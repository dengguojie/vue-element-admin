/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file a_reduce_mean_fusion_pass.cpp
 * \brief reducemean fusion pass
 */
#include "a_reduce_mean_fusion_pass.h"
#include "tbe_ops_pass_util.h"

namespace fe {
static const string PATTERN_FUSEDNODE = "FusedNodeReduceMean";
static const string FUSED_NODE = "ReduceMean";

Status AReduceMeanFusionPass::CheckMeanFussionOrNot(vector<int64_t> tensor_info, vector<int64_t> axis_info,
                                                    Operator& op) {
  bool keep_dims = false;
  const string keep_dims_name = "keep_dims";
  if (GRAPH_SUCCESS != op.GetAttr(keep_dims_name, keep_dims)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "can't get keep_dims attr.");
  }

  if (tensor_info.empty()) {
    return SUCCESS;
  }

  for (auto& input_shape_value : tensor_info) {
    if (input_shape_value < 0 && !keep_dims) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Dynamic shape process and not keep dim, shouldn't delete.");
      return FAILED;
    }
  }
  for (size_t i = 0; i < axis_info.size(); ++i) {
    if (tensor_info[axis_info[i]] != 1) {
      return FAILED;
    }
  }
  return SUCCESS;
}

vector<FusionPattern*> AReduceMeanFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("AReduceMeanFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                   "New a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

Status AReduceMeanFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define AReduceMeanFusionPass fusion begin.");
  ge::NodePtr meanNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(meanNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                   "meanNode is null, fusion failed."),
                    return PARAM_INVALID);

  FUSION_PASS_CHECK(meanNode->GetOpDesc() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "meanNode get output failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(meanNode->GetOpDesc()->GetInputsSize() < 2, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "meanNode input size small than 2"),
                    return PARAM_INVALID);
  ge::GeTensorDesc tensor_input = meanNode->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc axis_input = meanNode->GetOpDesc()->GetInputDesc(1);

  vector<int64_t> tensor_info = tensor_input.GetShape().GetDims();
  size_t tensor_size = tensor_input.GetShape().GetDimNum();
  vector<int64_t> axis_info = axis_input.GetShape().GetDims();

  Operator op = ge::OpDescUtils::CreateOperatorFromNode(meanNode);
  Tensor data;
  if (GRAPH_SUCCESS != op.GetInputConstData("axes", data)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "GetInputConstData of axes failed.");
    return NOT_CHANGED;
  }

  std::vector<int64_t> const_data;
  int32_t* const_data_ptr = (int32_t*)data.GetData();
  size_t const_data_size = data.GetSize() / sizeof(int32_t);
  for (size_t i = 0; i < const_data_size; ++i) {
    const_data.push_back((int32_t)((*(const_data_ptr + i))));
  }

  if (const_data_size == 0) {
    for (size_t i = 0; i < tensor_info.size(); ++i) {
      const_data.push_back(i);
    }
  }

  for (size_t i = 0; i < const_data_size; ++i) {
    if (const_data[i] < 0) {
      const_data[i] = tensor_size + const_data[i];
    }
    if (const_data[i] > (static_cast<int64_t>(tensor_size)) && (!IsUnknownRankShape(tensor_info))) {
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "const_data is not right");
        return FAILED;
    }
  }

  if (!(CheckMeanFussionOrNot(tensor_info, const_data, op) == SUCCESS)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Not need delete meanNode");
    return NOT_CHANGED;
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "delete edge of afterNode and mean. connect beforeNode and afterNode");
  for (auto inDataAnchor : meanNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(meanNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove mean and outnode edge failed."),
                      return FAILED);
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(meanNode->GetInDataAnchor(0)->GetPeerOutAnchor(), inDataAnchor) != SUCCESS,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add innode and outnode edge failed."), return FAILED);
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "delete reducemean edge.");
  FUSION_PASS_CHECK(graph.RemoveNode(meanNode) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                   "Remove meanNode failed."),
                    return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define AReduceMeanFusionPass fusion end");

  return SUCCESS;
}

REGISTER_PASS("AReduceMeanFusionPass", BUILT_IN_GRAPH_PASS, AReduceMeanFusionPass);
}  // namespace fe
