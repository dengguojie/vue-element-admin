/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file a_reduce_prod_fusion_pass.cpp
 * \brief reduceprod fusion pass
 */
#include "a_reduce_prod_fusion_pass.h"
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>
#include <algorithm>
#include "op_log.h"
#include "error_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"

namespace fe {
static const string PATTERN_FUSEDNODE = "FusedNodeReduceProd";
static const string FUSED_NODE = "ReduceProd";

Status CheckProdFussionOrNot(vector<int64_t> tensor_info, vector<int64_t> axis_info) {
  for (size_t i = 0; i < axis_info.size(); ++i) {
    if (axis_info[i] > tensor_info.size()) {
      return FAILED;
    }
    if (tensor_info[axis_info[i]] != 1) {
      return FAILED;
    }
  }
  return SUCCESS;
}

vector<FusionPattern*> AReduceProdFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("AReduceProdFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "New a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

Status AReduceProdFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define AReduceProdFusionPass fusion begin.");
  ge::NodePtr prodNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(prodNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "prodNode is null, fusion failed."),
                    return PARAM_INVALID);

  FUSION_PASS_CHECK(prodNode->GetOpDesc() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "prodNode get output failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(prodNode->GetOpDesc()->GetInputsSize() < 2, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "prodNode input size small than 2"),
                    return PARAM_INVALID);
  ge::GeTensorDesc tensor_input = prodNode->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc axis_input = prodNode->GetOpDesc()->GetInputDesc(1);

  vector<int64_t> tensor_info = tensor_input.GetShape().GetDims();
  size_t tensor_size = tensor_input.GetShape().GetDimNum();

  vector<int64_t> axis_info = axis_input.GetShape().GetDims();

  Operator op = ge::OpDescUtils::CreateOperatorFromNode(prodNode);
  Tensor data;
  if (GRAPH_SUCCESS != op.GetInputConstData("axes", data)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "GetInputConstData of axes failed.");
    return false;
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

  if (!(CheckProdFussionOrNot(tensor_info, const_data) == SUCCESS)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Not need delete prodNode");
    return NOT_CHANGED;
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "delete edge of afterNode and prod. connect beforeNode and afterNode");
  for (auto inDataAnchor : prodNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(prodNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove prod and outnode edge failed."), return FAILED);
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(prodNode->GetInDataAnchor(0)->GetPeerOutAnchor(), inDataAnchor) != SUCCESS,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add innode and outnode edge failed."), return FAILED);
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "delete reduceprod edge.");
  FUSION_PASS_CHECK(graph.RemoveNode(prodNode) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove prodNode failed."),
                    return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define AReduceProdFusionPass fusion end");

  return SUCCESS;
}

REGISTER_PASS("AReduceProdFusionPass", BUILT_IN_GRAPH_PASS, AReduceProdFusionPass);
}  // namespace fe
