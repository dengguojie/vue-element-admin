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
 * \file a_reduce_sum_fusion_pass.cpp
 * \brief reducesum fusion pass
 */
#include "a_reduce_sum_fusion_pass.h"
#include "tbe_ops_pass_util.h"

namespace fe {
static const string PATTERN_FUSEDNODE = "FusedNodeReduceSum";
static const string FUSED_NODE = "ReduceSum";
static const int32_t INT_NUM_ZERO = 0;
static const std::string CONSTANTOP = "Const";

template <typename Dtype>
Status AssitHelp(const int32_t n, vector<int64_t> shape, Dtype& output1) {
  Dtype* output = &output1;
  for (int32_t i = 0; i < n; ++i) {
    output[i] = shape[i];
  }
  return SUCCESS;
}

Status AReduceSumFusionPass::CheckSumFussionOrNot(vector<int64_t> tensor_info, vector<int64_t> axis_info,
                                                  Operator& op) {
  bool keep_dims = false;
  const string keep_dims_name = "keep_dims";
  if (GRAPH_SUCCESS != op.GetAttr(keep_dims_name, keep_dims)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "can't get keep_dims attr.");
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

vector<FusionPattern*> AReduceSumFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("AReduceSumFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "New a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

Status AReduceSumFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define AReduceSumFusionPass fusion begin.");
  ge::NodePtr sumNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(sumNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "sumNode is null, fusion failed."),
                    return PARAM_INVALID);

  FUSION_PASS_CHECK(sumNode->GetOpDesc() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "sumNode get output failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(sumNode->GetOpDesc()->GetInputsSize() < 2, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "sumNode input size small than 2"),
                    return PARAM_INVALID);
  NOT_CHANGED_WITH_DYNAMIC_NODE({sumNode}); // dynamic not changed
  ge::GeTensorDesc tensor_input = sumNode->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc axis_input = sumNode->GetOpDesc()->GetInputDesc(1);

  vector<int64_t> tensor_info = tensor_input.GetShape().GetDims();
  size_t tensor_size = tensor_input.GetShape().GetDimNum();

  vector<int64_t> axis_info = axis_input.GetShape().GetDims();
  size_t axis_size = axis_input.GetShape().GetDimNum();

  Operator op = ge::OpDescUtils::CreateOperatorFromNode(sumNode);
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

  int axis_value = axis_input.GetShape().GetDim(0);

  if (const_data_size == 0) {
    for (size_t i = 0; i < tensor_info.size(); ++i) {
      const_data.push_back(i);
    }
  }

  for (size_t i = 0; i < const_data_size; ++i) {
    if (const_data[i] < 0) {
      const_data[i] = tensor_size + const_data[i];
    }
    if (const_data[i] >= (static_cast<int64_t>(tensor_size)) && (!IsUnknownRankShape(tensor_info))) {
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "const_data is not right");
        return FAILED;
    }
  }

  if (!(CheckSumFussionOrNot(tensor_info, const_data, op) == SUCCESS) && (axis_size != 1 || axis_value != 0)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Not need delete sumNode");
    return NOT_CHANGED;
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "delete edge of afterNode and sum. connect beforeNode and afterNode");
  ge::GeTensorDesc out_desc = sumNode->GetOpDesc()->GetOutputDesc(0);
  ge::GeShape sum_out_shape = out_desc.GetShape();
  vector<int64_t> dim_sum = sum_out_shape.GetDims();
  int32_t len_dim = dim_sum.size();
  unique_ptr<int32_t[]> inputAssit(new (std::nothrow) int32_t[len_dim]());
  FUSION_PASS_CHECK(inputAssit.get() == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputassist is null."),
                    return PARAM_INVALID);
  
  Status ret = NnSet(len_dim, INT_NUM_ZERO, *reinterpret_cast<int32_t*>(inputAssit.get()));
  FUSION_PASS_CHECK(ret != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "NnSet failed."),
                    return ret);
  ret = AssitHelp(len_dim, dim_sum, *inputAssit.get());
  FUSION_PASS_CHECK(ret != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AssitHelp failed."),
                    return ret);
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(sumNode->GetInDataAnchor(1)->GetPeerOutAnchor(), sumNode->GetInDataAnchor(1)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove sum and outnode edge failed."), return FAILED);

  ge::InDataAnchorPtr sum_anchor_ptr = sumNode->GetInDataAnchor(1);
  ge::NodeUtils::ClearInDataAnchor(sumNode, sum_anchor_ptr);
  ge::OpDescUtils::ClearInputDesc(sumNode->GetOpDesc(), 1);

  vector<int64_t> assitDimInfo = {len_dim};
  Format assitMatrixFormat = out_desc.GetFormat();
  ge::GeTensorPtr assitPtr = nullptr;
  ge::GeShape assitShape(assitDimInfo);
  ge::GeTensorDesc tensorDesc(GeShape(), ge::FORMAT_NCHW, ge::DT_INT32);
  tensorDesc.SetShape(assitShape);
  tensorDesc.SetOriginShape(assitShape);
  tensorDesc.SetDataType(ge::DT_INT32);
  tensorDesc.SetOriginDataType(ge::DT_INT32);
  tensorDesc.SetFormat(assitMatrixFormat);
  tensorDesc.SetOriginFormat(assitMatrixFormat);

  FUSION_PASS_MAKE_SHARED((assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()), 
                          len_dim * sizeof(int32_t))), assitPtr = nullptr;
                          return PARAM_INVALID);
  vector<ge::GeTensorPtr> weights = {assitPtr};
  ge::OpDescUtils::SetWeights(sumNode, weights);
  auto constInputNodes = OpDescUtils::GetConstInputs(sumNode);
  if (constInputNodes.empty()) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "sum_node get const failed.");
    return FAILED;
  }
  NodePtr constInput = constInputNodes[0];
  constInput->GetOpDesc()->SetType(CONSTANTOP);
  sumNode->GetOpDesc()->SetType("Reshape");

  std::map<string, uint32_t> input_name_id = {{"x", 0}, {"shape", 1}};
  sumNode->GetOpDesc()->UpdateInputName(input_name_id);
  std::vector<string> dep_inputs = {"shape"};
  sumNode->GetOpDesc()->SetOpInferDepends(dep_inputs);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define AReduceSumFusionPass fusion end");

  return SUCCESS;
}

REGISTER_PASS("AReduceSumFusionPass", BUILT_IN_GRAPH_PASS, AReduceSumFusionPass);
}  // namespace fe
