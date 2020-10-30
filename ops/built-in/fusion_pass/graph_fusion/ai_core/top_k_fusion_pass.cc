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
 * \file top_k_fusion_pass.cpp
 * \brief TopK fusion pass(TopK --> TopKD)
 */
#include "top_k_fusion_pass.h"
#include "op_log.h"
#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "securec.h"
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace std;
using namespace ge;

namespace fe {
static const string kPatternTopK = "TopK";
static const std::string kConstOp = "Constant";
static const char* kTopK = "TopK";

Status AssitHelp(const int32_t n, uint16_t* output) {
  for (int32_t i = 0; i < n; ++i) {
    fp16_t t;
    t.val = 0;
    t = i;
    output[i] = t.val;
  }
  for (int32_t i = 0; i < n; ++i) {
    fp16_t t;
    t.val = 0;
    t = i;
    int32_t idx = t;
    int32_t gap = i - idx;
    fp16_t tmp;
    tmp.val = 0;
    tmp = gap;
    output[i + n] = tmp.val;
  }
  return SUCCESS;
}

vector<FusionPattern*> TopKFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  // TopK->TopKD
  FusionPattern* pattern = new (std::nothrow) FusionPattern("TopKFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  // define origin graph
  pattern->AddOpDesc(kPatternTopK, {kTopK}).SetOutput(kPatternTopK);

  patterns.push_back(pattern);

  return patterns;
}

Status TopKFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr topk_node = GetNodeFromMapping(kPatternTopK, mapping);
  FUSION_PASS_CHECK(topk_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "topk_node is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr topk_desc = topk_node->GetOpDesc();
  FUSION_PASS_CHECK(topk_desc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "topk_desc is null, fusion failed."),
                    return PARAM_INVALID);

  // first input of topkv2 is non-constant, second is constant
  ge::InDataAnchorPtr topk_anchor_ptr0 = topk_node->GetInDataAnchor(0);
  ge::OutDataAnchorPtr data_anchor_ptr = topk_anchor_ptr0->GetPeerOutAnchor();
  ge::NodePtr data_node = data_anchor_ptr->GetOwnerNode();
  ge::GeTensorDesc topk_data_tensor = data_node->GetOpDesc()->GetOutputDesc(0);
  ge::GeShape topk_data_shape = topk_data_tensor.GetShape();
  vector<int64_t> dim_info = topk_data_shape.GetDims();
  if (dim_info.size() < 1) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "dim_info size error.");
    return PARAM_INVALID;
  }
  // 4096 indicates the length of index in assist matrix.
  constexpr int64_t kAssistLen{4096};

  ge::OutDataAnchorPtr topk_anchor_out_ptr0 = topk_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(topk_anchor_out_ptr0 == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "topk_anchor_out_ptr0 is null, fusion failed."),
                    return PARAM_INVALID);
  ge::NodePtr data_node_out = topk_anchor_out_ptr0->GetOwnerNode();
  FUSION_PASS_CHECK(data_node_out == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "data_node_out is null, fusion failed."),
                    return PARAM_INVALID);
  ge::GeTensorDesc topk_data_out_tensor = data_node_out->GetOpDesc()->GetOutputDesc(0);
  ge::GeShape topk_data_out_shape = topk_data_out_tensor.GetShape();
  vector<int64_t> dim_info_out = topk_data_out_shape.GetDims();
  FUSION_PASS_CHECK(dim_info_out.size() == 0, OP_LOGE(FUSED_OP_TYPE.c_str(), "dim_info_out size is 0, fusion failed."),
                    return PARAM_INVALID);
  int64_t last_out_dim = dim_info_out[dim_info_out.size() - 1];
  if (last_out_dim == UNKNOWN_DIM) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "It's unkown shape, graph not changed.");
    return NOT_CHANGED;
  }

  std::string fusion_op_type = "TopKD";
  std::vector<PassAttrInfo> topk_attr_info;
  PassAttrInfo k_attr = {1, "k", "SetInt"};
  topk_attr_info.push_back(k_attr);
  ge::NodePtr fusion_node = nullptr;
  std::string node_name = topk_node->GetName();

  ge::OpDescPtr fusion_desc_ptr = AttrUtils::CloneOpDesc(topk_desc);
  fusion_desc_ptr->SetType("TopKD");
  std::vector<int> attr_index_vec;
  for (size_t i = 0; i < topk_attr_info.size(); i++) {
    attr_index_vec.push_back(topk_attr_info[i].attrIndex);
  }
  std::sort(attr_index_vec.begin(), attr_index_vec.end());

  // remove the inputdesc which need to be removed
  for (int i = attr_index_vec.size() - 1; i >= 0; i--) {
    unsigned int index = attr_index_vec[i];
    if (index >= fusion_desc_ptr->GetInputsSize()) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Index[%u] is beyond the size[%d] of input desc", index,
              fusion_desc_ptr->GetInputsSize());
      continue;
    }
    if (!OpDescUtils::ClearInputDesc(fusion_desc_ptr, index)) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Fail to clear input desc[%u]", index);
    }
  }

  Operator op = ge::OpDescUtils::CreateOperatorFromNode(topk_node);
  Tensor const_tensor;
  (void)op.GetInputConstData("k", const_tensor);
  int32_t* const_data_ptr = (int32_t*)const_tensor.GetData();
  if (const_data_ptr == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "GetData NULL.");
    return NOT_CHANGED;
  }

  ge::AttrUtils::SetInt(fusion_desc_ptr, "k", *(const_data_ptr));
  vector<int64_t> dims = {1};
  ge::GeShape input1_shape(dims);
  ge::GeTensorDesc in_desc1(input1_shape);
  in_desc1.SetFormat(ge::FORMAT_NCHW);
  in_desc1.SetDataType(ge::DT_FLOAT16);
  fusion_desc_ptr->AddInputDesc("assic_seq", in_desc1);
  FUSION_PASS_CHECK(!CheckOpSupported(fusion_desc_ptr), OP_LOGI(FUSED_OP_TYPE.c_str(), "Op Not Supported."),
                    return NOT_CHANGED);

  ge::NodePtr fusionNode = nullptr;
  Status ret = PatternFusionUtil::ConstToAttrWithNode(graph, topk_node, fusion_op_type, topk_attr_info, fusionNode);
  fusionNodes.push_back(fusionNode);
  for (auto node : graph.GetDirectNode()) {
    if (node_name == node->GetName()) {
      fusion_node = node;
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Find FusionNode");
      break;
    }
  }
  FUSION_PASS_CHECK(topk_desc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "FusionNode is null, fusion failed."),
                    return PARAM_INVALID);
  ge::GeTensorPtr assit_ptr = nullptr;
  unique_ptr<uint16_t> inputAssit(new (std::nothrow) uint16_t[kAssistLen * 2]());
  FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                    return PARAM_INVALID);
  ret = AssitHelp(kAssistLen, inputAssit.get());
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return ret);

  // define shape
  vector<int64_t> assit_dim_info;
  assit_dim_info.push_back(kAssistLen * 2);
  ge::GeShape assit_shape(assit_dim_info);
  ge::GeTensorDesc tensor_desc(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensor_desc.SetShape(assit_shape);
  tensor_desc.SetFormat(ge::FORMAT_NCHW);
  FUSION_PASS_MAKE_SHARED(
      (assit_ptr = std::make_shared<ge::GeTensor>(tensor_desc, reinterpret_cast<uint8_t*>(inputAssit.get()),
                                                  kAssistLen * 2 * sizeof(uint16_t))),
      assit_ptr = nullptr;
      return PARAM_INVALID);

  vector<ge::GeTensorPtr> weights = {assit_ptr};
  (void)ge::OpDescUtils::SetWeights(fusion_node, weights);
  auto const_input_nodes = OpDescUtils::GetConstInputs(fusion_node);
  if (const_input_nodes.size() <= 0) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "GetConstInputs Error");
    return PARAM_INVALID;
  }
  NodePtr const_input = const_input_nodes[0];
  const_input->GetOpDesc()->SetType(kConstOp);
  topk_desc->SetType("TopKD");

  return SUCCESS;
}

REGISTER_PASS("TopKFusionPass", BUILT_IN_GRAPH_PASS, TopKFusionPass);
}  // namespace fe
