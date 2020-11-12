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
 * \brief if dim = -1. TopK --> TopKD.
 * \brief if dim != -1. TransposeD -> TopkD -> TransposeD.
 */
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "securec.h"
#include "top_k_fusion_pass.h"

using namespace std;
using namespace ge;

namespace fe {
static const string kPatternTopK = "TopK";
static const string kConstantOp = "Constant";
static const string kPatternTopKD = "TopKD";
static const string kPatternTranspose = "TransposeD";

Status PermVecGen(int64_t dim_size, int64_t dim_aim, vector<int64_t>& perm) {
  if (dim_aim > dim_size) {
    return FAILED;
  }
  for (int64_t i = 0; i < dim_size; i++) {
    perm.push_back(i);
  }
  std::swap(perm[dim_aim], perm[dim_size - 1]);
  return SUCCESS;
}

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
  FusionPattern* pattern = new (nothrow) FusionPattern("TopKFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(kFusedOpType.c_str(), "New a pattern object failed."), return patterns);
  // define origin graph
  pattern->AddOpDesc(kPatternTopK, {"TopK"}).SetOutput(kPatternTopK);
  patterns.push_back(pattern);
  return patterns;
}

Status TopKFusionPass::Fusion(ComputeGraph& graph, Mapping& mapping, vector<NodePtr>& fusion_nodes) {
  NodePtr topk_node = GetNodeFromMapping(kPatternTopK, mapping);
  FUSION_PASS_CHECK(topk_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "The topk_node is null, fusion failed."),
                    return PARAM_INVALID);
  OpDescPtr topk_desc = topk_node->GetOpDesc();
  FUSION_PASS_CHECK(topk_desc == nullptr, OP_LOGE(kFusedOpType.c_str(), "The topk_desc is null, fusion failed."),
                    return PARAM_INVALID);

  // first input of topkv2 is non-constant, second is constant
  InDataAnchorPtr topk_anchor_ptr0 = topk_node->GetInDataAnchor(0);
  OutDataAnchorPtr data_anchor_ptr = topk_anchor_ptr0->GetPeerOutAnchor();
  NodePtr data_node = data_anchor_ptr->GetOwnerNode();
  GeTensorDesc topk_data_tensor = data_node->GetOpDesc()->GetOutputDesc(0);
  GeShape topk_data_shape = topk_data_tensor.GetShape();
  vector<int64_t> dim_info = topk_data_shape.GetDims();
  if (dim_info.size() < 1) {
    OP_LOGW(kFusedOpType.c_str(), "The dim_info size error.");
    return NOT_CHANGED;
  }
  // 4096 indicates the length of index in assist matrix.
  constexpr int64_t kAssistLen{4096};

  OutDataAnchorPtr topk_anchor_out_ptr0 = topk_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(topk_anchor_out_ptr0 == nullptr,
                    OP_LOGE(kFusedOpType.c_str(), "The topk_anchor_out_ptr0 is null, fusion failed."),
                    return PARAM_INVALID);
  NodePtr data_node_out = topk_anchor_out_ptr0->GetOwnerNode();
  FUSION_PASS_CHECK(data_node_out == nullptr,
                    OP_LOGE(kFusedOpType.c_str(), "The data_node_out is null, fusion failed."), return PARAM_INVALID);
  GeTensorDesc topk_data_out_tensor = data_node_out->GetOpDesc()->GetOutputDesc(0);
  GeShape topk_data_out_shape = topk_data_out_tensor.GetShape();
  vector<int64_t> dim_info_out = topk_data_out_shape.GetDims();
  FUSION_PASS_CHECK(dim_info_out.size() == 0,
                    OP_LOGE(kFusedOpType.c_str(), "The dim_info_out size is 0, fusion failed."), return PARAM_INVALID);
  int64_t last_out_dim = dim_info_out[dim_info_out.size() - 1];
  if (last_out_dim == UNKNOWN_DIM) {
    OP_LOGI(kFusedOpType.c_str(),
            "When the last dimension is unknown, topk does not support this kind of unknown shape, graph not changed.");
    return NOT_CHANGED;
  }

  vector<PassAttrInfo> topk_attr_info;
  PassAttrInfo k_attr = {1, "k", "SetInt"};
  topk_attr_info.push_back(k_attr);
  NodePtr fusion_node = nullptr;
  string node_name = topk_node->GetName();

  OpDescPtr fusion_desc_ptr = AttrUtils::CloneOpDesc(topk_desc);
  fusion_desc_ptr->SetType(kPatternTopKD);
  vector<int> attr_index_vec;
  for (size_t i = 0; i < topk_attr_info.size(); i++) {
    attr_index_vec.push_back(topk_attr_info[i].attrIndex);
  }
  sort(attr_index_vec.begin(), attr_index_vec.end());

  // remove the inputdesc which need to be removed
  for (int i = attr_index_vec.size() - 1; i >= 0; i--) {
    unsigned int index = attr_index_vec[i];
    if (index >= fusion_desc_ptr->GetInputsSize()) {
      OP_LOGI(kFusedOpType.c_str(), "Index[%u] is beyond the size[%d] of input desc", index,
              fusion_desc_ptr->GetInputsSize());
      continue;
    }
    if (!OpDescUtils::ClearInputDesc(fusion_desc_ptr, index)) {
      OP_LOGI(kFusedOpType.c_str(), "Fail to clear input desc[%u]", index);
    }
  }

  Operator op = OpDescUtils::CreateOperatorFromNode(topk_node);
  Tensor const_tensor;
  (void)op.GetInputConstData("k", const_tensor);
  auto k_tensor_desc = op.GetInputDesc("k");
  DataType input_k_dtype = k_tensor_desc.GetDataType();
  int64_t const_data_val = 0;
  uint8_t* const_data_ptr = const_tensor.GetData();
  if (const_data_ptr == nullptr) {
    OP_LOGI(kFusedOpType.c_str(), "GetData NULL.");
    return NOT_CHANGED;
  }

  if (input_k_dtype == DT_INT32) {
    const_data_val = static_cast<int64_t>(*(reinterpret_cast<int32_t*>(const_data_ptr)));
  } else if (input_k_dtype == DT_INT64) {
    const_data_val = *(reinterpret_cast<int64_t*>(const_data_ptr));
  } else {
    OP_LOGI(kFusedOpType.c_str(), "K only support int32 and int64 in AICORE");
    return NOT_CHANGED;
  }

  AttrUtils::SetInt(fusion_desc_ptr, "k", const_data_val);
  vector<int64_t> dims = {1};
  GeShape input1_shape(dims);
  GeTensorDesc in_desc1(input1_shape);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  fusion_desc_ptr->AddInputDesc("assic_seq", in_desc1);
  FUSION_PASS_CHECK(!CheckOpSupported(fusion_desc_ptr), OP_LOGI(kFusedOpType.c_str(), "Op Not Supported."),
                    return NOT_CHANGED);

  NodePtr fusionNode = nullptr;
  Status ret = PatternFusionUtil::ConstToAttrWithNode(graph, topk_node, kPatternTopKD, topk_attr_info, fusionNode);
  fusion_nodes.push_back(fusionNode);
  for (auto node : graph.GetDirectNode()) {
    if (node_name == node->GetName()) {
      fusion_node = node;
      OP_LOGI(kFusedOpType.c_str(), "Find FusionNode");
      break;
    }
  }

  FUSION_PASS_CHECK(topk_desc == nullptr, OP_LOGE(kFusedOpType.c_str(), "FusionNode is null, fusion failed."),
                    return PARAM_INVALID);
  GeTensorPtr assit_ptr{nullptr};
  unique_ptr<uint16_t> inputAssit(new (nothrow) uint16_t[kAssistLen * 2]());
  FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(kFusedOpType.c_str(), "InputAssit is NULL"),
                    return PARAM_INVALID);
  ret = AssitHelp(kAssistLen, inputAssit.get());
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(kFusedOpType.c_str(), "AssitHelp failed."), return NOT_CHANGED);

  // define shape
  vector<int64_t> assit_dim_info;
  assit_dim_info.push_back(kAssistLen * 2);
  GeShape assit_shape(assit_dim_info);
  GeTensorDesc tensor_desc(GeShape(), FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetShape(assit_shape);
  tensor_desc.SetFormat(FORMAT_NCHW);
  FUSION_PASS_MAKE_SHARED((assit_ptr = make_shared<GeTensor>(tensor_desc, reinterpret_cast<uint8_t*>(inputAssit.get()),
                                                             kAssistLen * 2 * sizeof(uint16_t))),
                          assit_ptr = nullptr;
                          return PARAM_INVALID);

  vector<GeTensorPtr> weights = {assit_ptr};
  (void)OpDescUtils::SetWeights(fusion_node, weights);
  auto const_input_nodes = OpDescUtils::GetConstInputs(fusion_node);
  if (const_input_nodes.size() <= 0) {
    OP_LOGE(kFusedOpType.c_str(), "GetConstInputs Error");
    return PARAM_INVALID;
  }
  NodePtr const_input = const_input_nodes[0];
  const_input->GetOpDesc()->SetType(kConstantOp);
  topk_desc->SetType(kPatternTopKD);

  OpDescPtr topkd_desc = fusion_node->GetOpDesc();
  int64_t dim_size = dim_info.size();
  int64_t dim_aim;
  AttrUtils::GetInt(topkd_desc, "dim", dim_aim);
  if (dim_aim < 0) {
    dim_aim = dim_size + dim_aim;
  }
  if (dim_aim == dim_size - 1) {
    return SUCCESS;
  }

  NodePtr trans_input_node =
      PatternFusionUtil::InsertSingleNode(graph, fusion_node, kPatternTranspose, true, 0, fusion_nodes);
  OpDescPtr trans_input_desc = trans_input_node->GetOpDesc();
  GeTensorDesc trans_data_tensor = trans_input_desc->GetInputDesc(0);
  GeShape trans_data_shape = trans_data_tensor.GetShape();
  vector<int64_t> trans_dim_info = trans_data_shape.GetDims();
  int64_t trans_dim_info_size = trans_dim_info.size();
  if (dim_aim >= trans_dim_info_size) {
    OP_LOGE(kFusedOpType.c_str(), "Dim index is out of shape range.");
  }

  std::swap(trans_dim_info[dim_aim], trans_dim_info[dim_size - 1]);

  // get input_transpose perm
  std::vector<int64_t> perm;
  ret = PermVecGen(dim_size, dim_aim, perm);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(kFusedOpType.c_str(), "PermVecGen failed."), return ret);

  // set input_transpose perm
  FUSION_PASS_CHECK(!AttrUtils::SetListInt(trans_input_desc, "perm", perm),
                    OP_LOGE(kFusedOpType.c_str(), "Input transporse set perm failed"), return FAILED);
  // set input_transpose output shape range
  std::vector<std::pair<int64_t, int64_t>> shape_range_after_sorted;
  trans_data_tensor.GetShapeRange(shape_range_after_sorted);

  if (shape_range_after_sorted.size() > 0) {
    int64_t tmp = shape_range_after_sorted[dim_aim].second;
    shape_range_after_sorted[dim_aim].second = shape_range_after_sorted[dim_size - 1].second;
    shape_range_after_sorted[dim_size - 1].second = tmp;
  }
  GeTensorDesc out_trans_data_tensor = trans_input_desc->GetOutputDesc(0);
  out_trans_data_tensor.SetShapeRange(shape_range_after_sorted);

  // set input_transpose output shape
  GeShape transpose_assit_shape(trans_dim_info);
  trans_input_desc->MutableOutputDesc(0)->SetShape(ge::GeShape(transpose_assit_shape));
  trans_input_desc->MutableOutputDesc(0)->SetOriginShape(ge::GeShape(transpose_assit_shape));

  // set topk dim and input desc
  AttrUtils::SetInt(topkd_desc, "dim", -1);
  fusion_node->GetOpDesc()->MutableInputDesc(0)->SetShape(ge::GeShape(transpose_assit_shape));
  fusion_node->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(ge::GeShape(transpose_assit_shape));
  // set topkd input shape range
  OpDescPtr topkd_input_desc = fusion_node->GetOpDesc();
  GeTensorDesc topkd_input_data_tensor = topkd_input_desc->GetInputDesc(0);
  topkd_input_data_tensor.SetShapeRange(shape_range_after_sorted);

  // set topkd output desc according to transpose
  vector<int64_t> topk_out_shape;
  GeTensorDesc topkd_data_tensor = topkd_desc->GetOutputDesc(0);
  GeShape topkd_data_shape = topkd_data_tensor.GetShape();
  topk_out_shape = topkd_data_shape.GetDims();
  vector<int64_t> topkd_dim_info = topk_out_shape;
  std::swap(topkd_dim_info[dim_aim], topkd_dim_info[dim_size - 1]);

  // set topkd val output shape
  GeShape topk_out_ge_shape(topkd_dim_info);
  fusion_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(ge::GeShape(topk_out_ge_shape));
  fusion_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(ge::GeShape(topk_out_ge_shape));
  // set topkd val output shape range
  std::vector<std::pair<int64_t, int64_t>> shape_range_val_k;
  shape_range_val_k = shape_range_after_sorted;
  if (shape_range_val_k.size() > 0) {
    shape_range_val_k[shape_range_val_k.size() - 1].second = const_data_val;
  }
  GeTensorDesc topkd_data_out_tensor = topkd_input_desc->GetOutputDesc(0);
  topkd_data_out_tensor.SetShapeRange(shape_range_val_k);

  // set topkd index output shape
  fusion_node->GetOpDesc()->MutableOutputDesc(1)->SetShape(ge::GeShape(topk_out_ge_shape));
  fusion_node->GetOpDesc()->MutableOutputDesc(1)->SetOriginShape(ge::GeShape(topk_out_ge_shape));
  // set topkd index output shape range
  GeTensorDesc topkd_data_out_index_tensor = topkd_input_desc->GetOutputDesc(1);
  topkd_data_out_index_tensor.SetShapeRange(shape_range_val_k);

  NodePtr trans_output_node =
      PatternFusionUtil::InsertSingleNode(graph, fusion_node, kPatternTranspose, false, 0, fusion_nodes);
  OpDescPtr trans_output_desc = trans_output_node->GetOpDesc();

  // set val transpose perm
  FUSION_PASS_CHECK(!AttrUtils::SetListInt(trans_output_desc, "perm", perm),
                    OP_LOGE(kFusedOpType.c_str(), "Output val transporse set perm failed"), return FAILED);

  // set val transepose output shape
  GeShape out_transpose_ouput_assit_shape(topk_out_shape);
  trans_output_desc->MutableOutputDesc(0)->SetShape(ge::GeShape(out_transpose_ouput_assit_shape));
  trans_output_desc->MutableOutputDesc(0)->SetOriginShape(ge::GeShape(out_transpose_ouput_assit_shape));
  // set val transepose output shape input_shape_range
  GeTensorDesc trans_output_tensor_input = trans_output_desc->GetInputDesc(0);
  trans_output_tensor_input.SetShapeRange(shape_range_val_k);
  // set val transepose output shape output_shape_range
  std::vector<std::pair<int64_t, int64_t>> shape_range_val_k_sorted;
  shape_range_val_k_sorted = shape_range_val_k;

  if (shape_range_val_k_sorted.size() > 0) {
    int64_t tmp = shape_range_val_k_sorted[dim_aim].second;
    shape_range_val_k_sorted[dim_aim].second = shape_range_val_k_sorted[shape_range_val_k_sorted.size() - 1].second;
    shape_range_val_k_sorted[shape_range_val_k_sorted.size() - 1].second = tmp;
  }
  GeTensorDesc trans_output_tensor_output = trans_output_desc->GetOutputDesc(0);
  trans_output_tensor_output.SetShapeRange(shape_range_val_k_sorted);

  NodePtr trans_output_index_node =
      PatternFusionUtil::InsertSingleNode(graph, fusion_node, kPatternTranspose, false, 1, fusion_nodes);
  OpDescPtr trans_output_index_desc = trans_output_index_node->GetOpDesc();

  // set index transpose perm
  FUSION_PASS_CHECK(!AttrUtils::SetListInt(trans_output_index_desc, "perm", perm),
                    OP_LOGE(kFusedOpType.c_str(), "Output index transporse set perm failed"), return FAILED);
  // set index transepose output shape
  GeShape out_index_transpose_output_assit_shape(topk_out_shape);
  trans_output_index_desc->MutableOutputDesc(0)->SetShape(ge::GeShape(out_index_transpose_output_assit_shape));
  trans_output_index_desc->MutableOutputDesc(0)->SetOriginShape(ge::GeShape(out_index_transpose_output_assit_shape));
  // set index transepose output shape input_shape_range
  GeTensorDesc trans_output_index_tensor_input = trans_output_index_desc->GetInputDesc(0);
  trans_output_index_tensor_input.SetShapeRange(shape_range_val_k);
  // set index transepose output shape output_shape_range
  GeTensorDesc trans_output_index_tensor_output = trans_output_index_desc->GetOutputDesc(0);
  trans_output_index_tensor_output.SetShapeRange(shape_range_val_k_sorted);

  return SUCCESS;
}

REGISTER_PASS("TopKFusionPass", BUILT_IN_GRAPH_PASS, TopKFusionPass);
}  // namespace fe
