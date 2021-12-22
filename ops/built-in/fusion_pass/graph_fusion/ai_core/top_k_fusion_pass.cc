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
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "securec.h"
#include "top_k_fusion_pass.h"

using namespace std;
using namespace ge;

namespace fe {
static const string kPatternTopK = "topk";
static const string kConstantOp = "Constant";
static const string kPatternTranspose = "TransposeD";

Status PermVecGen(int64_t dim_size, int64_t dim_aim, vector<int64_t>& perm) {
  if (dim_aim > dim_size) {
    return FAILED;
  }
  for (int64_t i = 0; i < dim_size; i++) {
    perm.push_back(i);
  }
  swap(perm[dim_aim], perm[dim_size - 1]);
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
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(),
                    "New a pattern object failed."),
                    return patterns);
  // define origin graph
  pattern->AddOpDesc(kPatternTopK, {"TopK", "TopKV2"}).SetOutput(kPatternTopK);
  patterns.push_back(pattern);
  return patterns;
}

Status TopKFusionPass::Fusion(ComputeGraph& graph, Mapping& mapping, vector<NodePtr>& fusion_nodes) {
  NodePtr topk_node = GetNodeFromMapping(kPatternTopK, mapping);
  FUSION_PASS_CHECK(topk_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(),
                    "The topk_node is null, fusion failed."),
                    return PARAM_INVALID);
  OpDescPtr topk_desc = topk_node->GetOpDesc();
  FUSION_PASS_CHECK(topk_desc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(),
                    "The topk_desc is null, fusion failed."),
                    return PARAM_INVALID);
  // may find TopKV2, use TopK instead
  topk_desc->SetType("TopK");
  auto input_desc_k = topk_desc->GetInputDesc(1);
  if (input_desc_k.GetDataType() == ge::DT_INT64) {
    input_desc_k.SetDataType(ge::DT_INT32);
    input_desc_k.SetOriginDataType(ge::DT_INT32);
    topk_desc->UpdateInputDesc(1, input_desc_k);
  }

  // The value of sorted cannot be false in aicore
  bool sorted = true;
  // attr sorted is optional
  AttrUtils::GetBool(topk_desc, "sorted", sorted);
  FUSION_PASS_CHECK(!sorted,
                    OP_LOGW(kFusedOpType.c_str(), "The value of sorted must be true in aicore, fusion failed."),
                    return NOT_CHANGED);

  // first input of topkv2 is non-constant, second is constant
  InDataAnchorPtr topk_anchor_ptr0 = topk_node->GetInDataAnchor(0);
  FUSION_PASS_CHECK(topk_anchor_ptr0 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(),
                                                   "The topk_anchor_ptr0 is null, fusion failed."),
                    return PARAM_INVALID);
  OutDataAnchorPtr data_anchor_ptr = topk_anchor_ptr0->GetPeerOutAnchor();
  FUSION_PASS_CHECK(data_anchor_ptr == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The data_anchor_ptr is null, fusion failed."),
                                                   return PARAM_INVALID);
  NodePtr data_node = data_anchor_ptr->GetOwnerNode();
  auto data_node_desc = data_node->GetOpDesc();
  FUSION_PASS_CHECK(data_node_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The data_node_desc is null, fusion failed."),
                                                   return PARAM_INVALID);
  GeTensorDesc topk_data_tensor = data_node_desc->GetOutputDesc(0);
  GeShape topk_data_shape = topk_data_tensor.GetShape();
  vector<int64_t> dim_info = topk_data_shape.GetDims();
  FUSION_PASS_CHECK(dim_info.size() < 1, OP_LOGW(kFusedOpType.c_str(), "The dim_info size error."), return NOT_CHANGED);
  // 4096 indicates the length of index in assist matrix.
  constexpr int64_t kAssistLen{4096};

  OutDataAnchorPtr topk_anchor_out_ptr0 = topk_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(topk_anchor_out_ptr0 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(),
                                                   "The topk_anchor_out_ptr0 is null, fusion failed."),
                    return PARAM_INVALID);
  NodePtr data_node_out = topk_anchor_out_ptr0->GetOwnerNode();
  FUSION_PASS_CHECK(data_node_out == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The data_node_out is null, fusion failed."),
                                                   return PARAM_INVALID);
  auto topk_data_out_tensor_desc = data_node_out->GetOpDesc();
  FUSION_PASS_CHECK(topk_data_out_tensor_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(),
                                                   "The topk_data_out_tensor_desc is null, fusion failed."),
                    return PARAM_INVALID);
  GeTensorDesc topk_data_out_tensor = topk_data_out_tensor_desc->GetOutputDesc(0);
  GeShape topk_data_out_shape = topk_data_out_tensor.GetShape();
  vector<int64_t> dim_info_out = topk_data_out_shape.GetDims();
  FUSION_PASS_CHECK(dim_info_out.size() == 0,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The dim_info_out size is 0, fusion failed."),
                                                   return PARAM_INVALID);

  vector<PassAttrInfo> topk_attr_info;
  PassAttrInfo k_attr = {1, "k", "SetInt"};
  topk_attr_info.push_back(k_attr);
  string node_name = topk_node->GetName();

  OpDescPtr fusion_desc_ptr = AttrUtils::CloneOpDesc(topk_desc);
  FUSION_PASS_CHECK(fusion_desc_ptr == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The fusion_desc_ptr is null, fusion failed."),
                                                   return PARAM_INVALID);
  fusion_desc_ptr->SetType("TopKD");
  vector<int> attr_index_vec;
  for (size_t i = 0; i < topk_attr_info.size(); i++) {
    attr_index_vec.push_back(topk_attr_info[i].attrIndex);
  }
  sort(attr_index_vec.begin(), attr_index_vec.end());

  // remove the inputdesc which need to be removed
  for (int i = attr_index_vec.size() - 1; i >= 0; i--) {
    unsigned int index = attr_index_vec[i];
    if (index >= fusion_desc_ptr->GetInputsSize()) {
      OP_LOGI(kFusedOpType.c_str(), "Index[%u] is beyond the size[%u] of input desc", index,
              fusion_desc_ptr->GetInputsSize());
      continue;
    }
    if (!OpDescUtils::ClearInputDesc(fusion_desc_ptr, index)) {
      OP_LOGI(kFusedOpType.c_str(), "Fail to clear input desc[%u]", index);
    }
  }

  Operator op = OpDescUtils::CreateOperatorFromNode(topk_node);
  int64_t const_data_val = 0;
  Tensor const_tensor;
  bool is_topk_v2 = true;
  if (op.GetInputConstData("k", const_tensor) == GRAPH_SUCCESS) {
    // top_k_v2 use k = 0
    is_topk_v2 = false;
    auto k_tensor_desc = op.GetInputDesc("k");
    DataType input_k_dtype = k_tensor_desc.GetDataType();
    uint8_t* const_data_ptr = const_tensor.GetData();
    FUSION_PASS_CHECK(const_data_ptr == nullptr, OP_LOGW(kFusedOpType.c_str(), "Get k const data failed."),
                      return NOT_CHANGED);
    if (input_k_dtype == DT_INT32) {
      const_data_val = static_cast<int64_t>(*(reinterpret_cast<int32_t*>(const_data_ptr)));
    } else if (input_k_dtype == DT_INT64) {
      const_data_val = *(reinterpret_cast<int64_t*>(const_data_ptr));
    } else {
      OP_LOGW(kFusedOpType.c_str(), "K only support int32 and int64 in AICORE");
      return NOT_CHANGED;
    }
  }

  FUSION_PASS_CHECK(!AttrUtils::SetInt(fusion_desc_ptr, "k", const_data_val),
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Set attr k failed"), return FAILED);
  vector<int64_t> dims = {1};
  GeShape input1_shape(dims);
  GeTensorDesc in_desc1(input1_shape);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  FUSION_PASS_CHECK(fusion_desc_ptr->AddInputDesc("assic_seq", in_desc1) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "AddInputDesc failed"), return FAILED);
  FUSION_PASS_CHECK(!CheckOpSupported(fusion_desc_ptr), OP_LOGW(kFusedOpType.c_str(), "Op Not Supported."),
                    return NOT_CHANGED);

  Status ret = SUCCESS;
  NodePtr fusion_node = topk_node;
  if (!is_topk_v2) {
    ret = PatternFusionUtil::ConstToAttrWithNode(graph, topk_node, "TopKD", topk_attr_info, fusion_node);
  }
  fusion_nodes.push_back(fusion_node);

  FUSION_PASS_CHECK(topk_desc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(),
                    "FusionNode is null, fusion failed."),
                    return PARAM_INVALID);
  GeTensorPtr assit_ptr{nullptr};
  unique_ptr<uint16_t[]> inputAssit(new (nothrow) uint16_t[kAssistLen * 2]());
  FUSION_PASS_CHECK(inputAssit.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(),
                    "InputAssit is NULL"),
                    return FAILED);
  ret = AssitHelp(kAssistLen, inputAssit.get());
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(kFusedOpType.c_str(), "AssitHelp failed."), return NOT_CHANGED);

  // define shape
  vector<int64_t> assit_dim_info;
  assit_dim_info.push_back(kAssistLen * 2);
  GeShape assit_shape(assit_dim_info);
  GeTensorDesc tensor_desc(GeShape(), FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetShape(assit_shape);
  tensor_desc.SetFormat(FORMAT_ND);
  tensor_desc.SetOriginFormat(FORMAT_ND);
  FUSION_PASS_MAKE_SHARED((assit_ptr = make_shared<GeTensor>(tensor_desc, reinterpret_cast<uint8_t*>(inputAssit.get()),
                                                             kAssistLen * 2 * sizeof(uint16_t))),
                          assit_ptr = nullptr;
                          return PARAM_INVALID);

  vector<GeTensorPtr> weights = {assit_ptr};
  FUSION_PASS_CHECK(OpDescUtils::SetWeights(fusion_node, weights) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "SetWeights failed"), return FAILED);
  auto const_input_nodes = OpDescUtils::GetConstInputs(fusion_node);
  FUSION_PASS_CHECK(const_input_nodes.size() <= 0, VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(),
                    "GetConstInputs Error"),
                    return PARAM_INVALID);
  NodePtr const_input = const_input_nodes[0];
  FUSION_PASS_CHECK(const_input == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(),
                    "The const_input is null, fusion failed."),
                    return PARAM_INVALID);
  auto const_input_desc = const_input->GetOpDesc();
  FUSION_PASS_CHECK(const_input_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(),
                                                   "The const_input_desc is null, fusion failed."),
                    return PARAM_INVALID);
  const_input_desc->SetType(kConstantOp);
  if (is_topk_v2) {
    topk_desc->SetType("TopKV2D");
  } else {
    topk_desc->SetType("TopKD");
  }

  OpDescPtr topkd_desc = fusion_node->GetOpDesc();
  int64_t dim_size = dim_info.size();
  int64_t dim_aim;
  if (!AttrUtils::GetInt(topkd_desc, "dim", dim_aim)) {
    OP_LOGI(kFusedOpType.c_str(), "Cannot get attr dim, fusion success, no need do more");
    return SUCCESS;
  }
  if (dim_aim < 0) {
    dim_aim = dim_size + dim_aim;
  }
  if (dim_aim == dim_size - 1) {
    return SUCCESS;
  }

  NodePtr trans_input_node =
      PatternFusionUtil::InsertSingleNode(graph, fusion_node, kPatternTranspose, true, 0, fusion_nodes);
  OpDescPtr trans_input_desc = trans_input_node->GetOpDesc();
  FUSION_PASS_CHECK(trans_input_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(),
                                                   "The trans_input_desc is null, fusion failed."),
                    return PARAM_INVALID);
  GeTensorDesc trans_data_tensor = trans_input_desc->GetInputDesc(0);
  GeShape trans_data_shape = trans_data_tensor.GetShape();
  vector<int64_t> trans_dim_info = trans_data_shape.GetDims();
  int64_t trans_dim_info_size = trans_dim_info.size();
  FUSION_PASS_CHECK(dim_aim >= trans_dim_info_size, VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(),
                    "Dim index is out of shape range."),
                    return PARAM_INVALID);
  swap(trans_dim_info[dim_aim], trans_dim_info[dim_size - 1]);

  // get input_transpose perm
  vector<int64_t> perm;
  ret = PermVecGen(dim_size, dim_aim, perm);
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "PermVecGen failed."),
                    return ret);

  // set input_transpose perm
  FUSION_PASS_CHECK(!AttrUtils::SetListInt(trans_input_desc, "perm", perm),
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Input transporse set perm failed"),
                                                   return FAILED);
  // set input_transpose output shape range
  vector<pair<int64_t, int64_t>> shape_range_after_sorted;
  if (trans_data_tensor.GetShapeRange(shape_range_after_sorted) != GRAPH_SUCCESS) {
    OP_LOGD(kFusedOpType.c_str(), "GetShapeRange failed. However the process is fine.");
  }
  if (shape_range_after_sorted.size() > 0) {
    int64_t tmp = shape_range_after_sorted[dim_aim].second;
    shape_range_after_sorted[dim_aim].second = shape_range_after_sorted[dim_size - 1].second;
    shape_range_after_sorted[dim_size - 1].second = tmp;
  }
  GeTensorDesc out_trans_data_tensor = trans_input_desc->GetOutputDesc(0);
  FUSION_PASS_CHECK(out_trans_data_tensor.SetShapeRange(shape_range_after_sorted) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "SetShapeRange failed"), return FAILED);
  // set input_transpose output shape
  GeShape transpose_assit_shape(trans_dim_info);
  auto transin_mutable_output0 = trans_input_desc->MutableOutputDesc(0);
  FUSION_PASS_CHECK(transin_mutable_output0 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(),
                                                   "The transin_mutable_output0 is null, fusion failed."),
                    return PARAM_INVALID);
  transin_mutable_output0->SetShape(GeShape(transpose_assit_shape));
  transin_mutable_output0->SetOriginShape(GeShape(transpose_assit_shape));

  // set topk dim and input desc
  FUSION_PASS_CHECK(!AttrUtils::SetInt(topkd_desc, "dim", -1), VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(),
                    "Set attr dim failed"),
                    return FAILED);
  auto fusion_mutable_input0 = topkd_desc->MutableInputDesc(0);
  FUSION_PASS_CHECK(fusion_mutable_input0 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(),
                                                   "The fusion_mutable_input0 is null, fusion failed."),
                    return PARAM_INVALID);

  fusion_mutable_input0->SetShape(GeShape(transpose_assit_shape));
  fusion_mutable_input0->SetOriginShape(GeShape(transpose_assit_shape));
  // set topkd input shape range

  GeTensorDesc topkd_input_data_tensor = topkd_desc->GetInputDesc(0);
  FUSION_PASS_CHECK(topkd_input_data_tensor.SetShapeRange(shape_range_after_sorted) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "SetShapeRange failed"), return FAILED);

  // set topkd output desc according to transpose
  vector<int64_t> topk_out_shape;
  GeTensorDesc topkd_data_tensor = topkd_desc->GetOutputDesc(0);
  GeShape topkd_data_shape = topkd_data_tensor.GetShape();
  topk_out_shape = topkd_data_shape.GetDims();
  vector<int64_t> topkd_dim_info = topk_out_shape;
  swap(topkd_dim_info[dim_aim], topkd_dim_info[dim_size - 1]);

  // set topkd val output shape
  GeShape topk_out_ge_shape(topkd_dim_info);
  auto fusion_mutable_output0 = topkd_desc->MutableOutputDesc(0);
  FUSION_PASS_CHECK(fusion_mutable_output0 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(),
                                                   "The fusion_mutable_output0 is null, fusion failed."),
                    return PARAM_INVALID);
  fusion_mutable_output0->SetShape(GeShape(topk_out_ge_shape));
  fusion_mutable_output0->SetOriginShape(GeShape(topk_out_ge_shape));
  // set topkd val output shape range
  vector<pair<int64_t, int64_t>> shape_range_val_k;
  shape_range_val_k = shape_range_after_sorted;
  if (shape_range_val_k.size() > 0) {
    shape_range_val_k[shape_range_val_k.size() - 1].second = const_data_val;
  }
  GeTensorDesc topkd_data_out_tensor = topkd_desc->GetOutputDesc(0);
  FUSION_PASS_CHECK(topkd_data_out_tensor.SetShapeRange(shape_range_val_k) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "SetShapeRange failed"), return FAILED);

  // set topkd index output shape
  auto fusion_mutable_output1 = topkd_desc->MutableOutputDesc(1);
  FUSION_PASS_CHECK(fusion_mutable_output1 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(),
                                                   "The fusion_mutable_output1 is null, fusion failed."),
                    return PARAM_INVALID);
  fusion_mutable_output1->SetShape(GeShape(topk_out_ge_shape));
  fusion_mutable_output1->SetOriginShape(GeShape(topk_out_ge_shape));
  // set topkd index output shape range
  GeTensorDesc topkd_data_out_index_tensor = topkd_desc->GetOutputDesc(1);
  FUSION_PASS_CHECK(topkd_data_out_index_tensor.SetShapeRange(shape_range_val_k) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "SetShapeRange failed"), return FAILED);

  NodePtr trans_output_node =
      PatternFusionUtil::InsertSingleNode(graph, fusion_node, kPatternTranspose, false, 0, fusion_nodes);
  FUSION_PASS_CHECK(trans_output_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(),
                                                   "The trans_output_node is null, fusion failed."),
                    return PARAM_INVALID);
  OpDescPtr trans_output_desc = trans_output_node->GetOpDesc();

  // set val transpose perm
  FUSION_PASS_CHECK(!AttrUtils::SetListInt(trans_output_desc, "perm", perm),
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Output val transporse set perm failed"),
                                                   return FAILED);

  // set val transepose output shape
  GeShape out_transpose_output_assit_shape(topk_out_shape);
  auto transout_mutable_output0 = trans_output_desc->MutableOutputDesc(0);
  FUSION_PASS_CHECK(transout_mutable_output0 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(),
                                                   "The transout_mutable_output0 is null, fusion failed."),
                    return PARAM_INVALID);
  transout_mutable_output0->SetShape(GeShape(out_transpose_output_assit_shape));
  transout_mutable_output0->SetOriginShape(GeShape(out_transpose_output_assit_shape));
  // set val transepose output shape input_shape_range
  GeTensorDesc trans_output_tensor_input = trans_output_desc->GetInputDesc(0);
  FUSION_PASS_CHECK(trans_output_tensor_input.SetShapeRange(shape_range_val_k) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "SetShapeRange failed"), return FAILED);
  // set val transepose output shape output_shape_range
  vector<pair<int64_t, int64_t>> shape_range_val_k_sorted;
  shape_range_val_k_sorted = shape_range_val_k;

  if (shape_range_val_k_sorted.size() > 0) {
    int64_t tmp = shape_range_val_k_sorted[dim_aim].second;
    shape_range_val_k_sorted[dim_aim].second = shape_range_val_k_sorted[shape_range_val_k_sorted.size() - 1].second;
    shape_range_val_k_sorted[shape_range_val_k_sorted.size() - 1].second = tmp;
  }
  GeTensorDesc trans_output_tensor_output = trans_output_desc->GetOutputDesc(0);
  FUSION_PASS_CHECK(trans_output_tensor_output.SetShapeRange(shape_range_val_k_sorted) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "SetShapeRange failed"), return FAILED);

  NodePtr trans_output_index_node =
      PatternFusionUtil::InsertSingleNode(graph, fusion_node, kPatternTranspose, false, 1, fusion_nodes);
  FUSION_PASS_CHECK(trans_output_index_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(),
                                                   "The trans_output_index_node is null, fusion failed."),
                    return PARAM_INVALID);
  OpDescPtr trans_output_index_desc = trans_output_index_node->GetOpDesc();

  // set index transpose perm
  FUSION_PASS_CHECK(!AttrUtils::SetListInt(trans_output_index_desc, "perm", perm),
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Output index transporse set perm failed"),
                                                   return FAILED);
  // set index transepose output shape
  GeShape out_index_transpose_output_assit_shape(topk_out_shape);
  auto trans_index_mutable_output0 = trans_output_index_desc->MutableOutputDesc(0);
  FUSION_PASS_CHECK(trans_index_mutable_output0 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(),
                                                   "The trans_index_mutable_output0 is null, fusion failed."),
                    return PARAM_INVALID);
  trans_index_mutable_output0->SetShape(GeShape(out_index_transpose_output_assit_shape));
  trans_index_mutable_output0->SetOriginShape(GeShape(out_index_transpose_output_assit_shape));
  // set index transepose output shape input_shape_range
  GeTensorDesc trans_output_index_tensor_input = trans_output_index_desc->GetInputDesc(0);
  FUSION_PASS_CHECK(trans_output_index_tensor_input.SetShapeRange(shape_range_val_k) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "SetShapeRange failed"), return FAILED);
  // set index transepose output shape output_shape_range
  GeTensorDesc trans_output_index_tensor_output = trans_output_index_desc->GetOutputDesc(0);
  FUSION_PASS_CHECK(trans_output_index_tensor_output.SetShapeRange(shape_range_val_k_sorted) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "SetShapeRange failed"), return FAILED);

  return SUCCESS;
}

REGISTER_PASS("TopKFusionPass", BUILT_IN_GRAPH_PASS, TopKFusionPass);
}  // namespace fe
