/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
 * \file swin_attention_matmul_fusion_pass.h
 * \brief batchmatmulv2 + add + reshape + reshape + transpose + reshape + reshape fusion pass
 */
#include "swin_attention_ffn_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {
static const char *RESHAPE = "Reshape";
static const char *MATMUL = "BatchMatMulV2";
static const char *ADD = "Add";
static const char *TRANSPOSE = "TransposeD";
static const char *STRIDEDSLICE = "StridedSliceD";
static const char *CONCAT = "ConcatD";
static const std::string PATTERN_RESHAPE_1 = "Reshape_1";
static const std::string PATTERN_RESHAPE_2 = "Reshape_2";
static const std::string PATTERN_RESHAPE_3 = "Reshape_3";
static const std::string PATTERN_RESHAPE_4 = "Reshape_4";
static const std::string PATTERN_MATMUL = "BatchMatMulV2";
static const std::string PATTERN_ADD = "Add";
static const std::string PATTERN_TRANSPOSE = "TransposeD";
static const std::string PATTERN_STRIDEDSLICE_1 = "StridedSliceD_1";
static const std::string PATTERN_STRIDEDSLICE_2 = "StridedSliceD_2";
static const std::string PATTERN_STRIDEDSLICE_3 = "StridedSliceD_3";
static const std::string PATTERN_STRIDEDSLICE_4 = "StridedSliceD_4";
static const std::string PATTERN_CONCAT_1 = "ConcatD_1";
static const std::string PATTERN_CONCAT_2 = "ConcatD_2";
static const std::string SwinAttentionFFN = "SwinAttentionFFN";

static const std::vector<string> patten_node_all = {
  PATTERN_RESHAPE_1, PATTERN_RESHAPE_2,
  PATTERN_RESHAPE_3, PATTERN_RESHAPE_4,
  PATTERN_MATMUL, PATTERN_ADD,
  PATTERN_TRANSPOSE, PATTERN_STRIDEDSLICE_1,
  PATTERN_STRIDEDSLICE_2, PATTERN_STRIDEDSLICE_3,
  PATTERN_STRIDEDSLICE_4, PATTERN_CONCAT_1,
  PATTERN_CONCAT_2
};

vector<FusionPattern*> SwinAttentionFFNFusionPass::DefinePatterns() {
  vector < FusionPattern *> patterns;
  FusionPattern *pattern =
      new (std::nothrow) FusionPattern("SwinAttentionFFNFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(),
                    "new a pattern object failed."), return patterns);
  pattern->AddOpDesc(PATTERN_MATMUL, {MATMUL})
      .AddOpDesc(PATTERN_ADD, {ADD})
      .AddOpDesc(PATTERN_RESHAPE_1, {RESHAPE})
      .AddOpDesc(PATTERN_RESHAPE_2, {RESHAPE})
      .AddOpDesc(PATTERN_TRANSPOSE, {TRANSPOSE})
      .AddOpDesc(PATTERN_RESHAPE_3, {RESHAPE})
      .AddOpDesc(PATTERN_RESHAPE_4, {RESHAPE})
      .SetInputs(PATTERN_ADD, {PATTERN_MATMUL})
      .SetInputs(PATTERN_RESHAPE_1, {PATTERN_ADD})
      .SetInputs(PATTERN_RESHAPE_2, {PATTERN_RESHAPE_1})
      .SetInputs(PATTERN_TRANSPOSE, {PATTERN_RESHAPE_2})
      .SetInputs(PATTERN_RESHAPE_3, {PATTERN_TRANSPOSE})
      .SetInputs(PATTERN_RESHAPE_4, {PATTERN_RESHAPE_3})
      .SetOutput(PATTERN_RESHAPE_4);
  patterns.push_back(pattern);

  FusionPattern *pattern1 = new (std::nothrow) FusionPattern("SwinAttentionFFNFusionPass");
  FUSION_PASS_CHECK(pattern1 == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "new a pattern object fail."),
    return patterns);
  pattern1->AddOpDesc(PATTERN_MATMUL, {MATMUL})
      .AddOpDesc(PATTERN_ADD, {ADD})
      .AddOpDesc(PATTERN_RESHAPE_1, {RESHAPE})
      .AddOpDesc(PATTERN_RESHAPE_2, {RESHAPE})
      .AddOpDesc(PATTERN_TRANSPOSE, {TRANSPOSE})
      .AddOpDesc(PATTERN_RESHAPE_3, {RESHAPE})
      .AddOpDesc(PATTERN_STRIDEDSLICE_1, {STRIDEDSLICE})
      .AddOpDesc(PATTERN_STRIDEDSLICE_2, {STRIDEDSLICE})
      .AddOpDesc(PATTERN_CONCAT_1, {CONCAT})
      .AddOpDesc(PATTERN_STRIDEDSLICE_3, {STRIDEDSLICE})
      .AddOpDesc(PATTERN_STRIDEDSLICE_4, {STRIDEDSLICE})
      .AddOpDesc(PATTERN_CONCAT_2, {CONCAT})
      .AddOpDesc(PATTERN_RESHAPE_4, {RESHAPE})
      .SetInputs(PATTERN_ADD, {PATTERN_MATMUL})
      .SetInputs(PATTERN_RESHAPE_1, {PATTERN_ADD})
      .SetInputs(PATTERN_RESHAPE_2, {PATTERN_RESHAPE_1})
      .SetInputs(PATTERN_TRANSPOSE, {PATTERN_RESHAPE_2})
      .SetInputs(PATTERN_RESHAPE_3, {PATTERN_TRANSPOSE})
      .SetInputs(PATTERN_STRIDEDSLICE_1, {PATTERN_RESHAPE_3})
      .SetInputs(PATTERN_STRIDEDSLICE_2, {PATTERN_RESHAPE_3})
      .SetInputs(PATTERN_CONCAT_1, {PATTERN_STRIDEDSLICE_1, PATTERN_STRIDEDSLICE_2})
      .SetInputs(PATTERN_STRIDEDSLICE_3, {PATTERN_CONCAT_1})
      .SetInputs(PATTERN_STRIDEDSLICE_4, {PATTERN_CONCAT_1})
      .SetInputs(PATTERN_CONCAT_2, {PATTERN_STRIDEDSLICE_3, PATTERN_STRIDEDSLICE_4})
      .SetInputs(PATTERN_RESHAPE_4, {PATTERN_CONCAT_2})
      .SetOutput(PATTERN_RESHAPE_4);
  patterns.push_back(pattern1);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SwinAttentionFFNFusionPass pattern end");
  return patterns;
}

Status SwinAttentionFFNFusionPass::GetNodeFromPattern(Mapping &mapping, vector<ge::NodePtr> &node_ptr_all) {
  for (auto &node_name : patten_node_all) {
    ge::NodePtr node_ptr = GetNodeFromMapping(node_name, mapping);
    node_ptr_all.push_back(node_ptr);
  }
  return SUCCESS;
}

bool SwinAttentionFFNFusionPass::CheckNodeShape(vector<int64_t>& label_shape, vector<int64_t>& check_shape) {
 if (label_shape.size() != check_shape.size()) {
   return false;
 }
 for (size_t dims_num=0; dims_num < label_shape.size(); dims_num++){
   if (label_shape[dims_num] != check_shape[dims_num]) {
     return false;
   }
 }
 return true;
}

Status SwinAttentionFFNFusionPass::CheckBatchMatmulNode(ge::NodePtr& batchmatmul_node) {
  FUSION_PASS_CHECK(batchmatmul_node == nullptr,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "batchmatmul_node is null."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(batchmatmul_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "batchmatmul_node output 0 size is [%lu], which not equal to 1.",
                            batchmatmul_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  return SUCCESS;
}

Status SwinAttentionFFNFusionPass::CheckReshapeNode(vector<ge::NodePtr>& node_ptr_all) {
  ge::NodePtr reshape_1_node = node_ptr_all[0];
  ge::NodePtr transpose_node = node_ptr_all[6];
  ge::NodePtr reshape_2_node = node_ptr_all[1];
  ge::NodePtr reshape_3_node = node_ptr_all[2];
  ge::NodePtr reshape_4_node = node_ptr_all[3];
  FUSION_PASS_CHECK(reshape_1_node == nullptr,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_1_node is null."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(reshape_1_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_1_node output 0 size is [%lu], which not equal to 1.",
                            reshape_1_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(transpose_node == nullptr,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "transpose_node is null."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(transpose_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "transpose_node output 0 size is [%lu], which not equal to 1.",
                            transpose_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  std::vector<int64_t> perm;
  AttrUtils::GetListInt(transpose_node->GetOpDesc(), "perm", perm);
  std::vector<int64_t> perm_label = {0, 1, 3, 2, 4, 5};
  FUSION_PASS_CHECK(!CheckNodeShape(perm_label, perm),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "transpose_node node attr perm not support, fusion failed."),
                      return NOT_CHANGED);

  FUSION_PASS_CHECK(reshape_2_node == nullptr,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_2_node is null."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(reshape_2_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_2_node output 0 size is [%lu], which not equal to 1.",
                            reshape_3_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(reshape_3_node == nullptr,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_3_node is null."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(reshape_4_node == nullptr,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_4_node is null."),
                    return NOT_CHANGED);

  vector<int64_t> reshape_2_input_shape = reshape_2_node->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();
  int64_t reshape_2_input_size = reshape_2_input_shape.size();
  FUSION_PASS_CHECK((reshape_2_input_size != 3 && reshape_2_input_size != 4),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_2_node output shape not support."),
                    return NOT_CHANGED);

  vector<int64_t> reshape_2_output_shape = reshape_2_node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();
  FUSION_PASS_CHECK(reshape_2_output_shape.size() != 6,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_1_node output shape not support."),
                    return NOT_CHANGED);
  return CheckRollNode(node_ptr_all);
}

Status SwinAttentionFFNFusionPass::CheckConcatNode(vector<ge::NodePtr>& node_ptr_all) {
  ge::NodePtr reshape_3_node = node_ptr_all[2];
  ge::NodePtr slice_1_node = node_ptr_all[7];
  ge::NodePtr slice_2_node = node_ptr_all[8];
  ge::NodePtr concat_1_node = node_ptr_all[11];
  ge::NodePtr slice_3_node = node_ptr_all[9];
  ge::NodePtr slice_4_node = node_ptr_all[10];
  ge::NodePtr concat_2_node = node_ptr_all[12];

  FUSION_PASS_CHECK(reshape_3_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 2,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_3_node output 0 size is [%lu], which not equal to 2.",
                            reshape_3_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(slice_1_node == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "slice_1_node is null."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(slice_1_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "slice_1_node output 0 size is [%lu], which not equal to 1.",
                            slice_1_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(slice_2_node == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "slice_2_node is null."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(slice_2_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "slice_2_node output 0 size is [%lu], which not equal to 1.",
                            slice_2_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(concat_1_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 2,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "concat_1_node output 0 size is [%lu], which not equal to 2.",
                            concat_1_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(slice_3_node == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "slice_3_node is null."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(slice_3_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "slice_3_node output 0 size is [%lu], which not equal to 1.",
                            slice_3_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(slice_4_node == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "slice_4_node is null."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(slice_4_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "slice_4_node output 0 size is [%lu], which not equal to 1.",
                            slice_4_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(concat_2_node == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "concat_2_node is null."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(concat_2_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "concat_2_node output 0 size is [%lu], which not equal to 1.",
                            concat_2_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  return CheckRollNode(node_ptr_all);
}

Status SwinAttentionFFNFusionPass::CheckRollNode(vector<ge::NodePtr>& node_ptr_all) {
  ge::NodePtr reshape_1_node = node_ptr_all[0];
  vector<int64_t> reshape_input_shape = reshape_1_node->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();
  vector<int64_t> reshape_output_shape = reshape_1_node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();
  FUSION_PASS_CHECK(reshape_input_shape.size() != 3,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_1 node input shape not support."),
                    return NOT_CHANGED);
  int64_t dim_1_num = (int64_t)sqrt(reshape_input_shape[1]);
  vector<int64_t> label_output_shape = {reshape_input_shape[0], dim_1_num, dim_1_num, reshape_input_shape[2]};
  FUSION_PASS_CHECK(!CheckNodeShape(label_output_shape, reshape_output_shape),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_1 node shape not support, fusion failed."),
                    return NOT_CHANGED);
  return SUCCESS;
}

Status SwinAttentionFFNFusionPass::CheckPatternNode(vector<ge::NodePtr>& node_ptr_all) {
  ge::NodePtr batchmatmul_node = node_ptr_all[4];
  ge::NodePtr concat_1_node = node_ptr_all[11];
  FUSION_PASS_CHECK(CheckBatchMatmulNode(batchmatmul_node) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "batchmatmul_node check failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(CheckReshapeNode(node_ptr_all) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Reshape_node check failed."),
                    return NOT_CHANGED);
  if (concat_1_node != nullptr) {
    FUSION_PASS_CHECK(CheckConcatNode(node_ptr_all) != SUCCESS,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "concat check failed."),
                      return NOT_CHANGED);
  }

  return SUCCESS;
}

Status SwinAttentionFFNFusionPass::SetInputOutputDesc(vector<ge::NodePtr>& node_ptr_all,
                                                      std::shared_ptr<ge::OpDesc>& attention_matmul_desc) {
  // get all node's desc
  ge::NodePtr matmul_node = node_ptr_all[4];
  ge::NodePtr add_node = node_ptr_all[5];
  ge::NodePtr reshape_node4 = node_ptr_all[3];

  ge::GeTensorDesc input_desc_0 = matmul_node->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc input_desc_1 = matmul_node->GetOpDesc()->GetInputDesc(1);
  ge::GeTensorDesc input_desc_2 = add_node->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc output_desc_0 = reshape_node4->GetOpDesc()->GetOutputDesc(0);

  // add input
  FUSION_PASS_CHECK(attention_matmul_desc->AddInputDesc(input_desc_0) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input_0 failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(attention_matmul_desc->AddInputDesc(input_desc_1) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input_1 failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(attention_matmul_desc->AddInputDesc(input_desc_2) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input_2 failed."),
                    return NOT_CHANGED);
  // add output
  FUSION_PASS_CHECK(attention_matmul_desc->AddOutputDesc("y", output_desc_0) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(),
                    "add output failed."), return NOT_CHANGED);
  return SUCCESS;
}

Status SwinAttentionFFNFusionPass::SetAttrPatternNode(vector<ge::NodePtr>& node_ptr_all,
                                                      ge::NodePtr& attention_matmul_node) {
  ge::NodePtr reshape_node4 = node_ptr_all[3];
  ge::NodePtr concat_1_node = node_ptr_all[11];
  ge::NodePtr concat_2_node = node_ptr_all[12];
  int64_t dim_second = 1;
  int64_t dim_third = 2;
  // get attr
  std::vector<int64_t> slice_dim;
  slice_dim.push_back(0);
  int64_t slice_dim_h = 0;
  if (concat_1_node != nullptr) {
    OpDescPtr concat_1_desc = concat_1_node->GetOpDesc();
    GeTensorDesc concat_1_input_desc = concat_1_desc->GetInputDesc(0);
    vector<int64_t> concat_1_input_shape = concat_1_input_desc.GetShape().GetDims();
    slice_dim_h = concat_1_input_shape[dim_second];
  }
  int64_t slice_dim_w = 0;
  if (concat_2_node != nullptr) {
    OpDescPtr concat_2_desc = concat_2_node->GetOpDesc();
    GeTensorDesc concat_2_input_desc = concat_2_desc->GetInputDesc(0);
    vector<int64_t> concat_2_input_shape = concat_2_input_desc.GetShape().GetDims();
    slice_dim_w = concat_2_input_shape[dim_third];
  }

  slice_dim.push_back(slice_dim_h);
  slice_dim.push_back(slice_dim_w);
  slice_dim.push_back(0);

  OpDescPtr reshape_desc4 = reshape_node4->GetOpDesc();
  // set attr
  AttrUtils::SetListInt(attention_matmul_node->GetOpDesc(), "shifts",slice_dim);
  return SUCCESS;
}

Status SwinAttentionFFNFusionPass::NewNodeAddEdge(ge::NodePtr& attention_matmul_node,
                                                  vector<ge::NodePtr>& node_ptr_all){
  ge::NodePtr matmul_node = node_ptr_all[4];
  ge::NodePtr add_node = node_ptr_all[5];
  ge::NodePtr reshape_node4 = node_ptr_all[3];
  // add input edge
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
      matmul_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
      attention_matmul_node->GetInDataAnchor(0)) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
          matmul_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
          attention_matmul_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
      matmul_node->GetInDataAnchor(1)->GetPeerOutAnchor(),
      attention_matmul_node->GetInDataAnchor(1)) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
          matmul_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
          attention_matmul_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
      add_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
      attention_matmul_node->GetInDataAnchor(2)) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
          add_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
          attention_matmul_node->GetName().c_str()),
      return FAILED);

  // add output edge
  for (auto &inDataAnchor : reshape_node4->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(reshape_node4->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "Remove out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(attention_matmul_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "Add out data edge failed."), return FAILED);
  }
  return SUCCESS;
}

Status SwinAttentionFFNFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr> &fusionNodes) {
  // get all nodes
  vector<ge::NodePtr> node_ptr_all;
  FUSION_PASS_CHECK(GetNodeFromPattern(mapping, node_ptr_all) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "SwinAttentionFFNFusionPass fusion failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(CheckPatternNode(node_ptr_all) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "SwinAttentionFFNFusionPass fusion failed."),
                    return NOT_CHANGED);

  ge::NodePtr matmul_node = node_ptr_all[4];
  // set new op attention_matmul
  std::shared_ptr<ge::OpDesc> attention_matmul_desc = nullptr;
  attention_matmul_desc = std::make_shared<ge::OpDesc>(matmul_node->GetName() + "/" + SwinAttentionFFN,
                                                         SwinAttentionFFN);
  FUSION_PASS_CHECK(attention_matmul_desc == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(),
                   "attention_desc is null, fusion failed."), return NOT_CHANGED);

  // set op desc
  FUSION_PASS_CHECK(SetInputOutputDesc(node_ptr_all, attention_matmul_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "SwinAttentionFFNFusionPass fusion failed."),
                    return NOT_CHANGED);

  // add attention_matmul node
  ge::NodePtr attention_matmul_node = graph.AddNode(attention_matmul_desc);
  fusionNodes.push_back(attention_matmul_node);

   // set attr
  FUSION_PASS_CHECK(SetAttrPatternNode(node_ptr_all, attention_matmul_node) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "SwinAttentionFFNFusionPass fusion failed."),
                    return NOT_CHANGED);

  // check whether op is supported
  FUSION_PASS_CHECK(!CheckOpSupported(attention_matmul_node->GetOpDesc()),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Op Not Supported."),
                    return NOT_CHANGED);

  // add edge
  FUSION_PASS_CHECK(NewNodeAddEdge(attention_matmul_node, node_ptr_all) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "SwinAttentionFFNFusionPass fusion failed."),
                    return FAILED);
  // delete fused nodes
  for (auto &remove_node : node_ptr_all) {
    if (remove_node != nullptr) {
      FUSION_PASS_CHECK(graph.RemoveNode(remove_node) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove node failed."),
                        return FAILED);
    }
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "SwinAttentionFFNFusionPass graph fusion success!");
  return SUCCESS;
}

REGISTER_PASS("SwinAttentionFFNFusionPass", BUILT_IN_GRAPH_PASS, SwinAttentionFFNFusionPass);
}
