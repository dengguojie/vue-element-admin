/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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
 * \file swin_transformer_ln_qkv_fusion_pass.cc
 * \brief the pass will turn three conjuction matmul_confusionTranspose into a swin_transformer_ln_qkv
 *  *  * Training pattern:
 *                        LayerNorm
 *                      /     |     \
 *                   var   Reshape  mean
 *                            |
 *                        TransposeD
 *                            |
 *                         Reshape
 *                            |
 *                         Reshape          =>   swin_transformer_ln_qkv
 *                            |
 *                       BatchMatMulV2
 *                            |
 *                    ConfusionTransposeD
 *                            |
 *                         SplitVD
 *                    /       |       \
 *                Squeeze  Squeeze  Squeeze
 *
 */
#include "swin_transformer_ln_qkv_fusion_pass.h"

#include <string>
#include <vector>
#include <cmath>

#include "common/util/platform_info.h"
#include "error_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"
namespace fe {
namespace {
static const string kPatternLayerNorm0 = "LayerNorm0";
static const string kPatternReshape0 = "Reshape0";
static const string kPatternReshape1 = "Reshape1";
static const string kPatternTransposeD0 = "TransposeD0";
static const string kPatternReshape2 = "Reshape2";
static const string kPatternReshape3 = "Reshape3";
static const string kPatternBatchMatMulV20 = "BatchMatMulV20";
static const string kPatternConfusionTransposeD0 = "ConfusionTransposeD0";
static const string kPatternSplitVD0 = "SplitVD0";

static const string kPatternStridedSliceD0 = "StridedSliceD0";
static const string kPatternStridedSliceD1 = "StridedSliceD1";
static const string kPatternConcatD0 = "ConcatD0";
static const string kPatternStridedSliceD2 = "StridedSliceD2";
static const string kPatternStridedSliceD3 = "StridedSliceD3";
static const string kPatternConcatD1 = "ConcatD1";

static const string kOpLayerNorm = "LayerNorm";
static const string kOpReshape = "Reshape";
static const string kOpTransposeD = "TransposeD";
static const string kOpBatchMatMulV2 = "BatchMatMulV2";
static const string kOpConfusionTransposeD = "ConfusionTransposeD";
static const string kOpSplitVD = "SplitVD";
static const string kOpSqueeze = "Squeeze";

static const string kOpStridedSliceD = "StridedSliceD";
static const string kOpConcatD = "ConcatD";

static const string kOpSwinTransformerLnQKV = "SwinTransformerLnQKV";
static const std::vector<string> patten_node_name_all = {
  kPatternLayerNorm0, kPatternReshape0,
  kPatternStridedSliceD0, kPatternStridedSliceD1,
  kPatternConcatD0,
  kPatternStridedSliceD2, kPatternStridedSliceD3,
  kPatternConcatD1,
  kPatternReshape1, kPatternTransposeD0,
  kPatternReshape2, kPatternReshape3,
  kPatternBatchMatMulV20,
  kPatternConfusionTransposeD0,
  kPatternSplitVD0
};
}

vector<FusionPattern*> SwinTransformerLnQKVFusionPass::DefineFirstPatterns(vector<FusionPattern*>& patterns) {
  FusionPattern *pattern1 = new (std::nothrow) FusionPattern("SwinTransformerLnQKVFusionPass");
  FUSION_PASS_CHECK(pattern1 == nullptr,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "new a pattern object fail."),
                    return patterns);
  pattern1->AddOpDesc(kPatternLayerNorm0, {kOpLayerNorm})
           .AddOpDesc(kPatternReshape0, {kOpReshape})
           .AddOpDesc(kPatternReshape1, {kOpReshape})
           .AddOpDesc(kPatternTransposeD0, {kOpTransposeD})
           .AddOpDesc(kPatternReshape2, {kOpReshape})
           .AddOpDesc(kPatternReshape3, {kOpReshape})
           .AddOpDesc(kPatternBatchMatMulV20, {kOpBatchMatMulV2})
           .AddOpDesc(kPatternConfusionTransposeD0, {kOpConfusionTransposeD})
           .AddOpDesc(kPatternSplitVD0, {kOpSplitVD})
           .SetInputs(kPatternReshape0, {kPatternLayerNorm0})
           .SetInputs(kPatternReshape1, {kPatternReshape0})
           .SetInputs(kPatternTransposeD0, {kPatternReshape1})
           .SetInputs(kPatternReshape2, {kPatternTransposeD0})
           .SetInputs(kPatternReshape3, {kPatternReshape2})
           .SetInputs(kPatternBatchMatMulV20, {kPatternReshape3})
           .SetInputs(kPatternConfusionTransposeD0, {kPatternBatchMatMulV20})
           .SetInputs(kPatternSplitVD0, {kPatternConfusionTransposeD0})
           .SetOutput(kPatternSplitVD0);
  patterns.push_back(pattern1);
  return patterns;
}

vector<FusionPattern*> SwinTransformerLnQKVFusionPass::DefineSecondPatterns(vector<FusionPattern*>& patterns) {
  FusionPattern *pattern2 = new (std::nothrow) FusionPattern("SwinTransformerLnQKVFusionPass");
  FUSION_PASS_CHECK(pattern2 == nullptr,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "new a pattern object fail."),
                    return patterns);
  pattern2->AddOpDesc(kPatternLayerNorm0, {kOpLayerNorm})
           .AddOpDesc(kPatternReshape0, {kOpReshape})
           .AddOpDesc(kPatternStridedSliceD0, {kOpStridedSliceD})
           .AddOpDesc(kPatternStridedSliceD1, {kOpStridedSliceD})
           .AddOpDesc(kPatternConcatD0, {kOpConcatD})
           .AddOpDesc(kPatternStridedSliceD2, {kOpStridedSliceD})
           .AddOpDesc(kPatternStridedSliceD3, {kOpStridedSliceD})
           .AddOpDesc(kPatternConcatD1, {kOpConcatD})
           .AddOpDesc(kPatternReshape1, {kOpReshape})
           .AddOpDesc(kPatternTransposeD0, {kOpTransposeD})
           .AddOpDesc(kPatternReshape2, {kOpReshape})
           .AddOpDesc(kPatternReshape3, {kOpReshape})
           .AddOpDesc(kPatternBatchMatMulV20, {kOpBatchMatMulV2})
           .AddOpDesc(kPatternConfusionTransposeD0, {kOpConfusionTransposeD})
           .AddOpDesc(kPatternSplitVD0, {kOpSplitVD})
           .SetInputs(kPatternReshape0, {kPatternLayerNorm0})
           .SetInputs(kPatternStridedSliceD0, {kPatternReshape0})
           .SetInputs(kPatternStridedSliceD1, {kPatternReshape0})
           .SetInputs(kPatternConcatD0, {kPatternStridedSliceD0, kPatternStridedSliceD1})
           .SetInputs(kPatternStridedSliceD2, {kPatternConcatD0})
           .SetInputs(kPatternStridedSliceD3, {kPatternConcatD0})
           .SetInputs(kPatternConcatD1, {kPatternStridedSliceD2, kPatternStridedSliceD3})
           .SetInputs(kPatternReshape1, {kPatternConcatD1})
           .SetInputs(kPatternTransposeD0, {kPatternReshape1})
           .SetInputs(kPatternReshape2, {kPatternTransposeD0})
           .SetInputs(kPatternReshape3, {kPatternReshape2})
           .SetInputs(kPatternBatchMatMulV20, {kPatternReshape3})
           .SetInputs(kPatternConfusionTransposeD0, {kPatternBatchMatMulV20})
           .SetInputs(kPatternSplitVD0, {kPatternConfusionTransposeD0})
           .SetOutput(kPatternSplitVD0);
  patterns.push_back(pattern2);

  return patterns;
}

vector<FusionPattern*> SwinTransformerLnQKVFusionPass::DefineThirdPatterns(vector<FusionPattern*>& patterns) {
  FusionPattern *pattern3 = new (std::nothrow) FusionPattern("SwinTransformerLnQKVFusionPass");
  FUSION_PASS_CHECK(pattern3 == nullptr,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "new a pattern object fail."),
                    return patterns);
  pattern3->AddOpDesc(kPatternLayerNorm0, {kOpLayerNorm})
           .AddOpDesc(kPatternReshape0, {kOpReshape})
           .AddOpDesc(kPatternStridedSliceD0, {kOpStridedSliceD})
           .AddOpDesc(kPatternStridedSliceD1, {kOpStridedSliceD})
           .AddOpDesc(kPatternConcatD0, {kOpConcatD})
           .AddOpDesc(kPatternStridedSliceD2, {kOpStridedSliceD})
           .AddOpDesc(kPatternStridedSliceD3, {kOpStridedSliceD})
           .AddOpDesc(kPatternConcatD1, {kOpConcatD})
           .AddOpDesc(kPatternReshape1, {kOpReshape})
           .AddOpDesc(kPatternTransposeD0, {kOpTransposeD})
           .AddOpDesc(kPatternReshape3, {kOpReshape})
           .AddOpDesc(kPatternBatchMatMulV20, {kOpBatchMatMulV2})
           .AddOpDesc(kPatternConfusionTransposeD0, {kOpConfusionTransposeD})
           .AddOpDesc(kPatternSplitVD0, {kOpSplitVD})
           .SetInputs(kPatternReshape0, {kPatternLayerNorm0})
           .SetInputs(kPatternStridedSliceD0, {kPatternReshape0})
           .SetInputs(kPatternStridedSliceD1, {kPatternReshape0})
           .SetInputs(kPatternConcatD0, {kPatternStridedSliceD0, kPatternStridedSliceD1})
           .SetInputs(kPatternStridedSliceD2, {kPatternConcatD0})
           .SetInputs(kPatternStridedSliceD3, {kPatternConcatD0})
           .SetInputs(kPatternConcatD1, {kPatternStridedSliceD2, kPatternStridedSliceD3})
           .SetInputs(kPatternReshape1, {kPatternConcatD1})
           .SetInputs(kPatternTransposeD0, {kPatternReshape1})
           .SetInputs(kPatternReshape3, {kPatternTransposeD0})
           .SetInputs(kPatternBatchMatMulV20, {kPatternReshape3})
           .SetInputs(kPatternConfusionTransposeD0, {kPatternBatchMatMulV20})
           .SetInputs(kPatternSplitVD0, {kPatternConfusionTransposeD0})
           .SetOutput(kPatternSplitVD0);
  patterns.push_back(pattern3);
  return patterns;
}

vector<FusionPattern*> SwinTransformerLnQKVFusionPass::DefineFourthPatterns(vector<FusionPattern*>& patterns) {
  FusionPattern *pattern4 = new (std::nothrow) FusionPattern("SwinTransformerLnQKVFusionPass");
  FUSION_PASS_CHECK(pattern4 == nullptr,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "new a pattern object fail."),
                    return patterns);
  pattern4->AddOpDesc(kPatternLayerNorm0, {kOpLayerNorm})
           .AddOpDesc(kPatternReshape1, {kOpReshape})
           .AddOpDesc(kPatternTransposeD0, {kOpTransposeD})
           .AddOpDesc(kPatternReshape3, {kOpReshape})
           .AddOpDesc(kPatternBatchMatMulV20, {kOpBatchMatMulV2})
           .AddOpDesc(kPatternConfusionTransposeD0, {kOpConfusionTransposeD})
           .AddOpDesc(kPatternSplitVD0, {kOpSplitVD})
           .SetInputs(kPatternReshape1, {kPatternLayerNorm0})
           .SetInputs(kPatternTransposeD0, {kPatternReshape1})
           .SetInputs(kPatternReshape3, {kPatternTransposeD0})
           .SetInputs(kPatternBatchMatMulV20, {kPatternReshape3})
           .SetInputs(kPatternConfusionTransposeD0, {kPatternBatchMatMulV20})
           .SetInputs(kPatternSplitVD0, {kPatternConfusionTransposeD0})
           .SetOutput(kPatternSplitVD0);
  patterns.push_back(pattern4);
  return patterns;
}


vector<FusionPattern*> SwinTransformerLnQKVFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  patterns = DefineFirstPatterns(patterns);
  patterns = DefineSecondPatterns(patterns);
  patterns = DefineThirdPatterns(patterns);
  patterns = DefineFourthPatterns(patterns);
  return patterns;
}

bool SwinTransformerLnQKVFusionPass::CheckNodeShape(vector<int64_t>& label_shape, vector<int64_t>& check_shape) {
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

Status SwinTransformerLnQKVFusionPass::GetNodeFromPatten(Mapping &mapping,
                                                         vector<ge::NodePtr> &node_ptr_all) {
  for (auto &node_name : patten_node_name_all) {
    ge::NodePtr node_ptr = GetNodeFromMapping(node_name, mapping);
    node_ptr_all.push_back(node_ptr);
  }
  NodePtr split_node = node_ptr_all[14];
  FUSION_PASS_CHECK(split_node->GetAllOutDataAnchors().size() != 3,
                    OP_LOGI(FUSED_OP_TYPE.c_str(),
                            "split_node output size is [%ld], which not equal to 3.",
                            split_node->GetAllOutDataAnchors().size()),
                    return NOT_CHANGED);

  vector<int64_t> split_node_input_shape = split_node->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();
  split_node_input_shape[0] = 1;
  int64_t output_index = 3;
  for (int64_t spilt_index=0; spilt_index < output_index; spilt_index++) {
    NodePtr output_node = split_node->GetOutDataAnchor(spilt_index)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
    FUSION_PASS_CHECK(output_node == nullptr,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "output_node %ld node is null, fusion failed.", spilt_index),
                      return NOT_CHANGED);
    FUSION_PASS_CHECK(output_node->GetType() != kOpSqueeze,
                      OP_LOGI(FUSED_OP_TYPE.c_str(),
                      "output_node %ld node op is not Squeeze, fusion failed.", spilt_index),
                      return NOT_CHANGED);
    vector<int64_t> split_node_output_shape = split_node->GetOpDesc()->GetOutputDesc(spilt_index).GetShape().GetDims();
    FUSION_PASS_CHECK(!CheckNodeShape(split_node_input_shape, split_node_output_shape),
                      OP_LOGI(FUSED_OP_TYPE.c_str(),
                      "output_node %ld node shape not support, fusion failed.", spilt_index),
                      return NOT_CHANGED);

    std::vector<int64_t> axis;
    AttrUtils::GetListInt(output_node->GetOpDesc(), "axis", axis);
    FUSION_PASS_CHECK(axis.size() != 1,
                      OP_LOGI(FUSED_OP_TYPE.c_str(),
                      "output_node %ld node attr axis not match, fusion failed.", spilt_index),
                      return NOT_CHANGED);
    FUSION_PASS_CHECK(axis[0] != 0,
                      OP_LOGI(FUSED_OP_TYPE.c_str(),
                      "output_node %ld node attr axis not match, fusion failed.", spilt_index),
                      return NOT_CHANGED);
    node_ptr_all.push_back(output_node);
  }
  return SUCCESS;
}

Status SwinTransformerLnQKVFusionPass::CheckLayerNormNode(ge::NodePtr& ln_node) {
  FUSION_PASS_CHECK(ln_node == nullptr, 
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "ln_node is null."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(ln_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "ln_node output 0 size is [%lu], which not equal to 1.",
                            ln_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(ln_node->GetOutDataAnchor(1)->GetPeerAnchorsSize() != 0,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "ln_node output 1 size is [%lu], which not equal to 0.",
                            ln_node->GetOutDataAnchor(1)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(ln_node->GetOutDataAnchor(2)->GetPeerAnchorsSize() != 0,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "ln_node output 2 size is [%lu], which not equal to 0.",
                            ln_node->GetOutDataAnchor(2)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);

  ge::GeTensorDesc ln_input_desc_0 = ln_node->GetOpDesc()->GetInputDesc(0);
  int64_t ln_input_size = ln_input_desc_0.GetShape().GetDims().size();

  int64_t begin_norm_axis = 0;
  FUSION_PASS_CHECK(!AttrUtils::GetInt(ln_node->GetOpDesc(), "begin_norm_axis", begin_norm_axis),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "ln node Failed to get begin_norm_axis."),
                    return NOT_CHANGED);
  begin_norm_axis = begin_norm_axis >= 0 ? begin_norm_axis : begin_norm_axis + ln_input_size;
  FUSION_PASS_CHECK(begin_norm_axis != ln_input_size-1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "ln node attr begin_norm_axis not support."),
                    return NOT_CHANGED);

  int64_t begin_params_axis = 0;
  FUSION_PASS_CHECK(!AttrUtils::GetInt(ln_node->GetOpDesc(), "begin_params_axis", begin_params_axis),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "ln node Failed to get begin_params_axis."),
                    return NOT_CHANGED);
  begin_params_axis = begin_params_axis >= 0 ? begin_params_axis : begin_params_axis + ln_input_size;
  FUSION_PASS_CHECK(begin_params_axis != ln_input_size-1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "ln node attr begin_params_axis not support."),
                    return NOT_CHANGED);
  return SUCCESS;
}

Status SwinTransformerLnQKVFusionPass::CheckBatchMatmulNode(ge::NodePtr& batchmatmul_node) {
  FUSION_PASS_CHECK(batchmatmul_node == nullptr,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "batchmatmul_node is null."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(batchmatmul_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "batchmatmul_node output 0 size is [%lu], which not equal to 1.",
                            batchmatmul_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  return SUCCESS;
}

Status SwinTransformerLnQKVFusionPass::CheckReshapeNode(vector<ge::NodePtr>& node_ptr_all) {
  ge::NodePtr reshape_1_node = node_ptr_all[8];
  ge::NodePtr transpose_node = node_ptr_all[9];
  ge::NodePtr reshape_2_node = node_ptr_all[10];
  ge::NodePtr reshape_3_node = node_ptr_all[11];
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

  if (reshape_2_node != nullptr) {
    FUSION_PASS_CHECK(reshape_2_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_2_node output 0 size is [%lu], which not equal to 1.",
                            reshape_2_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  }
  FUSION_PASS_CHECK(reshape_3_node == nullptr,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_3_node is null."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(reshape_3_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_3_node output 0 size is [%lu], which not equal to 1.",
                            reshape_3_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  vector<int64_t> reshape_1_input_shape = reshape_1_node->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();
  int64_t reshape_1_input_size = reshape_1_input_shape.size();
  FUSION_PASS_CHECK((reshape_1_input_size != 3 && reshape_1_input_size != 4),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_1_node output shape not support."),
                    return NOT_CHANGED);

  vector<int64_t> reshape_1_output_shape = reshape_1_node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();
  FUSION_PASS_CHECK(reshape_1_output_shape.size() != 6,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_1_node output shape not support."),
                    return NOT_CHANGED);
  vector<int64_t> reshape_1_output_shape_label = {
    reshape_1_input_shape[0], reshape_1_output_shape[1], reshape_1_output_shape[2],
    reshape_1_output_shape[1], reshape_1_output_shape[2], reshape_1_input_shape[reshape_1_input_size-1]
    };
  FUSION_PASS_CHECK(!CheckNodeShape(reshape_1_output_shape_label, reshape_1_output_shape),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_1 node shape not support, fusion failed."),
                      return NOT_CHANGED);

  vector<int64_t> reshape_3_input_shape = reshape_3_node->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();
  int64_t reshape_3_input_size = reshape_3_input_shape.size();
  FUSION_PASS_CHECK((reshape_3_input_size != 6 && reshape_3_input_size != 4),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_3_node output shape not support."),
                    return NOT_CHANGED);
  vector<int64_t> reshape_3_output_shape = reshape_3_node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();
  FUSION_PASS_CHECK(reshape_3_output_shape.size() != 3,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_2_node output shape not support."),
                    return NOT_CHANGED);

  vector<int64_t> reshape_3_output_shape_label = {
    reshape_3_output_shape[0],
    reshape_3_input_shape[reshape_3_input_size-3] * reshape_3_input_shape[reshape_3_input_size-2],
    reshape_3_input_shape[reshape_3_input_size-1]
    };
  FUSION_PASS_CHECK(!CheckNodeShape(reshape_3_output_shape_label, reshape_3_output_shape),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_3 node shape not support, fusion failed."),
                      return NOT_CHANGED);
  return SUCCESS;
}

Status SwinTransformerLnQKVFusionPass::CheckConfusionTransposeNode(ge::NodePtr& confuse_node) {
  FUSION_PASS_CHECK(confuse_node == nullptr,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "confuse_node is null."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(confuse_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "confuse_node output 0 size is [%lu], which not equal to 1.",
                            confuse_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  vector<int64_t> confuse_input_shape = confuse_node->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();
  FUSION_PASS_CHECK(confuse_input_shape.size() != 3,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "confuse_node input shape not support."),
                    return NOT_CHANGED);

  vector<int64_t> confuse_output_shape = confuse_node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();
  FUSION_PASS_CHECK(confuse_output_shape.size() != 5,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "confuse_node output shape not support."),
                    return NOT_CHANGED);

  vector<int64_t> confuse_output_shape_label = {
    3, confuse_input_shape[0], confuse_output_shape[2], confuse_input_shape[1], confuse_output_shape[4]
    };
  FUSION_PASS_CHECK(!CheckNodeShape(confuse_output_shape_label, confuse_output_shape),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_3 node shape not support, fusion failed."),
                    return NOT_CHANGED);
  std::vector<int64_t> perm;
  AttrUtils::GetListInt(confuse_node->GetOpDesc(), "perm", perm);
  std::vector<int64_t> perm_label = {2, 0, 3, 1, 4};
  FUSION_PASS_CHECK(!CheckNodeShape(perm_label, perm),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "confuse_node node attr perm not support, fusion failed."),
                      return NOT_CHANGED);
  return SUCCESS;
}

Status SwinTransformerLnQKVFusionPass::CheckRollNode(vector<ge::NodePtr>& node_ptr_all) {
  ge::NodePtr reshape_0_node = node_ptr_all[1];
  ge::NodePtr concat_0_node = node_ptr_all[4];
  ge::NodePtr concat_1_node = node_ptr_all[7];
  vector<int64_t> reshape_input_shape = reshape_0_node->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();
  vector<int64_t> reshape_output_shape = reshape_0_node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();
  FUSION_PASS_CHECK(reshape_input_shape.size() != 3,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_0 node input shape not support."),
                    return NOT_CHANGED);
  int64_t dim_1_num = (int64_t)sqrt(reshape_input_shape[1]);
  vector<int64_t> label_output_shape = {reshape_input_shape[0], dim_1_num, dim_1_num, reshape_input_shape[2]};
  FUSION_PASS_CHECK(!CheckNodeShape(label_output_shape, reshape_output_shape),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_0 node shape not support, fusion failed."),
                    return NOT_CHANGED);

  vector<int64_t> concat_0_shape_0 = concat_0_node->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();
  vector<int64_t> concat_0_shape_1 = concat_0_node->GetOpDesc()->GetInputDesc(1).GetShape().GetDims();
  vector<int64_t> concat_0_output_shape = concat_0_node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();
  concat_0_shape_0[1] = dim_1_num;
  concat_0_shape_1[1] = dim_1_num;
  FUSION_PASS_CHECK(!CheckNodeShape(label_output_shape, concat_0_shape_0),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "concat_0 node input 0 shape not support, fusion failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(!CheckNodeShape(label_output_shape, concat_0_shape_1),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "concat_0 node input 1 shape not support, fusion failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(!CheckNodeShape(label_output_shape, concat_0_output_shape),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "concat_0 node output shape not support, fusion failed."),
                    return NOT_CHANGED);
  vector<int64_t> concat_1_shape_0 = concat_1_node->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();
  vector<int64_t> concat_1_shape_1 = concat_1_node->GetOpDesc()->GetInputDesc(1).GetShape().GetDims();
  vector<int64_t> concat_1_output_shape = concat_1_node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();
  int64_t concat_1_dim = 2;
  concat_1_shape_0[concat_1_dim] = dim_1_num;
  concat_1_shape_1[concat_1_dim] = dim_1_num;
  FUSION_PASS_CHECK(!CheckNodeShape(label_output_shape, concat_1_shape_0),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "concat_1 node input 0 shape not support, fusion failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(!CheckNodeShape(label_output_shape, concat_1_shape_1),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "concat_1 node input 1 shape not support, fusion failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(!CheckNodeShape(label_output_shape, concat_1_output_shape),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "concat_1 node output shape not support, fusion failed."),
                    return NOT_CHANGED);

  return SUCCESS;
}

Status SwinTransformerLnQKVFusionPass::CheckConcatNode(vector<ge::NodePtr>& node_ptr_all) {
  ge::NodePtr reshape_0_node = node_ptr_all[1];
  ge::NodePtr slice_0_node = node_ptr_all[2];
  ge::NodePtr slice_1_node = node_ptr_all[3];
  ge::NodePtr concat_0_node = node_ptr_all[4];
  ge::NodePtr slice_2_node = node_ptr_all[5];
  ge::NodePtr slice_3_node = node_ptr_all[6];
  ge::NodePtr concat_1_node = node_ptr_all[7];
  if (concat_0_node == nullptr) {
    if (reshape_0_node != nullptr) {
      FUSION_PASS_CHECK(reshape_0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                        OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_0_node output 0 size is [%lu], which not equal to 1.",
                                reshape_0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                        return NOT_CHANGED);
    }
    return SUCCESS;
  }

  FUSION_PASS_CHECK(reshape_0_node == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_0_node is null."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(reshape_0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 2,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_0_node output 0 size is [%lu], which not equal to 2.",
                            reshape_0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(slice_0_node == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "slice_0_node is null."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(slice_0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "slice_0_node output 0 size is [%lu], which not equal to 1.",
                            slice_0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(slice_1_node == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "slice_1_node is null."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(slice_1_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "slice_1_node output 0 size is [%lu], which not equal to 1.",
                            slice_1_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(concat_0_node == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "concat_0_node is null."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(concat_0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 2,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "concat_0_node output 0 size is [%lu], which not equal to 2.",
                            concat_0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(slice_2_node == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "slice_2_node is null."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(slice_2_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "slice_2_node output 0 size is [%lu], which not equal to 1.",
                            slice_2_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(slice_3_node == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "slice_3_node is null."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(slice_3_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "slice_3_node output 0 size is [%lu], which not equal to 1.",
                            slice_3_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(concat_1_node == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "concat_1_node is null."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(concat_1_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "concat_1_node output 0 size is [%lu], which not equal to 1.",
                            concat_1_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  return CheckRollNode(node_ptr_all);
}

Status SwinTransformerLnQKVFusionPass::SetInputOutputDesc(vector<ge::NodePtr>& node_ptr_all,
                                                          std::shared_ptr<ge::OpDesc>& ln_qkv_desc) {
  ge::NodePtr ln_node = node_ptr_all[0];
  ge::NodePtr batchmatmul_node = node_ptr_all[12];
  ge::NodePtr output_node_0 = node_ptr_all[15];
  ge::NodePtr output_node_1 = node_ptr_all[16];
  ge::NodePtr output_node_2 = node_ptr_all[17];
  ge::GeTensorDesc input_desc_0 = ln_node->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc input_desc_1 = ln_node->GetOpDesc()->GetInputDesc(1);
  ge::GeTensorDesc input_desc_2 = ln_node->GetOpDesc()->GetInputDesc(2);
  ge::GeTensorDesc input_desc_3 = batchmatmul_node->GetOpDesc()->GetInputDesc(1);
  ge::GeTensorDesc input_desc_4 = batchmatmul_node->GetOpDesc()->GetInputDesc(2);

  ge::GeTensorDesc output_desc_0 = output_node_0->GetOpDesc()->GetOutputDesc(0);
  ge::GeTensorDesc output_desc_1 = output_node_1->GetOpDesc()->GetOutputDesc(0);
  ge::GeTensorDesc output_desc_2 = output_node_2->GetOpDesc()->GetOutputDesc(0);

  // input desc
  FUSION_PASS_CHECK(ln_qkv_desc->AddInputDesc(input_desc_0) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input_0 failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(ln_qkv_desc->AddInputDesc(input_desc_1) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input_1 failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(ln_qkv_desc->AddInputDesc(input_desc_2) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input_2 failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(ln_qkv_desc->AddInputDesc(input_desc_3) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input_3 failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(ln_qkv_desc->AddInputDesc(input_desc_4) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input_4 failed."),
                    return NOT_CHANGED);
  // output desc
  FUSION_PASS_CHECK(ln_qkv_desc->AddOutputDesc(output_desc_0) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add output_0 failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(ln_qkv_desc->AddOutputDesc(output_desc_1) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add output_1 failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(ln_qkv_desc->AddOutputDesc(output_desc_2) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add output_2 failed."),
                    return NOT_CHANGED);
  return SUCCESS;
}

Status SwinTransformerLnQKVFusionPass::CheckPattenNode(vector<ge::NodePtr>& node_ptr_all) {
  ge::NodePtr ln_node = node_ptr_all[0];
  FUSION_PASS_CHECK(CheckLayerNormNode(ln_node) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "ln_node check failed."),
                    return NOT_CHANGED);
  ge::NodePtr batchmatmul_node = node_ptr_all[12];
  FUSION_PASS_CHECK(CheckBatchMatmulNode(batchmatmul_node) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "batchmatmul_node check failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(CheckReshapeNode(node_ptr_all) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Reshape_node check failed."),
                    return NOT_CHANGED);
  ge::NodePtr confuse_node = node_ptr_all[13];
  FUSION_PASS_CHECK(CheckConfusionTransposeNode(confuse_node) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "confuse_node check failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(CheckConcatNode(node_ptr_all) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "concat check failed."),
                    return NOT_CHANGED);
  return SUCCESS;
}

Status SwinTransformerLnQKVFusionPass::SetAttrPattenNode(vector<ge::NodePtr>& node_ptr_all,
                                                         ge::NodePtr& ln_qkv_node) {
  ge::NodePtr ln_node = node_ptr_all[0];
  ge::NodePtr transposed_node = node_ptr_all[13];
  // get attr
  std::vector<int64_t> attr_shape;
  AttrUtils::GetListInt(transposed_node->GetOpDesc(), "shape", attr_shape);

  float epsilon = 0;
  AttrUtils::GetFloat(ln_node->GetOpDesc(), "epsilon", epsilon);

  ge::NodePtr concat_0_node = node_ptr_all[4];
  ge::NodePtr concat_1_node = node_ptr_all[7];
  std::vector<int64_t> slice_dim;
  slice_dim.push_back(0);
  int64_t slice_dim_h = 0;
  if (concat_0_node != nullptr) {
    OpDescPtr concat_0_desc = concat_0_node->GetOpDesc();
    GeTensorDesc concat_0_input_1_desc = concat_0_desc->GetInputDesc(1);
    vector<int64_t> concat_0_input_1_shape = concat_0_input_1_desc.GetShape().GetDims();
    slice_dim_h = concat_0_input_1_shape[1];
  }
  int64_t concat_1_concat_dim = 2;
  int64_t slice_dim_w = 0;
  if (concat_1_node != nullptr) {
    OpDescPtr concat_1_desc = concat_1_node->GetOpDesc();
    GeTensorDesc concat_1_input_1_desc = concat_1_desc->GetInputDesc(1);
    vector<int64_t> concat_1_input_1_shape = concat_1_input_1_desc.GetShape().GetDims();
    slice_dim_w = concat_1_input_1_shape[concat_1_concat_dim];
  }
  slice_dim.push_back(slice_dim_h);
  slice_dim.push_back(slice_dim_w);
  slice_dim.push_back(0);
  // set attr
  int64_t head_num_dim = 3;
  int64_t head_dim_dim = 4;
  AttrUtils::SetInt(ln_qkv_node->GetOpDesc(), "head_num", attr_shape[head_num_dim]);
  AttrUtils::SetInt(ln_qkv_node->GetOpDesc(), "head_dim", attr_shape[head_dim_dim]);
  AttrUtils::SetInt(ln_qkv_node->GetOpDesc(), "seq_length", attr_shape[1]);
  AttrUtils::SetListInt(ln_qkv_node->GetOpDesc(), "shifts",slice_dim);
  AttrUtils::SetFloat(ln_qkv_node->GetOpDesc(), "epsilon", epsilon);
  return SUCCESS;
}

Status SwinTransformerLnQKVFusionPass::NewNodeAddEdge(ge::NodePtr& ln_qkv_node,
                                                      vector<ge::NodePtr>& node_ptr_all){
  ge::NodePtr ln_node = node_ptr_all[0];
  ge::NodePtr batchmatmul_node = node_ptr_all[12];
  ge::NodePtr output_node_0 = node_ptr_all[15];
  ge::NodePtr output_node_1 = node_ptr_all[16];
  ge::NodePtr output_node_2 = node_ptr_all[17];
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
      ln_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
      ln_qkv_node->GetInDataAnchor(0)) != SUCCESS,
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
          ln_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
          ln_qkv_node->GetName().c_str()),
      return PARAM_INVALID);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
      ln_node->GetInDataAnchor(1)->GetPeerOutAnchor(),
      ln_qkv_node->GetInDataAnchor(1)) != SUCCESS,
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
          ln_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
          ln_qkv_node->GetName().c_str()),
      return PARAM_INVALID);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
      ln_node->GetInDataAnchor(2)->GetPeerOutAnchor(),
      ln_qkv_node->GetInDataAnchor(2)) != SUCCESS,
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
          ln_node->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
          ln_qkv_node->GetName().c_str()),
      return PARAM_INVALID);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
      batchmatmul_node->GetInDataAnchor(1)->GetPeerOutAnchor(),
      ln_qkv_node->GetInDataAnchor(3)) != SUCCESS,
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
          batchmatmul_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
          ln_qkv_node->GetName().c_str()),
      return PARAM_INVALID);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
      batchmatmul_node->GetInDataAnchor(2)->GetPeerOutAnchor(),
      ln_qkv_node->GetInDataAnchor(4)) != SUCCESS,
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
          batchmatmul_node->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
          ln_qkv_node->GetName().c_str()),
      return PARAM_INVALID);
  // set output edge
  for (auto &inDataAnchor : output_node_0->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(output_node_0->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGI(FUSED_OP_TYPE.c_str(),
                      "Remove out data edge 0 failed."), return PARAM_INVALID);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(ln_qkv_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGI(FUSED_OP_TYPE.c_str(),
                      "Add out data edge 0 failed."), return PARAM_INVALID);
  }

  for (auto &inDataAnchor : output_node_1->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(output_node_1->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGI(FUSED_OP_TYPE.c_str(),
                      "Remove out data edge 1 failed."), return PARAM_INVALID);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(ln_qkv_node->GetOutDataAnchor(1), inDataAnchor) != SUCCESS,
                      OP_LOGI(FUSED_OP_TYPE.c_str(),
                      "Add out data edge 1 failed."), return PARAM_INVALID);
  }
  for (auto &inDataAnchor : output_node_2->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(output_node_2->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGI(FUSED_OP_TYPE.c_str(),
                      "Remove out data edge 2 failed."), return PARAM_INVALID);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(ln_qkv_node->GetOutDataAnchor(2), inDataAnchor) != SUCCESS,
                      OP_LOGI(FUSED_OP_TYPE.c_str(),
                      "Add out data edge 2 failed."), return PARAM_INVALID);
  }
  return SUCCESS;
}

Status SwinTransformerLnQKVFusionPass::Fusion(ge::ComputeGraph &graph,
                                              Mapping &mapping,
                                              vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Start SwinTransformerLnQKVFusionPass.");
  vector<ge::NodePtr> node_ptr_all;
  FUSION_PASS_CHECK(GetNodeFromPatten(mapping, node_ptr_all) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "SwinTransformerLnQKVFusionPass fusion failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(CheckPattenNode(node_ptr_all) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "SwinTransformerLnQKVFusionPass fusion failed."),
                    return NOT_CHANGED);

  std::shared_ptr<ge::OpDesc> ln_qkv_desc = nullptr;
  ge::NodePtr batchmatmul_node = node_ptr_all[12];
  FUSION_PASS_MAKE_SHARED(
    (ln_qkv_desc = std::make_shared<ge::OpDesc>(
      batchmatmul_node->GetName() + "/" + kOpSwinTransformerLnQKV, kOpSwinTransformerLnQKV)
      ), return NOT_CHANGED
  );
  FUSION_PASS_CHECK(ln_qkv_desc == nullptr,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "ln_qkv_desc is null, fusion failed."),
                    return NOT_CHANGED);

  // set input edge
  FUSION_PASS_CHECK(SetInputOutputDesc(node_ptr_all, ln_qkv_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "SwinTransformerLnQKVFusionPass fusion failed."),
                    return NOT_CHANGED);

  // add node
  ge::NodePtr ln_qkv_node = graph.AddNode(ln_qkv_desc);
  fusion_nodes.push_back(ln_qkv_node);

  // set attr
  FUSION_PASS_CHECK(SetAttrPattenNode(node_ptr_all, ln_qkv_node) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "SwinTransformerLnQKVFusionPass fusion failed."),
                    return NOT_CHANGED);

  FUSION_PASS_CHECK(!CheckOpSupported(ln_qkv_node->GetOpDesc()),
                    OP_LOGI(FUSED_OP_TYPE.c_str(),
                    "CheckOpSupported failed, SwinTransformerLnQKVFusionPass fusion failed."),
                    return NOT_CHANGED);

  // set input edge
  FUSION_PASS_CHECK(NewNodeAddEdge(ln_qkv_node, node_ptr_all) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "SwinTransformerLnQKVFusionPass fusion failed."),
                    return PARAM_INVALID);
  for (auto &remove_node : node_ptr_all) {
    if (remove_node != nullptr) {
      FUSION_PASS_CHECK(graph.RemoveNode(remove_node) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove node failed."),
                        return PARAM_INVALID);
    }
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "end SwinTransformerLnQKVFusionPass.");
  return SUCCESS;
}

REGISTER_PASS("SwinTransformerLnQKVFusionPass", BUILT_IN_GRAPH_PASS, SwinTransformerLnQKVFusionPass);
}  // namespace fe
