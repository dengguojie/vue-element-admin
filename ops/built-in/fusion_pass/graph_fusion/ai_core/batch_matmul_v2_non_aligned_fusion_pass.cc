/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
 * \file batch_matmul_v2_non_aligned_fusion_pass.cc
 * \brief
 *
 * pattern 1
 *                                                                                   w
 *                                                                                   |
 *                                                                                 reshape       const
 *                                                                                   |             |
 *                                                                                  pad         reashape
 *                                                                                   |             |
 *             x        w                                                    x     reshape        pad
 *              \       /                                                      \      /            |
 *             batchmatmul_1   const                                       batchmatmul_1       reshape
 *                      \     /                                                         \     /
 *                       add_1                                                           add_1
 *                        |                                                                |
 *                     reshape_1                                                        reahspe_1
 *                        |                                                                |
 *      input_0        transpose_1             -------->                input_0         transpose_1    input_1
 *          \              /                                                  \          /               |
 *           batchmatmul_2                                                    batchmatmul_2            reshape
 *                |                                                                |                     |
 *           transpose_2                                                        transpose               pad
 *                |                                                                |                     |
 *           reshape_2       input_1                                            reshape_2              reshape
 *                \            /                                                       \               /
 *                batchmatmul_3                                                          batchmatmul_3
 *
 *
 * pattern 2
 *                                                                               w                          w
 *                                                                               |                          |
 *                                                                             reshape  const            reshape  const
 *                                                                               |        |                 |       |
 *                                                                              pad     reshape            pad reshape
 *                                                                               |        |                 |       |
 *    x          w                  x         w                         x      reshape   pad       x      reshape  pad
 *      \        /                   \       /                            \       /       |         \      /        |
 *     batchmatmul_1  const         batchmatmul_2   const                batchmatmul_1  reshape    batchmatmul_2 reshape
 *              \    /                       \      /                              \     /                  \     /
 *               add_1                        add_2             -------->           add_1                    add_2
 *                |                             |                                      |                       |
 *             reshape_1                     reshape_2                              reshape_1               reshape_2
 *                |                             |                                      |                        |
 *            transpose_1                   transpsoe_2                            transpose_1             transpose_2
 *                        \               /                                                 \               /
 *                          batchmatmul_3                                                      batchmatmul_3
 */

#include "batch_matmul_v2_non_aligned_fusion_pass.h"

#include "anchor_util.h"
#include "external/graph/operator_factory.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

namespace fe {
static const string PATTERN_BATCHMATMUL_1 = "BatchMatMul_1";
static const string PATTERN_BATCHMATMUL_2 = "BatchMatMul_2";
static const string PATTERN_BATCHMATMUL_3 = "BatchMatMul_3";
static const string PATTERN_TRANSPOSE_1 = "Transpose_1";
static const string PATTERN_TRANSPOSE_2 = "Transpose_2";
static const string PATTERN_INPUT_0 = "Input0";
static const string PATTERN_INPUT_1 = "Input1";
static const string PATTERN_ADD_1 = "Add_1";
static const string PATTERN_ADD_2 = "Add_2";
static const string PATTERN_RESHAPE_1 = "Reshape_1";
static const string PATTERN_RESHAPE_2 = "Reshape_2";

static const string BATCHMATMULV2 = "BatchMatMulV2";
static const string BATCHMATMUL = "BatchMatMul";
static const string TRANSPOSE = "TransposeD";
static const string RESHAPE = "Reshape";
static const string ADD = "Add";

static const int ALIGN_UNIT = 16;
static const int kNumTwo = 2;
static const int kNumThree = 3;

vector<FusionPattern *> BatchMatMulNonAlignedFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern1 = new (std::nothrow) FusionPattern(kNameFusionPass);
  FUSION_PASS_CHECK(pattern1 == nullptr, OP_LOGW(kNameFusionPass.c_str(), "Failed to create pattern 1"),
                    return patterns);

  OP_LOGD(kNameFusionPass.c_str(), "Start to define pattern 1.");
  pattern1->AddOpDesc(PATTERN_BATCHMATMUL_3, {BATCHMATMULV2, BATCHMATMUL})
      .AddOpDesc(PATTERN_RESHAPE_2, {RESHAPE})
      .AddOpDesc(PATTERN_INPUT_1)
      .AddOpDesc(PATTERN_TRANSPOSE_2, {TRANSPOSE})
      .AddOpDesc(PATTERN_BATCHMATMUL_2, {BATCHMATMULV2, BATCHMATMUL})
      .AddOpDesc(PATTERN_INPUT_0)
      .AddOpDesc(PATTERN_TRANSPOSE_1, {TRANSPOSE})
      .AddOpDesc(PATTERN_RESHAPE_1, {RESHAPE})
      .AddOpDesc(PATTERN_ADD_1, {ADD})
      .AddOpDesc(PATTERN_BATCHMATMUL_1, {BATCHMATMULV2, BATCHMATMUL})
      .SetInputs(PATTERN_BATCHMATMUL_3, {PATTERN_RESHAPE_2, PATTERN_INPUT_1})
      .SetInputs(PATTERN_RESHAPE_2, {PATTERN_TRANSPOSE_2})
      .SetInputs(PATTERN_TRANSPOSE_2, {PATTERN_BATCHMATMUL_2})
      .SetInputs(PATTERN_BATCHMATMUL_2, {PATTERN_INPUT_0, PATTERN_TRANSPOSE_1})
      .SetInputs(PATTERN_TRANSPOSE_1, {PATTERN_RESHAPE_1})
      .SetInputs(PATTERN_RESHAPE_1, {PATTERN_ADD_1})
      .SetInputs(PATTERN_ADD_1, {PATTERN_BATCHMATMUL_1})
      .SetOutput(PATTERN_BATCHMATMUL_3);
  patterns.push_back(pattern1);
  OP_LOGD(kNameFusionPass.c_str(), "End to define pattern 1.");

  FusionPattern *pattern2 = new (std::nothrow) FusionPattern(kNameFusionPass);
  FUSION_PASS_CHECK(pattern2 == nullptr, OP_LOGW(kNameFusionPass.c_str(), "Failed to create pattern 2"),
                    return patterns);

  OP_LOGD(kNameFusionPass.c_str(), "Start to define pattern 2.");
  pattern2->AddOpDesc(PATTERN_BATCHMATMUL_3, {BATCHMATMULV2, BATCHMATMUL})
      .AddOpDesc(PATTERN_TRANSPOSE_1, {TRANSPOSE})
      .AddOpDesc(PATTERN_RESHAPE_1, {RESHAPE})
      .AddOpDesc(PATTERN_ADD_1, {ADD})
      .AddOpDesc(PATTERN_BATCHMATMUL_1, {BATCHMATMULV2, BATCHMATMUL})
      .AddOpDesc(PATTERN_TRANSPOSE_2, {TRANSPOSE})
      .AddOpDesc(PATTERN_RESHAPE_2, {RESHAPE})
      .AddOpDesc(PATTERN_ADD_2, {ADD})
      .AddOpDesc(PATTERN_BATCHMATMUL_2, {BATCHMATMULV2, BATCHMATMUL})
      .SetInputs(PATTERN_BATCHMATMUL_3, {PATTERN_TRANSPOSE_1, PATTERN_TRANSPOSE_2})
      .SetInputs(PATTERN_TRANSPOSE_1, {PATTERN_RESHAPE_1})
      .SetInputs(PATTERN_TRANSPOSE_2, {PATTERN_RESHAPE_2})
      .SetInputs(PATTERN_RESHAPE_1, {PATTERN_ADD_1})
      .SetInputs(PATTERN_RESHAPE_2, {PATTERN_ADD_2})
      .SetInputs(PATTERN_ADD_1, {PATTERN_BATCHMATMUL_1})
      .SetInputs(PATTERN_ADD_2, {PATTERN_BATCHMATMUL_2})
      .SetOutput(PATTERN_BATCHMATMUL_3);
  patterns.push_back(pattern2);
  OP_LOGD(kNameFusionPass.c_str(), "End to define pattern 2.");
  return patterns;
}

Status BatchMatMulNonAlignedFusionPass::CheckTransposeDPerm() const {
  vector<int64_t> perm_list_1 = {0, 2, 1, 3};
  FUSION_PASS_CHECK(CheckPerm(transpose_1_node, perm_list_1) != SUCCESS,
                    OP_LOGW(kNameFusionPass.c_str(), "Check %s perm, not match the fusion condition.",
                            transpose_1_node->GetName().c_str()),
                    return NOT_CHANGED);
  vector<int64_t> perm_list_2;
  if (add_2_node == nullptr) {
    perm_list_2 = perm_list_1;
  } else {
    perm_list_2 = {0, 2, 3, 1};
  }
  FUSION_PASS_CHECK(CheckPerm(transpose_2_node, perm_list_2) != SUCCESS,
                    OP_LOGW(kNameFusionPass.c_str(), "Check %s perm, not match the fusion condition.",
                            transpose_2_node->GetName().c_str()),
                    return NOT_CHANGED);
  bool insert_loc_pattern1 = add_2_node == nullptr && CheckInsertLocPattern() == SUCCESS;
  bool insert_loc_pattern2 = add_2_node != nullptr && CheckInsertLocPattern() == SUCCESS;
  bool insert_loc_pattern_fail = !insert_loc_pattern1 && !insert_loc_pattern2;
  FUSION_PASS_CHECK(insert_loc_pattern_fail,
                    OP_LOGW(kNameFusionPass.c_str(), "Check insert position, not match the fusion condition."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(CheckReshapePattern() != SUCCESS,
                    OP_LOGW(kNameFusionPass.c_str(), "Check Reshape node, not match the fusion condition."),
                    return NOT_CHANGED);
  return SUCCESS;
}

Status BatchMatMulNonAlignedFusionPass::CheckBatchMatmulInputNode(const ge::NodePtr& batchmatmul_node) const {
  auto batchmatmul_inputs = batchmatmul_node->GetInDataNodes();
  FUSION_PASS_CHECK(batchmatmul_inputs.size() != kNumTwo,
                    OP_LOGE(kNameFusionPass.c_str(), "%s should only have 2 input, actual %zu.",
                            batchmatmul_node->GetName().c_str(), batchmatmul_inputs.size()),
                    return FAILED);
  ge::NodePtr batchmatmul_input_1_node = batchmatmul_inputs.at(1);
  FUSION_PASS_CHECK(batchmatmul_input_1_node == nullptr,
                    OP_LOGE(kNameFusionPass.c_str(), "The 1 input of %s is null, fusion failed.",
                            batchmatmul_node->GetName().c_str()),
                    return PARAM_INVALID);
  return SUCCESS;
}

Status BatchMatMulNonAlignedFusionPass::GetBatchMatMulShape(map<std::string, int64_t>& batch_matmul_shape_info) {
  bool batchmatmul_1_adj_x2 = false;
  FUSION_PASS_CHECK(
      !AttrUtils::GetBool(batchmatmul_1_node->GetOpDesc(), "adj_x2", batchmatmul_1_adj_x2),
      OP_LOGW(kNameFusionPass.c_str(), "Failed to get adj_x2 of %s.", batchmatmul_1_node->GetName().c_str()),
      return NOT_CHANGED);
  bool batchmatmul_2_adj_x2 = false;
  FUSION_PASS_CHECK(
      !AttrUtils::GetBool(batchmatmul_2_node->GetOpDesc(), "adj_x2", batchmatmul_2_adj_x2),
      OP_LOGW(kNameFusionPass.c_str(), "Failed to get adj_x2 of %s.", batchmatmul_2_node->GetName().c_str()),
      return NOT_CHANGED);
  auto bmm_1_input_1_shape = batchmatmul_1_node->GetOpDesc()->MutableInputDesc(1)->GetOriginShape();
  auto len_bmm_1_input_1_shape = bmm_1_input_1_shape.GetDimNum();
  int64_t bmm_1_n_dim = bmm_1_input_1_shape.GetDim(len_bmm_1_input_1_shape - 1);
  if (batchmatmul_1_adj_x2) {
    bmm_1_n_dim = bmm_1_input_1_shape.GetDim(len_bmm_1_input_1_shape - kNumTwo);
  }

  auto bmm_2_input_1_shape = batchmatmul_2_node->GetOpDesc()->MutableInputDesc(1)->GetOriginShape();
  auto len_bmm_2_input_1_shape = bmm_2_input_1_shape.GetDimNum();
  int64_t bmm_2_n_dim = bmm_2_input_1_shape.GetDim(len_bmm_2_input_1_shape - 1);
  if (batchmatmul_2_adj_x2) {
    bmm_2_n_dim = bmm_2_input_1_shape.GetDim(len_bmm_2_input_1_shape - kNumTwo);
  }
  int64_t bmm_2_n_dim_align;
  int64_t bmm_1_n_dim_align;
  if (add_2_node != nullptr){
    bool batchmatmul_3_adj_x2 = false;
    FUSION_PASS_CHECK(
        !AttrUtils::GetBool(batchmatmul_3_node->GetOpDesc(), "adj_x2", batchmatmul_3_adj_x2),
        OP_LOGW(kNameFusionPass.c_str(), "Failed to get adj_x2 of %s.", batchmatmul_3_node->GetName().c_str()),
        return NOT_CHANGED);
    auto bmm_3_input_1_shape = batchmatmul_3_node->GetOpDesc()->MutableInputDesc(1)->GetOriginShape();
    auto len_bmm_3_input_1_shape = bmm_3_input_1_shape.GetDimNum();
    int64_t bmm_3_k_dim = bmm_3_input_1_shape.GetDim(len_bmm_3_input_1_shape - kNumTwo);
    if (batchmatmul_3_adj_x2) {
      bmm_3_k_dim = bmm_3_input_1_shape.GetDim(len_bmm_3_input_1_shape - 1);
    }

    int64_t bmm_3_k_dim_align = (bmm_3_k_dim + ALIGN_UNIT - 1) / ALIGN_UNIT * ALIGN_UNIT;
    bmm_1_n_dim_align = bmm_1_n_dim / bmm_3_k_dim * bmm_3_k_dim_align;
    bmm_2_n_dim_align = bmm_2_n_dim / bmm_3_k_dim * bmm_3_k_dim_align;
    batch_matmul_shape_info["bmm_3_k_dim"] = bmm_3_k_dim;
    batch_matmul_shape_info["bmm_3_k_dim_align"] = bmm_3_k_dim_align;
    OP_LOGD(kNameFusionPass.c_str(), "%s K dim is %ld", batchmatmul_3_node->GetName().c_str(), bmm_3_k_dim);
    OP_LOGD(kNameFusionPass.c_str(), "After %s K dim alignment is %ld", batchmatmul_3_node->GetName().c_str(),
          bmm_3_k_dim_align);
  } else {
    bmm_2_n_dim_align = (bmm_2_n_dim + ALIGN_UNIT - 1) / ALIGN_UNIT * ALIGN_UNIT;
    bmm_1_n_dim_align = bmm_1_n_dim / bmm_2_n_dim * bmm_2_n_dim_align;
  }
  batch_matmul_shape_info["bmm_2_n_dim"] = bmm_2_n_dim;
  batch_matmul_shape_info["bmm_2_n_dim_align"] = bmm_2_n_dim_align;
  batch_matmul_shape_info["bmm_1_n_dim"] = bmm_1_n_dim;
  batch_matmul_shape_info["bmm_1_n_dim_align"] = bmm_1_n_dim_align;
  OP_LOGD(kNameFusionPass.c_str(), "%s N dim is %ld", batchmatmul_2_node->GetName().c_str(), bmm_2_n_dim);
  OP_LOGD(kNameFusionPass.c_str(), "After %s N dim alignment is %ld", batchmatmul_2_node->GetName().c_str(),
          bmm_2_n_dim_align);
  OP_LOGD(kNameFusionPass.c_str(), "%s N dim is %ld", batchmatmul_1_node->GetName().c_str(), bmm_1_n_dim);
  OP_LOGD(kNameFusionPass.c_str(), "After %s N dim alignment is %ld", batchmatmul_1_node->GetName().c_str(),
          bmm_1_n_dim_align);
  Status ret = CheckBatchMatmulInputNode(batchmatmul_1_node);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(kNameFusionPass.c_str(), "Check BatchMatmul InputNode failed."), return ret);
  return SUCCESS;
}

Status BatchMatMulNonAlignedFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                                               vector<ge::NodePtr> & fusion_nodes) {
  OP_LOGI(kNameFusionPass.c_str(), "Enter BatchMatMulNonAlignedFusionPass.");
  FUSION_PASS_CHECK(GetNodes(mapping) != SUCCESS, OP_LOGW(kNameFusionPass.c_str(), "Failed to get Nodes."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(CheckStaticShape() != SUCCESS, OP_LOGW(kNameFusionPass.c_str(), "There is an unknown shape node."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(CheckBatchMatMul() != SUCCESS, OP_LOGW(kNameFusionPass.c_str(), "Failed to check BatchMatMul."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(CheckTransposeDPerm() != SUCCESS,
                  OP_LOGW(kNameFusionPass.c_str(), "Check failed."), return NOT_CHANGED);
  int64_t bmm_1_n_dim, bmm_2_n_dim, bmm_3_k_dim, bmm_1_n_dim_align, bmm_2_n_dim_align, bmm_3_k_dim_align;
  map<std::string, int64_t> batch_matmul_shape_info = {
    {"bmm_1_n_dim", bmm_1_n_dim}, {"bmm_2_n_dim", bmm_2_n_dim}, {"bmm_3_k_dim", bmm_3_k_dim},
    {"bmm_1_n_dim_align", bmm_1_n_dim_align}, {"bmm_2_n_dim_align", bmm_2_n_dim_align},
    {"bmm_3_k_dim_align", bmm_3_k_dim_align}
  };
  Status ret = GetBatchMatMulShape(batch_matmul_shape_info);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(kNameFusionPass.c_str(), "get bacthmatmul shape failed."),
                    return ret);
  if (add_2_node == nullptr) {
    FUSION_PASS_CHECK(DoFusionPattern1(graph, batch_matmul_shape_info) != SUCCESS,
                      OP_LOGW(kNameFusionPass.c_str(), "Pattern 1, fusion failed."), return FAILED);
  } else {
    FUSION_PASS_CHECK(DoFusionPattern2(graph, batch_matmul_shape_info) != SUCCESS,
                      OP_LOGW(kNameFusionPass.c_str(), "Pattern 2, fusion failed."),  return FAILED);
  }

  // modify ori_format
  batchmatmul_1_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  batchmatmul_2_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  batchmatmul_3_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  OP_LOGI(kNameFusionPass.c_str(), "BatchMatMulNonAlignedFusionPass Success.");
  return SUCCESS;
}

Status BatchMatMulNonAlignedFusionPass::CheckPerm(const ge::NodePtr &transpose_node,
                                                  const vector<int64_t> &perm_list) const {
  OP_LOGI(kNameFusionPass.c_str(), "Enter CheckPerm.");
  Operator transpose_op = ge::OpDescUtils::CreateOperatorFromNode(transpose_node);
  vector<int64_t> cur_perm_list;
  FUSION_PASS_CHECK(transpose_op.GetAttr("perm", cur_perm_list) != ge::GRAPH_SUCCESS,
                    OP_LOGW(kNameFusionPass.c_str(), "Failed to get perm of %s.", transpose_node->GetName().c_str()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(
      cur_perm_list.size() != 4,
      OP_LOGW(kNameFusionPass.c_str(), "%s, length of perm should be 4.", transpose_node->GetName().c_str()),
      return NOT_CHANGED);
  FUSION_PASS_CHECK(
      perm_list != cur_perm_list,
      OP_LOGW(kNameFusionPass.c_str(), "%s, Support perm is (%ld, %ld, %ld, %ld), but actual is (%ld, %ld, %ld, %ld).",
              transpose_node->GetName().c_str(), perm_list[0], perm_list[1], perm_list[kNumTwo], perm_list[kNumThree],
              cur_perm_list[0], cur_perm_list[1], cur_perm_list[kNumTwo], cur_perm_list[kNumThree]),
      return NOT_CHANGED);
  OP_LOGI(kNameFusionPass.c_str(), "End CheckPerm.");
  return SUCCESS;
}

Status BatchMatMulNonAlignedFusionPass::CheckBatchMatMul() const {
  OP_LOGI(kNameFusionPass.c_str(), "Enter CheckBatchMatMul.");
  // Check whether the K of the batchmatmul_3_node is not aligned with 16
  bool batchmatmul_3_adj_x1 = false;
  FUSION_PASS_CHECK(
      !AttrUtils::GetBool(batchmatmul_3_node->GetOpDesc(), "adj_x1", batchmatmul_3_adj_x1),
      OP_LOGW(kNameFusionPass.c_str(), "Failed to get adj_x1 of %s.", batchmatmul_3_node->GetName().c_str()),
      return NOT_CHANGED);

  auto bmm_3_in_shape_1 = batchmatmul_3_node->GetOpDesc()->MutableInputDesc(0)->GetOriginShape();
  auto len_bmm_3_in_shape_1 = bmm_3_in_shape_1.GetDimNum();
  int64_t bmm_3_k_dim = bmm_3_in_shape_1.GetDim(len_bmm_3_in_shape_1 - 1);
  int64_t bmm_3_m_dim = bmm_3_in_shape_1.GetDim(len_bmm_3_in_shape_1 - kNumTwo);
  if (batchmatmul_3_adj_x1) {
    bmm_3_k_dim = bmm_3_in_shape_1.GetDim(len_bmm_3_in_shape_1 - kNumTwo);
    bmm_3_m_dim = bmm_3_in_shape_1.GetDim(len_bmm_3_in_shape_1 - 1);
  }
  FUSION_PASS_CHECK(bmm_3_k_dim % ALIGN_UNIT == 0,
                    OP_LOGW(kNameFusionPass.c_str(), "K dim of %s is aligned, not match the fusion condition.",
                            batchmatmul_3_node->GetName().c_str()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(bmm_3_m_dim % ALIGN_UNIT != 0,
                    OP_LOGW(kNameFusionPass.c_str(), "M dim of %s is not aligned, not match the fusion condition.",
                            batchmatmul_3_node->GetName().c_str()),
                    return NOT_CHANGED);
  OP_LOGI(kNameFusionPass.c_str(), "End CheckBatchMatMul.");
  return SUCCESS;
}

Status BatchMatMulNonAlignedFusionPass::CheckStaticShape() const {
  OP_LOGI(kNameFusionPass.c_str(), "Enter CheckStaticShape.");
  FUSION_PASS_CHECK(
      CheckNodeShape(batchmatmul_1_node) != SUCCESS,
      OP_LOGW(kNameFusionPass.c_str(), "Check %s orishape and shape fail.", batchmatmul_1_node->GetName().c_str()),
      return NOT_CHANGED);
  FUSION_PASS_CHECK(
      CheckNodeShape(batchmatmul_2_node) != SUCCESS,
      OP_LOGW(kNameFusionPass.c_str(), "Check %s orishape and shape fail.", batchmatmul_2_node->GetName().c_str()),
      return NOT_CHANGED);
  FUSION_PASS_CHECK(
      CheckNodeShape(batchmatmul_3_node) != SUCCESS,
      OP_LOGW(kNameFusionPass.c_str(), "Check %s orishape and shape fail.", batchmatmul_3_node->GetName().c_str()),
      return NOT_CHANGED);
  FUSION_PASS_CHECK(
      CheckNodeShape(transpose_1_node) != SUCCESS,
      OP_LOGW(kNameFusionPass.c_str(), "Check %s orishape and shape fail.", transpose_1_node->GetName().c_str()),
      return NOT_CHANGED);
  FUSION_PASS_CHECK(
      CheckNodeShape(transpose_2_node) != SUCCESS,
      OP_LOGW(kNameFusionPass.c_str(), "Check %s orishape and shape fail.", transpose_2_node->GetName().c_str()),
      return NOT_CHANGED);
  FUSION_PASS_CHECK(
      CheckNodeShape(reshape_1_node) != SUCCESS,
      OP_LOGW(kNameFusionPass.c_str(), "Check %s orishape and shape fail.", reshape_1_node->GetName().c_str()),
      return NOT_CHANGED);
  FUSION_PASS_CHECK(
      CheckNodeShape(reshape_2_node) != SUCCESS,
      OP_LOGW(kNameFusionPass.c_str(), "Check %s orishape and shape fail.", reshape_2_node->GetName().c_str()),
      return NOT_CHANGED);
  FUSION_PASS_CHECK(
      CheckNodeShape(add_1_node) != SUCCESS,
      OP_LOGW(kNameFusionPass.c_str(), "Check %s orishape and shape fail.", add_1_node->GetName().c_str()),
      return NOT_CHANGED);
  if (add_2_node != nullptr) {
    FUSION_PASS_CHECK(
        CheckNodeShape(add_2_node) != SUCCESS,
        OP_LOGW(kNameFusionPass.c_str(), "Check %s orishape and shape fail.", add_2_node->GetName().c_str()),
        return NOT_CHANGED);
  }
  OP_LOGI(kNameFusionPass.c_str(), "End CheckStaticShape.");
  return SUCCESS;
}

Status BatchMatMulNonAlignedFusionPass::CheckNodeShape(const ge::NodePtr &node) const {
  auto node_inputs = node->GetInDataNodes();
  ge::OpDescPtr node_desc = node->GetOpDesc();
  for (size_t i = 0; i < node_inputs.size(); i++) {
    ge::GeShape input_ori_shape = node_desc->MutableInputDesc(i)->GetOriginShape();
    FUSION_PASS_CHECK(
        input_ori_shape.IsUnknownShape(),
        OP_LOGW(kNameFusionPass.c_str(), "%zu input of %s origin shape is unknow shape.", i, node->GetName().c_str()),
        return NOT_CHANGED);
    ge::GeShape input_shape = node_desc->MutableInputDesc(i)->GetShape();
    FUSION_PASS_CHECK(
        input_shape.IsUnknownShape(),
        OP_LOGW(kNameFusionPass.c_str(), "%zu input of %s shape is unknow shape.", i, node->GetName().c_str()),
        return NOT_CHANGED);
  }
  return SUCCESS;
}

Status BatchMatMulNonAlignedFusionPass::CheckInsertLocPattern() const {
  FUSION_PASS_CHECK(OpDescUtils::IsNonConstInput(batchmatmul_1_node, 1),
                    OP_LOGW(kNameFusionPass.c_str(), "input of %s is not const node, not match the fusion condition.",
                            batchmatmul_1_node->GetName().c_str()),
                    return NOT_CHANGED);
  auto in_nodes_add_1 = add_1_node->GetInNodes();
  auto in_node_1_add_1 = in_nodes_add_1.at(0);
  size_t const_add_input_index = 0;
  if (in_node_1_add_1->GetType() == BATCHMATMULV2 || in_node_1_add_1->GetType() == BATCHMATMUL) {
    const_add_input_index = 1;
  }
  FUSION_PASS_CHECK(OpDescUtils::IsNonConstInput(add_1_node, const_add_input_index),
                    OP_LOGW(kNameFusionPass.c_str(), "input of %s is not const node, not match the fusion condition.",
                            add_1_node->GetName().c_str()),
                    return NOT_CHANGED);
  if (add_2_node == nullptr) {
    FUSION_PASS_CHECK(OpDescUtils::IsNonConstInput(batchmatmul_3_node, 1),
                      OP_LOGW(kNameFusionPass.c_str(), "input of %s is not const node, not match the fusion condition.",
                              batchmatmul_3_node->GetName().c_str()),
                      return NOT_CHANGED);
  } else {
    FUSION_PASS_CHECK(OpDescUtils::IsNonConstInput(batchmatmul_2_node, 1),
                      OP_LOGW(kNameFusionPass.c_str(), "input of %s is not const node, not match the fusion condition.",
                              batchmatmul_2_node->GetName().c_str()),
                      return NOT_CHANGED);
    auto in_nodes_add_2 = add_2_node->GetInNodes();
    auto in_node_1_add_2 = in_nodes_add_2.at(0);
    size_t const_add_2_input_index = 0;
    if (in_node_1_add_2->GetType() == BATCHMATMULV2 || in_node_1_add_2->GetType() == BATCHMATMUL) {
      const_add_2_input_index = 1;
    }
    FUSION_PASS_CHECK(OpDescUtils::IsNonConstInput(add_2_node, const_add_2_input_index),
                      OP_LOGW(kNameFusionPass.c_str(), "input of %s is not const node, not match the fusion condition.",
                              add_2_node->GetName().c_str()),
                      return NOT_CHANGED);
  }
  return SUCCESS;
}

Status BatchMatMulNonAlignedFusionPass::CheckReshapePattern() const {
  size_t dim_3 = 3;
  size_t dim_4 = 4;
  GeShape &reshape_1_in_shape = reshape_1_node->GetOpDesc()->MutableInputDesc(0)->MutableShape();
  auto len_reshape_1_in_shape = reshape_1_in_shape.GetDimNum();
  FUSION_PASS_CHECK(
      len_reshape_1_in_shape != dim_3,
      OP_LOGW(kNameFusionPass.c_str(), "Input of %s is not 3 dimensional, not match the fusion condition.",
              reshape_1_node->GetName().c_str()),
      return NOT_CHANGED);
  int64_t before_reshape_dim = reshape_1_in_shape.GetDim(len_reshape_1_in_shape - 1);

  GeShape &reshape_1_out_shape = reshape_1_node->GetOpDesc()->MutableOutputDesc(0)->MutableShape();
  auto len_reshape_1_out_shape = reshape_1_out_shape.GetDimNum();
  FUSION_PASS_CHECK(
      len_reshape_1_out_shape != dim_4,
      OP_LOGW(kNameFusionPass.c_str(), "output of %s is not 4 dimensional, not match the fusion condition.",
              reshape_1_node->GetName().c_str()),
      return NOT_CHANGED);
  int64_t after_reshape_dim_1 = reshape_1_out_shape.GetDim(len_reshape_1_out_shape - 1);
  int64_t after_reshape_dim_2 = reshape_1_out_shape.GetDim(len_reshape_1_out_shape - kNumTwo);
  FUSION_PASS_CHECK(
      (after_reshape_dim_1 * after_reshape_dim_2) != before_reshape_dim,
      OP_LOGW(kNameFusionPass.c_str(), "%s , not match the fusion condition.", reshape_1_node->GetName().c_str()),
      return NOT_CHANGED);

  GeShape &reshape_2_in_shape = reshape_2_node->GetOpDesc()->MutableInputDesc(0)->MutableShape();
  auto len_reshape_2_in_shape = reshape_2_in_shape.GetDimNum();

  GeShape &reshape_2_out_shape = reshape_2_node->GetOpDesc()->MutableOutputDesc(0)->MutableShape();
  auto len_reshape_2_out_shape = reshape_2_out_shape.GetDimNum();
  if (add_2_node == nullptr) {
    FUSION_PASS_CHECK(
        len_reshape_2_in_shape != dim_4,
        OP_LOGW(kNameFusionPass.c_str(), "Input of %s is not 4 dimensional, not match the fusion condition.",
                reshape_2_node->GetName().c_str()),
        return NOT_CHANGED);
    int64_t before_reshape_dim_1 = reshape_2_in_shape.GetDim(len_reshape_2_in_shape - 1);
    int64_t before_reshape_dim_2 = reshape_2_in_shape.GetDim(len_reshape_2_in_shape - kNumTwo);
    FUSION_PASS_CHECK(
        len_reshape_2_out_shape != dim_3,
        OP_LOGW(kNameFusionPass.c_str(), "output of %s is not 3 dimensional, not match the fusion condition.",
                reshape_2_node->GetName().c_str()),
        return NOT_CHANGED);
    int64_t after_reshape_dim = reshape_2_out_shape.GetDim(len_reshape_2_out_shape - 1);
    FUSION_PASS_CHECK(
        (before_reshape_dim_1 * before_reshape_dim_2) != after_reshape_dim,
        OP_LOGW(kNameFusionPass.c_str(), "%s , not match the fusion condition.", reshape_2_node->GetName().c_str()),
        return NOT_CHANGED);
  } else {
    FUSION_PASS_CHECK(
        len_reshape_2_in_shape != dim_3,
        OP_LOGW(kNameFusionPass.c_str(), "Input of %s is not 3 dimensional, not match the fusion condition.",
                reshape_2_node->GetName().c_str()),
        return NOT_CHANGED);
    before_reshape_dim = reshape_2_in_shape.GetDim(len_reshape_2_in_shape - 1);
    FUSION_PASS_CHECK(
        len_reshape_2_out_shape != dim_4,
        OP_LOGW(kNameFusionPass.c_str(), "output of %s is not 4 dimensional, not match the fusion condition.",
                reshape_2_node->GetName().c_str()),
        return NOT_CHANGED);
    after_reshape_dim_1 = reshape_2_out_shape.GetDim(len_reshape_2_out_shape - 1);
    after_reshape_dim_2 = reshape_2_out_shape.GetDim(len_reshape_2_out_shape - kNumTwo);
    FUSION_PASS_CHECK(
        (after_reshape_dim_1 * after_reshape_dim_2) != before_reshape_dim,
        OP_LOGW(kNameFusionPass.c_str(), "%s , not match the fusion condition.", reshape_2_node->GetName().c_str()),
        return NOT_CHANGED);
  }
  return SUCCESS;
}

Status BatchMatMulNonAlignedFusionPass::GetNodes(const Mapping &mapping) {
  batchmatmul_1_node = GetNodeFromMapping(PATTERN_BATCHMATMUL_1, mapping);
  FUSION_PASS_CHECK(batchmatmul_1_node == nullptr, OP_LOGE(kNameFusionPass.c_str(), "Fuse node is null, fusion failed"),
                    return PARAM_INVALID);
  batchmatmul_2_node = GetNodeFromMapping(PATTERN_BATCHMATMUL_2, mapping);
  FUSION_PASS_CHECK(batchmatmul_2_node == nullptr, OP_LOGE(kNameFusionPass.c_str(), "Fuse node is null, fusion failed"),
                    return PARAM_INVALID);
  batchmatmul_3_node = GetNodeFromMapping(PATTERN_BATCHMATMUL_3, mapping);
  FUSION_PASS_CHECK(batchmatmul_3_node == nullptr, OP_LOGE(kNameFusionPass.c_str(), "Fuse node is null, fusion failed"),
                    return PARAM_INVALID);
  transpose_1_node = GetNodeFromMapping(PATTERN_TRANSPOSE_1, mapping);
  FUSION_PASS_CHECK(transpose_1_node == nullptr, OP_LOGE(kNameFusionPass.c_str(), "Fuse node is null, fusion failed"),
                    return PARAM_INVALID);
  transpose_2_node = GetNodeFromMapping(PATTERN_TRANSPOSE_2, mapping);
  FUSION_PASS_CHECK(transpose_2_node == nullptr, OP_LOGE(kNameFusionPass.c_str(), "Fuse node is null, fusion failed"),
                    return PARAM_INVALID);
  reshape_1_node = GetNodeFromMapping(PATTERN_RESHAPE_1, mapping);
  FUSION_PASS_CHECK(reshape_1_node == nullptr, OP_LOGE(kNameFusionPass.c_str(), "Fuse node is null, fusion failed"),
                    return PARAM_INVALID);
  reshape_2_node = GetNodeFromMapping(PATTERN_RESHAPE_2, mapping);
  FUSION_PASS_CHECK(reshape_2_node == nullptr, OP_LOGE(kNameFusionPass.c_str(), "Fuse node is null, fusion failed"),
                    return PARAM_INVALID);
  add_1_node = GetNodeFromMapping(PATTERN_ADD_1, mapping);
  FUSION_PASS_CHECK(add_1_node == nullptr, OP_LOGE(kNameFusionPass.c_str(), "Fuse node is null, fusion failed"),
                    return PARAM_INVALID);
  add_2_node = GetNodeFromMapping(PATTERN_ADD_2, mapping);

  return SUCCESS;
}

Status BatchMatMulNonAlignedFusionPass::DoFusionPattern1(ge::ComputeGraph &graph,
                                                         map<std::string, int64_t>& batch_matmul_shape_info) const {
  OP_LOGI(kNameFusionPass.c_str(), "Enter DoFusionPattern1.");

  // Create reshape1 pad reshape2 for batchmatmul_1_input_1_node
  auto bmm_1_input_1_shape = batchmatmul_1_node->GetOpDesc()->MutableInputDesc(1)->GetOriginShape();
  auto bmm_1_input_1_dims = bmm_1_input_1_shape.GetDims();
  int64_t tmp_dim = batch_matmul_shape_info["bmm_1_n_dim"] / batch_matmul_shape_info["bmm_2_n_dim"];
  vector<vector<int64_t>> paddings;
  paddings = {{0, 0}, {0, 0},
              {0, batch_matmul_shape_info["bmm_2_n_dim_align"] - batch_matmul_shape_info["bmm_2_n_dim"]}};
  map<string, vector<int64_t>> shape_info;
  shape_info["reshape_shape_1"] = {bmm_1_input_1_dims[0], tmp_dim, batch_matmul_shape_info["bmm_2_n_dim"]};
  shape_info["pad_shape"] = {bmm_1_input_1_dims[0], tmp_dim, batch_matmul_shape_info["bmm_2_n_dim_align"]};
  shape_info["reshape_shape_2"] = {bmm_1_input_1_dims[0], batch_matmul_shape_info["bmm_1_n_dim_align"]};
  auto dst_anchor_batchmatmul = batchmatmul_1_node->GetInDataAnchor(1);
  FUSION_PASS_CHECK(
      CreateReshapePadReshape(graph, dst_anchor_batchmatmul, shape_info, paddings) != SUCCESS,
      OP_LOGW(kNameFusionPass.c_str(), "Create Reshape Pad Reshape for %s fail", batchmatmul_1_node->GetName().c_str()),
      return FAILED);

  // Create reshape1 pad reshape2 for add_1_node
  paddings = {{0, 0}, {0, batch_matmul_shape_info["bmm_2_n_dim_align"] - batch_matmul_shape_info["bmm_2_n_dim"]}};
  shape_info.clear();
  shape_info["reshape_shape_1"] = {tmp_dim, batch_matmul_shape_info["bmm_2_n_dim"]};
  shape_info["pad_shape"] = {tmp_dim, batch_matmul_shape_info["bmm_2_n_dim_align"]};
  shape_info["reshape_shape_2"] = {batch_matmul_shape_info["bmm_1_n_dim_align"]};
  auto in_nodes_add_1 = add_1_node->GetInNodes();
  auto in_node_1_add_1 = in_nodes_add_1.at(0);
  size_t const_add_input_index = 0;
  if (in_node_1_add_1->GetType() == BATCHMATMULV2 || in_node_1_add_1->GetType() == BATCHMATMUL) {
    const_add_input_index = 1;
  }
  auto dst_anchor_add = add_1_node->GetInDataAnchor(const_add_input_index);
  FUSION_PASS_CHECK(
      CreateReshapePadReshape(graph, dst_anchor_add, shape_info, paddings) != SUCCESS,
      OP_LOGW(kNameFusionPass.c_str(), "Create Reshape Pad Reshape for %s fail", add_1_node->GetName().c_str()),
      return FAILED);

  // Create reshape1 pad reshape2 for batchmatmul_3_node
  auto bmm_3_input_1_shape = batchmatmul_3_node->GetOpDesc()->MutableInputDesc(1)->GetOriginShape();
  auto len_bmm_3_input_1_shape = bmm_3_input_1_shape.GetDimNum();
  auto bmm_3_input_1_dims = bmm_3_input_1_shape.GetDims();
  paddings = {{0, 0},
              {0, batch_matmul_shape_info["bmm_2_n_dim_align"] - batch_matmul_shape_info["bmm_2_n_dim"]}, {0, 0}};
  shape_info.clear();
  shape_info["reshape_shape_1"] = {tmp_dim, batch_matmul_shape_info["bmm_2_n_dim"],
                                   bmm_3_input_1_dims[len_bmm_3_input_1_shape - 1]};
  shape_info["pad_shape"] = {tmp_dim, batch_matmul_shape_info["bmm_2_n_dim_align"],
                             bmm_3_input_1_dims[len_bmm_3_input_1_shape - 1]};
  shape_info["reshape_shape_2"] = {batch_matmul_shape_info["bmm_1_n_dim_align"],
                                   bmm_3_input_1_dims[len_bmm_3_input_1_shape - 1]};
  auto dst_anchor_bmm3 = batchmatmul_3_node->GetInDataAnchor(1);
  FUSION_PASS_CHECK(
      CreateReshapePadReshape(graph, dst_anchor_bmm3, shape_info, paddings) != SUCCESS,
      OP_LOGW(kNameFusionPass.c_str(), "Create Reshape Pad Reshape for %s fail", batchmatmul_3_node->GetName().c_str()),
      return FAILED);

  // Update the const input of the reshape node
  auto reshape_const_shape = reshape_1_node->GetOpDesc()->MutableOutputDesc(0)->GetOriginShape();
  auto reshape_const_shape_dims = reshape_const_shape.GetDims();
  vector<int64_t> tmp_dims = reshape_const_shape_dims;
  tmp_dims[kNumThree] = batch_matmul_shape_info["bmm_2_n_dim_align"];
  FUSION_PASS_CHECK(
      UpdateConst(reshape_1_node, tmp_dims) != SUCCESS,
      OP_LOGW(kNameFusionPass.c_str(), "Update the const input of %s fail", reshape_1_node->GetName().c_str()),
      return FAILED);

  auto reshape_2_const_shape = reshape_2_node->GetOpDesc()->MutableOutputDesc(0)->GetOriginShape();
  auto reshape_2_const_shape_dims = reshape_2_const_shape.GetDims();
  vector<int64_t> tmp_2_dims = reshape_2_const_shape_dims;
  tmp_2_dims[kNumTwo] = batch_matmul_shape_info["bmm_1_n_dim_align"];
  FUSION_PASS_CHECK(
      UpdateConst(reshape_2_node, tmp_2_dims) != SUCCESS,
      OP_LOGW(kNameFusionPass.c_str(), "Update the const input of %s fail", reshape_2_node->GetName().c_str()),
      return FAILED);

  ge::NodePtr cur_node = batchmatmul_1_node;
  auto end_node_outputs = batchmatmul_3_node->GetOutDataNodes();
  ge::NodePtr end_node = nullptr;
  if (end_node_outputs.size() == 1) {
    end_node = end_node_outputs.at(0);
  }
  FUSION_PASS_CHECK(UpdateAllShape(cur_node, end_node) != SUCCESS,
                    OP_LOGW(kNameFusionPass.c_str(), "Failed to update shape in pattern 1"), return FAILED);
  OP_LOGI(kNameFusionPass.c_str(), "End DoFusionPattern1.");
  return SUCCESS;
}

Status BatchMatMulNonAlignedFusionPass::DoFusionPattern2(ge::ComputeGraph &graph,
                                                         map<std::string, int64_t>& batch_matmul_shape_info) const {
  OP_LOGI(kNameFusionPass.c_str(), "Enter DoFusionPattern2.");

  // Create reshape1 pad reshape2 for batchmatmul_1_input_1_node
  auto bmm_1_input_1_shape = batchmatmul_1_node->GetOpDesc()->MutableInputDesc(1)->GetOriginShape();
  auto bmm_1_input_1_dims = bmm_1_input_1_shape.GetDims();
  int64_t tmp_dim = batch_matmul_shape_info["bmm_1_n_dim"] / batch_matmul_shape_info["bmm_3_k_dim"];
  vector<vector<int64_t>> paddings;
  paddings = {{0, 0}, {0, 0},
              {0, batch_matmul_shape_info["bmm_3_k_dim_align"] - batch_matmul_shape_info["bmm_3_k_dim"]}};
  map<string, vector<int64_t>> shape_info;
  shape_info["reshape_shape_1"] = {bmm_1_input_1_dims[0], tmp_dim, batch_matmul_shape_info["bmm_3_k_dim"]};
  shape_info["pad_shape"] = {bmm_1_input_1_dims[0], tmp_dim, batch_matmul_shape_info["bmm_3_k_dim_align"]};
  shape_info["reshape_shape_2"] = {bmm_1_input_1_dims[0], batch_matmul_shape_info["bmm_1_n_dim_align"]};
  auto dst_anchor_batchmatmul = batchmatmul_1_node->GetInDataAnchor(1);
  FUSION_PASS_CHECK(
      CreateReshapePadReshape(graph, dst_anchor_batchmatmul, shape_info, paddings) != SUCCESS,
      OP_LOGW(kNameFusionPass.c_str(), "Create Reshape Pad Reshape for %s fail", batchmatmul_1_node->GetName().c_str()),
      return FAILED);

  // Create reshape1 pad reshape2 for add_1_node
  paddings = {{0, 0},
              {0, batch_matmul_shape_info["bmm_3_k_dim_align"] - batch_matmul_shape_info["bmm_3_k_dim"]}};
  shape_info.clear();
  shape_info["reshape_shape_1"] = {tmp_dim, batch_matmul_shape_info["bmm_3_k_dim"]};
  shape_info["pad_shape"] = {tmp_dim, batch_matmul_shape_info["bmm_3_k_dim_align"]};
  shape_info["reshape_shape_2"] = {batch_matmul_shape_info["bmm_1_n_dim_align"]};
  auto in_nodes_add_1 = add_1_node->GetInNodes();
  auto in_node_1_add_1 = in_nodes_add_1.at(0);
  size_t const_add_input_index = 0;
  if (in_node_1_add_1->GetType() == BATCHMATMULV2 || in_node_1_add_1->GetType() == BATCHMATMUL) {
    const_add_input_index = 1;
  }
  auto dst_anchor_add = add_1_node->GetInDataAnchor(const_add_input_index);
  FUSION_PASS_CHECK(
      CreateReshapePadReshape(graph, dst_anchor_add, shape_info, paddings) != SUCCESS,
      OP_LOGW(kNameFusionPass.c_str(), "Create Reshape Pad Reshape for %s fail", add_1_node->GetName().c_str()),
      return FAILED);

  // Insert reshape_pad_reshape at position 2, w input of batchmatmul_2_node
  Status ret = CheckBatchMatmulInputNode(batchmatmul_2_node);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(kNameFusionPass.c_str(), "Check BatchMatmul InputNode failed."), return ret);
  // Create reshape1 pad reshape2 for batchmatmul_2_input_1_node
  auto bmm_2_input_1_shape = batchmatmul_2_node->GetOpDesc()->MutableInputDesc(1)->GetOriginShape();
  auto bmm_2_input_1_dims = bmm_2_input_1_shape.GetDims();
  tmp_dim = batch_matmul_shape_info["bmm_2_n_dim"] / batch_matmul_shape_info["bmm_3_k_dim"];
  paddings = {{0, 0}, {0, 0},
              {0, batch_matmul_shape_info["bmm_3_k_dim_align"] - batch_matmul_shape_info["bmm_3_k_dim"]}};
  shape_info.clear();
  shape_info["reshape_shape_1"] = {bmm_2_input_1_dims[0], tmp_dim, batch_matmul_shape_info["bmm_3_k_dim"]};
  shape_info["pad_shape"] = {bmm_2_input_1_dims[0], tmp_dim, batch_matmul_shape_info["bmm_3_k_dim_align"]};
  shape_info["reshape_shape_2"] = {bmm_2_input_1_dims[0], batch_matmul_shape_info["bmm_2_n_dim_align"]};
  auto dst_anchor_batchmatmul_2 = batchmatmul_2_node->GetInDataAnchor(1);
  FUSION_PASS_CHECK(
      CreateReshapePadReshape(graph, dst_anchor_batchmatmul_2, shape_info, paddings) != SUCCESS,
      OP_LOGW(kNameFusionPass.c_str(), "Create Reshape Pad Reshape for %s fail", batchmatmul_2_node->GetName().c_str()),
      return FAILED);

  // Create reshape1 pad reshape2 for add_2_node
  paddings = {{0, 0}, {0, batch_matmul_shape_info["bmm_3_k_dim_align"] - batch_matmul_shape_info["bmm_3_k_dim"]}};
  auto in_nodes_add_2 = add_2_node->GetInNodes();
  auto in_node_1_add_2 = in_nodes_add_2.at(0);
  size_t const_add_2_input_index = 0;
  if (in_node_1_add_2->GetType() == BATCHMATMULV2 || in_node_1_add_2->GetType() == BATCHMATMUL) {
    const_add_2_input_index = 1;
  }
  auto dst_anchor_add_2 = add_2_node->GetInDataAnchor(const_add_2_input_index);
  shape_info.clear();
  shape_info["reshape_shape_1"] = {tmp_dim, batch_matmul_shape_info["bmm_3_k_dim"]};
  shape_info["pad_shape"] = {tmp_dim, batch_matmul_shape_info["bmm_3_k_dim_align"]};
  shape_info["reshape_shape_2"] = {batch_matmul_shape_info["bmm_2_n_dim_align"]};
  FUSION_PASS_CHECK(
      CreateReshapePadReshape(graph, dst_anchor_add_2, shape_info, paddings) != SUCCESS,
      OP_LOGW(kNameFusionPass.c_str(), "Create Reshape Pad Reshape for %s fail", add_2_node->GetName().c_str()),
      return FAILED);

  // Update the const input of the reshape node
  auto reshape_const_shape = reshape_1_node->GetOpDesc()->MutableOutputDesc(0)->GetOriginShape();
  auto reshape_const_shape_dims = reshape_const_shape.GetDims();
  vector<int64_t> tmp_dims = reshape_const_shape_dims;
  tmp_dims[kNumThree] = batch_matmul_shape_info["bmm_3_k_dim_align"];
  FUSION_PASS_CHECK(
      UpdateConst(reshape_1_node, tmp_dims) != SUCCESS,
      OP_LOGW(kNameFusionPass.c_str(), "Update the const input of %s fail", reshape_1_node->GetName().c_str()),
      return FAILED);

  auto reshape_2_const_shape = reshape_2_node->GetOpDesc()->MutableOutputDesc(0)->GetOriginShape();
  auto reshape_2_const_shape_dims = reshape_2_const_shape.GetDims();
  vector<int64_t> tmp_2_dims = reshape_2_const_shape_dims;
  tmp_2_dims[kNumThree] = batch_matmul_shape_info["bmm_3_k_dim_align"];
  FUSION_PASS_CHECK(
      UpdateConst(reshape_2_node, tmp_2_dims) != SUCCESS,
      OP_LOGW(kNameFusionPass.c_str(), "Update the const input of %s fail", reshape_2_node->GetName().c_str()),
      return FAILED);

  ge::NodePtr cur_node = batchmatmul_1_node;
  ge::NodePtr end_node = batchmatmul_3_node;
  FUSION_PASS_CHECK(UpdateAllShape(cur_node, end_node) != SUCCESS,
                    OP_LOGW(kNameFusionPass.c_str(), "Failed to update shape in pattern 2"), return FAILED);

  cur_node = batchmatmul_2_node;
  auto end_node_outputs = batchmatmul_3_node->GetOutDataNodes();
  if (end_node_outputs.size() == 1) {
    end_node = end_node_outputs.at(0);
  }
  FUSION_PASS_CHECK(UpdateAllShape(cur_node, end_node) != SUCCESS,
                    OP_LOGW(kNameFusionPass.c_str(), "Failed to update shape in pattern 2"), return FAILED);
  OP_LOGI(kNameFusionPass.c_str(), "End DoFusionPattern2.");
  return SUCCESS;
}

Status BatchMatMulNonAlignedFusionPass::CreatePadDNode(ge::ComputeGraph &graph, const OutDataAnchorPtr &out_anchor,
                                                       const vector<int64_t> &shape, ge::NodePtr &pad_node,
                                                       vector<vector<int64_t>> &paddings) const {
  auto previous_node = out_anchor->GetOwnerNode();
  int idx = out_anchor->GetIdx();
  auto previous_node_desc = previous_node->GetOpDesc()->MutableOutputDesc(idx);

  std::string op_name(previous_node->GetName() + "/PadD");
  auto pad_op = ge::OperatorFactory::CreateOperator(op_name.c_str(), "PadD");
  FUSION_PASS_CHECK(pad_op.IsEmpty(), OP_LOGE("Create PadD Op operator error"), return FAILED);
  auto pad_desc = ge::OpDescUtils::GetOpDescFromOperator(pad_op);
  pad_op.BreakConnect();

  pad_desc->MutableInputDesc(0)->SetDataType(previous_node_desc->GetDataType());
  pad_desc->MutableInputDesc(0)->SetFormat(previous_node_desc->GetFormat());
  pad_desc->MutableInputDesc(0)->SetOriginFormat(previous_node_desc->GetOriginFormat());
  pad_desc->MutableInputDesc(0)->SetShape(previous_node_desc->GetShape());
  pad_desc->MutableInputDesc(0)->SetOriginShape(previous_node_desc->GetOriginShape());

  pad_desc->MutableOutputDesc(0)->SetDataType(previous_node_desc->GetDataType());
  pad_desc->MutableOutputDesc(0)->SetFormat(previous_node_desc->GetFormat());
  pad_desc->MutableOutputDesc(0)->SetOriginFormat(previous_node_desc->GetOriginFormat());
  pad_desc->MutableOutputDesc(0)->SetShape(ge::GeShape(shape));
  pad_desc->MutableOutputDesc(0)->SetOriginShape(ge::GeShape(shape));

  ge::AttrUtils::SetListListInt(pad_desc, "paddings", paddings);

  auto new_pad_node = graph.AddNode(pad_desc);
  FUSION_PASS_CHECK(new_pad_node == nullptr, OP_LOGE(kNameFusionPass.c_str(), "failed to add PadD to graph."),
                    return FAILED);
  pad_node = new_pad_node;
  return SUCCESS;
}

Status BatchMatMulNonAlignedFusionPass::CreateReshapeNode(ge::ComputeGraph &graph, const OutDataAnchorPtr &out_anchor,
                                                          const vector<int64_t> &shape, ge::NodePtr &shape_node) const {
  auto previous_node = out_anchor->GetOwnerNode();
  int idx = out_anchor->GetIdx();
  auto previous_node_desc = previous_node->GetOpDesc()->MutableOutputDesc(idx);

  std::string op_name(previous_node->GetName() + "/Reshape");
  auto reshape_op = ge::OperatorFactory::CreateOperator(op_name.c_str(), "Reshape");
  FUSION_PASS_CHECK(reshape_op.IsEmpty(), OP_LOGE("Create Reshape Op operator error"), return FAILED);
  auto reshape_desc = ge::OpDescUtils::GetOpDescFromOperator(reshape_op);
  reshape_op.BreakConnect();

  reshape_desc->MutableInputDesc(0)->SetDataType(previous_node_desc->GetDataType());
  reshape_desc->MutableInputDesc(0)->SetFormat(previous_node_desc->GetFormat());
  reshape_desc->MutableInputDesc(0)->SetOriginFormat(previous_node_desc->GetOriginFormat());
  reshape_desc->MutableInputDesc(0)->SetShape(previous_node_desc->GetShape());
  reshape_desc->MutableInputDesc(0)->SetOriginShape(previous_node_desc->GetOriginShape());

  reshape_desc->MutableOutputDesc(0)->SetDataType(previous_node_desc->GetDataType());
  reshape_desc->MutableOutputDesc(0)->SetFormat(previous_node_desc->GetFormat());
  reshape_desc->MutableOutputDesc(0)->SetOriginFormat(previous_node_desc->GetOriginFormat());
  reshape_desc->MutableOutputDesc(0)->SetShape(ge::GeShape(shape));
  reshape_desc->MutableOutputDesc(0)->SetOriginShape(ge::GeShape(shape));

  int64_t dim = shape.size();
  ge::GeShape shapeShape = ge::GeShape({dim});

  reshape_desc->MutableInputDesc(1)->SetDataType(ge::DT_INT64);
  reshape_desc->MutableInputDesc(1)->SetFormat(ge::FORMAT_ND);
  reshape_desc->MutableInputDesc(1)->SetOriginFormat(ge::FORMAT_ND);
  reshape_desc->MutableInputDesc(1)->SetShape(shapeShape);
  reshape_desc->MutableInputDesc(1)->SetOriginShape(shapeShape);

  std::vector<string> dep_inputs = {"shape"};
  reshape_desc->SetOpInferDepends(dep_inputs);

  ge::OpDescPtr const_opdesc = CreateListConstDesc(previous_node->GetName() + "/Reshape_Const", shape);
  FUSION_PASS_CHECK(const_opdesc == nullptr, OP_LOGE("Create Const Op operator error"), return FAILED);
  auto new_shape_node = graph.AddNode(reshape_desc);
  FUSION_PASS_CHECK(new_shape_node == nullptr, OP_LOGE(kNameFusionPass.c_str(), "failed to add Reshape to graph."),
                    return FAILED);
  auto new_shape_const_node = graph.AddNode(const_opdesc);
  FUSION_PASS_CHECK(new_shape_const_node == nullptr,
                    OP_LOGE(kNameFusionPass.c_str(), "failed to add Reshape_Const to graph."), return FAILED);
  ge::GraphUtils::AddEdge(new_shape_const_node->GetOutDataAnchor(0), new_shape_node->GetInDataAnchor(1));

  shape_node = new_shape_node;
  return SUCCESS;
}

ge::OpDescPtr BatchMatMulNonAlignedFusionPass::CreateListConstDesc(const string &name, vector<int64_t> values) const {
  OpDescPtr const_op_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((const_op_desc = std::make_shared<ge::OpDesc>(name, "Const")), return nullptr);

  GeTensorDesc data_desc(GeShape({static_cast<int64_t>(values.size())}), FORMAT_ND, DT_INT64);
  GeTensorPtr const_value = nullptr;
  FUSION_PASS_MAKE_SHARED((const_value = std::make_shared<ge::GeTensor>(
                               data_desc, reinterpret_cast<uint8_t *>(values.data()), sizeof(int64_t) * values.size())),
                          return nullptr);
  if (const_value == nullptr) {
    return nullptr;
  }
  if (!AttrUtils::SetTensor(const_op_desc, ATTR_NAME_WEIGHTS, const_value)) {
    return nullptr;
  }

  if (const_op_desc->AddOutputDesc("y", data_desc) != GRAPH_SUCCESS) {
    return nullptr;
  }

  return const_op_desc;
}

Status BatchMatMulNonAlignedFusionPass::CreateReshapePadReshape(ge::ComputeGraph &graph,
                                                                const InDataAnchorPtr &dst_anchor,
                                                                map<string, vector<int64_t>> &shape_dict,
                                                                vector<vector<int64_t>> &paddings) const {
  vector<int64_t> reshape_shape_1 = shape_dict["reshape_shape_1"];
  vector<int64_t> pad_shape = shape_dict["pad_shape"];
  vector<int64_t> reshape_shape_2 = shape_dict["reshape_shape_2"];
  ge::NodePtr reshape_node_1 = nullptr;
  ge::NodePtr pad_node = nullptr;
  ge::NodePtr reshape_node_2 = nullptr;
  auto src_anchor = dst_anchor->GetPeerOutAnchor();
  FUSION_PASS_CHECK(CreateReshapeNode(graph, src_anchor, reshape_shape_1, reshape_node_1) != SUCCESS,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to create Reshape node"), return FAILED);
  FUSION_PASS_CHECK(
      ge::GraphUtils::InsertNodeBetweenDataAnchors(src_anchor, dst_anchor, reshape_node_1) != ge::GRAPH_SUCCESS,
      OP_LOGE(kNameFusionPass.c_str(), "Failed to insert Reshape node"), return FAILED);

  src_anchor = dst_anchor->GetPeerOutAnchor();
  FUSION_PASS_CHECK(CreatePadDNode(graph, src_anchor, pad_shape, pad_node, paddings) != SUCCESS,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to create PadD node"), return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::InsertNodeBetweenDataAnchors(src_anchor, dst_anchor, pad_node) != ge::GRAPH_SUCCESS,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to insert PadD node"), return FAILED);

  src_anchor = dst_anchor->GetPeerOutAnchor();
  FUSION_PASS_CHECK(CreateReshapeNode(graph, src_anchor, reshape_shape_2, reshape_node_2) != SUCCESS,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to create Reshape node"), return FAILED);
  FUSION_PASS_CHECK(
      ge::GraphUtils::InsertNodeBetweenDataAnchors(src_anchor, dst_anchor, reshape_node_2) != ge::GRAPH_SUCCESS,
      OP_LOGE(kNameFusionPass.c_str(), "Failed to insert Reshape node"), return FAILED);
  return SUCCESS;
}

Status BatchMatMulNonAlignedFusionPass::UpdateConst(const ge::NodePtr &shape_node, vector<int64_t> &const_shape) const {
  vector<GeTensorPtr> const_reshape = ge::OpDescUtils::MutableWeights(shape_node);
  FUSION_PASS_CHECK(const_shape.empty(), OP_LOGE(kNameFusionPass.c_str(), "const of reshape is nullptr"),
                    return PARAM_INVALID);
  GeTensorPtr const_ptr = const_reshape[0];
  FUSION_PASS_CHECK(const_ptr == nullptr, OP_LOGE(kNameFusionPass.c_str(), "const_ptr of reshape is nullptr"),
                    return PARAM_INVALID);
  ge::DataType const_type = const_ptr->GetTensorDesc().GetDataType();
  if (const_type == ge::DT_INT32) {
    int32_t *const_data = (int32_t *)(const_ptr->GetData().GetData());
    FUSION_PASS_CHECK(const_data == nullptr, OP_LOGE(kNameFusionPass.c_str(), "const_data of reshape is nullptr"),
                      return PARAM_INVALID);
    size_t dim = const_shape.size();
    for (size_t i = 0; i < dim; i++) {
      const_data[i] = (int32_t)const_shape[i];
    }
    const_ptr->SetData(reinterpret_cast<uint8_t *>(const_data), dim * sizeof(int32_t));
  } else {
    int64_t *const_data = (int64_t *)(const_ptr->GetData().GetData());
    FUSION_PASS_CHECK(const_data == nullptr, OP_LOGE(kNameFusionPass.c_str(), "const_data of reshape is nullptr"),
                      return PARAM_INVALID);
    size_t dim = const_shape.size();
    for (size_t i = 0; i < dim; i++) {
      const_data[i] = const_shape[i];
    }
    const_ptr->SetData(reinterpret_cast<uint8_t *>(const_data), dim * sizeof(int64_t));
  }

  return SUCCESS;
}

Status BatchMatMulNonAlignedFusionPass::UpdateAllShape(ge::NodePtr &cur_node, const ge::NodePtr &end_node) const {
  while (cur_node != end_node) {
    // get cur node input size
    auto node_inputs = cur_node->GetInDataNodes();
    for (size_t i = 0; i < node_inputs.size(); i++) {
      // get output shape of input node
      auto node_input = node_inputs.at(i);
      auto node_input_shape_ori = node_input->GetOpDesc()->MutableOutputDesc(0)->GetOriginShape();
      cur_node->GetOpDesc()->MutableInputDesc(i)->SetOriginShape(node_input_shape_ori);
      auto node_input_shape = node_input->GetOpDesc()->MutableOutputDesc(0)->GetShape();
      cur_node->GetOpDesc()->MutableInputDesc(i)->SetShape(node_input_shape);
    }
    FUSION_PASS_CHECK(cur_node->InferShapeAndType() != ge::GRAPH_SUCCESS,
                      OP_LOGE(kNameFusionPass.c_str(), "%s infershape failed", cur_node->GetName().c_str()),
                      return FAILED);
    cur_node = cur_node->GetOutDataNodes().at(0);
  }
  return SUCCESS;
}

REGISTER_PASS("BatchMatMulNonAlignedFusionPass", BUILT_IN_GRAPH_PASS, BatchMatMulNonAlignedFusionPass);
}  // namespace fe
