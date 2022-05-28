/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file matmul_biasadd_fusion_pass.cpp
 * \brief matmul biasadd fusion pass(matmul --> biasadd)
 */
#include "matmul_biasadd_fusion_pass.h"

#include <string>
#include <vector>

#include "anchor_util.h"
#include "error_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

namespace fe {
static const string kHasBias = "has_bias";
static const string kPatternMatmul = "mat_mul";
static const string kPatternBiasadd = "bias_add";
static const string kPatternBias = "bias";
static const int kMatmulInputNum = 2;

static const char* kTfMatmul = "MatMul";
static const char* kTfMatmulV2 = "MatMulV2";
static const char* kTfBatchMatmul = "BatchMatMul";
static const char* kTfBatchMatmulV2 = "BatchMatMulV2";
static const char* kBiasAdd = "BiasAdd";
static const char* kAdd = "Add";

vector<FusionPattern*> MatMulBiasAddFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("MatMulBiasAddFusion");
  if (pattern == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "pattern is nullptr,Create pattern not success.");
    return patterns;
  }

  pattern->AddOpDesc(kPatternMatmul, {kTfMatmul, kTfMatmulV2, kTfBatchMatmul, kTfBatchMatmulV2})
      .AddOpDesc(kPatternBias)
      .AddOpDesc(kPatternBiasadd, {kBiasAdd, kAdd})
      .SetInputs(kPatternBiasadd, {kPatternMatmul, kPatternBias})
      .SetOutput(kPatternBiasadd);
  patterns.push_back(pattern);

  return patterns;
}

bool MatMulBiasAddFusionPass::CheckRange(const std::vector<std::pair<int64_t, int64_t>> &ranges) const {
  for (auto cur_range : ranges) {
    if (cur_range.second == -1) {
      return false;
    }
  }
  return true;
}

bool MatMulBiasAddFusionPass::IsNorange(ge::OpDescPtr &matmul_opdesc) const {
  ge::GeShape first_input_shape = matmul_opdesc->MutableInputDesc(0)->GetShape();
  ge::GeShape second_input_shape = matmul_opdesc->MutableInputDesc(1)->GetShape();
  if (first_input_shape.IsUnknownShape() || second_input_shape.IsUnknownShape()) {
    return true;
  }

  std::vector<std::pair<int64_t, int64_t>> input0_ranges;
  std::vector<std::pair<int64_t, int64_t>> input1_ranges;
  matmul_opdesc->MutableInputDesc(0)->GetShapeRange(input0_ranges);
  matmul_opdesc->MutableInputDesc(1)->GetShapeRange(input1_ranges);
  return (!CheckRange(input0_ranges) || !CheckRange(input1_ranges));
}

Status MatMulBiasAddFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusion_nodes) {
  ge::NodePtr node_matmul = GetNodeFromMapping(kPatternMatmul, mapping);
  ge::NodePtr node_bias = GetNodeFromMapping(kPatternBias, mapping);
  ge::NodePtr node_biasadd = GetNodeFromMapping(kPatternBiasadd, mapping);
  FUSION_PASS_CHECK(node_matmul == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Parameter[node_matmul] must not be null."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(node_bias == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Parameter[node_bias] must not be null."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(node_biasadd == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Parameter[node_biasadd] must not be null."), return NOT_CHANGED);

  auto biasadd_opdesc = node_biasadd->GetOpDesc();
  auto matmul_opdesc = node_matmul->GetOpDesc();
  FUSION_PASS_CHECK(biasadd_opdesc == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Parameter[biasadd_opdesc] must not be null."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(matmul_opdesc == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Parameter[matmul_opdesc] must not be null."), return NOT_CHANGED);
  FUSION_PASS_CHECK(
      !CheckOpSupported(matmul_opdesc),
      OP_LOGW(node_matmul, "Matmul[%s] is not supported by FE, fusion abort.", matmul_opdesc->GetName().c_str()),
      return NOT_CHANGED);
  if (node_biasadd->GetType() == kAdd) {
    auto input0_desc = GetCurrNodeInputDesc(node_biasadd, DIMENSION_0);
    auto input1_desc = GetCurrNodeInputDesc(node_biasadd, DIMENSION_1);
    FUSION_PASS_CHECK(input0_desc == nullptr, OP_LOGW(node_biasadd, "inputDesc0 is null"), return NOT_CHANGED);
    FUSION_PASS_CHECK(input1_desc == nullptr, OP_LOGW(node_biasadd, "inputDesc1 is null"), return NOT_CHANGED);
    ge::GeShape first_input_shape = input0_desc->GetShape();
    ge::GeShape second_input_shape = input1_desc->GetShape();
    FUSION_PASS_CHECK(first_input_shape.GetDims().size() != 1 && second_input_shape.GetDims().size() != 1,
                      OP_LOGI(node_biasadd, "Add input is not scalar"), return NOT_CHANGED);
    int64_t bias_dim;
    int64_t input_ndim;
    if (second_input_shape.GetDims().size() == 1) {
      bool valid_op_type = node_matmul->GetType() == kTfMatmul || node_matmul->GetType() == kTfMatmulV2;
      if (valid_op_type) {
        FUSION_PASS_CHECK(first_input_shape.GetDims().size() != 2,
                          OP_LOGI(node_biasadd, "Matmul output shape no match."), return NOT_CHANGED);
      }
      bool unvalid_shape = PatternFusionUtil::IsUnknownShape(second_input_shape.GetDim(DIMENSION_0)) ||
                           PatternFusionUtil::IsUnknownShape(first_input_shape.GetDim(DIMENSION_1));
      if (unvalid_shape) {
        OP_LOGI(node_biasadd, "MatMulBiasAddFusionPass cannot be applied for unknown shape.");
        return NOT_CHANGED;
      }
      bias_dim = second_input_shape.GetDim(DIMENSION_0);
      uint32_t input_dim_length = first_input_shape.GetDims().size();
      input_ndim = first_input_shape.GetDim(input_dim_length - 1);
    } else {
      bool valid_op_type = node_matmul->GetType() == kTfMatmul || node_matmul->GetType() == kTfMatmulV2;
      if (valid_op_type) {
        FUSION_PASS_CHECK(second_input_shape.GetDims().size() != 2,
                          OP_LOGI(node_matmul, "Matmul output shape no match."), return NOT_CHANGED);
      }
      bool unvalid_shape = PatternFusionUtil::IsUnknownShape(second_input_shape.GetDim(DIMENSION_1)) ||
                           PatternFusionUtil::IsUnknownShape(first_input_shape.GetDim(DIMENSION_0));
      if (unvalid_shape) {
        OP_LOGI(node_biasadd, "MatMulBiasAddFusionPass cannot be applied for unknown shape.");
        return NOT_CHANGED;
      }
      bias_dim = first_input_shape.GetDim(DIMENSION_0);
      uint32_t input_dim_length = second_input_shape.GetDims().size();
      input_ndim = second_input_shape.GetDim(input_dim_length - 1);
    }
    FUSION_PASS_CHECK(
        bias_dim != input_ndim,
        OP_LOGI(node_biasadd, "bias shape %ld, is not equal to input second dim %ld.", bias_dim, input_ndim),
        return NOT_CHANGED);
  }
  // to add node bias as third input, node_matmul must have 2 InDataAnchor
  // and 2 InputDesc(referenced AddLinkFrom())
  FUSION_PASS_CHECK(
      matmul_opdesc->GetInputsSize() != kMatmulInputNum,
      OP_LOGI(node_matmul, "MatMul node should have 2 inputs, acutal %zu", node_matmul->GetInAllNodes().size()),
      return NOT_CHANGED);

  // check nodeMatMul must have range
  FUSION_PASS_CHECK(
      IsNorange(matmul_opdesc) && node_bias->GetOpDesc()->MutableOutputDesc(0)->GetDataType() == DT_FLOAT16,
      OP_LOGI(FUSED_OP_TYPE.c_str(), "MatMul node should have range when bias is fp16."), return NOT_CHANGED);

  // check node_matmul must have only one output to node_biasadd
  FUSION_PASS_CHECK(node_matmul->GetOutDataNodes().size() != 1,
                    OP_LOGI(node_matmul, "MatMul node should only have 1 output, actual %zu",
                            node_matmul->GetOutDataNodes().size()), return NOT_CHANGED);

  // check biasadd_opdesc should only have one outputTensroDesc
  FUSION_PASS_CHECK(biasadd_opdesc->GetAllOutputsDesc().size() != 1,
                    OP_LOGI(node_biasadd, "BiasAdd node should only have 1 output, actual %zu",
                            biasadd_opdesc->GetAllOutputsDesc().size()),
                    return NOT_CHANGED);

  // check Bias node should only have 1 output, because ge::graph haven't offer
  // method to modify node anchor, only way to add anchor is AddLinkFrom
  FUSION_PASS_CHECK(node_bias->GetAllOutDataAnchors().size() != 1,
                    OP_LOGI(node_bias, "now don't support fusion Bias with over 1 output"), return NOT_CHANGED);

  // add has_bias attr to MatMul, and set value with "true"
  FUSION_PASS_CHECK(ge::AttrUtils::SetBool(matmul_opdesc, kHasBias, true) == false,
                    CUBE_INNER_ERR_REPORT(node_matmul, "set attr:has_bias=true to matmul failed"), return FAILED);

  // add link from node_bias to node_matmul,x3 is the name of third input of
  // MatMul in IR matmul.h
  FUSION_PASS_CHECK(node_matmul->AddLinkFrom("bias", node_bias) != ge::GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(node_matmul, "add link from Bias to MatMul failed"), return FAILED);

  vector<bool> is_input_const;
  for (auto &anchor : node_matmul->GetAllInDataAnchors()) {
    auto peer_anchor = anchor->GetPeerOutAnchor();
    if (peer_anchor == nullptr) {
      continue;
    }
    auto node = peer_anchor->GetOwnerNode();
    string node_type = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(node);
    if (node_type == CONSTANT || node_type == CONSTANTOP) {
      is_input_const.push_back(true);
    } else {
      is_input_const.push_back(false);
    }
  }
  node_matmul->GetOpDesc()->SetIsInputConst(is_input_const);
  // replace src (BiasAdd(0) -> OtherNode) to (MatMul -> OtherNode)
  auto matmul_outanchor = node_matmul->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(matmul_outanchor == nullptr,
                    CUBE_CALL_ERR_REPORT(node_matmul, "Parameter[matmul_outanchor] must not be null."), return FAILED);
  auto biasadd_outanchor0 = node_biasadd->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(biasadd_outanchor0 == nullptr,
                    CUBE_CALL_ERR_REPORT(node_matmul, "Parameter[biasadd_outanchor0] must not be null."),
                    return FAILED);
  for (auto &dst_anchor : biasadd_outanchor0->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(dst_anchor == nullptr,
                      CUBE_CALL_ERR_REPORT(node_matmul, "Parameter[dst_anchor] must not be null."), return FAILED);
    if (ge::GraphUtils::RemoveEdge(biasadd_outanchor0, dst_anchor) != ge::GRAPH_SUCCESS ||
        ge::GraphUtils::AddEdge(matmul_outanchor, dst_anchor) != ge::GRAPH_SUCCESS) {
      CUBE_CALL_ERR_REPORT(node_matmul, "Replace edge src Failed.");
      return FAILED;
    }
  }

  // delete BiasAdd node
  if (graph.RemoveNode(node_biasadd) != ge::GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(node_biasadd, "delete BiasAdd failed");
    return FAILED;
  }
  fusion_nodes.push_back(node_matmul);
  OP_LOGI(node_matmul, "matmul biasadd fusion success!");
  return SUCCESS;
}
REGISTER_PASS("MatMulBiasAddFusionPass", BUILT_IN_GRAPH_PASS, MatMulBiasAddFusionPass);
}  // namespace fe
