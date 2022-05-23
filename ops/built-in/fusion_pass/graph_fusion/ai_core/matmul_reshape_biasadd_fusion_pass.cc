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
 * \file matmul_reshape_biasadd_fusion_pass.cpp
 * \brief matmul reshape biasadd fusion (Matmul--reshape--biasadd)
 */
#include "matmul_reshape_biasadd_fusion_pass.h"

#include <string>
#include <vector>

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "error_util.h"

namespace fe {
static const string HAS_BIAS = "has_bias";
static const string PATTERN_MATMUL = "mat_mul";
static const string PATTERN_BIASADD = "bias_add";
static const string PATTERN_RESHAPE = "reshape";
static const string PATTERN_BIAS = "bias";
static const int MATMUL_INPUT_NUM = 2;

static const char* TF_MATMUL = "MatMul";
static const char* TF_MATMULV2 = "MatMulV2";
static const char* BIASADD = "BiasAdd";
static const char* ADD = "Add";
static const char* RESHAPE = "Reshape";


/*
    fusion pattern
            node
                \
                 \
                Matmul---Reshape---BiasAdd/Add
                /
               /
            node
*/
vector<FusionPattern*> MatMulReshapeBiasAddFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  string passName = "MatmulReshapeBiasAddFusion";
  FusionPattern* pattern = new (std::nothrow) FusionPattern(passName);
  FUSION_PASS_CHECK((pattern == nullptr),
                    OP_LOGW(FUSED_OP_TYPE.c_str(),
                    "pattern is nullptr,Create pattern not success!"),
                    return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName.c_str());
  pattern->AddOpDesc(PATTERN_MATMUL, {TF_MATMUL, TF_MATMULV2})
      .AddOpDesc(PATTERN_BIAS)
      .AddOpDesc(PATTERN_BIASADD, {BIASADD, ADD})
      .AddOpDesc(PATTERN_RESHAPE, {RESHAPE})
      .SetInputs(PATTERN_RESHAPE, {PATTERN_MATMUL})
      .SetInputs(PATTERN_BIASADD, {PATTERN_RESHAPE, PATTERN_BIAS})
      .SetOutput(PATTERN_BIASADD);
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName.c_str());
  return patterns;
}

Status MatMulReshapeBiasAddFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                              vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr node_matmul = GetNodeFromMapping(PATTERN_MATMUL, mapping);
  ge::NodePtr node_bias = GetNodeFromMapping(PATTERN_BIAS, mapping);
  ge::NodePtr node_biasadd = GetNodeFromMapping(PATTERN_BIASADD, mapping);
  ge::NodePtr node_reshape = GetNodeFromMapping(PATTERN_RESHAPE, mapping);

  // check matmul node
  FUSION_PASS_CHECK((node_matmul == nullptr), OP_LOGW(FUSED_OP_TYPE.c_str(), "Parameter[node_matmul] can not be null"),
                    return fe::NOT_CHANGED);

  // check node_matmul must have only one output to node_biasadd
  FUSION_PASS_CHECK((node_matmul->GetOutDataNodes().size() != 1),
                    OP_LOGI(node_matmul, "MatMul node should only have 1 output, actual %zu.",
                            node_matmul->GetOutDataNodes().size()),
                    return fe::NOT_CHANGED);

  FUSION_PASS_CHECK((node_bias == nullptr), OP_LOGW(FUSED_OP_TYPE.c_str(), "Parameter[node_bias] can not be null"),
                    return fe::NOT_CHANGED);

  FUSION_PASS_CHECK((node_biasadd == nullptr), OP_LOGW(node_biasadd, "Parameter[node_biasadd] can not be null"),
                    return fe::NOT_CHANGED);

  auto matmul_op_desc = node_matmul->GetOpDesc();
  FUSION_PASS_CHECK((matmul_op_desc == nullptr), OP_LOGW(node_matmul, "Parameter[matmul_op_desc] can not be null"),
                    return fe::NOT_CHANGED);
  // to check support of matmul fusion
  FUSION_PASS_CHECK(!CheckOpSupported(matmul_op_desc),
                    OP_LOGW(node_matmul, "MatMul[%s] is not supported by FE, fusion pass abort.",
                            matmul_op_desc->GetName().c_str()),
                    return fe::NOT_CHANGED);
  // to add node bias as third input, node_matmul must have 2 InDataAnchor
  // and 2 InputDesc(referenced AddLinkFrom())
  FUSION_PASS_CHECK((matmul_op_desc->GetInputsSize() != MATMUL_INPUT_NUM),
                    OP_LOGI(node_matmul, "MatMul node should have 2 inputs, acutal %zu.",
                            node_matmul->GetInAllNodes().size()),
                    return fe::NOT_CHANGED);
  // check whether the input is dynamic mode
  ge::GeShape matmul_output_shape = matmul_op_desc->GetOutputDesc(0).GetShape();
  FUSION_PASS_CHECK((matmul_output_shape.GetDimNum() != 2),
                    OP_LOGI(node_matmul, "The output dim of matmul node must be 2, actual %zu.",
                            matmul_output_shape.GetDimNum()),
                    return fe::NOT_CHANGED);
  FUSION_PASS_CHECK((PatternFusionUtil::IsUnknownShape(matmul_output_shape.GetDim(1)) ||
                     PatternFusionUtil::IsUnknownShape(matmul_output_shape.GetDim(0))),
                    OP_LOGI(node_matmul, "MatmulReshapeBiasAddFusion cannot be applied for unknown shape"),
                    return fe::NOT_CHANGED);

  // check reshape node
  auto reshape_op_desc = node_reshape->GetOpDesc();
  FUSION_PASS_CHECK((reshape_op_desc == nullptr), OP_LOGW(node_reshape, "Parameter[reshape_op_desc] can not be null"),
                    return fe::NOT_CHANGED);
  // check whether reshape node has split the last dim of matmul node output
  ge::GeShape reshape_output_shape = reshape_op_desc->GetOutputDesc(0).GetShape();
  size_t reshape_out_dim = reshape_output_shape.GetDimNum();
  FUSION_PASS_CHECK((reshape_out_dim < 1),
                    OP_LOGI(node_reshape, "The dim ofreshape out shape must be larger than 1."),
                    return fe::NOT_CHANGED);
  // check bias node
  auto biasadd_op_desc = node_biasadd->GetOpDesc();
  FUSION_PASS_CHECK((biasadd_op_desc == nullptr), OP_LOGW(node_biasadd, "Parameter[biasadd_op_desc] can not be null"),
                    return fe::NOT_CHANGED);
  // check biasadd_op_desc should only have one outputTensroDesc
  FUSION_PASS_CHECK((biasadd_op_desc->GetAllOutputsDesc().size() != 1),
                    OP_LOGI(node_biasadd, "BiasAdd node should only have 1 output, actual %zu",
                            biasadd_op_desc->GetAllOutputsDesc().size()),
                    return fe::NOT_CHANGED);
  // check Bias node should only have 1 output, because ge::graph haven't offer
  // method to modify node anchor, only way to add anchor is AddLinkFrom
  FUSION_PASS_CHECK((node_bias->GetAllOutDataAnchors().size() != 1),
                    OP_LOGI(node_bias, "now don't support fusion Bias with over 1 output"),
                    return fe::NOT_CHANGED);

  auto node_biasadd_output_desc = node_biasadd->GetOpDesc()->GetOutputDesc(0);
  auto node_reshape_output_desc = node_reshape->GetOpDesc()->GetOutputDesc(0);
  // check output dtype
  auto node_biasadd_output_dtype = node_biasadd_output_desc.GetDataType();
  auto node_reshape_output_dtype = node_reshape_output_desc.GetDataType();
  FUSION_PASS_CHECK((node_reshape_output_dtype != node_biasadd_output_dtype),
                    OP_LOGI(node_biasadd, "The output dtype of reshape and biasadd is different."),
                    return fe::NOT_CHANGED);

  if (node_biasadd->GetType() == ADD) {
    auto bias_op_desc = node_bias->GetOpDesc();
    FUSION_PASS_CHECK((bias_op_desc == nullptr),
                      OP_LOGW(node_bias, "Parameter[bias_op_desc] can not be null"),
                      return fe::NOT_CHANGED);
    auto bias_output_tensor = bias_op_desc->GetOutputDesc(0);
    auto bias_output_shape_vec = bias_output_tensor.GetShape().GetDims();
    auto reshape_output_shape_vec = reshape_output_shape.GetDims();
    FUSION_PASS_CHECK(reshape_output_shape_vec.size() < bias_output_shape_vec.size(),
                      OP_LOGI(node_reshape, "reshape output dim num should be larger than bias dim num."),
                      return fe::NOT_CHANGED);

    int64_t bias_length = 1;
    for (auto i : bias_output_shape_vec) {
      FUSION_PASS_CHECK((i <= 0), OP_LOGW(node_bias, "bias output shape is invalid."), return fe::NOT_CHANGED);
      FUSION_PASS_CHECK((bias_length * i / i != bias_length),
                        OP_LOGW(node_bias, "bias output shape size exceed int64."), return fe::NOT_CHANGED);
      bias_length *= i;
    }
    FUSION_PASS_CHECK(bias_length != matmul_output_shape.GetDim(1),
                      OP_LOGI(node_bias, "bias dim is not equal with matmul output n dim."), return fe::NOT_CHANGED);
    for (size_t i = 0; i < bias_output_shape_vec.size(); i++) {
      FUSION_PASS_CHECK(bias_output_shape_vec[bias_output_shape_vec.size() - i - 1] !=
                            reshape_output_shape_vec[reshape_output_shape_vec.size() - i - 1],
                        OP_LOGI(node_bias, "bias shape can not broadcast to reshape output shape."),
                        return fe::NOT_CHANGED);
    }
  } else {
    FUSION_PASS_CHECK((matmul_output_shape.GetDim(1) != reshape_output_shape.GetDim(reshape_out_dim - 1)),
                      OP_LOGI(node_matmul, "MatmulReshapeBiasAddFusion only support that do not "
                              "split the last dim of matmul node output when fusion with bias_add node."),
                      return fe::NOT_CHANGED);
  }

  // add HAS_BIAS attr to MatMul, and set value with "true"
  if (ge::AttrUtils::SetBool(matmul_op_desc, HAS_BIAS, true) == false) {
    CUBE_INNER_ERR_REPORT(node_matmul, "set attr:has_bias=true to matmul failed");
    return fe::FAILED;
  }

  // add link from node_bias to node_matmul
  if (node_matmul->AddLinkFrom("bias", node_bias) != ge::GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(node_matmul, "add link from Bias to MatMul failed");
    return fe::FAILED;
  }

  vector<bool> is_input_const;
  for (auto &anchor : node_matmul->GetAllInDataAnchors()) {
    auto peerAnchor = anchor->GetPeerOutAnchor();
    if (peerAnchor == nullptr) {
      continue;
    }
    auto node = peerAnchor->GetOwnerNode();
    string node_type = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(node);
    if (node_type == CONSTANT || node_type == CONSTANTOP) {
      is_input_const.push_back(true);
    } else {
      is_input_const.push_back(false);
    }
  }
  node_matmul->GetOpDesc()->SetIsInputConst(is_input_const);

  // replace src (Reshape -> BiasAdd(0) -> OtherNode) to (MatMul -> OtherNode)
  auto reshapeOutAnchor = node_reshape->GetOutDataAnchor(0);
  if (reshapeOutAnchor == nullptr) {
    CUBE_CALL_ERR_REPORT(node_reshape, "Parameter[reshapeOutAnchor] must not be null.");
    return fe::FAILED;
  }

  auto biasadd_out_anchor0 = node_biasadd->GetOutDataAnchor(0);
  if (biasadd_out_anchor0 == nullptr) {
    CUBE_CALL_ERR_REPORT(node_biasadd, "Parameter[biasadd_out_anchor0] must not be null.");
    return fe::FAILED;
  }
  for (auto &dstAnchor : biasadd_out_anchor0->GetPeerInDataAnchors()) {
    if (dstAnchor == nullptr) {
      CUBE_CALL_ERR_REPORT(node_biasadd, "Parameter[dstAnchor] must not be null.");
      return fe::FAILED;
    }
    if (ge::GraphUtils::RemoveEdge(biasadd_out_anchor0, dstAnchor) != ge::GRAPH_SUCCESS ||
        ge::GraphUtils::AddEdge(reshapeOutAnchor, dstAnchor) != ge::GRAPH_SUCCESS) {
      CUBE_CALL_ERR_REPORT(node_biasadd, "Replace edge src Failed.");
      return fe::FAILED;
    }
  }

  // ensure the format of output does not change
  auto node_biasadd_output_format = node_biasadd_output_desc.GetFormat();
  auto node_reshape_output_format = node_reshape_output_desc.GetFormat();
  OP_LOGD(FUSED_OP_TYPE.c_str(), "node bias_add output format is [%s]",
          ge::TypeUtils::FormatToSerialString(node_biasadd_output_format).c_str());
  OP_LOGD(FUSED_OP_TYPE.c_str(), "node reshape output format is [%s]",
          ge::TypeUtils::FormatToSerialString(node_reshape_output_format).c_str());

  if (node_biasadd_output_format != node_reshape_output_format) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "The output format of biasadd and reshape is different");
    node_reshape_output_desc.SetFormat(node_biasadd_output_format);
    node_reshape_output_desc.SetOriginFormat(node_biasadd_output_format);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Now the format of reshape output is [%s]",
            ge::TypeUtils::FormatToSerialString(node_reshape_output_desc.GetFormat()).c_str());
    FUSION_PASS_CHECK(node_reshape->GetOpDesc()->UpdateOutputDesc(0, node_reshape_output_desc) != ge::GRAPH_SUCCESS,
                      CUBE_CALL_ERR_REPORT(node_reshape, "update reshape output desc of [%s] not success.",
                            node_reshape->GetName().c_str()),
                      return fe::FAILED);
  }

  // delete BiasAdd node
  if (graph.RemoveNode(node_biasadd) != ge::GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(node_biasadd, "delete BiasAdd failed");
    return fe::FAILED;
  }
  fusionNodes.push_back(node_matmul);
  OP_LOGI(node_matmul, "matmul reshape biasadd fusion success!");
  return fe::SUCCESS;
}
REGISTER_PASS("MatMulReshapeBiasAddFusionPass", BUILT_IN_GRAPH_PASS, MatMulReshapeBiasAddFusionPass);
}  // namespace fe
