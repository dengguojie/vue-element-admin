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
static const string HAS_BIAS = "has_bias";
static const string PATTERN_MATMUL = "mat_mul";
static const string PATTERN_BIASADD = "bias_add";
static const string PATTERN_BIAS = "bias";
static const int MATMUL_INPUT_NUM = 2;

static const char* TF_MATMUL = "MatMul";
static const char* TF_MATMULV2 = "MatMulV2";
static const char* BIASADD = "BiasAdd";
static const char* ADD = "Add";

vector<FusionPattern*> MatMulBiasAddFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("MatMulBiasAddFusion");
  if (pattern == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "pattern is nullptr,Create pattern not success.");
    return patterns;
  }

  pattern->AddOpDesc(PATTERN_MATMUL, {TF_MATMUL, TF_MATMULV2})
      .AddOpDesc(PATTERN_BIAS)
      .AddOpDesc(PATTERN_BIASADD, {BIASADD, ADD})
      .SetInputs(PATTERN_BIASADD, {PATTERN_MATMUL, PATTERN_BIAS})
      .SetOutput(PATTERN_BIASADD);
  patterns.push_back(pattern);

  return patterns;
}

Status MatMulBiasAddFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr nodeMatMul = GetNodeFromMapping(PATTERN_MATMUL, mapping);
  ge::NodePtr nodeBias = GetNodeFromMapping(PATTERN_BIAS, mapping);
  ge::NodePtr nodeBiasAdd = GetNodeFromMapping(PATTERN_BIASADD, mapping);

  if (nodeMatMul == nullptr) {
    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[nodeMatMul] must not be null.");
    return fe::PARAM_INVALID;
  }
  if (nodeBias == nullptr) {
    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[nodeBias] must not be null.");
    return fe::PARAM_INVALID;
  }
  if (nodeBiasAdd == nullptr) {
    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[nodeBiasAdd] must not be null.");
    return fe::PARAM_INVALID;
  }

  auto biasAddOpDesc = nodeBiasAdd->GetOpDesc();
  if (biasAddOpDesc == nullptr) {
    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[biasAddOpDesc] must not be null.");
    return fe::PARAM_INVALID;
  }
  auto matMulOpDesc = nodeMatMul->GetOpDesc();
  if (matMulOpDesc == nullptr) {
    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[matMulOpDesc] must not be null.");
    return fe::PARAM_INVALID;
  }

  FUSION_PASS_CHECK(!CheckOpSupported(matMulOpDesc),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Matmul[%s] is not supported by FE, fusion abort.",
                            matMulOpDesc->GetName().c_str()),
                    return NOT_CHANGED);

  if (nodeBiasAdd->GetType() == ADD) {
    string nodeType = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(nodeBias);
    FUSION_PASS_CHECK(nodeType != CONSTANT && nodeType != CONSTANTOP,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "bias is not const node"), return NOT_CHANGED);
    auto input0desc = GetCurrNodeInputDesc(nodeBiasAdd, 0);
    auto input1desc = GetCurrNodeInputDesc(nodeBiasAdd, 1);
    FUSION_PASS_CHECK(input0desc == nullptr,
                  CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputDesc0 is null"),
                  return FAILED);
    FUSION_PASS_CHECK(input1desc == nullptr,
                  CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputDesc1 is null"),
                  return FAILED);
    ge::GeShape inputShape = input0desc->GetShape();
    ge::GeShape biasShape = input1desc->GetShape();
    FUSION_PASS_CHECK(biasShape.GetDims().size() != 1 && inputShape.GetDims().size() != 1,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "Add input is not scalar"), return NOT_CHANGED);
    if (biasShape.GetDims().size() == 1) {
      FUSION_PASS_CHECK(inputShape.GetDims().size() != 2,
                        OP_LOGI(FUSED_OP_TYPE.c_str(), "Matmul output shape not martch."), return NOT_CHANGED);
      if (PatternFusionUtil::IsUnknownShape(biasShape.GetDim(0)) ||
          PatternFusionUtil::IsUnknownShape(inputShape.GetDim(1))) {
        OP_LOGW(FUSED_OP_TYPE.c_str(), "MatMulBiasAddFusionPass cannot be applied for unknown shape.");
        return NOT_CHANGED;
      }
      uint32_t biasDim = biasShape.GetDim(0);
      FUSION_PASS_CHECK(biasDim != inputShape.GetDim(1),
                        OP_LOGI(FUSED_OP_TYPE.c_str(), "bias shape is not equal to input second dim."),
                        return NOT_CHANGED);
    } else {
      FUSION_PASS_CHECK(biasShape.GetDims().size() != 2,
                        OP_LOGI(FUSED_OP_TYPE.c_str(), "Matmul output shape not martch."), return NOT_CHANGED);
      if (PatternFusionUtil::IsUnknownShape(biasShape.GetDim(1)) ||
          PatternFusionUtil::IsUnknownShape(inputShape.GetDim(0))) {
        OP_LOGW(FUSED_OP_TYPE.c_str(), "MatMulBiasAddFusionPass cannot be applied for unknown shape.");
        return NOT_CHANGED;
      }
      uint32_t biasDim = inputShape.GetDim(0);
      FUSION_PASS_CHECK(biasDim != biasShape.GetDim(1),
                        OP_LOGI(FUSED_OP_TYPE.c_str(), "bias shape is not equal to input second dim."),
                        return NOT_CHANGED);
    }
  }
  // to add node bias as third input, nodeMatMul must have 2 InDataAnchor
  // and 2 InputDesc(referenced AddLinkFrom())
  if (matMulOpDesc->GetInputsSize() != MATMUL_INPUT_NUM) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "MatMul node should have 2 inputs, acutal %zu",
            nodeMatMul->GetInAllNodes().size());
    return NOT_CHANGED;
  }

  // check nodeMatMul must have only one output to nodeBiasAdd
  if (nodeMatMul->GetOutDataNodes().size() != 1) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "MatMul node should only have 1 output, actual %zu",
            nodeMatMul->GetOutDataNodes().size());
    return NOT_CHANGED;
  }

  // check biasAddOpDesc should only have one outputTensroDesc
  if (biasAddOpDesc->GetAllOutputsDesc().size() != 1) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "BiasAdd node should only have 1 output, actual %zu",
            biasAddOpDesc->GetAllOutputsDesc().size());
    return NOT_CHANGED;
  }

  // check Bias node should only have 1 output, because ge::graph haven't offer
  // method to modify node anchor, only way to add anchor is AddLinkFrom
  if (nodeBias->GetAllOutDataAnchors().size() != 1) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "now don't support fusion Bias with over 1 output");
    return NOT_CHANGED;
  }

  // add HAS_BIAS attr to MatMul, and set value with "true"
  if (ge::AttrUtils::SetBool(matMulOpDesc, HAS_BIAS, true) == false) {
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "set attr:has_bias=true to matmul failed");
    return FAILED;
  }

  // add link from nodeBias to nodeMatMul,x3 is the name of third input of
  // MatMul in IR matmul.h
  if (nodeMatMul->AddLinkFrom("bias", nodeBias) != ge::GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add link from Bias to MatMul failed");
    return FAILED;
  }

  vector<bool> isInputConst;
  for (auto &anchor : nodeMatMul->GetAllInDataAnchors()) {
    auto peerAnchor = anchor->GetPeerOutAnchor();
    if (peerAnchor == nullptr) {
      continue;
    }
    auto node = peerAnchor->GetOwnerNode();
    string nodeType = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(node);
    if (nodeType == CONSTANT || nodeType == CONSTANTOP) {
      isInputConst.push_back(true);
    } else {
      isInputConst.push_back(false);
    }
  }
  nodeMatMul->GetOpDesc()->SetIsInputConst(isInputConst);
  // replace src (BiasAdd(0) -> OtherNode) to (MatMul -> OtherNode)
  auto matMulOutAnchor = nodeMatMul->GetOutDataAnchor(0);
  if (matMulOutAnchor == nullptr) {
    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[matMulOutAnchor] must not be null.");
    return fe::PARAM_INVALID;
  }
  auto biasAddOutAnchor0 = nodeBiasAdd->GetOutDataAnchor(0);
  if (biasAddOutAnchor0 == nullptr) {
    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[biasAddOutAnchor0] must not be null.");
    return fe::PARAM_INVALID;
  }
  for (auto &dstAnchor : biasAddOutAnchor0->GetPeerInDataAnchors()) {
    if (dstAnchor == nullptr) {
      CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[dstAnchor] must not be null.");
      return fe::PARAM_INVALID;
    }
    if (ge::GraphUtils::RemoveEdge(biasAddOutAnchor0, dstAnchor) != ge::GRAPH_SUCCESS ||
        ge::GraphUtils::AddEdge(matMulOutAnchor, dstAnchor) != ge::GRAPH_SUCCESS) {
      CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Replace edge src Failed.");
      return FAILED;
    }
  }

  // delete BiasAdd node
  if (graph.RemoveNode(nodeBiasAdd) != ge::GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "delete BiasAdd failed");
    return FAILED;
  }
  fusionNodes.push_back(nodeMatMul);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "matmul biasadd fusion success!");
  return SUCCESS;
}
REGISTER_PASS("MatMulBiasAddFusionPass", BUILT_IN_GRAPH_PASS, MatMulBiasAddFusionPass);
}  // namespace fe
