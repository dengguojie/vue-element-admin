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
 * \file attention_score_grad_fusion_pass.cc
 * pattern:
 *    x        const0                           const0
 *     \      /                                 /
 *   batch_matmul    const1          x       pad0      const1
 *              \    /                \      /         /
 *               add                 batch_matmul    pad1
 *                |                             \    /
 *               relu    const2  ->              add
 *                 \    /                         |
 *               reduce_mean                     relu    const2
 *                                                 \    /
 *                                               reduce_mean
 *                                                   |
 *                                                 slice
 */
#include <stdlib.h>
#include "attention_score_grad_fusion_pass.h"

#include "anchor_util.h"
#include "error_util.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "external/graph/operator_factory.h"

namespace fe {
namespace {
static const std::string PATTERN_INPUTCONFUSIONTRANSPOSE = "inputConfusionTranspose";
static const std::string PATTERN_BATCHMATMUL = "batch_matmul";
static const std::string PATTERN_BATCHMATMUL2 = "batch_matmul2";
static const std::string PATTERN_DROPOUTDOMASKV3 = "drop_out_do_mask_v3";
static const std::string PATTERN_SOFTMAXGRADEXT = "softmax_grad_ext";
static const std::string PATTERN_CONFUSIONTRANSPOSE1 = "confusion_transpose1";
static const std::string PATTERN_CONFUSIONTRANSPOSE2 = "confusion_transpose2";
static const std::string PATTERN_CONFUSIONTRANSPOSE3 = "confusion_transpose3";


static const std::string BATCHMATMULV2 = "BatchMatMulV2";
static const std::string DROPOUTDOMASKV3D = "DropOutDoMaskV3D";
static const std::string SOFTMAXGRADEXT = "SoftmaxGradExt";
static const std::string CONFUSIONTRANSPOSE = "ConfusionTransposeD";

static const int ALIGN_UNIT = 16;
static const int CONFUSION_DIM_ONE = 12288;
static const int CONFUSION_DIM_TWO = 768;
static const int NUM_TWO = 2;
static const int NUM_THREE = 3;
static const int NUM_FOUR = 4;
static const int NUM_FIVE = 5;
static const int NUM_SIX = 6;

static const std::string kNameFusionPass = "ZAttentionScoreGradFusionPass";
}  // namespace

vector<FusionPattern *> ZAttentionScoreGradFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern = new (std::nothrow) FusionPattern(kNameFusionPass);
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGW(kNameFusionPass, "Failed to create pattern."),
                    return patterns);
  OP_LOGD(kNameFusionPass, "Start to define pattern.");
  pattern->AddOpDesc(PATTERN_INPUTCONFUSIONTRANSPOSE, {CONFUSIONTRANSPOSE})
    .AddOpDesc(PATTERN_BATCHMATMUL, {BATCHMATMULV2})
    .AddOpDesc(PATTERN_BATCHMATMUL2, {BATCHMATMULV2})
    .AddOpDesc(PATTERN_DROPOUTDOMASKV3, {DROPOUTDOMASKV3D})
    .AddOpDesc(PATTERN_SOFTMAXGRADEXT, {SOFTMAXGRADEXT})
    .AddOpDesc(PATTERN_CONFUSIONTRANSPOSE1, {CONFUSIONTRANSPOSE})
    .SetInputs(PATTERN_BATCHMATMUL, {PATTERN_INPUTCONFUSIONTRANSPOSE})
    .SetInputs(PATTERN_DROPOUTDOMASKV3, {PATTERN_BATCHMATMUL})
    .SetInputs(PATTERN_SOFTMAXGRADEXT, {PATTERN_DROPOUTDOMASKV3})
    .SetInputs(PATTERN_BATCHMATMUL2, {PATTERN_SOFTMAXGRADEXT})
    .SetInputs(PATTERN_CONFUSIONTRANSPOSE1, {PATTERN_BATCHMATMUL2})
    .SetOutput(PATTERN_CONFUSIONTRANSPOSE1);
  patterns.push_back(pattern);
  OP_LOGD(kNameFusionPass, "End to define pattern.");

  return patterns;
}

Status ZAttentionScoreGradFusionPass::AddInputDescForBsb(std::shared_ptr<ge::OpDesc> bsbDesc) {
  FUSION_PASS_CHECK(bsbDesc->AddInputDesc("attention_score",
                                          *(batchMatmulNode4->GetOpDesc()->MutableInputDesc(0))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input0 of AttentionScore failed."),
                    return FAILED);

  FUSION_PASS_CHECK(bsbDesc->AddInputDesc("dx",
                                          *(batchMatmulNode4->GetOpDesc()->MutableInputDesc(1))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input1 of AttentionScore failed."),
                    return FAILED);

  FUSION_PASS_CHECK(bsbDesc->AddInputDesc("query",
                                          *(batchMatmulNode1->GetOpDesc()->MutableInputDesc(1))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input2 of AttentionScore failed."),
                    return FAILED);

  FUSION_PASS_CHECK(bsbDesc->AddInputDesc("key",
                                          *(batchMatmulNode2->GetOpDesc()->MutableInputDesc(1))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input3 of AttentionScore failed."),
                    return FAILED);

  FUSION_PASS_CHECK(bsbDesc->AddInputDesc("value",
                                          *(batchMatmulNode3->GetOpDesc()->MutableInputDesc(1))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input4 of AttentionScore failed."),
                    return FAILED);

  FUSION_PASS_CHECK(bsbDesc->AddInputDesc("scale",
                                           *(softmaxGradExtNode->GetOpDesc()->MutableInputDesc(NUM_TWO))) != SUCCESS,
                      OP_LOGE(kNameFusionPass, "Add input5 of AttentionScore failed."),
                      return FAILED);
  FUSION_PASS_CHECK(bsbDesc->AddInputDesc("drop_mask",
                                           *(dropOutDoMaskV3Node->GetOpDesc()->MutableInputDesc(1))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input6 of AttentionScore failed."),
                    return FAILED);
  return SUCCESS;
}

Status ZAttentionScoreGradFusionPass::AddOutputDescForBsb(std::shared_ptr<ge::OpDesc> bsbDesc) {
  FUSION_PASS_CHECK(bsbDesc->AddOutputDesc("value_dw",
                                            *(confusionTransposeNode2->GetOpDesc()->MutableOutputDesc(0))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add output0 of AttentionScore failed."),
                    return FAILED);
  FUSION_PASS_CHECK(bsbDesc->AddOutputDesc("query_dx",
                                            *(confusionTransposeNode->GetOpDesc()->MutableOutputDesc(0))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add output0 of AttentionScore failed."),
                    return FAILED);
  FUSION_PASS_CHECK(bsbDesc->AddOutputDesc("key_dw",
                                            *(confusionTransposeNode1->GetOpDesc()->MutableOutputDesc(0))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add output0 of AttentionScore failed."),
                    return FAILED);
  return SUCCESS;
}

Status ZAttentionScoreGradFusionPass::DeleteOldNode(ge::ComputeGraph &graph) {
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(batchMatmulNode1),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                     batchMatmulNode1->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(batchMatmulNode2),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                     batchMatmulNode2->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(batchMatmulNode3),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                     batchMatmulNode3->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(batchMatmulNode4),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                     batchMatmulNode4->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(confusionTransposeNode),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                     confusionTransposeNode->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(confusionTransposeNode1),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                     confusionTransposeNode1->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(confusionTransposeNode2),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                     confusionTransposeNode2->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(softmaxGradExtNode),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                     softmaxGradExtNode->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(dropOutDoMaskV3Node),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                     dropOutDoMaskV3Node->GetName().c_str()),
      return FAILED);
  return SUCCESS;
}

Status ZAttentionScoreGradFusionPass::AddControlEdgesForBsbNode(ge::NodePtr bsbNode) {
  // add control edge
  if (softmaxGradExtNode->GetInControlAnchor()) {
    for (auto outControlAnchor : softmaxGradExtNode->GetInControlAnchor()->GetPeerOutControlAnchors()) {
      if (dropOutDoMaskV3Node->GetName().c_str() == outControlAnchor->GetOwnerNode()->GetName().c_str()) {
        continue;
      } else {
        FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(outControlAnchor, bsbNode->GetInControlAnchor()) != SUCCESS,
                            CUBE_CALL_ERR_REPORT(kNameFusionPass, "Add new node input control edge failed."),
                            return FAILED);
          FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(outControlAnchor,
                                                       softmaxGradExtNode->GetInControlAnchor()) != SUCCESS,
                            CUBE_CALL_ERR_REPORT(kNameFusionPass,
                                                           "Remove softmax_node node input control edge failed."),
                            return FAILED);
      }
    }
  }
  if (softmaxGradExtNode->GetOutControlAnchor()) {
    for (auto inControlAnchor : softmaxGradExtNode->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(softmaxGradExtNode->GetOutControlAnchor(),
                                                   inControlAnchor) != SUCCESS,
                        CUBE_CALL_ERR_REPORT(kNameFusionPass, "Remove softmax node out control edge failed."),
                        return FAILED);
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(bsbNode->GetOutControlAnchor(), inControlAnchor) != SUCCESS,
          CUBE_CALL_ERR_REPORT(kNameFusionPass, "Add new node out control edge failed."), return FAILED);
    }
  }

  if (dropOutDoMaskV3Node->GetInControlAnchor()) {
    for (auto outControlAnchor : dropOutDoMaskV3Node->GetInControlAnchor()->GetPeerOutControlAnchors()) {
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(outControlAnchor, bsbNode->GetInControlAnchor()) != SUCCESS,
          CUBE_CALL_ERR_REPORT(kNameFusionPass, "Add new node input control edge failed."),
          return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(outControlAnchor,
                                                   dropOutDoMaskV3Node->GetInControlAnchor()) != SUCCESS,
                        CUBE_CALL_ERR_REPORT(kNameFusionPass,
                                             "Remove drop_out node input control edge failed."),
                        return FAILED);
    }
  }
  if (dropOutDoMaskV3Node->GetOutControlAnchor()) {
    for (auto inControlAnchor : dropOutDoMaskV3Node->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(dropOutDoMaskV3Node->GetOutControlAnchor(),
                                                   inControlAnchor) != SUCCESS,
                        CUBE_CALL_ERR_REPORT(kNameFusionPass, "Remove drop_out node out control edge failed."),
                        return FAILED);
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(bsbNode->GetOutControlAnchor(), inControlAnchor) != SUCCESS,
          CUBE_CALL_ERR_REPORT(kNameFusionPass, "Add new node out control edge failed."), return FAILED);
    }
  }
  return SUCCESS;
}

Status ZAttentionScoreGradFusionPass::AddEdgesForBsbNode(ge::NodePtr bsbNode) {
 // add edge
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(softmaxGradExtNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            bsbNode->GetInDataAnchor(0)) != SUCCESS,
                    OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(batchMatmulNode4->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            bsbNode->GetInDataAnchor(1)) != SUCCESS,
                    OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(batchMatmulNode1->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            bsbNode->GetInDataAnchor(NUM_TWO)) != SUCCESS,
                    OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(batchMatmulNode2->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            bsbNode->GetInDataAnchor(NUM_THREE)) != SUCCESS,
                    OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(batchMatmulNode3->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            bsbNode->GetInDataAnchor(NUM_FOUR)) != SUCCESS,
                    OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(softmaxGradExtNode->GetInDataAnchor(NUM_TWO)->GetPeerOutAnchor(),
                                            bsbNode->GetInDataAnchor(NUM_FIVE)) != SUCCESS,
                      OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                      return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(dropOutDoMaskV3Node->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            bsbNode->GetInDataAnchor(NUM_SIX)) != SUCCESS,
                      OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                      return FAILED);

  AddOutputEdgeForNode(confusionTransposeNode2, bsbNode, 0, 0);

  AddOutputEdgeForNode(confusionTransposeNode, bsbNode, 0, 1);

  AddOutputEdgeForNode(confusionTransposeNode1, bsbNode, 0, NUM_TWO);

  return SUCCESS;
}

Status ZAttentionScoreGradFusionPass::SetAttrsForBsb(std::shared_ptr<ge::OpDesc> bsbDesc) {
  float keepProb = 0;
  FUSION_PASS_CHECK(
    !ge::AttrUtils::GetFloat(dropOutDoMaskV3Node->GetOpDesc(), "keep_prob", keepProb),
    OP_LOGE(kNameFusionPass, "Failed to get keep_prod of %s.", dropOutDoMaskV3Node->GetName().c_str()),
    return FAILED);

  ge::AttrUtils::SetFloat(bsbDesc, "keep_prob", keepProb);

  return SUCCESS;
}

bool ZAttentionScoreGradFusionPass::CheckNodeShapeSupported() {
  return (CheckSoftmaxGradShapeSupported() || CheckBatchMatMulShapeSupported(batchMatmulNode1) ||
          CheckBatchMatMulShapeSupported(batchMatmulNode2) || CheckBatchMatMulShapeSupported(batchMatmulNode3) ||
          CheckSpcBatchMatMulShapeSupported(batchMatmulNode4));
}

bool ZAttentionScoreGradFusionPass::CheckSpcBatchMatMulShapeSupported(const ge::NodePtr bmmNode) {
  OP_LOGD(kNameFusionPass, "Start check shape of %s.", bmmNode->GetName().c_str());
  vector<int64_t> bmmInput1Dims = bmmNode->GetOpDesc()->MutableInputDesc(1)->GetShape().GetDims();
  FUSION_PASS_CHECK(bmmInput1Dims.empty(),
                    OP_LOGD(kNameFusionPass, "Get bmm input1 shape not success."),
                    return false);
  vector<int64_t> bmmInput0Dims = bmmNode->GetOpDesc()->MutableInputDesc(0)->GetShape().GetDims();
  FUSION_PASS_CHECK(bmmInput0Dims.empty(),
                    OP_LOGD(kNameFusionPass, "Get bmm input0 shape not success."),
                    return false);
  vector<int64_t> bmm1Dims = batchMatmulNode1->GetOpDesc()->MutableInputDesc(1)->GetShape().GetDims();
  vector<int64_t> softmaxShapeDims = softmaxGradExtNode->GetOpDesc()->MutableOutputDesc(0)->GetShape().GetDims();
  if ((bmmInput1Dims != bmm1Dims) || (bmmInput0Dims != softmaxShapeDims)) {
    return true;
  }
  return false;
}

bool ZAttentionScoreGradFusionPass::CheckBatchMatMulShapeSupported(const ge::NodePtr bmmNode) {
  OP_LOGD(kNameFusionPass, "Start check shape of %s.", bmmNode->GetName().c_str());
  vector<int64_t> bmmDims = bmmNode->GetOpDesc()->MutableInputDesc(1)->GetShape().GetDims();
  FUSION_PASS_CHECK(bmmDims.empty(),
                    OP_LOGD(kNameFusionPass, "Get bmm shape not success."),
                    return false);
  int64_t bmmDim2 = bmmDims[NUM_TWO];
  int64_t bmmDim3 = bmmDims[NUM_THREE];
  int64_t batchDim3Limit = 64;
  bool shapeNotMatched = false;
  int64_t dimSize = bmmDims.size();
  auto bmmFormat = bmmNode->GetOpDesc()->GetInputDesc(1).GetFormat();
  if (dimSize == NUM_FOUR && bmmDim3 == batchDim3Limit) {
    shapeNotMatched = false;
  } else if ((dimSize == NUM_SIX) && (bmmDim2 == NUM_FOUR) && (bmmFormat == ge::FORMAT_FRACTAL_NZ)) {
    shapeNotMatched = false;
  } else {
    shapeNotMatched = true;
  }

  return shapeNotMatched;
}

bool ZAttentionScoreGradFusionPass::CheckSoftmaxGradShapeSupported() {
  OP_LOGD(kNameFusionPass, "Start check shape of %s.", softmaxGradExtNode->GetName().c_str());
  vector<int64_t> softmaxShapeDims = softmaxGradExtNode->GetOpDesc()->MutableOutputDesc(0)->GetShape().GetDims();
  FUSION_PASS_CHECK(softmaxShapeDims.empty(),
                    OP_LOGD(kNameFusionPass, "Get softmax_dims not success."),
                    return NOT_CHANGED);
  int64_t batchDim0 = softmaxShapeDims[0];
  int64_t batchDim1 = softmaxShapeDims[1];
  int64_t batchDim2 = softmaxShapeDims[NUM_TWO];
  int64_t batchDim3 = softmaxShapeDims[NUM_THREE];
  int64_t dimSize = softmaxShapeDims.size();
  int64_t seqDimLimitFactor = 32;
  int64_t seqDimLimitFactorNz = 2;
  bool shapeNotMatched = false;
  int64_t batchMaxValue = 128;
  if (batchDim0 > batchMaxValue) {
    OP_LOGD(kNameFusionPass, "BathSize is too large.");
    shapeNotMatched = true;
  }
  if (dimSize != NUM_FOUR && dimSize != NUM_SIX) {
    OP_LOGD(kNameFusionPass, "Shape not match, graph not change.");
    shapeNotMatched = true;
  }

  if (batchDim1 != ALIGN_UNIT || batchDim2 != batchDim3) {
    shapeNotMatched = true;
  }
  auto softmaxFormat = softmaxGradExtNode->GetOpDesc()->GetInputDesc(0).GetFormat();
  if (dimSize == NUM_FOUR) {
    if (batchDim2 % seqDimLimitFactor != 0) {
      shapeNotMatched = true;
    }
    if (batchDim0 * batchDim2 != CONFUSION_DIM_ONE) {
      shapeNotMatched = true;
    }
  } else if (dimSize == NUM_SIX && softmaxFormat == ge::FORMAT_FRACTAL_NZ) {
    if (batchDim2 % seqDimLimitFactorNz != 0) {
      shapeNotMatched = true;
    }
    if (batchDim0 * batchDim3 != CONFUSION_DIM_TWO) {
      shapeNotMatched = true;
    }
  } else {
    shapeNotMatched = true;
  }

  return shapeNotMatched;
}

Status ZAttentionScoreGradFusionPass::CreateAttentionDesc() {
  FUSION_PASS_MAKE_SHARED(
    bsbDesc = std::make_shared<ge::OpDesc>(softmaxGradExtNode->GetName() + "/AttentionScoreGrad",
                                            "AttentionScoreGrad"),
    return FAILED);

  FUSION_PASS_CHECK(AddInputDescForBsb(bsbDesc) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input desc for bsb failed."),
                    return FAILED);
  FUSION_PASS_CHECK(AddOutputDescForBsb(bsbDesc) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add output desc for bsb failed."),
                    return FAILED);
  FUSION_PASS_CHECK(SetAttrsForBsb(bsbDesc) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Set Attr for bsb failed."),
                    return FAILED);
  return SUCCESS;
}

Status ZAttentionScoreGradFusionPass::GetFusionNode(Mapping &mapping, ge::NodePtr inputConfusionTranspose) {
  batchMatmulNode1 = GetNodeFromMapping(PATTERN_BATCHMATMUL, mapping);
  FUSION_PASS_CHECK(batchMatmulNode1 == nullptr,
                    OP_LOGD(kNameFusionPass, "The graph has been fusion."),
                    return NOT_CHANGED);

  FUSION_PASS_CHECK(batchMatmulNode1->GetInNodes().size() == 0,
                    OP_LOGD(kNameFusionPass, "Get batchMatmulNode1 not success."),
                    return NOT_CHANGED);

  for (auto netoutInDataAnchor : inputConfusionTranspose->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    ge::NodePtr tempBatchNode = netoutInDataAnchor->GetOwnerNode();
    if (tempBatchNode->GetName().c_str() != batchMatmulNode1->GetName().c_str()) {
      batchMatmulNode4 = tempBatchNode;
      break;
    }
  }
  FUSION_PASS_CHECK(batchMatmulNode4 == nullptr,
                    OP_LOGD(kNameFusionPass, "The batchMatmulNode4 is nullptr."),
                    return NOT_CHANGED);
  for (auto netoutInDataAnchor : batchMatmulNode4->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    ge::NodePtr tempBatchNode = netoutInDataAnchor->GetOwnerNode();
    confusionTransposeNode2 = tempBatchNode;
  }
  FUSION_PASS_CHECK(confusionTransposeNode2 == nullptr,
                    OP_LOGD(kNameFusionPass, "The confusionTransposeNode2 is nullptr."),
                    return NOT_CHANGED);
  dropOutDoMaskV3Node = GetNodeFromMapping(PATTERN_DROPOUTDOMASKV3, mapping);
  FUSION_PASS_CHECK(dropOutDoMaskV3Node == nullptr,
                    OP_LOGD(kNameFusionPass, "Get dropOutDoMaskV3Node not success."),
                    return NOT_CHANGED);

  softmaxGradExtNode = GetNodeFromMapping(PATTERN_SOFTMAXGRADEXT, mapping);
  FUSION_PASS_CHECK(softmaxGradExtNode == nullptr,
                    OP_LOGD(kNameFusionPass, "Get softmaxGradExtNode not success."),
                    return NOT_CHANGED);

  batchMatmulNode2 = GetNodeFromMapping(PATTERN_BATCHMATMUL2, mapping);
  FUSION_PASS_CHECK(batchMatmulNode2 == nullptr,
                    OP_LOGD(kNameFusionPass, "Get batchMatmulNode2 not success."),
                    return NOT_CHANGED);

  confusionTransposeNode = GetNodeFromMapping(PATTERN_CONFUSIONTRANSPOSE1, mapping);
  FUSION_PASS_CHECK(confusionTransposeNode == nullptr,
                    OP_LOGD(kNameFusionPass, "Get confusionTransposeNode not success."),
                    return NOT_CHANGED);

  for (auto netoutInDataAnchor : softmaxGradExtNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    ge::NodePtr tempBatchNode = netoutInDataAnchor->GetOwnerNode();
    if (tempBatchNode->GetName().c_str() != batchMatmulNode2->GetName().c_str()) {
      batchMatmulNode3 = tempBatchNode;
      break;
    }
  }
  FUSION_PASS_CHECK(batchMatmulNode3 == nullptr,
                    OP_LOGD(kNameFusionPass, "The batchMatmulNode3 is nullptr."),
                    return NOT_CHANGED);
  bool bmm2TransposeA = false;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetBool(batchMatmulNode2->GetOpDesc(), "adj_x1", bmm2TransposeA),
                    OP_LOGE(kNameFusionPass, "Failed to get adj_x1 of %s.", batchMatmulNode2->GetName().c_str()),
                    return NOT_CHANGED);
  for (auto netoutInDataAnchor : batchMatmulNode3->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    ge::NodePtr tempBatchNode = netoutInDataAnchor->GetOwnerNode();
    confusionTransposeNode1 = tempBatchNode;
  }
  FUSION_PASS_CHECK(confusionTransposeNode1 == nullptr,
                    OP_LOGD(kNameFusionPass, "The confusionTransposeNode1 is nullptr."),
                    return NOT_CHANGED);
  if (bmm2TransposeA) {
    ge::NodePtr tempBmmNode = batchMatmulNode2;
    batchMatmulNode2 = batchMatmulNode3;
    batchMatmulNode3 = tempBmmNode;

    tempBmmNode = confusionTransposeNode1;
    confusionTransposeNode1 = confusionTransposeNode;
    confusionTransposeNode = tempBmmNode;
  }
  return SUCCESS;
}

Status ZAttentionScoreGradFusionPass::Fusion(ge::ComputeGraph &graph,
                                             Mapping &mapping, vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(kNameFusionPass, "Start ZAttentionScoreGradFusionPass.");
  ge::NodePtr inputConfusionTranspose = GetNodeFromMapping(PATTERN_INPUTCONFUSIONTRANSPOSE, mapping);
  FUSION_PASS_CHECK(inputConfusionTranspose == nullptr,
                    OP_LOGD(kNameFusionPass, "Get inputConfusionTranspose not success."),
                    return NOT_CHANGED);

  /*
  * Get fused node in mapping assign global var.
  */
  OP_LOGD(kNameFusionPass, "Start GetFusionNode.");
  FUSION_PASS_CHECK(SUCCESS != GetFusionNode(mapping, inputConfusionTranspose),
                    OP_LOGD(kNameFusionPass, "Get fused node not success."),
                    return NOT_CHANGED);
  /*
  * Check node Shape is supported.
  */
  OP_LOGD(kNameFusionPass, "Start CheckNodeShapeSupported.");
  if (CheckNodeShapeSupported()) {
    OP_LOGD(kNameFusionPass, "Shape not match, graph not change.");
    return NOT_CHANGED;
  }

  /*
  * Create new Desc AttentionScore.
  */
  OP_LOGD(kNameFusionPass, "Start CreateAttentionDesc.");
  FUSION_PASS_CHECK(CreateAttentionDesc() != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input desc for bsb failed."),
                    return FAILED);
  /*
  * Create new Node AttentionScore.
  */
  OP_LOGD(kNameFusionPass, "Start Create new Node AttentionScore.");
  ge::NodePtr bsbNode = graph.AddNode(bsbDesc);
  FUSION_PASS_CHECK(bsbNode == nullptr,
                    OP_LOGE(kNameFusionPass, "Add AttentionScoreGrad to graph failed."),
                    return FAILED);
  /*
  * Add Edges for AttentionScore.
  */
  OP_LOGD(kNameFusionPass, "Start AddEdgesForBsbNode.");
  FUSION_PASS_CHECK(AddEdgesForBsbNode(bsbNode) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add Edges For BsbNode failed."),
                    return FAILED);
  /*
  * Add Control Edges for AttentionScore.
  */
  OP_LOGD(kNameFusionPass, "Start AddControlEdgesForBsbNode.");
  FUSION_PASS_CHECK(AddControlEdgesForBsbNode(bsbNode) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add Edges For BsbNode failed."),
                    return FAILED);
  /*
  * Delete fused node in graph.
  */
  OP_LOGD(kNameFusionPass, "Start DeleteOldNode.");
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != DeleteOldNode(graph),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node failed"),
      return FAILED);

  return SUCCESS;
}

Status ZAttentionScoreGradFusionPass::AddOutputEdgeForNode(ge::NodePtr oriNode, ge::NodePtr newNode,
                                                           int unlinkIndex, int newNodeIndex) const {
  if (oriNode->GetOutDataAnchor(unlinkIndex)->GetPeerInDataAnchors().size() > 0) {
    for (InDataAnchorPtr inAnchorPtr : oriNode->GetOutDataAnchor(unlinkIndex)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(inAnchorPtr == nullptr,
                        OP_LOGE(kNameFusionPass, "delete Edges For old node failed."),
                        return FAILED);
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(newNode->GetOutDataAnchor(newNodeIndex), inAnchorPtr),
                        CUBE_CALL_ERR_REPORT(
                          kNameFusionPass,
                          "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                          newNode->GetName().c_str(), 0, newNode->GetName().c_str(), 0),
                        return FAILED);
    }
  }
  return SUCCESS;
}


REGISTER_PASS("ZAttentionScoreGradFusionPass",
              SECOND_ROUND_BUILT_IN_GRAPH_PASS,
              ZAttentionScoreGradFusionPass);
}  // namespace fe
