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
 * \file attention_score_fusion_pass.cc
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
#include "attention_score_fusion_pass.h"
#include "common/util/platform_info.h"
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
static const std::string PATTERN_BATCHMATMUL = "batch_matmul";
static const std::string PATTERN_BATCHMATMUL2 = "batch_matmul2";
static const std::string PATTERN_BATCHMATMUL3 = "batch_matmul3";
static const std::string PATTERN_FUSEDMULADD = "fused_mul_add";
static const std::string PATTERN_SOFTMAXV2WITHDROPOUT = "softmax_v2_with_dropout";
static const std::string PATTERN_CONFUSIONTRANSPOSE = "confusion_transpose";
static const std::string PATTERN_SOFTMAXV2 = "softmax_v2";

static const std::string BATCHMATMULV2 = "BatchMatMulV2";
static const std::string FUSEDMULADD = "FusedMulAdd";
static const std::string SOFTMAXV2WITHDROPOUTDOMASKV3D = "SoftmaxV2WithDropOutDoMaskV3D";
static const std::string CONFUSIONTRANSPOSE = "ConfusionTransposeD";
static const std::string SOFTMAXV2 = "SoftmaxV2";
static const int ALIGN_UNIT = 16;
static const int ALIGN_UNIT_BASE = 12;
static const int CONFUSION_DIM_ONE = 12288;
static const int CONFUSION_DIM_TWO = 768;
static const int NUM_TWO = 2;
static const int NUM_THREE = 3;
static const int NUM_FOUR = 4;
static const int NUM_FIVE = 5;
static const int NUM_SIX = 6;
static bool isInferencePlateform = false;
static bool isTraningPlateform = false;
static const std::vector<std::string> SUPPORT_PLATFORM_PATTERN = {"Ascend310P", "Ascend910"};
static const std::string kNameFusionPass = "ZAttentionScoreFusionPass";
}  // namespace

vector<FusionPattern *> ZAttentionScoreFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern = new (std::nothrow) FusionPattern(kNameFusionPass);
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGW(kNameFusionPass, "Failed to create pattern."),
                    return patterns);
  OP_LOGD(kNameFusionPass, "Start to define pattern.");
  pattern->AddOpDesc(PATTERN_BATCHMATMUL, {BATCHMATMULV2})
    .AddOpDesc(PATTERN_FUSEDMULADD, {FUSEDMULADD})
    .AddOpDesc(PATTERN_SOFTMAXV2WITHDROPOUT, {SOFTMAXV2WITHDROPOUTDOMASKV3D})
    .AddOpDesc(PATTERN_BATCHMATMUL2, {BATCHMATMULV2})
    .AddOpDesc(PATTERN_CONFUSIONTRANSPOSE, {CONFUSIONTRANSPOSE})
    .SetInputs(PATTERN_FUSEDMULADD, {PATTERN_BATCHMATMUL})
    .SetInputs(PATTERN_SOFTMAXV2WITHDROPOUT, {PATTERN_FUSEDMULADD})
    .SetInputs(PATTERN_BATCHMATMUL2, {PATTERN_SOFTMAXV2WITHDROPOUT})
    .SetInputs(PATTERN_CONFUSIONTRANSPOSE, {PATTERN_BATCHMATMUL2})
    .SetOutput(PATTERN_CONFUSIONTRANSPOSE);
  patterns.push_back(pattern);
  OP_LOGD(kNameFusionPass, "End to define pattern.");

  FusionPattern *pattern1 = new (std::nothrow) FusionPattern(kNameFusionPass);
  FUSION_PASS_CHECK(pattern1 == nullptr, OP_LOGW(kNameFusionPass, "Failed to create pattern1."),
                    return patterns);
  OP_LOGD(kNameFusionPass, "Start to define pattern1.");
  pattern1->AddOpDesc(PATTERN_BATCHMATMUL, {BATCHMATMULV2})
    .AddOpDesc(PATTERN_FUSEDMULADD, {FUSEDMULADD})
    .AddOpDesc(PATTERN_SOFTMAXV2, {SOFTMAXV2})
    .AddOpDesc(PATTERN_BATCHMATMUL2, {BATCHMATMULV2})
    .AddOpDesc(PATTERN_CONFUSIONTRANSPOSE, {CONFUSIONTRANSPOSE})
    .SetInputs(PATTERN_FUSEDMULADD, {PATTERN_BATCHMATMUL})
    .SetInputs(PATTERN_SOFTMAXV2, {PATTERN_FUSEDMULADD})
    .SetInputs(PATTERN_BATCHMATMUL2, {PATTERN_SOFTMAXV2})
    .SetInputs(PATTERN_CONFUSIONTRANSPOSE, {PATTERN_BATCHMATMUL2})
    .SetOutput(PATTERN_CONFUSIONTRANSPOSE);
  patterns.push_back(pattern1);
  OP_LOGD(kNameFusionPass, "End to define pattern1.");

  return patterns;
}

Status ZAttentionScoreFusionPass::CheckPlatformInfo() {
  OP_LOGD(kNameFusionPass, "CheckPlatformInfo begin");
  PlatformInfo platformInfo;
  OptionalInfo optionalInfo;
  FUSION_PASS_CHECK(
      PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != fe::SUCCESS,
      OP_LOGD(kNameFusionPass, "Failed to get platform info"), return NOT_CHANGED);

  std::string socVersion = optionalInfo.soc_version;
  OP_LOGD(kNameFusionPass, "Get soc version: %s", socVersion.c_str());

  bool isSupport = false;
  isInferencePlateform = IsTargetPlateform(SUPPORT_PLATFORM_PATTERN[0]);
  isTraningPlateform = IsTargetPlateform(SUPPORT_PLATFORM_PATTERN[1]);
  if (isInferencePlateform || isTraningPlateform) {
    isSupport = true;
  }
  FUSION_PASS_CHECK(!isSupport, OP_LOGD(kNameFusionPass, "Only support 310p, 910 series platform"),
                    return NOT_CHANGED);

  OP_LOGD(kNameFusionPass, "CheckPlatformInfo end");
  return SUCCESS;
}

bool ZAttentionScoreFusionPass::IsTargetPlateform(const std::string plateform) {
  OP_LOGD(kNameFusionPass, "IsTargetPlateform begin");
  PlatformInfo platformInfo;
  OptionalInfo optionalInfo;
  FUSION_PASS_CHECK(
      PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != fe::SUCCESS,
      OP_LOGD(kNameFusionPass, "Failed to get platform info"), return NOT_CHANGED);

  std::string socVersion = optionalInfo.soc_version;
  OP_LOGD(kNameFusionPass, "Get soc version: %s", socVersion.c_str());

  bool isTarget = false;
  if (socVersion == plateform || socVersion.find(plateform) != string::npos) {
    isTarget = true;
  }

  OP_LOGD(kNameFusionPass, "IsTargetPlateform end");
  return isTarget;
}

Status ZAttentionScoreFusionPass::SetAttrForBsbDesc(std::shared_ptr<ge::OpDesc> bsbDesc) {
  bool firstTransposeA = false;
  bool firstTransposeB = false;
  bool secondTransposeA = false;
  bool secondTransposeB = false;
  float keepProb = 0;
  vector<int64_t> axes = {};
  FUSION_PASS_CHECK(
    !ge::AttrUtils::GetBool(batchMatmulNode1->GetOpDesc(), "adj_x1", firstTransposeA),
    OP_LOGE(kNameFusionPass, "Failed to get adj_x1 of %s.", batchMatmulNode1->GetName().c_str()),
    return FAILED);
  FUSION_PASS_CHECK(
    !ge::AttrUtils::GetBool(batchMatmulNode1->GetOpDesc(), "adj_x2", firstTransposeB),
    OP_LOGE(kNameFusionPass, "Failed to get adj_x2 of %s.", batchMatmulNode1->GetName().c_str()),
    return FAILED);
  FUSION_PASS_CHECK(
    !ge::AttrUtils::GetBool(batchMatmulNode2->GetOpDesc(), "adj_x1", secondTransposeA),
    OP_LOGE(kNameFusionPass, "Failed to get adj_x1 of %s.", batchMatmulNode2->GetName().c_str()),
    return FAILED);
  FUSION_PASS_CHECK(
    !ge::AttrUtils::GetBool(batchMatmulNode2->GetOpDesc(), "adj_x2", secondTransposeB),
    OP_LOGE(kNameFusionPass, "Failed to get adj_x2 of %s.", batchMatmulNode2->GetName().c_str()),
    return FAILED);
  FUSION_PASS_CHECK(
    !ge::AttrUtils::GetListInt(softmaxNode->GetOpDesc(), "axes", axes),
    OP_LOGE(kNameFusionPass, "Failed to get axes of %s.", softmaxNode->GetName().c_str()),
    return FAILED);
  if (traning) {
      FUSION_PASS_CHECK(
        !ge::AttrUtils::GetFloat(softmaxNode->GetOpDesc(), "keep_prob", keepProb),
        OP_LOGE(kNameFusionPass, "Failed to get keep_prod of %s.", softmaxNode->GetName().c_str()),
        return FAILED);
  }
  ge::AttrUtils::SetFloat(bsbDesc, "keep_prob", keepProb);
  ge::AttrUtils::SetBool(bsbDesc, "query_transpose", firstTransposeA);
  ge::AttrUtils::SetBool(bsbDesc, "key_transpose", firstTransposeB);
  ge::AttrUtils::SetBool(bsbDesc, "bmm_score_transpose_a", secondTransposeA);
  ge::AttrUtils::SetBool(bsbDesc, "bmm_score_transpose_b", secondTransposeB);
  ge::AttrUtils::SetListInt(bsbDesc, "softmax_axes", axes);

  return SUCCESS;
}

Status ZAttentionScoreFusionPass::DeleteFusionNode(ge::ComputeGraph &graph) {
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(batchMatmulNode1),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                           batchMatmulNode1->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(fusedMulAddNode),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                           fusedMulAddNode->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(batchMatmulNode2),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                           batchMatmulNode2->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(softmaxNode),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                           softmaxNode->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(confusionTransposeNode),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                           confusionTransposeNode->GetName().c_str()),
      return FAILED);

  return SUCCESS;
}

Status ZAttentionScoreFusionPass::AddControlEdgesForBsbNode(ge::NodePtr bsbNode) {
  // add control edge
  if (softmaxNode->GetInControlAnchor()) {
    for (auto outControlAnchor : softmaxNode->GetInControlAnchor()->GetPeerOutControlAnchors()) {
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(outControlAnchor, bsbNode->GetInControlAnchor()) != SUCCESS,
          CUBE_CALL_ERR_REPORT(kNameFusionPass, "Add new node input control edge failed."),
          return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(outControlAnchor, softmaxNode->GetInControlAnchor()) != SUCCESS,
                        CUBE_CALL_ERR_REPORT(kNameFusionPass,
                                             "Remove softmaxNode node input control edge failed."),
                        return FAILED);
    }
  }
  if (softmaxNode->GetOutControlAnchor()) {
    for (auto inControlAnchor : softmaxNode->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(softmaxNode->GetOutControlAnchor(), inControlAnchor) != SUCCESS,
                        CUBE_CALL_ERR_REPORT(kNameFusionPass, "Remove softmax node out control edge failed."),
                        return FAILED);
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(bsbNode->GetOutControlAnchor(), inControlAnchor) != SUCCESS,
          CUBE_CALL_ERR_REPORT(kNameFusionPass, "Add new node out control edge failed."), return FAILED);
    }
  }
  return SUCCESS;
}

Status ZAttentionScoreFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(kNameFusionPass, "Start ZAttentionScoreFusionPass.");

  FUSION_PASS_CHECK(CheckPlatformInfo() != SUCCESS, OP_LOGD(kNameFusionPass, "Failed to check platform info"),
                    return NOT_CHANGED);

  batchMatmulNode1 = GetNodeFromMapping(PATTERN_BATCHMATMUL, mapping);
  FUSION_PASS_CHECK(batchMatmulNode1 == nullptr,
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "Get batchMatmulNode1 not success."),
                    return PARAM_INVALID);

  ge::NodePtr outputBatchMatmulNode1 = GetNodeFromMapping(PATTERN_BATCHMATMUL2, mapping);
  FUSION_PASS_CHECK(outputBatchMatmulNode1 == nullptr,
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "Get batchMatmulNode2 not success."),
                    return PARAM_INVALID);

  fusedMulAddNode = GetNodeFromMapping(PATTERN_FUSEDMULADD, mapping);
  FUSION_PASS_CHECK(fusedMulAddNode == nullptr,
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "Get fusedMulAddNode not success."),
                    return PARAM_INVALID);

  confusionTransposeNode = GetNodeFromMapping(PATTERN_CONFUSIONTRANSPOSE, mapping);
  FUSION_PASS_CHECK(confusionTransposeNode == nullptr,
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "Get confusionTransposeNode not success."),
                    return PARAM_INVALID);

  vector<int64_t> confusionTransposeOutDims = confusionTransposeNode->GetOpDesc()->
                                                       MutableOutputDesc(0)->GetShape().GetDims();
  FUSION_PASS_CHECK(confusionTransposeOutDims.empty(),
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "Get confusionTransposeOutDims not success."),
                    return PARAM_INVALID);
  batchMatmulNode2 = outputBatchMatmulNode1;
  std::string batchMatmul2Name = batchMatmulNode2->GetName().c_str();
  std::string tarName = "gradient_tape";
  size_t found = batchMatmul2Name.find(tarName);
  if (found != std::string::npos) {
    OP_LOGD(kNameFusionPass, "Graph not change.");
    return NOT_CHANGED;
  }

  vector<int64_t> batchMatmulNode2Dims = batchMatmulNode2->GetOpDesc()->MutableInputDesc(1)->GetShape().GetDims();
  vector<int64_t> batchMatmulNode1Input2Dims = batchMatmulNode1->GetOpDesc()->
                                                      MutableInputDesc(1)->GetShape().GetDims();
  vector<int64_t> batchMatmulNode1Dims = batchMatmulNode1->GetOpDesc()->MutableInputDesc(0)->GetShape().GetDims();
  if (batchMatmulNode1Dims != batchMatmulNode1Input2Dims || batchMatmulNode1Dims != batchMatmulNode2Dims) {
    return NOT_CHANGED;
  }

  OP_LOGD(kNameFusionPass, "Check BatchMatMul node input dims.");
  softmaxNode = GetNodeFromMapping(PATTERN_SOFTMAXV2WITHDROPOUT, mapping);
  if (softmaxNode == nullptr) {
    traning = false;
    OP_LOGD(kNameFusionPass, "The traning flag is false.");
    softmaxNode = GetNodeFromMapping(PATTERN_SOFTMAXV2, mapping);
    FUSION_PASS_CHECK(softmaxNode == nullptr,
                      CUBE_CALL_ERR_REPORT(kNameFusionPass, "Get softmaxNode not success."),
                      return PARAM_INVALID);
  }

  if (!traning && isTraningPlateform) {
    OP_LOGD(kNameFusionPass, "Not supported 910 inference.");
    return NOT_CHANGED;
  }

  std::string tempOutName = "";
  std::string quantName = "Quant";
  OP_LOGD(kNameFusionPass, "Check Quant graph.");
  for (auto netoutInDataAnchor : confusionTransposeNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    ge::NodePtr tempOutNode = netoutInDataAnchor->GetOwnerNode();
    tempOutName = tempOutNode->GetName().c_str();
    size_t foundQuant = tempOutName.find(quantName);
    if (foundQuant != std::string::npos) {
      OP_LOGD(kNameFusionPass, "Graph not support quant.");
      return NOT_CHANGED;
    }
  }

  if (traning) {
    for (auto netoutInDataAnchor : softmaxNode->GetOutDataAnchor(1)->GetPeerInDataAnchors()) {
      ge::NodePtr tempBatchNode = netoutInDataAnchor->GetOwnerNode();
      if (tempBatchNode->GetName().c_str() != batchMatmulNode2->GetName().c_str()) {
        batchMatmulNode3 = tempBatchNode;
        break;
      }
    }
  }

  OP_LOGD(kNameFusionPass, "Check the training net.");
  vector<int64_t> softmaxShapeDims = softmaxNode->GetOpDesc()->MutableOutputDesc(0)->GetShape().GetDims();
  FUSION_PASS_CHECK(softmaxShapeDims.empty(),
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "Get softmax_dims not success."),
                    return PARAM_INVALID);

  if (softmaxNode->GetOpDesc()->MutableOutputDesc(0)->GetDataType() != ge::DT_FLOAT16) {
    OP_LOGD(kNameFusionPass, "DataType not match, graph not change.");
    return NOT_CHANGED;
  }

  auto dimSize = softmaxShapeDims.size();
  if (dimSize != NUM_FOUR && dimSize != NUM_SIX) {
    OP_LOGD(kNameFusionPass, "Shape not match, graph not change.");
    return NOT_CHANGED;
  }
  int64_t batchDim0 = softmaxShapeDims[0];
  int64_t batchDim1 = softmaxShapeDims[1];
  int64_t batchDim2 = softmaxShapeDims[NUM_TWO];
  int64_t batchDim3 = softmaxShapeDims[NUM_THREE];
  int64_t batchMaxValue = 128;
  int64_t batchDim3Target = 64;
  int64_t dimThree = 3;
  bool shapeNotMatched = false;
  if (batchDim0 > batchMaxValue) {
    OP_LOGD(kNameFusionPass, "BathSize is too large, graph not change.");
    return NOT_CHANGED;
  }
  if ((batchDim1 != ALIGN_UNIT && batchDim1 != ALIGN_UNIT_BASE) || batchDim2 != batchDim3 ||
    batchMatmulNode1Dims[dimThree] != batchDim3Target) {
    shapeNotMatched = true;
  }
  vector<int64_t> fusedMulAddNodeDims = fusedMulAddNode->GetOpDesc()->MutableInputDesc(NUM_TWO)
                                               ->GetShape().GetDims();
  vector<int64_t> addTargetDims = {batchDim0, 1, batchDim2, batchDim3};
  if (dimSize == NUM_SIX) {
    addTargetDims = {batchDim0, 1, batchDim2 * ALIGN_UNIT, batchDim3 * ALIGN_UNIT};
  }
  if (fusedMulAddNodeDims != addTargetDims) {
    shapeNotMatched = true;
  }

  int64_t seqValueFactor = 32;
  int64_t maxSeqValue = 512;
  int64_t miniBatchValue = 8;
  auto softmaxFormat = softmaxNode->GetOpDesc()->GetInputDesc(0).GetFormat();
  if ((dimSize == NUM_FOUR) && (batchDim2 % seqValueFactor != 0)) {
    shapeNotMatched = true;
  }
  OP_LOGD(kNameFusionPass, "The seq_value is %d.", batchDim2);
  if (dimSize == NUM_FOUR) {
    if (traning && batchDim0 * batchDim2 != CONFUSION_DIM_ONE) {
      shapeNotMatched = true;
    }
    if (((batchDim0 * batchDim1)% miniBatchValue != 0) || (batchDim2 > maxSeqValue)) {
      shapeNotMatched = true;
    }
  } else if (dimSize == NUM_SIX && softmaxFormat == ge::FORMAT_FRACTAL_NZ) {
    if (batchDim0 * batchDim3 != CONFUSION_DIM_TWO) {
      shapeNotMatched = true;
    }
  } else {
    shapeNotMatched = true;
  }

  if (shapeNotMatched) {
    OP_LOGD(kNameFusionPass, "Shape not match, graph not change.");
    return NOT_CHANGED;
  }
  std::shared_ptr<ge::OpDesc> bsbDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
    bsbDesc = std::make_shared<ge::OpDesc>(softmaxNode->GetName() + "/AttentionScore",
                                            "AttentionScore"),
    return FAILED);

  FUSION_PASS_CHECK(bsbDesc->AddInputDesc("query",
                                           *(batchMatmulNode1->GetOpDesc()->MutableInputDesc(0))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input0 of AttentionScore failed."),
                    return FAILED);

  FUSION_PASS_CHECK(bsbDesc->AddInputDesc("key",
                                           *(batchMatmulNode1->GetOpDesc()->MutableInputDesc(1))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input1 of AttentionScore failed."),
                    return FAILED);

  FUSION_PASS_CHECK(bsbDesc->AddInputDesc("value",
                                           *(batchMatmulNode2->GetOpDesc()->MutableInputDesc(1))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input2 of AttentionScore failed."),
                    return FAILED);

  FUSION_PASS_CHECK(bsbDesc->AddInputDesc("padding_mask",
                                           *(fusedMulAddNode->GetOpDesc()->MutableInputDesc(NUM_TWO))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input3 of AttentionScore failed."),
                    return FAILED);

  FUSION_PASS_CHECK(bsbDesc->AddInputDesc("scale",
                                           *(fusedMulAddNode->GetOpDesc()->MutableInputDesc(1))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input4 of AttentionScore failed."),
                    return FAILED);
  if (traning) {
    FUSION_PASS_CHECK(bsbDesc->AddInputDesc("drop_mask",
                                             *(softmaxNode->GetOpDesc()->MutableInputDesc(1))) != SUCCESS,
                      OP_LOGE(kNameFusionPass, "Add input5 of AttentionScore failed."),
                      return FAILED);
  } else {
    FUSION_PASS_CHECK(bsbDesc->AddInputDesc("drop_mask",
                                             *(fusedMulAddNode->GetOpDesc()->MutableInputDesc(NUM_TWO))) != SUCCESS,
                      OP_LOGE(kNameFusionPass, "Add input5 of AttentionScore failed."),
                      return FAILED);
    OP_LOGD(kNameFusionPass, "Add fused mul_add input to drop_mask.");
  }

  FUSION_PASS_CHECK(bsbDesc->AddOutputDesc("attention_score",
                                             *(confusionTransposeNode->GetOpDesc()->MutableOutputDesc(0))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add output0 of AttentionScore failed."),
                    return FAILED);
  if (traning) {
    FUSION_PASS_CHECK(bsbDesc->AddOutputDesc("softmax_output",
                                             *(softmaxNode->GetOpDesc()->MutableOutputDesc(0))) != SUCCESS,
                      OP_LOGE(kNameFusionPass, "Add output1 of AttentionScore failed."),
                      return FAILED);
  }
  OP_LOGD(kNameFusionPass, "Start to SetAttrForBsbDesc.");
  FUSION_PASS_CHECK(SUCCESS != SetAttrForBsbDesc(bsbDesc),
                    OP_LOGE(kNameFusionPass, "SetAttrForBsbDesc failed."),
                    return FAILED);
  OP_LOGD(kNameFusionPass, "End to SetAttrForBsbDesc.");
  ge::NodePtr bsbNode = graph.AddNode(bsbDesc);
  FUSION_PASS_CHECK(bsbNode == nullptr,
                    OP_LOGE(kNameFusionPass, "Add AttentionScore to graph failed."),
                    return FAILED);

  // add edge
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(batchMatmulNode1->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            bsbNode->GetInDataAnchor(0)) != SUCCESS,
                    OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(batchMatmulNode1->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            bsbNode->GetInDataAnchor(1)) != SUCCESS,
                    OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(batchMatmulNode2->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            bsbNode->GetInDataAnchor(NUM_TWO)) != SUCCESS,
                    OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(fusedMulAddNode->GetInDataAnchor(NUM_TWO)->GetPeerOutAnchor(),
                                            bsbNode->GetInDataAnchor(NUM_THREE)) != SUCCESS,
                    OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(fusedMulAddNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            bsbNode->GetInDataAnchor(NUM_FOUR)) != SUCCESS,
                    OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                    return FAILED);
  if (traning) {
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(softmaxNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                              bsbNode->GetInDataAnchor(NUM_FIVE)) != SUCCESS,
                      OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                      return FAILED);
  } else {
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(fusedMulAddNode->GetInDataAnchor(NUM_TWO)->GetPeerOutAnchor(),
                                              bsbNode->GetInDataAnchor(NUM_FIVE)) != SUCCESS,
                      OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                      return FAILED);
    OP_LOGD(kNameFusionPass, "Add edge mul_add input to drop_mask.");
  }
  if (traning) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(softmaxNode->GetOutDataAnchor(1),
                                                 batchMatmulNode3->GetInDataAnchor(0)) != SUCCESS,
                      OP_LOGW(kNameFusionPass, "Failed to remove edge softmax to bmm3."),
                      return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(bsbNode->GetOutDataAnchor(1),
                                              batchMatmulNode3->GetInDataAnchor(0)) != SUCCESS,
                      OP_LOGW(kNameFusionPass, "Failed to add edge bsbNode to bmm3."),
                      return FAILED);
  }
  OP_LOGD(kNameFusionPass, "Start AddOutputEdgeForNode.");
  AddOutputEdgeForNode(confusionTransposeNode, bsbNode, 0, 0);
  if (traning) {
    AddOutputEdgeForNode(softmaxNode, bsbNode, 0, 1);
  }
  OP_LOGD(kNameFusionPass, "Start to AddControlEdgesForBsbNode.");
  FUSION_PASS_CHECK(SUCCESS != AddControlEdgesForBsbNode(bsbNode),
                    OP_LOGE(kNameFusionPass, "Failed to AddControlEdgesForBsbNode."),
                    return FAILED);
  OP_LOGD(kNameFusionPass, "End to AddControlEdgesForBsbNode.");
  FUSION_PASS_CHECK(SUCCESS != DeleteFusionNode(graph),
                    OP_LOGE(kNameFusionPass, "Failed to DeleteFusionNode."),
                    return FAILED);
  OP_LOGD(kNameFusionPass, "End to DeleteFusionNode.");
  return SUCCESS;
}

Status ZAttentionScoreFusionPass::AddOutputEdgeForNode(ge::NodePtr oriNode, ge::NodePtr newNode,
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


REGISTER_PASS("ZAttentionScoreFusionPass",
              BUILT_IN_GRAPH_PASS,
              ZAttentionScoreFusionPass);
}  // namespace fe
