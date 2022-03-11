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
 * \file concatv2_mean_mul_gatherv2_fusion_pass.cc
 * \brief concatv2_mean_mul_gatherv2 fusion pass(enable VectorCore: Aiore --> Aicore + VectorCore)
 */
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../../../op_proto/util/error_util.h"
#include "common/util/error_manager/error_manager.h"
#include "common/util/platform_info.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"
#include "concatv2_mean_mul_gatherv2_fusion_pass.h"

namespace fe {
static const std::string PATTERN_CONCATV2 = "ConcatV2";
static const std::string PATTERN_REDUCEMEAN = "ReduceMean";
static const std::string PATTERN_MUL = "Mul";
static const std::string PATTERN_EXPANDDIMS = "ExpandDims";
static const std::string PATTERN_GATHERV2 = "GatherV2";

static const std::string OP_TYPE_CONCATV2 = "ConcatV2";
static const std::string OP_TYPE_CONCATV2D = "ConcatV2D";
static const std::string OP_TYPE_REDUCEMEAN = "ReduceMean";
static const std::string OP_TYPE_MUL = "Mul";
static const std::string OP_TYPE_EXPANDDIMS = "ExpandDims";
static const std::string OP_TYPE_GATHERV2 = "GatherV2";

static const std::string ATTR_OP_SPECIFIED_ENGINE_NAME = "_specified_engine_name";
static const std::string ATTR_OP_SPECIFIED_KERNEL_LIB_NAME = "_specified_kernel_lib_name";
static const std::string ATTR_NAME_STREAM_LABEL = "_stream_label";
static const std::string AIC_LABEL = "AIcoreEngine";
static const std::string AIV_LABEL = "VectorEngine";
static const std::string ASCEND_710 = "Ascend710";
static const std::string CONCATV2_N = "N";
static const int64_t MIN_N = 2;
static const int64_t MAX_N = 63;
static const int64_t NUM_2 = 2;

/**
 * @brief pattern define
 *
 * ExpandDims  GatherV2   ExpandDims  GatherV2 ... ExpandDims  GatherV2
 *       \       /              \       /                \       /
 *        \     /                \     /                  \     /
 *          Mul                    Mul         ...          Mul
 *           |                      |                        |
 *      ReduceMean             ReduceMean      ...      ReduceMean
 *            \                     |                      /
 *               \                  |                   /
 *                  \               |                /
 *                     \            |             /
 *                              ConcatV2
 */
vector<FusionPattern*> ConcatV2MeanMulGatherV2FusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ConcatV2MeanMulGatherV2Fusion");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to new a pattern object"),
                    return patterns);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "start to define pass pattern");
  pattern->AddOpDesc(PATTERN_CONCATV2, {OP_TYPE_CONCATV2, OP_TYPE_CONCATV2D})
      .AddOpDesc(PATTERN_REDUCEMEAN, {OP_TYPE_REDUCEMEAN})
      .AddOpDesc(PATTERN_MUL, {OP_TYPE_MUL})
      .AddOpDesc(PATTERN_EXPANDDIMS, {OP_TYPE_EXPANDDIMS})
      .AddOpDesc(PATTERN_GATHERV2, {OP_TYPE_GATHERV2})
      .SetInputs(PATTERN_CONCATV2, {PATTERN_REDUCEMEAN})
      .SetInputs(PATTERN_REDUCEMEAN, {PATTERN_MUL})
      .SetInputs(PATTERN_MUL, {PATTERN_EXPANDDIMS, PATTERN_GATHERV2})
      .SetOutput(PATTERN_CONCATV2);
  patterns.push_back(pattern);

  return patterns;
}

Status ConcatV2MeanMulGatherV2FusionPass::CheckPlatformSupported() {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckPlatformSupported begin");

  PlatformInfo platformInfo;
  OptionalInfo optionalInfo;
  FUSION_PASS_CHECK(
      PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS,
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Failed to get platform info"),
      return NOT_CHANGED);

  std::string socVersion = optionalInfo.soc_version;
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Get soc version: %s", socVersion.c_str());
  FUSION_PASS_CHECK(socVersion != ASCEND_710,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Only 710 is supported, it has vector engine."),
                    return NOT_CHANGED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckPlatformSupported end");
  return SUCCESS;
}

Status ConcatV2MeanMulGatherV2FusionPass::CheckIsMatch(ge::NodePtr concatV2Node, int64_t numN) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckIsMatch begin");
  FUSION_PASS_CHECK(HasUnKnowShape(concatV2Node),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "not support dynamic shape, ConcatV2"),
                    return NOT_CHANGED);

  int64_t inputNum = concatV2Node->GetInDataNodes().size();
  if (concatV2Node->GetType() == OP_TYPE_CONCATV2) {
    // last input node is concat_dim of ConcatV2
    FUSION_PASS_CHECK((inputNum - 1) != numN,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "ConcatV2 input num is %ld, fusion failed", inputNum),
                      return NOT_CHANGED);

    ge::NodePtr concatDimNodePtr = concatV2Node->GetInDataNodes().at(numN);
    FUSION_PASS_CHECK(concatDimNodePtr == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "the input concat_dim of ConcatV2 is null"),
                      return PARAM_INVALID);
    FUSION_PASS_CHECK(HasUnKnowShape(concatDimNodePtr),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "not support dynamic shape, concat_dim"),
                      return NOT_CHANGED);
  } else {
    // ConcatV2D, input num should be equal to attr N
    FUSION_PASS_CHECK(inputNum != numN,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "ConcatV2D input num is %ld, fusion failed", inputNum),
                      return NOT_CHANGED);
  }

  for (int64_t i = 0; i < numN; i++) {
    ge::NodePtr meanNodePtr = concatV2Node->GetInDataNodes().at(i);
    FUSION_PASS_CHECK(meanNodePtr == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "the input %ld of ConcatV2 is null", i),
                      return PARAM_INVALID);
    FUSION_PASS_CHECK(meanNodePtr->GetType() != OP_TYPE_REDUCEMEAN,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "the input %ld of ConcatV2 is not ReduceMean", i),
                      return NOT_CHANGED);
    FUSION_PASS_CHECK(HasUnKnowShape(meanNodePtr),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "the input %ld of ConcatV2 is dynamic shape, ReduceMean", i),
                      return NOT_CHANGED);

    // check whether the label is set already
    ge::OpDescPtr meanDesc = meanNodePtr->GetOpDesc();
    FUSION_PASS_CHECK(meanDesc == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "OpDesc is null, ReduceMean"),
                      return PARAM_INVALID);
    std::string specifiedEngineName;
    FUSION_PASS_CHECK(ge::AttrUtils::GetStr(meanDesc, ATTR_OP_SPECIFIED_ENGINE_NAME, specifiedEngineName) &&
                          specifiedEngineName == AIV_LABEL,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "the label of engine has been set"),
                      return NOT_CHANGED);

    // input node of Mean
    for (auto preNodePtr : meanNodePtr->GetInDataNodes()) {
      FUSION_PASS_CHECK(preNodePtr == nullptr,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "the input of Mean(%ld) is null", i),
                        return PARAM_INVALID);
      FUSION_PASS_CHECK(HasUnKnowShape(preNodePtr),
                        OP_LOGI(FUSED_OP_TYPE.c_str(), "the input of Mean(%ld) is dynamic shape", i),
                        return NOT_CHANGED);
    }
    ge::NodePtr mulNodePtr = meanNodePtr->GetInDataNodes().at(0);
    FUSION_PASS_CHECK(mulNodePtr->GetType() != OP_TYPE_MUL,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "the input node of Mean(%ld) is not Mul", i),
                      return NOT_CHANGED);

    // input node of Mul
    ge::NodePtr expandDimsNodePtr = mulNodePtr->GetInDataNodes().at(0);
    ge::NodePtr gatherV2NodePtr = mulNodePtr->GetInDataNodes().at(1);
    FUSION_PASS_CHECK(
        (expandDimsNodePtr == nullptr) || (gatherV2NodePtr == nullptr),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "the input node of Mul(%ld) is null", i),
        return PARAM_INVALID);
    FUSION_PASS_CHECK(
        (expandDimsNodePtr->GetType() != OP_TYPE_EXPANDDIMS) || (gatherV2NodePtr->GetType() != OP_TYPE_GATHERV2),
        OP_LOGI(FUSED_OP_TYPE.c_str(), "the input node of Mul(%ld) is not ExpandDims and GatherV2", i),
        return NOT_CHANGED);
    FUSION_PASS_CHECK(HasUnKnowShape(expandDimsNodePtr) || HasUnKnowShape(gatherV2NodePtr),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "the input node of Mul(%ld) is dynamic shape", i),
                      return NOT_CHANGED);
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckIsMatch end");
  return SUCCESS;
}

Status ConcatV2MeanMulGatherV2FusionPass::SetLabelInNode(ge::NodePtr nodePtr, bool isAic) {
  ge::OpDescPtr opDesc = nodePtr->GetOpDesc();
  FUSION_PASS_CHECK(
      opDesc == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "SetLabelInNode, node's OpDesc is null"),
      return PARAM_INVALID);

  if (isAic) {
    FUSION_PASS_CHECK(
        !ge::AttrUtils::SetStr(opDesc, ATTR_OP_SPECIFIED_ENGINE_NAME, AIC_LABEL),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to set attribute _specified_engine_name, aic"),
        return FAILED);
    FUSION_PASS_CHECK(
        !ge::AttrUtils::SetStr(opDesc, ATTR_OP_SPECIFIED_KERNEL_LIB_NAME, AIC_LABEL),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                       "failed to set attribute _specified_kernel_lib_name, aic"),
        return FAILED);
  } else {
    FUSION_PASS_CHECK(
        !ge::AttrUtils::SetStr(opDesc, ATTR_OP_SPECIFIED_ENGINE_NAME, AIV_LABEL),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to set attribute _specified_engine_name, aiv"),
        return FAILED);
    FUSION_PASS_CHECK(
        !ge::AttrUtils::SetStr(opDesc, ATTR_OP_SPECIFIED_KERNEL_LIB_NAME, AIV_LABEL),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                       "failed to set attribute _specified_kernel_lib_name, aiv"),
        return FAILED);
    FUSION_PASS_CHECK(
        !ge::AttrUtils::SetStr(opDesc, ATTR_NAME_STREAM_LABEL, AIV_LABEL),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to set attribute _stream_label, aiv"),
        return FAILED);
  }

  return SUCCESS;
}

Status ConcatV2MeanMulGatherV2FusionPass::SetEngineLabel(ge::NodePtr nodePtr, bool isAic,
                                                         int64_t start, int64_t end) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "SetEngineLabel begin");

  for (int64_t i = start; i < end; i++) {
    ge::NodePtr meanNodePtr = nodePtr->GetInDataNodes().at(i);
    FUSION_PASS_CHECK(SetLabelInNode(meanNodePtr, isAic) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to set engine label, Mean node"),
                      return FAILED);

    ge::NodePtr mulNodePtr = meanNodePtr->GetInDataNodes().at(0);
    FUSION_PASS_CHECK(SetLabelInNode(mulNodePtr, isAic) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to set engine label, Mul node"),
                      return FAILED);

    ge::NodePtr expandDimsNodePtr = mulNodePtr->GetInDataNodes().at(0);
    ge::NodePtr gatherV2NodePtr = mulNodePtr->GetInDataNodes().at(1);
    FUSION_PASS_CHECK(
        SetLabelInNode(expandDimsNodePtr, isAic) != SUCCESS,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to set engine label, ExpandDims node"),
        return FAILED);

    FUSION_PASS_CHECK(
          SetLabelInNode(gatherV2NodePtr, true) != SUCCESS,
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to set engine label, GatherV2 node"),
          return FAILED);
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "SetEngineLabel end");
  return SUCCESS;
}

/*!
 * @brief: parse nodes matched in mapping and call graph DoFusion
 * @param [in] graph: original graph
 * @param [in] mapping: matched pattern
 * @param [out] newNodes: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status ConcatV2MeanMulGatherV2FusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                                 vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter into AConcatV2MeanMulGatherV2FusionPass");
  FUSION_PASS_CHECK(CheckPlatformSupported() != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "failed to check platform info, fusion failed"),
                    return NOT_CHANGED);

  ge::NodePtr concatV2Node = GetNodeFromMapping(PATTERN_CONCATV2, mapping);
  FUSION_PASS_CHECK(concatV2Node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to new a pattern object"),
                    return PARAM_INVALID);

  ge::OpDescPtr concatV2Desc = concatV2Node->GetOpDesc();
  FUSION_PASS_CHECK(
      concatV2Desc == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ConcatV2's desc is null, fusion failed."),
      return PARAM_INVALID);

  int64_t numN;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetInt(concatV2Desc, CONCATV2_N, numN),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "failed to get N of ConcatV2"),
                    return NOT_CHANGED);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "ConcatV2 input number is %ld", numN);
  FUSION_PASS_CHECK((numN < MIN_N) || (numN > MAX_N),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "N is %ld, min N is %ld, max N is %ld", numN, MIN_N, MAX_N),
                    return NOT_CHANGED);

  // check whether dynamic shape exist
  FUSION_PASS_CHECK(CheckIsMatch(concatV2Node, numN) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "match check failed"),
                    return NOT_CHANGED);

  // divide N groups evenly, and then label them with aic and aiv labels respectively
  int64_t aivNum = numN / NUM_2;
  FUSION_PASS_CHECK(SetEngineLabel(concatV2Node, false, 0, aivNum) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to set vector engine label"),
                    return FAILED);
  FUSION_PASS_CHECK(SetEngineLabel(concatV2Node, true, aivNum, numN) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to set aicore engine label"),
                    return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "End to AConcatV2MeanMulGatherV2FusionPass");
  return SUCCESS;
}

REGISTER_PASS("AConcatV2MeanMulGatherV2FusionPass", BUILT_IN_GRAPH_PASS, ConcatV2MeanMulGatherV2FusionPass);
}  // namespace fe
