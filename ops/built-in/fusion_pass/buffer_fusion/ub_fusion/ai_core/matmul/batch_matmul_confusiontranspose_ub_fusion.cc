/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file batch_matmul_confusiontranspose_ub_fusion.cpp
 * \brief tbe batch_matmul + confusiontransposed ops fusion pattern
 */
#include "batch_matmul_confusiontranspose_ub_fusion.h"

#include <string>
#include "pattern_fusion_util.h"
#include "op_log.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "anchor_util.h"

namespace fe {

static const char PATTERN_BATCHMATMUL[] = "batch_matmul";
static const char PATTERN_BATCH_MATMUL_CONFUSION_TRANSPOSE[] = "batchmatmul_transpose";
/*
 * @brief:  define batchmatmul op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    batchmatmul --> confusiontransposed
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern*> BatchMatmulConfusiontransposeUbFusion::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;
  string passName = "BatchMatmulConfusiontransposeUbFusion";
  BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(passName);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName.c_str());
  pattern->AddOpDesc(PATTERN_BATCHMATMUL, {OP_PATTERN_BATCH_MATMUL}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_BATCH_MATMUL_CONFUSION_TRANSPOSE, {OP_PATTERN_CONFUSION_TRANSPOSE}, TBE_PATTERN_NUM_DEFAULT,
                 TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_BATCHMATMUL})
      .SetOutputs(PATTERN_BATCHMATMUL, {PATTERN_BATCH_MATMUL_CONFUSION_TRANSPOSE});
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName.c_str());

  return patterns;
}

/*
 * @brief: check input parameters before DoFusion
 * @param [in] matmulNodes: nodes matched pattern of matmul
 * @param [int] transposeNodes: nodes matched pattern of transpose
 * @return bool: check status ok or not.
 */
Status BatchMatmulConfusiontransposeUbFusion::CheckInputParameters(const vector<ge::NodePtr>& matmulNodes,
                                                                   const vector<ge::NodePtr>& transposeNodes) {
  for (auto matmulNode : matmulNodes) {
    for (auto matmulControlNode : matmulNode->GetOutControlNodes()) {
      FUSION_PASS_CHECK(matmulControlNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "out control of matmul is null"),
                        return FAILED);
      if (matmulControlNode->GetType() != "ConfusionTransposeD" && matmulControlNode->GetType() != "DropOutDoMaskV3D" &&
          matmulControlNode->GetType() != "Add") {
        continue;
      }
      FUSION_PASS_CHECK(
          ge::GraphUtils::RemoveEdge(matmulNode->GetOutControlAnchor(), matmulControlNode->GetInControlAnchor()) !=
              SUCCESS,
          OP_LOGD(FUSED_OP_TYPE.c_str(), "remove edge between batch_matmul and confusion_transpose_d error"),
          return FAILED);
      for (auto transposeOutNode : matmulControlNode->GetOutAllNodes()) {
        FUSION_PASS_CHECK(transposeOutNode == nullptr,
                          OP_LOGD(FUSED_OP_TYPE.c_str(), "output node of transpose is null"), return FAILED);
        FUSION_PASS_CHECK(
            ge::GraphUtils::AddEdge(matmulNode->GetOutControlAnchor(), transposeOutNode->GetInControlAnchor()) !=
                SUCCESS,
            OP_LOGD(FUSED_OP_TYPE.c_str(), "add edge between batch_matmul and confusion_transpose_d's output error"),
            return FAILED);
      }
    }
  }

  for (auto tranposeNode : transposeNodes) {
    for (auto transposeControlNode : tranposeNode->GetInControlNodes()) {
      FUSION_PASS_CHECK(transposeControlNode == nullptr,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "in control of transpose is null"), return FAILED);
      if (transposeControlNode->GetType() != "BatchMatMul") {
        continue;
      }
      FUSION_PASS_CHECK(
          ge::GraphUtils::RemoveEdge(transposeControlNode->GetOutControlAnchor(), tranposeNode->GetInControlAnchor()) !=
              SUCCESS,
          OP_LOGD(FUSED_OP_TYPE.c_str(), "remove edge between batch_matmul and confusion_transpose_d error"),
          return FAILED);
      for (auto transposeOutNode : tranposeNode->GetOutAllNodes()) {
        FUSION_PASS_CHECK(transposeOutNode == nullptr,
                          OP_LOGD(FUSED_OP_TYPE.c_str(), "output node of transpose is null"), return FAILED);
        FUSION_PASS_CHECK(
            ge::GraphUtils::AddEdge(transposeControlNode->GetOutControlAnchor(),
                                    transposeOutNode->GetInControlAnchor()) != SUCCESS,
            OP_LOGD(FUSED_OP_TYPE.c_str(), "add edge between batch_matmul and confusion_transpose_d's output error"),
            return FAILED);
      }
    }
  }

  return SUCCESS;
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] mapping: nodes matched by pattern
 * @param [out] fusionNodes: nodes to be fusioned
 * @return bool: fusion status ok or not.
 */
Status BatchMatmulConfusiontransposeUbFusion::GetFusionNodes(const BufferFusionMapping& mapping,
                                                             vector<ge::NodePtr>& fusionNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to do BatchMatmulConfusiontransposeUbFusion!");
  vector<ge::NodePtr> matmulNodes = GetMatchedNodesByDescName(PATTERN_BATCHMATMUL, mapping);
  vector<ge::NodePtr> transposeNodes = GetMatchedNodesByDescName(PATTERN_BATCH_MATMUL_CONFUSION_TRANSPOSE, mapping);

  FUSION_PASS_CHECK(CheckInputParameters(matmulNodes, transposeNodes) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "check parameter failed"), return FAILED);

  fusionNodes = GetMatchedNodes(mapping);

  // buffer fusion do not support dynamic shape now
  for (const auto& matmulNode : matmulNodes){
    auto input0desc = GetCurrNodeInputDesc(matmulNode, 0);
    auto input1desc = GetCurrNodeInputDesc(matmulNode, 1);
    FUSION_PASS_CHECK(input0desc == nullptr,
                  CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputDesc0 is null"),
                  return FAILED);
    FUSION_PASS_CHECK(input1desc == nullptr,
                  CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputDesc1 is null"),
                  return FAILED);
    vector<int64_t> input0Dims = input0desc->GetOriginShape().GetDims();
    vector<int64_t> input1Dims = input1desc->GetOriginShape().GetDims();
    vector<int64_t> allDims;
    allDims.resize(input0Dims.size() + input1Dims.size());
    merge(input0Dims.begin(), input0Dims.end(), input1Dims.begin(), input1Dims.end(), allDims.begin());
    for (auto singleDim : allDims) {
      if (singleDim < 0) {
        fusionNodes.clear();
        OP_LOGW(FUSED_OP_TYPE.c_str(), "ub fusion not support dynamic shape");
        return SUCCESS;
      }
    }
  }

  // multi input node can not be fused except head node
  for (auto& item : mapping) {
    auto opdesc = find(item.first->types.begin(), item.first->types.end(), TBE_PATTERN_OUTPUT_NODE);
    if (opdesc != item.first->types.end()) {
      for (auto& node : item.second) {
        auto nodePtr = find(fusionNodes.begin(), fusionNodes.end(), node);
        if (nodePtr != fusionNodes.end()) {
          fusionNodes.erase(nodePtr);
        }
      }
    }
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do BatchMatmulConfusiontransposeUbFusion!");
  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("BatchMatmulConfusiontransposeUbFusion", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            BatchMatmulConfusiontransposeUbFusion);
}  // namespace fe
