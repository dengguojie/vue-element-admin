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
 * \file matmul_confusiontranspose_ub_fusion.cpp
 * \brief tbe matmul + confusiontransposed ops fusion pattern
 */
#include "matmul_confusiontranspose_ub_fusion.h"

#include <string>
#include "pattern_fusion_util.h"
#include "op_log.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {

static const char PATTERN_MATMUL[] = "matmul";
static const char PATTERN_CONFUSION_TRANSPOSE[] = "matmul_transpose";
/*
 * @brief:  define matmul op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    matmul --> confusiontransposed
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern*> MatmulConfusiontransposeUbFusion::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;
  string passName = "MatmulConfusiontransposeUbFusion";
  BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(passName);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName.c_str());
  pattern->AddOpDesc(PATTERN_MATMUL, {OP_PATTERN_MATMUL}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_CONFUSION_TRANSPOSE, {OP_PATTERN_CONFUSION_TRANSPOSE}, TBE_PATTERN_NUM_DEFAULT,
                 TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_MATMUL})
      .SetOutputs(PATTERN_MATMUL, {PATTERN_CONFUSION_TRANSPOSE});
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName.c_str());

  return patterns;
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status MatmulConfusiontransposeUbFusion::GetFusionNodes(const BufferFusionMapping& mapping,
                                                        vector<ge::NodePtr>& fusionNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to do MatmulConfusiontransposeUbFusion!");
  vector<ge::NodePtr> matmulNodes = GetMatchedNodesByDescName(PATTERN_MATMUL, mapping);

  for (auto matmulNode : matmulNodes) {
    for (auto matmulControlNode : matmulNode->GetOutControlNodes()) {
      FUSION_PASS_CHECK(matmulControlNode == nullptr, OP_LOGD(FUSED_OP_TYPE.c_str(), "out control of matmul is null"),
                        return FAILED);
      if (matmulControlNode->GetType() != "ConfusionTransposeD") {
        continue;
      }
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(matmulNode->GetOutControlAnchor(),
                                                   matmulControlNode->GetInControlAnchor()) != SUCCESS,
                        OP_LOGD(FUSED_OP_TYPE.c_str(), "remove edge between matmul and confusion_transpose_d error"),
                        return FAILED);
      for (auto transposeOutNode : matmulControlNode->GetOutAllNodes()) {
        FUSION_PASS_CHECK(transposeOutNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "output of transpose is null"),
                          return FAILED);
        FUSION_PASS_CHECK(
            ge::GraphUtils::AddEdge(matmulNode->GetOutControlAnchor(), transposeOutNode->GetInControlAnchor()) !=
                SUCCESS,
            OP_LOGD(FUSED_OP_TYPE.c_str(), "add edge between matmul and confusion_transpose_d's output error"),
            return FAILED);
      }
    }
  }
  fusionNodes = GetMatchedNodes(mapping);

  // buffer fusion do not support dynamic shape now
  for (const auto& matmulNode : matmulNodes){
    vector<int64_t> input0Dims = matmulNode->GetOpDesc()->GetInputDesc(0).GetOriginShape().GetDims();
    vector<int64_t> input1Dims = matmulNode->GetOpDesc()->GetInputDesc(1).GetOriginShape().GetDims();
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
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do MatmulConfusiontransposeUbFusion!");
  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("MatmulConfusiontransposeUbFusion", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            MatmulConfusiontransposeUbFusion);
}  // namespace fe
