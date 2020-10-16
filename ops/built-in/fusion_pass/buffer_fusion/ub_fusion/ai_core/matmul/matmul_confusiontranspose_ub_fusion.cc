/**
 * @file matmul_confusiontranspose_ub_fusion.cpp
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief tbe matmul + confusiontransposed ops fusion pattern
 *
 * @version 1.0
 *
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
vector<BufferFusionPattern *> MatmulConfusiontransposeUbFusion::DefinePatterns() {

  vector<BufferFusionPattern *> patterns;
  string passName = "MatmulConfusiontransposeUbFusion";
  BufferFusionPattern *pattern =
      new (std::nothrow) BufferFusionPattern(passName);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."),
           return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName.c_str());
  pattern
      ->AddOpDesc(PATTERN_MATMUL, {OP_PATTERN_MATMUL}, TBE_PATTERN_NUM_DEFAULT,
                  TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_CONFUSION_TRANSPOSE, {OP_PATTERN_CONFUSION_TRANSPOSE},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
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
Status
MatmulConfusiontransposeUbFusion::GetFusionNodes(const BufferFusionMapping &mapping,
                                       vector<ge::NodePtr> &fusionNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to do MatmulConfusiontransposeUbFusion!");
  fusionNodes = GetMatchedNodes(mapping);
  // multi input node can not be fused except head node
  for (auto &item : mapping) {
    auto opdesc = find(item.first->types.begin(), item.first->types.end(),
                       TBE_PATTERN_OUTPUT_NODE);
    if (opdesc != item.first->types.end()) {
      for (auto &node : item.second) {
        auto nodePtr = find(fusionNodes.begin(), fusionNodes.end(), node);
        fusionNodes.erase(nodePtr);
      }
    }
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do MatmulConfusiontransposeUbFusion!");
  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("MatmulConfusiontransposeUbFusion",
                            BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            MatmulConfusiontransposeUbFusion);
} // namespace fe
