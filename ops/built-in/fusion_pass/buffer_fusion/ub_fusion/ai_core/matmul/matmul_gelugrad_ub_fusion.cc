/**
 * @file matmul_gelugrad_ub_fusion.cpp
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief tbe matmul + elemwise ops fusion pattern
 *
 * @version 1.0
 *
 */

#include "matmul_gelugrad_ub_fusion.h"

#include <string>
#include "pattern_fusion_util.h"
#include "op_log.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {

static const char PATTERN_MATMUL[] = "matmul";
static const char PATTERN_ELTWISE[] = "eltwise";
static const char PATTERN_OTHER_INPUT1[] = "otherInput1";
static const char PATTERN_OTHER_INPUT2[] = "otherInput2";
static const char PATTERN_OUTPUT1[] = "OUTPUT1";
static const char PATTERN_OUTPUT2[] = "OUTPUT2";

/*
 * @brief:  define matmul op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    matmul --> elemwise
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern *> MatmulGelugradUbFusion::DefinePatterns() {

  vector<BufferFusionPattern *> patterns;
  string passName = "MatmulGelugradUbFusion";
  BufferFusionPattern *pattern =
      new (std::nothrow) BufferFusionPattern(passName);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."),
           return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName.c_str());
  pattern
      ->AddOpDesc(PATTERN_MATMUL, {OP_PATTERN_MATMUL}, TBE_PATTERN_NUM_DEFAULT,
                  TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELTWISE, {OP_PATTERN_ELEMWISE},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OUTPUT1, {TBE_PATTERN_OUTPUT_NODE},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OUTPUT2, {TBE_PATTERN_OUTPUT_NODE},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT1, {TBE_PATTERN_INPUT_NODE},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT2, {TBE_PATTERN_INPUT_NODE},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_MATMUL})
      .SetOutputs(PATTERN_MATMUL,
                  {PATTERN_ELTWISE, PATTERN_OUTPUT1, PATTERN_OUTPUT2},
                  TBE_OUTPUT_BRANCH_MULTI)
      .SetOutputs(PATTERN_OTHER_INPUT1, {PATTERN_ELTWISE})
      .SetOutputs(PATTERN_OTHER_INPUT2, {PATTERN_ELTWISE});
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName.c_str());

  string passName1 = "TbeConvEltwiseReluQuantFusion1";
  BufferFusionPattern *pattern1 = new (std::nothrow) BufferFusionPattern(passName1);
  FUSION_PASS_CHECK((pattern1 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."),
           return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName1.c_str());
  // define pattern rules Convolution-->Eltwise-->Leakyrelu-->AcendQuant
  pattern1
      ->AddOpDesc(PATTERN_MATMUL, {OP_PATTERN_MATMUL}, TBE_PATTERN_NUM_DEFAULT,
                  TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELTWISE, {OP_PATTERN_ELEMWISE},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OUTPUT1, {TBE_PATTERN_OUTPUT_NODE},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT1, {TBE_PATTERN_INPUT_NODE},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT2, {TBE_PATTERN_INPUT_NODE},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_MATMUL})
      .SetOutputs(PATTERN_MATMUL, {PATTERN_ELTWISE, PATTERN_OUTPUT1},
                  TBE_OUTPUT_BRANCH_MULTI)
      .SetOutputs(PATTERN_OTHER_INPUT1, {PATTERN_ELTWISE})
      .SetOutputs(PATTERN_OTHER_INPUT2, {PATTERN_ELTWISE});
  patterns.push_back(pattern1);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName1.c_str());

  string passName2 = "TbeConvEltwiseReluQuantFusion2";
  BufferFusionPattern *pattern2 = new (std::nothrow) BufferFusionPattern(passName2);
  FUSION_PASS_CHECK((pattern2 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."),
           return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName2.c_str());
  // define pattern rules Convolution-->Eltwise-->Leakyrelu-->AcendQuant
  pattern2
      ->AddOpDesc(PATTERN_MATMUL, {OP_PATTERN_MATMUL}, TBE_PATTERN_NUM_DEFAULT,
                  TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELTWISE, {OP_PATTERN_ELEMWISE},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT1, {TBE_PATTERN_INPUT_NODE},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT2, {TBE_PATTERN_INPUT_NODE},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_MATMUL})
      .SetOutputs(PATTERN_MATMUL, {PATTERN_ELTWISE})
      .SetOutputs(PATTERN_OTHER_INPUT1, {PATTERN_ELTWISE})
      .SetOutputs(PATTERN_OTHER_INPUT2, {PATTERN_ELTWISE});
  patterns.push_back(pattern2);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName2.c_str());

  string passName3 = "TbeMatmulAddFusionPass";
  BufferFusionPattern *pattern3 = new (std::nothrow) BufferFusionPattern(passName3);
  FUSION_PASS_CHECK((pattern3 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."),
           return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName3.c_str());
  // define pattern rules Convolution-->Eltwise-->Leakyrelu-->AcendQuant
  pattern3
      ->AddOpDesc(PATTERN_MATMUL, {OP_PATTERN_MATMUL}, TBE_PATTERN_NUM_DEFAULT,
                  TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELTWISE, {OP_PATTERN_ELEMWISE},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT1, {TBE_PATTERN_INPUT_NODE},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_MATMUL})
      .SetOutputs(PATTERN_MATMUL, {PATTERN_ELTWISE})
      .SetOutputs(PATTERN_OTHER_INPUT1, {PATTERN_ELTWISE});
  patterns.push_back(pattern3);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName3.c_str());
  return patterns;
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status
MatmulGelugradUbFusion::GetFusionNodes(const BufferFusionMapping &mapping,
                                       vector<ge::NodePtr> &fusionNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to do MatmulGelugradUbFusion!");
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
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do MatmulGelugradUbFusion!");
  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("MatmulGelugradUbFusion",
                            BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            MatmulGelugradUbFusion);
} // namespace fe
