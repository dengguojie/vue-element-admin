/**
 * @file tbe_aipp_conv_fusion_pass.cpp
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief tbe aipp convolution ops fusion pattern
 *
 * @version 1.0
 *
 */

#include "tbe_aipp_conv_fusion_pass.h"
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include<math.h>
#include "common/util/platform_info.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "tbe_aipp_fusion_rule.h"

namespace fe {

static const char PATTERN_AIPP[] = "aipp";
static const string PATTERN_CONV = "convolution";
static const string PATTERN_OTHER_INPUT = "otherInput";
static const char PATTERN_OTHER_INPUT1[] = "otherInput1";

/*
 * @brief:  define aipp and convolution op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    1) Aipp-->Convolution
 *
 * fusion node: Aipp, Convolution
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern *> TbeAippConvFusionPass::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;
  string passName1 = "TbeAippConvFusion1";

  BufferFusionPattern *pattern1 = new (std::nothrow) BufferFusionPattern(passName1);
  FUSION_PASS_CHECK((pattern1 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."),
           return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName1.c_str());
  // define pattern rules
  pattern1
      ->AddOpDesc(PATTERN_AIPP, {OP_PATTERN_AIPP}, TBE_PATTERN_NUM_DEFAULT,
                  TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_CONV, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT,
                 TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_AIPP})
      .SetOutputs(PATTERN_AIPP, {PATTERN_CONV})
      .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_CONV});

  patterns.push_back(pattern1);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName1.c_str());

  string passName2 = "TbeAippConvFusion2";
  BufferFusionPattern *pattern2 = new (std::nothrow) BufferFusionPattern(passName2);
  FUSION_PASS_CHECK((pattern2 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."),
           return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName2.c_str());
  // define pattern rules
  pattern2
      ->AddOpDesc(PATTERN_AIPP, {OP_PATTERN_AIPP}, TBE_PATTERN_NUM_DEFAULT,
                  TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT1, {TBE_PATTERN_INPUT_NODE},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_CONV, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT,
                 TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_AIPP})
      .SetOutputs(PATTERN_AIPP, {PATTERN_CONV})
      .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_CONV})
      .SetOutputs(PATTERN_OTHER_INPUT1, {PATTERN_CONV});

  patterns.push_back(pattern2);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName2.c_str());
  return patterns;
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status TbeAippConvFusionPass::GetFusionNodes(
    const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusionNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to do TbeConvReluFusionPass!");
  vector<ge::NodePtr> convNodes = GetMatchedNodesByDescName(PATTERN_CONV, mapping);
  vector<ge::NodePtr> aippNodes = GetMatchedNodesByDescName(PATTERN_AIPP, mapping);

  string inputFormat = "";

  for (auto aippNode : aippNodes) {
    string aippConfigStr = "";
    FUSION_PASS_CHECK(!ge::AttrUtils::GetStr(aippNode->GetOpDesc(), "aipp_config_path",
                                    aippConfigStr),
             OP_LOGI(FUSED_OP_TYPE.c_str(), "Get node[%s]'s aipp_config_path attr not success.",
                     aippNode->GetName().c_str()), return FAILED);

    nlohmann::json aippConfigJson = nlohmann::json::parse(aippConfigStr);
    FUSION_PASS_CHECK(!aippConfigJson.is_object(),
             OP_LOGE(FUSED_OP_TYPE.c_str(), "The aippConfigStr is not an object, the aippConfigStr is %s.",
                     aippConfigStr.c_str()),
             return FAILED);
    inputFormat = aippConfigJson["input_format"];
    OP_LOGI(FUSED_OP_TYPE.c_str(), "aipp input_format is %s!", inputFormat.c_str());
  }

  for (auto convNode : convNodes) {
    if (!TbeAippFusionRule::CheckConvload2dNodeValidation(convNode)) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s] not satisfied with fusion condition.",
              convNode->GetName().c_str());
      return SUCCESS;
    }
    if (!TbeAippFusionRule::CheckAippConvEltwiseFusionValidation(convNode, inputFormat)){
      OP_LOGI(FUSED_OP_TYPE.c_str(), "The AIPP YUV exceed the L1 buffer, "
          "Node[%s] not satisfied with fusion condition.",
              convNode->GetName().c_str());
      return SUCCESS;
    }
  if (!TbeAippFusionRule::CheckAippConvStridehValidation(convNode)){
      OP_LOGI(FUSED_OP_TYPE.c_str(), "The case is the strideh optim. "
          "Node[%s] not satisfied with fusion condition.",
              convNode->GetName().c_str());
      return SUCCESS;
    }
  }

  fusionNodes = GetMatchedNodes(mapping);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do TbeAippConvFusionPass!");
  return SUCCESS;

}
REGISTER_BUFFER_FUSION_PASS("TbeAippConvZ", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            TbeAippConvFusionPass);
}  // namespace fe
