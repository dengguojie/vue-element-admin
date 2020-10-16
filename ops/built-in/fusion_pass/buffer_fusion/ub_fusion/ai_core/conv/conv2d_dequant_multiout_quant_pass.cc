/**
 * @file conv2d_dequant_multiout_quant_pass.cpp
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief tbe conv2d + dequant + quant ops fusion pattern
 *
 * @version 1.0
 *
 */

#include "conv2d_dequant_multiout_quant_pass.h"

#include <string>
#include "pattern_fusion_util.h"
#include "op_log.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {

static const char PATTERN_CONV[] = "conv2d";
static const char PATTERN_DEQUANT[] = "dequant";
static const char PATTERN_QUANT[] = "quant";
static const char PATTERN_OTHER_INPUT[] = "otherInput";
static const char PATTERN_OUTPUT1[] = "OUTPUT1";
/*
 * @brief:  define conv2d op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    Convolution-->AcendDeQuant(out)-->AscendQuant(out)
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern *> TbeConv2DDequantMultiOutQuantPass::DefinePatterns() {

    vector<BufferFusionPattern *> patterns;
    string passName = "TbeConv2DDequantMultiOutQuantFusion";
    BufferFusionPattern *pattern =
        new (std::nothrow) BufferFusionPattern(passName);
    FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."),
             return patterns);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName.c_str());
    pattern->AddOpDesc(PATTERN_CONV, {OP_PATTERN_CONV},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .AddOpDesc(PATTERN_DEQUANT, {OP_PATTERN_DEQUANT},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .AddOpDesc(PATTERN_QUANT, {OP_PATTERN_QUANT},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .AddOpDesc(PATTERN_OUTPUT1, {TBE_PATTERN_OUTPUT_NODE},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .SetHead({PATTERN_CONV})
            .SetOutputs(PATTERN_CONV, {PATTERN_DEQUANT})
            .SetOutputs(PATTERN_DEQUANT, {PATTERN_QUANT,
                        PATTERN_OUTPUT1}, TBE_OUTPUT_BRANCH_MULTI)
            .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_DEQUANT});
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
Status TbeConv2DDequantMultiOutQuantPass::GetFusionNodes(
    const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusionNodes) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to do TbeConv2DDequantMultiOutQuantFusion!");
    fusionNodes = GetMatchedNodes(mapping);
    // the outputData can't be fused
    for (auto &item : mapping) {
        auto opdesc = find(item.first->types.begin(),
                           item.first->types.end(),
                           TBE_PATTERN_OUTPUT_NODE);

        if (opdesc != item.first->types.end()) {
            for (auto &node : item.second) {
                auto nodePtr = find(fusionNodes.begin(), fusionNodes.end(),
                                    node);
                if (nodePtr != fusionNodes.end()) {
                    fusionNodes.erase(nodePtr);
                }
            }
        }
    }
    for (auto &item : mapping) {
        // judge AscendDequant node
        if (item.first->desc_name == PATTERN_DEQUANT) {
            for (auto &node : item.second) {
                if (node->GetType() == "AscendDequant") {
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "ascend_dequant is vaild, "
                            "support ub fusion.");
                } else {
                    fusionNodes.clear();
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "op type is not vaild, only support "
                            "AscendDequant.");
                    return SUCCESS;
                }
            }
        }
        // judge AscendQuant node
        if (item.first->desc_name == PATTERN_QUANT) {
            for (auto &node : item.second) {
                if (node->GetType() == "AscendQuant") {
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "ascend_quant is vaild, support ub fusion.");
                } else {
                    fusionNodes.clear();
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "we only support op type: AscendQuant, "
                            "not support %s.",
                            node->GetType().c_str());
                    return SUCCESS;
                }
            }
        }
    }
    OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do TbeConv2DDequantMultiOutQuantFusion!");

    return SUCCESS;
}
REGISTER_BUFFER_FUSION_PASS("TbeConv2DDequantMultiOutQuantPass",
                            BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            TbeConv2DDequantMultiOutQuantPass);
}  // namespace fe
