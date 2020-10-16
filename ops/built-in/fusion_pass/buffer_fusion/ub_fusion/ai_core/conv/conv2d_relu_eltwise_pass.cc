/**
 * @file conv2d_relu_eltwise_pass.cpp
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief tbe conv2d + relu + eltwise ops fusion pattern
 *
 * @version 1.0
 *
 */

#include "conv2d_relu_eltwise_pass.h"

#include <string>
#include "pattern_fusion_util.h"
#include "op_log.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {

static const char PATTERN_CONV[] = "conv2d";
static const char PATTERN_RELU[] = "relu";
static const char PATTERN_ELTWISE[] = "eltwise";
static const char PATTERN_OTHER_INPUT[] = "otherInput";
/*
 * @brief:  define conv2d op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    conv2d --> relu/leaklyrelu --> eltwise/addv2
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern *> TbeConv2DReluEltwisePass::DefinePatterns() {

    vector<BufferFusionPattern *> patterns;
    string passName = "TbeConv2DReluEltwiseFusion";
    BufferFusionPattern *pattern =
        new (std::nothrow) BufferFusionPattern(passName);
    FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."),
             return patterns);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName.c_str());
    pattern->AddOpDesc(PATTERN_CONV, {OP_PATTERN_CONV},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .AddOpDesc(PATTERN_RELU, {OP_PATTERN_ELEMWISE},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .AddOpDesc(PATTERN_ELTWISE, {OP_PATTERN_ELEMWISE},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .SetHead({PATTERN_CONV})
            .SetOutputs(PATTERN_CONV, {PATTERN_RELU})
            .SetOutputs(PATTERN_RELU, {PATTERN_ELTWISE})
            .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_ELTWISE});
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
Status TbeConv2DReluEltwisePass::GetFusionNodes(
    const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusionNodes) {

    OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to do Conv2DReluEltwise!");
    fusionNodes= GetMatchedNodes(mapping);
    // the outputData can't be fused
    for (auto &item : mapping) {
        auto opdesc = find(item.first->types.begin(),
                           item.first->types.end(),
                           TBE_PATTERN_OUTPUT_NODE);

        if (opdesc != item.first->types.end()) {
            for (auto &node : item.second) {
                auto nodePtr = find(fusionNodes.begin(), fusionNodes.end(),
                                    node);
                fusionNodes.erase(nodePtr);
            }
        }
    }
    for (auto &item : mapping) {
        // judge LeakyRelu/Relu node
        if (item.first->desc_name == PATTERN_RELU) {
            for (auto &node : item.second) {
                if (node->GetType() == "Relu" ||
                    node->GetType() == "LeakyRelu") {
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "relu or leakly_relu is vaild, "
                            "support ub fusion.");
                } else {
                    fusionNodes.clear();
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "relu is not vaild, only support "
                            "Relu or LeakyRelu.");
                    return SUCCESS;
                }
            }
        }
        // judge Eltwise/Add node
        if (item.first->desc_name == PATTERN_ELTWISE) {
            for (auto &node : item.second) {
                if (node->GetType() == "Add" ||
                    node->GetType() == "Eltwise") {
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "eltwise or add is vaild, support ub fusion.");
                } else {
                    fusionNodes.clear();
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "we only support op type : Eltwise or Add, "
                            "not support %s.",
                            node->GetType().c_str());
                    return SUCCESS;
                }
            }
        }
    }
    OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do Conv2DReluEltwise!");

    return SUCCESS;
}
REGISTER_BUFFER_FUSION_PASS("TbeConv2DReluEltwisePass",
                            BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            TbeConv2DReluEltwisePass);
}  // namespace fe
