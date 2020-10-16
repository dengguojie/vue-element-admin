/**
 * @file conv2d_dequant_add_mul_quant_pass.cpp
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief tbe conv2d + add + mul + quant ops fusion pattern
 *
 * @version 1.0
 *
 */

#include "conv2d_dequant_add_mul_quant_pass.h"
#include <string>
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {

static const string PATTERN_CONV = "conv2d";
static const string PATTERN_DEQ = "dequant";
static const string PATTERN_ADD = "add";
static const string PATTERN_MUL = "mul";
static const string PATTERN_MUL1 = "mul1";
static const string PATTERN_ADD1 = "add1";
static const string PATTERN_QUANT = "quant";
static const string PATTERN_OTHER_INPUT = "otherInput";
static const string PATTERN_OTHER_INPUT1 = "otherInput1";
static const string PATTERN_OTHER_INPUT2 = "otherInput2";
static const string PATTERN_OTHER_INPUT3 = "otherInput3";
static const string PATTERN_OTHER_INPUT4 = "otherInput4";
static const string PATTERN_OUTPUT1 = "OUTPUT1";
static const string PATTERN_OUTPUT2 = "OUTPUT2";
static const int64_t MAX_FUSE_NODE = 7;

/*
 * @brief:  define conv2d op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    conv2d --> dequant --> add --> mul --> add --> quant
 *    conv2d --> dequant --> add --> quant
 *                            | --> other
 *                            \ --> other
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern *> TbeConv2DAddMulQuantPass::DefinePatterns() {

    vector<BufferFusionPattern *> patterns;
    string passName = "TbeConv2DAddMulAddQuantFusion";
    BufferFusionPattern *pattern =
        new (std::nothrow) BufferFusionPattern(passName, MAX_FUSE_NODE);
    FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "create new pattern failed."),
             return patterns);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName.c_str());
    pattern->AddOpDesc(PATTERN_CONV, {OP_PATTERN_CONV},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .AddOpDesc(PATTERN_DEQ, {OP_PATTERN_DEQUANT},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .AddOpDesc(PATTERN_ADD, {OP_PATTERN_ELEMWISE},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .AddOpDesc(PATTERN_MUL, {OP_PATTERN_ELEMWISE},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .AddOpDesc(PATTERN_ADD1, {OP_PATTERN_ELEMWISE},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .AddOpDesc(PATTERN_QUANT, {OP_PATTERN_QUANT},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .AddOpDesc(PATTERN_OTHER_INPUT1, {TBE_PATTERN_INPUT_NODE},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .AddOpDesc(PATTERN_OTHER_INPUT2, {TBE_PATTERN_INPUT_NODE},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .AddOpDesc(PATTERN_OTHER_INPUT3, {TBE_PATTERN_INPUT_NODE},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .SetHead({PATTERN_CONV})
            .SetOutputs(PATTERN_CONV, {PATTERN_DEQ})
            .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_DEQ})
            .SetOutputs(PATTERN_DEQ, {PATTERN_ADD})
            .SetOutputs(PATTERN_OTHER_INPUT1, {PATTERN_ADD})
            .SetOutputs(PATTERN_ADD, {PATTERN_MUL})
            .SetOutputs(PATTERN_OTHER_INPUT2, {PATTERN_MUL})
            .SetOutputs(PATTERN_MUL, {PATTERN_ADD1})
            .SetOutputs(PATTERN_OTHER_INPUT3, {PATTERN_ADD1})
            .SetOutputs(PATTERN_ADD1, {PATTERN_QUANT});
    patterns.push_back(pattern);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName.c_str());

    string passName1 = "TbeConv2DAddMutioutQuantFusion";
    BufferFusionPattern *pattern1 =
        new (std::nothrow) BufferFusionPattern(passName1);
    FUSION_PASS_CHECK((pattern1 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "create new pattern failed."),
             return patterns);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName1.c_str());
    pattern1->AddOpDesc(PATTERN_CONV, {OP_PATTERN_CONV},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .AddOpDesc(PATTERN_DEQ, {OP_PATTERN_DEQUANT},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .AddOpDesc(PATTERN_ADD, {OP_PATTERN_ELEMWISE},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .AddOpDesc(PATTERN_QUANT, {OP_PATTERN_QUANT},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .AddOpDesc(PATTERN_OTHER_INPUT1, {TBE_PATTERN_INPUT_NODE},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .AddOpDesc(PATTERN_OUTPUT1, {TBE_PATTERN_OUTPUT_NODE},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .AddOpDesc(PATTERN_OUTPUT2, {TBE_PATTERN_OUTPUT_NODE},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .SetHead({PATTERN_CONV})
            .SetOutputs(PATTERN_CONV, {PATTERN_DEQ})
            .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_DEQ})
            .SetOutputs(PATTERN_DEQ, {PATTERN_ADD})
            .SetOutputs(PATTERN_OTHER_INPUT1, {PATTERN_ADD})
            .SetOutputs(PATTERN_ADD, {PATTERN_QUANT, PATTERN_OUTPUT1,
                        PATTERN_OUTPUT2}, TBE_OUTPUT_BRANCH_MULTI);
    patterns.push_back(pattern1);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName1.c_str());

    return patterns;
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status TbeConv2DAddMulQuantPass::GetFusionNodes(
    const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusionNodes) {

    OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to do Conv2DAddMulQuant!");
    fusionNodes = GetMatchedNodes(mapping);
    // the outputData can't be fusd
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
    OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do Conv2DAddMulQuant!");

    return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("TbeConv2DAddMulQuantPass",
                            BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            TbeConv2DAddMulQuantPass);
}  // namespace fe
