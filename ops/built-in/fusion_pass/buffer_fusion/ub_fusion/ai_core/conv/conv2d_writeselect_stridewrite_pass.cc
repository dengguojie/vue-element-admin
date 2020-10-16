/**
 * @file conv2d_writeselect_stridewrite_pass.cpp
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief tbe conv2d + write_select + stride_write ops fusion pattern
 *
 * @version 1.0
 *
 */

#include "conv2d_writeselect_stridewrite_pass.h"
#include <string>
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {

static const char PATTERN_CONV[] = "conv2d";
static const char PATTERN_DEQUANT[] = "dequant";
static const char PATTERN_QUANT[] = "quant";
static const char PATTERN_WRITESELECT[] = "writeselect";
static const char PATTERN_STRIDED_WRITE[] = "stridedwrite";
static const char PATTERN_OTHER_INPUT[] = "otherInput";
static const int64_t MAX_FUSE_NODE = 6;

/*
 * @brief:  define conv2d op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    conv2d --> dequant --> quant --> write_select --> stride_write
 *    conv2d --> dequant --> write_select --> stride_write
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern *> TbeConv2dWrtselStridewrtPass::DefinePatterns() {

    vector<BufferFusionPattern *> patterns;
    string passName1 = "TbeConvDequantQuantWriteselectStridewriteFusion";
    BufferFusionPattern *pattern1 =
      new (std::nothrow) BufferFusionPattern(passName1, MAX_FUSE_NODE);
    FUSION_PASS_CHECK((pattern1 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."),
             return patterns);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName1.c_str());
    // conv2d --> dequant --> quant --> write_select --> stride_write
    pattern1
        ->AddOpDesc(PATTERN_CONV, {OP_PATTERN_CONV},
                    TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(PATTERN_DEQUANT, {OP_PATTERN_DEQUANT},
                   TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(PATTERN_QUANT, {OP_PATTERN_QUANT},
                   TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(PATTERN_STRIDED_WRITE, {OP_PATTERN_STRIDED_WRITE},
                   TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE},
                   TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(PATTERN_WRITESELECT, {OP_PATTERN_WRITE_SELECT},
                   TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .SetHead({PATTERN_CONV})
        .SetOutputs(PATTERN_CONV, {PATTERN_DEQUANT})
        .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_DEQUANT})
        .SetOutputs(PATTERN_DEQUANT, {PATTERN_QUANT})
        .SetOutputs(PATTERN_QUANT, {PATTERN_WRITESELECT})
        .SetOutputs(PATTERN_WRITESELECT, {PATTERN_STRIDED_WRITE});
      patterns.push_back(pattern1);

    string passName2 = "TbeConvDequantWriteselectStridewriteFusion";
    BufferFusionPattern *pattern2 =
      new (std::nothrow) BufferFusionPattern(passName2);
    FUSION_PASS_CHECK((pattern2 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."),
             return patterns);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName2.c_str());
    // conv2d --> dequant --> write_select --> stride_write
    pattern2
        ->AddOpDesc(PATTERN_CONV, {OP_PATTERN_CONV},
                    TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(PATTERN_DEQUANT, {OP_PATTERN_DEQUANT},
                   TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(PATTERN_STRIDED_WRITE, {OP_PATTERN_STRIDED_WRITE},
                   TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE},
                   TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(PATTERN_WRITESELECT, {OP_PATTERN_WRITE_SELECT},
                   TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .SetHead({PATTERN_CONV})
        .SetOutputs(PATTERN_CONV, {PATTERN_DEQUANT})
        .SetOutputs(PATTERN_DEQUANT, {PATTERN_WRITESELECT})
        .SetOutputs(PATTERN_WRITESELECT, {PATTERN_STRIDED_WRITE})
        .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_DEQUANT});
    patterns.push_back(pattern2);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName2.c_str());

    return patterns;
}
REGISTER_BUFFER_FUSION_PASS("TbeConv2dWrtselStridewrtPass",
                            BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            TbeConv2dWrtselStridewrtPass);
}  // namespace fe
