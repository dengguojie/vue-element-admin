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
 * \file conv2d_writeselect_stridewrite_pass.cpp
 * \brief  tbe conv2d + write_select + stride_write ops fusion pattern
 */
#include "conv2d_writeselect_stridewrite_pass.h"
#include <string>
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {

static const char kPatternConv[] = "conv2d";
static const char kPatternDequant[] = "dequant";
static const char kPatternQuant[] = "quant";
static const char kPatternWriteselect[] = "writeselect";
static const char kPatternStridedWrite[] = "stridedwrite";
static const char kPatternOtherInput[] = "otherInput";
static const int64_t kMaxFuseNode = 6;

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
vector<BufferFusionPattern*> TbeConv2dWrtselStridewrtPass::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;
  string pass_name1 = "TbeConvDequantQuantWriteselectStridewriteFusion";
  BufferFusionPattern* pattern1 = new (std::nothrow) BufferFusionPattern(pass_name1, kMaxFuseNode);
  FUSION_PASS_CHECK((pattern1 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", pass_name1.c_str());
  // conv2d --> dequant --> quant --> write_select --> stride_write
  pattern1->AddOpDesc(kPatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternDequant, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternQuant, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternStridedWrite, {OP_PATTERN_STRIDED_WRITE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternWriteselect, {OP_PATTERN_WRITE_SELECT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternDequant})
      .SetOutputs(kPatternOtherInput, {kPatternDequant})
      .SetOutputs(kPatternDequant, {kPatternQuant})
      .SetOutputs(kPatternQuant, {kPatternWriteselect})
      .SetOutputs(kPatternWriteselect, {kPatternStridedWrite});
  patterns.push_back(pattern1);

  string pass_name2 = "TbeConvDequantWriteselectStridewriteFusion";
  BufferFusionPattern* pattern2 = new (std::nothrow) BufferFusionPattern(pass_name2);
  FUSION_PASS_CHECK((pattern2 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", pass_name2.c_str());
  // conv2d --> dequant --> write_select --> stride_write
  pattern2->AddOpDesc(kPatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternDequant, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternStridedWrite, {OP_PATTERN_STRIDED_WRITE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternWriteselect, {OP_PATTERN_WRITE_SELECT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternDequant})
      .SetOutputs(kPatternDequant, {kPatternWriteselect})
      .SetOutputs(kPatternWriteselect, {kPatternStridedWrite})
      .SetOutputs(kPatternOtherInput, {kPatternDequant});
  patterns.push_back(pattern2);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", pass_name2.c_str());

  return patterns;
}
REGISTER_BUFFER_FUSION_PASS("TbeConv2dWrtselStridewrtPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            TbeConv2dWrtselStridewrtPass);
}  // namespace fe
