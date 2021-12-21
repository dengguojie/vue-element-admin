/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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

#include "tbe_depthwiseconv_requant_pass.h"
#include <string>
#include <vector>
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {
using std::vector;

static const char kPatternConv[] = "convolution";
static const char kPatternRequant[] = "requant";
static const char kPatternOtherInput[] = "otherInput";
static const char kPatternStridedWrite[] = "strided_write";
static const string kFusedOpType = "FusedOp";
/*
 * @brief:  define convolution and single input op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    Convolution-->AscendReQuant
 *
 * fusion node: AcendReQuantS16, Convolution
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern *> TbeDepthwiseConvRequantFusionPass::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;

  string pass_name = "TbeConvRequantFusion";
  BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(pass_name);
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(kFusedOpType.c_str(), "new an object failed."), return patterns);
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name.c_str());
  // define pattern rules Convolution-->AcendDeQuant-->AscendQuant
  pattern
      ->AddOpDesc(kPatternConv, {OP_PATTERN_DEPTHWISE_CONV},
                  TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternRequant, {OP_PATTERN_REQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternStridedWrite, {OP_PATTERN_STRIDED_WRITE}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternRequant})
      .SetOutputs(kPatternRequant, {kPatternStridedWrite})
      .SetOutputs(kPatternOtherInput, {kPatternRequant});
  patterns.push_back(pattern);
  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name.c_str());
  return patterns;
}
REGISTER_BUFFER_FUSION_PASS("TbeDepthwiseConvRequantFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            TbeDepthwiseConvRequantFusionPass);
}  // namespace fe
