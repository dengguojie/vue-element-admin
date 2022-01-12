/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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

#include "tbe_matmul_requant_pass.h"
#include <string>
#include <vector>
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {
using std::vector;

static const char kPattternMatmul[] = "convolution";
static const char kPattternRequant[] = "requant";
static const char kPattternOtherInput[] = "otherInput";
static const char kPattternStrideWrite[] = "strided_write";
static const string kFusedOpType = "FusedOp";

/*
 * @brief:  define convolution and single input op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    Matmul-->AscendReQuant
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern *> TbeMatmulRequantFusionPass::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;

  string pass_name = "TbeMatmulRequantFusion";
  BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(pass_name);
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(kFusedOpType.c_str(), "new an object failed."), return patterns);
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name.c_str());
  // define pattern rules Matmul-->AcendReQuant
  pattern
      ->AddOpDesc(kPattternMatmul, {OP_PATTERN_MATMUL, OP_PATTERN_GEMM, OP_PATTERN_BATCH_MATMUL},
                  TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPattternRequant, {OP_PATTERN_REQUANT},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPattternOtherInput, {TBE_PATTERN_INPUT_NODE},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPattternStrideWrite, {OP_PATTERN_STRIDED_WRITE},
                 TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .SetHead({kPattternMatmul})
      .SetOutputs(kPattternMatmul, {kPattternRequant})
      .SetOutputs(kPattternRequant, {kPattternStrideWrite})
      .SetOutputs(kPattternOtherInput, {kPattternRequant});
  patterns.push_back(pattern);
  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name.c_str());
  return patterns;
}
REGISTER_BUFFER_FUSION_PASS("TbeMatmulRequantFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            TbeMatmulRequantFusionPass);
}  // namespace fe
