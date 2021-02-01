/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file batch_matmul_fusedmuladd_ub_fusion.cpp
 * \brief tbe batchmatmul and fusedmuladd ops fusion pattern
 */
#include "batch_matmul_fusedmuladd_ub_fusion.h"
#include <string>
#include <vector>
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {
namespace {
static const char PATTERN_BATCH_MATMUL[] = "batchmatmul";
static const char PATTERN_ELEM[] = "elemwise";
}  // namespace

/*
 * @brief:  define Matmul and element-wise op fusion pattern
 *
 *   Matmul + FusedMulAdd
 *
 * fusion node:  Matmul, FusedMulAdd
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern*> TbeBatchMatmulFusedMulAddFusionPass::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;
  string passName = "TbeBatchMatmulFusedMulAddFusionPass";

  BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(passName);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName.c_str());
  // define pattern rules
  pattern->AddOpDesc(PATTERN_BATCH_MATMUL, {OP_PATTERN_BATCH_MATMUL}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELEM, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_BATCH_MATMUL})
      .SetOutputs(PATTERN_BATCH_MATMUL, {PATTERN_ELEM}, TBE_OUTPUT_BRANCH_SINGLE, true)
      .SetOutputs(PATTERN_ELEM, {}, TBE_OUTPUT_BRANCH_SINGLE, true);
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName.c_str());

  return patterns;
}

REGISTER_BUFFER_FUSION_PASS("TbeBatchMatmulFusedMulAddFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            TbeBatchMatmulFusedMulAddFusionPass);
}  // namespace fe
