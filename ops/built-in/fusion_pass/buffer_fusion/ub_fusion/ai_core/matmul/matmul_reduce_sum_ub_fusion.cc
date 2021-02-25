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
 * \file matmul_gelugrad_ub_fusion.cpp
 * \brief tbe matmul + reducesum ops fusion pattern
 */
#include "matmul_reduce_sum_ub_fusion.h"

#include <string>
#include "pattern_fusion_util.h"
#include "op_log.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {

static const char PATTERN_MATMUL[] = "batch_matmul";
static const char PATTERN_REDUCESUM[] = "reduce_sum_d";
static const char PATTERN_CAST[] = "cast";

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
vector<BufferFusionPattern*> MatmulReduceSumUbFusion::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;
  string passName = "MatmulReduceSumUbFusion";
  BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(passName);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName.c_str());
  pattern->AddOpDesc(PATTERN_MATMUL, {OP_PATTERN_BATCH_MATMUL}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_REDUCESUM, {OP_PATTERN_COMMONREDUCE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_MATMUL})
      .SetOutputs(PATTERN_MATMUL, {PATTERN_REDUCESUM}, TBE_OUTPUT_BRANCH_SINGLE, true);
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
Status MatmulReduceSumUbFusion::GetFusionNodes(const BufferFusionMapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to do MatmulReduceSumUbFusion!");
  vector<ge::NodePtr> matmulNodes = GetMatchedNodesByDescName(PATTERN_MATMUL, mapping);
  vector<ge::NodePtr> reduceSumDNodes = GetMatchedNodesByDescName(PATTERN_REDUCESUM, mapping);
  for (auto matmulNode : matmulNodes) {
    if (matmulNode->GetType() != "BatchMatMul") {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "ub fusion not support this OP, skip fusion.");
      return SUCCESS;
    }
  }

  for (auto reduceNode : reduceSumDNodes) {
    if (reduceNode->GetType() != "ReduceSumD") {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "ub fusion not support this OP, skip fusion.");
      return SUCCESS;
    }

    if (reduceNode->GetOpDesc()->GetOutputDesc(0).GetDataType() != ge::DT_FLOAT) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "ub fusion not support reduce output type not fp32, skip fusion.");
      return SUCCESS;
    }
  }

  fusionNodes = GetMatchedNodes(mapping);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do MatmulReduceSumUbFusion!");

  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("MatmulReduceSumUbFusion", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, MatmulReduceSumUbFusion);
}  // namespace fe
