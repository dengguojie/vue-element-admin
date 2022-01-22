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

/*!
 * \file matmul_gelugrad_ub_fusion.cpp
 * \brief tbe matmul + reducesum ops fusion pattern
 */
#include "matmul_reduce_sum_ub_fusion.h"

#include <string>
#include "pattern_fusion_util.h"
#include "op_log.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "common/lxfusion_json_util.h"
#include "graph/utils/attr_utils.h"
#include "lx_fusion_func.h"
#include "anchor_util.h"

namespace fe {
static const char PATTERN_MATMUL[] = "batch_matmul";
static const char PATTERN_REDUCESUM[] = "reduce_sum_d";
static const char PATTERN_CAST[] = "cast";
static const int kNumTwo = 2;

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

void MatmulReduceSumUbFusion::SetSplitInfo(const BufferFusionMapping &mapping, std::vector<ge::NodePtr> &fusion_nodes) {
  vector<ge::NodePtr> matmulNodes = GetMatchedNodesByDescName(PATTERN_MATMUL, mapping);
  vector<ge::NodePtr> ReduceSumNodes = GetMatchedNodesByDescName(PATTERN_REDUCESUM, mapping);
  if (matmulNodes.empty()) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Matmul node not matched");
    return;
  }
  if (ReduceSumNodes.empty()) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "ReduceSum node not matched");
    return;
  }

  vector<AxisSplitMap> split_maps;
  OpL1FusionType fusion_type = L1FUSION_DISABLE;
  int64_t min_tbe_l1space = 0;
  if (!GetSplitMap(split_maps, matmulNodes[0], FUSED_OP_TYPE, fusion_type, min_tbe_l1space)) {
    return;
  }
  auto output0desc = GetCurrNodeOutputDesc(matmulNodes[0], 0);
  FUSION_PASS_CHECK(output0desc == nullptr,
              CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "output0desc is null"),
              return);
  FUSION_PASS_CHECK(output0desc->GetOriginShape().GetDims().size() < kNumTwo,
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Matmul output shape dims < 2."), return);
  int batch_lenth = output0desc->GetOriginShape().GetDims().size() - kNumTwo;

  for (int batch_index = 0; batch_index < batch_lenth; batch_index++){
    DelSplitInfoByOutputAxis(split_maps, batch_index);
  }

  SetSplitMap(split_maps, fusion_nodes, FUSED_OP_TYPE, fusion_type, min_tbe_l1space);
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
    if (matmulNode->GetType() != "BatchMatMul" && matmulNode->GetType() != "BatchMatMulV2") {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "ub fusion not support this OP, skip fusion.");
      return SUCCESS;
    }
    vector<int64_t> matmul_output_shape = matmulNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDims();
    int batch_lenth = matmul_output_shape.size() - kNumTwo;
    if (batch_lenth > 1) {
      OP_LOGD(FUSED_OP_TYPE, "Only support BatchMatMul shape batch dim is 1.");
      return SUCCESS;
    }
    vector<int64_t> matmul_x1_shape = matmulNode->GetOpDesc()->GetInputDesc(0).GetOriginShape().GetDims();
    if (matmul_x1_shape.size() > kNumTwo && matmul_x1_shape[0] == 1) {
      OP_LOGD(FUSED_OP_TYPE, "Do not support BatchMatMul shape batch=1.");
      return SUCCESS;
    }
    vector<int64_t> matmul_x2_shape = matmulNode->GetOpDesc()->GetInputDesc(1).GetOriginShape().GetDims();
    if (matmul_x2_shape.size() > kNumTwo && matmul_x2_shape[0] == 1) {
      OP_LOGD(FUSED_OP_TYPE, "Do not support BatchMatMul shape batch=1.");
      return SUCCESS;
    }
  }

  for (auto reduceNode : reduceSumDNodes) {
    if (reduceNode->GetType() != "ReduceSumD") {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "ub fusion not support this OP, skip fusion.");
      return SUCCESS;
    }
    auto output0desc = GetCurrNodeOutputDesc(reduceNode, 0);
    FUSION_PASS_CHECK(output0desc == nullptr,
              CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "output0desc is null"),
              return FAILED);
    if (output0desc->GetDataType() != ge::DT_FLOAT) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "ub fusion not support reduce output type not fp32, skip fusion.");
      return SUCCESS;
    }
    bool keep_dims;
    if (ge::AttrUtils::GetBool(reduceNode->GetOpDesc(), "keep_dims", keep_dims) && keep_dims) {
      OP_LOGW(FUSED_OP_TYPE, "ub fusion do not support keep_dims is true.");
      return SUCCESS;
    }
  }

  fusionNodes = GetMatchedNodes(mapping);
  SetSplitInfo(mapping, fusionNodes);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do MatmulReduceSumUbFusion!");

  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("MatmulReduceSumUbFusion", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, MatmulReduceSumUbFusion);
}  // namespace fe
