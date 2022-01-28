/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
 * \file batch_matmul_quant_ub_fusion.cpp
 * \brief tbe batchmatmul and quant ops fusion pattern
 */
#include "batch_matmul_quant_ub_fusion.h"

#include <string>
#include <vector>

#include "anchor_util.h"
#include "common/lxfusion_json_util.h"
#include "graph/utils/attr_utils.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "lx_fusion_func.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

namespace fe {
namespace {
static const char PATTERN_BATCH_MATMUL[] = "batchmatmul";
static const char PATTERN_OTHER_INPUT[] = "InputData";
static const char PATTERN_DEQUANT[] = "dequant";
static const char PATTERN_QUANT[] = "quant";
static const vector<string> MATMUL_WHITELIST = {"MatMul", "MatMulV2", "BatchMatMul", "BatchMatMulV2"};
}  // namespace

/*
 * @brief:  define Matmul and dequant quant op fusion pattern
 *
 *   Matmul + (dequant) + quant
 *
 * fusion node:  Matmul, (dequant), quant
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern *> TbeBatchMatmulQuantFusionPass::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;

  string passName = "TbeBatchMatmulQuantPass";
  BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(passName);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(FUSED_OP_TYPE, "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE, "Start to define %s pass pattern.", passName.c_str());
  // define pattern rules
  pattern
      ->AddOpDesc(PATTERN_BATCH_MATMUL, {OP_PATTERN_MATMUL, OP_PATTERN_BATCH_MATMUL}, TBE_PATTERN_NUM_DEFAULT,
                  TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_DEQUANT, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_QUANT, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_BATCH_MATMUL})
      .SetOutputs(PATTERN_BATCH_MATMUL, {PATTERN_DEQUANT})
      .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_DEQUANT})
      .SetOutputs(OP_PATTERN_DEQUANT, {PATTERN_QUANT});
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE, "End to define %s pass pattern.", passName.c_str());
  return patterns;
}

const void TbeBatchMatmulQuantFusionPass::SetSplitInfo(const BufferFusionMapping &mapping,
                                                       std::vector<ge::NodePtr> &fusion_nodes) {
  vector<ge::NodePtr> matmul_nodes = GetMatchedNodesByDescName(PATTERN_BATCH_MATMUL, mapping);
  vector<ge::NodePtr> dequant_nodes = GetMatchedNodesByDescName(PATTERN_DEQUANT, mapping);
  FUSION_PASS_CHECK(matmul_nodes.empty(), OP_LOGW(FUSED_OP_TYPE, "Matmul node not matched"), return);
  FUSION_PASS_CHECK(matmul_nodes[0]->GetInDataNodes().size() <= 0,
                    OP_LOGE(FUSED_OP_TYPE, "Matmul Nodes's inputsize can not smaller or equal to zero."), return);
  size_t pre = matmul_nodes[0]->GetInDataNodes().size() - 1;
  vector<AxisSplitMap> split_maps;
  OpL1FusionType fusion_type = L1FUSION_DISABLE;
  int64_t min_tbe_l1space = 0;
  if (!GetSplitMap(split_maps, matmul_nodes[0], FUSED_OP_TYPE, fusion_type, min_tbe_l1space)) {
    return;
  }
  bool tensor_mode = false;
  if (!dequant_nodes.empty()) {
    pre += 1;
    auto deq_scale = GetCurrNodeMutableInputDesc(dequant_nodes[0], "deq_scale");
    vector<int64_t> scalar = {1};
    tensor_mode = deq_scale != nullptr && deq_scale->GetOriginShape().GetDims() != scalar;
  }
  // the dequant is scalar mode, can not split c_dim
  if (!tensor_mode) {
    DelSplitInfoByOutputAxis(split_maps, static_cast<int>(pre));
  }
  SetSplitMap(split_maps, fusion_nodes, FUSED_OP_TYPE, fusion_type, min_tbe_l1space);
}

Status TbeBatchMatmulQuantFusionPass::GetFusionNodes(const BufferFusionMapping &mapping,
                                                     vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(FUSED_OP_TYPE, "Begin to do TbeBatchMatmulQuantFusionPass!");

  fusion_nodes.clear();
  vector<ge::NodePtr> matmul_nodes = GetMatchedNodesByDescName(PATTERN_BATCH_MATMUL, mapping);
  vector<ge::NodePtr> dequant_nodes = GetMatchedNodesByDescName(PATTERN_DEQUANT, mapping);
  vector<ge::NodePtr> quant_nodes = GetMatchedNodesByDescName(PATTERN_QUANT, mapping);

  FUSION_PASS_CHECK(quant_nodes.empty(), OP_LOGW(FUSED_OP_TYPE, "Quant node not match!"), return SUCCESS);

  // check whether matmul/batchmatmul op and if dynamic mode or not
  for (const auto &matmul_node : matmul_nodes) {
    if (find(MATMUL_WHITELIST.begin(), MATMUL_WHITELIST.end(), matmul_node->GetType()) == MATMUL_WHITELIST.end()) {
      OP_LOGD(FUSED_OP_TYPE, "fcNode op[%s] type[%s] is not supported for this ub fusion pass, skip fusion.",
              matmul_node->GetName().c_str(), matmul_node->GetType().c_str());
      return SUCCESS;
    }
    auto input0desc = GetCurrNodeInputDesc(matmul_node, 0);
    auto input1desc = GetCurrNodeInputDesc(matmul_node, 1);
    FUSION_PASS_CHECK(input0desc == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE, "inputDesc0 is null"),
                      return SUCCESS);
    FUSION_PASS_CHECK(input1desc == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE, "inputDesc1 is null"),
                      return SUCCESS);
    vector<int64_t> input0_dims = input0desc->GetOriginShape().GetDims();
    vector<int64_t> input1_dims = input1desc->GetOriginShape().GetDims();
    vector<int64_t> all_dim;
    all_dim.resize(input0_dims.size() + input1_dims.size());
    merge(input0_dims.begin(), input0_dims.end(), input1_dims.begin(), input1_dims.end(), all_dim.begin());
    for (auto single_dim : all_dim) {
      if (single_dim < 0) {
        OP_LOGW(FUSED_OP_TYPE, "UB fusion not support dynamic shape");
        return SUCCESS;
      }
    }
  }
  fusion_nodes = GetMatchedNodes(mapping);
  SetSplitInfo(mapping, fusion_nodes);
  OP_LOGD(FUSED_OP_TYPE, "End to do TbeBatchMatmulQuantFusionPass!");

  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("TbeBatchMatmulQuantFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            TbeBatchMatmulQuantFusionPass);
}  // namespace fe
