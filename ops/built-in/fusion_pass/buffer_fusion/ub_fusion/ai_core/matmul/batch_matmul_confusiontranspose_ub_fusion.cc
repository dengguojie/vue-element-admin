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
 * \file batch_matmul_confusiontranspose_ub_fusion.cpp
 * \brief tbe batch_matmul + confusiontransposed ops fusion pattern
 */
#include "batch_matmul_confusiontranspose_ub_fusion.h"

#include <string>

#include "anchor_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

namespace fe {
static const char PATTERN_BATCHMATMUL[] = "batch_matmul";
static const char PATTERN_BATCH_MATMUL_CONFUSION_TRANSPOSE[] = "batchmatmul_transpose";
/*
 * @brief:  define batchmatmul op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    batchmatmul --> confusiontransposed
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern *> BatchMatmulConfusiontransposeUbFusion::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;
  string pass_name = "BatchMatmulConfusiontransposeUbFusion";
  BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(pass_name);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGW(FUSED_OP_TYPE.c_str(), "can not new an object."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", pass_name.c_str());
  pattern->AddOpDesc(PATTERN_BATCHMATMUL, {OP_PATTERN_BATCH_MATMUL}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_BATCH_MATMUL_CONFUSION_TRANSPOSE, {OP_PATTERN_CONFUSION_TRANSPOSE}, TBE_PATTERN_NUM_DEFAULT,
                 TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_BATCHMATMUL})
      .SetOutputs(PATTERN_BATCHMATMUL, {PATTERN_BATCH_MATMUL_CONFUSION_TRANSPOSE});
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", pass_name.c_str());

  return patterns;
}

/*
 * @brief: check input parameters before DoFusion
 * @param [in] matmul_nodes: nodes matched pattern of matmul
 * @param [int] transpose_nodes: nodes matched pattern of transpose
 * @return bool: check status ok or not.
 */
Status BatchMatmulConfusiontransposeUbFusion::CheckInputParameters(const vector<ge::NodePtr> &matmul_nodes,
                                                                   const vector<ge::NodePtr> &transpose_nodes) {
  for (auto matmul_node : matmul_nodes) {
    for (auto matmul_control_node : matmul_node->GetOutControlNodes()) {
      FUSION_PASS_CHECK(matmul_control_node == nullptr, OP_LOGW(matmul_node, "out control of matmul is null"),
                        return FAILED);
      if (matmul_control_node->GetType() != "ConfusionTransposeD" &&
          matmul_control_node->GetType() != "DropOutDoMaskV3D" && matmul_control_node->GetType() != "Add") {
        continue;
      }
      FUSION_PASS_CHECK(
          ge::GraphUtils::RemoveEdge(matmul_node->GetOutControlAnchor(), matmul_control_node->GetInControlAnchor()) !=
              SUCCESS,
          OP_LOGD(FUSED_OP_TYPE.c_str(), "remove edge between batch_matmul and confusion_transpose_d error"),
          return FAILED);
      for (auto transpose_out_node : matmul_control_node->GetOutAllNodes()) {
        FUSION_PASS_CHECK(transpose_out_node == nullptr,
                          OP_LOGD(FUSED_OP_TYPE.c_str(), "output node of transpose is null"), return FAILED);
        FUSION_PASS_CHECK(
            ge::GraphUtils::AddEdge(matmul_node->GetOutControlAnchor(),transpose_out_node->GetInControlAnchor()) !=
                SUCCESS,
            OP_LOGD(FUSED_OP_TYPE.c_str(), "add edge between batch_matmul and confusion_transpose_d's output error"),
            return FAILED);
      }
    }
  }

  for (auto tranpose_node : transpose_nodes) {
    for (auto transpose_control_node : tranpose_node->GetInControlNodes()) {
      FUSION_PASS_CHECK(transpose_control_node == nullptr, OP_LOGW(tranpose_node, "in control of transpose is null"),
                        return SUCCESS);
      if (transpose_control_node->GetType() != "BatchMatMul" && transpose_control_node->GetType() != "BatchMatMulV2") {
        continue;
      }
      FUSION_PASS_CHECK(
          ge::GraphUtils::RemoveEdge(transpose_control_node->GetOutControlAnchor(),
                                     tranpose_node->GetInControlAnchor()) != SUCCESS,
          OP_LOGD(FUSED_OP_TYPE.c_str(), "can not remove edge between batch_matmul and confusion_transpose_d"),
          return FAILED);
      for (auto transpose_out_node : tranpose_node->GetOutAllNodes()) {
        FUSION_PASS_CHECK(transpose_out_node == nullptr,
                          OP_LOGD(FUSED_OP_TYPE.c_str(), "output node of transpose is null"), return FAILED);
        FUSION_PASS_CHECK(
            ge::GraphUtils::AddEdge(transpose_control_node->GetOutControlAnchor(),
                                    transpose_out_node->GetInControlAnchor()) != SUCCESS,
            OP_LOGD(FUSED_OP_TYPE.c_str(), "can not add edge between batch_matmul and confusion_transpose_d's output"),
            return FAILED);
      }
    }
  }

  return SUCCESS;
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] mapping: nodes matched by pattern
 * @param [out] fusionNodes: nodes to be fusioned
 * @return bool: fusion status ok or not.
 */
Status BatchMatmulConfusiontransposeUbFusion::GetFusionNodes(const BufferFusionMapping &mapping,
                                                             vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to do BatchMatmulConfusiontransposeUbFusion!");
  vector<ge::NodePtr> matmul_nodes = GetMatchedNodesByDescName(PATTERN_BATCHMATMUL, mapping);
  vector<ge::NodePtr> transpose_nodes = GetMatchedNodesByDescName(PATTERN_BATCH_MATMUL_CONFUSION_TRANSPOSE, mapping);

  FUSION_PASS_CHECK(CheckInputParameters(matmul_nodes, transpose_nodes) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "check parameter failed"), return SUCCESS);

  // buffer fusion do not support dynamic shape now
  for (const auto &matmul_node : matmul_nodes) {
    auto input0_desc = GetCurrNodeInputDesc(matmul_node, 0);
    auto input1_desc = GetCurrNodeInputDesc(matmul_node, 1);
    FUSION_PASS_CHECK(input0_desc == nullptr, OP_LOGW(matmul_node, "inputDesc0 is null"), return SUCCESS);
    FUSION_PASS_CHECK(input1_desc == nullptr, OP_LOGW(matmul_node, "inputDesc1 is null"), return SUCCESS);
    vector<int64_t> input0_dims = input0_desc->GetOriginShape().GetDims();
    vector<int64_t> input1_dims = input1_desc->GetOriginShape().GetDims();
    vector<int64_t> allDims;
    allDims.resize(input0_dims.size() + input1_dims.size());
    merge(input0_dims.begin(), input0_dims.end(), input1_dims.begin(), input1_dims.end(), allDims.begin());
    for (auto singleDim : allDims) {
      if (singleDim < 0) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "ub fusion not support dynamic shape");
        return SUCCESS;
      }
    }
  }

  fusion_nodes = GetMatchedNodes(mapping);

  // multi input node can not be fused except head node
  for (auto &item : mapping) {
    auto opdesc = find(item.first->types.begin(), item.first->types.end(), TBE_PATTERN_OUTPUT_NODE);
    if (opdesc != item.first->types.end()) {
      for (auto &node : item.second) {
        auto nodePtr = find(fusion_nodes.begin(), fusion_nodes.end(), node);
        if (nodePtr != fusion_nodes.end()) {
          fusion_nodes.erase(nodePtr);
        }
      }
    }
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do BatchMatmulConfusiontransposeUbFusion!");
  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("BatchMatmulConfusiontransposeUbFusion", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            BatchMatmulConfusiontransposeUbFusion);
}  // namespace fe
