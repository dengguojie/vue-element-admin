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
 * \file batch_matmul_dropout_do_mask_v3_d_ub_fusion.cpp
 * \brief batch_matmul + dropout_do_mask_v3_d ub fusion pass
 */
#include "batch_matmul_dropout_do_mask_v3_d_ub_fusion.h"
#include <string>
#include <vector>
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "common/lxfusion_json_util.h"
#include "graph/utils/attr_utils.h"
#include "lx_fusion_func.h"

namespace fe {
namespace {
static const string PATTERN_BATCH_MATMUL = "batch_matmul";
static const string PATTERN_DROPOUTDOMASKV3D = "dropout_do_mask_v3_d";
static const string PATTERN_OTHER_INPUT = "other_input";
}  // namespace

vector<BufferFusionPattern*> BatchMatmulDropOutDoMaskV3DFusionPass::DefinePatterns() {
  /*
  * ================= pattern =================
  *
  * --> BatchMatmul_dx --> DropOutDoMaskV3D -->
  *
  * ===========================================
  */
  vector<BufferFusionPattern*> patterns;
  string pass_name = "BatchMatmulDropOutDoMaskV3DFusion";

  BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pass_name);
  FUSION_PASS_CHECK(pattern == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_BATCH_MATMUL, {OP_PATTERN_BATCH_MATMUL})
    .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT)
    .AddOpDesc(PATTERN_DROPOUTDOMASKV3D, {OP_PATTERN_DROPOUTDOMASKV3D})
    .SetHead({PATTERN_BATCH_MATMUL})
    .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_DROPOUTDOMASKV3D})
    .SetOutputs(PATTERN_BATCH_MATMUL, {PATTERN_DROPOUTDOMASKV3D});
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define pattern %s success.", pass_name.c_str());

  return patterns;
}

void BatchMatmulDropOutDoMaskV3DFusionPass::SetSplitInfo(const BufferFusionMapping &mapping, std::vector<ge::NodePtr> &fusion_nodes) {
  vector<ge::NodePtr> matmulNodes = GetMatchedNodesByDescName(PATTERN_BATCH_MATMUL, mapping);
  vector<ge::NodePtr> elemWiseNodes = GetMatchedNodesByDescName(PATTERN_DROPOUTDOMASKV3D, mapping);
  if (matmulNodes.empty()) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Matmul node not matched");
    return;
  }
  if (elemWiseNodes.empty()) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "DropOutV3 node not matched");
    return;
  }

  int pre = matmulNodes[0]->GetInDataNodes().size() - 1;
  vector<AxisSplitMap> split_maps;
  if (!GetSplitMap(split_maps, matmulNodes[0], FUSED_OP_TYPE)) {
    return;
  }
  AddElemwiseSplitMap(split_maps, elemWiseNodes[0], pre);
  SetSplitMap(split_maps, fusion_nodes, FUSED_OP_TYPE);
}

Status BatchMatmulDropOutDoMaskV3DFusionPass::GetFusionNodes(const BufferFusionMapping& mapping,
                                                             vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to do BatchMatmulDropOutDoMaskV3DFusion!");

  vector<ge::NodePtr> batch_matmul_nodes = GetMatchedNodesByDescName(PATTERN_BATCH_MATMUL, mapping);
  vector<ge::NodePtr> dropout_nodes = GetMatchedNodesByDescName(PATTERN_DROPOUTDOMASKV3D, mapping);

  FUSION_PASS_CHECK(batch_matmul_nodes.empty(),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "BatchMatMul node is not matched."),
                    return SUCCESS);
  FUSION_PASS_CHECK(dropout_nodes.empty(),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "DropOutDoMaskV3D node is not matched."),
                    return SUCCESS);

  // adjust control-edges to destroy the loop in BERT
  // "dropout_do_mask -> batch_matmul1 (control-node) -> fusion_transpose
  // -> batch_matmul2 (control-node) -> dropout_do_mask"
  for (const auto& batch_matmul_node : batch_matmul_nodes) {
    if (batch_matmul_node->GetType() != "BatchMatMul") {
      OP_LOGD(FUSED_OP_TYPE.c_str(),
              "The op_type of node [%s] should be BatchMatMul, but actually is [%s].",
              batch_matmul_node->GetName().c_str(), batch_matmul_node->GetType().c_str());
      return SUCCESS;
    }
    for (const auto& batch_matmul_control_node : batch_matmul_node->GetOutControlNodes()) {
      if (batch_matmul_control_node->GetType() != "ConfusionTransposeD") {
        continue;
      }
      // batch_matmul_control_node is confusion_transpose_node in this situation
      FUSION_PASS_CHECK(
        ge::GraphUtils::RemoveEdge(batch_matmul_node->GetOutControlAnchor(),
          batch_matmul_control_node->GetInControlAnchor()) != SUCCESS,
        OP_LOGD(FUSED_OP_TYPE.c_str(),
                "Removing control-edge between BatchMatMul and ConfusionTransposeD is failed."),
        return FAILED);
      for (const auto& confusion_transpose_out_node : batch_matmul_control_node->GetOutAllNodes()) {
        FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(batch_matmul_node->GetOutControlAnchor(),
            confusion_transpose_out_node->GetInControlAnchor()) != SUCCESS,
          OP_LOGD(FUSED_OP_TYPE.c_str(),
                  "Adding control-edge between BatchMatMul and ConfusionTransposeD's output node is failed."),
          return FAILED);
      }
    }
  }
  for (const auto& dropout_node : dropout_nodes) {
    if (dropout_node->GetType() != "DropOutDoMaskV3D") {
      OP_LOGD(FUSED_OP_TYPE.c_str(),
              "The op_type of node [%s] should be DropOutDoMaskV3D, but actually is [%s].",
              dropout_node->GetName().c_str(), dropout_node->GetType().c_str());
      return SUCCESS;
    }
    for (const auto& dropout_control_node : dropout_node->GetInControlNodes()) {
      if (dropout_control_node->GetType() != "BatchMatMul") {
        continue;
      }
      // dropout_control_node is batch_matmul_node in this situation
      FUSION_PASS_CHECK(
        ge::GraphUtils::RemoveEdge(dropout_control_node->GetOutControlAnchor(),
          dropout_node->GetInControlAnchor()) != SUCCESS,
        OP_LOGD(FUSED_OP_TYPE.c_str(),
                "Removing control-edge between BatchMatMul and DropOutDoMaskV3D is failed."),
        return FAILED);
      for (const auto& dropout_out_node : dropout_node->GetOutAllNodes()) {
        FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(dropout_control_node->GetOutControlAnchor(),
            dropout_out_node->GetInControlAnchor()) != SUCCESS,
          OP_LOGD(FUSED_OP_TYPE.c_str(),
                  "Adding control-edge between BatchMatMul and DropOutDoMaskV3D's output node is failed."),
          return FAILED);
      }
    }
  }

  fusion_nodes = GetMatchedNodes(mapping);

  // buffer fusion do not support dynamic shape now
  for (const auto& batch_matmul_node : batch_matmul_nodes) {
    vector<int64_t> input0_dims = batch_matmul_node->GetOpDesc()->GetInputDesc(0).GetOriginShape().GetDims();
    vector<int64_t> input1_dims = batch_matmul_node->GetOpDesc()->GetInputDesc(1).GetOriginShape().GetDims();
    vector<int64_t> all_dims;
    all_dims.resize(input0_dims.size() + input1_dims.size());
    merge(input0_dims.begin(), input0_dims.end(), input1_dims.begin(), input1_dims.end(), all_dims.begin());
    for (auto single_dim : all_dims) {
      if (single_dim < 0) {
        fusion_nodes.clear();
        OP_LOGW(FUSED_OP_TYPE.c_str(), "Ub fusion do not support dynamic shape.");
        return SUCCESS;
      }
    }
  }

  for (auto& batch_matmul_node : batch_matmul_nodes) {
    ge::AttrUtils::SetStr(batch_matmul_node->GetOpDesc(), UB_FUSION_OP_TYPE, "DropOutDoMaskV3D");
  }

  SetSplitInfo(mapping, fusion_nodes);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do BatchMatmulDropOutDoMaskV3DFusion!");
  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("BatchMatmulDropOutDoMaskV3DFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            BatchMatmulDropOutDoMaskV3DFusionPass);
}  // namespace fe
