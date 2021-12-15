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
 * \file matmul_dropout_do_mask_v3_d_ub_fusion.cpp
 * \brief matmul + dropout_do_mask_v3_d ub fusion pass
 */
#include "matmul_dropout_do_mask_v3_d_ub_fusion.h"
#include <string>
#include <vector>
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "common/lxfusion_json_util.h"
#include "graph/utils/attr_utils.h"
#include "lx_fusion_func.h"
#include "anchor_util.h"

namespace fe {
namespace {
static const string PATTERN_MATMUL = "matmul";
static const string PATTERN_DROPOUTDOMASKV3D = "dropout_do_mask_v3_d";
static const string PATTERN_OTHER_INPUT = "other_input";
static const string PATTERN_OTHER_INPUT1 = "other_input1";
static const string PATTERN_ADD = "add";
}  // namespace

vector<BufferFusionPattern*> MatmulDropOutDoMaskV3DFusionPass::DefinePatterns() {
  /*
  * ===================== pattern =====================
  *
  * --> Matmul (BiasAdd) --> DropOutDoMaskV3D --> Add
  *
  * ===================================================
  */
  vector<BufferFusionPattern*> patterns;
  string pass_name = "MatmulDropOutDoMaskV3DFusion";

  BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pass_name);
  FUSION_PASS_CHECK(pattern == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_MATMUL, {OP_PATTERN_MATMUL})
    .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT)
    .AddOpDesc(PATTERN_OTHER_INPUT1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT)
    .AddOpDesc(PATTERN_DROPOUTDOMASKV3D, {OP_PATTERN_DROPOUTDOMASKV3D})
    .AddOpDesc(PATTERN_ADD, {OP_PATTERN_ELEMWISE})
    .SetHead({PATTERN_MATMUL})
    .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_DROPOUTDOMASKV3D})
    .SetOutputs(PATTERN_MATMUL, {PATTERN_DROPOUTDOMASKV3D})
    .SetOutputs(PATTERN_OTHER_INPUT1, {PATTERN_ADD})
    .SetOutputs(PATTERN_DROPOUTDOMASKV3D, {PATTERN_ADD});
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define pattern %s success.", pass_name.c_str());

  return patterns;
}

void MatmulDropOutDoMaskV3DFusionPass::SetSplitInfo(const BufferFusionMapping &mapping, std::vector<ge::NodePtr> &fusion_nodes) {
  vector<ge::NodePtr> matmulNodes = GetMatchedNodesByDescName(PATTERN_MATMUL, mapping);
  vector<ge::NodePtr> elemWiseNodes = GetMatchedNodesByDescName(PATTERN_DROPOUTDOMASKV3D, mapping);
  vector<ge::NodePtr> elemWiseNodes1 = GetMatchedNodesByDescName(PATTERN_ADD, mapping);
  if (matmulNodes.empty()) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Matmul node not matched");
    return;
  }
  if (elemWiseNodes.empty()) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "DropOutV3 node not matched");
    return;
  }
  if (elemWiseNodes1.empty()) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "add node not matched");
    return;
  }

  int pre = matmulNodes[0]->GetInDataNodes().size() - 1;
  vector<AxisSplitMap> split_maps;
  OpL1FusionType fusion_type = L1FUSION_DISABLE;
  int64_t min_tbe_l1space = 0;
  if (!GetSplitMap(split_maps, matmulNodes[0], FUSED_OP_TYPE, fusion_type, min_tbe_l1space)) {

    return;
  }
  AddElemwiseSplitMap(split_maps, elemWiseNodes[0], pre);
  AddElemwiseSplitMap(split_maps, elemWiseNodes1[0], pre);
  SetSplitMap(split_maps, fusion_nodes, FUSED_OP_TYPE, fusion_type, min_tbe_l1space);
}

Status MatmulDropOutDoMaskV3DFusionPass::GetFusionNodes(const BufferFusionMapping& mapping,
                                                        vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to do MatmulDropOutDoMaskV3DFusion!");

  vector<ge::NodePtr> matmul_nodes = GetMatchedNodesByDescName(PATTERN_MATMUL, mapping);
  vector<ge::NodePtr> dropout_nodes = GetMatchedNodesByDescName(PATTERN_DROPOUTDOMASKV3D, mapping);
  vector<ge::NodePtr> add_nodes = GetMatchedNodesByDescName(PATTERN_ADD, mapping);

  FUSION_PASS_CHECK(matmul_nodes.empty(),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "MatMul node is not matched."),
                    return SUCCESS);
  FUSION_PASS_CHECK(dropout_nodes.empty(),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "DropOutDoMaskV3D node is not matched."),
                    return SUCCESS);
  FUSION_PASS_CHECK(add_nodes.empty(),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Elemwise node is not matched."),
                    return SUCCESS);

  for (const auto& matmul_node : matmul_nodes) {
    if (matmul_node->GetType() != "MatMulV2") {
      OP_LOGD(FUSED_OP_TYPE.c_str(),
              "The op_type of node [%s] should be MatMulV2, but actually is [%s].",
              matmul_node->GetName().c_str(), matmul_node->GetType().c_str());
      return SUCCESS;
    }
  }
  for (const auto& dropout_node : dropout_nodes) {
    if (dropout_node->GetType() != "DropOutDoMaskV3D") {
      OP_LOGD(FUSED_OP_TYPE.c_str(),
              "The op_type of node [%s] should be DropOutDoMaskV3D, but actually is [%s].",
              dropout_node->GetName().c_str(), dropout_node->GetType().c_str());
      return SUCCESS;
    }
  }
  for (const auto& add_node : add_nodes) {
    if (add_node->GetType() != "Add") {
      OP_LOGD(FUSED_OP_TYPE.c_str(),
              "The op_type of node [%s] should be Add, but actually is [%s].",
              add_node->GetName().c_str(), add_node->GetType().c_str());
      return SUCCESS;
    }
  }

  fusion_nodes = GetMatchedNodes(mapping);

  // buffer fusion do not support dynamic shape now
  for (const auto& matmul_node : matmul_nodes) {
    auto input0desc = GetCurrNodeInputDesc(matmul_node, 0);
    auto input1desc = GetCurrNodeInputDesc(matmul_node, 1);
    FUSION_PASS_CHECK(input0desc == nullptr,
                  CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputDesc0 is null"),
                  return FAILED);
    FUSION_PASS_CHECK(input1desc == nullptr,
                  CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputDesc1 is null"),
                  return FAILED);
    vector<int64_t> input0_dims = input0desc->GetOriginShape().GetDims();
    vector<int64_t> input1_dims = input1desc->GetOriginShape().GetDims();
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

  for (auto& matmul_node : matmul_nodes) {
    ge::AttrUtils::SetStr(matmul_node->GetOpDesc(), UB_FUSION_OP_TYPE, "DropOutDoMaskV3D");
  }

  SetSplitInfo(mapping, fusion_nodes);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do MatmulDropOutDoMaskV3DFusion!");
  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("MatmulDropOutDoMaskV3DFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            MatmulDropOutDoMaskV3DFusionPass);
}  // namespace fe
