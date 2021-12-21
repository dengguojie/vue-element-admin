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
 * \file arg_max_with_k_fusion_pass.cpp
 * \brief
 */
#include "arg_max_with_k_fusion_pass.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>

#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

static const char FUSED_NODE[] = "ArgMaxWithK";
static const char PATTERN_FUSEDNODE[] = "ArgMaxWithK";

namespace fe {
vector<FusionPattern*> ArgMaxWithKFusionPass::DefinePatterns() {

  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("ArgMaxWithKFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

Status ArgMaxWithKFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  // get the NodePtr of ArgMaxWithK
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);
  // get the OpDescPtr of ArgMaxWithK
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  // get the attr
  int32_t topk = 1;
  ge::AttrUtils::GetInt(fusedDesc, "topk", topk);

  if (topk < 1) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Node:%s's attr topk is not valid, fusion failed.", fusedNode->GetName().c_str());
    return NOT_CHANGED;
  }

  // when topk == 1 or not need output index, no need x_index
  if (topk == 1) {
    fusedDesc->SetType("ArgMaxWithKD");
    OP_LOGI(FUSED_OP_TYPE.c_str(), "ArgMaxWithKFusionPass pass handle success!!!!");
    return SUCCESS;
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "ArgMaxWithKFusionPass pass handle success!!!!");
    return NOT_CHANGED;
  }
}

REGISTER_PASS("ArgMaxWithKFusionPass", BUILT_IN_GRAPH_PASS, ArgMaxWithKFusionPass);
}  // namespace fe
