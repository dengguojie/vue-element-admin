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

/*!
 * \file one_hot_fusion_pass.cpp
 * \brief split fusion pass(one_hot --> one_hot_d)
 */
#include "one_hot_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {
static const std::string PATTERN_ONEHOT = "OneHot";
static const char* ONEHOT = "OneHot";

vector<FusionPattern*> OneHotFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  // one_hot fusion to one_hot_d
  FusionPattern* pattern = new (std::nothrow) FusionPattern("OneHotFusion");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_ONEHOT, {ONEHOT}).SetOutput(PATTERN_ONEHOT);

  patterns.push_back(pattern);

  return patterns;
}

template <typename SRCTYPE, typename DESTYPE>
static DESTYPE GetOneHotConstValue(const uint8_t* const_data) {
  DESTYPE result;
  SRCTYPE* data = (SRCTYPE*)const_data;
  result = *data;
  return result;
}

Status OneHotFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  // get one_hot node and node-desc
  ge::NodePtr one_hot_node = GetNodeFromMapping(PATTERN_ONEHOT, mapping);
  FUSION_PASS_CHECK(one_hot_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "one_hot_node is null, fusion failed."),
                    return PARAM_INVALID);

  ge::OpDescPtr one_hot_desc = one_hot_node->GetOpDesc();
  FUSION_PASS_CHECK(
      one_hot_desc == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "one_hot_node's OpDesc is null, fusion failed."),
      return PARAM_INVALID);

  std::string fusionOpType = "OneHotD";
  std::vector<PassAttrInfo> transposeAttrInfo;
  PassAttrInfo axis = {1, "depth", "SetInt"};
  transposeAttrInfo.push_back(axis);

  ge::OpDescPtr fusionDescPtr = PatternFusionUtil::GetFusionOpDesc(one_hot_node, fusionOpType, transposeAttrInfo);
  FUSION_PASS_CHECK(fusionDescPtr == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "op_type not changed to OneHotD."),
                    return NOT_CHANGED);

  // check op support
  FUSION_PASS_CHECK(!CheckOpSupported(fusionDescPtr), OP_LOGI(FUSED_OP_TYPE.c_str(), "Op OneHotD Not Supported."),
                    return NOT_CHANGED);

  ge::NodePtr fusion_node = nullptr;
  Status ret =
      PatternFusionUtil::ConstToAttrWithNode(graph, one_hot_node, fusionOpType, transposeAttrInfo, fusion_node);
  if (ret != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "OneHot has input which is not a CONST, graph not changed.");
    return NOT_CHANGED;
  }
  fusionNodes.push_back(fusion_node);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "one_hot_node fusion SUCCESSS!!!!!");

  return SUCCESS;
}

REGISTER_PASS("OneHotFusionPass", BUILT_IN_GRAPH_PASS, OneHotFusionPass);
}  // namespace fe
