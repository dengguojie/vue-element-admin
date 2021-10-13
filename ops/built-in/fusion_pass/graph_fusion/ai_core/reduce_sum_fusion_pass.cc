/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file reduce_sum_fusion_pass.cpp
 * \brief split fusion pass(reduce_sum --> reduce_sum_d)
 */
#include "reduce_sum_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "tbe_fusion_pass_util.h"
#include "op_log.h"
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "register/op_registry.h"

using namespace ge;
namespace fe {

static const char* FUSED_NODE = "ReduceSum";

static const std::string PATTERN_FUSEDNODE = "FusedNodeReduceSum";

vector<FusionPattern*> ConstToAttrReduceSumPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("ConstToAttrReduceSumFusion");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

Status ConstToAttrReduceSumPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  std::string fusionOpType = "ReduceSumD";
  std::vector<PassAttrInfo> reduce_sumAttrInfo;
  PassAttrInfo axes = {1, "axes", "SetListInt"};
  reduce_sumAttrInfo.push_back(axes);

  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);

  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  ge::GeTensorDesc axis_input = fusedDesc->GetInputDesc(1);
  
  FUSION_PASS_CHECK(
      TbeFusionPassUtil::IsEmptyTensor(axis_input),
      OP_LOGI(FUSED_OP_TYPE.c_str(), "axis dim num is 1 and axis shape vector[0] is 0, not need fusion."),
      return NOT_CHANGED);

  FUSION_PASS_CHECK(
      fusedDesc == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.", fusedNode->GetName().c_str()),
      return PARAM_INVALID);

  ge::OpDescPtr fusionDescPtr = PatternFusionUtil::GetFusionOpDesc(fusedNode, fusionOpType, reduce_sumAttrInfo);
  FUSION_PASS_CHECK(fusionDescPtr == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "Fusion OP Desc is nullptr."),
                    return NOT_CHANGED);
  // check op support
  FUSION_PASS_CHECK(!CheckOpSupported(fusionDescPtr), OP_LOGI(FUSED_OP_TYPE.c_str(), "Op ReduceSumD Not Supported."),
                    return NOT_CHANGED);
  // PatternFusionUtil patternFusionUtil;
  if (domi::OpRegistry::Instance()->GetImplyType(fusedDesc->GetType()) == domi::ImplyType::CCE) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Fusion Op %s is CCE, no need to fusion.", fusedNode->GetName().c_str());
    return NOT_CHANGED;
  }
  ge::NodePtr fusion_node = nullptr;
  Status ret = PatternFusionUtil::ConstToAttrWithNode(graph, fusedNode, fusionOpType, reduce_sumAttrInfo, fusion_node);
  if (ret != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "ReduceSum has input which is not a CONST, graph not changed.");
    return NOT_CHANGED;
  }
  fusionNodes.push_back(fusion_node);
  return SUCCESS;
}
REGISTER_PASS("ConstToAttrReduceSumFusion", BUILT_IN_GRAPH_PASS, ConstToAttrReduceSumPass);
}  // namespace fe
