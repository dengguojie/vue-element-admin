/* Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file batchnormgrad_preprocess_fusion_pass.cc
 * \brief
 */
#include "batchnormgrad_preprocess_fusion_pass.h"
#include <vector>
#include <memory>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

namespace fe {

static const string PATTERN_BATCHNORMGRAD = "batchnormgrad";
static const string PASS_OP_TYPE_BATCHNORMGRAD = "BatchNormGrad";

vector<FusionPattern*> BatchNormGradPreprocessFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchNormGradPreprocessFusionPass pattern begin");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("BatchNormGradPreprocessFusionPass");

  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_BATCHNORMGRAD, {PASS_OP_TYPE_BATCHNORMGRAD}).SetOutput(PATTERN_BATCHNORMGRAD);
  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchNormGradPreprocessFusionPass pattern end");
  return patterns;
}

Status BatchNormGradPreprocessFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                                 vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchNormGradPreprocessFusionPass fusion begin");
  ge::NodePtr batchNormNode = GetNodeFromMapping(PATTERN_BATCHNORMGRAD, mapping);

  FUSION_PASS_CHECK(batchNormNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "batchNorm is null, fusion failed."),
                    return PARAM_INVALID);

  ge::OpDescPtr bn_desc = batchNormNode->GetOpDesc();
  size_t inputs_size = bn_desc->GetInputsSize();

  if (inputs_size <= 5) {
    return NOT_CHANGED;
  }

  // remove reserve_space_3 input
  auto in_anchor = batchNormNode->GetInDataAnchor(5);
  ge::GraphUtils::RemoveEdge(in_anchor->GetPeerOutAnchor(), in_anchor);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchNormGradPreprocessFusionPass fusion end");
  return SUCCESS;
}
REGISTER_PASS("BatchNormGradPreprocessFusionPass", BUILT_IN_GRAPH_PASS, BatchNormGradPreprocessFusionPass);
}  // namespace fe
