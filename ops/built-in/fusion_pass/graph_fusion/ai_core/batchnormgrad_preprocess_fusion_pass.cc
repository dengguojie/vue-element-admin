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
                                                 vector<ge::NodePtr>& new_nodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "BatchNormGradPreprocessFusionPass fusion begin");
  ge::NodePtr bn_grad_node = GetNodeFromMapping(PATTERN_BATCHNORMGRAD, mapping);

  FUSION_PASS_CHECK(bn_grad_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "batchNorm is null, fusion failed."),
                    return PARAM_INVALID);

  ge::OpDescPtr bn_grad_desc = bn_grad_node->GetOpDesc();
  // GetAllInputsSize contains optional_input, GetInputsSie not
  size_t inputs_size = bn_grad_desc->GetAllInputsSize();
  size_t outputs_size = bn_grad_desc->GetOutputsSize();
  // num inputs defined in op_proto, exclude optional inputs
  const int32_t REAL_INPUTS_SIZE = 5;
  if (inputs_size <= REAL_INPUTS_SIZE) {
    return NOT_CHANGED;
  }

  ge::OpDescPtr new_bn_grad_desc = AttrUtils::CopyOpDesc(bn_grad_desc);
  OpDescUtils::ClearInputDesc(new_bn_grad_desc, REAL_INPUTS_SIZE);
  ge::NodePtr new_bn_grad_node = graph.AddNode(new_bn_grad_desc);
  FUSION_PASS_CHECK(
    new_bn_grad_node == nullptr,
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusion node %s is null",
                                   new_bn_grad_desc->GetName().c_str()),
    return PARAM_INVALID);

  // add input edge to new node
  for(int32_t i = 0; i < REAL_INPUTS_SIZE; i++) {
    ge::GraphUtils::AddEdge(bn_grad_node->GetInDataAnchor(i)->GetPeerOutAnchor(), new_bn_grad_node->GetInDataAnchor(i));
  }

  // add in control edge to new nodea
  if (bn_grad_node->GetInControlAnchor() != nullptr) {
    for (unsigned int i = 0; i < bn_grad_node->GetInControlAnchor()->GetPeerOutControlAnchors().size(); i++) {
      ge::GraphUtils::AddEdge(bn_grad_node->GetInControlAnchor()->GetPeerOutControlAnchors().at(i),
                              new_bn_grad_node->GetInControlAnchor());
    }
  }

  // add output edge from new node
  for(size_t i = 0; i < outputs_size; i++) {
    auto anchor_out = bn_grad_node->GetOutDataAnchor(i);
    for (InDataAnchorPtr anchor_out_in : anchor_out->GetPeerInDataAnchors()) {
      anchor_out_in ->UnlinkAll();
      ge::GraphUtils::AddEdge(new_bn_grad_node->GetOutDataAnchor(i), anchor_out_in);
    }
  }

  // add out control edge to new node
  if (bn_grad_node->GetOutControlAnchor() != nullptr) {
    for (unsigned int i = 0; i < bn_grad_node->GetOutControlAnchor()->GetPeerInControlAnchors().size(); i++) {
      ge::GraphUtils::AddEdge(new_bn_grad_node->GetOutControlAnchor(),
                              bn_grad_node->GetOutControlAnchor()->GetPeerInControlAnchors().at(i));
    }
  }

  graph.RemoveNode(bn_grad_node);
  new_nodes.push_back(new_bn_grad_node);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "BatchNormGradPreprocessFusionPass fusion end");
  return SUCCESS;
}
REGISTER_PASS("BatchNormGradPreprocessFusionPass", BUILT_IN_GRAPH_PASS, BatchNormGradPreprocessFusionPass);
}  // namespace fe
