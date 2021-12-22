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
 * \file batchnorm_preprocess_fusion_pass.cc
 * \brief
 */
#include "batchnorm_preprocess_fusion_pass.h"
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

static const string PATTERN_BATCHNORM = "batchnorm";
static const string PASS_OP_TYPE_BATCHNORM = "BatchNorm";

vector<FusionPattern*> BatchNormPreprocessFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchNormPreprocessFusionPass pattern begin");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("BatchNormPreprocessFusionPass");

  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_BATCHNORM, {PASS_OP_TYPE_BATCHNORM}).SetOutput(PATTERN_BATCHNORM);
  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchNormPreprocessFusionPass pattern end");
  return patterns;
}

Status BatchNormPreprocessFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                             vector<ge::NodePtr>& new_nodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "BatchNormPreprocessFusionPass fusion begin");
  ge::NodePtr bn_node = GetNodeFromMapping(PATTERN_BATCHNORM, mapping);

  FUSION_PASS_CHECK(bn_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "batchnorm is null, fusion failed."),
                    return PARAM_INVALID);

  ge::OpDescPtr bn_desc = bn_node->GetOpDesc();
  size_t outputs_size = bn_desc->GetOutputsSize();
  // num outputs defined in op_proto, exclude reserve_space_3
  const int32_t REAL_OUTPUTS_SIZE = 5;
  if (outputs_size <= REAL_OUTPUTS_SIZE) {
    return NOT_CHANGED;
  }

  ge::OpDescPtr new_bn_desc = AttrUtils::CopyOpDesc(bn_desc);
  OpDescUtils::ClearOutputDesc(new_bn_desc, REAL_OUTPUTS_SIZE);
  ge::NodePtr new_bn_node = graph.AddNode(new_bn_desc);
  FUSION_PASS_CHECK(
    new_bn_node == nullptr,
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusion node %s is null", new_bn_desc->GetName().c_str()),
    return PARAM_INVALID);

  // add input edge to new node
  for (unsigned int i=0; i<bn_node->GetAllInDataAnchors().size(); i++) {
    if (bn_node->GetInDataAnchor(i)->GetPeerOutAnchor() != nullptr) {
      ge::GraphUtils::AddEdge(bn_node->GetInDataAnchor(i)->GetPeerOutAnchor(), new_bn_node->GetInDataAnchor(i));
    }
  }
  // add in control edge to new nodea
  if (bn_node->GetInControlAnchor() != nullptr) {
    for (unsigned int i = 0; i < bn_node->GetInControlAnchor()->GetPeerOutControlAnchors().size(); i++) {
      ge::GraphUtils::AddEdge(bn_node->GetInControlAnchor()->GetPeerOutControlAnchors().at(i),
                              new_bn_node->GetInControlAnchor());
    }
  }

  // add output edge from new node
  for (unsigned int i=0; i<REAL_OUTPUTS_SIZE; i++) {
    auto anchor_out = bn_node->GetOutDataAnchor(i);
    for(InDataAnchorPtr anchor_out_in : anchor_out->GetPeerInDataAnchors()) {
      anchor_out_in->UnlinkAll();
      ge::GraphUtils::AddEdge(new_bn_node->GetOutDataAnchor(i), anchor_out_in);
    }
  }

  // add out control edge to new node
  if (bn_node->GetOutControlAnchor() != nullptr) {
    for (unsigned int i = 0; i < bn_node->GetOutControlAnchor()->GetPeerInControlAnchors().size(); i++) {
      ge::GraphUtils::AddEdge(new_bn_node->GetOutControlAnchor(),
                              bn_node->GetOutControlAnchor()->GetPeerInControlAnchors().at(i));
    }
  }

  OpDescPtr const_op_desc = std::make_shared<ge::OpDesc>("Constant", "Constant");
  
  vector<int64_t> dims{1};
  GeShape shape(dims);
  GeTensorDesc const_output_desc(shape, FORMAT_ND, DT_FLOAT);
  const_output_desc.SetOriginShape(shape);
  const_output_desc.SetOriginFormat(FORMAT_ND);
  const_op_desc->AddOutputDesc("y", const_output_desc);

  std::vector<int64_t> values {0};
  GeTensorDesc data_desc(GeShape({1}), FORMAT_ND, DT_FLOAT);
  ge::GeTensorPtr const_value = std::make_shared<ge::GeTensor>(data_desc, reinterpret_cast<uint8_t *>(values.data()),
                                                               sizeof(float)*values.size());
  AttrUtils::SetTensor(const_op_desc, "value", const_value);

  const_op_desc->SetType("Constant");
  NodePtr const_node = graph.AddNode(const_op_desc);

  // unlink old add a new edge form const node to next node
  auto anchor_last_out = bn_node->GetOutDataAnchor(REAL_OUTPUTS_SIZE);
  auto const_node_out_anchor = const_node->GetOutDataAnchor(0);
  for (auto anchor_out_in : anchor_last_out->GetPeerInDataAnchors()) {
    anchor_out_in->UnlinkAll();
    ge::GraphUtils::AddEdge(const_node_out_anchor, anchor_out_in);
  }

  graph.RemoveNode(bn_node);
  new_nodes.push_back(new_bn_node);
  new_nodes.push_back(const_node);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "BatchNormPreprocessFusionPass fusion end");
  return SUCCESS;
}
REGISTER_PASS("BatchNormPreprocessFusionPass", BUILT_IN_GRAPH_PASS, BatchNormPreprocessFusionPass);
}  // namespace fe
