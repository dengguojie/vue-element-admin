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
 * \file batchnormgrad_bninfergrad_fusion_pass.cpp
 * \brief BatchNormGrad BnInferGrad fusion pass
 */
#include <memory>
#include <string>
#include "batchnormgrad_bninfergrad_fusion_pass.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

namespace fe {
static const int32_t INT_NUM_FIVE = 5;
static const string PATTERN_BATCHNORMGRAD = "batchNormGrad";
static const string PATTERN_INPUTS1 = "input1";
static const string PATTERN_INPUTS2 = "input2";
static const string PATTERN_INPUTS3 = "input3";
static const string PATTERN_INPUTS4 = "input4";
static const string PATTERN_INPUTS5 = "input5";
static const string BNINFERGRAD = "BNInferGrad";
static const string BATCHNORMGRAD = "BatchNormGrad";
static const string EPSILON = "epsilon";
static const string IS_TRAING = "is_training";

vector<FusionPattern*> BatchNormGradBnInferGradFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchNormGradBnInferGradFusionPass pattern begin");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("BatchNormGradBnInferGradFusionPass");

  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_BATCHNORMGRAD, {BATCHNORMGRAD})
      .AddOpDesc(PATTERN_INPUTS1)
      .AddOpDesc(PATTERN_INPUTS2)
      .AddOpDesc(PATTERN_INPUTS3)
      .AddOpDesc(PATTERN_INPUTS4)
      .AddOpDesc(PATTERN_INPUTS5)
      .SetInputs(PATTERN_BATCHNORMGRAD,
                 {PATTERN_INPUTS1, PATTERN_INPUTS2, PATTERN_INPUTS3, PATTERN_INPUTS4, PATTERN_INPUTS5})
      .SetOutput(PATTERN_BATCHNORMGRAD);
  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchNormGradBnInferGradFusionPass pattern end");
  return patterns;
}

// vector<ge::NodePtr> &fusionNodes: Store fusion nodes,
//       including newly added nodes and fused but not deleted nodes
Status BatchNormGradBnInferGradFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                                  vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchNormGradBnInferGradFusionPass fusion begin");
  ge::NodePtr batchNormGradNode = GetNodeFromMapping(PATTERN_BATCHNORMGRAD, mapping);

  FUSION_PASS_CHECK(batchNormGradNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "batchNormGrad is null, fusion failed."), return PARAM_INVALID);
  bool isTraing = true;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetBool(batchNormGradNode->GetOpDesc(), IS_TRAING, isTraing),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Get is_traing attr failed."), return FAILED);

  FUSION_PASS_CHECK(isTraing, OP_LOGI(FUSED_OP_TYPE.c_str(), "is_traing is true, no need fusion."), return NOT_CHANGED);

  auto out_anchors_size = batchNormGradNode->GetAllOutDataAnchorsSize();
  if (out_anchors_size < INT_NUM_FIVE) {
    return FAILED;
  }
  FUSION_PASS_CHECK(batchNormGradNode->GetOutDataAnchor(1)->GetPeerInDataAnchors().size() != 0 ||
                        batchNormGradNode->GetOutDataAnchor(2)->GetPeerInDataAnchors().size() != 0 ||
                        batchNormGradNode->GetOutDataAnchor(3)->GetPeerInDataAnchors().size() != 0 ||
                        batchNormGradNode->GetOutDataAnchor(4)->GetPeerInDataAnchors().size() != 0,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "batchNormGrad Node intput(2-5) not null, no need fusion."),
                    return NOT_CHANGED);

  // copy Opdesc
  std::shared_ptr<ge::OpDesc> newOpdesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (newOpdesc = std::make_shared<ge::OpDesc>(batchNormGradNode->GetName(), BNINFERGRAD)),
      return FAILED);

  FUSION_PASS_CHECK(newOpdesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "newOpdesc is null, fusion failed."),
                    return PARAM_INVALID);

  // add input
  string newOpName = newOpdesc->GetName();
  ge::GeTensorDesc input_tensor1 = batchNormGradNode->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(
      newOpdesc->AddInputDesc(input_tensor1) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Op[%s]: add the input desc for the input grads failed.", newOpName.c_str()),
      return FAILED);

  ge::GeTensorDesc input_tensor3 = batchNormGradNode->GetOpDesc()->GetInputDesc(2);
  FUSION_PASS_CHECK(
      newOpdesc->AddInputDesc(input_tensor3) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Op[%s]: add the input desc for the input scale failed.", newOpName.c_str()),
      return FAILED);

  ge::GeTensorDesc input_tensor5 = batchNormGradNode->GetOpDesc()->GetInputDesc(4);
  FUSION_PASS_CHECK(newOpdesc->AddInputDesc(input_tensor5) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Op[%s]: add the input desc for the input batch_variance failed.",
                            newOpName.c_str()),
                    return FAILED);

  // add output
  ge::GeTensorDesc tensor1 = batchNormGradNode->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(newOpdesc->AddOutputDesc(tensor1) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Op[%s]: add the output desc for the output x_backprop failed.",
                            newOpName.c_str()),
                    return FAILED);

  ge::NodePtr newNode = graph.AddNode(newOpdesc);
  FUSION_PASS_CHECK(newNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "newNode is null, fusion failed."),
                    return PARAM_INVALID);

  // copy attr
  float epsilon;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetFloat(batchNormGradNode->GetOpDesc(), EPSILON, epsilon),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Get epsilon attr failed."), return FAILED);

  FUSION_PASS_CHECK(!ge::AttrUtils::SetFloat(newNode->GetOpDesc(), EPSILON, epsilon),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Set epsilon attr failed"), return FAILED);

  // copy output edge
  for (auto inDataAnchor : batchNormGradNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(batchNormGradNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(newNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
  }

  if (batchNormGradNode->GetOutControlAnchor()) {
    for (auto inControlAnchor : batchNormGradNode->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(
          ge::GraphUtils::RemoveEdge(batchNormGradNode->GetOutControlAnchor(), inControlAnchor) != SUCCESS,
          OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out control edge failed."), return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(newNode->GetOutControlAnchor(), inControlAnchor) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out control edge failed."), return FAILED);
    }
  }

  // copy BatchNormGrad inputs to BNInferGrad nodes
  auto peer_out_data_anchor = batchNormGradNode->GetInDataAnchor(0)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(peer_out_data_anchor == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "concatd_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(batchNormGradNode->GetInDataAnchor(0)->GetPeerOutAnchor(), newNode->GetInDataAnchor(0)) !=
          SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
              batchNormGradNode->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
              newNode->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(batchNormGradNode->GetInDataAnchor(2)->GetPeerOutAnchor(), newNode->GetInDataAnchor(1)) !=
          SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
              batchNormGradNode->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
              newNode->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(batchNormGradNode->GetInDataAnchor(4)->GetPeerOutAnchor(), newNode->GetInDataAnchor(2)) !=
          SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
              batchNormGradNode->GetInDataAnchor(4)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
              newNode->GetName().c_str()),
      return FAILED);

  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(batchNormGradNode->GetOutControlAnchor(), newNode->GetInControlAnchor()) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add control edge between node %s. and node %s failed.",
              batchNormGradNode->GetName().c_str(), newNode->GetName().c_str()),
      return FAILED);

  // set grad op type to BNInferGrad
  newNode->GetOpDesc()->SetType(BNINFERGRAD);

  FUSION_PASS_CHECK(graph.RemoveNode(batchNormGradNode) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove identity node failed."), return FAILED);

  fusionNodes.push_back(newNode);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchNormGradBnInferGradFusionPass fusion end");
  return SUCCESS;
}
REGISTER_PASS("BatchNormGradBnInferGradFusion", BUILT_IN_GRAPH_PASS, BatchNormGradBnInferGradFusionPass);
}  // namespace fe
