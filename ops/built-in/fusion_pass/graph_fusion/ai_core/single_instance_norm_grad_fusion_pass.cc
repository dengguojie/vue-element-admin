/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file single_instance_norm_grad_fusion_pass.cpp
 * \brief
 */
#include "single_instance_norm_grad_fusion_pass.h"
#include <iostream>
#include <vector>
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
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

static const string PATTERN_INSTANCENORMGRAD = "instanceNormGrad";
static const string PASS_OP_TYPE_INSTANCENORMGRAD = "InstanceNormGrad";
static const string INTUPDATEGRAD = "INTrainingUpdateGrad";
static const string INTREDUCEGRAD = "INTrainingReduceGrad";
static const string INTUPDATEGRADGAMMABETA = "INTrainingUpdateGradGammaBeta";

vector<FusionPattern*> SingleInstanceNormGradFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SingleInstanceNormGradFusionPass pattern begin");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("SingleInstanceNormGradFusionPass");

  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_INSTANCENORMGRAD, {PASS_OP_TYPE_INSTANCENORMGRAD}).SetOutput(PATTERN_INSTANCENORMGRAD);
  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SingleInstanceNormGradFusionPass pattern end");
  return patterns;
}

Status SingleInstanceNormGradFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                                vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SingleInstanceNormGradFusionPass fusion begin");
  ge::NodePtr instanceNormGradNode = GetNodeFromMapping(PATTERN_INSTANCENORMGRAD, mapping);

  FUSION_PASS_CHECK(instanceNormGradNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "instanceNormGrad is null, fusion failed."), return PARAM_INVALID);

  // copy Opdesc
  std::shared_ptr<ge::OpDesc> newINTrainingUpdateGradOpdesc = nullptr;
  newINTrainingUpdateGradOpdesc =
      std::make_shared<ge::OpDesc>(instanceNormGradNode->GetName() + "_INTrainingUpdateGrad", INTUPDATEGRAD);
  FUSION_PASS_CHECK(newINTrainingUpdateGradOpdesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "newINTrainingUpdateGradOpdesc is null, fusion failed."),
                    return PARAM_INVALID);

  std::shared_ptr<ge::OpDesc> newINTrainingReduceGradOpdesc = nullptr;
  newINTrainingReduceGradOpdesc =
      std::make_shared<ge::OpDesc>(instanceNormGradNode->GetName() + "_INTrainingReduceGrad", INTREDUCEGRAD);
  FUSION_PASS_CHECK(newINTrainingReduceGradOpdesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "newINTrainingReduceGradOpdesc is null, fusion failed."),
                    return PARAM_INVALID);

  std::shared_ptr<ge::OpDesc> newINTrainingUpdateGradGammaBetaOpdesc = nullptr;
  newINTrainingUpdateGradGammaBetaOpdesc = std::make_shared<ge::OpDesc>(
      instanceNormGradNode->GetName() + "_INTrainingUpdateGradGammaBeta", INTUPDATEGRADGAMMABETA);
  FUSION_PASS_CHECK(newINTrainingUpdateGradGammaBetaOpdesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "newINTrainingUpdateGradGammaBetaOpdesc is null, fusion failed."),
                    return PARAM_INVALID);

  // add inputs for INTrainingUpdateGrad
  ge::GeTensorDesc dy_tensor_desc = instanceNormGradNode->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc x_tensor_desc = instanceNormGradNode->GetOpDesc()->GetInputDesc(1);
  ge::GeTensorDesc var_tensor_desc = instanceNormGradNode->GetOpDesc()->GetInputDesc(2);
  ge::GeTensorDesc mean_tensor_desc = instanceNormGradNode->GetOpDesc()->GetInputDesc(3);
  ge::GeTensorDesc gamma_tensor_desc = instanceNormGradNode->GetOpDesc()->GetInputDesc(4);

  FUSION_PASS_CHECK(newINTrainingUpdateGradOpdesc->AddInputDesc("dy", dy_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input dy failed."), return NOT_CHANGED);

  FUSION_PASS_CHECK(newINTrainingUpdateGradOpdesc->AddInputDesc("x", x_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input x failed."), return NOT_CHANGED);

  FUSION_PASS_CHECK(newINTrainingUpdateGradOpdesc->AddInputDesc("variance", var_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input variance failed."), return NOT_CHANGED);

  FUSION_PASS_CHECK(newINTrainingUpdateGradOpdesc->AddInputDesc("mean", mean_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input mean failed."), return NOT_CHANGED);

  // add inputs for INTrainingReduceGrad
  FUSION_PASS_CHECK(newINTrainingReduceGradOpdesc->AddInputDesc("dy", dy_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input dy failed."), return NOT_CHANGED);

  FUSION_PASS_CHECK(newINTrainingReduceGradOpdesc->AddInputDesc("x", x_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input x failed."), return NOT_CHANGED);

  FUSION_PASS_CHECK(newINTrainingReduceGradOpdesc->AddInputDesc("variance", var_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input variance failed."), return NOT_CHANGED);

  FUSION_PASS_CHECK(newINTrainingReduceGradOpdesc->AddInputDesc("mean", mean_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input mean failed."), return NOT_CHANGED);

  FUSION_PASS_CHECK(newINTrainingReduceGradOpdesc->AddInputDesc("res_gamma", var_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input res_gamma failed."), return NOT_CHANGED);

  FUSION_PASS_CHECK(newINTrainingReduceGradOpdesc->AddInputDesc("res_beta", var_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input res_beta failed."), return NOT_CHANGED);

  FUSION_PASS_CHECK(newINTrainingReduceGradOpdesc->AddInputDesc("gamma", gamma_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input gamma failed."), return NOT_CHANGED);

  // add inputs for INTrainingUpdateGradGammaBeta
  FUSION_PASS_CHECK(newINTrainingUpdateGradGammaBetaOpdesc->AddInputDesc("res_gamma", var_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input res_gamma failed."), return NOT_CHANGED);

  FUSION_PASS_CHECK(newINTrainingUpdateGradGammaBetaOpdesc->AddInputDesc("res_beta", var_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input res_beta failed."), return NOT_CHANGED);

  // add output for INTrainingUpdateGrad
  FUSION_PASS_CHECK(newINTrainingUpdateGradOpdesc->AddOutputDesc("res_gamma", var_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add output res_gamma failed."), return NOT_CHANGED);

  FUSION_PASS_CHECK(newINTrainingUpdateGradOpdesc->AddOutputDesc("res_beta", var_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add output res_beta failed."), return NOT_CHANGED);

  // add output for INTrainingReduceGrad
  FUSION_PASS_CHECK(newINTrainingReduceGradOpdesc->AddOutputDesc("pd_x", x_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add output pd_x failed."), return NOT_CHANGED);

  // add output for INTrainingUpdateGradGammaBeta
  FUSION_PASS_CHECK(newINTrainingUpdateGradGammaBetaOpdesc->AddOutputDesc("pd_gamma", gamma_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add output pd_gamma failed."), return NOT_CHANGED);

  FUSION_PASS_CHECK(newINTrainingUpdateGradGammaBetaOpdesc->AddOutputDesc("pd_beta", gamma_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add output pd_beta failed."), return NOT_CHANGED);

  // check op supported
  FUSION_PASS_CHECK(!CheckOpSupported(newINTrainingUpdateGradOpdesc),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "INTrainingUpdateGrad Not Supported."), return NOT_CHANGED);

  FUSION_PASS_CHECK(!CheckOpSupported(newINTrainingReduceGradOpdesc),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "INTrainingReduceGrad Not Supported."), return NOT_CHANGED);

  FUSION_PASS_CHECK(!CheckOpSupported(newINTrainingUpdateGradGammaBetaOpdesc),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "INTrainingUpdateGradGammaBeta Not Supported."), return NOT_CHANGED);

  // add nodes in graph
  ge::NodePtr INTrainingUpdateGradNode = graph.AddNode(newINTrainingUpdateGradOpdesc);
  ge::NodePtr INTrainingReduceGradNode = graph.AddNode(newINTrainingReduceGradOpdesc);
  ge::NodePtr INTrainingUpdateGradGammaBetaNode = graph.AddNode(newINTrainingUpdateGradGammaBetaOpdesc);
  newNodes.push_back(INTrainingUpdateGradNode);
  newNodes.push_back(INTrainingReduceGradNode);
  newNodes.push_back(INTrainingUpdateGradGammaBetaNode);

  // connect output edge for INTrainingReduceGrad
  string instanceNormGradNodeName = instanceNormGradNode->GetName();
  if (instanceNormGradNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    for (auto inDataAnchor : instanceNormGradNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(instanceNormGradNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: remove out data edge failed, index=[0].",
                                instanceNormGradNodeName.c_str()),
                        return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(INTrainingReduceGradNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: add out data edge failed, index=[0].",
                                instanceNormGradNodeName.c_str()),
                        return FAILED);
    }
  }

  // connect output edge for INTrainingUpdateGradGammaBeta
  if (instanceNormGradNode->GetOutDataAnchor(1)->GetPeerInDataAnchors().size() > 0) {
    for (auto inDataAnchor : instanceNormGradNode->GetOutDataAnchor(1)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(instanceNormGradNode->GetOutDataAnchor(1), inDataAnchor) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: remove out data edge failed, index=[1].",
                                instanceNormGradNodeName.c_str()),
                        return FAILED);
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(INTrainingUpdateGradGammaBetaNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: add out data edge failed, index=[0].",
                  instanceNormGradNodeName.c_str()),
          return FAILED);
    }
  }

  if (instanceNormGradNode->GetOutDataAnchor(2)->GetPeerInDataAnchors().size() > 0) {
    for (auto inDataAnchor : instanceNormGradNode->GetOutDataAnchor(2)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(instanceNormGradNode->GetOutDataAnchor(2), inDataAnchor) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: remove out data edge failed, index=[2].",
                                instanceNormGradNodeName.c_str()),
                        return FAILED);
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(INTrainingUpdateGradGammaBetaNode->GetOutDataAnchor(1), inDataAnchor) != SUCCESS,
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: add out data edge failed, index=[1].",
                  instanceNormGradNodeName.c_str()),
          return FAILED);
    }
  }

  // connect inputs edge for INTrainingUpdateGrad
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(instanceNormGradNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                              INTrainingUpdateGradNode->GetInDataAnchor(0)) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
              instanceNormGradNode->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
              INTrainingUpdateGradNode->GetName().c_str()),
      return FAILED);

  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(instanceNormGradNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
                              INTrainingUpdateGradNode->GetInDataAnchor(1)) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
              instanceNormGradNode->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
              INTrainingUpdateGradNode->GetName().c_str()),
      return FAILED);

  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(instanceNormGradNode->GetInDataAnchor(2)->GetPeerOutAnchor(),
                              INTrainingUpdateGradNode->GetInDataAnchor(2)) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
              instanceNormGradNode->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
              INTrainingUpdateGradNode->GetName().c_str()),
      return FAILED);

  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(instanceNormGradNode->GetInDataAnchor(3)->GetPeerOutAnchor(),
                              INTrainingUpdateGradNode->GetInDataAnchor(3)) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
              instanceNormGradNode->GetInDataAnchor(3)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
              INTrainingUpdateGradNode->GetName().c_str()),
      return FAILED);

  // connect inputs edge for INTrainingReduceGrad
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(instanceNormGradNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                              INTrainingReduceGradNode->GetInDataAnchor(0)) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
              instanceNormGradNode->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
              INTrainingReduceGradNode->GetName().c_str()),
      return FAILED);

  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(instanceNormGradNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
                              INTrainingReduceGradNode->GetInDataAnchor(1)) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
              instanceNormGradNode->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
              INTrainingReduceGradNode->GetName().c_str()),
      return FAILED);

  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(instanceNormGradNode->GetInDataAnchor(2)->GetPeerOutAnchor(),
                              INTrainingReduceGradNode->GetInDataAnchor(2)) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
              instanceNormGradNode->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
              INTrainingReduceGradNode->GetName().c_str()),
      return FAILED);

  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(instanceNormGradNode->GetInDataAnchor(3)->GetPeerOutAnchor(),
                              INTrainingReduceGradNode->GetInDataAnchor(3)) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
              instanceNormGradNode->GetInDataAnchor(3)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
              INTrainingReduceGradNode->GetName().c_str()),
      return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(INTrainingUpdateGradNode->GetOutDataAnchor(0),
                                            INTrainingReduceGradNode->GetInDataAnchor(4)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: add out data edge failed, index=[0].",
                            INTrainingUpdateGradNode->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(INTrainingUpdateGradNode->GetOutDataAnchor(1),
                                            INTrainingReduceGradNode->GetInDataAnchor(5)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: add out data edge failed, index=[1].",
                            INTrainingUpdateGradNode->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(instanceNormGradNode->GetInDataAnchor(4)->GetPeerOutAnchor(),
                              INTrainingReduceGradNode->GetInDataAnchor(6)) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
              instanceNormGradNode->GetInDataAnchor(4)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
              INTrainingReduceGradNode->GetName().c_str()),
      return FAILED);

  // connect inputs edge for INTrainingUpdateGradGammaBeta
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(INTrainingUpdateGradNode->GetOutDataAnchor(0),
                                            INTrainingUpdateGradGammaBetaNode->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: add out data edge failed, index=[0].",
                            INTrainingUpdateGradNode->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(INTrainingUpdateGradNode->GetOutDataAnchor(1),
                                            INTrainingUpdateGradGammaBetaNode->GetInDataAnchor(1)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: add out data edge failed, index=[1].",
                            INTrainingUpdateGradNode->GetName().c_str()),
                    return FAILED);

  // set grad op type to INTrainingUpdateGrad, INTrainingReduceGrad and INTrainingUpdateGradGammaBeta
  INTrainingUpdateGradNode->GetOpDesc()->SetType(INTUPDATEGRAD);
  INTrainingReduceGradNode->GetOpDesc()->SetType(INTREDUCEGRAD);
  INTrainingUpdateGradGammaBetaNode->GetOpDesc()->SetType(INTUPDATEGRADGAMMABETA);

  FUSION_PASS_CHECK(graph.RemoveNode(instanceNormGradNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove instanceNormGrad node failed."), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SingleInstanceNormGradFusionPass fusion end");
  return SUCCESS;
}
REGISTER_PASS("SingleInstanceNormGradFusion", BUILT_IN_GRAPH_PASS, SingleInstanceNormGradFusionPass);
}  // namespace fe
