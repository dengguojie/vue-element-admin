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
 * \file single_batch_norm_fusion_pass.cpp
 * \brief
 */
#include "single_batch_norm_fusion_pass.h"
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

static const string PATTERN_BATCHNORM = "batchNorm";
static const string PASS_OP_TYPE_BATCHNORM = "BatchNorm";
static const string IS_TRAINING = "is_training";
static const string BNREDUCE = "BNTrainingReduce";
static const string BNUPDATE = "BNTrainingUpdateV3";
static const string EPSILON = "epsilon";

vector<FusionPattern*> SingleBatchNormFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SingleBatchNormFusionPass pattern begin");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("SingleBatchNormFusionPass");

  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_BATCHNORM, {PASS_OP_TYPE_BATCHNORM}).SetOutput(PATTERN_BATCHNORM);
  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SingleBatchNormFusionPass pattern end");
  return patterns;
}

Status SingleBatchNormFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SingleBatchNormFusionPass fusion begin");
  ge::NodePtr batchNormNode = GetNodeFromMapping(PATTERN_BATCHNORM, mapping);

  FUSION_PASS_CHECK(batchNormNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "batchNorm is null, fusion failed."),
                    return PARAM_INVALID);
  // validation
  bool isTraining = false;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetBool(batchNormNode->GetOpDesc(), IS_TRAINING, isTraining),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Get is_training attr failed."), return NOT_CHANGED);

  FUSION_PASS_CHECK(!isTraining, OP_LOGI(FUSED_OP_TYPE.c_str(), "is_training is false, no need fusion."),
                    return NOT_CHANGED);
  // validation ends

  // copy Opdesc
  std::shared_ptr<ge::OpDesc> newReduceOpdesc = nullptr;
  newReduceOpdesc = std::make_shared<ge::OpDesc>(batchNormNode->GetName() + "_Reduce", BNREDUCE);

  FUSION_PASS_CHECK(newReduceOpdesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "newReduceOpdesc is null, fusion failed."), return PARAM_INVALID);

  std::shared_ptr<ge::OpDesc> newUpdateOpdesc = nullptr;
  newUpdateOpdesc = std::make_shared<ge::OpDesc>(batchNormNode->GetName() + "_UpdateV3", BNUPDATE);

  FUSION_PASS_CHECK(newUpdateOpdesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "newUpdateOpdesc is null, fusion failed."), return PARAM_INVALID);

  // add inputs for bnreduce
  ge::GeTensorDesc reduce_input_tensor1 = batchNormNode->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(newReduceOpdesc->AddInputDesc("x", reduce_input_tensor1) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input failed."), return FAILED);

  // add inputs for bnupdatev3
  ge::GeTensorDesc update_input_tensor1 = batchNormNode->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(newUpdateOpdesc->AddInputDesc("x", update_input_tensor1) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input tensor0 failed."), return FAILED);

  ge::GeTensorDesc update_input_tensor2 = batchNormNode->GetOpDesc()->GetInputDesc(1);
  FUSION_PASS_CHECK(newUpdateOpdesc->AddInputDesc("sum", update_input_tensor2) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input tensor1 failed."), return FAILED);
  FUSION_PASS_CHECK(newUpdateOpdesc->AddInputDesc("square_sum", update_input_tensor2) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input tensor2 failed."), return FAILED);
  FUSION_PASS_CHECK(newUpdateOpdesc->AddInputDesc("scale", update_input_tensor2) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input tensor3 failed."), return FAILED);
  FUSION_PASS_CHECK(newUpdateOpdesc->AddInputDesc("offset", update_input_tensor2) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input tensor4 failed."), return FAILED);

  // add output for bnreduce
  ge::GeTensorDesc reduce_tensor1 = batchNormNode->GetOpDesc()->GetOutputDesc(1);
  FUSION_PASS_CHECK(newReduceOpdesc->AddOutputDesc("sum", reduce_tensor1) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output tensor0 failed."), return FAILED);
  FUSION_PASS_CHECK(newReduceOpdesc->AddOutputDesc("square_sum", reduce_tensor1) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output tensor1 failed."), return FAILED);

  // add output for bnupdate
  ge::GeTensorDesc update_tensor1 = batchNormNode->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(newUpdateOpdesc->AddOutputDesc("y", update_tensor1) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output tensor0 failed."), return FAILED);

  ge::GeTensorDesc update_tensor2 = batchNormNode->GetOpDesc()->GetOutputDesc(1);
  FUSION_PASS_CHECK(newUpdateOpdesc->AddOutputDesc("batch_mean", update_tensor2) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output tensor1 failed."), return FAILED);
  FUSION_PASS_CHECK(newUpdateOpdesc->AddOutputDesc("batch_variance", update_tensor2) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output tensor2 failed."), return FAILED);
  FUSION_PASS_CHECK(newUpdateOpdesc->AddOutputDesc("reserve_1", update_tensor2) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output tensor3 failed."), return FAILED);
  FUSION_PASS_CHECK(newUpdateOpdesc->AddOutputDesc("reserve_2", update_tensor2) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output tensor4 failed."), return FAILED);

  // add nodes in graph
  ge::NodePtr reduceNode = graph.AddNode(newReduceOpdesc);
  ge::NodePtr updateNode = graph.AddNode(newUpdateOpdesc);
  newNodes.push_back(reduceNode);
  newNodes.push_back(updateNode);

  // copy attr
  float epsilon;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetFloat(batchNormNode->GetOpDesc(), EPSILON, epsilon),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Get epsilon attr failed."), return NOT_CHANGED);

  FUSION_PASS_CHECK(!ge::AttrUtils::SetFloat(updateNode->GetOpDesc(), EPSILON, epsilon),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Set epsilon attr failed"), return FAILED);

  // connect output edge for bnreduce
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(reduceNode->GetOutDataAnchor(0), updateNode->GetInDataAnchor(1)) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: add out data edge failed, index=[0].", reduceNode->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(reduceNode->GetOutDataAnchor(1), updateNode->GetInDataAnchor(2)) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: add out data edge failed, index=[1].", reduceNode->GetName().c_str()),
      return FAILED);

  // connect output edge for bnupdate
  string batchNormNodeName = batchNormNode->GetName();
  if (batchNormNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    for (auto inDataAnchor : batchNormNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(
          ge::GraphUtils::RemoveEdge(batchNormNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: remove out data edge failed, index=[0].", batchNormNodeName.c_str()),
          return FAILED);
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(updateNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: add out data edge failed, index=[0].", batchNormNodeName.c_str()),
          return FAILED);
    }
  }
  if (batchNormNode->GetOutDataAnchor(1)->GetPeerInDataAnchors().size() > 0) {
    for (auto inDataAnchor : batchNormNode->GetOutDataAnchor(1)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(
          ge::GraphUtils::RemoveEdge(batchNormNode->GetOutDataAnchor(1), inDataAnchor) != SUCCESS,
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: remove out data edge failed, index=[1].", batchNormNodeName.c_str()),
          return FAILED);
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(updateNode->GetOutDataAnchor(1), inDataAnchor) != SUCCESS,
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: add out data edge failed, index=[1].", batchNormNodeName.c_str()),
          return FAILED);
    }
  }

  if (batchNormNode->GetOutDataAnchor(2)->GetPeerInDataAnchors().size() > 0) {
    for (auto inDataAnchor : batchNormNode->GetOutDataAnchor(2)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(
          ge::GraphUtils::RemoveEdge(batchNormNode->GetOutDataAnchor(2), inDataAnchor) != SUCCESS,
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: remove out data edge failed, index=[2].", batchNormNodeName.c_str()),
          return FAILED);
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(updateNode->GetOutDataAnchor(2), inDataAnchor) != SUCCESS,
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: add out data edge failed, index=[2].", batchNormNodeName.c_str()),
          return FAILED);
    }
  }

  if (batchNormNode->GetOutDataAnchor(3)->GetPeerInDataAnchors().size() > 0) {
    for (auto inDataAnchor : batchNormNode->GetOutDataAnchor(3)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(
          ge::GraphUtils::RemoveEdge(batchNormNode->GetOutDataAnchor(3), inDataAnchor) != SUCCESS,
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: remove out data edge failed, index=[3].", batchNormNodeName.c_str()),
          return FAILED);
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(updateNode->GetOutDataAnchor(3), inDataAnchor) != SUCCESS,
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: add out data edge failed, index=[3].", batchNormNodeName.c_str()),
          return FAILED);
    }
  }

  if (batchNormNode->GetOutDataAnchor(4)->GetPeerInDataAnchors().size() > 0) {
    for (auto inDataAnchor : batchNormNode->GetOutDataAnchor(4)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(
          ge::GraphUtils::RemoveEdge(batchNormNode->GetOutDataAnchor(4), inDataAnchor) != SUCCESS,
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: remove out data edge failed, index=[4].", batchNormNodeName.c_str()),
          return FAILED);
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(updateNode->GetOutDataAnchor(4), inDataAnchor) != SUCCESS,
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: add out data edge failed, index=[4].", batchNormNodeName.c_str()),
          return FAILED);
    }
  }

  if (batchNormNode->GetOutControlAnchor()) {
    for (auto inControlAnchor : batchNormNode->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(batchNormNode->GetOutControlAnchor(), inControlAnchor) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: remove out control edge failed, index=[0].",
                                batchNormNodeName.c_str()),
                        return FAILED);
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(updateNode->GetOutControlAnchor(), inControlAnchor) != SUCCESS,
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: add out control edge failed, index=[0].", batchNormNodeName.c_str()),
          return FAILED);
    }
  }

  // connect input edge for bnreduce
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(batchNormNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            reduceNode->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            batchNormNode->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                            reduceNode->GetName().c_str()),
                    return FAILED);

  // connect inputs edge for bnupdate
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(batchNormNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            updateNode->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            batchNormNode->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                            updateNode->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(batchNormNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            updateNode->GetInDataAnchor(3)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            batchNormNode->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                            updateNode->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(batchNormNode->GetInDataAnchor(2)->GetPeerOutAnchor(),
                                            updateNode->GetInDataAnchor(4)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            batchNormNode->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                            updateNode->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(batchNormNode->GetOutControlAnchor(), updateNode->GetInControlAnchor()) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add control edge between node %s. and node %s failed.",
              batchNormNode->GetName().c_str(), updateNode->GetName().c_str()),
      return FAILED);

  // set grad op type to bnreduce and bnupdate
  reduceNode->GetOpDesc()->SetType(BNREDUCE);
  updateNode->GetOpDesc()->SetType(BNUPDATE);

  FUSION_PASS_CHECK(graph.RemoveNode(batchNormNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove batchNorm node failed."), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SingleBatchNormFusionPass fusion end");
  return SUCCESS;
}
REGISTER_PASS("SingleBatchNormFusion", BUILT_IN_GRAPH_PASS, SingleBatchNormFusionPass);
}  // namespace fe
