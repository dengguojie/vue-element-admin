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
 * \file single_instance_norm_fusion_pass.cpp
 * \brief
 */
#include "single_instance_norm_fusion_pass.h"
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
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

namespace fe {

static const string PATTERN_INSTANCENORM = "instanceNorm";
static const string PASS_OP_TYPE_INSTANCENORM = "InstanceNorm";
static const string INREDUCE = "INTrainingReduceV2";
static const string INUPDATE = "INTrainingUpdateV2";
static const string EPSILON = "epsilon";

vector<FusionPattern*> SingleInstanceNormFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SingleInstanceNormFusionPass pattern begin");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("SingleInstanceNormFusionPass");

  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_INSTANCENORM, {PASS_OP_TYPE_INSTANCENORM}).SetOutput(PATTERN_INSTANCENORM);
  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SingleInstanceNormFusionPass pattern end");
  return patterns;
}

Status SingleInstanceNormFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SingleInstanceNormFusionPass fusion begin");
  ge::NodePtr instanceNormNode = GetNodeFromMapping(PATTERN_INSTANCENORM, mapping);

  FUSION_PASS_CHECK(instanceNormNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "instanceNorm is null, fusion failed."),
                    return PARAM_INVALID);

  // copy Opdesc
  std::shared_ptr<ge::OpDesc> newReduceOpdesc = nullptr;
  newReduceOpdesc = std::make_shared<ge::OpDesc>(instanceNormNode->GetName() + "_ReduceV2", INREDUCE);

  FUSION_PASS_CHECK(newReduceOpdesc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "newReduceOpdesc is null, fusion failed."), return PARAM_INVALID);

  std::shared_ptr<ge::OpDesc> newUpdateOpdesc = nullptr;
  newUpdateOpdesc = std::make_shared<ge::OpDesc>(instanceNormNode->GetName() + "_UpdateV2", INUPDATE);

  FUSION_PASS_CHECK(newUpdateOpdesc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "newUpdateOpdesc is null, fusion failed."), return PARAM_INVALID);

  // add inputs for inreducev2
  ge::GeTensorDesc x_tensor_desc = instanceNormNode->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(newReduceOpdesc->AddInputDesc("x", x_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input failed."), return NOT_CHANGED);

  // add inputs for inupdatev2
  FUSION_PASS_CHECK(newUpdateOpdesc->AddInputDesc("x", x_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input tensor0 failed."), return NOT_CHANGED);

  // verify input format
  FUSION_PASS_CHECK(x_tensor_desc.GetFormat() == FORMAT_ND,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "input format not support ND."), return NOT_CHANGED);

  size_t dimCnt = x_tensor_desc.GetShape().GetDimNum();
  auto sumShape = x_tensor_desc.GetShape();
  if (x_tensor_desc.GetFormat() == FORMAT_NCHW || x_tensor_desc.GetFormat() == FORMAT_NCDHW) {
    for (int64_t i = 0; i < static_cast<int64_t>(dimCnt); i++) {
      if (i != 0 && i != 1) {
        sumShape.SetDim(i, 1);
      } else {
        continue;
      }
    }
  } else {
    for (int64_t i = 0; i < static_cast<int64_t>(dimCnt); i++) {
      if (i != 0 && i != (static_cast<int64_t>(dimCnt) - 1)) {
        sumShape.SetDim(i, 1);
      } else {
        continue;
      }
    }
  }

  ge::GeTensorDesc sum_tensor_desc = instanceNormNode->GetOpDesc()->GetInputDesc(0);
  sum_tensor_desc.SetShape(sumShape);
  sum_tensor_desc.SetOriginShape(sumShape);
  FUSION_PASS_CHECK(newUpdateOpdesc->AddInputDesc("sum", sum_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input tensor1 failed."), return NOT_CHANGED);
  FUSION_PASS_CHECK(newUpdateOpdesc->AddInputDesc("square_sum", sum_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input tensor2 failed."), return NOT_CHANGED);

  ge::GeTensorDesc gamma_beta_tensor_desc = instanceNormNode->GetOpDesc()->GetInputDesc(1);
  FUSION_PASS_CHECK(newUpdateOpdesc->AddInputDesc("gamma", gamma_beta_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input tensor3 failed."), return NOT_CHANGED);
  FUSION_PASS_CHECK(newUpdateOpdesc->AddInputDesc("beta", gamma_beta_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input tensor4 failed."), return NOT_CHANGED);

  // add output for inreducev2
  FUSION_PASS_CHECK(newReduceOpdesc->AddOutputDesc("sum", sum_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add output tensor0 failed."), return NOT_CHANGED);
  FUSION_PASS_CHECK(newReduceOpdesc->AddOutputDesc("square_sum", sum_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add output tensor1 failed."), return NOT_CHANGED);

  // add output for inupdate
  FUSION_PASS_CHECK(newUpdateOpdesc->AddOutputDesc("y", x_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add output tensor0 failed."), return NOT_CHANGED);
  FUSION_PASS_CHECK(newUpdateOpdesc->AddOutputDesc("batch_mean", sum_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add output tensor1 failed."), return NOT_CHANGED);
  FUSION_PASS_CHECK(newUpdateOpdesc->AddOutputDesc("batch_variance", sum_tensor_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add output tensor2 failed."), return NOT_CHANGED);

  // copy attr
  float epsilon;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetFloat(instanceNormNode->GetOpDesc(), EPSILON, epsilon),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Get epsilon attr failed."), return NOT_CHANGED);

  FUSION_PASS_CHECK(!ge::AttrUtils::SetFloat(newUpdateOpdesc, EPSILON, epsilon),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Set epsilon attr failed"), return NOT_CHANGED);

  FUSION_PASS_CHECK(!ge::AttrUtils::SetFloat(newUpdateOpdesc, "momentum", epsilon),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Set momentum attr failed"), return NOT_CHANGED);

  // check op supported
  FUSION_PASS_CHECK(!CheckOpSupported(newReduceOpdesc), OP_LOGI(FUSED_OP_TYPE.c_str(), "ReduceOp Not Supported."),
                    return NOT_CHANGED);

  FUSION_PASS_CHECK(!CheckOpSupported(newUpdateOpdesc), OP_LOGI(FUSED_OP_TYPE.c_str(), "UpdateOp Not Supported."),
                    return NOT_CHANGED);

  // add nodes in graph
  ge::NodePtr reduceNode = graph.AddNode(newReduceOpdesc);
  ge::NodePtr updateNode = graph.AddNode(newUpdateOpdesc);
  newNodes.push_back(reduceNode);
  newNodes.push_back(updateNode);

  // connect output edge for inreducev2
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(reduceNode->GetOutDataAnchor(0), updateNode->GetInDataAnchor(1)) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Op[%s]: add out data edge failed, index=[0].", reduceNode->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(reduceNode->GetOutDataAnchor(1), updateNode->GetInDataAnchor(2)) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Op[%s]: add out data edge failed, index=[1].", reduceNode->GetName().c_str()),
      return FAILED);

  // connect output edge for inupdate
  string instanceNormNodeName = instanceNormNode->GetName();
  if (instanceNormNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    for (auto inDataAnchor : instanceNormNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(instanceNormNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Op[%s]: remove out data edge failed, index=[0].",
                                instanceNormNodeName.c_str()),
                        return FAILED);
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(updateNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
          OP_LOGE(FUSED_OP_TYPE.c_str(), "Op[%s]: add out data edge failed, index=[0].", instanceNormNodeName.c_str()),
          return FAILED);
    }
  }
  if (instanceNormNode->GetOutDataAnchor(1)->GetPeerInDataAnchors().size() > 0) {
    for (auto inDataAnchor : instanceNormNode->GetOutDataAnchor(1)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(instanceNormNode->GetOutDataAnchor(1), inDataAnchor) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Op[%s]: remove out data edge failed, index=[1].",
                                instanceNormNodeName.c_str()),
                        return FAILED);
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(updateNode->GetOutDataAnchor(1), inDataAnchor) != SUCCESS,
          OP_LOGE(FUSED_OP_TYPE.c_str(), "Op[%s]: add out data edge failed, index=[1].", instanceNormNodeName.c_str()),
          return FAILED);
    }
  }

  if (instanceNormNode->GetOutDataAnchor(2)->GetPeerInDataAnchors().size() > 0) {
    for (auto inDataAnchor : instanceNormNode->GetOutDataAnchor(2)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(instanceNormNode->GetOutDataAnchor(2), inDataAnchor) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Op[%s]: remove out data edge failed, index=[2].",
                                instanceNormNodeName.c_str()),
                        return FAILED);
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(updateNode->GetOutDataAnchor(2), inDataAnchor) != SUCCESS,
          OP_LOGE(FUSED_OP_TYPE.c_str(), "Op[%s]: add out data edge failed, index=[2].", instanceNormNodeName.c_str()),
          return FAILED);
    }
  }

  // connect input edge for inreducev2
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(instanceNormNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            reduceNode->GetInDataAnchor(0)) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            instanceNormNode->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                            reduceNode->GetName().c_str()),
                    return FAILED);

  // connect inputs edge for inupdate
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(instanceNormNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            updateNode->GetInDataAnchor(0)) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            instanceNormNode->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                            updateNode->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(instanceNormNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            updateNode->GetInDataAnchor(3)) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            instanceNormNode->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                            updateNode->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(instanceNormNode->GetInDataAnchor(2)->GetPeerOutAnchor(),
                                            updateNode->GetInDataAnchor(4)) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            instanceNormNode->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                            updateNode->GetName().c_str()),
                    return FAILED);

  // set grad op type to inreducev2 and inupdate
  reduceNode->GetOpDesc()->SetType(INREDUCE);
  updateNode->GetOpDesc()->SetType(INUPDATE);

  FUSION_PASS_CHECK(graph.RemoveNode(instanceNormNode) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove instanceNorm node failed."), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SingleInstanceNormFusionPass fusion end");
  return SUCCESS;
}
REGISTER_PASS("SingleInstanceNormFusion", BUILT_IN_GRAPH_PASS, SingleInstanceNormFusionPass);
}  // namespace fe
