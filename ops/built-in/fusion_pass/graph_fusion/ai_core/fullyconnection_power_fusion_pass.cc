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
 * \file fullyconnection_power_fusion_pass.cpp
 * \brief
 */
#include "fullyconnection_power_fusion_pass.h"

#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

#include "anchor_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "securec.h"


namespace fe {
static const char PATTERN_POWER[] = "Power";
static const char PATTERN_FULLYCONNECTION[] = "FullyConnection";
static const int BIAS_INDEX = 2;
static const char FUSED_OP_TYPE[] = "FullyConnection";
static const char CONSTANTOPTAB[] = "Const";

static const std::string FC_POWER_OP_INPUT = "fc_power_input";
static const std::string FC_POWER_OP_OUTPUT = "fc_power_output";
static const std::string POWER_SHIFT = "shift";
static const int FC_BIAS_INDEX = 2;
static const int FC_POWER_HOST_OP_BIAS_INDEX = 0;
static const int kInputNumTwo = 2;
static const std::string FullyConnectionPowerPassHostOp = "FullyConnectionPowerPassHostOp";

uint64_t GetHostCpuAtomicId() {
  static std::atomic<uint64_t> globalTransAtomicId(0);
  return globalTransAtomicId.fetch_add(1, std::memory_order_relaxed);
}

Status CreateFullyPowerPassHostOp(const string &opType, const ge::NodePtr &fcNode, ge::ComputeGraph &graph,
                                  vector<ge::NodePtr> &newNodes, const float &shift) {
    std::stringstream opNameTemp;
    // the atomic id of trans nodes must be unique.(start from 0)
    opNameTemp << opType << "_" << GetHostCpuAtomicId();
    ge::OpDescPtr fcPowerHostOp = nullptr;
    FUSION_PASS_MAKE_SHARED((fcPowerHostOp = std::make_shared<ge::OpDesc>(opNameTemp.str(), opType)), return FAILED);
    FUSION_PASS_CHECK(fcPowerHostOp == nullptr, OP_LOGW(fcNode, "create new host op failed"), return fe::NOT_CHANGED);
    // add input and output desc of new host op
    vector<ge::GeTensorPtr> weights = ge::OpDescUtils::MutableWeights(fcNode);
    FUSION_PASS_CHECK(weights.size() < 2,
                      OP_LOGW(fcNode, "fc weights get failed, weights size can not be less than 2, current size is %u.",
                              weights.size()),
                      return fe::NOT_CHANGED);
    ge::GeTensorPtr biasTensorPtr = weights[1];
    ge::GeTensorDesc biasTensorDesc = biasTensorPtr->GetTensorDesc();

    FUSION_PASS_CHECK(fcPowerHostOp->AddInputDesc(FC_POWER_OP_INPUT, biasTensorDesc) != GRAPH_SUCCESS,
                      OP_LOGW(fcNode, "failed to add input desc to fcPowerHostOp."), return fe::NOT_CHANGED);
    FUSION_PASS_CHECK(fcPowerHostOp->MutableInputDesc(0) == nullptr, OP_LOGW(fcNode, "get input desc 0 failed"),
                      return fe::NOT_CHANGED);
    fcPowerHostOp->MutableInputDesc(0)->SetOriginDataType(biasTensorDesc.GetDataType());
    fcPowerHostOp->MutableInputDesc(0)->SetOriginFormat(
        static_cast<ge::Format>(ge::GetPrimaryFormat(biasTensorDesc.GetFormat())));
    fcPowerHostOp->MutableInputDesc(0)->SetOriginShape(biasTensorDesc.GetShape());

    FUSION_PASS_CHECK(fcPowerHostOp->AddOutputDesc(FC_POWER_OP_OUTPUT, biasTensorDesc) != GRAPH_SUCCESS,
                      OP_LOGW(fcNode, "failed to add output desc to fcPowerHostOp."), return fe::NOT_CHANGED);
    FUSION_PASS_CHECK(fcPowerHostOp->MutableOutputDesc(0) == nullptr, OP_LOGW(fcNode, "get output desc 0 failed"),
                      return fe::NOT_CHANGED);
    fcPowerHostOp->MutableOutputDesc(0)->SetOriginFormat(
        static_cast<ge::Format>(ge::GetPrimaryFormat(biasTensorDesc.GetFormat())));
    fcPowerHostOp->MutableOutputDesc(0)->SetOriginShape(biasTensorDesc.GetShape());
    fcPowerHostOp->MutableOutputDesc(0)->SetDataType(biasTensorDesc.GetDataType());
    fcPowerHostOp->MutableOutputDesc(0)->SetShape(biasTensorDesc.GetShape());

    auto fcPowerNode = graph.AddNode(fcPowerHostOp);
    FUSION_PASS_CHECK(fcPowerNode == nullptr, OP_LOGW(fcNode, "add new host op to graph failed"),
                      return fe::NOT_CHANGED);
    newNodes.emplace_back(fcPowerNode);

    // Add edges between fc bias <--> new_host_cpu_op:0
    ge::InDataAnchorPtr fcInputAnchor = fcNode->GetInDataAnchor(FC_BIAS_INDEX);
    FUSION_PASS_CHECK(fcInputAnchor == nullptr, OP_LOGW(fcNode, "fc get const anchor failed"), return fe::NOT_CHANGED);
    ge::OutDataAnchorPtr fcBiasPeerOutAnchor = fcInputAnchor->GetPeerOutAnchor();
    FUSION_PASS_CHECK(fcBiasPeerOutAnchor == nullptr, OP_LOGW(fcNode, "fc get const failed"), return fe::NOT_CHANGED);
    auto fcPowerHostOpInputAnchor = fcPowerNode->GetInDataAnchor(FC_POWER_HOST_OP_BIAS_INDEX);
    if (ge::GraphUtils::AddEdge(fcBiasPeerOutAnchor, fcPowerHostOpInputAnchor) != ge::GRAPH_SUCCESS) {
      OP_LOGW(fcNode, "Add Edge between const and new host op failed.");
      return FAILED;
    }
    if (ge::GraphUtils::RemoveEdge(fcBiasPeerOutAnchor, fcInputAnchor) != ge::GRAPH_SUCCESS) {
      OP_LOGW(fcNode, "Remove Edge between FC and bias const op failed.");
      return FAILED;
    }

    auto fcPowerHostOpOutputAnchor = fcPowerNode->GetOutDataAnchor(0);
    if (ge::GraphUtils::AddEdge(fcPowerHostOpOutputAnchor, fcInputAnchor) != ge::GRAPH_SUCCESS) {
      OP_LOGW(fcNode, "Add Edge between FC and new host op failed.");
      return FAILED;
    }

    if (!ge::AttrUtils::SetFloat(fcPowerNode->GetOpDesc(), POWER_SHIFT, shift)) {
      OP_LOGW(fcNode, "set float failed.");
      return FAILED;
    }
    return SUCCESS;
}

/*
            Data(input)     shift
                \            \
                 \            \
                  v            v
      Const(filter)--->FC----->Power(caffe)------>output
                 ^              ^       ^
                /              /       /
               /              /       /
            Const(bias)     power  scale
    */

vector<FusionPattern*> FullyConnectionPowerPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern1 = new (std::nothrow) FusionPattern("FullyConnectionPowerPass");
  FUSION_PASS_CHECK(pattern1 == nullptr, OP_LOGW(FUSED_OP_TYPE, "new an object failed"), return patterns);

  pattern1->AddOpDesc(PATTERN_FULLYCONNECTION, {"FullyConnection"})
      .AddOpDesc(PATTERN_POWER, {"Power"})
      .SetInputs(PATTERN_POWER, {PATTERN_FULLYCONNECTION})
      .SetOutput(PATTERN_POWER);
  patterns.push_back(pattern1);
  return patterns;
}

Status FullyConnectionPowerPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr powerNode = GetNodeFromMapping(PATTERN_POWER, mapping);
  ge::NodePtr fullyConnectionNode = GetNodeFromMapping(PATTERN_FULLYCONNECTION, mapping);

  FUSION_PASS_CHECK(powerNode == nullptr, OP_LOGW(FUSED_OP_TYPE, "powerNode is null"), return fe::NOT_CHANGED);
  FUSION_PASS_CHECK(fullyConnectionNode == nullptr, OP_LOGW(FUSED_OP_TYPE, "fullyConnectionNode is null"),
                    return fe::NOT_CHANGED);
  FUSION_PASS_CHECK(fullyConnectionNode->GetOutDataNodes().size() > 1,
                    OP_LOGW(fullyConnectionNode, "output data nodes can not greater than 1."), return fe::NOT_CHANGED);
  ge::ConstGeTensorDescPtr fullyConnectionNodeDescPtr = GetCurrNodeInputDesc(fullyConnectionNode, 0);
  FUSION_PASS_CHECK(fullyConnectionNodeDescPtr == nullptr,
                    OP_LOGW(fullyConnectionNode, "fcNode's OpDesc is null, fusion failed."), return fe::NOT_CHANGED);
  ge::DataType inputType = fullyConnectionNodeDescPtr->GetDataType();
  if (inputType == ge::DT_INT8 || inputType == ge::DT_UINT8) {
    OP_LOGI(fullyConnectionNode, "fc dataType is not float16 , graph not changed.");
    return NOT_CHANGED;
  }

  string fullyConnectionNodeName = fullyConnectionNode->GetName();
  // Check fc's output is power node
  Status ret = PatternFusionUtil::LinkControlEdge(powerNode, fullyConnectionNode);

  FUSION_PASS_CHECK(
      ret != SUCCESS,
      OP_LOGW(fullyConnectionNode, "FcNode[%s]: LinkControlEdge not success.", fullyConnectionNodeName.c_str()),
      return fe::NOT_CHANGED);

  ge::OpDescPtr powerDesc = powerNode->GetOpDesc();
  FUSION_PASS_CHECK(powerDesc == nullptr, OP_LOGW(powerNode, "powerNode's OpDesc is null, fusion failed."),
                    return fe::NOT_CHANGED);

  float power = -1;
  float scale = -1;
  float shift = -1;
  int64_t num_output = 0;

  ge::AttrUtils::GetFloat(powerNode->GetOpDesc(), "power", power);
  ge::AttrUtils::GetFloat(powerNode->GetOpDesc(), "scale", scale);
  ge::AttrUtils::GetFloat(powerNode->GetOpDesc(), "shift", shift);

  ge::AttrUtils::GetInt(fullyConnectionNode->GetOpDesc(), "num_output", num_output);

  if (std::fabs(power - 1.0F) > std::numeric_limits<float>::epsilon() ||
      std::fabs(scale - 1.0F) > std::numeric_limits<float>::epsilon()) {
    OP_LOGI(powerNode, "fc and power nodes is not meet the conditions , graph not changed.");
    return NOT_CHANGED;
  }

  // Get size for fc, check it
  int64_t inputs_num = fullyConnectionNode->GetOpDesc()->GetInputsSize();
  // create bias node
  bool hasBias = true;
  if (inputs_num == kInputNumTwo) {
    AddBiasNode(graph, fullyConnectionNode, shift, num_output, fusionNodes);
    hasBias = false;
  }

  if (hasBias) {
    Tensor data;
    Operator op = ge::OpDescUtils::CreateOperatorFromNode(fullyConnectionNode);
    if (GRAPH_SUCCESS != op.GetInputConstData("b", data)) {
      OP_LOGI(fullyConnectionNode, "GetInputConstData of bias failed.");
      return NOT_CHANGED;
    }
    /* Create Host Cpu Op */
    vector<ge::NodePtr> fusion_nodes;
    ret = CreateFullyPowerPassHostOp(FullyConnectionPowerPassHostOp, fullyConnectionNode, graph,
        fusion_nodes, shift);
    if (ret != SUCCESS || fusion_nodes.empty()) {
      OP_LOGW(fullyConnectionNode, "Create host cpu op for fc node %s failed", fullyConnectionNode->GetName().c_str());
      return ret;
    }
  }

  auto outDataAnchor0 = powerNode->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(outDataAnchor0 == nullptr, OP_LOGW(powerNode, "get out data anchor of 0"), return fe::NOT_CHANGED);
  for (auto inDataAnchor : outDataAnchor0->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(outDataAnchor0, inDataAnchor) != SUCCESS,
                      OP_LOGW(powerNode, "Remove power and outnode edge failed."), return fe::NOT_CHANGED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(GetPeerOutAnchorWithInDataAnchor(powerNode, 0), inDataAnchor) != SUCCESS,
                      OP_LOGW(powerNode, "Add innode and outnode edge failed."), return fe::NOT_CHANGED);
  }

  return SUCCESS;
}

Status FullyConnectionPowerPass::AddBiasNode(ge::ComputeGraph& graph, ge::NodePtr& fcNode, float shift,
                                             int64_t num_output, vector<ge::NodePtr>& fusionNodes) {
  ge::OpDescPtr fcOp = fcNode->GetOpDesc();
  // if the former fc has no bias, newBiasData = transBias
  ge::OpDescPtr constOpDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((constOpDesc = std::make_shared<ge::OpDesc>(fcNode->GetName() + "_bias", CONSTANTOPTAB)),
                          return FAILED);
  ge::GeTensorDesc constOutDesc;
  ge::GeShape biasShape({num_output});
  constOutDesc.SetShape(biasShape);
  constOutDesc.SetOriginFormat(ge::FORMAT_NCHW);
  constOutDesc.SetOriginShape(biasShape);
  constOutDesc.SetOriginDataType(ge::DT_FLOAT);
  constOutDesc.SetDataType(ge::DT_FLOAT);

  FUSION_PASS_CHECK(constOpDesc->AddOutputDesc(constOutDesc) != GRAPH_SUCCESS, OP_LOGW(fcNode, "AddOutputDesc failed!"),
                    return fe::NOT_CHANGED);
  constOpDesc->SetType(CONSTANTOPTAB);

  ge::GeTensorPtr biasPtr = nullptr;
  std::unique_ptr<float[]> biasDataTemp(new (std::nothrow) float[num_output]());
  for (int64_t i = 0; i < num_output; i++) {
    biasDataTemp[i] = shift;
  }

  FUSION_PASS_MAKE_SHARED(
      biasPtr = std::make_shared<ge::GeTensor>(constOutDesc, reinterpret_cast<uint8_t *>(biasDataTemp.get()),
                                               num_output * sizeof(float)),
      biasPtr = nullptr;
      return PARAM_INVALID);

  auto ret = biasPtr->SetData(reinterpret_cast<uint8_t*>(biasDataTemp.get()), num_output * sizeof(float));
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(fcNode, "set bias data failed"), return ret);

  ge::NodePtr constNode = graph.AddNode(constOpDesc);
  FUSION_PASS_CHECK(constNode == nullptr, OP_LOGW(fcNode, "constNode is nullptr"), return fe::NOT_CHANGED);

  vector<ge::GeTensorPtr> tensorVec = {biasPtr};
  FUSION_PASS_CHECK(ge::OpDescUtils::SetWeights(constNode, tensorVec) != GRAPH_SUCCESS,
                    OP_LOGW(fcNode, "set weight failed"), return fe::NOT_CHANGED);

  fusionNodes.push_back(constNode);
  // bias is the name of the third input of fc in IR fc.h
  ret = fcNode->AddLinkFrom("b", constNode);
  FUSION_PASS_CHECK(
      ret != SUCCESS,
      OP_LOGW(fcNode, "FcNode[%s]: add edge between new const node and fc node failed!", fcNode->GetName().c_str()),
      return ret);

  return SUCCESS;
}

REGISTER_PASS("FullyConnectionPowerPass", BUILT_IN_GRAPH_PASS, FullyConnectionPowerPass);
} // namespace fe
