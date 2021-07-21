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
static const std::string FullyConnectionPowerPassHostOp = "FullyConnectionPowerPassHostOp";

uint64_t GetHostCpuAtomicId() {
  static std::atomic<uint64_t> globalTransAtomicId(0);
  return globalTransAtomicId.fetch_add(1, std::memory_order_relaxed);
}

Status CreateFullyPowerPassHostOp(const string &opType, const ge::NodePtr &fcNode, ge::ComputeGraph &graph,
                                vector<ge::NodePtr> &newNodes, const float &shift) {
    OP_LOGI(FUSED_OP_TYPE, "Create new fully power pass host op for dequant node [%s].", fcNode->GetName().c_str());
    std::stringstream opNameTemp;
    // the atomic id of trans nodes must be unique.(start from 0)
    opNameTemp << opType << "_" << GetHostCpuAtomicId();
    ge::OpDescPtr fcPowerHostOp = nullptr;
    FUSION_PASS_MAKE_SHARED((fcPowerHostOp = std::make_shared<ge::OpDesc>(opNameTemp.str(), opType)), return FAILED);
    FUSION_PASS_CHECK(fcPowerHostOp == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "create new host op failed"), return FAILED);
    // add input and output desc of new host op
    vector<ge::GeTensorPtr> weights = ge::OpDescUtils::MutableWeights(fcNode);
    FUSION_PASS_CHECK(weights.size() < 2, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "fc weights get failed"), return FAILED);
    ge::GeTensorPtr biasTensorPtr = weights[1];
    ge::GeTensorDesc biasTensorDesc = biasTensorPtr->GetTensorDesc();

    FUSION_PASS_CHECK(fcPowerHostOp->AddInputDesc(FC_POWER_OP_INPUT, biasTensorDesc) != GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "failed to add input desc to fcPowerHostOp."), return FAILED);
    FUSION_PASS_CHECK(fcPowerHostOp->MutableInputDesc(0) == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "get input desc 0 failed"),
                      return FAILED);
    fcPowerHostOp->MutableInputDesc(0)->SetOriginDataType(biasTensorDesc.GetDataType());
    fcPowerHostOp->MutableInputDesc(0)->SetOriginFormat(static_cast<ge::Format>(ge::GetPrimaryFormat(biasTensorDesc.GetFormat())));
    fcPowerHostOp->MutableInputDesc(0)->SetOriginShape(biasTensorDesc.GetShape());

    FUSION_PASS_CHECK(fcPowerHostOp->AddOutputDesc(FC_POWER_OP_OUTPUT, biasTensorDesc) != GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "failed to add output desc to fcPowerHostOp."), return FAILED);
    FUSION_PASS_CHECK(fcPowerHostOp->MutableOutputDesc(0) == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "get output desc 0 failed"),
                      return FAILED);
    fcPowerHostOp->MutableOutputDesc(0)->SetOriginFormat(static_cast<ge::Format>(ge::GetPrimaryFormat(biasTensorDesc.GetFormat())));
    fcPowerHostOp->MutableOutputDesc(0)->SetOriginShape(biasTensorDesc.GetShape());
    fcPowerHostOp->MutableOutputDesc(0)->SetDataType(biasTensorDesc.GetDataType());
    fcPowerHostOp->MutableOutputDesc(0)->SetShape(biasTensorDesc.GetShape());

    auto fcPowerNode = graph.AddNode(fcPowerHostOp);
    FUSION_PASS_CHECK(fcPowerNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "add new host op to graph failed"), return FAILED);
    newNodes.emplace_back(fcPowerNode);

    // Add edges between fc bias <--> new_host_cpu_op:0
    ge::InDataAnchorPtr fcInputAnchor = fcNode->GetInDataAnchor(FC_BIAS_INDEX);
    FUSION_PASS_CHECK(fcInputAnchor == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "fc get const anchor failed"), return false);
    ge::OutDataAnchorPtr fcBiasPeerOutAnchor = fcInputAnchor->GetPeerOutAnchor();
    FUSION_PASS_CHECK(fcBiasPeerOutAnchor == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "fc get const failed"), return false);
    auto fcPowerHostOpInputAnchor = fcPowerNode->GetInDataAnchor(FC_POWER_HOST_OP_BIAS_INDEX);
    if (ge::GraphUtils::AddEdge(fcBiasPeerOutAnchor, fcPowerHostOpInputAnchor) != ge::GRAPH_SUCCESS) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "Add Edge between const and new host op failed.");
      return FAILED;
    }
    if (ge::GraphUtils::RemoveEdge(fcBiasPeerOutAnchor, fcInputAnchor) != ge::GRAPH_SUCCESS) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "Remove Edge between FC and bias const op failed.");
      return FAILED;
    }

    auto fcPowerHostOpOutputAnchor = fcPowerNode->GetOutDataAnchor(0);
    if (ge::GraphUtils::AddEdge(fcPowerHostOpOutputAnchor, fcInputAnchor) != ge::GRAPH_SUCCESS) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "Add Edge between FC and new host op failed.");
      return FAILED;
    }

    if (!ge::AttrUtils::SetFloat(fcPowerNode->GetOpDesc(), POWER_SHIFT, shift)) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "set float failed.");
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
  OP_LOGI(FUSED_OP_TYPE, "Define FullyConnectionPowerPass pattern begin");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern1 = new (std::nothrow) FusionPattern("FullyConnectionPowerPass");
  FUSION_PASS_CHECK(pattern1 == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "new an object failed"), return patterns);

  pattern1->AddOpDesc(PATTERN_FULLYCONNECTION, {"FullyConnection"})
      .AddOpDesc(PATTERN_POWER, {"Power"})
      .SetInputs(PATTERN_POWER, {PATTERN_FULLYCONNECTION})
      .SetOutput(PATTERN_POWER);
  patterns.push_back(pattern1);
  OP_LOGI(FUSED_OP_TYPE, "Define FullyConnectionPowerPass pattern end");
  return patterns;
}

Status FullyConnectionPowerPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE, "Define FullyConnectionPowerPass fusion begin");
  ge::NodePtr powerNode = GetNodeFromMapping(PATTERN_POWER, mapping);
  ge::NodePtr fullyConnectionNode = GetNodeFromMapping(PATTERN_FULLYCONNECTION, mapping);

  FUSION_PASS_CHECK(powerNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "powerNode is null"), return PARAM_INVALID);
  FUSION_PASS_CHECK(fullyConnectionNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "fullyConnectionNode is null"),
                    return PARAM_INVALID);

  if (fullyConnectionNode->GetOutDataNodes().size() > 1) {
    return SUCCESS;
  }

  ge::ConstGeTensorDescPtr fullyConnectionNodeDescPtr = GetCurrNodeInputDesc(fullyConnectionNode, 0);
  FUSION_PASS_CHECK(fullyConnectionNodeDescPtr == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "fcNode's OpDesc is null, fusion failed."),
                    return FAILED);
  ge::DataType inputType = fullyConnectionNodeDescPtr->GetDataType();
  if (inputType == ge::DT_INT8 || inputType == ge::DT_UINT8) {
    OP_LOGI(FUSED_OP_TYPE, "fc dataType is not float16 , graph not changed.");
    return NOT_CHANGED;
  }

  string fullyConnectionNodeName = fullyConnectionNode->GetName();
  // Check fc's output is power node
  Status ret = PatternFusionUtil::LinkControlEdge(powerNode, fullyConnectionNode);

  FUSION_PASS_CHECK(
      ret != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "FcNode[%s]: LinkControlEdge not success.", fullyConnectionNodeName.c_str()),
      return ret);

  

  ge::OpDescPtr powerDesc = powerNode->GetOpDesc();
  FUSION_PASS_CHECK(powerDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "powerNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  float power = -1;
  float scale = -1;
  float shift = -1;
  int64_t num_output = 0;

  ge::AttrUtils::GetFloat(powerNode->GetOpDesc(), "power", power);
  ge::AttrUtils::GetFloat(powerNode->GetOpDesc(), "scale", scale);
  ge::AttrUtils::GetFloat(powerNode->GetOpDesc(), "shift", shift);

  ge::AttrUtils::GetInt(fullyConnectionNode->GetOpDesc(), "num_output", num_output);

  if (power != 1 || scale != 1) {
    OP_LOGI(FUSED_OP_TYPE, "fc and power nodes is not meet the conditions , graph not changed.");
    return NOT_CHANGED;
  }

  // Get size for fc, check it
  int64_t inputs_num = fullyConnectionNode->GetOpDesc()->GetInputsSize();
  // create bias node
  bool hasBias = true;
  if (inputs_num == 2) {
    AddBiasNode(graph, fullyConnectionNode, shift, num_output, fusionNodes);
    hasBias = false;
  }

  if (hasBias) {
    Tensor data;
    Operator op = ge::OpDescUtils::CreateOperatorFromNode(fullyConnectionNode);
    if (GRAPH_SUCCESS != op.GetInputConstData("b", data)) {
      OP_LOGI(FUSED_OP_TYPE, "GetInputConstData of bias failed.");
      return NOT_CHANGED;
    }
    /* Create Host Cpu Op */
    OP_LOGI(FUSED_OP_TYPE, "Create host op to calc shift of node:[%s].", fullyConnectionNode->GetName().c_str());
    vector<ge::NodePtr> fusion_nodes;
    Status ret = CreateFullyPowerPassHostOp(FullyConnectionPowerPassHostOp, fullyConnectionNode, graph,
        fusion_nodes, shift);
    if (ret != SUCCESS || fusion_nodes.empty()) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "Create host cpu op for fc node %s failed", fullyConnectionNode->GetName().c_str());
      return ret;
    }
  }

  auto outDataAnchor0 = powerNode->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(outDataAnchor0 == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "get out data anchor of 0"), return FAILED);
  for (auto inDataAnchor : outDataAnchor0->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(outDataAnchor0, inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "Remove power and outnode edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(GetPeerOutAnchorWithInDataAnchor(powerNode, 0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "Add innode and outnode edge failed."), return FAILED);
  }

  OP_LOGI(FUSED_OP_TYPE, "FullyConnectionPowerPass fusion success");

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

  FUSION_PASS_CHECK(constOpDesc->AddOutputDesc(constOutDesc) != GRAPH_SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "AddOutputDesc failed!"),
                    return FAILED);
  constOpDesc->SetType(CONSTANTOPTAB);

  ge::GeTensorPtr biasPtr = nullptr;
  std::unique_ptr<float[]> biasDataTemp(new (std::nothrow) float[num_output]());
  for (int64_t i = 0; i < num_output; i++) {
    biasDataTemp[i] = shift;
  }

  FUSION_PASS_MAKE_SHARED(biasPtr = std::make_shared<ge::GeTensor>(constOutDesc, reinterpret_cast<uint8_t*>(biasDataTemp.get()),
                                                                   num_output * sizeof(float)),
                          biasPtr = nullptr;
                          return PARAM_INVALID);

  auto ret = biasPtr->SetData(reinterpret_cast<uint8_t*>(biasDataTemp.get()), num_output * sizeof(float));
  if (ret != SUCCESS){
    biasPtr = nullptr;
    OP_LOGW(FUSED_OP_TYPE, "set bias data failed!");
    return ret;
  }
  ge::NodePtr constNode = graph.AddNode(constOpDesc);
  if (constNode == nullptr){
    biasPtr = nullptr;
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "constNode is nullptr");
    return PARAM_INVALID;
  }
  vector<ge::GeTensorPtr> tensorVec = {biasPtr};
  if ((ge::OpDescUtils::SetWeights(constNode, tensorVec)) != GRAPH_SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "set weight failed!");
    return FAILED;
  }
  fusionNodes.push_back(constNode);
  // bias is the name of the third input of fc in IR fc.h
  ret = fcNode->AddLinkFrom("b", constNode);
  if (ret != SUCCESS){
    biasPtr = nullptr;
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "FcNode[%s]: add edge between new const node and fc node failed!", fcNode->GetName().c_str());
    return ret;
  }
  return SUCCESS;
}

REGISTER_PASS("FullyConnectionPowerPass", BUILT_IN_GRAPH_PASS, FullyConnectionPowerPass);
}  // namespace fe
