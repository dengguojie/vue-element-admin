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
#include <vector>
#include <map>
#include <memory>
#include <string>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"


namespace fe {
static const char PATTERN_POWER[] = "Power";
static const char PATTERN_FULLYCONNECTION[] = "FullyConnection";
static const int BIAS_INDEX = 2;
static const char FUSED_OP_TYPE[] = "FullyConnection";
static const char CONSTANTOPTAB[] = "Const";
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
  FUSION_PASS_CHECK(pattern1 == nullptr, OP_LOGE(FUSED_OP_TYPE, "new an object failed"), return patterns);

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

  FUSION_PASS_CHECK(powerNode == nullptr, OP_LOGE(FUSED_OP_TYPE, "powerNode is null"), return PARAM_INVALID);
  FUSION_PASS_CHECK(fullyConnectionNode == nullptr, OP_LOGE(FUSED_OP_TYPE, "fullyConnectionNode is null"),
                    return PARAM_INVALID);

  if (fullyConnectionNode->GetOutDataNodes().size() > 1) {
    return SUCCESS;
  }
  ge::DataType inputType = fullyConnectionNode->GetOpDesc()->GetInputDesc(0).GetDataType();
  if (inputType == ge::DT_INT8 || inputType == ge::DT_UINT8) {
    OP_LOGI(FUSED_OP_TYPE, "fc dataType is not float16 , graph not changed.");
    return NOT_CHANGED;
  }

  string fullyConnectionNodeName = fullyConnectionNode->GetName();
  // Check fc's output is power node
  Status ret = PatternFusionUtil::LinkControlEdge(powerNode, fullyConnectionNode);

  FUSION_PASS_CHECK(
      ret != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE, "FcNode[%s]: LinkControlEdge not success.", fullyConnectionNodeName.c_str()),
      return ret);

  // Find fc and power operation
  ge::OpDescPtr fullyConnectionDesc = fullyConnectionNode->GetOpDesc();
  FUSION_PASS_CHECK(fullyConnectionDesc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE, "fcNode's OpDesc is null, fusion failed."), return PARAM_INVALID);

  ge::OpDescPtr powerDesc = powerNode->GetOpDesc();
  FUSION_PASS_CHECK(powerDesc == nullptr, OP_LOGE(FUSED_OP_TYPE, "powerNode's OpDesc is null, fusion failed."),
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
  int64_t inputs_num = fullyConnectionDesc->GetInputsSize();
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
    std::vector<float> const_data;
    float* const_data_ptr = reinterpret_cast<float*>(data.GetData());
    FUSION_PASS_CHECK(const_data_ptr == nullptr,
                      OP_LOGE(FUSED_OP_TYPE, "const_data_ptr is null, fusion failed."), return PARAM_INVALID);
    int64_t size = 0;
    size = data.GetSize() / sizeof(float);
    for (int64_t i = 0; i < size; ++i) {
      const_data.push_back(static_cast<float>((*(const_data_ptr + i))));
    }
    for (int64_t i = 0; i < size; ++i) {
      const_data[i] = shift + const_data[i];
    }

    vector<ge::GeTensorPtr> sliceTensorPtr = ge::OpDescUtils::MutableWeights(fullyConnectionNode);
    FUSION_PASS_CHECK(sliceTensorPtr.size() < 2, OP_LOGW(FUSED_OP_TYPE, "fc weights get failed"), return false);
    ge::GeTensorPtr offsetsTensorPtr = sliceTensorPtr[1];
    offsetsTensorPtr->SetData(reinterpret_cast<uint8_t*>(const_data.data()), num_output * sizeof(float));
  }

  for (auto inDataAnchor : powerNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(powerNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE, "Remove power and outnode edge failed."), return FAILED);
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(powerNode->GetInDataAnchor(0)->GetPeerOutAnchor(), inDataAnchor) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE, "Add innode and outnode edge failed."), return FAILED);
  }

  FUSION_PASS_CHECK(graph.RemoveNode(powerNode) != SUCCESS, OP_LOGE(FUSED_OP_TYPE, "Remove powerNode failed."),
                    return FAILED);

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

  FUSION_PASS_CHECK(constOpDesc->AddOutputDesc(constOutDesc) != SUCCESS, OP_LOGE("AddOutputDesc failed!"),
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
    OP_LOGE(FUSED_OP_TYPE, "constNode is nullptr");
    return PARAM_INVALID;
  }
  vector<ge::GeTensorPtr> tensorVec = {biasPtr};
  ge::OpDescUtils::SetWeights(constNode, tensorVec);
  fusionNodes.push_back(constNode);
  // bias is the name of the third input of fc in IR fc.h
  ret = fcNode->AddLinkFrom("b", constNode);
  if (ret != SUCCESS){
    biasPtr = nullptr;
    OP_LOGE("FcNode[%s]: add edge between new const node and fc node failed!", fcNode->GetName().c_str());
    return ret;
  }
  return SUCCESS;
}

REGISTER_PASS("FullyConnectionPowerPass", BUILT_IN_GRAPH_PASS, FullyConnectionPowerPass);
}  // namespace fe
