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

#include "dynamic_rnn_v3_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "external/graph/operator_factory.h"
#include "common/util/platform_info.h"

using namespace ge;
namespace fe {
static const char *FUSED_NODE = "DynamicRNNV3";
static const std::string PATTERN_FUSEDNODE = "DynamicRNNV3";

vector<FusionPattern *> DynamicRNNV3FusionPass::DefinePatterns()
{
  vector<FusionPattern *> patterns;

  FusionPattern *pattern = new (std::nothrow) FusionPattern("DynamicRNNV3FusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(),
        "DynamicRNNV3FusionPass pattern object failed."), return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, { FUSED_NODE }).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

ge::NodePtr DynamicRNNV3FusionPass::AddBroadCastForCt(ge::ComputeGraph &graph, ge::NodePtr fusedNode, bool &failStatus,
                                                      int64_t batchSize, int64_t hiddenSize, int64_t stateSize){
  ge::OpDescPtr broadcast_op_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (broadcast_op_desc = std::make_shared<ge::OpDesc>(fusedNode->GetName() + "BroadCast", "BroadcastToD")),
      failStatus=true; return nullptr);
  std::vector<int64_t> data_dim = {stateSize, batchSize, 1};
  ge::GeTensorDesc inputTensorDesc = ge::GeTensorDesc(ge::GeShape(data_dim), ge::FORMAT_ND, ge::DT_FLOAT16);
  inputTensorDesc.SetOriginShape(ge::GeShape(data_dim));
  inputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  broadcast_op_desc->AddInputDesc("x", inputTensorDesc);

  std::vector<int64_t> res_nd_dim = {stateSize, batchSize, 16};
  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(ge::GeShape(res_nd_dim), ge::FORMAT_ND, ge::DT_FLOAT16);
  outputTensorDesc.SetOriginShape(ge::GeShape(res_nd_dim));
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  broadcast_op_desc->AddOutputDesc("y", outputTensorDesc);

  ge::AttrUtils::SetListInt(broadcast_op_desc, "shape", res_nd_dim);

  ge::NodePtr broadcast_node1 = graph.AddNode(broadcast_op_desc);
  FUSION_PASS_CHECK(!broadcast_node1, OP_LOGE(FUSED_OP_TYPE.c_str(),
        "add broadcat fail."), return nullptr);
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  fusedDesc->UpdateInputDesc(10, outputTensorDesc);

  ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(10)->GetPeerOutAnchor(), broadcast_node1->GetInDataAnchor(0));

  InDataAnchorPtr inAnchorPtr = fusedNode->GetInDataAnchor(10);
  inAnchorPtr->UnlinkAll();
  ge::GraphUtils::AddEdge(broadcast_node1->GetOutDataAnchor(0), fusedNode->GetInDataAnchor(10));

  return broadcast_node1;
}

ge::GeTensorPtr DynamicRNNV3FusionPass::ProcessDynamicRnnV3Wdate(ge::NodePtr fusedNode, bool &failStatus,
                                                                 int64_t index, int64_t batchSize, int64_t hiddenSize){
  std::vector<int64_t> wcIn = {batchSize, hiddenSize};
  ge::GeShape wcShape(wcIn);
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  DataType dataType = fusedDesc->GetInputDesc(index).GetDataType();
  ge::GeTensorDesc wcTensorDesc(wcShape, ge::FORMAT_ND, dataType);

  wcTensorDesc.SetOriginShape(wcShape);
  wcTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  fusedNode->GetOpDesc()->UpdateInputDesc(index, wcTensorDesc);

  ge::InDataAnchorPtr wcInputAnchorPtr0 = fusedNode->GetInDataAnchor(index);
  ge::OutDataAnchorPtr constWcAnchorPtr0 = wcInputAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr wcNode = constWcAnchorPtr0->GetOwnerNode();
  vector<ge::GeTensorPtr> wcT = ge::OpDescUtils::MutableWeights(wcNode);
  ge::GeTensorPtr wcTensorPtr = wcT[0];

  float *wcData = (float *)wcTensorPtr->GetData().data();
  unique_ptr<float[]> dstWcData(new (std::nothrow) float[batchSize * hiddenSize]());

  auto retMem = memset_s(dstWcData.get(), batchSize*hiddenSize, 0, batchSize * hiddenSize);
  FUSION_PASS_CHECK(retMem != EOK, OP_LOGE("DynamicRnnV3", "Failed to operate memset_s function."), return nullptr);
  float *dstWc = dstWcData.get();

  for (int i = 0; i < batchSize; i++) {
    for (int j = 0; j < hiddenSize; j++) {
      dstWc[i*hiddenSize + j] = *(wcData + j);
    }
  }

  wcTensorPtr->SetData(reinterpret_cast<uint8_t *>(dstWcData.get()), (batchSize * hiddenSize) * sizeof(float));
  wcTensorPtr->SetTensorDesc(wcTensorDesc);

  return wcTensorPtr;
}

Status DynamicRNNV3FusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &newNodes)
{
  OP_LOGI(FUSED_OP_TYPE.c_str(), "DynamicRNNV3 start fusion.");

  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(),
        "fusedNode OpDesc is null, fusion failed."), return PARAM_INVALID);
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(),
        "fusedDesc OpDesc is null, fusion failed."), return PARAM_INVALID);
  int64_t wciIndex = 6;
  int64_t wcfIndex = 7;
  int64_t wcoIndex = 8;
  int64_t batchSize = fusedDesc->GetInputDesc(0).GetShape().GetDim(1);
  int64_t hiddenSize = fusedDesc->GetInputDesc(1).GetShape().GetDim(1) / 4;
  int64_t stateSize = fusedDesc->GetInputDesc(0).GetShape().GetDim(0);

  bool failStatus = true;

  ProcessDynamicRnnV3Wdate(fusedNode, failStatus, wciIndex, batchSize, hiddenSize);
  FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "Process wci fail."), return FAILED);
  ProcessDynamicRnnV3Wdate(fusedNode, failStatus, wcfIndex, batchSize, hiddenSize);
  FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "Process wcf fail."), return FAILED);
  ProcessDynamicRnnV3Wdate(fusedNode, failStatus, wcoIndex, batchSize, hiddenSize);
  FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "Process wco fail."), return FAILED);

  AddBroadCastForCt(graph, fusedNode, failStatus, batchSize, hiddenSize, stateSize);
  FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "Process wco fail."), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "DynamicRNNV3 end fusion");

  return SUCCESS;

}

REGISTER_PASS("DynamicRNNV3FusionPass", BUILT_IN_GRAPH_PASS, DynamicRNNV3FusionPass);
}