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
 * \file map_index_pass.cpp
 * \brief avgPool fusion pass
 */
#include "map_index_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "quant_host_cpu_op_common.h"
#include "op_log.h"
#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "securec.h"

using namespace std;
using namespace ge;

namespace fe {
static const string PATTERN_MAPINDEX = "MapIndex";
static const char* MAPINDEX = "MapIndex";

vector<FusionPattern*> MapIndexFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  // define AvgPoolFusion
  FusionPattern* pattern = new (std::nothrow) FusionPattern("MapIndexFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  // define origin graph
  pattern->AddOpDesc(PATTERN_MAPINDEX, {MAPINDEX}).SetOutput(PATTERN_MAPINDEX);

  patterns.push_back(pattern);

  return patterns;
}

Status MapIndexFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "enter into MapIndexFusionPass");
  // mapindex node
  ge::NodePtr mapIndexNode = GetNodeFromMapping(PATTERN_MAPINDEX, mapping);
  FUSION_PASS_CHECK(mapIndexNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "mapIndexNode is null, fusion failed."),
                    return PARAM_INVALID);

  // input of MapIndex
  ge::OpDescPtr mapIndexDesc = mapIndexNode->GetOpDesc();
  FUSION_PASS_CHECK(mapIndexDesc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "mapIndexNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  auto xShape = mapIndexNode->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();
  FUSION_PASS_CHECK(xShape.empty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "xShape is empty!"), return PARAM_INVALID);
  int64_t xLength = xShape[0];
  OP_LOGI(FUSED_OP_TYPE.c_str(), "xLength = %ld", xLength);

  auto dataSeqShape = mapIndexNode->GetOpDesc()->GetInputDesc(1).GetShape().GetDims();
  FUSION_PASS_CHECK(dataSeqShape.empty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "dataSeqShape is empty!"),
                    return PARAM_INVALID);
  int64_t dataSeqLength = dataSeqShape[0];
  int64_t number = ((dataSeqLength / xLength + 7) / 8) * 8;
  int64_t dataSeqNewLength = number * xLength;
  OP_LOGI(FUSED_OP_TYPE.c_str(), "dataSeqLength = %ld, number = %ld, dataSeqNewLength = %ld", dataSeqLength, number,
          dataSeqNewLength);

  ge::InDataAnchorPtr dataSeqPtr = mapIndexNode->GetInDataAnchor(1);
  ge::OutDataAnchorPtr dataSeqAnchorPtr = dataSeqPtr->GetPeerOutAnchor();
  ge::NodePtr dataSeqNode = dataSeqAnchorPtr->GetOwnerNode();

  ge::GeShape dataSeqNewShape({dataSeqNewLength});
  ge::GeTensorDesc dataSeqTensorDesc = mapIndexDesc->GetInputDesc(1);
  dataSeqTensorDesc.SetOriginShape(dataSeqNewShape);
  dataSeqTensorDesc.SetShape(dataSeqNewShape);
  dataSeqTensorDesc.SetDataType(ge::DT_INT32);
  mapIndexDesc->UpdateInputDesc(1, dataSeqTensorDesc);

  vector<ge::GeTensorPtr> weightsDataSeq = ge::OpDescUtils::MutableWeights(dataSeqNode);
  FUSION_PASS_CHECK(weightsDataSeq.empty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "weightsDataSeq is empty!"),
                    return PARAM_INVALID);

  ge::GeTensorPtr dataSeqTensorPtr = weightsDataSeq[0];
  FUSION_PASS_CHECK(dataSeqTensorPtr == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "dataSeq is null ptr!"),
                    return PARAM_INVALID);

  dataSeqTensorPtr->SetTensorDesc(dataSeqTensorDesc);
  dataSeqNode->GetOpDesc()->UpdateOutputDesc(0, dataSeqTensorDesc);

  std::unique_ptr<int32_t[]> NewDataSeq(new (std::nothrow) int32_t[dataSeqNewLength]());
  memset_s(NewDataSeq.get(), dataSeqNewLength, 0, dataSeqNewLength);

  int32_t* dataSeqNewData = NewDataSeq.get();
  int32_t* dataSeqOldData = (int32_t*)(dataSeqTensorPtr->GetData().data());

  int64_t i = 0;
  int64_t j = 0;
  for (i = 0; i < dataSeqLength; i++) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "dataSeqData = %d", dataSeqOldData[i]);
  }

  for (i = 0; i < dataSeqLength; i = i + xLength) {
    for (j = 0; j < xLength; j++) {
      dataSeqNewData[i / xLength + j * number] = dataSeqOldData[i + j];
    }
  }

  for (i = 0; i < dataSeqNewLength; i++) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "dataSeqNewData = %d", dataSeqNewData[i]);
  }

  dataSeqTensorPtr->SetData(reinterpret_cast<uint8_t*>(NewDataSeq.get()), (dataSeqNewLength) * sizeof(int32_t));

  /*vector< ge::GeTensorPtr > weights = { assitPtr };
  auto inputDataAnchor = mapIndexDesc->GetInDataAnchor(1);
  auto peerOutDataAnchor = inputDataAnchor->GetPeerOutAnchor();
  ge::NodePtr const0 = peerOutDataAnchor->GetOwnerNode();
  ge::OpDescUtils::SetWeights(const0, weights);*/

  // inputDesc1.SetOriginFormat(ge::FORMAT_ND);
  return SUCCESS;
}

REGISTER_PASS("MapIndexFusionPass", BUILT_IN_GRAPH_PASS, MapIndexFusionPass);
}  // namespace fe
