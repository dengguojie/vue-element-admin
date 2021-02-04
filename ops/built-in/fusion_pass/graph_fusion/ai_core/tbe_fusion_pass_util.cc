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
 * \file tbe_fusion_pass_util.cpp
 * \brief
 */
#include "tbe_fusion_pass_util.h"

#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <functional>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

using namespace ge;
using namespace std;

namespace fe {
static const string FUSED_TRANSPOSE_NODE = "TransposeD";

Status AddTransposeBeforeNode(const ge::NodePtr& fusedNode, const int64_t& inputIndex, const vector<int64_t>& permList,
                              ge::ComputeGraph& graph) {
  string fuseNodeType = fusedNode->GetType();
  OP_LOGI(fuseNodeType.c_str(), "begin to insert Transpose before %s.", fusedNode->GetName().c_str());
  ge::GeTensorDesc inputDesc = fusedNode->GetOpDesc()->GetInputDesc(inputIndex);

  std::shared_ptr<ge::OpDesc> beforeTransposeDesc = nullptr;
  std::string beforeTransposeName = fusedNode->GetName() + "_Input" + to_string(inputIndex) + "_TransposeBefore";
  beforeTransposeDesc = std::make_shared<ge::OpDesc>(beforeTransposeName, FUSED_TRANSPOSE_NODE);
  FUSION_PASS_CHECK(beforeTransposeDesc == nullptr,
                    OP_LOGE(fuseNodeType.c_str(), "beforeTransposeDesc is null, fusion failed."), return FAILED);
  FUSION_PASS_CHECK(beforeTransposeDesc->AddInputDesc("x", inputDesc) != SUCCESS,
                    OP_LOGE(fuseNodeType.c_str(), "add before transpose of %s failed.", fusedNode->GetName().c_str()),
                    return FAILED);
  vector<int64_t> inputShape = inputDesc.GetShape().GetDims();
  size_t inputShapeDims = inputShape.size();
  FUSION_PASS_CHECK(permList.size() < inputShapeDims,
                    OP_LOGE(fuseNodeType.c_str(), "permList size less then %d .", inputShapeDims),
                    return FAILED);
  vector<int64_t> outputShapeVec;
  for (size_t n = 0; n < inputShapeDims; n++) {
    outputShapeVec.push_back(inputShape[permList[n]]);
  }
  ge::GeShape outputShape(outputShapeVec);
  inputDesc.SetShape(outputShape);
  inputDesc.SetOriginShape(outputShape);
  FUSION_PASS_CHECK(beforeTransposeDesc->AddOutputDesc("y", inputDesc) != SUCCESS,
                    OP_LOGE(fuseNodeType.c_str(), "add before transpose of %s failed.", fusedNode->GetName().c_str()),
                    return FAILED);
  ge::AttrUtils::SetListInt(beforeTransposeDesc, "perm", permList);

  // add node to graph
  ge::NodePtr beforeTransposeNode = graph.AddNode(beforeTransposeDesc);
  // add input for transpose
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(inputIndex)->GetPeerOutAnchor(),
                                            beforeTransposeNode->GetInDataAnchor(0)) != SUCCESS,
                    OP_LOGE(fuseNodeType.c_str(), "AddEdge edge failed."), return FAILED);
  // remove fused node input edge
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fusedNode->GetInDataAnchor(inputIndex)->GetPeerOutAnchor(),
                                               fusedNode->GetInDataAnchor(inputIndex)) != SUCCESS,
                    OP_LOGE(fuseNodeType.c_str(), "Remove out data edge failed."), return FAILED);
  // add output for transpose
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(beforeTransposeNode->GetOutDataAnchor(0),
                                            fusedNode->GetInDataAnchor(inputIndex)) != SUCCESS,
                    OP_LOGE(fuseNodeType.c_str(), "AddEdge edge failed."), return FAILED);
  // add transpose input boxes end

  // update fused node input info
  auto opInputDesc = fusedNode->GetOpDesc();
  opInputDesc->UpdateInputDesc(inputIndex, inputDesc);
  OP_LOGI(fuseNodeType.c_str(), "end to insert Transpose before %s.", fusedNode->GetName().c_str());
  return SUCCESS;
}

Status AddTransposeAfterNode(const ge::NodePtr& fusedNode, const int64_t& outputIndex, const vector<int64_t>& permList,
                             ge::ComputeGraph& graph) {
  string fuseNodeType = fusedNode->GetType();
  OP_LOGI(fuseNodeType.c_str(), "begin to insert Transpose after %s.", fusedNode->GetName().c_str());
  ge::GeTensorDesc outputDesc = fusedNode->GetOpDesc()->GetOutputDesc(outputIndex);

  std::shared_ptr<ge::OpDesc> afterTransposeDesc = nullptr;
  std::string TransposeName = fusedNode->GetName() + "_Output" + to_string(outputIndex) + "TransposeAfter";
  afterTransposeDesc = std::make_shared<ge::OpDesc>(TransposeName, FUSED_TRANSPOSE_NODE);
  FUSION_PASS_CHECK(afterTransposeDesc == nullptr,
                    OP_LOGE(fuseNodeType.c_str(), "afterTransposeDesc is null, fusion failed."), return FAILED);
  FUSION_PASS_CHECK(afterTransposeDesc->AddInputDesc("x", outputDesc) != SUCCESS,
                    OP_LOGE(fuseNodeType.c_str(), "add after transpose of %s failed.", fusedNode->GetName().c_str()),
                    return FAILED);
  vector<int64_t> inputShape = outputDesc.GetShape().GetDims();
  size_t inputShapeDims = inputShape.size();
  FUSION_PASS_CHECK(permList.size() < inputShapeDims,
                    OP_LOGE(fuseNodeType.c_str(), "permList size less then %d .", inputShapeDims),
                    return FAILED);
  vector<int64_t> outputShapeVec;
  for (size_t n = 0; n < inputShapeDims; n++) {
    outputShapeVec.push_back(inputShape[permList[n]]);
  }
  ge::GeShape outputShape(outputShapeVec);
  outputDesc.SetShape(outputShape);
  outputDesc.SetOriginShape(outputShape);
  FUSION_PASS_CHECK(afterTransposeDesc->AddOutputDesc("y", outputDesc) != SUCCESS,
                    OP_LOGE(fuseNodeType.c_str(), "add after transpose of %s failed.", fusedNode->GetName().c_str()),
                    return FAILED);
  ge::AttrUtils::SetListInt(afterTransposeDesc, "perm", permList);

  // add node to graph
  ge::NodePtr transposeNode = graph.AddNode(afterTransposeDesc);

  // add edge transpose output with other node input
  for (auto inDataAnchor : fusedNode->GetOutDataAnchor(outputIndex)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fusedNode->GetOutDataAnchor(outputIndex), inDataAnchor) != SUCCESS,
                      OP_LOGE(fuseNodeType.c_str(), "Remove edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(transposeNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGE(fuseNodeType.c_str(), , "Add edge failed."), return FAILED);
  }

  // add input for transpose
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(fusedNode->GetOutDataAnchor(outputIndex), transposeNode->GetInDataAnchor(0)) != SUCCESS,
      OP_LOGE(fuseNodeType.c_str(), , "AddEdge edge failed."), return FAILED);
  OP_LOGI(fuseNodeType.c_str(), "end to insert Transpose after %s.", fusedNode->GetName().c_str());

  return SUCCESS;
}

template <typename T>
static std::vector<int64_t> GetConstIntValue(const uint8_t* const_data, size_t data_size) {
  size_t size = data_size / sizeof(T);
  std::vector<int64_t> result(size);
  T* data = (T*)const_data;
  for (size_t i = 0; i < size; i++) {
    result[i] = *(data + i);
  }

  return result;
}

bool TbeFusionPassUtil::GetConstIntData(const ge::Tensor& data, ge::DataType data_type,
                                        std::vector<int64_t>& const_values) {
  using namespace std;
  using namespace std::placeholders;
  const std::map<DataType, std::function<vector<int64_t>(const uint8_t*, size_t)>> type_call_map = {
      {DT_INT8, std::bind(GetConstIntValue<int8_t>, _1, _2)},
      {DT_INT16, std::bind(GetConstIntValue<int16_t>, _1, _2)},
      {DT_INT32, std::bind(GetConstIntValue<int32_t>, _1, _2)},
      {DT_INT64, std::bind(GetConstIntValue<int64_t>, _1, _2)},
  };

  auto found = type_call_map.find(data_type);
  if (found == type_call_map.end()) {
    OP_LOGE("GetConstIntData", "GetConstIntData is not support data_type[%d]!", (int)data_type);
    return false;
  }

  const_values = found->second(data.GetData(), data.GetSize());

  return true;
}

bool TbeFusionPassUtil::GetConstIntData(const ge::Operator& op, const std::string& name, std::vector<int64_t>& values) {
  ge::Tensor tensor;
  if (GRAPH_SUCCESS != op.GetInputConstData(name, tensor)) {
    OP_LOGI("GetConstIntData", "GetInputConstData failed, op name is %s and input name is %s", op.GetName().c_str(),
            name.c_str());
    return false;
  }

  if (!GetConstIntData(tensor, op.GetInputDesc(name).GetDataType(), values)) {
    OP_LOGI("GetConstIntData", "GetInputConstData failed, op name is %s and input name is %s", op.GetName().c_str(),
            name.c_str());
    return false;
  }

  return true;
}
}  // namespace fe
