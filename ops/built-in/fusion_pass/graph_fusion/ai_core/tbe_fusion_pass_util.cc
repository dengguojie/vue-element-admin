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
 * \file tbe_fusion_pass_util.cpp
 * \brief
 */
#include "tbe_fusion_pass_util.h"

#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <functional>
#include <memory>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"

using namespace ge;
using namespace std;

namespace fe {
static const string FUSED_TRANSPOSE_NODE = "TransposeD";
static const string FUSED_CAST_NODE = "Cast";

Status AddTransposeBeforeNode(const ge::NodePtr& fusedNode, const int64_t& inputIndex, const vector<int64_t>& permList,
                              ge::ComputeGraph& graph) {
  string fuseNodeType = fusedNode->GetType();
  OP_LOGI(fuseNodeType.c_str(), "begin to insert Transpose before %s.", fusedNode->GetName().c_str());

  std::shared_ptr<ge::OpDesc> beforeTransposeDesc = nullptr;
  std::string beforeTransposeName = fusedNode->GetName() + "_Input" + to_string(inputIndex) + "_TransposeBefore";
  beforeTransposeDesc = std::make_shared<ge::OpDesc>(beforeTransposeName, FUSED_TRANSPOSE_NODE);
  FUSION_PASS_CHECK(
      beforeTransposeDesc == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(fuseNodeType.c_str(), "beforeTransposeDesc is null, fusion failed."),
      return FAILED);
  ge::GeTensorDesc inputDesc = fusedNode->GetOpDesc()->GetInputDesc(inputIndex);
  FUSION_PASS_CHECK(beforeTransposeDesc->AddInputDesc("x", inputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(fuseNodeType.c_str(), "add before transpose of %s failed.",
                                                   fusedNode->GetName().c_str()),
                    return FAILED);
  vector<int64_t> inputShape = inputDesc.GetShape().GetDims();
  size_t inputShapeDims = inputShape.size();
  FUSION_PASS_CHECK(
      permList.size() < inputShapeDims,
      VECTOR_FUSION_INNER_ERR_REPORT(fuseNodeType.c_str(), "permList size less then %lu .", inputShapeDims),
      return FAILED);
  vector<int64_t> outputShapeVec;
  for (size_t n = 0; n < inputShapeDims; n++) {
    outputShapeVec.push_back(inputShape[permList[n]]);
  }
  ge::GeShape outputShape(outputShapeVec);
  inputDesc.SetShape(outputShape);
  inputDesc.SetOriginShape(outputShape);
  FUSION_PASS_CHECK(beforeTransposeDesc->AddOutputDesc("y", inputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(fuseNodeType.c_str(), "add before transpose of %s failed.",
                                                   fusedNode->GetName().c_str()),
                    return FAILED);
  ge::AttrUtils::SetListInt(beforeTransposeDesc, "perm", permList);

  // add node to graph
  ge::NodePtr beforeTransposeNode = graph.AddNode(beforeTransposeDesc);
  // add input for transpose
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(inputIndex)->GetPeerOutAnchor(),
                                            beforeTransposeNode->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(fuseNodeType.c_str(), "AddEdge edge failed."), return FAILED);
  // remove fused node input edge
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fusedNode->GetInDataAnchor(inputIndex)->GetPeerOutAnchor(),
                                               fusedNode->GetInDataAnchor(inputIndex)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(fuseNodeType.c_str(), "Remove out data edge failed."),
                    return FAILED);
  // add output for transpose
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(beforeTransposeNode->GetOutDataAnchor(0),
                                            fusedNode->GetInDataAnchor(inputIndex)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(fuseNodeType.c_str(), "AddEdge edge failed."), return FAILED);
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

  std::shared_ptr<ge::OpDesc> afterTransposeDesc = nullptr;
  std::string TransposeName = fusedNode->GetName() + "_Output" + to_string(outputIndex) + "TransposeAfter";
  afterTransposeDesc = std::make_shared<ge::OpDesc>(TransposeName, FUSED_TRANSPOSE_NODE);
  FUSION_PASS_CHECK(afterTransposeDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(fuseNodeType.c_str(), "afterTransposeDesc is null, fusion failed."),
                    return FAILED);
  ge::GeTensorDesc outputDesc = fusedNode->GetOpDesc()->GetOutputDesc(outputIndex);
  FUSION_PASS_CHECK(afterTransposeDesc->AddInputDesc("x", outputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(fuseNodeType.c_str(), "add after transpose of %s failed.",
                                                   fusedNode->GetName().c_str()),
                    return FAILED);
  vector<int64_t> inputShape = outputDesc.GetShape().GetDims();
  size_t inputShapeDims = inputShape.size();
  FUSION_PASS_CHECK(
      permList.size() < inputShapeDims,
      VECTOR_FUSION_INNER_ERR_REPORT(fuseNodeType.c_str(), "permList size less then %lu .", inputShapeDims),
      return FAILED);
  vector<int64_t> outputShapeVec;
  for (size_t n = 0; n < inputShapeDims; n++) {
    outputShapeVec.push_back(inputShape[permList[n]]);
  }
  ge::GeShape outputShape(outputShapeVec);
  outputDesc.SetShape(outputShape);
  outputDesc.SetOriginShape(outputShape);
  FUSION_PASS_CHECK(afterTransposeDesc->AddOutputDesc("y", outputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(fuseNodeType.c_str(), "add after transpose of %s failed.",
                                                   fusedNode->GetName().c_str()),
                    return FAILED);
  ge::AttrUtils::SetListInt(afterTransposeDesc, "perm", permList);

  // add node to graph
  ge::NodePtr transposeNode = graph.AddNode(afterTransposeDesc);

  // add edge transpose output with other node input
  for (auto inDataAnchor : fusedNode->GetOutDataAnchor(outputIndex)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fusedNode->GetOutDataAnchor(outputIndex), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(fuseNodeType.c_str(), "Remove edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(transposeNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(fuseNodeType.c_str(), "Add edge failed."), return FAILED);
  }

  // add input for transpose
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(fusedNode->GetOutDataAnchor(outputIndex), transposeNode->GetInDataAnchor(0)) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(fuseNodeType.c_str(), "Add edge failed."), return FAILED);
  OP_LOGI(fuseNodeType.c_str(), "end to insert Transpose after %s.", fusedNode->GetName().c_str());

  return SUCCESS;
}

Status AddCastAfterNode(const ge::NodePtr& fused_node, const int64_t& output_index, const ge::DataType& dst_type,
                        ge::ComputeGraph& graph) {
  string fuse_nodetype = fused_node->GetType();
  OP_LOGI(fuse_nodetype.c_str(), "begin to insert Cast after %s.", fused_node->GetName().c_str());

  std::shared_ptr<ge::OpDesc> after_cast_desc = nullptr;
  std::string CastName = fused_node->GetName() + "_Output" + to_string(output_index) + "CastAfter";
  after_cast_desc = std::make_shared<ge::OpDesc>(CastName, FUSED_CAST_NODE);
  FUSION_PASS_CHECK(after_cast_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(fuse_nodetype.c_str(), "after_cast_desc is null, fusion failed."),
                    return FAILED);
  ge::GeTensorDesc output_desc = fused_node->GetOpDesc()->GetOutputDesc(output_index);
  FUSION_PASS_CHECK(after_cast_desc->AddInputDesc("x", output_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(fuse_nodetype.c_str(), "add after cast of %s failed.",
                                                   fused_node->GetName().c_str()),
                    return FAILED);

  ge::GeTensorDesc aftercast_output_desc = fused_node->GetOpDesc()->GetOutputDesc(output_index).Clone();
  aftercast_output_desc.SetDataType(dst_type);
  FUSION_PASS_CHECK(after_cast_desc->AddOutputDesc("y", aftercast_output_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(fuse_nodetype.c_str(), "add after cast of %s failed.",
                                                   fused_node->GetName().c_str()),
                    return FAILED);
  ge::AttrUtils::SetInt(after_cast_desc, "dst_type", dst_type);
  // add node to graph
  ge::NodePtr cast_node = graph.AddNode(after_cast_desc);

  // add edge cast output with other node input
  for (auto in_data_anchor : fused_node->GetOutDataAnchor(output_index)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(
        ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(output_index), in_data_anchor) != SUCCESS,
        VECTOR_FUSION_INNER_ERR_REPORT(fuse_nodetype.c_str(), "Remove edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(cast_node->GetOutDataAnchor(0), in_data_anchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(fuse_nodetype.c_str(), "Add edge failed."), return FAILED);
  }

  // add input for cast
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(fused_node->GetOutDataAnchor(output_index), cast_node->GetInDataAnchor(0)) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(fuse_nodetype.c_str(), "Add edge failed."), return FAILED);
  OP_LOGI(fuse_nodetype.c_str(), "end to insert cast after %s.", fused_node->GetName().c_str());

  return SUCCESS;
}

template <typename T>
static std::vector<int64_t> GetConstIntValue(const uint8_t* const_data, size_t data_size) {
  size_t size = data_size / sizeof(T);
  std::vector<int64_t> result(size);
  const T* data = reinterpret_cast<const T*>(const_data);
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
    VECTOR_FUSION_INNER_ERR_REPORT("GetConstIntData", "GetConstIntData is not support data_type[%d]!", (int)data_type);
    return false;
  }

  const_values = found->second(data.GetData(), data.GetSize());

  return true;
}

bool TbeFusionPassUtil::GetConstIntData(const ge::Operator& op, const std::string& name,
                                        std::vector<int64_t>& values) {
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

bool TbeFusionPassUtil::UpdateAttrIsInputConst(const ge::NodePtr& fuse_node) {
  Operator fuse_op = ge::OpDescUtils::CreateOperatorFromNode(fuse_node);
  ge::OpDescPtr fuse_desc = fuse_node->GetOpDesc();
  std::vector<bool> is_input_const = fuse_desc->GetIsInputConst();

  FUSION_PASS_CHECK(!is_input_const.empty(),
                    OP_LOGI("UpdateAttrIsInputConst", "The node(%s) have attr is_input_const, will not update again.",
                            fuse_op.GetName().c_str()),
                    return true);
  OP_LOGI("UpdateAttrIsInputConst", "will update the node(%s) attr(is_input_const).", TbeGetName(fuse_op).c_str());
  auto input_size = fuse_desc->GetInputsSize();
  FUSION_PASS_CHECK(
      input_size < 1,
      OP_LOGI("UpdateAttrIsInputConst", "The node(%s) have no input desc, will not update is_input_const.",
              fuse_op.GetName().c_str()),
      return true);
  for (size_t i = 0; i < input_size; i++) {
    auto peer_node = fuse_node->GetInDataAnchor(i)->GetPeerOutAnchor()->GetOwnerNode();
    auto peer_op_type = peer_node->GetType();
    bool is_const = false;
    if (peer_op_type == "Const" || peer_op_type == "Constant") {
      is_const = true;
      OP_LOGI("UpdateAttrIsInputConst", "the %d input is const node.", i);
    }
    is_input_const.push_back(is_const);
  }
  fuse_desc->SetIsInputConst(is_input_const);
  OP_LOGI("UpdateAttrIsInputConst", "update the node(%s) attr(is_input_const) end.", fuse_op.GetName().c_str());
  return true;
}

bool TbeFusionPassUtil::IsEmptyTensor(const ge::GeTensorDesc& tensor_desc) {
  vector<int64_t> shape_info = tensor_desc.GetShape().GetDims();
  int64_t shape_dim = shape_info.size();
  for (int64_t dim = 0; dim < shape_dim; dim++) {
    if (shape_info[dim] == 0) {
      OP_LOGI("IsEmptyTensor", "shape dim num is 1 and shape vector[0] is 0, is empty tensor.");
      return true;
    }
  }
  return false;
}
}  // namespace fe
