/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
 * \file gru_fusion_pass.cpp
 * \brief GRU fusion pass
 *   (CommonGRU --> DynamicGRUV2)
 */
#include "gru_fusion_pass.h"

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
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "external/graph/operator_factory.h"

using namespace ge;
namespace fe {
static const char* FUSED_NODE = "CommonGRU";
static const std::string PATTERN_FUSEDNODE = "CommonGRU";
static const int BIAS_INPUT_INDEX = 3;
static const int BIAS_SPLIT_GROUP = 2;
static const int BIAS_CHANNEL_INDEX = 1;


vector<FusionPattern*> GRUFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("GRUFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

Status GRUFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  // get the NodePtr of GRU
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode is null."), return PARAM_INVALID);

  // get the OpDescPtr of GRU
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode's OpDesc is null."),
                    return PARAM_INVALID);

  auto gruOp = ge::OperatorFactory::CreateOperator(fusedDesc->GetName() + "_splitD_layer", "DynamicGRUV2");
  FUSION_PASS_CHECK(gruOp.IsEmpty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create DynamicGRUV2 operator error"),
                    return FAILED);

  // create DynamicGRUV2 OpDesc
  std::shared_ptr<ge::OpDesc> gruOpDesc = nullptr;
  gruOpDesc = ge::OpDescUtils::GetOpDescFromOperator(gruOp);
  gruOp.BreakConnect();
  FUSION_PASS_CHECK(gruOpDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "gruOpDesc is null, DynamicGRUV2 failed."),
                    return PARAM_INVALID);

  // process x
  GeTensorDesc xInput = fusedDesc->GetInputDesc(0);
  std::vector<int64_t> xInputDims = xInput.GetShape().GetDims();

  GeShape xInputOriginShape(xInputDims);
  xInput.SetOriginShape(xInputOriginShape);
  (void)ProcessNZFormat(xInputDims);
  GeShape xInputShape(xInputDims);
  xInput.Update(xInputShape, ge::FORMAT_FRACTAL_NZ, xInput.GetDataType());
  gruOpDesc->UpdateInputDesc("x", xInput);

  FUSION_PASS_CHECK(AddTransposNode(fusedNode, 1, graph) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add transpos failed"), return FAILED);
  FUSION_PASS_CHECK(AddTransposNode(fusedNode, 2, graph) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add transpos failed"), return FAILED);

  // process weight_input
  GeTensorDesc weightInput = fusedDesc->GetInputDesc(1);
  std::vector<int64_t> weightInputDims = RemoveNumDirectionsDim(weightInput.GetShape().GetDims(), true);

  GeShape weightInputOriginShape(weightInputDims);
  weightInput.SetOriginShape(weightInputOriginShape);
  (void)ProcessZFormat(weightInputDims);
  GeShape weightInputShape(weightInputDims);
  weightInput.Update(weightInputShape, ge::FORMAT_FRACTAL_Z, weightInput.GetDataType());
  gruOpDesc->UpdateInputDesc("weight_input", weightInput);

  // process weight_hidden
  GeTensorDesc weightHidden = fusedDesc->GetInputDesc(2);
  std::vector<int64_t> weightHiddenDims = RemoveNumDirectionsDim(weightHidden.GetShape().GetDims(), true);

  GeShape weightHiddenOriginShape(weightHiddenDims);
  weightHidden.SetOriginShape(weightHiddenOriginShape);
  (void)ProcessZFormat(weightHiddenDims);
  GeShape weightHiddenShape(weightHiddenDims);
  weightHidden.Update(weightHiddenShape, ge::FORMAT_FRACTAL_Z, weightHidden.GetDataType());
  gruOpDesc->UpdateInputDesc("weight_hidden", weightHidden);

  bool hasSeqLength = fusedDesc->MutableInputDesc("sequence_lens") != nullptr;
  OP_LOGI(FUSED_OP_TYPE.c_str(), "hasSeqLength");
  if (hasSeqLength) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "yes hasSeqLength");
    gruOpDesc->UpdateInputDesc("seq_length", *fusedDesc->MutableInputDesc("sequence_lens"));
  }

  bool hasInitH = fusedDesc->MutableInputDesc("initial_h") != nullptr;
  OP_LOGI(FUSED_OP_TYPE.c_str(), "init_h");
  if (hasInitH) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "yes hasInitH");
    GeTensorDesc initialH = *fusedDesc->MutableInputDesc("initial_h");
    std::vector<int64_t> initialHDims = RemoveNumDirectionsDim(initialH.GetShape().GetDims(), false);

    GeShape initialHOriginShape(initialHDims);
    initialH.SetOriginShape(initialHOriginShape);
    (void)ProcessNZFormat(initialHDims);
    GeShape initialHShape(initialHDims);
    initialH.Update(initialHShape, ge::FORMAT_FRACTAL_NZ, initialH.GetDataType());
    gruOpDesc->UpdateInputDesc("init_h", initialH);
  }

  GeTensorDesc y = fusedDesc->GetOutputDesc(0);
  std::vector<int64_t> yDims = ProcessOutputDim(y.GetShape().GetDims());
  GeShape yOriginShape(yDims);
  y.SetOriginShape(yOriginShape);
  (void)ProcessNZFormat(yDims);
  GeShape yShape(yDims);
  y.Update(yShape, ge::FORMAT_FRACTAL_NZ, y.GetDataType());
  gruOpDesc->UpdateOutputDesc("y", y);
  gruOpDesc->UpdateOutputDesc("output_h", y);
  gruOpDesc->UpdateOutputDesc("update", y);
  gruOpDesc->UpdateOutputDesc("reset", y);
  gruOpDesc->UpdateOutputDesc("new", y);
  gruOpDesc->UpdateOutputDesc("hidden_new", y);

  // create a splitD Op for bias
  bool hasBias = fusedDesc->MutableInputDesc("b") != nullptr;
  ge::NodePtr splitNode = nullptr;
  if (hasBias) {
    // add bias Node
    OP_LOGI(FUSED_OP_TYPE.c_str(), "CommonGRU has bias input.");
    FUSION_PASS_CHECK(AddBiasSplitNode(graph, fusedNode, splitNode) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add bias split node failed."), return FAILED);
    // splitNode must not be nullptr when AddBiasSplit returns SUCCESS
    ge::OpDescPtr splitDesc = splitNode->GetOpDesc();
    FUSION_PASS_CHECK(splitDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "splitNode's OpDesc is null."),
                      return PARAM_INVALID);
    GeTensorDesc splitOutDesc = splitDesc->GetOutputDesc(0);
    gruOpDesc->UpdateInputDesc("bias_input", splitOutDesc);
    gruOpDesc->UpdateInputDesc("bias_hidden", splitOutDesc);
  }

  // create DynamicGRUV2 Node
  ge::NodePtr gruNode = graph.AddNode(gruOpDesc);
  FUSION_PASS_CHECK(gruNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "DynamicGRUV2 node is null, fusion failed."),
                    return FAILED);

  // connect bias(splitD) to gru bias input
  if (hasBias) {
    for (int i = 0; i < BIAS_SPLIT_GROUP; i++) {
      graphStatus status = GraphUtils::AddEdge(splitNode->GetOutDataAnchor(i),
                                               gruNode->GetInDataAnchor(i + BIAS_INPUT_INDEX));
      FUSION_PASS_CHECK(status != GRAPH_SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add slice to conv edge fail"),
                        return FAILED);
    }
  }

  // connect x
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                       gruNode->GetInDataAnchor(0)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add DynamicLSTMV2 edge to fusion node x failed."), return FAILED);

  // connect weight_input
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                                       gruNode->GetInDataAnchor(1)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add DynamicLSTMV2 edge to fusion node weight_input failed."),
                    return FAILED);

  // connect weight_hidden
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(2)->GetPeerOutAnchor(),
                                                       gruNode->GetInDataAnchor(2)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add DynamicLSTMV2 edge to fusion node weight_hidden failed."),
                    return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "hasSeqLength");
  if (hasSeqLength) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "yes hasSeqLength");
    ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(4)->GetPeerOutAnchor(), gruNode->GetInDataAnchor(5));
  }

  int64_t first_dim_value = 1;
  ge::NodePtr output_node = gruNode;
  int anchor_index = 1;
  if (yDims[0] > first_dim_value) {
    ge::NodePtr slice_node = nullptr;
    auto ret = CreateSliceNode(graph, gruNode, slice_node);
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Create slice node fail."), return FAILED);
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(gruNode->GetOutDataAnchor(1), slice_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddEdge for slice node fail"), return FAILED);
    output_node = slice_node;
    anchor_index = 0;
  }

  ge::OutDataAnchorPtr outputY = fusedNode->GetOutDataAnchor(0);
  auto yOriTopPeerAnchors = outputY->GetPeerInDataAnchors();
  ge::OutDataAnchorPtr outputYH = fusedNode->GetOutDataAnchor(1);
  auto yhOriTopPeerAnchors = outputYH->GetPeerInDataAnchors();

  OP_LOGI(FUSED_OP_TYPE.c_str(), "init_h");
  if (hasInitH) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "yes hasInitH");
    ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(5)->GetPeerOutAnchor(), gruNode->GetInDataAnchor(6));
  }

  // unlink all input of CommonGRU
  for (auto inAnchor : fusedNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }

  // unlink all output of CommonGRU
  for (auto outAnchor : fusedNode->GetAllOutDataAnchors()) {
    if (outAnchor != nullptr) {
      outAnchor->UnlinkAll();
    }
  }

  for (uint64_t i = 0; i < yOriTopPeerAnchors.size(); ++i) {
    ge::InDataAnchorPtr oriTopPeerAnchorPtri = yOriTopPeerAnchors.at(i);
    ge::NodePtr outputNode = oriTopPeerAnchorPtri->GetOwnerNode();
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(gruNode->GetOutDataAnchor(0), oriTopPeerAnchorPtri),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add DynamicLSTMV2 edge to fusion node output y failed."),
                      return FAILED);
  }

  for (uint64_t i = 0; i < yhOriTopPeerAnchors.size(); ++i) {
    ge::InDataAnchorPtr oriTopPeerAnchorPtri = yhOriTopPeerAnchors.at(i);
    ge::NodePtr outputNode = oriTopPeerAnchorPtri->GetOwnerNode();
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(output_node->GetOutDataAnchor(anchor_index), oriTopPeerAnchorPtri),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add DynamicLSTMV2 edge to fusion node output y_h failed."), return FAILED);
  }

  return SUCCESS;
}

Status GRUFusionPass::AddBiasSplitNode(ge::ComputeGraph& graph, ge::NodePtr& fusedNode, ge::NodePtr& splitNode) {
  OpDescPtr splitDesc = std::make_shared<ge::OpDesc>(fusedNode->GetName() + "/DynamicGRUV2_split", "SplitD");
  FUSION_PASS_CHECK(splitDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "splitD is null, SplitD failed."),
                    return PARAM_INVALID);
  AttrUtils::SetInt(splitDesc, "split_dim", 1);
  AttrUtils::SetInt(splitDesc, "num_split", BIAS_SPLIT_GROUP);

  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode's OpDesc is null."),
                    return PARAM_INVALID);
  ge::GeTensorDesc bias = fusedDesc->GetInputDesc(BIAS_INPUT_INDEX);
  FUSION_PASS_CHECK(splitDesc->AddInputDesc(bias) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add SplitD input"),
                    return FAILED);

  // build split node Output Desc
  GeTensorDesc inputDesc = fusedDesc->GetInputDesc(BIAS_INPUT_INDEX);
  GeShape inputShape = inputDesc.GetShape();
  int newInputChn = inputShape.GetDim(BIAS_CHANNEL_INDEX);
  GeShape splitOutShape = inputShape;
  splitOutShape.SetDim(BIAS_CHANNEL_INDEX, newInputChn / BIAS_SPLIT_GROUP);
  std::vector<int64_t> splitOutDims = RemoveNumDirectionsDim(splitOutShape.GetDims(), false);
  GeShape splitDShape(splitOutDims);
  GeTensorDesc splitOutDesc = inputDesc;
  splitOutDesc.SetShape(splitDShape);
  splitOutDesc.SetOriginShape(splitDShape);
  for (int i = 0; i < BIAS_SPLIT_GROUP; i++) {
    FUSION_PASS_CHECK(splitDesc->AddOutputDesc(splitOutDesc) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add bias split output failed."), return FAILED);
  }

  // create SplitD Node
  splitNode = graph.AddNode(splitDesc);
  FUSION_PASS_CHECK(splitNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "SplitD node is null, fusion failed."),
                    return FAILED);
  // connect bias to Split input
  graphStatus status = GraphUtils::AddEdge(fusedNode->GetInDataAnchor(BIAS_INPUT_INDEX)->GetPeerOutAnchor(),
                                           splitNode->GetInDataAnchor(0));
  FUSION_PASS_CHECK(status != GRAPH_SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add data to Split edge fail"),
                    return FAILED);
  return SUCCESS;
}

Status GRUFusionPass::CreateSliceNode(ge::ComputeGraph& graph, ge::NodePtr& gru_node, ge::NodePtr& new_node) {
  ge::OpDescPtr new_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((new_desc = std::make_shared<ge::OpDesc>(gru_node->GetName() + "_SliceD", "SliceD")),
                          return INTERNAL_ERROR);
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(gru_node);
  auto output_desc1 = op.GetOutputDesc(1);
  std::vector<int64_t> dims = output_desc1.GetShape().GetDims();
  ge::GeShape input_shape(dims);
  std::vector<int64_t> origin_dims = output_desc1.GetOriginShape().GetDims();
  ge::GeShape origin_shape(origin_dims);
  ge::Format data_format = output_desc1.GetFormat();
  ge::DataType data_type = output_desc1.GetDataType();
  auto ret = new_desc->AddInputDesc(GeTensorDesc(input_shape, data_format, data_type));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT("GRUFusionPass", "CreateSliceNode AddInputDesc fail"), return FAILED);
  auto input_desc = new_desc->GetInputDesc(0);
  input_desc.SetOriginShape(origin_shape);
  input_desc.SetOriginDataType(data_type);
  input_desc.SetOriginFormat(output_desc1.GetOriginFormat());
  new_desc->UpdateInputDesc(0, input_desc);
  int dims_size = origin_dims.size();
  std::vector<int64_t> offsets(dims_size, 0);
  offsets[0] = origin_dims[0] - 1;
  std::vector<int64_t> origin_output_dims = {1};
  for (int i = 1; i < dims_size; ++i) {
    origin_output_dims.push_back(origin_dims[i]);
  }
  ge::GeShape origin_output_shape(origin_output_dims);
  std::vector<int64_t> output_dims = {1};
  for (size_t i = 1; i < dims.size(); ++i) {
    output_dims.push_back(dims[i]);
  }
  ge::GeShape output_shape(output_dims);
  ret = new_desc->AddOutputDesc(GeTensorDesc(output_shape, data_format, data_type));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT("GRUFusionPass", "CreateSliceNode AddOutputDesc fail"), return FAILED);
  auto output_desc = new_desc->GetOutputDesc(0);
  output_desc.SetOriginShape(origin_output_shape);
  output_desc.SetOriginDataType(data_type);
  output_desc.SetOriginFormat(output_desc1.GetOriginFormat());
  new_desc->UpdateOutputDesc(0, output_desc);
  AttrUtils::SetListInt(new_desc, "offsets", offsets);
  AttrUtils::SetListInt(new_desc, "size", origin_output_dims);
  new_node = graph.AddNode(new_desc);
  return SUCCESS;
}

Status GRUFusionPass::AddTransposNode(ge::NodePtr gruNode, int anchorIndex, ge::ComputeGraph& graph) {
  ge::NodePtr weightNode = gruNode->GetInDataAnchor(anchorIndex)->GetPeerOutAnchor()->GetOwnerNode();
  std::shared_ptr<ge::OpDesc> transposeOpdesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (transposeOpdesc = std::make_shared<ge::OpDesc>(weightNode->GetName() + "_transpose_b", "TransposeD")),
      return FAILED);

  vector<int64_t> perm = {0, 2, 1};
  FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(transposeOpdesc, "perm", perm),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Set perm to %s failed.", transposeOpdesc->GetName().c_str()),
                    return FAILED);

  ge::GeTensorDesc inputDesc = weightNode->GetOpDesc()->GetOutputDesc(0).Clone();
  std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
  FUSION_PASS_CHECK(dims.size() != 3, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "weight dim size is not 3."), return FAILED);
  std::vector<int64_t> newDim = {dims[0], dims[2], dims[1]};

  ge::GeTensorDesc outputDesc = gruNode->GetOpDesc()->GetInputDesc(anchorIndex).Clone();
  outputDesc.SetOriginShape(GeShape(newDim));
  outputDesc.SetShape(GeShape(newDim));

  FUSION_PASS_CHECK(transposeOpdesc->AddInputDesc("x", inputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "%s add inputDesc failed.", transposeOpdesc->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(transposeOpdesc->AddOutputDesc("y", outputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "%s add outputDesc failed.", transposeOpdesc->GetName().c_str()),
                    return FAILED);

  ge::NodePtr transposeNode = graph.AddNode(transposeOpdesc);

  ge::OutDataAnchorPtr src = weightNode->GetOutDataAnchor(0);
  ge::InDataAnchorPtr dst = gruNode->GetInDataAnchor(anchorIndex);

  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(src, dst) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove %s input edge error", gruNode->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(src, transposeNode->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            weightNode->GetName().c_str(), transposeNode->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(transposeNode->GetOutDataAnchor(0), dst) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            transposeNode->GetName().c_str(), gruNode->GetName().c_str()),
                    return FAILED);
  return SUCCESS;
}

std::vector<int64_t> GRUFusionPass::RemoveNumDirectionsDim(const std::vector<int64_t>& dims, bool isReverse) {
  std::vector<int64_t> res;
  if (isReverse) {
    for (int i = dims.size() - 1; i > 0; --i) {
      res.push_back(dims[i]);
    }
    return res;
  }
  for (size_t i = 1; i < dims.size(); ++i) {
    res.push_back(dims[i]);
  }
  return res;
}

std::vector<int64_t> GRUFusionPass::ProcessOutputDim(const std::vector<int64_t>& dims) {
  std::vector<int64_t> res;
  int n = dims.size();
  FUSION_PASS_CHECK(n < 2, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dim size less then 2."), return res);
  int64_t numStep = dims[0];
  int64_t last = dims[n - 1];
  int64_t second = dims[n - 2];
  res.push_back(numStep);
  res.push_back(second);
  res.push_back(last);
  return res;
}

void GRUFusionPass::ProcessNZFormat(std::vector<int64_t>& dims) {
  int n = dims.size();
  FUSION_PASS_CHECK(n < 2, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dim size less then 2."), return );
  int64_t first = dims[n - 1];
  int64_t second = dims[n - 2];
  dims[n - 1] = (second + 15) / 16;
  dims[n - 2] = (first + 15) / 16;
  dims.push_back(16);
  dims.push_back(16);
}

void GRUFusionPass::ProcessZFormat(std::vector<int64_t>& dims) {
  for (auto& elem : dims) {
    elem = (elem + 15) / 16;
  }
  dims.push_back(16);
  dims.push_back(16);
}

REGISTER_PASS("GRUFusionPass", BUILT_IN_GRAPH_PASS, GRUFusionPass);
}  // namespace fe