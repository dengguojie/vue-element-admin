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
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include "external/graph/operator_factory.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/fusion_common/graph_pass_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "fp16_t.hpp"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "gru_fusion_pass.h"


using namespace ge;
namespace fe {
static const char* FUSED_NODE = "CommonGRU";
static const std::string PATTERN_FUSED_NODE = "CommonGRU";
static const std::string GRU_DEFAULT_DIRECTION = "UNIDIRECTIONAL";
static const std::string GRU_BIDI_DIRECTION = "REDIRECTIONAL";
static const std::string ATTR_NAME_OP_INFER_DEPENDS = "_op_infer_depends";

static const int X_INDEX = 0;
static const int W_INDEX = 1;
static const int R_INDEX = 2;
static const int B_INDEX = 3;
static const int SEQUENCE_LENS_INDEX = 4;
static const int INITIAL_H_INDEX = 5;

static const int WEIGHT_INPUT_INDEX = 1;
static const int WEIGHT_HIDDEN_INDEX = 2;
static const int BIAS_INPUT_INDEX = 3;
static const int BIAS_HIDDEN_INDEX = 4;
static const int SEQ_LENGTH_INDEX = 5;
static const int INIT_H_INDEX = 6;

static const int SPLIT_DIM_INDEX = 1;
static const int SIZE_SPLITS_INDEX = 2;
static const int SPLIT_FORWARD_INDEX = 0;
static const int SPLIT_REVERSE_INDEX = 1;
static const int NUM_DIRECTIONS_INDEX = 0;
static const int BIAS_DIRECTION_INDEX = 0;
static const int BIAS_CHANNEL_INDEX = 1;
static const int INIT_H_SPLIT_INDEX = 0;
static const int SPLIT_GROUP = 2;
static const int W_INPUT_SIZE = 3;
static const int SINGLE_OUTPUT_INDEX = 0;
static const int Y_OUTPUT_INDEX = 0;
static const int H_OUTPUT_INDEX = 1;
static const int CONCAT_NUM = 2;
static const int INPUT_SIZE_INDEX = 2;

static const int64_t DIM_5HD_UNIT_SIZE = 16;
static const int64_t DIM_5HD_DIV_FACTOR = 15;
static const int64_t BIDI_NUM_DIRECTIONS = 2;
static const int64_t DEFAULT_NUM_DIRECTIONS = 1;


vector<FusionPattern*> GRUFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("CommonGRUFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "new a pattern object failed."), return patterns);

  pattern->AddOpDesc(PATTERN_FUSED_NODE, {FUSED_NODE}).SetOutput(PATTERN_FUSED_NODE);

  patterns.push_back(pattern);

  return patterns;
}

Status GRUFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  // get the NodePtr of GRU
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSED_NODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode is null."), return PARAM_INVALID);

  // get the OpDescPtr of GRU
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode's OpDesc is null."),
                    return PARAM_INVALID);

  FUSION_PASS_CHECK(SUCCESS != CheckParams(fusedDesc),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CheckParams fail"),
                    return FAILED);
  std::string direction;
  ge::AttrUtils::GetStr(fusedDesc, "direction", direction);
  if (direction == "bidirectional") {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "CommonGRU enter bidirectional process!");
    FUSION_PASS_CHECK(SUCCESS != ProcessBidiFusion(graph, fusedNode, fusedDesc, newNodes),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ProcessBidiFusion fail"),
                      return FAILED);
    return SUCCESS;
  }
  OP_LOGI(FUSED_OP_TYPE.c_str(), "CommonGRU enter UNIDIRECTIONAL process!");
  auto gruOp = ge::OperatorFactory::CreateOperator((fusedDesc->GetName() + "_splitD_layer").c_str(), "DynamicGRUV2");
  FUSION_PASS_CHECK(gruOp.IsEmpty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create DynamicGRUV2 operator error"),
                    return FAILED);

  // create DynamicGRUV2 OpDesc
  std::shared_ptr<ge::OpDesc> gruOpDesc = nullptr;
  gruOpDesc = ge::OpDescUtils::GetOpDescFromOperator(gruOp);
  gruOp.BreakConnect();
  FUSION_PASS_CHECK(gruOpDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "gruOpDesc is null, DynamicGRUV2 failed."),
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
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add bias split node failed."),
                      return FAILED);
    // splitNode must not be nullptr when AddBiasSplit returns SUCCESS
    ge::OpDescPtr splitDesc = splitNode->GetOpDesc();
    FUSION_PASS_CHECK(splitDesc == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "splitNode's OpDesc is null."),
                      return PARAM_INVALID);
    GeTensorDesc splitOutDesc = splitDesc->GetOutputDesc(0);
    gruOpDesc->UpdateInputDesc("bias_input", splitOutDesc);
    gruOpDesc->UpdateInputDesc("bias_hidden", splitOutDesc);
  }

  // create DynamicGRUV2 Node
  ge::NodePtr gruNode = graph.AddNode(gruOpDesc);
  FUSION_PASS_CHECK(gruNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "DynamicGRUV2 node is null, fusion failed."),
                    return FAILED);

  // connect bias(splitD) to gru bias input
  if (hasBias) {
    for (int i = 0; i < SPLIT_GROUP; i++) {
      graphStatus status = GraphUtils::AddEdge(splitNode->GetOutDataAnchor(i),
                                               gruNode->GetInDataAnchor(i + BIAS_INPUT_INDEX));
      FUSION_PASS_CHECK(status != GRAPH_SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add slice to conv edge fail"),
                        return FAILED);
    }
  }

  // connect x
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                       gruNode->GetInDataAnchor(0)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add DynamicLSTMV2 edge to fusion node x failed."),
                    return FAILED);

  // connect weight_input
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                                       gruNode->GetInDataAnchor(1)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add DynamicLSTMV2 edge to fusion node weight_input failed."),
                    return FAILED);

  // connect weight_hidden
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(2)->GetPeerOutAnchor(),
                                                       gruNode->GetInDataAnchor(2)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add DynamicLSTMV2 edge to fusion node weight_hidden failed."),
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
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddEdge for slice node fail"),
                      return FAILED);
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
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "add DynamicLSTMV2 edge to fusion node output y failed."),
                      return FAILED);
  }

  for (uint64_t i = 0; i < yhOriTopPeerAnchors.size(); ++i) {
    ge::InDataAnchorPtr oriTopPeerAnchorPtri = yhOriTopPeerAnchors.at(i);
    ge::NodePtr outputNode = oriTopPeerAnchorPtri->GetOwnerNode();
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(output_node->GetOutDataAnchor(anchor_index), oriTopPeerAnchorPtri),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                       "add DynamicLSTMV2 edge to fusion node output y_h failed."),
        return FAILED);
  }

  return SUCCESS;
}

Status GRUFusionPass::CheckParams(const ge::OpDescPtr& fusedDesc) {
  GeTensorDesc wDesc = fusedDesc->GetInputDesc(W_INDEX).Clone();
  std::vector<int64_t> wInputDims = wDesc.GetShape().GetDims();
  std::string direction;

  FUSION_PASS_CHECK(wInputDims.size() != W_INPUT_SIZE,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "weight dim size is not 3."),
                    return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::GetStr(fusedDesc, "direction", direction),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "get attr direction fail."),
                    return PARAM_INVALID);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "direction is %s.", direction.c_str());
  bool directionFlag = (direction == "bidirectional") || (direction == "forward") || (direction == "reverse");
  FUSION_PASS_CHECK(!directionFlag,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "direction attr invalid."),
                    return FAILED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "num_directions is %d.", wInputDims[NUM_DIRECTIONS_INDEX]);
  if (direction == "bidirectional") {
    FUSION_PASS_CHECK(wInputDims[NUM_DIRECTIONS_INDEX] != BIDI_NUM_DIRECTIONS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "num_directions attr invalid."),
                      return FAILED);
  } else {
    FUSION_PASS_CHECK(wInputDims[NUM_DIRECTIONS_INDEX] != DEFAULT_NUM_DIRECTIONS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "num_directions attr invalid."),
                      return FAILED);
  }

  return SUCCESS;
}

void GRUFusionPass::SetTensorDescription(ge::GeTensorDesc& tensorDesc, vector<int64_t>& dims,
                                         const ge::Format& format, const ge::DataType& dtype) {
  ge::GeShape shape(dims);
  tensorDesc.SetShape(shape);
  tensorDesc.SetDataType(dtype);
  tensorDesc.SetFormat(format);
  tensorDesc.SetOriginShape(shape);
  tensorDesc.SetOriginDataType(dtype);
  tensorDesc.SetOriginFormat(format);
}

Status GRUFusionPass::ProcessBidiFusion(ge::ComputeGraph& graph, ge::NodePtr& fusedNode, ge::OpDescPtr& fusedDesc,
                                        vector<ge::NodePtr>& newNodes) {
  std::string dynamicGruV2OpForwardName = fusedDesc->GetName() + "/DynamicGRUV2" + "Forward";
  auto dynamicGruV2OpForward = ge::OperatorFactory::CreateOperator(dynamicGruV2OpForwardName.c_str(), "DynamicGRUV2");
  FUSION_PASS_CHECK(dynamicGruV2OpForward.IsEmpty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CommonGRU create Forward operator error"),
                    return FAILED);
  auto dynamicGruV2DescForward = ge::OpDescUtils::GetOpDescFromOperator(dynamicGruV2OpForward);
  dynamicGruV2OpForward.BreakConnect();

  std::string dynamicGruV2OpReverseName = fusedDesc->GetName() + "/DynamicGRUV2" + "Reverse";
  auto dynamicGruV2OpReverse = ge::OperatorFactory::CreateOperator(dynamicGruV2OpReverseName.c_str(), "DynamicGRUV2");
  FUSION_PASS_CHECK(dynamicGruV2OpReverse.IsEmpty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CommonGRU create Reverse operator error"),
                    return FAILED);
  auto dynamicGruV2DescReverse = ge::OpDescUtils::GetOpDescFromOperator(dynamicGruV2OpReverse);
  dynamicGruV2OpReverse.BreakConnect();

  // process x
  GeTensorDesc xDesc = fusedDesc->GetInputDesc(X_INDEX).Clone();
  std::vector<int64_t> xInputDims = xDesc.GetShape().GetDims();
  // w shape [num_directions, 3*hidden_size, input_size]
  int64_t inputSize = fusedDesc->GetInputDesc(W_INDEX).GetShape().GetDim(INPUT_SIZE_INDEX);
  xInputDims[INPUT_SIZE_INDEX] = inputSize;
  GeShape xInputShape(xInputDims);
  xDesc.SetOriginShape(xInputShape);
  xDesc.SetShape(xInputShape);
  dynamicGruV2DescForward->UpdateInputDesc("x", xDesc);
  dynamicGruV2DescReverse->UpdateInputDesc("x", xDesc);

  // process weight_input
  ge::NodePtr weightInputSplitNode = nullptr;
  FUSION_PASS_CHECK(SUCCESS != AddBidiWeightSplitNode(graph, fusedNode, W_INDEX, weightInputSplitNode, newNodes),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CommonGRU add weight_input node failed."),
                    return FAILED);
  ge::OpDescPtr weightInputSplitDesc = weightInputSplitNode->GetOpDesc();
  FUSION_PASS_CHECK(nullptr == weightInputSplitDesc,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "weight_input splitNode's OpDesc is null."),
                    return PARAM_INVALID);
  GeTensorDesc weightInputOutDesc = weightInputSplitDesc->GetOutputDesc(SPLIT_FORWARD_INDEX);
  dynamicGruV2DescForward->UpdateInputDesc("weight_input", weightInputOutDesc);
  dynamicGruV2DescReverse->UpdateInputDesc("weight_input", weightInputOutDesc);

  // process weight_hidden
  ge::NodePtr weightHiddenSplitNode = nullptr;
  FUSION_PASS_CHECK(SUCCESS != AddBidiWeightSplitNode(graph, fusedNode, R_INDEX, weightHiddenSplitNode, newNodes),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CommonGRU add weight_hidden failed."),
                    return FAILED);
  ge::OpDescPtr weightHiddenSplitDesc = weightHiddenSplitNode->GetOpDesc();
  FUSION_PASS_CHECK(nullptr == weightHiddenSplitDesc,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "weight_hidden splitNode's OpDesc is null."),
                    return PARAM_INVALID);
  GeTensorDesc weightHiddenOutDesc = weightHiddenSplitDesc->GetOutputDesc(SPLIT_FORWARD_INDEX);
  dynamicGruV2DescForward->UpdateInputDesc("weight_hidden", weightHiddenOutDesc);
  dynamicGruV2DescReverse->UpdateInputDesc("weight_hidden", weightHiddenOutDesc);

  // process bias
  bool hasBias = fusedDesc->MutableInputDesc("b") != nullptr;
  OP_LOGI(FUSED_OP_TYPE.c_str(), "CommonGRU hasBias is %d", hasBias);
  ge::NodePtr inputBiasSplitNode = nullptr;
  ge::NodePtr hiddenBiasSplitNode = nullptr;
  if (hasBias) {
    // add bias split Node
    FUSION_PASS_CHECK(SUCCESS != AddBidiBiasSplitNode(graph, fusedNode, B_INDEX, inputBiasSplitNode,
                                                      hiddenBiasSplitNode, newNodes),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CommonGRU add bias split node failed."),
                      return FAILED);

    // splitNode must not be nullptr when AddBiasSplit returns SUCCESS
    ge::OpDescPtr inputBiasSplitDesc = inputBiasSplitNode->GetOpDesc();
    ge::OpDescPtr hiddenBiasSplitDesc = hiddenBiasSplitNode->GetOpDesc();
    FUSION_PASS_CHECK(nullptr == inputBiasSplitDesc,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "splitNode's OpDesc is null."),
                      return PARAM_INVALID);
    FUSION_PASS_CHECK(nullptr == hiddenBiasSplitDesc,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "splitNode's OpDesc is null."),
                      return PARAM_INVALID);

    GeTensorDesc inputBiasOutDesc = inputBiasSplitDesc->GetOutputDesc(SPLIT_FORWARD_INDEX);
    dynamicGruV2DescForward->UpdateInputDesc("bias_input", inputBiasOutDesc);
    dynamicGruV2DescReverse->UpdateInputDesc("bias_input", inputBiasOutDesc);

    GeTensorDesc hiddenBiasOutDesc = hiddenBiasSplitDesc->GetOutputDesc(SPLIT_FORWARD_INDEX);
    dynamicGruV2DescForward->UpdateInputDesc("bias_hidden", hiddenBiasOutDesc);
    dynamicGruV2DescReverse->UpdateInputDesc("bias_hidden", hiddenBiasOutDesc);
  }

  // process seq_length
  bool hasSeqLength = fusedDesc->MutableInputDesc("sequence_lens") != nullptr;
  OP_LOGI(FUSED_OP_TYPE.c_str(), "CommonGRU hasSeqLength is %d", hasSeqLength);
  if (hasSeqLength) {
    ge::GeTensorDesc seqLengthDesc = *fusedDesc->MutableInputDesc("sequence_lens");
    dynamicGruV2DescForward->UpdateInputDesc("seq_length", seqLengthDesc);
    dynamicGruV2DescReverse->UpdateInputDesc("seq_length", seqLengthDesc);
  }

  // process init_h
  bool hasInitH = fusedDesc->MutableInputDesc("initial_h") != nullptr;
  OP_LOGI(FUSED_OP_TYPE.c_str(), "CommonGRU hasInitH is %d", hasInitH);
  ge::NodePtr initHSplitNode = nullptr;
  if (hasInitH) {
    FUSION_PASS_CHECK(SUCCESS != AddBidiInitHSplitNode(graph, fusedNode, INITIAL_H_INDEX, initHSplitNode, newNodes),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CommonGRU add init_h split node failed."),
                      return FAILED);
    ge::OpDescPtr initHSplitDesc = initHSplitNode->GetOpDesc();
    FUSION_PASS_CHECK(nullptr == initHSplitDesc,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "init_h splitNode's OpDesc is null."),
                      return PARAM_INVALID);
    GeTensorDesc initHOutDesc = initHSplitDesc->GetOutputDesc(SPLIT_FORWARD_INDEX);
    dynamicGruV2DescForward->UpdateInputDesc("init_h", initHOutDesc);
    dynamicGruV2DescReverse->UpdateInputDesc("init_h", initHOutDesc);
  }

  // process direction attr
  ge::AttrUtils::SetStr(dynamicGruV2DescForward, "direction", GRU_DEFAULT_DIRECTION);
  ge::AttrUtils::SetStr(dynamicGruV2DescReverse, "direction", GRU_BIDI_DIRECTION);

  // process output
  GeTensorDesc outputYDesc = fusedDesc->GetOutputDesc(Y_OUTPUT_INDEX);
  std::vector<int64_t> yDims = ProcessOutputDim(outputYDesc.GetShape().GetDims());
  GeShape yOriginShape(yDims);
  outputYDesc.SetOriginShape(yOriginShape);
  outputYDesc.SetShape(yOriginShape);

  dynamicGruV2DescForward->UpdateOutputDesc("y", outputYDesc);
  dynamicGruV2DescForward->UpdateOutputDesc("output_h", outputYDesc);
  dynamicGruV2DescForward->UpdateOutputDesc("update", outputYDesc);
  dynamicGruV2DescForward->UpdateOutputDesc("reset", outputYDesc);
  dynamicGruV2DescForward->UpdateOutputDesc("new", outputYDesc);
  dynamicGruV2DescForward->UpdateOutputDesc("hidden_new", outputYDesc);

  dynamicGruV2DescReverse->UpdateOutputDesc("y", outputYDesc);
  dynamicGruV2DescReverse->UpdateOutputDesc("output_h", outputYDesc);
  dynamicGruV2DescReverse->UpdateOutputDesc("update", outputYDesc);
  dynamicGruV2DescReverse->UpdateOutputDesc("reset", outputYDesc);
  dynamicGruV2DescReverse->UpdateOutputDesc("new", outputYDesc);
  dynamicGruV2DescReverse->UpdateOutputDesc("hidden_new", outputYDesc);

  // create DynamicGRUV2 forward node
  ge::NodePtr dynamicGruV2ForwardNode = graph.AddNode(dynamicGruV2DescForward);
  FUSION_PASS_CHECK(nullptr == dynamicGruV2ForwardNode,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "DynamicGRUV2 forward node is null, fusion failed."),
                    return FAILED);
  newNodes.push_back(dynamicGruV2ForwardNode);

  // create DynamicGRUV2 reverse node
  ge::NodePtr dynamicGRUV2ReverseNode = graph.AddNode(dynamicGruV2DescReverse);
  FUSION_PASS_CHECK(nullptr == dynamicGRUV2ReverseNode,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "DynamicGRUV2 reverse node is null, fusion failed."),
                    return FAILED);
  newNodes.push_back(dynamicGRUV2ReverseNode);

  // connect x
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(X_INDEX)->GetPeerOutAnchor(),
                                                       dynamicGruV2ForwardNode->GetInDataAnchor(X_INDEX)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "forword add x edge to fusion node failed."),
                    return FAILED);

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(X_INDEX)->GetPeerOutAnchor(),
                                                       dynamicGRUV2ReverseNode->GetInDataAnchor(X_INDEX)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "reverse add x edge to fusion node failed."),
                    return FAILED);

  // connect weight_input
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(weightInputSplitNode->GetOutDataAnchor(SPLIT_FORWARD_INDEX),
                                                       dynamicGruV2ForwardNode->GetInDataAnchor(WEIGHT_INPUT_INDEX)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "forward add wi edge to fusion node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(weightInputSplitNode->GetOutDataAnchor(SPLIT_REVERSE_INDEX),
                                                       dynamicGRUV2ReverseNode->GetInDataAnchor(WEIGHT_INPUT_INDEX)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "reverse add wi edge to fusion node failed."),
                    return FAILED);

  // connect weight_hidden
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(weightHiddenSplitNode->GetOutDataAnchor(SPLIT_FORWARD_INDEX),
                                                       dynamicGruV2ForwardNode->GetInDataAnchor(WEIGHT_HIDDEN_INDEX)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "forward add wh edge to fusion node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(weightHiddenSplitNode->GetOutDataAnchor(SPLIT_REVERSE_INDEX),
                                                       dynamicGRUV2ReverseNode->GetInDataAnchor(WEIGHT_HIDDEN_INDEX)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "reverse add wh edge to fusion node failed."),
                    return FAILED);

  // connect bias
  if (hasBias) {
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(inputBiasSplitNode->GetOutDataAnchor(SPLIT_FORWARD_INDEX),
                                                         dynamicGruV2ForwardNode->GetInDataAnchor(BIAS_INPUT_INDEX)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "forward add input bias edge to fusion node failed."),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(inputBiasSplitNode->GetOutDataAnchor(SPLIT_REVERSE_INDEX),
                                                         dynamicGRUV2ReverseNode->GetInDataAnchor(BIAS_INPUT_INDEX)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "reverse add input bias edge to fusion node failed."),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(hiddenBiasSplitNode->GetOutDataAnchor(SPLIT_FORWARD_INDEX),
                                                         dynamicGruV2ForwardNode->GetInDataAnchor(BIAS_HIDDEN_INDEX)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "forward add hidden bias edge to fusion node failed."),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(hiddenBiasSplitNode->GetOutDataAnchor(SPLIT_REVERSE_INDEX),
                                                         dynamicGRUV2ReverseNode->GetInDataAnchor(BIAS_HIDDEN_INDEX)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "reverse add hidden bias edge to fusion node failed."),
                      return FAILED);
  }

  // connect seq_length
  if (hasSeqLength) {
    ge::NodePtr seqLengthNode = fusedNode->GetInDataAnchor(SEQUENCE_LENS_INDEX)->GetPeerOutAnchor()->GetOwnerNode();
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(seqLengthNode->GetOutDataAnchor(SINGLE_OUTPUT_INDEX),
                                                         dynamicGruV2ForwardNode->GetInDataAnchor(SEQ_LENGTH_INDEX)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "forward add seqLength edge to fusion node failed."),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(seqLengthNode->GetOutDataAnchor(SINGLE_OUTPUT_INDEX),
                                                         dynamicGRUV2ReverseNode->GetInDataAnchor(SEQ_LENGTH_INDEX)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "reverse add seqLength edge to fusion node failed."),
                      return FAILED);
  }

  // connect init_h
  if (hasInitH) {
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(initHSplitNode->GetOutDataAnchor(SPLIT_FORWARD_INDEX),
                                                         dynamicGruV2ForwardNode->GetInDataAnchor(INIT_H_INDEX)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "forward add init_h edge to fusion node failed."),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(initHSplitNode->GetOutDataAnchor(SPLIT_REVERSE_INDEX),
                                                         dynamicGRUV2ReverseNode->GetInDataAnchor(INIT_H_INDEX)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "reverse add init_h edge to fusion node failed."),
                      return FAILED);
  }

  // add concat
  FUSION_PASS_CHECK(SUCCESS != AddExpandDimsAndConcatNode(graph, fusedNode, dynamicGruV2ForwardNode,
                                                          dynamicGRUV2ReverseNode, outputYDesc, newNodes),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create ExpandDimsY Op operator error"),
                    return FAILED);

  // add slice
  FUSION_PASS_CHECK(SUCCESS != AddSliceAndConcatNode(graph, fusedNode, dynamicGruV2ForwardNode, dynamicGRUV2ReverseNode,
                                                     outputYDesc, newNodes, H_OUTPUT_INDEX),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create slice H Op operator error"),
                    return FAILED);

  // unlink all control input of CommonGRU
  if (fusedNode->GetInControlAnchor() != nullptr) {
    fusedNode->GetInControlAnchor()->UnlinkAll();
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

  return SUCCESS;
}

Status GRUFusionPass::SetSplitVNodeInfo(ge::GeTensorDesc& tensorDesc, ge::OpDescPtr& outOpDesc, vector<int64_t>& dimIn,
                                        vector<int32_t>& inputDims) {
  SetTensorDescription(tensorDesc, dimIn, ge::FORMAT_ND, ge::DT_INT32);
  ge::GeTensorPtr tensorPtr = nullptr;
  FUSION_PASS_MAKE_SHARED((tensorPtr = std::make_shared<ge::GeTensor>(tensorDesc,
                                                                      reinterpret_cast<uint8_t *>(inputDims.data()),
                                                                      inputDims.size() * sizeof(int32_t))),
                          return FAILED);
  outOpDesc = ge::OpDescUtils::CreateConstOp(tensorPtr);
  FUSION_PASS_CHECK(nullptr == outOpDesc,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "SetSplitVNodeInfo CreateConstOp failed."),
                    return FAILED);
  return SUCCESS;
}

Status GRUFusionPass::AddSplitVNode(const std::string& nodeName, ge::GeTensorDesc& inputDesc,
                                    ge::GeTensorDesc& outputDesc, ge::NodePtr& splitNode, ge::NodePtr& peerOutNode,
                                    vector<int32_t>& splitDimAxis, vector<int32_t>& sizeSplitAxis, int splitIndex,
                                    vector<ge::NodePtr>& newNodes, ge::ComputeGraph& graph) {
  // get splitDesc
  std::shared_ptr<ge::OpDesc> splitDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((splitDesc = std::make_shared<ge::OpDesc>(nodeName + "/DynamicGRUV2_split_v", "SplitV")),
                          return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(splitDesc, "num_split", SPLIT_GROUP),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Set num_split to %s failed.", splitDesc->GetName().c_str()),
                    return FAILED);

  ge::GeTensorDesc splitDimDesc;
  ge::OpDescPtr splitDimOutDesc;
  std::vector<int64_t> splitDimIn = {};
  FUSION_PASS_CHECK(SUCCESS != SetSplitVNodeInfo(splitDimDesc, splitDimOutDesc, splitDimIn, splitDimAxis),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "splitDim set info failed."),
                    return FAILED);

  ge::GeTensorDesc sizeSplitDesc;
  ge::OpDescPtr sizeSplitOutDesc;
  std::vector<int64_t> sizeSplitIn = {SPLIT_GROUP};
  FUSION_PASS_CHECK(SUCCESS != SetSplitVNodeInfo(sizeSplitDesc, sizeSplitOutDesc, sizeSplitIn, sizeSplitAxis),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "sizeSplit set info failed."),
                    return FAILED);

  // split_v
  FUSION_PASS_CHECK(SUCCESS != splitDesc->AddInputDesc("x", inputDesc),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "SplitV add x input failed."),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != splitDesc->AddInputDesc("size_splits", sizeSplitDesc),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "SplitV add size_splits input failed."),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != splitDesc->AddInputDesc("split_dim", splitDimDesc),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "SplitV add split_dim input failed."),
                    return FAILED);
  for (int i = 0; i < SPLIT_GROUP; i++) {
    FUSION_PASS_CHECK(SUCCESS != splitDesc->AddOutputDesc(outputDesc),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "SplitV add SplitD output failed."),
                      return FAILED);
  }

  // create node
  splitNode = graph.AddNode(splitDesc);
  ge::NodePtr sizeSplitNode = graph.AddNode(sizeSplitOutDesc);
  ge::NodePtr splitDimNode = graph.AddNode(splitDimOutDesc);
  FUSION_PASS_CHECK(nullptr == splitNode,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add SplitD node is null, fusion failed."),
                    return FAILED);
  FUSION_PASS_CHECK(nullptr == sizeSplitNode,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add sizeSplitNode node is null, fusion failed."),
                    return FAILED);
  FUSION_PASS_CHECK(nullptr == splitDimNode,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add splitDimNode node is null, fusion failed."),
                    return FAILED);
  newNodes.push_back(splitNode);
  newNodes.push_back(sizeSplitNode);
  newNodes.push_back(splitDimNode);

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(peerOutNode->GetOutDataAnchor(splitIndex),
                                                       splitNode->GetInDataAnchor(X_INDEX)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Add edge between node %s. and node %s failed.",
                                                    peerOutNode->GetName().c_str(), splitNode->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(sizeSplitNode->GetOutDataAnchor(SINGLE_OUTPUT_INDEX),
                                                       splitNode->GetInDataAnchor(SPLIT_DIM_INDEX)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Add edge between node %s. and node %s failed.",
                                                    sizeSplitNode->GetName().c_str(), splitNode->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(splitDimNode->GetOutDataAnchor(SINGLE_OUTPUT_INDEX),
                                                       splitNode->GetInDataAnchor(SIZE_SPLITS_INDEX)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Add edge between node %s. and node %s failed.",
                                                    splitDimNode->GetName().c_str(), splitNode->GetName().c_str()),
                    return FAILED);
  return SUCCESS;
}

Status GRUFusionPass::AddBidiWeightSplitNode(ge::ComputeGraph& graph, const ge::NodePtr& fusedNode,
                                             int weightIndex, ge::NodePtr& splitNode,
                                             vector<ge::NodePtr>& newNodes) {
  ge::NodePtr weightNode = fusedNode->GetInDataAnchor(weightIndex)->GetPeerOutAnchor()->GetOwnerNode();

  // get transposeDesc
  std::shared_ptr<ge::OpDesc> transposeDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((transposeDesc = std::make_shared<ge::OpDesc>(weightNode->GetName() + "_transpose_b",
                                                                        "TransposeD")),
                          return FAILED);
  vector<int64_t> perm = {0, 2, 1};
  FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(transposeDesc, "perm", perm),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Set perm to %s failed.", transposeDesc->GetName().c_str()),
                    return FAILED);

  ge::GeTensorDesc inputDesc = weightNode->GetOpDesc()->GetOutputDesc(0).Clone();
  std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
  FUSION_PASS_CHECK(dims.size() != W_INPUT_SIZE,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "weight dim size is not 3."),
                    return FAILED);

  ge::GeTensorDesc midDesc = weightNode->GetOpDesc()->GetOutputDesc(0).Clone();
  std::vector<int64_t> newDim = {dims[0], dims[2], dims[1]};
  midDesc.SetOriginShape(GeShape(newDim));
  midDesc.SetShape(GeShape(newDim));

  ge::GeTensorDesc outputDesc = fusedNode->GetOpDesc()->GetInputDesc(weightIndex).Clone();
  std::vector<int64_t> outDim = {dims[2], dims[1]};
  outputDesc.SetOriginShape(GeShape(outDim));
  outputDesc.SetShape(GeShape(outDim));

  // transpose
  FUSION_PASS_CHECK(SUCCESS != transposeDesc->AddInputDesc("x", inputDesc),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "%s add inputDesc failed.", transposeDesc->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != transposeDesc->AddOutputDesc("y", midDesc),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "%s add outputDesc failed.", transposeDesc->GetName().c_str()),
                    return FAILED);

  ge::NodePtr transposeNode = graph.AddNode(transposeDesc);
  FUSION_PASS_CHECK(nullptr == transposeNode,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add TransposeD node is null, fusion failed."),
                    return FAILED);
  newNodes.push_back(transposeNode);

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(weightNode->GetOutDataAnchor(SINGLE_OUTPUT_INDEX),
                                                          fusedNode->GetInDataAnchor(weightIndex)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "remove %s input edge error", fusedNode->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(weightNode->GetOutDataAnchor(SINGLE_OUTPUT_INDEX),
                                                       transposeNode->GetInDataAnchor(X_INDEX)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Add edge between node %s. and node %s failed.",
                                                   weightNode->GetName().c_str(), transposeNode->GetName().c_str()),
                    return FAILED);

  // add split_v node
  std::vector<int32_t> splitDimAxis = {0};
  std::vector<int32_t> sizeSplitAxis = {1, 1};
  FUSION_PASS_CHECK(SUCCESS != AddSplitVNode(weightNode->GetName(), midDesc, outputDesc, splitNode, transposeNode,
                                             splitDimAxis, sizeSplitAxis, 0, newNodes, graph),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "w add split node failed."),
                    return FAILED);

  return SUCCESS;
}

Status GRUFusionPass::AddBidiBiasSplitNode(ge::ComputeGraph& graph, const ge::NodePtr& fusedNode, int biasIndex,
                                           ge::NodePtr& inputSplitNode, ge::NodePtr& hiddenSplitNode,
                                           vector<ge::NodePtr>& newNodes) {
  ge::NodePtr biasNode = fusedNode->GetInDataAnchor(biasIndex)->GetPeerOutAnchor()->GetOwnerNode();

  ge::GeTensorDesc inputDesc = biasNode->GetOpDesc()->GetOutputDesc(0).Clone();
  std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
  FUSION_PASS_CHECK(dims.size() != 2,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "bias dim size is not 2."), return FAILED);

  ge::GeTensorDesc midDesc = biasNode->GetOpDesc()->GetOutputDesc(0).Clone();
  std::vector<int64_t> midDim = {dims[0], dims[1] / SPLIT_GROUP};
  midDesc.SetOriginShape(GeShape(midDim));
  midDesc.SetShape(GeShape(midDim));

  ge::GeTensorDesc outputDesc = fusedNode->GetOpDesc()->GetInputDesc(biasIndex).Clone();
  std::vector<int64_t> outDim = {dims[1] / SPLIT_GROUP};
  outputDesc.SetOriginShape(GeShape(outDim));
  outputDesc.SetShape(GeShape(outDim));

  // add split_v node
  ge::NodePtr biasSplitNode;
  std::vector<int32_t> biasSplitDimAxis = {1};
  std::vector<int32_t> biasSizeSplitAxis = {static_cast<int32_t>(dims[1]) / SPLIT_GROUP,
                                            static_cast<int32_t>(dims[1]) / SPLIT_GROUP};
  FUSION_PASS_CHECK(SUCCESS != AddSplitVNode(biasNode->GetName() + "_b", inputDesc, midDesc, biasSplitNode, biasNode,
                                             biasSplitDimAxis, biasSizeSplitAxis, 0, newNodes, graph),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "b add split node failed."),
                    return FAILED);

  std::vector<int32_t> numSplitDimAxis = {0};
  std::vector<int32_t> numSizeSplitAxis = {1, 1};
  FUSION_PASS_CHECK(SUCCESS != AddSplitVNode(biasNode->GetName() + "_i", midDesc, outputDesc, inputSplitNode,
                                             biasSplitNode, numSplitDimAxis, numSizeSplitAxis, 0, newNodes, graph),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "b_i add split node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != AddSplitVNode(biasNode->GetName() + "_h", midDesc, outputDesc, hiddenSplitNode,
                                             biasSplitNode, numSplitDimAxis, numSizeSplitAxis, 1, newNodes,graph),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "b_h add split node failed."),
                    return FAILED);

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(biasNode->GetOutDataAnchor(SINGLE_OUTPUT_INDEX),
                                                          fusedNode->GetInDataAnchor(biasIndex)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "remove %s input edge error", fusedNode->GetName().c_str()),
                    return FAILED);

  return SUCCESS;
}

Status GRUFusionPass::AddBidiInitHSplitNode(ge::ComputeGraph& graph, const ge::NodePtr& fusedNode,
                                            int initHIndex, ge::NodePtr& splitNode, vector<ge::NodePtr>& newNodes) {
  ge::NodePtr initHNode = fusedNode->GetInDataAnchor(initHIndex)->GetPeerOutAnchor()->GetOwnerNode();
  string nodeType = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(initHNode);

  ge::GeTensorDesc inputDesc = initHNode->GetOpDesc()->GetOutputDesc(SINGLE_OUTPUT_INDEX).Clone();
  std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
  FUSION_PASS_CHECK(dims.size() != 3,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "init_h dim size is not 3."),
                    return FAILED);

  ge::GeTensorDesc outputDesc = fusedNode->GetOpDesc()->GetInputDesc(initHIndex).Clone();
  std::vector<int64_t> outDim = {dims[1], dims[2]};
  outputDesc.SetOriginShape(GeShape(outDim));
  outputDesc.SetShape(GeShape(outDim));

  if (nodeType == "Const" || nodeType == "Constant") {
    // add split_v node
    std::vector<int32_t> splitDimAxis = {0};
    std::vector<int32_t> sizeSplitAxis = {1, 1};
    FUSION_PASS_CHECK(SUCCESS != AddSplitVNode(initHNode->GetName(), inputDesc, outputDesc, splitNode, initHNode,
                                               splitDimAxis, sizeSplitAxis, SINGLE_OUTPUT_INDEX, newNodes, graph),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "w add split node failed."),
                      return FAILED);
    if (initHNode->GetInControlAnchor() != nullptr) {
      initHNode->GetInControlAnchor()->UnlinkAll();
    }
  } else {
    // get splitDesc
    std::shared_ptr<ge::OpDesc> splitDesc = nullptr;
    FUSION_PASS_MAKE_SHARED((splitDesc = std::make_shared<ge::OpDesc>(initHNode->GetName() + "/DynamicGRUV2_h_split",
                                                                      "SplitD")),
                            return FAILED);
    FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(splitDesc, "split_dim", INIT_H_SPLIT_INDEX),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "Set split_dim to %s failed.", splitDesc->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(splitDesc, "num_split", SPLIT_GROUP),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "Set num_split to %s failed.", splitDesc->GetName().c_str()),
                      return FAILED);

    FUSION_PASS_CHECK(splitDesc->AddInputDesc(inputDesc) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add init_h SplitD input failed."),
                      return FAILED);
    for (int i = 0; i < SPLIT_GROUP; i++) {
      FUSION_PASS_CHECK(splitDesc->AddOutputDesc(outputDesc) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add init_h SplitD output failed."),
                        return FAILED);
    }

    // create node
    splitNode = graph.AddNode(splitDesc);
    FUSION_PASS_CHECK(nullptr == splitNode,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add SplitD node is null, fusion failed."),
                      return FAILED);
    newNodes.push_back(splitNode);

    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(initHNode->GetOutDataAnchor(SINGLE_OUTPUT_INDEX),
                                              splitNode->GetInDataAnchor(X_INDEX)) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "Add edge between node %s. and node %s failed.",
                                                     initHNode->GetName().c_str(), splitNode->GetName().c_str()),
                      return FAILED);
  }

  // remove and add edge
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(initHNode->GetOutDataAnchor(SINGLE_OUTPUT_INDEX),
                                               fusedNode->GetInDataAnchor(initHIndex)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "remove %s input edge error", fusedNode->GetName().c_str()),
                    return FAILED);
  return SUCCESS;
}

Status GRUFusionPass::AddExpandDimsAndConcatNode(ge::ComputeGraph& graph, ge::NodePtr& fusedNode,
                                                 ge::NodePtr& forwardNode, ge::NodePtr& reverseNode,
                                                 ge::GeTensorDesc& outputDesc, vector<ge::NodePtr>& newNodes) {
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  std::string forwardExdName = fusedDesc->GetName() + "/ExpandDims" + "_Forward";
  std::string reverseExdName = fusedDesc->GetName() + "/ExpandDims" + "_Reverse";
  auto forwardExdOp = ge::OperatorFactory::CreateOperator(forwardExdName.c_str(), "ExpandDims");
  auto reverseExdOp = ge::OperatorFactory::CreateOperator(reverseExdName.c_str(), "ExpandDims");

  FUSION_PASS_CHECK(forwardExdOp.IsEmpty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create ExpandDims Forward Op error"),
                    return FAILED);
  auto forwardExdDesc = ge::OpDescUtils::GetOpDescFromOperator(forwardExdOp);
  forwardExdOp.BreakConnect();

  FUSION_PASS_CHECK(reverseExdOp.IsEmpty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create ExpandDims Reverse Op error"),
                    return FAILED);
  auto reverseExdDesc = ge::OpDescUtils::GetOpDescFromOperator(reverseExdOp);
  reverseExdOp.BreakConnect();

  // create axis tensor for ExpandDims
  ge::GeTensorDesc tensorDesc;
  vector<int64_t> dimsIn = {1};
  SetTensorDescription(tensorDesc, dimsIn, ge::FORMAT_ND, ge::DT_INT32);

  ge::GeTensorDesc exdTensorDesc = fusedDesc->GetOutputDesc(0).Clone();
  std::vector<int64_t> dims = exdTensorDesc.GetShape().GetDims();
  dims[1] = 1;
  ge::GeShape outputShape(dims);
  exdTensorDesc.SetShape(outputShape);
  exdTensorDesc.SetOriginShape(outputShape);

  forwardExdDesc->UpdateInputDesc("x", outputDesc);
  forwardExdDesc->UpdateInputDesc("axis", tensorDesc);
  forwardExdDesc->UpdateOutputDesc("y", exdTensorDesc);

  reverseExdDesc->UpdateInputDesc("x", outputDesc);
  reverseExdDesc->UpdateInputDesc("axis", tensorDesc);
  reverseExdDesc->UpdateOutputDesc("y", exdTensorDesc);

  ge::GeTensorPtr axisTensorPtr = nullptr;
  vector<int32_t> axis = {1};
  FUSION_PASS_MAKE_SHARED((axisTensorPtr = std::make_shared<ge::GeTensor>(tensorDesc,
                                                                          reinterpret_cast<uint8_t *>(axis.data()),
                                                                          1 * sizeof(int32_t))),
                          return FAILED);
  ge::OpDescPtr axisDesc = ge::OpDescUtils::CreateConstOp(axisTensorPtr);

  // create ExpandDims and axis node
  ge::NodePtr forwardExdNode = graph.AddNode(forwardExdDesc);
  ge::NodePtr reverseExdNode = graph.AddNode(reverseExdDesc);
  ge::NodePtr axisNode = graph.AddNode(axisDesc);

  FUSION_PASS_CHECK(nullptr == forwardExdNode,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create ExpandDims forward node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(nullptr == reverseExdNode,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create ExpandDims reverse node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(nullptr == axisNode,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "axis node is null, fusion failed."),
                    return FAILED);

  // set attr _op_infer_depends for ExpandDims
  vector<std::string> original_names = {"axis"};
  bool ret = ge::AttrUtils::SetListStr(forwardExdNode->GetOpDesc(), ATTR_NAME_OP_INFER_DEPENDS, original_names);
  ret = ge::AttrUtils::SetListStr(reverseExdNode->GetOpDesc(), ATTR_NAME_OP_INFER_DEPENDS, original_names);
  if (!ret) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "ExpandDimNode set ATTR_NAME_OP_INFER_DEPENDS error.");
  }

  newNodes.push_back(forwardExdNode);
  newNodes.push_back(reverseExdNode);
  newNodes.push_back(axisNode);

  // add forward output y edge to expand_dims node
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(forwardNode->GetOutDataAnchor(0),
                                                       forwardExdNode->GetInDataAnchor(0)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add edge to fusion ExpandDim x failed."),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(axisNode->GetOutDataAnchor(0),
                                                       forwardExdNode->GetInDataAnchor(1)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add axis edge to fusion ExpandDim axis failed."),
                    return FAILED);

  // add reverse output y edge to expand_dims node
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(reverseNode->GetOutDataAnchor(0),
                                                       reverseExdNode->GetInDataAnchor(0)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add edge to fusion ExpandDim x failed."),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(axisNode->GetOutDataAnchor(0),
                                                       reverseExdNode->GetInDataAnchor(1)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add axis edge to fusion ExpandDim axis failed."),
                    return FAILED);

  // create concat node
  std::string concatOpName = fusedDesc->GetName() + "/ConcatD_" + "Y";
  auto concatOp = ge::OperatorFactory::CreateOperator(concatOpName.c_str(), "ConcatD");
  FUSION_PASS_CHECK(concatOp.IsEmpty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create y concat operator error"),
                    return FAILED);
  auto concatDesc = ge::OpDescUtils::GetOpDescFromOperator(concatOp);
  concatOp.BreakConnect();

  ge::GeTensorDesc originTensorDesc = fusedDesc->GetOutputDesc(0);
  concatDesc->AddInputDesc("x0", exdTensorDesc);
  concatDesc->AddInputDesc("x1", exdTensorDesc);
  concatDesc->UpdateOutputDesc("y", originTensorDesc);

  ge::AttrUtils::SetInt(concatDesc, "concat_dim", 1);
  ge::AttrUtils::SetInt(concatDesc, "N", CONCAT_NUM);

  ge::NodePtr concatNode = graph.AddNode(concatDesc);
  newNodes.push_back(concatNode);

  // add ExpandDims edge to concat node
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(forwardExdNode->GetOutDataAnchor(0),
                                                       concatNode->GetInDataAnchor(0)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add expanddims forward edge to fusion concat node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(reverseExdNode->GetOutDataAnchor(0),
                                                       concatNode->GetInDataAnchor(1)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add expanddims backward edge to fusion concat node failed."),
                    return FAILED);

  // add concat edge to output node
  for (InDataAnchorPtr oriTopPeerAnchorPtr : fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(fusedNode->GetOutDataAnchor(0), oriTopPeerAnchorPtr),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "remove concat output edge to output node failed."),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(concatNode->GetOutDataAnchor(0), oriTopPeerAnchorPtr),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "add concat output edge to output node failed."),
                      return FAILED);
  }

  return SUCCESS;
}

Status GRUFusionPass::AddSliceAndConcatNode(ge::ComputeGraph& graph, ge::NodePtr& fusedNode,
                                            ge::NodePtr& forwardNode, ge::NodePtr& reverseNode,
                                            ge::GeTensorDesc& outputDesc, vector<ge::NodePtr>& newNodes,
                                            int nodeIndex) {
  // forward strided_slice
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  std::string operatorName = fusedDesc->GetName() + "/StridedSliceDForward_" + "H";
  auto sliceOpForward = ge::OperatorFactory::CreateOperator(operatorName.c_str(), "StridedSliceD");
  FUSION_PASS_CHECK(sliceOpForward.IsEmpty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create slice_op forward Op error"),
                    return FAILED);
  auto sliceDescForward = ge::OpDescUtils::GetOpDescFromOperator(sliceOpForward);
  sliceOpForward.BreakConnect();

  ge::GeTensorDesc sliceTensorDesc = fusedDesc->GetOutputDesc(nodeIndex).Clone();
  std::vector<int64_t> sliceDims = sliceTensorDesc.GetShape().GetDims();
  sliceDims[0] = 1;
  ge::GeShape sliceShape(sliceDims);
  sliceTensorDesc.SetShape(sliceShape);
  sliceTensorDesc.SetOriginShape(sliceShape);

  sliceDescForward->UpdateInputDesc("x", outputDesc);
  sliceDescForward->UpdateOutputDesc("y", sliceTensorDesc);
  ge::AttrUtils::SetListInt(sliceDescForward, "begin", {-1, 0, 0});
  ge::AttrUtils::SetListInt(sliceDescForward, "end", {-2, sliceDims[1], sliceDims[2]});
  ge::AttrUtils::SetListInt(sliceDescForward, "strides", {-1, 1, 1});

  ge::NodePtr sliceNodeForward = graph.AddNode(sliceDescForward);
  newNodes.push_back(sliceNodeForward);

  // reverse strided_slice
  operatorName = fusedDesc->GetName() + "/StridedSliceDReverse_" + "H";
  auto sliceOpReverse = ge::OperatorFactory::CreateOperator(operatorName.c_str(), "StridedSliceD");
  FUSION_PASS_CHECK(sliceOpReverse.IsEmpty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create slice_op reverse Op error"),
                    return FAILED);
  auto sliceDescReverse = ge::OpDescUtils::GetOpDescFromOperator(sliceOpReverse);
  sliceOpReverse.BreakConnect();

  sliceDescReverse->UpdateInputDesc("x", outputDesc);
  sliceDescReverse->UpdateOutputDesc("y", sliceTensorDesc);
  ge::AttrUtils::SetListInt(sliceDescReverse, "begin", {-1, 0, 0});
  ge::AttrUtils::SetListInt(sliceDescReverse, "end", {-2, sliceDims[1], sliceDims[2]});
  ge::AttrUtils::SetListInt(sliceDescReverse, "strides", {-1, 1, 1});

  ge::NodePtr sliceNodeReverse = graph.AddNode(sliceDescReverse);
  newNodes.push_back(sliceNodeReverse);

  // connect output hc -> slice
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(forwardNode->GetOutDataAnchor(nodeIndex),
                                                       sliceNodeForward->GetInDataAnchor(0)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add edge to fusion slice_node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(reverseNode->GetOutDataAnchor(nodeIndex),
                                                       sliceNodeReverse->GetInDataAnchor(0)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add edge to fusion slice_node failed."),
                    return FAILED);

  // create output concat node
  std::string concatOpName = fusedDesc->GetName() + "/ConcatD_" + "H";
  auto concatOp = ge::OperatorFactory::CreateOperator(concatOpName.c_str(), "ConcatD");
  FUSION_PASS_CHECK(concatOp.IsEmpty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create concat operator error"),
                    return FAILED);
  auto concat_desc = ge::OpDescUtils::GetOpDescFromOperator(concatOp);
  concatOp.BreakConnect();

  ge::GeTensorDesc originTensorDesc = fusedDesc->GetOutputDesc(nodeIndex);
  concat_desc->AddInputDesc("x_forward", sliceTensorDesc);
  concat_desc->AddInputDesc("x_reverse", sliceTensorDesc);
  concat_desc->UpdateOutputDesc("y", originTensorDesc);

  ge::AttrUtils::SetInt(concat_desc, "concat_dim", 0);
  ge::AttrUtils::SetInt(concat_desc, "N", CONCAT_NUM);

  ge::NodePtr concat_node = graph.AddNode(concat_desc);
  newNodes.push_back(concat_node);

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(sliceNodeForward->GetOutDataAnchor(0),
                                                       concat_node->GetInDataAnchor(0)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add silce forward edge to fusion slice_node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(sliceNodeReverse->GetOutDataAnchor(0),
                                                       concat_node->GetInDataAnchor(1)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add slice reverse edge to fusion slice_node failed."),
                    return FAILED);

  for (InDataAnchorPtr oriTopPeerAnchorPtr : fusedNode->GetOutDataAnchor(nodeIndex)->GetPeerInDataAnchors()) {
    oriTopPeerAnchorPtr->UnlinkAll();
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0), oriTopPeerAnchorPtr),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "add concat node to fusion node output failed."),
                      return FAILED);
  }

  return SUCCESS;
}

Status GRUFusionPass::AddBiasSplitNode(ge::ComputeGraph& graph, const ge::NodePtr& fusedNode,
                                       ge::NodePtr& splitNode) const {
  OpDescPtr splitDesc = std::make_shared<ge::OpDesc>(fusedNode->GetName() + "/DynamicGRUV2_split", "SplitD");
  FUSION_PASS_CHECK(splitDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "splitD is null, SplitD failed."),
                    return PARAM_INVALID);
  AttrUtils::SetInt(splitDesc, "split_dim", 1);
  AttrUtils::SetInt(splitDesc, "num_split", SPLIT_GROUP);

  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode's OpDesc is null."),
                    return PARAM_INVALID);
  ge::GeTensorDesc bias = fusedDesc->GetInputDesc(BIAS_INPUT_INDEX);
  FUSION_PASS_CHECK(splitDesc->AddInputDesc(bias) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add SplitD input"),
                    return FAILED);

  // build split node Output Desc
  GeTensorDesc inputDesc = fusedDesc->GetInputDesc(BIAS_INPUT_INDEX);
  GeShape inputShape = inputDesc.GetShape();
  int newInputChn = inputShape.GetDim(BIAS_CHANNEL_INDEX);
  GeShape splitOutShape = inputShape;
  splitOutShape.SetDim(BIAS_CHANNEL_INDEX, newInputChn / SPLIT_GROUP);
  std::vector<int64_t> splitOutDims = RemoveNumDirectionsDim(splitOutShape.GetDims(), false);
  GeShape splitDShape(splitOutDims);
  GeTensorDesc splitOutDesc = inputDesc;
  splitOutDesc.SetShape(splitDShape);
  splitOutDesc.SetOriginShape(splitDShape);
  for (int i = 0; i < SPLIT_GROUP; i++) {
    FUSION_PASS_CHECK(splitDesc->AddOutputDesc(splitOutDesc) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add bias split output failed."),
                      return FAILED);
  }

  // create SplitD Node
  splitNode = graph.AddNode(splitDesc);
  FUSION_PASS_CHECK(splitNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "SplitD node is null, fusion failed."),
                    return FAILED);
  // connect bias to Split input
  graphStatus status = GraphUtils::AddEdge(fusedNode->GetInDataAnchor(BIAS_INPUT_INDEX)->GetPeerOutAnchor(),
                                           splitNode->GetInDataAnchor(0));
  FUSION_PASS_CHECK(status != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add data to Split edge fail"),
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
  FUSION_PASS_CHECK(ret != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT("GRUFusionPass", "CreateSliceNode AddInputDesc fail"),
                    return FAILED);
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
  FUSION_PASS_CHECK(ret != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT("GRUFusionPass", "CreateSliceNode AddOutputDesc fail"),
                    return FAILED);
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
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Set perm to %s failed.", transposeOpdesc->GetName().c_str()),
                    return FAILED);

  ge::GeTensorDesc inputDesc = weightNode->GetOpDesc()->GetOutputDesc(0).Clone();
  std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
  FUSION_PASS_CHECK(dims.size() != 3,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "weight dim size is not 3."), return FAILED);
  std::vector<int64_t> newDim = {dims[0], dims[2], dims[1]};

  ge::GeTensorDesc outputDesc = gruNode->GetOpDesc()->GetInputDesc(anchorIndex).Clone();
  outputDesc.SetOriginShape(GeShape(newDim));
  outputDesc.SetShape(GeShape(newDim));

  FUSION_PASS_CHECK(transposeOpdesc->AddInputDesc("x", inputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "%s add inputDesc failed.", transposeOpdesc->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(transposeOpdesc->AddOutputDesc("y", outputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "%s add outputDesc failed.", transposeOpdesc->GetName().c_str()),
                    return FAILED);

  ge::NodePtr transposeNode = graph.AddNode(transposeOpdesc);

  ge::OutDataAnchorPtr src = weightNode->GetOutDataAnchor(0);
  ge::InDataAnchorPtr dst = gruNode->GetInDataAnchor(anchorIndex);

  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(src, dst) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "remove %s input edge error", gruNode->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(src, transposeNode->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                            "Add edge between node %s. and node %s failed.",
                            weightNode->GetName().c_str(), transposeNode->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(transposeNode->GetOutDataAnchor(0), dst) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                            "Add edge between node %s. and node %s failed.",
                            transposeNode->GetName().c_str(), gruNode->GetName().c_str()),
                    return FAILED);
  return SUCCESS;
}

std::vector<int64_t> GRUFusionPass::RemoveNumDirectionsDim(const std::vector<int64_t>& dims, bool isReverse) const {
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
  FUSION_PASS_CHECK(n < 2, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dim size less then 2."), return);
  int64_t first = dims[n - 1];
  int64_t second = dims[n - 2];
  dims[n - 1] = (second + DIM_5HD_DIV_FACTOR) / DIM_5HD_UNIT_SIZE;
  dims[n - 2] = (first + DIM_5HD_DIV_FACTOR) / DIM_5HD_UNIT_SIZE;
  dims.push_back(DIM_5HD_UNIT_SIZE);
  dims.push_back(DIM_5HD_UNIT_SIZE);
}

void GRUFusionPass::ProcessZFormat(std::vector<int64_t>& dims) {
  for (auto& elem : dims) {
    elem = (elem + DIM_5HD_DIV_FACTOR) / DIM_5HD_UNIT_SIZE;
  }
  dims.push_back(DIM_5HD_UNIT_SIZE);
  dims.push_back(DIM_5HD_UNIT_SIZE);
}

REGISTER_PASS("CommonGRUFusionPass", BUILT_IN_GRAPH_PASS, GRUFusionPass);
}  // namespace fe