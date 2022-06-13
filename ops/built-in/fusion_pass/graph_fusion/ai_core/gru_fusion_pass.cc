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

static const int FIRST_INPUT = 1;
static const int SECOND_INPUT = 2;
static const int THIRD_INPUT  = 3;

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
  FUSION_PASS_CHECK(SUCCESS != InitParams(graph, mapping, newNodes),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "InitParams fail"), return FAILED);

  std::string direction;
  ge::AttrUtils::GetStr(fusedDesc_, "direction", direction);
  if (direction == "bidirectional") {
    FUSION_PASS_CHECK(SUCCESS != ProcessBidiFusion(),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ProcessBidiFusion fail"), return FAILED);
    return SUCCESS;
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CommonGRU enter UNIDIRECTIONAL process!");
  auto gruOp = ge::OperatorFactory::CreateOperator((fusedDesc_->GetName() + "_splitD_layer").c_str(), "DynamicGRUV2");
  FUSION_PASS_CHECK(gruOp.IsEmpty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "GRUV2 CreateOperator error"), return FAILED);

  // create DynamicGRUV2 OpDesc
  std::shared_ptr<ge::OpDesc> gruOpDesc = ge::OpDescUtils::GetOpDescFromOperator(gruOp);
  gruOp.BreakConnect();
  FUSION_PASS_CHECK(gruOpDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "gruOpDesc is null."), return PARAM_INVALID);

  // process x
  GeTensorDesc xInput = fusedDesc_->GetInputDesc(0);
  std::vector<int64_t> xInputDims = xInput.GetShape().GetDims();

  xInput.SetOriginShape(GeShape(xInputDims));
  (void)ProcessNZFormat(xInputDims);
  xInput.Update(GeShape(xInputDims), ge::FORMAT_FRACTAL_NZ, xInput.GetDataType());
  gruOpDesc->UpdateInputDesc("x", xInput);

  ge::NodePtr transNode;
  FUSION_PASS_CHECK(AddTransposNode(1, 1, transNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add transpos failed"), return FAILED);
  ge::NodePtr secondNode;
  FUSION_PASS_CHECK(AddTransposNode(2, 1, secondNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add transpos failed"), return FAILED);

  // process weight_input
  GeTensorDesc weightInput = fusedDesc_->GetInputDesc(1);
  std::vector<int64_t> weightInputDims = RemoveNumDirectionsDim(weightInput.GetShape().GetDims(), true);

  weightInput.SetOriginShape(GeShape(weightInputDims));
  (void)ProcessZFormat(weightInputDims);
  weightInput.Update(GeShape(weightInputDims), ge::FORMAT_FRACTAL_Z, weightInput.GetDataType());
  gruOpDesc->UpdateInputDesc("weight_input", weightInput);

  // process weight_hidden
  GeTensorDesc weightHidden = fusedDesc_->GetInputDesc(2);
  std::vector<int64_t> weightHiddenDims = RemoveNumDirectionsDim(weightHidden.GetShape().GetDims(), true);

  weightHidden.SetOriginShape(GeShape(weightHiddenDims));
  (void)ProcessZFormat(weightHiddenDims);
  weightHidden.Update(GeShape(weightHiddenDims), ge::FORMAT_FRACTAL_Z, weightHidden.GetDataType());
  gruOpDesc->UpdateInputDesc("weight_hidden", weightHidden);

  if (fusedDesc_->MutableInputDesc("sequence_lens") != nullptr) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "yes hasSeqLength");
    gruOpDesc->UpdateInputDesc("seq_length", *fusedDesc_->MutableInputDesc("sequence_lens"));
  }

  if (fusedDesc_->MutableInputDesc("initial_h") != nullptr) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "yes hasInitH");
    GeTensorDesc initialH = *fusedDesc_->MutableInputDesc("initial_h");
    std::vector<int64_t> initialHDims = RemoveNumDirectionsDim(initialH.GetShape().GetDims(), false);

    initialH.SetOriginShape(GeShape(initialHDims));
    (void)ProcessNZFormat(initialHDims);
    initialH.Update(GeShape(initialHDims), ge::FORMAT_FRACTAL_NZ, initialH.GetDataType());
    gruOpDesc->UpdateInputDesc("init_h", initialH);
  }

  GeTensorDesc y = fusedDesc_->GetOutputDesc(0);
  std::vector<int64_t> yDims = ProcessOutputDim(y.GetShape().GetDims());
  y.SetOriginShape(GeShape(yDims));
  (void)ProcessNZFormat(yDims);
  y.Update(GeShape(yDims), ge::FORMAT_FRACTAL_NZ, y.GetDataType());
  UpdateOutputDesc(gruOpDesc, y);

  // create a splitD Op for bias
  ge::NodePtr splitNode = nullptr;
  if (fusedDesc_->MutableInputDesc("b") != nullptr) {
    // add bias Node
    OP_LOGD(FUSED_OP_TYPE.c_str(), "CommonGRU has bias input.");
    FUSION_PASS_CHECK(AddBiasSplitNode(splitNode) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddBiasSplitNode failed."), return FAILED);
    // splitNode must not be nullptr when AddBiasSplit returns SUCCESS
    ge::OpDescPtr splitDesc = splitNode->GetOpDesc();
    FUSION_PASS_CHECK(splitDesc == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "splitDesc is null."), return FAILED);
    GeTensorDesc splitOutDesc = splitDesc->GetOutputDesc(0);
    gruOpDesc->UpdateInputDesc("bias_input", splitOutDesc);
    gruOpDesc->UpdateInputDesc("bias_hidden", splitOutDesc);
  }

  // create DynamicGRUV2 Node
  ge::NodePtr gruNode = graph_->AddNode(gruOpDesc);
  FUSION_PASS_CHECK(gruNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add gruNode failed."), return FAILED);
  newNodes_->push_back(gruNode);

  // connect all input nodes
  FUSION_PASS_CHECK(SUCCESS != AddInputNodes(splitNode, gruNode),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddInputNodes fail"), return FAILED);
  ge::NodePtr output_node = gruNode;
  int anchor_index = 1;
  if (yDims[0] > 1) {
    ge::NodePtr slice_node = nullptr;
    FUSION_PASS_CHECK(CreateSliceNode(gruNode, slice_node) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Create slice node fail."), return FAILED);
    output_node = slice_node;
    anchor_index = 0;
  }

  auto yOriTopPeerAnchors = fusedNode_->GetOutDataAnchor(0)->GetPeerInDataAnchors();
  auto yhOriTopPeerAnchors = fusedNode_->GetOutDataAnchor(1)->GetPeerInDataAnchors();

  // unlink all input of CommonGRU
  UnlinkAllAnchors();

  for (uint64_t i = 0; i < yOriTopPeerAnchors.size(); ++i) {
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(gruNode->GetOutDataAnchor(0), yOriTopPeerAnchors.at(i)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output_y edge to gruNode failed."), return FAILED);
  }

  for (uint64_t i = 0; i < yhOriTopPeerAnchors.size(); ++i) {
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(output_node->GetOutDataAnchor(anchor_index), yhOriTopPeerAnchors.at(i)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output_h edge to gruNode failed."), return FAILED);
  }
  return SUCCESS;
}

Status GRUFusionPass::ProcessBidiFusion() {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CommonGRU enter bidirectional process!");
  auto dynamicGruV2OpForward = ge::OperatorFactory::CreateOperator(
      (fusedDesc_->GetName() + "/DynamicGRUV2" + "Forward").c_str(), "DynamicGRUV2");
  FUSION_PASS_CHECK(dynamicGruV2OpForward.IsEmpty(),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CommonGRU create Forward operator error"), return FAILED);
  auto dynamicGruV2DescForward = ge::OpDescUtils::GetOpDescFromOperator(dynamicGruV2OpForward);
  dynamicGruV2OpForward.BreakConnect();

  auto dynamicGruV2OpReverse = ge::OperatorFactory::CreateOperator(
      (fusedDesc_->GetName() + "/DynamicGRUV2" + "Reverse").c_str(), "DynamicGRUV2");
  FUSION_PASS_CHECK(dynamicGruV2OpReverse.IsEmpty(),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CommonGRU create Reverse operator error"), return FAILED);
  auto dynamicGruV2DescReverse = ge::OpDescUtils::GetOpDescFromOperator(dynamicGruV2OpReverse);
  dynamicGruV2OpReverse.BreakConnect();

  // process x
  GeTensorDesc xDesc = fusedDesc_->GetInputDesc(X_INDEX).Clone();
  std::vector<int64_t> xInputDims = xDesc.GetShape().GetDims();
  // w shape [num_directions, 3*hidden_size, input_size]
  int64_t inputSize = fusedDesc_->GetInputDesc(W_INDEX).GetShape().GetDim(INPUT_SIZE_INDEX);
  xInputDims[INPUT_SIZE_INDEX] = inputSize;
  GeShape xInputShape(xInputDims);
  xDesc.SetOriginShape(xInputShape);
  xDesc.SetShape(xInputShape);
  dynamicGruV2DescForward->UpdateInputDesc("x", xDesc);
  dynamicGruV2DescReverse->UpdateInputDesc("x", xDesc);

  // process weight_input
  ge::NodePtr weightInputSplitNode = nullptr;
  FUSION_PASS_CHECK(SUCCESS != AddBidiWeightSplitNode(W_INDEX, weightInputSplitNode),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CommonGRU add weight_input node failed."), return FAILED);
  ge::OpDescPtr weightInputSplitDesc = weightInputSplitNode->GetOpDesc();
  FUSION_PASS_CHECK(nullptr == weightInputSplitDesc,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "weightInputSplitDesc is null."), return FAILED);
  dynamicGruV2DescForward->UpdateInputDesc("weight_input", weightInputSplitDesc->GetOutputDesc(SPLIT_FORWARD_INDEX));
  dynamicGruV2DescReverse->UpdateInputDesc("weight_input", weightInputSplitDesc->GetOutputDesc(SPLIT_FORWARD_INDEX));

  // process weight_hidden
  ge::NodePtr weightHiddenSplitNode = nullptr;
  FUSION_PASS_CHECK(SUCCESS != AddBidiWeightSplitNode(R_INDEX, weightHiddenSplitNode),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CommonGRU add weight_hidden failed."), return FAILED);
  ge::OpDescPtr weightHiddenSplitDesc = weightHiddenSplitNode->GetOpDesc();
  FUSION_PASS_CHECK(nullptr == weightHiddenSplitDesc,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "weightHiddenSplitDesc is null."), return FAILED);
  dynamicGruV2DescForward->UpdateInputDesc("weight_hidden", weightHiddenSplitDesc->GetOutputDesc(SPLIT_FORWARD_INDEX));
  dynamicGruV2DescReverse->UpdateInputDesc("weight_hidden", weightHiddenSplitDesc->GetOutputDesc(SPLIT_FORWARD_INDEX));

  // process bias
  bool hasBias = fusedDesc_->MutableInputDesc("b") != nullptr;
  ge::NodePtr inputBiasSplitNode = nullptr;
  ge::NodePtr hiddenBiasSplitNode = nullptr;
  if (hasBias) {
    // add bias split Node
    FUSION_PASS_CHECK(SUCCESS != AddBidiBiasSplitNode(B_INDEX, inputBiasSplitNode, hiddenBiasSplitNode),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CommonGRU add bias split node failed."), return FAILED);

    // splitNode must not be nullptr when AddBiasSplit returns SUCCESS
    ge::OpDescPtr inputBiasSplitDesc = inputBiasSplitNode->GetOpDesc();
    ge::OpDescPtr hiddenBiasSplitDesc = hiddenBiasSplitNode->GetOpDesc();
    FUSION_PASS_CHECK(nullptr == inputBiasSplitDesc,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "splitNode's OpDesc is null."), return PARAM_INVALID);
    FUSION_PASS_CHECK(nullptr == hiddenBiasSplitDesc,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "splitNode's OpDesc is null."), return PARAM_INVALID);

    dynamicGruV2DescForward->UpdateInputDesc("bias_input", inputBiasSplitDesc->GetOutputDesc(SPLIT_FORWARD_INDEX));
    dynamicGruV2DescReverse->UpdateInputDesc("bias_input", inputBiasSplitDesc->GetOutputDesc(SPLIT_FORWARD_INDEX));

    dynamicGruV2DescForward->UpdateInputDesc("bias_hidden", hiddenBiasSplitDesc->GetOutputDesc(SPLIT_FORWARD_INDEX));
    dynamicGruV2DescReverse->UpdateInputDesc("bias_hidden", hiddenBiasSplitDesc->GetOutputDesc(SPLIT_FORWARD_INDEX));
  }

  // process seq_length
  bool hasSeqLength = fusedDesc_->MutableInputDesc("sequence_lens") != nullptr;
  if (hasSeqLength) {
    dynamicGruV2DescForward->UpdateInputDesc("seq_length", *fusedDesc_->MutableInputDesc("sequence_lens"));
    dynamicGruV2DescReverse->UpdateInputDesc("seq_length", *fusedDesc_->MutableInputDesc("sequence_lens"));
  }

  // process init_h
  bool hasInitH = fusedDesc_->MutableInputDesc("initial_h") != nullptr;
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CommonGRU hasBias,hasSeqLength,hasInitH is %d,%d,%d",
      hasBias, hasSeqLength, hasInitH);
  ge::NodePtr initHSplitNode = nullptr;
  if (hasInitH) {
    FUSION_PASS_CHECK(SUCCESS != AddBidiInitHSplitNode(INITIAL_H_INDEX, initHSplitNode),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddBidiInitHSplitNode failed."), return FAILED);
    ge::OpDescPtr initHSplitDesc = initHSplitNode->GetOpDesc();
    FUSION_PASS_CHECK(nullptr == initHSplitDesc,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "initHSplitDesc is null."), return PARAM_INVALID);
    dynamicGruV2DescForward->UpdateInputDesc("init_h", initHSplitDesc->GetOutputDesc(SPLIT_FORWARD_INDEX));
    dynamicGruV2DescReverse->UpdateInputDesc("init_h", initHSplitDesc->GetOutputDesc(SPLIT_FORWARD_INDEX));
  }

  // process direction attr
  ge::AttrUtils::SetStr(dynamicGruV2DescForward, "direction", GRU_DEFAULT_DIRECTION);
  ge::AttrUtils::SetStr(dynamicGruV2DescReverse, "direction", GRU_BIDI_DIRECTION);

  // process output
  GeTensorDesc outputYDesc = fusedDesc_->GetOutputDesc(Y_OUTPUT_INDEX);
  GeShape yOriginShape(ProcessOutputDim(outputYDesc.GetShape().GetDims()));
  outputYDesc.SetOriginShape(yOriginShape);
  outputYDesc.SetShape(yOriginShape);

  UpdateOutputDesc(dynamicGruV2DescForward, outputYDesc);
  UpdateOutputDesc(dynamicGruV2DescReverse, outputYDesc);

  // create DynamicGRUV2 forward node
  ge::NodePtr dynamicGruV2ForwardNode = graph_->AddNode(dynamicGruV2DescForward);
  FUSION_PASS_CHECK(nullptr == dynamicGruV2ForwardNode,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "DynamicGRUV2 forward node is null."), return FAILED);
  newNodes_->push_back(dynamicGruV2ForwardNode);

  // create DynamicGRUV2 reverse node
  ge::NodePtr dynamicGRUV2ReverseNode = graph_->AddNode(dynamicGruV2DescReverse);
  FUSION_PASS_CHECK(nullptr == dynamicGRUV2ReverseNode,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "DynamicGRUV2 reverse node is null."), return FAILED);
  newNodes_->push_back(dynamicGRUV2ReverseNode);

  // add all input nodes
  FUSION_PASS_CHECK(
      SUCCESS != AddBidiInputNodes({dynamicGruV2ForwardNode, dynamicGRUV2ReverseNode, weightInputSplitNode,
      weightHiddenSplitNode, inputBiasSplitNode, hiddenBiasSplitNode, initHSplitNode}),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "connect input nodes error"), return FAILED);

  // add concat
  FUSION_PASS_CHECK(SUCCESS != AddExpandDimsAndConcatNode(dynamicGruV2ForwardNode, dynamicGRUV2ReverseNode),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create ExpandDimsY Op operator error"), return FAILED);

  // add slice
  FUSION_PASS_CHECK(SUCCESS != AddSliceAndConcatNode(dynamicGruV2ForwardNode, dynamicGRUV2ReverseNode, H_OUTPUT_INDEX),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create slice H Op operator error"), return FAILED);

  UnlinkAllAnchors();
  return SUCCESS;
}

Status GRUFusionPass::CheckParams() {
  GeTensorDesc wDesc = fusedDesc_->GetInputDesc(W_INDEX).Clone();
  std::vector<int64_t> wInputDims = wDesc.GetShape().GetDims();
  std::string direction;

  FUSION_PASS_CHECK(wInputDims.size() != W_INPUT_SIZE,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "weight dim size is not 3."),
                    return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::GetStr(fusedDesc_, "direction", direction),
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

Status GRUFusionPass::AddSplitVNode(SplitInfo splitInfo, ge::NodePtr& splitNode, ge::NodePtr peerOutNode) {
  // get splitDesc
  std::shared_ptr<ge::OpDesc> splitDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (splitDesc = std::make_shared<ge::OpDesc>(splitInfo.nodeName + "/DynamicGRUV2_split_v", "SplitV")),
      return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(splitDesc, "num_split", SPLIT_GROUP),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Set num_split to %s failed.", splitDesc->GetName().c_str()),
                    return FAILED);

  ge::GeTensorDesc splitDimDesc;
  ge::OpDescPtr splitDimOutDesc;
  std::vector<int64_t> splitDimIn = {};
  FUSION_PASS_CHECK(SUCCESS != SetSplitVNodeInfo(splitDimDesc, splitDimOutDesc, splitDimIn, splitInfo.splitDimAxis),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "splitDim set info failed."),
                    return FAILED);

  ge::GeTensorDesc sizeSplitDesc;
  ge::OpDescPtr sizeSplitOutDesc;
  std::vector<int64_t> sizeSplitIn = {SPLIT_GROUP};
  FUSION_PASS_CHECK(SUCCESS != SetSplitVNodeInfo(sizeSplitDesc, sizeSplitOutDesc, sizeSplitIn, splitInfo.sizeSplitAxis),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "sizeSplit set info failed."),
                    return FAILED);

  // split_v
  FUSION_PASS_CHECK(SUCCESS != splitDesc->AddInputDesc("x", splitInfo.inputDesc),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "SplitV add x input failed."),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != splitDesc->AddInputDesc("size_splits", sizeSplitDesc),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "SplitV add size_splits input failed."),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != splitDesc->AddInputDesc("split_dim", splitDimDesc),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "SplitV add split_dim input failed."),
                    return FAILED);
  for (int i = 0; i < SPLIT_GROUP; i++) {
    FUSION_PASS_CHECK(SUCCESS != splitDesc->AddOutputDesc(splitInfo.outputDesc),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "SplitV add SplitD output failed."),
                      return FAILED);
  }

  // create node
  splitNode = graph_->AddNode(splitDesc);
  ge::NodePtr sizeSplitNode = graph_->AddNode(sizeSplitOutDesc);
  ge::NodePtr splitDimNode = graph_->AddNode(splitDimOutDesc);
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
  newNodes_->push_back(splitNode);
  newNodes_->push_back(sizeSplitNode);
  newNodes_->push_back(splitDimNode);

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(peerOutNode->GetOutDataAnchor(splitInfo.splitIndex),
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

Status GRUFusionPass::AddBidiWeightSplitNode(int weightIndex, ge::NodePtr& splitNode) {
  ge::NodePtr weightNode = fusedNode_->GetInDataAnchor(weightIndex)->GetPeerOutAnchor()->GetOwnerNode();
  ge::NodePtr transposeNode;
  FUSION_PASS_CHECK(SUCCESS != AddTransposNode(weightIndex, 2, transposeNode),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddBidiWeightSplitNode transNode failed."),
                    return FAILED);

  ge::GeTensorDesc midDesc = weightNode->GetOpDesc()->GetOutputDesc(0).Clone();
  std::vector<int64_t> dims = midDesc.GetShape().GetDims();
  std::vector<int64_t> newDim = {dims[0], dims[2], dims[1]};
  midDesc.SetOriginShape(GeShape(newDim));
  midDesc.SetShape(GeShape(newDim));

  ge::GeTensorDesc outputDesc = fusedNode_->GetOpDesc()->GetInputDesc(weightIndex).Clone();
  std::vector<int64_t> outDim = {dims[2], dims[1]};
  outputDesc.SetOriginShape(GeShape(outDim));
  outputDesc.SetShape(GeShape(outDim));

  // add split_v node
  std::vector<int32_t> splitDimAxis = {0};
  std::vector<int32_t> sizeSplitAxis = {1, 1};
  FUSION_PASS_CHECK(SUCCESS != AddSplitVNode({weightNode->GetName(), midDesc, outputDesc, splitDimAxis,
                                              sizeSplitAxis, 0}, splitNode, transposeNode),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "w add split node failed."),
                    return FAILED);
  return SUCCESS;
}

Status GRUFusionPass::AddBidiInitHSplitNode(int initHIndex, ge::NodePtr& splitNode) {
  ge::NodePtr initHNode = fusedNode_->GetInDataAnchor(initHIndex)->GetPeerOutAnchor()->GetOwnerNode();
  string nodeType = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(initHNode);

  ge::GeTensorDesc inputDesc = initHNode->GetOpDesc()->GetOutputDesc(SINGLE_OUTPUT_INDEX).Clone();
  std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
  FUSION_PASS_CHECK(dims.size() != 3,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "init_h dim size is not 3."),
                    return FAILED);

  ge::GeTensorDesc outputDesc = fusedNode_->GetOpDesc()->GetInputDesc(initHIndex).Clone();
  std::vector<int64_t> outDim = {dims[1], dims[2]};
  outputDesc.SetOriginShape(GeShape(outDim));
  outputDesc.SetShape(GeShape(outDim));

  if (nodeType == "Const" || nodeType == "Constant") {
    // add split_v node
    std::vector<int32_t> splitDimAxis = {0};
    std::vector<int32_t> sizeSplitAxis = {1, 1};
    FUSION_PASS_CHECK(SUCCESS != AddSplitVNode({initHNode->GetName(), inputDesc, outputDesc,
                                               splitDimAxis, sizeSplitAxis, SINGLE_OUTPUT_INDEX}, splitNode, initHNode),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "w add split node failed."),
                      return FAILED);
    if (initHNode->GetInControlAnchor() != nullptr) {
      initHNode->GetInControlAnchor()->UnlinkAll();
    }
  } else {
    // get splitDesc
    std::shared_ptr<ge::OpDesc> splitDesc = nullptr;
    FUSION_PASS_MAKE_SHARED((splitDesc = std::make_shared<ge::OpDesc>(initHNode->GetName() + "/DynamicGRUV2_h_split",
                                                                      "Split")),
                            return FAILED);
    std::vector<int32_t> splitDims = {INIT_H_SPLIT_INDEX};
    vector<ge::OpDescPtr> splitVector = {splitDesc};
    ge::NodePtr splitDimNode = AttrToConstNode(splitVector, splitDims, "split_dim");
    FUSION_PASS_CHECK(nullptr == splitDimNode,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add split_dim node failed."), return FAILED);
    FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(splitDesc, "num_split", SPLIT_GROUP),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "Set num_split to %s failed.", splitDesc->GetName().c_str()),
                      return FAILED);

    FUSION_PASS_CHECK(splitDesc->AddInputDesc("x", inputDesc) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add init_h SplitD input failed."),
                      return FAILED);
    for (int i = 0; i < SPLIT_GROUP; i++) {
      FUSION_PASS_CHECK(splitDesc->AddOutputDesc(outputDesc) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add init_h SplitD output failed."),
                        return FAILED);
    }

    // create node
    splitNode = graph_->AddNode(splitDesc);
    FUSION_PASS_CHECK(nullptr == splitNode,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add SplitD node is null, fusion failed."),
                      return FAILED);
    newNodes_->push_back(splitNode);
    newNodes_->push_back(splitDimNode);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(splitDimNode->GetOutDataAnchor(SINGLE_OUTPUT_INDEX),
                                              splitNode->GetInDataAnchor(0)) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "Add edge between node %s. and node %s failed.",
                                                     initHNode->GetName().c_str(), splitNode->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(initHNode->GetOutDataAnchor(SINGLE_OUTPUT_INDEX),
                                              splitNode->GetInDataAnchor(FIRST_INPUT)) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "Add edge between node %s. and node %s failed.",
                                                     initHNode->GetName().c_str(), splitNode->GetName().c_str()),
                      return FAILED);
  }

  // remove and add edge
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(initHNode->GetOutDataAnchor(SINGLE_OUTPUT_INDEX),
                                               fusedNode_->GetInDataAnchor(initHIndex)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "remove %s input edge error", fusedNode_->GetName().c_str()),
                    return FAILED);
  return SUCCESS;
}

Status GRUFusionPass::AddBidiBiasSplitNode(int biasIndex, ge::NodePtr& inputSplitNode, ge::NodePtr& hiddenSplitNode) {
  ge::NodePtr biasNode = fusedNode_->GetInDataAnchor(biasIndex)->GetPeerOutAnchor()->GetOwnerNode();

  ge::GeTensorDesc inputDesc = biasNode->GetOpDesc()->GetOutputDesc(0).Clone();
  std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
  FUSION_PASS_CHECK(dims.size() != 2,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "bias dim size is not 2."), return FAILED);

  ge::GeTensorDesc midDesc = biasNode->GetOpDesc()->GetOutputDesc(0).Clone();
  std::vector<int64_t> midDim = {dims[0], dims[1] / SPLIT_GROUP};
  midDesc.SetOriginShape(GeShape(midDim));
  midDesc.SetShape(GeShape(midDim));

  ge::GeTensorDesc outputDesc = fusedNode_->GetOpDesc()->GetInputDesc(biasIndex).Clone();
  std::vector<int64_t> outDim = {dims[1] / SPLIT_GROUP};
  outputDesc.SetOriginShape(GeShape(outDim));
  outputDesc.SetShape(GeShape(outDim));

  // add split_v node
  ge::NodePtr biasSplitNode;
  std::vector<int32_t> biasSplitDimAxis = {1};
  std::vector<int32_t> biasSizeSplitAxis = {static_cast<int32_t>(dims[1]) / SPLIT_GROUP,
                                            static_cast<int32_t>(dims[1]) / SPLIT_GROUP};
  FUSION_PASS_CHECK(SUCCESS != AddSplitVNode({biasNode->GetName() + "_b", inputDesc, midDesc,
                                             biasSplitDimAxis, biasSizeSplitAxis, 0}, biasSplitNode, biasNode),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "b add split node failed."),
                    return FAILED);

  std::vector<int32_t> numSplitDimAxis = {0};
  std::vector<int32_t> numSizeSplitAxis = {1, 1};
  FUSION_PASS_CHECK(SUCCESS != AddSplitVNode({biasNode->GetName() + "_i", midDesc, outputDesc,
                                             numSplitDimAxis, numSizeSplitAxis, 0}, inputSplitNode, biasSplitNode),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "b_i add split node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != AddSplitVNode({biasNode->GetName() + "_h", midDesc, outputDesc,
                                             numSplitDimAxis, numSizeSplitAxis, 1}, hiddenSplitNode, biasSplitNode),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "b_h add split node failed."),
                    return FAILED);

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(biasNode->GetOutDataAnchor(SINGLE_OUTPUT_INDEX),
                                                          fusedNode_->GetInDataAnchor(biasIndex)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "remove %s input edge error", fusedNode_->GetName().c_str()),
                    return FAILED);
  return SUCCESS;
}

Status GRUFusionPass::AddExpandDimsAndConcatNode(ge::NodePtr forwardNode, ge::NodePtr reverseNode) {
  ge::OpDescPtr fusedDesc_ = fusedNode_->GetOpDesc();
  std::string forwardExdName = fusedDesc_->GetName() + "/ExpandDims" + "_Forward";
  std::string reverseExdName = fusedDesc_->GetName() + "/ExpandDims" + "_Reverse";
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

  ge::GeTensorDesc exdTensorDesc = fusedDesc_->GetOutputDesc(0).Clone();
  std::vector<int64_t> dims = exdTensorDesc.GetShape().GetDims();
  dims[1] = 1;
  ge::GeShape outputShape(dims);
  exdTensorDesc.SetShape(outputShape);
  exdTensorDesc.SetOriginShape(outputShape);

  ge::GeTensorDesc outputDesc = forwardNode->GetOpDesc()->GetOutputDesc(0).Clone();
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
  ge::NodePtr forwardExdNode = graph_->AddNode(forwardExdDesc);
  ge::NodePtr reverseExdNode = graph_->AddNode(reverseExdDesc);
  ge::NodePtr axisNode = graph_->AddNode(axisDesc);

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

  newNodes_->push_back(forwardExdNode);
  newNodes_->push_back(reverseExdNode);
  newNodes_->push_back(axisNode);

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
  vector<ge::NodePtr> fusedNodes = {forwardExdNode, reverseExdNode};
  vector<int64_t> nodeDims = {0, 1};
  FUSION_PASS_CHECK(SUCCESS != AddConcatNode(fusedNodes, fusedDesc_->GetName() + "/Concat_" + "Y", nodeDims),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add concat node to fusion node failed."),
                    return FAILED);
  return SUCCESS;
}

Status GRUFusionPass::AddSliceAndConcatNode(ge::NodePtr forwardNode, ge::NodePtr reverseNode, int nodeIndex) {
  // forward strided_slice
  auto sliceOpForward = ge::OperatorFactory::CreateOperator(
      (fusedDesc_->GetName() + "/StridedSliceDForward_" + "H").c_str(), "StridedSlice");
  FUSION_PASS_CHECK(sliceOpForward.IsEmpty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create slice_op forward Op error"),
                    return FAILED);
  auto sliceDescForward = ge::OpDescUtils::GetOpDescFromOperator(sliceOpForward);
  sliceOpForward.BreakConnect();

  ge::GeTensorDesc sliceTensorDesc = fusedDesc_->GetOutputDesc(nodeIndex).Clone();
  std::vector<int64_t> sliceDims = sliceTensorDesc.GetShape().GetDims();
  sliceDims[0] = 1;
  ge::GeShape sliceShape(sliceDims);
  sliceTensorDesc.SetShape(sliceShape);
  sliceTensorDesc.SetOriginShape(sliceShape);

  // reverse strided_slice
  auto sliceOpReverse = ge::OperatorFactory::CreateOperator(
      (fusedDesc_->GetName() + "/StridedSliceDReverse_" + "H").c_str(), "StridedSlice");
  FUSION_PASS_CHECK(sliceOpReverse.IsEmpty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create slice_op reverse Op error"),
                    return FAILED);
  auto sliceDescReverse = ge::OpDescUtils::GetOpDescFromOperator(sliceOpReverse);
  sliceOpReverse.BreakConnect();

  ge::GeTensorDesc outputDesc = forwardNode->GetOpDesc()->GetOutputDesc(0).Clone();
  sliceDescForward->UpdateInputDesc("x", outputDesc);
  sliceDescReverse->UpdateInputDesc("x", outputDesc);

  vector<ge::OpDescPtr> strideSliceVector = {sliceDescForward, sliceDescReverse};
  std::vector<int64_t> beginDims = {-1, 0, 0};
  ge::NodePtr beginNode = AttrToConstNode(strideSliceVector, beginDims, "begin");
  std::vector<int64_t> endDims = {-2, sliceDims[1], sliceDims[2]};
  ge::NodePtr endNode = AttrToConstNode(strideSliceVector, endDims, "end");
  std::vector<int64_t> stridesDims = {-1, 1, 1};
  ge::NodePtr stridesNode = AttrToConstNode(strideSliceVector, stridesDims, "strides");
  FUSION_PASS_CHECK(nullptr == beginNode,
                VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add begin node failed."), return FAILED);
  FUSION_PASS_CHECK(nullptr == endNode,
                VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add end node failed."), return FAILED);
  FUSION_PASS_CHECK(nullptr == stridesNode,
                VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add strides node failed."), return FAILED);

  sliceDescForward->UpdateOutputDesc("y", sliceTensorDesc);
  sliceDescReverse->UpdateOutputDesc("y", sliceTensorDesc);

  ge::NodePtr sliceNodeForward = graph_->AddNode(sliceDescForward);
  ge::NodePtr sliceNodeReverse = graph_->AddNode(sliceDescReverse);
  newNodes_->push_back(sliceNodeForward);
  newNodes_->push_back(sliceNodeReverse);
  newNodes_->push_back(beginNode);
  newNodes_->push_back(endNode);
  newNodes_->push_back(stridesNode);

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

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(beginNode->GetOutDataAnchor(0),
                                                       sliceNodeForward->GetInDataAnchor(FIRST_INPUT)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add begin edge to fusion slice_node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(beginNode->GetOutDataAnchor(0),
                                                       sliceNodeReverse->GetInDataAnchor(FIRST_INPUT)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add begin edge to fusion slice_node failed."),
                    return FAILED);

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(endNode->GetOutDataAnchor(0),
                                                       sliceNodeForward->GetInDataAnchor(SECOND_INPUT)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add end edge to fusion slice_node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(endNode->GetOutDataAnchor(0),
                                                       sliceNodeReverse->GetInDataAnchor(SECOND_INPUT)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add end edge to fusion slice_node failed."),
                    return FAILED);

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(stridesNode->GetOutDataAnchor(0),
                                                       sliceNodeForward->GetInDataAnchor(THIRD_INPUT)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add strides edge to fusion slice_node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(stridesNode->GetOutDataAnchor(0),
                                                       sliceNodeReverse->GetInDataAnchor(THIRD_INPUT)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add strides edge to fusion slice_node failed."),
                    return FAILED);

  // create output concat node
  vector<ge::NodePtr> fusedNodes = {sliceNodeForward, sliceNodeReverse};
  vector<int64_t> nodeDims = {nodeIndex, 0};
  FUSION_PASS_CHECK(SUCCESS != AddConcatNode(fusedNodes, fusedDesc_->GetName() + "/Concat_" + "H", nodeDims),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add concat node to fusion node failed."),
                    return FAILED);
  return SUCCESS;
}

Status GRUFusionPass::AddBiasSplitNode(ge::NodePtr& splitNode) {
  OpDescPtr splitDesc = std::make_shared<ge::OpDesc>(fusedNode_->GetName() + "/DynamicGRUV2_split", "Split");
  FUSION_PASS_CHECK(splitDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "splitD is null, SplitD failed."),
                    return PARAM_INVALID);
  std::vector<int64_t> splitDims = {1};
  vector<ge::OpDescPtr> splitVector = {splitDesc};
  ge::NodePtr splitDimNode = AttrToConstNode(splitVector, splitDims, "split_dim");
  FUSION_PASS_CHECK(nullptr == splitDimNode,
                VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add split_dim node failed."), return FAILED);
  AttrUtils::SetInt(splitDesc, "num_split", SPLIT_GROUP);

  ge::GeTensorDesc bias = fusedDesc_->GetInputDesc(BIAS_INPUT_INDEX);
  FUSION_PASS_CHECK(splitDesc->AddInputDesc("x", bias) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add SplitD input"),
                    return FAILED);

  // build split node Output Desc
  GeTensorDesc inputDesc = fusedDesc_->GetInputDesc(BIAS_INPUT_INDEX);
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
  splitNode = graph_->AddNode(splitDesc);
  FUSION_PASS_CHECK(splitNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "SplitD node is null, fusion failed."),
                    return FAILED);
  newNodes_->push_back(splitNode);
  newNodes_->push_back(splitDimNode);
  // connect bias to Split input
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(splitDimNode->GetOutDataAnchor(0),
                                                       splitNode->GetInDataAnchor(0)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add split_dim edge to splitnode failed."),
                    return FAILED);
  graphStatus status = GraphUtils::AddEdge(fusedNode_->GetInDataAnchor(BIAS_INPUT_INDEX)->GetPeerOutAnchor(),
                                           splitNode->GetInDataAnchor(FIRST_INPUT));
  FUSION_PASS_CHECK(status != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add data to Split edge fail"),
                    return FAILED);
  return SUCCESS;
}

Status GRUFusionPass::CreateSliceNode(ge::NodePtr gru_node, ge::NodePtr& new_node) {
  ge::OpDescPtr new_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((new_desc = std::make_shared<ge::OpDesc>(gru_node->GetName() + "_SliceD", "Slice")),
                          return INTERNAL_ERROR);
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(gru_node);
  auto output_desc1 = op.GetOutputDesc(1);
  std::vector<int64_t> dims = output_desc1.GetShape().GetDims();
  ge::GeShape input_shape(dims);
  std::vector<int64_t> origin_dims = output_desc1.GetOriginShape().GetDims();
  ge::GeShape origin_shape(origin_dims);
  ge::Format data_format = output_desc1.GetFormat();
  ge::DataType data_type = output_desc1.GetDataType();
  auto ret = new_desc->AddInputDesc("x", GeTensorDesc(input_shape, data_format, data_type));
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

  vector<ge::OpDescPtr> sliceVector = {new_desc};
  ge::NodePtr offsetsNode = AttrToConstNode(sliceVector, offsets, "offsets");
  FUSION_PASS_CHECK(nullptr == offsetsNode,
                VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add offsets node failed."), return FAILED);
  ge::NodePtr sizeNode = AttrToConstNode(sliceVector, origin_output_dims, "size");
  FUSION_PASS_CHECK(nullptr == sizeNode,
                VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add size node failed."), return FAILED);

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

  new_node = graph_->AddNode(new_desc);
  newNodes_->push_back(new_node);
  newNodes_->push_back(offsetsNode);
  newNodes_->push_back(sizeNode);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(gru_node->GetOutDataAnchor(1),
                                                       new_node->GetInDataAnchor(0)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddEdge for slice node fail"),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(offsetsNode->GetOutDataAnchor(0),
                                                       new_node->GetInDataAnchor(FIRST_INPUT)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddEdge for slice node fail"),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(sizeNode->GetOutDataAnchor(0),
                                                       new_node->GetInDataAnchor(SECOND_INPUT)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddEdge for slice node fail"),
                    return FAILED);
  return SUCCESS;
}

Status GRUFusionPass::AddTransposNode(int anchorIndex, int nodeNum, ge::NodePtr& transposeNode) {
  ge::NodePtr weightNode = fusedNode_->GetInDataAnchor(anchorIndex)->GetPeerOutAnchor()->GetOwnerNode();
  std::shared_ptr<ge::OpDesc> transposeOpdesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (transposeOpdesc = std::make_shared<ge::OpDesc>(weightNode->GetName() + "_transpose_b", "Transpose")),
      return FAILED);

  ge::GeTensorDesc inputDesc = weightNode->GetOpDesc()->GetOutputDesc(0).Clone();
  std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
  FUSION_PASS_CHECK(dims.size() != 3,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "weight dim size is not 3."), return FAILED);
  std::vector<int64_t> newDim = {dims[0], dims[2], dims[1]};

  ge::GeTensorDesc outputDesc;
  if (nodeNum == 1) {
    outputDesc = fusedNode_->GetOpDesc()->GetInputDesc(anchorIndex).Clone();
  } else {
    outputDesc = weightNode->GetOpDesc()->GetOutputDesc(0).Clone();
  }
  outputDesc.SetOriginShape(GeShape(newDim));
  outputDesc.SetShape(GeShape(newDim));

  FUSION_PASS_CHECK(transposeOpdesc->AddInputDesc("x", inputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "%s add inputDesc failed.", transposeOpdesc->GetName().c_str()),
                    return FAILED);
  vector<int64_t> perm = {0, 2, 1};
  vector<ge::OpDescPtr> transVector = {transposeOpdesc};
  ge::NodePtr permNode = AttrToConstNode(transVector, perm, "perm");
  FUSION_PASS_CHECK(nullptr == permNode,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add perm node failed."), return FAILED);
  FUSION_PASS_CHECK(transposeOpdesc->AddOutputDesc("y", outputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "%s add outputDesc failed.", transposeOpdesc->GetName().c_str()),
                    return FAILED);

  transposeNode = graph_->AddNode(transposeOpdesc);

  ge::OutDataAnchorPtr src = weightNode->GetOutDataAnchor(0);
  ge::InDataAnchorPtr dst = fusedNode_->GetInDataAnchor(anchorIndex);
  newNodes_->push_back(transposeNode);
  newNodes_->push_back(permNode);
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(src, dst) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "remove %s input edge error", fusedNode_->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(src, transposeNode->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                            "Add edge between node %s. and node %s failed.",
                            weightNode->GetName().c_str(), transposeNode->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(permNode->GetOutDataAnchor(0),
                                                       transposeNode->GetInDataAnchor(FIRST_INPUT)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add perm edge to transpose failed."),
                    return FAILED);
  if (nodeNum == 1) {
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(transposeNode->GetOutDataAnchor(0), dst) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                              "Add edge between node %s. and node %s failed.",
                              transposeNode->GetName().c_str(), fusedNode_->GetName().c_str()),
                      return FAILED);
  }
  return SUCCESS;
}

template<typename T>
ge::NodePtr GRUFusionPass::AttrToConstNode(vector<ge::OpDescPtr>& tensorDesc, T& inputDims,
                                           const std::string& nodeName) {
  ge::GeTensorDesc constDesc;
  std::vector<int64_t> dimIn = {};
  ge::GeTensorPtr constPtr;
  if (nodeName == "split_dim") {
    SetTensorDescription(constDesc, dimIn, ge::FORMAT_ND, ge::DT_INT32);
    FUSION_PASS_MAKE_SHARED(
      (constPtr = std::make_shared<ge::GeTensor>(constDesc, reinterpret_cast<uint8_t *>(inputDims.data()),
                                                 inputDims.size() * sizeof(int32_t))),
      return nullptr);
  } else {
    SetTensorDescription(constDesc, dimIn, ge::FORMAT_ND, ge::DT_INT64);
    FUSION_PASS_MAKE_SHARED(
      (constPtr = std::make_shared<ge::GeTensor>(constDesc, reinterpret_cast<uint8_t *>(inputDims.data()),
                                                 inputDims.size() * sizeof(int64_t))),
      return nullptr);
  }

  for (auto desc : tensorDesc) {
    FUSION_PASS_CHECK(SUCCESS != desc->AddInputDesc(nodeName, constDesc),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AttrToConstNode AddInputDesc failed."),
                      return nullptr);
  }

  ge::OpDescPtr constOpPtr = ge::OpDescUtils::CreateConstOp(constPtr);
  FUSION_PASS_CHECK(nullptr == constOpPtr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AttrToConstNode CreateConstOp failed."),
                    return nullptr);
  ge::NodePtr constNode = graph_->AddNode(constOpPtr);
  FUSION_PASS_CHECK(nullptr == constNode,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AttrToConstNode AddNode failed."),
                    return nullptr);
  return  constNode;
}

Status GRUFusionPass::AddConcatNode(vector<ge::NodePtr>& fusedNodes, const std::string& nodeName,
                                    vector<int64_t>& nodeDims) {
  // create output concat node
  auto concatOp = ge::OperatorFactory::CreateOperator(nodeName.c_str(), "Concat");
  FUSION_PASS_CHECK(concatOp.IsEmpty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create concat operator error"),
                    return FAILED);
  auto concat_desc = ge::OpDescUtils::GetOpDescFromOperator(concatOp);
  concatOp.BreakConnect();

  ge::GeTensorDesc originTensorDesc = fusedDesc_->GetOutputDesc(nodeDims[0]);
  std::vector<int64_t> concatDims = {nodeDims[1]};
  vector<ge::OpDescPtr> concatVector = {concat_desc};
  ge::NodePtr concatDimNode = AttrToConstNode(concatVector, concatDims, "concat_dim");
  FUSION_PASS_CHECK(nullptr == concatDimNode,
                VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add concat_dim node failed."), return FAILED);

  ge::GeTensorDesc inputTensorDesc = fusedNodes[0]->GetOpDesc()->GetOutputDesc(0).Clone();
  concat_desc->AddInputDesc("x0", inputTensorDesc);
  concat_desc->AddInputDesc("x1", inputTensorDesc);
  concat_desc->UpdateOutputDesc("y", originTensorDesc);
  ge::AttrUtils::SetInt(concat_desc, "N", CONCAT_NUM);

  ge::NodePtr concat_node = graph_->AddNode(concat_desc);
  newNodes_->push_back(concat_node);
  newNodes_->push_back(concatDimNode);

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(concatDimNode->GetOutDataAnchor(0),
                                                       concat_node->GetInDataAnchor(0)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add silce forward edge to fusion slice_node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusedNodes[0]->GetOutDataAnchor(0),
                                                       concat_node->GetInDataAnchor(FIRST_INPUT)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add silce forward edge to fusion slice_node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusedNodes[1]->GetOutDataAnchor(0),
                                                       concat_node->GetInDataAnchor(SECOND_INPUT)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add slice reverse edge to fusion slice_node failed."),
                    return FAILED);

  for (InDataAnchorPtr oriTopPeerAnchorPtr : fusedNode_->GetOutDataAnchor(nodeDims[0])->GetPeerInDataAnchors()) {
    oriTopPeerAnchorPtr->UnlinkAll();
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0), oriTopPeerAnchorPtr),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "add concat node to fusion node output failed."),
                      return FAILED);
  }
  return SUCCESS;
}

Status GRUFusionPass::AddInputNodes(ge::NodePtr splitNode, ge::NodePtr gruNode) {
  bool hasBias = fusedDesc_->MutableInputDesc("b") != nullptr;
  bool hasSeqLength = fusedDesc_->MutableInputDesc("sequence_lens") != nullptr;
  bool hasInitH = fusedDesc_->MutableInputDesc("initial_h") != nullptr;

  // connect bias(splitD) to gru bias input
  if (hasBias) {
    for (int i = 0; i < SPLIT_GROUP; i++) {
      FUSION_PASS_CHECK(
          GRAPH_SUCCESS != GraphUtils::AddEdge(splitNode->GetOutDataAnchor(i),
                                               gruNode->GetInDataAnchor(i + BIAS_INPUT_INDEX)),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add slice to conv edge fail"), return FAILED);
    }
  }

  // connect x
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(fusedNode_->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                         gruNode->GetInDataAnchor(0)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add x edge to gruNode failed."), return FAILED);

  // connect weight_input
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(fusedNode_->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                         gruNode->GetInDataAnchor(1)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add weightInput edge to gruNode failed."), return FAILED);

  // connect weight_hidden
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(fusedNode_->GetInDataAnchor(2)->GetPeerOutAnchor(),
                                         gruNode->GetInDataAnchor(2)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add weightHidden edge to gruNode failed."), return FAILED);

  if (hasSeqLength) {
    ge::GraphUtils::AddEdge(fusedNode_->GetInDataAnchor(SEQUENCE_LENS_INDEX)->GetPeerOutAnchor(),
                            gruNode->GetInDataAnchor(SEQ_LENGTH_INDEX));
  }
  if (hasInitH) {
    ge::GraphUtils::AddEdge(fusedNode_->GetInDataAnchor(INITIAL_H_INDEX)->GetPeerOutAnchor(),
                            gruNode->GetInDataAnchor(INIT_H_INDEX));
  }
  return SUCCESS;
}

Status GRUFusionPass::AddBidiInputNodes(InputNodes nodes) {
  bool hasBias = fusedDesc_->MutableInputDesc("b") != nullptr;
  bool hasSeqLength = fusedDesc_->MutableInputDesc("sequence_lens") != nullptr;
  bool hasInitH = fusedDesc_->MutableInputDesc("initial_h") != nullptr;

  // connect x
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(fusedNode_->GetInDataAnchor(X_INDEX)->GetPeerOutAnchor(),
                                         nodes.dynamicGruV2ForwardNode->GetInDataAnchor(X_INDEX)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Forward add x edge failed."), return FAILED);

  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(fusedNode_->GetInDataAnchor(X_INDEX)->GetPeerOutAnchor(),
                                         nodes.dynamicGRUV2ReverseNode->GetInDataAnchor(X_INDEX)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Reverse add x edge failed."), return FAILED);

  // connect weight_input
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(nodes.weightInputSplitNode->GetOutDataAnchor(SPLIT_FORWARD_INDEX),
                                         nodes.dynamicGruV2ForwardNode->GetInDataAnchor(WEIGHT_INPUT_INDEX)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Forward add weightInput edge failed."), return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(nodes.weightInputSplitNode->GetOutDataAnchor(SPLIT_REVERSE_INDEX),
                                         nodes.dynamicGRUV2ReverseNode->GetInDataAnchor(WEIGHT_INPUT_INDEX)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Reverse add weightInput edge failed."), return FAILED);

  // connect weight_hidden
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(nodes.weightHiddenSplitNode->GetOutDataAnchor(SPLIT_FORWARD_INDEX),
                                         nodes.dynamicGruV2ForwardNode->GetInDataAnchor(WEIGHT_HIDDEN_INDEX)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Forward add weightHidden edge failed."), return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(nodes.weightHiddenSplitNode->GetOutDataAnchor(SPLIT_REVERSE_INDEX),
                                         nodes.dynamicGRUV2ReverseNode->GetInDataAnchor(WEIGHT_HIDDEN_INDEX)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Reverse add weightHidden edge failed."), return FAILED);

  // connect bias
  if (hasBias) {
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(nodes.inputBiasSplitNode->GetOutDataAnchor(SPLIT_FORWARD_INDEX),
                                           nodes.dynamicGruV2ForwardNode->GetInDataAnchor(BIAS_INPUT_INDEX)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Forward add inputBias edge failed."), return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(nodes.inputBiasSplitNode->GetOutDataAnchor(SPLIT_REVERSE_INDEX),
                                           nodes.dynamicGRUV2ReverseNode->GetInDataAnchor(BIAS_INPUT_INDEX)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Reverse add inputBias edge failed."), return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(nodes.hiddenBiasSplitNode->GetOutDataAnchor(SPLIT_FORWARD_INDEX),
                                           nodes.dynamicGruV2ForwardNode->GetInDataAnchor(BIAS_HIDDEN_INDEX)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Forward add hiddenBias edge failed."), return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(nodes.hiddenBiasSplitNode->GetOutDataAnchor(SPLIT_REVERSE_INDEX),
                                           nodes.dynamicGRUV2ReverseNode->GetInDataAnchor(BIAS_HIDDEN_INDEX)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Reverse add hiddenBias edge failed."), return FAILED);
  }

  // connect seq_length
  if (hasSeqLength) {
    ge::NodePtr seqLengthNode = fusedNode_->GetInDataAnchor(SEQUENCE_LENS_INDEX)->GetPeerOutAnchor()->GetOwnerNode();
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(seqLengthNode->GetOutDataAnchor(SINGLE_OUTPUT_INDEX),
                                           nodes.dynamicGruV2ForwardNode->GetInDataAnchor(SEQ_LENGTH_INDEX)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Forward add seqLength edge failed."), return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(seqLengthNode->GetOutDataAnchor(SINGLE_OUTPUT_INDEX),
                                           nodes.dynamicGRUV2ReverseNode->GetInDataAnchor(SEQ_LENGTH_INDEX)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Reverse add seqLength edge failed."), return FAILED);
  }

  // connect init_h
  if (hasInitH) {
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(nodes.initHSplitNode->GetOutDataAnchor(SPLIT_FORWARD_INDEX),
                                           nodes.dynamicGruV2ForwardNode->GetInDataAnchor(INIT_H_INDEX)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Forward add initH edge failed."), return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(nodes.initHSplitNode->GetOutDataAnchor(SPLIT_REVERSE_INDEX),
                                           nodes.dynamicGRUV2ReverseNode->GetInDataAnchor(INIT_H_INDEX)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Reverse add initH edge failed."), return FAILED);
  }
  return SUCCESS;
}

void GRUFusionPass::UpdateOutputDesc(ge::OpDescPtr gruOpDesc, ge::GeTensorDesc& tensorDesc) {
  gruOpDesc->UpdateOutputDesc("y", tensorDesc);
  gruOpDesc->UpdateOutputDesc("output_h", tensorDesc);
  gruOpDesc->UpdateOutputDesc("update", tensorDesc);
  gruOpDesc->UpdateOutputDesc("reset", tensorDesc);
  gruOpDesc->UpdateOutputDesc("new", tensorDesc);
  gruOpDesc->UpdateOutputDesc("hidden_new", tensorDesc);
}

void GRUFusionPass::UnlinkAllAnchors() {
  // unlink all control input of CommonGRU
  if (fusedNode_->GetInControlAnchor() != nullptr) {
    fusedNode_->GetInControlAnchor()->UnlinkAll();
  }

  // unlink all input of CommonGRU
  for (auto inAnchor : fusedNode_->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }

  // unlink all output of CommonGRU
  for (auto outAnchor : fusedNode_->GetAllOutDataAnchors()) {
    if (outAnchor != nullptr) {
      outAnchor->UnlinkAll();
    }
  }
}

Status GRUFusionPass::InitParams(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  // get the NodePtr of GRU
  fusedNode_ = GetNodeFromMapping(PATTERN_FUSED_NODE, mapping);
  FUSION_PASS_CHECK(fusedNode_ == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode_ is null."), return PARAM_INVALID);

  // get the OpDescPtr of GRU
  fusedDesc_ = fusedNode_->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc_ == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedDesc_ is null."), return PARAM_INVALID);
  graph_ = &graph;
  newNodes_ = &newNodes;
  FUSION_PASS_CHECK(SUCCESS != CheckParams(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CheckParams fail"), return FAILED);

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