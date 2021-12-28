/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 *
 * @brief DynamicLSTM fusion pass
 * (DynamicRNN(BIDIRECTIONAL) --> split + [DynamicRNN + DynamicRNN(REDIRECTIONAL)] + concat)
 *
 */

#include "dynamic_rnn_fusion_pass.h"

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

using namespace ge;
namespace fe {
static const char* FUSED_NODE = "DynamicRNN";
static const std::string PATTERN_FUSEDNODE = "DynamicRNN";
static const int64_t concatNodeIndex = 2;
static const int64_t concatNodeNums = 3;
static const int64_t inputShapeDim = 2;
static const int64_t numSplit = 2;
vector<FusionPattern*> DynamicRNNFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("DynamicRNNFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "DynamicRNN pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);
  return patterns;
}

Status DynamicRNNFusionPass::AddSplitEdge(ge::NodePtr splitNode, ge::NodePtr forwardRNNNode,
                                          ge::NodePtr backwardRNNNode, string nodeName, int64_t index) {
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(splitNode->GetOutDataAnchor(0), forwardRNNNode->GetInDataAnchor(index)),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "add " + nodeName + "'s y to forwardRNNNode's w failed."), return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(splitNode->GetOutDataAnchor(1), backwardRNNNode->GetInDataAnchor(index)),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "add " + nodeName + "'s y to backwardRNNNode's w failed."), return FAILED);

  return SUCCESS;
}

Status DynamicRNNFusionPass::AddConcatEdge(ge::NodePtr concatNode, ge::NodePtr dynamicRNNNode,
                                           ge::NodePtr forwardRNNNode, ge::NodePtr backwardRNNNode, string nodeName,
                                           int64_t index) {
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(forwardRNNNode->GetOutDataAnchor(index), concatNode->GetInDataAnchor(0)),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "add forwardRNNNode's y to " + nodeName + "'s x failed."), return FAILED);

  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(backwardRNNNode->GetOutDataAnchor(index), concatNode->GetInDataAnchor(1)),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "add backwardRNNNode's y to " + nodeName + "'s x failed."), return FAILED);

  OutDataAnchor::Vistor<InDataAnchorPtr> outputAnchors =
      dynamicRNNNode->GetOutDataAnchor(index)->GetPeerInDataAnchors();

  for (auto& in_anchor : outputAnchors) {
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(dynamicRNNNode->GetOutDataAnchor(index), in_anchor),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "remove dynamicRNNNode's outDataAnchor failed."), return FAILED);

    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(concatNode->GetOutDataAnchor(0), in_anchor),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add " + nodeName + "'s y to dynamicRNNNode's y failed."),
                      return FAILED);
  }

  return SUCCESS;
}

Status DynamicRNNFusionPass::AddInputEdge(ge::NodePtr dynamicRNNNode, ge::NodePtr forwardRNNNode,
                                          ge::NodePtr backwardRNNNode, ge::NodePtr splitWNode, ge::NodePtr splitBNode,
                                          ge::NodePtr splitHNode, ge::NodePtr splitCNode, InputIndexInfo inputIndexInfo,
                                          bool has_seq, bool has_h0, bool has_c0) {
  // edge for tensor x
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(dynamicRNNNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                       forwardRNNNode->GetInDataAnchor(inputIndexInfo.xIndex)),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add dynamicRNNNode's x edge to forwardRNNNode's x failed."),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(dynamicRNNNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                       backwardRNNNode->GetInDataAnchor(inputIndexInfo.xIndex)),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add dynamicRNNNode's x edge to backwardRNNNode's x failed."),
                    return FAILED);

  // edge for tensor w
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(dynamicRNNNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                                       splitWNode->GetInDataAnchor(0)),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add dynamicRNNNode's w edge to splitWNode's x failed."),
                    return FAILED);

  // edge for tensor b
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(dynamicRNNNode->GetInDataAnchor(2)->GetPeerOutAnchor(),
                                                       splitBNode->GetInDataAnchor(0)),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add dynamicRNNNode's b edge to splitBNode's x failed."),
                    return FAILED);

  // edge for tensor seq
  if (has_seq) {
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(dynamicRNNNode->GetInDataAnchor(3)->GetPeerOutAnchor(),
                                                         forwardRNNNode->GetInDataAnchor(inputIndexInfo.sIndex)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add dynamicRNNNode's seq edge to forwardRNNNode's seq failed."),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(dynamicRNNNode->GetInDataAnchor(3)->GetPeerOutAnchor(),
                                                         backwardRNNNode->GetInDataAnchor(inputIndexInfo.sIndex)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add dynamicRNNNode's seq edge to backwardRNNNode's seq failed."),
                      return FAILED);
  }

  // edge for tensor init_h
  if (has_h0) {
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(dynamicRNNNode->GetInDataAnchor(4)->GetPeerOutAnchor(),
                                                         splitHNode->GetInDataAnchor(0)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add dynamicRNNNode's init_h edge to splitHNode's x failed."),
                      return FAILED);
  }

  // edge for tensor init_c
  if (has_c0) {
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(dynamicRNNNode->GetInDataAnchor(5)->GetPeerOutAnchor(),
                                                         splitCNode->GetInDataAnchor(0)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add dynamicRNNNode's init_c edge to splitCNode's x failed."),
                      return FAILED);
  }
  return SUCCESS;
}

ge::OpDescPtr DynamicRNNFusionPass::CreateSplitDesc(ge::OpDescPtr splitDesc, ge::OpDescPtr dynamicRNNDesc,
                                                    string tensorName, int64_t splitDim) {
  ge::GeTensorDesc tensorDesc = dynamicRNNDesc->GetInputDesc(tensorName).Clone();
  splitDesc->AddInputDesc(tensorDesc);
  ge::AttrUtils::SetInt(splitDesc, "split_dim", splitDim);
  ge::AttrUtils::SetInt(splitDesc, "num_split", 2);
  vector<int64_t> tensorDims = tensorDesc.GetShape().GetDims();

  tensorDims[splitDim] = tensorDims[splitDim] / numSplit;

  ge::GeShape tensorShape(tensorDims);
  tensorDesc.SetShape(tensorShape);
  tensorDesc.SetOriginShape(tensorShape);

  splitDesc->AddOutputDesc(tensorDesc);
  splitDesc->AddOutputDesc(tensorDesc);

  return splitDesc;
}

ge::OpDescPtr DynamicRNNFusionPass::CreateConcatDesc(ge::OpDescPtr concatDesc, ge::OpDescPtr dynamicRNNDesc,
                                                     string tensorName, int64_t concatDim) {
  ge::GeTensorDesc tensorDesc = dynamicRNNDesc->GetOutputDesc(tensorName).Clone();
  concatDesc->AddOutputDesc(tensorDesc);
  vector<int64_t> tensorDims = tensorDesc.GetShape().GetDims();

  tensorDims[concatDim] = tensorDims[concatDim] / numSplit;

  ge::GeShape tensorShape(tensorDims);
  tensorDesc.SetShape(tensorShape);
  tensorDesc.SetOriginShape(tensorShape);
  tensorDesc.SetOriginFormat(ge::FORMAT_ND);

  concatDesc->AddInputDesc(tensorDesc);
  concatDesc->AddInputDesc(tensorDesc);

  ge::AttrUtils::SetInt(concatDesc, "concat_dim", concatDim);

  return concatDesc;
}

ge::OpDescPtr DynamicRNNFusionPass::CreateRNNDesc(ge::OpDescPtr RNNDesc, ge::OpDescPtr dynamicRNNDesc, string direction,
                                                  bool has_seq, bool has_h0, bool has_c0) {
  // for inputs
  vector<int64_t> tensorXDims = dynamicRNNDesc->GetInputDesc(0).GetShape().GetDims();
  int64_t inputSize = tensorXDims[inputShapeDim];
  ge::AttrUtils::SetInt(RNNDesc, "input_size", inputSize);
  vector<int64_t> tensorOutputDims = dynamicRNNDesc->GetOutputDesc(0).GetShape().GetDims();
  int64_t hiddenSize = tensorOutputDims[inputShapeDim];
  ge::AttrUtils::SetInt(RNNDesc, "hidden_size", hiddenSize);

  // update x
  ge::GeTensorDesc tensorXDesc = dynamicRNNDesc->GetInputDesc("x").Clone();
  RNNDesc->AddInputDesc("x", tensorXDesc);

  // update w
  ge::GeTensorDesc tensorWDesc = dynamicRNNDesc->GetInputDesc("w").Clone();
  vector<int64_t> tensorWDims = tensorWDesc.GetShape().GetDims();
  tensorWDims[0] = tensorWDims[0] / numSplit;
  ge::GeShape tensorWShape(tensorWDims);
  tensorWDesc.SetShape(tensorWShape);
  tensorWDesc.SetOriginShape(tensorWShape);
  RNNDesc->AddInputDesc("w", tensorWDesc);

  // update b
  ge::GeTensorDesc tensorBDesc = dynamicRNNDesc->GetInputDesc("b").Clone();
  vector<int64_t> tensorBDims = tensorBDesc.GetShape().GetDims();
  tensorBDims[0] = tensorBDims[0] / numSplit;
  ge::GeShape tensorBShape(tensorBDims);
  tensorBDesc.SetShape(tensorBShape);
  tensorBDesc.SetOriginShape(tensorBShape);
  RNNDesc->AddInputDesc("b", tensorBDesc);

  // update s
  if (has_seq) {
    ge::GeTensorDesc tensorSDesc = dynamicRNNDesc->GetInputDesc("seq_length").Clone();
    RNNDesc->AddInputDesc("seq_length", tensorSDesc);
  }

  // update h0
  if (has_h0) {
    ge::GeTensorDesc tensorHDesc = dynamicRNNDesc->GetInputDesc("init_h").Clone();
    vector<int64_t> tensorHDims = tensorHDesc.GetShape().GetDims();
    tensorHDims[0] = tensorHDims[0] / numSplit;
    ge::GeShape tensorHShape(tensorHDims);
    tensorHDesc.SetShape(tensorHShape);
    tensorHDesc.SetOriginShape(tensorHShape);
    RNNDesc->AddInputDesc("init_h", tensorHDesc);
  }

  // update c0
  if (has_c0) {
    ge::GeTensorDesc tensorCDesc = dynamicRNNDesc->GetInputDesc("init_c").Clone();
    vector<int64_t> tensorCDims = tensorCDesc.GetShape().GetDims();
    tensorCDims[0] = tensorCDims[0] / numSplit;
    ge::GeShape tensorCShape(tensorCDims);
    tensorCDesc.SetShape(tensorCShape);
    tensorCDesc.SetOriginShape(tensorCShape);
    RNNDesc->AddInputDesc("init_c", tensorCDesc);
  }

  // for outputs

  vector<string> outputs = {"y", "output_h", "output_c", "i", "j", "f", "o", "tanhc"};

  ge::GeTensorDesc tensorYDesc = dynamicRNNDesc->GetOutputDesc("y").Clone();
  vector<int64_t> tensorYDims = tensorYDesc.GetShape().GetDims();
  tensorYDims[inputShapeDim] = tensorYDims[inputShapeDim] / numSplit;
  ge::GeShape tensorYShape(tensorYDims);
  tensorYDesc.SetShape(tensorYShape);
  tensorYDesc.SetOriginShape(tensorYShape);
  tensorYDesc.SetOriginFormat(ge::FORMAT_ND);

  for (size_t index = 0; index < outputs.size(); index++) {
    RNNDesc->AddOutputDesc(outputs[index], tensorYDesc);
  }

  // for attrs

  // cell_type
  string cell_type = "LSTM";
  if (ge::AttrUtils::GetStr(dynamicRNNDesc, "cell_type", cell_type)) {
    ge::AttrUtils::SetStr(RNNDesc, "cell_type", cell_type);
  }

  // direction
  ge::AttrUtils::SetStr(RNNDesc, "direction", direction);

  // cell_depth
  int64_t cell_depth = 1;
  if (ge::AttrUtils::GetInt(dynamicRNNDesc, "cell_depth", cell_depth)) {
    ge::AttrUtils::SetInt(RNNDesc, "cell_depth", cell_depth);
  }

  // use_peephole
  bool use_peephole = false;
  if (ge::AttrUtils::GetBool(dynamicRNNDesc, "use_peephole", use_peephole)) {
    ge::AttrUtils::SetBool(RNNDesc, "use_peephole", use_peephole);
  }

  // keep_prob
  float keep_prob = 1.0;
  if (ge::AttrUtils::GetFloat(dynamicRNNDesc, "keep_prob", keep_prob)) {
    ge::AttrUtils::SetFloat(RNNDesc, "keep_prob", keep_prob);
  }
  // cell_clip
  float cell_clip = -1.0;
  if (ge::AttrUtils::GetFloat(dynamicRNNDesc, "cell_clip", cell_clip)) {
    ge::AttrUtils::SetFloat(RNNDesc, "cell_clip", cell_clip);
  }

  // num_proj
  int64_t num_proj = 0;
  if (ge::AttrUtils::GetInt(dynamicRNNDesc, "num_proj", num_proj)) {
    ge::AttrUtils::SetInt(RNNDesc, "num_proj", num_proj);
  }

  // time_major
  bool time_major = true;
  if (ge::AttrUtils::GetBool(dynamicRNNDesc, "time_major", time_major)) {
    ge::AttrUtils::SetBool(RNNDesc, "time_major", time_major);
  }

  // activation
  string activation = "tanh";
  if (ge::AttrUtils::GetStr(dynamicRNNDesc, "activation", activation)) {
    ge::AttrUtils::SetStr(RNNDesc, "activation", activation);
  }

  // forget_bias
  float forget_bias = 0.0;
  if (ge::AttrUtils::GetFloat(dynamicRNNDesc, "forget_bias", forget_bias)) {
    ge::AttrUtils::SetFloat(RNNDesc, "forget_bias", forget_bias);
  }

  // is_training
  bool is_training = true;
  if (ge::AttrUtils::GetBool(dynamicRNNDesc, "is_training", is_training)) {
    ge::AttrUtils::SetBool(RNNDesc, "is_training", is_training);
  }

  return RNNDesc;
}

Status DynamicRNNFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  // get the NodePtr of DynamicRNN
  OP_LOGI(FUSED_OP_TYPE.c_str(), "DynamicRNN fusion start fusion");
  ge::NodePtr dynamicRNNNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(dynamicRNNNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "dynamicRNNNode is null, fusion failed."),
                    return PARAM_INVALID);

  // get the OpDescPtr of DynamicRNN
  ge::OpDescPtr dynamicRNNDesc = dynamicRNNNode->GetOpDesc();
  FUSION_PASS_CHECK(dynamicRNNDesc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "dynamicRNNDesc OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  auto wxhTensorDesc = dynamicRNNDesc->MutableInputDesc("w");
  wxhTensorDesc->SetFormat(ge::FORMAT_ND);
  wxhTensorDesc->SetOriginFormat(ge::FORMAT_ND);
  auto biasTensorDesc = dynamicRNNDesc->MutableInputDesc("b");
  biasTensorDesc->SetFormat(ge::FORMAT_ND);
  biasTensorDesc->SetOriginFormat(ge::FORMAT_ND);

  vector<int64_t> tensorXDims = dynamicRNNDesc->GetInputDesc(0).GetShape().GetDims();
  int64_t inputSize = tensorXDims[inputShapeDim];
  ge::AttrUtils::SetInt(dynamicRNNDesc, "input_size", inputSize);
  vector<int64_t> tensorYDims = dynamicRNNDesc->GetOutputDesc(0).GetShape().GetDims();
  int64_t hiddenSize = tensorYDims[inputShapeDim];
  ge::AttrUtils::SetInt(dynamicRNNDesc, "hidden_size", hiddenSize);

  // check attr_direction
  string direction = "UNIDIRECTIONAL";
  ge::AttrUtils::GetStr(dynamicRNNDesc, "direction", direction);
  if (direction != "BIDIRECTIONAL") {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "direction is BIDIRECTIONAL, fusion failed.");
    return NOT_CHANGED;
  }

  // optional input
  bool has_seq = dynamicRNNDesc->MutableInputDesc("seq_length") != nullptr;
  bool has_h0 = dynamicRNNDesc->MutableInputDesc("init_h") != nullptr;
  bool has_c0 = dynamicRNNDesc->MutableInputDesc("init_c") != nullptr;

  InputIndexInfo inputIndexInfo;

  // add forwardRNN node
  ge::OpDescPtr forwardRNNDesc = nullptr;
  OP_LOGD(FUSED_OP_TYPE.c_str(), "DynamicRNN forwardRNNDesc created");
  FUSION_PASS_MAKE_SHARED(
      (forwardRNNDesc = std::make_shared<ge::OpDesc>(dynamicRNNDesc->GetName() + "/forwardRNN", "DynamicRNN")),
      return INTERNAL_ERROR);
  forwardRNNDesc = CreateRNNDesc(forwardRNNDesc, dynamicRNNDesc, "UNIDIRECTIONAL", has_seq, has_h0, has_c0);
  ge::NodePtr forwardRNNNode = graph.AddNode(forwardRNNDesc);

  // add backwardRNN node
  ge::OpDescPtr backwardRNNDesc = nullptr;
  OP_LOGD(FUSED_OP_TYPE.c_str(), "DynamicRNN backwardRNNDesc created");
  FUSION_PASS_MAKE_SHARED(
      (backwardRNNDesc = std::make_shared<ge::OpDesc>(dynamicRNNDesc->GetName() + "/backwardRNN", "DynamicRNN")),
      return INTERNAL_ERROR);
  backwardRNNDesc = CreateRNNDesc(backwardRNNDesc, dynamicRNNDesc, "REDIRECTIONAL", has_seq, has_h0, has_c0);
  ge::NodePtr backwardRNNNode = graph.AddNode(backwardRNNDesc);

  // add split node for w
  ge::OpDescPtr splitWDesc = nullptr;
  OP_LOGD(FUSED_OP_TYPE.c_str(), "DynamicRNN splitWDesc created");
  FUSION_PASS_MAKE_SHARED((splitWDesc = std::make_shared<ge::OpDesc>(dynamicRNNDesc->GetName() + "/splitW", "SplitD")),
                          return INTERNAL_ERROR);
  splitWDesc = CreateSplitDesc(splitWDesc, dynamicRNNDesc, "w", 0);
  ge::NodePtr splitWNode = graph.AddNode(splitWDesc);

  // add split node for b

  ge::OpDescPtr splitBDesc = nullptr;
  OP_LOGD(FUSED_OP_TYPE.c_str(), "DynamicRNN splitBDesc created");
  FUSION_PASS_MAKE_SHARED((splitBDesc = std::make_shared<ge::OpDesc>(dynamicRNNDesc->GetName() + "/splitB", "SplitD")),
                          return INTERNAL_ERROR);
  splitBDesc = CreateSplitDesc(splitBDesc, dynamicRNNDesc, "b", 0);
  ge::NodePtr splitBNode = graph.AddNode(splitBDesc);

  // add split node for h0
  ge::NodePtr splitHNode = nullptr;
  OP_LOGD(FUSED_OP_TYPE.c_str(), "DynamicRNN splitHNode created");
  if (has_h0) {
    ge::OpDescPtr splitHDesc = nullptr;
    FUSION_PASS_MAKE_SHARED(
        (splitHDesc = std::make_shared<ge::OpDesc>(dynamicRNNDesc->GetName() + "/splitH", "SplitD")),
        return INTERNAL_ERROR);
    splitHDesc = CreateSplitDesc(splitHDesc, dynamicRNNDesc, "init_h", 0);
    splitHNode = graph.AddNode(splitHDesc);
  }

  // add split node for c0
  ge::NodePtr splitCNode = nullptr;
  OP_LOGD(FUSED_OP_TYPE.c_str(), "DynamicRNN splitCNode created");
  if (has_c0) {
    ge::OpDescPtr splitCDesc = nullptr;
    FUSION_PASS_MAKE_SHARED(
        (splitCDesc = std::make_shared<ge::OpDesc>(dynamicRNNDesc->GetName() + "/splitC", "SplitD")),
        return INTERNAL_ERROR);
    splitCDesc = CreateSplitDesc(splitCDesc, dynamicRNNDesc, "init_c", 0);
    splitCNode = graph.AddNode(splitCDesc);
  }

  // add edge for dynamicRNNNode's inputs
  AddInputEdge(dynamicRNNNode, forwardRNNNode, backwardRNNNode, splitWNode, splitBNode, splitHNode, splitCNode,
               inputIndexInfo, has_seq, has_h0, has_c0);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "DynamicRNN AddInputEdge finished");
  // edge for op splitw
  AddSplitEdge(splitWNode, forwardRNNNode, backwardRNNNode, "splitWNode", inputIndexInfo.wIndex);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "DynamicRNN AddSplitEdge finished");
  // edge for op splitB
  AddSplitEdge(splitBNode, forwardRNNNode, backwardRNNNode, "splitBNode", inputIndexInfo.bIndex);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "DynamicRNN AddSplitEdge finished");
  // edge for op splitH
  if (has_h0) {
    AddSplitEdge(splitHNode, forwardRNNNode, backwardRNNNode, "splitHNode", inputIndexInfo.hIndex);
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "DynamicRNN AddSplitEdge finished");
  // edge for op splitC
  if (has_c0) {
    AddSplitEdge(splitCNode, forwardRNNNode, backwardRNNNode, "splitCNode", inputIndexInfo.cIndex);
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "DynamicRNN AddSplitEdge finished");

  vector<string> concatNodes = {"y", "output_h", "output_c", "i", "j", "f", "o", "tanhc"};
  for (size_t index = 0; index < concatNodes.size(); index++) {
    ge::OpDescPtr concatDesc = nullptr;
    OP_LOGD(FUSED_OP_TYPE.c_str(), "DynamicRNN concatDesc of %s created", concatNodes[index].c_str());
    FUSION_PASS_MAKE_SHARED((concatDesc = std::make_shared<ge::OpDesc>(
                                 dynamicRNNDesc->GetName() + "/concat_" + concatNodes[index], "ConcatD")),
                            return INTERNAL_ERROR);
    if (index < concatNodeNums) {
      concatDesc = CreateConcatDesc(concatDesc, dynamicRNNDesc, concatNodes[index], concatNodeIndex);
    } else {
      concatDesc = CreateConcatDesc(concatDesc, dynamicRNNDesc, concatNodes[index], 0);
    }
    ge::NodePtr concatNode = graph.AddNode(concatDesc);
    AddConcatEdge(concatNode, dynamicRNNNode, forwardRNNNode, backwardRNNNode, concatNodes[index], index);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "DynamicRNN AddConcatEdge of %s finished", concatNodes[index].c_str());
  }

  // unlink all input for dynamicRNNNode
  for (auto inAnchor : dynamicRNNNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "DynamicRNN UnlinkAll inanchor finished");
  // unlink all output for dynamicRNNNode
  for (auto outAnchor : dynamicRNNNode->GetAllOutDataAnchors()) {
    if (outAnchor != nullptr) {
      outAnchor->UnlinkAll();
    }
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "DynamicRNN UnlinkAll outAnchor finished");
  // remove dynamicRNNNode
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(dynamicRNNNode),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "remove dynamicRNNNode node[%s] failed", dynamicRNNNode->GetName().c_str()),
      return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "DynamicRNN remove finished");
  return SUCCESS;
}

REGISTER_PASS("DynamicRNNFusionPass", BUILT_IN_GRAPH_PASS, DynamicRNNFusionPass);
}  // namespace fe