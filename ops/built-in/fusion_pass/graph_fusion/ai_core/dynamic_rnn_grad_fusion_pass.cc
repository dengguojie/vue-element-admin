/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 *
 * @brief DynamicRNNGrad fusion pass(DynamicRNNGrad --> LSTMIInputGrad & LSTMWeightGrad(Concat&Matmul&Reduce))
 *
 */

#include "dynamic_rnn_grad_fusion_pass.h"

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

static const int BLOCKSIZE = 16;
static const char* FUSED_NODE = "DynamicRNNGrad";
static const std::string PATTERN_FUSEDNODE = "DynamicRNNGrad";

vector<FusionPattern*> DynamicRNNGradFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("DynamicRNNGradFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

ge::NodePtr DynamicRNNGradFusionPass::AddLSTMInputGradNode(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph,
                                                           vector<ge::NodePtr>& newNodes, bool& failStatus) {
  ge::OpDescPtr dynamicRNNGradDesc = dynamicRNNGradNode->GetOpDesc();
  // create lstm_input_grad desc
  ge::OpDescPtr lstmInputGradDesc =
      std::make_shared<ge::OpDesc>(dynamicRNNGradDesc->GetName() + "/LSTMInputGrad", "LSTMInputGrad");


  //add Input
  lstmInputGradDesc->AddInputDesc("w", dynamicRNNGradDesc->GetInputDesc(1));
  lstmInputGradDesc->AddInputDesc("init_c", dynamicRNNGradDesc->GetInputDesc(5));
  lstmInputGradDesc->AddInputDesc("c", dynamicRNNGradDesc->GetInputDesc(7));
  lstmInputGradDesc->AddInputDesc("dy", dynamicRNNGradDesc->GetInputDesc(8));
  lstmInputGradDesc->AddInputDesc("dh", dynamicRNNGradDesc->GetInputDesc(9));
  lstmInputGradDesc->AddInputDesc("dc", dynamicRNNGradDesc->GetInputDesc(10));
  lstmInputGradDesc->AddInputDesc("i", dynamicRNNGradDesc->GetInputDesc(11));
  lstmInputGradDesc->AddInputDesc("j", dynamicRNNGradDesc->GetInputDesc(12));
  lstmInputGradDesc->AddInputDesc("f", dynamicRNNGradDesc->GetInputDesc(13));
  lstmInputGradDesc->AddInputDesc("o", dynamicRNNGradDesc->GetInputDesc(14));
  lstmInputGradDesc->AddInputDesc("tanhct", dynamicRNNGradDesc->GetInputDesc(15));
  // input

  ge::GeTensorDesc inputTensorDescC = lstmInputGradDesc->GetInputDesc(2);
  // output
  lstmInputGradDesc->AddOutputDesc("dx", dynamicRNNGradDesc->GetOutputDesc(2));
  lstmInputGradDesc->AddOutputDesc("dh_prev", dynamicRNNGradDesc->GetOutputDesc(3));
  lstmInputGradDesc->AddOutputDesc("dc_prev", dynamicRNNGradDesc->GetOutputDesc(4));

  // shape:{t,batch_size,4*hidden_size}
  vector<int64_t> outputDims;
  outputDims.push_back(inputTensorDescC.GetShape().GetDim(0));
  outputDims.push_back(inputTensorDescC.GetShape().GetDim(1));
  outputDims.push_back(4 * inputTensorDescC.GetShape().GetDim(2));
  ge::GeShape outputOriginShape(outputDims);
  ge::GeShape outputShape(outputDims);
  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_ND, inputTensorDescC.GetDataType());
  outputTensorDesc.SetOriginShape(outputOriginShape);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  lstmInputGradDesc->AddOutputDesc("dgate", outputTensorDesc);

  // create lstm_input_grad node
  ge::NodePtr lstmInputGradNode = graph.AddNode(lstmInputGradDesc);

  FUSION_PASS_CHECK(
      lstmInputGradNode == nullptr,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", lstmInputGradNode->GetName().c_str()),
      failStatus = true);
  newNodes.push_back(lstmInputGradNode);

  // Edge
  // add w
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
                          lstmInputGradNode->GetInDataAnchor(0));
  // add init_c
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(5)->GetPeerOutAnchor(),
                          lstmInputGradNode->GetInDataAnchor(1));
  // add c
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(7)->GetPeerOutAnchor(),
                          lstmInputGradNode->GetInDataAnchor(2));
  // add dy and others
  for (unsigned int i = 3; i < dynamicRNNGradNode->GetAllInDataAnchors().size() - 10; i++) {
    ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(i + 5)->GetPeerOutAnchor(),
                            lstmInputGradNode->GetInDataAnchor(i));
  }

  for (unsigned int i = 2; i < 5; i++) {
    if (dynamicRNNGradNode->GetOutDataAnchor(i)->GetPeerInDataAnchors().size() > 0) {
      for (InDataAnchorPtr inAnchorPtr : dynamicRNNGradNode->GetOutDataAnchor(i)->GetPeerInDataAnchors()) {
        inAnchorPtr->UnlinkAll();
        ge::GraphUtils::AddEdge(lstmInputGradNode->GetOutDataAnchor(i - 2), inAnchorPtr);
      }
    }
  }

  return lstmInputGradNode;
}

ge::NodePtr DynamicRNNGradFusionPass::AddSplitNode(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph,
                                                   vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create concat desc
  ge::OpDescPtr splitDesc =
      std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Dw/SplitVD", "SplitVD");

  // input
  ge::GeTensorDesc inputTensorDescH = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6);  // h
  splitDesc->AddInputDesc("input_h", inputTensorDescH);
  // output shape:{t-1,batch_size,hidden_size}
  vector<int64_t> outputDims;
  outputDims.push_back(inputTensorDescH.GetShape().GetDim(0) - 1);
  outputDims.push_back(inputTensorDescH.GetShape().GetDim(1));
  outputDims.push_back(inputTensorDescH.GetShape().GetDim(2));
  ge::GeShape outputShape(outputDims);
  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_ND, inputTensorDescH.GetDataType());
  outputTensorDesc.SetOriginShape(outputShape);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  splitDesc->AddOutputDesc("split_t_1", outputTensorDesc);

  // output shape:{1,batch_size,hidden_size}
  vector<int64_t> outputLastDims;
  outputLastDims.push_back(1);
  outputLastDims.push_back(inputTensorDescH.GetShape().GetDim(1));
  outputLastDims.push_back(inputTensorDescH.GetShape().GetDim(2));
  ge::GeShape outputLastShape(outputLastDims);
  ge::GeTensorDesc outputLastTensorDesc =
      ge::GeTensorDesc(outputLastShape, ge::FORMAT_ND, inputTensorDescH.GetDataType());
  outputTensorDesc.SetOriginShape(outputLastShape);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  splitDesc->AddOutputDesc("split_1", outputLastTensorDesc);

  // attr
  vector<int64_t> size_splits;
  size_splits.push_back(inputTensorDescH.GetShape().GetDim(0) - 1);
  size_splits.push_back(1);

  ge::AttrUtils::SetListInt(splitDesc, "size_splits", size_splits);
  ge::AttrUtils::SetInt(splitDesc, "split_dim", 0);
  ge::AttrUtils::SetInt(splitDesc, "num_split", 2);

  // create concat node
  ge::NodePtr splitNode = graph.AddNode(splitDesc);
  FUSION_PASS_CHECK(
      splitNode == nullptr,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", splitNode->GetName().c_str()),
      failStatus = true);
  newNodes.push_back(splitNode);

  // Edge
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(6)->GetPeerOutAnchor(),
                          splitNode->GetInDataAnchor(0));  // h

  return splitNode;
}

ge::NodePtr DynamicRNNGradFusionPass::AddHConcatNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr splitNode,
                                                     ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                     bool& failStatus) {
  // create concat desc
  ge::OpDescPtr concatDesc =
      std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Dw/HConcatD", "ConcatD");

  // input
  ge::GeTensorDesc inputTensorDescInitH = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(4);  // init_h
  ge::GeTensorDesc inputTensorDescSplitH = splitNode->GetOpDesc()->GetOutputDesc(0).Clone();

  vector<int64_t> input_h;
  input_h.push_back(1);
  input_h.push_back(inputTensorDescInitH.GetShape().GetDim(0));
  input_h.push_back(inputTensorDescInitH.GetShape().GetDim(1));
  ge::GeShape init_hShape(input_h);
  inputTensorDescInitH.SetShape(init_hShape);
  inputTensorDescInitH.SetOriginShape(init_hShape);

  concatDesc->AddInputDesc("input_init_h", inputTensorDescInitH);
  concatDesc->AddInputDesc("input_split_h", inputTensorDescSplitH);

  // output shape:{t,batch_size,hidden_size}
  vector<int64_t> outputDims;
  outputDims.push_back(inputTensorDescSplitH.GetShape().GetDim(0) + 1);
  outputDims.push_back(inputTensorDescInitH.GetShape().GetDim(1));
  outputDims.push_back(inputTensorDescInitH.GetShape().GetDim(2));
  ge::GeShape outputShape(outputDims);
  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_ND, inputTensorDescInitH.GetDataType());
  outputTensorDesc.SetOriginShape(outputShape);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  concatDesc->AddOutputDesc("concat_h", outputTensorDesc);
  // attr
  ge::AttrUtils::SetInt(concatDesc, "concat_dim", 0);
  ge::AttrUtils::SetInt(concatDesc, "N", 2);

  // create concat node
  ge::NodePtr concatNode = graph.AddNode(concatDesc);
  FUSION_PASS_CHECK(
      concatNode == nullptr,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", concatNode->GetName().c_str()),
      failStatus = true);
  newNodes.push_back(concatNode);

  // Edge
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(4)->GetPeerOutAnchor(),
                          concatNode->GetInDataAnchor(0));  // Init_h
  ge::GraphUtils::AddEdge(splitNode->GetOutDataAnchor(0), concatNode->GetInDataAnchor(1));

  return concatNode;
}

ge::NodePtr DynamicRNNGradFusionPass::AddConcatNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr hConcatNode,
                                                    ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                    bool& failStatus) {
  // create concat desc
  ge::OpDescPtr concatDesc =
      std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Dw/ConcatD", "ConcatD");

  // input
  ge::GeTensorDesc inputTensorDescX = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(0);  // x
  ge::GeTensorDesc inputTensorDescH = hConcatNode->GetOpDesc()->GetOutputDesc(0).Clone();
  concatDesc->AddInputDesc("input_x", inputTensorDescX);
  concatDesc->AddInputDesc("input_h", inputTensorDescH);
  // output shape:{t,batch_size,input_size+hidden_size}
  vector<int64_t> outputDims;
  outputDims.push_back(inputTensorDescX.GetShape().GetDim(0));
  outputDims.push_back(inputTensorDescX.GetShape().GetDim(1));
  outputDims.push_back(inputTensorDescX.GetShape().GetDim(2) + inputTensorDescH.GetShape().GetDim(2));
  ge::GeShape outputShape(outputDims);
  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_ND, inputTensorDescX.GetDataType());
  outputTensorDesc.SetOriginShape(outputShape);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  concatDesc->AddOutputDesc("concat_xh", outputTensorDesc);
  // attr
  ge::AttrUtils::SetInt(concatDesc, "concat_dim", 2);
  ge::AttrUtils::SetInt(concatDesc, "N", 2);

  // create concat node
  ge::NodePtr concatNode = graph.AddNode(concatDesc);
  FUSION_PASS_CHECK(
      concatNode == nullptr,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", concatNode->GetName().c_str()),
      failStatus = true);
  newNodes.push_back(concatNode);

  // Edge
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                          concatNode->GetInDataAnchor(0));  // x
  ge::GraphUtils::AddEdge(hConcatNode->GetOutDataAnchor(0), concatNode->GetInDataAnchor(1));

  return concatNode;
}
ge::NodePtr DynamicRNNGradFusionPass::AddConcatNodeT_1(ge::NodePtr dynamicRNNGradNode,
                                                       ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                       bool& failStatus) {
  // create concat desc
  ge::OpDescPtr concatDesc =
      std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Dw/ConcatD", "ConcatD");

  // input
  ge::GeTensorDesc inputTensorDescX = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(0);  // x
  //ge::GeTensorDesc inputTensorDescH = hConcatNode->GetOpDesc()->GetOutputDesc(0).Clone();
  ge::GeTensorDesc inputTensorDescInitH = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(4);
  concatDesc->AddInputDesc("input_x", inputTensorDescX);

  vector<int64_t> input_h;
  input_h.push_back(1);
  input_h.push_back(inputTensorDescInitH.GetShape().GetDim(0));
  input_h.push_back(inputTensorDescInitH.GetShape().GetDim(1));
  ge::GeShape init_hShape(input_h);
  inputTensorDescInitH.SetShape(init_hShape);
  inputTensorDescInitH.SetOriginShape(init_hShape);
  concatDesc->AddInputDesc("input_init_h", inputTensorDescInitH);
  // concatDesc->AddInputDesc("input_h", inputTensorDescH);
  // output shape:{t,batch_size,input_size+hidden_size}
  vector<int64_t> outputDims;
  outputDims.push_back(inputTensorDescX.GetShape().GetDim(0));
  outputDims.push_back(inputTensorDescX.GetShape().GetDim(1));
  outputDims.push_back(inputTensorDescX.GetShape().GetDim(2) + inputTensorDescInitH.GetShape().GetDim(2));
  ge::GeShape outputShape(outputDims);
  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_ND, inputTensorDescX.GetDataType());
  outputTensorDesc.SetOriginShape(outputShape);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  concatDesc->AddOutputDesc("concat_xh", outputTensorDesc);
  // attr
  ge::AttrUtils::SetInt(concatDesc, "concat_dim", 2);
  ge::AttrUtils::SetInt(concatDesc, "N", 2);

  // create concat node
  ge::NodePtr concatNode = graph.AddNode(concatDesc);
  FUSION_PASS_CHECK(concatNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                                                   concatNode->GetName().c_str()),
                    failStatus = true);
  newNodes.push_back(concatNode);

  // Edge
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                          concatNode->GetInDataAnchor(0));  // x
  //ge::GraphUtils::AddEdge(hConcatNode->GetOutDataAnchor(0), concatNode->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(4)->GetPeerOutAnchor(),
                          concatNode->GetInDataAnchor(1));
  return concatNode;
}
ge::NodePtr DynamicRNNGradFusionPass::AddMatmulNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr concatNode,
                                                    ge::NodePtr lstmInputGradNode, ge::ComputeGraph& graph,
                                                    vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create matmul desc
  ge::OpDescPtr matmulDesc =
      std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "LSTMWeightGrad/BatchMatmul", "BatchMatMul");

  // input
  ge::GeTensorDesc inputTensorDescXh = concatNode->GetOpDesc()->GetOutputDesc(0).Clone();
  ge::GeTensorDesc inputTensorDescDgate = lstmInputGradNode->GetOpDesc()->GetOutputDesc(3).Clone();  // dgate
  inputTensorDescXh.SetDataType(ge::DT_FLOAT16);
  inputTensorDescDgate.SetDataType(ge::DT_FLOAT16);

  matmulDesc->AddInputDesc("input_xh", inputTensorDescXh);
  matmulDesc->AddInputDesc("input_dgate", inputTensorDescDgate);
  // output shape:{t,input_size+hidden_size,4*hide_size}
  vector<int64_t> outputDims;
  outputDims.push_back(inputTensorDescXh.GetShape().GetDim(0));
  outputDims.push_back(inputTensorDescXh.GetShape().GetDim(2));
  outputDims.push_back(inputTensorDescDgate.GetOriginShape().GetDim(2));
  ge::GeShape outputOriginShape(outputDims);
  ge::GeShape outputShape(outputDims);
  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_ND, ge::DT_FLOAT16);
  outputTensorDesc.SetOriginShape(outputOriginShape);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  matmulDesc->AddOutputDesc("y", outputTensorDesc);
  // attr
  ge::AttrUtils::SetBool(matmulDesc, "adj_x1", true);
  ge::AttrUtils::SetBool(matmulDesc, "adj_x2", false);

  // create matmul node
  ge::NodePtr matmulNode = graph.AddNode(matmulDesc);
  FUSION_PASS_CHECK(
      matmulNode == nullptr,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", matmulNode->GetName().c_str()),
      failStatus = true);
  newNodes.push_back(matmulNode);

  // Edge
  ge::GraphUtils::AddEdge(concatNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(lstmInputGradNode->GetOutDataAnchor(3), matmulNode->GetInDataAnchor(1));  // dgate

  return matmulNode;
}

Status DynamicRNNGradFusionPass::AddDwReduceSumNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr matmulNode,
                                                    ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes) {
  // create reduce_sum desc
  ge::OpDescPtr reduceSumDesc =
      std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Dw/ReduceSumD", "ReduceSumD");

  // input
  ge::GeTensorDesc inputTensorDescMatmul = matmulNode->GetOpDesc()->GetOutputDesc(0).Clone();
  inputTensorDescMatmul.SetDataType(dynamicRNNGradNode->GetOpDesc()->GetInputDesc(0).GetDataType());  // x
  reduceSumDesc->AddInputDesc("input_matmul", inputTensorDescMatmul);
  // output
  ge::GeTensorDesc outputTensorDesc = dynamicRNNGradNode->GetOpDesc()->GetOutputDesc(0).Clone();  // dw
  reduceSumDesc->AddOutputDesc("y", outputTensorDesc);
  // attr
  ge::AttrUtils::SetListInt(reduceSumDesc, "axes", {0});
  ge::AttrUtils::SetBool(reduceSumDesc, "keep_dims", false);

  // create reduce_sum node
  ge::NodePtr reduceSumNode = graph.AddNode(reduceSumDesc);
  FUSION_PASS_CHECK(
      reduceSumNode == nullptr,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", reduceSumNode->GetName().c_str()),
      return FAILED);
  newNodes.push_back(reduceSumNode);

  // Edge
  ge::GraphUtils::AddEdge(matmulNode->GetOutDataAnchor(0), reduceSumNode->GetInDataAnchor(0));
  if (dynamicRNNGradNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    for (InDataAnchorPtr inAnchorPtr : dynamicRNNGradNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {  // dw
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(reduceSumNode->GetOutDataAnchor(0), inAnchorPtr);
    }
  }

  return SUCCESS;
}

Status DynamicRNNGradFusionPass::AddDbReduceSumNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr lstmInputGradNode,
                                                    ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes) {
  // create reduce_sum desc
  ge::OpDescPtr reduceSumDesc =
      std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Db/ReduceSumD", "ReduceSumD");

  // input
  ge::GeTensorDesc inputTensorDescDgate = lstmInputGradNode->GetOpDesc()->GetOutputDesc(3).Clone();  // dgate
  inputTensorDescDgate.SetDataType(dynamicRNNGradNode->GetOpDesc()->GetInputDesc(0).GetDataType());  // x
  reduceSumDesc->AddInputDesc("input_dgate", inputTensorDescDgate);

  // output
  ge::GeTensorDesc outputTensorDesc = dynamicRNNGradNode->GetOpDesc()->GetOutputDesc(1).Clone();  // db
  reduceSumDesc->AddOutputDesc("y", outputTensorDesc);

  // attr
  ge::AttrUtils::SetListInt(reduceSumDesc, "axes", {0, 1});
  ge::AttrUtils::SetBool(reduceSumDesc, "keep_dims", false);

  // create reduce_sum node
  ge::NodePtr reduceSumNode = graph.AddNode(reduceSumDesc);
  FUSION_PASS_CHECK(
      reduceSumNode == nullptr,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", reduceSumNode->GetName().c_str()),
      return FAILED);
  newNodes.push_back(reduceSumNode);

  // Edge
  ge::GraphUtils::AddEdge(lstmInputGradNode->GetOutDataAnchor(3), reduceSumNode->GetInDataAnchor(0));      // dgate
  if (dynamicRNNGradNode->GetOutDataAnchor(1)->GetPeerInDataAnchors().size() > 0) {                        // db
    for (InDataAnchorPtr inAnchorPtr : dynamicRNNGradNode->GetOutDataAnchor(1)->GetPeerInDataAnchors()) {  // db
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(reduceSumNode->GetOutDataAnchor(0), inAnchorPtr);
    }
  }

  return SUCCESS;
}

Status DynamicRNNGradFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  bool failStatus = false;
  // get dynamicRNNGradNode
  ge::NodePtr dynamicRNNGradNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);

  // add lstmInputGrad
  ge::NodePtr lstmInputGradNode = AddLSTMInputGradNode(dynamicRNNGradNode, graph, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddLSTMInputGradNode:check failed, fusion failed."),
                    return FAILED);

  int t_size = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6).GetShape().GetDim(0);
  // add split
  ge::NodePtr concatNode = nullptr;
  if (t_size != 1)
  {
        ge::NodePtr splitNode = AddSplitNode(dynamicRNNGradNode, graph, newNodes, failStatus);
        FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddSplitNode:check failed, fusion failed."),
                          return FAILED);
        // add concat
        ge::NodePtr hConcatNode = AddHConcatNode(dynamicRNNGradNode, splitNode, graph, newNodes, failStatus);
        FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddHConcatNode:check failed, fusion failed."),
                          return FAILED);
        // add concat
        concatNode = AddConcatNode(dynamicRNNGradNode, hConcatNode, graph, newNodes, failStatus);
        FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddConcatNode:check failed, fusion failed."),
                          return FAILED);
  }else{
        // add concat
        concatNode = AddConcatNodeT_1(dynamicRNNGradNode, graph, newNodes, failStatus);
        FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddConcatNode:check failed, fusion failed."),
                          return FAILED);

  }
  // add matmul
  ge::NodePtr matmulNode =
      AddMatmulNode(dynamicRNNGradNode, concatNode, lstmInputGradNode, graph, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddMatmulNode:check failed, fusion failed."),
                    return FAILED);
  // add dw reduce_sum
  AddDwReduceSumNode(dynamicRNNGradNode, matmulNode, graph, newNodes);
  // add db reduce_sum
  AddDbReduceSumNode(dynamicRNNGradNode, lstmInputGradNode, graph, newNodes);
  // unlink all control input of dynamicRNNGradNode
  if (dynamicRNNGradNode->GetInControlAnchor() != nullptr) {
    dynamicRNNGradNode->GetInControlAnchor()->UnlinkAll();
  }
  // unlink all input of dynamicRNNGradNode
  for (auto inAnchor : dynamicRNNGradNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }
  // remove dynamicRNNGradNode from graph
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(dynamicRNNGradNode),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "remove fusedNode node[%s] failed", dynamicRNNGradNode->GetName().c_str()),
      return FAILED);

  return SUCCESS;
}

REGISTER_PASS("DynamicRNNGradFusionPass", BUILT_IN_GRAPH_PASS, DynamicRNNGradFusionPass);
}  // namespace fe
