/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @file  a_conv2d_mul_fusion_pass.cpp
 *
 * @brief conv-mul fusion pass(conv2d-mul --> conv)
 *
 */

#include "a_conv2d_mul_fusion_pass.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "pattern_fusion_util.h"
#include "op_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace ge;
namespace fe {
  static const string PATTERN_SRC = "src";
  static const string PATTERN_MUL = "mul";
  static const string CONVOLUTION = "Conv2D";
  static const string MUL = "Mul";
  static const std::string CONSTANT = "Const";
  const std::string CONSTANTOP = "Constant";
  static const int FILTER_INDEX = 1;
  static const int BIAS_INDEX = 2;

/* The graph struct need to adapt is shown as follows:
 *
 *               const  conv2d
 *                    \  |
 *                      mul
 *                       |
 *                     output
 *
 *  Notice: the struct can be captured by
 *          conv2d + mul pattern
*/

vector<FusionPattern *> Conv2DMulFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  string passName = "TbeConv2DMulFusion";
  FusionPattern *pattern = new (std::nothrow) FusionPattern(passName);
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."),
           return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName.c_str());
  pattern->AddOpDesc(PATTERN_MUL, {MUL})
          .AddOpDesc(PATTERN_SRC, {CONVOLUTION})
          .SetInputs(PATTERN_MUL, {PATTERN_SRC})
          .SetOutput(PATTERN_MUL);
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName.c_str());

  return patterns;
}

static Status GetInputChannel(ge::NodePtr input_node, int64_t& inputChannel)
{
  ge::GeTensorDesc nodeFm =
           ge::OpDescUtils::GetNonConstInputTensorDesc(input_node, 0);
  ge::GeShape nodeFmShape = nodeFm.GetShape();
  ge::Format nodeFmFormat = nodeFm.GetFormat();
  FUSION_PASS_CHECK(nodeFmShape.GetDims().size() != 4 ||
           (nodeFmFormat != FORMAT_NCHW &&
            nodeFmFormat != FORMAT_NHWC),
           OP_LOGE("Conv2DMulFusionPass", "Mul node's fm shape_dim is %d, format is %d,\
                  fusion failed, valid fm shape_dim is 4, and valid format is NCHW(0) or NHWC(1).",
              nodeFmShape.GetDims().size(), nodeFmFormat),
           return FAILED);
  if (nodeFmFormat == FORMAT_NCHW) {
    inputChannel = nodeFmShape.GetDims()[1];
  } else if (nodeFmFormat == FORMAT_NHWC) {
    inputChannel = nodeFmShape.GetDims()[3];
  }

  return SUCCESS;
}

/*
 * @brief: parse nodes matched in mapping and call graph DoFusion
 * @param [in] graph: original graph
 * @param [in] mapping: matched pattern
 * @param [out] newNodes: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status Conv2DMulFusionPass::Fusion(ge::ComputeGraph &graph,
                                     Mapping &mapping,
                                     vector<ge::NodePtr> &newNodes)
{
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter Conv2DMulFusionPass");
  ge::NodePtr convNode = GetNodeFromMapping(PATTERN_SRC, mapping);
  ge::NodePtr mul_node = GetNodeFromMapping(PATTERN_MUL, mapping);
  FUSION_PASS_CHECK(convNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Node conv2d is null, fusion failed."),
           return PARAM_INVALID);
  FUSION_PASS_CHECK(mul_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Node mul is null, fusion failed."),
           return PARAM_INVALID);

  if (convNode->GetOutDataNodes().size() > 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "out data size is invalid.");
    return NOT_CHANGED;
  }
  OutDataAnchorPtr biasAnchor = nullptr;
  InDataAnchorPtr convBiasInAnchor = nullptr;
  NodePtr biasNode = nullptr;

  vector<ge::NodePtr> mulConstNodes =
            ge::OpDescUtils::GetConstInputs(mul_node);
  vector<ge::NodePtr> convConstNodes =
            ge::OpDescUtils::GetConstInputs(convNode);
  auto mulConstNodeSize = mulConstNodes.size();
  FUSION_PASS_CHECK(mulConstNodeSize != 1,
            OP_LOGI(FUSED_OP_TYPE.c_str(), "Mul_Node:[%s]'s const_node's size %u is invalid.",
                    mul_node->GetName().c_str(), mulConstNodeSize),
            return NOT_CHANGED);

  Status result =
           PatternFusionUtil::CopyMultiReferenceConstNode(graph, convNode);
  FUSION_PASS_CHECK(result != SUCCESS,
           OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv_Node[%s]: can not copy multiReference const node.",
                    convNode->GetName().c_str()),
           return NOT_CHANGED);

  ge::OpDescPtr src_op = convNode->GetOpDesc();
  FUSION_PASS_CHECK(src_op == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.",
           convNode->GetName().c_str()),
           return PARAM_INVALID);
  std::map<string, uint32_t> inputNameMap = src_op->GetAllInputName();
  ge::OpDescPtr mul_op = mul_node->GetOpDesc();
  FUSION_PASS_CHECK(mul_op == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.",
           mul_node->GetName().c_str()),
           return PARAM_INVALID);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2DMulFusionPass: conv2d [%s] has %u input anchor.",
           convNode->GetName().c_str(), convNode->GetAllInDataAnchors().size());
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2DMulFusionPass: conv2d [%s] has %u input desc.",
           convNode->GetName().c_str(), src_op->GetAllInputsDesc().size());
  int32_t in_edges_size = convNode->GetInDataNodes().size();
  if (in_edges_size < 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "inEdges size is invalid.");
    return NOT_CHANGED;
  }

  // get conv2d's weights & bias
  vector<ge::ConstGeTensorPtr> conv2dWeights =
           ge::OpDescUtils::GetWeights(convNode);
  bool hasBias = true;
  if (conv2dWeights.size() < BIAS_INDEX) {
    hasBias = false;
  }

  OutDataAnchorPtr filterAnchor = convNode->GetInDataAnchor(1)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(filterAnchor == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "filter output anchor is null"),
           return PARAM_INVALID);
  NodePtr filterNode = filterAnchor->GetOwnerNode();
  FUSION_PASS_CHECK(filterNode == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Conv2DMulFusionPass: filterNode is not exist."),
           return PARAM_INVALID);
  FUSION_PASS_CHECK(filterNode->GetType() != "Const",
           OP_LOGW(FUSED_OP_TYPE.c_str(), "Conv2DMulFusionPass: filter is not const Node."),
           return NOT_CHANGED);
  if (hasBias) {
      biasAnchor = convNode->GetInDataAnchor(2)->GetPeerOutAnchor();
      FUSION_PASS_CHECK(biasAnchor == nullptr,
              OP_LOGE(FUSED_OP_TYPE.c_str(), "bias anchor is null"),
              return PARAM_INVALID);
      biasNode = biasAnchor->GetOwnerNode();
      FUSION_PASS_CHECK(biasNode == nullptr,
              OP_LOGE(FUSED_OP_TYPE.c_str(), "Conv2DMulFusionPass: biasNode is not exist."),
              return PARAM_INVALID);
      FUSION_PASS_CHECK(biasNode->GetType() != "Const",
              OP_LOGW(FUSED_OP_TYPE.c_str(), "Conv2DMulFusionPass: bias is not const Node."),
              return NOT_CHANGED);
  }

  // get inputChannel
  int64_t inputChannel = 0;
  FUSION_PASS_CHECK(SUCCESS != GetInputChannel(mul_node, inputChannel),
           OP_LOGW(FUSED_OP_TYPE.c_str(), "Conv2DMulAddFusionPass: Get Mul's fm input_channel failed."),
           return NOT_CHANGED);

  // get Mul's const input, should be scalar
  vector<ge::ConstGeTensorPtr> mulweights =
            ge::OpDescUtils::GetWeights(mul_node);
  ge::ConstGeTensorPtr mul_weight = mulweights[0];
  FUSION_PASS_CHECK(mul_weight == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(),
          "Mul node's weight is null, fusion failed."),
            return PARAM_INVALID);
  FUSION_PASS_CHECK(mul_weight->GetTensorDesc().GetShape().GetDims().size() != 0 &&
            !(mul_weight->GetTensorDesc().GetShape().GetDims().size() == 1 &&
            (mul_weight->GetTensorDesc().GetShape().GetDims()[0] == 1 ||
              mul_weight->GetTensorDesc().GetShape().GetDims()[0] == inputChannel)),
            OP_LOGW(FUSED_OP_TYPE.c_str(), "Conv2DMulFusionPass: mul's weight should be scalar input or channel_wise input."),
            return NOT_CHANGED);

  size_t nonConstMulInputIndex = 0;
  size_t constMulInputIndex = 0;
  FUSION_PASS_CHECK(true != ge::OpDescUtils::GetNonConstInputIndex(mul_node,
             0, nonConstMulInputIndex),
           OP_LOGW(FUSED_OP_TYPE.c_str(), "get mul's non-const input failed"),
           return NOT_CHANGED);
  size_t maxMulInputIndex = mul_op->GetAllInputName().size() - 1;
  constMulInputIndex = maxMulInputIndex - nonConstMulInputIndex;

  // copy filter mul op, update input_shape & out_shape
  OpDescPtr filterMulOpDesc = AttrUtils::CloneOpDesc(mul_op);
  FUSION_PASS_CHECK(filterMulOpDesc == nullptr,
           OP_LOGI(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, clone filter mul desc failed.",
           mul_node->GetName().c_str()),
           return PARAM_INVALID);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "filterMulOpDesc %s, optye %s, input %ld, output %ld",
          filterMulOpDesc->GetName().c_str(),
          filterMulOpDesc->GetType().c_str(),
          filterMulOpDesc->GetAllInputsDesc().size(),
          filterMulOpDesc->GetAllOutputsDesc().size());

  GeTensorDesc convfilterconstTensor = filterNode->GetOpDesc()->GetOutputDesc(0);
  convfilterconstTensor.SetOriginShape(convfilterconstTensor.GetShape());
  convfilterconstTensor.SetOriginDataType(convfilterconstTensor.GetDataType());
  convfilterconstTensor.SetOriginFormat(convfilterconstTensor.GetOriginFormat());
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2D's filter_node input datatype is %d.", convfilterconstTensor.GetDataType());
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2D's filter_node origin datatype is %d.", convfilterconstTensor.GetOriginDataType());
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2D's filter_node input format is %d.", convfilterconstTensor.GetFormat());
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2D's filter_node input origin format is %d.", convfilterconstTensor.GetOriginFormat());
  GeTensorDesc convfilterTensor = src_op->GetInputDesc(FILTER_INDEX);
  convfilterTensor.SetOriginShape(convfilterTensor.GetShape());
  convfilterTensor.SetOriginDataType(convfilterTensor.GetDataType());
  convfilterTensor.SetOriginFormat(convfilterTensor.GetOriginFormat());
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2D's 2nd input datatype is %d.", convfilterTensor.GetDataType());
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2D's 2nd origin datatype is %d.", convfilterTensor.GetOriginDataType());
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2D's 2nd input format is %d.", convfilterTensor.GetFormat());
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2D's 2nd input origin format is %d.", convfilterTensor.GetOriginFormat());
  filterMulOpDesc->SetName(mul_node->GetName() + "_filter");
  filterMulOpDesc->SetType(MUL);
  filterMulOpDesc->UpdateInputName(mul_op->GetAllInputName());
  filterMulOpDesc->UpdateOutputName(mul_op->GetAllOutputName());
  filterMulOpDesc->UpdateInputDesc(nonConstMulInputIndex, convfilterconstTensor);
  filterMulOpDesc->UpdateOutputDesc(0, convfilterTensor);

  // connect conv2d to output
  auto xAnchor = mul_node->GetOutDataAnchor(0);
  InDataAnchorPtr outputInAnchor = xAnchor->GetPeerInDataAnchors().at(0);
  xAnchor->UnlinkAll();
  OutDataAnchorPtr yAnchor = convNode->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(yAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "conv2d output anchor is null"),
            return FAILED);
  FUSION_PASS_CHECK(outputInAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "output input anchor is null"),
            return FAILED);
  FUSION_PASS_CHECK(GraphUtils::AddEdge(yAnchor, outputInAnchor) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from conv2d to output failed"),
            return FAILED);

  // unlink filter to conv2d
  InDataAnchorPtr convFilterInAnchor = convNode->GetInDataAnchor(FILTER_INDEX);
  FUSION_PASS_CHECK(convFilterInAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "convNode filter anchor is null"),
            return FAILED);
  FUSION_PASS_CHECK(GraphUtils::RemoveEdge(filterAnchor, convFilterInAnchor) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "remove edge from input to conv2d failed"),
            return FAILED);

  // unlink bias to conv2d
  if (hasBias) {
    convBiasInAnchor = convNode->GetInDataAnchor(BIAS_INDEX);
    FUSION_PASS_CHECK(convBiasInAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "convNode bias anchor is null"),
            return FAILED);
    FUSION_PASS_CHECK(GraphUtils::RemoveEdge(biasAnchor, convBiasInAnchor) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "remove edge from input to conv2d failed"),
            return FAILED);
  }

  // get mulConstNode
  NodePtr mulConstNode = mulConstNodes[0];
  OutDataAnchorPtr constMulAnchor = mulConstNode->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(constMulAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "Node [%s]: const output anchor is null",
                    mulConstNode->GetName().c_str()),
            return FAILED);

  // add edge for filter->filterMul
  ge::NodePtr filterMulOpNode = graph.AddNode(filterMulOpDesc);
  newNodes.push_back(filterMulOpNode);
  InDataAnchorPtr filterMul0Anchor = filterMulOpNode->GetInDataAnchor(nonConstMulInputIndex);
  FUSION_PASS_CHECK(filterMul0Anchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "filter_mul input_0 anchor is null"),
            return FAILED);
  FUSION_PASS_CHECK(GraphUtils::AddEdge(filterAnchor, filterMul0Anchor) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from filter to filter_mul failed"),
            return FAILED);
  InDataAnchorPtr filterMul1Anchor = filterMulOpNode->GetInDataAnchor(constMulInputIndex);
  FUSION_PASS_CHECK(filterMul1Anchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "filter_mul input_1 anchor is null"),
            return FAILED);
  FUSION_PASS_CHECK(GraphUtils::AddEdge(constMulAnchor, filterMul1Anchor) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from filter to filter_mul failed"),
            return FAILED);
  OutDataAnchorPtr filterMulOutAnchor = filterMulOpNode->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(GraphUtils::AddEdge(filterMulOutAnchor, convFilterInAnchor) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from filter_mul to conv failed"),
            return FAILED);

  if (hasBias) {
    // copy bias mul op, update input_shape & out_shape
    OpDescPtr biasMulOpDesc = AttrUtils::CloneOpDesc(mul_op);
    FUSION_PASS_CHECK(biasMulOpDesc == nullptr,
            OP_LOGI(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, clone bias mul desc failed.",
            mul_node->GetName().c_str()),
            return PARAM_INVALID);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "biasMulOpDesc %s, optye %s, input %ld, output %ld",
            biasMulOpDesc->GetName().c_str(),
            biasMulOpDesc->GetType().c_str(),
            biasMulOpDesc->GetAllInputsDesc().size(),
            biasMulOpDesc->GetAllOutputsDesc().size());
    GeTensorDesc convbiasconstTensor = biasNode->GetOpDesc()->GetOutputDesc(0);
    convbiasconstTensor.SetOriginShape(convbiasconstTensor.GetShape());
    convbiasconstTensor.SetOriginDataType(convbiasconstTensor.GetDataType());
    convbiasconstTensor.SetOriginFormat(convbiasconstTensor.GetOriginFormat());
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2D's bias_node input datatype is %d.", convbiasconstTensor.GetDataType());
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2D's bias_node origin datatype is %d.", convbiasconstTensor.GetOriginDataType());
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2D's bias_node input format is %d.", convbiasconstTensor.GetFormat());
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2D's bias_node input origin format is %d.", convbiasconstTensor.GetOriginFormat());
    GeTensorDesc convbiasTensor = src_op->GetInputDesc(BIAS_INDEX);
    convbiasTensor.SetOriginShape(convbiasTensor.GetShape());
    convbiasTensor.SetOriginDataType(convbiasTensor.GetDataType());
    convbiasTensor.SetOriginFormat(convbiasTensor.GetOriginFormat());
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2D's 3nd input datatype is %d.", convbiasTensor.GetDataType());
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2D's 3nd origin datatype is %d.", convbiasTensor.GetOriginDataType());
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2D's 3nd input format is %d.", convbiasTensor.GetFormat());
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2D's 3nd input origin format is %d.", convbiasTensor.GetOriginFormat());
    biasMulOpDesc->SetName(mul_node->GetName() + "_bias");
    biasMulOpDesc->SetType(MUL);
    biasMulOpDesc->UpdateInputName(mul_op->GetAllInputName());
    biasMulOpDesc->UpdateOutputName(mul_op->GetAllOutputName());
    biasMulOpDesc->UpdateInputDesc(nonConstMulInputIndex, convbiasconstTensor);
    biasMulOpDesc->UpdateOutputDesc(0, convbiasTensor);

    // add edge for bias->biasMul
    ge::NodePtr biasMulOpNode = graph.AddNode(biasMulOpDesc);
    newNodes.push_back(biasMulOpNode);
    InDataAnchorPtr biasMul0Anchor = biasMulOpNode->GetInDataAnchor(nonConstMulInputIndex);
    FUSION_PASS_CHECK(biasMul0Anchor == nullptr,
              OP_LOGE(FUSED_OP_TYPE.c_str(), "bias_mul input_0 anchor is null"),
              return FAILED);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(biasAnchor, biasMul0Anchor) != GRAPH_SUCCESS,
              OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from bias to bias_mul failed"),
              return FAILED);
    InDataAnchorPtr biasMul1Anchor = biasMulOpNode->GetInDataAnchor(constMulInputIndex);
    FUSION_PASS_CHECK(biasMul1Anchor == nullptr,
              OP_LOGE(FUSED_OP_TYPE.c_str(), "bias_mul input_1 anchor is null"),
              return FAILED);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(constMulAnchor, biasMul1Anchor) != GRAPH_SUCCESS,
              OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from bias to bias_mul failed"),
              return FAILED);
    OutDataAnchorPtr biasMulOutAnchor = biasMulOpNode->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(biasMulOutAnchor, convBiasInAnchor) != GRAPH_SUCCESS,
              OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from bias_mul to conv failed"),
              return FAILED);
  }

  FUSION_PASS_CHECK(graph.RemoveNode(mul_node) == ge::GRAPH_FAILED,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "remove node %s failed.", mul_node->GetName().c_str()),
            return FAILED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Conv2DMulFusionPass fusion success.");
  return SUCCESS;
}
REGISTER_PASS("AConv2dMulFusion", BUILT_IN_GRAPH_PASS, Conv2DMulFusionPass);
}