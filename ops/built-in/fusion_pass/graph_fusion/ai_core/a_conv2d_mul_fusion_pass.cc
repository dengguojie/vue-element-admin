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
 * \file a_conv2d_mul_fusion_pass.cpp
 * \brief  conv-mul fusion pass(conv2d-mul --> conv)
 */
#include "a_conv2d_mul_fusion_pass.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

#include "pattern_fusion_util.h"
#include "op_log.h"
#include "../../../op_proto/util/error_util.h"
#include "common/util/error_manager/error_manager.h"

using namespace ge;
namespace fe {
static const char kPatternSrc[] = "src";
static const char kPatternMul[] = "mul";
static const char kConvolution[] = "Conv2D";
static const char kConvolution3D[] = "Conv3D";
static const char kMul[] = "Mul";
static const int kFilterIndex = 1;
static const int kBiasIndex = 2;

/*!
  * @brief Define conv+mul pattern.
  * The graph struct need to adapt is shown as follows:
  *
  *               const  conv2d
  *                    \  |
  *                      mul
  *                       |
  *                     output
  *
  *  Notice: the struct can be captured by
  *          conv2d + mul pattern
  * @return vector<FusionPattern*> All valid patterns.
  */

vector<FusionPattern*> Conv2DMulFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  string pass_name = "TbeConv2DMulFusion";
  FusionPattern* pattern = new (std::nothrow) FusionPattern(pass_name);
  FUSION_PASS_CHECK(pattern == nullptr, CommonRuntimeErrLog(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pass_name.c_str());
  pattern->AddOpDesc(kPatternMul, {kMul})
      .AddOpDesc(kPatternSrc, {kConvolution, kConvolution3D})
      .SetInputs(kPatternMul, {kPatternSrc})
      .SetOutput(kPatternMul);
  patterns.push_back(pattern);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pass_name.c_str());

  return patterns;
}

/*!
  * @brief get input node channel num
  * @param input_node The pointer of input node
  * @param input_channel The cahnnel num of input node
  * @return success for get input channel
  */
Status Conv2DMulFusionPass::GetInputChannel(const ge::NodePtr input_node, int64_t& input_channel) {
  ge::GeTensorDesc node_fm = ge::OpDescUtils::GetNonConstInputTensorDesc(input_node, 0);
  ge::GeShape node_fm_shape = node_fm.GetShape();
  ge::Format node_fm_format = node_fm.GetFormat();
  auto dims = node_fm_shape.GetDims();
  FUSION_PASS_CHECK(dims.size() != 4 ||
                    (node_fm_format != FORMAT_NCHW && node_fm_format != FORMAT_NHWC),
                     CommonRuntimeErrLog(fused_op_type_.c_str(), ConcatString("Conv2DMulFusionPass",
                          " Mul node's fm shape_dim is ", dims.size(), " format is ", node_fm_format,
                          " fusion failed, valid fm shape_dim is 4, and valid format is NCHW(0) or NHWC(1).")),
                    return FAILED);
  if (node_fm_format == FORMAT_NCHW) {
    input_channel = dims[1];
  } else if (node_fm_format == FORMAT_NHWC) {
    input_channel = dims[3];
  }

  return SUCCESS;
}

/*
 * @brief: parse nodes matched in mapping and call graph DoFusion
 * @param [in] graph: original graph
 * @param [in] mapping: matched pattern
 * @param [out] new_nodes: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status Conv2DMulFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& new_nodes) {
  OP_LOGD(fused_op_type_.c_str(), "Enter Conv2DMulFusionPass");
  ge::NodePtr conv_node = GetNodeFromMapping(kPatternSrc, mapping);
  ge::NodePtr mul_node = GetNodeFromMapping(kPatternMul, mapping);
  FUSION_PASS_CHECK(conv_node == nullptr, CommonRuntimeErrLog(fused_op_type_.c_str(), "Node conv2d is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(mul_node == nullptr, CommonRuntimeErrLog(fused_op_type_.c_str(), "Node mul is null, fusion failed."),
                    return PARAM_INVALID);

  if (conv_node->GetOutDataNodes().size() > 1) {
    OP_LOGI(fused_op_type_.c_str(), "Conv_Node:[%s]: Conv outdata size is invalid.", conv_node->GetName().c_str());
    return NOT_CHANGED;
  }

  if (mul_node->GetInDataNodes().size() != 2) {
    OP_LOGI(fused_op_type_.c_str(), "Mul_Node:[%s]: Conv_Mul graph fusion only support 2 input for Mul Node.",
            mul_node->GetName().c_str());
    return NOT_CHANGED;
  }

  if ((conv_node->GetType() == "Conv3D") && (mul_node->GetOutDataNodes().size() > 1)) {
    OP_LOGI(fused_op_type_.c_str(), "Conv2DMulFusionPass: mul node [%s] has %u output node,out data size is invalid.",
            mul_node->GetName().c_str(), mul_node->GetOutDataNodes().size());
    return NOT_CHANGED;
  }

  OutDataAnchorPtr bias_anchor = nullptr;
  InDataAnchorPtr conv_bias_in_anchor = nullptr;
  NodePtr bias_node = nullptr;

  vector<ge::NodePtr> mul_const_nodes = ge::OpDescUtils::GetConstInputs(mul_node);
  auto mul_const_node_size = mul_const_nodes.size();
  FUSION_PASS_CHECK((mul_const_nodes.empty() || mul_const_node_size != 1),
                    OP_LOGI(fused_op_type_.c_str(), "Mul_Node:[%s]'s const_node's size %u is invalid.",
                            mul_node->GetName().c_str(), mul_const_node_size),
                    return NOT_CHANGED);

  ge::OpDescPtr src_op = conv_node->GetOpDesc();
  FUSION_PASS_CHECK(src_op == nullptr,
                    CommonRuntimeErrLog(fused_op_type_.c_str(), ConcatString("Node: ",conv_node->GetName().c_str(),
                                                           "'s OpDesc is null, fusion failed.")),
                    return PARAM_INVALID);
  ge::OpDescPtr mul_op = mul_node->GetOpDesc();
  FUSION_PASS_CHECK(mul_op == nullptr,
                    CommonRuntimeErrLog(fused_op_type_.c_str(), ConcatString("Node: ",conv_node->GetName().c_str(),
                                                           "'s OpDesc is null, fusion failed.")),
                    return PARAM_INVALID);

  OP_LOGI(fused_op_type_.c_str(), "Conv2DMulFusionPass: conv2d [%s] has %u input anchor.", conv_node->GetName().c_str(),
          conv_node->GetAllInDataAnchors().size());
  OP_LOGI(fused_op_type_.c_str(), "Conv2DMulFusionPass: conv2d [%s] has %u input desc.", conv_node->GetName().c_str(),
          src_op->GetAllInputsDesc().size());
  int32_t in_edges_size = conv_node->GetInDataNodes().size();
  if (in_edges_size < 0) {
    OP_LOGI(fused_op_type_.c_str(), "inEdges size is invalid.");
    return NOT_CHANGED;
  }

  // get conv2d's weights & bias
  vector<ge::ConstGeTensorPtr> conv2d_weights = ge::OpDescUtils::GetWeights(conv_node);
  bool hasBias = true;
  if (conv2d_weights.size() < kBiasIndex) {
    hasBias = false;
  }

  int anchor_num = 0;
  for (auto in_data_anchor : conv_node->GetAllInDataAnchors()) {
    if ((in_data_anchor != nullptr) &&
        (in_data_anchor->GetPeerOutAnchor() != nullptr) &&
        (in_data_anchor->GetPeerOutAnchor()->GetOwnerNode() != nullptr)) {
      anchor_num++;
    }
  }

  if (anchor_num == 3) {
    hasBias = true;
  }

  OP_LOGI(fused_op_type_.c_str(), "convWeights  size %u .", conv2d_weights.size());

  auto conv_filter_anchor = conv_node->GetInDataAnchor(kFilterIndex);
  FUSION_PASS_CHECK(conv_filter_anchor == nullptr, CommonRuntimeErrLog(fused_op_type_.c_str(), "filter in anchor is null"),
                    return PARAM_INVALID);
  OutDataAnchorPtr filter_anchor = conv_filter_anchor->GetPeerOutAnchor();
  FUSION_PASS_CHECK(filter_anchor == nullptr, CommonRuntimeErrLog(fused_op_type_.c_str(), "filter output anchor is null"),
                    return PARAM_INVALID);
  NodePtr filter_node = filter_anchor->GetOwnerNode();
  FUSION_PASS_CHECK(filter_node == nullptr,
                    CommonRuntimeErrLog(fused_op_type_.c_str(), "Conv2DMulFusionPass: filter_node is not exist."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(filter_node) != "Const",
                    OP_LOGW(fused_op_type_.c_str(), "Conv2DMulFusionPass: filter is not const Node."),
                    return NOT_CHANGED);
  if (hasBias) {
    auto conv_bias_anchor = conv_node->GetInDataAnchor(kBiasIndex);
    FUSION_PASS_CHECK(conv_bias_anchor == nullptr, CommonRuntimeErrLog(fused_op_type_.c_str(), "bias in anchor is null"),
                      return PARAM_INVALID);
    bias_anchor = conv_bias_anchor->GetPeerOutAnchor();
    FUSION_PASS_CHECK(bias_anchor == nullptr, CommonRuntimeErrLog(fused_op_type_.c_str(), "bias anchor is null"),
                      return PARAM_INVALID);
    bias_node = bias_anchor->GetOwnerNode();
    FUSION_PASS_CHECK(bias_node == nullptr,
                      CommonRuntimeErrLog(fused_op_type_.c_str(), "Conv2DMulFusionPass: bias_node is not exist."),
                      return PARAM_INVALID);
    FUSION_PASS_CHECK(ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(bias_node) != "Const",
                      OP_LOGW(fused_op_type_.c_str(), "Conv2DMulFusionPass: bias is not const Node."),
                      return NOT_CHANGED);
  }

  if (conv_node->GetType() == "Conv2D") {
    // get input_channel
    int64_t input_channel = 0;
    FUSION_PASS_CHECK(SUCCESS != GetInputChannel(mul_node, input_channel),
                      OP_LOGW(fused_op_type_.c_str(), "Conv2DMulAddFusionPass: Get Mul's fm input_channel failed."),
                      return NOT_CHANGED);

    // get Mul's const input, should be scalar or channel_wise
    vector<ge::ConstGeTensorPtr> mul_weights = ge::OpDescUtils::GetWeights(mul_node);
    FUSION_PASS_CHECK(mul_weights.empty(), OP_LOGW(fused_op_type_.c_str(), "Mul's weight is empty."),
                      return NOT_CHANGED);
    ge::ConstGeTensorPtr mul_weight = mul_weights[0];
    FUSION_PASS_CHECK(mul_weight == nullptr,
                      CommonRuntimeErrLog(fused_op_type_.c_str(), "Mul node's weight is null, fusion failed."),
                      return PARAM_INVALID);
    FUSION_PASS_CHECK(mul_weight->GetTensorDesc().GetShape().GetDims().size() != 0 &&
                          !(mul_weight->GetTensorDesc().GetShape().GetDims().size() == 1 &&
                            (mul_weight->GetTensorDesc().GetShape().GetDims()[0] == 1 ||
                             mul_weight->GetTensorDesc().GetShape().GetDims()[0] == input_channel)),
                      OP_LOGW(fused_op_type_.c_str(),
                              "Conv2DMulFusionPass: mul's weight should be scalar input or channel_wise input."),
                      return NOT_CHANGED);
    if (mul_weight->GetTensorDesc().GetShape().GetDims().size() != 0 &&
        PatternFusionUtil::IsUnknownShape(mul_weight->GetTensorDesc().GetShape().GetDims()[0])) {
      OP_LOGW(fused_op_type_.c_str(), "Conv2DMulFusionPass cannot be applied for unknown shape.");
      return NOT_CHANGED;
    }
  } else if (conv_node->GetType() == "Conv3D") {
    GeTensorDesc input_desc0 = mul_op->GetInputDesc(0);
    GeTensorDesc input_desc1 = mul_op->GetInputDesc(1);
    auto dims1 = input_desc1.GetShape().GetDims();
    auto dims0 = input_desc0.GetShape().GetDims();
    FUSION_PASS_CHECK(dims0.size() != 5,
                      OP_LOGW(fused_op_type_.c_str(), "Mul's input0 dims size is not 5."),
                      return NOT_CHANGED);
    if (dims1.size() == 1) {
      if (PatternFusionUtil::IsUnknownShape(dims0.at(4)) ||
          PatternFusionUtil::IsUnknownShape(dims1.at(0))) {
        OP_LOGW(fused_op_type_.c_str(), "Conv2DMulFusionPass cannot be applied for unknown shape.");
        return NOT_CHANGED;
      }
      if (dims0.at(4) != dims1.at(0)) {
        return NOT_CHANGED;
      }
    } else if (dims1.size() == 5) {
      // NDHWC
      if (PatternFusionUtil::IsUnknownShape(dims0.at(4)) ||
          PatternFusionUtil::IsUnknownShape(dims1.at(0)) ||
          PatternFusionUtil::IsUnknownShape(dims1.at(1)) ||
          PatternFusionUtil::IsUnknownShape(dims1.at(2)) ||
          PatternFusionUtil::IsUnknownShape(dims1.at(3)) ||
          PatternFusionUtil::IsUnknownShape(dims1.at(4))) {
        OP_LOGW(fused_op_type_.c_str(), "Conv2DMulFusionPass cannot be applied for unknown shape.");
        return NOT_CHANGED;
      }
      if (dims1.at(0) != 1 or dims1.at(1) != 1 or dims1.at(2) != 1 or dims1.at(3) != 1 or dims0.at(4) != dims1.at(4)) {
        OP_LOGW(fused_op_type_.c_str(), "Match failed! tensor is not ChannelWise");
        return NOT_CHANGED;
      }
    } else {
      OP_LOGW(fused_op_type_.c_str(), "fusion failed, Mul's input1 dims size is %d.", dims1.size());
      return NOT_CHANGED;
    }
  }

  size_t non_const_mul_input_index = 0;
  size_t const_mul_input_index = 0;
  FUSION_PASS_CHECK(true != ge::OpDescUtils::GetNonConstInputIndex(mul_node, 0, non_const_mul_input_index),
                    OP_LOGW(fused_op_type_.c_str(), "get mul's non-const input failed."), return FAILED);
  FUSION_PASS_CHECK((non_const_mul_input_index != 0 && non_const_mul_input_index != 1),
                    OP_LOGW(fused_op_type_.c_str(), "mul's non-const input index is invalid."), return FAILED);
  const_mul_input_index = 1 - non_const_mul_input_index;

  // get mul_const_node
  NodePtr mul_const_node = mul_node->GetInDataNodes().at(const_mul_input_index);
  FUSION_PASS_CHECK(mul_const_node == nullptr, CommonRuntimeErrLog(fused_op_type_.c_str(), "mul's const node is null"),
                    return FAILED);
  OutDataAnchorPtr const_mul_anchor = mul_const_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(const_mul_anchor == nullptr,
                    CommonRuntimeErrLog(fused_op_type_.c_str(), ConcatString("Node [", mul_const_node->GetName().c_str(),
                                                                 "]: const output anchor is null")),
                    return FAILED);

  // copy filter mul op, update input_shape & out_shape
  OpDescPtr filter_mul_op_desc = AttrUtils::CloneOpDesc(mul_op);
  FUSION_PASS_CHECK(filter_mul_op_desc == nullptr,
                    OP_LOGI(fused_op_type_.c_str(), "Node:%s's OpDesc is null, clone filter mul desc failed.",
                            mul_node->GetName().c_str()),
                    return PARAM_INVALID);

  // update filter-mul non-const input Desc and output Desc
  auto conv_filter_op_desc = filter_node->GetOpDesc();
  FUSION_PASS_CHECK(conv_filter_op_desc == nullptr,
                    CUBE_INNER_ERR_REPORT(fused_op_type_.c_str(), "filter's OpDesc is null"), return FAILED);
  GeTensorDesc conv_filter_const_tensor = conv_filter_op_desc->GetOutputDesc(0);
  conv_filter_const_tensor.SetOriginShape(conv_filter_const_tensor.GetShape());
  conv_filter_const_tensor.SetOriginDataType(conv_filter_const_tensor.GetDataType());
  conv_filter_const_tensor.SetOriginFormat(conv_filter_const_tensor.GetOriginFormat());
  GeTensorDesc conv_filter_tensor = src_op->GetInputDesc(kFilterIndex);
  conv_filter_tensor.SetOriginShape(conv_filter_tensor.GetShape());
  conv_filter_tensor.SetOriginDataType(conv_filter_tensor.GetDataType());
  conv_filter_tensor.SetOriginFormat(conv_filter_tensor.GetOriginFormat());
  filter_mul_op_desc->SetName(mul_node->GetName() + "_filter");
  filter_mul_op_desc->SetType(kMul);
  filter_mul_op_desc->UpdateInputName(mul_op->GetAllInputName());
  filter_mul_op_desc->UpdateOutputName(mul_op->GetAllOutputName());
  filter_mul_op_desc->UpdateInputDesc(non_const_mul_input_index, conv_filter_const_tensor);
  filter_mul_op_desc->UpdateOutputDesc(0, conv_filter_tensor);
  if (filter_mul_op_desc->HasAttr(ge::ATTR_NAME_BATCH_LABEL)) {
    filter_mul_op_desc->DelAttr(ge::ATTR_NAME_BATCH_LABEL);
  }

  // connect conv2d to output
  auto x_anchor = mul_node->GetOutDataAnchor(0);
  OutDataAnchor::Vistor<InDataAnchorPtr> output_in_anchors = x_anchor->GetPeerInDataAnchors();
  x_anchor->UnlinkAll();
  for (auto &in_anchor : output_in_anchors) {
    OutDataAnchorPtr y_anchor = conv_node->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(y_anchor == nullptr, CommonRuntimeErrLog(fused_op_type_.c_str(), "conv2d output anchor is null"),
                      return FAILED);
    FUSION_PASS_CHECK(in_anchor == nullptr, CommonRuntimeErrLog(fused_op_type_.c_str(), "output input anchor is null"),
                      return FAILED);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(y_anchor, in_anchor) != GRAPH_SUCCESS,
                      CommonRuntimeErrLog(fused_op_type_.c_str(), "add edge from conv2d to output failed"), return FAILED);
  }

  // unlink filter to conv2d
  InDataAnchorPtr conv_filter_in_anchor = conv_node->GetInDataAnchor(kFilterIndex);
  FUSION_PASS_CHECK(conv_filter_in_anchor == nullptr,
                    CommonRuntimeErrLog(fused_op_type_.c_str(), "conv_node filter anchor is null"), return FAILED);
  filter_anchor = conv_filter_in_anchor->GetPeerOutAnchor();
  FUSION_PASS_CHECK(filter_anchor == nullptr, CommonRuntimeErrLog(fused_op_type_.c_str(), "filter output anchor is null"),
                    return FAILED);
  FUSION_PASS_CHECK(GraphUtils::RemoveEdge(filter_anchor, conv_filter_in_anchor) != GRAPH_SUCCESS,
                    CommonRuntimeErrLog(fused_op_type_.c_str(), "remove edge from input to conv2d failed"), return FAILED);

  // unlink bias to conv2d
  if (hasBias) {
    conv_bias_in_anchor = conv_node->GetInDataAnchor(kBiasIndex);
    FUSION_PASS_CHECK(conv_bias_in_anchor == nullptr, CommonRuntimeErrLog(fused_op_type_.c_str(), "conv_node bias anchor is null"),
                      return FAILED);
    bias_anchor = conv_bias_in_anchor->GetPeerOutAnchor();
    FUSION_PASS_CHECK(bias_anchor == nullptr, CommonRuntimeErrLog(fused_op_type_.c_str(), "bias anchor is null"), return FAILED);
    FUSION_PASS_CHECK(GraphUtils::RemoveEdge(bias_anchor, conv_bias_in_anchor) != GRAPH_SUCCESS,
                      CommonRuntimeErrLog(fused_op_type_.c_str(), "remove edge from input to conv2d failed"), return FAILED);
  }

  // add edge for filter->filterMul
  ge::NodePtr filter_mul_op_node = graph.AddNode(filter_mul_op_desc);
  FUSION_PASS_CHECK(filter_mul_op_node == nullptr, CommonRuntimeErrLog(fused_op_type_.c_str(), "add filter mul op node failed"),
                    return FAILED);
  new_nodes.push_back(filter_mul_op_node);
  InDataAnchorPtr filter_mul_non_const_anchor = filter_mul_op_node->GetInDataAnchor(non_const_mul_input_index);
  FUSION_PASS_CHECK(filter_mul_non_const_anchor == nullptr,
                    CommonRuntimeErrLog(fused_op_type_.c_str(), "filter_mul input_0 anchor is null"), return FAILED);
  FUSION_PASS_CHECK(GraphUtils::AddEdge(filter_anchor, filter_mul_non_const_anchor) != GRAPH_SUCCESS,
                    CommonRuntimeErrLog(fused_op_type_.c_str(), "add edge from filter to filter_mul failed"), return FAILED);
  InDataAnchorPtr filter_mul_const_anchor = filter_mul_op_node->GetInDataAnchor(const_mul_input_index);
  FUSION_PASS_CHECK(filter_mul_const_anchor == nullptr,
                    CommonRuntimeErrLog(fused_op_type_.c_str(), "filter_mul input_1 anchor is null"), return FAILED);
  FUSION_PASS_CHECK(GraphUtils::AddEdge(const_mul_anchor, filter_mul_const_anchor) != GRAPH_SUCCESS,
                    CommonRuntimeErrLog(fused_op_type_.c_str(), "add edge from filter to filter_mul failed"), return FAILED);
  OutDataAnchorPtr filter_mul_out_anchor = filter_mul_op_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(GraphUtils::AddEdge(filter_mul_out_anchor, conv_filter_in_anchor) != GRAPH_SUCCESS,
                    CommonRuntimeErrLog(fused_op_type_.c_str(), "add edge from filter_mul to conv failed"), return FAILED);

  if (hasBias) {
    // copy bias mul op, update input_shape & out_shape
    OpDescPtr bias_mul_op_desc = AttrUtils::CloneOpDesc(mul_op);
    FUSION_PASS_CHECK(bias_mul_op_desc == nullptr,
                      OP_LOGI(fused_op_type_.c_str(), "Node:%s's OpDesc is null, clone bias mul desc failed.",
                              mul_node->GetName().c_str()),
                      return PARAM_INVALID);

    auto conv_bias_op_desc = bias_node->GetOpDesc();
    FUSION_PASS_CHECK(conv_bias_op_desc == nullptr,
                      CUBE_INNER_ERR_REPORT(fused_op_type_.c_str(), "bias's OpDesc is null"), return FAILED);
    GeTensorDesc conv_bias_const_tensor = conv_bias_op_desc->GetOutputDesc(0);
    conv_bias_const_tensor.SetOriginShape(conv_bias_const_tensor.GetShape());
    conv_bias_const_tensor.SetOriginDataType(conv_bias_const_tensor.GetDataType());
    conv_bias_const_tensor.SetOriginFormat(conv_bias_const_tensor.GetOriginFormat());
    GeTensorDesc conv_bias_tensor = src_op->GetInputDesc(kBiasIndex);
    conv_bias_tensor.SetOriginShape(conv_bias_tensor.GetShape());
    conv_bias_tensor.SetOriginDataType(conv_bias_tensor.GetDataType());
    conv_bias_tensor.SetOriginFormat(conv_bias_tensor.GetOriginFormat());
    bias_mul_op_desc->SetName(mul_node->GetName() + "_bias");
    bias_mul_op_desc->SetType(kMul);
    bias_mul_op_desc->UpdateInputName(mul_op->GetAllInputName());
    bias_mul_op_desc->UpdateOutputName(mul_op->GetAllOutputName());
    bias_mul_op_desc->UpdateInputDesc(non_const_mul_input_index, conv_bias_const_tensor);
    bias_mul_op_desc->UpdateOutputDesc(0, conv_bias_tensor);
    if (bias_mul_op_desc->HasAttr(ge::ATTR_NAME_BATCH_LABEL)) {
      bias_mul_op_desc->DelAttr(ge::ATTR_NAME_BATCH_LABEL);
    }

    // add edge for bias->biasMul
    ge::NodePtr bias_mul_op_node = graph.AddNode(bias_mul_op_desc);
    FUSION_PASS_CHECK(bias_mul_op_node == nullptr, CommonRuntimeErrLog(fused_op_type_.c_str(),
                                                   "add bias mul op node failed"),
                      return FAILED);
    new_nodes.push_back(bias_mul_op_node);
    InDataAnchorPtr bias_mul_non_const_anchor = bias_mul_op_node->GetInDataAnchor(non_const_mul_input_index);
    FUSION_PASS_CHECK(bias_mul_non_const_anchor == nullptr,
                      CommonRuntimeErrLog(fused_op_type_.c_str(), "bias_mul input_0 anchor is null"), return FAILED);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(bias_anchor, bias_mul_non_const_anchor) != GRAPH_SUCCESS,
                      CommonRuntimeErrLog(fused_op_type_.c_str(), "add edge from bias to bias_mul failed"), return FAILED);
    InDataAnchorPtr bias_mul_const_anchor = bias_mul_op_node->GetInDataAnchor(const_mul_input_index);
    FUSION_PASS_CHECK(bias_mul_const_anchor == nullptr,
                      CommonRuntimeErrLog(fused_op_type_.c_str(), "bias_mul input_1 anchor is null"), return FAILED);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(const_mul_anchor, bias_mul_const_anchor) != GRAPH_SUCCESS,
                      CommonRuntimeErrLog(fused_op_type_.c_str(), "add edge from bias to bias_mul failed"), return FAILED);
    OutDataAnchorPtr bias_mul_out_anchor = bias_mul_op_node->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(bias_mul_out_anchor, conv_bias_in_anchor) != GRAPH_SUCCESS,
                      CommonRuntimeErrLog(fused_op_type_.c_str(), "add edge from bias_mul to conv failed"), return FAILED);
  }
  OP_LOGI(fused_op_type_.c_str(), "Conv2DMulFusionPass: conv2d [%s] has %u input anchor.",
          conv_node->GetName().c_str(), conv_node->GetAllInDataAnchors().size());

  FUSION_PASS_CHECK(graph.RemoveNode(mul_node) == ge::GRAPH_FAILED,
                    CommonRuntimeErrLog(fused_op_type_.c_str(), ConcatString("remove node ",
                                                                 mul_node->GetName().c_str(), " failed.")),
                    return FAILED);

  OP_LOGD(fused_op_type_.c_str(), "Conv2DMulFusionPass fusion success.");
  return SUCCESS;
}
REGISTER_PASS("AConv2dMulFusion", BUILT_IN_GRAPH_PASS, Conv2DMulFusionPass);
}  // namespace fe
