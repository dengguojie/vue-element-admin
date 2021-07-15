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
 * \file conv2d_backprop_input_bias_add_fusion_pass.cc
 * \brief pseudo fusion pass of "dx + bias_add"
 */
#include "conv2d_backprop_input_bias_add_fusion_pass.h"

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "anchor_util.h"
#include "common/util/error_manager/error_manager.h"
#include "error_util.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

namespace fe {
static const string OP_CONV2DBACKPROPINPUTD = "Conv2DBackpropInputD";

static const string OP_BIASADD = "BiasAdd";
static const string OP_CONV2DTRANSPOSED = "Conv2DTransposeD";

static const string PATTERN_CUBENODE = "cube_node";
static const string PATTERN_BIASADD = "bias_add";

vector<FusionPattern*> Conv2DbpInputBiasAddFusionPass::DefinePatterns() {
  /*
  * ============== pattern ================
  *
  * --> Conv2DbpInputD --> BiasAdd -->
  *
  * =======================================
  */
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("Conv2DbpInputBiasAddFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new an object not success."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_CUBENODE, {OP_CONV2DBACKPROPINPUTD})
    .AddOpDesc(PATTERN_BIASADD, {OP_BIASADD})
    .SetInputs(PATTERN_BIASADD, {PATTERN_CUBENODE})
    .SetOutput(PATTERN_BIASADD);
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define pattern Conv2DbpInputBiasAddFusionPass success.");

  return patterns;
}

Status Conv2DbpInputBiasAddFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                              vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGD("Start to fuse Conv2DBackpropInputD and BiasAdd.");
  if (convert_dx_to_transpose(graph, mapping, fusion_nodes) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status Conv2DbpInputBiasAddFusionPass::convert_dx_to_transpose(ge::ComputeGraph &graph, Mapping &mapping,
                                                               vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD("Start to convert Conv2DBackpropInputD to Conv2DTransposeD.");
  ge::NodePtr conv_node = GetNodeFromMapping(PATTERN_CUBENODE, mapping);
  FUSION_PASS_CHECK(conv_node == nullptr,
                    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new conv_node not success."),
                    return PARAM_INVALID);
  ge::NodePtr bias_node = GetNodeFromMapping(PATTERN_BIASADD, mapping);
  FUSION_PASS_CHECK(bias_node == nullptr,
                    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new bias_node not success."),
                    return PARAM_INVALID);

  auto in_all_nodes = bias_node->GetInAllNodes();
  FUSION_PASS_CHECK(in_all_nodes.size() <= 1,
                    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "num of input nodes of bias must >= 2."),
                    return PARAM_INVALID);
  auto bias_const_node = in_all_nodes.at(1);
  FUSION_PASS_CHECK(bias_const_node == nullptr,
                    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new bias_const_node not success."),
                    return PARAM_INVALID);

  // build a new node named Conv2DTransposeD
  auto conv_op = conv_node->GetOpDesc();
  FUSION_PASS_CHECK(conv_op == nullptr,
                    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "conv_op is null."),
                    return FAILED);
  ge::OpDescPtr conv2d_transpose_d_op = nullptr;
  FUSION_PASS_MAKE_SHARED(
    conv2d_transpose_d_op = std::make_shared<ge::OpDesc>(conv_op->GetName(), "Conv2DTransposeD"),
    return FAILED);

  auto bias_const_op = bias_const_node->GetOpDesc();
  FUSION_PASS_CHECK(bias_const_op == nullptr,
                    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "bias_const_op is null."),
                    return FAILED);

  set_in_out_op_and_attr(conv_op, bias_const_op, conv2d_transpose_d_op);

  // add Conv2DTransposeD node and connect edges
  ge::NodePtr conv2d_transpose_d = graph.AddNode(conv2d_transpose_d_op);
  FUSION_PASS_CHECK(conv2d_transpose_d == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new Conv2DTransposeD not success."),
                    return PARAM_INVALID);
    
  fusion_nodes.push_back(conv2d_transpose_d);

  if (connect_edges(conv_node, bias_node, bias_const_node, conv2d_transpose_d) != SUCCESS) {
    return FAILED;
  }

  // delete Conv2DBackpropInputD & BiasAdd
  FUSION_PASS_CHECK(graph.RemoveNode(conv_node) == ge::GRAPH_FAILED,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Removing conv_node is failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(bias_node) == ge::GRAPH_FAILED,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Removing bias_node is failed."),
                    return FAILED);

  return SUCCESS;
}

void Conv2DbpInputBiasAddFusionPass::set_in_out_op_and_attr(ge::OpDescPtr &conv_op,
                                                            ge::OpDescPtr &bias_const_op,
                                                            ge::OpDescPtr &conv2d_transpose_d_op) {
  /*
  * Conv2DBackpropInputD: input0 (filter), input1 (out_backprop)
  *   attr_list: input_size, strides, pads, dilations, groups, data_format
  * Conv2DTransposeD: input0 (x), input1 (filter), input2 (bias), input3 (offset_w)
  *   attr_list: input_size, strides, pads, dilations, groups, data_format, output_padding, offset_x
  */
  FUSION_PASS_CHECK(conv_op->GetAllInputNames().size() < 2,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get InputDesc of conv2d_transposefailed."),
                    return);
  ge::GeTensorDesc conv2d_transpose_d_in_desc_0 = conv_op->GetInputDesc(1);
  ge::GeTensorDesc conv2d_transpose_d_in_desc_1 = conv_op->GetInputDesc(0);
  ge::GeTensorDesc conv2d_transpose_d_in_desc_2 = bias_const_op->GetOutputDesc(0);
  ge::GeTensorDesc conv2d_transpose_d_in_desc_3(ge::GeShape(), ge::FORMAT_RESERVED, ge::DT_UNDEFINED);

  conv2d_transpose_d_in_desc_0.SetDataType(conv_op->GetInputDesc(1).GetDataType());
  conv2d_transpose_d_in_desc_0.SetOriginDataType(conv_op->GetInputDesc(1).GetOriginDataType());
  conv2d_transpose_d_in_desc_0.SetFormat(conv_op->GetInputDesc(1).GetFormat());
  conv2d_transpose_d_in_desc_0.SetOriginFormat(conv_op->GetInputDesc(1).GetFormat());
  conv2d_transpose_d_in_desc_1.SetDataType(conv_op->GetInputDesc(0).GetDataType());
  conv2d_transpose_d_in_desc_1.SetOriginDataType(conv_op->GetInputDesc(0).GetOriginDataType());
  conv2d_transpose_d_in_desc_1.SetFormat(conv_op->GetInputDesc(0).GetFormat());
  conv2d_transpose_d_in_desc_1.SetOriginFormat(conv_op->GetInputDesc(0).GetFormat());
  conv2d_transpose_d_in_desc_2.SetDataType(bias_const_op->GetOutputDesc(0).GetDataType());
  conv2d_transpose_d_in_desc_2.SetOriginDataType(bias_const_op->GetOutputDesc(0).GetOriginDataType());
  conv2d_transpose_d_in_desc_2.SetFormat(bias_const_op->GetOutputDesc(0).GetFormat());
  conv2d_transpose_d_in_desc_2.SetOriginFormat(bias_const_op->GetOutputDesc(0).GetFormat());

  FUSION_PASS_CHECK(
    conv2d_transpose_d_op->AddInputDesc("x", conv2d_transpose_d_in_desc_0) != ge::GRAPH_SUCCESS,
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Adding input x for Conv2DTransposeD is failed."),
    return);
  FUSION_PASS_CHECK(
    conv2d_transpose_d_op->AddInputDesc("filter", conv2d_transpose_d_in_desc_1) != ge::GRAPH_SUCCESS,
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Adding input filter for Conv2DTransposeD is failed."),
    return);
  FUSION_PASS_CHECK(
    conv2d_transpose_d_op->AddInputDesc("bias", conv2d_transpose_d_in_desc_2) != ge::GRAPH_SUCCESS,
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Adding input bias for Conv2DTransposeD is failed."),
    return);
  FUSION_PASS_CHECK(
    conv2d_transpose_d_op->AddInputDesc("offset_w", conv2d_transpose_d_in_desc_3) != ge::GRAPH_SUCCESS,
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Adding input offset_w for Conv2DTransposeD is failed."),
    return);
  ge::GeTensorDesc conv2d_transpose_d_out_desc_0 = conv_op->GetOutputDesc(0);
  conv2d_transpose_d_out_desc_0.SetDataType(ge::DT_FLOAT16);
  conv2d_transpose_d_out_desc_0.SetOriginDataType(ge::DT_FLOAT16);
  conv2d_transpose_d_out_desc_0.SetShape(conv_op->GetOutputDesc(0).GetShape());
  conv2d_transpose_d_out_desc_0.SetOriginShape(conv_op->GetOutputDesc(0).GetShape());
  conv2d_transpose_d_out_desc_0.SetFormat(conv_op->GetOutputDesc(0).GetFormat());
  conv2d_transpose_d_out_desc_0.SetOriginFormat(conv_op->GetOutputDesc(0).GetFormat());

  FUSION_PASS_CHECK(
    conv2d_transpose_d_op->AddOutputDesc(conv2d_transpose_d_out_desc_0) != ge::GRAPH_SUCCESS,
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Adding output y for Conv2DTransposeD is failed."),
    return);
  std::vector<int64_t> input_size_index;
  if (ge::AttrUtils::GetListInt(conv_op, "input_size", input_size_index)) {
    ge::AttrUtils::SetListInt(conv2d_transpose_d_op, "input_size", input_size_index);
  }
  std::vector<int64_t> strides_index;
  if (ge::AttrUtils::GetListInt(conv_op, "strides", strides_index)) {
    ge::AttrUtils::SetListInt(conv2d_transpose_d_op, "strides", strides_index);
  }
  std::vector<int64_t> pads_index;
  if (ge::AttrUtils::GetListInt(conv_op, "pads", pads_index)) {
    ge::AttrUtils::SetListInt(conv2d_transpose_d_op, "pads", pads_index);
  }
  std::vector<int64_t> dilations_index;
  if (ge::AttrUtils::GetListInt(conv_op, "dilations", dilations_index)) {
    ge::AttrUtils::SetListInt(conv2d_transpose_d_op, "dilations", dilations_index);
  }
  int64_t groups_index;
  if (ge::AttrUtils::GetInt(conv_op, "groups", groups_index)) {
    ge::AttrUtils::SetInt(conv2d_transpose_d_op, "groups", groups_index);
  }
  string data_format_index;
  if (ge::AttrUtils::GetStr(conv_op, "data_format", data_format_index)) {
    ge::AttrUtils::SetStr(conv2d_transpose_d_op, "data_format", data_format_index);
  }
  // set default value
  ge::AttrUtils::SetListInt(conv2d_transpose_d_op, "output_padding", {0, 0, 0, 0});
  ge::AttrUtils::SetInt(conv2d_transpose_d_op, "offset_x", 0);
}

Status Conv2DbpInputBiasAddFusionPass::connect_edges(ge::NodePtr &conv_node,
                                                     ge::NodePtr &bias_node,
                                                     ge::NodePtr &bias_const_node,
                                                     ge::NodePtr &conv2d_transpose_d) {
  /*
  * ============= src edges ================
  *
  * pre_peer_output0 --> dx_input0 \
  *                                 dx \
  * pre_peer_output1 --> dx_input1 /    \
  *                                  biasadd -->
  *                               const /
  *
  * ============= dst edges ================
  *
  * pre_peer_output1 --> td_input0 \
  * pre_peer_output0 --> td_input1 \
  *                                td (biasadd) -->
  *            const --> td_input2 /
  *  offset_w (null) --> td_input3 /
  *
  * ========================================
  */
  FUSION_PASS_CHECK(
    ge::GraphUtils::AddEdge(GetPeerOutAnchorWithInDataAnchor(conv_node, 0),
                            conv2d_transpose_d->GetInDataAnchor(1)) != SUCCESS,
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Adding edge from fused node: %s's index[0] to fusion node: %s's index[1] is failed.",
            conv_node->GetName().c_str(), conv2d_transpose_d->GetName().c_str()),
    return FAILED);
  FUSION_PASS_CHECK(
    ge::GraphUtils::AddEdge(GetPeerOutAnchorWithInDataAnchor(conv_node, 1),
                            conv2d_transpose_d->GetInDataAnchor(0)) != SUCCESS,
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Adding edge from fused node: %s's index[1] to fusion node: %s's index[0] is failed.",
            conv_node->GetName().c_str(), conv2d_transpose_d->GetName().c_str()),
    return FAILED);
  FUSION_PASS_CHECK(
    ge::GraphUtils::AddEdge(bias_const_node->GetOutDataAnchor(0), conv2d_transpose_d->GetInDataAnchor(2)) != SUCCESS,
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Adding edge from fused node: %s's index[0] to fusion node: %s's index[2] is failed.",
            bias_const_node->GetName().c_str(), conv2d_transpose_d->GetName().c_str()),
    return FAILED);
  auto outdata_anchor = bias_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(outdata_anchor == nullptr,
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Getting data anchor 0 of output is failed."), return FAILED);

  for (auto in_anchor : outdata_anchor->GetPeerInDataAnchors()) {
    in_anchor->UnlinkAll();
    FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(conv2d_transpose_d->GetOutDataAnchor(0), in_anchor) != SUCCESS,
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Adding edge from fusion node: %s's index[0] to fused node: %s's indexes is failed.",
              conv2d_transpose_d->GetName().c_str(), bias_node->GetName().c_str()),
      return FAILED);
  }

  for (auto in_anchor : conv_node->GetAllInDataAnchors()) {
    if (in_anchor != nullptr) {
      in_anchor->UnlinkAll();
    }
  }

  for (auto in_anchor : bias_node->GetAllInDataAnchors()) {
    if (in_anchor != nullptr) {
      in_anchor->UnlinkAll();
    }
  }

  return SUCCESS;
}

REGISTER_PASS("Conv2DbpInputBiasAddFusionPass", BUILT_IN_GRAPH_PASS, Conv2DbpInputBiasAddFusionPass);
}  // namespace fe
