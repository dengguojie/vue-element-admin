/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file conv3d_backprop_input_bias_add_fusion_pass.cc
 * \brief pseudo fusion pass of "dx + bias_add"
 */
#include "conv3d_backprop_input_bias_add_fusion_pass.h"

#include <iostream>
#include <map>
#include <memory>
#include <vector>
#include <string>

#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "common/util/error_manager/error_manager.h"
#include "../../../op_proto/util/error_util.h"
#include "anchor_util.h"

namespace fe {
static const string OP_CONV3D_BACKPROP_INPUT_D = "Conv3DBackpropInputD";

static const string OP_BIASADD = "BiasAdd";
static const string OP_CONV3D_TRANSPOSE_D = "Conv3DTransposeD";

static const string PATTERN_CUBENODE = "conv3d_backprop_input";
static const string PATTERN_BIASADD = "bias_add";

vector<FusionPattern*> Conv3DbpInputBiasAddFusionPass::DefinePatterns() {
  /*
  * ============== pattern ================
  *
  * --> Conv3DbpInputD --> BiasAdd -->
  *
  * =======================================
  */
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("Conv3DbpInputBiassAddFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "create Conv3DbpInputBiassAddFusionPass pattern object fail."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_CUBENODE, {OP_CONV3D_BACKPROP_INPUT_D})
          .AddOpDesc(PATTERN_BIASADD, {OP_BIASADD})
          .SetInputs(PATTERN_BIASADD, {PATTERN_CUBENODE})
          .SetOutput(PATTERN_BIASADD);
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "define pattern Conv3DbpInputBiassAddFusionPass success.");
  return patterns;
}

Status Conv3DbpInputBiasAddFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                                              vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD("begin Conv3DbpInputBiassAddFusionPass.");
  ge::NodePtr conv_node = GetNodeFromMapping(PATTERN_CUBENODE, mapping);
  FUSION_PASS_CHECK(conv_node == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "new conv_node not success."),
                    return NOT_CHANGED);
  ge::NodePtr bias_node = GetNodeFromMapping(PATTERN_BIASADD, mapping);
  FUSION_PASS_CHECK(bias_node == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "new bias_node not success."),
                    return NOT_CHANGED);

  FUSION_PASS_CHECK(bias_node->GetInAllNodes().size() <= 1, OP_LOGW(bias_node, "bias input node num is incorrect."),
                    return NOT_CHANGED);
  ge::NodePtr bias_const_node = bias_node->GetInAllNodes().at(1);
  FUSION_PASS_CHECK(bias_const_node == nullptr, OP_LOGW(bias_node, "bias const node is null."), return NOT_CHANGED);

  // build a new node named Conv3DTransposeD
  auto conv_op = conv_node->GetOpDesc();
  FUSION_PASS_CHECK(conv_op == nullptr, OP_LOGW(conv_node, "get conv3d_transpose_d_op failed."),
                    return NOT_CHANGED);
  ge::OpDescPtr conv3d_transpose_d_op = nullptr;
  FUSION_PASS_MAKE_SHARED(conv3d_transpose_d_op = std::make_shared<ge::OpDesc>(conv_op->GetName(), "Conv3DTransposeD"),
                          return NOT_CHANGED);

  auto bias_const_op = bias_const_node->GetOpDesc();
  FUSION_PASS_CHECK(bias_const_op == nullptr, OP_LOGW(conv_node, "bias_const_op is null."),
                    return NOT_CHANGED);

  SetOpAttr(conv_op, bias_const_op, conv3d_transpose_d_op);

  // add Conv3DTransposeD node and connect edges
  ge::NodePtr conv3d_transpose_d = graph.AddNode(conv3d_transpose_d_op);
  FUSION_PASS_CHECK(conv3d_transpose_d == nullptr, CUBE_INNER_ERR_REPORT(conv_node, "failed to new Conv3DTransposeD."),
                    return PARAM_INVALID);

  fusion_nodes.push_back(conv3d_transpose_d);

  if (ConnectEdges(conv_node, bias_node, bias_const_node, conv3d_transpose_d) != SUCCESS) {
    return FAILED;
  }

  // delete Conv3DBackpropInputD & BiasAdd
  FUSION_PASS_CHECK(graph.RemoveNode(conv_node) == ge::GRAPH_FAILED,
                    CUBE_INNER_ERR_REPORT(conv_node, "failed to remove conv_node."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(bias_node) == ge::GRAPH_FAILED,
                    CUBE_INNER_ERR_REPORT(bias_node, "failed to remove bias_node."), return FAILED);

  OP_LOGI(conv_node, "succeed to execute fusion pass");
  return SUCCESS;
}

void Conv3DbpInputBiasAddFusionPass::SetOpAttr(const ge::OpDescPtr &conv_op,
                                               const ge::OpDescPtr &bias_const_op,
                                               ge::OpDescPtr &conv3d_transpose_d_op) const {
  /*
  * Conv3DBackpropInputD: input0 (filter), input1 (out_backprop)
  *   attr_list: input_size, strides, pads, dilations, groups, data_format
  * Conv3DTransposeD: input0 (x), input1 (filter), input2 (bias), input3 (offset_w)
  *   attr_list: input_size, strides, pads, dilations, groups, data_format, output_padding, offset_x
  */
  FUSION_PASS_CHECK(conv_op == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "conv_op is null."),
                    return);
  FUSION_PASS_CHECK(bias_const_op == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "bias_const_op is null."),
                    return);
  FUSION_PASS_CHECK(conv3d_transpose_d_op == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "conv3d_transpose_d_op is null."),
                    return);
  ge::GeTensorDesc conv3d_transpose_d_in_desc_0 = conv_op->GetInputDesc(1);
  ge::GeTensorDesc conv3d_transpose_d_in_desc_1 = conv_op->GetInputDesc(0);
  ge::GeTensorDesc conv3d_transpose_d_in_desc_2 = bias_const_op->GetOutputDesc(0);
  ge::GeTensorDesc conv3d_transpose_d_in_desc_3(ge::GeShape(), ge::FORMAT_RESERVED, ge::DT_UNDEFINED);

  FUSION_PASS_CHECK(conv3d_transpose_d_op->AddInputDesc("x", conv3d_transpose_d_in_desc_0) != ge::GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(conv_op, "Adding input x for Conv3DTransposeD is failed."), return );
  FUSION_PASS_CHECK(conv3d_transpose_d_op->AddInputDesc("filter", conv3d_transpose_d_in_desc_1) != ge::GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(conv_op, "Adding input filter for Conv3DTransposeD is failed."), return );
  FUSION_PASS_CHECK(conv3d_transpose_d_op->AddInputDesc("bias", conv3d_transpose_d_in_desc_2) != ge::GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(conv_op, "Adding input bias for Conv3DTransposeD is failed."), return );
  FUSION_PASS_CHECK(conv3d_transpose_d_op->AddInputDesc("offset_w", conv3d_transpose_d_in_desc_3) != ge::GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(conv_op, "Adding input offset_w for Conv3DTransposeD is failed."), return );
  ge::GeTensorDesc conv3d_transpose_d_out_desc_0 = conv_op->GetOutputDesc(0);
  conv3d_transpose_d_out_desc_0.SetDataType(ge::DT_FLOAT16);
  conv3d_transpose_d_out_desc_0.SetOriginDataType(ge::DT_FLOAT16);

  FUSION_PASS_CHECK(conv3d_transpose_d_op->AddOutputDesc("y", conv3d_transpose_d_out_desc_0) != ge::GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(conv_op, "Adding output y for Conv3DTransposeD is failed."), return );
  std::vector<int64_t> input_size_index;
  if (ge::AttrUtils::GetListInt(conv_op, "input_size", input_size_index)) {
    ge::AttrUtils::SetListInt(conv3d_transpose_d_op, "input_size", input_size_index);
  }
  std::vector<int64_t> strides_index;
  if (ge::AttrUtils::GetListInt(conv_op, "strides", strides_index)) {
    ge::AttrUtils::SetListInt(conv3d_transpose_d_op, "strides", strides_index);
  }
  std::vector<int64_t> pads_index;
  if (ge::AttrUtils::GetListInt(conv_op, "pads", pads_index)) {
    ge::AttrUtils::SetListInt(conv3d_transpose_d_op, "pads", pads_index);
  }
  std::vector<int64_t> dilations_index;
  if (ge::AttrUtils::GetListInt(conv_op, "dilations", dilations_index)) {
    ge::AttrUtils::SetListInt(conv3d_transpose_d_op, "dilations", dilations_index);
  }
  int64_t groups_index;
  if (ge::AttrUtils::GetInt(conv_op, "groups", groups_index)) {
    ge::AttrUtils::SetInt(conv3d_transpose_d_op, "groups", groups_index);
  }
  string data_format_index;
  if (ge::AttrUtils::GetStr(conv_op, "data_format", data_format_index)) {
    ge::AttrUtils::SetStr(conv3d_transpose_d_op, "data_format", data_format_index);
  }
  // set default value
  ge::AttrUtils::SetListInt(conv3d_transpose_d_op, "output_padding", {0, 0, 0, 0, 0});
  ge::AttrUtils::SetInt(conv3d_transpose_d_op, "offset_x", 0);
}

Status Conv3DbpInputBiasAddFusionPass::ConnectEdges(const ge::NodePtr &conv_node,
                                                    const ge::NodePtr &bias_node,
                                                    const ge::NodePtr &bias_const_node,
                                                    const ge::NodePtr &conv3d_transpose_d) const {
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
  ge::OutDataAnchorPtr outDataPtr0 = GetPeerOutAnchorWithInDataAnchor(conv_node, 0);
  FUSION_PASS_CHECK(outDataPtr0 == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "outDataPtr0 is null."),
                    return FAILED);
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(outDataPtr0, conv3d_transpose_d->GetInDataAnchor(1)) != SUCCESS,
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                            "Adding edge from fused node: %s's index[0] to fusion node: %s's index[1] is failed.",
                            conv_node->GetName().c_str(), conv3d_transpose_d->GetName().c_str()),
      return FAILED);
  ge::OutDataAnchorPtr outDataPtr1 = GetPeerOutAnchorWithInDataAnchor(conv_node, 1);
  FUSION_PASS_CHECK(outDataPtr1 == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "outDataPtr1 is null."),
                    return FAILED);
  FUSION_PASS_CHECK(
    ge::GraphUtils::AddEdge(outDataPtr1, conv3d_transpose_d->GetInDataAnchor(0)) != SUCCESS,
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                          "Adding edge from fused node: %s's index[1] to fusion node: %s's index[0] is failed.",
                          conv_node->GetName().c_str(), conv3d_transpose_d->GetName().c_str()),
    return FAILED);
  FUSION_PASS_CHECK(
    ge::GraphUtils::AddEdge(bias_const_node->GetOutDataAnchor(0), conv3d_transpose_d->GetInDataAnchor(2)) != SUCCESS,
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                          "Adding edge from fused node: %s's index[0] to fusion node: %s's index[2] is failed.",
                          bias_const_node->GetName().c_str(), conv3d_transpose_d->GetName().c_str()),
    return FAILED);
  ge::OutDataAnchorPtr biasNodePtr = bias_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(biasNodePtr == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "biasNodePtr is null."),
                    return FAILED);
  auto in_anchors = biasNodePtr->GetPeerInDataAnchors();
  for (auto in_anchor : in_anchors) {
    in_anchor->UnlinkAll();
    FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(conv3d_transpose_d->GetOutDataAnchor(0), in_anchor) != SUCCESS,
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                            "Adding edge from fusion node: %s's index[0] to fused node: %s's indexes is failed.",
                            conv3d_transpose_d->GetName().c_str(), bias_node->GetName().c_str()),
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

REGISTER_PASS("Conv3DbpInputBiasAddFusionPass", BUILT_IN_GRAPH_PASS, Conv3DbpInputBiasAddFusionPass);
}  // namespace fe
