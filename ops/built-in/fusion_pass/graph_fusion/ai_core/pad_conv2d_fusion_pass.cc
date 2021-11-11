/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file pad_conv2d_fusion_pass.cpp
 * \brief pad conv2d fusion pass
 */
#include "pad_conv2d_fusion_pass.h"

#include <memory>
#include <sstream>
#include <string>

#include "anchor_util.h"
#include "error_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph/utils/op_desc_utils.h"
#include "tbe_ops_pass_util.h"

namespace fe {

static const char PATTERN_INPUTS1[] = "input1";
static const char PATTERN_PADD[] = "pad";
static const char PATTERN_CONV2D[] = "conv2d";
static const char PADD[] = "Pad";
static const char PADDINGS[] = "paddings";
static const char PADS[] = "pads";
static const char PADDING[] = "padding";
static const char CONV2D[] = "Conv2D";
static const char INPUT_SIZE[] = "input_size";
static const char CONV2DBACKPROPFILTERD[] = "Conv2DBackpropFilterD";
static const char FUSEBATCHNORMGRADD[] = "BNTrainingReduceGrad";
static const char CONV2DBACKPROPINPUTD[] = "Conv2DBackpropInputD";
static const char SLICE[] = "SliceD";
static const int DIM_NUM4 = 4;
static const int DIRECTION_COUNT = 2;
vector<FusionPattern*> PadConv2dFusionPass::DefinePatterns() {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define PadConv2dFusionPass pattern begin");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("PadConv2dFusionPass");

  FUSION_PASS_CHECK(pattern == nullptr, ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "New a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_PADD, {PADD})
      .AddOpDesc(PATTERN_CONV2D, {CONV2D})
      .AddOpDesc(PATTERN_INPUTS1)
      .SetInputs(PATTERN_CONV2D, {PATTERN_PADD, PATTERN_INPUTS1})
      .SetOutput(PATTERN_CONV2D);
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define PadConv2dFusionPass pattern end");
  return patterns;
}

Status PadConv2dFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define PadConv2dFusionPass fusion begin");
  ge::NodePtr padd_node = GetNodeFromMapping(PATTERN_PADD, mapping);
  FUSION_PASS_CHECK(padd_node == nullptr, ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "padD Node is null, fusion failed."),
                    return PARAM_INVALID);
  NOT_CHANGED_WITH_DYNAMIC_NODE({padd_node});

  ge::NodePtr conv2d_node = GetNodeFromMapping(PATTERN_CONV2D, mapping);
  FUSION_PASS_CHECK(conv2d_node == nullptr, ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "Conv2D Node is null, fusion failed."),
                    return PARAM_INVALID);

  int64_t conv_count = 0;
  int64_t dw_count = 0;

  auto out_data_anchor_padd = padd_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(out_data_anchor_padd == nullptr,
                    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "OutdataAnchor 0 of padd is null, fusion failed."),
                    return PARAM_INVALID);
  for (auto peer_in_data_anchor : out_data_anchor_padd->GetPeerInDataAnchors()) {
    ge::NodePtr next_node = peer_in_data_anchor->GetOwnerNode();
    FUSION_PASS_CHECK(next_node == nullptr,
                      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "next_node is null."),
                      return PARAM_INVALID);
    if (next_node->GetType() == CONV2D) {
      conv_count++;
    }
    if (next_node->GetType() == CONV2DBACKPROPFILTERD) {
      dw_count++;
    }
  }
  FUSION_PASS_CHECK(conv_count > 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Padnode have multiple conv2d outputs, can not fusion."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(dw_count > 1, OP_LOGI(FUSED_OP_TYPE.c_str(), "Padnode have multiple dw outputs, can not fusion."),
                    return NOT_CHANGED);

  std::vector<int64_t> pad_value;
  FUSION_PASS_CHECK(!GetIntConstValue(padd_node, "paddings", pad_value),
                    OP_LOGW(padd_node->GetName().c_str(), "Get const value of paddings failed"),
                    return FAILED);
  vector<vector<int64_t>> paddings;
  for (size_t i = 1; i < pad_value.size(); i += 2) {
    vector<int64_t> one_value;
    one_value.push_back(pad_value[i - 1]);
    one_value.push_back(pad_value[i]);
    paddings.push_back(one_value);
  }

  vector<int64_t> conv_pads;
  (void)ge::AttrUtils::GetListInt(conv2d_node->GetOpDesc(), PADS, conv_pads);
  FUSION_PASS_CHECK(conv_pads.size() != DIM_NUM4,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "pad dims not 4."), return NOT_CHANGED);
  if ((conv_pads[0] < 0) ||
      (conv_pads[1] < 0) ||
      (conv_pads[2] < 0) ||
      (conv_pads[3] < 0)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The number of convPads less than 0, can not fusion.");
    return NOT_CHANGED;
  }

  if (paddings.size() < DIM_NUM4 || paddings[0].size() < DIRECTION_COUNT || paddings[1].size() < DIRECTION_COUNT ||
      paddings[2].size() < DIRECTION_COUNT || paddings[3].size() < DIRECTION_COUNT) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The number of paddings not valid, can not fusion.");
    return NOT_CHANGED;
  }

  int64_t paddings_t;
  int64_t paddings_b;
  int64_t paddings_l;
  int64_t paddings_r;
  if (padd_node->GetOpDesc()->GetInputDesc(0).GetOriginFormat() == ge::FORMAT_NCHW) {
    paddings_t = paddings[2][0];
    paddings_b = paddings[2][1];
    paddings_l = paddings[3][0];
    paddings_r = paddings[3][1];
  } else if (padd_node->GetOpDesc()->GetInputDesc(0).GetOriginFormat() == ge::FORMAT_NHWC) {
    paddings_t = paddings[1][0];
    paddings_b = paddings[1][1];
    paddings_l = paddings[2][0];
    paddings_r = paddings[2][1];
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Padd intput Format is not NCHW or NHWC, can not fusion.");
    return NOT_CHANGED;
  }
  int64_t conv_pads_t;
  int64_t conv_pads_b;
  int64_t conv_pads_l;
  int64_t conv_pads_r;
  conv_pads_t = paddings_t + conv_pads[0];
  conv_pads_b = paddings_b + conv_pads[1];
  conv_pads_l = paddings_l + conv_pads[2];
  conv_pads_r = paddings_r + conv_pads[3];

  if (paddings_t < 0 || conv_pads_t > 255 || paddings_b < 0 || conv_pads_b > 255 ||
      paddings_l < 0 || conv_pads_l > 255 || paddings_r < 0 || conv_pads_r > 255) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Paddings value not in [0,255], can not fusion.");
    return NOT_CHANGED;
  }

  auto kernel_node = GetPeerOutNodeWithInDataAnchor(conv2d_node, 0);
  FUSION_PASS_CHECK(kernel_node == nullptr,
                    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "Kernel is null, fusion failed."),
                    return FAILED);

  auto kernel_desc = GetCurrNodeOutputDesc(kernel_node, 0);
  FUSION_PASS_CHECK(kernel_desc == nullptr,
                    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "Output 0 of kernel is null, fusion failed."),
                    return FAILED);
  if (kernel_desc->GetOriginFormat() == ge::FORMAT_NCHW &&
      (kernel_desc->GetShape().GetDim(2) <= conv_pads_t || kernel_desc->GetShape().GetDim(2) <= conv_pads_b)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Filter_H more than pad_H, can not fusion.");
    return NOT_CHANGED;
  }

  if (kernel_desc->GetOriginFormat() == ge::FORMAT_HWCN &&
      (kernel_desc->GetShape().GetDim(0) <= conv_pads_t || kernel_desc->GetShape().GetDim(0) <= conv_pads_b)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Filter_H more than pad_H, can not fusion.");
    return NOT_CHANGED;
  }

  vector<int64_t> pads;
  vector<int64_t> conv_pads_all;
  pads.push_back(paddings_t);
  pads.push_back(paddings_b);
  pads.push_back(paddings_l);
  pads.push_back(paddings_r);
  conv_pads_all.push_back(conv_pads_t);
  conv_pads_all.push_back(conv_pads_b);
  conv_pads_all.push_back(conv_pads_l);
  conv_pads_all.push_back(conv_pads_r);
  ge::OutControlAnchorPtr outCtrlAnchorPtr = padd_node->GetOutControlAnchor();
  FUSION_PASS_CHECK(outCtrlAnchorPtr == nullptr,
                    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "node is null."),
                    return PARAM_INVALID);
  if (!padd_node->GetOutControlAnchor()->GetPeerInControlAnchors().empty()) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "padd_node has control edge, can not fusion.");
    return NOT_CHANGED;
  }

  // Get conv2d_backprop_filter_d_node and check the graph
  ge::NodePtr conv2d_backprop_filter_d_node = nullptr;
  ge::NodePtr objNodePtr = nullptr;
  for (auto in_data_anchor : out_data_anchor_padd->GetPeerInDataAnchors()) {
    objNodePtr = in_data_anchor->GetOwnerNode();
    FUSION_PASS_CHECK(objNodePtr == nullptr,
                      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "objNodePtr is null."),
                      return PARAM_INVALID);
    if (in_data_anchor->GetOwnerNode()->GetType() == CONV2DBACKPROPFILTERD) {
      conv2d_backprop_filter_d_node = in_data_anchor->GetOwnerNode();
    }
    FUSION_PASS_CHECK(
        in_data_anchor->GetOwnerNode()->GetType() != CONV2D &&
            in_data_anchor->GetOwnerNode()->GetType() != CONV2DBACKPROPFILTERD,
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Output node is not Conv2D or Conv2DBackpropFilterD, can not fusion."),
        return NOT_CHANGED);
  }

  // Get batch_norm_grad_node and check the graph
  std::ostringstream description;
  if (conv2d_backprop_filter_d_node != nullptr) {
    ge::NodePtr batch_norm_grad_node = nullptr;
    auto node = GetPeerOutNodeWithInDataAnchor(conv2d_backprop_filter_d_node, 1);
    FUSION_PASS_CHECK(
        node == nullptr,
        ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
                                "Peer out node of conv2d_backprop_filter input 1 is null, fusion failed."),
        return PARAM_INVALID);

    if (node->GetType() == FUSEBATCHNORMGRADD) {
      batch_norm_grad_node = node;
    }
    if (batch_norm_grad_node != nullptr) {
      ge::NodePtr conv2d_backpropinput_node = nullptr;
      auto out_data_anchor_dbn = batch_norm_grad_node->GetOutDataAnchor(0);
      FUSION_PASS_CHECK(
          out_data_anchor_dbn == nullptr,
          ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "OutdataAnchor 0 of batch_norm_grad is null, fusion failed."),
          return PARAM_INVALID);

      for (auto in_data_anchor : out_data_anchor_dbn->GetPeerInDataAnchors()) {
        if (in_data_anchor->GetOwnerNode()->GetType() == CONV2DBACKPROPINPUTD) {
          conv2d_backpropinput_node = in_data_anchor->GetOwnerNode();
        }
      }
      if (conv2d_backpropinput_node != nullptr) {
        // Get slice_node and check the graph
        ge::NodePtr slice_node = nullptr;
        int flag_slice = 0;

        auto out_data_anchor_dx = conv2d_backpropinput_node->GetOutDataAnchor(0);
        FUSION_PASS_CHECK(out_data_anchor_dx == nullptr,
                          ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
                                                  "OutdataAnchor 0 of conv2d_backpropinput is null, fusion failed."),
                          return PARAM_INVALID);
        for (auto in_data_anchor : out_data_anchor_dx->GetPeerInDataAnchors()) {
          if (in_data_anchor->GetOwnerNode()->GetType() == SLICE) {
            slice_node = in_data_anchor->GetOwnerNode();
          }
          if (in_data_anchor->GetOwnerNode()->GetType() != SLICE) {
            flag_slice = 1;
          }
        }
        if (slice_node != nullptr && flag_slice == 0) {
          FUSION_PASS_CHECK(conv2d_backpropinput_node->GetOpDesc()->UpdateOutputDesc(
                                0, slice_node->GetOpDesc()->GetOutputDesc(0)) != SUCCESS,
                            ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "Update output failed."), return FAILED);
          // change out edge of conv2dbackpropinput to slice
          FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(GetPeerOutAnchorWithInDataAnchor(slice_node, 0),
                                                       slice_node->GetInDataAnchor(0)) != SUCCESS,
                            ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "Remove slice input0 edge error"), return FAILED);
          description.str("");
          description << "Set paddings to " << conv2d_backpropinput_node->GetName().c_str() << " failed.";
          FUSION_PASS_CHECK(
              !ge::AttrUtils::SetListInt(conv2d_backpropinput_node->GetOpDesc(), PADS, conv_pads_all),
              ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), description.str().c_str()), return FAILED);
          FUSION_PASS_CHECK(!ge::AttrUtils::SetStr(conv2d_backpropinput_node->GetOpDesc(), PADDING, "SAME"),
                            ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "Set padding attr failed."), return FAILED);
          vector<int64_t> input_size = slice_node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();
          description.str("");
          description << "Set input_size to " << conv2d_backpropinput_node->GetName().c_str() << " failed.";
          FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(conv2d_backpropinput_node->GetOpDesc(), INPUT_SIZE, input_size),
                            ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), description.str().c_str()),
                            return FAILED);

          // remove slice_node output
          auto out_data_anchor_slice = slice_node->GetOutDataAnchor(0);
          FUSION_PASS_CHECK(
              out_data_anchor_slice == nullptr,
              ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "OutdataAnchor 0 of slice is null, fusion failed."),
              return PARAM_INVALID);

          for (auto out_data_anchor : out_data_anchor_slice->GetPeerInDataAnchors()) {
            FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(out_data_anchor_slice, out_data_anchor) != SUCCESS,
                              ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
            description.str("");
            description << "Add edge between node " << conv2d_backpropinput_node->GetName().c_str()
                              << ". and node " << out_data_anchor->GetOwnerNode()->GetName().c_str() << "failed.";
            FUSION_PASS_CHECK(
                ge::GraphUtils::AddEdge(conv2d_backpropinput_node->GetOutDataAnchor(0), out_data_anchor) != SUCCESS,
                ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), description.str().c_str()),
                return FAILED);
          }
          FUSION_PASS_CHECK(graph.RemoveNode(slice_node) != SUCCESS,
                            ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "Remove slice node failed."), return FAILED);
        }
      }
    }
  }
  vector<ge::NodePtr> node_vector;
  node_vector.push_back(conv2d_node);
  if (conv2d_backprop_filter_d_node != nullptr) {
    node_vector.push_back(conv2d_backprop_filter_d_node);
  }
  for (ge::NodePtr node_ptr : node_vector) {
    string node_name = node_ptr->GetType();
    // update input desc

    description.str("");
    description << "Update "<< node_name.c_str() << " input failed.";
    FUSION_PASS_CHECK(node_ptr->GetOpDesc()->UpdateInputDesc(0, padd_node->GetOpDesc()->GetInputDesc(0)) != SUCCESS,
                      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), description.str().c_str()), return FAILED);
    // change input edge of padd to conv2d/conv2DBackpropFilterD
    description.str("");
    description << "Remove " << node_name.c_str() << " input0 edge error.";
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(GetPeerOutAnchorWithInDataAnchor(node_ptr, 0),
                                                 node_ptr->GetInDataAnchor(0)) != SUCCESS,
                      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), description.str().c_str()), return FAILED);

    auto peer_out_0_padd_node = GetPeerOutNodeWithInDataAnchor(padd_node, 0);
    FUSION_PASS_CHECK(peer_out_0_padd_node == nullptr,
                      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "Get peer out of padd failed"), return FAILED);
    description.str("");
    description << "Add edge between node "
                << peer_out_0_padd_node->GetName().c_str()
                << ". and node " << node_ptr->GetName() << "failed.";
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(GetPeerOutAnchorWithInDataAnchor(padd_node, 0),
                                              node_ptr->GetInDataAnchor(0)) != SUCCESS,
                      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), description.str().c_str()),
                      return FAILED);
    description.str(""); 
    description << "Set paddings to " << node_name.c_str() << " failed.";
    FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(node_ptr->GetOpDesc(), PADS, conv_pads_all),
                      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), description.str().c_str()), return FAILED);

    FUSION_PASS_CHECK(!ge::AttrUtils::SetStr(node_ptr->GetOpDesc(), PADDING, "SAME"),
                      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "Set padding attr failed."), return FAILED);
  }
  // remove padd_node output
  for (auto in_data_anchor : out_data_anchor_padd->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(padd_node->GetOutDataAnchor(0), in_data_anchor) != SUCCESS,
                      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
  }
  if (padd_node->GetOutControlAnchor()) {
    for (auto in_control_anchor : padd_node->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(padd_node->GetOutControlAnchor(), in_control_anchor) != SUCCESS,
                        ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "Remove out control edge failed."), return FAILED);
    }
  }
  FUSION_PASS_CHECK(graph.RemoveNode(padd_node) != SUCCESS, ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "Remove PadD node failed."),
                    return FAILED);
  fusion_nodes.push_back(conv2d_node);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define PadConv2dFusionPass fusion end");
  return SUCCESS;
}
REGISTER_PASS("PadConv2dFusionPass", BUILT_IN_GRAPH_PASS, PadConv2dFusionPass);
}  // namespace fe
