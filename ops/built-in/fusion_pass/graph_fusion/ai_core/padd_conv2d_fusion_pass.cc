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
 * \file padd_conv2d_fusion_pass.cpp
 * \brief padd conv2d fusion pass
 */
#include "padd_conv2d_fusion_pass.h"
#include <memory>
#include <string>
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph/utils/graph_utils.h"

namespace fe {

static const char PATTERN_INPUTS1[] = "input1";
static const char PATTERN_PADD[] = "padd";
static const char PATTERN_CONV2D[] = "conv2d";
static const char PADD[] = "PadD";
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
vector<FusionPattern*> PaddConv2dFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define PaddConv2dFusionPass pattern begin");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("PaddConv2dFusionPass");

  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "New a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_PADD, {PADD})
      .AddOpDesc(PATTERN_CONV2D, {CONV2D})
      .AddOpDesc(PATTERN_INPUTS1)
      .SetInputs(PATTERN_CONV2D, {PATTERN_PADD, PATTERN_INPUTS1})
      .SetOutput(PATTERN_CONV2D);
  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define PaddConv2dFusionPass pattern end");
  return patterns;
}

Status PaddConv2dFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define PaddConv2dFusionPass fusion begin");
  ge::NodePtr padd_node = GetNodeFromMapping(PATTERN_PADD, mapping);
  FUSION_PASS_CHECK(padd_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "padD Node is null, fusion failed."),
                    return PARAM_INVALID);

  ge::NodePtr conv2d_node = GetNodeFromMapping(PATTERN_CONV2D, mapping);
  FUSION_PASS_CHECK(conv2d_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Conv2D Node is null, fusion failed."),
                    return PARAM_INVALID);

  int64_t conv_count = 0;
  int64_t dw_count = 0;
  for (auto peer_in_data_anchor : padd_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    ge::NodePtr next_node = peer_in_data_anchor->GetOwnerNode();
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
  vector<vector<int64_t>> paddings;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetListListInt(padd_node->GetOpDesc(), PADDINGS, paddings),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Get paddings attr failed."), return NOT_CHANGED);

  if (paddings.size() < DIM_NUM4 || paddings[0].size() < DIRECTION_COUNT || paddings[1].size() < DIRECTION_COUNT ||
      paddings[2].size() < DIRECTION_COUNT || paddings[3].size() < DIRECTION_COUNT) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The number of paddings not valid, can not fusion.");
    return NOT_CHANGED;
  }

  int64_t paddings_t;
  int64_t paddings_b;
  int64_t paddings_l;
  int64_t paddings_r;
  if (padd_node->GetOpDesc()->GetInputDesc(0).GetFormat() == ge::FORMAT_NCHW) {
    paddings_t = paddings[2][0];
    paddings_b = paddings[2][1];
    paddings_l = paddings[3][0];
    paddings_r = paddings[3][1];
  } else if (padd_node->GetOpDesc()->GetInputDesc(0).GetFormat() == ge::FORMAT_NHWC) {
    paddings_t = paddings[1][0];
    paddings_b = paddings[1][1];
    paddings_l = paddings[2][0];
    paddings_r = paddings[2][1];
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Padd intput Format is not NCHW or NHWC, can not fusion.");
    return NOT_CHANGED;
  }

  if (paddings_t < 0 || paddings_t > 255 || paddings_b < 0 || paddings_b > 255 || paddings_l < 0 || paddings_l > 255 ||
      paddings_r < 0 || paddings_r > 255) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Paddings value not in [0,255], can not fusion.");
    return NOT_CHANGED;
  }
  ge::NodePtr kernel_node = conv2d_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  if (kernel_node->GetOpDesc()->GetOutputDesc(0).GetFormat() == ge::FORMAT_NCHW &&
      (kernel_node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDim(2) <= paddings_t ||
       kernel_node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDim(2) <= paddings_b)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Filter_H more than pad_H, can not fusion.");
    return NOT_CHANGED;
  }

  if (kernel_node->GetOpDesc()->GetOutputDesc(0).GetFormat() == ge::FORMAT_HWCN &&
      (kernel_node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDim(0) <= paddings_t ||
       kernel_node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDim(0) <= paddings_b)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Filter_H more than pad_H, can not fusion.");
    return NOT_CHANGED;
  }

  vector<int64_t> pads;
  pads.push_back(paddings_t);
  pads.push_back(paddings_b);
  pads.push_back(paddings_l);
  pads.push_back(paddings_r);
  if (!padd_node->GetOutControlAnchor()->GetPeerInControlAnchors().empty()) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "padd_node has control edge, can not fusion.");
    return NOT_CHANGED;
  }

  // Get conv2d_backprop_filter_d_node and check the graph
  ge::NodePtr conv2d_backprop_filter_d_node = nullptr;
  for (auto in_data_anchor : padd_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    if (in_data_anchor->GetOwnerNode()->GetOpDesc()->GetType() == CONV2DBACKPROPFILTERD) {
      conv2d_backprop_filter_d_node = in_data_anchor->GetOwnerNode();
    }
    FUSION_PASS_CHECK(
        in_data_anchor->GetOwnerNode()->GetOpDesc()->GetType() != CONV2D &&
            in_data_anchor->GetOwnerNode()->GetOpDesc()->GetType() != CONV2DBACKPROPFILTERD,
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Output node is not Conv2D or Conv2DBackpropFilterD, can not fusion."),
        return NOT_CHANGED);
  }
  // Get batch_norm_grad_node and check the graph
  if (conv2d_backprop_filter_d_node != nullptr) {
    ge::NodePtr batch_norm_grad_node = nullptr;
    if (conv2d_backprop_filter_d_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc()->GetType() ==
        FUSEBATCHNORMGRADD) {
      batch_norm_grad_node = conv2d_backprop_filter_d_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode();
    }
    if (batch_norm_grad_node != nullptr) {
      ge::NodePtr conv2d_backpropinput_node = nullptr;
      for (auto in_data_anchor : batch_norm_grad_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
        if (in_data_anchor->GetOwnerNode()->GetOpDesc()->GetType() == CONV2DBACKPROPINPUTD) {
          conv2d_backpropinput_node = in_data_anchor->GetOwnerNode();
        }
      }
      if (conv2d_backpropinput_node != nullptr) {
        // Get slice_node and check the graph
        ge::NodePtr slice_node = nullptr;
        int flag_slice = 0;
        for (auto in_data_anchor : conv2d_backpropinput_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
          if (in_data_anchor->GetOwnerNode()->GetType() == SLICE) {
            slice_node = in_data_anchor->GetOwnerNode();
          }
          if (in_data_anchor->GetOwnerNode()->GetOpDesc()->GetType() != SLICE) {
            flag_slice = 1;
          }
        }
        if (slice_node != nullptr && flag_slice == 0) {
          FUSION_PASS_CHECK(conv2d_backpropinput_node->GetOpDesc()->UpdateOutputDesc(
                                0, slice_node->GetOpDesc()->GetOutputDesc(0)) != SUCCESS,
                            OP_LOGE(FUSED_OP_TYPE.c_str(), "Update output failed."), return FAILED);
          // change out edge of conv2dbackpropinput to slice
          FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(slice_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                       slice_node->GetInDataAnchor(0)) != SUCCESS,
                            OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove slice input0 edge error"), return FAILED);
          FUSION_PASS_CHECK(
              !ge::AttrUtils::SetListInt(conv2d_backpropinput_node->GetOpDesc(), PADS, pads),
              OP_LOGE(FUSED_OP_TYPE.c_str(), "Set paddings to %s failed.",
                      conv2d_backpropinput_node->GetName().c_str()),
              return FAILED);
          FUSION_PASS_CHECK(!ge::AttrUtils::SetStr(conv2d_backpropinput_node->GetOpDesc(), PADDING, "SAME"),
                            OP_LOGE(FUSED_OP_TYPE.c_str(), "Set padding attr failed."), return FAILED);
          vector<int64_t> input_size = slice_node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();
          FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(conv2d_backpropinput_node->GetOpDesc(), INPUT_SIZE, input_size),
                            OP_LOGE(FUSED_OP_TYPE.c_str(), "Set input_size to %s failed.",
                                    conv2d_backpropinput_node->GetName().c_str()),
                            return FAILED);

          // remove slice_node output
          for (auto out_data_anchor : slice_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
            FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(slice_node->GetOutDataAnchor(0), out_data_anchor) != SUCCESS,
                              OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
            FUSION_PASS_CHECK(
                ge::GraphUtils::AddEdge(conv2d_backpropinput_node->GetOutDataAnchor(0), out_data_anchor) != SUCCESS,
                OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                        conv2d_backpropinput_node->GetName().c_str(),
                        out_data_anchor->GetOwnerNode()->GetName().c_str()),
                return FAILED);
          }
          FUSION_PASS_CHECK(graph.RemoveNode(slice_node) != SUCCESS,
                            OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove slice node failed."), return FAILED);
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
    string node_name = node_ptr->GetOpDesc()->GetType();
    // update input desc
    FUSION_PASS_CHECK(node_ptr->GetOpDesc()->UpdateInputDesc(0, padd_node->GetOpDesc()->GetInputDesc(0)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Update %s input failed.", node_name.c_str()), return FAILED);
    // change input edge of padd to conv2d/conv2DBackpropFilterD
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(node_ptr->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                 node_ptr->GetInDataAnchor(0)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove %s input0 edge error", node_name.c_str()), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(padd_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                              node_ptr->GetInDataAnchor(0)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                              padd_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                              node_ptr->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(node_ptr->GetOpDesc(), PADS, pads),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Set paddings to %s failed.", node_name.c_str()), return FAILED);
    FUSION_PASS_CHECK(!ge::AttrUtils::SetStr(node_ptr->GetOpDesc(), PADDING, "SAME"),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Set padding attr failed."), return FAILED);
  }
  // remove padd_node output
  for (auto in_data_anchor : padd_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(padd_node->GetOutDataAnchor(0), in_data_anchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
  }
  if (padd_node->GetOutControlAnchor()) {
    for (auto in_control_anchor : padd_node->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(padd_node->GetOutControlAnchor(), in_control_anchor) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out control edge failed."), return FAILED);
    }
  }
  FUSION_PASS_CHECK(graph.RemoveNode(padd_node) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove PadD node failed."),
                    return FAILED);
  fusion_nodes.push_back(conv2d_node);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define PaddConv2dFusionPass fusion end");
  return SUCCESS;
}
REGISTER_PASS("PaddConv2dFusionPass", BUILT_IN_GRAPH_PASS, PaddConv2dFusionPass);
}  // namespace fe
