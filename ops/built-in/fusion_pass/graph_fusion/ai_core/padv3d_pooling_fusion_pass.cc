/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
 * \file padv3d_pooling_fusion_pass.h
 * \brief padv3d + pooling fusion pass
 */
#include "padv3d_pooling_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {
static const char *PADV3D = "PadV3D";
static const char *POOLING = "Pooling";
static const std::string PATTERN_PADV3D = "PadV3D";
static const std::string PATTERN_POOLING = "Pooling";

vector<FusionPattern*> Padv3dPoolingFusionPass::DefinePatterns() {
  vector < FusionPattern* > patterns;
  FusionPattern *pattern =
      new (std::nothrow) FusionPattern("Padv3dPoolingFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), 
                    "new a pattern object failed."), return patterns);

  pattern->AddOpDesc(PATTERN_PADV3D, {PADV3D})
      .AddOpDesc(PATTERN_POOLING, {POOLING})
      .SetInputs(PATTERN_POOLING, {PATTERN_PADV3D})
      .SetOutput(PATTERN_POOLING);

  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define Padv3dPoolingFusionPass pattern end");
  return patterns;
}

Status Padv3dPoolingFusionPass::Fusion(ge::ComputeGraph& graph,
                                        Mapping& mapping,
                                        vector<ge::NodePtr> &fusionNodes) {
  // get all nodes
  ge::NodePtr pad_node = GetNodeFromMapping(PATTERN_PADV3D, mapping);
  ge::NodePtr Pooling_node = GetNodeFromMapping(PATTERN_POOLING, mapping);
  FUSION_PASS_CHECK(pad_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), 
                    "pad_node is null, fusion failed."), return PARAM_INVALID);
  FUSION_PASS_CHECK(Pooling_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), 
                    "Pooling_node is null, fusion failed."), return PARAM_INVALID);

  // check output link
  FUSION_PASS_CHECK(pad_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "PADV3D_node output size is [%d], which not equal to 1.",
                            pad_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);

  // get all node's desc
  ge::OpDescPtr pad_desc = pad_node->GetOpDesc();
  ge::OpDescPtr Pooling_desc = Pooling_node->GetOpDesc();
  FUSION_PASS_CHECK(pad_desc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), 
                    "pad_node's OpDesc is null, fusion failed."), return PARAM_INVALID);
  FUSION_PASS_CHECK(Pooling_desc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), 
                    "Pooling_node's OpDesc is null, fusion failed."), return PARAM_INVALID);

  // get shape and format
  ge::GeTensorDesc input_desc = pad_desc->GetInputDesc(0);
  ge::GeTensorDesc output_desc = Pooling_desc->GetOutputDesc(0);
  ge::GeShape input_shape = input_desc.GetShape();
  ge::Format input_format = input_desc.GetFormat();

  // get op
  Operator op_pad = ge::OpDescUtils::CreateOperatorFromNode(pad_node);
  Operator op_Pooling = ge::OpDescUtils::CreateOperatorFromNode(Pooling_node);

  // attr:paddings
  std::vector<std::vector<int64_t>> paddings;
  if (ge::GRAPH_SUCCESS != op_pad.GetAttr("paddings", paddings)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "get attr padddings failed.");
    return GRAPH_FAILED;
  }
  // attr:window
  std::vector<int32_t> window;
  if (GRAPH_SUCCESS != op_Pooling.GetAttr("window", window)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "get attr window failed.");
    return GRAPH_FAILED;
  }
  // attr:stride
  std::vector<int32_t> stride;
  if (GRAPH_SUCCESS != op_Pooling.GetAttr("stride", stride)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "get attr stride failed.");
    return GRAPH_FAILED;
  }
  // attr:pad
  std::vector<int32_t> pad;
  if (ge::GRAPH_SUCCESS != op_Pooling.GetAttr("pad", pad)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "get attr pad failed.");
    return GRAPH_FAILED;
  }
  // attr:mode
  std::int32_t mode;
  if (ge::GRAPH_SUCCESS != op_Pooling.GetAttr("mode", mode)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "get attr mode failed.");
    return GRAPH_FAILED;
  }

  // attr:ceil_mode
  std::int32_t ceil_mode;
  if (ge::GRAPH_SUCCESS != op_Pooling.GetAttr("ceil_mode", ceil_mode)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "get attr ceil_mode failed.");
    return GRAPH_FAILED;
  }

  // verify
  if ((input_format != FORMAT_NHWC) && (input_format != FORMAT_NCHW)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "input format is not match.");
    return NOT_CHANGED;
  }
  if (paddings.size() != 4) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the len of paddings is not match.");
    return NOT_CHANGED;
  }
  if (window.size() != 2) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the len of window is not match.");
    return NOT_CHANGED;
  }
  if (stride.size() != 2) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the len of stride is not match.");
    return NOT_CHANGED;
  }
  if (pad.size() != 4) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the len of pad is not match.");
    return NOT_CHANGED;
  }

  if (input_format == FORMAT_NHWC) {
    if (paddings[0].size() != 2 || paddings[3].size() != 2) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "the len of padding[0] and padding[3] is not match.");
      return NOT_CHANGED;
    }
    if ((paddings[0][0] != 0) || (paddings[0][1] != 0) || (paddings[3][0] != 0) || (paddings[3][1] != 0)) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "the values of paddings are not match.");
      return NOT_CHANGED;
    }
    if (window[0] != window[1]) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "the values of window are not match.");
      return NOT_CHANGED;
    }
    if (stride[0] != stride[1]) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "the values of stride are not match.");
      return NOT_CHANGED;
    }
  } else {
    if (paddings[0].size() != 2 || paddings[2].size() != 2) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "the len of padding[0] and padding[2] is not match.");
      return NOT_CHANGED;
    }
    if ((paddings[0][0] != 0) || (paddings[0][1] != 0) || (paddings[2][0] != 0) || (paddings[2][1] != 0)) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "the values of paddings are not match.");
      return NOT_CHANGED;
    }
    if (window[0] != window[1]) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "the values of window are not match.");
      return NOT_CHANGED;
    }
    if (stride[0] != stride[1]) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "the values of stride are not match.");
      return NOT_CHANGED;
    }
  }

  // create node
  std::shared_ptr<ge::OpDesc> pool_desc = nullptr;
  pool_desc = std::make_shared<ge::OpDesc>(pad_node->GetName() + "_pooling", "Pooling");
  FUSION_PASS_CHECK(pool_desc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "pool_desc is null, fusion failed."),
           return PARAM_INVALID);
  FUSION_PASS_CHECK(pool_desc->AddInputDesc(input_desc) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "add input failed."), return FAILED);
  FUSION_PASS_CHECK(pool_desc->AddOutputDesc(output_desc) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "add output failed."), return FAILED);
  ge::NodePtr pool_node = graph.AddNode(pool_desc);
  fusionNodes.push_back(pool_node);

  // get op
  Operator op_pool = ge::OpDescUtils::CreateOperatorFromNode(pool_node);

  // attr:window
  op_pool.SetAttr("window", window);

  // attr:stride
  op_pool.SetAttr("stride", stride);

  // attr:pad
  std::vector<int32_t> new_pad;
  if (input_format == FORMAT_NHWC) {
    new_pad.push_back(paddings[1][0]);
    new_pad.push_back(paddings[1][1]);
    new_pad.push_back(paddings[2][0]);
    new_pad.push_back(paddings[2][1]);
  } else {
    new_pad.push_back(paddings[1][0]);
    new_pad.push_back(paddings[3][0]);
    new_pad.push_back(paddings[1][1]);
    new_pad.push_back(paddings[3][1]);
  }
  op_pool.SetAttr("pad", new_pad);

  // attr:mode
  op_pool.SetAttr("mode", mode);

  // attr:ceil_mode
  op_pool.SetAttr("ceil_mode", ceil_mode);

  // attr:global_pooling
  op_pool.SetAttr("global_pooling", false);

  // attr:dilation
  std::vector<int32_t> dilation {1,1,1,1};
  op_pool.SetAttr("dilation", dilation);

  // connect input edge
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
               pad_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
               pool_node->GetInDataAnchor(0)) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                   pad_node->GetInDataAnchor(0)
                       ->GetPeerOutAnchor()
                       ->GetOwnerNode()
                       ->GetName()
                       .c_str(),
                   pool_node->GetName().c_str()),
           return FAILED);

  // connect output edge
  for (auto inDataAnchor :
       Pooling_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(Pooling_node->GetOutDataAnchor(0),
                                        inDataAnchor) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(pool_node->GetOutDataAnchor(0),
                                     inDataAnchor) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
  }

  // set node type
  pool_node->GetOpDesc()->SetType("Pooling");

  // delete fused nodes
  FUSION_PASS_CHECK(graph.RemoveNode(pad_node) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove pad_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(Pooling_node) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove Pooling_node failed."), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Padv3dPoolingFusionPass graph fusion success!");
  return SUCCESS;
}
REGISTER_PASS("Padv3dPoolingFusionPass", BUILT_IN_GRAPH_PASS, Padv3dPoolingFusionPass);
}
