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
 * \file padd_maxpool_fusion_pass.cpp
 * \brief padd maxpool fusion pass
 */
#include "padd_maxpool_fusion_pass.h"

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
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"

using namespace ge;
namespace fe {
static const char* PADD = "Pad";
static const char* MAXPOOL = "MaxPool";
static const std::string PATTERN_PADD = "FusedNodePadD";
static const std::string PATTERN_MAXPOOL = "FusedNodeMaxPool";

vector<FusionPattern*> PaddMaxPoolFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("PaddMaxPoolFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_PADD, {PADD})
      .AddOpDesc(PATTERN_MAXPOOL, {MAXPOOL})
      .SetInputs(PATTERN_MAXPOOL, {PATTERN_PADD})
      .SetOutput(PATTERN_MAXPOOL);

  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define PaddMaxPoolFusionPass pattern end");
  return patterns;
}

Status PaddMaxPoolFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  // get all nodes
  ge::NodePtr pad_node = GetNodeFromMapping(PATTERN_PADD, mapping);
  NOT_CHANGED_WITH_DYNAMIC_NODE({pad_node});
  ge::NodePtr maxpool_node = GetNodeFromMapping(PATTERN_MAXPOOL, mapping);
  FUSION_PASS_CHECK(pad_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "pad_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(maxpool_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "maxpool_node is null, fusion failed."),
                    return PARAM_INVALID);

  // check output link
  FUSION_PASS_CHECK(pad_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "pad_node output size is [%d], which not equal to 1.",
                            pad_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);

  // get all node's desc
  ge::OpDescPtr pad_desc = pad_node->GetOpDesc();
  ge::OpDescPtr maxpool_desc = maxpool_node->GetOpDesc();
  FUSION_PASS_CHECK(pad_desc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "pad_node's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(maxpool_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "maxpool_node's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  // get shape and format
  ge::GeTensorDesc input_desc = pad_desc->GetInputDesc(0);
  ge::GeTensorDesc output_desc = maxpool_desc->GetOutputDesc(0);
  ge::GeShape input_shape = input_desc.GetShape();
  ge::Format input_format = input_desc.GetFormat();

  // get op
  Operator op_pad = ge::OpDescUtils::CreateOperatorFromNode(pad_node);
  Operator op_maxpool = ge::OpDescUtils::CreateOperatorFromNode(maxpool_node);

  // get const paddings
  std::vector<int64_t> pad_value;
  FUSION_PASS_CHECK(!GetIntConstValue(pad_node, "paddings", pad_value),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Get const value of paddings failed"),
                    return NOT_CHANGED);
  std::vector<std::vector<int64_t>> paddings;
  for (size_t i = 1; i < pad_value.size(); i += 2) {
    vector<int64_t> one_value;
    one_value.push_back(pad_value[i - 1]);
    one_value.push_back(pad_value[i]);
    paddings.push_back(one_value);
  }

  // attr:ksize
  std::vector<int32_t> ksize;
  if (GRAPH_SUCCESS != op_maxpool.GetAttr("ksize", ksize)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "get attr ksize failed.");
    return NOT_CHANGED;
  }
  // attr:strides
  std::vector<int32_t> strides;
  if (GRAPH_SUCCESS != op_maxpool.GetAttr("strides", strides)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "get attr strides failed.");
    return NOT_CHANGED;
  }
  std::string padding;
  if (ge::GRAPH_SUCCESS != op_maxpool.GetAttr("padding", padding)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "get attr padding failed.");
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
  for (int i = 0; i < 4; i++) {
    if (paddings[i].size() != 2) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "the len of paddings[%d] is not match.", i);
      return NOT_CHANGED;
    }
  }
  if (ksize.size() != 4) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the len of ksize is not match.");
    return NOT_CHANGED;
  }
  if (strides.size() != 4) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the len of strides is not match.");
    return NOT_CHANGED;
  }
  if (strcmp(padding.c_str(), "VALID") != 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "padding is not VALID.");
    return NOT_CHANGED;
  }

  if (input_format == FORMAT_NHWC) {
    if ((paddings[0][0] != 0) || (paddings[0][1] != 0) || (paddings[3][0] != 0) || (paddings[3][1] != 0)) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "the values of paddings are not match.");
      return NOT_CHANGED;
    }
    if ((ksize[0] != 1) || (ksize[3] != 1)) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "the values of ksize are not match.");
      return NOT_CHANGED;
    }
    if ((strides[0] != 1) || (strides[3] != 1)) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "the values of strides are not match.");
      return NOT_CHANGED;
    }
  } else {
    if ((paddings[0][0] != 0) || (paddings[0][1] != 0) || (paddings[1][0] != 0) || (paddings[1][1] != 0)) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "the values of paddings are not match.");
      return NOT_CHANGED;
    }
    if ((ksize[0] != 1) || (ksize[1] != 1)) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "the values of ksize are not match.");
      return NOT_CHANGED;
    }
    if ((strides[0] != 1) || (strides[1] != 1)) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "the values of strides are not match.");
      return NOT_CHANGED;
    }
  }

  // create node
  std::shared_ptr<ge::OpDesc> pool_desc = nullptr;
  pool_desc = std::make_shared<ge::OpDesc>(pad_node->GetName() + "_pooling", "Pooling");
  FUSION_PASS_CHECK(pool_desc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "pool_desc is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(pool_desc->AddInputDesc(input_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "add input failed."),
                    return FAILED);
  FUSION_PASS_CHECK(pool_desc->AddOutputDesc(output_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output failed."), return FAILED);
  ge::NodePtr pool_node = graph.AddNode(pool_desc);
  fusionNodes.push_back(pool_node);

  // get op
  Operator op_pool = ge::OpDescUtils::CreateOperatorFromNode(pool_node);

  // attr:window
  std::vector<int32_t> window;
  if (input_format == FORMAT_NHWC) {
    window.push_back(ksize[1]);
    window.push_back(ksize[2]);
  } else {
    window.push_back(ksize[2]);
    window.push_back(ksize[3]);
  }
  op_pool.SetAttr("window", window);

  // attr:stride
  std::vector<int32_t> stride;
  if (input_format == FORMAT_NHWC) {
    stride.push_back(strides[1]);
    stride.push_back(strides[2]);
  } else {
    stride.push_back(strides[2]);
    stride.push_back(strides[3]);
  }
  op_pool.SetAttr("stride", stride);

  // attr:pad
  std::vector<int32_t> pad;
  if (input_format == FORMAT_NHWC) {
    pad.push_back(paddings[1][0]);
    pad.push_back(paddings[1][1]);
    pad.push_back(paddings[2][0]);
    pad.push_back(paddings[2][1]);
  } else {
    pad.push_back(paddings[2][0]);
    pad.push_back(paddings[2][1]);
    pad.push_back(paddings[3][0]);
    pad.push_back(paddings[3][1]);
  }
  op_pool.SetAttr("pad", pad);

  // attr:mode
  op_pool.SetAttr("mode", 0);

  // attr:offset_x
  op_pool.SetAttr("offset_x", 0);

  // attr:ceil_mode
  op_pool.SetAttr("ceil_mode", 1);

  // attr:global_pooling
  op_pool.SetAttr("global_pooling", false);

  // attr:dilation
  std::vector<int32_t> dilation{1, 1, 1, 1};
  op_pool.SetAttr("dilation", dilation);

  // connect input edge
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(pad_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            pool_node->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Add edge between node %s. and node %s failed.",
                            pad_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                            pool_node->GetName().c_str()),
                    return FAILED);

  // connect output edge
  for (auto inDataAnchor : maxpool_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(maxpool_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."),
                                                     return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(pool_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge failed."),
                                                     return FAILED);
  }

  // set node type
  pool_node->GetOpDesc()->SetType("Pooling");

  // try to delete Edge between paddings const node and pad node
  ge::NodePtr paddings_node = pad_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode();
  FUSION_PASS_CHECK(paddings_node == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "get paddings const node failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(paddings_node->GetOutDataAnchor(0),
                                               pad_node->GetInDataAnchor(1)) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Remove edge between const node and pad failed."),
                    return NOT_CHANGED);
  // try to delete paddings const node if const node have no Edge
  if (paddings_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() == 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the paddings const node have no output edge, will be deleted!");
    FUSION_PASS_CHECK(graph.RemoveNode(paddings_node) != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(),
                      "Remove paddings_node failed."), return NOT_CHANGED);
  }

  // delete fused nodes
  FUSION_PASS_CHECK(graph.RemoveNode(pad_node) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Remove pad_node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(maxpool_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove maxpool_node failed."),
                                                   return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "PaddMaxPoolFusionPass graph fusion success!");
  return SUCCESS;
}
REGISTER_PASS("PadMaxPoolFusionPass", BUILT_IN_GRAPH_PASS, PaddMaxPoolFusionPass);
}  // namespace fe
