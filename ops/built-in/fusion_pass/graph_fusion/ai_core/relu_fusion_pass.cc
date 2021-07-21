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
 * \file relu_fusion_pass.cpp
 * \brief relu fusion pass(src --> relu)
 */
#include "relu_fusion_pass.h"
#include <iostream>
#include <map>
#include "cce/dnn_base_def.hpp"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

namespace fe {
static const char PATTERN_SRC[] = "src";
static const char PATTERN_INPUT[] = "input";
static const char PATTERN_RELU[] = "relu";

static const char FULLCONNECTION[] = "FullConnection";
static const char CONV2D[] = "Conv2D";
static const char ELTWISE[] = "Eltwise";

static const char ADD[] = "Add";
static const char ACTIVATION[] = "Activation";
static const char STREAMSWITCH[] = "StreamSwitch";

static const std::set<string> add_relu_fusion_input_op = {"Convolution", "Activation", "FusionBatchNorm",
                                                          "BatchNorm",   "Pooling",    "Eltwise"};

vector<FusionPattern*> ReluFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern1 = new (std::nothrow) FusionPattern("ReluFusion1");
  FUSION_PASS_CHECK(pattern1 == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern1->AddOpDesc(PATTERN_RELU, {ACTIVATION})
      .AddOpDesc(PATTERN_INPUT, {STREAMSWITCH})
      .AddOpDesc(PATTERN_SRC, {CONV2D, ELTWISE, FULLCONNECTION, ADD})
      .SetInputs(PATTERN_RELU, {PATTERN_INPUT, PATTERN_SRC})
      .SetOutput(PATTERN_RELU);
  patterns.push_back(pattern1);

  FusionPattern* pattern = new (std::nothrow) FusionPattern("ReluFusion");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_RELU, {ACTIVATION})
      .AddOpDesc(PATTERN_SRC, {CONV2D, ELTWISE, FULLCONNECTION, ADD})
      .SetInputs(PATTERN_RELU, {PATTERN_SRC})
      .SetOutput(PATTERN_RELU);
  patterns.push_back(pattern);

  return patterns;
}

Status ReluFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr src_node = GetNodeFromMapping(PATTERN_SRC, mapping);
  ge::NodePtr relu_node = GetNodeFromMapping(PATTERN_RELU, mapping);

  FUSION_PASS_CHECK(src_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "src_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(relu_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "relu_node is null, fusion failed."),
                    return PARAM_INVALID);

  int64_t mode = 0;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetInt(relu_node->GetOpDesc(), ge::ACTIVATION_ATTR_MODE, mode),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "get mode fail."), return NOT_CHANGED);

  FUSION_PASS_CHECK(src_node->GetOutDataNodes().size() > 1 || mode != ge::DOMI_ACTIVATION_RELU ||
                        graph.GetGraphOutNodes().find(relu_node->GetName()) != graph.GetGraphOutNodes().end() ||
                        relu_node->GetOutDataNodes().size() == 0,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "Extra conditions are not satisfied, graph is not changed"),
                    return SUCCESS);

  // do fuion of src node and relu node
  Status ret = DoFusion(src_node, relu_node);
  FUSION_PASS_CHECK(ret != SUCCESS && ret != NOT_CHANGED,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                            "Fail to do fusion between node[%s] and node[%s]: the reason"
                            "is that the parameters were not successfully obtained",
                            src_node->GetName().c_str(), relu_node->GetName().c_str()),
                    return ret);
  FUSION_PASS_CHECK(ret == NOT_CHANGED,
                    OP_LOGD(FUSED_OP_TYPE.c_str(),
                            "Can't do fusion between node[%s] and node[%s]: the reason"
                            "is that not all of check conditions are met.",
                            src_node->GetName().c_str(), relu_node->GetName().c_str()),
                    return ret);

  // connect switch node to src node
  int32_t in_edges_size = src_node->GetInDataNodes().size();
  FUSION_PASS_CHECK(in_edges_size < 0, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inEdges size is invalid"), return FAILED);
  Status ret2 = PatternFusionUtil::LinkControlEdge(relu_node, src_node);
  FUSION_PASS_CHECK(ret2 != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "LinkControlEdge failed."), return ret2);
  // delete relu
  Status ret3 = graph.RemoveNode(relu_node);
  FUSION_PASS_CHECK(ret3 != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Isolate relu node failed"), return ret3);
  fusionNodes.push_back(src_node);
  return SUCCESS;
}

Status ReluFusionPass::DoFusion(ge::NodePtr src_node, ge::NodePtr relu_node) {
  FUSION_PASS_CHECK(src_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "src_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(relu_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "relu_node is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr src_op = src_node->GetOpDesc();
  ge::OpDescPtr relu_op = relu_node->GetOpDesc();

  FUSION_PASS_CHECK(src_op == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "src_op is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(relu_op == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "relu_op is null, fusion failed."),
                    return PARAM_INVALID);

  if (CONV2D == src_op->GetType()) {
    // set relu flag to true
    FUSION_PASS_CHECK(!ge::AttrUtils::SetBool(src_op, ge::CONV_ATTR_NAME_RELU_FLAG, true),
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "set CONV_ATTR_NAME_RELU_FLAG fail."), return NOT_CHANGED);
  } else if (FULLCONNECTION == src_op->GetType()) {
    // set relu flag to true
    FUSION_PASS_CHECK(!ge::AttrUtils::SetBool(src_op, ge::FULL_CONNECTION_ATTR_RELU_FLAG, true),
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "set FULLCONNECTION_ATTR_RELU_FLAG fail."), return NOT_CHANGED);
  } else if (ELTWISE == src_op->GetType()) {
    // set relu flag to true
    FUSION_PASS_CHECK(!ge::AttrUtils::SetBool(src_op, ge::ELTWISE_ATTR_RELU_FLAG, true),
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "set ELTWISE_ATTR_RELU_FLAG fail."), return NOT_CHANGED);
  } else if (ADD == src_op->GetType()) {
    for (auto &in_anchor : src_node->GetAllInDataAnchors()) {
      FUSION_PASS_CHECK(in_anchor == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "in_anchor is null, fusion failed."),
                        return PARAM_INVALID);
      if (in_anchor->GetPeerOutAnchor() == nullptr || in_anchor->GetPeerOutAnchor()->GetOwnerNode() == nullptr ||
          in_anchor->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc() == nullptr) {
        continue;
      }
      string input_op_type = in_anchor->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc()->GetType();
      if (add_relu_fusion_input_op.find(input_op_type) == add_relu_fusion_input_op.end()) {
        return NOT_CHANGED;
      }
    }
    src_op->SetType(ELTWISE);
    FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(src_op, ge::ELTWISE_ATTR_MODE, cce::CC_ELTWISE_SUM),
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "set ELTWISE_ATTR_MODE fail."), return NOT_CHANGED);
    FUSION_PASS_CHECK(!ge::AttrUtils::SetBool(src_op, ge::ELTWISE_ATTR_RELU_FLAG, true),
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "set ELTWISE_ATTR_RELU_FLAG fail."), return NOT_CHANGED);
  }

  return SUCCESS;
}
REGISTER_PASS("ReluFusionPass", BUILT_IN_GRAPH_PASS, ReluFusionPass);
}  // namespace fe
