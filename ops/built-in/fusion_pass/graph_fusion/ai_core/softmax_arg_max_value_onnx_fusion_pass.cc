/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file softmax_arg_max_value_onnx_fusion_pass.cpp
 * \brief softmax ArgMax with value onnx fusion pass
 */
#include "softmax_arg_max_value_onnx_fusion_pass.h"

#include <numeric>
#include <sstream>
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
#include "fp16_t.hpp"

using namespace ge;
namespace fe {
static const char PATTERN_INPUT[] = "Input0";
static const char PATTERN_SOFTMAX_V2[] = "SoftmaxV2";
static const char PATTERN_ARG_MAX_D[] = "ArgMaxD";
static const char PATTERN_REDUCE_MAX_D[] = "ReduceMaxD";

static const char OPTYPE_SOFTMAX_V2[] = "SoftmaxV2";
static const char OPTYPE_ARG_MAX_D[] = "ArgMaxD";
static const char OPTYPE_REDUCE_MAX_D[] = "ReduceMaxD";

static const char ATTR_NAME_DIMENSION[] = "dimension";
static const char ATTR_NAME_KEEP_DIMS[] = "keep_dims";

/*!
 * @brief Define pattern.
 * The graph struct need to adapt and target is shown as follows:
 *
 * case1:
 *            x                           x
 *         /     \                        |
 *   SoftmaxV2  ArgMaxD               SoftmaxV2
 *      |          |      ==>             |
 *  ReduceMaxD     |               ArgMaxWithValue
 *      |       output1               /       \
 *   output0                      output0   output1
 *
 * case2:
 *            x                           x
 *            |                           |
 *        SoftmaxV2                   SoftmaxV2
 *         /     \        ==>             |
 *  ReduceMaxD  ArgMaxD            ArgMaxWithValue
 *      |         |                    /       \
 *   output0    output1            output0   output1
 *
 * @return vector<FusionPattern*> All valid patterns.
 */
vector<FusionPattern*> SoftmaxArgMaxValueONNXFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define pattern begin");

  vector<FusionPattern*> patterns;
  FusionPattern* pattern1 = new (std::nothrow) FusionPattern("SoftmaxArgMaxValueONNXFusionPass");
  FUSION_PASS_CHECK(pattern1 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to new a pattern object."),
                    return patterns);
  pattern1->AddOpDesc(PATTERN_INPUT)
      .AddOpDesc(PATTERN_SOFTMAX_V2, {OPTYPE_SOFTMAX_V2})
      .AddOpDesc(PATTERN_ARG_MAX_D, {OPTYPE_ARG_MAX_D})
      .AddOpDesc(PATTERN_REDUCE_MAX_D, {OPTYPE_REDUCE_MAX_D})
      .SetInputs(PATTERN_SOFTMAX_V2, {PATTERN_INPUT})
      .SetInputs(PATTERN_ARG_MAX_D, {PATTERN_INPUT})
      .SetInputs(PATTERN_REDUCE_MAX_D, {PATTERN_SOFTMAX_V2})
      .SetOutput(PATTERN_REDUCE_MAX_D);
  patterns.push_back(pattern1);

  FusionPattern* pattern2 = new (std::nothrow) FusionPattern("SoftmaxArgMaxValueONNXFusionPass");
  FUSION_PASS_CHECK(pattern2 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to new a pattern object."),
                    return patterns);
  pattern2->AddOpDesc(PATTERN_INPUT)
      .AddOpDesc(PATTERN_SOFTMAX_V2, {OPTYPE_SOFTMAX_V2})
      .AddOpDesc(PATTERN_ARG_MAX_D, {OPTYPE_ARG_MAX_D})
      .AddOpDesc(PATTERN_REDUCE_MAX_D, {OPTYPE_REDUCE_MAX_D})
      .SetInputs(PATTERN_SOFTMAX_V2, {PATTERN_INPUT})
      .SetInputs(PATTERN_ARG_MAX_D, {PATTERN_SOFTMAX_V2})
      .SetInputs(PATTERN_REDUCE_MAX_D, {PATTERN_SOFTMAX_V2})
      .SetOutput(PATTERN_REDUCE_MAX_D);
  patterns.push_back(pattern2);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define pattern end");
  return patterns;
}

Status SoftmaxArgMaxValueONNXFusionPass::GetArgMaxNode(const ge::NodePtr& preNode, ge::NodePtr& argMaxNode,
                                                       const string& postOpType) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "GetArgMaxNode begin. postOpType: %s", postOpType.c_str());

  auto preOutAnchors = preNode->GetAllOutDataAnchors();
  for (auto preOutAnchor : preOutAnchors) {
    if (preOutAnchor == nullptr) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "OutDataAnchor is null");
      continue;
    }

    auto inAnchors = preOutAnchor->GetPeerInDataAnchors();
    for (auto inAnchor : inAnchors) {
      if (inAnchor == nullptr) {
        OP_LOGD(FUSED_OP_TYPE.c_str(), "InDataAnchor is null");
        continue;
      }

      ge::NodePtr node = inAnchor->GetOwnerNode();
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Search node name: %s. node type: %s", node->GetName().c_str(),
              node->GetType().c_str());
      if (node->GetType() == postOpType) {
        argMaxNode = node;
        OP_LOGD(FUSED_OP_TYPE.c_str(), "GetArgMaxNode success");
        return SUCCESS;
      }
    }
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "GetArgMaxNode end");
  return PARAM_INVALID;
}

Status SoftmaxArgMaxValueONNXFusionPass::GetFusedNodes(const ge::ComputeGraph& graph, const Mapping& mapping,
                                                       ge::NodePtr& softmaxNode, ge::NodePtr& argMaxNode,
                                                       ge::NodePtr& reduceMaxNode) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "GetFusedNodes begin");

  softmaxNode = GetNodeFromMapping(PATTERN_SOFTMAX_V2, mapping);
  FUSION_PASS_CHECK(softmaxNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to get SoftmaxV2 node."),
                    return PARAM_INVALID);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "SoftmaxV2 nodeName: %s", softmaxNode->GetName().c_str());

  reduceMaxNode = GetNodeFromMapping(PATTERN_REDUCE_MAX_D, mapping);
  FUSION_PASS_CHECK(reduceMaxNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to get ArgMaxD node."),
                    return PARAM_INVALID);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "ReduceMaxD nodeName: %s", reduceMaxNode->GetName().c_str());

  ge::NodePtr inputNode = GetNodeFromMapping(PATTERN_INPUT, mapping);
  FUSION_PASS_CHECK(inputNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to get Input0 node."),
                    return PARAM_INVALID);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Input0 nodeName: %s", inputNode->GetName().c_str());

  string opType = PATTERN_ARG_MAX_D;
  Status getArgMaxRet = GetArgMaxNode(inputNode, argMaxNode, opType);  // Firstly, try to get ArgMaxD after Input0 node.
  if (getArgMaxRet == SUCCESS && argMaxNode != nullptr) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Got ArgMaxD node after Input0 node.");
  } else {
    getArgMaxRet = GetArgMaxNode(softmaxNode, argMaxNode, opType);  // Secondly, try to get ArgMaxD after SoftmaxV2.
    FUSION_PASS_CHECK(getArgMaxRet != SUCCESS || argMaxNode == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to get ReduceMaxD node."),
                      return PARAM_INVALID);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Got ArgMaxD node after SoftMaxV2 node.");
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "ArgMaxD nodeName: %s", argMaxNode->GetName().c_str());

  OP_LOGD(FUSED_OP_TYPE.c_str(), "GetFusedNodes end");
  return SUCCESS;
}

Status SoftmaxArgMaxValueONNXFusionPass::AddArgMaxWithValueNode(ge::ComputeGraph& graph, ge::NodePtr& softmaxNode,
                                                                ge::NodePtr& argMaxNode, ge::NodePtr& reduceMaxNode,
                                                                vector<ge::NodePtr>& newNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add ArgMaxWithValue begin");

  ge::OpDescPtr softmaxOpDesc = softmaxNode->GetOpDesc();
  FUSION_PASS_CHECK(softmaxOpDesc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to get op desc of SoftmaxV2 node."), return FAILED);
  std::string nodeName = softmaxOpDesc->GetName() + "/ArgMaxWithValue";
  ge::OpDescPtr argMaxWithValueOpDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((argMaxWithValueOpDesc = std::make_shared<ge::OpDesc>(nodeName, "ArgMaxWithValue")),
                          return INTERNAL_ERROR);

  ge::GeTensorDesc xDesc = softmaxOpDesc->GetOutputDesc(0).Clone();
  FUSION_PASS_CHECK(argMaxWithValueOpDesc->AddInputDesc("x", xDesc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to add input x for ArgMaxWithValue node."), return FAILED);

  ge::OpDescPtr argMaxOpDesc = argMaxNode->GetOpDesc();
  FUSION_PASS_CHECK(argMaxOpDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to get op desc of ArgMaxD node."),
                    return FAILED);
  ge::GeTensorDesc indiceDesc = argMaxOpDesc->GetOutputDesc(0).Clone();
  FUSION_PASS_CHECK(argMaxWithValueOpDesc->AddOutputDesc("indice", indiceDesc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to add output indice for ArgMaxWithValue node."),
                    return FAILED);

  ge::OpDescPtr reduceMaxOpDesc = reduceMaxNode->GetOpDesc();
  FUSION_PASS_CHECK(reduceMaxOpDesc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to get op desc of ReduceMaxD node."), return FAILED);
  ge::GeTensorDesc valuesDesc = reduceMaxOpDesc->GetOutputDesc(0).Clone();
  FUSION_PASS_CHECK(argMaxWithValueOpDesc->AddOutputDesc("values", valuesDesc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to add output values for ArgMaxWithValue node."),
                    return FAILED);

  ge::GeAttrValue dimension;
  FUSION_PASS_CHECK(argMaxOpDesc->GetAttr(ATTR_NAME_DIMENSION, dimension) == ge::GRAPH_FAILED,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to get attr %s from node %s", ATTR_NAME_DIMENSION,
                            argMaxOpDesc->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(argMaxWithValueOpDesc->SetAttr(ATTR_NAME_DIMENSION, dimension) == ge::GRAPH_FAILED,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to set attr %s to node %s", ATTR_NAME_DIMENSION,
                            argMaxOpDesc->GetName().c_str()),
                    return FAILED);

  ge::GeAttrValue keepDims;
  FUSION_PASS_CHECK(reduceMaxOpDesc->GetAttr(ATTR_NAME_KEEP_DIMS, keepDims) == ge::GRAPH_FAILED,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to get attr %s from node %s", ATTR_NAME_KEEP_DIMS,
                            reduceMaxOpDesc->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(argMaxWithValueOpDesc->SetAttr(ATTR_NAME_KEEP_DIMS, keepDims) == ge::GRAPH_FAILED,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to set attr %s to node %s", ATTR_NAME_KEEP_DIMS,
                            reduceMaxOpDesc->GetName().c_str()),
                    return FAILED);

  ge::NodePtr argMaxWithValueNode = graph.AddNode(argMaxWithValueOpDesc);
  FUSION_PASS_CHECK(argMaxWithValueNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to add ArgMaxWithValue node."), return FAILED);

  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(softmaxNode->GetOutDataAnchor(0), argMaxWithValueNode->GetInDataAnchor(0)),
      OP_LOGE(FUSED_OP_TYPE.c_str(),
              "Failed to add edge from Softmax(V2) node's output to ArgMaxWithValue node's input."),
      return FAILED);

  auto postArgMaxInAnchors = argMaxNode->GetOutDataAnchor(0)->GetPeerInDataAnchors();
  if (postArgMaxInAnchors.size() > 0) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "The output edge size of ArgMaxD Node is %d.", postArgMaxInAnchors.size());
    for (InDataAnchorPtr inAnchorPtr : postArgMaxInAnchors) {
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(argMaxWithValueNode->GetOutDataAnchor(0), inAnchorPtr),
                        OP_LOGE(FUSED_OP_TYPE.c_str(),
                                "Failed to add edge from ArgMaxWithValue node's output to postArgMaxD node's input."),
                        return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(),
              "Success to add edge from ArgMaxWithValue node's output to postArgMaxD node's input.");
    }
  }

  auto postReduceMaxInAnchors = reduceMaxNode->GetOutDataAnchor(0)->GetPeerInDataAnchors();
  if (postReduceMaxInAnchors.size() > 0) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "The output edge size of ReduceMaxD Node is %d.", postReduceMaxInAnchors.size());
    for (InDataAnchorPtr inAnchorPtr : postReduceMaxInAnchors) {
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(argMaxWithValueNode->GetOutDataAnchor(1), inAnchorPtr),
          OP_LOGE(FUSED_OP_TYPE.c_str(),
                  "Failed to add edge from ArgMaxWithValue node's output to postReduceMaxD node's input."),
          return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(),
              "Success to add edge from ArgMaxWithValue node's output to postReduceMaxD node's input.");
    }
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add ArgMaxWithValue end");
  return SUCCESS;
}

Status SoftmaxArgMaxValueONNXFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                                vector<ge::NodePtr>& newNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Fusion begin");

  ge::NodePtr softmaxNode;
  ge::NodePtr argMaxNode;
  ge::NodePtr reduceMaxNode;
  Status getNodesRet = GetFusedNodes(graph, mapping, softmaxNode, argMaxNode, reduceMaxNode);
  FUSION_PASS_CHECK(SUCCESS != getNodesRet, OP_LOGD(FUSED_OP_TYPE.c_str(), "Failed to get fused node."),
                    return getNodesRet);

  Status addNodeRet = AddArgMaxWithValueNode(graph, softmaxNode, argMaxNode, reduceMaxNode, newNodes);
  FUSION_PASS_CHECK(SUCCESS != addNodeRet, OP_LOGD(FUSED_OP_TYPE.c_str(), "Failed to add ArgMaxWithValue node."),
                    return addNodeRet);

  FUSION_PASS_CHECK(graph.RemoveNode(argMaxNode) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to remove ArgMaxD node."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(reduceMaxNode) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to remove ReduceMaxD node."), return FAILED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Fusion end");
  return SUCCESS;
}
REGISTER_PASS("SoftmaxArgMaxValueONNXFusionPass", BUILT_IN_GRAPH_PASS, SoftmaxArgMaxValueONNXFusionPass);
}  // namespace fe
