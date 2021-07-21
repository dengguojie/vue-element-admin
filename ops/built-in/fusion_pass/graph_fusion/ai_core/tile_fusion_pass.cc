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
 * \file tile_fusion_pass.cpp
 * \brief tile fusion pass(tile --> tile_d)
 */
#include "tile_fusion_pass.h"

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

static const char* FUSED_NODE = "Tile";

static const std::string PATTERN_FUSEDNODE = "FusedNodeTile";

static const int64_t MULTIPLE_NUM = 1;

vector<FusionPattern*> ConstToAttrTilePass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("TileConstToAttrFusion");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

// vector<ge::NodePtr> &fusionNodes: Store fusion nodes,
//       including newly added nodes and fused but not deleted nodes
Status ConstToAttrTilePass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  std::string fusionOpType = "TileD";
  std::vector<PassAttrInfo> tileAttrInfo;
  PassAttrInfo multiples = {1, "multiples", "SetListInt"};
  tileAttrInfo.push_back(multiples);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Begin to do tile fusion pass.");

  //  PatternFusionUtil patternFusionUtil;
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode is nullptr, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr fusionDescPtr = PatternFusionUtil::GetFusionOpDesc(fusedNode, fusionOpType, tileAttrInfo);
  FUSION_PASS_CHECK(fusionDescPtr == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "Fusion op desc is nullptr."),
                    return NOT_CHANGED);
  if (HasUnKnowDimShape(fusedNode)) {
    FUSION_PASS_CHECK(CheckOpSupported(fusionDescPtr), OP_LOGI(FUSED_NODE, "tile dynamic shape supported"),
                      return NOT_CHANGED);
    OP_LOGI(FUSED_NODE, "CheckOpSupported fail, tile dynamic");
  }

  // get const_data
  Operator tileOp = ge::OpDescUtils::CreateOperatorFromNode(fusedNode);
  Tensor data;
  if (tileOp.GetInputConstData("multiples", data) != GRAPH_SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "GetInputConstData of multiples failed.");
    return false;
  }
  DataType dtype = tileOp.GetInputDesc("multiples").GetDataType();
  std::vector<int64_t> const_data;
  size_t size = 0;
  if (dtype == ge::DT_INT32) {
    int32_t* const_data_ptr = (int32_t*)data.GetData();
    size = data.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back((int32_t)((*(const_data_ptr + i))));
    }
  } else if (dtype == ge::DT_INT64) {
    int64_t* const_data_ptr = (int64_t*)data.GetData();
    size = data.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back(((int64_t)(*(const_data_ptr + i))));
    }
  } else {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Input multiples not support this type");
  }

  // get with control edge
  auto constControlAnchors = fusedNode->GetInAllNodes().at(1)->GetInControlAnchor()->GetPeerOutControlAnchors();
  auto tileControlAnchors = fusedNode->GetInControlAnchor()->GetPeerOutControlAnchors();
  ge::NodePtr fusion_node = nullptr;
  for (size_t index = 0; index < const_data.size(); ++index) {
    if (const_data[index] != MULTIPLE_NUM) {
      FUSION_PASS_CHECK(!CheckOpSupported(fusionDescPtr), OP_LOGI(FUSED_OP_TYPE.c_str(), "Op TileD Not Supported."),
                        return NOT_CHANGED);
      Status result = PatternFusionUtil::ConstToAttrWithNode(graph, fusedNode, fusionOpType, tileAttrInfo, fusion_node);
      if (result != SUCCESS) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Tile has input which is not a CONST, graph not changed.");
        return NOT_CHANGED;
      }
      // link control anchor
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to link control anchor.");
      for (auto outControlAnchor : constControlAnchors) {
        FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(outControlAnchor, fusion_node->GetInControlAnchor()) != SUCCESS,
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out control edge failed."), return FAILED);
      }
      for (auto outControlAnchor : tileControlAnchors) {
        FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(outControlAnchor, fusion_node->GetInControlAnchor()) != SUCCESS,
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out control edge failed."), return FAILED);
      }
      fusionNodes.push_back(fusion_node);
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Tile fusion pass end.");
      return SUCCESS;
    }
  }
  // connect data anchor
  auto preOutDataAnchor = fusedNode->GetInDataAnchor(0)->GetPeerOutAnchor();
  for (auto inDataAnchor : fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fusedNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(preOutDataAnchor, inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
  }
  // connect control anchor
  ge::NodePtr preNode = nullptr;
  preNode = fusedNode->GetInDataAnchor(0)->GetOwnerNode();
  auto preOutControlAnchor = preNode->GetOutControlAnchor();
  if (fusedNode->GetOutControlAnchor() != nullptr) {
    for (auto inControlAnchor : fusedNode->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fusedNode->GetOutControlAnchor(), inControlAnchor) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out control edge failed."), return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(preOutControlAnchor, inControlAnchor) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out control edge failed."), return FAILED);
    }
  }

  // remove constNode
  ge::InDataAnchorPtr tileAnchorPtr1 = fusedNode->GetInDataAnchor(1);
  ge::OutDataAnchorPtr constAnchorPtr1 = tileAnchorPtr1->GetPeerOutAnchor();
  ge::NodePtr constNode = constAnchorPtr1->GetOwnerNode();
  if (PatternFusionUtil::GetOutEdgeSize(constNode) == 0) {
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(constNode),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove Node[%s] failed", constNode->GetName().c_str()),
                      return FAILED);
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Remove const Node:[%s].", constNode->GetName().c_str());
  }
  // remove node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(fusedNode),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove Node[%s] failed", fusedNode->GetName().c_str()),
                    return FAILED);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Remove Node:[%s].", fusedNode->GetName().c_str());

  return SUCCESS;
}
REGISTER_PASS("TileConstToAttrFusion", BUILT_IN_GRAPH_PASS, ConstToAttrTilePass);
}  // namespace fe
