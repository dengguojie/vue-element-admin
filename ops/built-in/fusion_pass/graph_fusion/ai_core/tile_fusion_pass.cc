/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @file  tile_fusion_pass.cpp
 *
 * @brief tile fusion pass(tile --> tile_d)
 *
 * author z00512353
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
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {

static const char *FUSED_NODE = "Tile";

static const std::string PATTERN_FUSEDNODE = "FusedNodeTile";

static const int64_t MULTIPLE_NUM = 1;

vector<FusionPattern *> ConstToAttrTilePass::DefinePatterns() {
  vector < FusionPattern * > patterns;

  FusionPattern *pattern = new(std::nothrow) FusionPattern("TileConstToAttrFusion");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
  return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE})
      .SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

// vector<ge::NodePtr> &fusionNodes: Store fusion nodes,
//       including newly added nodes and fused but not deleted nodes
Status ConstToAttrTilePass::Fusion(ge::ComputeGraph &graph,
                                   Mapping &mapping,
                                   vector<ge::NodePtr> &fusionNodes) {

  std::string fusionOpType = "TileD";
  std::vector<PassAttrInfo> tileAttrInfo;
  PassAttrInfo multiples = {1, "multiples", "SetListInt"};
  tileAttrInfo.push_back(multiples);

//  PatternFusionUtil patternFusionUtil;
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
           return PARAM_INVALID);
  ge::OpDescPtr fusionDescPtr =
      PatternFusionUtil::GetFusionOpDesc(fusedNode, fusionOpType, tileAttrInfo);
  FUSION_PASS_CHECK(fusionDescPtr == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Fusion OP Desc is nullptr."),return PARAM_INVALID);

  // get const_data
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fusedNode);
  Tensor data;
  if (GRAPH_SUCCESS != op.GetInputConstData("multiples", data)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "GetInputConstData of multiples failed.");
    return false;
  }
  DataType dtype = op.GetInputDesc("multiples").GetDataType();
  std::vector<int64_t> const_data;
  size_t size = 0;
  if (dtype == ge::DT_INT32) {
    int32_t* const_data_ptr = (int32_t*) data.GetData();
    size = data.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back((int32_t) ((*(const_data_ptr + i))));
    }
  } else if (dtype == ge::DT_INT64) {
    int64_t* const_data_ptr = (int64_t*) data.GetData();
    size = data.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back(((int64_t) (*(const_data_ptr + i))));
    }
  } else {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Input multiples not support this type");
  }

  ge::NodePtr fusion_node = nullptr;
  for (size_t i = 0; i < const_data.size(); ++i) {
    if (const_data[i] != MULTIPLE_NUM) {
      FUSION_PASS_CHECK(!CheckOpSupported(fusionDescPtr), OP_LOGI(FUSED_OP_TYPE.c_str(), "Op TileD Not Supported."),
          return NOT_CHANGED);
      Status ret = PatternFusionUtil::ConstToAttrWithNode(graph, fusedNode, fusionOpType, tileAttrInfo, fusion_node);
      if (ret != SUCCESS) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Tile has input which is not a CONST, graph not changed.");
        return NOT_CHANGED;
      }
      fusionNodes.push_back(fusion_node);
      return SUCCESS;
    }
  }
  // connect data anchor
  auto preOutDataAnchor = fusedNode->GetInDataAnchor(0)->GetPeerOutAnchor();
  for (auto inDataAnchor :
       fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fusedNode->GetOutDataAnchor(0),
                                        inDataAnchor) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(preOutDataAnchor,
                                    inDataAnchor) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
  }
  // connect control anchor
  ge::NodePtr preNode = nullptr;
  preNode = fusedNode->GetInDataAnchor(0)->GetOwnerNode();
  auto preOutControlAnchor = preNode->GetOutControlAnchor();
  if (fusedNode->GetOutControlAnchor() != nullptr) {
    for (auto inControlAnchor :
         fusedNode->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fusedNode->GetOutControlAnchor(),
                                          inControlAnchor) != SUCCESS,
               OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out control edge failed."), return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(preOutControlAnchor,
                                       inControlAnchor) != SUCCESS,
               OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out control edge failed."), return FAILED);
    }
  }

  // remove constNode
  ge::InDataAnchorPtr tileAnchorPtr1 = fusedNode->GetInDataAnchor(1);
  ge::OutDataAnchorPtr constAnchorPtr1 = tileAnchorPtr1->GetPeerOutAnchor();
  ge::NodePtr constNode = constAnchorPtr1->GetOwnerNode();
  if (PatternFusionUtil::GetOutEdgeSize(constNode) == 0) {
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(constNode),
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove Node[%s] failed", constNode->GetName().c_str()),
             return FAILED);
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Remove const Node:[%s].", constNode->GetName().c_str());
  }
  // remove node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(fusedNode),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove Node[%s] failed", fusedNode->GetName().c_str()),
           return FAILED);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Remove Node:[%s].", fusedNode->GetName().c_str());

  return SUCCESS;
}
REGISTER_PASS("TileConstToAttrFusion", BUILT_IN_GRAPH_PASS, ConstToAttrTilePass
);
}
