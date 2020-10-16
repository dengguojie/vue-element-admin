/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @flie   add_n_fusion_pass.cpp
 *
 * @brief  AddN fusion pass(ADDN --> ADDN)
 *
 */

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>

#include "add_n_fusion_pass.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {
  static const char *FUSED_NODE = "AddN";
  static const std::string PATTERN_FUSEDNODE = "AddN";

vector<FusionPattern *>AddNFusionPass::DefinePatterns() {
  vector <FusionPattern *> patterns;
  FusionPattern *pattern = new(std::nothrow) FusionPattern("AddNFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
           return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE})
          .SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);

  return patterns;
}

// vector<ge::NodePtr> &fusionNodes: Store fusion nodes,
//       including newly added nodes and fused but not deleted nodes
Status AddNFusionPass::Fusion(ge::ComputeGraph &graph,
                              Mapping &mapping,
                              vector<ge::NodePtr> &fusionNodes) {
  ge::NodePtr fused_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fused_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed"),
           return PARAM_INVALID);
  ge::OpDescPtr fusedDesc = fused_node->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "fused_node's OpDesc is null, fusion failed."),
           return PARAM_INVALID);

  size_t inputs_num = fusedDesc->GetInputsSize();
  FUSION_PASS_CHECK(inputs_num <= 63,
           OP_LOGD(FUSED_OP_TYPE.c_str(), "The amount of input of AddN node is less than 63."),
           return NOT_CHANGED);

  if (inputs_num > 63) {
    size_t nodes_num, nodes_num1;
    nodes_num1 = inputs_num % 62;
    if (nodes_num1 == 0) {
      nodes_num = inputs_num / 62;
    }else {
      nodes_num = inputs_num / 62 + 1;
    }
    size_t last_node_inputs_num = inputs_num - (62*(nodes_num-1));

    if (last_node_inputs_num == 1) {
      nodes_num -= 1;
    }

    ge::OpDescPtr addNBaseDesc = AttrUtils::CopyOpDesc(fusedDesc);
    addNBaseDesc->SetName(addNBaseDesc->GetName() + "/AddN" + "Base_node");
    addNBaseDesc->SetType("AddN");
    for (size_t c=inputs_num-1; c>=nodes_num; c--) {
      OpDescUtils::ClearInputDesc(addNBaseDesc, c);
    }
    ge::NodePtr add_n_base_node = graph.AddNode(addNBaseDesc);
    fusionNodes.push_back(add_n_base_node);
    ge::AttrUtils::SetInt(add_n_base_node->GetOpDesc(), "N", nodes_num);
    FUSION_PASS_CHECK(add_n_base_node == nullptr,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "add_n_base_node:%s is null, fusion failed.",
             add_n_base_node->GetName().c_str()), return PARAM_INVALID);
    for (InDataAnchorPtr inAnchorPtr :
         fused_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(0), inAnchorPtr),
               OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(add_n_base_node->GetOutDataAnchor(0), inAnchorPtr),
               OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
    }

    for (size_t i = 0; i < nodes_num; i++) {
      if (i<nodes_num-1) {
        ge::OpDescPtr addNDesc = AttrUtils::CopyOpDesc(fusedDesc);
        addNDesc->SetName(addNDesc->GetName()+"/AddN"+ to_string(i));
        addNDesc->SetType("AddN");

        for (size_t a =inputs_num-1; a>=62; a--) {
          OpDescUtils::ClearInputDesc(addNDesc, a);
        }
        ge::NodePtr add_n_node = graph.AddNode(addNDesc);
        fusionNodes.push_back(add_n_node);
        ge::AttrUtils::SetInt(add_n_node->GetOpDesc(), "N", 62);
        FUSION_PASS_CHECK(add_n_node == nullptr,
                 OP_LOGE(FUSED_OP_TYPE.c_str(), "add_n_node:%s is null, fusion failed.",
                 add_n_node->GetName().c_str()), return PARAM_INVALID);

        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(add_n_node->GetOutDataAnchor(0),
                 add_n_base_node->GetInDataAnchor(i)),
                 OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                 add_n_base_node->GetName().c_str(), i,
                 add_n_node->GetName().c_str(), i),  return FAILED);

        for (size_t m=0; m<62; m++) {
          FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(m+i*62)->GetPeerOutAnchor(),
                   add_n_node->GetInDataAnchor(m)),
                   OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                   fused_node->GetName().c_str(), (m+i*62),
                   add_n_node->GetName().c_str(), m), return FAILED);
        }
      } else {
        if (last_node_inputs_num == 1) {
          last_node_inputs_num = 63;
        }
        ge::OpDescPtr LastAddNDesc = AttrUtils::CopyOpDesc(fusedDesc);
        LastAddNDesc->SetName(LastAddNDesc->GetName()+"/AddN"+ to_string(nodes_num-1));
        LastAddNDesc->SetType("AddN");

        for (size_t b=inputs_num-1; b>=last_node_inputs_num; b--) {
          OpDescUtils::ClearInputDesc(LastAddNDesc, b);
        }
        ge::NodePtr last_add_n_node = graph.AddNode(LastAddNDesc);
        fusionNodes.push_back(last_add_n_node);
        ge::AttrUtils::SetInt(last_add_n_node->GetOpDesc(), "N", last_node_inputs_num);
        FUSION_PASS_CHECK(last_add_n_node == nullptr,
                 OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                 last_add_n_node->GetName().c_str()), return PARAM_INVALID);
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(last_add_n_node->GetOutDataAnchor(0),
                 add_n_base_node->GetInDataAnchor(i)),
                 OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                 add_n_base_node->GetName().c_str(), i,
                 last_add_n_node->GetName().c_str(), i), return FAILED);

        for (size_t n=0; n< last_node_inputs_num; n++) {
          FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(n+i*62)->GetPeerOutAnchor(),
                   last_add_n_node->GetInDataAnchor(n)),
                   OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                   fused_node->GetName().c_str(), (n+i*62),
                   last_add_n_node->GetName().c_str(), n), return FAILED);
        }
      }
    }
  }
  for (auto inAnchor : fused_node->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }
  for (auto outAnchor : fused_node->GetAllOutDataAnchors()) {
    if (outAnchor != nullptr) {
      outAnchor->UnlinkAll();
    }
  }
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(fused_node),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove Node [%s] failed", fused_node->GetName().c_str()), return FAILED);

  return SUCCESS;
}
REGISTER_PASS("AddNFusionPass", BUILT_IN_GRAPH_PASS, AddNFusionPass);
}
