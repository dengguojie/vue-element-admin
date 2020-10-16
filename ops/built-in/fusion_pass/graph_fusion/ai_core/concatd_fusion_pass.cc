/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @file  concatd_fusion_pass.cpp
 *
 * @brief ConcatD fusion pass(ConcatD --> ConcatD)
 *
 */

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>

#include "concatd_fusion_pass.h"
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
static const char *FUSED_NODE = "ConcatD";
static const std::string PATTERN_FUSEDNODE = "FusedNodeConcat";
vector<FusionPattern *>ConcatDFusionPass::DefinePatterns() {
  vector <FusionPattern *> patterns;
  FusionPattern *pattern = new(std::nothrow) FusionPattern("ConcatDFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
           return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE})
          .SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);

  return patterns;
}

// vector<ge::NodePtr> &fusionNodes: Store fusion nodes,
//       including newly added nodes and fused but not deleted nodes
Status ConcatDFusionPass::Fusion(ge::ComputeGraph &graph,
                              Mapping &mapping,
                              vector<ge::NodePtr> &fusionNodes) {

  NodePtr fused_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  ge::OpDescPtr fusedDesc = fused_node->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "fused_node's OpDesc is null, fusion failed."),
           return PARAM_INVALID);
//A maximum of 63 tensors are supported in mini mode.
  size_t inputs_num = fusedDesc->GetInputsSize();
  FUSION_PASS_CHECK(inputs_num <= 63,
           OP_LOGD(FUSED_OP_TYPE.c_str(), "The amount of input of ConcatD node is less than 63."),
           return NOT_CHANGED);

  if (inputs_num > 63) {
    size_t auto_num;
    if (inputs_num <= 126) {
        auto_num = inputs_num / 2;
    } else {
        auto_num = 63;
    }
    size_t nodes_num, nodes_num1;
    nodes_num1 = inputs_num % auto_num;
    if (nodes_num1 == 0) {
      nodes_num = inputs_num / auto_num;
    }else {
      nodes_num = inputs_num / auto_num + 1;
    }
    size_t last_node_inputs_num = inputs_num - (auto_num*(nodes_num-1));

    ge::OpDescPtr ConcatdBaseDesc = AttrUtils::CopyOpDesc(fusedDesc);
    ConcatdBaseDesc->SetName(ConcatdBaseDesc->GetName() + "/ConcatD" + "Base_node");
    ConcatdBaseDesc->SetType("ConcatD");
    int64_t concat_dim;
    ge::AttrUtils::GetInt(fusedDesc, "concat_dim", concat_dim);
    ge::AttrUtils::SetInt(ConcatdBaseDesc, "concat_dim", concat_dim);

    for (size_t c=inputs_num-1; c>=nodes_num; c--) {
      OpDescUtils::ClearInputDesc(ConcatdBaseDesc, c);
    }

    ge::NodePtr concatd_base_node = graph.AddNode(ConcatdBaseDesc);
    fusionNodes.push_back(concatd_base_node);
    ge::AttrUtils::SetInt(concatd_base_node->GetOpDesc(), "N", nodes_num);
    FUSION_PASS_CHECK(concatd_base_node == nullptr,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "concatd_base_node:%s is null, fusion failed.",
             concatd_base_node->GetName().c_str()), return PARAM_INVALID);
    for (InDataAnchorPtr inAnchorPtr :
         fused_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(0), inAnchorPtr),
               OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(concatd_base_node->GetOutDataAnchor(0), inAnchorPtr),
               OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
    }

    for (size_t i = 0; i < nodes_num; i++) {
      if (i<nodes_num-1) {
        ge::OpDescPtr ConcatdDesc = AttrUtils::CopyOpDesc(fusedDesc);
        ConcatdDesc->SetName(ConcatdDesc->GetName()+"/ConcatD"+ to_string(i));
        ConcatdDesc->SetType("ConcatD");

        for (size_t a =inputs_num-1; a>=auto_num; a--) {
          OpDescUtils::ClearInputDesc(ConcatdDesc, a);
        }
        ge::NodePtr concatd_node = graph.AddNode(ConcatdDesc);
        fusionNodes.push_back(concatd_node);
        ge::AttrUtils::SetInt(concatd_node->GetOpDesc(), "N", auto_num);
//infershape begin
        ge::GeTensorDesc ConcatDInputTensor_1 = ConcatdDesc->GetInputDesc(0);
        ge::GeShape ConcatDInputShape_1 = ConcatDInputTensor_1.GetShape();
        int64_t dimnum = ConcatDInputShape_1.GetDimNum();
        int64_t concat_dim;
        int64_t num_concat = auto_num;
        ge::AttrUtils::GetInt(concatd_node->GetOpDesc(), "concat_dim", concat_dim);
        auto axis = concat_dim;

        if (axis < 0) {
          axis += (dimnum);
        }
        int64_t dim_axis_value = ConcatDInputShape_1.GetDim(axis);
        int32_t size = 0;
        for (int32_t i = 0; i < num_concat; i++) {
        size += dim_axis_value;
        }

        ge::GeTensorDesc ConcatDOutputTensor_1 = ConcatdDesc->GetOutputDesc(0);
        ge::GeShape ConcatDOutputShape_1 = ConcatDOutputTensor_1.GetShape();
        ConcatDOutputShape_1.SetDim(axis, size);
        ConcatDOutputTensor_1.SetShape(ConcatDOutputShape_1);
        ConcatDOutputTensor_1.SetOriginShape(ConcatDOutputShape_1);
        ConcatdDesc->UpdateOutputDesc(0, ConcatDOutputTensor_1);
        ConcatdBaseDesc->UpdateInputDesc(i,ConcatDOutputTensor_1);
//infershape end
        FUSION_PASS_CHECK(concatd_node == nullptr,
                 OP_LOGE(FUSED_OP_TYPE.c_str(), "concatd_node:%s is null, fusion failed.",
                concatd_node->GetName().c_str()), return PARAM_INVALID);

        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(concatd_node->GetOutDataAnchor(0),
                 concatd_base_node->GetInDataAnchor(i)),
                 OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                 concatd_base_node->GetName().c_str(), i,
                 concatd_node->GetName().c_str(), i),  return FAILED);

        for (size_t m=0; m<auto_num; m++) {
          FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(m+i*auto_num)->GetPeerOutAnchor(),
                   concatd_node->GetInDataAnchor(m)),
                   OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                   fused_node->GetName().c_str(), (m+i*auto_num),
                   concatd_node->GetName().c_str(), m), return FAILED);
        }
      }else {
        ge::OpDescPtr LastConcatDDesc = AttrUtils::CopyOpDesc(fusedDesc);
        LastConcatDDesc->SetName(LastConcatDDesc->GetName()+"/ConcatD"+ to_string(nodes_num-1));
        LastConcatDDesc->SetType("ConcatD");

        for (size_t b=inputs_num-1; b>=last_node_inputs_num; b--) {
          OpDescUtils::ClearInputDesc(LastConcatDDesc, b);
        }
        ge::NodePtr last_concatd_node = graph.AddNode(LastConcatDDesc);
        fusionNodes.push_back(last_concatd_node);
        ge::AttrUtils::SetInt(last_concatd_node->GetOpDesc(), "N", last_node_inputs_num);
//the last_node infershape begin
        ge::GeTensorDesc ConcatDInputTensor_2 = LastConcatDDesc->GetInputDesc(0);
        ge::GeShape ConcatDInputShape_2 = ConcatDInputTensor_2.GetShape();
        int64_t dimnum = ConcatDInputShape_2.GetDimNum();
        int64_t concat_dim;
        int64_t num_concat = last_node_inputs_num;
        ge::AttrUtils::GetInt(last_concatd_node->GetOpDesc(), "concat_dim", concat_dim);
        auto axis = concat_dim;
        if (axis < 0) {
          axis += (dimnum);
        }
        int64_t dim_axis_value = ConcatDInputShape_2.GetDim(axis);
        int32_t size = 0;
        for (int32_t i = 0; i < num_concat; i++) {
        size += dim_axis_value;
        }

        ge::GeTensorDesc ConcatDOutputTensor_2 = LastConcatDDesc->GetOutputDesc(0);
        ge::GeShape ConcatDOutputShape_2 = ConcatDOutputTensor_2.GetShape();
        ConcatDOutputShape_2.SetDim(axis, size);
        ConcatDOutputTensor_2.SetShape(ConcatDOutputShape_2);
        ConcatDOutputTensor_2.SetOriginShape(ConcatDOutputShape_2);
        LastConcatDDesc->UpdateOutputDesc(0, ConcatDOutputTensor_2);
        ConcatdBaseDesc->UpdateInputDesc(i,ConcatDOutputTensor_2);
//the last_node infershape end

        FUSION_PASS_CHECK(last_concatd_node == nullptr,
                 OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                 last_concatd_node->GetName().c_str()), return PARAM_INVALID);
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(last_concatd_node->GetOutDataAnchor(0),
                 concatd_base_node->GetInDataAnchor(i)),
                 OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                 concatd_base_node->GetName().c_str(), i,
                 last_concatd_node->GetName().c_str(), i), return FAILED);

        for (size_t n=0; n< last_node_inputs_num; n++) {
          FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(n+i*auto_num)->GetPeerOutAnchor(),
                   last_concatd_node->GetInDataAnchor(n)),
                   OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                   fused_node->GetName().c_str(), (n+i*auto_num),
                   last_concatd_node->GetName().c_str(), n), return FAILED);
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

  REGISTER_PASS("ZConcatDFusionPass", BUILT_IN_GRAPH_PASS, ConcatDFusionPass);
}
