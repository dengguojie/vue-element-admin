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
 * \file concat_v2_fusion_pass.cpp
 * \brief ConcatExt2 fusion pass(ConcatExt2 --> ConcatExt2)
 */
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>

#include "concat_v2_fusion_pass.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"

using namespace ge;
namespace fe {

static const char* FUSED_NODE = "ConcatV2";
static const std::string PATTERN_FUSEDNODE = "FusedNodeConcatV2";

vector<FusionPattern*> ConcatExt2FusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ConcatExt2FusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);

  return patterns;
}

// vector<ge::NodePtr> &fusionNodes: Store fusion nodes,
//including newly added nodes and fused but not deleted nodes
Status ConcatExt2FusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  std::string fusionOpType = "ConcatV2D";
  std::vector<PassAttrInfo> concatv2AttrInfo;
  ge::NodePtr fused_node = nullptr;
  ge::NodePtr fused_node1 = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  int32_t inputsize = fused_node1->GetAllInDataAnchors().size();
  PassAttrInfo axis = {inputsize - 1, "concat_dim", "SetInt"};
  concatv2AttrInfo.push_back(axis);
  FUSION_PASS_CHECK(fused_node1 == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed"),
                    return PARAM_INVALID);

  Status ret = PatternFusionUtil::ConstToAttrWithNode(graph, fused_node1, fusionOpType, concatv2AttrInfo, fused_node);
  if (ret != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Concatv2 has input which is not a constant, graph not changed.");
    return NOT_CHANGED;
  }

  ClearOpInferDepends(fused_node1);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Concatv2 fusion SUCCESSS!!!!!");

  ge::OpDescPtr fusedDesc = fused_node->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fused_node's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  // A maximum of 63 tensors are supported in mini mode.
  size_t NeedTangent = 63;
  size_t inputs_num = fusedDesc->GetInputsSize();
  if (inputs_num <= NeedTangent) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "The amount of input of ConcatV2D node is less than 63.");
    fusionNodes.push_back(fused_node);
    return SUCCESS;
  }

  if (inputs_num > NeedTangent) {
    size_t auto_num;
    if (inputs_num <= 126) {
      auto_num = inputs_num / 2;
    } else {
      auto_num = NeedTangent;
    }
    size_t nodes_num, nodes_num1;
    nodes_num1 = inputs_num % auto_num;
    if (nodes_num1 == 0) {
      nodes_num = inputs_num / auto_num;
    } else {
      nodes_num = inputs_num / auto_num + 1;
    }
    size_t last_node_inputs_num = inputs_num - (auto_num * (nodes_num - 1));

    ge::OpDescPtr ConcatExt2BaseDesc = AttrUtils::CopyOpDesc(fusedDesc);
    ConcatExt2BaseDesc->SetName(ConcatExt2BaseDesc->GetName() + "/ConcatV2D" + "Base_node");
    ConcatExt2BaseDesc->SetType("ConcatV2D");
    int64_t axis;
    ge::AttrUtils::GetInt(fusedDesc, "concat_dim", axis);
    ge::AttrUtils::SetInt(ConcatExt2BaseDesc, "concat_dim", axis);

    for (size_t c = inputs_num - 1; c >= nodes_num; c--) {
      OpDescUtils::ClearInputDesc(ConcatExt2BaseDesc, c);
    }

    ge::NodePtr concatext2_base_node = graph.AddNode(ConcatExt2BaseDesc);
    FUSION_PASS_CHECK(concatext2_base_node == nullptr,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "concatv2d_base_node:%s is null, fusion failed.",
                              concatext2_base_node->GetName().c_str()),
                      return PARAM_INVALID);
    fusionNodes.push_back(concatext2_base_node);
    ge::AttrUtils::SetInt(concatext2_base_node->GetOpDesc(), "N", nodes_num);

    for (InDataAnchorPtr inAnchorPtr : fused_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(0), inAnchorPtr),
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(concatext2_base_node->GetOutDataAnchor(0), inAnchorPtr),
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
    }

    for (size_t i = 0; i < nodes_num; i++) {
      if (i < nodes_num - 1) {
        ge::OpDescPtr ConcatExt2Desc = AttrUtils::CopyOpDesc(fusedDesc);
        ConcatExt2Desc->SetName(ConcatExt2Desc->GetName() + "/ConcatV2D" + to_string(i));
        ConcatExt2Desc->SetType("ConcatV2D");

        for (size_t a = inputs_num - 1; a >= auto_num; a--) {
          OpDescUtils::ClearInputDesc(ConcatExt2Desc, a);
        }

        ge::NodePtr concatext2_node = graph.AddNode(ConcatExt2Desc);
        FUSION_PASS_CHECK(concatext2_node == nullptr,
                          OP_LOGE(FUSED_OP_TYPE.c_str(), "concatv2d_node:%s is null, fusion failed.",
                                  concatext2_node->GetName().c_str()),
                          return PARAM_INVALID);
        fusionNodes.push_back(concatext2_node);
        ge::AttrUtils::SetInt(concatext2_node->GetOpDesc(), "N", auto_num);
        // infershape begin

        ge::GeTensorDesc ConcatExt2InputTensor_1 = ConcatExt2Desc->GetInputDesc(0);
        ge::GeShape ConcatExt2InputShape_1 = ConcatExt2InputTensor_1.GetShape();
        int64_t dimnum = ConcatExt2InputShape_1.GetDimNum();
        int64_t axis;
        int64_t num_concatext2 = auto_num;
        ge::AttrUtils::GetInt(concatext2_node->GetOpDesc(), "concat_dim", axis);

        if (axis < 0) {
          axis += (dimnum);
        }
        int64_t dim_axis_value = ConcatExt2InputShape_1.GetDim(axis);
        int32_t size = 0;
        for (int32_t i = 0; i < num_concatext2; i++) {
          size += dim_axis_value;
        }

        ge::GeTensorDesc ConcatExt2OutputTensor_1 = ConcatExt2Desc->GetOutputDesc(0);
        ge::GeShape ConcatExt2OutputShape_1 = ConcatExt2OutputTensor_1.GetShape();
        ConcatExt2OutputShape_1.SetDim(axis, size);
        ConcatExt2OutputTensor_1.SetShape(ConcatExt2OutputShape_1);
        ConcatExt2OutputTensor_1.SetOriginShape(ConcatExt2OutputShape_1);
        ConcatExt2Desc->UpdateOutputDesc(0, ConcatExt2OutputTensor_1);
        ConcatExt2BaseDesc->UpdateInputDesc(i, ConcatExt2OutputTensor_1);
        // infershape end

        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(concatext2_node->GetOutDataAnchor(0),
                                                             concatext2_base_node->GetInDataAnchor(i)),
                          OP_LOGE(FUSED_OP_TYPE.c_str(),
                                  "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                                  concatext2_base_node->GetName().c_str(), i, concatext2_node->GetName().c_str(), i),
                          return FAILED);

        for (size_t m = 0; m < auto_num; m++) {
          FUSION_PASS_CHECK(
              SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(m + i * auto_num)->GetPeerOutAnchor(),
                                                 concatext2_node->GetInDataAnchor(m)),
              OP_LOGE(FUSED_OP_TYPE.c_str(),
                      "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                      fused_node->GetName().c_str(), (m + i * auto_num), concatext2_node->GetName().c_str(), m),
              return FAILED);
        }
      } else {
        ge::OpDescPtr LastConcatExt2Desc = AttrUtils::CopyOpDesc(fusedDesc);
        LastConcatExt2Desc->SetName(LastConcatExt2Desc->GetName() + "/ConcatV2D" + to_string(nodes_num - 1));
        LastConcatExt2Desc->SetType("ConcatV2D");

        for (size_t b = inputs_num - 1; b >= last_node_inputs_num; b--) {
          OpDescUtils::ClearInputDesc(LastConcatExt2Desc, b);
        }
        ge::NodePtr last_concatext2_node = graph.AddNode(LastConcatExt2Desc);
        FUSION_PASS_CHECK(last_concatext2_node == nullptr,
                          OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                                  last_concatext2_node->GetName().c_str()),
                          return PARAM_INVALID);

        fusionNodes.push_back(last_concatext2_node);
        ge::AttrUtils::SetInt(last_concatext2_node->GetOpDesc(), "N", last_node_inputs_num);
        // infershape begin
        ge::GeTensorDesc ConcatExt2InputTensor_2 = LastConcatExt2Desc->GetInputDesc(0);
        ge::GeShape ConcatExt2InputShape_2 = ConcatExt2InputTensor_2.GetShape();
        int64_t dimnum = ConcatExt2InputShape_2.GetDimNum();
        int64_t axis;
        int64_t num_concatext2 = last_node_inputs_num;
        ge::AttrUtils::GetInt(last_concatext2_node->GetOpDesc(), "concat_dim", axis);

        if (axis < 0) {
          axis += (dimnum);
        }
        int64_t dim_axis_value = ConcatExt2InputShape_2.GetDim(axis);
        int32_t size = 0;
        for (int32_t i = 0; i < num_concatext2; i++) {
          size += dim_axis_value;
        }

        ge::GeTensorDesc ConcatExt2OutputTensor_2 = LastConcatExt2Desc->GetOutputDesc(0);
        ge::GeShape ConcatExt2OutputShape_2 = ConcatExt2OutputTensor_2.GetShape();
        ConcatExt2OutputShape_2.SetDim(axis, size);
        ConcatExt2OutputTensor_2.SetShape(ConcatExt2OutputShape_2);
        ConcatExt2OutputTensor_2.SetOriginShape(ConcatExt2OutputShape_2);
        LastConcatExt2Desc->UpdateOutputDesc(0, ConcatExt2OutputTensor_2);
        ConcatExt2BaseDesc->UpdateInputDesc(i, ConcatExt2OutputTensor_2);
        // infershape end

        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(last_concatext2_node->GetOutDataAnchor(0),
                    concatext2_base_node->GetInDataAnchor(i)),
                    OP_LOGE(FUSED_OP_TYPE.c_str(),
                    "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                    concatext2_base_node->GetName().c_str(), i, last_concatext2_node->GetName().c_str(), i),
            return FAILED);

        for (size_t n = 0; n < last_node_inputs_num; n++) {
          FUSION_PASS_CHECK(
              SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(n + i * auto_num)->GetPeerOutAnchor(),
                                                 last_concatext2_node->GetInDataAnchor(n)),
              OP_LOGE(FUSED_OP_TYPE.c_str(),
                      "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                      fused_node->GetName().c_str(), (n + i * auto_num), last_concatext2_node->GetName().c_str(), n),
              return FAILED);
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
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "remove fused_node node[%s] failed", fused_node->GetName().c_str()),
                    return FAILED);
  return SUCCESS;
}

REGISTER_PASS("ZConcatExt2FusionPass", BUILT_IN_GRAPH_PASS, ConcatExt2FusionPass);
}  // namespace fe
