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
 * \file pack_fusion_pass.cpp
 * \brief pack fusion pass(Pack --> Pack & Pack)
 */
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>

#include "pack_fusion_pass.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"

using namespace ge;
namespace fe {
static const char* FUSED_NODE = "Pack";
static const std::string PATTERN_FUSEDNODE = "Pack";
vector<FusionPattern*> PackFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("PackFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);

  return patterns;
}

Status PackFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed"),
                    return PARAM_INVALID);
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  size_t inputs_num = fusedDesc->GetInputsSize();
  // A maximum of 63 tensors are supported in mini mode.
  size_t NeedTangent = 63;
  if (HasUnKnowShape(fusedNode)) {
    // Maximum of 48 tensors are supported in mini mode for dynamic shape of pack
    NeedTangent = 48;
  }
  FUSION_PASS_CHECK(inputs_num <= NeedTangent,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "The amount of input of Pack node is less than %lld.",
                            NeedTangent),
                    return NOT_CHANGED);

  if (inputs_num > NeedTangent) {
    size_t nodes_num, nodes_num1;
    nodes_num1 = inputs_num % NeedTangent;
    if (nodes_num1 == 0) {
      nodes_num = inputs_num / NeedTangent;
    } else {
      nodes_num = inputs_num / NeedTangent + 1;
    }
    size_t final_clear_node_num = inputs_num - (NeedTangent * (nodes_num - 1));

    ge::OpDescPtr packBaseDesc = AttrUtils::CopyOpDesc(fusedDesc);
    packBaseDesc->SetName(packBaseDesc->GetName() + "/ConcatD" + "Base_node");
    packBaseDesc->SetType("ConcatD");

    int64_t axis;
    ge::AttrUtils::GetInt(fusedDesc, "axis", axis);
    ge::AttrUtils::SetInt(packBaseDesc, "concat_dim", axis);

    for (size_t c = inputs_num - 1; c >= nodes_num; c--) {
      OpDescUtils::ClearInputDesc(packBaseDesc, c);
    }
    ge::NodePtr pack_base_node = graph.AddNode(packBaseDesc);
    fusionNodes.push_back(pack_base_node);
    ge::AttrUtils::SetInt(pack_base_node->GetOpDesc(), "N", nodes_num);
    FUSION_PASS_CHECK(
        pack_base_node == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode:%s is null, fusion failed.", pack_base_node->GetName().c_str()),
        return PARAM_INVALID);
    for (InDataAnchorPtr inAnchorPtr : fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(fusedNode->GetOutDataAnchor(0), inAnchorPtr),
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(pack_base_node->GetOutDataAnchor(0), inAnchorPtr),
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
    }

    for (size_t i = 0; i < nodes_num; i++) {
      if (i < nodes_num - 1) {
        ge::OpDescPtr packDesc = AttrUtils::CopyOpDesc(fusedDesc);
        packDesc->SetName(fusedDesc->GetName() + "/Pack" + to_string(i));
        packDesc->SetType("Pack");

        for (size_t a = inputs_num - 1; a >= NeedTangent; a--) {
          OpDescUtils::ClearInputDesc(packDesc, a);
        }
        ge::NodePtr pack_node = graph.AddNode(packDesc);
        fusionNodes.push_back(pack_node);
        ge::AttrUtils::SetInt(pack_node->GetOpDesc(), "N", NeedTangent);

        ge::GeTensorDesc PackOutputTensor_1 = packDesc->GetOutputDesc(0);
        ge::GeShape PackOutputShape_1 = PackOutputTensor_1.GetShape();
        int64_t dimnum = PackOutputShape_1.GetDimNum();

        int64_t axis;
        const int64_t pack_num = static_cast<int64_t>(NeedTangent);
        ge::AttrUtils::GetInt(pack_node->GetOpDesc(), "axis", axis);
        if (axis < 0) {
          axis += (dimnum);
        }
        vector<int64_t> dimVector;
        for (int64_t i = 0; i < dimnum + 1; i++) {
          if (i < axis) {
            dimVector.push_back(PackOutputShape_1.GetDim(i));
          } else if (i == axis) {
            dimVector.push_back(pack_num);
          } else {
            dimVector.push_back(PackOutputShape_1.GetDim(i - 1));
          }
        }

        dimVector.erase(std::begin(dimVector) + axis + 1);

        ge::GeShape x_shape(dimVector);
        PackOutputTensor_1.SetShape(x_shape);
        packDesc->UpdateOutputDesc(0, PackOutputTensor_1);

        ge::GeTensorDesc PackInputTensor_1 = packBaseDesc->GetInputDesc(i);
        ge::GeShape PackInputShape_1 = PackInputTensor_1.GetShape();
        packBaseDesc->UpdateInputDesc(i, PackOutputTensor_1);

        FUSION_PASS_CHECK(
            pack_node == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode:%s is null, fusion failed.", pack_node->GetName().c_str()),
            return PARAM_INVALID);

        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(pack_node->GetOutDataAnchor(0), pack_base_node->GetInDataAnchor(i)),
            OP_LOGE(FUSED_OP_TYPE.c_str(),
                    "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                    pack_base_node->GetName().c_str(), i, pack_node->GetName().c_str(), i),
            return FAILED);

        for (size_t m = 0; m < pack_num; m++) {
          FUSION_PASS_CHECK(
              SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(m + i * pack_num)->GetPeerOutAnchor(),
                                                 pack_node->GetInDataAnchor(m)),
              OP_LOGE(FUSED_OP_TYPE.c_str(),
                      "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                      fusedNode->GetName().c_str(), (m + i * pack_num), pack_node->GetName().c_str(), m),
              return FAILED);
        }
      } else {
        ge::OpDescPtr LastPackDesc = AttrUtils::CopyOpDesc(fusedDesc);
        LastPackDesc->SetName(fusedDesc->GetName() + "/Pack" + to_string(nodes_num - 1));
        LastPackDesc->SetType("Pack");

        for (size_t b = inputs_num - 1; b >= final_clear_node_num; b--) {
          OpDescUtils::ClearInputDesc(LastPackDesc, b);
        }
        ge::NodePtr last_pack_node = graph.AddNode(LastPackDesc);
        fusionNodes.push_back(last_pack_node);
        ge::AttrUtils::SetInt(last_pack_node->GetOpDesc(), "N", final_clear_node_num);
        ge::GeTensorDesc PackOutputTensor_2 = LastPackDesc->GetOutputDesc(0);
        ge::GeShape PackOutputShape_2 = PackOutputTensor_2.GetShape();
        int64_t dimnum = PackOutputShape_2.GetDimNum();
        int64_t axis;
        int64_t pack_num = final_clear_node_num;
        ge::AttrUtils::GetInt(last_pack_node->GetOpDesc(), "axis", axis);
        if (axis < 0) {
          axis += (dimnum);
        }

        vector<int64_t> dimVector;
        for (int64_t i = 0; i < dimnum + 1; i++) {
          if (i < axis) {
            dimVector.push_back(PackOutputShape_2.GetDim(i));
          } else if (i == axis) {
            dimVector.push_back(pack_num);
          } else {
            dimVector.push_back(PackOutputShape_2.GetDim(i - 1));
          }
        }
        dimVector.erase(std::begin(dimVector) + axis + 1);
        ge::GeShape x_shape(dimVector);
        PackOutputTensor_2.SetShape(x_shape);
        LastPackDesc->UpdateOutputDesc(0, PackOutputTensor_2);
        packBaseDesc->UpdateInputDesc(i, PackOutputTensor_2);

        FUSION_PASS_CHECK(
            last_pack_node == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", last_pack_node->GetName().c_str()),
            return PARAM_INVALID);
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(last_pack_node->GetOutDataAnchor(0), pack_base_node->GetInDataAnchor(i)),
            OP_LOGE(FUSED_OP_TYPE.c_str(),
                    "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                    pack_base_node->GetName().c_str(), i, last_pack_node->GetName().c_str(), i),
            return FAILED);

        for (size_t n = 0; n < final_clear_node_num; n++) {
          FUSION_PASS_CHECK(
              SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(n + i * pack_num)->GetPeerOutAnchor(),
                                                 last_pack_node->GetInDataAnchor(n)),
              OP_LOGE(FUSED_OP_TYPE.c_str(),
                      "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                      fusedNode->GetName().c_str(), (n + i * pack_num), last_pack_node->GetName().c_str(), n),
              return FAILED);
        }
      }
    }
  }
  for (auto inAnchor : fusedNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }
  for (auto outAnchor : fusedNode->GetAllOutDataAnchors()) {
    if (outAnchor != nullptr) {
      outAnchor->UnlinkAll();
    }
  }
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(fusedNode),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove Node [%s] failed", fusedNode->GetName().c_str()),
                    return FAILED);

  return SUCCESS;
}
std::string PackPassName = "PackFusionPass";
REGISTER_PASS(PackPassName, BUILT_IN_GRAPH_PASS, PackFusionPass);
}  // namespace fe
