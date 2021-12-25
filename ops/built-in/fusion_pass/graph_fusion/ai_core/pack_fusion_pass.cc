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
 * \file pack_fusion_pass.cpp
 * \brief pack fusion pass(Pack --> Pack & Pack)
 */
#include "pack_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"

using namespace ge;
namespace fe {
static const char* FUSED_NODE = "Pack";
static const std::string PATTERN_FUSEDNODE = "Pack";
vector<FusionPattern*> PackFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("PackFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);

  return patterns;
}

Status PackFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed"),
                    return PARAM_INVALID);
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  size_t inputs_num = fusedDesc->GetInputsSize();
  // A maximum of 63 tensors are supported in mini mode.
  size_t NeedTangent = 63;
  if (HasUnKnowShape(fusedNode)) {
    // Maximum of 48 tensors are supported in mini mode for dynamic shape of pack
    NeedTangent = 48;
  }
  const int64_t max_inputs = static_cast<int64_t>(NeedTangent);
  FUSION_PASS_CHECK(static_cast<int64_t>(inputs_num) <= max_inputs,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "The amount of input of Pack node is less than %lld.", max_inputs),
                    return NOT_CHANGED);

  if (static_cast<int64_t>(inputs_num) > max_inputs) {
    size_t nodes_num = 0;
    size_t nodes_num1 = 0;
    nodes_num1 = inputs_num % max_inputs;
    if (nodes_num1 == 0) {
      nodes_num = inputs_num / max_inputs;
    } else {
      nodes_num = inputs_num / max_inputs + 1;
    }
    size_t final_clear_node_num = inputs_num - (max_inputs * (nodes_num - 1));

    // create Base_node concat node op description
    std::string Base_nodeName = fusedNode->GetName() + "/ConcatD" + "Base_node";
    std::shared_ptr<ge::OpDesc> concatBaseDesc = std::make_shared<ge::OpDesc>(Base_nodeName, "ConcatD");
    FUSION_PASS_CHECK(concatBaseDesc == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "concatBaseDesc is null, fusion failed."),
                      return PARAM_INVALID);

    for (size_t c = 0; c < nodes_num; c++) {
      ge::GeTensorDesc fusedDesc1 = fusedNode->GetOpDesc()->GetInputDesc(c);
      FUSION_PASS_CHECK(concatBaseDesc->AddInputDesc(c, fusedDesc1) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add fusedDesc1  failed."),
                        return FAILED);
    }
    ge::GeTensorDesc addOutputDesc0 = fusedNode->GetOpDesc()->GetOutputDesc(0);
    FUSION_PASS_CHECK(concatBaseDesc->AddOutputDesc(addOutputDesc0) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add addOutputDesc0  output desc failed."),
                      return FAILED);

    concatBaseDesc->SetType("ConcatD");

    int64_t axis;
    ge::AttrUtils::GetInt(fusedDesc, "axis", axis);
    ge::AttrUtils::SetInt(concatBaseDesc, "concat_dim", axis);

    ge::NodePtr pack_base_node = graph.AddNode(concatBaseDesc);
    FUSION_PASS_CHECK(
        pack_base_node == nullptr,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
        return PARAM_INVALID);
    fusionNodes.push_back(pack_base_node);
    ge::AttrUtils::SetInt(pack_base_node->GetOpDesc(), "N", nodes_num);
    for (InDataAnchorPtr inAnchorPtr : fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(fusedNode->GetOutDataAnchor(0), inAnchorPtr),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."),
                        return FAILED);
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(pack_base_node->GetOutDataAnchor(0), inAnchorPtr),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge failed."),
                        return FAILED);
    }

    // create pack node op description of max_inputs
    std::string Pack_nodeName = fusedNode->GetName() + "/Pack" + "Base_node1";
    std::shared_ptr<ge::OpDesc> packBaseDesc = std::make_shared<ge::OpDesc>(Pack_nodeName, "Pack");
    FUSION_PASS_CHECK(packBaseDesc == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "packBaseDesc is null, fusion failed."),
                      return PARAM_INVALID);

    for (size_t c = 0; c < static_cast<int64_t>(max_inputs); c++) {
      ge::GeTensorDesc fusedDesc2 = fusedNode->GetOpDesc()->GetInputDesc(c);
      FUSION_PASS_CHECK(packBaseDesc->AddInputDesc(c, fusedDesc2) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add packBaseDesc failed."),
                        return FAILED);
    }
    ge::GeTensorDesc addOutputDesc1 = fusedNode->GetOpDesc()->GetOutputDesc(0);
    FUSION_PASS_CHECK(packBaseDesc->AddOutputDesc(addOutputDesc1) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add addOutputDesc1  output desc failed."),
                      return FAILED);

    packBaseDesc->SetType("Pack");

    int64_t axis2;
    ge::AttrUtils::GetInt(fusedDesc, "axis", axis2);
    ge::AttrUtils::SetInt(packBaseDesc, "axis", axis2);
    ge::NodePtr pack_base_node1 = graph.AddNode(packBaseDesc);
    FUSION_PASS_CHECK(
        pack_base_node1 == nullptr,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
        return PARAM_INVALID);
    fusionNodes.push_back(pack_base_node1);
    ge::AttrUtils::SetInt(pack_base_node1->GetOpDesc(), "N", max_inputs);

    // create pack node op description of max_inputs
    std::string Pack_nodeName2 = fusedNode->GetName() + "/Pack" + "Base_node_last";
    std::shared_ptr<ge::OpDesc> packLastDesc = std::make_shared<ge::OpDesc>(Pack_nodeName2, "Pack");
    FUSION_PASS_CHECK(packLastDesc == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "packLastDesc is null, fusion failed."),
                      return PARAM_INVALID);

    for (size_t c = 0; c < final_clear_node_num; c++) {
      ge::GeTensorDesc fusedDesc3 = fusedNode->GetOpDesc()->GetInputDesc(c);
      FUSION_PASS_CHECK(packLastDesc->AddInputDesc(c, fusedDesc3) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add fusedDesc3 failed."), return FAILED);
    }
    ge::GeTensorDesc addOutputDesc2 = fusedNode->GetOpDesc()->GetOutputDesc(0);
    FUSION_PASS_CHECK(packLastDesc->AddOutputDesc(addOutputDesc2) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add addOutputDesc2 output desc failed."),
                      return FAILED);

    packLastDesc->SetType("Pack");

    int64_t axis3;
    ge::AttrUtils::GetInt(fusedDesc, "axis", axis3);
    ge::AttrUtils::SetInt(packLastDesc, "axis", axis3);
    ge::NodePtr pack_base_node3 = graph.AddNode(packLastDesc);
    FUSION_PASS_CHECK(
        pack_base_node3 == nullptr,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
        return PARAM_INVALID);
    fusionNodes.push_back(pack_base_node3);
    ge::AttrUtils::SetInt(pack_base_node3->GetOpDesc(), "N", final_clear_node_num);
    for (size_t i = 0; i < nodes_num; i++) {
      if (i < nodes_num - 1) {
        ge::OpDescPtr packDesc = AttrUtils::CopyOpDesc(packBaseDesc);
        packDesc->SetName(fusedDesc->GetName() + "/Pack" + to_string(i));
        packDesc->SetType("Pack");

        ge::NodePtr pack_node = graph.AddNode(packDesc);
        FUSION_PASS_CHECK(
            pack_node == nullptr,
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
            return PARAM_INVALID);
        fusionNodes.push_back(pack_node);
        ge::AttrUtils::SetInt(pack_node->GetOpDesc(), "N", max_inputs);

        ge::GeTensorDesc PackOutputTensor_1 = packDesc->GetOutputDesc(0);
        ge::GeShape PackOutputShape_1 = PackOutputTensor_1.GetShape();
        int64_t dimnum = PackOutputShape_1.GetDimNum();

        int64_t axis4;
        const int64_t pack_num = static_cast<int64_t>(max_inputs);
        ge::AttrUtils::GetInt(pack_node->GetOpDesc(), "axis", axis4);
        if (axis4 < 0) {
          axis4 += (dimnum);
        }
        vector<int64_t> dimVector;
        for (int64_t j = 0; j < dimnum + 1; j++) {
          if (j < axis4) {
            dimVector.push_back(PackOutputShape_1.GetDim(j));
          } else if (j == axis4) {
            dimVector.push_back(pack_num);
          } else {
            dimVector.push_back(PackOutputShape_1.GetDim(j - 1));
          }
        }

        dimVector.erase(std::begin(dimVector) + axis4 + 1);

        ge::GeShape x_shape(dimVector);
        PackOutputTensor_1.SetShape(x_shape);
        PackOutputTensor_1.SetOriginShape(x_shape);
        packDesc->UpdateOutputDesc(0, PackOutputTensor_1);

        ge::GeTensorDesc PackInputTensor_1 = concatBaseDesc->GetInputDesc(i);
        ge::GeShape PackInputShape_1 = PackInputTensor_1.GetShape();
        concatBaseDesc->UpdateInputDesc(i, PackOutputTensor_1);

        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(pack_node->GetOutDataAnchor(0), pack_base_node->GetInDataAnchor(i)),
            VECTOR_FUSION_INNER_ERR_REPORT(
                FUSED_OP_TYPE.c_str(),
                "Add edge from fused node:%s's index[%lu] to fusion node:%s's index[%lu] failed.",
                pack_base_node->GetName().c_str(), i, pack_node->GetName().c_str(), i),
            return FAILED);

        for (size_t m = 0; static_cast<int64_t>(m) < max_inputs; m++) {
          FUSION_PASS_CHECK(
              SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(m + i * max_inputs)->GetPeerOutAnchor(),
                                                 pack_node->GetInDataAnchor(m)),
              VECTOR_FUSION_INNER_ERR_REPORT(
                  FUSED_OP_TYPE.c_str(),
                  "Add edge from fused node:%s's index[%lu] to fusion node:%s's index[%lu] failed.",
                  fusedNode->GetName().c_str(), (m + i * max_inputs), pack_node->GetName().c_str(), m),
              return FAILED);
        }
      } else {
        ge::OpDescPtr LastPackDesc = AttrUtils::CopyOpDesc(packLastDesc);
        LastPackDesc->SetName(fusedDesc->GetName() + "/Pack" + to_string(nodes_num - 1));
        LastPackDesc->SetType("Pack");

        ge::NodePtr last_pack_node = graph.AddNode(LastPackDesc);
        FUSION_PASS_CHECK(
            last_pack_node == nullptr,
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode is null, fusion failed."),
            return PARAM_INVALID);
        fusionNodes.push_back(last_pack_node);
        ge::AttrUtils::SetInt(last_pack_node->GetOpDesc(), "N", final_clear_node_num);
        ge::GeTensorDesc PackOutputTensor_2 = LastPackDesc->GetOutputDesc(0);
        ge::GeShape PackOutputShape_2 = PackOutputTensor_2.GetShape();
        int64_t dimnum = PackOutputShape_2.GetDimNum();
        int64_t axis5;
        int64_t pack_num = final_clear_node_num;
        ge::AttrUtils::GetInt(last_pack_node->GetOpDesc(), "axis", axis5);
        if (axis5 < 0) {
          axis5 += (dimnum);
        }

        vector<int64_t> dimVector;
        for (int64_t j = 0; j < dimnum + 1; j++) {
          if (j < axis5) {
            dimVector.push_back(PackOutputShape_2.GetDim(j));
          } else if (j == axis5) {
            dimVector.push_back(pack_num);
          } else {
            dimVector.push_back(PackOutputShape_2.GetDim(j - 1));
          }
        }
        dimVector.erase(std::begin(dimVector) + axis5 + 1);
        ge::GeShape x_shape(dimVector);
        PackOutputTensor_2.SetShape(x_shape);
        PackOutputTensor_2.SetOriginShape(x_shape);
        LastPackDesc->UpdateOutputDesc(0, PackOutputTensor_2);
        concatBaseDesc->UpdateInputDesc(i, PackOutputTensor_2);

        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(last_pack_node->GetOutDataAnchor(0), pack_base_node->GetInDataAnchor(i)),
            VECTOR_FUSION_INNER_ERR_REPORT(
                FUSED_OP_TYPE.c_str(),
                "Add edge from fused node:%s's index[%lu] to fusion node:%s's index[%lu] failed.",
                pack_base_node->GetName().c_str(), i, last_pack_node->GetName().c_str(), i),
            return FAILED);

        for (size_t n = 0; n < final_clear_node_num; n++) {
          FUSION_PASS_CHECK(
              SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(n + i * max_inputs)->GetPeerOutAnchor(),
                                                 last_pack_node->GetInDataAnchor(n)),
              VECTOR_FUSION_INNER_ERR_REPORT(
                  FUSED_OP_TYPE.c_str(),
                  "Add edge from fused node:%s's index[%lu] to fusion node:%s's index[%lu] failed.",
                  fusedNode->GetName().c_str(), (n + i * max_inputs), last_pack_node->GetName().c_str(), n),
              return FAILED);
        }
      }
    }
  }
  ge::NodeUtils::UnlinkAll(*(fusedNode.get()));
  Status ret = ge::GraphUtils::RemoveJustNode(graph, fusedNode);
  FUSION_PASS_CHECK(
      ret != ge::GRAPH_SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove Node [%s] failed", fusedNode->GetName().c_str()),
      return FAILED);

  return SUCCESS;
}
std::string PackPassName = "PackFusionPass";
REGISTER_PASS(PackPassName, BUILT_IN_GRAPH_PASS, PackFusionPass);
}  // namespace fe
