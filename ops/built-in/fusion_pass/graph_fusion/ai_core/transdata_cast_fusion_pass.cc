/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file transdata_cast_fusion_pass.cpp
 * \brief transdata cast fusion pass
 *   (TransData --> Cast & TransData & Cast)
 */
#include "transdata_cast_fusion_pass.h"

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
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace ge;
namespace fe {

static const std::string PATTERN_TRANSDATA = "TransData";
static const std::string PATTERN_CAST_1 = "CastInt8ToFloat16";
static const std::string PATTERN_CAST_2 = "CastFloat16ToBool";

vector<FusionPattern*> TransdataCastFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("TransdataCastFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_TRANSDATA, {"TransData"}).SetOutput(PATTERN_TRANSDATA);

  patterns.push_back(pattern);

  return patterns;
}

Status TransdataCastFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_TRANSDATA, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  if (fusedDesc->GetOutputDesc(0).GetDataType() != ge::DT_BOOL) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "The output dtype is not bool, it is %d, not changed",
            fusedDesc->GetOutputDesc(0).GetDataType());
    return NOT_CHANGED;
  }

  ge::OpDescPtr castInt8ToFloat16Desc = AttrUtils::CloneOpDesc(fusedDesc);
  FUSION_PASS_CHECK(
      castInt8ToFloat16Desc == nullptr,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.", fusedNode->GetName().c_str()),
      return PARAM_INVALID);
  ge::OpDescPtr castFloat16ToBoolDesc = AttrUtils::CloneOpDesc(fusedDesc);
  FUSION_PASS_CHECK(
      castFloat16ToBoolDesc == nullptr,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.", fusedNode->GetName().c_str()),
      return PARAM_INVALID);
  castInt8ToFloat16Desc->SetName(fusedDesc->GetName() + "/" + PATTERN_CAST_1);
  castFloat16ToBoolDesc->SetName(fusedDesc->GetName() + "/" + PATTERN_CAST_2);
  castInt8ToFloat16Desc->SetType("Cast");
  castFloat16ToBoolDesc->SetType("Cast");

  // delete useless attribute
  FUSION_PASS_CHECK(SUCCESS != castInt8ToFloat16Desc->DelAttr("src_format"),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Delete the attr of src_format from castInt8ToFloat16Desc failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(SUCCESS != castInt8ToFloat16Desc->DelAttr("dst_format"),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Delete the attr of dst_format from castInt8ToFloat16Desc failed."),
                    return PARAM_INVALID);
  if (SUCCESS != castInt8ToFloat16Desc->DelAttr("groups")) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Delete the attr of groups from castInt8ToFloat16Desc failed.");
  }

  FUSION_PASS_CHECK(SUCCESS != castFloat16ToBoolDesc->DelAttr("src_format"),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Delete the attr of src_format from castFloat16ToBoolDesc failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(SUCCESS != castFloat16ToBoolDesc->DelAttr("dst_format"),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Delete the attr of dst_format from castFloat16ToBoolDesc failed."),
                    return PARAM_INVALID);
  if (SUCCESS != castFloat16ToBoolDesc->DelAttr("groups")) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Delete the attr of groups from castFloat16ToBoolDesc failed.");
  }

  // add dst_type attribute
  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(castInt8ToFloat16Desc, "dst_type", ge::DT_FLOAT16),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Fail to set attr to float16 from cast."),
                    return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(castFloat16ToBoolDesc, "dst_type", ge::DT_BOOL),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Fail to set attr to float16 from cast."),
                    return FAILED);

  // update input and output for cast_1, cast_2 and transdata
  ge::GeTensorDesc cast1OutputDesc = fusedDesc->GetInputDesc(0);
  ge::GeTensorDesc cast2InputDesc = fusedDesc->GetOutputDesc(0);
  cast1OutputDesc.SetDataType(ge::DT_FLOAT16);
  cast2InputDesc.SetDataType(ge::DT_FLOAT16);
  castInt8ToFloat16Desc->UpdateOutputDesc(0, cast1OutputDesc);
  castFloat16ToBoolDesc->UpdateInputDesc(0, cast2InputDesc);
  fusedDesc->UpdateInputDesc(0, cast1OutputDesc);
  fusedDesc->UpdateOutputDesc(0, cast2InputDesc);

  ge::NodePtr castInt8ToFloat16Node = graph.AddNode(castInt8ToFloat16Desc);
  ge::NodePtr castFloat16ToBoolNode = graph.AddNode(castFloat16ToBoolDesc);
  FUSION_PASS_CHECK(
      castInt8ToFloat16Node == nullptr,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
              castInt8ToFloat16Node->GetName().c_str()),
      return PARAM_INVALID);
  FUSION_PASS_CHECK(castFloat16ToBoolNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                            castFloat16ToBoolNode->GetName().c_str()),
                    return PARAM_INVALID);

  // save fusedNode input anchor
  ge::InDataAnchorPtr transdataInDataAnchor = fusedNode->GetInDataAnchor(0);
  ge::OutDataAnchorPtr trandataInDataPeerAnchor = transdataInDataAnchor->GetPeerOutAnchor();
  std::vector<InDataAnchorPtr> fusedNodeOutAnchorPeerInAnchors;
  for (auto inDataAnchor : fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
          fusedNodeOutAnchorPeerInAnchors.push_back(inDataAnchor);
  }

  // set edge for before fusedNode -> cast_1 -> fusedNode
  for (unsigned int i = 0; i < fusedNode->GetAllInDataAnchors().size(); i++) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(trandataInDataPeerAnchor, transdataInDataAnchor) != ge::GRAPH_SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "remove node failed."), return FAILED);
    OP_LOGI(FUSED_OP_TYPE.c_str(), "remove edge %u of node %s", i, fusedNode->GetName().c_str());

    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(trandataInDataPeerAnchor,
                                           castInt8ToFloat16Node->GetInDataAnchor(i)),
        OP_LOGE(FUSED_OP_TYPE.c_str(),
                "Add edge from before fused node:%s's index[%u] to fusion node:%s's index[%u] failed.",
                fusedNode->GetName().c_str(), i, castInt8ToFloat16Node->GetName().c_str(), i),
        return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from before fused node:%s's index[%u] to fusion node:%s's index[%u].",
            fusedNode->GetName().c_str(), i, castInt8ToFloat16Node->GetName().c_str(), i);

    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(castInt8ToFloat16Node->GetOutDataAnchor(i),
                                           fusedNode->GetInDataAnchor(i)),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index[%u] to fusion node:%s's index[%u] failed.",
                castInt8ToFloat16Node->GetName().c_str(), i, fusedNode->GetName().c_str(), i),
        return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index[%u] to fusion node:%s's index[%u].",
            castInt8ToFloat16Node->GetName().c_str(), i, fusedNode->GetName().c_str(), i);
  }

  // set edge for fusedNode -> cast_2
  FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetOutDataAnchor(0),
                                           castFloat16ToBoolNode->GetInDataAnchor(0)),
        OP_LOGE(FUSED_OP_TYPE.c_str(),
                "Add edge from before fused node:%s's index[%u] to fusion node:%s's index[%u] failed.",
                fusedNode->GetName().c_str(), 0, castFloat16ToBoolNode->GetName().c_str(), 0),
        return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index[%u] to fusion node:%s's index[%u].",
            fusedNode->GetName().c_str(), 0, castFloat16ToBoolNode->GetName().c_str(), 0);

  // set edge for cast_2 -> after fusedNode
  unsigned int i = 0;
  for (auto inDataAnchor : fusedNodeOutAnchorPeerInAnchors) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fusedNode->GetOutDataAnchor(0), inDataAnchor) != ge::GRAPH_SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "remove node failed."), return FAILED);
    OP_LOGI(FUSED_OP_TYPE.c_str(), "remove edge %u of node %s", i, fusedNode->GetName().c_str());

    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(castFloat16ToBoolNode->GetOutDataAnchor(0), inDataAnchor),
                      OP_LOGE(FUSED_OP_TYPE.c_str(),
                              "Add edge from fused node:%s's index[%u] to after fusion node:%s's index[%u] failed.",
                              castFloat16ToBoolNode->GetName().c_str(), i, fusedNode->GetName().c_str(), i),
                      return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index[%u] to after fusion node:%s's index[%u].",
            castFloat16ToBoolNode->GetName().c_str(), i, fusedNode->GetName().c_str(), i);
      i++;
  }

  fusionNodes.push_back(castInt8ToFloat16Node);
  fusionNodes.push_back(castFloat16ToBoolNode);
  return SUCCESS;
}

REGISTER_PASS("TransdataCastFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, TransdataCastFusionPass);
}  // namespace fe
