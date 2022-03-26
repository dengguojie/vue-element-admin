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
 * \file roi_extractor_fusion_pass.cc
 */
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "fp16_t.hpp"
#include "common/util/platform_info.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "securec.h"
#include "roi_extractor_fusion_pass.h"

using namespace std;
using namespace ge;

namespace fe {
static const string PatternRoi = "RoiExtractor";
static const int FeatureNum = 4;
static const int IndexDim = 0;

vector<FusionPattern*> RoiExtractorFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (nothrow) FusionPattern("RoiExtractorFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "New a pattern object failed."), return patterns);
  // define origin graph
  pattern->AddOpDesc(PatternRoi, {"RoiExtractor"}).SetOutput(PatternRoi);
  patterns.push_back(pattern);
  return patterns;
}

Status RoiExtractorFusionPass::Fusion(ComputeGraph& graph, Mapping& mapping, vector<NodePtr>& fusion_nodes) {
  PlatformInfo platform_info;
  OptionalInfo optional_info;
  if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) != SUCCESS) {
     OP_LOGW("RoiExtractor", "Fail to get platform info.");
     optional_info.soc_version == "";
  }
  OP_LOGD("RoiExtractor", "Get soc_version is: [%s].", optional_info.soc_version.c_str());
  if (optional_info.soc_version != "Ascend710") {
     OP_LOGW("RoiExtractor", "not support this soc_version");
     return FAILED;
  }

  NodePtr roi_node = GetNodeFromMapping(PatternRoi, mapping);
  FUSION_PASS_CHECK(roi_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "The roi_node is null, fusion failed."),
                    return PARAM_INVALID);
  OpDescPtr roi_desc = roi_node->GetOpDesc();
  FUSION_PASS_CHECK(roi_desc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "The roi_desc is null, fusion failed."),
                    return PARAM_INVALID);

  std::shared_ptr<ge::OpDesc> balancerois_desc = std::make_shared<ge::OpDesc>(roi_node->GetName() + "_balancerois",
                                                                              "BalanceRois");
  FUSION_PASS_CHECK(
      balancerois_desc == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "balancerois_desc is null, Build BalanceRois Op failed."),
      return FAILED);
  GeTensorDesc rois_data_desc = roi_node->GetOpDesc()->GetInputDesc("rois");

  FUSION_PASS_CHECK(roi_node->GetOpDesc()->GetInputDesc("index").GetShape().GetDim(0) != IndexDim,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "index dim is not 0."),
                    return FAILED);
  FUSION_PASS_CHECK(balancerois_desc->AddInputDesc("rois", rois_data_desc) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "BalanceRois add rois_data desc failed."),
                    return FAILED);

  ge::GeTensorDesc index_desc;
  auto rois_shape = rois_data_desc.GetShape();
  vector<int64_t> index_shape;
  index_shape.push_back(rois_shape.GetDim(0));
  index_desc.SetShape(ge::GeShape(index_shape));
  index_desc.SetFormat(ge::FORMAT_ND);
  index_desc.SetDataType(ge::DT_INT32);

  FUSION_PASS_CHECK(balancerois_desc->AddOutputDesc("balance_rois", rois_data_desc) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "BalanceRois add balance_rois_data desc failed."),
                    return FAILED);
  FUSION_PASS_CHECK(balancerois_desc->AddOutputDesc("index", index_desc) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "BalanceRois add index desc failed."),
                    return FAILED);

  NodePtr BalanceRoisNode = graph.AddNode(balancerois_desc);
  FUSION_PASS_CHECK(
      BalanceRoisNode == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "The BalanceRoisNode is null, Build BalanceRois Op failed."),
      return PARAM_INVALID);

  fusion_nodes.push_back(BalanceRoisNode);

  ge::OpDescPtr fpnroiextractor_desc = AttrUtils::CopyOpDesc(roi_desc);
  fpnroiextractor_desc->SetName(roi_node->GetName() + "_fpnroiextractor");
  fpnroiextractor_desc->SetType("RoiExtractor");
  FUSION_PASS_CHECK(fpnroiextractor_desc->AddInputDesc("index", index_desc) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "FPNRoiExtractor add index_data desc failed."),
                    return FAILED);
  NodePtr FPNRoiExtractorNode = graph.AddNode(fpnroiextractor_desc);
  fusion_nodes.push_back(FPNRoiExtractorNode);

  FUSION_PASS_CHECK(
      GraphUtils::AddEdge(roi_node->GetInDataAnchor(4)->GetPeerOutAnchor(),
                          BalanceRoisNode->GetInDataAnchor(0)) != GRAPH_SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add rois_data node to BalanceRoisNode node edge failed."),
      return FAILED);

  for (unsigned int i = 0; i < FeatureNum; ++i) {
      FUSION_PASS_CHECK(
            GraphUtils::AddEdge(roi_node->GetInDataAnchor(i)->GetPeerOutAnchor(), FPNRoiExtractorNode->GetInDataAnchor(i)) != GRAPH_SUCCESS,
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add fm_data to FPNRoiExtractorNode edge failed."),
            return FAILED);
  }

  FUSION_PASS_CHECK(
        GraphUtils::AddEdge(BalanceRoisNode->GetOutDataAnchor(0), FPNRoiExtractorNode->GetInDataAnchor(4)) != GRAPH_SUCCESS,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add balance_roi to FPNRoiExtractorNode edge failed."),
        return FAILED);

  FUSION_PASS_CHECK(
        GraphUtils::AddEdge(BalanceRoisNode->GetOutDataAnchor(1), FPNRoiExtractorNode->GetInDataAnchor(5)) != GRAPH_SUCCESS,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add index to FPNRoiExtractorNode edge failed."),
        return FAILED);

  for (auto inDataAnchor : roi_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(
        GraphUtils::RemoveEdge(roi_node->GetOutDataAnchor(0), inDataAnchor) != GRAPH_SUCCESS,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."),
        return FAILED);
    FUSION_PASS_CHECK(
        GraphUtils::AddEdge(FPNRoiExtractorNode->GetOutDataAnchor(0), inDataAnchor) != GRAPH_SUCCESS,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                       "Add last output data to next node edge failed."),
        return FAILED);
  }

  FUSION_PASS_CHECK(graph.RemoveNode(roi_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove roiextractor node failed."),
                    return FAILED);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "add BalanceRois and FPNRoiExtractor end.");
  return SUCCESS;
}
REGISTER_PASS("RoiExtractorFusionPass", BUILT_IN_GRAPH_PASS, RoiExtractorFusionPass);
} // namespace fe