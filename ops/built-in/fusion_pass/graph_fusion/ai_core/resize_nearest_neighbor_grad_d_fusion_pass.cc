/**
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief ResizeNearestNeighborV2Grad fusion pass
 * ResizeNearestNeighborV2Grad --> ResizeNearestNeighborV2GradD
 *
 */

#include "resize_nearest_neighbor_grad_d_fusion_pass.h"
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
static const char *FUSED_NODE = "ResizeNearestNeighborV2Grad";
static const std::string PATTERN_FUSEDNODE = "FusedNodeResizeNearestNeighborGrad";

vector<FusionPattern*> ConstToAttrResizeNearestNeighborGradPass::DefinePatterns() {
  vector < FusionPattern * > patterns;
  FusionPattern* pattern = new(std::nothrow) FusionPattern("ConstToAttrResizeNearestNeighborGradFusion");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
       return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE})
      .SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

Status ConstToAttrResizeNearestNeighborGradPass::Fusion(ge::ComputeGraph& graph,
                                                        Mapping& mapping,
                                                        vector<ge::NodePtr> &fusionNodes) {
  // get fused node
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."), return PARAM_INVALID);

  //build attr infos
  std::string fusionOpType = "ResizeNearestNeighborV2GradD";
  std::vector<PassAttrInfo> attrInfos;
  PassAttrInfo size = {1, "size", "SetListInt"};
  attrInfos.push_back(size);

  // build a fusion node op desc
  ge::OpDescPtr fusionDescPtr = PatternFusionUtil::GetFusionOpDesc(fusedNode, fusionOpType, attrInfos);
  FUSION_PASS_CHECK(fusionDescPtr == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Fusion OP Desc is nullptr."),
           return PARAM_INVALID);

  // check op support
  FUSION_PASS_CHECK(!CheckOpSupported(fusionDescPtr), OP_LOGI(FUSED_OP_TYPE.c_str(), "Op ResizeNearestNeighborGrad Not Supported."),
           return NOT_CHANGED);

  // const to attr
  ge::NodePtr fusionNode = nullptr;
  Status ret = PatternFusionUtil::ConstToAttrWithNode(graph, fusedNode, fusionOpType, attrInfos, fusionNode);
  if (ret != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "ResizeNearestNeighborGrad has input which is not a constant, graph not changed.");
    return NOT_CHANGED;
  }
  fusionNodes.push_back(fusedNode);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "ResizeNearestNeighborGradD fusion SUCCESSS!!!!!");
  return SUCCESS;
}
REGISTER_PASS("ConstToAttrResizeNearestNeighborGradFusion",
              BUILT_IN_GRAPH_PASS,
              ConstToAttrResizeNearestNeighborGradPass);
}

