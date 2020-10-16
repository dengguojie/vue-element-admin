/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief split fusion pass(conv3d_backprop_input --> conv3d_backprop_input_d)
 *
 */

#include "conv3d_backprop_filter_fusion_pass.h"

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
static const char* CONV3DBACKPROPFILTER = "Conv3DBackpropFilter";
static const std::string PATTERN_CONV3DBACKPROPFILTER = "Conv3DBackpropFilter";

vector<FusionPattern*> ConstToAttrConv3dBackpropFilterPass::DefinePatterns() {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter ConstToAttrConv3dBackpropFilterPass::DefinePatterns.");
    vector < FusionPattern * > patterns;
    FusionPattern* pattern = new(std::nothrow) FusionPattern("ConstToAttrConv3dBackpropFilterFusion");
    FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
           return patterns);

    pattern->AddOpDesc(PATTERN_CONV3DBACKPROPFILTER, {CONV3DBACKPROPFILTER})
          .SetOutput(PATTERN_CONV3DBACKPROPFILTER);

    patterns.push_back(pattern);

    return patterns;
}

Status ConstToAttrConv3dBackpropFilterPass::Fusion(ge::ComputeGraph& graph,
                                                  Mapping& mapping,
                                                  vector<ge::NodePtr> &newNodes) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter ConstToAttrConv3dBackpropFilterPass::Fusion.");
    std::string Conv3DBackpropFilterD = "Conv3DBackpropFilterD";
    std::vector<PassAttrInfo> conv3dBackpropFilterAttrInfo;
    PassAttrInfo filter_size = {1, "filter_size", "SetListInt"};
    conv3dBackpropFilterAttrInfo.push_back(filter_size);
    ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_CONV3DBACKPROPFILTER, mapping);
    FUSION_PASS_CHECK(fusedNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."), return PARAM_INVALID);
    Status ret = PatternFusionUtil::ConstToAttrWithType(graph, fusedNode, Conv3DBackpropFilterD, conv3dBackpropFilterAttrInfo);
    if (ret != SUCCESS) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv3DBackpropFilter has input which is not a constant, graph not changed.");
        return NOT_CHANGED;
    }
    return SUCCESS;
}

REGISTER_PASS("ConstToAttrConv3dBackpropFilterPass",
              BUILT_IN_GRAPH_PASS,
              ConstToAttrConv3dBackpropFilterPass);
}
