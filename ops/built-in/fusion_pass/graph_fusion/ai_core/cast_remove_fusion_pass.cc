/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
 * \file cast_remove_fusion_pass.cpp
 * \brief cast fusion pass(cast --> null)
 */
#include "cast_remove_fusion_pass.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "error_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "tbe_fusion_pass_util.h"
#include "tbe_ops_pass_util.h"

using namespace ge;
namespace fe
{

static const char *FUSED_NODE = "Cast";
static const std::string PATTERN_FUSEDNODE = "Cast";

vector<FusionPattern *> CastRemoveFusionPass::DefinePatterns() {
    vector<FusionPattern *> patterns;

    FusionPattern *pattern = new (std::nothrow) FusionPattern("CastRemoveFusionPass");
    FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "new a pattern object failed."), return patterns);
    pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE})
        .SetOutput(PATTERN_FUSEDNODE);
    patterns.push_back(pattern);

    return patterns;
}

Status CastRemoveFusionPass::ReLinkControlAnchor(ge::NodePtr castNode, ge::NodePtr nextNode) {
    InControlAnchorPtr node1InControlAnchorPtr = castNode->GetInControlAnchor();
    InControlAnchorPtr node2InControlAnchorPtr = nextNode->GetInControlAnchor();
    if (node1InControlAnchorPtr == nullptr || node2InControlAnchorPtr == nullptr) {
        return SUCCESS;
    }
    for (OutControlAnchorPtr outControlAnchorPtr : node1InControlAnchorPtr->GetPeerOutControlAnchors()) {
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(outControlAnchorPtr, node1InControlAnchorPtr),
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove input control edge failed"),
                          return FAILED);
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(outControlAnchorPtr, node2InControlAnchorPtr),
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input control edge failed"),
                          return FAILED);
    }
    return SUCCESS;
}

Status CastRemoveFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                                      vector<ge::NodePtr> &fusionNodes) {
    // PatternFusionUtil patternFusionUtil;
    ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
    FUSION_PASS_CHECK(fusedNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "fusedNode is null, fusion failed."), return PARAM_INVALID);
    ge::OpDescPtr fuseDesc = fusedNode->GetOpDesc();
    FUSION_PASS_CHECK(fuseDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "fused_node's OpDesc is null, fusion failed."), return PARAM_INVALID);

    DataType castInputDataType = fuseDesc->GetInputDesc(0).GetDataType();
    DataType castOutputDataType = fuseDesc->GetOutputDesc(0).GetDataType();

    vector<int64_t> outputDims = fuseDesc->GetOutputDesc(0).GetShape().GetDims();
    if (castInputDataType != castOutputDataType) {
        OP_LOGD(FUSED_OP_TYPE.c_str(), "cast input dataType is not equal cast output type.");
        return NOT_CHANGED;
    }
    // if exist control anchor, need relink to next node.
    for (auto outAnchor : fusedNode->GetAllOutDataAnchors()) {
        for (InDataAnchorPtr inAnchorPtr : outAnchor->GetPeerInDataAnchors()) {
            ge::NodePtr nextNode = inAnchorPtr->GetOwnerNode();
            if (nextNode == nullptr) {
                continue;
            }
            if (ReLinkControlAnchor(fusedNode, nextNode) != SUCCESS) {
                OP_LOGD(FUSED_OP_TYPE.c_str(), "process control link failed");
                return FAILED;
            }
        }
    }

    // remove castNode1 from graph
    if (ge::GRAPH_SUCCESS != graph.RemoveNode(fusedNode)) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "remove Cast node failed");
        return FAILED;
    }
    return SUCCESS;
}

REGISTER_PASS("CastRemoveFusionPass", BUILT_IN_GRAPH_PASS, CastRemoveFusionPass);
} // namespace fe