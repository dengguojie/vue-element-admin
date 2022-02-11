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
 * \file strided_slice_fusion_pass.cpp
 * \brief strided_slice fusion pass(strided_slice --> null)
 */
#include "strided_slice_remove_fusion_pass.h"

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
namespace fe {
static const char *FUSED_NODE_1 = "StridedSlice";
static const char *FUSED_NODE_2 = "StridedSliceD";
static const std::string PATTERN_FUSEDNODE = "StridedSlice";

vector<FusionPattern *> StridedSliceRemovePass::DefinePatterns() {
    vector<FusionPattern *> patterns;

    FusionPattern *pattern = new (std::nothrow) FusionPattern("StridedSliceRemovePass");
    FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "new a pattern object failed."), return patterns);
    pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE_1, FUSED_NODE_2})
        .SetOutput(PATTERN_FUSEDNODE);
    patterns.push_back(pattern);

    return patterns;
}

Status StridedSliceRemovePass::ReLinkControlAnchor(ge::NodePtr stridedSliceNode, ge::NodePtr nextNode) {
    InControlAnchorPtr node1InControlAnchorPtr = stridedSliceNode->GetInControlAnchor();
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

Status StridedSliceRemovePass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusionNodes) {
    ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
    FUSION_PASS_CHECK(fusedNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "fusedNode is null, fusion failed."),  return PARAM_INVALID);
    ge::OpDescPtr fuseDesc = fusedNode->GetOpDesc();
    FUSION_PASS_CHECK(fuseDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "fused_node's OpDesc is null, fusion failed."), return PARAM_INVALID);

    if (HasUnKnowShape(fusedNode)) {
        OP_LOGD(FUSED_OP_TYPE.c_str(), "StridedSliceRemovePass cannot be applied for unknow shape.");
        return NOT_CHANGED;
    }
    if (fusedNode->GetOutDataNodes().size() > 1) {
        OP_LOGD(FUSED_OP_TYPE.c_str(), "node %s have more than 1 data nodes.", fusedNode->GetName().c_str());
        return NOT_CHANGED;
    }
    vector<int64_t> inputDims = fuseDesc->GetInputDesc(0).GetShape().GetDims();
    vector<int64_t> outputDims = fuseDesc->GetOutputDesc(0).GetShape().GetDims();
    if (inputDims != outputDims) {
        OP_LOGD(FUSED_OP_TYPE.c_str(), "StrideSlice inputDims is not equal outputDims.");
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
                OP_LOGD(FUSED_OP_TYPE.c_str(), "process %s control link failed", fusedNode->GetName().c_str());
                return FAILED;
            }
        }
    }

    // remove castNode1 from graph
    if (ge::GRAPH_SUCCESS != graph.RemoveNode(fusedNode)) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "remove StridedSlice node failed");
        return FAILED;
    }
    return SUCCESS;
}

REGISTER_PASS("StridedSliceRemovePass", BUILT_IN_GRAPH_PASS, StridedSliceRemovePass);
} // namespace fe