/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file partitionedcall_fusion_pass.cc
 * \brief partitionedcall fusion pass
 */

#include "partitionedcall_fusion_pass.h"

#include <memory>
#include <string>
#include <vector>

#include "error_util.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "graph_optimizer/fusion_common/fusion_turbo.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

namespace fe {
static const char *PARTITIONEDCALL = "PartitionedCall";
static const string TRANSDATA = "TransData";
static const string PATTERN_PARTITIONEDCALL = "PartitionedCall";
static const string PATTERN_TRANSDATA = "TransData";
static const string NETOUTPUT = "NetOutput";
static const string BATCHMATMULV2 = "BatchMatMulV2";
static const string MATMULV2 = "MatMulV2";

vector<FusionPattern *> PartitionedCallFusionPass::DefinePatterns() {
    vector<FusionPattern *> patterns;
    FusionPattern *pattern = new (std::nothrow) FusionPattern("PartitionedCallFusionPass");
    if (pattern == nullptr) {
        OP_LOGW(FUSED_OP_TYPE.c_str(), "pattern is nullptr, Create pattern failed.");
        return patterns;
    }
    pattern->AddOpDesc(PATTERN_PARTITIONEDCALL, { PARTITIONEDCALL })
        .AddOpDesc(PATTERN_TRANSDATA, { TRANSDATA })
        .SetInputs(PATTERN_TRANSDATA, { PATTERN_PARTITIONEDCALL })
        .SetOutput(PATTERN_TRANSDATA);
    patterns.push_back(pattern);
    return patterns;
}

Status PartitionedCallFusionPass::IsMatch(ge::NodePtr &partitionedCallNode, ge::NodePtr &transDataNode) const {
    FUSION_PASS_CHECK(partitionedCallNode->GetOutDataNodes().size() > 1,
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "PartitionedCall Out Nodes is more than 1, skip fusion."),
                      return NOT_CHANGED);
    ge::OpDescPtr partitionedCallDesc = partitionedCallNode->GetOpDesc();
    ge::OpDescPtr transDataDesc = transDataNode->GetOpDesc();
    FUSION_PASS_CHECK(partitionedCallDesc == nullptr, OP_LOGD(FUSED_OP_TYPE.c_str(), "partitionedCallDesc is null"),
                      return PARAM_INVALID);
    FUSION_PASS_CHECK(transDataDesc == nullptr, OP_LOGD(FUSED_OP_TYPE.c_str(), "transDataDesc is null"),
                      return PARAM_INVALID);

    ge::GeTensorDesc transDataInputDesc = transDataDesc->GetInputDesc(0);
    ge::GeTensorDesc transDataOutputDesc = transDataDesc->GetOutputDesc(0);
    FUSION_PASS_CHECK(
        transDataOutputDesc.GetFormat() != ge::FORMAT_ND ||
        ge::GetPrimaryFormat(transDataInputDesc.GetFormat()) != ge::FORMAT_FRACTAL_NZ,
        OP_LOGD(FUSED_OP_TYPE.c_str(),
                "For TransData node, input format should be FRACTAL_NZ, and output fromat should be ND"),
        return NOT_CHANGED);
    // get PartitionedCall GetSubgraph
    auto partitionCallSubgraph = ge::NodeUtils::GetSubgraph(*partitionedCallNode, 0);
    FUSION_PASS_CHECK(partitionCallSubgraph == nullptr,
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "PartitionedCall SubGraph is null"),
                      return NOT_CHANGED);
    ge::NodePtr subGraphOutputNode = partitionCallSubgraph->FindFirstNodeMatchType(NETOUTPUT);
    FUSION_PASS_CHECK(subGraphOutputNode == nullptr,
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "NetOutput Node is null"), return NOT_CHANGED);
    FUSION_PASS_CHECK(subGraphOutputNode->GetInDataNodes().size() != 1,
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "NetOutput input nodes is more than 1, skip fusion."),
                      return NOT_CHANGED);
    ge::NodePtr netOuptInNode = subGraphOutputNode->GetInDataNodes().at(0);
    FUSION_PASS_CHECK(netOuptInNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "NetOutput input Node is null"),
                      return PARAM_INVALID);
    if (netOuptInNode->GetType() != BATCHMATMULV2 && netOuptInNode->GetType() != MATMULV2) {
        OP_LOGD(FUSED_OP_TYPE.c_str(), "NetOutput input Node is not BatchMatMulV2 or MatMulV2, skip fusion.");
        return NOT_CHANGED;
    }
    OP_LOGD(FUSED_OP_TYPE.c_str(), "fusion Match success.");
    return SUCCESS;
}

Status PartitionedCallFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusionNodes) {
    ge::NodePtr partitionedCallNode = GetNodeFromMapping(PATTERN_PARTITIONEDCALL, mapping);
    ge::NodePtr transDataNode = GetNodeFromMapping(PATTERN_TRANSDATA, mapping);
    FUSION_PASS_CHECK(partitionedCallNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "PartitionedCall node is null"),
                      return PARAM_INVALID);
    FUSION_PASS_CHECK(transDataNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "TransData node is null"),
                      return PARAM_INVALID);

    if (IsMatch(partitionedCallNode, transDataNode) != SUCCESS) {
        OP_LOGD(FUSED_OP_TYPE.c_str(), "Node[s%] and node[s%] don't match Cast + Cast fusion pattern.",
                partitionedCallNode->GetName().c_str(), transDataNode->GetName().c_str());
        return NOT_CHANGED;
    }
    // call the FE API
    FusionTurbo ft(graph);
    if (ft.GraphNodeUpMigration(transDataNode, 0) != SUCCESS) {
        OP_LOGD(FUSED_OP_TYPE.c_str(), "Node[%s] call GraphNodeForwardMigration not success!",
                transDataNode->GetName().c_str());
        return NOT_CHANGED;
    }
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Node[%s] PartitionedCallFusionPass fusion success!",
            partitionedCallNode->GetName().c_str());
    return SUCCESS;
}

REGISTER_PASS("PartitionedCallFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, PartitionedCallFusionPass);
} // namespace fe