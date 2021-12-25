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
 * \file cast_cast_fusion_pass.cc
 * \brief cast cast fusion pass(Cast+Cast --> Cast)
 */

#include "cast_cast_fusion_pass.h"

#include <memory>
#include <string>
#include <vector>

#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph/utils/graph_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "error_util.h"

namespace fe {
static const char* CAST = "Cast";
static const string PATTERN_CAST1 = "cast1";
static const string PATTERN_CAST2 = "cast2";

vector<FusionPattern *> CastCastFusionPass::DefinePatterns() {
    vector<FusionPattern*> patterns;
    FusionPattern *pattern = new (std::nothrow) FusionPattern("CastCastFusionPass");
    if (pattern == nullptr) {
        OP_LOGW(FUSED_OP_TYPE.c_str(), "pattern is nullptr,Create pattern failed.");
        return patterns;
    }
    pattern->AddOpDesc(PATTERN_CAST1, { CAST })
        .AddOpDesc(PATTERN_CAST2, { CAST })
        .SetInputs(PATTERN_CAST2, {PATTERN_CAST1})
        .SetOutput(PATTERN_CAST2);
    patterns.push_back(pattern);
    return patterns;
}

Status CastCastFusionPass::IsMatch(ge::NodePtr castNode1, ge::NodePtr castNode2) const {
    std::shared_ptr<ge::OpDesc> cast1Op = castNode1->GetOpDesc();
    std::shared_ptr<ge::OpDesc> cast2Op = castNode2->GetOpDesc();
    FUSION_PASS_CHECK(cast1Op == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "cast1Op is null"), return FAILED);
    FUSION_PASS_CHECK(cast2Op == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "cast2Op is null"), return FAILED);

    ge::GeTensorDesc cast1InputTensor = cast1Op->GetInputDesc(0);
    ge::GeTensorDesc cast1OutputTensor = cast1Op->GetOutputDesc(0);
    ge::GeTensorDesc cast2OutputTensor = cast2Op->GetOutputDesc(0);
    DataType cast1InputDataType = cast1InputTensor.GetDataType();
    DataType cast1OutputDataType = cast1OutputTensor.GetDataType();
    DataType cast2OutputDataType = cast2OutputTensor.GetDataType();
    if (cast1OutputDataType == ge::DT_BOOL) {
        OP_LOGD(FUSED_OP_TYPE.c_str(), "Cast1 output type is bool, can not do fusion.");
        return FAILED;
    }
    if (castNode1->GetOutDataNodes().size() > 1) {
        OP_LOGD(FUSED_OP_TYPE.c_str(), "Cast1 out data node size is more than 1");
        return FAILED;
    }
    if ((cast1InputDataType == ge::DT_FLOAT16 && cast2OutputDataType == ge::DT_INT32) ||
        (cast1InputDataType == ge::DT_INT32 && cast2OutputDataType == ge::DT_FLOAT16)) {
        OP_LOGD(FUSED_OP_TYPE.c_str(), "Cast1 input type is u% Cast2 output data type is u%",
                cast1InputDataType, cast2OutputDataType);
        return SUCCESS;
    }
    return FAILED;
}

Status CastCastFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusionNodes) {
    ge::NodePtr castNode1 = GetNodeFromMapping(PATTERN_CAST1, mapping);
    ge::NodePtr castNode2 = GetNodeFromMapping(PATTERN_CAST2, mapping);
    FUSION_PASS_CHECK(castNode1 == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "cast1 node is null"), return FAILED);
    FUSION_PASS_CHECK(castNode2 == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "cast2 node is null"), return FAILED);

    if (IsMatch(castNode1, castNode2) != SUCCESS) {
        OP_LOGD(FUSED_OP_TYPE.c_str(), "Node[s%] and node[s%] don't match Cast + Cast fusion pattern.",
                castNode1->GetName().c_str(), castNode2->GetName().c_str());
        return NOT_CHANGED;
    }
    if (ReLinkControlAnchor(castNode1, castNode2) != SUCCESS) {
        OP_LOGD(FUSED_OP_TYPE.c_str(), "process %s and %s control link failed", castNode1->GetName().c_str(), castNode2->GetName().c_str());
        return FAILED;
    }
    ge::GeTensorDesc cast1InputDesc = castNode1->GetOpDesc()->GetInputDesc(0).Clone();
    ge::OpDescPtr cast2Desc = castNode2->GetOpDesc();
    cast2Desc->UpdateInputDesc("x", cast1InputDesc);

    // remove castNode1 from graph
    if (ge::GRAPH_SUCCESS != graph.RemoveNode(castNode1)) {
        OP_LOGD(FUSED_OP_TYPE.c_str(), "remove cast node failed");
        return FAILED;
    }

    fusionNodes.push_back(castNode2);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Node[%s] do cast + cast fusion success!", castNode1->GetName().c_str());
    return SUCCESS;
}

Status CastCastFusionPass::ReLinkControlAnchor(ge::NodePtr castNode1, ge::NodePtr castNode2) {
    InControlAnchorPtr cast1InControlAnchorPtr = castNode1->GetInControlAnchor();
    InControlAnchorPtr cast2InControlAnchorPtr = castNode2->GetInControlAnchor();
    if (cast1InControlAnchorPtr != nullptr && cast2InControlAnchorPtr != nullptr) {
        for (OutControlAnchorPtr outControlAnchorPtr : cast1InControlAnchorPtr->GetPeerOutControlAnchors()) {
            FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(outControlAnchorPtr, cast1InControlAnchorPtr),
                VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove input control edge failed"), return FAILED);
            FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(outControlAnchorPtr, cast2InControlAnchorPtr),
                VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input control edge failed"), return FAILED);
        }
    }
    return SUCCESS;
}

REGISTER_PASS("CastCastFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, CastCastFusionPass);
}