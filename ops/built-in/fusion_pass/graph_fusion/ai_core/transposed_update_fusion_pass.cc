/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file transposed_update_fusion_pass.cc
 * \brief transposed update fusion pass
 */
#include "transposed_update_fusion_pass.h"
#include "op_log.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace std;
using namespace ge;

namespace fe {
vector<FusionPattern*> TransposedUpdateFusionPass::DefinePatterns() {
    vector<FusionPattern*> patterns;
    FusionPattern* pattern = new (std::nothrow) FusionPattern("TransposedUpdatePattern");
    FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                      return patterns);
    pattern->AddOpDesc("TransposeD", {"TransposeD"}).SetOutput("TransposeD");
    patterns.push_back(pattern);
    return patterns;
}

Status TransposedUpdateFusionPass::Fusion(ge::ComputeGraph& graph,
                                          Mapping& mapping,
                                          vector<ge::NodePtr>& fusionNodes) {
    ge::NodePtr transposeNode = GetNodeFromMapping("TransposeD", mapping);
    if (transposeNode == nullptr) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Transpose not exist");
        return NOT_CHANGED;
    }

    ge::OpDescPtr transposeOpDesc = transposeNode->GetOpDesc();
    if (transposeOpDesc == nullptr) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Failed to get op desc");
        return FAILED;
    }

    transposeOpDesc->SetType("Transpose");
    vector<string> permVec;
    permVec.push_back("perm");
    transposeOpDesc->SetOpInferDepends(permVec);
    std::vector<int64_t> perm;
    ge::AttrUtils::GetListInt(transposeOpDesc, "perm", perm);
    transposeOpDesc->DelAttr("perm");
    vector<int64_t> permShape;
    permShape.push_back(perm.size());
    ge::GeShape constInPerm = ge::GeShape(permShape);
    auto permInputDesc = ge::GeTensorDesc(constInPerm, ge::FORMAT_ND, ge::DT_INT32);
    ge::GeTensorPtr outTensor = std::make_shared<ge::GeTensor>(permInputDesc);
    vector<int32_t> permB32;
    for (auto ele : perm){
        permB32.push_back(static_cast<int32_t>(ele));
    }
    outTensor->SetData(reinterpret_cast<uint8_t *>(permB32.data()), permB32.size() * sizeof(int32_t));
    ge::OpDescPtr outOpDesc = ge::OpDescUtils::CreateConstOp(outTensor);
    auto constNode = graph.AddNode(outOpDesc);
    if (transposeNode->AddLinkFrom("perm", constNode) != SUCCESS) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to AddEdge");
        return FAILED;
    }

    bool isTransposSupported = CheckOpSupported(transposeNode);

    if (!isTransposSupported) {
        auto anchor = transposeNode->GetInDataAnchor(1);
        anchor->UnlinkAll();
        if (graph.RemoveNode(constNode) != SUCCESS) {
            OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to remove const node");
            return FAILED;
        }
        if (ge::NodeUtils::RemoveInputAnchor(transposeNode, 1) != SUCCESS) {
            OP_LOGE(FUSED_OP_TYPE.c_str(), "Fail to remove input anchor");
            return FAILED;
        }
        ge::AttrUtils::SetListInt(transposeOpDesc, "perm", perm);
        permVec.clear();
        transposeOpDesc->SetOpInferDepends(permVec);
        transposeOpDesc->SetType("TransposeD");
    }
    return SUCCESS;
}

REGISTER_PASS("TransposedUpdateFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, TransposedUpdateFusionPass);
}  // namespace fe

