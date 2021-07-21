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
 * \file z_unsortedsegmentsum_update_fusion_pass.cc
 * \brief z_unsortedsegmentsum update fusion pass
 */
#include "z_unsortedsegmentsum_update_fusion_pass.h"
#include "op_log.h"
#include "error_util.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace std;
using namespace ge;

namespace fe {
vector<FusionPattern*> ZUnsortedSegmentSumUpdateFusionPass::DefinePatterns() {
    vector<FusionPattern*> patterns;
    FusionPattern* pattern = new (std::nothrow) FusionPattern("UnsortedSegmentSumUpdatePattern");
    FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                      return patterns);
    pattern->AddOpDesc("UnsortedSegmentSumD", {"UnsortedSegmentSumD"}).SetOutput("UnsortedSegmentSumD");
    patterns.push_back(pattern);
    return patterns;
}

// to change unsortedsegmentsumd node to unsortedsegmentsum node when check supported
// partly realize the static2dynamic process of unsortedsegmentsum
Status ZUnsortedSegmentSumUpdateFusionPass::Fusion(ge::ComputeGraph& graph,
                                          Mapping& mapping,
                                          vector<ge::NodePtr>& fusionNodes) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "ZUnsortedSegmentSumUpdateFusionPass is running.");
    ge::NodePtr unsortedSegmentSumdNode = GetNodeFromMapping("UnsortedSegmentSumD", mapping);
    if (unsortedSegmentSumdNode == nullptr) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "unsortedsegmentsumd not exist");
        return NOT_CHANGED;
    }

    ge::OpDescPtr unsortedSegmentSumdOpDesc = unsortedSegmentSumdNode->GetOpDesc();
    if (unsortedSegmentSumdOpDesc == nullptr) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Failed to get op desc");
        return FAILED;
    }

    unsortedSegmentSumdOpDesc->SetType("UnsortedSegmentSum");
    int32_t num_segments;
    std::vector<int64_t> num_segments01 = {1};
    ge::AttrUtils::GetInt(unsortedSegmentSumdOpDesc, "num_segments", num_segments);
    unsortedSegmentSumdOpDesc->DelAttr("num_segments");
    ge::GeShape constInsegment = ge::GeShape(num_segments01);
    auto segmentInputDesc = ge::GeTensorDesc(constInsegment, ge::FORMAT_ND, ge::DT_INT32);
    ge::GeTensorPtr outTensor = std::make_shared<ge::GeTensor>(segmentInputDesc);
    vector<int32_t> permB32 = {static_cast<int32_t>(num_segments)};

    outTensor->SetData(reinterpret_cast<uint8_t *>(permB32.data()), permB32.size() * sizeof(int32_t));
    ge::OpDescPtr outOpDesc = ge::OpDescUtils::CreateConstOp(outTensor);
    auto constNode = graph.AddNode(outOpDesc);
    if (unsortedSegmentSumdNode->AddLinkFrom("num_segments", constNode) != SUCCESS) {
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to AddEdge");
        return FAILED;
    }

    // empty the format/dtype info. in opdesc.
    const std::map<std::string, vector<ge::Format>> format_map;
    const std::map<std::string, vector<ge::DataType>> data_type_map;
    unsortedSegmentSumdOpDesc->SetExtAttr("ext_dynamic_format", format_map);
    unsortedSegmentSumdOpDesc->SetExtAttr("ext_dynamic_datatype", data_type_map);

    bool isSegmentSupported = CheckOpSupported(unsortedSegmentSumdOpDesc);

    OP_LOGD(FUSED_OP_TYPE.c_str(), "is_segment_supported=%d.", isSegmentSupported);

    if (!isSegmentSupported) {
        auto anchor = unsortedSegmentSumdNode->GetInDataAnchor(2);
        anchor->UnlinkAll();
        if (graph.RemoveNode(constNode) != SUCCESS) {
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove const node");
            return FAILED;
        }
        if (ge::NodeUtils::RemoveInputAnchor(unsortedSegmentSumdNode, 2) != SUCCESS) {
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Fail to remove input anchor");
            return FAILED;
        }
        ge::AttrUtils::SetInt(unsortedSegmentSumdOpDesc, "num_segments", num_segments);
        unsortedSegmentSumdOpDesc->SetType("UnsortedSegmentSumD");
    }

    OP_LOGD(FUSED_OP_TYPE.c_str(), "ZUnsortedSegmentSumUpdateFusionPass run success.");
    return SUCCESS;
}

REGISTER_PASS("ZUnsortedSegmentSumUpdateFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS,
              ZUnsortedSegmentSumUpdateFusionPass);
}  // namespace fe
