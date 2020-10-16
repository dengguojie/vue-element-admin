/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief fusion pass(Add a pad op before NMSWithMask for input of box_scores)
 *
 */

#include "nms_with_mask_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "op_log.h"

using namespace ge;
namespace fe {
static const std::string PATTERN_NMS_WITH_MASK = "NMSWithMask";
static const char *NMS_WITH_MASK = "NMSWithMask";

vector<FusionPattern*> NMSWithMaskFusionPass::DefinePatterns() {
    vector <FusionPattern*> patterns;

    FusionPattern* pattern = new(std::nothrow) FusionPattern("NMSWithMaskFusion");
    FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
             return patterns);

    pattern->AddOpDesc(PATTERN_NMS_WITH_MASK, {NMS_WITH_MASK})
            .SetOutput(PATTERN_NMS_WITH_MASK);

    patterns.push_back(pattern);

    return patterns;
}

Status NMSWithMaskFusionPass::Fusion(ge::ComputeGraph &graph,
                                     Mapping &mapping,
                                     vector<ge::NodePtr> &fusionNodes)
{
    ge::NodePtr nmsNodePtr = GetNodeFromMapping(PATTERN_NMS_WITH_MASK, mapping);
    FUSION_PASS_CHECK(nmsNodePtr == nullptr,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "NMSWithMask Node is null, fusion failed."),
             return PARAM_INVALID);
    ge::OpDescPtr nmsDescPtr = nmsNodePtr->GetOpDesc();
    FUSION_PASS_CHECK(nmsNodePtr == nullptr,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "NMSWithMask desc is null, FE fusion failed."),
             return PARAM_INVALID);

    // clone pad node desc from nms
    ge::OpDescPtr padDescPtr = AttrUtils::CloneOpDesc(nmsDescPtr);
    FUSION_PASS_CHECK(padDescPtr == nullptr,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Create PadD OpDesc failed, fusion failed."),
             return PARAM_INVALID);
    padDescPtr->SetType("PadD");
    padDescPtr->SetName(padDescPtr->GetName() + "_PadD");

    // delete output desc of pad node
    int tmpOutputSize = padDescPtr->GetOutputsSize();
    if (tmpOutputSize < 1) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "The output of %s is zero", nmsNodePtr->GetName().c_str());
        return FAILED;
    }
    while (tmpOutputSize > 0) {
        tmpOutputSize--;
        OpDescUtils::ClearOutputDesc(padDescPtr, tmpOutputSize);
    }

    // add the output edge of pad node, and update the info of pad
    int nmsInputSize = nmsDescPtr->GetInputsSize();
    if (nmsInputSize < 1) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "The input of %s is zero", nmsNodePtr->GetName().c_str());
        return FAILED;
    }
    ge::GeTensorDesc padOutputTensorDesc = nmsDescPtr->GetInputDesc(0);

    // update pad node info
    auto nmsInputDims = padOutputTensorDesc.GetShape().GetDims();
    if (!(nmsInputDims.size() == 2 && nmsInputDims[1] == 5)) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "The input dim of %s is not 2 dims or the second dimension of \
                 input is not 5", nmsNodePtr->GetName().c_str());
        return FAILED;
    }
    // set pad output shape
    nmsInputDims[1] = 8;
    padOutputTensorDesc.SetShape(ge::GeShape(nmsInputDims));

    // update output origin shape of pad
    padOutputTensorDesc.SetOriginShape(ge::GeShape(nmsInputDims));
    padDescPtr->AddOutputDesc("NMSWithMask", padOutputTensorDesc);
    padDescPtr->UpdateOutputDesc(0, padOutputTensorDesc);
    nmsDescPtr->UpdateInputDesc(0, padOutputTensorDesc);

    // delete attr from nms
    FUSION_PASS_CHECK(SUCCESS != padDescPtr->DelAttr("iou_threshold"),
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Delete the attr of iou_threshold from nms."),
             return PARAM_INVALID);

    // set paddings attr for pad node
    FUSION_PASS_CHECK(true != ge::AttrUtils::SetListListInt(padDescPtr, "paddings",
             std::vector<std::vector<int64_t>>{{0, 0}, {0, 3}}),
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Set paddings attr for pad node."), return PARAM_INVALID);

    // add pad node to graph
    ge::NodePtr padNodePtr = graph.AddNode(padDescPtr);
    fusionNodes.push_back(padNodePtr);
    FUSION_PASS_CHECK(padNodePtr == nullptr,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode: padNodePtr is null, fusion failed."),
             return FAILED);

    // add the original edge of nms to pad
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(
             nmsNodePtr->GetInDataAnchor(0)->GetPeerOutAnchor(),
             padNodePtr->GetInDataAnchor(0)),
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s to fusion node:%s failed.",
             nmsNodePtr->GetName().c_str(), nmsNodePtr->GetName().c_str()),
             return FAILED);

    // delete the first edge of nms
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(
             nmsNodePtr->GetInDataAnchor(0)->GetPeerOutAnchor(),
             nmsNodePtr->GetInDataAnchor(0)),
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove input edge from fused node:%s.",
             nmsNodePtr->GetName().c_str()),
             return FAILED);

    // add the output of pad edge to nms
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(padNodePtr->GetOutDataAnchor(0),
             nmsNodePtr->GetInDataAnchor(0)),
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from node:%s to node:%s failed.",
             padNodePtr->GetName().c_str(), nmsNodePtr->GetName().c_str()),
             return FAILED);

    fusionNodes.push_back(nmsNodePtr);

    return SUCCESS;
}

REGISTER_PASS("NMSWithMaskFusionPass", BUILT_IN_GRAPH_PASS, NMSWithMaskFusionPass);
}
