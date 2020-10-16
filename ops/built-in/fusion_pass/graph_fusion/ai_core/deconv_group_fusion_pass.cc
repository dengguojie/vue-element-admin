/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief split fusion pass(conv2d_backprop_input --> conv2d_backprop_input_d)
 *
 * @author c00523597
 */

#include "deconv_group_fusion_pass.h"

#include <vector>
#include <string>

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
namespace {
const string DECONV = "Deconvolution";
const string PATTERN_DECONV = "Deconvolution";
}

vector<FusionPattern *> DeconvGroupFusionPass::DefinePatterns()
{
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter DeconvGroupPass::DefinePatterns.");
    vector<FusionPattern *> patterns;
    FusionPattern *pattern = new(std::nothrow) FusionPattern("Deconvolution");
    FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
           return patterns);

    pattern->AddOpDesc(PATTERN_DECONV, {DECONV})
        .SetOutput(PATTERN_DECONV);

    patterns.push_back(pattern);

    return patterns;
}

Status DeconvGroupFusionPass::Fusion(ge::ComputeGraph &graph, Mapping& mapping, vector<ge::NodePtr> &fusionNodes)
{
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter DeconvGroupPass::Fusion.");
    NodePtr deconvNode = GetNodeFromMapping(PATTERN_DECONV, mapping);
    OpDescPtr deconvDesc = deconvNode->GetOpDesc();

    // 1.if the deconv node doesn't have the attribute groups or the value is 1, just return not changed.
    int64_t groups = 1;
    bool hasGroup = ge::AttrUtils::GetInt(deconvDesc, "groups", groups);
    if (!hasGroup || groups == 1) {
        OP_LOGD(FUSED_OP_TYPE.c_str(), "The deconv node[name=%s, type=%s] doesn't have the attribute groups, or the value is 1.",
                deconvDesc->GetName().c_str(), deconvDesc->GetType().c_str());
        return NOT_CHANGED;
    }

    GeTensorDesc inputDesc = deconvDesc->GetInputDesc(0);
    size_t inChannelIdx = -1;
    FUSION_PASS_CHECK(SUCCESS != PatternFusionUtil::ParseChannelIdx(inputDesc, inChannelIdx),
            OP_LOGE(FUSED_OP_TYPE.c_str(), "The original format of the deconv node[name=%s, type=%s]'s input0 is %s, which is unsupportable.",
                    deconvDesc->GetName().c_str(), deconvDesc->GetType().c_str(),
                    ge::TypeUtils::FormatToSerialString(inputDesc.GetFormat()).c_str()),
                    return FAILED);
    int64_t inChn = inputDesc.GetOriginShape().GetDim(inChannelIdx);

    GeTensorDesc outputDesc = deconvDesc->GetOutputDesc(0);
    size_t outChannelIdx = -1;
    FUSION_PASS_CHECK(SUCCESS != PatternFusionUtil::ParseChannelIdx(outputDesc, outChannelIdx),
             OP_LOGE(FUSED_OP_TYPE.c_str(), "The original format of the deconv node[name=%s, type=%s]'s output0 is %s, which is unsupportable.",
                     deconvDesc->GetName().c_str(), deconvDesc->GetType().c_str(),
                     ge::TypeUtils::FormatToSerialString(outputDesc.GetFormat()).c_str()),
             return FAILED);
    int64_t outChn = outputDesc.GetOriginShape().GetDim(outChannelIdx);

    // 2. if the number of input channel and output channel are both divisible by groups,
    // then process group padding, otherwise, return failed.
    if (inChn % groups == 0 && outChn % groups == 0) {
        return PatternFusionUtil::ProcessGroupPadding(graph, deconvNode, groups);
    } else {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "The number of input channel(%lld) or output channel(%lld) of "
                "the deconv node[name=%s, type=%s] is not divisible by groups(%lld)",
                inChn, outChn, deconvDesc->GetName().c_str(), deconvDesc->GetType().c_str(), groups);
        return FAILED;
    }
}

REGISTER_PASS("ADeconvGroupFusionPass",
              BUILT_IN_GRAPH_PASS,
              DeconvGroupFusionPass);
}
