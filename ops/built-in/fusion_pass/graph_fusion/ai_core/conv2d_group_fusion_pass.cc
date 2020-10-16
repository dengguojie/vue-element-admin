/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief conv2d group fusion pass(conv2d --> conv2d/splited conv2d/depthwise conv2d)
 *
 */
#include "conv2d_group_fusion_pass.h"
#include <vector>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace ge;

namespace fe {
const string PATTERN_CONV2D_ID = "conv2d_group_id";
const string CONV2D_TYPE = "Conv2D";
const string ATTR_GROUPS = "groups";

enum {
    DIM_N = 0,
    DIM_C = 1,
    DIM_H = 2,
    DIM_W = 3
};

vector<FusionPattern *> Conv2DGroupFusionPass::DefinePatterns()
{
    vector<FusionPattern *> patterns;
    FusionPattern* pattern = new(std::nothrow) FusionPattern("Conv2DGroupFusionPass");
    FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
           return patterns);
    pattern->AddOpDesc(PATTERN_CONV2D_ID, {CONV2D_TYPE}).SetOutput(PATTERN_CONV2D_ID);
    patterns.push_back(pattern);
    return patterns;
}

namespace {
const int MAX_DIM_NUM = 4;
}

Status Conv2DGroupFusionPass::SwapNumChn(OpDescPtr opDesc, bool bInput, uint32_t index)
{
    ge::GeTensorDesc tensorDesc;
    if (bInput) {
        tensorDesc = opDesc->GetInputDesc(index);
    } else {
        tensorDesc = opDesc->GetOutputDesc(index);
    }
    FUSION_PASS_CHECK(tensorDesc.GetShape().GetDimNum() != MAX_DIM_NUM,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "dim count not illegal, need:4 real:%d",
            tensorDesc.GetShape().GetDimNum()),
        return PARAM_INVALID);
    // Refresh the variable format and shape
    int64_t n = tensorDesc.GetShape().GetDim(DIM_C);
    int64_t c = tensorDesc.GetShape().GetDim(DIM_N);
    int64_t h = tensorDesc.GetShape().GetDim(DIM_H);
    int64_t w = tensorDesc.GetShape().GetDim(DIM_W);
    tensorDesc.SetShape(ge::GeShape({ n, c, h, w }));
    tensorDesc.SetOriginShape(ge::GeShape({ n, c, h, w }));
    graphStatus retRes;
    if (bInput) {
        retRes = opDesc->UpdateInputDesc(index, tensorDesc);
    } else {
        retRes = opDesc->UpdateOutputDesc(index, tensorDesc);
    }
    FUSION_PASS_CHECK(retRes != ge::GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Update matmul variable failed"),
        return PARAM_INVALID);
    return SUCCESS;
}

Status Conv2DGroupFusionPass::ProcessDepthwiseConv(NodePtr convNode)
{
    FUSION_PASS_CHECK(convNode->GetInAllNodes().size() < 2,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "The number of input of the node[name=%s, type=%s] is less than 2, there is no weight input.",
                     convNode->GetName().c_str(), convNode->GetType().c_str()),
             return FAILED);
    OpDescPtr filterDesc = convNode->GetInAllNodes().at(1)->GetOpDesc();
    FUSION_PASS_CHECK(filterDesc == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "Filter GetOpDesc fail"),
            return PARAM_INVALID);

    OpDescPtr convDesc = convNode->GetOpDesc();
    FUSION_PASS_CHECK(convDesc->GetInputDesc(1).GetShape().GetDim(1) != 1,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Filter channel must be 1 in depthwise conv"),
        return PARAM_INVALID);
    FUSION_PASS_CHECK(SwapNumChn(filterDesc, false, 0) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Conv parent const node out 0 change nc failed"), return FAILED);

    FUSION_PASS_CHECK(SwapNumChn(convDesc, true, 1) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Conv node input 1 change nc failed"), return FAILED);

    // change op type to depthwise
    OP_LOGI(FUSED_OP_TYPE.c_str(), "change the conv type");
    convDesc->SetType("DepthwiseConv2D");
    // because conv2d no data_format and padding setting but depthwise has
    FUSION_PASS_CHECK(!ge::AttrUtils::SetStr(convDesc, "data_format", "NCHW"),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "set data_format NCHW fail"), return FAILED);
    convDesc->DelAttr("groups");
    return SUCCESS;
}

Status Conv2DGroupFusionPass::Fusion(ge::ComputeGraph &graph,
                               Mapping &mapping,
                               vector<ge::NodePtr> &newNodes)
{
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter Conv2DGroupPass::Fusion.");
  NodePtr convNode = GetNodeFromMapping(PATTERN_CONV2D_ID, mapping);
  OpDescPtr convDesc = convNode->GetOpDesc();

  // 1.if the deconv node doesn't have the attribute groups or the value is 1, just return not changed.
  int64_t groups = 1;
  bool hasGroup = ge::AttrUtils::GetInt(convDesc, "groups", groups);
  if (!hasGroup || groups == 1) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "The conv node[name=%s, type=%s] doesn't have the attribute groups, or the value is 1.",
            convDesc->GetName().c_str(), convDesc->GetType().c_str());
    return NOT_CHANGED;
  }

  GeTensorDesc inputDesc = convDesc->GetInputDesc(0);
  size_t inChannelIdx = -1;
  FUSION_PASS_CHECK(SUCCESS != PatternFusionUtil::ParseChannelIdx(inputDesc, inChannelIdx),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "The original format of the conv node[name=%s, type=%s]'s input0 is %s, which is unsupportable.",
                   convDesc->GetName().c_str(), convDesc->GetType().c_str(),
                   ge::TypeUtils::FormatToSerialString(inputDesc.GetFormat()).c_str()),
           return FAILED);
  int64_t inChn = inputDesc.GetOriginShape().GetDim(inChannelIdx);

  GeTensorDesc outputDesc = convDesc->GetOutputDesc(0);
  size_t outChannelIdx = -1;
  FUSION_PASS_CHECK(SUCCESS != PatternFusionUtil::ParseChannelIdx(outputDesc, outChannelIdx),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "The original format of the conv node[name=%s, type=%s]'s output0 is %s, which is unsupportable.",
                   convDesc->GetName().c_str(), convDesc->GetType().c_str(),
                   ge::TypeUtils::FormatToSerialString(outputDesc.GetFormat()).c_str()),
           return FAILED);
  int64_t outChn = outputDesc.GetOriginShape().GetDim(outChannelIdx);

  if (groups == inChn && groups == outChn) {
    return ProcessDepthwiseConv(convNode);
  } else if (inChn % groups == 0 && outChn % groups == 0) {
    return PatternFusionUtil::ProcessGroupPadding(graph, convNode, groups);
  } else {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "The number of input channel(%lld) or output channel(%lld) of "
            "the conv node[name=%s, type=%s] is not divisible by groups(%lld)",
            inChn, outChn, convDesc->GetName().c_str(), convDesc->GetType().c_str(), groups);
    return FAILED;
  }
}
REGISTER_PASS("GroupConv2DFusionPass",
        BUILT_IN_GRAPH_PASS, Conv2DGroupFusionPass);
}
