/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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
 * \file spacetobatch_conv2d_batchtospace_fusion_pass.cc
 * \brief spacetobatch_conv2d_batchtospace fusion pass(spacetobatch + conv2d + batchtospace --> conv2d)
 */

#include "spacetobatch_conv2d_batchtospace_fusion_pass.h"
#include <vector>
#include <string>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "error_util.h"
#include "common/util/platform_info.h"

namespace fe {
static const std::string PATTERN_SPACETOBATCH = "spacetobatch";
static const std::string PATTERN_CONV2D = "conv2d";
static const std::string PATTERN_BATCHTOSPACE = "batchtospace";
static const std::string SPACETOBATCH_TYPE = "SpaceToBatchND";
static const std::string CONV2D_TYPE = "Conv2D";
static const std::string BATCHTOSPACE_TYPE = "BatchToSpaceND";

constexpr uint32_t BLOCK_SIZE = 2;
constexpr uint32_t BLOCK_H_POS = 0;
constexpr uint32_t BLOCK_W_POS = 1;
constexpr uint32_t PAD_SIZE = 4;
constexpr uint32_t STRIDES_SIZE = 4;
constexpr uint32_t DILATIONS_SIZE = 4;
constexpr uint32_t CROPS_SIZE = 4;
constexpr uint32_t PAD_MAX_VALUE = 255;
constexpr uint32_t DILATIONS_MAX_VALUE = 255;
constexpr uint32_t SPACETOBATCH_CONST_INPUT = 2;
constexpr uint32_t SPACETOBATCH_BLOCK = 0;
constexpr uint32_t SPACETOBATCH_PAD = 1;
constexpr uint32_t FILTER_SHAPE_SIZE = 4;
constexpr int32_t KERNEL_SIZE_MAX = 255;

/*!
  * @brief Define pattern.
  * The graph struct need to adapt is shown as follows:
  *
  *          x
  *          |
  *     spacetobatch                x
  *          |                      |
  *        conv2d        ==>      conv2d
  *          |                      |
  *     batchtospace              output
  *          |
  *        output
  *
  *  Notice: the struct can be captured by
  *          spacetobatch + conv2d + batchtospace pattern
  *  @return vector<FusionPattern*> All valid patterns.
  */

std::vector<FusionPattern*> SpacetobatchConv2dBatchtospacePass::DefinePatterns()
{
    OP_LOGI(fusedOpType_.c_str(), "SpacetobatchConv2dBatchtospacePass define patterns start.");
    std::vector<FusionPattern*> patterns;
    FusionPattern* pattern = new(std::nothrow)FusionPattern("SpacetobatchConv2dBatchtospace");
    FUSION_PASS_CHECK(pattern == nullptr,
        OP_LOGE(fusedOpType_.c_str(), "new a pattern object failed."),
        return patterns);
    pattern->AddOpDesc(PATTERN_SPACETOBATCH, {SPACETOBATCH_TYPE})
        .AddOpDesc(PATTERN_CONV2D, {CONV2D_TYPE})
        .AddOpDesc(PATTERN_BATCHTOSPACE, {BATCHTOSPACE_TYPE})
        .SetInputs(PATTERN_CONV2D, {PATTERN_SPACETOBATCH})
        .SetInputs(PATTERN_BATCHTOSPACE, {PATTERN_CONV2D})
        .SetOutput(PATTERN_BATCHTOSPACE);
    patterns.push_back(pattern);

    OP_LOGI(fusedOpType_.c_str(), "SpacetobatchConv2dBatchtospacePass define patterns end.");
    return patterns;
}

Status SpacetobatchConv2dBatchtospacePass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
    std::vector<ge::NodePtr>& newNodes)
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    auto platRet = PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    FUSION_PASS_CHECK(platRet != SUCCESS,
        OP_LOGW(fusedOpType_.c_str(), "get platform info failed, no fusion."),
        return SUCCESS);
    FUSION_PASS_CHECK(optionalInfo.soc_version == "SD3403" || optionalInfo.soc_version == "Hi3796CV300CS",
        OP_LOGI(fusedOpType_.c_str(), "soc version SD3403/Hi3796CV300CS, no fusion."),
        return SUCCESS);

    OP_LOGI(fusedOpType_.c_str(), "enter SpacetobatchConv2dBatchtospacePass.");
    auto spacetobatchNode = GetNodeFromMapping(PATTERN_SPACETOBATCH, mapping);
    auto batchtospaceNode = GetNodeFromMapping(PATTERN_BATCHTOSPACE, mapping);
    auto conv2dNode = GetNodeFromMapping(PATTERN_CONV2D, mapping);
    auto ret = CheckNodes(spacetobatchNode, batchtospaceNode, conv2dNode);
    if (ret != SUCCESS) {
        return ret;
    }

    ret = UpdateConv2dAttr(spacetobatchNode, conv2dNode);
    if (ret != SUCCESS) {
        return ret;
    }

    ret = UpdateConv2dDesc(spacetobatchNode->GetOpDesc(), batchtospaceNode->GetOpDesc(), conv2dNode->GetOpDesc());
    if (ret != SUCCESS) {
        return ret;
    }

    ret = LinkConv2d(spacetobatchNode, batchtospaceNode, conv2dNode);
    if (ret != SUCCESS) {
        return ret;
    }

    ret = RemoveNodes(graph, spacetobatchNode, batchtospaceNode);
    if (ret != SUCCESS) {
        return ret;
    }

    OP_LOGI(fusedOpType_.c_str(), "leave SpacetobatchConv2dBatchtospacePass.");
    return SUCCESS;
}

Status SpacetobatchConv2dBatchtospacePass::CheckNodes(ge::NodePtr spacetobatchNode,
    ge::NodePtr batchtospaceNode, ge::NodePtr conv2dNode) const
{
    FUSION_PASS_CHECK(spacetobatchNode == nullptr,
        OP_LOGE(fusedOpType_.c_str(), "spacetobatch node is null, fusion failed."),
        return PARAM_INVALID);
    FUSION_PASS_CHECK(batchtospaceNode == nullptr,
        OP_LOGE(fusedOpType_.c_str(), "batchtospace node is null, fusion failed."),
        return PARAM_INVALID);
    FUSION_PASS_CHECK(conv2dNode == nullptr,
        OP_LOGE(fusedOpType_.c_str(), "conv2d node is null, fusion failed."),
        return PARAM_INVALID);

    FUSION_PASS_CHECK(spacetobatchNode->GetOutDataNodes().size() > 1,
        OP_LOGI(fusedOpType_.c_str(), "spacetobatch output multi nodes, no fusion."),
        return NOT_CHANGED);
    FUSION_PASS_CHECK(batchtospaceNode->GetOutDataNodes().size() > 1,
        OP_LOGI(fusedOpType_.c_str(), "batchtospace output multi nodes, no fusion."),
        return NOT_CHANGED);
    FUSION_PASS_CHECK(conv2dNode->GetOutDataNodes().size() > 1,
        OP_LOGI(fusedOpType_.c_str(), "conv output multi nodes, no fusion."),
        return NOT_CHANGED);

    FUSION_PASS_CHECK(spacetobatchNode->GetOpDesc() == nullptr,
        OP_LOGE(fusedOpType_.c_str(), "spacetobatch desc is null, fusion failed."),
        return PARAM_INVALID);
    FUSION_PASS_CHECK(batchtospaceNode->GetOpDesc() == nullptr,
        OP_LOGE(fusedOpType_.c_str(), "batchtospace desc is null, fusion failed."),
        return PARAM_INVALID);
    FUSION_PASS_CHECK(conv2dNode->GetOpDesc() == nullptr,
        OP_LOGE(fusedOpType_.c_str(), "conv2d desc is null, fusion failed."),
        return PARAM_INVALID);

    auto ret = CheckCrops(batchtospaceNode);
    if (ret != SUCCESS) {
        return ret;
    }

    return SUCCESS;
}

Status SpacetobatchConv2dBatchtospacePass::CheckCrops(ge::NodePtr batchtospaceNode) const
{
    auto batchWeight = ge::OpDescUtils::GetWeights(batchtospaceNode);
    FUSION_PASS_CHECK(batchWeight.size() != 2,
        OP_LOGI(fusedOpType_.c_str(), "batchtospace weight %zu, no fusion.", batchWeight.size()),
        return NOT_CHANGED);

    auto cropsPtr = batchWeight[1];
    FUSION_PASS_CHECK(cropsPtr == nullptr,
        OP_LOGE(fusedOpType_.c_str(), "batchtospace crops is null, fusion failed."),
        return PARAM_INVALID);
    auto cropsData = cropsPtr->GetData().GetData();
    FUSION_PASS_CHECK(cropsData == nullptr,
        OP_LOGE(fusedOpType_.c_str(), "batchtospace crops data is null, fusion failed."),
        return PARAM_INVALID);
    auto dataSize = cropsPtr->GetData().GetSize();
    auto dataType = cropsPtr->GetTensorDesc().GetDataType();
    if (dataType == DT_INT32 && dataSize / sizeof(int32_t) == CROPS_SIZE) {
        auto cropsValue = reinterpret_cast<const int32_t*>(cropsData);
        for (uint32_t i = 0; i < CROPS_SIZE; ++i) {
            FUSION_PASS_CHECK(cropsValue[i] != 0,
                OP_LOGI(fusedOpType_.c_str(), "crops not 0, no fusion."), return NOT_CHANGED);
        }
    } else if (dataType == DT_INT64 && dataSize / sizeof(int64_t) == CROPS_SIZE) {
        auto cropsValue = reinterpret_cast<const int64_t*>(cropsData);
        for (uint32_t i = 0; i < CROPS_SIZE; ++i) {
            FUSION_PASS_CHECK(cropsValue[i] != 0,
                OP_LOGI(fusedOpType_.c_str(), "crops not 0, no fusion."), return NOT_CHANGED);
        }
    } else {
        return NOT_CHANGED;
    }

    return SUCCESS;
}

Status SpacetobatchConv2dBatchtospacePass::CheckKernelSize(ge::OpDescPtr convDesc,
    int64_t dilationH, int64_t dilationW) const
{
    auto filterDesc = convDesc->GetInputDesc("filter");
    FUSION_PASS_CHECK(filterDesc.IsValid() != GRAPH_SUCCESS,
        OP_LOGI(fusedOpType_.c_str(), "no filter no fusion."), return NOT_CHANGED);
    auto shape = filterDesc.GetShape().GetDims();
    FUSION_PASS_CHECK(shape.size() != FILTER_SHAPE_SIZE,
        OP_LOGI(fusedOpType_.c_str(), "invalid filter shape dim %zu, no fusion.", shape.size()),
        return NOT_CHANGED);

    std::string format = TypeUtils::FormatToSerialString(filterDesc.GetFormat());
    size_t kernelH = format.find('H');
    size_t kernelW = format.find('W');
    FUSION_PASS_CHECK(kernelH >= shape.size() || kernelW >= shape.size(),
        OP_LOGI(fusedOpType_.c_str(), "invalid format, no fusion."), return NOT_CHANGED);

    int64_t kernelSizeH = (shape[kernelH] - 1) * dilationH + 1;
    int64_t kernelSizeW = (shape[kernelW] - 1) * dilationW + 1;
    FUSION_PASS_CHECK(kernelSizeH > KERNEL_SIZE_MAX || kernelSizeW > KERNEL_SIZE_MAX,
        OP_LOGI(fusedOpType_.c_str(), "kernel size over size, no fusion."), return NOT_CHANGED);

    return SUCCESS;
}

Status SpacetobatchConv2dBatchtospacePass::CheckConvStrides(ge::OpDescPtr convDesc) const
{
    std::vector<int64_t> strides;
    ge::AttrUtils::GetListInt(convDesc, "strides", strides);
    FUSION_PASS_CHECK(strides.size() != STRIDES_SIZE,
        OP_LOGE(fusedOpType_.c_str(), "invalid conv strides, fusion failed."),
        return PARAM_INVALID);

    std::string format;
    ge::AttrUtils::GetStr(convDesc, "data_format", format);
    size_t fondH = format.find('H');
    size_t fondW = format.find('W');
    FUSION_PASS_CHECK(fondH >= strides.size() || fondW >= strides.size(),
        OP_LOGE(fusedOpType_.c_str(), "invalid conv format, fusion failed."),
        return PARAM_INVALID);

    if (strides[fondH] != 1 && strides[fondW] != 1) {
        OP_LOGI(fusedOpType_.c_str(), "stride not 1x1, no fusion.");
        return NOT_CHANGED;
    }

    return SUCCESS;
}

Status SpacetobatchConv2dBatchtospacePass::CheckConvPads(ge::ConstGeTensorPtr spacePadPtr,
    ge::OpDescPtr convDesc, std::vector<int64_t>& convPads) const
{
    FUSION_PASS_CHECK(spacePadPtr == nullptr,
        OP_LOGE(fusedOpType_.c_str(), "space pad is null, fusion failed."), return PARAM_INVALID);

    auto spacePadData = spacePadPtr->GetData().GetData();
    auto dataSize = spacePadPtr->GetData().GetSize();
    auto dataType = spacePadPtr->GetTensorDesc().GetDataType();
    FUSION_PASS_CHECK(spacePadData == nullptr,
        OP_LOGE(fusedOpType_.c_str(), "space pad data is null, fusion failed."), return PARAM_INVALID);
    std::vector<int64_t> spacePads;
    if (dataType == DT_INT32 && dataSize / sizeof(int32_t) == PAD_SIZE) {
        auto padValue = reinterpret_cast<const int32_t*>(spacePadData);
        for (uint32_t i = 0; i < PAD_SIZE; ++i) {
            spacePads.emplace_back(padValue[i]);
        }
    } else if (dataType == DT_INT64 && dataSize / sizeof(int32_t) == PAD_SIZE) {
        auto padValue = reinterpret_cast<const int64_t*>(spacePadData);
        for (uint32_t i = 0; i < PAD_SIZE; ++i) {
            spacePads.emplace_back(padValue[i]);
        }
    } else {
        OP_LOGE(fusedOpType_.c_str(), "invalid space pad data, fusion failed.");
        return PARAM_INVALID;
    }
    for (auto& pad : spacePads) {
        FUSION_PASS_CHECK(pad < 0,
            OP_LOGI(fusedOpType_.c_str(), "space pad < 0, no fusion."), return NOT_CHANGED);
    }

    ge::AttrUtils::GetListInt(convDesc, "pads", convPads);
    FUSION_PASS_CHECK(convPads.size() != spacePads.size(),
        OP_LOGE(fusedOpType_.c_str(), "invalid conv pads, fusion failed."),
        return PARAM_INVALID);
    for (auto& pad : convPads) {
        FUSION_PASS_CHECK(pad != 0,
            OP_LOGI(fusedOpType_.c_str(), "conv pad not 0, no fusion."), return NOT_CHANGED);
    }

    for (size_t i = 0; i < convPads.size(); ++i) {
        convPads[i] += spacePads[i];
        FUSION_PASS_CHECK(convPads[i] > PAD_MAX_VALUE,
            OP_LOGI(fusedOpType_.c_str(), "pad > 255, no fusion."), return NOT_CHANGED);
        OP_LOGI(fusedOpType_.c_str(), "fusion pad %lu.", convPads[i]);
    }

    return SUCCESS;
}

Status SpacetobatchConv2dBatchtospacePass::CheckConvDilations(ge::ConstGeTensorPtr blockPtr,
    ge::OpDescPtr convDesc, std::vector<int64_t>& dilations) const
{
    FUSION_PASS_CHECK(blockPtr == nullptr,
        OP_LOGE(fusedOpType_.c_str(), "block is null, fusion failed."), return PARAM_INVALID);

    auto blockData = blockPtr->GetData().GetData();
    auto dataSize = blockPtr->GetData().GetSize();
    auto dataType = blockPtr->GetTensorDesc().GetDataType();
    FUSION_PASS_CHECK(blockData == nullptr,
        OP_LOGE(fusedOpType_.c_str(), "space block data is null, fusion failed."), return PARAM_INVALID);
    int64_t blockH;
    int64_t blockW;
    if (dataType == DT_INT32 && dataSize / sizeof(int32_t) == BLOCK_SIZE) {
        auto blockDataValue = reinterpret_cast<const int32_t*>(blockData);
        blockH = blockDataValue[BLOCK_H_POS];
        blockW = blockDataValue[BLOCK_W_POS];
    } else if (dataType == DT_INT64 && dataSize / sizeof(int64_t) == BLOCK_SIZE) {
        auto blockDataValue = reinterpret_cast<const int64_t*>(blockData);
        blockH = blockDataValue[BLOCK_H_POS];
        blockW = blockDataValue[BLOCK_W_POS];
    } else {
        OP_LOGE(fusedOpType_.c_str(), "invalid block data, fusion failed.");
        return PARAM_INVALID;
    }

    ge::AttrUtils::GetListInt(convDesc, "dilations", dilations);
    FUSION_PASS_CHECK(dilations.size() != DILATIONS_SIZE,
        OP_LOGE(fusedOpType_.c_str(), "invalid dilations, fusion failed."), return PARAM_INVALID);
    for (auto& dilation : dilations) {
        FUSION_PASS_CHECK(dilation < 0,
            OP_LOGI(fusedOpType_.c_str(), "conv dilations < 0, no fusion."), return NOT_CHANGED);
    }

    std::string format;
    ge::AttrUtils::GetStr(convDesc, "data_format", format);
    size_t fondH = format.find('H');
    size_t fondW = format.find('W');
    FUSION_PASS_CHECK(fondH >= dilations.size() || fondW >= dilations.size(),
        OP_LOGE(fusedOpType_.c_str(), "invalid format, fusion failed."), return PARAM_INVALID);
    dilations[fondH] *= blockH;
    dilations[fondW] *= blockW;
    FUSION_PASS_CHECK(dilations[fondH] > DILATIONS_MAX_VALUE || dilations[fondW] > DILATIONS_MAX_VALUE,
        OP_LOGI(fusedOpType_.c_str(), "dilations > max, no fusion."), return NOT_CHANGED);
    FUSION_PASS_CHECK(CheckKernelSize(convDesc, dilations[fondH], dilations[fondW]) != SUCCESS,
        OP_LOGI(fusedOpType_.c_str(), "over kernel size, no fusion."), return NOT_CHANGED);

    OP_LOGI(fusedOpType_.c_str(), "fusion dilations %lu %lu.", dilations[fondH], dilations[fondW]);
    return SUCCESS;
}

Status SpacetobatchConv2dBatchtospacePass::UpdateConv2dAttr(ge::NodePtr spaceNode, ge::NodePtr convNode) const
{
    auto convDesc = convNode->GetOpDesc();
    auto ret = CheckConvStrides(convDesc);
    if (ret != SUCCESS) {
        return ret;
    }

    auto spaceWeight = ge::OpDescUtils::GetWeights(spaceNode);
    FUSION_PASS_CHECK(spaceWeight.size() != SPACETOBATCH_CONST_INPUT,
        OP_LOGI(fusedOpType_.c_str(), "spacetobatch weight size %zu, no fusion.", spaceWeight.size()),
        return NOT_CHANGED);

    std::vector<int64_t> dilations;
    ret = CheckConvDilations(spaceWeight[SPACETOBATCH_BLOCK], convDesc, dilations);
    if (ret != SUCCESS) {
        return ret;
    }

    std::vector<int64_t> convPads;
    ret = CheckConvPads(spaceWeight[SPACETOBATCH_PAD], convDesc, convPads);
    if (ret != SUCCESS) {
        return ret;
    }

    FUSION_PASS_CHECK(!AttrUtils::SetListInt(convDesc, "dilations", dilations),
        OP_LOGE(fusedOpType_.c_str(), "set dilations failed, fusion failed."),
        return FAILED);

    FUSION_PASS_CHECK(!AttrUtils::SetListInt(convDesc, "pads", convPads),
        OP_LOGE(fusedOpType_.c_str(), "set pads failed, fusion failed."),
        return FAILED);

    return SUCCESS;
}

Status SpacetobatchConv2dBatchtospacePass::UpdateConv2dDesc(ge::OpDescPtr spaceDesc,
    ge::OpDescPtr batchDesc, ge::OpDescPtr convDesc) const
{
    // update conv input desc
    auto spaceInTensor = spaceDesc->GetInputDesc(0);
    auto spaceOriShape = spaceInTensor.GetOriginShape().GetDims();
    auto spaceShape = spaceInTensor.GetShape().GetDims();
    auto convInTensor = convDesc->GetInputDesc(0);
    convInTensor.SetOriginShape(ge::GeShape(spaceOriShape));
    convInTensor.SetShape(ge::GeShape(spaceShape));
    convDesc->UpdateInputDesc(0, convInTensor);

    // update conv output desc
    auto batchOutTensor = batchDesc->GetOutputDesc(0);
    auto batchOriShape = batchOutTensor.GetOriginShape().GetDims();
    auto batchShape = batchOutTensor.GetShape().GetDims();
    auto convOutTensor = convDesc->GetOutputDesc(0);
    convOutTensor.SetOriginShape(ge::GeShape(batchOriShape));
    convOutTensor.SetShape(ge::GeShape(batchShape));
    convDesc->UpdateOutputDesc(0, convOutTensor);

    return SUCCESS;
}

Status SpacetobatchConv2dBatchtospacePass::LinkConv2d(ge::NodePtr spacetobatchNode,
    ge::NodePtr batchtospaceNode, ge::NodePtr conv2dNode) const
{
    // remove input--spacetobatch
    auto spaceInAnchor = spacetobatchNode->GetInDataAnchor(0);
    FUSION_PASS_CHECK(spaceInAnchor == nullptr,
        OP_LOGE(fusedOpType_.c_str(), "spacetobatch input data anchor is null, fusion failed."),
        return PARAM_INVALID);
    auto spacePreAnchor = spaceInAnchor->GetPeerOutAnchor();
    FUSION_PASS_CHECK(spacePreAnchor == nullptr,
        OP_LOGE(fusedOpType_.c_str(), "spacetobatch input peer anchor is null, fusion failed."),
        return PARAM_INVALID);
    FUSION_PASS_CHECK(GraphUtils::RemoveEdge(spacePreAnchor, spaceInAnchor) != GRAPH_SUCCESS,
        OP_LOGE(fusedOpType_.c_str(), "remove edge from input to spacetobatch failed, fusion failed."),
        return PARAM_INVALID);

    // remove spacetobatch--conv2d, add input--conv2d
    auto convInAnchor = conv2dNode->GetInDataAnchor(0);
    FUSION_PASS_CHECK(convInAnchor == nullptr,
        OP_LOGE(fusedOpType_.c_str(), "conv2d input data anchor is null, fusion failed."),
        return PARAM_INVALID);
    auto convPreAnchor = convInAnchor->GetPeerOutAnchor();  // spacetobatch out anchor
    FUSION_PASS_CHECK(convPreAnchor == nullptr,
        OP_LOGE(fusedOpType_.c_str(), "conv2d input data anchor is null, fusion failed."),
        return PARAM_INVALID);
    FUSION_PASS_CHECK(GraphUtils::RemoveEdge(convPreAnchor, convInAnchor) != GRAPH_SUCCESS,
        OP_LOGE(fusedOpType_.c_str(), "remove edge from input to spacetobatch failed, fusion failed."),
        return PARAM_INVALID);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(spacePreAnchor, convInAnchor) != GRAPH_SUCCESS,
        OP_LOGE(fusedOpType_.c_str(), "remove edge from input to spacetobatch failed, fusion failed."),
        return PARAM_INVALID);

    // remove conv--batchtospace
    auto convOutAnchor = conv2dNode->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(convOutAnchor == nullptr,
        OP_LOGE(fusedOpType_.c_str(), "conv out data anchor is null, fusion failed."),
        return PARAM_INVALID);
    auto convPeerInAnchor = convOutAnchor->GetPeerInDataAnchors();
    for (const auto& nextAnchor : convPeerInAnchor) {
        FUSION_PASS_CHECK(GraphUtils::RemoveEdge(convOutAnchor, nextAnchor) != GRAPH_SUCCESS,
            OP_LOGE(fusedOpType_.c_str(), "remove edge from conv to batchtospace failed, fusion failed."),
            return PARAM_INVALID);
    }

    // remove batchtospace--output, add conv--output
    auto batchOutAnchor = batchtospaceNode->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(batchOutAnchor == nullptr,
        OP_LOGE(fusedOpType_.c_str(), "batchtospace out data anchor is null, fusion failed."),
        return PARAM_INVALID);
    auto batchPeerInAnchor = batchOutAnchor->GetPeerInDataAnchors();
    for (const auto& nextAnchor : batchPeerInAnchor) {
        FUSION_PASS_CHECK(GraphUtils::RemoveEdge(batchOutAnchor, nextAnchor) != GRAPH_SUCCESS,
            OP_LOGE(fusedOpType_.c_str(), "remove edge from batchtospace to ouput failed, fusion failed."),
            return PARAM_INVALID);
        FUSION_PASS_CHECK(GraphUtils::AddEdge(convOutAnchor, nextAnchor) != GRAPH_SUCCESS,
            OP_LOGE(fusedOpType_.c_str(), "add edge from conv to output failed, fusion failed."),
            return PARAM_INVALID);
    }

    return SUCCESS;
}

Status SpacetobatchConv2dBatchtospacePass::RemoveNodes(ge::ComputeGraph& graph,
    ge::NodePtr spacetobatchNode, ge::NodePtr batchtospaceNode) const
{
    FUSION_PASS_CHECK(graph.RemoveNode(spacetobatchNode) != GRAPH_SUCCESS,
        OP_LOGE(fusedOpType_.c_str(), "remove spacetobatch failed, fusion failed."),
        return PARAM_INVALID);

    FUSION_PASS_CHECK(graph.RemoveNode(batchtospaceNode) != GRAPH_SUCCESS,
        OP_LOGE(fusedOpType_.c_str(), "remove batchtospace failed, fusion failed."),
        return PARAM_INVALID);

    return SUCCESS;
}

REGISTER_PASS("SpaceToBatchConv2dBatchToSpacePass", BUILT_IN_GRAPH_PASS, SpacetobatchConv2dBatchtospacePass);
}
