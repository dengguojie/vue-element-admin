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
 * \file same_input_conv2d_fusion_pass.cc
 * \brief same_input_conv2d fusion pass
 */

#include "same_input_conv2d_fusion_pass.h"
#include <vector>
#include <string>
#include <numeric>
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

namespace fe {
constexpr uint32_t CONV2D_INPUT_SIZE_MIN = 2;
constexpr uint32_t FILTER_SHAPE_SIZE = 4;
constexpr uint32_t BIAS_SHAPE_SIZE = 1;
constexpr uint32_t FILTER_POS = 1;
constexpr uint32_t BIAS_POS = 2;
constexpr uint32_t SIZE_SPLIT_POS = 1;
constexpr uint32_t DIM_SPLIT_POS = 2;
constexpr uint32_t DATA_ALIGN = 16;
constexpr uint32_t INT8_ALIGN = 32;
constexpr float DELTA = 0.001;

/*!
  * @brief Define pattern.
  * The graph struct need to adapt is shown as follows:
  * pattern:
  *          x                     x
  *       /     \                  |
  *    conv      conv            conv
  *      |        |      ==>       |
  *    relu      relu            relu
  *      |        |                |
  *    conv      conv            split
  *      |        |              /   \
  *    relu      relu         conv   conv
  *                            |       |
  *                           relu   relu
  *
  * pattern quant:
  *          x                     x
  *       /     \                  |
  *    conv     conv             conv
  *     |        |                 |
  *  dequant  dequant   ==>     dequant
  *     |        |                 |
  *   quant    quant             quant
  *     |        |                 |
  *   conv      conv             split
  *     |        |               /   \
  * dequant    dequant        conv   conv
  *                            |       |
  *                        dequant   dequant
  *
  * 
  * pattern requant:
  *          x                     x
  *       /     \                  |
  *    conv    conv               conv
  *      |      |                  |
  *  requant  requant           requant
  *      |      |                  |
  *    conv    conv              split
  *      |      |                /   \
  *  requant  requant         conv   conv
  *                            |       |
  *                        requant   requant
  *
  *  Notice: the struct can be captured by
  *          input + conv + relu + conv pattern
  *          input + conv + ascend_dequant + relu + ascend_quant + conv
  *  @return vector<FusionPattern*> All valid patterns.
  */

std::vector<FusionPattern*> SameInputConv2dPass::DefinePatterns()
{
    OP_LOGD(FUSED_OP_TYPE.c_str(), "SameInputConv2dPass define patterns start.");
    std::vector<FusionPattern*> patterns;
    FusionPattern* pattern = new(std::nothrow)FusionPattern("SameInputConv2dPass");
    FUSION_PASS_CHECK(pattern == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."), return patterns);
    pattern->AddOpDesc(PATTERN_INPUT)
        .AddOpDesc(PATTERN_CONV2D_0, {CONV2D_TYPE})
        .AddOpDesc(PATTERN_CONV2D_1, {CONV2D_TYPE})
        .AddOpDesc(PATTERN_RELU_REQUANT, {RELU_TYPE, REQUANT_TYPE})
        .SetInputs(PATTERN_CONV2D_0, {PATTERN_INPUT})
        .SetInputs(PATTERN_RELU_REQUANT, {PATTERN_CONV2D_0})
        .SetInputs(PATTERN_CONV2D_1, {PATTERN_RELU_REQUANT})
        .SetOutput(PATTERN_CONV2D_1);
    patterns.push_back(pattern);

    FusionPattern* patternQuant = new(std::nothrow)FusionPattern("SameInputConv2dPass");
    FUSION_PASS_CHECK(patternQuant == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern quant object failed."), return patterns);
    patternQuant->AddOpDesc(PATTERN_INPUT)
        .AddOpDesc(PATTERN_CONV2D_0, {CONV2D_TYPE})
        .AddOpDesc(PATTERN_CONV2D_1, {CONV2D_TYPE})
        .AddOpDesc(PATTERN_QUANT, {QUANT_TYPE})
        .AddOpDesc(PATTERN_DEQUANT, {DEQUANT_TYPE})
        .SetInputs(PATTERN_CONV2D_0, {PATTERN_INPUT})
        .SetInputs(PATTERN_DEQUANT, {PATTERN_CONV2D_0})
        .SetInputs(PATTERN_QUANT, {PATTERN_DEQUANT})
        .SetInputs(PATTERN_CONV2D_1, {PATTERN_QUANT})
        .SetOutput(PATTERN_CONV2D_1);
    patterns.push_back(patternQuant);

    OP_LOGD(FUSED_OP_TYPE.c_str(), "SameInputConv2dPass define patterns end.");
    return patterns;
}

Status SameInputConv2dPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, std::vector<ge::NodePtr>& newNodes)
{
    fusionNodes_.clear();
    quantPattern_ = false;
    requantPattern_ = false;

    OP_LOGI(FUSED_OP_TYPE.c_str(), "enter SameInputConv2dPass.");
    auto ret = CheckFusion(mapping);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGI(FUSED_OP_TYPE.c_str(), "check fusion."), return ret);

    ret = AddConcatNodes(graph, newNodes);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGI(FUSED_OP_TYPE.c_str(), "add concat failed."), return ret);

    ret = AddSplitNode(graph, newNodes);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGI(FUSED_OP_TYPE.c_str(), "add split failed."), return ret);

    ret = UpdateConvNode(graph);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGI(FUSED_OP_TYPE.c_str(), "update conv failed."), return ret);

    OP_LOGI(FUSED_OP_TYPE.c_str(), "leave SameInputConv2dPass.");
    return SUCCESS;
}

Status SameInputConv2dPass::CheckFusion(Mapping& mapping)
{
    auto inputNode = GetNodeFromMapping(PATTERN_INPUT, mapping);
    FUSION_PASS_CHECK(inputNode == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "input node is null, fusion failed."), return PARAM_INVALID);

    auto ret = GetAllPatternNodes(mapping, inputNode);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGI(FUSED_OP_TYPE.c_str(), "get pattern nodes."), return ret);

    ret = CheckAllConvNodes(inputNode);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGI(FUSED_OP_TYPE.c_str(), "check pattern nodes."), return ret);

    ret = CheckConvAttr();
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGI(FUSED_OP_TYPE.c_str(), "check conv attr."), return ret);

    ret = CheckDequantNodes();
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGI(FUSED_OP_TYPE.c_str(), "check dequant node."), return ret);

    ret = CheckQuantNodes();
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGI(FUSED_OP_TYPE.c_str(), "check quant node."), return ret);

    OP_LOGI(FUSED_OP_TYPE.c_str(), "conv fusion num %zu.", fusionNodes_.size());
    return SUCCESS;
}

bool SameInputConv2dPass::CheckLastNextNode(ge::NodePtr lastNode) const
{
    // check whether the last node is conv
    FUSION_PASS_CHECK(lastNode == nullptr,
        OP_LOGI(FUSED_OP_TYPE.c_str(), "last node is null, no fusion."), return false);
    FUSION_PASS_CHECK(lastNode->GetType() != CONV2D_TYPE,
        OP_LOGI(FUSED_OP_TYPE.c_str(), "last node %s is not conv, no fusion.", lastNode->GetType().c_str()),
        return false);

    // check last node is output
    FUSION_PASS_CHECK(lastNode->GetOutDataNodes().empty(),
        OP_LOGI(FUSED_OP_TYPE.c_str(), "last node is last."), return true);

    FUSION_PASS_CHECK(lastNode->GetOutDataNodes().size() != 1,
        OP_LOGI(FUSED_OP_TYPE.c_str(), "last node multi out."), return false);
    ge::NodePtr nextNode = lastNode->GetOutDataNodes().at(0);
    FUSION_PASS_CHECK(nextNode == nullptr,
        OP_LOGI(FUSED_OP_TYPE.c_str(), "next node is null."), return false);

    // check next node type
    auto nextType = nextNode->GetType();
    FUSION_PASS_CHECK(quantPattern_ && nextType != DEQUANT_TYPE,
        OP_LOGI(FUSED_OP_TYPE.c_str(), "next not deuant %s.", nextNode->GetType().c_str()), return false);
    FUSION_PASS_CHECK(requantPattern_ && nextType != REQUANT_TYPE,
        OP_LOGI(FUSED_OP_TYPE.c_str(), "next not requant %s, no fusion.", nextNode->GetType().c_str()), return false);
    FUSION_PASS_CHECK(!quantPattern_ && !requantPattern_ && nextType != RELU_TYPE,
        OP_LOGI(FUSED_OP_TYPE.c_str(), "next not relu, %s, no fusion.", nextNode->GetType().c_str()), return false);

    if (quantPattern_) {
        // check next node is output
        FUSION_PASS_CHECK(nextNode->GetOutDataNodes().empty(),
            OP_LOGI(FUSED_OP_TYPE.c_str(), "next node is last."), return true);
        FUSION_PASS_CHECK(nextNode->GetOutDataNodes().size() != 1,
            OP_LOGI(FUSED_OP_TYPE.c_str(), "next node multi out."), return false);
        ge::NodePtr nextNextNode = nextNode->GetOutDataNodes().at(0);
        FUSION_PASS_CHECK(nextNextNode == nullptr,
            OP_LOGI(FUSED_OP_TYPE.c_str(), "next next node is null."), return false);

        auto nextNextType = nextNextNode->GetType();
        FUSION_PASS_CHECK(nextNextType != QUANT_TYPE && nextNextType != CONCATV2D_TYPE,
            OP_LOGI(FUSED_OP_TYPE.c_str(), "next next not quant/concatv2d no fusion."), return false);
    }

    return true;
}

void SameInputConv2dPass::GetPatternNodes(ge::NodePtr patternConv)
{
    FUSION_PASS_CHECK(patternConv == nullptr,
        OP_LOGI(FUSED_OP_TYPE.c_str(), "pattern conv node is null, no fusion."), return);
    FUSION_PASS_CHECK(patternConv->GetType() != CONV2D_TYPE,
        OP_LOGI(FUSED_OP_TYPE.c_str(), "first not conv, %s.", patternConv->GetType().c_str()), return);

    // check whether the node after conv is relu/requant/dequant
    FUSION_PASS_CHECK(patternConv->GetOutDataNodes().size() != 1,
        OP_LOGI(FUSED_OP_TYPE.c_str(), "conv out data node not 1, no fusion."), return);
    ge::NodePtr convOut = patternConv->GetOutDataNodes().at(0);
    FUSION_PASS_CHECK(convOut == nullptr,
        OP_LOGI(FUSED_OP_TYPE.c_str(), "conv out data node is null, no fusion."), return);
    ConvFusionNodes node;
    node.convNode = patternConv;
    ge::NodePtr lastNode = nullptr;
    if (quantPattern_) {
        node.dequantNode = convOut;
        FUSION_PASS_CHECK(node.dequantNode->GetType() != DEQUANT_TYPE,
            OP_LOGI(FUSED_OP_TYPE.c_str(), "not dequant, %s.", node.dequantNode->GetType().c_str()), return);
        FUSION_PASS_CHECK(node.dequantNode->GetOutDataNodes().size() != 1,
            OP_LOGI(FUSED_OP_TYPE.c_str(), "dequant out data node not 1, no fusion."), return);
        node.quantNode = node.dequantNode->GetOutDataNodes().at(0);
        FUSION_PASS_CHECK(node.quantNode == nullptr,
            OP_LOGI(FUSED_OP_TYPE.c_str(), "quant is null, no fusion."), return);
        FUSION_PASS_CHECK(node.quantNode->GetType() != QUANT_TYPE,
            OP_LOGI(FUSED_OP_TYPE.c_str(), "not quant, %s.", node.quantNode->GetType().c_str()), return);
        FUSION_PASS_CHECK(node.quantNode->GetOutDataNodes().size() != 1,
            OP_LOGI(FUSED_OP_TYPE.c_str(), "quant out data node not 1, no fusion."), return);
        lastNode = node.quantNode->GetOutDataNodes().at(0);
    } else {
        node.reluRequantNode = convOut;
        FUSION_PASS_CHECK(requantPattern_ && node.reluRequantNode->GetType() != REQUANT_TYPE,
            OP_LOGI(FUSED_OP_TYPE.c_str(), "%s not requant, no fusion.", node.reluRequantNode->GetType().c_str()),
            return);
        FUSION_PASS_CHECK(!requantPattern_ && node.reluRequantNode->GetType() != RELU_TYPE,
            OP_LOGI(FUSED_OP_TYPE.c_str(), "%s not relu, no fusion.", node.reluRequantNode->GetType().c_str()),
            return);
        FUSION_PASS_CHECK(node.reluRequantNode->GetOutDataNodes().size() != 1,
            OP_LOGI(FUSED_OP_TYPE.c_str(), "relu out data node not 1, no fusion."), return);
        lastNode = node.reluRequantNode->GetOutDataNodes().at(0);
    }

    FUSION_PASS_CHECK(!CheckLastNextNode(lastNode),
        OP_LOGI(FUSED_OP_TYPE.c_str(), "check last/next node, no fusion."), return);

    fusionNodes_.emplace_back(node);
}

Status SameInputConv2dPass::GetAllPatternNodes(Mapping& mapping, ge::NodePtr inputNode)
{
    auto patternConv = GetNodeFromMapping(PATTERN_CONV2D_0, mapping);
    FUSION_PASS_CHECK(patternConv == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "conv node is null, fusion failed."), return PARAM_INVALID);

    auto quantNode = GetNodeFromMapping(PATTERN_QUANT, mapping);
    quantPattern_ = (quantNode != nullptr) ? true : false;
    auto requantNode = GetNodeFromMapping(PATTERN_RELU_REQUANT, mapping);
    requantPattern_ = (requantNode != nullptr && requantNode->GetType() == REQUANT_TYPE) ? true : false;

    GetPatternNodes(patternConv);
    FUSION_PASS_CHECK(fusionNodes_.size() < 1,
        OP_LOGI(FUSED_OP_TYPE.c_str(), "pattern no fusion."), return NOT_CHANGED);

    for (auto& convNode : inputNode->GetOutDataNodes()) {
        if (convNode == patternConv) {
            continue;
        }

        GetPatternNodes(convNode);
    }

    OP_LOGI(FUSED_OP_TYPE.c_str(), "pattern size %zu.", fusionNodes_.size());
    FUSION_PASS_CHECK(fusionNodes_.size() <= 1,
        OP_LOGI(FUSED_OP_TYPE.c_str(), "no conv nodes to fuse, no fusion."), return NOT_CHANGED);

    return SUCCESS;
}

Status SameInputConv2dPass::CheckConvNode(ge::NodePtr convNode, ge::NodePtr inputNode) const
{
    FUSION_PASS_CHECK(convNode == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "conv node is null, fusion failed."), return PARAM_INVALID);
    FUSION_PASS_CHECK(convNode->GetOpDesc() == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "conv desc is null, fusion failed."), return PARAM_INVALID);

    FUSION_PASS_CHECK(convNode->GetInDataNodes().size() < CONV2D_INPUT_SIZE_MIN,
        OP_LOGI(FUSED_OP_TYPE.c_str(), "conv input < size min, no fusion."), return NOT_CHANGED);
    FUSION_PASS_CHECK(convNode->GetOpDesc()->GetInputDesc("filter").IsValid() != GRAPH_SUCCESS,
        OP_LOGI(FUSED_OP_TYPE.c_str(), "conv has no filter, no fusion."), return NOT_CHANGED);

    FUSION_PASS_CHECK(convNode->GetInDataNodes().at(0) != inputNode,
        OP_LOGI(FUSED_OP_TYPE.c_str(), "conv has no common fmap, no fusion."), return NOT_CHANGED);

    auto filterType = convNode->GetInDataNodes().at(FILTER_POS)->GetType();
    FUSION_PASS_CHECK(filterType != CONST_TYPE && filterType != CONSTANT_TYPE
        && filterType != FILTER_HOST_TYPE && filterType != WEIGHT_QUANT && filterType != FILTER_WEIGHT_TYPE,
        OP_LOGI(FUSED_OP_TYPE.c_str(), "conv filter: %s, no fusion.", filterType.c_str()), return NOT_CHANGED);

    auto convDesc = convNode->GetOpDesc()->GetOutputDesc(0);
    auto convDim = convDesc.GetShape().GetDims();
    for (auto& dimValue : convDim) {
        FUSION_PASS_CHECK(dimValue < 0,
            OP_LOGI(FUSED_OP_TYPE.c_str(), "dynamic shape, no fusion."), return NOT_CHANGED);
    }

    return SUCCESS;
}

Status SameInputConv2dPass::CheckAllConvNodes(ge::NodePtr inputNode)
{
    std::vector<ConvFusionNodes> updateNodes;
    for (auto& node : fusionNodes_) {
        auto ret = CheckConvNode(node.convNode, inputNode);
        if (ret != SUCCESS) {
            continue;
        }

        updateNodes.emplace_back(node);
    }

    FUSION_PASS_CHECK(updateNodes.size() <= 1,
        OP_LOGI(FUSED_OP_TYPE.c_str(), "no conv nodes to fuse, no fusion."), return NOT_CHANGED);
    FUSION_PASS_CHECK(updateNodes[0].convNode != fusionNodes_[0].convNode,
        OP_LOGI(FUSED_OP_TYPE.c_str(), "invalid pattern conv, no fusion."), return NOT_CHANGED);

    int64_t dimValue = 0;
    FUSION_PASS_CHECK(GetConvOutDimValue(dimValue) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get conv out dim failed, fusion failed."), return FAILED);
    auto convDesc = updateNodes.at(0).convNode->GetOpDesc()->GetInputDesc(0);
    uint32_t align = (convDesc.GetDataType() == DT_INT8)? INT8_ALIGN : DATA_ALIGN;
    OP_LOGI(FUSED_OP_TYPE.c_str(), "align %u, type %u.", align, convDesc.GetDataType());
    FUSION_PASS_CHECK(dimValue % align != 0,
        OP_LOGI(FUSED_OP_TYPE.c_str(), "conv cout not aligned, no fusion."), return NOT_CHANGED);

    fusionNodes_.swap(updateNodes);

    return SUCCESS;
}

Status SameInputConv2dPass::CheckDequantNodes()
{
    FUSION_PASS_CHECK(!quantPattern_, OP_LOGI(FUSED_OP_TYPE.c_str(), "not dequant."), return SUCCESS);
    std::vector<ConvFusionNodes> updateNodes;
    int32_t dstType = 0;
    bool sqrtMode = false;
    bool reluFlag = false;
    for (size_t i = 0; i < fusionNodes_.size(); ++i) {
        ge::NodePtr dequantNode = fusionNodes_[i].dequantNode;
        FUSION_PASS_CHECK(dequantNode == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "dequant node is null, fusion failed."), return PARAM_INVALID);
        FUSION_PASS_CHECK(dequantNode->GetOpDesc() == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "dequant node desc is null, fusion failed."), return PARAM_INVALID);

        int32_t nodeDstType = 0;
        bool nodeSqrtMode = false;
        bool nodeReluFlag = false;
        ge::AttrUtils::GetInt(dequantNode->GetOpDesc(), "dtype", nodeDstType);
        ge::AttrUtils::GetBool(dequantNode->GetOpDesc(), "sqrt_mode", nodeSqrtMode);
        ge::AttrUtils::GetBool(dequantNode->GetOpDesc(), "relu_flag", nodeReluFlag);
        OP_LOGI(FUSED_OP_TYPE.c_str(), "dequant dtype %u, mode %u, %u.", nodeDstType, nodeSqrtMode, nodeReluFlag);
        if (i == 0) {
            dstType = nodeDstType;
            sqrtMode = nodeSqrtMode;
            reluFlag = nodeReluFlag;
        }

        if ((dstType != nodeDstType) || (sqrtMode != nodeSqrtMode) || (reluFlag != nodeReluFlag)) {
            continue;
        }

        updateNodes.emplace_back(fusionNodes_[i]);
    }
    FUSION_PASS_CHECK(updateNodes.size() <= 1,
        OP_LOGI(FUSED_OP_TYPE.c_str(), "dequant check no fusion."), return NOT_CHANGED);
    fusionNodes_.swap(updateNodes);

    return SUCCESS;
}

Status SameInputConv2dPass::CheckQuantNodes()
{
    FUSION_PASS_CHECK(!quantPattern_, OP_LOGI(FUSED_OP_TYPE.c_str(), "not quant."), return SUCCESS);
    std::vector<ConvFusionNodes> updateNodes;
    int32_t dstType = 0;
    float offset = 0.0;
    float scale = 0.0;
    bool sqrtMode = false;
    std::string roundMode;
    for (size_t i = 0; i < fusionNodes_.size(); ++i) {
        ge::NodePtr quantNode = fusionNodes_[i].quantNode;
        FUSION_PASS_CHECK(quantNode == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "quant node is null, fusion failed."), return PARAM_INVALID);
        FUSION_PASS_CHECK(quantNode->GetOpDesc() == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "quant node desc is null, fusion failed."), return PARAM_INVALID);

        int32_t nodeDstType = 0;
        float nodeOffset = 0.0;
        float nodeScale = 0.0;
        bool nodeSqrtMode = false;
        std::string nodeRoundMode;
        ge::AttrUtils::GetInt(quantNode->GetOpDesc(), "dst_type", nodeDstType);
        ge::AttrUtils::GetFloat(quantNode->GetOpDesc(), "offset", nodeOffset);
        ge::AttrUtils::GetFloat(quantNode->GetOpDesc(), "scale", nodeScale);
        ge::AttrUtils::GetBool(quantNode->GetOpDesc(), "sqrt_mode", nodeSqrtMode);
        ge::AttrUtils::GetStr(quantNode->GetOpDesc(), "round_mode", nodeRoundMode);
        if (i == 0) {
            dstType = nodeDstType;
            offset = nodeOffset;
            scale = nodeScale;
            sqrtMode = nodeSqrtMode;
            roundMode = nodeRoundMode;
        }

        if ((dstType != nodeDstType) || (sqrtMode != nodeSqrtMode) || (roundMode != nodeRoundMode)) {
            continue;
        }
        if (abs(nodeOffset - offset) > DELTA || abs(nodeScale - scale) > DELTA) {
            continue;
        }

        updateNodes.emplace_back(fusionNodes_[i]);
    }
    FUSION_PASS_CHECK(updateNodes.size() <= 1,
        OP_LOGI(FUSED_OP_TYPE.c_str(), "quant check no fusion."), return NOT_CHANGED);
    fusionNodes_.swap(updateNodes);

    return SUCCESS;
}

ConvFusionAttr SameInputConv2dPass::GetConvAttr(ge::NodePtr node) const
{
    ConvFusionAttr attr;
    ge::AttrUtils::GetListInt(node->GetOpDesc(), "pads", attr.pads);
    ge::AttrUtils::GetListInt(node->GetOpDesc(), "strides", attr.strides);
    ge::AttrUtils::GetListInt(node->GetOpDesc(), "dilations", attr.dilations);
    ge::AttrUtils::GetInt(node->GetOpDesc(), "groups", attr.groups);
    ge::AttrUtils::GetStr(node->GetOpDesc(), "data_format", attr.format);
    ge::AttrUtils::GetInt(node->GetOpDesc(), "offset_x", attr.offsetX);

    return attr;
}

Status SameInputConv2dPass::GetConvInput(ge::NodePtr conv, ConvFusionInput& convInput) const
{
    auto filterDesc = conv->GetOpDesc()->GetInputDesc("filter");
    FUSION_PASS_CHECK(filterDesc.IsValid() != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "invalid filter desc, fusion failed."), return PARAM_INVALID);
    auto shape = filterDesc.GetShape().GetDims();
    FUSION_PASS_CHECK(shape.size() != FILTER_SHAPE_SIZE,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "invalid filter shape dim %zu, fusion failed.", shape.size()),
        return PARAM_INVALID);

    std::string format = TypeUtils::FormatToSerialString(filterDesc.GetFormat());
    size_t fondH = format.find('H');
    size_t fondW = format.find('W');
    FUSION_PASS_CHECK(fondH >= shape.size() || fondW >= shape.size(),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "invalid format, fusion failed."), return PARAM_INVALID);
    convInput.kernel.emplace_back(shape[fondH]);
    convInput.kernel.emplace_back(shape[fondW]);
    convInput.format = format;

    auto biasDesc = conv->GetOpDesc()->GetInputDesc("bias");
    if (biasDesc.IsValid() == GRAPH_SUCCESS) {
        convInput.bias = biasDesc.GetShape().GetDims();
        FUSION_PASS_CHECK(convInput.bias.size() != BIAS_SHAPE_SIZE,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "invalid bais shape, fusion failed."), return PARAM_INVALID);
    }

    auto offsetDesc = conv->GetOpDesc()->GetInputDesc("offset_w");
    if (offsetDesc.IsValid() == GRAPH_SUCCESS) {
        convInput.offset = offsetDesc.GetShape().GetDims();
    }

    return SUCCESS;
}

Status SameInputConv2dPass::CheckConvAttr()
{
    auto convNode = fusionNodes_.at(0).convNode;
    ConvFusionAttr attr = GetConvAttr(convNode);
    ConvFusionInput input;
    auto ret = GetConvInput(convNode, input);
    if (ret != SUCCESS) {
        return ret;
    }

    std::vector<ConvFusionNodes> fusionNodes {fusionNodes_.at(0)};
    for (size_t i = 1; i < fusionNodes_.size(); ++i) {
        auto node = fusionNodes_[i].convNode;
        ConvFusionAttr attrCheck = GetConvAttr(node);
        if (attrCheck != attr) {
            continue;
        }

        ConvFusionInput inputCheck;
        ret = GetConvInput(node, inputCheck);
        if (ret != SUCCESS) {
            return ret;
        }
        if (inputCheck != input) {
            continue;
        }

        fusionNodes.emplace_back(fusionNodes_[i]);
    }
    FUSION_PASS_CHECK(fusionNodes.size() <= 1,
        OP_LOGI(FUSED_OP_TYPE.c_str(), "conv attr check no fusion."), return NOT_CHANGED);

    fusionNodes_.swap(fusionNodes);

    return SUCCESS;
}

Status SameInputConv2dPass::GetSplitAttr(const std::vector<ge::NodePtr>& preNodes, std::vector<int32_t>& sizeSplits,
    int32_t& splitDim) const
{
    auto preOutDesc = preNodes.at(0)->GetOpDesc()->GetOutputDesc(0);
    std::string format = TypeUtils::FormatToSerialString(preOutDesc.GetFormat());
    size_t axis = format.find('C');
    FUSION_PASS_CHECK(axis == std::string::npos,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "invalid split axis, fusion failed."), return PARAM_INVALID);

    for (auto& preNode : preNodes) {
        auto preOutDesc = preNode->GetOpDesc()->GetOutputDesc(0);
        auto preDim = preOutDesc.GetShape().GetDims();
        FUSION_PASS_CHECK(axis >= preDim.size(),
            OP_LOGE(FUSED_OP_TYPE.c_str(), "invalid split shape, fusion failed."), return PARAM_INVALID);
        sizeSplits.emplace_back(preDim[axis]);
    }

    splitDim = static_cast<int32_t>(axis);

    return SUCCESS;
}

Status SameInputConv2dPass::GetConvInDimValue(int64_t& dimValue) const
{
    auto filterDesc = fusionNodes_.at(0).convNode->GetOpDesc()->GetInputDesc(FILTER_POS);
    std::string format = TypeUtils::FormatToSerialString(filterDesc.GetFormat());
    size_t axis = format.find('N');
    FUSION_PASS_CHECK(axis == std::string::npos,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get conv axis failed, fusion failed."), return PARAM_INVALID);

    dimValue = 0;
    for (auto& node : fusionNodes_) {
        auto filterDesc = node.convNode->GetOpDesc()->GetInputDesc(FILTER_POS);
        auto filterDim = filterDesc.GetShape().GetDims();
        FUSION_PASS_CHECK(axis >= filterDim.size(),
            OP_LOGE(FUSED_OP_TYPE.c_str(), "invalid filter shape, fusion failed."), return PARAM_INVALID);
        dimValue += filterDim[axis];
    }

    return SUCCESS;
}

Status SameInputConv2dPass::GetConvOutDimValue(int64_t& dimValue) const
{
    auto convDesc = fusionNodes_.at(0).convNode->GetOpDesc()->GetOutputDesc(0);
    std::string format = TypeUtils::FormatToSerialString(convDesc.GetFormat());
    size_t axis = format.find('C');
    FUSION_PASS_CHECK(axis == std::string::npos,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get conv out axis failed, fusion failed."), return PARAM_INVALID);

    dimValue = 0;
    for (auto& node : fusionNodes_) {
        auto convDesc = node.convNode->GetOpDesc()->GetOutputDesc(0);
        auto convDim = convDesc.GetShape().GetDims();
        FUSION_PASS_CHECK(axis >= convDim.size(),
            OP_LOGE(FUSED_OP_TYPE.c_str(), "invalid conv shape, fusion failed."), return PARAM_INVALID);
        dimValue += convDim[axis];
    }

    return SUCCESS;
}

Status SameInputConv2dPass::GetReluDimValue(int64_t& dimValue) const
{
    auto reluOutDesc = fusionNodes_.at(0).reluRequantNode->GetOpDesc()->GetOutputDesc(0);
    std::string format = TypeUtils::FormatToSerialString(reluOutDesc.GetFormat());
    size_t axis = format.find('C');
    FUSION_PASS_CHECK(axis == std::string::npos,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get relu axis failed, fusion failed."), return PARAM_INVALID);

    dimValue = 0;
    for (auto& node : fusionNodes_) {
        auto reluDesc = node.reluRequantNode->GetOpDesc()->GetOutputDesc(0);
        auto reluDim = reluDesc.GetShape().GetDims();
        FUSION_PASS_CHECK(axis >= reluDim.size(),
            OP_LOGE(FUSED_OP_TYPE.c_str(), "invalid relu shape, fusion failed."), return PARAM_INVALID);
        dimValue += reluDim[axis];
    }

    return SUCCESS;
}

Status SameInputConv2dPass::GetDequantDimValue(int64_t& dimValue) const
{
    auto scaleDesc = fusionNodes_.at(0).dequantNode->GetOpDesc()->GetInputDesc(1);
    std::string format = TypeUtils::FormatToSerialString(scaleDesc.GetFormat());
    size_t axis = format.find('N');
    FUSION_PASS_CHECK(axis == std::string::npos,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get scale axis failed, fusion failed."), return PARAM_INVALID);

    dimValue = 0;
    for (auto& node : fusionNodes_) {
        auto scaleDesc = node.dequantNode->GetOpDesc()->GetInputDesc(1);
        auto dim = scaleDesc.GetShape().GetDims();
        FUSION_PASS_CHECK(axis >= dim.size(),
            OP_LOGE(FUSED_OP_TYPE.c_str(), "invalid dequant shape, fusion failed."), return PARAM_INVALID);
        dimValue += dim[axis];
    }

    return SUCCESS;
}

Status SameInputConv2dPass::GetQuantDimValue(int64_t& dimValue) const
{
    auto quantDesc = fusionNodes_.at(0).quantNode->GetOpDesc()->GetOutputDesc(0);
    std::string format = TypeUtils::FormatToSerialString(quantDesc.GetFormat());
    size_t axis = format.find('C');
    FUSION_PASS_CHECK(axis == std::string::npos,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get quant axis failed, fusion failed."), return PARAM_INVALID);

    dimValue = 0;
    for (auto& node : fusionNodes_) {
        auto quantDesc = node.dequantNode->GetOpDesc()->GetOutputDesc(0);
        auto dim = quantDesc.GetShape().GetDims();
        FUSION_PASS_CHECK(axis >= dim.size(),
            OP_LOGE(FUSED_OP_TYPE.c_str(), "invalid quant shape, fusion failed."), return PARAM_INVALID);
        dimValue += dim[axis];
    }

    return SUCCESS;
}

Status SameInputConv2dPass::GetRequantDimValue(int64_t& dimValue) const
{
    auto scaleDesc = fusionNodes_.at(0).reluRequantNode->GetOpDesc()->GetInputDesc(1);
    std::string format = TypeUtils::FormatToSerialString(scaleDesc.GetFormat());
    size_t axis = format.find('N');
    FUSION_PASS_CHECK(axis == std::string::npos,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get scale axis failed, fusion failed."), return PARAM_INVALID);

    dimValue = 0;
    for (auto& node : fusionNodes_) {
        auto scaleDesc = node.reluRequantNode->GetOpDesc()->GetInputDesc(1);
        auto dim = scaleDesc.GetShape().GetDims();
        FUSION_PASS_CHECK(axis >= dim.size(),
            OP_LOGE(FUSED_OP_TYPE.c_str(), "invalid requant shape, fusion failed."), return PARAM_INVALID);
        dimValue += dim[axis];
    }

    return SUCCESS;
}

Status SameInputConv2dPass::GetConvInAxis(ge::NodePtr convNode, int32_t &axis) const
{
    auto convInFilterDesc = convNode->GetOpDesc()->GetInputDesc(FILTER_POS);
    std::string format = TypeUtils::FormatToSerialString(convInFilterDesc.GetFormat());
    size_t coutAxis = format.find('N');
    FUSION_PASS_CHECK(coutAxis == std::string::npos,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get filter axis failed %s, fusion failed.", format.c_str()),
        return PARAM_INVALID);

    axis = static_cast<int32_t>(coutAxis);

    OP_LOGI(FUSED_OP_TYPE.c_str(), "conv in axis %d.", axis);
    return SUCCESS;
}

Status SameInputConv2dPass::GetNodeCoutAxis(const ge::GeTensorDesc& nodeDesc, int32_t &axis) const
{
    std::string format = TypeUtils::FormatToSerialString(nodeDesc.GetFormat());
    size_t coutAxis = format.find('C');
    FUSION_PASS_CHECK(coutAxis == std::string::npos,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get axis failed, fusion failed."), return PARAM_INVALID);

    axis = static_cast<int32_t>(coutAxis);
    OP_LOGI(FUSED_OP_TYPE.c_str(), "node axis %d.", axis);

    return SUCCESS;
}

ge::NodePtr SameInputConv2dPass::CreateSizeSplitNode(ge::ComputeGraph& graph, std::vector<ge::NodePtr>& newNodes,
    const std::vector<int32_t>& sizeSplits, const std::string& name) const
{
    // create const size_split op
    int64_t shapeValue = static_cast<int64_t>(sizeSplits.size());
    ge::GeShape sizeSplitShape = ge::GeShape({shapeValue});
    auto sizeSplitDesc = ge::GeTensorDesc(sizeSplitShape, ge::FORMAT_ND, ge::DT_INT32);
    sizeSplitDesc.SetOriginShape(sizeSplitShape);
    sizeSplitDesc.SetOriginFormat(ge::FORMAT_ND);
    sizeSplitDesc.SetOriginDataType(ge::DT_INT32);
    sizeSplitDesc.SetShape(sizeSplitShape);
    sizeSplitDesc.SetFormat(ge::FORMAT_ND);
    sizeSplitDesc.SetDataType(ge::DT_INT32);

    ge::GeTensorPtr sizeSplitTensorPtr = nullptr;
    FUSION_PASS_MAKE_SHARED(sizeSplitTensorPtr = std::make_shared<ge::GeTensor>(sizeSplitDesc), return nullptr);
    auto data = reinterpret_cast<const uint8_t*>(sizeSplits.data());
    size_t len = sizeSplits.size() * sizeof(int32_t);
    FUSION_PASS_CHECK(sizeSplitTensorPtr->SetData(data, len) != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "size_split set data failed, fusion failed."), return nullptr);

    auto sizeSplitOpDesc = ge::OpDescUtils::CreateConstOp(sizeSplitTensorPtr);
    FUSION_PASS_CHECK(sizeSplitOpDesc == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "create const op size split failed, fusion failed."), return nullptr);
    sizeSplitOpDesc->SetName(name);

    auto sizeSplitNode = graph.AddNode(sizeSplitOpDesc);
    FUSION_PASS_CHECK(sizeSplitNode == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add size_split node failed, fusion failed."), return nullptr);
    newNodes.emplace_back(sizeSplitNode);
    sizeSplitNode->GetOpDesc()->UpdateOutputDesc(0, sizeSplitDesc);

    return sizeSplitNode;
}

ge::NodePtr SameInputConv2dPass::CreateDimSplitNode(ge::ComputeGraph& graph, std::vector<ge::NodePtr>& newNodes,
    int32_t splitDim, const std::string& name) const
{
    // create const split_dim op, shape is 1
    ge::GeShape splitDimShape = ge::GeShape({1});
    auto splitDimDesc = ge::GeTensorDesc(splitDimShape, ge::FORMAT_ND, ge::DT_INT32);
    splitDimDesc.SetOriginShape(splitDimShape);
    splitDimDesc.SetOriginFormat(ge::FORMAT_ND);
    splitDimDesc.SetOriginDataType(ge::DT_INT32);
    splitDimDesc.SetShape(splitDimShape);
    splitDimDesc.SetFormat(ge::FORMAT_ND);
    splitDimDesc.SetDataType(ge::DT_INT32);

    ge::GeTensorPtr tensorPtr = nullptr;
    FUSION_PASS_MAKE_SHARED(tensorPtr = std::make_shared<ge::GeTensor>(splitDimDesc), return nullptr);
    FUSION_PASS_CHECK(tensorPtr->SetData(reinterpret_cast<uint8_t*>(&splitDim), sizeof(int32_t)) != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "dim_split set data failed, fusion failed."), return nullptr);

    auto dimSplitOpDesc = ge::OpDescUtils::CreateConstOp(tensorPtr);
    FUSION_PASS_CHECK(dimSplitOpDesc == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "create const op split dim failed, fusion failed."), return nullptr);
    dimSplitOpDesc->SetName(name);

    auto dimSplitNode = graph.AddNode(dimSplitOpDesc);
    FUSION_PASS_CHECK(dimSplitNode == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add size_split node failed, fusion failed."), return nullptr);
    newNodes.emplace_back(dimSplitNode);
    dimSplitNode->GetOpDesc()->UpdateOutputDesc(0, splitDimDesc);

    return dimSplitNode;
}

ge::NodePtr SameInputConv2dPass::CreateSplitNode(ge::ComputeGraph& graph, std::vector<ge::NodePtr>& newNodes,
    std::vector<ge::NodePtr>& reluNodes, ge::NodePtr& sizeSplitNode, ge::NodePtr& dimSplitNode) const
{
    ge::OpDescPtr splitOpDesc = nullptr;
    std::string splitName = reluNodes.at(0)->GetName() + SPLIT;
    FUSION_PASS_MAKE_SHARED(splitOpDesc = std::make_shared<ge::OpDesc>(splitName, SPLIT_TYPE), return nullptr);

    std::vector<int32_t> sizeSplits;
    int32_t splitDim = 0;
    FUSION_PASS_CHECK(GetSplitAttr(reluNodes, sizeSplits, splitDim) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get split attr failed, fusion failed."), return nullptr);
    int64_t dimValue = std::accumulate(sizeSplits.begin(), sizeSplits.end(), 0);

    auto convOutDesc = reluNodes.at(0)->GetOpDesc()->GetOutputDesc(0).Clone();
    FUSION_PASS_CHECK(SetShapeDims(splitDim, dimValue, convOutDesc) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "split set dims failed, fusion failed."), return nullptr);
    FUSION_PASS_CHECK(splitOpDesc->AddInputDesc("x", convOutDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "split add input desc failed, fusion failed."), return nullptr);

    for (size_t i = 0; i < reluNodes.size(); ++i) {
        std::string outName = "y" + to_string(i);
        auto reluNode = reluNodes[i]->GetOutDataNodes().at(0);
        FUSION_PASS_CHECK(splitOpDesc->AddOutputDesc(outName, reluNode->GetOpDesc()->GetInputDesc(0)) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "split add output desc failed, fusion failed."), return nullptr);
    }
    splitOpDesc->AddInputDesc("size_splits", sizeSplitNode->GetOpDesc()->GetOutputDesc(0));
    splitOpDesc->AddInputDesc("split_dim", dimSplitNode->GetOpDesc()->GetOutputDesc(0));
    FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(splitOpDesc, "num_split", sizeSplits.size()),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add size_split node failed, fusion failed."), return nullptr);

    auto splitNode = graph.AddNode(splitOpDesc);
    FUSION_PASS_CHECK(splitNode == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add size_split node failed, fusion failed."), return nullptr);
    newNodes.emplace_back(splitNode);

    return splitNode;
}

Status SameInputConv2dPass::LinkSplitConst(ge::NodePtr sizeSplitNode, ge::NodePtr dimSplitNode,
    ge::NodePtr splitNode) const
{
    // add edge split-size_splits
    auto splitInSizeAnchor = splitNode->GetInDataAnchor(SIZE_SPLIT_POS);
    FUSION_PASS_CHECK(splitInSizeAnchor == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "split in data anchor 1 is null, fusion failed."), return FAILED);
    auto sizeSplitOutAnchor = sizeSplitNode->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(sizeSplitOutAnchor == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "size split out data anchor is null, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(sizeSplitOutAnchor, splitInSizeAnchor),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from split--size failed, fusion failed."), return FAILED);

    // add edge split-split_dim
    auto splitInDimAnchor = splitNode->GetInDataAnchor(DIM_SPLIT_POS);
    FUSION_PASS_CHECK(splitInDimAnchor == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "split in data anchor 2 is null, fusion failed."), return FAILED);
    auto dimSplitOutAnchor = dimSplitNode->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(dimSplitOutAnchor == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "dim split out data anchor is null, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(dimSplitOutAnchor, splitInDimAnchor),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from split--dim failed, fusion failed."), return FAILED);

    return SUCCESS;
}

Status SameInputConv2dPass::AddSplitNode(ge::ComputeGraph& graph, std::vector<ge::NodePtr>& newNodes) const
{
    std::vector<ge::NodePtr> preSplitNodes;
    for (auto& node : fusionNodes_) {
        if (quantPattern_) {
            preSplitNodes.emplace_back(node.quantNode);
        } else {
            preSplitNodes.emplace_back(node.reluRequantNode);
        }
    }

    std::vector<int32_t> sizeSplits;
    int32_t splitDim;
    FUSION_PASS_CHECK(GetSplitAttr(preSplitNodes, sizeSplits, splitDim) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get split attr failed, fusion failed."), return FAILED);

    std::string sizeSplitName = preSplitNodes.at(0)->GetName() + SPLIT_SIZE_CONST;
    ge::NodePtr sizeSplitNode = CreateSizeSplitNode(graph, newNodes, sizeSplits, sizeSplitName);
    FUSION_PASS_CHECK(sizeSplitNode == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "create size_splits node failed, fusion failed."), return FAILED);

    std::string dimSplitName = preSplitNodes.at(0)->GetName() + SPLIT_DIM_CONST;
    ge::NodePtr dimSplitNode = CreateDimSplitNode(graph, newNodes, splitDim, dimSplitName);
    FUSION_PASS_CHECK(dimSplitNode == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "create split_dim node failed, fusion failed."), return FAILED);

    ge::NodePtr splitNode = CreateSplitNode(graph, newNodes, preSplitNodes, sizeSplitNode, dimSplitNode);
    FUSION_PASS_CHECK(splitNode == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "create split node failed, fusion failed."), return FAILED);

    FUSION_PASS_CHECK(LinkSplitConst(sizeSplitNode, dimSplitNode, splitNode) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "link split const failed, fusion failed."), return FAILED);

    FUSION_PASS_CHECK(LinkReluSplit(graph, preSplitNodes, splitNode) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "link relu split failed, fusion failed."), return FAILED);

    FUSION_PASS_CHECK(AddStrideNode(graph, newNodes, splitNode) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add stride failed, fusion failed."), return FAILED);

    OP_LOGI(FUSED_OP_TYPE.c_str(), "add split node success.");
    return SUCCESS;
}

Status SameInputConv2dPass::UpdateConvShape() const
{
    ge::NodePtr updateConvNode = fusionNodes_.at(0).convNode;

    // update conv shape
    int64_t dimOutValue;
    FUSION_PASS_CHECK(GetConvOutDimValue(dimOutValue) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get conv out dim failed, fusion failed."), return FAILED);
    int32_t convOutAxis;
    FUSION_PASS_CHECK(GetNodeCoutAxis(updateConvNode->GetOpDesc()->GetOutputDesc(0), convOutAxis) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get conv out axis failed, fusion failed."), return FAILED);
    auto convOutDesc = updateConvNode->GetOpDesc()->GetOutputDesc(0).Clone();
    FUSION_PASS_CHECK(SetShapeDims(convOutAxis, dimOutValue, convOutDesc) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "set conv out dim failed, fusion failed."), return FAILED);
    updateConvNode->GetOpDesc()->UpdateOutputDesc(0, convOutDesc);

    int64_t dimInValue;
    FUSION_PASS_CHECK(GetConvInDimValue(dimInValue) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get conv in dim failed, fusion failed."), return FAILED);
    auto convInFilterDesc = updateConvNode->GetOpDesc()->GetInputDesc("filter").Clone();
    int32_t concatAxis;
    FUSION_PASS_CHECK(GetConvInAxis(updateConvNode, concatAxis) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get filter axis failed, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(SetShapeDims(concatAxis, dimInValue, convInFilterDesc) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "set conv in dim failed, fusion failed."), return FAILED);
    updateConvNode->GetOpDesc()->UpdateInputDesc(FILTER_POS, convInFilterDesc);

    auto convInBiasDesc = updateConvNode->GetOpDesc()->GetInputDesc("bias").Clone();
    if (convInBiasDesc.IsValid() == GRAPH_SUCCESS) {
        FUSION_PASS_CHECK(SetShapeDims(0, dimInValue, convInBiasDesc) != SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "set shape dim failed, fusion failed."), return FAILED);
        updateConvNode->GetOpDesc()->UpdateInputDesc(BIAS_POS, convInBiasDesc);
    }

    return SUCCESS;
}

Status SameInputConv2dPass::UpdateReluShape() const
{
    FUSION_PASS_CHECK(requantPattern_ || quantPattern_, OP_LOGI(FUSED_OP_TYPE.c_str(), "no relu."), return SUCCESS);

    // update relu shape
    ge::NodePtr updateReluNode = fusionNodes_.at(0).reluRequantNode;

    int64_t dimValue;
    FUSION_PASS_CHECK(GetReluDimValue(dimValue) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get relu dim failed, fusion failed."), return FAILED);

    auto reluInDesc = updateReluNode->GetOpDesc()->GetInputDesc(0).Clone();
    int32_t reluInAxis;
    FUSION_PASS_CHECK(GetNodeCoutAxis(reluInDesc, reluInAxis) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get relu axis failed, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(SetShapeDims(reluInAxis, dimValue, reluInDesc) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "set relu shape dim failed, fusion failed."), return FAILED);
    updateReluNode->GetOpDesc()->UpdateInputDesc(0, reluInDesc);

    auto reluOutDesc = updateReluNode->GetOpDesc()->GetOutputDesc(0).Clone();
    int32_t reluOutAxis;
    FUSION_PASS_CHECK(GetNodeCoutAxis(reluOutDesc, reluOutAxis) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get relu axis failed, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(SetShapeDims(reluOutAxis, dimValue, reluOutDesc) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "set relu shape dim failed, fusion failed."), return FAILED);
    updateReluNode->GetOpDesc()->UpdateOutputDesc(0, reluOutDesc);

    return SUCCESS;
}

Status SameInputConv2dPass::UpdateDequantShape() const
{
    FUSION_PASS_CHECK(!quantPattern_, OP_LOGI(FUSED_OP_TYPE.c_str(), "no dequant."), return SUCCESS);

    // update dequant shape
    ge::NodePtr updateDequantNode = fusionNodes_.at(0).dequantNode;

    int64_t dimValue;
    FUSION_PASS_CHECK(GetDequantDimValue(dimValue) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get value failed, fusion failed."), return FAILED);

    auto dequantInDesc = updateDequantNode->GetOpDesc()->GetInputDesc(0).Clone();
    int32_t dequantInAxis;
    FUSION_PASS_CHECK(GetNodeCoutAxis(dequantInDesc, dequantInAxis) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get concat axis failed, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(SetShapeDims(dequantInAxis, dimValue, dequantInDesc) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "set shape dim failed, fusion failed."), return FAILED);
    updateDequantNode->GetOpDesc()->UpdateInputDesc(0, dequantInDesc);

    auto deqScaleDesc = updateDequantNode->GetOpDesc()->GetInputDesc(1).Clone();
    FUSION_PASS_CHECK(SetShapeDims(0, dimValue, deqScaleDesc) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "set deq dim failed, fusion failed."), return FAILED);
    updateDequantNode->GetOpDesc()->UpdateInputDesc(1, deqScaleDesc);

    auto dequantOutDesc = updateDequantNode->GetOpDesc()->GetOutputDesc(0).Clone();
    int32_t dequantOutAxis;
    FUSION_PASS_CHECK(GetNodeCoutAxis(dequantOutDesc, dequantOutAxis) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get concat axis failed, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(SetShapeDims(dequantOutAxis, dimValue, dequantOutDesc) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "set shape dim failed, fusion failed."), return FAILED);
    updateDequantNode->GetOpDesc()->UpdateOutputDesc(0, dequantOutDesc);

    return SUCCESS;
}

Status SameInputConv2dPass::UpdateQuantShape() const
{
    FUSION_PASS_CHECK(!quantPattern_, OP_LOGI(FUSED_OP_TYPE.c_str(), "no quant."), return SUCCESS);

    // update quant shape
    int64_t dimValue;
    FUSION_PASS_CHECK(GetQuantDimValue(dimValue) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get value failed, fusion failed."), return FAILED);

    ge::NodePtr updateQuantNode = fusionNodes_.at(0).quantNode;
    auto quantInDesc = updateQuantNode->GetOpDesc()->GetInputDesc(0).Clone();
    int32_t quantInAxis;
    FUSION_PASS_CHECK(GetNodeCoutAxis(quantInDesc, quantInAxis) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get axis failed, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(SetShapeDims(quantInAxis, dimValue, quantInDesc) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "set shape dim failed, fusion failed."), return FAILED);
    updateQuantNode->GetOpDesc()->UpdateInputDesc(0, quantInDesc);

    auto quantOutDesc = updateQuantNode->GetOpDesc()->GetOutputDesc(0).Clone();
    int32_t quantOutAxis;
    FUSION_PASS_CHECK(GetNodeCoutAxis(quantOutDesc, quantOutAxis) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get axis failed, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(SetShapeDims(quantOutAxis, dimValue, quantOutDesc) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "set shape dim failed, fusion failed."), return FAILED);
    updateQuantNode->GetOpDesc()->UpdateOutputDesc(0, quantOutDesc);

    return SUCCESS;
}

Status SameInputConv2dPass::UpdateRequantShape() const
{
    FUSION_PASS_CHECK(!requantPattern_, OP_LOGI(FUSED_OP_TYPE.c_str(), "no requant."), return SUCCESS);

    // update quant shape
    int64_t dimValue;
    FUSION_PASS_CHECK(GetRequantDimValue(dimValue) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get requant dim failed, fusion failed."), return FAILED);

    ge::NodePtr updateRequantNode = fusionNodes_.at(0).reluRequantNode;
    auto requantInDesc = updateRequantNode->GetOpDesc()->GetInputDesc(0).Clone();
    int32_t requantInAxis;
    FUSION_PASS_CHECK(GetNodeCoutAxis(requantInDesc, requantInAxis) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get requant axis failed, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(SetShapeDims(requantInAxis, dimValue, requantInDesc) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "set requant dim failed, fusion failed."), return FAILED);
    updateRequantNode->GetOpDesc()->UpdateInputDesc(0, requantInDesc);

    auto reqScaleDesc = updateRequantNode->GetOpDesc()->GetInputDesc(1).Clone();
    FUSION_PASS_CHECK(SetShapeDims(0, dimValue, reqScaleDesc) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "set reqScale dim failed, fusion failed."), return FAILED);
    updateRequantNode->GetOpDesc()->UpdateInputDesc(1, reqScaleDesc);

    auto requantOutDesc = updateRequantNode->GetOpDesc()->GetOutputDesc(0).Clone();
    int32_t requantOutAxis;
    FUSION_PASS_CHECK(GetNodeCoutAxis(requantOutDesc, requantOutAxis) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get req out axis failed, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(SetShapeDims(requantOutAxis, dimValue, requantOutDesc) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "set req out dim failed, fusion failed."), return FAILED);
    updateRequantNode->GetOpDesc()->UpdateOutputDesc(0, requantOutDesc);

    return SUCCESS;
}

Status SameInputConv2dPass::UpdateConvEdge() const
{
    // remove edge fmap--conv
    for (size_t i = 1; i < fusionNodes_.size(); ++i) {
        auto fmapInAnchor = fusionNodes_[i].convNode->GetInDataAnchor(0);
        FUSION_PASS_CHECK(fmapInAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "fmap anchor is null, fusion failed."), return FAILED);
        auto fmapPeerAnchor = fmapInAnchor->GetPeerOutAnchor();
        FUSION_PASS_CHECK(fmapPeerAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "fmap peer out data anchor is null, fusion failed."), return FAILED);
        FUSION_PASS_CHECK(GraphUtils::RemoveEdge(fmapPeerAnchor, fmapInAnchor) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "remove edge from fmap--conv failed, fusion failed."), return FAILED);
    }

    return SUCCESS;
}

Status SameInputConv2dPass::UpdateReluEdge(ge::ComputeGraph& graph) const
{
    // remove edge conv-relu/requant/dequant
    for (size_t i = 1; i < fusionNodes_.size(); ++i) {
        auto removeNode = quantPattern_ ? fusionNodes_[i].dequantNode : fusionNodes_[i].reluRequantNode;
        auto removeInAnchor = removeNode->GetInDataAnchor(0);
        FUSION_PASS_CHECK(removeInAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "relu/requant in anchor is null, fusion failed."), return FAILED);
        auto removePeerOutAnchor = removeInAnchor->GetPeerOutAnchor();
        FUSION_PASS_CHECK(removePeerOutAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "relu/quant peer out data anchor is null, fusion failed."), return FAILED);
        FUSION_PASS_CHECK(GraphUtils::RemoveEdge(removePeerOutAnchor, removeInAnchor) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "remove edge conv-relu/quant failed, fusion failed."), return FAILED);

        FUSION_PASS_CHECK(graph.RemoveNode(fusionNodes_[i].convNode) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "remove conv node failed, fusion failed."), return FAILED);
    }

    return SUCCESS;
}

Status SameInputConv2dPass::UpdateDequantEdge(ge::ComputeGraph& graph) const
{
    // remove edge dequant-quant
    FUSION_PASS_CHECK(!quantPattern_, OP_LOGI(FUSED_OP_TYPE.c_str(), "no dequant."), return SUCCESS);

    for (size_t i = 1; i < fusionNodes_.size(); ++i) {
        auto quantInAnchor = fusionNodes_[i].quantNode->GetInDataAnchor(0);
        FUSION_PASS_CHECK(quantInAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "quant in anchor is null, fusion failed."), return FAILED);
        auto quantPeerOutAnchor = quantInAnchor->GetPeerOutAnchor();
        FUSION_PASS_CHECK(quantPeerOutAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "quant peer out data anchor is null, fusion failed."), return FAILED);
        FUSION_PASS_CHECK(GraphUtils::RemoveEdge(quantPeerOutAnchor, quantInAnchor) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "remove edge from input--conv failed, fusion failed."), return FAILED);

        FUSION_PASS_CHECK(graph.RemoveNode(fusionNodes_[i].dequantNode) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "remove conv node failed, fusion failed."), return FAILED);
    }

    return SUCCESS;
}

Status SameInputConv2dPass::UpdateQuantEdge(ge::ComputeGraph& graph) const
{
    // remove quant/relu/requant-conv
    for (size_t i = 1; i < fusionNodes_.size(); ++i) {
        auto removeNode = quantPattern_ ? fusionNodes_[i].quantNode : fusionNodes_[i].reluRequantNode;
        auto removeOutAnchor = removeNode->GetOutDataAnchor(0);
        FUSION_PASS_CHECK(removeOutAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "relu/requant in anchor is null, fusion failed."), return FAILED);
        auto removePeerInAnchor = removeOutAnchor->GetPeerInDataAnchors();
        for (auto nextAnchor : removePeerInAnchor) {
            FUSION_PASS_CHECK(GraphUtils::RemoveEdge(removeOutAnchor, nextAnchor) != GRAPH_SUCCESS,
                OP_LOGE(FUSED_OP_TYPE.c_str(), "remove edge conv--relu/quant failed, fusion failed."), return FAILED);
        }

        FUSION_PASS_CHECK(graph.RemoveNode(removeNode) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "remove relu/quant node failed, fusion failed."), return FAILED);
    }

    return SUCCESS;
}

Status SameInputConv2dPass::UpdateConvNode(ge::ComputeGraph& graph) const
{
    UpdateConvShape();
    UpdateReluShape();
    UpdateQuantShape();
    UpdateDequantShape();
    UpdateRequantShape();

    UpdateConvEdge();
    UpdateReluEdge(graph);
    UpdateDequantEdge(graph);
    UpdateQuantEdge(graph);

    OP_LOGI(FUSED_OP_TYPE.c_str(), "update conv node success.");
    return SUCCESS;
}

ge::NodePtr SameInputConv2dPass::CreateConcatDimNode(ge::ComputeGraph& graph, std::vector<ge::NodePtr>& newNodes,
    const std::string& name, int32_t dimValue) const
{
    // create const concat dim op
    ge::GeShape concatDimShape = ge::GeShape({1});
    auto concatDimDesc = ge::GeTensorDesc(concatDimShape, ge::FORMAT_ND, ge::DT_INT32);
    concatDimDesc.SetOriginShape(concatDimShape);
    concatDimDesc.SetOriginFormat(ge::FORMAT_ND);
    concatDimDesc.SetOriginDataType(ge::DT_INT32);
    concatDimDesc.SetShape(concatDimShape);
    concatDimDesc.SetFormat(ge::FORMAT_ND);
    concatDimDesc.SetDataType(ge::DT_INT32);

    ge::GeTensorPtr tensorPtr = nullptr;
    FUSION_PASS_MAKE_SHARED(tensorPtr = std::make_shared<ge::GeTensor>(concatDimDesc), return nullptr);
    FUSION_PASS_CHECK(tensorPtr->SetData(reinterpret_cast<uint8_t*>(&dimValue), sizeof(int32_t)) != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "concat dim set data failed, fusion failed."), return nullptr);

    auto concaDimOpDesc = ge::OpDescUtils::CreateConstOp(tensorPtr);
    FUSION_PASS_CHECK(concaDimOpDesc == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "create const op concat dim failed, fusion failed."), return nullptr);
    concaDimOpDesc->SetName(name);

    auto concatDimNode = graph.AddNode(concaDimOpDesc);
    FUSION_PASS_CHECK(concatDimNode == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add concat dim node failed, fusion failed."), return nullptr);
    newNodes.emplace_back(concatDimNode);
    concatDimNode->GetOpDesc()->UpdateOutputDesc(0, concatDimDesc);

    return concatDimNode;
}

Status SameInputConv2dPass::SetShapeDims(int32_t dim, int64_t dimValue, ge::GeTensorDesc &tensorDesc) const
{
    auto shape = tensorDesc.GetShape();
    FUSION_PASS_CHECK((shape.SetDim(dim, dimValue) != GRAPH_SUCCESS),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "shape set dim failed, fusion failed."), return FAILED);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    return SUCCESS;
}

ge::NodePtr SameInputConv2dPass::AddFilterConcatNode(ge::ComputeGraph& graph, std::vector<ge::NodePtr>& newNodes) const
{
    std::vector<ge::NodePtr> filterNodes;
    for (auto& node : fusionNodes_) {
        filterNodes.emplace_back(node.convNode->GetInDataNodes().at(FILTER_POS));
    }

    // create concat node
    ge::OpDescPtr concatDesc = nullptr;
    std::string concatName = fusionNodes_.at(0).convNode->GetName() + FILTER_CONCAT;
    FUSION_PASS_MAKE_SHARED(concatDesc = std::make_shared<ge::OpDesc>(concatName, CONCAT_TYPE), return nullptr);
    for (size_t i = 0; i < filterNodes.size(); ++i) {
        std::string constName = "filter_x" + to_string(i);
        auto filterOutDesc = filterNodes[i]->GetOpDesc()->GetOutputDesc(0);
        FUSION_PASS_CHECK(concatDesc->AddInputDesc(constName, filterOutDesc) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "concat add input desc failed, fusion failed."), return nullptr);
    }

    int64_t dimValue;
    FUSION_PASS_CHECK(GetConvInDimValue(dimValue) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get concat value failed, fusion failed."), return nullptr);
    int32_t axis;
    FUSION_PASS_CHECK(GetConvInAxis(fusionNodes_.at(0).convNode, axis) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get concat axis failed, fusion failed."), return nullptr);

    auto convInFilterDesc = fusionNodes_.at(0).convNode->GetOpDesc()->GetInputDesc("filter").Clone();
    FUSION_PASS_CHECK(SetShapeDims(axis, dimValue, convInFilterDesc) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "set shape dim failed, fusion failed."), return nullptr);

    FUSION_PASS_CHECK(concatDesc->AddOutputDesc("y", convInFilterDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "concat add output desc failed, fusion failed."), return nullptr);

    std::string concatDimName = fusionNodes_.at(0).convNode->GetName() + "_filter_concat_dim";
    auto concatDimNode = CreateConcatDimNode(graph, newNodes, concatDimName, axis);
    FUSION_PASS_CHECK(concatDimNode == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "create concat dim node failed, fusion failed."), return nullptr);

    concatDesc->AddInputDesc("concat_dim", concatDimNode->GetOpDesc()->GetOutputDesc(0));
    ge::AttrUtils::SetInt(concatDesc, "N", filterNodes.size());

    // add concat node
    ge::NodePtr concatNode = graph.AddNode(concatDesc);
    FUSION_PASS_CHECK(concatNode == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add concat dim node failed, fusion failed."), return nullptr);
    newNodes.emplace_back(concatNode);

    auto concatInAnchor = concatNode->GetInDataAnchor(filterNodes.size());
    auto concatDimAnchor = concatDimNode->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(concatDimAnchor, concatInAnchor) != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add filter-concat failed, fusion failed."), return nullptr);

    OP_LOGI(FUSED_OP_TYPE.c_str(), "add filter concat node success.");
    return concatNode;
}

ge::NodePtr SameInputConv2dPass::AddBiasConcatNode(ge::ComputeGraph& graph, std::vector<ge::NodePtr>& newNodes) const
{
    if (fusionNodes_.at(0).convNode->GetOpDesc()->GetInputDesc("bias").IsValid() != GRAPH_SUCCESS) {
        return nullptr;
    }
    std::vector<ge::NodePtr> biasNodes;
    for (auto& node : fusionNodes_) {
        biasNodes.emplace_back(node.convNode->GetInDataNodes().at(BIAS_POS));
    }

    // create concat node
    ge::OpDescPtr concatDesc = nullptr;
    std::string concatName = fusionNodes_.at(0).convNode->GetName() + BIAS_CONCAT;
    FUSION_PASS_MAKE_SHARED(concatDesc = std::make_shared<ge::OpDesc>(concatName, CONCAT_TYPE), return nullptr);
    for (size_t i = 0; i < biasNodes.size(); ++i) {
        std::string constName = "bias_x" + to_string(i);
        auto biasOutDesc = biasNodes[i]->GetOpDesc()->GetOutputDesc(0);
        FUSION_PASS_CHECK(concatDesc->AddInputDesc(constName, biasOutDesc) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "concat add input desc failed, fusion failed."), return nullptr);
    }

    int64_t dimValue;
    FUSION_PASS_CHECK(GetConvInDimValue(dimValue) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get concat value failed, fusion failed."), return nullptr);

    auto convInBiasDesc = fusionNodes_.at(0).convNode->GetOpDesc()->GetInputDesc("bias").Clone();
    FUSION_PASS_CHECK(SetShapeDims(0, dimValue, convInBiasDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "set shape dim failed, fusion failed."), return nullptr);

    FUSION_PASS_CHECK(concatDesc->AddOutputDesc("y", convInBiasDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "concat add output desc failed, fusion failed."), return nullptr);

    std::string concatDimName = fusionNodes_.at(0).convNode->GetName() + "_bias_concat_dim";
    auto concatDimNode = CreateConcatDimNode(graph, newNodes, concatDimName, 0);
    FUSION_PASS_CHECK(concatDimNode == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "create concat dim node failed, fusion failed."), return nullptr);

    concatDesc->AddInputDesc("concat_dim", concatDimNode->GetOpDesc()->GetOutputDesc(0));
    ge::AttrUtils::SetInt(concatDesc, "N", biasNodes.size());

    // add concat node
    ge::NodePtr concatNode = graph.AddNode(concatDesc);
    FUSION_PASS_CHECK(concatNode == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add concat dim node failed, fusion failed."), return nullptr);
    newNodes.emplace_back(concatNode);

    auto concatInAnchor = concatNode->GetInDataAnchor(biasNodes.size());
    auto concatDimAnchor = concatDimNode->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(concatDimAnchor, concatInAnchor) != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add filter-concat failed, fusion failed."), return nullptr);

    OP_LOGI(FUSED_OP_TYPE.c_str(), "add bias concat node success.");
    return concatNode;
}

Status SameInputConv2dPass::UpdateFilterConcat(const std::vector<ge::NodePtr>& filterNodes,
    ge::NodePtr filterConcatNode) const
{
    // update concat filter edge
    for (size_t i = 0; i < filterNodes.size(); ++i) {
        // remove filter-conv
        auto convInAnchor = fusionNodes_[i].convNode->GetInDataAnchor(FILTER_POS);
        FUSION_PASS_CHECK(convInAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "conv data anchor is null, fusion failed."), return FAILED);
        auto peerFilterAnchor = convInAnchor->GetPeerOutAnchor();
        FUSION_PASS_CHECK(peerFilterAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "filter peer data anchor is null, fusion failed."), return FAILED);
        FUSION_PASS_CHECK(GraphUtils::RemoveEdge(peerFilterAnchor, convInAnchor) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "remove filter-conv failed, fusion failed."), return FAILED);

        // add filter-concat
        auto concatInAnchor = filterConcatNode->GetInDataAnchor(i);
        FUSION_PASS_CHECK(concatInAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "concat in anchor is null, fusion failed."), return FAILED);
        auto filterOutAnchor = filterNodes[i]->GetOutDataAnchor(0);
        FUSION_PASS_CHECK(filterOutAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "filter out anchor is null, fusion failed."), return FAILED);
        FUSION_PASS_CHECK(GraphUtils::AddEdge(filterOutAnchor, concatInAnchor) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "add filter-concat failed, fusion failed."), return FAILED);
    }

    // add concat-conv
    auto concatOutAnchor = filterConcatNode->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(concatOutAnchor == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "concat out anchor null, fusion failed."), return FAILED);
    auto convInAnchor = fusionNodes_.at(0).convNode->GetInDataAnchor(FILTER_POS);
    FUSION_PASS_CHECK(convInAnchor == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "conv in anchor null, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(concatOutAnchor, convInAnchor) != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add concat-conv failed, fusion failed."), return FAILED);

    return SUCCESS;
}

Status SameInputConv2dPass::UpdateBiasConcat(const std::vector<ge::NodePtr>& biasNodes,
    ge::NodePtr biasConcatNode) const
{
    // update concat bias edge
    FUSION_PASS_CHECK(biasConcatNode == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "no bias."), return SUCCESS);
    FUSION_PASS_CHECK(fusionNodes_.size() != biasNodes.size(),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "conv and bias not match, fusion failed."), return FAILED);

    for (size_t i = 0; i < fusionNodes_.size(); ++i) {
        // remove bias-conv
        auto convInAnchor = fusionNodes_[i].convNode->GetInDataAnchor(BIAS_POS);
        FUSION_PASS_CHECK(convInAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "conv data anchor is null, fusion failed."), return FAILED);
        auto peerBiasAnchor = convInAnchor->GetPeerOutAnchor();
        FUSION_PASS_CHECK(peerBiasAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "conv peer data anchor is null, fusion failed."), return FAILED);
        FUSION_PASS_CHECK(GraphUtils::RemoveEdge(peerBiasAnchor, convInAnchor) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "remove bias-conv failed, fusion failed."), return FAILED);

        // add bias-concat
        auto concatInAnchor = biasConcatNode->GetInDataAnchor(i);
        FUSION_PASS_CHECK(concatInAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "concat in anchor is null, fusion failed."), return FAILED);
        auto biasOutAnchor = biasNodes[i]->GetOutDataAnchor(0);
        FUSION_PASS_CHECK(biasOutAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "bias out anchor is null, fusion failed."), return FAILED);
        FUSION_PASS_CHECK(GraphUtils::AddEdge(biasOutAnchor, concatInAnchor) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "add bias-concat failed, fusion failed."), return FAILED);
    }

    // add concat-conv
    auto biasConcatOutAnchor = biasConcatNode->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(biasConcatOutAnchor == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "concat out anchor null, fusion failed."), return FAILED);
    auto biasConvInAnchor = fusionNodes_.at(0).convNode->GetInDataAnchor(BIAS_POS);
    FUSION_PASS_CHECK(biasConvInAnchor == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "conv in anchor null, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(biasConcatOutAnchor, biasConvInAnchor) != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add concat-conv failed, fusion failed."), return FAILED);

    return SUCCESS;
}

ge::NodePtr SameInputConv2dPass::AddDequantConcatNode(ge::ComputeGraph& graph,
    std::vector<ge::NodePtr>& newNodes) const
{
    std::vector<ge::NodePtr> scaleNodes;
    for (auto& node : fusionNodes_) {
        scaleNodes.emplace_back(node.dequantNode->GetInDataNodes().at(1));
    }

    // create concat node
    ge::OpDescPtr concatDesc = nullptr;
    std::string concatName = fusionNodes_.at(0).dequantNode->GetName() + DEQUANT_CONCAT;
    FUSION_PASS_MAKE_SHARED(concatDesc = std::make_shared<ge::OpDesc>(concatName, CONCAT_TYPE), return nullptr);
    for (size_t i = 0; i < scaleNodes.size(); ++i) {
        std::string scaleName = "dequant_x" + to_string(i);
        auto scaleDesc = scaleNodes[i]->GetOpDesc()->GetOutputDesc(0);
        FUSION_PASS_CHECK(concatDesc->AddInputDesc(scaleName, scaleDesc) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "concat add input desc failed, fusion failed."), return nullptr);
    }

    int64_t dimValue;
    FUSION_PASS_CHECK(GetDequantDimValue(dimValue) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "dequant concat dim failed, fusion failed."), return nullptr);

    auto dequantConstNode = fusionNodes_.at(0).dequantNode->GetInNodes().at(1);
    auto constDesc = dequantConstNode->GetOpDesc()->GetOutputDesc(0).Clone();
    FUSION_PASS_CHECK(SetShapeDims(0, dimValue, constDesc) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "dequant set dim failed, fusion failed."), return nullptr);

    FUSION_PASS_CHECK(concatDesc->AddOutputDesc("y", constDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "concat add output desc failed, fusion failed."), return nullptr);

    std::string concatDimName = fusionNodes_.at(0).dequantNode->GetName() + "/dequant_concat_dim";
    auto concatDimNode = CreateConcatDimNode(graph, newNodes, concatDimName, 0);
    FUSION_PASS_CHECK(concatDimNode == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "create concat dim node failed, fusion failed."), return nullptr);

    concatDesc->AddInputDesc("concat_dim", concatDimNode->GetOpDesc()->GetOutputDesc(0));
    ge::AttrUtils::SetInt(concatDesc, "N", scaleNodes.size());

    // add concat node
    ge::NodePtr concatNode = graph.AddNode(concatDesc);
    FUSION_PASS_CHECK(concatNode == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add dequant concat dim node failed, fusion failed."), return nullptr);
    newNodes.emplace_back(concatNode);

    auto concatInAnchor = concatNode->GetInDataAnchor(scaleNodes.size());
    auto concatDimAnchor = concatDimNode->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(concatDimAnchor, concatInAnchor) != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add const-concat failed, fusion failed."), return nullptr);

    return concatNode;
}

ge::NodePtr SameInputConv2dPass::AddRequantConcatNode(ge::ComputeGraph& graph,
    std::vector<ge::NodePtr>& newNodes) const
{
    std::vector<ge::NodePtr> constNodes;
    for (auto& node : fusionNodes_) {
        constNodes.emplace_back(node.reluRequantNode->GetInDataNodes().at(1));
    }

    // create concat node
    ge::OpDescPtr concatDesc = nullptr;
    std::string concatName = fusionNodes_.at(0).reluRequantNode->GetName() + REQUANT_CONCAT;
    FUSION_PASS_MAKE_SHARED(concatDesc = std::make_shared<ge::OpDesc>(concatName, CONCAT_TYPE), return nullptr);
    for (size_t i = 0; i < constNodes.size(); ++i) {
        std::string scaleName = "requant_x" + to_string(i);
        auto constOutDesc = constNodes[i]->GetOpDesc()->GetOutputDesc(0);
        FUSION_PASS_CHECK(concatDesc->AddInputDesc(scaleName, constOutDesc) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "concat add input desc failed, fusion failed."), return nullptr);
    }

    int64_t dimValue;
    FUSION_PASS_CHECK(GetRequantDimValue(dimValue) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get concat value failed, fusion failed."), return nullptr);

    auto requantInDesc = fusionNodes_.at(0).reluRequantNode->GetOpDesc()->GetInputDesc(1).Clone();
    FUSION_PASS_CHECK(SetShapeDims(0, dimValue, requantInDesc) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "set shape dim failed, fusion failed."), return nullptr);

    FUSION_PASS_CHECK(concatDesc->AddOutputDesc("y", requantInDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "concat add output desc failed, fusion failed."), return nullptr);

    std::string concatDimName = fusionNodes_.at(0).reluRequantNode->GetName() + "_requant_concat_dim";
    auto concatDimNode = CreateConcatDimNode(graph, newNodes, concatDimName, 0);
    FUSION_PASS_CHECK(concatDimNode == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "create concat dim node failed, fusion failed."), return nullptr);

    auto dimDesc = concatDimNode->GetOpDesc()->GetOutputDesc(0);
    FUSION_PASS_CHECK(concatDesc->AddInputDesc("concat_dim", dimDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add concat dim desc failed, fusion failed."), return nullptr);
    ge::AttrUtils::SetInt(concatDesc, "N", constNodes.size());

    // add concat node
    ge::NodePtr concatNode = graph.AddNode(concatDesc);
    FUSION_PASS_CHECK(concatNode == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add requant concat dim node failed, fusion failed."), return nullptr);
    newNodes.emplace_back(concatNode);

    auto concatInAnchor = concatNode->GetInDataAnchor(constNodes.size());
    auto concatDimAnchor = concatDimNode->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(concatDimAnchor, concatInAnchor) != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add const-concat failed, fusion failed."), return nullptr);

    OP_LOGI(FUSED_OP_TYPE.c_str(), "add requant concat node success.");
    return concatNode;
}

Status SameInputConv2dPass::UpdateDequantConcat(const std::vector<ge::NodePtr>& constNodes,
    ge::NodePtr concatNode) const
{
    // update concat quant edge
    FUSION_PASS_CHECK(concatNode == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "no dequant."), return SUCCESS);
    FUSION_PASS_CHECK(!quantPattern_, OP_LOGI(FUSED_OP_TYPE.c_str(), "no dequant."), return SUCCESS);
    FUSION_PASS_CHECK(fusionNodes_.size() != constNodes.size(),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "dequant const node not match."), return FAILED);

    for (size_t i = 0; i < fusionNodes_.size(); ++i) {
        // remove const-dequant
        auto dequantInAnchor = fusionNodes_[i].dequantNode->GetInDataAnchor(1);
        FUSION_PASS_CHECK(dequantInAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "dequant data anchor is null, fusion failed."), return FAILED);
        auto peerDequantAnchor = dequantInAnchor->GetPeerOutAnchor();
        FUSION_PASS_CHECK(peerDequantAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "dequant peer data anchor is null, fusion failed."), return FAILED);
        FUSION_PASS_CHECK(GraphUtils::RemoveEdge(peerDequantAnchor, dequantInAnchor) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "remove const-dequant failed, fusion failed."), return FAILED);

        // add const-concat
        auto concatInAnchor = concatNode->GetInDataAnchor(i);
        FUSION_PASS_CHECK(concatInAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "concat in anchor is null, fusion failed."), return FAILED);
        auto constOutAnchor = constNodes[i]->GetOutDataAnchor(0);
        FUSION_PASS_CHECK(constOutAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "const out anchor is null, fusion failed."), return FAILED);
        FUSION_PASS_CHECK(GraphUtils::AddEdge(constOutAnchor, concatInAnchor) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "add const-concat failed, fusion failed."), return FAILED);
    }

    // add concat-dequant
    auto concatOutAnchor = concatNode->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(concatOutAnchor == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "concat out anchor null, fusion failed."), return FAILED);
    auto dequantInAnchor = fusionNodes_.at(0).dequantNode->GetInDataAnchor(1);
    FUSION_PASS_CHECK(dequantInAnchor == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "dequant in anchor null, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(concatOutAnchor, dequantInAnchor) != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add concat-conv failed, fusion failed."), return FAILED);

    return SUCCESS;
}

Status SameInputConv2dPass::UpdateRequantConcat(const std::vector<ge::NodePtr>& constNodes,
    ge::NodePtr concatNode) const
{
    // update concat quant edge
    FUSION_PASS_CHECK(concatNode == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "no requant."), return SUCCESS);
    FUSION_PASS_CHECK(fusionNodes_.size() != constNodes.size(),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "requant const node not match."), return FAILED);

    for (size_t i = 0; i < fusionNodes_.size(); ++i) {
        // remove const-requant/dequant
        auto requantInAnchor = fusionNodes_[i].reluRequantNode->GetInDataAnchor(1);
        FUSION_PASS_CHECK(requantInAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "requant data anchor is null, fusion failed."), return FAILED);
        auto peerRequantAnchor = requantInAnchor->GetPeerOutAnchor();
        FUSION_PASS_CHECK(peerRequantAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "requant peer data anchor is null, fusion failed."), return FAILED);
        FUSION_PASS_CHECK(GraphUtils::RemoveEdge(peerRequantAnchor, requantInAnchor) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "remove const-requant failed, fusion failed."), return FAILED);

        // add const-concat
        auto concatInAnchor = concatNode->GetInDataAnchor(i);
        FUSION_PASS_CHECK(concatInAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "concat in anchor is null, fusion failed."), return FAILED);
        auto constOutAnchor = constNodes[i]->GetOutDataAnchor(0);
        FUSION_PASS_CHECK(constOutAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "const out anchor is null, fusion failed."), return FAILED);
        FUSION_PASS_CHECK(GraphUtils::AddEdge(constOutAnchor, concatInAnchor) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "add const-concat failed, fusion failed."), return FAILED);
    }

    // add concat-requant
    auto concatOutAnchor = concatNode->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(concatOutAnchor == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "concat out anchor null, fusion failed."), return FAILED);
    auto requantInAnchor = fusionNodes_.at(0).reluRequantNode->GetInDataAnchor(1);
    FUSION_PASS_CHECK(requantInAnchor == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "requant in anchor null, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(concatOutAnchor, requantInAnchor) != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add concat-requant failed, fusion failed."), return FAILED);

    return SUCCESS;
}

Status SameInputConv2dPass::UpdateConcatNodes(ge::NodePtr filterConcatNode, ge::NodePtr biasConcatNode) const
{
    std::vector<ge::NodePtr> filterNodes;
    std::vector<ge::NodePtr> biasNodes;
    for (auto& node : fusionNodes_) {
        filterNodes.emplace_back(node.convNode->GetInDataNodes().at(FILTER_POS));
        if (biasConcatNode != nullptr && node.convNode->GetInDataNodes().size() > BIAS_POS) {
            biasNodes.emplace_back(node.convNode->GetInDataNodes().at(BIAS_POS));
        }
    }

    FUSION_PASS_CHECK(UpdateFilterConcat(filterNodes, filterConcatNode) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "update filter concat failed, fusion failed."), return FAILED);

    FUSION_PASS_CHECK(UpdateBiasConcat(biasNodes, biasConcatNode) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "update bias concat failed, fusion failed."), return FAILED);

    return SUCCESS;
}

Status SameInputConv2dPass::AddConcatNodes(ge::ComputeGraph& graph, std::vector<ge::NodePtr>& newNodes) const
{
    auto filterConcat = AddFilterConcatNode(graph, newNodes);
    FUSION_PASS_CHECK(filterConcat == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add filter concat node failed, fusion failed."), return FAILED);

    ge::NodePtr biasConcat = nullptr;
    if (fusionNodes_.at(0).convNode->GetOpDesc()->GetInputDesc("bias").IsValid() == GRAPH_SUCCESS) {
        biasConcat = AddBiasConcatNode(graph, newNodes);
        FUSION_PASS_CHECK(biasConcat == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "add bias concat node failed, fusion failed."), return FAILED);
    }

    FUSION_PASS_CHECK(UpdateConcatNodes(filterConcat, biasConcat) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add bias concat node failed, fusion failed."), return FAILED);

    if (quantPattern_) {
        auto dequantConcat = AddDequantConcatNode(graph, newNodes);
        FUSION_PASS_CHECK(dequantConcat == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "add dequant concat node failed, fusion failed."), return FAILED);

        std::vector<ge::NodePtr> dequantConstNodes;
        for (auto& node : fusionNodes_) {
            dequantConstNodes.emplace_back(node.dequantNode->GetInDataNodes().at(1));
        }
        FUSION_PASS_CHECK(UpdateDequantConcat(dequantConstNodes, dequantConcat) != SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "update dequant concat failed, fusion failed."), return FAILED);
    }

    if (requantPattern_) {
        auto requantConcat = AddRequantConcatNode(graph, newNodes);
        FUSION_PASS_CHECK(requantConcat == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "add requant concat node failed, fusion failed."), return FAILED);

        std::vector<ge::NodePtr> requantConstNodes;
        for (auto& node : fusionNodes_) {
            requantConstNodes.emplace_back(node.reluRequantNode->GetInDataNodes().at(1));
        }
        FUSION_PASS_CHECK(UpdateRequantConcat(requantConstNodes, requantConcat) != SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "update requant concat failed, fusion failed."), return FAILED);
    }

    return SUCCESS;
}

Status SameInputConv2dPass::LinkReluSplit(ge::ComputeGraph& graph, const std::vector<ge::NodePtr>& reluNodes,
    ge::NodePtr splitNode) const
{
    // remove edge relu-conv, add edge split-conv
    for (size_t i = 0; i < reluNodes.size(); ++i) {
        auto splitOutAnchor = splitNode->GetOutDataAnchor(i);
        FUSION_PASS_CHECK(splitOutAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "split out data anchor is null, fusion failed."), return FAILED);
        auto reluOutAnchor = reluNodes[i]->GetOutDataAnchor(0);
        FUSION_PASS_CHECK(reluOutAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "relu out data anchor is null, fusion failed."), return FAILED);
        auto reluPeerInAnchor = reluOutAnchor->GetPeerInDataAnchors();
        for (auto& nextAnchor : reluPeerInAnchor) {
            FUSION_PASS_CHECK(GraphUtils::RemoveEdge(reluOutAnchor, nextAnchor),
                OP_LOGE(FUSED_OP_TYPE.c_str(), "remove edge from conv--relu failed, fusion failed."), return FAILED);
            FUSION_PASS_CHECK(GraphUtils::AddEdge(splitOutAnchor, nextAnchor),
                OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from split--relu failed, fusion failed."), return FAILED);
        }
    }

    // add edge relu-split
    auto reluOutAnchor = reluNodes.at(0)->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(reluOutAnchor == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "conv out data anchor is null, fusion failed."), return FAILED);
    auto splitInAnchor = splitNode->GetInDataAnchor(0);
    FUSION_PASS_CHECK(splitInAnchor == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "split in data anchor is null, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(reluOutAnchor, splitInAnchor),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from conv--split failed, fusion failed."), return FAILED);

    return SUCCESS;
}

ge::NodePtr SameInputConv2dPass::CreateStridedReadNode(ge::ComputeGraph& graph, std::vector<ge::NodePtr>& newNodes,
    const std::string& strideName, int64_t strideValue) const
{
    ge::OpDescPtr stridedReadOpDesc = nullptr;
    FUSION_PASS_MAKE_SHARED(stridedReadOpDesc = std::make_shared<ge::OpDesc>(strideName, "StridedRead"),
        return nullptr);

    ge::GeTensorDesc inputDesc;
    ge::GeTensorDesc outputDesc;
    FUSION_PASS_CHECK(stridedReadOpDesc->AddInputDesc("x", inputDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add stride input desc failed, fusion failed."), return nullptr);
    FUSION_PASS_CHECK(stridedReadOpDesc->AddOutputDesc("y", outputDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add stride output desc failed, fusion failed."), return nullptr);

    ge::AttrUtils::SetInt(stridedReadOpDesc, "axis", 1);
    ge::AttrUtils::SetInt(stridedReadOpDesc, "stride", strideValue);

    auto strideNode = graph.AddNode(stridedReadOpDesc);
    FUSION_PASS_CHECK(strideNode == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add stride node failed, fusion failed."), return nullptr);
    newNodes.emplace_back(strideNode);

    return strideNode;
}

Status SameInputConv2dPass::AddStrideNode(ge::ComputeGraph& graph, std::vector<ge::NodePtr>& newNodes,
    ge::NodePtr splitNode) const
{
    std::vector<ge::NodePtr> strideReads;
    for (size_t i = 0; i < splitNode->GetOutDataNodes().size(); ++i) {
        auto dims = splitNode->GetOpDesc()->GetOutputDesc(i).GetShape().GetDims();
        auto axis = 0;
        GetNodeCoutAxis(splitNode->GetOpDesc()->GetOutputDesc(i), axis);
        auto strideValue = dims[axis];
        OP_LOGI(FUSED_OP_TYPE.c_str(), "stride value is %u.", strideValue);
        std::string name = splitNode->GetName() + "/stride_read_" + to_string(i);
        auto strideNode = CreateStridedReadNode(graph, newNodes, name, strideValue);
        FUSION_PASS_CHECK(strideNode == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "create stride node failed, fusion failed."), return FAILED);
        strideReads.emplace_back(strideNode);
    }

    FUSION_PASS_CHECK(LinkSplitStrideRead(strideReads, splitNode) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "link split-stride failed, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(TransSplitStrideRead(strideReads) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "link trans split-stride failed, fusion failed."), return FAILED);

    return SUCCESS;
}

Status SameInputConv2dPass::LinkSplitStrideRead(const std::vector<ge::NodePtr>& strideReads,
    ge::NodePtr splitNode) const
{
    for (size_t i = 0; i < strideReads.size(); ++i) {
        auto splitOutAnchor = splitNode->GetOutDataAnchor(i);
        FUSION_PASS_CHECK(splitOutAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "split out data anchor is null, fusion failed."), return FAILED);
        auto strideOutAnchor = strideReads[i]->GetOutDataAnchor(0);
        FUSION_PASS_CHECK(strideOutAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "stride out data anchor is null, fusion failed."), return FAILED);

        // remove edge split-conv, add edge split-strided_read
        auto splitPeerInAnchor = splitOutAnchor->GetPeerInDataAnchors();
        for (auto& nextAnchor : splitPeerInAnchor) {
            FUSION_PASS_CHECK(GraphUtils::RemoveEdge(splitOutAnchor, nextAnchor),
                OP_LOGE(FUSED_OP_TYPE.c_str(), "remove split-conv failed, fusion failed."), return FAILED);
            FUSION_PASS_CHECK(GraphUtils::AddEdge(strideOutAnchor, nextAnchor),
                OP_LOGE(FUSED_OP_TYPE.c_str(), "add stride-conv failed, fusion failed."), return FAILED);
        }

        // add edge strided_read-conv
        auto strideInAnchor = strideReads[i]->GetInDataAnchor(0);
        FUSION_PASS_CHECK(GraphUtils::AddEdge(splitOutAnchor, strideInAnchor),
            OP_LOGE(FUSED_OP_TYPE.c_str(), "add  split--relu failed, fusion failed."), return FAILED);
    }

    return SUCCESS;
}

Status SameInputConv2dPass::TransferShapeToNC1HWC0(ge::Format oldFormat, ge::DataType dataType,
    ge::GeShape originShape, ge::GeShape& newShape) const
{
    FUSION_PASS_CHECK(oldFormat >= ge::FORMAT_RESERVED,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "old format %u is invalid!", oldFormat), return FAILED);
    FUSION_PASS_CHECK(dataType >= ge::DT_UNDEFINED,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "dataType %u is invalid!", dataType), return FAILED);

    std::string format = TypeUtils::FormatToSerialString(oldFormat);
    size_t fondH = format.find('H');
    size_t fondW = format.find('W');
    size_t fondN = format.find('N');
    size_t fondC = format.find('C');
    auto oldDims = originShape.GetDims();
    FUSION_PASS_CHECK(fondH >= oldDims.size() || fondW >= oldDims.size() ||
        fondN >= oldDims.size() || fondC >= oldDims.size(),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "invalid shape."), return FAILED);

    uint32_t c0 = (dataType == DT_INT8) ? INT8_ALIGN : DATA_ALIGN;
    uint32_t c1 = oldDims[fondC] / c0;
    OP_LOGI(FUSED_OP_TYPE.c_str(), "type %u, c0 %u, c1 %u", dataType, c0, c1);

    std::vector<int64_t> newDims {oldDims[fondN], c1, oldDims[fondH], oldDims[fondW], c0};
    ge::GeShape shapeTmp(newDims);
    newShape = shapeTmp;

    return SUCCESS;
}

Status SameInputConv2dPass::GetNC1HWC0Shape(ge::GeTensorDescPtr tensorDesc, const ge::DataType& quantDataType) const
{
    ge::Format originFormat = tensorDesc->GetFormat();
    ge::GeShape originShape = tensorDesc->GetShape();
    std::vector<int64_t> oldDimVec = originShape.GetDims();
    FUSION_PASS_CHECK(oldDimVec.empty(),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "oldDimVec is empty."), return FAILED);

    ge::DataType dataType = tensorDesc->GetDataType();
    OP_LOGI(FUSED_OP_TYPE.c_str(), "NC1HWC0: dataType %u, quantDataType %u", dataType, quantDataType);
    if (quantDataType == ge::DT_INT8 || quantDataType == ge::DT_INT4) {
        dataType = quantDataType;
        tensorDesc->SetDataType(quantDataType);
    }

    ge::GeShape newShape;
    FUSION_PASS_CHECK(TransferShapeToNC1HWC0(originFormat, dataType, originShape, newShape) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "transfer NC1HWC0 failed."), return FAILED);
    tensorDesc->SetShape(newShape);

    if ((tensorDesc->GetDataType() == ge::DT_FLOAT || tensorDesc->GetDataType() == ge::DT_FLOAT) &&
        quantDataType != ge::DT_INT8 && quantDataType != ge::DT_INT4) {
        tensorDesc->SetDataType(ge::DT_FLOAT16);
    }
    tensorDesc->SetFormat(ge::FORMAT_NC1HWC0);

    return SUCCESS;
}

Status SameInputConv2dPass::JudgeOp(ge::NodePtr node) const
{
    FUSION_PASS_CHECK(node == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "node is null, fusion failed."), return FAILED);

    // update input desc
    ge::InDataAnchorPtr srcInData = node->GetInDataAnchor(0);
    FUSION_PASS_CHECK(srcInData == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "node in data anchor is null, fusion failed."), return FAILED);
    ge::OutDataAnchorPtr src = srcInData->GetPeerOutAnchor();
    FUSION_PASS_CHECK(src == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "src is null, fusion failed."), return FAILED);
    ge::NodePtr srcNode = src->GetOwnerNode();
    FUSION_PASS_CHECK(srcNode == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "srcNode is null, fusion failed."), return FAILED);

    auto srcDesc = srcNode->GetOpDesc()->GetOutputDesc(src->GetIdx());
    FUSION_PASS_CHECK(node->GetOpDesc()->UpdateInputDesc(0, srcDesc) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "%s update input desc failed.", node->GetName().c_str()), return FAILED);

    // update ouput desc
    ge::OutDataAnchorPtr dstOutData = node->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(dstOutData == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "dst out data is null, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(dstOutData->GetPeerInDataAnchors().empty(),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "dst peer in empty, fusion failed."), return FAILED);
    ge::InDataAnchorPtr dst = dstOutData->GetPeerInDataAnchors().at(0);
    ge::NodePtr dstNode = dst->GetOwnerNode();
    FUSION_PASS_CHECK(dstNode == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "dstNode is null, fusion failed."), return FAILED);

    auto dstDesc = dstNode->GetOpDesc()->GetInputDesc(dst->GetIdx());
    FUSION_PASS_CHECK(node->GetOpDesc()->UpdateOutputDesc(0, dstDesc) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "%s update output desc failed.", node->GetName().c_str()), return FAILED);

    return SUCCESS;
}

Status SameInputConv2dPass::TransSplitStrideRead(const std::vector<ge::NodePtr>& strideReads) const
{
    for (auto& stride : strideReads) {
        FUSION_PASS_CHECK(JudgeOp(stride) != SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "judge stride failed."), return FAILED);

        auto strideDesc = stride->GetOpDesc()->MutableOutputDesc(0);
        auto dataType = stride->GetOpDesc()->GetOutputDesc(0).GetDataType();
        FUSION_PASS_CHECK(GetNC1HWC0Shape(strideDesc, dataType) != SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "get out shape of NC1HWC0 failed."), return FAILED);

        auto strideInDesc = stride->GetOpDesc()->MutableInputDesc(0);
        auto dataInType = stride->GetOpDesc()->GetInputDesc(0).GetDataType();
        FUSION_PASS_CHECK(GetNC1HWC0Shape(strideInDesc, dataInType) != SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "get in shape of NC1HWC0 failed."), return FAILED);

        auto splitShape = stride->GetOpDesc()->MutableInputDesc(0)->MutableShape();
        ge::AttrUtils::SetInt(stride->GetOpDesc(), "stride", splitShape.GetDim(1));
    }

    return SUCCESS;
}

REGISTER_PASS("SameInputConv2dPass", BUILT_IN_GRAPH_PASS, SameInputConv2dPass);
}
