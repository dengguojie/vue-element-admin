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

/*!
  * @brief Define pattern.
  * The graph struct need to adapt is shown as follows:
  *
  *          x                     x
  *       /     \                  |
  *   conv2d0  conv2d1           conv2d
  *      |        |      ==>       |
  *    relu      relu            split
  *      |        |              /    \
  *   conv2d2   conv2d3       relu    relu
  *                             |       |
  *                         conv2d2   conv2d3
  *
  *  Notice: the struct can be captured by
  *          input + conv2d0 + relu + conv2d2 pattern
  *  @return vector<FusionPattern*> All valid patterns.
  */

std::vector<FusionPattern*> SameInputConv2dPass::DefinePatterns()
{
    OP_LOGD(FUSED_OP_TYPE.c_str(), "SameInputConv2dPass define patterns start.");
    std::vector<FusionPattern*> patterns;
    FusionPattern* pattern = new(std::nothrow)FusionPattern("SameInputConv2dPass");
    FUSION_PASS_CHECK(pattern == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
        return patterns);
    pattern->AddOpDesc(PATTERN_INPUT)
        .AddOpDesc(PATTERN_CONV2D_0, {CONV2D_TYPE})
        .AddOpDesc(PATTERN_CONV2D_2, {CONV2D_TYPE})
        .AddOpDesc(PATTERN_RELU_0, {RELU_TYPE})
        .SetInputs(PATTERN_CONV2D_0, {PATTERN_INPUT})
        .SetInputs(PATTERN_RELU_0, {PATTERN_CONV2D_0})
        .SetInputs(PATTERN_CONV2D_2, {PATTERN_RELU_0})
        .SetOutput(PATTERN_CONV2D_2);
    patterns.push_back(pattern);

    OP_LOGD(FUSED_OP_TYPE.c_str(), "SameInputConv2dPass define patterns end.");
    return patterns;
}

Status SameInputConv2dPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, std::vector<ge::NodePtr>& newNodes)
{
    GE_DUMP(make_shared<ComputeGraph>(graph), "same_input_conv2d_fusion_pass_begin");
    OP_LOGD(FUSED_OP_TYPE.c_str(), "enter SameInputConv2dPass.");
    auto inputNode = GetNodeFromMapping(PATTERN_INPUT, mapping);
    FUSION_PASS_CHECK(inputNode == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "input node is null, fusion failed."), return PARAM_INVALID);
    auto convNode = GetNodeFromMapping(PATTERN_CONV2D_0, mapping);
    FUSION_PASS_CHECK(convNode == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "conv node is null, fusion failed."), return PARAM_INVALID);

    // the matching conv node in the first
    std::vector<ge::NodePtr> convNodes {convNode};
    auto ret = CheckFusion(inputNode, convNodes);
    if (ret != SUCCESS) {
        return ret;
    }

    ret = AddConcatNodes(graph, newNodes, convNodes);
    if (ret != SUCCESS) {
        return ret;
    }

    ret = AddSplitNode(graph, newNodes, convNodes);
    if (ret != SUCCESS) {
        return ret;
    }

    ret = UpdateConvNode(graph, convNodes);
    if (ret != SUCCESS) {
        return ret;
    }

    GE_DUMP(make_shared<ComputeGraph>(graph), "same_input_conv2d_fusion_pass_finish");
    OP_LOGD(FUSED_OP_TYPE.c_str(), "leave SameInputConv2dPass.");
    return SUCCESS;
}

Status SameInputConv2dPass::CheckFusion(ge::NodePtr inputNode, std::vector<ge::NodePtr>& convNodes) const
{
    FUSION_PASS_CHECK(convNodes.empty(),
        OP_LOGI(FUSED_OP_TYPE.c_str(), "no conv from pattern, no fusion."), return NOT_CHANGED);
    FUSION_PASS_CHECK(convNodes.at(0)->GetOutDataNodes().empty(),
        OP_LOGD(FUSED_OP_TYPE.c_str(), "conv out data empty, no fusion."), return NOT_CHANGED);

    auto ret = GetConvNodes(inputNode, convNodes);
    if (ret != SUCCESS) {
        return ret;
    }

    ret = CheckConvNodes(inputNode, convNodes);
    if (ret != SUCCESS) {
        return ret;
    }

    ret = CheckConvAttr(convNodes);
    if (ret != SUCCESS) {
        return ret;
    }

    ret = CheckConvInputLink(convNodes);
    if (ret != SUCCESS) {
        return ret;
    }

    FUSION_PASS_CHECK(convNodes.size() <= 1,
        OP_LOGD(FUSED_OP_TYPE.c_str(), "no conv nodes to fuse, no fusion."), return NOT_CHANGED);

    OP_LOGI(FUSED_OP_TYPE.c_str(), "conv fusion num %zu.", convNodes.size());
    return SUCCESS;
}

Status SameInputConv2dPass::GetConvNodes(ge::NodePtr inputNode, std::vector<ge::NodePtr>& convNodes) const
{
    for (auto& inputOutNode : inputNode->GetOutDataNodes()) {
        // check whether the node after input is conv
        FUSION_PASS_CHECK(inputOutNode == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "input out data node is null, fusion failed."),
            return PARAM_INVALID);
        if (inputOutNode->GetType() != CONV2D_TYPE) {
            continue;
        }
        if (inputOutNode == convNodes.at(0)) {
            continue;
        }

        // check whether the node after conv is only relu
        auto convOutNode = inputOutNode->GetOutDataNodes();
        if (convOutNode.size() != 1) {
            continue;
        }
        FUSION_PASS_CHECK(convOutNode.at(0) == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "conv out data node is null, fusion failed."), return PARAM_INVALID);
        if (convOutNode.at(0)->GetType() != RELU_TYPE) {
            continue;
        }

        // check whether the node after relu is only conv
        auto reluOutNode = convOutNode.at(0)->GetOutDataNodes();
        if (reluOutNode.size() != 1) {
            continue;
        }
        FUSION_PASS_CHECK(reluOutNode.at(0) == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "relu out data node is null, fusion failed."), return PARAM_INVALID);
        if (reluOutNode.at(0)->GetType() == CONV2D_TYPE) {
            convNodes.emplace_back(inputOutNode);
        }
    }

    return SUCCESS;
}

Status SameInputConv2dPass::CheckConvNodes(ge::NodePtr inputNode, const std::vector<ge::NodePtr>& convNodes) const
{
    FUSION_PASS_CHECK(convNodes.size() <= 1,
        OP_LOGD(FUSED_OP_TYPE.c_str(), "no conv nodes to fuse, no fusion."), return NOT_CHANGED);

    for (auto& node : convNodes) {
        FUSION_PASS_CHECK(node == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "conv node is null, fusion failed."), return PARAM_INVALID);
        FUSION_PASS_CHECK(node->GetOpDesc() == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "conv desc is null, fusion failed."), return PARAM_INVALID);
        FUSION_PASS_CHECK(node->GetInDataNodes().size() < CONV2D_INPUT_SIZE_MIN,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "conv input < size min, fusion failed."), return PARAM_INVALID);

        FUSION_PASS_CHECK(node->GetOpDesc()->GetInputDesc("filter").IsValid() != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "conv has no filter, fusion failed."), return PARAM_INVALID);

        FUSION_PASS_CHECK(node->GetInDataNodes().at(0) != inputNode,
            OP_LOGD(FUSED_OP_TYPE.c_str(), "conv has no common fmap, no fusion."), return NOT_CHANGED);

        auto filterType = node->GetInDataNodes().at(FILTER_POS)->GetType();
        FUSION_PASS_CHECK(filterType != CONST_TYPE && filterType != CONSTANT_TYPE && filterType != FILTER_HOST_TYPE,
            OP_LOGD(FUSED_OP_TYPE.c_str(), "conv filter not const/host, no fusion."), return NOT_CHANGED);

        auto convShape = node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();
        for (auto& value : convShape) {
            FUSION_PASS_CHECK(value < 0,
                OP_LOGD(FUSED_OP_TYPE.c_str(), "dynamic shape, no fusion."), return NOT_CHANGED);
        }
    }

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

Status SameInputConv2dPass::GetConvInput(ge::NodePtr conv, ConvFusionInput &convInput) const
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
            OP_LOGE(FUSED_OP_TYPE.c_str(), "invalid bais shape, fusion failed."),
            return PARAM_INVALID);
    }

    auto offsetDesc = conv->GetOpDesc()->GetInputDesc("offset_w");
    if (offsetDesc.IsValid() == GRAPH_SUCCESS) {
        convInput.offset = offsetDesc.GetShape().GetDims();
    }

    return SUCCESS;
}

Status SameInputConv2dPass::CheckConvAttr(std::vector<ge::NodePtr>& convNodes) const
{
    ConvFusionAttr attr = GetConvAttr(convNodes.at(0));
    ConvFusionInput input;
    auto ret = GetConvInput(convNodes.at(0), input);
    if (ret != SUCCESS) {
        return ret;
    }

    std::vector<ge::NodePtr> fusionConvNodes {convNodes.at(0)};
    for (size_t i = 1; i < convNodes.size(); ++i) {
        ConvFusionAttr attrCheck = GetConvAttr(convNodes[i]);
        if (attrCheck != attr) {
            continue;
        }

        ConvFusionInput inputCheck;
        ret = GetConvInput(convNodes[i], inputCheck);
        if (ret != SUCCESS) {
            return ret;
        }
        if (inputCheck != input) {
            continue;
        }

        fusionConvNodes.emplace_back(convNodes[i]);
    }
    convNodes.swap(fusionConvNodes);

    return SUCCESS;
}

Status SameInputConv2dPass::CheckConvInputLink(std::vector<ge::NodePtr>& convNodes) const
{
    auto ret = CheckConvInputLink(convNodes.at(0), convNodes);
    if (ret != SUCCESS) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "the first filter/bias links to other node, no fusion.");
        return ret;
    }

    std::vector<ge::NodePtr> preConvNodes {convNodes};
    std::vector<ge::NodePtr> result;
    while (preConvNodes != result) {
        result.emplace_back(convNodes.at(0));
        for (size_t i = 1; i < preConvNodes.size(); ++i) {
            auto ret = CheckConvInputLink(preConvNodes[i], preConvNodes);
            if (ret != SUCCESS) {
                continue;
            }
            result.emplace_back(preConvNodes[i]);
        }
        if (result == preConvNodes) {
            convNodes.swap(result);
            break;
        } else {
            preConvNodes.swap(result);
            result.clear();
        }
    }

    return SUCCESS;
}

Status SameInputConv2dPass::CheckConvInputLink(ge::NodePtr node, std::vector<ge::NodePtr>& convNodes) const
{
    auto filterNode = node->GetInDataNodes().at(FILTER_POS);
    for (auto &outNode : filterNode->GetOutDataNodes()) {
        auto iter = find(convNodes.begin(), convNodes.end(), outNode);
        if (iter == convNodes.end()) {
            OP_LOGI(FUSED_OP_TYPE.c_str(), "filter links to others, no fusion.");
            return NOT_CHANGED;
        }
    }
    if (filterNode->GetOutDataNodes().size() > 1) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "conv has common filter.");
    }

    if (node->GetOpDesc()->GetInputDesc("bias").IsValid() != GRAPH_SUCCESS) {
        return SUCCESS;
    }
    auto biasNode = node->GetInDataNodes().at(BIAS_POS);
    for (auto &outNode : biasNode->GetOutDataNodes()) {
        auto iter = find(convNodes.begin(), convNodes.end(), outNode);
        if (iter == convNodes.end()) {
            OP_LOGI(FUSED_OP_TYPE.c_str(), "bias links to others, no fusion.");
            return NOT_CHANGED;
        }
    }
    if (biasNode->GetOutDataNodes().size() > 1) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "conv has common bias.");
    }

    return SUCCESS;
}

Status SameInputConv2dPass::GetSplitAttr(const std::vector<ge::NodePtr>& reluNodes, std::vector<int32_t> &sizeSplits,
    int32_t &splitDim) const
{
    auto reluOutDesc = reluNodes.at(0)->GetOpDesc()->GetOutputDesc(0);
    std::string format = TypeUtils::FormatToSerialString(reluOutDesc.GetFormat());
    size_t axis = format.find('C');
    FUSION_PASS_CHECK(axis == std::string::npos,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get split axis, fusion failed."), return PARAM_INVALID);

    for (auto& reluNode : reluNodes) {
        auto reluOutDesc = reluNode->GetOpDesc()->GetOutputDesc(0);
        auto reluDim = reluOutDesc.GetShape().GetDims();
        FUSION_PASS_CHECK(axis >= reluDim.size(),
            OP_LOGE(FUSED_OP_TYPE.c_str(), "invalid relu shape, fusion failed."), return PARAM_INVALID);
        sizeSplits.emplace_back(reluDim[axis]);
        OP_LOGI(FUSED_OP_TYPE.c_str(), "split dim %d value %d.", axis, reluDim[axis]);
    }

    splitDim = static_cast<int32_t>(axis);
    OP_LOGI(FUSED_OP_TYPE.c_str(), "split dim %d, split num %zu.", splitDim, sizeSplits.size());

    return SUCCESS;
}

Status SameInputConv2dPass::GetConvInDimValue(const std::vector<ge::NodePtr>& convNodes, int64_t& dimValue) const
{
    auto filterDesc = convNodes.at(0)->GetOpDesc()->GetInputDesc(FILTER_POS);
    std::string format = TypeUtils::FormatToSerialString(filterDesc.GetFormat());
    size_t axis = format.find('N');
    FUSION_PASS_CHECK(axis == std::string::npos,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get conv axis failed, fusion failed."), return PARAM_INVALID);

    dimValue = 0;
    for (auto& convNode : convNodes) {
        auto filterDesc = convNode->GetOpDesc()->GetInputDesc(FILTER_POS);
        auto filterDim = filterDesc.GetShape().GetDims();
        FUSION_PASS_CHECK(axis >= filterDim.size(),
            OP_LOGE(FUSED_OP_TYPE.c_str(), "invalid filter shape, fusion failed."), return PARAM_INVALID);
        dimValue += filterDim[axis];
        OP_LOGI(FUSED_OP_TYPE.c_str(), "concat dim %d value %d.", axis, filterDim[axis]);
    }

    OP_LOGI(FUSED_OP_TYPE.c_str(), "concat dim sum value %d.", dimValue);
    return SUCCESS;
}

Status SameInputConv2dPass::GetConvInAxis(ge::NodePtr convNode, int32_t &axis) const
{
    auto convInFilterDesc = convNode->GetOpDesc()->GetInputDesc(FILTER_POS);
    std::string format = TypeUtils::FormatToSerialString(convInFilterDesc.GetFormat());
    size_t coutAxis = format.find('N');
    FUSION_PASS_CHECK(coutAxis == std::string::npos,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get filter axis failed %s, fusion failed.", format.c_str()), return PARAM_INVALID);

    axis = static_cast<int32_t>(coutAxis);

    OP_LOGI(FUSED_OP_TYPE.c_str(), "concat axis %d.", axis);
    return SUCCESS;
}

Status SameInputConv2dPass::GetConvOutAxis(ge::NodePtr convNode, int32_t &axis) const
{
    auto convOutDesc = convNode->GetOpDesc()->GetOutputDesc(0);
    std::string format = TypeUtils::FormatToSerialString(convOutDesc.GetFormat());
    size_t coutAxis = format.find('C');
    FUSION_PASS_CHECK(coutAxis == std::string::npos,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get split axis, fusion failed."), return PARAM_INVALID);

    axis = static_cast<int32_t>(coutAxis);

    OP_LOGI(FUSED_OP_TYPE.c_str(), "split axis %d.", axis);
    return SUCCESS;
}

Status SameInputConv2dPass::GetReluInAxis(const ge::GeTensorDesc& reluInDesc, int32_t &axis) const
{
    std::string format = TypeUtils::FormatToSerialString(reluInDesc.GetFormat());
    size_t coutAxis = format.find('C');
    FUSION_PASS_CHECK(coutAxis == std::string::npos,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get relu in axis failed, fusion failed."), return PARAM_INVALID);

    axis = static_cast<int32_t>(coutAxis);

    OP_LOGI(FUSED_OP_TYPE.c_str(), "relu in axis %d.", axis);
    return SUCCESS;
}

Status SameInputConv2dPass::GetReluOutAxis(const ge::GeTensorDesc& reluOutDesc, int32_t &axis) const
{
    std::string format = TypeUtils::FormatToSerialString(reluOutDesc.GetFormat());
    size_t coutAxis = format.find('C');
    FUSION_PASS_CHECK(coutAxis == std::string::npos,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get relu out axis failed, fusion failed."), return PARAM_INVALID);
    
    axis = static_cast<int32_t>(coutAxis);

    OP_LOGI(FUSED_OP_TYPE.c_str(), "relu out axis %d.", axis);
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

Status SameInputConv2dPass::AddSplitNode(ge::ComputeGraph& graph, std::vector<ge::NodePtr>& newNodes,
    std::vector<ge::NodePtr>& convNodes) const
{
    std::vector<ge::NodePtr> reluNodes;
    for (auto& conv : convNodes) {
        reluNodes.emplace_back(conv->GetOutDataNodes().at(0));
    }

    std::vector<int32_t> sizeSplits;
    int32_t splitDim;
    FUSION_PASS_CHECK(GetSplitAttr(reluNodes, sizeSplits, splitDim) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get split attr failed, fusion failed."), return FAILED);

    std::string sizeSplitName = reluNodes.at(0)->GetName() + SPLIT_SIZE_CONST;
    ge::NodePtr sizeSplitNode = CreateSizeSplitNode(graph, newNodes, sizeSplits, sizeSplitName);
    FUSION_PASS_CHECK(sizeSplitNode == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "create size_splits node failed, fusion failed."), return FAILED);

    std::string dimSplitName = reluNodes.at(0)->GetName() + SPLIT_DIM_CONST;
    ge::NodePtr dimSplitNode = CreateDimSplitNode(graph, newNodes, splitDim, dimSplitName);
    FUSION_PASS_CHECK(dimSplitNode == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "create split_dim node failed, fusion failed."), return FAILED);

    ge::NodePtr splitNode = CreateSplitNode(graph, newNodes, reluNodes, sizeSplitNode, dimSplitNode);
    FUSION_PASS_CHECK(splitNode == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "create split node failed, fusion failed."), return FAILED);

    FUSION_PASS_CHECK(LinkSplitConst(sizeSplitNode, dimSplitNode, splitNode) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "link split const failed, fusion failed."), return FAILED);

    FUSION_PASS_CHECK(LinkReluSplit(graph, reluNodes, splitNode) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "link relu split failed, fusion failed."), return FAILED);

    OP_LOGI(FUSED_OP_TYPE.c_str(), "add split node success.");
    return SUCCESS;
}

Status SameInputConv2dPass::UpdateConvNode(ge::ComputeGraph& graph, const std::vector<ge::NodePtr>& convNodes) const
{
    ge::NodePtr updateConvNode = convNodes.at(0);

    // update conv shape
    int64_t dimValue;
    FUSION_PASS_CHECK(GetConvInDimValue(convNodes, dimValue) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get concat value failed, fusion failed."), return FAILED);

    int32_t convOutAxis;
    FUSION_PASS_CHECK(GetConvOutAxis(updateConvNode, convOutAxis) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get split axis failed, fusion failed."), return FAILED);
    auto convOutDesc = updateConvNode->GetOpDesc()->GetOutputDesc(0).Clone();
    FUSION_PASS_CHECK(SetShapeDims(convOutAxis, dimValue, convOutDesc) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "set shape dim failed, fusion failed."), return FAILED);
    updateConvNode->GetOpDesc()->UpdateOutputDesc(0, convOutDesc);

    auto convInFilterDesc = updateConvNode->GetOpDesc()->GetInputDesc("filter").Clone();
    int32_t concatAxis;
    FUSION_PASS_CHECK(GetConvInAxis(updateConvNode, concatAxis) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get concat axis failed, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(SetShapeDims(concatAxis, dimValue, convInFilterDesc) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "set shape dim failed, fusion failed."), return FAILED);
    updateConvNode->GetOpDesc()->UpdateInputDesc(FILTER_POS, convInFilterDesc);

    auto convInBiasDesc = updateConvNode->GetOpDesc()->GetInputDesc("bias").Clone();
    if (convInBiasDesc.IsValid() == GRAPH_SUCCESS) {
        FUSION_PASS_CHECK(SetShapeDims(0, dimValue, convInBiasDesc) != SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "set shape dim failed, fusion failed."), return FAILED);
        updateConvNode->GetOpDesc()->UpdateInputDesc(BIAS_POS, convInBiasDesc);
    }

    // update relu shape
    ge::NodePtr updateReluNode = updateConvNode->GetOutDataNodes().at(0);
    auto reluInDesc = updateReluNode->GetOpDesc()->GetInputDesc(0).Clone();
    int32_t reluInAxis;
    FUSION_PASS_CHECK(GetReluInAxis(reluInDesc, reluInAxis) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get concat axis failed, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(SetShapeDims(reluInAxis, dimValue, reluInDesc) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "set shape dim failed, fusion failed."), return FAILED);
    updateReluNode->GetOpDesc()->UpdateInputDesc(0, reluInDesc);

    auto reluOutDesc = updateReluNode->GetOpDesc()->GetOutputDesc(0).Clone();
    int32_t reluOutAxis;
    FUSION_PASS_CHECK(GetReluOutAxis(reluOutDesc, reluOutAxis) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get concat axis failed, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(SetShapeDims(reluOutAxis, dimValue, reluOutDesc) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "set shape dim failed, fusion failed."), return FAILED);
    updateReluNode->GetOpDesc()->UpdateOutputDesc(0, reluOutDesc);

    // remove edge conv-relu
    for (size_t i = 1; i < convNodes.size(); ++i) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "relu node in anchor %zu.", convNodes[i]->GetOutDataNodes().size());
        auto reluNode = convNodes[i]->GetOutDataNodes().at(0);
        auto reluInAnchor = reluNode->GetInDataAnchor(0);
        FUSION_PASS_CHECK(reluInAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "relu in anchor is null, fusion failed."), return FAILED);
        auto reluPeerOutAnchor = reluInAnchor->GetPeerOutAnchor();
        FUSION_PASS_CHECK(reluPeerOutAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "relu peer out data anchor is null, fusion failed."), return FAILED);
        FUSION_PASS_CHECK(GraphUtils::RemoveEdge(reluPeerOutAnchor, reluInAnchor) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "remove edge from input--conv failed, fusion failed."), return FAILED);

        FUSION_PASS_CHECK(graph.RemoveNode(reluNode) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "remove relu node failed, fusion failed."), return FAILED);
    }

    // remove edge input--conv
    for (size_t i = 1; i < convNodes.size(); ++i) {
        auto fmapConvInAnchor = convNodes[i]->GetInDataAnchor(0);
        FUSION_PASS_CHECK(fmapConvInAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "conv fmap anchor is null, fusion failed."), return FAILED);
        auto fmapPeerAnchor = fmapConvInAnchor->GetPeerOutAnchor();
        FUSION_PASS_CHECK(fmapPeerAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "fmap peer out data anchor is null, fusion failed."), return FAILED);
        FUSION_PASS_CHECK(GraphUtils::RemoveEdge(fmapPeerAnchor, fmapConvInAnchor) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "remove edge from input--conv failed, fusion failed."), return FAILED);

        FUSION_PASS_CHECK(graph.RemoveNode(convNodes[i]) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "remove conv node failed, fusion failed."), return FAILED);
    }

    OP_LOGD(FUSED_OP_TYPE.c_str(), "update conv node success.");
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

ge::NodePtr SameInputConv2dPass::AddFilterConcatNode(ge::ComputeGraph& graph, std::vector<ge::NodePtr>& newNodes,
    std::vector<ge::NodePtr>& convNodes) const
{
    std::vector<ge::NodePtr> filterNodes;
    for (auto& convNode : convNodes) {
        filterNodes.emplace_back(convNode->GetInDataNodes().at(FILTER_POS));
    }

    // create concat node
    ge::OpDescPtr concatDesc = nullptr;
    std::string concatName = convNodes.at(0)->GetName() + FILTER_CONCAT;
    FUSION_PASS_MAKE_SHARED(concatDesc = std::make_shared<ge::OpDesc>(concatName, CONCAT_TYPE), return nullptr);
    for (size_t i = 0; i < filterNodes.size(); ++i) {
        std::string concatName = "filter_conv_input_" + to_string(i);
        auto filterOutDesc = filterNodes[i]->GetOpDesc()->GetOutputDesc(0);
        FUSION_PASS_CHECK(concatDesc->AddInputDesc(concatName, filterOutDesc) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "concat add input desc failed, fusion failed."), return nullptr);
    }

    int64_t dimValue;
    FUSION_PASS_CHECK(GetConvInDimValue(convNodes, dimValue) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get concat value failed, fusion failed."), return nullptr);
    int32_t axis;
    FUSION_PASS_CHECK(GetConvInAxis(convNodes.at(0), axis) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get concat axis failed, fusion failed."), return nullptr);

    auto convInFilterDesc = convNodes.at(0)->GetOpDesc()->GetInputDesc("filter").Clone();
    FUSION_PASS_CHECK(SetShapeDims(axis, dimValue, convInFilterDesc) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "set shape dim failed, fusion failed."), return nullptr);

    FUSION_PASS_CHECK(concatDesc->AddOutputDesc("y", convInFilterDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "concat add output desc failed, fusion failed."), return nullptr);

    std::string concatDimName = convNodes.at(0)->GetName() + "_filter_concat_dim";
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

    OP_LOGD(FUSED_OP_TYPE.c_str(), "add filter concat node success.");
    return concatNode;
}

ge::NodePtr SameInputConv2dPass::AddBiasConcatNode(ge::ComputeGraph& graph, std::vector<ge::NodePtr>& newNodes,
    std::vector<ge::NodePtr>& convNodes) const
{
    if (convNodes.at(0)->GetOpDesc()->GetInputDesc("bias").IsValid() != GRAPH_SUCCESS) {
        return nullptr;
    }
    std::vector<ge::NodePtr> biasNodes;
    for (auto& convNode : convNodes) {
        biasNodes.emplace_back(convNode->GetInDataNodes().at(BIAS_POS));
    }

    // create concat node
    ge::OpDescPtr concatDesc = nullptr;
    std::string concatName = convNodes.at(0)->GetName() + BIAS_CONCAT;
    FUSION_PASS_MAKE_SHARED(concatDesc = std::make_shared<ge::OpDesc>(concatName, CONCAT_TYPE), return nullptr);
    for (size_t i = 0; i < biasNodes.size(); ++i) {
        std::string concatName = "bias_conv_input_" + to_string(i);
        auto biasOutDesc = biasNodes[i]->GetOpDesc()->GetOutputDesc(0);
        FUSION_PASS_CHECK(concatDesc->AddInputDesc(concatName, biasOutDesc) != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "concat add input desc failed, fusion failed."), return nullptr);
    }

    int64_t dimValue;
    FUSION_PASS_CHECK(GetConvInDimValue(convNodes, dimValue) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "get concat value failed, fusion failed."), return nullptr);

    auto convInBiasDesc = convNodes.at(0)->GetOpDesc()->GetInputDesc("bias").Clone();
    FUSION_PASS_CHECK(SetShapeDims(0, dimValue, convInBiasDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "set shape dim failed, fusion failed."), return nullptr);

    FUSION_PASS_CHECK(concatDesc->AddOutputDesc("y", convInBiasDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "concat add output desc failed, fusion failed."), return nullptr);

    std::string concatDimName = convNodes.at(0)->GetName() + "_bias_concat_dim";
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

    OP_LOGD(FUSED_OP_TYPE.c_str(), "add bias concat node success.");
    return concatNode;
}

Status SameInputConv2dPass::UpdateFilterConcat(const std::vector<ge::NodePtr>& convNodes,
    const std::vector<ge::NodePtr>& filterNodes, ge::NodePtr filterConcatNode) const
{
    // update concat filter edge
    for (size_t i = 0; i < filterNodes.size(); ++i) {
        // remove filter-conv
        auto convInAnchor = convNodes[i]->GetInDataAnchor(FILTER_POS);
        FUSION_PASS_CHECK(convInAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "conv data anchor is null, fusion failed."), return FAILED);
        auto peerFilterAnchor = convInAnchor->GetPeerOutAnchor();
        FUSION_PASS_CHECK(peerFilterAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "conv peer data anchor is null, fusion failed."), return FAILED);
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
    auto convInAnchor = convNodes.at(0)->GetInDataAnchor(FILTER_POS);
    FUSION_PASS_CHECK(convInAnchor == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "conv in anchor null, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(concatOutAnchor, convInAnchor) != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add concat-conv failed, fusion failed."), return FAILED);

    return SUCCESS;
}

Status SameInputConv2dPass::UpdateBiasConcat(const std::vector<ge::NodePtr>& convNodes,
    const std::vector<ge::NodePtr>& biasNodes, ge::NodePtr biasConcatNode) const
{
    // update concat bias edge
    FUSION_PASS_CHECK(biasConcatNode == nullptr, OP_LOGD(FUSED_OP_TYPE.c_str(), "no bias."), return SUCCESS);
    FUSION_PASS_CHECK(convNodes.size() != biasNodes.size(),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "conv and bias not match, fusion failed."), return FAILED);

    for (size_t i = 0; i < convNodes.size(); ++i) {
        // remove bias-conv
        auto convInAnchor = convNodes[i]->GetInDataAnchor(BIAS_POS);
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
    auto biasConvInAnchor = convNodes.at(0)->GetInDataAnchor(BIAS_POS);
    FUSION_PASS_CHECK(biasConvInAnchor == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "conv in anchor null, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(biasConcatOutAnchor, biasConvInAnchor) != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add concat-conv failed, fusion failed."), return FAILED);

    return SUCCESS;
}

Status SameInputConv2dPass::UpdateConcatNodes(std::vector<ge::NodePtr>& convNodes, ge::NodePtr filterConcatNode,
    ge::NodePtr biasConcatNode) const
{
    std::vector<ge::NodePtr> filterNodes;
    std::vector<ge::NodePtr> biasNodes;
    for (auto& convNode : convNodes) {
        filterNodes.emplace_back(convNode->GetInDataNodes().at(FILTER_POS));
        if (biasConcatNode != nullptr && convNode->GetInDataNodes().size() > BIAS_POS) {
            biasNodes.emplace_back(convNode->GetInDataNodes().at(BIAS_POS));
        }
    }

    FUSION_PASS_CHECK(UpdateFilterConcat(convNodes, filterNodes, filterConcatNode) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "update filter concat failed, fusion failed."), return FAILED);

    FUSION_PASS_CHECK(UpdateBiasConcat(convNodes, biasNodes, biasConcatNode) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "update bias concat failed, fusion failed."), return FAILED);

    return SUCCESS;
}

Status SameInputConv2dPass::AddConcatNodes(ge::ComputeGraph& graph, std::vector<ge::NodePtr>& newNodes,
    std::vector<ge::NodePtr>& convNodes) const
{
    auto filterConcat = AddFilterConcatNode(graph, newNodes, convNodes);
    FUSION_PASS_CHECK(filterConcat == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add filter concat node failed, fusion failed."), return FAILED);

    ge::NodePtr biasConcat = nullptr;
    if (convNodes.at(0)->GetOpDesc()->GetInputDesc("bias").IsValid() == GRAPH_SUCCESS) {
        biasConcat = AddBiasConcatNode(graph, newNodes, convNodes);
        FUSION_PASS_CHECK(biasConcat == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "add bias concat node failed, fusion failed."), return FAILED);
    }

    FUSION_PASS_CHECK(UpdateConcatNodes(convNodes, filterConcat, biasConcat) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add bias concat node failed, fusion failed."), return FAILED);

    return SUCCESS;
}

Status SameInputConv2dPass::LinkReluSplit(ge::ComputeGraph& graph, const std::vector<ge::NodePtr>& reluNodes,
    ge::NodePtr splitNode) const
{
    // remove edge relu-conv, add edge relu-split
    for (size_t i = 0; i < reluNodes.size(); ++i) {
        auto splitOutAnchor = splitNode->GetOutDataAnchor(i);
        FUSION_PASS_CHECK(splitOutAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "split out data anchor is null, fusion failed."), return FAILED);
        auto convOutAnchor = reluNodes[i]->GetOutDataAnchor(0);
        FUSION_PASS_CHECK(convOutAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "conv out data anchor is null, fusion failed."), return FAILED);
        auto convPeerInAnchor = convOutAnchor->GetPeerInDataAnchors();
        for (auto& nextAnchor : convPeerInAnchor) {
            FUSION_PASS_CHECK(GraphUtils::RemoveEdge(convOutAnchor, nextAnchor),
                OP_LOGE(FUSED_OP_TYPE.c_str(), "remove edge from conv--relu failed, fusion failed."), return FAILED);
            FUSION_PASS_CHECK(GraphUtils::AddEdge(splitOutAnchor, nextAnchor),
                OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from split--relu failed, fusion failed."), return FAILED);
        }
    }

    // add edge relu-split
    auto convOutAnchor = reluNodes.at(0)->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(convOutAnchor == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "conv out data anchor is null, fusion failed."), return FAILED);
    auto splitInAnchor = splitNode->GetInDataAnchor(0);
    FUSION_PASS_CHECK(splitInAnchor == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "split in data anchor is null, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(convOutAnchor, splitInAnchor),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from conv--split failed, fusion failed."), return FAILED);

    return SUCCESS;
}

REGISTER_PASS("SameInputConv2dPass", BUILT_IN_GRAPH_PASS, SameInputConv2dPass);
}
