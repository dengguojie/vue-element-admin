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
 * \file same_input_conv2d_fusion_pass.h
 * \brief same_input_conv2d fusion pass
 */

#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_SAME_INPUT_CONV2D_FUSION_PASS_H
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_SAME_INPUT_CONV2D_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "graph/node.h"

namespace fe {

struct ConvFusionAttr {
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    int64_t groups;
    std::string format;
    int64_t offsetX;

    bool operator != (const ConvFusionAttr &attr) const
    {
        return ((this->pads != attr.pads) ||
            (this->strides != attr.strides) ||
            (this->dilations != attr.dilations) ||
            (this->groups != attr.groups) ||
            (this->format != attr.format) ||
            (this->offsetX != attr.offsetX));
    }
};

struct ConvFusionInput {
    std::vector<int64_t> kernel;
    std::vector<int64_t> bias;
    std::vector<int64_t> offset;
    std::string format;

    bool operator != (const ConvFusionInput &input)
    {
        return ((this->kernel != input.kernel) ||
            (this->bias.size() != input.bias.size()) ||
            (this->offset != input.offset) ||
            (this->format != input.format));
    }
};

class SameInputConv2dPass : public PatternFusionBasePass {
protected:
    std::vector<FusionPattern*> DefinePatterns() override;
    Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, std::vector<ge::NodePtr>& newNodes) override;

private:
    Status CheckFusion(ge::NodePtr inputNode, std::vector<ge::NodePtr>& convNodes) const;
    Status GetConvNodes(ge::NodePtr inputNode, std::vector<ge::NodePtr>& convNodes) const;
    Status CheckConvNodes(ge::NodePtr inputNode, const std::vector<ge::NodePtr>& convNodes) const;
    Status CheckConvAttr(std::vector<ge::NodePtr>& convNodes) const;
    ConvFusionAttr GetConvAttr(ge::NodePtr node) const;
    Status GetConvInput(ge::NodePtr conv, ConvFusionInput &convInput) const;
    Status CheckConvInputLink(std::vector<ge::NodePtr>& convNodes) const;
    Status CheckConvInputLink(ge::NodePtr node, std::vector<ge::NodePtr>& convNodes) const;

    Status GetSplitAttr(const std::vector<ge::NodePtr>& convNodes, std::vector<int32_t> &sizeSplits,
        int32_t &splitDim) const;
    Status GetConvInDimValue(const std::vector<ge::NodePtr>& convNodes, int64_t& dimValue) const;
    Status GetConvInAxis(ge::NodePtr convNode, int32_t &axis) const;
    Status GetConvOutAxis(ge::NodePtr convNode, int32_t &axis) const;
    Status GetReluInAxis(const ge::GeTensorDesc& reluInDesc, int32_t &axis) const;
    Status GetReluOutAxis(const ge::GeTensorDesc& reluOutDesc, int32_t &axis) const;

    ge::NodePtr CreateSizeSplitNode(ge::ComputeGraph& graph, std::vector<ge::NodePtr>& newNodes,
        const std::vector<int32_t>& sizeSplit, const std::string& name) const;
    ge::NodePtr CreateDimSplitNode(ge::ComputeGraph& graph, std::vector<ge::NodePtr>& newNodes,
        int32_t splitDim, const std::string& name) const;
    ge::NodePtr CreateSplitNode(ge::ComputeGraph& graph, std::vector<ge::NodePtr>& newNodes,
        std::vector<ge::NodePtr>& convNodes, ge::NodePtr& sizeSplitNode, ge::NodePtr& dimSplitNode) const;

    Status LinkSplitConst(ge::NodePtr sizeSplitNode, ge::NodePtr dimSplitNode,
        ge::NodePtr splitNode) const;
    Status SetShapeDims(int32_t dim, int64_t dimValue, ge::GeTensorDesc &tensorDesc) const;

    Status AddSplitNode(ge::ComputeGraph& graph, std::vector<ge::NodePtr>& newNodes,
        std::vector<ge::NodePtr>& convNodes) const;
    Status LinkReluSplit(ge::ComputeGraph& graph, const std::vector<ge::NodePtr>& convNodes,
        ge::NodePtr splitNode) const;

    ge::NodePtr CreateConcatDimNode(ge::ComputeGraph& graph, std::vector<ge::NodePtr>& newNodes,
        const std::string& name, int32_t dimValue) const;

    ge::NodePtr AddFilterConcatNode(ge::ComputeGraph& graph, std::vector<ge::NodePtr>& newNodes,
        std::vector<ge::NodePtr>& convNodes) const;
    ge::NodePtr AddBiasConcatNode(ge::ComputeGraph& graph, std::vector<ge::NodePtr>& newNodes,
        std::vector<ge::NodePtr>& convNodes) const;
    Status AddConcatNodes(ge::ComputeGraph& graph, std::vector<ge::NodePtr>& newNodes,
        std::vector<ge::NodePtr>& convNodes) const;

    Status UpdateFilterConcat(const std::vector<ge::NodePtr>& convNodes,
        const std::vector<ge::NodePtr>& filterNodes, ge::NodePtr filterConcatNode) const;
    Status UpdateBiasConcat(const std::vector<ge::NodePtr>& convNodes,
        const std::vector<ge::NodePtr>& biasNodes, ge::NodePtr biasConcatNode) const;
    Status UpdateConcatNodes(std::vector<ge::NodePtr>& convNodes, ge::NodePtr filterConcatNode,
        ge::NodePtr biasConcatNode) const;

    Status UpdateConvNode(ge::ComputeGraph& graph, const std::vector<ge::NodePtr>& convNodes) const;

    const std::string FUSED_OP_TYPE = "same_input_conv2d";
    const std::string PATTERN_INPUT = "input";
    const std::string PATTERN_CONV2D_0 = "conv2d0";
    const std::string PATTERN_CONV2D_2 = "conv2d2";
    const std::string PATTERN_RELU_0 = "relu0";
    const std::string CONV2D_TYPE = "Conv2D";
    const std::string RELU_TYPE = "Relu";
    const std::string SPLIT = "/same_input_conv_split";
    const std::string SPLIT_TYPE = "SplitV";
    const std::string SPLIT_SIZE_CONST = "/size_split_const";
    const std::string SPLIT_DIM_CONST = "/split_dim_const";
    const std::string FILTER_CONCAT = "/same_input_conv_filter_concat";
    const std::string BIAS_CONCAT = "/same_input_conv_bias_concat";
    const std::string CONCAT_TYPE = "ConcatV2";
    const std::string FILTER_HOST_TYPE = "ConvBnFilterHost";
    const std::string CONST_TYPE = "Const";
    const std::string CONSTANT_TYPE = "Constant";
};

}

#endif
