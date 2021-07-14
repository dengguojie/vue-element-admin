/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef FUSION_ENGINE_INC_COMMON_FUSION_PASS_UTIL_H_
#define FUSION_ENGINE_INC_COMMON_FUSION_PASS_UTIL_H_

#include <string>
#include <unordered_set>
namespace fe {

static const std::unordered_set<std::string> ForbiddenClosedPass = {
        // Graph Fusion
        "ArgMaxWithFusionPass",
        "COPYPass",
        "ADeformableConv2dPass",
        "DynamicGRUV2GradFusionPass",
        "DynamicRNNGradFusionPass",
        "DynamicRNNInsertTransposePass",
        "FlattenV2Pass",
        "HostBNFusionPass",
        "ALSTMFusionPass",
        "MapIndexFusionPass",
        "NormalizeFusionPass",
        "PassThroughFusionPass",
        "PassThroughSecondFusionPass",
        "PriorBoxPass",
        "ProposalFusionPass",
        "RNNFusionPass",
        "SpatialTransformerDPass",
        "SPPPass",
        "YoloPass",
        "YoloV2DetectionOutputPass",
        "YoloV3DetectionOutputV2Pass",
        "PermuteFusionPass",
        "CommonLSTMFusionPass",
        "PReluGradFusionPass",
        "ConstToAttrReduceSumFusion",
        "TfTagNoConstFoldingFusionPass",
        "TfMergeConv2DBackpropInputFusionPass",
        "TfMergeSubFusionPass",
        "AvgPoolQuantProcessFusionPass",
        "Conv2DQuantProcessFusionPass",
        "GroupConv2DQuantProcessFusionPass",
        "Conv2DTDQuantProcessFusionPass",
        "DeConvQuantProcessFusionPass",
        "DWConv2DQuantProcessFusionPass",
        "FCQuantProcessFusionPass",
        "MatmulV2QuantProcessFusionPass",
        "PoolingQuantProcessFusionPass",
        "BatchMatmulV2QuantProcessFusionPass",
        "V100NotRequantFusionPass",
        "V200NotRequantFusionPass",
        "DepthwiseInsertTransDataFusionPass",

        // UB Fusion
        "TbeCommonRules2FusionPass",
        "TbeCommonRules0FusionPass",
        "TbeAntiquantMaxpoolingFusionPass",
        "TbeConvRequantFusionPass",
        "TbeConvDequantS16FusionPass"
};
}  // namespace fe
#endif  // FUSION_ENGINE_INC_COMMON_FUSION_PASS_UTIL_H_
