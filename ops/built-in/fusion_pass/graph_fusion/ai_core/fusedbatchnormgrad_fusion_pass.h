/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
 * \file fusedbatchnormgrad_fusion_pass.h
 * \brief fusedbatchnormgrad fusion pass
 *   (fusedbatchnormgrad --> BNTrainingReduceGrad & BNTrainingUpdateGrad)
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_FUSEDBATCHNORMGRAD_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_FUSEDBATCHNORMGRAD_FUSION_PASS_H_

#include <vector>
#include <string>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {

class FusedBatchNormGradFusionPass : public PatternFusionBasePass {
    public:
    FusedBatchNormGradFusionPass() {
        FUSED_OP_TYPE = "BNTrainingReduceGrad_BNTrainingUpdateGrad";
        PATTERN_FUSEDBATCHNORMGRAD = "BatchNormGrad";

        PASS_OP_TYPE_BATCHNORMGRAD = "BatchNormGrad";
        PASS_OP_TYPE_BNREDUCEGRAD = "BNTrainingReduceGrad";
        PASS_OP_TYPE_BNUPDATEGRAD = "BNTrainingUpdateGrad";

        /* BatchNormGrad */
        BATCHNORMGRAD_ATTR_MODE = "mode";
        BATCHNORMGRAD_ATTR_EPSILON = "epsilon";
        BATCHNORMGRAD_ATTR_USE_GLOBAL_STATS = "use_global_stats";
        BATCHNORMGRAD_ATTR_SCALE = "scale";
        BATCHNORMGRAD_ATTR_BIAS = "bias";
        BATCHNORMGRAD_ATTR_TRAINING = "is_training";
        STREAM_LABEL = "_stream_label";
    }
    ~FusedBatchNormGradFusionPass() {
    }
    protected:
    vector<FusionPattern*> DefinePatterns() override;
    Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) override;

    vector<ge::NodePtr> GetNodesFromMapping(const string& id, Mapping& mapping);
    string FUSED_OP_TYPE;
    string PATTERN_FUSEDBATCHNORMGRAD;

    string PASS_OP_TYPE_BATCHNORMGRAD;
    string PASS_OP_TYPE_BNREDUCEGRAD;
    string PASS_OP_TYPE_BNUPDATEGRAD;

    /* BatchNormGrad */
    std::string BATCHNORMGRAD_ATTR_MODE;
    std::string BATCHNORMGRAD_ATTR_EPSILON;
    std::string BATCHNORMGRAD_ATTR_USE_GLOBAL_STATS;
    std::string BATCHNORMGRAD_ATTR_SCALE;
    std::string BATCHNORMGRAD_ATTR_BIAS;
    std::string BATCHNORMGRAD_ATTR_TRAINING;
    std::string STREAM_LABEL;
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_FUSEDBATCHNORMGRAD_FUSION_PASS_H_
