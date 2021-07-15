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
 * \file fusedBatchNorm3DGrad_fusion_pass.h
 * \brief fusedBatchNorm3DGrad fusion pass
 *   (fusedBatchNorm3DGrad --> BN3DTrainingReduceGrad & BN3DTrainingUpdateGrad)
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_FUSEDBatchNorm3DGrad_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_FUSEDBatchNorm3DGrad_FUSION_PASS_H_

#include <vector>
#include <string>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "fusedbatchnormgrad_fusion_pass.h"

namespace fe {
class FusedBatchNorm3DGradFusionPass : public FusedBatchNormGradFusionPass {
    public:
        FusedBatchNorm3DGradFusionPass() {
            FUSED_OP_TYPE = "BN3DTrainingReduceGrad_BN3DTrainingUpdateGrad";
            PATTERN_FUSEDBATCHNORMGRAD = "BatchNorm3DGrad";

            PASS_OP_TYPE_BATCHNORMGRAD = "BatchNorm3DGrad";
            PASS_OP_TYPE_BNREDUCEGRAD = "BN3DTrainingReduceGrad";
            PASS_OP_TYPE_BNUPDATEGRAD = "BN3DTrainingUpdateGrad";

            /* BatchNorm3DGrad */
            BATCHNORMGRAD_ATTR_MODE = "mode";
            BATCHNORMGRAD_ATTR_EPSILON = "epsilon";
            BATCHNORMGRAD_ATTR_USE_GLOBAL_STATS = "use_global_stats";
            BATCHNORMGRAD_ATTR_SCALE = "scale";
            BATCHNORMGRAD_ATTR_BIAS = "bias";
            BATCHNORMGRAD_ATTR_TRAINING = "is_training";
            STREAM_LABEL = "_stream_label";
        }
        ~FusedBatchNorm3DGradFusionPass() {
        }
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_FUSEDBatchNorm3DGrad_FUSION_PASS_H_
