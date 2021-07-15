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
 * \file fusedbatchnorm_fusion_pass.h
 * \brief fusedbatchnorm fusion pass(BatchNorm --> BNTrainingReduce & BNTrainingUpdate)
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_FUSEDBATCHNORM3D_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_FUSEDBATCHNORM3D_FUSION_PASS_H_

#include <vector>
#include <string>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "fusedbatchnorm_fusion_pass.h"

namespace fe {
    class FusedBatchnorm3DFusionPass : public FusedBatchnormFusionPass{
        public:
        FusedBatchnorm3DFusionPass() {
            PASS_OP_TYPE_BATCHNORM = "BatchNorm3D";
            PASS_OP_TYPE_SUB = "Sub";
            PASS_OP_TYPE_BNREDUCE = "BN3DTrainingReduce";
            PASS_OP_TYPE_BNUPDATE = "BN3DTrainingUpdate";
            STREAM_LABEL = "_stream_label";
            FUSED_OP_TYPE = "BN3DTrainingReduce_BN3DTrainingUpdate";
        }
        ~FusedBatchnorm3DFusionPass() {
        }
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_FUSEDBATCHNORM3D_FUSION_PASS_H_
