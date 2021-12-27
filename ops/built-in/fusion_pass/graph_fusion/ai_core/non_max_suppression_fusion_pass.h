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
 * \file non_max_suppression_fusion_pass.h
 * \brief non_max_suppressionv6 --> non_max_suppressionv6)
 */
#ifndef NON_MAX_SUPPRESSION_FUSION_PASS_H
#define NON_MAX_SUPPRESSION_FUSION_PASS_H
#include <string>
#include <vector>
#include "pattern_fusion_util.h"
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "graph/tensor.h"

namespace fe {
    class NonMaxSuppressionV6Pass : public PatternFusionBasePass {
        protected:
            vector<FusionPattern *> DefinePatterns() override;
            Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusion_nodes) override;

            Status SetConstDesc(vector<int64_t> &tensorShape,
                ge::GeTensorDesc &tensorDesc,
                const ge::GeTensorDesc &desDesc) const;
            Status IdxValueConstNode(vector<int64_t> &OnValueTensorShape,
                const ge::GeTensorDesc &inputDesc1,
                ge::GeTensorPtr &assitOnValuePtr,
                ge::GeTensorDesc &OnValueTensorDesc) const;
            bool GetConstValue(const Tensor &const_tensor, const DataType &dtype,
                std::vector<int32_t> &const_data);
        private:
            static const std::string PATTERN_FUSEDNODE;
    };
}

#endif
