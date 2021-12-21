/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
 * \file sub_sample_fusion_pass.h
 * \brief sub_sample --> sub_sample)
 */
#ifndef SUB_SAMPLE_FUSION_PASS_H
#define SUB_SAMPLE_FUSION_PASS_H
#include <string>
#include <vector>
#include "pattern_fusion_util.h"
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "graph/tensor.h"

namespace fe {
    class SubSamplePass : public PatternFusionBasePass {
        protected:
            vector<FusionPattern *> DefinePatterns() override;
            Status Fusion(ge::ComputeGraph &graph,
                          Mapping &mapping,
                          vector<ge::NodePtr> &fusion_nodes) override;
            void set_node_tensor_desc(ge::GeTensorDesc &tensorDesc,
                                      vector<int64_t> &dims,
                                      const ge::DataType &dtype,
                                      const ge::Format &format) const;
        private:
            static const std::string PATTERN_FUSEDNODE;
    };
}

#endif // SUB_SAMPLE_FUSION_PASS_H
