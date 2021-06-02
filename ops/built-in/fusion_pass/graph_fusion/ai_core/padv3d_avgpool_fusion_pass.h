/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file padv3d_avgpool_fusion_pass.h
 * \brief padv3d + avgpoolv2/avgpool3d fusion pass
 */
#ifndef _OPTIMIZER_FUSION_PADV3D_AVGPOOL_FUSION_H_
#define _OPTIMIZER_FUSION_PADV3D_AVGPOOL_FUSION_H_

#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class Padv3dAvgpoolFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;
private:
    Status CheckFormatAndPading(ge::Format& input_format, std::vector<std::vector<int64_t>>& paddings,
                              bool paddings_contiguous);
    void UpdateAttrPads(ge::Format& input_format, std::vector<std::vector<int64_t>>& paddings,
                        std::vector<int32_t>& new_pad, bool paddings_contiguous);
    const string FUSED_OP_TYPE = "Avgpool_Pad3d";
};
}  // namespace fe
#endif  // _OPTIMIZER_FUSION_PADV3D_AVGPOOL_FUSION_H_
