/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file spatial_transformer_d_fusion_pass.h
 * \brief
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_SPATIAL_TRANSFORMER_D_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_SPATIAL_TRANSFORMER_D_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class SpatialTransformerDPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;
  Status TbeFusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes);
  Status AicpuFusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes);

 private:
  Status StnPreAddConst(ge::NodePtr& thisNode, ge::OpDescPtr& thisOpDesc);
  Status StnHIndexFP16(const int32_t h, const int32_t w, uint16_t* output1);
  Status StnWIndexFP16(const int32_t h, const int32_t w, uint16_t* output1);
  Status MakeStnComputeLayer(ge::OpDescPtr& thisOpDesc, const ge::OpDescPtr& bottomOpDesc,
                             const ge::OpDescPtr& formerOpDesc);
  Status MakeStnPreLayer(ge::OpDescPtr& thisOpDesc, const ge::OpDescPtr& formerOpDesc, bool hasInput1);
  const string TBE_FUSED_OP_TYPE = "StnPre_StnCompute";
  const string AICPU_FUSED_OP_TYPE = "Aicpu_SpatialTransformer";
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_SPATIAL_TRANSFORMER_D_FUSION_PASS_H_
