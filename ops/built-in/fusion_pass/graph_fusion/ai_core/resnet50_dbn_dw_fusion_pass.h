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
 * \file resnet50_dbn_dw_fusion_pass.h
 * \brief resnet50_dbn_dw_fusion_pass pass
 */

#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_RESNET50_DBN_DW_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_RESNET50_DBN_DW_FUSION_PASS_H_

#include <vector>
#include <string>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class Resnet50DbnDwFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusion_nodes) override;

private:
  const string FUSED_OP_TYPE = "FusedDbnDw";
  Status SetOpDesc(ge::OpDescPtr &dw_op_desc, ge::OpDescPtr &dbn_op_desc,
                   ge::OpDescPtr &fused_dbn_dw_desc);
  Status CheckSupportCase(ge::OpDescPtr &dw_op_desc);
  vector<int64_t> GetNchwVec(vector<int64_t> &dim_info, ge::Format &origin_format);
};
} // namespace fe

#endif // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_RESNET50_DBN_DW_FUSION_PASS_H_