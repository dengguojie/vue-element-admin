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
 * \file matmul_cast_fusion_pass.h
 * \brief matmul cast fusion (Matmul--Cast)
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_MATMUL_CAST_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_MATMUL_CAST_FUSION_PASS_H_

#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class MatmulCastFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusion_nodes) override;

private:
  Status LinkOutputEdgeWithoutControl(const ge::NodePtr &matmul_node, const ge::NodePtr &cast_node) const;
  Status IsMatch(const ge::NodePtr &matmul_node, const ge::NodePtr &cast_node) const;
  Status DoFusion(const ge::NodePtr &matmul_node, ge::DataType &matmul_output_dtype,
                  ge::DataType &matmul_output_ori_dtype) const;
  void RestoreDtype(ge::NodePtr &matmul_node, const ge::DataType &matmul_output_dtype,
                    const ge::DataType &matmul_output_ori_dtype) const;
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_MATMUL_CAST_FUSION_PASS_H_
