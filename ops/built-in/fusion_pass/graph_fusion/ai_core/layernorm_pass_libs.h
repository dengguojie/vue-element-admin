/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file layernorm_pass_libs.h
 * \brief used for layernorm & layernormgrad fusion
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LAYERNORM_LIBS_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LAYERNORM_LIBS_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "graph_optimizer/fusion_common/fusion_turbo.h"

namespace fe {
Status CheckNullPtr(const std::vector<ge::NodePtr>& pattern_nodes);
Status GetReduceOpAttr(std::vector<int64_t>& axes, bool& keep_dims, const ge::NodePtr& node);
Status CheckReduceOpAttr(const std::string& fused_op_type, const std::vector<int64_t>& axes_0, const bool keep_dims_1,
                         const ge::NodePtr& mean_node_1, const ge::NodePtr& mean_node_2);
Status SetLayerNormAttr(const std::string& fused_op_type, const ge::NodePtr& node,
                        const std::vector<int64_t>& axes_vec, const ge::NodePtr& add_1);
void GetInputRelations(Relations& input_relations, const std::vector<std::pair<ge::NodePtr, ge::NodePtr>>& pairs);
void GetOutputRelations(Relations& output_relations, const std::vector<ge::NodePtr>& nodes);
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LAYERNORM_LIBS_FUSION_PASS_H_
