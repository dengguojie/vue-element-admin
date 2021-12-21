/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permission and
 * limitations under the License.
 **/
 
#ifndef HARD_MAX_FUSION_PASS_H
#define HARD_MAX_FUSION_PASS_H
#include <string>
#include <vector>
#include "pattern_fusion_util.h"
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "graph/tensor.h"

namespace fe{
class HardMaxPass : public PatternFusionBasePass{
protected:
	vector<FusionPattern *>DefinePatterns() override;
	Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusion_nodes)override;
	Status CreateArgMaxDNode(ge::ComputeGraph &graph, ge::NodePtr &fused_node,ge::NodePtr &new_node,int64_t &depth,
			int64_t &dim);
	Status CreateOneHotDNode(ge::ComputeGraph &graph,ge::NodePtr &fused_node,ge::NodePtr &argmax_node,
		ge::NodePtr &new_node,int64_t depth,int64_t dim);
	Status SetConstDesc(vector<int64_t> &tensor_shape,ge::GeTensorDesc &tensor_desc,ge::GeTensorDesc &des_desc)const;
	Status OnValueConstNode(vector<int64_t> &on_value_tensor_shape,ge::GeTensorDesc &input_desc_one,
		ge::GeTensorPtr &assit_on_value_ptr, int32_t on_value, ge::GeTensorDesc &on_value_tensor_desc)const;
	Status OffValueConstNode(vector<int64_t> &off_value_tensor_shape,ge::GeTensorDesc &input_desc_one,
		ge::GeTensorPtr &assit_off_value_ptr, int32_t off_value, ge::GeTensorDesc &off_value_tensor_desc)const;
	Status AddEdgeToOneHotDForOut(ge::NodePtr &fused_node,ge::NodePtr &one_hot_d_node)const;
	Status RemoveFusedNode(ge::ComputeGraph &graph, ge::NodePtr &fused_node)const;
	
private:
	static const std::string PATTERN_FUSEDNODE;
};
} //  namespace fe

#endif //  HARD_MAX_FUSION_PASS_H

