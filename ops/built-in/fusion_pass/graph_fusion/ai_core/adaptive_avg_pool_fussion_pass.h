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
 * \file adaptive_avg_pool_fussion_pass.h
 * \brief adaptive_avg_pool fusion pass
 */

#ifndef ADAPTIVE_AVG_POOL2D_PASS_H
#define ADAPTIVE_AVG_POOL2D_PASS_H

#include <string>
#include <vector>

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class AdaptiveAvgPool2dPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                vector<ge::NodePtr> &new_nodes) override;

 private:
  Status AdaptiveValueGen(vector<int64_t> &input_shape,
                          vector<int64_t> &output_shape,
                          vector<float> &left_tensor,
                          vector<float> &right_tensor,
                          vector<float> &mul_tensor) const;
  Status SetConstDesc(vector<int64_t> &tensor_shape,
                      ge::GeTensorDesc &tensor_desc,
                      ge::GeTensorDesc &des_desc) const;
  Status RemoveNodes(ge::NodePtr &data_node, ge::ComputeGraph &graph) const;
  Status Bridge(ge::NodePtr &fuse_node, ge::NodePtr &one_node,
                ge::NodePtr &two_node, ge::NodePtr &mul_node) const;
  Status CreatOneNode(ge::NodePtr &one_node, ge::NodePtr &fuse_node,
                      ge::ComputeGraph &graph, vector<ge::NodePtr> &new_nodes,
                      ge::GeShape &bat_one_outshape) const;
  Status CreatTwoNode(ge::NodePtr &two_node, ge::NodePtr &fuse_node,
                      ge::ComputeGraph &graph, vector<ge::NodePtr> &new_nodes,
                      ge::GeShape &bat_one_outshape) const;
  Status CreatMulNode(ge::NodePtr &mul_node, ge::NodePtr &fuse_node,
                      ge::ComputeGraph &graph,
                      vector<ge::NodePtr> &new_nodes) const;
  Status CreatFuseNode(ge::NodePtr &fuse_node, vector<int64_t> &input_shape,
                       vector<int64_t> &output_shape,
                       vector<int64_t> &bat_one_shape) const;
  Status LeftConstNode(vector<int64_t> &left_tensor_shape,
                       ge::GeTensorDesc &input_desc1,
                       ge::GeTensorPtr &assit_left_ptr,
                       vector<float> &left_tensor,
                       ge::GeTensorDesc &left_tensor_desc) const;
  Status MidConstNode(vector<int64_t> &input_shape,
                      ge::GeTensorDesc &input_desc1,
                      ge::GeTensorPtr &assit_mid_ptr,
                      ge::GeTensorDesc &mid_tensor_desc) const;
  Status RightConstNode(vector<int64_t> &right_tensor_shape,
                        ge::GeTensorDesc &input_desc1,
                        ge::GeTensorPtr &assit_right_ptr,
                        vector<float> &right_tensor,
                        ge::GeTensorDesc &right_tensor_desc) const;
  Status MulConstNode(vector<int64_t> &output_shape,
                      ge::GeTensorDesc &input_desc1,
                      ge::GeTensorPtr &assit_mul_ptr, vector<float> &mul_tensor,
                      ge::GeTensorDesc &mul_tensor_desc) const;
  const string FUSED_OP_TYPE = "AdaptiveAvgPool2d";
};
}  // namespace fe
#endif  // ADAPTIVE_AVG_POOL2D_PASS_H
