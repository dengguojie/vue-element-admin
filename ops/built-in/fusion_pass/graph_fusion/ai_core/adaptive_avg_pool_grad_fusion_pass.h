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

/* !
 * \file adaptive_avg_pool_grad_fusion_pass.h
 * \brief adaptive_avg_pool_grad fusion pass
 */

#ifndef FE_ADAPTIVE_AVG_POOL_GRAD_FUSION_H
#define FE_ADAPTIVE_AVG_POOL_GRAD_FUSION_H

#include <string>
#include <vector>

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class AdaptiveAvgPoolGradFusionPass : public PatternFusionBasePass {
    protected:
    vector<FusionPattern *> DefinePatterns() override;
    Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &new_nodes) override;

    private:
    // format is NCHW
    vector<int64_t> input_dims;
    vector<int64_t> output_dims;
    vector<vector<int>> h_kernel_index;
    vector<vector<int>> w_kernel_index;
    const string FUSED_OP_TYPE = "AdaptiveAvgPool2dGrad";

    void GetNodeInfo(ge::NodePtr node);
    int StartIndex(int a, int b, int c) const;
    int EndIndex(int a, int b, int c) const;
    void GenKernelIndex(vector<vector<int>> &kernel, int inSize, int outSize) const;
    void GenBatchMatMulAssistMatrix(uint16_t &matrix, bool is_left_mat) const;
    bool GenVectorMulAssistMatrix(uint16_t &matrix) const;
    void SetTensorDesc(ge::GeTensorDesc &tensorDesc, vector<int64_t> &dims, const ge::DataType &dtype,
                       const ge::Format &format) const;
    Status CreateVectorMulWeightNode(ge::NodePtr &vmul_node) const;
    Status CreateBatchMatMulWeightNode(ge::NodePtr &matmul_node, vector<int64_t> &weight_dims, bool is_left_mat) const;
    ge::NodePtr AddNewNode(ge::ComputeGraph &graph, ge::OpDescPtr &op_desc, vector<ge::NodePtr> &new_nodes) const;
    ge::NodePtr AddVectorMulNode(ge::NodePtr avgpool_grad_node, ge::ComputeGraph &graph,
                                 vector<ge::NodePtr> &new_nodes) const;
    ge::NodePtr AddLeftMatmulNode(ge::NodePtr avgpool_grad_node, ge::NodePtr mul_node, ge::ComputeGraph &graph,
                                  vector<ge::NodePtr> &new_nodes) const;
    ge::NodePtr AddRightMatmulNode(ge::NodePtr avgpool_grad_node, ge::NodePtr left_matmul_node, ge::ComputeGraph &graph,
                                   vector<ge::NodePtr> &new_nodes) const;
};
}  // namespace fe

#endif  // FE_ADAPTIVE_AVG_POOL_GRAD_FUSION_H
