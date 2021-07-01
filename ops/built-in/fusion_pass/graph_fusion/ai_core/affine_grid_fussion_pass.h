/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file affine_grid_fussion_pass.h
 * \brief affine grid fusion pass
 */

#ifndef FE_AFFINE_GRID_FUSION_H
#define FE_AFFINE_GRID_FUSION_H

#include <string>
#include <vector>

#include "graph/tensor.h"
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "pattern_fusion_util.h"

namespace fe {
    class AffineGridFusionPass : public PatternFusionBasePass {
    protected:
        vector<FusionPattern *> DefinePatterns() override;
        Status Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                      vector<ge::NodePtr> &new_nodes) override;

    private:
        vector<int64_t> output_size_vector;
        vector<int64_t> assist_shape;
        vector<int64_t> theta_shape;
        vector<int64_t> bmm_output_shape;
        bool align_corner = false;
        const string FUSED_OP_TYPE = "AffineGrid";

        Status get_fuse_node_info(ge::NodePtr node);
        void get_node_const_value(const Tensor &const_tensor, const DataType &dtype,
                               std::vector<int64_t> &const_data);
        void init_graph_shape(vector<int64_t> &output_size,
                            vector<int64_t> &assist_shape,
                            vector<int64_t> &bmm_output_shape);

        void grid_linspace(int64_t num_step, bool align_corners,
                          vector<float> &grid_line);
        void gen_assist_vector_4d(const vector<int64_t> output_size,
                                vector<float> &grid, bool align_corners);
        void gen_assist_vector_5d(const vector<int64_t> output_size,
                                vector<float> &grid, bool align_corners);

        void assist_matrix_gen(vector<float> data, uint16_t *output);

        void set_node_tensor_desc(ge::GeTensorDesc &tensorDesc, vector<int64_t> &dims,
                               const ge::DataType &dtype,
                               const ge::Format &format) const;

        int64_t get_grid_size(const vector<int64_t> output_size);
        float get_step_size(int64_t s, bool align_corners);

        Status create_batch_mat_mul_weight_node(ge::NodePtr &matmul_node,
                                           vector<int64_t> &weight_dims,
                                           ge::GeTensorPtr &weight_ptr,
                                           bool is_assist);

        Status remove_nodes(ge::NodePtr &data_node, ge::ComputeGraph &graph);

        ge::NodePtr add_batch_matmul_node(ge::NodePtr affine_node,
                                       ge::ComputeGraph &graph,
                                       vector<ge::NodePtr> &new_nodes,
                                       bool &fail_status);

        ge::NodePtr add_new_node(ge::ComputeGraph &graph, ge::OpDescPtr &op_desc,
                               vector<ge::NodePtr> &new_nodes, bool &fail_status);
    };
} // namespace fe

#endif // FE_AFFINE_GRID_FUSION_H