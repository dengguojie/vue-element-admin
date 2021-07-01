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

#include "affine_grid_fussion_pass.h"

#include <algorithm>
#include <map>
#include <memory>
#include <numeric>

#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "tbe_fusion_pass_util.h"

using namespace std;
using namespace ge;

namespace {
    const string PATTERN_FUSEDNODE = "AffineGrid";
    const string FUSED_NODE = "AffineGrid";
    const int BATCH_INDEX = 0;
    const int CHANNEL_INDEX = 1;
    const int H_INDEX = 2;
    const int W_INDEX = 3;
    const int D_INDEX = 4;

} // namespace

namespace fe {

    // get grid total size
    int64_t AffineGridFusionPass::get_grid_size(const vector<int64_t> output_size) {
        int64_t size = 0;
        if (output_size.size() == 4) {
            size = output_size[0] * output_size[2] * output_size[3] * 3;
        } else {
            size = output_size[0] * output_size[2] * output_size[3] * output_size[4] * 4;
        }
        return size;
    }

    // get grid step
    float AffineGridFusionPass::get_step_size(int64_t s, bool align_corners) {
        if (s == 1) {
            // FIXME throw error
            throw "division by zero condition";
            return 0.0;
        }
        float step_size = 0;
        if (align_corners) {
            step_size = 2.0 / (s - 1);
        } else {
            step_size = 2.0 / s;
        }
        return step_size;
    }

    // generate grid line
    void AffineGridFusionPass::grid_linspace(int64_t num_step, bool align_corners,
                                            vector<float> &grid_line) {
        if (num_step <= 1) {
            grid_line.push_back(0.0);
            return;
        }
        float step_size = get_step_size(num_step, align_corners);
        float start_index = -1.0 + 0.5 * step_size;
        if (align_corners) {
            start_index = -1.0;
        }
        for (int64_t i = 0; i < num_step; i++) {
            float x = start_index + step_size * i;
            grid_line.push_back(x);
        }
        return;
    }

    // generate 4d grid const data
    void AffineGridFusionPass::gen_assist_vector_4d(const vector<int64_t> output_size,
                                                  vector<float> &grid,
                                                  bool align_corners) {
        // get grid line
        vector<float> grid_x;
        vector<float> grid_y;
        grid_linspace(output_size[3], align_corners, grid_x);
        grid_linspace(output_size[2], align_corners, grid_y);

        int64_t temp_size = output_size[2] * output_size[3] * 3;
        vector<float> grid_temp(temp_size, 0.0);

        int64_t index = 0;
        for (int64_t i = 0; i < output_size[2]; i++) {
            for (int64_t j = 0; j < output_size[3]; j++) {
                grid_temp[index++] = grid_x[j];
                grid_temp[index++] = grid_y[i];
                grid_temp[index++] = 1;
            }
        }
        int64_t offset = 0;
        for (int64_t i = 0; i < output_size[0]; i++) {
            offset = temp_size * i;
            copy(grid_temp.begin(), grid_temp.end(), &grid[offset]);
        }
    }

    // generate 5d grid const data
    void AffineGridFusionPass::gen_assist_vector_5d(const vector<int64_t> output_size,
                                                  vector<float> &grid,
                                                  bool align_corners) {
        vector<float> grid_x;
        vector<float> grid_y;
        vector<float> grid_z;
        grid_linspace(output_size[4], align_corners, grid_x);
        grid_linspace(output_size[3], align_corners, grid_y);
        grid_linspace(output_size[2], align_corners, grid_z);

        int64_t temp_size = output_size[2] * output_size[3] * output_size[4] * 4;
        vector<float> grid_temp(temp_size, 0.0);

        int64_t index = 0;
        for (int64_t i = 0; i < output_size[2]; i++) {
            for (int64_t j = 0; j < output_size[3]; j++) {
                for (int64_t k = 0; k < output_size[4]; k++) {
                    grid_temp[index++] = grid_x[k];
                    grid_temp[index++] = grid_y[j];
                    grid_temp[index++] = grid_z[i];
                    grid_temp[index++] = 1;
                }
            }
        }
        int64_t offset = 0;
        for (int64_t i = 0; i < output_size[0]; i++) {
            offset = temp_size * i;
            copy(grid_temp.begin(), grid_temp.end(), &grid[offset]);
        }
    }

    // get output size from const node
    void AffineGridFusionPass::get_node_const_value(const Tensor &const_tensor,
                                                 const DataType &dtype,
                                                 std::vector<int64_t> &const_data) {
        size_t size = 0;
        if (dtype == ge::DT_INT32) {
            int32_t *const_data_ptr = (int32_t *)const_tensor.GetData();
            size = const_tensor.GetSize() / sizeof(int32_t);
            for (size_t i = 0; i < size; ++i) {
                const_data.push_back((static_cast<int64_t>(*(const_data_ptr + i))));
            }
        } else {
            OP_LOGW("AffineGrid", "The output_size dtype only surpport INT32.");
            return;
        }
        return;
    }

    void AffineGridFusionPass::assist_matrix_gen(vector<float> data,
                                               uint16_t *output) {
        if (output == nullptr) {
            OP_LOGE("batchMatmul", "output pointer is null!");
            return;
        }
        auto size_data = data.size();
        for (size_t i = 0; i < size_data; i++) {
            fp16_t tmp;
            tmp = data[i];
            output[i] = tmp.val;
        }
        return;
    }

    Status AffineGridFusionPass::create_batch_mat_mul_weight_node(
        ge::NodePtr &matmul_node, vector<int64_t> &weight_dims,
        ge::GeTensorPtr &weight_ptr, bool is_assist) {
        int64_t size = std::accumulate(weight_dims.begin(), weight_dims.end(), 1, std::multiplies<int64_t>());
        OP_LOGI(FUSED_OP_TYPE.c_str(), "calculate weight size. %d", size);

        unique_ptr<uint16_t[]> data(new (std::nothrow) uint16_t[size]());
        const uint16_t init_value = 0;
        if (NnSet(size, init_value, *reinterpret_cast<uint16_t *>(data.get())) !=
            SUCCESS) {
            OP_LOGE(FUSED_OP_TYPE.c_str(), "NnSet data failed.");
            return FAILED;
        }

        if (is_assist) {
            int64_t n = this->get_grid_size(this->output_size_vector);
            vector<float> grid(n, 0.0);
            if (this->output_size_vector.size() == 4) {
                this->gen_assist_vector_4d(this->output_size_vector, grid,
                                         this->align_corner);
            } else {
                this->gen_assist_vector_5d(this->output_size_vector, grid,
                                         this->align_corner);
            }
            this->assist_matrix_gen(grid, data.get());
        }

        ge::GeTensorDesc desc;
        this->set_node_tensor_desc(desc, weight_dims, ge::DT_FLOAT16, ge::FORMAT_ND);
        weight_ptr = std::make_shared<ge::GeTensor>(
            desc, reinterpret_cast<uint8_t *>(data.get()), size * sizeof(uint16_t));
        if (!weight_ptr) {
            OP_LOGE(FUSED_OP_TYPE.c_str(), "create weight failed.");
            return FAILED;
        }
        return SUCCESS;
    }

    // init graph shape
    void AffineGridFusionPass::init_graph_shape(vector<int64_t> &output_size,
                                              vector<int64_t> &assist_shape,
                                              vector<int64_t> &bmm_output_shape) {
        size_t size = output_size.size();
        assist_shape.push_back(output_size[BATCH_INDEX]);
        bmm_output_shape.push_back(output_size[BATCH_INDEX]);
        if (size == 4) {
            // 4d
            assist_shape.push_back(output_size[H_INDEX] * output_size[W_INDEX]);
            assist_shape.push_back(W_INDEX);
            bmm_output_shape.push_back(output_size[H_INDEX] * output_size[W_INDEX]);
            bmm_output_shape.push_back(H_INDEX);
        } else {
            // 5d
            assist_shape.push_back(output_size[H_INDEX] * output_size[H_INDEX] *
                                   output_size[D_INDEX]);
            assist_shape.push_back(D_INDEX);
            bmm_output_shape.push_back(output_size[H_INDEX] * output_size[H_INDEX] *
                                       output_size[D_INDEX]);
            bmm_output_shape.push_back(W_INDEX);
        }
        return;
    }

    Status AffineGridFusionPass::remove_nodes(ge::NodePtr &data_node,
                                             ge::ComputeGraph &graph) {
        for (auto in_anchor : data_node->GetAllInDataAnchors()) {
            if (in_anchor != nullptr) {
                in_anchor->UnlinkAll();
            }
        }
        for (auto out_anchor : data_node->GetAllOutDataAnchors()) {
            if (out_anchor != nullptr) {
                out_anchor->UnlinkAll();
            }
        }
        FUSION_PASS_CHECK(graph.RemoveNode(data_node) != SUCCESS,
                          OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove data_node failed."),
                          return FAILED);
        return SUCCESS;
    }

    ge::NodePtr AffineGridFusionPass::add_new_node(ge::ComputeGraph &graph,
                                                 ge::OpDescPtr &op_desc,
                                                 vector<ge::NodePtr> &new_nodes,
                                                 bool &fail_status) {
        ge::NodePtr node = graph.AddNode(op_desc);
        if (!node) {
            OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                    node->GetName().c_str());
            fail_status = true;
        }
        new_nodes.push_back(node);
        return node;
    }

    void AffineGridFusionPass::set_node_tensor_desc(ge::GeTensorDesc &tensorDesc,
                                                 vector<int64_t> &dims,
                                                 const ge::DataType &dtype,
                                                 const ge::Format &format) const {
        ge::GeShape shape(dims);
        tensorDesc.SetShape(shape);
        tensorDesc.SetDataType(dtype);
        tensorDesc.SetFormat(format);
        tensorDesc.SetOriginShape(shape);
        tensorDesc.SetOriginDataType(dtype);
        tensorDesc.SetOriginFormat(format);
        return;
    }

    // get node info from fused node
    Status AffineGridFusionPass::get_fuse_node_info(ge::NodePtr node) {
        ge::OpDescPtr node_desc = node->GetOpDesc();

        // get theta shape
        ge::GeTensorDesc input_desc = node->GetOpDesc()->GetInputDesc(0);
        DataType input_dtype = input_desc.GetDataType();
        if (input_dtype == ge::DT_DOUBLE || input_dtype == ge::DT_INT64 || input_dtype == ge::DT_UINT64) {
            OP_LOGI("AffineGridFusionPass", "Type of input data is double, int64 or uint64");
            return NOT_CHANGED;
        }

        this->theta_shape = input_desc.GetShape().GetDims();

        ge::Tensor ouputSizeTensor;
        Operator op = ge::OpDescUtils::CreateOperatorFromNode(node);

        if (node_desc->MutableInputDesc(
                node_desc->GetInputIndexByName("output_size")) != nullptr) {
            if (op.GetInputConstData("output_size", ouputSizeTensor) != GRAPH_SUCCESS) {
                OP_LOGE("AffineGridFusionPass", "Get constValue failed of [output_size]");
                return FAILED;
            }
        }
        DataType dtype = op.GetInputDesc("output_size").GetDataType();
        // get output size
        get_node_const_value(ouputSizeTensor, dtype, this->output_size_vector);
        init_graph_shape(this->output_size_vector, this->assist_shape,
                       this->bmm_output_shape);

        // get attr align_corners
        if (op.GetAttr("align_corners", this->align_corner) != ge::GRAPH_SUCCESS) {
            OP_LOGW(op.GetName().c_str(), "GetOpAttr align_corners failed! Default align_corner is false");
        }
        return SUCCESS;
    }

    //  Operator: BatchMatMul(assist,input_data.T)
    ge::NodePtr AffineGridFusionPass::add_batch_matmul_node(
        ge::NodePtr affine_node, ge::ComputeGraph &graph,
        vector<ge::NodePtr> &new_nodes, bool &fail_status) {
        // create matmul desc
        ge::OpDescPtr matmul_op_desc = nullptr;
        FUSION_PASS_MAKE_SHARED(
            (matmul_op_desc = std::make_shared<ge::OpDesc>(affine_node->GetName() + "affineBatchMatmul", "BatchMatMul")),
            return nullptr);

        ge::GeTensorDesc out_tensor_desc;
        this->set_node_tensor_desc(out_tensor_desc, this->bmm_output_shape,
                                ge::DT_FLOAT16, ge::FORMAT_ND);
        matmul_op_desc->AddOutputDesc("BMM", out_tensor_desc);

        // attr
        ge::AttrUtils::SetBool(matmul_op_desc, "adj_x1", false);
        ge::AttrUtils::SetBool(matmul_op_desc, "adj_x2", true);

        // create matmul node
        ge::NodePtr matmul_node =
            this->add_new_node(graph, matmul_op_desc, new_nodes, fail_status);

        ge::GeTensorPtr assit_ptr = nullptr;
        ge::GeTensorPtr theta_ptr = nullptr;

        this->create_batch_mat_mul_weight_node(matmul_node, this->assist_shape, assit_ptr,
                                          true);
        this->create_batch_mat_mul_weight_node(matmul_node, this->theta_shape, theta_ptr,
                                          false);
        vector<ge::GeTensorPtr> weights = {assit_ptr, theta_ptr};
        ge::OpDescUtils::SetWeights(matmul_node, weights);
        auto const_input_nodes = OpDescUtils::GetConstInputs(matmul_node);

        NodePtr const_assist_input = const_input_nodes[0];
        const_assist_input->GetOpDesc()->SetType("Const");

        NodePtr const_theta_input = const_input_nodes[1];
        const_theta_input->GetOpDesc()->SetType("Const");
        // remove theta const node
        this->remove_nodes(const_theta_input, graph);

        // input Edge to bmm node
        ge::GraphUtils::AddEdge(affine_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                matmul_node->GetInDataAnchor(1));

        // output Edge
        for (auto in_anchor_ptr :
             affine_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
            in_anchor_ptr->UnlinkAll();
            ge::GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), in_anchor_ptr);
        }

        return matmul_node;
    }

    vector<FusionPattern *> AffineGridFusionPass::DefinePatterns() {
        vector<FusionPattern *> patterns;
        FusionPattern *pattern = new (std::nothrow) FusionPattern("AffineGridFusion");
        if (!pattern) {
            OP_LOGE(FUSED_OP_TYPE.c_str(), "New a pattern object failed.");
            return patterns;
        }
        // FUSED_NODE should be the OpType
        pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE})
            .SetOutput(PATTERN_FUSEDNODE);
        patterns.push_back(pattern);
        return patterns;
    }

    Status AffineGridFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                                        vector<ge::NodePtr> &new_node) {
        OP_LOGD(FUSED_OP_TYPE.c_str(), "Define AffineGridFusionPass fusion begin.");
        ge::NodePtr affine_node =
            this->GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
        Status ret = this->get_fuse_node_info(affine_node);
        if (ret != SUCCESS) {
            return ret;
        }

        bool is_failure = false;

        // add batch_mat_mul
        ge::NodePtr batch_matmul_node =
            this->add_batch_matmul_node(affine_node, graph, new_node, is_failure);
        if (is_failure) {
            OP_LOGE(FUSED_OP_TYPE.c_str(),
                    "BatchMatMulNode:check failed, fusion failed.");
            return FAILED;
        }

        // unlink all input of grad_node
        for (auto inAnchor : affine_node->GetAllInDataAnchors()) {
            if (inAnchor != nullptr) {
                inAnchor->UnlinkAll();
            }
        }

        // remove affine_node from graph
        if (graph.RemoveNode(affine_node) != ge::GRAPH_SUCCESS) {
            OP_LOGE(FUSED_OP_TYPE.c_str(), "remove fusedNode node[%s] failed",
                    affine_node->GetName().c_str());
            return FAILED;
        }
        return SUCCESS;
    }
    // register pass rule
    REGISTER_PASS("AffineGridFusionPass", BUILT_IN_GRAPH_PASS,
                  AffineGridFusionPass);
} // namespace fe