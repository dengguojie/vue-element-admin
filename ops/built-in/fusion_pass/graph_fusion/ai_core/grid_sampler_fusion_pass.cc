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

#include "grid_sampler_fusion_pass.h"

#include <map>
#include <memory>
#include <numeric>
#include <typeinfo>

#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

namespace fe {
const std::string PATTERN_FUSEDNODE = "GridSampler2D";
const std::string FUSED_OP_TYPE = "GridSampler2D";
const int INDEX_N = 0;
const int INDEX_C = 1;
const int INDEX_H = 2;
const int INDEX_W = 3;

// weight node
template <class T>
static Status CreateWeight(ge::NodePtr &node, std::map<std::string, std::vector<int64_t>> &dims,
                           ge::GeTensorDesc &weight_desc, Status (*func)(T &, int, std::vector<int64_t> &)) {
    std::vector<int64_t> weight_dims = dims["weight"];
    int64_t size = std::accumulate(weight_dims.begin(), weight_dims.end(), 1, std::multiplies<int64_t>());
    OP_LOGI(FUSED_OP_TYPE.c_str(), "calculate weight size. %d", size);
    std::unique_ptr<T[]> data(new (std::nothrow) T[size]());

    FUSION_PASS_CHECK(weight_dims.size() < 3, OP_LOGE(FUSED_OP_TYPE.c_str(), "dim of weight should be greater than 3."),
                      return FAILED);

    // note: out_shape = N * Hout * Wout
    int out_shape = weight_dims[0] * weight_dims[1] * weight_dims[2];
    if (weight_dims.size() == 6) {  // it is KMatrix
        out_shape = out_shape * weight_dims[3];
    }
    std::vector<int64_t> values;
    std::map<std::string, std::vector<int64_t>>::iterator iter = dims.find("input");
    if (iter != dims.end()) {
        values = iter->second;
    }

    FUSION_PASS_CHECK(func(*data.get(), out_shape, values) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "create weight failed."), return FAILED);

    ge::GeTensorPtr weight_ptr =
        std::make_shared<ge::GeTensor>(weight_desc, reinterpret_cast<uint8_t *>(data.get()), size * sizeof(T));
    FUSION_PASS_CHECK(!weight_ptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "create weight failed."), return FAILED);

    std::vector<ge::GeTensorPtr> weights = {weight_ptr};
    ge::OpDescUtils::SetWeights(node, weights);
    auto const_input_nodes = OpDescUtils::GetConstInputs(node);
    FUSION_PASS_CHECK(const_input_nodes.empty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "ConstInputNodes is empty."),
                      return PARAM_INVALID);
    const_input_nodes[0]->GetOpDesc()->SetType("Const");
    return SUCCESS;
}

// note: create [[-1,-1,1,1],[0,1,-1,0],[1,0,-1,0],[0,0,1,0]]
template <class T>
static Status GenKMatrix(T &mat, int size, std::vector<int64_t> &value = {}) {
    /* sub_assist is
      [[1, -1, -1, 1],
      [-1, 0, 1, 0],
      [-1, 1, 0, 0],
      [ 1, 0, 0, 0]]*/
    constexpr int LEN = 16;
    float sub_assist[LEN] = {1, -1, -1, 1, -1, 0, 1, 0, -1, 1, 0, 0, 1, 0, 0, 0};
    uint16_t sub_assist_fp16[LEN];
    for (int i = 0; i < LEN; i++) {
        fp16_t tmp;
        tmp = sub_assist[i];
        sub_assist_fp16[i] = tmp.val;
    }

    T *assist = &mat;
    // note: weight_dims[0] = N * C * Hout * Wout
    for (int n = 0; n < size; n++) {
        errno_t ret = memcpy_s(assist + n * LEN, LEN * sizeof(T), sub_assist_fp16, LEN * sizeof(uint16_t));
        FUSION_PASS_CHECK(ret != EOK, OP_LOGE(FUSED_OP_TYPE.c_str(), "memcpy_s fail."), return FAILED);
    }
    return SUCCESS;
}

// note: create numpy.ones([N,Hout,Wout,1])
template <class T>
static Status GenOnesMatrix(T &mat, int size, std::vector<int64_t> &value = {}) {
    T init_value = 1;
    if (typeid(T) == typeid(uint16_t)) {
        fp16_t tmp;
        tmp = 1.0;
        init_value = tmp.val;
    }

    T *assist = &mat;
    Status ret = NnSet(size, init_value, *reinterpret_cast<T *>(assist));
    if (ret != SUCCESS) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "create assist fail.");
    }
    return ret;
}

// note: create attr matrix (value is[Win, Hin], shape is [N,Hout,Wout,2])
template <class T>
static Status GenInputAttr(T &mat, int size, std::vector<int64_t> &input_dims = {}) {
    constexpr int LEN = 2;
    // note: the value of source_data is [Win, Hin]
    int32_t value[LEN] = {input_dims[INDEX_W], input_dims[INDEX_H]};
    T *source_data = new T[LEN];
    if (typeid(T) == typeid(uint16_t)) {
        fp16_t tmp;
        tmp = value[0];
        source_data[0] = tmp.val;
        tmp = value[1];
        source_data[1] = tmp.val;
    } else {
        source_data[0] = value[0];
        source_data[1] = value[1];
    }

    T *assist = &mat;
    // note: size is N * Hout * Wout * 2
    for (int n = 0; n < size; n++) {
        errno_t ret = memcpy_s(assist + n * LEN, LEN * sizeof(T), source_data, LEN * sizeof(T));
        FUSION_PASS_CHECK(ret != EOK, OP_LOGE(FUSED_OP_TYPE.c_str(), "memcpy_s fail."), return FAILED);
    }
    delete[] source_data;
    return SUCCESS;
}
std::vector<FusionPattern *> GridSamplerFusionPass::DefinePatterns() {
    std::vector<FusionPattern *> patterns;
    FusionPattern *pattern = new (std::nothrow) FusionPattern("GridSamplerFusion");
    if (!pattern) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "New a pattern object failed.");
        return patterns;
    }

    pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_OP_TYPE}).SetOutput(PATTERN_FUSEDNODE);
    patterns.push_back(pattern);
    return patterns;
}

Status GridSamplerFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, std::vector<ge::NodePtr> &new_node) {
    ge::NodePtr grid_sampler_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
    GetNodeInfo(grid_sampler_node);
    // add GridUnnormal
    // note: the result of grid unnormal part is position
    ge::NodePtr position_node = nullptr;
    // note: the result of concat is N matrix
    ge::NodePtr concat_node = nullptr;
    Status ret = AddGridUnnormalNode(grid_sampler_node, position_node, concat_node, graph, new_node);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "GridUnnormal:check failed."), return ret);

    // add ImageUnfold
    ge::NodePtr unfold_node = AddImageUnfoldNode(grid_sampler_node, position_node, graph, new_node);
    FUSION_PASS_CHECK(!unfold_node, OP_LOGE(FUSED_OP_TYPE.c_str(), "ImageUnfold:check failed."), return FAILED);

    ge::NodePtr broadcast_node = AddBroadCastNode(grid_sampler_node, concat_node, graph, new_node);
    FUSION_PASS_CHECK(!broadcast_node, OP_LOGE(FUSED_OP_TYPE.c_str(), "BroadCastNode:check failed."), return FAILED);

    ge::NodePtr right_matmul_node = AddRightMatmulNode(grid_sampler_node, broadcast_node, graph, new_node);
    FUSION_PASS_CHECK(!right_matmul_node, OP_LOGE(FUSED_OP_TYPE.c_str(), "RightMatMulNode:check failed."),
                      return FAILED);

    ge::NodePtr left_matmul_node =
        AddLeftMatmulNode(grid_sampler_node, unfold_node, right_matmul_node, graph, new_node);
    FUSION_PASS_CHECK(!left_matmul_node, OP_LOGE(FUSED_OP_TYPE.c_str(), "LeftMatMulNode:check failed."), return FAILED);

    // unlink all input of grid_sampler_node
    for (auto in_anchor_ptr : grid_sampler_node->GetAllInDataAnchors()) {
        if (in_anchor_ptr != nullptr) {
            in_anchor_ptr->UnlinkAll();
        }
    }

    // output Edge
    for (auto in_anchor_ptr : grid_sampler_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
        in_anchor_ptr->UnlinkAll();
        ge::GraphUtils::AddEdge(left_matmul_node->GetOutDataAnchor(0), in_anchor_ptr);
    }

    // remove grid_sampler_node from graph
    FUSION_PASS_CHECK(graph.RemoveNode(grid_sampler_node) != ge::GRAPH_SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "remove fusedNode node failed."), return FAILED);
    return SUCCESS;
}

void GridSamplerFusionPass::GetNodeInfo(ge::NodePtr node) {
    ge::OpDescPtr node_desc = node->GetOpDesc();

    input_dims = node_desc->GetInputDesc(0).GetShape().GetDims();
    std::vector<int64_t> grid_dims = node_desc->GetInputDesc(1).GetShape().GetDims();
    if (grid_dims.size() != 4) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "grid dims is %d, while excepted dims is 4.", grid_dims.size());
    }
    output_dims = {input_dims[INDEX_N], input_dims[INDEX_C], grid_dims[1], grid_dims[2]};
    inputX_type = node_desc->GetInputDesc(0).GetDataType();
    grid_type = node_desc->GetInputDesc(1).GetDataType();
    return;
}

void GridSamplerFusionPass::SetTensorDesc(ge::GeTensorDesc &tensor_desc, const std::vector<int64_t> &dims,
                                          const ge::Format &format, const ge::DataType &dtype) const {
    ge::GeShape shape(dims);
    tensor_desc.SetShape(shape);
    tensor_desc.SetFormat(format);
    tensor_desc.SetDataType(dtype);
    tensor_desc.SetOriginShape(shape);
    tensor_desc.SetOriginFormat(format);
    tensor_desc.SetOriginDataType(dtype);
    return;
}

void GridSamplerFusionPass::AddInputNodeDesc(ge::OpDescPtr opDesc, const std::string &name, const vector<int64_t> &dims,
                                             const ge::Format &format, const ge::DataType &dtype) const {
    ge::GeTensorDesc tensor_desc;
    SetTensorDesc(tensor_desc, dims, format, dtype);
    opDesc->AddInputDesc(name, tensor_desc);
    return;
}

void GridSamplerFusionPass::AddOutputNodeDesc(ge::OpDescPtr opDesc, const std::string &name,
                                              const vector<int64_t> &dims, const ge::Format &format,
                                              const ge::DataType &dtype) const {
    ge::GeTensorDesc tensor_desc;
    SetTensorDesc(tensor_desc, dims, format, dtype);
    opDesc->AddOutputDesc(name, tensor_desc);
    return;
}

ge::NodePtr GridSamplerFusionPass::AddNewNode(ge::ComputeGraph &graph, ge::OpDescPtr &op_desc,
                                              std::vector<ge::NodePtr> &new_nodes) const {
    ge::NodePtr node = graph.AddNode(op_desc);
    FUSION_PASS_CHECK(!node, OP_LOGE(FUSED_OP_TYPE.c_str(), "add new node failed."), return nullptr);
    new_nodes.push_back(node);
    return node;
}

ge::NodePtr GridSamplerFusionPass::AddImageUnfoldNode(ge::NodePtr grid_sampler_node, ge::NodePtr unnormal_node,
                                                      ge::ComputeGraph &graph,
                                                      std::vector<ge::NodePtr> &new_nodes) const {
    // create unfold desc
    ge::OpDescPtr unfold_op_desc =
        std::make_shared<ge::OpDesc>(grid_sampler_node->GetName() + "ImageUnfold", "ImageUnfold");

    // add input:x desc, input tensor of grid_sampler_node
    ge::OpDescPtr sampler_desc = grid_sampler_node->GetOpDesc();
    ge::GeTensorDesc x_desc = sampler_desc->GetInputDesc(0).Clone();
    unfold_op_desc->AddInputDesc("x", x_desc);

    // add input:position desc, output tensor of grid unnormal
    ge::GeTensorDesc position_desc = unnormal_node->GetOpDesc()->GetOutputDesc(1).Clone();
    unfold_op_desc->AddInputDesc("position", position_desc);

    // check padding mode
    std::string padding = "zeros";
    ge::AttrUtils::GetStr(sampler_desc, "padding_mode", padding);
    FUSION_PASS_CHECK(padding != "zeros",
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "padding mode is %s, which excepted is zeros.", padding.c_str()),
                      return nullptr);
    ge::AttrUtils::SetStr(unfold_op_desc, "padding_mode", padding);

    int64_t out_size = std::accumulate(output_dims.begin(), output_dims.end(), 1, std::multiplies<int64_t>());
    OP_LOGI(FUSED_OP_TYPE.c_str(), "ImageUnfold size is %d.", out_size);
    std::vector<int64_t> res_dim = {out_size, 4};
    AddOutputNodeDesc(unfold_op_desc, "Unfold", res_dim, ge::FORMAT_ND, inputX_type);

    // create ImageUnfold node
    ge::NodePtr unfold_node = AddNewNode(graph, unfold_op_desc, new_nodes);
    FUSION_PASS_CHECK(!unfold_node, OP_LOGE(FUSED_OP_TYPE.c_str(), "add unfold node fail."), return nullptr);

    // input Edge, connect input_data with unfold
    ge::GraphUtils::AddEdge(grid_sampler_node->GetInDataAnchor(0)->GetPeerOutAnchor(), unfold_node->GetInDataAnchor(0));
    // input Edge, connect position with unfold
    ge::GraphUtils::AddEdge(unnormal_node->GetOutDataAnchor(1), unfold_node->GetInDataAnchor(1));
    return unfold_node;
}

ge::NodePtr GridSamplerFusionPass::AddRightMatmulNode(ge::NodePtr grid_sampler_node, ge::NodePtr broadcast_node,
                                                      ge::ComputeGraph &graph,
                                                      std::vector<ge::NodePtr> &new_nodes) const {
    // create matmul desc : (KN).t = N.t * K.t
    ge::OpDescPtr bmm_op_desc =
        std::make_shared<ge::OpDesc>(grid_sampler_node->GetName() + "RightBatchMatmul", "BatchMatMul");
    // add input desc, the output0 tensor of grid unnormal node
    int batch_size = output_dims[INDEX_N] * output_dims[INDEX_C] * output_dims[INDEX_H] * output_dims[INDEX_W];
    std::vector<int64_t> diff_nd_dim = {batch_size, 4, 1};
    AddInputNodeDesc(bmm_op_desc, "diff", diff_nd_dim, ge::FORMAT_ND, ge::DT_FLOAT16);

    // add output desc: (KN).t shape: [N, C, Hout, Wout, 1, 4]
    std::vector<int64_t> res_nd_dim = {batch_size, 1, 4};
    AddOutputNodeDesc(bmm_op_desc, "RightBMM", res_nd_dim, ge::FORMAT_ND, ge::DT_FLOAT16);

    // attr
    ge::AttrUtils::SetBool(bmm_op_desc, "adj_x1", true);
    ge::AttrUtils::SetBool(bmm_op_desc, "adj_x2", true);

    // create matmul node
    ge::NodePtr matmul_node = AddNewNode(graph, bmm_op_desc, new_nodes);
    FUSION_PASS_CHECK(!matmul_node, OP_LOGE(FUSED_OP_TYPE.c_str(), "add right BMM fail."), return nullptr);

    std::vector<int64_t> w_nd_dims = {batch_size, 4, 4};
    ge::GeTensorDesc desc;
    SetTensorDesc(desc, w_nd_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
    std::map<std::string, std::vector<int64_t>> dim_map;
    dim_map["weight"] = {output_dims[INDEX_N], output_dims[INDEX_C], output_dims[INDEX_H], output_dims[INDEX_W], 4, 4};

    Status ret = CreateWeight<uint16_t>(matmul_node, dim_map, desc, GenKMatrix);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "create assist fail."), return nullptr);
    // input Edge
    ge::GraphUtils::AddEdge(broadcast_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(0));
    return matmul_node;
}

ge::NodePtr GridSamplerFusionPass::AddBroadCastNode(ge::NodePtr grid_sampler_node, ge::NodePtr concat_node,
                                                    ge::ComputeGraph &graph,
                                                    std::vector<ge::NodePtr> &new_nodes) const {
    // create broadcast desc [N,H,W,1,4] -> [N,C,H,W,1,4]
    ge::OpDescPtr broadcast_op_desc =
        std::make_shared<ge::OpDesc>(grid_sampler_node->GetName() + "BroadCast", "BroadcastToD");
    // add input desc, the output0 tensor of grid unnormal node
    std::vector<int64_t> data_dim = {output_dims[INDEX_N], 1, output_dims[INDEX_H], output_dims[INDEX_W], 1, 4};
    AddInputNodeDesc(broadcast_op_desc, "x", data_dim, ge::FORMAT_ND, ge::DT_FLOAT16);

    // add output desc. shape: [N, C, Hout, Wout, 1, 4]
    std::vector<int64_t> res_nd_dim = {
        output_dims[INDEX_N], output_dims[INDEX_C], output_dims[INDEX_H], output_dims[INDEX_W], 1, 4};
    AddOutputNodeDesc(broadcast_op_desc, "y", res_nd_dim, ge::FORMAT_ND, ge::DT_FLOAT16);

    // attr
    ge::AttrUtils::SetListInt(broadcast_op_desc, "shape", res_nd_dim);

    // create matmul node
    ge::NodePtr broadcast_node = AddNewNode(graph, broadcast_op_desc, new_nodes);
    FUSION_PASS_CHECK(!broadcast_node, OP_LOGE(FUSED_OP_TYPE.c_str(), "add broadcast fail."), return nullptr);
    // input Edge
    ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0), broadcast_node->GetInDataAnchor(0));
    return broadcast_node;
}

ge::NodePtr GridSamplerFusionPass::AddLeftMatmulNode(ge::NodePtr grid_sampler_node, ge::NodePtr unfold_node,
                                                     ge::NodePtr rbmm_node, ge::ComputeGraph &graph,
                                                     std::vector<ge::NodePtr> &new_nodes) const {
    // create matmul desc : Q*rightBMM.t
    ge::OpDescPtr bmm_op_desc =
        std::make_shared<ge::OpDesc>(grid_sampler_node->GetName() + "LeftBatchMatmul", "BatchMatMul");
    // add input0 desc, the output tensor of unfold node

    int batch_size = output_dims[INDEX_N] * output_dims[INDEX_C] * output_dims[INDEX_H] * output_dims[INDEX_W];

    std::vector<int64_t> matmul_input_nd_dim = {batch_size, 1, 4};
    AddInputNodeDesc(bmm_op_desc, "unfold", matmul_input_nd_dim, ge::FORMAT_ND, ge::DT_FLOAT16);

    // add input1 desc, the output tensor of broadcast node
    AddInputNodeDesc(bmm_op_desc, "broadcast", matmul_input_nd_dim, ge::FORMAT_ND, ge::DT_FLOAT16);

    // add output desc. shape: [N, C, Hout, Wout, 1, 1]
    std::vector<int64_t> res_nd_dim = {batch_size, 1, 1};
    AddOutputNodeDesc(bmm_op_desc, "LeftBMM", res_nd_dim, ge::FORMAT_ND, inputX_type);

    // attr
    ge::AttrUtils::SetBool(bmm_op_desc, "adj_x1", false);
    ge::AttrUtils::SetBool(bmm_op_desc, "adj_x2", true);

    // create matmul node
    ge::NodePtr matmul_node = AddNewNode(graph, bmm_op_desc, new_nodes);
    FUSION_PASS_CHECK(!matmul_node, OP_LOGE(FUSED_OP_TYPE.c_str(), "add right BMM fail."), return nullptr);
    // input Edge
    ge::GraphUtils::AddEdge(unfold_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(rbmm_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(1));
    return matmul_node;
}

Status GridSamplerFusionPass::AddGridUnnormalNode(ge::NodePtr grid_sampler_node, ge::NodePtr &unnormal_part_node,
                                                  ge::NodePtr &concat_node, ge::ComputeGraph &graph,
                                                  vector<ge::NodePtr> &new_nodes) const {
    // note: the result of grid unnormal part is position
    unnormal_part_node = AddGridUnnormalPartNode(grid_sampler_node, graph, new_nodes);
    FUSION_PASS_CHECK(!unnormal_part_node, OP_LOGE(FUSED_OP_TYPE.c_str(), "add unnormal part node fail."),
                      return FAILED);

    ge::NodePtr split_node = AddSplitNode(grid_sampler_node, unnormal_part_node, graph, new_nodes);
    FUSION_PASS_CHECK(!split_node, OP_LOGE(FUSED_OP_TYPE.c_str(), "add split node fail."), return FAILED);

    ge::NodePtr vmul_node = AddVmulNode(grid_sampler_node, split_node, graph, new_nodes);
    FUSION_PASS_CHECK(!vmul_node, OP_LOGE(FUSED_OP_TYPE.c_str(), "add vmul node fail."), return FAILED);

    // note: the result of concat is N matrix
    concat_node = AddConcatNode(grid_sampler_node, vmul_node, unnormal_part_node, graph, new_nodes);
    FUSION_PASS_CHECK(!concat_node, OP_LOGE(FUSED_OP_TYPE.c_str(), "add concat node fail."), return FAILED);
    return SUCCESS;
}

ge::NodePtr GridSamplerFusionPass::AddGridUnnormalPartNode(ge::NodePtr grid_sampler_node, ge::ComputeGraph &graph,
                                                           std::vector<ge::NodePtr> &new_nodes) const {
    ge::OpDescPtr unnormal_op_desc =
        std::make_shared<ge::OpDesc>(grid_sampler_node->GetName() + "GridUnnormalPart", "GridUnnormal");
    // add input: grid desc, input tensor of grid_sampler_node
    ge::OpDescPtr sampler_desc = grid_sampler_node->GetOpDesc();
    ge::GeTensorDesc grid_desc = sampler_desc->GetInputDesc(1).Clone();
    unnormal_op_desc->AddInputDesc("x", grid_desc);

    // add output desc.
    unnormal_op_desc->AddOutputDesc("diff", grid_desc);
    ge::GeTensorDesc position_desc = sampler_desc->GetInputDesc(1).Clone();
    position_desc.SetDataType(ge::DT_INT32);
    position_desc.SetOriginDataType(ge::DT_INT32);
    unnormal_op_desc->AddOutputDesc("position", position_desc);

    // check align_corners mode
    bool align_corners = false;
    ge::AttrUtils::GetBool(sampler_desc, "align_corners", align_corners);
    OP_LOGI(FUSED_OP_TYPE.c_str(), "align_corners is %d.", align_corners);
    ge::AttrUtils::SetBool(unnormal_op_desc, "align_corners", align_corners);

    // create unnormal node
    ge::NodePtr unnormal_node = AddNewNode(graph, unnormal_op_desc, new_nodes);
    FUSION_PASS_CHECK(!unnormal_node, OP_LOGE(FUSED_OP_TYPE.c_str(), "add unnormal part fail."), return nullptr);

    Status ret = SUCCESS;
    std::vector<int64_t> dim = {output_dims[INDEX_N], output_dims[INDEX_H], output_dims[INDEX_W], 2};
    std::map<std::string, std::vector<int64_t>> dim_map;
    dim_map["weight"] = dim;
    dim_map["input"] = input_dims;
    if (grid_desc.GetDataType() == ge::DT_FLOAT) {
        ret = CreateWeight<float>(unnormal_node, dim_map, grid_desc, GenInputAttr);
    } else {
        ret = CreateWeight<uint16_t>(unnormal_node, dim_map, grid_desc, GenInputAttr);
    }
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "create assist fail."), return nullptr);
    // input Edge
    ge::GraphUtils::AddEdge(grid_sampler_node->GetInDataAnchor(1)->GetPeerOutAnchor(),
                            unnormal_node->GetInDataAnchor(0));
    return unnormal_node;
}

ge::NodePtr GridSamplerFusionPass::AddSplitNode(ge::NodePtr grid_sampler_node, ge::NodePtr unnormal_part_node,
                                                ge::ComputeGraph &graph, std::vector<ge::NodePtr> &new_nodes) const {
    // create split desc
    ge::OpDescPtr split_desc = std::make_shared<ge::OpDesc>(grid_sampler_node->GetName() + "SplitVD", "SplitVD");

    // add input
    ge::GeTensorDesc unnormal_desc = unnormal_part_node->GetOpDesc()->GetOutputDesc(0).Clone();
    split_desc->AddInputDesc("x", unnormal_desc);

    // add output desc split_y and split_x, shape:{n, h, w, 1}
    ge::GeTensorDesc out_tensor_desc;
    std::vector<int64_t> res_dim = {output_dims[INDEX_N], output_dims[INDEX_H], output_dims[INDEX_W], 1};
    AddOutputNodeDesc(split_desc, "split_y", res_dim, ge::FORMAT_ND, grid_type);
    AddOutputNodeDesc(split_desc, "split_x", res_dim, ge::FORMAT_ND, grid_type);

    // attr
    vector<int64_t> size_splits{1, 1};
    ge::AttrUtils::SetListInt(split_desc, "size_splits", size_splits);
    ge::AttrUtils::SetInt(split_desc, "split_dim", -1);
    ge::AttrUtils::SetInt(split_desc, "num_split", 2);

    // create split node
    ge::NodePtr split_node = AddNewNode(graph, split_desc, new_nodes);
    FUSION_PASS_CHECK(!split_node, OP_LOGE(FUSED_OP_TYPE.c_str(), "add split fail."), return nullptr);
    // Edge
    ge::GraphUtils::AddEdge(unnormal_part_node->GetOutDataAnchor(0), split_node->GetInDataAnchor(0));
    return split_node;
}

ge::NodePtr GridSamplerFusionPass::AddVmulNode(ge::NodePtr grid_sampler_node, ge::NodePtr split_node,
                                               ge::ComputeGraph &graph, std::vector<ge::NodePtr> &new_nodes) const {
    // create vmul desc
    ge::OpDescPtr mul_desc = std::make_shared<ge::OpDesc>(grid_sampler_node->GetName() + "mul", "Mul");

    // add input
    ge::GeTensorDesc split1_desc = split_node->GetOpDesc()->GetOutputDesc(0).Clone();
    mul_desc->AddInputDesc("dy", split1_desc);
    ge::GeTensorDesc split2_desc = split_node->GetOpDesc()->GetOutputDesc(1).Clone();
    mul_desc->AddInputDesc("dx", split2_desc);

    // add output desc
    mul_desc->AddOutputDesc("dxy", split1_desc);

    // create split node
    ge::NodePtr mul_node = AddNewNode(graph, mul_desc, new_nodes);
    FUSION_PASS_CHECK(!mul_node, OP_LOGE(FUSED_OP_TYPE.c_str(), "add vmul fail."), return nullptr);

    // Edge
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(0), mul_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(1), mul_node->GetInDataAnchor(1));
    return mul_node;
}

ge::NodePtr GridSamplerFusionPass::AddConcatNode(ge::NodePtr grid_sampler, ge::NodePtr vmul_node,
                                                 ge::NodePtr unnormal_part, ge::ComputeGraph &graph,
                                                 std::vector<ge::NodePtr> &new_nodes) const {
    // create concat desc
    ge::OpDescPtr concat_desc = std::make_shared<ge::OpDesc>(grid_sampler->GetName() + "Concat", "ConcatD");

    // note: input desc
    ge::GeTensorDesc dxy_desc = vmul_node->GetOpDesc()->GetOutputDesc(0).Clone();
    concat_desc->AddInputDesc("dxy", dxy_desc);
    ge::GeTensorDesc dx_dy_desc = unnormal_part->GetOpDesc()->GetOutputDesc(0).Clone();
    concat_desc->AddInputDesc("dx_dy", dx_dy_desc);

    // output concat, shape:{N, Hout, Wout, 4}
    std::vector<int64_t> res_dim = {output_dims[INDEX_N], output_dims[INDEX_H], output_dims[INDEX_W], 4};
    AddOutputNodeDesc(concat_desc, "diff", res_dim, ge::FORMAT_ND, grid_type);

    // attr
    ge::AttrUtils::SetInt(concat_desc, "concat_dim", 3);
    ge::AttrUtils::SetInt(concat_desc, "N", 3);
    // create concat node
    ge::NodePtr concat_node = AddNewNode(graph, concat_desc, new_nodes);
    FUSION_PASS_CHECK(!concat_node, OP_LOGE(FUSED_OP_TYPE.c_str(), "add concat fail."), return nullptr);

    std::vector<int64_t> w_dims = {output_dims[INDEX_N], output_dims[INDEX_H], output_dims[INDEX_W], 1};
    ge::GeTensorDesc desc;
    SetTensorDesc(desc, w_dims, ge::FORMAT_ND, grid_type);

    Status ret = SUCCESS;
    std::map<std::string, std::vector<int64_t>> dim_map;
    dim_map["weight"] = w_dims;
    if (grid_type == ge::DT_FLOAT) {
        ret = CreateWeight<float>(concat_node, dim_map, desc, GenOnesMatrix);
    } else {
        ret = CreateWeight<uint16_t>(concat_node, dim_map, desc, GenOnesMatrix);
    }
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "create assist fail."), return nullptr);

    // Edge
    ge::GraphUtils::AddEdge(vmul_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(unnormal_part->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
    return concat_node;
}

// register pass rule
REGISTER_PASS("GridSamplerFusionPass", BUILT_IN_GRAPH_PASS, GridSamplerFusionPass);
}  // namespace fe
