/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
 * \file multi_head_attention_grad_fusion_pass.cc
 *
 */

#include "multi_head_attention_grad_fusion_pass.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "fp16_t.hpp"

using namespace std;
using namespace ge;

namespace fe {
static void SetNDTensorDesc(ge::GeTensorDesc &tensorDesc, const vector<int64_t> &ori_dims, const ge::DataType dtype = DT_FLOAT16) {
    tensorDesc.SetShape(ge::GeShape(ori_dims));
    tensorDesc.SetDataType(dtype);
    tensorDesc.SetFormat(FORMAT_ND);
    tensorDesc.SetOriginShape(ge::GeShape(ori_dims));
    tensorDesc.SetOriginDataType(dtype);
    tensorDesc.SetOriginFormat(FORMAT_ND);
}

static void SetNZTensorDesc(ge::GeTensorDesc &tensorDesc, const vector<int64_t> &ori_dims, const ge::DataType dtype = DT_FLOAT16) {
    vector<int64_t> dims;
    int32_t dim = ori_dims.size();
    for (auto i = 0; i < dim - 2; i++) {
        dims.push_back(ori_dims[i]);
    }
    dims.push_back((ori_dims[dim-1] + 15) / 16);
    dims.push_back((ori_dims[dim-2] + 15) / 16);
    dims.push_back(16);
    dims.push_back(16);
    tensorDesc.SetShape(ge::GeShape(dims));
    tensorDesc.SetDataType(dtype);
    tensorDesc.SetFormat(FORMAT_FRACTAL_NZ);
    tensorDesc.SetOriginShape(ge::GeShape(ori_dims));
    tensorDesc.SetOriginDataType(dtype);
    tensorDesc.SetOriginFormat(FORMAT_ND);
}

vector<FusionPattern*> MultiHeadAttentionGradFusionPass::DefinePatterns()
{
    vector<FusionPattern*> patterns;
    FusionPattern* pattern = new (std::nothrow) FusionPattern("MultiHeadAttentionGradFusionPass");
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter MultiHeadAttentionGradFusionPass");
    FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                        return patterns);
    pattern->AddOpDesc(FUSED_OP_TYPE, {FUSED_OP_TYPE}).SetOutput(FUSED_OP_TYPE);
    patterns.push_back(pattern);
    return patterns;
}

template<typename _InAnchor, typename _OutAnchor>
static Status AddNodeLinkOut(_InAnchor in_anchor, _OutAnchor out_anchor, const string& out_node_name) {
    // link out
    OP_LOGI("MultiHeadAttentionGrad Define %s link out begin", out_node_name.c_str());
    for (auto anchor : out_anchor->GetPeerInDataAnchors()) {
        GraphUtils::RemoveEdge(out_anchor, anchor);
        GraphUtils::AddEdge(in_anchor, anchor);
    }
    OP_LOGI("MultiHeadAttentionGrad Define %s link out end", out_node_name.c_str());
    return SUCCESS;
}

template<typename _InAnchor1, typename _InAnchor2>
static Status AddMatmulNode(ge::ComputeGraph& graph, const ge::GeTensorDesc& x1_desc, const ge::GeTensorDesc& x2_desc,
    const ge::GeTensorDesc& y_desc, ge::NodePtr& new_node, bool transpose_x1,
    bool transpose_x2, const string& node_name, _InAnchor1 in_anchor1, _InAnchor2 in_anchor2)
{
    OP_LOGI("MultiHeadAttentionGrad Define %s begin", node_name.c_str());
    OpDescPtr matmulOpDesc;
    FUSION_PASS_MAKE_SHARED((matmulOpDesc = std::make_shared<ge::OpDesc>(node_name, "MatMulV2")), return INTERNAL_ERROR);
    matmulOpDesc->AddInputDesc("x1", x1_desc);
    matmulOpDesc->AddInputDesc("x2", x2_desc);
    AttrUtils::SetBool(matmulOpDesc, "transpose_x1", transpose_x1);
    AttrUtils::SetBool(matmulOpDesc, "transpose_x2", transpose_x2);
    matmulOpDesc->AddOutputDesc("y", y_desc);
    new_node = graph.AddNode(matmulOpDesc);
    GraphUtils::AddEdge(in_anchor1, new_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(in_anchor2, new_node->GetInDataAnchor(1));
    OP_LOGI("MultiHeadAttentionGrad Define %s end", node_name.c_str());
    return SUCCESS;
}

template<typename _InAnchor>
static Status AddReduceSumNode(ge::ComputeGraph& graph, const ge::GeTensorDesc& x_desc,
    const ge::GeTensorDesc& y_desc, ge::NodePtr& new_node, bool keep_dims,
    const string& node_name, _InAnchor in_anchor)
{
    OP_LOGI("MultiHeadAttentionGrad Define %s begin", node_name.c_str());
    OpDescPtr reducesumOpDesc;
    FUSION_PASS_MAKE_SHARED((reducesumOpDesc = std::make_shared<ge::OpDesc>(node_name, "ReduceSumD")), return INTERNAL_ERROR);
    reducesumOpDesc->AddInputDesc("x", x_desc);
    AttrUtils::SetListInt(reducesumOpDesc, "axes", {0});
    AttrUtils::SetBool(reducesumOpDesc, "keep_dims", keep_dims);
    reducesumOpDesc->AddOutputDesc("y", y_desc);
    new_node = graph.AddNode(reducesumOpDesc);
    GraphUtils::AddEdge(in_anchor, new_node->GetInDataAnchor(0));
    OP_LOGI("MultiHeadAttentionGrad Define %s end", node_name.c_str());
    return SUCCESS;
}

template<typename _InAnchor>
static Status AddTransposeNode(ge::ComputeGraph& graph, const ge::GeTensorDesc& x_desc, const ge::GeTensorDesc& y_desc, 
    ge::NodePtr& new_node, const vector<int64_t>& perm, const vector<int64_t>& shape, 
    bool transpose_first, const string& node_name, _InAnchor in_anchor)
{
    OP_LOGI("MultiHeadAttentionGrad Define %s begin", node_name.c_str());
    OpDescPtr transOpDesc;
    FUSION_PASS_MAKE_SHARED((transOpDesc = std::make_shared<ge::OpDesc>(node_name, "ConfusionTransposeD")), return INTERNAL_ERROR);
    transOpDesc->AddInputDesc("x", x_desc);
    AttrUtils::SetListInt(transOpDesc, "perm", perm);
    AttrUtils::SetListInt(transOpDesc, "shape", shape);
    AttrUtils::SetBool(transOpDesc, "transpose_first", transpose_first);
    transOpDesc->AddOutputDesc("y", y_desc);
    new_node = graph.AddNode(transOpDesc);
    GraphUtils::AddEdge(in_anchor, new_node->GetInDataAnchor(0));
    OP_LOGI("MultiHeadAttentionGrad Define %s end", node_name.c_str());
    return SUCCESS;
}

template<typename _InAnchor1, typename _InAnchor2>
static Status AddBatchMatmulNode(ge::ComputeGraph& graph, ge::OpDescPtr& opDesc, const ge::GeTensorDesc& x1_desc, const ge::GeTensorDesc& x2_desc,
    const ge::GeTensorDesc& y_desc, ge::NodePtr& new_node, bool adj_x1, bool adj_x2, 
    const string& node_name, _InAnchor1 in_anchor1, _InAnchor2 in_anchor2)
{
    OP_LOGI("MultiHeadAttentionGrad Define %s begin", node_name.c_str());
    FUSION_PASS_MAKE_SHARED((opDesc = std::make_shared<ge::OpDesc>(node_name, "BatchMatMul")), return INTERNAL_ERROR);
    opDesc->AddInputDesc("x1", x1_desc);
    opDesc->AddInputDesc("x2", x2_desc);
    AttrUtils::SetBool(opDesc, "adj_x1", adj_x1);
    AttrUtils::SetBool(opDesc, "adj_x2", adj_x2);
    opDesc->AddOutputDesc("y", y_desc);
    new_node = graph.AddNode(opDesc);
    GraphUtils::AddEdge(in_anchor1, new_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(in_anchor2, new_node->GetInDataAnchor(1));
    OP_LOGI("MultiHeadAttentionGrad Define %s end", node_name.c_str());
    return SUCCESS;
}

static Status AddConstNode(ge::ComputeGraph& graph, const ge::GeTensorDesc& y_desc, ge::NodePtr& new_node, 
    uint8_t* data_ptr, size_t size, const string& node_name)
{
    OP_LOGI("MultiHeadAttentionGrad Define %s begin", node_name.c_str());
    OpDescPtr constOpDesc;
    FUSION_PASS_MAKE_SHARED((constOpDesc = std::make_shared<ge::OpDesc>(node_name, "Const")), return INTERNAL_ERROR);
    GeTensorPtr constValue = std::make_shared<ge::GeTensor>(y_desc, data_ptr, size);
    AttrUtils::SetTensor(constOpDesc, ATTR_NAME_WEIGHTS, constValue);
    constOpDesc->AddOutputDesc("y", y_desc);
    new_node = graph.AddNode(constOpDesc);
    OP_LOGI("MultiHeadAttentionGrad Define %s end", node_name.c_str());
    return SUCCESS;
}

template<typename _InAnchor>
static Status AddCastNode(ge::ComputeGraph& graph, ge::OpDescPtr& opDesc, const ge::GeTensorDesc& x_desc, 
    const ge::GeTensorDesc& y_desc, ge::NodePtr& new_node, int32_t dst_type,
    const string& node_name, _InAnchor in_anchor)
{
    OP_LOGI("MultiHeadAttentionGrad Define %s begin", node_name.c_str());
    FUSION_PASS_MAKE_SHARED((opDesc = std::make_shared<ge::OpDesc>(node_name, "Cast")), return INTERNAL_ERROR);
    opDesc->AddInputDesc("x", x_desc);
    AttrUtils::SetInt(opDesc, "dst_type", dst_type);
    opDesc->AddOutputDesc("y", y_desc);
    new_node = graph.AddNode(opDesc);
    GraphUtils::AddEdge(in_anchor, new_node->GetInDataAnchor(0));
    OP_LOGI("MultiHeadAttentionGrad Define %s end", node_name.c_str());
    return SUCCESS;
}

template<typename _InAnchor1, typename _InAnchor2>
static Status AddSoftmaxGradNode(ge::ComputeGraph& graph, ge::OpDescPtr& opDesc, const ge::GeTensorDesc& softmax_desc, 
    const ge::GeTensorDesc& grad_softmax_desc, const ge::GeTensorDesc& y_desc, ge::NodePtr& new_node, vector<int64_t> axes,
    const string& node_name, _InAnchor1 in_anchor1, _InAnchor2 in_anchor2)
{
    OP_LOGI("MultiHeadAttentionGrad Define %s begin", node_name.c_str());
    FUSION_PASS_MAKE_SHARED((opDesc = std::make_shared<ge::OpDesc>(node_name, "SoftmaxGrad")), return INTERNAL_ERROR);
    opDesc->AddInputDesc("softmax", softmax_desc);
    opDesc->AddInputDesc("grad_softmax", grad_softmax_desc);
    AttrUtils::SetListInt(opDesc, "axes", axes);
    opDesc->AddOutputDesc("y", y_desc);
    new_node = graph.AddNode(opDesc);
    GraphUtils::AddEdge(in_anchor1, new_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(in_anchor2, new_node->GetInDataAnchor(1));
    OP_LOGI("MultiHeadAttentionGrad Define %s end", node_name.c_str());
    return SUCCESS;
}

Status MultiHeadAttentionGradFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes)
{
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Define MultiHeadAttentionGradFusionPass fusion begin");
    ge::NodePtr multiHeadAttentionGradNode = GetNodeFromMapping(FUSED_OP_TYPE, mapping);
    FUSION_PASS_CHECK(multiHeadAttentionGradNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "MultiHeadAttentionGrad node is null, fusion failed."),
                        return PARAM_INVALID);
    ge::OpDescPtr multiHeadAttentionGradDesc = multiHeadAttentionGradNode->GetOpDesc();
    FUSION_PASS_CHECK(multiHeadAttentionGradDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "MultiHeadAttentionGrad's Op_desc is null, fusion failed."),
                        return PARAM_INVALID);
    // shape
    vector<int64_t> query_shape = multiHeadAttentionGradDesc->GetInputDesc("query").GetShape().GetDims();
    FUSION_PASS_CHECK(query_shape.size() !=2, OP_LOGE(FUSED_OP_TYPE.c_str(), "MultiHeadAttentionGrad's Query origin shape should be 2D, fusion failed."),
                        return PARAM_INVALID);
    int64_t attn_head_num, attn_dim_per_head, src_len, tgt_len;
    float keep_prob;
    bool softmax_use_float;
    vector<bool> bias_grad_mask;
    AttrUtils::GetInt(multiHeadAttentionGradDesc, "attn_head_num", attn_head_num);
    AttrUtils::GetInt(multiHeadAttentionGradDesc, "attn_dim_per_head", attn_dim_per_head);
    AttrUtils::GetInt(multiHeadAttentionGradDesc, "src_len", src_len);
    AttrUtils::GetInt(multiHeadAttentionGradDesc, "tgt_len", tgt_len);
    AttrUtils::GetFloat(multiHeadAttentionGradDesc, "keep_prob", keep_prob);
    AttrUtils::GetBool(multiHeadAttentionGradDesc, "softmax_use_float", softmax_use_float);
    AttrUtils::GetListBool(multiHeadAttentionGradDesc, "bias_grad_mask", bias_grad_mask);
    FUSION_PASS_CHECK((attn_head_num == 0 || attn_dim_per_head ==0 || src_len == 0 || tgt_len==0), 
        OP_LOGE(FUSED_OP_TYPE.c_str(), "MultiHeadAttention's attn_head_num, attn_dim_per_head, src_len, tgt_len should not be 0, fusion failed."),
                    return PARAM_INVALID);
    FUSION_PASS_CHECK(!(attn_head_num % 16 == 0 && attn_dim_per_head % 16 ==  0 && src_len % 16 ==  0 && tgt_len % 16 == 0), 
        OP_LOGE(FUSED_OP_TYPE.c_str(), "MultiHeadAttention's attn_head_num, attn_dim_per_head, src_len, tgt_len should align of 16, fusion failed."),
                    return PARAM_INVALID);
    const int64_t batch = query_shape[0] / tgt_len;
    const int64_t weight_col = attn_head_num * attn_dim_per_head;
    const float scale = 1.0 / sqrt(attn_dim_per_head);

    const vector<int64_t> perm({0,2,1,3});
    const vector<int64_t> out_proj_input_matmul_shape({batch * tgt_len, weight_col});
    const vector<int64_t> out_proj_weight_matmul_shape({weight_col, weight_col});
    const vector<int64_t> bias_reducesum_shape({1, weight_col});
    const vector<int64_t> context_trans_shape({batch, tgt_len, attn_head_num, attn_dim_per_head});
    const vector<int64_t> new_context_trans_shape({batch, attn_head_num, tgt_len, attn_dim_per_head});
    const vector<int64_t> attn_res_batch_shape({batch, attn_head_num, tgt_len, src_len});
    const vector<int64_t> kv_res_batch_shape({batch, attn_head_num, src_len, attn_dim_per_head});
    const vector<int64_t> query_res_batch_shape({batch, attn_head_num, tgt_len, attn_dim_per_head});
    const vector<int64_t> kv_trans_batch_shape({batch * src_len, weight_col});
    const vector<int64_t> query_trans_batch_shape({batch * tgt_len, weight_col});
    const vector<int64_t> query_matmul_shape({batch * tgt_len, weight_col});
    const vector<int64_t> query_weight_matmul_shape({weight_col, weight_col});
    const vector<int64_t> kv_matmul_shape({batch * src_len, weight_col});
    const vector<int64_t> kv_weight_matmul_shape({weight_col, weight_col});

    // out_proj_input_matmul
    GeTensorDesc outProjInputMatmulOutputDesc;
    SetNZTensorDesc(outProjInputMatmulOutputDesc, out_proj_input_matmul_shape);
    NodePtr outProjInputMatmulNode;
    AddMatmulNode(graph, multiHeadAttentionGradDesc->GetInputDesc("y_grad"), multiHeadAttentionGradDesc->GetInputDesc("out_proj_weight"),
        outProjInputMatmulOutputDesc, outProjInputMatmulNode, false, false, "out_proj_input_matmul",
        multiHeadAttentionGradNode->GetInDataAnchor(13)->GetPeerOutAnchor(),
        multiHeadAttentionGradNode->GetInDataAnchor(6)->GetPeerOutAnchor()
    );

    // out_proj_weight_matmul
    GeTensorDesc outProjWeightMatmulOutputDesc;
    SetNZTensorDesc(outProjWeightMatmulOutputDesc, out_proj_input_matmul_shape);
    NodePtr outProjWeightMatmulNode;
    AddMatmulNode(graph, multiHeadAttentionGradDesc->GetInputDesc("y_grad"), multiHeadAttentionGradDesc->GetInputDesc("context"),
        outProjWeightMatmulOutputDesc, outProjWeightMatmulNode, true, false, "out_proj_weight_matmul",
        multiHeadAttentionGradNode->GetInDataAnchor(13)->GetPeerOutAnchor(),
        multiHeadAttentionGradNode->GetInDataAnchor(12)->GetPeerOutAnchor()
    );
    AddNodeLinkOut(outProjWeightMatmulNode->GetOutDataAnchor(0), 
        multiHeadAttentionGradNode->GetOutDataAnchor(3), "out_proj_weight_matmul");

    // bias_empty
    GeTensorDesc biasEmptyTensorDesc = GeTensorDesc(GeShape(), FORMAT_ND, DT_FLOAT16);
    NodePtr biasEmptyNode;
    AddConstNode(graph, biasEmptyTensorDesc, biasEmptyNode, nullptr, 0, "bias_empty");

    // out_proj_bias
    NodePtr outProjBiasNode;
    GeTensorDesc biasReducesumTensorDesc;
    SetNDTensorDesc(biasReducesumTensorDesc, bias_reducesum_shape, DT_FLOAT16);
    if (bias_grad_mask[3]) {
        AddReduceSumNode(graph, multiHeadAttentionGradDesc->GetInputDesc("y_grad"), biasReducesumTensorDesc, outProjBiasNode,
            true, "out_proj_bias", multiHeadAttentionGradNode->GetInDataAnchor(13)->GetPeerOutAnchor());
    } else {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Define out_proj_bias empty begin");
        outProjBiasNode = biasEmptyNode;
    }
    AddNodeLinkOut(outProjBiasNode->GetOutDataAnchor(0), 
        multiHeadAttentionGradNode->GetOutDataAnchor(10), "out_proj_bias");

    // context_trans
    GeTensorDesc contextTransOutputDesc;
    SetNZTensorDesc(contextTransOutputDesc, new_context_trans_shape);
    NodePtr contextTransNode;
    AddTransposeNode(graph, outProjInputMatmulOutputDesc, contextTransOutputDesc, contextTransNode, perm, context_trans_shape,
        false, "context_trans", outProjInputMatmulNode->GetOutDataAnchor(0)
    );

    // attn_res_batch
    GeTensorDesc attnResBatchOutputDesc;
    SetNZTensorDesc(attnResBatchOutputDesc, attn_res_batch_shape);
    OpDescPtr attnResBatchOpDesc;
    NodePtr attnResBatchNode;
    AddBatchMatmulNode(graph, attnResBatchOpDesc, contextTransOutputDesc, multiHeadAttentionGradDesc->GetInputDesc("value_res"), attnResBatchOutputDesc,
        attnResBatchNode, false, true, "attn_res_batch", contextTransNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetInDataAnchor(9)->GetPeerOutAnchor()
    );

    // value_res_batch
    GeTensorDesc valueResBatchOutputDesc;
    SetNZTensorDesc(valueResBatchOutputDesc, kv_res_batch_shape);
    OpDescPtr valueResBatchOpDesc;
    NodePtr valueResBatchNode;
    AddBatchMatmulNode(graph, valueResBatchOpDesc, multiHeadAttentionGradDesc->GetInputDesc("attn_res"), contextTransOutputDesc, valueResBatchOutputDesc,
        valueResBatchNode, true, false, "value_res_batch", multiHeadAttentionGradNode->GetInDataAnchor(11)->GetPeerOutAnchor(),
        contextTransNode->GetOutDataAnchor(0)
    );

    // attn_res_dropout
    OpDescPtr attnResDropoutOpDesc;
    NodePtr attnResDropoutNode;
    if (keep_prob < 1.0) {
        GeTensorDesc probTensorDesc = GeTensorDesc(GeShape(), FORMAT_ND, DT_FLOAT);
        NodePtr probNode;
        AddConstNode(graph, probTensorDesc, probNode, reinterpret_cast<uint8_t*>(&keep_prob), sizeof(float), "keep_prob");
        // dropout_do_mask
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Define attn_res_dropout begin");
        FUSION_PASS_MAKE_SHARED((attnResDropoutOpDesc = std::make_shared<ge::OpDesc>("dropout_do_mask", "DropOutDoMask")), return INTERNAL_ERROR);
        attnResDropoutOpDesc->AddInputDesc("x", attnResBatchOutputDesc);
        attnResDropoutOpDesc->AddInputDesc("mask", multiHeadAttentionGradDesc->GetInputDesc("dropout_mask"));
        attnResDropoutOpDesc->AddInputDesc("keep_prob", probTensorDesc);
        attnResDropoutOpDesc->AddOutputDesc("y", attnResBatchOutputDesc);
        attnResDropoutNode = graph.AddNode(attnResDropoutOpDesc);
        GraphUtils::AddEdge(attnResBatchNode->GetOutDataAnchor(0), attnResDropoutNode->GetInDataAnchor(0));
        GraphUtils::AddEdge(multiHeadAttentionGradNode->GetInDataAnchor(14)->GetPeerOutAnchor(), attnResDropoutNode->GetInDataAnchor(1));
        GraphUtils::AddEdge(probNode->GetOutDataAnchor(0), attnResDropoutNode->GetInDataAnchor(2));
    } else {
        attnResDropoutOpDesc = attnResBatchOpDesc;
        attnResDropoutNode = attnResBatchNode;
    }
    // attn_weight_softmax
    OpDescPtr softmaxGradOpDesc;
    NodePtr softmaxGradNode;
    GeTensorDesc softmaxGradOutputDesc;
    SetNZTensorDesc(softmaxGradOutputDesc, attn_res_batch_shape);
    if (softmax_use_float) {
        OpDescPtr castOpDesc, beforeCastOpDesc;
        // cast_before_softmax
        GeTensorDesc castOutputDesc;
        SetNZTensorDesc(castOutputDesc, attn_res_batch_shape, DT_FLOAT);
        NodePtr castNode;
        AddCastNode(graph, castOpDesc, attnResBatchOutputDesc, castOutputDesc, castNode, DT_FLOAT, "cast_before_softmax",
            attnResDropoutNode->GetOutDataAnchor(0));
        // attn_weight_softmax
        NodePtr beforeCastNode;
        AddSoftmaxGradNode(graph, beforeCastOpDesc, multiHeadAttentionGradDesc->GetInputDesc("attn_scores"), castOutputDesc,
            castOutputDesc, beforeCastNode, {-1}, "attn_weight_softmax",
            multiHeadAttentionGradNode->GetInDataAnchor(10)->GetPeerOutAnchor(), castNode->GetOutDataAnchor(0)
        );
        // cast_after_softmax
        AddCastNode(graph, softmaxGradOpDesc, castOutputDesc, softmaxGradOutputDesc, softmaxGradNode, DT_FLOAT16, "cast_after_softmax",
            beforeCastNode->GetOutDataAnchor(0));
    } else {
        // softmax
        AddSoftmaxGradNode(graph, softmaxGradOpDesc, multiHeadAttentionGradDesc->GetInputDesc("attn_scores"), attnResBatchOutputDesc,
            softmaxGradOutputDesc, softmaxGradNode, {-1}, "attn_weight_softmax",
            multiHeadAttentionGradNode->GetInDataAnchor(10)->GetPeerOutAnchor(), attnResDropoutNode->GetOutDataAnchor(0)
        );
    }

    // query_res_batch
    GeTensorDesc queryResBatchOutputDesc;
    SetNZTensorDesc(queryResBatchOutputDesc, query_res_batch_shape);
    OpDescPtr queryResBatchOpDesc;
    NodePtr queryResBatchNode;
    AddBatchMatmulNode(graph, queryResBatchOpDesc, softmaxGradOutputDesc, multiHeadAttentionGradDesc->GetInputDesc("key_res"), queryResBatchOutputDesc,
        queryResBatchNode, false, false, "query_res_batch", softmaxGradNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetInDataAnchor(8)->GetPeerOutAnchor()
    );

    // key_res_batch
    GeTensorDesc keyResBatchOutputDesc;
    SetNZTensorDesc(keyResBatchOutputDesc, kv_res_batch_shape);
    OpDescPtr keyResBatchOpDesc;
    NodePtr keyResBatchNode;
    AddBatchMatmulNode(graph, keyResBatchOpDesc, softmaxGradOutputDesc, multiHeadAttentionGradDesc->GetInputDesc("query_res"), keyResBatchOutputDesc,
        keyResBatchNode, true, false, "key_res_batch", softmaxGradNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetInDataAnchor(7)->GetPeerOutAnchor()
    );

    // query_trans
    GeTensorDesc queryTransOutputDesc;
    SetNZTensorDesc(queryTransOutputDesc, query_trans_batch_shape);
    NodePtr queryTransNode;
    AddTransposeNode(graph, queryResBatchOutputDesc, queryTransOutputDesc, queryTransNode, perm, query_trans_batch_shape,
        true, "query_trans", queryResBatchNode->GetOutDataAnchor(0)
    );

    // attn_scores_muls
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Define attn_scores_muls begin");
    OpDescPtr attnScoresMulsOpDesc;
    FUSION_PASS_MAKE_SHARED((attnScoresMulsOpDesc = std::make_shared<ge::OpDesc>("attn_scores_muls", "Muls")), return INTERNAL_ERROR);
    attnScoresMulsOpDesc->AddInputDesc("x", queryTransOutputDesc);
    AttrUtils::SetFloat(attnScoresMulsOpDesc, "value", scale);
    attnScoresMulsOpDesc->AddOutputDesc("y", queryTransOutputDesc);
    NodePtr attnScoresMulsNode = graph.AddNode(attnScoresMulsOpDesc);
    GraphUtils::AddEdge(queryTransNode->GetOutDataAnchor(0), attnScoresMulsNode->GetInDataAnchor(0));

    // key_trans
    GeTensorDesc keyTransOutputDesc;
    SetNZTensorDesc(keyTransOutputDesc, kv_trans_batch_shape);
    NodePtr keyTransNode;
    AddTransposeNode(graph, keyResBatchOutputDesc, keyTransOutputDesc, keyTransNode, perm, kv_trans_batch_shape,
        true, "key_trans", keyResBatchNode->GetOutDataAnchor(0)
    );

    // value_trans
    GeTensorDesc valueTransOutputDesc;
    SetNZTensorDesc(valueTransOutputDesc, kv_trans_batch_shape);
    NodePtr valueTransNode;
    AddTransposeNode(graph, valueResBatchOutputDesc, valueTransOutputDesc, valueTransNode, perm, kv_trans_batch_shape,
        true, "value_trans", valueResBatchNode->GetOutDataAnchor(0)
    );

    // query_matmul
    GeTensorDesc queryMatmulOutputDesc;
    SetNZTensorDesc(queryMatmulOutputDesc, query_matmul_shape);
    NodePtr queryMatmulNode;
    AddMatmulNode(graph, queryTransOutputDesc, multiHeadAttentionGradDesc->GetInputDesc("query_weight"),
        queryMatmulOutputDesc, queryMatmulNode, false, false, "query_matmul",
        attnScoresMulsNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetInDataAnchor(3)->GetPeerOutAnchor()
    );
    AddNodeLinkOut(queryMatmulNode->GetOutDataAnchor(0), 
        multiHeadAttentionGradNode->GetOutDataAnchor(4), "query_matmul");

    // query_weight_matmul
    GeTensorDesc queryWeightMatmulOutputDesc;
    SetNZTensorDesc(queryWeightMatmulOutputDesc, query_weight_matmul_shape);
    NodePtr queryWeightMatmulNode;
    AddMatmulNode(graph, queryTransOutputDesc, multiHeadAttentionGradDesc->GetInputDesc("query"),
        queryWeightMatmulOutputDesc, queryWeightMatmulNode, true, false, "query_weight_matmul",
        attnScoresMulsNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetInDataAnchor(0)->GetPeerOutAnchor()
    );
    AddNodeLinkOut(queryWeightMatmulNode->GetOutDataAnchor(0), 
        multiHeadAttentionGradNode->GetOutDataAnchor(0), "query_weight_matmul");

    // query_bias
    NodePtr queryBiasNode;
    if (bias_grad_mask[0]) {
        AddReduceSumNode(graph, queryTransOutputDesc, biasReducesumTensorDesc, queryBiasNode,
            true, "query_bias", attnScoresMulsNode->GetOutDataAnchor(0));
    } else {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Define query_bias empty begin");
        queryBiasNode = biasEmptyNode;
    }
    AddNodeLinkOut(queryBiasNode->GetOutDataAnchor(0), 
        multiHeadAttentionGradNode->GetOutDataAnchor(7), "query_bias");

    // key_matmul
    GeTensorDesc keyMatmulOutputDesc;
    SetNZTensorDesc(keyMatmulOutputDesc, kv_matmul_shape);
    NodePtr keyMatmulNode;
    AddMatmulNode(graph, keyTransOutputDesc, multiHeadAttentionGradDesc->GetInputDesc("key_weight"),
        keyMatmulOutputDesc, keyMatmulNode, false, false, "key_matmul",
        keyTransNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetInDataAnchor(4)->GetPeerOutAnchor()
    );
    AddNodeLinkOut(keyMatmulNode->GetOutDataAnchor(0), 
        multiHeadAttentionGradNode->GetOutDataAnchor(5), "key_matmul");

    // key_weight_matmul
    GeTensorDesc keyWeightMatmulOutputDesc;
    SetNZTensorDesc(keyWeightMatmulOutputDesc, kv_weight_matmul_shape);
    NodePtr keyWeightMatmulNode;
    AddMatmulNode(graph, keyTransOutputDesc, multiHeadAttentionGradDesc->GetInputDesc("key"),
        keyWeightMatmulOutputDesc, keyWeightMatmulNode, true, false, "key_weight_matmul",
        keyTransNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetInDataAnchor(1)->GetPeerOutAnchor()
    );
    AddNodeLinkOut(keyWeightMatmulNode->GetOutDataAnchor(0), 
        multiHeadAttentionGradNode->GetOutDataAnchor(1), "key_weight_matmul");

    // key_bias
    NodePtr keyBiasNode;
    if (bias_grad_mask[1]) {
        AddReduceSumNode(graph, keyTransOutputDesc, biasReducesumTensorDesc, keyBiasNode,
            true, "key_bias", keyTransNode->GetOutDataAnchor(0));
    } else {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Define key_bias empty begin");
        keyBiasNode = biasEmptyNode;
    }
    AddNodeLinkOut(keyBiasNode->GetOutDataAnchor(0), 
        multiHeadAttentionGradNode->GetOutDataAnchor(8), "key_bias");

    // value_matmul
    GeTensorDesc valueMatmulOutputDesc;
    SetNZTensorDesc(valueMatmulOutputDesc, kv_matmul_shape);
    NodePtr valueMatmulNode;
    AddMatmulNode(graph, valueTransOutputDesc, multiHeadAttentionGradDesc->GetInputDesc("value_weight"),
        valueMatmulOutputDesc, valueMatmulNode, false, false, "value_matmul",
        valueTransNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetInDataAnchor(5)->GetPeerOutAnchor()
    );
    AddNodeLinkOut(valueMatmulNode->GetOutDataAnchor(0), 
        multiHeadAttentionGradNode->GetOutDataAnchor(6), "value_matmul");

    // value_weight_matmul
    GeTensorDesc valueWeightMatmulOutputDesc;
    SetNZTensorDesc(valueWeightMatmulOutputDesc, kv_weight_matmul_shape);
    NodePtr valueWeightMatmulNode;
    AddMatmulNode(graph, valueTransOutputDesc, multiHeadAttentionGradDesc->GetInputDesc("value"),
        valueWeightMatmulOutputDesc, valueWeightMatmulNode, true, false, "value_weight_matmul",
        valueTransNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetInDataAnchor(2)->GetPeerOutAnchor()
    );
    AddNodeLinkOut(valueWeightMatmulNode->GetOutDataAnchor(0), 
        multiHeadAttentionGradNode->GetOutDataAnchor(2), "value_weight_matmul");

    // value_bias
    NodePtr valueBiasNode;
    if (bias_grad_mask[2]) {
        AddReduceSumNode(graph, valueTransOutputDesc, biasReducesumTensorDesc, valueBiasNode,
            true, "value_bias", valueTransNode->GetOutDataAnchor(0));
    } else {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Define value_bias empty begin");
        valueBiasNode = biasEmptyNode;
    }
    AddNodeLinkOut(valueBiasNode->GetOutDataAnchor(0), 
        multiHeadAttentionGradNode->GetOutDataAnchor(9), "value_bias");

    // unlink all control input
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Define unlink control input begin");
    if (multiHeadAttentionGradNode->GetInControlAnchor() != nullptr) {
        multiHeadAttentionGradNode->GetInControlAnchor()->UnlinkAll();
    }

    // unlink all input
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Define unlinks input begin");
    for (auto inAnchor : multiHeadAttentionGradNode->GetAllInDataAnchors()) {
        if (inAnchor != nullptr) {
            inAnchor->UnlinkAll();
        }
    }
    // remove all output
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Define unlinks output begin");
    for (auto outAnchor : multiHeadAttentionGradNode->GetAllInDataAnchors()) {
        if (outAnchor != nullptr) {
            outAnchor->UnlinkAll();
        }
    }
    if (bias_grad_mask[0]&&bias_grad_mask[1]&&bias_grad_mask[2]&&bias_grad_mask[3]) {
        graph.RemoveNode(biasEmptyNode);
    }
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(multiHeadAttentionGradNode),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove multiHeadAttentionGradNode failed"), return FAILED);


    return SUCCESS;
}

REGISTER_PASS("MultiHeadAttentionGradFusionPass", BUILT_IN_GRAPH_PASS, MultiHeadAttentionGradFusionPass);
} // namespace 