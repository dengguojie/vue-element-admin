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
 * \file multi_head_attention_fusion_pass.cc
 *
 */

#include "multi_head_attention_fusion_pass.h"
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

using namespace std;
using namespace ge;

namespace fe {
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

vector<FusionPattern*> MultiHeadAttentionFusionPass::DefinePatterns()
{
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("MultiHeadAttentionFusionPass");
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter MultiHeadAttentionFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(FUSED_OP_TYPE, {FUSED_OP_TYPE}).SetOutput(FUSED_OP_TYPE);
  patterns.push_back(pattern);
  return patterns;
}

template<typename _InAnchor, typename _OutAnchor>
static Status AddNodeLinkOut(_InAnchor in_anchor, _OutAnchor out_anchor, const string& out_node_name) {
    // link out
    OP_LOGI("MultiHeadAttention Define %s link out begin", out_node_name.c_str());
    for (auto anchor : out_anchor->GetPeerInDataAnchors()) {
        GraphUtils::RemoveEdge(out_anchor, anchor);
        GraphUtils::AddEdge(in_anchor, anchor);
    }
    OP_LOGI("MultiHeadAttention Define %s link out end", out_node_name.c_str());
    return SUCCESS;
}

template<typename _InAnchor1, typename _InAnchor2, typename _InAnchor3>
static Status AddMatmulNode(ge::ComputeGraph& graph, const ge::GeTensorDesc& x1_desc, const ge::GeTensorDesc& x2_desc,
    const ge::GeTensorDescPtr& bias_desc, const ge::GeTensorDesc& y_desc, ge::NodePtr& new_node, bool transpose_x1,
    bool transpose_x2, const string& node_name, _InAnchor1 in_anchor1, _InAnchor2 in_anchor2, _InAnchor3 in_anchor3)
{
    OP_LOGI("MultiHeadAttention Define %s begin", node_name.c_str());
    OpDescPtr matmulOpDesc;
    FUSION_PASS_MAKE_SHARED((matmulOpDesc = std::make_shared<ge::OpDesc>(node_name, "MatMulV2")), return INTERNAL_ERROR);
    matmulOpDesc->AddInputDesc("x1", x1_desc);
    matmulOpDesc->AddInputDesc("x2", x2_desc);
    if (bias_desc) {
        matmulOpDesc->AddInputDesc("bias", *bias_desc);
    }
    AttrUtils::SetBool(matmulOpDesc, "transpose_x1", transpose_x1);
    AttrUtils::SetBool(matmulOpDesc, "transpose_x2", transpose_x2);
    matmulOpDesc->AddOutputDesc("y", y_desc);
    new_node = graph.AddNode(matmulOpDesc);
    GraphUtils::AddEdge(in_anchor1, new_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(in_anchor2, new_node->GetInDataAnchor(1));
    if (bias_desc) {
        GraphUtils::AddEdge(in_anchor3, new_node->GetInDataAnchor(2));
    }
    OP_LOGI("MultiHeadAttention Define %s end", node_name.c_str());
    return SUCCESS;
}

template<typename _InAnchor, typename _OutAnchor>
static Status AddTransposeNode(ge::ComputeGraph& graph, const ge::GeTensorDesc& x_desc, const ge::GeTensorDesc& y_desc, 
    ge::NodePtr& new_node, const vector<int64_t>& perm, const vector<int64_t>& shape, 
    bool transpose_first, const string& node_name, _InAnchor in_anchor, _OutAnchor out_anchor)
{
    OP_LOGI("MultiHeadAttention Define %s begin", node_name.c_str());
    OpDescPtr transOpDesc;
    FUSION_PASS_MAKE_SHARED((transOpDesc = std::make_shared<ge::OpDesc>(node_name, "ConfusionTransposeD")), return INTERNAL_ERROR);
    transOpDesc->AddInputDesc("x", x_desc);
    AttrUtils::SetListInt(transOpDesc, "perm", perm);
    AttrUtils::SetListInt(transOpDesc, "shape", shape);
    AttrUtils::SetBool(transOpDesc, "transpose_first", transpose_first);
    transOpDesc->AddOutputDesc("y", y_desc);
    new_node = graph.AddNode(transOpDesc);
    GraphUtils::AddEdge(in_anchor, new_node->GetInDataAnchor(0));
    AddNodeLinkOut(new_node->GetOutDataAnchor(0), out_anchor, node_name);
    OP_LOGI("MultiHeadAttention Define %s end", node_name.c_str());
    return SUCCESS;
}

template<typename _InAnchor1, typename _InAnchor2>
static Status AddBatchMatmulNode(ge::ComputeGraph& graph, const ge::GeTensorDesc& x1_desc, const ge::GeTensorDesc& x2_desc,
    const ge::GeTensorDesc& y_desc, ge::NodePtr& new_node, bool adj_x1, bool adj_x2, 
    const string& node_name, _InAnchor1 in_anchor1, _InAnchor2 in_anchor2)
{
    OP_LOGI("MultiHeadAttention Define %s begin", node_name.c_str());
    OpDescPtr batchOpDesc;
    FUSION_PASS_MAKE_SHARED((batchOpDesc = std::make_shared<ge::OpDesc>(node_name, "BatchMatMul")), return INTERNAL_ERROR);
    batchOpDesc->AddInputDesc("x1", x1_desc);
    batchOpDesc->AddInputDesc("x2", x2_desc);
    AttrUtils::SetBool(batchOpDesc, "adj_x1", adj_x1);
    AttrUtils::SetBool(batchOpDesc, "adj_x2", adj_x2);
    batchOpDesc->AddOutputDesc("y", y_desc);
    new_node = graph.AddNode(batchOpDesc);
    GraphUtils::AddEdge(in_anchor1, new_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(in_anchor2, new_node->GetInDataAnchor(1));
    OP_LOGI("MultiHeadAttention Define %s end", node_name.c_str());
    return SUCCESS;
}

static Status AddConstNode(ge::ComputeGraph& graph, const ge::GeTensorDesc& y_desc, ge::NodePtr& new_node, 
    uint8_t* data_ptr, size_t size, const string& node_name)
{
    OP_LOGI("MultiHeadAttention Define %s begin", node_name.c_str());
    OpDescPtr constOpDesc;
    FUSION_PASS_MAKE_SHARED((constOpDesc = std::make_shared<ge::OpDesc>(node_name, "Const")), return INTERNAL_ERROR);
    GeTensorPtr constValue = std::make_shared<ge::GeTensor>(y_desc, data_ptr, size);
    AttrUtils::SetTensor(constOpDesc, ATTR_NAME_WEIGHTS, constValue);
    constOpDesc->AddOutputDesc("y", y_desc);
    new_node = graph.AddNode(constOpDesc);
    OP_LOGI("MultiHeadAttention Define %s end", node_name.c_str());
    return SUCCESS;
}

template<typename _InAnchor1, typename _InAnchor2, typename _InAnchor3>
static Status AddDropOutDoMaskNode(ge::ComputeGraph& graph, ge::OpDescPtr& opDesc, const ge::GeTensorDesc& x_desc, 
    const ge::GeTensorDesc& mask_desc, const ge::GeTensorDesc& prob_desc, const ge::GeTensorDesc& y_desc, ge::NodePtr& new_node, 
    const string& node_name, _InAnchor1 in_anchor1, _InAnchor2 in_anchor2, _InAnchor3 in_anchor3)
{
    OP_LOGI("MultiHeadAttention Define %s begin", node_name.c_str());
    FUSION_PASS_MAKE_SHARED((opDesc = std::make_shared<ge::OpDesc>(node_name, "DropOutDoMask")), return INTERNAL_ERROR);
    opDesc->AddInputDesc("x", x_desc);
    opDesc->AddInputDesc("mask", mask_desc);
    opDesc->AddInputDesc("keep_prob", prob_desc);
    opDesc->AddOutputDesc("y", y_desc);
    new_node = graph.AddNode(opDesc);
    GraphUtils::AddEdge(in_anchor1, new_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(in_anchor2, new_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(in_anchor3, new_node->GetInDataAnchor(2));
    OP_LOGI("MultiHeadAttention Define %s end", node_name.c_str());
    return SUCCESS;
}

template<typename _InAnchor>
static Status AddCastNode(ge::ComputeGraph& graph, ge::OpDescPtr& opDesc, const ge::GeTensorDesc& x_desc, 
    const ge::GeTensorDesc& y_desc, ge::NodePtr& new_node, int32_t dst_type,
    const string& node_name, _InAnchor in_anchor)
{
    OP_LOGI("MultiHeadAttention Define %s begin", node_name.c_str());
    FUSION_PASS_MAKE_SHARED((opDesc = std::make_shared<ge::OpDesc>(node_name, "Cast")), return INTERNAL_ERROR);
    opDesc->AddInputDesc("x", x_desc);
    AttrUtils::SetInt(opDesc, "dst_type", dst_type);
    opDesc->AddOutputDesc("y", y_desc);
    new_node = graph.AddNode(opDesc);
    GraphUtils::AddEdge(in_anchor, new_node->GetInDataAnchor(0));
    OP_LOGI("MultiHeadAttention Define %s end", node_name.c_str());
    return SUCCESS;
}

template<typename _InAnchor>
static Status AddSoftmaxNode(ge::ComputeGraph& graph, ge::OpDescPtr& opDesc, const ge::GeTensorDesc& x_desc, 
    const ge::GeTensorDesc& y_desc, ge::NodePtr& new_node, vector<int64_t> axes,
    const string& node_name, _InAnchor in_anchor)
{
    OP_LOGI("MultiHeadAttention Define %s begin", node_name.c_str());
    FUSION_PASS_MAKE_SHARED((opDesc = std::make_shared<ge::OpDesc>(node_name, "SoftmaxV2")), return INTERNAL_ERROR);
    opDesc->AddInputDesc("x", x_desc);
    AttrUtils::SetListInt(opDesc, "axes", axes);
    opDesc->AddOutputDesc("y", y_desc);
    new_node = graph.AddNode(opDesc);
    GraphUtils::AddEdge(in_anchor, new_node->GetInDataAnchor(0));
    OP_LOGI("MultiHeadAttention Define %s end", node_name.c_str());
    return SUCCESS;
}

Status MultiHeadAttentionFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes)
{
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Define MultiHeadAttentionFusionPass fusion begin");
    ge::NodePtr multiHeadAttentionNode = GetNodeFromMapping(FUSED_OP_TYPE, mapping);
    FUSION_PASS_CHECK(multiHeadAttentionNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "MultiHeadAttention node is null, fusion failed."),
                        return PARAM_INVALID);
    ge::OpDescPtr multiHeadAttentionDesc = multiHeadAttentionNode->GetOpDesc();
    FUSION_PASS_CHECK(multiHeadAttentionDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "MultiHeadAttention's Op_desc is null, fusion failed."),
                        return PARAM_INVALID);
    // shape
    vector<int64_t> query_shape = multiHeadAttentionDesc->GetInputDesc("query").GetOriginShape().GetDims();
    FUSION_PASS_CHECK(query_shape.size() !=2, OP_LOGE(FUSED_OP_TYPE.c_str(), "MultiHeadAttention's Query origin shape should be 2D, fusion failed."),
                        return PARAM_INVALID);
    ge::Format format = multiHeadAttentionDesc->GetInputDesc("query").GetOriginFormat();
    FUSION_PASS_CHECK(format != FORMAT_ND, OP_LOGE(FUSED_OP_TYPE.c_str(), "MultiHeadAttention's Query origin format should be nd, fusion failed."),
                        return PARAM_INVALID);
    int64_t attn_head_num, attn_dim_per_head, src_len, tgt_len;
    float keep_prob;
    bool softmax_use_float;
    AttrUtils::GetInt(multiHeadAttentionDesc, "attn_head_num", attn_head_num);
    AttrUtils::GetInt(multiHeadAttentionDesc, "attn_dim_per_head", attn_dim_per_head);
    AttrUtils::GetInt(multiHeadAttentionDesc, "src_len", src_len);
    AttrUtils::GetInt(multiHeadAttentionDesc, "tgt_len", tgt_len);
    AttrUtils::GetFloat(multiHeadAttentionDesc, "keep_prob", keep_prob);
    AttrUtils::GetBool(multiHeadAttentionDesc, "softmax_use_float", softmax_use_float);
    FUSION_PASS_CHECK((attn_head_num <= 0 || attn_dim_per_head <= 0 || src_len <= 0 || tgt_len <= 0), 
        OP_LOGE(FUSED_OP_TYPE.c_str(), "MultiHeadAttention's attn_head_num, attn_dim_per_head, src_len, tgt_len should greater than 0, fusion failed."),
                    return PARAM_INVALID);
    FUSION_PASS_CHECK(!(attn_head_num % 16 == 0 && attn_dim_per_head % 16 ==  0 && src_len % 16 ==  0 && tgt_len % 16 == 0), 
        OP_LOGE(FUSED_OP_TYPE.c_str(), "MultiHeadAttention's attn_head_num, attn_dim_per_head, src_len, tgt_len should align of 16, fusion failed."),
                    return PARAM_INVALID);
    const int64_t batch = query_shape[0] / tgt_len;
    const int64_t weight_col = attn_head_num * attn_dim_per_head;
    const int64_t gen_mask_shape = batch * attn_head_num * src_len * tgt_len / 8;
    const int64_t attn_res_shape = batch * attn_head_num * src_len * tgt_len;
    const float scale = 1.0 / sqrt(attn_dim_per_head);
    
    const vector<int64_t> perm({0,2,1,3});
    const vector<int64_t> query_matmul_shape({batch*tgt_len, weight_col}); 
    const vector<int64_t> kv_matmul_shape({batch*src_len, weight_col});
    const vector<int64_t> new_query_shape({batch, tgt_len, attn_head_num, attn_dim_per_head});
    const vector<int64_t> transpose_new_query_shape({batch, attn_head_num, tgt_len, attn_dim_per_head});
    const vector<int64_t> new_kv_shape({batch, src_len, attn_head_num, attn_dim_per_head});
    const vector<int64_t> transpose_new_kv_shape({batch, attn_head_num, src_len, attn_dim_per_head});
    const vector<int64_t> softmax_scores_shape({batch, attn_head_num, tgt_len, src_len});
    const vector<int64_t> attn_batchmatmul_shape({batch, attn_head_num, tgt_len, src_len});
    const vector<int64_t> context_batchmatmul_shape({batch, attn_head_num, tgt_len, attn_dim_per_head});
    const vector<int64_t> context_shape({batch*tgt_len, weight_col});
    // query_matmul
    GeTensorDesc queryMatmulOutputDesc;
    SetNZTensorDesc(queryMatmulOutputDesc, query_matmul_shape);
    NodePtr queryMatmulNode;
    AddMatmulNode(graph, multiHeadAttentionDesc->GetInputDesc("query"), multiHeadAttentionDesc->GetInputDesc("query_weight"),
        multiHeadAttentionDesc->MutableInputDesc("query_bias"), queryMatmulOutputDesc, queryMatmulNode, false, true, "query_matmul",
        multiHeadAttentionNode->GetInDataAnchor(0)->GetPeerOutAnchor(), multiHeadAttentionNode->GetInDataAnchor(3)->GetPeerOutAnchor(),
        multiHeadAttentionNode->GetInDataAnchor(8)->GetPeerOutAnchor()
    );
    // key_matmul
    GeTensorDesc kvMatmulOutputDesc;
    SetNZTensorDesc(kvMatmulOutputDesc, kv_matmul_shape);
    NodePtr keyMatmulNode;
    AddMatmulNode(graph, multiHeadAttentionDesc->GetInputDesc("key"), multiHeadAttentionDesc->GetInputDesc("key_weight"),
        multiHeadAttentionDesc->MutableInputDesc("key_bias"), kvMatmulOutputDesc, keyMatmulNode, false, true, "key_matmul",
        multiHeadAttentionNode->GetInDataAnchor(1)->GetPeerOutAnchor(), multiHeadAttentionNode->GetInDataAnchor(4)->GetPeerOutAnchor(),
        multiHeadAttentionNode->GetInDataAnchor(9)->GetPeerOutAnchor()
    );

    // value_matmul
    NodePtr valueMatmulNode;
    AddMatmulNode(graph, multiHeadAttentionDesc->GetInputDesc("value"), multiHeadAttentionDesc->GetInputDesc("value_weight"),
        multiHeadAttentionDesc->MutableInputDesc("value_bias"), kvMatmulOutputDesc, valueMatmulNode, false, true, "value_matmul",
        multiHeadAttentionNode->GetInDataAnchor(2)->GetPeerOutAnchor(), multiHeadAttentionNode->GetInDataAnchor(5)->GetPeerOutAnchor(),
        multiHeadAttentionNode->GetInDataAnchor(10)->GetPeerOutAnchor()
    );

    // query_muls
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Define query_muls begin");
    OpDescPtr queryMulsOpDesc;
    FUSION_PASS_MAKE_SHARED((queryMulsOpDesc = std::make_shared<ge::OpDesc>("query_muls", "Muls")), return INTERNAL_ERROR);
    queryMulsOpDesc->AddInputDesc("x", queryMatmulOutputDesc);
    AttrUtils::SetFloat(queryMulsOpDesc, "value", scale);
    queryMulsOpDesc->AddOutputDesc("y", queryMatmulOutputDesc);
    NodePtr queryMulsNode = graph.AddNode(queryMulsOpDesc);
    GraphUtils::AddEdge(queryMatmulNode->GetOutDataAnchor(0), queryMulsNode->GetInDataAnchor(0));

    
    // query_trans
    GeTensorDesc queryTransOutputDesc;
    SetNZTensorDesc(queryTransOutputDesc, transpose_new_query_shape);
    NodePtr queryTransNode;
    AddTransposeNode(graph, queryMatmulOutputDesc, queryTransOutputDesc, queryTransNode, perm, new_query_shape, false, "query_trans",
        queryMulsNode->GetOutDataAnchor(0), multiHeadAttentionNode->GetOutDataAnchor(2)
    );

    // key_trans
    GeTensorDesc keyTransOutputDesc;
    SetNZTensorDesc(keyTransOutputDesc, transpose_new_kv_shape);
    NodePtr keyTransNode;
    AddTransposeNode(graph, kvMatmulOutputDesc, keyTransOutputDesc, keyTransNode, perm, new_kv_shape, false, "key_trans",
        keyMatmulNode->GetOutDataAnchor(0), multiHeadAttentionNode->GetOutDataAnchor(3)
    );

    // value_trans
    GeTensorDesc valueTransOutputDesc;
    SetNZTensorDesc(valueTransOutputDesc, transpose_new_kv_shape);
    NodePtr valueTransNode;
    AddTransposeNode(graph, kvMatmulOutputDesc, valueTransOutputDesc, valueTransNode, perm, new_kv_shape, false, "value_trans",
        valueMatmulNode->GetOutDataAnchor(0), multiHeadAttentionNode->GetOutDataAnchor(4)
    );

    // attn_scores_batchmatmul
    GeTensorDesc attnScoresBatchOutputDesc;
    SetNZTensorDesc(attnScoresBatchOutputDesc, attn_batchmatmul_shape);
    NodePtr attnScoresBatchNode;
    AddBatchMatmulNode(graph, queryTransOutputDesc, keyTransOutputDesc, attnScoresBatchOutputDesc, attnScoresBatchNode, 
        false, true, "attn_scores_batchmatmul", queryTransNode->GetOutDataAnchor(0), keyTransNode->GetOutDataAnchor(0)
    );
 
    // attn_scores_add
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Define attn_scores_add begin");
    OpDescPtr attnScoresAddOpDesc;
    FUSION_PASS_MAKE_SHARED((attnScoresAddOpDesc = std::make_shared<ge::OpDesc>("attn_scores_add", "Add")), return INTERNAL_ERROR);
    attnScoresAddOpDesc->AddInputDesc("x1", multiHeadAttentionDesc->GetInputDesc("attn_mask"));
    attnScoresAddOpDesc->AddInputDesc("x2", attnScoresBatchOutputDesc);
    attnScoresAddOpDesc->AddOutputDesc("y", attnScoresBatchOutputDesc);
    NodePtr attnScoresAddNode = graph.AddNode(attnScoresAddOpDesc);
    GraphUtils::AddEdge(multiHeadAttentionNode->GetInDataAnchor(6)->GetPeerOutAnchor(), attnScoresAddNode->GetInDataAnchor(0));
    GraphUtils::AddEdge(attnScoresBatchNode->GetOutDataAnchor(0), attnScoresAddNode->GetInDataAnchor(1));

    // attn_scores_softmax
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Define attn_scores_softmax begin");
    OpDescPtr softmaxOpDesc;
    NodePtr softmaxNode;
    if (softmax_use_float) {
        OpDescPtr castOpDesc, beforeCastOpDesc;
        NodePtr castNode, beforeCastNode;
        GeTensorDesc castOutputDesc;
        SetNZTensorDesc(castOutputDesc, attn_batchmatmul_shape, DT_FLOAT);
        // cast_before_softmax
        AddCastNode(graph, castOpDesc, attnScoresBatchOutputDesc, castOutputDesc, castNode, DT_FLOAT, "cast_before_softmax",
            attnScoresAddNode->GetOutDataAnchor(0));
        // attn_scores_softmax
        AddSoftmaxNode(graph, beforeCastOpDesc, castOutputDesc, castOutputDesc, beforeCastNode, {-1}, "attn_scores_softmax",
            castNode->GetOutDataAnchor(0));
        AddNodeLinkOut(beforeCastNode->GetOutDataAnchor(0), 
            multiHeadAttentionNode->GetOutDataAnchor(5), "attn_scores_softmax");
        // cast_after_softmax
        AddCastNode(graph, softmaxOpDesc, castOutputDesc, attnScoresBatchOutputDesc, softmaxNode, DT_FLOAT16, "cast_after_softmax",
            beforeCastNode->GetOutDataAnchor(0));
    } else {
        AddSoftmaxNode(graph, softmaxOpDesc, attnScoresBatchOutputDesc, attnScoresBatchOutputDesc, 
            softmaxNode, {-1}, "attn_scores_softmax", attnScoresAddNode->GetOutDataAnchor(0));
        AddNodeLinkOut(softmaxNode->GetOutDataAnchor(0), 
            multiHeadAttentionNode->GetOutDataAnchor(5), "attn_scores_softmax");
    }

    // dropout
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Define dropout begin");
    OpDescPtr attnResOpDesc;
    NodePtr attnResNode;
    GeTensorDesc dropoutMaskTensorDesc = GeTensorDesc(GeShape({gen_mask_shape}), FORMAT_ND, DT_UINT8);
    if (keep_prob < 1.0) {
        // keep_prob
        GeTensorDesc probTensorDesc = GeTensorDesc(GeShape(), FORMAT_ND, DT_FLOAT);
        NodePtr probNode;
        AddConstNode(graph, probTensorDesc, probNode, reinterpret_cast<uint8_t*>(&keep_prob), sizeof(float), "keep_prob");
        if (!multiHeadAttentionDesc->MutableInputDesc("dropout_mask")) {
            // attn_res_shape
            GeTensorDesc attnResOutputDesc = GeTensorDesc(GeShape({1}), FORMAT_ND, DT_INT64);
            NodePtr attnResShapeNode;
            AddConstNode(graph, attnResOutputDesc, attnResShapeNode, reinterpret_cast<uint8_t*>(const_cast<int64_t*>(&attn_res_shape)), 
                sizeof(int64_t), "attn_res_shape");

            // dropout_gen_mask
            OP_LOGI(FUSED_OP_TYPE.c_str(), "Define attn_res_shape begin");
            OpDescPtr dropoutGenMaskOpDesc;
            FUSION_PASS_MAKE_SHARED((dropoutGenMaskOpDesc = std::make_shared<ge::OpDesc>("dropout_gen_mask", "DropOutGenMask")), return INTERNAL_ERROR);
            dropoutGenMaskOpDesc->AddInputDesc("shape", attnResOutputDesc);
            dropoutGenMaskOpDesc->AddInputDesc("prob", probTensorDesc);
            AttrUtils::SetInt(dropoutGenMaskOpDesc, "seed", 0);
            AttrUtils::SetInt(dropoutGenMaskOpDesc, "seed2", 0);
            dropoutGenMaskOpDesc->AddOutputDesc("y", dropoutMaskTensorDesc);
            NodePtr dropoutMaskNode = graph.AddNode(dropoutGenMaskOpDesc);
            GraphUtils::AddEdge(attnResShapeNode->GetOutDataAnchor(0), dropoutMaskNode->GetInDataAnchor(0));
            GraphUtils::AddEdge(probNode->GetOutDataAnchor(0), dropoutMaskNode->GetInDataAnchor(1));
            AddNodeLinkOut(dropoutMaskNode->GetOutDataAnchor(0), 
                multiHeadAttentionNode->GetOutDataAnchor(1), "dropout_gen_mask");
            // dropout_do_mask
            AddDropOutDoMaskNode(graph, attnResOpDesc, attnScoresBatchOutputDesc, dropoutMaskTensorDesc, probTensorDesc,
                attnScoresBatchOutputDesc, attnResNode, "dropout_do_mask", softmaxNode->GetOutDataAnchor(0),
                dropoutMaskNode->GetOutDataAnchor(0), probNode->GetOutDataAnchor(0)
            );
        } else {
            AddNodeLinkOut(multiHeadAttentionNode->GetInDataAnchor(12)->GetPeerOutAnchor(), 
                multiHeadAttentionNode->GetOutDataAnchor(1), "dropout_mask");
            // dropout_do_mask
            AddDropOutDoMaskNode(graph, attnResOpDesc, attnScoresBatchOutputDesc, 
                *(multiHeadAttentionDesc->MutableInputDesc("dropout_mask")), 
                probTensorDesc, attnScoresBatchOutputDesc, attnResNode, "dropout_do_mask", 
                softmaxNode->GetOutDataAnchor(0), multiHeadAttentionNode->GetInDataAnchor(12)->GetPeerOutAnchor(), 
                probNode->GetOutDataAnchor(0)
            );
        }
    } else {
        // dropout_empty
        uint8_t outmask[gen_mask_shape];
        memset(outmask, 0xff, gen_mask_shape);
        GeTensorPtr maskValue = std::make_shared<ge::GeTensor>(dropoutMaskTensorDesc, reinterpret_cast<uint8_t*>(&outmask), gen_mask_shape);
        NodePtr dropoutMaskNode;
        AddConstNode(graph, dropoutMaskTensorDesc, dropoutMaskNode, 
            reinterpret_cast<uint8_t*>(&outmask), gen_mask_shape, "dropout_empty");
        AddNodeLinkOut(dropoutMaskNode->GetOutDataAnchor(0), 
            multiHeadAttentionNode->GetOutDataAnchor(1), "dropout_empty");
        attnResOpDesc = softmaxOpDesc;
        attnResNode = softmaxNode;
    }
    // relink out
    AddNodeLinkOut(attnResNode->GetOutDataAnchor(0), 
        multiHeadAttentionNode->GetOutDataAnchor(6), "attn_res");
    // context_batchmatmul
    GeTensorDesc contextBatchOutputDesc;
    SetNZTensorDesc(contextBatchOutputDesc, context_batchmatmul_shape);
    NodePtr contextBatchNode;
    AddBatchMatmulNode(graph, attnScoresBatchOutputDesc, valueTransOutputDesc, contextBatchOutputDesc, contextBatchNode, 
        false, false, "context_batchmatmul", attnResNode->GetOutDataAnchor(0), valueTransNode->GetOutDataAnchor(0)
    );

    // context_trans
    GeTensorDesc contextTransOutputDesc;
    SetNZTensorDesc(contextTransOutputDesc, context_shape);
    NodePtr contextTransNode;
    AddTransposeNode(graph, contextBatchOutputDesc, contextTransOutputDesc, contextTransNode, perm, context_shape, true, "context_trans",
        contextBatchNode->GetOutDataAnchor(0), multiHeadAttentionNode->GetOutDataAnchor(7)
    );

    // result
    NodePtr resultNode;
    AddMatmulNode(graph, contextTransOutputDesc, multiHeadAttentionDesc->GetInputDesc("out_proj_weight"),
        multiHeadAttentionDesc->MutableInputDesc("out_proj_bias"), multiHeadAttentionDesc->GetOutputDesc("y"), 
        resultNode, false, true, "result",
        contextTransNode->GetOutDataAnchor(0), multiHeadAttentionNode->GetInDataAnchor(7)->GetPeerOutAnchor(),
        multiHeadAttentionNode->GetInDataAnchor(11)->GetPeerOutAnchor()
    );
    // link out
    AddNodeLinkOut(resultNode->GetOutDataAnchor(0), 
        multiHeadAttentionNode->GetOutDataAnchor(0), "result");

    // unlink all control input
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Define unlink control input begin");
    if (multiHeadAttentionNode->GetInControlAnchor() != nullptr) {
        multiHeadAttentionNode->GetInControlAnchor()->UnlinkAll();
    }

    // unlink all input
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Define unlinks input begin");
    for (auto inAnchor : multiHeadAttentionNode->GetAllInDataAnchors()) {
        if (inAnchor != nullptr) {
            inAnchor->UnlinkAll();
        }
    }
    // remove all output
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Define unlinks output begin");
    for (auto outAnchor : multiHeadAttentionNode->GetAllOutDataAnchors()) {
        if (outAnchor != nullptr) {
            outAnchor->UnlinkAll();
        }
    }
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(multiHeadAttentionNode),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove multiHeadAttentionNode failed"), return FAILED);
        
    return SUCCESS;
}

REGISTER_PASS("MultiHeadAttentionFusionPass", BUILT_IN_GRAPH_PASS, MultiHeadAttentionFusionPass);
} // namespace 