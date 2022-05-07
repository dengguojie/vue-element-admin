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

namespace {

constexpr int OFFSET_FOR_ALIGNMENT = 15;
constexpr int ALIGNMENT = 16;

constexpr int INPUT_QUERY = 0;
constexpr int INPUT_KEY = 1;
constexpr int INPUT_VALUE = 2;
constexpr int INPUT_QUERY_WEIGHT = 3;
constexpr int INPUT_KEY_WEIGHT = 4;
constexpr int INPUT_VALUE_WEIGHT = 5;
constexpr int INPUT_ATTN_MASK = 6;
constexpr int INPUT_OUT_PROJ_WEIGHT = 7;
constexpr int INPUT_QUERY_BIAS = 8;
constexpr int INPUT_KEY_BIAS = 9;
constexpr int INPUT_VALUE_BIAS = 10;
constexpr int INPUT_OUT_PROJ_BIAS = 11;
constexpr int INPUT_DROPOUT_MASK = 12;

constexpr int OUTPUT_Y = 0;
constexpr int OUTPUT_DROPOUT_MASK = 1;
constexpr int OUTPUT_QUERY_RES = 2;
constexpr int OUTPUT_KEY_RES = 3;
constexpr int OUTPUT_VALUE_RES = 4;
constexpr int OUTPUT_ATTN_SCORES = 5;
constexpr int OUTPUT_ATTN_RES = 6;
constexpr int OUTPUT_CONTEXT = 7;

}
namespace fe {

static void SetNZTensorDesc(ge::GeTensorDesc &tensorDesc, const vector<int64_t> &oriDims,
    const ge::DataType dtype = DT_FLOAT16) {
    vector<int64_t> dims;
    int32_t dim = oriDims.size();
    for (auto i = 0; i < dim - 2; i++) {
        dims.push_back(oriDims[i]);
    }
    dims.push_back((oriDims[dim-1] + OFFSET_FOR_ALIGNMENT) / ALIGNMENT);
    dims.push_back((oriDims[dim-2] + OFFSET_FOR_ALIGNMENT) / ALIGNMENT); // dim-2: the last second element.
    dims.push_back(16);  // 16 means nz-format alignment
    dims.push_back(16);  // 16 means nz-format alignment
    tensorDesc.SetShape(ge::GeShape(dims));
    tensorDesc.SetDataType(dtype);
    tensorDesc.SetFormat(FORMAT_FRACTAL_NZ);
    tensorDesc.SetOriginShape(ge::GeShape(oriDims));
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
static Status AddNodeLinkOut(_InAnchor inAnchor, _OutAnchor outAnchor, const string& outNodeName) {
    // link out
    OP_LOGI("MultiHeadAttention", "Define %s link out begin", outNodeName.c_str());
    for (auto anchor : outAnchor->GetPeerInDataAnchors()) {
        GraphUtils::RemoveEdge(outAnchor, anchor);
        GraphUtils::AddEdge(inAnchor, anchor);
    }
    OP_LOGI("MultiHeadAttention", "Define %s link out end", outNodeName.c_str());
    return SUCCESS;
}

template<typename _InAnchor1, typename _InAnchor2, typename _InAnchor3>
static Status AddMatmulNode(ge::ComputeGraph& graph, const ge::GeTensorDescPtr& x1Desc,
    const ge::GeTensorDescPtr& x2Desc, const ge::GeTensorDescPtr& biasDesc,
    const ge::GeTensorDesc& yDesc, ge::NodePtr& newNode, bool transposeX1,
    bool transposeX2, const string& nodeName, _InAnchor1 inAnchor1, _InAnchor2 inAnchor2, _InAnchor3 inAnchor3)
{
    OP_LOGI("MultiHeadAttention", "Define %s begin", nodeName.c_str());
    OpDescPtr matmulOpDesc;
    FUSION_PASS_MAKE_SHARED((matmulOpDesc = std::make_shared<ge::OpDesc>(nodeName, "MatMulV2")),
    return INTERNAL_ERROR);
    matmulOpDesc->AddInputDesc("x1", *x1Desc);
    matmulOpDesc->AddInputDesc("x2", *x2Desc);
    if (biasDesc) {
        matmulOpDesc->AddInputDesc("bias", *biasDesc);
    }
    AttrUtils::SetBool(matmulOpDesc, "transpose_x1", transposeX1);
    AttrUtils::SetBool(matmulOpDesc, "transpose_x2", transposeX2);
    matmulOpDesc->AddOutputDesc("y", yDesc);
    newNode = graph.AddNode(matmulOpDesc);
    GraphUtils::AddEdge(inAnchor1, newNode->GetInDataAnchor(0));
    GraphUtils::AddEdge(inAnchor2, newNode->GetInDataAnchor(1));
    if (biasDesc) {
        GraphUtils::AddEdge(inAnchor3, newNode->GetInDataAnchor(2)); // get the newNode's 2nd input
    }
    OP_LOGI("MultiHeadAttention", "Define %s end", nodeName.c_str());
    return SUCCESS;
}

template<typename _InAnchor, typename _OutAnchor>
static Status AddTransposeNode(ge::ComputeGraph& graph, const ge::GeTensorDesc& xDesc, const ge::GeTensorDesc& yDesc,
    ge::NodePtr& newNode, const vector<int64_t>& perm, const vector<int64_t>& shape,
    bool transposeFirst, const string& nodeName, _InAnchor inAnchor, _OutAnchor outAnchor)
{
    OP_LOGI("MultiHeadAttention", "Define %s begin", nodeName.c_str());
    OpDescPtr transOpDesc;
    FUSION_PASS_MAKE_SHARED((transOpDesc = std::make_shared<ge::OpDesc>(nodeName, "ConfusionTransposeD")),
        return INTERNAL_ERROR);
    transOpDesc->AddInputDesc("x", xDesc);
    AttrUtils::SetListInt(transOpDesc, "perm", perm);
    AttrUtils::SetListInt(transOpDesc, "shape", shape);
    AttrUtils::SetBool(transOpDesc, "transpose_first", transposeFirst);
    transOpDesc->AddOutputDesc("y", yDesc);
    newNode = graph.AddNode(transOpDesc);
    GraphUtils::AddEdge(inAnchor, newNode->GetInDataAnchor(0));
    AddNodeLinkOut(newNode->GetOutDataAnchor(0), outAnchor, nodeName);
    OP_LOGI("MultiHeadAttention", "Define %s end", nodeName.c_str());
    return SUCCESS;
}

template<typename _InAnchor1, typename _InAnchor2>
static Status AddBatchMatmulNode(ge::ComputeGraph& graph, const ge::GeTensorDesc& x1Desc,
    const ge::GeTensorDesc& x2Desc,
    const ge::GeTensorDesc& yDesc, ge::NodePtr& newNode, bool adjX1, bool adjX2,
    const string& nodeName, _InAnchor1 inAnchor1, _InAnchor2 inAnchor2)
{
    OP_LOGI("MultiHeadAttention", "Define %s begin", nodeName.c_str());
    OpDescPtr batchOpDesc;
    FUSION_PASS_MAKE_SHARED((batchOpDesc = std::make_shared<ge::OpDesc>(nodeName, "BatchMatMul")),
        return INTERNAL_ERROR);
    batchOpDesc->AddInputDesc("x1", x1Desc);
    batchOpDesc->AddInputDesc("x2", x2Desc);
    AttrUtils::SetBool(batchOpDesc, "adj_x1", adjX1);
    AttrUtils::SetBool(batchOpDesc, "adj_x2", adjX2);
    batchOpDesc->AddOutputDesc("y", yDesc);
    newNode = graph.AddNode(batchOpDesc);
    GraphUtils::AddEdge(inAnchor1, newNode->GetInDataAnchor(0));
    GraphUtils::AddEdge(inAnchor2, newNode->GetInDataAnchor(1));
    OP_LOGI("MultiHeadAttention", "Define %s end", nodeName.c_str());
    return SUCCESS;
}

static Status AddConstNode(ge::ComputeGraph& graph, const ge::GeTensorDesc& yDesc, ge::NodePtr& newNode,
    const uint8_t* dataPtr, size_t size, const string& nodeName)
{
    OP_LOGI("MultiHeadAttention", "Define %s begin", nodeName.c_str());
    OpDescPtr constOpDesc;
    FUSION_PASS_MAKE_SHARED((constOpDesc = std::make_shared<ge::OpDesc>(nodeName, "Const")), return INTERNAL_ERROR);
    GeTensorPtr constValue = std::make_shared<ge::GeTensor>(yDesc, dataPtr, size);
    AttrUtils::SetTensor(constOpDesc, ATTR_NAME_WEIGHTS, constValue);
    constOpDesc->AddOutputDesc("y", yDesc);
    newNode = graph.AddNode(constOpDesc);
    OP_LOGI("MultiHeadAttention", "Define %s end", nodeName.c_str());
    return SUCCESS;
}

template<typename _InAnchor1, typename _InAnchor2, typename _InAnchor3>
static Status AddDropOutDoMaskNode(ge::ComputeGraph& graph, ge::OpDescPtr& opDesc, const ge::GeTensorDesc& xDesc,
    const ge::GeTensorDesc& maskDesc, const ge::GeTensorDesc& probDesc,
    const ge::GeTensorDesc& yDesc, ge::NodePtr& newNode,
    const string& nodeName, _InAnchor1 inAnchor1, _InAnchor2 inAnchor2, _InAnchor3 inAnchor3)
{
    OP_LOGI("MultiHeadAttention", "Define %s begin", nodeName.c_str());
    FUSION_PASS_MAKE_SHARED((opDesc = std::make_shared<ge::OpDesc>(nodeName, "DropOutDoMask")),
        return INTERNAL_ERROR);
    opDesc->AddInputDesc("x", xDesc);
    opDesc->AddInputDesc("mask", maskDesc);
    opDesc->AddInputDesc("keep_prob", probDesc);
    opDesc->AddOutputDesc("y", yDesc);
    newNode = graph.AddNode(opDesc);
    GraphUtils::AddEdge(inAnchor1, newNode->GetInDataAnchor(0));
    GraphUtils::AddEdge(inAnchor2, newNode->GetInDataAnchor(1));
    GraphUtils::AddEdge(inAnchor3, newNode->GetInDataAnchor(2)); // get the newNode's 2nd input
    OP_LOGI("MultiHeadAttention", "Define %s end", nodeName.c_str());
    return SUCCESS;
}

template<typename _InAnchor>
static Status AddCastNode(ge::ComputeGraph& graph, ge::OpDescPtr& opDesc, const ge::GeTensorDesc& xDesc,
    const ge::GeTensorDesc& yDesc, ge::NodePtr& newNode, int32_t dstType,
    const string& nodeName, _InAnchor inAnchor)
{
    OP_LOGI("MultiHeadAttention", "Define %s begin", nodeName.c_str());
    FUSION_PASS_MAKE_SHARED((opDesc = std::make_shared<ge::OpDesc>(nodeName, "Cast")), return INTERNAL_ERROR);
    opDesc->AddInputDesc("x", xDesc);
    AttrUtils::SetInt(opDesc, "dst_type", dstType);
    opDesc->AddOutputDesc("y", yDesc);
    newNode = graph.AddNode(opDesc);
    GraphUtils::AddEdge(inAnchor, newNode->GetInDataAnchor(0));
    OP_LOGI("MultiHeadAttention", "Define %s end", nodeName.c_str());
    return SUCCESS;
}

template<typename _InAnchor>
static Status AddSoftmaxNode(ge::ComputeGraph& graph, ge::OpDescPtr& opDesc, const ge::GeTensorDesc& xDesc,
    const ge::GeTensorDesc& yDesc, ge::NodePtr& newNode, vector<int64_t> axes,
    const string& nodeName, _InAnchor inAnchor)
{
    OP_LOGI("MultiHeadAttention", "Define %s begin", nodeName.c_str());
    FUSION_PASS_MAKE_SHARED((opDesc = std::make_shared<ge::OpDesc>(nodeName, "SoftmaxV2")), return INTERNAL_ERROR);
    opDesc->AddInputDesc("x", xDesc);
    AttrUtils::SetListInt(opDesc, "axes", axes);
    opDesc->AddOutputDesc("y", yDesc);
    newNode = graph.AddNode(opDesc);
    GraphUtils::AddEdge(inAnchor, newNode->GetInDataAnchor(0));
    OP_LOGI("MultiHeadAttention", "Define %s end", nodeName.c_str());
    return SUCCESS;
}

Status MultiHeadAttentionFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
    vector<ge::NodePtr>& fusionNodes)
{
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Define MultiHeadAttentionFusionPass fusion begin");
    ge::NodePtr multiHeadAttentionNode = GetNodeFromMapping(FUSED_OP_TYPE, mapping);
    FUSION_PASS_CHECK(multiHeadAttentionNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(),
    "MultiHeadAttention node is null, fusion failed."),
                        return PARAM_INVALID);
    ge::OpDescPtr multiHeadAttentionDesc = multiHeadAttentionNode->GetOpDesc();
    FUSION_PASS_CHECK(multiHeadAttentionDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(),
    "MultiHeadAttention's Op_desc is null, fusion failed."),
                        return PARAM_INVALID);
    // shape
    vector<int64_t> query_shape = multiHeadAttentionDesc->GetInputDesc("query").GetOriginShape().GetDims();
    FUSION_PASS_CHECK(query_shape.size() !=2, OP_LOGE(FUSED_OP_TYPE.c_str(),
    "MultiHeadAttention's Query origin shape should be 2D, fusion failed."),
                        return PARAM_INVALID);
    int64_t attnHeadNum;
    int64_t attnDimPerHead;
    int64_t srcLen;
    int64_t tgtLen;
    float keepProb;
    bool softmaxUseFloat;
    AttrUtils::GetInt(multiHeadAttentionDesc, "attn_head_num", attnHeadNum);
    AttrUtils::GetInt(multiHeadAttentionDesc, "attn_dim_per_head", attnDimPerHead);
    AttrUtils::GetInt(multiHeadAttentionDesc, "src_len", srcLen);
    AttrUtils::GetInt(multiHeadAttentionDesc, "tgt_len", tgtLen);
    AttrUtils::GetFloat(multiHeadAttentionDesc, "keep_prob", keepProb);
    AttrUtils::GetBool(multiHeadAttentionDesc, "softmax_use_float", softmaxUseFloat);
    FUSION_PASS_CHECK((attnHeadNum <= 0 || attnDimPerHead <= 0 || srcLen <= 0 || tgtLen <= 0),
        OP_LOGE(FUSED_OP_TYPE.c_str(),
        "MultiHeadAttention's attn_head_num, attn_dim_per_head, src_len, tgt_len should greater than 0, fusion failed."
        ),
                    return PARAM_INVALID);
    FUSION_PASS_CHECK(!(attnHeadNum % ALIGNMENT == 0 &&
    attnDimPerHead % ALIGNMENT ==  0 && srcLen % ALIGNMENT ==  0 && tgtLen % ALIGNMENT == 0),
        OP_LOGE(FUSED_OP_TYPE.c_str(),
        "MultiHeadAttention's attn_head_num, attn_dim_per_head, src_len, tgt_len should align of 16, fusion failed."),
                    return PARAM_INVALID);
    const int64_t batch = query_shape[0] / tgtLen;
    const int64_t weightCol = attnHeadNum * attnDimPerHead;
    const int64_t genMaskShape = batch * attnHeadNum * srcLen * tgtLen / 8;
    const int64_t attnResShape = batch * attnHeadNum * srcLen * tgtLen;
    const float scale = 1.0 / sqrt(attnDimPerHead);
    
    const vector<int64_t> perm({0, 2, 1, 3});
    const vector<int64_t> query_matmul_shape({batch*tgtLen, weightCol});
    const vector<int64_t> kv_matmul_shape({batch*srcLen, weightCol});
    const vector<int64_t> new_query_shape({batch, tgtLen, attnHeadNum, attnDimPerHead});
    const vector<int64_t> transpose_new_query_shape({batch, attnHeadNum, tgtLen, attnDimPerHead});
    const vector<int64_t> new_kv_shape({batch, srcLen, attnHeadNum, attnDimPerHead});
    const vector<int64_t> transpose_new_kv_shape({batch, attnHeadNum, srcLen, attnDimPerHead});
    const vector<int64_t> softmax_scores_shape({batch, attnHeadNum, tgtLen, srcLen});
    const vector<int64_t> attn_batchmatmul_shape({batch, attnHeadNum, tgtLen, srcLen});
    const vector<int64_t> context_batchmatmul_shape({batch, attnHeadNum, tgtLen, attnDimPerHead});
    const vector<int64_t> context_shape({batch*tgtLen, weightCol});
    static const uint8_t full_mask = 0xff;
    // query_matmul
    GeTensorDesc queryMatmulOutputDesc;
    SetNZTensorDesc(queryMatmulOutputDesc, query_matmul_shape);
    NodePtr queryMatmulNode;
    AddMatmulNode(graph, multiHeadAttentionDesc->MutableInputDesc("query"),
        multiHeadAttentionDesc->MutableInputDesc("query_weight"),
        multiHeadAttentionDesc->MutableInputDesc("query_bias"),
        queryMatmulOutputDesc, queryMatmulNode, false, true,
        multiHeadAttentionNode->GetName() + "_query_matmul",
        multiHeadAttentionNode->GetInDataAnchor(INPUT_QUERY)->GetPeerOutAnchor(),
        multiHeadAttentionNode->GetInDataAnchor(INPUT_QUERY_WEIGHT)->GetPeerOutAnchor(),
        multiHeadAttentionNode->GetInDataAnchor(INPUT_QUERY_BIAS)->GetPeerOutAnchor()
    );
    // key_matmul
    GeTensorDesc kvMatmulOutputDesc;
    SetNZTensorDesc(kvMatmulOutputDesc, kv_matmul_shape);
    NodePtr keyMatmulNode;
    AddMatmulNode(graph, multiHeadAttentionDesc->MutableInputDesc("key"),
        multiHeadAttentionDesc->MutableInputDesc("key_weight"),
        multiHeadAttentionDesc->MutableInputDesc("key_bias"), kvMatmulOutputDesc, keyMatmulNode, false, true,
        multiHeadAttentionNode->GetName() + "_key_matmul",
        multiHeadAttentionNode->GetInDataAnchor(INPUT_KEY)->GetPeerOutAnchor(),
        multiHeadAttentionNode->GetInDataAnchor(INPUT_KEY_WEIGHT)->GetPeerOutAnchor(),
        multiHeadAttentionNode->GetInDataAnchor(INPUT_KEY_BIAS)->GetPeerOutAnchor()
    );

    // value_matmul
    NodePtr valueMatmulNode;
    AddMatmulNode(graph, multiHeadAttentionDesc->MutableInputDesc("value"),
        multiHeadAttentionDesc->MutableInputDesc("value_weight"),
        multiHeadAttentionDesc->MutableInputDesc("value_bias"), kvMatmulOutputDesc, valueMatmulNode, false, true,
        multiHeadAttentionNode->GetName() + "_value_matmul",
        multiHeadAttentionNode->GetInDataAnchor(INPUT_VALUE)->GetPeerOutAnchor(),
        multiHeadAttentionNode->GetInDataAnchor(INPUT_VALUE_WEIGHT)->GetPeerOutAnchor(),
        multiHeadAttentionNode->GetInDataAnchor(INPUT_VALUE_BIAS)->GetPeerOutAnchor()
    );

    // query_muls
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Define query_muls begin");
    OpDescPtr queryMulsOpDesc;
    FUSION_PASS_MAKE_SHARED((queryMulsOpDesc = std::make_shared<ge::OpDesc>(
        multiHeadAttentionNode->GetName() + "_query_muls", "Muls")), return INTERNAL_ERROR);
    queryMulsOpDesc->AddInputDesc("x", queryMatmulOutputDesc);
    AttrUtils::SetFloat(queryMulsOpDesc, "value", scale);
    queryMulsOpDesc->AddOutputDesc("y", queryMatmulOutputDesc);
    NodePtr queryMulsNode = graph.AddNode(queryMulsOpDesc);
    GraphUtils::AddEdge(queryMatmulNode->GetOutDataAnchor(0), queryMulsNode->GetInDataAnchor(0));

    
    // query_trans
    GeTensorDesc queryTransOutputDesc;
    SetNZTensorDesc(queryTransOutputDesc, transpose_new_query_shape);
    NodePtr queryTransNode;
    AddTransposeNode(graph, queryMatmulOutputDesc, queryTransOutputDesc, queryTransNode, perm, new_query_shape, false,
        multiHeadAttentionNode->GetName() + "_query_trans",
        queryMulsNode->GetOutDataAnchor(0),
        multiHeadAttentionNode->GetOutDataAnchor(OUTPUT_QUERY_RES)
    );

    // key_trans
    GeTensorDesc keyTransOutputDesc;
    SetNZTensorDesc(keyTransOutputDesc, transpose_new_kv_shape);
    NodePtr keyTransNode;
    AddTransposeNode(graph, kvMatmulOutputDesc, keyTransOutputDesc, keyTransNode, perm, new_kv_shape, false,
        multiHeadAttentionNode->GetName() + "_key_trans",
        keyMatmulNode->GetOutDataAnchor(0),
        multiHeadAttentionNode->GetOutDataAnchor(OUTPUT_KEY_RES)
    );

    // value_trans
    GeTensorDesc valueTransOutputDesc;
    SetNZTensorDesc(valueTransOutputDesc, transpose_new_kv_shape);
    NodePtr valueTransNode;
    AddTransposeNode(graph, kvMatmulOutputDesc, valueTransOutputDesc, valueTransNode, perm, new_kv_shape, false,
        multiHeadAttentionNode->GetName() + "_value_trans",
        valueMatmulNode->GetOutDataAnchor(0),
        multiHeadAttentionNode->GetOutDataAnchor(OUTPUT_VALUE_RES)
    );

    // attn_scores_batchmatmul
    GeTensorDesc attnScoresBatchOutputDesc;
    SetNZTensorDesc(attnScoresBatchOutputDesc, attn_batchmatmul_shape);
    NodePtr attnScoresBatchNode;
    AddBatchMatmulNode(graph, queryTransOutputDesc, keyTransOutputDesc, attnScoresBatchOutputDesc,
        attnScoresBatchNode, false, true,
        multiHeadAttentionNode->GetName() + "_attn_scores_batchmatmul",
        queryTransNode->GetOutDataAnchor(0), keyTransNode->GetOutDataAnchor(0)
    );
 
    // attn_scores_add
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Define attn_scores_add begin");
    OpDescPtr attnScoresAddOpDesc;
    FUSION_PASS_MAKE_SHARED((attnScoresAddOpDesc = std::make_shared<ge::OpDesc>(
        multiHeadAttentionNode->GetName() + "_attn_scores_add", "Add")), return INTERNAL_ERROR);
    attnScoresAddOpDesc->AddInputDesc("x1", multiHeadAttentionDesc->GetInputDesc("attn_mask"));
    attnScoresAddOpDesc->AddInputDesc("x2", attnScoresBatchOutputDesc);
    attnScoresAddOpDesc->AddOutputDesc("y", attnScoresBatchOutputDesc);
    NodePtr attnScoresAddNode = graph.AddNode(attnScoresAddOpDesc);
    GraphUtils::AddEdge(multiHeadAttentionNode->GetInDataAnchor(INPUT_ATTN_MASK)->GetPeerOutAnchor(),
        attnScoresAddNode->GetInDataAnchor(0));
    GraphUtils::AddEdge(attnScoresBatchNode->GetOutDataAnchor(0), attnScoresAddNode->GetInDataAnchor(1));

    // attn_scores_softmax
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Define attn_scores_softmax begin");
    OpDescPtr softmaxOpDesc;
    NodePtr softmaxNode;
    if (softmaxUseFloat) {
        OpDescPtr castOpDesc, beforeCastOpDesc;
        NodePtr castNode, beforeCastNode;
        GeTensorDesc castOutputDesc;
        SetNZTensorDesc(castOutputDesc, attn_batchmatmul_shape, DT_FLOAT);
        // cast_before_softmax
        AddCastNode(graph, castOpDesc, attnScoresBatchOutputDesc, castOutputDesc, castNode, DT_FLOAT,
            multiHeadAttentionNode->GetName() + "_cast_before_softmax",
            attnScoresAddNode->GetOutDataAnchor(0));
        // attn_scores_softmax
        AddSoftmaxNode(graph, beforeCastOpDesc, castOutputDesc, castOutputDesc, beforeCastNode, {-1},
            multiHeadAttentionNode->GetName() + "_attn_scores_softmax",
            castNode->GetOutDataAnchor(0));
        AddNodeLinkOut(beforeCastNode->GetOutDataAnchor(0),
            multiHeadAttentionNode->GetOutDataAnchor(OUTPUT_ATTN_SCORES),
            multiHeadAttentionNode->GetName() + "_attn_scores_softmax");
        // cast_after_softmax
        AddCastNode(graph, softmaxOpDesc, castOutputDesc, attnScoresBatchOutputDesc, softmaxNode, DT_FLOAT16,
            multiHeadAttentionNode->GetName() + "_cast_after_softmax",
            beforeCastNode->GetOutDataAnchor(0));
    } else {
        AddSoftmaxNode(graph, softmaxOpDesc, attnScoresBatchOutputDesc, attnScoresBatchOutputDesc, softmaxNode, {-1},
            multiHeadAttentionNode->GetName() + "_attn_scores_softmax",
            attnScoresAddNode->GetOutDataAnchor(0));
        AddNodeLinkOut(softmaxNode->GetOutDataAnchor(0),
            multiHeadAttentionNode->GetOutDataAnchor(OUTPUT_ATTN_SCORES),
            multiHeadAttentionNode->GetName() + "_attn_scores_softmax");
    }

    // dropout
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Define dropout begin");
    OpDescPtr attnResOpDesc;
    NodePtr attnResNode;
    GeTensorDesc dropoutMaskTensorDesc = GeTensorDesc(GeShape({genMaskShape}), FORMAT_ND, DT_UINT8);
    if (keepProb < 1.0) {
        // keep_prob
        GeTensorDesc probTensorDesc = GeTensorDesc(GeShape(), FORMAT_ND, DT_FLOAT);
        NodePtr probNode;
        AddConstNode(graph, probTensorDesc, probNode, reinterpret_cast<uint8_t*>(&keepProb), sizeof(float),
            multiHeadAttentionNode->GetName() + "_keep_prob");
        if (!multiHeadAttentionDesc->MutableInputDesc("dropout_mask_input")) {
            // attn_res_shape
            GeTensorDesc attnResOutputDesc = GeTensorDesc(GeShape({1}), FORMAT_ND, DT_INT64);
            NodePtr attnResShapeNode;
            AddConstNode(graph, attnResOutputDesc, attnResShapeNode,
                reinterpret_cast<uint8_t*>(const_cast<int64_t*>(&attnResShape)),
                sizeof(int64_t), multiHeadAttentionNode->GetName() + "_attn_res_shape");

            // dropout_gen_mask
            OP_LOGI(FUSED_OP_TYPE.c_str(), "Define attn_res_shape begin");
            OpDescPtr dropoutGenMaskOpDesc;
            FUSION_PASS_MAKE_SHARED((dropoutGenMaskOpDesc = std::make_shared<ge::OpDesc>(
                multiHeadAttentionNode->GetName() + "_dropout_gen_mask", "DropOutGenMask")), return INTERNAL_ERROR);
            dropoutGenMaskOpDesc->AddInputDesc("shape", attnResOutputDesc);
            dropoutGenMaskOpDesc->AddInputDesc("prob", probTensorDesc);
            AttrUtils::SetInt(dropoutGenMaskOpDesc, "seed", 0);
            AttrUtils::SetInt(dropoutGenMaskOpDesc, "seed2", 0);
            dropoutGenMaskOpDesc->AddOutputDesc("y", dropoutMaskTensorDesc);
            NodePtr dropoutMaskNode = graph.AddNode(dropoutGenMaskOpDesc);
            GraphUtils::AddEdge(attnResShapeNode->GetOutDataAnchor(0), dropoutMaskNode->GetInDataAnchor(0));
            GraphUtils::AddEdge(probNode->GetOutDataAnchor(0), dropoutMaskNode->GetInDataAnchor(1));
            AddNodeLinkOut(dropoutMaskNode->GetOutDataAnchor(0),
                multiHeadAttentionNode->GetOutDataAnchor(OUTPUT_DROPOUT_MASK),
                multiHeadAttentionNode->GetName() + "_dropout_gen_mask");
            // dropout_do_mask
            AddDropOutDoMaskNode(graph, attnResOpDesc, attnScoresBatchOutputDesc, dropoutMaskTensorDesc,
                probTensorDesc, attnScoresBatchOutputDesc, attnResNode,
                multiHeadAttentionNode->GetName() + "_dropout_do_mask", softmaxNode->GetOutDataAnchor(0),
                dropoutMaskNode->GetOutDataAnchor(0), probNode->GetOutDataAnchor(0)
            );
        } else {
            AddNodeLinkOut(multiHeadAttentionNode->GetInDataAnchor(INPUT_DROPOUT_MASK)->GetPeerOutAnchor(),
                multiHeadAttentionNode->GetOutDataAnchor(OUTPUT_DROPOUT_MASK),
                multiHeadAttentionNode->GetName() + "_dropout_mask");
            // dropout_do_mask
            AddDropOutDoMaskNode(graph, attnResOpDesc, attnScoresBatchOutputDesc,
                *(multiHeadAttentionDesc->MutableInputDesc("dropout_mask_input")),
                probTensorDesc, attnScoresBatchOutputDesc, attnResNode,
                multiHeadAttentionNode->GetName() + "_dropout_do_mask",
                softmaxNode->GetOutDataAnchor(0),
                multiHeadAttentionNode->GetInDataAnchor(INPUT_DROPOUT_MASK)->GetPeerOutAnchor(),
                probNode->GetOutDataAnchor(0)
            );
        }
    } else {
        // dropout_empty
        vector<uint8_t> outmask(genMaskShape);
        std::fill(outmask.begin(), outmask.end(), full_mask);
        GeTensorPtr maskValue = std::make_shared<ge::GeTensor>(dropoutMaskTensorDesc, outmask.data(), genMaskShape);
        NodePtr dropoutMaskNode;
        AddConstNode(graph, dropoutMaskTensorDesc, dropoutMaskNode,
            outmask.data(), genMaskShape, multiHeadAttentionNode->GetName() + "_dropout_empty");
        AddNodeLinkOut(dropoutMaskNode->GetOutDataAnchor(0),
            multiHeadAttentionNode->GetOutDataAnchor(OUTPUT_DROPOUT_MASK),
            multiHeadAttentionNode->GetName() + "_dropout_empty");
        attnResOpDesc = softmaxOpDesc;
        attnResNode = softmaxNode;
    }
    // relink out
    AddNodeLinkOut(attnResNode->GetOutDataAnchor(0),
        multiHeadAttentionNode->GetOutDataAnchor(OUTPUT_ATTN_RES),
        multiHeadAttentionNode->GetName() + "_attn_res");
    // context_batchmatmul
    GeTensorDesc contextBatchOutputDesc;
    SetNZTensorDesc(contextBatchOutputDesc, context_batchmatmul_shape);
    NodePtr contextBatchNode;
    AddBatchMatmulNode(graph, attnScoresBatchOutputDesc, valueTransOutputDesc, contextBatchOutputDesc,
        contextBatchNode, false, false,
        multiHeadAttentionNode->GetName() + "_context_batchmatmul",
        attnResNode->GetOutDataAnchor(0), valueTransNode->GetOutDataAnchor(0)
    );

    // context_trans
    GeTensorDesc contextTransOutputDesc;
    SetNZTensorDesc(contextTransOutputDesc, context_shape);
    NodePtr contextTransNode;
    AddTransposeNode(graph, contextBatchOutputDesc, contextTransOutputDesc, contextTransNode,
        perm, context_shape, true,
        multiHeadAttentionNode->GetName() + "_context_trans",
        contextBatchNode->GetOutDataAnchor(0),
        multiHeadAttentionNode->GetOutDataAnchor(OUTPUT_CONTEXT)
    );

    // result
    NodePtr resultNode;
    AddMatmulNode(graph, std::make_shared<GeTensorDesc>(contextTransOutputDesc),
        multiHeadAttentionDesc->MutableInputDesc("out_proj_weight"),
        multiHeadAttentionDesc->MutableInputDesc("out_proj_bias"), multiHeadAttentionDesc->GetOutputDesc("y"),
        resultNode, false, true, multiHeadAttentionNode->GetName() + "_result",
        contextTransNode->GetOutDataAnchor(0),
        multiHeadAttentionNode->GetInDataAnchor(INPUT_OUT_PROJ_WEIGHT)->GetPeerOutAnchor(),
        multiHeadAttentionNode->GetInDataAnchor(INPUT_OUT_PROJ_BIAS)->GetPeerOutAnchor()
    );
    // link out
    AddNodeLinkOut(resultNode->GetOutDataAnchor(0),
        multiHeadAttentionNode->GetOutDataAnchor(OUTPUT_Y),
        multiHeadAttentionNode->GetName() + "_result");

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