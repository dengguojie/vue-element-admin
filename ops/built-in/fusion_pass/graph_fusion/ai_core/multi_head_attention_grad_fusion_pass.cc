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

namespace {

constexpr int OFFSET_FOR_ALIGNMENT = 15;
constexpr int ALIGNMENT = 16;

constexpr int INPUT_QUERY = 0;
constexpr int INPUT_KEY = 1;
constexpr int INPUT_VALUE = 2;
constexpr int INPUT_QUERY_WEIGHT = 3;
constexpr int INPUT_KEY_WEIGHT = 4;
constexpr int INPUT_VALUE_WEIGHT = 5;
constexpr int INPUT_OUT_PROJ_WEIGHT = 6;
constexpr int INPUT_QUERY_RES = 7;
constexpr int INPUT_KEY_RES = 8;
constexpr int INPUT_VALUE_RES = 9;
constexpr int INPUT_ATTN_SCORES = 10;
constexpr int INPUT_ATTN_RES = 11;
constexpr int INPUT_CONTEXT = 12;
constexpr int INPUT_Y_GRAD = 13;
constexpr int INPUT_DROPOUT_MASK = 14;

constexpr int OUTPUT_QUERY_WEIGHT_GRAD = 0;
constexpr int OUTPUT_KEY_WEIGHT_GRAD = 1;
constexpr int OUTPUT_VALUE_WEIGHT_GRAD = 2;
constexpr int OUTPUT_OUT_PROJ_WEIGHT_GRAD = 3;
constexpr int OUTPUT_QUERY_GRAD = 4;
constexpr int OUTPUT_KEY_GRAD = 5;
constexpr int OUTPUT_VALUE_GRAD = 6;
constexpr int OUTPUT_QUERY_BIAS_GRAD = 7;
constexpr int OUTPUT_KEY_BIAS_GRAD = 8;
constexpr int OUTPUT_VALUE_BIAS_GRAD = 9;
constexpr int OUTPUT_OUT_PROJ_BIAS_GRAD = 10;

}

namespace fe {
static void SetNDTensorDesc(ge::GeTensorDesc &tensorDesc, const vector<int64_t> &oriDims,
    const ge::DataType dtype = DT_FLOAT16) {
    tensorDesc.SetShape(ge::GeShape(oriDims));
    tensorDesc.SetDataType(dtype);
    tensorDesc.SetFormat(FORMAT_ND);
    tensorDesc.SetOriginShape(ge::GeShape(oriDims));
    tensorDesc.SetOriginDataType(dtype);
    tensorDesc.SetOriginFormat(FORMAT_ND);
}

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
static Status AddNodeLinkOut(_InAnchor inAnchor, _OutAnchor outAnchor, const string& outNodeName) {
    // link out
    OP_LOGI("MultiHeadAttentionGrad", "Define %s link out begin", outNodeName.c_str());
    for (auto anchor : outAnchor->GetPeerInDataAnchors()) {
        GraphUtils::RemoveEdge(outAnchor, anchor);
        GraphUtils::AddEdge(inAnchor, anchor);
    }
    OP_LOGI("MultiHeadAttentionGrad", "Define %s link out end", outNodeName.c_str());
    return SUCCESS;
}

template<typename _InAnchor1, typename _InAnchor2>
static Status AddMatmulNode(ge::ComputeGraph& graph, const ge::GeTensorDesc& x1Desc, const ge::GeTensorDesc& x2Desc,
    const ge::GeTensorDesc& yDesc, ge::NodePtr& newNode, bool transposeX1,
    bool transposeX2, const string& nodeName, _InAnchor1 inAnchor1, _InAnchor2 inAnchor2)
{
    OP_LOGI("MultiHeadAttentionGrad", "Define %s begin", nodeName.c_str());
    OpDescPtr matmulOpDesc;
    FUSION_PASS_MAKE_SHARED((matmulOpDesc = std::make_shared<ge::OpDesc>(nodeName, "MatMulV2")),
        return INTERNAL_ERROR);
    matmulOpDesc->AddInputDesc("x1", x1Desc);
    matmulOpDesc->AddInputDesc("x2", x2Desc);
    AttrUtils::SetBool(matmulOpDesc, "transpose_x1", transposeX1);
    AttrUtils::SetBool(matmulOpDesc, "transpose_x2", transposeX2);
    matmulOpDesc->AddOutputDesc("y", yDesc);
    newNode = graph.AddNode(matmulOpDesc);
    GraphUtils::AddEdge(inAnchor1, newNode->GetInDataAnchor(0));
    GraphUtils::AddEdge(inAnchor2, newNode->GetInDataAnchor(1));
    OP_LOGI("MultiHeadAttentionGrad", "Define %s end", nodeName.c_str());
    return SUCCESS;
}

template<typename _InAnchor>
static Status AddReduceSumNode(ge::ComputeGraph& graph, const ge::GeTensorDesc& xDesc,
    const ge::GeTensorDesc& yDesc, ge::NodePtr& newNode, bool keepDims,
    const string& nodeName, _InAnchor inAnchor)
{
    OP_LOGI("MultiHeadAttentionGrad", "Define %s begin", nodeName.c_str());
    OpDescPtr reducesumOpDesc;
    FUSION_PASS_MAKE_SHARED((reducesumOpDesc = std::make_shared<ge::OpDesc>(nodeName, "ReduceSumD")),
        return INTERNAL_ERROR);
    reducesumOpDesc->AddInputDesc("x", xDesc);
    AttrUtils::SetListInt(reducesumOpDesc, "axes", {0});
    AttrUtils::SetBool(reducesumOpDesc, "keep_dims", keepDims);
    reducesumOpDesc->AddOutputDesc("y", yDesc);
    newNode = graph.AddNode(reducesumOpDesc);
    GraphUtils::AddEdge(inAnchor, newNode->GetInDataAnchor(0));
    OP_LOGI("MultiHeadAttentionGrad", "Define %s end", nodeName.c_str());
    return SUCCESS;
}

template<typename _InAnchor>
static Status AddTransposeNode(ge::ComputeGraph& graph, const ge::GeTensorDesc& xDesc, const ge::GeTensorDesc& yDesc,
    ge::NodePtr& newNode, const vector<int64_t>& perm, const vector<int64_t>& shape,
    bool transposeFirst, const string& nodeName, _InAnchor inAnchor)
{
    OP_LOGI("MultiHeadAttentionGrad", "Define %s begin", nodeName.c_str());
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
    OP_LOGI("MultiHeadAttentionGrad", "Define %s end", nodeName.c_str());
    return SUCCESS;
}

template<typename _InAnchor1, typename _InAnchor2>
static Status AddBatchMatmulNode(ge::ComputeGraph& graph, ge::OpDescPtr& opDesc, const ge::GeTensorDesc& x1Desc,
    const ge::GeTensorDesc& x2Desc,
    const ge::GeTensorDesc& yDesc, ge::NodePtr& newNode, bool adjX1, bool adjX2,
    const string& nodeName, _InAnchor1 inAnchor1, _InAnchor2 inAnchor2)
{
    OP_LOGI("MultiHeadAttentionGrad", "Define %s begin", nodeName.c_str());
    FUSION_PASS_MAKE_SHARED((opDesc = std::make_shared<ge::OpDesc>(nodeName, "BatchMatMul")), return INTERNAL_ERROR);
    opDesc->AddInputDesc("x1", x1Desc);
    opDesc->AddInputDesc("x2", x2Desc);
    AttrUtils::SetBool(opDesc, "adj_x1", adjX1);
    AttrUtils::SetBool(opDesc, "adj_x2", adjX2);
    opDesc->AddOutputDesc("y", yDesc);
    newNode = graph.AddNode(opDesc);
    GraphUtils::AddEdge(inAnchor1, newNode->GetInDataAnchor(0));
    GraphUtils::AddEdge(inAnchor2, newNode->GetInDataAnchor(1));
    OP_LOGI("MultiHeadAttentionGrad", "Define %s end", nodeName.c_str());
    return SUCCESS;
}

static Status AddConstNode(ge::ComputeGraph& graph, const ge::GeTensorDesc& yDesc, ge::NodePtr& newNode,
    const uint8_t* dataPtr, size_t size, const string& nodeName)
{
    OP_LOGI("MultiHeadAttentionGrad", "Define %s begin", nodeName.c_str());
    OpDescPtr constOpDesc;
    FUSION_PASS_MAKE_SHARED((constOpDesc = std::make_shared<ge::OpDesc>(nodeName, "Const")), return INTERNAL_ERROR);
    GeTensorPtr constValue = std::make_shared<ge::GeTensor>(yDesc, dataPtr, size);
    AttrUtils::SetTensor(constOpDesc, ATTR_NAME_WEIGHTS, constValue);
    constOpDesc->AddOutputDesc("y", yDesc);
    newNode = graph.AddNode(constOpDesc);
    OP_LOGI("MultiHeadAttentionGrad", "Define %s end", nodeName.c_str());
    return SUCCESS;
}

template<typename _InAnchor>
static Status AddCastNode(ge::ComputeGraph& graph, ge::OpDescPtr& opDesc, const ge::GeTensorDesc& xDesc,
    const ge::GeTensorDesc& yDesc, ge::NodePtr& newNode, int32_t dstType,
    const string& nodeName, _InAnchor inAnchor)
{
    OP_LOGI("MultiHeadAttentionGrad", "Define %s begin", nodeName.c_str());
    FUSION_PASS_MAKE_SHARED((opDesc = std::make_shared<ge::OpDesc>(nodeName, "Cast")), return INTERNAL_ERROR);
    opDesc->AddInputDesc("x", xDesc);
    AttrUtils::SetInt(opDesc, "dst_type", dstType);
    opDesc->AddOutputDesc("y", yDesc);
    newNode = graph.AddNode(opDesc);
    GraphUtils::AddEdge(inAnchor, newNode->GetInDataAnchor(0));
    OP_LOGI("MultiHeadAttentionGrad", "Define %s end", nodeName.c_str());
    return SUCCESS;
}

template<typename _InAnchor1, typename _InAnchor2>
static Status AddSoftmaxGradNode(ge::ComputeGraph& graph, ge::OpDescPtr& opDesc, const ge::GeTensorDesc& softmaxDesc,
    const ge::GeTensorDesc& gradSoftmaxDesc, const ge::GeTensorDesc& yDesc, ge::NodePtr& newNode,
    vector<int64_t> axes, const string& nodeName, _InAnchor1 inAnchor1, _InAnchor2 inAnchor2)
{
    OP_LOGI("MultiHeadAttentionGrad", "Define %s begin", nodeName.c_str());
    FUSION_PASS_MAKE_SHARED((opDesc = std::make_shared<ge::OpDesc>(nodeName, "SoftmaxGrad")), return INTERNAL_ERROR);
    opDesc->AddInputDesc("softmax", softmaxDesc);
    opDesc->AddInputDesc("grad_softmax", gradSoftmaxDesc);
    AttrUtils::SetListInt(opDesc, "axes", axes);
    opDesc->AddOutputDesc("y", yDesc);
    newNode = graph.AddNode(opDesc);
    GraphUtils::AddEdge(inAnchor1, newNode->GetInDataAnchor(0));
    GraphUtils::AddEdge(inAnchor2, newNode->GetInDataAnchor(1));
    OP_LOGI("MultiHeadAttentionGrad", "Define %s end", nodeName.c_str());
    return SUCCESS;
}

Status MultiHeadAttentionGradFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
    vector<ge::NodePtr>& fusionNodes)
{
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Define MultiHeadAttentionGradFusionPass fusion begin");
    ge::NodePtr multiHeadAttentionGradNode = GetNodeFromMapping(FUSED_OP_TYPE, mapping);
    FUSION_PASS_CHECK(multiHeadAttentionGradNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(),
        "MultiHeadAttentionGrad node is null, fusion failed."),
                        return PARAM_INVALID);
    ge::OpDescPtr multiHeadAttentionGradDesc = multiHeadAttentionGradNode->GetOpDesc();
    FUSION_PASS_CHECK(multiHeadAttentionGradDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(),
        "MultiHeadAttentionGrad's Op_desc is null, fusion failed."),
                        return PARAM_INVALID);
    // shape
    vector<int64_t> query_shape = multiHeadAttentionGradDesc->GetInputDesc("query").GetShape().GetDims();
    FUSION_PASS_CHECK(query_shape.size() !=2, OP_LOGE(FUSED_OP_TYPE.c_str(),
        "MultiHeadAttentionGrad's Query origin shape should be 2D, fusion failed."),
                        return PARAM_INVALID);
    int64_t attnHeadNum;
    int64_t attnDimPerHead;
    int64_t srcLen;
    int64_t tgtLen;
    float keepProb;
    bool softmaxUseFloat;
    vector<bool> bias_grad_mask;
    AttrUtils::GetInt(multiHeadAttentionGradDesc, "attn_head_num", attnHeadNum);
    AttrUtils::GetInt(multiHeadAttentionGradDesc, "attn_dim_per_head", attnDimPerHead);
    AttrUtils::GetInt(multiHeadAttentionGradDesc, "src_len", srcLen);
    AttrUtils::GetInt(multiHeadAttentionGradDesc, "tgt_len", tgtLen);
    AttrUtils::GetFloat(multiHeadAttentionGradDesc, "keep_prob", keepProb);
    AttrUtils::GetBool(multiHeadAttentionGradDesc, "softmax_use_float", softmaxUseFloat);
    AttrUtils::GetListBool(multiHeadAttentionGradDesc, "bias_grad_mask", bias_grad_mask);
    FUSION_PASS_CHECK((attnHeadNum == 0 || attnDimPerHead == 0 || srcLen == 0 || tgtLen== 0),
        OP_LOGE(FUSED_OP_TYPE.c_str(),
        "MultiHeadAttention's attn_head_num, attn_dim_per_head, src_len, tgt_len should not be 0, fusion failed."),
                    return PARAM_INVALID);
    FUSION_PASS_CHECK(!(
        attnHeadNum % ALIGNMENT == 0 && attnDimPerHead % ALIGNMENT ==  0 &&
        srcLen % ALIGNMENT ==  0 && tgtLen % ALIGNMENT == 0),
        OP_LOGE(FUSED_OP_TYPE.c_str(),
        "MultiHeadAttention's attn_head_num, attn_dim_per_head, src_len, tgt_len should align of 16, fusion failed."),
                    return PARAM_INVALID);
    const int64_t batch = query_shape[0] / tgtLen;
    const int64_t weightCol = attnHeadNum * attnDimPerHead;
    const float scale = 1.0 / sqrt(attnDimPerHead);

    const vector<int64_t> perm({0, 2, 1, 3});
    const vector<int64_t> out_proj_input_matmul_shape({batch * tgtLen, weightCol});
    const vector<int64_t> out_proj_weight_matmul_shape({weightCol, weightCol});
    const vector<int64_t> bias_reducesum_shape({1, weightCol});
    const vector<int64_t> context_trans_shape({batch, tgtLen, attnHeadNum, attnDimPerHead});
    const vector<int64_t> new_context_trans_shape({batch, attnHeadNum, tgtLen, attnDimPerHead});
    const vector<int64_t> attn_res_batch_shape({batch, attnHeadNum, tgtLen, srcLen});
    const vector<int64_t> kv_res_batch_shape({batch, attnHeadNum, srcLen, attnDimPerHead});
    const vector<int64_t> query_res_batch_shape({batch, attnHeadNum, tgtLen, attnDimPerHead});
    const vector<int64_t> kv_trans_batch_shape({batch * srcLen, weightCol});
    const vector<int64_t> query_trans_batch_shape({batch * tgtLen, weightCol});
    const vector<int64_t> query_matmul_shape({batch * tgtLen, weightCol});
    const vector<int64_t> query_weight_matmul_shape({weightCol, weightCol});
    const vector<int64_t> kv_matmul_shape({batch * srcLen, weightCol});
    const vector<int64_t> kv_weight_matmul_shape({weightCol, weightCol});

    // out_proj_input_matmul
    GeTensorDesc outProjInputMatmulOutputDesc;
    SetNZTensorDesc(outProjInputMatmulOutputDesc, out_proj_input_matmul_shape);
    NodePtr outProjInputMatmulNode;
    AddMatmulNode(graph, multiHeadAttentionGradDesc->GetInputDesc("y_grad"),
        multiHeadAttentionGradDesc->GetInputDesc("out_proj_weight"),
        outProjInputMatmulOutputDesc, outProjInputMatmulNode, false, false,
        multiHeadAttentionGradNode->GetName() + "_out_proj_input_matmul",
        multiHeadAttentionGradNode->GetInDataAnchor(INPUT_Y_GRAD)->GetPeerOutAnchor(),
        multiHeadAttentionGradNode->GetInDataAnchor(INPUT_OUT_PROJ_WEIGHT)->GetPeerOutAnchor()
    );

    // out_proj_weight_matmul
    GeTensorDesc outProjWeightMatmulOutputDesc;
    SetNZTensorDesc(outProjWeightMatmulOutputDesc, out_proj_weight_matmul_shape);
    NodePtr outProjWeightMatmulNode;
    AddMatmulNode(graph, multiHeadAttentionGradDesc->GetInputDesc("y_grad"),
        multiHeadAttentionGradDesc->GetInputDesc("context"),
        outProjWeightMatmulOutputDesc, outProjWeightMatmulNode, true, false,
        multiHeadAttentionGradNode->GetName() + "_out_proj_weight_matmul",
        multiHeadAttentionGradNode->GetInDataAnchor(INPUT_Y_GRAD)->GetPeerOutAnchor(),
        multiHeadAttentionGradNode->GetInDataAnchor(INPUT_CONTEXT)->GetPeerOutAnchor()
    );
    AddNodeLinkOut(outProjWeightMatmulNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetOutDataAnchor(OUTPUT_OUT_PROJ_WEIGHT_GRAD),
        multiHeadAttentionGradNode->GetName() + "_out_proj_weight_matmul");

    // bias_empty
    GeTensorDesc biasEmptyTensorDesc = GeTensorDesc(GeShape(), FORMAT_ND, DT_FLOAT16);
    NodePtr biasEmptyNode;
    AddConstNode(graph, biasEmptyTensorDesc, biasEmptyNode, nullptr, 0,
        multiHeadAttentionGradNode->GetName() + "_bias_empty");

    // out_proj_bias
    NodePtr outProjBiasNode;
    GeTensorDesc biasReducesumTensorDesc;
    SetNDTensorDesc(biasReducesumTensorDesc, bias_reducesum_shape, DT_FLOAT16);
    int theThirdElement = 3;
    if (bias_grad_mask[theThirdElement]) {
        AddReduceSumNode(graph,
            multiHeadAttentionGradDesc->GetInputDesc("y_grad"), biasReducesumTensorDesc, outProjBiasNode,
            true, multiHeadAttentionGradNode->GetName() + "_out_proj_bias",
            multiHeadAttentionGradNode->GetInDataAnchor(INPUT_Y_GRAD)->GetPeerOutAnchor());
    } else {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Define out_proj_bias empty begin");
        outProjBiasNode = biasEmptyNode;
    }
    AddNodeLinkOut(outProjBiasNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetOutDataAnchor(OUTPUT_OUT_PROJ_BIAS_GRAD),
        multiHeadAttentionGradNode->GetName() + "_out_proj_bias");

    // context_trans
    GeTensorDesc contextTransOutputDesc;
    SetNZTensorDesc(contextTransOutputDesc, new_context_trans_shape);
    NodePtr contextTransNode;
    AddTransposeNode(graph, outProjInputMatmulOutputDesc, contextTransOutputDesc,
        contextTransNode, perm, context_trans_shape,
        false, multiHeadAttentionGradNode->GetName() + "_context_trans",
        outProjInputMatmulNode->GetOutDataAnchor(0)
    );

    // attn_res_batch
    GeTensorDesc attnResBatchOutputDesc;
    SetNZTensorDesc(attnResBatchOutputDesc, attn_res_batch_shape);
    OpDescPtr attnResBatchOpDesc;
    NodePtr attnResBatchNode;
    AddBatchMatmulNode(graph, attnResBatchOpDesc, contextTransOutputDesc,
        multiHeadAttentionGradDesc->GetInputDesc("value_res"), attnResBatchOutputDesc,
        attnResBatchNode, false, true,
        multiHeadAttentionGradNode->GetName() + "_attn_res_batch",
        contextTransNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetInDataAnchor(INPUT_VALUE_RES)->GetPeerOutAnchor()
    );

    // value_res_batch
    GeTensorDesc valueResBatchOutputDesc;
    SetNZTensorDesc(valueResBatchOutputDesc, kv_res_batch_shape);
    OpDescPtr valueResBatchOpDesc;
    NodePtr valueResBatchNode;
    AddBatchMatmulNode(graph, valueResBatchOpDesc,
        multiHeadAttentionGradDesc->GetInputDesc("attn_res"),
        contextTransOutputDesc, valueResBatchOutputDesc,
        valueResBatchNode, true, false,
        multiHeadAttentionGradNode->GetName() + "_value_res_batch",
        multiHeadAttentionGradNode->GetInDataAnchor(INPUT_ATTN_RES)->GetPeerOutAnchor(),
        contextTransNode->GetOutDataAnchor(0)
    );

    // attn_res_dropout
    OpDescPtr attnResDropoutOpDesc;
    NodePtr attnResDropoutNode;
    if (keepProb < 1.0) {
        GeTensorDesc probTensorDesc = GeTensorDesc(GeShape(), FORMAT_ND, DT_FLOAT);
        NodePtr probNode;
        AddConstNode(graph, probTensorDesc, probNode, reinterpret_cast<uint8_t*>(&keepProb), sizeof(float),
            "keep_prob");
        // dropout_do_mask
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Define attn_res_dropout begin");
        FUSION_PASS_MAKE_SHARED((attnResDropoutOpDesc = std::make_shared<ge::OpDesc>(
            multiHeadAttentionGradNode->GetName() + "_dropout_do_mask", "DropOutDoMask")), return INTERNAL_ERROR);
        attnResDropoutOpDesc->AddInputDesc("x", attnResBatchOutputDesc);
        attnResDropoutOpDesc->AddInputDesc("mask", multiHeadAttentionGradDesc->GetInputDesc("dropout_mask"));
        attnResDropoutOpDesc->AddInputDesc("keep_prob", probTensorDesc);
        attnResDropoutOpDesc->AddOutputDesc("y", attnResBatchOutputDesc);
        attnResDropoutNode = graph.AddNode(attnResDropoutOpDesc);
        GraphUtils::AddEdge(attnResBatchNode->GetOutDataAnchor(0), attnResDropoutNode->GetInDataAnchor(0));
        GraphUtils::AddEdge(multiHeadAttentionGradNode->GetInDataAnchor(INPUT_DROPOUT_MASK)->GetPeerOutAnchor(),
            attnResDropoutNode->GetInDataAnchor(1));
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
    if (softmaxUseFloat) {
        OpDescPtr castOpDesc, beforeCastOpDesc;
        // cast_before_softmax
        GeTensorDesc castOutputDesc;
        SetNZTensorDesc(castOutputDesc, attn_res_batch_shape, DT_FLOAT);
        NodePtr castNode;
        AddCastNode(graph, castOpDesc, attnResBatchOutputDesc, castOutputDesc, castNode, DT_FLOAT,
            multiHeadAttentionGradNode->GetName() + "_cast_before_softmax",
            attnResDropoutNode->GetOutDataAnchor(0));
        // attn_weight_softmax
        NodePtr beforeCastNode;
        AddSoftmaxGradNode(graph, beforeCastOpDesc,
            multiHeadAttentionGradDesc->GetInputDesc("attn_scores"), castOutputDesc,
            castOutputDesc, beforeCastNode, {-1},
            multiHeadAttentionGradNode->GetName() + "_attn_weight_softmax",
            multiHeadAttentionGradNode->GetInDataAnchor(INPUT_ATTN_SCORES)->GetPeerOutAnchor(),
            castNode->GetOutDataAnchor(0)
        );
        // cast_after_softmax
        AddCastNode(graph, softmaxGradOpDesc, castOutputDesc, softmaxGradOutputDesc, softmaxGradNode, DT_FLOAT16,
            multiHeadAttentionGradNode->GetName() + "_cast_after_softmax",
            beforeCastNode->GetOutDataAnchor(0));
    } else {
        // softmax
        AddSoftmaxGradNode(graph, softmaxGradOpDesc,
            multiHeadAttentionGradDesc->GetInputDesc("attn_scores"),
            attnResBatchOutputDesc,
            softmaxGradOutputDesc, softmaxGradNode, {-1},
            multiHeadAttentionGradNode->GetName() + "_attn_weight_softmax",
            multiHeadAttentionGradNode->GetInDataAnchor(INPUT_ATTN_SCORES)->GetPeerOutAnchor(),
            attnResDropoutNode->GetOutDataAnchor(0)
        );
    }

    // query_res_batch
    GeTensorDesc queryResBatchOutputDesc;
    SetNZTensorDesc(queryResBatchOutputDesc, query_res_batch_shape);
    OpDescPtr queryResBatchOpDesc;
    NodePtr queryResBatchNode;
    AddBatchMatmulNode(graph, queryResBatchOpDesc, softmaxGradOutputDesc,
        multiHeadAttentionGradDesc->GetInputDesc("key_res"), queryResBatchOutputDesc,
        queryResBatchNode, false, false,
        multiHeadAttentionGradNode->GetName() + "_query_res_batch",
        softmaxGradNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetInDataAnchor(INPUT_KEY_RES)->GetPeerOutAnchor()
    );

    // key_res_batch
    GeTensorDesc keyResBatchOutputDesc;
    SetNZTensorDesc(keyResBatchOutputDesc, kv_res_batch_shape);
    OpDescPtr keyResBatchOpDesc;
    NodePtr keyResBatchNode;
    AddBatchMatmulNode(graph, keyResBatchOpDesc, softmaxGradOutputDesc,
        multiHeadAttentionGradDesc->GetInputDesc("query_res"), keyResBatchOutputDesc,
        keyResBatchNode, true, false,
        multiHeadAttentionGradNode->GetName() + "_key_res_batch",
        softmaxGradNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetInDataAnchor(INPUT_QUERY_RES)->GetPeerOutAnchor()
    );

    // query_trans
    GeTensorDesc queryTransOutputDesc;
    SetNZTensorDesc(queryTransOutputDesc, query_trans_batch_shape);
    NodePtr queryTransNode;
    AddTransposeNode(graph, queryResBatchOutputDesc, queryTransOutputDesc, queryTransNode, perm,
        query_trans_batch_shape, true,
        multiHeadAttentionGradNode->GetName() + "_query_trans",
        queryResBatchNode->GetOutDataAnchor(0)
    );

    // attn_scores_muls
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Define attn_scores_muls begin");
    OpDescPtr attnScoresMulsOpDesc;
    FUSION_PASS_MAKE_SHARED((attnScoresMulsOpDesc =
        std::make_shared<ge::OpDesc>(multiHeadAttentionGradNode->GetName() + "_attn_scores_muls", "Muls")),
        return INTERNAL_ERROR);
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
        true, multiHeadAttentionGradNode->GetName() + "_key_trans", keyResBatchNode->GetOutDataAnchor(0)
    );

    // value_trans
    GeTensorDesc valueTransOutputDesc;
    SetNZTensorDesc(valueTransOutputDesc, kv_trans_batch_shape);
    NodePtr valueTransNode;
    AddTransposeNode(graph, valueResBatchOutputDesc, valueTransOutputDesc, valueTransNode, perm, kv_trans_batch_shape,
        true, multiHeadAttentionGradNode->GetName() + "_value_trans", valueResBatchNode->GetOutDataAnchor(0)
    );

    // query_matmul
    GeTensorDesc queryMatmulOutputDesc;
    SetNZTensorDesc(queryMatmulOutputDesc, query_matmul_shape);
    NodePtr queryMatmulNode;
    AddMatmulNode(graph, queryTransOutputDesc, multiHeadAttentionGradDesc->GetInputDesc("query_weight"),
        queryMatmulOutputDesc, queryMatmulNode, false, false, multiHeadAttentionGradNode->GetName() + "_query_matmul",
        attnScoresMulsNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetInDataAnchor(INPUT_QUERY_WEIGHT)->GetPeerOutAnchor()
    );
    AddNodeLinkOut(queryMatmulNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetOutDataAnchor(OUTPUT_QUERY_GRAD),
        multiHeadAttentionGradNode->GetName() + "_query_matmul");

    // query_weight_matmul
    GeTensorDesc queryWeightMatmulOutputDesc;
    SetNZTensorDesc(queryWeightMatmulOutputDesc, query_weight_matmul_shape);
    NodePtr queryWeightMatmulNode;
    AddMatmulNode(graph, queryTransOutputDesc, multiHeadAttentionGradDesc->GetInputDesc("query"),
        queryWeightMatmulOutputDesc, queryWeightMatmulNode, true, false,
        multiHeadAttentionGradNode->GetName() + "_query_weight_matmul",
        attnScoresMulsNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetInDataAnchor(INPUT_QUERY)->GetPeerOutAnchor()
    );
    AddNodeLinkOut(queryWeightMatmulNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetOutDataAnchor(OUTPUT_QUERY_WEIGHT_GRAD),
        multiHeadAttentionGradNode->GetName() + "_query_weight_matmul");

    // query_bias
    NodePtr queryBiasNode;
    if (bias_grad_mask[0]) {
        AddReduceSumNode(graph, queryTransOutputDesc, biasReducesumTensorDesc, queryBiasNode,
            true, multiHeadAttentionGradNode->GetName() + "_query_bias",
            attnScoresMulsNode->GetOutDataAnchor(0));
    } else {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Define query_bias empty begin");
        queryBiasNode = biasEmptyNode;
    }
    AddNodeLinkOut(queryBiasNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetOutDataAnchor(OUTPUT_QUERY_BIAS_GRAD),
        multiHeadAttentionGradNode->GetName() + "_query_bias");

    // key_matmul
    GeTensorDesc keyMatmulOutputDesc;
    SetNZTensorDesc(keyMatmulOutputDesc, kv_matmul_shape);
    NodePtr keyMatmulNode;
    AddMatmulNode(graph, keyTransOutputDesc, multiHeadAttentionGradDesc->GetInputDesc("key_weight"),
        keyMatmulOutputDesc, keyMatmulNode, false, false, multiHeadAttentionGradNode->GetName() + "_key_matmul",
        keyTransNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetInDataAnchor(INPUT_KEY_WEIGHT)->GetPeerOutAnchor()
    );
    AddNodeLinkOut(keyMatmulNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetOutDataAnchor(OUTPUT_KEY_GRAD),
        multiHeadAttentionGradNode->GetName() + "_key_matmul");

    // key_weight_matmul
    GeTensorDesc keyWeightMatmulOutputDesc;
    SetNZTensorDesc(keyWeightMatmulOutputDesc, kv_weight_matmul_shape);
    NodePtr keyWeightMatmulNode;
    AddMatmulNode(graph, keyTransOutputDesc, multiHeadAttentionGradDesc->GetInputDesc("key"),
        keyWeightMatmulOutputDesc, keyWeightMatmulNode, true, false,
        multiHeadAttentionGradNode->GetName() + "_key_weight_matmul",
        keyTransNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetInDataAnchor(INPUT_KEY)->GetPeerOutAnchor()
    );
    AddNodeLinkOut(keyWeightMatmulNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetOutDataAnchor(OUTPUT_KEY_WEIGHT_GRAD),
        multiHeadAttentionGradNode->GetName() + "_key_weight_matmul");

    // key_bias
    NodePtr keyBiasNode;
    if (bias_grad_mask[1]) {
        AddReduceSumNode(graph, keyTransOutputDesc, biasReducesumTensorDesc, keyBiasNode,
            true, multiHeadAttentionGradNode->GetName() + "_key_bias", keyTransNode->GetOutDataAnchor(0));
    } else {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Define key_bias empty begin");
        keyBiasNode = biasEmptyNode;
    }
    AddNodeLinkOut(keyBiasNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetOutDataAnchor(OUTPUT_KEY_BIAS_GRAD),
        multiHeadAttentionGradNode->GetName() + "_key_bias");

    // value_matmul
    GeTensorDesc valueMatmulOutputDesc;
    SetNZTensorDesc(valueMatmulOutputDesc, kv_matmul_shape);
    NodePtr valueMatmulNode;
    AddMatmulNode(graph, valueTransOutputDesc, multiHeadAttentionGradDesc->GetInputDesc("value_weight"),
        valueMatmulOutputDesc, valueMatmulNode, false, false, multiHeadAttentionGradNode->GetName() + "_value_matmul",
        valueTransNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetInDataAnchor(INPUT_VALUE_WEIGHT)->GetPeerOutAnchor()
    );
    AddNodeLinkOut(valueMatmulNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetOutDataAnchor(OUTPUT_VALUE_GRAD),
        multiHeadAttentionGradNode->GetName() + "_value_matmul");

    // value_weight_matmul
    GeTensorDesc valueWeightMatmulOutputDesc;
    SetNZTensorDesc(valueWeightMatmulOutputDesc, kv_weight_matmul_shape);
    NodePtr valueWeightMatmulNode;
    AddMatmulNode(graph, valueTransOutputDesc, multiHeadAttentionGradDesc->GetInputDesc("value"),
        valueWeightMatmulOutputDesc, valueWeightMatmulNode, true, false,
        multiHeadAttentionGradNode->GetName() + "_value_weight_matmul",
        valueTransNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetInDataAnchor(INPUT_VALUE)->GetPeerOutAnchor()
    );
    AddNodeLinkOut(valueWeightMatmulNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetOutDataAnchor(OUTPUT_VALUE_WEIGHT_GRAD),
        multiHeadAttentionGradNode->GetName() + "_value_weight_matmul");

    // value_bias
    NodePtr valueBiasNode;
    if (bias_grad_mask[2]) {
        AddReduceSumNode(graph, valueTransOutputDesc, biasReducesumTensorDesc, valueBiasNode,
            true, multiHeadAttentionGradNode->GetName() + "_value_bias", valueTransNode->GetOutDataAnchor(0));
    } else {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Define value_bias empty begin");
        valueBiasNode = biasEmptyNode;
    }
    AddNodeLinkOut(valueBiasNode->GetOutDataAnchor(0),
        multiHeadAttentionGradNode->GetOutDataAnchor(OUTPUT_VALUE_BIAS_GRAD),
        multiHeadAttentionGradNode->GetName() + "_value_bias");

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
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove multiHeadAttentionGradNode failed"),
        return FAILED);


    return SUCCESS;
}

REGISTER_PASS("MultiHeadAttentionGradFusionPass", BUILT_IN_GRAPH_PASS, MultiHeadAttentionGradFusionPass);
} // namespace