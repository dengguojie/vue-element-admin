/**
* Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
*
* @brief DynamicLSTM fusion pass(LSTM --> DynamicLSTM & FullyConnection)
*
*/

#include "dynamic_lstm_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace ge;
namespace fe {
static const char *FUSED_NODE = "LSTM";
static const std::string PATTERN_FUSEDNODE = "LSTM";
static const int64_t nodeOutputSize = 3;
static const int64_t ctOutputSize = 2;
static const int64_t dimTwo = 2;
static const int64_t dimThree = 3;
static const int64_t input0Size = 4;
static const int64_t fzDim = 16;
vector<FusionPattern *> DynamicLSTMFusionPass::DefinePatterns()
{
    vector<FusionPattern *> patterns;

    FusionPattern *pattern = new (std::nothrow) FusionPattern("DynamicLSTMV2FusionPass");
    FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
        "lstm pattern object failed."), return patterns);

    pattern->AddOpDesc(PATTERN_FUSEDNODE, { FUSED_NODE }).SetOutput(PATTERN_FUSEDNODE);

    patterns.push_back(pattern);
    return patterns;
}

Status DynamicLSTMFusionPass::ProcessLSTMStatic(ge::NodePtr fusedNode, ge::NodePtr &innerproductNode,
                                                ge::ComputeGraph &graph,
                                                vector<ge::NodePtr> &newNodes, const InputIndexInfo &inputIndexInfo)
{
    ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
    ge::GeTensorDesc inputTensorDesc = fusedDesc->GetInputDesc(inputIndexInfo.xStaticIndex);
    DataType dataType = inputTensorDesc.GetDataType();

    // create the OpDescPtr for InnerProduct
    ge::OpDescPtr innerProductStaticDesc = nullptr;
    FUSION_PASS_MAKE_SHARED((innerProductStaticDesc =
       std::make_shared<ge::OpDesc>(string("FullyConnection/") + fusedDesc->GetName(),
                                    "FullyConnection")), return INTERNAL_ERROR);

    ge::GeShape outputShape = inputTensorDesc.GetShape();
    std::vector<int64_t> dimsInputXShape;
    dimsInputXShape.push_back(inputTensorDesc.GetShape().GetDim(0));
    dimsInputXShape.push_back(inputTensorDesc.GetShape().GetDim(1));

    ge::GeShape inputXShape(dimsInputXShape);
    inputTensorDesc.SetShape(inputXShape);
    innerProductStaticDesc->AddInputDesc("x", inputTensorDesc);

    ge::InDataAnchorPtr inputWAnchorPtr0 = fusedNode->GetInDataAnchor(inputIndexInfo.wxStaticIndex);
    ge::OutDataAnchorPtr constAnchorPtr0 = inputWAnchorPtr0->GetPeerOutAnchor();
    ge::NodePtr inputWNode = constAnchorPtr0->GetOwnerNode();
    Operator constWxStaticOp = OpDescUtils::CreateOperatorFromNode(inputWNode);
    bool constAdjustFlag = false;
    vector<ge::GeTensorPtr> weights = ge::OpDescUtils::MutableWeights(inputWNode);
    if (weights.empty()) {
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "LSTM weights is null, fusion failed.");
        return PARAM_INVALID;
    }
    ge::GeTensorPtr inputWConstGeTensor = weights[0];
    ge::GeTensorDesc inputWTensorDesc = inputWConstGeTensor->GetTensorDesc();
    constWxStaticOp.GetAttr("const_adjust_flag", constAdjustFlag);
    if (!constAdjustFlag) {
        int32_t c0 = 16;
        int32_t wRow = inputWTensorDesc.GetShape().GetDim(1);
        int32_t wCol = inputWTensorDesc.GetShape().GetDim(0);
        int32_t destWRow = (wRow + fzDim - 1) / fzDim * fzDim;

        // there why
        int32_t destWCol = input0Size * ((wCol / input0Size + c0 - 1) / c0 * c0);
        std::vector<int64_t> dimsInputWDim;

        // no need padding
        dimsInputWDim.push_back(destWCol);
        dimsInputWDim.push_back(destWRow);
        dimsInputWDim.push_back(1);
        dimsInputWDim.push_back(1);

        std::vector<int64_t> dimsOriInputWDim;
        // no need padding
        dimsOriInputWDim.push_back(wCol);
        dimsOriInputWDim.push_back(wRow);
        dimsOriInputWDim.push_back(1);
        dimsOriInputWDim.push_back(1);

        ge::GeShape dimsInputWShape(dimsInputWDim);
        ge::GeShape dimsOriInputWShape(dimsOriInputWDim);

        inputWTensorDesc.SetShape(dimsInputWShape);
        inputWTensorDesc.SetOriginShape(dimsOriInputWShape);
        inputWTensorDesc.SetFormat(ge::FORMAT_NCHW);
        inputWTensorDesc.SetOriginFormat(ge::FORMAT_NCHW);
        fusedNode->GetInDataAnchor(inputIndexInfo.wxStaticIndex)
            ->GetPeerOutAnchor()
            ->GetOwnerNode()
            ->GetOpDesc()
            ->UpdateOutputDesc(0, inputWTensorDesc);
        constWxStaticOp.SetAttr("const_adjust_flag", true);
    }
    innerProductStaticDesc->AddInputDesc("w", inputWTensorDesc);
    inputWConstGeTensor->SetTensorDesc(inputWTensorDesc);

    // output todo shape product output
    ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_NCHW, dataType);
    std::vector<int64_t> dimsY;

    dimsY.push_back(inputTensorDesc.GetShape().GetDim(0));
    dimsY.push_back(inputWTensorDesc.GetShape().GetDim(0));

    ge::GeShape dimsYShape(dimsY);
    outputTensorDesc.SetShape(dimsYShape);
    outputTensorDesc.SetOriginShape(dimsYShape);
    outputTensorDesc.SetFormat(ge::FORMAT_NCHW);
    outputTensorDesc.SetOriginFormat(ge::FORMAT_NCHW);
    innerProductStaticDesc->AddOutputDesc("y", outputTensorDesc);
    int32_t num_output = 0;
    ge::AttrUtils::GetInt(fusedDesc, "num_output", num_output);
    ge::AttrUtils::SetInt(innerProductStaticDesc, "num_output", num_output);
    ge::AttrUtils::SetBool(innerProductStaticDesc, "transpose", false);
    ge::AttrUtils::SetBool(innerProductStaticDesc, "bias_term", false);
    ge::AttrUtils::SetInt(innerProductStaticDesc, "axis", 1);

    // add the sub operators to the graph
    innerproductNode = graph.AddNode(innerProductStaticDesc);
    FUSION_PASS_CHECK(innerproductNode == nullptr,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                                       innerProductStaticDesc->GetName().c_str()), return FAILED);
    newNodes.push_back(innerproductNode);
    return SUCCESS;
}

ge::GeTensorPtr DynamicLSTMFusionPass::ProcessLSTMWxh(ge::NodePtr fusedNode, bool &failStatus,
                                                      const InputIndexInfo &inputIndexInfo)
{
    OP_LOGI(FUSED_OP_TYPE.c_str(), "has enter process lstm wxh");
    ge::InDataAnchorPtr inputWxAnchorPtr0 = fusedNode->GetInDataAnchor(inputIndexInfo.wxIndex);
    ge::OutDataAnchorPtr constWxAnchorPtr0 = inputWxAnchorPtr0->GetPeerOutAnchor();
    ge::NodePtr inputWxNode = constWxAnchorPtr0->GetOwnerNode();
    bool isExistSubGraph = false;
    FUSION_PASS_CHECK(inputWxNode == nullptr,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "weights Wx node is null."),
                      return nullptr);
    if (inputWxNode->GetType() == "Data") {
        ge::NodePtr parentNode = NodeUtils::GetParentInput(inputWxNode);
        FUSION_PASS_CHECK((parentNode == nullptr) || (parentNode->GetType() != "Const"),
                          OP_LOGE(FUSED_OP_TYPE.c_str(), "weights Wx get parent node failed."),
                          return nullptr);
        isExistSubGraph = true;
        inputWxNode = parentNode;
    }
    Operator constWxOp = OpDescUtils::CreateOperatorFromNode(inputWxNode);
    bool constAdjustFlag = false;
    vector<ge::GeTensorPtr> weightsWx = ge::OpDescUtils::MutableWeights(inputWxNode);
    FUSION_PASS_CHECK(weightsWx.empty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "LSTM weights Wx is null, fusion failed"),
                       return nullptr);
    ge::GeTensorPtr wxTensorPtr = weightsWx[0];
    constWxOp.GetAttr("const_adjust_flag", constAdjustFlag);
    if (constAdjustFlag) {
        // Data exist different subgraph, need update data output desc
        if (isExistSubGraph) {
            ge::GeTensorDesc wxhTensorDesc = wxTensorPtr->GetTensorDesc();
            fusedNode->GetInDataAnchor(inputIndexInfo.wxIndex)
                ->GetPeerOutAnchor()
                ->GetOwnerNode()
                ->GetOpDesc()
                ->UpdateOutputDesc(0, wxhTensorDesc);
        }
      OP_LOGD(FUSED_OP_TYPE.c_str(), "dynamic LSTM const_adjust_flag is true, no need adjust again.");
      return wxTensorPtr;
    }
    ge::InDataAnchorPtr inputWhAnchorPtr0 = fusedNode->GetInDataAnchor(inputIndexInfo.whIndex);
    ge::OutDataAnchorPtr constWhAnchorPtr0 = inputWhAnchorPtr0->GetPeerOutAnchor();
    ge::NodePtr inputWhNode = constWhAnchorPtr0->GetOwnerNode();
    vector<ge::GeTensorPtr> weightsWh = ge::OpDescUtils::MutableWeights(inputWhNode);
    if (weightsWh.empty()) {
        failStatus = true;
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "LSTM weightsWh is null, fusion failed.");
        return nullptr;
    }
    ge::GeTensorPtr whTensorPtr = weightsWh[0];

    ge::GeTensorDesc wxConstTensorDesc = wxTensorPtr->GetTensorDesc();
    ge::GeTensorDesc whConstTensorDesc = whTensorPtr->GetTensorDesc();

    ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
    DataType dataType = fusedDesc->GetInputDesc(inputIndexInfo.c0Index).GetDataType();
    int32_t wxRow = wxConstTensorDesc.GetShape().GetDim(0);
    int32_t wxCol = wxConstTensorDesc.GetShape().GetDim(1);
    int32_t whRow = whConstTensorDesc.GetShape().GetDim(0);
    int32_t whCol = whConstTensorDesc.GetShape().GetDim(1);
    FUSION_PASS_CHECK(wxCol == 0, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                 "wxCol can not 0"), return nullptr);
    FUSION_PASS_CHECK(whCol == 0, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                 "whCol can not 0"), return nullptr);

    // wxRow == whRow
    std::vector<int64_t> dimsIn;
    int32_t targetCol = wxCol + whCol;
    dimsIn.push_back(targetCol);
    dimsIn.push_back(wxRow);

    ge::GeShape wxhShape(dimsIn);
    ge::GeTensorDesc wxhTensorDesc(wxhShape, ge::FORMAT_ND, dataType);
    wxhTensorDesc.SetOriginShape(wxhShape);
    wxhTensorDesc.SetOriginFormat(ge::FORMAT_ND);

    fusedNode->GetInDataAnchor(inputIndexInfo.wxIndex)
        ->GetPeerOutAnchor()
        ->GetOwnerNode()
        ->GetOpDesc()
        ->UpdateOutputDesc(0, wxhTensorDesc);
    wxTensorPtr->SetTensorDesc(wxhTensorDesc);
    ge::GeTensorPtr weightTensor = nullptr;
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_FLOAT) {
        // the wx + wh matrix
        unique_ptr<float[]> wxhMergeData(new (std::nothrow) float[targetCol * wxRow]());
        FUSION_PASS_CHECK(wxhMergeData.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                                        "wxhMergeData is NULL"),
                          return nullptr);
        FUSION_PASS_CHECK(wxTensorPtr->GetData().data() == nullptr,
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                          "wxTensorPtr->GetData().data() is NULL"), return nullptr);
        FUSION_PASS_CHECK(whTensorPtr->GetData().data() == nullptr,
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                          "whTensorPtr->GetData().data() is NULL"), return nullptr);
        float *wxData = (float *)wxTensorPtr->GetData().data();
        float *whData = (float *)whTensorPtr->GetData().data();

        auto retMem = memset_s(wxhMergeData.get(), targetCol * wxRow, 0, targetCol * wxRow);
        FUSION_PASS_CHECK(retMem != EOK, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
        "Failed to operate memset_s function!"), return nullptr);

        // wx transpose, assign to merge data
        float *dstWeight = wxhMergeData.get();
        for (int32_t i = 0; i < wxRow * wxCol; ++i) {
            *(dstWeight + i / wxCol + wxRow * (i % wxCol)) = *(wxData + i);
        }

        // wh transpose, assign to merge data
        for (int32_t i = 0; i < whRow * whCol; ++i) {
            *(dstWeight + wxRow * wxCol + i / whCol + whRow * (i % whCol)) = *(whData + i);
        }
        FUSION_PASS_MAKE_SHARED(
            (weightTensor = std::make_shared<GeTensor>(wxhTensorDesc, reinterpret_cast<uint8_t*>(wxhMergeData.get()),
                                                       targetCol * wxRow * sizeof(float))),
            weightTensor = nullptr;
            return weightTensor);
        ge::AttrUtils::SetTensor(inputWxNode->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, weightTensor);
        constWxOp.SetAttr("const_adjust_flag", true);
    } else {
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                       "Node:%s's dtype is not in (float16, float32), fusion failed.",
                                       fusedDesc->GetName().c_str());
        failStatus = true;
    }
    return weightTensor;
}

Status DynamicLSTMFusionPass::AddDynamicLSTMNode(ge::OpDescPtr &thisOpDesc, const ge::OpDescPtr &formerOpDesc,
                                                 const ge::GeTensorDesc &wxhTensorDesc,
                                                 const InputIndexInfo &inputIndexInfo, bool expose_hidden,
                                                 ge::GeTensorDesc &staticTensorDesc,
                                                 int32_t outputSize)
{
    OP_LOGI(FUSED_OP_TYPE.c_str(),"Enter add DynamicLSTM node");
    // get x
    ge::GeTensorDesc inputX = formerOpDesc->GetInputDesc(0);
    // set x input
    ge::GeTensorDesc x = inputX;
    x.SetFormat(ge::FORMAT_ND);
    x.SetOriginFormat(ge::FORMAT_ND);
    thisOpDesc->AddInputDesc("x",x);

    vector<int64_t> tensorXDims = inputX.GetShape().GetDims();
    if (tensorXDims.size() == dimThree) {
      int64_t inputSize = tensorXDims[dimTwo];
      ge::AttrUtils::SetInt(thisOpDesc, "input_size", inputSize);
    }
    // set w input
    thisOpDesc->AddInputDesc("w",wxhTensorDesc);

    // get bias to b
    ge::GeTensorDesc biasDesc = formerOpDesc->GetInputDesc(inputIndexInfo.biasIndex);
    ge::GeTensorDesc bias = biasDesc;
    bias.SetFormat(ge::FORMAT_ND);
    bias.SetOriginFormat(ge::FORMAT_ND);
    thisOpDesc->AddInputDesc("b",bias);

    // get cont to cont
    ge::GeTensorDesc contDesc = formerOpDesc->GetInputDesc(1);
    ge::GeTensorDesc cont = contDesc;
    cont.SetFormat(ge::FORMAT_ND);
    cont.SetOriginFormat(ge::FORMAT_ND);
    thisOpDesc->AddInputDesc("cont",cont);

    // set static
    if (inputIndexInfo.hasStatic){
        thisOpDesc->AddInputDesc("w_xc_x_static",staticTensorDesc);
    }

    // set h0, c0
    if (expose_hidden){
        thisOpDesc->AddInputDesc("h0",formerOpDesc->GetInputDesc(inputIndexInfo.h0Index));
        thisOpDesc->AddInputDesc("c0",formerOpDesc->GetInputDesc(inputIndexInfo.c0Index));
    }

    int32_t num_output = 0;
    ge::AttrUtils::GetInt(formerOpDesc,"num_output",num_output);
    // set num_output
    ge::AttrUtils::SetInt(thisOpDesc,"num_output",num_output);
    // set expose_hidden
    ge::AttrUtils::SetBool(thisOpDesc,"expose_hidden",expose_hidden);

    // set output y
    ge::GeTensorDesc outputY = formerOpDesc->GetOutputDesc(0);
    ge::GeTensorDesc outputYTensorDesc = outputY;
    thisOpDesc->AddOutputDesc("y",outputYTensorDesc);
    thisOpDesc->AddOutputDesc("output_h",outputYTensorDesc);
    thisOpDesc->AddOutputDesc("output_c",outputYTensorDesc);

    vector<int64_t> tensorYDims = outputY.GetShape().GetDims();
    if (tensorYDims.size() == dimThree) {
      int64_t hiddenSize = tensorYDims[dimTwo];
      ge::AttrUtils::SetInt(thisOpDesc, "hidden_size", hiddenSize);
    }

    // set last_h and last_c
    if (outputSize == dimThree) {
        // set need_output_last
        ge::AttrUtils::SetBool(thisOpDesc,"need_output_last",true);
        ge::GeShape shapeY = outputY.GetShape();
        std::vector<int64_t> lastOutputShape;
        lastOutputShape.push_back(1);
        lastOutputShape.push_back(shapeY.GetDim(1));
        lastOutputShape.push_back(shapeY.GetDim(dimTwo));
        GeShape input0ShapeNew(lastOutputShape);
        ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(input0ShapeNew, ge::FORMAT_NCHW,
                                                             staticTensorDesc.GetDataType());
        outputTensorDesc.SetShape(input0ShapeNew);
        outputTensorDesc.SetOriginShape(input0ShapeNew);
        outputTensorDesc.SetFormat(ge::FORMAT_NCHW);
        outputTensorDesc.SetOriginFormat(ge::FORMAT_NCHW);
        thisOpDesc->AddOutputDesc("last_output_h", outputTensorDesc);
        thisOpDesc->AddOutputDesc("last_output_c", outputTensorDesc);
    }

    return SUCCESS;
}

Status DynamicLSTMFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &newNodes) 
{
    // get the NodePtr of LSTM
    OP_LOGI(FUSED_OP_TYPE.c_str(), "lstm fusion start fusion");
    ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
    FUSION_PASS_CHECK(fusedNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                           "fusedNode is null, fusion failed."),
                      return PARAM_INVALID);
    int32_t inputSize = fusedNode->GetInDataNodes().size();
    int32_t outputSize = fusedNode->GetOutDataNodes().size();
    // get the OpDescPtr of LSTM
    ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
    FUSION_PASS_CHECK(fusedNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                           "fusedNode OpDesc is null, fusion failed."),
                      return PARAM_INVALID);

    // LSTM input X support 3 dim and 4 dim
    ge::GeTensorDesc tempInput0Desc = fusedDesc->GetInputDesc(0);
    ge::GeShape shapeInput0 = tempInput0Desc.GetShape();
    int64_t last_dim_value = shapeInput0.GetDim(dimTwo);
    if (shapeInput0.GetDims().size() == input0Size) {
        last_dim_value = shapeInput0.GetDim(dimTwo) * shapeInput0.GetDim(dimThree);
    }
    std::vector<int64_t> input0ShapeDim0;
    input0ShapeDim0.push_back(shapeInput0.GetDim(0));
    input0ShapeDim0.push_back(shapeInput0.GetDim(1));
    input0ShapeDim0.push_back(last_dim_value);

    GeShape input0ShapeNew(input0ShapeDim0);
    tempInput0Desc.SetShape(input0ShapeNew);
    tempInput0Desc.SetOriginShape(input0ShapeNew);
    fusedDesc->UpdateInputDesc(0, tempInput0Desc);

    bool hasStatic = false;
    bool expose_hidden = false;
    InputIndexInfo inputIndexInfo;
    ge::AttrUtils::GetBool(fusedDesc, "expose_hidden", expose_hidden);

    if (inputSize == 9) {
        inputIndexInfo.xStaticIndex = 2;
        inputIndexInfo.h0Index = 3;
        inputIndexInfo.c0Index = 4;
        inputIndexInfo.wxIndex = 5;
        inputIndexInfo.biasIndex = 6;
        inputIndexInfo.wxStaticIndex = 7;
        inputIndexInfo.whIndex = 8;
        inputIndexInfo.hasStatic = true;
        hasStatic = true;
    } else if (inputSize == 7 && expose_hidden) {
        inputIndexInfo.xStaticIndex = -1;
        inputIndexInfo.h0Index = 2;
        inputIndexInfo.c0Index = 3;
        inputIndexInfo.wxIndex = 4;
        inputIndexInfo.whIndex = 6;
        inputIndexInfo.biasIndex = 5;
        inputIndexInfo.wxStaticIndex = -1;
    } else if (inputSize == 7) {
        inputIndexInfo.xStaticIndex = 2;
        inputIndexInfo.h0Index = -1;
        inputIndexInfo.c0Index = -1;
        inputIndexInfo.wxIndex = 3;
        inputIndexInfo.biasIndex = 4;
        inputIndexInfo.wxStaticIndex = 5;
        inputIndexInfo.whIndex = 6;
        inputIndexInfo.hasStatic = true;
        hasStatic = true;
    } else if (inputSize == 5) {
        inputIndexInfo.wxIndex = 2;
        inputIndexInfo.biasIndex = 3;
        inputIndexInfo.whIndex = 4;
    }

    ge::NodePtr innerproductNode = nullptr;
    ge::GeTensorDesc outInnerProductTensorDesc;
    bool failStatus = false;
    if (hasStatic) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "has static start build static");
        Status resStatus = ProcessLSTMStatic(fusedNode, innerproductNode, graph, newNodes, inputIndexInfo);
        FUSION_PASS_CHECK(resStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                    "Process Static fail."), return PARAM_INVALID);
        ge::OpDescPtr InnerProductOpDesc = innerproductNode->GetOpDesc();
        outInnerProductTensorDesc = InnerProductOpDesc->GetOutputDesc(0);
    }

    // process w_xh
    ge::GeTensorPtr wxTensorPtr = ProcessLSTMWxh(fusedNode, failStatus, inputIndexInfo);
    FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                 "Process wxh fail."), return FAILED);
    FUSION_PASS_CHECK(wxTensorPtr == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                             "Process wxTensorPtr fail."),
                                                                             return FAILED);

    // add dynamicLSTM
    ge::OpDescPtr dynamicLSTMOpDesc = nullptr;
    ge::NodePtr dynamicLSTMNode = nullptr;
    FUSION_PASS_MAKE_SHARED((dynamicLSTMOpDesc = std::make_shared<ge::OpDesc>(fusedDesc->GetName() + "/DynamicLSTMV2",
                                                                              "DynamicLSTMV2")), return INTERNAL_ERROR);
    FUSION_PASS_CHECK(SUCCESS != AddDynamicLSTMNode(dynamicLSTMOpDesc, fusedDesc, wxTensorPtr->GetTensorDesc(),
                                                    inputIndexInfo, expose_hidden,
                                                    outInnerProductTensorDesc, outputSize),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "make DynamicLSTMV2 layer fail."), return FAILED);
    dynamicLSTMNode = graph.AddNode(dynamicLSTMOpDesc);
    FUSION_PASS_CHECK(dynamicLSTMNode == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "dynamicLSTM node is null, fusion failed."), return FAILED);

    // connect x
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                           dynamicLSTMNode->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                       "add DynamicLSTMV2 edge to fusion node x failed."), return FAILED);

    // connect w
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(inputIndexInfo.wxIndex)->GetPeerOutAnchor(),
                                           dynamicLSTMNode->GetInDataAnchor(1)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                       "add DynamicLSTMV2 edge to fusion node w failed."), return FAILED);
    
    // connect bias
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(inputIndexInfo.biasIndex)->GetPeerOutAnchor(),
                                           dynamicLSTMNode->GetInDataAnchor(2)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                       "add DynamicLSTMV2 edge to fusion node bias failed."), return FAILED);
    
    // connect cont
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                           dynamicLSTMNode->GetInDataAnchor(3)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                       "add DynamicLSTMV2 edge to fusion node bias failed."), return FAILED);
    
    int32_t inputxIndex = 4;

    // connect static
    if (hasStatic) {
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(
                           fusedNode->GetInDataAnchor(inputIndexInfo.xStaticIndex)->GetPeerOutAnchor(),
                           innerproductNode->GetInDataAnchor(0)),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                           "add Fc Node edge to fusion node xstatic failed."), return FAILED);
        
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(
                                       fusedNode->GetInDataAnchor(inputIndexInfo.wxStaticIndex)->GetPeerOutAnchor(),
                                               innerproductNode->GetInDataAnchor(1)),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                           "add Fc Node edge to fusion node wxstatic failed."), return FAILED);
        
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(innerproductNode->GetOutDataAnchor(0),
                                               dynamicLSTMNode->GetInDataAnchor(inputxIndex)),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                           "add Fc Node edge to fusion node x failed."), return FAILED);
        inputxIndex += 1;
    }

    // connect h_0 and c_0
    if (expose_hidden) {
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(inputIndexInfo.h0Index)->GetPeerOutAnchor(),
                                               dynamicLSTMNode->GetInDataAnchor(inputxIndex)),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
            "add dynamicLSTMV2 Node edge to fusion node h0 failed."), return FAILED);
        inputxIndex += 1;
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(inputIndexInfo.c0Index)->GetPeerOutAnchor(),
            dynamicLSTMNode->GetInDataAnchor(inputxIndex)),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
            "add dynamicLSTMV2 Node edge to fusion node c0 failed."), return FAILED);
    }

    ge::OutDataAnchorPtr outputY = fusedNode->GetOutDataAnchor(0);
    auto hOriTopPeerAnchors = outputY->GetPeerInDataAnchors();
    ge::OutDataAnchorPtr outputH = fusedNode->GetOutDataAnchor(1);
    auto htOriTopPeerAnchors = outputH->GetPeerInDataAnchors();
    ge::OutDataAnchorPtr outputC = fusedNode->GetOutDataAnchor(ctOutputSize);
    auto ctOriTopPeerAnchors = outputC->GetPeerInDataAnchors();

    // unlink all control input of LSTMD
    if (fusedNode->GetInControlAnchor() != nullptr) {
        fusedNode->GetInControlAnchor()->UnlinkAll();
    }

    // unlink all input of LSTMD
    for (auto inAnchor : fusedNode->GetAllInDataAnchors()) {
        if (inAnchor != nullptr) {
            inAnchor->UnlinkAll();
        }
    }

    // unlink all output
    for (auto outAnchor : fusedNode->GetAllOutDataAnchors()) {
        if (outAnchor != nullptr) {
            outAnchor->UnlinkAll();
        }
    }

    // Get Output Node
    for (uint64_t i = 0; i < hOriTopPeerAnchors.size(); i++) {
        ge::InDataAnchorPtr oriTopPeerAnchorPtri = hOriTopPeerAnchors.at(i);
        ge::NodePtr outputNode = oriTopPeerAnchorPtri->GetOwnerNode();
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(dynamicLSTMNode->GetOutDataAnchor(0), oriTopPeerAnchorPtri),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
            "add dynamicLSTMV2 Node edge to fusion node output y failed."), return FAILED);
    }

    if (outputSize == nodeOutputSize) {
        for (uint64_t i = 0; i < htOriTopPeerAnchors.size(); i++) {
            ge::InDataAnchorPtr oriTopPeerAnchorPtri = htOriTopPeerAnchors.at(i);
            ge::NodePtr outputNode = oriTopPeerAnchorPtri->GetOwnerNode();
            FUSION_PASS_CHECK(
                SUCCESS != ge::GraphUtils::AddEdge(dynamicLSTMNode->GetOutDataAnchor(3), oriTopPeerAnchorPtri),
                VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                "add dynamicLSTMV2 Node edge to fusion node last output h failed."),
                return FAILED);
        }

        for (uint64_t i = 0; i < ctOriTopPeerAnchors.size(); i++) {
            ge::InDataAnchorPtr oriTopPeerAnchorPtri = ctOriTopPeerAnchors.at(i);
            ge::NodePtr outputNode = oriTopPeerAnchorPtri->GetOwnerNode();
            FUSION_PASS_CHECK(
                SUCCESS != ge::GraphUtils::AddEdge(dynamicLSTMNode->GetOutDataAnchor(4), oriTopPeerAnchorPtri),
                VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                               "add dynamicLSTMV2 Node edge to fusion node last output c failed."),
                return FAILED);
        }
    }

    // remove LSTMD from graph
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(fusedNode),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "remove fusedNode node[%s] failed", fusedNode->GetName().c_str()),
                    return FAILED);
    return SUCCESS;
}

REGISTER_PASS("DynamicLSTMFusionPass", BUILT_IN_GRAPH_PASS, DynamicLSTMFusionPass);
} // namespace fe
