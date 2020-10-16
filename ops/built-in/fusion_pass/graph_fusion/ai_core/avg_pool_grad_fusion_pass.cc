/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @flie   avg_pool_grad_fusion_pass.h
 *
 * @brief  avg_pool_grad fusion pass(avg_pool_grad --> avg_pool_grad_d)
 *
 */
#include "avg_pool_grad_fusion_pass.h"
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include "fp16_t.hpp"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {
const std::string AVGPOOLGRAD = "AvgPoolGrad";
static const std::string PATTERN_AVGPOOLGRAD = "AvgPoolGrad";
static const float FLOAT_NUM_ZERO = 0;
const std::string CONSTANTOP = "Constant";

Status AvgPoolGradFusionPass::WindowedOutputSize(int32_t input, int32_t kSize, int32_t stride, string padding, int32_t &output,
    int32_t &padBefor, int32_t &padAfter)
{
    int32_t tmpOutput = 0;
    int32_t tmpPadneed = 0;
    int32_t tmpPadBefor = 0;
    int32_t tmpPadAfter = 0;
    if (stride <= 0) {
         OP_LOGE(FUSED_OP_TYPE.c_str(), "stride less or equal than zero");
         return FAILED;
    }

    if (padding == "VALID") {
        tmpOutput = (input - kSize + stride) / stride;
        tmpPadBefor = 0;
        tmpPadAfter = 0;
    } else if (padding == "SAME") {
        tmpOutput = (input + stride - 1) / stride;
        tmpPadneed = max(0, ((tmpOutput - 1) * stride + kSize - input));
        tmpPadBefor = tmpPadneed / 2;
        tmpPadAfter = tmpPadneed - tmpPadBefor;
    } else {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "AvgPoolGrad padding arg not surport padding model");
        return FAILED;
    }
    output = tmpOutput;
    padBefor = tmpPadBefor;
    padAfter = tmpPadAfter;
    return SUCCESS;
}

Status AvgPoolGradFusionPass::TransposeNCHW2NHWC(int32_t nOutput, int32_t hOutput, int32_t wOutput, int32_t cOutput, uint16_t* avgpoolout)
{
    uint64_t len = static_cast<uint64_t>(nOutput) * static_cast<uint64_t>(hOutput) * static_cast<uint64_t>(wOutput)
        * static_cast<uint64_t>(cOutput);
    if ((len > INT_MAX) || (len <= 0)) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "malloc memory too large");
        return FAILED;
    }
    uint16_t* tmp = new(std::nothrow) uint16_t[len];
    if (tmp == nullptr) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "malloc memory failed");
        return FAILED;
    }
    for (int32_t n = 0; n < nOutput; n++) {
        for (int32_t h = 0; h < hOutput; h++) {
            for (int32_t w = 0; w < wOutput; w++) {
                for (int32_t c = 0; c < cOutput; c++) {
                    tmp[n * hOutput * wOutput * cOutput + h * wOutput * cOutput + w * cOutput + c] =
                        avgpoolout[n * cOutput * hOutput * wOutput + c * hOutput * wOutput + h * wOutput + w];
                }
            }
        }
    }
    errno_t ret = memcpy_s(avgpoolout, len * sizeof(uint16_t), tmp, len * sizeof(uint16_t));
    if (ret != EOK) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "memcpy_s fail!");
        delete[] tmp;
        return FAILED;
    }
    delete[] tmp;
    return SUCCESS;
}

Status AvgPoolGradFusionPass::AvgValueTableGen(vector<int64_t> dimInfo, vector<int64_t> Ksize, vector<int64_t> strides, string padding,
    string data_format, vector<int64_t> &assitDimInfo, uint16_t *output)
{
    fp16_t tmp;
    tmp.val = 0;
    int64_t nInput = 0;
    int64_t cInput = 0;
    int64_t hInput = 0;
    int64_t wInput = 0;
    int64_t hKsize = 0;
    int64_t wKsize = 0;
    int64_t hStride = 0;
    int64_t wStride = 0;
    // dimInfo must NHWC
    nInput = dimInfo[0];
    hInput = dimInfo[1];
    wInput = dimInfo[2];
    cInput = dimInfo[3];

    if ((Ksize[0] == 1) || (Ksize[3] == 1)) {
        hKsize = Ksize[1];
        wKsize = Ksize[2];
    } else {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "AvgPoolGrad ksize error");
        return FAILED;
    }
    if ((strides[0] == 1) || (strides[3] == 1)) {
        hStride = strides[1];
        wStride = strides[2];
    } else {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "AvgPoolGrad strides arg error");
        return FAILED;
    }
    int32_t nOutput = nInput;
    int32_t cOutput = cInput;

    int32_t hOutput = 0;
    int32_t padTop = 0;
    int32_t padBottom = 0;
    WindowedOutputSize(hInput, hKsize, hStride, padding, hOutput, padTop, padBottom);
    int32_t wOutput = 0;
    int32_t padLeft = 0;
    int32_t padRight = 0;
    int32_t add_flag_h = 0;
    int32_t add_flag_w = 0;
    WindowedOutputSize(wInput, wKsize, wStride, padding, wOutput, padLeft, padRight);
    int64_t outOffsetPoint = 0;
    for (int n = 0; n < nOutput; n++) {
        for (int c = 0; c < cOutput; c++) {
            for (int h = 0; h < hOutput; h++) {
                for (int w = 0; w < wOutput; w++) {
                    for (int hk = 0; hk < hKsize; hk++) {
                        for (int wk = 0; wk < wKsize; wk++) {
                            add_flag_h = 0;
                            add_flag_w = 0;
                            outOffsetPoint = n * cOutput * hOutput * wOutput + c * hOutput * wOutput + h * wOutput + w;
                            if ((padTop == 0) && (padBottom == 1)) {
                                if ((padTop <= (h * hStride + hk)) && ((h * hStride + hk - hInput + 1) < padBottom)) {
                                    add_flag_h = 1;
                                }
                            }
                            else {
                                if ((padTop <= (h * hStride + hk)) && ((h * hStride + hk - hInput + 1) <= padBottom)) {
                                    add_flag_h = 1;
                                }
                            }
                            if ((padLeft == 0) && (padRight == 1)) {
                                if ((padLeft <= (w * wStride + wk)) && ((w * wStride + wk - wInput + 1) < padRight)) {
                                    add_flag_w = 1;
                                }
                            }
                            else {
                                if ((padLeft <= (w * wStride + wk)) && ((w * wStride + wk - wInput + 1) <= padRight)) {
                                    add_flag_w = 1;
                                }
                            }
                            if ((add_flag_h == 1) && (add_flag_w == 1)) {
                                output[outOffsetPoint] += 1;
                            }
                        }
                    }
                    fp16_t tmp;
                    tmp.val = 0;
                    tmp.val = output[outOffsetPoint];
                    fp16_t tmp2;
                    tmp2.val = 0;
                    tmp2 = 1 / (float)tmp.val;
                    output[outOffsetPoint] = tmp2.val;
                }
            }
        }
    }
    if (data_format == "NHWC") {
        TransposeNCHW2NHWC(nOutput, hOutput, wOutput, cOutput, output);
    }
    if (data_format == "NHWC") {
        assitDimInfo.push_back(nOutput);
        assitDimInfo.push_back(hOutput);
        assitDimInfo.push_back(wOutput);
        assitDimInfo.push_back(cOutput);
    } else if (data_format == "NCHW") {
        assitDimInfo.push_back(nOutput);
        assitDimInfo.push_back(cOutput);
        assitDimInfo.push_back(hOutput);
        assitDimInfo.push_back(wOutput);
    }
    return SUCCESS;
}

Status KernelGen(int32_t hKsize, int32_t wKsize, int32_t cInput, vector<int64_t> assitDimInfo, uint16_t *kernelTable)
{
    // this 6D is not safe ,because gragh donot know this info
    // from depthwise, filter is HWNC, but ge get shape by NHWC, so, plugin set format HWNC.
    int64_t len = static_cast<int64_t>(hKsize) * static_cast<int64_t>(wKsize) * static_cast<int64_t>(cInput);
    fp16_t tmp;
    tmp.val = 0;
    for (int64_t i = 0; i < len; i++) {
        tmp.val = 1.0;
        fp16_t tmp2;
        tmp2.val = 0;
        tmp2 = (float)tmp.val;
        kernelTable[i] = tmp2.val;
    }
    return SUCCESS;
}

vector<FusionPattern *> AvgPoolGradFusionPass::DefinePatterns()
{
    vector<FusionPattern *> patterns;
    FusionPattern *pattern = new (std::nothrow) FusionPattern("AvgPoolGradFusion");
    FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."), return patterns);
    pattern->AddOpDesc(PATTERN_AVGPOOLGRAD, { AVGPOOLGRAD }).SetOutput(PATTERN_AVGPOOLGRAD);
    patterns.push_back(pattern);
    return patterns;
}

// vector<ge::NodePtr> &fusionNodes: Store fusion nodes,
// including newly added nodes and fused but not deleted nodes
Status AvgPoolGradFusionPass::Fusion(ge::ComputeGraph &graph,
                                     Mapping &mapping,
                                     vector<ge::NodePtr> &fusionNodes)
{
    std::string fusionOpType = "AvgPoolGradD";
    std::map<int16_t, std::string> avgPoolGradAttrInfo;
    avgPoolGradAttrInfo[0] = "orig_input_shape";
    PatternFusionUtil patternFusionUtil;
    // get node pointer
    ge::NodePtr avpPoolGradfusedNode = GetNodeFromMapping(PATTERN_AVGPOOLGRAD, mapping);
    FUSION_PASS_CHECK(avpPoolGradfusedNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "avpPoolGradfusedNode is null, fusion failed."),
        return PARAM_INVALID);
    // get opdesc pointer
    ge::OpDescPtr avgPoolGradDesc = avpPoolGradfusedNode->GetOpDesc();
    FUSION_PASS_CHECK(avgPoolGradDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "avgPoolGradDesc is null, fusion failed."), return PARAM_INVALID);
    string dataFormat;
    vector<int64_t> Ksize;
    vector<int64_t> strides;
    string padding;

    // get ksize padding strides value dataFormat
    ge::AttrUtils::GetStr(avgPoolGradDesc, "data_format", dataFormat);
    ge::AttrUtils::GetListInt(avgPoolGradDesc, "ksize", Ksize);
    ge::AttrUtils::GetListInt(avgPoolGradDesc, "strides", strides);
    ge::AttrUtils::GetStr(avgPoolGradDesc, "padding", padding);

    // get const org_input_shape desc, dtype, format, dims
    ge::InDataAnchorPtr avgPoolGradConstAnchorPtr0 = avpPoolGradfusedNode->GetInDataAnchor(0);
    ge::OutDataAnchorPtr constAnchorPtr = avgPoolGradConstAnchorPtr0->GetPeerOutAnchor();
    ge::NodePtr constNode = constAnchorPtr->GetOwnerNode();
    ge::GeTensorDesc orgInputShapeTensor = constNode->GetOpDesc()->GetOutputDesc(0);
    DataType dataType = orgInputShapeTensor.GetDataType();
    ge::GeShape orgInputShape = orgInputShapeTensor.GetShape();
    int64_t dimNums = orgInputShape.GetShapeSize();
    if (dimNums != 4) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "org_input_shape dimNums must be 4.");
        return FAILED;
    }
    vector<int64_t> dimInfo = orgInputShape.GetDims();
    // get orig_input_shape value
    Operator op = ge::OpDescUtils::CreateOperatorFromNode(avpPoolGradfusedNode);
    Tensor origInputShapeConstTensor;
    op.GetInputConstData(avgPoolGradAttrInfo[0], origInputShapeConstTensor);
    if (dataType != ge::DT_INT32) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "orig_input_shape dtype only surpport INT32.");
        return FAILED;
    }
    int32_t *origInputShapeConstTensorPrt = (int32_t *)origInputShapeConstTensor.GetData();
    if (dimInfo[0] != 4) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "orig_input_shape must be list of 4.");
        return FAILED;
    }

    // gen avgtable matrix
    ge::GeTensorDesc avpPoolInputShapeTensor = avpPoolGradfusedNode->GetOpDesc()->GetInputDesc(1);
    ge::GeShape avgPoolShape = avpPoolInputShapeTensor.GetShape();
    vector<int64_t> avgPoolDimInfo = avgPoolShape.GetDims();
    if (Ksize.size() != 4) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Ksize must list of 4 element.");
        return FAILED;
    }
    if (strides.size() != 4) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "strides must list of 4 element.");
        return FAILED;
    }
    if (dataFormat == "NHWC") {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "AvgPoolGrad dataFormat NHWC.");
        if ((Ksize[0] != 1) || (Ksize[3] != 1)) {
            OP_LOGE(FUSED_OP_TYPE.c_str(), "AvgPoolGrad NHWC,ksize only surpport ksize[0]==ksize[3]==1.");
            return FAILED;
        }
        if ((strides[0] != 1) || (strides[3] != 1)) {
            OP_LOGE(FUSED_OP_TYPE.c_str(), "AvgPoolGrad NHWC stride only surpport strides[0]==strides[3]==1.");
            return FAILED;
        }
    } else if (dataFormat == "NCHW") {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "AvgPoolGrad dataFormat NCHW.");
        int64_t Ksize_h = 0;
        int64_t Ksize_w = 0;
        int64_t stride_h = 0;
        int64_t stride_w = 0;
        int64_t origInputShapeH = 0;
        int64_t origInputShapeW = 0;
        int64_t origInputShapeC = 0;

        if ((Ksize[0] != 1) || (Ksize[1] != 1)) {
            OP_LOGE(FUSED_OP_TYPE.c_str(), "AvgPoolGrad NCHW, stride only surpport ksize[0]==ksize[3]==1.");
            return FAILED;
        }
        Ksize_h = Ksize[2];
        Ksize_w = Ksize[3];
        Ksize[0] = 1;
        Ksize[1] = Ksize_h;
        Ksize[2] = Ksize_w;
        Ksize[3] = 1;
        if ((strides[0] != 1) || (strides[1] != 1)) {
            OP_LOGE(FUSED_OP_TYPE.c_str(), "AvgPoolGrad NCHW, stride only surpport strides[0]==strides[1]==1.");
            return FAILED;
        }
        stride_h = strides[2];
        stride_w = strides[3];
        strides[0] = 1;
        strides[1] = stride_h;
        strides[2] = stride_w;
        strides[3] = 1;
        origInputShapeH = origInputShapeConstTensorPrt[2];
        origInputShapeW = origInputShapeConstTensorPrt[3];
        origInputShapeC = origInputShapeConstTensorPrt[1];
        origInputShapeConstTensorPrt[1] = origInputShapeH;
        origInputShapeConstTensorPrt[2] = origInputShapeW;
        origInputShapeConstTensorPrt[3] = origInputShapeC;
    }
    vector<int64_t> origInputShapeV;
    origInputShapeV.push_back((int64_t)origInputShapeConstTensorPrt[0]);
    origInputShapeV.push_back((int64_t)origInputShapeConstTensorPrt[1]);
    origInputShapeV.push_back((int64_t)origInputShapeConstTensorPrt[2]);
    origInputShapeV.push_back((int64_t)origInputShapeConstTensorPrt[3]);
    // origInputShapeV must NHWC, Ksize(1,h,w,1) strides(1,h,w,1) origInputShapeConstTensorPrt(N,H,W,C)
    if (!((origInputShapeConstTensorPrt[1] == Ksize[1]) && (origInputShapeConstTensorPrt[2] == Ksize[2]) &&
        (padding == "VALID"))) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "AvgPoolGrad now is no global mode");
        ge::GeTensorPtr AvgTableAssitPtr = nullptr;
        int64_t valueTableSize = avgPoolDimInfo[0] * avgPoolDimInfo[1] * avgPoolDimInfo[2] * avgPoolDimInfo[3];

        FUSION_PASS_CHECK((((avgPoolDimInfo[1] * avgPoolDimInfo[2] * avgPoolDimInfo[3]) == 0) || (valueTableSize <= 0)),\
         OP_LOGE(FUSED_OP_TYPE.c_str(), "valueTableSize have 0 element"), return FAILED);
        FUSION_PASS_CHECK((avgPoolDimInfo[0] != valueTableSize/(avgPoolDimInfo[1] * avgPoolDimInfo[2] * avgPoolDimInfo[3])),\
         OP_LOGE(FUSED_OP_TYPE.c_str(), "valueTableSize overlap , over int64"), return FAILED);

        unique_ptr<uint16_t> inputAssit(new (std::nothrow) uint16_t[valueTableSize]());
        FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"), return PARAM_INVALID);
        vector<int64_t> avgPoolAssitDimInfo;
        Status ret = AvgValueTableGen(origInputShapeV, Ksize, strides, padding, dataFormat, avgPoolAssitDimInfo,
            inputAssit.get());
        FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return ret);
        ge::GeShape avpPoolAssitShape(avgPoolAssitDimInfo);
        ge::GeTensorDesc tensorDesc(GeShape(), ge::FORMAT_NHWC, ge::DT_FLOAT16);
        if (dataFormat == "NHWC") {
            tensorDesc.SetFormat(ge::FORMAT_NHWC);
            tensorDesc.SetOriginFormat(ge::FORMAT_NHWC);
        } else if (dataFormat == "NCHW") {
            tensorDesc.SetFormat(ge::FORMAT_NCHW);
            tensorDesc.SetOriginFormat(ge::FORMAT_NCHW);
        }
        tensorDesc.SetShape(avpPoolAssitShape);
        tensorDesc.SetOriginShape(avpPoolAssitShape);

        FUSION_PASS_MAKE_SHARED((AvgTableAssitPtr = std::make_shared<ge::GeTensor>(tensorDesc,
                            reinterpret_cast<uint8_t*>(inputAssit.get()), valueTableSize * sizeof(uint16_t))),
                       AvgTableAssitPtr = nullptr;
                       return PARAM_INVALID);
        vector<ge::GeTensorPtr> avgPoolGradWeights = { AvgTableAssitPtr };
        ge::OpDescUtils::SetWeights(avpPoolGradfusedNode, avgPoolGradWeights);
        auto avgPoolConstInputNodes = OpDescUtils::GetConstInputs(avpPoolGradfusedNode);
        NodePtr avgPoolConstInput = avgPoolConstInputNodes[0];
        avgPoolConstInput->GetOpDesc()->SetType(CONSTANTOP);
        avgPoolGradDesc->SetType("AVGPOOLGRAD");

        // gen kernel matrix
        // origInputShapeV[1] must be channel
        int64_t kernelTableSize = origInputShapeV[3] * Ksize[1] * Ksize[2];
        FUSION_PASS_CHECK((((Ksize[1] * Ksize[2]) == 0) || (kernelTableSize <= 0)),\
         OP_LOGE(FUSED_OP_TYPE.c_str(), "kernelTableSize have O element"), return FAILED);
        FUSION_PASS_CHECK((origInputShapeV[3] != kernelTableSize/(Ksize[1] * Ksize[2])),\
         OP_LOGE(FUSED_OP_TYPE.c_str(), "kernelTableSize overlap , over int64"), return FAILED);
        ge::GeTensorPtr kernelTableassitPtr = nullptr;

        unique_ptr<uint16_t> kernelTableinputAssit(new (std::nothrow) uint16_t[kernelTableSize]());
        FUSION_PASS_CHECK(kernelTableinputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "kernelTableinputAssit is NULL"),
            return PARAM_INVALID);
        vector<int64_t> kernelTableassitDimInfo;
        ret = KernelGen(Ksize[1], Ksize[2], origInputShapeV[3], kernelTableassitDimInfo, kernelTableinputAssit.get());
        FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "kernelTable matrix AssitHelp failed."), return ret);
        kernelTableassitDimInfo.push_back((int64_t)Ksize[1]);
        kernelTableassitDimInfo.push_back((int64_t)Ksize[2]);
        kernelTableassitDimInfo.push_back((int64_t)origInputShapeConstTensorPrt[3]);
        kernelTableassitDimInfo.push_back((int64_t)1);
        ge::GeShape assitShape(kernelTableassitDimInfo);
        ge::GeTensorDesc kernelTableTensorDesc(GeShape(), ge::FORMAT_HWCN, ge::DT_FLOAT16);
        kernelTableTensorDesc.SetShape(assitShape);
        kernelTableTensorDesc.SetOriginShape(assitShape);
        kernelTableTensorDesc.SetOriginFormat(ge::FORMAT_HWCN);
        kernelTableTensorDesc.SetFormat(ge::FORMAT_HWCN);
        FUSION_PASS_MAKE_SHARED((kernelTableassitPtr = std::make_shared<ge::GeTensor>(kernelTableTensorDesc,
                            reinterpret_cast<uint8_t*>(kernelTableinputAssit.get()), kernelTableSize * sizeof(uint16_t))),
                       kernelTableassitPtr = nullptr;
                       return PARAM_INVALID);
        vector<ge::GeTensorPtr> kernelWeights = { kernelTableassitPtr };
        ge::OpDescUtils::SetWeights(avpPoolGradfusedNode, kernelWeights);
        auto kernelConstInputNodes = OpDescUtils::GetConstInputs(avpPoolGradfusedNode);
        NodePtr kernelConstInput = kernelConstInputNodes[0];
        kernelConstInput->GetOpDesc()->SetType(CONSTANTOP);
        avgPoolGradDesc->SetType("AVGPOOLGRAD");

    } else {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "AvgPoolGrad now is global mode");
    }

    std::vector<PassAttrInfo> avgPoolGradPassInfo;
    ge::NodePtr fusion_node = nullptr;
    PassAttrInfo orig_input_shape = {0, "orig_input_shape", "SetListInt"};
    avgPoolGradPassInfo.push_back(orig_input_shape);
    // const org input shape change to attr
    Status ret = patternFusionUtil.ConstToAttrWithNode(graph, avpPoolGradfusedNode, fusionOpType, avgPoolGradPassInfo, fusion_node);
    if (ret != SUCCESS) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "AvgPoolGrad has input which is not a CONST, graph not changed.");
        return NOT_CHANGED;
    }
    fusionNodes.push_back(fusion_node);
    return SUCCESS;
}
REGISTER_PASS("AvgPoolGradFusionPass", BUILT_IN_GRAPH_PASS, AvgPoolGradFusionPass);
}
