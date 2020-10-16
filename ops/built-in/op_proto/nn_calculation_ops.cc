/**
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file nn_calculation_ops.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#define CHECK_FORMAT(format)  \
{                                         \
    if (ge::FORMAT_RESERVED == format) {    \
        OP_LOGE(op.GetName().c_str(), "get format failed:%s:%d", #format, format); \
        return false;     \
    }                     \
}

#include "inc/nn_calculation_ops.h"
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include "util/util.h"
#include "op_log.h"
#include "./util/error_util.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/utils/type_utils.h"

namespace ge
{
// ----------------LSTM begin-------------------

IMPLEMT_VERIFIER(LSTM,LSTMInferShape) {
  return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(LSTM, LSTMVerify)
{
    int32_t output_size = 0;
    int32_t input_size = op.GetInputsSize();
    bool expose_hidden = false;

    if (ge::GRAPH_SUCCESS != op.GetAttr("expose_hidden", expose_hidden))
    {
        OP_LOGE(op.GetName().c_str(),"GetOpAttr expose_hidden failed!");
    }

    if (input_size == 9)
    {
        output_size = op.GetInputDesc(3).GetShape().GetDim(2);
    }else if (input_size == 7 and expose_hidden) {
        output_size = op.GetInputDesc(2).GetShape().GetDim(2);
    }else if (input_size == 7) {
        output_size = op.GetInputDesc(6).GetShape().GetDim(1);
    }else{
        output_size = op.GetInputDesc(4).GetShape().GetDim(1);
    }

    ge::TensorDesc inputXTensorDesc = op.GetInputDesc(0);

    vector<int64_t> hDims;

    hDims.push_back(inputXTensorDesc.GetShape().GetDim(0));
    hDims.push_back(inputXTensorDesc.GetShape().GetDim(1));
    hDims.push_back(output_size);

    TensorDesc outputHTensorDesc = op.GetOutputDesc(0);
    outputHTensorDesc.SetShape(ge::Shape(hDims));
    (void) op.UpdateOutputDesc("h", outputHTensorDesc);

    if (expose_hidden)
    {
       int32_t c_index = 0;
       int32_t h_index = 0;
       if (input_size == 9)
       {
          h_index = 3;
          c_index = 4;
       }else{
          h_index = 2;
          c_index = 3;
       }
       ge::TensorDesc inputHTensorDesc = op.GetInputDesc(h_index);
       ge::TensorDesc inputCTensorDesc = op.GetInputDesc(c_index);
       ge::Shape shape = inputCTensorDesc.GetShape();
       ge::Shape shapeH = inputHTensorDesc.GetShape();

       TensorDesc outputHtTensorDesc = op.GetOutputDesc(1);
       TensorDesc outputCtTensorDesc = op.GetOutputDesc(2);

       outputCtTensorDesc.SetShape(shape);
       outputHtTensorDesc.SetShape(shapeH);

       (void) op.UpdateOutputDesc("h_t", outputHtTensorDesc);
       (void) op.UpdateOutputDesc("c_t", outputCtTensorDesc);
    }

    return GRAPH_SUCCESS;
}
INFER_FUNC_REG(LSTM, LSTMInferShape);
VERIFY_FUNC_REG(LSTM, LSTMVerify);
// ----------------LSTM end------------------
//----------------DepthwiseConv2d Op-------------------
// Obtains the value of the constant tensor.
static std::vector<int64_t> GetAttrValue(const ge::Operator &op,
                                   const std::string &key_name)
{
    std::vector<int64_t> list;
    if (ge::GRAPH_SUCCESS != op.GetAttr(key_name, list))
    {
        OP_LOGE(op.GetName().c_str(),"GetOpAttr ConstValue failed!");
    }

  return list;
}

static bool CheckListEmpty(const std::string& opName, const std::vector<int64_t>& list, const std::string& attrName)
{
    if (list.empty())
    {
        OP_LOGE(opName.c_str(),"the %s is empty !", attrName.c_str());
        return false;
    }

  return true;
}

static bool GetPadDepthwiseConv2D(ge::Operator& op,
                           int64_t inH, int64_t inW,
                           int64_t filterH, int64_t filterW,
                           int64_t strideH, int64_t strideW,
                           int64_t dilationH, int64_t dilationW,
                           int64_t& padtop, int64_t& padbottom,
                           int64_t& padleft, int64_t& padright) {
    std::string padStr;
    std::vector<int64_t> padList;
    if (GRAPH_SUCCESS == op.GetAttr("_padding", padList)) {
        op.SetAttr("pads", padList);
    } else if (GRAPH_SUCCESS == op.GetAttr("padding", padStr)) {
        if (padStr.compare("SAME") == 0) {
            int64_t effective_filter_h = (filterH - 1) * dilationH + 1;
            int64_t effective_filter_w = (filterW - 1) * dilationW + 1;
            int64_t out_h = (inH + strideH - 1) / strideH;
            int64_t out_w = (inW + strideW - 1) / strideW;
            int64_t pad_h = std::max((out_h - 1) * strideH + effective_filter_h - inH,(int64_t)0);
            int64_t pad_w = std::max((out_w - 1) * strideW + effective_filter_w - inW,(int64_t)0);
            padList.push_back(pad_h / 2);
            padList.push_back(pad_h / 2 + pad_h % 2);
            padList.push_back(pad_w / 2);
            padList.push_back(pad_w / 2 + pad_w % 2);
        } else if (padStr.compare("VALID") == 0) {
            padList.push_back(0);
            padList.push_back(0);
            padList.push_back(0);
            padList.push_back(0);
        } else {
            OP_LOGE(op.GetName().c_str(), "padding should be SAME or VALID."
                    " actual is: %s.", padStr.c_str());
            return false;
        }
        op.SetAttr("pads", padList);
    }
    std::vector<int64_t> padVec;
    if (op.GetAttr("pads", padVec) == ge::GRAPH_FAILED) {
        OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue padding failed!");
        return false;
    }
    auto pSize = padVec.size();
    if (pSize != 4) {
        OP_LOGE(op.GetName().c_str(), "pads list should be 4d."
                " actual is: %d.", (int)pSize);
        return false;
    }
    padtop = padVec[0];
    padbottom = padVec[1];
    padleft = padVec[2];
    padright = padVec[3];
    if (padtop < 0 || padbottom < 0 || padleft < 0 || padright < 0) {
        OP_LOGE(op.GetName().c_str(), "pads should be positive, "
                " actual is [%d,%d,%d,%d].", padtop, padbottom, padleft, padright);
        return false;
    }

    return true;
}

//Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(DepthwiseConv2D, DepthwiseConv2DVerify)
{
    auto xTensor = op.GetInputDesc(0);
    auto wTensor = op.GetInputDesc(1);

    auto xShape = xTensor.GetShape().GetDims();
    auto wShape = wTensor.GetShape().GetDims();

    if (xShape.size() != 4) {
        OP_LOGE(op.GetName().c_str(),
            "input x shape should be 4d. input x shape size is %d",
            (int)xShape.size());
        return GRAPH_FAILED;
    }

    if (wShape.size() != 4) {
        OP_LOGE(op.GetName().c_str(),
            "input filter shape should be 4d. input filter shape size is %d",
            (int)wShape.size());
        return GRAPH_FAILED;
    }

    auto xDtype = xTensor.GetDataType();
    auto wDtype = wTensor.GetDataType();

    if (xDtype != wDtype) {
        OP_LOGE(op.GetName().c_str(),
            "input x dtype(%d) is differ from filter dtype(%d).",
            (int)xDtype, (int)wDtype);
        return GRAPH_FAILED;
    }

    std::vector<int64_t> dilation;
    dilation = GetAttrValue(op, "dilations");
    if (!CheckListEmpty(op.GetName(), dilation, "dilations")) {
        OP_LOGE(op.GetName().c_str(), "Get dilations failed!");
        return GRAPH_FAILED;
    }
    std::vector<int64_t> stride;
    stride = GetAttrValue(op, "strides");
    if (!CheckListEmpty(op.GetName(), stride, "strides")) {
        OP_LOGE(op.GetName().c_str(), "Get stride failed!");
        return GRAPH_FAILED;
    }
    if (stride.size() != 4) {
        OP_LOGE(op.GetName().c_str(), "stride dim(%d) must be 4!",
            (int)stride.size());
        return GRAPH_FAILED;
    }

    std::vector<int64_t> pads;
    pads = GetAttrValue(op, "pads");
    if (!CheckListEmpty(op.GetName(), pads, "pads")) {
        OP_LOGE(op.GetName().c_str(), "Get pads failed!");
        return GRAPH_FAILED;
    }
    if (pads.size() != 4) {
        OP_LOGE(op.GetName().c_str(), "attr pads(%d) is too large",
            (int)pads.size());
        return GRAPH_FAILED;
    }
    std::string data_format;
    if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
        if (data_format != "NCHW" && data_format != "NHWC") {
            OP_LOGE(op.GetName().c_str(),
                "attr data_format(%s) only support NCHW and NHWC",
                data_format.c_str());
            return GRAPH_FAILED;
        }
    }
    // attr offset_x need not check
    return GRAPH_SUCCESS;
}

static map<int, std::string> format2str={
            {ge::FORMAT_NCHW, "NCHW"},
            {ge::FORMAT_NHWC, "NHWC"},
            {ge::FORMAT_HWCN, "HWCN"},
            {ge::FORMAT_DHWNC, "DHWNC"},
            {ge::FORMAT_DHWCN, "DHWCN"},
            {ge::FORMAT_NDHWC, "NDHWC"},
            {ge::FORMAT_NCDHW, "NCDHW"}
};

static bool GetDimInFormat(const std::string& opName, const std::string& formatStr,
                           const std::string& dimName, int64_t& dimPosition) {
    dimPosition = formatStr.find(dimName);
    if (dimPosition < 0) {
        OP_LOGE(opName.c_str(),"Position(%s) is invalid: %d, which format is %s.",
                dimName.c_str(), dimPosition, formatStr.c_str());
        return false;
    }
    return true;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DepthwiseConv2DInferShape)
{
    OP_LOGI(op.GetName().c_str(), "Enter op_proto inferfunction!");

    int64_t nPosition = 0;
    int64_t cPosition = 0;
    int64_t hPosition = 0;
    int64_t wPosition = 0;
    int64_t inN = 0;
    int64_t inC = 0;
    int64_t inH = 0;
    int64_t inW = 0;
    int64_t outH = 0;
    int64_t outW = 0;
    int64_t filterH = 0;
    int64_t filterW = 0;
    int64_t dilationH = 0;
    int64_t dilationW = 0;
    int64_t strideH = 0;
    int64_t strideW = 0;
    int64_t effectiveFilterH = 0;
    int64_t effectiveFilterW = 0;
    int64_t padtop = 0;
    int64_t padbottom = 0;
    int64_t padleft = 0;
    int64_t padright = 0;

    std::vector<int64_t> dilation;
    dilation = GetAttrValue(op, "dilations");
    if (!CheckListEmpty(op.GetName(), dilation, "dilations"))
    {
        OP_LOGE(op.GetName().c_str(),"Get dilations failed!");
        return GRAPH_FAILED;
    }

    std::vector<int64_t> stride;
    stride = GetAttrValue(op, "strides");
    if (!CheckListEmpty(op.GetName(), stride, "strides"))
    {
        OP_LOGE(op.GetName().c_str(),"Get stride failed!");
        return GRAPH_FAILED;
    }

    std::string dataFormat = "";
    if (ge::GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
        OP_LOGE(op.GetName().c_str(),"get data_format attr failed");
        return GRAPH_FAILED;
    }

    auto tensorDescIn = op.GetInputDesc(0);
    auto tensorDescW = op.GetInputDesc(1);

    auto shapeIn = tensorDescIn.GetShape();
    auto shapeW = tensorDescW.GetShape();

    auto dataTypeIn = tensorDescIn.GetDataType();

    if (!GetDimInFormat(op.GetName(), dataFormat, "N", nPosition)) {
        return GRAPH_FAILED;
    }

    if (!GetDimInFormat(op.GetName(), dataFormat, "C", cPosition)) {
        return GRAPH_FAILED;
    }

    if (!GetDimInFormat(op.GetName(), dataFormat, "H", hPosition)) {

        return GRAPH_FAILED;
    }
    if (!GetDimInFormat(op.GetName(), dataFormat, "W", wPosition)) {
        return GRAPH_FAILED;
    }

    // NC1HWC0(NCHW)
    inN = shapeIn.GetDim(nPosition);
    inC = shapeIn.GetDim(cPosition);
    inH = shapeIn.GetDim(hPosition);
    inW = shapeIn.GetDim(wPosition);

    Format filterFormat = tensorDescW.GetFormat();
    std::string filterFormatStr = format2str[filterFormat];
    int64_t fhPosition = 0;
    int64_t fwPosition = 0;
    if (!GetDimInFormat(op.GetName(), filterFormatStr, "H", fhPosition)) {
        return GRAPH_FAILED;
    }
    if (!GetDimInFormat(op.GetName(), filterFormatStr, "W", fwPosition)) {
        return GRAPH_FAILED;
    }

    filterH = shapeW.GetDim(fhPosition);
    filterW = shapeW.GetDim(fwPosition);

    dilationH = dilation.at(hPosition);
    dilationW = dilation.at(wPosition);
    strideH = stride.at(hPosition);
    strideW = stride.at(wPosition);

    if (false == GetPadDepthwiseConv2D(op, inH, inW, filterH, filterW, \
                              strideH, strideW, dilationH, dilationW, \
                              padtop, padbottom, padleft, padright)) {
        OP_LOGE(op.GetName().c_str(), "get pads attrs failed.");
        return GRAPH_FAILED;
    }

    effectiveFilterH = (filterH - 1) * dilationH + 1;
    effectiveFilterW = (filterW - 1) * dilationW + 1;
    outH = (inH + padtop + padbottom - effectiveFilterH) / strideH + 1;
    outW = (inW + padleft + padright - effectiveFilterW) / strideW + 1;

    vector<int64_t> shapeOut;
    TensorDesc tensordesc_output = op.GetOutputDesc("y");
    auto formatOut = tensordesc_output.GetFormat();

    // NC1HWC0(NCHW/NHWC)
    if (formatOut == FORMAT_NCHW) {
        shapeOut.push_back(inN);
        shapeOut.push_back(inC);
        shapeOut.push_back(outH);
        shapeOut.push_back(outW);
    } else if (formatOut == FORMAT_NHWC) {
        shapeOut.push_back(inN);
        shapeOut.push_back(outH);
        shapeOut.push_back(outW);
        shapeOut.push_back(inC);
    } else {
        OP_LOGE(op.GetName().c_str(), "output y format should be NCHW or NHWC."
                " actual is: %d", (int)formatOut);
        return GRAPH_FAILED;
    }

    tensordesc_output.SetShape(Shape(shapeOut));
    if(dataTypeIn == ge::DT_INT8){
        tensordesc_output.SetDataType(ge::DT_INT32);
    }else{
        tensordesc_output.SetDataType(dataTypeIn);
    }
    (void)op.UpdateOutputDesc("y", tensordesc_output);

    return GRAPH_SUCCESS;
}

//Registered inferfunction
COMMON_INFER_FUNC_REG(DepthwiseConv2D, DepthwiseConv2DInferShape);

//Registered verify function
VERIFY_FUNC_REG(DepthwiseConv2D, DepthwiseConv2DVerify);

static graphStatus VerifyDepthwiseConv2DbpPadding(ge::Operator& op) {
    std::string pad;
    if (GRAPH_SUCCESS == op.GetAttr("padding", pad)) {
        if (pad.compare("SAME") != 0 && pad.compare("VALID") != 0) {
            OP_LOGE(op.GetName().c_str(), "padding must be SAME or VALID. actual is: %s", pad.c_str());
            return GRAPH_FAILED;
        }
    } else {
        OP_LOGE(op.GetName().c_str(), "get padding failed.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

static graphStatus VerifyDepthwiseConv2DbpPads(ge::Operator& op) {
    std::vector<int64_t> pads;
    if (GRAPH_SUCCESS == op.GetAttr("pads", pads)) {
        if (pads.size() < 4) {
            OP_LOGE(op.GetName().c_str(), "op pads's size is illegal,pads.size:%d", pads.size());
            return GRAPH_FAILED;
        }
        OP_LOGI(op.GetName().c_str(), "op pads: top:%d,bottom:%d,left:%d,right:%d", pads[0], pads[1], pads[2], pads[3]);
        if (pads[0] < 0 || pads[1] < 0 || pads[2] < 0 || pads[3] < 0 ) {
            OP_LOGE(op.GetName().c_str(), "op get pads is illegal");
            return GRAPH_FAILED;
        }
    } else{
        OP_LOGE(op.GetName().c_str(), "op get pads failed.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

//----------------DepthwiseConv2DBackpropInputD Op-------------------
//Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(DepthwiseConv2DBackpropInputD, DepthwiseConv2DBackpropInputDVerify)
{
    std::vector<int64_t> input_size;
    input_size = GetAttrValue(op, "input_size");
    if (!CheckListEmpty(op.GetName(), input_size, "input_size")) {
        return GRAPH_FAILED;
    }

    std::vector<int64_t> strides;
    strides = GetAttrValue(op, "strides");
    if (!CheckListEmpty(op.GetName(), strides, "strides")) {
        return GRAPH_FAILED;
    }
    if (strides.size() != DIM_SIZE4) {
        OP_LOGE(op.GetName().c_str(), "strides must be NCHW!");
        return GRAPH_FAILED;
    }

    std::vector<int64_t> dilations;
    dilations = GetAttrValue(op, "dilations");
    if (!CheckListEmpty(op.GetName(), dilations, "dilations")) {
        return GRAPH_FAILED;
    }
    if (dilations.size() != DIM_SIZE4) {
        OP_LOGE(op.GetName().c_str(), "dilations must be NCHW!");
        return GRAPH_FAILED;
    }
    std::string data_format;
    if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
        if (data_format != "NCHW" && data_format != "NHWC") {
            OP_LOGE(op.GetName().c_str(),
                "attr data_format(%s) only support NCHW and NHWC",
                data_format.c_str());
            return GRAPH_FAILED;
        }
    }
    if (GRAPH_SUCCESS != VerifyDepthwiseConv2DbpPads(op)) {
        return GRAPH_FAILED;
    }

    if (op.GetInputDesc(0).GetDataType() != op.GetInputDesc(1).GetDataType()) {
        OP_LOGE(op.GetName().c_str(),
            "The type of filter and out_backprop must be same!");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DepthwiseConv2DBackpropInputDInferShape)
{
    std::vector<int64_t> input_size;
    input_size = GetAttrValue(op, "input_size");

    DataType output_dtype = op.GetInputDesc("out_backprop").GetDataType();
    TensorDesc tensordesc_output = op.GetOutputDesc("input_grad");
    tensordesc_output.SetShape(Shape(input_size));
    tensordesc_output.SetDataType(output_dtype);
    (void)op.UpdateOutputDesc("input_grad", tensordesc_output);

    std::vector<int64_t> strides;
    strides = GetAttrValue(op, "strides");
    if (!CheckListEmpty(op.GetName(), strides, "strides"))
    {
        return GRAPH_FAILED;
    }
    if (strides.size() != DIM_SIZE4)
    {
        OP_LOGE(op.GetName().c_str(),"strides must be NCHW!");
        return GRAPH_FAILED;
    }

    std::vector<int64_t> dilations;
    dilations = GetAttrValue(op, "dilations");
    if (!CheckListEmpty(op.GetName(), dilations, "dilations"))
    {
        return GRAPH_FAILED;
    }
    if (dilations.size() != DIM_SIZE4)
    {
        OP_LOGE(op.GetName().c_str(),"dilations must be NCHW!");
        return GRAPH_FAILED;
    }

    std::string dataFormat = "";
    if (ge::GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
        OP_LOGE(op.GetName().c_str(),"get data_format attr failed");
        return GRAPH_FAILED;
    }

    int64_t hPosition = 0;
    int64_t wPosition = 0;
    int64_t fhPosition = 0;
    int64_t fwPosition = 0;
    int64_t inH = 0;
    int64_t inW = 0;
    int64_t filterH = 0;
    int64_t filterW = 0;
    int64_t dilationH = 0;
    int64_t dilationW = 0;
    int64_t strideH = 0;
    int64_t strideW = 0;
    int64_t padtop = 0;
    int64_t padbottom = 0;
    int64_t padleft = 0;
    int64_t padright = 0;

    auto tensorDescW = op.GetInputDesc(0);
    auto shapeW = tensorDescW.GetShape();

    Format filterFormat = tensorDescW.GetFormat();
    std::string filterFormatStr = format2str[filterFormat];
    if (!GetDimInFormat(op.GetName(), filterFormatStr, "H", fhPosition)) {
        return GRAPH_FAILED;
    }
    if (!GetDimInFormat(op.GetName(), filterFormatStr, "W", fwPosition)) {
        return GRAPH_FAILED;
    }

    filterH = shapeW.GetDim(fhPosition);
    filterW = shapeW.GetDim(fwPosition);

    if (!GetDimInFormat(op.GetName(), dataFormat, "H", hPosition)) {
        return GRAPH_FAILED;
    }
    if (!GetDimInFormat(op.GetName(), dataFormat, "W", wPosition)) {
        return GRAPH_FAILED;
    }

    // NC1HWC0(NCHW)
    inH = input_size[hPosition];
    inW = input_size[wPosition];

    dilationH = dilations.at(hPosition);
    dilationW = dilations.at(wPosition);
    strideH = strides.at(hPosition);
    strideW = strides.at(wPosition);

    if (false == GetPadDepthwiseConv2D(op, inH, inW, filterH, filterW, \
                              strideH, strideW, dilationH, dilationW, \
                              padtop, padbottom, padleft, padright)) {
        OP_LOGE(op.GetName().c_str(), "update pads attrs failed.");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

//Registered inferfunction
COMMON_INFER_FUNC_REG(DepthwiseConv2DBackpropInputD, DepthwiseConv2DBackpropInputDInferShape);
//Registered verify function
VERIFY_FUNC_REG(DepthwiseConv2DBackpropInputD, DepthwiseConv2DBackpropInputDVerify);

//----------------DepthwiseConv2DBackpropInput Op-------------------
// Obtains the value of the constant tensor.
static void GetConstValue(const Tensor &const_tensor, const DataType &dtype,
                              std::vector<int64_t> &const_data)
{
    const uint8_t *constData = const_tensor.GetData();
    size_t size;
    if (dtype == ge::DT_INT32)
    {
        size = const_tensor.GetSize() / sizeof(int32_t);
        for (size_t i = 0; i < size; ++i)
        {
            const_data.push_back(*((int32_t*)constData + i));
        }
    }
    else
    {
        size = const_tensor.GetSize() / sizeof(int64_t);
        for (size_t i = 0; i < size; ++i)
        {
            const_data.push_back(*((int64_t*)constData + i));
        }
    }
}

//Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(DepthwiseConv2DBackpropInput, DepthwiseConv2DBackpropInputVerify)
{
    Tensor input_size_tensor;
    if (GRAPH_SUCCESS != op.GetInputConstData("input_size", input_size_tensor)) {
        OP_LOGE(op.GetName().c_str(), "Get constdata failed");
        return GRAPH_FAILED;
    }
    DataType dtype = op.GetInputDesc("input_size").GetDataType();

    std::vector<int64_t> input_size;
    GetConstValue(input_size_tensor, dtype, input_size);
    if (!CheckListEmpty(op.GetName(), input_size, "input_size")) {
        return GRAPH_FAILED;
    }

    std::vector<int64_t> strides;
    strides = GetAttrValue(op, "strides");
    if (!CheckListEmpty(op.GetName(), strides, "strides")) {
        return GRAPH_FAILED;
    }
    if (strides.size() != DIM_SIZE4) {
        OP_LOGE(op.GetName().c_str(), "strides must be NCHW!");
        return GRAPH_FAILED;
    }

    std::vector<int64_t> dilations;
    dilations = GetAttrValue(op, "dilations");
    if (!CheckListEmpty(op.GetName(), dilations, "dilations")) {
        return GRAPH_FAILED;
    }
    if (dilations.size() != DIM_SIZE4) {
        OP_LOGE(op.GetName().c_str(), "dilations must be NCHW!");
        return GRAPH_FAILED;
    }
    std::string data_format;
    if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
        if (data_format != "NCHW" && data_format != "NHWC") {
            OP_LOGE(op.GetName().c_str(),
                "attr data_format(%s) only support NCHW and NHWC",
                data_format.c_str());
            return GRAPH_FAILED;
        }
    }

    if (VerifyDepthwiseConv2DbpPadding(op) == GRAPH_SUCCESS ||
        VerifyDepthwiseConv2DbpPads(op) == GRAPH_SUCCESS) {
        return GRAPH_SUCCESS;
    }
    else {
        return GRAPH_FAILED;
    }

    if (op.GetInputDesc(1).GetDataType() != op.GetInputDesc(2).GetDataType()) {
        OP_LOGE(op.GetName().c_str(),
            "The type of filter and out_backprop must be same!");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DepthwiseConv2DBackpropInputInferShape)
{
    Tensor input_size_tensor;
    if (ge::GRAPH_SUCCESS != op.GetInputConstData("input_size", input_size_tensor)) {
      OP_LOGE(op.GetName().c_str(), "get constdata failed");
      return GRAPH_FAILED;
    }
    DataType dtype = op.GetInputDesc("input_size").GetDataType();
    std::vector<int64_t> input_size;
    GetConstValue(input_size_tensor, dtype, input_size);

    DataType output_dtype = op.GetInputDesc("out_backprop").GetDataType();
    TensorDesc tensordesc_output = op.GetOutputDesc("input_grad");
    tensordesc_output.SetShape(Shape(input_size));
    tensordesc_output.SetDataType(output_dtype);
    (void)op.UpdateOutputDesc("input_grad", tensordesc_output);

    std::vector<int64_t> strides;
    strides = GetAttrValue(op, "strides");
    if (!CheckListEmpty(op.GetName(), strides, "strides"))
    {
        return GRAPH_FAILED;
    }
    if (strides.size() != DIM_SIZE4)
    {
        OP_LOGE(op.GetName().c_str(),"strides must be NCHW!");
        return GRAPH_FAILED;
    }

    std::vector<int64_t> dilations;
    dilations = GetAttrValue(op, "dilations");
    if (!CheckListEmpty(op.GetName(), dilations, "dilations"))
    {
        return GRAPH_FAILED;
    }
    if (dilations.size() != DIM_SIZE4)
    {
        OP_LOGE(op.GetName().c_str(),"dilations must be NCHW!");
        return GRAPH_FAILED;
    }

    std::string dataFormat = "";
    if (ge::GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
        OP_LOGE(op.GetName().c_str(),"get data_format attr failed");
        return GRAPH_FAILED;
    }

    int64_t hPosition = 0;
    int64_t wPosition = 0;
    int64_t fhPosition = 0;
    int64_t fwPosition = 0;
    int64_t inH = 0;
    int64_t inW = 0;
    int64_t filterH = 0;
    int64_t filterW = 0;
    int64_t dilationH = 0;
    int64_t dilationW = 0;
    int64_t strideH = 0;
    int64_t strideW = 0;
    int64_t padtop = 0;
    int64_t padbottom = 0;
    int64_t padleft = 0;
    int64_t padright = 0;

    auto tensorDescW = op.GetInputDesc(1);
    auto shapeW = tensorDescW.GetShape();

    Format filterFormat = tensorDescW.GetFormat();
    std::string filterFormatStr = format2str[filterFormat];
    if (!GetDimInFormat(op.GetName(), filterFormatStr, "H", fhPosition)) {
        return GRAPH_FAILED;
    }
    if (!GetDimInFormat(op.GetName(), filterFormatStr, "W", fwPosition)) {
        return GRAPH_FAILED;
    }

    filterH = shapeW.GetDim(fhPosition);
    filterW = shapeW.GetDim(fwPosition);

    if (!GetDimInFormat(op.GetName(), dataFormat, "H", hPosition)) {
        return GRAPH_FAILED;
    }
    if (!GetDimInFormat(op.GetName(), dataFormat, "W", wPosition)) {
        return GRAPH_FAILED;
    }

    // NC1HWC0(NCHW)
    inH = input_size[hPosition];
    inW = input_size[wPosition];

    dilationH = dilations.at(hPosition);
    dilationW = dilations.at(wPosition);
    strideH = strides.at(hPosition);
    strideW = strides.at(wPosition);

    if (false == GetPadDepthwiseConv2D(op, inH, inW, filterH, filterW, \
                              strideH, strideW, dilationH, dilationW, \
                              padtop, padbottom, padleft, padright)) {
        OP_LOGE(op.GetName().c_str(), "update pads attrs failed.");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

//Registered inferfunction
COMMON_INFER_FUNC_REG(DepthwiseConv2DBackpropInput, DepthwiseConv2DBackpropInputInferShape);
//Registered verify function
VERIFY_FUNC_REG(DepthwiseConv2DBackpropInput, DepthwiseConv2DBackpropInputVerify);

//----------------DepthwiseConv2DBackpropFilterD Op-------------------
//Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(DepthwiseConv2DBackpropFilterD, DepthwiseConv2DBackpropFilterDVerify)
{
    std::vector<int64_t> filter_size;
    filter_size = GetAttrValue(op, "filter_size");
    if (!CheckListEmpty(op.GetName(), filter_size, "filter_size")) {
        return GRAPH_FAILED;
    }
    if (filter_size.size() != DIM_SIZE4) {
        OP_LOGE(op.GetName().c_str(), "Filter_size must be HWCK!");
        return GRAPH_FAILED;
    }

    std::vector<int64_t> strides;
    strides = GetAttrValue(op, "strides");
    if (!CheckListEmpty(op.GetName(), strides, "strides")) {
        return GRAPH_FAILED;
    }
    if (strides.size() != DIM_SIZE4) {
        OP_LOGE(op.GetName().c_str(), "strides must be 4!");
        return GRAPH_FAILED;
    }

    std::vector<int64_t> dilations;
    dilations = GetAttrValue(op, "dilations");
    if (!CheckListEmpty(op.GetName(), dilations, "dilations")) {
        return GRAPH_FAILED;
    }
    if (dilations.size() != DIM_SIZE4) {
        OP_LOGE(op.GetName().c_str(), "dilations must be 4!");
        return GRAPH_FAILED;
    }
    std::string data_format;
    if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
        if (data_format != "NCHW" && data_format != "NHWC") {
            OP_LOGE(op.GetName().c_str(),
                "attr data_format(%s) only support NCHW and NHWC",
                data_format.c_str());
            return GRAPH_FAILED;
        }
    }
    if (GRAPH_SUCCESS != VerifyDepthwiseConv2DbpPads(op)) {
        return GRAPH_FAILED;
    }

    if (op.GetInputDesc(0).GetDataType() != op.GetInputDesc(1).GetDataType()) {
        OP_LOGE(op.GetName().c_str(), "The type of input and out_backprop must be same!");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DepthwiseConv2DBackpropFilterDInferShape)
{
    std::vector<int64_t> filter_size;
    filter_size = GetAttrValue(op, "filter_size");

    DataType output_dtype = op.GetInputDesc("out_backprop").GetDataType();
    TensorDesc tensordesc_output = op.GetOutputDesc("filter_grad");
    tensordesc_output.SetShape(Shape(filter_size));
    tensordesc_output.SetDataType(output_dtype);
    (void)op.UpdateOutputDesc("filter_grad", tensordesc_output);

    std::vector<int64_t> strides;
    strides = GetAttrValue(op, "strides");
    if (!CheckListEmpty(op.GetName(), strides, "strides")) {
        return GRAPH_FAILED;
    }
    if (strides.size() != DIM_SIZE4) {
        OP_LOGE(op.GetName().c_str(),"strides must be NCHW!");
        return GRAPH_FAILED;
    }

    std::vector<int64_t> dilations;
    dilations = GetAttrValue(op, "dilations");
    if (!CheckListEmpty(op.GetName(), dilations, "dilations")) {
        return GRAPH_FAILED;
    }
    if (dilations.size() != DIM_SIZE4) {
        OP_LOGE(op.GetName().c_str(),"dilations must be NCHW!");
        return GRAPH_FAILED;
    }

    std::string dataFormat = "";
    if (ge::GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
        OP_LOGE(op.GetName().c_str(),"get data_format attr failed");
        return GRAPH_FAILED;
    }

    int64_t hPosition = 0;
    int64_t wPosition = 0;
    int64_t fhPosition = 0;
    int64_t fwPosition = 0;
    int64_t inH = 0;
    int64_t inW = 0;
    int64_t filterH = 0;
    int64_t filterW = 0;
    int64_t dilationH = 0;
    int64_t dilationW = 0;
    int64_t strideH = 0;
    int64_t strideW = 0;
    int64_t padtop = 0;
    int64_t padbottom = 0;
    int64_t padleft = 0;
    int64_t padright = 0;

    Format filterFormat = tensordesc_output.GetFormat();
    std::string filterFormatStr = format2str[filterFormat];
    if (!GetDimInFormat(op.GetName(), filterFormatStr, "H", fhPosition)) {
        return GRAPH_FAILED;
    }
    if (!GetDimInFormat(op.GetName(), filterFormatStr, "W", fwPosition)) {
        return GRAPH_FAILED;
    }
    filterH = filter_size[fhPosition];
    filterW = filter_size[fwPosition];

    auto tensorDescIn = op.GetInputDesc(0);
    auto shapeIn = tensorDescIn.GetShape();

    if (!GetDimInFormat(op.GetName(), dataFormat, "H", hPosition)) {
        return GRAPH_FAILED;
    }
    if (!GetDimInFormat(op.GetName(), dataFormat, "W", wPosition)) {
        return GRAPH_FAILED;
    }

    // NC1HWC0(NCHW)
    inH = shapeIn.GetDim(hPosition);
    inW = shapeIn.GetDim(wPosition);

    dilationH = dilations.at(hPosition);
    dilationW = dilations.at(wPosition);
    strideH = strides.at(hPosition);
    strideW = strides.at(wPosition);

    if (false == GetPadDepthwiseConv2D(op, inH, inW, filterH, filterW, \
                              strideH, strideW, dilationH, dilationW, \
                              padtop, padbottom, padleft, padright)) {
        OP_LOGE(op.GetName().c_str(), "update pads attrs failed.");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

//Registered inferfunction
COMMON_INFER_FUNC_REG(DepthwiseConv2DBackpropFilterD, DepthwiseConv2DBackpropFilterDInferShape);
//Registered verify function
VERIFY_FUNC_REG(DepthwiseConv2DBackpropFilterD, DepthwiseConv2DBackpropFilterDVerify);


//----------------DepthwiseConv2DBackpropFilter Op-------------------
//Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(DepthwiseConv2DBackpropFilter, DepthwiseConv2DBackpropFilterVerify)
{
    Tensor filter_size_tensor;
    if (GRAPH_SUCCESS != op.GetInputConstData("filter_size", filter_size_tensor)) {
        OP_LOGE(op.GetName().c_str(), "Get constdata failed");
        return GRAPH_FAILED;
    }
    DataType dtype = op.GetInputDesc("filter_size").GetDataType();
    std::vector<int64_t> filter_size;
    GetConstValue(filter_size_tensor, dtype, filter_size);
    if (!CheckListEmpty(op.GetName(), filter_size, "filter_size")) {
        return GRAPH_FAILED;
    }
    if (filter_size.size() != DIM_SIZE4) {
        OP_LOGE(op.GetName().c_str(), "Filter_size must be 4!");
        return GRAPH_FAILED;
    }

    std::vector<int64_t> strides;
    strides = GetAttrValue(op, "strides");
    if (!CheckListEmpty(op.GetName(), strides, "strides")) {
        return GRAPH_FAILED;
    }
    if (strides.size() != DIM_SIZE4) {
        OP_LOGE(op.GetName().c_str(), "strides must be 4!");
        return GRAPH_FAILED;
    }

    std::vector<int64_t> dilations;
    dilations = GetAttrValue(op, "dilations");
    if (!CheckListEmpty(op.GetName(), dilations, "dilations")) {
        return GRAPH_FAILED;
    }
    if (dilations.size() != DIM_SIZE4) {
        OP_LOGE(op.GetName().c_str(), "dilations must be 4!");
        return GRAPH_FAILED;
    }
    std::string data_format;
    if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
        if (data_format != "NCHW" && data_format != "NHWC") {
            OP_LOGE(op.GetName().c_str(),
                "attr data_format(%s) only support NCHW and NHWC",
                data_format.c_str());
            return GRAPH_FAILED;
        }
    }
    if (GRAPH_SUCCESS == VerifyDepthwiseConv2DbpPadding(op) ||
        GRAPH_SUCCESS == VerifyDepthwiseConv2DbpPads(op)) {
        return GRAPH_SUCCESS;
    }
    else {
        return GRAPH_FAILED;
    }

    if (op.GetInputDesc(0).GetDataType() != op.GetInputDesc(2).GetDataType()) {
        OP_LOGE(op.GetName().c_str(),
            "The type of input and out_backprop must be same!");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DepthwiseConv2DBackpropFilterInferShape)
{
    Tensor filter_size_tensor;
    if (GRAPH_SUCCESS != op.GetInputConstData("filter_size", filter_size_tensor)) {
     OP_LOGE(op.GetName().c_str(), "Get constdata failed");
     return GRAPH_FAILED;
    }
    DataType dtype = op.GetInputDesc("filter_size").GetDataType();
    std::vector<int64_t> filter_size;
    GetConstValue(filter_size_tensor, dtype, filter_size);

    DataType output_dtype = op.GetInputDesc("out_backprop").GetDataType();
    TensorDesc tensordesc_output = op.GetOutputDesc("filter_grad");
    tensordesc_output.SetShape(Shape(filter_size));
    tensordesc_output.SetDataType(output_dtype);
    (void)op.UpdateOutputDesc("filter_grad", tensordesc_output);

    std::vector<int64_t> strides;
    strides = GetAttrValue(op, "strides");
    if (!CheckListEmpty(op.GetName(), strides, "strides")) {
        return GRAPH_FAILED;
    }
    if (strides.size() != DIM_SIZE4) {
        OP_LOGE(op.GetName().c_str(),"strides must be 4!");
        return GRAPH_FAILED;
    }

    std::vector<int64_t> dilations;
    dilations = GetAttrValue(op, "dilations");
    if (!CheckListEmpty(op.GetName(), dilations, "dilations")) {
        return GRAPH_FAILED;
    }
    if (dilations.size() != DIM_SIZE4) {
        OP_LOGE(op.GetName().c_str(),"dilations must be 4!");
        return GRAPH_FAILED;
    }

    std::string dataFormat = "";
    if (ge::GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
        OP_LOGE(op.GetName().c_str(),"get data_format attr failed");
        return GRAPH_FAILED;
    }

    int64_t hPosition = 0;
    int64_t wPosition = 0;
    int64_t fhPosition = 0;
    int64_t fwPosition = 0;
    int64_t inH = 0;
    int64_t inW = 0;
    int64_t filterH = 0;
    int64_t filterW = 0;
    int64_t dilationH = 0;
    int64_t dilationW = 0;
    int64_t strideH = 0;
    int64_t strideW = 0;
    int64_t padtop = 0;
    int64_t padbottom = 0;
    int64_t padleft = 0;
    int64_t padright = 0;

    Format filterFormat = tensordesc_output.GetFormat();
    std::string filterFormatStr = format2str[filterFormat];
    if (!GetDimInFormat(op.GetName(), filterFormatStr, "H", fhPosition)) {
        return GRAPH_FAILED;
    }
    if (!GetDimInFormat(op.GetName(), filterFormatStr, "W", fwPosition)) {
        return GRAPH_FAILED;
    }
    filterH = filter_size[fhPosition];
    filterW = filter_size[fwPosition];

    auto tensorDescIn = op.GetInputDesc(0);
    auto shapeIn = tensorDescIn.GetShape();

    if (!GetDimInFormat(op.GetName(), dataFormat, "H", hPosition)) {
        return GRAPH_FAILED;
    }
    if (!GetDimInFormat(op.GetName(), dataFormat, "W", wPosition)) {
        return GRAPH_FAILED;
    }

    // NC1HWC0(NCHW)
    inH = shapeIn.GetDim(hPosition);
    inW = shapeIn.GetDim(wPosition);

    dilationH = dilations.at(hPosition);
    dilationW = dilations.at(wPosition);
    strideH = strides.at(hPosition);
    strideW = strides.at(wPosition);

    if (false == GetPadDepthwiseConv2D(op, inH, inW, filterH, filterW, \
                              strideH, strideW, dilationH, dilationW, \
                              padtop, padbottom, padleft, padright)) {
        OP_LOGE(op.GetName().c_str(), "update pads attrs failed.");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

//Registered inferfunction
COMMON_INFER_FUNC_REG(DepthwiseConv2DBackpropFilter, DepthwiseConv2DBackpropFilterInferShape);
//Registered verify function
VERIFY_FUNC_REG(DepthwiseConv2DBackpropFilter, DepthwiseConv2DBackpropFilterVerify);

// --------------------------------BiasAddGrad---------------------------------
IMPLEMT_COMMON_INFERFUNC(BiasAddGradInferShape) {
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  ge::TensorDesc tensor_desc_x = op.GetInputDesc("x");
  ge::Shape shape = tensor_desc_x.GetShape();
  size_t dim_num = shape.GetDimNum();

  std::string data_format;
  if (ge::GRAPH_SUCCESS != op.GetAttr("data_format", data_format)) {
      OpsGetAttrErrReport(op.GetName(), "data_format");
      printf("[Plugin][ERROR]The bias add grad op GetOpAttr"
             "data_format failed!");
      return GRAPH_FAILED;
  }

  std::vector<int64_t> dim_vec;
  if (data_format == "NHWC") {
      if (dim_num < DIM_SIZE2 || dim_num > DIM_SIZE8) {
          OpsInputShapeDimErrReport(op.GetName(), "x", Strcat(DIM_SIZE8), Strcat(DIM_SIZE2), Strcat(dim_num));
          OP_LOGE("[Plugin][ERROR]The bias add grad op dimension(%lu) is not"
                  "supported when format is NHWC!", dim_num);
            return GRAPH_FAILED;
      }
      dim_vec.push_back(shape.GetDim(dim_num - 1));
  } else if (data_format == "NCHW") {
      if (dim_num < DIM_SIZE2) {
        OP_LOGE("[Plugin][ERROR]The bias add grad op dimension(%lu) is not"
                "supported when format is NCHW!", dim_num);
        return GRAPH_FAILED;
      }
      dim_vec.push_back(shape.GetDim(1));
    } else {
      string expected_format_list = Strcat("NHWC, NCHW");
      OpsInputFormatErrReport(op.GetName(), "x", expected_format_list, data_format);
      OP_LOGE("[Plugin][ERROR]The bias add grad op data format(%s) is not"
                "supported!", data_format.c_str());
      return GRAPH_FAILED;
      }

  tensordesc_output.SetShape(ge::Shape(dim_vec));
  tensordesc_output.SetRealDimCnt(1);

  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BiasAddGrad, BiasAddGradInferShape);
// ------------------------------BiasAddGrad END-----------------------------------

//============================Conv2Dbackprop===============================
#define ALIGN_CONV2DBP(x_1, x_2) ((((x_1) + (x_2) - 1) / (x_2)) * (x_2))

#define CHECK_POSITION(position) \
{                                   \
    if (position < 0) {        \
        OP_LOGE(op.GetName().c_str(), "get position failed:%s:%d", #position, position); \
        return false;     \
    }                     \
}

static bool getStrideDilationHW(ge::Operator& op, int32_t& stride_h, int32_t& stride_w,
                                int32_t& dilation_h, int32_t& dilation_w) {
    const int32_t DIM_SIZE_LIMIT = 4;
    std::string dataFormat = "";
    int32_t hPosition = 0;
    int32_t wPosition = 0;
    if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", dataFormat)) {
        if (dataFormat != "NCHW" && dataFormat != "NHWC") {
            OP_LOGE(op.GetName().c_str(), "conv2DBackprop's dataFormat error, should be HCHW OR NHWC!");
            map<string, string> err_map;
            err_map["param"] = "data_Format";
            err_map["op_name"] = "conv2Dbp";
            err_map["expected_format_list"] = "HCHW OR NHWC";
            err_map["format"] = dataFormat;
            std::string report_error_code = "E50002";
            ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return false;
        }

    } else{
        OP_LOGE(op.GetName().c_str(), "GetOpAttr dataFormat failed!");
        map<string, string> err_map;
        err_map["op_name"] = "conv2Dbp";
        err_map["param_name"] = "data_Format";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }
    // get H & W position
    hPosition = dataFormat.find("H");
    wPosition = dataFormat.find("W");
    if (hPosition < 0 || wPosition < 0) {
        OP_LOGE(op.GetName().c_str(), "Get hPosition or wPosition failed!");
        map<string, string> err_map;
        err_map["op_name"] = "conv2Dbp";
        err_map["param_name"] = "hPosition and wPosition of dataFormat";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    std::vector<int32_t> strideList;
    if (GRAPH_SUCCESS == op.GetAttr("strides", strideList)) {
        if (strideList.empty()) {
            OP_LOGE(op.GetName().c_str(), "strideList from op is empty!");
            map<string, string> err_map;
            err_map["op_name"] = "conv2Dbp";
            err_map["param_name"] = "stride_List";
            std::string report_error_code = "E50030";
            ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return false;
        }
        stride_h = strideList[hPosition];
        stride_w = strideList[wPosition];

    } else {
        OP_LOGE(op.GetName().c_str(), "Get strides list failed!");
        map<string, string> err_map;
        err_map["op_name"] = "conv2Dbp";
        err_map["param_name"] = "stride_List";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }
     // check dilations shape
    std::vector<int32_t> dilationsList;
    if (GRAPH_SUCCESS == op.GetAttr("dilations", dilationsList)) {
        if (dilationsList.size() != DIM_SIZE_LIMIT) {
            OP_LOGE(op.GetName().c_str(), "dilationsList list should be 4d");
            map<string, string> err_map;
            err_map["op_name"] = "Conv2DBackpropInput";
            err_map["param_name"] = "dilationsList";
            err_map["expected_length"] = std::to_string(DIM_SIZE_LIMIT);
            err_map["length"] = std::to_string(dilationsList.size());
            std::string report_error_code = "E50035";
            (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return false;
        }
        dilation_h = dilationsList[hPosition];
        dilation_w = dilationsList[wPosition];

    } else {
        OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DBackpropInput";
        err_map["param_name"] = "dilations";
        std::string report_error_code = "E50030";
        (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }
    return true;
}

template<typename T1, typename T2>
static bool SetPadListByPaddingConv2dbp(ge::Operator& op, std::vector<T1>& inputSizes, Format inputFormat,
                                        std::vector<T2>& filterSizes, Format filterFormat) {
    OP_LOGI(op.GetName().c_str(), "SetPadListByPaddingConv2dbp begin.");
    if (filterSizes.size() < 4 || inputSizes.size() < 4) {
        OP_LOGE(op.GetName().c_str(), "filter_sizes or inputSizes is illegal");
        map<string, string> err_map;
        err_map["op_name"] = "conv2Dbp";
        err_map["param_name"] = "filterSizes and inputSizes";
        err_map["expected_length"] = "4";
        err_map["length"] = std::to_string(filterSizes.size()) + " and "\
                            + std::to_string(inputSizes.size());
        std::string report_error_code = "E50035";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }
    CHECK_FORMAT(inputFormat);
    CHECK_FORMAT(filterFormat);

    int32_t stride_h = 0;
    int32_t stride_w = 0;
    int32_t dilation_h = 0;
    int32_t dilation_w = 0;
    if (false == getStrideDilationHW(op, stride_h, stride_w, dilation_h, dilation_w)) {
        OP_LOGE(op.GetName().c_str(), "op get strides failed.");
        map<string, string> err_map;
        err_map["op_name"] = "conv2Dbp";
        err_map["param_name"] = "stridet";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    std::string inputFormatStr = format2str[inputFormat];
    int32_t hInputPosition = inputFormatStr.find("H");
    CHECK_POSITION(hInputPosition);
    int32_t wInputPosition = inputFormatStr.find("W");
    CHECK_POSITION(wInputPosition);
    int32_t dx_h = inputSizes[hInputPosition];
    int32_t dx_w = inputSizes[wInputPosition];

    std::string filterFormatStr = format2str[filterFormat];
    int32_t hFilterPosition = filterFormatStr.find("H");
    CHECK_POSITION(hFilterPosition);
    int32_t wFilterPosition = filterFormatStr.find("W");
    CHECK_POSITION(wFilterPosition);
    int32_t filter_h = filterSizes[hFilterPosition];
    int32_t filter_w = filterSizes[wFilterPosition];

    int32_t filter_dilation_h = (filter_h - 1) * dilation_h + 1;
    int32_t filter_dilation_w = (filter_w - 1) * dilation_w + 1;
    std::string padding;
    std::vector<int32_t> pads;
    if (GRAPH_SUCCESS == op.GetAttr("padding", padding)) {
        int pad_h = 0;
        int32_t pad_up = 0;
        int32_t pad_down = 0;
        int pad_w = 0;
        int32_t pad_left = 0;
        int32_t pad_right = 0;
        if (padding == "SAME") {
            pad_h = std::max(ALIGN_CONV2DBP(dx_h, stride_h) - stride_h + filter_dilation_h - dx_h, 0);
            pad_up = pad_h / 2 ;
            pad_down = pad_h - pad_up;
            pad_w = std::max(ALIGN_CONV2DBP(dx_w, stride_w) - stride_w + filter_dilation_w - dx_w, 0);
            pad_left = pad_w / 2 ;
            pad_right = pad_w - pad_left;
        }
        pads.push_back(pad_up);
        pads.push_back(pad_down);
        pads.push_back(pad_left);
        pads.push_back(pad_right);

        op.SetAttr("pads", pads);
    }

    if (GRAPH_SUCCESS == op.GetAttr("pads", pads)) {
        if (pads.size() < 4) {
            OP_LOGE(op.GetName().c_str(), "op pads's size is illegal.");
            map<string, string> err_map;
            err_map["op_name"] = "conv2Dbp";
            err_map["param_name"] = "pads";
            err_map["expected_length"] = "4";
            err_map["length"] = std::to_string(pads.size()) ;
            std::string report_error_code = "E50035";
            ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return false;
        }
        if (pads[0] < 0 || pads[1] < 0 || pads[2] < 0 || pads[3] < 0 ) {
            OP_LOGE(op.GetName().c_str(), "op get pads is illegal");
            map<string, string> err_map;
            err_map["op_name"] = "conv2Dbp";
            err_map["param_name"] = "pads";
            err_map["expected_value"] = ">= 0";
            err_map["input_value"] = "[" + std::to_string(pads[0]) + ", "\
                                     + std::to_string(pads[1]) + ", "\
                                     + std::to_string(pads[2]) + ", "\
                                     + std::to_string(pads[3]) + ']';
            std::string report_error_code = "E50029";
            (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return false;
        }
    } else {
        OP_LOGE(op.GetName().c_str(), "op get pads failed.");
        map<string, string> err_map;
        err_map["op_name"] = "conv2Dbp";
        err_map["param_name"] = "pads";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }
    OP_LOGI(op.GetName().c_str(), "op set pads succ.");
    return true;
}

static graphStatus VerifyConvPadding(ge::Operator& op) {
    std::string pad;
    if (GRAPH_SUCCESS == op.GetAttr("padding", pad)) {
        if (pad.compare("SAME") != 0 && pad.compare("VALID") != 0) {
            OP_LOGE(op.GetName().c_str(), "padding must be SAME or VALID.");
            return GRAPH_FAILED;
        }
    } else {
        OP_LOGE(op.GetName().c_str(), "get padding failed.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

static graphStatus VerifyConv2dbpPads(ge::Operator& op) {
    std::vector<int> pads;
    if (GRAPH_SUCCESS == op.GetAttr("pads", pads)) {
        if (pads.size() < 4) {
            OP_LOGE(op.GetName().c_str(), "op pads's size is illegal.");
            map<string, string> err_map;
            err_map["op_name"] = "conv2Dbp";
            err_map["param_name"] = "pads";
            err_map["expected_length"] = "4";
            err_map["length"] = std::to_string(pads.size()) ;
            std::string report_error_code = "E50035";
            ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return GRAPH_FAILED;
        }
        if (pads[0] < 0 || pads[1] < 0 || pads[2] < 0 || pads[3] < 0 ) {
            OP_LOGE(op.GetName().c_str(), "op get pads is illegal");
            map<string, string> err_map;
            err_map["op_name"] = "conv2Dbp";
            err_map["param_name"] = "pads";
            err_map["expected_value"] = ">= 0";
            err_map["input_value"] = "[" + std::to_string(pads[0]) + ", "\
                                     + std::to_string(pads[1]) + ", "\
                                     + std::to_string(pads[2]) + ", "\
                                     + std::to_string(pads[3]) + ']';
            std::string report_error_code = "E50029";
            ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return GRAPH_FAILED;
        }
    } else{
        OP_LOGE(op.GetName().c_str(), "op get pads failed.");
        map<string, string> err_map;
        err_map["op_name"] = "conv2Dbp";
        err_map["param_name"] = "pads";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}
//----------------Conv2DBackpropInput-------------------
static graphStatus VerifyConv2dbpInputCommon(ge::Operator& op) {
    auto filterDesc = op.GetInputDesc("filter");
    auto outBackpropDesc = op.GetInputDesc("out_backprop");

    auto filterDtype = filterDesc.GetDataType();
    auto outBackpropDtype = outBackpropDesc.GetDataType();
    auto filterShape = filterDesc.GetShape().GetDims();
    auto outBackpropShape = outBackpropDesc.GetShape().GetDims();

    const int32_t DIM_SIZE_LIMIT = 4;
    const int32_t DIM_STRIDES_LIMIT = 4;

    //check input dtype
    if (filterDtype != outBackpropDtype)
    {
        OP_LOGE(op.GetName().c_str(), "filter's dtype should equal to out_backprop's dtype.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DBackpropInput";
        err_map["param1_name"] = "filter";
        err_map["param2_name"] = "out_backprop";
        err_map["param1_value"] = std::to_string(filterDtype);
        err_map["param2_value"] = std::to_string(outBackpropDtype);
        err_map["attr_name"] = "dtype";
	std::string report_error_code = "E50031";
        (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // check input tensor shape
    if (filterShape.size() != DIM_SIZE_LIMIT)
    {
        OP_LOGE(op.GetName().c_str(), "filter's shape should be 4d.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DBackpropInput";
        err_map["param_name"] = "filterShape";
        err_map["expected_length"] = "4";
        err_map["length"] = std::to_string(filterShape.size());
        std::string report_error_code = "E50035";
        (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    if (outBackpropShape.size() != DIM_SIZE_LIMIT)
    {
        OP_LOGE(op.GetName().c_str(), "out_backprop's shape should be 4d.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DBackpropInput";
        err_map["param_name"] = "out_backprop's shape";
        err_map["expected_length"] = "4";
        err_map["length"] = std::to_string(outBackpropShape.size());
        std::string report_error_code = "E50035";
        (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // check strides shape
    std::vector<int32_t> strideList;
    if (GRAPH_SUCCESS == op.GetAttr("strides", strideList)) {
        if (strideList.size() != DIM_STRIDES_LIMIT) {
            OP_LOGE(op.GetName().c_str(), "strides should be 4d.");
            map<string, string> err_map;
            err_map["op_name"] = "Conv2DBackpropInput";
            err_map["param_name"] = "strides's shape";
            err_map["expected_length"] = std::to_string(DIM_STRIDES_LIMIT);
            err_map["length"] = std::to_string(strideList.size());
            std::string report_error_code = "E50035";
            return GRAPH_FAILED;
        }
    } else {
        OP_LOGE(op.GetName().c_str(), "get strides list failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DBackpropInput";
        err_map["param_name"] = "strides list";
        std::string report_error_code = "E50030";
        (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // check dilations shape
    std::vector<int32_t> dilationsList;
    if (GRAPH_SUCCESS == op.GetAttr("dilations", dilationsList)) {
        if (dilationsList.size() != DIM_SIZE_LIMIT) {
            OP_LOGE(op.GetName().c_str(), "dilationsList list should be 4d");
            map<string, string> err_map;
            err_map["op_name"] = "Conv2DBackpropInput";
            err_map["param_name"] = "dilationsList";
            err_map["expected_length"] = std::to_string(DIM_SIZE_LIMIT);
            err_map["length"] = std::to_string(dilationsList.size());
            std::string report_error_code = "E50035";
            (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return GRAPH_FAILED;
        }
    } else {
        OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DBackpropInput";
        err_map["param_name"] = "dilations";
        std::string report_error_code = "E50030";
        (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(Conv2DBackpropInput, Conv2DBackpropInputInfer)
{
    OP_LOGI(op.GetName().c_str(), "Enter Conv2DBackpropInput inferfunction!");

    auto inputSizesDesc = op.GetInputDesc("input_size");
    auto yDesc = op.GetOutputDesc("y");

    Tensor inputSizesTensor;
    if (GRAPH_SUCCESS != op.GetInputConstData("input_size", inputSizesTensor)) {
        OP_LOGE(op.GetName().c_str(), "get input_size tensor failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DBackpropInput";
        err_map["param_name"] = "input_size";
        std::string report_error_code = "E50030";
        (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
    // get shape for output from input_size
    DataType dtype = inputSizesDesc.GetDataType();
    std::vector<int64_t> inputSizes;
    GetConstValue(inputSizesTensor, dtype, inputSizes);

    int64_t groups = 1;
    if (GRAPH_SUCCESS != op.GetAttr("groups", groups)) {
        OP_LOGI(op.GetName().c_str(), "no groups setting, use groups as 1");
    }

    // set dtype of output desc
    auto outBackpropDtype = op.GetInputDesc("out_backprop").GetDataType();
    yDesc.SetDataType(outBackpropDtype);
    // set shape of output desc, input_size should match the format of y
    std::vector<int64_t> yShape;
    yShape.push_back(inputSizes[0]);
    yShape.push_back(inputSizes[1]);
    yShape.push_back(inputSizes[2]);
    yShape.push_back(inputSizes[3]);
    yDesc.SetShape(ge::Shape(yShape));

    // update output desc
    if (GRAPH_SUCCESS != op.UpdateOutputDesc("y", yDesc)) {
        OP_LOGE(op.GetName().c_str(), "update output desc failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DBackpropInput";
        err_map["param_name"] = "updating result";
        err_map["rule_desc"] = "updata OutputDesc";
        err_map["param_value"] = "failed";
        std::string report_error_code = "E50012";
        (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    std::vector<int64_t> filterSizes = op.GetInputDesc("filter").GetShape().GetDims();
    Format filterFormat = op.GetInputDesc("filter").GetFormat();
    Format inputFormat = yDesc.GetFormat();
    CHECK_FORMAT(filterFormat);
    CHECK_FORMAT(inputFormat);

    // update pads list by padding[SAME,VALID]
    if (false == SetPadListByPaddingConv2dbp(op, inputSizes, inputFormat, filterSizes, filterFormat)) {
        OP_LOGE(op.GetName().c_str(), "update pads list by padding failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DBackpropInput";
        err_map["param_name"] = "updding result";
        err_map["rule_desc"] = "updata pads list by padding";
        err_map["param_value"] = "failed";
        std::string report_error_code = "E50012";
        (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    OP_LOGI(op.GetName().c_str(), "Leaving Conv2DBackpropInput inferfunction!");
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv2DBackpropInput, Conv2DBackpropInputVerify)
{
    OP_LOGI(op.GetName().c_str(), "Enter Conv2DBackpropInput verifyfunction!");
    if (GRAPH_SUCCESS != VerifyConv2dbpInputCommon(op)) {
        return GRAPH_FAILED;
    }
    // check padding value
    if (GRAPH_SUCCESS == VerifyConvPadding(op) || GRAPH_SUCCESS == VerifyConv2dbpPads(op)) {
        OP_LOGI(op.GetName().c_str(), "Leaving Conv2DBackpropInput verifyfunction!");
        return GRAPH_SUCCESS;
    }else{
        OP_LOGE(op.GetName().c_str(), "Leaving Conv2DBackpropInput verifyfunction!");
        map<string, string> err_map;
        err_map["param_name"] = "verify pads and verify padding";
        err_map["op_name"] = "Conv2DBackpropInput";
        err_map["excepted_value"] = GRAPH_SUCCESS;
        err_map["output_value"] = VerifyConvPadding(op) + " and "\
                                  + VerifyConv2dbpPads(op);
        std::string report_error_code = "E50029";
        (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
}

INFER_FUNC_REG(Conv2DBackpropInput, Conv2DBackpropInputInfer);
VERIFY_FUNC_REG(Conv2DBackpropInput, Conv2DBackpropInputVerify);

//----------------Conv2DBackpropInputD-------------------
IMPLEMT_INFERFUNC(Conv2DBackpropInputD, Conv2DBackpropInputDInfer)
{
    OP_LOGI(op.GetName().c_str(), "Enter Conv2DBackpropInputD inferfunction!");
    const int32_t DIM_SIZE_LIMIT = 4;

    auto outBackpropDesc = op.GetInputDesc("out_backprop");
    auto yDesc = op.GetOutputDesc("y");

    // get dtype for output from out_backprop
    auto outBackpropDtype = outBackpropDesc.GetDataType();
    // get shape for output from input_size
    std::vector<int32_t> inputSizes;
    if (GRAPH_SUCCESS == op.GetAttr("input_size", inputSizes)) {
        if (inputSizes.size() != DIM_SIZE_LIMIT) {
            OP_LOGE(op.GetName().c_str(), "input_size list should be 4d.");
            map<string, string> err_map;
            err_map["op_name"] = "Conv2DBackpropInput";
            err_map["param_name"] = "input_size";
            err_map["expected_length"] = "4";
            err_map["length"] = std::to_string(inputSizes.size());
            std::string report_error_code = "E50035";
            (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return GRAPH_FAILED;
        }
    } else {
        OP_LOGE(op.GetName().c_str(), "get input_size list failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DBackpropInput";
        err_map["param_name"] = "input_size";
        std::string report_error_code = "E50030";
        (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // set dtype of output desc
    yDesc.SetDataType(outBackpropDtype);
    // set shape of output desc, input_size should match the format of y
    std::vector<int64_t> yShape;
    yShape.push_back(inputSizes[0]);
    yShape.push_back(inputSizes[1]);
    yShape.push_back(inputSizes[2]);
    yShape.push_back(inputSizes[3]);
    yDesc.SetShape(ge::Shape(yShape));

    // update output desc
    if (GRAPH_SUCCESS != op.UpdateOutputDesc("y", yDesc)) {
        OP_LOGE(op.GetName().c_str(), "update output desc failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DBackpropInput";
        err_map["param_name"] = "updating result";
        err_map["rule_desc"] = "updata OutputDesc";
        err_map["param_value"] = "failed";
        std::string report_error_code = "E50012";
        (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    std::vector<int64_t> filterSizes =  op.GetInputDesc("filter").GetShape().GetDims();
    Format filterFormat = op.GetInputDesc("filter").GetFormat();
    Format inputFormat = yDesc.GetFormat();
    CHECK_FORMAT(filterFormat);
    CHECK_FORMAT(inputFormat);

    // update pads list by padding[SAME,VALID]
    if (false == SetPadListByPaddingConv2dbp(op, inputSizes, inputFormat, filterSizes, filterFormat)) {
        OP_LOGE(op.GetName().c_str(), "Conv2DBackpropInputD update pads list by padding failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DBackpropInput";
        err_map["param_name"] = "updating result";
        err_map["rule_desc"] = "updata pads list by padding";
        err_map["param_value"] = "failed";
        std::string report_error_code = "E50012";
        (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
    OP_LOGI(op.GetName().c_str(), "Leaving Conv2DBackpropInputD inferfunction!");
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv2DBackpropInputD, Conv2DBackpropInputDVerify)
{
    OP_LOGI(op.GetName().c_str(), "Enter Conv2DBackpropInputD verifyfunction!");
    if (GRAPH_SUCCESS != VerifyConv2dbpInputCommon(op)) {
        return GRAPH_FAILED;
    }
    // check padding value
    if (GRAPH_SUCCESS != VerifyConv2dbpPads(op)) {
        return GRAPH_FAILED;
    }
    OP_LOGI(op.GetName().c_str(), "Leaving Conv2DBackpropInputD verifyfunction!");
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Conv2DBackpropInputD, Conv2DBackpropInputDInfer);
VERIFY_FUNC_REG(Conv2DBackpropInputD, Conv2DBackpropInputDVerify);

//----------------Conv2DBackpropFilter-------------------
static graphStatus VerifyConv2dbpFilterCommon(ge::Operator& op) {
    auto xDesc = op.GetInputDesc("x");
    auto outBackpropDesc = op.GetInputDesc("out_backprop");
    auto xDtype = xDesc.GetDataType();
    auto outBackpropDtype = outBackpropDesc.GetDataType();
    auto xShape = xDesc.GetShape().GetDims();
    auto outBackpropShape = outBackpropDesc.GetShape().GetDims();

    const int32_t DIM_SIZE_LIMIT = 4;
    const int32_t DIM_STRIDES_LIMIT = 4;

    //check input dtype
    if (xDtype != outBackpropDtype)
    {
        OP_LOGE(op.GetName().c_str(), "filter's dtype should equal to out_backprop's dtype.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DBackpropFilter";
        err_map["param1_name"] = "filter";
        err_map["param2_name"] = "out_backprop";
        err_map["param1_value"] = std::to_string(xDtype);
        err_map["param2_value"] = std::to_string(outBackpropDtype);
        err_map["attr_name"] = "dtype";
        std::string report_error_code = "E50031";
        (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // check input tensor shape
    if (xShape.size() != DIM_SIZE_LIMIT)
    {
        OP_LOGE(op.GetName().c_str(), "x's shape should be 4d.");
        auto x_shape = xShape.size();
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DBackpropFilter";
        err_map["param_name"] = "x's shape";
        err_map["expected_length"] = "4";
        err_map["length"] = std::to_string(x_shape);
        std::string report_error_code = "E50035";
        (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    if (outBackpropShape.size() != DIM_SIZE_LIMIT)
    {
        OP_LOGE(op.GetName().c_str(), "out_backprop's shape should be 4d.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DBackpropFilter";
        err_map["param_name"] = "out_backprop's shape";
        err_map["expected_length"] = "4";
        err_map["length"] = std::to_string(outBackpropShape.size());
        std::string report_error_code = "E50035";
        (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // check strides shape
    std::vector<int32_t> strideList;
    if (GRAPH_SUCCESS == op.GetAttr("strides", strideList)) {
        if (strideList.size() != DIM_STRIDES_LIMIT) {
            OP_LOGE(op.GetName().c_str(), "strides should be 4d.");
            map<string, string> err_map;
            err_map["op_name"] = "Conv2DBackpropFilter";
            err_map["param_name"] = "strides's shape";
            err_map["expected_length"] = std::to_string(DIM_STRIDES_LIMIT);
            err_map["length"] = std::to_string(strideList.size());
            std::string report_error_code = "E50035";
            (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return GRAPH_FAILED;
        }
    } else {
        OP_LOGE(op.GetName().c_str(), "get strides list failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DBackpropFilter";
        err_map["param_name"] = "strides list";
        std::string report_error_code = "E50030";
        (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // check dilations shape
    std::vector<int32_t> dilationsList;
    if (GRAPH_SUCCESS == op.GetAttr("dilations", dilationsList)) {
        if (dilationsList.size() != DIM_SIZE_LIMIT) {
            OP_LOGE(op.GetName().c_str(), "dilationsList list should be 4d.");
            map<string, string> err_map;
            err_map["op_name"] = "Conv2DBackpropFilter";
            err_map["param_name"] = "dilationsList";
            err_map["expected_length"] = std::to_string(DIM_SIZE_LIMIT);
            err_map["length"] = std::to_string(dilationsList.size());
            std::string report_error_code = "E50035";
            (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return GRAPH_FAILED;
        }
    } else {
        OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DBackpropFilter";
        err_map["param_name"] = "dilations";
        std::string report_error_code = "E50035";
        (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}
IMPLEMT_INFERFUNC(Conv2DBackpropFilter, Conv2DBackpropFilterInfer)
{
    OP_LOGI(op.GetName().c_str(), "Enter Conv2DBackpropFilter inferfunction!");

    auto filterSizesDesc = op.GetInputDesc("filter_size");
    auto yDesc = op.GetOutputDesc("y");

    Tensor filterSizesTensor;
    if (GRAPH_SUCCESS != op.GetInputConstData("filter_size", filterSizesTensor)) {
        OP_LOGE(op.GetName().c_str(), "get filter_size tensor failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DBackpropFilter";
        err_map["param_name"] = "filter_size";
        std::string report_error_code = "E50030";
        (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
    // get shape for output from filter_size
    DataType dtype = filterSizesDesc.GetDataType();
    std::vector<int64_t> filterSizes;
    GetConstValue(filterSizesTensor, dtype, filterSizes);

    // set dtype of output desc
    auto outBackpropDtype = op.GetInputDesc("out_backprop").GetDataType();
    yDesc.SetDataType(outBackpropDtype);
    // set shape of output desc, filter_size should match the format of y
    std::vector<int64_t> yShape;
    yShape.push_back(filterSizes[0]);
    yShape.push_back(filterSizes[1]);
    yShape.push_back(filterSizes[2]);
    yShape.push_back(filterSizes[3]);
    yDesc.SetShape(ge::Shape(yShape));

    // update output desc
    if (GRAPH_SUCCESS != op.UpdateOutputDesc("y", yDesc)) {
        OP_LOGE(op.GetName().c_str(), "update output desc failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DBackpropFilter";
        err_map["param_name"] = "updating result";
        err_map["rule_desc"] = "updata OutputDesc";
        err_map["param_value"] = "failed";
        std::string report_error_code = "E50012";
        (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    std::vector<int64_t> xSizes = op.GetInputDesc("x").GetShape().GetDims();
    Format xFormat = op.GetInputDesc("x").GetFormat();
    Format filterFormat = yDesc.GetFormat();
    CHECK_FORMAT(xFormat);
    CHECK_FORMAT(filterFormat);

    // update pads list by padding[SAME,VALID]
    if (false == SetPadListByPaddingConv2dbp(op, xSizes, xFormat, filterSizes, filterFormat)) {
        OP_LOGE(op.GetName().c_str(), "update pads list by padding failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DBackpropFilter";
        err_map["param_name"] = "updding result";
        err_map["rule_desc"] = "updata pads list by padding";
        err_map["param_value"] = "failed";
        std::string report_error_code = "E50012";
        (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
    OP_LOGI(op.GetName().c_str(), "Leaving Conv2DBackpropFilter inferfunction!");
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv2DBackpropFilter, Conv2DBackpropFilterVerify)
{
    OP_LOGI(op.GetName().c_str(), "Enter Conv2DBackpropFilter verifyfunction!");
    if (GRAPH_SUCCESS != VerifyConv2dbpFilterCommon(op)) {
        return GRAPH_FAILED;
    }
    // check padding value
    if (GRAPH_SUCCESS == VerifyConvPadding(op) || GRAPH_SUCCESS == VerifyConv2dbpPads(op)) {
        OP_LOGI(op.GetName().c_str(), "Leaving Conv2DBackpropFilter verifyfunction!");
        return GRAPH_SUCCESS;
    }else{
        OP_LOGE(op.GetName().c_str(), "Leaving Conv2DBackpropFilter verifyfunction!");
        map<string, string> err_map;
        err_map["param_name"] = "verify pads and verify padding result";
        err_map["op_name"] = "Conv2DBackpropInput";
        err_map["rule_desc"] = "verify pads and verify padding";
        err_map["param_value"] = "failed";
        std::string report_error_code = "E50012";
        (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
}

INFER_FUNC_REG(Conv2DBackpropFilter, Conv2DBackpropFilterInfer);
VERIFY_FUNC_REG(Conv2DBackpropFilter, Conv2DBackpropFilterVerify);

//----------------Conv2DBackpropFilterD-------------------
IMPLEMT_INFERFUNC(Conv2DBackpropFilterD, Conv2DBackpropFilterDInfer)
{
    OP_LOGI(op.GetName().c_str(), "Enter Conv2DBackpropFilterD inferfunction!");
    const int32_t DIM_SIZE_LIMIT = 4;

    auto outBackpropDesc = op.GetInputDesc("out_backprop");
    auto yDesc = op.GetOutputDesc("y");

    // get shape for output from filter_size
    std::vector<int32_t> filterSizes;
    if (GRAPH_SUCCESS == op.GetAttr("filter_size", filterSizes)) {
        if (filterSizes.size() != DIM_SIZE_LIMIT) {
            OP_LOGE(op.GetName().c_str(), "filter_size list should be 4d.");
            map<string, string> err_map;
            err_map["op_name"] = "Conv2DBackpropFilter";
            err_map["param_name"] = "filter_size";
            err_map["expected_length"] = "4";
            err_map["length"] = std::to_string(filterSizes.size());
            std::string report_error_code = "E50035";
            (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return GRAPH_FAILED;
        }
    } else {
        OP_LOGE(op.GetName().c_str(), "get filter_size list failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DBackpropFilter";
        err_map["param_name"] = "filter_size";
        std::string report_error_code = "E50030";
        (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // set dtype of output desc
    yDesc.SetDataType(DT_FLOAT);

    // set shape of output desc, filter_size should match the format of y
    std::vector<int64_t> yShape;
    yShape.push_back(filterSizes[0]);
    yShape.push_back(filterSizes[1]);
    yShape.push_back(filterSizes[2]);
    yShape.push_back(filterSizes[3]);
    yDesc.SetShape(ge::Shape(yShape));

    // update output desc
    if (GRAPH_SUCCESS != op.UpdateOutputDesc("y", yDesc)) {
        OP_LOGE(op.GetName().c_str(), "update output desc failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DBackpropFilter";
        err_map["param_name"] = "updating result";
        err_map["rule_desc"] = "updata OutputDesc";
        err_map["param_value"] = "failed";
        std::string report_error_code = "E50012";
        (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    std::vector<int64_t> xSizes = op.GetInputDesc("x").GetShape().GetDims();
    Format xFormat = op.GetInputDesc("x").GetFormat();
    Format filterFormat = yDesc.GetFormat();
    CHECK_FORMAT(xFormat);
    CHECK_FORMAT(filterFormat);

    // update pads list by padding[SAME,VALID]
    if (false == SetPadListByPaddingConv2dbp(op, xSizes, xFormat, filterSizes, filterFormat)) {
        OP_LOGE(op.GetName().c_str(), "update pads list by padding failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DBackpropFilter";
        err_map["param_name"] = "updding result";
        err_map["rule_desc"] = "updata pads list by padding";
        err_map["param_value"] = "failed";
        std::string report_error_code = "E50012";
        (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    OP_LOGI(op.GetName().c_str(), "Leaving Conv2DBackpropFilterD inferfunction!");
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv2DBackpropFilterD, Conv2DBackpropFilterDVerify)
{
    OP_LOGI(op.GetName().c_str(), "Enter Conv2DBackpropFilterD verifyfunction!");
    if (GRAPH_SUCCESS != VerifyConv2dbpFilterCommon(op)) {
        return GRAPH_FAILED;
    }
    // check padding value
    if (GRAPH_SUCCESS != VerifyConv2dbpPads(op)) {
        return GRAPH_FAILED;
    }
    OP_LOGI(op.GetName().c_str(), "Leaving Conv2DBackpropFilterD verifyfunction!");
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Conv2DBackpropFilterD, Conv2DBackpropFilterDInfer);
VERIFY_FUNC_REG(Conv2DBackpropFilterD, Conv2DBackpropFilterDVerify);

//--------------------------Conv2D------------------------------
/*
 * Convert different framework pad param to ir pads:
 *   [_padding]: 4D lsit, format sensitive, need convert to pads
 *   [padding]: 'SAME' or 'VALID', need convert to pads
 *   [pads]: 4D list, format sensitive, no need convert
*/
static bool GetPadConv2D(ge::Operator& op,
                         int32_t ih, int32_t iw,
                         int32_t kh, int32_t kw,
                         int32_t strh, int32_t strw,
                         int32_t dilh, int32_t dilw,
                         int32_t& padt, int32_t& padb,
                         int32_t& padl, int32_t& padr) {
    std::string padStr;
    std::vector<int32_t> padList;
    if (GRAPH_SUCCESS == op.GetAttr("padding", padStr)) {
        if (padStr.compare("SAME") == 0) {
            int32_t tails_h = ih % strh;
            int32_t tails_w = iw % strw;
            int32_t dkh = dilh*(kh - 1) + 1;
            int32_t dkw = dilw*(kw - 1) + 1;
            int32_t pad_h = \
                    std::max((tails_h > 0 ? dkh - tails_h : dkh - strh), 0);
            int32_t pad_w = \
                    std::max((tails_w > 0 ? dkw - tails_w : dkw - strw), 0);
            padList.push_back(pad_h / 2);
            padList.push_back(pad_h / 2 + pad_h % 2);
            padList.push_back(pad_w / 2);
            padList.push_back(pad_w / 2 + pad_w % 2);
        } else if (padStr.compare("VALID") == 0) {
            padList.push_back(0);
            padList.push_back(0);
            padList.push_back(0);
            padList.push_back(0);
        } else {
            OP_LOGE(op.GetName().c_str(), "padding should be SAME or VALID."
                    " actual is: %s.", padStr.c_str());
            map<string, string> err_map;
            err_map["op_name"] = op.GetName().c_str();
            err_map["expected_pad_mode"] = "SAME or VALID";
            err_map["actual_pad_mode"] = padStr;
            std::string report_error_code = "E50050";
            ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return false;
        }
        op.SetAttr("pads", padList);
    }

    // handle attr auto_pad from ONNX
    if (GRAPH_SUCCESS == op.GetAttr("auto_pad", padStr)){
        if (padStr.compare("SAME_UPPER") == 0){
            int32_t tails_h = ih % strh;
            int32_t tails_w = iw % strw;
            int32_t dkh = dilh * (kh - 1) + 1;
            int32_t dkw = dilw * (kw - 1) + 1;
            int32_t pad_h = \
                    std::max((tails_h > 0 ? dkh - tails_h : dkh - strh), 0);
            int32_t pad_w = \
                    std::max((tails_w > 0 ? dkw - tails_w : dkw - strw), 0);
            padList.push_back(pad_h / 2);
            padList.push_back(pad_h / 2 + pad_h % 2);
            padList.push_back(pad_w / 2);
            padList.push_back(pad_w / 2 + pad_w % 2);
            op.SetAttr("pads", padList);
        }
        else if (padStr.compare("SAME_LOWER") == 0){
            int32_t tails_h = ih % strh;
            int32_t tails_w = iw % strw;
            int32_t dkh = dilh * (kh - 1) + 1;
            int32_t dkw = dilw * (kw - 1) + 1;
            int32_t pad_h = \
                    std::max((tails_h > 0 ? dkh - tails_h : dkh - strh), 0);
            int32_t pad_w = \
                    std::max((tails_w > 0 ? dkw - tails_w : dkw - strw), 0);
            padList.push_back(pad_h / 2 + pad_h % 2);
            padList.push_back(pad_h / 2);
            padList.push_back(pad_w / 2 + pad_w % 2);
            padList.push_back(pad_w / 2);
            op.SetAttr("pads", padList);
        } else if (padStr.compare("NOTSET") == 0) {
        } else if (padStr.compare("VALID") == 0) {
            padList.push_back(0);
            padList.push_back(0);
            padList.push_back(0);
            padList.push_back(0);
            op.SetAttr("pads", padList);
        } else {
            OP_LOGE(op.GetName().c_str(), "padding should be SAME or VALID."
                    " actual is: %s.", padStr.c_str());
            map<string, string> err_map;
            err_map["op_name"] = op.GetName().c_str();
            err_map["expected_pad_mode"] = "NOTSET, SAME_UPPER, SAME_LOWER or VALID";
            err_map["actual_pad_mode"] = padStr;
            std::string report_error_code = "E50050";
            ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);

            return false;
        }
    }

    std::vector<int32_t> padVec;
    op.GetAttr("pads", padVec);
    auto pSize = padVec.size();
    if (pSize != 4) {
        OP_LOGE(op.GetName().c_str(), "pads list should be 4D."
                " actual is: %d.", (int)pSize);
        map<string, string> err_map;
        err_map["param_name"] = "pads";
        err_map["op_name"] = op.GetName().c_str();
        err_map["expected_value"] = "4D";
        err_map["input_value"] = std::to_string(pSize) + "D.";
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }
    padt = padVec[0];
    padb = padVec[1];
    padl = padVec[2];
    padr = padVec[3];
    if (padt < 0 || padb < 0 || padl < 0 || padr < 0) {
        OP_LOGE(op.GetName().c_str(), "pads should be positive, "
                " actual is [%d,%d,%d,%d].", padt, padb, padl, padr);
        map<string, string> err_map;
        err_map["param_name"] = "pads";
        err_map["op_name"] = op.GetName().c_str();
        err_map["expected_value"] = "positive";
        err_map["input_value"] = std::to_string(padt) + ", " + \
                                std::to_string(padb) + ", " + \
                                std::to_string(padl) + ", " + \
                                std::to_string(padr);
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    return true;
}

/*
 * Get 2D(H/W) stride and dilation params to infershape output
 *   [strides]: 4D list, format sensitive, according to first input
 *              tensor format
 *   [dilations]: 4D list, format sensitive
*/
static bool GetAttrsConv2D(ge::Operator& op, Format refer,
                           int32_t& strh, int32_t& strw,
                           int32_t& dilh, int32_t& dilw) {
    std::vector<int32_t> strideList;
    op.GetAttr("strides", strideList);
    auto sSize = strideList.size();
    if (sSize != 4) {
        OP_LOGE(op.GetName().c_str(), "strides list should be 4D."
                " actual is: %d.", (int)sSize);
        map<string, string> err_map;
        err_map["param_name"] = "strides";
        err_map["op_name"] = op.GetName().c_str();
        err_map["expected_value"] = "4D";
        err_map["input_value"] = std::to_string(sSize) + "D.";
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }
    std::vector<int32_t> dilationList;
    op.GetAttr("dilations", dilationList);
    auto dSize = dilationList.size();
    if (dSize != 4) {
        OP_LOGE(op.GetName().c_str(), "dilations list should be 4D."
                " actual is: %d.", (int)dSize);
        map<string, string> err_map;
        err_map["param_name"] = "dilations";
        err_map["op_name"] = op.GetName().c_str();
        err_map["expected_value"] = "4D";
        err_map["input_value"] = std::to_string(dSize) + "D.";
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    if (refer == FORMAT_NCHW) {
        strh = strideList[2];
        strw = strideList[3];
        dilh = dilationList[2];
        dilw = dilationList[3];
    } else if (refer == FORMAT_NHWC) {
        strh = strideList[1];
        strw = strideList[2];
        dilh = dilationList[1];
        dilw = dilationList[2];
    }
    if (strh <= 0 || strw <= 0) {
        OP_LOGE(op.GetName().c_str(), "strides should be positive,"
                " actual is [%d,%d].", strh, strw);
        map<string, string> err_map;
        err_map["param_name"] = "strides";
        err_map["op_name"] = op.GetName().c_str();
        err_map["expected_value"] = "positive";
        err_map["input_value"] = std::to_string(strh) + ", " + \
                                std::to_string(strw);
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }
    if (dilh <= 0 || dilw <= 0) {
        OP_LOGE(op.GetName().c_str(), "dilations should be positive,"
                " actual is [%d,%d].", dilh, dilw);
        map<string, string> err_map;
        err_map["param_name"] = "dilations";
        err_map["op_name"] = op.GetName().c_str();
        err_map["expected_value"] = "positive";
        err_map["input_value"] = std::to_string(dilh) + ", " + \
                                std::to_string(dilw);
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    return true;
}

/*
* Infer output shape and dtype, dtype is same to first input tensor
* Output format is set by ge parser process already
*/
IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(Conv2DInfer)
    OP_LOGD(op.GetName().c_str(), "Enter Conv2DInfer.");
    auto xTensor = op.GetInputDesc("x");
    auto wTensor = op.GetInputDesc("filter");

    auto xShape = xTensor.GetShape().GetDims();
    auto wShape = wTensor.GetShape().GetDims();
    auto xFormat = xTensor.GetFormat();
    auto wFormat  = wTensor.GetFormat();
    CHECK_FORMAT(xFormat);
    CHECK_FORMAT(wFormat);

    int32_t in = 0;
    int32_t ic = 0;
    int32_t ih = 0;
    int32_t iw = 0;
    int32_t kn = 0;
    int32_t kc = 0;
    int32_t kh = 0;
    int32_t kw = 0;
    if (xFormat == FORMAT_NCHW) {
        in = xShape[0];
        ic = xShape[1];
        ih = xShape[2];
        iw = xShape[3];
    } else if (xFormat == FORMAT_NHWC) {
        in = xShape[0];
        ic = xShape[3];
        ih = xShape[1];
        iw = xShape[2];
    } else {
        OP_LOGE(op.GetName().c_str(), "input x format should be NCHW or NHWC."
                " actual is: %s",
                TypeUtils::FormatToSerialString(xFormat).c_str());
        map<string, string> err_map;
        err_map["param"] = "x";
        err_map["op_name"] = op.GetName().c_str();
        err_map["expected_format_list"] = "NCHW or NHWC";
        err_map["format"] = std::to_string(xFormat);
        std::string report_error_code = "E50002";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    if (wFormat == FORMAT_NCHW) {
        kn = wShape[0];
        kc = wShape[1];
        kh = wShape[2];
        kw = wShape[3];
    } else if (wFormat == FORMAT_NHWC) {
        kn = wShape[0];
        kc = wShape[3];
        kh = wShape[1];
        kw = wShape[2];
    } else if (wFormat == FORMAT_HWCN) {
        kn = wShape[3];
        kc = wShape[2];
        kh = wShape[0];
        kw = wShape[1];
    } else {
        OP_LOGE(op.GetName().c_str(),
                "input filter format should be NCHW, NHWC or HWCN."
                " actual is: %s",
                TypeUtils::FormatToSerialString(wFormat).c_str());
        map<string, string> err_map;
        err_map["param"] = "filter";
        err_map["op_name"] = op.GetName().c_str();
        err_map["expected_format_list"] = "NCHW, NHWC or HWCN";
        err_map["format"] = std::to_string(wFormat);
        std::string report_error_code = "E50002";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // set data_format: copy value of xFormat to data_format
    // data_format will be used to infer position of H/W
    // in strides and dilations(previously used ori_format)
    std::string data_format;
    std::string attr_data_format = "data_format";
    std::string data_format_NCHW = "NCHW";
    std::string data_format_NHWC = "NHWC";

    if (GRAPH_SUCCESS == op.GetAttr(attr_data_format, data_format)) {
        OP_LOGI(data_format.c_str(), "conv before set data_format");
    }

    if (xFormat == ge::FORMAT_NCHW){
        op.SetAttr(attr_data_format, data_format_NCHW);
    } else {
        op.SetAttr(attr_data_format, data_format_NHWC);
    }

    op.GetAttr(attr_data_format, data_format);
    OP_LOGI(data_format.c_str(), "conv after set data_format");

    int64_t groups = 1;
    op.GetAttr("groups", groups);
    if (ic != kc*groups) {
        OP_LOGE(op.GetName().c_str(),
                "x channel should be equal to filter channel*groups. "
                "x format is: %s, filter format is: %s, "
                "x shape is: [%d,%d,%d,%d], filter shape is: [%d,%d,%d,%d], "
                "groups is: %d.",
                TypeUtils::FormatToSerialString(xFormat).c_str(),
                TypeUtils::FormatToSerialString(wFormat).c_str(),
                (int)xShape[0], (int)xShape[1],
                (int)xShape[2], (int)xShape[3],
                (int)wShape[0], (int)wShape[1],
                (int)wShape[2], (int)wShape[3],
                (int)groups);
        map<string, string> err_map;
        err_map["op_name"] = op.GetName().c_str();
        err_map["x_shape"] = std::to_string(xShape[0]) + ", " + \
                             std::to_string(xShape[1]) + ", " + \
                             std::to_string(xShape[2]) + ", " + \
                             std::to_string(xShape[3]);
        err_map["filter_shape"] = std::to_string(wShape[0]) + ", " + \
                                  std::to_string(wShape[1]) + ", " + \
                                  std::to_string(wShape[2]) + ", " + \
                                  std::to_string(wShape[3]);
        err_map["groups"] = std::to_string(groups);

        std::string report_error_code = "E50059";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    int32_t strh = 0;
    int32_t strw = 0;
    int32_t dilh = 0;
    int32_t dilw = 0;
    int32_t padt = 0;
    int32_t padb = 0;
    int32_t padl = 0;
    int32_t padr = 0;
    if (!GetAttrsConv2D(op, xFormat, strh, strw, dilh, dilw) ||
        !GetPadConv2D(op, ih, iw, kh, kw, strh, strw, dilh, dilw,
                      padt, padb, padl, padr)) {
        return GRAPH_FAILED;
    }

    int64_t ihPad = (ih + padt + padb - dilh * (kh - 1) - 1);
    int64_t iwPad = (iw + padl + padr - dilw * (kw - 1) - 1);
    int64_t oh = ihPad / strh + 1;
    int64_t ow = iwPad / strw + 1;
    if ((ih > 0) && (kh > 0) && (iw > 0) && (kw > 0)) {
        if ((ihPad < 0) || (iwPad < 0)) {
            OP_LOGE(op.GetName().c_str(), "image size after padding should be greater than or equal to filter size.");
            map<string, string> err_map;
            err_map["op_name"] = op.GetName().c_str();
            err_map["description"] = "image size after padding should be greater than or equal to filter size.";
            std::string report_error_code = "E50060";
            ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        }
    }

    vector<int64_t> yShape;
    auto yTensor = op.GetOutputDesc("y");
    auto yFormat = yTensor.GetFormat();
    CHECK_FORMAT(yFormat)
    if (yFormat == FORMAT_NCHW) {
        yShape.push_back(in);
        yShape.push_back(kn);
        yShape.push_back(oh);
        yShape.push_back(ow);
    } else if (yFormat == FORMAT_NHWC) {
        yShape.push_back(in);
        yShape.push_back(oh);
        yShape.push_back(ow);
        yShape.push_back(kn);
    } else {
        OP_LOGE(op.GetName().c_str(), "output y format should be NCHW or NHWC."
                " actual is: %s",
                TypeUtils::FormatToSerialString(yFormat).c_str());
        map<string, string> err_map;
        err_map["param"] = "y";
        err_map["op_name"] = op.GetName().c_str();
        err_map["expected_format_list"] = "NCHW or NHWC";
        err_map["format"] = TypeUtils::FormatToSerialString(yFormat);
        std::string report_error_code = "E50002";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
    yTensor.SetShape(Shape(yShape));
    auto xDtype = xTensor.GetDataType();
    if (xDtype == ge::DT_INT8) {
        yTensor.SetDataType(ge::DT_INT32);
    }else{
        yTensor.SetDataType(xDtype);
    }

    // set Range
    std::vector<std::pair<int64_t, int64_t>> fmapRange;
    xTensor.GetShapeRange(fmapRange);
    if (fmapRange.empty() || fmapRange.size() == 5) {
        OP_LOGD(op.GetName().c_str(), "Do not set range when fmapRange size is %d",
                (int32_t)fmapRange.size());
    } else {
        for (size_t i = 0; i < fmapRange.size(); i++) {
            OP_LOGD(op.GetName().c_str(), "fmap Range[%u] is (%lld, %lld)",
                    i, fmapRange[i].first, fmapRange[i].second);
        }

        size_t idxH = 0;
        size_t idxW = 0;
        size_t idxC = 0;
        if (xFormat == FORMAT_NHWC) {
            idxH = 1;
            idxW = 2;
            idxC = 3;
        } else if (xFormat == FORMAT_NCHW) {
            idxC = 1;
            idxH = 2;
            idxW = 3;
        }

        std::vector<std::pair<int64_t, int64_t>> outRange(fmapRange);
        outRange[idxC] = std::make_pair((int64_t)kn, (int64_t)kn);
        if (xShape[idxH] == -1) {
            yShape[idxH] = -1;
            int64_t lowH = fmapRange[idxH].first;
            int64_t highH = fmapRange[idxH].second;
            outRange[idxH].first = (lowH + padt + padb - dilh * (kh - 1) - 1) / strh + 1;
            outRange[idxH].second = (highH + padt + padb - dilh * (kh - 1) - 1) / strh + 1;
        }
        if (xShape[idxW] == -1) {
            yShape[idxW] = -1;
            int64_t lowW = fmapRange[idxW].first;
            int64_t highW = fmapRange[idxW].second;
            outRange[idxW].first = (lowW + padl + padr - dilw * (kw - 1) - 1) / strw + 1;
            outRange[idxW].second = (highW + padl + padr - dilw * (kw - 1) - 1) / strw + 1;
        }
        yTensor.SetShape(Shape(yShape));
        yTensor.SetShapeRange(outRange);
        for (size_t i = 0; i < outRange.size(); i++) {
            OP_LOGD(op.GetName().c_str(), "output Range[%u] is (%lld, %lld)",
                    i, outRange[i].first, outRange[i].second);
        }
    }

    if (GRAPH_SUCCESS != op.UpdateOutputDesc("y", yTensor)) {
        OP_LOGE(op.GetName().c_str(), "update output desc failed.");
        map<string, string> err_map;
        err_map["op_name"] = op.GetName().c_str();
        err_map["description"] = "update output desc failed.";
        std::string report_error_code = "E50058";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    OP_LOGD(op.GetName().c_str(), "Leave Conv2DInfer.");
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

/*
 * Verify the required 2 input tensor, optional bias ignored
 * Verify strides and dilations attrs, pads ignored
*/
IMPLEMT_VERIFIER(Conv2D, Conv2DVerify) {

    OP_LOGD(op.GetName().c_str(), "Enter Conv2DVerify.");
    auto xTensor = op.GetInputDesc("x");
    auto wTensor = op.GetInputDesc("filter");
    auto xShape = xTensor.GetOriginShape().GetDims();
    auto wShape = wTensor.GetOriginShape().GetDims();

    if (xShape.size() != 4) {
        if (xShape.size() == 0) {
            OP_LOGE(op.GetName().c_str(), "input x shape is empty.");
            map<string, string> err_map;
            err_map["op_name"] = op.GetName().c_str();
            err_map["description"] = "input x shape is empty.";
            std::string report_error_code = "E50060";
            ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        } else {
            OP_LOGE(op.GetName().c_str(), "input x shape shoule be 4D.");
            map<string, string> err_map;
            err_map["op_name"] = op.GetName().c_str();
            err_map["description"] = "input x shape shoule be 4D.";
            std::string report_error_code = "E50060";
            ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        }
        return GRAPH_FAILED;
    }
    if (wShape.size() != 4) {
        if (wShape.size() == 0) {
            OP_LOGE(op.GetName().c_str(), "input filter shape is empty.");
            map<string, string> err_map;
            err_map["op_name"] = op.GetName().c_str();
            err_map["description"] = "input filter shape is empty.";
            std::string report_error_code = "E50060";
            ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        } else {
            OP_LOGE(op.GetName().c_str(), "input filter shape shoule be 4D.");
            map<string, string> err_map;
            err_map["op_name"] = op.GetName().c_str();
            err_map["description"] = "input filter shape shoule be 4D.";
            std::string report_error_code = "E50060";
            ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        }
        return GRAPH_FAILED;
    }

    auto xDtype = xTensor.GetDataType();
    auto wDtype = wTensor.GetDataType();

    if (xDtype != wDtype) {
        OP_LOGE(op.GetName().c_str(),
                "input x dtype is differ from filter dtype."
                " actual x dtype is: %d filter dtype is: %d",
                (int)xDtype, (int)wDtype);
        map<string, string> err_map;
        err_map["op_name"] = op.GetName().c_str();
        err_map["param1"] = "x";
        err_map["param1_data_type"] = std::to_string(xDtype);
        err_map["param2"] = "filter";
        err_map["param2_data_type"] = std::to_string(wDtype);
        err_map["rule"] = "input x dtype is same as filter dtype";
        std::string report_error_code = "E50004";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    std::vector<int32_t> strideList;
    if (GRAPH_SUCCESS != op.GetAttr("strides", strideList)) {
        OP_LOGE(op.GetName().c_str(), "get strides list failed.");
        map<string, string> err_map;
        err_map["op_name"] = op.GetName().c_str();
        err_map["description"] = "get strides list failed.";
        std::string report_error_code = "E50060";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
    std::vector<int32_t> dilationList;
    if (GRAPH_SUCCESS != op.GetAttr("dilations", dilationList)) {
        OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
        map<string, string> err_map;
        err_map["op_name"] = op.GetName().c_str();
        err_map["description"] = "get dilations list failed.";
        std::string report_error_code = "E50060";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    OP_LOGD(op.GetName().c_str(), "Leave Conv2DVerify.");
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Conv2D, Conv2DInfer);
VERIFY_FUNC_REG(Conv2D, Conv2DVerify);

//--------------------------Conv2DCompress------------------------------

/*
* Infer output shape and dtype, dtype is same to first input tensor
* Output format is set by ge parser process already
*/
IMPLEMT_INFERFUNC(Conv2DCompress, Conv2DCompressInfer) {

    OP_LOGD(op.GetName().c_str(), "Enter Conv2DCompressInfer.");
    auto xTensor = op.get_input_desc_x();
    auto wTensor = op.get_input_desc_filter_compress();

    auto xShape = xTensor.GetShape().GetDims();
    auto wShape = wTensor.GetShape().GetDims();
    auto xFormat = xTensor.GetFormat();
    auto wFormat  = wTensor.GetFormat();
    CHECK_FORMAT(xFormat);
    CHECK_FORMAT(wFormat);

    int32_t in = 0;
    int32_t ic = 0;
    int32_t ih = 0;
    int32_t iw = 0;
    int32_t kn = 0;
    int32_t kc = 0;
    int32_t kh = 0;
    int32_t kw = 0;
    if (xFormat == FORMAT_NCHW) {
        in = xShape[0];
        ic = xShape[1];
        ih = xShape[2];
        iw = xShape[3];
    } else if (xFormat == FORMAT_NHWC) {
        in = xShape[0];
        ic = xShape[3];
        ih = xShape[1];
        iw = xShape[2];
    } else {
        OP_LOGE(op.GetName().c_str(), "input x format should be NCHW or NHWC.");
        return GRAPH_FAILED;
    }

    if (wFormat == FORMAT_NCHW) {
        kn = wShape[0];
        kc = wShape[1];
        kh = wShape[2];
        kw = wShape[3];
    } else if (wFormat == FORMAT_NHWC) {
        kn = wShape[0];
        kc = wShape[3];
        kh = wShape[1];
        kw = wShape[2];
    } else if (wFormat == FORMAT_HWCN) {
        kn = wShape[3];
        kc = wShape[2];
        kh = wShape[0];
        kw = wShape[1];
    } else {
        OP_LOGE(op.GetName().c_str(),
                "input filter format should be NCHW, NHWC or HWCN.");
        return GRAPH_FAILED;
    }

    // set data_format: copy value of xFormat to data_format
    // data_format will be used to infer position of H/W
    // in strides and dilations(previously used ori_format)
    std::string data_format;
    std::string attr_data_format = "data_format";
    std::string data_format_NCHW = "NCHW";
    std::string data_format_NHWC = "NHWC";

    if (GRAPH_SUCCESS == op.GetAttr(attr_data_format, data_format)) {
        OP_LOGI(data_format.c_str(), "conv compress before set data_format");
    }

    if (xFormat == ge::FORMAT_NCHW){
        op.SetAttr(attr_data_format, data_format_NCHW);
    } else {
        op.SetAttr(attr_data_format, data_format_NHWC);
    }

    op.GetAttr(attr_data_format, data_format);
    OP_LOGI(data_format.c_str(), "conv compress after set data_format");

    int64_t groups = 1;
    if (GRAPH_SUCCESS != op.GetAttr("groups", groups)) {
        OP_LOGI(op.GetName().c_str(), "no groups setting, use groups as 1");
    }
    OP_LOGI(op.GetName().c_str(), "groups is %lld", groups);

    if (ic != kc*groups) {
        OP_LOGE(op.GetName().c_str(),
                "input x channel should be equal to filter. ");
        return GRAPH_FAILED;
    }

    int32_t strh = 0;
    int32_t strw = 0;
    int32_t dilh = 0;
    int32_t dilw = 0;
    int32_t padt = 0;
    int32_t padb = 0;
    int32_t padl = 0;
    int32_t padr = 0;
    if (false == GetAttrsConv2D(op, xFormat, strh, strw, dilh, dilw)) {
        OP_LOGE(op.GetName().c_str(), "get attrs failed.");
        return GRAPH_FAILED;
    }
    if (false == GetPadConv2D(op, ih, iw, kh, kw, strh, strw, dilh, dilw,
                              padt, padb, padl, padr)) {
        OP_LOGE(op.GetName().c_str(), "get pads attrs failed.");
        return GRAPH_FAILED;
    }

    int64_t oh = (ih + padt + padb - dilh * (kh - 1) - 1) / strh + 1;
    int64_t ow = (iw + padl + padr - dilw * (kw - 1) - 1) / strw + 1;

    vector<int64_t> yShape;
    auto yTensor = op.get_output_desc_y();
    auto yFormat = yTensor.GetFormat();
    CHECK_FORMAT(yFormat)
    if (yFormat == FORMAT_NCHW) {
        yShape.push_back(in);
        yShape.push_back(kn);
        yShape.push_back(oh);
        yShape.push_back(ow);
    } else if (yFormat == FORMAT_NHWC) {
        yShape.push_back(in);
        yShape.push_back(oh);
        yShape.push_back(ow);
        yShape.push_back(kn);
    } else {
        OP_LOGE(op.GetName().c_str(), "output y format should be NCHW or NHWC.");
        return GRAPH_FAILED;
    }
    yTensor.SetShape(Shape(yShape));
    auto xDtype = xTensor.GetDataType();
    if (xDtype == ge::DT_INT8) {
        yTensor.SetDataType(ge::DT_INT32);
    }else{
        yTensor.SetDataType(xDtype);
    }
    if (GRAPH_SUCCESS != op.update_output_desc_y(yTensor)) {
        OP_LOGE(op.GetName().c_str(), "update output desc failed.");
        return GRAPH_FAILED;
    }

    OP_LOGD(op.GetName().c_str(), "Leave Conv2DCompressInfer.");
    return GRAPH_SUCCESS;
}

/*
 * Verify the required 2 input tensor, optional bias ignored
 * Verify strides and dilations attrs, pads ignored
*/
IMPLEMT_VERIFIER(Conv2DCompress, Conv2DCompressVerify) {

    OP_LOGD(op.GetName().c_str(), "Enter Conv2DCompressVerify.");
    auto xTensor = op.get_input_desc_x();
    auto wTensor = op.get_input_desc_filter_compress();

    auto xShape = xTensor.GetShape().GetDims();
    auto wShape = wTensor.GetShape().GetDims();

    if (xShape.size() != 4) {
        if (xShape.size() == 0) {
            OP_LOGE(op.GetName().c_str(), "input x shape is empty.");
        } else {
            OP_LOGE(op.GetName().c_str(), "only support 2D compress convolution.");
        }
        return GRAPH_FAILED;
    }
    if (wShape.size() != 4) {
        if (wShape.size() == 0) {
            OP_LOGE(op.GetName().c_str(), "input filter_compress shape is empty.");
        } else {
            OP_LOGE(op.GetName().c_str(), "only support 2D compress convolution.");
        }
        return GRAPH_FAILED;
    }

    auto xDtype = xTensor.GetDataType();
    auto wDtype = wTensor.GetDataType();

    if (xDtype != wDtype) {
        OP_LOGE(op.GetName().c_str(),
                "input x dtype is differ from filter_compress dtype."
                " actual x dtype is: %d filter dtype is: %d",
                (int)xDtype, (int)wDtype);
        return GRAPH_FAILED;
    }

    std::vector<int32_t> strideList;
    if (GRAPH_SUCCESS != op.GetAttr("strides", strideList)) {
        OP_LOGE(op.GetName().c_str(), "get strides list failed.");
        return GRAPH_FAILED;
    }
    std::vector<int32_t> dilationList;
    if (GRAPH_SUCCESS != op.GetAttr("dilations", dilationList)) {
        OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
        return GRAPH_FAILED;
    }

    OP_LOGD(op.GetName().c_str(), "Leave Conv2DCompressVerify.");
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Conv2DCompress, Conv2DCompressInfer);
VERIFY_FUNC_REG(Conv2DCompress, Conv2DCompressVerify);

//----------------Deconvolution--------------------------
static bool GetAttrsDeconv(ge::Operator& op, Format refer,
                           int32_t& strh, int32_t& strw,
                           int32_t& dilh, int32_t& dilw) {
    std::vector<int32_t> strideList;
    op.GetAttr("strides", strideList);
    auto sSize = strideList.size();
    if (sSize != 2) {
        OP_LOGE(op.GetName().c_str(), "strides list should be 2d."
            " actual is: %d.", sSize);
        string sizealue = Strcat(sSize);
        map<string, string> err_map;
        err_map["op_name"] = op.GetName().c_str();
        err_map["param_name"] = "sSize";
        err_map["expected_value"] = "2d";
        err_map["input_value"] = sizealue;
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }
    std::vector<int32_t> dilationList;
    op.GetAttr("dilations", dilationList);
    auto dSize = dilationList.size();
    if (dSize != 4) {
        OP_LOGE(op.GetName().c_str(), "dilations list should be 4d."
            " actual is: %d.", dSize);
        string realvalue = Strcat(dSize);
        map<string, string> err_map;
        err_map["op_name"] = op.GetName().c_str();
        err_map["param_name"] = "dSize";
        err_map["expected_value"] = "4d";
        err_map["input_value"] = realvalue;
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    strh = strideList[0];
    strw = strideList[1];
    if (refer == FORMAT_NCHW) {
        dilh = dilationList[2];
        dilw = dilationList[3];
    } else {
        return false;
    }
    if (strh <= 0 || strw <= 0) {
        OP_LOGE(op.GetName().c_str(), "strides should be positive, "
            " actual is [%d,%d].", strh, strw);
        map<string, string> err_map;
        err_map["op_name"] = op.GetName().c_str();
        err_map["param_name"] = "strides";
        err_map["expected_value"] = "positive";
        err_map["input_value"] = "[" + std::to_string(strh) + "," + std::to_string(strw) + "]";
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    return true;
}

IMPLEMT_INFERFUNC(Deconvolution, DeconvolutionInfer)
{
    OP_LOGD(op.GetName().c_str(), "Enter DeconvolutionInfer.");
    auto xTensor = op.get_input_desc_x();
    auto wTensor = op.get_input_desc_filter();

    auto xShape = xTensor.GetShape().GetDims();
    auto wShape = wTensor.GetShape().GetDims();
    auto xFormat = xTensor.GetFormat();
    auto wFormat  = wTensor.GetFormat();
    CHECK_FORMAT(xFormat);
    CHECK_FORMAT(wFormat);

    int32_t in = 0;
    int32_t ic = 0;
    int32_t ih = 0;
    int32_t iw = 0;
    int32_t kn = 0;
    int32_t kc = 0;
    int32_t kh = 0;
    int32_t kw = 0;
    if (xFormat == FORMAT_NCHW) {
        in = xShape[0];
        ic = xShape[1];
        ih = xShape[2];
        iw = xShape[3];
    } else {
        OP_LOGE(op.GetName().c_str(), "input x format should be NCHW"
            " actual is: %d", xFormat);
        map<string, string> err_map;
        err_map["op_name"] = op.GetName().c_str();
        err_map["param_name"] = "xFormat";
        err_map["expected_format_list"] = "[NCHW]";
        err_map["format"] = Strcat(xFormat);
        std::string report_error_code = "E50033";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    if (wFormat == FORMAT_NCHW) {
        kn = wShape[0];
        kc = wShape[1];
        kh = wShape[2];
        kw = wShape[3];
    } else {
        OP_LOGE(op.GetName().c_str(), "input filter format should be NCHW"
            " actual is: %d", wFormat);
        map<string, string> err_map;
        err_map["op_name"] = op.GetName().c_str();
        err_map["param_name"] = "wFormat";
        err_map["expected_format_list"] = "[NCHW]";
        err_map["format"] = Strcat(wFormat);
        std::string report_error_code = "E50033";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    int64_t groups = 1;
    if (GRAPH_SUCCESS != op.GetAttr("groups", groups)) {
        OP_LOGI(op.GetName().c_str(), "no groups setting, use groups as 1");
    }
    OP_LOGI(op.GetName().c_str(), "groups is %lld", groups);

    if (ic != kn) {
        OP_LOGE(op.GetName().c_str(), "input x channel should be equal to filter. "
            "x format is: %d, filter format is: %d "
            "x shape is: [%d,%d,%d,%d], filter shape is: [%d,%d,%d,%d].", \
            xFormat, wFormat, \
            xShape[0], xShape[1], xShape[2], xShape[3], \
            wShape[0], wShape[1], wShape[2], wShape[3]);
        map<string, string> err_map;
        err_map["op_name"] = op.GetName().c_str();
        err_map["attr_name"] = "channel";
        err_map["param1_name"] = "x";
        err_map["param1_value"] = std::to_string(ic);
        err_map["param2_name"] = "filter";
        err_map["param2_value"] = std::to_string(kn);
        std::string report_error_code = "E50031";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    int32_t strh = 0;
    int32_t strw = 0;
    int32_t dilh = 0;
    int32_t dilw = 0;
    int32_t padt = 0;
    int32_t padb = 0;
    int32_t padl = 0;
    int32_t padr = 0;
    if (false == GetAttrsDeconv(op, xFormat, strh, strw, dilh, dilw)) {
        OP_LOGE(op.GetName().c_str(), "get attrs failed.");
        map<string, string> err_map;
        err_map["op_name"] = op.GetName().c_str();
        err_map["description"] = "get attrs failed.";
        std::string report_error_code = "E50058";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
    if (false == GetPadConv2D(op, ih, iw, kh, kw, strh, strw, dilh, dilw, \
                              padt, padb, padl, padr)) {
        OP_LOGE(op.GetName().c_str(), "get pads attrs failed.");
        map<string, string> err_map;
        err_map["op_name"] = op.GetName().c_str();
        err_map["description"] = "get pads attrs failed.";
        std::string report_error_code = "E50058";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    int khext = dilh*(kh-1)+1;
    int kwext = dilw*(kw-1)+1;
    int64_t oh = strh*(ih-1)+khext-padt-padb;
    int64_t ow = strw*(iw-1)+kwext-padl-padr;

    vector<int64_t> yShape;
    auto yTensor = op.get_output_desc_y();
    auto yFormat = yTensor.GetFormat();
    CHECK_FORMAT(yFormat)
    if (yFormat == FORMAT_NCHW) {
        yShape.push_back(in);
        yShape.push_back(kc*groups);
        yShape.push_back(oh);
        yShape.push_back(ow);
    } else {
        OP_LOGE(op.GetName().c_str(), "output y format should be NCHW."
            " actual is: %d", yFormat);
        map<string, string> err_map;
        err_map["op_name"] = op.GetName().c_str();
        err_map["param_name"] = "yFormat";
        err_map["expected_format_list"] = "[NCHW]";
        err_map["format"] = Strcat(yFormat);
        std::string report_error_code = "E50033";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
    yTensor.SetShape(Shape(yShape));
    auto xDtype = xTensor.GetDataType();
    if (xDtype == DT_INT8) {
      yTensor.SetDataType(DT_INT32);
    } else {
      yTensor.SetDataType(xDtype);
    }
    if (GRAPH_SUCCESS != op.update_output_desc_y(yTensor)) {
        OP_LOGE(op.GetName().c_str(), "update output desc failed.");
        map<string, string> err_map;
        err_map["op_name"] = op.GetName().c_str();
        err_map["description"] = "update output desc failed.";
        std::string report_error_code = "E50058";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    OP_LOGD(op.GetName().c_str(), "Leave DeconvolutionInfer.");
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Deconvolution, DeconvolutionVerify)
{
    OP_LOGD(op.GetName().c_str(), "Enter DeconvolutionVerify.");
    auto xTensor = op.get_input_desc_x();
    auto wTensor = op.get_input_desc_filter();

    auto xShape = xTensor.GetShape().GetDims();
    auto wShape = wTensor.GetShape().GetDims();
    if (xShape.size() != 4) {
        OP_LOGE(op.GetName().c_str(), "input x shape should be 4d.");
        string xvalue =  Strcat(xShape.size());
        map<string, string> err_map;
        err_map["op_name"] = op.GetName().c_str();
        err_map["param_name"] = "xShape";
        err_map["expected_value"] = "4d";
        err_map["input_value"] = xvalue;
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
    if (wShape.size() != 4) {
        OP_LOGE(op.GetName().c_str(), "input filter shape should be 4d.");
        string wvalue =  Strcat(wShape.size());
        map<string, string> err_map;
        err_map["op_name"] = op.GetName().c_str();
        err_map["param_name"] = "wShape";
        err_map["expected_value"] = "4d";
        err_map["input_value"] = wvalue;
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    std::vector<int32_t> strideList;
    if (GRAPH_SUCCESS != op.GetAttr("strides", strideList)) {
        OP_LOGE(op.GetName().c_str(), "get strides list failed.");
        map<string, string> err_map;
        err_map["op_name"] = op.GetName().c_str();
        err_map["description"] = "get strides list failed.";
        std::string report_error_code = "E50058";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
    std::vector<int32_t> dilationList;
    if (GRAPH_SUCCESS != op.GetAttr("dilations", dilationList)) {
        OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
        map<string, string> err_map;
        err_map["op_name"] = op.GetName().c_str();
        err_map["description"] = "get dilations list failed.";
        std::string report_error_code = "E50058";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    OP_LOGD(op.GetName().c_str(), "Leave DeconvolutionVerify.");
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Deconvolution, DeconvolutionInfer);
VERIFY_FUNC_REG(Deconvolution, DeconvolutionVerify);

// ---------------------------Conv3D---------------------------
static bool GetPadConv3D(ge::Operator& op,
                           int32_t id, int32_t ih, int32_t iw,
                           int32_t kd, int32_t kh, int32_t kw,
                           int32_t strd, int32_t strh, int32_t strw,
                           int32_t dild, int32_t dilh, int32_t dilw,
                           int32_t& padf, int32_t& padba, int32_t& padt,
                           int32_t& padb, int32_t& padl, int32_t& padr) {
    std::string padStr;
    std::vector<int32_t> padList;
    if (GRAPH_SUCCESS == op.GetAttr("_padding", padList)) {
        op.SetAttr("pads", padList);
    } else if (GRAPH_SUCCESS == op.GetAttr("padding", padStr)){
        if (padStr.compare("SAME") == 0){
            int32_t tails_d = id % strd;
            int32_t tails_h = ih % strh;
            int32_t tails_w = iw % strw;
            int32_t pad_d = std::max((tails_d > 0 ? kd - tails_d : kd - strd), 0);
            int32_t pad_h = std::max((tails_h > 0 ? kh - tails_h : kh - strh), 0);
            int32_t pad_w = std::max((tails_w > 0 ? kw - tails_w : kw - strw), 0);
            padList.push_back(pad_d / 2);
            padList.push_back(pad_d / 2 + pad_d % 2);
            padList.push_back(pad_h / 2);
            padList.push_back(pad_h / 2 + pad_h % 2);
            padList.push_back(pad_w / 2);
            padList.push_back(pad_w / 2 + pad_w % 2);
        } else if (padStr.compare("VALID") == 0) {
            for(int32_t i=0;i<6;i++)
                padList.push_back(0);
        } else {
            OP_LOGE(op.GetName().c_str(), "padding should be SAME or VALID.");
            map<string, string> err_map;
            err_map["param_name"] = "padding";
            err_map["op_name"] = "Conv3d";
            err_map["Expected_value"] = "SAME or VALID";
            err_map["input_value"] = padStr;
            std::string report_error_code = "E50029";
            ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return false;
        }
        op.SetAttr("pads", padList);
    }
    std::vector<int32_t> padVec;
    if (GRAPH_SUCCESS != op.GetAttr("pads", padVec)) {
        OP_LOGE(op.GetName().c_str(), "get pads failed");
        return GRAPH_FAILED;
    }
    auto pSize = padVec.size();
    if (pSize != 6) {
        OP_LOGE(op.GetName().c_str(), "pads list should be 6d.");
        map<string, string> err_map;
        err_map["param_name"] = "pads list";
        err_map["op_name"] = "Conv3d";
        err_map["excepted_value"] = "6d";
        err_map["input_value"] = std::to_string(pSize);
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }
    padf = padVec[0];
    padba = padVec[1];
    padt = padVec[2];
    padb = padVec[3];
    padl = padVec[4];
    padr = padVec[5];
    if (padf < 0 || padba <0 || padt < 0 || padb < 0 || padl < 0 || padr < 0) {
        OP_LOGE(op.GetName().c_str(), "pads should be positive");
        map<string, string> err_map;
        err_map["param_name"] = "pads_list";
        err_map["op_name"] = "Conv3d";
        err_map["excepted_value"] = "positive";
        err_map["input_value"] = std::to_string(padf) + " " + \
                                 std::to_string(padba) + " " + \
                                 std::to_string(padt) + " " + \
                                 std::to_string(padb) + " " + \
                                 std::to_string(padl) + " " + \
                                 std::to_string(padr);
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    return true;
}

static bool GetAttrsConv3D(ge::Operator& op, Format refer,
                           int32_t& strd, int32_t& strh, int32_t& strw,
                           int32_t& dild, int32_t& dilh, int32_t& dilw) {
    std::vector<int32_t> strideList;
    if (GRAPH_SUCCESS != op.GetAttr("strides", strideList)) {
        OP_LOGE(op.GetName().c_str(), "get strides list failed.");
        return GRAPH_FAILED;
    }
    auto sSize = strideList.size();
    if (sSize != 5) {
        OP_LOGE(op.GetName().c_str(), "strides list should be 5d.");
        map<string, string> err_map;
        err_map["param_name"] = "strides_list";
        err_map["op_name"] = "Conv3d";
        err_map["excepted_value"] = "5d";
        err_map["input_value"] = std::to_string(sSize);
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    // get data_format, not used for now temporarily
    std::string dataFormat;
    if (GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
        OP_LOGE(op.GetName().c_str(), "The Conv3D op GetOpAttr data_format failed!");
        map<string, string> err_map;
        err_map["param_name"] = "data_format";
        err_map["op_name"] = "Conv3d";
        err_map["excepted_value"] = "NDHWC";
        err_map["input_value"] = dataFormat;
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    std::vector<int32_t> dilationList;
    if (GRAPH_SUCCESS != op.GetAttr("dilations", dilationList)) {
        OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
        return GRAPH_FAILED;
    }
    auto dSize = dilationList.size();
    if (dSize != 5) {
        OP_LOGE(op.GetName().c_str(), "dilations list should be 5d.");
        map<string, string> err_map;
        err_map["param_name"] = "dilation_list";
        err_map["op_name"] = "Conv3d";
        err_map["excepted_value"] = "5d";
        err_map["input_value"] = std::to_string(dSize);
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    if (refer == FORMAT_NCDHW) {
        strd = strideList[2];
        strh = strideList[3];
        strw = strideList[4];
        dild = dilationList[2];
        dilh = dilationList[3];
        dilw = dilationList[4];
    } else if (refer == FORMAT_NDHWC) {
        strd = strideList[1];
        strh = strideList[2];
        strw = strideList[3];
        dild = dilationList[1];
        dilh = dilationList[2];
        dilw = dilationList[3];
    }
    if (strd <= 0 || strh <= 0 || strw <= 0) {
        OP_LOGE(op.GetName().c_str(), "strides should be positive.");
        map<string, string> err_map;
        err_map["param_name"] = "strides";
        err_map["op_name"] = "Conv3d";
        err_map["excepted_value"] = "positive";
        err_map["input_value"] = std::to_string(strd) + " " + \
                                 std::to_string(strh) + " " + \
                                 std::to_string(strw);
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }
    if (dild != 1 || dilh != 1 || dilw != 1) {
        OP_LOGE(op.GetName().c_str(), "dilations only support 1 now.");
        map<string, string> err_map;
        err_map["param_name"] = "dilations";
        err_map["op_name"] = "Conv3d";
        err_map["excepted_value"] = "1, 1, 1";
        err_map["input_value"] = std::to_string(dild) + " " + \
                                 std::to_string(dilh) + " " + \
                                 std::to_string(dilw);
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    return true;
}
IMPLEMT_INFERFUNC(Conv3D, Conv3DInfer) {
    OP_LOGD(op.GetName().c_str(), "Enter Conv3DInfer.");
    auto xTensor = op.get_input_desc_x();
    auto wTensor = op.get_input_desc_filter();
    auto xShape = xTensor.GetShape().GetDims();
    auto wShape = wTensor.GetShape().GetDims();
    auto xFormat = xTensor.GetFormat();
    auto wFormat  = wTensor.GetFormat();
    CHECK_FORMAT(xFormat);
    CHECK_FORMAT(wFormat);

    int32_t in = 0;
    int32_t ic = 0;
    int32_t id = 0;
    int32_t ih = 0;
    int32_t iw = 0;
    int32_t kn = 0;
    int32_t kc = 0;
    int32_t kd = 0;
    int32_t kh = 0;
    int32_t kw = 0;
    if (xFormat == FORMAT_NCDHW) {
        in = xShape[0];
        ic = xShape[1];
        id = xShape[2];
        ih = xShape[3];
        iw = xShape[4];
    } else if (xFormat == FORMAT_NDHWC) {
        in = xShape[0];
        ic = xShape[4];
        id = xShape[1];
        ih = xShape[2];
        iw = xShape[3];
    } else {
        OP_LOGE(op.GetName().c_str(), "input x format should be NCDHW or NDHWC.");
        map<string, string> err_map;
        err_map["param_name"] = "xFormat";
        err_map["op_name"] = "Conv3d";
        err_map["excepted_value"] = "NCDHW or NDHWC";
        err_map["input_value"] = xFormat;
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
    if (wFormat == FORMAT_NCDHW) {
        kn = wShape[0];
        kc = wShape[1];
        kd = wShape[2];
        kh = wShape[3];
        kw = wShape[4];
    } else if (wFormat == FORMAT_NDHWC) {
        kn = wShape[0];
        kc = wShape[4];
        kd = wShape[1];
        kh = wShape[2];
        kw = wShape[3];
    } else if (wFormat == FORMAT_DHWCN) {
        kn = wShape[4];
        kc = wShape[3];
        kd = wShape[0];
        kh = wShape[1];
        kw = wShape[2];
    } else {
        OP_LOGE(op.GetName().c_str(), "input filter format should be NCDHW, NDHWC or DHWCN.");
        map<string, string> err_map;
        err_map["param_name"] = "wFormat";
        err_map["op_name"] = "Conv3d";
        err_map["excepted_value"] = "NCDHW or NDHWC or DHWCN";
        err_map["input_value"] = wFormat;
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    int64_t group = 1;
    if (GRAPH_SUCCESS != op.GetAttr("group", group)) {
        OP_LOGI(op.GetName().c_str(), "no group setting, use group as 1");
    }

    if (ic != kc*group) {
        OP_LOGE(op.GetName().c_str(), "input x channel should be equal to filter.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3d";
        err_map["channel_of_x"] = std::to_string(ic);
        err_map["channel_of_filter"] = std::to_string(kc*group);
        std::string report_error_code = "E50039";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    int32_t strd = 0;
    int32_t strh = 0;
    int32_t strw = 0;
    int32_t dild = 0;
    int32_t dilh = 0;
    int32_t dilw = 0;
    int32_t padf = 0;
    int32_t padba = 0;
    int32_t padt = 0;
    int32_t padb = 0;
    int32_t padl = 0;
    int32_t padr = 0;
    if (false == GetAttrsConv3D(op, xFormat, strd, strh, strw, dild, dilh, dilw)) {
        OP_LOGE(op.GetName().c_str(), "get attrs failed.");
        return GRAPH_FAILED;
    }
    if (false == GetPadConv3D(op, id, ih, iw, kd, kh, kw, strd, strh, strw, \
                              dild, dilh, dilw, padf, padba, padt, padb, padl, padr)) {
        OP_LOGE(op.GetName().c_str(), "get pads attrs failed.");
        return GRAPH_FAILED;
    }

    int64_t od = (id + padf + padba - dild * (kd - 1) - 1) / strd + 1;
    int64_t oh = (ih + padt + padb - dilh * (kh - 1) - 1) / strh + 1;
    int64_t ow = (iw + padl + padr - dilw * (kw - 1) - 1) / strw + 1;

    vector<int64_t> yShape;
    auto yTensor = op.get_output_desc_y();
    auto yFormat = yTensor.GetFormat();
    CHECK_FORMAT(yFormat)
    if (yFormat == FORMAT_NCDHW) {
        yShape.push_back(in);
        yShape.push_back(kn);
        yShape.push_back(od);
        yShape.push_back(oh);
        yShape.push_back(ow);
    } else if (yFormat == FORMAT_NDHWC) {
        yShape.push_back(in);
        yShape.push_back(od);
        yShape.push_back(oh);
        yShape.push_back(ow);
        yShape.push_back(kn);
    } else {
        OP_LOGE(op.GetName().c_str(), "output y format should be NCDHW or NDHWC.");
        map<string, string> err_map;
        err_map["param_name"] = "yFormat";
        err_map["op_name"] = "Conv3d";
        err_map["excepted_value"] = "NCDHW or NDHWC";
        err_map["input_value"] = wFormat;
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
    yTensor.SetShape(Shape(yShape));
    auto xDtype = xTensor.GetDataType();
    yTensor.SetDataType(xDtype);
    if (GRAPH_SUCCESS != op.update_output_desc_y(yTensor)) {
        OP_LOGE(op.GetName().c_str(), "update output desc failed.");
        map<string, string> err_map;
        err_map["param_name"] = "output_desc_y";
        err_map["op_name"] = "Conv3d";
        err_map["excepted_value"] = GRAPH_SUCCESS;
        err_map["output_value"] = op.update_output_desc_y(yTensor);
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
    OP_LOGD(op.GetName().c_str(), "Leave Conv3DInfer.");
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv3D, Conv3DVerify) {
    OP_LOGD(op.GetName().c_str(), "Enter Conv3DVerify.");
    auto xTensor = op.get_input_desc_x();
    auto wTensor = op.get_input_desc_filter();

    auto xShape = xTensor.GetShape().GetDims();
    auto wShape = wTensor.GetShape().GetDims();
    if (xShape.size() != 5) {
        OP_LOGE(op.GetName().c_str(), "input x shape should be 5d.");
        map<string, string> err_map;
        err_map["param_name"] = "xShape_size";
        err_map["op_name"] = "Conv3d";
        err_map["excepted_value"] = std::to_string(5);
        err_map["output_value"] = xShape.size();
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
    if (wShape.size() != 5) {
        OP_LOGE(op.GetName().c_str(), "input filter shape should be 5d.");
        map<string, string> err_map;
        err_map["param_name"] = "wShape_size";
        err_map["op_name"] = "Conv3d";
        err_map["excepted_value"] = std::to_string(5);
        err_map["output_value"] = wShape.size();
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    auto xDtype = xTensor.GetDataType();
    auto wDtype = wTensor.GetDataType();

    if(xDtype != wDtype) {
        OP_LOGE(op.GetName().c_str(), "input x dtype is differ from filter dtype.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3d";
        err_map["attr_name"] = "dtype";
        err_map["param1_name"] = "input x";
        err_map["param2_name"] = "weight";
        err_map["param1_value"] = std::to_string(xDtype);
        err_map["param2_value"] = std::to_string(wDtype);
        std::string report_error_code = "E50031";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    std::vector<int32_t> strideList;
    if (GRAPH_SUCCESS != op.GetAttr("strides", strideList)) {
        OP_LOGE(op.GetName().c_str(), "get strides list failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3d";
        err_map["op_name"] = "strides";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
    std::vector<int32_t> dilationList;
    if (GRAPH_SUCCESS != op.GetAttr("dilations", dilationList)) {
        OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3d";
        err_map["op_name"] = "dilations";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
    OP_LOGD(op.GetName().c_str(), "Leave Conv3DVerify.");
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Conv3D, Conv3DInfer);
VERIFY_FUNC_REG(Conv3D, Conv3DVerify);

// -----------------------------conv3dbp_common_check-----------------------------
template<typename T1, typename T2>
static bool SetPadListByPaddingConv3dbp(ge::Operator& op,
                                        const std::vector<T1>& inputSizes,
                                        Format inputFormat,
                                        const std::vector<T2>& filterSizes,
                                        Format filterFormat)
{
    const int32_t INPUT_SIZE_LIMIT = 5;
    const int32_t PADS_SIZE_LIMIT = 6;
    if(filterSizes.size() < INPUT_SIZE_LIMIT || inputSizes.size() < INPUT_SIZE_LIMIT){
        OP_LOGE(op.GetName().c_str(), "filter_sizes or inputSizes is illegal");
        map<string, string> err_map;
        err_map["param_name"] = "filter_size and inputsize";
        err_map["op_name"] = "Conv3dbp";
        err_map["excepted_value"] = std::to_string(5);
        err_map["input_value"] = std::to_string(filterSizes.size()) + " " + \
                                 std::to_string(inputSizes.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }
    CHECK_FORMAT(inputFormat);
    CHECK_FORMAT(filterFormat);

    std::vector<int32_t> strideList;
    if (GRAPH_FAILED == op.GetAttr("strides", strideList)){
        OP_LOGE(op.GetName().c_str(), "op get strides failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dbp";
        err_map["param_name"] = "strides";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    if (strideList.size() < INPUT_SIZE_LIMIT){
        OP_LOGE(op.GetName().c_str(), "op get strides failed.");
        map<string, string> err_map;
        err_map["param_name"] = "strides";
        err_map["op_name"] = "Conv3dbp";
        err_map["excepted_value"] = std::to_string(3);
        err_map["input_value"] = std::to_string(strideList.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }
    std::string inputFormatStr = format2str[inputFormat];
    int32_t hInputPosition = inputFormatStr.find("H");
    CHECK_POSITION(hInputPosition);
    int32_t wInputPosition = inputFormatStr.find("W");
    CHECK_POSITION(wInputPosition);
    int32_t dInputPosition = inputFormatStr.find("D");
    CHECK_POSITION(dInputPosition);
    int32_t dx_h = inputSizes[hInputPosition];
    int32_t dx_w = inputSizes[wInputPosition];
    int32_t dx_d = inputSizes[dInputPosition];

    int32_t stride_h = strideList[hInputPosition];
    int32_t stride_w = strideList[wInputPosition];
    int32_t stride_d = strideList[dInputPosition];


    std::string filterFormatStr = format2str[filterFormat];
    int32_t hFilterPosition = filterFormatStr.find("H");
    CHECK_POSITION(hFilterPosition);
    int32_t wFilterPosition = filterFormatStr.find("W");
    CHECK_POSITION(wFilterPosition);
    int32_t dFilterPosition = filterFormatStr.find("D");
    CHECK_POSITION(dFilterPosition);

    int32_t filter_h = filterSizes[hFilterPosition];
    int32_t filter_w = filterSizes[wFilterPosition];
    int32_t filter_d = filterSizes[dFilterPosition];

    std::string padding;
    std::vector<int32_t> pads;
    if (GRAPH_SUCCESS == op.GetAttr("padding", padding)){
        OP_LOGI(op.GetName().c_str(), "op get padding succ.");
        int pad_h = 0;
        int32_t pad_up = 0;
        int32_t pad_down = 0;
        int pad_w = 0;
        int32_t pad_left = 0;
        int32_t pad_right = 0;
        int pad_d = 0;
        int32_t pad_head = 0;
        int32_t pad_tail = 0;
        if (padding == "SAME"){
            pad_h = std::max(ALIGN_CONV2DBP(dx_h, stride_h) - stride_h + filter_h - dx_h, 0);
            pad_up = pad_h / 2 ;
            pad_down = pad_h - pad_up;
            pad_w = std::max(ALIGN_CONV2DBP(dx_w, stride_w) - stride_w + filter_w - dx_w, 0);
            pad_left = pad_w / 2 ;
            pad_right = pad_w - pad_left;
            pad_d = std::max(ALIGN_CONV2DBP(dx_d, stride_d) - stride_d + filter_d - dx_d, 0);
            pad_head = pad_d / 2 ;
            pad_tail = pad_d - pad_head;

        }

        pads.push_back(pad_head);
        pads.push_back(pad_tail);
        pads.push_back(pad_up);
        pads.push_back(pad_down);
        pads.push_back(pad_left);
        pads.push_back(pad_right);

        op.SetAttr("pads", pads);
    }
    if (GRAPH_SUCCESS == op.GetAttr("pads", pads)){
        if (pads.size() < PADS_SIZE_LIMIT){
            OP_LOGE(op.GetName().c_str(), "op pads's size is illegal,pads.");
            map<string, string> err_map;
            err_map["param_name"] = "pads";
            err_map["op_name"] = "Conv3dbp";
            err_map["excepted_value"] = std::to_string(6);
            err_map["input_value"] = std::to_string(pads.size());
            std::string report_error_code = "E50029";
            ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return false;
        }
        if (pads[0] < 0 || pads[1] < 0 ||
            pads[2] < 0 || pads[3] < 0 ||
            pads[4] < 0 || pads[5] < 0){
            OP_LOGE(op.GetName().c_str(), "op get pads is illegal");
            map<string, string> err_map;
            err_map["param_name"] = "pads";
            err_map["op_name"] = "Conv3dbp";
            err_map["excepted_value"] = "Non-negative";
            err_map["input_value"] = std::to_string(pads[0]) + " " + \
                                     std::to_string(pads[1]) + " " + \
                                     std::to_string(pads[2]) + " " + \
                                     std::to_string(pads[3]) + " " + \
                                     std::to_string(pads[4]) + " " + \
                                     std::to_string(pads[5]);
            std::string report_error_code = "E50029";
            ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return false;
        }
    } else{
        OP_LOGE(op.GetName().c_str(), "op get pads failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dbp";
        err_map["param_name"] = "pads";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }
    OP_LOGI(op.GetName().c_str(), "op set pads succ.");
    return true;
}

static graphStatus VerifyConv3dbpInputCommon(ge::Operator& op){
    auto filterDesc = op.GetInputDesc("filter");
    auto outBackpropDesc = op.GetInputDesc("out_backprop");

    auto filterDtype = filterDesc.GetDataType();
    auto outBackpropDtype = outBackpropDesc.GetDataType();
    auto filterShape = filterDesc.GetShape().GetDims();
    auto outBackpropShape = outBackpropDesc.GetShape().GetDims();

    const int32_t DIM_SIZE_LIMIT = 5;

    //check input dtype
    if (filterDtype != outBackpropDtype)
    {
        OP_LOGE(op.GetName().c_str(), "filter's dtype should equal to outBackprop's dtype.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dbpInput";
        err_map["attr_name"] = "dtype";
        err_map["param1_name"] = "filter";
        err_map["param2_name"] = "outBackprop";
        err_map["param1_value"] = std::to_string(filterDtype);
        err_map["param2_value"] = std::to_string(outBackpropDtype);
        std::string report_error_code = "E50031";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // check input tensor shape
    if (filterShape.size() != DIM_SIZE_LIMIT)
    {
        OP_LOGE(op.GetName().c_str(), "filter's shape should be 5d.");
        map<string, string> err_map;
        err_map["param_name"] = "filterShape_size";
        err_map["op_name"] = "Conv3dbpInput";
        err_map["excepted_value"] = std::to_string(5);
        err_map["input_value"] = std::to_string(filterShape.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    if (outBackpropShape.size() != DIM_SIZE_LIMIT)
    {
        OP_LOGE(op.GetName().c_str(), "outBackprop's shape should be 5d.");
        map<string, string> err_map;
        err_map["param_name"] = "outBackpropShape_size";
        err_map["op_name"] = "Conv3dbpInput";
        err_map["excepted_value"] = std::to_string(5);
        err_map["input_value"] = std::to_string(outBackpropShape.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // check strides shape
    std::vector<int32_t> strideList;
    if (GRAPH_SUCCESS == op.GetAttr("strides", strideList)) {
        if (strideList.size() != DIM_SIZE_LIMIT) {
            OP_LOGE(op.GetName().c_str(), "strides should be 5d.");
            map<string, string> err_map;
            err_map["param_name"] = "strides";
            err_map["op_name"] = "Conv3dbpInput";
            err_map["excepted_value"] = std::to_string(5);
            err_map["input_value"] = std::to_string(strideList.size());
            std::string report_error_code = "E50029";
            ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return GRAPH_FAILED;
        }
    } else {
        OP_LOGE(op.GetName().c_str(), "get strides list failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dbpInput";
        err_map["param_name"] = "strides";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // check dilations shape
    std::vector<int32_t> dilationsList;
    if (GRAPH_SUCCESS == op.GetAttr("dilations", dilationsList)) {
        if (dilationsList.size() != DIM_SIZE_LIMIT) {
            OP_LOGE(op.GetName().c_str(), "dilationsList list should be 5d.");
            map<string, string> err_map;
            err_map["param_name"] = "dilations";
            err_map["op_name"] = "Conv3dbpInput";
            err_map["excepted_value"] = std::to_string(5);
            err_map["input_value"] = std::to_string(dilationsList.size());
            std::string report_error_code = "E50029";
            ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return GRAPH_FAILED;
        }
    } else {
        OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dbpInput";
        err_map["param_name"] = "dilations";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}
static graphStatus VerifyConv3dbpPads(ge::Operator& op)
{
    const int32_t LENGTH_PADS_LIMIT = 6;

    std::vector<int> pads;
    if (GRAPH_SUCCESS == op.GetAttr("pads", pads)){
        if (pads.size() < LENGTH_PADS_LIMIT){
            OP_LOGE(op.GetName().c_str(), "op pads's size is illegal,pads.");
            map<string, string> err_map;
            err_map["param_name"] = "pads";
            err_map["op_name"] = "Conv3dbpInput";
            err_map["excepted_value"] = std::to_string(6);
            err_map["input_value"] = std::to_string(pads.size());
            std::string report_error_code = "E50029";
            ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return GRAPH_FAILED;
        }

        if (pads[0] < 0 || pads[1] < 0 ||
            pads[2] < 0 || pads[3] < 0 ||
            pads[4] < 0 || pads[5] < 0){
            OP_LOGE(op.GetName().c_str(), "op get pads is illegal");
            map<string, string> err_map;
            err_map["param_name"] = "pads";
            err_map["op_name"] = "Conv3dbpInput";
            err_map["excepted_value"] = "positive";
            err_map["input_value"] = std::to_string(pads[0]) + " " + \
                                     std::to_string(pads[1]) + " " + \
                                     std::to_string(pads[2]) + " " + \
                                     std::to_string(pads[3]) + " " + \
                                     std::to_string(pads[4]) + " " + \
                                     std::to_string(pads[5]);
            std::string report_error_code = "E50029";
            ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return GRAPH_FAILED;
        }
    } else{
        OP_LOGE(op.GetName().c_str(), "op get pads failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dbpInput";
        err_map["param_name"] = "pads";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}


IMPLEMT_INFERFUNC(Conv3DBackpropInput, Conv3DBackpropInputInfer)
{
    OP_LOGI(op.GetName().c_str(), "Enter Conv3DBackpropInput inferfunction!");

    auto inputSizesDesc = op.GetInputDesc("input_size");
    auto yDesc = op.GetOutputDesc("y");

    Tensor inputSizesTensor;
    if (GRAPH_SUCCESS != op.GetInputConstData("input_size", inputSizesTensor)) {
        OP_LOGE(op.GetName().c_str(), "get input_size tensor failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dbpInput";
        err_map["param_name"] = "inputSizesTensor";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // get shape for output from input_sizes
    DataType dtype = inputSizesDesc.GetDataType();
    std::vector<int64_t> inputSizes;
    GetConstValue(inputSizesTensor, dtype, inputSizes);
    // std::vector<int64_t> inputSizes = op.GetInputDesc("input_sizes").GetShape().GetDims();

    // set dtype of output desc
    auto outBackpropDtype = op.GetInputDesc("out_backprop").GetDataType();
    yDesc.SetDataType(outBackpropDtype);
    // set shape of output desc, input_sizes should match the format of y
    std::vector<int64_t> yShape;
    for (auto i : inputSizes) {
        yShape.push_back(i);
    }
    yDesc.SetShape(ge::Shape(yShape));

    // update output desc
    if (GRAPH_SUCCESS != op.UpdateOutputDesc("y", yDesc)) {
        OP_LOGE(op.GetName().c_str(), "update output desc failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dbpInput";
        err_map["param_name"] = "output y";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    std::vector<int64_t> filterSizes = op.GetInputDesc("filter").GetShape().GetDims();
    Format filterFormat = op.GetInputDesc("filter").GetFormat();
    Format inputFormat = yDesc.GetFormat();
    CHECK_FORMAT(filterFormat);
    CHECK_FORMAT(inputFormat);
    // update pads list by padding[SAME,VALID]
    if(false == SetPadListByPaddingConv3dbp(op, inputSizes, inputFormat, filterSizes, filterFormat)){
        OP_LOGE(op.GetName().c_str(), "update pads list by padding failed.");
        return GRAPH_FAILED;
    }

    OP_LOGI(op.GetName().c_str(), "Leaving Conv3DBackpropInput inferfunction!");
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv3DBackpropInput, Conv3DBackpropInputVerify)
{
    OP_LOGI(op.GetName().c_str(), "Enter Conv3DBackpropInput verifyfunction!");
    if(GRAPH_SUCCESS != VerifyConv3dbpInputCommon(op)){
        return GRAPH_FAILED;
    }
    // check padding value
    if(GRAPH_SUCCESS == VerifyConvPadding(op) || GRAPH_SUCCESS == VerifyConv3dbpPads(op)){
        OP_LOGI(op.GetName().c_str(), "Leaving Conv3DBackpropInput verifyfunction!");
        return GRAPH_SUCCESS;
    }else{
        OP_LOGE(op.GetName().c_str(), "Leaving Conv3DBackpropInput verifyfunction!");
        return GRAPH_FAILED;
    }
}

INFER_FUNC_REG(Conv3DBackpropInput, Conv3DBackpropInputInfer);
VERIFY_FUNC_REG(Conv3DBackpropInput, Conv3DBackpropInputVerify);


//----------------Conv3DBackpropInputD-------------------
IMPLEMT_INFERFUNC(Conv3DBackpropInputD, Conv3DBackpropInputDInfer)
{
    OP_LOGI(op.GetName().c_str(), "Enter Conv3DBackpropInputD inferfunction!");
    const int32_t DIM_SIZE_LIMIT = 5;

    auto outBackpropDesc = op.GetInputDesc("out_backprop");
    auto yDesc = op.GetOutputDesc("y");

    // get dtype for output from out_backprop
    auto outBackpropDtype =outBackpropDesc.GetDataType();
    // get shape for output from input_size
    std::vector<int32_t> inputSizes;
    if (GRAPH_SUCCESS == op.GetAttr("input_size", inputSizes)) {
        if (inputSizes.size() != DIM_SIZE_LIMIT) {
            OP_LOGE(op.GetName().c_str(), "input_size list should be 5d.");
            map<string, string> err_map;
            err_map["param_name"] = "input_size";
            err_map["op_name"] = "Conv3dbpInput";
            err_map["excepted_value"] = std::to_string(5);
            err_map["input_value"] = std::to_string(inputSizes.size());
            std::string report_error_code = "E50029";
            ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return GRAPH_FAILED;
        }
    } else {
        OP_LOGE(op.GetName().c_str(), "get input_size list failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dbpInput";
        err_map["param_name"] = "input_size";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // set dtype of output desc
    yDesc.SetDataType(outBackpropDtype);
    // set shape of output desc, input_size should match the format of y
    std::vector<int64_t> outShape;
    for (auto i : inputSizes) {
        outShape.push_back(i);
    }
    yDesc.SetShape(ge::Shape(outShape));

    // update output desc
    if (GRAPH_SUCCESS != op.UpdateOutputDesc("y", yDesc)) {
        OP_LOGE(op.GetName().c_str(), "update output desc failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dbpInput";
        err_map["param_name"] = "output y";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    std::vector<int64_t> filterSizes =  op.GetInputDesc("filter").GetShape().GetDims();
    Format filterFormat = op.GetInputDesc("filter").GetFormat();
    Format inputFormat = yDesc.GetFormat();
    CHECK_FORMAT(filterFormat);
    CHECK_FORMAT(inputFormat);
    // update pads list by padding[SAME,VALID]
    if(false == SetPadListByPaddingConv3dbp(op, inputSizes, inputFormat, filterSizes, filterFormat)){
        OP_LOGE(op.GetName().c_str(), "Conv3DBackpropInputD update pads list by padding failed.");
        return GRAPH_FAILED;
    }
    OP_LOGI(op.GetName().c_str(), "Leaving Conv3DBackpropInputD inferfunction!");
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv3DBackpropInputD, Conv3DBackpropInputDVerify)
{
    OP_LOGI(op.GetName().c_str(), "Enter Conv3DBackpropInputD verifyfunction!");
    if(GRAPH_SUCCESS != VerifyConv3dbpInputCommon(op)){
        return GRAPH_FAILED;
    }
    // check padding value
    if(GRAPH_SUCCESS != VerifyConv3dbpPads(op)){
        return GRAPH_FAILED;
    }
    OP_LOGI(op.GetName().c_str(), "Leaving Conv3DBackpropInputD verifyfunction!");
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Conv3DBackpropInputD, Conv3DBackpropInputDInfer);
VERIFY_FUNC_REG(Conv3DBackpropInputD, Conv3DBackpropInputDVerify);


//----------------Conv3DBackpropFilter-------------------
static graphStatus VerifyConv3dbpFilterCommon(ge::Operator& op){
    auto xDesc = op.GetInputDesc("x");
    auto outBackpropDesc = op.GetInputDesc("out_backprop");
    auto xDtype = xDesc.GetDataType();
    auto outBackpropDtype = outBackpropDesc.GetDataType();
    auto xShape = xDesc.GetShape().GetDims();
    auto outBackpropShape = outBackpropDesc.GetShape().GetDims();

    const int32_t DIM_SIZE_LIMIT = 5;
    const int32_t DIM_STRIDES_LIMIT = 5;

    //check input dtype
    if (xDtype != outBackpropDtype)
    {
        OP_LOGE(op.GetName().c_str(), "x's dtype should equal to out_backprop's dtype.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dbpFilter";
        err_map["attr_name"] = "dtype";
        err_map["param1_name"] = "input x";
        err_map["param2_name"] = "outBackprop";
        err_map["param1_value"] = std::to_string(xDtype);
        err_map["param2_value"] = std::to_string(outBackpropDtype);
        std::string report_error_code = "E50031";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // check input tensor shape
    if (xShape.size() != DIM_SIZE_LIMIT)
    {
        OP_LOGE(op.GetName().c_str(), "x's shape should be 5d.");
        map<string, string> err_map;
        err_map["param_name"] = "xShape_size";
        err_map["op_name"] = "Conv3dbpFilter";
        err_map["excepted_value"] = std::to_string(5);
        err_map["input_value"] = std::to_string(xShape.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    if (outBackpropShape.size() != DIM_SIZE_LIMIT)
    {
        OP_LOGE(op.GetName().c_str(), "out_backprop's shape should be 5d.");
        map<string, string> err_map;
        err_map["param_name"] = "outBackpropShape_size";
        err_map["op_name"] = "Conv3dbpFilter";
        err_map["excepted_value"] = std::to_string(5);
        err_map["input_value"] = std::to_string(outBackpropShape.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // check strides shape
    std::vector<int32_t> strideList;
    if (GRAPH_SUCCESS == op.GetAttr("strides", strideList)) {
        if (strideList.size() != DIM_STRIDES_LIMIT) {
            OP_LOGE(op.GetName().c_str(), "strides should be 5d.");
            map<string, string> err_map;
            err_map["param_name"] = "strides";
            err_map["op_name"] = "Conv3dbpFilter";
            err_map["excepted_value"] = std::to_string(5);
            err_map["input_value"] = std::to_string(strideList.size());
            std::string report_error_code = "E50029";
            ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return GRAPH_FAILED;
        }
    } else {
        OP_LOGE(op.GetName().c_str(), "get strides list failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dbpFilter";
        err_map["param_name"] = "strides";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // check dilations shape
    std::vector<int32_t> dilationsList;
    if (GRAPH_SUCCESS == op.GetAttr("dilations", dilationsList)) {
        if (dilationsList.size() != DIM_SIZE_LIMIT) {
            OP_LOGE(op.GetName().c_str(), "dilationsList list should be 5d.");
            map<string, string> err_map;
            err_map["param_name"] = "dilations";
            err_map["op_name"] = "Conv3dbpFilter";
            err_map["excepted_value"] = std::to_string(5);
            err_map["input_value"] = std::to_string(dilationsList.size());
            std::string report_error_code = "E50029";
            ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return GRAPH_FAILED;
        }
    } else {
        OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dbpFilter";
        err_map["param_name"] = "dilations";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}
IMPLEMT_INFERFUNC(Conv3DBackpropFilter, Conv3DBackpropFilterInfer)
{
    OP_LOGI(op.GetName().c_str(), "Enter Conv3DBackpropFilter inferfunction!");

    auto filterSizesDesc = op.GetInputDesc("filter_size");
    auto yDesc = op.GetOutputDesc("y");

    Tensor filterSizesTensor;
    if (GRAPH_SUCCESS != op.GetInputConstData("filter_size", filterSizesTensor)) {
        OP_LOGE(op.GetName().c_str(), "get filter_size tensor failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dbpFilter";
        err_map["param_name"] = "filter tensor";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
    // get shape for output from filter_size
    DataType dtype = filterSizesDesc.GetDataType();
    std::vector<int64_t> filterSizes;
    GetConstValue(filterSizesTensor, dtype, filterSizes);

    // set dtype of output desc
    auto outBackpropDtype = op.GetInputDesc("out_backprop").GetDataType();
    yDesc.SetDataType(outBackpropDtype);
    // set shape of output desc, filter_size should match the format of y
    std::vector<int64_t> yShape;
    yShape.push_back(filterSizes[0]);
    yShape.push_back(filterSizes[1]);
    yShape.push_back(filterSizes[2]);
    yShape.push_back(filterSizes[3]);
    yShape.push_back(filterSizes[4]);
    yDesc.SetShape(ge::Shape(yShape));

    // update output desc
    if (GRAPH_SUCCESS != op.UpdateOutputDesc("y", yDesc)) {
        OP_LOGE(op.GetName().c_str(), "update output desc failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dbpFilter";
        err_map["param_name"] = "output y";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
    std::vector<int64_t> xSizes = op.GetInputDesc("x").GetShape().GetDims();
    Format xFormat = op.GetInputDesc("x").GetFormat();
    Format filterFormat = yDesc.GetFormat();
    CHECK_FORMAT(xFormat);
    CHECK_FORMAT(filterFormat);
    // update pads list by padding[SAME,VALID]
    if(false == SetPadListByPaddingConv3dbp(op, xSizes, xFormat, filterSizes, filterFormat)){
        OP_LOGE(op.GetName().c_str(), "update pads list by padding failed.");
        return GRAPH_FAILED;
    }
    OP_LOGI(op.GetName().c_str(), "Leaving Conv3DBackpropFilter inferfunction!");
    return GRAPH_SUCCESS;
}
static graphStatus VerifyConv3dbpFilterPads(ge::Operator& op)
{
    const int32_t LENGTH_PADS_LIMIT = 6;
    std::vector<int> pads;
    if (GRAPH_SUCCESS == op.GetAttr("pads", pads)){
        if (pads.size() < LENGTH_PADS_LIMIT){
            OP_LOGE(op.GetName().c_str(), "op pads's size is illegal.");
            map<string, string> err_map;
            err_map["param_name"] = "pads";
            err_map["op_name"] = "Conv3dbpFilter";
            err_map["excepted_value"] = std::to_string(6);
            err_map["input_value"] = std::to_string(pads.size());
            std::string report_error_code = "E50029";
            ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return GRAPH_FAILED;
        }

        if (pads[0] < 0 || pads[1] < 0 ||
            pads[2] < 0 || pads[3] < 0 ||
            pads[4] < 0 || pads[5] < 0){
            OP_LOGE(op.GetName().c_str(), "op get pads is illegal");
            map<string, string> err_map;
            err_map["param_name"] = "pads";
            err_map["op_name"] = "Conv3dbpFilter";
            err_map["excepted_value"] = "positive";
            err_map["input_value"] = std::to_string(pads[0]) + " " + \
                                     std::to_string(pads[1]) + " " + \
                                     std::to_string(pads[2]) + " " + \
                                     std::to_string(pads[3]) + " " + \
                                     std::to_string(pads[4]) + " " + \
                                     std::to_string(pads[5]);
            std::string report_error_code = "E50029";
            ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return GRAPH_FAILED;
        }
    } else{
        OP_LOGE(op.GetName().c_str(), "op get pads failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dbpFilter";
        err_map["param_name"] = "pads";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv3DBackpropFilter, Conv3DBackpropFilterVerify)
{
    OP_LOGI(op.GetName().c_str(), "Enter Conv3DBackpropFilter verifyfunction!");
    if(GRAPH_SUCCESS != VerifyConv3dbpFilterCommon(op)){
        return GRAPH_FAILED;
    }
    // check padding value
    if(GRAPH_SUCCESS == VerifyConvPadding(op) || GRAPH_SUCCESS == VerifyConv3dbpPads(op)){
        OP_LOGI(op.GetName().c_str(), "Leaving Conv3DBackpropFilter verifyfunction!");
        return GRAPH_SUCCESS;
    }else{
        OP_LOGE(op.GetName().c_str(), "Leaving Conv3DBackpropFilter verifyfunction!");
        return GRAPH_FAILED;
    }
}

INFER_FUNC_REG(Conv3DBackpropFilter, Conv3DBackpropFilterInfer);
VERIFY_FUNC_REG(Conv3DBackpropFilter, Conv3DBackpropFilterVerify);

//----------------Conv3DBackpropFilterD-------------------
IMPLEMT_INFERFUNC(Conv3DBackpropFilterD, Conv3DBackpropFilterDInfer)
{
    OP_LOGI(op.GetName().c_str(), "Enter Conv3DBackpropFilterD inferfunction!");
    const int32_t DIM_SIZE_LIMIT = 5;

    auto outBackpropDesc = op.GetInputDesc("out_backprop");
    auto yDesc = op.GetOutputDesc("y");

    // get dtype for output
    auto outBackpropDtype = outBackpropDesc.GetDataType();
    // get shape for output from filter_size
    std::vector<int32_t> filterSizes;
    if (GRAPH_SUCCESS == op.GetAttr("filter_size", filterSizes)) {
        if (filterSizes.size() != DIM_SIZE_LIMIT) {
            OP_LOGE(op.GetName().c_str(), "filter_size list should be 5d.");
            map<string, string> err_map;
            err_map["param_name"] = "filter_sizes";
            err_map["op_name"] = "Conv3dbpFilter";
            err_map["excepted_value"] = std::to_string(5);
            err_map["input_value"] = std::to_string(filterSizes.size());
            std::string report_error_code = "E50029";
            ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return GRAPH_FAILED;
        }
    } else {
        OP_LOGE(op.GetName().c_str(), "get filter_size list failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dbpFilter";
        err_map["param_name"] = "filter_size";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // set dtype of output desc
    yDesc.SetDataType(outBackpropDtype);

    // set shape of output desc, filter_size should match the format of y
    std::vector<int64_t> yShape;
    yShape.push_back(filterSizes[0]);
    yShape.push_back(filterSizes[1]);
    yShape.push_back(filterSizes[2]);
    yShape.push_back(filterSizes[3]);
    yShape.push_back(filterSizes[4]);
    yDesc.SetShape(ge::Shape(yShape));

    // update output desc
    if (GRAPH_SUCCESS != op.UpdateOutputDesc("y", yDesc)) {
        OP_LOGE(op.GetName().c_str(), "update output desc failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dbpFilter";
        err_map["param_name"] = "output y";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    std::vector<int64_t> xSizes = op.GetInputDesc("x").GetShape().GetDims();
    Format xFormat = op.GetInputDesc("x").GetFormat();
    Format filterFormat = yDesc.GetFormat();
    CHECK_FORMAT(xFormat);
    CHECK_FORMAT(filterFormat);
    // update pads list by padding[SAME,VALID]
    if(false == SetPadListByPaddingConv3dbp(op, xSizes, xFormat, filterSizes, filterFormat)){
        OP_LOGE(op.GetName().c_str(), "update pads list by padding failed.");
        return GRAPH_FAILED;
    }

    OP_LOGI(op.GetName().c_str(), "Leaving Conv3DBackpropFilterD inferfunction!");
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv3DBackpropFilterD, Conv3DBackpropFilterDVerify)
{
    OP_LOGI(op.GetName().c_str(), "Enter Conv3DBackpropFilterD verifyfunction!");
    if(GRAPH_SUCCESS != VerifyConv3dbpFilterCommon(op)){
        return GRAPH_FAILED;
    }
    // check padding value
    if(GRAPH_SUCCESS != VerifyConv3dbpFilterPads(op)){
        return GRAPH_FAILED;
    }
    OP_LOGI(op.GetName().c_str(), "Leaving Conv3DBackpropFilterD verifyfunction!");
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Conv3DBackpropFilterD, Conv3DBackpropFilterDInfer);
VERIFY_FUNC_REG(Conv3DBackpropFilterD, Conv3DBackpropFilterDVerify);

//-----------------------------conv3d_transpose_common_check----------------------------
template<typename T1>
static bool CheckAllZero(const std::vector<T1>& list)
{
    if (list[0] == 0 && list[1] == 0 && list[2] == 0 && list[3] == 0 && list[4] == 0)
    {
        return true;
    }

  return false;
}

template<typename T1, typename T2, typename T3>
static bool SetInputsizeListConv3dtranspose(ge::Operator& op,
                                        const std::vector<T1>& xSizes,
                                        Format xFormat,
                                        const std::vector<T2>& filterSizes,
                                        Format filterFormat,
                                        const std::vector<T3>& inputSizes,
                                        Format inputFormat)
{
    //the shape of input_size may be 5
    const int32_t INPUT_SIZE_LIMIT = 5;
    const int32_t PADS_SIZE_LIMIT = 6;

    if(filterSizes.size() != INPUT_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "filter_sizes is illegal");
        map<string, string> err_map;
        err_map["param_name"] = "filter_size";
        err_map["op_name"] = "Conv3dTranspose";
        err_map["excepted_value"] = std::to_string(5);
        err_map["input_value"] = std::to_string(filterSizes.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    std::vector<int32_t> strideList;
    if (op.GetAttr("strides", strideList) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "op get strides failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dTranspose";
        err_map["param_name"] = "strides";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    if (strideList.size() != INPUT_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "op get strides failed.");
        map<string, string> err_map;
        err_map["param_name"] = "strides";
        err_map["op_name"] = "Conv3dTranspose";
        err_map["excepted_value"] = std::to_string(5);
        err_map["input_value"] = std::to_string(strideList.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    std::vector<int32_t> dilationsList;
    if (op.GetAttr("dilations", dilationsList) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "op get dilation failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dTranspose";
        err_map["param_name"] = "dilations";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    if (dilationsList.size() != INPUT_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "op get dilation failed.");
        map<string, string> err_map;
        err_map["param_name"] = "dilations";
        err_map["op_name"] = "Conv3dTranspose";
        err_map["excepted_value"] = std::to_string(5);
        err_map["input_value"] = std::to_string(strideList.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    std::vector<int32_t> outputPaddingList;
    if (op.GetAttr("output_padding", outputPaddingList) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "op get outputpadding failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dTranspose";
        err_map["param_name"] = "output_padding";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    if (outputPaddingList.size() != INPUT_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "op get outputpadding failed.");
        map<string, string> err_map;
        err_map["param_name"] = "output_padding";
        err_map["op_name"] = "Conv3dTranspose";
        err_map["excepted_value"] = std::to_string(5);
        err_map["input_value"] = std::to_string(outputPaddingList.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    std::string xFormatStr = format2str[xFormat];
    int32_t hInputPosition = xFormatStr.find("H");
    CHECK_POSITION(hInputPosition);
    int32_t wInputPosition = xFormatStr.find("W");
    CHECK_POSITION(wInputPosition);
    int32_t dInputPosition = xFormatStr.find("D");
    CHECK_POSITION(dInputPosition);
    int32_t cInputPosition = xFormatStr.find("C");
    CHECK_POSITION(cInputPosition);
    int32_t nInputPosition = xFormatStr.find("N");
    CHECK_POSITION(nInputPosition);
    int32_t dy_h = xSizes[hInputPosition];
    int32_t dy_w = xSizes[wInputPosition];
    int32_t dy_d = xSizes[dInputPosition];
    int32_t dy_n = xSizes[nInputPosition];

    int32_t stride_h = strideList[hInputPosition];
    int32_t stride_w = strideList[wInputPosition];
    int32_t stride_d = strideList[dInputPosition];

    int32_t dilation_h = dilationsList[hInputPosition];
    int32_t dilation_w = dilationsList[wInputPosition];
    int32_t dilation_d = dilationsList[dInputPosition];

    int32_t outputpadding_h = outputPaddingList[hInputPosition];
    int32_t outputpadding_w = outputPaddingList[wInputPosition];
    int32_t outputpadding_d = outputPaddingList[dInputPosition];

    std::string filterFormatStr = format2str[filterFormat];
    int32_t hFilterPosition = filterFormatStr.find("H");
    CHECK_POSITION(hFilterPosition);
    int32_t wFilterPosition = filterFormatStr.find("W");
    CHECK_POSITION(wFilterPosition);
    int32_t dFilterPosition = filterFormatStr.find("D");
    CHECK_POSITION(dFilterPosition);
    int32_t cFilterPosition = filterFormatStr.find("C");
    CHECK_POSITION(cFilterPosition);

    int32_t filter_h = filterSizes[hFilterPosition];
    int32_t filter_w = filterSizes[wFilterPosition];
    int32_t filter_d = filterSizes[dFilterPosition];
    int32_t filter_c = filterSizes[cFilterPosition];

    std::vector<int32_t> padsList;
    if (op.GetAttr("pads", padsList) == GRAPH_FAILED) {
        OP_LOGE(op.GetName().c_str(), "op get pads failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dTranspose";
        err_map["param_name"] = "pads";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    if (padsList.size() != PADS_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "op get outputpadding failed.");
        map<string, string> err_map;
        err_map["param_name"] = "output_padding";
        err_map["op_name"] = "Conv3dTranspose";
        err_map["excepted_value"] = std::to_string(5);
        err_map["input_value"] = std::to_string(outputPaddingList.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    int32_t pad_head = padsList[0];
    int32_t pad_tail = padsList[1];
    int32_t pad_up = padsList[2];
    int32_t pad_down = padsList[3];
    int32_t pad_left = padsList[4];
    int32_t pad_right = padsList[5];

    std::vector<int32_t> output;
    int32_t output_h = 0;
    int32_t output_w = 0;
    int32_t output_d = 0;
    int32_t output_n = 0;
    int32_t output_c = 0;
    if (!CheckAllZero(inputSizes)) {
        output_h = inputSizes[hInputPosition];
        output_w = inputSizes[wInputPosition];
        output_d = inputSizes[dInputPosition];
        output_n = inputSizes[nInputPosition];
        output_c = inputSizes[cInputPosition];

    } else {
        output_d = stride_d * (dy_d - 1) + outputpadding_d + ((filter_d - 1) * dilation_d + 1) - pad_head - pad_tail;
        output_h = stride_h * (dy_h - 1) + outputpadding_h + ((filter_h - 1) * dilation_h + 1) - pad_up - pad_down;
        output_w = stride_w * (dy_w - 1) + outputpadding_w + ((filter_w - 1) * dilation_w + 1) - pad_left - pad_right;
        output_n = dy_n;
        output_c = filter_c;
    }

    if (xFormat == FORMAT_NCDHW) {
        output.push_back(output_n);
        output.push_back(output_c);
        output.push_back(output_d);
        output.push_back(output_h);
        output.push_back(output_w);
    } else if (xFormat == FORMAT_NDHWC) {
        output.push_back(output_n);
        output.push_back(output_d);
        output.push_back(output_h);
        output.push_back(output_w);
        output.push_back(output_c);
    } else {
        OP_LOGE(op.GetName().c_str(), "inputSize format should be NCDHW or NDHWC.");
        map<string, string> err_map;
        err_map["param_name"] = "inputSize";
        err_map["op_name"] = "Conv3d";
        err_map["excepted_value"] = "NCDHW or NDHWC";
        err_map["input_value"] = xFormat;
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    //set input_size shape to dedx
    op.SetAttr("dedx", output);

    return true;
}

static graphStatus VerifyConv3dTransposeInput(ge::Operator& op) {
    auto filterDesc = op.GetInputDesc("filter");
    auto xDesc = op.GetInputDesc("x");

    auto filterDtype = filterDesc.GetDataType();
    auto xDtype = xDesc.GetDataType();
    auto filterShape = filterDesc.GetShape().GetDims();
    auto xShape = xDesc.GetShape().GetDims();

    const int32_t DIM_SIZE_LIMIT = 5;

    //check input dtype
    if (filterDtype != xDtype) {
        OP_LOGE(op.GetName().c_str(), "filter's dtype should equal to x's dtype.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dTranspose";
        err_map["attr_name"] = "dtype";
        err_map["param1_name"] = "filter";
        err_map["param2_name"] = "x";
        err_map["param1_value"] = std::to_string(filterDtype);
        err_map["param2_value"] = std::to_string(xDtype);
        std::string report_error_code = "E50031";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // check input tensor shape
    if (filterShape.size() != DIM_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "filter's shape should be 5d.");
        map<string, string> err_map;
        err_map["param_name"] = "filterShape_size";
        err_map["op_name"] = "Conv3dTranspose";
        err_map["excepted_value"] = std::to_string(5);
        err_map["input_value"] = std::to_string(filterShape.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    if (xShape.size() != DIM_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "x's shape should be 5d.");
        map<string, string> err_map;
        err_map["param_name"] = "xShape_size";
        err_map["op_name"] = "Conv3dTranspose";
        err_map["excepted_value"] = std::to_string(5);
        err_map["input_value"] = std::to_string(xShape.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // check strides shape
    std::vector<int32_t> strideList;
    if (op.GetAttr("strides", strideList) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "get strides list failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dTranspose";
        err_map["param_name"] = "strides";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    if (strideList.size() != DIM_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "strides should be 5d.");
        map<string, string> err_map;
        err_map["param_name"] = "strides";
        err_map["op_name"] = "Conv3dTranspose";
        err_map["excepted_value"] = std::to_string(5);
        err_map["input_value"] = std::to_string(strideList.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // check dilations shape
    std::vector<int32_t> dilationsList;
    if (op.GetAttr("dilations", dilationsList) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dTranspose";
        err_map["param_name"] = "dilations";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    if (dilationsList.size() != DIM_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "dilationsList list should be 5d.");
        map<string, string> err_map;
        err_map["param_name"] = "dilations";
        err_map["op_name"] = "Conv3dTranspose";
        err_map["excepted_value"] = std::to_string(5);
        err_map["input_value"] = std::to_string(dilationsList.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    std::vector<int32_t> outputPaddingList;
    if (op.GetAttr("output_padding",outputPaddingList) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "get output_padding list failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dTranspose";
        err_map["param_name"] = "output_padding";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    if (outputPaddingList.size() != DIM_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "output_paddingList list should be 5d.");
        map<string, string> err_map;
        err_map["param_name"] = "output_padding";
        err_map["op_name"] = "Conv3dTranspose";
        err_map["excepted_value"] = std::to_string(5);
        err_map["input_value"] = std::to_string(outputPaddingList.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}
//----------------Conv3DTranspose-------------------
IMPLEMT_INFERFUNC(Conv3DTranspose, Conv3DTransposeInfer)
{
    auto inputSizesDesc = op.GetInputDesc("input_size");
    auto yDesc = op.GetOutputDesc("y");

    Tensor inputSizesTensor;
    if (op.GetInputConstData("input_size", inputSizesTensor) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "get input_size tensor failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dTranspose";
        err_map["param_name"] = "inputSizesTensor";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // get shape for output from input_sizes
    DataType dtype = inputSizesDesc.GetDataType();
    std::vector<int64_t> inputSizes;
    GetConstValue(inputSizesTensor, dtype, inputSizes);

    // set dtype of x
    auto xDtype = op.GetInputDesc("x").GetDataType();
    std::vector<int64_t> filterSizes = op.GetInputDesc("filter").GetShape().GetDims();
    std::vector<int64_t> xSizes = op.GetInputDesc("x").GetShape().GetDims();
    Format filterFormat = op.GetInputDesc("filter").GetFormat();
    Format inputFormat = yDesc.GetFormat();
    Format xFormat = op.GetInputDesc("x").GetFormat();
    CHECK_FORMAT(filterFormat);
    CHECK_FORMAT(inputFormat);
    CHECK_FORMAT(xFormat);

    if (SetInputsizeListConv3dtranspose(op, xSizes, xFormat, filterSizes, filterFormat, inputSizes, inputFormat) != false) {
        OP_LOGE(op.GetName().c_str(), "Conv3DTransposeD update pads list by padding failed or calculate input sizes failed.");
        return GRAPH_FAILED;
    }

    yDesc.SetDataType(xDtype);
    // set shape of output desc, input_sizes should match the format of y
    std::vector<int64_t> yShape;
    for (auto i : inputSizes) {
        yShape.push_back(i);
    }
    yDesc.SetShape(ge::Shape(yShape));

    // update output desc
    if (op.UpdateOutputDesc("y", yDesc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update output desc failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3DTranspose";
        err_map["param_name"] = "output y";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv3DTranspose, Conv3DTransposeVerify)
{
    if (VerifyConv3dTransposeInput(op) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    // check padding value
    if (VerifyConvPadding(op) == GRAPH_SUCCESS || VerifyConv3dbpPads(op) == GRAPH_SUCCESS) {
        return GRAPH_SUCCESS;
    } else {
        OP_LOGE(op.GetName().c_str(), "Leaving Conv3DTranspose verifyfunction!");
        return GRAPH_FAILED;
    }
}

INFER_FUNC_REG(Conv3DTranspose, Conv3DTransposeInfer);
VERIFY_FUNC_REG(Conv3DTranspose, Conv3DTransposeVerify);
//----------------Conv3DTransposeD-------------------
IMPLEMT_INFERFUNC(Conv3DTransposeD, Conv3DTransposeDInfer)
{
    const int32_t DIM_SIZE_LIMIT = 5;
    auto xDesc = op.GetInputDesc("x");
    auto yDesc = op.GetOutputDesc("y");

    // get dtype for output from x
    auto xDtype = xDesc.GetDataType();
    // get shape for output from input_size
    std::vector<int32_t> inputSizes;
    if (op.GetAttr("input_size", inputSizes) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "get input_size list failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dTranspose";
        err_map["param_name"] = "input_size";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
    if (inputSizes.size() != DIM_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "input_size list should be 5d.");
        map<string, string> err_map;
        err_map["param_name"] = "input_size";
        err_map["op_name"] = "Conv3dTranspose";
        err_map["excepted_value"] = std::to_string(5);
        err_map["input_value"] = std::to_string(inputSizes.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    std::vector<int64_t> filterSizes =  op.GetInputDesc("filter").GetShape().GetDims();
    std::vector<int64_t> xSizes = op.GetInputDesc("x").GetShape().GetDims();
    Format filterFormat = op.GetInputDesc("filter").GetFormat();
    Format inputFormat = yDesc.GetFormat();
    Format xFormat = op.GetInputDesc("x").GetFormat();
    CHECK_FORMAT(filterFormat);
    CHECK_FORMAT(inputFormat);
    CHECK_FORMAT(xFormat);
    // update pads list by padding[SAME,VALID] and calculate input_size
    if (SetInputsizeListConv3dtranspose(op, xSizes, xFormat, filterSizes, filterFormat, inputSizes, inputFormat) == false) {
        OP_LOGE(op.GetName().c_str(), "Conv3DTransposeD update pads list by padding failed or calculate input sizes failed.");
        return GRAPH_FAILED;
    }
    // set dtype of output desc
    yDesc.SetDataType(xDtype);
    // set shape of output desc, input_size should match the format of y
    std::vector<int32_t> dedx;
    if (op.GetAttr("dedx", dedx) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "get dedx list failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dTranspose";
        err_map["param_name"] = "dedx";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    if (dedx.size() != DIM_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "dedx list should be 5d.");
        map<string, string> err_map;
        err_map["param_name"] = "dedx";
        err_map["op_name"] = "Conv3dTranspose";
        err_map["excepted_value"] = std::to_string(5);
        err_map["input_value"] = std::to_string(dedx.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    std::vector<int64_t> outShape;
    for (auto i : dedx) {
        outShape.push_back(i);
    }

    yDesc.SetShape(ge::Shape(outShape));
    // update input_size shape
    op.SetAttr("input_size",dedx);

    // update output desc
    if (op.UpdateOutputDesc("y", yDesc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update output desc failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dTranspose";
        err_map["param_name"] = "output y";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv3DTransposeD, Conv3DTransposeDVerify)
{
    if(VerifyConv3dTransposeInput(op) != GRAPH_SUCCESS){
        return GRAPH_FAILED;
    }
    // check padding value
    if(VerifyConv3dbpPads(op) != GRAPH_SUCCESS){
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Conv3DTransposeD, Conv3DTransposeDInfer);
VERIFY_FUNC_REG(Conv3DTransposeD, Conv3DTransposeDVerify);

//----------------Conv2DTransposeD-------------------
template<typename T1, typename T2, typename T3>
static bool SetInputsizeListConv2DTranspose(ge::Operator& op,
                                            const std::vector<T1>& xSizes,
                                            Format xFormat,
                                            const std::vector<T2>& filterSizes,
                                            Format filterFormat,
                                            const std::vector<T3>& inputSizes,
                                            Format inputFormat)
{
    //the shape of input_size may be 4
    const int32_t INPUT_SIZE_LIMIT = 4;
    const int32_t PADS_SIZE_LIMIT = 4;

    if (filterSizes.size() != INPUT_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "filter_sizes is illegal.");
        map<string, string> err_map;
        err_map["param_name"] = "filter_size";
        err_map["op_name"] = "Conv2DTranspose";
        err_map["excepted_value"] = std::to_string(4);
        err_map["input_value"] = std::to_string(filterSizes.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    std::vector<int32_t> strideList;
    if (op.GetAttr("strides", strideList) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "op get strides failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DTranspose";
        err_map["param_name"] = "strides";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    if (strideList.size() != INPUT_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "strideList size is illegal.");
        map<string, string> err_map;
        err_map["param_name"] = "strides";
        err_map["op_name"] = "Conv2DTranspose";
        err_map["excepted_value"] = std::to_string(4);
        err_map["input_value"] = std::to_string(strideList.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    std::vector<int32_t> dilationsList;
    if (op.GetAttr("dilations", dilationsList) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "op get dilation failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DTranspose";
        err_map["param_name"] = "dilations";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    if (dilationsList.size() != INPUT_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "dilationsList size is illegal.");
        map<string, string> err_map;
        err_map["param_name"] = "dilations";
        err_map["op_name"] = "Conv2DTranspose";
        err_map["excepted_value"] = std::to_string(4);
        err_map["input_value"] = std::to_string(strideList.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    std::vector<int32_t> outputPaddingList;
    if (op.GetAttr("output_padding", outputPaddingList) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "op get outputpadding failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DTranspose";
        err_map["param_name"] = "output_padding";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    if (outputPaddingList.size() != INPUT_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "outputpadding size is illegal.");
        map<string, string> err_map;
        err_map["param_name"] = "output_padding";
        err_map["op_name"] = "Conv2DTranspose";
        err_map["excepted_value"] = std::to_string(4);
        err_map["input_value"] = std::to_string(outputPaddingList.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    std::string xFormatStr = format2str[xFormat];
    int32_t hInputPosition = xFormatStr.find("H");
    CHECK_POSITION(hInputPosition);
    int32_t wInputPosition = xFormatStr.find("W");
    CHECK_POSITION(wInputPosition);
    int32_t cInputPosition = xFormatStr.find("C");
    CHECK_POSITION(cInputPosition);
    int32_t nInputPosition = xFormatStr.find("N");
    CHECK_POSITION(nInputPosition);
    int32_t dy_h = xSizes[hInputPosition];
    int32_t dy_w = xSizes[wInputPosition];
    int32_t dy_n = xSizes[nInputPosition];

    int32_t stride_h = strideList[hInputPosition];
    int32_t stride_w = strideList[wInputPosition];

    int32_t dilation_h = dilationsList[hInputPosition];
    int32_t dilation_w = dilationsList[wInputPosition];

    int32_t outputpadding_h = outputPaddingList[hInputPosition];
    int32_t outputpadding_w = outputPaddingList[wInputPosition];

    std::string filterFormatStr = format2str[filterFormat];
    int32_t hFilterPosition = filterFormatStr.find("H");
    CHECK_POSITION(hFilterPosition);
    int32_t wFilterPosition = filterFormatStr.find("W");
    CHECK_POSITION(wFilterPosition);
    int32_t cFilterPosition = filterFormatStr.find("C");
    CHECK_POSITION(cFilterPosition);

    int32_t filter_h = filterSizes[hFilterPosition];
    int32_t filter_w = filterSizes[wFilterPosition];
    int32_t filter_c = filterSizes[cFilterPosition];

    std::vector<int32_t> padsList;
    if (op.GetAttr("pads", padsList) == GRAPH_FAILED) {
        OP_LOGE(op.GetName().c_str(), "op get pads failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DTranspose";
        err_map["param_name"] = "pads";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    if (padsList.size() != PADS_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "op get padsList failed.");
        map<string, string> err_map;
        err_map["param_name"] = "output_padding";
        err_map["op_name"] = "Conv2DTranspose";
        err_map["excepted_value"] = std::to_string(4);
        err_map["input_value"] = std::to_string(padsList.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    int32_t pad_up = padsList[0];
    int32_t pad_down = padsList[1];
    int32_t pad_left = padsList[2];
    int32_t pad_right = padsList[3];

    std::vector<int32_t> output;
    int32_t output_h = 0;
    int32_t output_w = 0;
    int32_t output_n = 0;
    int32_t output_c = 0;
    if (!CheckAllZero(inputSizes)) {
        output_h = inputSizes[hInputPosition];
        output_w = inputSizes[wInputPosition];
        output_n = inputSizes[nInputPosition];
        output_c = inputSizes[cInputPosition];

    } else {
        output_h = stride_h * (dy_h - 1) + outputpadding_h + ((filter_h - 1) * dilation_h + 1) - pad_up - pad_down;
        output_w = stride_w * (dy_w - 1) + outputpadding_w + ((filter_w - 1) * dilation_w + 1) - pad_left - pad_right;
        output_n = dy_n;
        output_c = filter_c;
    }

    if (xFormat == FORMAT_NCHW) {
        output.push_back(output_n);
        output.push_back(output_c);
        output.push_back(output_h);
        output.push_back(output_w);
    } else if (xFormat == FORMAT_NHWC) {
        output.push_back(output_n);
        output.push_back(output_h);
        output.push_back(output_w);
        output.push_back(output_c);
    } else {
        OP_LOGE(op.GetName().c_str(), "inputSize format should be NCHW or NHWC.");
        map<string, string> err_map;
        err_map["param_name"] = "inputSize";
        err_map["op_name"] = "Conv2DTranspose";
        err_map["excepted_value"] = "NCHW or NHWC" ;
        err_map["input_value"] = xFormat;
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    //set input_size shape to dedx
    op.SetAttr("dedx", output);

    return true;
}

static graphStatus VerifyConv2DTransposeInput(ge::Operator& op) {
    auto filterDesc = op.GetInputDesc("filter");
    auto xDesc = op.GetInputDesc("x");

    auto filterDtype = filterDesc.GetDataType();
    auto xDtype = xDesc.GetDataType();
    auto filterShape = filterDesc.GetShape().GetDims();
    auto xShape = xDesc.GetShape().GetDims();

    const int32_t DIM_SIZE_LIMIT = 4;

    //check input dtype
    if (filterDtype != xDtype) {
        OP_LOGE(op.GetName().c_str(), "filter's dtype should equal to x's dtype.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DTranspose";
        err_map["attr_name"] = "dtype";
        err_map["param1_name"] = "filter";
        err_map["param2_name"] = "x";
        err_map["param1_value"] = std::to_string(filterDtype);
        err_map["param2_value"] = std::to_string(xDtype);
        std::string report_error_code = "E50031";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // check input tensor shape
    if (filterShape.size() != DIM_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "filter's shape should be 4d.");
        map<string, string> err_map;
        err_map["param_name"] = "filterShape_size";
        err_map["op_name"] = "Conv2DTranspose";
        err_map["excepted_value"] = std::to_string(4);
        err_map["input_value"] = std::to_string(filterShape.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    if (xShape.size() != DIM_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "x's shape should be 4d.");
        map<string, string> err_map;
        err_map["param_name"] = "xShape_size";
        err_map["op_name"] = "Conv2DTranspose";
        err_map["excepted_value"] = std::to_string(4);
        err_map["input_value"] = std::to_string(xShape.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // check strides shape
    std::vector<int32_t> strideList;
    if (op.GetAttr("strides", strideList) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "get strides list failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DTranspose";
        err_map["param_name"] = "strides";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    if (strideList.size() != DIM_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "strides should be 4d.");
        map<string, string> err_map;
        err_map["param_name"] = "strides";
        err_map["op_name"] = "Conv2DTranspose";
        err_map["excepted_value"] = std::to_string(4);
        err_map["input_value"] = std::to_string(strideList.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // check dilations shape
    std::vector<int32_t> dilationsList;
    if (op.GetAttr("dilations", dilationsList) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DTranspose";
        err_map["param_name"] = "dilations";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    if (dilationsList.size() != DIM_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "dilationsList list should be 4d.");
        map<string, string> err_map;
        err_map["param_name"] = "dilations";
        err_map["op_name"] = "Conv2DTranspose";
        err_map["excepted_value"] = std::to_string(4);
        err_map["input_value"] = std::to_string(dilationsList.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    std::vector<int32_t> outputPaddingList;
    if (op.GetAttr("output_padding",outputPaddingList) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "get output_padding list failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DTranspose";
        err_map["param_name"] = "output_padding";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    if (outputPaddingList.size() != DIM_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "output_paddingList list should be 4d.");
        map<string, string> err_map;
        err_map["param_name"] = "output_padding";
        err_map["op_name"] = "Conv2DTranspose";
        err_map["excepted_value"] = std::to_string(4);
        err_map["input_value"] = std::to_string(outputPaddingList.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}
//----------------Conv2DTranspose-------------------
IMPLEMT_INFERFUNC(Conv2DTranspose, Conv2DTransposeInfer)
{
    auto inputSizesDesc = op.GetInputDesc("input_size");
    auto yDesc = op.GetOutputDesc("y");

    Tensor inputSizesTensor;
    if (op.GetInputConstData("input_size", inputSizesTensor) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "get input_size tensor failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DTranspose";
        err_map["param_name"] = "inputSizesTensor";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // get shape for output from input_sizes
    DataType dtype = inputSizesDesc.GetDataType();
    std::vector<int64_t> inputSizes;
    GetConstValue(inputSizesTensor, dtype, inputSizes);

    // set dtype of x
    auto xDtype = op.GetInputDesc("x").GetDataType();
    std::vector<int64_t> filterSizes = op.GetInputDesc("filter").GetShape().GetDims();
    std::vector<int64_t> xSizes = op.GetInputDesc("x").GetShape().GetDims();
    Format filterFormat = op.GetInputDesc("filter").GetFormat();
    Format inputFormat = yDesc.GetFormat();
    Format xFormat = op.GetInputDesc("x").GetFormat();
    CHECK_FORMAT(filterFormat);
    CHECK_FORMAT(inputFormat);
    CHECK_FORMAT(xFormat);
    // update pads list by padding[SAME,VALID] and calculate input_size
    if (!SetInputsizeListConv2DTranspose(op, xSizes, xFormat, filterSizes, filterFormat,
        inputSizes, inputFormat)) {
        OP_LOGE(op.GetName().c_str(), "Set Conv2DTranspose InputsizeList failed.");
        return GRAPH_FAILED;
    }

    yDesc.SetDataType(xDtype);
    // set shape of output desc, input_sizes should match the format of y
    std::vector<int64_t> yShape;
    for (auto i : inputSizes) {
        yShape.push_back(i);
    }
    yDesc.SetShape(ge::Shape(yShape));

    // update output desc
    if (op.UpdateOutputDesc("y", yDesc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update output desc failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DTranspose";
        err_map["param_name"] = "output y";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv2DTranspose, Conv2DTransposeVerify)
{
    if (VerifyConv2DTransposeInput(op) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    // check padding value
    if (VerifyConvPadding(op) == GRAPH_SUCCESS || VerifyConv2dbpPads(op) == GRAPH_SUCCESS) {
        return GRAPH_SUCCESS;
    } else {
        OP_LOGE(op.GetName().c_str(), "Leaving Conv2DTranspose verifyfunction!");
        return GRAPH_FAILED;
    }
}

INFER_FUNC_REG(Conv2DTranspose, Conv2DTransposeInfer);
VERIFY_FUNC_REG(Conv2DTranspose, Conv2DTransposeVerify);
//----------------Conv2DTransposeD-------------------
IMPLEMT_INFERFUNC(Conv2DTransposeD, Conv2DTransposeDInfer)
{
    const int32_t DIM_SIZE_LIMIT = 4;
    auto xDesc = op.GetInputDesc("x");
    auto yDesc = op.GetOutputDesc("y");

    // get dtype for output from x
    auto xDtype = xDesc.GetDataType();
    // get shape for output from input_size
    std::vector<int32_t> inputSizes;
    if (op.GetAttr("input_size", inputSizes) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "get input_size list failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DTranspose";
        err_map["param_name"] = "input_size";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    if (inputSizes.size() != DIM_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "input_size list should be 4d.");
        map<string, string> err_map;
        err_map["param_name"] = "input_size";
        err_map["op_name"] = "Conv2DTranspose";
        err_map["excepted_value"] = std::to_string(4);
        err_map["input_value"] = std::to_string(inputSizes.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    std::vector<int64_t> filterSizes = op.GetInputDesc("filter").GetShape().GetDims();
    std::vector<int64_t> xSizes = op.GetInputDesc("x").GetShape().GetDims();
    Format filterFormat = op.GetInputDesc("filter").GetFormat();
    Format inputFormat = yDesc.GetFormat();
    Format xFormat = op.GetInputDesc("x").GetFormat();
    CHECK_FORMAT(filterFormat);
    CHECK_FORMAT(inputFormat);
    CHECK_FORMAT(xFormat);
    // update pads list by padding[SAME,VALID] and calculate input_size
    if (!SetInputsizeListConv2DTranspose(op, xSizes, xFormat, filterSizes, filterFormat,
        inputSizes, inputFormat)) {
        OP_LOGE(op.GetName().c_str(), "Set Conv2DTranspose InputsizeList failed.");
        return GRAPH_FAILED;
    }
    // set dtype of output desc
    yDesc.SetDataType(xDtype);
    // set shape of output desc, input_size should match the format of y
    std::vector<int32_t> dedx;
    if (op.GetAttr("dedx", dedx) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "get dedx list failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DTranspose";
        err_map["param_name"] = "dedx";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    if (dedx.size() != DIM_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "dedx list should be 4d.");
        map<string, string> err_map;
        err_map["param_name"] = "dedx";
        err_map["op_name"] = "Conv2DTranspose";
        err_map["excepted_value"] = std::to_string(4);
        err_map["input_value"] = std::to_string(dedx.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    std::vector<int64_t> outShape;
    for (auto i : dedx) {
        outShape.push_back(i);
    }

    yDesc.SetShape(ge::Shape(outShape));
    // update input_size shape
    op.SetAttr("input_size", dedx);

    // update output desc
    if (op.UpdateOutputDesc("y", yDesc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update output desc failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2DTranspose";
        err_map["param_name"] = "output y";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv2DTransposeD, Conv2DTransposeDVerify)
{
    if (VerifyConv2DTransposeInput(op) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    // check padding value
    if (VerifyConv2dbpPads(op) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Conv2DTransposeD, Conv2DTransposeDInfer);
VERIFY_FUNC_REG(Conv2DTransposeD, Conv2DTransposeDVerify);

} // namespace ge
