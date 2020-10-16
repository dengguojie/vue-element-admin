/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
/* reslove the complexity of pooling fuction. */

#include "inc/nn_pooling_ops.h"
#include <string.h>
#include <cmath>
#include <string>
#include <vector>
#include "graph/operator.h"
#include "op_log.h"
#include "common/util/error_manager/error_manager.h"
#include "util/common_shape_fns.h"
#include "util/error_util.h"
#include "util/util.h"

namespace ge {
// Obtains the value of the attr.
static std::vector<int64_t> GetAttrValue(const ge::Operator &op, const std::string &key_name)
{
    std::vector<int64_t> list;
    if (ge::GRAPH_SUCCESS != op.GetAttr(key_name, list)) {
        OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue failed!");
    }
    return list;
}

static bool CheckListEmpty(const std::string &opName, const std::vector<int64_t> &list, const std::string &attrName)
{
    if (list.empty()) {
        OP_LOGE(opName.c_str(), "the %s is empty !", attrName.c_str());
        return false;
    }
    return true;
}

static bool Construct3DPadsByPadding(std::string opName, ge::Operator &op, int32_t id, int32_t ih, int32_t iw,
    int32_t kd, int32_t kh, int32_t kw, int32_t strd, int32_t strh, int32_t strw)
{
    std::string padStr;
    std::vector<int32_t> padList;
    int32_t padf = 0;
    int32_t padba = 0;
    int32_t padt = 0;
    int32_t padb = 0;
    int32_t padl = 0;
    int32_t padr = 0;
    if (GRAPH_SUCCESS == op.GetAttr("padding", padStr)) {
        if (padStr.compare("SAME") == 0) {
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
            for (int32_t i = 0; i < 6; i++) {
                padList.push_back(0);
            }
        } else {
            map<string, string> err_map;
            err_map["param_name"] = "padding";
            err_map["op_name"] = opName;
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
        OP_LOGE(op.GetName().c_str(), "Failed to get pads!");
        return false;
    }

    auto pSize = padVec.size();
    // Check padVec.empty() for CODEX which is unnecessary
    if (padVec.empty() || (pSize != 6)) {
        map<string, string> err_map;
        err_map["param_name"] = "pads list";
        err_map["op_name"] = opName;
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
    if (padf < 0 || padba < 0 || padt < 0 || padb < 0 || padl < 0 || padr < 0) {
        map<string, string> err_map;
        err_map["param_name"] = "pads_list";
        err_map["op_name"] = "MaxPool3DGrad";
        err_map["excepted_value"] = "positive";
        err_map["input_value"] = std::to_string(padf) + " " + std::to_string(padba) + " " + std::to_string(padt) + " " +
            std::to_string(padb) + " " + std::to_string(padl) + " " + std::to_string(padr);
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    return true;
}


// ----------------Pooling-------------------
IMPLEMT_INFERFUNC(Pooling, PoolingInferShape)
{
    auto globalPooling = op.get_attr_global_pooling();
    auto window = op.get_attr_window();
    auto pad = op.get_attr_pad();
    auto stride = op.get_attr_stride();
    auto ceilMode = op.get_attr_ceil_mode();

    auto xShape = op.get_input_desc_x().GetShape().GetDims();
    auto xDtype = op.get_input_desc_x().GetDataType();
    auto xFormat = op.get_input_desc_x().GetFormat();

    int64_t inputN = 0;
    int64_t inputC = 0;
    int64_t inputH = 0;
    int64_t inputW = 0;

    if (xFormat == FORMAT_NCHW) {
        inputN = xShape[0];
        inputC = xShape[1];
        inputH = xShape[2];
        inputW = xShape[3];
    } else if (xFormat == FORMAT_NHWC) {
        inputN = xShape[0];
        inputC = xShape[3];
        inputH = xShape[1];
        inputW = xShape[2];
    } else {
        OP_LOGE(op.GetName().c_str(),
            "xFormat should be NCHW or NHWC."
            " actual is: %d",
            (int)xFormat);
        OpsInputFormatErrReport(op.GetName().c_str(), "xFormat", "NCHW or NHWC", Strcat(xFormat));
        return GRAPH_FAILED;
    }

    int64_t strH = stride[0];
    int64_t strW = stride[1];

    int64_t windowH = window[0];
    int64_t windowW = window[1];

    // update globalPooling default value
    if (globalPooling) {
        windowH = inputH;
        windowW = inputW;
        (void)op.set_attr_window({ windowH, windowW });
    }

    int64_t padT = pad[0];
    int64_t padB = pad[1];
    int64_t padL = pad[2];
    int64_t padR = pad[3];

    if ((padT != padB) || (padL != padR)) {
        OP_LOGE(op.GetName().c_str(), "padT should equals padB, and padL should equals padR in caffe");
        return GRAPH_FAILED;
    }

    // init output
    int64_t outputH = inputH;
    int64_t outputW = inputW;
    int64_t outputN = inputN;
    int64_t outputC = inputC;

    // update output
    if (ceilMode == 0) {
        outputH = static_cast<int64_t>(std::ceil((inputH + padT + padB - windowH) * 1.0f / strH)) + 1;
        outputW = static_cast<int64_t>(std::ceil((inputW + padL + padR - windowW) * 1.0f / strW)) + 1;
    } else if (ceilMode == 1) {
        outputH = static_cast<int64_t>(std::floor((inputH + padT + padB - windowH) * 1.0f / strH)) + 1;
        outputW = static_cast<int64_t>(std::floor((inputW + padL + padR - windowW) * 1.0f / strW)) + 1;
    } else {
        OP_LOGE(op.GetName().c_str(), "Unknown rounding mode.");
        return GRAPH_FAILED;
    }

    bool hasPad = padT || padL;
    if (hasPad) {
        // If we have padding, ensure that the last pooling starts strictly
        // inside the image (instead of at the padding); otherwise clip the last.
        if ((outputH - 1) * strH >= inputH + padT) {
            --outputH;
        }
        if ((outputW - 1) * strW >= inputW + padL) {
            --outputW;
        }

        // CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
        bool conditionH = ((outputH - 1) * strH) <= inputH + padT;
        if (!conditionH) {
            OP_LOGE(op.GetName().c_str(), "CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_)");
            return GRAPH_FAILED;
        }

        // CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
        bool conditionW = ((outputW - 1) * strW) <= inputW + padL;
        if (!conditionW) {
            OP_LOGE(op.GetName().c_str(), "CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_)");
            return GRAPH_FAILED;
        }
    }

    auto outdesc = op.GetOutputDesc("y");
    auto yFormat = outdesc.GetFormat();
    vector<int64_t> yShape;
    if (yFormat == FORMAT_NCHW) {
        yShape.push_back(outputN);
        yShape.push_back(outputC);
        yShape.push_back(outputH);
        yShape.push_back(outputW);
    } else if (yFormat == FORMAT_NHWC) {
        yShape.push_back(outputN);
        yShape.push_back(outputH);
        yShape.push_back(outputW);
        yShape.push_back(outputC);
    } else {
        OP_LOGE(op.GetName().c_str(),
            "yFormat should be NCHW or NHWC."
            " actual is: %d",
            (int)yFormat);
        OpsInputFormatErrReport(op.GetName().c_str(), "xFormat", "NCHW or NHWC", Strcat(xFormat));
        return GRAPH_FAILED;
    }

    outdesc.SetShape(Shape(yShape));
    outdesc.SetDataType(ge::DataType(xDtype));

    if (GRAPH_SUCCESS != op.update_output_desc_y(outdesc)) {
        OP_LOGE(op.GetName().c_str(), "update output desc failed.");
        return GRAPH_FAILED;
    }

    OP_LOGD(op.GetName().c_str(), "Leave PoolingInfer.");
    return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(Pooling, PoolingVerify)
{
    auto window = op.get_attr_window();
    auto stride = op.get_attr_stride();
    auto xShape = op.get_input_desc_x().GetShape().GetDims();
    auto pad = op.get_attr_pad();
    if (xShape.size() != 4) {
        OP_LOGE(op.GetName().c_str(), "pooling check input size is invalid");
        return GRAPH_FAILED;
    }
    if (window.size() != 2) {
        OP_LOGE(op.GetName().c_str(), "pooling check window size is invalid");
        return GRAPH_FAILED;
    }
    if (stride.size() != 2) {
        OP_LOGE(op.GetName().c_str(), "pooling check stride size is invalid");
        return GRAPH_FAILED;
    }
    if (pad.size() != 4) {
        OP_LOGE(op.GetName().c_str(), "pooling check pad size is invalid");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}
INFER_FUNC_REG(Pooling, PoolingInferShape);
VERIFY_FUNC_REG(Pooling, PoolingVerify);
// ----------------Pooling-------------------

// ----------------AvgPool-------------------
IMPLEMT_VERIFIER(AvgPool, AvgPoolVerify)
{
    auto ksize = op.get_attr_ksize();
    auto strides = op.get_attr_strides();
    auto xShape = op.get_input_desc_x().GetShape().GetDims();
    bool invalidParam = (xShape.size() != 4 || ksize.size() != 4 || strides.size() != 4);
    if (invalidParam) {
        OP_LOGE(op.GetName().c_str(), "avgpool check x ksize strides size is invalid");
        return GRAPH_FAILED;
    }
    std::string dataFormat;
    if (GRAPH_SUCCESS == op.GetAttr("data_format", dataFormat)) {
        if (dataFormat != "NHWC" && dataFormat != "NCHW" && dataFormat != "NC1HWC0") {
            string expected_format_list = Strcat("NHWC,NCHW,NC1HWC0");
            OpsInputFormatErrReport(op.GetName(), "data_format", expected_format_list, dataFormat);
            OP_LOGE(op.GetName().c_str(), "dataFormat only "
                "support 'NHWC', 'NCHW' and 'NC1HWC0'.");
            return GRAPH_FAILED;
        } else {
            if (dataFormat == "NHWC") {
                if (ksize[0] != 1 || ksize[3] != 1) {
                    OP_LOGE(op.GetName().c_str(), "Only supports pooling across "
                        "width/height, and other ksize dimension should be one");
                    return GRAPH_FAILED;
                }
                if (strides[0] != 1 || strides[3] != 1) {
                    OP_LOGE(op.GetName().c_str(), "Only supports pooling across "
                        "width/height, and other strides dimension should be one");
                    return GRAPH_FAILED;
                }
            } else {
                if (ksize[0] != 1 || ksize[1] != 1) {
                    OP_LOGE(op.GetName().c_str(), "Only supports pooling across "
                        "width/height, and other ksize dimension should be one");
                    return GRAPH_FAILED;
                }
                if (strides[0] != 1 || strides[1] != 1) {
                    OP_LOGE(op.GetName().c_str(), "Only supports pooling across "
                        "width/height, and other strides dimension should be one");
                    return GRAPH_FAILED;
                }
            }
        }
    }
    return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(AvgPool, AvgPoolInferShape)
{
    const size_t DIM_SIZE1 = 1;
    const size_t DIM_SIZE2 = 2;
    const size_t DIM_SIZE3 = 3;
    const size_t DIM_SIZE4 = 4;
    auto inputTensorDesc = op.GetInputDesc("x");
    auto shape = inputTensorDesc.GetShape();
    Format input_format = inputTensorDesc.GetFormat();
    // get input ksize
    std::vector<int32_t> ksizeList;
    if (GRAPH_SUCCESS != op.GetAttr("ksize", ksizeList)) {
        OpsGetAttrErrReport(op.GetName(), "ksize");
        OP_LOGE(op.GetName().c_str(), "GetOpAttr ksizeList failed!");
        return GRAPH_FAILED;
    }

    if (ksizeList.size() != DIM_SIZE4) {
        OpsAttrValueErrReport(op.GetName(), "length of ksize", Strcat(DIM_SIZE4), Strcat((size_t)ksizeList.size()));
        OP_LOGE(op.GetName().c_str(), "length of ksize must be equal to the "
            "length of shape!");
        return GRAPH_FAILED;
    }

    // get input strides
    std::vector<int32_t> stridesList;
    if (GRAPH_SUCCESS != op.GetAttr("strides", stridesList)) {
        OpsGetAttrErrReport(op.GetName(), "strides");
        OP_LOGE(op.GetName().c_str(), "GetOpAttr stridesList failed!");
        return GRAPH_FAILED;
    }

    if (stridesList.size() != DIM_SIZE4) {
        OpsAttrValueErrReport(op.GetName(), "length of strides", Strcat(DIM_SIZE4), Strcat((size_t)stridesList.size()));
        OP_LOGE(op.GetName().c_str(), "length of strides must be equal to "
            "the length of shape!");
        return GRAPH_FAILED;
    }

    // get input data_format
    std::string dataFormat;
    if (GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
        OpsGetAttrErrReport(op.GetName(), "data_format");
        OP_LOGE(op.GetName().c_str(), "The AvgPool op GetOpAttr data_format "
            "failed!");
        return GRAPH_FAILED;
    }

    // get input paddingMode
    std::string paddingMode;
    if (GRAPH_SUCCESS != op.GetAttr("padding", paddingMode)) {
        OpsGetAttrErrReport(op.GetName(), "padding");
        OP_LOGE(op.GetName().c_str(), "GetOpAttr padding failed!");
        return GRAPH_FAILED;
    }

    if (paddingMode != "SAME" && paddingMode != "VALID") {
        string expected_format_list = Strcat("SAME,VALID");
        OpsInputFormatErrReport(op.GetName(), "padding", expected_format_list, paddingMode);
        OP_LOGE(op.GetName().c_str(), "AvgPool can only support SAME or VALID "
            "padding mode!");
        return GRAPH_FAILED;
    }

    std::vector<int64_t> dims_input = shape.GetDims();

    // set for global avg pool
    if (dataFormat == "NHWC") {
        if (ksizeList[1] == -1 && ksizeList[2] == -1) {
            ksizeList[1] = dims_input[1];
            ksizeList[2] = dims_input[2];
        }
    } else {
        if (ksizeList[2] == -1 && ksizeList[3] == -1) {
            ksizeList[2] = dims_input[2];
            ksizeList[3] = dims_input[3];
        }
    }
    op.SetAttr("ksize", ksizeList);

    // set output shape
    std::vector<int64_t> dimVector;
    if (input_format == FORMAT_NHWC) {
        if (paddingMode == "SAME") {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE1) || (i == DIM_SIZE2)) {
                    if (dataFormat == "NHWC") {
                        int64_t dims = (dims_input[i] + stridesList[i] - 1) / stridesList[i];
                        dimVector.push_back(dims);
                    } else {
                        int64_t dims = (dims_input[i] + stridesList[i + 1] - 1) / stridesList[i + 1];
                        dimVector.push_back(dims);
                    }
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        } else {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE1) || (i == DIM_SIZE2)) {
                    if (dataFormat == "NHWC") {
                        int64_t dims = (dims_input[i] - ksizeList[i] + 1 + (stridesList[i] - 1)) / stridesList[i];
                        dimVector.push_back(dims);
                    } else {
                        int64_t dims =
                            (dims_input[i] - ksizeList[i + 1] + 1 + (stridesList[i + 1] - 1)) / stridesList[i + 1];
                        dimVector.push_back(dims);
                    }
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        }
    } else {
        if (paddingMode == "SAME") {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE2) || (i == DIM_SIZE3)) {
                    if (dataFormat == "NHWC") {
                        int64_t dims = (dims_input[i] + stridesList[i - 1] - 1) / stridesList[i - 1];
                        dimVector.push_back(dims);
                    } else {
                        int64_t dims = (dims_input[i] + stridesList[i] - 1) / stridesList[i];
                        dimVector.push_back(dims);
                    }
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        } else {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE2) || (i == DIM_SIZE3)) {
                    if (dataFormat == "NHWC") {
                        int64_t dims =
                            (dims_input[i] - ksizeList[i - 1] + 1 + (stridesList[i - 1] - 1)) / stridesList[i - 1];
                        dimVector.push_back(dims);
                    } else {
                        int64_t dims = (dims_input[i] - ksizeList[i] + 1 + (stridesList[i] - 1)) / stridesList[i];
                        dimVector.push_back(dims);
                    }
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        }
    }
    TensorDesc td = op.GetOutputDesc("y");
    DataType inputDtype = inputTensorDesc.GetDataType();
    Shape outputShape(dimVector);
    td.SetShape(outputShape);
    td.SetDataType(inputDtype);
    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AvgPool, AvgPoolInferShape);
VERIFY_FUNC_REG(AvgPool, AvgPoolVerify);
// ----------------AvgPool-------------------

// ----------------AvgPool3D-------------------
IMPLEMT_VERIFIER(AvgPool3D, AvgPool3DVerify)
{
    auto ksize = op.get_attr_ksize();
    auto strides = op.get_attr_strides();
    auto xShape = op.get_input_desc_x().GetShape().GetDims();
    bool invalidParam = (xShape.size() != 5 || ksize.size() != 5 || strides.size() != 5);
    if (invalidParam) {
        OP_LOGE(op.GetName().c_str(), "avgpool3d check x or ksize or strides size is invalid");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(AvgPool3D, AvgPool3DInferShape)
{
    auto inputTensorDesc = op.GetInputDesc("x");
    auto shape = inputTensorDesc.GetShape();
    Format input_format = inputTensorDesc.GetFormat();
    // get input ksize
    std::vector<int32_t> ksizeList;
    if (GRAPH_SUCCESS != op.GetAttr("ksize", ksizeList)) {
        OP_LOGE(op.GetName().c_str(), "GetOpAttr ksizeList failed!");
        return GRAPH_FAILED;
    }

    // get input strides
    std::vector<int32_t> stridesList;
    if (GRAPH_SUCCESS != op.GetAttr("strides", stridesList)) {
        OP_LOGE(op.GetName().c_str(), "GetOpAttr stridesList failed!");
        return GRAPH_FAILED;
    }

    // get input data_format
    std::string dataFormat;
    if (GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
        OP_LOGE(op.GetName().c_str(), "The AvgPool3D op GetOpAttr data_format failed!");
        return GRAPH_FAILED;
    }

    std::vector<int64_t> dims_input = shape.GetDims();
    // set output shape
    std::vector<int64_t> dimVector;
    if (input_format == FORMAT_NDHWC) {
        for (size_t i = 0; i < dims_input.size(); i++) {
            int64_t dims = dims_input[i];
            if (i == 1 || i == 2 || i == 3) {
                dims = (dims_input[i] - ksizeList[i] + 1 + (stridesList[i] - 1)) / stridesList[i];
            }
            dimVector.push_back(dims);
        }
    }

    TensorDesc td = op.GetOutputDesc("y");
    DataType inputDtype = inputTensorDesc.GetDataType();
    Shape outputShape(dimVector);
    td.SetShape(outputShape);
    td.SetDataType(inputDtype);
    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AvgPool3D, AvgPool3DInferShape);
VERIFY_FUNC_REG(AvgPool3D, AvgPool3DVerify);
// ----------------AvgPool3D-------------------

// ----------------MaxPool-------------------
IMPLEMT_INFERFUNC(MaxPool, MaxPoolInferShape)
{
    const size_t DIM_SIZE1 = 1;
    const size_t DIM_SIZE2 = 2;
    const size_t DIM_SIZE3 = 3;
    const size_t DIM_SIZE4 = 4;
    auto inputTensorDesc = op.GetInputDesc("x");
    auto shape = inputTensorDesc.GetShape();
    Format input_format = inputTensorDesc.GetFormat();
    // get input ksize
    std::vector<int32_t> ksizeList;
    if (GRAPH_SUCCESS != op.GetAttr("ksize", ksizeList)) {
        OpsGetAttrErrReport(op.GetName(), "ksize");
        OP_LOGE(op.GetName().c_str(), "GetOpAttr ksizeList failed!");
        return GRAPH_FAILED;
    }

    if (ksizeList.size() != DIM_SIZE4) {
        OpsAttrValueErrReport(op.GetName(), "length of ksize", Strcat(DIM_SIZE4), Strcat((size_t)ksizeList.size()));
        OP_LOGE(op.GetName().c_str(), "length of ksize must be equal to the "
            "length of shape!");
        return GRAPH_FAILED;
    }

    // get input strides
    std::vector<int32_t> stridesList;
    if (GRAPH_SUCCESS != op.GetAttr("strides", stridesList)) {
        OpsGetAttrErrReport(op.GetName(), "strides");
        OP_LOGE(op.GetName().c_str(), "GetOpAttr stridesList failed!");
        return GRAPH_FAILED;
    }

    if (stridesList.size() != DIM_SIZE4) {
        OpsAttrValueErrReport(op.GetName(), "length of strides", Strcat(DIM_SIZE4), Strcat(stridesList.size()));
        OP_LOGE(op.GetName().c_str(), "length of strides must be equal to "
            "the length of shape!");
        return GRAPH_FAILED;
    }
    // get input data_format
    std::string dataFormat;
    if (GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
        OpsGetAttrErrReport(op.GetName(), "data_format");
        OP_LOGE(op.GetName().c_str(), "The MaxPool op GetOpAttr data_format "
            "failed!");
        return GRAPH_FAILED;
    }
    if (dataFormat != "NHWC" && dataFormat != "NCHW" && dataFormat != "NC1HWC0") {
        string expected_format_list = Strcat("NHWC,NCHW,NC1HWC0");
        OpsInputFormatErrReport(op.GetName(), "data_format", expected_format_list, dataFormat);
        OP_LOGE(op.GetName().c_str(), "data_format only "
            "support 'NHWC','NCHW' and 'NC1HWC0'.");
        return GRAPH_FAILED;
    }
    if (dataFormat == "NHWC") {
        if ((ksizeList[0] != 1) || (ksizeList[3] != 1) || (stridesList[0] != 1) || (stridesList[3] != 1)) {
            OP_LOGE(op.GetName().c_str(), "MaxPool only supports pooling across width/height"
                "and other ksize dimension should be one");
            return GRAPH_FAILED;
        }
    }
    if ((dataFormat == "NCHW") || (dataFormat == "NC1HWC0")) {
        if ((ksizeList[0] != 1) || (ksizeList[1] != 1) || (stridesList[0] != 1) || (stridesList[1] != 1)) {
            OP_LOGE(op.GetName().c_str(), "MaxPool only supports pooling across width/height"
                "and other ksize dimension should be one");
            return GRAPH_FAILED;
        }
    }
    // get input paddingMode
    std::string paddingMode;
    if (GRAPH_SUCCESS != op.GetAttr("padding", paddingMode)) {
        OpsGetAttrErrReport(op.GetName(), "padding");
        OP_LOGE(op.GetName().c_str(), "GetOpAttr padding failed!");
        return GRAPH_FAILED;
    }

    if (paddingMode != "SAME" && paddingMode != "VALID") {
        string expected_format_list = Strcat("SAME,VALID");
        OpsInputFormatErrReport(op.GetName(), "padding", expected_format_list, paddingMode);
        OP_LOGE(op.GetName().c_str(), "MaxPool can only support SAME or VALID "
            "padding mode!");
        return GRAPH_FAILED;
    }

    std::vector<int64_t> dims_input = shape.GetDims();
    // set output shape
    std::vector<int64_t> dimVector;
    if (input_format == FORMAT_NHWC) {
        if (paddingMode == "SAME") {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE1) || (i == DIM_SIZE2)) {
                    if (dataFormat == "NHWC") {
                        int64_t dims = (dims_input[i] + stridesList[i] - 1) / stridesList[i];
                        dimVector.push_back(dims);
                    } else {
                        int64_t dims = (dims_input[i] + stridesList[i + 1] - 1) / stridesList[i + 1];
                        dimVector.push_back(dims);
                    }
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        } else {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE1) || (i == DIM_SIZE2)) {
                    if (dataFormat == "NHWC") {
                        int64_t dims = (dims_input[i] - ksizeList[i] + 1 + (stridesList[i] - 1)) / stridesList[i];
                        dimVector.push_back(dims);
                    } else {
                        int64_t dims =
                            (dims_input[i] - ksizeList[i + 1] + 1 + (stridesList[i + 1] - 1)) / stridesList[i + 1];
                        dimVector.push_back(dims);
                    }
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        }
    } else {
        if (paddingMode == "SAME") {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE2) || (i == DIM_SIZE3)) {
                    if (dataFormat == "NHWC") {
                        int64_t dims = (dims_input[i] + stridesList[i - 1] - 1) / stridesList[i - 1];
                        dimVector.push_back(dims);
                    } else {
                        int64_t dims = (dims_input[i] + stridesList[i] - 1) / stridesList[i];
                        dimVector.push_back(dims);
                    }
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        } else {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE2) || (i == DIM_SIZE3)) {
                    if (dataFormat == "NHWC") {
                        int64_t dims =
                            (dims_input[i] - ksizeList[i - 1] + 1 + (stridesList[i - 1] - 1)) / stridesList[i - 1];
                        dimVector.push_back(dims);
                    } else {
                        int64_t dims = (dims_input[i] - ksizeList[i] + 1 + (stridesList[i] - 1)) / stridesList[i];
                        dimVector.push_back(dims);
                    }
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        }
    }
    TensorDesc td = op.GetOutputDesc("y");
    DataType inputDtype = inputTensorDesc.GetDataType();
    Shape outputShape(dimVector);
    td.SetShape(outputShape);
    td.SetDataType(inputDtype);
    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MaxPool, MaxPoolInferShape);
// ----------------MaxPool-------------------

// ----------------MaxPool3D-------------------
IMPLEMT_INFERFUNC(MaxPool3D, MaxPool3DInferShape)
{
    const size_t DIM_SIZE1 = 1;
    const size_t DIM_SIZE3 = 3;
    const size_t DIM_SIZE5 = 5;
    auto inputTensorDesc = op.GetInputDesc("x");
    auto shape = inputTensorDesc.GetShape();
    Format input_format = inputTensorDesc.GetFormat();
    // get input ksize
    std::vector<int32_t> ksizeList;
    if (GRAPH_SUCCESS != op.GetAttr("ksize", ksizeList)) {
        OpsGetAttrErrReport(op.GetName(), "ksize");
        OP_LOGE(op.GetName().c_str(), "GetOpAttr ksizeList failed!");
        return GRAPH_FAILED;
    }

    if ((ksizeList.size() != DIM_SIZE1) && (ksizeList.size() != DIM_SIZE3) && (ksizeList.size() != DIM_SIZE5)) {
        string excepted_value = Strcat(DIM_SIZE1, DIM_SIZE3, DIM_SIZE5);
        OpsAttrValueErrReport(op.GetName(), "length of ksize", excepted_value, Strcat((size_t)ksizeList.size()));
        OP_LOGE(op.GetName().c_str(), "length of ksize must be  1 or 3 or 5!");
        return GRAPH_FAILED;
    }

    std::vector<int64_t> ksizeTempList;
    if (ksizeList.size() == DIM_SIZE1) {
        ksizeTempList.push_back(1);
        ksizeTempList.push_back(ksizeList[0]);
        ksizeTempList.push_back(ksizeList[0]);
        ksizeTempList.push_back(ksizeList[0]);
        ksizeTempList.push_back(1);
    }

    if (ksizeList.size() == DIM_SIZE3) {
        ksizeTempList.push_back(1);
        ksizeTempList.push_back(ksizeList[0]);
        ksizeTempList.push_back(ksizeList[1]);
        ksizeTempList.push_back(ksizeList[2]);
        ksizeTempList.push_back(1);
    }

    if (ksizeList.size() == DIM_SIZE5) {
        ksizeTempList.push_back(ksizeList[0]);
        ksizeTempList.push_back(ksizeList[1]);
        ksizeTempList.push_back(ksizeList[2]);
        ksizeTempList.push_back(ksizeList[3]);
        ksizeTempList.push_back(ksizeList[4]);
    }

    // get input strides
    std::vector<int32_t> stridesList;
    if (GRAPH_SUCCESS != op.GetAttr("strides", stridesList)) {
        OpsGetAttrErrReport(op.GetName(), "strides");
        OP_LOGE(op.GetName().c_str(), "GetOpAttr stridesList failed!");
        return GRAPH_FAILED;
    }

    if ((stridesList.size() != DIM_SIZE1) && (stridesList.size() != DIM_SIZE3) && (stridesList.size() != DIM_SIZE5)) {
        string excepted_value = Strcat(DIM_SIZE1, DIM_SIZE3, DIM_SIZE5);
        OpsAttrValueErrReport(op.GetName(), "length of strides", excepted_value, Strcat((size_t)stridesList.size()));
        OP_LOGE(op.GetName().c_str(), "length of strides must be  1 or 3 or 5!");
        return GRAPH_FAILED;
    }

    std::vector<int64_t> stridesTempList;
    if (stridesList.size() == DIM_SIZE1) {
        stridesTempList.push_back(1);
        stridesTempList.push_back(stridesList[0]);
        stridesTempList.push_back(stridesList[0]);
        stridesTempList.push_back(stridesList[0]);
        stridesTempList.push_back(1);
    }

    if (stridesList.size() == DIM_SIZE3) {
        stridesTempList.push_back(1);
        stridesTempList.push_back(stridesList[0]);
        stridesTempList.push_back(stridesList[1]);
        stridesTempList.push_back(stridesList[2]);
        stridesTempList.push_back(1);
    }

    if (stridesList.size() == DIM_SIZE5) {
        stridesTempList.push_back(stridesList[0]);
        stridesTempList.push_back(stridesList[1]);
        stridesTempList.push_back(stridesList[2]);
        stridesTempList.push_back(stridesList[3]);
        stridesTempList.push_back(stridesList[4]);
    }

    // get input data_format
    std::string dataFormat;
    if (GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
        OpsGetAttrErrReport(op.GetName(), "data_format");
        OP_LOGE(op.GetName().c_str(), "The MaxPool3D op GetOpAttr data_format "
            "failed!");
        return GRAPH_FAILED;
    }

    // get input paddingMode
    std::string paddingMode;
    if (GRAPH_SUCCESS != op.GetAttr("padding", paddingMode)) {
        OpsGetAttrErrReport(op.GetName(), "padding");
        OP_LOGE(op.GetName().c_str(), "GetOpAttr padding failed!");
        return GRAPH_FAILED;
    }

    if (paddingMode != "SAME" && paddingMode != "VALID") {
        string excepted_value = Strcat("SAME,VALID");
        OpsAttrValueErrReport(op.GetName(), "padding", excepted_value, paddingMode);
        OP_LOGE(op.GetName().c_str(), "MaxPool3D can only support SAME or VALID "
            "padding mode!");
        return GRAPH_FAILED;
    }

    std::vector<int64_t> dims_input = shape.GetDims();
    // set output shape
    std::vector<int64_t> dimVector;
    if (input_format == FORMAT_NDHWC) {
        if (paddingMode == "SAME") {
            for (size_t i = 0; i < dims_input.size(); i++) {
                // D H W calculate, N C stride default 1
                int64_t dims = (dims_input[i] + stridesTempList[i] - 1) / stridesTempList[i];
                dimVector.push_back(dims);
            }
        } else {
            for (size_t i = 0; i < dims_input.size(); i++) {
                // D H W calculate, N C stride default 1
                int64_t dims = (dims_input[i] - ksizeTempList[i] + 1 + (stridesTempList[i] - 1)) / stridesTempList[i];
                dimVector.push_back(dims);
            }
        }
    }

    TensorDesc td = op.GetOutputDesc("y");
    DataType inputDtype = inputTensorDesc.GetDataType();
    Shape outputShape(dimVector);
    td.SetShape(outputShape);
    td.SetDataType(inputDtype);
    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MaxPool3D, MaxPool3DVerify)
{
    // verify in infer func
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MaxPool3D, MaxPool3DInferShape);
VERIFY_FUNC_REG(MaxPool3D, MaxPool3DVerify);
// ----------------MaxPool3D-------------------

// ---------------------MaxPool3DGradGrad---------------------
static bool GetAttrsMaxPool3DGradGrad(ge::Operator &op, Format refer, int32_t &strd, int32_t &strh, int32_t &strw,
    int32_t &kd, int32_t &kh, int32_t &kw)
{
    std::vector<int32_t> strideList;
    if (GRAPH_SUCCESS != op.GetAttr("strides", strideList)) {
        OP_LOGE(op.GetName().c_str(), "Failed to get strides!");
        return GRAPH_FAILED;
    }

    std::vector<int32_t> ksizeList;
    if (GRAPH_SUCCESS != op.GetAttr("ksize", ksizeList)) {
        OP_LOGE(op.GetName().c_str(), "Failed to get ksize!");
        return GRAPH_FAILED;
    }

    if (ksizeList.size() == 1) {
        kd = kh = kw = ksizeList[0];
    } else if (ksizeList.size() == 3) {
        kd = ksizeList[0];
        kh = ksizeList[1];
        kw = ksizeList[2];
    } else if (ksizeList.size() == 5) {
        if (refer == FORMAT_NCDHW) {
            kd = ksizeList[2];
            kh = ksizeList[3];
            kw = ksizeList[4];
        } else {
            kd = ksizeList[1];
            kh = ksizeList[2];
            kw = ksizeList[3];
        }
    }

    if (strideList.size() == 1) {
        strd = strh = strw = strideList[0];
    } else if (strideList.size() == 3) {
        strd = strideList[0];
        strh = strideList[1];
        strw = strideList[2];
    } else if (strideList.size() == 5) {
        if (refer == FORMAT_NCDHW) {
            strd = strideList[2];
            strh = strideList[3];
            strw = strideList[4];
        } else {
            strd = strideList[1];
            strh = strideList[2];
            strw = strideList[3];
        }
    }

    return true;
}

IMPLEMT_VERIFIER(MaxPool3DGradGrad, MaxPool3DGradGradVerify)
{
    OP_LOGD(op.GetName().c_str(), "Entering MaxPool3DGradGradDVerify");
    if (!CheckTwoInputDtypeSame(op, "orig_x", "orig_y") || !CheckTwoInputDtypeSame(op, "orig_x", "grads")) {
        OP_LOGE(op.GetName().c_str(), "MaxPool3DGradGrad, two input dtypes must be same");
        return GRAPH_FAILED;
    }
    std::vector<int64_t> ksize;
    ksize = GetAttrValue(op, "ksize");
    if (!CheckListEmpty(op.GetName(), ksize, "ksize")) {
        OP_LOGE(op.GetName().c_str(), "get attr ksize failed");
        return GRAPH_FAILED;
    }
    if (ksize.size() != 1 and ksize.size() != 3 and ksize.size() != 5) {
        OP_LOGE(op.GetName().c_str(), "attr ksize(%d) must be 1 3 or 5", (int)ksize.size());
        return GRAPH_FAILED;
    }
    std::vector<int64_t> strides;
    strides = GetAttrValue(op, "strides");
    if (!CheckListEmpty(op.GetName(), strides, "strides")) {
        OP_LOGE(op.GetName().c_str(), "get attr strides failed");
        return GRAPH_FAILED;
    }
    if (strides.size() != 1 and strides.size() != 3 and strides.size() != 5) {
        OP_LOGE(op.GetName().c_str(), "attr strides(%d) must be 1 3 or 5", (int)strides.size());
        return GRAPH_FAILED;
    }

    std::string data_format;
    if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
        if (data_format != "NCDHW" && data_format != "NDHWC") {
            OP_LOGE(op.GetName().c_str(), "attr data_format(%s) only support NCDHW and NDHWC", data_format.c_str());
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(MaxPool3DGradGrad, MaxPool3DGradGradInferShape)
{
    OP_LOGD(op.GetName().c_str(), "Entering MaxPool3DGradGradInferShape");
    auto input_orig_out = op.GetInputDesc("orig_y");
    auto shape_orig_out = input_orig_out.GetShape();
    auto type_orig_out = input_orig_out.GetDataType();

    TensorDesc td = op.GetOutputDesc("y");
    td.SetShape(shape_orig_out);
    td.SetDataType(type_orig_out);
    (void)op.UpdateOutputDesc("y", td);

    // get input DHW
    auto xShape = op.GetInputDesc("orig_x").GetShape().GetDims();
    auto xFormat = op.GetInputDesc("orig_x").GetFormat();
    int32_t id = 0;
    int32_t ih = 0;
    int32_t iw = 0;

    if ((xFormat == FORMAT_NCDHW) && (xShape.size() >= 5)) {
        id = xShape[2];
        ih = xShape[3];
        iw = xShape[4];
    } else if ((xFormat == FORMAT_NDHWC) && (xShape.size() >= 5)) {
        id = xShape[1];
        ih = xShape[2];
        iw = xShape[3];
    } else {
        map<string, string> err_map;
        err_map["param_name"] = "xFormat";
        err_map["op_name"] = "MaxPool3DGradGrad";
        err_map["excepted_value"] = "NCDHW or NDHWC";
        err_map["input_value"] = xFormat;
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // get ksize and strides
    int32_t kd = 0;
    int32_t kh = 0;
    int32_t kw = 0;
    int32_t strd = 0;
    int32_t strh = 0;
    int32_t strw = 0;
    if (false == GetAttrsMaxPool3DGradGrad(op, xFormat, strd, strh, strw, kd, kh, kw)) {
        OP_LOGE(op.GetName().c_str(), "Failed to get attr in MaxPool3DGradGrad");
        return GRAPH_FAILED;
    }
    if ((strd == 0) || (strh == 0) || (strw == 0)) {
        OP_LOGE(op.GetName().c_str(), "strd/strh/strw should not be zero");
        return GRAPH_FAILED;
    }
    // construct pads attr
    if (false == Construct3DPadsByPadding("MaxPool3DGradGradD", op, id, ih, iw, kd, kh, kw, strd, strh, strw)) {
        OP_LOGE(op.GetName().c_str(), "Failed to get pads in MaxPool3DGradGrad");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MaxPool3DGradGrad, MaxPool3DGradGradInferShape);
VERIFY_FUNC_REG(MaxPool3DGradGrad, MaxPool3DGradGradVerify);
// ---------------------MaxPool3DGradGrad---------------------


// ----------------MaxPoolGrad-------------------
IMPLEMT_VERIFIER(MaxPoolGrad, MaxPoolGradVerify)
{
    if (!CheckTwoInputDtypeSame(op, "x1", "x2") || !CheckTwoInputDtypeSame(op, "x1", "grad")) {
        return GRAPH_FAILED;
    }
    std::vector<int64_t> ksize;
    ksize = GetAttrValue(op, "ksize");
    if (!CheckListEmpty(op.GetName(), ksize, "ksize")) {
        OP_LOGE(op.GetName().c_str(), "get attr ksize failed");
        return GRAPH_FAILED;
    }
    if (ksize.size() != 4) {
        OP_LOGE(op.GetName().c_str(), "attr ksize(%d) must be 4", (int)ksize.size());
        return GRAPH_FAILED;
    }
    std::vector<int64_t> strides;
    strides = GetAttrValue(op, "strides");
    if (!CheckListEmpty(op.GetName(), strides, "strides")) {
        OP_LOGE(op.GetName().c_str(), "get attr strides failed");
        return GRAPH_FAILED;
    }
    if (strides.size() != 4) {
        OP_LOGE(op.GetName().c_str(), "attr strides(%d) must be 4", (int)strides.size());
        return GRAPH_FAILED;
    }
    std::string padding;
    if (ge::GRAPH_SUCCESS != op.GetAttr("padding", padding)) {
        OP_LOGE(op.GetName().c_str(), "Get padding failed!");
        return GRAPH_FAILED;
    }
    if (padding != "SAME" && padding != "VALID") {
        OP_LOGE(op.GetName().c_str(), "attr padding(%s) only support SAME and VALID", padding.c_str());
        return GRAPH_FAILED;
    }
    std::string data_format;
    if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
        if (data_format != "NCHW" && data_format != "NHWC") {
            OP_LOGE(op.GetName().c_str(), "attr data_format(%s) only support NCHW and NHWC", data_format.c_str());
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(MaxPoolGrad, MaxPoolGradInferShape)
{
    auto shapeX1 = op.GetInputDesc("x1").GetShape();
    auto inputType = op.GetInputDesc("x1").GetDataType();

    TensorDesc td = op.GetOutputDesc("y");
    td.SetShape(shapeX1);
    td.SetDataType(inputType);
    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MaxPoolGrad, MaxPoolGradInferShape);
VERIFY_FUNC_REG(MaxPoolGrad, MaxPoolGradVerify);
// ---------------------MaxPoolGrad---------------------

// ---------------------MaxPoolGradGrad---------------------
IMPLEMT_VERIFIER(MaxPoolGradGrad, MaxPoolGradGradVerify)
{
    if (!CheckTwoInputDtypeSame(op, "x1", "x2") || !CheckTwoInputDtypeSame(op, "x1", "grad")) {
        OP_LOGE(op.GetName().c_str(), "two input dtypes must be same");
        return GRAPH_FAILED;
    }
    std::vector<int64_t> ksize;
    ksize = GetAttrValue(op, "ksize");
    if (!CheckListEmpty(op.GetName(), ksize, "ksize")) {
        OP_LOGE(op.GetName().c_str(), "get attr ksize failed");
        return GRAPH_FAILED;
    }
    if (ksize.size() != 4) {
        OP_LOGE(op.GetName().c_str(), "attr ksize(%d) must be 4", (int)ksize.size());
        return GRAPH_FAILED;
    }
    std::vector<int64_t> strides;
    strides = GetAttrValue(op, "strides");
    if (!CheckListEmpty(op.GetName(), strides, "strides")) {
        OP_LOGE(op.GetName().c_str(), "get attr strides failed");
        return GRAPH_FAILED;
    }
    if (strides.size() != 4) {
        OP_LOGE(op.GetName().c_str(), "attr strides(%d) must be 4", (int)strides.size());
        return GRAPH_FAILED;
    }
    std::string padding;
    if (ge::GRAPH_SUCCESS != op.GetAttr("padding", padding)) {
        OP_LOGE(op.GetName().c_str(), "Get padding failed!");
        return GRAPH_FAILED;
    }
    if (padding != "SAME" && padding != "VALID") {
        OP_LOGE(op.GetName().c_str(), "attr padding(%s) only support SAME and VALID", padding.c_str());
        return GRAPH_FAILED;
    }
    std::string data_format;
    if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
        if (data_format != "NCHW" && data_format != "NHWC") {
            OP_LOGE(op.GetName().c_str(), "attr data_format(%s) only support NCHW and NHWC", data_format.c_str());
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(MaxPoolGradGrad, MaxPoolGradGradInferShape)
{
    auto inputX2 = op.GetInputDesc("x2");
    auto shapeX2 = inputX2.GetShape();
    auto typeX2 = inputX2.GetDataType();

    TensorDesc td = op.GetOutputDesc("y");
    td.SetShape(shapeX2);
    td.SetDataType(typeX2);
    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MaxPoolGradGrad, MaxPoolGradGradInferShape);
VERIFY_FUNC_REG(MaxPoolGradGrad, MaxPoolGradGradVerify);
// ---------------------MaxPoolGradGrad---------------------

// ---------------------MaxPoolExt2---------------------
IMPLEMT_INFERFUNC(MaxPoolExt2, MaxPoolExt2InferShape)
{
    const size_t DIM_SIZE2 = 2;
    const size_t DIM_SIZE3 = 3;
    const size_t DIM_SIZE4 = 4;
    auto inputTensorDesc = op.GetInputDesc("x");
    auto shape = inputTensorDesc.GetShape();
    Format input_format = inputTensorDesc.GetFormat();
    // get input ksize
    std::vector<int32_t> ksizeList;
    if (GRAPH_SUCCESS != op.GetAttr("ksize", ksizeList)) {
        OpsGetAttrErrReport(op.GetName(), "ksize");
        OP_LOGW(op.GetName().c_str(), "GetOpAttr ksizeList failed!");
        std::vector<int64_t> outputVec = { -1, -1, -1, -1, -1 };
        TensorDesc td = op.GetOutputDesc("y");
        DataType inputDtype = inputTensorDesc.GetDataType();
        Shape outputShape(outputVec);
        td.SetShape(outputShape);
        td.SetDataType(inputDtype);
        (void)op.UpdateOutputDesc("y", td);
        return GRAPH_SUCCESS;
    }

    if (ksizeList.size() != DIM_SIZE4) {
        string excepted_value = Strcat("equal to the length of x'shape[", DIM_SIZE4, "]");
        OpsAttrValueErrReport(op.GetName(), "ksize'length", excepted_value, Strcat(ksizeList.size()));
        OP_LOGE(op.GetName().c_str(), "length of ksize must be equal to the "
            "length of shape!");
        return GRAPH_FAILED;
    }

    // get input strides
    std::vector<int32_t> stridesList;
    if (GRAPH_SUCCESS != op.GetAttr("strides", stridesList)) {
        OpsGetAttrErrReport(op.GetName(), "strides");
        OP_LOGE(op.GetName().c_str(), "GetOpAttr stridesList failed!");
        return GRAPH_FAILED;
    }

    if (stridesList.size() != DIM_SIZE4) {
        string excepted_value = Strcat("equal to the length of x'shape[", DIM_SIZE4, "]");
        OpsAttrValueErrReport(op.GetName(), "strides'length", excepted_value, Strcat(stridesList.size()));
        OP_LOGE(op.GetName().c_str(), "length of strides must be equal to the "
            "length of shape!");
        return GRAPH_FAILED;
    }

    // get input data_format
    std::string dataFormat;
    if (GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
        OpsGetAttrErrReport(op.GetName(), "data_format");
        OP_LOGE(op.GetName().c_str(), "The MaxPool op GetOpAttr data_format "
            "failed!");
        return GRAPH_FAILED;
    }

    // get input paddingMode
    std::string paddingMode;
    if (GRAPH_SUCCESS != op.GetAttr("padding", paddingMode)) {
        OpsGetAttrErrReport(op.GetName(), "padding");
        OP_LOGE(op.GetName().c_str(), "GetOpAttr padding failed!");
        return GRAPH_FAILED;
    }

    if (paddingMode != "SAME" && paddingMode != "VALID") {
        string excepted_value = Strcat("SAME, VALID");
        OpsAttrValueErrReport(op.GetName(), "padding", excepted_value, paddingMode);
        OP_LOGE(op.GetName().c_str(), "MaxPool can only support SAME or VALID "
            "padding mode!");
        return GRAPH_FAILED;
    }

    std::vector<int64_t> dims_input = shape.GetDims();
    // set output shape
    std::vector<int64_t> dimVector;
    if (input_format == FORMAT_NHWC) {
        if (paddingMode == "SAME") {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE1) || (i == DIM_SIZE2)) {
                    if (dataFormat == "NHWC") {
                        int64_t dims = (dims_input[i] + stridesList[i] - 1) / stridesList[i];
                        dimVector.push_back(dims);
                    } else {
                        int64_t dims = (dims_input[i] + stridesList[i + 1] - 1) / stridesList[i + 1];
                        dimVector.push_back(dims);
                    }
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        } else {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE1) || (i == DIM_SIZE2)) {
                    if (dataFormat == "NHWC") {
                        int64_t dims = (dims_input[i] - ksizeList[i] + 1 + (stridesList[i] - 1)) / stridesList[i];
                        dimVector.push_back(dims);
                    } else {
                        int64_t dims =
                            (dims_input[i] - ksizeList[i + 1] + 1 + (stridesList[i + 1] - 1)) / stridesList[i + 1];
                        dimVector.push_back(dims);
                    }
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        }
    } else {
        if (paddingMode == "SAME") {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE2) || (i == DIM_SIZE3)) {
                    if (dataFormat == "NHWC") {
                        int64_t dims = (dims_input[i] + stridesList[i - 1] - 1) / stridesList[i - 1];
                        dimVector.push_back(dims);
                    } else {
                        int64_t dims = (dims_input[i] + stridesList[i] - 1) / stridesList[i];
                        dimVector.push_back(dims);
                    }
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        } else {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE2) || (i == DIM_SIZE3)) {
                    if (dataFormat == "NHWC") {
                        int64_t dims =
                            (dims_input[i] - ksizeList[i - 1] + 1 + (stridesList[i - 1] - 1)) / stridesList[i - 1];
                        dimVector.push_back(dims);
                    } else {
                        int64_t dims = (dims_input[i] - ksizeList[i] + 1 + (stridesList[i] - 1)) / stridesList[i];
                        dimVector.push_back(dims);
                    }
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        }
    }
    TensorDesc td = op.GetOutputDesc("y");
    DataType inputDtype = inputTensorDesc.GetDataType();
    Shape outputShape(dimVector);
    td.SetShape(outputShape);
    td.SetDataType(inputDtype);
    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MaxPoolExt2, MaxPoolExt2InferShape);
// ---------------------MaxPoolExt2---------------------

// ---------------------MaxPoolGradWithArgmax---------------------
IMPLEMT_VERIFIER(MaxPoolGradWithArgmax, MaxPoolGradWithArgmaxVerify)
{
    const size_t DIM_SIZE4 = 4;
    // get input ksize
    std::vector<int32_t> ksizeList;
    if (GRAPH_SUCCESS != op.GetAttr("ksize", ksizeList)) {
        OpsGetAttrErrReport(op.GetName(), "ksize");
        OP_LOGE(op.GetName().c_str(), "GetOpAttr ksizeList failed!");
        return GRAPH_FAILED;
    }
    if (ksizeList.size() < DIM_SIZE4) {
        string excepted_value = Strcat("more than[", DIM_SIZE4, "]");
        OpsAttrValueErrReport(op.GetName(), "ksize'length", excepted_value, Strcat(ksizeList.size()));
        OP_LOGE(op.GetName().c_str(), "length of ksize must be more than 4");
        return GRAPH_FAILED;
    }
    // get input strides
    std::vector<int32_t> stridesList;
    if (GRAPH_SUCCESS != op.GetAttr("strides", stridesList)) {
        OpsGetAttrErrReport(op.GetName(), "strides");
        OP_LOGE(op.GetName().c_str(), "GetOpAttr stridesList failed!");
        return GRAPH_FAILED;
    }

    if (stridesList.size() < DIM_SIZE4) {
        string excepted_value = Strcat("more than[", DIM_SIZE4, "]");
        OpsAttrValueErrReport(op.GetName(), "strides'length", excepted_value, Strcat(stridesList.size()));
        OP_LOGE(op.GetName().c_str(), "length of strides must be more than 4");
        return GRAPH_FAILED;
    }
    if ((ksizeList[0] != 1) || (ksizeList[3] != 1) || (stridesList[0] != 1) || (stridesList[3] != 1)) {
        OP_LOGE(op.GetName().c_str(), "MaxPoolGradWithArgmax only supports pooling "
            "across width/height, and other ksize "
            "dimension should be one");
        return GRAPH_FAILED;
    }
    if ((ksizeList[1] * ksizeList[2]) > 255) {
        OpsAttrValueErrReport(op.GetName(), "window", "<= 255", Strcat((ksizeList[1] * ksizeList[2])));
        OP_LOGE(op.GetName().c_str(), "invalid window params, window_h*window_w "
            "should be <= 255");
        return GRAPH_FAILED;
    }
    // get input paddingMode
    std::string paddingMode;
    if (GRAPH_SUCCESS != op.GetAttr("padding", paddingMode)) {
        OpsGetAttrErrReport(op.GetName(), "padding");
        OP_LOGE(op.GetName().c_str(), "GetOpAttr padding failed!");
        return GRAPH_FAILED;
    }
    if (paddingMode != "SAME" && paddingMode != "VALID") {
        string excepted_value = Strcat("SAME, VALID");
        OpsAttrValueErrReport(op.GetName(), "padding", excepted_value, paddingMode);
        OP_LOGE(op.GetName().c_str(), "MaxPool can only support SAME or VALID "
            "padding mode!");
        return GRAPH_FAILED;
    }
    if (!CheckTwoInputDtypeSame(op, "x", "grad")) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(MaxPoolGradWithArgmax, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
VERIFY_FUNC_REG(MaxPoolGradWithArgmax, MaxPoolGradWithArgmaxVerify);
// ---------------------MaxPoolGradWithArgmax---------------------

// ---------------------MaxPoolV2---------------------
static void GetMaxPoolV2ConstData(const Tensor &data, const DataType &dtype, std::vector<int64_t> &const_vec)
{
    const uint8_t *constData = data.GetData();
    size_t size;
    if (dtype == ge::DT_INT32) {
        size = data.GetSize() / sizeof(int32_t);
        for (size_t i = 0; i < size; ++i) {
            const_vec.push_back(*((int32_t *)constData + i));
        }
    } else {
        size = data.GetSize() / sizeof(int64_t);
        for (size_t i = 0; i < size; ++i) {
            const_vec.push_back(*((int64_t *)constData + i));
        }
    }
}

IMPLEMT_INFERFUNC(MaxPoolV2, MaxPoolV2InferShape)
{
    const size_t DIM_SIZE2 = 2;
    const size_t DIM_SIZE3 = 3;
    const size_t DIM_SIZE4 = 4;
    auto inputTensorDesc = op.GetInputDesc("x");
    auto shape = inputTensorDesc.GetShape();
    Format input_format = inputTensorDesc.GetFormat();
    // get input ksize
    Tensor ksizeData;
    if (ge::GRAPH_SUCCESS != op.GetInputConstData("ksize", ksizeData)) {
        OP_LOGE(op.GetName().c_str(), "get const data failed");
        return GRAPH_FAILED;
    }
    DataType ksizeDtype = op.GetInputDesc("ksize").GetDataType();
    std::vector<int64_t> ksizeList;
    GetMaxPoolV2ConstData(ksizeData, ksizeDtype, ksizeList);
    if (ksizeList.size() != DIM_SIZE4) {
        OP_LOGE(op.GetName().c_str(),
            "length of ksize %zu must be equal to the "
            "length of shape!",
            ksizeList.size());
        return GRAPH_FAILED;
    }

    // get input strides
    Tensor stridesData;
    if (ge::GRAPH_SUCCESS != op.GetInputConstData("strides", stridesData)) {
        OP_LOGE(op.GetName().c_str(), "get constdata failed");
        return GRAPH_FAILED;
    }
    DataType stridesDtype = op.GetInputDesc("strides").GetDataType();
    std::vector<int64_t> stridesList;
    GetMaxPoolV2ConstData(stridesData, stridesDtype, stridesList);
    if (stridesList.size() != DIM_SIZE4) {
        OP_LOGE(op.GetName().c_str(), "length of strides must be equal to the "
            "length of shape!");
        return GRAPH_FAILED;
    }
    // get input data_format
    std::string dataFormat;
    if (GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
        OP_LOGE(op.GetName().c_str(), "The MaxPool op GetOpAttr data_format "
            "failed!");
        return GRAPH_FAILED;
    }
    if (dataFormat != "NHWC" && dataFormat != "NCHW" && dataFormat != "NC1HWC0") {
        OP_LOGE(op.GetName().c_str(), "data_format only "
            "support 'NHWC','NCHW' and 'NC1HWC0'.");
        return GRAPH_FAILED;
    }
    if (dataFormat == "NHWC") {
        if ((ksizeList[0] != 1) || (ksizeList[3] != 1) || (stridesList[0] != 1) || (stridesList[3] != 1)) {
            OP_LOGE(op.GetName().c_str(), "MaxPool only supports pooling across width/height"
                "and other ksize dimension should be one");
            return GRAPH_FAILED;
        }
    }
    if ((dataFormat == "NCHW") || (dataFormat == "NC1HWC0")) {
        if ((ksizeList[0] != 1) || (ksizeList[1] != 1) || (stridesList[0] != 1) || (stridesList[1] != 1)) {
            OP_LOGE(op.GetName().c_str(), "MaxPool only supports pooling across width/height"
                "and other ksize dimension should be one");
            return GRAPH_FAILED;
        }
    }


    // get input paddingMode
    std::string paddingMode;
    if (GRAPH_SUCCESS != op.GetAttr("padding", paddingMode)) {
        OP_LOGE(op.GetName().c_str(), "GetOpAttr padding failed!");
        return GRAPH_FAILED;
    }

    if (paddingMode != "SAME" && paddingMode != "VALID") {
        OP_LOGE(op.GetName().c_str(), "MaxPool can only support SAME or VALID "
            "padding mode!");
        return GRAPH_FAILED;
    }

    std::vector<int64_t> dims_input = shape.GetDims();
    // set output shape
    std::vector<int64_t> dimVector;
    if (input_format == FORMAT_NHWC) {
        if (paddingMode == "SAME") {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE1) || (i == DIM_SIZE2)) {
                    if (dataFormat == "NHWC") {
                        int64_t dims = (dims_input[i] + stridesList[i] - 1) / stridesList[i];
                        dimVector.push_back(dims);
                    } else {
                        int64_t dims = (dims_input[i] + stridesList[i + 1] - 1) / stridesList[i + 1];
                        dimVector.push_back(dims);
                    }
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        } else {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE1) || (i == DIM_SIZE2)) {
                    if (dataFormat == "NHWC") {
                        int64_t dims = (dims_input[i] - ksizeList[i] + 1 + (stridesList[i] - 1)) / stridesList[i];
                        dimVector.push_back(dims);
                    } else {
                        int64_t dims =
                            (dims_input[i] - ksizeList[i + 1] + 1 + (stridesList[i + 1] - 1)) / stridesList[i + 1];
                        dimVector.push_back(dims);
                    }
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        }
    } else {
        if (paddingMode == "SAME") {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE2) || (i == DIM_SIZE3)) {
                    if (dataFormat == "NHWC") {
                        int64_t dims = (dims_input[i] + stridesList[i - 1] - 1) / stridesList[i - 1];
                        dimVector.push_back(dims);
                    } else {
                        int64_t dims = (dims_input[i] + stridesList[i] - 1) / stridesList[i];
                        dimVector.push_back(dims);
                    }
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        } else {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE2) || (i == DIM_SIZE3)) {
                    if (dataFormat == "NHWC") {
                        int64_t dims =
                            (dims_input[i] - ksizeList[i - 1] + 1 + (stridesList[i - 1] - 1)) / stridesList[i - 1];
                        dimVector.push_back(dims);
                    } else {
                        int64_t dims = (dims_input[i] - ksizeList[i] + 1 + (stridesList[i] - 1)) / stridesList[i];
                        dimVector.push_back(dims);
                    }
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        }
    }
    TensorDesc td = op.GetOutputDesc("y");
    DataType inputDtype = inputTensorDesc.GetDataType();
    Shape outputShape(dimVector);
    td.SetShape(outputShape);
    td.SetDataType(inputDtype);
    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MaxPoolV2, MaxPoolV2InferShape);
// ---------------------MaxPoolV2---------------------

int64_t CeilDev(int64_t value, int64_t factor)
{
    int64_t value_num = 0;
    if (value % factor == 0) {
        value_num = value / factor;
    } else {
        value_num = value / factor + 1;
    }
    return value_num;
}


// ---------------------MaxPoolWithArgmax---------------------
IMPLEMT_INFERFUNC(MaxPoolWithArgmax, MaxPoolWithArgmaxInferShape)
{
    const size_t DIM_SIZE1 = 1;
    const size_t DIM_SIZE2 = 2;
    const size_t DIM_SIZE3 = 3;
    const size_t DIM_SIZE4 = 4;
    ge::TensorDesc inputTensorDesc = op.GetInputDesc(0);
    Format input_format = inputTensorDesc.GetFormat();
    ge::Shape shape = inputTensorDesc.GetShape();
    int32_t in_size_h = 0;
    int32_t in_size_w = 0;
    if (input_format == FORMAT_NHWC) {
        in_size_h = shape.GetDim(1);
        in_size_w = shape.GetDim(2);
    } else {
        in_size_h = shape.GetDim(2);
        in_size_w = shape.GetDim(3);
    }
    // get input ksize
    std::vector<int32_t> ksizeList;
    if (op.GetAttr("ksize", ksizeList) != ge::GRAPH_SUCCESS) {
        OpsGetAttrErrReport(op.GetName(), "ksize");
        OP_LOGE(op.GetName().c_str(), "GetOpAttr ksizeList failed!");
        return GRAPH_FAILED;
    }

    if (ksizeList.size() != DIM_SIZE4) {
        string excepted_value = Strcat("equal to the length of x'shape[", DIM_SIZE4, "]");
        OpsAttrValueErrReport(op.GetName(), "ksize'length", excepted_value, Strcat(ksizeList.size()));
        OP_LOGE(op.GetName().c_str(), "length of ksize must be equal to"
            "the length of shape!");
        return GRAPH_FAILED;
    }

    // get input strides
    std::vector<int32_t> stridesList;
    if (op.GetAttr("strides", stridesList) != ge::GRAPH_SUCCESS) {
        OpsGetAttrErrReport(op.GetName(), "strides");
        OP_LOGE(op.GetName().c_str(), "GetOpAttr stridesList failed!");
        return GRAPH_FAILED;
    }

    if (stridesList.size() != DIM_SIZE4) {
        string excepted_value = Strcat("equal to the length of x'shape[", DIM_SIZE4, "]");
        OpsAttrValueErrReport(op.GetName(), "strides'length", excepted_value, Strcat(stridesList.size()));
        OP_LOGE(op.GetName().c_str(), "length of strides must be equal to"
            "the length of shape!");
        return GRAPH_FAILED;
    }
    if ((ksizeList[0] != 1) || (ksizeList[3] != 1) || (stridesList[0] != 1) || (stridesList[3] != 1)) {
        OP_LOGE(op.GetName().c_str(), "MaxPoolWithArgmax only supports pooling "
            "across width/height, and other ksize "
            "dimension should be one");
        return GRAPH_FAILED;
    }
    if ((ksizeList[1] * ksizeList[2]) > 255) {
        OpsAttrValueErrReport(op.GetName(), "window", "<= 255", Strcat((ksizeList[1] * ksizeList[2])));
        OP_LOGE(op.GetName().c_str(), "invalid window params, window_h*window_w "
            "should be <= 255");
        return GRAPH_FAILED;
    }
    if ((ksizeList[1] > in_size_h) || (ksizeList[2] > in_size_w)) {
        OP_LOGE(op.GetName().c_str(), "can not support global pooling now");
        return GRAPH_FAILED;
    }

    // get input paddingMode
    std::string paddingMode;
    if (op.GetAttr("padding", paddingMode) != ge::GRAPH_SUCCESS) {
        OpsGetAttrErrReport(op.GetName(), "padding");
        OP_LOGE(op.GetName().c_str(), "GetOpAttr padding failed!");
        return GRAPH_FAILED;
    }

    if (paddingMode != "SAME" && paddingMode != "VALID") {
        string excepted_value = Strcat("SAME, VALID");
        OpsAttrValueErrReport(op.GetName(), "padding", excepted_value, paddingMode);
        OP_LOGE(op.GetName().c_str(), "MaxPoolWithArgmax can only support"
            "SAME or VALID padding mode!");
        return GRAPH_FAILED;
    }
    std::vector<int64_t> dims_input = shape.GetDims();
    // set output max shape
    std::vector<int64_t> dimVector;
    if (input_format == FORMAT_NHWC) {
        if (paddingMode == "SAME") {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE1) || (i == DIM_SIZE2)) {
                    int64_t dims = (dims_input[i] + stridesList[i] - 1) / stridesList[i];
                    dimVector.push_back(dims);
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        } else {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE1) || (i == DIM_SIZE2)) {
                    int64_t dims = (dims_input[i] - ksizeList[i] + 1 + (stridesList[i] - 1)) / stridesList[i];
                    dimVector.push_back(dims);
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        }
    } else {
        if (paddingMode == "SAME") {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE2) || (i == DIM_SIZE3)) {
                    int64_t dims = (dims_input[i] + stridesList[i - 1] - 1) / stridesList[i - 1];
                    dimVector.push_back(dims);
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        } else {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE2) || (i == DIM_SIZE3)) {
                    int64_t dims =
                        (dims_input[i] - ksizeList[i - 1] + 1 + (stridesList[i - 1] - 1)) / stridesList[i - 1];
                    dimVector.push_back(dims);
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        }
    }
    TensorDesc outputMaxTensorDesc = op.GetOutputDesc("y");
    TensorDesc outputMaskTensorDesc = op.GetOutputDesc("argmax");
    Shape outputMaxShape(dimVector);
    DataType inputDtype = inputTensorDesc.GetDataType();
    outputMaxTensorDesc.SetShape(outputMaxShape);
    outputMaxTensorDesc.SetDataType(inputDtype);
    outputMaskTensorDesc.SetShape(outputMaxShape);
    outputMaskTensorDesc.SetDataType(DT_INT64);
    (void)op.UpdateOutputDesc("y", outputMaxTensorDesc);
    (void)op.UpdateOutputDesc("argmax", outputMaskTensorDesc);

    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MaxPoolWithArgmax, MaxPoolWithArgmaxInferShape);
// ---------------------MaxPoolWithArgmax---------------------

// ---------------------Mask2Argmax---------------------
IMPLEMT_INFERFUNC(Mask2Argmax, Mask2ArgmaxInferShape)
{
    const size_t DIM_SIZE1 = 1;
    const size_t DIM_SIZE2 = 2;
    const size_t DIM_SIZE3 = 3;
    const size_t DIM_SIZE4 = 4;
    ge::TensorDesc inputTensorDesc = op.GetInputDesc(0);
    Format input_format = inputTensorDesc.GetFormat();
    ge::Shape shape = inputTensorDesc.GetShape();
    int32_t in_size_h = 0;
    int32_t in_size_w = 0;
    if (input_format == FORMAT_NHWC) {
        in_size_h = shape.GetDim(1);
        in_size_w = shape.GetDim(2);
    } else {
        in_size_h = shape.GetDim(2);
        in_size_w = shape.GetDim(3);
    }
    // get input ksize
    std::vector<int32_t> ksizeList;
    if (op.GetAttr("ksize", ksizeList) != ge::GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "GetOpAttr ksizeList failed!");
        return GRAPH_FAILED;
    }

    if (ksizeList.size() != DIM_SIZE4) {
        OP_LOGE(op.GetName().c_str(), "length of ksize must be equal to"
            "the length of shape!");
        return GRAPH_FAILED;
    }

    // get input strides
    std::vector<int32_t> stridesList;
    if (op.GetAttr("strides", stridesList) != ge::GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "GetOpAttr stridesList failed!");
        return GRAPH_FAILED;
    }

    if (stridesList.size() != DIM_SIZE4) {
        OP_LOGE(op.GetName().c_str(), "length of strides must be equal to"
            "the length of shape!");
        return GRAPH_FAILED;
    }
    if ((ksizeList[0] != 1) || (ksizeList[3] != 1) || (stridesList[0] != 1) || (stridesList[3] != 1)) {
        OP_LOGE(op.GetName().c_str(), "Mask2Argmax only supports pooling "
            "across width/height, and other ksize "
            "dimension should be one");
        return GRAPH_FAILED;
    }
    if ((ksizeList[1] * ksizeList[2]) > 255) {
        OP_LOGE(op.GetName().c_str(), "invalid window params, window_h*window_w "
            "should be <= 255");
        return GRAPH_FAILED;
    }
    if ((ksizeList[1] >= in_size_h) || (ksizeList[2] >= in_size_w)) {
        OP_LOGE(op.GetName().c_str(), "can not support global pooling now");
        return GRAPH_FAILED;
    }

    // get input paddingMode
    std::string paddingMode;
    if (op.GetAttr("padding", paddingMode) != ge::GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "GetOpAttr padding failed!");
        return GRAPH_FAILED;
    }

    if (paddingMode != "SAME" && paddingMode != "VALID") {
        OP_LOGE(op.GetName().c_str(), "Mask2Argmax can only support"
            "SAME or VALID padding mode!");
        return GRAPH_FAILED;
    }
    std::vector<int64_t> dims_input = shape.GetDims();
    // set output max shape
    std::vector<int64_t> dimVector;
    if (input_format == FORMAT_NHWC) {
        if (paddingMode == "SAME") {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE1) || (i == DIM_SIZE2)) {
                    int64_t dims = (dims_input[i] + stridesList[i] - 1) / stridesList[i];
                    dimVector.push_back(dims);
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        } else {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE1) || (i == DIM_SIZE2)) {
                    int64_t dims = (dims_input[i] - ksizeList[i] + 1 + (stridesList[i] - 1)) / stridesList[i];
                    dimVector.push_back(dims);
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        }
    } else {
        if (paddingMode == "SAME") {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE2) || (i == DIM_SIZE3)) {
                    int64_t dims = (dims_input[i] + stridesList[i - 1] - 1) / stridesList[i - 1];
                    dimVector.push_back(dims);
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        } else {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE2) || (i == DIM_SIZE3)) {
                    int64_t dims =
                        (dims_input[i] - ksizeList[i - 1] + 1 + (stridesList[i - 1] - 1)) / stridesList[i - 1];
                    dimVector.push_back(dims);
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        }
    }

    TensorDesc outputArgmaxTensorDesc = op.GetOutputDesc("argmax");
    Shape outputArgmaxShape(dimVector);
    DataType outputArgmaxDtype = DT_FLOAT;
    outputArgmaxTensorDesc.SetShape(outputArgmaxShape);
    outputArgmaxTensorDesc.SetDataType(outputArgmaxDtype);

    (void)op.UpdateOutputDesc("argmax", outputArgmaxTensorDesc);

    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Mask2Argmax, Mask2ArgmaxInferShape);
// ---------------------Mask2Argmax---------------------

// ----------------------MaxPoolGradGradWithArgmax-----------------------
IMPLEMT_VERIFIER(MaxPoolGradGradWithArgmax, MaxPoolGradGradWithArgmaxVerify)
{
    DataType input_x_type = op.GetInputDesc("x").GetDataType();
    DataType input_grad_type = op.GetInputDesc("grad").GetDataType();
    if (input_x_type != input_grad_type) {
        OP_LOGE(op.GetName().c_str(), "the max_pool_grad_grad_with_argmax op inputs"
            "should have the same dtype!\n");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(MaxPoolGradGradWithArgmax, MaxPoolGradGradWithArgmaxInferShape)
{
    const size_t DIM_SIZE1 = 1;
    const size_t DIM_SIZE2 = 2;
    const size_t DIM_SIZE3 = 3;
    const size_t DIM_SIZE4 = 4;
    auto inputTensorDesc = op.GetInputDesc("x");
    auto shape = inputTensorDesc.GetShape();
    Format input_format = inputTensorDesc.GetFormat();
    // get input ksize
    std::vector<int32_t> ksizeList;
    if (op.GetAttr("ksize", ksizeList) != GRAPH_SUCCESS) {
        OpsGetAttrErrReport(op.GetName(), "ksize");
        OP_LOGE(op.GetName().c_str(), "GetOpAttr ksizeList failed!\n");
        return GRAPH_FAILED;
    }

    if (ksizeList.size() != DIM_SIZE4) {
        OpsAttrValueErrReport(op.GetName(), "ksize", Strcat(DIM_SIZE4), Strcat(ksizeList.size()));
        OP_LOGE(op.GetName().c_str(), "length of ksize must be equal to the "
            "length of shape!\n");
        return GRAPH_FAILED;
    }

    // get input strides
    std::vector<int32_t> stridesList;
    if (op.GetAttr("strides", stridesList) != GRAPH_SUCCESS) {
        OpsGetAttrErrReport(op.GetName(), "strides");
        OP_LOGE(op.GetName().c_str(), "GetOpAttr stridesList failed!\n");
        return GRAPH_FAILED;
    }

    if (stridesList.size() != DIM_SIZE4) {
        OpsAttrValueErrReport(op.GetName(), "strides", Strcat(DIM_SIZE4), Strcat(stridesList.size()));
        OP_LOGE(op.GetName().c_str(), "length of strides must be equal to "
            "the length of shape!\n");
        return GRAPH_FAILED;
    }

    // get input paddingMode
    std::string paddingMode;
    if (op.GetAttr("padding", paddingMode) != GRAPH_SUCCESS) {
        OpsGetAttrErrReport(op.GetName(), "padding");
        OP_LOGE(op.GetName().c_str(), "GetOpAttr padding failed!\n");
        return GRAPH_FAILED;
    }

    if (paddingMode != "SAME" && paddingMode != "VALID") {
        string excepted_value = Strcat("SAME, VALID");
        OpsAttrValueErrReport(op.GetName(), "paddingMode", excepted_value, paddingMode);
        OP_LOGE(op.GetName().c_str(), "MaxPoolGradGradWithArgmax can only support"
            " SAME or VALID  padding mode!\n");
        return GRAPH_FAILED;
    }

    std::vector<int64_t> dims_input = shape.GetDims();
    // set output shape
    std::vector<int64_t> dimVector;
    if (input_format == FORMAT_NHWC) {
        if (paddingMode == "SAME") {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE1) || (i == DIM_SIZE2)) {
                    int64_t dims = (dims_input[i] + stridesList[i] - 1) / stridesList[i];
                    dimVector.push_back(dims);
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        } else {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE1) || (i == DIM_SIZE2)) {
                    int64_t dims = (dims_input[i] - ksizeList[i] + 1 + (stridesList[i] - 1)) / stridesList[i];
                    dimVector.push_back(dims);
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        }
    } else {
        if (paddingMode == "SAME") {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE2) || (i == DIM_SIZE3)) {
                    int64_t dims = (dims_input[i] + stridesList[i - 1] - 1) / stridesList[i - 1];
                    dimVector.push_back(dims);
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        } else {
            for (size_t i = 0; i < dims_input.size(); i++) {
                if ((i == DIM_SIZE2) || (i == DIM_SIZE3)) {
                    int64_t dims =
                        (dims_input[i] - ksizeList[i - 1] + 1 + (stridesList[i - 1] - 1)) / stridesList[i - 1];
                    dimVector.push_back(dims);
                } else {
                    int64_t dims = dims_input[i];
                    dimVector.push_back(dims);
                }
            }
        }
    }
    TensorDesc td = op.GetOutputDesc("y");
    DataType inputDtype = inputTensorDesc.GetDataType();
    Shape outputShape(dimVector);
    td.SetShape(outputShape);
    td.SetDataType(inputDtype);
    (void)op.UpdateOutputDesc("y", td);

    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MaxPoolGradGradWithArgmax, MaxPoolGradGradWithArgmaxInferShape);
VERIFY_FUNC_REG(MaxPoolGradGradWithArgmax, MaxPoolGradGradWithArgmaxVerify);
// ----------------------MaxPoolGradGradWithArgmax-----------------------

// ---------------------AvgPoolGrad---------------------

IMPLEMT_VERIFIER(AvgPoolGrad, AvgPoolGradVerify)
{
    Tensor orig_input_shape_tensor;
    if (GRAPH_SUCCESS != op.GetInputConstData("orig_input_shape", orig_input_shape_tensor)) {
        OP_LOGE(op.GetName().c_str(), "get constdata filed");
        return GRAPH_FAILED;
    }

    DataType dtype = op.GetInputDesc("orig_input_shape").GetDataType();

    std::vector<int64_t> orig_input_size;
    GetConstValue(op, orig_input_shape_tensor, dtype, orig_input_size);
    if (!CheckListEmpty(op.GetName(), orig_input_size, "orig_input_shape")) {
        return GRAPH_FAILED;
    }
    std::vector<int64_t> ksize;
    ksize = GetAttrValue(op, "ksize");
    if (!CheckListEmpty(op.GetName(), ksize, "ksize")) {
        return GRAPH_FAILED;
    }
    if (ksize.size() < 4) {
        OP_LOGE(op.GetName().c_str(), "attr ksize(%d) is too small", (int)ksize.size());
    }
    std::vector<int64_t> strides;
    strides = GetAttrValue(op, "strides");
    if (!CheckListEmpty(op.GetName(), strides, "strides")) {
        return GRAPH_FAILED;
    }
    if (strides.size() < 4) {
        OP_LOGE(op.GetName().c_str(), "attr strides(%d) is too small", (int)strides.size());
    }

    std::string padding;
    if (ge::GRAPH_SUCCESS != op.GetAttr("padding", padding)) {
        OP_LOGE(op.GetName().c_str(), "Get padding failed!");
        return GRAPH_FAILED;
    }
    if (padding != "SAME" && padding != "VALID") {
        OP_LOGE(op.GetName().c_str(), "attr padding(%s) only support SAME and VALID", padding.c_str());
        return GRAPH_FAILED;
    }
    std::string data_format;
    if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
        if (data_format != "NCHW" && data_format != "NHWC") {
            OP_LOGE(op.GetName().c_str(), "attr data_format(%s) only support NCHW and NHWC", data_format.c_str());
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AvgPoolGradInferShape)
{
    Tensor orig_input_shape_tensor;
    if (GRAPH_SUCCESS != op.GetInputConstData("orig_input_shape", orig_input_shape_tensor)) {
        OP_LOGE(op.GetName().c_str(), "Get constdata failed");
        return GRAPH_FAILED;
    }
    DataType dtype = op.GetInputDesc("orig_input_shape").GetDataType();
    std::vector<int64_t> orig_input_size;
    GetConstValue(op, orig_input_shape_tensor, dtype, orig_input_size);
    DataType output_dtype = op.GetInputDesc("input_grad").GetDataType();
    TensorDesc tensordesc_output = op.GetOutputDesc("out_grad");
    tensordesc_output.SetShape(Shape(orig_input_size));
    tensordesc_output.SetDataType(output_dtype);
    (void)op.UpdateOutputDesc("out_grad", tensordesc_output);
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AvgPoolGrad, AvgPoolGradInferShape);
VERIFY_FUNC_REG(AvgPoolGrad, AvgPoolGradVerify);
// ---------------------AvgPoolGrad---------------------


// ---------------------AvgPoolGradD---------------------
IMPLEMT_VERIFIER(AvgPoolGradD, AvgPoolGradDVerify)
{
    std::vector<int64_t> orig_input_size;
    orig_input_size = GetAttrValue(op, "orig_input_shape");
    if (!CheckListEmpty(op.GetName(), orig_input_size, "orig_input_shape")) {
        return GRAPH_FAILED;
    }
    std::vector<int64_t> ksize;
    ksize = GetAttrValue(op, "ksize");
    if (!CheckListEmpty(op.GetName(), ksize, "ksize")) {
        return GRAPH_FAILED;
    }
    if (ksize.size() < 4) {
        OP_LOGE(op.GetName().c_str(), "attr ksize(%d) is too small", (int)ksize.size());
    }
    std::vector<int64_t> strides;
    strides = GetAttrValue(op, "strides");
    if (!CheckListEmpty(op.GetName(), strides, "strides")) {
        return GRAPH_FAILED;
    }
    if (!CheckListEmpty(op.GetName(), strides, "strides")) {
        return GRAPH_FAILED;
    }
    std::string padding;
    if (ge::GRAPH_SUCCESS != op.GetAttr("padding", padding)) {
        OP_LOGE(op.GetName().c_str(), "Get padding failed!");
        return GRAPH_FAILED;
    }
    if (padding != "SAME" && padding != "VALID") {
        OP_LOGE(op.GetName().c_str(), "attr padding(%s) must in SAME and VALID", padding.c_str());
        return GRAPH_FAILED;
    }
    std::string data_format;
    if (op.GetAttr("data_format", data_format) == ge::GRAPH_SUCCESS) {
        if (data_format != "NCHW" && data_format != "NHWC") {
            OP_LOGE(op.GetName().c_str(), "attr data_format(%s) only support NCHW and NHWC", data_format.c_str());
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AvgPoolGradDInferShape)
{
    std::vector<int64_t> orig_input_size;
    orig_input_size = GetAttrValue(op, "orig_input_shape");
    DataType output_dtype = op.GetInputDesc("input_grad").GetDataType();
    TensorDesc tensordesc_output = op.GetOutputDesc("out_grad");
    tensordesc_output.SetShape(Shape(orig_input_size));
    tensordesc_output.SetDataType(output_dtype);
    (void)op.UpdateOutputDesc("out_grad", tensordesc_output);
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AvgPoolGradD, AvgPoolGradDInferShape);
VERIFY_FUNC_REG(AvgPoolGradD, AvgPoolGradDVerify);
// ---------------------AvgPoolGradD---------------------

IMPLEMT_VERIFIER(Upsample, UpsampleVerify)
{
    return GRAPH_SUCCESS;
}
IMPLEMT_INFERFUNC(Upsample, UpsampleInferShape)
{
    TensorDesc tensordesc_output = op.GetInputDesc("x");
    uint32_t stride_h = 2;
    uint32_t stride_w = 2;
    if (op.GetAttr("stride_h", stride_h) != ge::GRAPH_SUCCESS) {
        OP_LOGI(op.GetName().c_str(), "GetOpAttr stride failed, set stride_h default value");
        stride_h = 2;
    }
    if (op.GetAttr("stride_w", stride_w) != ge::GRAPH_SUCCESS) {
        OP_LOGI(op.GetName().c_str(), "GetOpAttr stride failed, set stride_w default value");
        stride_w = 2;
    }
    ge::Shape shape = tensordesc_output.GetShape();
    std::vector<int64_t> dims_input = shape.GetDims();
    std::vector<int64_t> dimVector;
    for (size_t i = 0; i < dims_input.size(); i++) {
        if (i == 2) {
            int64_t dims = dims_input[i] * stride_h;
            dimVector.push_back(dims);
        } else if (i == 3) {
            int64_t dims = dims_input[i] * stride_w;
            dimVector.push_back(dims);
        } else {
            int64_t dims = dims_input[i];
            dimVector.push_back(dims);
        }
    }

    Shape outputMaxShape(dimVector);
    tensordesc_output.SetShape(outputMaxShape);
    (void)op.UpdateOutputDesc("y", tensordesc_output);

    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Upsample, UpsampleInferShape);
VERIFY_FUNC_REG(Upsample, UpsampleVerify);

IMPLEMT_INFERFUNC(FractionalMaxPoolGrad, FractionalMaxPoolGradInfer)
{
    Shape input_shape;
    if (WithRank(op.GetInputDesc(0), 4, input_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        ShapeErrReport(0, op.GetName(), DebugString(op.GetInputDesc(0).GetShape().GetDims()), "4D");
        return GRAPH_FAILED;
    }

    auto type = op.GetInputDesc("orig_input").GetDataType();
    TensorDesc output_desc = op.GetOutputDesc("y");
    output_desc.SetShape(Shape(input_shape));
    output_desc.SetDataType(type);
    if (op.UpdateOutputDesc("y", output_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Update y desc failed.");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(FractionalMaxPoolGrad, FractionalMaxPoolGradInfer);

IMPLEMT_INFERFUNC(FractionalAvgPool, FractionalAvgPoolInfer)
{
    Shape input;
    if (WithRank(op.GetInputDesc(0), 4, input, op.GetName().c_str()) != GRAPH_SUCCESS) {
        ShapeErrReport(0, op.GetName(), DebugString(op.GetInputDesc(0).GetShape().GetDims()), "4D");
        return GRAPH_FAILED;
    }
    std::vector<float> pooling_ratio;
    op.GetAttr("pooling_ratio", pooling_ratio);
    if (pooling_ratio.size() != 4) {
        AttrSizeErrReport("pooling_ratio", op.GetName(), Strcat(pooling_ratio.size()), "4");
        OP_LOGE(op.GetName().c_str(), "pooling_ratio field must specify 4 dimensions.");
        return GRAPH_PARAM_INVALID;
    }
    auto x_dims = op.GetInputDesc(0).GetShape().GetDims();
    std::vector<int64_t> dims;
    dims.reserve(4);
    for (int i = 0; i < 4; ++i) {
        auto val = static_cast<int64_t>(x_dims[i] / pooling_ratio[i]);
        if (val < 0) {
            string err_msg = Strcat("size computed for ", i, "th dim is ", val, ", please check");
            OP_LOGE(op.GetName().c_str(), "%s.", err_msg.c_str());
            InferShapeOtherErrReport(op.GetName(), err_msg);
            return GRAPH_PARAM_INVALID;
        }
        dims.push_back(val);
    }
    Shape out(dims);
    Shape row_pooling_sequence;
    (void)Vector(dims[1] + 1, row_pooling_sequence);
    Shape col_pooling_sequence;
    (void)Vector(dims[2] + 1, col_pooling_sequence);

    DataType type = op.GetInputDesc("x").GetDataType();

    TensorDesc y_desc = op.GetOutputDesc("y");
    y_desc.SetShape(out);
    y_desc.SetDataType(type);
    op.UpdateOutputDesc("y", y_desc);

    TensorDesc row_desc = op.GetOutputDesc("row_pooling_sequence");
    row_desc.SetShape(row_pooling_sequence);
    row_desc.SetDataType(DT_INT64);
    op.UpdateOutputDesc("row_pooling_sequence", row_desc);

    TensorDesc col_desc = op.GetOutputDesc("col_pooling_sequence");
    col_desc.SetShape(col_pooling_sequence);
    col_desc.SetDataType(DT_INT64);
    op.UpdateOutputDesc("col_pooling_sequence", col_desc);

    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(FractionalAvgPool, FractionalAvgPoolInfer);

IMPLEMT_INFERFUNC(FractionalMaxPool, FractionalMaxPoolInfer)
{
    auto tensor = op.get_input_desc_x();
    Shape input_value;

    if (WithRank(tensor, 4, input_value, op.GetName().c_str()) != GRAPH_SUCCESS) {
        ShapeErrReport(0, op.GetName(), DebugString(tensor.GetShape().GetDims()), "4D");
        OP_LOGE(op.GetName().c_str(), "input value must be 4-D.");
        return GRAPH_FAILED;
    }
    std::vector<float> pooling_ratio;
    pooling_ratio = op.get_attr_pooling_ratio();
    if (pooling_ratio.size() != 4) {
        AttrSizeErrReport("pooling_ratio", op.GetName(), Strcat(pooling_ratio.size()), "4");
        OP_LOGE(op.GetName().c_str(), "pooling_ratio field must specify 4-D.");
        return GRAPH_FAILED;
    }

    std::vector<int64_t> output_dims;
    for (int i = 0; i < 4; ++i) {
        int64_t dim = input_value.GetDim(i);
        if (dim != UNKNOWN_DIM) {
            auto real_dim = static_cast<int64_t>(dim / pooling_ratio[i]);
            if (real_dim < 0) {
                string err_msg = Strcat("size computed for ", i, "th dim is ", real_dim, ", please check");
                OP_LOGE(op.GetName().c_str(), "%s.", err_msg.c_str());
                InferShapeOtherErrReport(op.GetName(), err_msg);
                return GRAPH_FAILED;
            }
            output_dims.push_back(real_dim);
        } else {
            output_dims.push_back(UNKNOWN_DIM);
        }
    }

    TensorDesc y_desc = op.GetOutputDesc("y");
    y_desc.SetShape(Shape(output_dims));
    y_desc.SetDataType(op.GetInputDesc("x").GetDataType());
    if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "fail to update output y.");
        return GRAPH_FAILED;
    }

    TensorDesc row_pooling_desc = op.GetOutputDesc("row_pooling_sequence");
    row_pooling_desc.SetShape(Shape({ output_dims[1] + 1 }));
    row_pooling_desc.SetDataType(DT_INT64);
    if (op.UpdateOutputDesc("row_pooling_sequence", row_pooling_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "fail to  update output row_pooling_sequence.");
        return GRAPH_FAILED;
    }

    TensorDesc col_pooling_desc = op.GetOutputDesc("col_pooling_sequence");
    col_pooling_desc.SetShape(Shape({ output_dims[2] + 1 }));
    col_pooling_desc.SetDataType(DT_INT64);
    if (op.UpdateOutputDesc("col_pooling_sequence", col_pooling_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "fail to update output col_pooling_sequence.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(FractionalMaxPool, FractionalMaxPoolInfer);

IMPLEMT_INFERFUNC(DataFormatVecPermute, DataFormatVecPermuteInfer)
{
    return UnchangedShape(op, "x", "y");
}

INFER_FUNC_REG(DataFormatVecPermute, DataFormatVecPermuteInfer);

IMPLEMT_INFERFUNC(FractionalAvgPoolGrad, FractionalAvgPoolGradInfer)
{
    Tensor tensor;
    if (op.GetInputConstData("orig_input_tensor_shape", tensor) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input orig_input_tensor_shape GetInputConstData failed");
        return GRAPH_FAILED;
    }

    Shape result;
    if (MakeShapeFromShapeTensor(tensor, result, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Make shape from ShapeTensor failed");
        return GRAPH_FAILED;
    }

    DataType y_type = op.GetInputDesc("out_backprop").GetDataType();
    TensorDesc out_desc = op.GetOutputDesc("y");
    out_desc.SetShape(Shape(result));
    out_desc.SetDataType(y_type);
    if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update y desc failed.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(FractionalAvgPoolGrad, FractionalAvgPoolGradInfer);

IMPLEMT_INFERFUNC(NthElement, NthElementInfer)
{
    Shape x_shape;
    auto x_tensor = op.get_input_desc_x();
    if (WithRankAtLeast(x_tensor, 1, x_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        ShapeErrReport(0, op.GetName(), DebugString(op.GetInputDesc(0).GetShape().GetDims()), "at least 1D");
        OP_LOGE(op.GetName().c_str(), "The rank of input x must be at least 1.");
        return GRAPH_FAILED;
    }

    Tensor n_tensor;
    if (op.GetInputConstData("n", n_tensor) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Get input n value error.");
        return GRAPH_FAILED;
    }

    int64_t n_dim = 0;
    if (MakeDimForScalarInput(n_tensor, n_dim, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "MakeDimForScalarInput get input n value error.");
        return GRAPH_FAILED;
    }

    int64_t existing = x_shape.GetDimNum();
    int64_t last_input_dim = x_shape.GetDim(existing - 1);
    if ((last_input_dim != ge::UNKNOWN_DIM) && (n_dim != ge::UNKNOWN_DIM) && (last_input_dim <= n_dim)) {
        OP_LOGE(op.GetName().c_str(), "Input must have last dim > n=%lld,but inputLastDim is %lld", n_dim,
            last_input_dim);
        return GRAPH_FAILED;
    }

    Shape output_shape;
    if (SubShape(x_shape, 0, -1, 1, output_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Get SubShape error.");
        return GRAPH_FAILED;
    }

    TensorDesc y_tensor = op.GetOutputDesc("y");
    y_tensor.SetDataType(x_tensor.GetDataType());
    y_tensor.SetShape(output_shape);
    return op.UpdateOutputDesc("y", y_tensor);
}

INFER_FUNC_REG(NthElement, NthElementInfer);

// ----------------MaxPool3DGrad-------------------
static bool GetPadMaxPool3DGrad(ge::Operator &op, int32_t id, int32_t ih, int32_t iw, int32_t kd, int32_t kh,
    int32_t kw, int32_t strd, int32_t strh, int32_t strw, int32_t &padf, int32_t &padba, int32_t &padt, int32_t &padb,
    int32_t &padl, int32_t &padr)
{
    std::string padStr;
    std::vector<int32_t> padList;
    if (GRAPH_SUCCESS == op.GetAttr("padding", padStr)) {
        if (padStr.compare("SAME") == 0) {
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
            for (int32_t i = 0; i < 6; i++)
                padList.push_back(0);
        } else {
            map<string, string> err_map;
            err_map["param_name"] = "padding";
            err_map["op_name"] = "MaxPool3DGrad";
            err_map["Expected_value"] = "SAME or VALID";
            err_map["input_value"] = padStr;
            std::string report_error_code = "E50029";
            ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
            return false;
        }
        op.SetAttr("pads", padList);
    }
    std::vector<int32_t> padVec;
    if (op.GetAttr("pads", padVec) != ge::GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Failed to get pads!");
        return false;
    }
    auto pSize = padVec.size();
    if (pSize != 6) {
        map<string, string> err_map;
        err_map["param_name"] = "pads list";
        err_map["op_name"] = "MaxPool3DGrad";
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
    if (padf < 0 || padba < 0 || padt < 0 || padb < 0 || padl < 0 || padr < 0) {
        map<string, string> err_map;
        err_map["param_name"] = "pads_list";
        err_map["op_name"] = "MaxPool3DGrad";
        err_map["excepted_value"] = "positive";
        err_map["input_value"] = std::to_string(padf) + " " + std::to_string(padba) + " " + std::to_string(padt) + " " +
            std::to_string(padb) + " " + std::to_string(padl) + " " + std::to_string(padr);
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    return true;
}

static bool GetAttrsMaxPool3DGrad(ge::Operator &op, Format refer, int32_t &strd, int32_t &strh, int32_t &strw,
    int32_t &kd, int32_t &kh, int32_t &kw)
{
    std::vector<int32_t> strideList;
    if (GRAPH_SUCCESS != op.GetAttr("strides", strideList)) {
        OP_LOGE(op.GetName().c_str(), "Failed to get strides!");
        return GRAPH_FAILED;
    }
    std::vector<int32_t> ksizeList;
    if (GRAPH_SUCCESS != op.GetAttr("ksize", ksizeList)) {
        OP_LOGE(op.GetName().c_str(), "Failed to get ksize!");
        return GRAPH_FAILED;
    }

    if (refer == FORMAT_NCDHW) {
        strd = strideList[2];
        strh = strideList[3];
        strw = strideList[4];
        kd = ksizeList[2];
        kh = ksizeList[3];
        kw = ksizeList[4];
    } else if (refer == FORMAT_NDHWC) {
        strd = strideList[1];
        strh = strideList[2];
        strw = strideList[3];
        kd = ksizeList[1];
        kh = ksizeList[2];
        kw = ksizeList[3];
    } else {
        map<string, string> err_map;
        err_map["param_name"] = "refer";
        err_map["op_name"] = "MaxPool3DGrad";
        err_map["excepted_value"] = "NCDHW or NDHWC";
        err_map["input_value"] = refer;
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
    return true;
}

IMPLEMT_VERIFIER(MaxPool3DGrad, MaxPool3DGradVerify)
{
    if (!CheckTwoInputDtypeSame(op, "orig_x", "orig_y") || !CheckTwoInputDtypeSame(op, "orig_x", "grads")) {
        return GRAPH_FAILED;
    }
    std::vector<int64_t> ksize;
    ksize = GetAttrValue(op, "ksize");
    if (!CheckListEmpty(op.GetName(), ksize, "ksize")) {
        OP_LOGE(op.GetName().c_str(), "get attr ksize failed");
        return GRAPH_FAILED;
    }
    if (ksize.size() != 5) {
        OP_LOGE(op.GetName().c_str(), "attr ksize(%d) must be 5", (int)ksize.size());
        return GRAPH_FAILED;
    }
    std::vector<int64_t> strides;
    strides = GetAttrValue(op, "strides");
    if (!CheckListEmpty(op.GetName(), strides, "strides")) {
        OP_LOGE(op.GetName().c_str(), "get attr strides failed");
        return GRAPH_FAILED;
    }
    if (strides.size() != 5) {
        OP_LOGE(op.GetName().c_str(), "attr strides(%d) must be 5", (int)strides.size());
        return GRAPH_FAILED;
    }
    std::string data_format;
    if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
        if (data_format != "NDHWC" && data_format != "NDCHW") {
            OP_LOGE(op.GetName().c_str(), "attr data_format(%s) only support NDHWC and NDCHW", data_format.c_str());
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(MaxPool3DGrad, MaxPool3DGradInferShape)
{
    auto shapeX1 = op.GetInputDesc("orig_x").GetShape();
    TensorDesc td = op.GetOutputDesc("y");
    td.SetShape(shapeX1);
    td.SetDataType(DT_FLOAT);
    (void)op.UpdateOutputDesc("y", td);

    // get input DHW
    auto xShape = op.GetInputDesc("orig_x").GetShape().GetDims();
    auto xFormat = op.GetInputDesc("orig_x").GetFormat();
    int32_t id = 0;
    int32_t ih = 0;
    int32_t iw = 0;

    if (xFormat == FORMAT_NCDHW) {
        id = xShape[2];
        ih = xShape[3];
        iw = xShape[4];
    } else if (xFormat == FORMAT_NDHWC) {
        id = xShape[1];
        ih = xShape[2];
        iw = xShape[3];
    } else {
        map<string, string> err_map;
        err_map["param_name"] = "xFormat";
        err_map["op_name"] = "MaxPool3DGrad";
        err_map["excepted_value"] = "NCDHW or NDHWC";
        err_map["input_value"] = xFormat;
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // get ksize and strides
    int32_t kd = 0;
    int32_t kh = 0;
    int32_t kw = 0;
    int32_t strd = 0;
    int32_t strh = 0;
    int32_t strw = 0;
    if (false == GetAttrsMaxPool3DGrad(op, xFormat, strd, strh, strw, kd, kh, kw)) {
        OP_LOGE(op.GetName().c_str(), "get attrs failed.");
        return GRAPH_FAILED;
    }
    // get pad
    int32_t padf = 0;
    int32_t padba = 0;
    int32_t padt = 0;
    int32_t padb = 0;
    int32_t padl = 0;
    int32_t padr = 0;
    if ((strd == 0) || (strh == 0) || (strw == 0)) {
        OP_LOGE(op.GetName().c_str(), "strd/strh/strw should not be zero");
        return GRAPH_FAILED;
    }
    if (false ==
        GetPadMaxPool3DGrad(op, id, ih, iw, kd, kh, kw, strd, strh, strw, padf, padba, padt, padb, padl, padr)) {
        OP_LOGE(op.GetName().c_str(), "get pads attrs failed.");
        return GRAPH_FAILED;
    }

    OP_LOGD(op.GetName().c_str(), "Leave MaxPool3DGrad.");
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MaxPool3DGrad, MaxPool3DGradInferShape);
VERIFY_FUNC_REG(MaxPool3DGrad, MaxPool3DGradVerify);
// ---------------------MaxPool3DGrad---------------------

// ----------------AvgPool1D-------------------
IMPLEMT_VERIFIER(AvgPool1D, AvgPool1DVerify) {
  std::vector<int64_t> pads;
  pads = GetAttrValue(op, "pads");
  if (!CheckListEmpty(op.GetName(), pads, "pads")) {
    return GRAPH_FAILED;
  }
  int64_t kSize;
  if (ge::GRAPH_SUCCESS != op.GetAttr("ksize", kSize)) {
    OP_LOGE(op.GetName().c_str(), "Get kSize failed!");
    return GRAPH_FAILED;
  }
  int64_t strides;
  if (ge::GRAPH_SUCCESS != op.GetAttr("strides", strides)) {
    OP_LOGE(op.GetName().c_str(), "Get strides failed!");
    return GRAPH_FAILED;
  }
  bool ceilMode;
  if (ge::GRAPH_SUCCESS != op.GetAttr("ceil_mode", ceilMode)) {
    OP_LOGE(op.GetName().c_str(), "Get ceilMode failed!");
    return GRAPH_FAILED;
  }
  bool countIncludePad;
  if (ge::GRAPH_SUCCESS != op.GetAttr("count_include_pad", countIncludePad)) {
    OP_LOGE(op.GetName().c_str(), "Get countIncludePad failed!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
IMPLEMT_INFERFUNC(AvgPool1D, AvgPool1DInfer) {
  TensorDesc outputTensorDesc = op.GetOutputDesc("y");
  auto inputTensor = op.GetInputDesc("x");
  Format inputFormat = inputTensor.GetFormat();
  auto inputShape = inputTensor.GetShape();
  auto inputWSize = 0;

  if (inputFormat == FORMAT_NHWC) {
    inputWSize = inputShape.GetDim(2);
  } else if (inputFormat == FORMAT_NCHW) {
    inputWSize = inputShape.GetDim(3);
  }
  DataType inputType = inputTensor.GetDataType();
  uint32_t ksize = 0;
  uint32_t strides = 1;
  bool ceilMode = false;
  if (op.GetAttr("ksize", ksize) != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ksize failed");
  }
  if (op.GetAttr("strides", strides) != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr strides failed");
  }
  // get input ksize
  std::vector<int32_t> padsList;
  if (GRAPH_SUCCESS != op.GetAttr("pads", padsList)) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr padsList failed!");
    return GRAPH_FAILED;
  }
  if (op.GetAttr("ceil_mode", ceilMode) != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ceil_mode failed");
  }
  uint32_t padl = padsList[0];
  uint32_t padr = padsList[1];
  uint32_t outputWSize = 0;
  if (ceilMode) {
    outputWSize =
        (inputWSize + padl + padr - ksize + strides - 1) / strides + 1;
  } else {
    outputWSize = ((inputWSize + padl + padr) - ksize) / strides + 1;
  }
  if (padl) {
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
    // existing bug in pytorch code
    // padl = 0 and strides is big, but kernel is small, return nan
    if (((outputWSize - 1) * strides) >= (inputWSize + padl)) {
      outputWSize--;
    }
  }
  padr = (outputWSize - 1) * strides + ksize - inputWSize - padl;

  if (inputFormat != FORMAT_NHWC && inputFormat != FORMAT_NCHW) {
    OP_LOGE(op.GetName().c_str(), "inputFormat only support NCHW or NHWC");
    return GRAPH_FAILED;
  }

  vector<int64_t> dimVec;
  if (inputFormat == FORMAT_NHWC) {
    dimVec.push_back(inputShape.GetDim(0));
    dimVec.push_back(inputShape.GetDim(1));
    dimVec.push_back(outputWSize);
    dimVec.push_back(inputShape.GetDim(3));
  } else if (inputFormat == FORMAT_NCHW) {
    dimVec.push_back(inputShape.GetDim(0));
    dimVec.push_back(inputShape.GetDim(1));
    dimVec.push_back(inputShape.GetDim(2));
    dimVec.push_back(outputWSize);
  }
  if (dimVec.size() == 0) {
    OP_LOGE(op.GetName().c_str(), "inputFormat is not NCHW or NHWC");
    return GRAPH_FAILED;
  }

  Shape outputShape = ge::Shape(dimVec);
  DataType outputDtype = inputType;
  outputTensorDesc.SetShape(outputShape);
  outputTensorDesc.SetDataType(outputDtype);
  op.UpdateOutputDesc("y", outputTensorDesc);
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(AvgPool1D, AvgPool1DInfer);
VERIFY_FUNC_REG(AvgPool1D, AvgPool1DVerify);
// ----------------AvgPool1D END-------------------

// ----------------AvgPool1DD-------------------
IMPLEMT_VERIFIER(AvgPool1DD, AvgPool1DDVerify) {
  if (!CheckTwoInputDtypeSame(op, "x", "assist_matrix")) {
      OP_LOGE(op.GetName().c_str(), "Check matrix input dtype!");
      return GRAPH_FAILED;
  }
  std::vector<int64_t> pads;
  pads = GetAttrValue(op, "pads");
  if (!CheckListEmpty(op.GetName(), pads, "pads")) {
    return GRAPH_FAILED;
  }
  if (pads.size() != 2) {
    OP_LOGE(op.GetName().c_str(), "pads size should be two");
    return GRAPH_FAILED;
  }
  int64_t kSize;
  if (ge::GRAPH_SUCCESS != op.GetAttr("ksize", kSize)) {
    OP_LOGE(op.GetName().c_str(), "Get kSize failed!");
    return GRAPH_FAILED;
  }
  int64_t strides;
  if (ge::GRAPH_SUCCESS != op.GetAttr("strides", strides)) {
    OP_LOGE(op.GetName().c_str(), "Get strides failed!");
    return GRAPH_FAILED;
  }
  bool ceilMode;
  if (ge::GRAPH_SUCCESS != op.GetAttr("ceil_mode", ceilMode)) {
    OP_LOGE(op.GetName().c_str(), "Get ceilMode failed!");
    return GRAPH_FAILED;
  }
  bool countIncludePad;
  if (ge::GRAPH_SUCCESS != op.GetAttr("count_include_pad", countIncludePad)) {
    OP_LOGE(op.GetName().c_str(), "Get countIncludePad failed!");
    return GRAPH_FAILED;
  }

  int64_t padl = pads[0];
  int64_t padr = pads[1];
  int64_t wOutput = 0;

  auto inputTensor = op.GetInputDesc("x");
  Format inputFormat = inputTensor.GetFormat();
  auto inputShape = inputTensor.GetShape();
  auto inputWSize = 0;
  if (inputFormat == FORMAT_NHWC) {
    inputWSize = inputShape.GetDim(2);
  } else if (inputFormat == FORMAT_NCHW) {
    inputWSize = inputShape.GetDim(3);
  }

  if (ceilMode) {
    wOutput = (inputWSize + padl + padr - kSize + strides - 1) / strides + 1;
  } else {
    wOutput = ((inputWSize + padl + padr) - kSize) / strides + 1;
  }
  if (padl) {
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
    // existing bug in pytorch code
    // padl = 0 and strides is big, but kernel is small, return Nan
    if (((wOutput - 1) * strides) >= (inputWSize + padl)) {
      wOutput--;
    }
  }
  auto matrixTensor = op.GetInputDesc("assist_matrix");
  auto matrixShape = matrixTensor.GetShape();
  if (wOutput != matrixShape.GetDim(3)) {
    OP_LOGE(op.GetName().c_str(), "Check matrix shape W dimension");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(AvgPool1DD, AvgPool1DDInfer) {
  TensorDesc outputTensorDesc = op.GetOutputDesc("y");
  auto inputTensor = op.GetInputDesc("x");
  Format inputFormat = inputTensor.GetFormat();
  auto inputShape = inputTensor.GetShape();
  auto inputWSize = 0;

  if (inputFormat == FORMAT_NHWC) {
    inputWSize = inputShape.GetDim(2);
  } else if (inputFormat == FORMAT_NCHW) {
    inputWSize = inputShape.GetDim(3);
  }
  DataType inputType = inputTensor.GetDataType();
  uint32_t ksize = 0;
  uint32_t strides = 1;
  bool ceilMode = false;

  if (op.GetAttr("ksize", ksize) != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(),
            "GetOpAttr ksize failed, set ksize default value");
  }
  if (op.GetAttr("strides", strides) != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(),
            "GetOpAttr strides failed, set strides default value");
  }
  // get input ksize
  std::vector<int32_t> padsList;
  if (GRAPH_SUCCESS != op.GetAttr("pads", padsList)) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr padsList failed!");
    return GRAPH_FAILED;
  }

  if (op.GetAttr("ceil_mode", ceilMode) != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(),
            "GetOpAttr ceil_mode failed, set ceil_mode default value");
  }
  uint32_t padl = padsList[0];
  uint32_t padr = padsList[1];
  uint32_t outputWSize = 0;
  if (ceilMode) {
    outputWSize =
        (inputWSize + padl + padr - ksize + strides - 1) / strides + 1;
  } else {
    outputWSize = ((inputWSize + padl + padr) - ksize) / strides + 1;
  }
  if (padl) {
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
    // existing bug in pytorch code
    // padl = 0 and strides is big, but kernel is small, return nan
    if (((outputWSize - 1) * strides) >= (inputWSize + padl)) {
      outputWSize--;
    }
  }
  padr = (outputWSize - 1) * strides + ksize - inputWSize - padl;

  vector<int64_t> dimVec;
  if (inputFormat == FORMAT_NHWC) {
    dimVec.push_back(inputShape.GetDim(0));
    dimVec.push_back(inputShape.GetDim(1));
    dimVec.push_back(outputWSize);
    dimVec.push_back(inputShape.GetDim(3));
  } else if (inputFormat == FORMAT_NCHW) {
    dimVec.push_back(inputShape.GetDim(0));
    dimVec.push_back(inputShape.GetDim(1));
    dimVec.push_back(inputShape.GetDim(2));
    dimVec.push_back(outputWSize);
  }
  if (dimVec.size() == 0) {
    OP_LOGE(op.GetName().c_str(), "inputFormat is not NCHW or NHWC");
    return GRAPH_FAILED;
  }
  Shape outputShape = ge::Shape(dimVec);
  DataType outputDtype = inputType;
  outputTensorDesc.SetShape(outputShape);
  outputTensorDesc.SetDataType(outputDtype);
  op.UpdateOutputDesc("y", outputTensorDesc);
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(AvgPool1DD, AvgPool1DDInfer);
VERIFY_FUNC_REG(AvgPool1DD, AvgPool1DDVerify);
// ----------------AvgPool1DD END-------------------

// ----------------------MaxPoolGradWithArgmaxV2-----------------------
IMPLEMT_VERIFIER(MaxPoolGradWithArgmaxV2, MaxPoolGradWithArgmaxV2Verify)
{
    return GRAPH_SUCCESS;
}
IMPLEMT_COMMON_INFERFUNC(MaxPoolGradWithArgmaxV2InferShape)
{
    TensorDesc output_y = op.GetOutputDesc("y");

    auto tensorDesc = op.GetInputDesc("x");
    auto shape = tensorDesc.GetShape();
    output_y.SetShape(shape);

    (void)op.UpdateOutputDesc("y", output_y);
    return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(MaxPoolGradWithArgmaxV2, MaxPoolGradWithArgmaxV2InferShape);
VERIFY_FUNC_REG(MaxPoolGradWithArgmaxV2, MaxPoolGradWithArgmaxV2Verify);
// ----------------------MaxPoolGradWithArgmaxV2-----------------------

// ---------------------MaxPoolWithArgmaxV2---------------------

int cal_max(int input_size, int pad, int dilation, int kernel_size, int stride, bool ceil_mode)
{
    int max_size = 0;
    int temp = 0;

    if (stride == 0) {
        return 0;
    }

    temp = input_size + 2 * pad - dilation * (kernel_size - 1) - 1;
    if (ceil_mode) {
        max_size = (temp + stride - 1) / stride + 1;
    } else {
        max_size = temp / stride + 1;
    }
    return max_size;
}

int ceil(int a, int b)
{
    int r = 0;
    if (b == 0) {
        return 0;
    }

    if (a % b == 0) {
        r = a / b;
    } else {
        r = a / b + 1;
    }

    return r;
}

void cal_mask(int max_h, int max_w, int kernel_h, int kernel_w, int input_c0, int *mask_h, int *mask_w)
{
    int max_mul = 0;
    max_mul = max_h * max_w;
    *mask_h = kernel_h * kernel_w;
    *mask_w = ceil(max_mul, input_c0) + 1;
}

IMPLEMT_VERIFIER(MaxPoolWithArgmaxV2, MaxPoolWithArgmaxV2Verify)
{
    return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(MaxPoolWithArgmaxV2InferShape)
{
    TensorDesc output_max = op.GetOutputDesc("y");
    TensorDesc output_mask = op.GetOutputDesc("argmax");

    auto tensorDesc = op.GetInputDesc(0);
    auto shape = tensorDesc.GetShape();

    std::vector<int64_t> max_vec;
    std::vector<int64_t> mask_vec;
    std::vector<int64_t> pads;
    std::vector<int64_t> dilation;
    std::vector<int64_t> kernel_size;
    std::vector<int64_t> strides;

    int batch_size, c1_size, input_h, input_w, pad_h, pad_w, dilation_h, dilation_w, kernel_h, kernel_w;
    int max_h, max_w, mask_h, mask_w, stride_h, stride_w;
    int input_c0 = 16;
    bool ceil_mode;

    op.GetAttr("ksize", kernel_size);
    op.GetAttr("strides", strides);
    op.GetAttr("pads", pads);
    op.GetAttr("dilation", dilation);
    op.GetAttr("ceil_mode", ceil_mode);
    batch_size = shape.GetDim(0);
    c1_size = shape.GetDim(1);
    input_h = shape.GetDim(2);
    input_w = shape.GetDim(3);

    pad_h = pads[1];
    pad_w = pads[2];
    dilation_h = dilation[1];
    dilation_w = dilation[2];
    stride_h = strides[1];
    stride_w = strides[2];
    kernel_h = kernel_size[1];
    kernel_w = kernel_size[2];

    max_h = cal_max(input_h, pad_h, dilation_h, kernel_h, stride_h, ceil_mode);
    max_w = cal_max(input_w, pad_w, dilation_w, kernel_w, stride_w, ceil_mode);

    cal_mask(max_h, max_w, kernel_h, kernel_w, input_c0, &mask_h, &mask_w);

    max_vec.push_back(batch_size);
    max_vec.push_back(c1_size);
    max_vec.push_back(max_h);
    max_vec.push_back(max_w);

    mask_vec.push_back(batch_size);
    mask_vec.push_back(c1_size);
    mask_vec.push_back(mask_h);
    mask_vec.push_back(mask_w);

    ge::Shape max_shape = ge::Shape(max_vec);
    ge::Shape mask_shape = ge::Shape(mask_vec);

    output_max.SetShape(max_shape);
    output_max.SetDataType(op.GetInputDesc("x").GetDataType());
    output_max.SetFormat(op.GetInputDesc("x").GetFormat());
    output_mask.SetShape(mask_shape);
    output_mask.SetFormat(op.GetInputDesc("x").GetFormat());

    (void)op.UpdateOutputDesc("y", output_max);
    (void)op.UpdateOutputDesc("argmax", output_mask);
    return GRAPH_SUCCESS;
}
// Registered inferfunction
COMMON_INFER_FUNC_REG(MaxPoolWithArgmaxV2, MaxPoolWithArgmaxV2InferShape);

// Registered verify function
VERIFY_FUNC_REG(MaxPoolWithArgmaxV2, MaxPoolWithArgmaxV2Verify);
// ---------------------MaxPoolWithArgmaxV2---------------------
} // namespace ge
