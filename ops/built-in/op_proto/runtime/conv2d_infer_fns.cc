/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
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

#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "../util/util.h"
#include "graph/common_error_codes.h"
#include "register/op_impl_registry.h"

namespace gert {
// Op Indices
// proto input
const size_t X_IDX_CONV2D = 0;
const size_t W_IDX_CONV2D = 1;
const size_t BIAS_IDX_CONV2D = 2;
const size_t OFFSET_W_IDX_CONV2D = 3;
// proto output
const size_t Y_IDX_CONV2D = 0;
// proto attribute
const size_t STRIDES_IDX_CONV2D = 0;
const size_t PADS_IDX_CONV2D = 1;
const size_t DILATIONS_IDX_CONV2D = 2;
const size_t GROUPS_IDX_CONV2D = 3;
const size_t DATA_FORMAT_IDX_CONV2D = 4;
const size_t OFFSET_X_IDX_CONV2D = 5;
// customized attribute
const size_t PADDING_IDX_CONV2D = 6;
const size_t AUTO_PAD_IDX_CONV2D = 7;

// NCHW shape
const int32_t N_DIM_IDX_NCHW = 0;
const int32_t C_DIM_IDX_NCHW = 1;
const int32_t H_DIM_IDX_NCHW = 2;
const int32_t W_DIM_IDX_NCHW = 3;
// NHWC shape
const int32_t N_DIM_IDX_NHWC = 0;
const int32_t C_DIM_IDX_NHWC = 3;
const int32_t H_DIM_IDX_NHWC = 1;
const int32_t W_DIM_IDX_NHWC = 2;
// HWCN shape
const int32_t N_DIM_IDX_HWCN = 3;
const int32_t C_DIM_IDX_HWCN = 2;
const int32_t H_DIM_IDX_HWCN = 0;
const int32_t W_DIM_IDX_HWCN = 1;
// PAD IDX
const int32_t TOP_IDX_PAD = 0;
const int32_t BOTTOM_IDX_PAD = 1;
const int32_t LEFT_IDX_PAD = 2;
const int32_t RIGHT_IDX_PAD = 3;
// STRIDES IDX
const int32_t STRD_H_NCHW = 2;
const int32_t STRD_W_NCHW = 3;
const int32_t STRD_H_NHWC = 1;
const int32_t STRD_W_NHWC = 2;

// support information
const size_t SUPPORTED_DIM_NUM = 4;
const size_t PAD_SIZE_LIMIT = 4;
const size_t STRIDE_SIZE_LIMIT = 4;
const size_t DILATION_SIZE_LIMIT = 4;

struct Conv2DInputShapes {
    int32_t in = 0;
    int32_t ic = 0;
    int32_t ih = 0;
    int32_t iw = 0;
    int32_t kn = 0;
    int32_t kc = 0;
    int32_t kh = 0;
    int32_t kw = 0;
};

struct Conv2DAttrs {
    int32_t strh = 0;
    int32_t strw = 0;
    int32_t dilh = 0;
    int32_t dilw = 0;
    int32_t padt = 0;
    int32_t padb = 0;
    int32_t padl = 0;
    int32_t padr = 0;
};

ge::graphStatus GetConv2DXShapeDim(InferShapeContext* context, Conv2DInputShapes& shapes)
{
    // Get x format
    const gert::CompileTimeTensorDesc* xDescPtr = context->GetInputDesc(X_IDX_CONV2D);
    OP_LOGE_IF(xDescPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "Get input x desc failed.");
    const ge::Format xFormat = xDescPtr->GetOriginFormat();
    OP_LOGE_IF(xFormat == ge::Format::FORMAT_RESERVED, ge::GRAPH_FAILED,
        context->GetNodeName(), "Get format failed: %d.", xFormat);
    // Get x shape
    const gert::Shape* xShape = context->GetInputShape(X_IDX_CONV2D);
    OP_LOGE_IF(xShape == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "Get xShape failed.");
    OP_LOGE_IF(xShape->GetDimNum() != SUPPORTED_DIM_NUM, ge::GRAPH_FAILED,
        context->GetNodeName(), "Not support input xShape dimnum %lu.", xShape->GetDimNum());
    // Set shapes
    if (xFormat == ge::Format::FORMAT_NCHW) {
        shapes.in = xShape->GetDim(N_DIM_IDX_NCHW);
        shapes.ic = xShape->GetDim(C_DIM_IDX_NCHW);
        shapes.ih = xShape->GetDim(H_DIM_IDX_NCHW);
        shapes.iw = xShape->GetDim(W_DIM_IDX_NCHW);
    } else if (xFormat == ge::Format::FORMAT_NHWC) {
        shapes.in = xShape->GetDim(N_DIM_IDX_NHWC);
        shapes.ic = xShape->GetDim(C_DIM_IDX_NHWC);
        shapes.ih = xShape->GetDim(H_DIM_IDX_NHWC);
        shapes.iw = xShape->GetDim(W_DIM_IDX_NHWC);
    } else {
        OP_LOGE(context->GetNodeName(), "input x format should be NCHW or NHWC, but the actual is: %s",
            ge::TypeUtils::FormatToSerialString(xFormat).c_str());
        map<string, string> errMap;
        errMap["param"] = "x";
        errMap["op_name"] = context->GetNodeName();
        errMap["expected_format_list"] = "NCHW or NHWC";
        errMap["format"] = ge::TypeUtils::FormatToSerialString(xFormat);
        std::string reportErrorCode = "E50002";
        ErrorManager::GetInstance().ReportErrMessage(reportErrorCode, errMap);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetConv2DWShapeDim(InferShapeContext* context, Conv2DInputShapes& shapes)
{
    // Get w format
    const gert::CompileTimeTensorDesc* wDescPtr = context->GetInputDesc(W_IDX_CONV2D);
    OP_LOGE_IF(wDescPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "Get input w desc failed.");
    ge::Format wFormat = wDescPtr->GetOriginFormat();
    OP_LOGE_IF(wFormat == ge::Format::FORMAT_RESERVED, ge::GRAPH_FAILED,
        context->GetNodeName(), "Get format failed: %d", wFormat);
    // Get w shape
    const gert::Shape* wShape = context->GetInputShape(W_IDX_CONV2D);
    OP_LOGE_IF(wShape == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "Get wShape failed.");
    OP_LOGE_IF(wShape->GetDimNum() != SUPPORTED_DIM_NUM, ge::GRAPH_FAILED,
        context->GetNodeName(), "Not support input wShape dimnum %lu", wShape->GetDimNum());
    // Set shapes
    if (wFormat == ge::Format::FORMAT_NCHW) {
        shapes.kn = wShape->GetDim(N_DIM_IDX_NCHW);
        shapes.kc = wShape->GetDim(C_DIM_IDX_NCHW);
        shapes.kh = wShape->GetDim(H_DIM_IDX_NCHW);
        shapes.kw = wShape->GetDim(W_DIM_IDX_NCHW);
    } else if (wFormat == ge::Format::FORMAT_NHWC) {
        shapes.kn = wShape->GetDim(N_DIM_IDX_NHWC);
        shapes.kc = wShape->GetDim(C_DIM_IDX_NHWC);
        shapes.kh = wShape->GetDim(H_DIM_IDX_NHWC);
        shapes.kw = wShape->GetDim(W_DIM_IDX_NHWC);
    } else if (wFormat == ge::Format::FORMAT_HWCN) {
        shapes.kn = wShape->GetDim(N_DIM_IDX_HWCN);
        shapes.kc = wShape->GetDim(C_DIM_IDX_HWCN);
        shapes.kh = wShape->GetDim(H_DIM_IDX_HWCN);
        shapes.kw = wShape->GetDim(W_DIM_IDX_HWCN);
    } else {
        OP_LOGE(context->GetNodeName(), "input filter format should be NCHW, NHWC or HWCN, but the actual is: %s",
            ge::TypeUtils::FormatToSerialString(wFormat).c_str());
        map<string, string> errMap;
        errMap["param"] = "filter";
        errMap["op_name"] = context->GetNodeName();
        errMap["expected_format_list"] = "NCHW, NHWC or HWCN";
        errMap["format"] = ge::TypeUtils::FormatToSerialString(wFormat);
        std::string reportErrorCode = "E50002";
        ErrorManager::GetInstance().ReportErrMessage(reportErrorCode, errMap);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetBiasChannelIdx(InferShapeContext* context, size_t biasDimNum, size_t& idxC)
{
    if (biasDimNum == SUPPORTED_DIM_NUM) {
        ge::Format biasFormat = context->GetInputDesc(BIAS_IDX_CONV2D)->GetOriginFormat();
        if (biasFormat == ge::Format::FORMAT_NCHW) {
            idxC = C_DIM_IDX_NCHW;
        } else if (biasFormat == ge::Format::FORMAT_NHWC) {
            idxC = C_DIM_IDX_NHWC;
        } else {
            OP_LOGE(context->GetNodeName(), "Input bias format should be NCHW or NHWC when shape is 4D.");
            map<string, string> errMap;
            errMap["op_name"]= context->GetNodeName();
            errMap["description"] = "Input bias format should be NCHW or NHWC when shape is 4D.";
            std::string reportErrorCode = "E50060";
            ErrorManager::GetInstance().ReportErrMessage(reportErrorCode, errMap);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckConv2DBias(InferShapeContext* context, int32_t outChannel)
{
    const gert::Shape* biasShape = context->GetInputShape(BIAS_IDX_CONV2D);
    if (biasShape == nullptr) {
        OP_LOGD(context->GetNodeName(), "No bias.");
        return ge::GRAPH_SUCCESS;
    }
    // Check bias dim num
    size_t biasDimNum = biasShape->GetDimNum();
    if (biasDimNum != 1 && biasDimNum != SUPPORTED_DIM_NUM) {
        OP_LOGE(context->GetNodeName(), "Input bias shape should be 1D or 4D.");
        map<string, string> errMap;
        errMap["op_name"]= context->GetNodeName();
        errMap["description"] = "Input bias format shoud be 1D or 4D.";
        std::string reportErrorCode = "E50060";
        ErrorManager::GetInstance().ReportErrMessage(reportErrorCode, errMap);
        return ge::GRAPH_FAILED;
    }
    // Get bias channel index
    size_t idxC = 0;
    if (GetBiasChannelIdx(context, biasDimNum, idxC) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // Check bias channel
    if (biasShape->GetDim(idxC) != outChannel) {
        OP_LOGE(context->GetNodeName(), "Input bias size of dim_c should be equal to out_channels.");
        map<string, string> errMap;
        errMap["op_name"]= context->GetNodeName();
        errMap["description"] = "Input bias size of dim_c should be equal to out_channels.";
        std::string reportErrorCode = "E50060";
        ErrorManager::GetInstance().ReportErrMessage(reportErrorCode, errMap);
        return ge::GRAPH_FAILED;
    }
    // Check bias other dim
    for (size_t i = 0; i < biasDimNum; i++) {
        if (i == idxC) {
            continue;
        }
        if (biasShape->GetDim(i) != 1) {
            OP_LOGE(context->GetNodeName(), "Input bias size of dim [N, H, W] should be equal to 1.");
            map<string, string> errMap;
            errMap["op_name"]= context->GetNodeName();
            errMap["description"] = "Input bias size of dim [N, H, W] should be equal to 1.";
            std::string reportErrorCode = "E50060";
            ErrorManager::GetInstance().ReportErrMessage(reportErrorCode, errMap);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckGroupsAndInputChannel(InferShapeContext* context,
    const int32_t ic, const int32_t kc, const int32_t groups)
{
    if (ic != kc * groups) {
        const gert::Shape* xShape = context->GetInputShape(X_IDX_CONV2D);
        OP_LOGE_IF(xShape == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "Get xShape failed.");
        const gert::Shape* wShape = context->GetInputShape(W_IDX_CONV2D);
        OP_LOGE_IF(wShape == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "Get wShape failed.");
        const gert::CompileTimeTensorDesc* xDescPtr = context->GetInputDesc(X_IDX_CONV2D);
        OP_LOGE_IF(xDescPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "Get input x desc failed.");
        const gert::CompileTimeTensorDesc* wDescPtr = context->GetInputDesc(W_IDX_CONV2D);
        OP_LOGE_IF(wDescPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "Get input filter desc failed.");
        const ge::Format xFormat = xDescPtr->GetOriginFormat();
        const ge::Format wFormat = wDescPtr->GetOriginFormat();
        std::string xStr = ge::Shape2String(*xShape);
        std::string wStr = ge::Shape2String(*wShape);

        OP_LOGE(context->GetNodeName(), "x channel should be equal to filter_channel*groups. "
            "x format is: %s, x shape is: [%s], filter format is: %s, filter shape is: [%s], groups is: %d.",
            ge::TypeUtils::FormatToSerialString(xFormat).c_str(), ge::TypeUtils::FormatToSerialString(wFormat).c_str(),
            xStr.c_str(), wStr.c_str(), (int)groups);
        map<string, string> errMap;
        errMap["op_name"] = context->GetNodeName();
        errMap["x_shape"] = xStr;
        errMap["filter_shape"] = wStr;
        errMap["groups"] = std::to_string(groups);
        std::string reportErrorCode = "E50059";
        ErrorManager::GetInstance().ReportErrMessage(reportErrorCode, errMap);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckGroupsAndOutChannel(InferShapeContext* context, const int32_t outChannel, const int32_t groups)
{
    if (groups != 0 && outChannel % groups != 0) {
        OP_LOGE(context->GetNodeName(), "out_channels should be divisible by groups.");
        map<string, string> errMap;
        errMap["op_name"] = context->GetNodeName();
        errMap["description"] = "out_channels should be divisible by groups.";
        std::string reportErrorCode = "E50060";
        ErrorManager::GetInstance().ReportErrMessage(reportErrorCode, errMap);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckConv2DGroups(InferShapeContext* context, const Conv2DInputShapes& shapes)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_LOGE_IF(attrs == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "Get attrs failed.");
    const int64_t* groupsPtr = attrs->GetAttrPointer<int64_t>(GROUPS_IDX_CONV2D);
    OP_LOGE_IF(groupsPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "Get groups failed.");
    int32_t groups = *groupsPtr;

    if (groups == 1) {
        if (shapes.ic % shapes.kc == 0) {
            groups = shapes.ic / shapes.kc;
            // runtime2.0 disallows set attrs.
            OP_LOGD(context->GetNodeName(), "Attr groups is implicitly changed.");
        } else {
            OP_LOGE(context->GetNodeName(), "in_channels(>0) should be divisible by kernel_channels when groups = 1.");
            map<string, string> errMap;
            errMap["op_name"]= context->GetNodeName();
            errMap["description"] = "in_channels(>0) should be divisible by kernel_channels when groups = 1.";
            std::string reportErrorCode = "E50060";
            ErrorManager::GetInstance().ReportErrMessage(reportErrorCode, errMap);
            return ge::GRAPH_FAILED;
        }
    }
    if (CheckGroupsAndInputChannel(context, shapes.ic, shapes.kc, groups) != ge::GRAPH_SUCCESS ||
        CheckGroupsAndOutChannel(context, shapes.kn, groups) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetConv2DStrides(InferShapeContext* context, Conv2DAttrs& conv2DAttrs)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_LOGE_IF(attrs == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "Get attrs failed.");
    const gert::ContinuousVector* stridesPtr = attrs->GetAttrPointer<gert::ContinuousVector>(STRIDES_IDX_CONV2D);
    OP_LOGE_IF(stridesPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "Get strides failed.");
    if (stridesPtr->GetSize() != STRIDE_SIZE_LIMIT) {
        OP_LOGE(context->GetNodeName(), "strides list should be 4D. actual is: %lu.", stridesPtr->GetSize());
        map<string, string> errMap;
        errMap["param_name"] = "strides";
        errMap["op_name"] = context->GetNodeName();
        errMap["expected_value"] = "4D";
        errMap["input_value"] = std::to_string(stridesPtr->GetSize()) + "D.";
        std::string reportErrorCode = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(reportErrorCode, errMap);
        return ge::GRAPH_FAILED;
    }

    const int64_t* stridesArray = reinterpret_cast<const int64_t*>(stridesPtr->GetData());
    OP_LOGE_IF(stridesArray == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "Stride is null.");
    ge::Format xFormat = context->GetInputDesc(X_IDX_CONV2D)->GetOriginFormat();
    if (xFormat == ge::Format::FORMAT_NCHW) {
        conv2DAttrs.strh = stridesArray[H_DIM_IDX_NCHW];
        conv2DAttrs.strw = stridesArray[W_DIM_IDX_NCHW];
    } else if (xFormat == ge::Format::FORMAT_NHWC) {
        conv2DAttrs.strh = stridesArray[H_DIM_IDX_NHWC];
        conv2DAttrs.strw = stridesArray[W_DIM_IDX_NHWC];
    }

    if (conv2DAttrs.strh <= 0 || conv2DAttrs.strw <= 0) {
        OP_LOGE(context->GetNodeName(),
            "strides should be positive, actual is [%d,%d].", conv2DAttrs.strh, conv2DAttrs.strw);
        map<string, string> errMap;
        errMap["param_name"] = "strides";
        errMap["op_name"] = context->GetNodeName();
        errMap["expected_value"] = "positive";
        errMap["input_value"] = std::to_string(conv2DAttrs.strh) + ", " + std::to_string(conv2DAttrs.strw);
        std::string reportErrorCode = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(reportErrorCode, errMap);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetConv2DDilations(InferShapeContext* context, Conv2DAttrs& conv2DAttrs)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_LOGE_IF(attrs == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "Get attrs failed.");
    const gert::ContinuousVector* dilationsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(DILATIONS_IDX_CONV2D);
    OP_LOGE_IF(dilationsPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "Get dilations failed.");
    if (dilationsPtr->GetSize() != DILATION_SIZE_LIMIT) {
        OP_LOGE(context->GetNodeName(), "dilations list should be 4D. actual is: %lu.", dilationsPtr->GetSize());
        map<string, string> errMap;
        errMap["param_name"] = "dilations";
        errMap["op_name"] = context->GetNodeName();
        errMap["expected_value"] = "4D";
        errMap["input_value"] = std::to_string(dilationsPtr->GetSize()) + "D.";
        std::string reportErrorCode = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(reportErrorCode, errMap);
        return ge::GRAPH_FAILED;
    }

    const int64_t* dilationsArray = reinterpret_cast<const int64_t*>(dilationsPtr->GetData());
    OP_LOGE_IF(dilationsArray == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "Dilation is null.");
    ge::Format xFormat = context->GetInputDesc(X_IDX_CONV2D)->GetOriginFormat();
    if (xFormat == ge::Format::FORMAT_NCHW) {
        conv2DAttrs.dilh = dilationsArray[H_DIM_IDX_NCHW];
        conv2DAttrs.dilw = dilationsArray[W_DIM_IDX_NCHW];
    } else if (xFormat == ge::Format::FORMAT_NHWC) {
        conv2DAttrs.dilh = dilationsArray[H_DIM_IDX_NHWC];
        conv2DAttrs.dilw = dilationsArray[W_DIM_IDX_NHWC];
    }

    if (conv2DAttrs.dilh <= 0 || conv2DAttrs.dilw <= 0) {
        OP_LOGE(context->GetNodeName(),
            "dilations should be positive, actual is [%d,%d].", conv2DAttrs.dilh, conv2DAttrs.dilw);
        map<string, string> errMap;
        errMap["param_name"] = "dilations";
        errMap["op_name"] = context->GetNodeName();
        errMap["expected_value"] = "positive";
        errMap["input_value"] = std::to_string(conv2DAttrs.dilh) + ", " + std::to_string(conv2DAttrs.dilw);
        std::string reportErrorCode = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(reportErrorCode, errMap);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferConv2DPadsWithPadding(const InferShapeContext* context, const gert::ContinuousVector* padsPtr,
    const Conv2DInputShapes& shapes, const std::string& paddingStr, Conv2DAttrs& conv2DAttrs)
{
    if (paddingStr.compare("EXPLICIT") == 0) {
        OP_LOGE_IF(padsPtr->GetSize() != PAD_SIZE_LIMIT, ge::GRAPH_FAILED, context->GetNodeName(),
            "pads list should be 4D. actual is: %lu.", padsPtr->GetSize());
    } else if (paddingStr.compare("SAME") == 0) {
        int32_t tailsH = shapes.ih % conv2DAttrs.strh;
        int32_t tailsW = shapes.iw % conv2DAttrs.strw;
        int32_t dkH = conv2DAttrs.dilh * (shapes.kh - 1) + 1;
        int32_t dkW = conv2DAttrs.dilw * (shapes.kw - 1) + 1;
        int32_t padH = std::max((tailsH > 0 ? dkH - tailsH : dkH - conv2DAttrs.strh), 0);
        int32_t padW = std::max((tailsW > 0 ? dkW - tailsW : dkW - conv2DAttrs.strw), 0);
        conv2DAttrs.padt = (padH >> 1);
        conv2DAttrs.padb = (padH >> 1) + (padH & 1);
        conv2DAttrs.padl = (padW >> 1);
        conv2DAttrs.padr = (padW >> 1) + (padW & 1);
    } else if (paddingStr.compare("VALID") == 0) {
        conv2DAttrs.padt = 0;
        conv2DAttrs.padb = 0;
        conv2DAttrs.padl = 0;
        conv2DAttrs.padr = 0;
    } else {
        OP_LOGE(context->GetNodeName(), "padding should be SAME or VALID, but the actual is: %s.", paddingStr.c_str());
        map<string, string> errMap;
        errMap["op_name"] = context->GetNodeName();
        errMap["expected_pad_mode"] = "SAME or VALID";
        errMap["actual_pad_mode"] = paddingStr;
        std::string reportErrorCode = "E50050";
        ErrorManager::GetInstance().ReportErrMessage(reportErrorCode, errMap);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferConv2DPadsWithAutoPad(const InferShapeContext* context, const gert::ContinuousVector* padsPtr,
    const Conv2DInputShapes& shapes, const std::string& autoPadStr, Conv2DAttrs& conv2DAttrs)
{
    if (autoPadStr.compare("SAME_UPPER") == 0) {
        int32_t tailsH = shapes.ih % conv2DAttrs.strh;
        int32_t tailsW = shapes.iw % conv2DAttrs.strw;
        int32_t dkH = conv2DAttrs.dilh * (shapes.kh - 1) + 1;
        int32_t dkW = conv2DAttrs.dilw * (shapes.kw - 1) + 1;
        int32_t padH = std::max((tailsH > 0 ? dkH - tailsH : dkH - conv2DAttrs.strh), 0);
        int32_t padW = std::max((tailsW > 0 ? dkW - tailsW : dkW - conv2DAttrs.strw), 0);
        conv2DAttrs.padt = (padH >> 1);
        conv2DAttrs.padb = (padH >> 1) + (padH & 1);
        conv2DAttrs.padl = (padW >> 1);
        conv2DAttrs.padr = (padW >> 1) + (padW & 1);
    } else if (autoPadStr.compare("SAME_LOWER") == 0) {
        int32_t tailsH = shapes.ih % conv2DAttrs.strh;
        int32_t tailsW = shapes.iw % conv2DAttrs.strw;
        int32_t dkH = conv2DAttrs.dilh * (shapes.kh - 1) + 1;
        int32_t dkW = conv2DAttrs.dilw * (shapes.kw - 1) + 1;
        int32_t padH = std::max((tailsH > 0 ? dkH - tailsH : dkH - conv2DAttrs.strh), 0);
        int32_t padW = std::max((tailsW > 0 ? dkW - tailsW : dkW - conv2DAttrs.strw), 0);
        conv2DAttrs.padt = (padH >> 1) + (padH & 1);
        conv2DAttrs.padb = (padH >> 1);
        conv2DAttrs.padl = (padW >> 1) + (padW & 1);
        conv2DAttrs.padr = (padW >> 1);
    } else if (autoPadStr.compare("NOTSET") == 0) {
        OP_LOGE_IF(padsPtr->GetSize() != PAD_SIZE_LIMIT, ge::GRAPH_FAILED, context->GetNodeName(),
            "pads list should be 4D. actual is: %lu.", padsPtr->GetSize());
    } else if (autoPadStr.compare("VALID") == 0) {
        conv2DAttrs.padt = 0;
        conv2DAttrs.padb = 0;
        conv2DAttrs.padl = 0;
        conv2DAttrs.padr = 0;
    } else {
        OP_LOGE(context->GetNodeName(), "Not support auto_pad %s.", autoPadStr.c_str());
        map<string, string> errMap;
        errMap["op_name"] = context->GetNodeName();
        errMap["expected_pad_mode"] = "NOTSET, SAME_UPPER, SAME_LOWER or VALID";
        errMap["actual_pad_mode"] = autoPadStr;
        std::string reportErrorCode = "E50050";
        ErrorManager::GetInstance().ReportErrMessage(reportErrorCode, errMap);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CutPads(InferShapeContext* context, const Conv2DInputShapes& shapes, Conv2DAttrs& attrs)
{
    int32_t ho = (shapes.ih + attrs.padt + attrs.padb - (shapes.kh - 1) * attrs.dilh - 1) / attrs.strh + 1;
    int32_t hr = (shapes.ih + attrs.padt + attrs.padb - (shapes.kh - 1) * attrs.dilh - 1) % attrs.strh;
    if ((ho == 1) && (hr <= attrs.padb)) {
        attrs.padb -= hr;
    }
    int32_t wo = (shapes.iw + attrs.padl + attrs.padr - (shapes.kw - 1) * attrs.dilw - 1) / attrs.strw + 1;
    int32_t wr = (shapes.iw + attrs.padl + attrs.padr - (shapes.kw - 1) * attrs.dilw - 1) % attrs.strw;
    if ((wo == 1) && (wr <= attrs.padr)) {
        attrs.padr -= wr;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckPositivePads(InferShapeContext* context, Conv2DAttrs& conv2DAttrs)
{
    if (conv2DAttrs.padt < 0 || conv2DAttrs.padb < 0 || conv2DAttrs.padl < 0 || conv2DAttrs.padr < 0) {
        OP_LOGE(context->GetNodeName(), "pads should be positive, but the actual is [%d,%d,%d,%d].",
            conv2DAttrs.padt, conv2DAttrs.padb, conv2DAttrs.padl, conv2DAttrs.padr);
        map<string, string> errMap;
        errMap["param_name"] = "pads";
        errMap["op_name"] = context->GetNodeName();
        errMap["expected_value"] = "positive";
        errMap["input_value"] = std::to_string(conv2DAttrs.padt) + ", " + std::to_string(conv2DAttrs.padb) + ", " +
            std::to_string(conv2DAttrs.padl) + ", " + std::to_string(conv2DAttrs.padr);
        std::string reportErrorCode = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(reportErrorCode, errMap);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetConv2DPads(InferShapeContext* context, const Conv2DInputShapes& shapes, Conv2DAttrs& conv2DAttrs)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_LOGE_IF(attrs == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "Get attrs failed.");
    const gert::ContinuousVector* padsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(PADS_IDX_CONV2D);
    OP_LOGE_IF(padsPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "Get pads failed.");
    if (padsPtr->GetSize() == PAD_SIZE_LIMIT) {
        const int64_t* padsArray = reinterpret_cast<const int64_t*>(padsPtr->GetData());
        OP_LOGE_IF(padsArray == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "Pad is null.");
        conv2DAttrs.padt = static_cast<int32_t>(padsArray[TOP_IDX_PAD]);
        conv2DAttrs.padb = static_cast<int32_t>(padsArray[BOTTOM_IDX_PAD]);
        conv2DAttrs.padl = static_cast<int32_t>(padsArray[LEFT_IDX_PAD]);
        conv2DAttrs.padr = static_cast<int32_t>(padsArray[RIGHT_IDX_PAD]);
    }

    // Infer pads if "padding" is defined.
    if (attrs->GetAttrNum() > PADDING_IDX_CONV2D) {
        const char* paddingPtr = attrs->GetAttrPointer<char>(PADDING_IDX_CONV2D);
        std::string paddingStr = paddingPtr == nullptr ? "NULL" : paddingPtr;
        if (InferConv2DPadsWithPadding(context, padsPtr, shapes, paddingStr, conv2DAttrs) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    // Infer pads if "auto_pad" from ONNX is defined.
    if (attrs->GetAttrNum() > AUTO_PAD_IDX_CONV2D) {
        const char* autoPadPtr = attrs->GetAttrPointer<char>(AUTO_PAD_IDX_CONV2D);
        std::string autoPadStr = autoPadPtr == nullptr ? "NULL" : autoPadPtr;
        if (InferConv2DPadsWithAutoPad(context, padsPtr, shapes, autoPadStr, conv2DAttrs) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    // >>> start: cut off right/bottom pad when output size is 1
    CutPads(context, shapes, conv2DAttrs);
    // <<< end: cut off right/bottom pad when output size is 1

    return CheckPositivePads(context, conv2DAttrs);
}

ge::graphStatus CheckConv2DInputWithPad(const char* nodeName, int64_t ihPad, int64_t iwPad)
{
    if ((ihPad < 0) || (iwPad < 0)) {
        const char* opName = (nodeName == nullptr) ? "nil" : nodeName;
        OP_LOGE(opName, "image size after padding should be greater than or equal to filter size.");
        map<string, string> errMap;
        errMap["op_name"] = opName;
        errMap["description"] = "image size after padding should be greater than or equal to filter size.";
        std::string reportErrorCode = "E50060";
        ErrorManager::GetInstance().ReportErrMessage(reportErrorCode, errMap);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetConv2DYShape(InferShapeContext* context, int64_t in, int64_t kn, int64_t oh, int64_t ow)
{
    const char* opName = (context->GetNodeName() == nullptr) ? "nil" : context->GetNodeName();

    gert::Shape* yShape = context->GetOutputShape(Y_IDX_CONV2D);
    OP_LOGE_IF(yShape == nullptr, ge::GRAPH_FAILED, opName, "Get yShape failed.");
    const gert::CompileTimeTensorDesc* yDescPtr = context->GetOutputDesc(Y_IDX_CONV2D);
    OP_LOGE_IF(yDescPtr == nullptr, ge::GRAPH_FAILED, opName, "Get y desc failed.");
    ge::Format yFormat = yDescPtr->GetOriginFormat();
    OP_LOGE_IF(ge::Format::FORMAT_RESERVED == yFormat, ge::GRAPH_FAILED, opName, "get format failed: %d", yFormat);

    yShape->SetDimNum(SUPPORTED_DIM_NUM);
    if (yFormat == ge::Format::FORMAT_NCHW) {
        yShape->SetDim(N_DIM_IDX_NCHW, in);
        yShape->SetDim(C_DIM_IDX_NCHW, kn);
        yShape->SetDim(H_DIM_IDX_NCHW, oh);
        yShape->SetDim(W_DIM_IDX_NCHW, ow);
    } else if (yFormat == ge::Format::FORMAT_NHWC) {
        yShape->SetDim(N_DIM_IDX_NHWC, in);
        yShape->SetDim(C_DIM_IDX_NHWC, kn);
        yShape->SetDim(H_DIM_IDX_NHWC, oh);
        yShape->SetDim(W_DIM_IDX_NHWC, ow);
    } else {
        OP_LOGE(opName, "output y format should be NCHW or NHWC, but the actual is: %s",
            ge::TypeUtils::FormatToSerialString(yFormat).c_str());
        map<string, string> errMap;
        errMap["param"] = "y";
        errMap["op_name"] = opName;
        errMap["expected_format_list"] = "NCHW or NHWC";
        errMap["format"] = ge::TypeUtils::FormatToSerialString(yFormat);
        std::string reportErrorCode = "E50002";
        ErrorManager::GetInstance().ReportErrMessage(reportErrorCode, errMap);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeForConv2D(InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter shape infer.");

    Conv2DInputShapes shapes;
    if (GetConv2DXShapeDim(context, shapes) != ge::GRAPH_SUCCESS ||
        GetConv2DWShapeDim(context, shapes) != ge::GRAPH_SUCCESS ||
        CheckConv2DBias(context, shapes.kn) != ge::GRAPH_SUCCESS ||
        CheckConv2DGroups(context, shapes) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    Conv2DAttrs conv2DAttrs;
    if (GetConv2DStrides(context, conv2DAttrs) != ge::GRAPH_SUCCESS ||
        GetConv2DDilations(context, conv2DAttrs) != ge::GRAPH_SUCCESS ||
        GetConv2DPads(context, shapes, conv2DAttrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    int64_t ihPad = shapes.ih + conv2DAttrs.padt + conv2DAttrs.padb - conv2DAttrs.dilh * (shapes.kh - 1) - 1;
    int64_t iwPad = shapes.iw + conv2DAttrs.padl + conv2DAttrs.padr - conv2DAttrs.dilw * (shapes.kw - 1) - 1;
    int64_t oh = ihPad / conv2DAttrs.strh + 1;
    int64_t ow = iwPad / conv2DAttrs.strw + 1;

    if (CheckConv2DInputWithPad(context->GetNodeName(), ihPad, iwPad) != ge::GRAPH_SUCCESS ||
        SetConv2DYShape(context, (int64_t)shapes.in, (int64_t)shapes.kn, oh, ow) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "Leave shape infer.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(Conv2D).InferShape(InferShapeForConv2D);
} // namespace gert