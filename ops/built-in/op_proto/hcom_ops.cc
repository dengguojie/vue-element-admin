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
 * @file hcom_ops.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "inc/hcom_ops.h"
#include "op_log.h"
#include <string>
#include <vector>
#include <algorithm>

namespace ge {
// HcomAllGather 算子
IMPLEMT_INFERFUNC(HcomAllGather, HcomAllGatherInferShape) {
    auto inTensorDesc = op.get_input_desc_x();
    auto outTensorDesc = inTensorDesc;
    auto inShape = inTensorDesc.GetShape();
    std::vector<int64_t> inDims = inShape.GetDims();
    int64_t rankSize = op.get_attr_rank_size();
    std::vector<int64_t> outDims;
    if (rankSize <= 0) {
        OP_LOGE(op.GetName().c_str(), "attr rank_size is illegal, expected: > 0, actual: %ld.", rankSize);
        return GRAPH_FAILED;
    }
    if (inDims.size() == 0) {
        OP_LOGE(op.GetName().c_str(), "input tensor's first dim is illegal, expected: > 0, actual: %zu.", \
            inDims.size());
        return GRAPH_FAILED;
    }
    outDims = inDims;
    outDims[0] = inDims[0] * rankSize;
    ge::Shape outputShape = ge::Shape(outDims);
    ge::DataType outputDtype = inTensorDesc.GetDataType();
    outTensorDesc.SetShape(outputShape);
    outTensorDesc.SetDataType(outputDtype);
    op.update_output_desc_y(outTensorDesc);
    OP_LOGI(op.GetName().c_str(), "the op infershape end");
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HcomAllGather, HcomAllGatherVerify) {
    std::vector<int64_t> inDims = op.get_input_desc_x().GetShape().GetDims();
    int64_t rankSize = op.get_attr_rank_size();
    if (rankSize <= 0) {
        OP_LOGE(op.GetName().c_str(), "attr rank_size is illegal, expected: > 0, actual: %ld.", rankSize);
        return GRAPH_FAILED;
    }
    if (inDims.size() == 0) {
        OP_LOGE(op.GetName().c_str(), "input tensor's first dim is illegal, expected: > 0, actual: %zu.", \
            inDims.size());
        return GRAPH_FAILED;
    }
    OP_LOGI(op.GetName().c_str(), "the op verify end");
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(HcomAllGather, HcomAllGatherInferShape);
VERIFY_FUNC_REG(HcomAllGather, HcomAllGatherVerify);

// HcomAllReduce 算子
IMPLEMT_VERIFIER(HcomAllReduce, HcomAllReduceVerify) {
    constexpr int64_t fusionAttrMinVal = 0;
    constexpr int64_t fusionAttrMaxVal = 2;
    constexpr int64_t fusionIdMinVal = -1;
    constexpr int64_t fusionIdMaxVal = 0x7fffffff;
    std::string reduction = op.get_attr_reduction();
    const std::vector<std::string> SUPPORTED_REDUCTION = {
        "min", "max", "prod", "sum"
    };
    auto it = std::find(SUPPORTED_REDUCTION.begin(), SUPPORTED_REDUCTION.end(), reduction);
    if (it == SUPPORTED_REDUCTION.end()) {
        OP_LOGE(op.GetName().c_str(), "Attr reduction [%s] is not supported. expecttd: min, max, prod, sum", \
            reduction.c_str());
        return GRAPH_FAILED;
    }
    int64_t fusionAttr;
    if (op.GetAttr("fusion", fusionAttr) == GRAPH_SUCCESS) {
        if ((fusionAttr < fusionAttrMinVal) || (fusionAttr > fusionAttrMaxVal)) {
            OP_LOGE(op.GetName().c_str(), "Attr fusion [%lld] is not supported. expecttd: [%lld ~ %lld]", \
                fusionAttr, fusionAttrMinVal, fusionAttrMaxVal);
            return GRAPH_FAILED;
        }
    }
    int64_t fusionIdAttr;
    if (op.GetAttr("fusion_id", fusionIdAttr) == GRAPH_SUCCESS) {
        if ((fusionIdAttr < fusionIdMinVal) || (fusionIdAttr > fusionIdMaxVal)) {
            OP_LOGE(op.GetName().c_str(), "Attr fusion_id [%lld] is not supported. expecttd: [%lld ~ %lld]", \
                fusionIdAttr, fusionIdMinVal, fusionIdMaxVal);
            return GRAPH_FAILED;
        }
    }
    OP_LOGI(op.GetName().c_str(), "the op verify end");
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(HcomAllReduce, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
VERIFY_FUNC_REG(HcomAllReduce, HcomAllReduceVerify);

// HcomBroadcast 算子
IMPLEMT_INFERFUNC(HcomBroadcast, HcomBroadcastInferShape) {
    const unsigned int UINT_MAX_VALUE = 0xFFFFFFFF;
    auto inputsSize = op.GetInputsSize();
    if (inputsSize >= UINT_MAX_VALUE) {
        OP_LOGE(op.GetName().c_str(), "GetInputsSize [%zu] is more than %u", inputsSize, UINT_MAX_VALUE);
        return GRAPH_FAILED;
    }
    for (size_t i = 0; i < inputsSize; i++) {
        auto outputDesc = op.get_dynamic_input_desc_x(i);
        op.update_dynamic_output_desc_y(i, outputDesc);
    }
    OP_LOGI(op.GetName().c_str(), "the op infershape end");
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HcomBroadcast, HcomBroadcastVerify) {
    OP_LOGI(op.GetName().c_str(), "the op verify end");
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(HcomBroadcast, HcomBroadcastInferShape);
VERIFY_FUNC_REG(HcomBroadcast, HcomBroadcastVerify);

// HcomReduceScatter 算子
IMPLEMT_INFERFUNC(HcomReduceScatter, HcomReduceScatterInferShape) {
    auto inTensorDesc = op.get_input_desc_x();
    auto outTensorDesc = inTensorDesc;
    auto inShape = inTensorDesc.GetShape();
    std::vector<int64_t> inDims = inShape.GetDims();
    int64_t rankSize = op.get_attr_rank_size();
    std::vector<int64_t> outDims;
    if (rankSize <= 0) {
        OP_LOGE(op.GetName().c_str(), "attr rank_size is illegal, expected: > 0, actual: %ld.", rankSize);
        return GRAPH_FAILED;
    }
    if (inDims.size() == 0) {
        OP_LOGE(op.GetName().c_str(), "input tensor's first dim is illegal, expected: > 0, actual: %zu.", \
            inDims.size());
        return GRAPH_FAILED;
    }
    if (inDims[0] % rankSize) {
        OP_LOGE(op.GetName().c_str(), "input tensor's first dim is illegal, expected: rankSize[%ld] * N " \
            "(N is positive integer), actual: %ld.", rankSize, inDims[0]);
        return GRAPH_FAILED;
    }
    outDims = inDims;
    outDims[0] = inDims[0] / rankSize;
    ge::Shape outputShape = ge::Shape(outDims);
    ge::DataType outputDtype = inTensorDesc.GetDataType();
    outTensorDesc.SetShape(outputShape);
    outTensorDesc.SetDataType(outputDtype);
    op.update_output_desc_y(outTensorDesc);
    OP_LOGI(op.GetName().c_str(), "the op infershape end");
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HcomReduceScatter, HcomReduceScatterVerify) {
    std::string reduction = op.get_attr_reduction();
    const std::vector<std::string> SUPPORTED_REDUCTION = {
        "min", "max", "prod", "sum"
    };
    auto it = std::find(SUPPORTED_REDUCTION.begin(), SUPPORTED_REDUCTION.end(), reduction);
    if (it == SUPPORTED_REDUCTION.end()) {
        OP_LOGE(op.GetName().c_str(), "Attr reduction [%s] is not supported. expected: min, max, prod, sum", \
            reduction.c_str());
        return GRAPH_FAILED;
    }
    std::vector<int64_t> inDims = op.get_input_desc_x().GetShape().GetDims();
    int64_t rankSize = op.get_attr_rank_size();
    if (rankSize <= 0) {
        OP_LOGE(op.GetName().c_str(), "attr rank_size is illegal, expected: > 0, actual: %ld.", rankSize);
        return GRAPH_FAILED;
    }
    if (inDims.size() == 0) {
        OP_LOGE(op.GetName().c_str(), "input tensor's first dim is illegal, expected: > 0, actual: %zu.", \
            inDims.size());
        return GRAPH_FAILED;
    }
    if (inDims[0] % rankSize) {
        OP_LOGE(op.GetName().c_str(), "input tensor's first dim is illegal, expected: rankSize[%ld] * N " \
            "(N is positive integer), actual:%ld.", rankSize, inDims[0]);
        return GRAPH_FAILED;
    }
    OP_LOGI(op.GetName().c_str(), "the op verify end");
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(HcomReduceScatter, HcomReduceScatterInferShape);
VERIFY_FUNC_REG(HcomReduceScatter, HcomReduceScatterVerify);

// HcomSend 算子
IMPLEMT_INFERFUNC(HcomSend, HcomSendInferShape) {
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HcomSend, HcomSendVerify) {
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(HcomSend, HcomSendInferShape);
VERIFY_FUNC_REG(HcomSend, HcomSendVerify);

// HcomReceive 算子
IMPLEMT_INFERFUNC(HcomReceive, HcomReceiveInferShape) {
    TensorDesc outTensorDesc = op.get_output_desc_y();
    std::vector<int64_t> shapeSize{};
    op.GetAttr("shape", shapeSize);
    outTensorDesc.SetShape(ge::Shape(shapeSize));
    uint32_t dataType = op.get_attr_dtype();
    outTensorDesc.SetDataType((DataType)dataType);
    op.update_output_desc_y(outTensorDesc);
    OP_LOGI(op.GetName().c_str(), "the op infershape end");
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HcomReceive, HcomReceiveVerify) {
    TensorDesc outTensorDesc = op.get_output_desc_y();
    std::vector<int64_t> shapeSize{};
    op.GetAttr("shape", shapeSize);
    if (shapeSize.size() == 0) {
        OP_LOGE(op.GetName().c_str(), "Attr shape is illegal, mast be > 0");
    }
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(HcomReceive, HcomReceiveInferShape);
VERIFY_FUNC_REG(HcomReceive, HcomReceiveVerify);
} // namespace ge