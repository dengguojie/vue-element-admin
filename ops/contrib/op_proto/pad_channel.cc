/**
Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
Description: op_proto for padchannel caffe operator
Author:
Create: 2020-6-15
*/


#include "pad_channel.h"
#include <iostream>
#include <string>
#include <vector>


namespace ge {
const uint32_t DIM0 = 0;
const uint32_t DIM1 = 1;
const uint32_t DIM2 = 2;
const uint32_t DIM3 = 3;

IMPLEMT_VERIFIER(PadChannel, PadChannelVerify) {
    auto tensorDesc = op.GetInputDesc(0);
    auto shape = tensorDesc.GetShape();
    uint32_t numChannelsToPad = 0;
    op.GetAttr("num_channels_to_pad", numChannelsToPad);

    if ((shape.GetDim(1) % 16 != 0) || (numChannelsToPad % 16 != 0)) {
        printf(" Input shape and num_channels_to_pad should align to C0 ");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(PadChannelInferShape) {
    printf("PadChannel infer shape begin.\n");
    // 获取算子的输入张量描述赋值给输出张量，获取输入张量的形状
    auto tensorDesc = op.GetInputDesc(0);
    auto shape = tensorDesc.GetShape();
    TensorDesc tensordesc_output = op.GetOutputDesc("output0");

    uint32_t num_channels_to_pad = 0;
    op.GetAttr("num_channels_to_pad", num_channels_to_pad);

    // 计算输出张量的shape，并赋值
    shape.SetDim(DIM0, shape.GetDim(DIM0));
    shape.SetDim(DIM1, num_channels_to_pad + shape.GetDim(DIM1));
    shape.SetDim(DIM2, shape.GetDim(DIM2));
    shape.SetDim(DIM3, shape.GetDim(DIM3));
    tensordesc_output.SetShape(shape);
    tensordesc_output.SetDataType(tensorDesc.GetDataType());

    (void)op.UpdateOutputDesc("output0", tensorDesc);
    return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(PadChannel, PadChannelInferShape);

// Registered verify function
VERIFY_FUNC_REG(PadChannel, PadChannelVerify);
}