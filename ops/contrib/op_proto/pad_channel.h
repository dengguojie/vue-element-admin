/**
* Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
* Description: op_proto for padchannel caffe operator
* Author:huawei
* Create: 2020-6-15
*
* @file pad_channel.h
*
* @version 1.0
*/

#ifndef GE_OP_PADCHANNEL_H
#define GE_OP_PADCHANNEL_H

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief pad channel
*
* @par Inputs:
* @li input0: input data, float16, NC1HWC0
*
* @par Attributes:
* @li num_channels_to_pad: optional, int
*
* @par Outputs:
* @li output0: output data, float16, NC1HWC0
*
* @attention Constraints:
* @li only match Attention-Resnet(JD_AI)
*/

REG_OP(PadChannel)
    .INPUT(input0, TensorType({DT_FLOAT16}))
    .OUTPUT(output0, TensorType({DT_FLOAT16}))
    .ATTR(num_channels_to_pad, Int, 0)
    .OP_END_FACTORY_REG(PadChannel)
}

#endif