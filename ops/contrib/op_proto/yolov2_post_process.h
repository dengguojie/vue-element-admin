/**
* Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
* Description: op_proto for yolov2_post_process
* Author: Huawei
* Create: 20200310
*
* @file postprocess.h
*
* @version 1.0
*/


#ifndef GE_OP_POSTPROCESS_H
#define GE_OP_POSTPROCESS_H

#include "graph/operator_reg.h"

namespace ge {
REG_OP(Yolov2PostProcess)
    .INPUT(box_encoding_gm, TensorType({DT_FLOAT16}))   // box_encoding_gm
    .INPUT(scores_gm, TensorType({DT_FLOAT16}))         // scores_gm
    .INPUT(anchorData, TensorType({DT_FLOAT16}))        // anchor_data
    .OUTPUT(boxout_gm, TensorType({DT_FLOAT16}))        // boxout_gm
    .OP_END_FACTORY_REG(Yolov2PostProcess);
} // namespace ge

#endif // GE_OP_POSTPROCESS_H
