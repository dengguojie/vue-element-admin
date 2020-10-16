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
 * @file copy_ops.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include "inc/array_ops.h"
#include <string>
#include <vector>
#include "util/util.h"
#include "op_log.h"

namespace ge {

IMPLEMT_INFERFUNC(Copy,CopyInferShape){

    TensorDesc tensordesc = op.GetInputDesc("x");
    Shape input_shape = tensordesc.GetShape();
    DataType input_dtype = tensordesc.GetDataType();
    Format input_format = tensordesc.GetFormat();
    TensorDesc td = op.GetOutputDesc("y");

    int64_t top_size;
    if (GRAPH_SUCCESS != op.GetAttr("N", top_size)) {
        OP_LOGE(op.GetName().c_str(), "GetAttr of N failed.");
        return GRAPH_FAILED;
    }

    if (top_size < 1){
        OP_LOGE(op.GetName().c_str(), "the number of top need greater than or equals to 1.");
        return GRAPH_FAILED;
    }

    for (int64_t i = 0; i < top_size; ++i) {
        td.SetShape(input_shape);
        td.SetDataType(input_dtype);
        td.SetFormat(input_format);
        op.UpdateDynamicOutputDesc("y", i, td);
    }
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Copy,CopyVerify){

    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Copy,CopyInferShape);
VERIFY_FUNC_REG(Copy,CopyVerify);

}  // namespace ge
