/* *
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file  condtake_ops.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "inc/condtake_ops.h"
#include <unordered_set>
#include "op_log.h"
#include "util/common_shape_fns.h"

namespace ge {
IMPLEMT_INFERFUNC(CondTake, CondTakeInfer)
{
    TensorDesc output_data_desc = op.GetOutputDesc("out_data");
    output_data_desc.SetDataType(DT_FLOAT);
    TensorDesc output_index_desc = op.GetOutputDesc("out_index");
    output_index_desc.SetDataType(DT_INT32);
    TensorDesc output_num_desc = op.GetOutputDesc("valid_num");
    output_num_desc.SetDataType(DT_INT32);

    Shape input_data_shape = op.GetInputDesc("data").GetShape();
    output_data_desc.SetShape(input_data_shape);
    output_index_desc.SetShape(input_data_shape);
    output_num_desc.SetShape(Shape({ 1 }));
    if (op.UpdateOutputDesc("out_data", output_data_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "out_data update failed!\n");
        return GRAPH_FAILED;
    }
    if (op.UpdateOutputDesc("out_index", output_index_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "out_index update failed!\n");
        return GRAPH_FAILED;
    }
    if (op.UpdateOutputDesc("valid_num", output_num_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "valid_num update failed!\n");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(CondTake, CondTakeInfer);
}
