/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

/*!
 * \file wts_arq.cpp
 * \brief
 */
#include <vector>
#include <string>

#include "math_ops.h"
#include "op_log.h"


namespace ge {

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(WtsARQInferShape) {
    Shape w_shape = op.GetInputDesc("w").GetShape();
    Shape w_min_shape = op.GetInputDesc("w_min").GetShape();
    Shape w_max_shape = op.GetInputDesc("w_max").GetShape();

    if (w_shape.GetDimNum() != w_min_shape.GetDimNum()) {
        OP_LOGE(TbeGetName(op).c_str(), "The dimension of w_min must be the same as w!");
        return GRAPH_FAILED;
    }

    if (w_shape.GetDimNum() != w_max_shape.GetDimNum()) {
        OP_LOGE(TbeGetName(op).c_str(), "The dimension of w_max must be the same as w!");
        return GRAPH_FAILED;
    }

    std::vector<int64_t> w_dims = w_shape.GetDims();
    std::vector<int64_t> w_min_dims = w_min_shape.GetDims();
    std::vector<int64_t> w_max_dims = w_max_shape.GetDims();

    if (w_min_dims != w_max_dims) {
        OP_LOGE(TbeGetName(op).c_str(), "The shape of w_min must be the same as w_max!");
        return GRAPH_FAILED;
    }

    for (size_t i = 0; i < w_dims.size(); i++) {
        if ((w_min_dims[i] != w_dims[i]) && (w_min_dims[i] != 1)) {
            OP_LOGE(TbeGetName(op).c_str(), "The shape of w_min&w_max must be the same as w or equal to 1!");
            return GRAPH_FAILED;
        }
    }

    TensorDesc y = op.GetOutputDesc("y");
    y.SetShape(w_shape);
    y.SetDataType(op.GetInputDesc("w").GetDataType());
    if (op.UpdateOutputDesc("y", y) != GRAPH_SUCCESS) {
        OP_LOGE(TbeGetName(op).c_str(), "Update output[y] failed!");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(WtsARQ, WtsARQVerify) {
    DataType w_type = op.GetInputDesc("w").GetDataType();
    DataType w_min_type = op.GetInputDesc("w_min").GetDataType();
    DataType w_max_type = op.GetInputDesc("w_max").GetDataType();

    if (w_type != w_min_type) {
        OP_LOGE(TbeGetName(op).c_str(), "The type of w_min must be the same as w!");
        return GRAPH_FAILED;
    }

    if (w_type != w_max_type) {
        OP_LOGE(TbeGetName(op).c_str(), "The type of w_max must be the same as w!");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(WtsARQ, WtsARQInferShape);

// Registered verify function
VERIFY_FUNC_REG(WtsARQ, WtsARQVerify);
}  // namespace ge
