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
 * \file axpy_v1.cpp
 * \brief
 */
#include "axpy_v1.h"
#include <cstdint>
#include <string>
#include <vector>
#include "graph/ge_error_codes.h"
#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
IMPLEMT_VERIFIER(AxpyV1, AxpyV1Verity)
{
  DataType input_type_a = op.GetInputDesc("input_a").GetDataType();
  DataType input_type_x = op.GetInputDesc("input_x").GetDataType();
  DataType input_type_y = op.GetInputDesc("input_y").GetDataType();

  if ((input_type_a != input_type_x) || (input_type_x != input_type_y)) {
    printf("[Plugin][ERROR] input type is not consistent\n");
    return GRAPH_FAILED;
  }

  Shape a_shape = op.GetInputDesc("input_a").GetShape();
  Shape x_shape = op.GetInputDesc("input_x").GetShape();
  Shape y_shape = op.GetInputDesc("input_y").GetShape();

  std::vector<int64_t> dims_a = a_shape.GetDims();
  std::vector<int64_t> dims_x = x_shape.GetDims();
  std::vector<int64_t> dims_y = y_shape.GetDims();

  if (dims_a.size() != 2 and dims_a.size() != 4) {
    printf("[Plugin][ERROR] size of input1 shape is not 2 or 4\n");
    return GRAPH_FAILED;
  }
  for (size_t i = 0; i < dims_x.size(); i++) {
    if ((dims_x[i] != dims_y[i])) {
      printf("[Plugin][ERROR] input2 shape is not same as input3\n");
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AxpyV1InferShape)
{
  TensorDesc out_desc = op.GetOutputDesc("output_z");

  Shape x_shape = op.GetInputDesc("input_x").GetShape();
  DataType datatype_x = op.GetInputDesc("input_x").GetDataType();
  Format input_format_x = op.GetInputDesc("input_x").GetFormat();
  out_desc.SetShape(x_shape);
  out_desc.SetDataType(datatype_x);
  out_desc.SetFormat(input_format_x);
  (void)op.UpdateOutputDesc("output_z", out_desc);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AxpyV1, AxpyV1InferShape);
VERIFY_FUNC_REG(AxpyV1, AxpyV1Verity);
}
