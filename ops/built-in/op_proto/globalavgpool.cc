/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file globalavgpool.cpp
 * \brief
 */
#include "inc/nn_detect_ops.h"

#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#include "util/util.h"
#include "util/error_util.h"
#include "op_log.h"
#include "./inc/globalavgpool.h"

namespace ge {

IMPLEMT_INFERFUNC(GlobalAveragePool, GlobalAveragePoolInferShape) {
  TensorDesc input_desc = op.GetInputDesc("x");
  TensorDesc output_desc = op.GetOutputDesc("y");

  int64_t num_shape = 1;
  vector<int64_t> output_shape;
  DataType input_dtype = input_desc.GetDataType();
  Format input_format = input_desc.GetFormat();
  vector<int64_t> input_shape = input_desc.GetShape().GetDims();
  int64_t x_shape = input_desc.GetShape().GetDims().size();
  output_shape.push_back(input_shape[0]);
  output_shape.push_back(input_shape[1]);

  if (x_shape == 5 && input_format == FORMAT_NCDHW) {
    output_shape.push_back(num_shape);
    output_shape.push_back(num_shape);
    output_shape.push_back(num_shape);
  } else if (x_shape == 4 && input_format == FORMAT_NCHW) {
    output_shape.push_back(num_shape);
    output_shape.push_back(num_shape);
  } else if (x_shape == 3 && input_format == FORMAT_ND) {
    output_shape.push_back(num_shape);
  } else {
    OP_LOGE(op.GetName().c_str(), "x_shape error or format error");
    return GRAPH_FAILED;
  }

  output_desc.SetShape(ge::Shape(output_shape));
  output_desc.SetDataType(input_dtype);
  output_desc.SetFormat(input_format);

  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(GlobalAveragePool, GlobalAveragePoolVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(GlobalAveragePool, GlobalAveragePoolInferShape);
VERIFY_FUNC_REG(GlobalAveragePool, GlobalAveragePoolVerify);

}  // namespace ge