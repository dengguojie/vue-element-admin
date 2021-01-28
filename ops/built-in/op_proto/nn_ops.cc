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
 * \file nn_ops.cpp
 * \brief
 */
#include "inc/nn_ops.h"
#include <cmath>
#include <string>
#include <vector>

#include "util/util.h"
#include "util/error_util.h"
#include "op_log.h"
#include "strided_slice_infer_shape.h"
#include "graph/utils/op_desc_utils.h"
namespace ge {
bool InTopKV2CheckInput(const Operator& op) {
  Shape shape_prediction = op.GetInputDesc("predictions").GetShape();
  Shape shape_target = op.GetInputDesc("targets").GetShape();
  int prediction_dim = shape_prediction.GetDimNum();
  if (prediction_dim != DIM_SIZE2) {
    OP_LOGE(op.GetName().c_str(), "Predictions must be 2-dimensional, but get [%d]", prediction_dim);
    return false;
  }
  size_t target_dim = shape_target.GetDimNum();
  if (target_dim != DIM_SIZE1) {
    OP_LOGE(op.GetName().c_str(), "Targets must be 1-dimensional but get [%u]", target_dim);
    return false;
  }
  if (shape_prediction.GetDim(0) != shape_target.GetDim(0)) {
    OP_LOGE(op.GetName().c_str(),
            "First dimension of predictions must match length of targets, but first dimension of predictions get [%d] "
            "and targets get [%u]", shape_prediction.GetDim(0), shape_target.GetDim(0));
    return false;
  }
  return true;
}

IMPLEMT_VERIFIER(InTopKV2, InTopKV2Verify) {
  if (!InTopKV2CheckInput(op)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(InTopKV2InferShape) {
  DataType output_dtype = DT_BOOL;
  Shape shape_target = op.GetInputDesc("targets").GetShape();
  TensorDesc tensordesc_output = op.GetOutputDesc("precision");
  tensordesc_output.SetShape(shape_target);
  tensordesc_output.SetDataType(output_dtype);
  if (op.UpdateOutputDesc("precision", tensordesc_output) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(InTopKV2, InTopKV2InferShape);
VERIFY_FUNC_REG(InTopKV2, InTopKV2Verify);
}//namespace ge