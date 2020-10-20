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
 * \file condtake_ops.cpp
 * \brief
 */
#include "inc/condtake_ops.h"
#include <unordered_set>
#include "op_log.h"
#include "util/common_shape_fns.h"

namespace ge {
IMPLEMT_INFERFUNC(CondTake, CondTakeInfer) {
  TensorDesc output_data_desc = op.GetOutputDesc("out_data");
  output_data_desc.SetDataType(DT_FLOAT);
  TensorDesc output_index_desc = op.GetOutputDesc("out_index");
  output_index_desc.SetDataType(DT_INT32);
  TensorDesc output_num_desc = op.GetOutputDesc("valid_num");
  output_num_desc.SetDataType(DT_INT32);

  Shape input_data_shape = op.GetInputDesc("data").GetShape();
  output_data_desc.SetShape(input_data_shape);
  output_index_desc.SetShape(input_data_shape);
  output_num_desc.SetShape(Shape({1}));
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
}  // namespace ge
