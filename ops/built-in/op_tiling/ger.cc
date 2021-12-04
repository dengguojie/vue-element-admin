/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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

#include <iostream>
#include "error_log.h"
#include "op_log.h"
#include "vector_tiling.h"

namespace optiling {
bool GerTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
               OpRunInfo& run_info) {
  OP_TILING_CHECK(op_paras.inputs.empty(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs cannot be empty"),
                  return false);
  OP_TILING_CHECK(op_paras.outputs.empty(),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.outputs cannot be empty"), return false);
  OP_TILING_CHECK(op_paras.inputs.size() != 2,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs size needs to be 2"), return false);
  OP_TILING_CHECK(op_paras.inputs[0].tensor.empty(),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs[0].tensor cannot be empty"), return false);
  OP_TILING_CHECK(op_paras.inputs[1].tensor.empty(),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs[1].tensor cannot be empty"), return false);
  OP_TILING_CHECK(op_paras.outputs[0].tensor.empty(),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.outputs[0].tensor cannot be empty"), return false);

  const std::vector<int64_t> shape_x1 = op_paras.inputs[0].tensor[0].shape;
  const std::vector<int64_t> shape_x2 = op_paras.inputs[1].tensor[0].shape;
  std::vector<int64_t> shape_x1_new = shape_x1;
  std::vector<int64_t> shape_x2_new = shape_x2;
  std::vector<int64_t> shape_y_new = {};

  shape_x1_new.push_back(1);
  shape_x2_new.insert(shape_x2_new.begin(), 1, 1);
  shape_y_new.push_back(shape_x1[0]);
  shape_y_new.push_back(shape_x2[0]);

  TeOpParas op_paras_tmp;
  TeOpTensor x1_tensor, x2_tensor, y_tensor;
  TeOpTensorArg x1_arg, x2_arg, y_arg;

  x1_tensor.shape = shape_x1_new;
  x2_tensor.shape = shape_x2_new;
  y_tensor.shape = shape_y_new;
  x1_arg.tensor.push_back(x1_tensor);
  x2_arg.tensor.push_back(x2_tensor);
  y_arg.tensor.push_back(y_tensor);
  op_paras_tmp.inputs.push_back(x1_arg);
  op_paras_tmp.inputs.push_back(x2_arg);
  op_paras_tmp.outputs.push_back(y_arg);

  bool ret = EletwiseTiling(op_type, op_paras_tmp, op_info, run_info);
  return ret;
}

REGISTER_OP_TILING_FUNC_BUFFERED(Ger, GerTiling);
}  // namespace optiling
