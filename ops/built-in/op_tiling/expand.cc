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
 * \file expand.cc
 * \brief
 */
#include <sstream>
#include <cctype>
#include <iostream>

#include "error_log.h"
#include "graph/debug/ge_log.h"
#include "eletwise.h"
#include "vector_tiling.h"

namespace optiling {

bool ExpandTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                  OpRunInfo& run_info) {

  std::vector<int64_t> x_runtime_shape = op_paras.inputs[0].tensor[0].shape;
  std::vector<int64_t> shape_value;

  std::string shape_dtype = op_paras.inputs[1].tensor[0].dtype;
  auto pointer = std::get<0>(op_paras.const_inputs.at("shape"));
  auto size = std::get<1>(op_paras.const_inputs.at("shape"));

  uint32_t count =
    (shape_dtype == "int64") ? size / sizeof(int64_t) : (shape_dtype == "int32") ? size / sizeof(int32_t) : 0;

  if (shape_dtype == "int64") {
    auto* data = (int64_t*)pointer;
    while (count--) {
      shape_value.push_back(*data++);
    }
  }
  if (shape_dtype == "int32") {
    auto* data = (int32_t*)pointer;
    while (count--) {
      shape_value.push_back(*data++);
    }
  }

  std::vector<int64_t> broadcast_input = {};
  std::vector<int64_t> broadcast_shape = {};
  std::vector<int64_t> output_shape = {};

  // align shape for shape and input shapes
  uint64_t len_diff = shape_value.size() - x_runtime_shape.size();
  if (len_diff >= 0) {
    for (uint64_t i = 0; i < len_diff; i++){
      broadcast_input.push_back(1);
    }

    for (uint64_t i = 0; i < x_runtime_shape.size(); i++){
       broadcast_input.push_back(x_runtime_shape[i]);
    }

    for (uint64_t i = 0; i < broadcast_input.size(); i++){
       if (broadcast_input[i] > shape_value[i]){
         output_shape.push_back(broadcast_input[i]);
       } else {
         output_shape.push_back(shape_value[i]);
         }
    }
  }
  else {
    len_diff = -len_diff;
    for (uint64_t i = 0; i < len_diff; i++){
      output_shape.push_back(1);
    }

    for (uint64_t i = 0; i < shape_value.size(); i++){
      output_shape.push_back(shape_value[i]);
    }

    for (uint64_t i = 0; i < output_shape.size(); i++){
      if (broadcast_input[i] > shape_value[i]){
        output_shape[i] = broadcast_input[i];
      }
    }
  }

  TeOpParas op_paras_tmp = op_paras;
  // update new shape
  broadcast_shape.push_back(output_shape.size());

  op_paras_tmp.inputs[0].tensor[0].shape = std::move(shape_value);
  op_paras_tmp.inputs[1].tensor[0].shape = std::move(x_runtime_shape);
  op_paras_tmp.outputs[0].tensor[0].shape = std::move(output_shape);

  bool ret = EletwiseTiling(op_type, const_cast<TeOpParas&>(op_paras_tmp), op_info, run_info);
  return ret;
}

// register tiling interface of the expand op.
REGISTER_OP_TILING_FUNC_BUFFERED(Expand, ExpandTiling);
}  // namespace optiling
