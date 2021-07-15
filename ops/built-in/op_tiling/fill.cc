/*
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <algorithm>
#include <unordered_map>
#include "error_log.h"
#include "vector_tiling.h"

namespace optiling {

bool FillTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                OpRunInfo& run_info) {
  std::vector<int64_t> shapes;

  OP_TILING_CHECK(op_paras.inputs.empty(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs cannot be empty"),
                  return false);
  OP_TILING_CHECK(op_paras.inputs[0].tensor.empty(),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs[0].tensor cannot be empty"), return false);

  std::string dims_dtype = op_paras.inputs[0].tensor[0].dtype;
  auto pointer = std::get<0>(op_paras.const_inputs.at("dims"));
  auto size = std::get<1>(op_paras.const_inputs.at("dims"));
  uint32_t count =
      (dims_dtype == "int64") ? size / sizeof(int64_t) : (dims_dtype == "int32") ? size / sizeof(int32_t) : 0;
  OP_TILING_CHECK(!count, VECTOR_INNER_ERR_REPORT_TILIING(op_type, " input dims shape cannot be empty"), return false);

  if (dims_dtype == "int64") {
    auto* data = (int64_t*)pointer;
    while (count--) {
      shapes.push_back(*data++);
    }
  }
  if (dims_dtype == "int32") {
    auto* data = (int32_t*)pointer;
    while (count--) {
      shapes.push_back(*data++);
    }
  }

  int64_t fused_output = std::accumulate(shapes.begin(), shapes.end(), 1ll, std::multiplies<int64_t>());

  TeOpParas op_paras_tmp = std::move(op_paras);
  OP_TILING_CHECK(op_paras_tmp.inputs.empty(),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras_tmp.inputs cannot be empty"), return false);
  OP_TILING_CHECK(op_paras_tmp.inputs[0].tensor.empty(),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras_tmp.inputs[0].tensor cannot be empty"),
                  return false);
  op_paras_tmp.inputs[0].tensor[0].shape = {fused_output};

  OP_TILING_CHECK(op_paras_tmp.outputs.empty(),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras_tmp.outputs cannot be empty"), return false);
  OP_TILING_CHECK(op_paras_tmp.outputs[0].tensor.empty(),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras_tmp.outputs[0].tensor cannot be empty"),
                  return false);
  op_paras_tmp.outputs[0].tensor[0].shape.clear();
  op_paras_tmp.outputs[0].tensor[0].shape.push_back(fused_output);
  GELOGD("fill get dims fused_output is [%d], and fuse shape size is [%d]", fused_output,
         op_paras_tmp.outputs[0].tensor[0].shape.size());

  bool ret = EletwiseTiling(op_type, const_cast<TeOpParas&>(op_paras_tmp), op_info, run_info);
  return ret;
}

REGISTER_OP_TILING_FUNC_BUFFERED(Fill, FillTiling);
}  // namespace optiling
