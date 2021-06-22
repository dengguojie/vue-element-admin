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
#include "reduce_tiling.h"
#include "../fusion_pass/common/fp16_t.hpp"
#include "error_log.h"

namespace optiling {

bool IsInVector(const std::vector<int32_t>& input, int32_t value) {
  for (uint32_t i = 0; i < input.size(); i++) {
    if (input[i] == value) {
      return true;
    }
  }
  return false;
}

bool IsPureMove(const std::vector<int64_t>& input_shape, const std::vector<int32_t>& reduce_axis) {
  for (uint32_t i = 0; i < input_shape.size(); i++) {
    if (IsInVector(reduce_axis, i)) {
      if (input_shape[i] != 1) {
        return false;
      }
    }
  }
  return true;
}

bool GetInputShape(const std::string& op_type, const TeOpParas& op_paras,
                   const nlohmann::json& op_info, std::vector<int64_t>& input_shape_ori) {
  int idx = op_info.count("_idx_before_reduce") > 0 ?
            op_info.at("_idx_before_reduce").get<int>() : 0;
  // CHECK INPUT
  V_OP_TILING_CHECK(!op_paras.inputs.empty(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "inputs cannot be empty"),
                    return false);
  V_OP_TILING_CHECK(!(op_paras.inputs.size() <= uint(idx) || idx < 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "idx is invalid index for inputs"),
                    return false);
  V_OP_TILING_CHECK(!op_paras.inputs[uint(idx)].tensor.empty(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "tensor cannot be empty"),
                    return false);
  input_shape_ori = op_paras.inputs[idx].tensor[0].shape;
  return true;
}

bool GetReduceAxis(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                   const std::vector<int64_t>& input_shape_ori, std::vector<int32_t>& reduce_axis_ori) {
  // Get ori reduce aixs
  if (op_paras.const_inputs.find("axes") != op_paras.const_inputs.end() || op_info.count("axes_idx") > 0) {
    // axis_unknown
    TeConstTensorData reduce_axis_info;
    if (op_info.count("axes_idx") > 0) {
      int axes_idx = op_info.at("axes_idx").get<int>();
      std::string axes_name = op_paras.inputs[axes_idx].tensor[0].name;
      for (auto &item: op_paras.const_inputs) {
          const std::string key = item.first;
      }
      reduce_axis_info = op_paras.const_inputs.at(axes_name);
    } else {
      reduce_axis_info = op_paras.const_inputs.at("axes");
    }
    auto size = std::get<1>(reduce_axis_info);
    ge::DataType axis_type = std::get<2>(reduce_axis_info).GetTensorDesc().GetDataType();
    V_OP_TILING_CHECK(!(axis_type != ge::DT_INT32 && axis_type != ge::DT_INT64),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "axis_type is not belong to [int32, int64]"),
                      return false);

    if (axis_type == ge::DT_INT32) {
      int count = size / sizeof(int32_t);
      const int32_t* data_addr = reinterpret_cast<const int32_t*>(std::get<0>(reduce_axis_info));
      reduce_axis_ori.resize(count);
      for (int i = 0; i < count; i++) {
        reduce_axis_ori[i] = *data_addr;
        data_addr++;
      }
    } else {
      int count = size / sizeof(int64_t);
      const int64_t* data_addr = reinterpret_cast<const int64_t*>(std::get<0>(reduce_axis_info));
      reduce_axis_ori.resize(count);
      for (int i = 0; i < count; i++) {
        reduce_axis_ori[i] = (int32_t)*data_addr;
        data_addr++;
      }
    }
    // clear reduce_axis_ori when shape of input axis is (0, )
    if (size == 0) {
      reduce_axis_ori.clear();
    }
  } else {
    // axis_known
    reduce_axis_ori = op_info.at("_ori_axis").get<std::vector<int32_t>>();
  }

  // Convert reduce axis (-1 -> length+1)
  // CHECK AXIS VALUE
  int32_t max_value = int32_t(input_shape_ori.size());
  int32_t min_value = -1 * max_value;
  for (size_t i = 0; i < reduce_axis_ori.size(); i++) {
    bool is_illegal_case = reduce_axis_ori[i] >= max_value || reduce_axis_ori[i] < min_value;
    if (is_illegal_case) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "value of axis is illegal.");
      return false;
    }
    if (reduce_axis_ori[i] < 0) {
      reduce_axis_ori[i] = input_shape_ori.size() + reduce_axis_ori[i];
    }
  }
  return true;
}

bool ReduceMeanTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                      OpRunInfo& run_info) {
  bool ret = ReduceTiling(op_type, op_paras, op_info, run_info);
  std::vector<int64_t> input_shape{std::vector<int64_t>(10, 0)};
  std::vector<int32_t> reduce_axis{std::vector<int32_t>(10, 0)};
  ret = ret && GetInputShape(op_type, op_paras, op_info, input_shape);
  ret = ret && GetReduceAxis(op_type, op_paras, op_info, input_shape, reduce_axis);

  // reduce_mean_cof is not required when handling pure dma_copy case
  if (IsPureMove(input_shape, reduce_axis)) {
    return ret;
  }

  float reduce_mean_cof = 1.0;
  if (op_info.count("reduce_mean_cof_dtype") > 0) {
    const std::string& reduce_mean_cof_dtype = op_info.at("reduce_mean_cof_dtype").get<std::string>();
    if (reduce_mean_cof_dtype == "float32") {
      for (uint32_t i = 0; i < input_shape.size(); i++) {
        if (IsInVector(reduce_axis, i)) {
          reduce_mean_cof = reduce_mean_cof / input_shape[i];
        }
      }
      ByteBufferPut(run_info.tiling_data, (float)reduce_mean_cof);
      OP_LOGD(op_type.c_str(), "reduce mean cof:%f", reduce_mean_cof);
    } else if (reduce_mean_cof_dtype == "float16") {
      for (uint32_t i = 0; i < input_shape.size(); i++) {
        if (IsInVector(reduce_axis, i)) {
          reduce_mean_cof = reduce_mean_cof / input_shape[i];
        }
      }
      fe::fp16_t reduce_mean_cof_fp16;
      reduce_mean_cof_fp16 = reduce_mean_cof;
      ByteBufferPut(run_info.tiling_data, (fe::fp16_t)reduce_mean_cof_fp16);
      ByteBufferPut(run_info.tiling_data, (uint16_t)0);
      OP_LOGD(op_type.c_str(), "reduce mean cof:%f", reduce_mean_cof);
    }
  }

  return ret;
}

REGISTER_OP_TILING_FUNC_BUFFERED(ReduceMean, ReduceMeanTiling);
REGISTER_OP_TILING_FUNC_BUFFERED(ReduceMeanD, ReduceMeanTiling);
}  // namespace optiling
