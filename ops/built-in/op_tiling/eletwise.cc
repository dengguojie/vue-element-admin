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

#include "eletwise.hpp"
#include <string>
#include <algorithm>
#include <nlohmann/json.hpp>
#include "graph/debug/ge_log.h"
#include "vector_tiling.hpp"

namespace optiling {

static std::unordered_map<int32_t, int32_t> split_factors{
        {1, 32767},
        {2, 32767},
        {4, 16383},
        {8, 8191},
};

bool ProduceShapes(const std::string &op_type,
                   std::vector<int64_t> &input_shape_x,
                   std::vector<int64_t> &input_shape_y,
                   std::vector<int64_t> &output_shape) {

  size_t shape_len_x = input_shape_x.size();
  size_t shape_len_y = input_shape_y.size();
  size_t max_len = shape_len_x > shape_len_y ? shape_len_x : shape_len_y;

  std::vector<int64_t> output(max_len, 1);
  if (shape_len_x < max_len) {
    input_shape_x.insert(input_shape_x.begin(),
                       output.begin(), output.begin() + (max_len - shape_len_x));
  } else {
    input_shape_y.insert(input_shape_y.begin(),
                       output.begin(), output.begin() + (max_len - shape_len_y));
  }

  for (size_t i = 0; i < max_len; i++) {
    if (input_shape_x[i] != input_shape_y[i] &&
        input_shape_x[i] != 1 && input_shape_y[i] != 1) {
      GE_LOGE("op [%s] : input shapes not match!", op_type.c_str());
      return false;
    }
    output[i] = input_shape_x[i] == 1 ? input_shape_y[i] : input_shape_x[i];
  }
  output_shape = std::move(output);

  return true;
}

bool RefineShapesForBroadcast(const std::string &op_type,
                              std::vector<int64_t> &input_shape_x,
                              std::vector<int64_t> &input_shape_y,
                              std::vector<int64_t> &output_shape,
                              const std::vector<std::vector<size_t>> &fusion_index) {
  if (input_shape_x.size() != input_shape_y.size()) {
    GE_LOGE("op [%s] : input shapes not match, can't broadcast!", op_type.c_str());
    return false;
  }
  size_t fusion_len = fusion_index.size();
  std::vector<int64_t> fused_shape_x(fusion_len);
  std::vector<int64_t> fused_shape_y(fusion_len);
  for (size_t i = 0; i < fusion_len; i++) {
    int64_t fused_x = 1;
    int64_t fused_y = 1;
    for (size_t j = 0; j < fusion_index[i].size(); j++) {
      fused_x *= input_shape_x[fusion_index[i][j]];
      fused_y *= input_shape_y[fusion_index[i][j]];
    }
    fused_shape_x[i] = fused_x;
    fused_shape_y[i] = fused_y;
  }
  bool ret = ProduceShapes(op_type, fused_shape_x, fused_shape_y, output_shape);
  input_shape_x = std::move(fused_shape_x);
  input_shape_y = std::move(fused_shape_y);
  return ret;
}

int32_t GetTypeSize(const std::string dtype) {
  int32_t type_size = 2;
  if (dtype == "int8" || dtype == "uint8") {
    type_size = 1;
  } else if (dtype == "float32" || dtype == "int32" || dtype == "uint32") {
    type_size = 4;
  } else if (dtype == "int64" || dtype == "uint64") {
    type_size = 8;
  } else if (dtype == "bool") {
    type_size = 1;
  }
  return type_size;
}

int32_t GetBlockTiling(std::vector<int64_t> &output_shape,
                       const int32_t core_num,
                       std::unordered_map<std::string, int32_t> &var_names,
                       int32_t &block_axis,
                       const std::string &dtype) {
  int32_t cur_core = core_num;
  size_t len = output_shape.size();
  const int32_t block_size = 32;
  int32_t type_size = GetTypeSize(dtype);
  int32_t ele_in_block = block_size / type_size;
  int32_t block_dims = 1;
  if (len == 1) {
    block_axis = 0;
    if (output_shape.empty()) {
      return -1;
    }
    int32_t block_factor = std::ceil(output_shape[0] * 1.0 / cur_core);
    block_factor = std::ceil(block_factor * 1.0 / ele_in_block) * ele_in_block;
    block_dims = std::ceil(output_shape[0] * 1.0 / block_factor);
    var_names["block_factor_0"] = block_factor;
    output_shape[0] = block_factor;
  } else {
    for (size_t i = 0; i < output_shape.size(); i++) {
      if (output_shape[i] > cur_core) {
        block_axis = i;
        int32_t block_factor = std::ceil(output_shape[i] * 1.0 / cur_core);
        block_dims *= std::ceil(output_shape[i] * 1.0 / block_factor);
        var_names["block_factor_" + std::to_string(i)] = block_factor;
        output_shape[i] = block_factor;
        break;
      } else {
        cur_core /= output_shape[i];
        block_dims *= output_shape[i];
        output_shape[i] = 1;
      }
    }
  }
  return block_dims;
}

int32_t GetUbTiling(const std::vector<int64_t> &output_shape,
                    const int32_t ub_limit,
                    std::unordered_map<std::string, int32_t> &var_names,
                    int32_t &ub_axis,
                    int32_t &block_axis,
                    const int32_t &broadcast_axis,
                    const std::string &dtype,
                    const int32_t &max_dtype) {
  int32_t limit = ub_limit;
  int32_t ub_factor = 1;
  if (output_shape.size() == 1) {
    ub_axis = 0;
    ub_factor = static_cast<int32_t>(output_shape[0]);
    if (limit < ub_factor) {
      int32_t type_size = GetTypeSize(dtype);
      int32_t ele_in_block = BLOCK_SIZE / type_size;
      int32_t ub_for_num = std::ceil(ub_factor * 1.0 / limit);
      int32_t adjust_factor = std::ceil(ub_factor * 1.0 / ub_for_num);
      int32_t align_factor = std::ceil(adjust_factor * 1.0 / ele_in_block);
      ub_factor = align_factor * ele_in_block;
    }
    ub_factor = std::min(ub_factor, split_factors[max_dtype]);
    var_names["ub_factor_0"] = ub_factor;
  } else {
    int32_t shape_len = static_cast<int32_t>(output_shape.size()) - 1;
    for (int32_t i = shape_len; i >= block_axis; i--) {
      if (output_shape[i] >= limit) {
        ub_axis = i;
        ub_factor = limit;
        var_names["ub_factor_" + std::to_string(i)] = ub_factor;
        break;
      } else {
        limit /= output_shape[i];
      }
    }
    if (ub_axis < 0) {
      ub_axis = block_axis;
      ub_factor = output_shape[block_axis];
      var_names["ub_factor_" + std::to_string(ub_axis)] = ub_factor;
    }
  }
  return ub_factor;
}

int32_t GetKey(const int32_t &ub_axis,
               const int32_t &block_axis,
               const int32_t &shape_size,
               const int32_t &ub_factor,
               const std::string &dtype) {
  int32_t key = 100000000;
  if (shape_size == 1) {
    return key;
  } else {
    key = 10000000;
    if (ub_axis == -1 && block_axis == -1) {
      return 2;
    } else {
      return key + block_axis * shape_size + ub_axis;
    }
  }
}

bool AddShapeInfo(const std::vector<int64_t> &input_shape_x,
                  const std::vector<int64_t> &input_shape_y,
                  std::unordered_map<std::string, int32_t>& var_names) {
  if (input_shape_x.size() != input_shape_y.size()) {
    return false;
  }
  string suffix_x = "_0";
  string suffix_y = "_1";
  for (size_t i = 0; i < input_shape_x.size(); i++) {
    std::string prefix = "dim_" + std::to_string(i);
    var_names[prefix + suffix_x] = input_shape_x[i];
    var_names[prefix + suffix_y] = input_shape_y[i];
  }
  return true;
}

bool GetOutputShape(const std::string &op_type,
                    const TeOpParas &op_paras,
                    const nlohmann::json &op_info,
                    bool &has_scalar,
                    std::vector<int64_t> &output_shape,
                    int32_t &broadcast_axis,
                    std::unordered_map<std::string, int32_t>& var_names) {
  size_t input_num = op_paras.inputs.size();

  if (input_num == 2) {
    bool valid_input = op_paras.inputs.size() == 2 &&
                       op_paras.inputs[0].tensor.size() > 0 &&
                       op_paras.inputs[1].tensor.size() > 0;
    if (!valid_input) {
      GE_LOGE("op [%s] : input shape error", op_type.c_str());
      return false;
    }
    std::vector<int64_t> input_shape_x = op_paras.inputs[0].tensor[0].shape;
    std::vector<int64_t> input_shape_y = op_paras.inputs[1].tensor[0].shape;

    bool ret = ProduceShapes(op_type, input_shape_x, input_shape_y, output_shape);
    ret = ret && AddShapeInfo(input_shape_x, input_shape_y, var_names);
    if (!ret) {
      return ret;
    }

    const std::string &fusion_flag = op_info["_fusion"].get<std::string>();
    if (fusion_flag != "disable") {
      const std::vector<std::vector<size_t>>& fusion_index =
              op_info["_fusion_index"].get<std::vector<std::vector<size_t>>>();

      ret = RefineShapesForBroadcast(op_type, input_shape_x, input_shape_y, output_shape, fusion_index);

      if (!ret) {
        return ret;
      }

      string suffix_z = "_2";
      for (size_t i = 0; i < output_shape.size(); i++) {
        std::string prefix = "dim_" + std::to_string(i);
        var_names[prefix + suffix_z] = output_shape[i];
      }
      // operator is scalar
      has_scalar = (input_shape_x.size() == 1 && input_shape_x[0] == 1) ||
                   (input_shape_y.size() == 1 && input_shape_y[0] == 1);
    }
    for (int32_t i = static_cast<int32_t>(input_shape_x.size()) - 1; i >= 0; i--) {
      if (input_shape_x[i] != input_shape_y[i] &&
          (input_shape_x[i] == 1 || input_shape_y[i] == 1)) {
        broadcast_axis = i;
        break;
      }
    }
  } else {
    size_t input_num = op_paras.inputs.size();
    for (size_t i = 0; i < input_num; i++) {
      string suffix = "_" + std::to_string(i);
      if (op_paras.inputs[i].tensor.empty()){
        GE_LOGE("op [%s] : inputs tensor of op_paras is empty", op_type.c_str());
        return false;
      }
      const std::vector<int64_t> &shapes = op_paras.inputs[i].tensor[0].shape;
      for (size_t j = 0; j < shapes.size(); j++) {
        std::string name = "dim_" + std::to_string(j) + suffix;
        var_names[name] = shapes[j];
      }
    }
    if (op_paras.outputs.empty() || op_paras.outputs[0].tensor.empty()) {
      GE_LOGE("op [%s] : output or output[0] tensor of op_paras is empty", op_type.c_str());
      return false;
    }
    const std::vector<int64_t> &outputs = op_paras.outputs[0].tensor[0].shape;
    int64_t fused_output = std::accumulate(outputs.begin(), outputs.end(),
                                           1, std::multiplies<int64_t>());
    output_shape.push_back(fused_output);
  }

  return true;
}

bool CalcMultiCore(const std::string &op_type,
                   const nlohmann::json &op_info,
                   const std::vector<int64_t> &output_shape,
                   bool &need_multi_core,
                   int32_t &ub_limit) {
  int32_t ub_size = op_info["_ub_size"].get<std::int32_t>();
  int32_t max_dtype = op_info["_max_dtype_bytes"].get<std::int32_t>();
  int32_t coex_quantity = op_info["_coexisting_quantity"].get<std::int32_t>();
  ub_limit = (((ub_size / coex_quantity) / BLOCK_SIZE) * BLOCK_SIZE) / max_dtype;
  int64_t output_size = std::accumulate(output_shape.begin(), output_shape.end(),
                                        1, std::multiplies<int64_t>());
  if (output_size > INT32_MAX) {
    GE_LOGE("[ERROR]op [%s] : The input shape is too large",
            op_type.c_str());
    return false;
  }
  const int32_t multi_core_threshold = 1024;
  if (output_size < multi_core_threshold) {
    need_multi_core = false;
  }
  return true;
}

bool EletwiseTiling(const std::string &op_type,
                    const TeOpParas &op_paras,
                    const nlohmann::json &op_info,
                    OpRunInfo &run_info) {
  GELOGI("op [%s]: tiling running", op_type.c_str());

  if (op_paras.outputs.size() <= 0 ||
      op_paras.outputs[0].tensor.size() <= 0) {
    GE_LOGE("op [%s] : output shape error", op_type.c_str());
    return false;
  }
  const std::string &dtype = op_paras.outputs[0].tensor[0].dtype;

  std::unordered_map<std::string, int32_t> var_names;

  bool has_scalar = false;
  std::vector<int64_t> output_shape;
  int32_t broadcast_axis = -2;
  bool ret = GetOutputShape(op_type, op_paras, op_info,
          has_scalar, output_shape, broadcast_axis, var_names);
  if (!ret) {
    return ret;
  }

  bool need_multi_core = true;
  int32_t ub_limit;
  ret = CalcMultiCore(op_type, op_info, output_shape, need_multi_core, ub_limit);
  if (!ret) {
    return ret;
  }

  int32_t key = -1;
  int32_t block_axis = -1;
  int32_t ub_axis = -1;
  int32_t block_dims = 1;
  int32_t ub_factor = 1;
  if (need_multi_core) {
    // cut block
    int32_t core_num = op_info["_core_num"].get<std::int32_t>();
    block_dims = GetBlockTiling(output_shape, core_num, var_names, block_axis, dtype);

    // cut ub
    int32_t max_dtype = op_info["_max_dtype_bytes"].get<std::int32_t>();
    ub_factor = GetUbTiling(output_shape, ub_limit, var_names,
            ub_axis, block_axis, broadcast_axis, dtype, max_dtype);
  } else {
    if (output_shape.size() == 1) {
      var_names["block_factor_0"] = output_shape[0];
      ub_factor = output_shape[0];
      var_names["ub_factor_0"] = ub_factor;
      block_axis = 0;
      ub_axis = 0;
    }
  }
  GELOGD("op [%s]: DoBlockTiling&GetUbTiling", op_type.c_str());

  key = GetKey(ub_axis, block_axis, output_shape.size(), ub_factor, dtype);


  GELOGD("op [%s] tiling key:%d", op_type.c_str(), key);
  GELOGD("op [%s] tiling block_dims:%d", op_type.c_str(), block_dims);
  GELOGD("op [%s] tiling ub_factor:%d", op_type.c_str(), ub_factor);
  GELOGD("op [%s] tiling block_axis:%d", op_type.c_str(), block_axis);
  GELOGD("op [%s] tiling ub_axis:%d", op_type.c_str(), ub_axis);

  run_info.block_dim = block_dims;
  ByteBufferPut(run_info.tiling_data, key);

  const auto& all_vars = op_info["_vars"][std::to_string(key)];
  for (const auto& var: all_vars) {
    if (var_names.count(var) == 0) {
      GE_LOGE("op [%s] : Compile info error", op_type.c_str());
      return false;
    }
    ByteBufferPut(run_info.tiling_data, var_names[var]);
  }

  return true;
}

}
