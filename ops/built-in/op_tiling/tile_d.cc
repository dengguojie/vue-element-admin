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
 * \file tile_d.cpp
 * \brief
 */
#include <nlohmann/json.hpp>
#include <sstream>
#include <cctype>
#include "register/op_tiling.h"
#include "error_log.h"
#include "graph/debug/ge_log.h"

namespace optiling {

const int32_t BLOCK_SIZE = 32;
const int32_t N_LAST_BROADCAST_THRESHOLD = 512;

struct CompileInfo {
  int32_t ub_size;
  int32_t max_dtype;
  int32_t coexisting_quantity;
  int32_t core_num;
  vector<int64_t> origin_multiples;
  vector<int64_t> multiples_adapt;
  vector<int64_t> unknown_shape;
};

int32_t GetDtypeSize(const std::string& dtype) {
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

bool CheckInput(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                std::string& in_dtype, std::string& out_dtype, std::vector<int64_t>& input_shape,
                struct CompileInfo& compileInfo) {
  // Check input and output info
  CHECK((!op_paras.inputs.empty() && !op_paras.inputs[0].tensor.empty()), "op [%s] : input shape cannot be empty",
        op_type.c_str());
  CHECK((!op_paras.outputs.empty() && !op_paras.outputs[0].tensor.empty()), "op [%s] : output shape cannot be empty",
        op_type.c_str());
  in_dtype = op_paras.inputs[0].tensor[0].dtype;
  out_dtype = op_paras.outputs[0].tensor[0].dtype;
  input_shape = op_paras.inputs[0].tensor[0].shape;

  // Check and get the compile info
  CHECK((op_info.count("_vars") > 0), "op [%s] : compile info not contain [_vars]", op_type.c_str());
  CHECK((op_info.count("_ub_size") > 0), "op [%s] : compile info not contain [_ub_size]", op_type.c_str());
  CHECK((op_info.count("_max_dtype_bytes") > 0), "op [%s] : compile info not contain [_max_dtype_bytes]",
        op_type.c_str());
  CHECK((op_info.count("_coexisting_quantity") > 0), "op [%s] : compile info not contain [_coexisting_quantity]",
        op_type.c_str());
  CHECK((op_info.count("_core_num") > 0), "op [%s] : compile info not contain [_core_num]", op_type.c_str());

  compileInfo.ub_size = op_info["_ub_size"].get<std::int32_t>();
  compileInfo.max_dtype = op_info["_max_dtype_bytes"].get<std::int32_t>();
  compileInfo.coexisting_quantity = op_info["_coexisting_quantity"].get<std::int32_t>();
  compileInfo.core_num = op_info["_core_num"].get<std::int32_t>();

  CHECK((op_info.count("_origin_multiples") > 0 && !op_info["_origin_multiples"].empty()),
        "op [%s] : compile info"
        "not contain[_origin_multiples]",
        op_type.c_str());
  compileInfo.origin_multiples = op_info["_origin_multiples"].get<std::vector<int64_t>>();
  //  for (uint64_t i = 0; i < op_info["_origin_multiples"].size(); i++) {
  //    compileInfo.origin_multiples.push_back(static_cast<int64_t> (op_info["_origin_multiples"][i]));
  //  }

  CHECK((op_info.count("_multiples_adapt") > 0 && !op_info["_multiples_adapt"].empty()),
        "op [%s] : compile info not"
        "contain[_multiples]",
        op_type.c_str());
  compileInfo.multiples_adapt = op_info["_multiples_adapt"].get<std::vector<int64_t>>();
  //  for (uint64_t i = 0; i < op_info["_multiples_adapt"].size(); i++) {
  //    compileInfo.multiples_adapt.push_back(static_cast<int64_t> (op_info["_multiples_adapt"][i]));
  //  }

  CHECK((op_info.count("_unknown_shape") > 0 && !op_info["_unknown_shape"].empty()),
        "op [%s]: compile info not"
        "contain[_unknown_shape]",
        op_type.c_str());
  compileInfo.unknown_shape = op_info["_unknown_shape"].get<std::vector<int64_t>>();

  return true;
}

bool GenOutputShape(const std::string& op_type, struct CompileInfo& compileInfo, std::vector<int64_t>& input_shape,
                    std::vector<int64_t>& output_shape, int32_t& broadcast_axis) {
  /*
  input:(1, -1), origin_multiples:(3, 4, 6)
  the compile input will be(1, -1)->(1, 1, -1)->(1, 1, 1, -1), and the origin_multiples will be multiples_adapt:
  (3, 4, 6, 1). So output shape will be (3, 4, 6, -1). whatever if the runtime unknown_shape is 1, the output dim  will
  be 4 not be 3.
  */
  uint64_t input_len_diff = compileInfo.origin_multiples.size() - input_shape.size();

  // input_shape dim align with origin_multiples
  input_shape.insert(input_shape.begin(), input_len_diff, 1);
  std::vector<int64_t> align_input(input_len_diff, 1);

  std::vector<int64_t> align_multiples;
  align_multiples.insert(align_multiples.begin(), compileInfo.origin_multiples.begin(),
                         compileInfo.origin_multiples.begin() + input_len_diff);

  for (uint64_t i = input_len_diff; i < compileInfo.origin_multiples.size(); i++) {
    std::vector<int64_t>::iterator iter =
        std::find(compileInfo.unknown_shape.begin(), compileInfo.unknown_shape.end(), i - input_len_diff);
    if (compileInfo.origin_multiples[i] == 1 || (input_shape[i] == 1 && iter == compileInfo.unknown_shape.end())) {
      align_input.push_back(input_shape[i]);
      align_multiples.push_back(compileInfo.origin_multiples[i]);
    } else {
      align_input.push_back(1);
      align_input.push_back(input_shape[i]);
      align_multiples.push_back(compileInfo.origin_multiples[i]);
      align_multiples.push_back(1);
    }
  }

  // Infer the output_shape and the dims
  for (uint64_t i = 0; i < align_input.size(); i++) {
    // output_shape insert the bigger element between align_input[i] and align_multiples[i]
    int64_t output_shape_i = (align_input[i] >= align_multiples[i]) ? align_input[i] : align_multiples[i];
    output_shape.push_back(output_shape_i);
  }

  // Calculate the broadcast axis by input and output
  for (uint64_t i = output_shape.size() - 1; i >= 0; i--) {
    if (align_input[i] == 1 && output_shape[i] != 1) {
      broadcast_axis = static_cast<int32_t>(i);
      break;
    }
  }

  return true;
}

void CalcTiling(struct CompileInfo& compileInfo, vector<int64_t>& output_shape, int32_t& max_available_ub,
                bool& need_multi_core) {
  max_available_ub =
      (((compileInfo.ub_size / compileInfo.coexisting_quantity) / BLOCK_SIZE) * BLOCK_SIZE) / compileInfo.max_dtype;
  int64_t output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
  const int64_t multi_core_threshold = 1024;
  if (output_size < multi_core_threshold) {
    need_multi_core = false;
  }
}

bool DoBlockTiling(std::vector<int64_t>& output_shape, int32_t& core_num, std::string& out_dtype,
                   std::unordered_map<std::string, int32_t>& var_names, int32_t& block_axis, int32_t& block_dims) {
  int32_t cur_core = core_num;
  size_t out_shape_len = output_shape.size();
  for (size_t i = 0; i < out_shape_len; i++) {
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
  return true;
}

bool DoUbTiling(const std::string& op_type, int32_t& max_available_ub, vector<int64_t>& output_shape,
                std::string& in_dtype, int32_t& block_axis, int32_t& ub_axis, int32_t& broadcast_axis,
                int32_t& ub_factor, std::unordered_map<std::string, int32_t>& var_names) {
  int32_t limit = max_available_ub;
  CHECK((!output_shape.empty()), "op [%s] : output_shape size cannot be empty", op_type.c_str());
  int32_t shape_len = static_cast<int32_t>(output_shape.size()) - 1;
  int64_t under_broadcast_shape = 1;
  int32_t ele_in_block = BLOCK_SIZE / GetDtypeSize(in_dtype);
  for (int32_t i = shape_len; i >= block_axis; i--) {
    if (broadcast_axis == i && broadcast_axis != shape_len) {
      bool is_cut_under_b = (under_broadcast_shape > N_LAST_BROADCAST_THRESHOLD ||
                             (under_broadcast_shape > (ele_in_block * 2) && output_shape[i] < ele_in_block)) &&
                            under_broadcast_shape % ele_in_block != 0;
      if (is_cut_under_b) {
        ub_axis = i + 1;
        ub_factor = output_shape[i + 1];
        var_names["ub_factor_" + std::to_string(i + 1)] = ub_factor;
        break;
      } else if (output_shape[i] > (ele_in_block * 3) && under_broadcast_shape % ele_in_block != 0) {
        ub_axis = i;
        ub_factor = std::min(static_cast<int32_t>(output_shape[i]), limit);
        var_names["ub_factor_" + std::to_string(i)] = ub_factor;
        break;
      }
    }
    if (output_shape[i] >= limit) {
      ub_axis = i;
      ub_factor = limit;
      var_names["ub_factor_" + std::to_string(i)] = ub_factor;
      break;
    } else {
      limit /= output_shape[i];
      under_broadcast_shape *= output_shape[i];
    }
  }
  if (ub_axis < 0) {
    ub_axis = block_axis;
    ub_factor = output_shape[block_axis];
    var_names["ub_factor_" + std::to_string(ub_axis)] = ub_factor;
  }
  return true;
}

void CalcKey(int32_t& key, int32_t& block_axis, int32_t& ub_axis, vector<int64_t>& output_shape) {
  int32_t base_key = 0;
  // pattern is original, and out len is bigger than 2, so only need the following branch
  if (ub_axis == -1 && block_axis == -1) {
    key = base_key;
  } else {
    key = base_key + block_axis * output_shape.size() + ub_axis + 1;
  }
}

bool WriteTilingData(const std::string& op_type, const nlohmann::json& op_info, int32_t& key, int32_t& block_dims,
                     int32_t& ub_factor, int32_t& block_axis, int32_t& ub_axis,
                     std::unordered_map<std::string, int32_t>& var_names, OpRunInfo& run_info) {
  GELOGD("op [%s] tiling key is: [%d]", op_type.c_str(), key);
  GELOGD("op [%s] tiling block_dims:%d", op_type.c_str(), block_dims);
  GELOGD("op [%s] tiling ub_factor:%d", op_type.c_str(), ub_factor);
  GELOGD("op [%s] tiling block_axis:%d", op_type.c_str(), block_axis);
  GELOGD("op [%s] tiling ub_axis:%d", op_type.c_str(), ub_axis);

  run_info.block_dim = block_dims;
  ByteBufferPut(run_info.tiling_data, key);

  const auto& all_vars = op_info["_vars"][std::to_string(key)];
  for (const auto& var : all_vars) {
    std::string _var = var;
    CHECK((var_names.count(var) > 0), "op [%s] : Compile info error", op_type.c_str());
    ByteBufferPut(run_info.tiling_data, var_names[var]);
  }
  return true;
}

bool TileDTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                 OpRunInfo& run_info) {
  using namespace ge;
  GELOGI("op[%s] TileDTiling running.", op_type.c_str());

  bool ret = true;
  bool need_multi_core = true;
  int32_t max_available_ub = 0;
  int32_t block_axis = -1;
  int32_t ub_axis = -1;
  int32_t block_dims = 1;
  int32_t ub_factor = 1;
  std::unordered_map<std::string, int32_t> var_names;

  CompileInfo compileInfo;
  std::string in_dtype;
  std::string out_dtype;
  std::vector<int64_t> input_shape;
  ret = ret && CheckInput(op_type, op_paras, op_info, in_dtype, out_dtype, input_shape, compileInfo);

  // add the variable_shape dim info
  std::string suffix = "_0";
  for (uint64_t i = 0; i < compileInfo.unknown_shape.size(); i++) {
    int64_t temp_shape = static_cast<int64_t>(compileInfo.unknown_shape[i]);
    std::string dim_name = "dim_" + std::to_string(temp_shape) + suffix;
    var_names[dim_name] = static_cast<int32_t>(input_shape[temp_shape]);
  }

  std::vector<int64_t> output_shape;
  int32_t broadcast_axis = -2;
  ret = ret && GenOutputShape(op_type, compileInfo, input_shape, output_shape, broadcast_axis);

  // Calculate the tiling params for the block_tiling and ub_tiling
  CalcTiling(compileInfo, output_shape, max_available_ub, need_multi_core);

  if (need_multi_core) {
    ret = ret && DoBlockTiling(output_shape, compileInfo.core_num, out_dtype, var_names, block_axis, block_dims);
    ret = ret && DoUbTiling(op_type, max_available_ub, output_shape, in_dtype, block_axis, ub_axis, broadcast_axis,
                            ub_factor, var_names);
  }

  int32_t key;
  if (ret) {
    CalcKey(key, block_axis, ub_axis, output_shape);
  }

  ret = ret && WriteTilingData(op_type, op_info, key, block_dims, ub_factor, block_axis, ub_axis, var_names, run_info);
  return ret;
}

// register tiling interface of the TileD op.
REGISTER_OP_TILING_FUNC(TileD, TileDTiling);
}  // namespace optiling
