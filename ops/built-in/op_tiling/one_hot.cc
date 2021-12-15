/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#include <string>
#include <math.h>
#include <nlohmann/json.hpp>
#include "op_tiling_util.h"
#include "op_log.h"
#include "error_log.h"
#include "vector_tiling_profiling.h"

namespace {
  constexpr int32_t DEPTH_SIZE = 1;
}

namespace optiling {
struct OneHotTilingParams {
  int32_t is_zero_off_value;
  int32_t not_last_core_numel;
  int32_t mode_of_cal_with_axis;
  int32_t core_used;
  int32_t numel_shape_x;
  int32_t first_dim_x;
  int32_t last_dim_x;
  int32_t numel_shape_off_value_tensor;
  int32_t last_core_numel;
  int32_t not_last_core_index;
  int32_t last_core_index;
};

void InitTilingParams(OneHotTilingParams& params) {
  OP_LOGD("InitTilingParams is running");
  params.is_zero_off_value = 0;
  params.not_last_core_numel = 0;
  params.mode_of_cal_with_axis = 0;
  params.core_used = 0;
  params.numel_shape_x = 0;
  params.first_dim_x = 0;
  params.last_dim_x = 0;
  params.numel_shape_off_value_tensor = 0;
  params.last_core_numel = 0;
  params.not_last_core_index = 0;
  params.last_core_index = 0;
}

bool MergeAxis(int32_t axis, int32_t depth, std::vector<int64_t> x_shape, std::vector<int64_t>& merged_x_shape) {
  OP_LOGD("MergeAxis is running");
  int32_t x_shape_size = x_shape.size();
  int32_t first_dim_size = 1;
  int32_t last_dim_size = 1;
  int32_t i = 0;
  int32_t axis_size = depth;
  if (axis == 0) {
    while (i < x_shape_size) {
      last_dim_size *= x_shape[i];
      ++i;
    }
    merged_x_shape.push_back(axis_size);
    merged_x_shape.push_back(last_dim_size);
    return true;
  } else if (axis > 0 && axis < x_shape_size) {
    while (i < axis) {
      first_dim_size = first_dim_size * x_shape[i];
      ++i;
    }
    i = axis;
    merged_x_shape.push_back(first_dim_size);
    merged_x_shape.push_back(axis_size);
    while (i < x_shape_size) {
      last_dim_size *= x_shape[i];
      ++i;
    }
    merged_x_shape.push_back(last_dim_size);
    return true;
  } else if (axis == -1 || axis == x_shape_size) {
    while (i < x_shape_size) {
      first_dim_size *= x_shape[i];
      ++i;
    }
    merged_x_shape.push_back(first_dim_size);
    merged_x_shape.push_back(axis_size);
    return true;
  }
  return false;
}

static int32_t CeilDiv(int32_t first_value, int32_t second_value) {
  OP_TILING_CHECK(second_value == 0, VECTOR_INNER_ERR_REPORT_TILIING("one_hot", "second_value = 0 is not support"),
                  return -1);
  int32_t result = 1;
  result = (first_value + second_value - 1) / second_value;
  return result;
}

int32_t CalTensorNumel(std::vector<int64_t> tensor_shape) {
  int32_t numel = std::accumulate(tensor_shape.begin(), tensor_shape.end(), 1, std::multiplies<int>());
  return numel;
}

int32_t get_core_num(int32_t numel, std::vector<int64_t> x_shape, int32_t depth, int32_t axis, int32_t core_num) {
  OP_TILING_CHECK(core_num == 0, VECTOR_INNER_ERR_REPORT_TILIING("one_hot", "core_num = 0 is not support"), return -1);
  auto ele_per_core = (numel - 1) / core_num + 1;
  auto core_used = (numel - 1) / ele_per_core + 1;
  auto numel_x = CalTensorNumel(x_shape);
  std::vector<int64_t> merged_x_shape;
  MergeAxis(axis, depth, x_shape, merged_x_shape);
  int32_t block = 16;
  int32_t x_shape_size = x_shape.size();

  if (axis == 0) {
    auto per_core_index = CeilDiv(depth, core_used);
    while ((per_core_index * numel_x < block) && (core_used > 1)) {
      core_used -= 1;
      per_core_index = CeilDiv(depth, core_used);
    }
    core_used = CeilDiv(depth, per_core_index);
  } else if (axis == x_shape_size || axis == -1) {
    auto per_core_numel = CeilDiv(numel_x, core_used);
    while (per_core_numel * depth < block && core_used > 1) {
      core_used -= 1;
      per_core_numel = CeilDiv(numel_x, core_used);
    }
    core_used = CeilDiv(numel_x, per_core_numel);
  } else {
    auto first_dim_x = merged_x_shape[0];
    auto last_dim_x = merged_x_shape[2];
    auto per_core_numel = CeilDiv(first_dim_x, core_used);
    while (per_core_numel * last_dim_x * depth < block && core_used > 1) {
      core_used -= 1;
      per_core_numel = CeilDiv(first_dim_x, core_used);
    }
    core_used = CeilDiv(first_dim_x, per_core_numel);
  }
  return core_used;
}

// Calcuted acording to the min size of radio of ub and tensor
#define RADIO_UB_SIZE_MIN 19792
// Calcuted acording to the max size of radio of ub and tensor
#define RADIO_UB_SIZE_MAX 39584

int32_t CalTilingMode(std::vector<int64_t> x_shape, int32_t depth, int32_t axis, int32_t core_num) {
  OP_LOGD("CalTilingMode is running");
  int32_t x_shape_size = x_shape.size();
  int32_t x_numel = CalTensorNumel(x_shape);
  int32_t tiling_mode = 0;

  if (axis == -1 || axis == x_shape_size) {
    auto core_used = get_core_num(x_numel, x_shape, depth, axis, core_num);
    auto per_core_numel = CeilDiv(x_numel, core_used);
    if (per_core_numel <= RADIO_UB_SIZE_MIN && per_core_numel * depth <= RADIO_UB_SIZE_MAX) {
      tiling_mode = 1;
    } else if (per_core_numel <= RADIO_UB_SIZE_MIN && per_core_numel * depth > RADIO_UB_SIZE_MAX && depth >= DEPTH_SIZE &&
               depth <= RADIO_UB_SIZE_MAX) {
      tiling_mode = 2;
    } else if (per_core_numel <= RADIO_UB_SIZE_MIN && depth > RADIO_UB_SIZE_MAX) {
      tiling_mode = 3;
    } else if (per_core_numel > RADIO_UB_SIZE_MIN && depth >= DEPTH_SIZE && depth <= RADIO_UB_SIZE_MAX) {
      tiling_mode = 4;
    } else {
      tiling_mode = 5;
    }
  } else if (axis == 0) {
    if (x_numel <= RADIO_UB_SIZE_MIN && x_numel * depth <= RADIO_UB_SIZE_MAX) {
      tiling_mode = 6;
    } else if (x_numel <= RADIO_UB_SIZE_MIN && x_numel * depth > RADIO_UB_SIZE_MAX) {
      tiling_mode = 7;
    } else if (x_numel <= RADIO_UB_SIZE_MAX && x_numel > RADIO_UB_SIZE_MIN && x_numel * depth > RADIO_UB_SIZE_MAX) {
      tiling_mode = 8;
    } else {
      tiling_mode = 9;
    }
  } else if (axis < x_shape_size && axis > 0) {
    std::vector<int64_t> merged_x_shape;
    MergeAxis(axis, depth, x_shape, merged_x_shape);
    auto first_dim_x = merged_x_shape[0];
    auto core_used = get_core_num(first_dim_x, x_shape, depth, axis, core_num);
    auto per_core_numel = CeilDiv(first_dim_x, core_used);
    auto last_dim_x = merged_x_shape[2];
    if (per_core_numel * last_dim_x <= RADIO_UB_SIZE_MIN && per_core_numel * last_dim_x * depth <= RADIO_UB_SIZE_MAX) {
      tiling_mode = 10;
    } else if (per_core_numel * last_dim_x <= RADIO_UB_SIZE_MIN &&
               per_core_numel * last_dim_x * depth > RADIO_UB_SIZE_MAX && last_dim_x >= 1 &&
               last_dim_x <= RADIO_UB_SIZE_MAX) {
      tiling_mode = 11;
    } else if (per_core_numel * last_dim_x > RADIO_UB_SIZE_MIN &&
               per_core_numel * last_dim_x * depth > RADIO_UB_SIZE_MAX && last_dim_x >= 1 &&
               last_dim_x <= RADIO_UB_SIZE_MAX) {
      tiling_mode = 12;
    } else {
      tiling_mode = 13;
    }
  }

  return tiling_mode;
}

void CalCoreInfo(OneHotTilingParams& tiling_params, int32_t core_num, int32_t depth, int32_t axis,
                 std::vector<int64_t> x_shape) {
  OP_LOGD("CalCoreInfo is running");
  std::vector<int64_t> merged_x_shape;
  MergeAxis(axis, depth, x_shape, merged_x_shape);

  int32_t per_core_index = 0;
  int32_t last_core_index = 0;
  int32_t core_used = 0;
  int32_t per_core_numel = 0;
  int32_t last_core_numel = 0;
  int32_t numel_x = 0;
  int32_t x_shape_size = x_shape.size();

  numel_x = CalTensorNumel(x_shape);

  if (axis == 0) {
    core_used = get_core_num(depth, x_shape, depth, axis, core_num);
    per_core_index = CeilDiv(depth, core_used);
    last_core_index = depth - (core_used - 1) * per_core_index;
  } else if (axis == -1 || axis == x_shape_size) {
    core_used = get_core_num(numel_x, x_shape, depth, axis, core_num);
    per_core_numel = CeilDiv(numel_x, core_used);
    last_core_numel = numel_x - (core_used - 1) * per_core_numel;
  } else {
    auto first_dim_x = merged_x_shape[0];
    core_used = get_core_num(first_dim_x, x_shape, depth, axis, core_num);
    per_core_numel = CeilDiv(first_dim_x, core_used);
    last_core_numel = first_dim_x - (core_used - 1) * per_core_numel;
  }

  tiling_params.last_core_index = last_core_index;
  tiling_params.not_last_core_index = per_core_index;
  tiling_params.not_last_core_numel = per_core_numel;
  tiling_params.core_used = core_used;
  tiling_params.last_core_numel = last_core_numel;
}

void CalRunningInfo(OneHotTilingParams& tiling_params, int32_t core_num, int32_t depth, int32_t axis,
                    std::vector<int64_t> x_shape) {
  OP_LOGD("CalRunningInfo is running");
  std::vector<int64_t> merged_x_shape;
  MergeAxis(axis, depth, x_shape, merged_x_shape);

  int32_t first_dim_x = 1;
  int32_t last_dim_x = 1;
  int32_t numel_x = 1;
  int32_t numel_merged_x = 1;
  int32_t x_shape_size = x_shape.size();

  numel_x = CalTensorNumel(x_shape);
  numel_merged_x = CalTensorNumel(merged_x_shape);
  tiling_params.numel_shape_x = numel_x;
  if (axis > 0 && axis < x_shape_size) {
    first_dim_x = merged_x_shape[0];
    last_dim_x = merged_x_shape[2];
  } else if (axis == x_shape_size || axis == -1) {
    first_dim_x = merged_x_shape[0];
  } else {
    last_dim_x = merged_x_shape[1];
  }
  tiling_params.first_dim_x = first_dim_x;
  tiling_params.last_dim_x = last_dim_x;
  tiling_params.numel_shape_off_value_tensor = numel_merged_x;
  tiling_params.mode_of_cal_with_axis = CalTilingMode(x_shape, depth, axis, core_num);
  CalCoreInfo(tiling_params, core_num, depth, axis, x_shape);
}

void SetRunningInfo(const OneHotTilingParams& tiling_params, utils::OpRunInfo& run_info) {
  OP_LOGD("SetRunningInfo is running");
  run_info.AddTilingData(tiling_params.is_zero_off_value);
  run_info.AddTilingData(tiling_params.not_last_core_numel);
  run_info.AddTilingData(tiling_params.mode_of_cal_with_axis);
  run_info.AddTilingData(tiling_params.core_used);
  run_info.AddTilingData(tiling_params.numel_shape_x);
  run_info.AddTilingData(tiling_params.first_dim_x);
  run_info.AddTilingData(tiling_params.last_dim_x);
  run_info.AddTilingData(tiling_params.numel_shape_off_value_tensor);
  run_info.AddTilingData(tiling_params.last_core_numel);
  run_info.AddTilingData(tiling_params.not_last_core_index);
  run_info.AddTilingData(tiling_params.last_core_index);
}

void PrintTilingParams(const OneHotTilingParams& tiling_params) {
  OP_LOGD("OneHotTiling", "PrintTilingParams is running");
  OP_LOGD("OneHotTiling", "is_zero_off_value=%d.", tiling_params.is_zero_off_value);
  OP_LOGD("OneHotTiling", "not_last_core_numel=%d.", tiling_params.not_last_core_numel);
  OP_LOGD("OneHotTiling", "mode_of_cal_with_axis=%d.", tiling_params.mode_of_cal_with_axis);
  OP_LOGD("OneHotTiling", "core_used=%d.", tiling_params.core_used);
  OP_LOGD("OneHotTiling", "numel_shape_x=%d.", tiling_params.numel_shape_x);
  OP_LOGD("OneHotTiling", "first_dim_x=%d.", tiling_params.first_dim_x);
  OP_LOGD("OneHotTiling", "last_dim_x=%d.", tiling_params.last_dim_x);
  OP_LOGD("OneHotTiling", "numel_shape_off_value_tensor=%d.", tiling_params.numel_shape_off_value_tensor);
  OP_LOGD("OneHotTiling", "last_core_numel=%d.", tiling_params.last_core_numel);
  OP_LOGD("OneHotTiling", "not_last_core_index=%d.", tiling_params.not_last_core_index);
  OP_LOGD("OneHotTiling", "last_core_index=%d.", tiling_params.last_core_index);
}

static const std::vector<std::string> COMPILE_INFO_KEY = {"core_num", "axis"};

bool OneHotTiling(const std::string& op_type, const ge::Operator& op_paras, const std::vector<int64_t>& op_info,
                  utils::OpRunInfo& run_info) {
  OP_LOGD("OneHotTiling is running");
  PROFILING_TILING_INIT(op_type.c_str());
  using namespace ge;
  OP_TILING_CHECK(COMPILE_INFO_KEY.size() != op_info.size(),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "parse op_info failed."), return false);
  int32_t core_num = static_cast<int32_t>(op_info[0]);
  int32_t axis = static_cast<int32_t>(op_info[1]);

  OneHotTilingParams tiling_params;
  InitTilingParams(tiling_params);
  PROFILING_TILING_AFTER_GET_SHAPE_REG();

  auto operator_info = OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get op_info failed."),
                  return false);

  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_desc failed."),
                  return false);
  std::vector<int64_t> x_shape = input_desc->MutableShape().GetDims();

  int32_t depth;
  // input depth index is 1
  OP_TILING_CHECK(!ops::GetConstInt(op_paras, 1, depth),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get input depth Const value error."), return false);
  OP_LOGD("depth=%d.", depth);
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();
  CalRunningInfo(tiling_params, core_num, depth, axis, x_shape);
  SetRunningInfo(tiling_params, run_info);
  PrintTilingParams(tiling_params);
  PROFILING_TILING_AFTER_CALCU_TILING_REG();

  run_info.SetBlockDim(tiling_params.core_used);
  PROFILING_TILING_END();
  return true;
}
REGISTER_OP_TILING_V3_WITH_VECTOR(OneHot, OneHotTiling, COMPILE_INFO_KEY, NO_OPTIONAL_VALUE);
}  // namespace optiling.
