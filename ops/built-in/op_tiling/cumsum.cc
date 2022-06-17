/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file cumsum.cc
 * \brief
 */
#include <nlohmann/json.hpp>
#include "graph/debug/ge_log.h"
#include "error_util.h"
#include "error_log.h"
#include "op_log.h"
#include "op_tiling.h"
#include "op_tiling_util.h"

#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"

namespace optiling {
struct CumsumTilingParam {
  int64_t tiling_mode = 0;
  // use aicore num
  int64_t num_act_core = 0;
  // total outer loops
  int64_t num_outer_total = 0;
  // outer loop times
  int64_t ceil_outer_loop_times = 0;
  // outer loop times for total num
  int64_t floor_outer_loop_times = 0;
  // outer loop tail
  int64_t num_outer_cores_tail = 0;
  // nums to calculate in each outer loop
  int64_t nums_per_outer_loop = 0;
  // inner loop times
  int64_t inner_floor_loop_times = 0;
  // numbers to calculate in one inner loop
  int64_t num_each_inner_loop = 0;
  // numbers to calculate in last inner loop
  int64_t num_inner_last_loop = 0;
  // numbers front cores to deal
  int64_t num_per_core = 0;
  // offset num for cal back loop index
  int64_t offset_back_loop = 0;
  // shape of axis
  int64_t axis_shape = 0;
  // repeat times of inner tail loop
  int64_t inner_tail_repeat_times = 0;
  // equal shape of shape before axis
  int64_t equal_shape_before_axis = 0;
  // equal shape of shape after axis
  int64_t equal_shape_after_axis = 0;
};

void InItRunningParams(CumsumTilingParam& param) {
  param.tiling_mode = 0;
  param.num_act_core = 0;
  param.num_outer_total = 0;
  param.ceil_outer_loop_times = 0;
  param.floor_outer_loop_times = 0;
  param.num_outer_cores_tail = 0;
  param.nums_per_outer_loop = 0;
  param.inner_floor_loop_times = 0;
  param.num_each_inner_loop = 0;
  param.num_inner_last_loop = 0;
  param.num_per_core = 0;
  param.offset_back_loop = 0;
  param.axis_shape = 0;
  param.inner_tail_repeat_times = 0;
  param.equal_shape_before_axis = 0;
  param.equal_shape_after_axis = 0;
}

bool GetAxis(const ge::Operator& op_paras, int64_t& axis) {
  int64_t input_shape_size = op_paras.GetInputDesc(0).GetShape().GetDims().size();
  std::vector<int64_t> values;
  // input axis index is 1
  if (!ops::GetConstIntData(op_paras, 1, values)) {
    VECTOR_INNER_ERR_REPORT_TILIING("Cumsum", "axis not exists.");
    return false;
  }
  if (values.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING("Cumsum", "values is empty.");
    return false;
  }
  axis = values[0];

  if (axis < 0) {
    axis += input_shape_size;
  }

  if (axis < 0 || axis >= input_shape_size) {
    VECTOR_INNER_ERR_REPORT_TILIING("Cumsum", "axis [%ld] is out of range.", axis);
    return false;
  }

  OP_LOGD("Cumsum", "axis=%ld.", axis);

  return true;
}

static void PrintTilingParam(const CumsumTilingParam& param) {
  OP_LOGD("CumSumTiling",
          "tiling_mode=%ld, num_act_core=%ld, num_outer_total=%ld, "
          "ceil_outer_loop_times=%ld, floor_outer_loop_times=%ld, "
          "num_outer_cores_tail=%ld, nums_per_outer_loop=%ld, "
          "inner_floor_loop_times=%ld, num_each_inner_loop=%ld, "
          "num_inner_last_loop=%ld, num_per_core=%ld, offset_back_loop=%ld, "
          "axis_shape=%ld, inner_tail_repeat_times=%ld, "
          "equal_shape_before_axis=%ld, equal_shape_after_axis=%ld.",
          param.tiling_mode, param.num_act_core, param.num_outer_total, param.ceil_outer_loop_times,
          param.floor_outer_loop_times, param.num_outer_cores_tail, param.nums_per_outer_loop,
          param.inner_floor_loop_times, param.num_each_inner_loop, param.num_inner_last_loop, param.num_per_core,
          param.offset_back_loop, param.axis_shape, param.inner_tail_repeat_times, param.equal_shape_before_axis,
          param.equal_shape_after_axis);
}

static std::string GetVectorData(const std::vector<int64_t>& vec_data) {
  std::string dims_info = "";
  for (int64_t i = 0; i < static_cast<int64_t>(vec_data.size()); i++) {
    dims_info += " " + std::to_string(vec_data[i]);
  }

  return dims_info;
}

vector<int64_t> CalEqualShape(std::vector<int64_t> tensor_shape, int64_t axis) {
  int64_t m_num = std::accumulate(tensor_shape.begin(), tensor_shape.begin() + axis, 1, std::multiplies<int>());
  int64_t n_num = std::accumulate(tensor_shape.begin() + axis + 1, tensor_shape.end(), 1, std::multiplies<int>());
  vector<int64_t> array = {m_num, tensor_shape[axis], n_num};
  std::string array_str = GetVectorData(array);
  OP_LOGD("CumSumTiling", "CalEqualShape: %s", array_str.c_str());
  return array;
}

static void CalTilingParam(CumsumTilingParam& param, vector<int64_t> input_shape, int64_t aicore_num,
                           ge::DataType dtype, int64_t axis) {
  if (aicore_num == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("Cumsum", "aicore_num = 0 is not support");
    return;
  }
  vector<int64_t> equivalent_shape = CalEqualShape(input_shape, axis);
  int64_t m_index = 0;
  int64_t axis_index = 1;
  int64_t n_index = 2;
  // equivalent_shape [m, j, n]
  param.equal_shape_before_axis = equivalent_shape[m_index];
  param.axis_shape = equivalent_shape[axis_index];
  param.equal_shape_after_axis = equivalent_shape[n_index];

  int32_t dtype_size = ge::GetSizeByDataType(dtype);
  if (dtype_size == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("Cumsum", "dtype_size = 0 is not support");
    return;
  }
  param.num_outer_total = param.equal_shape_before_axis;
  param.nums_per_outer_loop = param.axis_shape * param.equal_shape_after_axis;

  int64_t one_block_size = 32;
  if (param.equal_shape_after_axis * dtype_size < one_block_size) {
    param.tiling_mode = 1;
  }

  if (param.num_outer_total >= aicore_num) {
    param.num_act_core = aicore_num;
  } else {
    param.num_act_core = param.num_outer_total;
  }

  param.ceil_outer_loop_times = (param.num_outer_total + aicore_num - 1) / aicore_num;
  param.floor_outer_loop_times = param.num_outer_total / aicore_num;
  param.num_outer_cores_tail = param.num_outer_total % aicore_num;
  param.num_per_core = param.equal_shape_after_axis * param.ceil_outer_loop_times * param.axis_shape;

  int64_t max_compute_size = 65280;  // a vector cal cann deal 8 block，max repeat time 255 65280=32*8*255
  param.num_each_inner_loop = max_compute_size / dtype_size;
  param.inner_floor_loop_times = param.equal_shape_after_axis / param.num_each_inner_loop;
  param.num_inner_last_loop = param.equal_shape_after_axis % param.num_each_inner_loop;

  param.offset_back_loop = aicore_num - param.num_outer_cores_tail;
  int64_t vector_byte_size = 256; // vector cal can deal 8 block
  param.inner_tail_repeat_times = (param.num_inner_last_loop * dtype_size + vector_byte_size - 1) / vector_byte_size;
}

static void SetTilingParam(const CumsumTilingParam& param, utils::OpRunInfo& run_info) {
  run_info.AddTilingData(param.tiling_mode);
  run_info.AddTilingData(param.num_act_core);
  run_info.AddTilingData(param.num_outer_total);
  run_info.AddTilingData(param.ceil_outer_loop_times);
  run_info.AddTilingData(param.floor_outer_loop_times);
  run_info.AddTilingData(param.num_outer_cores_tail);
  run_info.AddTilingData(param.nums_per_outer_loop);
  run_info.AddTilingData(param.inner_floor_loop_times);
  run_info.AddTilingData(param.num_each_inner_loop);
  run_info.AddTilingData(param.num_inner_last_loop);
  run_info.AddTilingData(param.num_per_core);
  run_info.AddTilingData(param.offset_back_loop);
  run_info.AddTilingData(param.axis_shape);
  run_info.AddTilingData(param.inner_tail_repeat_times);
  run_info.AddTilingData(param.equal_shape_before_axis);
  run_info.AddTilingData(param.equal_shape_after_axis);
}

bool CumsumTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                  utils::OpRunInfo& run_info) {
  OP_LOGD(op_type.c_str(), "CumSumTiling running, op_info:%s", op_info.dump().c_str());
  PROFILING_TILING_INIT(op_type.c_str());

  vector<int64_t> input_shape = op_paras.GetInputDesc(0).GetShape().GetDims();
  ge::DataType input_dtype = op_paras.GetInputDesc(0).GetDataType();

  int64_t core_num = op_info["vars"]["core_num"].get<std::int64_t>();

  int64_t axis = -1;
  if (!GetAxis(op_paras, axis)) {
    return false;
  }
  CumsumTilingParam param;
  InItRunningParams(param);

  CalTilingParam(param, input_shape, core_num, input_dtype, axis);
  SetTilingParam(param, run_info);
  PrintTilingParam(param);

  run_info.SetBlockDim(param.num_act_core);
  PROFILING_TILING_END();
  OP_LOGD(op_type.c_str(), "CumSumTiling run success.");
  return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED_V2(Cumsum, CumsumTiling);
}  // namespace optiling
