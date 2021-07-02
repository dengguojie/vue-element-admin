/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file arg_max_v2.cc
 * \brief dynamic shape tiling of arg_max_v2
 */
#include <cmath>
#include <map>
#include <nlohmann/json.hpp>

#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "op_tiling.h"
#include "error_log.h"

namespace optiling {
using namespace ge;
using namespace std;

const int32_t MAX_SEGMENT_LEN = 2048 * 4;

struct TilingParam {
  int32_t tiling_mode;
  int32_t first_dim_size;
  int32_t axis_size;
  int32_t last_dim_size;
  int32_t act_core_num;
  int32_t one_core_ele;
  int32_t last_core_ele;
  // for arg last dim
  int32_t align_num;
  int32_t axis_size_one_time;
  int32_t loop_times;
  int32_t tail_size;
  // for arg last dim and not last dim
  int32_t one_core_segment_loop;
  int32_t one_core_segment_tail;
  int32_t one_core_segment_tail_data;
  int32_t one_core_offset;
  int32_t last_core_segment_loop;
  int32_t last_core_segment_tail;
  int32_t last_core_segment_tail_data;
  int32_t last_core_offset;
};

static void SetTilingParam(const TilingParam& param, OpRunInfo& run_info) {
  ByteBufferPut(run_info.tiling_data, param.tiling_mode);
  ByteBufferPut(run_info.tiling_data, param.first_dim_size);
  ByteBufferPut(run_info.tiling_data, param.axis_size);
  ByteBufferPut(run_info.tiling_data, param.last_dim_size);
  ByteBufferPut(run_info.tiling_data, param.act_core_num);
  ByteBufferPut(run_info.tiling_data, param.one_core_ele);
  ByteBufferPut(run_info.tiling_data, param.last_core_ele);
  ByteBufferPut(run_info.tiling_data, param.align_num);
  ByteBufferPut(run_info.tiling_data, param.axis_size_one_time);
  ByteBufferPut(run_info.tiling_data, param.loop_times);
  ByteBufferPut(run_info.tiling_data, param.tail_size);
  ByteBufferPut(run_info.tiling_data, param.one_core_segment_loop);
  ByteBufferPut(run_info.tiling_data, param.one_core_segment_tail);
  ByteBufferPut(run_info.tiling_data, param.one_core_segment_tail_data);
  ByteBufferPut(run_info.tiling_data, param.one_core_offset);
  ByteBufferPut(run_info.tiling_data, param.last_core_segment_loop);
  ByteBufferPut(run_info.tiling_data, param.last_core_segment_tail);
  ByteBufferPut(run_info.tiling_data, param.last_core_segment_tail_data);
  ByteBufferPut(run_info.tiling_data, param.last_core_offset);
}

static void PrintParam(const TilingParam& param) {
  OP_LOGD("ArgOpsTiling",
          "(tiling_mode,first_dim_size,axis_size,last_dim_size,act_core_num,one_core_ele,"
          "last_core_ele,align_num,axis_size_one_time,loop_times,tail_size,one_core_segment_loop,"
          "one_core_segment_tail,one_core_segment_tail_data,one_core_offset,last_core_segment_loop,"
          "last_core_segment_tail,last_core_segment_tail_data,last_core_offset):"
          "(%d,%d,%d,% d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d)",
          param.tiling_mode, param.first_dim_size, param.axis_size, param.last_dim_size, param.act_core_num,
          param.one_core_ele, param.last_core_ele, param.align_num, param.axis_size_one_time, param.loop_times,
          param.tail_size, param.one_core_segment_loop, param.one_core_segment_tail, param.one_core_segment_tail_data,
          param.one_core_offset, param.last_core_segment_loop, param.last_core_segment_tail,
          param.last_core_segment_tail_data, param.last_core_offset);
}

static int32_t GetCeilInt(int32_t value1, int32_t value2) {
  return (int32_t)(value1 + value2 - 1) / value2;
}

static int GetAlignNum(int32_t dim_size, int32_t align_size) {
  int32_t align_num = align_size;
  for (int32_t i = 1; i <= align_size; i++) {
    align_num = i;
    if (dim_size * i % align_size == 0)
      break;
  }
  return align_num;
}

static void CalTilingParam(TilingParam& param, const vector<int64_t>& input_shape, const string& dtype, int32_t axis,
                           int32_t ub_ele, int32_t core_num) {
  // set block and vector
  int32_t data_each_block = 8;
  int32_t data_each_vector = 64;
  int32_t segment = MAX_SEGMENT_LEN;  // for arg at last dim
  if (dtype == "float16") {
    data_each_block = 16;
    data_each_vector = 128;
    segment = MAX_SEGMENT_LEN * 3;  // for arg at last dim
  }

  int32_t first_dim_size = 1;
  int64_t input_dims = input_shape.size();
  if (axis == input_dims - 1) {
    // calc first dim size and axis size
    int32_t i = 0;
    while (i < input_dims - 1) {
      first_dim_size = first_dim_size * input_shape[i];
      i++;
    }
    param.first_dim_size = first_dim_size;
    param.axis_size = input_shape[axis];
    param.last_dim_size = param.axis_size;

    // calc core number at first dim size
    int32_t core_number = core_num;
    if (param.first_dim_size < 8)
      core_number = 1;
    int32_t core_segment = GetCeilInt(param.first_dim_size, core_number);
    core_segment = GetCeilInt(core_segment, 8) * 8;

    param.one_core_ele = core_segment;
    param.act_core_num = GetCeilInt(param.first_dim_size, core_segment);
    if (param.first_dim_size % param.one_core_ele == 0) {
      param.last_core_ele = param.one_core_ele;
    } else {
      param.last_core_ele = param.first_dim_size % param.one_core_ele;
    }

    // tiling at first dim
    param.one_core_segment_loop = param.one_core_ele / MAX_SEGMENT_LEN;
    param.one_core_segment_tail = param.one_core_ele % MAX_SEGMENT_LEN;
    param.last_core_segment_loop = param.last_core_ele / MAX_SEGMENT_LEN;
    param.last_core_segment_tail = param.last_core_ele % MAX_SEGMENT_LEN;

    // select branch
    if (dtype == "float16" && param.axis_size < MAX_SEGMENT_LEN) {
      param.tiling_mode = 0;  // compute_argxxx_last_axis_fp16_copy_one_time
      if (param.axis_size <= data_each_vector * 2) {
        param.align_num = GetAlignNum(param.axis_size, data_each_block);
        // calc axis size one time: the size one move can copy to ub at core_segment
        int32_t vector_size = 0;
        int32_t axis_size_one_time = 0;
        if (param.axis_size >= data_each_vector) {
          vector_size = GetCeilInt(param.axis_size, data_each_vector) * data_each_vector;
          axis_size_one_time = (int32_t)(segment / vector_size);
          axis_size_one_time = axis_size_one_time * param.align_num;
        } else {
          vector_size = GetCeilInt(param.axis_size, data_each_block) * data_each_block;
          axis_size_one_time = (int32_t)(segment / vector_size);
          axis_size_one_time = ((axis_size_one_time > 248) || (param.align_num == 1))
                                   ? 248
                                   : axis_size_one_time;  // 248: (int)(255 / 8) * 8
          axis_size_one_time = (int32_t)(axis_size_one_time * param.align_num / 248) * 248;
        }
        // change tiling at first dim
        param.axis_size_one_time = axis_size_one_time;
        param.one_core_segment_loop = param.one_core_ele / axis_size_one_time;
        param.one_core_segment_tail = param.one_core_ele % axis_size_one_time;
        param.last_core_segment_loop = param.last_core_ele / axis_size_one_time;
        param.last_core_segment_tail = param.last_core_ele % axis_size_one_time;
        if (param.axis_size >= data_each_vector) {
          param.tiling_mode = 1;  // compute_argxxx_last_axis_fp16_more_vector
        } else {
          param.tiling_mode = 2;  // compute_argxxx_last_axis_fp16_less_vector
        }
      }
    } else {
      // tiling at last dim
      param.loop_times = param.axis_size / segment;
      param.tail_size = param.axis_size % segment;
      if (dtype == "float32") {
        param.tiling_mode = 4;  // compute_argxxx_last_axis_fp32
        if (param.loop_times == 0) {
          param.tiling_mode = 12;  // compute_argxxx_last_axis_fp32 for zero loop
        }
      } else {
        param.tiling_mode = 3;  // compute_argxxx_last_axis_fp16
      }
    }
  } else {
    // calc first dim size and axis size and last dim size
    int32_t i = 0;
    int32_t last_dim_size = 1;
    while (i < axis) {
      first_dim_size = first_dim_size * input_shape[i];
      i++;
    }
    i = axis + 1;
    while (i < input_dims) {
      last_dim_size = last_dim_size * input_shape[i];
      i++;
    }
    param.first_dim_size = first_dim_size;
    param.axis_size = input_shape[axis];
    param.last_dim_size = last_dim_size;

    if ((param.first_dim_size >= core_num) || (param.first_dim_size >= (param.last_dim_size / data_each_vector))) {
      // calc core number at first_dim
      param.one_core_ele = GetCeilInt(param.first_dim_size, core_num);
      param.act_core_num = param.first_dim_size / param.one_core_ele;
      if (param.first_dim_size % param.one_core_ele != 0) {
        param.act_core_num = param.act_core_num + 1;
      }
      param.last_core_ele = param.first_dim_size - (param.act_core_num - 1) * param.one_core_ele;
      // calc tiling info
      param.one_core_segment_loop = param.last_dim_size / MAX_SEGMENT_LEN;
      param.one_core_segment_tail = param.last_dim_size % MAX_SEGMENT_LEN;
      param.one_core_segment_tail_data = param.one_core_segment_tail;
      if (param.one_core_segment_tail % data_each_block != 0 && param.one_core_segment_loop != 0) {
        param.one_core_segment_tail_data = GetCeilInt(param.one_core_segment_tail, data_each_block) * data_each_block;
        param.one_core_offset = param.one_core_segment_tail - param.one_core_segment_tail_data;
      }
      param.last_core_segment_loop = param.one_core_segment_loop;
      param.last_core_segment_tail = param.one_core_segment_tail;
      param.last_core_segment_tail_data = param.one_core_segment_tail_data;
      param.last_core_offset = param.one_core_offset;

      if (param.last_dim_size < data_each_block) {
        param.act_core_num = 1;
        param.one_core_ele = param.first_dim_size;
        param.last_core_ele = param.first_dim_size;
      }
      // select branch
      if (dtype == "float32") {
        param.tiling_mode = 5;  // do_not_last
      } else {
        param.tiling_mode = 10;  // do_not_last
        if (param.axis_size <= 2048) {
          param.tiling_mode = 6;  // do_not_last_fp16_default
          if (param.last_dim_size % data_each_block == 0 && param.last_dim_size / data_each_block <= 4 * 8) {
            param.tiling_mode = 7;  // do_not_last_fp16_aglin
          }
        }
      }
    } else {
      // calc core number at last_dim
      int32_t core_number = core_num;
      if (param.last_dim_size < data_each_vector)
        core_number = 1;
      int32_t core_segment = GetCeilInt(param.last_dim_size, core_number);
      core_segment = GetCeilInt(core_segment, data_each_vector) * data_each_vector;

      param.one_core_ele = core_segment;
      param.act_core_num = GetCeilInt(param.last_dim_size, core_segment);
      if (param.last_dim_size % param.one_core_ele == 0) {
        param.last_core_ele = param.one_core_ele;
      } else {
        param.last_core_ele = param.last_dim_size % param.one_core_ele;
      }
      // calc tiling info
      param.one_core_segment_loop = param.one_core_ele / MAX_SEGMENT_LEN;
      param.one_core_segment_tail = param.one_core_ele % MAX_SEGMENT_LEN;
      if (param.one_core_segment_tail != 0) {
        param.one_core_segment_tail_data = GetCeilInt(param.one_core_segment_tail, data_each_vector) * data_each_vector;
        param.one_core_offset = param.one_core_segment_tail_data - param.one_core_segment_tail;
      }
      param.last_core_segment_loop = param.last_core_ele / MAX_SEGMENT_LEN;
      param.last_core_segment_tail = param.last_core_ele % MAX_SEGMENT_LEN;
      if (param.last_core_segment_tail != 0) {
        param.last_core_segment_tail_data =
            GetCeilInt(param.last_core_segment_tail, data_each_vector) * data_each_vector;
        param.last_core_offset = param.last_core_segment_tail_data - param.last_core_segment_tail;
      }
      // select branch
      if (dtype == "float32") {
        param.tiling_mode = 8;  // do_not_last
      } else {
        param.tiling_mode = 11;  // do_not_last
        if (param.axis_size <= 2048) {
          param.tiling_mode = 9;  // do_not_last_fp16_default
        }
      }
    }
  }
}

template <typename T>
static bool GetCompileInfo(const nlohmann::json& op_info, const string& name, T& value) {
  const nlohmann::json& all_vars = op_info["vars"];
  if (all_vars.empty()) {
    return false;
  }
  if (all_vars.count(name) == 0) {
    return false;
  }
  value = all_vars[name].get<T>();
  return true;
}

static bool ArgOpsTiling(const string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                         OpRunInfo& run_info) {
  OP_LOGI(op_type.c_str(), "ArgOpsTiling running.");

  // get axis value
  int32_t axis;
  string axis_type = op_paras.inputs[1].tensor[0].dtype;
  if (op_paras.const_inputs.count("dimension") == 0 || get<0>(op_paras.const_inputs.at("dimension")) == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "const_inputs not include dimension or dimension is null");
    return false;
  }
  if (axis_type == "int32") {
    const int32_t* axis_ptr = reinterpret_cast<const int32_t*>(std::get<0>(op_paras.const_inputs.at("dimension")));
    axis = *axis_ptr;
  } else if (axis_type == "int64") {
    const int64_t* axis_ptr = reinterpret_cast<const int64_t*>(std::get<0>(op_paras.const_inputs.at("dimension")));
    axis = *axis_ptr;
  } else {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ArgOpsTiling: axis can only support int32, int64, but got %s", axis_type.c_str());
    return false;
  }
  // check axis value and set to positive value
  vector<int64_t> input_shape = op_paras.inputs[0].tensor[0].shape;
  int64_t input_dims = input_shape.size();
  if (axis < -input_dims || axis >= input_dims) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ArgOpsTiling: axis is invalid, axis:%d, dims:%d", axis, input_dims);
    return false;
  }
  axis = (axis + input_dims) % input_dims;
  OP_LOGD(op_type.c_str(), "ArgOpsTiling", "axis is %d.", axis);

  // get compile info
  int32_t ub_ele = 0;
  int32_t core_num = 1;
  const map<string, int32_t&> compile_params = {
      {"ub_ele", ub_ele},
      {"core_num", core_num},
  };
  for (auto& param : compile_params) {
    const auto& name = param.first;
    OP_LOGD(op_type.c_str(), "GetCompileInfo %s.", name.c_str());
    if (!GetCompileInfo<int32_t>(op_info, name, param.second)) {
      return false;
    }
    OP_LOGD(op_type.c_str(), "%s=%d.", name.c_str(), param.second);
  }

  // calc and set and print tiling param
  TilingParam param;
  memset(&param, 0, sizeof(param));
  string dtype = op_paras.inputs[0].tensor[0].dtype;
  CalTilingParam(param, input_shape, dtype, axis, ub_ele, core_num);
  SetTilingParam(param, run_info);
  PrintParam(param);

  // reserve
  run_info.block_dim = param.act_core_num;
  vector<int64_t> workspace;
  run_info.workspaces = workspace;

  OP_LOGI(op_type.c_str(), "ArgOpsTiling run success.");
  return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(ArgMin, ArgOpsTiling);
REGISTER_OP_TILING_FUNC_BUFFERED(ArgMaxV2, ArgOpsTiling);
}  // namespace optiling
