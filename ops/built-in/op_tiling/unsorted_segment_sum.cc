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
 * \file unsorted_segment_sum.cpp
 * \brief
 */
#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"

#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "error_log.h"

namespace optiling {

const std::string UNSORTED_SEGMENT_SUM_OP_TYPE = "UnsortedSegmentSum";
const int32_t BYTE_BLOCK = 32;
const int32_t MIN_ELE_SIZE_USING_ALL_CORE = 1024;
const int32_t MASK_FP32 = 64;
const int32_t MASK_INT32 = 64;
const int32_t MASK_FP16 = 128;
const int32_t MAX_REPEAT_TIME = 255;
const int32_t FP32_ELE_NUM_ALIGN_32B = 8;
const int32_t BYTE_FULL_MASK = 256;
const int32_t MULTI = 4;
const int32_t FP16_BLOCK_NUM = 16;
const int32_t INT32_BLOCK_NUM = 8;
// dtype
const std::string DTYPE_FP32 = "float32";
const std::string DTYPE_FP16 = "float16";
const std::string DTYPE_INT32 = "int32";
const std::string DTYPE_INT8 = "int8";
const std::string DTYPE_UINT8 = "uint8";

// ub tensor num
const int32_t UB_TENSOR_NUM_FP32_INPUT_LAST_AXIS_ALIGN = 2;
const int32_t UB_TENSOR_NUM_FP32_INPUT_LAST_AXIS_ONE = 3;
const int32_t UB_TENSOR_NUM_FP32_INPUT_ONE_DIM = 3;
const int32_t UB_TENSOR_NUM_FP32_INPUT_LAST_AXIS_NOT_ALIGN = 3;

// fp32 select key
const int32_t SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ALIGN_SMALL_E = 1;
const int32_t SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ONE = 2;
const int32_t SELECT_KEY_MODE_FP32_INPUT_ONE_DIM = 2;
const int32_t SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN_SMALL_E = 4;
const int32_t SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ALIGN_BIG_E = 5;
const int32_t SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN_BIG_E = 6;
const int32_t SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ONE_MODIFY = 7;
const int32_t SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ONE_MULTI = 8;
const int32_t SELECT_KEY_MODE_FP32_INPUT_NUM_SEGMENT_ONE = 17;

// int32 select key
const int32_t SELECT_KEY_MODE_NO_ATOMIC_SMALL_E_SMALL_ID = 9;
const int32_t SELECT_KEY_MODE_NO_ATOMIC_SMALL_E_BIG_ID = 10;
const int32_t SELECT_KEY_MODE_NO_ATOMIC_BIG_E_SMALL_ID = 11;
const int32_t SELECT_KEY_MODE_NO_ATOMIC_BIG_E_BIG_ID = 12;
const int32_t SELECT_KEY_MODE_NO_ATOMIC_SMALL_E_SMALL_ID_SMALLBLOCK = 13;
const int32_t SELECT_KEY_MODE_NO_ATOMIC_SMALL_E_BIG_ID_SMALLBLOCK = 14;
const int32_t SELECT_KEY_MODE_NO_ATOMIC_NUM_SEGMENT_ONE = 15;
const int32_t SELECT_KEY_MODE_NO_ATOMIC_ALL_IN_ALIGN = 16;
enum EleByte { FP16_BYTE = 2, FP32_BYTE = 4, INT32_BYTE = 4, INT8_BYTE = 1, UINT8_BYTE = 1 };

struct TilingParamsFp32 {
  // common params
  int32_t select_key_input_scalar;
  int32_t need_core_num_input_scalar;

  // input data params
  // front core
  int32_t input_ele_num_front_core_input_scalar;
  // front part front core
  int32_t input_mov_times_gm2ub_front_part_front_core_input_scalar;
  int32_t input_front_burst_len_front_part_front_core_input_scalar;
  int32_t input_last_burst_len_front_part_front_core_input_scalar;
  int32_t input_front_ele_num_ub_front_part_front_core_input_scalar;
  int32_t input_last_ele_num_ub_front_part_front_core_input_scalar;
  int32_t input_front_rows_front_part_front_core_input_scalar;
  int32_t input_last_rows_front_part_front_core_input_scalar;
  // last part front core
  int32_t input_mov_times_gm2ub_last_part_front_core_input_scalar;
  int32_t input_front_burst_len_last_part_front_core_input_scalar;
  int32_t input_last_burst_len_last_part_front_core_input_scalar;
  int32_t input_front_ele_num_ub_last_part_front_core_input_scalar;
  int32_t input_last_ele_num_ub_last_part_front_core_input_scalar;
  int32_t input_front_rows_last_part_front_core_input_scalar;
  int32_t input_last_rows_last_part_front_core_input_scalar;
  // last core
  int32_t input_ele_num_last_core_input_scalar;
  // front part last core
  int32_t input_mov_times_gm2ub_front_part_last_core_input_scalar;
  int32_t input_front_burst_len_front_part_last_core_input_scalar;
  int32_t input_last_burst_len_front_part_last_core_input_scalar;
  int32_t input_front_ele_num_ub_front_part_last_core_input_scalar;
  int32_t input_last_ele_num_ub_front_part_last_core_input_scalar;
  int32_t input_front_rows_front_part_last_core_input_scalar;
  int32_t input_last_rows_front_part_last_core_input_scalar;
  // last part last core
  int32_t input_mov_times_gm2ub_last_part_last_core_input_scalar;
  int32_t input_front_burst_len_last_part_last_core_input_scalar;
  int32_t input_last_burst_len_last_part_last_core_input_scalar;
  int32_t input_front_ele_num_ub_last_part_last_core_input_scalar;
  int32_t input_last_ele_num_ub_last_part_last_core_input_scalar;
  int32_t input_front_rows_last_part_last_core_input_scalar;
  int32_t input_last_rows_last_part_last_core_input_scalar;

  // e num params
  int32_t e_num_input_scalar;
  int32_t e_mov_times_gm2ub_input_scalar;
  int32_t e_ub2gm_front_burst_len_input_scalar;
  int32_t e_num_front_part_input_scalar;
  int32_t e_ub2gm_last_burst_len_input_scalar;
  int32_t e_gm2ub_last_burst_len_input_scalar;
  int32_t e_num_last_part_input_scalar;

  // ids params
  int32_t ids_size_input_scalar;
  int32_t ids_ele_num_front_core_input_scalar;
  int32_t ids_mov_times_gm2ub_front_core_input_scalar;
  int32_t ids_front_burst_len_front_core_input_scalar;
  int32_t ids_last_burst_len_front_core_input_scalar;
  int32_t ids_ele_num_ub_front_part_front_core_input_scalar;
  int32_t ids_ele_num_ub_last_part_front_core_input_scalar;
  int32_t ids_ele_num_last_core_input_scalar;
  int32_t ids_mov_times_gm2ub_last_core_input_scalar;
  int32_t ids_front_burst_len_last_core_input_scalar;
  int32_t ids_last_burst_len_last_core_input_scalar;
  int32_t ids_ele_num_ub_front_part_last_core_input_scalar;
  int32_t ids_ele_num_ub_last_part_last_core_input_scalar;

  // output init params
  int32_t output_ub_init_last_repeat_time_front_part_front_core_input_scalar;
  int32_t output_ub_init_times_front_part_front_core_input_scalar;
  int32_t output_ub_init_last_repeat_time_last_part_front_core_input_scalar;
  int32_t output_ub_init_times_last_part_front_core_input_scalar;
  int32_t output_ub_init_last_repeat_time_front_part_last_core_input_scalar;
  int32_t output_ub_init_times_front_part_last_core_input_scalar;
  int32_t output_ub_init_last_repeat_time_last_part_last_core_input_scalar;
  int32_t output_ub_init_times_last_part_last_core_input_scalar;
  int32_t input_last_axis_align_front_part_ele_num_input_scalar;
  int32_t input_last_axis_align_floor_ele_num_input_scalar;
  int32_t last_part_vadd_mask_input_scalar;
  int32_t output_ub_init_last_row_last_repeat_time_front_part_front_core_input_scalar;
  int32_t output_ub_init_last_row_times_front_part_front_core_input_scalar;
  int32_t output_ub_init_last_row_last_repeat_time_last_part_front_core_input_scalar;
  int32_t output_ub_init_last_row_times_last_part_front_core_input_scalar;
  int32_t output_ub_init_last_row_last_repeat_time_front_part_last_core_input_scalar;
  int32_t output_ub_init_last_row_times_front_part_last_core_input_scalar;
  int32_t output_ub_init_last_row_last_repeat_time_last_part_last_core_input_scalar;
  int32_t output_ub_init_last_row_times_last_part_last_core_input_scalar;
};

struct TilingParamsInt32 {
  // common params
  int32_t select_key_input_scalar;
  int32_t need_core_num_input_scalar;
  int32_t num_segments_front_core_input_scalar;
  int32_t num_segments_last_core_input_scalar;

  // ids params
  int32_t ids_size_input_scalar;
  int32_t ids_mov_times_gm2ub_input_scalar;
  int32_t ids_ele_num_ub_front_part_input_scalar;
  int32_t ids_front_burst_len_input_scalar;
  int32_t ids_ele_num_ub_last_part_input_scalar;
  int32_t ids_last_burst_len_input_scalar;

  // e num params
  int32_t e_num_input_scalar;
  int32_t e_mov_times_gm2ub_input_scalar;
  int32_t e_ub2gm_front_burst_len_input_scalar;
  int32_t e_num_front_part_input_scalar;
  int32_t repeat_time_front_part_input_scalar;
  int32_t e_ub2gm_last_burst_len_input_scalar;
  int32_t e_num_last_part_input_scalar;
  int32_t repeat_time_last_part_input_scalar;
  int32_t align_scalar;
  int32_t align_scalar_lastcore;

  int32_t e_gm2ub_front_burst_len_input_scalar;
  int32_t e_gm2ub_last_burst_len_input_scalar;
  int32_t num_segment_max;
  int32_t num_segment_max_time;
  int32_t num_segment_max_time_lastcore;
  int32_t front_num_segment;
  int32_t front_num_segment_last;
  int32_t front_num_segment_lastcore;
  int32_t front_num_segment_last_lastcore;
  int32_t e_ub2gm_front_burst_len_input_scalar_lastcore;
  int32_t e_ub2gm_last_burst_len_input_scalar_lastcore;
  int32_t repeat_times;
  int32_t repeat_times_last_part;
  int32_t repeat_times_last_part_lastcore;
  int32_t e_mov_times_gm2ub_input_scalar_lastcore;
  int32_t repeat_time_front_part_input_scalar_lastcore;
};

/******************COMMON_FUNCTION******************/

int32_t ComputeDivRemainders(const int32_t& num, const int32_t& factor, const int32_t& times) {
  int32_t res = num - factor * times;
  return res;
}

int32_t UssCeil(const int32_t& num, const int32_t& factor) {
  int32_t res = (num % factor == 0) ? num : factor * (num / factor + 1);
  return res;
}

int32_t UssCeilDiv(const int32_t& num, const int32_t& factor) {
  int32_t res = (num % factor == 0) ? num / factor : num / factor + 1;
  return res;
}

int32_t UssCeilDivNoAtomic(const int32_t& num, const int32_t& factor, const int32_t& e_size,
                           const int32_t& output_ub_ele_num_one_row) {
  int32_t res = num / factor;
  if(factor>1 && e_size < output_ub_ele_num_one_row) {
    if (e_size * res % output_ub_ele_num_one_row != 0) {
      res = res / output_ub_ele_num_one_row * output_ub_ele_num_one_row;
    }
    int32_t last = num - (factor - 1) * res;
    if(e_size * last < output_ub_ele_num_one_row) {
      last = output_ub_ele_num_one_row / e_size + 1;
      res = (num - last) / (factor - 1);
      if (res < 1) {
        res = 1;
      }
      res = e_size * res / output_ub_ele_num_one_row * output_ub_ele_num_one_row / e_size;
    }
  }
  return res;
}

bool GetUssCompileParams(const std::string& op_type, const nlohmann::json& op_compile_info_json, int32_t& core_num,
                         int32_t& ub_size, int32_t& ub_tensor_num) {
  using namespace nlohmann;
  if (op_compile_info_json == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_compile_info_json is null");
    return false;
  }
  const auto& allVars = op_compile_info_json["vars"];
  // core num
  if (allVars.count("core_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "core_num is null");
    return false;
  }
  core_num = allVars["core_num"].get<std::int32_t>();
  // ub size
  if (allVars.count("ub_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ub_size is null");
    return false;
  }
  ub_size = allVars["ub_size"].get<std::int32_t>();
  // ub tensor num
  if (allVars.count("ub_tensor_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ub_tensor_num is null");
    return false;
  }
  ub_tensor_num = allVars["ub_tensor_num"].get<std::int32_t>();
  GELOGD("op [%s] : GetCompileParams, core_num[%d], ub_size[%d].", UNSORTED_SEGMENT_SUM_OP_TYPE.c_str(), core_num,
         ub_size);
  return true;
}

bool GetTilingMode(const std::vector<int64_t>& input_shape, const int32_t& e_size, const std::string& input_dtype,
                   const int32_t& ub_tensor_ele_num, int32_t& select_key, int32_t& num_segments) {
  int input_dim = input_shape.size();
  if (input_shape.empty()) {
    return false;
  }
  if (num_segments > 1) {
    if (input_dim == 1) {
      select_key = SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ONE_MULTI;
    } else if (input_dim > 1) {
      if (e_size == 1) {
        select_key = SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ONE_MULTI;
      } else if (e_size % FP32_ELE_NUM_ALIGN_32B == 0) {
        if (e_size < ub_tensor_ele_num) {
          select_key = SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ALIGN_SMALL_E;
        } else {
          select_key = SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ALIGN_BIG_E;
        }
      } else {
        if (e_size < ub_tensor_ele_num) {
          select_key = SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN_SMALL_E;
        } else {
          select_key = SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN_BIG_E;
        }
      }
    }
    return true;
  } else {
    select_key = SELECT_KEY_MODE_FP32_INPUT_NUM_SEGMENT_ONE;
    return true;
  }
  return false;
}

bool GetTilingModeNoAtomic(
  const std::vector<int64_t>& input_shape, const int32_t& e_size, const int32_t& ids_size,
  const std::string& input_dtype, const std::string& ids_dtype, int32_t& ub_tensor_size,
  int32_t& ub_tensor_size_input, int32_t& select_key, int32_t& e_once_num,int32_t& id_once_num,
  const int32_t& need_core, const int32_t& output_ub_ele_num_one_row, int32_t& num_segment_max,
  int32_t& mask, const int32_t& all_size, int32_t& num_segments) {
  int input_byte = 0;
  if (input_dtype  == DTYPE_FP16) {
    input_byte = 2;
  } else {
    input_byte = 4;
  }
  int32_t e_once_ubsize = (ub_tensor_size_input / BYTE_FULL_MASK) * BYTE_FULL_MASK;
  e_once_num = e_once_ubsize / input_byte;
  id_once_num = ((ub_tensor_size / BYTE_BLOCK) * BYTE_BLOCK) / INT32_BYTE;
  num_segment_max = ((ub_tensor_size / BYTE_BLOCK) * BYTE_BLOCK) / input_byte;
  num_segment_max = num_segment_max / e_size;

  if (input_shape.empty()) {
    OP_LOGD("input shape is empty");
    return false;
  }
  if(num_segments == 1) {
    select_key = SELECT_KEY_MODE_NO_ATOMIC_NUM_SEGMENT_ONE;
    return true;
  } else if (e_size > e_once_num && ids_size > id_once_num) {
    // e big id big
    select_key = SELECT_KEY_MODE_NO_ATOMIC_BIG_E_BIG_ID;
    return true;
  } else if(e_size > e_once_num && ids_size < id_once_num) {
    // e nig id small
    select_key = SELECT_KEY_MODE_NO_ATOMIC_BIG_E_SMALL_ID;
    return true;
  } else if(e_size < e_once_num && ids_size < id_once_num) {
    // e small id small
    select_key = SELECT_KEY_MODE_NO_ATOMIC_SMALL_E_SMALL_ID;
    if (need_core > 1 && e_size < output_ub_ele_num_one_row) {
      select_key = SELECT_KEY_MODE_NO_ATOMIC_SMALL_E_SMALL_ID_SMALLBLOCK;
    } else if(e_size % output_ub_ele_num_one_row == 0 && all_size < e_once_num) {
      select_key = SELECT_KEY_MODE_NO_ATOMIC_ALL_IN_ALIGN;
    }
    return true;
  } else if(ids_size > id_once_num && e_size < e_once_num) {
    // e small id big
    select_key = SELECT_KEY_MODE_NO_ATOMIC_SMALL_E_BIG_ID;
    if(need_core > 1 && e_size < output_ub_ele_num_one_row) {
      select_key = SELECT_KEY_MODE_NO_ATOMIC_SMALL_E_BIG_ID_SMALLBLOCK;
    }
    return true;
  }
  return false;
}

bool GetEleDtype(const std::string& dtype, EleByte& elebyte) {
  if (dtype == "float32") {
    elebyte = FP32_BYTE;
    return true;
  } else if (dtype == "float16") {
    elebyte = FP16_BYTE;
    return true;
  } else if (dtype == "int32") {
    elebyte = INT32_BYTE;
    return true;
  } else if (dtype == "int8") {
    elebyte = INT8_BYTE;
    return true;
  } else if (dtype == "uint8") {
    elebyte = UINT8_BYTE;
    return true;
  }
  return false;
}

bool IsUsingAllCore(const int32_t& ids_size, const int32_t& core_num, int32_t& need_core_num, int32_t& e_size,
                    int32_t& num_segments) {
  int32_t ele_num = ids_size / core_num;
  if(num_segments > 1) {
    if (e_size > 1) {
      if (ele_num >= 1) {
        need_core_num = core_num;
        return true;
      } else {
        need_core_num = ids_size;
        return true;
      }
    } else {
      if (ids_size <= 64 || ele_num <= 0) {
        need_core_num = 1;
        return true;
      } else {
        if(ele_num >= 64) {
          need_core_num = core_num;
          return true;
        } else {
          need_core_num = ids_size / 64;
          return true;
        }
      }
    }
  } else {
    if(ele_num >= 1) {
      need_core_num = core_num;
      return true;
    } else {
      need_core_num = ids_size;
      return true;
    }
  }
  need_core_num = 1;
  return false;
}

bool IsUsingAllCoreByNumSegments(const int32_t& num_segments, const int32_t& core_num,
                                 int32_t& need_core_num,int32_t& e_size, int32_t& output_ub_ele_num_one_row) {
  int32_t ele_num = num_segments / core_num;
  if(num_segments == 1) {
    if(e_size <= output_ub_ele_num_one_row) {
      need_core_num = 1;
      return true;
    } else {
      int32_t core_one = e_size / output_ub_ele_num_one_row;
      if(core_one >= core_num) {
        need_core_num = core_num;
        return true;
      } else {
        if(e_size % output_ub_ele_num_one_row == 0) {
          need_core_num = core_one;
          return true;
        } else {
          need_core_num = core_one + 1;
          return true;
        }
      }
    }
  }
  if(e_size < output_ub_ele_num_one_row && ele_num < output_ub_ele_num_one_row) {
    need_core_num = 1;
    return true;
  }
  if(e_size >= output_ub_ele_num_one_row) {
    if(ele_num >= 1) {
      need_core_num = core_num;
    } else {
      need_core_num = num_segments;
    }
    return true;
  }
  if(e_size < output_ub_ele_num_one_row && ele_num >= output_ub_ele_num_one_row) {
    need_core_num = core_num;
    return true;
  }
  need_core_num = 1;
  return false;
}

void ComputeUbTensorSizeNoAtomic(
  const int32_t& ub_size, const std::vector<int64_t>& input_shape,
  const std::string& input_dtype, const int32_t& e_size, int32_t& ub_tensor_size_id,int32_t& ub_tensor_size_input,
  int32_t& ub_tensor_size_output, const int32_t& output_ub_ele_num_one_row,
  const int32_t & need_core_num, int32_t& mask, int32_t num_segments) {
  int32_t input_ele_byte = (input_dtype == DTYPE_FP16)? 2 : 4;
  if(num_segments == 1) {
    ub_tensor_size_id = ub_size / 2;
    ub_tensor_size_input = ub_tensor_size_id;
    ub_tensor_size_output = ub_tensor_size_id;
  } else if(need_core_num > 1 && e_size < output_ub_ele_num_one_row) {
    int32_t ub_tensor_num = 2;
    ub_tensor_size_input = mask * input_ele_byte;
    ub_tensor_size_id = (ub_size - ub_tensor_size_input) / ub_tensor_num;
    ub_tensor_size_output = ub_tensor_size_id;
  } else {
    ub_tensor_size_input = 16000 * input_ele_byte;
    ub_tensor_size_output = ub_tensor_size_input;
    ub_tensor_size_id = ub_size - 2 * ub_tensor_size_input;
  }
}

void NumSegmentOne(
  int32_t& e_mov_times_gm2ub_input_scalar,int32_t& max_ele_num_one_ub_tensor,
  int32_t& e_num_front_part_input_scalar, int32_t& e_ub2gm_front_burst_len_input_scalar,
  int32_t& repeat_times,int32_t& repeat_time_front_part_input_scalar,int32_t& e_num_last_part_input_scalar,
  int32_t& e_ub2gm_last_burst_len_input_scalar,int32_t& repeat_times_last_part, int32_t& mask,
  const EleByte& ele_byte, int32_t& num_segments_front_core_input_scalar,int32_t& repeat_time_last_part_input_scalar) {
  if(e_mov_times_gm2ub_input_scalar > 1) {
    e_num_front_part_input_scalar = max_ele_num_one_ub_tensor;
    e_ub2gm_front_burst_len_input_scalar = UssCeilDiv(e_num_front_part_input_scalar * ele_byte, BYTE_BLOCK);
    repeat_times = UssCeilDiv(e_num_front_part_input_scalar,mask * 255);

    repeat_time_front_part_input_scalar = UssCeilDiv(e_num_front_part_input_scalar -
                                          (repeat_times - 1) * mask * 255, mask);

    e_num_last_part_input_scalar = ComputeDivRemainders(num_segments_front_core_input_scalar,
                                   e_num_front_part_input_scalar, e_mov_times_gm2ub_input_scalar - 1);
    e_ub2gm_last_burst_len_input_scalar = UssCeilDiv(e_num_last_part_input_scalar * ele_byte, BYTE_BLOCK);

    repeat_times_last_part = UssCeilDiv(e_num_last_part_input_scalar, mask * 255);
    if(repeat_times_last_part > 1) {
      repeat_time_last_part_input_scalar = UssCeilDiv(e_num_last_part_input_scalar -
                                           (repeat_times_last_part - 1) * mask * 255, mask);
    } else {
      repeat_time_last_part_input_scalar = UssCeilDiv(e_num_last_part_input_scalar, mask);
    }
  } else {
    e_num_front_part_input_scalar = num_segments_front_core_input_scalar;

    e_ub2gm_front_burst_len_input_scalar = UssCeilDiv(e_num_front_part_input_scalar * ele_byte, BYTE_BLOCK);
    repeat_times = UssCeilDiv(e_num_front_part_input_scalar, mask * 255);
    if(repeat_times > 1) {

      repeat_time_front_part_input_scalar = UssCeilDiv(e_num_front_part_input_scalar -
                                            (repeat_times - 1) * mask * 255, mask);
    } else {
      repeat_time_front_part_input_scalar = UssCeilDiv(e_num_front_part_input_scalar, mask);
    }
    repeat_time_last_part_input_scalar = repeat_time_front_part_input_scalar;
    e_num_last_part_input_scalar = e_num_front_part_input_scalar;
    e_ub2gm_last_burst_len_input_scalar = e_ub2gm_front_burst_len_input_scalar;
    repeat_times_last_part = repeat_times;
  }
}
void ComputeUbTensorSize(const int32_t& ub_size, const std::vector<int64_t>& input_shape,
                         const std::string& input_dtype, int32_t& e_size,
                         int32_t& ub_tensor_size, int32_t& num_segments) {
  if(num_segments > 1) {
    if (e_size == 1) {
      // input is one dim or last axis is one
      int32_t one_row_size = FP32_BYTE + INT32_BYTE + FP32_BYTE * FP32_ELE_NUM_ALIGN_32B;
      ub_tensor_size = UssCeil(ub_size / one_row_size, BYTE_BLOCK);
    } else if (e_size > 1) {
      int32_t ub_tensor_num = 0;
      if (e_size % FP32_ELE_NUM_ALIGN_32B == 0) {
        // align
        ub_tensor_num = UB_TENSOR_NUM_FP32_INPUT_LAST_AXIS_ALIGN;
        ub_tensor_size = ((ub_size / ub_tensor_num) / BYTE_BLOCK) * BYTE_BLOCK;
      } else if (e_size % FP32_ELE_NUM_ALIGN_32B > 0) {
        // not align
        ub_tensor_num = UB_TENSOR_NUM_FP32_INPUT_LAST_AXIS_NOT_ALIGN;
        ub_tensor_size = ((ub_size / ub_tensor_num) / BYTE_BLOCK) * BYTE_BLOCK;
      }
    }
  } else {
    ub_tensor_size = ub_size / BYTE_BLOCK * BYTE_BLOCK;
  }
}

/******************MODE_FP32_INPUT_LAST_AXIS_ALIGN******************/
void ComputeEleNumOneCore(const int32_t& min_ele_num, const int32_t& ids_num, const int32_t& core_num,
                          const int32_t& e_size, int32_t& ids_ele_num_front_core, int32_t& ids_ele_num_last_core,
                          int32_t& input_ele_num_front_core, int32_t& input_ele_num_last_core, int32_t& num_segments) {
  int32_t ids_num_align = UssCeil(ids_num, min_ele_num);
  if(num_segments > 1) {
    if (e_size == 1) {
      ids_ele_num_front_core = ids_num_align / core_num;
      ids_ele_num_front_core = ids_ele_num_front_core / MASK_FP32 * MASK_FP32;
      ids_ele_num_last_core = ComputeDivRemainders(ids_num, ids_ele_num_front_core, core_num - 1);
      input_ele_num_front_core = ids_ele_num_front_core;
      input_ele_num_last_core = ids_ele_num_last_core;
      return;
    }
    ids_ele_num_front_core = ids_num / core_num;

    ids_ele_num_last_core = ComputeDivRemainders(ids_num, ids_ele_num_front_core, core_num - 1);
    input_ele_num_front_core = ids_ele_num_front_core * e_size;
    input_ele_num_last_core = ids_ele_num_last_core * e_size;
  } else {
    ids_ele_num_front_core = ids_num / core_num;
    ids_ele_num_last_core = ComputeDivRemainders(ids_num, ids_ele_num_front_core, core_num - 1);
  }
}

void ComputeInputParamsMovGm2ub(const int32_t& ub_tensor_size, const EleByte& input_ele_byte,
                                const int32_t& input_ele_num, const int32_t& e_size, const int32_t& e_mov_times_gm2ub,
                                const int32_t& ids_ele_num_ub, int32_t& input_mov_times_gm2ub,
                                int32_t& input_front_burst_len, int32_t& input_last_burst_len,
                                int32_t& input_ele_num_ub_front_part, int32_t& input_ele_num_ub_last_part,
                                int32_t& input_rows_front_part, int32_t& input_rows_last_part) {
  if (e_mov_times_gm2ub == 1) {
    // e_size is small, ub_tensor_size is enough for e_num
    int32_t e_byte_align_32B = UssCeil(e_size, FP32_ELE_NUM_ALIGN_32B) * input_ele_byte;
    input_rows_front_part = ub_tensor_size / e_byte_align_32B;
    input_mov_times_gm2ub = UssCeilDiv(ids_ele_num_ub, input_rows_front_part);
    if (input_mov_times_gm2ub > 1) {
      input_front_burst_len = e_byte_align_32B * input_rows_front_part / BYTE_BLOCK;
      input_ele_num_ub_front_part = e_size * input_rows_front_part;
      input_rows_last_part = ComputeDivRemainders(ids_ele_num_ub, input_rows_front_part, input_mov_times_gm2ub - 1);
      input_last_burst_len = e_byte_align_32B * input_rows_last_part / BYTE_BLOCK;
      input_ele_num_ub_last_part = input_rows_last_part * e_size;
    } else if (input_mov_times_gm2ub == 1) {
      input_rows_front_part = ids_ele_num_ub;
      input_rows_last_part = input_rows_front_part;
      input_ele_num_ub_front_part = e_size * input_rows_front_part;
      input_ele_num_ub_last_part = input_ele_num_ub_front_part;
      input_front_burst_len = UssCeil(input_ele_num_ub_front_part * input_ele_byte, BYTE_BLOCK) / BYTE_BLOCK;
      input_last_burst_len = input_front_burst_len;
    }
  } else if (e_mov_times_gm2ub > 1) {
    // e_size is big, ub_tensor_size is not enough for e_num, use e params to move data
    input_mov_times_gm2ub = ids_ele_num_ub;
  }
}

void ComputeIdsParamsMovGm2ub(const int32_t& ids_ele_num_one_core, const int32_t& ub_tensor_size,
                              const EleByte& ids_ele_byte, int32_t& ids_mov_times_gm2ub, int32_t& ids_front_burst_len,
                              int32_t& ids_last_burst_len, int32_t& ids_ele_num_ub_front_part,
                              int32_t& ids_ele_num_ub_last_part) {
  int32_t max_ids_ele_num_one_ub_tensor = ub_tensor_size / ids_ele_byte;
  if (ids_ele_num_one_core <= max_ids_ele_num_one_ub_tensor) {
    // mov_times = 1, ub tensor is enough for ele one core
    ids_mov_times_gm2ub = 1;
    ids_ele_num_ub_front_part = ids_ele_num_one_core;
    ids_ele_num_ub_last_part = ids_ele_num_ub_front_part;
  } else {
    // mov_times > 1
    if (ids_ele_num_one_core % max_ids_ele_num_one_ub_tensor == 0) {
      // no last part
      ids_mov_times_gm2ub = ids_ele_num_one_core / max_ids_ele_num_one_ub_tensor;
      ids_ele_num_ub_front_part = max_ids_ele_num_one_ub_tensor;
      ids_ele_num_ub_last_part = ids_ele_num_ub_front_part;
    } else {
      // exist last part
      ids_mov_times_gm2ub = ids_ele_num_one_core / max_ids_ele_num_one_ub_tensor + 1;
      ids_ele_num_ub_front_part = max_ids_ele_num_one_ub_tensor;
      ids_ele_num_ub_last_part =
        ComputeDivRemainders(ids_ele_num_one_core, max_ids_ele_num_one_ub_tensor, ids_mov_times_gm2ub - 1);
    }
  }
  ids_front_burst_len = UssCeilDiv(ids_ele_num_ub_front_part * ids_ele_byte, BYTE_BLOCK);
  ids_last_burst_len = UssCeilDiv(ids_ele_num_ub_last_part * ids_ele_byte, BYTE_BLOCK);
}
void ComputeIdsParamsMovGm2ubNoAtomic(const int32_t& ids_ele_num_one_core, const int32_t& id_once_num,
                                      const EleByte& ids_ele_byte, int32_t& ids_mov_times_gm2ub,
                                      int32_t& ids_front_burst_len, int32_t& ids_last_burst_len,
                                      int32_t& ids_ele_num_ub_front_part, int32_t& ids_ele_num_ub_last_part) {
  int32_t max_ids_ele_num_one_ub_tensor = id_once_num;
  if (ids_ele_num_one_core <= max_ids_ele_num_one_ub_tensor) {
    // mov_times = 1, ub tensor is enough for ele one core
    ids_mov_times_gm2ub = 1;
    ids_ele_num_ub_front_part = ids_ele_num_one_core;
    ids_ele_num_ub_last_part = ids_ele_num_ub_front_part;
  } else {
    // mov_times > 1
    if (ids_ele_num_one_core % max_ids_ele_num_one_ub_tensor == 0) {
      // no last part
      ids_mov_times_gm2ub = ids_ele_num_one_core / max_ids_ele_num_one_ub_tensor;
      ids_ele_num_ub_front_part = max_ids_ele_num_one_ub_tensor;
      ids_ele_num_ub_last_part = ids_ele_num_ub_front_part;
    } else {
      // exist last part
      ids_mov_times_gm2ub = ids_ele_num_one_core / max_ids_ele_num_one_ub_tensor + 1;
      ids_ele_num_ub_front_part = max_ids_ele_num_one_ub_tensor;
      ids_ele_num_ub_last_part =
        ComputeDivRemainders(ids_ele_num_one_core, max_ids_ele_num_one_ub_tensor, ids_mov_times_gm2ub - 1);
    }
  }
  ids_front_burst_len = UssCeilDiv(ids_ele_num_ub_front_part * ids_ele_byte, BYTE_BLOCK);
  ids_last_burst_len = UssCeilDiv(ids_ele_num_ub_last_part * ids_ele_byte, BYTE_BLOCK);
}

void ComputeInitOutputUbParams(const int32_t& ids_ele_num, const int32_t& output_ub_ele_num_one_row,
                               int32_t& output_ub_init_last_repeat_time, int32_t& output_ub_init_times) {
  int32_t repeat_times = UssCeilDiv(ids_ele_num * output_ub_ele_num_one_row, MASK_FP32);
  if (repeat_times % MAX_REPEAT_TIME == 0) {
    output_ub_init_times = repeat_times / MAX_REPEAT_TIME;
    output_ub_init_last_repeat_time = MAX_REPEAT_TIME;
  } else {
    output_ub_init_times = repeat_times / MAX_REPEAT_TIME + 1;
    output_ub_init_last_repeat_time =
      ComputeDivRemainders(repeat_times, repeat_times / MAX_REPEAT_TIME, MAX_REPEAT_TIME);
  }
}

void ComputeNumSegmentsParams(
  const int32_t& need_core_num, const int32_t& num_segmens, int32_t& num_segmens_front_core,
  int32_t& num_segmens_last_core, const int32_t& e_size,
  int32_t& output_ub_ele_num_one_row) {
  if (need_core_num == 1 && num_segmens > 1) {
    num_segmens_front_core = num_segmens;
    num_segmens_last_core = num_segmens_front_core;
  } else if (need_core_num > 1 && num_segmens > 1) {
    num_segmens_front_core = UssCeilDivNoAtomic(num_segmens, need_core_num,e_size,output_ub_ele_num_one_row);
    num_segmens_last_core = ComputeDivRemainders(num_segmens, num_segmens_front_core, need_core_num - 1);
  } else if (num_segmens == 1) {
    if(e_size < output_ub_ele_num_one_row) {
      num_segmens_front_core = e_size;
      num_segmens_last_core = e_size;
    } else {
      if(e_size / need_core_num > output_ub_ele_num_one_row) {
        num_segmens_front_core = e_size / need_core_num / output_ub_ele_num_one_row * output_ub_ele_num_one_row;
      } else {
        num_segmens_front_core = output_ub_ele_num_one_row;
      }
      num_segmens_last_core = e_size - (num_segmens_front_core * (need_core_num - 1));
    }
  }
}

void ComputeENumParams(
  const std::string& input_dytpe,const int32_t& e_num, const EleByte& ele_byte,
  const int32_t& e_once_num, int32_t& e_mov_times_gm2ub_input_scalar, int32_t& e_ub2gm_front_burst_len_input_scalar,
  int32_t& e_num_front_part_input_scalar, int32_t& repeat_time_front_part_input_scalar,
  int32_t& e_ub2gm_last_burst_len_input_scalar, int32_t& e_num_last_part_input_scalar,
  int32_t& repeat_time_last_part_input_scalar, int32_t& align_scalar, int32_t& align_scalar_lastcore,
  int32_t& e_gm2ub_front_burst_len_input_scalar, int32_t& e_gm2ub_last_burst_len_input_scalar,
  int32_t& num_segments_front_core_input_scalar, int32_t& num_segments_last_core_input_scalar,
  int32_t& need_core, int32_t& num_segment_max, int32_t& num_segment_max_time,
  int32_t& num_segment_max_time_lastcore, int32_t&front_num_segment, int32_t& front_num_segment_last,
  int32_t&front_num_segment_lastcore, int32_t&front_num_segment_last_lastcore,
  int32_t& e_ub2gm_front_burst_len_input_scalar_lastcore, int32_t& e_ub2gm_last_burst_len_input_scalar_lastcore,
  const int32_t& all_size, int32_t& num_segments, int32_t& repeat_times, int32_t& repeat_times_last_part,
  int32_t& repeat_times_last_part_lastcore, int32_t& e_mov_times_gm2ub_input_scalar_lastcore,
  int32_t& repeat_time_front_part_input_scalar_lastcore) {
  int32_t max_ele_num_one_ub_tensor = e_once_num;
  int32_t mask = (input_dytpe == DTYPE_INT32) ? MASK_INT32 : MASK_FP16;
  int32_t byte = (input_dytpe == DTYPE_INT32) ? INT32_BLOCK_NUM : FP16_BLOCK_NUM;
  int32_t count = e_num * num_segments_front_core_input_scalar;
  int32_t lastcore_count = e_num * num_segments_last_core_input_scalar;
  if (e_num % byte == 0 && e_num > byte ) {
    align_scalar = 0;
  } else if(e_num % byte != 0 && e_num > byte) {
    align_scalar = byte - (e_num - (e_num / byte) * byte);
  }
  if (e_num >= max_ele_num_one_ub_tensor && num_segments > 1) {
    e_mov_times_gm2ub_input_scalar = UssCeilDiv(e_num, max_ele_num_one_ub_tensor);
    int32_t last = ComputeDivRemainders(e_num, max_ele_num_one_ub_tensor, e_mov_times_gm2ub_input_scalar - 1);
    if (last < byte) {
      max_ele_num_one_ub_tensor = e_once_num - mask;
      e_mov_times_gm2ub_input_scalar = UssCeilDiv(e_num, max_ele_num_one_ub_tensor);
    }
    // front part
    e_gm2ub_front_burst_len_input_scalar = UssCeilDiv(max_ele_num_one_ub_tensor * ele_byte, BYTE_BLOCK);
    e_ub2gm_front_burst_len_input_scalar = max_ele_num_one_ub_tensor * ele_byte / BYTE_BLOCK;
    e_num_front_part_input_scalar = max_ele_num_one_ub_tensor;
    repeat_time_front_part_input_scalar = UssCeilDiv(e_num_front_part_input_scalar, mask);
    // last part
    e_num_last_part_input_scalar =
      ComputeDivRemainders(e_num, e_num_front_part_input_scalar, e_mov_times_gm2ub_input_scalar - 1);
    e_gm2ub_last_burst_len_input_scalar = UssCeilDiv(e_num_last_part_input_scalar * ele_byte, BYTE_BLOCK);
    repeat_time_last_part_input_scalar = UssCeilDiv(e_num_last_part_input_scalar, mask);
    e_ub2gm_last_burst_len_input_scalar=e_num_last_part_input_scalar * ele_byte / BYTE_BLOCK;
  } else if(num_segments > 1 && e_num < max_ele_num_one_ub_tensor) {
    e_mov_times_gm2ub_input_scalar = 1;
    if(e_num >= byte || need_core == 1) {
      e_ub2gm_front_burst_len_input_scalar = e_num * ele_byte / BYTE_BLOCK;
      if(need_core == 1 && e_num * ele_byte % BYTE_BLOCK != 0) {
        e_ub2gm_front_burst_len_input_scalar = e_ub2gm_front_burst_len_input_scalar + 1;
      }
      e_gm2ub_front_burst_len_input_scalar = UssCeilDiv(e_num * ele_byte, BYTE_BLOCK);
      if(e_ub2gm_front_burst_len_input_scalar < 1) {
        e_ub2gm_front_burst_len_input_scalar = 1;
      }
      e_num_front_part_input_scalar = e_num;
      repeat_time_front_part_input_scalar = UssCeilDiv(e_num_front_part_input_scalar, mask);
      // last part
      e_num_last_part_input_scalar = e_num;
      e_ub2gm_last_burst_len_input_scalar = e_ub2gm_front_burst_len_input_scalar;
      e_gm2ub_last_burst_len_input_scalar = e_gm2ub_front_burst_len_input_scalar;
      if(e_num % byte == 0 && all_size <= max_ele_num_one_ub_tensor) {
        e_gm2ub_front_burst_len_input_scalar = UssCeilDiv(all_size * ele_byte, BYTE_BLOCK);
        e_gm2ub_last_burst_len_input_scalar = e_gm2ub_front_burst_len_input_scalar;
      }
      repeat_time_last_part_input_scalar = repeat_time_front_part_input_scalar;
    } else {
      if(num_segments_front_core_input_scalar <= num_segment_max) {
        num_segment_max_time = 1;
        e_ub2gm_front_burst_len_input_scalar = count * ele_byte / BYTE_BLOCK;
        if(e_ub2gm_front_burst_len_input_scalar < 1) {
          e_ub2gm_front_burst_len_input_scalar = 1;
        }
        front_num_segment = num_segments_front_core_input_scalar;
        front_num_segment_last = num_segments_front_core_input_scalar;
        e_gm2ub_front_burst_len_input_scalar = 1;
        repeat_time_front_part_input_scalar = 1;
        align_scalar = (count % byte == 0) ? 0 : byte - (count - (count / byte) * byte);
        e_num_front_part_input_scalar = e_num;
        e_num_last_part_input_scalar = e_num;
        e_ub2gm_last_burst_len_input_scalar = e_ub2gm_front_burst_len_input_scalar;
        e_gm2ub_last_burst_len_input_scalar = e_gm2ub_front_burst_len_input_scalar;
        repeat_time_last_part_input_scalar = repeat_time_front_part_input_scalar;
      } else {
        num_segment_max_time = num_segments_front_core_input_scalar / num_segment_max;
        num_segment_max_time =
        (num_segments_front_core_input_scalar % num_segment_max == 0) ? num_segment_max_time : num_segment_max_time + 1;
        front_num_segment = num_segment_max;
        front_num_segment_last = num_segments_front_core_input_scalar - num_segment_max * (num_segment_max_time - 1);
        if(front_num_segment_last * e_num < byte) {
          front_num_segment_last = byte;
          int32_t front = num_segments_front_core_input_scalar - front_num_segment_last;
          num_segment_max = (front / (num_segment_max_time - 1) / byte) * byte;
          front_num_segment = num_segment_max;
          front_num_segment_last = num_segments_front_core_input_scalar-num_segment_max * (num_segment_max_time - 1);
        }
        align_scalar =
          (front_num_segment_last * e_num % byte == 0) ? 0: byte -(front_num_segment_last * e_num -
              (front_num_segment_last * e_num / byte) * byte);
        e_ub2gm_front_burst_len_input_scalar = num_segment_max * e_num * ele_byte / BYTE_BLOCK;
        e_gm2ub_front_burst_len_input_scalar = 1;
        repeat_time_front_part_input_scalar = 1;

        e_num_front_part_input_scalar = e_num;
        e_num_last_part_input_scalar = e_num;
        e_ub2gm_last_burst_len_input_scalar = front_num_segment_last * e_num * ele_byte / BYTE_BLOCK;
        e_gm2ub_last_burst_len_input_scalar = e_gm2ub_front_burst_len_input_scalar;
        repeat_time_last_part_input_scalar = repeat_time_front_part_input_scalar;
      }
      if(num_segments_last_core_input_scalar < num_segment_max) {
        num_segment_max_time_lastcore = 1;
        align_scalar_lastcore =
          (lastcore_count % byte == 0) ? 0 : byte - (lastcore_count - (lastcore_count / byte) * byte);
        e_num_front_part_input_scalar = e_num;
        e_num_last_part_input_scalar = e_num;
        e_ub2gm_front_burst_len_input_scalar_lastcore = lastcore_count * ele_byte / BYTE_BLOCK;
        e_ub2gm_last_burst_len_input_scalar_lastcore = e_ub2gm_front_burst_len_input_scalar_lastcore;
        if(e_ub2gm_front_burst_len_input_scalar_lastcore < 1) {
          e_ub2gm_front_burst_len_input_scalar_lastcore = 1;
        }
        front_num_segment_lastcore = num_segments_last_core_input_scalar;
        front_num_segment_last_lastcore = num_segments_last_core_input_scalar;
        e_gm2ub_last_burst_len_input_scalar = 1;
        repeat_time_last_part_input_scalar = 1;
        e_gm2ub_front_burst_len_input_scalar = 1;
        repeat_time_front_part_input_scalar = 1;
      } else if(num_segments_last_core_input_scalar > num_segment_max) {
        num_segment_max_time_lastcore = num_segments_last_core_input_scalar / num_segment_max;
        num_segment_max_time_lastcore =
          (num_segments_last_core_input_scalar % num_segment_max ==
           0) ? num_segment_max_time_lastcore:num_segment_max_time_lastcore + 1;
        front_num_segment_lastcore = num_segment_max;
        front_num_segment_last_lastcore =
          num_segments_last_core_input_scalar - num_segment_max * (num_segment_max_time_lastcore - 1);
        if (front_num_segment_last_lastcore * e_num < byte) {
          front_num_segment_last_lastcore = byte;
          int32_t front = num_segments_last_core_input_scalar - front_num_segment_last_lastcore;
          int32_t num_segment_max_lastcore_front = (front / (num_segment_max_time_lastcore - 1) / byte) * byte;
          front_num_segment_lastcore = num_segment_max_lastcore_front;
          front_num_segment_last_lastcore =
            num_segments_last_core_input_scalar - front_num_segment_lastcore * (num_segment_max_time_lastcore - 1);
        }

        align_scalar_lastcore =
          (front_num_segment_last_lastcore * e_num % byte == 0) ? 0 : byte -(front_num_segment_last_lastcore * e_num -
              (front_num_segment_last_lastcore * e_num / byte) * byte);
        e_ub2gm_front_burst_len_input_scalar_lastcore = front_num_segment_lastcore * e_num * ele_byte / BYTE_BLOCK;
        e_gm2ub_front_burst_len_input_scalar = 1;
        repeat_time_front_part_input_scalar = 1;

        e_num_front_part_input_scalar = e_num;
        e_num_last_part_input_scalar = e_num;
        e_ub2gm_last_burst_len_input_scalar_lastcore = front_num_segment_last_lastcore * e_num * ele_byte / BYTE_BLOCK;
        e_gm2ub_last_burst_len_input_scalar = 1;
        repeat_time_last_part_input_scalar = 1;
      }
    }
  } else if(num_segments == 1) {
    e_mov_times_gm2ub_input_scalar = UssCeilDiv(num_segments_front_core_input_scalar, max_ele_num_one_ub_tensor);
    e_mov_times_gm2ub_input_scalar_lastcore = UssCeilDiv(num_segments_last_core_input_scalar,
        max_ele_num_one_ub_tensor);
    //front core
    NumSegmentOne(e_mov_times_gm2ub_input_scalar, max_ele_num_one_ub_tensor, e_num_front_part_input_scalar,
                  e_ub2gm_front_burst_len_input_scalar, repeat_times,
                  repeat_time_front_part_input_scalar, e_num_last_part_input_scalar,
                  e_ub2gm_last_burst_len_input_scalar, repeat_times_last_part, mask, ele_byte,
                  num_segments_front_core_input_scalar, repeat_time_last_part_input_scalar);
    //last core
    NumSegmentOne(e_mov_times_gm2ub_input_scalar_lastcore, max_ele_num_one_ub_tensor, e_num_front_part_input_scalar,
                  e_ub2gm_front_burst_len_input_scalar_lastcore,
                  repeat_times, repeat_time_front_part_input_scalar, e_num_last_part_input_scalar,
                  e_ub2gm_last_burst_len_input_scalar_lastcore, repeat_times_last_part_lastcore, mask, ele_byte,
                  num_segments_last_core_input_scalar, repeat_time_front_part_input_scalar_lastcore);
  }
}

void InitTilingParams(TilingParamsFp32& params) {
  // common params
  params.select_key_input_scalar = 0;
  params.need_core_num_input_scalar = 0;

  // input data params
  // front core
  params.input_ele_num_front_core_input_scalar = 0;
  // front part front core
  params.input_mov_times_gm2ub_front_part_front_core_input_scalar = 0;
  params.input_front_burst_len_front_part_front_core_input_scalar = 0;
  params.input_last_burst_len_front_part_front_core_input_scalar = 0;
  params.input_front_ele_num_ub_front_part_front_core_input_scalar = 0;
  params.input_last_ele_num_ub_front_part_front_core_input_scalar = 0;
  params.input_front_rows_front_part_front_core_input_scalar = 0;
  params.input_last_rows_front_part_front_core_input_scalar = 0;
  // last part front core
  params.input_mov_times_gm2ub_last_part_front_core_input_scalar = 0;
  params.input_front_burst_len_last_part_front_core_input_scalar = 0;
  params.input_last_burst_len_last_part_front_core_input_scalar = 0;
  params.input_front_ele_num_ub_last_part_front_core_input_scalar = 0;
  params.input_last_ele_num_ub_last_part_front_core_input_scalar = 0;
  params.input_front_rows_last_part_front_core_input_scalar = 0;
  params.input_last_rows_last_part_front_core_input_scalar = 0;
  // last core
  params.input_ele_num_last_core_input_scalar = 0;
  // front part last core
  params.input_mov_times_gm2ub_front_part_last_core_input_scalar = 0;
  params.input_front_burst_len_front_part_last_core_input_scalar = 0;
  params.input_last_burst_len_front_part_last_core_input_scalar = 0;
  params.input_front_ele_num_ub_front_part_last_core_input_scalar = 0;
  params.input_last_ele_num_ub_front_part_last_core_input_scalar = 0;
  params.input_front_rows_front_part_last_core_input_scalar = 0;
  params.input_last_rows_front_part_last_core_input_scalar = 0;
  // last part last core
  params.input_mov_times_gm2ub_last_part_last_core_input_scalar = 0;
  params.input_front_burst_len_last_part_last_core_input_scalar = 0;
  params.input_last_burst_len_last_part_last_core_input_scalar = 0;
  params.input_front_ele_num_ub_last_part_last_core_input_scalar = 0;
  params.input_last_ele_num_ub_last_part_last_core_input_scalar = 0;
  params.input_front_rows_last_part_last_core_input_scalar = 0;
  params.input_last_rows_last_part_last_core_input_scalar = 0;

  // e num params
  params.e_num_input_scalar = 0;
  params.e_mov_times_gm2ub_input_scalar = 0;
  params.e_ub2gm_front_burst_len_input_scalar = 0;
  params.e_num_front_part_input_scalar = 0;
  params.e_ub2gm_last_burst_len_input_scalar = 0;
  params.e_gm2ub_last_burst_len_input_scalar = 0;
  params.e_num_last_part_input_scalar = 0;

  // ids params
  params.ids_size_input_scalar = 0;
  params.ids_ele_num_front_core_input_scalar = 0;
  params.ids_mov_times_gm2ub_front_core_input_scalar = 0;
  params.ids_front_burst_len_front_core_input_scalar = 0;
  params.ids_last_burst_len_front_core_input_scalar = 0;
  params.ids_ele_num_ub_front_part_front_core_input_scalar = 0;
  params.ids_ele_num_ub_last_part_front_core_input_scalar = 0;
  params.ids_ele_num_last_core_input_scalar = 0;
  params.ids_mov_times_gm2ub_last_core_input_scalar = 0;
  params.ids_front_burst_len_last_core_input_scalar = 0;
  params.ids_last_burst_len_last_core_input_scalar = 0;
  params.ids_ele_num_ub_front_part_last_core_input_scalar = 0;
  params.ids_ele_num_ub_last_part_last_core_input_scalar = 0;

  // output init params
  params.output_ub_init_last_repeat_time_front_part_front_core_input_scalar = 0;
  params.output_ub_init_times_front_part_front_core_input_scalar = 0;
  params.output_ub_init_last_repeat_time_last_part_front_core_input_scalar = 0;
  params.output_ub_init_times_last_part_front_core_input_scalar = 0;
  params.output_ub_init_last_repeat_time_front_part_last_core_input_scalar = 0;
  params.output_ub_init_times_front_part_last_core_input_scalar = 0;
  params.output_ub_init_last_repeat_time_last_part_last_core_input_scalar = 0;
  params.output_ub_init_times_last_part_last_core_input_scalar = 0;
  params.input_last_axis_align_front_part_ele_num_input_scalar = 0;
  params.input_last_axis_align_floor_ele_num_input_scalar = 0;
  params.last_part_vadd_mask_input_scalar = 0;
  params.output_ub_init_last_row_last_repeat_time_front_part_front_core_input_scalar = 0;
  params.output_ub_init_last_row_times_front_part_front_core_input_scalar = 0;
  params.output_ub_init_last_row_last_repeat_time_last_part_front_core_input_scalar = 0;
  params.output_ub_init_last_row_times_last_part_front_core_input_scalar = 0;
  params.output_ub_init_last_row_last_repeat_time_front_part_last_core_input_scalar = 0;
  params.output_ub_init_last_row_times_front_part_last_core_input_scalar = 0;
  params.output_ub_init_last_row_last_repeat_time_last_part_last_core_input_scalar = 0;
  params.output_ub_init_last_row_times_last_part_last_core_input_scalar = 0;
}

void InitTilingParams(TilingParamsInt32& params) {
  // common params
  params.select_key_input_scalar = 0;
  params.need_core_num_input_scalar = 0;
  params.num_segments_front_core_input_scalar = 0;
  params.num_segments_last_core_input_scalar = 0;

  // ids params
  params.ids_size_input_scalar = 0;
  params.ids_mov_times_gm2ub_input_scalar = 0;
  params.ids_ele_num_ub_front_part_input_scalar = 0;
  params.ids_front_burst_len_input_scalar = 0;
  params.ids_ele_num_ub_last_part_input_scalar = 0;
  params.ids_last_burst_len_input_scalar = 0;

  // e num params
  params.e_num_input_scalar = 0;
  params.e_mov_times_gm2ub_input_scalar = 0;
  params.e_ub2gm_front_burst_len_input_scalar = 0;
  params.e_num_front_part_input_scalar = 0;
  params.repeat_time_front_part_input_scalar = 0;
  params.e_ub2gm_last_burst_len_input_scalar = 0;
  params.e_num_last_part_input_scalar = 0;
  params.repeat_time_last_part_input_scalar = 0;
  params.align_scalar = 0;
  params.align_scalar_lastcore = 0;
  params.e_gm2ub_front_burst_len_input_scalar = 0;
  params.e_gm2ub_last_burst_len_input_scalar = 0;
  params.num_segment_max = 0;
  params.num_segment_max_time = 0;
  params.num_segment_max_time_lastcore = 0;
  params.front_num_segment = 0;
  params.front_num_segment_last = 0;
  params.front_num_segment_lastcore = 0;
  params.front_num_segment_last_lastcore = 0;
  params.e_ub2gm_front_burst_len_input_scalar_lastcore = 0;
  params.e_ub2gm_last_burst_len_input_scalar_lastcore = 0;
  params.repeat_times = 0;
  params.repeat_times_last_part = 0;
  params.repeat_times_last_part_lastcore = 0;
  params.e_mov_times_gm2ub_input_scalar_lastcore = 0;
  params.repeat_time_front_part_input_scalar_lastcore = 0;

}

void WriteTilingParams(const TilingParamsFp32& params, OpRunInfo& run_info) {
  // common params
  ByteBufferPut(run_info.tiling_data, params.select_key_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.need_core_num_input_scalar);

  // input data params
  // front core
  ByteBufferPut(run_info.tiling_data, params.input_ele_num_front_core_input_scalar);
  // front part front core
  ByteBufferPut(run_info.tiling_data, params.input_mov_times_gm2ub_front_part_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_front_burst_len_front_part_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_last_burst_len_front_part_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_front_ele_num_ub_front_part_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_last_ele_num_ub_front_part_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_front_rows_front_part_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_last_rows_front_part_front_core_input_scalar);
  // last part front core
  ByteBufferPut(run_info.tiling_data, params.input_mov_times_gm2ub_last_part_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_front_burst_len_last_part_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_last_burst_len_last_part_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_front_ele_num_ub_last_part_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_last_ele_num_ub_last_part_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_front_rows_last_part_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_last_rows_last_part_front_core_input_scalar);
  // last core
  ByteBufferPut(run_info.tiling_data, params.input_ele_num_last_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_mov_times_gm2ub_front_part_last_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_front_burst_len_front_part_last_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_last_burst_len_front_part_last_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_front_ele_num_ub_front_part_last_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_last_ele_num_ub_front_part_last_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_front_rows_front_part_last_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_last_rows_front_part_last_core_input_scalar);
  // last part last core
  ByteBufferPut(run_info.tiling_data, params.input_mov_times_gm2ub_last_part_last_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_front_burst_len_last_part_last_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_last_burst_len_last_part_last_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_front_ele_num_ub_last_part_last_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_last_ele_num_ub_last_part_last_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_front_rows_last_part_last_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_last_rows_last_part_last_core_input_scalar);

  // e num params
  ByteBufferPut(run_info.tiling_data, params.e_num_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.e_mov_times_gm2ub_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.e_ub2gm_front_burst_len_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.e_num_front_part_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.e_ub2gm_last_burst_len_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.e_num_last_part_input_scalar);

  // ids params
  ByteBufferPut(run_info.tiling_data, params.ids_size_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.ids_ele_num_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.ids_mov_times_gm2ub_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.ids_front_burst_len_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.ids_last_burst_len_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.ids_ele_num_ub_front_part_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.ids_ele_num_ub_last_part_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.ids_ele_num_last_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.ids_mov_times_gm2ub_last_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.ids_front_burst_len_last_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.ids_last_burst_len_last_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.ids_ele_num_ub_front_part_last_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.ids_ele_num_ub_last_part_last_core_input_scalar);

  // output init params
  ByteBufferPut(run_info.tiling_data, params.output_ub_init_last_repeat_time_front_part_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.output_ub_init_times_front_part_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.output_ub_init_last_repeat_time_last_part_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.output_ub_init_times_last_part_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.output_ub_init_last_repeat_time_front_part_last_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.output_ub_init_times_front_part_last_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.output_ub_init_last_repeat_time_last_part_last_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.output_ub_init_times_last_part_last_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_last_axis_align_front_part_ele_num_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.input_last_axis_align_floor_ele_num_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.last_part_vadd_mask_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.e_gm2ub_last_burst_len_input_scalar);
  ByteBufferPut(run_info.tiling_data,
                params.output_ub_init_last_row_last_repeat_time_front_part_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.output_ub_init_last_row_times_front_part_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data,
                params.output_ub_init_last_row_last_repeat_time_last_part_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.output_ub_init_last_row_times_last_part_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data,
                params.output_ub_init_last_row_last_repeat_time_front_part_last_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.output_ub_init_last_row_times_front_part_last_core_input_scalar);
  ByteBufferPut(run_info.tiling_data,
                params.output_ub_init_last_row_last_repeat_time_last_part_last_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.output_ub_init_last_row_times_last_part_last_core_input_scalar);

}

void WriteTilingParams(const TilingParamsInt32& params, OpRunInfo& run_info) {
  // common params
  ByteBufferPut(run_info.tiling_data, params.select_key_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.need_core_num_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.num_segments_front_core_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.num_segments_last_core_input_scalar);

  // ids params
  ByteBufferPut(run_info.tiling_data, params.ids_size_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.ids_mov_times_gm2ub_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.ids_ele_num_ub_front_part_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.ids_front_burst_len_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.ids_ele_num_ub_last_part_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.ids_last_burst_len_input_scalar);

  // e num params
  ByteBufferPut(run_info.tiling_data, params.e_num_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.e_mov_times_gm2ub_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.e_ub2gm_front_burst_len_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.e_num_front_part_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.repeat_time_front_part_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.e_ub2gm_last_burst_len_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.e_num_last_part_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.repeat_time_last_part_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.align_scalar);
  ByteBufferPut(run_info.tiling_data, params.align_scalar_lastcore);
  ByteBufferPut(run_info.tiling_data, params.e_gm2ub_front_burst_len_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.e_gm2ub_last_burst_len_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.num_segment_max);
  ByteBufferPut(run_info.tiling_data, params.num_segment_max_time);
  ByteBufferPut(run_info.tiling_data, params.num_segment_max_time_lastcore);
  ByteBufferPut(run_info.tiling_data, params.front_num_segment);
  ByteBufferPut(run_info.tiling_data, params.front_num_segment_last);
  ByteBufferPut(run_info.tiling_data, params.front_num_segment_lastcore);
  ByteBufferPut(run_info.tiling_data, params.front_num_segment_last_lastcore);
  ByteBufferPut(run_info.tiling_data, params.e_ub2gm_front_burst_len_input_scalar_lastcore);
  ByteBufferPut(run_info.tiling_data, params.e_ub2gm_last_burst_len_input_scalar_lastcore);
  ByteBufferPut(run_info.tiling_data, params.repeat_times);
  ByteBufferPut(run_info.tiling_data, params.repeat_times_last_part);
  ByteBufferPut(run_info.tiling_data, params.repeat_times_last_part_lastcore);
  ByteBufferPut(run_info.tiling_data, params.e_mov_times_gm2ub_input_scalar_lastcore);
  ByteBufferPut(run_info.tiling_data, params.repeat_time_front_part_input_scalar_lastcore);
}

void PrintTilingParams(const std::string& op_type, const TilingParamsFp32& params) {
  GELOGD("op [%s] : params.select_key_input_scalar=%d", op_type.c_str(), params.select_key_input_scalar);
  GELOGD("op [%s] : params.need_core_num_input_scalar=%d", op_type.c_str(), params.need_core_num_input_scalar);
  GELOGD("op [%s] : params.input_ele_num_front_core_input_scalar=%d", op_type.c_str(),
         params.input_ele_num_front_core_input_scalar);
  GELOGD("op [%s] : params.input_mov_times_gm2ub_front_part_front_core_input_scalar=%d", op_type.c_str(),
         params.input_mov_times_gm2ub_front_part_front_core_input_scalar);
  GELOGD("op [%s] : params.input_front_burst_len_front_part_front_core_input_scalar=%d", op_type.c_str(),
         params.input_front_burst_len_front_part_front_core_input_scalar);
  GELOGD("op [%s] : params.input_last_burst_len_front_part_front_core_input_scalar=%d", op_type.c_str(),
         params.input_last_burst_len_front_part_front_core_input_scalar);
  GELOGD("op [%s] : params.input_front_ele_num_ub_front_part_front_core_input_scalar=%d", op_type.c_str(),
         params.input_front_ele_num_ub_front_part_front_core_input_scalar);
  GELOGD("op [%s] : params.input_last_ele_num_ub_front_part_front_core_input_scalar=%d", op_type.c_str(),
         params.input_last_ele_num_ub_front_part_front_core_input_scalar);
  GELOGD("op [%s] : params.input_front_rows_front_part_front_core_input_scalar=%d", op_type.c_str(),
         params.input_front_rows_front_part_front_core_input_scalar);
  GELOGD("op [%s] : params.input_last_rows_front_part_front_core_input_scalar=%d", op_type.c_str(),
         params.input_last_rows_front_part_front_core_input_scalar);
  GELOGD("op [%s] : params.input_mov_times_gm2ub_last_part_front_core_input_scalar=%d", op_type.c_str(),
         params.input_mov_times_gm2ub_last_part_front_core_input_scalar);
  GELOGD("op [%s] : params.input_front_burst_len_last_part_front_core_input_scalar=%d", op_type.c_str(),
         params.input_front_burst_len_last_part_front_core_input_scalar);
  GELOGD("op [%s] : params.input_last_burst_len_last_part_front_core_input_scalar=%d", op_type.c_str(),
         params.input_last_burst_len_last_part_front_core_input_scalar);
  GELOGD("op [%s] : params.input_front_ele_num_ub_last_part_front_core_input_scalar=%d", op_type.c_str(),
         params.input_front_ele_num_ub_last_part_front_core_input_scalar);
  GELOGD("op [%s] : params.input_last_ele_num_ub_last_part_front_core_input_scalar=%d", op_type.c_str(),
         params.input_last_ele_num_ub_last_part_front_core_input_scalar);
  GELOGD("op [%s] : params.input_front_rows_last_part_front_core_input_scalar=%d", op_type.c_str(),
         params.input_front_rows_last_part_front_core_input_scalar);
  GELOGD("op [%s] : params.input_last_rows_last_part_front_core_input_scalar=%d", op_type.c_str(),
         params.input_last_rows_last_part_front_core_input_scalar);
  GELOGD("op [%s] : params.input_ele_num_last_core_input_scalar=%d", op_type.c_str(),
         params.input_ele_num_last_core_input_scalar);
  GELOGD("op [%s] : params.input_mov_times_gm2ub_front_part_last_core_input_scalar=%d", op_type.c_str(),
         params.input_mov_times_gm2ub_front_part_last_core_input_scalar);
  GELOGD("op [%s] : params.input_front_burst_len_front_part_last_core_input_scalar=%d", op_type.c_str(),
         params.input_front_burst_len_front_part_last_core_input_scalar);
  GELOGD("op [%s] : params.input_last_burst_len_front_part_last_core_input_scalar=%d", op_type.c_str(),
         params.input_last_burst_len_front_part_last_core_input_scalar);
  GELOGD("op [%s] : params.input_front_ele_num_ub_front_part_last_core_input_scalar=%d", op_type.c_str(),
         params.input_front_ele_num_ub_front_part_last_core_input_scalar);
  GELOGD("op [%s] : params.input_last_ele_num_ub_front_part_last_core_input_scalar=%d", op_type.c_str(),
         params.input_last_ele_num_ub_front_part_last_core_input_scalar);
  GELOGD("op [%s] : params.input_front_rows_front_part_last_core_input_scalar=%d", op_type.c_str(),
         params.input_front_rows_front_part_last_core_input_scalar);
  GELOGD("op [%s] : params.input_last_rows_front_part_last_core_input_scalar=%d", op_type.c_str(),
         params.input_last_rows_front_part_last_core_input_scalar);
  GELOGD("op [%s] : params.input_mov_times_gm2ub_last_part_last_core_input_scalar=%d", op_type.c_str(),
         params.input_mov_times_gm2ub_last_part_last_core_input_scalar);
  GELOGD("op [%s] : params.input_front_burst_len_last_part_last_core_input_scalar=%d", op_type.c_str(),
         params.input_front_burst_len_last_part_last_core_input_scalar);
  GELOGD("op [%s] : params.input_last_burst_len_last_part_last_core_input_scalar=%d", op_type.c_str(),
         params.input_last_burst_len_last_part_last_core_input_scalar);
  GELOGD("op [%s] : params.input_front_ele_num_ub_last_part_last_core_input_scalar=%d", op_type.c_str(),
         params.input_front_ele_num_ub_last_part_last_core_input_scalar);
  GELOGD("op [%s] : params.input_last_ele_num_ub_last_part_last_core_input_scalar=%d", op_type.c_str(),
         params.input_last_ele_num_ub_last_part_last_core_input_scalar);
  GELOGD("op [%s] : params.input_front_rows_last_part_last_core_input_scalar=%d", op_type.c_str(),
         params.input_front_rows_last_part_last_core_input_scalar);
  GELOGD("op [%s] : params.input_last_rows_last_part_last_core_input_scalar=%d", op_type.c_str(),
         params.input_last_rows_last_part_last_core_input_scalar);
  GELOGD("op [%s] : params.e_num_input_scalar=%d", op_type.c_str(), params.e_num_input_scalar);
  GELOGD("op [%s] : params.e_mov_times_gm2ub_input_scalar=%d", op_type.c_str(), params.e_mov_times_gm2ub_input_scalar);
  GELOGD("op [%s] : params.e_ub2gm_front_burst_len_input_scalar=%d", op_type.c_str(),
         params.e_ub2gm_front_burst_len_input_scalar);
  GELOGD("op [%s] : params.e_num_front_part_input_scalar=%d", op_type.c_str(), params.e_num_front_part_input_scalar);
  GELOGD("op [%s] : params.e_ub2gm_last_burst_len_input_scalar=%d", op_type.c_str(),
         params.e_ub2gm_last_burst_len_input_scalar);
  GELOGD("op [%s] : params.e_num_last_part_input_scalar=%d", op_type.c_str(), params.e_num_last_part_input_scalar);
  GELOGD("op [%s] : params.ids_size_input_scalar=%d", op_type.c_str(), params.ids_size_input_scalar);
  GELOGD("op [%s] : params.ids_ele_num_front_core_input_scalar=%d", op_type.c_str(),
         params.ids_ele_num_front_core_input_scalar);
  GELOGD("op [%s] : params.ids_mov_times_gm2ub_front_core_input_scalar=%d", op_type.c_str(),
         params.ids_mov_times_gm2ub_front_core_input_scalar);
  GELOGD("op [%s] : params.ids_front_burst_len_front_core_input_scalar=%d", op_type.c_str(),
         params.ids_front_burst_len_front_core_input_scalar);
  GELOGD("op [%s] : params.ids_last_burst_len_front_core_input_scalar=%d", op_type.c_str(),
         params.ids_last_burst_len_front_core_input_scalar);
  GELOGD("op [%s] : params.ids_ele_num_ub_front_part_front_core_input_scalar=%d", op_type.c_str(),
         params.ids_ele_num_ub_front_part_front_core_input_scalar);
  GELOGD("op [%s] : params.ids_ele_num_ub_last_part_front_core_input_scalar=%d", op_type.c_str(),
         params.ids_ele_num_ub_last_part_front_core_input_scalar);
  GELOGD("op [%s] : params.ids_ele_num_last_core_input_scalar=%d", op_type.c_str(),
         params.ids_ele_num_last_core_input_scalar);
  GELOGD("op [%s] : params.ids_mov_times_gm2ub_last_core_input_scalar=%d", op_type.c_str(),
         params.ids_mov_times_gm2ub_last_core_input_scalar);
  GELOGD("op [%s] : params.ids_front_burst_len_last_core_input_scalar=%d", op_type.c_str(),
         params.ids_front_burst_len_last_core_input_scalar);
  GELOGD("op [%s] : params.ids_last_burst_len_last_core_input_scalar=%d", op_type.c_str(),
         params.ids_last_burst_len_last_core_input_scalar);
  GELOGD("op [%s] : params.ids_ele_num_ub_front_part_last_core_input_scalar=%d", op_type.c_str(),
         params.ids_ele_num_ub_front_part_last_core_input_scalar);
  GELOGD("op [%s] : params.ids_ele_num_ub_last_part_last_core_input_scalar=%d", op_type.c_str(),
         params.ids_ele_num_ub_last_part_last_core_input_scalar);
  GELOGD("op [%s] : params.output_ub_init_last_repeat_time_front_part_front_core_input_scalar=%d", op_type.c_str(),
         params.output_ub_init_last_repeat_time_front_part_front_core_input_scalar);
  GELOGD("op [%s] : params.output_ub_init_times_front_part_front_core_input_scalar=%d", op_type.c_str(),
         params.output_ub_init_times_front_part_front_core_input_scalar);
  GELOGD("op [%s] : params.output_ub_init_last_repeat_time_last_part_front_core_input_scalar=%d", op_type.c_str(),
         params.output_ub_init_last_repeat_time_last_part_front_core_input_scalar);
  GELOGD("op [%s] : params.output_ub_init_times_last_part_front_core_input_scalar=%d", op_type.c_str(),
         params.output_ub_init_times_last_part_front_core_input_scalar);
  GELOGD("op [%s] : params.output_ub_init_last_repeat_time_front_part_last_core_input_scalar=%d", op_type.c_str(),
         params.output_ub_init_last_repeat_time_front_part_last_core_input_scalar);
  GELOGD("op [%s] : params.output_ub_init_times_front_part_last_core_input_scalar=%d", op_type.c_str(),
         params.output_ub_init_times_front_part_last_core_input_scalar);
  GELOGD("op [%s] : params.output_ub_init_last_repeat_time_last_part_last_core_input_scalar=%d", op_type.c_str(),
         params.output_ub_init_last_repeat_time_last_part_last_core_input_scalar);
  GELOGD("op [%s] : params.output_ub_init_times_last_part_last_core_input_scalar=%d", op_type.c_str(),
         params.output_ub_init_times_last_part_last_core_input_scalar);
  GELOGD("op [%s] : params.input_last_axis_align_front_part_ele_num_input_scalar=%d", op_type.c_str(),
         params.input_last_axis_align_front_part_ele_num_input_scalar);
  GELOGD("op [%s] : params.input_last_axis_align_floor_ele_num_input_scalar=%d", op_type.c_str(),
         params.input_last_axis_align_floor_ele_num_input_scalar);
  GELOGD("op [%s] : params.last_part_vadd_mask_input_scalar=%d", op_type.c_str(),
         params.last_part_vadd_mask_input_scalar);
  GELOGD("op [%s] : params.e_gm2ub_last_burst_len_input_scalar=%d", op_type.c_str(),
         params.e_gm2ub_last_burst_len_input_scalar);

  GELOGD("op [%s] : params.output_ub_init_last_row_last_repeat_time_front_part_front_core_input_scalar=%d",
         op_type.c_str(), params.output_ub_init_last_row_last_repeat_time_front_part_front_core_input_scalar);
  GELOGD("op [%s] : params.output_ub_init_last_row_times_front_part_front_core_input_scalar=%d", op_type.c_str(),
         params.output_ub_init_last_row_times_front_part_front_core_input_scalar);
  GELOGD("op [%s] : params.output_ub_init_last_row_last_repeat_time_last_part_front_core_input_scalar=%d",
         op_type.c_str(), params.output_ub_init_last_row_last_repeat_time_last_part_front_core_input_scalar);
  GELOGD("op [%s] : params.output_ub_init_last_row_times_last_part_front_core_input_scalar=%d", op_type.c_str(),
         params.output_ub_init_last_row_times_last_part_front_core_input_scalar);
  GELOGD("op [%s] : params.output_ub_init_last_row_last_repeat_time_front_part_last_core_input_scalar=%d",
         op_type.c_str(), params.output_ub_init_last_row_last_repeat_time_front_part_last_core_input_scalar);
  GELOGD("op [%s] : params.output_ub_init_last_row_times_front_part_last_core_input_scalar=%d", op_type.c_str(),
         params.output_ub_init_last_row_times_front_part_last_core_input_scalar);
  GELOGD("op [%s] : params.output_ub_init_last_row_last_repeat_time_last_part_last_core_input_scalar=%d",
         op_type.c_str(), params.output_ub_init_last_row_last_repeat_time_last_part_last_core_input_scalar);
  GELOGD("op [%s] : params.output_ub_init_last_row_times_last_part_last_core_input_scalar=%d", op_type.c_str(),
         params.output_ub_init_last_row_times_last_part_last_core_input_scalar);
}

void PrintTilingParams(const std::string& op_type, const TilingParamsInt32& params) {
  GELOGD("op [%s] : params.select_key_input_scalar=%d", op_type.c_str(), params.select_key_input_scalar);
  GELOGD("op [%s] : params.need_core_num_input_scalar=%d", op_type.c_str(), params.need_core_num_input_scalar);
  GELOGD("op [%s] : params.num_segments_front_core_input_scalar=%d", op_type.c_str(),
         params.num_segments_front_core_input_scalar);
  GELOGD("op [%s] : params.num_segments_last_core_input_scalar=%d", op_type.c_str(),
         params.num_segments_last_core_input_scalar);
  GELOGD("op [%s] : params.ids_size_input_scalar=%d", op_type.c_str(), params.ids_size_input_scalar);
  GELOGD("op [%s] : params.ids_mov_times_gm2ub_input_scalar=%d", op_type.c_str(),
         params.ids_mov_times_gm2ub_input_scalar);
  GELOGD("op [%s] : params.ids_ele_num_ub_front_part_input_scalar=%d", op_type.c_str(),
         params.ids_ele_num_ub_front_part_input_scalar);
  GELOGD("op [%s] : params.ids_front_burst_len_input_scalar=%d", op_type.c_str(),
         params.ids_front_burst_len_input_scalar);
  GELOGD("op [%s] : params.ids_ele_num_ub_last_part_input_scalar=%d", op_type.c_str(),
         params.ids_ele_num_ub_last_part_input_scalar);
  GELOGD("op [%s] : params.ids_last_burst_len_input_scalar=%d", op_type.c_str(),
         params.ids_last_burst_len_input_scalar);
  GELOGD("op [%s] : params.e_num_input_scalar=%d", op_type.c_str(), params.e_num_input_scalar);
  GELOGD("op [%s] : params.e_mov_times_gm2ub_input_scalar=%d", op_type.c_str(), params.e_mov_times_gm2ub_input_scalar);
  GELOGD("op [%s] : params.e_ub2gm_front_burst_len_input_scalar=%d", op_type.c_str(),
         params.e_ub2gm_front_burst_len_input_scalar);
  GELOGD("op [%s] : params.e_num_front_part_input_scalar=%d", op_type.c_str(), params.e_num_front_part_input_scalar);
  GELOGD("op [%s] : params.repeat_time_front_part_input_scalar=%d", op_type.c_str(),
         params.repeat_time_front_part_input_scalar);
  GELOGD("op [%s] : params.e_ub2gm_last_burst_len_input_scalar=%d", op_type.c_str(),
         params.e_ub2gm_last_burst_len_input_scalar);
  GELOGD("op [%s] : params.e_num_last_part_input_scalar=%d", op_type.c_str(), params.e_num_last_part_input_scalar);
  GELOGD("op [%s] : params.repeat_time_last_part_input_scalar=%d", op_type.c_str(),
         params.repeat_time_last_part_input_scalar);

  GELOGD("op [%s] : params.align_scalar=%d", op_type.c_str(),params.align_scalar);
  GELOGD("op [%s] : params.align_scalar_lastcore=%d", op_type.c_str(),
         params.align_scalar_lastcore);
  GELOGD("op [%s] : params.e_gm2ub_front_burst_len_input_scalar=%d", op_type.c_str(),
         params.e_gm2ub_front_burst_len_input_scalar);
  GELOGD("op [%s] : params.e_gm2ub_last_burst_len_input_scalar=%d", op_type.c_str(),
         params.e_gm2ub_last_burst_len_input_scalar);
  GELOGD("op [%s] : params.num_segment_max=%d", op_type.c_str(),
         params.num_segment_max);
  GELOGD("op [%s] : params.num_segment_max_time=%d", op_type.c_str(),
         params.num_segment_max_time);
  GELOGD("op [%s] : params.num_segment_max_time_lastcore=%d", op_type.c_str(),
         params.num_segment_max_time_lastcore);
  GELOGD("op [%s] : params.front_num_segment=%d", op_type.c_str(),
         params.front_num_segment);
  GELOGD("op [%s] : params.front_num_segment_last=%d", op_type.c_str(),
         params.front_num_segment_last);
  GELOGD("op [%s] : params.front_num_segment_lastcore=%d", op_type.c_str(),
         params.front_num_segment_lastcore);
  GELOGD("op [%s] : params.front_num_segment_last_lastcore=%d", op_type.c_str(),
         params.front_num_segment_last_lastcore);
  GELOGD("op [%s] : params.e_ub2gm_front_burst_len_input_scalar_lastcore=%d", op_type.c_str(),
         params.e_ub2gm_front_burst_len_input_scalar_lastcore);
  GELOGD("op [%s] : params.e_ub2gm_last_burst_len_input_scalar_lastcore=%d", op_type.c_str(),
         params.e_ub2gm_last_burst_len_input_scalar_lastcore);
  GELOGD("op [%s] : params.repeat_times=%d", op_type.c_str(), params.repeat_times);
  GELOGD("op [%s] : params.repeat_times_last_part=%d", op_type.c_str(), params.repeat_times_last_part);
  GELOGD("op [%s] : params.repeat_times_last_part_lastcore=%d", op_type.c_str(),
         params.repeat_times_last_part_lastcore);
  GELOGD("op [%s] : params.e_mov_times_gm2ub_input_scalar_lastcore=%d", op_type.c_str(),
         params.e_mov_times_gm2ub_input_scalar_lastcore);
  GELOGD("op [%s] : params.repeat_time_front_part_input_scalar_lastcore=%d", op_type.c_str(),
         params.repeat_time_front_part_input_scalar_lastcore);

}

// tiling function
bool UnsortedSegmentSumTiling(const std::string& op_type, const TeOpParas& op_paras,
                              const nlohmann::json& op_compile_info_json, OpRunInfo& run_info) {
  GELOGI("op[%s] op tiling begin.", op_type.c_str());
  if (op_paras.inputs.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs is empty.");
    return false;
  }
  if (op_paras.inputs.size() < 2) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs.size() < 2.");
    return false;
  }
  if (op_paras.inputs[0].tensor.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input_data tensor is empty.");
    return false;
  }
  if (op_paras.inputs[1].tensor.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ids tensor is empty.");
    return false;
  }
  const std::vector<int64_t>& input_shape = op_paras.inputs[0].tensor[0].shape;
  const std::vector<int64_t>& ids_shape = op_paras.inputs[1].tensor[0].shape;
  const int32_t& input_size = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int>());
  const int32_t& ids_size = std::accumulate(ids_shape.begin(), ids_shape.end(), 1, std::multiplies<int>());
  GELOGD("op [%s] : input_size=%d, ids_size=%d", op_type.c_str(), input_size, ids_size);
  if (input_shape.size() < ids_shape.size()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "dim of input must be greater than or equal with dim of ids");
    return false;
  }
  for (unsigned i = 0; i < ids_shape.size(); i++) {
    GELOGD("op[%s] ids_shape[i] is %d",op_type.c_str(),ids_shape[i]);
    if (input_shape[i] != ids_shape[i]) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "front shape of input must be equal with ids shape");
      return false;
    }
  }
  std::string key_num_segments = "num_segments";
  if (op_paras.const_inputs.find(key_num_segments) == op_paras.const_inputs.end()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "num_segments not exists.");
    return false;
  }
  const int32_t* num_segments_ptr =
    reinterpret_cast<const int32_t*>(std::get<0>(op_paras.const_inputs.at(key_num_segments)));
  int32_t num_segments = *num_segments_ptr;
  if (num_segments <= 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "num_segments is small then 0");
    return false;
  }
  GELOGD("op [%s] : num_segments=%d", op_type.c_str(), num_segments);
  if (input_size == 0 || ids_size == 0) {
    TilingParamsFp32 params;
    InitTilingParams(params);
    params.select_key_input_scalar = 0;
    params.need_core_num_input_scalar = 1;
    WriteTilingParams(params, run_info);
    // cout tiling params
    PrintTilingParams(op_type, params);
    // BlockDim, core num used in tik op
    run_info.block_dim = params.need_core_num_input_scalar;
    // workspace, null for tik op
    std::vector<int64_t> workspace;
    run_info.workspaces = workspace;
    return true;
  }
  int32_t e_size = input_size / ids_size;
  GELOGD("op[%s] e_size is %d",op_type.c_str(), e_size);
  const std::string& input_dtype = op_paras.inputs[0].tensor[0].dtype;
  const std::string& ids_dtype = op_paras.inputs[1].tensor[0].dtype;
  bool flag = false;
  // get input dtype
  EleByte input_ele_byte = FP32_BYTE;
  flag = GetEleDtype(input_dtype, input_ele_byte);
  if (!flag) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_ele_byte failed.");
    return false;
  }

  EleByte output_ele_byte = input_ele_byte;
  int32_t output_ub_ele_num_one_row = BYTE_BLOCK / output_ele_byte;
  GELOGD("op[%s] output_ub_ele_num_one_row is %d",op_type.c_str(), output_ub_ele_num_one_row);
  // get ids dtype
  EleByte ids_ele_byte = FP32_BYTE;
  flag = GetEleDtype(ids_dtype, ids_ele_byte);
  if (!flag) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get ids_ele_byte failed.");
    return false;
  }
  // get compile info
  int32_t core_num = 1;
  int32_t ub_size = 256 * 1024;
  int32_t ub_tensor_num = 0;
  flag = GetUssCompileParams(op_type, op_compile_info_json, core_num, ub_size, ub_tensor_num);
  if (!flag) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetCompileParams failed.");
    return false;
  }

  if (input_dtype == DTYPE_FP32) {
    int32_t ub_tensor_size = 0;
    ComputeUbTensorSize(ub_size, input_shape, input_dtype, e_size, ub_tensor_size, num_segments);
    if (e_size == 1) {
      ub_tensor_size = ub_tensor_size / BYTE_FULL_MASK * BYTE_FULL_MASK;
    }
    GELOGD("op [%s] : ub_tensor_size=%d", op_type.c_str(), ub_tensor_size);
    int32_t ub_tensor_ele_num = ub_tensor_size / input_ele_byte;
    // fp32 tiling params
    TilingParamsFp32 params;
    InitTilingParams(params);
    // select key
    flag = GetTilingMode(input_shape, e_size, input_dtype, ub_tensor_ele_num, params.select_key_input_scalar,
                         num_segments);
    if (!flag) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetTilingMode failed.");
      return false;
    }
    params.e_num_input_scalar = e_size;
    // ids params compute is common
    params.ids_size_input_scalar = ids_size;
    int32_t ids_min_ele_num = BYTE_BLOCK / ids_ele_byte;
    // is using all core
    flag = IsUsingAllCore(ids_size, core_num, params.need_core_num_input_scalar, e_size, num_segments);
    ComputeEleNumOneCore(ids_min_ele_num, ids_size, params.need_core_num_input_scalar, e_size,
                         params.ids_ele_num_front_core_input_scalar, params.ids_ele_num_last_core_input_scalar,
                         params.input_ele_num_front_core_input_scalar, params.input_ele_num_last_core_input_scalar,
                         num_segments);
    // ids params front core
    ComputeIdsParamsMovGm2ub(
      params.ids_ele_num_front_core_input_scalar, ub_tensor_size, ids_ele_byte,
      params.ids_mov_times_gm2ub_front_core_input_scalar, params.ids_front_burst_len_front_core_input_scalar,
      params.ids_last_burst_len_front_core_input_scalar, params.ids_ele_num_ub_front_part_front_core_input_scalar,
      params.ids_ele_num_ub_last_part_front_core_input_scalar);
    // ids params last core
    ComputeIdsParamsMovGm2ub(
      params.ids_ele_num_last_core_input_scalar, ub_tensor_size, ids_ele_byte,
      params.ids_mov_times_gm2ub_last_core_input_scalar, params.ids_front_burst_len_last_core_input_scalar,
      params.ids_last_burst_len_last_core_input_scalar, params.ids_ele_num_ub_front_part_last_core_input_scalar,
      params.ids_ele_num_ub_last_part_last_core_input_scalar);

    if (params.select_key_input_scalar == SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ALIGN_SMALL_E) {
      // aign small e

      // e num params
      params.e_mov_times_gm2ub_input_scalar = 1;
      params.e_num_front_part_input_scalar = params.e_num_input_scalar;
      params.e_num_last_part_input_scalar = params.e_num_input_scalar;
      params.e_ub2gm_front_burst_len_input_scalar = params.e_num_front_part_input_scalar * input_ele_byte / BYTE_BLOCK;
      params.e_ub2gm_last_burst_len_input_scalar = params.e_ub2gm_front_burst_len_input_scalar;

      // input data params
      // front part front core
      ComputeInputParamsMovGm2ub(ub_tensor_size, input_ele_byte, params.input_ele_num_front_core_input_scalar, e_size,
                                 params.e_mov_times_gm2ub_input_scalar,
                                 params.ids_ele_num_ub_front_part_front_core_input_scalar,
                                 params.input_mov_times_gm2ub_front_part_front_core_input_scalar,
                                 params.input_front_burst_len_front_part_front_core_input_scalar,
                                 params.input_last_burst_len_front_part_front_core_input_scalar,
                                 params.input_front_ele_num_ub_front_part_front_core_input_scalar,
                                 params.input_last_ele_num_ub_front_part_front_core_input_scalar,
                                 params.input_front_rows_front_part_front_core_input_scalar,
                                 params.input_last_rows_front_part_front_core_input_scalar);
      // last part front core
      ComputeInputParamsMovGm2ub(ub_tensor_size, input_ele_byte, params.input_ele_num_front_core_input_scalar, e_size,
                                 params.e_mov_times_gm2ub_input_scalar,
                                 params.ids_ele_num_ub_last_part_front_core_input_scalar,
                                 params.input_mov_times_gm2ub_last_part_front_core_input_scalar,
                                 params.input_front_burst_len_last_part_front_core_input_scalar,
                                 params.input_last_burst_len_last_part_front_core_input_scalar,
                                 params.input_front_ele_num_ub_last_part_front_core_input_scalar,
                                 params.input_last_ele_num_ub_last_part_front_core_input_scalar,
                                 params.input_front_rows_last_part_front_core_input_scalar,
                                 params.input_last_rows_last_part_front_core_input_scalar);
      // front part last core
      ComputeInputParamsMovGm2ub(ub_tensor_size, input_ele_byte, params.input_ele_num_last_core_input_scalar, e_size,
                                 params.e_mov_times_gm2ub_input_scalar,
                                 params.ids_ele_num_ub_front_part_last_core_input_scalar,
                                 params.input_mov_times_gm2ub_front_part_last_core_input_scalar,
                                 params.input_front_burst_len_front_part_last_core_input_scalar,
                                 params.input_last_burst_len_front_part_last_core_input_scalar,
                                 params.input_front_ele_num_ub_front_part_last_core_input_scalar,
                                 params.input_last_ele_num_ub_front_part_last_core_input_scalar,
                                 params.input_front_rows_front_part_last_core_input_scalar,
                                 params.input_last_rows_front_part_last_core_input_scalar);
      // last part last core
      ComputeInputParamsMovGm2ub(ub_tensor_size, input_ele_byte, params.input_ele_num_last_core_input_scalar, e_size,
                                 params.e_mov_times_gm2ub_input_scalar,
                                 params.ids_ele_num_ub_last_part_last_core_input_scalar,
                                 params.input_mov_times_gm2ub_last_part_last_core_input_scalar,
                                 params.input_front_burst_len_last_part_last_core_input_scalar,
                                 params.input_last_burst_len_last_part_last_core_input_scalar,
                                 params.input_front_ele_num_ub_last_part_last_core_input_scalar,
                                 params.input_last_ele_num_ub_last_part_last_core_input_scalar,
                                 params.input_front_rows_last_part_last_core_input_scalar,
                                 params.input_last_rows_last_part_last_core_input_scalar);
    } else if (params.select_key_input_scalar == SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ONE) {
      // last axis is one

      // e num params
      params.e_num_input_scalar = 1;
      params.e_mov_times_gm2ub_input_scalar = 1;
      params.e_ub2gm_front_burst_len_input_scalar = 1;

      // input data params
      // front part front core
      params.input_front_burst_len_front_part_front_core_input_scalar =
        params.ids_front_burst_len_front_core_input_scalar;
      params.input_front_ele_num_ub_front_part_front_core_input_scalar =
        params.ids_ele_num_ub_front_part_front_core_input_scalar;
      params.input_front_rows_front_part_front_core_input_scalar =
        params.input_front_ele_num_ub_front_part_front_core_input_scalar;
      // last part front core
      params.input_front_burst_len_last_part_front_core_input_scalar =
        params.ids_last_burst_len_front_core_input_scalar;
      params.input_front_ele_num_ub_last_part_front_core_input_scalar =
        params.ids_ele_num_ub_last_part_front_core_input_scalar;
      params.input_front_rows_last_part_front_core_input_scalar =
        params.input_front_ele_num_ub_last_part_front_core_input_scalar;
      // front part last core
      params.input_front_burst_len_front_part_last_core_input_scalar =
        params.ids_front_burst_len_last_core_input_scalar;
      params.input_front_ele_num_ub_front_part_last_core_input_scalar =
        params.ids_ele_num_ub_front_part_last_core_input_scalar;
      params.input_front_rows_front_part_last_core_input_scalar =
        params.input_front_ele_num_ub_front_part_last_core_input_scalar;
      // last part last core
      params.input_front_burst_len_last_part_last_core_input_scalar = params.ids_last_burst_len_last_core_input_scalar;
      params.input_front_ele_num_ub_last_part_last_core_input_scalar =
        params.ids_ele_num_ub_last_part_last_core_input_scalar;
      params.input_front_rows_last_part_last_core_input_scalar =
        params.input_front_ele_num_ub_last_part_last_core_input_scalar;

      // output init params
      // front part front core
      ComputeInitOutputUbParams(params.ids_ele_num_ub_front_part_front_core_input_scalar, output_ub_ele_num_one_row,
                                params.output_ub_init_last_repeat_time_front_part_front_core_input_scalar,
                                params.output_ub_init_times_front_part_front_core_input_scalar);
      // last part front core
      ComputeInitOutputUbParams(params.ids_ele_num_ub_last_part_front_core_input_scalar, output_ub_ele_num_one_row,
                                params.output_ub_init_last_repeat_time_last_part_front_core_input_scalar,
                                params.output_ub_init_times_last_part_front_core_input_scalar);
      // front part last core
      ComputeInitOutputUbParams(params.ids_ele_num_ub_front_part_last_core_input_scalar, output_ub_ele_num_one_row,
                                params.output_ub_init_last_repeat_time_front_part_last_core_input_scalar,
                                params.output_ub_init_times_front_part_last_core_input_scalar);
      // last part last core
      ComputeInitOutputUbParams(params.ids_ele_num_ub_last_part_last_core_input_scalar, output_ub_ele_num_one_row,
                                params.output_ub_init_last_repeat_time_last_part_last_core_input_scalar,
                                params.output_ub_init_times_last_part_last_core_input_scalar);
    } else if (params.select_key_input_scalar == SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN_SMALL_E) {
      // not align small e
      // e num params
      params.e_mov_times_gm2ub_input_scalar = 1;
      params.e_ub2gm_front_burst_len_input_scalar = e_size * input_ele_byte / BYTE_BLOCK;
      params.e_num_front_part_input_scalar = e_size / FP32_ELE_NUM_ALIGN_32B * FP32_ELE_NUM_ALIGN_32B;
      params.e_ub2gm_last_burst_len_input_scalar = 1;
      params.e_num_last_part_input_scalar = e_size - params.e_num_front_part_input_scalar;

      // input data params
      // front part front core
      ComputeInputParamsMovGm2ub(ub_tensor_size, input_ele_byte, params.input_ele_num_front_core_input_scalar, e_size,
                                 params.e_mov_times_gm2ub_input_scalar,
                                 params.ids_ele_num_ub_front_part_front_core_input_scalar,
                                 params.input_mov_times_gm2ub_front_part_front_core_input_scalar,
                                 params.input_front_burst_len_front_part_front_core_input_scalar,
                                 params.input_last_burst_len_front_part_front_core_input_scalar,
                                 params.input_front_ele_num_ub_front_part_front_core_input_scalar,
                                 params.input_last_ele_num_ub_front_part_front_core_input_scalar,
                                 params.input_front_rows_front_part_front_core_input_scalar,
                                 params.input_last_rows_front_part_front_core_input_scalar);
      // last part front core
      ComputeInputParamsMovGm2ub(ub_tensor_size, input_ele_byte, params.input_ele_num_front_core_input_scalar, e_size,
                                 params.e_mov_times_gm2ub_input_scalar,
                                 params.ids_ele_num_ub_last_part_front_core_input_scalar,
                                 params.input_mov_times_gm2ub_last_part_front_core_input_scalar,
                                 params.input_front_burst_len_last_part_front_core_input_scalar,
                                 params.input_last_burst_len_last_part_front_core_input_scalar,
                                 params.input_front_ele_num_ub_last_part_front_core_input_scalar,
                                 params.input_last_ele_num_ub_last_part_front_core_input_scalar,
                                 params.input_front_rows_last_part_front_core_input_scalar,
                                 params.input_last_rows_last_part_front_core_input_scalar);
      // front part last core
      ComputeInputParamsMovGm2ub(ub_tensor_size, input_ele_byte, params.input_ele_num_last_core_input_scalar, e_size,
                                 params.e_mov_times_gm2ub_input_scalar,
                                 params.ids_ele_num_ub_front_part_last_core_input_scalar,
                                 params.input_mov_times_gm2ub_front_part_last_core_input_scalar,
                                 params.input_front_burst_len_front_part_last_core_input_scalar,
                                 params.input_last_burst_len_front_part_last_core_input_scalar,
                                 params.input_front_ele_num_ub_front_part_last_core_input_scalar,
                                 params.input_last_ele_num_ub_front_part_last_core_input_scalar,
                                 params.input_front_rows_front_part_last_core_input_scalar,
                                 params.input_last_rows_front_part_last_core_input_scalar);
      // last part last core
      ComputeInputParamsMovGm2ub(ub_tensor_size, input_ele_byte, params.input_ele_num_last_core_input_scalar, e_size,
                                 params.e_mov_times_gm2ub_input_scalar,
                                 params.ids_ele_num_ub_last_part_last_core_input_scalar,
                                 params.input_mov_times_gm2ub_last_part_last_core_input_scalar,
                                 params.input_front_burst_len_last_part_last_core_input_scalar,
                                 params.input_last_burst_len_last_part_last_core_input_scalar,
                                 params.input_front_ele_num_ub_last_part_last_core_input_scalar,
                                 params.input_last_ele_num_ub_last_part_last_core_input_scalar,
                                 params.input_front_rows_last_part_last_core_input_scalar,
                                 params.input_last_rows_last_part_last_core_input_scalar);

      // output init params
      // front row front part front core
      ComputeInitOutputUbParams(params.input_front_rows_front_part_front_core_input_scalar, output_ub_ele_num_one_row,
                                params.output_ub_init_last_repeat_time_front_part_front_core_input_scalar,
                                params.output_ub_init_times_front_part_front_core_input_scalar);
      // last row front part front core
      ComputeInitOutputUbParams(params.input_last_rows_front_part_front_core_input_scalar, output_ub_ele_num_one_row,
                                params.output_ub_init_last_row_last_repeat_time_front_part_front_core_input_scalar,
                                params.output_ub_init_last_row_times_front_part_front_core_input_scalar);
      // front row last part front core
      ComputeInitOutputUbParams(params.input_front_rows_last_part_front_core_input_scalar, output_ub_ele_num_one_row,
                                params.output_ub_init_last_repeat_time_last_part_front_core_input_scalar,
                                params.output_ub_init_times_last_part_front_core_input_scalar);
      // last row last part front core
      ComputeInitOutputUbParams(params.input_last_rows_last_part_front_core_input_scalar, output_ub_ele_num_one_row,
                                params.output_ub_init_last_row_last_repeat_time_last_part_front_core_input_scalar,
                                params.output_ub_init_last_row_times_last_part_front_core_input_scalar);
      // front row front part last core
      ComputeInitOutputUbParams(params.input_front_rows_front_part_last_core_input_scalar, output_ub_ele_num_one_row,
                                params.output_ub_init_last_repeat_time_front_part_last_core_input_scalar,
                                params.output_ub_init_times_front_part_last_core_input_scalar);
      // last row front part last core
      ComputeInitOutputUbParams(params.input_last_rows_front_part_last_core_input_scalar, output_ub_ele_num_one_row,
                                params.output_ub_init_last_row_last_repeat_time_front_part_last_core_input_scalar,
                                params.output_ub_init_last_row_times_front_part_last_core_input_scalar);
      // front row last part last core
      ComputeInitOutputUbParams(params.input_front_rows_last_part_last_core_input_scalar, output_ub_ele_num_one_row,
                                params.output_ub_init_last_repeat_time_last_part_last_core_input_scalar,
                                params.output_ub_init_times_last_part_last_core_input_scalar);
      // last row last part last core
      ComputeInitOutputUbParams(params.input_last_rows_last_part_last_core_input_scalar, output_ub_ele_num_one_row,
                                params.output_ub_init_last_row_last_repeat_time_last_part_last_core_input_scalar,
                                params.output_ub_init_last_row_times_last_part_last_core_input_scalar);
      params.input_last_axis_align_front_part_ele_num_input_scalar =
        e_size / FP32_ELE_NUM_ALIGN_32B * FP32_ELE_NUM_ALIGN_32B;
      params.input_last_axis_align_floor_ele_num_input_scalar = UssCeil(e_size, FP32_ELE_NUM_ALIGN_32B);
      params.last_part_vadd_mask_input_scalar = e_size - params.input_last_axis_align_front_part_ele_num_input_scalar;
    } else if (params.select_key_input_scalar == SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ALIGN_BIG_E) {
      ComputeEleNumOneCore(ids_min_ele_num, ids_size, params.need_core_num_input_scalar, e_size,
                           params.ids_ele_num_front_core_input_scalar, params.ids_ele_num_last_core_input_scalar,
                           params.input_ele_num_front_core_input_scalar, params.input_ele_num_last_core_input_scalar,
                           num_segments);
      // ids params front core
      ComputeIdsParamsMovGm2ub(
        params.ids_ele_num_front_core_input_scalar, ub_tensor_size, ids_ele_byte,
        params.ids_mov_times_gm2ub_front_core_input_scalar, params.ids_front_burst_len_front_core_input_scalar,
        params.ids_last_burst_len_front_core_input_scalar, params.ids_ele_num_ub_front_part_front_core_input_scalar,
        params.ids_ele_num_ub_last_part_front_core_input_scalar);
      // ids params last core
      ComputeIdsParamsMovGm2ub(
        params.ids_ele_num_last_core_input_scalar, ub_tensor_size, ids_ele_byte,
        params.ids_mov_times_gm2ub_last_core_input_scalar, params.ids_front_burst_len_last_core_input_scalar,
        params.ids_last_burst_len_last_core_input_scalar, params.ids_ele_num_ub_front_part_last_core_input_scalar,
        params.ids_ele_num_ub_last_part_last_core_input_scalar);
      // align big e
      // e num params
      params.e_mov_times_gm2ub_input_scalar = UssCeilDiv(e_size, ub_tensor_ele_num);
      params.e_ub2gm_front_burst_len_input_scalar = ub_tensor_size / BYTE_BLOCK;
      params.e_num_front_part_input_scalar = ub_tensor_ele_num;
      params.e_num_last_part_input_scalar =
        ComputeDivRemainders(e_size, params.e_num_front_part_input_scalar,
                             params.e_mov_times_gm2ub_input_scalar - 1);
      params.e_ub2gm_last_burst_len_input_scalar = params.e_num_last_part_input_scalar * input_ele_byte / BYTE_BLOCK;

      // input data params
      // front part front core
      params.input_mov_times_gm2ub_front_part_front_core_input_scalar =
        params.ids_ele_num_ub_front_part_front_core_input_scalar;
      params.input_front_ele_num_ub_front_part_front_core_input_scalar = ub_tensor_ele_num;
      params.input_last_ele_num_ub_front_part_front_core_input_scalar = params.e_num_last_part_input_scalar;
      // last part front core
      params.input_mov_times_gm2ub_last_part_front_core_input_scalar =
        params.ids_ele_num_ub_last_part_front_core_input_scalar;
      params.input_front_ele_num_ub_last_part_front_core_input_scalar = ub_tensor_ele_num;
      params.input_last_ele_num_ub_last_part_front_core_input_scalar = params.e_num_last_part_input_scalar;
      // front part last core
      params.input_mov_times_gm2ub_front_part_last_core_input_scalar =
        params.ids_ele_num_ub_front_part_last_core_input_scalar;
      params.input_front_ele_num_ub_front_part_last_core_input_scalar = ub_tensor_ele_num;
      params.input_last_ele_num_ub_front_part_last_core_input_scalar = params.e_num_last_part_input_scalar;
      // last part last core
      params.input_mov_times_gm2ub_last_part_last_core_input_scalar =
        params.ids_ele_num_ub_last_part_last_core_input_scalar;
      params.input_front_ele_num_ub_last_part_last_core_input_scalar = ub_tensor_ele_num;
      params.input_last_ele_num_ub_last_part_last_core_input_scalar = params.e_num_last_part_input_scalar;
    } else if (params.select_key_input_scalar == SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN_BIG_E) {
      // not align big e
      // e num params
      params.e_mov_times_gm2ub_input_scalar = UssCeilDiv(e_size, ub_tensor_ele_num);
      params.e_ub2gm_front_burst_len_input_scalar = ub_tensor_size / BYTE_BLOCK;
      params.e_num_front_part_input_scalar = ub_tensor_ele_num;
      params.e_num_last_part_input_scalar =
        ComputeDivRemainders(e_size, params.e_num_front_part_input_scalar, params.e_mov_times_gm2ub_input_scalar - 1);
      params.e_ub2gm_last_burst_len_input_scalar = params.e_num_last_part_input_scalar * input_ele_byte / BYTE_BLOCK;
      params.e_gm2ub_last_burst_len_input_scalar = UssCeilDiv(params.e_num_last_part_input_scalar * input_ele_byte,
          BYTE_BLOCK);

      // input data params
      // front part front core
      params.input_mov_times_gm2ub_front_part_front_core_input_scalar =
        params.ids_ele_num_ub_front_part_front_core_input_scalar;
      params.input_front_ele_num_ub_front_part_front_core_input_scalar = ub_tensor_ele_num;
      params.input_last_ele_num_ub_front_part_front_core_input_scalar = params.e_num_last_part_input_scalar;
      // last part front core
      params.input_mov_times_gm2ub_last_part_front_core_input_scalar =
        params.ids_ele_num_ub_last_part_front_core_input_scalar;
      params.input_front_ele_num_ub_last_part_front_core_input_scalar = ub_tensor_ele_num;
      params.input_last_ele_num_ub_last_part_front_core_input_scalar = params.e_num_last_part_input_scalar;
      // front part last core
      params.input_mov_times_gm2ub_front_part_last_core_input_scalar =
        params.ids_ele_num_ub_front_part_last_core_input_scalar;
      params.input_front_ele_num_ub_front_part_last_core_input_scalar = ub_tensor_ele_num;
      params.input_last_ele_num_ub_front_part_last_core_input_scalar = params.e_num_last_part_input_scalar;
      // last part last core
      params.input_mov_times_gm2ub_last_part_last_core_input_scalar =
        params.ids_ele_num_ub_last_part_last_core_input_scalar;
      params.input_front_ele_num_ub_last_part_last_core_input_scalar = ub_tensor_ele_num;
      params.input_last_ele_num_ub_last_part_last_core_input_scalar = params.e_num_last_part_input_scalar;

      // output init params
      params.input_last_axis_align_front_part_ele_num_input_scalar =
        params.e_num_last_part_input_scalar / FP32_ELE_NUM_ALIGN_32B * FP32_ELE_NUM_ALIGN_32B;
      params.input_last_axis_align_floor_ele_num_input_scalar =
        UssCeil(params.e_num_last_part_input_scalar, FP32_ELE_NUM_ALIGN_32B);
      params.last_part_vadd_mask_input_scalar =
        params.e_num_last_part_input_scalar - params.input_last_axis_align_front_part_ele_num_input_scalar;
    } else if (params.select_key_input_scalar == SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ONE_MODIFY) {
      // last axis is one modify
      // e num params
      params.e_num_input_scalar = 1;
      params.e_mov_times_gm2ub_input_scalar = 1;
      params.e_ub2gm_front_burst_len_input_scalar = 1;

      // input data params
      // front part front core
      params.input_front_burst_len_front_part_front_core_input_scalar =
        params.ids_front_burst_len_front_core_input_scalar;
      params.input_front_ele_num_ub_front_part_front_core_input_scalar =
        params.ids_ele_num_ub_front_part_front_core_input_scalar;
      params.input_front_rows_front_part_front_core_input_scalar =
        params.input_front_ele_num_ub_front_part_front_core_input_scalar;
      // last part front core
      params.input_front_burst_len_last_part_front_core_input_scalar =
        params.ids_last_burst_len_front_core_input_scalar;
      params.input_front_ele_num_ub_last_part_front_core_input_scalar =
        params.ids_ele_num_ub_last_part_front_core_input_scalar;
      params.input_front_rows_last_part_front_core_input_scalar =
        params.input_front_ele_num_ub_last_part_front_core_input_scalar;
      // front part last core
      params.input_front_burst_len_front_part_last_core_input_scalar =
        params.ids_front_burst_len_last_core_input_scalar;
      params.input_front_ele_num_ub_front_part_last_core_input_scalar =
        params.ids_ele_num_ub_front_part_last_core_input_scalar;
      params.input_front_rows_front_part_last_core_input_scalar =
        params.input_front_ele_num_ub_front_part_last_core_input_scalar;
      // last part last core
      params.input_front_burst_len_last_part_last_core_input_scalar = params.ids_last_burst_len_last_core_input_scalar;
      params.input_front_ele_num_ub_last_part_last_core_input_scalar =
        params.ids_ele_num_ub_last_part_last_core_input_scalar;
      params.input_front_rows_last_part_last_core_input_scalar =
        params.input_front_ele_num_ub_last_part_last_core_input_scalar;

      // output init params
      params.output_ub_init_times_front_part_front_core_input_scalar =
        UssCeilDiv(params.ids_ele_num_ub_front_part_front_core_input_scalar, MASK_FP32);
      params.output_ub_init_times_last_part_front_core_input_scalar =
        UssCeilDiv(params.ids_ele_num_ub_last_part_front_core_input_scalar, MASK_FP32);
      params.output_ub_init_times_front_part_last_core_input_scalar =
        UssCeilDiv(params.ids_ele_num_ub_front_part_last_core_input_scalar, MASK_FP32);
      params.output_ub_init_times_last_part_last_core_input_scalar =
        UssCeilDiv(params.ids_ele_num_ub_last_part_last_core_input_scalar, MASK_FP32);
      params.last_part_vadd_mask_input_scalar =
        params.ids_ele_num_ub_last_part_last_core_input_scalar -
        (params.output_ub_init_times_last_part_last_core_input_scalar - 1) * MASK_FP32;
    } else if (params.select_key_input_scalar == SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ONE_MULTI) {
      // last axis is one multi 64

      // e num params
      params.e_num_input_scalar = 1;
      params.e_mov_times_gm2ub_input_scalar = 1;
      params.e_ub2gm_front_burst_len_input_scalar = 1;

      // input data params
      // front part front core
      params.input_front_burst_len_front_part_front_core_input_scalar =
        params.ids_front_burst_len_front_core_input_scalar;
      params.input_front_ele_num_ub_front_part_front_core_input_scalar =
        params.ids_ele_num_ub_front_part_front_core_input_scalar;
      params.input_front_rows_front_part_front_core_input_scalar =
        params.input_front_ele_num_ub_front_part_front_core_input_scalar;
      // last part front core
      params.input_front_burst_len_last_part_front_core_input_scalar =
        params.ids_last_burst_len_front_core_input_scalar;
      params.input_front_ele_num_ub_last_part_front_core_input_scalar =
        params.ids_ele_num_ub_last_part_front_core_input_scalar;
      params.input_front_rows_last_part_front_core_input_scalar =
        params.input_front_ele_num_ub_last_part_front_core_input_scalar;
      // front part last core
      params.input_front_burst_len_front_part_last_core_input_scalar =
        params.ids_front_burst_len_last_core_input_scalar;
      params.input_front_ele_num_ub_front_part_last_core_input_scalar =
        params.ids_ele_num_ub_front_part_last_core_input_scalar;
      params.input_front_rows_front_part_last_core_input_scalar =
        params.input_front_ele_num_ub_front_part_last_core_input_scalar;
      // last part last core
      params.input_front_burst_len_last_part_last_core_input_scalar = params.ids_last_burst_len_last_core_input_scalar;
      params.input_front_ele_num_ub_last_part_last_core_input_scalar =
        params.ids_ele_num_ub_last_part_last_core_input_scalar;
      params.input_front_rows_last_part_last_core_input_scalar =
        params.input_front_ele_num_ub_last_part_last_core_input_scalar;

      // output init params
      // front part front core
      params.output_ub_init_times_front_part_front_core_input_scalar =
        params.ids_ele_num_ub_front_part_front_core_input_scalar / (MASK_FP32 * MULTI);
      params.output_ub_init_last_repeat_time_front_part_front_core_input_scalar =
        ComputeDivRemainders(params.ids_ele_num_ub_front_part_front_core_input_scalar, MASK_FP32 * MULTI,
                             params.output_ub_init_times_front_part_front_core_input_scalar) / MASK_FP32;
      // last part front core
      params.output_ub_init_times_last_part_front_core_input_scalar =
        params.ids_ele_num_ub_last_part_front_core_input_scalar / (MASK_FP32 * MULTI);
      params.output_ub_init_last_repeat_time_last_part_front_core_input_scalar =
        ComputeDivRemainders(params.ids_ele_num_ub_last_part_front_core_input_scalar, MASK_FP32 * MULTI,
                             params.output_ub_init_times_last_part_front_core_input_scalar) / MASK_FP32;
      // front part last core
      params.output_ub_init_times_front_part_last_core_input_scalar =
        params.ids_ele_num_ub_front_part_last_core_input_scalar / (MASK_FP32 * MULTI);
      params.output_ub_init_last_repeat_time_front_part_last_core_input_scalar =
        ComputeDivRemainders(params.ids_ele_num_ub_front_part_last_core_input_scalar, MASK_FP32 * MULTI,
                             params.output_ub_init_times_front_part_last_core_input_scalar) / MASK_FP32;
      // last part last core
      // multi 64 part
      params.output_ub_init_times_last_part_last_core_input_scalar =
        params.ids_ele_num_ub_last_part_last_core_input_scalar / (MASK_FP32 * MULTI);
      // single 64 part
      params.output_ub_init_last_repeat_time_last_part_last_core_input_scalar =
        ComputeDivRemainders(params.ids_ele_num_ub_last_part_last_core_input_scalar, MASK_FP32 * MULTI,
                             params.output_ub_init_times_last_part_last_core_input_scalar) / MASK_FP32;
      // last mask part
      params.last_part_vadd_mask_input_scalar =
        ComputeDivRemainders(params.ids_ele_num_ub_last_part_last_core_input_scalar, MASK_FP32 * MULTI,
                             params.output_ub_init_times_last_part_last_core_input_scalar) -
        params.output_ub_init_last_repeat_time_last_part_last_core_input_scalar * MASK_FP32;
    } else if (params.select_key_input_scalar == SELECT_KEY_MODE_FP32_INPUT_NUM_SEGMENT_ONE) {
      params.e_mov_times_gm2ub_input_scalar = UssCeilDiv(e_size, ub_tensor_ele_num);
      params.e_ub2gm_front_burst_len_input_scalar = ub_tensor_size / BYTE_BLOCK;
      params.e_num_front_part_input_scalar = ub_tensor_ele_num;
      params.e_num_last_part_input_scalar =
        ComputeDivRemainders(e_size, params.e_num_front_part_input_scalar, params.e_mov_times_gm2ub_input_scalar - 1);
      params.e_ub2gm_last_burst_len_input_scalar = UssCeilDiv(params.e_num_last_part_input_scalar * input_ele_byte,
          BYTE_BLOCK);
    }
    // write tiling params to run_info
    WriteTilingParams(params, run_info);
    // cout tiling params
    PrintTilingParams(op_type, params);
    // BlockDim, core num used in tik op
    run_info.block_dim = params.need_core_num_input_scalar;
    // workspace, null for tik op
    std::vector<int64_t> workspace;
    run_info.workspaces = workspace;
  } else if (input_dtype == DTYPE_INT32 || input_dtype == DTYPE_FP16) {
    // int32 tiling params
    TilingParamsInt32 params;
    InitTilingParams(params);
    // commmn params
    // select key
    int32_t e_once_num = 0;
    int32_t id_once_num = 0;
    int32_t mask = 0;
    if(input_dtype == DTYPE_INT32) {
      mask = MASK_INT32;
    } else {
      mask = MASK_FP16;
    }

    IsUsingAllCoreByNumSegments(num_segments, core_num, params.need_core_num_input_scalar,
                                e_size, output_ub_ele_num_one_row);
    int32_t ub_tensor_size = 0;
    int32_t ub_tensor_size_input = 0;
    int32_t ub_tensor_size_output = 0;
    ComputeUbTensorSizeNoAtomic(ub_size, input_shape, input_dtype, e_size, ub_tensor_size, ub_tensor_size_input,
                                ub_tensor_size_output, output_ub_ele_num_one_row,
                                params.need_core_num_input_scalar, mask, num_segments);
    GELOGD("op [%s] : ub_tensor_size_id is=%d,ub_tensor_size_input is %d,ub_tensor_size_output is %d",
           op_type.c_str(), ub_tensor_size, ub_tensor_size_input, ub_tensor_size_output);

    bool flag = GetTilingModeNoAtomic(input_shape, e_size, ids_size, input_dtype, ids_dtype, ub_tensor_size,
    ub_tensor_size_input, params.select_key_input_scalar, e_once_num, id_once_num, params.need_core_num_input_scalar,
    output_ub_ele_num_one_row, params.num_segment_max, mask, input_size, num_segments);
    GELOGD("op[%s]:e_once_num is %d ,id_once_num is %d ,params.num_segment_max is %d", op_type.c_str(), e_once_num,
           id_once_num, params.num_segment_max);
    if (!flag) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetTilingMode failed.");
      return false;
    }

    ComputeNumSegmentsParams(
      params.need_core_num_input_scalar, num_segments,
      params.num_segments_front_core_input_scalar, params.num_segments_last_core_input_scalar,
      e_size, output_ub_ele_num_one_row);
    // ids params
    params.ids_size_input_scalar = ids_size;
    ComputeIdsParamsMovGm2ubNoAtomic(
      ids_size, id_once_num, ids_ele_byte, params.ids_mov_times_gm2ub_input_scalar,
      params.ids_front_burst_len_input_scalar, params.ids_last_burst_len_input_scalar,
      params.ids_ele_num_ub_front_part_input_scalar,
      params.ids_ele_num_ub_last_part_input_scalar);
    // e num params
    params.e_num_input_scalar = e_size;
    ComputeENumParams(
      input_dtype,params.e_num_input_scalar, input_ele_byte, e_once_num,
      params.e_mov_times_gm2ub_input_scalar, params.e_ub2gm_front_burst_len_input_scalar,
      params.e_num_front_part_input_scalar, params.repeat_time_front_part_input_scalar,
      params.e_ub2gm_last_burst_len_input_scalar, params.e_num_last_part_input_scalar,
      params.repeat_time_last_part_input_scalar,
      params.align_scalar, params.align_scalar_lastcore, params.e_gm2ub_front_burst_len_input_scalar,
      params.e_gm2ub_last_burst_len_input_scalar, params.num_segments_front_core_input_scalar,
      params.num_segments_last_core_input_scalar, params.need_core_num_input_scalar,
      params.num_segment_max, params.num_segment_max_time, params.num_segment_max_time_lastcore,
      params.front_num_segment,params.front_num_segment_last, params.front_num_segment_lastcore,
      params.front_num_segment_last_lastcore, params.e_ub2gm_front_burst_len_input_scalar_lastcore,
      params.e_ub2gm_last_burst_len_input_scalar_lastcore, input_size, num_segments,
      params.repeat_times, params.repeat_times_last_part, params.repeat_times_last_part_lastcore,
      params.e_mov_times_gm2ub_input_scalar_lastcore, params.repeat_time_front_part_input_scalar_lastcore);
    // write tiling params to run_info
    WriteTilingParams(params, run_info);
    // cout tiling params
    PrintTilingParams(op_type, params);
    // BlockDim, core num used in tik op
    run_info.block_dim = params.need_core_num_input_scalar;
    // workspace, null for tik op
    std::vector<int64_t> workspace;
    run_info.workspaces = workspace;
  }
  GELOGI("op[%s] op tiling success.", op_type.c_str());
  return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(UnsortedSegmentSum, UnsortedSegmentSumTiling);
}  // namespace optiling
