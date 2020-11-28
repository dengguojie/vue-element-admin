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

namespace optiling {

const std::string UNSORTED_SEGMENT_SUM_OP_TYPE = "UnsortedSegmentSum";
const int32_t BYTE_BLOCK = 32;
const int32_t MIN_ELE_SIZE_USING_ALL_CORE = 1024;
const int32_t MASK_FP32 = 64;
const int32_t MASK_INT32 = 64;
const int32_t MAX_REPEAT_TIME = 255;
const int32_t FP32_ELE_NUM_ALIGN_32B = 8;
const int32_t BYTE_FULL_MASK = 256;
const int32_t MULTI = 4;

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

// int32 select key
const int32_t SELECT_KEY_MODE_INT32_SMALL_E = 11;
const int32_t SELECT_KEY_MODE_INT32_BIG_E = 12;

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
};

/******************COMMON_FUNCTION******************/

int32_t ComputeDivRemainders(const int32_t& num, const int32_t& factor, const int32_t& times) {
  int32_t res;
  res = num - factor * times;
  return res;
}

int32_t UssCeil(const int32_t& num, const int32_t& factor) {
  int32_t res;
  res = (num % factor == 0) ? num : factor * (num / factor + 1);
  return res;
}

int32_t UssCeilDiv(const int32_t& num, const int32_t& factor) {
  int32_t res;
  res = (num % factor == 0) ? num / factor : num / factor + 1;
  return res;
}

bool GetUssCompileParams(const std::string& op_type, const nlohmann::json& op_compile_info_json, int32_t& core_num,
                         int32_t& ub_size, int32_t& ub_tensor_num) {
  using namespace nlohmann;
  if (op_compile_info_json == nullptr) {
    ge::OpsGetCompileParamsErrReport("UnsortedSegmentSum", "op_compile_info_json");
    OP_LOGE(op_type.c_str(), "op_compile_info_json is null");
    return false;
  }
  const auto& allVars = op_compile_info_json["vars"];
  // core num
  if (allVars.count("core_num") == 0) {
    ge::OpsGetCompileParamsErrReport("UnsortedSegmentSum", "core_num");
    OP_LOGE(op_type.c_str(), "core_num is null");
    return false;
  }
  core_num = allVars["core_num"].get<std::int32_t>();
  // ub size
  if (allVars.count("ub_size") == 0) {
    ge::OpsGetCompileParamsErrReport("UnsortedSegmentSum", "ub_size");
    OP_LOGE(op_type.c_str(), "ub_size is null");
    return false;
  }
  ub_size = allVars["ub_size"].get<std::int32_t>();
  // ub tensor num
  if (allVars.count("ub_tensor_num") == 0) {
    ge::OpsGetCompileParamsErrReport("UnsortedSegmentSum", "ub_tensor_num");
    OP_LOGE(op_type.c_str(), "ub_tensor_num is null");
    return false;
  }
  ub_tensor_num = allVars["ub_tensor_num"].get<std::int32_t>();
  GELOGD("op [%s] : GetCompileParams, core_num[%d], ub_size[%d].", UNSORTED_SEGMENT_SUM_OP_TYPE.c_str(), core_num,
         ub_size);
  return true;
}

bool GetTilingMode(const std::vector<int64_t>& input_shape, const int32_t& e_size, const std::string& input_dtype,
                   const int32_t& ub_tensor_ele_num, int32_t& select_key) {
  int input_dim = input_shape.size();
  if (input_shape.empty()) {
    return false;
  }
  if (input_dtype == DTYPE_FP32) {
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
  } else if (input_dtype == DTYPE_INT32) {
    if (e_size > ub_tensor_ele_num) {
      // e big
      select_key = SELECT_KEY_MODE_INT32_BIG_E;
    } else {
      // e small
      select_key = SELECT_KEY_MODE_INT32_SMALL_E;
    }
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

bool IsUsingAllCore(const int32_t& ids_size, const int32_t& core_num, int32_t& need_core_num) {
  if (ids_size >= MIN_ELE_SIZE_USING_ALL_CORE) {
    need_core_num = core_num;
    return true;
  }
  need_core_num = 1;
  return false;
}

bool IsUsingAllCoreByNumSegments(const int32_t& num_segments, const int32_t& core_num, int32_t& need_core_num) {
  if (num_segments >= MIN_ELE_SIZE_USING_ALL_CORE) {
    need_core_num = core_num;
    return true;
  }
  need_core_num = 1;
  return false;
}

void ComputeUbTensorSize(const int32_t& ub_size, const std::vector<int64_t>& input_shape,
                         const std::string& input_dtype, const int32_t& e_size, int32_t& ub_tensor_size) {
  if (input_dtype == DTYPE_FP32) {
    if (e_size == 1) {
      // input is one dim or last axis is one
      int32_t one_row_size = FP32_BYTE + INT32_BYTE + FP32_BYTE * FP32_ELE_NUM_ALIGN_32B;
      ub_tensor_size = UssCeil(ub_size / one_row_size, BYTE_BLOCK);
    } else if (e_size > 1) {
      int32_t ub_tensor_num = 0;
      if (e_size % FP32_ELE_NUM_ALIGN_32B == 0) {
        // align
        ub_tensor_num = UB_TENSOR_NUM_FP32_INPUT_LAST_AXIS_ALIGN;
        ub_tensor_size = UssCeil(ub_size / ub_tensor_num, BYTE_BLOCK);
      } else if (e_size % FP32_ELE_NUM_ALIGN_32B > 0) {
        // not align
        ub_tensor_num = UB_TENSOR_NUM_FP32_INPUT_LAST_AXIS_NOT_ALIGN;
        ub_tensor_size = UssCeil(ub_size / ub_tensor_num, BYTE_BLOCK);
      }
    }
  }
}

/******************MODE_FP32_INPUT_LAST_AXIS_ALIGN******************/
void ComputeEleNumOneCore(const int32_t& min_ele_num, const int32_t& ids_num, const int32_t& core_num,
                          const int32_t& e_size, int32_t& ids_ele_num_front_core, int32_t& ids_ele_num_last_core,
                          int32_t& input_ele_num_front_core, int32_t& input_ele_num_last_core) {
  int32_t ids_num_align = UssCeil(ids_num, min_ele_num);
  if (e_size == 1) {
    ids_ele_num_front_core = ids_num_align / core_num;
    ids_ele_num_front_core = ids_ele_num_front_core / MASK_FP32 * MASK_FP32;
    ids_ele_num_last_core = ComputeDivRemainders(ids_num, ids_ele_num_front_core, core_num - 1);
    input_ele_num_front_core = ids_ele_num_front_core;
    input_ele_num_last_core = ids_ele_num_last_core;
    return;
  }
  ids_ele_num_front_core = ids_num_align / core_num;
  if (ids_num % ids_ele_num_front_core == 0) {
    ids_ele_num_last_core = ids_ele_num_front_core;
  } else {
    ids_ele_num_last_core = ComputeDivRemainders(ids_num, ids_ele_num_front_core, core_num - 1);
  }
  input_ele_num_front_core = ids_ele_num_front_core * e_size;
  input_ele_num_last_core = ids_ele_num_last_core * e_size;
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
  int32_t max_ids_ele_num_one_ub_tensor = UssCeilDiv(ub_tensor_size, ids_ele_byte);
  if (ids_ele_num_one_core <= max_ids_ele_num_one_ub_tensor) {
    // mov_times = 1, ub tensor is enough for ele one core
    ids_mov_times_gm2ub = 1;
    ids_ele_num_ub_front_part = ids_ele_num_one_core;
    ids_ele_num_ub_last_part = ids_ele_num_ub_front_part;
  } else if (ids_ele_num_one_core > max_ids_ele_num_one_ub_tensor) {
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

void ComputeUb2gmParams(const EleByte& ele_byte, const int32_t e_num, const int32_t ub_tensor_size,
                        int32_t& e_mov_times_gm2ub, int32_t& e_ub2gm_front_burst_len, int32_t& e_num_front_part,
                        int32_t& e_ub2gm_last_burst_len, int32_t& e_num_last_part) {
  int32_t max_e_one_ub_tensor = ub_tensor_size / ele_byte;
  if (e_num <= max_e_one_ub_tensor) {
    e_mov_times_gm2ub = 1;
    e_ub2gm_front_burst_len = UssCeilDiv(e_num * ele_byte, BYTE_BLOCK);
    e_ub2gm_last_burst_len = e_ub2gm_front_burst_len;
    e_num_front_part = e_num;
    e_num_last_part = e_num;
  } else {
    e_mov_times_gm2ub = UssCeilDiv(e_num, max_e_one_ub_tensor);
    e_ub2gm_front_burst_len = UssCeilDiv(ub_tensor_size, BYTE_BLOCK);
    e_num_front_part = ub_tensor_size / ele_byte;
    e_num_last_part = ComputeDivRemainders(e_num, e_num_front_part, e_mov_times_gm2ub - 1);
    e_ub2gm_last_burst_len = UssCeilDiv(e_num_last_part * ele_byte, BYTE_BLOCK);
  }
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

void ComputeNumSegmentsParams(const int32_t& need_core_num, const int32_t& num_segmens, int32_t& num_segmens_front_core,
                              int32_t& num_segmens_last_core) {
  if (need_core_num == 1) {
    num_segmens_front_core = num_segmens;
    num_segmens_last_core = num_segmens_front_core;
  } else if (need_core_num > 1) {
    num_segmens_front_core = UssCeilDiv(num_segmens, need_core_num);
    num_segmens_last_core = ComputeDivRemainders(num_segmens, num_segmens_front_core, need_core_num - 1);
  }
}

void ComputeENumParams(const int32_t& e_num, const EleByte& ele_byte, const int32_t& ub_tensor_size,
                       int32_t& e_mov_times_gm2ub_input_scalar, int32_t& e_ub2gm_front_burst_len_input_scalar,
                       int32_t& e_num_front_part_input_scalar, int32_t& repeat_time_front_part_input_scalar,
                       int32_t& e_ub2gm_last_burst_len_input_scalar, int32_t& e_num_last_part_input_scalar,
                       int32_t& repeat_time_last_part_input_scalar) {
  int32_t max_ele_num_one_ub_tensor = UssCeilDiv(ub_tensor_size, ele_byte);
  max_ele_num_one_ub_tensor = UssCeil(max_ele_num_one_ub_tensor, BYTE_BLOCK);
  if (e_num >= max_ele_num_one_ub_tensor) {
    // big e
    e_mov_times_gm2ub_input_scalar = UssCeilDiv(e_num, max_ele_num_one_ub_tensor);
    // front part
    e_ub2gm_front_burst_len_input_scalar = UssCeilDiv(max_ele_num_one_ub_tensor * ele_byte, BYTE_BLOCK);
    e_num_front_part_input_scalar = max_ele_num_one_ub_tensor;
    repeat_time_front_part_input_scalar = UssCeilDiv(e_num_front_part_input_scalar, MASK_INT32);
    // last part
    e_num_last_part_input_scalar =
        ComputeDivRemainders(e_num, e_num_front_part_input_scalar, e_mov_times_gm2ub_input_scalar - 1);
    e_ub2gm_last_burst_len_input_scalar = UssCeilDiv(e_num_last_part_input_scalar * ele_byte, BYTE_BLOCK);
    repeat_time_last_part_input_scalar = UssCeilDiv(e_num_last_part_input_scalar, MASK_INT32);
  } else {
    // small e
    e_mov_times_gm2ub_input_scalar = 1;
    // front part
    e_ub2gm_front_burst_len_input_scalar = UssCeilDiv(e_num * ele_byte, BYTE_BLOCK);
    e_num_front_part_input_scalar = e_num;
    repeat_time_front_part_input_scalar = UssCeilDiv(e_num_front_part_input_scalar, MASK_INT32);
    // last part
    e_num_last_part_input_scalar = e_ub2gm_front_burst_len_input_scalar;
    e_ub2gm_last_burst_len_input_scalar = e_ub2gm_front_burst_len_input_scalar;
    repeat_time_last_part_input_scalar = repeat_time_front_part_input_scalar;
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
}

// tiling function
bool UnsortedSegmentSumTiling(const std::string& op_type, const TeOpParas& op_paras,
                              const nlohmann::json& op_compile_info_json, OpRunInfo& run_info) {
  GELOGI("op[%s] op tiling begin.", op_type.c_str());
  if (op_paras.inputs.empty()) {
    ge::OpsOneInputShapeErrReport("UnsortedSegmentSum", "op_paras.inputs", "op_paras.inputs is empty.");
    OP_LOGE(op_type.c_str(), "op_paras.inputs is empty.");
    return false;
  }
  if (op_paras.inputs.size() < 2) {
    ge::OpsOneInputShapeErrReport("UnsortedSegmentSum", "op_paras.inputs", "op_paras.inputs.size() < 2.");
    OP_LOGE(op_type.c_str(), "op_paras.inputs.size() < 2.");
    return false;
  }
  if (op_paras.inputs[0].tensor.empty()) {
    ge::OpsOneInputShapeErrReport("UnsortedSegmentSum", "input_data", "input_data tensor is empty.");
    OP_LOGE(op_type.c_str(), "input_data tensor is empty.");
    return false;
  }
  if (op_paras.inputs[1].tensor.empty()) {
    ge::OpsOneInputShapeErrReport("UnsortedSegmentSum", "ids", "ids tensor is empty.");
    OP_LOGE(op_type.c_str(), "ids tensor is empty.");
    return false;
  }
  const std::vector<int64_t>& input_shape = op_paras.inputs[0].tensor[0].shape;
  const std::vector<int64_t>& ids_shape = op_paras.inputs[1].tensor[0].shape;
  const int32_t& input_size = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int>());
  const int32_t& ids_size = std::accumulate(ids_shape.begin(), ids_shape.end(), 1, std::multiplies<int>());
  GELOGD("op [%s] : input_size=%d, ids_size=%d", op_type.c_str(), input_size, ids_size);
  if (input_shape.size() < ids_shape.size()) {
    ge::OpsTwoInputShapeErrReport("UnsortedSegmentSum", "input_data", "ids",
                                  "dim of input must be greater than or equal with dim of ids");
    OP_LOGE(op_type.c_str(), "dim of input must be greater than or equal with dim of ids");
    return false;
  }
  for (unsigned i = 0; i < ids_shape.size(); i++) {
    if (input_shape[i] != ids_shape[i]) {
      ge::OpsTwoInputShapeErrReport("UnsortedSegmentSum", "input_data", "ids",
                                    "front shape of input must be equal with ids shape");
      OP_LOGE(op_type.c_str(), "front shape of input must be equal with ids shape");
      return false;
    }
  }
  int32_t e_size = input_size / ids_size;
  const std::string& input_dtype = op_paras.inputs[0].tensor[0].dtype;
  const std::string& ids_dtype = op_paras.inputs[1].tensor[0].dtype;
  bool flag;
  // get input dtype
  EleByte input_ele_byte;
  flag = GetEleDtype(input_dtype, input_ele_byte);
  if (!flag) {
    ge::OpsOneInputShapeErrReport("UnsortedSegmentSum", "input_data", "get input_ele_byte failed.");
    OP_LOGE("op[%s] get input_ele_byte failed.", op_type.c_str());
    return false;
  }
  GELOGI("op[%s] get input ele dtype success.", op_type.c_str());
  EleByte output_ele_byte = input_ele_byte;
  int32_t output_ub_ele_num_one_row = BYTE_BLOCK / output_ele_byte;
  // get ids dtype
  EleByte ids_ele_byte;
  flag = GetEleDtype(ids_dtype, ids_ele_byte);
  if (!flag) {
    ge::OpsOneInputShapeErrReport("UnsortedSegmentSum", "ids", "get ids_ele_byte failed.");
    OP_LOGE("op[%s] get ids_ele_byte failed.", op_type.c_str());
    return false;
  }
  GELOGI("op[%s] get ids ele dtype success.", op_type.c_str());
  // get compile info
  int32_t core_num = 1;
  int32_t ub_size = 256 * 1024;
  int32_t ub_tensor_num = 0;
  flag = GetUssCompileParams(op_type, op_compile_info_json, core_num, ub_size, ub_tensor_num);
  if (!flag) {
    OP_LOGE("op[%s] GetCompileParams failed.", op_type.c_str());
    return false;
  }
  GELOGI("op[%s] GetCompileParams success.", op_type.c_str());
  // compute ub tensor size
  int32_t ub_tensor_size = 0;
  ComputeUbTensorSize(ub_size, input_shape, input_dtype, e_size, ub_tensor_size);
  if (e_size == 1) {
    ub_tensor_size = ub_tensor_size / BYTE_FULL_MASK * BYTE_FULL_MASK;
  }
  GELOGD("op [%s] : ub_tensor_size=%d", op_type.c_str(), ub_tensor_size);
  int32_t ub_tensor_ele_num = ub_tensor_size / input_ele_byte;
  if (input_dtype == DTYPE_FP32) {
    // fp32 tiling params
    TilingParamsFp32 params;
    InitTilingParams(params);
    // select key
    flag = GetTilingMode(input_shape, e_size, input_dtype, ub_tensor_ele_num, params.select_key_input_scalar);
    if (!flag) {
      ge::OpsOneInputShapeErrReport("UnsortedSegmentSum", "tiling_mode", "GetTilingMode failed.");
      OP_LOGE("op[%s] GetTilingMode failed.", op_type.c_str());
      return false;
    }
    params.e_num_input_scalar = e_size;
    // ids params compute is common
    params.ids_size_input_scalar = ids_size;
    int32_t ids_min_ele_num = BYTE_BLOCK / ids_ele_byte;
    // is using all core
    flag = IsUsingAllCore(ids_size, core_num, params.need_core_num_input_scalar);
    ComputeEleNumOneCore(ids_min_ele_num, ids_size, params.need_core_num_input_scalar, e_size,
                         params.ids_ele_num_front_core_input_scalar, params.ids_ele_num_last_core_input_scalar,
                         params.input_ele_num_front_core_input_scalar, params.input_ele_num_last_core_input_scalar);
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
      params.input_last_axis_align_front_part_ele_num_input_scalar =
          e_size / FP32_ELE_NUM_ALIGN_32B * FP32_ELE_NUM_ALIGN_32B;
      params.input_last_axis_align_floor_ele_num_input_scalar = UssCeil(e_size, FP32_ELE_NUM_ALIGN_32B);
      params.last_part_vadd_mask_input_scalar = e_size - params.input_last_axis_align_front_part_ele_num_input_scalar;
    } else if (params.select_key_input_scalar == SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ALIGN_BIG_E) {
      // is using all core
      if (ids_size < core_num) {
        params.need_core_num_input_scalar = ids_size;
      } else {
        params.need_core_num_input_scalar = core_num;
      }
      ComputeEleNumOneCore(ids_min_ele_num, ids_size, params.need_core_num_input_scalar, e_size,
                           params.ids_ele_num_front_core_input_scalar, params.ids_ele_num_last_core_input_scalar,
                           params.input_ele_num_front_core_input_scalar, params.input_ele_num_last_core_input_scalar);
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
          ComputeDivRemainders(e_size, params.e_num_front_part_input_scalar, params.e_mov_times_gm2ub_input_scalar - 1);
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
      params.e_ub2gm_last_burst_len_input_scalar = params.e_num_last_part_input_scalar / BYTE_BLOCK;

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
                               params.output_ub_init_times_front_part_front_core_input_scalar) /
          MASK_FP32;
      // last part front core
      params.output_ub_init_times_last_part_front_core_input_scalar =
          params.ids_ele_num_ub_last_part_front_core_input_scalar / (MASK_FP32 * MULTI);
      params.output_ub_init_last_repeat_time_last_part_front_core_input_scalar =
          ComputeDivRemainders(params.ids_ele_num_ub_last_part_front_core_input_scalar, MASK_FP32 * MULTI,
                               params.output_ub_init_times_last_part_front_core_input_scalar) /
          MASK_FP32;
      // front part last core
      params.output_ub_init_times_front_part_last_core_input_scalar =
          params.ids_ele_num_ub_front_part_last_core_input_scalar / (MASK_FP32 * MULTI);
      params.output_ub_init_last_repeat_time_front_part_last_core_input_scalar =
          ComputeDivRemainders(params.ids_ele_num_ub_front_part_last_core_input_scalar, MASK_FP32 * MULTI,
                               params.output_ub_init_times_front_part_last_core_input_scalar) /
          MASK_FP32;
      // last part last core
      // multi 64 part
      params.output_ub_init_times_last_part_last_core_input_scalar =
          params.ids_ele_num_ub_last_part_last_core_input_scalar / (MASK_FP32 * MULTI);
      // single 64 part
      params.output_ub_init_last_repeat_time_last_part_last_core_input_scalar =
          ComputeDivRemainders(params.ids_ele_num_ub_last_part_last_core_input_scalar, MASK_FP32 * MULTI,
                               params.output_ub_init_times_last_part_last_core_input_scalar) /
          MASK_FP32;
      // last mask part
      params.last_part_vadd_mask_input_scalar =
          ComputeDivRemainders(params.ids_ele_num_ub_last_part_last_core_input_scalar, MASK_FP32 * MULTI,
                               params.output_ub_init_times_last_part_last_core_input_scalar) -
          params.output_ub_init_last_repeat_time_last_part_last_core_input_scalar * MASK_FP32;
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
  } else if (input_dtype == DTYPE_INT32) {
    std::string key_num_segments = "num_segments";
    if (op_paras.const_inputs.find(key_num_segments) == op_paras.const_inputs.end()) {
      ge::OpsOneInputShapeErrReport("UnsortedSegmentSum", "num_segments", "num_segments not exists.");
      OP_LOGE("op[%s] num_segments not exists.", op_type.c_str());
      return false;
    }
    const int32_t* num_segments_ptr =
        reinterpret_cast<const int32_t*>(std::get<0>(op_paras.const_inputs.at(key_num_segments)));
    int32_t num_segments = *num_segments_ptr;
    GELOGD("op [%s] : num_segments=%d", op_type.c_str(), num_segments);
    // int32 tiling params
    TilingParamsInt32 params;
    InitTilingParams(params);
    // commmn params
    // select key
    flag = GetTilingMode(input_shape, e_size, input_dtype, ub_tensor_ele_num, params.select_key_input_scalar);
    if (!flag) {
      ge::OpsOneInputShapeErrReport("UnsortedSegmentSum", "tiling_mode", "GetTilingMode failed.");
      OP_LOGE("op[%s] GetTilingMode failed.", op_type.c_str());
      return false;
    }
    IsUsingAllCoreByNumSegments(num_segments, core_num, params.need_core_num_input_scalar);
    ComputeNumSegmentsParams(params.need_core_num_input_scalar, num_segments,
                             params.num_segments_front_core_input_scalar, params.num_segments_last_core_input_scalar);
    // ids params
    params.ids_size_input_scalar = ids_size;
    ComputeIdsParamsMovGm2ub(ids_size, ub_tensor_size, ids_ele_byte, params.ids_mov_times_gm2ub_input_scalar,
                             params.ids_front_burst_len_input_scalar, params.ids_last_burst_len_input_scalar,
                             params.ids_ele_num_ub_front_part_input_scalar,
                             params.ids_ele_num_ub_last_part_input_scalar);
    // e num params
    params.e_num_input_scalar = e_size;
    ComputeENumParams(params.e_num_input_scalar, input_ele_byte, ub_tensor_size, params.e_mov_times_gm2ub_input_scalar,
                      params.e_ub2gm_front_burst_len_input_scalar, params.e_num_front_part_input_scalar,
                      params.repeat_time_front_part_input_scalar, params.e_ub2gm_last_burst_len_input_scalar,
                      params.e_num_last_part_input_scalar, params.repeat_time_last_part_input_scalar);
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
