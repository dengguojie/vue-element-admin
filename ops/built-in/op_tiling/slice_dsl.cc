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

/*!
 * \file slice_dsl.cc
 * \brief
 */
#include "slice_dsl.h"
#include <algorithm>
#include <unordered_map>
#include <numeric>
#include <cmath>

#include "tiling_handler.h"
#include "op_tiling_util.h"
#include "graph/utils/op_desc_utils.h"

namespace optiling {
namespace {
const std::unordered_map<int64_t, int64_t> SPLIT_FACTORS {
    {1, 32767},
    {2, 32767},
    {4, 16383},
    {8, 8191},
};

// depad dtype
constexpr int64_t DTYPE_B8 = 1;
constexpr int64_t DTYPE_B16 = 2;
constexpr int64_t DTYPE_B32 = 4;
constexpr int64_t DTYPE_B64 = 8;

// block size
constexpr int64_t BLOCK_BYTES = 32;

// key pattern
constexpr int64_t BASE_KEY = 500000000;
constexpr int64_t ZERO_SHAPE_BLOCK_DIMS = 1;

// tiling data pattern
constexpr size_t KEY_SIZE = 10;
constexpr int32_t DECIMAL_TEN = 10;
constexpr int32_t X_SHAPE_START = 10000;
constexpr int32_t BEGIN_START = 20000;
constexpr int32_t SIZE_START = 30000;
constexpr int32_t BLOCK_START = 40000;
constexpr int32_t UB_START = 50000;

// COEX IDX
constexpr size_t NORMAL_MODE_IDX = 0;
constexpr size_t DEPAD_MODE_IDX = 1;
constexpr size_t STRIDE_ALIGN_MODE_IDX = 2;

// INPUT IDX
constexpr size_t X_INPUT_DIX = 0;
constexpr size_t BEGIN_INPUT_DIX = 1;
constexpr size_t END_INPUT_DIX = 2;

// END MODE
constexpr int64_t MODE_SIZE = 0;
constexpr int64_t MODE_END = 1;

// MODE
constexpr int64_t MODE_DATA_MOV = 1;
constexpr int64_t MODE_DEPAD = 2;
constexpr int64_t MODE_UNALIGN_STRIDE = 3;
constexpr int64_t MODE_BOTH_ALIGN = 4;
constexpr int64_t MODE_ONE_DIM = 5;
constexpr int64_t MODE_SCALAR = 6;
constexpr int64_t MODE_LR_DEPAD = 7;

// OTHERS
constexpr size_t TWO_DIMS = 2;
constexpr int64_t STRIDE_ALIGN_THRESHOLD = 2048;
constexpr int64_t BLOCK_TILING_THRESHOLD = 2048;
constexpr int64_t ONE_DIM_TILING_THRESHOLD = 1024;
constexpr int64_t IMPROVE_TILING_THRESHOLD = 8;
constexpr int64_t BEFORE_LAST_DIM_THRESHOLD = 16;
}

SliceDslCompileInfo::SliceDslCompileInfo(const std::string &op_type, const nlohmann::json &org_compile_info) {
  try {
    // parase base info
    const auto &base_info = org_compile_info.at("_base_info");
    constexpr size_t base_info_size = 3;
    V_CHECK_EQ(base_info.size(), base_info_size,
               VECTOR_INNER_ERR_REPORT_TILIING(op_type, "base info must be 3 element"), return);
    constexpr size_t core_number_idx = 0;
    constexpr size_t ub_size_idx = 1;
    constexpr size_t x_type_idx = 2;
    core_num = base_info[core_number_idx];
    x_type_size = base_info[x_type_idx];
    int64_t tmp_ub_size = base_info[ub_size_idx];
    ub_size = tmp_ub_size / x_type_size;
    x_align = BLOCK_BYTES / x_type_size;

    if (org_compile_info.count("_is_static") > 0) {
      is_static = org_compile_info.at("_is_static").get<bool>();
    }

    if (org_compile_info.count("_end_mode") > 0) {
      end_mode = org_compile_info.at("_end_mode").get<int64_t>();
    }

    is_const_begins = org_compile_info.count("_const_begins") > 0;
    if (is_const_begins) {
      begin = org_compile_info.at("_const_begins").get<std::vector<int64_t>>();
    }

    is_const_ends = org_compile_info.count("_const_ends") > 0;
    if (is_const_ends) {
      end = org_compile_info.at("_const_ends").get<std::vector<int64_t>>();
    }

    is_const_sizes = org_compile_info.count("_const_sizes") > 0;
    if (is_const_sizes) {
      size = org_compile_info.at("_const_sizes").get<std::vector<int64_t>>();
    }

    if (org_compile_info.count("_slice_vars") > 0) {
      slice_vars = org_compile_info.at("_slice_vars").get<std::unordered_map<std::string, std::vector<int32_t>>>();
    }

    if (org_compile_info.count("_coex_list") > 0) {
      coex_list = org_compile_info.at("_coex_list").get<std::vector< int64_t>>();
    }

    if (org_compile_info.count("_is_const") > 0) {
      is_const = org_compile_info.at("_is_const").get<bool>();
      if (is_const) {
        const auto &const_info = org_compile_info.at("_const_info");
        constexpr size_t const_key_idx = 0;
        constexpr size_t const_block_dims_idx = 1;
        const_key = const_info[const_key_idx];
        const_block_dims = const_info[const_block_dims_idx];
      }
    }
  } catch (const std::exception &e) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "construct compile_info error. Error message: %s", e.what());
  }
  return;
}

bool SliceDsl::Init() {
  const ge::GeShape &org_x_ge_shape = ge::OpDescUtils::GetOpDescFromOperator(op_paras)->
      MutableInputDesc(X_INPUT_DIX)->MutableShape();
  size_t cur_x_dim_len = org_x_ge_shape.GetDimNum();
  if (cur_x_dim_len == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "slice x_shape values is empty.");
    return false;
  }
  for (size_t j = 0; j < cur_x_dim_len; j++) {
    input_x_shape[j] = org_x_ge_shape.GetDim(j);
  }
  input_x_shape.resize(cur_x_dim_len);

  if (slice_compile_info.is_static) {
    x_shape = input_x_shape;
    begin_list = slice_compile_info.begin;
    size_list = slice_compile_info.size;
    shape_len = cur_x_dim_len;
  } else {
    std::vector<int64_t> org_begin_list = {};
    if (slice_compile_info.is_const_begins) {
      org_begin_list = slice_compile_info.begin;
    } else {
      if (!ops::GetConstIntData(op_paras, BEGIN_INPUT_DIX, org_begin_list)) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get begin values failed.");
        return false;
      }
    }

    // get size
    std::vector<int64_t> org_size_list = {};
    bool is_thrid_input_const = slice_compile_info.is_const_sizes || slice_compile_info.is_const_ends;
    if (is_thrid_input_const) {
      org_size_list = slice_compile_info.size;
    } else {
      if (!ops::GetConstIntData(op_paras, END_INPUT_DIX, org_size_list)) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get size values failed.");
        return false;
      }
    }

    if (slice_compile_info.end_mode == MODE_END) {
      // change end to size
      for (size_t idx = 0; idx < org_size_list.size(); idx++) {
        org_size_list[idx] = org_size_list[idx] - org_begin_list[idx];
      }
    }

    SimplyShape(input_x_shape, org_begin_list, org_size_list);
  }

  last_dim = size_list[shape_len - 1];
  last_dim_align =
      (last_dim + slice_compile_info.x_align - 1) / slice_compile_info.x_align * slice_compile_info.x_align;

  x_last_dim_align = (x_shape[shape_len - 1] + slice_compile_info.x_align - 1) /
      slice_compile_info.x_align * slice_compile_info.x_align;

  pre_dim = std::accumulate(size_list.begin(), size_list.end() - 1, 1LL, std::multiplies<int64_t>());

  // check if zeros shape
  total_size = std::accumulate(size_list.begin(), size_list.end(), 1LL, std::multiplies<int64_t>());

  return true;
}

void SliceDsl::SimplyShape(std::vector<int64_t> org_x_shape,
                           std::vector<int64_t> org_begin_list,
                           std::vector<int64_t> org_size_list) {
  std::array<int64_t, SLICE_INIT_DIM_LEN> middle_x_shape{};
  std::array<int64_t, SLICE_INIT_DIM_LEN> middle_begin_list{};
  std::array<int64_t, SLICE_INIT_DIM_LEN> middle_size_list{};
  size_t real_len = 0;
  for (size_t i = 0; i < org_begin_list.size(); i++) {
    int64_t tmp_begin_value = org_begin_list[i];
    if (tmp_begin_value < 0) {
      tmp_begin_value = tmp_begin_value + org_x_shape[i];
    }
    int64_t tmp_size_value = org_size_list[i];
    if (tmp_size_value == -1) {
      tmp_size_value = org_x_shape[i] - tmp_begin_value;
    }

    if (org_x_shape[i] == tmp_size_value) {
      if (i == 0) {
        middle_x_shape[real_len] = org_x_shape[i];
        middle_begin_list[real_len] = tmp_begin_value;
        middle_size_list[real_len] = tmp_size_value;
        real_len = real_len + 1;
      } else {
        middle_x_shape[real_len - 1] = org_x_shape[i] * middle_x_shape[real_len - 1];
        middle_begin_list[real_len - 1] = org_x_shape[i] * middle_begin_list[real_len - 1];
        middle_size_list[real_len - 1] = org_x_shape[i] * middle_size_list[real_len - 1];
      }
    } else {
      middle_x_shape[real_len] = org_x_shape[i];
      middle_begin_list[real_len] = tmp_begin_value;
      middle_size_list[real_len] = tmp_size_value;
      real_len = real_len + 1;
    }
  }

  // fused all slice 1
  size_t x_shape_idx = 0;
  x_shape[x_shape_idx] = middle_x_shape[0];
  begin_list[x_shape_idx] = middle_begin_list[0];
  size_list[x_shape_idx] = middle_size_list[0];
  x_shape_idx = x_shape_idx + 1;
  for (size_t i = 1; i < real_len; i++) {
    if ((size_list[x_shape_idx - 1] == 1) && (middle_size_list[i] == 1)) {
      x_shape[x_shape_idx - 1] = middle_x_shape[i] * x_shape[x_shape_idx - 1];
      begin_list[x_shape_idx - 1] = middle_x_shape[i] * begin_list[x_shape_idx - 1] + middle_begin_list[i];
    } else {
      x_shape[x_shape_idx] = middle_x_shape[i];
      begin_list[x_shape_idx] = middle_begin_list[i];
      size_list[x_shape_idx] = middle_size_list[i];
      x_shape_idx = x_shape_idx + 1;
    }
  }

  shape_len = x_shape_idx;
  x_shape.resize(shape_len);
  begin_list.resize(shape_len);
  size_list.resize(shape_len);
  return;
}

bool SliceDsl::DoBaseTiling() {
  if (shape_len == 1) {
    ub_available = slice_compile_info.ub_size / slice_compile_info.coex_list[NORMAL_MODE_IDX] /
        slice_compile_info.x_align * slice_compile_info.x_align;
    mode = MODE_ONE_DIM;
  } else if (IsBothAlignTiling()) {
    mode = MODE_BOTH_ALIGN;
  } else if (IsLRDePadTiling()) {
    mode = MODE_LR_DEPAD;
  } else if (IsDePadTiling(last_dim_align)) {
    mode = MODE_DEPAD;
  } else if (IsStrideAlignTiling()) {
    mode = MODE_UNALIGN_STRIDE;
  } else {
    ub_available = slice_compile_info.ub_size / slice_compile_info.coex_list[NORMAL_MODE_IDX] /
        slice_compile_info.x_align * slice_compile_info.x_align;
    mode = MODE_DATA_MOV;
  }

  OP_LOGD(op_type.c_str(),
          "slice dsl DoBaseTiling: mode=%lld, ub_size=%lld, coex_list[0]=%lld, "
          "coex_list[1]=%lld, coex_list[2]=%lld, ub_available=%lld",
          mode, slice_compile_info.ub_size, slice_compile_info.coex_list[NORMAL_MODE_IDX],
          slice_compile_info.coex_list[DEPAD_MODE_IDX],
          slice_compile_info.coex_list[STRIDE_ALIGN_MODE_IDX], ub_available);

  return true;
}

bool SliceDsl::IsBothAlignTiling() {
  ub_available = slice_compile_info.ub_size / slice_compile_info.coex_list[NORMAL_MODE_IDX] /
      slice_compile_info.x_align * slice_compile_info.x_align;
  return (last_dim % slice_compile_info.x_align == 0) && (x_shape[shape_len - 1] % slice_compile_info.x_align == 0);
}

bool SliceDsl::IsLRDePadTiling() {
  return (shape_len == TWO_DIMS && size_list[0] == x_shape[0] &&
      begin_list[1] == 0 && IsDePadTiling(x_last_dim_align));
}

bool SliceDsl::IsDePadTiling(int64_t last_dim_align_value) {
  ub_available = slice_compile_info.ub_size / slice_compile_info.coex_list[DEPAD_MODE_IDX] /
      slice_compile_info.x_align * slice_compile_info.x_align;

  // b32
  constexpr int64_t b32_last_dim_max = 168;
  constexpr int64_t b32_co2co = 128;
  int64_t last_dim_max = b32_last_dim_max;
  int64_t co2co = b32_co2co;

  if (slice_compile_info.x_type_size == DTYPE_B8) {
    // b8
    constexpr int64_t b8_last_dim_max = 64;
    constexpr int64_t b8_co2co = 1024;
    last_dim_max = b8_last_dim_max;
    co2co = b8_co2co;
  } else if (slice_compile_info.x_type_size == DTYPE_B16) {
    // b16
    constexpr int64_t b16_last_dim_max = 160;
    constexpr int64_t b16_co2co = 256;
    last_dim_max = b16_last_dim_max;
    co2co = b16_co2co;
  } else if (slice_compile_info.x_type_size == DTYPE_B64) {
    // b64
    constexpr int64_t b64_last_dim_max = 168;
    constexpr int64_t b64_co2co = 64;
    last_dim_max = b64_last_dim_max;
    co2co = b64_co2co;
  }
  bool is_params_rows_ok = last_dim_align_value <= last_dim_max;
  bool is_params_num_ok = co2co * last_dim_align_value <= ub_available;

  OP_LOGD(op_type.c_str(),
          "slice dsl IsDepadTiling: last_dim_align_value=%lld, last_dim_max=%lld, co2co=%lld, "
          "ub_available=%lld",
          last_dim_align_value, last_dim_max, co2co, ub_available);

  return is_params_rows_ok && is_params_num_ok && shape_len >= TWO_DIMS && last_dim % slice_compile_info.x_align != 0;
}

bool SliceDsl::IsStrideAlignTiling() {
  ub_available = slice_compile_info.ub_size / slice_compile_info.coex_list[STRIDE_ALIGN_MODE_IDX] /
      slice_compile_info.x_align * slice_compile_info.x_align;
  return last_dim <= STRIDE_ALIGN_THRESHOLD && shape_len >= TWO_DIMS
      && pre_dim >= (slice_compile_info.core_num * BEFORE_LAST_DIM_THRESHOLD)
      && last_dim % slice_compile_info.x_align != 0;
}

bool SliceDsl::DoBlockUbTiling() {
  bool tiling_ret = DoBlockTiling();
  tiling_ret = tiling_ret && DoUbTiling();
  tiling_ret = tiling_ret && TryImproveTiling();
  return tiling_ret;
}

bool SliceDsl::DoBlockTiling() {
  int64_t second_total_size = std::accumulate(size_list.begin() + 1,
                                              size_list.end(), 1LL, std::multiplies<int64_t>());
  // block tiling
  if ((second_total_size >= slice_compile_info.x_align) || (total_size >= BLOCK_TILING_THRESHOLD)) {
    for (size_t i = 0; i < shape_len; i++) {
      if (size_list[i] > 1) {
        block_axis = i;
        block_factor = (size_list[block_axis] + slice_compile_info.core_num - 1) / slice_compile_info.core_num;
        // get block
        int64_t tmp_core_num = std::accumulate(size_list.begin() + i + 1,
                                               size_list.end(), 1LL, std::multiplies<int64_t>());
        if (tmp_core_num == 0) {
          VECTOR_INNER_ERR_REPORT_TILIING(op_type, "slice x_shape contains 0 and run normal tiling.");
          return false;
        }
        int64_t min_block_value = (slice_compile_info.x_align + tmp_core_num - 1) / tmp_core_num;
        block_factor = std::max(min_block_value, block_factor);
        block_factor = std::min(size_list[block_axis], block_factor);
        if (block_factor == 0) {
          VECTOR_INNER_ERR_REPORT_TILIING(op_type, "slice block_factor is 0.");
          return false;
        }
        block_dims = (size_list[block_axis] + block_factor - 1) / block_factor;
        break;
      }
    }
  } else {
    block_axis = 0;
    block_factor = size_list[block_axis];
    block_dims = 1;
  }
  OP_LOGD(op_type.c_str(),
          "slice dsl DoBlockUbTiling: block_axis=%lld, block_factor=%lld, block_dims=%lld",
          block_axis, block_factor, block_dims);
  return true;
}

bool SliceDsl::DoUbTiling() {
  int64_t last_dim_factor = last_dim_align;
  if (mode == MODE_LR_DEPAD) {
    last_dim_factor = x_last_dim_align;
  }
  if (block_axis == shape_len - 1) {
    last_dim_factor = block_factor;
  }
  OP_LOGD(op_type.c_str(), "slice dsl DoBlockUbTiling: last_dim_factor=%lld", last_dim_factor);

  // ub tiling
  int64_t row_num = ub_available / last_dim_factor;
  OP_LOGD(op_type.c_str(), "slice dsl DoBlockUbTiling: row_num=%lld", row_num);
  if (row_num == 0) {
    ub_axis = shape_len - 1;
    ub_factor = ub_available;
  } else {
    ub_axis = block_axis;
    ub_factor = block_factor;
    if (block_axis < shape_len - 1) {
      if (block_axis == shape_len - TWO_DIMS) {
        int64_t tmp_ub_num = (block_factor + row_num - 1) / row_num;
        ub_factor = block_factor / tmp_ub_num;
      } else {
        int64_t multi_num = 1;
        for (size_t i = shape_len - TWO_DIMS; i > block_axis; i--) {
          if (multi_num * size_list[i] >= row_num) {
            ub_axis = i;
            int64_t row_num_i = row_num / multi_num;
            int64_t tmp_ub_num = (size_list[i] + row_num_i - 1) / row_num_i;
            ub_factor = (size_list[i] + tmp_ub_num - 1) / tmp_ub_num;
            multi_num = multi_num * size_list[i];
            break;
          }
          multi_num = multi_num * size_list[i];
        }

        if (multi_num < row_num) {
          if (multi_num * block_factor > row_num) {
            ub_axis = block_axis;
            int64_t row_num_i = row_num / multi_num;
            int64_t tmp_ub_num = (block_factor + row_num_i - 1) / row_num_i;
            ub_factor = (block_factor + tmp_ub_num - 1) / tmp_ub_num;
          }
        }
      }
    }
  }

  // ub last dim only support data mov mode
  if (mode != MODE_DATA_MOV && ub_axis == shape_len - 1) {
    mode = MODE_DATA_MOV;
  }

  return true;
}

bool SliceDsl::TryImproveTiling() {
  OP_LOGD(op_type.c_str(),
          "slice dsl DoBlockUbTiling before improve: "
          "block_dims=%lld, block_axis=%lld, ub_axis=%lld, ub_factor==%lld",
          block_dims, block_axis, ub_axis, ub_factor);
  // try to ensure every ub data size align
  if ((ub_axis == (shape_len - TWO_DIMS)) && (ub_factor > IMPROVE_TILING_THRESHOLD * slice_compile_info.x_align) &&
      !(ub_axis == block_axis && ub_factor == block_factor)) {
    int64_t tmp_ub_factor = ub_factor / slice_compile_info.x_align * slice_compile_info.x_align;
    int64_t tail_ub_factor = 0;
    if (ub_axis == block_axis) {
      tail_ub_factor = block_factor % tmp_ub_factor;
    } else {
      tail_ub_factor = size_list[ub_axis] % tmp_ub_factor;
    }

    if (tail_ub_factor == 0) {
      tail_ub_factor = tmp_ub_factor;
    }

    if (tail_ub_factor * size_list[ub_axis + 1] >= slice_compile_info.x_align) {
      ub_factor = tmp_ub_factor;
    }
  }

  // if row number to small or scalar condtion change mode
  if (mode == MODE_DEPAD || mode == MODE_UNALIGN_STRIDE) {
    // cal row number
    int64_t second_last_factor = ub_factor;
    if (ub_axis < shape_len - TWO_DIMS) {
      second_last_factor = second_last_factor * std::accumulate(size_list.begin() + ub_axis + 1,
                                                                size_list.end() - 1, 1LL, std::multiplies<int64_t>());
    }

    if (second_last_factor < BEFORE_LAST_DIM_THRESHOLD) {
      mode = last_dim == 1 ? MODE_SCALAR : MODE_DATA_MOV;
    }
  }

  OP_LOGD(op_type.c_str(), "slice dsl DoBlockUbTiling after improve: "
                           "block_dims=%lld, block_axis=%lld, block_factor=%lld, ub_axis=%lld, ub_factor=%lld",
          block_dims, block_axis, block_factor, ub_axis, ub_factor);

  return true;
}

bool SliceDsl::DoOneDimTiling() {
  // one dim tiling
  int64_t dim_value = size_list[0];
  block_axis = 0;
  ub_axis = 0;
  if (dim_value > ONE_DIM_TILING_THRESHOLD) {
    // block tiling
    block_factor = std::ceil(dim_value * 1.0 / slice_compile_info.core_num);
    block_factor = std::ceil(block_factor * 1.0 / slice_compile_info.x_align) * slice_compile_info.x_align;
    if (block_factor == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "slice block_factor is 0.");
      return false;
    }
    block_dims = std::ceil(dim_value * 1.0 / block_factor);
    // ub tiling
    int64_t limit = std::min(ub_available, SPLIT_FACTORS.at(slice_compile_info.x_type_size));
    if (limit == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "slice limit is 0.");
      return false;
    }
    int64_t ub_for_num = std::ceil(block_factor * 1.0 / limit);
    if (ub_for_num == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "slice ub_for_num is 0.");
      return false;
    }
    int64_t adjust_factor = std::ceil(block_factor * 1.0 / ub_for_num);
    int64_t align_factor = std::ceil(adjust_factor * 1.0 / slice_compile_info.x_align);
    ub_factor = align_factor * slice_compile_info.x_align;
    if (ub_factor > limit) {
      ub_factor = std::floor(adjust_factor * 1.0 / slice_compile_info.x_align) * slice_compile_info.x_align;
    }
  } else {
    block_dims = 1;
    block_factor = dim_value;
    ub_factor = dim_value;
  }

  return true;
}

bool SliceDsl::CalcKey() {
  // base key
  key = BASE_KEY;

  constexpr int64_t shape_len_value = 10000000;
  constexpr int64_t mode_value = 1000000;
  key += shape_len * shape_len_value + mode * mode_value;

  // split info
  key += block_axis * shape_len + ub_axis;
  OP_LOGD(op_type.c_str(), "slice dsl CalcKey key=%lld", key);
  return true;
}

bool SliceDsl::WriteTilingData() {
  OP_LOGD(op_type.c_str(), "tiling key:%lld block_dims:%lld block_factor:%lld ub_factor:%lld "
                           "block_axis:%lld ub_axis:%lld",
          key, block_dims, block_factor, ub_factor, block_axis, ub_axis);

  if (slice_compile_info.is_static) {
    run_info.AddTilingData(static_cast<uint32_t>(key));
    run_info.AddTilingData(static_cast<int32_t>(block_axis));
    run_info.AddTilingData(static_cast<int32_t>(block_factor));
    run_info.AddTilingData(static_cast<int32_t>(ub_axis));
    run_info.AddTilingData(static_cast<int32_t>(ub_factor));
    run_info.AddTilingData(static_cast<int32_t>(mode));
    run_info.AddTilingData(static_cast<int32_t>(block_dims));
    return true;
  }

  run_info.SetBlockDim(static_cast<uint32_t>(block_dims));
  run_info.SetTilingKey(static_cast<uint32_t>(key));

  int64_t cur_key = key;
  int64_t key_len = 8;

  char keys[KEY_SIZE] = {'0', '0', '0', '0', '0', '0', '0', '0', '0', '\0'};
  while (cur_key && key_len >= 0) {
    keys[key_len] = '0' + cur_key % DECIMAL_TEN;
    cur_key /= DECIMAL_TEN;
    key_len--;
  }
  std::string str_key = keys;

  try {
    const auto &all_vars = slice_compile_info.slice_vars.at(str_key);
    for (const auto &var : all_vars) {
      if (var >= UB_START) {
        run_info.AddTilingData(static_cast<int32_t>(ub_factor));
      } else if (var >= BLOCK_START) {
        run_info.AddTilingData(static_cast<int32_t>(block_factor));
      } else if (var >= SIZE_START) {
        int64_t var_value = var;
        size_t dim_index = var_value % DECIMAL_TEN;
        run_info.AddTilingData(static_cast<int32_t>(size_list[dim_index]));
      } else if (var >= BEGIN_START) {
        int64_t var_value = var;
        size_t dim_index = var_value % DECIMAL_TEN;
        run_info.AddTilingData(static_cast<int32_t>(begin_list[dim_index]));
      } else {
        int64_t var_value = var;
        size_t dim_index = var_value % DECIMAL_TEN;
        run_info.AddTilingData(static_cast<int32_t>(x_shape[dim_index]));
      }
    }
  } catch (const std::exception &e) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile info[_slice_vars] error, error message: %s", e.what());
    return false;
  }
  return true;
}

bool SliceDsl::DoTiling() {
  OP_LOGD(op_type.c_str(), "slice dsl DoTiling");
  if (slice_compile_info.is_const) {
    run_info.SetBlockDim(static_cast<uint32_t>(slice_compile_info.const_block_dims));
    run_info.SetTilingKey(static_cast<uint32_t>(slice_compile_info.const_key));
    return true;
  }

  bool init_ret = Init();
  if (!init_ret) {
    return false;
  }

  if (total_size == 0) {
    run_info.SetBlockDim(static_cast<uint32_t>(ZERO_SHAPE_BLOCK_DIMS));
    run_info.SetTilingKey(static_cast<uint32_t>(BASE_KEY));
    return true;
  }

  bool tiling_ret = false;
  tiling_ret = DoBaseTiling();
  if (mode == MODE_ONE_DIM) {
    tiling_ret = DoOneDimTiling();
  } else {
    tiling_ret = DoBlockUbTiling();
  }
  tiling_ret = tiling_ret && CalcKey();
  tiling_ret = tiling_ret && WriteTilingData();

  return tiling_ret;
}

bool SliceTilingHandler::DoTiling(const ge::Operator &op_paras, utils::OpRunInfo &run_info) const {
  OP_LOGD(op_type.c_str(), "SliceTilingHandler DoTiling running");
  SliceDsl SliceDsl(op_type, op_paras, slice_compile_info, run_info);
  return SliceDsl.DoTiling();
}

bool SliceTilingHandler::DoTiling(const ge::Operator &op_paras, utils::OpRunInfo &run_info,
                                  const OpInfo &op_info) const {
  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Slice custom tiling is not supported yet");
  return false;
}

std::shared_ptr<AutoTilingHandler> CreateSliceTilingHandler(const std::string &op_type,
                                                            const std::string &pattern,
                                                            const nlohmann::json &parsed_compile_info) {
  return std::make_shared<SliceTilingHandler>(op_type, pattern, parsed_compile_info);
}
} // namespace optiling

