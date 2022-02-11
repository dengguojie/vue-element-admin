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
 * \file gather_schedule.cpp
 * \brief
 */
#include "gather_dsl.h"
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <iostream>

#include "tiling_handler.h"
#include "op_tiling_util.h"
#include "graph/utils/op_desc_utils.h"

namespace optiling {
namespace {
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t DOUBLE_BUFFER_SIZE = 2;
constexpr int64_t GATHER_ND_COMPUTE = 1;
constexpr int64_t GATHER_V2_COMPUTE = 2;
constexpr int64_t INDICES_COEXS = 1;
constexpr int64_t TENSOR_SWELLS_NUM = 4;
constexpr int64_t GATHER_V2_INPUTS_NUM = 3;
constexpr int64_t UB_HALF_DIV_NUM = 2;

// IDX
constexpr size_t PARAMS_BATCH_DIM_IDX = 0;
constexpr size_t PARAMS_LOOP_IDX = 1;
constexpr size_t PARAMS_AXIS_IDX = 2;
constexpr size_t PARAMS_ROWS_IDX = 3;
constexpr size_t PARAMS_SHAPE_SIZE = 4;
constexpr size_t INDICES_BATCH_DIM_IDX = 0;
constexpr size_t INDICES_LOOP_IDX = 1;
constexpr size_t INDICES_RANK_IDX = 2;
constexpr size_t INDICES_SHAPE_SIZE = 3;
constexpr size_t OUTPUT_BATCH_DIM_IDX = 0;
constexpr size_t OUTPUT_PARAMS_PRE_LOOP_IDX = 1;
constexpr size_t OUTPUT_INDICES_LOOP_IDX = 2;
constexpr size_t OUTPUT_PARAMS_ROW_IDX = 3;
constexpr size_t OUTPUT_SHAPE_SIZE = 4;
constexpr int64_t INPUT_AXIS_IDX = 2;

// TENSOR_SIZE
constexpr size_t TENSOR_SIZES_NUM = 2;
constexpr size_t TENSOR_SIZES_PARAMS_IDX = 0;
constexpr size_t TENSOR_SIZES_INDICES_IDX = 1;

// BLOCK_TILING_AXIS
constexpr size_t BLOCK_TILING_FIRST_AXIS = 0;
constexpr size_t BLOCK_TILING_SECOND_AXIS = 1;
constexpr size_t BLOCK_TILING_THIRD_AXIS = 2;
constexpr size_t BLOCK_TILING_LAST_AXIS = 3;

// UB_TILING_AXIS
constexpr size_t UB_TILING_FIRST_AXIS = 0;
constexpr size_t UB_TILING_SECOND_AXIS = 1;
constexpr size_t UB_TILING_THIRD_AXIS = 2;
constexpr size_t UB_TILING_LAST_AXIS = 3;

// SCHEDULE ENUM TYPE
const std::string BASE_SCHEDULE = "0";
const std::string PARAMS_UB_ALIGN_SCHEDULE = "1";
const std::string PARAMS_UB_NOT_ALIGN_SCHEDULE = "2";
const std::string DB_SCHEDULE = "5";
const std::string SCALAR_SCHEDULE = "6";
const std::string DEPAD_SCHEDULE = "7";

// KEY && TILING_KEY
constexpr int32_t PARAMS_UB_ALIGN_KEY = 1000;
constexpr int32_t PARAMS_UB_NO_ALIGN_KEY = 2000;
constexpr int32_t DB_MODULE_TILING_KEY = 5000;
constexpr int32_t SCALAR_TILING_KEY = 6000;
constexpr int32_t DEPAD_TILING_KEY = 7000;

constexpr int64_t BASE_KEY = 900000000;
constexpr int64_t ZERO_TILING_KEY = 990000000;
constexpr int64_t BROADCAST_TILING_KEY = 990000001;

// PARAM_DTYPE
constexpr int64_t PARAM_DTYPE_B8 = 1;
constexpr int64_t PARAM_DTYPE_B16 = 2;
constexpr int64_t PARAM_DTYPE_B64 = 8;

constexpr size_t KEY_SIZE = 10;
constexpr int32_t DECIMAL_TEN = 10;
constexpr int32_t INDICES_SHAPE_IDX = 20000;
constexpr int32_t MIN_BLOCK_FACTOR_IDX = 30000;
constexpr int32_t MIN_UB_FACTOR_IDX = 40000;
}
  GatherDslCompileInfo::GatherDslCompileInfo(const std::string &op_type, const nlohmann::json &org_compile_info) {
    try {
      // parse base info
      const auto &base_info = org_compile_info.at("_base_info");
      constexpr size_t base_info_size = 5;
      V_CHECK_EQ(base_info.size(), base_info_size,
                 VECTOR_INNER_ERR_REPORT_TILIING(op_type, "base info must be 6 element"),
                 return);
      constexpr size_t core_number_idx = 0;
      constexpr size_t ub_size_idx = 1;
      constexpr size_t gather_type_idx = 2;
      constexpr size_t params_dtype_idx = 3;
      constexpr size_t indices_dtype_idx = 4;
      constexpr int64_t block_size = 32;
      core_num = base_info[core_number_idx];
      ub_size = base_info[ub_size_idx];
      gather_type = base_info[gather_type_idx];
      params_dtype = base_info[params_dtype_idx];
      params_align = block_size / params_dtype;
      indices_dtype = base_info[indices_dtype_idx];

      // parse custom info
      const auto &custom_info = org_compile_info.at("_custom_info");
      constexpr size_t custom_info_size = 4;
      V_CHECK_EQ(custom_info.size(), custom_info_size,
                 VECTOR_INNER_ERR_REPORT_TILIING(op_type, "custom info must be 5 element"),
                 return);
      constexpr size_t params_ub_store_num_idx = 0;
      constexpr size_t batch_dims_idx = 1;
      constexpr size_t unknown_batch_dims_idx = 2;
      constexpr size_t org_batch_dims_idx = 3;
      params_ub_store_num = custom_info[params_ub_store_num_idx];
      batch_dims = custom_info[batch_dims_idx];
      unknown_batch_dims = custom_info[unknown_batch_dims_idx];
      org_batch_dims = custom_info[org_batch_dims_idx];

      OP_LOGD(op_type.c_str(), "GatherDslCompileInfo:%lld %lld %lld %lld",
              gather_type, batch_dims, unknown_batch_dims, org_batch_dims);

      // tensor sizes for special pattern
      tensor_sizes = org_compile_info.at("_tensor_sizes").get<std::unordered_map<std::string, std::vector<int64_t>>>();

      // const axis
      is_dynamic_const = org_compile_info.count("_const_axis") > 0;
      if (is_dynamic_const) {
        const_axis = org_compile_info.at("_const_axis").get<int32_t>();
      } else {
        // gather vars
        gather_vars = org_compile_info.at("_gather_vars").get<std::unordered_map<std::string, std::vector<int32_t>>>();
      }

      if (unknown_batch_dims) {
        attr_name = org_compile_info.at("attr_name").get<std::string>();
      }
    } catch (const std::exception &e) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "construct compile_info error. Error message: %s", e.what());
    }
    return;
  }

  bool GatherDsl::Init() {
    const ge::GeShape &org_params_ge_shape = ge::OpDescUtils::GetOpDescFromOperator(op_paras)->
            MutableInputDesc(0)->MutableShape();
    cur_params_dim_len = org_params_ge_shape.GetDimNum();
    if (cur_params_dim_len == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "gather org_params_shape values is empty.");
      return false;
    }
    for (size_t j = 0; j < cur_params_dim_len; j++) {
      org_params_shape[j] = org_params_ge_shape.GetDim(j);
    }

    const ge::GeShape &org_indices_ge_shape =
      ge::OpDescUtils::GetOpDescFromOperator(op_paras)->MutableInputDesc(1)->MutableShape();
    cur_indices_dim_len = org_indices_ge_shape.GetDimNum();
    if (cur_indices_dim_len == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "gather org_indices_shape values is empty.");
      return false;
    }
    for (size_t j = 0; j < cur_indices_dim_len; j++) {
      org_indices_shape[j] = org_indices_ge_shape.GetDim(j);
    }

    // gather rank
    if (gather_compile_info.gather_type == GATHER_ND_COMPUTE) {
      rank = org_indices_shape[cur_indices_dim_len - 1];
    }

    // batch dims
    GetRealBatchDims();

    if (gather_compile_info.is_dynamic_const) {
      // const condition shape if fused
      axis = gather_compile_info.const_axis;

      // const condition
      if (gather_compile_info.gather_type == GATHER_ND_COMPUTE) {
        // gather nd [batch, axes, row] [batch, loops, rank]
        params_shape[PARAMS_BATCH_DIM_IDX] = org_params_shape[PARAMS_BATCH_DIM_IDX];
        params_shape[PARAMS_LOOP_IDX] = 1;
        for (size_t i = 1; i < cur_params_dim_len; i++) {
          params_shape[PARAMS_LOOP_IDX + i] = org_params_shape[i];
        }
        params_shape.resize(cur_params_dim_len + 1);

        indices_shape[INDICES_BATCH_DIM_IDX] = org_indices_shape[INDICES_BATCH_DIM_IDX];
        indices_shape[INDICES_LOOP_IDX] = org_indices_shape[INDICES_LOOP_IDX];
        indices_shape[INDICES_RANK_IDX] = org_indices_shape[INDICES_RANK_IDX];

        params_rows = params_shape[cur_params_dim_len];
      } else {
        // gather/gather v2 [batch, pre_loops, axis, row] [batch, loops]
        params_shape[PARAMS_BATCH_DIM_IDX] = org_params_shape[PARAMS_BATCH_DIM_IDX];
        params_shape[PARAMS_LOOP_IDX] = org_params_shape[PARAMS_LOOP_IDX];
        params_shape[PARAMS_AXIS_IDX] = org_params_shape[PARAMS_AXIS_IDX];
        params_shape[PARAMS_ROWS_IDX] = org_params_shape[PARAMS_ROWS_IDX];
        params_shape.resize(PARAMS_SHAPE_SIZE);

        indices_shape[INDICES_BATCH_DIM_IDX] = org_indices_shape[INDICES_BATCH_DIM_IDX];
        indices_shape[INDICES_LOOP_IDX] = org_indices_shape[INDICES_LOOP_IDX];
        indices_shape[INDICES_RANK_IDX] = 1;

        params_rows = params_shape[PARAMS_SHAPE_SIZE - 1];
      }
    } else {
      // dynamic
      // gather_type = 0 gather
      // gather_type = 1 gather nd
      // gather_type = 2 gather v2
      SimplyParamsAndIndices();
    }

    // output shape
    output_shape[OUTPUT_BATCH_DIM_IDX] = params_shape[PARAMS_BATCH_DIM_IDX];
    output_shape[OUTPUT_PARAMS_PRE_LOOP_IDX] = params_shape[PARAMS_LOOP_IDX];
    output_shape[OUTPUT_INDICES_LOOP_IDX] = indices_shape[INDICES_LOOP_IDX];
    output_shape[OUTPUT_PARAMS_ROW_IDX] = params_rows;

    params_rows_align = (params_rows + gather_compile_info.params_align - 1) /
                         gather_compile_info.params_align * gather_compile_info.params_align;

    // cal total size
    params_size_total = std::accumulate(params_shape.begin(), params_shape.end(), 1LL, std::multiplies<int64_t>());
    indices_size_total = indices_shape[INDICES_BATCH_DIM_IDX] * indices_shape[INDICES_LOOP_IDX]
                         * indices_shape[INDICES_RANK_IDX];
    // indices shape batch dims value is same as params shape
    total_size = params_size_total * indices_shape[INDICES_LOOP_IDX];
    output_size = output_shape[OUTPUT_BATCH_DIM_IDX] * output_shape[OUTPUT_PARAMS_PRE_LOOP_IDX] *
                  output_shape[OUTPUT_INDICES_LOOP_IDX] * output_shape[OUTPUT_PARAMS_ROW_IDX];

    OP_LOGD(op_type.c_str(), "GatherDsl output_shape:%lld %lld %lld %lld",
            output_shape[OUTPUT_BATCH_DIM_IDX], output_shape[OUTPUT_PARAMS_PRE_LOOP_IDX],
            output_shape[OUTPUT_INDICES_LOOP_IDX], output_shape[OUTPUT_PARAMS_ROW_IDX]);
    return true;
  }

  void GatherDsl::GetRealBatchDims() {
    int64_t batch_dims = 0;
    if ((gather_compile_info.unknown_batch_dims) && (gather_compile_info.gather_type != GATHER_ND_COMPUTE)) {
      if (ge::GRAPH_SUCCESS !=
        static_cast<int64_t>(op_paras.GetAttr(gather_compile_info.attr_name.c_str(), batch_dims))) {
        OP_LOGW("Gather tiling GetAttr(batch_dims) failed, set default value to 0.");
      }
    } else {
      batch_dims = gather_compile_info.org_batch_dims;
    }

    if (batch_dims < 0) {
      real_batch_dims = batch_dims + cur_indices_dim_len;
    } else {
      real_batch_dims = batch_dims;
    }
    return;
  }

  void GatherDsl::SimplyParamsAndIndices() {
    uint32_t inputs_num = op_paras.GetInputsSize();
    if (inputs_num == GATHER_V2_INPUTS_NUM) {
      std::vector <int64_t> values;
      // input axis index is 2
      if (!ops::GetConstIntData(op_paras, INPUT_AXIS_IDX, values)) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "gather v2 axis not exists.");
        return;
      }

      if (values.empty()) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "gather v2 axis values is empty.");
        return;
      }

      axis = values[0];
      if (axis < 0) {
        axis = axis + cur_params_dim_len;
      }
    } else {
      axis = real_batch_dims;
    }
    // params batch dims
    int64_t params_batch_dims = 1;
    params_batch_dims = std::accumulate(org_params_shape.begin(), org_params_shape.begin() + real_batch_dims, 1LL,
                                        std::multiplies<int64_t>());
    params_shape[PARAMS_BATCH_DIM_IDX] = params_batch_dims;

    // params pre loop
    params_shape[PARAMS_LOOP_IDX] = std::accumulate(org_params_shape.begin() + real_batch_dims,
                                                    org_params_shape.begin() + axis,
                                                    1LL,
                                                    std::multiplies<int64_t>());

    // params gather axis
    for (int k = axis; k < axis + rank; k++) {
      params_shape[k - axis + PARAMS_AXIS_IDX] = org_params_shape[k];
    }

    // params rows
    params_rows = std::accumulate(org_params_shape.begin() + axis + rank,
                                  org_params_shape.begin() + cur_params_dim_len,
                                  1LL,
                                  std::multiplies<int64_t>());

    params_shape[PARAMS_AXIS_IDX + rank] = params_rows;
    params_shape.resize(PARAMS_AXIS_IDX + rank + 1);

    // indices batch dims
    indices_shape[INDICES_BATCH_DIM_IDX] = params_batch_dims;

    int64_t indices_loops = 1;
    if (real_batch_dims < cur_indices_dim_len) {
      indices_loops = std::accumulate(org_indices_shape.begin() + real_batch_dims,
                                      org_indices_shape.begin() + cur_indices_dim_len - 1,
                                      1LL,
                                      std::multiplies<int64_t>());
    }

    if ((gather_compile_info.gather_type != GATHER_ND_COMPUTE) && (real_batch_dims < cur_indices_dim_len)) {
      indices_loops = indices_loops * org_indices_shape[cur_indices_dim_len - 1];
    }

    indices_shape[INDICES_LOOP_IDX] = indices_loops;
    indices_shape[INDICES_RANK_IDX] = rank;

    return;
  }

  bool GatherDsl::IsZeroShapeTiling() {
    // total_size equal 0(zero shape)
    // only rank equal 0(broadcast shape)
    return (total_size == 0) || (rank == 0);
  }

  bool GatherDsl::DoZeroShapeTiling() {
    block_dims = 1;
    if ((gather_compile_info.gather_type == GATHER_ND_COMPUTE) && (total_size != 0) && (rank == 0)) {
      // broadcast shape
      key = BROADCAST_TILING_KEY;
      // real params shape [parmas_batch, params_data], but init is [batch, gather axis, after axis]
      params_shape[PARAMS_LOOP_IDX] = params_shape[PARAMS_LOOP_IDX] * params_shape[PARAMS_AXIS_IDX];
    } else {
      // zero shape
      key = ZERO_TILING_KEY;
    }
    return true;
  }

  bool GatherDsl::IsSpecialPattern() {
    return rank == 1;
  }

  bool GatherDsl::IsDePadTiling() {
    if (gather_compile_info.tensor_sizes.count(DEPAD_SCHEDULE) == 0) {
      return false;
    }

    if (gather_compile_info.tensor_sizes.at(DEPAD_SCHEDULE).size() != TENSOR_SIZES_NUM) {
      return false;
    }

    params_num_ub = gather_compile_info.tensor_sizes.at(DEPAD_SCHEDULE)[TENSOR_SIZES_PARAMS_IDX] / rank;

    real_params_row_num = params_num_ub / params_rows_align;
    if ((real_params_row_num < 1) && (params_rows % gather_compile_info.params_align == 0)) {
      return false;
    }

    bool is_shape_ok = ((params_size_total > gather_compile_info.params_ub_store_num) ||
                       (params_rows > 15 && params_rows < 64)) &&
                       (params_rows % gather_compile_info.params_align > 0);
    if (is_shape_ok) {
      // b32
      constexpr int64_t b32_last_dim_max = 168;
      constexpr int64_t b32_co2co = 128;
      int64_t last_dim_max = b32_last_dim_max;
      int64_t co2co = b32_co2co;

      if (gather_compile_info.params_dtype == PARAM_DTYPE_B8) {
        // b8
        constexpr int64_t b8_last_dim_max = 64;
        constexpr int64_t b8_co2co = 1024;
        last_dim_max = b8_last_dim_max;
        co2co = b8_co2co;
      } else if (gather_compile_info.params_dtype == PARAM_DTYPE_B16) {
        // b16
        constexpr int64_t b16_last_dim_max = 160;
        constexpr int64_t b16_co2co = 256;
        last_dim_max = b16_last_dim_max;
        co2co = b16_co2co;
      } else if (gather_compile_info.params_dtype == PARAM_DTYPE_B64) {
        // b64
        constexpr int64_t b64_last_dim_max = 168;
        constexpr int64_t b64_co2co = 64;
        last_dim_max = b64_last_dim_max;
        co2co = b64_co2co;
      }
      bool is_params_rows_ok = params_rows <= last_dim_max;
      bool is_params_num_ok = co2co * params_rows_align <= params_num_ub;

      return is_params_rows_ok && is_params_num_ok;
    }
    return false;
  }

  bool GatherDsl::DoDePadTiling() {
    DoBaseTiling();
    key_special_pattern = DEPAD_TILING_KEY;

    return true;
  }

  bool GatherDsl::IsScalarTiling() {
    if (gather_compile_info.tensor_sizes.count(SCALAR_SCHEDULE) == 0) {
      return false;
    }

    if (gather_compile_info.tensor_sizes.at(SCALAR_SCHEDULE).size() != TENSOR_SIZES_NUM) {
      return false;
    }

    params_num_ub = gather_compile_info.tensor_sizes.at(SCALAR_SCHEDULE)[TENSOR_SIZES_PARAMS_IDX] / rank;
    indices_num_ub = gather_compile_info.tensor_sizes.at(SCALAR_SCHEDULE)[TENSOR_SIZES_INDICES_IDX] / rank;
    real_params_row_num = params_num_ub / params_rows_align;

    return (params_size_total < gather_compile_info.params_ub_store_num) &&
           (params_rows <= 15) &&
           (params_rows % gather_compile_info.params_align > 0);
  }

  bool GatherDsl::DoScalarTiling() {
    DoBaseTiling();
    key_special_pattern = SCALAR_TILING_KEY;

    return true;
  }

  bool GatherDsl::IsStoreUB(int64_t params_total) {
    return params_total < gather_compile_info.params_ub_store_num;
  }

  bool GatherDsl::IsParamsUbTiling() {
    if ((gather_compile_info.tensor_sizes.count(PARAMS_UB_ALIGN_SCHEDULE) == 0 ||
        gather_compile_info.tensor_sizes.at(PARAMS_UB_ALIGN_SCHEDULE).size() != TENSOR_SIZES_NUM) &&
        (gather_compile_info.tensor_sizes.count(PARAMS_UB_NOT_ALIGN_SCHEDULE) == 0 ||
        gather_compile_info.tensor_sizes.at(PARAMS_UB_NOT_ALIGN_SCHEDULE).size() != TENSOR_SIZES_NUM)) {
      return false;
    }

    if (gather_compile_info.tensor_sizes.count(PARAMS_UB_ALIGN_SCHEDULE) > 0) {
      params_num_ub = gather_compile_info.tensor_sizes.at(PARAMS_UB_ALIGN_SCHEDULE)[TENSOR_SIZES_PARAMS_IDX] / rank;
    } else {
      params_num_ub = gather_compile_info.tensor_sizes.at(PARAMS_UB_NOT_ALIGN_SCHEDULE)[TENSOR_SIZES_PARAMS_IDX] / rank;
    }

    real_params_row_num = params_num_ub / params_rows_align;
    if (real_params_row_num < 1) {
      return false;
    }

    int64_t params_size_total_align = params_size_total / params_rows * params_rows_align;
    return IsStoreUB(params_size_total_align);
  }

  bool GatherDsl::DoParamsUbTiling() {
    // check need align
    DoBaseTiling();
    if ((block_axis == OUTPUT_PARAMS_ROW_IDX) && (block_factor % gather_compile_info.params_align != 0)) {
      return false;
    }
    if (((params_rows * gather_compile_info.params_dtype) % BLOCK_SIZE) == 0) {
      key_special_pattern = PARAMS_UB_ALIGN_KEY;
    } else {
      key_special_pattern = PARAMS_UB_NO_ALIGN_KEY;
    }

    return true;
  }

  bool GatherDsl::IsDbModule() {
    if (gather_compile_info.tensor_sizes.count(DB_SCHEDULE) == 0) {
      return false;
    }

    if (gather_compile_info.tensor_sizes.at(DB_SCHEDULE).size() != TENSOR_SIZES_NUM) {
      return false;
    }

    params_num_ub = gather_compile_info.tensor_sizes.at(DB_SCHEDULE)[TENSOR_SIZES_PARAMS_IDX] / rank;

    real_params_row_num = params_num_ub / params_rows_align;

    return params_rows > gather_compile_info.params_align && params_rows % gather_compile_info.params_align == 0;
  }

  bool GatherDsl::DoDbModule() {
    DoBaseTiling();
    key_special_pattern = DB_MODULE_TILING_KEY;
    return true;
  }

  bool GatherDsl::IsBaseTiling() {
    if (gather_compile_info.tensor_sizes.count(BASE_SCHEDULE) == 0) {
      return false;
    }

    if (gather_compile_info.tensor_sizes.at(BASE_SCHEDULE).size() != TENSOR_SIZES_NUM) {
      return false;
    }

    params_num_ub = gather_compile_info.tensor_sizes.at(BASE_SCHEDULE)[TENSOR_SIZES_PARAMS_IDX] / rank;
    indices_num_ub = gather_compile_info.tensor_sizes.at(BASE_SCHEDULE)[TENSOR_SIZES_INDICES_IDX] / rank;

    real_params_row_num = params_num_ub / params_rows_align;

    return true;
  }

  void GatherDsl::BlockFirstAxis() {
    real_params_row_num = params_num_ub / params_rows_align;

    block_axis = BLOCK_TILING_FIRST_AXIS;
    int64_t under_block = output_shape[OUTPUT_PARAMS_PRE_LOOP_IDX] *
            output_shape[OUTPUT_INDICES_LOOP_IDX] * params_rows;
    int64_t block_tmp = std::min((gather_compile_info.params_align + under_block -1) / under_block,
                                 output_shape[block_axis]);
    block_factor =
      std::max((output_shape[block_axis] + gather_compile_info.core_num - 1) / gather_compile_info.core_num, block_tmp);
    if (block_factor == 0) {
      return;
    }
    block_dims = (output_shape[block_axis] + block_factor - 1) / block_factor;

    // ub factor
    int64_t temp_ub_times;
    if (real_params_row_num == 0) {
      ub_axis = UB_TILING_LAST_AXIS;
      temp_ub_times = (output_shape[ub_axis] + params_num_ub - 1) / params_num_ub;
      ub_factor = (output_shape[ub_axis] + temp_ub_times - 1) / temp_ub_times;
    } else {
      if (real_params_row_num < output_shape[OUTPUT_INDICES_LOOP_IDX]) {
        ub_axis = UB_TILING_THIRD_AXIS;
        temp_ub_times = (output_shape[ub_axis] + real_params_row_num - 1) / real_params_row_num;
        ub_factor = (output_shape[ub_axis] + temp_ub_times - 1) / temp_ub_times;
      } else {
        // ub axis = 1
        int64_t real_axis2_params_row_num = real_params_row_num / output_shape[OUTPUT_INDICES_LOOP_IDX];
        if (real_axis2_params_row_num < output_shape[OUTPUT_PARAMS_PRE_LOOP_IDX]) {
          ub_axis = UB_TILING_SECOND_AXIS;
          temp_ub_times = (output_shape[ub_axis] + real_axis2_params_row_num - 1) / real_axis2_params_row_num;
          ub_factor = (output_shape[ub_axis] + temp_ub_times - 1) / temp_ub_times;
        } else {
          // ub axis = 0
          int64_t real_axis1_params_row_num = real_axis2_params_row_num / output_shape[1];
          ub_axis = UB_TILING_FIRST_AXIS;
          if (real_axis1_params_row_num < block_factor) {
            temp_ub_times = (block_factor + real_axis1_params_row_num - 1) / real_axis1_params_row_num;
            ub_factor = (block_factor + temp_ub_times - 1) / temp_ub_times;
          } else {
            ub_factor = block_factor;
          }
        }
      }
    }
    return;
  }

  void GatherDsl::BlockSecondAxis() {
    real_params_row_num = params_num_ub / params_rows_align;

    block_axis = BLOCK_TILING_SECOND_AXIS;
    int64_t under_block = output_shape[OUTPUT_INDICES_LOOP_IDX] * params_rows;
    int64_t block_tmp = std::min((gather_compile_info.params_align + under_block -1) / under_block,
                                 output_shape[block_axis]);
    block_factor =
      std::max((output_shape[block_axis] + gather_compile_info.core_num - 1) / gather_compile_info.core_num, block_tmp);
    if (block_factor == 0) {
      return;
    }
    block_dims = (output_shape[block_axis] + block_factor - 1) / block_factor;

    // ub factor
    int64_t temp_ub_times;
    if (real_params_row_num == 0) {
      ub_axis = UB_TILING_LAST_AXIS;
      temp_ub_times = (output_shape[ub_axis] + params_num_ub - 1) / params_num_ub;
      ub_factor = (output_shape[ub_axis] + temp_ub_times - 1) / temp_ub_times;
    } else {
      if (real_params_row_num < output_shape[OUTPUT_INDICES_LOOP_IDX]) {
        ub_axis = UB_TILING_THIRD_AXIS;
        temp_ub_times = (output_shape[ub_axis] + real_params_row_num - 1) / real_params_row_num;
        ub_factor = (output_shape[ub_axis] + temp_ub_times - 1) / temp_ub_times;
      } else {
        // ub axis = 1
        int64_t real_axis2_params_row_num = real_params_row_num / output_shape[OUTPUT_INDICES_LOOP_IDX];
        ub_axis = UB_TILING_SECOND_AXIS;
        if (real_axis2_params_row_num < block_factor) {
          temp_ub_times = (block_factor + real_axis2_params_row_num - 1) / real_axis2_params_row_num;
          ub_factor = (block_factor + temp_ub_times - 1) / temp_ub_times;
        } else {
          ub_factor = block_factor;
        }
      }
    }
    return;
  }

  void GatherDsl::BlockThirdAxis() {
    block_axis = BLOCK_TILING_THIRD_AXIS;
    int64_t block_tmp = std::min((gather_compile_info.params_align + params_rows - 1) / params_rows,
                                 output_shape[block_axis]);
    block_factor =
      std::max((output_shape[block_axis] + gather_compile_info.core_num - 1) / gather_compile_info.core_num, block_tmp);
    if (block_factor == 0) {
      return;
    }
    block_dims = (output_shape[block_axis] + block_factor - 1) / block_factor;

    int64_t temp_ub_times;
    if (real_params_row_num == 0) {
      ub_axis = UB_TILING_LAST_AXIS;
      temp_ub_times = (output_shape[ub_axis] + params_num_ub - 1) / params_num_ub;
      ub_factor = (output_shape[ub_axis] + temp_ub_times - 1) / temp_ub_times;
    } else {
      ub_axis = UB_TILING_THIRD_AXIS;
      if (real_params_row_num < block_factor) {
        temp_ub_times = (block_factor + real_params_row_num - 1) / real_params_row_num;
        ub_factor = (block_factor + temp_ub_times - 1) / temp_ub_times;
      } else {
        ub_factor = block_factor;
      }
    }
    return;
  }

  void GatherDsl::BlockLastAxis() {
    block_axis = BLOCK_TILING_LAST_AXIS;
    int64_t block_tmp = std::min(gather_compile_info.params_align, output_shape[block_axis]);
    block_factor =
      std::max((output_shape[block_axis] + gather_compile_info.core_num - 1) / gather_compile_info.core_num, block_tmp);
    if (block_factor == 0) {
      return;
    }
    block_dims = (output_shape[block_axis] + block_factor - 1) / block_factor;

    ub_axis = UB_TILING_LAST_AXIS;
    if (block_factor > params_num_ub) {
      ub_factor = params_num_ub;
    } else {
      ub_factor = block_factor;
    }
    return;
  }

  void GatherDsl::EnsureBlockUBTiling() {
    if (block_dims != 1) {
      // check if ub factor less than 32B
      int64_t total_ub_size = ub_factor;
      total_ub_size = std::accumulate(output_shape.begin() + ub_axis + 1, output_shape.end(),
                                      total_ub_size, std::multiplies<int64_t>());
      int64_t total_ub_size_tail = ub_factor;
      if (block_axis == ub_axis) {
        if (block_factor == ub_factor) {
          int64_t pre_core_num = std::accumulate(output_shape.begin(), output_shape.begin() + block_axis,
                                                 1LL, std::multiplies<int64_t>());
          if ((pre_core_num == 1) && (total_ub_size > gather_compile_info.params_align)) {
            return;
          }
          total_ub_size_tail = output_shape[ub_axis] % ub_factor;
        } else {
          total_ub_size_tail = block_factor % ub_factor;
        }
      } else {
        total_ub_size_tail = output_shape[ub_axis] % ub_factor;
      }

      total_ub_size_tail = std::accumulate(output_shape.begin() + ub_axis + 1, output_shape.end(),
                                           total_ub_size_tail, std::multiplies<int64_t>());
      if ((total_ub_size < gather_compile_info.params_align) ||
      ((total_ub_size_tail < gather_compile_info.params_align) && (total_ub_size_tail > 0))) {
        SafeTiling();
      }
    }

    return;
  }

  void GatherDsl::SafeTiling() {
    key_special_pattern = 0;
    block_dims = 1;
    block_axis = BLOCK_TILING_FIRST_AXIS;
    block_factor = output_shape[block_axis];

    if (params_rows_align > gather_compile_info.params_ub_store_num) {
      ub_axis = OUTPUT_PARAMS_ROW_IDX;
      ub_factor = gather_compile_info.params_ub_store_num;
    } else {
      int64_t ub_size_align = output_size / params_rows * params_rows_align;
      for(size_t idx = block_axis; idx < OUTPUT_PARAMS_ROW_IDX; idx++) {
        if (ub_size_align < gather_compile_info.params_ub_store_num) {
          ub_axis = idx;
          ub_factor = output_shape[ub_axis];
          break;
        } else {
          ub_size_align = ub_size_align / output_shape[idx];
        }
      }
    }
  }

  bool GatherDsl::DoBaseTiling() {
    // n last gather and params last dim > 1
    if ((output_shape[OUTPUT_BATCH_DIM_IDX] >= (gather_compile_info.core_num / 2)) ||
    ((output_size / output_shape[OUTPUT_BATCH_DIM_IDX] * gather_compile_info.params_dtype) < BLOCK_SIZE) ||
    ((output_shape[OUTPUT_BATCH_DIM_IDX] >= output_shape[OUTPUT_PARAMS_PRE_LOOP_IDX]) &&
    (output_shape[OUTPUT_INDICES_LOOP_IDX] * params_rows <
    gather_compile_info.core_num * gather_compile_info.params_align))) {
      BlockFirstAxis();
    } else {
      if ((output_shape[OUTPUT_PARAMS_PRE_LOOP_IDX] >= (gather_compile_info.core_num / 2)) ||
      (output_shape[OUTPUT_INDICES_LOOP_IDX] * params_rows <
      gather_compile_info.core_num * gather_compile_info.params_align)) {
        BlockSecondAxis();
      } else {
        if (output_shape[OUTPUT_INDICES_LOOP_IDX] > 1) {
          BlockThirdAxis();
        } else {
          BlockLastAxis();
        }
      }
    }

    return true;
  }

  bool GatherDsl::CalcKey() {
    // gather nd delete reduction axis
    if (gather_compile_info.gather_type == GATHER_ND_COMPUTE) {
      if (block_axis >= 1) {
        block_axis = block_axis - 1;
      }
      if (ub_axis >= 1) {
        ub_axis = ub_axis - 1;
      }
    }

    // base key
    key = BASE_KEY;

    // gather info
    constexpr int64_t rank_coeff = 10000;
    key += rank * rank_coeff + key_special_pattern;

    // split info
    if (gather_compile_info.gather_type == GATHER_ND_COMPUTE) {
      key += block_axis * (OUTPUT_SHAPE_SIZE - 1) + ub_axis;
    } else {
      key += block_axis * OUTPUT_SHAPE_SIZE + ub_axis;
    }

    return true;
  }

  bool GatherDsl::WriteTilingData() {
    OP_LOGD(op_type.c_str(), "tiling key:%lld block_dims:%lld block_factor:%lld ub_factor:%lld "
            "block_axis:%lld ub_axis:%lld", key, block_dims, block_factor, ub_factor, block_axis, ub_axis);

    if (gather_compile_info.is_dynamic_const) {
      run_info.AddTilingData(static_cast<uint32_t>(key));
      run_info.AddTilingData(static_cast<int32_t>(block_axis));
      run_info.AddTilingData(static_cast<int32_t>(block_factor));
      run_info.AddTilingData(static_cast<int32_t>(ub_axis));
      run_info.AddTilingData(static_cast<int32_t>(ub_factor));
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
      const auto &all_vars = gather_compile_info.gather_vars.at(str_key);
      for (const auto &var : all_vars) {
        if (var >= MIN_UB_FACTOR_IDX) {
          run_info.AddTilingData(static_cast<int32_t>(ub_factor));
        } else if (var >= MIN_BLOCK_FACTOR_IDX) {
          run_info.AddTilingData(static_cast<int32_t>(block_factor));
        } else if (var >= INDICES_SHAPE_IDX) {
          int64_t var_value = var;
          size_t dim_index = var_value % DECIMAL_TEN;
          run_info.AddTilingData(static_cast<int32_t>(indices_shape[dim_index]));
        } else {
          int64_t var_value = var;
          size_t dim_index = var_value % DECIMAL_TEN;
          if (total_size != 0) {
            if ((gather_compile_info.gather_type == GATHER_ND_COMPUTE) && (dim_index > 0)) {
              dim_index = dim_index + 1;
            }
          }
          run_info.AddTilingData(static_cast<int32_t>(params_shape[dim_index]));
        }
      }
    } catch (const std::exception &e) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile info[_gather_vars] error, error message: %s", e.what());
      return false;
    }
    return true;
  }

  bool GatherDsl::DoTiling() {
    OP_LOGD(op_type.c_str(), "GatherDsl tiling running");
    bool init_ret = Init();
    if (!init_ret) {
      return false;
    }

    bool tiling_ret = false;

    if (IsZeroShapeTiling()) {
      // zero schedule
      tiling_ret = DoZeroShapeTiling();
    } else {
      if (IsSpecialPattern()) {
        // special schedule
        if (IsDePadTiling()) {
          tiling_ret = DoDePadTiling();
        } else if (IsScalarTiling()) {
          tiling_ret = DoScalarTiling();
        } else if (IsParamsUbTiling()) {
          tiling_ret = DoParamsUbTiling();
        } else if (IsDbModule()) {
          tiling_ret = DoDbModule();
        }
      }

      // base schedule
      if (!tiling_ret && IsBaseTiling()) {
        tiling_ret = DoBaseTiling();
      }
      EnsureBlockUBTiling();
      tiling_ret = tiling_ret && CalcKey();
    }

    tiling_ret = tiling_ret && WriteTilingData();
    return tiling_ret;
  }

  bool GatherTilingHandler::DoTiling(const ge::Operator &op_paras, utils::OpRunInfo &run_info) const {
    OP_LOGD(op_type.c_str(), "GatherTilingHandler DoTiling running");
    GatherDsl GatherDsl(op_type, op_paras, gather_compile_info, run_info);
    return GatherDsl.DoTiling();
  }

  bool GatherTilingHandler::DoTiling(const ge::Operator &op_paras, utils::OpRunInfo &run_info,
                                     const OpInfo &op_info) const {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Gather custom tiling is not supported yet");
    return false;
  }

  std::shared_ptr <AutoTilingHandler> CreateGatherTilingHandler(const std::string &op_type,
                                                                const std::string &pattern,
                                                                const nlohmann::json &parsed_compile_info) {
    return std::make_shared<GatherTilingHandler>(op_type, pattern, parsed_compile_info);
  }
}  // namespace optiling
