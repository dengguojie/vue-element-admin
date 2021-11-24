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
 * \file gather_schedule.cpp
 * \brief
 */
#include "gather_dsl.h"
#include "op_tiling_util.h"

#include <algorithm>
#include <unordered_map>

#include "graph/utils/op_desc_utils.h"
#include "vector_tiling.h"
#include "error_log.h"
#include <math.h>

namespace optiling {
  GatherDslCompileInfo::GatherDslCompileInfo(const std::string& op_type, const nlohmann::json& org_compile_info) {
    try {
      // parse base info
      const auto &base_info = org_compile_info.at("_base_info");
      const size_t base_info_size = 6;
      V_CHECK_EQ(base_info.size(), base_info_size,
                 VECTOR_INNER_ERR_REPORT_TILIING(op_type, "base info must be 6 element"),
                 return;);
      const size_t core_number_idx = 0;
      const size_t ub_size_idx = 1;
      const size_t l1_size_idx = 2;
      const size_t gather_type_idx = 3;
      const size_t params_dtype_idx = 4;
      const size_t indices_dtype_idx = 5;
      const int64_t block_size = 32;
      core_num = base_info[core_number_idx];
      ub_size = base_info[ub_size_idx];
      l1_size = base_info[l1_size_idx];
      gather_type = base_info[gather_type_idx];
      params_dtype = base_info[params_dtype_idx];
      params_align = block_size / params_dtype;
      indices_dtype = base_info[indices_dtype_idx];

      // parse custom info
      const auto &custom_info = org_compile_info.at("_custom_info");
      const size_t custom_info_size = 5;
      V_CHECK_EQ(custom_info.size(), custom_info_size,
                 VECTOR_INNER_ERR_REPORT_TILIING(op_type, "custom info must be 5 element"),
                 return;);
      const size_t params_l1_num_idx = 0;
      const size_t params_ub_half_num_idx = 1;
      const size_t batch_dims_idx = 2;
      const size_t is_binary_shape_idx = 3;
      const size_t org_batch_dims_idx = 4;
      params_l1_num = custom_info[params_l1_num_idx];
      params_ub_half_num = custom_info[params_ub_half_num_idx];
      batch_dims = custom_info[batch_dims_idx];
      is_binary_shape = custom_info[is_binary_shape_idx];
      org_batch_dims = custom_info[org_batch_dims_idx];

      OP_LOGD(op_type.c_str(), "GatherDslCompileInfo:%lld %lld %lld %lld",
              gather_type, batch_dims, is_binary_shape, org_batch_dims);

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
    } catch (const std::exception &e) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "construct compile_info error. Error message: %s", e.what());
    }
    return;
  }

  bool GatherDsl::Init() {

    std::vector <int64_t> org_params_shape = op_paras.GetInputDesc(0).GetShape().GetDims();
    if (org_params_shape.empty()) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "gather org_params_shape values is empty.");
      return false;
    }

    std::vector <int64_t> org_indices_shape = op_paras.GetInputDesc(1).GetShape().GetDims();
    if (org_indices_shape.empty()) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "gather org_indices_shape values is empty.");
      return false;
    }

    // gather rank
    if (gather_compile_info.gather_type == GATHER_ND_COMPUTE) {
      rank = org_indices_shape[org_indices_shape.size() - 1];
    }

    // batch dims
    if ((gather_compile_info.is_binary_shape) && (gather_compile_info.org_batch_dims < 0)) {
      real_batch_dims = gather_compile_info.org_batch_dims + org_indices_shape.size();
    } else {
      real_batch_dims = gather_compile_info.org_batch_dims;
    }

    const size_t params_row_idx = 3;
    if (gather_compile_info.is_dynamic_const) {
      // const condition shape if fused
      axis = gather_compile_info.const_axis;

      // const condition
      if (gather_compile_info.gather_type == GATHER_ND_COMPUTE) {
        // gather nd [batch, axes, row] [batch, loops, rank]
        params_shape.push_back(org_params_shape[PARAMS_BATCH_DIM_IDX]);
        params_shape.push_back(1);
        for (size_t i = 1; i < org_params_shape.size(); i++) {
          params_shape.push_back(org_params_shape[i]);
        }

        indices_shape.push_back(org_indices_shape[INDICES_BATCH_DIM_IDX]);
        indices_shape.push_back(org_indices_shape[INDICES_LOOP_IDX]);
        indices_shape.push_back(org_indices_shape[INDICES_RANK_IDX]);
      } else {
        // gather/gather v2 [batch, pre_loops, axis, row] [batch, loops]
        params_shape.push_back(org_params_shape[PARAMS_BATCH_DIM_IDX]);
        params_shape.push_back(org_params_shape[PARAMS_LOOP_IDX]);
        params_shape.push_back(org_params_shape[PARAMS_AXIS_IDX]);
        params_shape.push_back(org_params_shape[params_row_idx]);

        indices_shape.push_back(org_indices_shape[INDICES_BATCH_DIM_IDX]);
        indices_shape.push_back(org_indices_shape[INDICES_LOOP_IDX]);
        indices_shape.push_back(1);
      }

      params_rows = params_shape[params_shape.size() - 1];
    } else {
      // dynamic
      // gather_type = 0 gather
      // gather_type = 1 gather nd
      // gather_type = 2 gather v2
      SimplyParamsAndIndices(org_params_shape, org_indices_shape);

    }

    // output shape
    output_shape.push_back(params_shape[PARAMS_BATCH_DIM_IDX]);
    output_shape.push_back(params_shape[PARAMS_LOOP_IDX]);
    output_shape.push_back(indices_shape[INDICES_LOOP_IDX]);
    output_shape.push_back(params_rows);

    // cal total size
    params_size_total = params_shape[PARAMS_BATCH_DIM_IDX] * params_shape[PARAMS_LOOP_IDX]
                        * params_shape[PARAMS_AXIS_IDX] * params_shape[PARAMS_ROWS_IDX];
    indices_size_total = indices_shape[INDICES_BATCH_DIM_IDX] * indices_shape[INDICES_LOOP_IDX]
                         * indices_shape[INDICES_RANK_IDX];
    total_size = params_size_total * indices_shape[INDICES_LOOP_IDX];

    OP_LOGD(op_type.c_str(), "GatherDsl output_shape:%lld %lld %lld %lld",
            output_shape[OUTPUT_BATCH_DIM_IDX], output_shape[OUTPUT_PARAMS_PRE_LOOP_IDX],
            output_shape[OUTPUT_INDICES_LOOP_IDX], output_shape[OUTPUT_PARAMS_ROW_IDX]);
    return true;
  }

  void GatherDsl::SimplyParamsAndIndices(std::vector <int64_t> org_params_shape,
                                              std::vector <int64_t> org_indices_shape) {

    if (org_indices_shape.empty()) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "gather org_indices_shape values is empty.");
      return;
    }

    uint32_t inputs_num = op_paras.GetInputsSize();
    if (inputs_num == GATHER_V2_INPUTS_NUM) {
      std::vector <int64_t> values;
      // input axis index is 2
      if (!ops::GetConstIntData(op_paras, 2, values)) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "gather v2 axis not exists.");
        return;
      }

      if (values.empty()) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "gather v2 axis values is empty.");
        return;
      }

      axis = values[0];
      if (axis < 0) {
        axis = axis + org_params_shape.size();
      }
    } else {
      axis = real_batch_dims;
    }
    // params batch dims
    int64_t params_batch_dims = 1;
    for (int i = 0; i < real_batch_dims; i++) {
      params_batch_dims = params_batch_dims * org_params_shape[i];
    }
    params_shape.push_back(params_batch_dims);

    // params pre loop
    int64_t params_pre_loop = 1;
    for (int k = real_batch_dims; k < axis; k++) {
      params_pre_loop = params_pre_loop * org_params_shape[k];
    }
    params_shape.push_back(params_pre_loop);

    // params gather axis
    for (int k = axis; k < axis + rank; k++) {
      params_shape.push_back(org_params_shape[k]);
    }

    // params rows
    params_rows = 1;
    for (size_t k = axis + rank; k < org_params_shape.size(); k++) {
      params_rows = params_rows * org_params_shape[k];
    }
    params_shape.push_back(params_rows);

    // indices batch dims
    indices_shape.push_back(params_batch_dims);

    int64_t indices_loops = 1;
    for (size_t j = real_batch_dims; j < org_indices_shape.size() - 1; j++) {
      indices_loops = indices_loops * org_indices_shape[j];
    }

    if (gather_compile_info.gather_type != GATHER_ND_COMPUTE) {
      indices_loops = indices_loops * org_indices_shape[org_indices_shape.size() - 1];
    }

    indices_shape.push_back(indices_loops);
    indices_shape.push_back(rank);

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
      key = 990000001;
      // real params shape [parmas_batch, params_data], but init is [batch, gather axis, after axis]
      params_shape[PARAMS_LOOP_IDX] = params_shape[PARAMS_LOOP_IDX] * params_shape[PARAMS_AXIS_IDX];
    } else {
      // zero shape
      key = 990000000;
    }
    return true;
  }

  bool GatherDsl::IsSpecialPattern() {
    return (output_shape[OUTPUT_BATCH_DIM_IDX] == 1) && (output_shape[OUTPUT_PARAMS_PRE_LOOP_IDX] == 1)
           && (output_shape[OUTPUT_INDICES_LOOP_IDX] > 1) && (rank == 1);
  }

  const std::string DEPAD_SCHEDULE = "7";
  bool GatherDsl::IsDePadTiling() {
    if (gather_compile_info.tensor_sizes.count(DEPAD_SCHEDULE) == 0) {
      return false;
    }

    if (gather_compile_info.tensor_sizes.at(DEPAD_SCHEDULE).size() != TENSOR_SIZES_NUM) {
      return false;
    }

    params_num_ub = gather_compile_info.tensor_sizes.at(DEPAD_SCHEDULE)[TENSOR_SIZES_PARAMS_IDX] / rank;

    real_params_row_num = params_num_ub / ((params_rows + gather_compile_info.params_align - 1)
                                           / gather_compile_info.params_align * gather_compile_info.params_align);
    if (real_params_row_num < 1) {
      return false;
    }

    int64_t params_size_total_align =
        params_size_total / params_rows * (params_rows + gather_compile_info.params_align - 1) /
        gather_compile_info.params_align * gather_compile_info.params_align * gather_compile_info.params_dtype;

    const int64_t depad_params_threshold = 16384;
    const int64_t depad_indices_threshold = 3072;
    bool is_shape_ok = (params_size_total_align > depad_params_threshold)
                       && (indices_size_total > depad_indices_threshold)
                       && (((params_rows * gather_compile_info.params_dtype) % BLOCK_SIZE) > 0);
    if (is_shape_ok) {
      // b32
      const int64_t b32_last_dim_max = 168;
      const int64_t b32_co2co = 128;
      int64_t last_dim_max = b32_last_dim_max;
      int64_t co2co = b32_co2co;

      if (gather_compile_info.params_dtype == 1) {
        // b8
        const int64_t b8_last_dim_max = 64;
        const int64_t b8_co2co = 1024;
        last_dim_max = b8_last_dim_max;
        co2co = b8_co2co;
      } else if (gather_compile_info.params_dtype == 2) {
        // b16
        const int64_t b16_last_dim_max = 160;
        const int64_t b16_co2co = 256;
        last_dim_max = b16_last_dim_max;
        co2co = b16_co2co;
      } else if (gather_compile_info.params_dtype == 8) {
        // b64
        const int64_t b64_last_dim_max = 168;
        const int64_t b64_co2co = 64;
        last_dim_max = b64_last_dim_max;
        co2co = b64_co2co;
      }
      bool is_params_rows_ok = params_rows <= last_dim_max;
      bool is_params_num_ok = (co2co * (params_rows + gather_compile_info.params_align - 1) /
                               gather_compile_info.params_align * gather_compile_info.params_align) <= params_num_ub;

      return is_params_rows_ok && is_params_num_ok;
    }
    return false;
  }

  bool GatherDsl::DoDePadTiling() {

    // check if remove pad can use
    int64_t max_ub_factor = params_num_ub /
                            ((params_rows + gather_compile_info.params_align - 1)
                             / gather_compile_info.params_align * gather_compile_info.params_align);

    BlockThirdAxis();

    if (ub_factor > max_ub_factor) {
      ub_factor = max_ub_factor;
    }
    key_special_pattern = 7000;

    return true;
  }

  bool GatherDsl::IsScalarTiling() {
    const std::string SCALAR_SCHEDULE = "6";
    if (gather_compile_info.tensor_sizes.count(SCALAR_SCHEDULE) == 0) {
      return false;
    }

    if (gather_compile_info.tensor_sizes.at(SCALAR_SCHEDULE).size() != TENSOR_SIZES_NUM) {
      return false;
    }

    params_num_ub = gather_compile_info.tensor_sizes.at(SCALAR_SCHEDULE)[TENSOR_SIZES_PARAMS_IDX] /rank;
    indices_num_ub = gather_compile_info.tensor_sizes.at(SCALAR_SCHEDULE)[TENSOR_SIZES_INDICES_IDX] / rank;

    const int64_t scalar_indices_threshold = 3072;
    return (params_size_total < gather_compile_info.params_ub_half_num / 2) && (indices_size_total < scalar_indices_threshold)
            && (output_shape[OUTPUT_PARAMS_ROW_IDX] == 1);
  }

  bool GatherDsl::DoScalarTiling() {
    block_axis = OUTPUT_INDICES_LOOP_IDX;
    ub_axis = OUTPUT_INDICES_LOOP_IDX;
    if (output_shape[OUTPUT_INDICES_LOOP_IDX] >= gather_compile_info.core_num) {
      block_factor = (output_shape[OUTPUT_INDICES_LOOP_IDX] + gather_compile_info.core_num - 1) / gather_compile_info.core_num;
      block_factor = (block_factor + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
      block_dims = (output_shape[OUTPUT_INDICES_LOOP_IDX] + block_factor - 1) / block_factor;

      if ((block_factor * gather_compile_info.params_dtype) >= BLOCK_SIZE) {
        int64_t min_ub_num = std::min(params_num_ub, indices_num_ub);
        min_ub_num = (min_ub_num + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

        if (block_factor > min_ub_num) {
          ub_factor = min_ub_num;
        } else {
          ub_factor = block_factor;
        }
      } else {
        block_factor = BLOCK_SIZE / gather_compile_info.params_dtype;
        block_dims = (output_shape[OUTPUT_INDICES_LOOP_IDX] + block_factor - 1) / block_factor;
        ub_factor = block_factor;
      }
    } else {
      // one core
      block_dims = 1;
      block_factor = output_shape[block_axis];
      ub_factor = output_shape[ub_axis];
    }

    key_special_pattern = 6000;

    return true;
  }

  bool GatherDsl::IsStoreUB(int64_t params_total) {
    int64_t indices_total = indices_shape[INDICES_BATCH_DIM_IDX] * indices_shape[INDICES_LOOP_IDX];
    const int64_t store_ub_indices_threshold = 3200;
    return ((params_total < (gather_compile_info.params_ub_half_num / 2)) && (indices_total > store_ub_indices_threshold));
  }

  bool GatherDsl::IsStoreL1(int64_t params_total) {
    int64_t indices_total = indices_shape[INDICES_BATCH_DIM_IDX] * indices_shape[INDICES_LOOP_IDX];
    const int64_t store_l1_indices_threshold = 16384;
    return ((params_total < gather_compile_info.params_l1_num)
            && (params_total >= (gather_compile_info.params_ub_half_num / 2))
            && (indices_total > store_l1_indices_threshold));
  }

  bool GatherDsl::IsParamsUbTiling() {
    const std::string PARAMS_UB_ALIGN_SCHEDULE = "1";
    const std::string PARAMS_UB_NOT_ALIGN_SCHEDULE = "2";
    if ((gather_compile_info.tensor_sizes.count(PARAMS_UB_ALIGN_SCHEDULE) == 0
    || gather_compile_info.tensor_sizes.at(PARAMS_UB_ALIGN_SCHEDULE).size() != TENSOR_SIZES_NUM)
    && (gather_compile_info.tensor_sizes.count(PARAMS_UB_NOT_ALIGN_SCHEDULE) == 0
    || gather_compile_info.tensor_sizes.at(PARAMS_UB_NOT_ALIGN_SCHEDULE).size() != TENSOR_SIZES_NUM)) {
      return false;
    }

    if (gather_compile_info.tensor_sizes.count(PARAMS_UB_ALIGN_SCHEDULE) > 0 ){
      params_num_ub = gather_compile_info.tensor_sizes.at(PARAMS_UB_ALIGN_SCHEDULE)[TENSOR_SIZES_PARAMS_IDX] / rank;
    } else {
      params_num_ub = gather_compile_info.tensor_sizes.at(PARAMS_UB_NOT_ALIGN_SCHEDULE)[TENSOR_SIZES_PARAMS_IDX] / rank;
    }

    real_params_row_num = params_num_ub / ((params_rows + gather_compile_info.params_align - 1)
                                           / gather_compile_info.params_align * gather_compile_info.params_align);
    if (real_params_row_num < 1) {
      return false;
    }

    int64_t params_size_total_align = params_size_total / params_rows *
                                      (params_rows + gather_compile_info.params_align - 1) / gather_compile_info.params_align * gather_compile_info.params_align;
    return IsStoreUB(params_size_total_align);
  }

  bool GatherDsl::DoParamsUbTiling() {
    // check need align
    BlockThirdAxis();

    if (((params_rows * gather_compile_info.params_dtype) % BLOCK_SIZE) == 0) {
      key_special_pattern = 1000;
    } else {
      key_special_pattern = 2000;
    }

    return true;
  }

  bool GatherDsl::IsParamsL1Tiling() {

    if (((params_rows * gather_compile_info.params_dtype) % BLOCK_SIZE) == 0) {
      return false;
    }

    const std::string PARAMS_L1_SCHEDULE = "3";
    if (gather_compile_info.tensor_sizes.count(PARAMS_L1_SCHEDULE) == 0) {
      return false;
    }

    if (gather_compile_info.tensor_sizes.at(PARAMS_L1_SCHEDULE).size() != TENSOR_SIZES_NUM) {
      return false;
    }

    params_num_ub = gather_compile_info.tensor_sizes.at(PARAMS_L1_SCHEDULE)[TENSOR_SIZES_PARAMS_IDX] / rank;

    real_params_row_num = params_num_ub / ((params_rows + gather_compile_info.params_align - 1)
                                           / gather_compile_info.params_align * gather_compile_info.params_align);

    int64_t params_size_total_align = params_size_total / params_rows
                                      * (params_rows + gather_compile_info.params_align - 1)
                                      / gather_compile_info.params_align * gather_compile_info.params_align;

    return IsStoreL1(params_size_total_align);
  }

  bool GatherDsl::DoParamsL1Tiling() {
    BlockThirdAxis();
    key_special_pattern = 3000;

    return true;
  }

  bool GatherDsl::IsDbModule() {

    const std::string DB_SCHEDULE = "5";
    if (gather_compile_info.tensor_sizes.count(DB_SCHEDULE) == 0) {
      return false;
    }

    if (gather_compile_info.tensor_sizes.at(DB_SCHEDULE).size() != TENSOR_SIZES_NUM) {
      return false;
    }

    params_num_ub = gather_compile_info.tensor_sizes.at(DB_SCHEDULE)[TENSOR_SIZES_PARAMS_IDX] / rank;

    real_params_row_num = params_num_ub / ((params_rows + gather_compile_info.params_align - 1)
                                           / gather_compile_info.params_align * gather_compile_info.params_align);

    return (real_params_row_num < 1) && (params_rows % gather_compile_info.params_align == 0);

  }

  bool GatherDsl::DoDbModule() {
    BlockThirdAxis();
    ub_factor = ub_factor / gather_compile_info.params_align * gather_compile_info.params_align;
    key_special_pattern = 5000;
    return true;
  }

  bool GatherDsl::IsBaseTiling() {
    const std::string BASE_SCHEDULE = "0";
    if (gather_compile_info.tensor_sizes.count(BASE_SCHEDULE) == 0) {
      return false;
    }

    if (gather_compile_info.tensor_sizes.at(BASE_SCHEDULE).size() != TENSOR_SIZES_NUM) {
      return false;
    }

    params_num_ub = gather_compile_info.tensor_sizes.at(BASE_SCHEDULE)[TENSOR_SIZES_PARAMS_IDX] / rank;
    indices_num_ub = gather_compile_info.tensor_sizes.at(BASE_SCHEDULE)[TENSOR_SIZES_INDICES_IDX] / rank;

    real_params_row_num = params_num_ub / ((params_rows + gather_compile_info.params_align - 1)
                                           / gather_compile_info.params_align * gather_compile_info.params_align);

    return true;

  }

  void GatherDsl::BlockFirstAxis() {
    int64_t real_params_row_num = params_num_ub / ((params_rows + gather_compile_info.params_align - 1)
                                                   / gather_compile_info.params_align * gather_compile_info.params_align);

    block_axis = 0;
    block_factor = (output_shape[block_axis] + gather_compile_info.core_num - 1) / gather_compile_info.core_num;
    block_dims = (output_shape[block_axis] + block_factor - 1) / block_factor;

    // ub factor
    int64_t temp_ub_times;
    if (real_params_row_num == 0) {
      ub_axis = 3;
      temp_ub_times = (output_shape[ub_axis] + params_num_ub - 1) / params_num_ub;
      ub_factor = (output_shape[ub_axis] + temp_ub_times - 1) / temp_ub_times;
    } else {
      if (real_params_row_num < output_shape[OUTPUT_INDICES_LOOP_IDX]) {
        ub_axis = 2;
        temp_ub_times = (output_shape[ub_axis] + real_params_row_num - 1) / real_params_row_num;
        ub_factor = (output_shape[ub_axis] + temp_ub_times - 1) / temp_ub_times;

      } else {
        // ub axis = 1
        int64_t real_axis2_params_row_num = real_params_row_num / output_shape[2];

        if (real_axis2_params_row_num < output_shape[OUTPUT_PARAMS_PRE_LOOP_IDX]) {
          ub_axis = 1;
          temp_ub_times = (output_shape[ub_axis] + real_axis2_params_row_num - 1) / real_axis2_params_row_num;
          ub_factor = (output_shape[ub_axis] + temp_ub_times - 1) / temp_ub_times;
        } else {
          // ub axis = 0
          int64_t real_axis1_params_row_num = real_axis2_params_row_num / output_shape[1];
          ub_axis = 0;
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
    int64_t real_params_row_num = params_num_ub /
                                  ((params_rows + gather_compile_info.params_align - 1) /
                                   gather_compile_info.params_align * gather_compile_info.params_align);

    block_axis = 1;
    block_factor = (output_shape[block_axis] + gather_compile_info.core_num - 1) / gather_compile_info.core_num;
    block_dims = (output_shape[block_axis] + block_factor - 1) / block_factor;

    // ub factor
    int64_t temp_ub_times;
    if (real_params_row_num == 0) {
      ub_axis = 3;
      temp_ub_times = (output_shape[ub_axis] + params_num_ub - 1) / params_num_ub;
      ub_factor = (output_shape[ub_axis] + temp_ub_times - 1) / temp_ub_times;
    } else {
      if (real_params_row_num < output_shape[OUTPUT_INDICES_LOOP_IDX]) {
        ub_axis = 2;
        temp_ub_times = (output_shape[ub_axis] + real_params_row_num - 1) / real_params_row_num;
        ub_factor = (output_shape[ub_axis] + temp_ub_times - 1) / temp_ub_times;
      } else {
        // ub axis = 1
        int64_t real_axis2_params_row_num = real_params_row_num / output_shape[2];
        ub_axis = 1;
        if (real_axis2_params_row_num < output_shape[OUTPUT_PARAMS_PRE_LOOP_IDX]) {
          temp_ub_times = (output_shape[ub_axis] + real_axis2_params_row_num - 1) / real_axis2_params_row_num;
          ub_factor = (output_shape[ub_axis] + temp_ub_times - 1) / temp_ub_times;
        } else {
          ub_factor = block_factor;
        }
      }
    }
    return;
  }

  void GatherDsl::BlockThirdAxis() {
    block_axis = 2;
    block_factor = (output_shape[block_axis] + gather_compile_info.core_num - 1) / gather_compile_info.core_num;
    block_dims = (output_shape[block_axis] + block_factor - 1) / block_factor;

    int64_t temp_ub_times;
    if (real_params_row_num == 0) {
      ub_axis = 3;
      temp_ub_times = (output_shape[ub_axis] + params_num_ub - 1) / params_num_ub;
      ub_factor = (output_shape[ub_axis] + temp_ub_times - 1) / temp_ub_times;
    } else {
      ub_axis = 2;
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
    block_axis = 3;
    block_factor = (output_shape[block_axis] + gather_compile_info.core_num - 1) / gather_compile_info.core_num;
    block_dims = (output_shape[block_axis] + block_factor - 1) / block_factor;

    ub_axis = 3;
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
      for (size_t i = ub_axis + 1; i < output_shape.size(); i++) {
        total_ub_size = total_ub_size * output_shape[i];
      }

      if (total_ub_size < gather_compile_info.params_align) {
        key_special_pattern = 0;
        if ((output_shape[OUTPUT_BATCH_DIM_IDX] == 1) && (output_shape[OUTPUT_PARAMS_PRE_LOOP_IDX] == 1)
            && (output_shape[OUTPUT_PARAMS_ROW_IDX] == 1)
            && (output_shape[OUTPUT_INDICES_LOOP_IDX] > gather_compile_info.params_align)
            && (output_shape[OUTPUT_INDICES_LOOP_IDX] < gather_compile_info.params_align * BLOCK_SIZE)) {
          block_axis = 2;
          ub_axis = 2;
          ub_factor = gather_compile_info.params_align;
          block_factor = gather_compile_info.params_align;
          block_dims = (output_shape[OUTPUT_INDICES_LOOP_IDX] + gather_compile_info.params_align - 1)
                       / gather_compile_info.params_align;
        } else {
          block_axis = 0;
          ub_axis = 0;
          block_dims = 1;
          block_factor = output_shape[OUTPUT_BATCH_DIM_IDX];
          ub_factor = block_factor;
        }
      }
    }

    // gather nd delete reduction axis
    if (gather_compile_info.gather_type == GATHER_ND_COMPUTE) {
      if (block_axis >= 1) {
        block_axis = block_axis - 1;
      }

      if (ub_axis >= 1) {
        ub_axis = ub_axis - 1;
      }
    }

    return;
  }

  bool GatherDsl::DoBaseTiling() {

    // n last gather and params last dim > 1
    if (output_shape[OUTPUT_BATCH_DIM_IDX] > 1) {
      BlockFirstAxis();
    } else {
      if (output_shape[OUTPUT_PARAMS_PRE_LOOP_IDX] > 1) {
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
    // base key
    key = 900000000;

    // gather info
    int64_t key_axis = axis;
    if (gather_compile_info.gather_type != GATHER_ND_COMPUTE) {
      key_axis = 2;
    }

    int64_t key_batch_dims = real_batch_dims;
    if (gather_compile_info.is_binary_shape) {
      key_batch_dims = 1;
    }

    key += key_batch_dims * 1000000 + key_axis * 100000 + rank * 10000 + key_special_pattern;

    // split info
    if (gather_compile_info.gather_type == GATHER_ND_COMPUTE) {
      key += block_axis * (output_shape.size() - 1) + ub_axis;
    } else {
      key += block_axis * output_shape.size() + ub_axis;
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
    char keys[10] = {'0', '0', '0', '0', '0', '0', '0', '0', '0', '\0'};
    while (cur_key && key_len >= 0) {
      keys[key_len] = '0' + cur_key % 10;
      cur_key /= 10;
      key_len--;
    }
    std::string str_key = keys;

    try {
      const auto &all_vars = gather_compile_info.gather_vars.at(str_key);
      for (const auto &var : all_vars) {
        if (var >= 40000) {
          run_info.AddTilingData(static_cast<int32_t>(ub_factor));
        } else if (var >= 30000) {
          run_info.AddTilingData(static_cast<int32_t>(block_factor));
        } else if (var >= 20000) {
          int64_t var_value = var;
          size_t dim_index = var_value % 10;
          run_info.AddTilingData(static_cast<int32_t>(indices_shape[dim_index]));
        } else {
          int64_t var_value = var;
          size_t dim_index = var_value % 10;
          if (total_size != 0) {
            if ((gather_compile_info.gather_type == GATHER_ND_COMPUTE) && (dim_index > 0)) {
              dim_index = dim_index + 1;
            }
          }
          run_info.AddTilingData(static_cast<int32_t>(params_shape[dim_index]));
        }
      }
    } catch (const std::exception &e) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile info[_gather_vars] error, error message: %s",
                                      e.what());
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
        if (IsDePadTiling()){
          tiling_ret = DoDePadTiling();
        } else if (IsScalarTiling()) {
          tiling_ret = DoScalarTiling();
        } else if (IsParamsUbTiling()) {
          tiling_ret = DoParamsUbTiling();
        } else if (IsParamsL1Tiling()) {
          tiling_ret = DoParamsL1Tiling();
        } else if (IsDbModule()) {
          tiling_ret = DoDbModule();
        }
      }

      // base schedule
      if (!tiling_ret && IsBaseTiling() ) {
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
}