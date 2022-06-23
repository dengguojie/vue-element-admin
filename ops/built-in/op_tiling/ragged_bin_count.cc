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
 * \file ragged_bin_count.cc
 * \brief dynamic shape tiling of ragged_bin_count
 */
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include "op_tiling_util.h"
#include "graph/debug/ge_log.h"

#include "error_util.h"
#include "op_log.h"
#include "error_log.h"
#include "op_const.h"
#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"

namespace optiling {
using namespace ge;

// one block size is 32Bytes
static const int32_t BLOCK_SIZE = 32;
// total tiling parameters number
static const int32_t TILING_PARAMS_NUM = 32;
static const int32_t BYTE_32BIT = 4;
static const int32_t BYTE_64BIT = 8;
static const std::vector<std::string> COMPILE_INFO_KEY = {"ub_size", "core_num"};

struct RaggedBinCountParams {
  int32_t need_core_num;
  int32_t size_data;
  int32_t splits_num;
  int32_t weights_num;
  int32_t output_total_num;
  int32_t values_num_each_core;
  int32_t values_num_tail_core;
  int32_t each_ub_block_num;
  int32_t max_ub_calc_values_num;
};

static void InitTilingParams(RaggedBinCountParams& params) {
  params.need_core_num = 0;
  params.size_data = 0;
  params.splits_num = 0;
  params.weights_num = 0;
  params.output_total_num = 0;
  params.values_num_each_core = 0;
  params.values_num_tail_core = 0;
  params.each_ub_block_num = 0;
  params.max_ub_calc_values_num = 0;
}

static void SetRuningInfo(const RaggedBinCountParams& tiling_params, utils::OpRunInfo& run_info) {
  run_info.AddTilingData(tiling_params.need_core_num);
  run_info.AddTilingData(tiling_params.size_data);
  run_info.AddTilingData(tiling_params.splits_num);
  run_info.AddTilingData(tiling_params.weights_num);
  run_info.AddTilingData(tiling_params.output_total_num);
  run_info.AddTilingData(tiling_params.values_num_each_core);
  run_info.AddTilingData(tiling_params.values_num_tail_core);
  run_info.AddTilingData(tiling_params.each_ub_block_num);
  run_info.AddTilingData(tiling_params.max_ub_calc_values_num);
}

static void PrintTilingParams(const RaggedBinCountParams& tiling_params) {
  OP_LOGD("RaggedBinCount", "[RaggedBinCountTiling] : need_core_num=%ld.", tiling_params.need_core_num);
  OP_LOGD("RaggedBinCount", "[RaggedBinCountTiling] : size_data=%ld.", tiling_params.size_data);
  OP_LOGD("RaggedBinCount", "[RaggedBinCountTiling] : splits_num=%ld.", tiling_params.splits_num);
  OP_LOGD("RaggedBinCount", "[RaggedBinCountTiling] : weights_num=%ld.", tiling_params.weights_num);
  OP_LOGD("RaggedBinCount", "[RaggedBinCountTiling] : output_total_num=%ld.", tiling_params.output_total_num);
  OP_LOGD("RaggedBinCount", "[RaggedBinCountTiling] : values_num_each_core=%ld.", tiling_params.values_num_each_core);
  OP_LOGD("RaggedBinCount", "[RaggedBinCountTiling] : values_num_tail_core=%ld.", tiling_params.values_num_tail_core);
  OP_LOGD("RaggedBinCount", "[RaggedBinCountTiling] : each_ub_block_num=%ld.", tiling_params.each_ub_block_num);
  OP_LOGD("RaggedBinCount", "[RaggedBinCountTiling] : max_ub_calc_values_num=%ld.",
          tiling_params.max_ub_calc_values_num);
}

/******************COMPUTE_FUNCTION******************/

static bool GetCompileParams(const std::string& op_type, const std::vector<int64_t>& op_compile_info, int32_t& ub_size,
                             int32_t& core_num) {
  OP_TILING_CHECK(COMPILE_INFO_KEY.size() != op_compile_info.size(),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "parse op_compile_info failed."), return false);
  ub_size = op_compile_info[0];
  core_num = op_compile_info[1];
  OP_TILING_CHECK(core_num == 0, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "core_num cannot be zero."), return false);

  OP_LOGD(op_type, "the compile info is: ub_size=[%ld], core_num=[%ld].", ub_size, core_num);
  return true;
}

static bool CalcNeedCoreNum(const int32_t values_total_num, const int32_t core_num, RaggedBinCountParams& tiling_pms) {
  if (core_num == 0) {
    return false;
  }
  tiling_pms.values_num_each_core = values_total_num / core_num;

  if (tiling_pms.values_num_each_core <= 1) {
    tiling_pms.need_core_num = values_total_num;
    tiling_pms.values_num_each_core = 1;
    tiling_pms.values_num_tail_core = 1;
  } else {
    tiling_pms.need_core_num = core_num;
    tiling_pms.values_num_tail_core = values_total_num - (tiling_pms.values_num_each_core * (core_num - 1));
  }
  return true;
}

/******************TILING_FUNCTION******************/

bool RaggedBinCountTiling(const std::string& op_type, const ge::Operator& op_params,
                          const std::vector<int64_t>& op_compile_info, utils::OpRunInfo& run_info) {
  OP_LOGI(op_type.c_str(), "RaggedBinCountTiling is running.");

  RaggedBinCountParams tiling_params;
  InitTilingParams(tiling_params);

  auto operator_info = OpDescUtils::GetOpDescFromOperator(op_params);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get op_info failed."),
                  return false);

  // the parameters order in op proto is: splits, values, size, weights, binary_output
  size_t input_splits_idx = 0;  // splits
  size_t input_values_idx = 1;  // values
  size_t input_size_idx = 2;  // size
  size_t input_weights_idx = 3;  // weights

  auto splits_desc = operator_info->MutableInputDesc(input_splits_idx);
  OP_TILING_CHECK(splits_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get splits failed."), return false);
  ge::GeShape splits_shape = splits_desc->MutableShape();
  OP_TILING_CHECK(
      splits_shape.GetDimNum() != 1,
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "dim of splits must be 1, but get %lu.", splits_shape.GetDimNum()),
      return false);
  tiling_params.splits_num = GetTensorSize(splits_shape);
  OP_TILING_CHECK(tiling_params.splits_num < 2,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "number of splits must more than 2, but get %d.",
                                                  tiling_params.splits_num),
                  return false);

  auto values_desc = operator_info->MutableInputDesc(input_values_idx);
  OP_TILING_CHECK(values_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get values failed."), return false);
  ge::GeShape values_shape = values_desc->MutableShape();
  ge::DataType values_dtype = values_desc->GetDataType();
  int32_t values_total_num = GetTensorSize(values_shape);
  int32_t values_each_size = GetSizeByDataType(values_dtype);

  // size is a scalar Tensor
  std::vector<int64_t> tmp_size_data;
  OP_TILING_CHECK(!ops::GetConstIntData(op_params, input_size_idx, tmp_size_data),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get size failed."), return false);
  OP_TILING_CHECK(
      tmp_size_data.size() != 1,
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "size must be 1 tensor scalar, but get %lu.", tmp_size_data.size()),
      return false);
  tiling_params.size_data = static_cast<int32_t>(tmp_size_data[0]);
  OP_TILING_CHECK(
      tiling_params.size_data < 0,
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "size must be a nagetive scalar, but get %d.", tiling_params.size_data),
      return false);

  auto weights_desc = operator_info->MutableInputDesc(input_weights_idx);
  OP_TILING_CHECK(weights_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get weights failed."),
                  return false);
  ge::GeShape weights_shape = weights_desc->MutableShape();
  tiling_params.weights_num = GetTensorSize(weights_shape);

  OP_TILING_CHECK(tiling_params.weights_num != 0 && tiling_params.weights_num != values_total_num,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "size of weights must be equal to values has."),
                  return false);

  int32_t ub_size = 0;
  int32_t core_num = 0;

  OP_TILING_CHECK(!GetCompileParams(op_type, op_compile_info, ub_size, core_num),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile information failed."), return false);

  int32_t ub_size_left = ub_size - TILING_PARAMS_NUM * BYTE_32BIT;
  int32_t max_ub_size = ub_size_left / 4;  // for splits, values, weights, output
  tiling_params.each_ub_block_num = max_ub_size / BLOCK_SIZE;

  OP_TILING_CHECK(!CalcNeedCoreNum(values_total_num, core_num, tiling_params),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get core need number failed."), return false);

  tiling_params.output_total_num = (tiling_params.splits_num - 1) * tiling_params.size_data;
  OP_TILING_CHECK(values_each_size == 0, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "values data size is invalid."),
                  return false);
  tiling_params.max_ub_calc_values_num = max_ub_size / values_each_size;

  SetRuningInfo(tiling_params, run_info);
  PrintTilingParams(tiling_params);

  run_info.SetBlockDim(tiling_params.need_core_num);
  OP_LOGI(op_type.c_str(), "RaggedBinCountTiling run success.");
  return true;
}

// register tiling interface of the RaggedBinCount op.
REGISTER_OP_TILING_V3_WITH_VECTOR(RaggedBinCount, RaggedBinCountTiling, COMPILE_INFO_KEY, NO_OPTIONAL_VALUE);
}  // namespace optiling