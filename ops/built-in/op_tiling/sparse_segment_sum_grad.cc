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
 * \file sparse_segment_sum_grad.cc
 * \brief dynamic shape tiling of sparse_segment_sum_grad
 */
#include <nlohmann/json.hpp>
#include "op_tiling_util.h"
#include "graph/debug/ge_log.h"

#include "error_util.h"
#include "op_log.h"
#include "error_log.h"
#include "op_const.h"
#include "vector_tiling_profiling.h"

namespace optiling {
// dtype
static const std::string DTYPE_FP32 = "float32";
static const std::string DTYPE_INT32 = "int32";

static const int32_t GRAD_IDX = 0;
static const int32_t INDICES_IDX = 1;
static const int32_t SEGMENT_IDX = 2;
// fp32 select key
static const int32_t SELECT_KEY_MODE_FP32_INPUT_INVALID = 0;
static const int32_t SELECT_KEY_MODE_FP32_INPUT_VALID = 1;

static const std::vector<std::string> COMPILE_INFO_KEY = {"core_num", "ub_size"};

struct SparseSegmentSumGradTilingParams {
  int32_t select_tiling_mode = 0;
  int32_t need_core_num = 1;
  int32_t segment_num_each_core = 0;
  int32_t segment_num_rest = 0;
  int32_t grad_second_dim_size = 0;
};

struct InputInfo {
  int32_t ids_size = 0;
  int32_t last_axis_size = 0;
};

void CalcNeedCoreNum(int32_t ids_size, int32_t core_num, int32_t& need_core_num) {
  int32_t ele_num = 0;
  if (core_num != 0) {
    ele_num = ids_size / core_num;
  }
  if (ele_num >= 1) {
    need_core_num = core_num;
  } else {
    need_core_num = ids_size;
  }
}

bool GetSparseSegmentSumGradCompileParams(const std::string& op_type, const std::vector<int64_t>& op_compile_info,
                                          int32_t& core_num, int32_t& ub_size) {
  OP_TILING_CHECK(
      op_compile_info.size() != COMPILE_INFO_KEY.size(),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "the compile info num is not equal expect compile_info(%zu), is %zu",
                                      COMPILE_INFO_KEY.size(), op_compile_info.size()),
      return false);

  core_num = op_compile_info[0];
  ub_size = op_compile_info[1];

  return true;
}

void WriteTilingParams(const std::string& opType, const SparseSegmentSumGradTilingParams& params,
                       utils::OpRunInfo& run_info) {
  // common params
  run_info.AddTilingData(params.select_tiling_mode);
  run_info.AddTilingData(params.need_core_num);
  run_info.AddTilingData(params.segment_num_each_core);
  run_info.AddTilingData(params.segment_num_rest);
  run_info.AddTilingData(params.grad_second_dim_size);
}

void PrintTilingParams(const std::string& op_type, const SparseSegmentSumGradTilingParams& params) {
  OP_LOGD(op_type, " : params.select_tiling_mode=%d", params.select_tiling_mode);
  OP_LOGD(op_type, " : params.need_core_num=%d", params.need_core_num);
  OP_LOGD(op_type, " : params.segment_num_each_core=%d", params.segment_num_each_core);
  OP_LOGD(op_type, " : params.segment_num_rest=%d", params.segment_num_rest);
  OP_LOGD(op_type, " : params.grad_second_dim_size=%d", params.grad_second_dim_size);
}

bool CalcOutputDimInvalid(const std::string& op_type, const ge::Operator& op_paras, utils::OpRunInfo& run_info,
                          int32_t ids_size) {
  int32_t output_dim0;
  OP_TILING_CHECK(!(ops::GetConstInt(op_paras, 3, output_dim0)),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input out_dim0 failed."), return false);
  OP_LOGI(op_type, " : output_dim0=%d", output_dim0);
  if (ids_size == 0 || output_dim0 == 0) {
    SparseSegmentSumGradTilingParams params;
    params.select_tiling_mode = SELECT_KEY_MODE_FP32_INPUT_INVALID;
    params.need_core_num = 1;
    WriteTilingParams(op_type, params, run_info);
    // cout tiling params
    PrintTilingParams(op_type, params);
    // BlockDim, core num used in tik op
    run_info.SetBlockDim(params.need_core_num);
    return true;
  }
  return false;
}

bool CalcOutputDimValid(const std::string& op_type, ge::OpDescPtr& operator_info, utils::OpRunInfo& run_info,
                        const std::vector<int64_t>& opCompileInfo, const InputInfo& info) {
  // get compile info
  int32_t core_num = 1;
  int32_t ub_size = 0;
  if (operator_info == nullptr) {
    return false;
  }

  if (!GetSparseSegmentSumGradCompileParams(op_type, opCompileInfo, core_num, ub_size)) {
    return false;
  }

  const ge::DataType input_dtype = operator_info->MutableInputDesc(0)->GetDataType();
  if (input_dtype == ge::DT_FLOAT || input_dtype == ge::DT_FLOAT16) {
    SparseSegmentSumGradTilingParams params;
    params.select_tiling_mode = SELECT_KEY_MODE_FP32_INPUT_VALID;
    CalcNeedCoreNum(info.ids_size, core_num, params.need_core_num);
    if (params.need_core_num != 0) {
      params.segment_num_each_core = info.ids_size / params.need_core_num;
      params.segment_num_rest = info.ids_size % params.need_core_num;
    } else {
      params.need_core_num = 1;
      params.segment_num_each_core = info.ids_size;
      params.segment_num_rest = 0;
    }
    params.grad_second_dim_size = info.last_axis_size;
    // write tiling params to run_info
    WriteTilingParams(op_type, params, run_info);
    // cout tiling params
    PrintTilingParams(op_type, params);
    // BlockDim, core num used in tik op
    run_info.SetBlockDim(params.need_core_num);
    OP_LOGI("op[%s] op tiling success.", op_type.c_str());
    return true;
  }
  return false;
}

// tiling function
bool SparseSegmentSumGradTiling(const std::string& op_type, const ge::Operator& op_paras,
                                const std::vector<int64_t>& opCompileInfo, utils::OpRunInfo& run_info) {
  using namespace ge;

  OP_LOGI(op_type.c_str(), "Tiling running.");

  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get op_info failed."),
                  return false);
  // get input grad Desc
  auto input_desc = operator_info->MutableInputDesc(GRAD_IDX);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get grad failed."), return false);
  const GeShape& input_shape = input_desc->MutableShape();
  int32_t grad_size = GetTensorSize(input_shape);
  int32_t grad_dim0_size = input_shape.GetDim(0);

  // get input indices Desc
  input_desc = operator_info->MutableInputDesc(INDICES_IDX);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get indices failed."), return false);
  const GeShape& indices_shape = input_desc->MutableShape();
  OP_TILING_CHECK(indices_shape.GetDimNum() != 1, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "dim of indices must be 1."),
                  return false);
  // get input segment_ids Desc
  input_desc = operator_info->MutableInputDesc(SEGMENT_IDX);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get ids failed."), return false);
  const GeShape& segment_ids_shape = input_desc->MutableShape();
  OP_TILING_CHECK(segment_ids_shape.GetDimNum() != 1,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "dim of segment_ids must be 1."), return false);
  int32_t ids_size = GetTensorSize(segment_ids_shape);
  int32_t indices_size = GetTensorSize(indices_shape);
  OP_TILING_CHECK(ids_size != indices_size,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "size of indices must be equal to size of segment_ids."),
                  return false);

  if (CalcOutputDimInvalid(op_type, op_paras, run_info, ids_size) == true) {
    return true;
  }
  InputInfo info;
  info.ids_size = ids_size;
  if (grad_dim0_size != 0) {
    info.last_axis_size = grad_size / grad_dim0_size;
  }
  return CalcOutputDimValid(op_type, operator_info, run_info, opCompileInfo, info);
}

// register tiling interface of the SparseSegmentSumGrad op.
REGISTER_OP_TILING_V3_WITH_VECTOR(SparseSegmentSumGrad, SparseSegmentSumGradTiling, COMPILE_INFO_KEY,
                                  NO_OPTIONAL_VALUE);
}  // namespace optiling
