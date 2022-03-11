/**
 * Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
 * \file sparse_apply_proximal_adagrad_d.cc
 * \brief
 */
#include <string>
#include <nlohmann/json.hpp>
#include "op_log.h"
#include "error_log.h"
#include "op_tiling.h"
#include "op_tiling_util.h"
#include "graph/debug/ge_log.h"
#include "graph/utils/op_desc_utils.h"
#include "../op_proto/util/error_util.h"

namespace optiling {
const int32_t BYTE_BLOCK = 32;
const int32_t MIN_ELE_SIZE_USING_ALL_CORE = 1024;
const int32_t ONE_KB = 1024;
const int32_t BYTE_FP32 = 4;
const int32_t MASK_FP32 = 64;
const int32_t MAX_REPEAT_TIME = 255;
const int32_t FP32_ELE_NUM_BLOCK = 8;
const int32_t OP_PARAS_NUM = 7;
const int32_t VAR_IDX = 0;
const int32_t ACCUM_IDX = 1;
const int32_t GRAD_IDX = 5;
const int32_t INDICES_IDX = 6;

static const std::vector<std::string> COMPILE_INFO_KEY = {"ub_size", "core_num", "ub_tensor_num"};

struct SapaTilingParamsFp32 {
  int32_t select_key;
  int32_t need_core_num;
  int32_t idx_mov_times;
  int32_t idx_front_num;
  int32_t idx_last_num;
  int32_t idx_front_burstlen;
  int32_t idx_last_burstlen;
  int32_t one_row_burstlen;
  int32_t one_row_num;
  int32_t vec_repeat_time;
};

void InitTilingParams(SapaTilingParamsFp32& params) {
  params.select_key = 0;
  params.need_core_num = 0;
  params.idx_mov_times = 0;
  params.idx_front_num = 0;
  params.idx_last_num = 0;
  params.idx_front_burstlen = 0;
  params.idx_last_burstlen = 0;
  params.one_row_burstlen = 0;
  params.one_row_num = 0;
  params.vec_repeat_time = 0;
}

int32_t SapaCeil(const int32_t& num, const int32_t& factor) {
  OP_TILING_CHECK(factor == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("sparse_apply_proximal_adagrad_d", "factor must not be zero"),
                  return -1);
  if (num % factor != 0) {
    return (num / factor + 1) * factor;
  }
  return num;
}

int32_t SapaCeilDiv(const int32_t& num, const int32_t& factor) {
  OP_TILING_CHECK(factor == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("sparse_apply_proximal_adagrad_d", "factor must not be zero"),
                  return -1);
  int32_t res = (num % factor == 0) ? num / factor : num / factor + 1;
  return res;
}

int32_t ComputeRowsInUb(const ge::DataType& var_dtype, const int32_t& ub_size, const int32_t& e_size,
                        const int32_t& ub_tesnor_num) {
  int32_t row = 0;
  if (var_dtype == DT_FLOAT) {
    int32_t e_size_ceil = SapaCeil(e_size, MASK_FP32);
    row = (ub_size - ub_tesnor_num * ONE_KB) / (e_size_ceil + 1);
    row = row / BYTE_FP32;
    row = row / FP32_ELE_NUM_BLOCK * FP32_ELE_NUM_BLOCK;
  }
  return row;
}

void WriteTilingParams(const SapaTilingParamsFp32& params, utils::OpRunInfo& run_info) {
  run_info.AddTilingData(params.select_key);
  run_info.AddTilingData(params.need_core_num);
  run_info.AddTilingData(params.idx_mov_times);
  run_info.AddTilingData(params.idx_front_num);
  run_info.AddTilingData(params.idx_last_num);
  run_info.AddTilingData(params.idx_front_burstlen);
  run_info.AddTilingData(params.idx_last_burstlen);
  run_info.AddTilingData(params.one_row_burstlen);
  run_info.AddTilingData(params.one_row_num);
  run_info.AddTilingData(params.vec_repeat_time);
}

void PrintTilingParams(const std::string& op_type, const SapaTilingParamsFp32& params) {
  GELOGD("op [%s] : params.select_key=%d", op_type.c_str(), params.select_key);
  GELOGD("op [%s] : params.need_core_num=%d", op_type.c_str(), params.need_core_num);
  GELOGD("op [%s] : params.idx_mov_times=%d", op_type.c_str(), params.idx_mov_times);
  GELOGD("op [%s] : params.idx_front_num=%d", op_type.c_str(), params.idx_front_num);
  GELOGD("op [%s] : params.idx_last_num=%d", op_type.c_str(), params.idx_last_num);
  GELOGD("op [%s] : params.idx_front_burstlen=%d", op_type.c_str(), params.idx_front_burstlen);
  GELOGD("op [%s] : params.idx_last_burstlen=%d", op_type.c_str(), params.idx_last_burstlen);
  GELOGD("op [%s] : params.one_row_burstlen=%d", op_type.c_str(), params.one_row_burstlen);
  GELOGD("op [%s] : params.one_row_num=%d", op_type.c_str(), params.one_row_num);
  GELOGD("op [%s] : params.vec_repeat_time=%d", op_type.c_str(), params.vec_repeat_time);
}

bool SparseApplyProximalAdagradTiling(const std::string& opType, const ge::Operator& opParas,
                                      const std::vector<int64_t>& op_info, utils::OpRunInfo& run_info) {
  OP_LOGD(opType, "op tiling begin.");
  auto operator_info = OpDescUtils::GetOpDescFromOperator(opParas);
  if (operator_info == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "operator_info is nullptr");
    return false;
  }

  if (operator_info->GetInputsSize() < OP_PARAS_NUM) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "operator_info.size() is less than 7");
    return false;
  }

  auto input_grad_dec = operator_info->MutableInputDesc(GRAD_IDX);
  if (input_grad_dec == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "grad tensor is null");
    return false;
  }
  auto input_indices_dec = operator_info->MutableInputDesc(INDICES_IDX);
  if (input_indices_dec == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "grad tensor is null");
    return false;
  }
  const ge::GeShape& grad_shape = input_grad_dec->MutableShape();
  const ge::GeShape& indices_shape = input_indices_dec->MutableShape();
  const int64_t& grad_size = GetTensorSize(grad_shape);
  const int64_t& indices_size =GetTensorSize(indices_shape);
  if (indices_size == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "indices_size is 0");
    return false;
  }

  int32_t e_size = (int32_t)(grad_size / indices_size);
  GELOGD("op [%s] : grad_size=%d, indices_size=%d, e_size=%d", opType.c_str(), grad_size, indices_size, e_size);
  if (grad_shape.GetDimNum() < indices_shape.GetDimNum()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "dim of grad must be greater than or equal with dim of indices");
    return false;
  }
  for (unsigned i = 0; i < indices_shape.GetDimNum(); i++) {
    if (grad_shape.GetDim(i) != indices_shape.GetDim(i)) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "front shape of grad must be equal with indices shape");
      return false;
    }
  }

  if (operator_info->MutableInputDesc(VAR_IDX) == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "grad tensor is null");
    return false;
  }
  const ge::GeShape& var_shape = operator_info->MutableInputDesc(VAR_IDX)->MutableShape();
  const int64_t var_shape_num = var_shape.GetDimNum();
  if (operator_info->MutableInputDesc(ACCUM_IDX) == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "grad tensor is null");
    return false;
  }
  const ge::GeShape& accum_shape = operator_info->MutableInputDesc(ACCUM_IDX)->MutableShape();
  const int64_t accum_shape_num = accum_shape.GetDimNum();

  if (var_shape_num != accum_shape_num) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "var_shape.size() must be equal with accum_shape.size()");
    return false;
  }
  for (int64_t i = 0; i < var_shape_num; i++) {
    if (var_shape.GetDim(i) != accum_shape.GetDim(i)) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "var_shape must be equal with accum_shape");
      return false;
    }
  }

  const int64_t grad_shape_num = grad_shape.GetDimNum();
  const int64_t indices_shape_num = indices_shape.GetDimNum();
  if (var_shape_num < (grad_shape_num - indices_shape_num)) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType,
                                    "dim of var_shape must be greater than or equal with \
                                    the difference between dim of grad and dim of indices");
    return false;
  }

  
  for (int64_t i = 1; i <= grad_shape_num - indices_shape_num; i++) {
    int32_t var_idx = var_shape_num - i;
    int32_t grad_idx = grad_shape_num - i;
    if (var_shape.GetDim(var_idx) != grad_shape.GetDim(grad_idx)) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType.c_str(),
                                      "second dimision in var_shape must be equal with  dimension in grad_shape");
      return false;
    }
  }

  // get compile params
  if (op_info.size() != COMPILE_INFO_KEY.size()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op_info size not expected");
    return false;
  }
  int32_t ub_size = static_cast<int32_t>(op_info[0]);
  int32_t core_num = static_cast<int32_t>(op_info[1]);
  int32_t ub_tesnor_num = static_cast<int32_t>(op_info[2]);

  // get dtype
  const ge::DataType grad_dtype = operator_info->MutableInputDesc(GRAD_IDX)->GetDataType();
  if (grad_dtype == DT_FLOAT) {
    OP_LOGD(opType, "op is float32");
    SapaTilingParamsFp32 params;
    InitTilingParams(params);
    if (e_size % FP32_ELE_NUM_BLOCK == 0) {
      // e_size 32B align
      OP_LOGD(opType, "op e_size 32B align");
      params.select_key = 1;
      if (e_size < MIN_ELE_SIZE_USING_ALL_CORE) {
        OP_LOGD(opType, "op need one core");
        params.need_core_num = 1;
        params.idx_front_num = ComputeRowsInUb(grad_dtype, ub_size, e_size, ub_tesnor_num);
        if (params.idx_front_num == 0) {
          VECTOR_INNER_ERR_REPORT_TILIING(opType, "op params.idx_front_num == 0");
          return false;
        }
        params.idx_mov_times = SapaCeilDiv(indices_size, params.idx_front_num);
        params.idx_last_num = indices_size - (params.idx_mov_times - 1) * params.idx_front_num;
        params.idx_front_burstlen = SapaCeilDiv(params.idx_front_num * BYTE_FP32, BYTE_BLOCK);
        params.idx_last_burstlen = SapaCeilDiv(params.idx_last_num * BYTE_FP32, BYTE_BLOCK);
        params.one_row_num = e_size;
        params.one_row_burstlen = SapaCeilDiv(params.one_row_num * BYTE_FP32, BYTE_BLOCK);
        params.vec_repeat_time = SapaCeilDiv(params.one_row_num, MASK_FP32);
      } else {
        OP_LOGD(opType, "op need full core");
        params.need_core_num = core_num;
      }
    }
    // write tiling params to run_info
    WriteTilingParams(params, run_info);
    // print tiling params
    PrintTilingParams(opType, params);
    // block_dim, core num used in tik op
    run_info.SetBlockDim(params.need_core_num);
  }
  OP_LOGD(opType, "op tiling success.");
  return true;
}

REGISTER_OP_TILING_V3_WITH_VECTOR(SparseApplyProximalAdagradD, SparseApplyProximalAdagradTiling, COMPILE_INFO_KEY,
                                  NO_OPTIONAL_VALUE);
}  // namespace optiling
