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
 * \file sparse_apply_proximal_adagrad_d.cpp
 * \brief
 */
#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"

#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "error_log.h"

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
const std::string DTYPE_FP32 = "float32";
const std::string STR_CORE_NUM = "core_num";
const std::string STR_UB_SIZE = "ub_size";
const std::string STR_UB_TENSOR_NUM = "ub_tensor_num";
const int32_t SELECT_KEY_FP32_E_ALIGN = 1;

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
  if (num % factor != 0) {
    return (num / factor + 1) * factor;
  }
  return num;
}

int32_t SapaCeilDiv(const int32_t& num, const int32_t& factor) {
  int32_t res;
  res = (num % factor == 0) ? num / factor : num / factor + 1;
  return res;
}

bool GetCompileParams(const std::string& op_type, const nlohmann::json& op_compile_info_json, int32_t& core_num,
                      int32_t& ub_size, int32_t& ub_tensor_num) {
  using namespace nlohmann;
  const auto& vars = op_compile_info_json["vars"];
  // core num
  if (vars.count(STR_CORE_NUM) == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op [%s] core_num is null", op_type.c_str());
    return false;
  }
  core_num = vars[STR_CORE_NUM].get<std::int32_t>();
  // ub size
  if (vars.count(STR_UB_SIZE) == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ub_size is null");
    return false;
  }
  ub_size = vars[STR_UB_SIZE].get<std::int32_t>();
  // ub tensor num
  if (vars.count(STR_UB_TENSOR_NUM) == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ub_tensor_num is null");
    return false;
  }
  ub_tensor_num = vars[STR_UB_TENSOR_NUM].get<std::int32_t>();
  return true;
}

int32_t ComputeRowsInUb(const std::string& var_dtype, const int32_t& ub_size, const int32_t& e_size,
                        const int32_t& ub_tesnor_num) {
  int32_t row = 0;
  if (var_dtype == DTYPE_FP32) {
    int32_t e_size_ceil = SapaCeil(e_size, MASK_FP32);
    row = (ub_size - ub_tesnor_num * ONE_KB) / (e_size_ceil + 1);
    row = row / BYTE_FP32;
    row = row / FP32_ELE_NUM_BLOCK * FP32_ELE_NUM_BLOCK;
  }
  return row;
}

void WriteTilingParams(const SapaTilingParamsFp32& params, OpRunInfo& run_info) {
  ByteBufferPut(run_info.tiling_data, params.select_key);
  ByteBufferPut(run_info.tiling_data, params.need_core_num);
  ByteBufferPut(run_info.tiling_data, params.idx_mov_times);
  ByteBufferPut(run_info.tiling_data, params.idx_front_num);
  ByteBufferPut(run_info.tiling_data, params.idx_last_num);
  ByteBufferPut(run_info.tiling_data, params.idx_front_burstlen);
  ByteBufferPut(run_info.tiling_data, params.idx_last_burstlen);
  ByteBufferPut(run_info.tiling_data, params.one_row_burstlen);
  ByteBufferPut(run_info.tiling_data, params.one_row_num);
  ByteBufferPut(run_info.tiling_data, params.vec_repeat_time);
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

bool SparseApplyProximalAdagradTiling(const std::string& op_type, const TeOpParas& op_paras,
                                      const nlohmann::json& op_compile_info_json, OpRunInfo& run_info) {
  GELOGI("op[%s] op tiling begin.", op_type.c_str());
  if (op_paras.inputs.size() < OP_PARAS_NUM) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.size() is less than 7");
    return false;
  }
  if (op_paras.inputs[GRAD_IDX].tensor.size() == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "grad tensor is null");
    return false;
  }
  const std::vector<int64_t>& grad_shape = op_paras.inputs[GRAD_IDX].tensor[0].shape;
  if (op_paras.inputs[INDICES_IDX].tensor.size() == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "indices tensor is null");
    return false;
  }
  // get size
  const std::vector<int64_t>& indices_shape = op_paras.inputs[INDICES_IDX].tensor[0].shape;
  const int32_t& grad_size = std::accumulate(grad_shape.begin(), grad_shape.end(), 1, std::multiplies<int>());
  const int32_t& indices_size = std::accumulate(indices_shape.begin(), indices_shape.end(), 1, std::multiplies<int>());
  int32_t e_size = grad_size / indices_size;
  GELOGD("op [%s] : grad_size=%d, indices_size=%d, e_size=%d", op_type.c_str(), grad_size, indices_size, e_size);
  if (grad_shape.size() < indices_shape.size()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "dim of grad must be greater than or equal with dim of indices");
    return false;
  }
  for (unsigned i = 0; i < indices_shape.size(); i++) {
    if (grad_shape[i] != indices_shape[i]) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "front shape of grad must be equal with indices shape");
      return false;
    }
  }
  if (op_paras.inputs[VAR_IDX].tensor.size() == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "var tensor is null");
    return false;
  }
  const std::vector<int64_t>& var_shape = op_paras.inputs[VAR_IDX].tensor[0].shape;
  if (op_paras.inputs[ACCUM_IDX].tensor.size() == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "accum tensor is null");
    return false;
  }
  const std::vector<int64_t>& accum_shape = op_paras.inputs[ACCUM_IDX].tensor[0].shape;
  if (var_shape.size() != accum_shape.size()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "var_shape.size() must be equal with accum_shape.size()");
    return false;
  }
  for (unsigned i = 0; i < var_shape.size(); i++) {
    if (var_shape[i] != accum_shape[i]) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "var_shape must be equal with accum_shape");
      return false;
    }
  }
  if (var_shape.size() < grad_shape.size() - indices_shape.size()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "dim of var_shape must be greater than or equal with \
                                    the difference between dim of grad and dim of indices");
    return false;
  }
  for (uint32_t i = 1; i <= grad_shape.size() - indices_shape.size(); i++) {
    int32_t var_idx = var_shape.size() - i;
    int32_t grad_idx = grad_shape.size() - i;
    if (var_shape[var_idx] != grad_shape[grad_idx]) {
      VECTOR_INNER_ERR_REPORT_TILIING( op_type.c_str(), "second dimision in var_shape must be equal with  dimension in grad_shape");
      return false;
    }
  }
  // get dtype
  const std::string& grad_dtype = op_paras.inputs[GRAD_IDX].tensor[0].dtype;
  // get compile params
  int32_t core_num = 1;
  int32_t ub_size = 0;
  int32_t ub_tesnor_num = 0;
  bool flag = true;
  flag = GetCompileParams(op_type, op_compile_info_json, core_num, ub_size, ub_tesnor_num);
  if (!flag) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile info params failed");
    return false;
  }
  if (grad_dtype == DTYPE_FP32) {
    GELOGI("op[%s] float32", op_type.c_str());
    SapaTilingParamsFp32 params;
    InitTilingParams(params);
    if (e_size % FP32_ELE_NUM_BLOCK == 0) {
      // e_size 32B align
      GELOGI("op[%s] e_size 32B align", op_type.c_str());
      params.select_key = 1;
      if (e_size < MIN_ELE_SIZE_USING_ALL_CORE) {
        GELOGI("op[%s] need one core", op_type.c_str());
        params.need_core_num = 1;
        params.idx_front_num = ComputeRowsInUb(grad_dtype, ub_size, e_size, ub_tesnor_num);
        if (params.idx_front_num == 0) {
          VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op params.idx_front_num == 0");
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
        GELOGI("op[%s] need full core", op_type.c_str());
        params.need_core_num = core_num;
      }
    }
    // write tiling params to run_info
    WriteTilingParams(params, run_info);
    // print tiling params
    PrintTilingParams(op_type, params);
    // BlockDim, core num used in tik op
    run_info.block_dim = params.need_core_num;
    // workspace, null for tik op
    std::vector<int64_t> workspace;
    run_info.workspaces = workspace;
  }
  GELOGI("op[%s] op tiling success.", op_type.c_str());
  return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(SparseApplyProximalAdagradD, SparseApplyProximalAdagradTiling);
}  // namespace optiling
