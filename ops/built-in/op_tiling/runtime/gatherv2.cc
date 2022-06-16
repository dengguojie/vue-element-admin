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
 * \file gather_v2.cpp
 * \brief
 */
#include "gatherv2.h"

#include <nlohmann/json.hpp>
#include <sstream>
#include <cctype>
#include "error_log.h"
#include "op_tiling_util.h"
#include "op_const.h"
#include "runtime2_util.h"

namespace optiling {
const size_t INPUT_IDX_AXIS = 2;
const int64_t HALF_UB = 2;
const int64_t DATA_VALUE = 1024;
const int64_t NUM_32 = 32;
const int64_t ACTUAL_NUM = 56.5;
const int64_t GATE_VALUE = 0.012;
const int64_t BLOCK_SIZE = 32;
const int64_t PARAMS_CACHED_UB = 100 * 1024;
const int64_t RESERVED_UB_SIZE = 6 * 1024;

// A. block tiling: indices tiling
// 1. one params row size is smaller than 32B
// params is not cache
const int64_t TILING_MODE_1 = 1;
// params is cache in UB
const int64_t TILING_MODE_4 = 4;
// params is cache in L1
const int64_t TILING_MODE_13 = 13;

// 2. one params row size is greater than or equal to 32B
// params_row is not 32B aligned
const int64_t TILING_MODE_2 = 2;
// the data of one params row can not store in half UB, need tiling
const int64_t TILING_MODE_5 = 5;

// 3. params_row is 32B aligned
// params is not cache in UB or L1
const int64_t TILING_MODE_3 = 3;
// params is cache in UB
const int64_t TILING_MODE_6 = 6;
// params is cache in L1
const int64_t TILING_MODE_7 = 7;

// B. block tiling: params_pre tiling
// 1. one params row size is smaller than 32B
// params is not cache
const int64_t TILING_MODE_8 = 8;
// params is cache in UB
const int64_t TILING_MODE_9 = 9;

// 2. params_row is 32B aligned
// params is not cache in UB or L1
const int64_t TILING_MODE_10 = 10;
// params is cache in UB
const int64_t TILING_MODE_11 = 11;
// params is cache in L1
const int64_t TILING_MODE_12 = 12;

// tiling_mode with impl_mode
const int64_t TILING_MODE_14 = 14;

// tiling_mode with batch_dims
// 1.one params row size is smaller than 32B
// 1.1 params is cached in UB
const int64_t TILING_MODE_20 = 20;
const int64_t TILING_MODE_21 = 21;
const int64_t TILING_MODE_22 = 22;
// 1.2 params is not cached in UB
const int64_t TILING_MODE_23 = 23;
const int64_t TILING_MODE_24 = 24;
const int64_t TILING_MODE_25 = 25;

// 2.one params row size is large than 32B and not align
const int64_t TILING_MODE_26 = 26;
const int64_t TILING_MODE_27 = 27;
const int64_t TILING_MODE_28 = 28;

// 3.one params row size is align
// 3.1 params is cached in UB
const int64_t TILING_MODE_29 = 29;
const int64_t TILING_MODE_30 = 30;
const int64_t TILING_MODE_31 = 31;
// 3.2 params is not cached in UB
const int64_t TILING_MODE_32 = 32;
const int64_t TILING_MODE_33 = 33;
const int64_t TILING_MODE_34 = 34;

// 4. large params row size
const int64_t TILING_MODE_35 = 35;
const int64_t TILING_MODE_36 = 36;
const int64_t TILING_MODE_37 = 37;

// 5. small indices row size
const int64_t TILING_MODE_38 = 38;
const int64_t TILING_MODE_39 = 39;

// 6. small params and indices row size
const int64_t TILING_MODE_40 = 40;
const int64_t TILING_MODE_41 = 41;

// define impl_mode of gather_v2 attr
static const int64_t IMPL_MODE_HIGH_PERFORMANCE_VALUE = 1;

bool CheckAndUpdateAxisAndBatchdims(int64_t& axis, gert::Shape& x_shape, gert::Shape& indies_shape, int64_t& batch_dims,
                                    int64_t params_dims, int64_t indices_dims) {
  OP_TILING_CHECK(params_dims <= 0 || indices_dims <= 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("GatherV2:", "GatherV2Tiling: params_dims or indices_dims is 0."),
                  return false);

  OP_TILING_CHECK(axis < -params_dims || axis >= params_dims,
                  VECTOR_INNER_ERR_REPORT_TILIING("GatherV2:", "op GatherV2Tiling: axis is invalid"), return false);

  if (axis < 0) {
    axis += params_dims;
  }

  if (batch_dims != 0) {
    OP_TILING_CHECK(batch_dims < -indices_dims || batch_dims >= indices_dims,
                    VECTOR_INNER_ERR_REPORT_TILIING("GatherV2:", "op GatherV2Tiling: batch_dims is invalid."),
                    return false);
    if (batch_dims < 0) {
      batch_dims += indices_dims;
    }
    OP_TILING_CHECK(
        batch_dims >= params_dims,
        VECTOR_INNER_ERR_REPORT_TILIING("GatherV2:", "op GatherV2Tiling: batch_dims must be less than rank(params)."),
        return false);
    OP_TILING_CHECK(batch_dims > axis,
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        "GatherV2:", "op GatherV2Tiling: batch_dims must be less than or equal to axis."),
                    return false);
    for (int64_t i = 0; i < batch_dims; i++) {
      if (x_shape.GetDim(i) != indies_shape.GetDim(i)) {
        VECTOR_INNER_ERR_REPORT_TILIING("GatherV2",
                                        "op GatherV2Tiling: Params.shape[:batch_dims] "
                                        "should be equal to indices.shape[:batch_dims].");
        return false;
      }
    }
  }
  return true;
}

bool DoImplModeTiling(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info) {
  OP_TILING_CHECK(
      compile_info->impl_mode != IMPL_MODE_HIGH_PERFORMANCE_VALUE,
      OP_LOGD("GatherV2", "[DoImplModeTiling] no need cache params row 0 for impl_mode is not high_performance"),
      return false);
  OP_TILING_CHECK(
      params->params_total * compile_info->params_dsize <= PARAMS_CACHED_UB,
      OP_LOGD("GatherV2", "[DoImplModeTiling] no need cache params row 0 for all params can be cached in UB"),
      return false);
  OP_TILING_CHECK(
      params->indices_num < compile_info->core_num * BLOCK_SIZE / compile_info->params_dsize,
      OP_LOGD("GatherV2", "[DoImplModeTiling] no need cache params row 0 for the num of indices is small"),
      return false);

  params->tiling_mode = TILING_MODE_14;
  params->need_core_num = compile_info->core_num;
  params->indices_num_each_core = (params->indices_num + params->need_core_num - 1) / params->need_core_num;
  params->indices_num_remaining = params->indices_num / params->need_core_num;

  params->tail_process_core = params->indices_num % params->need_core_num;
  if (params->tail_process_core == 0) {
    params->tail_process_core = params->need_core_num;
  }
  OP_LOGD("GatherV2", "[DoImplModeTiling] For the core which blockId <= %ld, %ld indices will be process",
          params->tail_process_core, params->indices_num_each_core);
  OP_LOGD("GatherV2", "[DoImplModeTiling] For the core which blockId > %ld, %ld indices will be process",
          params->tail_process_core, params->indices_num_remaining);

  return true;
}

// compute tiling params for tiling_mode 10&11&12
bool BlockAlignForParamsTiling(GatherV2TilingParams* params, int64_t indices_num_per_loop, int64_t res_ub_size,
                               int64_t params_dsize) {
  OP_TILING_CHECK(indices_num_per_loop == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("GatherV2:", "indices_num_per_loop = 0 is not support"),
                  return false);
  params->indices_loop_num = params->indices_num_each_core / indices_num_per_loop;
  params->indices_row_num_once = indices_num_per_loop;
  if (params->indices_num_each_core % params->indices_row_num_once != 0) {
    params->indices_row_num_last = params->indices_num_each_core % params->indices_row_num_once;
  }

  params->row_num_once_ub = res_ub_size / (params->params_row * params_dsize);
  OP_TILING_CHECK((params->row_num_once_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("GatherV2:", "Devide by row_num_once_ub[%ld] exception.",
                                                  params->row_num_once_ub),
                  return false);
  params->inner_loop_num = params->indices_row_num_once / params->row_num_once_ub;
  if (params->indices_row_num_once % params->row_num_once_ub != 0) {
    params->row_num_once_tail_ub = params->indices_row_num_once % params->row_num_once_ub;
  }

  params->row_num_last_ub = res_ub_size / (params->params_row * params_dsize);
  OP_TILING_CHECK((params->row_num_last_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("GatherV2:", "Devide by row_num_last_ub[%ld] exception.",
                                                  params->row_num_last_ub),
                  return false);
  params->inner_loop_num_last = params->indices_row_num_last / params->row_num_last_ub;
  if (params->indices_row_num_last % params->row_num_last_ub != 0) {
    params->row_num_last_tail_ub = params->indices_row_num_last % params->row_num_last_ub;
  }
  return true;
}

// compute tiling params for tiling_mode 1&4&13
bool BlockLessForIndicesTiling(GatherV2TilingParams* params, int64_t indices_num_per_loop, int64_t res_ub_size,
                               int64_t params_d_size, int64_t block_num) {
  OP_TILING_CHECK(indices_num_per_loop == 0 || block_num == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("GatherV2:", "indices_num_per_loop or block_num = 0 is not support"),
                  return false);
  params->indices_loop_num = params->indices_num_each_core / indices_num_per_loop;
  params->indices_row_num_once = indices_num_per_loop;
  if (params->indices_num_each_core % params->indices_row_num_once != 0) {
    params->indices_row_num_last = params->indices_num_each_core % params->indices_row_num_once;
  }

  params->row_num_once_ub = res_ub_size / (params->params_row * params_d_size);
  if (int(params->row_num_once_ub % block_num) != 0) {
    params->row_num_once_ub = int(params->row_num_once_ub / block_num) * block_num;
  }
  OP_TILING_CHECK((params->row_num_once_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("Gather Tiling:", "Devide by row_num_once_ub[%ld] exception.",
                                                  params->row_num_once_ub),
                  return false);
  params->inner_loop_num = params->indices_row_num_once / params->row_num_once_ub;
  if (params->indices_row_num_once % params->row_num_once_ub != 0) {
    params->row_num_once_tail_ub = params->indices_row_num_once % params->row_num_once_ub;
  }
  if (params->inner_loop_num > 0 && params->row_num_once_tail_ub > 0 &&
      params->row_num_once_tail_ub * params->params_row < block_num) {
    params->inner_loop_num = params->inner_loop_num - 1;
    params->row_num_once_tail_ub = params->row_num_once_tail_ub + params->row_num_once_ub;
  }

  params->row_num_last_ub = res_ub_size / (params->params_row * params_d_size);
  if (int(params->row_num_last_ub % block_num) != 0) {
    params->row_num_last_ub = int(params->row_num_last_ub / block_num) * block_num;
  }
  OP_TILING_CHECK((params->row_num_last_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("Gather Tiling:", "Devide by row_num_last_ub[%ld] exception.",
                                                  params->row_num_last_ub),
                  return false);
  params->inner_loop_num_last = params->indices_row_num_last / params->row_num_last_ub;
  if (params->indices_row_num_last % params->row_num_last_ub != 0) {
    params->row_num_last_tail_ub = params->indices_row_num_last % params->row_num_last_ub;
  }
  if (params->inner_loop_num_last > 0 && params->row_num_last_tail_ub > 0 &&
      params->row_num_last_tail_ub * params->params_row < block_num) {
    params->inner_loop_num_last = params->inner_loop_num_last - 1;
    params->row_num_last_tail_ub = params->row_num_last_tail_ub + params->row_num_once_ub;
  }
  OP_LOGD("gatherv2", "BlockLessForIndicesTiling END");
  return true;
}

// compute tiling params for tiling_mode 8&9
bool BlockLessForParamsTiling(GatherV2TilingParams* params, int64_t indices_num_per_loop, int64_t res_ub_size,
                              int64_t params_dsize, int64_t block_num) {
  OP_TILING_CHECK(
      indices_num_per_loop == 0 || block_num == 0,
      VECTOR_INNER_ERR_REPORT_TILIING("Gather Tiling:", "indices_num_per_loop or block_num = 0 is not support"),
      return false);
  params->indices_loop_num = params->indices_num_each_core / indices_num_per_loop;
  params->indices_row_num_once = indices_num_per_loop;
  if (params->indices_num_each_core % params->indices_row_num_once != 0) {
    params->indices_row_num_last = params->indices_num_each_core % params->indices_row_num_once;
  }

  params->row_num_once_ub = res_ub_size / (params->params_row * params_dsize);
  if (int(params->row_num_once_ub % block_num) != 0) {
    params->row_num_once_ub = int(params->row_num_once_ub / block_num) * block_num;
  }
  OP_TILING_CHECK((params->row_num_once_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("Gather Tiling:", "Devide by row_num_once_ub[%ld] exception.",
                                                  params->row_num_once_ub),
                  return false);
  params->inner_loop_num = params->indices_row_num_once / params->row_num_once_ub;
  if (params->indices_row_num_once % params->row_num_once_ub != 0) {
    params->row_num_once_tail_ub = params->indices_row_num_once % params->row_num_once_ub;
  }
  if (params->inner_loop_num > 0 && params->row_num_once_tail_ub > 0 &&
      params->row_num_once_tail_ub * params->params_row < block_num) {
    params->inner_loop_num = params->inner_loop_num - 1;
    params->row_num_once_tail_ub = params->row_num_once_tail_ub + params->row_num_once_ub;
  }

  params->row_num_last_ub = res_ub_size / (params->params_row * params_dsize);
  if (int(params->row_num_last_ub % block_num) != 0) {
    params->row_num_last_ub = int(params->row_num_last_ub / block_num) * block_num;
  }
  OP_TILING_CHECK((params->row_num_last_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("Gather Tiling:", "Devide by row_num_last_ub[%ld] exception.",
                                                  params->row_num_last_ub),
                  return false);
  params->inner_loop_num_last = params->indices_row_num_last / params->row_num_last_ub;
  if (params->indices_row_num_last % params->row_num_last_ub != 0) {
    params->row_num_last_tail_ub = params->indices_row_num_last % params->row_num_last_ub;
  }
  if (params->inner_loop_num_last > 0 && params->row_num_last_tail_ub > 0 &&
      params->row_num_last_tail_ub * params->params_row < block_num) {
    params->inner_loop_num_last = params->inner_loop_num_last - 1;
    params->row_num_last_tail_ub = params->row_num_last_tail_ub + params->row_num_once_ub;
  }
  return true;
}

void CalNeedCore(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info) {
  while (params->need_core_num > 1) {
    params->need_core_num = params->need_core_num / 2;
    params->indices_num_each_core = params->indices_num / params->need_core_num;
    params->indices_num_remaining = params->indices_num % params->need_core_num;
    if (params->indices_num_each_core * params->params_row * compile_info->params_dsize > BLOCK_SIZE) {
      break;
    }
  }
}

// compute tiling params for tiling_mode 3&6&7
bool BlockAlignForIndicesTiling(GatherV2TilingParams* params, int64_t indices_num_per_loop, int64_t res_ub_size,
                                int64_t params_d_size) {
  if (indices_num_per_loop == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("gather_v2", "indices_num_per_loop = 0 is not support");
    return false;
  }
  params->indices_loop_num = (params->indices_num_each_core) / indices_num_per_loop;
  params->indices_row_num_once = indices_num_per_loop;
  if ((params->indices_num_each_core) % (params->indices_row_num_once) != 0) {
    params->indices_row_num_last = (params->indices_num_each_core) % (params->indices_row_num_once);
  }

  params->row_num_once_ub = res_ub_size / ((params->params_row) * params_d_size);
  OP_TILING_CHECK((params->row_num_once_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("Gather Tiling:", "Devide by row_num_once_ub[%ld] exception.",
                                                  params->row_num_once_ub),
                  return false);
  params->inner_loop_num = (params->indices_row_num_once) / (params->row_num_once_ub);
  if ((params->indices_row_num_once) % (params->row_num_once_ub) != 0) {
    params->row_num_once_tail_ub = (params->indices_row_num_once) % (params->row_num_once_ub);
  }

  params->row_num_last_ub = res_ub_size / ((params->params_row) * params_d_size);
  OP_TILING_CHECK((params->row_num_last_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("Gather Tiling:", "Devide by row_num_last_ub[%ld] exception.",
                                                  params->row_num_last_ub),
                  return false);
  params->inner_loop_num_last = (params->indices_row_num_last) / (params->row_num_last_ub);
  if ((params->indices_row_num_last) % params->row_num_last_ub != 0) {
    params->row_num_last_tail_ub = (params->indices_row_num_last) % (params->row_num_last_ub);
  }
  return true;
}

bool ParamsPreTiling(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info, int64_t half_ub_size,
                     int64_t half_remain_ub_size, int64_t params_total_ceil, int64_t params_row_ceil) {
  params->need_core_num = compile_info->core_num;
  params->tail_process_core = 0;
  params->params_pre_each_core = (params->params_pre) / (params->need_core_num);
  params->params_pre_remaining = (params->params_pre) % (params->need_core_num);
  params->indices_num_each_core = params->indices_num;
  int64_t half_remain_params_elem = half_remain_ub_size / (compile_info->params_dsize);
  int64_t res_ub_size = half_ub_size;
  int64_t half_ub_indices_elem = half_ub_size / (compile_info->indices_dsize);
  int64_t indices_num_per_loop = half_ub_indices_elem;
  int64_t block_num = BLOCK_SIZE / (compile_info->params_dsize);

  if ((params->indices_num_each_core) * (params->params_row) * (compile_info->params_dsize) <= BLOCK_SIZE) {
    params->need_core_num = 1;
    params->tail_process_core = 0;
    params->params_pre_each_core = params->params_pre;
    params->params_pre_remaining = 0;
  }

  if ((params->params_row) * (compile_info->params_dsize) < BLOCK_SIZE) {
    if (params_total_ceil <= PARAMS_CACHED_UB / (compile_info->params_dsize)) {
      params->tiling_mode = TILING_MODE_8;
    } else {
      params->tiling_mode = TILING_MODE_9;
    }

    if (params->tiling_mode == TILING_MODE_8) {
      indices_num_per_loop = half_remain_ub_size / (compile_info->indices_dsize);
      res_ub_size = half_remain_ub_size;
    }

    if (!BlockLessForParamsTiling(params, indices_num_per_loop, res_ub_size, compile_info->params_dsize, block_num)) {
      return false;
    }
  } else {
    if (params_total_ceil <= PARAMS_CACHED_UB / (compile_info->params_dsize) &&
        params_row_ceil <= half_remain_params_elem) {
      params->tiling_mode = TILING_MODE_10;
    } else if (params_total_ceil <= (compile_info->l1_size) / (compile_info->params_dsize)) {
      params->tiling_mode = TILING_MODE_11;
    } else {
      params->tiling_mode = TILING_MODE_12;
    }

    if (params->tiling_mode == TILING_MODE_10) {
      indices_num_per_loop = half_remain_ub_size / (compile_info->indices_dsize);
      res_ub_size = half_remain_ub_size;
    }

    if (!BlockAlignForParamsTiling(params, indices_num_per_loop, res_ub_size, compile_info->params_dsize)) {
      return false;
    }
  }
  return true;
}

bool ParamsSmall32B(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info, int64_t params_total_ceil,
                    int64_t indices_num_per_loop, int64_t half_remain_ub_size, int64_t res_ub_size, int64_t block_num) {
  if (params_total_ceil <= PARAMS_CACHED_UB / (compile_info->params_dsize)) {
    params->tiling_mode = TILING_MODE_4;
  } else if (params_total_ceil <= ((compile_info->l1_size) / (compile_info->params_dsize))) {
    params->tiling_mode = TILING_MODE_13;
  } else {
    params->tiling_mode = TILING_MODE_1;
  }
  if (((params->params_row) < BLOCK_SIZE) &&
      ((params->indices_num_each_core) * (params->params_row) * (compile_info->params_dsize) <= BLOCK_SIZE)) {
    CalNeedCore(params, compile_info);
  }

  if (params->tiling_mode == TILING_MODE_4) {
    indices_num_per_loop = half_remain_ub_size / (compile_info->indices_dsize);
    res_ub_size = half_remain_ub_size;
  }

  if (!BlockLessForIndicesTiling(params, indices_num_per_loop, res_ub_size, compile_info->params_dsize, block_num)) {
    OP_LOGE("GatherV2", "BlockLessForIndicesTiling is false");
    return false;
  }
  return true;
}

bool ParamsGreater32B(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info,
                      int64_t half_ub_params_elem, int64_t half_remain_ub_size, int64_t half_ub_size,
                      int64_t params_total_ceil, int64_t params_row_ceil) {
  int64_t half_ub_indices_elem = half_ub_size / (compile_info->indices_dsize);
  int64_t half_remain_params_elem = half_remain_ub_size / (compile_info->params_dsize);
  int64_t indices_num_per_loop = half_ub_indices_elem;
  int64_t res_ub_size = half_ub_size;
  int64_t block_num = BLOCK_SIZE / (compile_info->params_dsize);
  float mode_7_gate_value = ACTUAL_NUM - GATE_VALUE * params->params_total / DATA_VALUE;
  if (params_row_ceil <= half_ub_params_elem) {
    if ((params->params_row) * (compile_info->params_dsize) % BLOCK_SIZE != 0) {  // not 32B aligned
      params->tiling_mode = TILING_MODE_2;

      params->indices_loop_num = (params->indices_num_each_core) / half_ub_indices_elem;
      params->indices_row_num_once = half_ub_indices_elem;
      if ((params->indices_num_each_core) % (params->indices_row_num_once) != 0) {
        params->indices_row_num_last = (params->indices_num_each_core) % (params->indices_row_num_once);
      }
    } else {  // 32B aligned
      if (params_total_ceil <= PARAMS_CACHED_UB / (compile_info->params_dsize) &&
          params_row_ceil <= half_remain_params_elem) {
        params->tiling_mode = TILING_MODE_6;
      } else if (params_total_ceil <= (compile_info->l1_size) / (compile_info->params_dsize) &&
                 (params->indices_num) > mode_7_gate_value) {
        params->tiling_mode = TILING_MODE_7;
      } else {
        params->tiling_mode = TILING_MODE_3;
      }
      if (params->tiling_mode == TILING_MODE_6) {
        indices_num_per_loop = half_remain_ub_size / (compile_info->indices_dsize);
        res_ub_size = half_remain_ub_size;
      }

      if (!BlockAlignForIndicesTiling(params, indices_num_per_loop, res_ub_size, compile_info->params_dsize)) {
        return false;
      }
    }
  } else {
    params->tiling_mode = TILING_MODE_5;  // one params row need tiling

    params->indices_loop_num = params->indices_num_each_core / half_ub_indices_elem;
    params->indices_row_num_once = indices_num_per_loop;
    if ((params->indices_num_each_core) % (params->indices_row_num_once) != 0) {
      params->indices_row_num_last = (params->indices_num_each_core) % (params->indices_row_num_once);
    }

    params->one_row_loop = (params->params_row) / half_ub_params_elem;
    params->one_row_tail = (params->params_row) % half_ub_params_elem;
    if (params->one_row_loop > 0 && (params->one_row_tail) > 0 && (params->one_row_tail) < block_num) {
      params->one_row_loop = (params->one_row_loop) - 1;
      params->one_row_tail = half_ub_params_elem + (params->one_row_tail);
    }
  }
  return true;
}

bool ParamsIndicesTiling(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info, int64_t half_ub_size,
                         int64_t half_remain_ub_size, int64_t half_ub_params_elem, int64_t params_total_ceil,
                         int64_t params_row_ceil) {
  params->need_core_num = compile_info->core_num;
  params->tail_process_core = 0;
  params->indices_num_each_core = (params->indices_num) / (params->need_core_num);
  params->indices_num_remaining = (params->indices_num) % (params->need_core_num);
  int64_t half_ub_indices_elem = half_ub_size / (compile_info->indices_dsize);
  int64_t indices_num_per_loop = half_ub_indices_elem;
  int64_t res_ub_size = half_ub_size;
  int64_t block_num = BLOCK_SIZE / (compile_info->params_dsize);
  if (params->indices_num <= params->need_core_num) {
    params->need_core_num = params->indices_num;
    params->tail_process_core = 0;
    params->indices_num_each_core = 1;
    params->indices_num_remaining = 0;
  }

  // one params row size is smaller than 32B
  if ((params->params_row) * (compile_info->params_dsize) < BLOCK_SIZE) {
    if (!ParamsSmall32B(params, compile_info, params_total_ceil, indices_num_per_loop, half_remain_ub_size, res_ub_size,
                        block_num)) {
      OP_LOGE("GatherV2", "ParamsSmall32B is false");
      return false;
    }
  } else {  // one params row size is greater than or equal to 32B
    if (!ParamsGreater32B(params, compile_info, half_ub_params_elem, half_remain_ub_size, half_ub_size,
                          params_total_ceil, params_row_ceil)) {
      OP_LOGE("GatherV2", "ParamsGreater32B is false");
      return false;
    }
  }
  return true;
}

bool TilingWithoutBatchDims(gert::TilingContext* context, const GatherV2CompileInfo* compile_info,
                            GatherV2TilingParams* params, int64_t axis, int64_t params_dims, int64_t indices_dims) {
  int64_t available_ub_size = (compile_info->ub_size) - 2 * 1024;  // reserved 2K
  int64_t half_ub_size = available_ub_size / 2;
  // params shape convert to 3D:[params_pre, params_axis, params_row]  indies_shape.GetDimNum();
  // indices shape convert to 1D:[indices_num]
  // output tensor, y shape convert to:[params_pre, indices_num, params_row]
  auto x_shape = context->GetInputShape(0)->GetStorageShape();
  auto indies_shape = context->GetInputShape(1)->GetStorageShape();
  if (axis == 0) {
    params->params_pre = 1;
  } else {
    for (int64_t i = 0; i < axis; i++) {
      params->params_pre *= x_shape.GetDim(i);
    }
  }
  params->params_axis = x_shape.GetDim(axis);

  if (axis + 1 < params_dims) {
    for (int64_t i = axis + 1; i < params_dims; i++) {
      params->params_row *= x_shape.GetDim(i);
    }
  } else {
    params->params_row = 1;
  }
  params->params_total = GetPartShapeSize(x_shape, 0, params_dims);
  int64_t block_num = BLOCK_SIZE / (compile_info->params_dsize);
  int64_t params_total_ceil = ((params->params_total) + block_num - 1) / block_num * block_num;
  int64_t params_row_ceil = ((params->params_row) + block_num - 1) / block_num * block_num;
  for (int i = 0; i < indices_dims; i++) {
    params->indices_num = (params->indices_num) * indies_shape.GetDim(i);
  }

  int64_t half_remain_ub_size = (available_ub_size - PARAMS_CACHED_UB) / HALF_UB;
  int64_t half_ub_params_elem = half_ub_size / (compile_info->params_dsize);
  if (half_ub_params_elem == 0) {
    OP_LOGE("GatherV2", "half_ub_params_elem is 0");
    return false;
  }

  // the data of the formula gained from actual tests
  // set a gate value for tiling_mode_7 to optimized some data_move processes

  if (DoImplModeTiling(params, compile_info)) {
    OP_LOGD("GatherV2", "[GatherV2TIKTiling] end of tiling for impl_mode is high_performance");
    return true;
  }

  if (params->params_pre >= (compile_info->core_num) && params_row_ceil <= half_ub_params_elem &&
      ((params->params_row) * (compile_info->params_dsize) < BLOCK_SIZE ||
       (params->params_row) * (compile_info->params_dsize) % BLOCK_SIZE == 0)) {
    if (!ParamsPreTiling(params, compile_info, half_ub_size, half_remain_ub_size, params_total_ceil, params_row_ceil)) {
      OP_LOGE("GatherV2", "ParamsPreTiling is false");
      return false;
    }
  } else {
    if (!ParamsIndicesTiling(params, compile_info, half_ub_size, half_remain_ub_size, half_ub_params_elem,
                             params_total_ceil, params_row_ceil)) {
      OP_LOGE("GatherV2", "ParamsIndicesTiling is false");
      return false;
    }
  }
  return true;
}

bool LargeRowProcess(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info, int64_t half_ub_params_elem,
                     int64_t half_size_ub) {
  OP_TILING_CHECK(half_ub_params_elem == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("GatherV2:", "half_ub_params_elem = 0 is not support"), return false);
  params->one_row_loop = (params->params_row) / half_ub_params_elem;
  params->one_row_tail = (params->params_row) % half_ub_params_elem;
  int64_t block_num = BLOCK_SIZE / (compile_info->params_dsize);
  if ((params->one_row_loop) > 0 && (params->one_row_tail) > 0 && (params->one_row_tail) < block_num) {
    params->one_row_loop = (params->one_row_loop) - 1;
    params->one_row_tail = half_ub_params_elem + (params->one_row_tail);
  }

  if ((params->params_batch_each_core) * (params->indices_row) * (compile_info->indices_dsize) <= half_size_ub) {
    params->indices_row_num_once = params->indices_row;
    params->tiling_mode = TILING_MODE_35;
  } else if ((params->indices_row) * (compile_info->indices_dsize) <= half_size_ub) {
    params->indices_row_num_once = params->indices_row;
    params->tiling_mode = TILING_MODE_36;
  } else {
    int64_t indices_num_per_loop = half_size_ub / (compile_info->indices_dsize);
    params->indices_loop_num = (params->indices_row) / indices_num_per_loop;
    params->indices_row_num_once = indices_num_per_loop;
    if ((params->indices_row) % (params->indices_row_num_once) != 0) {
      params->indices_row_num_last = (params->indices_num_each_core) % (params->indices_row_num_once);
    }
    params->tiling_mode = TILING_MODE_37;
  }
  return true;
}

bool CalcCacheIndices(GatherV2TilingParams* params, int64_t indices_num_per_loop, int64_t res_ub_size,
                      int64_t params_d_size, int64_t tiling_mode) {
  OP_TILING_CHECK(params_d_size == 0, VECTOR_INNER_ERR_REPORT_TILIING("GatherV2:", "params_d_size= 0 is not support"),
                  return false);
  params->indices_row_num_once = indices_num_per_loop;
  params->row_num_once_ub = res_ub_size / ((params->params_row) * params_d_size);
  int64_t block_num = BLOCK_SIZE / params_d_size;
  int64_t align_unit;
  if (tiling_mode == TILING_MODE_38 || tiling_mode == TILING_MODE_39) {
    align_unit = params->indices_row * block_num;
  } else if (tiling_mode == TILING_MODE_40 || tiling_mode == TILING_MODE_41) {
    align_unit = (params->params_pre) * (params->indices_row) * block_num;
  } else if ((params->params_row) * params_d_size >= BLOCK_SIZE) {
    align_unit = 1;
  } else {
    align_unit = block_num;
  }

  if (int((params->row_num_once_ub) % align_unit) != 0) {
    params->row_num_once_ub = int((params->row_num_once_ub) / align_unit) * align_unit;
  }
  OP_TILING_CHECK((params->row_num_once_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("Gather Tiling:", "Devide by row_num_once_ub[%ld] exception.",
                                                  params->row_num_once_ub),
                  return false);
  params->inner_loop_num = (params->indices_row_num_once) / (params->row_num_once_ub);
  if ((params->indices_row_num_once) % (params->row_num_once_ub) != 0) {
    params->row_num_once_tail_ub = (params->indices_row_num_once) % (params->row_num_once_ub);
  }
  if ((params->inner_loop_num) > 0 && (params->row_num_once_tail_ub) > 0 &&
      (params->row_num_once_tail_ub) * (params->params_row) < block_num) {
    params->inner_loop_num = (params->inner_loop_num) - 1;
    params->row_num_once_tail_ub = (params->row_num_once_tail_ub) + (params->row_num_once_ub);
  }
  params->tiling_mode = tiling_mode;
  return true;
}

bool CalcWithBatchDims(GatherV2TilingParams* params, int64_t indices_num_per_loop, int64_t res_ub_size,
                       int64_t params_d_size) {
  if (indices_num_per_loop == 0 || params_d_size == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("gather_v2", "indices_num_per_loop or params_d_size= 0 is not support");
    return false;
  }
  params->indices_loop_num = (params->indices_row) / indices_num_per_loop;
  params->indices_row_num_once = indices_num_per_loop;
  int64_t block_num = BLOCK_SIZE / params_d_size;
  if ((params->params_row) * params_d_size >= BLOCK_SIZE) {
    block_num = 1;
  }
  if ((params->indices_row) % (params->indices_row_num_once) != 0) {
    params->indices_row_num_last = (params->indices_num_each_core) % (params->indices_row_num_once);
  }
  if ((params->indices_loop_num) > 0 &&
      (params->indices_row_num_last) * (params->indices_row) * (params->params_row) < block_num) {
    params->indices_loop_num -= 1;
    params->indices_row_num_last += params->indices_row_num_once;
  }

  params->row_num_once_ub = res_ub_size / ((params->params_row) * params_d_size);
  if (int((params->row_num_once_ub) % block_num) != 0) {
    params->row_num_once_ub = int((params->row_num_once_ub) / block_num) * block_num;
  }
  OP_TILING_CHECK((params->row_num_once_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("Gather Tiling:", "Devide by row_num_once_ub[%ld] exception.",
                                                  params->row_num_once_ub),
                  return false);
  params->inner_loop_num = (params->indices_row_num_once) / (params->row_num_once_ub);
  if ((params->indices_row_num_once) % (params->row_num_once_ub) != 0) {
    params->row_num_once_tail_ub = (params->indices_row_num_once) % (params->row_num_once_ub);
  }
  if ((params->inner_loop_num) > 0 && (params->row_num_once_tail_ub) > 0 &&
      (params->row_num_once_tail_ub) * (params->params_row) < block_num) {
    params->inner_loop_num = params->inner_loop_num - 1;
    params->row_num_once_tail_ub = params->row_num_once_tail_ub + params->row_num_once_ub;
  }

  params->row_num_last_ub = params->row_num_once_ub;
  params->inner_loop_num_last = (params->indices_row_num_last) / (params->row_num_once_ub);
  if ((params->indices_row_num_last) % (params->row_num_once_ub) != 0) {
    params->row_num_last_tail_ub = (params->indices_row_num_last) % (params->row_num_once_ub);
  }
  if ((params->inner_loop_num_last) > 0 && (params->row_num_last_tail_ub) > 0 &&
      (params->row_num_last_tail_ub) * (params->params_row) < block_num) {
    params->inner_loop_num_last = params->inner_loop_num_last - 1;
    params->row_num_last_tail_ub = params->row_num_last_tail_ub + params->row_num_once_ub;
  }
  return true;
}

bool IndicesCachedProcess(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info, int64_t aval_ub_size,
                          int64_t mode_cache_all, int64_t mode_cache_row, int64_t mode_without_cache) {
  int64_t indices_num_per_loop = 1;
  if (params->params_batch_each_core * params->indices_row * compile_info->indices_dsize <= aval_ub_size) {
    indices_num_per_loop = params->indices_row;
    if (!CalcCacheIndices(params, indices_num_per_loop, aval_ub_size, compile_info->params_dsize, mode_cache_all)) {
      return false;
    }
  } else if (params->indices_row * compile_info->indices_dsize <= aval_ub_size) {
    indices_num_per_loop = params->indices_row;
    if (!CalcCacheIndices(params, indices_num_per_loop, aval_ub_size, compile_info->params_dsize, mode_cache_row)) {
      return false;
    }
  } else {
    indices_num_per_loop = aval_ub_size / compile_info->indices_dsize;
    params->tiling_mode = mode_without_cache;
    if (!CalcWithBatchDims(params, indices_num_per_loop, aval_ub_size, compile_info->params_dsize)) {
      return false;
    }
  }
  return true;
}

bool SmallRowProcess(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info, int64_t mode_with_cache,
                     int64_t mode_without_cache, int64_t half_remain_size_ub, int64_t half_size_ub) {
  if (mode_with_cache == TILING_MODE_38 || mode_without_cache == TILING_MODE_39) {
    params->params_batch_each_core = params->params_pre / params->need_core_num;
    params->params_batch_remaining = params->params_pre % params->need_core_num;
  }
  params->tail_process_core = params->need_core_num - 1;
  params->indices_num_each_core = params->params_batch_each_core * params->indices_row;
  params->indices_num_remaining = 0;
  int64_t block_num = BLOCK_SIZE / compile_info->params_dsize;
  int64_t indices_num_per_loop = params->indices_num_each_core;
  int64_t params_total_ceil = (params->params_total + block_num - 1) / block_num * block_num;
  int64_t params_row_ceil = (params->params_row + block_num - 1) / block_num * block_num;
  int64_t half_remain_params_elem = half_remain_size_ub / compile_info->params_dsize;
  if (params_total_ceil <= PARAMS_CACHED_UB / compile_info->params_dsize &&
      params_row_ceil <= half_remain_params_elem) {
    if (!CalcCacheIndices(params, indices_num_per_loop, half_remain_size_ub, compile_info->params_dsize,
                          mode_with_cache)) {
      return false;
    }
  } else {
    if (!CalcCacheIndices(params, indices_num_per_loop, half_size_ub, compile_info->params_dsize, mode_without_cache)) {
      return false;
    }
  }
  return true;
}

void CalNeedCoreWithBatchDims(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info) {
  while (params->need_core_num > 1) {
    params->need_core_num = params->need_core_num / 2;
    params->params_batch_each_core = params->params_batch / params->need_core_num;
    params->params_batch_remaining = params->params_batch % params->need_core_num;
    params->indices_num_each_core = params->params_batch_each_core * params->indices_row;
    params->indices_num_remaining = params->params_batch_remaining * params->indices_row;
    if (params->indices_num_each_core * params->params_pre * params->params_row * compile_info->params_dsize >
        BLOCK_SIZE) {
      break;
    }
  }
}

void ParasPreProcess(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info,
                     gert::TilingContext* context, int64_t axis, int64_t batch_dims, int64_t& indices_batch) {
  auto x_shape = context->GetInputShape(0)->GetStorageShape();
  auto indices_shape = context->GetInputShape(1)->GetStorageShape();
  int64_t indices_dims = indices_shape.GetDimNum();

  // params shape convert to 4D:[params_batch, params_pre, params_axis, params_row]
  // indices shape convert to 1D:[indices_batch, indices_row]
  // output tensor, y shape convert to:[params_batch, params_pre, indices_row, params_row]
  for (int64_t i = 0; i < batch_dims; i++) {
    indices_batch = indices_batch * indices_shape.GetDim(i);
  }
  params->params_batch = indices_batch;
  for (int64_t i = batch_dims; i < indices_dims; i++) {
    params->indices_row = (params->indices_row) * indices_shape.GetDim(i);
  }

  if (axis == batch_dims) {
    params->params_pre = 1;
  } else {
    for (int64_t i = batch_dims; i < axis; i++) {
      params->params_pre = (params->params_pre) * x_shape.GetDim(i);
    }
  }
  params->params_axis = x_shape.GetDim(axis);
  int64_t params_dims = x_shape.GetDimNum();
  if (axis + 1 < params_dims) {
    for (int64_t i = axis + 1; i < params_dims; i++) {
      params->params_row = (params->params_row) * x_shape.GetDim(i);
    }
  } else {
    params->params_row = 1;
  }

  for (int64_t i = 0; i < indices_dims; i++) {
    params->indices_num = (params->indices_num) * indices_shape.GetDim(i);
  }
}

bool WithBatchDimsSmall(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info,
                        int64_t half_remain_params_elem, int64_t half_size_ub, int64_t params_total_ceil,
                        int64_t params_row_ceil) {
  int64_t available_ub_size = compile_info->ub_size - 2 * 1024;  // reserved 2K
  int64_t half_remain_size_ub = (available_ub_size - PARAMS_CACHED_UB) / HALF_UB;
  if ((params->indices_row) * (params->params_row) * (compile_info->params_dsize) <= BLOCK_SIZE) {
    if ((params->params_pre) * (params->indices_row) * (params->params_row) * (compile_info->params_dsize) <=
        NUM_32 * BLOCK_SIZE) {
      if ((params->indices_num_each_core) * (params->params_row) * (compile_info->params_dsize) <= BLOCK_SIZE) {
        CalNeedCoreWithBatchDims(params, compile_info);
      }
      params->params_total =
          (params->params_batch_each_core) * (params->params_pre) * (params->params_axis) * (params->params_row);
      if (!SmallRowProcess(params, compile_info, TILING_MODE_40, TILING_MODE_41, half_remain_size_ub, half_size_ub)) {
        return false;
      }
    } else {
      params->need_core_num =
          ((params->params_pre) < (compile_info->core_num)) ? (params->params_pre) : (compile_info->core_num);
      params->params_total =
          (params->params_batch) * (params->params_pre) * (params->params_axis) * (params->params_row);
      if (!SmallRowProcess(params, compile_info, TILING_MODE_38, TILING_MODE_39, half_remain_size_ub, half_size_ub)) {
        return false;
      }
    }
    return true;
  }
  if (params_total_ceil <= PARAMS_CACHED_UB / (compile_info->params_dsize) &&
      params_row_ceil <= half_remain_params_elem) {
    if (!IndicesCachedProcess(params, compile_info, half_remain_size_ub, TILING_MODE_20, TILING_MODE_21,
                              TILING_MODE_22)) {
      return false;
    }
  } else {
    if (!IndicesCachedProcess(params, compile_info, half_size_ub, TILING_MODE_23, TILING_MODE_24, TILING_MODE_25)) {
      return false;
    }
  }
  return true;
}

bool WithBatchDimsSmallCeil(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info, int64_t half_size_ub,
                            int64_t half_remain_size_ub, int64_t half_remain_params_elem, int64_t params_total_ceil,
                            int64_t params_row_ceil) {
  if ((params->params_row) * (compile_info->params_dsize) % BLOCK_SIZE != 0) {
    if (!IndicesCachedProcess(params, compile_info, half_size_ub, TILING_MODE_26, TILING_MODE_27, TILING_MODE_28)) {
      return false;
    }
  } else {
    if (params_total_ceil <= PARAMS_CACHED_UB / (compile_info->params_dsize) &&
        params_row_ceil <= half_remain_params_elem) {
      if (!IndicesCachedProcess(params, compile_info, half_remain_size_ub, TILING_MODE_29, TILING_MODE_30,
                                TILING_MODE_31)) {
        return false;
      }
    } else {
      if (!IndicesCachedProcess(params, compile_info, half_size_ub, TILING_MODE_32, TILING_MODE_33, TILING_MODE_34)) {
        return false;
      }
    }
  }
  return true;
}

bool WithBatchDimsBig(GatherV2TilingParams* params, const GatherV2CompileInfo* compile_info, int64_t params_row_ceil,
                      int64_t half_size_ub, int64_t half_ub_params_elem, int64_t params_total_ceil,
                      int64_t half_remain_params_elem) {
  int64_t available_ub_size = compile_info->ub_size - 2 * 1024;  // reserved 2K
  int64_t half_remain_size_ub = (available_ub_size - PARAMS_CACHED_UB) / HALF_UB;
  if (params_row_ceil <= half_ub_params_elem) {
    if (!WithBatchDimsSmallCeil(params, compile_info, half_size_ub, half_remain_size_ub, half_remain_params_elem,
                                params_total_ceil, params_row_ceil)) {
      return false;
    }
  } else {
    if (!LargeRowProcess(params, compile_info, half_ub_params_elem, half_size_ub)) {
      return false;
    }
  }
  return true;
}

bool TilingWithBatchDims(gert::TilingContext* context, GatherV2TilingParams* params,
                         const GatherV2CompileInfo* compile_info, int64_t axis, int64_t batch_dims) {
  int64_t available_ub_size = compile_info->ub_size - 2 * 1024;  // reserved 2K
  int64_t half_size_ub = available_ub_size / 2;
  int64_t block_num = BLOCK_SIZE / compile_info->params_dsize;
  int64_t indices_batch = 1;
  int64_t half_remain_size_ub = 1;
  ParasPreProcess(params, compile_info, context, axis, batch_dims, indices_batch);

  half_remain_size_ub = (available_ub_size - PARAMS_CACHED_UB) / HALF_UB;
  int64_t half_remain_params_elem = half_remain_size_ub / (compile_info->params_dsize);
  int64_t half_ub_params_elem = half_size_ub / compile_info->params_dsize;
  params->need_core_num = (indices_batch < compile_info->core_num) ? indices_batch : compile_info->core_num;
  params->tail_process_core = 0;
  params->params_batch_each_core = (params->params_batch) / (params->need_core_num);
  params->params_batch_remaining = (params->params_batch) % (params->need_core_num);
  params->indices_num_each_core = (params->params_batch_each_core) * (params->indices_row);
  params->indices_num_remaining = (params->params_batch_remaining) * (params->indices_row);

  if ((params->indices_num_each_core) * (params->params_row) * (compile_info->params_dsize) <= BLOCK_SIZE) {
    params->need_core_num = 1;
    params->tail_process_core = 0;
    params->params_batch_each_core = params->params_batch;
    params->params_batch_remaining = 0;
    params->indices_num_each_core = (params->params_batch_each_core) * (params->indices_row);
    params->indices_num_remaining = (params->params_batch_remaining) * (params->indices_row);
  }
  params->params_total =
      (params->params_batch_each_core) * (params->params_pre) * (params->params_axis) * (params->params_row);
  int64_t params_total_ceil = ((params->params_total) + block_num - 1) / block_num * block_num;
  int64_t params_row_ceil = ((params->params_row) + block_num - 1) / block_num * block_num;

  if ((params->params_row) * (compile_info->params_dsize) < BLOCK_SIZE) {
    if (!WithBatchDimsSmall(params, compile_info, half_remain_params_elem, half_size_ub, params_total_ceil,
                            params_row_ceil)) {
      OP_LOGE("GatherV2", "WithBatchDimsSmall is false");
      return false;
    }
  } else {
    if (!WithBatchDimsBig(params, compile_info, params_row_ceil, half_size_ub, half_ub_params_elem, params_total_ceil,
                          half_remain_params_elem)) {
      OP_LOGE("GatherV2", "WithBatchDimsBig is false");
      return false;
    }
  }
  return true;
}

void InitGatherCompileParams(GatherV2TilingParams* params) {
  params->params_pre = 1;
  params->params_axis = 1;
  params->params_row = 1;
  params->indices_num = 1;
  params->indices_row = 1;
  params->params_batch_each_core = 1;
  params->params_batch = 1;
}

bool GatherTIKTiling(gert::TilingContext* context, const GatherV2CompileInfo* compile_info) {
  OP_LOGD(context->GetNodeName(), "GatherTIKTiling running");
  auto indeces_tensor = context->GetInputDesc(1);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, indeces_tensor, false);
  auto x_shape = context->GetInputShape(0)->GetStorageShape();
  auto indies_shape = context->GetInputShape(1)->GetStorageShape();
  int64_t x_dim = x_shape.GetDimNum();
  int64_t indices_dim = indies_shape.GetDimNum();
  int64_t batch_dims_idx = 1;
  int64_t axis = 0;
  if (compile_info->is_gather_v2) {
    OP_LOGD(context->GetNodeName(), "optype is gatherv2");
    OP_TILING_CHECK(!(ops::GetConstInt(context, INPUT_IDX_AXIS, axis)),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get const data axis failed"),
                    return false);
    batch_dims_idx = 0;
  }

  auto* attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, attrs, false);
  const auto* batchdims = attrs->GetAttrPointer<int64_t>(batch_dims_idx);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, batchdims, false);
  int64_t batch_dims = *batchdims;

  if (!(compile_info->is_gather_v2) && batch_dims != 0) {
    OP_LOGD(context->GetNodeName(), "optype is gather and batch_dims != 0");
    axis = batch_dims;
  }
  OP_LOGD(context->GetNodeName(), "axis is %d, batch_dims is %d", axis, batch_dims);
  if (!CheckAndUpdateAxisAndBatchdims(axis, x_shape, indies_shape, batch_dims, x_dim, indices_dim)) {
    VECTOR_INNER_ERR_REPORT_TILIING("GatherV2", "op GatherV2Tiling: [CheckAndUpdateAxisAndBatchdims] failed.");
    return false;
  }

  auto params = context->GetTilingData<GatherV2TilingParams>();
  InitGatherCompileParams(params);

  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, params, false);
  if (batch_dims == 0) {
    if (!TilingWithoutBatchDims(context, compile_info, params, axis, x_dim, indices_dim)) {
      VECTOR_INNER_ERR_REPORT_TILIING("GatherV2", "op GatherV2Tiling: [TilingWithoutBatchDims] failed.");
      return false;
    }
  } else {
    if (!TilingWithBatchDims(context, params, compile_info, axis, batch_dims)) {
      VECTOR_INNER_ERR_REPORT_TILIING("GatherV2", "op GatherV2Tiling: [TilingWithBatchDims] failed.");
      return false;
    }
  }
  // block_dim, core num used in tik op
  context->SetBlockDim(params->need_core_num);
  OP_LOGD(context->GetNodeName(), "GatherTIKTiling run success.");
  return true;
}

bool GatherDSLTiling(gert::TilingContext* context, const GatherV2CompileInfo* compile_info) {
  OP_LOGD(context->GetNodeName(), "GatherDSLTiling running");
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, compile_info->dsl_compile_info, false);
  OpInfo gatherv2_info(compile_info->dsl_compile_info.get());
  OP_TILING_CHECK(!DoAutoTiling(context, &gatherv2_info),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "call DoAutoTiling failed"), return false);
  OP_LOGD("gatherv2", "GatherDSLTiling end.");
  return true;
}

ge::graphStatus GatherTiling(gert::TilingContext* context) {
  OP_LOGD(context->GetNodeName(), "GatherTiling running begin");
  auto compile_info = reinterpret_cast<const GatherV2CompileInfo*>(context->GetCompileInfo());
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);

  if (compile_info->is_tik) {
    OP_TILING_CHECK(!GatherTIKTiling(context, compile_info),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "call TIKTiling failed"),
                    return ge::GRAPH_FAILED);
  } else {
    OP_TILING_CHECK(!GatherDSLTiling(context, compile_info),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "call DSLTiling failed"),
                    return ge::GRAPH_FAILED);
  }
  OP_LOGD(context->GetNodeName(), "GatherTiling running end");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForGatherV2(gert::TilingParseContext* context) {
  OP_LOGD(context->GetNodeName(), "TilingPrepareForGatherV2 running.");
  auto compile_info = MutableCompileInfo<GatherV2CompileInfo>(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  std::unique_ptr<nlohmann::json> parsed_object_cinfo = GetJsonObj(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, parsed_object_cinfo);
  if (GetCompileValue(*parsed_object_cinfo, "is_tik", compile_info->is_tik)) {
    const nlohmann::json& all_vars = (*parsed_object_cinfo)["vars"];
    OP_TILING_CHECK(
        !GetCompileValue(all_vars, "core_num", compile_info->core_num),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "TilingPrepareForGatherV2, get core_num error"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(compile_info->core_num < 1,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "GatherParseFunc, core_num should be greater than 0"),
        return false);
    OP_TILING_CHECK(
        !GetCompileValue(all_vars, "ub_size", compile_info->ub_size),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "TilingPrepareForGatherV2, get ub_size error"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !GetCompileValue(all_vars, "l1_size", compile_info->l1_size),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "TilingPrepareForGatherV2, get l1_size error"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !GetCompileValue(all_vars, "params_dsize", compile_info->params_dsize),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "TilingPrepareForGatherV2, get params_dsize error"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !GetCompileValue(all_vars, "indices_dsize", compile_info->indices_dsize),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "TilingPrepareForGatherV2, get indices_dsize error"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!GetCompileValue(all_vars, "impl_mode", compile_info->impl_mode, 0),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "GatherParseFunc, get impl_mode error"),
        return false);
    OP_TILING_CHECK(
        !GetCompileValue(*parsed_object_cinfo, "is_gather_v2", compile_info->is_gather_v2),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "TilingPrepareForGatherV2, get is_gather_v2 error"),
        return ge::GRAPH_FAILED);
  } else {
    OP_LOGD(context->GetNodeName(), "will use gather AotoTiling");
    compile_info->dsl_compile_info = ParseAutoTiling("GatherV2", *parsed_object_cinfo);
    OP_TILING_CHECK(compile_info->dsl_compile_info == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "CreateGatherTilingHandler failed"),
                    return ge::GRAPH_FAILED);

    compile_info->is_tik = false;
  }
  OP_LOGD(context->GetNodeName(), "TilingPrepareForGatherV2 GRAPH_SUCCESS.");
  return ge::GRAPH_SUCCESS;
}

// register tiling interface of the GatherV2 and Gather op.
IMPL_OP(GatherV2).Tiling(GatherTiling).TilingParse<GatherV2CompileInfo>(TilingPrepareForGatherV2);
}  // namespace optiling
