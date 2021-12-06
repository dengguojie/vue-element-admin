/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \brief tiling function of op
 */
#include <string>

#include <nlohmann/json.hpp>
#include "op_tiling_util.h"
#include "graph/debug/ge_log.h"

#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "error_log.h"
#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"

namespace {
  constexpr int32_t HALF_UB = 2;
  constexpr int32_t DATA_VALUE = 1024;
  constexpr int32_t NUM_32 = 32;
}

namespace optiling {
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

struct GatherCompileParams {
  int64_t ub_size;
  int64_t l1_size;
  int64_t core_num;
  int64_t params_d_size;
  int64_t indices_d_size;
  int64_t batch_dims;
};

struct GatherShapeInfo {
  std::vector<int64_t> params_shape;
  std::vector<int64_t> indices_shape;
  std::vector<int64_t> indices_ori_shape;
  std::vector<int64_t> y_shape;
};

struct GatherV2TilingParams {
  int64_t tiling_mode;
  int64_t params_pre;
  int64_t params_axis;
  int64_t params_row;
  int64_t indices_num;
  int64_t cache_params;
  int64_t need_core_num;
  int64_t tail_process_core;
  int64_t indices_num_each_core;
  int64_t indices_num_remaining;
  int64_t indices_loop_num;
  int64_t indices_row_num_once;
  int64_t indices_row_num_last;
  int64_t row_num_once_ub;
  int64_t row_num_once_tail_ub;
  int64_t inner_loop_num;
  int64_t row_num_last_ub;
  int64_t row_num_last_tail_ub;
  int64_t inner_loop_num_last;
  int64_t params_total;
  int64_t one_row_loop;
  int64_t one_row_tail;
  int64_t params_pre_each_core;
  int64_t params_pre_remaining;
  int64_t indices_row;
  int64_t params_batch_each_core;
  int64_t params_batch_remaining;
  int64_t params_batch;
  int64_t indicesBatch;
  int64_t half_remain_ub_size;
  int64_t half_ub_size;
};

void InitGatherCompileParams(GatherCompileParams& params) {
  params.ub_size = 0;
  params.l1_size = 0;
  params.core_num = 0;
  params.params_d_size = 0;
  params.indices_d_size = 0;
  params.batch_dims = 0;
}

void InitGatherShapeInfo(const std::string& op_type, GatherShapeInfo& params, const ge::Operator& op_paras) {
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op_paras);
  if (operator_info == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get operator_info error.");
    return;
  }
  auto opdesc = operator_info->MutableInputDesc(0);
  if (opdesc == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input shape error.");
    return;
  }
  auto opdesc_1 = operator_info->MutableInputDesc(1);
  if (opdesc_1 == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_1 shape error.");
    return;
  }
  auto outdesc = operator_info->MutableOutputDesc(0);
  if (outdesc == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get output shape error.");
    return;
  }
  params.params_shape = opdesc->GetShape().GetDims();
  params.indices_shape = opdesc_1->GetShape().GetDims();
  params.indices_ori_shape = opdesc_1->GetOriginShape().GetDims();
  params.y_shape = outdesc->MutableShape().GetDims();
}

void InitGatherV2Params(GatherV2TilingParams& params) {
  params.tiling_mode = 0;
  params.params_pre = 1;
  params.params_axis = 1;
  params.params_row = 1;
  params.indices_num = 1;
  params.cache_params = 0;
  params.need_core_num = 0;
  params.tail_process_core = 0;
  params.indices_num_each_core = 0;
  params.indices_num_remaining = 0;
  params.indices_loop_num = 0;
  params.indices_row_num_once = 0;
  params.indices_row_num_last = 0;
  params.row_num_once_ub = 0;
  params.row_num_once_tail_ub = 0;
  params.inner_loop_num = 0;
  params.row_num_last_ub = 0;
  params.row_num_last_tail_ub = 0;
  params.inner_loop_num_last = 0;
  params.params_total = 0;
  params.one_row_loop = 0;
  params.one_row_tail = 0;
  params.params_pre_each_core = 0;
  params.params_pre_remaining = 0;
  params.indices_row = 1;
  params.params_batch_each_core = 1;
  params.params_batch_remaining = 0;
  params.params_batch = 1;
  params.indicesBatch = 1;
  params.half_remain_ub_size = 1;
  params.half_ub_size = 1;
}

void SetGatherV2Params(GatherV2TilingParams& Params, utils::OpRunInfo& run_info) {
  // set tiling data
  run_info.AddTilingData(Params.tiling_mode);
  run_info.AddTilingData(Params.params_pre);
  run_info.AddTilingData(Params.params_axis);
  run_info.AddTilingData(Params.params_row);
  run_info.AddTilingData(Params.indices_num);
  run_info.AddTilingData(Params.cache_params);
  run_info.AddTilingData(Params.need_core_num);
  run_info.AddTilingData(Params.tail_process_core);
  run_info.AddTilingData(Params.indices_num_each_core);
  run_info.AddTilingData(Params.indices_num_remaining);
  run_info.AddTilingData(Params.indices_loop_num);
  run_info.AddTilingData(Params.indices_row_num_once);
  run_info.AddTilingData(Params.indices_row_num_last);
  run_info.AddTilingData(Params.row_num_once_ub);
  run_info.AddTilingData(Params.row_num_once_tail_ub);
  run_info.AddTilingData(Params.inner_loop_num);
  run_info.AddTilingData(Params.row_num_last_ub);
  run_info.AddTilingData(Params.row_num_last_tail_ub);
  run_info.AddTilingData(Params.inner_loop_num_last);
  run_info.AddTilingData(Params.params_total);
  run_info.AddTilingData(Params.one_row_loop);
  run_info.AddTilingData(Params.one_row_tail);
  run_info.AddTilingData(Params.params_pre_each_core);
  run_info.AddTilingData(Params.params_pre_remaining);
  run_info.AddTilingData(Params.indices_row);
  run_info.AddTilingData(Params.params_batch_each_core);
  run_info.AddTilingData(Params.params_batch_remaining);
  run_info.AddTilingData(Params.params_batch);
}

void PrintGatherV2Params(const GatherV2TilingParams& params, const std::string& op_type) {
  OP_LOGD(op_type.c_str(), "tiling_mode=%ld.", params.tiling_mode);
  OP_LOGD(op_type.c_str(), "params_pre=%ld.", params.params_pre);
  OP_LOGD(op_type.c_str(), "params_axis=%ld.", params.params_axis);
  OP_LOGD(op_type.c_str(), "params_row=%ld.", params.params_row);
  OP_LOGD(op_type.c_str(), "indices_num=%ld.", params.indices_num);
  OP_LOGD(op_type.c_str(), "cache_params=%ld.", params.cache_params);
  OP_LOGD(op_type.c_str(), "need_core_num=%ld.", params.need_core_num);
  OP_LOGD(op_type.c_str(), "tail_process_core=%ld.", params.tail_process_core);
  OP_LOGD(op_type.c_str(), "indices_num_each_core=%ld.", params.indices_num_each_core);
  OP_LOGD(op_type.c_str(), "indices_num_remaining=%ld.", params.indices_num_remaining);
  OP_LOGD(op_type.c_str(), "indices_loop_num=%ld.", params.indices_loop_num);
  OP_LOGD(op_type.c_str(), "indices_row_num_once=%ld.", params.indices_row_num_once);
  OP_LOGD(op_type.c_str(), "indices_row_num_last=%ld.", params.indices_row_num_last);
  OP_LOGD(op_type.c_str(), "row_num_once_ub=%ld.", params.row_num_once_ub);
  OP_LOGD(op_type.c_str(), "row_num_once_tail_ub=%ld.", params.row_num_once_tail_ub);
  OP_LOGD(op_type.c_str(), "inner_loop_num=%ld.", params.inner_loop_num);
  OP_LOGD(op_type.c_str(), "row_num_last_ub=%d.", params.row_num_last_ub);
  OP_LOGD(op_type.c_str(), "row_num_last_tail_ub=%ld.", params.row_num_last_tail_ub);
  OP_LOGD(op_type.c_str(), "inner_loop_num_last=%ld.", params.inner_loop_num_last);
  OP_LOGD(op_type.c_str(), "params_total=%ld.", params.params_total);
  OP_LOGD(op_type.c_str(), "one_row_loop=%ld.", params.one_row_loop);
  OP_LOGD(op_type.c_str(), "one_row_tail=%ld.", params.one_row_tail);
  OP_LOGD(op_type.c_str(), "params_pre_each_core=%ld.", params.params_pre_each_core);
  OP_LOGD(op_type.c_str(), "params_pre_remaining=%ld.", params.params_pre_remaining);
  OP_LOGD(op_type.c_str(), "indices_row=%ld.", params.indices_row);
  OP_LOGD(op_type.c_str(), "params_batch_each_core=%ld.", params.params_batch_each_core);
  OP_LOGD(op_type.c_str(), "params_batch_remaining=%ld.", params.params_batch_remaining);
  OP_LOGD(op_type.c_str(), "params_batch=%ld.", params.params_batch);
}

bool GetAxis(const std::string& op_type, const ge::Operator& op_paras, int64_t& axis) {
  std::vector<int64_t> values;
  // input axis index is 2
  if (!ops::GetConstIntData(op_paras, 2, values)) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "axis not exists.");
    return false;
  }
  if (values.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "values is empty.");
    return false;
  }
  axis = values[0];
  OP_LOGD(op_type.c_str(), "axis=%ld.", axis);

  return true;
}

bool CheckAxisAndBatchdims(const std::string& op_type, const GatherShapeInfo& shape_info, int64_t& axis,
                           GatherCompileParams& compile_params) {
  int64_t paramsDims = shape_info.params_shape.size();
  int64_t indices_dims = shape_info.indices_shape.size();
  if (paramsDims <= 0 || indices_dims <= 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GatherV2Tiling: paramsDims or indices_dims is 0.");
    return false;
  }
  if (axis < -paramsDims || axis >= paramsDims) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op GatherV2Tiling: axis is invalid.");
    return false;
  }
  if (axis < 0) {
    axis += paramsDims;
  }

  if (compile_params.batch_dims != 0) {
    if (compile_params.batch_dims < -indices_dims || compile_params.batch_dims >= indices_dims) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op GatherV2Tiling: batch_dims is invalid.");
      return false;
    }
    if (compile_params.batch_dims < 0) {
      compile_params.batch_dims += indices_dims;
    }
    if (compile_params.batch_dims >= paramsDims) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op GatherV2Tiling: batch_dims must be less than rank(params).");
      return false;
    }
    if (compile_params.batch_dims > axis) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op GatherV2Tiling: batch_dims must be less than or equal to axis.");
      return false;
    }
    for (int i = 0; i < compile_params.batch_dims; i++) {
      if (shape_info.params_shape[i] != shape_info.indices_shape[i]) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type,
                                        "op GatherV2Tiling: Params.shape[:batch_dims] "
                                        "should be equal to indices.shape[:batch_dims].");
        return false;
      }
    }
  }

  return true;
}

bool CheckTensorShape(const std::string& op_type, const GatherShapeInfo& shape_info, int64_t axis, int64_t batch_dims) {
  int64_t paramsDims = shape_info.params_shape.size();
  int64_t indices_dims = shape_info.indices_shape.size();
  int64_t indicesOriDims = shape_info.indices_ori_shape.size();

  std::vector<int64_t> outputShape;

  if (axis < 0) {
    axis += paramsDims;
  }

  if (axis > 0) {
    for (int64_t i = 0; i < axis; i++) {
      outputShape.push_back(shape_info.params_shape[i]);
    }
  }
  if (indicesOriDims > 0) {
    for (int64_t i = batch_dims; i < indices_dims; i++) {
      outputShape.push_back(shape_info.indices_shape[i]);
    }
  }
  if (axis + 1 < paramsDims) {
    for (int64_t i = axis + 1; i < paramsDims; i++) {
      outputShape.push_back(shape_info.params_shape[i]);
    }
  }

  if (outputShape != shape_info.y_shape) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op [GatherV2Tiling] : output shape is invalid.");
  }

  return true;
}

bool GetV2GatherCompileParams(const std::string& op_type, const std::vector<int64_t>& compile_info_vec,
                              GatherCompileParams& params) {
  OP_TILING_CHECK(
      compile_info_vec.size() != 6,
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "the compile info num is not 6, is %zu", compile_info_vec.size()),
      return false);

  params.core_num = compile_info_vec[0];
  params.ub_size = compile_info_vec[1];
  params.l1_size = compile_info_vec[2];
  params.params_d_size = compile_info_vec[3];
  params.indices_d_size = compile_info_vec[4];
  params.batch_dims = compile_info_vec[5];
  return true;
}

// compute tiling params for tiling_mode 8&9
bool BlockLessForParamsTiling(GatherV2TilingParams& run_params, int64_t indices_num_per_loop, int64_t res_ub_size,
                              int64_t params_d_size, int64_t block_num) {
  if(indices_num_per_loop == 0 || block_num == 0){
      VECTOR_INNER_ERR_REPORT_TILIING("gather_v2", "indices_num_per_loop or block_num = 0 is not support");
      return false;
  }
  run_params.indices_loop_num = run_params.indices_num_each_core / indices_num_per_loop;
  run_params.indices_row_num_once = indices_num_per_loop;
  if (run_params.indices_num_each_core % run_params.indices_row_num_once != 0) {
    run_params.indices_row_num_last = run_params.indices_num_each_core % run_params.indices_row_num_once;
  }

  run_params.row_num_once_ub = res_ub_size / (run_params.params_row * params_d_size);
  if (int(run_params.row_num_once_ub % block_num) != 0) {
    run_params.row_num_once_ub = int(run_params.row_num_once_ub / block_num) * block_num;
  }
  OP_TILING_CHECK((run_params.row_num_once_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("Gather Tiling:", "Devide by row_num_once_ub[%ld] exception.",
                                                  run_params.row_num_once_ub),
                  return false);
  run_params.inner_loop_num = run_params.indices_row_num_once / run_params.row_num_once_ub;
  if (run_params.indices_row_num_once % run_params.row_num_once_ub != 0) {
    run_params.row_num_once_tail_ub = run_params.indices_row_num_once % run_params.row_num_once_ub;
  }
  if (run_params.inner_loop_num > 0 && run_params.row_num_once_tail_ub > 0 &&
      run_params.row_num_once_tail_ub * run_params.params_row < block_num) {
    run_params.inner_loop_num = run_params.inner_loop_num - 1;
    run_params.row_num_once_tail_ub = run_params.row_num_once_tail_ub + run_params.row_num_once_ub;
  }

  run_params.row_num_last_ub = res_ub_size / (run_params.params_row * params_d_size);
  if (int(run_params.row_num_last_ub % block_num) != 0) {
    run_params.row_num_last_ub = int(run_params.row_num_last_ub / block_num) * block_num;
  }
  OP_TILING_CHECK((run_params.row_num_last_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("Gather Tiling:", "Devide by row_num_last_ub[%ld] exception.",
                                                  run_params.row_num_last_ub),
                  return false);
  run_params.inner_loop_num_last = run_params.indices_row_num_last / run_params.row_num_last_ub;
  if (run_params.indices_row_num_last % run_params.row_num_last_ub != 0) {
    run_params.row_num_last_tail_ub = run_params.indices_row_num_last % run_params.row_num_last_ub;
  }
  if (run_params.inner_loop_num_last > 0 && run_params.row_num_last_tail_ub > 0 &&
      run_params.row_num_last_tail_ub * run_params.params_row < block_num) {
    run_params.inner_loop_num_last = run_params.inner_loop_num_last - 1;
    run_params.row_num_last_tail_ub = run_params.row_num_last_tail_ub + run_params.row_num_once_ub;
  }

  return true;
}

// compute tiling params for tiling_mode 10&11&12
bool BlockAlignForParamsTiling(GatherV2TilingParams& run_params, int64_t indices_num_per_loop, int64_t res_ub_size,
                               int64_t params_d_size) {
  if(indices_num_per_loop == 0){
    VECTOR_INNER_ERR_REPORT_TILIING("gather_v2", "indices_num_per_loop = 0 is not support");
    return false;
  }
  run_params.indices_loop_num = run_params.indices_num_each_core / indices_num_per_loop;
  run_params.indices_row_num_once = indices_num_per_loop;
  if (run_params.indices_num_each_core % run_params.indices_row_num_once != 0) {
    run_params.indices_row_num_last = run_params.indices_num_each_core % run_params.indices_row_num_once;
  }

  run_params.row_num_once_ub = res_ub_size / (run_params.params_row * params_d_size);
  OP_TILING_CHECK((run_params.row_num_once_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("Gather Tiling:", "Devide by row_num_once_ub[%ld] exception.",
                                                  run_params.row_num_once_ub),
                  return false);
  run_params.inner_loop_num = run_params.indices_row_num_once / run_params.row_num_once_ub;
  if (run_params.indices_row_num_once % run_params.row_num_once_ub != 0) {
    run_params.row_num_once_tail_ub = run_params.indices_row_num_once % run_params.row_num_once_ub;
  }

  run_params.row_num_last_ub = res_ub_size / (run_params.params_row * params_d_size);
  OP_TILING_CHECK((run_params.row_num_last_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("Gather Tiling:", "Devide by row_num_last_ub[%ld] exception.",
                                                  run_params.row_num_last_ub),
                  return false);
  run_params.inner_loop_num_last = run_params.indices_row_num_last / run_params.row_num_last_ub;
  if (run_params.indices_row_num_last % run_params.row_num_last_ub != 0) {
    run_params.row_num_last_tail_ub = run_params.indices_row_num_last % run_params.row_num_last_ub;
  }

  return true;
}

// compute tiling params for tiling_mode 1&4&13
bool BlockLessForIndicesTiling(GatherV2TilingParams& run_params, int64_t indices_num_per_loop, int64_t res_ub_size,
                               int64_t params_d_size, int64_t block_num) {
  if(indices_num_per_loop == 0 || block_num == 0){
    VECTOR_INNER_ERR_REPORT_TILIING("gather_v2", "indices_num_per_loop or block_num = 0 is not support");
    return false;
  }
  run_params.indices_loop_num = run_params.indices_num_each_core / indices_num_per_loop;
  run_params.indices_row_num_once = indices_num_per_loop;
  if (run_params.indices_num_each_core % run_params.indices_row_num_once != 0) {
    run_params.indices_row_num_last = run_params.indices_num_each_core % run_params.indices_row_num_once;
  }

  run_params.row_num_once_ub = res_ub_size / (run_params.params_row * params_d_size);
  if (int(run_params.row_num_once_ub % block_num) != 0) {
    run_params.row_num_once_ub = int(run_params.row_num_once_ub / block_num) * block_num;
  }
  OP_TILING_CHECK((run_params.row_num_once_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("Gather Tiling:", "Devide by row_num_once_ub[%ld] exception.",
                                                  run_params.row_num_once_ub),
                  return false);
  run_params.inner_loop_num = run_params.indices_row_num_once / run_params.row_num_once_ub;
  if (run_params.indices_row_num_once % run_params.row_num_once_ub != 0) {
    run_params.row_num_once_tail_ub = run_params.indices_row_num_once % run_params.row_num_once_ub;
  }
  if (run_params.inner_loop_num > 0 && run_params.row_num_once_tail_ub > 0 &&
      run_params.row_num_once_tail_ub * run_params.params_row < block_num) {
    run_params.inner_loop_num = run_params.inner_loop_num - 1;
    run_params.row_num_once_tail_ub = run_params.row_num_once_tail_ub + run_params.row_num_once_ub;
  }

  run_params.row_num_last_ub = res_ub_size / (run_params.params_row * params_d_size);
  if (int(run_params.row_num_last_ub % block_num) != 0) {
    run_params.row_num_last_ub = int(run_params.row_num_last_ub / block_num) * block_num;
  }
  OP_TILING_CHECK((run_params.row_num_last_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("Gather Tiling:", "Devide by row_num_last_ub[%ld] exception.",
                                                  run_params.row_num_last_ub),
                  return false);
  run_params.inner_loop_num_last = run_params.indices_row_num_last / run_params.row_num_last_ub;
  if (run_params.indices_row_num_last % run_params.row_num_last_ub != 0) {
    run_params.row_num_last_tail_ub = run_params.indices_row_num_last % run_params.row_num_last_ub;
  }
  if (run_params.inner_loop_num_last > 0 && run_params.row_num_last_tail_ub > 0 &&
      run_params.row_num_last_tail_ub * run_params.params_row < block_num) {
    run_params.inner_loop_num_last = run_params.inner_loop_num_last - 1;
    run_params.row_num_last_tail_ub = run_params.row_num_last_tail_ub + run_params.row_num_once_ub;
  }

  return true;
}

// compute tiling params for tiling_mode 3&6&7
bool BlockAlignForIndicesTiling(GatherV2TilingParams& run_params, int64_t indices_num_per_loop, int64_t res_ub_size,
                                int64_t params_d_size) {
  if(indices_num_per_loop == 0){
    VECTOR_INNER_ERR_REPORT_TILIING("gather_v2", "indices_num_per_loop = 0 is not support");
    return false;
  }
  run_params.indices_loop_num = run_params.indices_num_each_core / indices_num_per_loop;
  run_params.indices_row_num_once = indices_num_per_loop;
  if (run_params.indices_num_each_core % run_params.indices_row_num_once != 0) {
    run_params.indices_row_num_last = run_params.indices_num_each_core % run_params.indices_row_num_once;
  }

  run_params.row_num_once_ub = res_ub_size / (run_params.params_row * params_d_size);
  OP_TILING_CHECK((run_params.row_num_once_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("Gather Tiling:", "Devide by row_num_once_ub[%ld] exception.",
                                                  run_params.row_num_once_ub),
                  return false);
  run_params.inner_loop_num = run_params.indices_row_num_once / run_params.row_num_once_ub;
  if (run_params.indices_row_num_once % run_params.row_num_once_ub != 0) {
    run_params.row_num_once_tail_ub = run_params.indices_row_num_once % run_params.row_num_once_ub;
  }

  run_params.row_num_last_ub = res_ub_size / (run_params.params_row * params_d_size);
  OP_TILING_CHECK((run_params.row_num_last_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("Gather Tiling:", "Devide by row_num_last_ub[%ld] exception.",
                                                  run_params.row_num_last_ub),
                  return false);
  run_params.inner_loop_num_last = run_params.indices_row_num_last / run_params.row_num_last_ub;
  if (run_params.indices_row_num_last % run_params.row_num_last_ub != 0) {
    run_params.row_num_last_tail_ub = run_params.indices_row_num_last % run_params.row_num_last_ub;
  }

  return true;
}

bool CalcWithBatchDims(GatherV2TilingParams& run_params, int64_t indices_num_per_loop, int64_t res_ub_size,
                       int64_t params_d_size) {
  if(indices_num_per_loop == 0 || params_d_size == 0){
    VECTOR_INNER_ERR_REPORT_TILIING("gather_v2", "indices_num_per_loop or params_d_size= 0 is not support");
    return false;
  }
  run_params.indices_loop_num = run_params.indices_row / indices_num_per_loop;
  run_params.indices_row_num_once = indices_num_per_loop;
  int64_t block_num = BLOCK_SIZE / params_d_size;
  if (run_params.params_row * params_d_size >= BLOCK_SIZE) {
    block_num = 1;
  }
  if (run_params.indices_row % run_params.indices_row_num_once != 0) {
    run_params.indices_row_num_last = run_params.indices_num_each_core % run_params.indices_row_num_once;
  }
  if (run_params.indices_loop_num > 0 &&
      run_params.indices_row_num_last * run_params.indices_row * run_params.params_row < block_num) {
    run_params.indices_loop_num -= 1;
    run_params.indices_row_num_last += run_params.indices_row_num_once;
  }

  run_params.row_num_once_ub = res_ub_size / (run_params.params_row * params_d_size);
  if (int(run_params.row_num_once_ub % block_num) != 0) {
    run_params.row_num_once_ub = int(run_params.row_num_once_ub / block_num) * block_num;
  }
  OP_TILING_CHECK((run_params.row_num_once_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("Gather Tiling:", "Devide by row_num_once_ub[%ld] exception.",
                                                  run_params.row_num_once_ub),
                  return false);
  run_params.inner_loop_num = run_params.indices_row_num_once / run_params.row_num_once_ub;
  if (run_params.indices_row_num_once % run_params.row_num_once_ub != 0) {
    run_params.row_num_once_tail_ub = run_params.indices_row_num_once % run_params.row_num_once_ub;
  }
  if (run_params.inner_loop_num > 0 && run_params.row_num_once_tail_ub > 0 &&
      run_params.row_num_once_tail_ub * run_params.params_row < block_num) {
    run_params.inner_loop_num = run_params.inner_loop_num - 1;
    run_params.row_num_once_tail_ub = run_params.row_num_once_tail_ub + run_params.row_num_once_ub;
  }

  run_params.row_num_last_ub = run_params.row_num_once_ub;
  run_params.inner_loop_num_last = run_params.indices_row_num_last / run_params.row_num_once_ub;
  if (run_params.indices_row_num_last % run_params.row_num_once_ub != 0) {
    run_params.row_num_last_tail_ub = run_params.indices_row_num_last % run_params.row_num_once_ub;
  }
  if (run_params.inner_loop_num_last > 0 && run_params.row_num_last_tail_ub > 0 &&
      run_params.row_num_last_tail_ub * run_params.params_row < block_num) {
    run_params.inner_loop_num_last = run_params.inner_loop_num_last - 1;
    run_params.row_num_last_tail_ub = run_params.row_num_last_tail_ub + run_params.row_num_once_ub;
  }

  return true;
}

bool CalcCacheIndices(GatherV2TilingParams& run_params, int64_t indices_num_per_loop, int64_t res_ub_size,
                      int64_t params_d_size, int64_t tiling_mode) {
  if(params_d_size == 0){
    VECTOR_INNER_ERR_REPORT_TILIING("gather_v2", "params_d_size= 0 is not support");
    return false;
  }
  run_params.indices_row_num_once = indices_num_per_loop;
  run_params.row_num_once_ub = res_ub_size / (run_params.params_row * params_d_size);
  int64_t block_num = BLOCK_SIZE / params_d_size;
  int64_t align_unit;
  if (tiling_mode == TILING_MODE_38 || tiling_mode == TILING_MODE_39) {
    align_unit = run_params.indices_row * block_num;
  } else if (tiling_mode == TILING_MODE_40 || tiling_mode == TILING_MODE_41) {
    align_unit = run_params.params_pre * run_params.indices_row * block_num;
  } else if (run_params.params_row * params_d_size >= BLOCK_SIZE) {
    align_unit = 1;
  } else {
    align_unit = block_num;
  }

  if (int(run_params.row_num_once_ub % align_unit) != 0) {
    run_params.row_num_once_ub = int(run_params.row_num_once_ub / align_unit) * align_unit;
  }
  OP_TILING_CHECK((run_params.row_num_once_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("Gather Tiling:", "Devide by row_num_once_ub[%ld] exception.",
                                                  run_params.row_num_once_ub),
                  return false);
  run_params.inner_loop_num = run_params.indices_row_num_once / run_params.row_num_once_ub;
  if (run_params.indices_row_num_once % run_params.row_num_once_ub != 0) {
    run_params.row_num_once_tail_ub = run_params.indices_row_num_once % run_params.row_num_once_ub;
  }
  if (run_params.inner_loop_num > 0 && run_params.row_num_once_tail_ub > 0 &&
      run_params.row_num_once_tail_ub * run_params.params_row < block_num) {
    run_params.inner_loop_num = run_params.inner_loop_num - 1;
    run_params.row_num_once_tail_ub = run_params.row_num_once_tail_ub + run_params.row_num_once_ub;
  }
  run_params.tiling_mode = tiling_mode;

  return true;
}

void CalNeedCore(GatherV2TilingParams& run_params, const GatherCompileParams& compile_params) {
  while (run_params.need_core_num > 1) {
    run_params.need_core_num = run_params.need_core_num / 2;
    run_params.indices_num_each_core = run_params.indices_num / run_params.need_core_num;
    run_params.indices_num_remaining = run_params.indices_num % run_params.need_core_num;
    if (run_params.indices_num_each_core * run_params.params_row * compile_params.params_d_size > BLOCK_SIZE) {
      break;
    }
  }
}

void CalNeedCoreWithBatchDims(GatherV2TilingParams& run_params, const GatherCompileParams& compile_params) {
  while (run_params.need_core_num > 1) {
    run_params.need_core_num = run_params.need_core_num / 2;
    run_params.params_batch_each_core = run_params.params_batch / run_params.need_core_num;
    run_params.params_batch_remaining = run_params.params_batch % run_params.need_core_num;
    run_params.indices_num_each_core = run_params.params_batch_each_core * run_params.indices_row;
    run_params.indices_num_remaining = run_params.params_batch_remaining * run_params.indices_row;
    if (run_params.indices_num_each_core * run_params.params_pre * run_params.params_row *
            compile_params.params_d_size >
        BLOCK_SIZE) {
      break;
    }
  }
}

bool TilingWithoutBatchDims(GatherV2TilingParams& run_params, const GatherCompileParams& compile_params,
                            const GatherShapeInfo& shape_info, int64_t axis) {
  int64_t available_ub_size = compile_params.ub_size - 2 * 1024;  // reserved 2K
  int64_t half_ub_size = available_ub_size / 2;
  int64_t block_num = BLOCK_SIZE / compile_params.params_d_size;

  // params shape convert to 3D:[params_pre, params_axis, params_row]
  // indices shape convert to 1D:[indices_num]
  // output tensor, y shape convert to:[params_pre, indices_num, params_row]
  if (axis == 0) {
    run_params.params_pre = 1;
  } else {
    for (int i = 0; i < axis; i++) {
      run_params.params_pre *= shape_info.params_shape[i];
    }
  }
  run_params.params_axis = shape_info.params_shape[axis];
  int64_t paramsDims = shape_info.params_shape.size();
  if (axis + 1 < paramsDims) {
    for (int i = axis + 1; i < paramsDims; i++) {
      run_params.params_row *= shape_info.params_shape[i];
    }
  } else {
    run_params.params_row = 1;
  }

  run_params.params_total =
      std::accumulate(shape_info.params_shape.begin(), shape_info.params_shape.end(), 1, std::multiplies<int64_t>());

  int64_t params_total_ceil = (run_params.params_total + block_num - 1) / block_num * block_num;
  int64_t params_row_ceil = (run_params.params_row + block_num - 1) / block_num * block_num;
  int64_t indices_dims = shape_info.indices_shape.size();
  for (int i = 0; i < indices_dims; i++) {
    run_params.indices_num *= shape_info.indices_shape[i];
  }

  int64_t res_ub_size = half_ub_size;  // store params row data
  int64_t half_ub_indices_elem = half_ub_size / compile_params.indices_d_size;
  int64_t indices_num_per_loop = half_ub_indices_elem;
  int64_t half_remain_ub_size = (available_ub_size - PARAMS_CACHED_UB) / HALF_UB;
  int64_t half_remain_params_elem = half_remain_ub_size / compile_params.params_d_size;
  int64_t half_ub_params_elem = half_ub_size / compile_params.params_d_size;

  // the data of the formula gained from actual tests
  // set a gate value for tiling_mode_7 to optimized some data_move processes
  float mode_7_gate_value = 56.5 - 0.012 * run_params.params_total / DATA_VALUE;

  if (run_params.params_pre >= compile_params.core_num && params_row_ceil <= half_ub_params_elem &&
      (run_params.params_row * compile_params.params_d_size < BLOCK_SIZE ||
       run_params.params_row * compile_params.params_d_size % BLOCK_SIZE == 0)) {
    // block tiling: params_pre tiling
    run_params.need_core_num = compile_params.core_num;
    run_params.tail_process_core = 0;
    run_params.params_pre_each_core = run_params.params_pre / run_params.need_core_num;
    run_params.params_pre_remaining = run_params.params_pre % run_params.need_core_num;
    run_params.indices_num_each_core = run_params.indices_num;

    if (run_params.indices_num_each_core * run_params.params_row * compile_params.params_d_size <= BLOCK_SIZE) {
      run_params.need_core_num = 1;
      run_params.tail_process_core = 0;
      run_params.params_pre_each_core = run_params.params_pre;
      run_params.params_pre_remaining = 0;
    }

    if (run_params.params_row * compile_params.params_d_size < BLOCK_SIZE) {
      if (params_total_ceil <= PARAMS_CACHED_UB / compile_params.params_d_size) {
        run_params.tiling_mode = TILING_MODE_8;
      } else {
        run_params.tiling_mode = TILING_MODE_9;
      }

      if (run_params.tiling_mode == TILING_MODE_8) {
        indices_num_per_loop = half_remain_ub_size / compile_params.indices_d_size;
        res_ub_size = half_remain_ub_size;
      }

      if (!BlockLessForParamsTiling(run_params, indices_num_per_loop, res_ub_size, compile_params.params_d_size,
                                    block_num)) {
        return false;
      }
    } else {
      if (params_total_ceil <= PARAMS_CACHED_UB / compile_params.params_d_size &&
          params_row_ceil <= half_remain_params_elem) {
        run_params.tiling_mode = TILING_MODE_10;
      } else if (params_total_ceil <= compile_params.l1_size / compile_params.params_d_size) {
        run_params.tiling_mode = TILING_MODE_11;
      } else {
        run_params.tiling_mode = TILING_MODE_12;
      }

      if (run_params.tiling_mode == TILING_MODE_10) {
        indices_num_per_loop = half_remain_ub_size / compile_params.indices_d_size;
        res_ub_size = half_remain_ub_size;
      }

      if (!BlockAlignForParamsTiling(run_params, indices_num_per_loop, res_ub_size, compile_params.params_d_size)) {
        return false;
      }
    }
  } else {
    // block tiling: indices tiling
    run_params.need_core_num = compile_params.core_num;
    run_params.tail_process_core = 0;
    run_params.indices_num_each_core = run_params.indices_num / run_params.need_core_num;
    run_params.indices_num_remaining = run_params.indices_num % run_params.need_core_num;
    if (run_params.indices_num <= run_params.need_core_num) {
      run_params.need_core_num = run_params.indices_num;
      run_params.tail_process_core = 0;
      run_params.indices_num_each_core = 1;
      run_params.indices_num_remaining = 0;
    }

    // one params row size is smaller than 32B
    if (run_params.params_row * compile_params.params_d_size < BLOCK_SIZE) {
      if (params_total_ceil <= PARAMS_CACHED_UB / compile_params.params_d_size) {
        run_params.tiling_mode = TILING_MODE_4;
      } else if (params_total_ceil <= compile_params.l1_size / compile_params.params_d_size) {
        run_params.tiling_mode = TILING_MODE_13;
      } else {
        run_params.tiling_mode = TILING_MODE_1;
      }

      if ((run_params.params_row < BLOCK_SIZE) &&
          run_params.indices_num_each_core * run_params.params_row * compile_params.params_d_size <= BLOCK_SIZE) {
        CalNeedCore(run_params, compile_params);
      }

      if (run_params.tiling_mode == TILING_MODE_4) {
        indices_num_per_loop = half_remain_ub_size / compile_params.indices_d_size;
        res_ub_size = half_remain_ub_size;
      }

      if (!BlockLessForIndicesTiling(run_params, indices_num_per_loop, res_ub_size, compile_params.params_d_size,
                                     block_num)) {
        return false;
      }
    } else {  // one params row size is greater than or equal to 32B
      if (params_row_ceil <= half_ub_params_elem) {
        if (run_params.params_row * compile_params.params_d_size % BLOCK_SIZE != 0) {  // not 32B aligned
          run_params.tiling_mode = TILING_MODE_2;

          run_params.indices_loop_num = run_params.indices_num_each_core / half_ub_indices_elem;
          run_params.indices_row_num_once = half_ub_indices_elem;
          if (run_params.indices_num_each_core % run_params.indices_row_num_once != 0) {
            run_params.indices_row_num_last = run_params.indices_num_each_core % run_params.indices_row_num_once;
          }
        } else {  // 32B aligned
          if (params_total_ceil <= PARAMS_CACHED_UB / compile_params.params_d_size &&
              params_row_ceil <= half_remain_params_elem) {
            run_params.tiling_mode = TILING_MODE_6;
          } else if (params_total_ceil <= compile_params.l1_size / compile_params.params_d_size &&
                     run_params.indices_num > mode_7_gate_value) {
            run_params.tiling_mode = TILING_MODE_7;
          } else {
            run_params.tiling_mode = TILING_MODE_3;
          }

          if (run_params.tiling_mode == TILING_MODE_6) {
            indices_num_per_loop = half_remain_ub_size / compile_params.indices_d_size;
            res_ub_size = half_remain_ub_size;
          }

          if (!BlockAlignForIndicesTiling(run_params, indices_num_per_loop, res_ub_size,
                                          compile_params.params_d_size)) {
            return false;
          }
        }
      } else {
        run_params.tiling_mode = TILING_MODE_5;  // one params row need tiling

        run_params.indices_loop_num = run_params.indices_num_each_core / half_ub_indices_elem;
        run_params.indices_row_num_once = indices_num_per_loop;
        if (run_params.indices_num_each_core % run_params.indices_row_num_once != 0) {
          run_params.indices_row_num_last = run_params.indices_num_each_core % run_params.indices_row_num_once;
        }

        run_params.one_row_loop = run_params.params_row / half_ub_params_elem;
        run_params.one_row_tail = run_params.params_row % half_ub_params_elem;
        if (run_params.one_row_loop > 0 && run_params.one_row_tail > 0 && run_params.one_row_tail < block_num) {
          run_params.one_row_loop = run_params.one_row_loop - 1;
          run_params.one_row_tail = half_ub_params_elem + run_params.one_row_tail;
        }
      }
    }
  }

  return true;
}

void ParasPreProcess(GatherV2TilingParams& run_params, const GatherCompileParams& compile_params,
                     const GatherShapeInfo& shape_info, int64_t axis) {
  int64_t batch_dims = compile_params.batch_dims;
  int64_t indices_dims = shape_info.indices_shape.size();

  // params shape convert to 4D:[params_batch, params_pre, params_axis, params_row]
  // indices shape convert to 1D:[indicesBatch, indices_row]
  // output tensor, y shape convert to:[params_batch, params_pre, indices_row, params_row]
  for (int i = 0; i < batch_dims; i++) {
    run_params.indicesBatch *= shape_info.indices_shape[i];
  }
  run_params.params_batch = run_params.indicesBatch;
  for (int i = batch_dims; i < indices_dims; i++) {
    run_params.indices_row *= shape_info.indices_shape[i];
  }

  if (axis == batch_dims) {
    run_params.params_pre = 1;
  } else {
    for (int i = batch_dims; i < axis; i++) {
      run_params.params_pre *= shape_info.params_shape[i];
    }
  }
  run_params.params_axis = shape_info.params_shape[axis];
  int64_t paramsDims = shape_info.params_shape.size();
  if (axis + 1 < paramsDims) {
    for (int i = axis + 1; i < paramsDims; i++) {
      run_params.params_row *= shape_info.params_shape[i];
    }
  } else {
    run_params.params_row = 1;
  }

  for (int i = 0; i < indices_dims; i++) {
    run_params.indices_num *= shape_info.indices_shape[i];
  }
}

bool SmallRowProcess(GatherV2TilingParams& run_params, const GatherCompileParams& compile_params,
                     int64_t mode_with_cache, int64_t mode_without_cache) {
  if (mode_with_cache == TILING_MODE_38 || mode_without_cache == TILING_MODE_39) {
    run_params.params_batch_each_core = run_params.params_pre / run_params.need_core_num;
    run_params.params_batch_remaining = run_params.params_pre % run_params.need_core_num;
  }
  run_params.tail_process_core = run_params.need_core_num - 1;
  run_params.indices_num_each_core = run_params.params_batch_each_core * run_params.indices_row;
  run_params.indices_num_remaining = 0;
  int64_t block_num = BLOCK_SIZE / compile_params.params_d_size;
  int64_t indices_num_per_loop = run_params.indices_num_each_core;
  int64_t params_total_ceil = (run_params.params_total + block_num - 1) / block_num * block_num;
  int64_t params_row_ceil = (run_params.params_row + block_num - 1) / block_num * block_num;
  int64_t half_remain_params_elem = run_params.half_remain_ub_size / compile_params.params_d_size;
  if (params_total_ceil <= PARAMS_CACHED_UB / compile_params.params_d_size &&
      params_row_ceil <= half_remain_params_elem) {
    if (!CalcCacheIndices(run_params, indices_num_per_loop, run_params.half_remain_ub_size,
                          compile_params.params_d_size, mode_with_cache)) {
      return false;
    }
  } else {
    if (!CalcCacheIndices(run_params, indices_num_per_loop, run_params.half_ub_size, compile_params.params_d_size,
                          mode_without_cache)) {
      return false;
    }
  }
  return true;
}

bool IndicesCachedProcess(GatherV2TilingParams& run_params, const GatherCompileParams& compile_params,
                          int64_t aval_ub_size, int64_t mode_cache_all, int64_t mode_cache_row,
                          int64_t mode_without_cache) {
  int64_t indices_num_per_loop = 1;
  if (run_params.params_batch_each_core * run_params.indices_row * compile_params.indices_d_size <= aval_ub_size) {
    indices_num_per_loop = run_params.indices_row;
    if (!CalcCacheIndices(run_params, indices_num_per_loop, aval_ub_size, compile_params.params_d_size,
                          mode_cache_all)) {
      return false;
    }
  } else if (run_params.indices_row * compile_params.indices_d_size <= aval_ub_size) {
    indices_num_per_loop = run_params.indices_row;
    if (!CalcCacheIndices(run_params, indices_num_per_loop, aval_ub_size, compile_params.params_d_size,
                          mode_cache_row)) {
      return false;
    }
  } else {
    indices_num_per_loop = aval_ub_size / compile_params.indices_d_size;
    run_params.tiling_mode = mode_without_cache;
    if (!CalcWithBatchDims(run_params, indices_num_per_loop, aval_ub_size, compile_params.params_d_size)) {
      return false;
    }
  }
  return true;
}

bool LargeRowProcess(GatherV2TilingParams& run_params, const GatherCompileParams& compile_params,
                     int64_t half_ub_params_elem) {
  if(half_ub_params_elem == 0){
    VECTOR_INNER_ERR_REPORT_TILIING("gather_v2", "half_ub_params_elem = 0 is not support");
    return false;
  }
  run_params.one_row_loop = run_params.params_row / half_ub_params_elem;
  run_params.one_row_tail = run_params.params_row % half_ub_params_elem;
  int64_t block_num = BLOCK_SIZE / compile_params.params_d_size;
  if (run_params.one_row_loop > 0 && run_params.one_row_tail > 0 && run_params.one_row_tail < block_num) {
    run_params.one_row_loop = run_params.one_row_loop - 1;
    run_params.one_row_tail = half_ub_params_elem + run_params.one_row_tail;
  }

  if (run_params.params_batch_each_core * run_params.indices_row * compile_params.indices_d_size <=
      run_params.half_ub_size) {
    run_params.indices_row_num_once = run_params.indices_row;
    run_params.tiling_mode = TILING_MODE_35;
  } else if (run_params.indices_row * compile_params.indices_d_size <= run_params.half_ub_size) {
    run_params.indices_row_num_once = run_params.indices_row;
    run_params.tiling_mode = TILING_MODE_36;
  } else {
    int64_t indices_num_per_loop = run_params.half_ub_size / compile_params.indices_d_size;
    run_params.indices_loop_num = run_params.indices_row / indices_num_per_loop;
    run_params.indices_row_num_once = indices_num_per_loop;
    if (run_params.indices_row % run_params.indices_row_num_once != 0) {
      run_params.indices_row_num_last = run_params.indices_num_each_core % run_params.indices_row_num_once;
    }
    run_params.tiling_mode = TILING_MODE_37;
  }
  return true;
}

bool TilingWithBatchDims(GatherV2TilingParams& run_params, const GatherCompileParams& compile_params,
                         const GatherShapeInfo& shape_info, int64_t axis) {
  int64_t available_ub_size = compile_params.ub_size - 2 * 1024;  // reserved 2K
  run_params.half_ub_size = available_ub_size / 2;
  int64_t block_num = BLOCK_SIZE / compile_params.params_d_size;
  ParasPreProcess(run_params, compile_params, shape_info, axis);

  run_params.half_remain_ub_size = (available_ub_size - PARAMS_CACHED_UB) / HALF_UB;
  int64_t half_remain_params_elem = run_params.half_remain_ub_size / compile_params.params_d_size;
  int64_t half_ub_params_elem = run_params.half_ub_size / compile_params.params_d_size;

  run_params.need_core_num =
      (run_params.indicesBatch < compile_params.core_num) ? run_params.indicesBatch : compile_params.core_num;
  run_params.tail_process_core = 0;
  run_params.params_batch_each_core = run_params.params_batch / run_params.need_core_num;
  run_params.params_batch_remaining = run_params.params_batch % run_params.need_core_num;
  run_params.indices_num_each_core = run_params.params_batch_each_core * run_params.indices_row;
  run_params.indices_num_remaining = run_params.params_batch_remaining * run_params.indices_row;

  if (run_params.indices_num_each_core * run_params.params_row * compile_params.params_d_size <= BLOCK_SIZE) {
    run_params.need_core_num = 1;
    run_params.tail_process_core = 0;
    run_params.params_batch_each_core = run_params.params_batch;
    run_params.params_batch_remaining = 0;
    run_params.indices_num_each_core = run_params.params_batch_each_core * run_params.indices_row;
    run_params.indices_num_remaining = run_params.params_batch_remaining * run_params.indices_row;
  }
  run_params.params_total =
      run_params.params_batch_each_core * run_params.params_pre * run_params.params_axis * run_params.params_row;
  int64_t params_total_ceil = (run_params.params_total + block_num - 1) / block_num * block_num;
  int64_t params_row_ceil = (run_params.params_row + block_num - 1) / block_num * block_num;

  if (run_params.params_row * compile_params.params_d_size < BLOCK_SIZE) {
    if (run_params.indices_row * run_params.params_row * compile_params.params_d_size <= BLOCK_SIZE) {
      if (run_params.params_pre * run_params.indices_row * run_params.params_row * compile_params.params_d_size <=
          NUM_32 * BLOCK_SIZE) {
        if (run_params.indices_num_each_core * run_params.params_row * compile_params.params_d_size <= BLOCK_SIZE) {
          CalNeedCoreWithBatchDims(run_params, compile_params);
        }
        run_params.params_total =
            run_params.params_batch_each_core * run_params.params_pre * run_params.params_axis * run_params.params_row;
        if (!SmallRowProcess(run_params, compile_params, TILING_MODE_40, TILING_MODE_41)) {
          return false;
        }
      } else {
        run_params.need_core_num =
            (run_params.params_pre < compile_params.core_num) ? run_params.params_pre : compile_params.core_num;
        run_params.params_total =
            run_params.params_batch * run_params.params_pre * run_params.params_axis * run_params.params_row;
        if (!SmallRowProcess(run_params, compile_params, TILING_MODE_38, TILING_MODE_39)) {
          return false;
        }
      }
      return true;
    }
    if (params_total_ceil <= PARAMS_CACHED_UB / compile_params.params_d_size &&
        params_row_ceil <= half_remain_params_elem) {
      if (!IndicesCachedProcess(run_params, compile_params, run_params.half_remain_ub_size, TILING_MODE_20,
                                TILING_MODE_21, TILING_MODE_22)) {
        return false;
      }
    } else {
      if (!IndicesCachedProcess(run_params, compile_params, run_params.half_ub_size, TILING_MODE_23, TILING_MODE_24,
                                TILING_MODE_25)) {
        return false;
      }
    }
  } else {
    if (params_row_ceil <= half_ub_params_elem) {
      if (run_params.params_row * compile_params.params_d_size % BLOCK_SIZE != 0) {
        if (!IndicesCachedProcess(run_params, compile_params, run_params.half_ub_size, TILING_MODE_26, TILING_MODE_27,
                                  TILING_MODE_28)) {
          return false;
        }
      } else {
        if (params_total_ceil <= PARAMS_CACHED_UB / compile_params.params_d_size &&
            params_row_ceil <= half_remain_params_elem) {
          if (!IndicesCachedProcess(run_params, compile_params, run_params.half_remain_ub_size, TILING_MODE_29,
                                    TILING_MODE_30, TILING_MODE_31)) {
            return false;
          }
        } else {
          if (!IndicesCachedProcess(run_params, compile_params, run_params.half_ub_size, TILING_MODE_32, TILING_MODE_33,
                                    TILING_MODE_34)) {
            return false;
          }
        }
      }
    } else {
      if (!LargeRowProcess(run_params, compile_params, half_ub_params_elem)) {
        return false;
      }
    }
  }
  return true;
}

/*
 * @brief: tiling function of op
 * @param [in] op_type: op_type of the op
 * @param [in] op_paras: inputs/outputs/atts of the op
 * @param [in] op_info: compile time generated info of the op
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool GatherV2Tiling(const std::string& op_type, const ge::Operator& op_paras, const std::vector<int64_t>& op_info,
                    utils::OpRunInfo& run_info) {
  OP_LOGI(op_type.c_str(), "GatherV2Tiling running.");
  PROFILING_TILING_INIT(op_type.c_str());
  using namespace ge;

  GatherShapeInfo shape_info;
  InitGatherShapeInfo(op_type, shape_info, op_paras);

  int64_t axis = 0;
  if (op_type == "GatherV2") {
    if (!GetAxis(op_type, op_paras, axis)) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op GatherV2Tiling: [GetAxis] failed.");
      return false;
    }
  }
  PROFILING_TILING_AFTER_GET_SHAPE_REG();
  // get compile info
  GatherCompileParams compile_params;
  InitGatherCompileParams(compile_params);
  if (!GetV2GatherCompileParams(op_type, op_info, compile_params)) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op GatherV2Tiling: [GetV2GatherCompileParams] failed.");
    return false;
  }
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();
  // Compatible with gather scenarios, set axis and batch_dims equal if gather operator has batch_dims attribute.
  if (op_type == "Gather" && compile_params.batch_dims != 0) {
    axis = compile_params.batch_dims;
  }

  if (!CheckAxisAndBatchdims(op_type, shape_info, axis, compile_params)) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op GatherV2Tiling: [CheckAxisAndBatchdims] failed.");
    return false;
  }

  if (!CheckTensorShape(op_type, shape_info, axis, compile_params.batch_dims)) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op GatherV2Tiling: [CheckTensorShape] failed.");
    return false;
  }

  GatherV2TilingParams run_params;
  InitGatherV2Params(run_params);
  if (compile_params.batch_dims == 0) {
    if (!TilingWithoutBatchDims(run_params, compile_params, shape_info, axis)) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op GatherV2Tiling: [TilingWithoutBatchDims] failed.");
      return false;
    }
  } else {
    if (!TilingWithBatchDims(run_params, compile_params, shape_info, axis)) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op GatherV2Tiling: [TilingWithBatchDims] failed.");
      return false;
    }
  }
  SetGatherV2Params(run_params, run_info);
  PROFILING_TILING_AFTER_CALCU_TILING_REG();
  PrintGatherV2Params(run_params, op_type);

  // block_dim, core num used in tik op
  run_info.SetBlockDim(run_params.need_core_num);
  PROFILING_TILING_END();
  OP_LOGI(op_type.c_str(), "tiling run success.");

  return true;
}

static const std::vector<std::string> COMPILE_INFO_KEY = {"core_num",     "ub_size",       "l1_size",
                                                          "params_dsize", "indices_dsize", "batch_dims"};
static const std::map<std::string, std::int64_t> OPTIONAL_VALUE = {{"batch_dims", 0}};
// register tiling interface of the GatherV2 op.
REGISTER_OP_TILING_V3_WITH_VECTOR(GatherV2, GatherV2Tiling, COMPILE_INFO_KEY, OPTIONAL_VALUE);
// register tiling interface of the Gather op.
REGISTER_OP_TILING_V3_WITH_VECTOR(Gather, GatherV2Tiling, COMPILE_INFO_KEY, OPTIONAL_VALUE);
}  // namespace optiling
