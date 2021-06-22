/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file nll_loss_grad.cpp
 * \brief dynamic shape tiling of nll_loss_grad
 */
#include <map>

#include <nlohmann/json.hpp>
#include "op_tiling.h"

#include "op_log.h"
#include "../op_proto/util/error_util.h"
#include "../op_proto/util/op_common_util.h"
#include "error_log.h"

namespace optiling {
using namespace ge;
using namespace std;

static const int64_t DIM_2 = 2;
static const int64_t NUM_4 = 4;
static const int64_t NUM_8 = 8;
static const int64_t NUM_64 = 64;

static string to_string(const ByteBuffer& tiling_data) {
  auto data = tiling_data.str();
  string result;
  const int64_t* data_addr = reinterpret_cast<const int64_t*>(data.c_str());
  for (size_t i = 0; i < data.length() / sizeof(int64_t); i++) {
    result += std::to_string(*data_addr);
    data_addr++;
    result += ", ";
  }

  return result;
}

// round down
static int64_t GetFloorDiv(int64_t u_value, int64_t d_value) {
  int64_t res_value = 0;
  if (d_value == 0) {
    return u_value;
  }

  res_value = u_value / d_value;

  return res_value;
}

// round up
static int64_t GetCeilDiv(int64_t u_value, int64_t d_value) {
  int64_t res_value = 0;
  if (d_value == 0) {
    return u_value;
  }

  res_value = (u_value + d_value - 1) / d_value;

  return res_value;
}

static bool CheckParams(const string& op, const vector<int64_t>& x_shape, const vector<int64_t>& y_grad_shape,
                        const vector<int64_t>& target_shape, const vector<int64_t>& weight_shape,
                        const vector<int64_t>& total_weight_shape, const std::string& reduction) {
  if (x_shape.size() > DIM_2) {
    VECTOR_INNER_ERR_REPORT_TILIING(op, "The dimension of x should be equal to or less than two.");
    return false;
  }

  if (x_shape.size() == 2 && x_shape[0] != target_shape[0]) {
    VECTOR_INNER_ERR_REPORT_TILIING(op, "The first dimension of x and target should be equal.");
    return false;
  }

  if (x_shape.back() != weight_shape[0]) {
    VECTOR_INNER_ERR_REPORT_TILIING(op, "The last dimension of x and the first dimension of weight should be equal.");
    return false;
  }

  if (y_grad_shape.size() != 1) {
    VECTOR_INNER_ERR_REPORT_TILIING(op, "The dimension of y_grad should be 1D.");
    return false;
  }

  if (weight_shape.size() != 1) {
    VECTOR_INNER_ERR_REPORT_TILIING(op, "The dimension of weight should be 1D.");
    return false;
  }

  if (target_shape.size() != 1) {
    VECTOR_INNER_ERR_REPORT_TILIING(op, "The dimension of target should be 1D.");
    return false;
  }

  if (total_weight_shape[0] != 1 || total_weight_shape.size() != 1) {
    VECTOR_INNER_ERR_REPORT_TILIING(op, "The shape of total_weight must be (1,)");
    return false;
  }

  if (x_shape.size() == 1 && y_grad_shape[0] != 1) {
    VECTOR_INNER_ERR_REPORT_TILIING(op, "The shape of y_grad must be (1,), while input x is 1D.");
    return false;
  }

  if ((reduction == "mean" || reduction == "sum") && y_grad_shape[0] != 1) {
    VECTOR_INNER_ERR_REPORT_TILIING(op, "The shape of y_grad must be (1,), while reduction is mean or sum.");
    return false;
  }

  return true;
}

struct TilingParam {
  int64_t c_dim = 0;
  int64_t n_dim = 0;
  int64_t invalid_target = 0;
  int64_t ignore_idx = 0;
  int64_t output_gm_size = 0;
  int64_t x_gm_size = 0;
  int64_t y_grad_gm_size = 0;
  int64_t target_gm_size = 0;
  int64_t data_total_weight_size = 0;
  int64_t weight_gm_size = 0;
  int64_t big_weight = 0;
  int64_t core_num = 0;
  int64_t max_line = 0;
  int64_t lower_line = 0;
  int64_t loop_time = 0;
  int64_t fake_core = 0;
  int64_t redundant_line = 0;
  int64_t max_total_num = 1;
  int64_t lower_total_num = 0;
  int64_t dup_ub_size = 0;
  int64_t target_ub_size = 0;
  int64_t weight_ub_size = 0;
  int64_t total_weight_ub_size = 0;
  int64_t refactor_weight_ub_size = 0;
  int64_t weight_burst = 0;
  int64_t target_burst = 0;
  int64_t lower_target_burst = 0;
  int64_t max_vmul_repeat = 0;
  int64_t lower_vmul_repeat = 0;
  int64_t last_target_burst = 0;
  int64_t last_vmul_repeat = 0;
  int64_t core_dup_repeat = 0;
  int64_t last_dup_repeat = 0;
  int64_t max_out_burst = 0;
  int64_t last_out_burst = 0;
  int64_t y_grad_ub_size = 0;
  int64_t tiling_key = 0;
  int64_t align_repeat_size = 0;
  int64_t move_out_time = 0;
  int64_t single_max_repeat = 0;
  int64_t tail_repeat = 0;
  int64_t offet = 0;
};

static bool GetTilingParamOfNormalTwoDim(const int64_t max_move_line, const std::string& reduction,
                                         TilingParam& tiling_param) {
  tiling_param.max_line = max_move_line;
  tiling_param.lower_line = tiling_param.n_dim % max_move_line;
  tiling_param.loop_time = GetCeilDiv(tiling_param.n_dim, tiling_param.max_line * tiling_param.core_num);
  tiling_param.fake_core = GetCeilDiv(tiling_param.n_dim, tiling_param.max_line);
  tiling_param.redundant_line = tiling_param.n_dim % tiling_param.core_num;

  if (tiling_param.loop_time == 1) {
    int64_t tmp_line = GetFloorDiv(tiling_param.n_dim, tiling_param.core_num);
    if (tiling_param.redundant_line == 0) {
      tiling_param.max_line = tmp_line;
    } else {
      tiling_param.max_line = tmp_line + 1;
    }
    tiling_param.lower_line = tmp_line;
  }
  if (tiling_param.lower_line == 0) {
    tiling_param.lower_line = tiling_param.max_line;
  }
  tiling_param.max_total_num = tiling_param.max_line * tiling_param.c_dim;
  tiling_param.lower_total_num = tiling_param.lower_line * tiling_param.c_dim;
  if (tiling_param.lower_total_num < 8) {
    tiling_param.core_num = 1;
    tiling_param.max_line = tiling_param.n_dim;
    tiling_param.lower_line = tiling_param.n_dim;
    tiling_param.loop_time = 1;
    tiling_param.max_total_num = tiling_param.n_dim * tiling_param.c_dim;
    tiling_param.lower_total_num = tiling_param.max_total_num;
    tiling_param.redundant_line = 0;
  }

  tiling_param.dup_ub_size = GetCeilDiv(tiling_param.max_total_num, NUM_64) * NUM_64;
  tiling_param.target_ub_size = GetCeilDiv(tiling_param.max_line, NUM_64) * NUM_64;
  tiling_param.refactor_weight_ub_size = tiling_param.target_ub_size;

  tiling_param.weight_burst = GetCeilDiv(tiling_param.c_dim, NUM_8);
  tiling_param.target_burst = GetCeilDiv(tiling_param.max_line, NUM_8);
  tiling_param.lower_target_burst = GetCeilDiv(tiling_param.lower_line, NUM_8);
  tiling_param.max_vmul_repeat = GetCeilDiv(tiling_param.max_line, NUM_64);
  tiling_param.lower_vmul_repeat = GetCeilDiv(tiling_param.lower_line, NUM_64);
  tiling_param.last_target_burst = GetCeilDiv(tiling_param.lower_line, NUM_8);
  tiling_param.last_vmul_repeat = GetCeilDiv(tiling_param.lower_line, NUM_64);
  tiling_param.core_dup_repeat = GetCeilDiv(tiling_param.max_total_num, NUM_64);
  tiling_param.last_dup_repeat = GetCeilDiv(tiling_param.lower_total_num, NUM_64);
  tiling_param.max_out_burst = GetCeilDiv(tiling_param.max_total_num, NUM_8);
  tiling_param.last_out_burst = GetCeilDiv(tiling_param.lower_total_num, NUM_8);

  if (reduction == "none") {
    tiling_param.y_grad_ub_size = tiling_param.target_ub_size;
  } else {
    tiling_param.y_grad_ub_size = NUM_64;
  }
  tiling_param.tiling_key = 2000;
  return true;
}

static bool GetTilingParamOfOneDimAndBigWeight(const int64_t ub_size_float, TilingParam& tiling_param) {
  tiling_param.tiling_key = 2001;
  tiling_param.refactor_weight_ub_size = NUM_8;
  tiling_param.loop_time = GetCeilDiv(tiling_param.n_dim, tiling_param.core_num);
  tiling_param.align_repeat_size = GetFloorDiv(ub_size_float, NUM_64) * NUM_64;
  tiling_param.move_out_time = GetCeilDiv(tiling_param.c_dim, tiling_param.align_repeat_size);
  tiling_param.single_max_repeat = GetFloorDiv(tiling_param.align_repeat_size, NUM_64);
  tiling_param.tail_repeat = GetCeilDiv(tiling_param.c_dim, NUM_64) % tiling_param.single_max_repeat;
  tiling_param.max_out_burst = tiling_param.single_max_repeat * NUM_8;
  tiling_param.last_out_burst =
      GetCeilDiv(tiling_param.c_dim, NUM_8) - (tiling_param.move_out_time - 1) * tiling_param.max_out_burst;
  if (tiling_param.move_out_time == 1) {
    tiling_param.tail_repeat = GetCeilDiv(tiling_param.c_dim, NUM_64);
    tiling_param.last_out_burst = GetCeilDiv(tiling_param.c_dim, NUM_8);
    tiling_param.single_max_repeat = tiling_param.tail_repeat;
    tiling_param.max_out_burst = tiling_param.last_out_burst;
  }
  tiling_param.offet = tiling_param.max_out_burst * NUM_8;
  tiling_param.dup_ub_size = GetCeilDiv(tiling_param.max_out_burst * NUM_8, NUM_64) * NUM_64;
  tiling_param.weight_ub_size = NUM_64;
  tiling_param.target_ub_size = NUM_64;
  tiling_param.y_grad_ub_size = NUM_64;

  return true;
}

static bool GetTilingParam(const vector<int64_t>& x_shape, const vector<int64_t>& y_grad_shape,
                           const vector<int64_t>& target_shape, const vector<int64_t>& weight_shape,
                           const int64_t block_dim, const int64_t ub_size, const std::string& reduction,
                           const int64_t ignore_idx, TilingParam& tiling_param) {
  int64_t c_dim = x_shape.back();
  int64_t n_dim = x_shape[0];
  if (ignore_idx < 0 || ignore_idx >= c_dim) {
    tiling_param.invalid_target = 1;
  }
  tiling_param.ignore_idx = ignore_idx;
  int64_t ub_size_float = GetFloorDiv(ub_size, NUM_4);
  if (x_shape.size() == DIM_2) {
    tiling_param.output_gm_size = c_dim * n_dim;
  } else {
    tiling_param.output_gm_size = n_dim;
  }

  tiling_param.n_dim = n_dim;
  tiling_param.c_dim = c_dim;
  tiling_param.x_gm_size = tiling_param.output_gm_size;
  tiling_param.y_grad_gm_size = y_grad_shape[0];
  tiling_param.target_gm_size = target_shape[0];
  tiling_param.data_total_weight_size = 1;
  tiling_param.weight_gm_size = weight_shape[0];
  tiling_param.weight_ub_size = GetCeilDiv(weight_shape[0], NUM_64) * NUM_64;
  tiling_param.total_weight_ub_size = NUM_64;
  int64_t last_ub_size = ub_size_float - tiling_param.weight_ub_size;
  int64_t one_line_size = c_dim + 3;
  int64_t max_move_line = GetFloorDiv(last_ub_size, one_line_size);
  if (max_move_line < 2) {
    tiling_param.big_weight = 1;
  }
  if (n_dim <= block_dim) {
    tiling_param.core_num = n_dim;
  } else {
    tiling_param.core_num = block_dim;
  }

  if (tiling_param.big_weight == 1) {
    bool flag = GetTilingParamOfOneDimAndBigWeight(ub_size_float, tiling_param);
    if (!flag) {
      VECTOR_INNER_ERR_REPORT_TILIING("NLLLossGrad", "NLLLossGradTiling: GetTilingParamOfOneDimAndBigWeight error.");
      return false;
    }
  } else {
    bool flag = GetTilingParamOfNormalTwoDim(max_move_line, reduction, tiling_param);
    if (!flag) {
      VECTOR_INNER_ERR_REPORT_TILIING("NLLLossGrad", "NLLLossGradTiling: GetTilingParamOfNormalTwoDim error.");
      return false;
    }
  }

  OP_LOGD("NLLLossGrad", "GetTilingParams, c_dim[%lld]", tiling_param.c_dim);
  OP_LOGD("NLLLossGrad", "GetTilingParams, n_dim[%lld]", tiling_param.n_dim);
  OP_LOGD("NLLLossGrad", "GetTilingParams, invalid_target[%lld]", tiling_param.invalid_target);
  OP_LOGD("NLLLossGrad", "GetTilingParams, ignore_idx[%lld]", tiling_param.ignore_idx);
  OP_LOGD("NLLLossGrad", "GetTilingParams, output_gm_size[%lld]", tiling_param.output_gm_size);
  OP_LOGD("NLLLossGrad", "GetTilingParams, x_gm_size[%lld]", tiling_param.x_gm_size);
  OP_LOGD("NLLLossGrad", "GetTilingParams, y_grad_gm_size[%lld]", tiling_param.y_grad_gm_size);
  OP_LOGD("NLLLossGrad", "GetTilingParams, target_gm_size[%lld]", tiling_param.target_gm_size);
  OP_LOGD("NLLLossGrad", "GetTilingParams, data_total_weight_size[%lld]", tiling_param.data_total_weight_size);
  OP_LOGD("NLLLossGrad", "GetTilingParams, weight_gm_size[%lld]", tiling_param.weight_gm_size);
  OP_LOGD("NLLLossGrad", "GetTilingParams, big_weight[%lld]", tiling_param.big_weight);
  OP_LOGD("NLLLossGrad", "GetTilingParams, core_num[%lld]", tiling_param.core_num);
  OP_LOGD("NLLLossGrad", "GetTilingParams, max_line[%lld]", tiling_param.max_line);
  OP_LOGD("NLLLossGrad", "GetTilingParams, lower_line[%lld]", tiling_param.lower_line);
  OP_LOGD("NLLLossGrad", "GetTilingParams, loop_time[%lld]", tiling_param.loop_time);
  OP_LOGD("NLLLossGrad", "GetTilingParams, fake_core[%lld]", tiling_param.fake_core);
  OP_LOGD("NLLLossGrad", "GetTilingParams, redundant_line[%lld]", tiling_param.redundant_line);
  OP_LOGD("NLLLossGrad", "GetTilingParams, max_total_num[%lld]", tiling_param.max_total_num);
  OP_LOGD("NLLLossGrad", "GetTilingParams, lower_total_num[%lld]", tiling_param.lower_total_num);
  OP_LOGD("NLLLossGrad", "GetTilingParams, dup_ub_size[%lld]", tiling_param.dup_ub_size);
  OP_LOGD("NLLLossGrad", "GetTilingParams, target_ub_size[%lld]", tiling_param.target_ub_size);
  OP_LOGD("NLLLossGrad", "GetTilingParams, weight_ub_size[%lld]", tiling_param.weight_ub_size);
  OP_LOGD("NLLLossGrad", "GetTilingParams, total_weight_ub_size[%lld]", tiling_param.total_weight_ub_size);
  OP_LOGD("NLLLossGrad", "GetTilingParams, refactor_weight_ub_size[%lld]", tiling_param.refactor_weight_ub_size);
  OP_LOGD("NLLLossGrad", "GetTilingParams, weight_burst[%lld]", tiling_param.weight_burst);
  OP_LOGD("NLLLossGrad", "GetTilingParams, target_burst[%lld]", tiling_param.target_burst);
  OP_LOGD("NLLLossGrad", "GetTilingParams, lower_target_burst[%lld]", tiling_param.lower_target_burst);
  OP_LOGD("NLLLossGrad", "GetTilingParams, max_vmul_repeat[%lld]", tiling_param.max_vmul_repeat);
  OP_LOGD("NLLLossGrad", "GetTilingParams, lower_vmul_repeat[%lld]", tiling_param.lower_vmul_repeat);
  OP_LOGD("NLLLossGrad", "GetTilingParams, last_target_burst[%lld]", tiling_param.last_target_burst);
  OP_LOGD("NLLLossGrad", "GetTilingParams, last_vmul_repeat[%lld]", tiling_param.last_vmul_repeat);
  OP_LOGD("NLLLossGrad", "GetTilingParams, core_dup_repeat[%lld]", tiling_param.core_dup_repeat);
  OP_LOGD("NLLLossGrad", "GetTilingParams, last_dup_repeat[%lld]", tiling_param.last_dup_repeat);
  OP_LOGD("NLLLossGrad", "GetTilingParams, max_out_burst[%lld]", tiling_param.max_out_burst);
  OP_LOGD("NLLLossGrad", "GetTilingParams, last_out_burst[%lld]", tiling_param.last_out_burst);
  OP_LOGD("NLLLossGrad", "GetTilingParams, y_grad_ub_size[%lld]", tiling_param.y_grad_ub_size);
  OP_LOGD("NLLLossGrad", "GetTilingParams, tiling_key[%lld]", tiling_param.tiling_key);
  OP_LOGD("NLLLossGrad", "GetTilingParams, align_repeat_size[%lld]", tiling_param.align_repeat_size);
  OP_LOGD("NLLLossGrad", "GetTilingParams, move_out_time[%lld]", tiling_param.move_out_time);
  OP_LOGD("NLLLossGrad", "GetTilingParams, single_max_repeat[%lld]", tiling_param.single_max_repeat);
  OP_LOGD("NLLLossGrad", "GetTilingParams, tail_repeat[%lld]", tiling_param.tail_repeat);
  OP_LOGD("NLLLossGrad", "GetTilingParams, offet[%lld]", tiling_param.offet);

  return true;
}

static bool GetCompileParams(const nlohmann::json& op_compile_info_json, std::string& reduction, std::string& dtype,
                             std::string& dtype_weight, int64_t& ignore_idx, int64_t& ub_size, int64_t& block_dim,
                             const std::string& op_type) {
  using namespace nlohmann;

  auto allVars = op_compile_info_json["vars"];
  if (allVars.count("reduction") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("NLLLossGradTiling", "GetCompileParams, get reduction error");
    return false;
  }
  reduction = allVars["reduction"].get<std::string>();

  if (allVars.count("dtype") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("NLLLossGradTiling", "GetCompileParams, get dtype error");
    return false;
  }
  dtype = allVars["dtype"].get<std::string>();

  if (allVars.count("dtype_weight") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("NLLLossGradTiling", "GetCompileParams, get dtype_weight error");
    return false;
  }
  dtype_weight = allVars["dtype_weight"].get<std::string>();

  if (allVars.count("ignore_idx") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("NLLLossGradTiling", "GetCompileParams, get ignore_idx error");
    return false;
  }
  ignore_idx = allVars["ignore_idx"].get<std::int64_t>();

  if (allVars.count("ub_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("NLLLossGradTiling", "GetCompileParams, get ub_size error");
    return false;
  }
  ub_size = allVars["ub_size"].get<std::int64_t>();

  if (allVars.count("block_dim") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("NLLLossGradTiling", "GetCompileParams, get block_dim error");
    return false;
  }
  block_dim = allVars["block_dim"].get<std::int64_t>();

  OP_LOGD(op_type,
          "GetCompileParams, reduction[%s], dtype[%s], dtype_weight[%s], ignore_idx[%lld], "
          "ub_size[%lld], block_dim[%lld].",
          reduction.c_str(), dtype.c_str(), dtype_weight.c_str(), ignore_idx, ub_size, block_dim);

  return true;
}

bool SetRunningInfo(const TilingParam& tiling_param, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, tiling_param.c_dim);
  ByteBufferPut(runInfo.tiling_data, tiling_param.n_dim);
  ByteBufferPut(runInfo.tiling_data, tiling_param.invalid_target);
  ByteBufferPut(runInfo.tiling_data, tiling_param.ignore_idx);
  ByteBufferPut(runInfo.tiling_data, tiling_param.output_gm_size);
  ByteBufferPut(runInfo.tiling_data, tiling_param.x_gm_size);
  ByteBufferPut(runInfo.tiling_data, tiling_param.y_grad_gm_size);
  ByteBufferPut(runInfo.tiling_data, tiling_param.target_gm_size);
  ByteBufferPut(runInfo.tiling_data, tiling_param.data_total_weight_size);
  ByteBufferPut(runInfo.tiling_data, tiling_param.weight_gm_size);
  ByteBufferPut(runInfo.tiling_data, tiling_param.big_weight);
  ByteBufferPut(runInfo.tiling_data, tiling_param.core_num);
  ByteBufferPut(runInfo.tiling_data, tiling_param.max_line);
  ByteBufferPut(runInfo.tiling_data, tiling_param.lower_line);
  ByteBufferPut(runInfo.tiling_data, tiling_param.loop_time);
  ByteBufferPut(runInfo.tiling_data, tiling_param.fake_core);
  ByteBufferPut(runInfo.tiling_data, tiling_param.redundant_line);
  ByteBufferPut(runInfo.tiling_data, tiling_param.max_total_num);
  ByteBufferPut(runInfo.tiling_data, tiling_param.lower_total_num);
  ByteBufferPut(runInfo.tiling_data, tiling_param.dup_ub_size);
  ByteBufferPut(runInfo.tiling_data, tiling_param.target_ub_size);
  ByteBufferPut(runInfo.tiling_data, tiling_param.weight_ub_size);
  ByteBufferPut(runInfo.tiling_data, tiling_param.total_weight_ub_size);
  ByteBufferPut(runInfo.tiling_data, tiling_param.refactor_weight_ub_size);
  ByteBufferPut(runInfo.tiling_data, tiling_param.weight_burst);
  ByteBufferPut(runInfo.tiling_data, tiling_param.target_burst);
  ByteBufferPut(runInfo.tiling_data, tiling_param.lower_target_burst);
  ByteBufferPut(runInfo.tiling_data, tiling_param.max_vmul_repeat);
  ByteBufferPut(runInfo.tiling_data, tiling_param.lower_vmul_repeat);
  ByteBufferPut(runInfo.tiling_data, tiling_param.last_target_burst);
  ByteBufferPut(runInfo.tiling_data, tiling_param.last_vmul_repeat);
  ByteBufferPut(runInfo.tiling_data, tiling_param.core_dup_repeat);
  ByteBufferPut(runInfo.tiling_data, tiling_param.last_dup_repeat);
  ByteBufferPut(runInfo.tiling_data, tiling_param.max_out_burst);
  ByteBufferPut(runInfo.tiling_data, tiling_param.last_out_burst);
  ByteBufferPut(runInfo.tiling_data, tiling_param.y_grad_ub_size);
  ByteBufferPut(runInfo.tiling_data, tiling_param.tiling_key);
  ByteBufferPut(runInfo.tiling_data, tiling_param.align_repeat_size);
  ByteBufferPut(runInfo.tiling_data, tiling_param.move_out_time);
  ByteBufferPut(runInfo.tiling_data, tiling_param.single_max_repeat);
  ByteBufferPut(runInfo.tiling_data, tiling_param.tail_repeat);
  ByteBufferPut(runInfo.tiling_data, tiling_param.offet);

  return true;
}

/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] opParas: inputs/outputs/atts of the op
 * @param [in] op_info: compile time generated info of the op
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool NLLLossGradTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& op_info,
                       OpRunInfo& runInfo) {
  OP_LOGD(opType, "NLLLossGradTiling running.");

  std::vector<int64_t> x_shape = opParas.inputs[0].tensor[0].shape;
  std::vector<int64_t> y_grad_shape = opParas.inputs[1].tensor[0].shape;
  std::vector<int64_t> target_shape = opParas.inputs[2].tensor[0].shape;
  std::vector<int64_t> weight_shape = opParas.inputs[3].tensor[0].shape;
  std::vector<int64_t> total_weight_shape = opParas.inputs[4].tensor[0].shape;
  std::vector<int64_t> out_shape = opParas.outputs[0].tensor[0].shape;

  OP_LOGD(
      opType,
      "Original x_shape is %s, y_grad_shape is %s, target_shape is %s, weight_shape is %s, total_weight_shape is %s, "
      "out_shape is %s.",
      ops::to_string(x_shape).c_str(), ops::to_string(y_grad_shape).c_str(), ops::to_string(target_shape).c_str(),
      ops::to_string(weight_shape).c_str(), ops::to_string(total_weight_shape).c_str(),
      ops::to_string(out_shape).c_str());

  std::string reduction;
  std::string dtype;
  std::string dtype_weight;
  int64_t ignore_idx = -100;
  int64_t ub_size = 0;
  int64_t block_dim = 0;

  // get compileinfo params
  bool flag = GetCompileParams(op_info, reduction, dtype, dtype_weight, ignore_idx, ub_size, block_dim, opType);
  if (!flag) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "NLLLossGradTiling: GetCompileParams error.");
    return false;
  }

  // check params
  if (x_shape.empty() || y_grad_shape.empty() || target_shape.empty() || weight_shape.empty() ||
      total_weight_shape.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "Get input shape failed.");
    return false;
  }

  OP_LOGD(opType, "to check params.");
  if (!CheckParams(opType, x_shape, y_grad_shape, target_shape, weight_shape, total_weight_shape, reduction)) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "NLLLossGradTiling: CheckParams error.");
    return false;
  }

  // One dim, converted to two dims processing; for example：x=(N,)->x=(1,N);target=(N,)->target=(1,)
  if (x_shape.size() == 1) {
    x_shape.insert(x_shape.begin(), 1);
    target_shape[0] = 1;
  }

  OP_LOGD(opType, "GetTilingParam.");
  TilingParam tiling_param;
  if (!GetTilingParam(x_shape, y_grad_shape, target_shape, weight_shape, block_dim, ub_size, reduction, ignore_idx,
                      tiling_param)) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "NLLLossGradTiling: GetTilingParam error.");
    return false;
  }

  OP_LOGD(opType, "encode TilingParam.");
  if (!SetRunningInfo(tiling_param, runInfo)) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "NLLLossGradTiling: SetRunningInfo error.");
    return false;
  }
  OP_LOGD(opType, "TilingParam:%s.", to_string(runInfo.tiling_data).c_str());

  // block_dim
  runInfo.block_dim = tiling_param.core_num;

  // workspace, null for tik op
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;
  OP_LOGD(opType, "tiling run success.");

  return true;
}

// register tiling interface of the NLLLossGrad op.
REGISTER_OP_TILING_FUNC_BUFFERED(NLLLossGrad, NLLLossGradTiling);
}  // namespace optiling
