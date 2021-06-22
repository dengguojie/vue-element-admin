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
 * \file max_pool.cc
 * \brief dynamic shape tiling of max_pool
 */
#include <map>
#include <nlohmann/json.hpp>

#include "../op_proto/util/error_util.h"
#include "error_log.h"
#include "op_log.h"
#include "op_tiling.h"

namespace optiling {
using namespace ge;
using namespace std;

struct TilingParam {
  int32_t tiling_mode = 0;
  int32_t act_core_num = 0;
  int32_t one_core_ele = 0;
  int32_t last_core_ele = 0;
  int32_t input_h = 0;
  int32_t input_w = 0;
  int32_t output_h = 0;
  int32_t output_w = 0;
  int32_t pad_h = 0;
  int32_t pad_w = 0;
  int32_t pad_t = 0;
  int32_t pad_b = 0;
  int32_t pad_l = 0;
  int32_t pad_r = 0;
  int32_t c_factor = 1;
  int32_t h_factor = 1;
  int32_t w_factor = 1;
  int32_t one_core_loop_num = 0;
  int32_t one_core_loop_left = 0;
  int32_t last_core_loop_num = 0;
  int32_t last_core_loop_left = 0;
  int32_t n_c1 = 0;
};

static void PrintTilingParam(const TilingParam& param) {
  OP_LOGD("MaxPoolTiling ",
          "(tiling_mode,act_core_num,one_core_ele,last_core_ele,input_h,input_w,output_h,output_w,pad_h,pad_w,pad_t,"
          "pad_b,pad_l,pad_r,c_factor,h_factor,w_factor,one_core_loop_num,one_core_loop_left,last_core_loop_num,"
          "last_core_loop_left,n_c1):(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d)",
          param.tiling_mode, param.act_core_num, param.one_core_ele, param.last_core_ele, param.input_h, param.input_w,
          param.output_h, param.output_w, param.pad_h, param.pad_w, param.pad_t, param.pad_b, param.pad_l, param.pad_r,
          param.c_factor, param.h_factor, param.w_factor, param.one_core_loop_num, param.one_core_loop_left,
          param.last_core_loop_num, param.last_core_loop_left, param.n_c1);
}

static void SetTilingParam(const TilingParam& param, OpRunInfo& run_info) {
  ByteBufferPut(run_info.tiling_data, param.tiling_mode);
  ByteBufferPut(run_info.tiling_data, param.act_core_num);
  ByteBufferPut(run_info.tiling_data, param.one_core_ele);
  ByteBufferPut(run_info.tiling_data, param.last_core_ele);
  ByteBufferPut(run_info.tiling_data, param.input_h);
  ByteBufferPut(run_info.tiling_data, param.input_w);
  ByteBufferPut(run_info.tiling_data, param.output_h);
  ByteBufferPut(run_info.tiling_data, param.output_w);
  ByteBufferPut(run_info.tiling_data, param.pad_h);
  ByteBufferPut(run_info.tiling_data, param.pad_w);
  ByteBufferPut(run_info.tiling_data, param.pad_t);
  ByteBufferPut(run_info.tiling_data, param.pad_b);
  ByteBufferPut(run_info.tiling_data, param.pad_l);
  ByteBufferPut(run_info.tiling_data, param.pad_r);
  ByteBufferPut(run_info.tiling_data, param.c_factor);
  ByteBufferPut(run_info.tiling_data, param.h_factor);
  ByteBufferPut(run_info.tiling_data, param.w_factor);
  ByteBufferPut(run_info.tiling_data, param.one_core_loop_num);
  ByteBufferPut(run_info.tiling_data, param.one_core_loop_left);
  ByteBufferPut(run_info.tiling_data, param.last_core_loop_num);
  ByteBufferPut(run_info.tiling_data, param.last_core_loop_left);
  ByteBufferPut(run_info.tiling_data, param.n_c1);
}

static void CalCoreNum(TilingParam& param, int32_t total_ele, int32_t core_num) {
  param.one_core_ele = (total_ele + core_num - 1) / core_num;
  param.act_core_num = total_ele / param.one_core_ele;
  if (total_ele % param.one_core_ele != 0) {
    param.act_core_num = param.act_core_num + 1;
  }
  param.last_core_ele = total_ele - (param.act_core_num - 1) * param.one_core_ele;
}

static void CalTilingParam(TilingParam& param, const vector<int64_t>& input_shape, int32_t ub_ele, int32_t core_num,
                           int32_t ksize_h, int32_t ksize_w, int32_t strides_h, int32_t strides_w, int32_t padding) {
  // calc output height and width, pad infos
  if (padding == 0) {
    param.output_h = (param.input_h + strides_h - 1) / strides_h;
    param.output_w = (param.input_w + strides_w - 1) / strides_w;
    param.pad_h = (param.output_h - 1) * strides_h + ksize_h;
    param.pad_w = (param.output_w - 1) * strides_w + ksize_w;
    param.pad_t = (param.pad_h - param.input_h) / 2 > 0 ? (param.pad_h - param.input_h) / 2 : 0;
    param.pad_b = param.pad_h - param.input_h - param.pad_t > 0 ? param.pad_h - param.input_h - param.pad_t : 0;
    param.pad_l = (param.pad_w - param.input_w) / 2 > 0 ? (param.pad_w - param.input_w) / 2 : 0;
    param.pad_r = param.pad_w - param.input_w - param.pad_l > 0 ? param.pad_w - param.input_w - param.pad_l : 0;
  } else {
    param.output_h = (param.input_h - (ksize_h - 1) + strides_h - 1) / strides_h;
    param.output_w = (param.input_w - (ksize_w - 1) + strides_w - 1) / strides_w;
    param.pad_h = (param.output_h - 1) * strides_h + ksize_h;
    param.pad_w = (param.output_w - 1) * strides_w + ksize_w;
    param.pad_t = 0;
    param.pad_b = 0;
    param.pad_l = 0;
    param.pad_r = 0;
  }

  // calc core_num, core_ele, loop_num and loop_left
  if ((ksize_h == 1) && (ksize_w == 1) && (strides_h == 1) && (strides_w == 1)) {
    param.tiling_mode = 0;
    int32_t max_ele = ub_ele / input_shape[4];
    int32_t total_ele = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
    CalCoreNum(param, total_ele, core_num);
    param.one_core_loop_num = param.one_core_ele / max_ele;
    param.one_core_loop_left = param.one_core_ele % max_ele;
    param.last_core_loop_num = param.last_core_ele / max_ele;
    param.last_core_loop_left = param.last_core_ele % max_ele;
  } else {
    int32_t one_sixth_ub_ele = ub_ele / 6;
    param.n_c1 = input_shape[0] * input_shape[1];
    if (param.pad_h * param.pad_w * input_shape[4] <= one_sixth_ub_ele) {
      param.tiling_mode = 1;
      CalCoreNum(param, param.n_c1, core_num);
      param.c_factor = one_sixth_ub_ele / (param.pad_h * param.pad_w * input_shape[4]);
      param.one_core_loop_num = param.one_core_ele / param.c_factor;
      param.one_core_loop_left = param.one_core_ele % param.c_factor;
      param.last_core_loop_num = param.last_core_ele / param.c_factor;
      param.last_core_loop_left = param.last_core_ele % param.c_factor;
    } else if (ksize_h * param.pad_w * input_shape[4] <= one_sixth_ub_ele) {
      param.h_factor = (one_sixth_ub_ele / (param.pad_w * input_shape[4]) - ksize_h) / strides_h + 1;
      int32_t h_loop = param.output_h / param.h_factor;
      if (h_loop <= param.n_c1) {
        param.tiling_mode = 2;
        CalCoreNum(param, param.n_c1, core_num);
        param.one_core_loop_num = param.output_h / param.h_factor;
        param.one_core_loop_left = param.output_h % param.h_factor;
        param.last_core_loop_num = param.one_core_loop_num;
        param.last_core_loop_left = param.one_core_loop_left;
      } else {
        param.tiling_mode = 4;
        CalCoreNum(param, param.output_h, core_num);
        param.one_core_loop_num = param.one_core_ele / param.h_factor;
        param.one_core_loop_left = param.one_core_ele % param.h_factor;
        param.last_core_loop_num = param.last_core_ele / param.h_factor;
        param.last_core_loop_left = param.last_core_ele % param.h_factor;
      }
    } else {
      param.w_factor = (one_sixth_ub_ele / input_shape[4] / ksize_h - ksize_w) / strides_w + 1;
      param.one_core_loop_num = param.output_w / param.w_factor;
      param.one_core_loop_left = param.output_w % param.w_factor;
      param.last_core_loop_num = param.one_core_loop_num;
      param.last_core_loop_left = param.one_core_loop_left;
      if (param.output_h <= param.n_c1) {
        param.tiling_mode = 3;
        CalCoreNum(param, param.n_c1, core_num);
      } else {
        param.tiling_mode = 5;
        CalCoreNum(param, param.output_h, core_num);
      }
    }
  }
}

template <typename T>
static bool GetCompileInfo(const nlohmann::json& op_info, const string& name, T& value) {
  const nlohmann::json& all_vars = op_info["vars"];
  if (all_vars.empty()) {
    return false;
  }
  if (all_vars.count(name) == 0) {
    return false;
  }
  value = all_vars[name].get<T>();
  return true;
}

/*
 * @brief: tiling function of op
 * @param [in] op_type: type of the op
 * @param [in] op_paras: inputs/outputs/attrs of the op
 * @param [in] op_info: compile time generated info of the op
 * @param [out] run_info: result data
 * @return bool: success or not success
 */
bool MaxPoolTiling(const string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                   OpRunInfo& run_info) {
  OP_LOGI(op_type.c_str(), "MaxPoolTiling running.");

  // get and check input format and shape
  string input_format = op_paras.inputs[0].tensor[0].format;
  OP_TILING_CHECK(
      input_format != "NC1HWC0",
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get input format failed, only support NC1HWC0, but got %s.", input_format.c_str()),
      return false);
  vector<int64_t> input_shape = op_paras.inputs[0].tensor[0].shape;
  OP_TILING_CHECK(input_shape.size() != 5,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get input shape failed, the length of input shape must be 5, but got %d.",
                          input_shape.size()),
                  return false);

  // get compile info
  int32_t ub_ele = 0;
  int32_t core_num = 1;
  int32_t ksize_h = 1;
  int32_t ksize_w = 1;
  int32_t strides_h = 1;
  int32_t strides_w = 1;
  int32_t padding = 0;  // SAME
  const map<string, int32_t&> compile_params = {
      {"ub_ele", ub_ele},       {"core_num", core_num},   {"ksize_h", ksize_h}, {"ksize_w", ksize_w},
      {"strides_h", strides_h}, {"strides_w", strides_w}, {"padding", padding},
  };
  for (auto& param : compile_params) {
    const auto& name = param.first;
    OP_LOGD(op_type.c_str(), "GetCompileInfo %s.", name.c_str());
    OP_TILING_CHECK(!GetCompileInfo<int32_t>(op_info, name, param.second),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetCompileInfo %s failed.", name.c_str()), return false);
    OP_LOGD(op_type.c_str(), "%s=%d.", name.c_str(), param.second);
  }

  // check ksize, strides and input shape
  TilingParam param;
  param.input_h = input_shape[2];
  param.input_w = input_shape[3];
  OP_TILING_CHECK(
      (padding == 1) && ((ksize_h > param.input_h) || (ksize_w > param.input_w)),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Input height or width must greater than or equal to ksize when padding mode is valid."),
      return false);
  int32_t one_sixth_ub_ele = ub_ele / 6;
  OP_TILING_CHECK(one_sixth_ub_ele / input_shape[4] / ksize_h < ksize_w,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get tiling failed, minimum processing unit must be ksize_h * ksize_w."),
                  return false);

  // calc tiling params, set tiling params, print tiling params
  CalTilingParam(param, input_shape, ub_ele, core_num, ksize_h, ksize_w, strides_h, strides_w, padding);
  SetTilingParam(param, run_info);
  PrintTilingParam(param);

  // block_dim, use fot tik op; workspace, null for tik op
  run_info.block_dim = param.act_core_num;
  vector<int64_t> workspace;
  run_info.workspaces = workspace;

  OP_LOGI(op_type.c_str(), "MaxPoolTiling run success.");
  return true;
}

// register tiling interface of maxpool op.
REGISTER_OP_TILING_FUNC_BUFFERED(MaxPool, MaxPoolTiling);
}  // namespace optiling
