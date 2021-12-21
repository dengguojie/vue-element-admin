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
 * \file maxpoolwithargmaxv2.cc
 * \brief dynamic shape tiling of max_pool_with_argmaxv2
 */
#include <map>
#include <nlohmann/json.hpp>

#include "../op_proto/util/error_util.h"
#include "error_log.h"
#include "op_log.h"
#include "op_tiling.h"

namespace {
  constexpr int32_t TILING_FACTOR_2 = 2;
  constexpr int32_t TILING_DIVIDE_6 = 6;
  constexpr int32_t TILING_MODE_1 = 1;
  constexpr int32_t TILING_MODE_2 = 2;
  constexpr int32_t TILING_MODE_3 = 3;
  constexpr int32_t TILING_MODE_5 = 5;
  constexpr int32_t ALLIGN_NUM = 16;
}

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
  int32_t align_w_factor = 0;
  int32_t align_w_loop_left = 0;
  int32_t align_output_w = 0;
  int32_t align_output_hw = 0;
};

struct CompileInfoParam {
  // get compile info
  int32_t ub_ele = 0;
  int32_t core_num = 1;
  int32_t ksize_h = 1;
  int32_t ksize_w = 1;
  int32_t strides_h = 1;
  int32_t strides_w = 1;
  int32_t padding = 0;
  int32_t ceil_mode = 0;
  int32_t pad_top = 0;
  int32_t pad_bottom = 0;
  int32_t pad_left = 0;
  int32_t pad_right = 0;
  int32_t global = 0;
};

static void PrintTilingParam(const TilingParam& param) {
  OP_LOGD("MaxPoolWithArgmaxV2Tiling ",
          "(tiling_mode,act_core_num,one_core_ele,last_core_ele,input_h,input_w,output_h,output_w,pad_h,pad_w,pad_t,"
          "pad_b,pad_l,pad_r,c_factor,h_factor,w_factor,one_core_loop_num,one_core_loop_left,last_core_loop_num,"
          "last_core_loop_left,n_c1,align_w_factor,align_w_loop_left,align_output_w,align_output_hw):"
          "(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d)",
          param.tiling_mode, param.act_core_num, param.one_core_ele, param.last_core_ele, param.input_h, param.input_w,
          param.output_h, param.output_w, param.pad_h, param.pad_w, param.pad_t, param.pad_b, param.pad_l, param.pad_r,
          param.c_factor, param.h_factor, param.w_factor, param.one_core_loop_num, param.one_core_loop_left,
          param.last_core_loop_num, param.last_core_loop_left, param.n_c1,
          param.align_w_factor, param.align_w_loop_left, param.align_output_w, param.align_output_hw);
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
  ByteBufferPut(run_info.tiling_data, param.align_w_factor);
  ByteBufferPut(run_info.tiling_data, param.align_w_loop_left);
  ByteBufferPut(run_info.tiling_data, param.align_output_w);
  ByteBufferPut(run_info.tiling_data, param.align_output_hw);
}


static int32_t DivRtn(int32_t x, int32_t y) {
  if (y == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("MaxPoolWithArgMaxV2", "y value cannot be zero");
    return 0;
  }
  int32_t q = x / y;
  int32_t r = x % y;
  if ((r != 0) && ((r < 0) != (y < 0))) {
    --q;
  }
  return q;
}

static int32_t GetRequireMemory(TilingParam& param, int32_t mode, int32_t input_memory, int32_t k_h, int32_t k_w) {
  int32_t align_output_w = 0;
  int32_t mask_memory = 0;
  int32_t require_memory = 0;
  if (param.output_w % ALLIGN_NUM != 0) {
    align_output_w = (param.output_w / ALLIGN_NUM + 1) * ALLIGN_NUM;
  } else {
    align_output_w = param.output_w;
  }
  if (mode == TILING_MODE_1) {
    mask_memory = align_output_w * param.output_h * k_h * k_w;
  } else if (mode == TILING_MODE_2) {
    mask_memory = align_output_w * k_h * k_w;
  }
  if (input_memory > mask_memory) {
    require_memory = input_memory;
  } else {
    require_memory = mask_memory;
  }
  return require_memory;
}

static int32_t GetFactor(TilingParam& param, int32_t one_sixth_ub_ele, int32_t c0,
                         int32_t k_h, int32_t k_w, int32_t stride_h) {
  int32_t align_output_w = 0;
  int32_t h_factor = 0;
  if (param.output_w % ALLIGN_NUM != 0) {
    align_output_w = (param.output_w / ALLIGN_NUM + 1) * ALLIGN_NUM;
  } else {
    align_output_w = param.output_w;
  }
  int32_t mask_memory = align_output_w * k_h * k_w;
  int32_t mask_factor = one_sixth_ub_ele / mask_memory;
  int32_t input_factor = (one_sixth_ub_ele / (param.pad_w * c0) - k_h) / stride_h + 1;
  if (input_factor < mask_factor) {
    h_factor = input_factor;
  } else {
    h_factor = mask_factor;
  }
  return h_factor;
}

static void CalCoreNum(TilingParam& param, int32_t total_ele, int32_t core_num) {
  if (core_num == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("MaxPoolWithArgMaxV2", "core_num value cannot be zero");
    return;
  }
  param.one_core_ele = (total_ele + core_num - 1) / core_num;
  param.act_core_num = total_ele / param.one_core_ele;
  if (total_ele % param.one_core_ele != 0) {
    ++param.act_core_num;
  }
  param.last_core_ele = total_ele - (param.act_core_num - 1) * param.one_core_ele;
}

static void CalTilingParam(TilingParam& param, const vector<int64_t>& input_shape,
                           const CompileInfoParam& compile_info_param) {
  int32_t ub_ele = compile_info_param.ub_ele;
  int32_t core_num = compile_info_param.core_num;
  int32_t ksize_h = compile_info_param.ksize_h;
  int32_t ksize_w = compile_info_param.ksize_w;
  int32_t strides_h = compile_info_param.strides_h;
  int32_t strides_w = compile_info_param.strides_w;
  int32_t ceil_mode = compile_info_param.ceil_mode;
  int32_t pad_top = compile_info_param.pad_top;
  int32_t pad_left = compile_info_param.pad_left;

  // calc output height and width, pad infos
  param.pad_t = pad_top;
  param.pad_l = pad_left;
  int32_t exact_h = param.input_h + TILING_FACTOR_2 * param.pad_t - (ksize_h - 1) - 1 +
                    ((ceil_mode == 1) ? (strides_h - 1) : 0);
  param.output_h = DivRtn(exact_h, strides_h) + 1;
  if (param.pad_t > 0) {
    if ((param.output_h - 1) * strides_h >= param.input_h + param.pad_t) {
      param.output_h = param.output_h - 1;
    }
  }
  int32_t exact_w = param.input_w + TILING_FACTOR_2 * param.pad_l - (ksize_w - 1) - 1 +
                    ((ceil_mode == 1) ? (strides_w - 1) : 0);
  param.output_w = DivRtn(exact_w, strides_w) + 1;
  if (param.pad_l > 0) {
    if ((param.output_w - 1) * strides_w >= (param.input_w + param.pad_l)) {
      param.output_w = param.output_w - 1;
    }
  }
  param.pad_h = (param.output_h - 1) * strides_h + ksize_h;
  param.pad_w = (param.output_w - 1) * strides_w + ksize_w;
  if ((param.pad_h - param.input_h - param.pad_t) >= 0) {
    param.pad_b = param.pad_h - param.input_h - param.pad_t;
  } else {
    param.pad_b = 0;
  }
  if ((param.pad_w - param.input_w - param.pad_l) >= 0) {
    param.pad_r = param.pad_w - param.input_w - param.pad_l;
  } else {
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
    int32_t one_sixth_ub_ele = ub_ele / TILING_DIVIDE_6;
    param.n_c1 = input_shape[0] * input_shape[1];
    int32_t require_memory_1 = GetRequireMemory(param, TILING_MODE_1,
                                                param.pad_h * param.pad_w * input_shape[4],
                                                ksize_h, ksize_w);
    int32_t require_memory_2 = GetRequireMemory(param, TILING_MODE_2,
                                                ksize_h * param.pad_w * input_shape[4],
                                                ksize_h, ksize_w);
    if (require_memory_1 <= one_sixth_ub_ele) {
      param.tiling_mode = TILING_MODE_1;
      CalCoreNum(param, param.n_c1, core_num);
      param.c_factor = one_sixth_ub_ele / require_memory_1;
      param.one_core_loop_num = param.one_core_ele / param.c_factor;
      param.one_core_loop_left = param.one_core_ele % param.c_factor;
      param.last_core_loop_num = param.last_core_ele / param.c_factor;
      param.last_core_loop_left = param.last_core_ele % param.c_factor;
    } else if (require_memory_2 <= one_sixth_ub_ele) {
      param.tiling_mode = TILING_MODE_2;
      param.h_factor = GetFactor(param, one_sixth_ub_ele, input_shape[4],
                                 ksize_h, ksize_w, strides_h);
      CalCoreNum(param, param.n_c1, core_num);
      param.one_core_loop_num = param.output_h / param.h_factor;
      param.one_core_loop_left = param.output_h % param.h_factor;
      param.last_core_loop_num = param.one_core_loop_num;
      param.last_core_loop_left = param.one_core_loop_left;
    } else {
      param.w_factor = (one_sixth_ub_ele / input_shape[4] / ksize_h - ksize_w) / strides_w + 1;
      param.one_core_loop_num = param.output_w / param.w_factor;
      param.one_core_loop_left = param.output_w % param.w_factor;
      param.last_core_loop_num = param.one_core_loop_num;
      param.last_core_loop_left = param.one_core_loop_left;
      param.tiling_mode = TILING_MODE_3;
      CalCoreNum(param, param.n_c1, core_num);
    }
  }
  if (param.w_factor % ALLIGN_NUM != 0) {
    param.align_w_factor = (param.w_factor / ALLIGN_NUM + 1) * ALLIGN_NUM;
  } else {
    param.align_w_factor = param.w_factor;
  }
  if (param.one_core_loop_left % ALLIGN_NUM != 0) {
    param.align_w_loop_left = (param.one_core_loop_left / ALLIGN_NUM + 1) * ALLIGN_NUM;
  } else {
    param.align_w_loop_left = param.one_core_loop_left;
  }
  if ((param.tiling_mode == TILING_MODE_3) || (param.tiling_mode == TILING_MODE_5)) {
    param.align_output_w = param.align_w_factor * param.one_core_loop_num + param.align_w_loop_left;
  } else {
    if (param.output_w % ALLIGN_NUM != 0) {
      param.align_output_w = (param.output_w / ALLIGN_NUM + 1) * ALLIGN_NUM;
    } else {
      param.align_output_w = param.output_w;
    }
  }
  param.align_output_hw = param.output_w * param.output_h;
  if (param.align_output_hw % ALLIGN_NUM != 0) {
    param.align_output_hw = (param.align_output_hw / ALLIGN_NUM + 1) * ALLIGN_NUM;
  }
  param.align_output_hw = (param.align_output_hw / ALLIGN_NUM + 1) * ALLIGN_NUM;
}

static bool GetCompileInfo(const nlohmann::json& op_info, const string& name, int32_t& value) {
  const nlohmann::json& all_vars = op_info["vars"];
  if (all_vars.empty()) {
    return false;
  }
  if (all_vars.count(name) == 0) {
    value = 0;
    OP_LOGW("Get compile info parameter failed, maybe need update om, set %s default value 0", name.c_str());
    return true;
  }
  value = all_vars[name].get<int32_t>();
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
bool MaxPoolWithArgmaxV2Tiling(const string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                               OpRunInfo& run_info) {
  OP_LOGI(op_type.c_str(), "MaxPoolWithArgmaxV2Tiling running.");

  // get and check input format and shape
  string input_format = op_paras.inputs[0].tensor[0].format;
  OP_TILING_CHECK(input_format != "NC1HWC0",
                  VECTOR_INNER_ERR_REPORT_TILIING(
                  op_type, "Get input format failed, only support NC1HWC0, but got %s.",
                  input_format.c_str()),
                  return false);
  vector<int64_t> input_shape = op_paras.inputs[0].tensor[0].shape;
  OP_TILING_CHECK(input_shape.size() != 5,
                  VECTOR_INNER_ERR_REPORT_TILIING(
                  op_type, "Get input shape failed, the length of input shape must be 5, but got %lu.",
                  input_shape.size()),
                  return false);
  OP_TILING_CHECK((input_shape[4] != 16),
                  VECTOR_INNER_ERR_REPORT_TILIING(
                  op_type, "Get input shape failed, dim 5 of input_shape must be 16, but got %lu.", input_shape[4]),
                  return false);

  CompileInfoParam compile_info_param;

  const map<string, int32_t&> compile_params = {
      {"ub_ele", compile_info_param.ub_ele},       {"core_num", compile_info_param.core_num},
      {"ksize_h", compile_info_param.ksize_h},     {"ksize_w", compile_info_param.ksize_w},
      {"strides_h", compile_info_param.strides_h}, {"strides_w", compile_info_param.strides_w},
      {"padding", compile_info_param.padding},     {"ceil_mode", compile_info_param.ceil_mode},
      {"pad_top", compile_info_param.pad_top},     {"pad_bottom", compile_info_param.pad_bottom},
      {"pad_left", compile_info_param.pad_left},   {"pad_right", compile_info_param.pad_right},
      {"global", compile_info_param.global}};
  for (auto& param : compile_params) {
    const auto& name = param.first;
    OP_LOGD(op_type.c_str(), "GetCompileInfo %s.", name.c_str());
    OP_TILING_CHECK(!GetCompileInfo(op_info, name, param.second),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetCompileInfo %s failed.", name.c_str()), return false);
    OP_LOGD(op_type.c_str(), "%s=%d.", name.c_str(), param.second);
  }
  // check compile info paramters
  OP_TILING_CHECK((compile_info_param.ub_ele <= 0), OP_LOGE(op_type.c_str(), "ub_ele must greater than 0."),
                  return false);
  OP_TILING_CHECK((compile_info_param.core_num <= 0), OP_LOGE(op_type.c_str(), "core_num must greater than 0."),
                  return false);
  OP_TILING_CHECK((compile_info_param.ksize_h <= 0) || (compile_info_param.ksize_w <= 0) ||
                  (compile_info_param.strides_h <= 0) || (compile_info_param.strides_w <= 0),
                  OP_LOGE(op_type.c_str(), "ksize and strides must greater than 0."), return false);
  // check kernel size limit
  OP_TILING_CHECK((compile_info_param.ub_ele / 6 / input_shape[4] / compile_info_param.ksize_h) <
                   compile_info_param.ksize_w,
                   VECTOR_INNER_ERR_REPORT_TILIING(op_type, "kernel_h * kernel_w exceeded limit."),
                   return false);

  // check ksize, strides and input shape
  TilingParam param;
  param.input_h = input_shape[2];
  param.input_w = input_shape[3];
  if (compile_info_param.global == 1) {
    compile_info_param.ksize_h = param.input_h;
    compile_info_param.ksize_w = param.input_w;
  }

  // calc tiling params, set tiling params, print tiling params
  CalTilingParam(param, input_shape, compile_info_param);
  // check output size for pytorch
  OP_TILING_CHECK((param.output_h <= 0) || (param.output_w <= 0),
                   VECTOR_INNER_ERR_REPORT_TILIING(
                   op_type, "Output size should be larger than zero."),
                   return false);

  OP_TILING_CHECK((param.output_h - 1) * compile_info_param.strides_h - param.pad_t >= param.input_h,
                  VECTOR_INNER_ERR_REPORT_TILIING(
                  op_type, "Last kernel is not in the scope of input feature on h dim."),
                  return false);

  OP_TILING_CHECK((param.output_w - 1) * compile_info_param.strides_w - param.pad_l >= param.input_w,
                VECTOR_INNER_ERR_REPORT_TILIING(
                op_type, "Last kernel is not in the scope of input feature on w dim."),
                return false);

  if ((compile_info_param.pad_left > 0) || (compile_info_param.pad_top > 0)) {
    OP_TILING_CHECK(((param.output_w - 1) * compile_info_param.strides_w >=
                    param.input_w + compile_info_param.pad_left) ||
                    ((param.output_h - 1) * compile_info_param.strides_h >=
                    param.input_h + compile_info_param.pad_top),
                    OP_LOGE(op_type.c_str(),
                    "Can not ensure that the last pooling starts strictly inside the image even after clip the last."),
                    return false);
  }
  SetTilingParam(param, run_info);
  PrintTilingParam(param);

  // block_dim, use fot tik op; workspace, null for tik op
  run_info.block_dim = param.act_core_num;
  vector<int64_t> workspace;
  run_info.workspaces = workspace;

  OP_LOGI(op_type.c_str(), "MaxPoolWithArgmaxV2Tiling run success.");
  return true;
}

// register tiling interface of maxpool op.
REGISTER_OP_TILING_FUNC_BUFFERED(MaxPoolWithArgmaxV2, MaxPoolWithArgmaxV2Tiling);
}  // namespace optiling
