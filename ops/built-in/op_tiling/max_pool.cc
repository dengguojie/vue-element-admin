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
 * \file max_pool.cc
 * \brief dynamic shape tiling of max_pool
 */
#include <map>
#include <nlohmann/json.hpp>

#include "../op_proto/util/error_util.h"
#include "error_log.h"
#include "op_log.h"
#include "op_tiling_util.h"
#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"

namespace {
constexpr int32_t PADDING_VALUE = 2;
constexpr int32_t TILING_MODE_6 = 6;
constexpr int32_t TILING_MODE_7 = 7;
}  // namespace

namespace optiling {
using namespace ge;
using namespace std;

static const std::vector<std::string> COMPILE_INFO_KEY = {
    "ub_ele",    "core_num", "ksize_h",    "ksize_w",  "strides_h", "strides_w", "padding",
    "ceil_mode", "pad_top",  "pad_bottom", "pad_left", "pad_right", "global"};

static const std::map<std::string, std::int64_t> OPTIONAL_VALUE = {
    {"ub_ele", 0},    {"core_num", 0},  {"ksize_h", 0},   {"ksize_w", 0}, {"strides_h", 0},
    {"strides_w", 0}, {"padding", 0},   {"ceil_mode", 0}, {"pad_top", 0}, {"pad_bottom", 0},
    {"pad_left", 0},  {"pad_right", 0}, {"global", 0}};

const int64_t INDEX_0 = 0;
const int64_t INDEX_1 = 1;
const int64_t INDEX_2 = 2;
const int64_t INDEX_3 = 3;
const int64_t INDEX_4 = 4;
const int64_t INDEX_5 = 5;
const int64_t INDEX_6 = 6;
const int64_t INDEX_7 = 7;
const int64_t INDEX_8 = 8;
const int64_t INDEX_9 = 9;
const int64_t INDEX_10 = 10;
const int64_t INDEX_11 = 11;
const int64_t INDEX_12 = 12;
const int64_t MODE_16 = 16;

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

struct CompileInfoParam {
  // get compile info
  int32_t ub_ele = 0;
  int32_t core_num = 1;
  int32_t ksize_h = 1;
  int32_t ksize_w = 1;
  int32_t strides_h = 1;
  int32_t strides_w = 1;
  int32_t padding = 0;    // SAME
  int32_t ceil_mode = 0;  // floor
  int32_t pad_top = 0;
  int32_t pad_bottom = 0;
  int32_t pad_left = 0;
  int32_t pad_right = 0;
  int32_t global = 0;
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

static void SetTilingParam(const TilingParam& param, utils::OpRunInfo& run_info) {
  run_info.AddTilingData(param.tiling_mode);
  run_info.AddTilingData(param.act_core_num);
  run_info.AddTilingData(param.one_core_ele);
  run_info.AddTilingData(param.last_core_ele);
  run_info.AddTilingData(param.input_h);
  run_info.AddTilingData(param.input_w);
  run_info.AddTilingData(param.output_h);
  run_info.AddTilingData(param.output_w);
  run_info.AddTilingData(param.pad_h);
  run_info.AddTilingData(param.pad_w);
  run_info.AddTilingData(param.pad_t);
  run_info.AddTilingData(param.pad_b);
  run_info.AddTilingData(param.pad_l);
  run_info.AddTilingData(param.pad_r);
  run_info.AddTilingData(param.c_factor);
  run_info.AddTilingData(param.h_factor);
  run_info.AddTilingData(param.w_factor);
  run_info.AddTilingData(param.one_core_loop_num);
  run_info.AddTilingData(param.one_core_loop_left);
  run_info.AddTilingData(param.last_core_loop_num);
  run_info.AddTilingData(param.last_core_loop_left);
  run_info.AddTilingData(param.n_c1);
}

static void CalCoreNum(TilingParam& param, int32_t total_ele, int32_t core_num) {
  OP_TILING_CHECK(core_num == 0, VECTOR_INNER_ERR_REPORT_TILIING("max_pool", "core_num = 0 is not support"), return);
  param.one_core_ele = (total_ele + core_num - 1) / core_num;
  param.act_core_num = total_ele / param.one_core_ele;
  if (total_ele % param.one_core_ele != 0) {
    param.act_core_num = param.act_core_num + 1;
  }
  param.last_core_ele = total_ele - (param.act_core_num - 1) * param.one_core_ele;
}

static void CalTilingParam(TilingParam& param, const GeShape& input_shape, const CompileInfoParam& compile_info_param) {
  int32_t ub_ele = compile_info_param.ub_ele;
  int32_t core_num = compile_info_param.core_num;
  int32_t ksize_h = compile_info_param.ksize_h;
  int32_t ksize_w = compile_info_param.ksize_w;
  int32_t strides_h = compile_info_param.strides_h;
  int32_t strides_w = compile_info_param.strides_w;
  int32_t padding = compile_info_param.padding;      // SAME
  int32_t ceil_mode = compile_info_param.ceil_mode;  // floor
  int32_t pad_top = compile_info_param.pad_top;
  int32_t pad_bottom = compile_info_param.pad_bottom;
  int32_t pad_left = compile_info_param.pad_left;
  int32_t pad_right = compile_info_param.pad_right;
  OP_TILING_CHECK(strides_h == 0, VECTOR_INNER_ERR_REPORT_TILIING("max_pool", "strides_h = 0 is not support"), return);
  OP_TILING_CHECK(strides_w == 0, VECTOR_INNER_ERR_REPORT_TILIING("max_pool", "strides_w = 0 is not support"), return);
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
  } else if (padding == PADDING_VALUE) {
    if (ceil_mode == 1) {
      param.output_h = (param.input_h + pad_top + pad_bottom - ksize_h + strides_h + strides_h - 1) / strides_h;
      param.output_w = (param.input_w + pad_left + pad_right - ksize_w + strides_w + strides_w - 1) / strides_w;
      if (pad_top != 0 || pad_left != 0) {
        if ((param.output_h - 1) * strides_h >= param.input_h + pad_top) {
          param.output_h -= 1;
        }
        if ((param.output_w - 1) * strides_w >= param.input_w + pad_left) {
          param.output_w -= 1;
        }
      }
      param.pad_h = (param.output_h - 1) * strides_h + ksize_h;
      param.pad_w = (param.output_w - 1) * strides_w + ksize_w;
      param.pad_t = pad_top;
      param.pad_b = (param.pad_h - param.input_h - pad_top) > 0 ? (param.pad_h - param.input_h - pad_top) : 0;
      param.pad_l = pad_left;
      param.pad_r = (param.pad_w - param.input_w - pad_left) > 0 ? (param.pad_w - param.input_w - pad_left) : 0;
    } else {
      param.output_h = (param.input_h + pad_top + pad_bottom - ksize_h + strides_h) / strides_h;
      param.output_w = (param.input_w + pad_left + pad_right - ksize_w + strides_w) / strides_w;
      if (pad_top != 0 || pad_left != 0) {
        if ((param.output_h - 1) * strides_h >= param.input_h + pad_top) {
          param.output_h -= 1;
        }
        if ((param.output_w - 1) * strides_w >= param.input_w + pad_left) {
          param.output_w -= 1;
        }
      }
      param.pad_h = (param.output_h - 1) * strides_h + ksize_h;
      param.pad_w = (param.output_w - 1) * strides_w + ksize_w;
      param.pad_t = pad_top;
      param.pad_b = (param.pad_h - param.input_h - pad_top) > 0 ? (param.pad_h - param.input_h - pad_top) : 0;
      param.pad_l = pad_left;
      param.pad_r = (param.pad_w - param.input_w - pad_left) > 0 ? (param.pad_w - param.input_w - pad_left) : 0;
    }
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
  // global pooling max_pool_v3
  if (ksize_h == param.input_h && ksize_w == param.input_w) {
    param.n_c1 = input_shape.GetDim(INDEX_0) * input_shape.GetDim(INDEX_1);
    CalCoreNum(param, param.n_c1, core_num);
    if (ub_ele >= (input_shape.GetDim(INDEX_2) * input_shape.GetDim(INDEX_3) * input_shape.GetDim(INDEX_4))) {
      param.tiling_mode = TILING_MODE_6;
    } else {
      param.h_factor = ub_ele / input_shape.GetDim(INDEX_4);  // acutal is hw_factor
      int32_t input_hw_num = param.input_h * param.input_w;
      param.one_core_loop_num = input_hw_num / param.h_factor;
      // dif from other tiling mode,this is used to tiling hw
      param.one_core_loop_left = input_hw_num % param.h_factor;
      param.last_core_loop_num = param.one_core_loop_num;
      param.last_core_loop_left = param.one_core_loop_left;
      param.tiling_mode = TILING_MODE_7;
    }
    return;
  }
  if ((ksize_h == 1) && (ksize_w == 1) && (strides_h == 1) && (strides_w == 1)) {
    param.tiling_mode = 0;
    int32_t max_ele = ub_ele / input_shape.GetDim(INDEX_4);
    int32_t total_ele = input_shape.GetDim(INDEX_0) * input_shape.GetDim(INDEX_1) * input_shape.GetDim(INDEX_2) *
                        input_shape.GetDim(INDEX_3);
    CalCoreNum(param, total_ele, core_num);
    param.one_core_loop_num = param.one_core_ele / max_ele;
    param.one_core_loop_left = param.one_core_ele % max_ele;
    param.last_core_loop_num = param.last_core_ele / max_ele;
    param.last_core_loop_left = param.last_core_ele % max_ele;
  } else {
    int32_t one_sixth_ub_ele = ub_ele / 6;
    param.n_c1 = input_shape.GetDim(INDEX_0) * input_shape.GetDim(INDEX_1);
    if (param.pad_h * param.pad_w * input_shape.GetDim(INDEX_4) <= one_sixth_ub_ele) {
      param.tiling_mode = 1;
      CalCoreNum(param, param.n_c1, core_num);
      param.c_factor = one_sixth_ub_ele / (param.pad_h * param.pad_w * input_shape.GetDim(INDEX_4));
      param.one_core_loop_num = param.one_core_ele / param.c_factor;
      param.one_core_loop_left = param.one_core_ele % param.c_factor;
      param.last_core_loop_num = param.last_core_ele / param.c_factor;
      param.last_core_loop_left = param.last_core_ele % param.c_factor;
    } else if (ksize_h * param.pad_w * input_shape.GetDim(INDEX_4) <= one_sixth_ub_ele) {
      param.h_factor = (one_sixth_ub_ele / (param.pad_w * input_shape.GetDim(INDEX_4)) - ksize_h) / strides_h + 1;
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
      param.w_factor = (one_sixth_ub_ele / input_shape.GetDim(INDEX_4) / ksize_h - ksize_w) / strides_w + 1;
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

bool GetCompileInfo(const std::string& opType, const std::vector<int64_t>& opCompileInfo,
                    CompileInfoParam& compile_info_param) {
  OP_TILING_CHECK(
      opCompileInfo.size() != COMPILE_INFO_KEY.size(),
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "the compile info num is not equal expect compile_info(%zu), is %zu",
                                      COMPILE_INFO_KEY.size(), opCompileInfo.size()),
      return false);

  compile_info_param.ub_ele = static_cast<int32_t>(opCompileInfo[INDEX_0]);
  compile_info_param.core_num = static_cast<int32_t>(opCompileInfo[INDEX_1]);
  compile_info_param.ksize_h = static_cast<int32_t>(opCompileInfo[INDEX_2]);
  compile_info_param.ksize_w = static_cast<int32_t>(opCompileInfo[INDEX_3]);
  compile_info_param.strides_h = static_cast<int32_t>(opCompileInfo[INDEX_4]);
  compile_info_param.strides_w = static_cast<int32_t>(opCompileInfo[INDEX_5]);
  compile_info_param.padding = static_cast<int32_t>(opCompileInfo[INDEX_6]);
  compile_info_param.ceil_mode = static_cast<int32_t>(opCompileInfo[INDEX_7]);
  compile_info_param.pad_top = static_cast<int32_t>(opCompileInfo[INDEX_8]);
  compile_info_param.pad_bottom = static_cast<int32_t>(opCompileInfo[INDEX_9]);
  compile_info_param.pad_left = static_cast<int32_t>(opCompileInfo[INDEX_10]);
  compile_info_param.pad_right = static_cast<int32_t>(opCompileInfo[INDEX_11]);
  compile_info_param.global = static_cast<int32_t>(opCompileInfo[INDEX_12]);

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
bool MaxPoolTiling(const string& op_type, const ge::Operator& op_paras, const std::vector<int64_t>& op_info,
                   utils::OpRunInfo& run_info) {
  OP_LOGI(op_type.c_str(), "MaxPoolTiling running.");
  PROFILING_TILING_INIT(op_type.c_str());
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get OpDesc failed."),
                  return false);

  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get InputDesc failed."),
                  return false);
  // get and check input format and shape
  ge::Format input_format = input_desc->GetFormat();
  OP_TILING_CHECK(input_format != FORMAT_NC1HWC0,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get input format failed, only support NC1HWC0, but got %s.",
                                                  to_string(input_format).c_str()),
                  return false);

  const GeShape& input_shape = input_desc->MutableShape();
  OP_TILING_CHECK(input_shape.GetDimNum() != 5,
                  VECTOR_INNER_ERR_REPORT_TILIING(
                      op_type, "Get input shape failed, the length of input shape must be 5, but got %lu.",
                      input_shape.GetDimNum()),
                  return false);

  OP_TILING_CHECK(
      input_shape.GetDim(INDEX_4) != MODE_16,
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get input shape failed, dim 5 of input_shape must be 16, but got %lu.",
                                      input_shape.GetDim(INDEX_4)),
      return false);
  PROFILING_TILING_AFTER_GET_SHAPE_REG();

  CompileInfoParam compile_info_param;
  // get compile info paramters
  OP_TILING_CHECK(!GetCompileInfo(op_type, op_info, compile_info_param),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetCompileInfo errror."), return false);

  // check compile info paramters
  OP_TILING_CHECK((compile_info_param.ub_ele <= 0), OP_LOGE(op_type.c_str(), "ub_ele must greater than 0."),
                  return false);
  OP_TILING_CHECK((compile_info_param.core_num <= 0), OP_LOGE(op_type.c_str(), "core_num must greater than 0."),
                  return false);
  OP_TILING_CHECK((compile_info_param.ksize_h <= 0) || (compile_info_param.ksize_w <= 0) ||
                      (compile_info_param.strides_h <= 0) || (compile_info_param.strides_w <= 0),
                  OP_LOGE(op_type.c_str(), "ksize and strides must greater than 0."), return false);
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  // check ksize, strides and input shape
  TilingParam param;
  param.input_h = input_shape.GetDim(INDEX_2);
  param.input_w = input_shape.GetDim(INDEX_3);
  if (compile_info_param.global == 1) {
    compile_info_param.ksize_h = param.input_h;
    compile_info_param.ksize_w = param.input_w;
  }
  OP_TILING_CHECK((compile_info_param.padding == 1) &&
                      ((compile_info_param.ksize_h > param.input_h) || (compile_info_param.ksize_w > param.input_w)),
                  VECTOR_INNER_ERR_REPORT_TILIING(
                      op_type, "Input height or width must greater than or equal to ksize when padding mode is valid."),
                  return false);

  // calc tiling params, set tiling params, print tiling params
  CalTilingParam(param, input_shape, compile_info_param);
  if ((compile_info_param.pad_left > 0) || (compile_info_param.pad_top > 0)) {
    OP_TILING_CHECK(
        ((param.output_w - 1) * compile_info_param.strides_w >= param.input_w + compile_info_param.pad_left) ||
            ((param.output_h - 1) * compile_info_param.strides_h >= param.input_h + compile_info_param.pad_top),
        OP_LOGE(op_type.c_str(),
                "Can not ensure that the last pooling starts strictly inside the image even after clip the last."),
        return false);
  }
  PROFILING_TILING_AFTER_CALCU_TILING_REG();

  SetTilingParam(param, run_info);
  PrintTilingParam(param);

  // block_dim, use fot tik op; workspace, null for tik op
  run_info.SetBlockDim(param.act_core_num);
  PROFILING_TILING_END();
  OP_LOGI(op_type.c_str(), "MaxPoolTiling run success.");
  return true;
}

// register tiling interface of maxpool op.
REGISTER_OP_TILING_V3_WITH_VECTOR(MaxPool, MaxPoolTiling, COMPILE_INFO_KEY, OPTIONAL_VALUE);
REGISTER_OP_TILING_V3_WITH_VECTOR(MaxPoolV3, MaxPoolTiling, COMPILE_INFO_KEY, OPTIONAL_VALUE);
}  // namespace optiling
