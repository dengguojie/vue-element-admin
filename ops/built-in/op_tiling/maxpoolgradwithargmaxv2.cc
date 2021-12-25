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
 * \file maxpoolgradwithargmaxv2.cc
 * \brief dynamic shape tiling of maxpoolgradwithargmaxv2
 */
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"
#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "error_log.h"

namespace {
  constexpr int32_t TILING_MODE_0 = 0;
  constexpr int32_t TILING_MODE_1 = 1;
  constexpr int32_t TILING_MODE_2 = 2;
  constexpr int32_t TILING_FACTOR_TWO = 2;
  constexpr int32_t C0 = 16;
  constexpr int32_t ALLING_MASK_128 = 128;
  constexpr int32_t INPUT_X_INDEX = 0;
  constexpr int32_t INPUT_GRAD_INDEX = 1;
  constexpr int32_t INPUT_ARGMAX_INDEX = 2;
  constexpr int32_t SHAPE_INDEX_N = 0;
  constexpr int32_t SHAPE_INDEX_C1 = 1;
  constexpr int32_t SHAPE_INDEX_H = 2;
  constexpr int32_t SHAPE_INDEX_W = 3;
  constexpr int64_t DTYPE_SIZE_FP32 = 4;
  constexpr int32_t BLOCK_ALLIGN = 16;
}

namespace optiling {
using namespace ge;
using namespace std;

struct MaxPoolGradWithArgmaxV2TilingParams {
  int32_t tiling_mode;
  int32_t real_block;
  int32_t block_cycle;
  int32_t ho_wo_16;
  int32_t mask_shape_128;
  int32_t pad_left;
  int32_t pad_right;
  int32_t pad_top;
  int32_t pad_bottom;
  int32_t each_process_wo;
  int32_t each_process_ho;
  int32_t each_process_wi;
  int32_t each_process_hi;
  int32_t c1;
  int32_t ho;
  int32_t wo;
  int32_t hi;
  int32_t wi;
  int32_t nc1;
  int32_t block_num;
  int32_t block_num_inner;
  int32_t block_num_outer;
  int32_t ho_inner;
  int32_t ho_outer;
  int32_t block;
  int32_t act_core_num;
  int32_t tile_h_to_block;
  int32_t if_block;
  int32_t shape_ho;
  int32_t shape_hi;
  int32_t one_window_size;
};

struct CompileInfoParams {
  int32_t ub_ele;
  int32_t core_num;
  int32_t kh;
  int32_t kw;
  int32_t stride_h;
  int32_t stride_w;
  int32_t pad_h;
  int32_t pad_w;
  int32_t dilation_h;
  int32_t dilation_w;
  int32_t ceil_mode;
};

static void InitTilingParams(MaxPoolGradWithArgmaxV2TilingParams& params) {
  params.tiling_mode = 0;
  params.real_block = 0;
  params.block_cycle = 0;
  params.ho_wo_16 = 0;
  params.mask_shape_128 = 0;
  params.pad_left = 0;
  params.pad_right = 0;
  params.pad_top = 0;
  params.pad_bottom = 0;
  params.each_process_wo = 0;
  params.each_process_ho = 0;
  params.each_process_wi = 0;
  params.each_process_hi = 0;
  params.c1 = 0;
  params.ho = 0;
  params.wo = 0;
  params.hi = 0;
  params.wi = 0;
  params.nc1 = 0;
  params.block_num = 0;
  params.block_num_inner = 0;
  params.block_num_outer = 0;
  params.ho_inner = 0;
  params.ho_outer = 0;
  params.block = 0;
  params.act_core_num = 0;
  params.tile_h_to_block = 0;
  params.if_block = 0;
  params.shape_ho = 0;
  params.shape_hi = 0;
  params.one_window_size = 0;
}

static bool GetCompileInfo(const std::string& op_type, const nlohmann::json& op_compile_info,
                           CompileInfoParams& compile_params) {
  using namespace nlohmann;
  auto all_vars = op_compile_info["vars"];
  if (all_vars.count("core_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("MaxPoolGradWithArgmaxV2Tiling", "GetCompileInfo, get core_num error");
    return false;
  }
  compile_params.ub_ele = all_vars["ub_ele"].get<std::int32_t>();
  compile_params.core_num = all_vars["core_num"].get<std::int32_t>();
  compile_params.kh = all_vars["kh"].get<std::int32_t>();
  compile_params.kw = all_vars["kw"].get<std::int32_t>();
  compile_params.stride_h = all_vars["stride_h"].get<std::int32_t>();
  compile_params.stride_w = all_vars["stride_w"].get<std::int32_t>();
  compile_params.pad_h = all_vars["pad_h"].get<std::int32_t>();
  compile_params.pad_w = all_vars["pad_w"].get<std::int32_t>();
  compile_params.dilation_h = all_vars["dilation_h"].get<std::int32_t>();
  compile_params.dilation_w = all_vars["dilation_w"].get<std::int32_t>();
  compile_params.ceil_mode = all_vars["ceil_mode"].get<std::int32_t>();
  return true;
}

static int32_t DivRtn(int32_t x, int32_t y) {
  if (y == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("MaxPoolGradWithArgMaxV2", "y value cannot be zero");
    return 0;
  }
  int32_t q = x / y;
  int32_t r = x % y;
  if ((r != 0) && ((r < 0) != (y < 0))) {
    q = q - 1;
  }
  return q;
}

static int32_t CeilDiv(int32_t num, int32_t divisor) {
  if (divisor == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("MaxPoolGradWithArgMaxV2", "divisor value cannot be zero");
    return 0;
  }
  if (num % divisor != 0) {
    return num / divisor + 1;
  }
  return num / divisor;
}

static void TilingFactor(MaxPoolGradWithArgmaxV2TilingParams& tiling_params, const CompileInfoParams& compile_info) {
  int32_t one_seventh_ub_ele = compile_info.ub_ele / 7;
  int32_t col2img_process_wo;
  int32_t grad_process_wo;
  // for one output pixel
  if (compile_info.stride_h > compile_info.kh) {
    tiling_params.each_process_hi = compile_info.stride_h;
  } else {
    tiling_params.each_process_hi = compile_info.kh;
  }
  int32_t col2img_process_wi = one_seventh_ub_ele / (tiling_params.each_process_hi * C0);
  if (compile_info.kw > compile_info.stride_w) {
    col2img_process_wo = (col2img_process_wi - compile_info.kw) / compile_info.stride_w + 1;
  } else {
    col2img_process_wo = col2img_process_wi / compile_info.stride_w;
  }
  grad_process_wo = one_seventh_ub_ele / C0;
  // in some case, grad_process_wo is smaller than col2img_process_wo
  if (col2img_process_wo < grad_process_wo) {
    tiling_params.each_process_wo = col2img_process_wo;
  } else {
    tiling_params.each_process_wo = grad_process_wo;
  }
  if (tiling_params.each_process_wo == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("max_pool_grad_with_argmax_v2",
                                    "kernel size or stride is too large and not support.");
  }

  if (tiling_params.each_process_wo >= tiling_params.wo) {
    // for the shape of col2img_ub
    int32_t wi = tiling_params.wi + tiling_params.pad_left + tiling_params.pad_right;
    tiling_params.each_process_hi = one_seventh_ub_ele / (wi * C0);
    // calc each_process_ho by each_process_hi
    if (compile_info.kh > compile_info.stride_h) {
      tiling_params.each_process_ho = (tiling_params.each_process_hi - compile_info.kh) / compile_info.stride_h + 1;
    } else {
      tiling_params.each_process_ho = tiling_params.each_process_hi / compile_info.stride_h;
    }

    tiling_params.each_process_wo = tiling_params.wo;
  } else {
    tiling_params.each_process_ho = 0;
  }

  if (tiling_params.each_process_ho >= tiling_params.ho) {
    // not tiling branch
    tiling_params.tiling_mode = TILING_MODE_0;
    tiling_params.each_process_ho = tiling_params.ho;
    tiling_params.each_process_wo = 0;
  } else if (tiling_params.each_process_ho > 1) {
    // tiling ho only branch
    tiling_params.tiling_mode = TILING_MODE_1;
    tiling_params.each_process_wo = 0;
  } else {
    // tiling wo branch, save temp overlap on gm
    tiling_params.each_process_ho = 1;
    tiling_params.tiling_mode = TILING_MODE_2;
  }
}

static bool IfBlock(const MaxPoolGradWithArgmaxV2TilingParams& tiling_params, const CompileInfoParams& compile_info,
                    const int32_t& ho_outer, const int32_t& ho_inner) {
  if (ho_inner <= 1) {
    return false;
  }
  if (compile_info.stride_h >= compile_info.kh) {
    return true;
  }
  int32_t overlap_num = ceil((compile_info.kh - compile_info.stride_h) * 1.0 / compile_info.stride_h);
  int32_t times = ceil(ho_inner * 1.0 / tiling_params.each_process_ho);
  int32_t overlaps = overlap_num * times;

  if ((overlaps + ho_inner) * 1.0 / ho_inner >= ho_outer) {
    return false;
  }
  return true;
}

static void CalTilingParam(MaxPoolGradWithArgmaxV2TilingParams& tiling_params, CompileInfoParams& compile_info,
                           std::vector<int64_t> grad_shape, std::vector<int64_t> input_shape) {
  // shape params
  int32_t hi = input_shape[SHAPE_INDEX_H];
  int32_t wi = input_shape[SHAPE_INDEX_W];
  int32_t ho = grad_shape[SHAPE_INDEX_H];
  int32_t wo = grad_shape[SHAPE_INDEX_W];
  int32_t n = grad_shape[SHAPE_INDEX_N];
  int32_t c1 = grad_shape[SHAPE_INDEX_C1];
  // tiling params
  tiling_params.hi = hi;
  tiling_params.wi = wi;
  tiling_params.c1 = c1;
  // compile info params
  int32_t core_num = compile_info.core_num;
  int32_t kh = compile_info.kh;
  int32_t kw = compile_info.kw;
  int32_t stride_h = compile_info.stride_h;
  int32_t stride_w = compile_info.stride_w;
  int32_t pad_top = compile_info.pad_h;
  int32_t pad_left = compile_info.pad_w;
  int32_t ceil_mode = compile_info.ceil_mode;

  OP_TILING_CHECK(stride_h == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("max_pool_grad_with_argmax_v2", "stride_h = 0 is not support"),
                  return);
  OP_TILING_CHECK(stride_w == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("max_pool_grad_with_argmax_v2", "stride_w = 0 is not support"),
                  return);

  // calc output height and width, pad infos
  tiling_params.pad_top = pad_top;
  tiling_params.pad_left = pad_left;
  int32_t exact_h =
      tiling_params.hi + TILING_FACTOR_TWO * tiling_params.pad_top - (kh - 1) - 1 + ((ceil_mode == 1) ? (stride_h - 1) : 0);
  tiling_params.ho = DivRtn(exact_h, stride_h) + 1;
  if (tiling_params.pad_top > 0) {
    if ((tiling_params.ho - 1) * stride_h >= tiling_params.hi + tiling_params.pad_top) {
      tiling_params.ho = tiling_params.ho - 1;
    }
  }
  int32_t exact_w =
      tiling_params.wi + TILING_FACTOR_TWO * tiling_params.pad_left - (kw - 1) - 1 + ((ceil_mode == 1) ? (stride_w - 1) : 0);
  tiling_params.wo = DivRtn(exact_w, stride_w) + 1;
  if (tiling_params.pad_left > 0) {
    if ((tiling_params.wo - 1) * stride_w >= (tiling_params.wi + tiling_params.pad_left)) {
      tiling_params.wo = tiling_params.wo - 1;
    }
  }
  int32_t pad_h = (tiling_params.ho - 1) * stride_h + kh;
  int32_t pad_w = (tiling_params.wo - 1) * stride_w + kw;
  if ((pad_h - tiling_params.hi - tiling_params.pad_top) >= 0) {
    tiling_params.pad_bottom = pad_h - tiling_params.hi - tiling_params.pad_top;
  } else {
    tiling_params.pad_bottom = 0;
  }
  if ((pad_w - tiling_params.wi - tiling_params.pad_left) >= 0) {
    tiling_params.pad_right = pad_w - tiling_params.wi - tiling_params.pad_left;
  } else {
    tiling_params.pad_right = 0;
  }

  if (tiling_params.ho != ho) {
    VECTOR_INNER_ERR_REPORT_TILIING("max_pool_grad_with_argmax_v2", "Wrong ori_output shape");
    VECTOR_INNER_ERR_REPORT_TILIING("max_pool_grad_with_argmax_v2", "ho is %d", ho);
    VECTOR_INNER_ERR_REPORT_TILIING("max_pool_grad_with_argmax_v2", "tiling_params.ho is %d", tiling_params.ho);
  }
  if (tiling_params.wo != wo) {
    VECTOR_INNER_ERR_REPORT_TILIING("max_pool_grad_with_argmax_v2", "Wrong ori_output shape");
    VECTOR_INNER_ERR_REPORT_TILIING("max_pool_grad_with_argmax_v2", "wo is %d", wo);
    VECTOR_INNER_ERR_REPORT_TILIING("max_pool_grad_with_argmax_v2", "tiling_params.wo is %d", tiling_params.wo);
  }

  // get block num
  tiling_params.block_num = n * c1;
  if (tiling_params.block_num > core_num) {
    tiling_params.real_block = core_num;
    tiling_params.block_cycle = (tiling_params.block_num + core_num - 1) / core_num;
  } else {
    tiling_params.real_block = tiling_params.block_num;
    tiling_params.block_cycle = 1;
  }

  // tiling factor
  if (tiling_params.real_block == core_num) {
    tiling_params.tile_h_to_block = 0;
    // calc tiling mode
    TilingFactor(tiling_params, compile_info);
    // set ceil Scalars
    if (tiling_params.tiling_mode == 0) {
      tiling_params.ho_wo_16 = CeilDiv(tiling_params.ho * tiling_params.wo, C0);
    } else if (tiling_params.tiling_mode == 1) {
      tiling_params.ho_wo_16 = CeilDiv(tiling_params.each_process_ho * tiling_params.wo, C0);
    } else {
      tiling_params.ho_wo_16 = CeilDiv(tiling_params.each_process_wo, C0);
    }
    tiling_params.mask_shape_128 = ALLING_MASK_128 * CeilDiv(tiling_params.ho_wo_16 * C0, ALLING_MASK_128);
    tiling_params.one_window_size = C0 * (CeilDiv(tiling_params.ho * tiling_params.wo, C0) + 1);
  } else {
    // get core divlist
    std::vector<int32_t> div_list;
    for (int i = 1; i < core_num + 1; i++) {
      if (core_num % i == 0) {
        if (std::find(div_list.begin(), div_list.end(), core_num / i) == div_list.end()) {
          div_list.push_back(core_num / i);
        }
      }
    }
    for (auto& iter : div_list) {
      if (tiling_params.block_num >= iter) {
        if (tiling_params.ho >= core_num / iter) {
          tiling_params.block_num_outer = iter;
          tiling_params.block_num_inner = (tiling_params.block_num + iter - 1) / iter;
          break;
        }
      }
    }
    if (tiling_params.block_num * tiling_params.ho < core_num) {
      tiling_params.ho_outer = tiling_params.ho;
      tiling_params.block_num_outer = tiling_params.block_num;
      tiling_params.block_num_inner = 1;
    } else {
      if (tiling_params.block_num_outer == 0) {
        tiling_params.ho_outer = core_num / tiling_params.block_num;
        tiling_params.block_num_outer = tiling_params.block_num;
        tiling_params.block_num_inner = 1;
      } else {
        tiling_params.ho_outer = core_num / tiling_params.block_num_outer;
      }
    }
    tiling_params.ho_inner = ceil(tiling_params.ho * 1.0 / tiling_params.ho_outer);
    tiling_params.ho_outer = ceil(tiling_params.ho * 1.0 / tiling_params.ho_inner);

    bool if_block = IfBlock(tiling_params, compile_info, tiling_params.ho_outer, tiling_params.ho_inner);
    if (if_block) {
      tiling_params.if_block = 1;
      tiling_params.tile_h_to_block = 1;
      // tiling factor
      int32_t overlap = kh - stride_h;
      int32_t overlap_num = ceil(overlap * 1.0 / stride_h);
      if (kh > stride_h) {
        tiling_params.shape_ho = tiling_params.ho_inner + overlap_num;
        tiling_params.shape_hi = (tiling_params.ho_inner + overlap_num - 1) * stride_h + kh;
      } else {
        tiling_params.shape_ho = tiling_params.ho_inner;
        tiling_params.shape_hi = tiling_params.ho_inner * stride_h;
      }
      if ((tiling_params.hi - tiling_params.ho_inner * stride_h * tiling_params.ho_outer) > 0) {
        tiling_params.shape_hi =
            tiling_params.shape_hi + (tiling_params.hi - tiling_params.ho_inner * stride_h * tiling_params.ho_outer);
      }
      // calc tiling mode
      TilingFactor(tiling_params, compile_info);
      // set ceil Scalars
      if (tiling_params.tiling_mode == 0) {
        tiling_params.ho_wo_16 = CeilDiv(tiling_params.shape_ho * tiling_params.wo, C0);
      } else if (tiling_params.tiling_mode == 1) {
        tiling_params.ho_wo_16 = CeilDiv(tiling_params.each_process_ho * tiling_params.wo, C0);
      } else {
        tiling_params.ho_wo_16 = CeilDiv(tiling_params.each_process_wo, C0);
      }
      tiling_params.mask_shape_128 = ALLING_MASK_128 * CeilDiv(tiling_params.ho_wo_16 * C0, ALLING_MASK_128);
      tiling_params.one_window_size = C0 * (CeilDiv(tiling_params.ho * tiling_params.wo, C0) + 1);
    } else {
      tiling_params.if_block = 0;
      tiling_params.nc1 = n * c1;
      tiling_params.block = core_num;
      while (tiling_params.nc1 % tiling_params.block != 0) {
        tiling_params.block = tiling_params.block - 1;
      }
      tiling_params.nc1 = tiling_params.nc1 / tiling_params.block;
      // calc tiling mode
      TilingFactor(tiling_params, compile_info);
      // set ceil Scalars
      if (tiling_params.tiling_mode == 0) {
        tiling_params.ho_wo_16 = CeilDiv(tiling_params.ho * tiling_params.wo, C0);
      } else if (tiling_params.tiling_mode == 1) {
        tiling_params.ho_wo_16 = CeilDiv(tiling_params.each_process_ho * tiling_params.wo, C0);
      } else {
        tiling_params.ho_wo_16 = CeilDiv(tiling_params.each_process_wo, C0);
      }
      tiling_params.mask_shape_128 = ALLING_MASK_128 * CeilDiv(tiling_params.ho_wo_16 * C0, ALLING_MASK_128);
      tiling_params.one_window_size = C0 * (CeilDiv(tiling_params.ho * tiling_params.wo, C0) + 1);
    }
  }
  // set loop params
  if (tiling_params.real_block == core_num) {
    tiling_params.act_core_num = CeilDiv(n * c1, tiling_params.block_cycle);
  } else {
    if (tiling_params.if_block == 1) {
      tiling_params.act_core_num = tiling_params.block_num_outer * tiling_params.ho_outer;
    } else {
      tiling_params.act_core_num = tiling_params.block;
    }
  }
}

static void SetTilingParam(const MaxPoolGradWithArgmaxV2TilingParams& tiling_params, OpRunInfo& run_info) {
  ByteBufferPut(run_info.tiling_data, tiling_params.tiling_mode);
  ByteBufferPut(run_info.tiling_data, tiling_params.real_block);
  ByteBufferPut(run_info.tiling_data, tiling_params.block_cycle);
  ByteBufferPut(run_info.tiling_data, tiling_params.ho_wo_16);
  ByteBufferPut(run_info.tiling_data, tiling_params.mask_shape_128);
  ByteBufferPut(run_info.tiling_data, tiling_params.pad_left);
  ByteBufferPut(run_info.tiling_data, tiling_params.pad_right);
  ByteBufferPut(run_info.tiling_data, tiling_params.pad_top);
  ByteBufferPut(run_info.tiling_data, tiling_params.pad_bottom);
  ByteBufferPut(run_info.tiling_data, tiling_params.each_process_wo);
  ByteBufferPut(run_info.tiling_data, tiling_params.each_process_ho);
  ByteBufferPut(run_info.tiling_data, tiling_params.each_process_wi);
  ByteBufferPut(run_info.tiling_data, tiling_params.each_process_hi);
  ByteBufferPut(run_info.tiling_data, tiling_params.c1);
  ByteBufferPut(run_info.tiling_data, tiling_params.ho);
  ByteBufferPut(run_info.tiling_data, tiling_params.wo);
  ByteBufferPut(run_info.tiling_data, tiling_params.hi);
  ByteBufferPut(run_info.tiling_data, tiling_params.wi);
  ByteBufferPut(run_info.tiling_data, tiling_params.nc1);
  ByteBufferPut(run_info.tiling_data, tiling_params.block_num);
  ByteBufferPut(run_info.tiling_data, tiling_params.block_num_inner);
  ByteBufferPut(run_info.tiling_data, tiling_params.block_num_outer);
  ByteBufferPut(run_info.tiling_data, tiling_params.ho_inner);
  ByteBufferPut(run_info.tiling_data, tiling_params.ho_outer);
  ByteBufferPut(run_info.tiling_data, tiling_params.block);
  ByteBufferPut(run_info.tiling_data, tiling_params.act_core_num);
  ByteBufferPut(run_info.tiling_data, tiling_params.tile_h_to_block);
  ByteBufferPut(run_info.tiling_data, tiling_params.if_block);
  ByteBufferPut(run_info.tiling_data, tiling_params.shape_ho);
  ByteBufferPut(run_info.tiling_data, tiling_params.shape_hi);
  ByteBufferPut(run_info.tiling_data, tiling_params.one_window_size);
}

static void PrintTilingParam(const MaxPoolGradWithArgmaxV2TilingParams& tiling_params) {
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "tiling_mode=%d.", tiling_params.tiling_mode);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "real_block=%d.", tiling_params.real_block);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "block_cycle=%d.", tiling_params.block_cycle);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "ho_wo_16=%d.", tiling_params.ho_wo_16);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "mask_shape_128=%d.", tiling_params.mask_shape_128);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "pad_left=%d.", tiling_params.pad_left);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "pad_right=%d.", tiling_params.pad_right);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "pad_top=%d.", tiling_params.pad_top);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "pad_bottom=%d.", tiling_params.pad_bottom);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "each_process_wo=%d.", tiling_params.each_process_wo);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "each_process_ho=%d.", tiling_params.each_process_ho);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "each_process_wi=%d.", tiling_params.each_process_wi);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "each_process_hi=%d.", tiling_params.each_process_hi);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "c1=%d.", tiling_params.c1);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "ho=%d.", tiling_params.ho);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "wo=%d.", tiling_params.wo);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "hi=%d.", tiling_params.hi);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "wi=%d.", tiling_params.wi);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "nc1=%d.", tiling_params.nc1);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "block_num=%d.", tiling_params.block_num);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "block_num_inner=%d.", tiling_params.block_num_inner);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "block_num_outer=%d.", tiling_params.block_num_outer);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "ho_inner=%d.", tiling_params.ho_inner);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "ho_outer=%d.", tiling_params.ho_outer);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "block=%d.", tiling_params.block);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "act_core_num=%d.", tiling_params.act_core_num);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "tile_h_to_block=%d.", tiling_params.tile_h_to_block);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "if_block=%d.", tiling_params.if_block);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "shape_ho=%d.", tiling_params.shape_ho);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "shape_hi=%d.", tiling_params.shape_hi);
  OP_LOGD("MaxPoolGradWithArgmaxV2Tiling", "one_window_size=%d.", tiling_params.one_window_size);
}

bool MaxPoolGradWithArgmaxV2Tiling(const std::string& op_type, const TeOpParas& op_paras,
                                   const nlohmann::json& op_compile_info, OpRunInfo& run_info) {
  OP_LOGI(op_type.c_str(), "MaxPoolGradWithArgmaxV2Tiling running.");

  if (op_paras.inputs.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs is empty.");
    return false;
  }
  if (op_paras.inputs[INPUT_X_INDEX].tensor.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ori_input tensor is empty.");
    return false;
  }
  if (op_paras.inputs[INPUT_GRAD_INDEX].tensor.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "grad tensor is empty.");
    return false;
  }
  if (op_paras.inputs[INPUT_ARGMAX_INDEX].tensor.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "argmax tensor is empty.");
    return false;
  }

  CompileInfoParams compile_params;

  bool get_compile_info = GetCompileInfo(op_type, op_compile_info, compile_params);
  if (!get_compile_info) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "MaxPoolGradWithArgmaxV2Tiling: GetCompileInfo error.");
    return false;
  }
  // check kernel size limit
  OP_TILING_CHECK((compile_params.ub_ele / 7 / C0 / compile_params.kh) < compile_params.kw,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "kernel size is too large to support."), return false);
  // check compile info paramters
  OP_TILING_CHECK((compile_params.ub_ele <= 0), OP_LOGE(op_type.c_str(), "ub_ele must greater than 0."), return false);
  OP_TILING_CHECK((compile_params.core_num <= 0), OP_LOGE(op_type.c_str(), "core_num must greater than 0."),
                  return false);
  OP_TILING_CHECK((compile_params.kh <= 0) || (compile_params.kw <= 0) || (compile_params.stride_h <= 0) ||
                  (compile_params.stride_w <= 0),
                  OP_LOGE(op_type.c_str(), "ksize and strides must greater than 0."), return false);

  MaxPoolGradWithArgmaxV2TilingParams tiling_params;
  InitTilingParams(tiling_params);

  const std::vector<int64_t>& input_shape = op_paras.inputs[0].tensor[0].shape;
  const std::vector<int64_t>& grad_shape = op_paras.inputs[1].tensor[0].shape;
  OP_TILING_CHECK(
      input_shape.size() != 5,
      VECTOR_INNER_ERR_REPORT_TILIING(
        op_type, "Get input shape failed, the length of input shape must be 5, but got %lu.", input_shape.size()),
      return false);
  OP_TILING_CHECK((input_shape[4] != 16),
                  VECTOR_INNER_ERR_REPORT_TILIING(
                    op_type, "Get input shape failed, dim 5 of input shape must be 16, but got %lu.", input_shape[4]),
                  return false);
  OP_TILING_CHECK(
      grad_shape.size() != 5,
      VECTOR_INNER_ERR_REPORT_TILIING(
        op_type, "Get grad shape failed, the length of grad shape must be 5, but got %lu.", grad_shape.size()),
      return false);
  OP_TILING_CHECK((grad_shape[4] != 16),
                  VECTOR_INNER_ERR_REPORT_TILIING(
                    op_type, "Get grad shape failed, dim 5 of grad shape must be 16, but got %lu.", grad_shape[4]),
                  return false);

  CalTilingParam(tiling_params, compile_params, grad_shape, input_shape);
  SetTilingParam(tiling_params, run_info);
  PrintTilingParam(tiling_params);

  // block_dim and workspace, use fot tik op
  run_info.block_dim = tiling_params.act_core_num;
  const int64_t WORKSPACE_DIM = 1;
  const int64_t WORKSPACE_SIZE = 1073741824;
  vector<int64_t> workspace(WORKSPACE_DIM, WORKSPACE_SIZE);
  run_info.workspaces = workspace;

  if (compile_params.kh > compile_params.stride_h) {
    // calc actual used workspace
    int64_t n = input_shape[SHAPE_INDEX_N];
    int64_t c1 = input_shape[SHAPE_INDEX_C1];
    int64_t hi = input_shape[SHAPE_INDEX_H];
    int64_t wi = input_shape[SHAPE_INDEX_W];
    int64_t actual_workspace = n * c1 * hi * wi * BLOCK_ALLIGN * DTYPE_SIZE_FP32;
    if (actual_workspace > WORKSPACE_SIZE) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Overlap is too large to support, please decrease input_shape.");
      return false;
    }
  }

  OP_LOGI(op_type.c_str(), "MaxPoolGradWithArgmaxV2Tiling run success.");
  return true;
}
// register tiling interface of the MaxPoolGradWithArgmaxV2 op.
REGISTER_OP_TILING_FUNC_BUFFERED(MaxPoolGradWithArgmaxV2, MaxPoolGradWithArgmaxV2Tiling);
}  // namespace optiling.
