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
 * \file resize_nearest_neighbor_v2.cpp
 * \brief
 */
#include "resize_common.h"

namespace optiling {
const int64_t TILING_100110_HW_THRESHOLD = 64;
const int64_t TILING_100110_NC1_THRESHOLD = 32;
const int64_t SHAPE_C0 = 16;

// tiling_key format: 000000
// 1. Reserved, default 1
// 2. h align flag, 0: h -> x.x*h, 1: h -> nh, 2: nh -> h, 3: h = h
// 3. w align flag, 0: w -> x.x*w, 1: w -> nw, 2: nw -> w, 3: w = w
// 4. Reserved, default 0, hdim <= TILING_100110_HW_THRESHOLD
// 5. Reserved, default 0, wdim <= TILING_100110_HW_THRESHOLD
// 6. Reserved, default 0

static void GetTilingForDefault(const ResizeClassCompileParams& compile_params,
                                ResizeClassTilingParams& tiling_params) {
  auto h_max = (tiling_params.output_height + compile_params.core_num - 1) / compile_params.core_num;
  h_max = (tiling_params.output_height + h_max - 1) / h_max;
  auto image_batch_c1 = tiling_params.input_batch * tiling_params.input_c1;
  auto cut_width_total = tiling_params.output_width;
  auto w_max = (cut_width_total + compile_params.core_num - 1) / compile_params.core_num;
  w_max = (cut_width_total + w_max - 1) / w_max;
  for (int64_t i = 0; i < compile_params.core_num; ++i) {
    int64_t cut_width_num_tmp = pow(2, i);
    if (cut_width_num_tmp > compile_params.core_num) {
      tiling_params.cut_width_num = min(cut_width_num_tmp / 2, w_max);
      break;
    }
    int64_t w_sigment = (cut_width_total + cut_width_num_tmp - 1) / cut_width_num_tmp;
    if (w_sigment <= 256) {
      tiling_params.cut_width_num = min(cut_width_num_tmp, w_max);
      break;
    }
  }
  auto left_core_num = compile_params.core_num / tiling_params.cut_width_num;
  // when NC1 > 8, will cut NC1 first
  auto nc_sigment = (image_batch_c1 + left_core_num - 1) / left_core_num;
  nc_sigment = max(nc_sigment, int64_t(8));
  tiling_params.cut_batch_c1_num = (image_batch_c1 + nc_sigment - 1) / nc_sigment;
  left_core_num = compile_params.core_num / (tiling_params.cut_width_num * tiling_params.cut_batch_c1_num);
  // cut h
  tiling_params.cut_height_num = min(left_core_num, h_max);
  tiling_params.cut_height_num = tiling_params.cut_height_num != 0 ? tiling_params.cut_height_num : 1;
}

static void GetTilingForNoBilinear(const ResizeClassCompileParams& compile_params,
                                   ResizeClassTilingParams& tiling_params) {
  tiling_params.tiling_key = 999999;
  tiling_params.input_batch = tiling_params.input_batch * tiling_params.input_c1;
  tiling_params.input_c1 = 1;
  if (tiling_params.input_height == tiling_params.output_height &&
      tiling_params.input_width == tiling_params.output_width) {
    tiling_params.input_batch = tiling_params.input_batch * tiling_params.input_height * tiling_params.input_width;
    tiling_params.input_height = 1;
    tiling_params.output_height = 1;
    tiling_params.input_width = 1;
    tiling_params.output_width = 1;
  }
  tiling_params.cut_batch_c1_num = compile_params.core_num;
  tiling_params.cut_height_num = 1;
  tiling_params.cut_width_num = 1;
}

static void GetTilingForNCProc(const ResizeClassCompileParams& compile_params, ResizeClassTilingParams& tiling_params) {
  // NC1(big) > H > W > NC1(small)
  tiling_params.tiling_key = 100110;
  tiling_params.input_batch = tiling_params.input_batch * tiling_params.input_c1;
  tiling_params.input_c1 = 1;

  // step 1: calcu batch* c1 cut num, the minest nc1 = 1024
  auto nc1_max = (tiling_params.input_batch + compile_params.core_num - 1) / compile_params.core_num;
  nc1_max = (tiling_params.input_batch + nc1_max - 1) / nc1_max;
  for (int64_t i = 0; i < compile_params.core_num; ++i) {
    int64_t tmp_cut_nc1_num = pow(2, i);
    if (tmp_cut_nc1_num > compile_params.core_num) {
      tiling_params.cut_batch_c1_num = min(tmp_cut_nc1_num / 2, nc1_max);
      break;
    }
    int64_t nc1_sigment = (tiling_params.input_batch + tmp_cut_nc1_num - 1) / tmp_cut_nc1_num;
    if (nc1_sigment <= 1024) {
      tiling_params.cut_batch_c1_num = min(tmp_cut_nc1_num, nc1_max);
      break;
    }
  }

  // step 2: calcu h/w cut num
  auto left_core_num = compile_params.core_num / tiling_params.cut_batch_c1_num;
  auto h_max = (tiling_params.output_height + left_core_num - 1) / left_core_num;
  h_max = (tiling_params.output_height + h_max - 1) / h_max;
  tiling_params.cut_height_num = h_max;

  left_core_num = compile_params.core_num / (tiling_params.cut_batch_c1_num * tiling_params.cut_height_num);
  auto w_max = (tiling_params.output_width + left_core_num - 1) / left_core_num;
  w_max = (tiling_params.output_width + w_max - 1) / w_max;
  tiling_params.cut_width_num = w_max;
}

bool GetResizeBilinearV2Tiling(const ResizeClassCompileParams& compile_params, ResizeClassTilingParams& tiling_params) {
  if (compile_params.tuneParams.tiling_key == 999999) {
    OP_LOGI(compile_params.op_type, "Start setting tiling params by tune params.");
    GetTilingForNoBilinear(compile_params, tiling_params);
    return true;
  }
  if ((compile_params.tuneParams.tiling_key == 100110 || compile_params.tuneParams.tiling_key == 100000) &&
      compile_params.tuneParams.cut_batch_c1_num * compile_params.tuneParams.cut_height_num *
              compile_params.tuneParams.cut_width_num <=
          compile_params.core_num) {
    OP_LOGI(compile_params.op_type, "Start setting tiling params by tune params.");
    tiling_params.tiling_key = compile_params.tuneParams.tiling_key;
    tiling_params.input_batch = tiling_params.input_batch * tiling_params.input_c1;
    tiling_params.input_c1 = 1;
    tiling_params.cut_batch_c1_num = compile_params.tuneParams.cut_batch_c1_num;
    tiling_params.cut_height_num = compile_params.tuneParams.cut_height_num;
    tiling_params.cut_width_num = compile_params.tuneParams.cut_width_num;
    return true;
  }
  OP_LOGI(compile_params.op_type, "Start calculating tiling parameters.");
  bool is_resize_with_no_bilinear = (tiling_params.input_height == tiling_params.output_height &&
                                     tiling_params.input_width == tiling_params.output_width) ||
                                    (tiling_params.input_height == 1 && tiling_params.input_width == 1);
  bool is_resize_with_nc_process = tiling_params.input_height < TILING_100110_HW_THRESHOLD &&
                                   tiling_params.output_height < TILING_100110_HW_THRESHOLD &&
                                   tiling_params.input_width < TILING_100110_HW_THRESHOLD &&
                                   tiling_params.output_width < TILING_100110_HW_THRESHOLD &&
                                   tiling_params.input_batch * tiling_params.input_c1 >= TILING_100110_NC1_THRESHOLD;
  if (is_resize_with_no_bilinear) {
    // this case do the resize with bilinear
    // (h_in = h_out w_in = w_out) or (h_in = 1  w_in = 1)
    // only cut by tiling_params.input_batch * tiling_params.input_c1
    GetTilingForNoBilinear(compile_params, tiling_params);
    return true;
  }
  if (is_resize_with_nc_process) {
    // process h * w, so cut by nc1 first from input
    GetTilingForNCProc(compile_params, tiling_params);
    return true;
  }
  // process nc1 * w
  // 1. first cut by w to get more nc1
  // 2. second cut h
  GetTilingForDefault(compile_params, tiling_params);

  return true;
}
}  // namespace optiling
