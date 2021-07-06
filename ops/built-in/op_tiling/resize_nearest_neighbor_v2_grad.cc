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
 * \file resize_nearest_neighbor_v2.cpp
 * \brief
 */
#include "resize_common.h"

namespace optiling {

static void GetTilingForNW2W(const ResizeClassCompileParams& compile_params, ResizeClassTilingParams& tiling_params) {
  // cut width first
  if (tiling_params.output_width <= compile_params.core_num) {
    tiling_params.cut_width_num = tiling_params.output_width;
  } else {
    for (int64_t i = (compile_params.core_num - 1); i > 0; i--) {
      if (tiling_params.output_width % i == 0) {
        tiling_params.cut_width_num = i;
        break;
      }
    }
  }
  auto left_core_num = compile_params.core_num - tiling_params.cut_width_num;
  if (left_core_num != 0) {
    // continue cut height
    left_core_num = compile_params.core_num / tiling_params.cut_width_num;
    auto cut_h_sigment = (tiling_params.input_height + left_core_num - 1) / left_core_num;
    auto h_max = (tiling_params.input_height + cut_h_sigment - 1) / cut_h_sigment;
    tiling_params.cut_height_num = min(left_core_num, h_max);
  }
  left_core_num = compile_params.core_num / (tiling_params.cut_width_num * tiling_params.input_height);
  // continue cut height
  auto image_batch_c1 = tiling_params.input_batch * tiling_params.input_c1;
  if (left_core_num > 1) {
    auto cut_batch_c1_sigment = (image_batch_c1 + left_core_num - 1) / left_core_num;
    tiling_params.cut_batch_c1_num = (image_batch_c1 + cut_batch_c1_sigment - 1) / cut_batch_c1_sigment;
    tiling_params.cut_batch_c1_num = min(left_core_num, tiling_params.cut_batch_c1_num);
  }
}

static void GetTilingDefault(const ResizeClassCompileParams& compile_params, ResizeClassTilingParams& tiling_params) {
  auto h_max = (tiling_params.input_height + compile_params.core_num - 1) / compile_params.core_num;
  h_max = (tiling_params.input_height + h_max - 1) / h_max;
  auto image_batch_c1 = tiling_params.input_batch * tiling_params.input_c1;
  auto w_max = (tiling_params.input_width + compile_params.core_num - 1) / compile_params.core_num;
  w_max = (tiling_params.input_width + w_max - 1) / w_max;
  for (int64_t i = 0; i < compile_params.core_num; ++i) {
    int64_t cut_width_num_tmp = pow(2, i);
    if (cut_width_num_tmp > compile_params.core_num) {
      tiling_params.cut_width_num = min(cut_width_num_tmp / 2, w_max);
      break;
    }
    int64_t w_sigment = (tiling_params.input_width + cut_width_num_tmp - 1) / cut_width_num_tmp;
    if (w_sigment <= 256) {
      tiling_params.cut_width_num = min(cut_width_num_tmp, w_max);
      break;
    }
  }
  auto left_core_num = compile_params.core_num / tiling_params.cut_width_num;
  auto nc_max = (image_batch_c1 + left_core_num - 1) / left_core_num;
  nc_max = (image_batch_c1 + nc_max - 1) / nc_max;
  // when w_cut * NC1 > compile_params.max_w_len, will cut NC1 first
  int64_t w_sigment = (tiling_params.input_width + tiling_params.cut_width_num - 1) / tiling_params.cut_width_num;
  auto image_batch_c1_w = image_batch_c1 * min(w_sigment, int64_t(128));
  auto nc_cut_by_compile = (image_batch_c1_w + compile_params.max_w_len - 1) / compile_params.max_w_len;
  if (image_batch_c1_w > compile_params.max_w_len) {
    tiling_params.cut_batch_c1_num = min(nc_cut_by_compile, nc_max);
    tiling_params.cut_batch_c1_num = max(int64_t(1), tiling_params.cut_batch_c1_num);
    left_core_num = compile_params.core_num / (tiling_params.cut_width_num * tiling_params.cut_batch_c1_num);
  }
  // cut h
  tiling_params.cut_height_num = min(left_core_num, h_max);
  tiling_params.cut_height_num = tiling_params.cut_height_num != 0 ? tiling_params.cut_height_num : 1;
  left_core_num = compile_params.core_num /
                  (tiling_params.cut_width_num * tiling_params.cut_batch_c1_num * tiling_params.cut_height_num);
  int64_t cut_width_height_num = tiling_params.cut_width_num * tiling_params.cut_height_num;
  if (left_core_num > 1 && image_batch_c1 > 1 && cut_width_height_num < 17) {
    left_core_num = compile_params.core_num / cut_width_height_num;
    nc_max = (image_batch_c1 + left_core_num - 1) / left_core_num;
    nc_max = (image_batch_c1 + nc_max - 1) / nc_max;
    tiling_params.cut_batch_c1_num = min(left_core_num, nc_max);
  }
}

bool GetResizeNearestNeighborV2GradTiling(const ResizeClassCompileParams& compile_params,
                                          ResizeClassTilingParams& tiling_params) {
  // check whether h,w to nh,nw
  bool is_h_nh = ((tiling_params.output_height % tiling_params.input_height == 0 &&
                   compile_params.align_corners + compile_params.half_pixel_centers == 0) ||
                  (tiling_params.output_height == tiling_params.input_height));
  bool is_w_nw = ((tiling_params.output_width % tiling_params.input_width == 0 &&
                   compile_params.align_corners + compile_params.half_pixel_centers == 0) ||
                  tiling_params.output_width == tiling_params.input_width);
  // h is not h-> mh and  w -> nw, will run output cut branch
  // is the n is too large (> 100), will not use hign performance branch
  is_w_nw = (!is_h_nh && tiling_params.output_width > tiling_params.input_width * 100) ? false : is_w_nw;
  // h is h-> mh and  w -> nw, n > max_w_len,  will not use hign performance branch
  is_w_nw =
      (is_h_nh && tiling_params.output_width > tiling_params.input_width * compile_params.max_w_len) ? false : is_w_nw;
  int64_t h_tiling_align_flag = 0;
  int64_t w_tiling_align_flag = is_w_nw ? 1 : 0;

  bool is_nw_w = ((tiling_params.input_width % tiling_params.output_width == 0) &&
                  (compile_params.align_corners + compile_params.half_pixel_centers == 0));
  is_nw_w = (tiling_params.input_width > tiling_params.output_width * 120) ? false : is_nw_w;
  bool is_big_to_small = tiling_params.input_width > tiling_params.output_width;
  int64_t w_align_flag = (is_big_to_small && is_nw_w) ? 1 : 0;
  int64_t is_big_to_small_flag = is_big_to_small ? 1 : 0;

  if (is_h_nh && is_w_nw) {
    // process h * w, so cut by nc1 first from input
    h_tiling_align_flag = 1;
    GetTilingForHW2MHNW(compile_params, tiling_params);
  } else if (w_align_flag) {
    GetTilingForNW2W(compile_params, tiling_params);
  } else {
    // process nc1 * w
    // 1. first cut by w to get more nc1
    // 2. second cut h
    GetTilingDefault(compile_params, tiling_params);
  }
  tiling_params.tiling_key = tiling_params.tiling_key + h_tiling_align_flag * HEIGHT_ALIGN_FLAG +
                             w_tiling_align_flag * width_ALIGN_FLAG + w_align_flag * WIDTH_ALIGN_FLAG +
                             is_big_to_small_flag * BIG_TO_SMALL_FLAG;
  return true;
}

}  // namespace optiling
