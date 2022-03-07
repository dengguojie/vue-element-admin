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
static void GetTilingForDefault(const ResizeClassCompileParams& compile_params, const bool is_w_nw,
                                ResizeClassTilingParams& tiling_params) {
  auto h_max = (tiling_params.output_height + compile_params.core_num - 1) / compile_params.core_num;
  h_max = (tiling_params.output_height + h_max - 1) / h_max;
  auto image_batch_c1 = tiling_params.input_batch * tiling_params.input_c1;
  auto cut_width_total = is_w_nw ? tiling_params.input_width : tiling_params.output_width;
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
  auto nc_max = (image_batch_c1 + left_core_num - 1) / left_core_num;
  nc_max = (image_batch_c1 + nc_max - 1) / nc_max;
  // when w_cut * NC1 > compile_params.max_w_sigment, will cut NC1 first
  int64_t w_sigment = (cut_width_total + tiling_params.cut_width_num - 1) / tiling_params.cut_width_num;
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
}

static bool IsSmallImageTilingBranch(const ResizeClassTilingParams& tiling_params, int64_t c0_blocks) {
  return tiling_params.cut_height_num * tiling_params.cut_width_num == 1 &&
         tiling_params.input_height * tiling_params.input_width < 100 &&
         tiling_params.output_height * tiling_params.output_width * c0_blocks <= 65535 &&
         tiling_params.output_width <= 128;
}

bool GetResizeNearestNeighborV2Tiling(const ResizeClassCompileParams& compile_params,
                                      ResizeClassTilingParams& tiling_params) {
  auto data_each_block = 8; // float32
  if (tiling_params.input_dtype == ge::DT_FLOAT16) {
    data_each_block = 16;
  }
  auto c0_blocks = tiling_params.input_c0 / data_each_block;

  // check whether h,w to nh,nw
  bool is_h_nh = IsIntegerResize(compile_params, tiling_params.input_height, tiling_params.output_height);
  bool is_w_nw = IsIntegerResize(compile_params, tiling_params.input_width, tiling_params.output_width);

  // h is not h -> mh and w -> nw, will run output cut branch
  // is the n is too large (> 100), will not use high performance branch
  is_w_nw = (!is_h_nh && (tiling_params.output_width / tiling_params.input_width) > 100) ? false : is_w_nw;

  // will use DEFAULT branch
  if (is_w_nw && is_h_nh) {
    auto h_scale = tiling_params.output_height / tiling_params.input_height;
    auto w_scale = tiling_params.output_width / tiling_params.input_width;
    // 1. if w_scale is larger than ub_max // C0, w_in_per_loop will less than 1
    // 2. data_move dst_gap stride should be less than 65536
    if (w_scale > compile_params.max_w_len ||
        h_scale * tiling_params.input_width * w_scale * c0_blocks > 65535) {
      is_w_nw = false;
    }
  }

  auto small_image_flag = 0;
  auto h_tiling_align_flag = 0;
  auto w_tiling_align_flag = is_w_nw ? 1 : 0;
  if (is_w_nw && tiling_params.output_width == tiling_params.input_width) {
    w_tiling_align_flag = 3;
  }

  if (is_h_nh && is_w_nw) {
    // process h * w, so cut by nc1 first from input
    h_tiling_align_flag = 1;
    GetTilingForHW2MHNW(compile_params, tiling_params);
    if (IsSmallImageTilingBranch(tiling_params, c0_blocks)) {
      small_image_flag = 1;
    }
  } else {
    // process nc1 * w
    // 1. first cut by w to get more nc1
    // 2. second cut h
    GetTilingForDefault(compile_params, is_w_nw, tiling_params);
  }

  /* Tiling key expression: 1abcde
      a: whether H is integer zoom-in.
      b: whether W is integer zoom-in. 3: srcW = dstW
      c/d: not used currently
      e: particular flag for SMALL images AlignHW

  Note: Attribute align_corners or half_pixel_centers with True value will be considered as 100000
  */
  tiling_params.tiling_key =
      tiling_params.tiling_key +
      h_tiling_align_flag * HEIGHT_ALIGN_FLAG +
      w_tiling_align_flag * width_ALIGN_FLAG +
      small_image_flag * SMALL_IMAGE_FLAG;
  return true;
}
}  // namespace optiling
