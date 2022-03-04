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
 * \file max_pool_grad.cc
 * \brief
 */
#include <nlohmann/json.hpp>
#include "op_tiling_util.h"
#include "graph/debug/ge_log.h"

#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "error_log.h"
#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"

namespace optiling {
const std::string MaxPoolGrad_OP_TYPE = "MaxPoolGrad";
const ge::DataType DTYPE_FP32 = DT_FLOAT;
const int64_t BYTE_BLOCK = 32;
const int64_t MASK_FP32 = 64;
const int64_t MASK_FP16 = 128;
const int64_t MAX_REPEAT_TIME = 255;
const int64_t BYTE16 = 2;
const int64_t BYTE32 = 4;
const int64_t FP16_BLOCK_NUM = 16;
const int64_t C0 = 16;
const int64_t CASE_NO_TILING = 0;
const int64_t CASE_TILING_HO = 1;
const int64_t CASE_TILING_HO_WO = 2;
const int64_t CASE_CORE_HO = 3;
const int64_t CASE_CORE_HO_WO = 4;
const int64_t CASE_SAME_NO_TILING = 5;
const int64_t CASE_SAME_TILING_HO = 6;
const int64_t CASE_SAME_TILING_HO_WO = 7;
const int64_t INDEX_0 = 0;
const int64_t INDEX_1 = 1;
const int64_t INDEX_2 = 2;
const int64_t INDEX_3 = 3;
const int64_t INDEX_4 = 4;
const int64_t INDEX_5 = 5;
const int64_t INDEX_6 = 6;
const int64_t INDEX_7 = 7;
const int64_t INDEX_8 = 8;
const int64_t MODE_TWO = 2;
const int64_t TILING_FACTOR_TWO = 2;
const int64_t VCMP_NUM_EACH_REPEAT = 128;
const int64_t LOAD3D_NUM_EACH_REPEAT = 16;

static const std::vector<std::string> COMPILE_INFO_KEY = {"core_num", "ub_size", "kh",      "kw",
                                                          "sh",       "sw",      "padding", "l1_size"};

struct TilingParams {
  // tiling params
  int64_t select_key;
  int64_t n;
  int64_t c1;
  int64_t h;
  int64_t w;
  int64_t ho;
  int64_t wo;
  int64_t pad_hw_top;
  int64_t pad_hw_bottom;
  int64_t pad_hw_left;
  int64_t pad_hw_right;
  int64_t overlap_h;
  int64_t overlap_w;
  int64_t hi_invalid;
  int64_t wi_invalid;
  int64_t total_num;
  int64_t core_num;
  int64_t core_ou_shape_h;
  int64_t core_ou_shape_w;
  int64_t core_in_shape_h;
  int64_t core_in_shape_w;
  int64_t new_ho;
  int64_t new_wo;
  int64_t total_num_div_core;
  int64_t total_num_div_core_1;
  int64_t core_loop_params;
  int64_t core_loop_params1;
  int64_t hi_batch;
  int64_t wi_batch;
  int64_t wi_tail;
  int64_t wo_tail;
  int64_t loop_ho;
  int64_t loop_wo;
  int64_t dup_repeat_merchant_f_map_fp32;
  int64_t dup_repeat_remainder_f_map_fp32;
  int64_t dup_remainder_f_map_fp32;
  int64_t repeats_f_map_fp32;
  int64_t forward_in_shape_h_w_c0;
  int64_t forward_ou_shape_h_w_c0;
  int64_t hi_val;
  int64_t wi_val;
  int64_t burst_len;
  int64_t src_stride;
  int64_t burst_len_src_orig_y;
  int64_t src_stride_src_orig_y;
  int64_t repeat_times;
  int64_t howo_co_ver;
  int64_t mask_size_16;
  int64_t mask_size_ver;
  int64_t repeat_max_time_grad_sel;
  int64_t remain_repeat_time_grad_sel;
  int64_t remain_ele_grad_sel;
  int64_t repeat_max_loop_vadd;
  int64_t remain_max_loop_vadd;
  int64_t remain_ele_vadd;
  int64_t src_stride_ub_2_gm;
  int64_t dst_stride_ub_2_gm;
  int64_t repeat_max_loop_f_map_fp32;
  int64_t remain_max_loop_f_map_fp32;
  int64_t remain_ele_f_map_fp32;
  int64_t wi_val_tail;
  int64_t burst_len_tail;
  int64_t src_stride_tail;
  int64_t pad_hw_top_neg;
  int64_t pad_hw_left_neg;
  int64_t forward_in_shape_h_w_2;
  int64_t burst_len_src_orig_y_tail;
  int64_t src_stride_src_orig_y_tail;
  int64_t repeat_times_tail;
  int64_t howo_co_ver_tail;
  int64_t repeat_max_loop_vadd_tail;
  int64_t remain_max_loop_vadd_tail;
  int64_t remain_ele_vadd_tail;
  int64_t src_stride_ub_2_gm_tail;
  int64_t dst_stride_ub_2_gm_tail;
  int64_t core_ho_times;
  int64_t core_wo_times;
  int64_t map_hi;
  int64_t map_wi;
  int64_t config;
  int64_t sh_wi_2;
  int64_t num_instr_loop_h;
  int64_t num_instr_loop_w;
  int64_t remain_mask;
  int64_t remain_repeat;
  int64_t num_instr_loop_w_1;
  int64_t num_instr_loop_h_1;
  int64_t ho_tail;
  int64_t hi_tail;
  int64_t dst_stride_tail;
  int64_t wo_2;
  int64_t boundary_h;
  int64_t burst_len_ub_2_gm;
  int64_t non_overlap_1;
  int64_t overlap_1;
  int64_t burst_len_over;
  int64_t src_stride_over;
  int64_t dst_stride_over;
  int64_t dup_repeat_merchant_non_overlap;
  int64_t dup_repeat_remainder_non_overlap;
  int64_t dup_remainder_non_overlap;
  int64_t repeats_non_overlap;
  int64_t burst_len_ub2gm_2;
  int64_t src_stride_ub2gm_2;
  int64_t dst_stride_ub2gm_2;
  int64_t burst_len_ub2gm_3;
  int64_t src_stride_ub2gm_3;
  int64_t dst_stride_ub2gm_3;
  int64_t hi_val_tail;
  int64_t burst_len_val;
  int64_t src_stride_val;
  int64_t dst_stride_val;
  int64_t burst_len_val_tail;
  int64_t src_stride_val_tail;
  int64_t dst_stride_val_tail;
  int64_t num_instr_loop_h_tail;
  int64_t remain_repeat_tail;
  int64_t num_instr_loop_h_1_tail;
  int64_t burst_len_ub_2_gm_tail;
  int64_t non_overlap_1_tail;
  int64_t src_stride_over_tail;
  int64_t dst_stride_over_tail;
  int64_t dup_repeat_merchant_non_overlap_tail;
  int64_t dup_repeat_remainder_non_overlap_tail;
  int64_t dup_remainder_non_overlap_tail;
  int64_t repeats_non_overlap_tail;
  int64_t burst_len_ub2gm_2_tail;
  int64_t src_stride_ub2gm_2_tail;
  int64_t dst_stride_ub2gm_2_tail;
  int64_t burst_len_ub2gm_3_tail;
  int64_t src_stride_ub2gm_3_tail;
  int64_t dst_stride_ub2gm_3_tail;
  int64_t forward_in_shape_w_c0;
  int64_t dst_stride;
};

void InitTilingParams(TilingParams& params) {
  // init params
  params.select_key = 0;
  params.n = 0;
  params.c1 = 0;
  params.h = 0;
  params.w = 0;
  params.ho = 0;
  params.wo = 0;
  params.pad_hw_top = 0;
  params.pad_hw_bottom = 0;
  params.pad_hw_left = 0;
  params.pad_hw_right = 0;
  params.overlap_h = 0;
  params.overlap_w = 0;
  params.hi_invalid = 0;
  params.wi_invalid = 0;
  params.total_num = 0;
  params.core_num = 0;
  params.core_ou_shape_h = 0;
  params.core_ou_shape_w = 0;
  params.core_in_shape_h = 0;
  params.core_in_shape_w = 0;
  params.new_ho = 0;
  params.new_wo = 0;
  params.total_num_div_core = 0;
  params.total_num_div_core_1 = 0;
  params.core_loop_params = 0;
  params.core_loop_params1 = 0;
  params.hi_batch = 0;
  params.wi_batch = 0;
  params.wi_tail = 0;
  params.wo_tail = 0;
  params.loop_ho = 0;
  params.loop_wo = 0;
  params.dup_repeat_merchant_f_map_fp32 = 0;
  params.dup_repeat_remainder_f_map_fp32 = 0;
  params.dup_remainder_f_map_fp32 = 0;
  params.repeats_f_map_fp32 = 0;
  params.forward_in_shape_h_w_c0 = 0;
  params.forward_ou_shape_h_w_c0 = 0;
  params.hi_val = 0;
  params.wi_val = 0;
  params.burst_len = 0;
  params.src_stride = 0;
  params.burst_len_src_orig_y = 0;
  params.src_stride_src_orig_y = 0;
  params.repeat_times = 0;
  params.howo_co_ver = 0;
  params.mask_size_16 = 0;
  params.mask_size_ver = 0;
  params.repeat_max_time_grad_sel = 0;
  params.remain_repeat_time_grad_sel = 0;
  params.remain_ele_grad_sel = 0;
  params.repeat_max_loop_vadd = 0;
  params.remain_max_loop_vadd = 0;
  params.remain_ele_vadd = 0;
  params.src_stride_ub_2_gm = 0;
  params.dst_stride_ub_2_gm = 0;
  params.repeat_max_loop_f_map_fp32 = 0;
  params.remain_max_loop_f_map_fp32 = 0;
  params.remain_ele_f_map_fp32 = 0;
  params.wi_val_tail = 0;
  params.burst_len_tail = 0;
  params.src_stride_tail = 0;
  params.pad_hw_top_neg = 0;
  params.pad_hw_left_neg = 0;
  params.forward_in_shape_h_w_2 = 0;
  params.burst_len_src_orig_y_tail = 0;
  params.src_stride_src_orig_y_tail = 0;
  params.repeat_times_tail = 0;
  params.howo_co_ver_tail = 0;
  params.repeat_max_loop_vadd_tail = 0;
  params.remain_max_loop_vadd_tail = 0;
  params.remain_ele_vadd_tail = 0;
  params.src_stride_ub_2_gm_tail = 0;
  params.dst_stride_ub_2_gm_tail = 0;
  params.core_ho_times = 0;
  params.core_wo_times = 0;
  params.map_hi = 0;
  params.map_wi = 0;
  params.config = 0;
  params.sh_wi_2 = 0;
  params.num_instr_loop_h = 0;
  params.num_instr_loop_w = 0;
  params.remain_mask = 0;
  params.remain_repeat = 0;
  params.num_instr_loop_w_1 = 0;
  params.num_instr_loop_h_1 = 0;
  params.ho_tail = 0;
  params.hi_tail = 0;
  params.dst_stride_tail = 0;
  params.wo_2 = 0;
  params.boundary_h = 0;
  params.burst_len_ub_2_gm = 0;
  params.non_overlap_1 = 0;
  params.overlap_1 = 0;
  params.burst_len_over = 0;
  params.src_stride_over = 0;
  params.dst_stride_over = 0;
  params.dup_repeat_merchant_non_overlap = 0;
  params.dup_repeat_remainder_non_overlap = 0;
  params.dup_remainder_non_overlap = 0;
  params.repeats_non_overlap = 0;
  params.burst_len_ub2gm_2 = 0;
  params.src_stride_ub2gm_2 = 0;
  params.dst_stride_ub2gm_2 = 0;
  params.burst_len_ub2gm_3 = 0;
  params.src_stride_ub2gm_3 = 0;
  params.dst_stride_ub2gm_3 = 0;
  params.hi_val_tail = 0;
  params.burst_len_val = 0;
  params.src_stride_val = 0;
  params.dst_stride_val = 0;
  params.burst_len_val_tail = 0;
  params.src_stride_val_tail = 0;
  params.dst_stride_val_tail = 0;
  params.num_instr_loop_h_tail = 0;
  params.remain_repeat_tail = 0;
  params.num_instr_loop_h_1_tail = 0;
  params.burst_len_ub_2_gm_tail = 0;
  params.non_overlap_1_tail = 0;
  params.src_stride_over_tail = 0;
  params.dst_stride_over_tail = 0;
  params.dup_repeat_merchant_non_overlap_tail = 0;
  params.dup_repeat_remainder_non_overlap_tail = 0;
  params.dup_remainder_non_overlap_tail = 0;
  params.repeats_non_overlap_tail = 0;
  params.burst_len_ub2gm_2_tail = 0;
  params.src_stride_ub2gm_2_tail = 0;
  params.dst_stride_ub2gm_2_tail = 0;
  params.burst_len_ub2gm_3_tail = 0;
  params.src_stride_ub2gm_3_tail = 0;
  params.dst_stride_ub2gm_3_tail = 0;
  params.forward_in_shape_w_c0 = 0;
  params.dst_stride = 0;
}

void MaxWriteTilingParams(const TilingParams& params, utils::OpRunInfo& run_info) {
  // write params
  run_info.AddTilingData(params.select_key);
  run_info.AddTilingData(params.n);
  run_info.AddTilingData(params.c1);
  run_info.AddTilingData(params.h);
  run_info.AddTilingData(params.w);
  run_info.AddTilingData(params.ho);
  run_info.AddTilingData(params.wo);
  run_info.AddTilingData(params.pad_hw_top);
  run_info.AddTilingData(params.pad_hw_bottom);
  run_info.AddTilingData(params.pad_hw_left);
  run_info.AddTilingData(params.pad_hw_right);
  run_info.AddTilingData(params.overlap_h);
  run_info.AddTilingData(params.overlap_w);
  run_info.AddTilingData(params.hi_invalid);
  run_info.AddTilingData(params.wi_invalid);
  run_info.AddTilingData(params.total_num);
  run_info.AddTilingData(params.core_num);
  run_info.AddTilingData(params.core_ou_shape_h);
  run_info.AddTilingData(params.core_ou_shape_w);
  run_info.AddTilingData(params.core_in_shape_h);
  run_info.AddTilingData(params.core_in_shape_w);
  run_info.AddTilingData(params.new_ho);
  run_info.AddTilingData(params.new_wo);
  run_info.AddTilingData(params.total_num_div_core);
  run_info.AddTilingData(params.total_num_div_core_1);
  run_info.AddTilingData(params.core_loop_params);
  run_info.AddTilingData(params.core_loop_params1);
  run_info.AddTilingData(params.hi_batch);
  run_info.AddTilingData(params.wi_batch);
  run_info.AddTilingData(params.wi_tail);
  run_info.AddTilingData(params.wo_tail);
  run_info.AddTilingData(params.loop_ho);
  run_info.AddTilingData(params.loop_wo);
  run_info.AddTilingData(params.dup_repeat_merchant_f_map_fp32);
  run_info.AddTilingData(params.dup_repeat_remainder_f_map_fp32);
  run_info.AddTilingData(params.dup_remainder_f_map_fp32);
  run_info.AddTilingData(params.repeats_f_map_fp32);
  run_info.AddTilingData(params.forward_in_shape_h_w_c0);
  run_info.AddTilingData(params.forward_ou_shape_h_w_c0);
  run_info.AddTilingData(params.hi_val);
  run_info.AddTilingData(params.wi_val);
  run_info.AddTilingData(params.burst_len);
  run_info.AddTilingData(params.src_stride);
  run_info.AddTilingData(params.burst_len_src_orig_y);
  run_info.AddTilingData(params.src_stride_src_orig_y);
  run_info.AddTilingData(params.repeat_times);
  run_info.AddTilingData(params.howo_co_ver);
  run_info.AddTilingData(params.mask_size_16);
  run_info.AddTilingData(params.mask_size_ver);
  run_info.AddTilingData(params.repeat_max_time_grad_sel);
  run_info.AddTilingData(params.remain_repeat_time_grad_sel);
  run_info.AddTilingData(params.remain_ele_grad_sel);
  run_info.AddTilingData(params.repeat_max_loop_vadd);
  run_info.AddTilingData(params.remain_max_loop_vadd);
  run_info.AddTilingData(params.remain_ele_vadd);
  run_info.AddTilingData(params.src_stride_ub_2_gm);
  run_info.AddTilingData(params.dst_stride_ub_2_gm);
  run_info.AddTilingData(params.repeat_max_loop_f_map_fp32);
  run_info.AddTilingData(params.remain_max_loop_f_map_fp32);
  run_info.AddTilingData(params.remain_ele_f_map_fp32);
  run_info.AddTilingData(params.wi_val_tail);
  run_info.AddTilingData(params.burst_len_tail);
  run_info.AddTilingData(params.src_stride_tail);
  run_info.AddTilingData(params.pad_hw_top_neg);
  run_info.AddTilingData(params.pad_hw_left_neg);
  run_info.AddTilingData(params.forward_in_shape_h_w_2);
  run_info.AddTilingData(params.burst_len_src_orig_y_tail);
  run_info.AddTilingData(params.src_stride_src_orig_y_tail);
  run_info.AddTilingData(params.repeat_times_tail);
  run_info.AddTilingData(params.howo_co_ver_tail);
  run_info.AddTilingData(params.repeat_max_loop_vadd_tail);
  run_info.AddTilingData(params.remain_max_loop_vadd_tail);
  run_info.AddTilingData(params.remain_ele_vadd_tail);
  run_info.AddTilingData(params.src_stride_ub_2_gm_tail);
  run_info.AddTilingData(params.dst_stride_ub_2_gm_tail);
  run_info.AddTilingData(params.core_ho_times);
  run_info.AddTilingData(params.core_wo_times);
  run_info.AddTilingData(params.map_hi);
  run_info.AddTilingData(params.map_wi);
  run_info.AddTilingData(params.config);
  run_info.AddTilingData(params.sh_wi_2);
  run_info.AddTilingData(params.num_instr_loop_h);
  run_info.AddTilingData(params.num_instr_loop_w);
  run_info.AddTilingData(params.remain_mask);
  run_info.AddTilingData(params.remain_repeat);
  run_info.AddTilingData(params.num_instr_loop_w_1);
  run_info.AddTilingData(params.num_instr_loop_h_1);
  run_info.AddTilingData(params.ho_tail);
  run_info.AddTilingData(params.hi_tail);
  run_info.AddTilingData(params.dst_stride_tail);
  run_info.AddTilingData(params.wo_2);
  run_info.AddTilingData(params.boundary_h);
  run_info.AddTilingData(params.burst_len_ub_2_gm);
  run_info.AddTilingData(params.non_overlap_1);
  run_info.AddTilingData(params.overlap_1);
  run_info.AddTilingData(params.burst_len_over);
  run_info.AddTilingData(params.src_stride_over);
  run_info.AddTilingData(params.dst_stride_over);
  run_info.AddTilingData(params.dup_repeat_merchant_non_overlap);
  run_info.AddTilingData(params.dup_repeat_remainder_non_overlap);
  run_info.AddTilingData(params.dup_remainder_non_overlap);
  run_info.AddTilingData(params.repeats_non_overlap);
  run_info.AddTilingData(params.burst_len_ub2gm_2);
  run_info.AddTilingData(params.src_stride_ub2gm_2);
  run_info.AddTilingData(params.dst_stride_ub2gm_2);
  run_info.AddTilingData(params.burst_len_ub2gm_3);
  run_info.AddTilingData(params.src_stride_ub2gm_3);
  run_info.AddTilingData(params.dst_stride_ub2gm_3);
  run_info.AddTilingData(params.hi_val_tail);
  run_info.AddTilingData(params.burst_len_val);
  run_info.AddTilingData(params.src_stride_val);
  run_info.AddTilingData(params.dst_stride_val);
  run_info.AddTilingData(params.burst_len_val_tail);
  run_info.AddTilingData(params.src_stride_val_tail);
  run_info.AddTilingData(params.dst_stride_val_tail);
  run_info.AddTilingData(params.num_instr_loop_h_tail);
  run_info.AddTilingData(params.remain_repeat_tail);
  run_info.AddTilingData(params.num_instr_loop_h_1_tail);
  run_info.AddTilingData(params.burst_len_ub_2_gm_tail);
  run_info.AddTilingData(params.non_overlap_1_tail);
  run_info.AddTilingData(params.src_stride_over_tail);
  run_info.AddTilingData(params.dst_stride_over_tail);
  run_info.AddTilingData(params.dup_repeat_merchant_non_overlap_tail);
  run_info.AddTilingData(params.dup_repeat_remainder_non_overlap_tail);
  run_info.AddTilingData(params.dup_remainder_non_overlap_tail);
  run_info.AddTilingData(params.repeats_non_overlap_tail);
  run_info.AddTilingData(params.burst_len_ub2gm_2_tail);
  run_info.AddTilingData(params.src_stride_ub2gm_2_tail);
  run_info.AddTilingData(params.dst_stride_ub2gm_2_tail);
  run_info.AddTilingData(params.burst_len_ub2gm_3_tail);
  run_info.AddTilingData(params.src_stride_ub2gm_3_tail);
  run_info.AddTilingData(params.dst_stride_ub2gm_3_tail);
  run_info.AddTilingData(params.forward_in_shape_w_c0);
  run_info.AddTilingData(params.dst_stride);
}

int64_t UssCeilDiv(const int64_t& num, const int64_t& factor) {
  int64_t res = (num % factor == 0) ? num / factor : num / factor + 1;
  return res;
}

vector<int64_t> PaddingMode(const GeShape& input_shape, const GeShape& output_shape, int64_t& kh, int64_t& kw,
                            int64_t& sh, int64_t& sw, std::string& padding) {
  int64_t pad_left = 0;
  int64_t pad_right = 0;
  int64_t pad_top = 0;
  int64_t pad_bottom = 0;
  int64_t ho = output_shape.GetDim(INDEX_2);
  int64_t fmap_h = input_shape.GetDim(INDEX_2);
  int64_t wo = output_shape.GetDim(INDEX_3);
  int64_t fmap_w = input_shape.GetDim(INDEX_3);
  std::vector<int64_t> pad;
  if (padding == "VALID") {
    pad = {pad_top, pad_bottom, pad_left, pad_right};
  } else {
    int64_t pad_h = (ho - 1) * sh + kh - fmap_h;
    if (pad_h < 0) {
      pad_h = 0;
    }
    pad_top = pad_h / TILING_FACTOR_TWO;
    pad_bottom = pad_h - pad_top;
    int64_t pad_w = (wo - 1) * sw + kw - fmap_w;
    if (pad_w < 0) {
      pad_w = 0;
    }
    pad_left = pad_w / TILING_FACTOR_TWO;
    pad_right = pad_w - pad_left;
    pad = {pad_top, pad_bottom, pad_left, pad_right};
  }
  return pad;
}
int64_t OverlapMode(int64_t stride, int64_t ksize, int64_t xo, int64_t xi) {
  if (xo == 1) {
    if (xi >= stride) {
      return ksize - stride;
    }
    return 0;
  }
  return ksize - stride;
}

void InferDimReturn(int64_t ho, int64_t wo, bool model, std::vector<int64_t>& ksize, std::vector<int64_t>& strides,
                    int64_t ho_ys, int64_t wo_ys, int64_t h_ys, int64_t w_ys, int64_t& hi, int64_t& wi) {
  int64_t kh = ksize[INDEX_1];
  int64_t sh = strides[INDEX_1];
  int64_t kw = ksize[INDEX_2];
  int64_t sw = strides[INDEX_2];
  if (kh > sh) {
    hi = kh + (ho - 1) * sh;
  } else {
    hi = ho * sh;
  }
  if (kw > sw) {
    wi = kw + (wo - 1) * sw;
  } else {
    wi = wo * sw;
  }
  if (model) {
    if (ho_ys == ho) {
      hi = h_ys;
    }
    if (wo_ys == wo) {
      wi = w_ys;
    }
  }
}

void GetInvalidPart(int64_t ho, int64_t wo, int64_t h_ys, int64_t w_ys, bool true_false, std::vector<int64_t>& ksize,
                    std::vector<int64_t>& strides, int64_t ho_ys, int64_t wo_ys, int64_t& invalid_h,
                    int64_t& invalid_w) {
  int64_t hi = 0;
  int64_t wi = 0;
  InferDimReturn(ho, wo, true_false, ksize, strides, ho_ys, wo_ys, h_ys, w_ys, hi, wi);
  invalid_h = h_ys - hi;
  invalid_w = w_ys - wi;
}

bool GetUssCompileParams(const std::string& op_type, const std::vector<int64_t>& op_compile_info, int64_t& core_num,
                         int64_t& ub_size, int64_t& kh, int64_t& kw, int64_t& sh, int64_t& sw, int64_t& padding_int,
                         int64_t& l1_size) {
  OP_TILING_CHECK(
      op_compile_info.size() != COMPILE_INFO_KEY.size(),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "the compile info num is not equal expect compile_info(%zu), is %zu",
                                      COMPILE_INFO_KEY.size(), op_compile_info.size()),
      return false);

  core_num = op_compile_info[INDEX_0];
  ub_size = op_compile_info[INDEX_1];
  kh = op_compile_info[INDEX_2];
  kw = op_compile_info[INDEX_3];
  sh = op_compile_info[INDEX_4];
  sw = op_compile_info[INDEX_5];
  padding_int = op_compile_info[INDEX_6];
  l1_size = op_compile_info[INDEX_7];
  return true;
}

void VectorDup(int64_t ele_num, ge::DataType dtype, int64_t& dup_repeat_merchant, int64_t& dup_repeat_remainder,
               int64_t& dup_remainder, int64_t& repeats) {
  int64_t mask = 0;
  if (dtype == DT_FLOAT16) {
    mask = MASK_FP16;
  } else {
    mask = MASK_FP32;
  }
  int64_t dup_psm = MAX_REPEAT_TIME * mask;
  dup_repeat_merchant = ele_num / dup_psm;
  dup_repeat_remainder = ele_num % dup_psm;
  if (dup_repeat_remainder != 0) {
    repeats = dup_repeat_remainder / mask;
    dup_remainder = dup_repeat_remainder % mask;
  }
}

int64_t CheckConfig(vector<int64_t>& config) {
  int64_t mark = 1;
  int64_t config_size = 6;
  for (int64_t i = 0; i < config_size; i++) {
    if (config[i] > MAX_REPEAT_TIME) {
      mark = 0;
      break;
    }
  }
  return mark;
}

void VectorDup2(int64_t ele_num, int64_t& repeat_max_time_grad_sel, int64_t& remain_repeat_time_grad_sel,
                int64_t& remain_ele_grad_sel) {
  int64_t total_repeat_time = ele_num / MASK_FP32;
  remain_ele_grad_sel = ele_num % MASK_FP32;
  repeat_max_time_grad_sel = total_repeat_time / MAX_REPEAT_TIME;
  remain_repeat_time_grad_sel = total_repeat_time % MAX_REPEAT_TIME;
}

void VectorDup3(int64_t ele_num, ge::DataType dtype, int64_t& repeat_max_loop, int64_t& remain_max_loop,
                int64_t& remain_ele) {
  int64_t repeat_times = 0;
  if (dtype == DT_FLOAT16) {
    repeat_times = ele_num / MASK_FP16;
    remain_ele = ele_num % MASK_FP16;
  } else {
    repeat_times = ele_num / MASK_FP32;
    remain_ele = ele_num % MASK_FP32;
  }
  repeat_max_loop = repeat_times / MAX_REPEAT_TIME;
  remain_max_loop = repeat_times % MAX_REPEAT_TIME;
}

bool CheckParam(const GeShape& ori_input_shape, const GeShape& ori_output_shape, const GeShape& grad_shape,
                const GeShape& ou_shape) {
  if (ori_input_shape.GetDimNum() != 5 || ori_output_shape.GetDimNum() != 5) {
    VECTOR_INNER_ERR_REPORT_TILIING("max_pool_grad", "ori_input_shape or ori_output_shape not 5D");
    return false;
  }
  if (!(grad_shape == ori_output_shape)) {
    VECTOR_INNER_ERR_REPORT_TILIING("max_pool_grad", "grad_shape not equal ori_output_shape");
    return false;
  }
  if (!(ou_shape == ori_input_shape)) {
    VECTOR_INNER_ERR_REPORT_TILIING("max_pool_grad", "ou_shape not equal ori_input_shape");
    return false;
  }
  return true;
}

void MaxPrintTilingParams(const std::string& op_type, const TilingParams& params) {
  OP_LOGD(op_type.c_str(), " params.select_key=%d", params.select_key);
  OP_LOGD(op_type.c_str(), " params.n=%d", params.n);
  OP_LOGD(op_type.c_str(), " params.c1=%d", params.c1);
  OP_LOGD(op_type.c_str(), " params.h=%d", params.h);
  OP_LOGD(op_type.c_str(), " params.w=%d", params.w);
  OP_LOGD(op_type.c_str(), " params.ho=%d", params.ho);
  OP_LOGD(op_type.c_str(), " params.wo=%d", params.wo);
  OP_LOGD(op_type.c_str(), " params.pad_hw_top=%d", params.pad_hw_top);
  OP_LOGD(op_type.c_str(), " params.pad_hw_bottom=%d", params.pad_hw_bottom);
  OP_LOGD(op_type.c_str(), " params.pad_hw_left=%d", params.pad_hw_left);
  OP_LOGD(op_type.c_str(), " params.pad_hw_right=%d", params.pad_hw_right);
  OP_LOGD(op_type.c_str(), " params.overlap_h=%d", params.overlap_h);
  OP_LOGD(op_type.c_str(), " params.overlap_w=%d", params.overlap_w);
  OP_LOGD(op_type.c_str(), " params.hi_invalid=%d", params.hi_invalid);
  OP_LOGD(op_type.c_str(), " params.wi_invalid=%d", params.wi_invalid);
  OP_LOGD(op_type.c_str(), " params.total_num=%d", params.total_num);
  OP_LOGD(op_type.c_str(), " params.core_num=%d", params.core_num);
  OP_LOGD(op_type.c_str(), " params.core_ou_shape_h=%d", params.core_ou_shape_h);
  OP_LOGD(op_type.c_str(), " params.core_ou_shape_w=%d", params.core_ou_shape_w);
  OP_LOGD(op_type.c_str(), " params.core_in_shape_h=%d", params.core_in_shape_h);
  OP_LOGD(op_type.c_str(), " params.core_in_shape_w=%d", params.core_in_shape_w);
  OP_LOGD(op_type.c_str(), " params.new_ho=%d", params.new_ho);
  OP_LOGD(op_type.c_str(), " params.new_wo=%d", params.new_wo);
  OP_LOGD(op_type.c_str(), " params.total_num_div_core=%d", params.total_num_div_core);
  OP_LOGD(op_type.c_str(), " params.total_num_div_core_1=%d", params.total_num_div_core_1);
  OP_LOGD(op_type.c_str(), " params.core_loop_params=%d", params.core_loop_params);
  OP_LOGD(op_type.c_str(), " params.core_loop_params1=%d", params.core_loop_params1);
  OP_LOGD(op_type.c_str(), " params.hi_batch=%d", params.hi_batch);
  OP_LOGD(op_type.c_str(), " params.wi_batch=%d", params.wi_batch);
  OP_LOGD(op_type.c_str(), " params.wi_tail=%d", params.wi_tail);
  OP_LOGD(op_type.c_str(), " params.wo_tail=%d", params.wo_tail);
  OP_LOGD(op_type.c_str(), " params.loop_ho=%d", params.loop_ho);
  OP_LOGD(op_type.c_str(), " params.loop_wo=%d", params.loop_wo);
  OP_LOGD(op_type.c_str(), " params.dup_repeat_merchant_f_map_fp32=%d", params.dup_repeat_merchant_f_map_fp32);
  OP_LOGD(op_type.c_str(), " params.dup_repeat_remainder_f_map_fp32=%d", params.dup_repeat_remainder_f_map_fp32);
  OP_LOGD(op_type.c_str(), " params.dup_remainder_f_map_fp32=%d", params.dup_remainder_f_map_fp32);
  OP_LOGD(op_type.c_str(), " params.repeats_f_map_fp32=%d", params.repeats_f_map_fp32);
  OP_LOGD(op_type.c_str(), " params.forward_in_shape_h_w_c0=%d", params.forward_in_shape_h_w_c0);
  OP_LOGD(op_type.c_str(), " params.forward_ou_shape_h_w_c0=%d", params.forward_ou_shape_h_w_c0);
  OP_LOGD(op_type.c_str(), " params.hi_val=%d", params.hi_val);
  OP_LOGD(op_type.c_str(), " params.wi_val=%d", params.wi_val);
  OP_LOGD(op_type.c_str(), " params.burst_len=%d", params.burst_len);
  OP_LOGD(op_type.c_str(), " params.src_stride=%d", params.src_stride);
  OP_LOGD(op_type.c_str(), " params.burst_len_src_orig_y=%d", params.burst_len_src_orig_y);
  OP_LOGD(op_type.c_str(), " params.src_stride_src_orig_y=%d", params.src_stride_src_orig_y);
  OP_LOGD(op_type.c_str(), " params.repeat_times=%d", params.repeat_times);
  OP_LOGD(op_type.c_str(), " params.howo_co_ver=%d", params.howo_co_ver);
  OP_LOGD(op_type.c_str(), " params.mask_size_16=%d", params.mask_size_16);
  OP_LOGD(op_type.c_str(), " params.mask_size_ver=%d", params.mask_size_ver);
  OP_LOGD(op_type.c_str(), " params.repeat_max_time_grad_sel=%d", params.repeat_max_time_grad_sel);
  OP_LOGD(op_type.c_str(), " params.remain_repeat_time_grad_sel=%d", params.remain_repeat_time_grad_sel);
  OP_LOGD(op_type.c_str(), " params.remain_ele_grad_sel=%d", params.remain_ele_grad_sel);
  OP_LOGD(op_type.c_str(), " params.repeat_max_loop_vadd=%d", params.repeat_max_loop_vadd);
  OP_LOGD(op_type.c_str(), " params.remain_max_loop_vadd=%d", params.remain_max_loop_vadd);
  OP_LOGD(op_type.c_str(), " params.remain_ele_vadd=%d", params.remain_ele_vadd);
  OP_LOGD(op_type.c_str(), " params.src_stride_ub_2_gm=%d", params.src_stride_ub_2_gm);
  OP_LOGD(op_type.c_str(), " params.dst_stride_ub_2_gm=%d", params.dst_stride_ub_2_gm);
  OP_LOGD(op_type.c_str(), " params.repeat_max_loop_f_map_fp32=%d", params.repeat_max_loop_f_map_fp32);
  OP_LOGD(op_type.c_str(), " params.remain_max_loop_f_map_fp32=%d", params.remain_max_loop_f_map_fp32);
  OP_LOGD(op_type.c_str(), " params.remain_ele_f_map_fp32=%d", params.remain_ele_f_map_fp32);
  OP_LOGD(op_type.c_str(), " params.wi_val_tail=%d", params.wi_val_tail);
  OP_LOGD(op_type.c_str(), " params.burst_len_tail=%d", params.burst_len_tail);
  OP_LOGD(op_type.c_str(), " params.src_stride_tail=%d", params.src_stride_tail);
  OP_LOGD(op_type.c_str(), " params.pad_hw_top_neg=%d", params.pad_hw_top_neg);
  OP_LOGD(op_type.c_str(), " params.pad_hw_left_neg=%d", params.pad_hw_left_neg);
  OP_LOGD(op_type.c_str(), " params.forward_in_shape_h_w_2=%d", params.forward_in_shape_h_w_2);
  OP_LOGD(op_type.c_str(), " params.burst_len_src_orig_y_tail=%d", params.burst_len_src_orig_y_tail);
  OP_LOGD(op_type.c_str(), " params.src_stride_src_orig_y_tail=%d", params.src_stride_src_orig_y_tail);
  OP_LOGD(op_type.c_str(), " params.repeat_times_tail=%d", params.repeat_times_tail);
  OP_LOGD(op_type.c_str(), " params.howo_co_ver_tail=%d", params.howo_co_ver_tail);
  OP_LOGD(op_type.c_str(), " params.repeat_max_loop_vadd_tail=%d", params.repeat_max_loop_vadd_tail);
  OP_LOGD(op_type.c_str(), " params.remain_max_loop_vadd_tail=%d", params.remain_max_loop_vadd_tail);
  OP_LOGD(op_type.c_str(), " params.remain_ele_vadd_tail=%d", params.remain_ele_vadd_tail);
  OP_LOGD(op_type.c_str(), " params.src_stride_ub_2_gm_tail=%d", params.src_stride_ub_2_gm_tail);
  OP_LOGD(op_type.c_str(), " params.dst_stride_ub_2_gm_tail=%d", params.dst_stride_ub_2_gm_tail);
  OP_LOGD(op_type.c_str(), " params.core_ho_times=%d", params.core_ho_times);
  OP_LOGD(op_type.c_str(), " params.core_wo_times=%d", params.core_wo_times);
  OP_LOGD(op_type.c_str(), " params.map_hi=%d", params.map_hi);
  OP_LOGD(op_type.c_str(), " params.map_wi=%d", params.map_wi);
  OP_LOGD(op_type.c_str(), " params.config=%d", params.config);
  OP_LOGD(op_type.c_str(), " params.sh_wi_2=%d", params.sh_wi_2);
  OP_LOGD(op_type.c_str(), " params.num_instr_loop_h=%d", params.num_instr_loop_h);
  OP_LOGD(op_type.c_str(), " params.num_instr_loop_w=%d", params.num_instr_loop_w);
  OP_LOGD(op_type.c_str(), " params.remain_mask=%d", params.remain_mask);
  OP_LOGD(op_type.c_str(), " params.remain_repeat=%d", params.remain_repeat);
  OP_LOGD(op_type.c_str(), " params.num_instr_loop_w_1=%d", params.num_instr_loop_w_1);
  OP_LOGD(op_type.c_str(), " params.num_instr_loop_h_1=%d", params.num_instr_loop_h_1);
  OP_LOGD(op_type.c_str(), " params.ho_tail=%d", params.ho_tail);
  OP_LOGD(op_type.c_str(), " params.hi_tail=%d", params.hi_tail);
  OP_LOGD(op_type.c_str(), " params.dst_stride_tail=%d", params.dst_stride_tail);
  OP_LOGD(op_type.c_str(), " params.wo_2=%d", params.wo_2);
  OP_LOGD(op_type.c_str(), " params.boundary_h=%d", params.boundary_h);
  OP_LOGD(op_type.c_str(), " params.burst_len_ub_2_gm=%d", params.burst_len_ub_2_gm);
  OP_LOGD(op_type.c_str(), " params.non_overlap_1=%d", params.non_overlap_1);
  OP_LOGD(op_type.c_str(), " params.overlap_1=%d", params.overlap_1);
  OP_LOGD(op_type.c_str(), " params.burst_len_over=%d", params.burst_len_over);
  OP_LOGD(op_type.c_str(), " params.src_stride_over=%d", params.src_stride_over);
  OP_LOGD(op_type.c_str(), " params.dst_stride_over=%d", params.dst_stride_over);
  OP_LOGD(op_type.c_str(), " params.dup_repeat_merchant_non_overlap=%d", params.dup_repeat_merchant_non_overlap);
  OP_LOGD(op_type.c_str(), " params.dup_repeat_remainder_non_overlap=%d", params.dup_repeat_remainder_non_overlap);
  OP_LOGD(op_type.c_str(), " params.dup_remainder_non_overlap=%d", params.dup_remainder_non_overlap);
  OP_LOGD(op_type.c_str(), " params.repeats_non_overlap=%d", params.repeats_non_overlap);
  OP_LOGD(op_type.c_str(), " params.burst_len_ub2gm_2=%d", params.burst_len_ub2gm_2);
  OP_LOGD(op_type.c_str(), " params.src_stride_ub2gm_2=%d", params.src_stride_ub2gm_2);
  OP_LOGD(op_type.c_str(), " params.dst_stride_ub2gm_2=%d", params.dst_stride_ub2gm_2);
  OP_LOGD(op_type.c_str(), " params.burst_len_ub2gm_3=%d", params.burst_len_ub2gm_3);
  OP_LOGD(op_type.c_str(), " params.src_stride_ub2gm_3=%d", params.src_stride_ub2gm_3);
  OP_LOGD(op_type.c_str(), " params.dst_stride_ub2gm_3=%d", params.dst_stride_ub2gm_3);
  OP_LOGD(op_type.c_str(), " params.hi_val_tail=%d", params.hi_val_tail);
  OP_LOGD(op_type.c_str(), " params.burst_len_val=%d", params.burst_len_val);
  OP_LOGD(op_type.c_str(), " params.src_stride_val=%d", params.src_stride_val);
  OP_LOGD(op_type.c_str(), " params.dst_stride_val=%d", params.dst_stride_val);
  OP_LOGD(op_type.c_str(), " params.burst_len_val_tail=%d", params.burst_len_val_tail);
  OP_LOGD(op_type.c_str(), " params.src_stride_val_tail=%d", params.src_stride_val_tail);
  OP_LOGD(op_type.c_str(), " params.dst_stride_val_tail=%d", params.dst_stride_val_tail);
  OP_LOGD(op_type.c_str(), " params.num_instr_loop_h_tail=%d", params.num_instr_loop_h_tail);
  OP_LOGD(op_type.c_str(), " params.remain_repeat_tail=%d", params.remain_repeat_tail);
  OP_LOGD(op_type.c_str(), " params.num_instr_loop_h_1_tail=%d", params.num_instr_loop_h_1_tail);
  OP_LOGD(op_type.c_str(), " params.burst_len_ub_2_gm_tail=%d", params.burst_len_ub_2_gm_tail);
  OP_LOGD(op_type.c_str(), " params.non_overlap_1_tail=%d", params.non_overlap_1_tail);
  OP_LOGD(op_type.c_str(), " params.src_stride_over_tail=%d", params.src_stride_over_tail);
  OP_LOGD(op_type.c_str(), " params.dst_stride_over_tail=%d", params.dst_stride_over_tail);
  OP_LOGD(op_type.c_str(), " params.dup_repeat_merchant_non_overlap_tail=%d",
          params.dup_repeat_merchant_non_overlap_tail);
  OP_LOGD(op_type.c_str(), " params.dup_repeat_remainder_non_overlap_tail=%d",
          params.dup_repeat_remainder_non_overlap_tail);
  OP_LOGD(op_type.c_str(), " params.dup_remainder_non_overlap_tail=%d", params.dup_remainder_non_overlap_tail);
  OP_LOGD(op_type.c_str(), " params.repeats_non_overlap_tail=%d", params.repeats_non_overlap_tail);
  OP_LOGD(op_type.c_str(), " params.burst_len_ub2gm_2_tail=%d", params.burst_len_ub2gm_2_tail);
  OP_LOGD(op_type.c_str(), " params.src_stride_ub2gm_2_tail=%d", params.src_stride_ub2gm_2_tail);
  OP_LOGD(op_type.c_str(), " params.dst_stride_ub2gm_2_tail=%d", params.dst_stride_ub2gm_2_tail);
  OP_LOGD(op_type.c_str(), " params.burst_len_ub2gm_3_tail=%d", params.burst_len_ub2gm_3_tail);
  OP_LOGD(op_type.c_str(), " params.src_stride_ub2gm_3_tail=%d", params.src_stride_ub2gm_3_tail);
  OP_LOGD(op_type.c_str(), " params.dst_stride_ub2gm_3_tail=%d", params.dst_stride_ub2gm_3_tail);
  OP_LOGD(op_type.c_str(), " params.forward_in_shape_w_c0=%d", params.forward_in_shape_w_c0);
  OP_LOGD(op_type.c_str(), " params.dst_stride=%d", params.dst_stride);
}

void DivisionNDearest(int64_t number, int64_t base_num, int64_t core_num, int64_t& n1, int64_t& new_base_num) {
  n1 = number;
  new_base_num = base_num;
  for (int n0 = 1; n0 < number + 1; n0 = n0 + 1) {
    if (number % n0 == 0) {
      new_base_num = base_num * n0;
      n1 = number / n0;
      if (new_base_num >= core_num) {
        break;
      }
    }
  }
}

vector<int64_t> SplitPore(int64_t& n, int64_t& c1, int64_t& core_num, vector<int64_t>& ksize, vector<int64_t>& strides,
                          int64_t& ho_ys, int64_t& wo_ys, int64_t& h_ys, int64_t& w_ys) {
  int64_t base_num = 0;
  int64_t total_num = 0;
  int64_t real_core = 0;
  int64_t core_branch = 0;
  int64_t core_ou_shape_h = 0;
  int64_t core_ou_shape_w = 0;
  int64_t new_ho = 0;
  int64_t new_wo = 0;
  int64_t core_in_shape_h = 0;
  int64_t core_in_shape_w = 0;
  base_num = n * c1;
  core_ou_shape_h = ho_ys;
  core_ou_shape_w = wo_ys;

  if (base_num >= core_num) {
    total_num = base_num;
    real_core = core_num;
    core_branch = 0;
  } else if (base_num * ho_ys >= core_num) {
    DivisionNDearest(ho_ys, base_num, core_num, new_ho, total_num);
    real_core = core_num;
    core_ou_shape_h = new_ho;
    core_branch = 2;
  } else {
    base_num = base_num * ho_ys;
    new_ho = 1;
    DivisionNDearest(wo_ys, base_num, core_num, new_wo, total_num);
    core_ou_shape_h = new_ho;
    core_ou_shape_w = new_wo;
    real_core = total_num;
    if (total_num >= core_num) {
      real_core = core_num;
    }
    core_branch = 3;
  }

  bool true_false = true;
  InferDimReturn(core_ou_shape_h, core_ou_shape_w, true_false, ksize, strides, ho_ys, wo_ys, h_ys, w_ys,
                 core_in_shape_h, core_in_shape_w);

  vector<int64_t> list_data = {total_num,       real_core,       core_ou_shape_h, core_ou_shape_w,
                               core_in_shape_h, core_in_shape_w, core_branch};
  return list_data;
}

void InferMapReturn(int64_t ho, int64_t wo, vector<int64_t>& ksize, vector<int64_t>& strides, int64_t ho_ys,
                    int64_t wo_ys, int64_t h_ys, int64_t w_ys, vector<int64_t>& pad, int64_t& hi, int64_t& wi) {
  int64_t kh = ksize[INDEX_1];
  int64_t sh = strides[INDEX_1];
  int64_t kw = ksize[INDEX_2];
  int64_t sw = strides[INDEX_2];
  if (kh >= sh) {
    hi = kh + (ho - 1) * sh;
  } else {
    hi = ho * sh;
  }
  if (kw >= sw) {
    wi = kw + (wo - 1) * sw;
  } else {
    wi = wo * sw;
  }
  if (ho_ys == ho) {
    hi = h_ys + pad[INDEX_0] + pad[INDEX_1];
  }
  if (wo_ys == wo) {
    wi = w_ys + pad[INDEX_2] + pad[INDEX_3];
  }
}

void CheckProcessSpace(int64_t ho, int64_t wo, vector<int64_t>& params_ub, vector<int64_t>& ksize,
                       vector<int64_t>& strides, int64_t ho_ys, int64_t wo_ys, int64_t h_ys, int64_t w_ys,
                       const string& padding, vector<int64_t>& pad, int64_t ub_size, int64_t l1_size, bool& ub_split) {
  bool true_false = true;
  int64_t infer_hi = 0;
  int64_t infer_wi = 0;
  InferDimReturn(ho, wo, true_false, ksize, strides, ho_ys, wo_ys, h_ys, w_ys, infer_hi, infer_wi);

  int64_t l1_in_size = infer_hi * infer_wi * C0;

  int64_t col_in_shape = ho * wo * C0;
  int64_t min_col_in_size = 256;
  int64_t min_output_size = 128;
  int64_t col_in_size = UssCeilDiv(col_in_shape, min_col_in_size) * min_col_in_size;

  int64_t forward_ou_size = UssCeilDiv(col_in_shape, min_output_size) * min_output_size;

  int64_t mask_shape = ho * wo;
  int64_t mask_size = UssCeilDiv(mask_shape, min_output_size) * min_output_size;
  int64_t grad_size = forward_ou_size;
  int64_t zero_size = 128;

  int64_t grad_sel_fp16_size = UssCeilDiv(col_in_shape, min_output_size) * min_output_size;
  int64_t grad_sel_fp32_size = grad_sel_fp16_size;
  int64_t f_map_fp32_size = 0;
  if (padding == "VALID") {
    f_map_fp32_size = infer_hi * infer_wi * C0;
  } else {
    int64_t map_hi = 0;
    int64_t map_wi = 0;
    InferMapReturn(ho, wo, ksize, strides, ho_ys, wo_ys, h_ys, w_ys, pad, map_hi, map_wi);
    f_map_fp32_size = map_hi * map_wi * C0;
  }

  int64_t used_ub_byte =
      (col_in_size + forward_ou_size + mask_size * 3 + grad_size + zero_size + grad_sel_fp16_size) * BYTE16 +
      (grad_sel_fp32_size + f_map_fp32_size) * BYTE32;
  bool l1_split = l1_in_size > (l1_size / TILING_FACTOR_TWO);
  int64_t col_in_size_ub = ((ub_size - min_col_in_size) / (198 + 64 * strides[INDEX_1] * strides[INDEX_2])) * C0;
  if (col_in_size_ub < min_col_in_size) {
    col_in_size_ub = min_col_in_size;
  }
  int64_t forward_ou_size_ub = col_in_size_ub;
  int64_t mask_size_ub = (ub_size - min_col_in_size) / (198 + 64 * strides[INDEX_1] * strides[INDEX_2]);
  if (mask_size_ub < min_output_size) {
    mask_size_ub = min_output_size;
  }
  int64_t grad_size_ub = col_in_size_ub;
  int64_t zero_size_ub = 128;
  int64_t grad_sel_fp16_size_ub = col_in_size_ub;
  int64_t grad_sel_fp32_size_ub = col_in_size_ub;
  int64_t used_ub_byte_ub =
      (col_in_size_ub + forward_ou_size_ub + mask_size_ub * 3 + grad_size_ub + zero_size_ub + grad_sel_fp16_size_ub) *
      BYTE16;

  int64_t f_map_fp32_size_ub = (ub_size - 288 - used_ub_byte_ub) / 4 - grad_sel_fp32_size_ub;

  bool col_in_size_split = col_in_size > col_in_size_ub;
  bool mask_size_split = mask_size > mask_size_ub;
  bool f_map_fp32_size_split = f_map_fp32_size > f_map_fp32_size_ub;
  bool split = used_ub_byte > ub_size;
  ub_split = split || col_in_size_split || mask_size_split || f_map_fp32_size_split || l1_split;

  params_ub = {l1_in_size, col_in_size,        forward_ou_size,    mask_size,      grad_size,
               zero_size,  grad_sel_fp16_size, grad_sel_fp32_size, f_map_fp32_size};
}

int64_t CheckCutModel(bool split_do, bool split_ho, bool split_wo, vector<int64_t>& split_model, int64_t core_branch) {
  int64_t model = -1;
  if (split_do && (!split_ho) && (!split_wo)) {
    model = 0;
  } else if (split_do && split_ho && (!split_wo)) {
    model = 2;
  } else {
    model = 3;
  }
  if (model < core_branch) {
    model = core_branch;
  }
  return model;
}

void Pattern(int64_t core_ou_shape_h, int64_t core_ou_shape_w, int64_t core_branch, vector<int64_t>& ksize,
             vector<int64_t>& strides, int64_t ho_ys, int64_t wo_ys, int64_t h_ys, int64_t w_ys, const string& padding,
             vector<int64_t>& pad, int64_t ub_size, int64_t l1_size, int64_t& branch, int64_t& ho, int64_t& wo,
             vector<int64_t>& params_ub, bool& support) {
  int64_t all_wo = core_ou_shape_w;
  int64_t all_ho = core_ou_shape_h;

  wo = all_wo;
  ho = all_ho;
  bool split_do = false;
  bool split_ho = false;
  bool split_wo = false;
  bool ub_split = false;

  CheckProcessSpace(ho, wo, params_ub, ksize, strides, ho_ys, wo_ys, h_ys, w_ys, padding, pad, ub_size, l1_size,
                    ub_split);

  if (!ub_split) {
    split_do = true;
  }
  if (!split_do) {
    for (int k = 0; k < all_ho; k++) {
      ho = all_ho - k;
      CheckProcessSpace(ho, wo, params_ub, ksize, strides, ho_ys, wo_ys, h_ys, w_ys, padding, pad, ub_size, l1_size,
                        ub_split);
      if (!ub_split) {
        split_do = true;
        split_ho = true;
        break;
      }
    }
  }
  if ((!split_do) && (!split_ho)) {
    ho = 1;
    for (int k = 0; k < all_wo; k++) {
      wo = all_wo - k;
      CheckProcessSpace(ho, wo, params_ub, ksize, strides, ho_ys, wo_ys, h_ys, w_ys, padding, pad, ub_size, l1_size,
                        ub_split);
      if (!ub_split) {
        split_do = true;
        split_ho = true;
        split_wo = true;
        break;
      }
    }
  }

  if (wo == 1 && ho != 1) {
    wo = 1;
    ho = 1;
    split_do = true;
    split_ho = true;
    split_wo = true;
  }
  if ((!split_do) && (!split_ho) && (!split_wo)) {
    support = false;
  }
  vector<int64_t> split_model = {ho, wo};
  branch = CheckCutModel(split_do, split_ho, split_wo, split_model, core_branch);
  OP_LOGD(MaxPoolGrad_OP_TYPE.c_str(),
          "GetCompileParams, split_do is %d, split_ho is %d,"
          "split_wo is %d, branch is %d",
          split_do, split_ho, split_wo, branch);
}

// tiling function
bool MaxPoolGradTiling(const std::string& op_type, const ge::Operator& op_paras, const std::vector<int64_t>& op_info,
                       utils::OpRunInfo& run_info) {
  OP_LOGD(op_type.c_str(), "Entering MaxPoolGradTiling.");
  PROFILING_TILING_INIT(op_type.c_str());

  auto operator_info = OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get OpDesc failed."),
                  return false);

  auto forward_in_desc = operator_info->MutableInputDesc(INDEX_0);
  auto forward_ou_desc = operator_info->MutableInputDesc(INDEX_1);
  auto grad_desc = operator_info->MutableInputDesc(INDEX_2);

  OP_TILING_CHECK(forward_in_desc == nullptr || forward_ou_desc == nullptr || grad_desc == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get inputs size < 3."), return false);

  const GeShape& forward_in_shape = forward_in_desc->MutableShape();
  const GeShape& forward_ou_shape = forward_ou_desc->MutableShape();
  const GeShape& grad_shape = grad_desc->MutableShape();
  const GeShape& ou_shape = forward_in_shape;
  PROFILING_TILING_AFTER_GET_SHAPE_REG();

  OP_TILING_CHECK(forward_in_shape.GetDimNum() == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ori_input tensor is empty."), return false);
  OP_TILING_CHECK(forward_ou_shape.GetDimNum() == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ori_output tensor is empty."), return false);
  OP_TILING_CHECK(grad_shape.GetDimNum() == 0, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "grad tensor is empty."),
                  return false);

  bool flag = true;
  int64_t core_num_ys = 0;
  int64_t ub_size = 0;
  int64_t l1_size = 0;
  int64_t kh;
  int64_t kw;
  int64_t sh;
  int64_t sw;
  int64_t padding_int = 0;
  string padding;
  flag = GetUssCompileParams(op_type, op_info, core_num_ys, ub_size, kh, kw, sh, sw, padding_int, l1_size);
  OP_LOGD(op_type.c_str(),
          "GetCompileParams, core_num is %d, ub_size is %d. l1_size is %d,"
          "padding_int is %d",
          core_num_ys, ub_size, l1_size, padding_int);
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  if (!flag) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetCompileParams failed.");
    return false;
  }
  flag = CheckParam(forward_in_shape, forward_ou_shape, grad_shape, ou_shape);
  if (!flag) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CheckParam failed.");
    return false;
  }
  TilingParams params;
  InitTilingParams(params);
  if (padding_int == 0) {
    padding = "VALID";
  } else {
    padding = "SAME";
  }
  OP_LOGD(op_type.c_str(), "GetCompileParams, kh is %d, kw is %d, sh is %d, sw is %d", kh, kw, sh, sw);

  params.n = forward_in_shape.GetDim(INDEX_0);
  params.c1 = forward_in_shape.GetDim(INDEX_1);
  params.h = forward_in_shape.GetDim(INDEX_2);
  params.w = forward_in_shape.GetDim(INDEX_3);
  params.ho = forward_ou_shape.GetDim(INDEX_2);
  params.wo = forward_ou_shape.GetDim(INDEX_3);
  std::vector<int64_t> pad = PaddingMode(forward_in_shape, grad_shape, kh, kw, sh, sw, padding);
  params.pad_hw_top = pad[INDEX_0];
  params.pad_hw_bottom = pad[INDEX_1];
  params.pad_hw_left = pad[INDEX_2];
  params.pad_hw_right = pad[INDEX_3];
  std::vector<int64_t> ksize = {1, kh, kw, 1};
  std::vector<int64_t> strides = {1, sh, sw, 1};
  bool bool_data = true;

  params.overlap_h = OverlapMode(sh, kh, params.ho, params.h);
  params.overlap_w = OverlapMode(sw, kw, params.wo, params.w);
  bool true_false = false;

  GetInvalidPart(params.ho, params.wo, params.h, params.w, true_false, ksize, strides, params.ho, params.wo,
                 params.hi_invalid, params.wi_invalid);
  int64_t core_branch = 0;
  std::vector<int64_t> list_data;

  list_data = SplitPore(params.n, params.c1, core_num_ys, ksize, strides, params.ho, params.wo, params.h, params.w);
  params.total_num = list_data[INDEX_0];
  params.core_num = list_data[INDEX_1];
  params.core_ou_shape_h = list_data[INDEX_2];
  params.core_ou_shape_w = list_data[INDEX_3];
  params.core_in_shape_h = list_data[INDEX_4];
  params.core_in_shape_w = list_data[INDEX_5];
  core_branch = list_data[INDEX_6];
  bool support = true;
  OP_LOGD(op_type.c_str(), "GetCompileParams, core_branch is %d ,padding is %s", core_branch, padding.c_str());
  std::vector<int64_t> params_ub = {};
  Pattern(params.core_ou_shape_h, params.core_ou_shape_w, core_branch, ksize, strides, params.ho, params.wo, params.h,
          params.w, padding, pad, ub_size, l1_size, params.select_key, params.new_ho, params.new_wo, params_ub,
          support);
  if (!support) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "kernel is too larger !!!");
    return false;
  }
  OP_LOGD(op_type.c_str(), "GetCompileParams, params.select_key is %d", params.select_key);
  OP_LOGD(op_type.c_str(),
          "GetCompileParams, l1_in_size is %d, col_in_size is %d, forward_ou_size is %d,"
          "mask_size is %d, grad_size is %d, zero_size is %d, grad_sel_fp16_size is %d, grad_sel_fp32_size is %d,"
          "f_map_fp32_size is %d",
          params_ub[INDEX_0], params_ub[INDEX_1], params_ub[INDEX_2], params_ub[INDEX_3], params_ub[INDEX_4],
          params_ub[INDEX_5], params_ub[INDEX_6], params_ub[INDEX_7], params_ub[INDEX_8]);

  if (padding == "VALID") {
    if (core_branch == 0) {
      if (params.select_key == 2) {
        params.select_key = CASE_TILING_HO;
      } else if (params.select_key != 0) {
        params.select_key = CASE_TILING_HO_WO;
      } else {
        params.select_key = CASE_NO_TILING;
      }
    } else {
      if (params.select_key == 2) {
        params.select_key = CASE_CORE_HO;
      } else {
        params.select_key = CASE_CORE_HO_WO;
      }
    }
  } else {
    if (params.select_key == 0) {
      params.select_key = CASE_SAME_NO_TILING;
    } else if (params.select_key == 2) {
      params.select_key = CASE_SAME_TILING_HO;
    } else {
      params.select_key = CASE_SAME_TILING_HO_WO;
    }
  }

  if (params.select_key == CASE_TILING_HO_WO) {
    params.total_num_div_core = params.total_num % params.core_num;
    params.total_num_div_core_1 = params.total_num_div_core;
    params.core_loop_params = params.total_num / params.core_num;
    params.core_loop_params1 = (params.total_num + params.core_num - 1) / params.core_num;

    VectorDup(params_ub[INDEX_8], DTYPE_FP32, params.dup_repeat_merchant_f_map_fp32,
              params.dup_repeat_remainder_f_map_fp32,
              params.dup_remainder_f_map_fp32, params.repeats_f_map_fp32);

    params.loop_ho = params.ho / params.new_ho;
    params.loop_wo = params.wo / params.new_wo;
    params.wo_tail = params.wo % params.new_wo;

    InferDimReturn(params.new_ho, params.new_wo, bool_data, ksize, strides, params.ho, params.wo, params.h, params.w,
                   params.hi_batch, params.wi_batch);

    InferDimReturn(params.new_ho, params.wo_tail, bool_data, ksize, strides, params.ho, params.wo, params.h, params.w,
                   params.hi_tail, params.wi_tail);
    if (params.overlap_h > 0) {
      params.hi_val = 0 + params.hi_batch;
    } else {
      params.hi_val = params.overlap_h + params.hi_batch;
    }
    if (params.overlap_w > 0) {
      params.wi_val = 0 + params.wi_batch;
    } else {
      params.wi_val = params.overlap_w + params.wi_batch;
    }
    params.burst_len = params.wi_val * C0 * BYTE16 / BYTE_BLOCK;

    params.src_stride = (params.w - params.wi_val) * C0 * BYTE16 / BYTE_BLOCK;

    params.burst_len_src_orig_y = params.new_ho * params.new_wo * C0 * BYTE16 / BYTE_BLOCK;
    params.src_stride_src_orig_y =
        ((forward_ou_shape.GetDim(INDEX_2) * forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4)) *
             (params.c1 - 1) +
         forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4) * (params.ho - params.new_ho) +
         C0 * (params.wo - params.new_wo)) *
        BYTE16 / BYTE_BLOCK;
    params.forward_in_shape_h_w_c0 = forward_in_shape.GetDim(INDEX_1) * forward_in_shape.GetDim(INDEX_2) *
                                     forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4);
    params.forward_ou_shape_h_w_c0 = forward_ou_shape.GetDim(INDEX_1) * forward_ou_shape.GetDim(INDEX_2) *
                                     forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4);
    params.repeat_times = UssCeilDiv(params.new_ho * params.new_wo, LOAD3D_NUM_EACH_REPEAT);
    params.howo_co_ver = UssCeilDiv(params.new_ho * params.new_wo * C0, VCMP_NUM_EACH_REPEAT);
    params.mask_size_16 = params_ub[INDEX_3] / FP16_BLOCK_NUM;
    params.mask_size_ver = params_ub[INDEX_3] / MASK_FP16;

    VectorDup2(params_ub[INDEX_6], params.repeat_max_time_grad_sel, params.remain_repeat_time_grad_sel,
               params.remain_ele_grad_sel);
    int64_t w_2 = params.new_wo * C0 / TILING_FACTOR_TWO;
    VectorDup3(w_2, DTYPE_FP32, params.repeat_max_loop_vadd, params.remain_max_loop_vadd, params.remain_ele_vadd);
    params.burst_len_ub_2_gm = params.wi_val * C0 * BYTE32 / BYTE_BLOCK;
    params.src_stride_ub_2_gm = (params.wi_batch - params.wi_val) * C0 * BYTE32 / BYTE_BLOCK;
    params.dst_stride_ub_2_gm = (params.w - params.wi_val) * C0 * BYTE32 / BYTE_BLOCK;
    VectorDup2(params_ub[INDEX_8], params.repeat_max_loop_f_map_fp32, params.remain_max_loop_f_map_fp32,
               params.remain_ele_f_map_fp32);

    if (params.wo_tail != 0) {
      if (params.overlap_w > 0) {
        params.wi_val_tail = 0 + params.wi_tail;
      } else {
        params.wi_val_tail = params.overlap_w + params.wi_tail;
      }
      if (params.overlap_h > 0) {
        params.hi_val_tail = 0 + params.hi_tail;
      } else {
        params.hi_val_tail = params.overlap_h + params.hi_tail;
      }

      params.burst_len_tail = params.wi_val_tail * C0 * BYTE16 / BYTE_BLOCK;

      params.src_stride_tail = (params.w - params.wi_val_tail) * C0 * BYTE16 / BYTE_BLOCK;

      params.burst_len_src_orig_y_tail = params.new_ho * params.wo_tail * C0 * BYTE16 / BYTE_BLOCK;
      params.src_stride_src_orig_y_tail =
          ((forward_ou_shape.GetDim(INDEX_2) * forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4)) *
               (params.c1 - 1) +
           forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4) * (params.ho - params.new_ho) +
           C0 * (params.wo - params.wo_tail)) *
          BYTE16 / BYTE_BLOCK;
      params.repeat_times_tail = UssCeilDiv(params.new_ho * params.wo_tail, LOAD3D_NUM_EACH_REPEAT);
      params.howo_co_ver_tail = UssCeilDiv(params.new_ho * params.wo_tail * C0, VCMP_NUM_EACH_REPEAT);
      int64_t w_2_tail = params.wo_tail * C0 / TILING_FACTOR_TWO;
      VectorDup3(w_2_tail, DTYPE_FP32, params.repeat_max_loop_vadd_tail, params.remain_max_loop_vadd_tail,
                 params.remain_ele_vadd_tail);
      params.burst_len_ub_2_gm_tail = params.wi_val_tail * C0 * BYTE32 / BYTE_BLOCK;
      params.src_stride_ub_2_gm_tail = (params.wi_batch - params.wi_val_tail) * C0 * BYTE32 / BYTE_BLOCK;
      params.dst_stride_ub_2_gm_tail = (params.w - params.wi_val_tail) * C0 * BYTE32 / BYTE_BLOCK;
    }
  } else if (params.select_key == CASE_CORE_HO_WO) {
    params.total_num_div_core = params.total_num % params.core_num;
    params.total_num_div_core_1 = params.total_num % core_num_ys;
    params.core_loop_params = params.total_num / params.core_num;
    params.core_loop_params1 = (params.total_num + params.core_num - 1) / params.core_num;
    params.loop_ho = params.core_ou_shape_h / params.new_ho;
    params.loop_wo = params.core_ou_shape_w / params.new_wo;
    params.wo_tail = params.core_ou_shape_w % params.new_wo;

    InferDimReturn(params.new_ho, params.new_wo, bool_data, ksize, strides, params.ho, params.wo, params.h, params.w,
                   params.hi_batch, params.wi_batch);

    InferDimReturn(params.new_ho, params.wo_tail, bool_data, ksize, strides, params.ho, params.wo, params.h, params.w,
                   params.hi_tail, params.wi_tail);
    VectorDup(params_ub[INDEX_8], DTYPE_FP32, params.dup_repeat_merchant_f_map_fp32,
              params.dup_repeat_remainder_f_map_fp32, params.dup_remainder_f_map_fp32, params.repeats_f_map_fp32);
    params.core_ho_times = params.ho / params.core_ou_shape_h;
    params.core_wo_times = params.wo / params.core_ou_shape_w;
    params.forward_in_shape_h_w_c0 = forward_in_shape.GetDim(INDEX_1) * forward_in_shape.GetDim(INDEX_2) *
                                     forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4);
    params.forward_ou_shape_h_w_c0 = forward_ou_shape.GetDim(INDEX_1) * forward_ou_shape.GetDim(INDEX_2) *
                                     forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4);
    if (params.overlap_h > 0) {
      params.hi_val = 0 + params.hi_batch;
    } else {
      params.hi_val = params.overlap_h + params.hi_batch;
    }
    if (params.overlap_w > 0) {
      params.wi_val = 0 + params.wi_batch;
    } else {
      params.wi_val = params.overlap_w + params.wi_batch;
    }
    params.burst_len = params.wi_val * C0 * BYTE16 / BYTE_BLOCK;

    params.src_stride = (params.w - params.wi_val) * C0 * BYTE16 / BYTE_BLOCK;
    params.burst_len_src_orig_y = params.new_ho * params.new_wo * C0 * BYTE16 / BYTE_BLOCK;
    params.src_stride_src_orig_y =
        ((forward_ou_shape.GetDim(INDEX_2) * forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4)) *
             (params.c1 - 1) +
         forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4) * (params.ho - params.new_ho) +
         C0 * (params.wo - params.new_wo)) *
        BYTE16 / BYTE_BLOCK;
    params.repeat_times = UssCeilDiv(params.new_ho * params.new_wo, LOAD3D_NUM_EACH_REPEAT);
    params.howo_co_ver = UssCeilDiv(params.new_ho * params.new_wo * C0, VCMP_NUM_EACH_REPEAT);
    params.mask_size_16 = params_ub[INDEX_3] / FP16_BLOCK_NUM;
    params.mask_size_ver = params_ub[INDEX_3] / MASK_FP16;
    VectorDup2(params_ub[INDEX_6], params.repeat_max_time_grad_sel, params.remain_repeat_time_grad_sel,
               params.remain_ele_grad_sel);
    int64_t w_2 = params.new_wo * C0 / TILING_FACTOR_TWO;
    VectorDup3(w_2, DTYPE_FP32, params.repeat_max_loop_vadd, params.remain_max_loop_vadd, params.remain_ele_vadd);
    params.burst_len_ub2gm_2 = params.wi_val * C0 * BYTE32 / BYTE_BLOCK;
    params.src_stride_ub_2_gm = (params.wi_batch - params.wi_val) * C0 * BYTE32 / BYTE_BLOCK;
    params.dst_stride_ub_2_gm = (params.w - params.wi_val) * C0 * BYTE32 / BYTE_BLOCK;
    VectorDup2(params_ub[INDEX_8], params.repeat_max_loop_f_map_fp32, params.remain_max_loop_f_map_fp32,
               params.remain_ele_f_map_fp32);
    if (params.wo_tail != 0) {
      if (params.overlap_w > 0) {
        params.wi_val_tail = 0 + params.wi_tail;
      } else {
        params.wi_val_tail = params.overlap_w + params.wi_tail;
      }
      if (params.overlap_h > 0) {
        params.hi_val_tail = 0 + params.hi_tail;
      } else {
        params.hi_val_tail = params.overlap_h + params.hi_tail;
      }
      params.burst_len_tail = params.wi_val_tail * C0 * BYTE16 / BYTE_BLOCK;
      params.src_stride_tail = (params.w - params.wi_val_tail) * C0 * BYTE16 / BYTE_BLOCK;
      params.burst_len_src_orig_y_tail = params.new_ho * params.wo_tail * C0 * BYTE16 / BYTE_BLOCK;
      params.src_stride_src_orig_y_tail =
          ((forward_ou_shape.GetDim(INDEX_2) * forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4)) *
               (params.c1 - 1) +
           forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4) * (params.ho - params.new_ho) +
           C0 * (params.wo - params.wo_tail)) *
          BYTE16 / BYTE_BLOCK;
      params.repeat_times_tail = UssCeilDiv(params.new_ho * params.wo_tail, LOAD3D_NUM_EACH_REPEAT);
      params.howo_co_ver_tail = UssCeilDiv(params.new_ho * params.wo_tail * C0, VCMP_NUM_EACH_REPEAT);
      int64_t w_2_tail = params.wo_tail * C0 / TILING_FACTOR_TWO;
      VectorDup3(w_2_tail, DTYPE_FP32, params.repeat_max_loop_vadd_tail, params.remain_max_loop_vadd_tail,
                 params.remain_ele_vadd_tail);
      params.burst_len_ub2gm_2_tail = params.wi_val_tail * C0 * BYTE32 / BYTE_BLOCK;
      params.src_stride_ub_2_gm_tail = (params.wi_batch - params.wi_val_tail) * C0 * BYTE32 / BYTE_BLOCK;
      params.dst_stride_ub_2_gm_tail = (params.w - params.wi_val_tail) * C0 * BYTE32 / BYTE_BLOCK;
    }
  } else if (params.select_key == CASE_SAME_TILING_HO_WO) {
    params.total_num_div_core = params.total_num % params.core_num;
    params.total_num_div_core_1 = params.total_num % core_num_ys;
    params.core_loop_params = params.total_num / params.core_num;
    params.core_loop_params1 = (params.total_num + params.core_num - 1) / params.core_num;
    params.loop_ho = params.core_ou_shape_h / params.new_ho;
    params.loop_wo = params.core_ou_shape_w / params.new_wo;
    params.wo_tail = params.core_ou_shape_w % params.new_wo;

    InferDimReturn(params.new_ho, params.new_wo, bool_data, ksize, strides, params.ho, params.wo, params.h, params.w,
                   params.hi_batch, params.wi_batch);

    InferDimReturn(params.new_ho, params.wo_tail, bool_data, ksize, strides, params.ho, params.wo, params.h, params.w,
                   params.hi_tail, params.wi_tail);
    InferMapReturn(params.new_ho, params.new_wo, ksize, strides, params.ho, params.wo, params.h, params.w, pad,
                   params.map_hi, params.map_wi);
    VectorDup(params_ub[INDEX_8], DTYPE_FP32, params.dup_repeat_merchant_f_map_fp32,
              params.dup_repeat_remainder_f_map_fp32, params.dup_remainder_f_map_fp32, params.repeats_f_map_fp32);

    params.core_ho_times = params.ho / params.core_ou_shape_h;

    params.core_wo_times = params.wo / params.core_ou_shape_w;
    params.forward_in_shape_h_w_c0 = forward_in_shape.GetDim(INDEX_1) * forward_in_shape.GetDim(INDEX_2) *
                                     forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4);
    params.forward_ou_shape_h_w_c0 = forward_ou_shape.GetDim(INDEX_1) * forward_ou_shape.GetDim(INDEX_2) *
                                     forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4);
    if (params.overlap_h > 0) {
      params.hi_val = 0 + params.hi_batch;
    } else {
      params.hi_val = params.overlap_h + params.hi_batch;
    }
    if (params.overlap_w > 0) {
      params.wi_val = 0 + params.wi_batch;
    } else {
      params.wi_val = params.overlap_w + params.wi_batch;
    }
    params.burst_len_src_orig_y = params.new_ho * params.new_wo * C0 * BYTE16 / BYTE_BLOCK;
    params.src_stride_src_orig_y =
        ((forward_ou_shape.GetDim(INDEX_2) * forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4)) *
             (params.c1 - 1) +
         forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4) * (params.ho - params.new_ho) +
         C0 * (params.wo - params.new_wo)) *
        BYTE16 / BYTE_BLOCK;
    params.repeat_times = UssCeilDiv(params.new_ho * params.new_wo, LOAD3D_NUM_EACH_REPEAT);
    params.howo_co_ver = UssCeilDiv(params.new_ho * params.new_wo * C0, VCMP_NUM_EACH_REPEAT);
    params.mask_size_16 = params_ub[INDEX_3] / FP16_BLOCK_NUM;
    params.mask_size_ver = params_ub[INDEX_3] / MASK_FP16;
    VectorDup2(params_ub[INDEX_6], params.repeat_max_time_grad_sel, params.remain_repeat_time_grad_sel,
               params.remain_ele_grad_sel);
    int64_t w_2 = params.new_wo * C0 / TILING_FACTOR_TWO;
    VectorDup3(w_2, DTYPE_FP32, params.repeat_max_loop_vadd, params.remain_max_loop_vadd, params.remain_ele_vadd);

    VectorDup(params_ub[INDEX_8], DTYPE_FP32, params.dup_repeat_merchant_f_map_fp32,
              params.dup_repeat_remainder_f_map_fp32, params.dup_remainder_f_map_fp32, params.repeats_f_map_fp32);

    if (params.wo_tail != 0) {
      if (params.overlap_w > 0) {
        params.wi_val_tail = 0 + params.wi_tail;
      } else {
        params.wi_val_tail = params.overlap_w + params.wi_tail;
      }
      if (params.overlap_h > 0) {
        params.hi_val_tail = 0 + params.hi_tail;
      } else {
        params.hi_val = params.overlap_h + params.hi_tail;
      }
      params.burst_len_src_orig_y_tail = params.new_ho * params.wo_tail * C0 * BYTE16 / BYTE_BLOCK;
      params.src_stride_src_orig_y_tail =
          ((forward_ou_shape.GetDim(INDEX_2) * forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4)) *
               (params.c1 - 1) +
           forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4) * (params.ho - params.new_ho) +
           C0 * (params.wo - params.wo_tail)) *
          BYTE16 / BYTE_BLOCK;
      params.repeat_times_tail = UssCeilDiv(params.new_ho * params.wo_tail, LOAD3D_NUM_EACH_REPEAT);
      params.howo_co_ver_tail = UssCeilDiv(params.new_ho * params.wo_tail * C0, VCMP_NUM_EACH_REPEAT);
      int64_t w_2_tail = params.wo_tail * C0 / TILING_FACTOR_TWO;
      VectorDup3(w_2_tail, DTYPE_FP32, params.repeat_max_loop_vadd_tail, params.remain_max_loop_vadd_tail,
                 params.remain_ele_vadd_tail);
    }
  } else if (params.select_key == CASE_NO_TILING) {
    params.total_num_div_core = params.total_num % params.core_num;
    params.total_num_div_core_1 = params.total_num_div_core;
    params.core_loop_params = params.total_num / params.core_num;
    params.core_loop_params1 = (params.total_num + params.core_num - 1) / params.core_num;

    InferDimReturn(params.new_ho, params.new_wo, bool_data, ksize, strides, params.ho, params.wo, params.h, params.w,
                   params.hi_batch, params.wi_batch);
    VectorDup(params_ub[INDEX_8], DTYPE_FP32, params.dup_repeat_merchant_f_map_fp32,
              params.dup_repeat_remainder_f_map_fp32, params.dup_remainder_f_map_fp32, params.repeats_f_map_fp32);
    params.forward_in_shape_h_w_c0 = forward_in_shape.GetDim(INDEX_1) * forward_in_shape.GetDim(INDEX_2) *
                                     forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4);
    params.forward_ou_shape_h_w_c0 = forward_ou_shape.GetDim(INDEX_1) * forward_ou_shape.GetDim(INDEX_2) *
                                     forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4);
    params.burst_len = params.hi_batch * params.wi_batch * C0 * BYTE16 / BYTE_BLOCK;

    params.src_stride =
        (forward_in_shape.GetDim(INDEX_2) * forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) *
             (params.c1 - 1) +
         forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) * (params.h - params.hi_batch)) *
        BYTE16 / BYTE_BLOCK;

    params.burst_len_src_orig_y = params.new_ho * params.new_wo * C0 * BYTE16 / BYTE_BLOCK;
    params.src_stride_src_orig_y =
        ((forward_ou_shape.GetDim(INDEX_2) * forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4)) *
             (params.c1 - 1) +
         forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4) * (params.ho - params.new_ho) +
         C0 * (params.wo - params.new_wo)) *
        BYTE16 / BYTE_BLOCK;
    params.repeat_times = UssCeilDiv(params.new_ho * params.new_wo, LOAD3D_NUM_EACH_REPEAT);
    params.howo_co_ver = UssCeilDiv(params.new_ho * params.new_wo * C0, VCMP_NUM_EACH_REPEAT);
    params.mask_size_16 = params_ub[INDEX_3] / FP16_BLOCK_NUM;
    params.mask_size_ver = params_ub[INDEX_3] / MASK_FP16;
    VectorDup2(params_ub[INDEX_6], params.repeat_max_time_grad_sel, params.remain_repeat_time_grad_sel,
               params.remain_ele_grad_sel);
    params.wo_2 = params.new_wo * TILING_FACTOR_TWO;
    params.sh_wi_2 = strides[1] * params.wi_batch * TILING_FACTOR_TWO;
    vector<int64_t> config_list = {strides[INDEX_2] * TILING_FACTOR_TWO,
                                   strides[INDEX_2] * TILING_FACTOR_TWO,
                                   MODE_TWO,
                                   params.sh_wi_2,
                                   params.sh_wi_2,
                                   params.wo_2};
    params.config = CheckConfig(config_list);
    if (params.config == 1) {
      params.num_instr_loop_h = UssCeilDiv(params.new_ho, MAX_REPEAT_TIME);
      params.num_instr_loop_w = UssCeilDiv(params.new_wo * C0 / TILING_FACTOR_TWO, MASK_FP32);
      params.remain_mask = (params.new_wo * C0 / TILING_FACTOR_TWO) % MASK_FP32;
      if (params.remain_mask == 0 && ((params.new_wo * C0 / TILING_FACTOR_TWO) != 0)) {
        params.remain_mask = MASK_FP32;
      }

      params.remain_repeat = params.new_ho % MAX_REPEAT_TIME;
      if (params.remain_repeat == 0 && (params.new_ho != 0)) {
        params.remain_repeat = MAX_REPEAT_TIME;
      }
      params.num_instr_loop_w_1 = params.num_instr_loop_w - 1;
      params.num_instr_loop_h_1 = params.num_instr_loop_h - 1;
    }
    int64_t w_2 = params.new_wo * C0 / TILING_FACTOR_TWO;
    VectorDup3(w_2, DTYPE_FP32, params.repeat_max_loop_vadd, params.remain_max_loop_vadd, params.remain_ele_vadd);
    params.dst_stride_ub_2_gm =
        (forward_in_shape.GetDim(INDEX_2) * forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) *
             (params.c1 - 1) +
         forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) * (params.h - params.hi_batch)) *
        BYTE32 / BYTE_BLOCK;
    params.burst_len_tail = params.hi_batch * params.wi_batch * C0 * BYTE32 / BYTE_BLOCK;
  } else if (params.select_key == CASE_TILING_HO) {
    params.total_num_div_core = params.total_num % params.core_num;
    params.total_num_div_core_1 = params.total_num_div_core;
    params.core_loop_params = params.total_num / params.core_num;
    params.core_loop_params1 = (params.total_num + params.core_num - 1) / params.core_num;
    params.loop_ho = params.ho / params.new_ho;
    params.ho_tail = params.ho % params.new_ho;
    InferDimReturn(params.new_ho, params.new_wo, bool_data, ksize, strides, params.ho, params.wo, params.h, params.w,
                   params.hi_batch, params.wi_batch);
    InferDimReturn(params.ho_tail, params.new_wo, bool_data, ksize, strides, params.ho, params.wo, params.h, params.w,
                   params.hi_tail, params.wi_tail);
    if (params.ho_tail == 0) {
      params.hi_tail = 0;
    }
    VectorDup(params_ub[INDEX_8], DTYPE_FP32, params.dup_repeat_merchant_f_map_fp32,
              params.dup_repeat_remainder_f_map_fp32, params.dup_remainder_f_map_fp32, params.repeats_f_map_fp32);
    params.forward_in_shape_h_w_c0 = forward_in_shape.GetDim(INDEX_1) * forward_in_shape.GetDim(INDEX_2) *
                                     forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4);
    params.forward_ou_shape_h_w_c0 = forward_ou_shape.GetDim(INDEX_1) * forward_ou_shape.GetDim(INDEX_2) *
                                     forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4);
    params.burst_len = params.hi_batch * params.wi_batch * C0 * BYTE16 / BYTE_BLOCK;

    params.src_stride =
        (forward_in_shape.GetDim(INDEX_2) * forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) *
             (params.c1 - 1) +
         forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) * (params.h - params.hi_batch)) *
        BYTE16 / BYTE_BLOCK;

    if (params.overlap_h < 0) {
      params.hi_val = params.overlap_h + params.hi_batch;
      params.burst_len_val = params.hi_val * params.wi_batch * C0 * BYTE16 / BYTE_BLOCK;
      params.src_stride_val =
          (forward_in_shape.GetDim(INDEX_2) * forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) *
               (params.c1 - 1) +
           forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) * (params.h - params.hi_val)) *
          BYTE16 / BYTE_BLOCK;
      params.dst_stride_val = (params.hi_batch - params.hi_val) * params.w * C0 * BYTE16 / BYTE_BLOCK;
    }

    params.burst_len_src_orig_y = params.new_ho * params.new_wo * C0 * BYTE16 / BYTE_BLOCK;
    params.src_stride_src_orig_y =
        ((forward_ou_shape.GetDim(INDEX_2) * forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4)) *
             (params.c1 - 1) +
         forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4) * (params.ho - params.new_ho) +
         C0 * (params.wo - params.new_wo)) *
        BYTE16 / BYTE_BLOCK;
    params.repeat_times = UssCeilDiv(params.new_ho * params.new_wo, LOAD3D_NUM_EACH_REPEAT);
    params.howo_co_ver = UssCeilDiv(params.new_ho * params.new_wo * C0, VCMP_NUM_EACH_REPEAT);
    params.mask_size_16 = params_ub[INDEX_3] / FP16_BLOCK_NUM;
    params.mask_size_ver = params_ub[INDEX_3] / MASK_FP16;
    VectorDup2(params_ub[INDEX_6], params.repeat_max_time_grad_sel, params.remain_repeat_time_grad_sel,
               params.remain_ele_grad_sel);
    params.wo_2 = params.new_wo * TILING_FACTOR_TWO;
    params.sh_wi_2 = strides[1] * params.wi_batch * TILING_FACTOR_TWO;
    vector<int64_t> config_list = {strides[INDEX_2] * TILING_FACTOR_TWO,
                                   strides[INDEX_2] * TILING_FACTOR_TWO,
                                   MODE_TWO,
                                   params.sh_wi_2,
                                   params.sh_wi_2,
                                   params.wo_2};
    params.config = CheckConfig(config_list);
    if (params.config == 1) {
      params.num_instr_loop_h = UssCeilDiv(params.new_ho, MAX_REPEAT_TIME);
      params.num_instr_loop_w = UssCeilDiv(params.new_wo * C0 / TILING_FACTOR_TWO, MASK_FP32);
      params.remain_mask = (params.new_wo * C0 / TILING_FACTOR_TWO) % MASK_FP32;
      if (params.remain_mask == 0 && ((params.new_wo * C0 / TILING_FACTOR_TWO) != 0)) {
        params.remain_mask = MASK_FP32;
      }

      params.remain_repeat = params.new_ho % MAX_REPEAT_TIME;
      if (params.remain_repeat == 0 && (params.new_ho != 0)) {
        params.remain_repeat = MAX_REPEAT_TIME;
      }

      params.num_instr_loop_w_1 = params.num_instr_loop_w - 1;
      params.num_instr_loop_h_1 = params.num_instr_loop_h - 1;
    }
    int64_t w_2 = params.new_wo * C0 / TILING_FACTOR_TWO;
    VectorDup3(w_2, DTYPE_FP32, params.repeat_max_loop_vadd, params.remain_max_loop_vadd, params.remain_ele_vadd);
    if (params.hi_invalid < 0) {
      params.hi_invalid = 0;
    }
    params.boundary_h = params.h - params.hi_invalid;
    // in_shape = [1, hi-self.overlap_h, wi, c0]
    params.burst_len_ub_2_gm = (params.hi_batch - params.overlap_h) * params.wi_batch * C0 * BYTE32 / BYTE_BLOCK;
    params.src_stride_ub_2_gm = params.overlap_h * params.w * C0 * BYTE32 / BYTE_BLOCK;
    params.dst_stride_ub_2_gm = (forward_in_shape.GetDim(INDEX_2) * forward_in_shape.GetDim(INDEX_3) *
                                     forward_in_shape.GetDim(INDEX_4) * (params.c1 - 1) +
                                 forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) *
                                     (params.h - (params.hi_batch - params.overlap_h))) *
                                BYTE32 / BYTE_BLOCK;
    // in_shape = [num_d, hi, wi, c0]
    // overlap = [num_d, self.overlap_h, wi, c0]
    // non_overlap = [num_d, hi-self.overlap_h, wi, c0]
    params.non_overlap_1 = (params.hi_batch - params.overlap_h) * params.wi_batch * C0;
    params.overlap_1 = params.overlap_h * params.wi_batch * C0;
    params.burst_len_over = params.overlap_1 * BYTE32 / BYTE_BLOCK;
    params.src_stride_over = params.non_overlap_1 * BYTE32 / BYTE_BLOCK;
    params.dst_stride_over = params.src_stride_over;
    VectorDup(params.non_overlap_1, DTYPE_FP32, params.dup_repeat_merchant_non_overlap,
              params.dup_repeat_remainder_non_overlap, params.dup_remainder_non_overlap, params.repeats_non_overlap);

    params.burst_len_ub2gm_2 = params.hi_batch * params.wi_batch * C0 * BYTE32 / BYTE_BLOCK;
    params.src_stride_ub2gm_2 = 0;
    params.dst_stride_ub2gm_2 =
        ((forward_in_shape.GetDim(INDEX_2) * forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4)) *
             (params.c1 - 1) +
         forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) * (params.h - params.hi_batch)) *
        BYTE32 / BYTE_BLOCK;

    params.burst_len_ub2gm_3 = (params.hi_batch + params.hi_invalid) * params.wi_batch * C0 * BYTE32 / BYTE_BLOCK;
    params.src_stride_ub2gm_3 =
        (params.hi_batch - (params.hi_batch + params.hi_invalid)) * params.w * C0 * BYTE32 / BYTE_BLOCK;
    params.dst_stride_ub2gm_3 =
        ((forward_in_shape.GetDim(INDEX_2) * forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4)) *
             (params.c1 - 1) +
         forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) *
             (params.h - (params.hi_batch + params.hi_invalid))) *
        BYTE32 / BYTE_BLOCK;
    if (params.ho_tail != 0) {
      params.burst_len_tail = params.hi_tail * params.wi_batch * C0 * BYTE16 / BYTE_BLOCK;

      params.src_stride_tail =
          (forward_in_shape.GetDim(INDEX_2) * forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) *
               (params.c1 - 1) +
           forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) * (params.h - params.hi_tail)) *
          BYTE16 / BYTE_BLOCK;
      params.dst_stride_tail = (params.hi_batch - params.hi_tail) * params.w * C0 * BYTE16 / BYTE_BLOCK;
      if (params.overlap_h < 0) {
        params.hi_val_tail = params.overlap_h + params.hi_tail;
        params.burst_len_val_tail = params.hi_val_tail * params.wi_batch * C0 * BYTE16 / BYTE_BLOCK;
        params.src_stride_val_tail =
            (forward_in_shape.GetDim(INDEX_2) * forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) *
                 (params.c1 - 1) +
             forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) * (params.h - params.hi_val_tail)) *
            BYTE16 / BYTE_BLOCK;
        params.dst_stride_val_tail = (params.hi_batch - params.hi_val_tail) * params.w * C0 * BYTE16 / BYTE_BLOCK;
      }
      params.burst_len_src_orig_y_tail = params.ho_tail * params.new_wo * C0 * BYTE16 / BYTE_BLOCK;
      params.src_stride_src_orig_y_tail =
          ((forward_ou_shape.GetDim(INDEX_2) * forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4)) *
               (params.c1 - 1) +
           forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4) * (params.ho - params.ho_tail) +
           C0 * (params.wo - params.new_wo)) *
          BYTE16 / BYTE_BLOCK;
      params.repeat_times_tail = UssCeilDiv(params.ho_tail * params.new_wo, LOAD3D_NUM_EACH_REPEAT);
      params.howo_co_ver_tail = UssCeilDiv(params.ho_tail * params.new_wo * C0, VCMP_NUM_EACH_REPEAT);
      if (params.config == 1) {
        params.num_instr_loop_h_tail = UssCeilDiv(params.ho_tail, MAX_REPEAT_TIME);
        params.remain_repeat_tail = params.ho_tail % MAX_REPEAT_TIME;
        if (params.remain_repeat_tail == 0 && (params.ho_tail != 0)) {
          params.remain_repeat_tail = MAX_REPEAT_TIME;
        }
        params.num_instr_loop_h_1_tail = params.num_instr_loop_h_tail - 1;
      }
      params.burst_len_ub_2_gm_tail = (params.hi_tail - params.overlap_h) * params.wi_batch * C0 * BYTE32 / BYTE_BLOCK;

      params.dst_stride_ub_2_gm_tail = (forward_in_shape.GetDim(INDEX_2) * forward_in_shape.GetDim(INDEX_3) *
                                            forward_in_shape.GetDim(INDEX_4) * (params.c1 - 1) +
                                        forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) *
                                            (params.h - (params.hi_tail - params.overlap_h))) *
                                       BYTE32 / BYTE_BLOCK;
      params.non_overlap_1_tail = (params.hi_tail - params.overlap_h) * params.wi_batch * C0;
      params.src_stride_over_tail = params.non_overlap_1_tail * BYTE32 / BYTE_BLOCK;
      params.dst_stride_over_tail = params.src_stride_over_tail;
      VectorDup(params.non_overlap_1_tail, DTYPE_FP32, params.dup_repeat_merchant_non_overlap_tail,
                params.dup_repeat_remainder_non_overlap_tail, params.dup_remainder_non_overlap_tail,
                params.repeats_non_overlap_tail);
      params.burst_len_ub2gm_2_tail = params.hi_tail * params.wi_batch * C0 * BYTE32 / BYTE_BLOCK;
      params.src_stride_ub2gm_2_tail = (params.hi_batch - params.hi_tail) * params.w * C0 * BYTE32 / BYTE_BLOCK;
      params.dst_stride_ub2gm_2_tail =
          ((forward_in_shape.GetDim(INDEX_2) * forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4)) *
               (params.c1 - 1) +
           forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) * (params.h - params.hi_tail)) *
          BYTE32 / BYTE_BLOCK;

      params.burst_len_ub2gm_3_tail = (params.hi_tail + params.hi_invalid) * params.wi_batch * C0 * BYTE32 / BYTE_BLOCK;
      params.src_stride_ub2gm_3_tail =
          (params.hi_batch - (params.hi_tail + params.hi_invalid)) * params.w * C0 * BYTE32 / BYTE_BLOCK;
      params.dst_stride_ub2gm_3_tail =
          ((forward_in_shape.GetDim(INDEX_2) * forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4)) *
               (params.c1 - 1) +
           forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) *
               (params.h - (params.hi_tail + params.hi_invalid))) *
          BYTE32 / BYTE_BLOCK;
    }
  } else if (params.select_key == CASE_CORE_HO) {
    params.total_num_div_core = params.total_num % params.core_num;
    params.total_num_div_core_1 = params.total_num % core_num_ys;
    params.core_loop_params = params.total_num / params.core_num;
    params.core_loop_params1 = (params.total_num + params.core_num - 1) / params.core_num;
    params.loop_ho = params.core_ou_shape_h / params.new_ho;
    params.ho_tail = params.core_ou_shape_h % params.new_ho;
    InferDimReturn(params.new_ho, params.new_wo, bool_data, ksize, strides, params.ho, params.wo, params.h, params.w,
                   params.hi_batch, params.wi_batch);
    InferDimReturn(params.ho_tail, params.new_wo, bool_data, ksize, strides, params.ho, params.wo, params.h, params.w,
                   params.hi_tail, params.wi_tail);
    VectorDup(params_ub[INDEX_8], DTYPE_FP32, params.dup_repeat_merchant_f_map_fp32,
              params.dup_repeat_remainder_f_map_fp32, params.dup_remainder_f_map_fp32, params.repeats_f_map_fp32);
    params.core_ho_times = params.ho / params.core_ou_shape_h;
    params.forward_in_shape_h_w_c0 = forward_in_shape.GetDim(INDEX_1) * forward_in_shape.GetDim(INDEX_2) *
                                     forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4);
    params.forward_ou_shape_h_w_c0 = forward_ou_shape.GetDim(INDEX_1) * forward_ou_shape.GetDim(INDEX_2) *
                                     forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4);
    if (params.overlap_h > 0) {
      params.hi_val = 0 + params.hi_batch;
    } else {
      params.hi_val = params.overlap_h + params.hi_batch;
    }
    params.burst_len = params.hi_val * params.wi_batch * C0 * BYTE16 / BYTE_BLOCK;

    params.src_stride =
        (forward_in_shape.GetDim(INDEX_2) * forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) *
             (params.c1 - 1) +
         forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) * (params.h - params.hi_val)) *
        BYTE16 / BYTE_BLOCK;
    params.dst_stride = (params.hi_batch - params.hi_val) * params.w * C0 * BYTE16 / BYTE_BLOCK;
    params.burst_len_src_orig_y = params.new_ho * params.new_wo * C0 * BYTE16 / BYTE_BLOCK;
    params.src_stride_src_orig_y =
        ((forward_ou_shape.GetDim(INDEX_2) * forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4)) *
             (params.c1 - 1) +
         forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4) * (params.ho - params.new_ho) +
         C0 * (params.wo - params.new_wo)) *
        BYTE16 / BYTE_BLOCK;
    params.repeat_times = UssCeilDiv(params.new_ho * params.new_wo, LOAD3D_NUM_EACH_REPEAT);
    params.howo_co_ver = UssCeilDiv(params.new_ho * params.new_wo * C0, VCMP_NUM_EACH_REPEAT);
    params.mask_size_16 = params_ub[INDEX_3] / FP16_BLOCK_NUM;
    params.mask_size_ver = params_ub[INDEX_3] / MASK_FP16;
    VectorDup2(params_ub[INDEX_6], params.repeat_max_time_grad_sel, params.remain_repeat_time_grad_sel,
               params.remain_ele_grad_sel);
    params.wo_2 = params.new_wo * TILING_FACTOR_TWO;
    params.sh_wi_2 = strides[1] * params.wi_batch * TILING_FACTOR_TWO;
    vector<int64_t> config_list = {strides[INDEX_2] * TILING_FACTOR_TWO,
                                   strides[INDEX_2] * TILING_FACTOR_TWO,
                                   MODE_TWO,
                                   params.sh_wi_2,
                                   params.sh_wi_2,
                                   params.wo_2};
    params.config = CheckConfig(config_list);
    if (params.config == 1) {
      params.num_instr_loop_h = UssCeilDiv(params.new_ho, MAX_REPEAT_TIME);
      params.num_instr_loop_w = UssCeilDiv(params.new_wo * C0 / TILING_FACTOR_TWO, MASK_FP32);
      params.remain_mask = (params.new_wo * C0 / TILING_FACTOR_TWO) % MASK_FP32;
      if (params.remain_mask == 0 && ((params.new_wo * C0 / TILING_FACTOR_TWO) != 0)) {
        params.remain_mask = MASK_FP32;
      }

      params.remain_repeat = params.new_ho % MAX_REPEAT_TIME;
      if (params.remain_repeat == 0 && (params.new_ho != 0)) {
        params.remain_repeat = MAX_REPEAT_TIME;
      }
      params.num_instr_loop_w_1 = params.num_instr_loop_w - 1;
      params.num_instr_loop_h_1 = params.num_instr_loop_h - 1;
    }
    int64_t w_2 = params.new_wo * C0 / TILING_FACTOR_TWO;
    VectorDup3(w_2, DTYPE_FP32, params.repeat_max_loop_vadd, params.remain_max_loop_vadd, params.remain_ele_vadd);
    params.burst_len_ub_2_gm = params.hi_val * params.wi_batch * C0 * BYTE32 / BYTE_BLOCK;
    params.src_stride_ub_2_gm = (params.hi_batch - params.hi_val) * params.w * C0 * BYTE32 / BYTE_BLOCK;
    params.dst_stride_ub_2_gm =
        (forward_in_shape.GetDim(INDEX_2) * forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) *
             (params.c1 - 1) +
         forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) * (params.h - params.hi_val)) *
        BYTE32 / BYTE_BLOCK;
    if (params.ho_tail != 0) {
      if (params.overlap_h > 0) {
        params.hi_val_tail = 0 + params.hi_tail;
      } else {
        params.hi_val_tail = params.overlap_h + params.hi_tail;
      }
      params.burst_len_tail = params.hi_val_tail * params.wi_batch * C0 * BYTE16 / BYTE_BLOCK;

      params.src_stride_tail =
          (forward_in_shape.GetDim(INDEX_2) * forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) *
               (params.c1 - 1) +
           forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) * (params.h - params.hi_val_tail)) *
          BYTE16 / BYTE_BLOCK;
      params.dst_stride_tail = (params.hi_batch - params.hi_val_tail) * params.w * C0 * BYTE16 / BYTE_BLOCK;
      params.burst_len_src_orig_y_tail = params.ho_tail * params.new_wo * C0 * BYTE16 / BYTE_BLOCK;
      params.src_stride_src_orig_y_tail =
          ((forward_ou_shape.GetDim(INDEX_2) * forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4)) *
               (params.c1 - 1) +
           forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4) * (params.ho - params.ho_tail) +
           C0 * (params.wo - params.new_wo)) *
          BYTE16 / BYTE_BLOCK;
      params.repeat_times_tail = UssCeilDiv(params.ho_tail * params.new_wo, LOAD3D_NUM_EACH_REPEAT);
      params.howo_co_ver_tail = UssCeilDiv(params.ho_tail * params.new_wo * C0, VCMP_NUM_EACH_REPEAT);
      if (params.config == 1) {
        params.num_instr_loop_h_tail = UssCeilDiv(params.ho_tail, MAX_REPEAT_TIME);
        params.remain_repeat_tail = params.ho_tail % MAX_REPEAT_TIME;
        if (params.remain_repeat_tail == 0 && (params.ho_tail != 0)) {
          params.remain_repeat_tail = MAX_REPEAT_TIME;
        }
        params.num_instr_loop_h_1_tail = params.num_instr_loop_h_tail - 1;
      }
      params.burst_len_ub_2_gm_tail = params.hi_val_tail * params.wi_batch * C0 * BYTE32 / BYTE_BLOCK;
      params.src_stride_ub_2_gm_tail = (params.hi_batch - params.hi_val_tail) * params.w * C0 * BYTE32 / BYTE_BLOCK;
      params.dst_stride_ub_2_gm_tail =
          (forward_in_shape.GetDim(INDEX_2) * forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) *
               (params.c1 - 1) +
           forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) * (params.h - params.hi_val_tail)) *
          BYTE32 / BYTE_BLOCK;
    }
  } else if (params.select_key == CASE_SAME_NO_TILING) {
    params.total_num_div_core = params.total_num % params.core_num;
    params.total_num_div_core_1 = params.total_num % core_num_ys;
    params.core_loop_params = params.total_num / params.core_num;
    params.core_loop_params1 = (params.total_num + params.core_num - 1) / params.core_num;
    InferDimReturn(params.new_ho, params.new_wo, bool_data, ksize, strides, params.ho, params.wo, params.h, params.w,
                   params.hi_batch, params.wi_batch);
    params.hi_tail = params.hi_batch;
    params.wi_tail = params.wi_batch;
    VectorDup(params_ub[INDEX_8], DTYPE_FP32, params.dup_repeat_merchant_f_map_fp32,
              params.dup_repeat_remainder_f_map_fp32, params.dup_remainder_f_map_fp32, params.repeats_f_map_fp32);
    InferMapReturn(params.new_ho, params.new_wo, ksize, strides, params.ho, params.wo, params.h, params.w, pad,
                   params.map_hi, params.map_wi);
    params.forward_in_shape_h_w_c0 = forward_in_shape.GetDim(INDEX_1) * forward_in_shape.GetDim(INDEX_2) *
                                     forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4);
    params.forward_ou_shape_h_w_c0 = forward_ou_shape.GetDim(INDEX_1) * forward_ou_shape.GetDim(INDEX_2) *
                                     forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4);
    params.burst_len = params.hi_batch * params.wi_batch * C0 * BYTE16 / BYTE_BLOCK;

    params.src_stride =
        (forward_in_shape.GetDim(INDEX_2) * forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) *
             (params.c1 - 1) +
         forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4) * (params.h - params.hi_batch)) *
        BYTE16 / BYTE_BLOCK;
    params.burst_len_src_orig_y = params.new_ho * params.new_wo * C0 * BYTE16 / BYTE_BLOCK;
    params.src_stride_src_orig_y =
        ((forward_ou_shape.GetDim(INDEX_2) * forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4)) *
             (params.c1 - 1) +
         forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4) * (params.ho - params.new_ho) +
         C0 * (params.wo - params.new_wo)) *
        BYTE16 / BYTE_BLOCK;
    params.repeat_times = UssCeilDiv(params.new_ho * params.new_wo, LOAD3D_NUM_EACH_REPEAT);
    params.pad_hw_left_neg = -params.pad_hw_left;
    params.pad_hw_top_neg = -params.pad_hw_top;
    params.howo_co_ver = UssCeilDiv(params.new_ho * params.new_wo * C0, VCMP_NUM_EACH_REPEAT);
    params.mask_size_16 = params_ub[INDEX_3] / FP16_BLOCK_NUM;
    params.mask_size_ver = params_ub[INDEX_3] / MASK_FP16;
    VectorDup2(params_ub[INDEX_6], params.repeat_max_time_grad_sel, params.remain_repeat_time_grad_sel,
               params.remain_ele_grad_sel);
    params.wo_2 = params.new_wo * TILING_FACTOR_TWO;
    params.sh_wi_2 = strides[1] * params.map_wi * TILING_FACTOR_TWO;
    vector<int64_t> config_list = {strides[INDEX_2] * TILING_FACTOR_TWO,
                                   strides[INDEX_2] * TILING_FACTOR_TWO,
                                   MODE_TWO,
                                   params.sh_wi_2,
                                   params.sh_wi_2,
                                   params.wo_2};
    params.config = CheckConfig(config_list);
    if (params.config == 1) {
      params.num_instr_loop_h = UssCeilDiv(params.new_ho, MAX_REPEAT_TIME);
      params.num_instr_loop_w = UssCeilDiv(params.new_wo * C0 / TILING_FACTOR_TWO, MASK_FP32);
      params.remain_mask = (params.new_wo * C0 / TILING_FACTOR_TWO) % MASK_FP32;
      if (params.remain_mask == 0 && ((params.new_wo * C0 / TILING_FACTOR_TWO) != 0)) {
        params.remain_mask = MASK_FP32;
      }

      params.remain_repeat = params.new_ho % MAX_REPEAT_TIME;
      if (params.remain_repeat == 0 && (params.new_ho != 0)) {
        params.remain_repeat = MAX_REPEAT_TIME;
      }

      params.num_instr_loop_w_1 = params.num_instr_loop_w - 1;
      params.num_instr_loop_h_1 = params.num_instr_loop_h - 1;
    }
    int64_t w_2 = params.new_wo * C0 / TILING_FACTOR_TWO;
    VectorDup3(w_2, DTYPE_FP32, params.repeat_max_loop_vadd, params.remain_max_loop_vadd, params.remain_ele_vadd);
    params.burst_len_over = params.wi_batch * C0 * BYTE32 / BYTE_BLOCK;
    params.src_stride_val = (params.pad_hw_left + params.pad_hw_right) * BYTE16;
  } else if (params.select_key == CASE_SAME_TILING_HO) {
    params.total_num_div_core = params.total_num % params.core_num;
    params.total_num_div_core_1 = params.total_num % core_num_ys;
    params.core_loop_params = params.total_num / params.core_num;
    params.core_loop_params1 = (params.total_num + params.core_num - 1) / params.core_num;
    params.loop_ho = params.core_ou_shape_h / params.new_ho;
    params.ho_tail = params.core_ou_shape_h % params.new_ho;
    InferDimReturn(params.new_ho, params.new_wo, bool_data, ksize, strides, params.ho, params.wo, params.h, params.w,
                   params.hi_batch, params.wi_batch);
    InferDimReturn(params.ho_tail, params.new_wo, bool_data, ksize, strides, params.ho, params.wo, params.h, params.w,
                   params.hi_tail, params.wi_tail);
    InferMapReturn(params.new_ho, params.new_wo, ksize, strides, params.ho, params.wo, params.h, params.w, pad,
                   params.map_hi, params.map_wi);
    VectorDup(params_ub[INDEX_8], DTYPE_FP32, params.dup_repeat_merchant_f_map_fp32,
              params.dup_repeat_remainder_f_map_fp32, params.dup_remainder_f_map_fp32, params.repeats_f_map_fp32);

    params.core_ho_times = params.ho / params.core_ou_shape_h;
    params.forward_in_shape_h_w_c0 = forward_in_shape.GetDim(INDEX_1) * forward_in_shape.GetDim(INDEX_2) *
                                     forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4);
    params.forward_ou_shape_h_w_c0 = forward_ou_shape.GetDim(INDEX_1) * forward_ou_shape.GetDim(INDEX_2) *
                                     forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4);
    if (params.overlap_h < 0) {
      params.hi_val = params.overlap_h + params.hi_batch;
    } else {
      params.hi_val = params.hi_batch;
    }
    params.forward_in_shape_h_w_2 = forward_in_shape.GetDim(INDEX_2) * forward_in_shape.GetDim(INDEX_3) *
                                    forward_in_shape.GetDim(INDEX_4) * (params.c1 - 1);
    params.forward_in_shape_w_c0 = forward_in_shape.GetDim(INDEX_3) * forward_in_shape.GetDim(INDEX_4);
    params.burst_len_src_orig_y = params.new_ho * params.new_wo * C0 * BYTE16 / BYTE_BLOCK;
    params.src_stride_src_orig_y =
        ((forward_ou_shape.GetDim(INDEX_2) * forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4)) *
             (params.c1 - 1) +
         forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4) * (params.ho - params.new_ho) +
         C0 * (params.wo - params.new_wo)) *
        BYTE16 / BYTE_BLOCK;
    params.repeat_times = UssCeilDiv(params.new_ho * params.new_wo, LOAD3D_NUM_EACH_REPEAT);
    params.pad_hw_left_neg = -params.pad_hw_left;
    params.pad_hw_top_neg = -params.pad_hw_top;
    params.howo_co_ver = UssCeilDiv(params.new_ho * params.new_wo * C0, VCMP_NUM_EACH_REPEAT);
    params.mask_size_16 = params_ub[INDEX_3] / FP16_BLOCK_NUM;
    params.mask_size_ver = params_ub[INDEX_3] / MASK_FP16;
    VectorDup2(params_ub[INDEX_6], params.repeat_max_time_grad_sel, params.remain_repeat_time_grad_sel,
               params.remain_ele_grad_sel);
    params.wo_2 = params.new_wo * TILING_FACTOR_TWO;
    params.sh_wi_2 = strides[1] * params.map_wi * TILING_FACTOR_TWO;
    vector<int64_t> config_list = {strides[INDEX_2] * TILING_FACTOR_TWO,
                                   strides[INDEX_2] * TILING_FACTOR_TWO,
                                   MODE_TWO,
                                   params.sh_wi_2,
                                   params.sh_wi_2,
                                   params.wo_2};
    params.config = CheckConfig(config_list);
    if (params.config == 1) {
      params.num_instr_loop_h = UssCeilDiv(params.new_ho, MAX_REPEAT_TIME);
      params.num_instr_loop_w = UssCeilDiv(params.new_wo * C0 / TILING_FACTOR_TWO, MASK_FP32);
      params.remain_mask = (params.new_wo * C0 / TILING_FACTOR_TWO) % MASK_FP32;
      if (params.remain_mask == 0 && ((params.new_wo * C0 / TILING_FACTOR_TWO) != 0)) {
        params.remain_mask = MASK_FP32;
      }

      params.remain_repeat = params.new_ho % MAX_REPEAT_TIME;
      if (params.remain_repeat == 0 && (params.new_ho != 0)) {
        params.remain_repeat = MAX_REPEAT_TIME;
      }
      params.num_instr_loop_w_1 = params.num_instr_loop_w - 1;
      params.num_instr_loop_h_1 = params.num_instr_loop_h - 1;
    }
    int64_t w_2 = params.new_wo * C0 / TILING_FACTOR_TWO;
    VectorDup3(w_2, DTYPE_FP32, params.repeat_max_loop_vadd, params.remain_max_loop_vadd, params.remain_ele_vadd);
    params.burst_len_ub_2_gm = params.wi_batch * C0 * BYTE32 / BYTE_BLOCK;
    params.src_stride_ub_2_gm = (params.pad_hw_left + params.pad_hw_right) * BYTE16;
    if (params.ho_tail != 0) {
      if (params.overlap_h > 0) {
        params.hi_val_tail = 0 + params.hi_tail;
      } else {
        params.hi_val_tail = params.overlap_h + params.hi_tail;
      }
      params.burst_len_src_orig_y_tail = params.ho_tail * params.new_wo * C0 * BYTE16 / BYTE_BLOCK;
      params.src_stride_src_orig_y_tail =
          ((forward_ou_shape.GetDim(INDEX_2) * forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4)) *
               (params.c1 - 1) +
           forward_ou_shape.GetDim(INDEX_3) * forward_ou_shape.GetDim(INDEX_4) * (params.ho - params.ho_tail) +
           C0 * (params.wo - params.new_wo)) *
          BYTE16 / BYTE_BLOCK;
      params.repeat_times_tail = UssCeilDiv(params.ho_tail * params.new_wo, LOAD3D_NUM_EACH_REPEAT);
      params.howo_co_ver_tail = UssCeilDiv(params.ho_tail * params.new_wo * C0, VCMP_NUM_EACH_REPEAT);
      if (params.config == 1) {
        params.num_instr_loop_h_tail = UssCeilDiv(params.ho_tail, MAX_REPEAT_TIME);
        params.remain_repeat_tail = params.ho_tail % MAX_REPEAT_TIME;
        if (params.remain_repeat_tail == 0 && (params.ho_tail != 0)) {
          params.remain_repeat_tail = MAX_REPEAT_TIME;
        }
        params.num_instr_loop_h_1_tail = params.num_instr_loop_h_tail - 1;
      }
    }
  } else {
    OP_LOGD(op_type.c_str(), "select_key is error");
    return false;
  }
  PROFILING_TILING_AFTER_CALCU_TILING_REG();
  // write tiling params to run_info
  MaxWriteTilingParams(params, run_info);
  // cout tiling params
  MaxPrintTilingParams(op_type, params);
  // BlockDim, core num used in tik op
  run_info.SetBlockDim(params.core_num);
  PROFILING_TILING_END();
  OP_LOGD(op_type.c_str(), "op tiling success");
  return true;
}
REGISTER_OP_TILING_V3_WITH_VECTOR(MaxPoolGrad, MaxPoolGradTiling, COMPILE_INFO_KEY, NO_OPTIONAL_VALUE);
}  // namespace optiling
