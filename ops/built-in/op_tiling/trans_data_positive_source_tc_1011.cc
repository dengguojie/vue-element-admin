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
 * \file trans_data_positive_source_tc_1011.cc
 * \brief dynamic TransData op tiling
 */
#include <string>
#include <algorithm>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "trans_data_common.h"
#include "error_log.h"

namespace optiling {

constexpr int64_t C0_16 = 16;
constexpr int64_t VNC_LINES = 16;

int64_t GetCeilFillC(int64_t u_value, int64_t d_value) {
  int64_t res_value = 0;
  if (d_value == 0) {
    return u_value;
  }

  res_value = (u_value + d_value - 1) / d_value * d_value;

  return res_value;
}

void GetMcInfoPositive1011(int64_t& axis_dst_r2nd_lp_cnt, int64_t axis_dst_r2nd_left, int64_t& c_lp_cnt, int64_t c_left,
                           int64_t& axis_src_cl_lp_cnt, int64_t axis_src_cl_left, int64_t& core_num,
                           TransDataMode1011Param& params) {
  int64_t tmp_full_loop_cnt_r2nd = GetFloorDiv(axis_dst_r2nd_lp_cnt, core_num) > 0 ? core_num : 0;
  int64_t reminder_loop_cnt_r2nd = axis_dst_r2nd_lp_cnt % core_num;
  if (reminder_loop_cnt_r2nd == 0) {
    tmp_full_loop_cnt_r2nd += core_num;
  }
  int64_t full_loop_cnt_r2nd = tmp_full_loop_cnt_r2nd + reminder_loop_cnt_r2nd;

  int64_t tmp_full_loop_cnt_c = GetFloorDiv(c_lp_cnt, core_num) > 0 ? core_num : 0;
  int64_t reminder_loop_cnt_c = c_lp_cnt % core_num;
  if (reminder_loop_cnt_c == 0) {
    tmp_full_loop_cnt_c += core_num;
  }
  int64_t full_loop_cnt_c = tmp_full_loop_cnt_c + reminder_loop_cnt_c;

  int64_t tmp_full_loop_cnt_left = GetFloorDiv(axis_src_cl_lp_cnt, core_num) > 0 ? core_num : 0;
  int64_t reminder_loop_cnt_left = axis_src_cl_lp_cnt % core_num;
  if (reminder_loop_cnt_left == 0) {
    tmp_full_loop_cnt_left += core_num;
  }
  int64_t full_loop_cnt_left = tmp_full_loop_cnt_left + reminder_loop_cnt_left;

  vector<int64_t> loop_cnt_list = {full_loop_cnt_r2nd, full_loop_cnt_left, full_loop_cnt_c};
  if (max_element(loop_cnt_list.begin(), loop_cnt_list.end()) - loop_cnt_list.begin() == 0) {
    params.mc_on_cl = 0;
    params.used_core_cnt = GetCeilDiv(axis_dst_r2nd_lp_cnt, GetCeilDiv(axis_dst_r2nd_lp_cnt, core_num));
    params.nlc_dst_r2nd_lp_cnt = GetCeilDiv(axis_dst_r2nd_lp_cnt, params.used_core_cnt);
    params.lc_dst_r2nd_lp_cnt = axis_dst_r2nd_lp_cnt - params.nlc_dst_r2nd_lp_cnt * (params.used_core_cnt - 1);
    params.core_step_in = params.nlc_dst_r2nd_lp_cnt * params.dst_r2nd_lp_step_in;
    params.core_step_out = params.nlc_dst_r2nd_lp_cnt * params.dst_r2nd_lp_step_out;
    params.nlc_dst_r2nd_left = 0;
    params.lc_dst_r2nd_left = axis_dst_r2nd_left;
    params.nlc_c_lp_cnt = c_lp_cnt;
    params.lc_c_lp_cnt = c_lp_cnt;
    params.nlc_c_left = c_left;
    params.lc_c_left = c_left;
    params.nlc_src_cl_lp_cnt = axis_src_cl_lp_cnt;
    params.lc_src_cl_lp_cnt = axis_src_cl_lp_cnt;
    params.nlc_src_cl_left = axis_src_cl_left;
    params.lc_src_cl_left = axis_src_cl_left;
  } else if (max_element(loop_cnt_list.begin(), loop_cnt_list.end()) - loop_cnt_list.begin() == 1) {
    params.mc_on_cl = 1;
    params.used_core_cnt = GetCeilDiv(axis_src_cl_lp_cnt, GetCeilDiv(axis_src_cl_lp_cnt, core_num));
    params.nlc_src_cl_lp_cnt = GetCeilDiv(axis_src_cl_lp_cnt, params.used_core_cnt);
    params.lc_src_cl_lp_cnt = axis_src_cl_lp_cnt - params.nlc_src_cl_lp_cnt * (params.used_core_cnt - 1);
    params.core_step_in = params.nlc_src_cl_lp_cnt * params.src_cl_lp_step_in;
    params.core_step_out = params.nlc_src_cl_lp_cnt * params.src_cl_lp_step_out;
    params.nlc_src_cl_left = 0;
    params.lc_src_cl_left = axis_src_cl_left;
    params.nlc_c_lp_cnt = c_lp_cnt;
    params.lc_c_lp_cnt = c_lp_cnt;
    params.nlc_c_left = c_left;
    params.lc_c_left = c_left;
    params.nlc_dst_r2nd_lp_cnt = axis_dst_r2nd_lp_cnt;
    params.lc_dst_r2nd_lp_cnt = axis_dst_r2nd_lp_cnt;
    params.nlc_dst_r2nd_left = axis_dst_r2nd_left;
    params.lc_dst_r2nd_left = axis_dst_r2nd_left;
  } else {
    params.mc_on_cl = 0;
    params.used_core_cnt = GetCeilDiv(c_lp_cnt, GetCeilDiv(c_lp_cnt, core_num));
    params.nlc_c_lp_cnt = GetCeilDiv(c_lp_cnt, params.used_core_cnt);
    params.lc_c_lp_cnt = c_lp_cnt - params.nlc_c_lp_cnt * (params.used_core_cnt - 1);
    params.core_step_in = params.nlc_c_lp_cnt * params.c_lp_step_in;
    params.core_step_out = params.nlc_c_lp_cnt * params.c_lp_step_out;
    params.nlc_c_left = 0;
    params.lc_c_left = c_left;
    params.nlc_src_cl_lp_cnt = axis_src_cl_lp_cnt;
    params.lc_src_cl_lp_cnt = axis_src_cl_lp_cnt;
    params.nlc_src_cl_left = axis_src_cl_left;
    params.lc_src_cl_left = axis_src_cl_left;
    params.nlc_dst_r2nd_lp_cnt = axis_dst_r2nd_lp_cnt;
    params.lc_dst_r2nd_lp_cnt = axis_dst_r2nd_lp_cnt;
    params.nlc_dst_r2nd_left = axis_dst_r2nd_left;
    params.lc_dst_r2nd_left = axis_dst_r2nd_left;
  }
}
void GetCommonParam(int64_t ub_size, int64_t block_elem_cnt, int64_t c0_len, int64_t axis_c_size,
                    TransDataMode1011Param& params) {
  if (block_elem_cnt == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "block_elem_cnt shoule not be 0");
    return;
  }

  int64_t half_ub_size;
  if (c0_len == C0_16) {
    half_ub_size = ub_size / 2;
  } else {
    half_ub_size = ub_size / 4;
  }
  params.vnc_line_size = half_ub_size / VNC_LINES / block_elem_cnt * block_elem_cnt;
  int64_t tmp_ub_offset = params.vnc_line_size * VNC_LINES;
  if (c0_len == C0_16) {
    params.ub_offset = tmp_ub_offset;
  } else {
    params.ub_offset = tmp_ub_offset * 2;
  }
  params.c_mod_c0 = axis_c_size % c0_len;
  params.c0_size = c0_len;
}

bool TillingPositiveMode1011(vector<int64_t>& in_shape, vector<int64_t>& out_shape, std::string& src_format,
                             std::string& dst_format, int64_t& core_num, int64_t& block_elem_cnt,
                             int64_t& ub_size, TransDataMode1011Param& params) {
  if ((src_format.length() != in_shape.size()) || (dst_format.length() != out_shape.size())) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "TillingPositiveMode1011 Failed.");
    return false;
  }
  int64_t axis_c_size = in_shape[in_shape.size() - 1];
  int64_t c0_len = out_shape[out_shape.size() - 1];
  GetCommonParam(ub_size, block_elem_cnt, c0_len, axis_c_size, params);

  params.tiling_mode = TILING_MODE_1011;

  // target axis -2 tiling parameters
  int32_t dst_axis_pos_c = std::strchr(dst_format.c_str(), 'C') - dst_format.c_str();
  int32_t src_axis_pos_c = std::strchr(src_format.c_str(), 'C') - src_format.c_str();
  int32_t dst_r2nd_in_src_idx = std::strchr(src_format.c_str(),
                                            dst_format[dst_format.length() - 2]) - src_format.c_str();
  int64_t axis_dst_r2nd_size = in_shape[dst_r2nd_in_src_idx];
  if (axis_dst_r2nd_size < VNC_LINES) {
    params.dst_r2nd_lp_unit = axis_dst_r2nd_size;
  } else {
    params.dst_r2nd_lp_unit = VNC_LINES;
  }
  int64_t axis_dst_r2nd_lp_cnt = GetCeilDiv(axis_dst_r2nd_size, params.dst_r2nd_lp_unit);
  int64_t axis_dst_r2nd_left = axis_dst_r2nd_size % params.dst_r2nd_lp_unit;
  params.dst_r2nd_lp_step_in = GetShapeSize(in_shape, dst_r2nd_in_src_idx + 1) * params.dst_r2nd_lp_unit;
  params.dst_r2nd_lp_step_out = GetShapeSize(out_shape, -1) * params.dst_r2nd_lp_unit;
  params.dst_r2nd_step_in = GetShapeSize(in_shape, dst_r2nd_in_src_idx + 1);

  // source axis c tiling parameters
  int64_t used_vnc_line_size = GetFloorDiv(params.vnc_line_size, params.c0_size) * params.c0_size;
  if (axis_c_size < used_vnc_line_size) {
    params.c_lp_unit = axis_c_size;
  } else {
    params.c_lp_unit = used_vnc_line_size;
  }
  params.c_lp_step_in = params.c_lp_unit;
  int64_t lp_c1_cnt = GetCeilDiv(params.c_lp_unit, c0_len);
  params.c_lp_step_out = lp_c1_cnt * GetShapeSize(out_shape, dst_axis_pos_c + 1);
  params.c_step_out = GetShapeSize(out_shape, dst_axis_pos_c + 1);
  int64_t c_lp_cnt = GetCeilDiv(axis_c_size, params.c_lp_unit);
  int64_t c_left = axis_c_size % params.c_lp_unit;

  // source axis left tiling parameters
  string src_format_left = src_format;
  src_format_left.replace(src_axis_pos_c, 1, "");
  int32_t chr_pos = std::strchr(src_format_left.c_str(), dst_format[dst_format.length() - 2]) - src_format_left.c_str();
  src_format_left.replace(chr_pos, 1, "");
  vector<int64_t> src_left_shape;
  for (size_t i = 0; i < src_format_left.length(); i++) {
    char cur_char = src_format_left[i];
    int32_t cur_pos = std::strchr(src_format.c_str(), cur_char) - src_format.c_str();
    src_left_shape.push_back(in_shape[cur_pos]);
  }
  src_left_shape.push_back(1);
  int64_t axis_src_cl_size = GetShapeSize(src_left_shape, 0);
  int64_t pln_src_cl_cnt = used_vnc_line_size / GetCeilFillC(params.c_lp_unit, c0_len);
  if (axis_src_cl_size < pln_src_cl_cnt) {
    params.src_cl_lp_unit = axis_src_cl_size;
  } else {
    params.src_cl_lp_unit = pln_src_cl_cnt;
  }
  int64_t axis_src_cl_lp_cnt = GetCeilDiv(axis_src_cl_size, params.src_cl_lp_unit);
  int64_t axis_src_cl_left = axis_src_cl_size % params.src_cl_lp_unit;
  params.src_cl_lp_step_in = GetShapeSize(in_shape, -1) * params.src_cl_lp_unit;
  params.src_cl_lp_step_out = 0;

  // parameters for output data
   reverse(src_format_left.begin(), src_format_left.end());
   for (size_t i = 0; i < src_format_left.length(); i++) {
    char chr = src_format_left[i];
    int32_t src_chr_pos = std::strchr(src_format.c_str(), chr) - src_format.c_str();
    int32_t dst_chr_pos = std::strchr(dst_format.c_str(), chr) - dst_format.c_str();
    if (i == 0) {
      params.cl_out_0_size = in_shape[src_chr_pos];
      params.cl_out_0_src_rsize = GetShapeSize(src_left_shape, -1 - i);
      params.cl_out_0_dst_asize = GetShapeSize(out_shape, dst_chr_pos + 1);
    } else if (i == 1) {
      params.cl_out_1_size = in_shape[src_chr_pos];
      params.cl_out_1_src_rsize = GetShapeSize(src_left_shape, -1 - i);
      params.cl_out_1_dst_asize = GetShapeSize(out_shape, dst_chr_pos + 1);
    }
  }

  GetMcInfoPositive1011(axis_dst_r2nd_lp_cnt, axis_dst_r2nd_left, c_lp_cnt, c_left, axis_src_cl_lp_cnt,
                        axis_src_cl_left, core_num, params);
  return true;
}

void SetRunningMode1011Params(const TransDataMode1011Param& run_params, utils::OpRunInfo& run_info) {
  run_info.AddTilingData(run_params.tiling_mode);
  run_info.AddTilingData(run_params.ub_offset);
  run_info.AddTilingData(run_params.used_core_cnt);
  run_info.AddTilingData(run_params.mc_on_cl);
  run_info.AddTilingData(run_params.core_step_in);
  run_info.AddTilingData(run_params.core_step_out);
  run_info.AddTilingData(run_params.dst_r2nd_lp_step_in);
  run_info.AddTilingData(run_params.dst_r2nd_lp_step_out);
  run_info.AddTilingData(run_params.dst_r2nd_step_in);
  run_info.AddTilingData(run_params.dst_r2nd_lp_unit);
  run_info.AddTilingData(run_params.src_cl_lp_step_in);
  run_info.AddTilingData(run_params.vnc_line_size);
  run_info.AddTilingData(run_params.src_cl_lp_unit);
  run_info.AddTilingData(run_params.src_cl_lp_step_out);
  run_info.AddTilingData(run_params.c_lp_step_in);
  run_info.AddTilingData(run_params.c_lp_step_out);
  run_info.AddTilingData(run_params.c_step_out);
  run_info.AddTilingData(run_params.c0_size);
  run_info.AddTilingData(run_params.c_mod_c0);
  run_info.AddTilingData(run_params.c_lp_unit);
  run_info.AddTilingData(run_params.nlc_dst_r2nd_lp_cnt);
  run_info.AddTilingData(run_params.nlc_dst_r2nd_left);
  run_info.AddTilingData(run_params.nlc_src_cl_lp_cnt);
  run_info.AddTilingData(run_params.nlc_src_cl_left);
  run_info.AddTilingData(run_params.nlc_c_lp_cnt);
  run_info.AddTilingData(run_params.nlc_c_left);
  run_info.AddTilingData(run_params.lc_dst_r2nd_lp_cnt);
  run_info.AddTilingData(run_params.lc_dst_r2nd_left);
  run_info.AddTilingData(run_params.lc_src_cl_lp_cnt);
  run_info.AddTilingData(run_params.lc_src_cl_left);
  run_info.AddTilingData(run_params.lc_c_lp_cnt);
  run_info.AddTilingData(run_params.lc_c_left);
  run_info.AddTilingData(run_params.cl_out_0_size);
  run_info.AddTilingData(run_params.cl_out_0_src_rsize);
  run_info.AddTilingData(run_params.cl_out_0_dst_asize);
  run_info.AddTilingData(run_params.cl_out_1_size);
  run_info.AddTilingData(run_params.cl_out_1_src_rsize);
  run_info.AddTilingData(run_params.cl_out_1_dst_asize);
}

void PrintTilingMode1011Params(const std::string& op_type, const TransDataMode1011Param& params) {
  OP_LOGD(op_type, "tiling_mode=%d", params.tiling_mode);
  OP_LOGD(op_type, "ub_offset=%d", params.ub_offset);
  OP_LOGD(op_type, "used_core_cnt=%d", params.used_core_cnt);
  OP_LOGD(op_type, "mc_on_cl=%d", params.mc_on_cl);
  OP_LOGD(op_type, "core_step_in=%d", params.core_step_in);
  OP_LOGD(op_type, "core_step_out=%d", params.core_step_out);
  OP_LOGD(op_type, "dst_r2nd_lp_step_in=%d", params.dst_r2nd_lp_step_in);
  OP_LOGD(op_type, "dst_r2nd_lp_step_out=%d", params.dst_r2nd_lp_step_out);
  OP_LOGD(op_type, "dst_r2nd_step_in=%d", params.dst_r2nd_step_in);
  OP_LOGD(op_type, "dst_r2nd_lp_unit=%d", params.dst_r2nd_lp_unit);
  OP_LOGD(op_type, "src_cl_lp_step_in=%d", params.src_cl_lp_step_in);
  OP_LOGD(op_type, "vnc_line_size=%d", params.vnc_line_size);
  OP_LOGD(op_type, "src_cl_lp_unit=%d", params.src_cl_lp_unit);
  OP_LOGD(op_type, "src_cl_lp_step_out=%d", params.src_cl_lp_step_out);
  OP_LOGD(op_type, "c_lp_step_in=%d", params.c_lp_step_in);
  OP_LOGD(op_type, "c_lp_step_out=%d", params.c_lp_step_out);
  OP_LOGD(op_type, "c_step_out=%d", params.c_step_out);
  OP_LOGD(op_type, "c0_size=%d", params.c0_size);
  OP_LOGD(op_type, "c_mod_c0=%d", params.c_mod_c0);
  OP_LOGD(op_type, "c_lp_unit=%d", params.c_lp_unit);
  OP_LOGD(op_type, "nlc_dst_r2nd_lp_cnt=%d", params.nlc_dst_r2nd_lp_cnt);
  OP_LOGD(op_type, "nlc_dst_r2nd_left=%d", params.nlc_dst_r2nd_left);
  OP_LOGD(op_type, "nlc_src_cl_lp_cnt=%d", params.nlc_src_cl_lp_cnt);
  OP_LOGD(op_type, "nlc_src_cl_left=%d", params.nlc_src_cl_left);
  OP_LOGD(op_type, "nlc_c_lp_cnt=%d", params.nlc_c_lp_cnt);
  OP_LOGD(op_type, "nlc_c_left=%d", params.nlc_c_left);
  OP_LOGD(op_type, "lc_dst_r2nd_lp_cnt=%d", params.lc_dst_r2nd_lp_cnt);
  OP_LOGD(op_type, "lc_dst_r2nd_left=%d", params.lc_dst_r2nd_left);
  OP_LOGD(op_type, "lc_src_cl_lp_cnt=%d", params.lc_src_cl_lp_cnt);
  OP_LOGD(op_type, "lc_src_cl_left=%d", params.lc_src_cl_left);
  OP_LOGD(op_type, "lc_c_lp_cnt=%d", params.lc_c_lp_cnt);
  OP_LOGD(op_type, "lc_c_left=%d", params.lc_c_left);
  OP_LOGD(op_type, "cl_out_0_size=%d", params.cl_out_0_size);
  OP_LOGD(op_type, "cl_out_0_src_rsize=%d", params.cl_out_0_src_rsize);
  OP_LOGD(op_type, "cl_out_0_dst_asize=%d", params.cl_out_0_dst_asize);
  OP_LOGD(op_type, "cl_out_1_size=%d", params.cl_out_1_size);
  OP_LOGD(op_type, "cl_out_1_src_rsize=%d", params.cl_out_1_src_rsize);
  OP_LOGD(op_type, "cl_out_1_dst_asize=%d", params.cl_out_1_dst_asize);
}
}  // namespace optiling
