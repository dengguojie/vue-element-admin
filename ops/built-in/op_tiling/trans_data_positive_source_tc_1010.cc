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
 * \file trans_data_positive_source_tc_1010.cc
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

int64_t GetCeilFillB(int64_t u_value, int64_t d_value) {
  int64_t res_value = 0;
  if (d_value == 0) {
    return u_value;
  }

  res_value = (u_value + d_value - 1) / d_value * d_value;

  return res_value;
}

void GetMcInfoPositive1010(int64_t& dst_cl_lp_cnt, int64_t& vnc_row_cl_left, int64_t& ll_dst_cl_left, int64_t& c_lp_cnt,
                           int64_t c_left, int64_t& dst_cr_lp_cnt, int64_t vnc_row_left, int64_t ll_dst_cr_left,
                           int64_t& core_num, TransDataMode1010Param& params) {
  int64_t tmp_full_loop_cnt_cr = GetFloorDiv(dst_cr_lp_cnt, core_num) > 0 ? core_num : 0;
  
  int64_t reminder_loop_cnt_cr = dst_cr_lp_cnt % core_num;
  if (reminder_loop_cnt_cr == 0) {
    tmp_full_loop_cnt_cr += core_num;
  }
  int64_t full_loop_cnt_cr = tmp_full_loop_cnt_cr + reminder_loop_cnt_cr;

  int64_t tmp_full_loop_cnt_c = GetFloorDiv(c_lp_cnt, core_num) > 0 ? core_num : 0;

  int64_t reminder_loop_cnt_c = c_lp_cnt % core_num;
  if (reminder_loop_cnt_c == 0) {
    tmp_full_loop_cnt_c += core_num;
  }
  int64_t full_loop_cnt_c = tmp_full_loop_cnt_c + reminder_loop_cnt_c;

  int64_t tmp_full_loop_cnt_left = GetFloorDiv(dst_cl_lp_cnt, core_num) > 0 ? core_num : 0;
  int64_t reminder_loop_cnt_left = dst_cl_lp_cnt % core_num;
  if (reminder_loop_cnt_left == 0) {
    tmp_full_loop_cnt_left += core_num;
  }
  int64_t full_loop_cnt_left = tmp_full_loop_cnt_left + reminder_loop_cnt_left;

  vector<int64_t> loop_cnt_list = {full_loop_cnt_left, full_loop_cnt_cr, full_loop_cnt_c};
  if (max_element(loop_cnt_list.begin(), loop_cnt_list.end()) - loop_cnt_list.begin() == 0) {
    params.used_core_cnt = GetCeilDiv(dst_cl_lp_cnt, GetCeilDiv(dst_cl_lp_cnt, core_num));
    params.nlc_dst_cl_lp_cnt = GetCeilDiv(dst_cl_lp_cnt, params.used_core_cnt);
    params.lc_dst_cl_lp_cnt = dst_cl_lp_cnt - params.nlc_dst_cl_lp_cnt * (params.used_core_cnt - 1);
    params.core_step_in = params.nlc_dst_cl_lp_cnt * params.dst_cl_lp_step_in;
    params.core_step_out = params.nlc_dst_cl_lp_cnt * params.dst_cl_lp_step_out;
    params.nlc_vnc_row_cl_left = 0;
    params.lc_vnc_row_cl_left = vnc_row_cl_left;
    params.nlc_last_line_cl_cnt = ll_dst_cl_left;
    params.lc_last_line_cl_cnt = ll_dst_cl_left;
    params.nlc_c_lp_cnt = c_lp_cnt;
    params.lc_c_lp_cnt = c_lp_cnt;
    params.nlc_c_left = c_left;
    params.lc_c_left = c_left;
    params.nlc_dst_cr_lp_cnt = dst_cr_lp_cnt;
    params.lc_dst_cr_lp_cnt = dst_cr_lp_cnt;
    params.nlc_vnc_row_left = vnc_row_left;
    params.lc_vnc_row_left = vnc_row_left;
    params.nlc_last_line_cr_cnt = ll_dst_cr_left;
    params.lc_last_line_cr_cnt = ll_dst_cr_left;
  } else if (max_element(loop_cnt_list.begin(), loop_cnt_list.end()) - loop_cnt_list.begin() == 1) {
    params.used_core_cnt = GetCeilDiv(dst_cr_lp_cnt, GetCeilDiv(dst_cr_lp_cnt, core_num));
    params.nlc_dst_cr_lp_cnt = GetCeilDiv(dst_cr_lp_cnt, params.used_core_cnt);
    params.lc_dst_cr_lp_cnt = dst_cr_lp_cnt - params.nlc_dst_cr_lp_cnt * (params.used_core_cnt - 1);
    params.core_step_in = params.nlc_dst_cr_lp_cnt * params.dst_cr_lp_step_in;
    params.core_step_out = params.nlc_dst_cr_lp_cnt * params.dst_cr_lp_step_out;
    params.nlc_vnc_row_left = 0;
    params.lc_vnc_row_left = vnc_row_left;
    params.nlc_last_line_cr_cnt = params.pln_dst_cr_size;
    params.lc_last_line_cr_cnt = ll_dst_cr_left;
    params.nlc_c_lp_cnt = c_lp_cnt;
    params.lc_c_lp_cnt = c_lp_cnt;
    params.nlc_c_left = c_left;
    params.lc_c_left = c_left;
    params.nlc_dst_cl_lp_cnt = dst_cl_lp_cnt;
    params.lc_dst_cl_lp_cnt = dst_cl_lp_cnt;
    params.nlc_vnc_row_cl_left = vnc_row_cl_left;
    params.lc_vnc_row_cl_left = vnc_row_cl_left;
    params.nlc_last_line_cl_cnt = ll_dst_cl_left;
    params.lc_last_line_cl_cnt = ll_dst_cl_left;
  } else {
    params.used_core_cnt = GetCeilDiv(c_lp_cnt, GetCeilDiv(c_lp_cnt, core_num));
    params.nlc_c_lp_cnt = GetCeilDiv(c_lp_cnt, params.used_core_cnt);
    params.lc_c_lp_cnt = c_lp_cnt - params.nlc_c_lp_cnt * (params.used_core_cnt - 1);
    params.core_step_in = params.nlc_c_lp_cnt * params.c_lp_step_in;
    params.core_step_out = params.nlc_c_lp_cnt * params.c_lp_step_out;
    params.nlc_c_left = 0;
    params.lc_c_left = c_left;
    params.nlc_dst_cl_lp_cnt = dst_cl_lp_cnt;
    params.lc_dst_cl_lp_cnt = dst_cl_lp_cnt;
    params.nlc_vnc_row_cl_left = vnc_row_cl_left;
    params.lc_vnc_row_cl_left = vnc_row_cl_left;
    params.nlc_last_line_cl_cnt = ll_dst_cl_left;
    params.lc_last_line_cl_cnt = ll_dst_cl_left;
    params.nlc_dst_cr_lp_cnt = dst_cr_lp_cnt;
    params.lc_dst_cr_lp_cnt = dst_cr_lp_cnt;
    params.nlc_vnc_row_left = vnc_row_left;
    params.lc_vnc_row_left = vnc_row_left;
    params.nlc_last_line_cr_cnt = ll_dst_cr_left;
    params.lc_last_line_cr_cnt = ll_dst_cr_left;
  }
}

void GetCommonParam(int64_t ub_size, int64_t block_elem_cnt, int64_t c0_len, int64_t axis_c_size,
                    TransDataMode1010Param& params) {
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

bool TillingPositiveMode1010(vector<int64_t>& in_shape, vector<int64_t>& out_shape, std::string& src_format,
                             std::string& dst_format, int64_t& core_num, int64_t& block_elem_cnt,
                             int64_t& ub_size, TransDataMode1010Param& params) {
  if ((src_format.length() != in_shape.size()) || (dst_format.length() != out_shape.size())) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "TillingPositiveMode1010 Failed.");
    return false;
  }
  int64_t axis_c_size = in_shape[in_shape.size() - 1];
  int64_t c0_len = out_shape[out_shape.size() - 1];
  GetCommonParam(ub_size, block_elem_cnt, c0_len, axis_c_size, params);

  params.tiling_mode = 1010;
  params.vnc_line_size = params.vnc_line_size / c0_len * c0_len;

  // source axis c tiling parameters
  int32_t dst_axis_pos_c = std::strchr(dst_format.c_str(), 'C') - dst_format.c_str();
  if (axis_c_size < params.vnc_line_size) {
    params.c_lp_unit = axis_c_size;
  } else {
    params.c_lp_unit = params.vnc_line_size;
  }
  params.c_lp_step_in = params.c_lp_unit;
  int64_t lp_c1_cnt = GetCeilDiv(params.c_lp_unit, c0_len);
  params.c_lp_step_out = lp_c1_cnt * GetShapeSize(out_shape, dst_axis_pos_c + 1);
  params.c_step_out = GetShapeSize(out_shape, dst_axis_pos_c + 1);
  int64_t c_lp_cnt = GetCeilDiv(axis_c_size, params.c_lp_unit);
  int64_t c_left = axis_c_size % params.c_lp_unit;

  // target axis c-right tiling parameters
  int64_t axis_dst_cl_size = 1;
  for (int32_t i = 0; i < dst_axis_pos_c; i++) {
    axis_dst_cl_size *= out_shape[i];
  }
  int32_t tmp_src_pos = std::strchr(src_format.c_str(), dst_format[dst_format.length() - 2]) - src_format.c_str();
  int64_t axis_dst_cr_size = GetShapeSize(in_shape, tmp_src_pos) / in_shape[in_shape.size() - 1];
  params.pln_dst_cr_size = params.vnc_line_size / GetCeilFillB(params.c_lp_unit, c0_len);
  params.vnc_row_size = VNC_LINES;
  int64_t per_vnc_dst_cr_cnt = params.pln_dst_cr_size * params.vnc_row_size;
  if (per_vnc_dst_cr_cnt >= axis_dst_cr_size && core_num > 1 && axis_dst_cl_size == 1) {
    int64_t new_vnc_lines = GetCeilDiv(axis_dst_cr_size, params.pln_dst_cr_size);
    if (new_vnc_lines > VNC_LINES) {
      new_vnc_lines = VNC_LINES;
    }
    int64_t vnc_per_core = new_vnc_lines > core_num ? GetCeilDiv(new_vnc_lines, core_num) : 1;
    params.vnc_row_size = vnc_per_core;
    per_vnc_dst_cr_cnt = params.pln_dst_cr_size * params.vnc_row_size;
  }
  int64_t dst_cr_lp_cnt = GetCeilDiv(axis_dst_cr_size, per_vnc_dst_cr_cnt);
  int64_t dst_cr_left = axis_dst_cr_size % per_vnc_dst_cr_cnt;
  int64_t vnc_row_left = GetCeilDiv(dst_cr_left, params.pln_dst_cr_size);
  int64_t tmp_dst_cr_left = dst_cr_left % params.pln_dst_cr_size;
  int64_t ll_dst_cr_left;
  if (tmp_dst_cr_left > 0) {
    ll_dst_cr_left = tmp_dst_cr_left;
  } else {
    ll_dst_cr_left = params.pln_dst_cr_size;
  }

  params.dst_cr_lp_step_in = in_shape[in_shape.size() - 1] * per_vnc_dst_cr_cnt;
  int32_t tmp_dst_pos = std::strchr(dst_format.c_str(), src_format[src_format.length() - 2]) - dst_format.c_str();
  params.dst_cr_lp_step_out = GetShapeSize(out_shape, tmp_dst_pos + 1) * per_vnc_dst_cr_cnt;
  params.dst_cr_step_in = GetShapeSize(in_shape, -1);

  // target axis c-left tiling parameters
  int64_t per_vnc_dst_cl_cnt = 1;
  int64_t dst_cl_lp_cnt = 1;
  int64_t dst_cl_left = 0;
  int64_t vnc_row_cl_left = 0;
  int64_t tmp_dst_cl_left = 0;
  int64_t ll_dst_cl_left = 0;
  char dst_cl_char = dst_format[dst_axis_pos_c - 1];

  if ((axis_c_size % c0_len == 0 && GetCeilDiv(params.c_lp_unit, block_elem_cnt) % C0_16 != 0) ||
      (axis_c_size % c0_len == 0 && params.pln_dst_cr_size % 2 == 0)) {
    // move in cl_cr_c in together
    if (params.c_lp_unit == axis_c_size && per_vnc_dst_cr_cnt >= axis_dst_cr_size) {
      params.nc_le_vcol = 3;
      per_vnc_dst_cl_cnt = GetFloorDiv(params.vnc_line_size * VNC_LINES, axis_c_size * axis_dst_cr_size);
    } else if (params.c_lp_unit == axis_c_size) {
      // move in cr_c in together
      params.nc_le_vcol = 4;
      per_vnc_dst_cl_cnt = 1;
    } else {
      // move in c
      params.nc_le_vcol = 5;
      per_vnc_dst_cl_cnt = 1;
    }
    params.pln_dst_cl_size = per_vnc_dst_cl_cnt;
    dst_cl_lp_cnt = GetCeilDiv(axis_dst_cl_size, params.pln_dst_cl_size);
    vnc_row_cl_left = axis_dst_cl_size % params.pln_dst_cl_size;
    ll_dst_cl_left = axis_dst_cl_size % params.pln_dst_cl_size;
  } else if (dst_cr_lp_cnt == 1 && params.c_lp_unit == axis_c_size && vnc_row_left <= GetFloorDiv(VNC_LINES, 2)) {
    // nc is less than vnchwconv col size
    if (vnc_row_left == 1) {
      params.nc_le_vcol = 1;
      params.pln_dst_cl_size = GetFloorDiv(params.pln_dst_cr_size, axis_dst_cr_size);
    } else {
      params.nc_le_vcol = 2;
      params.pln_dst_cl_size = 1;
      // adjust c-right parameters
      dst_cr_lp_cnt = GetCeilDiv(axis_dst_cr_size, params.pln_dst_cr_size);
      vnc_row_left = axis_dst_cr_size % params.pln_dst_cr_size;
      if (vnc_row_left > 0) {
        ll_dst_cr_left = vnc_row_left;
      } else {
        ll_dst_cr_left = params.pln_dst_cr_size;
      }
      params.dst_cr_lp_step_in = in_shape[in_shape.size() - 1] * params.pln_dst_cr_size;
      params.dst_cr_lp_step_out = GetShapeSize(out_shape, tmp_dst_pos + 1) * params.pln_dst_cr_size;
    }

    per_vnc_dst_cl_cnt = params.pln_dst_cl_size * params.vnc_row_size;
    dst_cl_lp_cnt = GetCeilDiv(axis_dst_cl_size, per_vnc_dst_cl_cnt);
    // adjust c-left parameters
    int64_t four_in_core_cnt = 4;
    int64_t pln_cl_gate = 64;
    if ((dst_cl_lp_cnt < GetFloorDiv(core_num, four_in_core_cnt)) && (params.pln_dst_cl_size > pln_cl_gate)) {
      params.pln_dst_cl_size = GetFloorDiv(params.pln_dst_cl_size, pln_cl_gate);
      per_vnc_dst_cl_cnt = params.pln_dst_cl_size * params.vnc_row_size;
      dst_cl_lp_cnt = GetCeilDiv(axis_dst_cl_size, per_vnc_dst_cl_cnt);
    }
    dst_cl_left = axis_dst_cl_size % per_vnc_dst_cl_cnt;
    vnc_row_cl_left = GetCeilDiv(dst_cl_left, params.pln_dst_cl_size);
    tmp_dst_cl_left = dst_cl_left % params.pln_dst_cl_size;
    if (tmp_dst_cl_left > 0) {
      ll_dst_cl_left = tmp_dst_cl_left;
    } else {
      ll_dst_cl_left = params.pln_dst_cl_size;
    }

  } else {
    params.nc_le_vcol = 0;
    params.pln_dst_cl_size = 1;
    dst_cl_lp_cnt = axis_dst_cl_size;
    vnc_row_cl_left = params.pln_dst_cl_size;
    ll_dst_cl_left = params.pln_dst_cl_size;
  }
  params.dst_cl_step_in = GetShapeSize(in_shape, std::strchr(src_format.c_str(), dst_cl_char) - src_format.c_str() + 1);
  params.dst_cl_step_out = GetShapeSize(out_shape, dst_axis_pos_c);
  if (params.nc_le_vcol == 0) {
    params.dst_cl_lp_step_in = params.dst_cl_step_in;
    params.dst_cl_lp_step_out = params.dst_cl_step_out;
  } else {
    params.dst_cl_lp_step_in = params.dst_cl_step_in * per_vnc_dst_cl_cnt;
    params.dst_cl_lp_step_out = params.dst_cl_step_out * per_vnc_dst_cl_cnt;
  }

  GetMcInfoPositive1010(dst_cl_lp_cnt, vnc_row_cl_left, ll_dst_cl_left, c_lp_cnt, c_left,
                        dst_cr_lp_cnt, vnc_row_left, ll_dst_cr_left, core_num, params);
  return true;
}

void SetRunningMode1010Params(const TransDataMode1010Param& run_params, utils::OpRunInfo& run_info) {
  run_info.AddTilingData(run_params.tiling_mode);
  run_info.AddTilingData(run_params.ub_offset);
  run_info.AddTilingData(run_params.used_core_cnt);
  run_info.AddTilingData(run_params.core_step_in);
  run_info.AddTilingData(run_params.core_step_out);

  run_info.AddTilingData(run_params.dst_cl_lp_step_in);
  run_info.AddTilingData(run_params.dst_cl_lp_step_out);
  run_info.AddTilingData(run_params.dst_cl_step_in);
  run_info.AddTilingData(run_params.dst_cl_step_out);
  run_info.AddTilingData(run_params.dst_cr_lp_step_in);
  run_info.AddTilingData(run_params.dst_cr_lp_step_out);
  run_info.AddTilingData(run_params.dst_cr_step_in);
  run_info.AddTilingData(run_params.nc_le_vcol);
  run_info.AddTilingData(run_params.vnc_line_size);

  run_info.AddTilingData(run_params.pln_dst_cl_size);
  run_info.AddTilingData(run_params.pln_dst_cr_size);
  run_info.AddTilingData(run_params.vnc_row_size);
  run_info.AddTilingData(run_params.c_lp_step_in);
  run_info.AddTilingData(run_params.c_lp_step_out);
  run_info.AddTilingData(run_params.c_step_out);
  run_info.AddTilingData(run_params.c0_size);
  run_info.AddTilingData(run_params.c_mod_c0);
  run_info.AddTilingData(run_params.c_lp_unit);
  run_info.AddTilingData(run_params.nlc_dst_cl_lp_cnt);
  run_info.AddTilingData(run_params.nlc_vnc_row_cl_left);
  run_info.AddTilingData(run_params.nlc_last_line_cl_cnt);
  run_info.AddTilingData(run_params.nlc_dst_cr_lp_cnt);
  run_info.AddTilingData(run_params.nlc_vnc_row_left);
  run_info.AddTilingData(run_params.nlc_last_line_cr_cnt);
  run_info.AddTilingData(run_params.nlc_c_lp_cnt);
  run_info.AddTilingData(run_params.nlc_c_left);
  run_info.AddTilingData(run_params.lc_dst_cl_lp_cnt);
  run_info.AddTilingData(run_params.lc_vnc_row_cl_left);
  run_info.AddTilingData(run_params.lc_last_line_cl_cnt);
  run_info.AddTilingData(run_params.lc_dst_cr_lp_cnt);
  run_info.AddTilingData(run_params.lc_vnc_row_left);
  run_info.AddTilingData(run_params.lc_last_line_cr_cnt);
  run_info.AddTilingData(run_params.lc_c_lp_cnt);
  run_info.AddTilingData(run_params.lc_c_left);
}

void PrintTilingMode1010Params(const std::string& op_type, const TransDataMode1010Param& params) {
  OP_LOGD(op_type, "tiling_mode=%d", params.tiling_mode);
  OP_LOGD(op_type, "ub_offset=%d", params.ub_offset);
  OP_LOGD(op_type, "used_core_cnt=%d", params.used_core_cnt);
  OP_LOGD(op_type, "core_step_in=%d", params.core_step_in);
  OP_LOGD(op_type, "core_step_out=%d", params.core_step_out);

  OP_LOGD(op_type, "dst_cl_lp_step_in=%d", params.dst_cl_lp_step_in);
  OP_LOGD(op_type, "dst_cl_lp_step_out=%d", params.dst_cl_lp_step_out);
  OP_LOGD(op_type, "dst_cl_step_in=%d", params.dst_cl_step_in);
  OP_LOGD(op_type, "dst_cl_step_out=%d", params.dst_cl_step_out);
  OP_LOGD(op_type, "dst_cr_lp_step_in=%d", params.dst_cr_lp_step_in);
  OP_LOGD(op_type, "dst_cr_lp_step_out=%d", params.dst_cr_lp_step_out);
  OP_LOGD(op_type, "dst_cr_step_in=%d", params.dst_cr_step_in);
  OP_LOGD(op_type, "nc_le_vcol=%d", params.nc_le_vcol);
  OP_LOGD(op_type, "vnc_line_size=%d", params.vnc_line_size);

  OP_LOGD(op_type, "pln_dst_cl_size=%d", params.pln_dst_cl_size);
  OP_LOGD(op_type, "pln_dst_cr_size=%d", params.pln_dst_cr_size);
  OP_LOGD(op_type, "vnc_row_size=%d", params.vnc_row_size);
  OP_LOGD(op_type, "c_lp_step_in=%d", params.c_lp_step_in);
  OP_LOGD(op_type, "c_lp_step_out=%d", params.c_lp_step_out);
  OP_LOGD(op_type, "c_step_out=%d", params.c_step_out);
  OP_LOGD(op_type, "c0_size=%d", params.c0_size);
  OP_LOGD(op_type, "c_mod_c0=%d", params.c_mod_c0);
  OP_LOGD(op_type, "c_lp_unit=%d", params.c_lp_unit);

  OP_LOGD(op_type, "nlc_dst_cl_lp_cnt=%d", params.nlc_dst_cl_lp_cnt);
  OP_LOGD(op_type, "nlc_vnc_row_cl_left=%d", params.nlc_vnc_row_cl_left);
  OP_LOGD(op_type, "nlc_last_line_cl_cnt=%d", params.nlc_last_line_cl_cnt);
  OP_LOGD(op_type, "nlc_dst_cr_lp_cnt=%d", params.nlc_dst_cr_lp_cnt);
  OP_LOGD(op_type, "nlc_vnc_row_left=%d", params.nlc_vnc_row_left);
  OP_LOGD(op_type, "nlc_last_line_cr_cnt=%d", params.nlc_last_line_cr_cnt);
  OP_LOGD(op_type, "nlc_c_lp_cnt=%d", params.nlc_c_lp_cnt);
  OP_LOGD(op_type, "nlc_c_left=%d", params.nlc_c_left);

  OP_LOGD(op_type, "lc_dst_cl_lp_cnt=%d", params.lc_dst_cl_lp_cnt);
  OP_LOGD(op_type, "lc_vnc_row_cl_left=%d", params.lc_vnc_row_cl_left);
  OP_LOGD(op_type, "lc_last_line_cl_cnt=%d", params.lc_last_line_cl_cnt);
  OP_LOGD(op_type, "lc_dst_cr_lp_cnt=%d", params.lc_dst_cr_lp_cnt);
  OP_LOGD(op_type, "lc_vnc_row_left=%d", params.lc_vnc_row_left);
  OP_LOGD(op_type, "lc_last_line_cr_cnt=%d", params.lc_last_line_cr_cnt);
  OP_LOGD(op_type, "lc_c_lp_cnt=%d", params.lc_c_lp_cnt);
  OP_LOGD(op_type, "lc_c_left=%d", params.lc_c_left);
}

}  // namespace optiling
