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
 * \file trans_data_negative_target_ntc.cc
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
const int32_t NTC_FRAME_LEVEL = 2;

int64_t GetCeilFillA(int64_t u_value, int64_t d_value) {
  int64_t res_value = 0;
  if (d_value == 0) {
    return u_value;
  }

  res_value = (u_value + d_value - 1) / d_value * d_value;

  return res_value;
}

bool GetMcInfoNegative200(int64_t& dst_cr_lp_cnt, int64_t dst_cr_left, int64_t& src_c_lp_cnt, int64_t src_c_left,
                          int64_t& dst_cl_lp_cnt, int64_t dst_cl_left, int64_t& core_num,
                          TransDataNtc200Param& params) {
  int64_t tmp_full_loop_cnt_cr;
  if (GetFloorDiv(dst_cr_lp_cnt, core_num) > 0) {
    tmp_full_loop_cnt_cr = core_num;
  } else {
    tmp_full_loop_cnt_cr = 0;
  }
  int64_t reminder_loop_cnt_cr = dst_cr_lp_cnt % core_num;
  if (reminder_loop_cnt_cr == 0 && dst_cr_left > params.dst_cr_lp_unit / 2) {
    tmp_full_loop_cnt_cr += core_num;
  }
  int64_t full_loop_cnt_cr = tmp_full_loop_cnt_cr + reminder_loop_cnt_cr;

  int64_t tmp_full_loop_cnt_c;
  if (GetFloorDiv(src_c_lp_cnt, core_num) > 0) {
    tmp_full_loop_cnt_c = core_num;
  } else {
    tmp_full_loop_cnt_c = 0;
  }
  int64_t reminder_loop_cnt_c = src_c_lp_cnt % core_num;
  if (reminder_loop_cnt_c == 0) {
    tmp_full_loop_cnt_c += core_num;
  }
  int64_t full_loop_cnt_c = tmp_full_loop_cnt_c + reminder_loop_cnt_c;

  int64_t tmp_full_loop_cnt_cl;
  if (GetFloorDiv(dst_cl_lp_cnt, core_num) > 0) {
    tmp_full_loop_cnt_cl = core_num;
  } else {
    tmp_full_loop_cnt_cl = 0;
  }
  int64_t reminder_loop_cnt_cl = dst_cl_lp_cnt % core_num;
  if (reminder_loop_cnt_cl == 0) {
    tmp_full_loop_cnt_cl += core_num;
  }
  int64_t full_loop_cnt_cl = tmp_full_loop_cnt_cl + reminder_loop_cnt_cl;
  vector<int64_t> loop_cnt_list = {full_loop_cnt_cl, full_loop_cnt_c, full_loop_cnt_cr};

  if (max_element(loop_cnt_list.begin(), loop_cnt_list.end()) - loop_cnt_list.begin() == 0) {
    params.mc_pos = 0;
    params.is_mc_cl = 1;
    params.is_mc_cr = 0;
    params.used_core_cnt = GetCeilDiv(dst_cl_lp_cnt, GetCeilDiv(dst_cl_lp_cnt, core_num));
    params.nlc_cl_lp_cnt = GetCeilDiv(dst_cl_lp_cnt, params.used_core_cnt);
    params.lc_cl_lp_cnt = dst_cl_lp_cnt - params.nlc_cl_lp_cnt * (params.used_core_cnt - 1);
    params.core_step_in = params.nlc_cl_lp_cnt * params.dst_cl_lp_step_in;
    params.core_step_out = params.nlc_cl_lp_cnt * params.dst_cl_lp_step_out;
    params.nlc_cl_left = 0;
    params.lc_cl_left = dst_cl_left;
    params.nlc_c_lp_cnt = src_c_lp_cnt;
    params.lc_c_lp_cnt = src_c_lp_cnt;
    params.nlc_c_left = src_c_left;
    params.lc_c_left = src_c_left;
    params.nlc_cr_lp_cnt = dst_cr_lp_cnt;
    params.lc_cr_lp_cnt = dst_cr_lp_cnt;
    params.nlc_cr_left = dst_cr_left;
    params.lc_cr_left = dst_cr_left;
  } else if (max_element(loop_cnt_list.begin(), loop_cnt_list.end()) - loop_cnt_list.begin() == 1) {
    params.mc_pos = 1;
    params.is_mc_cl = 0;
    params.is_mc_cr = 0;
    params.used_core_cnt = GetCeilDiv(src_c_lp_cnt, GetCeilDiv(src_c_lp_cnt, core_num));
    params.nlc_c_lp_cnt = GetCeilDiv(src_c_lp_cnt, params.used_core_cnt);
    params.lc_c_lp_cnt = src_c_lp_cnt - params.nlc_c_lp_cnt * (params.used_core_cnt - 1);
    params.nlc_c_left = 0;
    params.lc_c_left = src_c_left;
    params.core_step_in = params.nlc_c_lp_cnt * params.src_c_lp_step_in;
    params.core_step_out = params.nlc_c_lp_cnt * params.src_c_lp_step_out;
    params.nlc_cr_lp_cnt = dst_cr_lp_cnt;
    params.lc_cr_lp_cnt = dst_cr_lp_cnt;
    params.nlc_cr_left = dst_cr_left;
    params.lc_cr_left = dst_cr_left;
    params.nlc_cl_lp_cnt = dst_cl_lp_cnt;
    params.lc_cl_lp_cnt = dst_cl_lp_cnt;

    params.nlc_cl_left = dst_cl_left;
    params.lc_cl_left = dst_cl_left;
  } else {
    params.mc_pos = 2;
    params.is_mc_cl = 0;
    params.is_mc_cr = 1;
    params.used_core_cnt = GetCeilDiv(dst_cr_lp_cnt, GetCeilDiv(dst_cr_lp_cnt, core_num));
    params.nlc_cr_lp_cnt = GetCeilDiv(dst_cr_lp_cnt, params.used_core_cnt);
    params.lc_cr_lp_cnt = dst_cr_lp_cnt - params.nlc_cr_lp_cnt * (params.used_core_cnt - 1);
    params.nlc_cr_left = 0;
    params.lc_cr_left = dst_cr_left;
    params.core_step_in = params.nlc_cr_lp_cnt * params.dst_cr_lp_step_in;
    params.core_step_out = params.nlc_cr_lp_cnt * params.dst_cr_lp_step_out;
    params.nlc_c_lp_cnt = src_c_lp_cnt;
    params.lc_c_lp_cnt = src_c_lp_cnt;
    params.nlc_c_left = src_c_left;
    params.lc_c_left = src_c_left;
    params.nlc_cl_lp_cnt = dst_cl_lp_cnt;
    params.lc_cl_lp_cnt = dst_cl_lp_cnt;
    params.nlc_cl_left = dst_cl_left;
    params.lc_cl_left = dst_cl_left;
  }
  return true;
}

bool TilingNegativeNtc200(vector<int64_t>& in_shape, vector<int64_t>& out_shape, std::string& src_format,
                            std::string& dst_format, int64_t& core_num, int64_t& block_elem_cnt, DataType& dtype,
                            int64_t ub_size, int64_t& vnc_fp32_flag, TransDataNtc200Param& params) {
  if (src_format.length() < FORMAT_LEN_2D || dst_format.length() < 1) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "TilingNegativeNtc200 Failed.");
    return false;
  }
  OP_TILING_CHECK(block_elem_cnt == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "block_elem_cnt shoule not be 0"),
                  return false);

  int64_t c0_len = in_shape[in_shape.size() - 1];
  params.c0_len = c0_len;

  if (src_format[src_format.length() - 2] == dst_format[dst_format.length() - 1]) {
    params.src_r2nd_dst_r1st_same = 1;
  } else {
    params.src_r2nd_dst_r1st_same = 0;
  }
  params.ub_offset = ub_size / 2 / block_elem_cnt * block_elem_cnt;
  int64_t vnc_col_block_size = GetFloorDiv(params.ub_offset / VNC_LINES, block_elem_cnt);
  if (vnc_col_block_size % 2 == 0) {
    vnc_col_block_size -= 1;
  }
  int64_t vnc_col_size = vnc_col_block_size * block_elem_cnt;
  params.vnc_col_size = vnc_col_size;

  // dst axis C-RIGHT tiling parameters
  params.dst_cr_dims = 2;
  int32_t src_axis_pos_c = std::strchr(src_format.c_str(), 'C') - src_format.c_str();
  int32_t dst_axis_pos_c = std::strchr(dst_format.c_str(), 'C') - dst_format.c_str();
  int64_t axis_dst_cr_size = GetShapeSize(out_shape, dst_axis_pos_c + 1);
  int64_t axis_src_c_size = in_shape[src_axis_pos_c];
  int64_t cr_per_vnc_line = params.ub_offset / c0_len / c0_len * c0_len;
  // once vnchwconv flow
  int64_t tmp_dst_cr_lp_unit;
  int64_t cr_gate;
  if (axis_dst_cr_size % c0_len == 0) {
    cr_gate = 2 * VNC_LINES;
  } else if (GetFloorDiv(cr_per_vnc_line, GetCeilFillA(axis_dst_cr_size, c0_len)) <= axis_src_c_size) {
    cr_gate = 8 * VNC_LINES;
  } else {
    cr_gate = 15 * VNC_LINES;
  }

  if ((dtype == DT_FLOAT16 || dtype == DT_INT8 || dtype == DT_UINT8 ||
      ((dtype == DT_FLOAT || dtype == DT_INT32 || dtype == DT_UINT32) && vnc_fp32_flag == 1)) &&
      (axis_dst_cr_size >= cr_gate)) {
    tmp_dst_cr_lp_unit = params.ub_offset / c0_len / c0_len * c0_len;
  } else {
    // twice vnchwconv flow
    if (dtype == DT_INT8 || dtype == DT_UINT8) {
      tmp_dst_cr_lp_unit = vnc_col_size / 2 / c0_len / block_elem_cnt * block_elem_cnt;
    } else {
      tmp_dst_cr_lp_unit = vnc_col_size / c0_len / block_elem_cnt * block_elem_cnt;
    }
  }

  params.dst_cr_lp_unit = axis_dst_cr_size > tmp_dst_cr_lp_unit ? tmp_dst_cr_lp_unit : axis_dst_cr_size;
  int64_t dst_cr_lp_cnt = GetCeilDiv(axis_dst_cr_size, params.dst_cr_lp_unit);
  int64_t dst_cr_left = axis_dst_cr_size % params.dst_cr_lp_unit;
  string tmp_dst_cr_format = dst_format.substr(dst_axis_pos_c + 1, dst_format.length() - dst_axis_pos_c - 1);
  vector<int64_t> tmp_dst_cr_shape;
  for (size_t i = dst_axis_pos_c + 1; i < out_shape.size(); i++) {
    tmp_dst_cr_shape.push_back(out_shape[i]);
  }
  tmp_dst_cr_shape.push_back(1);
  reverse(tmp_dst_cr_format.begin(), tmp_dst_cr_format.end());
  for (size_t i = 0; i < tmp_dst_cr_format.length(); i++) {
    char chr = tmp_dst_cr_format[i];
    int32_t src_chr_pos = std::strchr(src_format.c_str(), chr) - src_format.c_str();
    int32_t dst_chr_pos = std::strchr(dst_format.c_str(), chr) - dst_format.c_str();
    if (i == 0) {
      params.cr_in_idx_0_size = out_shape[dst_chr_pos];
      params.cr_in_idx_0_dst_rsize = GetShapeSize(tmp_dst_cr_shape, -1 - i);
      params.cr_in_idx_0_src_asize = GetShapeSize(in_shape, src_chr_pos + 1);
    } else if (i == 1) {
      params.cr_in_idx_1_size = out_shape[dst_chr_pos];
      params.cr_in_idx_1_dst_rsize = GetShapeSize(tmp_dst_cr_shape, -1 - i);
      params.cr_in_idx_1_src_asize = GetShapeSize(in_shape, src_chr_pos + 1);
    }
  }
  // suppose there are 2 axises
  int32_t pad_axis_cnt = NTC_FRAME_LEVEL - tmp_dst_cr_format.length();
  if (pad_axis_cnt != 0) {
    params.dst_cr_dims = 1;
    if (tmp_dst_cr_format.length() == 0) {
      params.cr_in_idx_0_size = 1;
      params.cr_in_idx_0_dst_rsize = 1;
      params.cr_in_idx_0_src_asize = 0;
      params.cr_in_idx_1_size = 1;
      params.cr_in_idx_1_dst_rsize = 1;
      params.cr_in_idx_1_src_asize = 0;
    } else if (tmp_dst_cr_format.length() == 1) {
      params.cr_in_idx_1_size = 1;
      params.cr_in_idx_1_dst_rsize = 1;
      params.cr_in_idx_1_src_asize = 0;
    }
  }
  params.dst_cr_step_out = 1;
  params.dst_cr_lp_step_out = params.dst_cr_lp_unit * params.dst_cr_step_out;
  if (params.dst_cr_dims == SHAPE_LEN_2D) {
    params.dst_cr_step_in = 0;
  } else {
    char dst_cr_chr = dst_format[dst_format.length() - 1];
    int32_t dst_cr_in_src = std::strchr(src_format.c_str(), dst_cr_chr) - src_format.c_str();
    params.dst_cr_step_in = GetShapeSize(in_shape, dst_cr_in_src + 1);
  }
  params.dst_cr_lp_step_in = params.dst_cr_lp_unit * params.dst_cr_step_in;
  params.dst_cr_all_in = dst_cr_lp_cnt == 1 ? 1 : 0;

  // axis C tiling parameters
  int64_t axis_dst_c_size = out_shape[dst_axis_pos_c];
  int64_t tmp_src_c_lp_unit;
  if (dst_cr_lp_cnt > 1 || axis_src_c_size == 1) {
    tmp_src_c_lp_unit = 1;
  } else if ((dtype == DT_FLOAT16 || dtype == DT_INT8 || dtype == DT_UINT8 ||
             ((dtype == DT_FLOAT || dtype == DT_INT32 || dtype == DT_UINT32) && vnc_fp32_flag == 1)) &&
             (axis_dst_cr_size >= cr_gate)) {
    tmp_src_c_lp_unit = tmp_dst_cr_lp_unit / GetCeilFillA(params.dst_cr_lp_unit, c0_len);
  }else {
    tmp_src_c_lp_unit = tmp_dst_cr_lp_unit / GetCeilFillA(params.dst_cr_lp_unit, block_elem_cnt);
  }

  params.src_c_lp_unit = axis_src_c_size > tmp_src_c_lp_unit ? tmp_src_c_lp_unit : axis_src_c_size;
  int64_t src_c_lp_cnt = GetCeilDiv(axis_src_c_size, params.src_c_lp_unit);
  int64_t src_c_left = axis_src_c_size % params.src_c_lp_unit;
  params.src_c_step_in = GetShapeSize(in_shape, src_axis_pos_c + 1);
  params.src_c_step_out = GetShapeSize(out_shape, dst_axis_pos_c + 1);
  params.src_c_lp_step_in = params.src_c_lp_unit * params.src_c_step_in;
  params.src_c_lp_step_out = params.src_c_lp_unit * c0_len * params.src_c_step_out;
  params.c_mod_c0 = axis_dst_c_size % c0_len;
  params.dst_c_size = axis_dst_c_size;

  // dst axis C-LEFT tiling parameters
  params.dst_cl_dims = 2;
  int64_t axis_dst_cl_size = 1;
  for (int32_t i = 0; i < dst_axis_pos_c; i++) {
    axis_dst_cl_size *= out_shape[i];
  }
  int64_t src_c_dst_cr_size = axis_src_c_size * axis_dst_cr_size;
  int64_t dst_c_dst_cr_size = axis_dst_c_size * axis_dst_cr_size;
  int64_t tmp_dst_cl_lp_unit;
  if ((dtype == DT_FLOAT16 || dtype == DT_INT8 || dtype == DT_UINT8 ||
      ((dtype == DT_FLOAT || dtype == DT_INT32 || dtype == DT_UINT32) && vnc_fp32_flag == 1)) &&
      (axis_dst_cr_size >= cr_gate)) {
    params.tiling_mode = TILING_MODE_2001;
    tmp_dst_cl_lp_unit = params.ub_offset / (params.src_c_lp_unit * GetCeilFillA(params.dst_cr_lp_unit,
                                                                                 c0_len) * c0_len);
    params.dst_cl_lp_unit = axis_dst_cl_size > tmp_dst_cl_lp_unit ? tmp_dst_cl_lp_unit : axis_dst_cl_size;
  } else if (dst_c_dst_cr_size < 54 * block_elem_cnt && dst_cr_lp_cnt == 1 && src_c_lp_cnt == 1) {
    params.tiling_mode = TILING_MODE_2003;
    int64_t supposed_lp_unit = 4 * block_elem_cnt;
    tmp_dst_cl_lp_unit = tmp_dst_cr_lp_unit / (params.src_c_lp_unit * params.dst_cr_lp_unit);
    params.dst_cl_lp_unit = tmp_dst_cl_lp_unit > supposed_lp_unit ? supposed_lp_unit : tmp_dst_cl_lp_unit;
  } else {
    params.tiling_mode = TILING_MODE_2002;
    params.dst_cl_lp_unit = axis_dst_cl_size > VNC_LINES ? VNC_LINES : axis_dst_cl_size;
  }
  int64_t dst_cl_lp_cnt = GetCeilDiv(axis_dst_cl_size, params.dst_cl_lp_unit);
  int64_t dst_cl_left = axis_dst_cl_size % params.dst_cl_lp_unit;
  // for tiling mode 2003
  params.left_cl_c_cr_size = dst_cl_left * axis_dst_c_size * axis_dst_cr_size;
  string tmp_dst_cl_format = dst_format.substr(0, dst_axis_pos_c);
  vector<int64_t> tmp_c_left_shape;
  for (int32_t i = 0; i < dst_axis_pos_c; i++) {
    tmp_c_left_shape.push_back(out_shape[i]);
  }
  tmp_c_left_shape.push_back(1);

  reverse(tmp_dst_cl_format.begin(), tmp_dst_cl_format.end());
  for (size_t i = 0; i < tmp_dst_cl_format.length(); i++) {
    char chr = tmp_dst_cl_format[i];
    int32_t src_chr_pos = std::strchr(src_format.c_str(), chr) - src_format.c_str();
    int32_t dst_chr_pos = std::strchr(dst_format.c_str(), chr) - dst_format.c_str();
    if (i == 0) {
      params.cl_in_idx_0_size = out_shape[dst_chr_pos];
      params.cl_in_idx_0_dst_rsize = GetShapeSize(tmp_c_left_shape, -1 - i);
      params.cl_in_idx_0_src_asize = GetShapeSize(in_shape, src_chr_pos + 1);
    } else if (i == 1) {
      params.cl_in_idx_1_size = out_shape[dst_chr_pos];
      params.cl_in_idx_1_dst_rsize = GetShapeSize(tmp_c_left_shape, -1 - i);
      params.cl_in_idx_1_src_asize = GetShapeSize(in_shape, src_chr_pos + 1);
    }
  }
  // suppose there are 2 axises
  pad_axis_cnt = NTC_FRAME_LEVEL - tmp_dst_cl_format.length();
  if (pad_axis_cnt != 0) {
    params.dst_cl_dims = 1;
    if (tmp_dst_cl_format.length() == 0) {
      params.cl_in_idx_0_size = 1;
      params.cl_in_idx_0_dst_rsize = 1;
      params.cl_in_idx_0_src_asize = 0;
      params.cl_in_idx_1_size = 1;
      params.cl_in_idx_1_dst_rsize = 1;
      params.cl_in_idx_1_src_asize = 0;
    } else if (tmp_dst_cl_format.length() == 1) {
      params.cl_in_idx_1_size = 1;
      params.cl_in_idx_1_dst_rsize = 1;
      params.cl_in_idx_1_src_asize = 0;
    }
  }

  params.dst_cl_step_out = GetShapeSize(out_shape, dst_axis_pos_c);
  params.dst_cl_lp_step_out = params.dst_cl_lp_unit * params.dst_cl_step_out;
  if (params.dst_cl_dims == 2) {
    params.dst_cl_step_in = 0;
  } else {
    char dst_cl_chr = dst_format[0];
    params.dst_cl_step_in = GetShapeSize(in_shape,
                                         std::strchr(src_format.c_str(), dst_cl_chr) - src_format.c_str() + 1);
  }
  params.dst_cl_lp_step_in = params.dst_cl_lp_unit * params.dst_cl_step_in;

  bool ret = GetMcInfoNegative200(dst_cr_lp_cnt, dst_cr_left, src_c_lp_cnt, src_c_left, dst_cl_lp_cnt, dst_cl_left,
                                  core_num, params);
  if (!ret) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetMcInfoNegative200 Failed.");
    return ret;
  }
  return true;
}

void SetRunningNtc200Params(const TransDataNtc200Param& run_params, utils::OpRunInfo& run_info) {
  run_info.AddTilingData(run_params.tiling_mode);
  run_info.AddTilingData(run_params.ub_offset);
  run_info.AddTilingData(run_params.mc_pos);
  run_info.AddTilingData(run_params.used_core_cnt);
  run_info.AddTilingData(run_params.c0_len);
  run_info.AddTilingData(run_params.core_step_in);
  run_info.AddTilingData(run_params.core_step_out);

  run_info.AddTilingData(run_params.nlc_cr_lp_cnt);
  run_info.AddTilingData(run_params.nlc_c_lp_cnt);
  run_info.AddTilingData(run_params.nlc_cl_lp_cnt);
  run_info.AddTilingData(run_params.nlc_cr_left);
  run_info.AddTilingData(run_params.nlc_c_left);
  run_info.AddTilingData(run_params.nlc_cl_left);
  run_info.AddTilingData(run_params.lc_cr_lp_cnt);
  run_info.AddTilingData(run_params.lc_c_lp_cnt);
  run_info.AddTilingData(run_params.lc_cl_lp_cnt);
  run_info.AddTilingData(run_params.lc_cr_left);
  run_info.AddTilingData(run_params.lc_c_left);
  run_info.AddTilingData(run_params.lc_cl_left);
  run_info.AddTilingData(run_params.dst_cr_lp_unit);
  run_info.AddTilingData(run_params.src_c_lp_unit);
  run_info.AddTilingData(run_params.dst_cl_lp_unit);
  run_info.AddTilingData(run_params.vnc_col_size);
  run_info.AddTilingData(run_params.dst_cr_step_in);
  run_info.AddTilingData(run_params.dst_cr_step_out);
  run_info.AddTilingData(run_params.dst_cr_lp_step_in);
  run_info.AddTilingData(run_params.dst_cr_lp_step_out);
  run_info.AddTilingData(run_params.dst_c_size);
  run_info.AddTilingData(run_params.src_c_step_in);
  run_info.AddTilingData(run_params.src_c_step_out);
  run_info.AddTilingData(run_params.src_c_lp_step_in);
  run_info.AddTilingData(run_params.src_c_lp_step_out);
  run_info.AddTilingData(run_params.dst_cr_all_in);
  run_info.AddTilingData(run_params.dst_cl_step_in);
  run_info.AddTilingData(run_params.dst_cl_step_out);
  run_info.AddTilingData(run_params.dst_cl_lp_step_in);
  run_info.AddTilingData(run_params.dst_cl_lp_step_out);
  run_info.AddTilingData(run_params.c_mod_c0);
  run_info.AddTilingData(run_params.dst_cr_dims);
  run_info.AddTilingData(run_params.dst_cl_dims);
  run_info.AddTilingData(run_params.is_mc_cr);
  run_info.AddTilingData(run_params.is_mc_cl);
  run_info.AddTilingData(run_params.src_r2nd_dst_r1st_same);
  run_info.AddTilingData(run_params.left_cl_c_cr_size);

  run_info.AddTilingData(run_params.cl_in_idx_0_size);
  run_info.AddTilingData(run_params.cl_in_idx_0_dst_rsize);
  run_info.AddTilingData(run_params.cl_in_idx_0_src_asize);
  run_info.AddTilingData(run_params.cl_in_idx_1_size);
  run_info.AddTilingData(run_params.cl_in_idx_1_dst_rsize);
  run_info.AddTilingData(run_params.cl_in_idx_1_src_asize);
  run_info.AddTilingData(run_params.cr_in_idx_0_size);
  run_info.AddTilingData(run_params.cr_in_idx_0_dst_rsize);
  run_info.AddTilingData(run_params.cr_in_idx_0_src_asize);
  run_info.AddTilingData(run_params.cr_in_idx_1_size);
  run_info.AddTilingData(run_params.cr_in_idx_1_dst_rsize);
  run_info.AddTilingData(run_params.cr_in_idx_1_src_asize);
}

void PrintTilingModeNtc200Params(const std::string& op_type, const TransDataNtc200Param& params) {
  OP_LOGD(op_type, "tiling_mode=%d", params.tiling_mode);
  OP_LOGD(op_type, "ub_offset=%d", params.ub_offset);
  OP_LOGD(op_type, "mc_pos=%d", params.mc_pos);
  OP_LOGD(op_type, "used_core_cnt=%d", params.used_core_cnt);
  OP_LOGD(op_type, "c0_len=%d", params.c0_len);
  OP_LOGD(op_type, "core_step_in=%d", params.core_step_in);
  OP_LOGD(op_type, "core_step_out=%d", params.core_step_out);

  OP_LOGD(op_type, "nlc_cr_lp_cnt=%d", params.nlc_cr_lp_cnt);
  OP_LOGD(op_type, "nlc_c_lp_cnt=%d", params.nlc_c_lp_cnt);
  OP_LOGD(op_type, "nlc_cl_lp_cnt=%d", params.nlc_cl_lp_cnt);
  OP_LOGD(op_type, "nlc_cr_left=%d", params.nlc_cr_left);
  OP_LOGD(op_type, "nlc_c_left=%d", params.nlc_c_left);
  OP_LOGD(op_type, "nlc_cl_left=%d", params.nlc_cl_left);
  OP_LOGD(op_type, "lc_cr_lp_cnt=%d", params.lc_cr_lp_cnt);
  OP_LOGD(op_type, "lc_c_lp_cnt=%d", params.lc_c_lp_cnt);
  OP_LOGD(op_type, "lc_cl_lp_cnt=%d", params.lc_cl_lp_cnt);
  OP_LOGD(op_type, "lc_cr_left=%d", params.lc_cr_left);
  OP_LOGD(op_type, "lc_c_left=%d", params.lc_c_left);
  OP_LOGD(op_type, "lc_cl_left=%d", params.lc_cl_left);
  OP_LOGD(op_type, "dst_cr_lp_unit=%d", params.dst_cr_lp_unit);
  OP_LOGD(op_type, "src_c_lp_unit=%d", params.src_c_lp_unit);
  OP_LOGD(op_type, "dst_cl_lp_unit=%d", params.dst_cl_lp_unit);
  OP_LOGD(op_type, "vnc_col_size=%d", params.vnc_col_size);
  OP_LOGD(op_type, "dst_cr_step_in=%d", params.dst_cr_step_in);
  OP_LOGD(op_type, "dst_cr_step_out=%d", params.dst_cr_step_out);
  OP_LOGD(op_type, "dst_cr_lp_step_in=%d", params.dst_cr_lp_step_in);
  OP_LOGD(op_type, "dst_cr_lp_step_out=%d", params.dst_cr_lp_step_out);
  OP_LOGD(op_type, "dst_c_size=%d", params.dst_c_size);
  OP_LOGD(op_type, "src_c_step_in=%d", params.src_c_step_in);
  OP_LOGD(op_type, "src_c_step_out=%d", params.src_c_step_out);
  OP_LOGD(op_type, "src_c_lp_step_in=%d", params.src_c_lp_step_in);
  OP_LOGD(op_type, "src_c_lp_step_out=%d", params.src_c_lp_step_out);
  OP_LOGD(op_type, "dst_cr_all_in=%d", params.dst_cr_all_in);
  OP_LOGD(op_type, "dst_cl_step_in=%d", params.dst_cl_step_in);
  OP_LOGD(op_type, "dst_cl_step_out=%d", params.dst_cl_step_out);
  OP_LOGD(op_type, "dst_cl_lp_step_in=%d", params.dst_cl_lp_step_in);
  OP_LOGD(op_type, "dst_cl_lp_step_out=%d", params.dst_cl_lp_step_out);
  OP_LOGD(op_type, "c_mod_c0=%d", params.c_mod_c0);
  OP_LOGD(op_type, "dst_cr_dims=%d", params.dst_cr_dims);
  OP_LOGD(op_type, "dst_cl_dims=%d", params.dst_cl_dims);
  OP_LOGD(op_type, "is_mc_cr=%d", params.is_mc_cr);
  OP_LOGD(op_type, "is_mc_cl=%d", params.is_mc_cl);

  OP_LOGD(op_type, "src_r2nd_dst_r1st_same=%d", params.src_r2nd_dst_r1st_same);
  OP_LOGD(op_type, "left_cl_c_cr_size=%d", params.left_cl_c_cr_size);
  OP_LOGD(op_type, "cl_in_idx_0_size=%d", params.cl_in_idx_0_size);
  OP_LOGD(op_type, "cl_in_idx_0_dst_rsize=%d", params.cl_in_idx_0_dst_rsize);
  OP_LOGD(op_type, "cl_in_idx_0_src_asize=%d", params.cl_in_idx_0_src_asize);
  OP_LOGD(op_type, "cl_in_idx_1_size=%d", params.cl_in_idx_1_size);
  OP_LOGD(op_type, "cl_in_idx_1_dst_rsize=%d", params.cl_in_idx_1_dst_rsize);
  OP_LOGD(op_type, "cl_in_idx_1_src_asize=%d", params.cl_in_idx_1_src_asize);

  OP_LOGD(op_type, "cr_in_idx_0_size=%d", params.cr_in_idx_0_size);
  OP_LOGD(op_type, "cr_in_idx_0_dst_rsize=%d", params.cr_in_idx_0_dst_rsize);
  OP_LOGD(op_type, "cr_in_idx_0_src_asize=%d", params.cr_in_idx_0_src_asize);
  OP_LOGD(op_type, "cr_in_idx_1_size=%d", params.cr_in_idx_1_size);
  OP_LOGD(op_type, "cr_in_idx_1_dst_rsize=%d", params.cr_in_idx_1_dst_rsize);
  OP_LOGD(op_type, "cr_in_idx_1_src_asize=%d", params.cr_in_idx_1_src_asize);
}
}  // namespace optiling
