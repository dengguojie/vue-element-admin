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
 * \file trans_data_negative_target_tc_201.cc
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
const int32_t TC_FRAME_LEVEL = 2;
constexpr int64_t NI_16 = 16;
constexpr int64_t C0_16 = 16;
constexpr int64_t VNC_LINES = 16;

bool GetMcInfoNegative201(int64_t& dst_r2nd_lp_cnt, int64_t dst_r2nd_left, int64_t& src_cl_lp_cnt,
                          int64_t src_cl_left, int64_t& src_left_lp_cnt, int64_t src_left_left,
                          int64_t& core_num, TransDataTc201Param& params) {
  int64_t tmp_full_loop_cnt_r2nd;
  if (GetFloorDiv(dst_r2nd_lp_cnt, core_num) > 0) {
    tmp_full_loop_cnt_r2nd = core_num;
  } else {
    tmp_full_loop_cnt_r2nd = 0;
  }
  int64_t reminder_loop_cnt_r2nd = dst_r2nd_lp_cnt % core_num;
  if (reminder_loop_cnt_r2nd == 0) {
    tmp_full_loop_cnt_r2nd += core_num;
  }
  int64_t full_loop_cnt_r2nd = tmp_full_loop_cnt_r2nd + reminder_loop_cnt_r2nd;

  int64_t tmp_full_loop_cnt_c1;
  if (GetFloorDiv(src_cl_lp_cnt, core_num) > 0) {
    tmp_full_loop_cnt_c1 = core_num;
  } else {
    tmp_full_loop_cnt_c1 = 0;
  }
  int64_t reminder_loop_cnt_c1 = src_cl_lp_cnt % core_num;
  if (reminder_loop_cnt_c1 == 0) {
    tmp_full_loop_cnt_c1 += core_num;
  }
  int64_t full_loop_cnt_c1 = tmp_full_loop_cnt_c1 + reminder_loop_cnt_c1;

  int64_t tmp_full_loop_cnt_left;
  if (GetFloorDiv(src_left_lp_cnt, core_num) > 0) {
    tmp_full_loop_cnt_left = core_num;
  } else {
    tmp_full_loop_cnt_left = 0;
  }
  int64_t reminder_loop_cnt_left = src_left_lp_cnt % core_num;
  if (reminder_loop_cnt_left == 0) {
    tmp_full_loop_cnt_left += core_num;
  }
  int64_t full_loop_cnt_left = tmp_full_loop_cnt_left + reminder_loop_cnt_left;
  vector<int64_t> loop_cnt_list = {full_loop_cnt_left, full_loop_cnt_c1, full_loop_cnt_r2nd};

  if (max_element(loop_cnt_list.begin(), loop_cnt_list.end()) - loop_cnt_list.begin() == 0) {
    params.mc_pos = 0;
    params.used_core_cnt = GetCeilDiv(src_left_lp_cnt, GetCeilDiv(src_left_lp_cnt, core_num));
    params.nlc_src_left_lp_cnt = GetCeilDiv(src_left_lp_cnt, params.used_core_cnt);
    params.lc_src_left_lp_cnt = src_left_lp_cnt - params.nlc_src_left_lp_cnt * (params.used_core_cnt - 1);
    params.nlc_src_left_left = 0;
    params.lc_src_left_left = src_left_left;
    params.core_step_in = params.nlc_src_left_lp_cnt * params.src_left_lp_step_in;
    params.core_step_out = params.nlc_src_left_lp_cnt * params.src_left_lp_step_out;
    params.nlc_src_cl_lp_cnt = src_cl_lp_cnt;
    params.lc_src_cl_lp_cnt = src_cl_lp_cnt;
    params.nlc_src_cl_left = src_cl_left;
    params.lc_src_cl_left = src_cl_left;
    params.nlc_dst_r2nd_lp_cnt = dst_r2nd_lp_cnt;
    params.lc_dst_r2nd_lp_cnt = dst_r2nd_lp_cnt;
    params.nlc_dst_r2nd_left = dst_r2nd_left;
    params.lc_dst_r2nd_left = dst_r2nd_left;
  } else if (max_element(loop_cnt_list.begin(), loop_cnt_list.end()) - loop_cnt_list.begin() == 1) {
    params.mc_pos = 1;
    params.used_core_cnt = GetCeilDiv(src_cl_lp_cnt, GetCeilDiv(src_cl_lp_cnt, core_num));
    params.nlc_src_cl_lp_cnt = GetCeilDiv(src_cl_lp_cnt, params.used_core_cnt);
    params.lc_src_cl_lp_cnt = src_cl_lp_cnt - params.nlc_src_cl_lp_cnt * (params.used_core_cnt - 1);
    params.nlc_src_cl_left = 0;
    params.lc_src_cl_left = src_cl_left;
    params.core_step_in = params.nlc_src_cl_lp_cnt * params.src_cl_lp_step_in;
    params.core_step_out = params.nlc_src_cl_lp_cnt * params.src_cl_lp_step_out;
    params.nlc_dst_r2nd_lp_cnt = dst_r2nd_lp_cnt;
    params.lc_dst_r2nd_lp_cnt = dst_r2nd_lp_cnt;
    params.nlc_dst_r2nd_left = dst_r2nd_left;
    params.lc_dst_r2nd_left = dst_r2nd_left;
    params.nlc_src_left_lp_cnt = src_left_lp_cnt;
    params.lc_src_left_lp_cnt = src_left_lp_cnt;
    params.nlc_src_left_left = src_left_left;
    params.lc_src_left_left = src_left_left;
  } else {
    params.mc_pos = TRANSDATA_TILING_PARAM_2;
    params.used_core_cnt = GetCeilDiv(dst_r2nd_lp_cnt, GetCeilDiv(dst_r2nd_lp_cnt, core_num));
    params.nlc_dst_r2nd_lp_cnt = GetCeilDiv(dst_r2nd_lp_cnt, params.used_core_cnt);
    params.lc_dst_r2nd_lp_cnt = dst_r2nd_lp_cnt - params.nlc_dst_r2nd_lp_cnt * (params.used_core_cnt - 1);
    params.nlc_dst_r2nd_left = 0;
    params.lc_dst_r2nd_left = dst_r2nd_left;
    params.core_step_in = params.nlc_dst_r2nd_lp_cnt * params.dst_r2nd_lp_step_in;
    params.core_step_out = params.nlc_dst_r2nd_lp_cnt * params.dst_r2nd_lp_step_out;
    params.nlc_src_left_lp_cnt = src_left_lp_cnt;
    params.lc_src_left_lp_cnt = src_left_lp_cnt;
    params.nlc_src_left_left = src_left_left;
    params.lc_src_left_left = src_left_left;
    params.nlc_src_cl_lp_cnt = src_cl_lp_cnt;
    params.lc_src_cl_lp_cnt = src_cl_lp_cnt;
    params.nlc_src_cl_left = src_cl_left;
    params.lc_src_cl_left = src_cl_left;
  }
  return true;
}

bool TilingNegativeTc201(vector<int64_t>& in_shape, vector<int64_t>& out_shape, std::string& src_format,
                         std::string& dst_format, int64_t& core_num, int64_t& block_elem_cnt, DataType& dtype,
                         int64_t& ub_size, TransDataTc201Param& params) {
  if (src_format.length() < FORMAT_LEN_2D || dst_format.length() < FORMAT_LEN_2D) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "TilingNegativeTc201 Failed.");
    return false;
  }
  OP_TILING_CHECK(block_elem_cnt == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "block_elem_cnt shoule not be 0"),
                  return false);

  int64_t c0_len = in_shape[in_shape.size() - 1];
  params.c0_len = c0_len;

  int32_t src_axis_pos_c = std::strchr(src_format.c_str(), 'C') - src_format.c_str();
  int32_t dst_axis_pos_c = std::strchr(dst_format.c_str(), 'C') - dst_format.c_str();
  int64_t axis_dst_c_size = out_shape[dst_axis_pos_c];
  int64_t axis_src_c1_size = in_shape[src_axis_pos_c];
  vector<int64_t> dst_r2nd_shape;
  string dst_r2nd_format = "";
  int64_t axis_dst_r2nd_size;
  int64_t axis_src_left_size;
  string src_left_format = "";
  if (src_format[src_format.length() - 2] == dst_format[dst_format.length() - 2]) {
    params.src_r2nd_dst_r2nd_same = 1;
    dst_r2nd_format += dst_format[dst_format.length() - 2];
    dst_r2nd_shape.push_back(out_shape[out_shape.size() - 2]);
    axis_dst_r2nd_size = out_shape[out_shape.size() - 2];
    src_left_format += src_format[0];
    axis_src_left_size = out_shape[std::strchr(dst_format.c_str(), src_format[0]) - dst_format.c_str()];
  } else {
    params.src_r2nd_dst_r2nd_same = 0;
    src_left_format += src_format[src_format.length() - 2];
    axis_src_left_size = out_shape[std::strchr(dst_format.c_str(),
                                               src_format[src_format.length() - 2]) - dst_format.c_str()];
    dst_r2nd_format = src_format.substr(0, src_format.length() - 2);
    auto chr_c_pos = dst_r2nd_format.find('C');
    if (chr_c_pos != std::string::npos) {
      dst_r2nd_format.replace(chr_c_pos, 1, "");
    }
    axis_dst_r2nd_size = 1;
    for (size_t i = 0; i < dst_r2nd_format.length(); i++) {
      char chr = dst_r2nd_format[i];
      int32_t src_chr_pos = std::strchr(src_format.c_str(), chr) - src_format.c_str();
      axis_dst_r2nd_size *= in_shape[src_chr_pos];
      dst_r2nd_shape.push_back(in_shape[src_chr_pos]);
    }
  }
  dst_r2nd_shape.push_back(1);

  // output ub offset
  params.ub_offset = ub_size / TRANSDATA_TILING_FACTOR_2 / block_elem_cnt * block_elem_cnt;
  // axis c1 tiling parameters
  int64_t vnc_col_block_cnt = GetFloorDiv(params.ub_offset / VNC_LINES, block_elem_cnt);
  if (vnc_col_block_cnt % TRANSDATA_TILING_FACTOR_2 == 0) {
    vnc_col_block_cnt -= 1;
  }
  int64_t vnc_col_size = vnc_col_block_cnt * block_elem_cnt;
  params.vnc_col_size = vnc_col_size;
  int64_t tmp_src_cl_lp_unit;
  int64_t c_gate = 0;
  if (axis_dst_c_size % params.c0_len == 0) {
    c_gate = TRANSDATA_TILING_FACTOR_16 * params.c0_len;
  } else {
    c_gate = TRANSDATA_TILING_FACTOR_56 * params.c0_len;
  }

  if (axis_src_c1_size * c0_len >= c_gate || axis_dst_c_size == c0_len) {
    params.tiling_mode = TILING_MODE_2010;
    if (axis_dst_r2nd_size < NI_16) {
      tmp_src_cl_lp_unit = GetFloorDiv(params.ub_offset, axis_dst_r2nd_size * params.c0_len);
    } else {
      tmp_src_cl_lp_unit = GetFloorDiv(params.ub_offset, NI_16 * params.c0_len);
    }
  } else if (dtype != DT_INT8 && dtype != DT_UINT8) {
    if (axis_dst_c_size * axis_dst_r2nd_size >= vnc_col_size / VNC_LINES) {
      params.tiling_mode = TILING_MODE_2011;
    } else {
      params.tiling_mode = TILING_MODE_2012;
    }
    tmp_src_cl_lp_unit = vnc_col_size / c0_len / block_elem_cnt * block_elem_cnt;
  } else {
    if (axis_dst_c_size * axis_dst_r2nd_size >= vnc_col_size / TRANSDATA_TILING_FACTOR_2 / VNC_LINES) {
      params.tiling_mode = TILING_MODE_2011;
    } else {
      params.tiling_mode = TILING_MODE_2012;
    }
    tmp_src_cl_lp_unit = vnc_col_size / TRANSDATA_TILING_FACTOR_2 / c0_len / block_elem_cnt * block_elem_cnt;
  }

  params.src_cl_lp_unit = axis_src_c1_size > tmp_src_cl_lp_unit ? tmp_src_cl_lp_unit : axis_src_c1_size;
  int64_t src_cl_lp_cnt = GetCeilDiv(axis_src_c1_size, params.src_cl_lp_unit);
  int64_t src_cl_left = axis_src_c1_size % params.src_cl_lp_unit;
  params.src_cl_lp_step_in = params.src_cl_lp_unit * GetShapeSize(in_shape, src_axis_pos_c + 1);
  params.src_cl_lp_step_out = params.src_cl_lp_unit * c0_len;
  params.src_cl_step_in = GetShapeSize(in_shape, src_axis_pos_c + 1);
  params.src_cl_step_out = 1;
  params.c_mod_c0 = axis_dst_c_size % c0_len;
  if (src_cl_lp_cnt == 1) {
    params.all_c_in = 1;
  } else {
    params.all_c_in = 0;
  }

  // axis -2 tiling parameters
  params.dst_r2nd_dims = TRANSDATA_TILING_PARAM_2;
  int64_t tmp_dst_r2nd_lp_unit;
  int64_t dtype_factor = 1;
  // to make sure the rep_stride of vor is less than limit
  if (params.tiling_mode == TILING_MODE_2010) {
    int64_t max_r2nd_lp_size = TRANSDATA_TILING_PARAM_63;
    if (dtype == DT_FLOAT || dtype == DT_INT32 || dtype == DT_UINT32) {
      if (axis_dst_c_size == params.c0_len && axis_src_left_size <= C0_16) {
        // for vor in copy data in
        max_r2nd_lp_size = TRANSDATA_TILING_PARAM_63;
      } else {
        // for vor in reorder
        max_r2nd_lp_size = TRANSDATA_TILING_PARAM_31;
      }
      dtype_factor = TRANSDATA_TILING_FACTOR_2;
    } else if (axis_dst_c_size == params.c0_len && axis_src_left_size <= C0_16) {
      max_r2nd_lp_size = TRANSDATA_TILING_PARAM_127;
    }
    tmp_dst_r2nd_lp_unit = GetFloorDiv(params.ub_offset, params.src_cl_lp_unit * c0_len);
    if (tmp_dst_r2nd_lp_unit > max_r2nd_lp_size) {
      tmp_dst_r2nd_lp_unit = max_r2nd_lp_size;
    }
  } else if (dtype != DT_INT8 && dtype != DT_UINT8) {
    tmp_dst_r2nd_lp_unit = vnc_col_size / (params.src_cl_lp_unit * c0_len);
  } else {
    tmp_dst_r2nd_lp_unit = vnc_col_size / TRANSDATA_TILING_FACTOR_2 / (params.src_cl_lp_unit * c0_len);
  }
  params.dst_r2nd_lp_unit = axis_dst_r2nd_size > tmp_dst_r2nd_lp_unit ? tmp_dst_r2nd_lp_unit : axis_dst_r2nd_size;
  int64_t r2nd_c_mod_block = params.dst_r2nd_lp_unit * axis_dst_c_size % block_elem_cnt;
  if (params.tiling_mode == TILING_MODE_2011 && r2nd_c_mod_block > 0 &&
      axis_dst_r2nd_size > params.dst_r2nd_lp_unit && params.dst_r2nd_lp_unit > block_elem_cnt) {
    params.dst_r2nd_lp_unit = GetFloorDiv(params.dst_r2nd_lp_unit, block_elem_cnt) * block_elem_cnt;
  }
  // to avoid bank conflict
  if (params.tiling_mode == TILING_MODE_2010 && params.dst_r2nd_lp_unit * dtype_factor % NI_16 == 0 &&
      (params.dst_r2nd_lp_unit < params.src_cl_lp_unit || params.src_cl_lp_unit * dtype_factor % NI_16 == 0)) {
    params.dst_r2nd_lp_unit -= 1;
  }
  int64_t dst_r2nd_lp_cnt = GetCeilDiv(axis_dst_r2nd_size, params.dst_r2nd_lp_unit);
  int64_t dst_r2nd_left = axis_dst_r2nd_size % params.dst_r2nd_lp_unit;
  if (dst_r2nd_lp_cnt == 1) {
    params.all_r2nd_in = 1;
  } else {
    params.all_r2nd_in = 0;
  }

  reverse(dst_r2nd_format.begin(), dst_r2nd_format.end());
  for (size_t i = 0; i < dst_r2nd_format.length(); i++) {
    char chr = dst_r2nd_format[i];
    int32_t src_chr_pos = std::strchr(src_format.c_str(), chr) - src_format.c_str();
    if (i == 0) {
      params.dst_r2nd_in_0_size = in_shape[src_chr_pos];
      params.dst_r2nd_in_0_src_rsize = GetShapeSize(dst_r2nd_shape, -1 - i);
      params.dst_r2nd_in_0_src_asize = GetShapeSize(in_shape, src_chr_pos + 1);
    } else if (i == 1) {
      params.dst_r2nd_in_1_size = in_shape[src_chr_pos];
      params.dst_r2nd_in_1_src_rsize = GetShapeSize(dst_r2nd_shape, -1 - i);
      params.dst_r2nd_in_1_src_asize = GetShapeSize(in_shape, src_chr_pos + 1);
    }
  }
  int32_t pad_axis_cnt = TC_FRAME_LEVEL - dst_r2nd_format.length();
  if (pad_axis_cnt != 0) {
    params.dst_r2nd_dims = 1;
    if (dst_r2nd_format.length() == 0) {
      params.dst_r2nd_in_0_size = 1;
      params.dst_r2nd_in_0_src_rsize = 1;
      params.dst_r2nd_in_0_src_asize = 0;
      params.dst_r2nd_in_1_size = 1;
      params.dst_r2nd_in_1_src_rsize = 1;
      params.dst_r2nd_in_1_src_asize = 0;
    } else if (dst_r2nd_format.length() == 1) {
      params.dst_r2nd_in_1_size = 1;
      params.dst_r2nd_in_1_src_rsize = 1;
      params.dst_r2nd_in_1_src_asize = 0;
    }
  }
  if (params.dst_r2nd_dims == TRANSDATA_TILING_PARAM_2) {
    params.dst_r2nd_step_in = 0;
  } else {
    params.dst_r2nd_step_in = c0_len;
  }
  params.dst_r2nd_lp_step_in = params.dst_r2nd_lp_unit * params.dst_r2nd_step_in;
  params.dst_r2nd_step_out = axis_dst_c_size;
  params.dst_r2nd_lp_step_out = params.dst_r2nd_lp_unit * params.dst_r2nd_step_out;

  int64_t tmp_src_left_lp_unit;
  if (params.tiling_mode == TILING_MODE_2010) {
    tmp_src_left_lp_unit = params.ub_offset / (params.src_cl_lp_unit * params.dst_r2nd_lp_unit * c0_len);
    if (tmp_src_left_lp_unit > axis_src_left_size / core_num && axis_src_left_size >= core_num) {
      tmp_src_left_lp_unit = axis_src_left_size / core_num;
    }
  } else if (dtype != DT_INT8 && dtype != DT_UINT8) {
    tmp_src_left_lp_unit = vnc_col_size / (params.src_cl_lp_unit * params.dst_r2nd_lp_unit * c0_len);
  } else {
    tmp_src_left_lp_unit = vnc_col_size / TRANSDATA_TILING_FACTOR_2 / (
        params.src_cl_lp_unit * params.dst_r2nd_lp_unit * c0_len);
  }
  if (params.tiling_mode == TILING_MODE_2011) {
    tmp_src_left_lp_unit = NI_16;
  }
  params.src_left_lp_unit = axis_src_left_size > tmp_src_left_lp_unit ? tmp_src_left_lp_unit : axis_src_left_size;
  int64_t left_r2nd_c_mod_block = params.src_left_lp_unit * params.dst_r2nd_lp_unit * axis_dst_c_size % block_elem_cnt;
  if (params.tiling_mode == TILING_MODE_2012 && left_r2nd_c_mod_block > 0 &&
      axis_src_left_size > params.src_left_lp_unit && params.src_left_lp_unit > block_elem_cnt) {
    params.src_left_lp_unit = GetFloorDiv(params.src_left_lp_unit, block_elem_cnt) * block_elem_cnt;
  }
  int64_t src_left_lp_cnt = GetCeilDiv(axis_src_left_size, params.src_left_lp_unit);
  int64_t src_left_left = axis_src_left_size % params.src_left_lp_unit;
  params.src_left_step_in = GetShapeSize(in_shape, src_format.find(src_left_format) + 1);
  params.src_left_lp_step_in = params.src_left_lp_unit * params.src_left_step_in;
  params.src_left_step_out = GetShapeSize(out_shape, dst_format.find(src_left_format) + 1);
  params.src_left_lp_step_out = params.src_left_lp_unit * params.src_left_step_out;

  bool ret = GetMcInfoNegative201(dst_r2nd_lp_cnt, dst_r2nd_left, src_cl_lp_cnt, src_cl_left,
                                  src_left_lp_cnt, src_left_left, core_num, params);
  if (!ret) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetMcInfoNegative201 Failed.");
    return ret;
  }
  return true;
}

void SetRunningTc201Params(const TransDataTc201Param& run_params, utils::OpRunInfo& run_info) {
  run_info.AddTilingData(run_params.tiling_mode);
  run_info.AddTilingData(run_params.ub_offset);
  run_info.AddTilingData(run_params.mc_pos);
  run_info.AddTilingData(run_params.used_core_cnt);
  run_info.AddTilingData(run_params.src_r2nd_dst_r2nd_same);
  run_info.AddTilingData(run_params.c0_len);
  run_info.AddTilingData(run_params.core_step_in);
  run_info.AddTilingData(run_params.core_step_out);
  run_info.AddTilingData(run_params.nlc_dst_r2nd_lp_cnt);
  run_info.AddTilingData(run_params.nlc_src_cl_lp_cnt);
  run_info.AddTilingData(run_params.nlc_src_left_lp_cnt);
  run_info.AddTilingData(run_params.nlc_dst_r2nd_left);
  run_info.AddTilingData(run_params.nlc_src_cl_left);
  run_info.AddTilingData(run_params.nlc_src_left_left);
  run_info.AddTilingData(run_params.lc_dst_r2nd_lp_cnt);
  run_info.AddTilingData(run_params.lc_src_cl_lp_cnt);
  run_info.AddTilingData(run_params.lc_src_left_lp_cnt);
  run_info.AddTilingData(run_params.lc_dst_r2nd_left);
  run_info.AddTilingData(run_params.lc_src_cl_left);
  run_info.AddTilingData(run_params.lc_src_left_left);
  run_info.AddTilingData(run_params.dst_r2nd_lp_unit);
  run_info.AddTilingData(run_params.dst_r2nd_step_in);
  run_info.AddTilingData(run_params.dst_r2nd_step_out);
  run_info.AddTilingData(run_params.dst_r2nd_lp_step_in);
  run_info.AddTilingData(run_params.dst_r2nd_lp_step_out);
  run_info.AddTilingData(run_params.src_cl_lp_unit);
  run_info.AddTilingData(run_params.all_c_in);
  run_info.AddTilingData(run_params.src_cl_step_in);
  run_info.AddTilingData(run_params.src_cl_step_out);
  run_info.AddTilingData(run_params.src_cl_lp_step_in);
  run_info.AddTilingData(run_params.src_cl_lp_step_out);
  run_info.AddTilingData(run_params.c_mod_c0);
  run_info.AddTilingData(run_params.src_left_lp_unit);
  run_info.AddTilingData(run_params.src_left_step_in);
  run_info.AddTilingData(run_params.src_left_step_out);
  run_info.AddTilingData(run_params.src_left_lp_step_in);
  run_info.AddTilingData(run_params.src_left_lp_step_out);
  run_info.AddTilingData(run_params.dst_r2nd_in_0_size);
  run_info.AddTilingData(run_params.dst_r2nd_in_0_src_rsize);
  run_info.AddTilingData(run_params.dst_r2nd_in_0_src_asize);
  run_info.AddTilingData(run_params.dst_r2nd_in_1_size);
  run_info.AddTilingData(run_params.dst_r2nd_in_1_src_rsize);
  run_info.AddTilingData(run_params.dst_r2nd_in_1_src_asize);
  run_info.AddTilingData(run_params.dst_r2nd_dims);
  run_info.AddTilingData(run_params.vnc_col_size);
  run_info.AddTilingData(run_params.all_r2nd_in);
}

void PrintTilingModeTc201Params(const std::string& op_type, const TransDataTc201Param& params) {
  OP_LOGD(op_type, "tiling_mode=%d", params.tiling_mode);
  OP_LOGD(op_type, "ub_offset=%d", params.ub_offset);
  OP_LOGD(op_type, "mc_pos=%d", params.mc_pos);
  OP_LOGD(op_type, "used_core_cnt=%d", params.used_core_cnt);
  OP_LOGD(op_type, "src_r2nd_dst_r2nd_same=%d", params.src_r2nd_dst_r2nd_same);
  OP_LOGD(op_type, "c0_len=%d", params.c0_len);
  OP_LOGD(op_type, "core_step_in=%d", params.core_step_in);
  OP_LOGD(op_type, "core_step_out=%d", params.core_step_out);
  OP_LOGD(op_type, "nlc_dst_r2nd_lp_cnt=%d", params.nlc_dst_r2nd_lp_cnt);
  OP_LOGD(op_type, "nlc_src_cl_lp_cnt=%d", params.nlc_src_cl_lp_cnt);
  OP_LOGD(op_type, "nlc_src_left_lp_cnt=%d", params.nlc_src_left_lp_cnt);
  OP_LOGD(op_type, "nlc_dst_r2nd_left=%d", params.nlc_dst_r2nd_left);
  OP_LOGD(op_type, "nlc_src_cl_left=%d", params.nlc_src_cl_left);
  OP_LOGD(op_type, "nlc_src_left_left=%d", params.nlc_src_left_left);
  OP_LOGD(op_type, "lc_dst_r2nd_lp_cnt=%d", params.lc_dst_r2nd_lp_cnt);
  OP_LOGD(op_type, "lc_src_cl_lp_cnt=%d", params.lc_src_cl_lp_cnt);
  OP_LOGD(op_type, "lc_src_left_lp_cnt=%d", params.lc_src_left_lp_cnt);
  OP_LOGD(op_type, "lc_dst_r2nd_left=%d", params.lc_dst_r2nd_left);
  OP_LOGD(op_type, "lc_src_cl_left=%d", params.lc_src_cl_left);
  OP_LOGD(op_type, "lc_src_left_left=%d", params.lc_src_left_left);
  OP_LOGD(op_type, "dst_r2nd_lp_unit=%d", params.dst_r2nd_lp_unit);
  OP_LOGD(op_type, "dst_r2nd_step_in=%d", params.dst_r2nd_step_in);
  OP_LOGD(op_type, "dst_r2nd_step_out=%d", params.dst_r2nd_step_out);
  OP_LOGD(op_type, "dst_r2nd_lp_step_in=%d", params.dst_r2nd_lp_step_in);
  OP_LOGD(op_type, "dst_r2nd_lp_step_out=%d", params.dst_r2nd_lp_step_out);
  OP_LOGD(op_type, "src_cl_lp_unit=%d", params.src_cl_lp_unit);
  OP_LOGD(op_type, "all_c_in=%d", params.all_c_in);
  OP_LOGD(op_type, "src_cl_step_in=%d", params.src_cl_step_in);
  OP_LOGD(op_type, "src_cl_step_out=%d", params.src_cl_step_out);
  OP_LOGD(op_type, "src_cl_lp_step_in=%d", params.src_cl_lp_step_in);
  OP_LOGD(op_type, "src_cl_lp_step_out=%d", params.src_cl_lp_step_out);
  OP_LOGD(op_type, "c_mod_c0=%d", params.c_mod_c0);
  OP_LOGD(op_type, "src_left_lp_unit=%d", params.src_left_lp_unit);
  OP_LOGD(op_type, "src_left_step_in=%d", params.src_left_step_in);
  OP_LOGD(op_type, "src_left_step_out=%d", params.src_left_step_out);
  OP_LOGD(op_type, "src_left_lp_step_in=%d", params.src_left_lp_step_in);
  OP_LOGD(op_type, "src_left_lp_step_out=%d", params.src_left_lp_step_out);
  OP_LOGD(op_type, "dst_r2nd_in_0_size=%d", params.dst_r2nd_in_0_size);
  OP_LOGD(op_type, "dst_r2nd_in_0_src_rsize=%d", params.dst_r2nd_in_0_src_rsize);
  OP_LOGD(op_type, "dst_r2nd_in_0_src_asize=%d", params.dst_r2nd_in_0_src_asize);
  OP_LOGD(op_type, "dst_r2nd_in_1_size=%d", params.dst_r2nd_in_1_size);
  OP_LOGD(op_type, "dst_r2nd_in_1_src_rsize=%d", params.dst_r2nd_in_1_src_rsize);
  OP_LOGD(op_type, "dst_r2nd_in_1_src_asize=%d", params.dst_r2nd_in_1_src_asize);
  OP_LOGD(op_type, "dst_r2nd_dims=%d", params.dst_r2nd_dims);
  OP_LOGD(op_type, "vnc_col_size=%d", params.vnc_col_size);
  OP_LOGD(op_type, "all_r2nd_in=%d", params.all_r2nd_in);
}
}  // namespace optiling