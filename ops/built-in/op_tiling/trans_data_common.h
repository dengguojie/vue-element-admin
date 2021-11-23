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
 * \file trans_data_common.h
 * \brief dynamic TransData common function
 */

#ifndef CANN_OPS_BUILT_IN_OP_TILING_TRANSDATA_H_
#define CANN_OPS_BUILT_IN_OP_TILING_TRANSDATA_H_

#include <string>
#include <algorithm>

#include <nlohmann/json.hpp>
#include "op_tiling_util.h"
#include "../op_proto/util/error_util.h"
#include "op_log.h"

namespace optiling {
using namespace ge;

const vector<int64_t> PAD_IDX_LIST = {0, 1};

const std::map<ge::Format, int64_t> HW_IDX_MAP = {{FORMAT_NCHW, 2}, {FORMAT_NHWC, 1}, {FORMAT_NCDHW, 3},
                                                  {FORMAT_HWCN, 0}, {FORMAT_DHWCN, 1}, {FORMAT_NDHWC, 2}};
const std::map<ge::Format, int64_t> C_IDX_MAP = {{FORMAT_NCHW, 1}, {FORMAT_NHWC, 3}, {FORMAT_NCDHW, 1},
                                                 {FORMAT_HWCN, 2}, {FORMAT_DHWCN, 3}, {FORMAT_NDHWC, 4}};
const std::map<ge::Format, int64_t> N_IDX_MAP = {{FORMAT_NCHW, 0}, {FORMAT_NHWC, 0}, {FORMAT_NCDHW, 0},
                                                 {FORMAT_HWCN, 3}, {FORMAT_DHWCN, 4}, {FORMAT_NDHWC, 0}};

constexpr int64_t TILING_MODE_2001 = 2001;
constexpr int64_t TILING_MODE_2002 = 2002;
constexpr int64_t TILING_MODE_2003 = 2003;
constexpr int64_t TILING_MODE_2010 = 2010;
constexpr int64_t TILING_MODE_2011 = 2011;
constexpr int64_t TILING_MODE_2012 = 2012;
constexpr int64_t TILING_MODE_1000 = 1000;
constexpr int64_t TILING_MODE_1001 = 1001;
constexpr int64_t TILING_MODE_1010 = 1010;
constexpr int64_t TILING_MODE_1011 = 1011;

constexpr size_t SHAPE_LEN_2D = 2;
constexpr size_t SHAPE_LEN_4D = 4;
constexpr size_t SHAPE_LEN_5D = 5;
constexpr size_t SHAPE_LEN_6D = 6;

constexpr size_t SHAPE_LEN_CAPACITY_SIZE = 8;

constexpr size_t FORMAT_LEN_2D = 2;

struct HeadTilingParam {
  int64_t shape_loop_cnt;
};

struct TransDataMode100Param {
  int64_t tiling_mode;
  int64_t ub_offset;
  int64_t mc_flag;
  int64_t used_core_cnt;
  int64_t core_step_in;
  int64_t core_step_out;
  int64_t nlc_cr_lp_cnt;
  int64_t nlc_c_lp_cnt;
  int64_t nlc_left_lp_cnt;
  int64_t nlc_cr_left;
  int64_t nlc_c_left;
  int64_t lc_cr_lp_cnt;
  int64_t lc_c_lp_cnt;
  int64_t lc_left_lp_cnt;
  int64_t lc_cr_left;
  int64_t lc_c_left;

  int64_t src_cr_lp_unit;
  int64_t src_cr_lp_step_in;
  int64_t src_cr_lp_step_out;
  int64_t src_c_step_in;
  int64_t src_c_lp_unit;
  int64_t src_c_lp_step_in;
  int64_t src_c_lp_step_out;
  int64_t c_mod_c0;
  int64_t in_idx_0_size;
  int64_t in_idx_0_dst_rsize;
  int64_t in_idx_0_src_asize;
  int64_t in_idx_1_size;
  int64_t in_idx_1_dst_rsize;
  int64_t in_idx_1_src_asize;
  int64_t out_idx_0_size;
  int64_t out_idx_0_dst_rsize;
  int64_t out_idx_0_dst_asize;
  int64_t out_idx_1_size;
  int64_t out_idx_1_dst_rsize;
  int64_t out_idx_1_dst_asize;
  int64_t cr_out_idx_0_size;
  int64_t cr_out_idx_0_dst_rsize;
  int64_t cr_out_idx_0_dst_asize;
  int64_t cr_out_idx_1_size;
  int64_t cr_out_idx_1_dst_rsize;
  int64_t cr_out_idx_1_dst_asize;

  int64_t src_2_dst_flag;
  int64_t one_line_size;
};

struct TransDataMode1010Param {
  int64_t tiling_mode;
  int64_t ub_offset;
  int64_t used_core_cnt;
  int64_t core_step_in;
  int64_t core_step_out;
  int64_t dst_cl_lp_step_in;
  int64_t dst_cl_lp_step_out;
  int64_t dst_cl_step_in;
  int64_t dst_cl_step_out;
  int64_t dst_cr_lp_step_in;
  int64_t dst_cr_lp_step_out;
  int64_t dst_cr_step_in;
  int64_t nc_le_vcol;
  int64_t vnc_line_size;
  int64_t pln_dst_cl_size;
  int64_t pln_dst_cr_size;
  int64_t vnc_row_size;
  int64_t c_lp_step_in;
  int64_t c_lp_step_out;
  int64_t c_step_out;
  int64_t c0_size;
  int64_t c_mod_c0;
  int64_t c_lp_unit;
  int64_t nlc_dst_cl_lp_cnt;
  int64_t nlc_vnc_row_cl_left;
  int64_t nlc_last_line_cl_cnt;
  int64_t nlc_dst_cr_lp_cnt;
  int64_t nlc_vnc_row_left;
  int64_t nlc_last_line_cr_cnt;
  int64_t nlc_c_lp_cnt;
  int64_t nlc_c_left;
  int64_t lc_dst_cl_lp_cnt;
  int64_t lc_vnc_row_cl_left;
  int64_t lc_last_line_cl_cnt;
  int64_t lc_dst_cr_lp_cnt;
  int64_t lc_vnc_row_left;
  int64_t lc_last_line_cr_cnt;
  int64_t lc_c_lp_cnt;
  int64_t lc_c_left;
};

struct TransDataMode1011Param {
  int64_t tiling_mode;
  int64_t ub_offset;
  int64_t used_core_cnt;
  int64_t mc_on_cl;
  int64_t core_step_in;
  int64_t core_step_out;
  int64_t dst_r2nd_lp_step_in;
  int64_t dst_r2nd_lp_step_out;
  int64_t dst_r2nd_step_in;
  int64_t dst_r2nd_lp_unit;
  int64_t src_cl_lp_step_in;
  int64_t vnc_line_size;
  int64_t src_cl_lp_unit;
  int64_t src_cl_lp_step_out;
  int64_t c_lp_step_in;
  int64_t c_lp_step_out;
  int64_t c_step_out;
  int64_t c0_size;
  int64_t c_mod_c0;
  int64_t c_lp_unit;
  int64_t nlc_dst_r2nd_lp_cnt;
  int64_t nlc_dst_r2nd_left;
  int64_t nlc_src_cl_lp_cnt;
  int64_t nlc_src_cl_left;
  int64_t nlc_c_lp_cnt;
  int64_t nlc_c_left;
  int64_t lc_dst_r2nd_lp_cnt;
  int64_t lc_dst_r2nd_left;
  int64_t lc_src_cl_lp_cnt;
  int64_t lc_src_cl_left;
  int64_t lc_c_lp_cnt;
  int64_t lc_c_left;
  int64_t cl_out_0_size;
  int64_t cl_out_0_src_rsize;
  int64_t cl_out_0_dst_asize;
  int64_t cl_out_1_size;
  int64_t cl_out_1_src_rsize;
  int64_t cl_out_1_dst_asize;
};

struct TransDataNtc200Param {
  int64_t tiling_mode;
  int64_t ub_offset;
  int64_t mc_pos;
  int64_t used_core_cnt;
  int64_t c0_len;
  int64_t core_step_in;
  int64_t core_step_out;
  int64_t nlc_cr_lp_cnt;
  int64_t nlc_c_lp_cnt;
  int64_t nlc_cl_lp_cnt;
  int64_t nlc_cr_left;
  int64_t nlc_c_left;
  int64_t nlc_cl_left;
  int64_t lc_cr_lp_cnt;
  int64_t lc_c_lp_cnt;
  int64_t lc_cl_lp_cnt;
  int64_t lc_cr_left;
  int64_t lc_c_left;
  int64_t lc_cl_left;
  int64_t dst_cr_lp_unit;
  int64_t src_c_lp_unit;
  int64_t dst_cl_lp_unit;
  int64_t vnc_col_size;
  int64_t dst_cr_step_in;
  int64_t dst_cr_step_out;
  int64_t dst_cr_lp_step_in;
  int64_t dst_cr_lp_step_out;
  int64_t dst_c_size;
  int64_t src_c_step_in;
  int64_t src_c_step_out;
  int64_t src_c_lp_step_in;
  int64_t src_c_lp_step_out;
  int64_t dst_cr_all_in;
  int64_t dst_cl_step_in;
  int64_t dst_cl_step_out;
  int64_t dst_cl_lp_step_in;
  int64_t dst_cl_lp_step_out;
  int64_t c_mod_c0;
  int64_t dst_cr_dims;
  int64_t dst_cl_dims;
  int64_t is_mc_cr;
  int64_t is_mc_cl;
  int64_t src_r2nd_dst_r1st_same;
  int64_t left_cl_c_cr_size;

  int64_t cl_in_idx_0_size;
  int64_t cl_in_idx_0_dst_rsize;
  int64_t cl_in_idx_0_src_asize;
  int64_t cl_in_idx_1_size;
  int64_t cl_in_idx_1_dst_rsize;
  int64_t cl_in_idx_1_src_asize;
  int64_t cr_in_idx_0_size;
  int64_t cr_in_idx_0_dst_rsize;
  int64_t cr_in_idx_0_src_asize;
  int64_t cr_in_idx_1_size;
  int64_t cr_in_idx_1_dst_rsize;
  int64_t cr_in_idx_1_src_asize;
};


struct TransDataTc201Param {
  int64_t tiling_mode;
  int64_t ub_offset;
  int64_t mc_pos;
  int64_t used_core_cnt;
  int64_t src_r2nd_dst_r2nd_same;
  int64_t c0_len;
  int64_t core_step_in;
  int64_t core_step_out;
  int64_t nlc_dst_r2nd_lp_cnt;
  int64_t nlc_src_cl_lp_cnt;
  int64_t nlc_src_left_lp_cnt;
  int64_t nlc_dst_r2nd_left;
  int64_t nlc_src_cl_left;
  int64_t nlc_src_left_left;
  int64_t lc_dst_r2nd_lp_cnt;
  int64_t lc_src_cl_lp_cnt;
  int64_t lc_src_left_lp_cnt;
  int64_t lc_dst_r2nd_left;
  int64_t lc_src_cl_left;
  int64_t lc_src_left_left;
  int64_t dst_r2nd_lp_unit;
  int64_t dst_r2nd_step_in;
  int64_t dst_r2nd_step_out;
  int64_t dst_r2nd_lp_step_in;
  int64_t dst_r2nd_lp_step_out;
  int64_t src_cl_lp_unit;
  int64_t all_c_in;
  int64_t src_cl_step_in;
  int64_t src_cl_step_out;
  int64_t src_cl_lp_step_in;
  int64_t src_cl_lp_step_out;
  int64_t c_mod_c0;
  int64_t src_left_lp_unit;
  int64_t src_left_step_in;
  int64_t src_left_step_out;
  int64_t src_left_lp_step_in;
  int64_t src_left_lp_step_out;
  int64_t dst_r2nd_in_0_size;
  int64_t dst_r2nd_in_0_src_rsize;
  int64_t dst_r2nd_in_0_src_asize;
  int64_t dst_r2nd_in_1_size;
  int64_t dst_r2nd_in_1_src_rsize;
  int64_t dst_r2nd_in_1_src_asize;
  int64_t dst_r2nd_dims;
  int64_t vnc_col_size;
  int64_t all_r2nd_in;
};

struct TransDataNtc100Param {
  int64_t tiling_mode;
  int64_t ub_offset;
  /**
   * mc_pos, used_core_cnt, core_step_in, core_step_out
   **/
  std::vector<int64_t> core_params;
  int64_t vnc_line_size;
  int64_t c_mod_c0;
  int64_t c0_size;
  int64_t cl_dims;
  int64_t cr_dims;
  int64_t r1st_src_r2nd_dst_same;
  int64_t src_cl_step_in;
  int64_t src_cl_step_out;
  int64_t src_cl_lp_unit;
  int64_t src_cl_lp_step_in;
  int64_t src_cl_lp_step_out;
  int64_t src_c_step_in;
  int64_t src_c_lp_unit;
  int64_t src_c_lp_step_in;
  int64_t src_c_lp_step_out;
  int64_t src_cr_step_in;
  int64_t src_cr_step_out;
  int64_t src_cr_lp_unit;
  int64_t src_cr_lp_step_in;
  int64_t src_cr_lp_step_out;
  /**
   * nlc_cl_lp_cnt, nlc_cl_left, nlc_c_lp_cnt,nlc_c_left,nlc_cr_lp_cnt, nlc_cr_left,
   * lc_cl_lp_cnt, cl_cl_left, lc_c_lp_cnt,lc_c_left, lc_cr_lp_cnt,lc_cr_left
   **/
  std::vector<int64_t> lc_params;
  int64_t cl_out_idx_0_size;
  int64_t cl_out_idx_0_dst_rsize;
  int64_t cl_out_idx_0_dst_asize;
  int64_t cl_out_idx_1_size;
  int64_t cl_out_idx_1_dst_rsize;
  int64_t cl_out_idx_1_dst_asize;
  int64_t cr_out_idx_0_size;
  int64_t cr_out_idx_0_dst_rsize;
  int64_t cr_out_idx_0_dst_asize;
  int64_t cr_out_idx_1_size;
  int64_t cr_out_idx_1_dst_rsize;
  int64_t cr_out_idx_1_dst_asize;

  std::string to_string() const {
    std::string result = "tiling_mode:" + std::to_string(tiling_mode);
    result += " ub_offset:" + std::to_string(ub_offset);
    result += " mc_pos:" + std::to_string(core_params[0]);
    result += " used_core_cnt:" + std::to_string(core_params[1]);
    result += " core_step_in:" + std::to_string(core_params[2]);
    result += " core_step_out:" + std::to_string(core_params[3]);
    result += " vnc_line_size:" + std::to_string(vnc_line_size);
    result += " c_mod_c0:" + std::to_string(c_mod_c0);
    result += " c0_size:" + std::to_string(c0_size);
    result += " cl_dims:" + std::to_string(cl_dims);
    result += " cr_dims:" + std::to_string(cr_dims);
    result += " r1st_src_r2nd_dst_same:" + std::to_string(r1st_src_r2nd_dst_same);
    result += " src_cl_step_in:" + std::to_string(src_cl_step_in);
    result += " src_cl_step_out:" + std::to_string(src_cl_step_out);
    result += " src_cl_lp_unit:" + std::to_string(src_cl_lp_unit);
    result += " src_cl_lp_step_in:" + std::to_string(src_cl_lp_step_in);
    result += " src_cl_lp_step_out:" + std::to_string(src_cl_lp_step_out);
    result += " src_c_step_in:" + std::to_string(src_c_step_in);
    result += " src_c_lp_unit:" + std::to_string(src_c_lp_unit);
    result += " src_c_lp_step_in:" + std::to_string(src_c_lp_step_in);
    result += " src_c_lp_step_out:" + std::to_string(src_c_lp_step_out);
    result += " src_cr_step_in:" + std::to_string(src_cr_step_in);
    result += " src_cr_step_out:" + std::to_string(src_cr_step_out);
    result += " src_cr_lp_unit:" + std::to_string(src_cr_lp_unit);
    result += " src_cr_lp_step_in:" + std::to_string(src_cr_lp_step_in);
    result += " src_cr_lp_step_out:" + std::to_string(src_cr_lp_step_out);
    result += " nlc_cl_lp_cnt:" + std::to_string(lc_params[0]);
    result += " nlc_cl_left:" + std::to_string(lc_params[1]);
    result += " nlc_c_lp_cnt:" + std::to_string(lc_params[2]);
    result += " nlc_c_left:" + std::to_string(lc_params[3]);
    result += " nlc_cr_lp_cnt:" + std::to_string(lc_params[4]);
    result += " nlc_cr_left:" + std::to_string(lc_params[5]);
    result += " lc_cl_lp_cnt:" + std::to_string(lc_params[6]);
    result += " cl_cl_left:" + std::to_string(lc_params[7]);
    result += " lc_c_lp_cnt:" + std::to_string(lc_params[8]);
    result += " lc_c_left:" + std::to_string(lc_params[9]);
    result += " lc_cr_lp_cnt:" + std::to_string(lc_params[10]);
    result += " lc_cr_left:" + std::to_string(lc_params[11]);
    result += " cl_out_idx_0_size:" + std::to_string(cl_out_idx_0_size);
    result += " cl_out_idx_0_dst_rsize:" + std::to_string(cl_out_idx_0_dst_rsize);
    result += " cl_out_idx_0_dst_asize:" + std::to_string(cl_out_idx_0_dst_asize);
    result += " cl_out_idx_1_size:" + std::to_string(cl_out_idx_1_size);
    result += " cl_out_idx_1_dst_rsize:" + std::to_string(cl_out_idx_1_dst_rsize);
    result += " cl_out_idx_1_dst_asize:" + std::to_string(cl_out_idx_1_dst_asize);
    result += " cr_out_idx_0_size:" + std::to_string(cr_out_idx_0_size);
    result += " cr_out_idx_0_dst_rsize:" + std::to_string(cr_out_idx_0_dst_rsize);
    result += " cr_out_idx_0_dst_asize:" + std::to_string(cr_out_idx_0_dst_asize);
    result += " cr_out_idx_1_size:" + std::to_string(cr_out_idx_1_size);
    result += " cr_out_idx_1_dst_rsize:" + std::to_string(cr_out_idx_1_dst_rsize);
    result += " cr_out_idx_1_dst_asize:" + std::to_string(cr_out_idx_1_dst_asize);
    return result;
  }
};

static int64_t GetFloorDiv(int64_t u_value, int64_t d_value) {
  int64_t res_value = 0;
  if (d_value == 0) {
    return u_value;
  }

  res_value = u_value / d_value;

  return res_value;
}

static int64_t GetCeilDiv(int64_t u_value, int64_t d_value) {
  int64_t res_value = 0;
  if (d_value == 0) {
    return u_value;
  }

  res_value = (u_value + d_value - 1) / d_value;

  return res_value;
}

static int64_t GetIdxFromFormat(const std::map<ge::Format, int64_t>& format_map, const ge::Format data_format) {
  auto find_format_it = format_map.find(data_format);
  if (find_format_it != format_map.end()) {
    return find_format_it->second;
  }

  return -1;
}

static int64_t GetShapeSize(std::vector<int64_t> in_shape, int32_t pos) {
  int32_t n = in_shape.size();
  int64_t shape_size = 1;
  if (pos < 0) {
    pos = n + pos;
  }
  for (int32_t i = pos; i < n; i++) {
    shape_size *= in_shape[i];
  }
  return shape_size;
}

bool TillingPositiveMode1010(vector<int64_t>& in_shape, vector<int64_t>& out_shape, std::string& src_format,
                            std::string& dst_format, int64_t& core_num, int64_t& block_elem_cnt,
                            int64_t& ub_size, TransDataMode1010Param& params);

bool TillingPositiveMode1011(vector<int64_t>& in_shape, vector<int64_t>& out_shape, std::string& src_format,
                            std::string& dst_format, int64_t& core_num, int64_t& block_elem_cnt,
                            int64_t& ub_size, TransDataMode1011Param& params);

bool TilingNegativeNtc200(vector<int64_t>& in_shape, vector<int64_t>& out_shape, std::string& src_format,
                            std::string& dst_format, int64_t& core_num, int64_t& block_elem_cnt, DataType& dtype,
                            int64_t ub_size, int64_t& vnc_fp32_flag, TransDataNtc200Param& params);

bool TilingNegativeTc201(vector<int64_t>& in_shape, vector<int64_t>& out_shape, std::string& src_format,
                            std::string& dst_format, int64_t& core_num, int64_t& block_elem_cnt, DataType& dtype,
                            int64_t& ub_size, TransDataTc201Param& params);

bool TilingPositiveSourceNtc100(const vector<int64_t>& in_shape, const vector<int64_t>& out_shape,
                                      const ge::Format& src_format, const ge::Format& dst_format,
                                      const int64_t& core_num, const int64_t& block_elem_cnt, const int64_t& ub_size,
                                      const int64_t& c0_len, const DataType& dtype, TransDataNtc100Param& params);

void SetRunningMode1010Params(const TransDataMode1010Param& run_params, utils::OpRunInfo& run_info);
void SetRunningMode1011Params(const TransDataMode1011Param& run_params, utils::OpRunInfo& run_info);
void SetRunningNtc200Params(const TransDataNtc200Param& run_params, utils::OpRunInfo& run_info);
void SetRunningTc201Params(const TransDataTc201Param& run_params, utils::OpRunInfo& run_info);
void SetRunningNtc100Params(const TransDataNtc100Param& run_params, utils::OpRunInfo& run_info);

void PrintTilingMode1010Params(const std::string& op_type, const TransDataMode1010Param& params);
void PrintTilingMode1011Params(const std::string& op_type, const TransDataMode1011Param& params);
void PrintTilingModeNtc200Params(const std::string& op_type, const TransDataNtc200Param& params);
void PrintTilingModeTc201Params(const std::string& op_type, const TransDataTc201Param& params);
void PrintTilingNtc100Params(const std::string& op_type, const TransDataNtc100Param& params);

}  // namespace optiling

#endif  // CANN_OPS_BUILT_IN_OP_TILING_TRANSDATA_H_
