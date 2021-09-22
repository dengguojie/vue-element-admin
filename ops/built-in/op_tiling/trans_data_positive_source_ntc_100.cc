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
 * \file trans_data_positive_source_ntc_100.cc
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

namespace optiling
{

  const int32_t FRAME_LEVEL = 2;

  bool GetFullLpCnt(const int64_t &core_num, const int64_t &src_lp_cnt, int64_t &full_lp_cnt) {
    int64_t tmp_full_lp_cnt = GetFloorDiv(src_lp_cnt, core_num) > 0 ? core_num : 0;
    int64_t reminder_lp_cnt = src_lp_cnt % core_num;
    if (reminder_lp_cnt == 0)
    {
      tmp_full_lp_cnt += core_num;
    }
    full_lp_cnt = tmp_full_lp_cnt + reminder_lp_cnt;
    return true;
  }

  int64_t GetAxisIdx(std::string format, char axis) {
    size_t res_value = format.find(axis);
    if (res_value == std::string::npos) {
      res_value = 0;
      VECTOR_INNER_ERR_REPORT_TILIING("TransData", "Axis is not in format.");
    }
    return res_value;
  }


  bool GetMcInfoPositiveNtc100(const int64_t &src_cr_lp_cnt, const int64_t &src_cr_size, const int64_t &src_c_lp_cnt,
                               const int64_t &src_c_size, const int64_t &src_cl_lp_cnt, const int64_t &src_cl_size,
                               const int64_t &core_num, TransDataNtc100Param &params) {
    int64_t full_lp_cnt_cr = 0;
    int64_t full_lp_cnt_c = 0;
    int64_t full_lp_cnt_cl = 0;

    GetFullLpCnt(core_num, src_cr_lp_cnt, full_lp_cnt_cr);
    GetFullLpCnt(core_num, src_c_lp_cnt, full_lp_cnt_c);
    GetFullLpCnt(core_num, src_cl_lp_cnt, full_lp_cnt_cl);
    if (full_lp_cnt_cl >= full_lp_cnt_c && full_lp_cnt_cl >= full_lp_cnt_cr) {
      int64_t used_core_cnt = GetCeilDiv(src_cl_lp_cnt, GetCeilDiv(src_cl_lp_cnt, core_num));
      int64_t nlc_cl_lp_cnt = GetCeilDiv(src_cl_lp_cnt, used_core_cnt);
      int64_t lc_cl_lp_cnt = src_cl_lp_cnt - nlc_cl_lp_cnt * (used_core_cnt - 1);
      params.core_params.push_back(0);                                  // mc_pos
      params.core_params.push_back(used_core_cnt);                        // used_core_cnt
      params.core_params.push_back(nlc_cl_lp_cnt * params.src_cl_lp_step_in);  // core_step_in
      params.core_params.push_back(nlc_cl_lp_cnt * params.src_cl_lp_step_out); // core_step_out
      params.lc_params.push_back(nlc_cl_lp_cnt);                           // nlc_cl_lp_cnt
      params.lc_params.push_back(0);                                    // nlc_cl_left
      params.lc_params.push_back(src_c_lp_cnt);                            // nlc_c_lp_cnt
      params.lc_params.push_back(src_c_size % params.src_c_lp_unit);         // nlc_c_left
      params.lc_params.push_back(src_cr_lp_cnt);                           // nlc_cr_lp_cnt
      params.lc_params.push_back(src_cr_size % params.src_cr_lp_unit);       // nlc_cr_left
      params.lc_params.push_back(lc_cl_lp_cnt);                            // lc_cl_lp_cnt
      params.lc_params.push_back(src_cl_size % params.src_cl_lp_unit);       // lc_cl_left
      params.lc_params.push_back(src_c_lp_cnt);                            // lc_c_lp_cnt
      params.lc_params.push_back(src_c_size % params.src_c_lp_unit);         // lc_c_left
      params.lc_params.push_back(src_cr_lp_cnt);                           // lc_cr_lp_cnt
      params.lc_params.push_back(src_cr_size % params.src_cr_lp_unit);       // lc_cr_left
    } else if (full_lp_cnt_c >= full_lp_cnt_cr && full_lp_cnt_c >= full_lp_cnt_cl) {
      int64_t used_core_cnt = GetCeilDiv(src_c_lp_cnt, GetCeilDiv(src_c_lp_cnt, core_num));
      int64_t nlc_c_lp_cnt = GetCeilDiv(src_c_lp_cnt, used_core_cnt);
      int64_t lc_c_lp_cnt = src_c_lp_cnt - nlc_c_lp_cnt * (used_core_cnt - 1);
      params.core_params.push_back(1);                                // mc_pos
      params.core_params.push_back(used_core_cnt);                      // used_core_cnt
      params.core_params.push_back(nlc_c_lp_cnt * params.src_c_lp_step_in);  // core_step_in
      params.core_params.push_back(nlc_c_lp_cnt * params.src_c_lp_step_out); // core_step_out
      params.lc_params.push_back(src_cl_lp_cnt);                         // nlc_cl_lp_cnt
      params.lc_params.push_back(src_cl_size % params.src_cl_lp_unit);     // nlc_cl_left
      params.lc_params.push_back(nlc_c_lp_cnt);                          // nlc_c_lp_cnt
      params.lc_params.push_back(0);                                  // nlc_c_left
      params.lc_params.push_back(src_cr_lp_cnt);                         // nlc_cr_lp_cnt
      params.lc_params.push_back(src_cr_size % params.src_cr_lp_unit);     // nlc_cr_left
      params.lc_params.push_back(src_cl_lp_cnt);                         // lc_cl_lp_cnt
      params.lc_params.push_back(src_cl_size % params.src_cl_lp_unit);     // lc_cl_left
      params.lc_params.push_back(lc_c_lp_cnt);                           // lc_c_lp_cnt
      params.lc_params.push_back(src_c_size % params.src_c_lp_unit);       // lc_c_left
      params.lc_params.push_back(src_cr_lp_cnt);                         // lc_cr_lp_cnt
      params.lc_params.push_back(src_cr_size % params.src_cr_lp_unit);     // lc_cr_left
    } else if (full_lp_cnt_cr >= full_lp_cnt_c && full_lp_cnt_cr >= full_lp_cnt_cl) {
      int64_t used_core_cnt = GetCeilDiv(src_cr_lp_cnt, GetCeilDiv(src_cr_lp_cnt, core_num));
      int64_t nlc_cr_lp_cnt = GetCeilDiv(src_cr_lp_cnt, used_core_cnt);
      int64_t lc_cr_lp_cnt = src_cr_lp_cnt - nlc_cr_lp_cnt * (used_core_cnt - 1);
      params.core_params.push_back(2);                                  // mc_pos
      params.core_params.push_back(used_core_cnt);                        // used_core_cnt
      params.core_params.push_back(nlc_cr_lp_cnt * params.src_cr_lp_step_in);  // core_step_in
      params.core_params.push_back(nlc_cr_lp_cnt * params.src_cr_lp_step_out); // core_step_out
      params.lc_params.push_back(src_cl_lp_cnt);                           // nlc_cl_lp_cnt
      params.lc_params.push_back(src_cl_size % params.src_cl_lp_unit);       // nlc_cl_left
      params.lc_params.push_back(src_c_lp_cnt);                            // nlc_c_lp_cnt
      params.lc_params.push_back(src_c_size % params.src_c_lp_unit);         // nlc_c_left
      params.lc_params.push_back(nlc_cr_lp_cnt);                           // nlc_cr_lp_cnt
      params.lc_params.push_back(0);                                    // nlc_cr_left
      params.lc_params.push_back(src_cl_lp_cnt);                           // lc_cl_lp_cnt
      params.lc_params.push_back(src_cl_size % params.src_cl_lp_unit);       // lc_cl_left
      params.lc_params.push_back(src_c_lp_cnt);                            // lc_c_lp_cnt
      params.lc_params.push_back(src_c_size % params.src_c_lp_unit);         // lc_c_left
      params.lc_params.push_back(lc_cr_lp_cnt);                            // lc_cr_lp_cnt
      params.lc_params.push_back(src_cr_size % params.src_cr_lp_unit);       // lc_cr_left
    }
    return true;
  }

  bool RenewInputOutputShapeFormat(const std::vector<int64_t> &in_shape, const std::vector<int64_t> &out_shape,
                                   const ge::Format &src_format, const ge::Format &dst_format, const int64_t &c0_len,
                                   std::vector<int64_t> &in_shape_new, std::vector<int64_t> &out_shape_new,
                                   std::string &src_format_new, std::string &dst_format_new) {
    if (src_format == FORMAT_NCDHW && dst_format == FORMAT_NDC1HWC0) {
      if (in_shape.size() != 5) {
        VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
        return false;
      }
      src_format_new = "NCDH";
      dst_format_new = "NDCHT";
      in_shape_new.push_back(in_shape[0]);
      in_shape_new.push_back(in_shape[1]);
      in_shape_new.push_back(in_shape[2]);
      in_shape_new.push_back(in_shape[3] * in_shape[4]);
      int64_t c_idx = GetIdxFromFormat(C_IDX_MAP, src_format);
      int64_t axis_c1 = GetCeilDiv(in_shape[c_idx], c0_len);
      out_shape_new.push_back(in_shape[0]);
      out_shape_new.push_back(in_shape[2]);
      out_shape_new.push_back(axis_c1);
      out_shape_new.push_back(in_shape[3] * in_shape[4]);
      out_shape_new.push_back(c0_len);
    } else if (src_format == FORMAT_NCHW && dst_format == FORMAT_NC1HWC0) {
      if (in_shape.size() != 4) {
        VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
        return false;
      }
      src_format_new = "NCH";
      dst_format_new = "NCHT";
      in_shape_new.push_back(in_shape[0]);
      in_shape_new.push_back(in_shape[1]);
      in_shape_new.push_back(in_shape[2] * in_shape[3]);
      int64_t c_idx = GetIdxFromFormat(C_IDX_MAP, src_format);
      int64_t axis_c1 = GetCeilDiv(in_shape[c_idx], c0_len);
      out_shape_new.push_back(in_shape[0]);
      out_shape_new.push_back(axis_c1);
      out_shape_new.push_back(in_shape[2] * in_shape[3]);
      out_shape_new.push_back(c0_len);
    } else if (src_format == FORMAT_HWCN && dst_format == FORMAT_FRACTAL_Z) {
      if (in_shape.size() != 4) {
        VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
        return false;
      }
      src_format_new = "HCN";
      dst_format_new = "CHNT";
      in_shape_new.push_back(in_shape[0] * in_shape[1]);
      in_shape_new.push_back(in_shape[2]);
      in_shape_new.push_back(in_shape[3]);
      int64_t c_idx = GetIdxFromFormat(C_IDX_MAP, src_format);
      int64_t axis_c1 = GetCeilDiv(in_shape[c_idx], c0_len);
      int64_t n_idx = GetIdxFromFormat(N_IDX_MAP, src_format);
      int64_t axis_no = GetCeilDiv(in_shape[n_idx], NI_16);
      out_shape_new.push_back(axis_c1);
      out_shape_new.push_back(in_shape[0] * in_shape[1]);
      out_shape_new.push_back(NI_16 * axis_no);
      out_shape_new.push_back(c0_len);
    } else if (src_format == FORMAT_DHWCN && dst_format == FORMAT_FRACTAL_Z_3D) {
      if (in_shape.size() != 5) {
        VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
        return false;
      }
      src_format_new = "DHCN";
      dst_format_new = "DCHNT";
      in_shape_new.push_back(in_shape[0]);
      in_shape_new.push_back(in_shape[1] * in_shape[2]);
      in_shape_new.push_back(in_shape[3]);
      in_shape_new.push_back(in_shape[4]);
      int64_t c_idx = GetIdxFromFormat(C_IDX_MAP, src_format);
      int64_t axis_c1 = GetCeilDiv(in_shape[c_idx], c0_len);
      int64_t n_idx = GetIdxFromFormat(N_IDX_MAP, src_format);
      int64_t axis_no = GetCeilDiv(in_shape[n_idx], NI_16);
      out_shape_new.push_back(in_shape[0]);
      out_shape_new.push_back(axis_c1);
      out_shape_new.push_back(in_shape[1] * in_shape[2]);
      out_shape_new.push_back(NI_16 * axis_no);
      out_shape_new.push_back(c0_len);
    } else if (src_format == FORMAT_ND && dst_format == FORMAT_FRACTAL_Z) {
      int64_t axis_h = 1;
      int64_t axis_c = 1;
      int64_t axis_n = 1;
      if (in_shape.size() == 1) {
        axis_h = 1;
        axis_c = 1;
        axis_n = in_shape[0];
      } else if (in_shape.size() == 2) {
        axis_h = 1;
        axis_c = in_shape[0];
        axis_n = in_shape[1];
      } else {
        for (size_t i = 0; i < in_shape.size() - 2; i++) {
          axis_h *= in_shape[i];
        }
        axis_c = in_shape[in_shape.size() - 2];
        axis_n = in_shape[in_shape.size() - 1];
      }
      src_format_new = "HCN";
      dst_format_new = "HCNT";
      in_shape_new.push_back(axis_h);
      in_shape_new.push_back(axis_c);
      in_shape_new.push_back(axis_n);
      int64_t axis_c1 = GetCeilDiv(axis_c, c0_len);
      int64_t axis_no = GetCeilDiv(axis_n, NI_16);
      out_shape_new.push_back(axis_h);
      out_shape_new.push_back(axis_c1);
      out_shape_new.push_back(axis_no * NI_16);
      out_shape_new.push_back(c0_len);
    } else if (src_format == FORMAT_NCHW && dst_format == FORMAT_FRACTAL_Z) {
      if (in_shape.size() != 4) {
        VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
        return false;
      }
      src_format_new = "NCH";
      dst_format_new = "CHNT";
      in_shape_new.push_back(in_shape[0]);
      in_shape_new.push_back(in_shape[1]);
      in_shape_new.push_back(in_shape[2] * in_shape[3]);
      int64_t c_idx = GetIdxFromFormat(C_IDX_MAP, src_format);
      int64_t axis_c1 = GetCeilDiv(in_shape[c_idx], c0_len);
      int64_t n_idx = GetIdxFromFormat(N_IDX_MAP, src_format);
      int64_t axis_no = GetCeilDiv(in_shape[n_idx], NI_16);
      out_shape_new.push_back(axis_c1);
      out_shape_new.push_back(in_shape[2] * in_shape[3]);
      out_shape_new.push_back(NI_16 * axis_no);
      out_shape_new.push_back(c0_len);
    } else if (src_format == FORMAT_NCDHW && dst_format == FORMAT_FRACTAL_Z_3D) {
      if (in_shape.size() != 5) {
        VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
        return false;
      }
      src_format_new = "NCDH";
      dst_format_new = "DCHNT";
      in_shape_new.push_back(in_shape[0]);
      in_shape_new.push_back(in_shape[1]);
      in_shape_new.push_back(in_shape[2]);
      in_shape_new.push_back(in_shape[3] * in_shape[4]);
      int64_t c_idx = GetIdxFromFormat(C_IDX_MAP, src_format);
      int64_t axis_c1 = GetCeilDiv(in_shape[c_idx], c0_len);
      int64_t n_idx = GetIdxFromFormat(N_IDX_MAP, src_format);
      int64_t axis_no = GetCeilDiv(in_shape[n_idx], NI_16);
      out_shape_new.push_back(in_shape[2]);
      out_shape_new.push_back(axis_c1);
      out_shape_new.push_back(in_shape[3] * in_shape[4]);
      out_shape_new.push_back(NI_16 * axis_no);
      out_shape_new.push_back(c0_len);
    }
    return true;
  }

  bool TilingPositiveSourceNtc100(const vector<int64_t> &in_shape, const vector<int64_t> &out_shape,
                                  const ge::Format &src_format, const ge::Format &dst_format,
                                  const int64_t &core_num, const int64_t &block_elem_cnt, const int64_t &ub_size,
                                  const int64_t &c0_len, const DataType &dType, TransDataNtc100Param &params) {
    std::string src_format_new;
    std::string dst_format_new;
    std::vector<int64_t> in_shape_new;
    std::vector<int64_t> out_shape_new;
    RenewInputOutputShapeFormat(in_shape, out_shape, src_format, dst_format, c0_len,
                                in_shape_new, out_shape_new, src_format_new, dst_format_new);

    // get tiling params for using vnchwconv
    int64_t half_ub_size = c0_len == C0_16 ? ub_size / 2 : ub_size / 4;
    int64_t one_vnc_line_size = half_ub_size / VNC_LINES / block_elem_cnt * block_elem_cnt;
    int64_t tmp_ub_offset = one_vnc_line_size * VNC_LINES;
    params.ub_offset = c0_len == C0_16 ? tmp_ub_offset : tmp_ub_offset * 2;
    params.vnc_line_size = one_vnc_line_size;
    params.c0_size = c0_len;

    // axis c-right tiling parameters
    params.cr_dims = FRAME_LEVEL;
    params.r1st_src_r2nd_dst_same = 1;
    int64_t c_idx = GetAxisIdx(src_format_new, 'C');
    int64_t c1_idx = GetAxisIdx(dst_format_new, 'C');
    int64_t axis_src_cr_size = GetShapeSize(in_shape_new, c_idx + 1);
    int64_t tmp_src_cr_lp_unit = params.vnc_line_size / c0_len / block_elem_cnt * block_elem_cnt;
    const std::vector<DataType> dtype_list = {ge::DT_FLOAT, ge::DT_INT32, ge::DT_UINT32};
    if (axis_src_cr_size < 2 * block_elem_cnt || std::find(dtype_list.begin(), dtype_list.end(), dType) != dtype_list.end()) {
      params.tiling_mode = 1000;
      params.src_cr_lp_unit = axis_src_cr_size > tmp_src_cr_lp_unit ? tmp_src_cr_lp_unit : axis_src_cr_size;
    } else {
      params.tiling_mode = 1001;
      params.src_cr_lp_unit = axis_src_cr_size > params.vnc_line_size ? params.vnc_line_size : axis_src_cr_size;
    }

    // count method: cr_idx/dst_rsize%size*dst_asize
    std::string tmp_src_cr_format = src_format_new.substr(c_idx + 1);
    std::vector<int64_t> tmp_src_cr_shape;
    for (uint32_t i = 0; i < tmp_src_cr_format.length(); i++) {
      tmp_src_cr_shape.push_back(in_shape_new[i + c_idx + 1]);
    }
    tmp_src_cr_shape.push_back(1);
    std::reverse(tmp_src_cr_format.begin(), tmp_src_cr_format.end());
    for (uint32_t i = 0; i < tmp_src_cr_format.length(); i++) {
      int64_t tmp_src_idx = GetAxisIdx(src_format_new, tmp_src_cr_format[i]);
      int64_t tmp_dst_idx = GetAxisIdx(dst_format_new, tmp_src_cr_format[i]);
      if (i == 0) {
        params.cr_out_idx_0_size = in_shape_new[tmp_src_idx];
        params.cr_out_idx_0_dst_rsize = GetShapeSize(tmp_src_cr_shape, tmp_src_cr_shape.size() - i - 1);
        params.cr_out_idx_0_dst_asize = GetShapeSize(out_shape_new, tmp_dst_idx + 1);
      } else if (i == 1) {
        params.cr_out_idx_1_size = in_shape_new[tmp_src_idx];
        params.cr_out_idx_1_dst_rsize = GetShapeSize(tmp_src_cr_shape, tmp_src_cr_shape.size() - i - 1);
        params.cr_out_idx_1_dst_asize = GetShapeSize(out_shape_new, tmp_dst_idx + 1);
      }
    }

    // suppose there are 2 axises
    int64_t pad_axis_cnt = FRAME_LEVEL - tmp_src_cr_format.length();
    if (pad_axis_cnt) {
      params.cr_dims = 1;
      params.cr_out_idx_1_size = 1;
      params.cr_out_idx_1_dst_rsize = 1;
      params.cr_out_idx_1_dst_asize = 0;
    }
    if (*(src_format_new.rbegin()) != *(dst_format_new.rbegin() + 1)) {
      params.r1st_src_r2nd_dst_same = 0;
    }
    int64_t src_cr_lp_cnt = GetCeilDiv(axis_src_cr_size, params.src_cr_lp_unit);
    params.src_cr_step_in = 1;
    params.src_cr_lp_step_in = params.src_cr_step_in * params.src_cr_lp_unit;
    if (params.cr_dims == 2) {
      params.src_cr_step_out = 0;
      params.src_cr_lp_step_out = 0;
    } else {
      int64_t tmp_idx = std::strchr(dst_format_new.c_str(), *(src_format_new.rbegin())) - dst_format_new.c_str();
      params.src_cr_step_out = GetShapeSize(out_shape_new, tmp_idx + 1);
      params.src_cr_lp_step_out = params.src_cr_step_out * params.src_cr_lp_unit;
    }

    // axis c tiling parameters
    int64_t axis_src_c_size = in_shape_new[c_idx];
    params.src_c_lp_unit = c0_len;
    int64_t src_c_lp_cnt = GetCeilDiv(axis_src_c_size, params.src_c_lp_unit);
    params.src_c_step_in = GetShapeSize(in_shape_new, c_idx + 1);
    params.src_c_lp_step_in = params.src_c_step_in * params.src_c_lp_unit;
    params.src_c_lp_step_out = GetShapeSize(out_shape_new, c1_idx + 1);
    params.c_mod_c0 = axis_src_c_size % c0_len;

    // axis left parameters
    params.cl_dims = FRAME_LEVEL;
    int64_t axis_src_cl_size = GetShapeSize(in_shape_new, 0) / GetShapeSize(in_shape_new, c_idx);
    int64_t tmp_src_cl_lp_unit = 1;
    if (params.tiling_mode == 1000) {
      tmp_src_cl_lp_unit = NI_16;
    } else if (params.r1st_src_r2nd_dst_same == 0 && params.tiling_mode == 1001 && axis_src_cl_size > core_num) {
      tmp_src_cl_lp_unit = GetFloorDiv(params.vnc_line_size, GetCeilDiv(params.src_cr_lp_unit, c0_len) * c0_len);
    } else {
      tmp_src_cl_lp_unit = 1;
    }
    params.src_cl_lp_unit = axis_src_cl_size > tmp_src_cl_lp_unit ? tmp_src_cl_lp_unit : axis_src_cl_size;
    int64_t src_cl_lp_cnt = GetCeilDiv(axis_src_cl_size, params.src_cl_lp_unit);

    // count method: left_axis_size/dst_rsize%size*asize
    std::string tmp_src_cl_format = src_format_new.substr(0, c_idx);
    std::vector<int64_t> tmp_src_cl_shape;
    for (uint32_t i = 0; i < tmp_src_cl_format.length(); i++) {
      tmp_src_cl_shape.push_back(in_shape_new[i]);
    }
    tmp_src_cl_shape.push_back(1);
    std::reverse(tmp_src_cl_format.begin(), tmp_src_cl_format.end());
    for (uint32_t i = 0; i < tmp_src_cl_format.length(); i++) {
      int64_t tmp_src_cl_idx = GetAxisIdx(src_format_new, tmp_src_cl_format[i]);
      int64_t tmp_dst_cl_idx = GetAxisIdx(dst_format_new, tmp_src_cl_format[i]);
      if (i == 0) {
        params.cl_out_idx_0_size = in_shape_new[tmp_src_cl_idx];
        params.cl_out_idx_0_dst_rsize = GetShapeSize(tmp_src_cl_shape, tmp_src_cl_shape.size() - i - 1);
        params.cl_out_idx_0_dst_asize = GetShapeSize(out_shape_new, tmp_dst_cl_idx + 1);
      } else if (i == 1) {
        params.cl_out_idx_1_size = in_shape_new[tmp_src_cl_idx];
        params.cl_out_idx_1_dst_rsize = GetShapeSize(tmp_src_cl_shape, tmp_src_cl_shape.size() - i - 1);
        params.cl_out_idx_1_dst_asize = GetShapeSize(out_shape_new, tmp_dst_cl_idx + 1);
      }
    }

    // suppose there are 2 axises
    pad_axis_cnt = FRAME_LEVEL - tmp_src_cl_format.length();
    if (pad_axis_cnt) {
      params.cl_dims = 1;
      params.cl_out_idx_1_size = 1;
      params.cl_out_idx_1_dst_rsize = 1;
      params.cl_out_idx_1_dst_asize = 0;
    }
    params.src_cl_step_in = GetShapeSize(in_shape_new, c_idx);
    params.src_cl_lp_step_in = params.src_cl_step_in * params.src_cl_lp_unit;
    if (params.cl_dims == 2) {
      params.src_cl_step_out = 0;
      params.src_cl_lp_step_out = 0;
    } else {
      int64_t tmp_idx = GetAxisIdx(dst_format_new, src_format_new[0]);
      params.src_cl_step_out = GetShapeSize(out_shape_new, tmp_idx + 1);
      params.src_cl_lp_step_out = params.src_cl_step_out * params.src_cl_lp_unit;
    }

    // mulitple core parameters
    bool ret = GetMcInfoPositiveNtc100(src_cr_lp_cnt, axis_src_cr_size, src_c_lp_cnt, axis_src_c_size, src_cl_lp_cnt, axis_src_cl_size,
                                       core_num, params);
    if (!ret) {
      VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetMcInfoPositiveNtc100 Failed.");
      return ret;
    }

    return true;
  }

  void SetRunningNtc100Params(const TransDataNtc100Param &run_params, utils::OpRunInfo &run_info) {
    run_info.AddTilingData(run_params.tiling_mode);
    run_info.AddTilingData(run_params.ub_offset);
    for (auto i : run_params.core_params) {
      run_info.AddTilingData(int64_t(i));
    }
    run_info.AddTilingData(run_params.vnc_line_size);
    run_info.AddTilingData(run_params.c_mod_c0);
    run_info.AddTilingData(run_params.c0_size);
    run_info.AddTilingData(run_params.cl_dims);
    run_info.AddTilingData(run_params.cr_dims);
    run_info.AddTilingData(run_params.r1st_src_r2nd_dst_same);
    run_info.AddTilingData(run_params.src_cl_step_in);
    run_info.AddTilingData(run_params.src_cl_step_out);
    run_info.AddTilingData(run_params.src_cl_lp_unit);
    run_info.AddTilingData(run_params.src_cl_lp_step_in);
    run_info.AddTilingData(run_params.src_cl_lp_step_out);
    run_info.AddTilingData(run_params.src_c_step_in);
    run_info.AddTilingData(run_params.src_c_lp_unit);
    run_info.AddTilingData(run_params.src_c_lp_step_in);
    run_info.AddTilingData(run_params.src_c_lp_step_out);
    run_info.AddTilingData(run_params.src_cr_step_in);
    run_info.AddTilingData(run_params.src_cr_step_out);
    run_info.AddTilingData(run_params.src_cr_lp_unit);
    run_info.AddTilingData(run_params.src_cr_lp_step_in);
    run_info.AddTilingData(run_params.src_cr_lp_step_out);
    for (auto i : run_params.lc_params) {
      run_info.AddTilingData(int64_t(i));
    }
    run_info.AddTilingData(run_params.cl_out_idx_0_size);
    run_info.AddTilingData(run_params.cl_out_idx_0_dst_rsize);
    run_info.AddTilingData(run_params.cl_out_idx_0_dst_asize);
    run_info.AddTilingData(run_params.cl_out_idx_1_size);
    run_info.AddTilingData(run_params.cl_out_idx_1_dst_rsize);
    run_info.AddTilingData(run_params.cl_out_idx_1_dst_asize);
    run_info.AddTilingData(run_params.cr_out_idx_0_size);
    run_info.AddTilingData(run_params.cr_out_idx_0_dst_rsize);
    run_info.AddTilingData(run_params.cr_out_idx_0_dst_asize);
    run_info.AddTilingData(run_params.cr_out_idx_1_size);
    run_info.AddTilingData(run_params.cr_out_idx_1_dst_rsize);
    run_info.AddTilingData(run_params.cr_out_idx_1_dst_asize);
  }

} // namespace optiling
