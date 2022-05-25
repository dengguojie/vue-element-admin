/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "trans_data.h"

using namespace gert;
namespace optiling {
namespace transdata {
static const int32_t NTC_FRAME_LEVEL = 2;

void GetMcInfoNegative200(int64_t dst_cr_lp_cnt, int64_t dst_cr_left, int64_t src_c_lp_cnt,
                          int64_t src_c_left, int64_t dst_cl_lp_cnt, int64_t dst_cl_left,
                          int64_t core_num, TransDataNtc200Param& params) {
  int64_t tmp_full_loop_cnt_cr;
  if (ge::FloorDiv(dst_cr_lp_cnt, core_num) > 0) {
    tmp_full_loop_cnt_cr = core_num;
  } else {
    tmp_full_loop_cnt_cr = 0;
  }
  int64_t reminder_loop_cnt_cr = dst_cr_lp_cnt % core_num;
  if (reminder_loop_cnt_cr == 0 && dst_cr_left > params.dst_cr_lp_unit / TRANSDATA_TILING_FACTOR_2) {
    tmp_full_loop_cnt_cr += core_num;
  }
  int64_t full_loop_cnt_cr = tmp_full_loop_cnt_cr + reminder_loop_cnt_cr;

  int64_t tmp_full_loop_cnt_c;
  if (ge::FloorDiv(src_c_lp_cnt, core_num) > 0) {
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
  if (ge::FloorDiv(dst_cl_lp_cnt, core_num) > 0) {
    tmp_full_loop_cnt_cl = core_num;
  } else {
    tmp_full_loop_cnt_cl = 0;
  }
  int64_t reminder_loop_cnt_cl = dst_cl_lp_cnt % core_num;
  if (reminder_loop_cnt_cl == 0) {
    tmp_full_loop_cnt_cl += core_num;
  }
  int64_t full_loop_cnt_cl = tmp_full_loop_cnt_cl + reminder_loop_cnt_cl;
  auto max_value = std::max(std::max(full_loop_cnt_cl, full_loop_cnt_c), full_loop_cnt_cr);
  if (max_value == full_loop_cnt_cl) {
    params.mc_pos = 0;
    params.is_mc_cl = 1;
    params.is_mc_cr = 0;
    params.used_core_cnt = ge::CeilDiv(dst_cl_lp_cnt, ge::CeilDiv(dst_cl_lp_cnt, core_num));
    params.nlc_cl_lp_cnt = ge::CeilDiv(dst_cl_lp_cnt, params.used_core_cnt);
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
  } else if (max_value == full_loop_cnt_c) {
    params.mc_pos = 1;
    params.is_mc_cl = 0;
    params.is_mc_cr = 0;
    params.used_core_cnt = ge::CeilDiv(src_c_lp_cnt, ge::CeilDiv(src_c_lp_cnt, core_num));
    params.nlc_c_lp_cnt = ge::CeilDiv(src_c_lp_cnt, params.used_core_cnt);
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
    params.mc_pos = TRANSDATA_TILING_PARAM_2;
    params.is_mc_cl = 0;
    params.is_mc_cr = 1;
    params.used_core_cnt = ge::CeilDiv(dst_cr_lp_cnt, ge::CeilDiv(dst_cr_lp_cnt, core_num));
    params.nlc_cr_lp_cnt = ge::CeilDiv(dst_cr_lp_cnt, params.used_core_cnt);
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
}

ge::graphStatus TilingNegativeNtc200(TilingContext* context, const gert::Shape& in_shape, const gert::Shape& out_shape,
                                     const RealFormat& src_format, const RealFormat& dst_format,
                                     int64_t core_num, int64_t block_elem_cnt, int64_t ub_size,
                                     ge::DataType dtype, int64_t vnc_fp32_flag) {
  auto params = context->GetTilingData<TransDataNtc200Param>();
  OPS_CHECK_NULL_WITH_CONTEXT(context, params);
  int64_t c0_len = in_shape[in_shape.GetDimNum() - 1];
  params->c0_len = c0_len;

  if (GetAxisType(src_format, DIM_IDX_NEG_TWO) == GetAxisType(dst_format, -1)) {
    params->src_r2nd_dst_r1st_same = 1;
  } else {
    params->src_r2nd_dst_r1st_same = 0;
  }
  params->ub_offset = ub_size / TRANSDATA_TILING_FACTOR_2 / block_elem_cnt * block_elem_cnt;
  int64_t vnc_col_block_size = ge::FloorDiv(params->ub_offset / VNC_LINES, block_elem_cnt);
  if (vnc_col_block_size % TRANSDATA_TILING_FACTOR_2 == 0) {
    vnc_col_block_size -= 1;
  }
  int64_t vnc_col_size = vnc_col_block_size * block_elem_cnt;
  params->vnc_col_size = vnc_col_size;

  // dst axis C-RIGHT tiling parameters
  params->dst_cr_dims = TRANSDATA_TILING_PARAM_2;
  int32_t out_shape_dims = out_shape.GetDimNum();
  int32_t src_axis_pos_c = GetAxisIndex(src_format, RAT_C);
  int32_t dst_axis_pos_c = GetAxisIndex(dst_format, RAT_C);
  int64_t axis_dst_cr_size = GetShapeSize(out_shape, dst_axis_pos_c + 1);
  int64_t axis_src_c_size = in_shape[src_axis_pos_c];
  int64_t cr_per_vnc_line = params->ub_offset / c0_len / c0_len * c0_len;
  // once vnchwconv flow
  int64_t tmp_dst_cr_lp_unit;
  int64_t cr_gate;
  if (axis_dst_cr_size % c0_len == 0) {
    cr_gate = TRANSDATA_TILING_FACTOR_2 * VNC_LINES;
  } else if (ge::FloorDiv(cr_per_vnc_line, CeilAlign(axis_dst_cr_size, c0_len)) <= axis_src_c_size) {
    cr_gate = TRANSDATA_TILING_FACTOR_8 * VNC_LINES;
  } else {
    cr_gate = TRANSDATA_TILING_FACTOR_15 * VNC_LINES;
  }

  if ((dtype == ge::DT_FLOAT16 || dtype == ge::DT_INT8 || dtype == ge::DT_UINT8 ||
       ((dtype == ge::DT_FLOAT || dtype == ge::DT_INT32 || dtype == ge::DT_UINT32) && vnc_fp32_flag == 1)) &&
      (axis_dst_cr_size >= cr_gate)) {
    tmp_dst_cr_lp_unit = params->ub_offset / c0_len / c0_len * c0_len;
  } else {
    // twice vnchwconv flow
    if (dtype == ge::DT_INT8 || dtype == ge::DT_UINT8) {
      tmp_dst_cr_lp_unit = vnc_col_size / TRANSDATA_TILING_FACTOR_2 / c0_len / block_elem_cnt * block_elem_cnt;
    } else {
      tmp_dst_cr_lp_unit = vnc_col_size / c0_len / block_elem_cnt * block_elem_cnt;
    }
  }

  params->dst_cr_lp_unit = axis_dst_cr_size > tmp_dst_cr_lp_unit ? tmp_dst_cr_lp_unit : axis_dst_cr_size;
  int64_t dst_cr_lp_cnt = ge::CeilDiv(axis_dst_cr_size, params->dst_cr_lp_unit);
  int64_t dst_cr_left = axis_dst_cr_size % params->dst_cr_lp_unit;

  int32_t tmp_dst_cr_format_len = out_shape.GetDimNum() - dst_axis_pos_c - 1;
  gert::Shape tmp_dst_cr_shape;
  for (int32_t i = dst_axis_pos_c + 1; i < out_shape_dims; i++) {
    tmp_dst_cr_shape.AppendDim(out_shape[i]);
  }
  tmp_dst_cr_shape.AppendDim(1);
  for (int32_t i = 0; i < tmp_dst_cr_format_len; i++) {
    RealAxisType chr = GetAxisType(dst_format, out_shape.GetDimNum() - i - 1);
    int32_t src_chr_pos = GetAxisIndex(src_format, chr);
    int32_t dst_chr_pos = GetAxisIndex(dst_format, chr);
    if (i == 0) {
      params->cr_in_idx_0_size = out_shape[dst_chr_pos];
      params->cr_in_idx_0_dst_rsize = GetShapeSize(tmp_dst_cr_shape, -1 - i);
      params->cr_in_idx_0_src_asize = GetShapeSize(in_shape, src_chr_pos + 1);
    } else if (i == 1) {
      params->cr_in_idx_1_size = out_shape[dst_chr_pos];
      params->cr_in_idx_1_dst_rsize = GetShapeSize(tmp_dst_cr_shape, -1 - i);
      params->cr_in_idx_1_src_asize = GetShapeSize(in_shape, src_chr_pos + 1);
    }
  }

  // suppose there are 2 axises
  int32_t pad_axis_cnt = NTC_FRAME_LEVEL - tmp_dst_cr_format_len;
  if (pad_axis_cnt != 0) {
    params->dst_cr_dims = 1;
    if (tmp_dst_cr_format_len == 0) {
      params->cr_in_idx_0_size = 1;
      params->cr_in_idx_0_dst_rsize = 1;
      params->cr_in_idx_0_src_asize = 0;
      params->cr_in_idx_1_size = 1;
      params->cr_in_idx_1_dst_rsize = 1;
      params->cr_in_idx_1_src_asize = 0;
    } else if (tmp_dst_cr_format_len == 1) {
      params->cr_in_idx_1_size = 1;
      params->cr_in_idx_1_dst_rsize = 1;
      params->cr_in_idx_1_src_asize = 0;
    }
  }
  params->dst_cr_step_out = 1;
  params->dst_cr_lp_step_out = params->dst_cr_lp_unit * params->dst_cr_step_out;
  if (params->dst_cr_dims == SHAPE_LEN_2D) {
    params->dst_cr_step_in = 0;
  } else {
    RealAxisType dst_cr_chr = GetAxisType(dst_format, -1);
    int32_t dst_cr_in_src = GetAxisIndex(src_format, dst_cr_chr);
    params->dst_cr_step_in = GetShapeSize(in_shape, dst_cr_in_src + 1);
  }
  params->dst_cr_lp_step_in = params->dst_cr_lp_unit * params->dst_cr_step_in;
  params->dst_cr_all_in = dst_cr_lp_cnt == 1 ? 1 : 0;

  // axis C tiling parameters
  int64_t axis_dst_c_size = out_shape[dst_axis_pos_c];
  int64_t tmp_src_c_lp_unit;
  if (dst_cr_lp_cnt > 1 || axis_src_c_size == 1) {
    tmp_src_c_lp_unit = 1;
  } else if ((dtype == DT_FLOAT16 || dtype == DT_INT8 || dtype == DT_UINT8 ||
              ((dtype == DT_FLOAT || dtype == DT_INT32 || dtype == DT_UINT32) && vnc_fp32_flag == 1)) &&
             (axis_dst_cr_size >= cr_gate)) {
    tmp_src_c_lp_unit = tmp_dst_cr_lp_unit / CeilAlign(params->dst_cr_lp_unit, c0_len);
  } else {
    tmp_src_c_lp_unit = tmp_dst_cr_lp_unit / CeilAlign(params->dst_cr_lp_unit, block_elem_cnt);
  }

  params->src_c_lp_unit = axis_src_c_size > tmp_src_c_lp_unit ? tmp_src_c_lp_unit : axis_src_c_size;
  int64_t src_c_lp_cnt = ge::CeilDiv(axis_src_c_size, params->src_c_lp_unit);
  int64_t src_c_left = axis_src_c_size % params->src_c_lp_unit;
  params->src_c_step_in = GetShapeSize(in_shape, src_axis_pos_c + 1);
  params->src_c_step_out = GetShapeSize(out_shape, dst_axis_pos_c + 1);
  params->src_c_lp_step_in = params->src_c_lp_unit * params->src_c_step_in;
  params->src_c_lp_step_out = params->src_c_lp_unit * c0_len * params->src_c_step_out;
  params->c_mod_c0 = axis_dst_c_size % c0_len;
  params->dst_c_size = axis_dst_c_size;

  // dst axis C-LEFT tiling parameters
  params->dst_cl_dims = TRANSDATA_TILING_PARAM_2;
  int64_t axis_dst_cl_size = 1;
  for (int32_t i = 0; i < dst_axis_pos_c; i++) {
    axis_dst_cl_size *= out_shape[i];
  }
  int64_t dst_c_dst_cr_size = axis_dst_c_size * axis_dst_cr_size;
  int64_t tmp_dst_cl_lp_unit;
  if ((dtype == DT_FLOAT16 || dtype == DT_INT8 || dtype == DT_UINT8 ||
       ((dtype == DT_FLOAT || dtype == DT_INT32 || dtype == DT_UINT32) && vnc_fp32_flag == 1)) &&
      (axis_dst_cr_size >= cr_gate)) {
    params->tiling_mode = TILING_MODE_2001;
    tmp_dst_cl_lp_unit =
        params->ub_offset / (params->src_c_lp_unit * CeilAlign(params->dst_cr_lp_unit, c0_len) * c0_len);
    params->dst_cl_lp_unit = axis_dst_cl_size > tmp_dst_cl_lp_unit ? tmp_dst_cl_lp_unit : axis_dst_cl_size;
  } else if (dst_c_dst_cr_size < TRANSDATA_TILING_FACTOR_54 * block_elem_cnt && dst_cr_lp_cnt == 1 &&
             src_c_lp_cnt == 1) {
    params->tiling_mode = TILING_MODE_2003;
    int64_t supposed_lp_unit = TRANSDATA_TILING_FACTOR_4 * block_elem_cnt;
    tmp_dst_cl_lp_unit = tmp_dst_cr_lp_unit / (params->src_c_lp_unit * params->dst_cr_lp_unit);
    params->dst_cl_lp_unit = tmp_dst_cl_lp_unit > supposed_lp_unit ? supposed_lp_unit : tmp_dst_cl_lp_unit;
  } else {
    params->tiling_mode = TILING_MODE_2002;
    params->dst_cl_lp_unit = axis_dst_cl_size > VNC_LINES ? VNC_LINES : axis_dst_cl_size;
  }
  int64_t dst_cl_lp_cnt = ge::CeilDiv(axis_dst_cl_size, params->dst_cl_lp_unit);
  int64_t dst_cl_left = axis_dst_cl_size % params->dst_cl_lp_unit;

  // for tiling mode 2003
  params->left_cl_c_cr_size = dst_cl_left * axis_dst_c_size * axis_dst_cr_size;
  int32_t tmp_dst_cl_format_len = dst_axis_pos_c;
  gert::Shape tmp_c_left_shape;
  for (int32_t i = 0; i < dst_axis_pos_c; i++) {
    tmp_c_left_shape.AppendDim(out_shape[i]);
  }
  tmp_c_left_shape.AppendDim(1);

  for (int32_t i = 0; i < tmp_dst_cl_format_len; i++) {
    RealAxisType chr = GetAxisType(dst_format, dst_axis_pos_c - i - 1);
    int32_t src_chr_pos = GetAxisIndex(src_format, chr);
    int32_t dst_chr_pos = GetAxisIndex(dst_format, chr);
    if (i == 0) {
      params->cl_in_idx_0_size = out_shape[dst_chr_pos];
      params->cl_in_idx_0_dst_rsize = GetShapeSize(tmp_c_left_shape, -1 - i);
      params->cl_in_idx_0_src_asize = GetShapeSize(in_shape, src_chr_pos + 1);
    } else if (i == 1) {
      params->cl_in_idx_1_size = out_shape[dst_chr_pos];
      params->cl_in_idx_1_dst_rsize = GetShapeSize(tmp_c_left_shape, -1 - i);
      params->cl_in_idx_1_src_asize = GetShapeSize(in_shape, src_chr_pos + 1);
    }
  }
  // suppose there are 2 axises
  pad_axis_cnt = NTC_FRAME_LEVEL - tmp_dst_cl_format_len;
  if (pad_axis_cnt != 0) {
    params->dst_cl_dims = 1;
    if (tmp_dst_cl_format_len == 0) {
      params->cl_in_idx_0_size = 1;
      params->cl_in_idx_0_dst_rsize = 1;
      params->cl_in_idx_0_src_asize = 0;
      params->cl_in_idx_1_size = 1;
      params->cl_in_idx_1_dst_rsize = 1;
      params->cl_in_idx_1_src_asize = 0;
    } else if (tmp_dst_cl_format_len == 1) {
      params->cl_in_idx_1_size = 1;
      params->cl_in_idx_1_dst_rsize = 1;
      params->cl_in_idx_1_src_asize = 0;
    }
  }

  params->dst_cl_step_out = GetShapeSize(out_shape, dst_axis_pos_c);
  params->dst_cl_lp_step_out = params->dst_cl_lp_unit * params->dst_cl_step_out;
  if (params->dst_cl_dims == TRANSDATA_TILING_PARAM_2) {
    params->dst_cl_step_in = 0;
  } else {
    RealAxisType dst_cl_chr = GetAxisType(dst_format, 0);
    params->dst_cl_step_in = GetShapeSize(in_shape, GetAxisIndex(src_format, dst_cl_chr) + 1);
  }
  params->dst_cl_lp_step_in = params->dst_cl_lp_unit * params->dst_cl_step_in;

  GetMcInfoNegative200(dst_cr_lp_cnt, dst_cr_left, src_c_lp_cnt, src_c_left, dst_cl_lp_cnt, dst_cl_left, core_num,
                       *params);

  return ge::GRAPH_SUCCESS;
}
}  // namespace transdata
}  // namespace optiling
