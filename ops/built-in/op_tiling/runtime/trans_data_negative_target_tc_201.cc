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
static const int32_t TC_FRAME_LEVEL = 2;

void GetMcInfoNegative201(int64_t dst_r2nd_lp_cnt, int64_t dst_r2nd_left, int64_t src_cl_lp_cnt, int64_t src_cl_left,
                          int64_t src_left_lp_cnt, int64_t src_left_left, int64_t core_num,
                          TransDataTc201Param& params) {
  int64_t tmp_full_loop_cnt_r2nd;
  if (ge::FloorDiv(dst_r2nd_lp_cnt, core_num) > 0) {
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
  if (ge::FloorDiv(src_cl_lp_cnt, core_num) > 0) {
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
  if (ge::FloorDiv(src_left_lp_cnt, core_num) > 0) {
    tmp_full_loop_cnt_left = core_num;
  } else {
    tmp_full_loop_cnt_left = 0;
  }
  int64_t reminder_loop_cnt_left = src_left_lp_cnt % core_num;
  if (reminder_loop_cnt_left == 0) {
    tmp_full_loop_cnt_left += core_num;
  }
  int64_t full_loop_cnt_left = tmp_full_loop_cnt_left + reminder_loop_cnt_left;
  auto max_value = std::max(std::max(full_loop_cnt_left, full_loop_cnt_c1), full_loop_cnt_r2nd);
  if (max_value == full_loop_cnt_left) {
    params.mc_pos = 0;
    params.used_core_cnt = ge::CeilDiv(src_left_lp_cnt, ge::CeilDiv(src_left_lp_cnt, core_num));
    params.nlc_src_left_lp_cnt = ge::CeilDiv(src_left_lp_cnt, params.used_core_cnt);
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
  } else if (max_value == full_loop_cnt_c1) {
    params.mc_pos = 1;
    params.used_core_cnt = ge::CeilDiv(src_cl_lp_cnt, ge::CeilDiv(src_cl_lp_cnt, core_num));
    params.nlc_src_cl_lp_cnt = ge::CeilDiv(src_cl_lp_cnt, params.used_core_cnt);
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
    params.used_core_cnt = ge::CeilDiv(dst_r2nd_lp_cnt, ge::CeilDiv(dst_r2nd_lp_cnt, core_num));
    params.nlc_dst_r2nd_lp_cnt = ge::CeilDiv(dst_r2nd_lp_cnt, params.used_core_cnt);
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
}

ge::graphStatus TilingNegativeTc201(TilingContext* context, const gert::Shape& in_shape, const gert::Shape& out_shape,
                                    const RealFormat& src_format, const RealFormat& dst_format, int64_t core_num,
                                    int64_t block_elem_cnt, int64_t ub_size, ge::DataType dtype) {
  auto params = context->GetTilingData<TransDataTc201Param>();
  OPS_CHECK_NULL_WITH_CONTEXT(context, params);
  int64_t c0_len = in_shape[in_shape.GetDimNum() - 1];
  params->c0_len = c0_len;

  int32_t src_axis_pos_c = GetAxisIndex(src_format, RAT_C);
  int32_t dst_axis_pos_c = GetAxisIndex(dst_format, RAT_C);
  int64_t axis_dst_c_size = out_shape[dst_axis_pos_c];
  int64_t axis_src_c1_size = in_shape[src_axis_pos_c];

  gert::Shape dst_r2nd_shape{};
  int64_t axis_dst_r2nd_size;
  int64_t axis_src_left_size;
  RealAxisType src_left_format = RAT_END;
  int32_t dst_r2nd_format_len;
  int32_t src_format_c_pos = GetAxisIndex(src_format, RAT_C);
  if (GetAxisType(src_format, DIM_IDX_NEG_TWO) == GetAxisType(dst_format, DIM_IDX_NEG_TWO)) {
    params->src_r2nd_dst_r2nd_same = 1;
    dst_r2nd_format_len = 1;
    dst_r2nd_shape.AppendDim(out_shape[out_shape.GetDimNum() - DIM_NUM_2]);
    axis_dst_r2nd_size = out_shape[out_shape.GetDimNum() - DIM_NUM_2];
    src_left_format = GetAxisType(src_format, 0);
    axis_src_left_size = out_shape[GetAxisIndex(dst_format, GetAxisType(src_format, 0))];
  } else {
    params->src_r2nd_dst_r2nd_same = 0;
    src_left_format = GetAxisType(src_format, DIM_IDX_NEG_TWO);
    axis_src_left_size = out_shape[GetAxisIndex(dst_format, GetAxisType(src_format, DIM_IDX_NEG_TWO))];
    dst_r2nd_format_len = in_shape.GetDimNum() - DIM_NUM_2;
    axis_dst_r2nd_size = 1;
    for (int32_t i = 0; i < dst_r2nd_format_len; i++) {
      if (i == src_format_c_pos) {
        continue;
      }
      RealAxisType chr = GetAxisType(src_format, i);
      int32_t src_chr_pos = GetAxisIndex(src_format, chr);
      axis_dst_r2nd_size *= in_shape[src_chr_pos];
      dst_r2nd_shape.AppendDim(in_shape[src_chr_pos]);
    }
  }
  dst_r2nd_shape.AppendDim(1);

  // output ub offset
  params->ub_offset = ub_size / TRANSDATA_TILING_FACTOR_2 / block_elem_cnt * block_elem_cnt;
  // axis c1 tiling parameters
  int64_t vnc_col_block_cnt = ge::FloorDiv(params->ub_offset / VNC_LINES, block_elem_cnt);
  if (vnc_col_block_cnt % TRANSDATA_TILING_FACTOR_2 == 0) {
    vnc_col_block_cnt -= 1;
  }
  int64_t vnc_col_size = vnc_col_block_cnt * block_elem_cnt;
  params->vnc_col_size = vnc_col_size;
  int64_t tmp_src_cl_lp_unit;
  int64_t c_gate = 0;
  if (GetRemainder(axis_dst_c_size, params->c0_len) == 0) {
    c_gate = TRANSDATA_TILING_FACTOR_16 * params->c0_len;
  } else {
    c_gate = TRANSDATA_TILING_FACTOR_56 * params->c0_len;
  }

  if (axis_src_c1_size * c0_len >= c_gate || axis_dst_c_size == c0_len) {
    params->tiling_mode = TILING_MODE_2010;
    if (axis_dst_r2nd_size < NI_16) {
      tmp_src_cl_lp_unit = ge::FloorDiv(params->ub_offset, axis_dst_r2nd_size * params->c0_len);
    } else {
      tmp_src_cl_lp_unit = ge::FloorDiv(params->ub_offset, NI_16 * params->c0_len);
    }
  } else if (dtype != ge::DT_INT8 && dtype != ge::DT_UINT8) {
    if (axis_dst_c_size * axis_dst_r2nd_size >= vnc_col_size / VNC_LINES) {
      params->tiling_mode = TILING_MODE_2011;
    } else {
      params->tiling_mode = TILING_MODE_2012;
    }
    tmp_src_cl_lp_unit = vnc_col_size / c0_len / block_elem_cnt * block_elem_cnt;
  } else {
    if (axis_dst_c_size * axis_dst_r2nd_size >= vnc_col_size / TRANSDATA_TILING_FACTOR_2 / VNC_LINES) {
      params->tiling_mode = TILING_MODE_2011;
    } else {
      params->tiling_mode = TILING_MODE_2012;
    }
    tmp_src_cl_lp_unit = vnc_col_size / TRANSDATA_TILING_FACTOR_2 / c0_len / block_elem_cnt * block_elem_cnt;
  }

  params->src_cl_lp_unit = axis_src_c1_size > tmp_src_cl_lp_unit ? tmp_src_cl_lp_unit : axis_src_c1_size;
  int64_t src_cl_lp_cnt = ge::CeilDiv(axis_src_c1_size, params->src_cl_lp_unit);
  int64_t src_cl_left = GetRemainder(axis_src_c1_size, params->src_cl_lp_unit);
  params->src_cl_lp_step_in = params->src_cl_lp_unit * GetShapeSize(in_shape, src_axis_pos_c + 1);
  params->src_cl_lp_step_out = params->src_cl_lp_unit * c0_len;
  params->src_cl_step_in = GetShapeSize(in_shape, src_axis_pos_c + 1);
  params->src_cl_step_out = 1;
  params->c_mod_c0 = GetRemainder(axis_dst_c_size, c0_len);
  if (src_cl_lp_cnt == 1) {
    params->all_c_in = 1;
  } else {
    params->all_c_in = 0;
  }

  // axis -2 tiling parameters
  params->dst_r2nd_dims = TRANSDATA_TILING_PARAM_2;
  int64_t tmp_dst_r2nd_lp_unit;
  int64_t dtype_factor = 1;
  // to make sure the rep_stride of vor is less than limit
  if (params->tiling_mode == TILING_MODE_2010) {
    int64_t max_r2nd_lp_size = TRANSDATA_TILING_PARAM_63;
    if (dtype == ge::DT_FLOAT || dtype == ge::DT_INT32 || dtype == ge::DT_UINT32) {
      if (axis_dst_c_size == params->c0_len && axis_src_left_size <= C0_16) {
        // for vor in copy data in
        max_r2nd_lp_size = TRANSDATA_TILING_PARAM_63;
      } else {
        // for vor in reorder
        max_r2nd_lp_size = TRANSDATA_TILING_PARAM_31;
      }
      dtype_factor = TRANSDATA_TILING_FACTOR_2;
    } else if (axis_dst_c_size == params->c0_len && axis_src_left_size <= C0_16) {
      max_r2nd_lp_size = TRANSDATA_TILING_PARAM_127;
    }
    tmp_dst_r2nd_lp_unit = ge::FloorDiv(params->ub_offset, params->src_cl_lp_unit * c0_len);
    if (tmp_dst_r2nd_lp_unit > max_r2nd_lp_size) {
      tmp_dst_r2nd_lp_unit = max_r2nd_lp_size;
    }
  } else if (dtype != ge::DT_INT8 && dtype != ge::DT_UINT8) {
    tmp_dst_r2nd_lp_unit = vnc_col_size / (params->src_cl_lp_unit * c0_len);
  } else {
    tmp_dst_r2nd_lp_unit = vnc_col_size / TRANSDATA_TILING_FACTOR_2 / (params->src_cl_lp_unit * c0_len);
  }
  params->dst_r2nd_lp_unit = axis_dst_r2nd_size > tmp_dst_r2nd_lp_unit ? tmp_dst_r2nd_lp_unit : axis_dst_r2nd_size;
  int64_t r2nd_c_mod_block = params->dst_r2nd_lp_unit * axis_dst_c_size % block_elem_cnt;
  if (params->tiling_mode == TILING_MODE_2011 && r2nd_c_mod_block > 0 &&
      axis_dst_r2nd_size > params->dst_r2nd_lp_unit && params->dst_r2nd_lp_unit > block_elem_cnt) {
    params->dst_r2nd_lp_unit = ge::FloorDiv(params->dst_r2nd_lp_unit, block_elem_cnt) * block_elem_cnt;
  }
  // to avoid bank conflict
  if (params->tiling_mode == TILING_MODE_2010 && params->dst_r2nd_lp_unit * dtype_factor % NI_16 == 0 &&
      (params->dst_r2nd_lp_unit < params->src_cl_lp_unit || params->src_cl_lp_unit * dtype_factor % NI_16 == 0)) {
    params->dst_r2nd_lp_unit -= 1;
  }
  int64_t dst_r2nd_lp_cnt = ge::CeilDiv(axis_dst_r2nd_size, params->dst_r2nd_lp_unit);
  int64_t dst_r2nd_left = GetRemainder(axis_dst_r2nd_size, params->dst_r2nd_lp_unit);
  if (dst_r2nd_lp_cnt == 1) {
    params->all_r2nd_in = 1;
  } else {
    params->all_r2nd_in = 0;
  }
  if (GetAxisType(src_format, DIM_IDX_NEG_TWO) == GetAxisType(dst_format, DIM_IDX_NEG_TWO)) {
    int32_t src_chr_pos = GetAxisIndex(src_format, GetAxisType(dst_format, DIM_IDX_NEG_TWO));
    params->dst_r2nd_in_0_size = in_shape[src_chr_pos];
    params->dst_r2nd_in_0_src_rsize = GetShapeSize(dst_r2nd_shape, -1);
    params->dst_r2nd_in_0_src_asize = GetShapeSize(in_shape, src_chr_pos + 1);
  } else {
    int32_t excute_i = -1;
    int32_t format_len_for = dst_r2nd_format_len;
    for (int32_t i = 0; i < format_len_for; i++) {
      if (i == src_format_c_pos) {
        dst_r2nd_format_len = dst_r2nd_format_len - 1;
        continue;
      }
      excute_i = excute_i + 1;
      RealAxisType dst_r2nd_format_i = GetAxisType(src_format, format_len_for - i - 1);
      int32_t src_chr_pos = GetAxisIndex(src_format, dst_r2nd_format_i);
      if (excute_i == 0) {
        params->dst_r2nd_in_0_size = in_shape[src_chr_pos];
        params->dst_r2nd_in_0_src_rsize = GetShapeSize(dst_r2nd_shape, -1);
        params->dst_r2nd_in_0_src_asize = GetShapeSize(in_shape, src_chr_pos + 1);
      } else if (excute_i == 1) {
        params->dst_r2nd_in_1_size = in_shape[src_chr_pos];
        params->dst_r2nd_in_1_src_rsize = GetShapeSize(dst_r2nd_shape, DIM_IDX_NEG_TWO);
        params->dst_r2nd_in_1_src_asize = GetShapeSize(in_shape, src_chr_pos + 1);
      }
    }
  }

  int32_t pad_axis_cnt = TC_FRAME_LEVEL - dst_r2nd_format_len;
  if (pad_axis_cnt != 0) {
    params->dst_r2nd_dims = 1;
    if (dst_r2nd_format_len == 0) {
      params->dst_r2nd_in_0_size = 1;
      params->dst_r2nd_in_0_src_rsize = 1;
      params->dst_r2nd_in_0_src_asize = 0;
      params->dst_r2nd_in_1_size = 1;
      params->dst_r2nd_in_1_src_rsize = 1;
      params->dst_r2nd_in_1_src_asize = 0;
    } else if (dst_r2nd_format_len == 1) {
      params->dst_r2nd_in_1_size = 1;
      params->dst_r2nd_in_1_src_rsize = 1;
      params->dst_r2nd_in_1_src_asize = 0;
    }
  }

  if (params->dst_r2nd_dims == TRANSDATA_TILING_PARAM_2) {
    params->dst_r2nd_step_in = 0;
  } else {
    params->dst_r2nd_step_in = c0_len;
  }
  params->dst_r2nd_lp_step_in = params->dst_r2nd_lp_unit * params->dst_r2nd_step_in;
  params->dst_r2nd_step_out = axis_dst_c_size;
  params->dst_r2nd_lp_step_out = params->dst_r2nd_lp_unit * params->dst_r2nd_step_out;

  int64_t tmp_src_left_lp_unit;
  if (params->tiling_mode == TILING_MODE_2010) {
    tmp_src_left_lp_unit = params->ub_offset / (params->src_cl_lp_unit * params->dst_r2nd_lp_unit * c0_len);
    if (tmp_src_left_lp_unit > axis_src_left_size / core_num && axis_src_left_size >= core_num) {
      tmp_src_left_lp_unit = axis_src_left_size / core_num;
    }
  } else if (dtype != ge::DT_INT8 && dtype != ge::DT_UINT8) {
    tmp_src_left_lp_unit = vnc_col_size / (params->src_cl_lp_unit * params->dst_r2nd_lp_unit * c0_len);
  } else {
    tmp_src_left_lp_unit =
        vnc_col_size / TRANSDATA_TILING_FACTOR_2 / (params->src_cl_lp_unit * params->dst_r2nd_lp_unit * c0_len);
  }
  if (params->tiling_mode == TILING_MODE_2011) {
    tmp_src_left_lp_unit = NI_16;
  }
  params->src_left_lp_unit = axis_src_left_size > tmp_src_left_lp_unit ? tmp_src_left_lp_unit : axis_src_left_size;
  int64_t left_r2nd_c_mod_block =
      params->src_left_lp_unit * params->dst_r2nd_lp_unit * axis_dst_c_size % block_elem_cnt;
  if (params->tiling_mode == TILING_MODE_2012 && left_r2nd_c_mod_block > 0 &&
      axis_src_left_size > params->src_left_lp_unit && params->src_left_lp_unit > block_elem_cnt) {
    params->src_left_lp_unit = ge::FloorDiv(params->src_left_lp_unit, block_elem_cnt) * block_elem_cnt;
  }
  int64_t src_left_lp_cnt = ge::CeilDiv(axis_src_left_size, params->src_left_lp_unit);
  int64_t src_left_left = GetRemainder(axis_src_left_size, params->src_left_lp_unit);
  params->src_left_step_in = GetShapeSize(in_shape, GetAxisIndex(src_format, src_left_format) + 1);
  params->src_left_lp_step_in = params->src_left_lp_unit * params->src_left_step_in;
  params->src_left_step_out = GetShapeSize(out_shape, GetAxisIndex(dst_format, src_left_format) + 1);
  params->src_left_lp_step_out = params->src_left_lp_unit * params->src_left_step_out;

  GetMcInfoNegative201(dst_r2nd_lp_cnt, dst_r2nd_left, src_cl_lp_cnt, src_cl_left, src_left_lp_cnt, src_left_left,
                       core_num, *params);
  return ge::SUCCESS;
}
}  // namespace transdata
}  // namespace optiling
