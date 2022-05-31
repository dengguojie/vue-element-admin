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
const int64_t FOUR_IN_CORE_CNT = 4;
const int64_t PLN_CL_GATE = 64;
void GetMcInfoPositive1010(int64_t dst_cl_lp_cnt, int64_t vnc_row_cl_left, int64_t ll_dst_cl_left, int64_t c_lp_cnt,
                           int64_t c_left, int64_t dst_cr_lp_cnt, int64_t vnc_row_left, int64_t ll_dst_cr_left,
                           int64_t core_num, TransDataMode1010Param& params) {
  int64_t tmp_full_loop_cnt_cr = ge::FloorDiv(dst_cr_lp_cnt, core_num) > 0 ? core_num : 0;
  int64_t reminder_loop_cnt_cr = GetRemainder(dst_cr_lp_cnt, core_num);
  if (reminder_loop_cnt_cr == 0) {
    tmp_full_loop_cnt_cr += core_num;
  }
  int64_t full_loop_cnt_cr = tmp_full_loop_cnt_cr + reminder_loop_cnt_cr;

  int64_t tmp_full_loop_cnt_c = ge::FloorDiv(c_lp_cnt, core_num) > 0 ? core_num : 0;

  int64_t reminder_loop_cnt_c = GetRemainder(c_lp_cnt, core_num);
  if (reminder_loop_cnt_c == 0) {
    tmp_full_loop_cnt_c += core_num;
  }
  int64_t full_loop_cnt_c = tmp_full_loop_cnt_c + reminder_loop_cnt_c;

  int64_t tmp_full_loop_cnt_left = ge::FloorDiv(dst_cl_lp_cnt, core_num) > 0 ? core_num : 0;
  int64_t reminder_loop_cnt_left = GetRemainder(dst_cl_lp_cnt, core_num);
  if (reminder_loop_cnt_left == 0) {
    tmp_full_loop_cnt_left += core_num;
  }
  int64_t full_loop_cnt_left = tmp_full_loop_cnt_left + reminder_loop_cnt_left;

  auto max_value = std::max(std::max(full_loop_cnt_left, full_loop_cnt_cr), full_loop_cnt_c);
  if (max_value == full_loop_cnt_left) {
    params.used_core_cnt = ge::CeilDiv(dst_cl_lp_cnt, ge::CeilDiv(dst_cl_lp_cnt, core_num));
    params.nlc_dst_cl_lp_cnt = ge::CeilDiv(dst_cl_lp_cnt, params.used_core_cnt);
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
  } else if (max_value == full_loop_cnt_cr) {
    params.used_core_cnt = ge::CeilDiv(dst_cr_lp_cnt, ge::CeilDiv(dst_cr_lp_cnt, core_num));
    params.nlc_dst_cr_lp_cnt = ge::CeilDiv(dst_cr_lp_cnt, params.used_core_cnt);
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
    params.used_core_cnt = ge::CeilDiv(c_lp_cnt, ge::CeilDiv(c_lp_cnt, core_num));
    params.nlc_c_lp_cnt = ge::CeilDiv(c_lp_cnt, params.used_core_cnt);
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
    half_ub_size = ub_size / TRANSDATA_TILING_FACTOR_2;
  } else {
    half_ub_size = ub_size / TRANSDATA_TILING_FACTOR_4;
  }
  params.vnc_line_size = half_ub_size / VNC_LINES / block_elem_cnt * block_elem_cnt;
  int64_t tmp_ub_offset = params.vnc_line_size * VNC_LINES;
  if (c0_len == C0_16) {
    params.ub_offset = tmp_ub_offset;
  } else {
    params.ub_offset = tmp_ub_offset * TRANSDATA_TILING_FACTOR_2;
  }
  params.c_mod_c0 = GetRemainder(axis_c_size, c0_len);
  params.c0_size = c0_len;
}

ge::graphStatus TillingPositiveMode1010(TilingContext* context, const gert::Shape& in_shape,
                                        const gert::Shape& out_shape, const RealSrcDstFormat* real_formats,
                                        const TransDataCompileInfo* compile_info) {
  auto params = context->GetTilingData<TransDataMode1010Param>();
  OPS_CHECK_NULL_WITH_CONTEXT(context, params);
  auto src_td = context->GetInputDesc(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, src_td);
  auto dtype = src_td->GetDataType();
  int64_t block_elem_cnt = BLOCK_BYTE_SIZE / ge::GetSizeByDataType(dtype);
  RealFormat src_format = real_formats->src;
  RealFormat dst_format = real_formats->dst;
  int64_t axis_c_size = in_shape[in_shape.GetDimNum() - 1];
  int64_t c0_len = out_shape[out_shape.GetDimNum() - 1];
  GetCommonParam(compile_info->ub_size, block_elem_cnt, c0_len, axis_c_size, *params);
  params->tiling_mode = TILING_MODE_1010;
  params->vnc_line_size = params->vnc_line_size / c0_len * c0_len;

  // source axis c tiling parameters
  int32_t dst_axis_pos_c = GetAxisIndex(dst_format, RAT_C);
  if (axis_c_size < params->vnc_line_size) {
    params->c_lp_unit = axis_c_size;
  } else {
    params->c_lp_unit = params->vnc_line_size;
  }
  params->c_lp_step_in = params->c_lp_unit;
  int64_t lp_c1_cnt = ge::CeilDiv(params->c_lp_unit, c0_len);
  params->c_lp_step_out = lp_c1_cnt * GetShapeSize(out_shape, dst_axis_pos_c + 1);
  params->c_step_out = GetShapeSize(out_shape, dst_axis_pos_c + 1);
  int64_t c_lp_cnt = ge::CeilDiv(axis_c_size, params->c_lp_unit);
  int64_t c_left = GetRemainder(axis_c_size, params->c_lp_unit);

  // target axis c-right tiling parameters
  int64_t axis_dst_cl_size = 1;
  for (int32_t i = 0; i < dst_axis_pos_c; i++) {
    axis_dst_cl_size *= out_shape[i];
  }

  auto tmp_src_pos = GetAxisIndex(src_format, GetAxisType(dst_format, -2));
  int64_t axis_dst_cr_size = GetShapeSize(in_shape, tmp_src_pos) / in_shape[in_shape.GetDimNum() - 1];
  params->pln_dst_cr_size = params->vnc_line_size / CeilAlign(params->c_lp_unit, c0_len);
  params->vnc_row_size = VNC_LINES;
  int64_t per_vnc_dst_cr_cnt = params->pln_dst_cr_size * params->vnc_row_size;
  if (per_vnc_dst_cr_cnt >= axis_dst_cr_size && compile_info->block_dim > 1 && axis_dst_cl_size == 1) {
    int64_t new_vnc_lines = ge::CeilDiv(axis_dst_cr_size, params->pln_dst_cr_size);
    if (new_vnc_lines > VNC_LINES) {
      new_vnc_lines = VNC_LINES;
    }
    int64_t vnc_per_core =
        new_vnc_lines > compile_info->block_dim ? ge::CeilDiv(new_vnc_lines, compile_info->block_dim) : 1;
    params->vnc_row_size = vnc_per_core;
    per_vnc_dst_cr_cnt = params->pln_dst_cr_size * params->vnc_row_size;
  }
  int64_t dst_cr_lp_cnt = ge::CeilDiv(axis_dst_cr_size, per_vnc_dst_cr_cnt);
  int64_t dst_cr_left = GetRemainder(axis_dst_cr_size, per_vnc_dst_cr_cnt);
  int64_t vnc_row_left = ge::CeilDiv(dst_cr_left, params->pln_dst_cr_size);
  int64_t tmp_dst_cr_left = GetRemainder(dst_cr_left, params->pln_dst_cr_size);
  int64_t ll_dst_cr_left;
  if (tmp_dst_cr_left > 0) {
    ll_dst_cr_left = tmp_dst_cr_left;
  } else {
    ll_dst_cr_left = params->pln_dst_cr_size;
  }

  params->dst_cr_lp_step_in = in_shape[in_shape.GetDimNum() - 1] * per_vnc_dst_cr_cnt;
  int32_t tmp_dst_pos = GetAxisIndex(dst_format, GetAxisType(src_format, -2));
  params->dst_cr_lp_step_out = GetShapeSize(out_shape, tmp_dst_pos + 1) * per_vnc_dst_cr_cnt;
  params->dst_cr_step_in = GetShapeSize(in_shape, -1);

  // target axis c-left tiling parameters
  int64_t per_vnc_dst_cl_cnt = 1;
  int64_t dst_cl_lp_cnt = 1;
  int64_t dst_cl_left = 0;
  int64_t vnc_row_cl_left = 0;
  int64_t tmp_dst_cl_left = 0;
  int64_t ll_dst_cl_left = 0;
  RealAxisType dst_cl_char = GetAxisType(dst_format, dst_axis_pos_c - 1);

  if ((axis_c_size % c0_len == 0 && ge::CeilDiv(params->c_lp_unit, block_elem_cnt) % C0_16 != 0) ||
      (axis_c_size % c0_len == 0 && params->pln_dst_cr_size % TRANSDATA_TILING_FACTOR_2 == 0)) {
    // move in cl_cr_c in together
    if (params->c_lp_unit == axis_c_size && per_vnc_dst_cr_cnt >= axis_dst_cr_size) {
      params->nc_le_vcol = TRANSDATA_TILING_PARAM_3;
      per_vnc_dst_cl_cnt = ge::FloorDiv(params->vnc_line_size * VNC_LINES, axis_c_size * axis_dst_cr_size);
    } else if (params->c_lp_unit == axis_c_size) {
      // move in cr_c in together
      params->nc_le_vcol = TRANSDATA_TILING_PARAM_4;
      per_vnc_dst_cl_cnt = 1;
    } else {
      // move in c
      params->nc_le_vcol = TRANSDATA_TILING_PARAM_5;
      per_vnc_dst_cl_cnt = 1;
    }
    params->pln_dst_cl_size = per_vnc_dst_cl_cnt;
    dst_cl_lp_cnt = ge::CeilDiv(axis_dst_cl_size, params->pln_dst_cl_size);
    vnc_row_cl_left = GetRemainder(axis_dst_cl_size, params->pln_dst_cl_size);
    ll_dst_cl_left = GetRemainder(axis_dst_cl_size, params->pln_dst_cl_size);
  } else if (dst_cr_lp_cnt == 1 && params->c_lp_unit == axis_c_size &&
             vnc_row_left <= ge::FloorDiv(VNC_LINES, TRANSDATA_TILING_FACTOR_2)) {
    // nc is less than vnchwconv col size
    if (vnc_row_left == 1) {
      params->nc_le_vcol = 1;
      params->pln_dst_cl_size = ge::FloorDiv(params->pln_dst_cr_size, axis_dst_cr_size);
    } else {
      params->nc_le_vcol = TRANSDATA_TILING_PARAM_2;
      params->pln_dst_cl_size = 1;
      // adjust c-right parameters
      dst_cr_lp_cnt = ge::CeilDiv(axis_dst_cr_size, params->pln_dst_cr_size);
      vnc_row_left = GetRemainder(axis_dst_cr_size, params->pln_dst_cr_size);
      if (vnc_row_left > 0) {
        ll_dst_cr_left = vnc_row_left;
      } else {
        ll_dst_cr_left = params->pln_dst_cr_size;
      }
      params->dst_cr_lp_step_in = in_shape[in_shape.GetDimNum() - 1] * params->pln_dst_cr_size;
      params->dst_cr_lp_step_out = GetShapeSize(out_shape, tmp_dst_pos + 1) * params->pln_dst_cr_size;
    }
    per_vnc_dst_cl_cnt = params->pln_dst_cl_size * params->vnc_row_size;
    dst_cl_lp_cnt = ge::CeilDiv(axis_dst_cl_size, per_vnc_dst_cl_cnt);
    // adjust c-left parameters
    if ((ge::FloorDiv(compile_info->block_dim, FOUR_IN_CORE_CNT) > dst_cl_lp_cnt) &&
        (params->pln_dst_cl_size > PLN_CL_GATE)) {
      params->pln_dst_cl_size = ge::FloorDiv(params->pln_dst_cl_size, PLN_CL_GATE);
      per_vnc_dst_cl_cnt = params->pln_dst_cl_size * params->vnc_row_size;
      dst_cl_lp_cnt = ge::CeilDiv(axis_dst_cl_size, per_vnc_dst_cl_cnt);
    }
    dst_cl_left = GetRemainder(axis_dst_cl_size, per_vnc_dst_cl_cnt);
    vnc_row_cl_left = ge::CeilDiv(dst_cl_left, params->pln_dst_cl_size);

    tmp_dst_cl_left = GetRemainder(dst_cl_left, params->pln_dst_cl_size);
    if (tmp_dst_cl_left > 0) {
      ll_dst_cl_left = tmp_dst_cl_left;
    } else {
      ll_dst_cl_left = params->pln_dst_cl_size;
    }
  } else {
    params->nc_le_vcol = 0;
    params->pln_dst_cl_size = 1;
    dst_cl_lp_cnt = axis_dst_cl_size;
    vnc_row_cl_left = params->pln_dst_cl_size;
    ll_dst_cl_left = params->pln_dst_cl_size;
  }

  params->dst_cl_step_in = GetShapeSize(in_shape, GetAxisIndex(src_format, dst_cl_char) + 1);
  params->dst_cl_step_out = GetShapeSize(out_shape, dst_axis_pos_c);
  if (params->nc_le_vcol == 0) {
    params->dst_cl_lp_step_in = params->dst_cl_step_in;
    params->dst_cl_lp_step_out = params->dst_cl_step_out;
  } else {
    params->dst_cl_lp_step_in = params->dst_cl_step_in * per_vnc_dst_cl_cnt;
    params->dst_cl_lp_step_out = params->dst_cl_step_out * per_vnc_dst_cl_cnt;
  }

  GetMcInfoPositive1010(dst_cl_lp_cnt, vnc_row_cl_left, ll_dst_cl_left, c_lp_cnt, c_left, dst_cr_lp_cnt, vnc_row_left,
                        ll_dst_cr_left, compile_info->block_dim, *params);
  OP_LOGD(context->GetNodeName(), "TillingPositiveMode1010 tiling_data:%s",
          GetTilingDataString<int64_t>(context).c_str());
  return ge::SUCCESS;
}
}  // namespace transdata
}  // namespace optiling
