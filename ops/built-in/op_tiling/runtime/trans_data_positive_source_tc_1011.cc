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
void GetMcInfoPositive1011(int64_t axis_dst_r2nd_lp_cnt, int64_t axis_dst_r2nd_left, int64_t c_lp_cnt, int64_t c_left,
                           int64_t axis_src_cl_lp_cnt, int64_t axis_src_cl_left, int64_t core_num,
                           TransDataMode1011Param& params) {
  int64_t tmp_full_loop_cnt_r2nd = ge::FloorDiv(axis_dst_r2nd_lp_cnt, core_num) > 0 ? core_num : 0;
  int64_t reminder_loop_cnt_r2nd = GetRemainder(axis_dst_r2nd_lp_cnt, core_num);
  if (reminder_loop_cnt_r2nd == 0) {
    tmp_full_loop_cnt_r2nd += core_num;
  }
  int64_t full_loop_cnt_r2nd = tmp_full_loop_cnt_r2nd + reminder_loop_cnt_r2nd;

  int64_t tmp_full_loop_cnt_c = ge::FloorDiv(c_lp_cnt, core_num) > 0 ? core_num : 0;
  int64_t reminder_loop_cnt_c = GetRemainder(c_lp_cnt, core_num);
  if (reminder_loop_cnt_c == 0) {
    tmp_full_loop_cnt_c += core_num;
  }
  int64_t full_loop_cnt_c = tmp_full_loop_cnt_c + reminder_loop_cnt_c;

  int64_t tmp_full_loop_cnt_left = ge::FloorDiv(axis_src_cl_lp_cnt, core_num) > 0 ? core_num : 0;
  int64_t reminder_loop_cnt_left = GetRemainder(axis_src_cl_lp_cnt, core_num);
  if (reminder_loop_cnt_left == 0) {
    tmp_full_loop_cnt_left += core_num;
  }
  int64_t full_loop_cnt_left = tmp_full_loop_cnt_left + reminder_loop_cnt_left;

  auto max_value = std::max(std::max(full_loop_cnt_r2nd, full_loop_cnt_left), full_loop_cnt_c);
  if (max_value == full_loop_cnt_r2nd) {
    params.mc_on_cl = 0;
    params.used_core_cnt = ge::CeilDiv(axis_dst_r2nd_lp_cnt, ge::CeilDiv(axis_dst_r2nd_lp_cnt, core_num));
    params.nlc_dst_r2nd_lp_cnt = ge::CeilDiv(axis_dst_r2nd_lp_cnt, params.used_core_cnt);
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
  } else if (max_value == full_loop_cnt_left) {
    params.mc_on_cl = 1;
    params.used_core_cnt = ge::CeilDiv(axis_src_cl_lp_cnt, ge::CeilDiv(axis_src_cl_lp_cnt, core_num));
    params.nlc_src_cl_lp_cnt = ge::CeilDiv(axis_src_cl_lp_cnt, params.used_core_cnt);
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
    params.used_core_cnt = ge::CeilDiv(c_lp_cnt, ge::CeilDiv(c_lp_cnt, core_num));
    params.nlc_c_lp_cnt = ge::CeilDiv(c_lp_cnt, params.used_core_cnt);
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

ge::graphStatus TillingPositiveMode1011(TilingContext* context, const gert::Shape& in_shape,
                                        const gert::Shape& out_shape, const RealSrcDstFormat* real_formats,
                                        const TransDataCompileInfo* compile_info) {
  auto params = context->GetTilingData<TransDataMode1011Param>();
  OPS_CHECK_NULL_WITH_CONTEXT(context, params);
  auto src_td = context->GetInputDesc(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, src_td);
  auto dtype = src_td->GetDataType();
  int64_t block_elem_cnt = BLOCK_BYTE_SIZE / ge::GetSizeByDataType(dtype);
  RealFormat src_format = real_formats->src;
  RealFormat dst_format = real_formats->dst;
  int64_t axis_c_size = in_shape[in_shape.GetDimNum() - 1];
  int64_t c0_len = out_shape[out_shape.GetDimNum() - 1];
  int64_t ub_size = compile_info->ub_size;
  int64_t core_num = compile_info->block_dim;
  OP_TILING_CHECK(c0_len == 0, VECTOR_INNER_ERR_REPORT_TILIING("TransData", "invalid value c0_len = 0 "),
                  return ge::GRAPH_FAILED);
  GetCommonParam(ub_size, block_elem_cnt, c0_len, axis_c_size, *params);

  params->tiling_mode = TILING_MODE_1011;

  // target axis -2 tiling parameters
  int32_t dst_axis_pos_c = GetAxisIndex(dst_format, RAT_C);
  int32_t src_axis_pos_c = GetAxisIndex(src_format, RAT_C);
  int32_t dst_r2nd_in_src_idx = GetAxisIndex(src_format, GetAxisType(dst_format, -2));
  int64_t axis_dst_r2nd_size = in_shape[dst_r2nd_in_src_idx];
  if (axis_dst_r2nd_size < VNC_LINES) {
    params->dst_r2nd_lp_unit = axis_dst_r2nd_size;
  } else {
    params->dst_r2nd_lp_unit = VNC_LINES;
  }
  int64_t axis_dst_r2nd_lp_cnt = ge::CeilDiv(axis_dst_r2nd_size, params->dst_r2nd_lp_unit);
  int64_t axis_dst_r2nd_left = GetRemainder(axis_dst_r2nd_size, params->dst_r2nd_lp_unit);
  params->dst_r2nd_lp_step_in = GetShapeSize(in_shape, dst_r2nd_in_src_idx + 1) * params->dst_r2nd_lp_unit;
  params->dst_r2nd_lp_step_out = GetShapeSize(out_shape, -1) * params->dst_r2nd_lp_unit;
  params->dst_r2nd_step_in = GetShapeSize(in_shape, dst_r2nd_in_src_idx + 1);

  // source axis c tiling parameters
  int64_t used_vnc_line_size = ge::FloorDiv(params->vnc_line_size, params->c0_size) * params->c0_size;
  if (axis_c_size < used_vnc_line_size) {
    params->c_lp_unit = axis_c_size;
  } else {
    params->c_lp_unit = used_vnc_line_size;
  }
  params->c_lp_step_in = params->c_lp_unit;
  int64_t lp_c1_cnt = ge::CeilDiv(params->c_lp_unit, c0_len);
  params->c_lp_step_out = lp_c1_cnt * GetShapeSize(out_shape, dst_axis_pos_c + 1);
  params->c_step_out = GetShapeSize(out_shape, dst_axis_pos_c + 1);
  int64_t c_lp_cnt = ge::CeilDiv(axis_c_size, params->c_lp_unit);
  int64_t c_left = GetRemainder(axis_c_size, params->c_lp_unit);

  // source axis left tiling parameters
  int32_t chr_pos = GetAxisIndex(src_format, GetAxisType(dst_format, DIM_IDX_NEG_TWO));
  int32_t src_format_left_len = in_shape.GetDimNum();
  gert::Shape src_left_shape = {};
  for (int32_t i = 0; i < src_format_left_len; i++) {
    if (i == src_axis_pos_c || i == chr_pos) {
      continue;
    }
    RealAxisType cur_char = GetAxisType(src_format, i);
    int32_t cur_pos = GetAxisIndex(src_format, cur_char);
    src_left_shape.AppendDim(in_shape[cur_pos]);
  }
  src_left_shape.AppendDim(1);
  int64_t axis_src_cl_size = GetShapeSize(src_left_shape, 0);
  int64_t pln_src_cl_cnt = used_vnc_line_size / CeilAlign(params->c_lp_unit, c0_len);
  if (axis_src_cl_size < pln_src_cl_cnt) {
    params->src_cl_lp_unit = axis_src_cl_size;
  } else {
    params->src_cl_lp_unit = pln_src_cl_cnt;
  }
  int64_t axis_src_cl_lp_cnt = ge::CeilDiv(axis_src_cl_size, params->src_cl_lp_unit);
  int64_t axis_src_cl_left = GetRemainder(axis_src_cl_size, params->src_cl_lp_unit);
  params->src_cl_lp_step_in = GetShapeSize(in_shape, -1) * params->src_cl_lp_unit;
  params->src_cl_lp_step_out = 0;

  // parameters for output data
  int32_t excute_i = -1;
  for (int32_t i = src_format_left_len - 1; i >= 0; i = i - 1) {
    if (i == src_axis_pos_c || i == chr_pos) {
      continue;
    }
    excute_i = excute_i + 1;
    RealAxisType chr = GetAxisType(src_format, i);
    int32_t src_chr_pos = GetAxisIndex(src_format, chr);
    int32_t dst_chr_pos = GetAxisIndex(dst_format, chr);
    if (excute_i == 0) {
      params->cl_out_0_size = in_shape[src_chr_pos];
      params->cl_out_0_src_rsize = GetShapeSize(src_left_shape, -1);
      params->cl_out_0_dst_asize = GetShapeSize(out_shape, dst_chr_pos + 1);
    } else if (excute_i == 1) {
      params->cl_out_1_size = in_shape[src_chr_pos];
      params->cl_out_1_src_rsize = GetShapeSize(src_left_shape, DIM_IDX_NEG_TWO);
      params->cl_out_1_dst_asize = GetShapeSize(out_shape, dst_chr_pos + 1);
    }
  }

  GetMcInfoPositive1011(axis_dst_r2nd_lp_cnt, axis_dst_r2nd_left, c_lp_cnt, c_left, axis_src_cl_lp_cnt,
                        axis_src_cl_left, core_num, *params);
  OP_LOGD(context->GetNodeName(), "TillingPositiveMode1011 tiling_data:%s",
          GetTilingDataString<int64_t>(context).c_str());
  return ge::GRAPH_SUCCESS;
}
}  // namespace transdata
}  // namespace optiling
