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
const int32_t FRAME_LEVEL = 2;

void GetFullLpCnt(const int64_t& core_num, const int64_t& src_lp_cnt, int64_t& full_lp_cnt) {
  int64_t tmp_full_lp_cnt = ge::FloorDiv(src_lp_cnt, core_num) > 0 ? core_num : 0;
  int64_t reminder_lp_cnt = GetRemainder(src_lp_cnt, core_num);
  if (reminder_lp_cnt == 0) {
    tmp_full_lp_cnt += core_num;
  }
  full_lp_cnt = tmp_full_lp_cnt + reminder_lp_cnt;
}

void GetMcInfoPositiveNtc100(int64_t src_cr_lp_cnt, int64_t src_cr_size, int64_t src_c_lp_cnt,
                             int64_t src_c_size, int64_t src_cl_lp_cnt, int64_t src_cl_size,
                             int64_t core_num, TransDataNtc100Param& params) {
  int64_t full_lp_cnt_cr = 0;
  int64_t full_lp_cnt_c = 0;
  int64_t full_lp_cnt_cl = 0;

  GetFullLpCnt(core_num, src_cr_lp_cnt, full_lp_cnt_cr);
  GetFullLpCnt(core_num, src_c_lp_cnt, full_lp_cnt_c);
  GetFullLpCnt(core_num, src_cl_lp_cnt, full_lp_cnt_cl);
  if (full_lp_cnt_cl >= full_lp_cnt_c && full_lp_cnt_cl >= full_lp_cnt_cr) {
    int64_t used_core_cnt = ge::CeilDiv(src_cl_lp_cnt, ge::CeilDiv(src_cl_lp_cnt, core_num));
    int64_t nlc_cl_lp_cnt = ge::CeilDiv(src_cl_lp_cnt, used_core_cnt);
    int64_t lc_cl_lp_cnt = src_cl_lp_cnt - nlc_cl_lp_cnt * (used_core_cnt - 1);
    params.mc_pos = 0;                                                 // mc_pos
    params.used_core_cnt = used_core_cnt;                              // used_core_cnt
    params.core_step_in = nlc_cl_lp_cnt * params.src_cl_lp_step_in;    // core_step_in
    params.core_step_out = nlc_cl_lp_cnt * params.src_cl_lp_step_out;  // core_step_out
    params.nlc_cl_lp_cnt = nlc_cl_lp_cnt;                              // nlc_cl_lp_cnt
    params.nlc_cl_left = 0;                                            // nlc_cl_left
    params.nlc_c_lp_cnt = src_c_lp_cnt;                                // nlc_c_lp_cnt
    params.nlc_c_left = src_c_size % params.src_c_lp_unit;             // nlc_c_left
    params.nlc_cr_lp_cnt = src_cr_lp_cnt;                              // nlc_cr_lp_cnt
    params.nlc_cr_left = src_cr_size % params.src_cr_lp_unit;          // nlc_cr_left
    params.lc_cl_lp_cnt = lc_cl_lp_cnt;                                // lc_cl_lp_cnt
    params.lc_cl_left = src_cl_size % params.src_cl_lp_unit;           // lc_cl_left
    params.lc_c_lp_cnt = src_c_lp_cnt;                                 // lc_c_lp_cnt
    params.lc_c_left = src_c_size % params.src_c_lp_unit;              // lc_c_left
    params.lc_cr_lp_cnt = src_cr_lp_cnt;                               // lc_cr_lp_cnt
    params.lc_cr_left = src_cr_size % params.src_cr_lp_unit;           // lc_cr_left
  } else if (full_lp_cnt_c >= full_lp_cnt_cr && full_lp_cnt_c >= full_lp_cnt_cl) {
    int64_t used_core_cnt = ge::CeilDiv(src_c_lp_cnt, ge::CeilDiv(src_c_lp_cnt, core_num));
    int64_t nlc_c_lp_cnt = ge::CeilDiv(src_c_lp_cnt, used_core_cnt);
    int64_t lc_c_lp_cnt = src_c_lp_cnt - nlc_c_lp_cnt * (used_core_cnt - 1);
    params.mc_pos = 1;                                               // mc_pos
    params.used_core_cnt = used_core_cnt;                            // used_core_cnt
    params.core_step_in = nlc_c_lp_cnt * params.src_c_lp_step_in;    // core_step_in
    params.core_step_out = nlc_c_lp_cnt * params.src_c_lp_step_out;  // core_step_out
    params.nlc_cl_lp_cnt = src_cl_lp_cnt;                            // nlc_cl_lp_cnt
    params.nlc_cl_left = src_cl_size % params.src_cl_lp_unit;        // nlc_cl_left
    params.nlc_c_lp_cnt = nlc_c_lp_cnt;                              // nlc_c_lp_cnt
    params.nlc_c_left = 0;                                           // nlc_c_left
    params.nlc_cr_lp_cnt = src_cr_lp_cnt;                            // nlc_cr_lp_cnt
    params.nlc_cr_left = src_cr_size % params.src_cr_lp_unit;        // nlc_cr_left
    params.lc_cl_lp_cnt = src_cl_lp_cnt;                             // lc_cl_lp_cnt
    params.lc_cl_left = src_cl_size % params.src_cl_lp_unit;         // lc_cl_left
    params.lc_c_lp_cnt = lc_c_lp_cnt;                                // lc_c_lp_cnt
    params.lc_c_left = src_c_size % params.src_c_lp_unit;            // lc_c_left
    params.lc_cr_lp_cnt = src_cr_lp_cnt;                             // lc_cr_lp_cnt
    params.lc_cr_left = src_cr_size % params.src_cr_lp_unit;         // lc_cr_left
  } else if (full_lp_cnt_cr >= full_lp_cnt_c && full_lp_cnt_cr >= full_lp_cnt_cl) {
    int64_t used_core_cnt = ge::CeilDiv(src_cr_lp_cnt, ge::CeilDiv(src_cr_lp_cnt, core_num));
    int64_t nlc_cr_lp_cnt = ge::CeilDiv(src_cr_lp_cnt, used_core_cnt);
    int64_t lc_cr_lp_cnt = src_cr_lp_cnt - nlc_cr_lp_cnt * (used_core_cnt - 1);
    params.mc_pos = TRANSDATA_TILING_PARAM_2;                          // mc_pos
    params.used_core_cnt = used_core_cnt;                              // used_core_cnt
    params.core_step_in = nlc_cr_lp_cnt * params.src_cr_lp_step_in;    // core_step_in
    params.core_step_out = nlc_cr_lp_cnt * params.src_cr_lp_step_out;  // core_step_out
    params.nlc_cl_lp_cnt = src_cl_lp_cnt;                              // nlc_cl_lp_cnt
    params.nlc_cl_left = src_cl_size % params.src_cl_lp_unit;          // nlc_cl_left
    params.nlc_c_lp_cnt = src_c_lp_cnt;                                // nlc_c_lp_cnt
    params.nlc_c_left = src_c_size % params.src_c_lp_unit;             // nlc_c_left
    params.nlc_cr_lp_cnt = nlc_cr_lp_cnt;                              // nlc_cr_lp_cnt
    params.nlc_cr_left = 0;                                            // nlc_cr_left
    params.lc_cl_lp_cnt = src_cl_lp_cnt;                               // lc_cl_lp_cnt
    params.lc_cl_left = src_cl_size % params.src_cl_lp_unit;           // lc_cl_left
    params.lc_c_lp_cnt = src_c_lp_cnt;                                 // lc_c_lp_cnt
    params.lc_c_left = src_c_size % params.src_c_lp_unit;              // lc_c_left
    params.lc_cr_lp_cnt = lc_cr_lp_cnt;                                // lc_cr_lp_cnt
    params.lc_cr_left = src_cr_size % params.src_cr_lp_unit;           // lc_cr_left
  }
}

ge::graphStatus TilingPositiveSourceNtc100(TilingContext* context, const gert::Shape& in_shape,
                                           const gert::Shape& out_shape, const RealFormat& src_format,
                                           const RealFormat& dst_format, int64_t core_num,
                                           int64_t block_elem_cnt, int64_t ub_size,
                                           ge::DataType dtype, int64_t c0_len) {
  auto params = context->GetTilingData<TransDataNtc100Param>();
  OPS_CHECK_NULL_WITH_CONTEXT(context, params);
  // get tiling params for using vnchwconv
  int64_t half_ub_size = c0_len == C0_16 ? ub_size / TRANSDATA_TILING_FACTOR_2 : ub_size / TRANSDATA_TILING_FACTOR_4;
  int64_t one_vnc_line_size = half_ub_size / VNC_LINES / block_elem_cnt * block_elem_cnt;
  int64_t tmp_ub_offset = one_vnc_line_size * VNC_LINES;
  params->ub_offset = c0_len == C0_16 ? tmp_ub_offset : tmp_ub_offset * TRANSDATA_TILING_FACTOR_2;
  params->vnc_line_size = one_vnc_line_size;
  params->c0_size = c0_len;

  // axis c-right tiling parameters
  params->cr_dims = FRAME_LEVEL;
  params->r1st_src_r2nd_dst_same = 1;
  int64_t c_idx = GetAxisIndex(src_format, RAT_C);
  int64_t c1_idx = GetAxisIndex(dst_format, RAT_C);
  int64_t axis_src_cr_size = GetShapeSize(in_shape, c_idx + 1);
  int64_t tmp_src_cr_lp_unit = params->vnc_line_size / c0_len / block_elem_cnt * block_elem_cnt;
  const std::vector<ge::DataType> dtype_list = {ge::DT_FLOAT, ge::DT_INT32, ge::DT_UINT32};
  if (axis_src_cr_size < TRANSDATA_TILING_FACTOR_2 * block_elem_cnt ||
      std::find(dtype_list.begin(), dtype_list.end(), dtype) != dtype_list.end()) {
    params->tiling_mode = TILING_MODE_1000;
    params->src_cr_lp_unit = axis_src_cr_size > tmp_src_cr_lp_unit ? tmp_src_cr_lp_unit : axis_src_cr_size;
  } else {
    params->tiling_mode = TILING_MODE_1001;
    params->src_cr_lp_unit = axis_src_cr_size > params->vnc_line_size ? params->vnc_line_size : axis_src_cr_size;
  }

  int64_t in_shape_len = in_shape.GetDimNum();
  int64_t tmp_src_cr_format_len = in_shape_len - c_idx - 1;
  gert::Shape tmp_src_cr_shape{};

  for (uint32_t i = 0; i < tmp_src_cr_format_len; i++) {
    tmp_src_cr_shape.AppendDim(in_shape[i + c_idx + 1]);
  }
  tmp_src_cr_shape.AppendDim(1);
  for (uint32_t i = 0; i < tmp_src_cr_format_len; i++) {
    RealAxisType tmp_src_cr_format_i = GetAxisType(src_format, in_shape_len - i - 1);
    int64_t tmp_src_idx = GetAxisIndex(src_format, tmp_src_cr_format_i);
    int64_t tmp_dst_idx = GetAxisIndex(dst_format, tmp_src_cr_format_i);
    if (i == 0) {
      params->cr_out_idx_0_size = in_shape[tmp_src_idx];
      params->cr_out_idx_0_dst_rsize = GetShapeSize(tmp_src_cr_shape, tmp_src_cr_shape.GetDimNum() - i - 1);
      params->cr_out_idx_0_dst_asize = GetShapeSize(out_shape, tmp_dst_idx + 1);
    } else if (i == 1) {
      params->cr_out_idx_1_size = in_shape[tmp_src_idx];
      params->cr_out_idx_1_dst_rsize = GetShapeSize(tmp_src_cr_shape, tmp_src_cr_shape.GetDimNum() - i - 1);
      params->cr_out_idx_1_dst_asize = GetShapeSize(out_shape, tmp_dst_idx + 1);
    }
  }

  // suppose there are 2 axises
  int64_t pad_axis_cnt = FRAME_LEVEL - tmp_src_cr_format_len;
  if (pad_axis_cnt) {
    params->cr_dims = 1;
    params->cr_out_idx_1_size = 1;
    params->cr_out_idx_1_dst_rsize = 1;
    params->cr_out_idx_1_dst_asize = 0;
  }
  if (GetAxisType(src_format, -1) != GetAxisType(dst_format, DIM_IDX_NEG_TWO)) {
    params->r1st_src_r2nd_dst_same = 0;
  }
  int64_t src_cr_lp_cnt = ge::CeilDiv(axis_src_cr_size, params->src_cr_lp_unit);
  params->src_cr_step_in = 1;
  params->src_cr_lp_step_in = params->src_cr_step_in * params->src_cr_lp_unit;
  if (params->cr_dims == TRANSDATA_TILING_PARAM_2) {
    params->src_cr_step_out = 0;
    params->src_cr_lp_step_out = 0;
  } else {
    int64_t tmp_idx = GetAxisIndex(dst_format, GetAxisType(src_format, -1));
    params->src_cr_step_out = GetShapeSize(out_shape, tmp_idx + 1);
    params->src_cr_lp_step_out = params->src_cr_step_out * params->src_cr_lp_unit;
  }

  // axis c tiling parameters
  int64_t axis_src_c_size = in_shape[c_idx];
  params->src_c_lp_unit = c0_len;
  int64_t src_c_lp_cnt = ge::CeilDiv(axis_src_c_size, params->src_c_lp_unit);
  params->src_c_step_in = GetShapeSize(in_shape, c_idx + 1);
  params->src_c_lp_step_in = params->src_c_step_in * params->src_c_lp_unit;
  params->src_c_lp_step_out = GetShapeSize(out_shape, c1_idx + 1);
  params->c_mod_c0 = GetRemainder(axis_src_c_size, c0_len);

  // axis left parameters
  params->cl_dims = FRAME_LEVEL;
  int64_t axis_src_cl_size = GetShapeSize(in_shape, 0) / GetShapeSize(in_shape, c_idx);
  int64_t tmp_src_cl_lp_unit = 1;
  if (params->tiling_mode == TILING_MODE_1000) {
    tmp_src_cl_lp_unit = NI_16;
  } else if (params->r1st_src_r2nd_dst_same == 0 && params->tiling_mode == TILING_MODE_1001 &&
             axis_src_cl_size > core_num) {
    tmp_src_cl_lp_unit = ge::FloorDiv(params->vnc_line_size, ge::CeilDiv(params->src_cr_lp_unit, c0_len) * c0_len);
  } else {
    tmp_src_cl_lp_unit = 1;
  }
  params->src_cl_lp_unit = axis_src_cl_size > tmp_src_cl_lp_unit ? tmp_src_cl_lp_unit : axis_src_cl_size;
  int64_t src_cl_lp_cnt = ge::CeilDiv(axis_src_cl_size, params->src_cl_lp_unit);

  // count method: left_axis_size/dst_rsize%size*asize
  gert::Shape tmp_src_cl_shape = {};
  int64_t tmp_src_cl_format_len = c_idx;
  for (uint32_t i = 0; i < tmp_src_cl_format_len; i++) {
    tmp_src_cl_shape.AppendDim(in_shape[i]);
  }
  tmp_src_cl_shape.AppendDim(1);
  for (uint32_t i = 0; i < tmp_src_cl_format_len; i++) {
    RealAxisType tmp_src_cl_format_i = GetAxisType(src_format, c_idx - i - 1);
    int64_t tmp_src_cl_idx = GetAxisIndex(src_format, tmp_src_cl_format_i);
    int64_t tmp_dst_cl_idx = GetAxisIndex(dst_format, tmp_src_cl_format_i);
    if (i == 0) {
      params->cl_out_idx_0_size = in_shape[tmp_src_cl_idx];
      params->cl_out_idx_0_dst_rsize = GetShapeSize(tmp_src_cl_shape, tmp_src_cl_shape.GetDimNum() - i - 1);
      params->cl_out_idx_0_dst_asize = GetShapeSize(out_shape, tmp_dst_cl_idx + 1);
    } else if (i == 1) {
      params->cl_out_idx_1_size = in_shape[tmp_src_cl_idx];
      params->cl_out_idx_1_dst_rsize = GetShapeSize(tmp_src_cl_shape, tmp_src_cl_shape.GetDimNum() - i - 1);
      params->cl_out_idx_1_dst_asize = GetShapeSize(out_shape, tmp_dst_cl_idx + 1);
    }
  }

  // suppose there are 2 axises
  pad_axis_cnt = FRAME_LEVEL - tmp_src_cl_format_len;
  if (pad_axis_cnt) {
    params->cl_dims = 1;
    params->cl_out_idx_1_size = 1;
    params->cl_out_idx_1_dst_rsize = 1;
    params->cl_out_idx_1_dst_asize = 0;
  }
  params->src_cl_step_in = GetShapeSize(in_shape, c_idx);
  params->src_cl_lp_step_in = params->src_cl_step_in * params->src_cl_lp_unit;
  if (params->cl_dims == TRANSDATA_TILING_PARAM_2) {
    params->src_cl_step_out = 0;
    params->src_cl_lp_step_out = 0;
  } else {
    int64_t tmp_idx = GetAxisIndex(dst_format, GetAxisType(src_format, 0));
    params->src_cl_step_out = GetShapeSize(out_shape, tmp_idx + 1);
    params->src_cl_lp_step_out = params->src_cl_step_out * params->src_cl_lp_unit;
  }

  // mulitple core parameters
  GetMcInfoPositiveNtc100(src_cr_lp_cnt, axis_src_cr_size, src_c_lp_cnt, axis_src_c_size, src_cl_lp_cnt,
                          axis_src_cl_size, core_num, *params);

  return ge::GRAPH_SUCCESS;
}
}  // namespace transdata
}  // namespace optiling
