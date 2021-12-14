/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file as_strided.h
 * \brief
 */
#ifndef __AS_STRIDED_H__
#define __AS_STRIDED_H__

#include <string>
#include <algorithm>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "../op_proto/util/error_util.h"
#include "op_log.h"

namespace optiling {
struct AsStridedInfo {
  int64_t tiling_mode;
  int64_t used_core_cnt;
  int64_t out_ub_offset;
  int64_t vnc_col_size;
  int64_t m_axis_1_burst_unit;
  int64_t m_axis_1_lp_unit;
  int64_t m_axis_0_lp_unit;
  int64_t mc_pos;
  int64_t core_step_in;
  int64_t nlc_m_axis_1_lp_cnt;
  int64_t nlc_m_axis_1_lp_left;
  int64_t lc_m_axis_1_lp_cnt;
  int64_t lc_m_axis_1_lp_left;
  int64_t nlc_m_axis_0_lp_cnt;
  int64_t nlc_m_axis_0_lp_left;
  int64_t lc_m_axis_0_lp_cnt;
  int64_t lc_m_axis_0_lp_left;
  int64_t storage_offset;
  int64_t last_dim_size;
  int64_t last_dim_stride;
  int64_t rsecond_dim_size;
  int64_t rsecond_dim_stride;
  int64_t out_lp_step;
  int64_t nfirst_cnt_per_row;
  int64_t dim_num;  // except last dim
  std::vector<int64_t> dim_except_last_paras;  // the order is: rsize0, size0, stride0, rsize1, size1, stride1 ...
  AsStridedInfo() {
    storage_offset = 0;
    rsecond_dim_size = 1;
    rsecond_dim_stride = 0;
    nfirst_cnt_per_row = 1;
    tiling_mode = 1;
    used_core_cnt = 1;
    dim_num = 1;
    mc_pos = 0;
    out_ub_offset = 0;
    vnc_col_size = 16;
    core_step_in = 0;
  }
};

}// namespace optiling

#endif  //__AS_STRIDED_H__
