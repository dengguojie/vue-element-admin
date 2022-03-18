/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
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
 * \file cache_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_DX_CACHE_TILING_H_
#define OPS_BUILT_IN_OP_TILING_DX_CACHE_TILING_H_

#include <algorithm>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <climits>
#include <cmath>
#include <numeric>
#include <ratio>
#include <unistd.h>
#include <vector>
#include "op_log.h"
#include "op_tiling.h"

#define CUBE_INNER_ERR_REPORT(op_name, err_msg, ...) \
  do { \
      OP_LOGE(op_name, err_msg, ##__VA_ARGS__); \
      REPORT_INNER_ERROR("E69999", "op[%s], " err_msg, op_name, ##__VA_ARGS__); \
  } while(0)

#define CHECK_OP_FUNC(cond, post_action_expr, msg, ...)                                                          \
  {                                                                                                              \
    if (cond) {                                                                                                  \
      CUBE_INNER_ERR_REPORT("Conv2dBackpropInput", msg, ##__VA_ARGS__);                                      \
      post_action_expr;                                                                                          \
    }                                                                                                            \
  }

namespace optiling {

struct DxParas {
  int32_t core_num = 32;
  int32_t groups = 0;
  int32_t padu = 0;
  int32_t padd = 0;
  int32_t padl = 0;
  int32_t padr = 0;
  int32_t dilations_n = 0;
  int32_t dilations_c = 0;
  int32_t dilations_h = 0;
  int32_t dilations_w = 0;
  int32_t stride_h = 0;
  int32_t stride_w = 0;
  int32_t batch = 0;
  int32_t batch_o = 0;
  int32_t co1 = 0;
  int32_t ho = 0;
  int32_t wo = 0;
  int32_t co = 0;
  int32_t filter_cin1hw = 0;
  int32_t filter_cout1 = 0;
  int32_t kn = 0;
  int32_t kc = 0;
  int32_t kh = 0;
  int32_t kw = 0;
  int32_t c1 = 0;
  int32_t cin = 0;
  int32_t h = 0;
  int32_t w = 0;
  int32_t hw = 0;
  int32_t fmap_h_padding = 0;
  int32_t fmap_w_padding = 0;
  int32_t filter_h_dilation = 0;
  int32_t filter_w_dilation = 0;
  int32_t stride_expand_flag = 0;
  bool repo_seed_flag = false;
  bool repo_costmodel_flag = false;
  bool repo_binary_flag = false;
  bool dx_hw_align_flag = false;
  bool dx_c_align_flag = false;
  bool dx_no_overlap_condition_1 = false;
  bool dx_no_overlap_condition_2 = false;
};

struct Tiling {
  int32_t n_cub = 1;
  int32_t db_cub = 1;
  int32_t m_l0 = 1;
  int32_t k_l0 = 1;
  int32_t n_l0 = 1;
  int32_t m_l0_opti = 1;
  int32_t k_l0_opti = 1;
  int32_t n_l0_opti = 1;
  int32_t batch_dim = 1;
  int32_t n_dim = 1;
  int32_t m_dim = 1;
  int32_t batch_single_core_size = 1;
  int32_t n_single_core_size = 1;
  int32_t m_single_core_size = 1;
  int32_t k_single_core_size = 1;
  int32_t k_al1 = 1;
  int32_t k_bl1 = 1;
  int32_t m_al1 = 1;
  int32_t n_bl1 = 1;
  int32_t db_al1 = 1;
  int32_t db_bl1 = 1;
  int32_t k_aub = 1;
  int32_t m_aub = 1;
  int32_t db_aub = 1;
  int32_t k_org_dim = 1;
  int32_t db_l0c = 1;
  int32_t h_num = 1;
  int32_t init_db_al1 = 1;
  int32_t init_db_bl1 = 1;
  int32_t init_db_l0c = 1;
  bool update_l1 = false;
};

struct RunInfoRaras {
  int32_t g_extend;
  int32_t dx_c1_extend;
  int32_t dy_c_ori;
  int32_t multiple_extend;
  int32_t shape_up_modify;
  int32_t shape_left_modify;
  int32_t shape_down_modify;
  int32_t shape_right_modify;
  int32_t pad_up_before;
  int32_t pad_left_before;
  int32_t pad_down_after;
  int32_t pad_right_after;
  int32_t batch_single_core;
  int32_t n_single_core;
  int32_t m_single_core;
  int32_t n_l0_div_ub;
  int32_t k_al1_div_16;
  int32_t k_bl1_div_16;
  int32_t al1_bound;
  int32_t bl1_bound;
  int32_t aub_bound;
};

struct RunInfoParaStrideLargeOne {
  int32_t filter_cin1hw;
  int32_t filter_cout1;
  int32_t batch;
  int32_t co1;
  int32_t ho;
  int32_t wo;
  int32_t cin;
  int32_t c1;
  int32_t h;
  int32_t w;
  int32_t kh;
  int32_t kw;
  int32_t g_extend;
  int32_t dx_c1_extend;
  int32_t multiple_extend;
  int32_t padu;
  int32_t padd;
  int32_t padl;
  int32_t padr;
  int32_t stride_h;
  int32_t stride_w;
  int32_t shape_up_modify;
  int32_t shape_left_modify;
  int32_t shape_down_modify;
  int32_t shape_right_modify;
  int32_t pad_up_before;
  int32_t pad_left_before;
  int32_t pad_down_after;
  int32_t pad_right_after;
  int32_t batch_dim;
  int32_t n_dim;
  int32_t m_dim;
  int32_t batch_single_core;
  int32_t m_al1;
  int32_t n_bl1;
  int32_t k_aub;
  int32_t m_aub;
  int32_t m_l0;
  int32_t n_l0_div_ub;
  int32_t n_cub;
  int32_t k_l0;
  int32_t k_al1_div_16;
  int32_t k_bl1_div_16;
  int32_t al1_bound;
  int32_t bl1_bound;
  int32_t aub_bound;
};

struct RunInfoParaStrideEqualOne {
  int32_t batch;
  int32_t co;
  int32_t ho;
  int32_t wo;
  int32_t filter_cin1hw;
  int32_t filter_cout1;
  int32_t cin;
  int32_t c1;
  int32_t h;
  int32_t w;
  int32_t kh;
  int32_t kw;
  int32_t g_extend;
  int32_t dx_c1_extend;
  int32_t multiple_extend;
  int32_t padu;
  int32_t padd;
  int32_t padl;
  int32_t padr;
  int32_t shape_up_modify;
  int32_t shape_left_modify;
  int32_t shape_down_modify;
  int32_t shape_right_modify;
  int32_t pad_up_before;
  int32_t pad_left_before;
  int32_t pad_down_after;
  int32_t pad_right_after;
  int32_t batch_dim;
  int32_t n_dim;
  int32_t m_dim;
  int32_t batch_single_core;
  int32_t m_al1;
  int32_t n_bl1;
  int32_t k_aub;
  int32_t m_aub;
  int32_t m_l0;
  int32_t n_l0_div_ub;
  int32_t n_cub;
  int32_t k_l0;
  int32_t k_al1_div_16;
  int32_t k_bl1_div_16;
  int32_t al1_bound;
  int32_t bl1_bound;
  int32_t aub_bound;
};

bool GenTiling(const DxParas &params, Tiling &tiling, int32_t &tiling_id);
}; // namespace optiling
#endif
