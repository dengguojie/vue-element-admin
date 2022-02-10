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
 * \file dw_cache_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_DW_CACHE_TILING_H
#define OPS_BUILT_IN_OP_TILING_DW_CACHE_TILING_H

#include <algorithm>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <climits>
#include <map>
#include <cmath>
#include <numeric>
#include <ratio>
#include <unistd.h>
#include <vector>
#include "op_log.h"

namespace optiling::conv2d_dw {

struct Conv2dBpFilterParas {
  int32_t batch = 1;
  int32_t ho = 1;
  int32_t wo = 1;
  int32_t co = 1;
  int32_t co1 = 1;
  int32_t co0 = 16;
  int32_t hi = 1;
  int32_t wi = 1;
  int32_t ci = 1;
  int32_t ci1 = 1;
  int32_t ci0 = 16;
  int32_t k0 = 16;
  int32_t kh = 1;
  int32_t kw = 1;
  int32_t groups = 1;
  int32_t stride_h = 1;
  int32_t stride_w = 1;
  int32_t dilation_h = 1;
  int32_t dilation_w = 1;
  int32_t pad_u = 0;
  int32_t pad_d = 0;
  int32_t pad_l = 0;
  int32_t pad_r = 0;
  int32_t max_core_num = 64;
  uint32_t aub_fused_num = 1;
  uint32_t bub_fused_num = 1;
  uint32_t cub_fused_num = 0;
  uint32_t a_dtype = 1;
  uint32_t b_dtype = 1;
  uint32_t c_dtype = 2;
  std::string op_type;
};

struct Conv2dDwTiling {
  std::string tiling_id;
  int32_t batch_dim = 1;
  int32_t group_dim = 1;
  int32_t n_dim = 1;
  int32_t m_dim = 1;
  int32_t h_dim = 1;
  int32_t m_l0 = 1;
  int32_t k_l0 = 1;
  int32_t n_l0 = 1;
  int32_t db_l0c = 1;
  int32_t kal1_16 = 1;
  int32_t kbl1_16 = 1;
  int32_t m_al1 = 1;
  int32_t n_bl1 = 1;
  int32_t db_al1 = 1;
  int32_t db_bl1 = 1;
  int32_t ho_bl1 = 1;
  int32_t bl1_bound = 0;
  int32_t n_cub = 1;
  int32_t k_aub = 1;
  int32_t m_aub = 1;
  int32_t k_bub = 1;
  int32_t n_bub = 1;
  int32_t db_aub = 1;
  int32_t db_bub = 1;
  int32_t db_cub = 1;
  int32_t k_org_dim = 1;
  int32_t aub_multi_flag = 0;
  int32_t bub_multi_flag = 0;
  int32_t min_kl1_cmp_kl0 = 0;
  int32_t al1_attach_flag = 0;
  int32_t bl1_attach_flag = 0;
  int32_t abkl1_attach_flag = 0;
  int32_t reorder_l1_mn = 0;
  int32_t reorder_l0_mn = 0;
  bool al1_full_load = false;
  bool bl1_full_load = false;
};

struct SingleCoreStatus {
  int32_t batch1 = 1;
  int32_t m1 = 1;
  int32_t n1 = 1;
  int32_t k1 = 1;
  int32_t batch2 = 1;
  int32_t m2 = 1;
  int32_t k2 = 1;
  int32_t n2 = 1;
  int32_t batch_dim = 1;
  int32_t m_dim = 1;
  int32_t n_dim = 1;
  int32_t h_dim = 1;
  int32_t min_load_size = INT_MAX;
  int32_t core_use = 1;
  int32_t al1_full_load_size = 0;
  int32_t bl1_full_load_size = 0;
  int32_t al1_k_full_load_size = 0;
  int32_t bl1_k_full_load_size = 0;
};

struct BlockDimCalculator {
  int32_t batch2 = 1;
  int32_t m2 = 1;
  int32_t k2 = 1;
  int32_t n2 = 1;
  int32_t n_dim_factor = 1;
  int32_t batch_dim_factor = 1;
  int32_t m_dim_factor = 1;
  int32_t h_dim_factor = 1;
  int32_t batch_dim_cnt = 0;
  int32_t m_dim_cnt = 0;
  int32_t n_dim_cnt = 0;
  int32_t h_dim_cnt = 0;
  int32_t tmp_load_size = INT_MAX;
  int32_t tmp_load_size_al1k = INT_MAX;
  int32_t tmp_load_size_bl1k = INT_MAX;
  int32_t tmp_load_size_neither = INT_MAX;
  int32_t tmp_core_use = 1;
  int32_t* batch_dim_array;
  int32_t* m_dim_array;
  int32_t* n_dim_array;
  int32_t* h_dim_array;
  int32_t al1_full_load_size = 0;
  int32_t bl1_full_load_size = 0;
  int32_t al1_k_full_load_size = 0;
  int32_t bl1_k_full_load_size = 0;
  int32_t al1_min_load_size = 0;
  int32_t bl1_min_load_size = 0;
  int32_t single_core_ho = 0;
  int32_t single_core_hi = 0;
  bool al1_full_load = false;
  bool bl1_full_load = false;
  bool al1_k_full_load = false;
  bool bl1_k_full_load = false;
  int32_t l1_min_pnt = 0;
  int32_t max_ml1 = 0;
  int32_t max_nl1 = 0;
};

struct L0Status {
  int32_t m_l0 = 1;
  int32_t n_l0 = 1;
  int32_t k_l0 = 1;
  int32_t db_l0c = 1;
  int32_t db_l0a = 2;
  int32_t db_l0b = 2;
  int32_t final_ml0 = 0;
  int32_t final_kl0 = 0;
  int32_t final_nl0 = 0;
  int32_t final_load_size = INT_MAX;
  float final_l0c_use = 0;
  int32_t final_mul = 0;
  int32_t max_mk = 1;
  int32_t max_nk = 1;
  int32_t max_mn = 1;
  int32_t max_l0n = 1;
  int32_t min_l0n = 1;
  int32_t max_axis_num = 0;
  int32_t max_axis_pnt = 1;
  void SetInitLoadStatus()
  {
    final_ml0 = 0;
    final_kl0 = 0;
    final_nl0 = 0;
    final_load_size = INT_MAX;
    final_l0c_use = 0;
    final_mul = 0;
  }
};

struct L0Factors {
  int32_t final_ml0 = 0;
  int32_t final_kl0 = 0;
  int32_t final_nl0 = 0;
  int32_t final_load_size = INT_MAX;
  float final_l0c_use = 0;
  int32_t final_mul = 0;
};

struct L0Calculator {
  int32_t al1_min_load_size = 0;
  int32_t bl1_min_load_size = 0;
  int32_t min_ho = 0;
  int32_t min_hi = 0;
  int32_t tmp_load_size_al1k = 0;
  int32_t tmp_load_size_bl1k = 0;
};

struct MKNParasCombo {
  int32_t paras_combo[9];
};

struct L1Status {
  int32_t kal1_16 = 1;
  int32_t kbl1_16 = 1;
  int32_t m_al1 = 1;
  int32_t n_bl1 = 1;
  int32_t db_al1 = 1;
  int32_t db_bl1 = 1;
  int32_t al1_size = 0;
  int32_t bl1_size = 0;
  int32_t al1_times = 1;
  int32_t bl1_times = 1;
  int32_t all_times = 1;
  int32_t bl1_repeat = 1;
  int32_t al1_repeat = 1;
  int32_t load_size = 0;
  int32_t max_m_al1 = 1;
  int32_t max_n_bl1 = 1;
  int32_t max_k_al1 = 1;
  int32_t max_k_bl1 = 1;
  int32_t hi = 1;
  int32_t ho = 1;
  int32_t bl1_bound = 0;
  int32_t cur_l1_size = 0;
  bool both_full_load = false;
  bool al1_full_load = false;
  bool bl1_full_load = false;
  void SetStatus(int32_t kal1_16_input, int32_t kbl1_16_input, int32_t m_al1_input, int32_t n_bl1_input,
                 int32_t db_al1_input, int32_t db_bl1_input)
  {
    this->kal1_16 = kal1_16_input;
    this->kbl1_16 = kbl1_16_input;
    this->m_al1 = m_al1_input;
    this->n_bl1 = n_bl1_input;
    this->db_al1 = db_al1_input;
    this->db_bl1 = db_bl1_input;
  }
};

struct UbStatus {
  int32_t k_aub = 1;
  int32_t m_aub = 1;
  int32_t db_aub = 2;
  int32_t k_bub = 1;
  int32_t n_bub = 1;
  int32_t db_bub = 2;
  int32_t n_cub = 1;
  int32_t db_cub = 2;
};

class Conv2dDwCacheTiling {
 public:
  bool GenTiling(Conv2dDwTiling &tiling);
  Conv2dDwCacheTiling(const Conv2dBpFilterParas &params) : params(params) {}
  ~Conv2dDwCacheTiling() {
  }

 private:
  const Conv2dBpFilterParas params;
  BlockDimCalculator blockDimCalculator;
  L0Calculator l0Calculator;
  L0Status l0Status;
  L1Status l1Status;
  SingleCoreStatus singlecoreStatus;
  UbStatus ubStatus;

  bool GetL0Factors();
  void GetUbFactors();
  void GetL1Factors();
  void L1StatusBothFullLoad(int32_t res[][9]);
  void L1StatusAl1FullLoad(int32_t res[][9]);
  void L1StatusBl1FullLoad(int32_t res[][9]);
  void L1StatusNeitherFullLoad(int32_t res[][9]);
  int32_t GetL1Size();
  int32_t GetAL1Size();
  int32_t GetBL1Size();
  void GetkBL1Factor();
  void GetnBL1Factor();
  void GetkAL1Factor();
  void GetmAL1Factor();
  int32_t GetLoadSize();
  void GetL1LoadSize();
  void SetResFactors(L0Factors &res_factors);
  void GetParasCombo(MKNParasCombo *paras_combo_map);
  void GetL0StatusFromParasCombo(int32_t *paras_combo);
  void CheckSpecialTemplate();

  void GetFinalMkn();
  void GetBlockDim();
  void UpdateBlockDimRes();
  void NeitherFullLoadBlock();
  void GetL0FactorsCand(L0Factors &res_factors, int32_t *paras_combo);

  void SetParams(Conv2dDwTiling &tiling);
  void SetAttachFlag(Conv2dDwTiling &tiling);
  void FixTilingParam(Conv2dDwTiling &tiling);
  void GetTilingId(Conv2dDwTiling &tiling);

  static int32_t GetK2Ho(int32_t k, int32_t wo);
  static int32_t GetHo2Hi(int32_t ho, int32_t stride_h, int32_t kh);
  static void GetFactors(int32_t *cnt, int32_t *factor_list, int32_t num, int32_t max_num);
  static void GetNearestFactor(int32_t base, int32_t &factor);
  static void GetTwoFactors(int32_t *res, int32_t base, int32_t dim, int32_t max_num,
                            int32_t min_num, int32_t cur_factor, int32_t default_num);
};

}; // namespace optiling::conv2d_dw

#endif