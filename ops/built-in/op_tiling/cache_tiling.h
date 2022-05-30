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
 * \file cache_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_CACHE_TILING_H
#define OPS_BUILT_IN_OP_TILING_CACHE_TILING_H

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

namespace optiling {
struct BatchmatmulCompileParas {
  bool format_a_nd = false;
  bool format_b_nd = false;
  bool binary_mode_flag = false;
  bool bias_flag = false;
  bool nd_flag = false;
  bool trans_a_flag = false;
  bool trans_b_flag = false;
  bool at_l1_flag = true;
  bool split_k_flag = false;
  float fused_double_operand_num = 0;
  float aub_double_num = 0;
  float bub_double_num = 0;
  int32_t l2_size = (1024 * 1024);
  int32_t core_num = 32;
};

struct BatchmatmulRunParas {
  bool b_have_batch = false;  // dim num > 2
  bool is_batch_matmul_mode = false;  // (BatchMatMulV2 or BatchMatMul) and (dynamic_mode == "dynamic_mknb")
  bool used_aligned_pattern = false;
  bool non_factor_k = false;
  int32_t m_32 = 1;
  int32_t k_32 = 1;
  int32_t n_32 = 1;
  int32_t batch_32 = 1;
  int32_t m_mapped = 1;
  int32_t k_mapped = 1;
  int32_t n_mapped = 1;
  int32_t batch_mapped = 1;
  int64_t m = 1;
  int64_t k = 1;
  int64_t n = 1;
  int64_t batch = 1;
  int64_t ori_shape_m = 1;
  int64_t ori_shape_k = 1;
  int64_t ori_shape_n = 1;
};

struct BatchmatmulParas {
  const BatchmatmulCompileParas *compile_params = nullptr;
  BatchmatmulRunParas *run_params = nullptr;
};

struct CoreStatus {
  int32_t batch = 1;
  int32_t m = 1;
  int32_t k = 1;
  int32_t n = 1;
  int32_t batch_dim = 1;
  int32_t m_dim = 1;
  int32_t n_dim = 1;
  int32_t k_dim = 1;
  int32_t kal1_factor = 1;
  int32_t kbl1_factor = 1;
};

struct BlockDimCalculator {
  int32_t batch = 1;
  int32_t m = 1;
  int32_t k = 1;
  int32_t n = 1;
  int32_t k_num = 1;
  int32_t k_bytes = 1;
  int32_t n_dim_factor = 1;
  int32_t batch_dim_factor = 1;
  int32_t m_dim_factor = 1;
  int32_t k_dim_factor = 1;
  int32_t min_load_size = 1;
  int32_t core_use = 1;
  int32_t tmp_core_use = 1;
  int32_t batch_idx = 0;
  int32_t n_idx = 0;
  int32_t batch_dim_cnt = 0;
  int32_t m_dim_cnt = 0;
  int32_t n_dim_cnt = 0;
  int32_t k_dim_cnt = 0;
  int32_t batch_factor_cnt = 0;
  int32_t m_factor_less_64_cnt = 0;
  int32_t m_factor_less_1024_cnt = 0;
  int32_t k_factor_less_64_cnt = 0;
  int32_t k_factor_less_1024_cnt = 0;
  int32_t n_factor_less_64_cnt = 0;
  int32_t n_factor_less_1024_cnt = 0;
  int32_t ori_amat_size = 0;
  int32_t ori_bmat_size = 0;
  int32_t amat_size = 0;
  int32_t bmat_size = 0;
  int32_t tmp_amat_size = 0;
  int32_t tmp_bmat_size = 0;
  int32_t tmp_load_size = 0;
  int32_t total_load_size = 0;
  int32_t* batch_dim_array;
  int32_t* m_dim_array;
  int32_t* n_dim_array;
  int32_t* k_dim_array;
  int32_t tmp_value = 0;
  int32_t final_value = 0;
  bool final_blocking_flag = false;
  bool init_flag = false;
};

struct L0Status {
  int32_t m_l0 = 1;
  int32_t n_l0 = 1;
  int32_t k_l0 = 1;
  int32_t db_l0a = 1;
  int32_t db_l0b = 1;
  int32_t db_l0c = 1;
  int32_t db_cub = 1;
  int32_t final_ml0 = 0;
  int32_t final_kl0 = 0;
  int32_t final_nl0 = 0;
  int32_t final_load_size = INT_MAX;
  float final_l0c_use = 0;
  int32_t final_mul = 0;
  int32_t final_mte1Loop = INT_MAX;
  int32_t max_mk = 1;
  int32_t max_nk = 1;
  int32_t max_mn = 1;
  int32_t max_axis_idx = 0;
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
    final_mte1Loop = INT_MAX;
  }
};

struct L0Factors {
  int32_t final_ml0 = 0;
  int32_t final_kl0 = 0;
  int32_t final_nl0 = 0;
  int32_t final_load_size = INT_MAX;
  float final_l0c_use = 0;
  int32_t final_mul = 0;
  int32_t final_mte1Loop = INT_MAX;
};

struct MKNParasCombo {
  int32_t parasCombo[9];
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
  int32_t load_size = 0;
  int32_t max_m_al1 = 1;
  int32_t max_n_bl1 = 1;
  int32_t max_k_al1 = 1;
  int32_t max_k_bl1 = 1;
  bool both_full_load = false;
  bool al1_full_load = false;
  bool bl1_full_load = false;
  void SetStatus(const int32_t *tmp_l1_factors)
  {
    this->kal1_16 = tmp_l1_factors[0];
    this->kbl1_16 = tmp_l1_factors[1];
    this->m_al1 = tmp_l1_factors[2]; // 2 means m_al1 factor index
    this->n_bl1 = tmp_l1_factors[3]; // 3 means n_bl1 factor index
    this->db_al1 = tmp_l1_factors[4]; // 4 means db_al1 factor index
    this->db_bl1 = tmp_l1_factors[5]; // 5 means db_bl1 factor index
  }
};

struct UbStatus {
  int32_t k_aub = 1;
  int32_t m_aub = 1;
  int32_t db_aub = 1;
  int32_t k_bub = 1;
  int32_t n_bub = 1;
  int32_t db_bub = 1;
  int32_t n_cub = 1;
  int32_t db_cub = 1;
  int32_t max_dma_size = 0;
  int32_t min_dma_size = 0;
  int32_t min_load_size = 0;
  int32_t aub_size = 0;
  int32_t bub_size = 0;
  int32_t cub_size = 0;
  int32_t aub_multi_flag = 0;
  int32_t bub_multi_flag = 0;
  int32_t a_align_value = 1;
  int32_t b_align_value = 1;
  int32_t aub_bank_size = 0;
  int32_t bub_bank_size = 0;
  int32_t aub_align_bound = 0;
  int32_t bub_align_bound = 0;
  int32_t ub_rest_size = 0;
  int32_t safe_ub_rest_size = 0;
  // Ten tiling candidates for AUb. Two means tiling candidates results for M_aub and K_aub.
  int32_t aub_results[10][2] = {0};
  int32_t aub_cnt = 0;
  // Two tiling candidates for BUb. The second Two means tiling candidates results for K_bub and N_bub.
  int32_t bub_results[2][2] = {0};
  int32_t bub_cnt = 0;
  int32_t cub_dtype_multi = 1;
  bool cub_reuse_aub_flag = false;
  bool cub_reuse_bub_flag = false;
  bool a_bank_conflict = false;
  bool b_bank_conflict = false;
};

struct SingleCoreStatus {
  L0Status l0Status;
  L1Status l1Status;
  UbStatus ubStatus;
};

class Tiling {
public:
  std::string tiling_id;
  int32_t n_cub = 1;
  int32_t db_cub = 1;
  int32_t m_l0 = 1;
  int32_t k_l0 = 1;
  int32_t n_l0 = 1;
  int32_t batch_dim = 1;
  int32_t n_dim = 1;
  int32_t m_dim = 1;
  int32_t k_dim = 1;
  int32_t kal1_16 = 1;
  int32_t kbl1_16 = 1;
  int32_t kal1_factor = 1;
  int32_t kbl1_factor = 1;
  int32_t m_al1 = 1;
  int32_t n_bl1 = 1;
  int32_t db_al1 = 1;
  int32_t db_bl1 = 1;
  int32_t k_aub = 1;
  int32_t m_aub = 1;
  int32_t db_aub = 1;
  int32_t k_bub = 1;
  int32_t n_bub = 1;
  int32_t db_bub = 1;
  int32_t k_org_dim = 1;
  int32_t db_l0c = 1;
  int32_t aub_multi_flag = 0;
  int32_t bub_multi_flag = 0;
  int32_t a_align_value = 1;
  int32_t b_align_value = 1;
  int32_t aub_align_bound = 0;
  int32_t bub_align_bound = 0;
  int32_t min_kl1_cmp_kl0 = 0;
  int32_t al1_attach_flag = 0;
  int32_t bl1_attach_flag = 0;
  int32_t abkl1_attach_flag = 0;
  bool al1_full_load = false;
  bool bl1_full_load = false;
  Tiling() = default;
  void SetParams(const CoreStatus& coreStatus, const L0Status& l0Status, const L1Status& l1Status,
                 const UbStatus& ubStatus, const BatchmatmulParas& params);
  void SetAttachFlag();
  void GetTilingId(const BatchmatmulParas& params);
  ~Tiling() = default;
};

void GenTiling(const std::string &op_type, const BatchmatmulCompileParas &compile_params,
               BatchmatmulRunParas &run_params, Tiling &tiling, std::string &tilingId);
}; // namespace optiling

#endif