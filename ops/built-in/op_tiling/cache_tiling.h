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
#ifndef OPS_BUILT_IN_OP_TILING_FORMULA_TILING_H_
#define OPS_BUILT_IN_OP_TILING_FORMULA_TILING_H_

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

struct BatchmatmulParas {
  int64_t m = 1;
  int64_t k = 1;
  int64_t n = 1;
  int64_t batch = 1;
  int64_t ori_shape_M = 1;
  int64_t ori_shape_K = 1;
  int64_t ori_shape_N = 1;
  std::string format_a;
  std::string format_b;
};

struct L2Status {
  int64_t batch = 1;
  int64_t m = 1;
  int64_t k = 1;
  int64_t n = 1;
  int64_t batch_dim = 1;
  int64_t m_dim = 1;
  int64_t n_dim = 1;
};

struct BlockDimCalculator {
  int64_t batch = 1;
  int64_t m = 1;
  int64_t k = 1;
  int64_t n = 1;
  int64_t k_num = 1;
  int64_t k_bytes = 1;
  int64_t n_dim_factor = 1;
  int64_t batch_dim_factor = 1;
  int64_t m_dim_factor = 1;
  int64_t min_load_size = 1;
  int64_t core_use = 1;
  int64_t i_idx = 0;
  int64_t j_idx = 0;
  int64_t batch_dim_cnt = 0;
  int64_t m_dim_cnt = 0;
  int64_t n_dim_cnt = 0;
  int64_t ori_amat_size = 0;
  int64_t ori_bmat_size = 0;
  int64_t amat_size = 0;
  int64_t bmat_size = 0;
  int64_t tmp_amat_size = 0;
  int64_t tmp_bmat_size = 0;
  int64_t tmp_load_size = 0;
  int64_t total_load_size = 0;
  int64_t* batch_dim_array;
  int64_t* m_dim_array;
  int64_t* n_dim_array;
  int64_t tmp_value = 0;
  int64_t final_value = 0;
  bool final_blocking_flag = false;
};

struct L0Status {
  int64_t m_l0 = 1;
  int64_t n_l0 = 1;
  int64_t k_l0 = 1;
  int64_t db_l0a = 1;
  int64_t db_l0b = 1;
  int64_t db_l0c = 1;
  int64_t db_cub = 1;
  int64_t final_ml0 = 0;
  int64_t final_kl0 = 0;
  int64_t final_nl0 = 0;
  int64_t final_load_size = LONG_LONG_MAX;
  int64_t final_l0c_use = 0;
  int64_t final_mul = 0;
  int64_t final_mte1Loop = LONG_LONG_MAX;
  int64_t max_mk = 1;
  int64_t max_nk = 1;
  int64_t max_mn = 1;
  int64_t max_axis_idx = 0;
  int64_t max_axis_num = 0;
  int64_t max_axis_pnt = 1;
  void SetInitLoadStatus()
  {
    final_ml0 = 0;
    final_kl0 = 0;
    final_nl0 = 0;
    final_load_size = LLONG_MAX;
    final_l0c_use = 0;
    final_mul = 0;
    final_mte1Loop = LLONG_MAX;
  }
};

struct MKNParasCombo {
  int64_t parasCombo[9];
};

struct L1Status {
  int64_t kal1_16 = 1;
  int64_t kbl1_16 = 1;
  int64_t m_al1 = 1;
  int64_t n_bl1 = 1;
  int64_t db_al1 = 1;
  int64_t db_bl1 = 1;
  int64_t al1_size = 0;
  int64_t bl1_size = 0;
  int64_t al1_times = 1;
  int64_t bl1_times = 1;
  int64_t all_times = 1;
  int64_t load_size = 0;
  int64_t max_m_al1 = 1;
  int64_t max_n_bl1 = 1;
  int64_t max_k_al1 = 1;
  int64_t max_k_bl1 = 1;
  bool both_full_load = false;
  bool al1_full_load = false;
  bool bl1_full_load = false;
  void SetStatus(int64_t kal1_16_input, int64_t kbl1_16_input, int64_t m_al1_input, int64_t n_bl1_input,
                 int64_t db_al1_input, int64_t db_bl1_input)
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
  int64_t k_aub = 1;
  int64_t m_aub = 1;
  int64_t db_aub = 1;
  int64_t n_cub = 1;
  int64_t db_cub = 1;
};

class Tiling {
public:
  std::map<std::string, std::vector<int64_t>> mParam;
  std::map<std::string, std::map<std::string, int64_t>> mPingpongBuff;
  std::map<std::string, int64_t> pingpong;
  std::string tiling_id;
  int64_t n_cub = 1;
  int64_t db_cub = 1;
  int64_t m_l0 = 1;
  int64_t k_l0 = 1;
  int64_t n_l0 = 1;
  int64_t batch_dim = 1;
  int64_t n_dim = 1;
  int64_t m_dim = 1;
  int64_t kal1_16 = 1;
  int64_t kbl1_16 = 1;
  int64_t m_al1 = 1;
  int64_t n_bl1 = 1;
  int64_t db_al1 = 1;
  int64_t db_bl1 = 1;
  int64_t k_aub = 1;
  int64_t m_aub = 1;
  int64_t db_aub = 1;
  int64_t k_org_dim = 1;
  int64_t db_l0c = 1;
  Tiling() = default;
  void SetDoubleBufferParams(bool minKl1CmpKl0, std::map<std::string, int64_t> dbFlag);
  void SetParams(const L2Status& l2Status, const L0Status& l0Status, const L1Status& l1Status,
                 const UbStatus& ubStatus);
  void SetAttachFlag();
  void GetTilingId();
  ~Tiling() = default;
};

void GetFactors(int64_t* cnt, int64_t* factorList, const int64_t& num,
                const int64_t& maxNum);
void GetTwoFactors(int64_t* res, const int64_t& base, const int64_t& dim,
                   const int64_t& maxNum);
void GetNearestFactor(const int64_t& base, int64_t& factor);
void BL1FullLoadBlock(const L2Status& l2Status, BlockDimCalculator& blockDimCalculator, int64_t& n0);
void AL1FullLoadBlock(const L2Status& l2Status, BlockDimCalculator& blockDimCalculator, int64_t& m0);
void NeitherFullLoadBlock(const L2Status& l2Status, BlockDimCalculator& blockDimCalculator,
                          const int64_t nFactorTwoCandidates[][2], const int64_t mFactorTwoCandidates[][2],
                          const int64_t& nFactor, const int64_t& mFactor);
int64_t GetBlockDim(const std::string& op_type, const BatchmatmulParas& params, L2Status& l2Status,
                    const int64_t& coreNum);
void CheckUbDb(L0Status& l0Status);
int64_t GetLoadSize(const L2Status& l2Status, const L0Status& l0Status);
MKNParasCombo GetParasCombo(const int64_t& index, const int64_t& blockValue);
void GetFinalMkn(L0Status& l0Status, const L2Status& l2Status);
void GetL0StatusFromParasCombo(L0Status& l0Status, int64_t* parasCombo);
void SetResFactors(int64_t* resFactors, const L0Status& l0Status);
void GetL0FactorsCand(int64_t *resFactors, const L2Status &l2Status, L0Status &l0Status,
                      int64_t *parasCombo, int64_t sizeofParasCombo);
void GetL0Factors(const std::string& op_type, const L2Status& l2Status, const int64_t& blockValue,
                  L0Status& l0Status);
int64_t GetL1Size(const L1Status& l1Status, const L0Status& l0Status);
void CheckSpecialTemplate(const std::string& op_type, const L2Status& l2Status, const L0Status& l0Status,
                          L1Status& l1Status, const UbStatus& ubStatus);
void L1StatusBothFullLoad(const L2Status& l2Status, const L0Status& l0Status, L1Status& l1Status,
                          int64_t res[][7]);
void L1StatusAl1FullLoad(const L2Status& l2Status, const L0Status& l0Status, L1Status& l1Status,
                         int64_t res[][7]);
void L1StatusBl1FullLoad(const L2Status& l2Status, const L0Status& l0Status, L1Status& l1Status,
                         int64_t res[][7]);
void NeitherFullLoadDb(const L2Status& l2Status, const L0Status& l0Status, L1Status& l1Status,
                       const int64_t& kbl1Db);
void NeitherFullLoadMN(const L2Status& l2Status, const L0Status& l0Status, L1Status& l1Status,
                       const BatchmatmulParas& params);
void NeitherFullLoadK(const L2Status& l2Status, const L0Status& l0Status, L1Status& l1Status,
                      const BatchmatmulParas& params);
void L1StatusNeitherFullLoad(const L2Status& l2Status, const BatchmatmulParas& params,
                             const L0Status& l0Status, L1Status& l1Status, int64_t res[][7]);
void GetL1Factors(const std::string& op_type, const BatchmatmulParas& params, const L2Status& l2Status,
                  const L0Status& l0Status, L1Status& l1Status);
void GetUbFactors(const std::string& op_type, const L0Status& l0Status, UbStatus& ubStatus);
void GenTiling(const std::string& op_type, const BatchmatmulParas& params, Tiling& tiling, std::string& tilingId);
}; // namespace optiling

#endif
