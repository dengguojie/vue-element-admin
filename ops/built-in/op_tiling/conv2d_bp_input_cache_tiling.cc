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
 * \file conv2d_bp_input_cache_tiling.cc
 * \brief function of cacheTiling
 */
#include <algorithm>
#include <cmath>

#include "conv2d_bp_input_cache_tiling.h"
using namespace std;

namespace optiling {
static const int32_t kL1Size = (1024 * 1024);
static const int32_t kL0cSize = (256 * 1024);
static const int32_t kUbSize = 262000;
static const int32_t kBlockSize = 16;
static const int32_t kDecimal = 10;
static const int32_t kDbOn = 2;
static const int32_t kDbOff = 1;
static const int32_t kFrontUbFusionMulti = 2;
static const int32_t kAfterUbFusionMulti = 2;
static const int32_t khnumWNoDivided = 2;
static const float kFloatZero = 0.0f;
static const int32_t kAttachFlagZero = 0;
static const int32_t kAttachFlagOne = 1;
static const int32_t kAttachFlagTwo = 2;
static const int32_t kFp16Bytes = 2;
static const int32_t kFp32Bytes = 4;
static const int32_t kMmadComputeOneUs = 1000;
static const int32_t kMinCoreNum = 1;
static const double kLoadSizeThreshold = 0.00001;
static const int32_t kM2MaxSize = 1024;
static const int32_t kM2Size = 16;
static const int32_t kL0aSize = (64 * 1024);
static const int32_t kL0bSize = (64 * 1024);
static const int32_t kC0 = 16;
static const int32_t kM0N0OptimalNode = 11;
static const int32_t kL0cNzSize = 128;
static const int32_t kL0aNzSize = 64;
static const int32_t kL0bNzSize = 64;
static const int32_t kIdxTwo = 2;
static const int32_t kIdxThree = 3;
static const int32_t kIdxFour = 4;
static const int32_t kNumTwo = 2;
static const int32_t kL0CutOffPointNum = 6;

struct FactorArray {
  int32_t min_kl1_dim = 1;
  int32_t kn_factors[3L];
  int32_t size_para[3L];
};

inline bool EqualWith(const float& l_value, const float& r_value) {
  return std::fabs(l_value - r_value) <= std::numeric_limits<float>::epsilon();
}

inline int32_t CeilDivision(const int32_t& num1, const int32_t& num2) {
  if (num2 == 0) {
    return 0;
  }
  return (num1 + num2 - 1) / num2;
}

inline int32_t CeilAlign(const int32_t& num1, const int32_t& num2) {
  return CeilDivision(num1, num2) * num2;
}

inline bool IsSatisfyM(const DxParas &params, const int32_t &m_dim) {
  CHECK_OP_FUNC(m_dim == 0, return false, "m_dim is 0");
  int32_t m1 = (params.h * params.w + kBlockSize - 1) / kBlockSize;
  int32_t m_single_core_size_tmp = (m1 + m_dim - 1) / m_dim;
  int32_t m_single_core_size_16 = m_single_core_size_tmp * kBlockSize;
  int32_t m_core_remainder = params.hw % m_single_core_size_16;
  int32_t m_tail_core_size_tmp = m_core_remainder == 0 ? m_single_core_size_16 : m_core_remainder;
  bool satisfy_constraint_m = !params.dx_no_overlap_condition_1 ||
                              (params.dx_no_overlap_condition_1 && m_single_core_size_tmp >= kNumTwo &&
                               m_tail_core_size_tmp >= kBlockSize);
  return satisfy_constraint_m;
}

inline bool IsSatisfyN(const DxParas &params, const int32_t &n_dim) {
  CHECK_OP_FUNC(n_dim == 0, return false, "n_dim is 0");
  int32_t n_single_core_size_tmp = params.c1 / n_dim;
  bool satisfy_constraint_n = !params.dx_no_overlap_condition_2 || n_single_core_size_tmp >= kNumTwo;
  return satisfy_constraint_n;
}

bool MdimTune(const DxParas &params, const int32_t &m1, const int32_t &n_dim_factor, const int32_t &m_dim_factor,
              Tiling &tiling) {
  if (n_dim_factor == 0 || m_dim_factor == 0) {
    tiling.m_dim = 0;
    return false;
  }
  int32_t size_nk = params.co1 * params.kh * params.kw * (params.c1 / n_dim_factor);
  if (size_nk > kMmadComputeOneUs) {
    return false;
  }
  int32_t min_m = CeilDivision(kMmadComputeOneUs, size_nk);
  CHECK_OP_FUNC(min_m == 0, return false, "min_m is 0");
  bool min_hw = ((m1 + m_dim_factor - 1) / m_dim_factor) <= min_m;
  if (min_hw && m_dim_factor > 1) {
    int32_t m_dim_tmp = max(m1 / min_m, kMinCoreNum);
    m_dim_tmp = min(m_dim_tmp, params.core_num);
    if (IsSatisfyM(params, m_dim_tmp)) {
      tiling.m_dim = m_dim_tmp;
      return min_hw;
    }
    return false;
  }
  return min_hw;
}

bool GenNearestFactor(const int32_t& factor, const int32_t& dim, int32_t factor_optional[]) {
  int32_t cur_factor = min(factor + 1, dim);
  CHECK_OP_FUNC(cur_factor == 0, return false, "cur_factor is 0");
  while (dim % cur_factor != 0) {
    cur_factor++;
  }
  factor_optional[0] = cur_factor;
  cur_factor = factor;
  CHECK_OP_FUNC(cur_factor == 0, return false, "cur_factor is 0");
  while (dim % cur_factor != 0) {
    cur_factor--;
  }
  factor_optional[1] = cur_factor;
  return true;
}

void GetFactors(const int32_t& bm_dim, vector<int32_t>& bm_dim_opt) {
  // get all factors of num which smaller or equal to maxNum
  for (int32_t i = 1; i <= bm_dim / kNumTwo; i++) {
    if (bm_dim % i == 0) {
      bm_dim_opt.push_back(i);
    }
  }
  bm_dim_opt.push_back(bm_dim);
}

int32_t CalLoopNum(const DxParas &params, const int32_t &m1, const int32_t &m_dim, const int32_t &n_dim,
                   const int32_t &batch_dim) {
  int32_t l0c_size = kL0cNzSize;
  int32_t m_single_core_size = CeilDivision(m1, m_dim);
  if (m_single_core_size == 0 || n_dim == 0 || batch_dim == 0) {
    return 0;
  }
  l0c_size = min(m_single_core_size * (params.c1 / n_dim), l0c_size);
  if (l0c_size == 0) {
    return 0;
  }
  int32_t loop_num = (params.batch / batch_dim) * m_single_core_size * (params.c1 / n_dim) / l0c_size;
  // special scence n_single_core_size is 1 or m_single_core_size is 1
  if ((params.c1 / n_dim) == 1) {
    int32_t l0a_size = min(m_single_core_size, kL0aNzSize);
    int32_t m_single_ml0_num = CeilDivision(m_single_core_size, l0a_size);
    if (m_single_ml0_num == 0) {
      return 0;
    }
    loop_num = (params.batch / batch_dim) * m_single_ml0_num * (params.c1 / n_dim);
  }
  return loop_num;
}

void GetFactors(int32_t dim_factor[], const int32_t &factor_max, const int32_t &factor_min,
                const int32_t &target_num, size_t &index) {
  for (int32_t temp = factor_max; temp >= factor_min; temp--) {
    if (target_num % temp != 0) {
      continue;
    }
    dim_factor[index++] = temp;
  }
}

bool GetBlockDim(const DxParas &params, const int32_t &core_num, Tiling &tiling) {
  // get batch_dim, m_dim and n_dim for single core
  // not support multi cores slicing along k dim
  // single core batch_dim, m_dim, n_dim is a factor of input batch, m, n
  int32_t m1 = (params.h * params.w + kBlockSize - 1) / kBlockSize;
  int32_t k2 = params.co1 * params.kh * params.kw;
  if (params.hw * params.cin < kBlockSize) {
    // no overlap condition 4
    tiling.batch_dim = 1;
    tiling.n_dim = 1;
    tiling.m_dim = 1;
    tiling.batch_single_core_size = params.batch;
    tiling.m_single_core_size = 1;
    tiling.n_single_core_size = 1;
    tiling.k_single_core_size = k2;
    return true;
  }
  if (params.batch * m1 * params.c1 < core_num) {
    CHECK_OP_FUNC(!MdimTune(params, m1, params.c1, m1, tiling) && tiling.m_dim == 0, return false, "ndim, mdim is 0");
    tiling.batch_dim = params.batch;
    tiling.n_dim = params.c1;
    tiling.batch_single_core_size = 1;
    tiling.m_single_core_size = (m1 + tiling.m_dim - 1) / tiling.m_dim;
    tiling.n_single_core_size = 1;
    tiling.k_single_core_size = k2;
    return true;
  }
  int64_t a_size =
      static_cast<int64_t>(params.batch * params.co1) * static_cast<int64_t>(params.ho * params.wo) * kBlockSize;
  if ((params.kh - 1) / params.stride_h <= 1) {
    a_size = static_cast<int64_t>(params.batch * params.co1) * static_cast<int64_t>(params.kh * params.kw) *
             static_cast<int64_t>(params.h * params.w) * kBlockSize;
  }
  int64_t b_size = static_cast<int64_t>(params.co1 * params.c1 * params.kh * params.kw) *
                   static_cast<int64_t>(kBlockSize * kBlockSize);
  // get n_dim cand
  int32_t n_dim_factor[32L] = {0};
  size_t idx_n = 0;
  GetFactors(n_dim_factor, core_num, 1, params.c1, idx_n);
  // get batch_dim cand
  int32_t batch_dim_factor[32L] = {0};
  size_t idx_batch = 0;
  GetFactors(batch_dim_factor, core_num, 1, params.batch, idx_batch);
  // initial block dim
  tiling.batch_dim = 1;
  tiling.n_dim = 1;
  tiling.m_dim = 1;
  int64_t min_load_size = static_cast<int64_t>(tiling.batch_dim * tiling.m_dim) * b_size +
                          static_cast<int64_t>(tiling.n_dim) * a_size;
  int32_t core_use = tiling.batch_dim * tiling.n_dim * tiling.m_dim;
  int32_t loop_num = CalLoopNum(params, m1, tiling.m_dim, tiling.n_dim, tiling.batch_dim);
  CHECK_OP_FUNC(loop_num == 0, return false, "loop_num is 0");
  for (size_t i = 0; i < idx_n; i++) {
    for (size_t j = 0; j < idx_batch; j++) {
      int32_t n_dim = n_dim_factor[i];
      int32_t batch_dim = batch_dim_factor[j];
      if (batch_dim * n_dim > core_num) {
        continue;
      }
      int32_t m_dim = min(core_num / (batch_dim * n_dim), m1);
      int32_t loop_num_temp = CalLoopNum(params, m1, m_dim, n_dim, batch_dim);
      if (loop_num_temp == 0) {
        continue;
      }
      int64_t load_size = static_cast<int64_t>(batch_dim * m_dim) * b_size + static_cast<int64_t>(n_dim) * a_size;
      int32_t core_use_temp = batch_dim * n_dim * m_dim;
      bool modify_dim = loop_num_temp < loop_num || (loop_num_temp == loop_num && core_use < core_use_temp) ||
                        (core_use == core_use_temp && load_size < min_load_size) ||
                        (core_use == core_use_temp && abs(load_size - min_load_size) < kLoadSizeThreshold &&
                         tiling.batch_dim * tiling.n_dim < batch_dim * n_dim);
      bool satisfy_constraint_m = IsSatisfyM(params, m_dim);
      bool satisfy_constraint_n = IsSatisfyN(params, n_dim);
      if (modify_dim && satisfy_constraint_m && satisfy_constraint_n) {
        min_load_size = load_size;
        tiling.batch_dim = batch_dim;
        tiling.n_dim = n_dim;
        tiling.m_dim = m_dim;
        core_use = tiling.batch_dim * tiling.n_dim * tiling.m_dim;
        loop_num = loop_num_temp;
      }
    }
  }
  tiling.m_single_core_size = (m1 + tiling.m_dim - 1) / tiling.m_dim;
  tiling.n_single_core_size = params.c1 / tiling.n_dim;
  tiling.k_single_core_size = k2;
  tiling.batch_single_core_size = params.batch / tiling.batch_dim;
  return true;
}

inline bool CheckL0Overflow(const int32_t &m0, const int32_t &n0, const int32_t &k0) {
  bool l0_invalid = (m0 * k0 * kFp16Bytes * kDbOn * kBlockSize * kBlockSize > kL0aSize) ||
                    (n0 * m0 * kFp32Bytes * kBlockSize * kBlockSize * kDbOn > kL0cSize) ||
                    (n0 * k0 * kFp16Bytes * kDbOn * kBlockSize * kBlockSize > kL0bSize);
  return !l0_invalid;
}

int32_t Gcd(int32_t param1, int32_t param2) {
  // get greatest common divisor of param1 and param2
  if (param1 < param2) {
    swap(param1, param2);
  }
  if (param2 == 0) {
    return 0;
  }
  if (param1 % param2 == 0) {
    return param2;
  } else {
    return Gcd(param2, param1 - param2);
  }
}

inline int32_t InitialKl1(int32_t &k_l0, int32_t &k_hw) {
  int32_t gcd_num = Gcd(k_l0, k_hw);
  if (gcd_num == 0 || k_hw == 0) {
    return 0;
  }
  int32_t lcm_num = k_l0 * k_hw / gcd_num;
  int32_t k_l1 = lcm_num / k_hw;
  return k_l1;
}

int32_t GetBl1Bound(const DxParas &params, const Tiling &tiling, int32_t tiling_factor[]) {
  int32_t idx = 0;
  int32_t k_bl1 = tiling_factor[idx++];
  int32_t n0 = tiling_factor[idx++];
  int32_t n_bl1 = tiling_factor[idx++];
  int32_t db_bl1 = tiling_factor[idx++];
  int32_t b_l1_size =
      k_bl1 * kBlockSize * params.kh * params.kw * n_bl1 * n0 * kBlockSize * db_bl1;
  if (CeilDivision(tiling.n_single_core_size, (n_bl1 * n0)) == 1 && k_bl1 == params.co1) {
    b_l1_size = params.co1 * kBlockSize * params.kh * params.kw * tiling.n_single_core_size * kBlockSize;
  }
  return b_l1_size;
}

int32_t GetAl1Bound(const DxParas &params, const Tiling &tiling, const int32_t tiling_factor[]) {
  int32_t idx = 0;
  int32_t k_al1 = tiling_factor[idx++];
  int32_t m0 = tiling_factor[idx++];
  int32_t m_al1 = tiling_factor[idx++];
  int32_t db_al1 = tiling_factor[idx++];
  int32_t h_num = (params.kh - 1) + m_al1 * m0 * kBlockSize / params.w + khnumWNoDivided;
  if (m_al1 * m0 * kBlockSize < params.w) {
    h_num = (params.kh - 1) + khnumWNoDivided;
  } else if (m_al1 * m0 * kBlockSize % params.w == 0) {
    h_num = (params.kh - 1) + m_al1 * m0 * kBlockSize / params.w;
  }
  int32_t a_l1_size = k_al1 * params.wo * params.stride_w * kC0 * h_num * db_al1;
  if (m_al1 * m0 == tiling.m_single_core_size && k_al1 == params.co1 && tiling.m_dim == 1) {
    int32_t hw_ceil_align = CeilAlign(params.ho * params.stride_h * params.wo * params.stride_w, kBlockSize);
    a_l1_size = k_al1 * kBlockSize * hw_ceil_align;
  }
  return a_l1_size;
}

bool CheckL1Overflow(const DxParas &params, const Tiling &tiling,
                     const int32_t &m0, const int32_t &n0, int32_t &k0) {
  int32_t l1_fp16_size = kL1Size / kFp16Bytes;
  int32_t m_al1 = 1;
  int32_t n_bl1 = 1;
  int32_t tiling_factor[4L] = {tiling.k_al1, n0, n_bl1, kDbOff};
  int32_t b_l1_size = GetBl1Bound(params, tiling, tiling_factor);
  int32_t tiling_al1_factor[4L] = {tiling.k_al1, m0, m_al1, kDbOff};
  int32_t a_l1_size = GetAl1Bound(params, tiling, tiling_al1_factor);
  return a_l1_size + b_l1_size <= l1_fp16_size;
}

bool CheckUbDb(const DxParas &params, const Tiling &tiling, const int32_t &m0) {
  int32_t aub_h = 1;
  int32_t aub_k = 1;
  int32_t cub_n = 1;
  int32_t ub_fp16_size = kUbSize / kFp16Bytes;
  int32_t loadin_size = aub_k * aub_h * params.wo * kBlockSize * params.stride_w;
  int32_t copyout_size = kAfterUbFusionMulti * cub_n * m0 * kBlockSize * kBlockSize;
  if (params.stride_h == 1 && params.stride_w == 1) {
    loadin_size = kFrontUbFusionMulti * aub_k * kBlockSize *
              ((aub_h * params.wo + params.kw - 1 + kBlockSize - 1) / kBlockSize) * kBlockSize;
    return loadin_size * tiling.db_aub + copyout_size * tiling.db_cub <= ub_fp16_size;
  }
  return loadin_size * tiling.db_aub + copyout_size * tiling.db_cub <= ub_fp16_size;
}

bool GetM0(const int32_t &n0, int32_t &k0, Tiling &tiling, const DxParas & params, int32_t &m0) {
  CHECK_OP_FUNC(k0 == 0 || n0 == 0, return false, "k0 or n0 is 0");
  m0 = min(min(kL0aNzSize / k0, kL0cNzSize / n0), tiling.m_single_core_size);
  int32_t k_hw = params.kh * params.kw;
  tiling.k_al1 = InitialKl1(k0, k_hw);
  CHECK_OP_FUNC(tiling.k_al1 == 0, return false, "Initial kl1 failed");
  int32_t m_tail_size = params.hw % (m0 * kBlockSize);
  bool update_m = (params.dx_no_overlap_condition_1 && m_tail_size != 0 && m_tail_size < kBlockSize);
  while (m0 > 0 && (!CheckL1Overflow(params, tiling, m0, n0, k0) || update_m)) {
    m0--;
    if (m0 == 0) {
      break;
    }
    m_tail_size = params.hw % (m0 * kBlockSize);
    update_m = (params.dx_no_overlap_condition_1 && m_tail_size != 0 && m_tail_size < kBlockSize);
  }
  return true;
}

bool GetL0FactorsGeneral(const DxParas &params, Tiling &tiling) {
  tiling.k_l0 = 1;
  tiling.m_l0 = 1;
  tiling.n_l0 = 1;
  int32_t n_l0_min = 1;
  if (params.dx_no_overlap_condition_2) {
    tiling.n_l0 = kNumTwo;
    n_l0_min = kNumTwo;
  }
  if (params.dx_no_overlap_condition_1) {
    tiling.m_l0 = kNumTwo;
  }
  // get n0 factor
  int32_t n0_factor[64L] = {0};
  size_t idx_n = 0;
  int32_t n_l0_max = min(kL0aNzSize, tiling.n_single_core_size);
  GetFactors(n0_factor, n_l0_max, n_l0_min, tiling.n_single_core_size, idx_n);
  // get k0 factor
  int32_t k0_factor[64L] = {0};
  size_t idx_k = 0;
  int32_t k_l0_max = min(kL0aNzSize, tiling.k_single_core_size);
  GetFactors(k0_factor, k_l0_max, 1, tiling.k_single_core_size, idx_k);
  int32_t m_single_ml0_num = CeilDivision(tiling.m_single_core_size, tiling.m_l0);
  int32_t n_single_nl0_num = CeilDivision(tiling.n_single_core_size, tiling.n_l0);
  int32_t min_load_size = m_single_ml0_num * tiling.n_single_core_size + n_single_nl0_num * tiling.m_single_core_size;
  int32_t max_mkn = 1;
  int32_t max_mk = 1;
  int32_t load_size = 1;
  int32_t mkn = 1;
  int32_t mk = 1;
  bool l0_invalid = true;
  bool l0_valid_res = false;
  bool l0_update = false;
  for (size_t i = 0; i < idx_n; i++) {
    for (size_t j = 0; j < idx_k; j++) {
      int32_t n0 = n0_factor[i];
      int32_t k0 = k0_factor[j];
      if (k0 == 0 || n0 == 0) {
        continue;
      }
      int32_t m0 = 0;
      CHECK_OP_FUNC(!GetM0(n0, k0, tiling, params, m0), return false, "get m0 failed");
      m_single_ml0_num = CeilDivision(tiling.m_single_core_size, m0);
      n_single_nl0_num = CeilDivision(tiling.n_single_core_size, n0);
      l0_invalid = m0 <= 0 || !CheckL0Overflow(m0, n0, k0) || !CheckUbDb(params, tiling, m0) ||
                   m_single_ml0_num == 0 || n_single_nl0_num == 0;
      if (l0_invalid) {
        continue;
      }
      load_size = m_single_ml0_num * tiling.n_single_core_size + n_single_nl0_num * tiling.m_single_core_size;
      mkn = m0 * n0 * k0;
      mk = max(m0, k0) / min(m0, k0);
      l0_update = (load_size < min_load_size) ||
                 ((load_size == min_load_size) && (mkn > max_mkn || (mkn == max_mkn && mk > max_mk)));
      if (l0_update) {
        tiling.m_l0 = m0;
        tiling.n_l0 = n0;
        tiling.k_l0 = k0;
        min_load_size = load_size;
        max_mkn = mkn;
        max_mk = mk;
        l0_valid_res = true;
      }
    }
  }
  return l0_valid_res;
}

bool GetL0Factors(const DxParas &params, Tiling &tiling) {
  // get m_l0, n_l0, k_l0 factor when singlecore m, n, k is know
  // m_l0, n_l0, k_l0 is a factor of single core m, n, k
  tiling.db_l0c = kDbOn;
  tiling.db_cub = kDbOn;
  tiling.db_aub = kDbOff;
  if (GetL0FactorsGeneral(params, tiling)) {
    return true;
  }
  tiling.db_cub = kDbOff;
  return GetL0FactorsGeneral(params, tiling);
}

bool GetInitialL1(const DxParas &params, Tiling &tiling, int32_t &min_kl1_dim) {
  tiling.init_db_al1 = kDbOn;
  tiling.init_db_bl1 = kDbOn;
  tiling.m_al1 = 1;
  tiling.n_bl1 = 1;
  int32_t k_hw = (params.kh * params.kw);
  min_kl1_dim = InitialKl1(tiling.k_l0, k_hw);
  CHECK_OP_FUNC(min_kl1_dim == 0, return false, "initial kl1 failed");
  tiling.k_al1 = min_kl1_dim;
  tiling.k_bl1 = tiling.k_al1;
  return true;
}

int32_t GetAl1MExtent(const int32_t& al1_m, const DxParas &params) {
  if (al1_m == 0) {
    return 0;
  }
  int32_t al1_h_small = (params.w % al1_m == 0) ? params.kh : params.kh + 1;
  int32_t al1_h_large =
      (al1_m % params.w == 0) ? params.kh + (al1_m / params.w) - 1 : params.kh + (al1_m / params.w) + 1;
  int32_t al1_h = (al1_m < params.w) ? al1_h_small : al1_h_large;
  return al1_h;
}

bool Gethnum(const DxParas &params, const Tiling &tiling, const int32_t kn_factors[], const int32_t &h2,
             int32_t l1_para[]) {
  int32_t l1_fp16_size = kL1Size / kFp16Bytes;
  size_t idx = 0;
  int32_t k_bl1 = kn_factors[idx++];
  int32_t k_al1 = kn_factors[idx++];
  int32_t nbl1 = kn_factors[idx++];
  int32_t m_size = (l1_fp16_size -
                    k_bl1 * params.kh * params.kw * nbl1 * tiling.n_l0 * kBlockSize * kC0 * tiling.init_db_bl1) /
                   tiling.init_db_al1;
  if (k_bl1 == params.co1 && static_cast<int32_t>(ceil(static_cast<double>(tiling.n_single_core_size) /
                                                    static_cast<double>(nbl1 * tiling.n_l0))) == 1) {
    m_size = (l1_fp16_size - k_bl1 * params.kh * params.kw * tiling.n_single_core_size * kBlockSize * kC0) /
             tiling.init_db_al1;
  }
  int32_t h_num = h2 < params.kh + 1 ? params.kh + 1 : m_size / (k_al1 * params.wo * params.stride_w * kC0);
  if (k_al1 == params.co1) {
    if (h_num >= params.kh + 1) {
      int32_t m_1;
      if (h_num >= params.kh + khnumWNoDivided) {
        m_1 = max(((h_num - params.kh - 1 + 1) * params.w - 1) / (tiling.m_l0 * kBlockSize), 1);
      } else {
        m_1 = max((params.kh + 1 - params.kh) * params.w / (tiling.m_l0 * kBlockSize), 1);
      }
      int32_t m_1_factor[2L] = {0, 0};
      CHECK_OP_FUNC(
          !GenNearestFactor(m_1,
                            static_cast<int32_t>(ceil(static_cast<double>(params.h * params.w) /
                                                      static_cast<double>(tiling.m_dim * tiling.m_l0 * kBlockSize))),
                            m_1_factor),
          return false, "get m_1 failed");
      m_1 = m_1_factor[1];
      h_num = GetAl1MExtent(m_1 * tiling.m_l0 * kBlockSize, params);
      CHECK_OP_FUNC(h_num == 0, return false, "get h_num failed");
      l1_para[0] = h_num;
      l1_para[1] = m_1;
    } else {
      l1_para[0] = 0;
      l1_para[1] = 0;
    }
  } else {
    h_num = params.kh - 1 + (tiling.m_l0 * kBlockSize / params.w) + khnumWNoDivided;
    if (tiling.m_l0 * kBlockSize < params.w) {
      h_num = params.kh - 1 + khnumWNoDivided;
    }
    l1_para[0] = h_num;
    l1_para[1] = 1;
  }
  if (tiling.m_l0 * l1_para[1] == tiling.m_single_core_size && k_al1 == params.co1) {
    l1_para[0] = h2 * params.stride_h;
  }
  return true;
}

void GetMinloadSize(const FactorArray &factor_size, Tiling &tiling, const DxParas &params,
                    const int32_t &load_h, int32_t *db_size) {
  size_t idx = 0;
  int32_t k_bl1 = factor_size.kn_factors[idx++];
  int32_t k_al1 = factor_size.kn_factors[idx++];
  int32_t nbl1 = factor_size.kn_factors[idx++];
  idx = 0;
  int32_t h2 = factor_size.size_para[idx++];
  int32_t a_size = factor_size.size_para[idx++];
  int32_t b_size = factor_size.size_para[idx++];
  int32_t db_bl1 = tiling.init_db_bl1;
  int32_t db_al1 = tiling.init_db_al1;
  if (static_cast<int32_t>(
          ceil(static_cast<double>(tiling.n_single_core_size) / static_cast<double>(nbl1 * tiling.n_l0))) == 1 &&
      k_bl1 == params.co1) {
    db_bl1 = 1;
    db_size[0] = a_size + b_size;
    db_size[1] = db_al1;
    db_size[kIdxTwo] = db_bl1;
    return;
  }
  if (static_cast<int32_t>(ceil(static_cast<double>(h2) / static_cast<double>(load_h))) == 1 &&
      k_al1 == params.co1) {
    db_al1 = 1;
    db_size[0] = a_size + b_size;
    db_size[1] = db_al1;
    db_size[kIdxTwo] = db_bl1;
    return;
  }
  db_size[0] = static_cast<int32_t>(ceil(static_cast<double>(h2) / static_cast<double>(load_h))) * b_size +
               static_cast<int32_t>(
                   ceil(static_cast<double>(tiling.n_single_core_size) / static_cast<double>(nbl1 * tiling.n_l0))) *
                   a_size;
  db_size[1] = db_al1;
  db_size[kIdxTwo] = db_bl1;
  return;
}

bool CheckL1Size(const int32_t kn_factors[], const int32_t m_h[],
                 const Tiling &tiling, const DxParas &params) {
  size_t idx = 0;
  int32_t k_bl1 = kn_factors[idx++];
  int32_t k_al1 = kn_factors[idx++];
  int32_t nbl1 = kn_factors[idx++];
  idx = 0;
  int32_t m_1 = m_h[idx++];
  int32_t db_al1_end = m_h[idx++];
  int32_t db_bl1_end = m_h[idx++];
  int32_t tiling_factor[4L] = {k_bl1, tiling.n_l0, nbl1, db_bl1_end};
  int32_t b_size = GetBl1Bound(params, tiling, tiling_factor) * kFp16Bytes;
  int32_t tiling_al1_factor[4L] = {k_al1, tiling.m_l0, m_1, db_al1_end};
  int32_t a_size = GetAl1Bound(params, tiling, tiling_al1_factor) * kFp16Bytes;
  return a_size + b_size <= kL1Size;
}

bool GetL1FactorsOpti(const FactorArray &factor_size,
                      const DxParas &params, Tiling &tiling, int32_t &min_load_size, bool &first_flag) {
  size_t idx = 0;
  int32_t k_bl1 = factor_size.kn_factors[idx++];
  int32_t k_al1 = factor_size.kn_factors[idx++];
  int32_t nbl1 = factor_size.kn_factors[idx++];
  idx = 0;
  int32_t h2 = factor_size.size_para[idx++];
  if (nbl1 > 1 && k_bl1 < params.co1) {
    return true;
  }
  bool modify_l1 = (k_bl1 % k_al1 == 0 || k_al1 % k_bl1 == 0) &&
                   (k_bl1 >= factor_size.min_kl1_dim && k_al1 >= factor_size.min_kl1_dim) &&
                   (k_bl1 * params.kh * params.kw) % tiling.k_l0 == 0 &&
                   (k_al1 * params.kh * params.kw) % tiling.k_l0 == 0;
  if (modify_l1) {
    int32_t l1_para[2L];
    CHECK_OP_FUNC(!Gethnum(params, tiling, factor_size.kn_factors, h2, l1_para), return false, "get h_num failed");
    int32_t h_num = l1_para[0];
    int32_t m_1 = l1_para[1];
    if (h_num != 0 && m_1 != 0) {
      int32_t load_h = static_cast<int32_t>(ceil(static_cast<double>(h_num) / static_cast<double>(params.stride_h)));
      int32_t db_size[3L];
      GetMinloadSize(factor_size, tiling, params, load_h, db_size);
      int32_t load_size = db_size[0];
      int32_t db_al1_end = db_size[1];
      int32_t db_bl1_end = db_size[kIdxTwo];
      int32_t m_h[4L] = {m_1, db_al1_end, db_bl1_end};
      modify_l1 = CheckL1Size(factor_size.kn_factors, m_h, tiling, params) &&
                  (min_load_size > load_size ||
                   (static_cast<double>(abs(min_load_size - load_size)) < kLoadSizeThreshold &&
                    tiling.k_al1 < k_al1) ||
                   (static_cast<double>(abs(min_load_size - load_size)) < kLoadSizeThreshold &&
                    tiling.k_al1 == k_al1 && tiling.k_bl1 < k_bl1) || first_flag);
      if (modify_l1) {
        tiling.m_al1 = m_1;
        tiling.k_al1 = k_al1;
        tiling.k_bl1 = k_bl1;
        tiling.n_bl1 = nbl1;
        tiling.h_num = h_num;
        tiling.db_al1 = db_al1_end;
        tiling.db_bl1 = db_bl1_end;
        min_load_size = load_size;
        first_flag = false;
        tiling.update_l1 = true;
      }
    }
  }
  return true;
}

bool GetL1Factors(const DxParas &params, Tiling &tiling) {
  // get m_al1, n_bl1, kal1_16, kbl1_16 factors when L0, singlecore factor is know
  // get al1, bl1 double buffer factors
  struct FactorArray factor_size;
  tiling.db_al1 = kDbOn;
  tiling.db_bl1 = kDbOn;
  CHECK_OP_FUNC(!GetInitialL1(params, tiling, factor_size.min_kl1_dim), return false, "initial l1 failed");
  int32_t l1_fp16_size = kL1Size / kFp16Bytes;
  int32_t h_num;
  if (tiling.m_al1 * tiling.m_l0 * kBlockSize < params.w) {
    h_num = (params.kh - 1) + khnumWNoDivided;
  } else if ((tiling.m_al1 * tiling.m_l0 * kBlockSize) % params.w == 0) {
    h_num = (params.kh - 1) + (tiling.m_al1 * tiling.m_l0 * kBlockSize) / params.w;
  } else {
    h_num = (params.kh - 1) + (tiling.m_al1 * tiling.m_l0 * kBlockSize) / params.w + khnumWNoDivided;
  }
  int32_t b_l1_size =
      tiling.k_bl1 * params.kh * params.kw * tiling.n_bl1 * tiling.n_l0 * kC0 * tiling.init_db_bl1 * kBlockSize;
  int32_t a_l1_size = tiling.k_al1 * params.wo * params.stride_w * kC0 * h_num * tiling.init_db_al1;
  if (b_l1_size + a_l1_size > l1_fp16_size) {
    tiling.init_db_al1 = 1;
  }
  a_l1_size = tiling.k_al1 * params.wo * params.stride_w * kC0 * h_num * tiling.init_db_al1;
  if (b_l1_size + a_l1_size > l1_fp16_size) {
    tiling.init_db_bl1 = 1;
  }
  int32_t b_size = params.co1 * params.kh * params.kw * kC0 * kC0 * tiling.n_single_core_size;
  int32_t h2 = static_cast<int32_t>(
      ceil((ceil(static_cast<double>(tiling.m_single_core_size * kBlockSize) / static_cast<double>(params.w)) +
            static_cast<double>(params.kh - 1)) /
           static_cast<double>(params.stride_h)));
  h2 = min(h2, params.ho);
  int32_t a_size = h2 * params.co1 * params.wo * kC0;
  int32_t real_h = static_cast<int32_t>(ceil(static_cast<double>(h_num) / static_cast<double>(params.stride_h)));
  int32_t min_load_size =
      static_cast<int32_t>(ceil(static_cast<double>(h2) / static_cast<double>(real_h))) * b_size +
      static_cast<int32_t>(ceil(static_cast<double>(tiling.n_single_core_size) / static_cast<double>(tiling.n_l0))) *
          a_size;
  vector<int32_t> k_factors;
  GetFactors(params.co1, k_factors);
  vector<int32_t> nl1_factors;
  GetFactors(tiling.n_single_core_size / tiling.n_l0, nl1_factors);
  bool first_flag = true;
  for (auto &k_bl1 : k_factors) {
    for (auto &k_al1 : k_factors) {
      for (auto &nbl1 : nl1_factors) {
        factor_size.kn_factors[0] = k_bl1;
        factor_size.kn_factors[1] = k_al1;
        factor_size.kn_factors[2L] = nbl1;
        factor_size.size_para[0] = h2;
        factor_size.size_para[1] = a_size;
        factor_size.size_para[2L] = b_size;
        CHECK_OP_FUNC(!GetL1FactorsOpti(factor_size, params, tiling, min_load_size, first_flag), return false,
                      "get L1Factor failed");
      }
    }
  }
  int32_t max_kl1 = max(tiling.k_al1, tiling.k_bl1);
  int32_t min_kl1 = min(tiling.k_al1, tiling.k_bl1);
  CHECK_OP_FUNC(min_kl1 == 0, return false, "k_al1 is zero");
  CHECK_OP_FUNC(max_kl1 == 0, return false, "k_bl1 is zero");
  tiling.min_kl1_div_kl0 = min_kl1 * params.kh * params.kw / tiling.k_l0;
  tiling.max_kl1_div_min_kl1 = max_kl1 / min_kl1;
  tiling.k_div_max_kl1 = params.co1 / max_kl1;
  int32_t tiling_factor[4L] = {tiling.k_bl1, tiling.n_l0, tiling.n_bl1, kDbOff};
  tiling.bl1_bound = GetBl1Bound(params, tiling, tiling_factor);
  int32_t tiling_al1_factor[4L] = {tiling.k_al1, tiling.m_l0, tiling.m_al1, kDbOff};
  tiling.al1_bound = GetAl1Bound(params, tiling, tiling_al1_factor);
  return true;
}

void GetAubM(const int32_t &aub_size, const DxParas &params,
             const int32_t &k_aub, const Tiling &tiling, int32_t &aub_m) {
  aub_m = 1;
  int32_t aub_in = aub_size / (k_aub * params.wo * kC0);
  if (params.stride_h != 1 || params.stride_w != 1) {
    for (int32_t h_num_temp = tiling.h_num; h_num_temp >= aub_m + 1; h_num_temp--) {
      if (h_num_temp * params.stride_w <= aub_in) {
        aub_m = h_num_temp;
        break;
      }
    }
  } else {
    for (int32_t h_num_temp = tiling.h_num; h_num_temp >= aub_m + 1; h_num_temp--) {
      if (kFrontUbFusionMulti * k_aub * kBlockSize *
              ((h_num_temp * params.wo + params.kw - 1 + kBlockSize - 1) / kBlockSize) * kBlockSize <=
          aub_size) {
        aub_m = h_num_temp;
        break;
      }
    }
  }
}

bool InitUbDb(const DxParas &params, Tiling &tiling, int32_t &max_dma_size) {
  int32_t loadin_size = tiling.k_aub * tiling.m_aub * params.wo * kC0 * params.stride_w;
  int32_t copyout_size = kAfterUbFusionMulti * tiling.n_cub * tiling.m_l0 * kC0 * kC0;
  int32_t ub_fp16_size = kUbSize / kFp16Bytes;
  if (params.stride_h != 1 || params.stride_w != 1) {
    if (loadin_size * tiling.db_aub + copyout_size * tiling.db_cub > ub_fp16_size) {
      tiling.db_aub = 1;
    }
    if (loadin_size * tiling.db_aub + copyout_size * tiling.db_cub > ub_fp16_size) {
      tiling.db_cub = 1;
    }
  } else {
    loadin_size = kFrontUbFusionMulti * tiling.k_aub * kBlockSize *
                  ((tiling.m_aub * params.wo + params.kw - 1 + kBlockSize - 1) / kBlockSize) * kBlockSize;
    if (loadin_size * tiling.db_aub + copyout_size * tiling.db_cub > ub_fp16_size) {
      tiling.db_aub = 1;
    }
    if (loadin_size * tiling.db_aub + copyout_size * tiling.db_cub > ub_fp16_size) {
      tiling.db_cub = 1;
    }
  }
  CHECK_OP_FUNC(loadin_size * tiling.db_aub + copyout_size * tiling.db_cub > ub_fp16_size, return false,
                "ub factor exceed buffer");
  max_dma_size = loadin_size * tiling.db_aub + copyout_size * tiling.db_cub;
  return true;
}

bool GetUbFactors(const DxParas &params, Tiling &tiling) {
  tiling.m_aub = 1;
  tiling.k_aub = 1;
  tiling.n_cub = 1;
  int32_t ub_fp16_size = kUbSize / kFp16Bytes;
  vector<int32_t> n_l0_factors;
  vector<int32_t> k_al1_factors;
  GetFactors(tiling.n_l0, n_l0_factors);
  GetFactors(tiling.k_al1, k_al1_factors);

  int32_t max_dma_size;
  CHECK_OP_FUNC(!InitUbDb(params, tiling, max_dma_size), return false, "get ub factor fail");
  bool first_flag = true;
  int32_t aub_m;
  int32_t aub_size;
  int32_t aub_temp_size;
  int32_t dma_size;
  bool modify_ub;
  bool stride_equal_one = params.stride_h == 1 && params.stride_w == 1;
  for (auto &k_aub : k_al1_factors) {
    for (auto &n1 : n_l0_factors) {
      if (params.dx_no_overlap_condition_2 && n1 < kNumTwo) {
        continue;
      }
      if (tiling.db_cub == kDbOn && tiling.n_l0 / n1 == 1) {
        tiling.db_cub = kDbOff;
      }
      aub_size = (ub_fp16_size - kAfterUbFusionMulti * n1 * tiling.m_l0 * kC0 * kC0 * tiling.db_cub) / tiling.db_aub;
      GetAubM(aub_size, params, k_aub, tiling, aub_m);
      aub_temp_size = k_aub * aub_m * params.wo * params.stride_w * kC0;
      if (stride_equal_one) {
        aub_temp_size = kFrontUbFusionMulti * k_aub * kBlockSize *
                        ((aub_m * params.wo + params.kw - 1 + kBlockSize - 1) / kBlockSize) *
                        kBlockSize;
      }
      if (aub_temp_size > aub_size) {
        continue;
      }
      dma_size = aub_temp_size * tiling.db_aub + kAfterUbFusionMulti * n1 * tiling.m_l0 * kC0 * kC0 * tiling.db_cub;
      modify_ub =
          aub_m >= 1 && (tiling.k_aub < k_aub || (tiling.k_aub == k_aub && tiling.n_cub < n1) ||
                         (tiling.k_aub == k_aub && tiling.n_cub == n1 && dma_size > max_dma_size) or first_flag);
      if (modify_ub) {
        tiling.m_aub = aub_m;
        tiling.n_cub = n1;
        tiling.k_aub = k_aub;
        max_dma_size = dma_size;
        first_flag = false;
      }
    }
  }
  tiling.n_l0_div_ub = tiling.n_l0 / tiling.n_cub;
  tiling.aub_bound = tiling.k_aub * tiling.m_aub * params.wo * kBlockSize * params.stride_w;
  if (stride_equal_one) {
    tiling.aub_bound = tiling.k_aub * kBlockSize *
                       ((tiling.m_aub * params.wo + params.kw - 1 + kBlockSize - 1) / kBlockSize) * kBlockSize;
  }
  return true;
}

void CheckSpecialTemplate(const DxParas &params, Tiling &tiling) {
  int32_t k2 = params.co1 * params.kh * params.kw;
  if (tiling.m_al1 * tiling.m_l0 == tiling.m_single_core_size && tiling.k_al1 * params.kh * params.kw == k2 &&
      tiling.m_dim == 1) {
    tiling.m_al1 = 0;
    tiling.db_al1 = 1;
  }
  if (tiling.n_bl1 * tiling.n_l0 == tiling.n_single_core_size && tiling.k_bl1 * params.kh * params.kw == k2) {
    tiling.n_bl1 = 0;
    tiling.db_bl1 = 1;
  }
}

void SetTilingId(const DxParas &params, const Tiling &tiling, int32_t &tiling_id) {
  // set kernel ID
  bool k_al1_full_load = tiling.k_al1 * params.kh * params.kw == tiling.k_single_core_size;
  bool k_bl1_full_load = tiling.k_bl1 * params.kh * params.kw == tiling.k_single_core_size;
  bool condition2 = tiling.m_al1 == 0 && k_bl1_full_load && tiling.n_bl1 != 0;
  bool condition3 = tiling.m_al1 == 0 && tiling.n_bl1 == 1;
  bool condition4 = tiling.m_al1 != 0 && k_al1_full_load && tiling.n_bl1 == 0;
  bool condition5 = tiling.m_al1 != 0 && k_al1_full_load && k_bl1_full_load && tiling.n_bl1 != 0;
  bool condition6 = tiling.m_al1 != 0 && k_al1_full_load && tiling.n_bl1 == 1;
  bool condition7 = tiling.m_al1 == 1 && tiling.n_bl1 == 0;
  bool condition8 = tiling.m_al1 == 1 && k_bl1_full_load && tiling.n_bl1 != 0;
  bool condition9 = tiling.m_al1 == 1 && tiling.n_bl1 == 1;
  // default condition1 is m_al1 is 0 and n_bl1 is 0;
  int32_t min_kl1_cmp_kl0 = kAttachFlagOne;
  int32_t al1_attach_flag = kAttachFlagZero;
  int32_t bl1_attach_flag = kAttachFlagZero;
  int32_t abkl1_attach_flag = kAttachFlagZero;
  if (min(tiling.k_al1 * params.kh * params.kw, tiling.k_bl1 * params.kh * params.kw) == tiling.k_l0) {
    min_kl1_cmp_kl0 = kAttachFlagZero;
  }
  if (condition2) {
    al1_attach_flag = kAttachFlagZero;
    bl1_attach_flag = kAttachFlagOne;
    abkl1_attach_flag = kAttachFlagZero;
  } else if (condition3) {
    al1_attach_flag = kAttachFlagZero;
    bl1_attach_flag = kAttachFlagTwo;
    abkl1_attach_flag = kAttachFlagOne;
  } else if (condition4) {
    al1_attach_flag = kAttachFlagOne;
    bl1_attach_flag = kAttachFlagZero;
    abkl1_attach_flag = kAttachFlagZero;
  } else if (condition5) {
    al1_attach_flag = kAttachFlagOne;
    bl1_attach_flag = kAttachFlagOne;
    abkl1_attach_flag = kAttachFlagZero;
  } else if (condition6) {
    al1_attach_flag = kAttachFlagOne;
    bl1_attach_flag = kAttachFlagTwo;
    abkl1_attach_flag = kAttachFlagOne;
  } else if (condition7) {
    al1_attach_flag = kAttachFlagTwo;
    bl1_attach_flag = kAttachFlagZero;
    abkl1_attach_flag = kAttachFlagTwo;
  } else if (condition8) {
    al1_attach_flag = kAttachFlagTwo;
    bl1_attach_flag = kAttachFlagOne;
    abkl1_attach_flag = kAttachFlagTwo;
  } else if (condition9) {
    al1_attach_flag = kAttachFlagTwo;
    bl1_attach_flag = kAttachFlagTwo;
    if (tiling.k_al1 > tiling.k_bl1) {
      abkl1_attach_flag = kAttachFlagOne;
    } else if (tiling.k_al1 < tiling.k_bl1) {
      abkl1_attach_flag = kAttachFlagTwo;
    }
  }

  tiling_id = tiling.db_al1;
  tiling_id = tiling_id * kDecimal + tiling.db_bl1;
  tiling_id = tiling_id * kDecimal + tiling.db_l0c;
  tiling_id = tiling_id * kDecimal + tiling.db_cub;
  tiling_id = tiling_id * kDecimal + abkl1_attach_flag;
  tiling_id = tiling_id * kDecimal + al1_attach_flag;
  tiling_id = tiling_id * kDecimal + bl1_attach_flag;
  tiling_id = tiling_id * kDecimal + params.stride_expand_flag;
}

bool GenTiling(const DxParas &params, Tiling &tiling, int32_t &tiling_id) {
  CHECK_OP_FUNC(!GetBlockDim(params, params.core_num, tiling), return false, "get block dim failed");
  if (!GetL0Factors(params, tiling)) {
    tiling.k_l0 = 1;
    tiling.m_l0 = 1;
    tiling.n_l0 = 1;
  }
  CHECK_OP_FUNC(!GetL1Factors(params, tiling), return false, "get l1 factors failed");
  if (!tiling.update_l1) {
    tiling.k_l0 = 1;
    CHECK_OP_FUNC(!GetL1Factors(params, tiling), return false, "get l1 factors failed");
    CHECK_OP_FUNC(!tiling.update_l1, return false, "not have valid l1 factors failed");
  }
  CHECK_OP_FUNC(!GetUbFactors(params, tiling), return false, "get ub factors failed");
  CheckSpecialTemplate(params, tiling);
  SetTilingId(params, tiling, tiling_id);
  return true;
}
}  // namespace optiling
