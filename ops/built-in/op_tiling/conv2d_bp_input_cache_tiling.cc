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
  int32_t min_kl1_dim[3L];
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

bool ModifyBatchDim(Tiling &tiling, const DxParas &params, const bool &min_hw) {
  // modify batch dim
  CHECK_OP_FUNC(tiling.n_dim == 0, return false, "ndim is 0");
  bool modify_batch_dim = tiling.m_dim * tiling.batch_dim < params.core_num / tiling.n_dim && !min_hw;
  if (modify_batch_dim) {
    int32_t bm_dim = params.core_num / tiling.n_dim;
    vector<int32_t> bm_dim_factor_opt;
    GetFactors(bm_dim, bm_dim_factor_opt);
    for (auto &bm_dim_factor : bm_dim_factor_opt) {
      if (params.batch % (bm_dim / bm_dim_factor) == 0 && params.ho * params.stride_h % bm_dim_factor == 0 &&
          IsSatisfyM(params, bm_dim_factor)) {
        tiling.batch_dim = bm_dim / bm_dim_factor;
        tiling.m_dim = bm_dim_factor;
        break;
      }
    }
  }
  return true;
}

bool ModifyNDim(Tiling &tiling, const DxParas &params, const bool &min_hw, const int32_t m1) {
  // modify n dim
  CHECK_OP_FUNC(tiling.m_dim == 0, return false, "mdim is 0");
  bool modify_n_dim = m1 / tiling.m_dim <= kBlockSize && !min_hw;
  if (modify_n_dim) {
    vector<int32_t> n_dim_factor_opt;
    GetFactors(params.c1 / tiling.n_dim, n_dim_factor_opt);
    int32_t nm_dim = params.core_num / tiling.batch_dim;
    for (auto &n_dim_factor : n_dim_factor_opt) {
      int32_t n_temp = min(tiling.n_dim * n_dim_factor, params.core_num);
      CHECK_OP_FUNC(n_temp == 0, return false, "n_temp is 0");
      while (params.c1 % n_temp != 0) {
        n_temp--;
      }
      CHECK_OP_FUNC(n_temp == 0, return false, "n_temp is 0");
      int32_t m_dim_temp = max(nm_dim / n_temp, kMinCoreNum);
      if (m1 / m_dim_temp >= kBlockSize && IsSatisfyM(params, m_dim_temp) && IsSatisfyN(params, n_temp)) {
        tiling.m_dim = m_dim_temp;
        tiling.n_dim = n_temp;
        break;
      }
    }
  }
  return true;
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
  // Derivation of load size: a_size * n_dim + b_size * core_num / n_dim
  float n_factor =
      max(min(sqrt(static_cast<float>(core_num * b_size) / static_cast<float>(a_size)), static_cast<float>(params.c1)),
          1.0f);
  CHECK_OP_FUNC(EqualWith(n_factor, kFloatZero), return false, "n_factor is invalid");
  float bm_factor = min(static_cast<float>(core_num) / n_factor, static_cast<float>(m1 * params.batch));
  float b_factor = min(static_cast<float>(params.batch), bm_factor);
  float m_factor = min(static_cast<float>(core_num) / (n_factor * b_factor), static_cast<float>(m1));
  float factor_temp[3L] = {floor(n_factor), floor(b_factor), floor(m_factor)};
  int32_t factor[3L];
  for (size_t i = 0; i < 3L; i++) {
    factor_temp[i] = min(factor_temp[i], static_cast<float>(core_num));
    factor_temp[i] = max(factor_temp[i], static_cast<float>(kMinCoreNum));
    factor[i] = static_cast<int32_t>(factor_temp[i]);
  }
  n_factor = factor[0];
  b_factor = factor[1];
  m_factor = factor[kIdxTwo];
  int32_t batch_dim_opti[2L] = {0, 0};
  int32_t n_dim_opti[2L] = {0, 0};
  CHECK_OP_FUNC(!GenNearestFactor(b_factor, params.batch, batch_dim_opti), return false, "get b_factor failed");
  CHECK_OP_FUNC(!GenNearestFactor(n_factor, params.c1, n_dim_opti), return false, "get n_factor failed");
  tiling.batch_dim = 1;
  tiling.n_dim = 1;
  tiling.m_dim = 1;
  int64_t min_load_size = static_cast<int64_t>(tiling.batch_dim * tiling.m_dim) * b_size +
                          static_cast<int64_t>(tiling.n_dim) * a_size;
  int32_t core_use = tiling.batch_dim * tiling.n_dim * tiling.m_dim;
  for (auto &batch_dim : batch_dim_opti) {
    for (auto &n_dim : n_dim_opti) {
      if (batch_dim * n_dim > core_num) {
        continue;
      }
      int32_t m_dim_temp = min(core_num / (batch_dim * n_dim), m1);
      int64_t load_size = static_cast<int64_t>(batch_dim * m_dim_temp) * b_size + static_cast<int64_t>(n_dim) * a_size;
      int32_t core_use_temp = batch_dim * n_dim * m_dim_temp;
      bool modify_dim = core_use < core_use_temp || (core_use == core_use_temp and load_size < min_load_size) ||
                        (core_use == core_use_temp and abs(load_size - min_load_size) < kLoadSizeThreshold &&
                                                        tiling.batch_dim * tiling.n_dim < batch_dim * n_dim);
      bool satisfy_constraint_m = IsSatisfyM(params, m_dim_temp);
      bool satisfy_constraint_n = IsSatisfyN(params, n_dim);
      if (modify_dim && satisfy_constraint_m && satisfy_constraint_n) {
        min_load_size = load_size;
        tiling.batch_dim = batch_dim;
        tiling.n_dim = n_dim;
        tiling.m_dim = m_dim_temp;
        core_use = tiling.batch_dim * tiling.n_dim * tiling.m_dim;
      }
    }
  }
  bool min_hw = MdimTune(params, m1, tiling.n_dim, tiling.m_dim, tiling);
  CHECK_OP_FUNC(!min_hw && tiling.m_dim == 0, return false, "m_dim or n_dim is 0");
  CHECK_OP_FUNC(!ModifyBatchDim(tiling, params, min_hw), return false, "modify batch dim failed");
  CHECK_OP_FUNC(!ModifyNDim(tiling, params, min_hw, m1), return false, "modify n dim failed");
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

bool CheckL1Overflow(const DxParas &params, const Tiling &tiling,
                     const int32_t &m0, const int32_t &n0, int32_t &k0) {
  int32_t l1_fp16_size = kL1Size / kFp16Bytes;
  int32_t m_al1 = 1;
  int32_t n_bl1 = 1;
  int32_t k_hw = (params.kh * params.kw);
  int32_t k_al1 = InitialKl1(k0, k_hw);
  CHECK_OP_FUNC(k_al1 == 0, return false, "initial kl1 failed");
  int32_t h_num = (params.kh - 1) + m_al1 * m0 * kBlockSize / params.w + khnumWNoDivided;
  if (m_al1 * m0 * kBlockSize < params.w) {
    h_num = (params.kh - 1) + khnumWNoDivided;
  } else if (m_al1 * m0 * kBlockSize % params.w == 0) {
    h_num = (params.kh - 1) + m_al1 * m0 * kBlockSize / params.w;
  }
  int32_t b_l1_size =
      k_al1 * params.kh * params.kw * n_bl1 * n0 * kC0 * kBlockSize;
  int32_t a_l1_size = k_al1 * params.wo * params.stride_w * kC0 * h_num;
  if (m_al1 * m0 == tiling.m_single_core_size && k_al1 == params.co1 && tiling.m_dim == 1) {
    int32_t hw_ceil_align = CeilAlign(params.ho * params.stride_h * params.wo * params.stride_w, kBlockSize);
    CHECK_OP_FUNC(hw_ceil_align == 0, return false, "hw_ceil_align is invalid");
    a_l1_size = k_al1 * kBlockSize * hw_ceil_align;
  }
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

bool GetSameCutOff(const int32_t &first, const int32_t &second, const int32_t &factor,
                   int32_t cand[][2L], int32_t &arr_len) {
  CHECK_OP_FUNC(factor == 0, return false, "factor is 0");
  int32_t mt = first / factor;
  int32_t factor_fir = first / (mt + 1) + 1;
  int32_t factor_sec = min(kL0cNzSize / factor_fir, second);
  CHECK_OP_FUNC(factor_sec == 0, return false, "factor_sec is 0");
  mt = first / factor_fir;
  int32_t mt_1 = second / factor_sec;
  cand[0][0] = factor_fir;
  cand[0][1] = factor_sec;
  for (int32_t i = factor_fir + 1; i <= first; i++) {
    factor_sec = min(kL0cNzSize / i, second);
    if (first / i == mt && second / factor_sec == mt_1) {
      arr_len += 1;
      if (arr_len > 16L) {
        break;
      }
      cand[arr_len][0] = i;
      cand[arr_len][1] = factor_sec;
    } else {
      break;
    }
  }
  return true;
}

bool ModifyKl0(const DxParas &params, int32_t &k_l0) {
  CHECK_OP_FUNC(k_l0 == 0, return false, "k_l0 is 0");
  while ((params.co1 * params.kh * params.kw) % k_l0 != 0) {
    k_l0--;
  }
  return true;
}

bool PushMNCandidate(const DxParas &params, const int32_t cand[], Tiling &tiling,
                     int32_t mn_cand_mkn[], const int32_t &index) {
  int32_t cand_m = cand[0];
  int32_t cand_n = cand[1];
  if ((index & 0x100) != 0) {
    cand_m = cand[1];
    cand_n = cand[0];
  }

  if (cand_m == 0 || cand_n == 0) {
    return false;
  }

  int32_t cand_k =
      min(min(static_cast<int32_t>(floor(static_cast<float>(kL0aNzSize) / static_cast<float>(cand_m))),
              static_cast<int32_t>(floor(static_cast<float>(kL0bNzSize) / static_cast<float>(cand_n)))),
          tiling.k_single_core_size);
  CHECK_OP_FUNC(!ModifyKl0(params, cand_k), return false, "cand_k is 0");

  int32_t load_size = ((tiling.m_single_core_size - 1) / cand_m) * tiling.n_single_core_size +
                      ((tiling.n_single_core_size - 1) / cand_n) * tiling.m_single_core_size;
  int32_t mkn = cand_m * cand_n * cand_k;
  float max_mk = static_cast<float>(max(cand_m, cand_k)) / static_cast<float>(min(cand_m, cand_k));
  bool l0_invalid = !CheckL0Overflow(cand_m, cand_n, cand_k) ||
                    !CheckL1Overflow(params, tiling, cand_m, cand_n, cand_k) || !CheckUbDb(params, tiling, cand_m);
  if (l0_invalid) {
    return false;
  }
  if (mn_cand_mkn[0] == 0 && mn_cand_mkn[1] == 0 && mn_cand_mkn[kIdxTwo] == 0) {
    mn_cand_mkn[0] = cand_m;
    mn_cand_mkn[1L] = cand_n;
    mn_cand_mkn[kIdxTwo] = cand_k;
    mn_cand_mkn[kIdxThree] = load_size;
    mn_cand_mkn[kIdxFour] = mkn;
  } else {
    float max_mk_temp = static_cast<float>(max(mn_cand_mkn[0], mn_cand_mkn[kIdxTwo])) /
                        static_cast<float>(min(mn_cand_mkn[0], mn_cand_mkn[kIdxTwo]));
    bool l0_valid =
        (load_size < mn_cand_mkn[kIdxThree]) ||
        ((load_size == mn_cand_mkn[kIdxThree]) && (mkn > mn_cand_mkn[kIdxFour] ||
                                           (mkn == mn_cand_mkn[kIdxFour] && max_mk > max_mk_temp)));
    if (l0_valid) {
      mn_cand_mkn[0] = cand_m;
      mn_cand_mkn[1L] = cand_n;
      mn_cand_mkn[kIdxTwo] = cand_k;
      mn_cand_mkn[kIdxThree] = load_size;
      mn_cand_mkn[kIdxFour] = mkn;
    }
  }
  return true;
}

bool GetCandMKN(const DxParas &params, Tiling &tiling) {
  // get cand from m
  int32_t cand[16L][2L] = {{0, 0}};
  int32_t arr_len = 1;
  int32_t mn_cand_mkn[5L] = {0, 0, 0, 0, 0};
  int32_t factor = kM0N0OptimalNode;
  CHECK_OP_FUNC(!GetSameCutOff(tiling.m_single_core_size - 1, tiling.n_single_core_size - 1, factor, cand, arr_len),
                return false, "m get cut off failed");
  for (int32_t i = 0; i < arr_len; i++) {
    PushMNCandidate(params, cand[i], tiling, mn_cand_mkn, i);
  }
  // m right
  arr_len = 1;
  int32_t mt = (tiling.m_single_core_size - 1) / factor;
  int32_t mt_r = (tiling.m_single_core_size - 1) / (mt + 1);
  CHECK_OP_FUNC(!GetSameCutOff(tiling.m_single_core_size - 1, tiling.n_single_core_size - 1, mt_r, cand, arr_len),
                return false, "mt_r get cut off failed");
  for (int32_t i = 0; i < arr_len; i++) {
    PushMNCandidate(params, cand[i], tiling, mn_cand_mkn, i);
  }
  // m_left
  arr_len = 1;
  int32_t mt_l = mt > kNumTwo ? (tiling.m_single_core_size - 1) / (mt - 1) : tiling.m_single_core_size;
  CHECK_OP_FUNC(!GetSameCutOff(tiling.m_single_core_size - 1, tiling.n_single_core_size - 1, mt_l, cand, arr_len),
                return false, "mt_l get cut off failed");
  for (int32_t i = 0; i < arr_len; i++) {
    PushMNCandidate(params, cand[i], tiling, mn_cand_mkn, i);
  }
  // get cand from n
  arr_len = 1;
  CHECK_OP_FUNC(!GetSameCutOff(tiling.n_single_core_size - 1, tiling.m_single_core_size - 1, factor, cand, arr_len),
                return false, "n get cut off failed");
  for (int32_t i = 0; i < arr_len; i++) {
    PushMNCandidate(params, cand[i], tiling, mn_cand_mkn, i + (1 << 8L));
  }
  // n right
  arr_len = 1;
  int32_t nt = (tiling.n_single_core_size - 1) / factor;
  int32_t nt_r = (tiling.n_single_core_size - 1) / (nt + 1);
  CHECK_OP_FUNC(!GetSameCutOff(tiling.n_single_core_size - 1, tiling.m_single_core_size - 1, nt_r, cand, arr_len),
                return false, "nt_r get cut off failed");
  for (int32_t i = 0; i < arr_len; i++) {
    PushMNCandidate(params, cand[i], tiling, mn_cand_mkn, i + (1 << 8L));
  }
  // n_left
  arr_len = 1;
  int32_t nt_l = nt > kNumTwo ? (tiling.n_single_core_size - 1) / (nt - 1) : tiling.n_single_core_size;
  CHECK_OP_FUNC(!GetSameCutOff(tiling.n_single_core_size - 1, tiling.m_single_core_size - 1, nt_l, cand, arr_len),
                return false, "nt_l get cut off failed");
  for (int32_t i = 0; i < arr_len; i++) {
    PushMNCandidate(params, cand[i], tiling, mn_cand_mkn, i + (1 << 8L));
  }
  if (mn_cand_mkn[0] != 0) {
    tiling.m_l0_opti = mn_cand_mkn[0];
    tiling.n_l0_opti = mn_cand_mkn[1];
    tiling.k_l0_opti = mn_cand_mkn[kIdxTwo];
  }
  return true;
}

bool GetNMFactor(const int32_t &factor, const int32_t &k2, const int32_t &max_factor, int32_t &nm) {
  if (factor == 0) {
    return false;
  }
  int32_t k0 = min(kL0aNzSize / factor, k2);
  if (k0 == 0) {
    return false;
  }
  nm = min(min(kL0cNzSize / factor, kL0aNzSize / k0), max_factor);
  return true;
}

bool GetCutOffPoint(Tiling &tiling, int32_t m0_factor[], int32_t n0_factor[],
                    const int32_t &m0_factor_size, const int32_t &n0_factor_size) {
  int32_t m;
  int32_t n;
  int32_t idx_n = 1;
  int32_t idx_m = 1;
  CHECK_OP_FUNC(!GetNMFactor(tiling.m_l0_opti, tiling.k_single_core_size, tiling.n_single_core_size, n) ||
                !GetNMFactor(tiling.n_l0_opti, tiling.k_single_core_size, tiling.m_single_core_size, m),
                return false, "tiling param invalid");
  if (std::find(n0_factor, n0_factor + n0_factor_size, n) == (n0_factor + n0_factor_size)) {
    n0_factor[idx_n++] = n;
  }
  CHECK_OP_FUNC(!GetNMFactor(tiling.n_l0_opti, tiling.k_single_core_size, tiling.m_single_core_size, m),
                return false, "tiling param invalid");
  if (std::find(m0_factor, m0_factor + m0_factor_size, m) == (m0_factor + m0_factor_size)) {
    m0_factor[idx_m++] = m;
  }
  // m right
  int32_t mt = max((tiling.m_single_core_size - 1) / tiling.m_l0_opti, 1);
  int32_t mt_r = mt > 0 ? (tiling.m_single_core_size - 1) / mt + 1 : tiling.m_single_core_size;
  mt_r = min(mt_r, kL0aNzSize);
  if (std::find(m0_factor, m0_factor + m0_factor_size, mt_r) == (m0_factor + m0_factor_size)) {
    m0_factor[idx_m++] = mt_r;
    CHECK_OP_FUNC(!GetNMFactor(mt_r, tiling.k_single_core_size, tiling.n_single_core_size, n),
                  return false, "tiling param invalid");
    if (std::find(n0_factor, n0_factor + n0_factor_size, n) == (n0_factor + n0_factor_size)) {
      n0_factor[idx_n++] = n;
    }
  }
  // m left
  int32_t mt_l = max((tiling.m_single_core_size - 1) / (mt + 1), 1);
  mt_l = min(mt_l, kL0aNzSize);
  if (std::find(m0_factor, m0_factor + m0_factor_size, mt_l) == (m0_factor + m0_factor_size)) {
    m0_factor[idx_m++] = mt_l;
    CHECK_OP_FUNC(!GetNMFactor(mt_l, tiling.k_single_core_size, tiling.n_single_core_size, n),
                  return false, "tiling param invalid");
    if (std::find(n0_factor, n0_factor + n0_factor_size, n) == (n0_factor + n0_factor_size)) {
      n0_factor[idx_n++] = n;
    }
  }
  // n right
  int32_t nt = max((tiling.n_single_core_size - 1) / tiling.n_l0_opti, 1);
  int32_t nt_r = nt > 0 ? (tiling.n_single_core_size - 1) / nt + 1 : tiling.n_single_core_size;
  nt_r = min(nt_r, kL0bNzSize);
  if (std::find(n0_factor, n0_factor + n0_factor_size, nt_r) == (n0_factor + n0_factor_size)) {
    n0_factor[idx_n++] = nt_r;
    CHECK_OP_FUNC(!GetNMFactor(nt_r, tiling.k_single_core_size, tiling.m_single_core_size, m),
                  return false, "tiling param invalid");
    if (std::find(m0_factor, m0_factor + m0_factor_size, m) == (m0_factor + m0_factor_size)) {
      m0_factor[idx_m++] = m;
    }
  }
  // n left
  int32_t nt_l = max((tiling.n_single_core_size - 1) / (nt + 1), 1);
  nt_l = min(nt_l, kL0bNzSize);
  if (std::find(n0_factor, n0_factor + n0_factor_size, nt_l) == (n0_factor + n0_factor_size)) {
    n0_factor[idx_n++] = nt_l;
    CHECK_OP_FUNC(!GetNMFactor(nt_l, tiling.k_single_core_size, tiling.m_single_core_size, m),
                  return false, "tiling param invalid");
    if (std::find(m0_factor, m0_factor + m0_factor_size, m) == (m0_factor + m0_factor_size)) {
      m0_factor[idx_m++] = m;
    }
  }
  return true;
}

bool GetCin1Factor(int32_t n0_factor[], int32_t m0_factor[], const Tiling &tiling, const int32_t &n0_factor_size,
                   const int32_t &m0_factor_size) {
  int32_t n0_optional[2] = {0, 0};
  int32_t idx_m = kL0CutOffPointNum;
  int32_t idx_n = kL0CutOffPointNum;
  for (int32_t i = 0; i < kL0CutOffPointNum; i++) {
    if (n0_factor[i] == 0) {
      continue;
    }
    GenNearestFactor(n0_factor[i], tiling.n_single_core_size, n0_optional);
    if (std::find(n0_factor, n0_factor + n0_factor_size, n0_optional[0]) == (n0_factor + n0_factor_size)) {
      n0_factor[idx_n++] = n0_optional[0];
    }
    if (std::find(n0_factor, n0_factor + n0_factor_size, n0_optional[1]) == (n0_factor + n0_factor_size)) {
      n0_factor[idx_n++] = n0_optional[1];
    }
  }
  int32_t m0 = 0;
  for (int32_t i = kL0CutOffPointNum; i < n0_factor_size; i++) {
    if (n0_factor[i] == 0) {
      continue;
    }
    CHECK_OP_FUNC(!GetNMFactor(n0_factor[i], tiling.k_single_core_size, tiling.m_single_core_size, m0),
                  return false, "tiling param invalid");
    if (std::find(m0_factor, m0_factor + m0_factor_size, m0) == (m0_factor + m0_factor_size)) {
      m0_factor[idx_m++] = m0;
    }
  }
  return true;
}

bool GetL0FactorsOpti(const DxParas &params, Tiling &tiling) {
  // x0 is optimal of min load size equation of ((m2-1)/m0)*n2 + ((n2-1)/n0)*m2
  int32_t x0 = kM0N0OptimalNode;
  if (tiling.m_single_core_size <= x0 && tiling.n_single_core_size <= x0) {
    tiling.k_l0_opti = min(min(kL0aNzSize / tiling.m_single_core_size, kL0bNzSize / tiling.n_single_core_size),
                           tiling.k_single_core_size);
    CHECK_OP_FUNC(!ModifyKl0(params, tiling.k_l0_opti), return false, "k_l0 is 0");
    tiling.m_l0_opti = min(kL0aNzSize / tiling.k_l0_opti, tiling.m_single_core_size);
    tiling.n_l0_opti = min(kL0bNzSize / tiling.k_l0_opti, tiling.n_single_core_size);
  } else if (tiling.m_single_core_size > x0 && tiling.n_single_core_size > x0) {
    CHECK_OP_FUNC(!GetCandMKN(params, tiling), return false, "get candiate mkn factor failed");
  } else if (tiling.m_single_core_size <= x0 && tiling.n_single_core_size > x0) {
    int32_t m_temp = min(tiling.m_single_core_size, kL0aNzSize);
    CHECK_OP_FUNC(m_temp == 0, return false, "m_temp invalid");
    int32_t n_temp = min(min(kL0cNzSize / m_temp, tiling.n_single_core_size), kL0bNzSize);
    CHECK_OP_FUNC(n_temp == 0, return false, "n_temp invalid");
    tiling.k_l0_opti = min(min(kL0aNzSize / m_temp, kL0bNzSize /n_temp), tiling.k_single_core_size);
    CHECK_OP_FUNC(!ModifyKl0(params, tiling.k_l0_opti), return false, "k_l0 is 0");
    tiling.m_l0_opti = min(kL0aNzSize / tiling.k_l0_opti, m_temp);
    tiling.n_l0_opti = min(kL0bNzSize / tiling.k_l0_opti, n_temp);
  } else if (tiling.n_single_core_size <= x0 && tiling.m_single_core_size > x0) {
    int32_t n_temp = min(tiling.n_single_core_size, kL0bNzSize);
    CHECK_OP_FUNC(n_temp == 0, return false, "n_temp invalid");
    int32_t m_temp = min(min(kL0cNzSize / n_temp, tiling.m_single_core_size), kL0aNzSize);
    CHECK_OP_FUNC(m_temp == 0, return false, "m_temp invalid");
    tiling.k_l0_opti = min(min(kL0aNzSize / m_temp, kL0bNzSize / n_temp), tiling.k_single_core_size);
    CHECK_OP_FUNC(!ModifyKl0(params, tiling.k_l0_opti), return false, "k_l0 is 0");
    tiling.m_l0_opti = min(kL0aNzSize / tiling.k_l0_opti, m_temp);
    tiling.n_l0_opti = min(kL0bNzSize / tiling.k_l0_opti, n_temp);
  }
  return true;
}

bool GetL0FactorsGeneral(const DxParas &params, Tiling &tiling) {
  tiling.k_l0 = 1;
  tiling.m_l0 = 1;
  tiling.n_l0 = 1;
  CHECK_OP_FUNC(!GetL0FactorsOpti(params, tiling), return false, "get l0 optimal value failed");
  int32_t m0_factor[18L] = {tiling.m_l0_opti};
  int32_t n0_factor[18L] = {tiling.n_l0_opti};
  int32_t n0_factor_size = end(n0_factor) - begin(n0_factor);
  int32_t m0_factor_size = end(m0_factor) - begin(m0_factor);
  CHECK_OP_FUNC(!GetCutOffPoint(tiling, m0_factor, n0_factor, m0_factor_size, n0_factor_size), return false,
                "get cutoff of mkn factor failed");
  int32_t min_load_size = ((tiling.m_single_core_size - 1)) * tiling.n_single_core_size +
                          ((tiling.n_single_core_size - 1)) * tiling.m_single_core_size;
  int32_t max_mkn = 1;
  int32_t max_mk = 1;
  int32_t load_size;
  int32_t mkn;
  int32_t mk;
  bool l0_invalid = tiling.k_l0_opti <= 0 || tiling.m_l0_opti * tiling.n_l0_opti > kL0cNzSize ||
                    !CheckL0Overflow(tiling.m_l0_opti, tiling.n_l0_opti, tiling.k_l0_opti) ||
                    !CheckL1Overflow(params, tiling, tiling.m_l0_opti, tiling.n_l0_opti, tiling.k_l0_opti) ||
                    !CheckUbDb(params, tiling, tiling.m_l0_opti) ||
                    (params.co1 * params.kw * params.kh) % tiling.k_l0_opti != 0 ||
                    tiling.n_single_core_size % tiling.n_l0_opti != 0;
  if (params.dx_no_overlap_condition_2) {
    l0_invalid = l0_invalid || tiling.n_l0 < kNumTwo;
  }
  bool l0_valid_res = !l0_invalid;
  if (l0_invalid) {
    CHECK_OP_FUNC(!GetCin1Factor(n0_factor, m0_factor, tiling, n0_factor_size, m0_factor_size), return false,
                  "get cin1 facotr fail");
  } else {
    min_load_size = ((tiling.m_single_core_size - 1) / tiling.m_l0_opti) * tiling.n_single_core_size +
                    ((tiling.n_single_core_size - 1) / tiling.n_l0_opti) * tiling.m_single_core_size;
    max_mkn = tiling.m_l0_opti * tiling.n_l0_opti * tiling.k_l0_opti;
    max_mk = max(tiling.m_l0_opti, tiling.k_l0_opti) / min(tiling.m_l0_opti, tiling.k_l0_opti);
    tiling.m_l0 = tiling.m_l0_opti;
    tiling.n_l0 = tiling.n_l0_opti;
    tiling.k_l0 = tiling.k_l0_opti;
  }
  bool l0_valid = false;
  for (auto &m0 : m0_factor) {
    for (auto &n0 : n0_factor) {
      if (m0 == 0 || n0 == 0) {
        continue;
      }
      int32_t k0 = min(min(kL0aNzSize / m0, kL0bNzSize / n0), tiling.k_single_core_size);
      CHECK_OP_FUNC(!ModifyKl0(params, k0), return false, "k_l0 is 0");
      CHECK_OP_FUNC(k0 == 0, return false, "k_l0 is 0");
      l0_invalid = k0 <= 0 || m0 * n0 > kL0cNzSize || !CheckL0Overflow(m0, n0, k0) ||
                        !CheckL1Overflow(params, tiling, m0, n0, k0) || !CheckUbDb(params, tiling, m0) ||
                        (params.co1 * params.kw * params.kh) % k0 != 0 ||
                        tiling.n_single_core_size % n0 != 0;
      if (params.dx_no_overlap_condition_2) {
        l0_invalid = l0_invalid || n0 < kNumTwo;
      }
      if (l0_invalid) {
        continue;
      }
      load_size = ((tiling.m_single_core_size - 1) / m0) * tiling.n_single_core_size +
                  ((tiling.n_single_core_size - 1) / n0) * tiling.m_single_core_size;
      mkn = m0 * n0 * k0;
      mk = max(m0, k0) / min(m0, k0);
      l0_valid = (load_size < min_load_size) ||
                 ((load_size == min_load_size) && (mkn > max_mkn || (mkn == max_mkn && mk > max_mk)));
      if (l0_valid) {
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

bool GetL0FactorsNoOverlap(const DxParas &params, Tiling &tiling) {
  // It is necessary to ensure that the tail block of the tail core is greater than 1 block
  int32_t m_l0_max = min(kL0aNzSize, tiling.m_single_core_size);
  for (int32_t m_l0_temp = m_l0_max; m_l0_temp >= kNumTwo; m_l0_temp--) {
    int32_t m_tail_size = params.hw % (m_l0_temp * kBlockSize);
    if (m_tail_size != 0 && m_tail_size < kBlockSize) {
      continue;
    }
    int32_t n_l0_max = kL0bNzSize;
    int32_t n_l0_temp = tiling.n_single_core_size;
    CHECK_OP_FUNC(!GetNMFactor(m_l0_temp, tiling.k_single_core_size, tiling.n_single_core_size, n_l0_max),
                  return false, "tiling param invalid");
    // On the premise that m is determined, obtain the maximum N factor
    for (int32_t i = n_l0_max; i >= 1; i--) {
      if (tiling.n_single_core_size % i == 0) {
        n_l0_temp = i;
        break;
      }
    }
    // After MN is determined, the appropriate factor K is obtained
    int32_t k_l0_temp = min(min(kL0aNzSize / m_l0_temp, kL0bNzSize / n_l0_temp), tiling.k_single_core_size);
    CHECK_OP_FUNC(!ModifyKl0(params, k_l0_temp), return false, "k_l0 is 0");
    CHECK_OP_FUNC(k_l0_temp == 0, return false, "k_l0 is 0");
    bool l0_invalid = k_l0_temp <= 0 || m_l0_temp * n_l0_temp > kL0cNzSize ||
                      !CheckL0Overflow(m_l0_temp, n_l0_temp, k_l0_temp) ||
                      !CheckL1Overflow(params, tiling, m_l0_temp, n_l0_temp, k_l0_temp) ||
                      !CheckUbDb(params, tiling, m_l0_temp) ||
                      (params.co1 * params.kw * params.kh) % k_l0_temp != 0;
    if (l0_invalid) {
      continue;
    }
    tiling.m_l0 = m_l0_temp;
    tiling.n_l0 = n_l0_temp;
    tiling.k_l0 = k_l0_temp;
    return true;
  }
  return false;
}

bool GetL0Factors(const DxParas &params, Tiling &tiling) {
  // get m_l0, n_l0, k_l0 factor when singlecore m, n, k is know
  // m_l0, n_l0, k_l0 is a factor of single core m, n, k
  tiling.db_l0c = kDbOn;
  tiling.db_cub = kDbOff;
  tiling.db_aub = kDbOff;
  if (params.dx_no_overlap_condition_1) {
    return GetL0FactorsNoOverlap(params, tiling);
  } else {
    return GetL0FactorsGeneral(params, tiling);
  }
}

bool GetInitialL1(const DxParas &params, Tiling &tiling, int32_t min_kl1_dim[]) {
  CHECK_OP_FUNC(
      !GenNearestFactor((tiling.k_l0 + params.kh * params.kw - 1) / (params.kh * params.kw), params.co1, min_kl1_dim),
      return false, "get k_l1_factor failed");
  tiling.init_db_al1 = kDbOn;
  tiling.init_db_bl1 = kDbOn;
  tiling.m_al1 = 1;
  tiling.n_bl1 = 1;
  int32_t k_hw = (params.kh * params.kw);
  tiling.k_al1 = InitialKl1(tiling.k_l0, k_hw);
  CHECK_OP_FUNC(tiling.k_al1 == 0, return false, "initial kl1 failed");
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
  int32_t a_size;
  size_t idx = 0;
  int32_t k_bl1 = kn_factors[idx++];
  int32_t k_al1 = kn_factors[idx++];
  int32_t nbl1 = kn_factors[idx++];
  idx = 0;
  int32_t m_1 = m_h[idx++];
  int32_t h2 = m_h[idx++];
  int32_t db_al1_end = m_h[idx++];
  int32_t db_bl1_end = m_h[idx++];
  int32_t b_size =
      k_bl1 * kBlockSize * params.kh * params.kw * nbl1 * tiling.n_l0 * kBlockSize * db_bl1_end * kFp16Bytes;
  if (static_cast<int32_t>(
          ceil(static_cast<double>(tiling.n_single_core_size) / static_cast<double>(nbl1 * tiling.n_l0))) == 1 &&
      k_bl1 == params.co1) {
    b_size = params.co1 * kBlockSize * params.kh * params.kw * nbl1 * tiling.n_l0 * kBlockSize * kFp16Bytes;
  }
  if (m_1 * tiling.m_l0 == tiling.m_single_core_size && k_al1 == params.co1 && tiling.m_dim == 1) {
    a_size = h2 * params.stride_h * params.wo * params.stride_w * params.co1 * kBlockSize * kFp16Bytes;
  } else {
    int32_t h_num = (params.kh - 1) + khnumWNoDivided + (m_1 * tiling.m_l0 * kBlockSize) / params.w;
    if (m_1 * tiling.m_l0 * kBlockSize < params.w) {
      h_num = (params.kh - 1) + khnumWNoDivided;
    } else if ((m_1 * tiling.m_l0 * kBlockSize) % params.w == 0) {
      h_num = (params.kh - 1) + (m_1 * tiling.m_l0 * kBlockSize) / params.w;
    }
    a_size = h_num * params.wo * params.stride_w * k_al1 * kBlockSize * db_al1_end * kFp16Bytes;
  }
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
                   (k_bl1 >= factor_size.min_kl1_dim[1] && k_al1 >= factor_size.min_kl1_dim[1]) &&
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
      int32_t m_h[4L] = {m_1, h2, db_al1_end, db_bl1_end};
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
  factor_size.min_kl1_dim[0] = 0;
  factor_size.min_kl1_dim[1] = 0;
  factor_size.min_kl1_dim[2L] = 0;
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

bool GetUbFactors(const DxParas &params, Tiling &tiling) {
  tiling.m_aub = 1;
  tiling.k_aub = 1;
  tiling.n_cub = 1;
  int32_t ub_fp16_size = kUbSize / kFp16Bytes;
  vector<int32_t> n_l0_factors;
  vector<int32_t> k_al1_factors;
  GetFactors(tiling.n_l0, n_l0_factors);
  GetFactors(tiling.k_al1, k_al1_factors);
  int32_t loadin_size = tiling.k_aub * tiling.m_aub * params.wo * kC0 * params.stride_w;
  int32_t copyout_size = kAfterUbFusionMulti * tiling.n_cub * tiling.m_l0 * kC0 * kC0;
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
  int32_t max_dma_size = loadin_size * tiling.db_aub + copyout_size * tiling.db_cub;
  bool first_flag = true;
  int32_t aub_m;
  int32_t aub_size;
  int32_t aub_temp_size;
  int32_t dma_size;
  bool modify_ub;
  for (auto &k_aub : k_al1_factors) {
    for (auto &n1 : n_l0_factors) {
      if (params.dx_no_overlap_condition_2 && n1 < kNumTwo) {
        continue;
      }
      aub_size = (ub_fp16_size - kAfterUbFusionMulti * n1 * tiling.m_l0 * kC0 * kC0 * tiling.db_cub) / tiling.db_aub;
      GetAubM(aub_size, params, k_aub, tiling, aub_m);
      if (k_aub < params.co1) {
        aub_m = 1;
      }
      aub_temp_size = k_aub * aub_m * params.wo * params.stride_w * kC0;
      if (params.stride_h == 1 && params.stride_w == 1) {
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
  tiling_id = tiling_id * kDecimal + abkl1_attach_flag;
  tiling_id = tiling_id * kDecimal + al1_attach_flag;
  tiling_id = tiling_id * kDecimal + bl1_attach_flag;
  tiling_id = tiling_id * kDecimal + min_kl1_cmp_kl0;
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
