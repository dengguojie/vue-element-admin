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
#include <map>

#include "conv2d_bp_input_cache_tiling.h"
using namespace std;

namespace optiling {
static const int64_t kCoreNum = 32;
static const int64_t kL1Size = (1024 * 1024);
static const int64_t kL0cSize = (256 * 1024);
static const int64_t kUbSize = (256 * 1024);
static const int64_t kBlockSize = 16;
static const int64_t kDecimal = 10;
static const int64_t kDbOn = 2;
static const int64_t kDbOff = 1;
static const int64_t kFrontUbFusionMulti = 3;
static const int64_t kAfterUbFusionMulti = 2;
static const int64_t kHoshWNoDivided = 2;
static const float kFloatZero = 0.0f;
static const int64_t kAttachFlagZero = 0;
static const int64_t kAttachFlagOne = 1;
static const int64_t kAttachFlagTwo = 2;
static const int64_t kFp16Bytes = 2;
static const int64_t kFp32Bytes = 4;
static const int64_t kMmadComputeOneUs = 1000;
static const int64_t kMinCoreNum = 1;
static const double kLoadSizeThreshold = 0.00001;
static const int64_t kM2MaxSize = 1024;
static const int64_t kM2Size = 16;
static const int64_t kL0aSize = (64 * 1024);
static const int64_t kL0bSize = (64 * 1024);
static const int64_t kC0 = 16;
static const int64_t kM0N0OptimalNode = 11;
static const int64_t kL0cNzSize = 128;
static const int64_t kL0aNzSize = 64;
static const int64_t kL0bNzSize = 64;
static const int64_t kIdxTwo = 2;
static const int64_t kIdxThree = 3;
static const int64_t kIdxFour = 4;
static const int64_t kNumTwo = 2;

bool EqualWith(const float& l_value, const float& r_value) {
  return std::fabs(l_value - r_value) <= std::numeric_limits<float>::epsilon();
}

int64_t CeilDiv(const int64_t& num1, const int64_t& num2) {
  if (num2 == 0) {
    return 0;
  }
  return (num1 + num2 - 1) / num2;
}

int64_t CeilAlign(const int64_t& num1, const int64_t& num2) {
  return CeilDiv(num1, num2) * num2;
}

bool MdimTune(const DxParas &params, const int64_t &m1, const int64_t &n_dim_factor, const int64_t &m_dim_factor,
              Tiling &tiling) {
  if (n_dim_factor == 0 || m_dim_factor == 0) {
    tiling.m_dim = 0;
    return false;
  }
  int64_t min_m = static_cast<int64_t>(
      ceil(static_cast<double>(kMmadComputeOneUs) /
           static_cast<double>((params.co1 * params.kh * params.kw * (params.c1 / n_dim_factor)))));
  bool min_hw = m1 / m_dim_factor <= min_m;
  if (min_hw && m_dim_factor > 1) {
    tiling.m_dim = max(m1 / min_m, kMinCoreNum);
  }
  return min_hw;
}

bool GenNearestFactor(const int64_t& factor, const int64_t& dim, vector<int64_t>& factor_optional) {
  int64_t cur_factor = min(factor + 1, dim);
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

void GetFactors(const int64_t& bm_dim, vector<int64_t>& bm_dim_opt) {
  // get all factors of num which smaller or equal to maxNum
  for (int64_t i = 1; i <= bm_dim / kNumTwo; i++) {
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
    int64_t bm_dim = params.core_num / tiling.n_dim;
    vector<int64_t> bm_dim_factor_opt;
    GetFactors(bm_dim, bm_dim_factor_opt);
    for (auto &bm_dim_factor : bm_dim_factor_opt) {
      if (params.batch % (bm_dim / bm_dim_factor) == 0 && params.ho * params.stride_h % bm_dim_factor == 0) {
        tiling.batch_dim = bm_dim / bm_dim_factor;
        tiling.m_dim = bm_dim_factor;
        break;
      }
    }
  }
  return true;
}

bool ModifyNDim(Tiling &tiling, const DxParas &params, const bool &min_hw, const int64_t m1) {
  // modify n dim
  CHECK_OP_FUNC(tiling.m_dim == 0, return false, "mdim is 0");
  bool modify_n_dim = m1 / tiling.m_dim <= kBlockSize && !min_hw;
  if (modify_n_dim) {
    int64_t m_dim_temp = tiling.m_dim;
    vector<int64_t> n_dim_factor_opt;
    GetFactors(params.c1 / tiling.n_dim, n_dim_factor_opt);
    for (auto &n_dim_factor : n_dim_factor_opt) {
      if (m1 / max(kMinCoreNum, m_dim_temp / n_dim_factor) >= kBlockSize) {
        tiling.m_dim = max(m_dim_temp / n_dim_factor, kMinCoreNum);
        tiling.n_dim = min(tiling.n_dim * n_dim_factor, params.core_num);
        break;
      }
    }
  }
  return true;
}

bool GetBlockDim(const DxParas &params, const int64_t &core_num, Tiling &tiling) {
  // get batch_dim, m_dim and n_dim for single core
  // not support multi cores slicing along k dim
  // single core batch_dim, m_dim, n_dim is a factor of input batch, m, n
  int64_t m1 = (params.h * params.w + kBlockSize - 1) / kBlockSize;
  int64_t k2 = params.co1 * params.kh * params.kw;
  if (params.batch * m1 * params.c1 < core_num) {
    CHECK_OP_FUNC(!MdimTune(params, m1, params.c1, m1, tiling) && tiling.m_dim == 0, return false, "ndim, mdim is 0");
    tiling.batch_dim = params.batch;
    tiling.n_dim = params.c1;
    tiling.batch_single_core_size = 1;
    tiling.m_single_core_size = m1 / tiling.m_dim;
    tiling.n_single_core_size = 1;
    tiling.k_single_core_size = k2;
    return true;
  }
  int64_t a_size = params.batch * params.co1 * params.ho * params.wo * kBlockSize;
  if ((params.kh - 1) / params.stride_h <= 1) {
    a_size = params.batch * params.co1 * params.kh * params.kw * params.h * params.w * kBlockSize;
  }
  int64_t b_size = params.co1 * params.c1 * params.kh * params.kw * kBlockSize * kBlockSize;
  // Derivation of load size: a_size * n_dim + b_size * core_num / n_dim
  float n_factor =
      max(min(sqrt(static_cast<float>(core_num * b_size) / static_cast<float>(a_size)), static_cast<float>(params.c1)),
          1.0f);
  CHECK_OP_FUNC(EqualWith(n_factor, kFloatZero), return false, "n_factor is invalid");
  float bm_factor = min(min(static_cast<float>(core_num) / n_factor, static_cast<float>(m1 * params.batch)),
                        static_cast<float>(core_num));
  float b_factor = min(static_cast<float>(params.batch), bm_factor);
  float m_factor = min(static_cast<float>(core_num) / (n_factor * b_factor), static_cast<float>(m1));
  vector<float> factor_temp = {floor(n_factor), floor(b_factor), floor(m_factor)};
  vector<int64_t> factor;
  for (size_t i = 0; i < factor_temp.size(); i++) {
    factor_temp[i] = min(factor_temp[i], static_cast<float>(core_num));
    factor_temp[i] = max(factor_temp[i], static_cast<float>(kMinCoreNum));
    factor.push_back(static_cast<int64_t>(factor_temp[i]));
  }
  n_factor = factor[0];
  b_factor = factor[1];
  m_factor = factor[kIdxTwo];
  vector<int64_t>batch_dim_opti = {0, 0};
  vector<int64_t>n_dim_opti = {0, 0};
  CHECK_OP_FUNC(!GenNearestFactor(b_factor, params.batch, batch_dim_opti), return false, "get b_factor failed");
  CHECK_OP_FUNC(!GenNearestFactor(n_factor, params.c1, n_dim_opti), return false, "get n_factor failed");
  tiling.batch_dim = 1;
  tiling.n_dim = 1;
  tiling.m_dim = 1;
  int64_t min_load_size = tiling.batch_dim * tiling.m_dim * b_size + tiling.n_dim * a_size;
  int64_t core_use = tiling.batch_dim * tiling.n_dim * tiling.m_dim;
  for (auto &batch_dim : batch_dim_opti) {
    for (auto &n_dim : n_dim_opti) {
      if (batch_dim * n_dim > core_num) {
        continue;
      }
      int64_t m_dim_temp =
          min(static_cast<int64_t>(floor(static_cast<float>(core_num) / static_cast<float>(batch_dim * n_dim))), m1);
      int64_t load_size = batch_dim * m_dim_temp * b_size + n_dim * a_size;
      int64_t core_use_temp = batch_dim * n_dim * m_dim_temp;
      bool modify_dim = core_use < core_use_temp || (core_use == core_use_temp and load_size < min_load_size) ||
                        (core_use == core_use_temp and abs(load_size - min_load_size) < kLoadSizeThreshold &&
                                                        tiling.batch_dim * tiling.n_dim < batch_dim * n_dim);
      if (modify_dim) {
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

bool CheckL0Overflow(const int64_t &m0, const int64_t &n0, const int64_t &k0) {
  bool l0_invalid = (m0 * k0 * kFp16Bytes * kDbOn * kBlockSize * kBlockSize > kL0aSize) ||
                    (n0 * m0 * kFp32Bytes * kBlockSize * kBlockSize * kDbOn > kL0cSize) ||
                    (n0 * k0 * kFp16Bytes * kDbOn * kBlockSize * kBlockSize > kL0bSize);
  return !l0_invalid;
}

bool CheckL1Overflow(const DxParas &params, const Tiling &tiling,
                     const int64_t &m0, const int64_t &n0, const int64_t &k0) {
  int64_t l1_fp16_size = kL1Size / kFp16Bytes;
  int64_t m_al1 = 1;
  int64_t n_bl1 = 1;
  std::vector<int64_t> k_al1_optional = {0, 0};
  CHECK_OP_FUNC(
      !GenNearestFactor((k0 + params.kh * params.kw - 1) / (params.kh * params.kw), params.co1, k_al1_optional),
      return false, "get k_al1_factor failed");
  int64_t k_al1 = k_al1_optional[1];
  int64_t hosh = (params.kh - 1) + m_al1 * m0 * kBlockSize / params.w + kHoshWNoDivided;
  if (m_al1 * m0 * kBlockSize < params.w) {
    hosh = (params.kh - 1) + kHoshWNoDivided;
  } else if (m_al1 * m0 * kBlockSize % params.w == 0) {
    hosh = (params.kh - 1) + m_al1 * m0 * kBlockSize / params.w;
  }
  int64_t b_l1_size =
      k_al1 * params.kh * params.kw * n_bl1 * n0 * kC0 * kBlockSize;
  int64_t a_l1_size = k_al1 * params.wo * params.stride_w * kC0 * hosh;
  if (m_al1 * m0 == tiling.m_single_core_size && k_al1 == params.co1) {
    int64_t hw_ceil_align = CeilAlign(params.ho * params.stride_h * params.wo * params.stride_w, kBlockSize);
    CHECK_OP_FUNC(hw_ceil_align == 0, return false, "hw_ceil_align is invalid");
    a_l1_size = k_al1 * kBlockSize * hw_ceil_align;
  }
  return a_l1_size + b_l1_size <= l1_fp16_size;
}

bool CheckUbDb(const DxParas &params, const Tiling &tiling, const int64_t &m0) {
  int64_t aub_h = 1;
  int64_t aub_k = 1;
  int64_t cub_n = 1;
  int64_t ub_fp16_size = kUbSize / kFp16Bytes;
  int64_t loadin_size = aub_k * aub_h * params.wo * kBlockSize * params.stride_w;
  int64_t copyout_size = kAfterUbFusionMulti * cub_n * m0 * kBlockSize * kBlockSize;
  if (params.stride_h == 1 && params.stride_w == 1) {
    loadin_size = kFrontUbFusionMulti * aub_k * kBlockSize *
              ((aub_h * params.wo * params.stride_w + kBlockSize - 1) / kBlockSize) * kBlockSize;
    return loadin_size * tiling.db_aub + copyout_size * tiling.db_cub <= ub_fp16_size;
  }
  return loadin_size * tiling.db_aub + copyout_size * tiling.db_cub <= ub_fp16_size;
}

bool GetSameCutOff(const int64_t &first, const int64_t &seccond, const int64_t &factor,
                   vector<vector<int64_t>> &cand) {
  CHECK_OP_FUNC(factor == 0, return false, "factor is 0");
  int64_t mt = first / factor;
  int64_t factor_fir = first / (mt + 1) + 1;
  int64_t factor_sec = kL0cNzSize / factor_fir;
  mt = first / factor_fir;
  int64_t mt_1 = seccond / factor_sec;
  cand.push_back({factor_fir, factor_sec});
  for (int64_t i = factor_fir + 1; i <= first; i++) {
    factor_sec = kL0cNzSize / i;
    if (first / i == mt && seccond / factor_sec == mt_1) {
      cand.push_back({i, factor_sec});
    } else {
      break;
    }
  }
  return true;
}

bool PushMNCandidate(const DxParas &params, const int64_t &cand_m, const int64_t &cand_n, Tiling &tiling,
                     vector<int64_t>& mn_cand_mkn) {
  int64_t cand_k =
      min(min(static_cast<int64_t>(floor(static_cast<float>(kL0aNzSize) / static_cast<float>(cand_m))),
              static_cast<int64_t>(floor(static_cast<float>(kL0bNzSize) / static_cast<float>(cand_n)))),
          tiling.k_single_core_size);
  if (cand_m == 0 || cand_n == 0) {
    return false;
  }
  int64_t load_size = ((tiling.m_single_core_size - 1) / cand_m) * tiling.n_single_core_size +
                      ((tiling.n_single_core_size - 1) / cand_n) * tiling.m_single_core_size;
  int64_t mkn = cand_m * cand_n * cand_k;
  float max_mk = static_cast<float>(max(cand_m, cand_k)) / static_cast<float>(min(cand_m, cand_k));
  bool l0_invalid = !CheckL0Overflow(cand_m, cand_n, cand_k) ||
                    !CheckL1Overflow(params, tiling, cand_m, cand_n, cand_k) || !CheckUbDb(params, tiling, cand_m);
  if (l0_invalid) {
    return false;
  }
  if (mn_cand_mkn.size() == 0) {
    mn_cand_mkn = {cand_m, cand_n, cand_k, load_size, mkn};
  } else {
    float max_mk_temp = static_cast<float>(max(mn_cand_mkn[0], mn_cand_mkn[kIdxTwo])) /
                        static_cast<float>(min(mn_cand_mkn[0], mn_cand_mkn[kIdxTwo]));
    bool l0_valid =
        (load_size < mn_cand_mkn[kIdxThree]) ||
        ((load_size == mn_cand_mkn[kIdxThree]) && (mkn > mn_cand_mkn[kIdxFour] ||
                                           (mkn == mn_cand_mkn[kIdxFour] && max_mk > max_mk_temp)));
    if (l0_valid) {
      mn_cand_mkn = {cand_m, cand_n, cand_k, load_size, mkn};
    }
  }
  return true;
}

bool GetCandMKN(const DxParas &params, Tiling &tiling) {
  // get cand from m
  vector<vector<int64_t>> cand;
  vector<int64_t> mn_cand_mkn;
  int64_t factor = kM0N0OptimalNode;
  CHECK_OP_FUNC(!GetSameCutOff(tiling.m_single_core_size - 1, tiling.n_single_core_size - 1, factor, cand),
                return false, "m get cut off failed");
  for (auto &mn : cand) {
    PushMNCandidate(params, mn[0], mn[1], tiling, mn_cand_mkn);
  }
  // m right
  int64_t mt = (tiling.m_single_core_size - 1) / factor;
  int64_t mt_r = (tiling.m_single_core_size - 1) / (mt + 1);
  vector<vector<int64_t>> cand_mt_r;
  CHECK_OP_FUNC(!GetSameCutOff(tiling.m_single_core_size - 1, tiling.n_single_core_size - 1, mt_r, cand_mt_r),
                return false, "mt_r get cut off failed");
  for (auto &mn : cand_mt_r) {
    PushMNCandidate(params, mn[0], mn[1], tiling, mn_cand_mkn);
  }
  // m_left
  int64_t mt_l = mt > kNumTwo ? (tiling.m_single_core_size - 1) / (mt - 1) : tiling.m_single_core_size;
  vector<vector<int64_t>> cand_mt_l;
  CHECK_OP_FUNC(!GetSameCutOff(tiling.m_single_core_size - 1, tiling.n_single_core_size - 1, mt_l, cand_mt_l),
                return false, "mt_l get cut off failed");
  for (auto &mn : cand_mt_l) {
    PushMNCandidate(params, mn[0], mn[1], tiling, mn_cand_mkn);
  }
  // get cand from n
  vector<vector<int64_t>> cand_n;
  CHECK_OP_FUNC(!GetSameCutOff(tiling.n_single_core_size - 1, tiling.m_single_core_size - 1, factor, cand_n),
                return false, "n get cut off failed");
  for (auto &mn : cand_n) {
    PushMNCandidate(params, mn[1], mn[0], tiling, mn_cand_mkn);
  }
  // n right
  int64_t nt = (tiling.n_single_core_size - 1) / factor;
  int64_t nt_r = (tiling.n_single_core_size - 1) / (nt + 1);
  vector<vector<int64_t>> cand_nt_r;
  CHECK_OP_FUNC(!GetSameCutOff(tiling.n_single_core_size - 1, tiling.m_single_core_size - 1, nt_r, cand_nt_r),
                return false, "nt_r get cut off failed");
  for (auto &mn : cand_nt_r) {
    PushMNCandidate(params, mn[1], mn[0], tiling, mn_cand_mkn);
  }
  // n_left
  int64_t nt_l = nt > kNumTwo ? (tiling.n_single_core_size - 1) / (nt - 1) : tiling.n_single_core_size;
  vector<vector<int64_t>> cand_nt_l;
  CHECK_OP_FUNC(!GetSameCutOff(tiling.n_single_core_size - 1, tiling.m_single_core_size - 1, nt_l, cand_nt_l),
                return false, "nt_l get cut off failed");
  for (auto &mn : cand_nt_l) {
    PushMNCandidate(params, mn[1], mn[0], tiling, mn_cand_mkn);
  }
  if (mn_cand_mkn.size() != 0) {
    tiling.m_l0 = mn_cand_mkn[0];
    tiling.n_l0 = mn_cand_mkn[1];
    tiling.k_l0 = mn_cand_mkn[kIdxTwo];
  }
  return true;
}

bool GetNMFactor(const int64_t &factor, const int64_t &k2, const int64_t &max_factor, int64_t &nm) {
  if (factor == 0) {
    return false;
  }
  int64_t k0 = min(kL0aNzSize / factor, k2);
  if (k0 == 0) {
    return false;
  }
  nm = min(min(kL0cNzSize / factor, kL0aNzSize / k0), max_factor);
  return true;
}

bool GetCutOffPoint(Tiling &tiling, vector<int64_t> &m0_factor, vector<int64_t> &n0_factor) {
  int64_t nm;
  CHECK_OP_FUNC(!GetNMFactor(tiling.m_l0, tiling.k_single_core_size, tiling.n_single_core_size, nm),
                return false, "tiling param invalid");
  if (std::find(n0_factor.begin(), n0_factor.end(), nm) == n0_factor.end()) {
    n0_factor.push_back(nm);
  }
  CHECK_OP_FUNC(!GetNMFactor(tiling.n_l0, tiling.k_single_core_size, tiling.m_single_core_size, nm),
                return false, "tiling param invalid");
  if (std::find(m0_factor.begin(), m0_factor.end(), nm) == m0_factor.end()) {
    m0_factor.push_back(nm);
  }
  // m right
  int64_t mt = max((tiling.m_single_core_size - 1) / tiling.m_l0, 1L);
  int64_t mt_r = mt > 0 ? (tiling.m_single_core_size - 1) / mt + 1 : tiling.m_single_core_size;
  mt_r = min(mt_r, kL0aNzSize);
  if (std::find(m0_factor.begin(), m0_factor.end(), mt_r) == m0_factor.end()) {
    m0_factor.push_back(mt_r);
    CHECK_OP_FUNC(!GetNMFactor(mt_r, tiling.k_single_core_size, tiling.n_single_core_size, nm),
                  return false, "tiling param invalid");
    if (std::find(n0_factor.begin(), n0_factor.end(), nm) == n0_factor.end()) {
      n0_factor.push_back(nm);
    }
  }
  // m left
  int64_t mt_l = max((tiling.m_single_core_size - 1) / (mt + 1), 1L);
  mt_l = min(mt_l, kL0aNzSize);
  if (std::find(m0_factor.begin(), m0_factor.end(), mt_l) == m0_factor.end()) {
    m0_factor.push_back(mt_l);
    CHECK_OP_FUNC(!GetNMFactor(mt_l, tiling.k_single_core_size, tiling.n_single_core_size, nm),
                  return false, "tiling param invalid");
    if (std::find(n0_factor.begin(), n0_factor.end(), nm) == n0_factor.end()) {
      n0_factor.push_back(nm);
    }
  }
  // n right
  int64_t nt = max((tiling.n_single_core_size - 1) / tiling.n_l0, 1L);
  int64_t nt_r = nt > 0 ? (tiling.n_single_core_size - 1) / nt + 1 : tiling.n_single_core_size;
  nt_r = min(nt_r, kL0bNzSize);
  if (std::find(n0_factor.begin(), n0_factor.end(), nt_r) == n0_factor.end()) {
    n0_factor.push_back(nt_r);
    CHECK_OP_FUNC(!GetNMFactor(nt_r, tiling.k_single_core_size, tiling.m_single_core_size, nm),
                  return false, "tiling param invalid");
    if (std::find(m0_factor.begin(), m0_factor.end(), nm) == m0_factor.end()) {
      m0_factor.push_back(nm);
    }
  }
  // n left
  int64_t nt_l = max((tiling.n_single_core_size - 1) / (nt + 1), 1L);
  nt_l = min(nt_l, kL0bNzSize);
  if (std::find(n0_factor.begin(), n0_factor.end(), nt_l) == n0_factor.end()) {
    n0_factor.push_back(nt_l);
    CHECK_OP_FUNC(!GetNMFactor(nt_l, tiling.k_single_core_size, tiling.m_single_core_size, nm),
                  return false, "tiling param invalid");
    if (std::find(m0_factor.begin(), m0_factor.end(), nm) == m0_factor.end()) {
      n0_factor.push_back(nm);
    }
  }
  return true;
}

bool GetL0FactorsOptiNew(const DxParas &params, Tiling &tiling) {
  // x0 is optimal of min load size equation of ((m2-1)/m0)*n2 + ((n2-1)/n0)*m2
  int64_t x0 = kM0N0OptimalNode;
  tiling.k_l0 = 1;
  tiling.m_l0 = 1;
  tiling.n_l0 = 1;
  if (tiling.m_single_core_size <= x0 && tiling.n_single_core_size <= x0) {
    tiling.k_l0 = min(min(kL0aNzSize / tiling.m_single_core_size, kL0bNzSize / tiling.n_single_core_size),
                      tiling.k_single_core_size);
    tiling.m_l0 = min(kL0aNzSize / tiling.k_l0, tiling.m_single_core_size);
    tiling.n_l0 = min(kL0bNzSize / tiling.k_l0, tiling.n_single_core_size);
  } else if (tiling.m_single_core_size > x0 && tiling.n_single_core_size > x0) {
    CHECK_OP_FUNC(!GetCandMKN(params, tiling), return false, "get candiate mkn factor failed");
  } else if (tiling.m_single_core_size <= x0 && tiling.n_single_core_size > x0) {
    int64_t m_temp = min(tiling.m_single_core_size, kL0aNzSize);
    CHECK_OP_FUNC(m_temp == 0, return false, "m_temp invalid");
    int64_t n_temp = min(min(kL0cNzSize / m_temp, tiling.n_single_core_size), kL0bNzSize);
    CHECK_OP_FUNC(n_temp == 0, return false, "n_temp invalid");
    tiling.k_l0 = min(min(kL0aNzSize / m_temp, kL0bNzSize /n_temp), tiling.k_single_core_size);
    tiling.m_l0 = min(kL0aNzSize / tiling.k_l0, m_temp);
    tiling.n_l0 = min(kL0bNzSize / tiling.k_l0, n_temp);
  } else if (tiling.n_single_core_size <= x0 && tiling.m_single_core_size > x0) {
    int64_t n_temp = min(tiling.n_single_core_size, kL0bNzSize);
    CHECK_OP_FUNC(n_temp == 0, return false, "n_temp invalid");
    int64_t m_temp = min(min(kL0cNzSize / n_temp, tiling.m_single_core_size), kL0aNzSize);
    CHECK_OP_FUNC(m_temp == 0, return false, "m_temp invalid");
    tiling.k_l0 = min(min(kL0aNzSize / m_temp, kL0bNzSize / n_temp), tiling.k_single_core_size);
    tiling.m_l0 = min(kL0aNzSize / tiling.k_l0, m_temp);
    tiling.n_l0 = min(kL0bNzSize / tiling.k_l0, n_temp);
  }
  vector<int64_t> m0_factor = {tiling.m_l0};
  vector<int64_t> n0_factor = {tiling.n_l0};
  CHECK_OP_FUNC(!GetCutOffPoint(tiling, m0_factor, n0_factor), return false, "get cutoff of mkn factor failed");
  int64_t min_load_size = ((tiling.m_single_core_size - 1) / tiling.m_l0) * tiling.n_single_core_size +
                          ((tiling.n_single_core_size - 1) / tiling.n_l0) * tiling.m_single_core_size;
  int64_t max_mkn = tiling.m_l0 * tiling.k_l0 * tiling.n_l0;
  int64_t max_mk = max(tiling.m_l0, tiling.k_l0) / min(tiling.m_l0, tiling.k_l0);
  int64_t load_size;
  int64_t mkn;
  int64_t mk;
  bool l0_invalid = tiling.k_l0 <= 0 || tiling.m_l0 * tiling.n_l0 > kL0cNzSize ||
                    !CheckL0Overflow(tiling.m_l0, tiling.n_l0, tiling.k_l0) ||
                    !CheckL1Overflow(params, tiling, tiling.m_l0, tiling.n_l0, tiling.k_l0) ||
                    !CheckUbDb(params, tiling, tiling.m_l0);
  bool l0_valid_res = !l0_invalid;
  bool l0_valid = false;
  for (auto &m0 : m0_factor) {
    for (auto &n0 : n0_factor) {
      int64_t k0 = min(min(kL0aNzSize / m0, kL0bNzSize / n0), tiling.k_single_core_size);
      l0_invalid = k0 <= 0 || m0 * n0 > kL0cNzSize || !CheckL0Overflow(m0, n0, k0) ||
                        !CheckL1Overflow(params, tiling, m0, n0, k0) || !CheckUbDb(params, tiling, m0);
      if (l0_invalid) {
        continue;
      }
      load_size = ((tiling.m_single_core_size - 1) / m0) * tiling.n_single_core_size +
                  ((tiling.n_single_core_size - 1) / n0) * tiling.m_single_core_size;
      mkn = m0 * n0 * k0;
      mk = max(m0, k0) / min(m0, k0);
      l0_valid =
          (load_size < min_load_size) ||
          ((load_size == min_load_size) && (mkn > max_mkn ||
                                            (mkn == max_mkn && mk > max_mk)));
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

bool GetL0Factors(const DxParas &params, Tiling &tiling) {
  // get m_l0, n_l0, k_l0 factor when singlecore m, n, k is know
  // m_l0, n_l0, k_l0 is a factor of single core m, n, k
  tiling.db_cub = kDbOn;
  tiling.db_aub = kDbOn;
  tiling.db_l0c = kDbOn;
  if (GetL0FactorsOptiNew(params, tiling)) {
    return true;
  }
  tiling.db_cub = kDbOn;
  tiling.db_aub = kDbOff;
  if (GetL0FactorsOptiNew(params, tiling)) {
    return true;
  }
  tiling.db_cub = kDbOff;
  tiling.db_aub = kDbOff;
  if (GetL0FactorsOptiNew(params, tiling)) {
    return true;
  }
  return false;
}

bool GetInitialL1(const DxParas &params, Tiling &tiling, vector<int64_t> &min_kl1_dim) {
  CHECK_OP_FUNC(
      !GenNearestFactor((tiling.k_l0 + params.kh * params.kw - 1) / (params.kh * params.kw), params.co1, min_kl1_dim),
      return false, "get k_l1_factor failed");
  tiling.init_db_al1 = kDbOn;
  tiling.init_db_bl1 = kDbOn;
  tiling.m_al1 = 1;
  tiling.n_bl1 = 1;
  tiling.k_al1 = min_kl1_dim[1];
  tiling.k_bl1 = min_kl1_dim[1];
  return true;
}

int64_t GetAl1MExtent(const int64_t& al1_m, const DxParas &params) {
  if (al1_m == 0) {
    return 0;
  }
  int64_t al1_h_small = (params.w % al1_m == 0) ? params.kh : params.kh + 1;
  int64_t al1_h_large =
      (al1_m % params.w == 0) ? params.kh + (al1_m / params.w) - 1 : params.kh + (al1_m / params.w) + 1;
  int64_t al1_h = (al1_m < params.w) ? al1_h_small : al1_h_large;
  return al1_h;
}

bool GetHosh(const DxParas &params, const Tiling &tiling, const vector<int64_t> &kn_factors, const int64_t &h2,
             vector<int64_t> &l1_para) {
  int64_t l1_fp16_size = kL1Size / kFp16Bytes;
  size_t idx = 0;
  int64_t k_bl1 = kn_factors[idx++];
  int64_t k_al1 = kn_factors[idx++];
  int64_t nbl1 = kn_factors[idx++];
  int64_t m_size = (l1_fp16_size -
                    k_bl1 * params.kh * params.kw * nbl1 * tiling.n_l0 * kBlockSize * kC0 * tiling.init_db_bl1) /
                   tiling.init_db_al1;
  if (k_bl1 == params.co1 && static_cast<int64_t>(ceil(static_cast<double>(tiling.n_single_core_size) /
                                                    static_cast<double>(nbl1 * tiling.n_l0))) == 1) {
    m_size = (l1_fp16_size - k_bl1 * params.kh * params.kw * tiling.n_single_core_size * kBlockSize * kC0) /
             tiling.init_db_al1;
  }
  int64_t hosh = h2 < params.kh + 1 ? params.kh + 1 : m_size / (k_al1 * params.wo * params.stride_w * kC0);
  if (k_al1 == params.co1) {
    if (hosh >= params.kh + 1) {
      int64_t m_1;
      if (hosh >= params.kh + kHoshWNoDivided) {
        m_1 = max(((hosh - params.kh - 1 + 1) * params.w - 1) / (tiling.m_l0 * kBlockSize), 1L);
      } else {
        m_1 = max((params.kh + 1 - params.kh) * params.w / (tiling.m_l0 * kBlockSize), 1L);
      }
      vector<int64_t> m_1_factor = {0, 0};
      CHECK_OP_FUNC(
          !GenNearestFactor(m_1,
                            static_cast<int64_t>(ceil(static_cast<double>(params.h * params.w) /
                                                      static_cast<double>(tiling.m_dim * tiling.m_l0 * kBlockSize))),
                            m_1_factor),
          return false, "get m_1 failed");
      m_1 = m_1_factor[1];
      hosh = GetAl1MExtent(m_1 * tiling.m_l0 * kBlockSize, params);
      CHECK_OP_FUNC(hosh == 0, return false, "get hosh failed");
      l1_para = {hosh, m_1};
    } else {
      l1_para = {0, 0};
    }
  } else {
    hosh = params.kh - 1 + (tiling.m_l0 * kBlockSize / params.w) + kHoshWNoDivided;
    if (tiling.m_l0 * kBlockSize < params.w) {
      hosh = params.kh - 1 + kHoshWNoDivided;
    }
    l1_para = {hosh, 1};
  }
  if (tiling.m_l0 * l1_para[1] == tiling.m_single_core_size && k_al1 == params.co1) {
    l1_para[0] = h2 * params.stride_h;
  }
  return true;
}

vector<int64_t> GetMinloadSize(map<std::string, vector<int64_t>> &factor_size, Tiling &tiling, const DxParas &params,
                               const int64_t &load_h) {
  vector<int64_t> db_size = {0, 0, 0};
  vector<int64_t> kn_factors = factor_size["kn_factors"];
  size_t idx = 0;
  int64_t k_bl1 = kn_factors[idx++];
  int64_t k_al1 = kn_factors[idx++];
  int64_t nbl1 = kn_factors[idx++];
  vector<int64_t> size_para = factor_size["size_para"];
  idx = 0;
  int64_t h2 = size_para[idx++];
  int64_t a_size = size_para[idx++];
  int64_t b_size = size_para[idx++];
  int64_t db_bl1 = tiling.init_db_bl1;
  int64_t db_al1 = tiling.init_db_al1;
  if (static_cast<int64_t>(
          ceil(static_cast<double>(tiling.n_single_core_size) / static_cast<double>(nbl1 * tiling.n_l0))) == 1 &&
      k_bl1 == params.co1) {
    db_bl1 = 1;
    db_size[0] = a_size + b_size;
    db_size[1] = db_al1;
    db_size[kIdxTwo] = db_bl1;
    return db_size;
  }
  if (static_cast<int64_t>(ceil(static_cast<double>(h2) / static_cast<double>(load_h))) == 1 &&
      k_al1 == params.co1) {
    db_al1 = 1;
    db_size[0] = a_size + b_size;
    db_size[1] = db_al1;
    db_size[kIdxTwo] = db_bl1;
    return db_size;
  }
  db_size[0] = static_cast<int64_t>(ceil(static_cast<double>(h2) / static_cast<double>(load_h))) * b_size +
               static_cast<int64_t>(
                   ceil(static_cast<double>(tiling.n_single_core_size) / static_cast<double>(nbl1 * tiling.n_l0))) *
                   a_size;
  db_size[1] = db_al1;
  db_size[kIdxTwo] = db_bl1;
  return db_size;
}

bool CheckL1Size(const vector<int64_t> &kn_factors, const vector<int64_t> &m_h,
                 const Tiling &tiling, const int64_t &load_h, const DxParas &params) {
  int64_t a_size;
  size_t idx = 0;
  int64_t k_bl1 = kn_factors[idx++];
  int64_t k_al1 = kn_factors[idx++];
  int64_t nbl1 = kn_factors[idx++];
  idx = 0;
  int64_t m_1 = m_h[idx++];
  int64_t h2 = m_h[idx++];
  int64_t db_al1_end = m_h[idx++];
  int64_t db_bl1_end = m_h[idx++];
  int64_t b_size =
      k_bl1 * kBlockSize * params.kh * params.kw * nbl1 * tiling.n_l0 * kBlockSize * db_bl1_end * kFp16Bytes;
  if (static_cast<int64_t>(
          ceil(static_cast<double>(tiling.n_single_core_size) / static_cast<double>(nbl1 * tiling.n_l0))) == 1 &&
      k_bl1 == params.co1) {
    b_size = params.co1 * kBlockSize * params.kh * params.kw * nbl1 * tiling.n_l0 * kBlockSize * kFp16Bytes;
  }
  if (static_cast<int64_t>(ceil(static_cast<double>(h2) / static_cast<double>(load_h))) == 1 &&
      k_al1 == params.co1) {
    a_size = h2 * params.stride_h * params.wo * params.stride_w * params.co1 * kBlockSize * kFp16Bytes;
  } else {
    int64_t hosh = (params.kh - 1) + kHoshWNoDivided + (m_1 * tiling.m_l0 * kBlockSize) / params.w;
    if (m_1 * tiling.m_l0 * kBlockSize < params.w) {
      hosh = (params.kh - 1) + kHoshWNoDivided;
    } else if ((m_1 * tiling.m_l0 * kBlockSize) % params.w == 0) {
      hosh = (params.kh - 1) + (m_1 * tiling.m_l0 * kBlockSize) / params.w;
    }
    a_size = hosh * params.wo * params.stride_w * k_al1 * kBlockSize * db_al1_end * kFp16Bytes;
  }
  return a_size + b_size <= kL1Size;
}

bool GetL1FactorsOpti(map<std::string, vector<int64_t>> &factor_size,
                      const DxParas &params, Tiling &tiling, int64_t &min_load_size, bool &first_flag) {
  vector<int64_t> kn_factors = factor_size["kn_factors"];
  size_t idx = 0;
  int64_t k_bl1 = kn_factors[idx++];
  int64_t k_al1 = kn_factors[idx++];
  int64_t nbl1 = kn_factors[idx++];
  vector<int64_t> size_para = factor_size["size_para"];
  idx = 0;
  int64_t h2 = size_para[idx++];
  vector<int64_t> min_kl1_dim = factor_size["min_kl1_dim"];
  if (nbl1 > 1 && k_bl1 < params.co1) {
    return true;
  }
  bool modify_l1 = (k_bl1 % k_al1 == 0 || k_al1 % k_bl1 == 0) &&
                   (k_bl1 >= min_kl1_dim[1] && k_al1 >= min_kl1_dim[1]) &&
                   (k_bl1 * params.kh * params.kw) % tiling.k_l0 == 0 &&
                   (k_al1 * params.kh * params.kw) % tiling.k_l0 == 0;
  if (modify_l1) {
    vector<int64_t> l1_para;
    CHECK_OP_FUNC(!GetHosh(params, tiling, kn_factors, h2, l1_para), return false, "get hosh failed");
    int64_t hosh = l1_para[0];
    int64_t m_1 = l1_para[1];
    if (hosh != 0 && m_1 != 0) {
      int64_t load_h = static_cast<int64_t>(ceil(static_cast<double>(hosh) / static_cast<double>(params.stride_h)));
      vector<int64_t> db_size = GetMinloadSize(factor_size, tiling, params, load_h);
      int64_t load_size = db_size[0];
      int64_t db_al1_end = db_size[1];
      int64_t db_bl1_end = db_size[kIdxTwo];
      vector<int64_t> m_h = {m_1, h2, db_al1_end, db_bl1_end};
      modify_l1 = CheckL1Size(kn_factors, m_h, tiling, load_h, params) &&
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
        tiling.hosh = hosh;
        tiling.db_al1 = db_al1_end;
        tiling.db_bl1 = db_bl1_end;
        min_load_size = load_size;
        first_flag = false;
      }
    }
  }
  return true;
}

bool GetL1Factors(const DxParas &params, Tiling &tiling) {
  // get m_al1, n_bl1, kal1_16, kbl1_16 factors when L0, singlecore factor is know
  // get al1, bl1 double buffer factors
  tiling.db_al1 = kDbOn;
  tiling.db_bl1 = kDbOn;
  vector<int64_t> min_kl1_dim = {0, 0};
  CHECK_OP_FUNC(!GetInitialL1(params, tiling, min_kl1_dim), return false, "initial l1 failed");
  int64_t l1_fp16_size = kL1Size / kFp16Bytes;
  int64_t hosh;
  if (tiling.m_al1 * tiling.m_l0 * kBlockSize < params.w) {
    hosh = (params.kh - 1) + kHoshWNoDivided;
  } else if ((tiling.m_al1 * tiling.m_l0 * kBlockSize) % params.w == 0) {
    hosh = (params.kh - 1) + (tiling.m_al1 * tiling.m_l0 * kBlockSize) / params.w;
  } else {
    hosh = (params.kh - 1) + (tiling.m_al1 * tiling.m_l0 * kBlockSize) / params.w + kHoshWNoDivided;
  }
  int64_t b_l1_size =
      tiling.k_bl1 * params.kh * params.kw * tiling.n_bl1 * tiling.n_l0 * kC0 * tiling.init_db_bl1 * kBlockSize;
  int64_t a_l1_size = tiling.k_al1 * params.wo * params.stride_w * kC0 * hosh * tiling.init_db_al1;
  if (b_l1_size + a_l1_size > l1_fp16_size) {
    tiling.init_db_al1 = 1;
  }
  a_l1_size = tiling.k_al1 * params.wo * params.stride_w * kC0 * hosh * tiling.init_db_al1;
  if (b_l1_size + a_l1_size > l1_fp16_size) {
    tiling.init_db_bl1 = 1;
  }
  int64_t b_size = params.co1 * params.kh * params.kw * kC0 * kC0 * tiling.n_single_core_size;
  int64_t h2 = static_cast<int64_t>(
      ceil((ceil(static_cast<double>(tiling.m_single_core_size * kBlockSize) / static_cast<double>(params.w)) +
            static_cast<double>(params.kh - 1)) /
           static_cast<double>(params.stride_h)));
  h2 = min(h2, params.ho);
  int64_t a_size = h2 * params.co1 * params.wo * kC0;
  int64_t real_h = static_cast<int64_t>(ceil(static_cast<double>(hosh) / static_cast<double>(params.stride_h)));
  int64_t min_load_size =
      static_cast<int64_t>(ceil(static_cast<double>(h2) / static_cast<double>(real_h))) * b_size +
      static_cast<int64_t>(ceil(static_cast<double>(tiling.n_single_core_size) / static_cast<double>(tiling.n_l0))) *
          a_size;
  vector<int64_t> k_factors;
  GetFactors(params.co1, k_factors);
  vector<int64_t> nl1_factors;
  GetFactors(tiling.n_single_core_size / tiling.n_l0, nl1_factors);
  bool first_flag = true;
  for (auto &k_bl1 : k_factors) {
    for (auto &k_al1 : k_factors) {
      for (auto &nbl1 : nl1_factors) {
        vector<int64_t> kn_factors = {k_bl1, k_al1, nbl1};
        vector<int64_t> size_para = {h2, a_size, b_size};
        map<std::string, vector<int64_t>> factor_size = {
            {"kn_factors", kn_factors}, {"size_para", size_para}, {"min_kl1_dim", min_kl1_dim}};
        CHECK_OP_FUNC(!GetL1FactorsOpti(factor_size, params, tiling, min_load_size, first_flag), return false,
                      "get L1Factor failed");
      }
    }
  }
  return true;
}

void GetAubM(const int64_t &aub_size, const DxParas &params,
             const int64_t &k_aub, const Tiling &tiling, int64_t &aub_m) {
  aub_m = 1;
  int64_t aub_in = aub_size / (k_aub * params.wo * kC0);
  if (params.stride_h != 1 || params.stride_w != 1) {
    for (int64_t hosh_temp = tiling.hosh; hosh_temp >= aub_m + 1; hosh_temp--) {
      if (hosh_temp * params.stride_w <= aub_in) {
        aub_m = hosh_temp;
        break;
      }
    }
  } else {
    for (int64_t hosh_temp = tiling.hosh; hosh_temp >= aub_m + 1; hosh_temp--) {
      if (kFrontUbFusionMulti * k_aub * kBlockSize *
              ((hosh_temp * params.wo * params.stride_w + kBlockSize - 1) / kBlockSize) * kBlockSize <=
          aub_size) {
        aub_m = hosh_temp;
        break;
      }
    }
  }
}

bool GetUbFactors(const DxParas &params, Tiling &tiling) {
  tiling.m_aub = 1;
  tiling.k_aub = 1;
  tiling.n_cub = 1;
  int64_t ub_fp16_size = kUbSize / kFp16Bytes;
  vector<int64_t> n_l0_factors;
  vector<int64_t> k_al1_factors;
  GetFactors(tiling.n_l0, n_l0_factors);
  GetFactors(tiling.k_al1, k_al1_factors);
  int64_t loadin_size = tiling.k_aub * tiling.m_aub * params.wo * kC0 * params.stride_w;
  int64_t copyout_size = kAfterUbFusionMulti * tiling.n_cub * tiling.m_l0 * kC0 * kC0;
  if (params.stride_h != 1 || params.stride_w != 1) {
    if (loadin_size * tiling.db_aub + copyout_size * tiling.db_cub > ub_fp16_size) {
      tiling.db_aub = 1;
    }
    if (loadin_size * tiling.db_aub + copyout_size * tiling.db_cub > ub_fp16_size) {
      tiling.db_cub = 1;
    }
  } else {
    loadin_size = kFrontUbFusionMulti * tiling.k_aub * kBlockSize *
                  ((tiling.m_aub * params.wo * params.stride_w + kBlockSize - 1) / kBlockSize) * kBlockSize;
    if (loadin_size * tiling.db_aub + copyout_size * tiling.db_cub > ub_fp16_size) {
      tiling.db_aub = 1;
    }
    if (loadin_size * tiling.db_aub + copyout_size * tiling.db_cub > ub_fp16_size) {
      tiling.db_cub = 1;
    }
  }
  int64_t max_dma_size = loadin_size * tiling.db_aub + copyout_size * tiling.db_cub;

  bool first_flag = true;
  int64_t aub_m;
  int64_t aub_size;
  int64_t aub_temp_size;
  int64_t dma_size;
  bool modify_ub;
  for (auto &k_aub : k_al1_factors) {
    for (auto &n1 : n_l0_factors) {
      aub_size = (ub_fp16_size - kAfterUbFusionMulti * n1 * tiling.m_l0 * kC0 * kC0 * tiling.db_cub) / tiling.db_aub;
      GetAubM(aub_size, params, k_aub, tiling, aub_m);
      if (k_aub < params.co1) {
        aub_m = 1;
      }
      aub_temp_size = k_aub * aub_m * params.wo * params.stride_w * kC0;
      if (params.stride_h == 1 && params.stride_w == 1) {
        aub_temp_size = kFrontUbFusionMulti * k_aub * kBlockSize *
                        ((aub_m * params.wo * params.stride_w + kBlockSize - 1) / kBlockSize) *
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
  int64_t k2 = params.co1 * params.kh * params.kw;
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

void SetTilingId(const DxParas &params, const Tiling &tiling, string &tilingId) {
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
  int64_t min_kl1_cmp_kl0 = kAttachFlagOne;
  int64_t al1_attach_flag = kAttachFlagZero;
  int64_t bl1_attach_flag = kAttachFlagZero;
  int64_t abkl1_attach_flag = kAttachFlagZero;
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
  int64_t tiling_id_temp = tiling.db_al1;
  tiling_id_temp = tiling_id_temp * kDecimal + tiling.db_bl1;
  tiling_id_temp = tiling_id_temp * kDecimal + tiling.db_l0c;
  tiling_id_temp = tiling_id_temp * kDecimal + abkl1_attach_flag;
  tiling_id_temp = tiling_id_temp * kDecimal + al1_attach_flag;
  tiling_id_temp = tiling_id_temp * kDecimal + bl1_attach_flag;
  tiling_id_temp = tiling_id_temp * kDecimal + min_kl1_cmp_kl0;
  tiling_id_temp = tiling_id_temp * kDecimal + params.stride_expand_flag;
  tilingId = to_string(tiling_id_temp);
}

bool GenTiling(const DxParas &params, Tiling &tiling, string &tiling_id) {
  CHECK_OP_FUNC(!GetBlockDim(params, params.core_num, tiling), return false, "get block dim failed");
  if (!GetL0Factors(params, tiling)) {
    tiling.k_l0 = 1;
    tiling.m_l0 = 1;
    tiling.n_l0 = 1;
  }
  CHECK_OP_FUNC(!GetL1Factors(params, tiling), return false, "get l1 factors failed");
  CHECK_OP_FUNC(!GetUbFactors(params, tiling), return false, "get ub factors failed");
  CheckSpecialTemplate(params, tiling);
  SetTilingId(params, tiling, tiling_id);
  return true;
}
}  // namespace optiling
