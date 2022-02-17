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

/*!
 * \file dw_cache_tiling.cc
 * \brief function of cacheTiling
 */

#include "dw_cache_tiling.h"
using namespace std;

namespace optiling::conv2d_dw {
static const int32_t kNONE = -INT_MAX;
static const int32_t kBlockSize = 16;
static const int32_t kL1Size = (1024 * 1024);
static const int32_t kL0cSize = (256 * 1024);
static const int32_t kUbSize = (256 * 1024);
static const int32_t kFp16Bytes = 2;
static const int32_t kFp32Bytes = 4;
static const int32_t kL1Fp16Size = kL1Size / kFp16Bytes;
static const int32_t kL0ParasComboLen = 2;
static const int32_t kL0FactorCandLen = 2;
static const int32_t kL1FactorCandLen = 2;
static const int32_t kL1CandLen = 4;
static const int32_t kGetFactorParamsLen = 4;
static const int32_t kL1ParasComboLen = 9;
static const int32_t kL1FactorsLen = 6;
static const int32_t kDbOn = 2;
static const int32_t kDbOff = 1;
static const int32_t kAttachFlagZero = 0;
static const int32_t kAttachFlagOne = 1;
static const int32_t kAttachFlagTwo = 2;
static const int32_t kIdxZero = 0;
static const int32_t kIdxOne = 1;
static const int32_t kIdxTwo = 2;
static const int32_t kIdxThree = 3;
static const int32_t kIdxFour = 4;
static const int32_t kIdxFive = 5;
static const int32_t kIdxSix = 6;
static const int32_t kIdxSeven = 7;
static const int32_t kIdxEight = 8;
static const int32_t kDecimal = 5;

inline int32_t Align(int32_t val, int32_t mod) {
  if (mod == 0) {
    return -1;
  }
  return (val + mod - 1) / mod * mod;
}

void Conv2dDwCacheTiling::GetFactors(int32_t &cnt, int32_t *factor_list, int32_t num, int32_t max_num)
{
  // get all factors of num which smaller or equal to maxNum
  for (int32_t i = 1; i < max_num + 1; i++) {
    if (num % i == 0) {
      factor_list[cnt++] = i;
    }
  }
}

void Conv2dDwCacheTiling::GetTwoFactors(int32_t *res, int32_t base, int32_t dim, const int32_t *other_factors)
{
  // for up bigger or equal to base + 1, find the smallest num which is a factor of dim
  // form down smaller or equal to base, find the biggest num which is a factor of dim
  // the result number must be smaller than max_num and bigger than min_num
  // and cur_factor must be a factor of the result number
  // if there is no result number, return defaultNum
  int32_t max_num = other_factors[kIdxZero];
  int32_t min_num = other_factors[kIdxOne];
  int32_t cur_factor = other_factors[kIdxTwo];
  int32_t default_num = other_factors[kIdxThree];
  int32_t cnt = 0;
  int32_t up = base + 1;
  int32_t max_cnt = 2;
  while (up <= max_num) {
    if (dim % up == 0 && up % cur_factor == 0) {
      res[cnt++] = up;
      break;
    }
    up++;
  }
  int32_t down = base;
  while (down >= min_num) {
    if (dim % down == 0 && down % cur_factor == 0) {
      res[cnt++] = down;
      if (cnt == max_cnt) {
        break;
      }
    }
    down--;
  }
  if (cnt == 0) {
    res[cnt] = default_num;
  }
}

void Conv2dDwCacheTiling::GetNearestFactor(int32_t base, int32_t &factor)
{
  factor = min(factor, base);
  while (factor > 0 && base % factor != 0) {
    factor--;
  }
}

int32_t Conv2dDwCacheTiling::GetK2Ho(int32_t k, int32_t wo)
{
  int32_t ho = (k + wo - 1) / wo;
  if (k % wo == 0 || wo % k == 0) {
    return ho;
  } else {
    return ho + 1;
  }
}

int32_t Conv2dDwCacheTiling::GetHo2Hi(int32_t ho, int32_t stride_h, int32_t kh)
{
  int32_t hi = (ho - 1) * stride_h + kh;
  return hi;
}

void Conv2dDwCacheTiling::NeitherFullLoadBlock()
{
  int32_t ml1_factors[kL1FactorCandLen] = {0};
  int32_t nl1_factors[kL1FactorCandLen] = {0};
  int32_t ml1_other_factors[kGetFactorParamsLen] = {blockDimCalculator.max_ml1, 1, 1, 0};
  int32_t nl1_other_factors[kGetFactorParamsLen] = {blockDimCalculator.max_nl1, 1, 1, 0};
  GetTwoFactors(ml1_factors, blockDimCalculator.l1_min_pnt, blockDimCalculator.m2, ml1_other_factors);
  GetTwoFactors(nl1_factors, blockDimCalculator.l1_min_pnt, blockDimCalculator.n2, nl1_other_factors);
  for (auto const &ml1: ml1_factors) {
    if (ml1 == 0) {
      continue;
    }
    int32_t l1_rest_size = kL1Fp16Size - blockDimCalculator.al1_min_load_size * ml1;
    int32_t nl1 = min(l1_rest_size / blockDimCalculator.bl1_min_load_size, blockDimCalculator.max_nl1);
    while (blockDimCalculator.n2 % nl1 != 0) {
      nl1--;
    }
    blockDimCalculator.tmp_load_size_neither =
    blockDimCalculator.m2 * blockDimCalculator.n2 / ml1 + blockDimCalculator.m2 * blockDimCalculator.n2 / nl1;
    if (blockDimCalculator.tmp_load_size_neither < blockDimCalculator.tmp_load_size) {
      blockDimCalculator.tmp_load_size = blockDimCalculator.tmp_load_size_neither;
    }
  }
  for (auto const &nl1: nl1_factors) {
    if (nl1 == 0) {
      continue;
    }
    int32_t l1_rest_size = kL1Fp16Size - blockDimCalculator.bl1_min_load_size * nl1;
    int32_t ml1 = min(l1_rest_size / blockDimCalculator.al1_min_load_size, blockDimCalculator.max_ml1);
    while (blockDimCalculator.m2 % ml1 != 0) {
      ml1--;
    }
    blockDimCalculator.tmp_load_size_neither =
    blockDimCalculator.m2 * blockDimCalculator.n2 / ml1 + blockDimCalculator.m2 * blockDimCalculator.n2 / nl1;
    if (blockDimCalculator.tmp_load_size_neither < blockDimCalculator.tmp_load_size) {
      blockDimCalculator.tmp_load_size = blockDimCalculator.tmp_load_size_neither;
    }
  }
}

void Conv2dDwCacheTiling::UpdateBlockDimRes()
{
  // update block_dims if load_size less or the same load_size and core_use more
  blockDimCalculator.al1_k_full_load_size = blockDimCalculator.k2 * params.k0 * params.co0;
  blockDimCalculator.al1_full_load_size = blockDimCalculator.m2 * blockDimCalculator.al1_k_full_load_size;
  blockDimCalculator.single_core_ho = GetK2Ho(blockDimCalculator.k2 * params.k0, params.wo);
  blockDimCalculator.single_core_hi = GetHo2Hi(blockDimCalculator.single_core_ho, params.stride_h, params.kh);
  blockDimCalculator.bl1_k_full_load_size = blockDimCalculator.single_core_hi * params.wi * params.ci0;
  blockDimCalculator.bl1_full_load_size = params.ci1 * blockDimCalculator.bl1_k_full_load_size;
  blockDimCalculator.al1_full_load = blockDimCalculator.batch2 == 1 &&
    blockDimCalculator.al1_full_load_size + blockDimCalculator.bl1_min_load_size <= kL1Fp16Size;
  blockDimCalculator.bl1_full_load = blockDimCalculator.batch2 == 1 &&
    blockDimCalculator.bl1_full_load_size + blockDimCalculator.al1_min_load_size <= kL1Fp16Size;
  blockDimCalculator.al1_k_full_load = blockDimCalculator.batch2 == 1 &&
    blockDimCalculator.al1_k_full_load_size + blockDimCalculator.bl1_min_load_size <= kL1Fp16Size;
  blockDimCalculator.bl1_k_full_load = blockDimCalculator.batch2 == 1 &&
    blockDimCalculator.bl1_k_full_load_size + blockDimCalculator.al1_min_load_size <= kL1Fp16Size;
  if (blockDimCalculator.al1_k_full_load) {
    int32_t l1_rest_size = kL1Fp16Size - blockDimCalculator.bl1_min_load_size;
    int32_t m_l1 = min(l1_rest_size / blockDimCalculator.al1_k_full_load_size, blockDimCalculator.m2);
    if (m_l1 >= 1) {
      while (blockDimCalculator.m2 % m_l1 != 0) {
        m_l1--;
      }
    }
    int32_t m_outer = blockDimCalculator.m2 / m_l1;
    blockDimCalculator.tmp_load_size_al1k = m_outer * blockDimCalculator.n2 + blockDimCalculator.m2;
  }
  if (blockDimCalculator.bl1_k_full_load) {
    int32_t l1_rest_size = kL1Fp16Size - blockDimCalculator.al1_min_load_size;
    int32_t n_l1 = min(l1_rest_size / blockDimCalculator.bl1_k_full_load_size, blockDimCalculator.n2);
    if (n_l1 >= 1) {
      while (blockDimCalculator.n2 % n_l1 != 0) {
        n_l1--;
      }
    }
    int32_t n_outer = blockDimCalculator.n2 / n_l1;
    blockDimCalculator.tmp_load_size_bl1k = blockDimCalculator.n2 + n_outer * blockDimCalculator.m2;
  }
  bool condition_full_load = blockDimCalculator.al1_full_load || blockDimCalculator.bl1_full_load;
  bool condition_k_full_load = blockDimCalculator.al1_k_full_load && blockDimCalculator.bl1_k_full_load;
  if (condition_full_load) {
    blockDimCalculator.tmp_load_size = blockDimCalculator.m2 + blockDimCalculator.n2;
  } else if (condition_k_full_load) {
    blockDimCalculator.tmp_load_size =
      min(blockDimCalculator.tmp_load_size_al1k, blockDimCalculator.tmp_load_size_bl1k);
  } else if (blockDimCalculator.al1_k_full_load) {
    blockDimCalculator.tmp_load_size = blockDimCalculator.tmp_load_size_al1k;
  } else if (blockDimCalculator.bl1_k_full_load) {
    blockDimCalculator.tmp_load_size = blockDimCalculator.tmp_load_size_bl1k;
  } else {
    NeitherFullLoadBlock();
  }
  bool condition_update_res = blockDimCalculator.tmp_load_size < singlecoreStatus.min_load_size ||
    (blockDimCalculator.tmp_load_size == singlecoreStatus.min_load_size &&
    (blockDimCalculator.batch_dim_factor > singlecoreStatus.batch_dim ||
    (blockDimCalculator.batch_dim_factor == singlecoreStatus.batch_dim &&
    blockDimCalculator.tmp_core_use > singlecoreStatus.core_use)));
  if (condition_update_res) {
    singlecoreStatus.batch_dim = blockDimCalculator.batch_dim_factor;
    singlecoreStatus.n_dim = blockDimCalculator.n_dim_factor;
    singlecoreStatus.m_dim = blockDimCalculator.m_dim_factor;
    singlecoreStatus.h_dim = blockDimCalculator.h_dim_factor;
    singlecoreStatus.batch2 = blockDimCalculator.batch2;
    singlecoreStatus.n2 = blockDimCalculator.n2;
    singlecoreStatus.m2 = blockDimCalculator.m2;
    singlecoreStatus.k2 = blockDimCalculator.k2;
    singlecoreStatus.min_load_size = blockDimCalculator.tmp_load_size;
    singlecoreStatus.core_use = blockDimCalculator.tmp_core_use;
    singlecoreStatus.al1_full_load_size = blockDimCalculator.al1_full_load_size;
    singlecoreStatus.bl1_full_load_size = blockDimCalculator.bl1_full_load_size;
    singlecoreStatus.al1_k_full_load_size = blockDimCalculator.al1_k_full_load_size;
    singlecoreStatus.bl1_k_full_load_size = blockDimCalculator.bl1_k_full_load_size;
  }
}

void Conv2dDwCacheTiling::GetKDimHelper(const int32_t &m_idx, const int32_t &b_factor, const int32_t &n_factor)
{
  int32_t m_factor = blockDimCalculator.m_dim_array[m_idx];
  if (b_factor * n_factor * m_factor > params.max_core_num || b_factor * n_factor * m_factor == 0) {
    return;
  }
  for (int32_t h_idx = 0; h_idx < blockDimCalculator.h_dim_cnt; h_idx++) {
    int32_t h_factor = blockDimCalculator.h_dim_array[h_idx];
    if (b_factor * n_factor * m_factor * h_factor > params.max_core_num || h_factor == 0) {
      break;
    }
    blockDimCalculator.batch_dim_factor = b_factor;
    blockDimCalculator.n_dim_factor = n_factor;
    blockDimCalculator.m_dim_factor = m_factor;
    blockDimCalculator.h_dim_factor = h_factor;
    blockDimCalculator.batch2 = singlecoreStatus.batch1 / b_factor;
    blockDimCalculator.m2 = singlecoreStatus.m1 / m_factor;
    blockDimCalculator.n2 = singlecoreStatus.n1 / n_factor;
    blockDimCalculator.k2 = singlecoreStatus.k1 / h_factor;
    blockDimCalculator.tmp_core_use = b_factor * n_factor * m_factor * h_factor;
    blockDimCalculator.max_ml1 =
        min((kL1Fp16Size - blockDimCalculator.bl1_min_load_size) / blockDimCalculator.al1_min_load_size,
            blockDimCalculator.m2);
    blockDimCalculator.max_nl1 =
        min((kL1Fp16Size - blockDimCalculator.al1_min_load_size) / blockDimCalculator.bl1_min_load_size,
            blockDimCalculator.n2);
    UpdateBlockDimRes();
  }
}

void Conv2dDwCacheTiling::GetBlockDim()
{
  // get batch_dim, m_dim, n_dim, h_dim for single core
  // single core batch_dim, m_dim, n_dim, h_dim is a factor of input batch, co1, ci1, howo_1
  singlecoreStatus.batch1 = params.batch;
  singlecoreStatus.m1 = params.co1;
  singlecoreStatus.n1 = params.ci1 * params.kh * params.kw;
  singlecoreStatus.k1 = (params.ho * params.wo + kBlockSize - 1) / kBlockSize;
  OP_LOGD(params.op_type.c_str(), "batch1:%lld, m1:%lld, k1:%lld, n1:%lld",
          singlecoreStatus.batch1, singlecoreStatus.m1, singlecoreStatus.k1, singlecoreStatus.n1);
  if (singlecoreStatus.batch1 * params.co1 * params.ci1 * singlecoreStatus.k1 <= params.max_core_num) {
    singlecoreStatus.batch_dim = singlecoreStatus.batch1;
    singlecoreStatus.n_dim = params.ci1;
    singlecoreStatus.m_dim = params.co1;
    singlecoreStatus.h_dim = singlecoreStatus.k1;
    singlecoreStatus.batch2 = 1;
    singlecoreStatus.m2 = 1;
    singlecoreStatus.k2 = 1;
    singlecoreStatus.n2 = params.kh * params.kw;
    singlecoreStatus.al1_k_full_load_size = singlecoreStatus.k2 * params.k0 * params.co0;
    singlecoreStatus.al1_full_load_size = singlecoreStatus.m2 * singlecoreStatus.al1_k_full_load_size;
    int32_t single_core_ho = GetK2Ho(singlecoreStatus.k2 * params.k0, params.wo);
    int32_t single_core_hi = GetHo2Hi(single_core_ho, params.stride_h, params.kh);
    singlecoreStatus.bl1_k_full_load_size = single_core_hi * params.wi * params.ci0;
    singlecoreStatus.bl1_full_load_size = params.ci1 * singlecoreStatus.bl1_k_full_load_size;
    OP_LOGD(params.op_type.c_str(), "multi-core factors batch_dim:%lld, n_dim:%lld, m_dim:%lld, h_dim:%lld",
            singlecoreStatus.batch_dim, singlecoreStatus.n_dim, singlecoreStatus.m_dim, singlecoreStatus.h_dim);
    return;
  }
  int32_t batch_factor_array[params.max_core_num] = {0};
  int32_t n_factor_array[params.max_core_num] = {0};
  int32_t m_factor_array[params.max_core_num] = {0};
  int32_t h_factor_array[params.max_core_num] = {0};
  GetFactors(blockDimCalculator.batch_dim_cnt, batch_factor_array, singlecoreStatus.batch1, params.max_core_num);
  GetFactors(blockDimCalculator.n_dim_cnt, n_factor_array, params.ci1, params.max_core_num);
  GetFactors(blockDimCalculator.m_dim_cnt, m_factor_array, params.co1, params.max_core_num);
  GetFactors(blockDimCalculator.h_dim_cnt, h_factor_array, singlecoreStatus.k1, params.max_core_num);
  blockDimCalculator.batch_dim_array = batch_factor_array;
  blockDimCalculator.m_dim_array = m_factor_array;
  blockDimCalculator.n_dim_array = n_factor_array;
  blockDimCalculator.h_dim_array = h_factor_array;
  blockDimCalculator.al1_min_load_size = params.k0 * params.co0;
  int32_t min_single_core_ho = GetK2Ho(params.k0, params.wo);
  int32_t min_single_core_hi = GetHo2Hi(min_single_core_ho, params.stride_h, params.kh);
  blockDimCalculator.bl1_min_load_size = min_single_core_hi * params.wi * params.ci0;
  blockDimCalculator.l1_min_pnt =
    kL1Fp16Size / (blockDimCalculator.al1_min_load_size + blockDimCalculator.bl1_min_load_size);
  for (int32_t b_idx = 0; b_idx < blockDimCalculator.batch_dim_cnt; b_idx++) {
    int32_t b_factor = blockDimCalculator.batch_dim_array[b_idx];
    for (int32_t n_idx = 0; n_idx < blockDimCalculator.n_dim_cnt; n_idx++) {
      int32_t n_factor = blockDimCalculator.n_dim_array[n_idx];
      if (b_factor * n_factor > params.max_core_num) {
          break;
      }
      for (int32_t m_idx = 0; m_idx < blockDimCalculator.m_dim_cnt; m_idx++) {
        GetKDimHelper(m_idx, b_factor, n_factor);
      }
    }
  }
  OP_LOGD(params.op_type.c_str(), "multi-core factors batch_dim:%lld, n_dim:%lld, m_dim:%lld, h_dim:%lld",
          singlecoreStatus.batch_dim, singlecoreStatus.n_dim, singlecoreStatus.m_dim, singlecoreStatus.h_dim);
}

void Conv2dDwCacheTiling::SetResFactors(L0Factors &res_factors)
{
  res_factors.final_ml0 = l0Status.final_ml0;
  res_factors.final_kl0 = l0Status.final_kl0;
  res_factors.final_nl0 = l0Status.final_nl0;
  res_factors.final_load_size = l0Status.final_load_size;
  res_factors.final_l0c_use = l0Status.final_l0c_use;
  res_factors.final_mul = l0Status.final_mul;
}

int32_t Conv2dDwCacheTiling::GetLoadSize()
{
  l0Calculator.al1_min_load_size = l0Status.m_l0 * l0Status.k_l0 * params.k0 * params.co0;
  l0Calculator.min_ho = GetK2Ho(l0Status.k_l0 * params.k0, params.wo);
  l0Calculator.min_hi = GetHo2Hi(l0Calculator.min_ho, params.stride_h, params.kh);
  int32_t ci1_factor = (l0Status.n_l0 + params.kh * params.kw - 1) / (params.kh * params.kw);
  l0Calculator.bl1_min_load_size = l0Calculator.min_hi * params.wi * ci1_factor * params.ci0;
  bool al1_full_load = singlecoreStatus.batch2 == 1 &&
    singlecoreStatus.al1_full_load_size + l0Calculator.bl1_min_load_size <= kL1Fp16Size;
  bool bl1_full_load = singlecoreStatus.batch2 == 1 &&
    singlecoreStatus.bl1_full_load_size + l0Calculator.al1_min_load_size <= kL1Fp16Size;
  bool al1_k_full_load = singlecoreStatus.batch2 == 1 &&
    singlecoreStatus.al1_k_full_load_size + l0Calculator.bl1_min_load_size <= kL1Fp16Size;
  bool bl1_k_full_load = singlecoreStatus.batch2 == 1 &&
    singlecoreStatus.bl1_k_full_load_size + l0Calculator.al1_min_load_size <= kL1Fp16Size;
  bool condition_full_load = al1_full_load || bl1_full_load;
  bool condition_k_full_load = al1_k_full_load && bl1_k_full_load;
  if (al1_k_full_load) {
    int32_t l1_rest_size = kL1Fp16Size - l0Calculator.bl1_min_load_size;
    int32_t m_l1 = min(l1_rest_size / singlecoreStatus.al1_k_full_load_size, singlecoreStatus.m2);
    if (m_l1 >= l0Status.m_l0) {
      while (singlecoreStatus.m2 % m_l1 != 0 || m_l1 % l0Status.m_l0 != 0) {
        m_l1--;
      }
    }
    int32_t m_outer = singlecoreStatus.m2 / m_l1;
    l0Calculator.tmp_load_size_al1k = m_outer * singlecoreStatus.n2 + singlecoreStatus.m2;
  }
  if (bl1_k_full_load) {
    int32_t l1_rest_size = kL1Fp16Size - l0Calculator.al1_min_load_size;
    int32_t n_l1 = min(l1_rest_size / singlecoreStatus.bl1_k_full_load_size, singlecoreStatus.n2);
    if (n_l1 >= l0Status.n_l0) {
      while (singlecoreStatus.n2 % n_l1 != 0 || n_l1 % l0Status.n_l0 != 0) {
        n_l1--;
      }
    }
    int32_t n_outer = singlecoreStatus.n2 / n_l1;
    l0Calculator.tmp_load_size_bl1k = singlecoreStatus.n2 + n_outer * singlecoreStatus.m2;
  }
  if (condition_full_load) {
    return singlecoreStatus.m2 + singlecoreStatus.n2;
  } else if (condition_k_full_load) {
    return min(l0Calculator.tmp_load_size_al1k, l0Calculator.tmp_load_size_bl1k);
  } else if (al1_k_full_load) {
    return l0Calculator.tmp_load_size_al1k;
  } else if (bl1_k_full_load) {
    return l0Calculator.tmp_load_size_bl1k;
  } else {
    return singlecoreStatus.m2 * singlecoreStatus.n2 / l0Status.m_l0 +
      singlecoreStatus.m2 * singlecoreStatus.n2 / l0Status.n_l0;
  }
}

void Conv2dDwCacheTiling::GetFinalMkn()
{
  if (l0Status.k_l0 == 0) {
    return;
  }
  float tmp_l0c_use =
    l0Status.m_l0 * l0Status.n_l0 * l0Status.db_l0c * kBlockSize * kBlockSize * kFp32Bytes * 1.0 / kL0cSize;
  int32_t tmp_mul = l0Status.m_l0 * l0Status.k_l0 * l0Status.n_l0;
  int32_t tmp_loadsize = GetLoadSize();
  auto condition1 = l0Status.final_ml0 == 0;
  auto condition2 = tmp_loadsize < l0Status.final_load_size;
  auto condition3 = (tmp_loadsize == l0Status.final_load_size && tmp_mul > l0Status.final_mul &&
    tmp_mul * tmp_l0c_use >= l0Status.final_mul * l0Status.final_l0c_use);
  if (condition1 || condition2 || condition3) {
    l0Status.final_ml0 = l0Status.m_l0;
    l0Status.final_kl0 = l0Status.k_l0;
    l0Status.final_nl0 = l0Status.n_l0;
    l0Status.final_load_size = tmp_loadsize;
    l0Status.final_l0c_use = tmp_l0c_use;
    l0Status.final_mul = tmp_mul;
  }
}

void Conv2dDwCacheTiling::GetL0StatusFromParasCombo(int32_t *paras_combo)
{
  l0Status.SetInitLoadStatus();
  l0Status.min_l0n = params.kw;
  l0Status.db_l0a = paras_combo[kIdxZero];
  l0Status.db_l0b = paras_combo[kIdxOne];
  l0Status.db_l0c = paras_combo[kIdxTwo];
  // L1_min_loadsize <= L1Size
  l0Status.max_mk = min(paras_combo[kIdxThree],
                        (kL1Fp16Size - (params.stride_h + params.kh) * params.wi * kBlockSize)
                        / (kBlockSize * kBlockSize));
  l0Status.max_nk = paras_combo[kIdxFour];
  l0Status.max_mn = paras_combo[kIdxFive];
  // UB_min_loadsize <= UBSize
  int32_t cub_min_size = (params.cub_fused_num + 1) * kBlockSize * kBlockSize * ubStatus.db_cub * kFp32Bytes;
  int32_t aub_min_size = (params.aub_fused_num + 1) * kBlockSize * kBlockSize * ubStatus.db_aub * kFp16Bytes;
  int32_t bub_min_size = (params.bub_fused_num + 1) * kBlockSize * params.wi * ubStatus.db_bub * kFp16Bytes;
  int32_t max_ml0 = (kUbSize - aub_min_size - bub_min_size) / cub_min_size;
  l0Status.max_axis_num = min(min(min(paras_combo[kIdxSix], l0Status.max_mk), max_ml0),
                              l0Status.max_mn / l0Status.min_l0n);
  l0Status.max_axis_pnt = paras_combo[kIdxSeven];
  l0Status.max_axis_pnt = min(l0Status.max_axis_pnt, l0Status.max_axis_num);
}

void Conv2dDwCacheTiling::GetL0FactorsCand(L0Factors &res_factors, int32_t *paras_combo)
{
  GetL0StatusFromParasCombo(paras_combo);
  int32_t m_dim_factors[kL0FactorCandLen] = {0};
  int32_t ml0_other_factors[kGetFactorParamsLen] = {l0Status.max_axis_num, 1, 1, 0};
  GetTwoFactors(m_dim_factors, l0Status.max_axis_pnt, singlecoreStatus.m2, ml0_other_factors);
  for (auto &m_dim_factor: m_dim_factors) {
    if (m_dim_factor == 0) {
      continue;
    }
    int32_t max_ci1_factor = (kL1Fp16Size - m_dim_factor * kBlockSize * kBlockSize) /
                            ((params.stride_h + params.kh) * params.wi * kBlockSize);
    l0Status.max_l0n = max_ci1_factor * params.kh * params.kw;
    int32_t n_factor_max = min(min(l0Status.max_mn / m_dim_factor, l0Status.max_nk), l0Status.max_l0n);
    int32_t n_dim_factors[kL0FactorCandLen] = {0};
    int32_t nl0_other_factors[kGetFactorParamsLen] =
        {n_factor_max, l0Status.min_l0n, params.kh * params.kw, params.kw};
    GetTwoFactors(n_dim_factors, n_factor_max, singlecoreStatus.n2, nl0_other_factors);
    for (auto &n_dim_factor: n_dim_factors) {
      if (n_dim_factor == 0) {
        continue;
      }
      int32_t ci1_factor = (n_dim_factor + params.kh * params.kw - 1) / (params.kh * params.kw);
      int32_t k0_max_l1 = (kL1Fp16Size - (params.stride_h + params.kh) * params.wi * ci1_factor * kBlockSize) /
                          (m_dim_factor * kBlockSize * kBlockSize);
      int32_t k0_max = min(min(l0Status.max_mk / m_dim_factor, l0Status.max_nk / n_dim_factor), k0_max_l1);
      int32_t k0_factors[kL0FactorCandLen] = {0};
      int32_t kl0_other_factors[kGetFactorParamsLen] = {k0_max, 1, 1, 0};
      GetTwoFactors(k0_factors, k0_max, singlecoreStatus.k2, kl0_other_factors);
      for (auto &k0: k0_factors) {
        l0Status.m_l0 = m_dim_factor;
        l0Status.n_l0 = n_dim_factor;
        l0Status.k_l0 = k0;
        GetFinalMkn();
      }
    }
  }
  SetResFactors(res_factors);
}

void Conv2dDwCacheTiling::GetParasCombo(MKNParasCombo *paras_combo_map)
{
  MKNParasCombo combo_zero = {2, 2, 2, 64, 64, 128, 64, 11};
  MKNParasCombo combo_one = {2, 2, 1, 64, 64, 256, 64, 16};
  paras_combo_map[0] = combo_zero;
  paras_combo_map[1] = combo_one;
}

bool Conv2dDwCacheTiling::GetL0Factors()
{
  // get m_l0, n_l0, k_l0 factor when single core m2, n2, k2 is know
  // m_l0, n_l0, k_l0 is a factor of single core m2, n2, k2
  int32_t db_l0c_on_idx = 0;
  int32_t db_l0c_off_idx = 1;
  L0Factors res_factors[kL0ParasComboLen];
  MKNParasCombo paras_combo_map[kL0ParasComboLen];
  GetParasCombo(paras_combo_map);
  for (size_t i = 0; i < kL0ParasComboLen; ++i) {
    MKNParasCombo mkn_paras_combo = paras_combo_map[i];
    GetL0FactorsCand(res_factors[i], mkn_paras_combo.paras_combo);
  }
  // check both L0C utilization and loadsize to control L0C DB
  int32_t m0_on = res_factors[db_l0c_on_idx].final_ml0;
  int32_t k0_on = res_factors[db_l0c_on_idx].final_kl0;
  int32_t n0_on = res_factors[db_l0c_on_idx].final_nl0;
  int32_t loadsize_on = res_factors[db_l0c_on_idx].final_load_size;
  float l0c_use_on = res_factors[db_l0c_on_idx].final_l0c_use;

  int32_t m0_off = res_factors[db_l0c_off_idx].final_ml0;
  int32_t k0_off = res_factors[db_l0c_off_idx].final_kl0;
  int32_t n0_off = res_factors[db_l0c_off_idx].final_nl0;
  int32_t loadsize_off = res_factors[db_l0c_off_idx].final_load_size;
  float l0c_use_off = res_factors[db_l0c_off_idx].final_l0c_use;

  if ((l0c_use_off > l0c_use_on) || (loadsize_off < loadsize_on)) {
    int64_t db_l0a_off = kDbOn;
    int64_t db_l0b_off = kDbOn;
    l0Status.db_l0c = kDbOff;
    l0Status.db_l0a = db_l0a_off;
    l0Status.db_l0b = db_l0b_off;
    l0Status.m_l0 = m0_off;
    l0Status.k_l0 = k0_off;
    l0Status.n_l0 = n0_off;
  } else {
    int64_t db_l0a_on = kDbOn;
    int64_t db_l0b_on = kDbOn;
    l0Status.db_l0c = kDbOn;
    l0Status.db_l0a = db_l0a_on;
    l0Status.db_l0b = db_l0b_on;
    l0Status.m_l0 = m0_on;
    l0Status.k_l0 = k0_on;
    l0Status.n_l0 = n0_on;
  }
  OP_LOGE_IF(l0Status.m_l0 == 0, false, params.op_type, "l0Status.m_l0 is 0.");
  OP_LOGE_IF(l0Status.n_l0 == 0, false, params.op_type, "l0Status.n_l0 is 0.");
  OP_LOGE_IF(l0Status.k_l0 == 0, false, params.op_type, "l0Status.k_l0 is 0.");
  OP_LOGD(params.op_type.c_str(),
          "tiling m_l0:%lld, n_l0:%lld, k_l0:%lld", l0Status.m_l0, l0Status.n_l0, l0Status.k_l0);
  OP_LOGD(params.op_type.c_str(), "tiling db_l0a:%lld, db_l0b:%lld, db_l0c:%lld", l0Status.db_l0a, l0Status.db_l0b,
          l0Status.db_l0c);
  return true;
}

int32_t Conv2dDwCacheTiling::GetAL1Size()
{
  return l1Status.m_al1 * l0Status.m_l0 * params.co0 * l1Status.kal1_16 * kBlockSize * l1Status.db_al1;
}

int32_t Conv2dDwCacheTiling::GetBL1Size()
{
  int32_t cur_ho = GetK2Ho(l1Status.kbl1_16 * kBlockSize, params.wo);
  int32_t cur_hi = GetHo2Hi(cur_ho, params.stride_h, params.kh);
  int32_t cur_ci1_factor = (l1Status.n_bl1 * l0Status.n_l0 + params.kh * params.kw - 1) / (params.kh * params.kw);
  return cur_hi * params.wi * cur_ci1_factor * params.ci0 * l1Status.db_bl1;
}

int32_t Conv2dDwCacheTiling::GetL1Size()
{
  l1Status.al1_size = GetAL1Size();
  l1Status.bl1_size = GetBL1Size();
  return l1Status.al1_size + l1Status.bl1_size;
}

void Conv2dDwCacheTiling::L1StatusBothFullLoad(int32_t res[][kL1ParasComboLen])
{
  l1Status.cur_l1_size = GetL1Size();
  if (singlecoreStatus.batch2 == 1 && l1Status.cur_l1_size <= kL1Fp16Size) {
    l1Status.both_full_load = true;
    l1Status.load_size = singlecoreStatus.m2 + singlecoreStatus.n2;
    res[kIdxZero][kIdxZero] = l1Status.kal1_16;
    res[kIdxZero][kIdxOne] = l1Status.m_al1;
    res[kIdxZero][kIdxTwo] = l1Status.db_al1;
    res[kIdxZero][kIdxThree] = l1Status.kbl1_16;
    res[kIdxZero][kIdxFour] = l1Status.n_bl1;
    res[kIdxZero][kIdxFive] = l1Status.db_bl1;
    res[kIdxZero][kIdxSix] = l1Status.load_size;
    res[kIdxZero][kIdxSeven] = 1;
    res[kIdxZero][kIdxEight] = 1;
  }
}

void Conv2dDwCacheTiling::GetkBL1Factor()
{
  l1Status.bl1_times = l1Status.all_times;
  l1Status.kbl1_16 = l1Status.bl1_times * l0Status.k_l0;
  l1Status.ho = GetK2Ho(l1Status.kbl1_16 * kBlockSize, params.wo);
  l1Status.hi = GetHo2Hi(l1Status.ho, params.stride_h, params.kh);
  int32_t tmp_ci1_factor = (l1Status.n_bl1 * l0Status.n_l0 + params.kh * params.kw - 1) / (params.kh * params.kw);
  int32_t tmp_bl1_size = l1Status.hi * params.wi * tmp_ci1_factor * params.ci0 * l1Status.db_bl1;
  while (tmp_bl1_size > l1Status.bl1_size) {
    l1Status.bl1_times -= 1;
    GetNearestFactor(l1Status.all_times, l1Status.bl1_times);
    l1Status.kbl1_16 = l1Status.bl1_times * l0Status.k_l0;
    l1Status.ho = GetK2Ho(l1Status.kbl1_16 * kBlockSize, params.wo);
    l1Status.hi = GetHo2Hi(l1Status.ho, params.stride_h, params.kh);
    tmp_bl1_size = l1Status.hi * params.wi * tmp_ci1_factor * params.ci0 * l1Status.db_bl1;
  }
}

void Conv2dDwCacheTiling::GetnBL1Factor()
{
  if (singlecoreStatus.batch2 == 1 && l1Status.kbl1_16 == singlecoreStatus.k2) {
    int32_t tmp_ci1_factor = (l0Status.n_l0 + params.kh * params.kw - 1) / (params.kh * params.kw);
    l1Status.n_bl1 = min(l1Status.bl1_size /
                         (l1Status.hi * params.wi * tmp_ci1_factor * params.ci0 * l1Status.db_bl1),
                         l1Status.max_n_bl1);
    GetNearestFactor(l1Status.max_n_bl1, l1Status.n_bl1);
  }
}

void Conv2dDwCacheTiling::L1StatusAl1FullLoad(int32_t res[][kL1ParasComboLen])
{
  l1Status.cur_l1_size = GetL1Size();
  if (singlecoreStatus.batch2 == 1 && l1Status.cur_l1_size <= kL1Fp16Size) {
    l1Status.al1_full_load = true;
    l1Status.db_bl1 = kDbOn;
    if (GetL1Size() > kL1Fp16Size) {
      l1Status.db_bl1 = kDbOff;
    }
    l1Status.bl1_size = kL1Fp16Size - l1Status.al1_size;
    GetkBL1Factor();
    GetnBL1Factor();
    l1Status.load_size = singlecoreStatus.m2 + singlecoreStatus.n2;
    res[kIdxOne][kIdxZero] = l1Status.kal1_16;
    res[kIdxOne][kIdxOne] = l1Status.m_al1;
    res[kIdxOne][kIdxTwo] = l1Status.db_al1;
    res[kIdxOne][kIdxThree] = l1Status.kbl1_16;
    res[kIdxOne][kIdxFour] = l1Status.n_bl1;
    res[kIdxOne][kIdxFive] = l1Status.db_bl1;
    res[kIdxOne][kIdxSix] = l1Status.load_size;
    res[kIdxOne][kIdxSeven] = 1;
    res[kIdxOne][kIdxEight] = 1;
  }
}

void Conv2dDwCacheTiling::GetkAL1Factor()
{
  l1Status.kal1_16 =
    min(l1Status.al1_size / (l1Status.m_al1 * l0Status.m_l0 * params.co0 * kBlockSize * l1Status.db_al1),
        singlecoreStatus.k2);
  l1Status.al1_times = l1Status.kal1_16 / l0Status.k_l0;
  GetNearestFactor(l1Status.all_times, l1Status.al1_times);
  l1Status.kal1_16 = l1Status.al1_times * l0Status.k_l0;
}

void Conv2dDwCacheTiling::GetmAL1Factor()
{
  if (singlecoreStatus.batch2 == 1 && l1Status.kal1_16 == singlecoreStatus.k2) {
    l1Status.m_al1 =
      min(l1Status.al1_size / (l0Status.m_l0 * params.co0 * l1Status.kal1_16 * kBlockSize * l1Status.db_al1),
          l1Status.max_m_al1);
    GetNearestFactor(l1Status.max_m_al1, l1Status.m_al1);
  }
}

void Conv2dDwCacheTiling::L1StatusBl1FullLoad(int32_t res[][kL1ParasComboLen])
{
  l1Status.cur_l1_size = GetL1Size();
  if (singlecoreStatus.batch2 == 1 && l1Status.cur_l1_size <= kL1Fp16Size) {
    l1Status.bl1_full_load = true;
    l1Status.db_al1 = kDbOn;
    if (GetL1Size() > kL1Fp16Size) {
      l1Status.db_al1 = kDbOff;
    }
    l1Status.al1_size = kL1Fp16Size - l1Status.bl1_size;
    GetkAL1Factor();
    GetmAL1Factor();
    l1Status.load_size = singlecoreStatus.n2 + singlecoreStatus.m2;
    res[kIdxTwo][kIdxZero] = l1Status.kal1_16;
    res[kIdxTwo][kIdxOne] = l1Status.m_al1;
    res[kIdxTwo][kIdxTwo] = l1Status.db_al1;
    res[kIdxTwo][kIdxThree] = l1Status.kbl1_16;
    res[kIdxTwo][kIdxFour] = l1Status.n_bl1;
    res[kIdxTwo][kIdxFive] = l1Status.db_bl1;
    res[kIdxTwo][kIdxSix] = l1Status.load_size;
    res[kIdxTwo][kIdxSeven] = 1;
    res[kIdxTwo][kIdxEight] = 1;
  }
}

void Conv2dDwCacheTiling::GetL1LoadSize()
{
  bool al1_k_full_load = singlecoreStatus.batch2 == 1 && l1Status.kal1_16 == singlecoreStatus.k2;
  bool al1_full_load = al1_k_full_load && l1Status.m_al1 == l1Status.max_m_al1;
  bool bl1_k_full_load = singlecoreStatus.batch2 == 1 && l1Status.kbl1_16 == singlecoreStatus.k2;
  bool bl1_full_load = bl1_k_full_load && l1Status.n_bl1 == l1Status.max_n_bl1;
  if (al1_full_load || bl1_full_load) {
    l1Status.al1_repeat = 1;
    l1Status.bl1_repeat = 1;
  } else if (al1_k_full_load && bl1_k_full_load) {
    l1Status.al1_repeat = l1Status.max_n_bl1 / l1Status.n_bl1;
    l1Status.bl1_repeat = l1Status.max_m_al1 / l1Status.m_al1;
    if (singlecoreStatus.m2 + l1Status.bl1_repeat * singlecoreStatus.n2 <
        l1Status.al1_repeat * singlecoreStatus.m2 + singlecoreStatus.n2) {
      l1Status.al1_repeat = 1;
    } else {
      l1Status.bl1_repeat = 1;
    }
  } else if (al1_k_full_load) {
    l1Status.al1_repeat = 1;
    l1Status.bl1_repeat = l1Status.max_m_al1 / l1Status.m_al1;
  } else if (bl1_k_full_load) {
    l1Status.al1_repeat = l1Status.max_n_bl1 / l1Status.n_bl1;
    l1Status.bl1_repeat = 1;
  } else {
    l1Status.al1_repeat = l1Status.max_n_bl1 / l1Status.n_bl1;
    l1Status.bl1_repeat = l1Status.max_m_al1 / l1Status.m_al1;
  }
  l1Status.load_size = l1Status.al1_repeat * singlecoreStatus.m2 + l1Status.bl1_repeat * singlecoreStatus.n2;
}

void Conv2dDwCacheTiling::L1StatusNeitherFullLoad(int32_t res[][kL1ParasComboLen])
{
  l1Status.cur_l1_size = GetL1Size();
  l1Status.bl1_size = kL1Fp16Size - l1Status.al1_size;
  GetkBL1Factor();
  l1Status.cur_l1_size = GetL1Size();
  l1Status.al1_size = kL1Fp16Size - l1Status.bl1_size;
  GetkAL1Factor();
  l1Status.db_bl1 = kDbOn;
  if (GetL1Size() > kL1Fp16Size) {
    l1Status.db_bl1 = kDbOff;
  }
  l1Status.db_al1 = kDbOn;
  if (GetL1Size() > kL1Fp16Size) {
    l1Status.db_al1 = kDbOff;
  }
  l1Status.cur_l1_size = GetL1Size();
  l1Status.bl1_size = kL1Fp16Size - l1Status.al1_size;
  GetnBL1Factor();
  l1Status.cur_l1_size = GetL1Size();
  l1Status.al1_size = kL1Fp16Size - l1Status.bl1_size;
  GetmAL1Factor();
  // k_al1 and k_bl1 must be a factor of each other
  if (l1Status.kal1_16 > l1Status.kbl1_16 && l1Status.kal1_16 % l1Status.kbl1_16 != 0) {
    while (l1Status.kal1_16 % l1Status.kbl1_16 != 0 || singlecoreStatus.k2 % l1Status.kal1_16 != 0) {
      l1Status.kal1_16 -= 1;
    }
  } else if (l1Status.kal1_16 < l1Status.kbl1_16 && l1Status.kbl1_16 % l1Status.kal1_16 != 0) {
    while (l1Status.kbl1_16 % l1Status.kal1_16 != 0 || singlecoreStatus.k2 % l1Status.kbl1_16 != 0) {
      l1Status.kbl1_16 -= 1;
    }
  }
  GetL1LoadSize();
  res[kIdxThree][kIdxZero] = l1Status.kal1_16;
  res[kIdxThree][kIdxOne] = l1Status.m_al1;
  res[kIdxThree][kIdxTwo] = l1Status.db_al1;
  res[kIdxThree][kIdxThree] = l1Status.kbl1_16;
  res[kIdxThree][kIdxFour] = l1Status.n_bl1;
  res[kIdxThree][kIdxFive] = l1Status.db_bl1;
  res[kIdxThree][kIdxSix] = l1Status.load_size;
  res[kIdxThree][kIdxSeven] = l1Status.bl1_repeat;
  res[kIdxThree][kIdxEight] = l1Status.al1_repeat;
}

void Conv2dDwCacheTiling::GetL1Factors()
{
  // get m_al1, n_bl1, kal1_16, kbl1_16 factors when L0, singlecore factor is know
  // get al1, bl1 double buffer factors
  int32_t res[kL1CandLen][kL1ParasComboLen] = {0};
  l1Status.all_times = singlecoreStatus.k2 / l0Status.k_l0;
  l1Status.max_m_al1 = (singlecoreStatus.m2 + l0Status.m_l0 - 1) / l0Status.m_l0;
  l1Status.max_n_bl1 = (singlecoreStatus.n2 + l0Status.n_l0 - 1) / l0Status.n_l0;
  // both AL1 and Bl1 full load
  int32_t both_full_load_factors[kL1FactorsLen] =
      {singlecoreStatus.k2, singlecoreStatus.k2, l1Status.max_m_al1, l1Status.max_n_bl1, kDbOff, kDbOff};
  l1Status.SetStatus(both_full_load_factors);
  L1StatusBothFullLoad(res);
  // only AL1 full load
  int32_t al1_full_load_factors[kL1FactorsLen] =
      {singlecoreStatus.k2, l0Status.k_l0, l1Status.max_m_al1, 1, kDbOff, kDbOff};
  l1Status.SetStatus(al1_full_load_factors);
  L1StatusAl1FullLoad(res);
  // only BL1 full load
  int32_t bl1_full_load_factors[kL1FactorsLen] =
      {l0Status.k_l0, singlecoreStatus.k2, 1, l1Status.max_n_bl1, kDbOff, kDbOff};
  l1Status.SetStatus(bl1_full_load_factors);
  L1StatusBl1FullLoad(res);
  // neither AL1 nor Bl1 full load
  int32_t neither_full_load_factors[kL1FactorsLen] = {l0Status.k_l0, l0Status.k_l0, 1, 1, kDbOff, kDbOff};
  l1Status.SetStatus(neither_full_load_factors);
  L1StatusNeitherFullLoad(res);
  // choose the final factors
  int32_t *tmp_factors = res[kIdxThree];
  int32_t tmp_loadsize = tmp_factors[kIdxSix];
  int32_t tmp_mn = tmp_factors[kIdxOne] + tmp_factors[kIdxFour];
  if (l1Status.bl1_full_load && (res[kIdxTwo][kIdxSix] < tmp_loadsize ||
      (res[kIdxTwo][kIdxSix] == tmp_loadsize && res[kIdxTwo][kIdxOne] + res[kIdxTwo][kIdxFour] > tmp_mn))) {
    tmp_factors = res[kIdxTwo];
    tmp_loadsize = tmp_factors[kIdxSix];
    tmp_mn = tmp_factors[kIdxOne] + tmp_factors[kIdxFour];
  }
  if (l1Status.al1_full_load && (res[kIdxOne][kIdxSix] < tmp_loadsize ||
      (res[kIdxOne][kIdxSix] == tmp_loadsize && res[kIdxOne][kIdxOne] + res[kIdxOne][kIdxFour] > tmp_mn))) {
    tmp_factors = res[kIdxOne];
    tmp_loadsize = tmp_factors[kIdxSix];
    tmp_mn = tmp_factors[kIdxOne] + tmp_factors[kIdxFour];
  }
  if (l1Status.both_full_load && (res[kIdxZero][kIdxSix] < tmp_loadsize ||
      (res[kIdxZero][kIdxSix] == tmp_loadsize && res[kIdxZero][kIdxOne] + res[kIdxZero][kIdxFour] > tmp_mn))) {
    tmp_factors = res[kIdxZero];
  }
  int32_t res_l1_factors[kL1FactorsLen] = {tmp_factors[kIdxZero], tmp_factors[kIdxThree], tmp_factors[kIdxOne],
                                           tmp_factors[kIdxFour], tmp_factors[kIdxTwo], tmp_factors[kIdxFive]};
  l1Status.SetStatus(res_l1_factors);
  l1Status.bl1_repeat = tmp_factors[kIdxSeven];
  l1Status.al1_repeat = tmp_factors[kIdxEight];
  l1Status.cur_l1_size = GetL1Size();
  l1Status.ho = GetK2Ho(l1Status.kbl1_16 * kBlockSize, params.wo);
  l1Status.hi = GetHo2Hi(l1Status.ho, params.stride_h, params.kh);
  int32_t bl1_ci = (l0Status.n_l0 * l1Status.n_bl1 + params.kh * params.kw - 1) / (params.kh * params.kw);
  l1Status.bl1_bound = l1Status.hi * params.wi * bl1_ci * params.ci0;
  OP_LOGD(params.op_type.c_str(), "dw_tiling kal1_16:%lld, kbl1_16:%lld, k_l0:%lld", l1Status.kal1_16, l1Status.kbl1_16,
          l0Status.k_l0);
  OP_LOGD(params.op_type.c_str(), "dw_tiling m_al1:%lld, n_bl1:%lld", l1Status.m_al1, l1Status.n_bl1);
  OP_LOGD(params.op_type.c_str(), "dw_tiling db_al1:%lld, db_bl1:%lld", l1Status.db_al1, l1Status.db_bl1);
}

void Conv2dDwCacheTiling::GetUbFactors()
{
  ubStatus.db_aub = kDbOn;
  ubStatus.db_bub = kDbOn;
  ubStatus.db_cub = kDbOn;
  // aub, bub, cub min size
  int32_t aub_min_size = (params.aub_fused_num + 1) * kBlockSize * kBlockSize * ubStatus.db_aub;
  int32_t bub_min_size = (params.bub_fused_num + 1) * kBlockSize * Align(params.wi, kBlockSize) * ubStatus.db_bub;
  int32_t cub_min_size = (params.cub_fused_num + 1) * l0Status.m_l0 * kBlockSize * kBlockSize * ubStatus.db_cub;
  // aub add bub max size
  int32_t ub_rest_size = (kUbSize - cub_min_size * kFp32Bytes) / kFp16Bytes;
  // aub, bub total size
  int32_t al1_load_size = l1Status.al1_size * l1Status.al1_repeat * l1Status.db_al1;
  int32_t bl1_load_size = l1Status.bl1_size * l1Status.bl1_repeat * l1Status.db_bl1;
  int32_t aub_move_size = (params.aub_fused_num + 1) * al1_load_size * ubStatus.db_aub;
  int32_t bub_move_size = (params.bub_fused_num + 1) * bl1_load_size * ubStatus.db_bub;
  // maybe used_size > real_size
  int32_t used_aub_size = min(min((ub_rest_size * aub_move_size) / (aub_move_size + bub_move_size), l1Status.al1_size),
                              (ub_rest_size - bub_min_size));
  // AUB_size = mAUB * kAUB * 16 * 16
  // BUB_size = nBUB * 16 * align(kBUB * wi, 16)
  int32_t limit_mk = max(used_aub_size / aub_min_size, 1);
  ubStatus.m_aub = limit_mk;
  GetNearestFactor(l1Status.m_al1 * l0Status.m_l0, ubStatus.m_aub);
  ubStatus.k_aub = limit_mk / ubStatus.m_aub;
  GetNearestFactor(l1Status.kal1_16, ubStatus.k_aub);
  int32_t aub_size = ubStatus.m_aub * ubStatus.k_aub * aub_min_size;
  // align(kBUB * wi, 16) * nBUB < limit_hn
  int32_t limit_hn = max(static_cast<int32_t>((ub_rest_size - aub_size) /
                                              ((params.bub_fused_num + 1) * kBlockSize * ubStatus.db_bub)),
                         Align(params.wi, kBlockSize));
  ubStatus.k_bub = limit_hn;
  int32_t bl1_ho = GetK2Ho(l1Status.kbl1_16 * kBlockSize, params.wo);
  int32_t bl1_hi = GetHo2Hi(bl1_ho, params.stride_h, params.kh);
  GetNearestFactor(bl1_hi, ubStatus.k_bub);
  while (ubStatus.k_bub > 1 && Align(ubStatus.k_bub * params.wi, kBlockSize) > limit_hn) {
    --ubStatus.k_bub;
    GetNearestFactor(bl1_hi, ubStatus.k_bub);
  }
  ubStatus.n_bub = limit_hn / Align(ubStatus.k_bub * params.wi, kBlockSize);
  int32_t ci1_factor = (l1Status.n_bl1 * l0Status.n_l0 + params.kh * params.kw - 1) / (params.kh * params.kw);
  GetNearestFactor(ci1_factor, ubStatus.n_bub);
  int32_t bub_size = (params.bub_fused_num + 1) * ubStatus.n_bub * kBlockSize *
                     Align(ubStatus.k_bub * params.wi, kBlockSize) * ubStatus.db_bub;
  ubStatus.n_cub = (kUbSize - (aub_size + bub_size) * kFp16Bytes) / kFp32Bytes / cub_min_size;
  GetNearestFactor(l0Status.n_l0, ubStatus.n_cub);
  OP_LOGD(params.op_type.c_str(), "dw ub tiling -- m_aub:%lld, k_aub:%lld, n_bub:%lld, k_bub:%lld, n_cub:%lld",
          ubStatus.m_aub, ubStatus.k_aub, ubStatus.n_bub, ubStatus.k_bub, ubStatus.n_cub);
}

void Conv2dDwCacheTiling::CheckSpecialTemplate()
{
  if (singlecoreStatus.batch2 == 1 && singlecoreStatus.m2 / (l1Status.m_al1 * l0Status.m_l0) == 1 &&
      l1Status.kal1_16 == singlecoreStatus.k2) {
    l1Status.m_al1 = kNONE;
    OP_LOGD(params.op_type.c_str(), "check special template, tiling al1 changed to full load");
  }
  if (singlecoreStatus.batch2 == 1 && l1Status.n_bl1 * l0Status.n_l0 == singlecoreStatus.n2 &&
      l1Status.kbl1_16 == singlecoreStatus.k2) {
    l1Status.n_bl1 = kNONE;
    OP_LOGD(params.op_type.c_str(), "check special template, tiling bl1 changed to full load");
  }
}

void Conv2dDwCacheTiling::SetParams(Conv2dDwTiling &tiling)
{
  tiling.batch_dim = singlecoreStatus.batch_dim;
  tiling.n_dim = singlecoreStatus.n_dim;
  tiling.m_dim = singlecoreStatus.m_dim;
  tiling.h_dim = singlecoreStatus.h_dim;
  tiling.m_l0 = l0Status.m_l0;
  tiling.k_l0 = l0Status.k_l0;
  tiling.n_l0 = l0Status.n_l0;
  tiling.kal1_16 = l1Status.kal1_16;
  tiling.kbl1_16 = l1Status.kbl1_16;
  tiling.m_al1 = l1Status.m_al1;
  tiling.n_bl1 = l1Status.n_bl1;
  tiling.db_al1 = l1Status.db_al1;
  tiling.db_bl1 = l1Status.db_bl1;
  tiling.ho_bl1 = l1Status.ho;
  tiling.bl1_bound = l1Status.bl1_bound;
  tiling.n_cub = ubStatus.n_cub;
  tiling.db_cub = ubStatus.db_cub;
  tiling.k_org_dim = singlecoreStatus.k2 * kBlockSize;
  tiling.db_l0c = l0Status.db_l0c;
  tiling.k_aub = ubStatus.k_aub;
  tiling.m_aub = ubStatus.m_aub;
  tiling.db_aub = ubStatus.db_aub;
  tiling.k_bub = ubStatus.k_bub;
  tiling.n_bub = ubStatus.n_bub;
  tiling.db_bub = ubStatus.db_bub;
  if (tiling.m_al1 == kNONE) {
    tiling.al1_full_load = true;
    tiling.db_al1 = 1;
  }
  if (tiling.n_bl1 == kNONE) {
    tiling.bl1_full_load = true;
    tiling.db_bl1 = 1;
  }
  tiling.min_kl1_cmp_kl0 = (min(tiling.kal1_16, tiling.kbl1_16) == tiling.k_l0) ? 0 : 1;
}

void Conv2dDwCacheTiling::SetAttachFlag(Conv2dDwTiling &tiling)
{
  // find kernel ID
  bool k_al1_full_load = singlecoreStatus.batch2 == 1 && tiling.kal1_16 * kBlockSize == tiling.k_org_dim;
  bool k_bl1_full_load = singlecoreStatus.batch2 == 1 && tiling.kbl1_16 * kBlockSize == tiling.k_org_dim;
  bool condition1 = tiling.m_al1 == kNONE;
  bool condition2 = tiling.m_al1 != kNONE && k_al1_full_load;
  bool condition3 = tiling.m_al1 != kNONE && !k_al1_full_load;
  bool condition4 = tiling.n_bl1 == kNONE;
  bool condition5 = tiling.n_bl1 != kNONE && k_bl1_full_load;
  bool condition6 = tiling.n_bl1 != kNONE && !k_bl1_full_load;

  if (condition1) {
    tiling.al1_attach_flag = kAttachFlagZero;
  }
  if (condition2) {
    tiling.al1_attach_flag = kAttachFlagOne;
  }
  if (condition3) {
    tiling.al1_attach_flag = kAttachFlagTwo;
  }
  if (condition4) {
    tiling.bl1_attach_flag = kAttachFlagZero;
  }
  if (condition5) {
    tiling.bl1_attach_flag = kAttachFlagOne;
  }
  if (condition6) {
    tiling.bl1_attach_flag = kAttachFlagTwo;
  }

  if (tiling.kal1_16 == tiling.kbl1_16) {
    tiling.abkl1_attach_flag = kAttachFlagZero;
  } else if (tiling.kal1_16 > tiling.kbl1_16) {
    tiling.abkl1_attach_flag = kAttachFlagOne;
  } else if (tiling.kal1_16 < tiling.kbl1_16) {
    tiling.abkl1_attach_flag = kAttachFlagTwo;
  }
}

void Conv2dDwCacheTiling::FixTilingParam(Conv2dDwTiling &tiling) {
  // l0a_m = l0c_m * m_l0
  if (tiling.m_al1 == kNONE) {
    tiling.m_al1 = (params.co1 / tiling.m_dim / tiling.m_l0);
  }

  if (tiling.n_bl1 == kNONE) {
    tiling.n_bl1 = (params.ci1 / tiling.n_dim) * params.kh * params.kw / tiling.n_l0;
  }
}

void Conv2dDwCacheTiling::GetTilingId(Conv2dDwTiling &tiling)
{
  int32_t tiling_id_long = 0;
  tiling_id_long = tiling_id_long * kDecimal + tiling.db_al1;
  tiling_id_long = tiling_id_long * kDecimal + tiling.db_bl1;
  tiling_id_long = tiling_id_long * kDecimal + tiling.db_l0c;
  tiling_id_long = tiling_id_long * kDecimal + tiling.abkl1_attach_flag;
  tiling_id_long = tiling_id_long * kDecimal + tiling.al1_attach_flag;
  tiling_id_long = tiling_id_long * kDecimal + tiling.bl1_attach_flag;
  tiling_id_long = tiling_id_long * kDecimal + tiling.min_kl1_cmp_kl0;
  tiling_id_long = tiling_id_long * kDecimal + tiling.aub_multi_flag;
  tiling_id_long = tiling_id_long * kDecimal + tiling.bub_multi_flag;
  tiling_id_long = tiling_id_long * kDecimal + tiling.reorder_l1_mn;
  tiling_id_long = tiling_id_long * kDecimal + tiling.reorder_l0_mn;
  tiling.tiling_id = to_string(tiling_id_long);
}

bool Conv2dDwCacheTiling::GenTiling(Conv2dDwTiling &tiling)
{
  OP_LOGD(params.op_type.c_str(),
          "dw_cache_tiling input batch:%lld, ho:%lld, wo:%lld, co1:%lld, hi:%lld, wi:%lld, ci1:%lld, kh:%lld, kw:%lld",
          params.batch, params.ho, params.wo, params.co1, params.hi, params.wi, params.ci1, params.kh, params.kw);
  GetBlockDim();
  OP_LOGE_IF(!GetL0Factors(), false, params.op_type, "get L0 factors failed.");
  GetL1Factors();
  GetUbFactors();
  CheckSpecialTemplate();
  SetParams(tiling);
  SetAttachFlag(tiling);
  GetTilingId(tiling);
  FixTilingParam(tiling);
  OP_LOGD(params.op_type.c_str(), "the tiling id from cache tiling is: %s", tiling.tiling_id.c_str());
  return true;
}
} // namespace optiling::conv2d_dw