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
 * \file cache_tiling.cc
 * \brief function of cacheTiling
 */

#include "cache_tiling.h"
using namespace std;

namespace optiling {
static const int32_t kL1Size = (1024 * 1024);
static const int32_t kL0cSize = (256 * 1024);
static const int32_t kUbSize = (256 * 1024);
static const int32_t kAttachLabelLength = 9;
static const int32_t kBlockSize = 16;
static const int32_t kDecimal = 10;
static const int32_t kMinFractalSize = kBlockSize * kBlockSize;
static const int32_t kDbOn = 2;
static const int32_t kDbOff = 1;
static const int32_t kIdxZero = 0;
static const int32_t kIdxOne = 1;
static const int32_t kIdxTwo = 2;
static const int32_t kIdxThree = 3;
static const int32_t kIdxFour = 4;
static const int32_t kIdxFive = 5;
static const int32_t kIdxSix = 6;
static const int32_t kIdxSeven = 7;
static const int32_t kIdxEight = 8;
static const int32_t kAttachFlagZero = 0;
static const int32_t kAttachFlagOne = 1;
static const int32_t kAttachFlagTwo = 2;
static const int32_t kKbytes = 1024;
static const int32_t kMaxFactor = 128;
static const int32_t kFp16Bytes = 2;
static const int32_t kFp32Bytes = 4;
static const int32_t kMinMte1Load = 32;
static const bool kL0DbFlag = false;
static const int32_t kL0ParasComboLen = kL0DbFlag ? 8 : 2;
static const int32_t kLoadSizeRangeLow = 1000;
static const int32_t kLoadSizeRangeHigh = 4000;
static const int32_t kLoadSizeDiffRange = 400;
static const int32_t kMLowRange = 5;
static const int32_t kMHighRange = 6;
static const double kBlockingPctGate = 0.5;
static const double kLoadSizeGate = 0.13;
static const int32_t kCoreUseLowRange = 5;
static const int32_t kCoreUseHighRange = 9;
static const int32_t kUbFp16Size = kUbSize / kFp16Bytes;
static const int32_t kNumZero = 0;
static const int32_t kNumOne = 1;
static const int32_t kNumTwo = 2;
static const int32_t kNumThree = 3;
static const int32_t kNumFour = 4;
static const int32_t kBankConflictFactor = 4;
static const int32_t kL1FactorsLen = 6;
static const int32_t kCandidateLen = 2;
static const int32_t kMinSplitKCoreNum = 8;
static const double kWorstBandWidthUtilMulti = 8;
static const int32_t kHbmBandwidth8Core = 250;
static const int32_t kHbmBandwidth32Core = 1100;
static const int32_t kL2Bandwidth8Core = 1300;
static const int32_t kL2Bandwidth32Core = 3300;
static const int32_t kCoreNum32 = 32;
static const int32_t kMNPntMax = 16;
static const int32_t kSeedMapMin = 16;
static const int32_t kSeedMapMax = 1024;
static const int32_t kL0FactorNumLimit = 2;
static const int32_t kL1FactorNumLimit = 4;
static const int32_t kL0FactorLimit = 64;
static const int32_t kL1FactorLimit = 128;

struct AUbStatusCondition {
  /* This struct is used to storage the tiling condition of AUb.
  condition_m2_k2 : all data in m_dim and k_dim are loaded to Ub
  condition_ml1_kl1 : all data of m_dim and k_dim in L1 buffer are loaded to Ub
  condition_ml1_kl0: all data of m_dim in L1 buffer and the equivalent data of kl0 in L1 buffer are loaded to Ub.
  condition_ml1_k0: all data of m_dim in L1 buffer and partial data of kl0 in L1 buffer are loaded to Ub.
  condition_ml0_kl1 : all data of k_dim in L1 buffer and the equivalent data of ml0 in L1 buffer are loaded to Ub.
  condition_m0_kl1 : all data of k_dim in L1 buffer and the partial data of ml0 in L1 buffer are loaded to Ub.
  condition_ml0_kl0 : the equivalent data of kl0 and ml0 in L1 buffer are loaded to Ub.
  condition_ml0_k0 : the equivalent data of ml0 in m_dim and one block of data in k_dim are loaded to Ub.
  condition_m0_kl0 : the equivalent data of kl0 in m_dim and one block of data in m_dim are loaded to Ub.
  condition_m0_k0 : one block of data in m_dim and one block of data in k_dim are loaded to Ub.
  */
  bool condition_m2_k2 = false;
  bool condition_ml1_kl1 = false;
  bool condition_ml1_kl0 = false;
  bool condition_ml1_k0 = false;
  bool condition_ml0_kl1 = false;
  bool condition_m0_kl1 = false;
  bool condition_ml0_kl0 = false;
  bool condition_ml0_k0 = false;
  bool condition_m0_kl0 = false;
  bool condition_m0_k0 = false;
};

struct BUbStatusCondition {
  /* This struct is used to storage the tiling condition of BUb.
  condition_k2_n2 : all data in k_dim and n_dim are loaded to Ub
  condition_kl1_nl1 : all data of k_dim and n_dim in L1 buffer are loaded to Ub
  condition_kl0_nl1: all data of n_dim in L1 buffer and the equivalent data of kl0 in L1 buffer are loaded to Ub.
  condition_k0_nl1: all data of n_dim in L1 buffer and partial data of kl0 in L1 buffer are loaded to Ub.
  condition_kl1_nl0 : all data of k_dim in L1 buffer and the equivalent data of nl0 in L1 buffer are loaded to Ub.
  condition_kl1_n0 : all data of k_dim in L1 buffer and the partial data of nl0 in L1 buffer are loaded to Ub.
  condition_kl0_nl0 : the equivalent data of kl0 and nl0 in L1 buffer are loaded to Ub.
  condition_k0_nl0 : the equivalent data of nl0 in m_dim and one block of data in k_dim are loaded to Ub.
  condition_kl0_n0 : the equivalent data of kl0 in m_dim and one block of data in n_dim are loaded to Ub.
  condition_k0_n0 : one block of data in n_dim and one block of data in k_dim are loaded to Ub.
  */
  bool condition_k2_n2 = false;
  bool condition_kl1_nl1 = false;
  bool condition_kl0_nl1 = false;
  bool condition_k0_nl1 = false;
  bool condition_kl1_nl0 = false;
  bool condition_kl1_n0 = false;
  bool condition_kl0_nl0 = false;
  bool condition_k0_nl0 = false;
  bool condition_kl0_n0 = false;
  bool condition_k0_n0 = false;
};

class PreUbTiling
{
public:
  int32_t k_aub = 1;
  int32_t m_aub = 1;
  int32_t k_bub = 1;
  int32_t n_bub = 1;
  PreUbTiling() = default;
  void update_tiling(int32_t new_k_aub, int32_t new_m_aub, int32_t new_k_bub, int32_t new_n_bub)
  {
    k_aub = new_k_aub;
    m_aub = new_m_aub;
    k_bub = new_k_bub;
    n_bub = new_n_bub;
  }
  ~PreUbTiling() = default;
};

int32_t MapShape(int32_t shape) {
  int32_t seed = kSeedMapMin;
  if (shape < seed) {
    return shape;
  }
  while (seed < kSeedMapMax) {
    if (seed <= shape && (seed << 1) > shape) {
      break;
    }
    seed = seed << 1;
  }
  return seed;
}

void GetFactorCnt(const int32_t &shape, int32_t &factor_cnt, const int32_t &factor_start, const int32_t &factor_end) {
  for (int32_t i = factor_start; i <= factor_end; i++) {
    if (shape < i) {
      return;
    }
    if (shape % i == 0) {
      ++factor_cnt;
    }
  }
}

void NonFactorMap(const string &op_type, BatchmatmulParas &params, BlockDimCalculator &blockDimCalculator) {
  const BatchmatmulCompileParas &compile_params = *(params.compile_params);
  BatchmatmulRunParas &run_params = *(params.run_params);
  run_params.batch_mapped = run_params.batch_32;
  run_params.m_mapped = run_params.m_32;
  run_params.k_mapped = run_params.k_32;
  run_params.n_mapped = run_params.n_32;
  GetFactorCnt(run_params.batch_32, blockDimCalculator.batch_factor_cnt, 1, compile_params.core_num);
  GetFactorCnt(run_params.m_32, blockDimCalculator.m_factor_less_64_cnt, 1, kL0FactorLimit);
  GetFactorCnt(run_params.m_32, blockDimCalculator.m_factor_less_1024_cnt, kL0FactorLimit + 1, kL1FactorLimit);
  GetFactorCnt(run_params.k_32, blockDimCalculator.k_factor_less_64_cnt, 1, kL0FactorLimit);
  GetFactorCnt(run_params.k_32, blockDimCalculator.k_factor_less_1024_cnt, kL0FactorLimit + 1, kL1FactorLimit);
  GetFactorCnt(run_params.n_32, blockDimCalculator.n_factor_less_64_cnt, 1, kL0FactorLimit);
  GetFactorCnt(run_params.n_32, blockDimCalculator.n_factor_less_1024_cnt, kL0FactorLimit + 1, kL1FactorLimit);
  // Split k will introduce atomic_add which can't be used with shift_inwards.
  // Thus in split k mode, batch/m/n/ can't use non-factorial segmentation.
  if (compile_params.split_k_flag) {
    if ((run_params.k_32 > kL0FactorLimit && blockDimCalculator.k_factor_less_64_cnt <= kL0FactorNumLimit) ||
        (run_params.k_32 > kL1FactorLimit &&
         blockDimCalculator.k_factor_less_64_cnt + blockDimCalculator.k_factor_less_1024_cnt <= kL1FactorNumLimit)) {
      run_params.k_mapped = MapShape(run_params.k_32);
    }
  } else {
    if (run_params.batch_32 > 1 && blockDimCalculator.batch_factor_cnt <= kL0FactorNumLimit) {
      run_params.batch_mapped = MapShape(run_params.batch_32);
    }
    bool m_factor_not_enough = (run_params.m_32 > kL0FactorLimit &&
      blockDimCalculator.m_factor_less_64_cnt <= kL0FactorNumLimit) ||
      (run_params.m_32 > kL1FactorLimit &&
       blockDimCalculator.m_factor_less_64_cnt + blockDimCalculator.m_factor_less_1024_cnt <= kL1FactorNumLimit);
    if (m_factor_not_enough) {
      run_params.m_mapped = MapShape(run_params.m_32);
    }
    bool n_factor_not_enough = (run_params.n_32 > kL0FactorLimit &&
      blockDimCalculator.n_factor_less_64_cnt <= kL0FactorNumLimit) ||
      (run_params.n_32 > kL1FactorLimit &&
        blockDimCalculator.n_factor_less_64_cnt + blockDimCalculator.n_factor_less_1024_cnt <= kL1FactorNumLimit);
    if (n_factor_not_enough) {
      run_params.n_mapped = MapShape(run_params.n_32);
    }
  }
  OP_LOGD(op_type.c_str(),
          "NonFactorMap get mapped shape: batch_mapped: %d, m_mapped: %d, k_mapped: %d, n_mapped: %d.",
          run_params.batch_mapped, run_params.m_mapped, run_params.k_mapped, run_params.n_mapped);
}

void Tiling::SetParams(const CoreStatus &coreStatus, const L0Status &l0Status, const L1Status &l1Status,
                       const UbStatus &ubStatus, const BatchmatmulParas &params)
{
  batch_dim = coreStatus.batch_dim;
  n_dim = coreStatus.n_dim;
  m_dim = coreStatus.m_dim;
  k_dim = coreStatus.k_dim;
  m_l0 = l0Status.m_l0;
  k_l0 = l0Status.k_l0;
  n_l0 = l0Status.n_l0;
  kal1_16 = l1Status.kal1_16;
  kbl1_16 = l1Status.kbl1_16;
  kal1_factor = coreStatus.kal1_factor;
  kbl1_factor = coreStatus.kbl1_factor;
  m_al1 = l1Status.m_al1;
  n_bl1 = l1Status.n_bl1;
  db_al1 = l1Status.db_al1;
  db_bl1 = l1Status.db_bl1;
  n_cub = ubStatus.n_cub;
  db_cub = ubStatus.db_cub;
  k_org_dim = kal1_factor * kal1_16 * kBlockSize;
  db_l0c = l0Status.db_l0c;
  al1_full_load = l1Status.al1_full_load;
  bl1_full_load = l1Status.bl1_full_load;
  if (params.compile_params->nd_flag) {
    k_aub = ubStatus.k_aub;
    m_aub = ubStatus.m_aub;
    db_aub = ubStatus.db_aub;
    k_bub = ubStatus.k_bub;
    n_bub = ubStatus.n_bub;
    db_bub = ubStatus.db_bub;
    aub_multi_flag = ubStatus.aub_multi_flag;
    bub_multi_flag = ubStatus.bub_multi_flag;
    a_align_value = ubStatus.a_align_value;
    b_align_value = ubStatus.b_align_value;
    aub_align_bound = ubStatus.aub_align_bound;
    bub_align_bound = ubStatus.bub_align_bound;
  }

  if (al1_full_load && coreStatus.batch == 1) {
    db_al1 = 1;
  }
  if (bl1_full_load && (!params.run_params->b_have_batch || coreStatus.batch == 1)) {
    db_bl1 = 1;
  }
  min_kl1_cmp_kl0 = (min(kal1_16, kbl1_16) == k_l0) ? 0 : 1;
}

void Tiling::SetAttachFlag()
{
  // find kernel ID
  bool kAl1FullLoad = kal1_16 * kBlockSize == k_org_dim;
  bool kBl1FullLoad = kbl1_16 * kBlockSize == k_org_dim;
  bool template1 = al1_full_load && bl1_full_load;
  bool template2 = al1_full_load && !bl1_full_load && kBl1FullLoad;
  bool template3 = al1_full_load && !bl1_full_load && !kBl1FullLoad;
  bool template4 = !al1_full_load && bl1_full_load && kAl1FullLoad;
  bool template5 = !al1_full_load && bl1_full_load && !kAl1FullLoad;
  bool template6 = !al1_full_load && !bl1_full_load && kAl1FullLoad && kBl1FullLoad;
  bool template7 = !al1_full_load && !bl1_full_load && kAl1FullLoad && !kBl1FullLoad;
  bool template8 = !al1_full_load && !bl1_full_load && !kAl1FullLoad && kBl1FullLoad;
  bool template9 = !al1_full_load && !bl1_full_load && !kAl1FullLoad && !kBl1FullLoad;
  bool condition1 = template1 || template2 || template3;
  bool condition2 = template4 || template6 || template7;
  bool condition3 = template5 || template8 || template9;
  bool condition4 = template1 || template4 || template5;
  bool condition5 = template2 || template6 || template8;
  bool condition6 = template3 || template7 || template9;
  bool condition7 = template1 || template2 || template4 || template6;
  bool condition8 = template3 || template7;
  bool condition9 = template5 || template8;

  if (condition1) {
    al1_attach_flag = kAttachFlagZero;
  }
  if (condition2) {
    al1_attach_flag = kAttachFlagOne;
  }
  if (condition3) {
    al1_attach_flag = kAttachFlagTwo;
  }
  if (condition4) {
    bl1_attach_flag = kAttachFlagZero;
  }
  if (condition5) {
    bl1_attach_flag = kAttachFlagOne;
  }
  if (condition6) {
    bl1_attach_flag = kAttachFlagTwo;
  }
  if (condition7) {
    abkl1_attach_flag = kAttachFlagZero;
  }
  if (condition8) {
    abkl1_attach_flag = kAttachFlagOne;
  }
  if (condition9) {
    abkl1_attach_flag = kAttachFlagTwo;
  }
  if (template9) {
    if (kal1_16 == kbl1_16) {
      abkl1_attach_flag = kAttachFlagZero;
    } else if (kal1_16 > kbl1_16) {
      abkl1_attach_flag = kAttachFlagOne;
    } else if (kal1_16 < kbl1_16) {
      abkl1_attach_flag = kAttachFlagTwo;
    }
  }
}

void Tiling::GetTilingId(const BatchmatmulParas &params)
{
  const BatchmatmulCompileParas &compile_params = *(params.compile_params);
  const BatchmatmulRunParas &run_params = *(params.run_params);
  int32_t tilingIDLongLong = compile_params.split_k_flag ? 1 : 0;
  // db_flag includes db_al1, db_bl1 and db_l0c flags, its value from 1 to 8 represent 8 different combinations
  int32_t db_flag = ((db_al1 - 1) << 2) + ((db_bl1 - 1) << 1) + db_l0c;
  tilingIDLongLong = tilingIDLongLong * kDecimal + db_flag;
  tilingIDLongLong = tilingIDLongLong * kDecimal + abkl1_attach_flag;
  tilingIDLongLong = tilingIDLongLong * kDecimal + al1_attach_flag;
  tilingIDLongLong = tilingIDLongLong * kDecimal + bl1_attach_flag;
  tilingIDLongLong = tilingIDLongLong * kDecimal + min_kl1_cmp_kl0;
  if (compile_params.nd_flag) {
    tilingIDLongLong = tilingIDLongLong * kDecimal + aub_multi_flag;
    tilingIDLongLong = tilingIDLongLong * kDecimal + bub_multi_flag;
  }
  tilingIDLongLong = tilingIDLongLong * kDecimal + run_params.non_factor_k;
  tilingIDLongLong = tilingIDLongLong * kDecimal + run_params.non_factor_bmn;
  this->tiling_id = std::to_string(tilingIDLongLong);
}

void GetFactors(int32_t *cnt, int32_t *factorList, const int32_t &num, const int32_t &maxNum)
{
  // get all factors of num which smaller or equal to maxNum
  for (int32_t i = 1; i < maxNum + 1; i++) {
    if (num % i == 0) {
      factorList[(*cnt)++] = i;
    }
  }
}

void GetTwoFactors(int32_t *res, const int32_t &base, const int32_t &dim, const int32_t &maxNum = 32, int32_t cnt = 0)
{
  // for up bigger or equal to base + 1, find the smallest num which is a factor of dim
  // form down smaller or equal to base, find the biggest num which is a factor of dim
  int32_t up = base + 1;
  int32_t maxCnt = 2;
  while (up <= dim) {
    if (up > maxNum) {
      break;
    }
    if (dim % up == 0) {
      res[cnt++] = up;
      break;
    }
    up++;
  }
  int32_t down = base;
  while (down >= 1) {
    if (dim % down == 0) {
      res[cnt++] = down;
      if (cnt == maxCnt) {
        break;
      }
    }
    down--;
  }
}

void GetNearestFactor(const int32_t &base, int32_t &factor)
{
  while (factor > 0 && base % factor != 0) {
    factor--;
  }
}

void BL1FullLoadBlock(const CoreStatus &coreStatus, BlockDimCalculator &blockDimCalculator,
                      int32_t &n0, bool b_have_batch)
{
  if (n0 >= 1) {
    while (coreStatus.n % n0 != 0) {
      n0--;
    }
    blockDimCalculator.amat_size = blockDimCalculator.ori_amat_size * ceil(static_cast<double>(coreStatus.n) / n0);
    blockDimCalculator.bmat_size = b_have_batch ? coreStatus.batch * coreStatus.n : coreStatus.n;
    blockDimCalculator.total_load_size = blockDimCalculator.amat_size + blockDimCalculator.bmat_size;
    blockDimCalculator.tmp_value = n0;
  }
}

void UpdateBlockDimCalculator(BlockDimCalculator &blockDimCalculator)
{
  if (blockDimCalculator.total_load_size > blockDimCalculator.tmp_load_size) {
      blockDimCalculator.bmat_size = blockDimCalculator.tmp_bmat_size;
      blockDimCalculator.amat_size = blockDimCalculator.tmp_amat_size;
      blockDimCalculator.total_load_size = blockDimCalculator.tmp_load_size;
      blockDimCalculator.tmp_value = 0;
  }
}

void AL1FullLoadBlock(const CoreStatus &coreStatus, BlockDimCalculator &blockDimCalculator, int32_t &m0)
{
  if (m0 >= 1) {
    while (coreStatus.m % m0 != 0) {
      m0--;
    }
    blockDimCalculator.tmp_amat_size = blockDimCalculator.ori_amat_size;
    blockDimCalculator.tmp_bmat_size = coreStatus.n * ceil(static_cast<double>(blockDimCalculator.ori_amat_size) / m0);
    blockDimCalculator.tmp_load_size = blockDimCalculator.tmp_amat_size + blockDimCalculator.tmp_bmat_size;
    UpdateBlockDimCalculator(blockDimCalculator);
  }
}

void NeitherFullLoadBlock(const CoreStatus &coreStatus, BlockDimCalculator &blockDimCalculator,
                          const int32_t (&nFactorTwoCandidates)[kCandidateLen],
                          const int32_t (&mFactorTwoCandidates)[kCandidateLen])
{
  for (auto const &n0: nFactorTwoCandidates) {
    if (n0 <= 0) {
      continue;
    }
    int32_t max_m0 = kL0cSize / (kKbytes * n0);
    int32_t m0_arr[kCandidateLen] = {0};
    GetTwoFactors(m0_arr, max_m0, coreStatus.m, max_m0);
    for (auto const &m0: m0_arr) {
      if (m0 <= 0) {
        continue;
      }
      blockDimCalculator.tmp_amat_size =
          blockDimCalculator.ori_amat_size * ceil(static_cast<double>(coreStatus.n) / n0);
      blockDimCalculator.tmp_bmat_size =
          coreStatus.n * ceil(static_cast<double>(blockDimCalculator.ori_amat_size) / m0);
      blockDimCalculator.tmp_load_size = blockDimCalculator.tmp_amat_size + blockDimCalculator.tmp_bmat_size;
      UpdateBlockDimCalculator(blockDimCalculator);
    }
  }
  for (auto const &m0: mFactorTwoCandidates) {
    if (m0 <= 0) {
      continue;
    }
    int32_t max_n0 = kL0cSize / (kKbytes * m0);
    int32_t n0_arr[kCandidateLen] = {0};
    GetTwoFactors(n0_arr, max_n0, coreStatus.n, max_n0);
    for (auto const &n0: n0_arr) {
      if (n0 <= 0) {
        continue;
      }
      blockDimCalculator.tmp_amat_size =
          blockDimCalculator.ori_amat_size * ceil(static_cast<double>(coreStatus.n) / n0);
      blockDimCalculator.tmp_bmat_size =
          coreStatus.n * ceil(static_cast<double>(blockDimCalculator.ori_amat_size) / m0);
      blockDimCalculator.tmp_load_size = blockDimCalculator.tmp_amat_size + blockDimCalculator.tmp_bmat_size;
      UpdateBlockDimCalculator(blockDimCalculator);
    }
  }
}

void GetBlockDimHelper(CoreStatus &coreStatus, BlockDimCalculator &blockDimCalculator,
                       const int32_t m0s[][kCandidateLen], const int32_t n0s[][kCandidateLen],
                       const BatchmatmulParas &params)
{
  const BatchmatmulCompileParas &compile_params = *(params.compile_params);
  const BatchmatmulRunParas &run_params = *(params.run_params);
  int32_t bFactor = blockDimCalculator.batch_dim_array[blockDimCalculator.batch_idx];
  int32_t nFactor = blockDimCalculator.n_dim_array[blockDimCalculator.n_idx];
  blockDimCalculator.tmp_core_use = bFactor * nFactor;
  bool need_cal_load_size = blockDimCalculator.tmp_core_use > compile_params.core_num ||
    blockDimCalculator.tmp_core_use == 0;
  if (need_cal_load_size) {
    return;
  }
  for (int32_t mIdx = 0; mIdx < blockDimCalculator.m_dim_cnt; mIdx++) {
    int32_t mFactor = blockDimCalculator.m_dim_array[mIdx];
    blockDimCalculator.tmp_core_use = bFactor * nFactor * mFactor;
    need_cal_load_size = mFactor == 0 || blockDimCalculator.tmp_core_use > compile_params.core_num;
    if (need_cal_load_size) {
      continue;
    }
    for (int32_t kIdx = 0; kIdx < blockDimCalculator.k_dim_cnt; kIdx++) {
      int32_t kFactor = blockDimCalculator.k_dim_array[kIdx];
      blockDimCalculator.tmp_core_use = bFactor * nFactor * mFactor * kFactor;
      need_cal_load_size = kFactor == 0 || blockDimCalculator.tmp_core_use > compile_params.core_num;
      if (need_cal_load_size) {
        continue;
      }
      blockDimCalculator.k_num = run_params.k_32 / kFactor * kBlockSize * kBlockSize;
      blockDimCalculator.k_bytes = blockDimCalculator.k_num * kFp16Bytes;
      coreStatus.batch = ceil(static_cast<double>(run_params.batch_32) / bFactor);
      coreStatus.m = ceil(static_cast<double>(run_params.m_32) / mFactor);
      coreStatus.n = ceil(static_cast<double>(run_params.n_32) / nFactor);
      coreStatus.k = run_params.k_32 / kFactor;
      if (run_params.k_mapped != run_params.k_32 && kIdx < kIdxTwo) {
        blockDimCalculator.k_num = run_params.k_mapped / kFactor * kNumTwo * kBlockSize * kBlockSize;
        coreStatus.k = run_params.k_mapped / kFactor * kNumTwo;
      }
      // load size of A matrix is batch * m
      // load size of B matrix is n
      blockDimCalculator.ori_amat_size = coreStatus.batch * coreStatus.m;
      blockDimCalculator.ori_bmat_size = run_params.b_have_batch ? coreStatus.batch * coreStatus.n : coreStatus.n;
      blockDimCalculator.amat_size = blockDimCalculator.ori_amat_size;
      blockDimCalculator.bmat_size = blockDimCalculator.ori_bmat_size;
      blockDimCalculator.total_load_size = blockDimCalculator.amat_size + blockDimCalculator.bmat_size;
      blockDimCalculator.tmp_value = 0;
      if (blockDimCalculator.total_load_size * blockDimCalculator.k_bytes > kL1Size) {
        blockDimCalculator.total_load_size = INT_MAX;
        // BL1 full load
        int32_t n0 =
          min(min((kL1Size / kFp16Bytes - kMinFractalSize) / blockDimCalculator.k_num, coreStatus.n), kMaxFactor);
        BL1FullLoadBlock(coreStatus, blockDimCalculator, n0, run_params.b_have_batch);
        // AL1 full load
        int32_t m0 = min(min((kL1Size / kFp16Bytes - kMinFractalSize) /
                             (blockDimCalculator.k_num * blockDimCalculator.ori_amat_size),
                             blockDimCalculator.ori_amat_size),
                         kMaxFactor);
        AL1FullLoadBlock(coreStatus, blockDimCalculator, m0);
        // neither full load max_m max_n
        // closest m and n
        NeitherFullLoadBlock(coreStatus, blockDimCalculator, n0s[nFactor], m0s[mFactor]);
      }
      int32_t loadSizeKb = blockDimCalculator.total_load_size * blockDimCalculator.k_bytes / kKbytes;
      int32_t minLoadSizeKb = blockDimCalculator.min_load_size * blockDimCalculator.k_bytes / kKbytes;
      double tmpBlockingPct;
      if (nFactor > mFactor) {
        tmpBlockingPct = double(blockDimCalculator.amat_size) / blockDimCalculator.total_load_size;
      } else if (nFactor < mFactor) {
        tmpBlockingPct = double(blockDimCalculator.bmat_size) / blockDimCalculator.total_load_size;
      } else {
        tmpBlockingPct =
          double(max(blockDimCalculator.amat_size, blockDimCalculator.bmat_size)) / blockDimCalculator.total_load_size;
      }
      if (run_params.k_mapped != run_params.k_32) {
        blockDimCalculator.total_load_size *= coreStatus.k;
      }
      bool tmp_blocking_flag = (loadSizeKb < kLoadSizeRangeLow && max(nFactor, mFactor) > kMLowRange);
      // updateSolution: bool whether update to a new block factor solution
      // has smaller LoadSize or the same LoadSize but batch
      bool update_condition_loadsize = blockDimCalculator.total_load_size < blockDimCalculator.min_load_size;
      bool update_condition_batch_n_dim = (blockDimCalculator.total_load_size == blockDimCalculator.min_load_size) &&
        ((blockDimCalculator.n_dim_factor * blockDimCalculator.batch_dim_factor < bFactor * nFactor) ||
        (blockDimCalculator.n_dim_factor * blockDimCalculator.batch_dim_factor == bFactor * nFactor &&
        blockDimCalculator.batch_dim_factor < bFactor));
      auto update_solution = update_condition_loadsize || update_condition_batch_n_dim ||
        (blockDimCalculator.final_blocking_flag && (loadSizeKb - minLoadSizeKb) < kLoadSizeDiffRange &&
        max(nFactor, mFactor) < max(blockDimCalculator.n_dim_factor, blockDimCalculator.m_dim_factor));
      auto no_update_solution =
        (((loadSizeKb >= kLoadSizeRangeLow && loadSizeKb < kLoadSizeRangeHigh &&
        max(nFactor, mFactor) > kMHighRange && tmpBlockingPct > kBlockingPctGate) &&
        max(nFactor, mFactor) > max(blockDimCalculator.n_dim_factor, blockDimCalculator.m_dim_factor) &&
        double(blockDimCalculator.min_load_size - blockDimCalculator.total_load_size) /
        blockDimCalculator.min_load_size < kLoadSizeGate &&
        blockDimCalculator.core_use >= kCoreUseHighRange) ||
        ((loadSizeKb < kLoadSizeRangeLow && max(nFactor, mFactor) > kMLowRange) &&
        (max(nFactor, mFactor) > max(blockDimCalculator.n_dim_factor, blockDimCalculator.m_dim_factor)) &&
        ((minLoadSizeKb - loadSizeKb) < kLoadSizeDiffRange && blockDimCalculator.core_use > kCoreUseLowRange)));
      auto update_condition = update_solution && !no_update_solution;
      if (update_condition) {
        blockDimCalculator.min_load_size = blockDimCalculator.total_load_size;
        blockDimCalculator.n_dim_factor = nFactor;
        blockDimCalculator.batch_dim_factor = bFactor;
        blockDimCalculator.m_dim_factor = mFactor;
        blockDimCalculator.k_dim_factor = kFactor;
        blockDimCalculator.final_blocking_flag = tmp_blocking_flag;
        blockDimCalculator.core_use = blockDimCalculator.tmp_core_use;
        blockDimCalculator.final_value = blockDimCalculator.tmp_value;
      }
    }
  }
}

void GetBandwidth(const BatchmatmulCompileParas &params, const int64_t &use_out_buffer_size, int32_t &hbm_bandwidth,
                  int32_t &l2_bandwidth, int32_t &cur_bandwidth)
{
  int32_t abs_core_num_8 = abs(params.core_num - kMinSplitKCoreNum);
  int32_t abs_core_num_32 = abs(params.core_num - kCoreNum32);
  if (abs_core_num_8 < abs_core_num_32) {
    hbm_bandwidth = kHbmBandwidth8Core;
    l2_bandwidth = kL2Bandwidth8Core;
  } else {
    hbm_bandwidth = kHbmBandwidth32Core;
    l2_bandwidth = kL2Bandwidth32Core;
  }
  cur_bandwidth = use_out_buffer_size < params.l2_size ? l2_bandwidth : hbm_bandwidth;
}

void ComputePerfSplitK(const int32_t block_dims[], const int32_t single_core_shape[], int32_t &min_cost,
                       const BatchmatmulParas &params, CoreStatus &coreStatus)
{
  const BatchmatmulCompileParas &compile_params = *(params.compile_params);
  const BatchmatmulRunParas &run_params = *(params.run_params);
  int32_t m_dim = block_dims[0];
  int32_t k_dim = block_dims[1];
  int32_t n_dim = block_dims[kIdxTwo];
  int32_t batch_dim_max = block_dims[kIdxThree];
  if (k_dim * n_dim * m_dim > compile_params.core_num) {
    return;
  }
  for (int32_t batch_dim = 1; batch_dim <= batch_dim_max; batch_dim++) {
    if (k_dim * n_dim * m_dim * batch_dim > compile_params.core_num) {
      return;
    }
    int32_t single_core_m = single_core_shape[0];
    int32_t single_core_k = single_core_shape[1];
    int32_t single_core_n = single_core_shape[kIdxTwo];
    int32_t single_core_batch = run_params.batch_32 / batch_dim;
    int32_t atomic_add_bw_lose = k_dim == 1 ? 1 : kNumTwo;
    int32_t mte3_cost = k_dim * (single_core_batch * single_core_m * single_core_n * kFp32Bytes) * atomic_add_bw_lose;
    int32_t base_load_cost =
      single_core_batch * (single_core_m * single_core_k + single_core_k * single_core_n) * kFp16Bytes;
    int32_t b_repeat_load_cost = (batch_dim * m_dim - 1) * single_core_k * single_core_n * kFp16Bytes;
    int32_t a_repeat_load_cost = (batch_dim * n_dim - 1) * single_core_k * single_core_m * kFp16Bytes;
    int32_t cur_cost = base_load_cost + mte3_cost + a_repeat_load_cost + b_repeat_load_cost;
    if (cur_cost < min_cost) {
      min_cost = cur_cost;
      coreStatus.k_dim = k_dim;
    }
  }
}

void GetSplitKdim(const string &op_type, BatchmatmulParas &params, CoreStatus &coreStatus)
{
  // support multi cores slicing along k dim
  // get batch_dim, m_dim, n_dim and k_dim
  // batch_dim, m_dim, n_dim, k_dim is a factor of input batch, m, n, k
  const BatchmatmulCompileParas &compile_params = *(params.compile_params);
  const BatchmatmulRunParas &run_params = *(params.run_params);
  OP_LOGD(op_type.c_str(), "GetSplitKdim input shape batch:%d, m:%d, k:%d, n:%d", run_params.batch_32, run_params.m_32,
          run_params.k_32, run_params.n_32);
  if (compile_params.core_num < kMinSplitKCoreNum) {
    coreStatus.k_dim = 1;
    OP_LOGD(op_type.c_str(), "CORENUM < 8 so multi-core slicing factor k_dim:%d", coreStatus.k_dim);
    return;
  }
  int32_t use_out_buffer_size =
      run_params.batch_32 *
      (run_params.m_32 * run_params.k_32 + run_params.k_32 * run_params.n_32 + run_params.m_32 * run_params.n_32) *
      kFp16Bytes;
  int32_t cur_bandwidth = 0;
  int32_t hbm_bandwidth = 0;
  int32_t l2_bandwidth = 0;
  GetBandwidth(compile_params, use_out_buffer_size, hbm_bandwidth, l2_bandwidth, cur_bandwidth);
  int32_t min_cost = compile_params.core_num * use_out_buffer_size / hbm_bandwidth * cur_bandwidth;
  int32_t batch_dim_max = min(compile_params.core_num, run_params.batch_32);
  int32_t m_dim_max = min(compile_params.core_num, run_params.m_32);
  int32_t k_dim_max = min(compile_params.core_num, run_params.k_32);
  int32_t n_dim_max = min(compile_params.core_num, run_params.n_32);
  int32_t block_dims[kNumFour] = {1, 1, 1, 1};
  int32_t single_core_shape[kNumFour] = {run_params.m_32, run_params.k_32, run_params.n_32, run_params.batch_32};
  block_dims[kIdxThree] = batch_dim_max;
  for (int32_t k = 1; k <= k_dim_max; k++) {
    for (int32_t n = 1; n <= n_dim_max; n++) {
      if (k * n > compile_params.core_num) {
        break;
      }
      for (int32_t m = 1; m <= m_dim_max; m++) {
        block_dims[0] = m;
        block_dims[1] = k;
        block_dims[kIdxTwo] = n;
        single_core_shape[0] = run_params.m_32 / m;
        single_core_shape[1] = run_params.k_32 / k;
        single_core_shape[kIdxTwo] = run_params.n_32 / n;
        ComputePerfSplitK(block_dims, single_core_shape, min_cost, params, coreStatus);
      }
    }
  }
  OP_LOGD(op_type.c_str(), "multi-core slicing factor k_dim:%d", coreStatus.k_dim);
}

void SetNonfactorFlag(BatchmatmulRunParas &run_params, const CoreStatus &coreStatus,
                      bool all_full_load_flag, const int32_t kDimArray[]) {
  // non-factor split is only used for multi-core split, not considered when the number of core_dim is 1
  run_params.m_mapped = coreStatus.m_dim == 1 ? run_params.m_32 : run_params.m_mapped;
  run_params.n_mapped = coreStatus.n_dim == 1 ? run_params.n_32 : run_params.n_mapped;
  run_params.batch_mapped = coreStatus.batch_dim == 1 ? run_params.batch_32 : run_params.batch_mapped;
  if (all_full_load_flag) {
    run_params.k_mapped = coreStatus.k_dim == 1 ? run_params.k_32 : run_params.k_mapped;
  } else {
    run_params.k_mapped =
        (coreStatus.k_dim == 1 || coreStatus.k_dim == kDimArray[kIdxTwo] || coreStatus.k_dim == kDimArray[kIdxThree])
            ? run_params.k_32
            : run_params.k_mapped;
  }
  run_params.non_factor_k = run_params.k_mapped == run_params.k_32 ? false : true;
  run_params.non_factor_bmn = (run_params.batch_mapped == run_params.batch_32 &&
                               run_params.n_mapped == run_params.n_32 &&
                               run_params.m_mapped == run_params.m_32) ? false : true;
}

int32_t GetBlockDim(const string &op_type, BatchmatmulParas &params, CoreStatus &coreStatus,
                    BlockDimCalculator &blockDimCalculator)
{
  // get batch_dim, k_dim, m_dim and n_dim for single core
  // support multi cores slicing along k_dim
  // single core batch_dim, m_dim, n_dim, k_dim is a factor of input batch, m, n, k
  const BatchmatmulCompileParas &compile_params = *(params.compile_params);
  BatchmatmulRunParas &run_params = *(params.run_params);
  OP_LOGD(op_type.c_str(), "GetBlockDim input batch:%d, m:%d, k:%d, n:%d, k_dim:%d", run_params.batch_mapped,
          run_params.m_mapped, run_params.k_mapped, run_params.n_mapped, coreStatus.k_dim);
  // first get k_dim candidate
  int32_t kDimArray[compile_params.core_num] = {0};
  if (coreStatus.k_dim == 1) {
    kDimArray[0] = 1;
    blockDimCalculator.k_dim_cnt = 1;
    run_params.k_mapped = run_params.k_32;
  } else {
    if (run_params.k_mapped != run_params.k_32) {
      GetTwoFactors(kDimArray, coreStatus.k_dim, run_params.k_mapped, compile_params.core_num);
      blockDimCalculator.k_dim_cnt += kCandidateLen;
    }
    GetTwoFactors(kDimArray, coreStatus.k_dim, run_params.k_32, compile_params.core_num, blockDimCalculator.k_dim_cnt);
    blockDimCalculator.k_dim_cnt += kCandidateLen;
  }
  if (run_params.batch_32 * run_params.m_32 * run_params.n_32 * kDimArray[0] <= compile_params.core_num) {
    coreStatus.batch_dim = run_params.batch_32;
    coreStatus.n_dim = run_params.n_32;
    coreStatus.m_dim = run_params.m_32;
    coreStatus.k_dim = kDimArray[0];
    SetNonfactorFlag(run_params, coreStatus, true, kDimArray);
    coreStatus.batch = 1;
    coreStatus.m = 1;
    coreStatus.k = run_params.k_mapped / kDimArray[0];
    coreStatus.n = 1;
    OP_LOGD(op_type.c_str(), "multi-core slicing factor batch_dim:%d, n_dim:%d, m_dim:%d, k_dim:%d",
            coreStatus.batch_dim, coreStatus.n_dim, coreStatus.m_dim, coreStatus.k_dim);
    return 0;
  }
  int32_t batchDimArray[compile_params.core_num] = {0};
  int32_t nDimArray[compile_params.core_num] = {0};
  int32_t mDimArray[compile_params.core_num] = {0};
  GetFactors(&blockDimCalculator.batch_dim_cnt, batchDimArray, run_params.batch_mapped, compile_params.core_num);
  GetFactors(&blockDimCalculator.n_dim_cnt, nDimArray, run_params.n_mapped, compile_params.core_num);
  GetFactors(&blockDimCalculator.m_dim_cnt, mDimArray, run_params.m_mapped, compile_params.core_num);
  int32_t m0s[compile_params.core_num + 1][kCandidateLen] = {0};
  int32_t n0s[compile_params.core_num + 1][kCandidateLen] = {0};
  for (int32_t idx = 0; idx < blockDimCalculator.n_dim_cnt; idx++) {
    int32_t tmpNDim = nDimArray[idx];
    int32_t tmpNSingleCore = ceil(static_cast<double>(run_params.n_mapped) / tmpNDim);
    GetTwoFactors(n0s[tmpNDim], kMNPntMax, tmpNSingleCore, kMaxFactor);
  }
  for (int32_t idx = 0; idx < blockDimCalculator.m_dim_cnt; idx++) {
    int32_t tmpMDim = mDimArray[idx];
    int32_t tmpMSingleCore = ceil(static_cast<double>(run_params.m_mapped) / tmpMDim);
    GetTwoFactors(m0s[tmpMDim], kMNPntMax, tmpMSingleCore, kMaxFactor);
  }
  blockDimCalculator.n_dim_factor = 1;
  blockDimCalculator.batch_dim_factor = 1;
  blockDimCalculator.m_dim_factor = 1;
  blockDimCalculator.k_dim_factor = 1;
  blockDimCalculator.min_load_size = INT_MAX;
  blockDimCalculator.batch_dim_array = batchDimArray;
  blockDimCalculator.m_dim_array = mDimArray;
  blockDimCalculator.n_dim_array = nDimArray;
  blockDimCalculator.k_dim_array = kDimArray;
  for (int32_t batch_idx = 0; batch_idx < blockDimCalculator.batch_dim_cnt; batch_idx++) {
    for (int32_t n_idx = 0; n_idx < blockDimCalculator.n_dim_cnt; n_idx++) {
      blockDimCalculator.batch_idx = batch_idx;
      blockDimCalculator.n_idx = n_idx;
      GetBlockDimHelper(coreStatus, blockDimCalculator, m0s, n0s, params);
    }
  }
  coreStatus.batch_dim = blockDimCalculator.batch_dim_factor;
  coreStatus.n_dim = blockDimCalculator.n_dim_factor;
  coreStatus.m_dim = blockDimCalculator.m_dim_factor;
  coreStatus.k_dim = blockDimCalculator.k_dim_factor;
  SetNonfactorFlag(run_params, coreStatus, false, kDimArray);
  coreStatus.m = run_params.m_mapped / blockDimCalculator.m_dim_factor;
  coreStatus.n = run_params.n_mapped / blockDimCalculator.n_dim_factor;
  coreStatus.k = run_params.k_mapped / blockDimCalculator.k_dim_factor;
  coreStatus.batch = run_params.batch_mapped / blockDimCalculator.batch_dim_factor;
  OP_LOGD(op_type.c_str(),
          "multi-core slicing factor batch_dim:%d, n_dim:%d, m_dim:%d, k_dim:%d, m_block_pnt_point:%d",
          coreStatus.batch_dim, coreStatus.n_dim, coreStatus.m_dim, coreStatus.k_dim, blockDimCalculator.final_value);
  return blockDimCalculator.final_value;
}

int32_t GetLoadSize(const CoreStatus &coreStatus, const L0Status &l0Status)
{
  bool al1FullLoad =
    ((coreStatus.m * coreStatus.k + l0Status.n_l0 * l0Status.k_l0) * kBlockSize * kBlockSize * kFp16Bytes <=
      kL1Size);
  bool bl1FullLoad =
    ((l0Status.m_l0 * l0Status.k_l0 + l0Status.n_l0 * coreStatus.k) * kBlockSize * kBlockSize * kFp16Bytes <=
      kL1Size);
  bool bothFullLoad = ((coreStatus.m * coreStatus.k + l0Status.n_l0 * coreStatus.k) * kBlockSize *
    kBlockSize * kFp16Bytes <=
    kL1Size);
  int32_t num0a =
    bl1FullLoad ? coreStatus.n : ((coreStatus.m + l0Status.m_l0 - 1) / l0Status.m_l0) * coreStatus.n;
  int32_t num0b =
    al1FullLoad ? coreStatus.m : ((coreStatus.n + l0Status.n_l0 - 1) / l0Status.n_l0) * coreStatus.m;
  if ((al1FullLoad && bl1FullLoad) && !bothFullLoad) {
    return min(coreStatus.n + ((coreStatus.n + l0Status.n_l0 - 1) / l0Status.n_l0) * coreStatus.m,
               coreStatus.m + ((coreStatus.m + l0Status.m_l0 - 1) / l0Status.m_l0) * coreStatus.n);
  }
  return num0a + num0b;
}

void GetFinalMkn(L0Status &l0Status, const CoreStatus &coreStatus, const int32_t &k0,
                 const int32_t &majorDimFactor, const int32_t &minorDimFactor)
{
  if (k0 == 0) {
    return;
  }
  if (l0Status.max_axis_idx == 0) {
    l0Status.m_l0 = majorDimFactor;
    l0Status.n_l0 = minorDimFactor;
  } else {
    l0Status.m_l0 = minorDimFactor;
    l0Status.n_l0 = majorDimFactor;
  }
  l0Status.k_l0 = k0;
  float tmpL0cUse = l0Status.m_l0 * l0Status.n_l0 * l0Status.db_l0c * kBlockSize * kBlockSize * 4 * 1.0 / kL0cSize;
  int32_t tmpMte1Loop = ((l0Status.n_l0 != 1) ? l0Status.k_l0 : 1) + ((l0Status.k_l0 != 1) ? l0Status.m_l0 : 1);
  int32_t tmpMul = l0Status.m_l0 * l0Status.k_l0 * l0Status.n_l0;
  int32_t tmpLoadSize = GetLoadSize(coreStatus, l0Status);
  auto condition1 = l0Status.final_ml0 == 0;
  auto condition2 = tmpLoadSize < l0Status.final_load_size;
  auto condition3 = (tmpLoadSize == l0Status.final_load_size && tmpMul > l0Status.final_mul &&
    tmpMul * tmpL0cUse >= l0Status.final_mul * l0Status.final_l0c_use);
  auto condition4 =
    tmpMul == l0Status.final_mul && tmpLoadSize == l0Status.final_load_size && tmpMte1Loop < l0Status.final_mte1Loop;
  if (condition1 || condition2 || condition3 || condition4) {
    l0Status.final_ml0 = l0Status.m_l0;
    l0Status.final_kl0 = l0Status.k_l0;
    l0Status.final_nl0 = l0Status.n_l0;
    l0Status.final_load_size = tmpLoadSize;
    l0Status.final_l0c_use = tmpL0cUse;
    l0Status.final_mul = tmpMul;
    l0Status.final_mte1Loop = tmpMte1Loop;
  }
}

MKNParasCombo GetParasCombo(const int32_t &index, const int32_t &blockValue)
{
  map<int32_t, MKNParasCombo> parasComboMap;
  if (blockValue == 0) {
    MKNParasCombo comboZero = {2, 2, 2, 64, 64, 128, 0, 64, 11};
    MKNParasCombo comboOne = {2, 2, 1, 64, 64, 256, 0, 64, 16};
    parasComboMap = {{0, comboZero}, {1, comboOne}};
  } else {
    MKNParasCombo comboZero = {2, 2, 2, 64, 64, 128, 1, 64, blockValue};
    MKNParasCombo comboOne = {2, 2, 1, 64, 64, 256, 1, 64, blockValue};
    parasComboMap = {{0, comboZero}, {1, comboOne}};
  }
  return parasComboMap[index];
}

void GetL0StatusFromParasCombo(L0Status &l0Status, int32_t *parasCombo)
{
  l0Status.SetInitLoadStatus();
  l0Status.db_l0a = parasCombo[kIdxZero];
  l0Status.db_l0b = parasCombo[kIdxOne];
  l0Status.db_l0c = parasCombo[kIdxTwo];
  l0Status.max_mk = parasCombo[kIdxThree];
  l0Status.max_nk = parasCombo[kIdxFour];
  l0Status.max_mn = parasCombo[kIdxFive];
  l0Status.max_axis_idx = parasCombo[kIdxSix];
  l0Status.max_axis_num = parasCombo[kIdxSeven];
  l0Status.max_axis_pnt = parasCombo[kIdxEight];
  l0Status.max_axis_pnt = min(l0Status.max_axis_pnt, l0Status.max_axis_num);
}

void SetResFactors(L0Factors &resFactors, const L0Status &l0Status)
{
  resFactors.final_ml0 = l0Status.final_ml0;
  resFactors.final_kl0 = l0Status.final_kl0;
  resFactors.final_nl0 = l0Status.final_nl0;
  resFactors.final_load_size = l0Status.final_load_size;
  resFactors.final_l0c_use = l0Status.final_l0c_use;
  resFactors.final_mte1Loop = l0Status.final_mte1Loop;
  resFactors.final_mul = l0Status.final_mul;
}

void GetL0FactorsCand(L0Factors &resFactors, const CoreStatus &coreStatus, L0Status &l0Status, int32_t *parasCombo) {
  GetL0StatusFromParasCombo(l0Status, parasCombo);
  int32_t majorDim = coreStatus.m;
  int32_t minorDim = coreStatus.n;
  int32_t majorDimK = l0Status.max_mk;
  int32_t minorDimK = l0Status.max_nk;
  if (l0Status.max_axis_idx != 0) {
    majorDim = coreStatus.n;
    minorDim = coreStatus.m;
    majorDimK = l0Status.max_nk;
    minorDimK = l0Status.max_mk;
  }
  int32_t majorDimFactors[kCandidateLen] = {0};
  GetTwoFactors(majorDimFactors, l0Status.max_axis_pnt, majorDim, l0Status.max_axis_num);
  for (auto &majorDimFactor: majorDimFactors) {
    if (majorDimFactor == 0) {
      continue;
    }
    int32_t minorFactorMax = min(l0Status.max_mn / majorDimFactor, minorDimK);
    int32_t minorDimFactors[kCandidateLen] = {0};
    GetTwoFactors(minorDimFactors, minorFactorMax, minorDim, minorFactorMax);
    for (auto &minorDimFactor: minorDimFactors) {
      if (minorDimFactor == 0) {
        continue;
      }
      int32_t k0Max = min(majorDimK / majorDimFactor, minorDimK / minorDimFactor);
      int32_t k0Factors[kCandidateLen] = {0};
      GetTwoFactors(k0Factors, k0Max, coreStatus.k, k0Max);
      for (auto &k0: k0Factors) {
        GetFinalMkn(l0Status, coreStatus, k0, majorDimFactor, minorDimFactor);
      }
    }
  }
  SetResFactors(resFactors, l0Status);
}

void GetL0Factors(const string &op_type, const CoreStatus &coreStatus, const int32_t &blockValue,
                  L0Status &l0Status)
{
  // get m_l0, n_l0, k_l0 factor when singlecore m, n, k is know
  // m_l0, n_l0, k_l0 is a factor of single core m, n, k

  int32_t dbAOnBOnCOnIdx = 0;
  int32_t dbAOnBOnCOffIdx = 1;
  L0Factors resFactors[kL0ParasComboLen];
  for (int32_t i = 0; i < kL0ParasComboLen; ++i) {
    MKNParasCombo mknParasCombo = GetParasCombo(i, blockValue);
    GetL0FactorsCand(resFactors[i], coreStatus, l0Status, mknParasCombo.parasCombo);
  }

  // check both L0C utilization and loadsize to control LOC LOA LOB DB
  int32_t m0L0cDbOn = resFactors[dbAOnBOnCOnIdx].final_ml0;
  int32_t k0L0cDbOn = resFactors[dbAOnBOnCOnIdx].final_kl0;
  int32_t n0L0cDbOn = resFactors[dbAOnBOnCOnIdx].final_nl0;
  int32_t loadSizeL0cDbOn = resFactors[dbAOnBOnCOnIdx].final_load_size;
  float l0cUseL0cDbOn = resFactors[dbAOnBOnCOnIdx].final_l0c_use;

  int32_t m0L0cDbOff = resFactors[dbAOnBOnCOffIdx].final_ml0;
  int32_t k0L0cDbOff = resFactors[dbAOnBOnCOffIdx].final_kl0;
  int32_t n0L0cDbOff = resFactors[dbAOnBOnCOffIdx].final_nl0;
  int32_t loadSizeL0cDbOff = resFactors[dbAOnBOnCOffIdx].final_load_size;
  float l0cUseL0cDbOff = resFactors[dbAOnBOnCOffIdx].final_l0c_use;

  if ((l0cUseL0cDbOff > l0cUseL0cDbOn) || (loadSizeL0cDbOff < loadSizeL0cDbOn)) {
    int64_t dbL0aL0cDbOff = kDbOn;
    int64_t dbL0bL0cDbOff = kDbOn;
    l0Status.db_l0c = kDbOff;
    l0Status.db_l0a = dbL0aL0cDbOff;
    l0Status.db_l0b = dbL0bL0cDbOff;
    l0Status.m_l0 = m0L0cDbOff;
    l0Status.k_l0 = k0L0cDbOff;
    l0Status.n_l0 = n0L0cDbOff;
  } else {
    int64_t dbL0aL0cDbOn = kDbOn;
    int64_t dbL0bL0cDbOn = kDbOn;
    l0Status.db_l0c = kDbOn;
    l0Status.db_l0a = dbL0aL0cDbOn;
    l0Status.db_l0b = dbL0bL0cDbOn;
    l0Status.m_l0 = m0L0cDbOn;
    l0Status.k_l0 = k0L0cDbOn;
    l0Status.n_l0 = n0L0cDbOn;
  }
  l0Status.db_cub = kDbOn;
  OP_LOGD(op_type.c_str(), "tiling m_l0:%d, n_l0:%d, k_l0:%d", l0Status.m_l0, l0Status.n_l0, l0Status.k_l0);
  OP_LOGD(op_type.c_str(), "tiling db_l0a:%d, db_l0b:%d, db_l0c:%d", l0Status.db_l0a, l0Status.db_l0b,
          l0Status.db_l0c);
  OP_LOGD(op_type.c_str(), "tiling db_cub:%d", l0Status.db_cub);
}

int32_t GetL1Size(const L1Status &l1Status, const L0Status &l0Status) {
  int32_t curL1Size =
    l1Status.m_al1 * l0Status.m_l0 * kBlockSize * l1Status.kal1_16 * kBlockSize * l1Status.db_al1 * kFp16Bytes +
      l1Status.n_bl1 * l0Status.n_l0 * kBlockSize * l1Status.kbl1_16 * kBlockSize * l1Status.db_bl1 * kFp16Bytes;
  return curL1Size;
}

void L1StatusBothFullLoad(const CoreStatus &coreStatus, const L0Status &l0Status, L1Status &l1Status,
                          int32_t res[][7])
{
  int32_t curL1Size = GetL1Size(l1Status, l0Status);
  if (curL1Size <= kL1Size) {
    l1Status.both_full_load = true;
    l1Status.load_size = coreStatus.m + coreStatus.n;
    res[kIdxZero][kIdxZero] = l1Status.kal1_16;
    res[kIdxZero][kIdxOne] = l1Status.m_al1;
    res[kIdxZero][kIdxTwo] = l1Status.db_al1;
    res[kIdxZero][kIdxThree] = l1Status.kbl1_16;
    res[kIdxZero][kIdxFour] = l1Status.n_bl1;
    res[kIdxZero][kIdxFive] = l1Status.db_bl1;
    res[kIdxZero][kIdxSix] = l1Status.load_size;
  }
}

void L1StatusAl1FullLoad(const CoreStatus &coreStatus, const L0Status &l0Status, L1Status &l1Status,
                         int32_t res[][7])
{
  int32_t curL1Size;
  int32_t mRepeat = coreStatus.m / l0Status.m_l0;
  int32_t nRepeat = coreStatus.n / l0Status.n_l0;
  curL1Size = GetL1Size(l1Status, l0Status);
  if (curL1Size <= kL1Size) {
    l1Status.al1_full_load = true;
    l1Status.al1_size = coreStatus.k * coreStatus.m * kBlockSize * kBlockSize * kFp16Bytes;
    l1Status.bl1_size = kL1Size - l1Status.al1_size;
    l1Status.db_bl1 = kDbOn;
    if (GetL1Size(l1Status, l0Status) > kL1Size) {
      l1Status.db_bl1 = kDbOff;
    }
    l1Status.kbl1_16 = min(
        l1Status.bl1_size / (l1Status.n_bl1 * l0Status.n_l0 * kBlockSize * l1Status.db_bl1 * kFp16Bytes * kBlockSize),
        coreStatus.k);
    l1Status.bl1_times = min(l1Status.kbl1_16 / l0Status.k_l0, l1Status.max_k_bl1);
    GetNearestFactor(l1Status.all_times, l1Status.bl1_times);
    l1Status.kbl1_16 = l1Status.bl1_times * l0Status.k_l0;
    if (l1Status.kbl1_16 == coreStatus.k) {
      l1Status.n_bl1 = min(l1Status.bl1_size / (l1Status.kbl1_16 * l0Status.n_l0 * kBlockSize * l1Status.db_bl1 *
                             kFp16Bytes * kBlockSize),
                           l1Status.max_n_bl1);
      GetNearestFactor(nRepeat, l1Status.n_bl1);
    }
    l1Status.load_size = coreStatus.m + (l1Status.kbl1_16 == coreStatus.k ? 1 : mRepeat) * coreStatus.n;
    res[kIdxOne][kIdxZero] = l1Status.kal1_16;
    res[kIdxOne][kIdxOne] = l1Status.m_al1;
    res[kIdxOne][kIdxTwo] = l1Status.db_al1;
    res[kIdxOne][kIdxThree] = l1Status.kbl1_16;
    res[kIdxOne][kIdxFour] = l1Status.n_bl1;
    res[kIdxOne][kIdxFive] = l1Status.db_bl1;
    res[kIdxOne][kIdxSix] = l1Status.load_size;
  }
}

void L1StatusBl1FullLoad(const CoreStatus &coreStatus, const L0Status &l0Status, L1Status &l1Status,
                         int32_t res[][7])
{
  int32_t curL1Size;
  int32_t mRepeat = coreStatus.m / l0Status.m_l0;
  int32_t nRepeat = coreStatus.n / l0Status.n_l0;
  curL1Size = GetL1Size(l1Status, l0Status);
  if (curL1Size <= kL1Size) {
    l1Status.bl1_full_load = true;
    l1Status.bl1_size = coreStatus.k * coreStatus.n * kBlockSize * kBlockSize * kFp16Bytes;
    l1Status.al1_size = kL1Size - l1Status.bl1_size;
    l1Status.db_al1 = kDbOn;
    if (GetL1Size(l1Status, l0Status) > kL1Size) {
      l1Status.db_al1 = kDbOff;
    }
    l1Status.kal1_16 = min(
        l1Status.al1_size / (l1Status.m_al1 * l0Status.m_l0 * kBlockSize * l1Status.db_al1 * kFp16Bytes * kBlockSize),
        coreStatus.k);
    l1Status.al1_times = min(l1Status.kal1_16 / l0Status.k_l0, l1Status.max_k_al1);
    GetNearestFactor(l1Status.all_times, l1Status.al1_times);
    l1Status.kal1_16 = l1Status.al1_times * l0Status.k_l0;
    if (l1Status.kal1_16 == coreStatus.k) {
      l1Status.m_al1 = min(l1Status.al1_size / (l1Status.kal1_16 * l0Status.m_l0 * kBlockSize * l1Status.db_al1 *
                             kFp16Bytes * kBlockSize),
                           l1Status.max_m_al1);
      GetNearestFactor(mRepeat, l1Status.m_al1);
    }
    l1Status.load_size =
      coreStatus.n +
        ((coreStatus.m == l1Status.m_al1 * l0Status.m_l0 && l1Status.kal1_16 == coreStatus.k) ? 1 : nRepeat) *
          coreStatus.m;
    res[kIdxTwo][kIdxZero] = l1Status.kal1_16;
    res[kIdxTwo][kIdxOne] = l1Status.m_al1;
    res[kIdxTwo][kIdxTwo] = l1Status.db_al1;
    res[kIdxTwo][kIdxThree] = l1Status.kbl1_16;
    res[kIdxTwo][kIdxFour] = l1Status.n_bl1;
    res[kIdxTwo][kIdxFive] = l1Status.db_bl1;
    res[kIdxTwo][kIdxSix] = l1Status.load_size;
  }
}

void NeitherFullLoadDb(const CoreStatus &coreStatus, const L0Status &l0Status, L1Status &l1Status,
                       const int32_t &kbl1Db)
{
  int32_t tmpKbl116 = l1Status.kbl1_16;
  l1Status.kbl1_16 = kbl1Db;
  if (GetL1Size(l1Status, l0Status) > kL1Size) {
    l1Status.db_bl1 = kDbOff;
    if (GetL1Size(l1Status, l0Status) > kL1Size) {
      l1Status.db_al1 = kDbOff;
    }
  }
  l1Status.kbl1_16 = coreStatus.k;
  bool bothDoubleBuffer = coreStatus.m != l0Status.m_l0 && coreStatus.k > l0Status.k_l0 &&
    GetL1Size(l1Status, l0Status) > kL1Size;
  l1Status.kbl1_16 = tmpKbl116;
  if (bothDoubleBuffer) {
    l1Status.db_al1 = kDbOn;
    l1Status.db_bl1 = kDbOn;
    if (GetL1Size(l1Status, l0Status) > kL1Size) {
      l1Status.db_bl1 = kDbOff;
      if (GetL1Size(l1Status, l0Status) > kL1Size) {
        l1Status.db_al1 = kDbOff;
      }
    }
  }
}

void NeitherFullLoadMN(const CoreStatus &coreStatus, const L0Status &l0Status, L1Status &l1Status,
                       const BatchmatmulRunParas &params)
{
  int32_t mRepeat = coreStatus.m / l0Status.m_l0;
  int32_t nRepeat = coreStatus.n / l0Status.n_l0;
  if (l0Status.k_l0 == coreStatus.k) {
    if (params.m_mapped > params.n_mapped) {
      l1Status.bl1_size = coreStatus.k * l0Status.n_l0 * kBlockSize * kBlockSize * l1Status.db_bl1 * kFp16Bytes;
      l1Status.al1_size = kL1Size - l1Status.bl1_size;
      l1Status.m_al1 = min(l1Status.al1_size / (l1Status.kal1_16 * l0Status.m_l0 * kBlockSize * l1Status.db_al1 *
                             kFp16Bytes * kBlockSize),
                           l1Status.max_m_al1);
      GetNearestFactor(mRepeat, l1Status.m_al1);
      l1Status.al1_size =
        l1Status.kal1_16 * l1Status.m_al1 * l0Status.m_l0 * kBlockSize * kBlockSize * l1Status.db_al1 * kFp16Bytes;
      l1Status.bl1_size = kL1Size - l1Status.al1_size;
      l1Status.n_bl1 = min(l1Status.bl1_size / (l1Status.kbl1_16 * l0Status.n_l0 * kBlockSize * l1Status.db_bl1 *
                             kFp16Bytes * kBlockSize),
                           l1Status.max_n_bl1);
      GetNearestFactor(nRepeat, l1Status.n_bl1);
    } else {
      l1Status.al1_size = coreStatus.k * l0Status.m_l0 * kBlockSize * kBlockSize * l1Status.db_al1 * kFp16Bytes;
      l1Status.bl1_size = kL1Size - l1Status.al1_size;
      l1Status.n_bl1 = min(l1Status.bl1_size / (l1Status.kbl1_16 * l0Status.n_l0 * kBlockSize * l1Status.db_bl1 *
                             kFp16Bytes * kBlockSize),
                           l1Status.max_n_bl1);
      GetNearestFactor(nRepeat, l1Status.n_bl1);
      l1Status.bl1_size = coreStatus.k * l0Status.n_l0 * kBlockSize * kBlockSize * l1Status.db_bl1 * kFp16Bytes;
      l1Status.al1_size = kL1Size - l1Status.bl1_size;
      l1Status.m_al1 = min(l1Status.al1_size / (l1Status.kal1_16 * l0Status.m_l0 * kBlockSize * l1Status.db_al1 *
                             kFp16Bytes * kBlockSize),
                           l1Status.max_m_al1);
      GetNearestFactor(mRepeat, l1Status.m_al1);
    }
  }
}

void NeitherFullLoadKforNZ(const CoreStatus &coreStatus, const L0Status &l0Status, L1Status &l1Status)
{
  l1Status.kbl1_16 = coreStatus.k;
  if (GetL1Size(l1Status, l0Status) <= kL1Size) {
    l1Status.bl1_size = coreStatus.k * l0Status.n_l0 * kBlockSize * kBlockSize * l1Status.db_bl1 * kFp16Bytes;
    l1Status.al1_size = kL1Size - l1Status.bl1_size;
    l1Status.kal1_16 = min(
        l1Status.al1_size / (l1Status.m_al1 * l0Status.m_l0 * kBlockSize * l1Status.db_al1 * kFp16Bytes * kBlockSize),
        coreStatus.k);
    l1Status.al1_times = min(l1Status.kal1_16 / l0Status.k_l0, l1Status.max_k_al1);
    GetNearestFactor(l1Status.all_times, l1Status.al1_times);
    l1Status.kal1_16 = l1Status.al1_times * l0Status.k_l0;
  } else {
    int32_t perK = min(kL1Size /
                         (l0Status.m_l0 * kBlockSize * kBlockSize * l1Status.db_al1 * kFp16Bytes +
                           kBlockSize * l0Status.n_l0 * kBlockSize * l1Status.db_bl1 * kFp16Bytes) /
                         l0Status.k_l0 * l0Status.k_l0,
                       coreStatus.k);
    int32_t perTimes = min(perK / l0Status.k_l0, max(l1Status.max_k_al1, l1Status.max_k_bl1));
    GetNearestFactor(l1Status.all_times, perTimes);
    perK = perTimes * l0Status.k_l0;
    l1Status.kal1_16 = perK;
    l1Status.kbl1_16 = perK;
  }
}

void NeitherFullLoadKforND(const CoreStatus &coreStatus, const L0Status &l0Status, L1Status &l1Status,
                           const int &kmax_axis)
{
  if (kmax_axis == kNumOne) {
    // first get k_al1, second get k_bl1
    l1Status.kbl1_16 = l0Status.k_l0;
    l1Status.bl1_size = l1Status.kbl1_16 * l0Status.n_l0 * kBlockSize * kBlockSize * l1Status.db_bl1 * kFp16Bytes;
    l1Status.al1_size = kL1Size - l1Status.bl1_size;
    l1Status.kal1_16 = min(
        l1Status.al1_size / (l1Status.m_al1 * l0Status.m_l0 * kBlockSize * l1Status.db_al1 * kFp16Bytes * kBlockSize),
        coreStatus.k);
    l1Status.al1_times = l1Status.kal1_16 / l0Status.k_l0;
    GetNearestFactor(l1Status.all_times, l1Status.al1_times);
    l1Status.kal1_16 = l1Status.al1_times * l0Status.k_l0;
    l1Status.al1_size = l1Status.kal1_16 * l0Status.m_l0 * kBlockSize * kBlockSize * l1Status.db_al1 * kFp16Bytes;
    l1Status.bl1_size = kL1Size - l1Status.al1_size;
    l1Status.kbl1_16 = min(
        l1Status.bl1_size / (l1Status.n_bl1 * l0Status.n_l0 * kBlockSize * l1Status.db_bl1 * kFp16Bytes * kBlockSize),
        coreStatus.k);
    l1Status.bl1_times = min(l1Status.kbl1_16 / l0Status.k_l0, l1Status.max_k_bl1);
    GetNearestFactor(l1Status.all_times, l1Status.bl1_times);
    l1Status.kbl1_16 = l1Status.bl1_times * l0Status.k_l0;
  } else if (kmax_axis == kNumTwo) {
    // first get k_bl1, second get k_al1
    l1Status.kal1_16 = l0Status.k_l0;
    l1Status.al1_size = l1Status.kal1_16 * l0Status.m_l0 * kBlockSize * kBlockSize * l1Status.db_al1 * kFp16Bytes;
    l1Status.bl1_size = kL1Size - l1Status.al1_size;
    l1Status.kbl1_16 = min(
        l1Status.bl1_size / (l1Status.n_bl1 * l0Status.n_l0 * kBlockSize * l1Status.db_bl1 * kFp16Bytes * kBlockSize),
        coreStatus.k);
    l1Status.bl1_times = l1Status.kbl1_16 / l0Status.k_l0;
    GetNearestFactor(l1Status.all_times, l1Status.bl1_times);
    l1Status.kbl1_16 = l1Status.bl1_times * l0Status.k_l0;
    l1Status.bl1_size = l1Status.kbl1_16 * l0Status.n_l0 * kBlockSize * kBlockSize * l1Status.db_bl1 * kFp16Bytes;
    l1Status.al1_size = kL1Size - l1Status.bl1_size;
    l1Status.kal1_16 = min(
        l1Status.al1_size / (l1Status.m_al1 * l0Status.m_l0 * kBlockSize * l1Status.db_al1 * kFp16Bytes * kBlockSize),
        coreStatus.k);
    l1Status.al1_times = min(l1Status.kal1_16 / l0Status.k_l0, l1Status.max_k_al1);
    GetNearestFactor(l1Status.all_times, l1Status.al1_times);
    l1Status.kal1_16 = l1Status.al1_times * l0Status.k_l0;
  }
}

void NeitherFullLoadK(const CoreStatus &coreStatus, const L0Status &l0Status, L1Status &l1Status,
                      const BatchmatmulCompileParas &params)
{
  // 1 -> let k_al1 bigger, 2 -> let k_bl1 bigger, 0 -> no matter
  int kmax_axis = kNumZero;
  if (!params.trans_a_flag && !params.trans_b_flag) {
    kmax_axis = kNumOne;
  } else if (params.trans_a_flag && params.trans_b_flag) {
    kmax_axis = kNumTwo;
  } else if (!params.trans_a_flag && params.trans_b_flag) {
    kmax_axis = l0Status.m_l0 > l0Status.n_l0 ? kNumOne : kNumTwo;
  }

  if (params.nd_flag && kmax_axis != kNumZero) {
    NeitherFullLoadKforND(coreStatus, l0Status, l1Status, kmax_axis);
  } else {
    NeitherFullLoadKforNZ(coreStatus, l0Status, l1Status);
  }
}

void L1StatusNeitherFullLoad(const CoreStatus &coreStatus, const BatchmatmulParas &params,
                             const L0Status &l0Status, L1Status &l1Status, int32_t res[][7])
{
  int32_t mRepeat = coreStatus.m / l0Status.m_l0;
  int32_t nRepeat = coreStatus.n / l0Status.n_l0;
  int32_t kBl1Db = (coreStatus.m == l0Status.m_l0) ? l0Status.k_l0 : coreStatus.k;
  NeitherFullLoadDb(coreStatus, l0Status, l1Status, kBl1Db);
  NeitherFullLoadMN(coreStatus, l0Status, l1Status, *(params.run_params));
  NeitherFullLoadK(coreStatus, l0Status, l1Status, *(params.compile_params));
  // k_al1 and k_bl1 must be a factor of each other
  if (l1Status.kal1_16 > l1Status.kbl1_16 && l1Status.kal1_16 % l1Status.kbl1_16 != 0) {
    while (l1Status.kal1_16 % l1Status.kbl1_16 != 0 || coreStatus.k % l1Status.kal1_16 != 0) {
      l1Status.kal1_16 -= 1;
    }
  } else if (l1Status.kal1_16 < l1Status.kbl1_16 && l1Status.kbl1_16 % l1Status.kal1_16 != 0) {
    while (l1Status.kbl1_16 % l1Status.kal1_16 != 0 || coreStatus.k % l1Status.kbl1_16 != 0) {
      l1Status.kbl1_16 -= 1;
    }
  }
  l1Status.load_size =
    ((coreStatus.m == l1Status.m_al1 * l0Status.m_l0 && l1Status.kal1_16 == coreStatus.k) ? 1 : nRepeat) *
      coreStatus.m + (l1Status.kbl1_16 == coreStatus.k ? 1 : mRepeat) * coreStatus.n;
  res[kIdxThree][kIdxZero] = l1Status.kal1_16;
  res[kIdxThree][kIdxOne] = l1Status.m_al1;
  res[kIdxThree][kIdxTwo] = l1Status.db_al1;
  res[kIdxThree][kIdxThree] = l1Status.kbl1_16;
  res[kIdxThree][kIdxFour] = l1Status.n_bl1;
  res[kIdxThree][kIdxFive] = l1Status.db_bl1;
  res[kIdxThree][kIdxSix] = l1Status.load_size;
}

void GetL1Factors(const string &op_type, const BatchmatmulParas &params, const CoreStatus &coreStatus,
                  const L0Status &l0Status, L1Status &l1Status)
{
  // get m_al1, n_bl1, kal1_16, kbl1_16 factors when L0, singlecore factor is know
  // get al1, bl1 double buffer factors

  int32_t mte1Loop = 50 / ((l0Status.n_l0 == 1 ? 1 : l0Status.k_l0) + (l0Status.k_l0 == 1 ? 1 : l0Status.m_l0));
  int32_t res[4][7] = {0};
  l1Status.all_times = coreStatus.k / l0Status.k_l0;
  l1Status.max_m_al1 = (coreStatus.m + l0Status.m_l0 - 1) / l0Status.m_l0;
  l1Status.max_n_bl1 = (coreStatus.n + l0Status.n_l0 - 1) / l0Status.n_l0;
  l1Status.max_k_al1 =
    max(mte1Loop, ((kMinMte1Load + l0Status.m_l0 - 1) / l0Status.m_l0 + l0Status.k_l0 - 1) / l0Status.k_l0);
  l1Status.max_k_bl1 =
    max(mte1Loop, ((kMinMte1Load + l0Status.n_l0 - 1) / l0Status.n_l0 + l0Status.k_l0 - 1) / l0Status.k_l0);
  // both AL1 and Bl1 full load
  int32_t both_full_load_factors[kL1FactorsLen] =
      {coreStatus.k, coreStatus.k, l1Status.max_m_al1, l1Status.max_n_bl1, kDbOff, kDbOff};
  l1Status.SetStatus(both_full_load_factors);
  L1StatusBothFullLoad(coreStatus, l0Status, l1Status, res);
  // only AL1 full load
  int32_t al1_full_load_factors[kL1FactorsLen] = {coreStatus.k, l0Status.k_l0, l1Status.max_m_al1, 1, kDbOff, kDbOff};
  l1Status.SetStatus(al1_full_load_factors);
  L1StatusAl1FullLoad(coreStatus, l0Status, l1Status, res);
  // only BL1 full load
  int32_t bl1_full_load_factors[kL1FactorsLen] = {l0Status.k_l0, coreStatus.k, 1, l1Status.max_n_bl1, kDbOff, kDbOff};
  l1Status.SetStatus(bl1_full_load_factors);
  L1StatusBl1FullLoad(coreStatus, l0Status, l1Status, res);
  // neither AL1 nor Bl1 full load
  int32_t neither_full_load_factors[kL1FactorsLen] = {l0Status.k_l0, l0Status.k_l0, 1, 1, kDbOn, kDbOn};
  l1Status.SetStatus(neither_full_load_factors);
  L1StatusNeitherFullLoad(coreStatus, params, l0Status, l1Status, res);
  // choose the final factors
  int32_t *tmpFactors = res[kIdxThree];
  int32_t tmpLoadSize = tmpFactors[kIdxSix];
  if (l1Status.al1_full_load &&
    (res[kIdxOne][kIdxSix] < tmpLoadSize ||
      (res[kIdxOne][kIdxSix] == tmpLoadSize &&
        res[kIdxOne][kIdxOne] + res[kIdxOne][kIdxFour] > tmpFactors[kIdxOne] + tmpFactors[kIdxFour]))) {
    tmpFactors = res[kIdxOne];
    tmpLoadSize = tmpFactors[kIdxSix];
  }
  if (l1Status.bl1_full_load &&
    (res[kIdxTwo][kIdxSix] < tmpLoadSize ||
      (res[kIdxTwo][kIdxSix] == tmpLoadSize &&
        res[kIdxTwo][kIdxOne] + res[kIdxTwo][kIdxFour] > tmpFactors[kIdxOne] + tmpFactors[kIdxFour]))) {
    tmpFactors = res[kIdxTwo];
    tmpLoadSize = tmpFactors[kIdxSix];
  }
  if (l1Status.both_full_load &&
    (res[kIdxZero][kIdxSix] < tmpLoadSize ||
      (res[kIdxZero][kIdxSix] == tmpLoadSize &&
        res[kIdxZero][kIdxOne] + res[kIdxZero][kIdxFour] > tmpFactors[kIdxOne] + tmpFactors[kIdxFour]))) {
    tmpFactors = res[kIdxZero];
  }
  int32_t res_l1_factors[kL1FactorsLen] = {tmpFactors[kIdxZero], tmpFactors[kIdxThree], tmpFactors[kIdxOne],
                                           tmpFactors[kIdxFour], tmpFactors[kIdxTwo], tmpFactors[kIdxFive]};
  l1Status.SetStatus(res_l1_factors);
  OP_LOGD(op_type.c_str(), "tiling kal1_16:%d, kbl1_16:%d, k_l0:%d", l1Status.kal1_16, l1Status.kbl1_16,
          l0Status.k_l0);
  OP_LOGD(op_type.c_str(), "tiling m_al1:%d, n_bl1:%d", l1Status.m_al1, l1Status.n_bl1);
  OP_LOGD(op_type.c_str(), "tiling db_al1:%d, db_bl1:%d", l1Status.db_al1, l1Status.db_bl1);
}

void UpdateUbReuseFlagAndRestSize(const CoreStatus& coreStatus, const L1Status& l1Status, const L0Status& l0Status,
                                  UbStatus& ubStatus) {
  // Initialization
  ubStatus.aub_multi_flag = kNumZero;
  ubStatus.bub_multi_flag = kNumZero;
  ubStatus.cub_reuse_aub_flag = l1Status.al1_full_load;
  ubStatus.cub_reuse_bub_flag = l1Status.bl1_full_load;
  // Get AUB Full Load Flag
  if (l1Status.kal1_16 == ubStatus.k_aub && l1Status.m_al1 * l0Status.m_l0 == ubStatus.m_aub) {
    ubStatus.aub_multi_flag = kAttachFlagOne;
  }
  bool aub_full_load = ubStatus.aub_multi_flag == kAttachFlagOne;
  // remove invalid reused scenario(preload is effected)
  if (l1Status.al1_full_load && aub_full_load) {
    ubStatus.cub_reuse_aub_flag = false;
  }
  // Get BUB Full Load Flag
  if (l1Status.kbl1_16 == ubStatus.k_bub && l1Status.n_bl1 * l0Status.n_l0 == ubStatus.n_bub) {
    ubStatus.bub_multi_flag = kAttachFlagOne;
  }
  // Check how many ub spaces are left.
  bool bub_full_load = ubStatus.bub_multi_flag == kAttachFlagOne;
  // remove invalid reused scenario(preload is effected)
  if (l1Status.bl1_full_load && bub_full_load) {
    ubStatus.cub_reuse_bub_flag = false;
  }
  // Update UB rest Size ---> available space for AUb and BUb
  if (!ubStatus.cub_reuse_aub_flag && !ubStatus.cub_reuse_bub_flag) {
    ubStatus.ub_rest_size = kUbFp16Size - ubStatus.min_dma_size;
  } else {
    ubStatus.ub_rest_size = kUbFp16Size;
  }
}

bool CheckABUbSize(const PreUbTiling& pre_ub_tiling, const BatchmatmulCompileParas& params,
                   const CoreStatus& coreStatus, SingleCoreStatus& singleCoreStatus,
                   const bool unsafe_ubStatus = false) {
  const int32_t& k_aub = pre_ub_tiling.k_aub;
  const int32_t& m_aub = pre_ub_tiling.m_aub;
  const int32_t& k_bub = pre_ub_tiling.k_bub;
  const int32_t& n_bub = pre_ub_tiling.n_bub;
  const L0Status& l0Status = singleCoreStatus.l0Status;
  const L1Status& l1Status = singleCoreStatus.l1Status;
  UbStatus& ubStatus = singleCoreStatus.ubStatus;
  UbStatus tmp_ubStatus = ubStatus;
  tmp_ubStatus.aub_size = k_aub * kBlockSize * m_aub * kBlockSize * ubStatus.db_aub * (1 + params.aub_double_num);
  tmp_ubStatus.bub_size = k_bub * kBlockSize * n_bub * kBlockSize * ubStatus.db_bub * (1 + params.bub_double_num);
  tmp_ubStatus.k_aub = k_aub;
  tmp_ubStatus.m_aub = m_aub;
  tmp_ubStatus.k_bub = k_bub;
  tmp_ubStatus.n_bub = n_bub;
  UpdateUbReuseFlagAndRestSize(coreStatus, l1Status, l0Status, tmp_ubStatus);
  if (!unsafe_ubStatus && !params.split_k_flag) {
    if (tmp_ubStatus.cub_reuse_aub_flag && !tmp_ubStatus.cub_reuse_bub_flag) {
      tmp_ubStatus.aub_size = max(tmp_ubStatus.aub_size, ubStatus.min_dma_size);
    } else if (!tmp_ubStatus.cub_reuse_aub_flag && tmp_ubStatus.cub_reuse_bub_flag) {
      tmp_ubStatus.bub_size = max(tmp_ubStatus.bub_size, ubStatus.min_dma_size);
    } else if (tmp_ubStatus.cub_reuse_aub_flag && tmp_ubStatus.cub_reuse_bub_flag) {
      // AUB BUB and CUB used the same space
      tmp_ubStatus.aub_size = max(tmp_ubStatus.aub_size, max(tmp_ubStatus.bub_size, ubStatus.min_dma_size));
      tmp_ubStatus.bub_size = 0;
    }
    return (tmp_ubStatus.aub_size + tmp_ubStatus.bub_size) <= tmp_ubStatus.ub_rest_size;
  }
  // Dont know if reused can be enabled so do not reused.
  return (tmp_ubStatus.aub_size + tmp_ubStatus.bub_size) <= tmp_ubStatus.safe_ub_rest_size;
}

void GetABUbMax(const int32_t &k_aub, const int32_t &m_aub, const int32_t &k_bub, const int32_t &n_bub,
                const BatchmatmulCompileParas &params, UbStatus &ubStatus, const int max_num)
{
  // max_num: 0->k_aub, 1->k_bub, 2->m_aub, 3->n_bub
  ubStatus.aub_size = k_aub * kBlockSize * m_aub * kBlockSize * ubStatus.db_aub * (1 + params.aub_double_num);
  ubStatus.bub_size = k_bub * kBlockSize * n_bub * kBlockSize * ubStatus.db_bub * (1 + params.bub_double_num);
  if (max_num == kNumZero) {
    ubStatus.k_aub = (ubStatus.safe_ub_rest_size - ubStatus.bub_size) /
      (kBlockSize * m_aub * kBlockSize * ubStatus.db_aub * (1 + params.aub_double_num));
  } else if (max_num == kNumOne) {
    ubStatus.k_bub = (ubStatus.safe_ub_rest_size - ubStatus.aub_size) /
      (kBlockSize * n_bub * kBlockSize * ubStatus.db_bub * (1 + params.bub_double_num));
  } else if (max_num == kNumTwo) {
    ubStatus.m_aub = (ubStatus.safe_ub_rest_size - ubStatus.bub_size) /
      (k_aub * kBlockSize * kBlockSize * ubStatus.db_aub * (1 + params.aub_double_num));
  } else if (max_num == kNumThree) {
    ubStatus.n_bub = (ubStatus.safe_ub_rest_size - ubStatus.aub_size) /
      (k_bub * kBlockSize * kBlockSize * ubStatus.db_bub * (1 + params.bub_double_num));
  }
}

void UpdateAUbCandidateStatusPhase1(const CoreStatus& coreStatus, const BatchmatmulCompileParas& params,
                                    const AUbStatusCondition& ub_condition, SingleCoreStatus& singleCoreStatus) {
  // Get All AUB Candidate Tiling result.
  const L0Status& l0Status = singleCoreStatus.l0Status;
  const L1Status& l1Status = singleCoreStatus.l1Status;
  UbStatus& ubStatus = singleCoreStatus.ubStatus;
  int32_t al1_m = l1Status.m_al1 * l0Status.m_l0;

  if (ub_condition.condition_m2_k2) {
    ubStatus.aub_results[ubStatus.aub_cnt][0] = coreStatus.k;
    ubStatus.aub_results[ubStatus.aub_cnt][1] = coreStatus.m;
    ubStatus.aub_cnt += 1;
  }
  if (ub_condition.condition_ml1_kl1) {
    ubStatus.aub_results[ubStatus.aub_cnt][0] = l1Status.kal1_16;
    ubStatus.aub_results[ubStatus.aub_cnt][1] = al1_m;
    ubStatus.aub_cnt += 1;
  }
  if (ub_condition.condition_ml1_kl0) {
    ubStatus.aub_results[ubStatus.aub_cnt][0] = l0Status.k_l0;
    ubStatus.aub_results[ubStatus.aub_cnt][1] = al1_m;
    ubStatus.aub_cnt += 1;
  }
  if (ub_condition.condition_ml1_k0) {
    ubStatus.k_aub = 1;
    ubStatus.m_aub = al1_m;
    GetABUbMax(1, al1_m, ubStatus.k_bub, ubStatus.n_bub, params, ubStatus, kNumZero);
    GetNearestFactor(l1Status.kal1_16, ubStatus.k_aub);
    ubStatus.aub_results[ubStatus.aub_cnt][0] = ubStatus.k_aub;
    ubStatus.aub_results[ubStatus.aub_cnt][1] = ubStatus.m_aub;
    ubStatus.aub_cnt += 1;
  }
  if (ub_condition.condition_ml0_kl1) {
    ubStatus.aub_results[ubStatus.aub_cnt][0] = l1Status.kal1_16;
    ubStatus.aub_results[ubStatus.aub_cnt][1] = l0Status.m_l0;
    ubStatus.aub_cnt += 1;
  }
}

void UpdateAUbCandidateStatusPhase2(const CoreStatus& coreStatus, const BatchmatmulCompileParas& params,
                                    const AUbStatusCondition& ub_condition, SingleCoreStatus& singleCoreStatus) {
  // Get All AUB Candidate Tiling result.
  const L0Status& l0Status = singleCoreStatus.l0Status;
  const L1Status& l1Status = singleCoreStatus.l1Status;
  UbStatus& ubStatus = singleCoreStatus.ubStatus;
  int32_t al1_m = l1Status.m_al1 * l0Status.m_l0;
  if (ub_condition.condition_m0_kl1) {
    ubStatus.k_aub = l1Status.kal1_16;
    ubStatus.m_aub = 1;
    GetABUbMax(l1Status.kal1_16, 1, ubStatus.k_bub, ubStatus.n_bub, params, ubStatus, kNumTwo);
    GetNearestFactor(al1_m, ubStatus.m_aub);
    ubStatus.aub_results[ubStatus.aub_cnt][0] = ubStatus.k_aub;
    ubStatus.aub_results[ubStatus.aub_cnt][1] = ubStatus.m_aub;
    ubStatus.aub_cnt += 1;
  }
  if (ub_condition.condition_ml0_kl0) {
    ubStatus.aub_results[ubStatus.aub_cnt][0] = l0Status.k_l0;
    ubStatus.aub_results[ubStatus.aub_cnt][1] = l0Status.m_l0;
    ubStatus.aub_cnt += 1;
  }
  if (ub_condition.condition_ml0_k0) {
    ubStatus.aub_results[ubStatus.aub_cnt][0] = 1;
    ubStatus.aub_results[ubStatus.aub_cnt][1] = l0Status.m_l0;
    ubStatus.aub_cnt += 1;
  }
  if (ub_condition.condition_m0_kl0) {
    ubStatus.aub_results[ubStatus.aub_cnt][0] = l0Status.k_l0;
    ubStatus.aub_results[ubStatus.aub_cnt][1] = 1;
    ubStatus.aub_cnt += 1;
  }
  if (ub_condition.condition_m0_k0) {
    ubStatus.aub_results[ubStatus.aub_cnt][0] = 1;
    ubStatus.aub_results[ubStatus.aub_cnt][1] = 1;
    ubStatus.aub_cnt += 1;
  }
}

void GetAUbFactors(const CoreStatus& coreStatus, const BatchmatmulCompileParas& params,
                   SingleCoreStatus& singleCoreStatus) {
  // Get All AUB Candidate Tiling result.
  const L0Status& l0Status = singleCoreStatus.l0Status;
  const L1Status& l1Status = singleCoreStatus.l1Status;
  UbStatus& ubStatus = singleCoreStatus.ubStatus;
  AUbStatusCondition ub_condition;
  PreUbTiling pre_ub_tiling;
  int32_t al1_m = l1Status.m_al1 * l0Status.m_l0;
  pre_ub_tiling.update_tiling(coreStatus.k, coreStatus.m, ubStatus.k_bub, ubStatus.n_bub);
  ub_condition.condition_m2_k2 =
      (!params.at_l1_flag && CheckABUbSize(pre_ub_tiling, params, coreStatus, singleCoreStatus));
  pre_ub_tiling.update_tiling(l1Status.kal1_16, al1_m, ubStatus.k_bub, ubStatus.n_bub);
  ub_condition.condition_ml1_kl1 = CheckABUbSize(pre_ub_tiling, params, coreStatus, singleCoreStatus);
  pre_ub_tiling.update_tiling(l0Status.k_l0, al1_m, ubStatus.k_bub, ubStatus.n_bub);
  ub_condition.condition_ml1_kl0 =
      (params.trans_a_flag && CheckABUbSize(pre_ub_tiling, params, coreStatus, singleCoreStatus));
  pre_ub_tiling.update_tiling(1, al1_m, ubStatus.k_bub, ubStatus.n_bub);
  ub_condition.condition_ml1_k0 =
      (params.trans_a_flag && CheckABUbSize(pre_ub_tiling, params, coreStatus, singleCoreStatus, true));
  pre_ub_tiling.update_tiling(l1Status.kal1_16, l0Status.m_l0, ubStatus.k_bub, ubStatus.n_bub);
  ub_condition.condition_ml0_kl1 =
      (!params.trans_a_flag && CheckABUbSize(pre_ub_tiling, params, coreStatus, singleCoreStatus));
  pre_ub_tiling.update_tiling(l1Status.kal1_16, 1, ubStatus.k_bub, ubStatus.n_bub);
  ub_condition.condition_m0_kl1 =
      (!params.trans_a_flag && CheckABUbSize(pre_ub_tiling, params, coreStatus, singleCoreStatus, true));
  pre_ub_tiling.update_tiling(l0Status.k_l0, l0Status.m_l0, ubStatus.k_bub, ubStatus.n_bub);
  ub_condition.condition_ml0_kl0 = CheckABUbSize(pre_ub_tiling, params, coreStatus, singleCoreStatus);
  pre_ub_tiling.update_tiling(1, l0Status.m_l0, ubStatus.k_bub, ubStatus.n_bub);
  ub_condition.condition_ml0_k0 =
      (params.trans_a_flag && CheckABUbSize(pre_ub_tiling, params, coreStatus, singleCoreStatus));
  pre_ub_tiling.update_tiling(l0Status.k_l0, 1, ubStatus.k_bub, ubStatus.n_bub);
  ub_condition.condition_m0_kl0 =
      (!params.trans_a_flag && CheckABUbSize(pre_ub_tiling, params, coreStatus, singleCoreStatus));
  pre_ub_tiling.update_tiling(1, 1, ubStatus.k_bub, ubStatus.n_bub);
  ub_condition.condition_m0_k0 = CheckABUbSize(pre_ub_tiling, params, coreStatus, singleCoreStatus);

  UpdateAUbCandidateStatusPhase1(coreStatus, params, ub_condition, singleCoreStatus);
  UpdateAUbCandidateStatusPhase2(coreStatus, params, ub_condition, singleCoreStatus);
}

void UpdateBUbCandidateStatusPhase1(const CoreStatus& coreStatus, const BatchmatmulCompileParas& params,
                                    const BUbStatusCondition& ub_condition, SingleCoreStatus& singleCoreStatus) {
  const L0Status& l0Status = singleCoreStatus.l0Status;
  const L1Status& l1Status = singleCoreStatus.l1Status;
  UbStatus& ubStatus = singleCoreStatus.ubStatus;
  int32_t bl1_n = l1Status.n_bl1 * l0Status.n_l0;
  if (ub_condition.condition_k2_n2) {
    ubStatus.bub_results[ubStatus.bub_cnt][0] = coreStatus.k;
    ubStatus.bub_results[ubStatus.bub_cnt][1] = coreStatus.n;
    ubStatus.bub_cnt += 1;
  }
  if (ub_condition.condition_kl1_nl1 && ubStatus.bub_cnt < kNumTwo) {
    ubStatus.bub_results[ubStatus.bub_cnt][0] = l1Status.kbl1_16;
    ubStatus.bub_results[ubStatus.bub_cnt][1] = bl1_n;
    ubStatus.bub_cnt += 1;
  }
  if (ub_condition.condition_kl0_nl1 && ubStatus.bub_cnt < kNumTwo) {
    ubStatus.bub_results[ubStatus.bub_cnt][0] = l0Status.k_l0;
    ubStatus.bub_results[ubStatus.bub_cnt][1] = bl1_n;
    ubStatus.bub_cnt += 1;
  }
  if (ub_condition.condition_k0_nl1 && ubStatus.bub_cnt < kNumTwo) {
    ubStatus.k_bub = 1;
    ubStatus.n_bub = bl1_n;
    GetABUbMax(ubStatus.k_aub, ubStatus.m_aub, 1, bl1_n, params, ubStatus, kNumOne);
    GetNearestFactor(l1Status.kbl1_16, ubStatus.k_bub);
    ubStatus.bub_results[ubStatus.bub_cnt][0] = ubStatus.k_bub;
    ubStatus.bub_results[ubStatus.bub_cnt][1] = ubStatus.n_bub;
    ubStatus.bub_cnt += 1;
  }
  if (ub_condition.condition_kl1_nl0 && ubStatus.bub_cnt < kNumTwo) {
    ubStatus.bub_results[ubStatus.bub_cnt][0] = l1Status.kbl1_16;
    ubStatus.bub_results[ubStatus.bub_cnt][1] = l0Status.n_l0;
    ubStatus.bub_cnt += 1;
  }
}

void UpdateBUbCandidateStatusPhase2(const CoreStatus& coreStatus, const BatchmatmulCompileParas& params,
                                    const BUbStatusCondition& ub_condition, SingleCoreStatus& singleCoreStatus) {
  const L0Status& l0Status = singleCoreStatus.l0Status;
  const L1Status& l1Status = singleCoreStatus.l1Status;
  UbStatus& ubStatus = singleCoreStatus.ubStatus;
  int32_t bl1_n = l1Status.n_bl1 * l0Status.n_l0;
  if (ub_condition.condition_kl1_n0 && ubStatus.bub_cnt < kNumTwo) {
    ubStatus.k_bub = l1Status.kbl1_16;
    ubStatus.n_bub = 1;
    GetABUbMax(ubStatus.k_aub, ubStatus.m_aub, l1Status.kbl1_16, 1, params, ubStatus, kNumThree);
    GetNearestFactor(bl1_n, ubStatus.n_bub);
    ubStatus.bub_results[ubStatus.bub_cnt][0] = ubStatus.k_bub;
    ubStatus.bub_results[ubStatus.bub_cnt][1] = ubStatus.n_bub;
    ubStatus.bub_cnt += 1;
  }
  if (ub_condition.condition_kl0_nl0 && ubStatus.bub_cnt < kNumTwo) {
    ubStatus.bub_results[ubStatus.bub_cnt][0] = l0Status.k_l0;
    ubStatus.bub_results[ubStatus.bub_cnt][1] = l0Status.n_l0;
    ubStatus.bub_cnt += 1;
  }
  if (ub_condition.condition_k0_nl0 && ubStatus.bub_cnt < kNumTwo) {
    ubStatus.bub_results[ubStatus.bub_cnt][0] = 1;
    ubStatus.bub_results[ubStatus.bub_cnt][1] = l0Status.n_l0;
    ubStatus.bub_cnt += 1;
  }
  if (ub_condition.condition_kl0_n0 && ubStatus.bub_cnt < kNumTwo) {
    ubStatus.bub_results[ubStatus.bub_cnt][0] = l0Status.k_l0;
    ubStatus.bub_results[ubStatus.bub_cnt][1] = 1;
    ubStatus.bub_cnt += 1;
  }
  if (ub_condition.condition_k0_n0 && ubStatus.bub_cnt < kNumTwo) {
    ubStatus.bub_results[ubStatus.bub_cnt][0] = 1;
    ubStatus.bub_results[ubStatus.bub_cnt][1] = 1;
    ubStatus.bub_cnt += 1;
  }
}

void GetBUbFactors(const CoreStatus& coreStatus, const BatchmatmulCompileParas& params,
                   SingleCoreStatus& singleCoreStatus)
{
  // Initialize the candidate array. Data will be overwritten so we can keep it dirty
  // Get All AUB Candidate Tiling result.
  const L0Status& l0Status = singleCoreStatus.l0Status;
  const L1Status& l1Status = singleCoreStatus.l1Status;
  UbStatus& ubStatus = singleCoreStatus.ubStatus;
  BUbStatusCondition ub_condition;
  PreUbTiling pre_ub_tiling;
  ubStatus.bub_cnt = 0;
  int32_t bl1_n = l1Status.n_bl1 * l0Status.n_l0;
  pre_ub_tiling.update_tiling(ubStatus.k_aub, ubStatus.m_aub, coreStatus.k, coreStatus.n);
  ub_condition.condition_k2_n2 =
      (!params.at_l1_flag && CheckABUbSize(pre_ub_tiling, params, coreStatus, singleCoreStatus));
  pre_ub_tiling.update_tiling(ubStatus.k_aub, ubStatus.m_aub, l1Status.kbl1_16, bl1_n);
  ub_condition.condition_kl1_nl1 = CheckABUbSize(pre_ub_tiling, params, coreStatus, singleCoreStatus);
  pre_ub_tiling.update_tiling(ubStatus.k_aub, ubStatus.m_aub, l0Status.k_l0, bl1_n);
  ub_condition.condition_kl0_nl1 =
      (!params.trans_b_flag && CheckABUbSize(pre_ub_tiling, params, coreStatus, singleCoreStatus));
  pre_ub_tiling.update_tiling(ubStatus.k_aub, ubStatus.m_aub, 1, bl1_n);
  ub_condition.condition_k0_nl1 =
      (!params.trans_b_flag && CheckABUbSize(pre_ub_tiling, params, coreStatus, singleCoreStatus, true));
  pre_ub_tiling.update_tiling(ubStatus.k_aub, ubStatus.m_aub, l1Status.kbl1_16, l0Status.n_l0);
  ub_condition.condition_kl1_nl0 =
      (params.trans_b_flag && CheckABUbSize(pre_ub_tiling, params, coreStatus, singleCoreStatus));
  pre_ub_tiling.update_tiling(ubStatus.k_aub, ubStatus.m_aub, l1Status.kbl1_16, 1);
  ub_condition.condition_kl1_n0 =
      (params.trans_b_flag && CheckABUbSize(pre_ub_tiling, params, coreStatus, singleCoreStatus, true));
  pre_ub_tiling.update_tiling(ubStatus.k_aub, ubStatus.m_aub, l0Status.k_l0, l0Status.n_l0);
  ub_condition.condition_kl0_nl0 = CheckABUbSize(pre_ub_tiling, params, coreStatus, singleCoreStatus);
  pre_ub_tiling.update_tiling(ubStatus.k_aub, ubStatus.m_aub, 1, l0Status.n_l0);
  ub_condition.condition_k0_nl0 =
      (!params.trans_b_flag && CheckABUbSize(pre_ub_tiling, params, coreStatus, singleCoreStatus));
  pre_ub_tiling.update_tiling(ubStatus.k_aub, ubStatus.m_aub, l0Status.k_l0, 1);
  ub_condition.condition_kl0_n0 =
      (params.trans_b_flag && CheckABUbSize(pre_ub_tiling, params, coreStatus, singleCoreStatus));
  pre_ub_tiling.update_tiling(ubStatus.k_aub, ubStatus.m_aub, 1, 1);
  ub_condition.condition_k0_n0 = CheckABUbSize(pre_ub_tiling, params, coreStatus, singleCoreStatus);

  UpdateBUbCandidateStatusPhase1(coreStatus, params, ub_condition, singleCoreStatus);
  UpdateBUbCandidateStatusPhase2(coreStatus, params, ub_condition, singleCoreStatus);
}

void GetABUbSize(const int32_t &k_aub, const int32_t &m_aub, const int32_t &k_bub, const int32_t &n_bub,
                 const BatchmatmulCompileParas &params, UbStatus &ubStatus) {
  ubStatus.aub_bank_size = k_aub * kBlockSize * m_aub * kBlockSize * ubStatus.db_aub * (1 + params.aub_double_num);
  ubStatus.bub_bank_size = k_bub * kBlockSize * n_bub * kBlockSize * ubStatus.db_bub * (1 + params.bub_double_num);
  ubStatus.aub_size = ubStatus.aub_bank_size;
  ubStatus.bub_size = ubStatus.bub_bank_size;
  ubStatus.a_align_value = 1;
  ubStatus.b_align_value = 1;
  ubStatus.a_bank_conflict = false;
  ubStatus.b_bank_conflict = false;
  if (params.trans_a_flag && m_aub % kBankConflictFactor == 0) {
    ubStatus.a_bank_conflict = true;
    ubStatus.a_align_value = (m_aub + 1) * kBlockSize;
    ubStatus.aub_bank_size += k_aub * kBlockSize * kBlockSize * ubStatus.db_aub;
    ubStatus.aub_align_bound += k_aub * kBlockSize * kBlockSize;
  } else if (!params.trans_a_flag && k_aub % kBankConflictFactor == 0) {
    ubStatus.a_bank_conflict = true;
    ubStatus.a_align_value = (k_aub + 1) * kBlockSize;
    ubStatus.aub_bank_size += kBlockSize * m_aub * kBlockSize * ubStatus.db_aub;
    ubStatus.aub_align_bound += kBlockSize * m_aub * kBlockSize;
  }
  if (params.trans_b_flag && k_bub % kBankConflictFactor == 0) {
    ubStatus.b_bank_conflict = true;
    ubStatus.b_align_value = (k_bub + 1) * kBlockSize;
    ubStatus.bub_bank_size += kBlockSize * n_bub * kBlockSize * ubStatus.db_bub;
    ubStatus.bub_align_bound += kBlockSize * n_bub * kBlockSize;
  } else if (!params.trans_b_flag && n_bub % kBankConflictFactor == 0) {
    ubStatus.b_bank_conflict = true;
    ubStatus.b_align_value = (n_bub + 1) * kBlockSize;
    ubStatus.bub_bank_size += k_bub * kBlockSize * kBlockSize * ubStatus.db_bub;
    ubStatus.bub_align_bound += k_bub * kBlockSize * kBlockSize;
  }
}

void GetABUbStorageSize(const UbStatus& ubStatus, int32_t storage_array[2][2]) {
  // After Reused, the storage size is the actual tensor storage size used.
  // storage_array[0] contains aub_storage_size and aub_bank_storage_size;
  // storage_array[1] contains bub_storage_size and bub_bank_storage_size;
  storage_array[0][0] = ubStatus.cub_reuse_aub_flag
                            ? max(ubStatus.aub_size, ubStatus.min_dma_size) + ubStatus.cub_aub_ratio * ubStatus.aub_size
                            : ubStatus.aub_size;
  storage_array[1][0] = ubStatus.cub_reuse_bub_flag
                            ? max(ubStatus.bub_size, ubStatus.min_dma_size) + ubStatus.cub_bub_ratio * ubStatus.bub_size
                            : ubStatus.bub_size;
  storage_array[0][1] = ubStatus.cub_reuse_aub_flag
                            ? max(ubStatus.aub_bank_size, ubStatus.min_dma_size) +
                                  (ubStatus.aub_bank_size - ubStatus.cub_aub_ratio * ubStatus.aub_size)
                            : ubStatus.aub_bank_size;
  storage_array[1][1] = ubStatus.cub_reuse_bub_flag
                            ? max(ubStatus.bub_bank_size, ubStatus.min_dma_size) +
                                  (ubStatus.bub_bank_size - ubStatus.cub_bub_ratio * ubStatus.bub_size)
                            : ubStatus.bub_bank_size;
}

int GetAllUbReusedAlignMode(const UbStatus& ubStatus, const int& aub_bank_storage_size,
                            const int& bub_bank_storage_size) {
  int align_mode = kNumZero;
  int32_t max_bank_storage_size = max(max(ubStatus.aub_bank_size, ubStatus.min_dma_size), ubStatus.bub_bank_size);
  // When K_dim is split, the nd_to_nz tensor cannot be reused.
  max_bank_storage_size = max_bank_storage_size +
                          (ubStatus.aub_bank_size - ubStatus.cub_aub_ratio * ubStatus.aub_size) +
                          (ubStatus.bub_bank_size - ubStatus.cub_bub_ratio * ubStatus.bub_size);
  if (max_bank_storage_size <= ubStatus.ub_rest_size) {
    align_mode = kNumThree;
  } else if (ubStatus.a_align_value != 1 && ubStatus.b_align_value != 1) {
    if (ubStatus.aub_bank_size > ubStatus.bub_bank_size) {
      // Aub_bank_storage_size > Bub_bank_storage_size and AUB_bank_storage_size exceeds UB buffer size
      align_mode = (bub_bank_storage_size <= ubStatus.ub_rest_size) ? kNumTwo : kNumZero;
    } else {
      // Bub_bank_storage_size > Aub_bank_storage_size and Bub_bank_storage_size exceeds UB buffer size
      align_mode = (aub_bank_storage_size <= ubStatus.ub_rest_size) ? kNumOne : kNumZero;
    }
  }
  return align_mode;
}

int GetAlignMode(const UbStatus& ubStatus) {
  int align_mode = kNumZero;
  // storage_array[0] contains aub_storage_size and aub_bank_storage_size;
  // storage_array[1] contains bub_storage_size and bub_bank_storage_size;
  int32_t storage_array[2][2] = {0};
  GetABUbStorageSize(ubStatus, storage_array);
  // Parse the storage info from storage array.
  int32_t aub_storage_size = storage_array[0][0];
  int32_t bub_storage_size = storage_array[1][0];
  int32_t aub_bank_storage_size = storage_array[0][1];
  int32_t bub_bank_storage_size = storage_array[1][1];
  // Process storage_size when all ub tensors are reused together.
  bool a_b_c_ub_reused_together = ubStatus.cub_reuse_aub_flag && ubStatus.cub_reuse_bub_flag;
  // There‘s 4 align_mode here, 0->neither align, 1->AUB align and BUB not, 2->AUB not and BUB align, 3-> all aligned
  if (a_b_c_ub_reused_together) {
    align_mode = GetAllUbReusedAlignMode(ubStatus, aub_bank_storage_size, bub_bank_storage_size);
  } else {
    if (aub_bank_storage_size + bub_bank_storage_size <= ubStatus.ub_rest_size) {
      align_mode = kNumThree;
    } else if (ubStatus.a_align_value != 1 && ubStatus.b_align_value != 1) {
      if (ubStatus.aub_bank_size > ubStatus.bub_bank_size) {
        align_mode = (aub_bank_storage_size + bub_storage_size <= ubStatus.ub_rest_size)
                         ? kNumOne
                         : ((aub_storage_size + bub_bank_storage_size <= ubStatus.ub_rest_size) ? kNumTwo : kNumZero);
      } else {
        align_mode = (aub_storage_size + bub_bank_storage_size <= ubStatus.ub_rest_size)
                         ? kNumTwo
                         : ((aub_bank_storage_size + bub_storage_size <= ubStatus.ub_rest_size) ? kNumOne : kNumZero);
      }
    }
  }
  return align_mode;
}

void CheckBankConflict(const BatchmatmulCompileParas& params, UbStatus& ubStatus) {
  GetABUbSize(ubStatus.k_aub, ubStatus.m_aub, ubStatus.k_bub, ubStatus.n_bub, params, ubStatus);
  // There‘s 4 align_mode here, 0->neither align, 1->AUB align and BUB not, 2->AUB not and BUB align, 3-> all aligned
  int align_mode = GetAlignMode(ubStatus);
  if (align_mode == kNumZero) {
    ubStatus.a_align_value = 1;
    ubStatus.b_align_value = 1;
    ubStatus.aub_align_bound = ubStatus.k_aub * kBlockSize * ubStatus.m_aub * kBlockSize;
    ubStatus.bub_align_bound = ubStatus.k_bub * kBlockSize * ubStatus.n_bub * kBlockSize;
  } else if (align_mode == kNumOne) {
    ubStatus.a_bank_conflict = false;
    ubStatus.b_align_value = 1;
    ubStatus.bub_align_bound = ubStatus.k_bub * kBlockSize * ubStatus.n_bub * kBlockSize;
    ubStatus.aub_size = ubStatus.aub_bank_size;
  } else if (align_mode == kNumTwo) {
    ubStatus.b_bank_conflict = false;
    ubStatus.a_align_value = 1;
    ubStatus.aub_align_bound = ubStatus.k_aub * kBlockSize * ubStatus.m_aub * kBlockSize;
    ubStatus.bub_size = ubStatus.bub_bank_size;
  } else {
    // align_mode is 3-> all aligned.
    ubStatus.aub_size = ubStatus.aub_bank_size;
    ubStatus.bub_size = ubStatus.bub_bank_size;
    ubStatus.a_bank_conflict = false;
    ubStatus.b_bank_conflict = false;
  }
}

void GetCUbFactors(const L0Status& l0Status, const BatchmatmulCompileParas& params, UbStatus& ubStatus) {
  // Initialize n_cub status.
  ubStatus.n_cub = l0Status.n_l0;
  bool condition_cub_n = ubStatus.max_dma_size + ubStatus.min_load_size > kUbFp16Size;
  if (condition_cub_n) {
    ubStatus.max_dma_size = kUbFp16Size - ubStatus.min_load_size;
    if (params.bias_flag) {
      ubStatus.max_dma_size -= l0Status.n_l0 * kBlockSize * ubStatus.db_cub;
    }
    ubStatus.n_cub =
        ubStatus.max_dma_size / (l0Status.m_l0 * kBlockSize * kBlockSize * (1 + params.fused_double_operand_num) *
                                 ubStatus.db_cub * ubStatus.cub_dtype_multi);
    GetNearestFactor(l0Status.n_l0, ubStatus.n_cub);
  }
  // The following CUB Size has no need to be uploaded.
  ubStatus.cub_size = ubStatus.n_cub * l0Status.m_l0 * kBlockSize * kBlockSize * (1 + params.fused_double_operand_num) *
                      ubStatus.db_cub * ubStatus.cub_dtype_multi;
  if (params.bias_flag) {
    ubStatus.cub_size += l0Status.n_l0 * kBlockSize * ubStatus.db_cub;
  }
  // Solve CUB Bank Conflict need to check cub_size.
}

void UpdateUbLoadSize(const BatchmatmulCompileParas& params, const L0Status& l0Status, UbStatus& ubStatus) {
  if (!ubStatus.cub_reuse_aub_flag && !ubStatus.cub_reuse_bub_flag) {
    ubStatus.min_load_size = ubStatus.aub_size + ubStatus.bub_size;
  } else if (ubStatus.cub_reuse_aub_flag && !ubStatus.cub_reuse_bub_flag) {
    ubStatus.min_load_size =
        ubStatus.bub_size + static_cast<int32_t>(ubStatus.aub_size * (1.0 - ubStatus.cub_aub_ratio));
    ubStatus.max_dma_size =
        max(ubStatus.max_dma_size, static_cast<int32_t>(ubStatus.aub_size * ubStatus.cub_aub_ratio));
  } else if (!ubStatus.cub_reuse_aub_flag && ubStatus.cub_reuse_bub_flag) {
    ubStatus.min_load_size =
        ubStatus.aub_size + static_cast<int32_t>(ubStatus.bub_size * (1.0 - ubStatus.cub_bub_ratio));
    ubStatus.max_dma_size =
        max(ubStatus.max_dma_size, static_cast<int32_t>(ubStatus.bub_size * ubStatus.cub_bub_ratio));
  } else {
    ubStatus.min_load_size = static_cast<int32_t>(ubStatus.aub_size * (1.0 - ubStatus.cub_aub_ratio)) +
                             static_cast<int32_t>(ubStatus.bub_size * (1.0 - ubStatus.cub_bub_ratio));
    ubStatus.max_dma_size =
        max(ubStatus.max_dma_size, static_cast<int32_t>(max(ubStatus.aub_size * ubStatus.cub_aub_ratio,
                                                            ubStatus.bub_size * ubStatus.cub_bub_ratio)));
  }
}

void UpdateUbStatus(const UbStatus &src_ub, UbStatus &dst_ub) {
  // Update dst UbStatus from source ubStataus
  dst_ub.k_aub = src_ub.k_aub;
  dst_ub.m_aub = src_ub.m_aub;
  dst_ub.k_bub = src_ub.k_bub;
  dst_ub.n_bub = src_ub.n_bub;
  dst_ub.aub_multi_flag = src_ub.aub_multi_flag;
  dst_ub.bub_multi_flag = src_ub.bub_multi_flag;
  dst_ub.n_cub = src_ub.n_cub;

  dst_ub.a_align_value = src_ub.a_align_value;
  dst_ub.b_align_value = src_ub.b_align_value;
  dst_ub.aub_align_bound = src_ub.aub_align_bound;
  dst_ub.bub_align_bound = src_ub.bub_align_bound;
}

int32_t GetUbMTE2Cost(const BatchmatmulCompileParas& params, const SingleCoreStatus& singleCoreStatus) {
  // Get The Cost of UB MTE2 process which is constructed as copy_gm_to_ub and nd2nz
  const L0Status& l0Status = singleCoreStatus.l0Status;
  const L1Status& l1Status = singleCoreStatus.l1Status;
  const UbStatus& ubStatus = singleCoreStatus.ubStatus;
  // Calculate MTE2 cost for AUB
  int32_t multi_k_aub_l1 = l1Status.kal1_16 / ubStatus.k_aub;
  int32_t multi_m_ub_l1 = l1Status.m_al1 * l0Status.m_l0 / ubStatus.m_aub;
  // Do compensation if Pre Ub still have bank conflict.
  multi_m_ub_l1 = ubStatus.a_bank_conflict ? multi_m_ub_l1 * kNumFour : multi_m_ub_l1;
  double aub_brand_width_utilization = params.trans_a_flag
                                           ? max((kWorstBandWidthUtilMulti / static_cast<double>(ubStatus.m_aub)), 1.0)
                                           : max((kWorstBandWidthUtilMulti / static_cast<double>(ubStatus.k_aub)), 1.0);
  int32_t aub_mte2_cost = static_cast<int32_t>(multi_k_aub_l1 * multi_m_ub_l1 * aub_brand_width_utilization);
  // Calculate MTE2 cost for BUB
  int32_t multi_k_bub_l1 = l1Status.kbl1_16 / ubStatus.k_bub;
  int32_t multi_n_ub_l1 = l1Status.n_bl1 * l0Status.n_l0 / ubStatus.n_bub;
  // Do compensation if Pre Ub still have bank conflict.
  multi_n_ub_l1 = ubStatus.b_bank_conflict ? multi_n_ub_l1 * kNumFour : multi_n_ub_l1;
  double bub_brand_width_utilization = params.trans_b_flag
                                           ? max((kWorstBandWidthUtilMulti / static_cast<double>(ubStatus.k_bub)), 1.0)
                                           : max((kWorstBandWidthUtilMulti / static_cast<double>(ubStatus.n_bub)), 1.0);
  int32_t bub_mte2_cost = static_cast<int32_t>(multi_k_bub_l1 * multi_n_ub_l1 * bub_brand_width_utilization);
  return aub_mte2_cost * bub_mte2_cost;
}

void GetUbFactorsInND(const string &op_type, const CoreStatus& coreStatus, const BatchmatmulCompileParas& params,
                      SingleCoreStatus& singleCoreStatus) {
  const L0Status& l0Status = singleCoreStatus.l0Status;
  const L1Status& l1Status = singleCoreStatus.l1Status;
  UbStatus& ubStatus = singleCoreStatus.ubStatus;
  // multi_m_ub_l1 * multi_k_aub_l1 (with bank conflict kNumTwo and worst brand utilization) + multi_n_ub_l1 *
  // multi_k_bub_l1
  int32_t min_ub_cost = l1Status.kal1_16 * l1Status.m_al1 * l0Status.m_l0 * kNumFour * kWorstBandWidthUtilMulti *
                        l1Status.kbl1_16 * l1Status.n_bl1 * l0Status.n_l0 * kNumFour * kWorstBandWidthUtilMulti;
  // Two means both aub and bub have bank conflict.
  int8_t min_bank_conflict_level = kNumTwo;
  int32_t min_multi_n_ub_l0 = l0Status.n_l0;
  UbStatus final_ub_status;
  GetAUbFactors(coreStatus, params, singleCoreStatus);
  for (int32_t aub_idx = 0; aub_idx < ubStatus.aub_cnt; aub_idx++) {
    ubStatus.k_aub = ubStatus.aub_results[aub_idx][0];
    ubStatus.m_aub = ubStatus.aub_results[aub_idx][1];
    GetBUbFactors(coreStatus, params, singleCoreStatus);
    for (int32_t bub_idx = 0; bub_idx < ubStatus.bub_cnt; bub_idx++) {
      ubStatus.k_bub = ubStatus.bub_results[bub_idx][0];
      ubStatus.n_bub = ubStatus.bub_results[bub_idx][1];
      // Initialize min/max dma copy size before updating it.
      ubStatus.min_dma_size = l0Status.m_l0 * kBlockSize * kBlockSize * (1 + params.fused_double_operand_num) *
                              ubStatus.db_cub * ubStatus.cub_dtype_multi;
      ubStatus.max_dma_size = l0Status.n_l0 * ubStatus.min_dma_size;
      ubStatus.aub_align_bound = ubStatus.k_aub * kBlockSize * ubStatus.m_aub * kBlockSize;
      ubStatus.bub_align_bound = ubStatus.k_bub * kBlockSize * ubStatus.n_bub * kBlockSize;
      UpdateUbReuseFlagAndRestSize(coreStatus, l1Status, l0Status, ubStatus);
      CheckBankConflict(params, ubStatus);
      UpdateUbLoadSize(params, l0Status, ubStatus);
      GetCUbFactors(l0Status, params, ubStatus);
      // Calculate CUB cost
      OP_TILING_CHECK(ubStatus.n_cub == 0,
                      OP_LOGW(op_type.c_str(),
                              "The current Tiling Candidate in MatMul/BatchMatMaul optiling exist "
                              "one invalid zero n_cub tiling result."),
                      continue);
      int32_t multi_n_ub_l0 = l0Status.n_l0 / ubStatus.n_cub;
      int32_t tmp_ub_cost = GetUbMTE2Cost(params, singleCoreStatus);
      int8_t tmp_bank_conflict_level =
          static_cast<int8_t>(ubStatus.a_bank_conflict) + static_cast<int8_t>(ubStatus.b_bank_conflict);
      if (tmp_ub_cost < min_ub_cost ||
          (tmp_ub_cost == min_ub_cost && tmp_bank_conflict_level < min_bank_conflict_level) ||
          (tmp_ub_cost == min_ub_cost && tmp_bank_conflict_level == min_bank_conflict_level &&
           multi_n_ub_l0 < min_multi_n_ub_l0)) {
        min_ub_cost = tmp_ub_cost;
        min_bank_conflict_level = tmp_bank_conflict_level;
        min_multi_n_ub_l0 = multi_n_ub_l0;
        UpdateUbStatus(ubStatus, final_ub_status);
      }
    }
  }
  UpdateUbStatus(final_ub_status, ubStatus); // Update final result to ubStatus.
}

void SetUbReuseFlag(const L1Status& l1Status, UbStatus& ubStatus) {
  // Set UB Reused Flag
  if (l1Status.both_full_load) {
    ubStatus.cub_reuse_aub_flag = true;
    ubStatus.cub_reuse_bub_flag = true;
  } else if (l1Status.al1_full_load && !l1Status.bl1_full_load) {
    ubStatus.cub_reuse_aub_flag = true;
  } else if (!l1Status.al1_full_load && l1Status.bl1_full_load) {
    ubStatus.cub_reuse_bub_flag = true;
  }
}

void UpdateL1FullLoadFlag(const string &op_type, const BatchmatmulRunParas &params, CoreStatus &coreStatus,
                          SingleCoreStatus &singleCoreStatus) {
  const L0Status& l0Status = singleCoreStatus.l0Status;
  L1Status& l1Status = singleCoreStatus.l1Status;
  int32_t n_single_core = ceil(static_cast<double>(params.n_32) / (coreStatus.n_dim * l1Status.n_bl1 * l0Status.n_l0));
  int32_t m_single_core = ceil(static_cast<double>(params.m_32) / (coreStatus.m_dim * l1Status.m_al1 * l0Status.m_l0));
  if (l1Status.kal1_16 >= l1Status.kbl1_16) {
    coreStatus.kal1_factor = ceil(static_cast<double>(params.k_32) / (coreStatus.k_dim * l1Status.kal1_16));
    coreStatus.kbl1_factor = coreStatus.kal1_factor * l1Status.kal1_16 / l1Status.kbl1_16;
  }
  else {
    coreStatus.kbl1_factor = ceil(static_cast<double>(params.k_32) / (coreStatus.k_dim * l1Status.kbl1_16));
    coreStatus.kal1_factor = coreStatus.kbl1_factor * l1Status.kbl1_16 / l1Status.kal1_16;
  }
  // initialize the full load flag
  l1Status.both_full_load = false;
  l1Status.al1_full_load = false;
  l1Status.bl1_full_load = false;
  if (m_single_core == 1 && coreStatus.kal1_factor == 1) {
    l1Status.al1_full_load = true;
    OP_LOGD(op_type.c_str(), "check special template, tiling al1 changed to full load");
  }
  if (n_single_core == 1 && coreStatus.kbl1_factor == 1) {
    l1Status.bl1_full_load = true;
    OP_LOGD(op_type.c_str(), "check special template, tiling bl1 changed to full load");
  }
  // Update the full_load flag in l1Status to ensure they are correct.
  if (l1Status.al1_full_load && l1Status.bl1_full_load) {
    l1Status.both_full_load = true;
  }
}

void GetUbFactors(const string &op_type, const BatchmatmulCompileParas &params, const CoreStatus &coreStatus,
                  SingleCoreStatus &singleCoreStatus)
{
  const L0Status& l0Status = singleCoreStatus.l0Status;
  const L1Status& l1Status = singleCoreStatus.l1Status;
  UbStatus& ubStatus = singleCoreStatus.ubStatus;
  // Set reused condition based on L1 attach situation
  if (params.split_k_flag) {
    // data in cub is in fp32 so the size used is double.
    ubStatus.cub_dtype_multi = kNumTwo;
  }
  SetUbReuseFlag(l1Status, ubStatus);
  // UB double Buffer default On
  ubStatus.db_aub = kDbOn;
  ubStatus.db_bub = kDbOn;
  ubStatus.db_cub = kDbOn;

  ubStatus.n_cub = l0Status.n_l0;
  ubStatus.min_dma_size = l0Status.m_l0 * kBlockSize * kBlockSize * (1 + params.fused_double_operand_num) *
                          ubStatus.db_cub * ubStatus.cub_dtype_multi;
  ubStatus.max_dma_size = l0Status.n_l0 * ubStatus.min_dma_size;
  if (params.bias_flag) {
    ubStatus.min_dma_size += l0Status.n_l0 * kBlockSize * ubStatus.db_cub;
    ubStatus.max_dma_size += l0Status.n_l0 * kBlockSize * ubStatus.db_cub;
  }

  if (params.nd_flag) {
    ubStatus.safe_ub_rest_size = kUbFp16Size - ubStatus.min_dma_size;
    // Only valid when fused_double_operand_num is smaller than aub_double_num and bub_double_num
    ubStatus.cub_aub_ratio = (params.fused_double_operand_num + 1) / (params.aub_double_num + 1);
    ubStatus.cub_bub_ratio = (params.fused_double_operand_num + 1) / (params.bub_double_num + 1);
    GetUbFactorsInND(op_type, coreStatus, params, singleCoreStatus);
  } else {
    // Get CUB factors for NZ in Mode
    GetCUbFactors(l0Status, params, ubStatus);
  }
  OP_LOGD(op_type.c_str(), "tiling n_cub:%d, db_cub:%d", l0Status.n_l0, l0Status.db_cub);
}


void GenTiling(const string &op_type, const BatchmatmulCompileParas &compile_params,
               BatchmatmulRunParas &run_params, Tiling &tiling, string &tilingId)
{
  OP_LOGD(op_type.c_str(), "cache tiling input shape batch:%d, m:%d, k:%d, n:%d", run_params.batch_32, run_params.m_32,
          run_params.k_32, run_params.n_32);
  CoreStatus coreStatus;
  SingleCoreStatus singleCoreStatus;
  BatchmatmulParas params;
  params.compile_params = &compile_params;
  params.run_params = &run_params;
  singleCoreStatus.l0Status.SetInitLoadStatus();
  if (compile_params.split_k_flag) {
    GetSplitKdim(op_type, params, coreStatus);
  }
  BlockDimCalculator blockDimCalculator;
  NonFactorMap(op_type, params, blockDimCalculator);
  int32_t blockValue = GetBlockDim(op_type, params, coreStatus, blockDimCalculator);
  GetL0Factors(op_type, coreStatus, blockValue, singleCoreStatus.l0Status);
  GetL1Factors(op_type, params, coreStatus, singleCoreStatus.l0Status, singleCoreStatus.l1Status);
  UpdateL1FullLoadFlag(op_type, run_params, coreStatus, singleCoreStatus);
  GetUbFactors(op_type, compile_params, coreStatus, singleCoreStatus);

  tiling.SetParams(coreStatus, singleCoreStatus.l0Status, singleCoreStatus.l1Status, singleCoreStatus.ubStatus, params);
  tiling.SetAttachFlag();
  tiling.GetTilingId(params);
  tilingId = tiling.tiling_id;
  OP_LOGD(op_type.c_str(), "the tiling id from cache tiling is: %s", tilingId.c_str());
}
} // namespace optiling
