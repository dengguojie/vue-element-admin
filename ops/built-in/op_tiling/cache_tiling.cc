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
static const int32_t kNONE = -INT_MAX;
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
static const int32_t UbFp16Size = kUbSize / kFp16Bytes;
static const int32_t kTypeZero = 0;
static const int32_t kTypeOne = 1;
static const int32_t kTypeTwo = 2;
static const int32_t kTypeThree = 3;
static const int32_t kBankConflictFactor = 4;
static const int32_t kL1FactorsLen = 6;
static const int32_t kCandidateLen = 2;

void Tiling::SetParams(const L2Status &l2Status, const L0Status &l0Status, const L1Status &l1Status,
                       const UbStatus &ubStatus, const BatchmatmulParas &params)
{
  batch_dim = l2Status.batch_dim;
  n_dim = l2Status.n_dim;
  m_dim = l2Status.m_dim;
  m_l0 = l0Status.m_l0;
  k_l0 = l0Status.k_l0;
  n_l0 = l0Status.n_l0;
  kal1_16 = l1Status.kal1_16;
  kbl1_16 = l1Status.kbl1_16;
  m_al1 = l1Status.m_al1;
  n_bl1 = l1Status.n_bl1;
  db_al1 = l1Status.db_al1;
  db_bl1 = l1Status.db_bl1;
  n_cub = ubStatus.n_cub;
  db_cub = ubStatus.db_cub;
  k_org_dim = l2Status.k * kBlockSize;
  db_l0c = l0Status.db_l0c;
  if (params.nd_flag) {
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

  if (m_al1 == kNONE) {
    al1_full_load = true;
    if (l2Status.batch == 1) {
      db_al1 = 1;
    }
  }
  if (n_bl1 == kNONE) {
    bl1_full_load = true;
    if (!params.b_have_batch || l2Status.batch == 1) {
      db_bl1 = 1;
    }
  }
  min_kl1_cmp_kl0 = (min(kal1_16, kbl1_16) == k_l0) ? 0 : 1;
}

void Tiling::SetAttachFlag()
{
  // find kernel ID
  bool kAl1FullLoad = kal1_16 * kBlockSize == k_org_dim;
  bool kBl1FullLoad = kbl1_16 * kBlockSize == k_org_dim;
  bool template1 = m_al1 == kNONE && n_bl1 == kNONE;
  bool template2 = m_al1 == kNONE && n_bl1 != kNONE && kBl1FullLoad;
  bool template3 = m_al1 == kNONE && n_bl1 != kNONE && !kBl1FullLoad;
  bool template4 = m_al1 != kNONE && n_bl1 == kNONE && kAl1FullLoad;
  bool template5 = m_al1 != kNONE && n_bl1 == kNONE && !kAl1FullLoad;
  bool template6 = m_al1 != kNONE && n_bl1 != kNONE && kAl1FullLoad && kBl1FullLoad;
  bool template7 = m_al1 != kNONE && n_bl1 != kNONE && kAl1FullLoad && !kBl1FullLoad;
  bool template8 = m_al1 != kNONE && n_bl1 != kNONE && !kAl1FullLoad && kBl1FullLoad;
  bool template9 = m_al1 != kNONE && n_bl1 != kNONE && !kAl1FullLoad && !kBl1FullLoad;
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
  int32_t tilingIDLongLong = 0;
  tilingIDLongLong = tilingIDLongLong * kDecimal + db_al1;
  tilingIDLongLong = tilingIDLongLong * kDecimal + db_bl1;
  tilingIDLongLong = tilingIDLongLong * kDecimal + db_l0c;
  tilingIDLongLong = tilingIDLongLong * kDecimal + abkl1_attach_flag;
  tilingIDLongLong = tilingIDLongLong * kDecimal + al1_attach_flag;
  tilingIDLongLong = tilingIDLongLong * kDecimal + bl1_attach_flag;
  tilingIDLongLong = tilingIDLongLong * kDecimal + min_kl1_cmp_kl0;
  if (params.nd_flag) {
    tilingIDLongLong = tilingIDLongLong * kDecimal + aub_multi_flag;
    tilingIDLongLong = tilingIDLongLong * kDecimal + bub_multi_flag;
  }
  this->tiling_id = to_string(tilingIDLongLong);
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

void GetTwoFactors(int32_t *res, const int32_t &base, const int32_t &dim, const int32_t &maxNum = 32)
{
  // for up bigger or equal to base + 1, find the smallest num which is a factor of dim
  // form down smaller or equal to base, find the biggest num which is a factor of dim
  int32_t cnt = 0;
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

void BL1FullLoadBlock(const L2Status &l2Status, BlockDimCalculator &blockDimCalculator, int32_t &n0, bool b_have_batch)
{
  if (n0 >= 1) {
    while (l2Status.n % n0 != 0) {
      n0--;
    }
    blockDimCalculator.amat_size = blockDimCalculator.ori_amat_size * (l2Status.n / n0);
    blockDimCalculator.bmat_size = b_have_batch ? l2Status.batch * l2Status.n : l2Status.n;
    blockDimCalculator.total_load_size = blockDimCalculator.amat_size + blockDimCalculator.bmat_size;
    blockDimCalculator.tmp_value = n0;
  }
}

void AL1FullLoadBlock(const L2Status &l2Status, BlockDimCalculator &blockDimCalculator, int32_t &m0)
{
  if (m0 >= 1) {
    while (l2Status.m % m0 != 0) {
      m0--;
    }
    blockDimCalculator.tmp_amat_size = blockDimCalculator.ori_amat_size;
    blockDimCalculator.tmp_bmat_size = l2Status.n * (blockDimCalculator.ori_amat_size / m0);
    blockDimCalculator.tmp_load_size = blockDimCalculator.tmp_amat_size + blockDimCalculator.tmp_bmat_size;
    if (blockDimCalculator.total_load_size > blockDimCalculator.tmp_load_size) {
      blockDimCalculator.bmat_size = blockDimCalculator.tmp_bmat_size;
      blockDimCalculator.amat_size = blockDimCalculator.tmp_amat_size;
      blockDimCalculator.total_load_size = blockDimCalculator.tmp_load_size;
      blockDimCalculator.tmp_value = 0;
    }
  }
}

void NeitherFullLoadBlock(const L2Status &l2Status, BlockDimCalculator &blockDimCalculator,
                          const int32_t (&nFactorTwoCandidates)[kCandidateLen],
                          const int32_t (&mFactorTwoCandidates)[kCandidateLen])
{
  for (auto const &n0: nFactorTwoCandidates) {
    for (auto const &m0: mFactorTwoCandidates) {
      if (m0 <= 0 || n0 <= 0) {
        continue;
      }
      if (m0 * n0 * kKbytes > kL0cSize) {
        continue;
      }
      blockDimCalculator.tmp_amat_size = blockDimCalculator.ori_amat_size * (l2Status.n / n0);
      blockDimCalculator.tmp_bmat_size = l2Status.n * (blockDimCalculator.ori_amat_size / m0);
      blockDimCalculator.tmp_load_size = blockDimCalculator.tmp_amat_size + blockDimCalculator.tmp_bmat_size;
      if (blockDimCalculator.tmp_load_size < blockDimCalculator.total_load_size) {
        blockDimCalculator.amat_size = blockDimCalculator.tmp_amat_size;
        blockDimCalculator.bmat_size = blockDimCalculator.tmp_bmat_size;
        blockDimCalculator.total_load_size = blockDimCalculator.tmp_load_size;
        blockDimCalculator.tmp_value = 0;
      }
    }
  }
}

void GetBlockDimHelper(L2Status &l2Status, BlockDimCalculator &blockDimCalculator, const int32_t m0s[][kCandidateLen],
                       const int32_t n0s[][kCandidateLen], bool b_have_batch, int32_t core_num)
{
  int32_t iIdx = blockDimCalculator.i_idx;
  int32_t jIdx = blockDimCalculator.j_idx;
  int32_t bFactor = blockDimCalculator.batch_dim_array[iIdx];
  int32_t nFactor = blockDimCalculator.n_dim_array[jIdx];
  for (int32_t mIdx = 0; mIdx < blockDimCalculator.m_dim_cnt; mIdx++) {
    int32_t mFactor = blockDimCalculator.m_dim_array[mIdx];
    if (bFactor * nFactor * mFactor > core_num) {
      break;
    }
    l2Status.batch = blockDimCalculator.batch / bFactor;
    l2Status.m = blockDimCalculator.m / mFactor;
    l2Status.n = blockDimCalculator.n / nFactor;
    // load size of A matrix is batch * m
    // load size of B matrix is n
    blockDimCalculator.ori_amat_size = l2Status.batch * l2Status.m;
    blockDimCalculator.ori_bmat_size = b_have_batch ? l2Status.batch * l2Status.n : l2Status.n;
    blockDimCalculator.amat_size = blockDimCalculator.ori_amat_size;
    blockDimCalculator.bmat_size = blockDimCalculator.ori_bmat_size;
    blockDimCalculator.total_load_size = blockDimCalculator.amat_size + blockDimCalculator.bmat_size;
    blockDimCalculator.tmp_value = 0;
    if (blockDimCalculator.total_load_size * blockDimCalculator.k_bytes > kL1Size) {
      blockDimCalculator.total_load_size = INT_MAX;
      // BL1 full load
      int32_t n0 =
        min(min((kL1Size / kFp16Bytes - kMinFractalSize) / blockDimCalculator.k_num, l2Status.n), kMaxFactor);
      BL1FullLoadBlock(l2Status, blockDimCalculator, n0, b_have_batch);
      // AL1 full load
      int32_t m0 = min(min((kL1Size / kFp16Bytes - kMinFractalSize) /
                             (kMinFractalSize * blockDimCalculator.k * blockDimCalculator.ori_amat_size),
                           blockDimCalculator.ori_amat_size),
                       kMaxFactor);
      AL1FullLoadBlock(l2Status, blockDimCalculator, m0);
      // neither full load max_m max_n
      // closest m and n
      NeitherFullLoadBlock(l2Status, blockDimCalculator, n0s[nFactor], m0s[mFactor]);
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
    bool tmpBlockingFlag = (loadSizeKb < kLoadSizeRangeLow && max(nFactor, mFactor) > kMLowRange);

    // updateSolution: bool whether update to a new block factor solution
    // use more coreNums or use same core num but has smaller loadsize
    // or same core num same loadsize but has bigger batch_dim * n_dim
    // when loadsize in a predetermined range, do not block factor solution
    // these predetermined range parameters is only suitable for cloud 60 platform

    auto updateSolution =
      (blockDimCalculator.total_load_size < blockDimCalculator.min_load_size) ||
        ((blockDimCalculator.total_load_size == blockDimCalculator.min_load_size) &&
          ((blockDimCalculator.n_dim_factor * blockDimCalculator.batch_dim_factor < bFactor * nFactor) ||
            (blockDimCalculator.n_dim_factor * blockDimCalculator.batch_dim_factor == bFactor * nFactor &&
              blockDimCalculator.batch_dim_factor < bFactor))) ||
        (blockDimCalculator.final_blocking_flag && (loadSizeKb - minLoadSizeKb) < kLoadSizeDiffRange &&
          max(nFactor, mFactor) < max(blockDimCalculator.n_dim_factor, blockDimCalculator.m_dim_factor));
    auto noUpdateSolution =
      (((loadSizeKb >= kLoadSizeRangeLow && loadSizeKb < kLoadSizeRangeHigh &&
        max(nFactor, mFactor) > kMHighRange && tmpBlockingPct > kBlockingPctGate) &&
        max(nFactor, mFactor) > max(blockDimCalculator.n_dim_factor, blockDimCalculator.m_dim_factor) &&
        double(blockDimCalculator.min_load_size - blockDimCalculator.total_load_size) /
          blockDimCalculator.min_load_size <
          kLoadSizeGate &&
        blockDimCalculator.core_use >= kCoreUseHighRange) ||
        ((loadSizeKb < kLoadSizeRangeLow && max(nFactor, mFactor) > kMLowRange) &&
          (max(nFactor, mFactor) > max(blockDimCalculator.n_dim_factor, blockDimCalculator.m_dim_factor)) &&
          ((minLoadSizeKb - loadSizeKb) < kLoadSizeDiffRange && blockDimCalculator.core_use > kCoreUseLowRange)));
    auto updateCondition = updateSolution && !noUpdateSolution;
    if (updateCondition) {
      blockDimCalculator.min_load_size = blockDimCalculator.total_load_size;
      blockDimCalculator.n_dim_factor = nFactor;
      blockDimCalculator.batch_dim_factor = bFactor;
      blockDimCalculator.m_dim_factor = mFactor;
      blockDimCalculator.final_blocking_flag = tmpBlockingFlag;
      blockDimCalculator.core_use =
        blockDimCalculator.n_dim_factor * blockDimCalculator.batch_dim_factor * blockDimCalculator.m_dim_factor;
      blockDimCalculator.final_value = blockDimCalculator.tmp_value;
    }
  }
}

int32_t GetBlockDim(const string &op_type, const BatchmatmulParas &params, L2Status &l2Status)
{
  // get batch_dim, m_dim and n_dim for single core
  // not support multi cores slicing along k dim
  // single core batch_dim, m_dim, n_dim is a factor of input batch, m, n

  OP_LOGD(op_type.c_str(), "input shape batch:%lld, m:%lld, k:%lld, n:%lld", params.batch, params.m, params.k,
          params.n);
  if (static_cast<uint32_t>(params.batch_32 * params.m_32 * params.n_32) < params.core_num) {
    l2Status.batch_dim = params.batch_32;
    l2Status.n_dim = params.n_32;
    l2Status.m_dim = params.m_32;
    l2Status.batch = 1;
    l2Status.m = 1;
    l2Status.k = params.k_32;
    l2Status.n = 1;
    OP_LOGD(op_type.c_str(), "multi-core slicing factor batch dim:%lld, n dim:%lld, m dim:%lld, m block pnt point:0",
            l2Status.batch_dim, l2Status.n_dim, l2Status.m_dim);
    return 0;
  }
  BlockDimCalculator blockDimCalculator;
  blockDimCalculator.batch = params.batch_32;
  blockDimCalculator.m = params.m_32;
  blockDimCalculator.k = params.k_32;
  blockDimCalculator.n = params.n_32;
  int32_t batchDimArray[params.core_num] = {0};
  int32_t nDimArray[params.core_num] = {0};
  int32_t mDimArray[params.core_num] = {0};
  GetFactors(&blockDimCalculator.batch_dim_cnt, batchDimArray, params.batch_32, params.core_num);
  GetFactors(&blockDimCalculator.n_dim_cnt, nDimArray, params.n_32, params.core_num);
  GetFactors(&blockDimCalculator.m_dim_cnt, mDimArray, params.m_32, params.core_num);
  int32_t mnMaxPnt = 16;
  int32_t m0s[params.core_num + 1][kCandidateLen] = {0};
  int32_t n0s[params.core_num + 1][kCandidateLen] = {0};
  for (int32_t idx = 0; idx < blockDimCalculator.n_dim_cnt; idx++) {
    int32_t tmpNDim = nDimArray[idx];
    int32_t tmpNSingleCore = params.n_32 / tmpNDim;
    GetTwoFactors(n0s[tmpNDim], mnMaxPnt, tmpNSingleCore, kMaxFactor);
  }
  for (int32_t idx = 0; idx < blockDimCalculator.m_dim_cnt; idx++) {
    int32_t tmpMDim = mDimArray[idx];
    int32_t tmpMSingleCore = params.m_32 / tmpMDim;
    GetTwoFactors(m0s[tmpMDim], mnMaxPnt, tmpMSingleCore, kMaxFactor);
  }
  blockDimCalculator.k_num = params.k_32 * kBlockSize * kBlockSize;
  blockDimCalculator.k_bytes = blockDimCalculator.k_num * kFp16Bytes;
  blockDimCalculator.n_dim_factor = 1;
  blockDimCalculator.batch_dim_factor = 1;
  blockDimCalculator.m_dim_factor = 1;
  blockDimCalculator.min_load_size = kL1Size / kFp16Bytes;
  blockDimCalculator.batch_dim_array = batchDimArray;
  blockDimCalculator.m_dim_array = mDimArray;
  blockDimCalculator.n_dim_array = nDimArray;
  for (int32_t iIdx = 0; iIdx < blockDimCalculator.batch_dim_cnt; iIdx++) {
    for (int32_t jIdx = 0; jIdx < blockDimCalculator.n_dim_cnt; jIdx++) {
      blockDimCalculator.i_idx = iIdx;
      blockDimCalculator.j_idx = jIdx;
      GetBlockDimHelper(l2Status, blockDimCalculator, m0s, n0s, params.b_have_batch, params.core_num);
    }
  }
  l2Status.batch_dim = blockDimCalculator.batch_dim_factor;
  l2Status.n_dim = blockDimCalculator.n_dim_factor;
  l2Status.m_dim = blockDimCalculator.m_dim_factor;
  l2Status.m = params.m_32 / blockDimCalculator.m_dim_factor;
  l2Status.n = params.n_32 / blockDimCalculator.n_dim_factor;
  l2Status.k = params.k_32;
  l2Status.batch = params.batch_32 / blockDimCalculator.batch_dim_factor;
  OP_LOGD(op_type.c_str(),
          "multi-core slicing factor batch dim:%lld, n dim:%lld, m dim:%lld, m block pnt point:%lld",
          l2Status.batch_dim, l2Status.n_dim, l2Status.m_dim, blockDimCalculator.final_value);
  return blockDimCalculator.final_value;
}

int32_t GetLoadSize(const L2Status &l2Status, const L0Status &l0Status)
{
  bool al1FullLoad =
    ((l2Status.m * l2Status.k + l0Status.n_l0 * l0Status.k_l0) * kBlockSize * kBlockSize * kFp16Bytes <=
      kL1Size);
  bool bl1FullLoad =
    ((l0Status.m_l0 * l0Status.k_l0 + l0Status.n_l0 * l2Status.k) * kBlockSize * kBlockSize * kFp16Bytes <=
      kL1Size);
  bool bothFullLoad = ((l2Status.m * l2Status.k + l0Status.n_l0 * l2Status.k) * kBlockSize *
    kBlockSize * kFp16Bytes <=
    kL1Size);
  int32_t num0a =
    bl1FullLoad ? l2Status.n : ((l2Status.m + l0Status.m_l0 - 1) / l0Status.m_l0) * l2Status.n;
  int32_t num0b =
    al1FullLoad ? l2Status.m : ((l2Status.n + l0Status.n_l0 - 1) / l0Status.n_l0) * l2Status.m;
  if ((al1FullLoad && bl1FullLoad) && !bothFullLoad) {
    return min(l2Status.n + ((l2Status.n + l0Status.n_l0 - 1) / l0Status.n_l0) * l2Status.m,
               l2Status.m + ((l2Status.m + l0Status.m_l0 - 1) / l0Status.m_l0) * l2Status.n);
  }
  return num0a + num0b;
}

void GetFinalMkn(L0Status &l0Status, const L2Status &l2Status)
{
  float tmpL0cUse = l0Status.m_l0 * l0Status.n_l0 * l0Status.db_l0c * kBlockSize * kBlockSize * 4 * 1.0 / kL0cSize;
  int32_t tmpMte1Loop = ((l0Status.n_l0 != 1) ? l0Status.k_l0 : 1) + ((l0Status.k_l0 != 1) ? l0Status.m_l0 : 1);
  int32_t tmpMul = l0Status.m_l0 * l0Status.k_l0 * l0Status.n_l0;
  int32_t tmpLoadSize = GetLoadSize(l2Status, l0Status);
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

void GetL0FactorsCand(L0Factors &resFactors, const L2Status &l2Status, L0Status &l0Status, int32_t *parasCombo) {
  GetL0StatusFromParasCombo(l0Status, parasCombo);
  int32_t majorDim = l2Status.m;
  int32_t minorDim = l2Status.n;
  int32_t majorDimK = l0Status.max_mk;
  int32_t minorDimK = l0Status.max_nk;
  if (l0Status.max_axis_idx != 0) {
    majorDim = l2Status.n;
    minorDim = l2Status.m;
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
      GetTwoFactors(k0Factors, k0Max, l2Status.k, k0Max);
      for (auto &k0: k0Factors) {
        if (k0 == 0) {
          continue;
        }
        if (l0Status.max_axis_idx == 0) {
          l0Status.m_l0 = majorDimFactor;
          l0Status.n_l0 = minorDimFactor;
        } else {
          l0Status.m_l0 = minorDimFactor;
          l0Status.n_l0 = majorDimFactor;
        }
        l0Status.k_l0 = k0;
        GetFinalMkn(l0Status, l2Status);
      }
    }
  }
  SetResFactors(resFactors, l0Status);
}

void GetL0Factors(const string &op_type, const L2Status &l2Status, const int32_t &blockValue,
                  L0Status &l0Status)
{
  // get m_l0, n_l0, k_l0 factor when singlecore m, n, k is know
  // m_l0, n_l0, k_l0 is a factor of single core m, n, k

  int32_t dbAOnBOnCOnIdx = 0;
  int32_t dbAOnBOnCOffIdx = 1;
  L0Factors resFactors[kL0ParasComboLen];
  for (int32_t i = 0; i < kL0ParasComboLen; ++i) {
    MKNParasCombo mknParasCombo = GetParasCombo(i, blockValue);
    GetL0FactorsCand(resFactors[i], l2Status, l0Status, mknParasCombo.parasCombo);
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
  OP_LOGD(op_type.c_str(), "tiling m_l0:%lld, n_l0:%lld, k_l0:%lld", l0Status.m_l0, l0Status.n_l0, l0Status.k_l0);
  OP_LOGD(op_type.c_str(), "tiling db_l0a:%lld, db_l0b:%lld, db_l0c:%lld", l0Status.db_l0a, l0Status.db_l0b,
          l0Status.db_l0c);
  OP_LOGD(op_type.c_str(), "tiling db_cub:%lld", l0Status.db_cub);
}

int32_t GetL1Size(const L1Status &l1Status, const L0Status &l0Status) {
  int32_t curL1Size =
    l1Status.m_al1 * l0Status.m_l0 * kBlockSize * l1Status.kal1_16 * kBlockSize * l1Status.db_al1 * kFp16Bytes +
      l1Status.n_bl1 * l0Status.n_l0 * kBlockSize * l1Status.kbl1_16 * kBlockSize * l1Status.db_bl1 * kFp16Bytes;
  return curL1Size;
}

void L1StatusBothFullLoad(const L2Status &l2Status, const L0Status &l0Status, L1Status &l1Status,
                          int32_t res[][7])
{
  int32_t curL1Size = GetL1Size(l1Status, l0Status);
  if (curL1Size <= kL1Size) {
    l1Status.both_full_load = true;
    l1Status.load_size = l2Status.m + l2Status.n;
    res[kIdxZero][kIdxZero] = l1Status.kal1_16;
    res[kIdxZero][kIdxOne] = l1Status.m_al1;
    res[kIdxZero][kIdxTwo] = l1Status.db_al1;
    res[kIdxZero][kIdxThree] = l1Status.kbl1_16;
    res[kIdxZero][kIdxFour] = l1Status.n_bl1;
    res[kIdxZero][kIdxFive] = l1Status.db_bl1;
    res[kIdxZero][kIdxSix] = l1Status.load_size;
  }
}

void L1StatusAl1FullLoad(const L2Status &l2Status, const L0Status &l0Status, L1Status &l1Status,
                         int32_t res[][7])
{
  int32_t curL1Size;
  int32_t mRepeat = l2Status.m / l0Status.m_l0;
  int32_t nRepeat = l2Status.n / l0Status.n_l0;
  curL1Size = GetL1Size(l1Status, l0Status);
  if (curL1Size <= kL1Size) {
    l1Status.al1_full_load = true;
    l1Status.al1_size = l2Status.k * l2Status.m * kBlockSize * kBlockSize * kFp16Bytes;
    l1Status.bl1_size = kL1Size - l1Status.al1_size;
    l1Status.db_bl1 = kDbOn;
    if (GetL1Size(l1Status, l0Status) > kL1Size) {
      l1Status.db_bl1 = kDbOff;
    }
    l1Status.kbl1_16 = min(
        l1Status.bl1_size / (l1Status.n_bl1 * l0Status.n_l0 * kBlockSize * l1Status.db_bl1 * kFp16Bytes * kBlockSize),
        l2Status.k);
    l1Status.bl1_times = min(l1Status.kbl1_16 / l0Status.k_l0, l1Status.max_k_bl1);
    GetNearestFactor(l1Status.all_times, l1Status.bl1_times);
    l1Status.kbl1_16 = l1Status.bl1_times * l0Status.k_l0;
    if (l1Status.kbl1_16 == l2Status.k) {
      l1Status.n_bl1 = min(l1Status.bl1_size / (l1Status.kbl1_16 * l0Status.n_l0 * kBlockSize * l1Status.db_bl1 *
                             kFp16Bytes * kBlockSize),
                           l1Status.max_n_bl1);
      GetNearestFactor(nRepeat, l1Status.n_bl1);
    }
    l1Status.load_size = l2Status.m + (l1Status.kbl1_16 == l2Status.k ? 1 : mRepeat) * l2Status.n;
    res[kIdxOne][kIdxZero] = l1Status.kal1_16;
    res[kIdxOne][kIdxOne] = l1Status.m_al1;
    res[kIdxOne][kIdxTwo] = l1Status.db_al1;
    res[kIdxOne][kIdxThree] = l1Status.kbl1_16;
    res[kIdxOne][kIdxFour] = l1Status.n_bl1;
    res[kIdxOne][kIdxFive] = l1Status.db_bl1;
    res[kIdxOne][kIdxSix] = l1Status.load_size;
  }
}

void L1StatusBl1FullLoad(const L2Status &l2Status, const L0Status &l0Status, L1Status &l1Status,
                         int32_t res[][7])
{
  int32_t curL1Size;
  int32_t mRepeat = l2Status.m / l0Status.m_l0;
  int32_t nRepeat = l2Status.n / l0Status.n_l0;
  curL1Size = GetL1Size(l1Status, l0Status);
  if (curL1Size <= kL1Size) {
    l1Status.bl1_full_load = true;
    l1Status.bl1_size = l2Status.k * l2Status.n * kBlockSize * kBlockSize * kFp16Bytes;
    l1Status.al1_size = kL1Size - l1Status.bl1_size;
    l1Status.db_al1 = kDbOn;
    if (GetL1Size(l1Status, l0Status) > kL1Size) {
      l1Status.db_al1 = kDbOff;
    }
    l1Status.kal1_16 = min(
        l1Status.al1_size / (l1Status.m_al1 * l0Status.m_l0 * kBlockSize * l1Status.db_al1 * kFp16Bytes * kBlockSize),
        l2Status.k);
    l1Status.al1_times = min(l1Status.kal1_16 / l0Status.k_l0, l1Status.max_k_al1);
    GetNearestFactor(l1Status.all_times, l1Status.al1_times);
    l1Status.kal1_16 = l1Status.al1_times * l0Status.k_l0;
    if (l1Status.kal1_16 == l2Status.k) {
      l1Status.m_al1 = min(l1Status.al1_size / (l1Status.kal1_16 * l0Status.m_l0 * kBlockSize * l1Status.db_al1 *
                             kFp16Bytes * kBlockSize),
                           l1Status.max_m_al1);
      GetNearestFactor(mRepeat, l1Status.m_al1);
    }
    l1Status.load_size =
      l2Status.n +
        ((l2Status.m == l1Status.m_al1 * l0Status.m_l0 && l1Status.kal1_16 == l2Status.k) ? 1 : nRepeat) *
          l2Status.m;
    res[kIdxTwo][kIdxZero] = l1Status.kal1_16;
    res[kIdxTwo][kIdxOne] = l1Status.m_al1;
    res[kIdxTwo][kIdxTwo] = l1Status.db_al1;
    res[kIdxTwo][kIdxThree] = l1Status.kbl1_16;
    res[kIdxTwo][kIdxFour] = l1Status.n_bl1;
    res[kIdxTwo][kIdxFive] = l1Status.db_bl1;
    res[kIdxTwo][kIdxSix] = l1Status.load_size;
  }
}

void NeitherFullLoadDb(const L2Status &l2Status, const L0Status &l0Status, L1Status &l1Status,
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
  l1Status.kbl1_16 = l2Status.k;
  bool bothDoubleBuffer = l2Status.m != l0Status.m_l0 && l2Status.k > l0Status.k_l0 &&
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

void NeitherFullLoadMN(const L2Status &l2Status, const L0Status &l0Status, L1Status &l1Status,
                       const BatchmatmulParas &params)
{
  int32_t mRepeat = l2Status.m / l0Status.m_l0;
  int32_t nRepeat = l2Status.n / l0Status.n_l0;
  if (l0Status.k_l0 == params.k_32) {
    if (params.m_32 > params.n_32) {
      l1Status.bl1_size = params.k_32 * l0Status.n_l0 * kBlockSize * kBlockSize * l1Status.db_bl1 * kFp16Bytes;
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
      l1Status.al1_size = params.k_32 * l0Status.m_l0 * kBlockSize * kBlockSize * l1Status.db_al1 * kFp16Bytes;
      l1Status.bl1_size = kL1Size - l1Status.al1_size;
      l1Status.n_bl1 = min(l1Status.bl1_size / (l1Status.kbl1_16 * l0Status.n_l0 * kBlockSize * l1Status.db_bl1 *
                             kFp16Bytes * kBlockSize),
                           l1Status.max_n_bl1);
      GetNearestFactor(nRepeat, l1Status.n_bl1);
      l1Status.bl1_size = params.k_32 * l0Status.n_l0 * kBlockSize * kBlockSize * l1Status.db_bl1 * kFp16Bytes;
      l1Status.al1_size = kL1Size - l1Status.bl1_size;
      l1Status.m_al1 = min(l1Status.al1_size / (l1Status.kal1_16 * l0Status.m_l0 * kBlockSize * l1Status.db_al1 *
                             kFp16Bytes * kBlockSize),
                           l1Status.max_m_al1);
      GetNearestFactor(mRepeat, l1Status.m_al1);
    }
  }
}

void NeitherFullLoadKforNZ(const L2Status &l2Status, const L0Status &l0Status, L1Status &l1Status,
                           const BatchmatmulParas &params)
{
  l1Status.kbl1_16 = params.k_32;
  if (GetL1Size(l1Status, l0Status) <= kL1Size) {
    l1Status.bl1_size = params.k_32 * l0Status.n_l0 * kBlockSize * kBlockSize * l1Status.db_bl1 * kFp16Bytes;
    l1Status.al1_size = kL1Size - l1Status.bl1_size;
    l1Status.kal1_16 = min(
        l1Status.al1_size / (l1Status.m_al1 * l0Status.m_l0 * kBlockSize * l1Status.db_al1 * kFp16Bytes * kBlockSize),
        l2Status.k);
    l1Status.al1_times = min(l1Status.kal1_16 / l0Status.k_l0, l1Status.max_k_al1);
    GetNearestFactor(l1Status.all_times, l1Status.al1_times);
    l1Status.kal1_16 = l1Status.al1_times * l0Status.k_l0;
  } else {
    int32_t perK = min(kL1Size /
                         (l0Status.m_l0 * kBlockSize * kBlockSize * l1Status.db_al1 * kFp16Bytes +
                           kBlockSize * l0Status.n_l0 * kBlockSize * l1Status.db_bl1 * kFp16Bytes) /
                         l0Status.k_l0 * l0Status.k_l0,
                       l2Status.k);
    int32_t perTimes = min(perK / l0Status.k_l0, max(l1Status.max_k_al1, l1Status.max_k_bl1));
    GetNearestFactor(l1Status.all_times, perTimes);
    perK = perTimes * l0Status.k_l0;
    l1Status.kal1_16 = perK;
    l1Status.kbl1_16 = perK;
  }
}

void NeitherFullLoadKforND(const L2Status &l2Status, const L0Status &l0Status, L1Status &l1Status,
                           const int &kmax_axis)
{
  if (kmax_axis == kTypeOne) {
    // first get k_al1, second get k_bl1
    l1Status.kbl1_16 = l0Status.k_l0;
    l1Status.bl1_size = l1Status.kbl1_16 * l0Status.n_l0 * kBlockSize * kBlockSize * l1Status.db_bl1 * kFp16Bytes;
    l1Status.al1_size = kL1Size - l1Status.bl1_size;
    l1Status.kal1_16 = min(
        l1Status.al1_size / (l1Status.m_al1 * l0Status.m_l0 * kBlockSize * l1Status.db_al1 * kFp16Bytes * kBlockSize),
        l2Status.k);
    l1Status.al1_times = l1Status.kal1_16 / l0Status.k_l0;
    GetNearestFactor(l1Status.all_times, l1Status.al1_times);
    l1Status.kal1_16 = l1Status.al1_times * l0Status.k_l0;
    l1Status.al1_size = l1Status.kal1_16 * l0Status.m_l0 * kBlockSize * kBlockSize * l1Status.db_al1 * kFp16Bytes;
    l1Status.bl1_size = kL1Size - l1Status.al1_size;
    l1Status.kbl1_16 = min(
        l1Status.bl1_size / (l1Status.n_bl1 * l0Status.n_l0 * kBlockSize * l1Status.db_bl1 * kFp16Bytes * kBlockSize),
        l2Status.k);
    l1Status.bl1_times = min(l1Status.kbl1_16 / l0Status.k_l0, l1Status.max_k_bl1);
    GetNearestFactor(l1Status.all_times, l1Status.bl1_times);
    l1Status.kbl1_16 = l1Status.bl1_times * l0Status.k_l0;
  } else if (kmax_axis == kTypeTwo) {
    // first get k_bl1, second get k_al1
    l1Status.kal1_16 = l0Status.k_l0;
    l1Status.al1_size = l1Status.kal1_16 * l0Status.m_l0 * kBlockSize * kBlockSize * l1Status.db_al1 * kFp16Bytes;
    l1Status.bl1_size = kL1Size - l1Status.al1_size;
    l1Status.kbl1_16 = min(
        l1Status.bl1_size / (l1Status.n_bl1 * l0Status.n_l0 * kBlockSize * l1Status.db_bl1 * kFp16Bytes * kBlockSize),
        l2Status.k);
    l1Status.bl1_times = l1Status.kbl1_16 / l0Status.k_l0;
    GetNearestFactor(l1Status.all_times, l1Status.bl1_times);
    l1Status.kbl1_16 = l1Status.bl1_times * l0Status.k_l0;
    l1Status.bl1_size = l1Status.kbl1_16 * l0Status.n_l0 * kBlockSize * kBlockSize * l1Status.db_bl1 * kFp16Bytes;
    l1Status.al1_size = kL1Size - l1Status.bl1_size;
    l1Status.kal1_16 = min(
        l1Status.al1_size / (l1Status.m_al1 * l0Status.m_l0 * kBlockSize * l1Status.db_al1 * kFp16Bytes * kBlockSize),
        l2Status.k);
    l1Status.al1_times = min(l1Status.kal1_16 / l0Status.k_l0, l1Status.max_k_al1);
    GetNearestFactor(l1Status.all_times, l1Status.al1_times);
    l1Status.kal1_16 = l1Status.al1_times * l0Status.k_l0;
  }
}

void NeitherFullLoadK(const L2Status &l2Status, const L0Status &l0Status, L1Status &l1Status,
                      const BatchmatmulParas &params)
{
  // 1 -> let k_al1 bigger, 2 -> let k_bl1 bigger, 0 -> no matter
  int kmax_axis = kTypeZero;
  if (!params.trans_a_flag && !params.trans_b_flag) {
    kmax_axis = kTypeOne;
  } else if (params.trans_a_flag && params.trans_b_flag) {
    kmax_axis = kTypeTwo;
  } else if (!params.trans_a_flag && params.trans_b_flag) {
    kmax_axis = l0Status.m_l0 > l0Status.n_l0 ? kTypeOne : kTypeTwo;
  }

  if (params.nd_flag && kmax_axis != kTypeZero) {
    NeitherFullLoadKforND(l2Status, l0Status, l1Status, kmax_axis);
  } else {
    NeitherFullLoadKforNZ(l2Status, l0Status, l1Status, params);
  }
}

void L1StatusNeitherFullLoad(const L2Status &l2Status, const BatchmatmulParas &params,
                             const L0Status &l0Status, L1Status &l1Status, int32_t res[][7])
{
  int32_t mRepeat = l2Status.m / l0Status.m_l0;
  int32_t nRepeat = l2Status.n / l0Status.n_l0;
  int32_t kBl1Db = (l2Status.m == l0Status.m_l0) ? l0Status.k_l0 : l2Status.k;
  NeitherFullLoadDb(l2Status, l0Status, l1Status, kBl1Db);
  NeitherFullLoadMN(l2Status, l0Status, l1Status, params);
  NeitherFullLoadK(l2Status, l0Status, l1Status, params);
  // k_al1 and k_bl1 must be a factor of each other
  if (l1Status.kal1_16 > l1Status.kbl1_16 && l1Status.kal1_16 % l1Status.kbl1_16 != 0) {
    while (l1Status.kal1_16 % l1Status.kbl1_16 != 0 || l2Status.k % l1Status.kal1_16 != 0) {
      l1Status.kal1_16 -= 1;
    }
  } else if (l1Status.kal1_16 < l1Status.kbl1_16 && l1Status.kbl1_16 % l1Status.kal1_16 != 0) {
    while (l1Status.kbl1_16 % l1Status.kal1_16 != 0 || l2Status.k % l1Status.kbl1_16 != 0) {
      l1Status.kbl1_16 -= 1;
    }
  }
  l1Status.load_size =
    ((l2Status.m == l1Status.m_al1 * l0Status.m_l0 && l1Status.kal1_16 == params.k_32) ? 1 : nRepeat) *
      l2Status.m + (l1Status.kbl1_16 == params.k_32 ? 1 : mRepeat) * l2Status.n;
  res[kIdxThree][kIdxZero] = l1Status.kal1_16;
  res[kIdxThree][kIdxOne] = l1Status.m_al1;
  res[kIdxThree][kIdxTwo] = l1Status.db_al1;
  res[kIdxThree][kIdxThree] = l1Status.kbl1_16;
  res[kIdxThree][kIdxFour] = l1Status.n_bl1;
  res[kIdxThree][kIdxFive] = l1Status.db_bl1;
  res[kIdxThree][kIdxSix] = l1Status.load_size;
}

void GetL1Factors(const string &op_type, const BatchmatmulParas &params, const L2Status &l2Status,
                  const L0Status &l0Status, L1Status &l1Status)
{
  // get m_al1, n_bl1, kal1_16, kbl1_16 factors when L0, singlecore factor is know
  // get al1, bl1 double buffer factors

  int32_t mte1Loop = 50 / ((l0Status.n_l0 == 1 ? 1 : l0Status.k_l0) + (l0Status.k_l0 == 1 ? 1 : l0Status.m_l0));
  int32_t res[4][7] = {0};
  l1Status.all_times = l2Status.k / l0Status.k_l0;
  l1Status.max_m_al1 = (l2Status.m + l0Status.m_l0 - 1) / l0Status.m_l0;
  l1Status.max_n_bl1 = (l2Status.n + l0Status.n_l0 - 1) / l0Status.n_l0;
  l1Status.max_k_al1 =
    max(mte1Loop, ((kMinMte1Load + l0Status.m_l0 - 1) / l0Status.m_l0 + l0Status.k_l0 - 1) / l0Status.k_l0);
  l1Status.max_k_bl1 =
    max(mte1Loop, ((kMinMte1Load + l0Status.n_l0 - 1) / l0Status.n_l0 + l0Status.k_l0 - 1) / l0Status.k_l0);
  // both AL1 and Bl1 full load
  int32_t both_full_load_factors[kL1FactorsLen] =
      {l2Status.k, l2Status.k, l1Status.max_m_al1, l1Status.max_n_bl1, kDbOff, kDbOff};
  l1Status.SetStatus(both_full_load_factors);
  L1StatusBothFullLoad(l2Status, l0Status, l1Status, res);
  // only AL1 full load
  int32_t al1_full_load_factors[kL1FactorsLen] = {l2Status.k, l0Status.k_l0, l1Status.max_m_al1, 1, kDbOff, kDbOff};
  l1Status.SetStatus(al1_full_load_factors);
  L1StatusAl1FullLoad(l2Status, l0Status, l1Status, res);
  // only BL1 full load
  int32_t bl1_full_load_factors[kL1FactorsLen] = {l0Status.k_l0, l2Status.k, 1, l1Status.max_n_bl1, kDbOff, kDbOff};
  l1Status.SetStatus(bl1_full_load_factors);
  L1StatusBl1FullLoad(l2Status, l0Status, l1Status, res);
  // neither AL1 nor Bl1 full load
  int32_t neither_full_load_factors[kL1FactorsLen] = {l0Status.k_l0, l0Status.k_l0, 1, 1, kDbOn, kDbOn};
  l1Status.SetStatus(neither_full_load_factors);
  L1StatusNeitherFullLoad(l2Status, params, l0Status, l1Status, res);
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
  OP_LOGD(op_type.c_str(), "tiling kal1_16:%lld, kbl1_16:%lld, k_l0:%lld", l1Status.kal1_16, l1Status.kbl1_16,
          l0Status.k_l0);
  OP_LOGD(op_type.c_str(), "tiling m_al1:%lld, n_bl1:%lld", l1Status.m_al1, l1Status.n_bl1);
  OP_LOGD(op_type.c_str(), "tiling db_al1:%lld, db_bl1:%lld", l1Status.db_al1, l1Status.db_bl1);
}

bool CheckABUbSize(const int32_t &ub_rest_size, const int32_t &k_aub, const int32_t &m_aub,
                   const int32_t &k_bub, const int32_t &n_bub, const BatchmatmulParas &params, UbStatus &ubStatus)
{
  ubStatus.aub_size = k_aub * kBlockSize * m_aub * kBlockSize * ubStatus.db_aub * (1 + params.aub_double_num);
  ubStatus.bub_size = k_bub * kBlockSize * n_bub * kBlockSize * ubStatus.db_bub * (1 + params.bub_double_num);
  return (ubStatus.aub_size + ubStatus.bub_size) <= ub_rest_size;
}

void GetABUbMax(const int32_t &ub_rest_size, const int32_t &k_aub, const int32_t &m_aub, const int32_t &k_bub,
                const int32_t &n_bub, const BatchmatmulParas &params, UbStatus &ubStatus, const int max_num)
{
  // max_num: 0->k_aub, 1->k_bub, 2->m_aub, 3->n_bub
  ubStatus.aub_size = k_aub * kBlockSize * m_aub * kBlockSize * ubStatus.db_aub * (1 + params.aub_double_num);
  ubStatus.bub_size = k_bub * kBlockSize * n_bub * kBlockSize * ubStatus.db_bub * (1 + params.bub_double_num);
  if (max_num == kTypeZero) {
    ubStatus.k_aub = (ub_rest_size - ubStatus.bub_size) /
      (kBlockSize * m_aub * kBlockSize * ubStatus.db_aub * (1 + params.aub_double_num));
  } else if (max_num == kTypeOne) {
    ubStatus.k_bub = (ub_rest_size - ubStatus.aub_size) /
      (kBlockSize * n_bub * kBlockSize * ubStatus.db_bub * (1 + params.bub_double_num));
  } else if (max_num == kTypeTwo) {
    ubStatus.m_aub = (ub_rest_size - ubStatus.bub_size) /
      (k_aub * kBlockSize * kBlockSize * ubStatus.db_aub * (1 + params.aub_double_num));
  } else if (max_num == kTypeThree) {
    ubStatus.n_bub = (ub_rest_size - ubStatus.aub_size) /
      (k_bub * kBlockSize * kBlockSize * ubStatus.db_bub * (1 + params.bub_double_num));
  }
}

void GetAUbFactors(const int32_t &ub_rest_size, const L2Status &l2Status, const L1Status &l1Status,
                   const L0Status &l0Status, const BatchmatmulParas &params, UbStatus &ubStatus)
{
  int32_t al1_m = l1Status.m_al1 * l0Status.m_l0;
  bool condition_m2_k2 = (!params.at_l1_flag &&
    CheckABUbSize(ub_rest_size, l2Status.k, l2Status.m, ubStatus.k_bub, ubStatus.n_bub, params, ubStatus));
  bool condition_ml1_kl1 =
    CheckABUbSize(ub_rest_size, l1Status.kal1_16, al1_m, ubStatus.k_bub, ubStatus.n_bub, params, ubStatus);
  bool condition_ml1_kl0 = (params.trans_a_flag &&
    CheckABUbSize(ub_rest_size, l0Status.k_l0, al1_m, ubStatus.k_bub, ubStatus.n_bub, params, ubStatus));
  bool condition_ml1_k0 = (params.trans_a_flag &&
    CheckABUbSize(ub_rest_size, 1, al1_m, ubStatus.k_bub, ubStatus.n_bub, params, ubStatus));
  bool condition_ml0_kl1 = (!params.trans_a_flag &&
    CheckABUbSize(ub_rest_size, l1Status.kal1_16, l0Status.m_l0, ubStatus.k_bub, ubStatus.n_bub, params, ubStatus));
  bool condition_m0_kl1 = (!params.trans_a_flag &&
    CheckABUbSize(ub_rest_size, l1Status.kal1_16, 1, ubStatus.k_bub, ubStatus.n_bub, params, ubStatus));
  bool condition_ml0_kl0 =
    CheckABUbSize(ub_rest_size, l0Status.k_l0, l0Status.m_l0, ubStatus.k_bub, ubStatus.n_bub, params, ubStatus);
  bool condition_ml0_k0 = (params.trans_a_flag &&
    CheckABUbSize(ub_rest_size, 1, l0Status.m_l0, ubStatus.k_bub, ubStatus.n_bub, params, ubStatus));
  bool condition_m0_kl0 = (!params.trans_a_flag &&
    CheckABUbSize(ub_rest_size, l0Status.k_l0, 1, ubStatus.k_bub, ubStatus.n_bub, params, ubStatus));
  bool condition_m0_k0 = CheckABUbSize(ub_rest_size, 1, 1, ubStatus.k_bub, ubStatus.n_bub, params, ubStatus);
  if (condition_m2_k2) {
    ubStatus.k_aub = l2Status.k;
    ubStatus.m_aub = l2Status.m;
  } else if (condition_ml1_kl1) {
    ubStatus.k_aub = l1Status.kal1_16;
    ubStatus.m_aub = al1_m;
  } else if (condition_ml1_kl0) {
    ubStatus.k_aub = l0Status.k_l0;
    ubStatus.m_aub = al1_m;
  } else if (condition_ml1_k0) {
    ubStatus.k_aub = 1;
    ubStatus.m_aub = al1_m;
    GetABUbMax(ub_rest_size, 1, al1_m, ubStatus.k_bub, ubStatus.n_bub, params, ubStatus, kTypeZero);
    GetNearestFactor(l1Status.kal1_16, ubStatus.k_aub);
  } else if (condition_ml0_kl1) {
    ubStatus.k_aub = l1Status.kal1_16;
    ubStatus.m_aub = l0Status.m_l0;
  } else if (condition_m0_kl1) {
    ubStatus.k_aub = l1Status.kal1_16;
    ubStatus.m_aub = 1;
    GetABUbMax(ub_rest_size, l1Status.kal1_16, 1, ubStatus.k_bub, ubStatus.n_bub, params, ubStatus, kTypeTwo);
    GetNearestFactor(al1_m, ubStatus.m_aub);
  } else if (condition_ml0_kl0) {
    ubStatus.k_aub = l0Status.k_l0;
    ubStatus.m_aub = l0Status.m_l0;
  } else if (condition_ml0_k0) {
    ubStatus.k_aub = 1;
    ubStatus.m_aub = l0Status.m_l0;
  } else if (condition_m0_kl0) {
    ubStatus.k_aub = l0Status.k_l0;
    ubStatus.m_aub = 1;
  } else if (condition_m0_k0) {
    ubStatus.k_aub = 1;
    ubStatus.m_aub = 1;
  }
  ubStatus.aub_align_bound = ubStatus.k_aub * kBlockSize * ubStatus.m_aub * kBlockSize;
}

void GetBUbFactors(const int32_t &ub_rest_size, const L2Status &l2Status, const L1Status &l1Status,
                   const L0Status &l0Status, const BatchmatmulParas &params, UbStatus &ubStatus)
{
  int32_t bl1_n = l1Status.n_bl1 * l0Status.n_l0;
  bool condition_k2_n2 = (!params.at_l1_flag &&
    CheckABUbSize(ub_rest_size, ubStatus.k_aub, ubStatus.m_aub, l2Status.k, l2Status.n, params, ubStatus));
  bool condition_kl1_nl1 =
    CheckABUbSize(ub_rest_size, ubStatus.k_aub, ubStatus.m_aub, l1Status.kbl1_16, bl1_n, params, ubStatus);
  bool condition_kl0_nl1 = (!params.trans_b_flag &&
    CheckABUbSize(ub_rest_size, ubStatus.k_aub, ubStatus.m_aub, l0Status.k_l0, bl1_n, params, ubStatus));
  bool condition_k0_nl1 = (!params.trans_b_flag &&
    CheckABUbSize(ub_rest_size, ubStatus.k_aub, ubStatus.m_aub, 1, bl1_n, params, ubStatus));
  bool condition_kl1_nl0 = (params.trans_b_flag &&
    CheckABUbSize(ub_rest_size, ubStatus.k_aub, ubStatus.m_aub, l1Status.kbl1_16, l0Status.n_l0, params, ubStatus));
  bool condition_kl1_n0 = (params.trans_b_flag &&
    CheckABUbSize(ub_rest_size, ubStatus.k_aub, ubStatus.m_aub, l1Status.kbl1_16, 1, params, ubStatus));
  bool condition_kl0_nl0 =
    CheckABUbSize(ub_rest_size, ubStatus.k_aub, ubStatus.m_aub, l0Status.k_l0, l0Status.n_l0, params, ubStatus);
  bool condition_k0_nl0 = (!params.trans_b_flag &&
    CheckABUbSize(ub_rest_size, ubStatus.k_aub, ubStatus.m_aub, 1, l0Status.n_l0, params, ubStatus));
  bool condition_kl0_n0 = (params.trans_b_flag &&
    CheckABUbSize(ub_rest_size, ubStatus.k_aub, ubStatus.m_aub, l0Status.k_l0, 1, params, ubStatus));
  bool condition_k0_n0 = CheckABUbSize(ub_rest_size, ubStatus.k_aub, ubStatus.m_aub, 1, 1, params, ubStatus);
  if (condition_k2_n2) {
    ubStatus.k_bub = l2Status.k;
    ubStatus.n_bub = l2Status.n;
  } else if (condition_kl1_nl1) {
    ubStatus.k_bub = l1Status.kbl1_16;
    ubStatus.n_bub = bl1_n;
  } else if (condition_kl0_nl1) {
    ubStatus.k_bub = l0Status.k_l0;
    ubStatus.n_bub = bl1_n;
  } else if (condition_k0_nl1) {
    ubStatus.k_bub = 1;
    ubStatus.n_bub = bl1_n;
    GetABUbMax(ub_rest_size, ubStatus.k_aub, ubStatus.m_aub, 1, bl1_n, params, ubStatus, kTypeOne);
    GetNearestFactor(l1Status.kbl1_16, ubStatus.k_bub);
  } else if (condition_kl1_nl0) {
    ubStatus.k_bub = l1Status.kbl1_16;
    ubStatus.n_bub = l0Status.n_l0;
  } else if (condition_kl1_n0) {
    ubStatus.k_bub = l1Status.kbl1_16;
    ubStatus.n_bub = 1;
    GetABUbMax(ub_rest_size, ubStatus.k_aub, ubStatus.m_aub, l1Status.kbl1_16, 1, params, ubStatus, kTypeThree);
    GetNearestFactor(bl1_n, ubStatus.n_bub);
  } else if (condition_kl0_nl0) {
    ubStatus.k_bub = l0Status.k_l0;
    ubStatus.n_bub = l0Status.n_l0;
  } else if (condition_k0_nl0) {
    ubStatus.k_bub = 1;
    ubStatus.n_bub = l0Status.n_l0;
  } else if (condition_kl0_n0) {
    ubStatus.k_bub = l0Status.k_l0;
    ubStatus.n_bub = 1;
  } else if (condition_k0_n0) {
    ubStatus.k_bub = 1;
    ubStatus.n_bub = 1;
  }
  ubStatus.bub_align_bound = ubStatus.k_bub * kBlockSize * ubStatus.n_bub * kBlockSize;
}

void GetABUbSize(const int32_t &k_aub, const int32_t &m_aub, const int32_t &k_bub, const int32_t &n_bub,
                 const BatchmatmulParas &params, UbStatus &ubStatus) {
  ubStatus.aub_bank_size = k_aub * kBlockSize * m_aub * kBlockSize * ubStatus.db_aub * (1 + params.aub_double_num);
  ubStatus.bub_bank_size = k_bub * kBlockSize * n_bub * kBlockSize * ubStatus.db_bub * (1 + params.bub_double_num);
  ubStatus.aub_size = ubStatus.aub_bank_size;
  ubStatus.bub_size = ubStatus.bub_bank_size;
  if (params.trans_a_flag && m_aub % kBankConflictFactor == 0) {
    ubStatus.a_align_value = (m_aub + 1) * kBlockSize;
    ubStatus.aub_bank_size += k_aub * kBlockSize * kBlockSize * ubStatus.db_aub;
    ubStatus.aub_align_bound += k_aub * kBlockSize * kBlockSize;
  } else if (!params.trans_a_flag && k_aub % kBankConflictFactor == 0) {
    ubStatus.a_align_value = (k_aub + 1) * kBlockSize;
    ubStatus.aub_bank_size += kBlockSize * m_aub * kBlockSize * ubStatus.db_aub;
    ubStatus.aub_align_bound += kBlockSize * m_aub * kBlockSize;
  }
  if (params.trans_b_flag && k_bub % kBankConflictFactor == 0) {
    ubStatus.b_align_value = (k_bub + 1) * kBlockSize;
    ubStatus.bub_bank_size += kBlockSize * n_bub * kBlockSize * ubStatus.db_bub;
    ubStatus.bub_align_bound += kBlockSize * n_bub * kBlockSize;
  } else if (!params.trans_b_flag && n_bub % kBankConflictFactor == 0) {
    ubStatus.b_align_value = (n_bub + 1) * kBlockSize;
    ubStatus.bub_bank_size += k_bub * kBlockSize * kBlockSize * ubStatus.db_bub;
    ubStatus.bub_align_bound += k_bub * kBlockSize * kBlockSize;
  }
}

void CheckBankConflict(const int32_t &ub_rest_size, const BatchmatmulParas &params, UbStatus &ubStatus) {
  GetABUbSize(ubStatus.k_aub, ubStatus.m_aub, ubStatus.k_bub, ubStatus.n_bub, params, ubStatus);
  if (ubStatus.aub_bank_size + ubStatus.bub_bank_size <= ub_rest_size) {
    ubStatus.aub_size = ubStatus.aub_bank_size;
    ubStatus.bub_size = ubStatus.bub_bank_size;
  } else {
    // There‘s 3 align_mode here, 0->neither align, 1->AUB align and BUB not, 2->AUB not and BUB align
    int align_mode = kTypeZero;
    if (ubStatus.a_align_value != 1 && ubStatus.b_align_value != 1) {
      if (ubStatus.aub_bank_size > ubStatus.bub_bank_size) {
        align_mode = (ubStatus.aub_bank_size + ubStatus.bub_size <= ub_rest_size)
                         ? kTypeOne
                         : ((ubStatus.aub_size + ubStatus.bub_bank_size <= ub_rest_size) ? kTypeTwo : kTypeZero);
      } else {
        align_mode = (ubStatus.aub_size + ubStatus.bub_bank_size <= ub_rest_size)
                         ? kTypeTwo
                         : ((ubStatus.aub_bank_size + ubStatus.bub_size <= ub_rest_size) ? kTypeOne : kTypeZero);
      }
    }
    if (align_mode == kTypeZero) {
      ubStatus.a_align_value = 1;
      ubStatus.b_align_value = 1;
      ubStatus.aub_align_bound = ubStatus.k_aub * kBlockSize * ubStatus.m_aub * kBlockSize;
      ubStatus.bub_align_bound = ubStatus.k_bub * kBlockSize * ubStatus.n_bub * kBlockSize;
    } else if (align_mode == kTypeOne) {
      ubStatus.b_align_value = 1;
      ubStatus.bub_align_bound = ubStatus.k_bub * kBlockSize * ubStatus.n_bub * kBlockSize;
      ubStatus.aub_size = ubStatus.aub_bank_size;
    } else if (align_mode == kTypeTwo) {
      ubStatus.a_align_value = 1;
      ubStatus.aub_align_bound = ubStatus.k_aub * kBlockSize * ubStatus.m_aub * kBlockSize;
      ubStatus.bub_size = ubStatus.bub_bank_size;
    }
  }
}

void GetABUbFactors(const int32_t &ub_rest_size, const L2Status &l2Status, const L1Status &l1Status,
                    const L0Status &l0Status, const BatchmatmulParas &params, UbStatus &ubStatus)
{
  GetAUbFactors(ub_rest_size, l2Status, l1Status, l0Status, params, ubStatus);
  GetBUbFactors(ub_rest_size, l2Status, l1Status, l0Status, params, ubStatus);
  CheckBankConflict(ub_rest_size, params, ubStatus);
}

void GetUbFactors(const string &op_type, const BatchmatmulParas &params, const L2Status &l2Status,
                  const L1Status &l1Status, const L0Status &l0Status, UbStatus &ubStatus)
{
  if (!params.ubdb_flag) {
    ubStatus.db_aub = kDbOn;
    ubStatus.db_bub = kDbOn;
    ubStatus.db_cub = kDbOn;
  }
  ubStatus.n_cub = l0Status.n_l0;
  ubStatus.min_dma_size =
    l0Status.m_l0 * kBlockSize * kBlockSize * (1 + params.fused_double_operand_num) * ubStatus.db_cub;
  ubStatus.max_dma_size = l0Status.n_l0 * ubStatus.min_dma_size;
  if (params.bias_flag) {
    ubStatus.min_dma_size += l0Status.n_l0 * kBlockSize * ubStatus.db_cub;
    ubStatus.max_dma_size += l0Status.n_l0 * kBlockSize * ubStatus.db_cub;
  }
  // first get AUB BUB factors
  if (params.nd_flag) {
    int32_t ub_rest_size = UbFp16Size - ubStatus.min_dma_size;
    GetABUbFactors(ub_rest_size, l2Status, l1Status, l0Status, params, ubStatus);
    bool condition_db_bub = (params.ubdb_flag &&
      (ubStatus.bub_size * kDbOn + ubStatus.aub_size + ubStatus.min_dma_size <= UbFp16Size));
    bool condition_db_aub = (params.ubdb_flag &&
      (ubStatus.bub_size + ubStatus.aub_size * kDbOn + ubStatus.min_dma_size <= UbFp16Size));
    if (condition_db_bub) {
      ubStatus.db_bub = kDbOn;
      ubStatus.bub_size *= kDbOn;
    }
    if (condition_db_aub) {
      ubStatus.db_aub = kDbOn;
      ubStatus.aub_size *= kDbOn;
    }
    ubStatus.min_load_size = (params.cub_reused_flag ?
      (ubStatus.aub_size / (1 + params.aub_double_num) + ubStatus.bub_size / (1 + params.bub_double_num)) :
      (ubStatus.aub_size + ubStatus.bub_size));
    if (l1Status.kal1_16 == ubStatus.k_aub) {
      ubStatus.aub_multi_flag = kAttachFlagOne;
      if (l1Status.m_al1 * l0Status.m_l0 == ubStatus.m_aub) {
        ubStatus.aub_multi_flag = kAttachFlagTwo;
      }
    }
    if (l1Status.n_bl1 * l0Status.n_l0 == ubStatus.n_bub) {
      ubStatus.bub_multi_flag = kAttachFlagOne;
      if (l1Status.kbl1_16 == ubStatus.k_bub) {
        ubStatus.bub_multi_flag = kAttachFlagTwo;
      }
    }
  }
  // second get CUB factors
  bool condition_cub_n = (params.cub_reused_flag &&
    max(ubStatus.max_dma_size, ubStatus.min_load_size) + ubStatus.min_load_size > UbFp16Size) ||
    (!params.cub_reused_flag && ubStatus.max_dma_size + ubStatus.min_load_size > UbFp16Size);
  if (condition_cub_n) {
    ubStatus.max_dma_size = UbFp16Size - ubStatus.min_load_size;
    if (params.bias_flag) {
      ubStatus.max_dma_size -= l0Status.n_l0 * kBlockSize * ubStatus.db_cub;
    }
    ubStatus.n_cub = ubStatus.max_dma_size /
      (l0Status.m_l0 * kBlockSize * kBlockSize * (1 + params.fused_double_operand_num) * ubStatus.db_cub);
    GetNearestFactor(l0Status.n_l0, ubStatus.n_cub);
  }
  ubStatus.cub_size =
    ubStatus.n_cub * l0Status.m_l0 * kBlockSize * kBlockSize * (1 + params.fused_double_operand_num) * ubStatus.db_cub;
  if (params.bias_flag) {
    ubStatus.cub_size += l0Status.n_l0 * kBlockSize * ubStatus.db_cub;
  }
  bool condition_db_cub =
    params.ubdb_flag && (max(ubStatus.cub_size * kDbOn, ubStatus.min_load_size) + ubStatus.min_load_size <= UbFp16Size);
  if (condition_db_cub) {
    ubStatus.db_cub = kDbOn;
  }
  OP_LOGD(op_type.c_str(), "tiling n_cub:%lld, db_cub:%lld", l0Status.n_l0, l0Status.db_cub);
}

void CheckSpecialTemplate(const string &op_type, const L2Status &l2Status, const L0Status &l0Status,
                          L1Status &l1Status)
{
  if (l2Status.m / (l1Status.m_al1 * l0Status.m_l0) == 1 && l1Status.kal1_16 == l2Status.k) {
    l1Status.m_al1 = kNONE;
    OP_LOGD(op_type.c_str(), "check special template, tiling al1 changed to full load");
  }
  if (l1Status.n_bl1 * l0Status.n_l0 == l2Status.n && l1Status.kbl1_16 == l2Status.k) {
    l1Status.n_bl1 = kNONE;
    OP_LOGD(op_type.c_str(), "check special template, tiling bl1 changed to full load");
  }
}

void GenTiling(const string &op_type, const BatchmatmulParas &params, Tiling &tiling, string &tilingId)
{
  OP_LOGD(op_type.c_str(), "cache tiling input shape batch:%lld, m:%lld, k:%lld, n:%lld", params.batch, params.m,
          params.k, params.n);
  L2Status l2Status;
  L0Status l0Status;
  L1Status l1Status;
  UbStatus ubStatus;
  l0Status.SetInitLoadStatus();
  int32_t blockValue = GetBlockDim(op_type, params, l2Status);
  GetL0Factors(op_type, l2Status, blockValue, l0Status);
  GetL1Factors(op_type, params, l2Status, l0Status, l1Status);
  GetUbFactors(op_type, params, l2Status, l1Status, l0Status, ubStatus);
  CheckSpecialTemplate(op_type, l2Status, l0Status, l1Status);
  tiling.SetParams(l2Status, l0Status, l1Status, ubStatus, params);
  tiling.SetAttachFlag();
  tiling.GetTilingId(params);
  tilingId = tiling.tiling_id;
  OP_LOGD(op_type.c_str(), "the tiling id from cache tiling is: %s", tilingId.c_str());
}
} // namespace optiling
