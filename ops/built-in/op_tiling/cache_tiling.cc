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
static const int64_t NONE = -LLONG_MAX;
static const int64_t CORE_NUM = 32;
static const int64_t L1_Size = (1024 * 1024);
static const int64_t L0c_Size = (256 * 1024);
static const int64_t Ub_Size = (256 * 1024);
static const int32_t Attach_Label_Length = 7;
static const int64_t BLOCK_SIZE = 16;
static const int64_t DECIMAL = 10;
static const int64_t MIN_FRACTAL_SIZE = BLOCK_SIZE * BLOCK_SIZE;
static const int64_t DB_ON = 2;
static const int64_t DB_OFF = 1;
static const int64_t IDX_ZERO = 0;
static const int64_t IDX_ONE = 1;
static const int64_t IDX_TWO = 2;
static const int64_t IDX_THREE = 3;
static const int64_t IDX_FOUR = 4;
static const int64_t IDX_FIVE = 5;
static const int64_t IDX_SIX = 6;
static const int64_t IDX_SEVEN = 7;
static const int64_t IDX_EIGHT = 8;
static const int64_t ATTACH_FLAG_ZERO = 0;
static const int64_t ATTACH_FLAG_ONE = 1;
static const int64_t ATTACH_FLAG_TWO = 2;
static const int64_t KBYTES = 1024;
static const int64_t MAX_FACTOR = 128;
static const int64_t FP16_BYTES = 2;
static const int64_t MIN_MTE1_LOAD = 32;
static const int64_t L0_PARAS_COMBO_LEN = 8;
static const int64_t LOAD_SIZE_RANGE_LOW = 1000;
static const int64_t LOAD_SIZE_RANGE_HIGH = 4000;
static const int64_t LOAD_SIZE_DIFF_RANGE = 400;
static const int64_t M_LOW_RANGE = 5;
static const int64_t M_HIGH_RANGE = 6;
static const double BLOCKING_PCT_GATE = 0.5;
static const double LOAD_SIZE_GATE = 0.13;
static const int64_t CORE_USE_LOW_RANGE = 5;
static const int64_t CORE_USE_HIGH_RANGE = 9;
void Tiling::SetDoubleBufferParams(bool minKl1CmpKl0, map<string, int64_t> dbFlag)
{
  pingpong["AUB_pbuffer"] = dbFlag["db_aub"];
  pingpong["BUB_pbuffer"] = DB_OFF;
  pingpong["AL1_pbuffer"] = dbFlag["db_al1"];
  pingpong["BL1_pbuffer"] = dbFlag["db_bl1"];
  pingpong["AL0_pbuffer"] = DB_ON;
  pingpong["BL0_pbuffer"] = DB_ON;
  pingpong["CL0_pbuffer"] = dbFlag["db_l0c"];
  pingpong["CUB_pbuffer"] = dbFlag["db_cub"];
  pingpong["UBG_pbuffer"] = DB_OFF;

  if (minKl1CmpKl0) {
    pingpong["min_kl1_cmp_kl0"] = 0;
  } else {
    pingpong["min_kl1_cmp_kl0"] = 1;
  }
}

void Tiling::SetParams(const L2Status &l2Status, const L0Status &l0Status, const L1Status &l1Status,
                       const UbStatus &ubStatus)
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
  k_org_dim = l2Status.k * BLOCK_SIZE;
  db_l0c = l0Status.db_l0c;
  k_aub = ubStatus.k_aub;
  m_aub = ubStatus.m_aub;
  db_aub = ubStatus.db_aub;
  mParam.emplace("block_dim", move(vector<int64_t>({batch_dim, n_dim, m_dim, 1})));
  mParam.emplace("AL0_matrix", move(vector<int64_t>({m_l0, k_l0, BLOCK_SIZE, BLOCK_SIZE, 1, 1})));
  mParam.emplace("BL0_matrix", move(vector<int64_t>({k_l0, n_l0, BLOCK_SIZE, BLOCK_SIZE, 1, 1})));
  mParam.emplace("CL0_matrix", move(vector<int64_t>({n_l0, m_l0, BLOCK_SIZE, BLOCK_SIZE, 1, 1})));
  mParam.emplace("CUB_matrix", move(vector<int64_t>({n_cub, m_l0, BLOCK_SIZE, BLOCK_SIZE, 1, 1})));

  mParam["BUB_shape"] = {};
  mParam.emplace("AL1_shape", move(vector<int64_t>({kal1_16 * BLOCK_SIZE, m_al1, 1, 1})));
  mParam.emplace("BL1_shape", move(vector<int64_t>({kbl1_16 * BLOCK_SIZE, n_bl1, 1, 1})));
  mParam.emplace("AUB_shape", move(vector<int64_t>({k_aub * BLOCK_SIZE, m_aub, 1, 1})));

  mParam["n_bef_batch_flag"] = {0};
  mParam["n_bef_group_flag"] = {0};
  mParam["batch_bef_group_flag"] = {0};
  mParam["A_overhead_opt_flag"] = {0};
  mParam["B_overhead_opt_flag"] = {0};
  mParam["AUB_channel_wise_flag"] = {};
  mParam["BUB_channel_wise_flag"] = {};
  mParam["CUB_channel_wise_flag"] = {};

  if (m_al1 == NONE) {
    mParam["AL1_shape"] = vector<int64_t>({});
    db_al1 = 1;
  }
  if (n_bl1 == NONE) {
    mParam["BL1_shape"] = vector<int64_t>({});
    db_bl1 = 1;
  }
  map<string, int64_t> dbFlag = {
    {"db_aub", db_aub}, {"db_al1", db_al1}, {"db_bl1", db_bl1}, {"db_l0c", db_l0c}, {"db_cub", db_cub}};
  SetDoubleBufferParams(min(kal1_16, kbl1_16) == k_l0, dbFlag);
}

void Tiling::SetAttachFlag()
{
  // find kernel ID
  bool kAl1FullLoad = kal1_16 * BLOCK_SIZE == k_org_dim;
  bool kBl1FullLoad = kbl1_16 * BLOCK_SIZE == k_org_dim;
  bool template1 = m_al1 == NONE && n_bl1 == NONE;
  bool template2 = m_al1 == NONE && n_bl1 != NONE && kBl1FullLoad;
  bool template3 = m_al1 == NONE && n_bl1 != NONE && !kBl1FullLoad;
  bool template4 = m_al1 != NONE && n_bl1 == NONE && kAl1FullLoad;
  bool template5 = m_al1 != NONE && n_bl1 == NONE && !kAl1FullLoad;
  bool template6 = m_al1 != NONE && n_bl1 != NONE && kAl1FullLoad && kBl1FullLoad;
  bool template7 = m_al1 != NONE && n_bl1 != NONE && kAl1FullLoad && !kBl1FullLoad;
  bool template8 = m_al1 != NONE && n_bl1 != NONE && !kAl1FullLoad && kBl1FullLoad;
  bool template9 = m_al1 != NONE && n_bl1 != NONE && !kAl1FullLoad && !kBl1FullLoad;
  if (template1 || template2 || template3) {
    pingpong["al1_attach_flag"] = ATTACH_FLAG_ZERO;
  }
  if (template4 || template6 || template7) {
    pingpong["al1_attach_flag"] = ATTACH_FLAG_ONE;
  }
  if (template5 || template8 || template9) {
    pingpong["al1_attach_flag"] = ATTACH_FLAG_TWO;
  }
  if (template1 || template4 || template5) {
    pingpong["bl1_attach_flag"] = ATTACH_FLAG_ZERO;
  }
  if (template2 || template6 || template8) {
    pingpong["bl1_attach_flag"] = ATTACH_FLAG_ONE;
  }
  if (template3 || template7 || template9) {
    pingpong["bl1_attach_flag"] = ATTACH_FLAG_TWO;
  }
  if (template1 || template2 || template4 || template6) {
    pingpong["abkl1_attach_flag"] = ATTACH_FLAG_ZERO;
  }
  if (template3 || template7) {
    pingpong["abkl1_attach_flag"] = ATTACH_FLAG_ONE;
  }
  if (template5 || template8) {
    pingpong["abkl1_attach_flag"] = ATTACH_FLAG_TWO;
  }
  if (template9) {
    if (kal1_16 == kbl1_16) {
      pingpong["abkl1_attach_flag"] = ATTACH_FLAG_ZERO;
    } else if (kal1_16 > kbl1_16) {
      pingpong["abkl1_attach_flag"] = ATTACH_FLAG_ONE;
    } else if (kal1_16 < kbl1_16) {
      pingpong["abkl1_attach_flag"] = ATTACH_FLAG_TWO;
    }
  }
  mPingpongBuff["manual_pingpong_buffer"] = pingpong;
}

void Tiling::GetTilingId()
{
  string tilingKeywords[Attach_Label_Length] = {"AL1_pbuffer", "BL1_pbuffer", "CL0_pbuffer",
                                                "abkl1_attach_flag", "al1_attach_flag", "bl1_attach_flag",
                                                "min_kl1_cmp_kl0"};
  int64_t tilingIDLongLong = 0;
  for (const auto &tilingKeyword: tilingKeywords) {
    tilingIDLongLong = tilingIDLongLong * DECIMAL + this->mPingpongBuff["manual_pingpong_buffer"][tilingKeyword];
  }
  this->tiling_id = to_string(tilingIDLongLong);
}

void GetFactors(int64_t *cnt, int64_t *factorList, const int64_t &num, const int64_t &maxNum)
{
  // get all factors of num which smaller or equal to maxNum
  for (int64_t i = 1; i < maxNum + 1; i++) {
    if (num % i == 0) {
      factorList[(*cnt)++] = i;
    }
  }
}

void GetTwoFactors(int64_t *res, const int64_t &base, const int64_t &dim, const int64_t &maxNum = 32)
{
  // for up bigger or equal to base + 1, find the smallest num which is a factor of dim
  // form down smaller or equal to base, find the biggest num which is a factor of dim
  int64_t cnt = 0;
  int64_t up = base + 1;
  int64_t maxCnt = 2;
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
  int64_t down = base;
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

void GetNearestFactor(const int64_t &base, int64_t &factor)
{
  while (factor > 0 && base % factor != 0) {
    factor--;
  }
}

void BL1FullLoadBlock(const L2Status &l2Status, BlockDimCalculator &blockDimCalculator, int64_t &n0)
{
  if (n0 >= 1) {
    while (l2Status.n % n0 != 0) {
      n0--;
    }
    blockDimCalculator.amat_size = blockDimCalculator.ori_amat_size * (l2Status.n / n0);
    blockDimCalculator.bmat_size = l2Status.n;
    blockDimCalculator.total_load_size = blockDimCalculator.amat_size + blockDimCalculator.bmat_size;
    blockDimCalculator.tmp_value = n0;
  }
}

void AL1FullLoadBlock(const L2Status &l2Status, BlockDimCalculator &blockDimCalculator, int64_t &m0)
{
  if (m0 >= 1) {
    while (l2Status.m % m0 != 0) {
      m0--;
    }
    blockDimCalculator.tmp_amat_size = blockDimCalculator.ori_amat_size;
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

void NeitherFullLoadBlock(const L2Status &l2Status, BlockDimCalculator &blockDimCalculator,
                          const int64_t nFactorTwoCandidates[][2], const int64_t mFactorTwoCandidates[][2],
                          const int64_t &nFactor, const int64_t &mFactor)
{
  for (auto const &n0: nFactorTwoCandidates[nFactor]) {
    for (auto const &m0: mFactorTwoCandidates[mFactor]) {
      if (m0 <= 0 || n0 <= 0) {
        continue;
      }
      if (m0 * n0 * KBYTES <= L0c_Size) {
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
}

void GetBlockDimHelper(L2Status &l2Status, BlockDimCalculator &blockDimCalculator, const int64_t m0s[][2],
                       const int64_t n0s[][2])
{
  int64_t iIdx = blockDimCalculator.i_idx;
  int64_t jIdx = blockDimCalculator.j_idx;
  int64_t bFactor = blockDimCalculator.batch_dim_array[iIdx];
  int64_t nFactor = blockDimCalculator.n_dim_array[jIdx];
  for (int64_t mIdx = 0; mIdx < blockDimCalculator.m_dim_cnt; mIdx++) {
    int64_t mFactor = blockDimCalculator.m_dim_array[mIdx];
    if (bFactor * nFactor * mFactor > CORE_NUM) {
      break;
    }
    l2Status.batch = blockDimCalculator.batch / bFactor;
    l2Status.m = blockDimCalculator.m / mFactor;
    l2Status.n = blockDimCalculator.n / nFactor;
    // load size of A matrix is batch * m
    // load size of B matrix is n
    blockDimCalculator.ori_amat_size = l2Status.batch * l2Status.m;
    blockDimCalculator.ori_bmat_size = l2Status.n;
    blockDimCalculator.amat_size = blockDimCalculator.ori_amat_size;
    blockDimCalculator.bmat_size = blockDimCalculator.ori_bmat_size;
    blockDimCalculator.total_load_size = blockDimCalculator.amat_size + blockDimCalculator.bmat_size;
    blockDimCalculator.tmp_value = 0;
    if (blockDimCalculator.total_load_size * blockDimCalculator.k_bytes > L1_Size) {
      blockDimCalculator.total_load_size = LONG_LONG_MAX;
      // BL1 full load
      int64_t n0 =
        min(min((L1_Size / FP16_BYTES - MIN_FRACTAL_SIZE) / blockDimCalculator.k_num, l2Status.n), MAX_FACTOR);
      BL1FullLoadBlock(l2Status, blockDimCalculator, n0);
      // AL1 full load
      int64_t m0 = min(min((L1_Size / FP16_BYTES - MIN_FRACTAL_SIZE) /
                             (MIN_FRACTAL_SIZE * blockDimCalculator.k * blockDimCalculator.ori_amat_size),
                           blockDimCalculator.ori_amat_size),
                       MAX_FACTOR);
      AL1FullLoadBlock(l2Status, blockDimCalculator, m0);
      // neither full load max_m max_n
      // closest m and n
      NeitherFullLoadBlock(l2Status, blockDimCalculator, n0s, m0s, nFactor, mFactor);
    }
    int64_t loadSizeKb = blockDimCalculator.total_load_size * blockDimCalculator.k_bytes / KBYTES;
    int64_t minLoadSizeKb = blockDimCalculator.min_load_size * blockDimCalculator.k_bytes / KBYTES;
    double tmpBlockingPct;
    if (nFactor > mFactor) {
      tmpBlockingPct = double(blockDimCalculator.amat_size) / blockDimCalculator.total_load_size;
    } else if (nFactor < mFactor) {
      tmpBlockingPct = double(blockDimCalculator.bmat_size) / blockDimCalculator.total_load_size;
    } else {
      tmpBlockingPct =
        double(max(blockDimCalculator.amat_size, blockDimCalculator.bmat_size)) / blockDimCalculator.total_load_size;
    }
    bool tmpBlockingFlag = (loadSizeKb < LOAD_SIZE_RANGE_LOW && max(nFactor, mFactor) > M_LOW_RANGE);

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
        (blockDimCalculator.final_blocking_flag && (loadSizeKb - minLoadSizeKb) < LOAD_SIZE_DIFF_RANGE &&
          max(nFactor, mFactor) < max(blockDimCalculator.n_dim_factor, blockDimCalculator.m_dim_factor));
    auto noUpdateSolution =
      (((loadSizeKb >= LOAD_SIZE_RANGE_LOW && loadSizeKb < LOAD_SIZE_RANGE_HIGH &&
        max(nFactor, mFactor) > M_HIGH_RANGE && tmpBlockingPct > BLOCKING_PCT_GATE) &&
        max(nFactor, mFactor) > max(blockDimCalculator.n_dim_factor, blockDimCalculator.m_dim_factor) &&
        double(blockDimCalculator.min_load_size - blockDimCalculator.total_load_size) /
          blockDimCalculator.min_load_size <
          LOAD_SIZE_GATE &&
        blockDimCalculator.core_use >= CORE_USE_HIGH_RANGE) ||
        ((loadSizeKb < LOAD_SIZE_RANGE_LOW && max(nFactor, mFactor) > M_LOW_RANGE) &&
          (max(nFactor, mFactor) > max(blockDimCalculator.n_dim_factor, blockDimCalculator.m_dim_factor)) &&
          ((minLoadSizeKb - loadSizeKb) < LOAD_SIZE_DIFF_RANGE && blockDimCalculator.core_use > CORE_USE_LOW_RANGE)));
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

int64_t GetBlockDim(const string &op_type, const BatchmatmulParas &params, L2Status &l2Status,
                    const int64_t &coreNum)
{
  // get batch_dim, m_dim and n_dim for single core
  // not support multi cores slicing along k dim
  // single core batch_dim, m_dim, n_dim is a factor of input batch, m, n

  OP_LOGD(op_type.c_str(), "input batch dim:%lld", params.batch);
  OP_LOGD(op_type.c_str(), "input m dim:%lld", params.m);
  OP_LOGD(op_type.c_str(), "input k dim:%lld", params.k);
  OP_LOGD(op_type.c_str(), "input n dim:%lld", params.n);
  if (params.batch * params.m * params.n < coreNum) {
    l2Status.batch_dim = params.batch;
    l2Status.n_dim = params.n;
    l2Status.m_dim = params.m;
    l2Status.batch = 1;
    l2Status.m = 1;
    l2Status.k = params.k;
    l2Status.n = 1;
    OP_LOGD(op_type.c_str(), "singlecore batch dim factor:%lld", params.batch);
    OP_LOGD(op_type.c_str(), "singlecore batch n dim factor:%lld", params.n);
    OP_LOGD(op_type.c_str(), "singlecore m dim factor:%lld", params.m);
    OP_LOGD(op_type.c_str(), "singlecore m block pnt point:%lld", 0);
    return 0;
  }
  BlockDimCalculator blockDimCalculator;
  blockDimCalculator.batch = params.batch;
  blockDimCalculator.m = params.m;
  blockDimCalculator.k = params.k;
  blockDimCalculator.n = params.n;
  int64_t batchDimArray[coreNum] = {0};
  int64_t nDimArray[coreNum] = {0};
  int64_t mDimArray[coreNum] = {0};
  GetFactors(&blockDimCalculator.batch_dim_cnt, batchDimArray, params.batch, coreNum);
  GetFactors(&blockDimCalculator.n_dim_cnt, nDimArray, params.n, coreNum);
  GetFactors(&blockDimCalculator.m_dim_cnt, mDimArray, params.m, coreNum);
  int64_t mnMaxPnt = 16;
  int64_t m0s[coreNum + 1][2] = {0};
  int64_t n0s[coreNum + 1][2] = {0};
  for (int64_t idx = 0; idx < blockDimCalculator.n_dim_cnt; idx++) {
    int64_t tmpNDim = nDimArray[idx];
    int64_t tmpNSingleCore = params.n / tmpNDim;
    GetTwoFactors(n0s[tmpNDim], mnMaxPnt, tmpNSingleCore, MAX_FACTOR);
  }
  for (int64_t idx = 0; idx < blockDimCalculator.m_dim_cnt; idx++) {
    int64_t tmpMDim = mDimArray[idx];
    int64_t tmpMSingleCore = params.m / tmpMDim;
    GetTwoFactors(m0s[tmpMDim], mnMaxPnt, tmpMSingleCore, MAX_FACTOR);
  }
  blockDimCalculator.k_num = params.k * BLOCK_SIZE * BLOCK_SIZE;
  blockDimCalculator.k_bytes = blockDimCalculator.k_num * FP16_BYTES;
  blockDimCalculator.n_dim_factor = 1;
  blockDimCalculator.batch_dim_factor = 1;
  blockDimCalculator.m_dim_factor = 1;
  blockDimCalculator.min_load_size = L1_Size / FP16_BYTES;
  blockDimCalculator.batch_dim_array = batchDimArray;
  blockDimCalculator.m_dim_array = mDimArray;
  blockDimCalculator.n_dim_array = nDimArray;
  for (int64_t iIdx = 0; iIdx < blockDimCalculator.batch_dim_cnt; iIdx++) {
    for (int64_t jIdx = 0; jIdx < blockDimCalculator.n_dim_cnt; jIdx++) {
      blockDimCalculator.i_idx = iIdx;
      blockDimCalculator.j_idx = jIdx;
      GetBlockDimHelper(l2Status, blockDimCalculator, m0s, n0s);
    }
  }
  l2Status.batch_dim = blockDimCalculator.batch_dim_factor;
  l2Status.n_dim = blockDimCalculator.n_dim_factor;
  l2Status.m_dim = blockDimCalculator.m_dim_factor;
  l2Status.m = params.m / blockDimCalculator.m_dim_factor;
  l2Status.n = params.n / blockDimCalculator.n_dim_factor;
  l2Status.k = params.k;
  l2Status.batch = params.batch / blockDimCalculator.batch_dim_factor;
  OP_LOGD(op_type.c_str(), "singlecore batch dim factor:%lld", l2Status.batch_dim);
  OP_LOGD(op_type.c_str(), "singlecore batch n dim factor:%lld", l2Status.n_dim);
  OP_LOGD(op_type.c_str(), "singlecore m dim factor:%lld", l2Status.m_dim);
  OP_LOGD(op_type.c_str(), "singlecore m block pnt point:%lld", blockDimCalculator.final_value);
  return blockDimCalculator.final_value;
}

void CheckUbDb(L0Status &l0Status)
{
  int64_t nCub = 1;
  int64_t c0 = 16;
  int64_t ubFp16Size = Ub_Size / FP16_BYTES;
  int64_t copyoutSize = nCub * l0Status.m_l0 * c0;
  if (copyoutSize * l0Status.db_cub > ubFp16Size) {
    l0Status.db_cub = 1;
  }
}

int64_t GetLoadSize(const L2Status &l2Status, const L0Status &l0Status)
{
  bool al1FullLoad =
    ((l2Status.m * l2Status.k + l0Status.n_l0 * l0Status.k_l0) * BLOCK_SIZE * BLOCK_SIZE * FP16_BYTES <=
      L1_Size);
  bool bl1FullLoad =
    ((l0Status.m_l0 * l0Status.k_l0 + l0Status.n_l0 * l2Status.k) * BLOCK_SIZE * BLOCK_SIZE * FP16_BYTES <=
      L1_Size);
  bool bothFullLoad = ((l2Status.m * l2Status.k + l0Status.n_l0 * l2Status.k) * BLOCK_SIZE *
    BLOCK_SIZE * FP16_BYTES <=
    L1_Size);
  int64_t num0a =
    bl1FullLoad ? l2Status.n : ((l2Status.m + l0Status.m_l0 - 1) / l0Status.m_l0) * l2Status.n;
  int64_t num0b =
    al1FullLoad ? l2Status.m : ((l2Status.n + l0Status.n_l0 - 1) / l0Status.n_l0) * l2Status.m;
  if ((al1FullLoad && bl1FullLoad) && !bothFullLoad) {
    return min(l2Status.n + ((l2Status.n + l0Status.n_l0 - 1) / l0Status.n_l0) * l2Status.m,
               l2Status.m + ((l2Status.m + l0Status.m_l0 - 1) / l0Status.m_l0) * l2Status.n);
  }
  return num0a + num0b;
}

void GetFinalMkn(L0Status &l0Status, const L2Status &l2Status)
{
  int64_t tmpL0cUse = l0Status.m_l0 * l0Status.n_l0 * l0Status.db_l0c * BLOCK_SIZE * BLOCK_SIZE * 4 * 100 / L0c_Size;
  int64_t tmpMte1Loop = ((l0Status.n_l0 != 1) ? l0Status.k_l0 : 1) + ((l0Status.k_l0 != 1) ? l0Status.m_l0 : 1);
  int64_t tmpMul = l0Status.m_l0 * l0Status.k_l0 * l0Status.n_l0;
  int64_t tmpLoadSize = GetLoadSize(l2Status, l0Status);
  auto condition1 = l0Status.final_ml0 == 0;
  auto condition2 = tmpLoadSize < l0Status.final_load_size;
  auto condition3 = tmpLoadSize == l0Status.final_load_size && tmpMul > l0Status.final_mul;
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

MKNParasCombo GetParasCombo(const int64_t &index, const int64_t &blockValue)
{
  map<int64_t, MKNParasCombo> parasComboMap;
  if (blockValue == 0) {
    MKNParasCombo comboZero = {2, 2, 2, 64, 64, 128, 0, 64, 11};
    MKNParasCombo comboOne = {2, 2, 1, 64, 64, 256, 0, 64, 16};
    MKNParasCombo comboTwo = {2, 1, 2, 64, 128, 128, 1, 64, 16};
    MKNParasCombo comboThree = {1, 2, 2, 128, 64, 128, 0, 128, 16};
    MKNParasCombo comboFour = {1, 1, 2, 128, 128, 128, 0, 128, 11};
    MKNParasCombo comboFive = {1, 2, 1, 128, 64, 256, 0, 128, 22};
    MKNParasCombo comboSix = {2, 1, 1, 64, 128, 256, 1, 128, 22};
    MKNParasCombo comboSeven = {1, 1, 1, 128, 128, 256, 0, 128, 16};
    parasComboMap = {{0, comboZero}, {1, comboOne}, {2, comboTwo}, {3, comboThree},
                     {4, comboFour}, {5, comboFive}, {6, comboSix}, {7, comboSeven}};
  } else {
    MKNParasCombo comboZero = {2, 2, 2, 64, 64, 128, 1, 64, blockValue};
    MKNParasCombo comboOne = {2, 2, 1, 64, 64, 256, 1, 64, blockValue};
    MKNParasCombo comboTwo = {2, 1, 2, 64, 128, 128, 1, 128, blockValue};
    MKNParasCombo comboThree = {1, 2, 2, 128, 64, 128, 1, 64, blockValue};
    MKNParasCombo comboFour = {1, 1, 2, 128, 128, 128, 1, 128, blockValue};
    MKNParasCombo comboFive = {1, 2, 1, 128, 64, 256, 1, 128, blockValue};
    MKNParasCombo comboSix = {2, 1, 1, 64, 128, 256, 1, 128, blockValue};
    MKNParasCombo comboSeven = {1, 1, 1, 128, 128, 256, 1, 128, blockValue};
    parasComboMap = {{0, comboZero}, {1, comboOne}, {2, comboTwo}, {3, comboThree},
                     {4, comboFour}, {5, comboFive}, {6, comboSix}, {7, comboSeven}};
  }
  return parasComboMap[index];
}

void GetL0StatusFromParasCombo(L0Status &l0Status, int64_t *parasCombo)
{
  l0Status.SetInitLoadStatus();
  l0Status.db_l0a = parasCombo[IDX_ZERO];
  l0Status.db_l0b = parasCombo[IDX_ONE];
  l0Status.db_l0c = parasCombo[IDX_TWO];
  l0Status.max_mk = parasCombo[IDX_THREE];
  l0Status.max_nk = parasCombo[IDX_FOUR];
  l0Status.max_mn = parasCombo[IDX_FIVE];
  l0Status.max_axis_idx = parasCombo[IDX_SIX];
  l0Status.max_axis_num = parasCombo[IDX_SEVEN];
  l0Status.max_axis_pnt = parasCombo[IDX_EIGHT];
  l0Status.max_axis_pnt = min(l0Status.max_axis_pnt, l0Status.max_axis_num);
}

void SetResFactors(int64_t *resFactors, const L0Status &l0Status)
{
  resFactors[IDX_ZERO] = l0Status.final_ml0;
  resFactors[IDX_ONE] = l0Status.final_kl0;
  resFactors[IDX_TWO] = l0Status.final_nl0;
  resFactors[IDX_THREE] = l0Status.final_load_size;
  resFactors[IDX_FOUR] = l0Status.final_l0c_use;
  resFactors[IDX_FIVE] = l0Status.final_mte1Loop;
  resFactors[IDX_SIX] = l0Status.final_mul;
}

void GetL0FactorsCand(int64_t *resFactors, const L2Status &l2Status, L0Status &l0Status,
                      int64_t *parasCombo, int64_t sizeofParasCombo)
{
  GetL0StatusFromParasCombo(l0Status, parasCombo);
  int64_t majorDim = l2Status.m;
  int64_t minorDim = l2Status.n;
  int64_t majorDimK = l0Status.max_mk;
  int64_t minorDimK = l0Status.max_nk;
  if (l0Status.max_axis_idx != 0) {
    majorDim = l2Status.n;
    minorDim = l2Status.m;
    majorDimK = l0Status.max_nk;
    minorDimK = l0Status.max_mk;
  }
  int64_t majorDimFactors[2] = {0};
  GetTwoFactors(majorDimFactors, l0Status.max_axis_pnt, majorDim, l0Status.max_axis_num);
  for (auto &majorDimFactor: majorDimFactors) {
    if (majorDimFactor == 0) {
      continue;
    }
    int64_t minorFactorMax = min(l0Status.max_mn / majorDimFactor, minorDimK);
    int64_t minorDimFactors[2] = {0};
    GetTwoFactors(minorDimFactors, minorFactorMax, minorDim, minorFactorMax);
    for (auto &minorDimFactor: minorDimFactors) {
      if (minorDimFactor == 0) {
        continue;
      }
      int64_t k0Max = min(majorDimK / majorDimFactor, minorDimK / minorDimFactor);
      int64_t k0Factors[2] = {0};
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

void GetL0Factors(const string &op_type, const L2Status &l2Status, const int64_t &blockValue,
                  L0Status &l0Status)
{
  // get m_l0, n_l0, k_l0 factor when singlecore m, n, k is know
  // m_l0, n_l0, k_l0 is a factor of single core m, n, k

  int64_t dbAOnBOnCOnIdx = 0;
  int64_t dbAOnBOnCOffIdx = 1;
  int64_t resFactors[8][7] = {0};
  int64_t parasCombo[9];
  for (int32_t i = 0; i < L0_PARAS_COMBO_LEN; ++i) {
    MKNParasCombo mknParasCombo = GetParasCombo(i, blockValue);
    GetL0FactorsCand(resFactors[i], l2Status, l0Status, mknParasCombo.parasCombo,
                     sizeof(mknParasCombo.parasCombo));
  }

  // check both L0C utilization and loadsize to control LOC LOA LOB DB
  int64_t dbL0aL0cDbOn = 2;
  int64_t dbL0bL0cDbOn = 2;
  int64_t m0L0cDbOn = resFactors[dbAOnBOnCOnIdx][IDX_ZERO];
  int64_t k0L0cDbOn = resFactors[dbAOnBOnCOnIdx][IDX_ONE];
  int64_t n0L0cDbOn = resFactors[dbAOnBOnCOnIdx][IDX_TWO];
  int64_t loadSizeL0cDbOn = resFactors[dbAOnBOnCOnIdx][IDX_THREE];
  int64_t l0cUseL0cDbOn = resFactors[dbAOnBOnCOnIdx][IDX_FOUR];

  int64_t dbL0aL0cDbOff = 2;
  int64_t dbL0bL0cDbOff = 2;
  int64_t m0L0cDbOff = resFactors[dbAOnBOnCOffIdx][IDX_ZERO];
  int64_t k0L0cDbOff = resFactors[dbAOnBOnCOffIdx][IDX_ONE];
  int64_t n0L0cDbOff = resFactors[dbAOnBOnCOffIdx][IDX_TWO];
  int64_t loadSizeL0cDbOff = resFactors[dbAOnBOnCOffIdx][IDX_THREE];
  int64_t l0cUseL0cDbOff = resFactors[dbAOnBOnCOffIdx][IDX_FOUR];

  if ((l0cUseL0cDbOff > l0cUseL0cDbOn) || (loadSizeL0cDbOff < loadSizeL0cDbOn)) {
    l0Status.db_l0c = DB_OFF;
    l0Status.db_l0a = dbL0aL0cDbOff;
    l0Status.db_l0b = dbL0bL0cDbOff;
    l0Status.m_l0 = m0L0cDbOff;
    l0Status.k_l0 = k0L0cDbOff;
    l0Status.n_l0 = n0L0cDbOff;
  } else {
    l0Status.db_l0c = DB_ON;
    l0Status.db_l0a = dbL0aL0cDbOn;
    l0Status.db_l0b = dbL0bL0cDbOn;
    l0Status.m_l0 = m0L0cDbOn;
    l0Status.k_l0 = k0L0cDbOn;
    l0Status.n_l0 = n0L0cDbOn;
  }
  l0Status.db_cub = DB_ON;
  CheckUbDb(l0Status);
  OP_LOGD(op_type.c_str(), "tiling m_l0:%s, n_l0:%s, k_l0:%s", l0Status.m_l0, l0Status.n_l0, l0Status.k_l0);
  OP_LOGD(op_type.c_str(), "tiling db_l0a:%s, db_l0b:%s, db_l0c:%s", l0Status.db_l0a, l0Status.db_l0b, l0Status.db_l0c);
  OP_LOGD(op_type.c_str(), "tiling db_cub:%s", l0Status.db_cub);
}

int64_t GetL1Size(const L1Status &l1Status, const L0Status &l0Status)
{
  int64_t curL1Size;
  curL1Size =
    l1Status.m_al1 * l0Status.m_l0 * BLOCK_SIZE * l1Status.kal1_16 * BLOCK_SIZE * l1Status.db_al1 * FP16_BYTES +
      l1Status.n_bl1 * l0Status.n_l0 * BLOCK_SIZE * l1Status.kbl1_16 * BLOCK_SIZE * l1Status.db_bl1 * FP16_BYTES;
  return curL1Size;
}

void L1StatusBothFullLoad(const L2Status &l2Status, const L0Status &l0Status, L1Status &l1Status,
                          int64_t res[][7])
{
  int64_t curL1Size;
  curL1Size = GetL1Size(l1Status, l0Status);
  if (curL1Size <= L1_Size) {
    l1Status.both_full_load = true;
    l1Status.load_size = l2Status.m + l2Status.n;
    res[IDX_ZERO][IDX_ZERO] = l1Status.kal1_16;
    res[IDX_ZERO][IDX_ONE] = l1Status.m_al1;
    res[IDX_ZERO][IDX_TWO] = l1Status.db_al1;
    res[IDX_ZERO][IDX_THREE] = l1Status.kbl1_16;
    res[IDX_ZERO][IDX_FOUR] = l1Status.n_bl1;
    res[IDX_ZERO][IDX_FIVE] = l1Status.db_bl1;
    res[IDX_ZERO][IDX_SIX] = l1Status.load_size;
  }
}

void L1StatusAl1FullLoad(const L2Status &l2Status, const L0Status &l0Status, L1Status &l1Status,
                         int64_t res[][7])
{
  int64_t curL1Size;
  int64_t mRepeat = l2Status.m / l0Status.m_l0;
  int64_t nRepeat = l2Status.n / l0Status.n_l0;
  curL1Size = GetL1Size(l1Status, l0Status);
  if (curL1Size <= L1_Size) {
    l1Status.al1_full_load = true;
    l1Status.al1_size = l2Status.k * l2Status.m * BLOCK_SIZE * BLOCK_SIZE * FP16_BYTES;
    l1Status.bl1_size = L1_Size - l1Status.al1_size;
    l1Status.db_bl1 = DB_ON;
    if (GetL1Size(l1Status, l0Status) > L1_Size) {
      l1Status.db_bl1 = DB_OFF;
    }
    l1Status.kbl1_16 = min(
      l1Status.bl1_size / (l1Status.n_bl1 * l0Status.n_l0 * BLOCK_SIZE * l1Status.db_bl1 * FP16_BYTES * BLOCK_SIZE),
      l2Status.k);
    l1Status.bl1_times = min(l1Status.kbl1_16 / l0Status.k_l0, l1Status.max_k_bl1);
    GetNearestFactor(l1Status.all_times, l1Status.bl1_times);
    l1Status.kbl1_16 = l1Status.bl1_times * l0Status.k_l0;
    if (l1Status.kbl1_16 == l2Status.k) {
      l1Status.n_bl1 = min(l1Status.bl1_size / (l1Status.kbl1_16 * l0Status.n_l0 * BLOCK_SIZE * l1Status.db_bl1 *
                             FP16_BYTES * BLOCK_SIZE),
                           l1Status.max_n_bl1);
      GetNearestFactor(nRepeat, l1Status.n_bl1);
    }
    l1Status.load_size = l2Status.m + (l1Status.kbl1_16 == l2Status.k ? 1 : mRepeat) * l2Status.n;
    res[IDX_ONE][IDX_ZERO] = l1Status.kal1_16;
    res[IDX_ONE][IDX_ONE] = l1Status.m_al1;
    res[IDX_ONE][IDX_TWO] = l1Status.db_al1;
    res[IDX_ONE][IDX_THREE] = l1Status.kbl1_16;
    res[IDX_ONE][IDX_FOUR] = l1Status.n_bl1;
    res[IDX_ONE][IDX_FIVE] = l1Status.db_bl1;
    res[IDX_ONE][IDX_SIX] = l1Status.load_size;
  }
}

void L1StatusBl1FullLoad(const L2Status &l2Status, const L0Status &l0Status, L1Status &l1Status,
                         int64_t res[][7])
{
  int64_t curL1Size;
  int64_t mRepeat = l2Status.m / l0Status.m_l0;
  int64_t nRepeat = l2Status.n / l0Status.n_l0;
  curL1Size = GetL1Size(l1Status, l0Status);
  if (curL1Size <= L1_Size) {
    l1Status.bl1_full_load = true;
    l1Status.bl1_size = l2Status.k * l2Status.n * BLOCK_SIZE * BLOCK_SIZE * FP16_BYTES;
    l1Status.al1_size = L1_Size - l1Status.bl1_size;
    l1Status.db_al1 = DB_ON;
    if (GetL1Size(l1Status, l0Status) > L1_Size) {
      l1Status.db_al1 = DB_OFF;
    }
    l1Status.kal1_16 = min(
      l1Status.al1_size / (l1Status.m_al1 * l0Status.m_l0 * BLOCK_SIZE * l1Status.db_al1 * FP16_BYTES * BLOCK_SIZE),
      l2Status.k);
    l1Status.al1_times = min(l1Status.kal1_16 / l0Status.k_l0, l1Status.max_k_al1);
    GetNearestFactor(l1Status.all_times, l1Status.al1_times);
    l1Status.kal1_16 = l1Status.al1_times * l0Status.k_l0;
    if (l1Status.kal1_16 == l2Status.k) {
      l1Status.m_al1 = min(l1Status.al1_size / (l1Status.kal1_16 * l0Status.m_l0 * BLOCK_SIZE * l1Status.db_al1 *
                             FP16_BYTES * BLOCK_SIZE),
                           l1Status.max_m_al1);
      GetNearestFactor(mRepeat, l1Status.m_al1);
    }
    l1Status.load_size =
      l2Status.n +
        ((l2Status.m == l1Status.m_al1 * l0Status.m_l0 && l1Status.kal1_16 == l2Status.k) ? 1 : nRepeat) *
          l2Status.m;
    res[IDX_TWO][IDX_ZERO] = l1Status.kal1_16;
    res[IDX_TWO][IDX_ONE] = l1Status.m_al1;
    res[IDX_TWO][IDX_TWO] = l1Status.db_al1;
    res[IDX_TWO][IDX_THREE] = l1Status.kbl1_16;
    res[IDX_TWO][IDX_FOUR] = l1Status.n_bl1;
    res[IDX_TWO][IDX_FIVE] = l1Status.db_bl1;
    res[IDX_TWO][IDX_SIX] = l1Status.load_size;
  }
}

void NeitherFullLoadDb(const L2Status &l2Status, const L0Status &l0Status, L1Status &l1Status,
                       const int64_t &kbl1Db)
{
  int64_t tmpKbl116 = l1Status.kbl1_16;
  l1Status.kbl1_16 = kbl1Db;
  if (GetL1Size(l1Status, l0Status) > L1_Size) {
    l1Status.db_bl1 = DB_OFF;
    if (GetL1Size(l1Status, l0Status) > L1_Size) {
      l1Status.db_al1 = DB_OFF;
    }
  }
  l1Status.kbl1_16 = l2Status.k;
  bool bothDoubleBuffer = l2Status.m != l0Status.m_l0 && l2Status.k > l0Status.k_l0 &&
    GetL1Size(l1Status, l0Status) > L1_Size;
  l1Status.kbl1_16 = tmpKbl116;
  if (bothDoubleBuffer) {
    l1Status.db_al1 = DB_ON;
    l1Status.db_bl1 = DB_ON;
    if (GetL1Size(l1Status, l0Status) > L1_Size) {
      l1Status.db_bl1 = DB_OFF;
      if (GetL1Size(l1Status, l0Status) > L1_Size) {
        l1Status.db_al1 = DB_OFF;
      }
    }
  }
}

void NeitherFullLoadMN(const L2Status &l2Status, const L0Status &l0Status, L1Status &l1Status,
                       const BatchmatmulParas &params)
{
  int64_t mRepeat = l2Status.m / l0Status.m_l0;
  int64_t nRepeat = l2Status.n / l0Status.n_l0;
  if (l0Status.k_l0 == params.k) {
    if (params.m > params.n) {
      l1Status.bl1_size = params.k * l0Status.n_l0 * BLOCK_SIZE * BLOCK_SIZE * l1Status.db_bl1 * FP16_BYTES;
      l1Status.al1_size = L1_Size - l1Status.bl1_size;
      l1Status.m_al1 = min(l1Status.al1_size / (l1Status.kal1_16 * l0Status.m_l0 * BLOCK_SIZE * l1Status.db_al1 *
                             FP16_BYTES * BLOCK_SIZE),
                           l1Status.max_m_al1);
      GetNearestFactor(mRepeat, l1Status.m_al1);
      l1Status.al1_size =
        l1Status.kal1_16 * l1Status.m_al1 * l0Status.m_l0 * BLOCK_SIZE * BLOCK_SIZE * l1Status.db_al1 * FP16_BYTES;
      l1Status.bl1_size = L1_Size - l1Status.al1_size;
      l1Status.n_bl1 = min(l1Status.bl1_size / (l1Status.kbl1_16 * l0Status.n_l0 * BLOCK_SIZE * l1Status.db_bl1 *
                             FP16_BYTES * BLOCK_SIZE),
                           l1Status.max_n_bl1);
      GetNearestFactor(nRepeat, l1Status.n_bl1);
    } else {
      l1Status.al1_size = params.k * l0Status.m_l0 * BLOCK_SIZE * BLOCK_SIZE * l1Status.db_al1 * FP16_BYTES;
      l1Status.bl1_size = L1_Size - l1Status.al1_size;
      l1Status.n_bl1 = min(l1Status.bl1_size / (l1Status.kbl1_16 * l0Status.n_l0 * BLOCK_SIZE * l1Status.db_bl1 *
                             FP16_BYTES * BLOCK_SIZE),
                           l1Status.max_n_bl1);
      GetNearestFactor(nRepeat, l1Status.n_bl1);
      l1Status.bl1_size = params.k * l0Status.n_l0 * BLOCK_SIZE * BLOCK_SIZE * l1Status.db_bl1 * FP16_BYTES;
      l1Status.al1_size = L1_Size - l1Status.bl1_size;
      l1Status.m_al1 = min(l1Status.al1_size / (l1Status.kal1_16 * l0Status.m_l0 * BLOCK_SIZE * l1Status.db_al1 *
                             FP16_BYTES * BLOCK_SIZE),
                           l1Status.max_m_al1);
      GetNearestFactor(mRepeat, l1Status.m_al1);
    }
  }
}

void NeitherFullLoadK(const L2Status &l2Status, const L0Status &l0Status, L1Status &l1Status,
                      const BatchmatmulParas &params)
{
  l1Status.kbl1_16 = params.k;
  if (GetL1Size(l1Status, l0Status) <= L1_Size) {
    l1Status.bl1_size = params.k * l0Status.n_l0 * BLOCK_SIZE * BLOCK_SIZE * l1Status.db_bl1 * FP16_BYTES;
    l1Status.al1_size = L1_Size - l1Status.bl1_size;
    l1Status.kal1_16 = min(
      l1Status.al1_size / (l1Status.m_al1 * l0Status.m_l0 * BLOCK_SIZE * l1Status.db_al1 * FP16_BYTES * BLOCK_SIZE),
      l2Status.k);
    l1Status.al1_times = min(l1Status.kal1_16 / l0Status.k_l0, l1Status.max_k_al1);
    GetNearestFactor(l1Status.all_times, l1Status.al1_times);
    l1Status.kal1_16 = l1Status.al1_times * l0Status.k_l0;
  } else {
    int64_t perK = min(L1_Size /
                         (l0Status.m_l0 * BLOCK_SIZE * BLOCK_SIZE * l1Status.db_al1 * FP16_BYTES +
                           BLOCK_SIZE * l0Status.n_l0 * BLOCK_SIZE * l1Status.db_bl1 * FP16_BYTES) /
                         l0Status.k_l0 * l0Status.k_l0,
                       l2Status.k);
    int64_t perTimes = min(perK / l0Status.k_l0, max(l1Status.max_k_al1, l1Status.max_k_bl1));
    GetNearestFactor(l1Status.all_times, perTimes);
    perK = perTimes * l0Status.k_l0;
    l1Status.kal1_16 = perK;
    l1Status.kbl1_16 = perK;
  }
}

void L1StatusNeitherFullLoad(const L2Status &l2Status, const BatchmatmulParas &params,
                             const L0Status &l0Status, L1Status &l1Status, int64_t res[][7])
{
  int64_t mRepeat = l2Status.m / l0Status.m_l0;
  int64_t nRepeat = l2Status.n / l0Status.n_l0;
  int64_t kBl1Db = (l2Status.m == l0Status.m_l0) ? l0Status.k_l0 : l2Status.k;
  NeitherFullLoadDb(l2Status, l0Status, l1Status, kBl1Db);
  NeitherFullLoadMN(l2Status, l0Status, l1Status, params);
  NeitherFullLoadK(l2Status, l0Status, l1Status, params);
  l1Status.load_size =
    ((l2Status.m == l1Status.m_al1 * l0Status.m_l0 && l1Status.kal1_16 == params.k) ? 1 : nRepeat) *
      l2Status.m +
      (l1Status.kbl1_16 == params.k ? 1 : mRepeat) * l2Status.n;
  res[IDX_THREE][IDX_ZERO] = l1Status.kal1_16;
  res[IDX_THREE][IDX_ONE] = l1Status.m_al1;
  res[IDX_THREE][IDX_TWO] = l1Status.db_al1;
  res[IDX_THREE][IDX_THREE] = l1Status.kbl1_16;
  res[IDX_THREE][IDX_FOUR] = l1Status.n_bl1;
  res[IDX_THREE][IDX_FIVE] = l1Status.db_bl1;
  res[IDX_THREE][IDX_SIX] = l1Status.load_size;
}

void GetL1Factors(const string &op_type, const BatchmatmulParas &params, const L2Status &l2Status,
                  const L0Status &l0Status, L1Status &l1Status)
{
  // get m_al1, n_bl1, kal1_16, kbl1_16 factors when L0, singlecore factor is know
  // get al1, bl1 double buffer factors

  int64_t mte1Loop = 50 / ((l0Status.n_l0 == 1 ? 1 : l0Status.k_l0) + (l0Status.k_l0 == 1 ? 1 : l0Status.m_l0));
  int64_t res[4][7] = {0};
  l1Status.all_times = l2Status.k / l0Status.k_l0;
  l1Status.max_m_al1 = (l2Status.m + l0Status.m_l0 - 1) / l0Status.m_l0;
  l1Status.max_n_bl1 = (l2Status.n + l0Status.n_l0 - 1) / l0Status.n_l0;
  l1Status.max_k_al1 =
    max(mte1Loop, ((MIN_MTE1_LOAD + l0Status.m_l0 - 1) / l0Status.m_l0 + l0Status.k_l0 - 1) / l0Status.k_l0);
  l1Status.max_k_bl1 =
    max(mte1Loop, ((MIN_MTE1_LOAD + l0Status.n_l0 - 1) / l0Status.n_l0 + l0Status.k_l0 - 1) / l0Status.k_l0);
  // both AL1 and Bl1 full load
  l1Status.SetStatus(l2Status.k, l2Status.k, l1Status.max_m_al1, l1Status.max_n_bl1, DB_OFF, DB_OFF);
  L1StatusBothFullLoad(l2Status, l0Status, l1Status, res);
  // only AL1 full load
  l1Status.SetStatus(l2Status.k, l0Status.k_l0, l1Status.max_m_al1, 1, DB_OFF, DB_OFF);
  L1StatusAl1FullLoad(l2Status, l0Status, l1Status, res);
  // only BL1 full load
  l1Status.SetStatus(l0Status.k_l0, l2Status.k, 1, l1Status.max_n_bl1, DB_OFF, DB_OFF);
  L1StatusBl1FullLoad(l2Status, l0Status, l1Status, res);
  // neither AL1 nor Bl1 full load
  l1Status.SetStatus(l0Status.k_l0, l0Status.k_l0, 1, 1, DB_ON, DB_ON);
  L1StatusNeitherFullLoad(l2Status, params, l0Status, l1Status, res);
  // choose the final factors
  int64_t *tmpFactors;
  tmpFactors = res[IDX_THREE];
  int64_t tmpLoadSize = tmpFactors[IDX_SIX];
  if (l1Status.al1_full_load &&
    (res[IDX_ONE][IDX_SIX] < tmpLoadSize ||
      (res[IDX_ONE][IDX_SIX] == tmpLoadSize &&
        res[IDX_ONE][IDX_ONE] + res[IDX_ONE][IDX_FOUR] > tmpFactors[IDX_ONE] + tmpFactors[IDX_FOUR]))) {
    tmpFactors = res[IDX_ONE];
    tmpLoadSize = tmpFactors[IDX_SIX];
  }
  if (l1Status.bl1_full_load &&
    (res[IDX_TWO][IDX_SIX] < tmpLoadSize ||
      (res[IDX_TWO][IDX_SIX] == tmpLoadSize &&
        res[IDX_TWO][IDX_ONE] + res[IDX_TWO][IDX_FOUR] > tmpFactors[IDX_ONE] + tmpFactors[IDX_FOUR]))) {
    tmpFactors = res[IDX_TWO];
    tmpLoadSize = tmpFactors[IDX_SIX];
  }
  if (l1Status.both_full_load &&
    (res[IDX_ZERO][IDX_SIX] < tmpLoadSize ||
      (res[IDX_ZERO][IDX_SIX] == tmpLoadSize &&
        res[IDX_ZERO][IDX_ONE] + res[IDX_ZERO][IDX_FOUR] > tmpFactors[IDX_ONE] + tmpFactors[IDX_FOUR]))) {
    tmpFactors = res[IDX_ZERO];
  }
  l1Status.SetStatus(tmpFactors[IDX_ZERO], tmpFactors[IDX_THREE], tmpFactors[IDX_ONE], tmpFactors[IDX_FOUR],
                     tmpFactors[IDX_TWO], tmpFactors[IDX_FIVE]);
  OP_LOGD(op_type.c_str(), "tiling kal1_16:%s, kbl1_16:%s, k_l0:%s", l1Status.kal1_16, l1Status.kbl1_16);
  OP_LOGD(op_type.c_str(), "tiling m_al1:%s, n_bl1:%s", l1Status.m_al1, l1Status.n_bl1);
  OP_LOGD(op_type.c_str(), "tiling db_al1:%s, db_bl1:%s", l1Status.db_al1, l1Status.db_bl1);
}

void GetUbFactors(const string &op_type, const L0Status &l0Status, UbStatus &ubStatus)
{
  ubStatus.n_cub = l0Status.n_l0;
  ubStatus.db_cub = l0Status.db_cub;
  OP_LOGD(op_type.c_str(), "tiling n_cub:%s, db_cub:%s", l0Status.n_l0, l0Status.db_cub);
}

void CheckSpecialTemplate(const string &op_type, const L2Status &l2Status, const L0Status &l0Status,
                          L1Status &l1Status, const UbStatus &ubStatus)
{
  if (l2Status.m / (l1Status.m_al1 * l0Status.m_l0) == 1 && l1Status.kal1_16 == l2Status.k) {
    l1Status.m_al1 = NONE;
    OP_LOGD(op_type.c_str(), "check special template, tiling al1 changed to full load");
  }
  if (l1Status.n_bl1 * l0Status.n_l0 == l2Status.n && l1Status.kbl1_16 == l2Status.k) {
    l1Status.n_bl1 = NONE;
    OP_LOGD(op_type.c_str(), "check special template, tiling bl1 changed to full load");
  }
}

void GenTiling(const string &op_type, const BatchmatmulParas &params, Tiling &tiling, string &tilingId)
{
  OP_LOGD(op_type.c_str(), "cache tiling input shape batch:%s", params.batch);
  OP_LOGD(op_type.c_str(), "cache tiling input shape m:%s", params.m);
  OP_LOGD(op_type.c_str(), "cache tiling input shape k:%s", params.k);
  OP_LOGD(op_type.c_str(), "cache tiling input shape n:%s", params.n);
  L2Status l2Status;
  L0Status l0Status;
  L1Status l1Status;
  UbStatus ubStatus;
  l0Status.SetInitLoadStatus();
  int64_t blockValue = GetBlockDim(op_type, params, l2Status, CORE_NUM);
  GetL0Factors(op_type, l2Status, blockValue, l0Status);
  GetL1Factors(op_type, params, l2Status, l0Status, l1Status);
  GetUbFactors(op_type, l0Status, ubStatus);
  CheckSpecialTemplate(op_type, l2Status, l0Status, l1Status, ubStatus);
  tiling.SetParams(l2Status, l0Status, l1Status, ubStatus);
  tiling.SetAttachFlag();
  tiling.GetTilingId();
  tilingId = tiling.tiling_id;
  OP_LOGD(op_type.c_str(), "get tiling id %s from cache tiling", tilingId);
}
} // namespace optiling