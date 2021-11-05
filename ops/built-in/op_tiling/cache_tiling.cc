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
 * \brief function of cache_tiling
 */

#include "cache_tiling.h"

namespace optiling {
void Tiling::SetDoubleBufferParams(bool minKl1CmpKl0, std::map<std::string, int64_t> dbFlag)
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

void Tiling::SetParams(const SingleCoreParas &singleCoreParas, const L0Status &l0Status, const L1Status &l1Status,
                       const UbStatus &ubStatus)
{
  batch_dim = singleCoreParas.batch_dim;
  n_dim = singleCoreParas.n_dim;
  m_dim = singleCoreParas.m_dim;
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
  k_org_dim = singleCoreParas.k * BLOCK_SIZE;
  db_l0c = l0Status.db_l0c;
  k_aub = ubStatus.k_aub;
  m_aub = ubStatus.m_aub;
  db_aub = ubStatus.db_aub;
  mParam.emplace("block_dim", std::move(std::vector<int64_t>({batch_dim, n_dim, m_dim, 1})));
  mParam.emplace("AL0_matrix", std::move(std::vector<int64_t>({m_l0, k_l0, BLOCK_SIZE, BLOCK_SIZE, 1, 1})));
  mParam.emplace("BL0_matrix", std::move(std::vector<int64_t>({k_l0, n_l0, BLOCK_SIZE, BLOCK_SIZE, 1, 1})));
  mParam.emplace("CL0_matrix", std::move(std::vector<int64_t>({n_l0, m_l0, BLOCK_SIZE, BLOCK_SIZE, 1, 1})));
  mParam.emplace("CUB_matrix", std::move(std::vector<int64_t>({n_cub, m_l0, BLOCK_SIZE, BLOCK_SIZE, 1, 1})));

  mParam["BUB_shape"] = {};
  mParam.emplace("AL1_shape", std::move(std::vector<int64_t>({kal1_16 * BLOCK_SIZE, m_al1, 1, 1})));
  mParam.emplace("BL1_shape", std::move(std::vector<int64_t>({kbl1_16 * BLOCK_SIZE, n_bl1, 1, 1})));
  mParam.emplace("AUB_shape", std::move(std::vector<int64_t>({k_aub * BLOCK_SIZE, m_aub, 1, 1})));

  mParam["n_bef_batch_flag"] = {0};
  mParam["n_bef_group_flag"] = {0};
  mParam["batch_bef_group_flag"] = {0};
  mParam["A_overhead_opt_flag"] = {0};
  mParam["B_overhead_opt_flag"] = {0};
  mParam["AUB_channel_wise_flag"] = {};
  mParam["BUB_channel_wise_flag"] = {};
  mParam["CUB_channel_wise_flag"] = {};

  if (m_al1 == NONE) {
    mParam["AL1_shape"] = std::vector<int64_t>({});
    db_al1 = 1;
  }
  if (n_bl1 == NONE) {
    mParam["BL1_shape"] = std::vector<int64_t>({});
    db_bl1 = 1;
  }
  std::map<std::string, int64_t> dbFlag = {
    {"db_aub", db_aub}, {"db_al1", db_al1}, {"db_bl1", db_bl1}, {"db_l0c", db_l0c}, {"db_cub", db_cub}};
  SetDoubleBufferParams(std::min(kal1_16, kbl1_16) == k_l0, dbFlag);
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
  std::vector<std::string> tilingKeywords =
    std::vector<std::string>({
                               "AL1_pbuffer", "BL1_pbuffer", "CL0_pbuffer", "abkl1_attach_flag", "al1_attach_flag",
                               "bl1_attach_flag", "min_kl1_cmp_kl0"
                             });
  for (const auto &tilingKeyword: tilingKeywords) {
    this->tiling_id += std::to_string(this->mPingpongBuff["manual_pingpong_buffer"][tilingKeyword]);
  }
}

int64_t GetFactors(int64_t *cnt, int64_t *factorList, const int64_t &num, const int64_t &maxNum)
{
  // get all factors of num which smaller or equal to maxNum
  for (int64_t i = 1; i < maxNum + 1; i++) {
    if (num % i == 0) {
      factorList[(*cnt)++] = i;
    }
  }
  return 0;
}

int64_t GetTwoFactors(int64_t *res, const int64_t &base, const int64_t &dim, const int64_t &maxNum = 32)
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
  return 0;
}

int64_t GetNearestFactor(const int64_t &base, int64_t &factor)
{
  while (factor > 0 && base % factor != 0) {
    factor--;
  }
  return 0;
}

int64_t BL1FullLoadBlock(const SingleCoreParas &singleCoreParas, int64_t &n0, const int64_t &oriAMatrixSize,
                         int64_t &amatSize, int64_t &bmatSize, int64_t &loadSize, int64_t &tmpValue)
{
  if (n0 >= 1) {
    while (singleCoreParas.n % n0 != 0) {
      n0--;
    }
    amatSize = oriAMatrixSize * (singleCoreParas.n / n0);
    bmatSize = singleCoreParas.n;
    loadSize = amatSize + bmatSize;
    tmpValue = n0;
  }
  return 0;
}

int64_t AL1FullLoadBlock(int64_t &m0, const int64_t &oriAMatrixSize, int64_t &tmpAmatSize, int64_t &tmpBmatSize,
                         int64_t &tmpLoadSize, int64_t &amatSize, int64_t &bmatSize, int64_t &loadSize,
                         int64_t &tmpValue, const SingleCoreParas &singleCoreParas)
{
  if (m0 >= 1) {
    while (singleCoreParas.m % m0 != 0) {
      m0--;
    }
    tmpAmatSize = oriAMatrixSize;
    tmpBmatSize = singleCoreParas.n * (oriAMatrixSize / m0);
    tmpLoadSize = tmpAmatSize + tmpBmatSize;
    if (tmpLoadSize < loadSize) {
      amatSize = tmpAmatSize;
      bmatSize = tmpBmatSize;
      loadSize = tmpLoadSize;
      tmpValue = 0;
    }
  }
  return 0;
}

int64_t NeitherFullLoadBlock(const int64_t n0s[][2], const int64_t m0s[][2], int64_t &tmpAmatSize, int64_t &tmpBmatSize,
                             int64_t &tmpLoadSize, const int64_t &j, const int64_t &mF, const int64_t &oriAMatrixSize,
                             const SingleCoreParas &singleCoreParas, int64_t &amatSize, int64_t &bmatSize,
                             int64_t &loadSize, int64_t &tmpValue)
{
  for (auto const &n0: n0s[j]) {
    for (auto const &m0: m0s[mF]) {
      if (m0 <= 0 || n0 <= 0) {
        continue;
      }
      if (m0 * n0 * KBYTES <= L0c_Size) {
        tmpAmatSize = oriAMatrixSize * (singleCoreParas.n / n0);
        tmpBmatSize = singleCoreParas.n * (oriAMatrixSize / m0);
        tmpLoadSize = tmpAmatSize + tmpBmatSize;
        if (tmpLoadSize < loadSize) {
          amatSize = tmpAmatSize;
          bmatSize = tmpBmatSize;
          loadSize = tmpLoadSize;
          tmpValue = 0;
        }
      }
    }
  }
  return 0;
}

int64_t GetBlockDimHelper(SingleCoreParas &singleCoreParas, BlockDimCalculator &blockDimCalculator,
                          const int64_t *batchDimArray, const int64_t *mDimArray, const int64_t *nDimArray,
                          const int64_t &iIdx, const int64_t &jIdx, const int64_t &mDimCnt, const int64_t &nDimCnt,
                          const int64_t m0s[][2], const int64_t n0s[][2])
{
  int64_t bFactor = batchDimArray[iIdx];
  int64_t nFactor = nDimArray[jIdx];
  for (int64_t mIdx = 0; mIdx < mDimCnt; mIdx++) {
    int64_t mFactor = mDimArray[mIdx];
    if (bFactor * nFactor * mFactor > CORE_NUM) {
      break;
    }
    singleCoreParas.batch = blockDimCalculator.batch / bFactor;
    singleCoreParas.m = blockDimCalculator.m / mFactor;
    singleCoreParas.n = blockDimCalculator.n / nFactor;
    int64_t oriAMatrixSize = singleCoreParas.batch * singleCoreParas.m;
    int64_t oriBMatrixSize = singleCoreParas.n;
    int64_t amatSize = oriAMatrixSize;
    int64_t bmatSize = oriBMatrixSize;
    int64_t totalLoadSize = amatSize + bmatSize;
    int64_t tmpValue = 0;
    if (totalLoadSize * blockDimCalculator.kBytes > L1_Size) {
      totalLoadSize = LONG_LONG_MAX;
      int64_t tmpAmatSize, tmpBmatSize, tmpLoadSize;
      // BL1 k full load
      int64_t n0 = std::min(
        std::min((L1_Size / FP16_BYTES - MIN_FRACTAL_SIZE) / blockDimCalculator.kNum, singleCoreParas.n), MAX_FACTOR);
      BL1FullLoadBlock(singleCoreParas, n0, oriAMatrixSize, amatSize, bmatSize, totalLoadSize, tmpValue);
      // AL1 full load
      int64_t m0 = std::min(std::min((L1_Size / FP16_BYTES - MIN_FRACTAL_SIZE) /
                                       (MIN_FRACTAL_SIZE * blockDimCalculator.k * oriAMatrixSize),
                                     oriAMatrixSize),
                            MAX_FACTOR);
      AL1FullLoadBlock(m0, oriAMatrixSize, tmpAmatSize, tmpBmatSize, tmpLoadSize, amatSize, bmatSize, totalLoadSize,
                       tmpValue, singleCoreParas);
      // neither full load max_m max_n
      // closest m and n
      NeitherFullLoadBlock(n0s, m0s, tmpAmatSize, tmpBmatSize, tmpLoadSize, nFactor, mFactor, oriAMatrixSize,
                           singleCoreParas,
                           amatSize, bmatSize, totalLoadSize, tmpValue);
    }
    int64_t loadSizeKb = totalLoadSize * blockDimCalculator.kBytes / KBYTES;
    int64_t minLoadSizeKb = blockDimCalculator.minLoadSize * blockDimCalculator.kBytes / KBYTES;
    double tmpBlockingPct;
    if (nFactor > mFactor) {
      tmpBlockingPct = double(amatSize) / totalLoadSize;
    } else if (nFactor < mFactor) {
      tmpBlockingPct = double(bmatSize) / totalLoadSize;
    } else {
      tmpBlockingPct = double(std::max(amatSize, bmatSize)) / totalLoadSize;
    }
    bool tmpBlockingFlag = (loadSizeKb < LOAD_SIZE_RANGE_LOW && std::max(nFactor, mFactor) > M_LOW_RANGE);

    // updateSolution: bool whether update to a new block factor solution
    // use more coreNums or use same core num but has smaller loadsize
    // or same core num same loadsize but has bigger batch_dim * n_dim
    // when loadsize in a predetermined range, do not block factor solution
    // these predetermined range parameters is only suitable for cloud 60 platform

    auto updateSolution = (totalLoadSize < blockDimCalculator.minLoadSize) ||
      ((totalLoadSize == blockDimCalculator.minLoadSize) &&
        ((blockDimCalculator.nDimFactor * blockDimCalculator.batchDimFactor < bFactor * nFactor) ||
          (blockDimCalculator.nDimFactor * blockDimCalculator.batchDimFactor == bFactor * nFactor &&
            blockDimCalculator.batchDimFactor < bFactor))) ||
      (blockDimCalculator.finalBlockingFlag && (loadSizeKb - minLoadSizeKb) < LOAD_SIZE_DIFF_RANGE &&
        std::max(nFactor, mFactor) < std::max(blockDimCalculator.nDimFactor, blockDimCalculator.mDimFactor));
    auto noUpdateSolution =
      (((loadSizeKb >= LOAD_SIZE_RANGE_LOW && loadSizeKb < LOAD_SIZE_RANGE_HIGH
        && std::max(nFactor, mFactor) > M_HIGH_RANGE
        && tmpBlockingPct > BLOCKING_PCT_GATE) &&
        std::max(nFactor, mFactor) > std::max(blockDimCalculator.nDimFactor, blockDimCalculator.mDimFactor) &&
        double(blockDimCalculator.minLoadSize - totalLoadSize) / blockDimCalculator.minLoadSize < LOAD_SIZE_GATE &&
        blockDimCalculator.coreUse >= CORE_USE_HIGH_RANGE) ||
        ((loadSizeKb < LOAD_SIZE_RANGE_LOW && std::max(nFactor, mFactor) > M_LOW_RANGE) &&
          (std::max(nFactor, mFactor) > std::max(blockDimCalculator.nDimFactor, blockDimCalculator.mDimFactor)) &&
          ((minLoadSizeKb - loadSizeKb) < LOAD_SIZE_DIFF_RANGE && blockDimCalculator.coreUse > CORE_USE_LOW_RANGE)));
    auto updateCondition = updateSolution && !noUpdateSolution;
    if (updateCondition) {
      blockDimCalculator.minLoadSize = totalLoadSize;
      blockDimCalculator.nDimFactor = nFactor;
      blockDimCalculator.batchDimFactor = bFactor;
      blockDimCalculator.mDimFactor = mFactor;
      blockDimCalculator.finalBlockingFlag = tmpBlockingFlag;
      blockDimCalculator.coreUse =
        blockDimCalculator.nDimFactor * blockDimCalculator.batchDimFactor * blockDimCalculator.mDimFactor;
      blockDimCalculator.finalValue = tmpValue;
    }
  }
  return 0;
}

int64_t GetBlockDim(SingleCoreParas &singleCoreParas, const int64_t &batch, const int64_t &m, const int64_t &k,
                    const int64_t &n, const int64_t &coreNum = CORE_NUM)
{
  if (batch * m * n < coreNum) {
    singleCoreParas.batch_dim = batch;
    singleCoreParas.n_dim = n;
    singleCoreParas.m_dim = m;
    singleCoreParas.batch = 1;
    singleCoreParas.m = 1;
    singleCoreParas.k = k;
    singleCoreParas.n = 1;
    return 0;
  }
  BlockDimCalculator blockDimCalculator;
  blockDimCalculator.batch = batch;
  blockDimCalculator.m = m;
  blockDimCalculator.k = k;
  blockDimCalculator.n = n;
  int64_t batchDimArray[coreNum] = {0};
  int64_t nDimArray[coreNum] = {0};
  int64_t mDimArray[coreNum] = {0};
  int64_t batchDimCnt = 0;
  int64_t nDimCnt = 0;
  int64_t mDimCnt = 0;
  GetFactors(&batchDimCnt, batchDimArray, batch, coreNum);
  GetFactors(&nDimCnt, nDimArray, n, coreNum);
  GetFactors(&mDimCnt, mDimArray, m, coreNum);
  int64_t mnMaxPnt = 16;
  int64_t m0s[coreNum + 1][2] = {0};
  int64_t n0s[coreNum + 1][2] = {0};
  for (int64_t idx = 0; idx < nDimCnt; idx++) {
    int64_t tmpNDim = nDimArray[idx];
    int64_t tmpNSingleCore = n / tmpNDim;
    GetTwoFactors(n0s[tmpNDim], mnMaxPnt, tmpNSingleCore, MAX_FACTOR);
  }
  for (int64_t idx = 0; idx < mDimCnt; idx++) {
    int64_t tmpMDim = mDimArray[idx];
    int64_t tmpMSingleCore = m / tmpMDim;
    GetTwoFactors(m0s[tmpMDim], mnMaxPnt, tmpMSingleCore, MAX_FACTOR);
  }
  blockDimCalculator.kNum = k * BLOCK_SIZE * BLOCK_SIZE;
  blockDimCalculator.kBytes = blockDimCalculator.kNum * FP16_BYTES;
  blockDimCalculator.nDimFactor = 1;
  blockDimCalculator.batchDimFactor = 1;
  blockDimCalculator.mDimFactor = 1;
  blockDimCalculator.minLoadSize = L1_Size / FP16_BYTES;
  blockDimCalculator.finalBlockingFlag = false;
  blockDimCalculator.coreUse = 1;
  blockDimCalculator.finalValue = 0;
  for (int64_t iIdx = 0; iIdx < batchDimCnt; iIdx++) {
    for (int64_t jIdx = 0; jIdx < nDimCnt; jIdx++) {
      GetBlockDimHelper(singleCoreParas, blockDimCalculator, batchDimArray, mDimArray, nDimArray, iIdx, jIdx, mDimCnt,
                        nDimCnt, m0s, n0s);
    }
  }
  singleCoreParas.batch_dim = blockDimCalculator.batchDimFactor;
  singleCoreParas.n_dim = blockDimCalculator.nDimFactor;
  singleCoreParas.m_dim = blockDimCalculator.mDimFactor;
  singleCoreParas.m = m / blockDimCalculator.mDimFactor;
  singleCoreParas.n = n / blockDimCalculator.nDimFactor;
  singleCoreParas.k = k;
  singleCoreParas.batch = batch / blockDimCalculator.batchDimFactor;
  return blockDimCalculator.finalValue;
}

int64_t CheckUbDb(L0Status &l0Status)
{
  int64_t nCub = 1;
  int64_t c0 = 16;
  int64_t ubFp16Size = Ub_Size / FP16_BYTES;
  int64_t copyoutSize = nCub * l0Status.m_l0 * c0;
  if (copyoutSize * l0Status.db_cub > ubFp16Size) {
    l0Status.db_cub = 1;
  }
  return 0;
}

int64_t GetLoadSize(const SingleCoreParas &singleCoreParas, const L0Status &l0Status)
{
  bool al1FullLoad =
    ((singleCoreParas.m * singleCoreParas.k + l0Status.n_l0 * l0Status.k_l0) * BLOCK_SIZE * BLOCK_SIZE * FP16_BYTES <=
      L1_Size);
  bool bl1FullLoad =
    ((l0Status.m_l0 * l0Status.k_l0 + l0Status.n_l0 * singleCoreParas.k) * BLOCK_SIZE * BLOCK_SIZE * FP16_BYTES <=
      L1_Size);
  bool bothFullLoad = ((singleCoreParas.m * singleCoreParas.k + l0Status.n_l0 * singleCoreParas.k) * BLOCK_SIZE *
    BLOCK_SIZE * FP16_BYTES <= L1_Size);
  int64_t num0a =
    bl1FullLoad ? singleCoreParas.n : ((singleCoreParas.m + l0Status.m_l0 - 1) / l0Status.m_l0) * singleCoreParas.n;
  int64_t num0b =
    al1FullLoad ? singleCoreParas.m : ((singleCoreParas.n + l0Status.n_l0 - 1) / l0Status.n_l0) * singleCoreParas.m;
  if ((al1FullLoad && bl1FullLoad) && !bothFullLoad) {
    return std::min(singleCoreParas.n + ((singleCoreParas.n + l0Status.n_l0 - 1) / l0Status.n_l0) * singleCoreParas.m,
                    singleCoreParas.m + ((singleCoreParas.m + l0Status.m_l0 - 1) / l0Status.m_l0) * singleCoreParas.n);
  }
  return num0a + num0b;
}

int64_t GetFinalMkn(L0Status &l0Status, const SingleCoreParas &singleCoreParas)
{
  int64_t tmpL0cUse = l0Status.m_l0 * l0Status.n_l0 * l0Status.db_l0c * BLOCK_SIZE * BLOCK_SIZE * 4 * 100 / L0c_Size;
  int64_t tmpMte1Loop = ((l0Status.n_l0 != 1) ? l0Status.k_l0 : 1) + ((l0Status.k_l0 != 1) ? l0Status.m_l0 : 1);
  int64_t tmpMul = l0Status.m_l0 * l0Status.k_l0 * l0Status.n_l0;
  int64_t tmpLoadSize = GetLoadSize(singleCoreParas, l0Status);
  auto condition1 = l0Status.finalM0 == 0;
  auto condition2 = tmpLoadSize < l0Status.finalLoadSize;
  auto condition3 = tmpLoadSize == l0Status.finalLoadSize && tmpMul > l0Status.finalMul;
  auto condition4 =
    tmpMul == l0Status.finalMul && tmpLoadSize == l0Status.finalLoadSize && tmpMte1Loop < l0Status.finalMte1Loop;
  if (condition1 || condition2 || condition3 || condition4) {
    l0Status.finalM0 = l0Status.m_l0;
    l0Status.finalK0 = l0Status.k_l0;
    l0Status.finalN0 = l0Status.n_l0;
    l0Status.finalLoadSize = tmpLoadSize;
    l0Status.finalL0cUse = tmpL0cUse;
    l0Status.finalMul = tmpMul;
    l0Status.finalMte1Loop = tmpMte1Loop;
  }
  return 0;
}

MKNParasCombo GetParasCombo(const int64_t &index, const int64_t &blockValue)
{
  std::map<int64_t, MKNParasCombo> parasComboMap;
  if (blockValue == 0) {
    MKNParasCombo comboZero = {2, 2, 2, 64, 64, 128, 0, 64, 11};
    MKNParasCombo comboOne = {2, 2, 1, 64, 64, 256, 0, 64, 16};
    MKNParasCombo comboTwo = {2, 1, 2, 64, 128, 128, 1, 64, 16};
    MKNParasCombo comboThree = {1, 2, 2, 128, 64, 128, 0, 128, 16};
    MKNParasCombo comboFour = {1, 1, 2, 128, 128, 128, 0, 128, 11};
    MKNParasCombo comboFive = {1, 2, 1, 128, 64, 256, 0, 128, 22};
    MKNParasCombo comboSix = {2, 1, 1, 64, 128, 256, 1, 128, 22};
    MKNParasCombo comboSeven = {1, 1, 1, 128, 128, 256, 0, 128, 16};
    parasComboMap = {{0, comboZero}, {1, comboOne}, {2, comboTwo}, {3, comboThree}, {4, comboFour}, {5, comboFive},
                     {6, comboSix}, {7, comboSeven}};
  } else {
    MKNParasCombo comboZero = {2, 2, 2, 64, 64, 128, 1, 64, blockValue};
    MKNParasCombo comboOne = {2, 2, 1, 64, 64, 256, 1, 64, blockValue};
    MKNParasCombo comboTwo = {2, 1, 2, 64, 128, 128, 1, 128, blockValue};
    MKNParasCombo comboThree = {1, 2, 2, 128, 64, 128, 1, 64, blockValue};
    MKNParasCombo comboFour = {1, 1, 2, 128, 128, 128, 1, 128, blockValue};
    MKNParasCombo comboFive = {1, 2, 1, 128, 64, 256, 1, 128, blockValue};
    MKNParasCombo comboSix = {2, 1, 1, 64, 128, 256, 1, 128, blockValue};
    MKNParasCombo comboSeven = {1, 1, 1, 128, 128, 256, 1, 128, blockValue};
    parasComboMap = {{0, comboZero}, {1, comboOne}, {2, comboTwo}, {3, comboThree}, {4, comboFour}, {5, comboFive},
                     {6, comboSix}, {7, comboSeven}};
  }
  return parasComboMap[index];
}

int64_t GetL0StatusFromParasCombo(L0Status &l0Status, int64_t parasCombo[9])
{
  l0Status.SetInitLoadStatus();
  l0Status.db_l0a = parasCombo[IDX_ZERO];
  l0Status.db_l0b = parasCombo[IDX_ONE];
  l0Status.db_l0c = parasCombo[IDX_TWO];
  l0Status.maxMK = parasCombo[IDX_THREE];
  l0Status.maxNK = parasCombo[IDX_FOUR];
  l0Status.maxMN = parasCombo[IDX_FIVE];
  l0Status.maxAxisIdx = parasCombo[IDX_SIX];
  l0Status.maxAxisNum = parasCombo[IDX_SEVEN];
  l0Status.maxAxisPnt = parasCombo[IDX_EIGHT];
  l0Status.maxAxisPnt = std::min(l0Status.maxAxisPnt, l0Status.maxAxisNum);
  return 0;
}

int64_t SetResFactors(int64_t *resFactors, const L0Status &l0Status)
{
  resFactors[IDX_ZERO] = l0Status.finalM0;
  resFactors[IDX_ONE] = l0Status.finalK0;
  resFactors[IDX_TWO] = l0Status.finalN0;
  resFactors[IDX_THREE] = l0Status.finalLoadSize;
  resFactors[IDX_FOUR] = l0Status.finalL0cUse;
  resFactors[IDX_FIVE] = l0Status.finalMte1Loop;
  resFactors[IDX_SIX] = l0Status.finalMul;
  return 0;
}

int64_t GetL0FactorsCand(int64_t *resFactors, const SingleCoreParas &singleCoreParas, int64_t parasCombo[9],
                         L0Status &l0Status)
{
  GetL0StatusFromParasCombo(l0Status, parasCombo);
  int64_t majorDim = singleCoreParas.m;
  int64_t minorDim = singleCoreParas.n;
  int64_t majorDimK = l0Status.maxMK;
  int64_t minorDimK = l0Status.maxNK;
  if (l0Status.maxAxisIdx != 0) {
    majorDim = singleCoreParas.n;
    minorDim = singleCoreParas.m;
    majorDimK = l0Status.maxNK;
    minorDimK = l0Status.maxMK;
  }
  int64_t majorDimFactors[2] = {0};
  GetTwoFactors(majorDimFactors, l0Status.maxAxisPnt, majorDim, l0Status.maxAxisNum);
  for (auto &majorDimFactor: majorDimFactors) {
    if (majorDimFactor == 0) {
      continue;
    }
    int64_t minorFactorMax = std::min(l0Status.maxMN / majorDimFactor, minorDimK);
    int64_t minorDimFactors[2] = {0};
    GetTwoFactors(minorDimFactors, minorFactorMax, minorDim, minorFactorMax);
    for (auto &minorDimFactor: minorDimFactors) {
      if (minorDimFactor == 0) {
        continue;
      }
      int64_t k0Max = std::min(majorDimK / majorDimFactor, minorDimK / minorDimFactor);
      int64_t k0Factors[2] = {0};
      GetTwoFactors(k0Factors, k0Max, singleCoreParas.k, k0Max);
      for (auto &k0: k0Factors) {
        if (k0 == 0) {
          continue;
        }
        if (l0Status.maxAxisIdx == 0) {
          l0Status.m_l0 = majorDimFactor;
          l0Status.n_l0 = minorDimFactor;
        } else {
          l0Status.m_l0 = minorDimFactor;
          l0Status.n_l0 = majorDimFactor;
        }
        l0Status.k_l0 = k0;
        GetFinalMkn(l0Status, singleCoreParas);
      }
    }
  }
  SetResFactors(resFactors, l0Status);
  return 0;
}

int64_t GetMKN(const SingleCoreParas &singleCoreParas, const int64_t &blockValue, L0Status &l0Status)
{
  int64_t dbAOnBOnCOnIdx = 0;
  int64_t dbAOnBOnCOffIdx = 1;
  int64_t resFactors[8][7] = {0};
  int64_t parasCombo[9];
  for (int i = 0; i < L0_PARAS_COMBO_LEN; ++i) {
    MKNParasCombo mknParasCombo = GetParasCombo(i, blockValue);
    GetL0FactorsCand(resFactors[i], singleCoreParas, mknParasCombo.parasCombo, l0Status);
  }

  // check both L0C utilization and loadsize to control LOC LOA LOB DB
  int64_t dbL0aL0cDbOn = 2;
  int64_t dbL0bL0cDbOn = 2;
  int64_t m0L0cDbOn = resFactors[dbAOnBOnCOnIdx][IDX_ZERO];
  int64_t k0L0cDbOn = resFactors[dbAOnBOnCOnIdx][IDX_ONE];
  int64_t n0L0cDbOn = resFactors[dbAOnBOnCOnIdx][IDX_TWO];
  int64_t loadSizeL0cDbOn = resFactors[dbAOnBOnCOnIdx][IDX_THREE];
  int64_t l0cUseL0cDbOn = resFactors[dbAOnBOnCOnIdx][IDX_FOUR];
  int64_t mte1LoopL0cDbOn = resFactors[dbAOnBOnCOnIdx][IDX_FIVE];
  int64_t mulL0cDbOn = resFactors[dbAOnBOnCOnIdx][IDX_SIX];

  int64_t dbL0aL0cDbOff = 2;
  int64_t dbL0bL0cDbOff = 2;
  int64_t m0L0cDbOff = resFactors[dbAOnBOnCOffIdx][IDX_ZERO];
  int64_t k0L0cDbOff = resFactors[dbAOnBOnCOffIdx][IDX_ONE];
  int64_t n0L0cDbOff = resFactors[dbAOnBOnCOffIdx][IDX_TWO];
  int64_t loadSizeL0cDbOff = resFactors[dbAOnBOnCOffIdx][IDX_THREE];
  int64_t l0cUseL0cDbOff = resFactors[dbAOnBOnCOffIdx][IDX_FOUR];
  int64_t mte1LoopL0cDbOff = resFactors[dbAOnBOnCOffIdx][IDX_FIVE];
  int64_t mulL0cDbOff = resFactors[dbAOnBOnCOffIdx][IDX_SIX];

  if (l0cUseL0cDbOff > l0cUseL0cDbOn or loadSizeL0cDbOff < loadSizeL0cDbOn) {
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
  return 0;
}

int64_t GetL1Size(const L1Status &l1Status, const L0Status &l0Status)
{
  int64_t curL1Size;
  curL1Size =
    l1Status.m_al1 * l0Status.m_l0 * BLOCK_SIZE * l1Status.kal1_16 * BLOCK_SIZE * l1Status.db_al1 * FP16_BYTES +
      l1Status.n_bl1 * l0Status.n_l0 * BLOCK_SIZE * l1Status.kbl1_16 * BLOCK_SIZE * l1Status.db_bl1 * FP16_BYTES;
  return curL1Size;
}

int64_t L1StatusBothFullLoad(const SingleCoreParas &singleCoreParas, const L0Status &l0Status, L1Status &l1Status,
                             int64_t res[][7])
{
  int64_t curL1Size;
  curL1Size = GetL1Size(l1Status, l0Status);
  if (curL1Size <= L1_Size) {
    l1Status.both_full_load = true;
    l1Status.load_size = singleCoreParas.m + singleCoreParas.n;
    res[IDX_ZERO][IDX_ZERO] = l1Status.kal1_16;
    res[IDX_ZERO][IDX_ONE] = l1Status.m_al1;
    res[IDX_ZERO][IDX_TWO] = l1Status.db_al1;
    res[IDX_ZERO][IDX_THREE] = l1Status.kbl1_16;
    res[IDX_ZERO][IDX_FOUR] = l1Status.n_bl1;
    res[IDX_ZERO][IDX_FIVE] = l1Status.db_bl1;
    res[IDX_ZERO][IDX_SIX] = l1Status.load_size;
  }
  return 0;
}

int64_t L1StatusAl1FullLoad(const SingleCoreParas &singleCoreParas, const L0Status &l0Status, L1Status &l1Status,
                            int64_t res[][7])
{
  int64_t curL1Size;
  int64_t mRepeat = singleCoreParas.m / l0Status.m_l0;
  int64_t nRepeat = singleCoreParas.n / l0Status.n_l0;
  curL1Size = GetL1Size(l1Status, l0Status);
  if (curL1Size <= L1_Size) {
    l1Status.al1_full_load = true;
    l1Status.al1_size = singleCoreParas.k * singleCoreParas.m * BLOCK_SIZE * BLOCK_SIZE * FP16_BYTES;
    l1Status.bl1_size = L1_Size - l1Status.al1_size;
    l1Status.db_bl1 = DB_ON;
    if (GetL1Size(l1Status, l0Status) > L1_Size) {
      l1Status.db_bl1 = DB_OFF;
    }
    l1Status.kbl1_16 = std::min(
      l1Status.bl1_size / (l1Status.n_bl1 * l0Status.n_l0 * BLOCK_SIZE * l1Status.db_bl1 * FP16_BYTES * BLOCK_SIZE),
      singleCoreParas.k);
    l1Status.bl1_times = std::min(l1Status.kbl1_16 / l0Status.k_l0, l1Status.max_k_bl1);
    GetNearestFactor(l1Status.all_times, l1Status.bl1_times);
    l1Status.kbl1_16 = l1Status.bl1_times * l0Status.k_l0;
    if (l1Status.kbl1_16 == singleCoreParas.k) {
      l1Status.n_bl1 = std::min(l1Status.bl1_size / (l1Status.kbl1_16 * l0Status.n_l0 * BLOCK_SIZE * l1Status.db_bl1 *
                                  FP16_BYTES * BLOCK_SIZE),
                                l1Status.max_n_bl1);
      GetNearestFactor(nRepeat, l1Status.n_bl1);
    }
    l1Status.load_size = singleCoreParas.m + (l1Status.kbl1_16 == singleCoreParas.k ? 1 : mRepeat) * singleCoreParas.n;
    res[IDX_ONE][IDX_ZERO] = l1Status.kal1_16;
    res[IDX_ONE][IDX_ONE] = l1Status.m_al1;
    res[IDX_ONE][IDX_TWO] = l1Status.db_al1;
    res[IDX_ONE][IDX_THREE] = l1Status.kbl1_16;
    res[IDX_ONE][IDX_FOUR] = l1Status.n_bl1;
    res[IDX_ONE][IDX_FIVE] = l1Status.db_bl1;
    res[IDX_ONE][IDX_SIX] = l1Status.load_size;
  }
  return 0;
}

int64_t L1StatusBl1FullLoad(const SingleCoreParas &singleCoreParas, const L0Status &l0Status, L1Status &l1Status,
                            int64_t res[][7])
{
  int64_t curL1Size;
  int64_t mRepeat = singleCoreParas.m / l0Status.m_l0;
  int64_t nRepeat = singleCoreParas.n / l0Status.n_l0;
  curL1Size = GetL1Size(l1Status, l0Status);
  if (curL1Size <= L1_Size) {
    l1Status.bl1_full_load = true;
    l1Status.bl1_size = singleCoreParas.k * singleCoreParas.n * BLOCK_SIZE * BLOCK_SIZE * FP16_BYTES;
    l1Status.al1_size = L1_Size - l1Status.bl1_size;
    l1Status.db_al1 = DB_ON;
    if (GetL1Size(l1Status, l0Status) > L1_Size) {
      l1Status.db_al1 = DB_OFF;
    }
    l1Status.kal1_16 = std::min(
      l1Status.al1_size / (l1Status.m_al1 * l0Status.m_l0 * BLOCK_SIZE * l1Status.db_al1 * FP16_BYTES * BLOCK_SIZE),
      singleCoreParas.k);
    l1Status.al1_times = std::min(l1Status.kal1_16 / l0Status.k_l0, l1Status.max_k_al1);
    GetNearestFactor(l1Status.all_times, l1Status.al1_times);
    l1Status.kal1_16 = l1Status.al1_times * l0Status.k_l0;
    if (l1Status.kal1_16 == singleCoreParas.k) {
      l1Status.m_al1 = std::min(l1Status.al1_size / (l1Status.kal1_16 * l0Status.m_l0 * BLOCK_SIZE * l1Status.db_al1 *
                                  FP16_BYTES * BLOCK_SIZE),
                                l1Status.max_m_al1);
      GetNearestFactor(mRepeat, l1Status.m_al1);
    }
    l1Status.load_size =
      singleCoreParas.n +
        ((singleCoreParas.m == l1Status.m_al1 * l0Status.m_l0 && l1Status.kal1_16 == singleCoreParas.k) ? 1 : nRepeat) *
          singleCoreParas.m;
    res[IDX_TWO][IDX_ZERO] = l1Status.kal1_16;
    res[IDX_TWO][IDX_ONE] = l1Status.m_al1;
    res[IDX_TWO][IDX_TWO] = l1Status.db_al1;
    res[IDX_TWO][IDX_THREE] = l1Status.kbl1_16;
    res[IDX_TWO][IDX_FOUR] = l1Status.n_bl1;
    res[IDX_TWO][IDX_FIVE] = l1Status.db_bl1;
    res[IDX_TWO][IDX_SIX] = l1Status.load_size;
  }
  return 0;
}

int64_t NeitherFullLoadDb(const SingleCoreParas &singleCoreParas, const L0Status &l0Status, L1Status &l1Status,
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
  l1Status.kbl1_16 = singleCoreParas.k;
  bool bothDoubleBuffer = singleCoreParas.m != l0Status.m_l0 && singleCoreParas.k > l0Status.k_l0 &&
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
  return 0;
}

int64_t NeitherFullLoadMN(const SingleCoreParas &singleCoreParas, const L0Status &l0Status, L1Status &l1Status,
                          const int64_t &batch, const int64_t &m, const int64_t &k, const int64_t &n)
{
  int64_t mRepeat = singleCoreParas.m / l0Status.m_l0;
  int64_t nRepeat = singleCoreParas.n / l0Status.n_l0;
  if (l0Status.k_l0 == k) {
    if (m > n) {
      l1Status.bl1_size = k * l0Status.n_l0 * BLOCK_SIZE * BLOCK_SIZE * l1Status.db_bl1 * FP16_BYTES;
      l1Status.al1_size = L1_Size - l1Status.bl1_size;
      l1Status.m_al1 = std::min(l1Status.al1_size / (l1Status.kal1_16 * l0Status.m_l0 * BLOCK_SIZE * l1Status.db_al1 *
        FP16_BYTES * BLOCK_SIZE), l1Status.max_m_al1);
      GetNearestFactor(mRepeat, l1Status.m_al1);
      l1Status.al1_size =
        l1Status.kal1_16 * l1Status.m_al1 * l0Status.m_l0 * BLOCK_SIZE * BLOCK_SIZE * l1Status.db_al1 * FP16_BYTES;
      l1Status.bl1_size = L1_Size - l1Status.al1_size;
      l1Status.n_bl1 = std::min(l1Status.bl1_size / (l1Status.kbl1_16 * l0Status.n_l0 * BLOCK_SIZE * l1Status.db_bl1 *
                                  FP16_BYTES * BLOCK_SIZE),
                                l1Status.max_n_bl1);
      GetNearestFactor(nRepeat, l1Status.n_bl1);
    } else {
      l1Status.al1_size = k * l0Status.m_l0 * BLOCK_SIZE * BLOCK_SIZE * l1Status.db_al1 * FP16_BYTES;
      l1Status.bl1_size = L1_Size - l1Status.al1_size;
      l1Status.n_bl1 = std::min(l1Status.bl1_size / (l1Status.kbl1_16 * l0Status.n_l0 * BLOCK_SIZE * l1Status.db_bl1 *
                                  FP16_BYTES * BLOCK_SIZE),
                                l1Status.max_n_bl1);
      GetNearestFactor(nRepeat, l1Status.n_bl1);
      l1Status.bl1_size = k * l0Status.n_l0 * BLOCK_SIZE * BLOCK_SIZE * l1Status.db_bl1 * FP16_BYTES;
      l1Status.al1_size = L1_Size - l1Status.bl1_size;
      l1Status.m_al1 = std::min(l1Status.al1_size / (l1Status.kal1_16 * l0Status.m_l0 * BLOCK_SIZE * l1Status.db_al1 *
                                  FP16_BYTES * BLOCK_SIZE),
                                l1Status.max_m_al1);
      GetNearestFactor(mRepeat, l1Status.m_al1);
    }
  }
  return 0;
}

int64_t NeitherFullLoadK(const SingleCoreParas &singleCoreParas, const L0Status &l0Status, L1Status &l1Status,
                         const int64_t &batch, const int64_t &m, const int64_t &k, const int64_t &n)
{
  l1Status.kbl1_16 = k;
  if (GetL1Size(l1Status, l0Status) <= L1_Size) {
    l1Status.bl1_size = k * l0Status.n_l0 * BLOCK_SIZE * BLOCK_SIZE * l1Status.db_bl1 * FP16_BYTES;
    l1Status.al1_size = L1_Size - l1Status.bl1_size;
    l1Status.kal1_16 = std::min(
      l1Status.al1_size / (l1Status.m_al1 * l0Status.m_l0 * BLOCK_SIZE * l1Status.db_al1 * FP16_BYTES * BLOCK_SIZE),
      singleCoreParas.k);
    l1Status.al1_times = std::min(l1Status.kal1_16 / l0Status.k_l0, l1Status.max_k_al1);
    GetNearestFactor(l1Status.all_times, l1Status.al1_times);
    l1Status.kal1_16 = l1Status.al1_times * l0Status.k_l0;
  } else {
    int64_t perK = std::min(L1_Size /
                              (l0Status.m_l0 * BLOCK_SIZE * BLOCK_SIZE * l1Status.db_al1 * FP16_BYTES +
                                BLOCK_SIZE * l0Status.n_l0 * BLOCK_SIZE * l1Status.db_bl1 * FP16_BYTES) /
                              l0Status.k_l0 * l0Status.k_l0,
                            singleCoreParas.k);
    int64_t perTimes = std::min(perK / l0Status.k_l0, std::max(l1Status.max_k_al1, l1Status.max_k_bl1));
    GetNearestFactor(l1Status.all_times, perTimes);
    perK = perTimes * l0Status.k_l0;
    l1Status.kal1_16 = perK;
    l1Status.kbl1_16 = perK;
  }
  return 0;
}

int64_t L1StatusNeitherFullLoad(const SingleCoreParas &singleCoreParas, const L0Status &l0Status, L1Status &l1Status,
                                int64_t res[][7], const int64_t &batch, const int64_t &m, const int64_t &k,
                                const int64_t &n)
{
  int64_t mRepeat = singleCoreParas.m / l0Status.m_l0;
  int64_t nRepeat = singleCoreParas.n / l0Status.n_l0;
  int64_t kBl1Db = (singleCoreParas.m == l0Status.m_l0) ? l0Status.k_l0 : singleCoreParas.k;
  NeitherFullLoadDb(singleCoreParas, l0Status, l1Status, kBl1Db);
  NeitherFullLoadMN(singleCoreParas, l0Status, l1Status, batch, m, k, n);
  NeitherFullLoadK(singleCoreParas, l0Status, l1Status, batch, m, k, n);
  l1Status.load_size = ((singleCoreParas.m == l1Status.m_al1 * l0Status.m_l0 && l1Status.kal1_16 == k) ? 1 : nRepeat) *
    singleCoreParas.m +
    (l1Status.kbl1_16 == k ? 1 : mRepeat) * singleCoreParas.n;
  res[IDX_THREE][IDX_ZERO] = l1Status.kal1_16;
  res[IDX_THREE][IDX_ONE] = l1Status.m_al1;
  res[IDX_THREE][IDX_TWO] = l1Status.db_al1;
  res[IDX_THREE][IDX_THREE] = l1Status.kbl1_16;
  res[IDX_THREE][IDX_FOUR] = l1Status.n_bl1;
  res[IDX_THREE][IDX_FIVE] = l1Status.db_bl1;
  res[IDX_THREE][IDX_SIX] = l1Status.load_size;
  return 0;
}

int64_t GetL1Factors(const int64_t &batch, const int64_t &m, const int64_t &k, const int64_t &n,
                     const SingleCoreParas &singleCoreParas, const L0Status &l0Status, L1Status &l1Status)
{
  int64_t mte1Loop = 50 / ((l0Status.n_l0 == 1 ? 1 : l0Status.k_l0) + (l0Status.k_l0 == 1 ? 1 : l0Status.m_l0));
  int64_t res[4][7] = {0};
  l1Status.all_times = singleCoreParas.k / l0Status.k_l0;
  l1Status.max_m_al1 = (singleCoreParas.m + l0Status.m_l0 - 1) / l0Status.m_l0;
  l1Status.max_n_bl1 = (singleCoreParas.n + l0Status.n_l0 - 1) / l0Status.n_l0;
  l1Status.max_k_al1 =
    std::max(mte1Loop, ((MIN_MTE1_LOAD + l0Status.m_l0 - 1) / l0Status.m_l0 + l0Status.k_l0 - 1) / l0Status.k_l0);
  l1Status.max_k_bl1 =
    std::max(mte1Loop, ((MIN_MTE1_LOAD + l0Status.n_l0 - 1) / l0Status.n_l0 + l0Status.k_l0 - 1) / l0Status.k_l0);
  // both AL1 and Bl1 full load
  l1Status.SetStatus(singleCoreParas.k, singleCoreParas.k, l1Status.max_m_al1, l1Status.max_n_bl1, DB_OFF, DB_OFF);
  L1StatusBothFullLoad(singleCoreParas, l0Status, l1Status, res);
  // only AL1 full load
  l1Status.SetStatus(singleCoreParas.k, l0Status.k_l0, l1Status.max_m_al1, 1, DB_OFF, DB_OFF);
  L1StatusAl1FullLoad(singleCoreParas, l0Status, l1Status, res);
  // only BL1 full load
  l1Status.SetStatus(l0Status.k_l0, singleCoreParas.k, 1, l1Status.max_n_bl1, DB_OFF, DB_OFF);
  L1StatusBl1FullLoad(singleCoreParas, l0Status, l1Status, res);
  // neither AL1 nor Bl1 full load
  l1Status.SetStatus(l0Status.k_l0, l0Status.k_l0, 1, 1, DB_ON, DB_ON);
  L1StatusNeitherFullLoad(singleCoreParas, l0Status, l1Status, res, batch, m, k, n);
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
  return 0;
}

int64_t GetUbFactors(const L0Status &l0Status, UbStatus &ubStatus)
{
  ubStatus.n_cub = l0Status.n_l0;
  ubStatus.db_cub = l0Status.db_cub;
  return 0;
}

int64_t CheckSpecialTemplate(const SingleCoreParas &singleCoreParas, const L0Status &l0Status, L1Status &l1Status,
                             const UbStatus &ubStatus)
{
  if (singleCoreParas.m / (l1Status.m_al1 * l0Status.m_l0) == 1 && l1Status.kal1_16 == singleCoreParas.k) {
    l1Status.m_al1 = NONE;
  }
  if (l1Status.n_bl1 * l0Status.n_l0 == singleCoreParas.n && l1Status.kbl1_16 == singleCoreParas.k) {
    l1Status.n_bl1 = NONE;
  }
  return 0;
}

int64_t GenTiling(const int64_t &batch, const int64_t &m, const int64_t &k, const int64_t &n, Tiling &tiling,
                  std::string &tilingId)
{
  SingleCoreParas singleCoreParas;
  L0Status l0Status;
  L1Status l1Status;
  UbStatus ubStatus;
  l0Status.SetInitLoadStatus();
  int64_t blockValue = GetBlockDim(singleCoreParas, batch, m, k, n, CORE_NUM);
  GetMKN(singleCoreParas, blockValue, l0Status);
  GetL1Factors(batch, m, k, n, singleCoreParas, l0Status, l1Status);
  GetUbFactors(l0Status, ubStatus);
  CheckSpecialTemplate(singleCoreParas, l0Status, l1Status, ubStatus);
  tiling.SetParams(singleCoreParas, l0Status, l1Status, ubStatus);
  tiling.SetAttachFlag();
  tiling.GetTilingId();
  tilingId = tiling.tiling_id;
  return 0;
}

} // namespace optiling
