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
 * \file formula.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_FORMULA_TILING_H_
#define OPS_BUILT_IN_OP_TILING_FORMULA_TILING_H_

#include <algorithm>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits.h>
#include <map>
#include <math.h>
#include <numeric>
#include <ratio>
#include <unistd.h>
#include <vector>

namespace optiling {
static const int64_t NONE = -LLONG_MAX;
static const int64_t M2_SIZE = 16;
static const int64_t CORE_NUM = 32;
static const int64_t M2_MAX_SIZE = 1024;
static const int64_t L1_Size = (1024 * 1024);
static const int64_t L0a_Size = (64 * 1024);
static const int64_t L0b_Size = (64 * 1024);
static const int64_t L0c_Size = (256 * 1024);
static const int64_t Ub_Size = (256 * 1024);
static const int64_t L0c_NZsize = (L0c_Size / (4 * 2 * 16 * 16));
static const int64_t L0a_NZsize = (L0a_Size / (2 * 2 * 16 * 16));
static const int64_t L0b_NZsize = (L0b_Size / (2 * 2 * 16 * 16));
static const int64_t MMADCompute1us = 100;
static const int64_t SingleDimMin = 32;
const int64_t BLOCK_SIZE = 16;
const int64_t MIN_FRACTAL_SIZE = BLOCK_SIZE * BLOCK_SIZE;
const int64_t DB_ON = 2;
const int64_t DB_OFF = 1;
const int64_t IDX_ZERO = 0;
const int64_t IDX_ONE = 1;
const int64_t IDX_TWO = 2;
const int64_t IDX_THREE = 3;
const int64_t IDX_FOUR = 4;
const int64_t IDX_FIVE = 5;
const int64_t IDX_SIX = 6;
const int64_t IDX_SEVEN = 7;
const int64_t IDX_EIGHT = 8;
const int64_t ATTACH_FLAG_ZERO = 0;
const int64_t ATTACH_FLAG_ONE = 1;
const int64_t ATTACH_FLAG_TWO = 2;
const int64_t KBYTES = 1024;
const int64_t MAX_FACTOR = 128;
const int64_t FP16_BYTES = 2;
const int64_t MIN_MTE1_LOAD = 32;
const int64_t L0_PARAS_COMBO_LEN = 8;
const int64_t LOAD_SIZE_RANGE_LOW = 1000;
const int64_t LOAD_SIZE_RANGE_HIGH = 4000;
const int64_t LOAD_SIZE_DIFF_RANGE = 400;
const int64_t M_LOW_RANGE = 5;
const int64_t M_HIGH_RANGE = 6;
const double BLOCKING_PCT_GATE = 0.5;
const double LOAD_SIZE_GATE = 0.13;
const int64_t CORE_USE_LOW_RANGE = 5;
const int64_t CORE_USE_HIGH_RANGE = 9;

extern std::string BigAxis;
struct SingleCoreParas {
  int64_t batch{};
  int64_t m{};
  int64_t k{};
  int64_t n{};
  int64_t batch_dim{};
  int64_t m_dim{};
  int64_t n_dim{};
};

struct BlockDimCalculator{
  int64_t batch{};
  int64_t m{};
  int64_t k{};
  int64_t n{};
  int64_t kNum {};
  int64_t kBytes{};
  int64_t nDimFactor{};
  int64_t batchDimFactor{};
  int64_t mDimFactor{};
  int64_t minLoadSize{};
  int64_t coreUse{};
  int64_t finalValue{};
  bool finalBlockingFlag{};
};

struct L0Status {
  int64_t m_l0{};
  int64_t n_l0{};
  int64_t k_l0{};
  int64_t db_l0a{};
  int64_t db_l0b{};
  int64_t db_l0c{};
  int64_t db_cub{};
  int64_t finalM0{};
  int64_t finalK0{};
  int64_t finalN0{};
  int64_t finalLoadSize{};
  int64_t finalL0cUse{};
  int64_t finalMul{};
  int64_t finalMte1Loop{};
  int64_t maxMK{};
  int64_t maxNK{};
  int64_t maxMN{};
  int64_t maxAxisIdx{};
  int64_t maxAxisNum{};
  int64_t maxAxisPnt{};
  void SetInitLoadStatus()
  {
    finalM0 = 0;
    finalK0 = 0;
    finalN0 = 0;
    finalLoadSize = LLONG_MAX;
    finalL0cUse = 0;
    finalMul = 0;
    finalMte1Loop = LLONG_MAX;
  }
};

struct MKNParasCombo{
  int64_t parasCombo[9];
};

struct L1Status {
  int64_t kal1_16{};
  int64_t kbl1_16{};
  int64_t m_al1{};
  int64_t n_bl1{};
  int64_t db_al1{};
  int64_t db_bl1{};
  int64_t al1_size{};
  int64_t bl1_size{};
  int64_t al1_times{};
  int64_t bl1_times{};
  int64_t all_times{};
  int64_t load_size{};
  int64_t max_m_al1 = {};
  int64_t max_n_bl1 = {};
  int64_t max_k_al1 = {};
  int64_t max_k_bl1 = {};
  bool both_full_load{};
  bool al1_full_load{};
  bool bl1_full_load{};
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
  int64_t n_cub{};
  int64_t db_cub{};
};

class Tiling {
public:
  std::map<std::string, std::vector<int64_t>> mParam;
  std::map<std::string, std::map<std::string, int64_t>> mPingpongBuff;
  std::map<std::string, int64_t> pingpong;
  std::string tiling_id;
  int64_t n_cub{};
  int64_t db_cub{};
  int64_t m_l0{};
  int64_t k_l0{};
  int64_t n_l0{};
  int64_t batch_dim{};
  int64_t n_dim{};
  int64_t m_dim{};
  int64_t kal1_16{};
  int64_t kbl1_16{};
  int64_t m_al1{};
  int64_t n_bl1{};
  int64_t db_al1{};
  int64_t db_bl1{};
  int64_t k_aub{};
  int64_t m_aub{};
  int64_t db_aub{};
  int64_t k_org_dim{};
  int64_t db_l0c{};
  Tiling() = default;
  void SetDoubleBufferParams(bool minKl1CmpKl0, std::map<std::string, int64_t> dbFlag);
  void SetParams(const SingleCoreParas &singleCoreParas, const L0Status &l0Status, const L1Status &l1Status,
                 const UbStatus &ubStatus);
  void SetAttachFlag();
  void GetTilingId();
  ~Tiling() = default;;
};

int64_t GetFactors(int64_t *cnt, int64_t *factorList, const int64_t &num,
                   const int64_t &maxNum);
int64_t GetTwoFactors(int64_t *res, const int64_t &base, const int64_t &dim,
                      const int64_t &maxNum);
int64_t GetNearestFactor(const int64_t &base, int64_t &factor);
int64_t BL1FullLoadBlock(const SingleCoreParas &singleCoreParas, int64_t &n0, const int64_t &oriAMatrixSize,
                         int64_t &amatSize, int64_t &bmatSize, int64_t &loadSize, int64_t &tmpValue);
int64_t AL1FullLoadBlock(int64_t &m0, const int64_t &oriAMatrixSize, int64_t &tmpAmatSize, int64_t &tmpBmatSize,
                         int64_t &tmpLoadSize, int64_t &amatSize, int64_t &bmatSize, int64_t &loadSize,
                         int64_t &tmpValue, const SingleCoreParas &singleCoreParas);
int64_t NeitherFullLoadBlock(const int64_t n0s[][2], const int64_t m0s[][2], int64_t &tmpAmatSize, int64_t &tmpBmatSize,
                             int64_t &tmpLoadSize, const int64_t &j, const int64_t &mF, const int64_t &oriAMatrixSize,
                             const SingleCoreParas &singleCoreParas, int64_t &amatSize, int64_t &bmatSize,
                             int64_t &loadSize, int64_t &tmpValue);
int64_t GetBlockDim(SingleCoreParas &singleCoreParas, const int64_t &batch, const int64_t &m, const int64_t &k,
                    const int64_t &n, const int64_t &coreNum);
int64_t CheckUbDb(L0Status &l0Status);
int64_t GetLoadSize(const SingleCoreParas &singleCoreParas, const L0Status &l0Status);
MKNParasCombo GetParasCombo(const int64_t& index, const int64_t& blockValue);
int64_t GetFinalMkn(L0Status &l0Status, const SingleCoreParas &singleCoreParas);
int64_t GetL0StatusFromParasCombo(L0Status &l0Status, int64_t parasCombo[9]);
int64_t SetResFactors(int64_t *resFactors, const L0Status &l0Status);
int64_t GetL0FactorsCand(int64_t *resFactors, const SingleCoreParas &singleCoreParas, int64_t parasCombo[9],
                         L0Status &l0Status);
int64_t GetMKN(const SingleCoreParas &singleCoreParas, const int64_t &blockValue, L0Status &l0Status);
int64_t GetL1Size(const L1Status &l1Status, const L0Status &l0Status);
int64_t CheckSpecialTemplate(const SingleCoreParas &singleCoreParas, const L0Status &l0Status, L1Status &l1Status,
                             const UbStatus &ubStatus);
int64_t L1StatusBothFullLoad(const SingleCoreParas &singleCoreParas, const L0Status &l0Status, L1Status &l1Status,
                             int64_t res[][7]);
int64_t L1StatusAl1FullLoad(const SingleCoreParas &singleCoreParas, const L0Status &l0Status, L1Status &l1Status,
                            int64_t res[][7]);
int64_t L1StatusBl1FullLoad(const SingleCoreParas &singleCoreParas, const L0Status &l0Status, L1Status &l1Status,
                            int64_t res[][7]);
int64_t NeitherFullLoadDb(const SingleCoreParas &singleCoreParas, const L0Status &l0Status, L1Status &l1Status,
                          const int64_t &kbl1Db);
int64_t NeitherFullLoadMN(const SingleCoreParas &singleCoreParas, const L0Status &l0Status, L1Status &l1Status,
                          const int64_t &batch, const int64_t &m, const int64_t &k, const int64_t &n);
int64_t NeitherFullLoadK(const SingleCoreParas &singleCoreParas, const L0Status &l0Status, L1Status &l1Status,
                         const int64_t &batch, const int64_t &m, const int64_t &k, const int64_t &n);
int64_t L1StatusNeitherFullLoad(const SingleCoreParas &singleCoreParas, const L0Status &l0Status, L1Status &l1Status,
                                int64_t res[][7], const int64_t &batch, const int64_t &m, const int64_t &k,
                                const int64_t &n);
int64_t GetL1Factors(const int64_t &batch, const int64_t &m, const int64_t &k, const int64_t &n,
                     const SingleCoreParas &singleCoreParas, const L0Status &l0Status, L1Status &l1Status);
int64_t GetUbFactors(const L0Status &l0Status, UbStatus &ubStatus);
int64_t GenTiling(const int64_t &batch, const int64_t &m, const int64_t &k, const int64_t &n, Tiling &tiling,
                  std::string &tilingId);
}; // namespace optiling

#endif
