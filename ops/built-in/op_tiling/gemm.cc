/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file gemm.cc
 * \brief tiling function of gemm
 */
#include <algorithm>
#include <climits>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "../op_proto/util/axis_util.h"
#include "../op_proto/util/error_util.h"
#include "external/graph/operator.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/types.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "cache_tiling.h"
#include "map"
#include "op_log.h"
#include "op_tiling.h"
#include "op_tiling_util.h"
using namespace std;
using json = nlohmann::json;
using utils = ge::TypeUtils;

namespace {
const int64_t kIdxMLow = 0;
const int64_t kIdxMHigh = 1;
const int64_t kIdxKLow = 2;
const int64_t kIdxKHigh = 3;
const int64_t kIdxNLow = 4;
const int64_t kIdxNHigh = 5;
const int64_t kIdxBLow = 6;
const int64_t kIdxBHigh = 7;
const int64_t kIdxM = 0;
const int64_t kIdxK = 1;
const int64_t kIdxN = 2;
const int64_t kIdxBatch = 3;
const int64_t kBlockIn = 16;
const int64_t kBlockReduce = 16;
const int64_t kBlockReduceS8 = 32;
const int64_t kBlockOut = 16;
const int64_t kMinBatchDimNum = 3;
const int64_t kNumTwo = 2;
const int32_t BLOCK_SIZE = 16;
const int64_t DIM_NUM = 3;
const int32_t CACHE_TILING_ID_LEN_NZ = 7;
const int32_t CACHE_TILING_ID_LEN_NZ_SPLIT_K = 8;
const int32_t CACHE_TILING_ID_LEN_ND = 9;
const int32_t CACHE_TILING_ID_LEN_ND_SPLIT_K = 10;
const int64_t KBLOCK_SIZE = 16;
const int64_t KMULTI = 4;
/*
TilingID from 20000 to 30000 means this tilingkey is used for aligned shape only. The range
is ensured by gemm_tilingcase.py.
The general tilingkeys' pattern is '1xxxx' and the aligned tilingkeys' pattern is '2xxxx'.
The first char of the general tilingkey is modified to '2' so as to use aligned pattern.
*/
const char ALIGNED_FLAG = '2';
}
namespace optiling {
struct OpRunInfoParas {
  BatchmatmulParas params;
  int32_t batch_single_core = 1;
  int32_t m_single_core = 1;
  int32_t n_single_core = 1;
  int32_t batch_dim = 1;
  int32_t n_dim = 1;
  int32_t m_dim = 1;
  int32_t k_dim = 1;
  int32_t m_al1 = 1;
  int32_t n_bl1 = 1;
  int32_t cub_n1 = 1;
  int32_t m_l0 = 1;
  int32_t n_l0 = 1;
  int32_t k_l0 = 1;
  int32_t n_ub_l0_time = 1;
  int32_t kal0_factor = 1;
  int32_t kbl0_factor = 1;
  int32_t kal1_factor = 1;
  int32_t kbl1_factor = 1;
  int32_t kal1_16 = 1;
  int32_t kbl1_16 = 1;
  int32_t k_al1 = kal1_16 * BLOCK_SIZE;
  int32_t k_bl1 = kbl1_16 * BLOCK_SIZE;
  int32_t kl1_times = 1;
  int32_t m_aub = 1;
  int32_t n_bub = 1;
  int32_t k_aub = BLOCK_SIZE;
  int32_t k_bub = BLOCK_SIZE;
  int32_t multi_n_ub_l1 = 1;
  int32_t multi_m_ub_l1 = 1;
  int32_t multi_k_aub_l1 = 1;
  int32_t multi_k_bub_l1 = 1;
  int32_t a_align_value = 1;
  int32_t b_align_value = 1;
  int32_t aub_align_bound = 0;
  int32_t bub_align_bound = 0;
};

// parse function
struct GemmCompileInfo {
  uint32_t workspace_num = 0;
  uint32_t ub_size = 0;
  BatchmatmulParas params;
  string dynamic_mode;
  bool trans_a;
  bool trans_b;
  bool repo_seed_flag;
  map<string, vector<int64_t>> repo_seeds;
  map<string, vector<int64_t>> repo_range;
  bool repo_costmodel_flag;
  map<string, vector<int64_t>> cost_range;
  map<string, int64_t> block_dim;
};

bool GetGEMMBatch(const string &op_type, const ge::GeShape &shape_a, const ge::GeShape &shape_b,
                  BatchmatmulParas &params) {
  int32_t num_dima = shape_a.GetDimNum();
  int32_t num_dimb = shape_b.GetDimNum();
  if (num_dima < DIM_NUM && num_dimb < DIM_NUM) {
    params.batch = 1;
    params.batch_32 = 1;
    return true;
  }

  params.b_have_batch = num_dimb > kNumTwo;

  const ge::GeShape& shape_short = num_dima < num_dimb ? shape_a : shape_b;
  const ge::GeShape& shape_long = num_dima < num_dimb ? shape_b : shape_a;
  ge::GeShape shape_broadcast(shape_long);
  int32_t num_dim = shape_long.GetDimNum();
  int32_t offset = num_dim - shape_short.GetDimNum();

  for (int32_t i = num_dim - kMinBatchDimNum; i >= offset; --i) {
    int64_t short_value = shape_short.GetDim(i - offset);
    int64_t long_value = shape_long.GetDim(i);
    OP_TILING_CHECK((short_value != long_value && short_value != 1 && long_value != 1),
                    CUBE_INNER_ERR_REPORT(op_type.c_str(), "Tensor a and b do not meet the broadcast rule"),
                    return false);

    shape_broadcast.SetDim(i, max(short_value, long_value));
  }

  int64_t batch_value = 1;
  for (int32_t i = 0; i < num_dim - kNumTwo; ++i) {
    batch_value *= shape_broadcast.GetDim(i);
  }
  params.batch = batch_value;
  OP_TILING_CHECK(
      (batch_value > INT_MAX),
      CUBE_INNER_ERR_REPORT(op_type.c_str(), "The batch of a and b tensors' shape must not larger than INT_MAX"),
      return false);
  params.batch_32 = batch_value;
  return true;
}

bool CalcGEMMMknb(const string &op_type, ge::DataType dtype, const ge::GeShape &ori_shape_a,
                  const ge::GeShape &ori_shape_b, shared_ptr<GemmCompileInfo> &compile_value) {
  int32_t block_reduce = kBlockReduce, block_in = kBlockIn, block_out = kBlockOut;
  bool is_int8_type = dtype == ge::DT_INT8 || dtype == ge::DT_UINT8;
  if (is_int8_type) {
    block_reduce = kBlockReduceS8;
  }

  int32_t num_dima = ori_shape_a.GetDimNum();
  int32_t num_dimb = ori_shape_b.GetDimNum();
  int32_t idx_m_of_a = num_dima - kNumTwo;
  int32_t idx_k_of_a = num_dima - 1;
  int32_t idx_k_of_b = num_dimb - kNumTwo;
  int32_t idx_n_of_b = num_dimb - 1;
  if (compile_value->trans_a) {
    idx_m_of_a = num_dima - 1;
    idx_k_of_a = num_dima - kNumTwo;
  }
  if (compile_value->trans_b) {
    idx_k_of_b = num_dimb - 1;
    idx_n_of_b = num_dimb - kNumTwo;
  }

  OP_TILING_CHECK((ori_shape_a.GetDim(idx_k_of_a) != ori_shape_b.GetDim(idx_k_of_b)),
                  CUBE_INNER_ERR_REPORT(op_type.c_str(), "The k-axis of a and b tensors must be the same"),
                  return false);

  compile_value->params.m = ceil(static_cast<double>(ori_shape_a.GetDim(idx_m_of_a)) / block_in);
  compile_value->params.k = ceil(static_cast<double>(ori_shape_a.GetDim(idx_k_of_a)) / block_reduce);
  compile_value->params.n = ceil(static_cast<double>(ori_shape_b.GetDim(idx_n_of_b)) / block_out);
  bool unvalid_dim =
      compile_value->params.m > INT_MAX || compile_value->params.k > INT_MAX || compile_value->params.n > INT_MAX;
  OP_TILING_CHECK(
      unvalid_dim,
      CUBE_INNER_ERR_REPORT(op_type.c_str(), "The m,k,n of a and b tensors' shape must not larger than INT_MAX"),
      return false);
  compile_value->params.m_32 = static_cast<int32_t>(compile_value->params.m);
  compile_value->params.k_32 = static_cast<int32_t>(compile_value->params.k);
  compile_value->params.n_32 = static_cast<int32_t>(compile_value->params.n);

  if (compile_value->params.format_a == "ND") {
    compile_value->params.ori_shape_M = ori_shape_a.GetDim(idx_m_of_a);
    compile_value->params.ori_shape_K = ori_shape_a.GetDim(idx_k_of_a);
  }
  if (compile_value->params.format_b == "ND") {
    compile_value->params.ori_shape_N = ori_shape_b.GetDim(idx_n_of_b);
  }
  bool unvalid_ori_shape = compile_value->params.ori_shape_M > INT_MAX || compile_value->params.ori_shape_K > INT_MAX ||
                           compile_value->params.ori_shape_N > INT_MAX;
  OP_TILING_CHECK(
      unvalid_ori_shape,
      CUBE_INNER_ERR_REPORT(op_type.c_str(), "The m,k,n of a and b tensors' ori_shape must not larger than INT_MAX"),
      return false);
  bool is_nd_input = !compile_value->params.binary_mode_flag && compile_value->params.format_a == "ND" &&
                     compile_value->params.format_b == "ND";
  if (is_nd_input) {
    // Aligned schedule pattern selection is only enabled in ND input format
    bool aligned_m = compile_value->params.ori_shape_M % block_in == 0;
    bool aligned_k = compile_value->params.ori_shape_K % block_reduce == 0;
    bool aligned_n = compile_value->params.ori_shape_N % block_out == 0;
    bool aligned_mkn = aligned_m && aligned_k && aligned_n;
    if (aligned_mkn) {
      compile_value->params.used_aligned_pattern = true;
    }
  }
  return GetGEMMBatch(op_type, ori_shape_a, ori_shape_b, compile_value->params);
}

string StringTeOperator(const ge::TensorDesc &tensor) {
  ostringstream oss;
  oss << "\t\tori_shape: (";
  for (auto dim: tensor.GetOriginShape().GetDims()) {
    oss << dim << ", ";
  }
  oss << ")" << endl;

  oss << "\t\tshape: (";
  for (auto dim: tensor.GetShape().GetDims()) {
    oss << dim << ", ";
  }
  oss << ")" << endl;
  oss << "\t\tdtype: " << ge::TypeUtils::DataTypeToSerialString(tensor.GetDataType()) << endl;
  oss << "\t\tformat: " << ge::TypeUtils::FormatToSerialString(tensor.GetFormat()) << endl;
  oss << "\t\tori_format: " << ge::TypeUtils::FormatToSerialString(tensor.GetOriginFormat()) << endl;

  return oss.str();
}

string DebugOpParasGEMM(const ge::Operator &op_paras) {
  ostringstream oss;
  oss << "inputs:" << endl;
  for (size_t i = 0; i < op_paras.GetInputsSize(); ++i) {
    oss << "\tinput " << i << endl;
    auto tensor = op_paras.GetInputDesc(i);
    oss << StringTeOperator(tensor);
  }

  oss << "outputs:" << endl;
  for (size_t i = 0; i < op_paras.GetOutputsSize(); ++i) {
    oss << "\toutput " << i << endl;
    auto tensor = op_paras.GetOutputDesc(i);
    oss << StringTeOperator(tensor);
  }
  return oss.str();
}

string DebugCompileInfoGEMM(const json &compile_info) {
  ostringstream oss;
  oss << "compile_info:" << endl;
  oss << compile_info.dump() << endl;
  return oss.str();
}

string CheckTilingInRepo(const string &op_type, const map<string, vector<int64_t>> &repo_seeds,
                         const map<string, vector<int64_t>> &repo_range, const BatchmatmulParas &params,
                         bool isBatchMatmulMode) {
  string tiling_id("-1");
  int64_t min_distance = LLONG_MAX;
  auto element_seed = repo_seeds.begin();
  auto element_range = repo_range.begin();
  while (element_seed != repo_seeds.end() && element_range != repo_range.end()) {
    OP_TILING_CHECK((element_seed->first != element_range->first),
          CUBE_INNER_ERR_REPORT(op_type.c_str(), "repo_seeds is inconsistent with repo_range"),
    return tiling_id);

    auto& seed = element_seed->second;
    auto& range = element_range->second;

    auto in_range = range[kIdxMLow] <= params.m && params.m <= range[kIdxMHigh] && range[kIdxKLow] <= params.k &&
      params.k <= range[kIdxKHigh] && range[kIdxNLow] <= params.n && params.n <= range[kIdxNHigh];
    if (isBatchMatmulMode) {
      in_range = in_range && range[kIdxBLow] <= params.batch && params.batch <= range[kIdxBHigh];
    }
    BatchmatmulParas seed_params;
    seed_params.m = seed[kIdxM];
    seed_params.k = seed[kIdxK];
    seed_params.n = seed[kIdxN];
    if (in_range) {
      int64_t dist = (params.m - seed_params.m) * (params.m - seed_params.m) +
        (params.k - seed_params.k) * (params.k - seed_params.k) +
        (params.n - seed_params.n) * (params.n - seed_params.n);
      if (isBatchMatmulMode) {
        seed_params.batch = seed[kIdxBatch];
        dist += (params.batch - seed_params.batch) * (params.batch - seed_params.batch);
      }
      if (dist < min_distance) {
        min_distance = dist;
        tiling_id = element_seed->first;
      }
    }
    ++element_seed;
    ++element_range;
  }
  return tiling_id;
}

string CheckTilingInCostModel(const string &op_type, const map<string, vector<int64_t>> &cost_range,
                              const BatchmatmulParas &params, bool isBatchMatmulMode) {
  string tiling_id("-1");
  auto element = cost_range.begin();
  while (element != cost_range.end()) {
    const auto& range = element->second;
    auto in_range = range[kIdxMLow] <= params.m && params.m <= range[kIdxMHigh] && range[kIdxKLow] <= params.k &&
      params.k <= range[kIdxKHigh] && range[kIdxNLow] <= params.n && params.n <= range[kIdxNHigh];
    if (isBatchMatmulMode) {
      in_range = in_range && range[kIdxBLow] <= params.batch && params.batch <= range[kIdxBHigh];
    }
    if (in_range) {
      tiling_id = element->first;
      OP_LOGD(op_type.c_str(), "match tiling_id(%s) in costmodel", tiling_id.c_str());
      break;
    }
    ++element;
  }
  return tiling_id;
}

void FillRunInfoParas(const Tiling &tiling, OpRunInfoParas &runinfoparas) {
  runinfoparas.batch_dim = tiling.batch_dim;
  runinfoparas.n_dim = tiling.n_dim;
  runinfoparas.m_dim = tiling.m_dim;
  runinfoparas.k_dim = tiling.k_dim;
  runinfoparas.m_l0 = tiling.m_l0;
  runinfoparas.k_l0 = tiling.k_l0;
  runinfoparas.n_l0 = tiling.n_l0;
  if (!tiling.al1_full_load) {
    runinfoparas.k_al1 = tiling.kal1_16 * ::BLOCK_SIZE;
    runinfoparas.m_al1 = tiling.m_al1;
  } else {
    runinfoparas.k_al1 = runinfoparas.params.k_32 / runinfoparas.k_dim * ::BLOCK_SIZE;
    runinfoparas.m_al1 = ceil(static_cast<double>(runinfoparas.params.m_32 / runinfoparas.m_dim) / runinfoparas.m_l0);
  }
  if (!tiling.bl1_full_load) {
    runinfoparas.k_bl1 = tiling.kbl1_16 * ::BLOCK_SIZE;
    runinfoparas.n_bl1 = tiling.n_bl1;
  } else {
    runinfoparas.k_bl1 = runinfoparas.params.k_32 / runinfoparas.k_dim * ::BLOCK_SIZE;
    runinfoparas.n_bl1 = ceil(static_cast<double>(runinfoparas.params.n_32 / runinfoparas.n_dim) / runinfoparas.n_l0);
  }
  runinfoparas.cub_n1 = tiling.n_cub;
  if (runinfoparas.params.nd_flag) {
    runinfoparas.k_aub = tiling.k_aub;
    runinfoparas.m_aub = tiling.m_aub;
    runinfoparas.multi_m_ub_l1 = runinfoparas.m_al1 * runinfoparas.m_l0 / tiling.m_aub;
    runinfoparas.multi_k_aub_l1 = runinfoparas.k_al1 / (tiling.k_aub * ::BLOCK_SIZE);
    runinfoparas.a_align_value = tiling.a_align_value;
    runinfoparas.aub_align_bound = tiling.aub_align_bound;
    runinfoparas.k_bub = tiling.k_bub;
    runinfoparas.n_bub = tiling.n_bub;
    runinfoparas.multi_n_ub_l1 = runinfoparas.n_bl1 * runinfoparas.n_l0 / tiling.n_bub;
    runinfoparas.multi_k_bub_l1 = runinfoparas.k_bl1 / (tiling.k_bub * ::BLOCK_SIZE);
    runinfoparas.b_align_value = tiling.b_align_value;
    runinfoparas.bub_align_bound = tiling.bub_align_bound;
  }
  runinfoparas.kal1_16 = runinfoparas.k_al1 / ::BLOCK_SIZE;
  runinfoparas.kbl1_16 = runinfoparas.k_bl1 / ::BLOCK_SIZE;
  runinfoparas.kal0_factor = runinfoparas.kal1_16 / runinfoparas.k_l0;
  runinfoparas.kbl0_factor = runinfoparas.kbl1_16 / runinfoparas.k_l0;
  runinfoparas.kal1_factor = runinfoparas.params.k_32 / runinfoparas.k_dim / runinfoparas.kal1_16;
  runinfoparas.kbl1_factor = runinfoparas.params.k_32 / runinfoparas.k_dim / runinfoparas.kbl1_16;
  runinfoparas.kl1_times = (runinfoparas.kal1_16 > runinfoparas.kbl1_16)
                           ? (runinfoparas.kal1_16 / runinfoparas.kbl1_16)
                           : (runinfoparas.kbl1_16 / runinfoparas.kal1_16);
  runinfoparas.n_ub_l0_time = runinfoparas.n_l0 / runinfoparas.cub_n1;
  runinfoparas.batch_single_core = runinfoparas.params.batch_32 / runinfoparas.batch_dim;
  runinfoparas.m_single_core =
    max(runinfoparas.params.m_32 / (runinfoparas.m_dim * runinfoparas.m_al1 * runinfoparas.m_l0), int32_t(1));
  runinfoparas.n_single_core = max(runinfoparas.params.n_32 / (runinfoparas.n_dim * runinfoparas.n_bl1 *
                                   runinfoparas.n_ub_l0_time * runinfoparas.cub_n1),
                                   int32_t(1));
}

void SetRunInfoForCacheTiling(const OpRunInfoParas &runinfoparas, utils::OpRunInfo &run_info) {
  if (runinfoparas.params.format_a == "ND") {
    run_info.AddTilingData(runinfoparas.params.m_32);
    run_info.AddTilingData(runinfoparas.params.k_32);
  }
  if (runinfoparas.params.format_b == "ND") {
    run_info.AddTilingData(runinfoparas.params.n_32);
  }
  run_info.AddTilingData(runinfoparas.batch_single_core);
  run_info.AddTilingData(runinfoparas.m_single_core);
  run_info.AddTilingData(runinfoparas.n_single_core);
  run_info.AddTilingData(runinfoparas.batch_dim);
  run_info.AddTilingData(runinfoparas.n_dim);
  run_info.AddTilingData(runinfoparas.m_dim);
  run_info.AddTilingData(runinfoparas.k_dim);
  run_info.AddTilingData(runinfoparas.m_al1);
  run_info.AddTilingData(runinfoparas.n_bl1);
  run_info.AddTilingData(runinfoparas.cub_n1);
  run_info.AddTilingData(runinfoparas.m_l0);
  run_info.AddTilingData(runinfoparas.k_l0);
  run_info.AddTilingData(runinfoparas.n_ub_l0_time);
  run_info.AddTilingData(runinfoparas.kal0_factor);
  run_info.AddTilingData(runinfoparas.kbl0_factor);
  run_info.AddTilingData(runinfoparas.kal1_factor);
  run_info.AddTilingData(runinfoparas.kbl1_factor);
  run_info.AddTilingData(runinfoparas.kal1_16);
  run_info.AddTilingData(runinfoparas.kbl1_16);
  run_info.AddTilingData(runinfoparas.kl1_times);
  if (runinfoparas.params.nd_flag) {
    run_info.AddTilingData(runinfoparas.m_aub);
    run_info.AddTilingData(runinfoparas.n_bub);
    run_info.AddTilingData(runinfoparas.k_aub);
    run_info.AddTilingData(runinfoparas.k_bub);
    run_info.AddTilingData(runinfoparas.multi_n_ub_l1);
    run_info.AddTilingData(runinfoparas.multi_m_ub_l1);
    run_info.AddTilingData(runinfoparas.multi_k_aub_l1);
    run_info.AddTilingData(runinfoparas.multi_k_bub_l1);
    run_info.AddTilingData(runinfoparas.a_align_value);
    run_info.AddTilingData(runinfoparas.b_align_value);
    run_info.AddTilingData(runinfoparas.aub_align_bound);
    run_info.AddTilingData(runinfoparas.bub_align_bound);
  }
}

void SetRunInfo(const bool &isBatchMatmulMode, string &tiling_id, const map<string, int64_t> &block_dim_info,
                const OpRunInfoParas &runinfoparas, utils::OpRunInfo &run_info) {
  bool is_cache_tiling = tiling_id.length() == CACHE_TILING_ID_LEN_NZ ||
      tiling_id.length() == CACHE_TILING_ID_LEN_ND || tiling_id.length() == CACHE_TILING_ID_LEN_NZ_SPLIT_K ||
      tiling_id.length() == CACHE_TILING_ID_LEN_ND_SPLIT_K;
  if (is_cache_tiling) {
    int32_t block_dim = runinfoparas.batch_dim * runinfoparas.n_dim * runinfoparas.m_dim * runinfoparas.k_dim;
    run_info.SetBlockDim(static_cast<uint32_t>(block_dim));
  } else {
    auto block_dim_value = block_dim_info.find(tiling_id);
    if (block_dim_value != block_dim_info.end()) {
      run_info.SetBlockDim(static_cast<uint32_t>(block_dim_value->second));
    }
  }
  // Used Aligned Pattern if the input shape is aligned. Only enabled in ND input format.
  if (runinfoparas.params.used_aligned_pattern && !is_cache_tiling) {
    tiling_id[0] = ALIGNED_FLAG;
  }
  run_info.SetTilingKey(stoi(tiling_id));
  if (runinfoparas.params.format_a == "ND") {
    run_info.AddTilingData(static_cast<int32_t>(runinfoparas.params.ori_shape_M));
    run_info.AddTilingData(static_cast<int32_t>(runinfoparas.params.ori_shape_K));
  } else {
    run_info.AddTilingData(runinfoparas.params.m_32);
    run_info.AddTilingData(runinfoparas.params.k_32);
  }
  if (runinfoparas.params.format_b == "ND") {
    run_info.AddTilingData(static_cast<int32_t>(runinfoparas.params.ori_shape_N));
  } else {
    run_info.AddTilingData(runinfoparas.params.n_32);
  }
  if (isBatchMatmulMode) {
    run_info.AddTilingData(runinfoparas.params.batch_32);
  }
  if (is_cache_tiling) {
    SetRunInfoForCacheTiling(runinfoparas, run_info);
  }
}

bool GetGemmCompileValue(const json &compile_info, shared_ptr<GemmCompileInfo> &compile_value) {
  compile_value->dynamic_mode = compile_info["dynamic_mode"];
  compile_value->params.format_a = compile_info["format_a"];
  compile_value->params.format_b = compile_info["format_b"];
  const auto& repo_attr = compile_info["attrs"];
  compile_value->trans_a = repo_attr["transpose_a"];
  compile_value->trans_b = repo_attr["transpose_b"];
  compile_value->block_dim = compile_info["block_dim"].get<map<string, int64_t>>();

  if (compile_info.contains("binary_mode_flag")) {
    compile_value->params.binary_mode_flag = compile_info["binary_mode_flag"];
    compile_value->params.core_num = compile_value->block_dim["CORE_NUM"];
    auto& binary_attrs = compile_info["binary_attrs"];
    compile_value->params.bias_flag = binary_attrs["bias_flag"];
    compile_value->params.nd_flag = binary_attrs["nd_flag"];
    compile_value->params.split_k_flag = binary_attrs["split_k_flag"];
    compile_value->params.l2_size = binary_attrs["l2_size"];
    compile_value->params.trans_a_flag = compile_value->trans_a;
    compile_value->params.trans_b_flag = compile_value->trans_b;
    compile_value->params.ubdb_flag = false;
    compile_value->params.at_l1_flag = true;
    compile_value->params.fused_double_operand_num = compile_value->params.nd_flag ? 1 : 0;
    compile_value->params.aub_double_num = compile_value->params.nd_flag ? 1 : 0;
    compile_value->params.bub_double_num = compile_value->params.nd_flag ? 1 : 0;
    compile_value->params.cub_reused_flag = false;
  }

  if (compile_info.contains("repo_seeds") && compile_info.contains("repo_range")) {
    compile_value->repo_seed_flag = true;
    compile_value->repo_seeds = compile_info["repo_seeds"].get<map<string, vector<int64_t>>>();
    compile_value->repo_range = compile_info["repo_range"].get<map<string, vector<int64_t>>>();
  }

  if (compile_info.contains("cost_range")) {
    compile_value->repo_costmodel_flag = true;
    compile_value->cost_range = compile_info["cost_range"].get<map<string, vector<int64_t>>>();
  }

  return true;
}

bool GemmParseFunc(const string &op_type, const json &compile_info, vector<shared_ptr<GemmCompileInfo>> &compile_data) {
  OP_LOGD(op_type.c_str(), "%s", DebugCompileInfoGEMM(compile_info).c_str());
  string tiling_id("-1");
  if (compile_info.type() == json::value_t::object) {
    compile_data.resize(1);
    OP_TILING_MAKE_SHARED(compile_data[0] = make_shared<GemmCompileInfo>(), return false);
    OP_TILING_CHECK(!(GetGemmCompileValue(compile_info, compile_data[0])),
                    CUBE_INNER_ERR_REPORT(op_type.c_str(), "Parse compile value fail"), return false);
  } else {
    compile_data.resize(compile_info.size());
    for (size_t i = 0; i < compile_info.size(); i++) {
      OP_TILING_MAKE_SHARED(compile_data[i] = make_shared<GemmCompileInfo>(), return false);
      OP_TILING_CHECK(!(GetGemmCompileValue(compile_info[i], compile_data[i])),
                      CUBE_INNER_ERR_REPORT(op_type.c_str(), "Parse compile value fail"), return false);
    }
  }
  return true;
}

bool UpdateGemmCompileValue(const string &op_type, const ge::Operator &op_paras,
                            shared_ptr<GemmCompileInfo> &compile_value) {
  OP_TILING_CHECK((op_paras.GetInputsSize() < 2), CUBE_INNER_ERR_REPORT(op_type.c_str(), "op_paras is unvalid"),
                  return false);

  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(op_desc == nullptr, CUBE_INNER_ERR_REPORT(op_type.c_str(), "op_desc is nullptr"), return false);

  ge::ConstGeTensorDescPtr tensor_a = op_desc->GetInputDescPtr(0);
  ge::ConstGeTensorDescPtr tensor_b = op_desc->GetInputDescPtr(1);
  ge::GeShape ori_shape_a = tensor_a->GetOriginShape();
  ge::GeShape ori_shape_b = tensor_b->GetOriginShape();
  ge::DataType dtype = tensor_a->GetDataType();

  int64_t input_size = 0;
  int64_t hidden_size = 0;
  bool input_size_flag = ge::AttrUtils::GetInt(op_desc, "input_size", input_size);
  bool hidden_size_flag = ge::AttrUtils::GetInt(op_desc, "hidden_size", hidden_size);
  if (input_size_flag && hidden_size_flag) {
    int64_t hidden_size_align = (hidden_size + KBLOCK_SIZE - 1) / KBLOCK_SIZE * KBLOCK_SIZE;
    ori_shape_a.SetDim(1, hidden_size_align * KMULTI);
    int64_t align_dim = (input_size + KBLOCK_SIZE - 1) / KBLOCK_SIZE * KBLOCK_SIZE + hidden_size_align;
    ori_shape_b.SetDim(0, align_dim);
    ori_shape_b.SetDim(1, hidden_size_align * KMULTI);
  }
  OP_TILING_CHECK((!CalcGEMMMknb(op_type, dtype, ori_shape_a, ori_shape_b, compile_value)),
        CUBE_INNER_ERR_REPORT(op_type.c_str(), "Failed to calculate m, k, n, batch"),
  return false);
  return true;
}

string GEMMTilingSelect(const string &op_type, const ge::Operator &op_paras, shared_ptr<GemmCompileInfo> &compile_value,
                        utils::OpRunInfo &run_info) {
  string tiling_id("-1");
  Tiling tiling;
  const auto& dynamic_mode = compile_value->dynamic_mode;
  bool isBatchMatmulMode = (dynamic_mode == "dynamic_mknb");
  OP_TILING_CHECK((dynamic_mode != "dynamic_mkn" && !isBatchMatmulMode),
        CUBE_INNER_ERR_REPORT(op_type.c_str(), "Only support dynamic_mode: dynamic_mkn, dynamic_mknb"),
  return tiling_id);

  OP_TILING_CHECK((!UpdateGemmCompileValue(op_type, op_paras, compile_value)),
                  CUBE_INNER_ERR_REPORT(op_type.c_str(), "Failed to update compile value"), return tiling_id);
  if (compile_value->repo_seed_flag) {
    tiling_id = CheckTilingInRepo(op_type, compile_value->repo_seeds, compile_value->repo_range, compile_value->params,
                                  isBatchMatmulMode);
  }
  OpRunInfoParas runinfoparas;
  runinfoparas.params = compile_value->params;
  if (tiling_id == "-1") {
    if (compile_value->repo_costmodel_flag) {
      tiling_id = CheckTilingInCostModel(op_type, compile_value->cost_range, runinfoparas.params, isBatchMatmulMode);
    }
    if (tiling_id == "-1") {
      GenTiling(op_type, runinfoparas.params, tiling, tiling_id);
      if (tiling_id != "-1") {
        OP_LOGD(op_type.c_str(), "match tiling_id(%s) in cache tiling mode", tiling_id.c_str());
      } else {
        CUBE_INNER_ERR_REPORT(op_type.c_str(), "Failed to calculate tiling from cache tiling mode");
        return tiling_id;
      }
      FillRunInfoParas(tiling, runinfoparas);
    }
  }
  if (tiling_id != "-1") {
    SetRunInfo(isBatchMatmulMode, tiling_id, compile_value->block_dim, runinfoparas, run_info);
  }
  return tiling_id;
}

/*
 * @brief: tiling function of gemm
 * @param [in] op_type: op_type of the gemm
 * @param [in] op_paras: inputs/outputs/atts of the gemm
 * @param [in] op_compile_info: compile time generated info of the gemm
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool GEMMTiling(const string &op_type, const ge::Operator &op_paras, vector<shared_ptr<GemmCompileInfo>> &compile_data,
                utils::OpRunInfo &run_info) {
  try {
    OP_LOGD(op_type.c_str(), "%s", DebugOpParasGEMM(op_paras).c_str());
    string tiling_id("-1");
    for (size_t i = 0; i < compile_data.size(); i++) {
      tiling_id = GEMMTilingSelect(op_type, op_paras, compile_data[i], run_info);
      if (tiling_id != "-1") {
        break;
      }
    }

    if (tiling_id == "-1") {
      CUBE_INNER_ERR_REPORT(op_type.c_str(),
                            "This shape is not covered by any tiling, "
                            "please modify range and recompile");
      return false;
    }

    return true;
  } catch (...) {
    CUBE_INNER_ERR_REPORT(op_type.c_str(), "get unknown exception, please check compile info json.");
    return false;
  }
}

// register tiling interface of the gemm
REGISTER_OP_TILING_V4_CUSTOM(MatMul, GEMMTiling, GemmParseFunc, vector<shared_ptr<GemmCompileInfo>>);
REGISTER_OP_TILING_V4_CUSTOM(MatMulV2, GEMMTiling, GemmParseFunc, vector<shared_ptr<GemmCompileInfo>>);
REGISTER_OP_TILING_V4_CUSTOM(BatchMatMul, GEMMTiling, GemmParseFunc, vector<shared_ptr<GemmCompileInfo>>);
REGISTER_OP_TILING_V4_CUSTOM(BatchMatMulV2, GEMMTiling, GemmParseFunc, vector<shared_ptr<GemmCompileInfo>>);
}  // namespace optiling