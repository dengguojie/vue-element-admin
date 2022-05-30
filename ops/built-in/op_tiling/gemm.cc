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
 * \file gemm.cc
 * \brief tiling function of gemm
 */
#include <string>
#include <vector>
#include <map>

#include <nlohmann/json.hpp>

#include "cube_tiling_runtime.h"
#include "error_util.h"
#include "external/graph/operator.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/types.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "cache_tiling.h"
#include "op_log.h"
#include "op_tiling.h"
#include "op_tiling_util.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_impl_registry.h"
using namespace std;
using json = nlohmann::json;
using utils = ge::TypeUtils;

namespace {
const int64_t kMinDimNum = 3;
const uint64_t kAlignedFlag = 2;
const uint64_t kMinCacheTilingId = 1000000ULL;
const uint64_t kMaxCacheTilingId = 9999999999ULL;

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
const int32_t kBlockSize = 16;
const int64_t DIM_NUM = 3;
const int32_t CACHE_TILING_ID_LEN_NZ = 7;
const int32_t CACHE_TILING_ID_LEN_NZ_SPLIT_K = 8;
const int32_t CACHE_TILING_ID_LEN_ND = 9;
const int32_t CACHE_TILING_ID_LEN_ND_SPLIT_K = 10;
const int64_t KBLOCK_SIZE = 16;
const int64_t KMULTI = 4;
const size_t kInputSizeLimit = 2;
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
  BatchmatmulRunParas params;
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
  int32_t k_al1 = kal1_16 * kBlockSize;
  int32_t k_bl1 = kbl1_16 * kBlockSize;
  int32_t kl1_times = 1;
  int32_t m_aub = 1;
  int32_t n_bub = 1;
  int32_t k_aub = kBlockSize;
  int32_t k_bub = kBlockSize;
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
  BatchmatmulCompileParas params;
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
                  BatchmatmulRunParas &params) {
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

bool CalcGEMMMknb(const ge::OpDescPtr &op_desc, const ge::GeShape &ori_shape_a,
                  const ge::GeShape &ori_shape_b, shared_ptr<GemmCompileInfo> &compile_value,
                  BatchmatmulRunParas &params) {
  ge::DataType dtype = op_desc->GetInputDescPtr(0)->GetDataType();
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
                  CUBE_INNER_ERR_REPORT(op_desc->GetName().c_str(), "The k-axis of a and b tensors must be the same"),
                  return false);

  params.m = ceil(static_cast<double>(ori_shape_a.GetDim(idx_m_of_a)) / block_in);
  params.k = ceil(static_cast<double>(ori_shape_a.GetDim(idx_k_of_a)) / block_reduce);
  params.n = ceil(static_cast<double>(ori_shape_b.GetDim(idx_n_of_b)) / block_out);
  bool unvalid_dim = params.m > INT_MAX || params.k > INT_MAX || params.n > INT_MAX;
  OP_TILING_CHECK(unvalid_dim,
                  CUBE_INNER_ERR_REPORT(op_desc->GetName().c_str(),
                                        "The m,k,n of a and b tensors' shape must not larger than INT_MAX"),
                  return false);
  params.m_32 = static_cast<int32_t>(params.m);
  params.k_32 = static_cast<int32_t>(params.k);
  params.n_32 = static_cast<int32_t>(params.n);

  if (compile_value->params.format_a_nd) {
    params.ori_shape_m = ori_shape_a.GetDim(idx_m_of_a);
    params.ori_shape_k = ori_shape_a.GetDim(idx_k_of_a);
  }
  if (compile_value->params.format_b_nd) {
    params.ori_shape_n = ori_shape_b.GetDim(idx_n_of_b);
  }
  bool unvalid_ori_shape = params.ori_shape_m > INT_MAX || params.ori_shape_k > INT_MAX || params.ori_shape_n > INT_MAX;
  OP_TILING_CHECK(unvalid_ori_shape,
                  CUBE_INNER_ERR_REPORT(op_desc->GetName().c_str(),
                                        "The m,k,n of a and b tensors' ori_shape must not larger than INT_MAX"),
                  return false);
  bool is_nd_input = !compile_value->params.binary_mode_flag && compile_value->params.format_a_nd &&
                     compile_value->params.format_b_nd;
  if (is_nd_input) {
    // Aligned schedule pattern selection is only enabled in ND input format
    bool aligned_m = params.ori_shape_m % block_in == 0;
    bool aligned_k = params.ori_shape_k % block_reduce == 0;
    bool aligned_n = params.ori_shape_n % block_out == 0;
    bool aligned_mkn = aligned_m && aligned_k && aligned_n;
    if (aligned_mkn) {
      params.used_aligned_pattern = true;
    }
  }
  return GetGEMMBatch(op_desc->GetName(), ori_shape_a, ori_shape_b, params);
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
                         const map<string, vector<int64_t>> &repo_range, const BatchmatmulRunParas &params,
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
    BatchmatmulRunParas seed_params;
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
                              const BatchmatmulRunParas &params, bool isBatchMatmulMode) {
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

void FillRunInfoParas(const BatchmatmulCompileParas &params, const Tiling &tiling, OpRunInfoParas &runinfoparas) {
  runinfoparas.batch_dim = tiling.batch_dim;
  runinfoparas.n_dim = tiling.n_dim;
  runinfoparas.m_dim = tiling.m_dim;
  runinfoparas.k_dim = tiling.k_dim;
  runinfoparas.m_l0 = tiling.m_l0;
  runinfoparas.k_l0 = tiling.k_l0;
  runinfoparas.n_l0 = tiling.n_l0;
  if (!tiling.al1_full_load) {
    runinfoparas.k_al1 = tiling.kal1_16 * kBlockSize;
    runinfoparas.m_al1 = tiling.m_al1;
  } else {
    runinfoparas.k_al1 = runinfoparas.params.k_32 / runinfoparas.k_dim * kBlockSize;
    runinfoparas.m_al1 = ceil(static_cast<double>(runinfoparas.params.m_32 / runinfoparas.m_dim) / runinfoparas.m_l0);
  }
  if (!tiling.bl1_full_load) {
    runinfoparas.k_bl1 = tiling.kbl1_16 * kBlockSize;
    runinfoparas.n_bl1 = tiling.n_bl1;
  } else {
    runinfoparas.k_bl1 = runinfoparas.params.k_32 / runinfoparas.k_dim * kBlockSize;
    runinfoparas.n_bl1 = ceil(static_cast<double>(runinfoparas.params.n_32 / runinfoparas.n_dim) / runinfoparas.n_l0);
  }
  runinfoparas.cub_n1 = tiling.n_cub;
  if (params.nd_flag) {
    runinfoparas.k_aub = tiling.k_aub;
    runinfoparas.m_aub = tiling.m_aub;
    runinfoparas.multi_m_ub_l1 = runinfoparas.m_al1 * runinfoparas.m_l0 / tiling.m_aub;
    runinfoparas.multi_k_aub_l1 = runinfoparas.k_al1 / (tiling.k_aub * kBlockSize);
    runinfoparas.a_align_value = tiling.a_align_value;
    runinfoparas.aub_align_bound = tiling.aub_align_bound;
    runinfoparas.k_bub = tiling.k_bub;
    runinfoparas.n_bub = tiling.n_bub;
    runinfoparas.multi_n_ub_l1 = runinfoparas.n_bl1 * runinfoparas.n_l0 / tiling.n_bub;
    runinfoparas.multi_k_bub_l1 = runinfoparas.k_bl1 / (tiling.k_bub * kBlockSize);
    runinfoparas.b_align_value = tiling.b_align_value;
    runinfoparas.bub_align_bound = tiling.bub_align_bound;
  }
  runinfoparas.kal1_16 = runinfoparas.k_al1 / kBlockSize;
  runinfoparas.kbl1_16 = runinfoparas.k_bl1 / kBlockSize;
  runinfoparas.kal0_factor = runinfoparas.kal1_16 / runinfoparas.k_l0;
  runinfoparas.kbl0_factor = runinfoparas.kbl1_16 / runinfoparas.k_l0;
  runinfoparas.kal1_factor = tiling.kal1_factor;
  runinfoparas.kbl1_factor = tiling.kbl1_factor;
  runinfoparas.kl1_times = (runinfoparas.kal1_16 > runinfoparas.kbl1_16)
                           ? (runinfoparas.kal1_16 / runinfoparas.kbl1_16)
                           : (runinfoparas.kbl1_16 / runinfoparas.kal1_16);
  runinfoparas.n_ub_l0_time = runinfoparas.n_l0 / runinfoparas.cub_n1;
  runinfoparas.batch_single_core = ceil(static_cast<double>(runinfoparas.params.batch_32) / runinfoparas.batch_dim);
  runinfoparas.m_single_core = ceil(static_cast<double>(runinfoparas.params.m_32) /
                                    (runinfoparas.m_dim * runinfoparas.m_al1 * runinfoparas.m_l0));
  runinfoparas.n_single_core =
      ceil(static_cast<double>(runinfoparas.params.n_32) /
           (runinfoparas.n_dim * runinfoparas.n_bl1 * runinfoparas.n_ub_l0_time * runinfoparas.cub_n1));
}

void SetRunInfoForCacheTiling(const BatchmatmulCompileParas &params, const OpRunInfoParas &runinfoparas,
                              utils::OpRunInfo &run_info) {
  run_info.AddTilingData(runinfoparas.params.m_32);
  // need to get origin_k to limit the k real length in non-factorial segmentation.
  if (params.format_a_nd) {
    run_info.AddTilingData(static_cast<int32_t>(runinfoparas.params.ori_shape_k));
  } else {
    run_info.AddTilingData(runinfoparas.params.k_32);
  }
  run_info.AddTilingData(runinfoparas.params.n_32);
  if (runinfoparas.params.is_batch_matmul_mode) {
    run_info.AddTilingData(runinfoparas.params.batch_32);
  }
  run_info.AddTilingData(runinfoparas.params.k_32);
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
  if (params.nd_flag) {
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

void SetRunInfo(const BatchmatmulCompileParas &compile_params, string &tiling_id,
                const map<string, int64_t> &block_dim_info, const OpRunInfoParas &runinfoparas,
                utils::OpRunInfo &run_info) {
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
  if (is_cache_tiling) {
    SetRunInfoForCacheTiling(compile_params, runinfoparas, run_info);
  } else {
    if (compile_params.format_a_nd) {
      run_info.AddTilingData(static_cast<int32_t>(runinfoparas.params.ori_shape_m));
      run_info.AddTilingData(static_cast<int32_t>(runinfoparas.params.ori_shape_k));
    } else {
      run_info.AddTilingData(runinfoparas.params.m_32);
      run_info.AddTilingData(runinfoparas.params.k_32);
    }
    if (compile_params.format_b_nd) {
      run_info.AddTilingData(static_cast<int32_t>(runinfoparas.params.ori_shape_n));
    } else {
      run_info.AddTilingData(runinfoparas.params.n_32);
    }
    if (runinfoparas.params.is_batch_matmul_mode) {
      run_info.AddTilingData(runinfoparas.params.batch_32);
    }
  }
}

bool GetGemmCompileValue(const json &compile_info, shared_ptr<GemmCompileInfo> &compile_value) {
  compile_value->dynamic_mode = compile_info["dynamic_mode"];

  const string &format_a = compile_info["format_a"];
  const string &format_b = compile_info["format_b"];
  compile_value->params.format_a_nd = format_a == "ND";
  compile_value->params.format_b_nd = format_b == "ND";
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
    compile_value->params.at_l1_flag = true;
    compile_value->params.fused_double_operand_num =
        compile_value->params.nd_flag && !compile_value->params.split_k_flag ? 1 : 0;
    compile_value->params.aub_double_num = compile_value->params.nd_flag ? 1 : 0;
    compile_value->params.bub_double_num = compile_value->params.nd_flag ? 1 : 0;
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
                            shared_ptr<GemmCompileInfo> &compile_value, BatchmatmulRunParas &run_params) {
  size_t gemm_input_size = op_paras.GetInputsSize();
  OP_TILING_CHECK(
      (gemm_input_size < kInputSizeLimit),
      CUBE_INNER_ERR_REPORT(
          op_type.c_str(),
          "The number of inputs to the operator should not be less than 2, but it is actually %zu.", gemm_input_size),
      return false);

  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(op_desc == nullptr, CUBE_INNER_ERR_REPORT(op_type.c_str(), "op_desc is nullptr"), return false);

  ge::ConstGeTensorDescPtr tensor_a = op_desc->GetInputDescPtr(0);
  ge::ConstGeTensorDescPtr tensor_b = op_desc->GetInputDescPtr(1);
  ge::GeShape ori_shape_a = tensor_a->GetOriginShape();
  ge::GeShape ori_shape_b = tensor_b->GetOriginShape();

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
  OP_TILING_CHECK((!CalcGEMMMknb(op_desc, ori_shape_a, ori_shape_b, compile_value, run_params)),
        CUBE_INNER_ERR_REPORT(op_type.c_str(), "Failed to calculate m, k, n, batch"),
  return false);
  return true;
}

string GEMMTilingSelect(const string &op_type, const ge::Operator &op_paras, shared_ptr<GemmCompileInfo> &compile_value,
                        utils::OpRunInfo &run_info) {
  string tiling_id("-1");
  Tiling tiling;
  OpRunInfoParas runinfoparas;
  const auto& dynamic_mode = compile_value->dynamic_mode;
  bool is_batch_matmul_mode = (dynamic_mode == "dynamic_mknb" || op_type == "BatchMatMulV2" || op_type == "BatchMatMul");
  OP_TILING_CHECK((dynamic_mode != "dynamic_mkn" && !is_batch_matmul_mode),
        CUBE_INNER_ERR_REPORT(op_type.c_str(), "Only support dynamic_mode: dynamic_mkn, dynamic_mknb"),
  return tiling_id);
  runinfoparas.params.is_batch_matmul_mode = is_batch_matmul_mode;
  OP_TILING_CHECK((!UpdateGemmCompileValue(op_type, op_paras, compile_value, runinfoparas.params)),
                  CUBE_INNER_ERR_REPORT(op_type.c_str(), "Failed to update compile value"), return tiling_id);
  if (compile_value->repo_seed_flag) {
    tiling_id = CheckTilingInRepo(op_type, compile_value->repo_seeds, compile_value->repo_range, runinfoparas.params,
                                  is_batch_matmul_mode);
  }

  if (tiling_id == "-1") {
    if (compile_value->repo_costmodel_flag) {
      tiling_id = CheckTilingInCostModel(op_type, compile_value->cost_range, runinfoparas.params, is_batch_matmul_mode);
    }
    if (tiling_id == "-1") {
      GenTiling(op_type, compile_value->params, runinfoparas.params, tiling, tiling_id);
      if (tiling_id != "-1") {
        OP_LOGD(op_type.c_str(), "match tiling_id(%s) in cache tiling mode", tiling_id.c_str());
      } else {
        CUBE_INNER_ERR_REPORT(op_type.c_str(), "Failed to calculate tiling from cache tiling mode");
        return tiling_id;
      }
      FillRunInfoParas(compile_value->params, tiling, runinfoparas);
    }
  }
  if (tiling_id != "-1") {
    SetRunInfo(compile_value->params, tiling_id, compile_value->block_dim, runinfoparas, run_info);
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

namespace gert {
enum DynamicMode {
  DYNAMIC_MKN,
  DYNAMIC_MKNB
};

class GemmCompileInfo : public optiling::CubeCompileInfo {
 public:
  GemmCompileInfo() = default;
  ~GemmCompileInfo() override = default;

  bool AnalyzeExtendInfo(const nlohmann::json &compile_info) override;

  bool trans_a = false;
  bool trans_b = false;
  bool repo_seed_flag = false;
  bool repo_costmodel_flag = false;
  uint32_t workspace_num = 0;
  uint32_t ub_size = 0;
  optiling::BatchmatmulCompileParas params;
  DynamicMode dynamic_mode = DYNAMIC_MKN;
};

bool GemmCompileInfo::AnalyzeExtendInfo(const json &compile_info) {
  const auto &dynamic_mode_str = compile_info["dynamic_mode"];
  if (dynamic_mode_str == "dynamic_mkn") {
    dynamic_mode = DYNAMIC_MKN;
  } else if (dynamic_mode_str == "dynamic_mknb") {
    dynamic_mode = DYNAMIC_MKNB;
  } else {
    return false;
  }

  const string &format_a = compile_info["format_a"];
  const string &format_b = compile_info["format_b"];
  params.format_a_nd = format_a == "ND";
  params.format_b_nd = format_b == "ND";

  const auto &repo_attr = compile_info["attrs"];
  trans_a = repo_attr["transpose_a"];
  trans_b = repo_attr["transpose_b"];

  if (compile_info.contains("binary_mode_flag")) {
    params.binary_mode_flag = compile_info["binary_mode_flag"];
    const auto &block_dim = compile_info["block_dim"].get<map<string, uint32_t>>();
    const auto core_num = block_dim.find("CORE_NUM");
    if (core_num != block_dim.end()) {
      params.core_num = core_num->second;
    }
    auto &binary_attrs = compile_info["binary_attrs"];
    params.bias_flag = binary_attrs["bias_flag"];
    params.nd_flag = binary_attrs["nd_flag"];
    params.split_k_flag = binary_attrs["split_k_flag"];
    params.l2_size = binary_attrs["l2_size"];
    params.trans_a_flag = trans_a;
    params.trans_b_flag = trans_b;
    params.at_l1_flag = true;
    params.fused_double_operand_num = (params.nd_flag && !params.split_k_flag) ? 1 : 0;
    params.aub_double_num = params.nd_flag ? 1 : 0;
    params.bub_double_num = params.nd_flag ? 1 : 0;
  }

  return true;
}

bool GetGEMMBatch(const char *op_name, const Shape &shape_a, const Shape &shape_b,
                  optiling::BatchmatmulRunParas &params) {
  int32_t num_dima = static_cast<int32_t>(shape_a.GetDimNum());
  int32_t num_dimb = static_cast<int32_t>(shape_b.GetDimNum());
  if (num_dima < kMinDimNum && num_dimb < kMinDimNum) {
    params.batch = 1;
    params.batch_32 = 1;
    return true;
  }

  params.b_have_batch = num_dimb > kNumTwo;

  const auto &shape_short = num_dima < num_dimb ? shape_a : shape_b;
  const auto &shape_long = num_dima < num_dimb ? shape_b : shape_a;
  auto shape_broadcast(shape_long);
  int32_t num_dim = shape_long.GetDimNum();
  int32_t offset = num_dim - shape_short.GetDimNum();

  for (int32_t i = num_dim - kMinBatchDimNum; i >= offset; --i) {
    int64_t short_value = shape_short.GetDim(i - offset);
    int64_t long_value = shape_long.GetDim(i);
    OP_TILING_CHECK((short_value != long_value && short_value != 1 && long_value != 1),
                    CUBE_INNER_ERR_REPORT(op_name, "Tensor a and b do not meet the broadcast rule"), return false);

    shape_broadcast.SetDim(i, std::max(short_value, long_value));
  }

  int64_t batch_value = 1;
  for (int32_t i = 0; i < num_dim - kNumTwo; ++i) {
    batch_value *= shape_broadcast.GetDim(i);
  }
  params.batch = batch_value;
  OP_TILING_CHECK((batch_value > INT_MAX),
                  CUBE_INNER_ERR_REPORT(op_name, "The batch of a and b tensors' shape must not larger than INT_MAX"),
                  return false);
  params.batch_32 = batch_value;
  return true;
}

bool CalcGEMMMknb(const TilingContext *context, const Shape &ori_shape_a, const Shape &ori_shape_b,
                  const GemmCompileInfo &compile_value, optiling::BatchmatmulRunParas &params) {
  auto dtype = context->GetInputDesc(0)->GetDataType();
  auto op_name = context->GetNodeName();
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
  if (compile_value.trans_a) {
    idx_m_of_a = num_dima - 1;
    idx_k_of_a = num_dima - kNumTwo;
  }
  if (compile_value.trans_b) {
    idx_k_of_b = num_dimb - 1;
    idx_n_of_b = num_dimb - kNumTwo;
  }

  OP_TILING_CHECK((ori_shape_a.GetDim(idx_k_of_a) != ori_shape_b.GetDim(idx_k_of_b)),
        CUBE_INNER_ERR_REPORT(op_name, "The k-axis of a and b tensors must be the same"), return false);

  params.m = ceil(static_cast<double>(ori_shape_a.GetDim(idx_m_of_a)) / block_in);
  params.k = ceil(static_cast<double>(ori_shape_a.GetDim(idx_k_of_a)) / block_reduce);
  params.n = ceil(static_cast<double>(ori_shape_b.GetDim(idx_n_of_b)) / block_out);
  bool unvalid_dim = params.m > INT_MAX || params.k > INT_MAX || params.n > INT_MAX;
  OP_TILING_CHECK(unvalid_dim,
                  CUBE_INNER_ERR_REPORT(op_name, "The m,k,n of a and b tensors' shape must not larger than INT_MAX"),
                  return false);
  params.m_32 = static_cast<int32_t>(params.m);
  params.k_32 = static_cast<int32_t>(params.k);
  params.n_32 = static_cast<int32_t>(params.n);

  if (compile_value.params.format_a_nd) {
    params.ori_shape_m = ori_shape_a.GetDim(idx_m_of_a);
    params.ori_shape_k = ori_shape_a.GetDim(idx_k_of_a);
  }
  if (compile_value.params.format_b_nd) {
    params.ori_shape_n = ori_shape_b.GetDim(idx_n_of_b);
  }
  bool unvalid_ori_shape = params.ori_shape_m > INT_MAX || params.ori_shape_k > INT_MAX || params.ori_shape_n > INT_MAX;
  OP_TILING_CHECK(unvalid_ori_shape,
        CUBE_INNER_ERR_REPORT(op_name, "The m,k,n of a and b tensors' ori_shape must not larger than INT_MAX"),
        return false);
  bool is_nd_input =
      !compile_value.params.binary_mode_flag && compile_value.params.format_a_nd && compile_value.params.format_b_nd;
  if (is_nd_input) {
    // Aligned schedule pattern selection is only enabled in ND input format
    bool aligned_m = params.ori_shape_m % block_in == 0;
    bool aligned_k = params.ori_shape_k % block_reduce == 0;
    bool aligned_n = params.ori_shape_n % block_out == 0;
    bool aligned_mkn = aligned_m && aligned_k && aligned_n;
    if (aligned_mkn) {
      params.used_aligned_pattern = true;
    }
  }
  return GetGEMMBatch(op_name, ori_shape_a, ori_shape_b, params);
}

bool UpdateGemmRunTimeParams(const char *op_name, TilingContext *context, const GemmCompileInfo &compile_value,
                             optiling::BatchmatmulRunParas &params) {
  // use copy, maybe modify for input_size and hidden_size
  gert::Shape ori_shape_a = context->GetInputShape(0)->GetOriginShape();
  gert::Shape ori_shape_b = context->GetInputShape(1)->GetOriginShape();

  auto *attrs = context->GetAttrs();
  if (attrs != nullptr && attrs->GetAttrNum() >= 4) {         // 4 temp logic, remove when GE implement private attr
    size_t input_size_idx = attrs->GetAttrNum() >= 5 ? 3: 2;  // V1 have 5 attr, idx: 3; V2 have 4 attr, idx: 2
    size_t hidden_size_idx = attrs->GetAttrNum() >= 5 ? 4: 3; // V1 have 5 attr, idx: 4; V2 have 4 attr, idx: 3
    // if idx over limit, we get nullptr
    auto input_size = attrs->GetAttrPointer<int64_t>(input_size_idx);
    auto hidden_size = attrs->GetAttrPointer<int64_t>(hidden_size_idx);
    if (input_size != nullptr && hidden_size != nullptr && *input_size > 0 && *hidden_size > 0) {
      OP_LOGD(op_name, "get private attr, input_size: %ld, hidden_size: %ld", *input_size, *hidden_size);
      int64_t hidden_size_align = (*hidden_size + kBlockSize - 1) / kBlockSize * kBlockSize;
      ori_shape_a.SetDim(1, hidden_size_align * KMULTI);
      int64_t align_dim = (*input_size + kBlockSize - 1) / kBlockSize * kBlockSize + hidden_size_align;
      ori_shape_b.SetDim(0, align_dim);
      ori_shape_b.SetDim(1, hidden_size_align * KMULTI);
    }
  }

  return CalcGEMMMknb(context, ori_shape_a, ori_shape_b, compile_value, params);
}

void SetRunInfoForCacheTiling(const optiling::BatchmatmulCompileParas &compile_params,
                              const optiling::OpRunInfoParas &runinfoparas, TilingData *tiling_data) {
  if (compile_params.format_a_nd) {
    tiling_data->Append(runinfoparas.params.m_32);
    tiling_data->Append(runinfoparas.params.k_32);
  }
  if (compile_params.format_b_nd) {
    tiling_data->Append(runinfoparas.params.n_32);
  }
  tiling_data->Append(runinfoparas.batch_single_core);
  tiling_data->Append(runinfoparas.m_single_core);
  tiling_data->Append(runinfoparas.n_single_core);
  tiling_data->Append(runinfoparas.batch_dim);
  tiling_data->Append(runinfoparas.n_dim);
  tiling_data->Append(runinfoparas.m_dim);
  tiling_data->Append(runinfoparas.k_dim);
  tiling_data->Append(runinfoparas.m_al1);
  tiling_data->Append(runinfoparas.n_bl1);
  tiling_data->Append(runinfoparas.cub_n1);
  tiling_data->Append(runinfoparas.m_l0);
  tiling_data->Append(runinfoparas.k_l0);
  tiling_data->Append(runinfoparas.n_ub_l0_time);
  tiling_data->Append(runinfoparas.kal0_factor);
  tiling_data->Append(runinfoparas.kbl0_factor);
  tiling_data->Append(runinfoparas.kal1_factor);
  tiling_data->Append(runinfoparas.kbl1_factor);
  tiling_data->Append(runinfoparas.kal1_16);
  tiling_data->Append(runinfoparas.kbl1_16);
  tiling_data->Append(runinfoparas.kl1_times);
  if (compile_params.nd_flag) {
    tiling_data->Append(runinfoparas.m_aub);
    tiling_data->Append(runinfoparas.n_bub);
    tiling_data->Append(runinfoparas.k_aub);
    tiling_data->Append(runinfoparas.k_bub);
    tiling_data->Append(runinfoparas.multi_n_ub_l1);
    tiling_data->Append(runinfoparas.multi_m_ub_l1);
    tiling_data->Append(runinfoparas.multi_k_aub_l1);
    tiling_data->Append(runinfoparas.multi_k_bub_l1);
    tiling_data->Append(runinfoparas.a_align_value);
    tiling_data->Append(runinfoparas.b_align_value);
    tiling_data->Append(runinfoparas.aub_align_bound);
    tiling_data->Append(runinfoparas.bub_align_bound);
  }
}

void SetRunInfo(uint64_t tiling_id, const map<uint64_t, uint32_t> &block_dim_info,
                const optiling::BatchmatmulCompileParas &compile_params, const optiling::OpRunInfoParas &runinfoparas,
                TilingContext *context) {
  bool is_cache_tiling = tiling_id >= kMinCacheTilingId && tiling_id <= kMaxCacheTilingId;
  if (is_cache_tiling) {
    int32_t block_dim = runinfoparas.batch_dim * runinfoparas.n_dim * runinfoparas.m_dim * runinfoparas.k_dim;
    context->SetBlockDim(static_cast<uint32_t>(block_dim));
  } else {
    auto block_dim_value = block_dim_info.find(tiling_id);
    if (block_dim_value != block_dim_info.end()) {
      context->SetBlockDim(block_dim_value->second);
    }
  }

  // Used Aligned Pattern if the input shape is aligned. Only enabled in ND input format.
  if (runinfoparas.params.used_aligned_pattern && !is_cache_tiling && tiling_id > 0) {
    std::array<int64_t, 20> tiling_id_seq;  // 20: uint64_t max length
    size_t length = 0;
    const int64_t kBase = 10;
    while (tiling_id > 0) {
      tiling_id_seq[length++] = tiling_id % kBase;
      tiling_id /= kBase;
    }

    tiling_id_seq[--length] = kAlignedFlag;
    while (length != 0) {
      tiling_id = tiling_id * kBase + tiling_id_seq[length--];
    }
    tiling_id = tiling_id * kBase + tiling_id_seq[0];
  }
  context->SetTilingKey(tiling_id);

  TilingData *tiling_data = context->GetRawTilingData();
  if (compile_params.format_a_nd) {
    tiling_data->Append(static_cast<int32_t>(runinfoparas.params.ori_shape_m));
    tiling_data->Append(static_cast<int32_t>(runinfoparas.params.ori_shape_k));
  } else {
    tiling_data->Append(runinfoparas.params.m_32);
    tiling_data->Append(runinfoparas.params.k_32);
  }
  if (compile_params.format_b_nd) {
    tiling_data->Append(static_cast<int32_t>(runinfoparas.params.ori_shape_n));
  } else {
    tiling_data->Append(runinfoparas.params.n_32);
  }
  if (runinfoparas.params.is_batch_matmul_mode) {
    tiling_data->Append(runinfoparas.params.batch_32);
  }
  if (is_cache_tiling) {
    SetRunInfoForCacheTiling(compile_params, runinfoparas, tiling_data);
  }
}

uint64_t GEMMTilingSelect(bool is_batch_matmul_op, TilingContext *context, const GemmCompileInfo &compile_value) {
  const char *op_name = context->GetNodeName();
  uint64_t tiling_id = optiling::kInvalidTilingId;
  optiling::OpRunInfoParas runinfoparas;
  const auto &dynamic_mode = compile_value.dynamic_mode;
  bool is_batch_matmul_mode = is_batch_matmul_op || (dynamic_mode == DYNAMIC_MKNB);

  OP_TILING_CHECK((dynamic_mode != DYNAMIC_MKN && !is_batch_matmul_mode),
        CUBE_INNER_ERR_REPORT(op_name, "Only support dynamic_mode: dynamic_mkn, dynamic_mknb"), return tiling_id);

  runinfoparas.params.is_batch_matmul_mode = is_batch_matmul_mode;
  OP_TILING_CHECK((!UpdateGemmRunTimeParams(op_name, context, compile_value, runinfoparas.params)),
        CUBE_INNER_ERR_REPORT(op_name, "Failed to update runtime params"), return tiling_id);

  size_t dim_num = 4;  // 4 items at most: mkn and batch
  int64_t shape_for_range_match[dim_num] = {runinfoparas.params.m, runinfoparas.params.k, runinfoparas.params.n,
                                            runinfoparas.params.batch};
  dim_num = is_batch_matmul_mode ? dim_num : dim_num - 1;
  if (!compile_value.repo_range.empty()) {
    tiling_id = compile_value.CheckTilingInRepo(op_name, shape_for_range_match, dim_num);
  }

  if (tiling_id == optiling::kInvalidTilingId) {
    if (!compile_value.cost_range.empty()) {
      tiling_id = compile_value.CheckTilingInCostModel(op_name, shape_for_range_match, dim_num);
    }
    if (tiling_id == optiling::kInvalidTilingId) {
      optiling::Tiling tiling;
      std::string str_tiling_id;  // when remove V4 interface, modify GenTiling
      optiling::GenTiling(op_name, compile_value.params, runinfoparas.params, tiling, str_tiling_id);
      tiling_id = stoull(str_tiling_id);
      if (tiling_id != optiling::kInvalidTilingId) {
        OP_LOGD(op_name, "match tiling_id(%llu) in cache tiling mode", tiling_id);
      } else {
        CUBE_INNER_ERR_REPORT(op_name, "Failed to calculate tiling from cache tiling mode");
        return tiling_id;
      }
      optiling::FillRunInfoParas(compile_value.params, tiling, runinfoparas);
    }
  }
  if (tiling_id != optiling::kInvalidTilingId) {
    SetRunInfo(tiling_id, compile_value.block_dim, compile_value.params, runinfoparas, context);
  }
  return tiling_id;
}

ge::graphStatus TilingForGemm(TilingContext *context, bool is_batch_matmul_op) {
  auto shape_x1 = context->GetInputShape(0);
  auto shape_x2 = context->GetInputShape(1);
  auto src_td = context->GetInputDesc(0);
  if (shape_x1 == nullptr || shape_x2 == nullptr || src_td == nullptr) {
    return ge::GRAPH_FAILED;
  }

  auto compile_info = reinterpret_cast<const GemmCompileInfo *>(context->GetCompileInfo());
  if (compile_info == nullptr) {
    return ge::GRAPH_FAILED;
  }

  OP_LOGD(context->GetNodeName(), "%s", optiling::DebugTilingContext(context).c_str());
  uint64_t tiling_id = GEMMTilingSelect(is_batch_matmul_op, context, *compile_info);
  if (tiling_id == optiling::kInvalidTilingId) {
    CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                          "This shape is not covered by any tiling, please modify range and recompile");
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForMatMul(TilingContext *context) {
  OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("MatMul", "context is null"), return ge::GRAPH_FAILED);
  return TilingForGemm(context, false);
}

ge::graphStatus TilingForBatchMatMul(TilingContext *context) {
  OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("BatchMatMul", "context is null"), return ge::GRAPH_FAILED);
  return TilingForGemm(context, true);
}

ge::graphStatus GemmParseFunc(KernelContext *context) {
  OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("MatMul", "context is null"), return ge::GRAPH_FAILED);
  auto compile_info = context->GetOutputPointer<GemmCompileInfo>(0);
  auto json_str = context->GetInputStrPointer(0);
  OP_TILING_CHECK(compile_info == nullptr || json_str == nullptr,
                  CUBE_INNER_ERR_REPORT("nil", "compile_info or json is null"), return ge::GRAPH_FAILED);

  auto compute_node = reinterpret_cast<const ComputeNodeInfo *>(context->GetComputeNodeExtend());
  OP_TILING_CHECK(compute_node == nullptr, CUBE_INNER_ERR_REPORT("nil", "compute_node is null"),
                  return ge::GRAPH_FAILED);
  auto op_name = compute_node->GetNodeName();
  OP_TILING_CHECK(!compile_info->AnalyzeCompileInfo(op_name, json_str),
                  CUBE_INNER_ERR_REPORT(op_name, "failed to analyze compile info"), return ge::GRAPH_FAILED);

  bool is_batch_matmul = (compile_info->dynamic_mode == DYNAMIC_MKNB ||
                          strncmp(compute_node->GetNodeType(), "BatchMatMul", sizeof("BatchMatMul") - 1) == 0);
  OP_TILING_CHECK(!compile_info->CheckRangeSize(is_batch_matmul ? 4 : 3),  // 3: mkn, 4: mkn and batch
                  CUBE_INNER_ERR_REPORT(op_name, "repo_range/repo_seeds/cost_range invalid"), return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

IMPL_OP(BatchMatMul)
    .Tiling(TilingForBatchMatMul)
    .TilingParse<GemmCompileInfo>(GemmParseFunc);

IMPL_OP(BatchMatMulV2)
    .Tiling(TilingForBatchMatMul)
    .TilingParse<GemmCompileInfo>(GemmParseFunc);

IMPL_OP(MatMul)
    .Tiling(TilingForMatMul)
    .TilingParse<GemmCompileInfo>(GemmParseFunc);

IMPL_OP(MatMulV2)
    .Tiling(TilingForMatMul)
    .TilingParse<GemmCompileInfo>(GemmParseFunc);
}  // namespace gert