/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "external/graph/operator.h"
#include "graph/types.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "op_tiling.h"
#include "../op_proto/util/error_util.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/op_desc_utils.h"
#include "../op_proto/util/axis_util.h"
using namespace std;
using json = nlohmann::json;
using utils = ge::TypeUtils;

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

struct BatchmatmulParas {
  int64_t m;
  int64_t k;
  int64_t n;
  int64_t batch;
  int64_t ori_shape_M;
  int64_t ori_shape_K;
  int64_t ori_shape_N;
  std::string format_a;
  std::string format_b;
};

namespace optiling {

bool GetGEMMBatch(const string& op_type, const ge::GeShape& shape_a, const ge::GeShape& shape_b, BatchmatmulParas& params) {
  int32_t num_dima = shape_a.GetDimNum();
  int32_t num_dimb = shape_b.GetDimNum();
  if (num_dima < 3 && num_dimb < 3) {
    params.batch = 1;
    return true;
  }

  const ge::GeShape& shape_short = num_dima < num_dimb ? shape_a : shape_b;
  const ge::GeShape& shape_long = num_dima < num_dimb ? shape_b : shape_a;
  ge::GeShape shape_broadcast(shape_long);
  int32_t num_dim = shape_long.GetDimNum();
  int32_t offset = num_dim - shape_short.GetDimNum();

  for (int i = num_dim - 3; i >= offset; --i) {
    int64_t short_value = shape_short.GetDim(i - offset);
    int64_t long_value = shape_long.GetDim(i);
    CHECK((short_value != long_value && short_value != 1 && long_value != 1),
          CUBE_INNER_ERR_REPORT(op_type.c_str(), "Tensor a and b do not meet the broadcast rule"), return false);

    shape_broadcast.SetDim(i, max(short_value, long_value));
  }

  int64_t batch_value = 1;
  for (int i = 0; i < num_dim - 2; ++i) {
    batch_value *= shape_broadcast.GetDim(i);
  }
  params.batch = batch_value;
  return true;
}

bool CalcGEMMMknb(const string& op_type, const json& compile_info, ge::DataType dtype, const ge::GeShape& ori_shape_a,
                  const ge::GeShape& ori_shape_b, BatchmatmulParas& params) {
  int32_t block_reduce = kBlockReduce, block_in = kBlockIn, block_out = kBlockOut;
  if (dtype == ge::DT_INT8 || dtype == ge::DT_UINT8) {
    block_reduce = kBlockReduceS8;
  }

  int32_t num_dima = ori_shape_a.GetDimNum();
  int32_t num_dimb = ori_shape_b.GetDimNum();
  int32_t idx_m_of_a = num_dima - 2;
  int32_t idx_k_of_a = num_dima - 1;
  int32_t idx_k_of_b = num_dimb - 2;
  int32_t idx_n_of_b = num_dimb - 1;
  auto& repo_attr = compile_info["attrs"];
  auto trans_a = repo_attr["transpose_a"];
  auto trans_b = repo_attr["transpose_b"];
  if (trans_a) {
    idx_m_of_a = num_dima - 1;
    idx_k_of_a = num_dima - 2;
  }
  if (trans_b) {
    idx_k_of_b = num_dimb - 1;
    idx_n_of_b = num_dimb - 2;
  }

  CHECK((ori_shape_a.GetDim(idx_k_of_a) != ori_shape_b.GetDim(idx_k_of_b)),
        CUBE_INNER_ERR_REPORT(op_type.c_str(), "The k-axis of a and b tensors must be the same"), return false);

  params.m = std::ceil(static_cast<double>(ori_shape_a.GetDim(idx_m_of_a)) / block_in);
  params.k = std::ceil(static_cast<double>(ori_shape_a.GetDim(idx_k_of_a)) / block_reduce);
  params.n = std::ceil(static_cast<double>(ori_shape_b.GetDim(idx_n_of_b)) / block_out);
  if (params.format_a == "ND") {
      params.ori_shape_M = ori_shape_a.GetDim(idx_m_of_a);
      params.ori_shape_K = ori_shape_a.GetDim(idx_k_of_a);
    }
    if (params.format_b == "ND") {
      params.ori_shape_N = ori_shape_b.GetDim(idx_n_of_b);
    }
  return GetGEMMBatch(op_type, ori_shape_a, ori_shape_b, params);
}

string StringTeOperator(const ge::TensorDesc &tensor) {
    std::ostringstream oss;
    oss << "\t\tori_shape: (";
    for (auto dim : tensor.GetOriginShape().GetDims()) {
      oss << dim << ", ";
    }
    oss << ")" << endl;

    oss << "\t\tshape: (";
    for (auto dim : tensor.GetShape().GetDims()) {
      oss << dim << ", ";
    }
    oss << ")" << endl;
    oss << "\t\tdtype: " << ge::TypeUtils::DataTypeToSerialString(tensor.GetDataType()) << endl;
    oss << "\t\tformat: " << ge::TypeUtils::FormatToSerialString(tensor.GetFormat()) << endl;
    oss << "\t\tori_format: " << ge::TypeUtils::FormatToSerialString(tensor.GetOriginFormat()) << endl;

    return oss.str();
}

string DebugInfoGEMM(const ge::Operator &op_paras, const json &compile_info) {
  std::ostringstream oss;
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

  oss << "compile_info:" << endl;
  oss << compile_info.dump() << endl;
  return oss.str();
}

std::string CheckTilingInRepo(const std::string &op_type, const json &compile_info, const BatchmatmulParas& params,
                               bool isBatchMatmulMode) {
  string tiling_id("-1");
  int64_t min_distance = LLONG_MAX;
  auto &repo_seed = compile_info["repo_seeds"];
  auto &repo_range = compile_info["repo_range"];
  auto element_seed = repo_seed.begin();
  auto element_range = repo_range.begin();
  while (element_seed != repo_seed.end() && element_range != repo_range.end()) {
    CHECK((element_seed.key() != element_range.key()),
          CUBE_INNER_ERR_REPORT(op_type.c_str(), "repo_seed is inconsistent with repo_range"), return tiling_id);

    auto& seed = element_seed.value();
    auto& range = element_range.value();

    auto in_range = range[kIdxMLow] <= params.m && params.m <= range[kIdxMHigh] &&
                    range[kIdxKLow] <= params.k && params.k <= range[kIdxKHigh] &&
                    range[kIdxNLow] <= params.n && params.n <= range[kIdxNHigh];
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
        tiling_id = element_seed.key();
      }
    }
    ++element_seed;
    ++element_range;
  }
  return tiling_id;
}

std::string GEMMTilingSelect(const std::string &op_type, const ge::Operator &op_paras, const json &compile_info,
                             utils::OpRunInfo& run_info) {
  string tiling_id("-1");
  const auto &dynamic_mode = compile_info["dynamic_mode"];
  bool isBatchMatmulMode = dynamic_mode == "dynamic_mknb";
  // Update ori_shape info
  string format_a("FRACTAL_NZ");
  string format_b("FRACTAL_NZ");
  if (compile_info.contains("format_a") && compile_info.contains("format_b"))
  {
    format_a = compile_info["format_a"];
    format_b = compile_info["format_b"];
  }

  CHECK((dynamic_mode != "dynamic_mkn" && !isBatchMatmulMode),
        CUBE_INNER_ERR_REPORT(op_type.c_str(), "Only support dynamic_mode: dynamic_mkn, dynamic_mknb"),
        return tiling_id);

  CHECK((op_paras.GetInputsSize() < 2), CUBE_INNER_ERR_REPORT(op_type.c_str(), "op_paras is null"), return tiling_id);

  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  CHECK(op_desc == nullptr, CUBE_INNER_ERR_REPORT(op_type.c_str(), "op_desc is nullptr"), return tiling_id);

  ge::ConstGeTensorDescPtr tensor_a = op_desc->GetInputDescPtr(0);
  ge::ConstGeTensorDescPtr tensor_b = op_desc->GetInputDescPtr(1);
  const ge::GeShape& ori_shape_a = tensor_a->GetOriginShape();
  const ge::GeShape& ori_shape_b = tensor_b->GetOriginShape();
  ge::DataType dtype = tensor_a->GetDataType();
  BatchmatmulParas params;
  params.format_a = format_a;
  params.format_b = format_b;
  CHECK((!CalcGEMMMknb(op_type, compile_info, dtype, ori_shape_a, ori_shape_b, params)),
        CUBE_INNER_ERR_REPORT(op_type.c_str(), "Failed to calculate m, k, n, batch"), return tiling_id);
  tiling_id = CheckTilingInRepo(op_type, compile_info, params, isBatchMatmulMode);
  // check tiling of costmodel
  if (tiling_id == "-1") {
    for (auto &element : compile_info["cost_range"].items()) {
      const auto &range = element.value();
      auto in_range = range[kIdxMLow] <= params.m && params.m <= range[kIdxMHigh] &&
                      range[kIdxKLow] <= params.k && params.k <= range[kIdxKHigh] &&
                      range[kIdxNLow] <= params.n && params.n <= range[kIdxNHigh];
      if (isBatchMatmulMode) {
        in_range = in_range && range[kIdxBLow] <= params.batch && params.batch <= range[kIdxBHigh];
      }
      if (in_range) {
        tiling_id = element.key();
        OP_LOGD(op_type.c_str(), "match tiling_id(%s) in costmodel", tiling_id.c_str());
        break;
      }
    }
  } else {
    OP_LOGD(op_type.c_str(), "match tiling_id(%s) in repository", tiling_id.c_str());
  }

  if (tiling_id != "-1") {
    run_info.SetBlockDim(static_cast<int32_t>(compile_info["block_dim"][tiling_id]));
    run_info.SetTilingKey(stoi(tiling_id));
    if (format_a == "ND"){
      run_info.AddTilingData(static_cast<int32_t>(params.ori_shape_M));
      run_info.AddTilingData(static_cast<int32_t>(params.ori_shape_K));
    } else {
      run_info.AddTilingData(static_cast<int32_t>(params.m));
      run_info.AddTilingData(static_cast<int32_t>(params.k));
    }
    if (format_b == "ND"){
      run_info.AddTilingData(static_cast<int32_t>(params.ori_shape_N));
    } else {
      run_info.AddTilingData(static_cast<int32_t>(params.n));
    }
    if (isBatchMatmulMode) {
      run_info.AddTilingData(static_cast<int32_t>(params.batch));
    }
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
bool GEMMTiling(const std::string &op_type, const ge::Operator &op_paras, const json &compile_info,
                utils::OpRunInfo& run_info) {
  try {
    OP_LOGD(op_type.c_str(), "%s", DebugInfoGEMM(op_paras, compile_info).c_str());
    std::string tiling_id("-1");
    if (compile_info.type() == json::value_t::object) {
      tiling_id = GEMMTilingSelect(op_type, op_paras, compile_info, run_info);
    } else {
      for (std::size_t i = 0; i < compile_info.size(); i++) {
        tiling_id = GEMMTilingSelect(op_type, op_paras, compile_info[i], run_info);
        if (tiling_id != "-1") {
          break;
        }
      }
    }

    if (tiling_id == "-1") {
      CUBE_INNER_ERR_REPORT(op_type.c_str(), "This shape is not covered by any tiling, "
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
REGISTER_OP_TILING_FUNC_BUFFERED_V2(MatMul, GEMMTiling);
REGISTER_OP_TILING_FUNC_BUFFERED_V2(MatMulV2, GEMMTiling);
REGISTER_OP_TILING_FUNC_BUFFERED_V2(BatchMatMul, GEMMTiling);
REGISTER_OP_TILING_FUNC_BUFFERED_V2(BatchMatMulV2, GEMMTiling);
}  // namespace optiling
