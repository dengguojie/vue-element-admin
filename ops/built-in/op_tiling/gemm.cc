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


namespace optiling {

bool GetGEMMBatch(const string &op_type, const vector<int64_t> &shape_a, const vector<int64_t> &shape_b,
                  int64_t *batch) {
  if (shape_a.size() < 3 && shape_b.size() < 3) {
    *batch = 1;
    return true;
  }

  const auto &shape_short = shape_a.size() <= shape_b.size() ? shape_a : shape_b;
  const auto &shape_long = shape_a.size() > shape_b.size() ? shape_a : shape_b;
  vector<int64_t> shape_broadcast(shape_long);

  for (int i = shape_broadcast.size() - 3; i >= static_cast<int>(shape_broadcast.size() - shape_short.size()); --i) {
    if (shape_short[i] != shape_long[i] && shape_short[i] != 1 && shape_long[i] != 1) {
      CUBE_INNER_ERR_REPORT(op_type.c_str(), "Tensor a and b do not meet the broadcast rule");
      return false;
    }
    shape_broadcast[i] = max(shape_short[i], shape_long[i]);
  }

  *batch = std::accumulate(shape_broadcast.begin(), shape_broadcast.end() - 2, 1, std::multiplies<double>());
  return true;
}

bool CalcGEMMMknb(const string &op_type, const json &compile_info,
                  const ge::TensorDesc &tensor_a, const ge::TensorDesc &tensor_b,
                  int64_t *m, int64_t *k, int64_t *n, int64_t *batch) {
    int32_t block_reduce = kBlockReduce, block_in = kBlockIn, block_out = kBlockOut;
    if (tensor_a.GetDataType() == ge::DT_INT8 || tensor_a.GetDataType() == ge::DT_UINT8) {
      block_reduce = kBlockReduceS8;
    }

    int32_t idx_m_of_a = -2;
    int32_t idx_k_of_a = -1;
    int32_t idx_k_of_b = -2;
    int32_t idx_n_of_b = -1;

    auto trans_a = compile_info["attrs"]["transpose_a"].get<bool>();
    auto trans_b = compile_info["attrs"]["transpose_b"].get<bool>();
    if (trans_a) {
      idx_m_of_a = -1;
      idx_k_of_a = -2;
    }
    if (trans_b) {
      idx_k_of_b = -1;
      idx_n_of_b = -2;
    }

    auto tansor_a_ori_shape = tensor_a.GetOriginShape().GetDims();
    auto tansor_b_ori_shape = tensor_b.GetOriginShape().GetDims();

    idx_m_of_a += tansor_a_ori_shape.size();
    idx_k_of_a += tansor_a_ori_shape.size();
    idx_k_of_b += tansor_b_ori_shape.size();
    idx_n_of_b += tansor_b_ori_shape.size();

    if (tansor_a_ori_shape[idx_k_of_a] != tansor_b_ori_shape[idx_k_of_b]) {
      CUBE_INNER_ERR_REPORT(op_type.c_str(), "The k-axis of a and b tensors must be the same");
      return false;
    }
    
    *m = std::ceil(static_cast<double>(tansor_a_ori_shape[idx_m_of_a]) / block_in);
    *k = std::ceil(static_cast<double>(tansor_a_ori_shape[idx_k_of_a]) / block_reduce);
    *n = std::ceil(static_cast<double>(tansor_b_ori_shape[idx_n_of_b]) / block_out);

    return GetGEMMBatch(op_type, tansor_a_ori_shape, tansor_b_ori_shape, batch);
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

std::string GEMMTilingSelect(const std::string &op_type, const ge::Operator &op_paras, const json &compile_info,
                             utils::OpRunInfo& run_info) {
  string tiling_id("-1");
  auto dynamic_mode = compile_info["dynamic_mode"].get<std::string>();
  if (dynamic_mode != "dynamic_mkn" && dynamic_mode != "dynamic_mknb") {
    CUBE_INNER_ERR_REPORT(op_type.c_str(), "Only support dynamic_mode: dynamic_mkn, dynamic_mknb");
    return tiling_id;
  }
  if (op_paras.GetInputsSize() < 2) {
    CUBE_INNER_ERR_REPORT(op_type.c_str(), "op_paras is null");
    return tiling_id;
  }
  auto tensor_a = op_paras.GetInputDesc(0);
  auto tensor_b = op_paras.GetInputDesc(1);
  int64_t m, k, n, batch;
  if (!CalcGEMMMknb(op_type, compile_info, tensor_a, tensor_b, &m, &k, &n, &batch)) {
    CUBE_INNER_ERR_REPORT(op_type.c_str(), "Failed to calculate m, k, n, batch");
    return tiling_id;
  }

  int64_t min_distance = LLONG_MAX;
  // check tiling of repository
  for (auto &element : compile_info["repo_seeds"].items()) {
    auto seed_mkn = element.value().get<std::vector<int64_t>>();
    auto &range = compile_info["repo_range"][element.key()];

    auto in_range = range[kIdxMLow] <= m && m <= range[kIdxMHigh] &&
                    range[kIdxKLow] <= k && k <= range[kIdxKHigh] &&
                    range[kIdxNLow] <= n && n <= range[kIdxNHigh];
    if (in_range) {
      int64_t dist = (m - seed_mkn[kIdxM]) * (m - seed_mkn[kIdxM]) +
                     (k - seed_mkn[kIdxK]) * (k - seed_mkn[kIdxK]) +
                     (n - seed_mkn[kIdxN]) * (n - seed_mkn[kIdxN]);
      if (dynamic_mode == "dynamic_mknb") {
        dist += (batch - seed_mkn[kIdxBatch]) * (batch - seed_mkn[kIdxBatch]);
      }
      if (dist < min_distance) {
        min_distance = dist;
        tiling_id = element.key();
      }
    }
  }

  // check tiling of costmodel
  if (tiling_id == "-1") {
    for (auto &element : compile_info["cost_range"].items()) {
      auto range = element.value().get<std::vector<int64_t>>();
      auto in_range = range[kIdxMLow] <= m && m <= range[kIdxMHigh] &&
                      range[kIdxKLow] <= k && k <= range[kIdxKHigh] &&
                      range[kIdxNLow] <= n && n <= range[kIdxNHigh];
      if (dynamic_mode == "dynamic_mknb") {
        in_range = in_range && range[kIdxBLow] <= batch && batch <= range[kIdxBHigh];
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
    run_info.AddTilingData(static_cast<int32_t>(m));
    run_info.AddTilingData(static_cast<int32_t>(k));
    run_info.AddTilingData(static_cast<int32_t>(n));
    if (dynamic_mode == "dynamic_mknb") {
      run_info.AddTilingData(static_cast<int32_t>(batch));
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
