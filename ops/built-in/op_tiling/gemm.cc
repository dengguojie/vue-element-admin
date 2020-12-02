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
#include <climits>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "graph/types.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "op_tiling.h"

using namespace std;
using json = nlohmann::json;
using utils = ge::TypeUtils;

const int64_t kIdxMLow = 0;
const int64_t kIdxMHigh = 1;
const int64_t kIdxKLow = 2;
const int64_t kIdxKHigh = 3;
const int64_t kIdxNLow = 4;
const int64_t kIdxNHigh = 5;

const int64_t kIdxM = 0;
const int64_t kIdxK = 1;
const int64_t kIdxN = 2;

const int64_t kBlockIn = 16;
const int64_t kBlockReduce = 16;
const int64_t kBlockReduceS8 = 32;
const int64_t kBlockOut = 16;

namespace optiling {

bool CheckCompileInfo(const std::string &op_type, const json &compile_info, const vector<string> &keys) {
  for (auto& key : keys) {
    if (compile_info.find(key) == compile_info.end()) {
      OP_LOGE(op_type.c_str(), "compile info does not contain %s", key.c_str());
      return false;
    }
  }
  return true;
}

bool CheckGEMMCompileInfo(const std::string &op_type, const json &compile_info) {
  const vector<string> keys = {"dynamic_mode", "repo_seeds", "repo_range", "cost_range", "block_dim", "attrs"};
  if (!CheckCompileInfo(op_type, compile_info, keys)) {
    return false;
  }
  return CheckCompileInfo(op_type, compile_info["attrs"], {"transpose_a", "transpose_b"});
}

bool CheckGEMMOpPara(const std::string &op_type, const TeOpParas &op_paras) {
  if (op_type != "MatMul" && op_type != "MatMulV2") {
    OP_LOGE(op_type.c_str(), "cannot support opType %s", op_type.c_str());
    return false;
  }

  // input: a, b, [bias]
  if (op_paras.inputs.empty() || op_paras.inputs.size() < 2) {
    OP_LOGE(op_type.c_str(), "MatMul/MatMulV2 requires at least 2 inputs, actually is %d",
            op_paras.inputs.size());
    return false;
  }
  if (op_paras.inputs[0].tensor.empty() || op_paras.inputs[0].tensor.empty()) {
    OP_LOGE(op_type.c_str(), "Input tensor is empty");
    return false;
  }

  return true;
}

bool CalcGEMMMkn(const string &op_type, const json &compile_info,
                 const TeOpTensor &tensor_a, const TeOpTensor &tensor_b,
                 int64_t *m, int64_t *k, int64_t *n) {
    int32_t block_reduce = kBlockReduce, block_in = kBlockIn, block_out = kBlockOut;
    if (tensor_a.dtype == "int8" || tensor_a.dtype == "uint8") {
      block_reduce = kBlockReduceS8;
    }

    int32_t idx_m_of_a = 0;
    int32_t idx_k_of_a = 1;
    int32_t idx_k_of_b = 0;
    int32_t idx_n_of_b = 1;

    auto trans_a = compile_info["attrs"]["transpose_a"].get<bool>();
    auto trans_b = compile_info["attrs"]["transpose_b"].get<bool>();
    if (trans_a) {
      idx_m_of_a = 1;
      idx_k_of_a = 0;
    }
    if (trans_b) {
      idx_k_of_b = 1;
      idx_n_of_b = 0;
    }

    if (tensor_a.ori_shape[idx_k_of_a] != tensor_b.ori_shape[idx_k_of_b]) {
      OP_LOGE(op_type.c_str(), "axis k must be equal, in facta k of a is %d, k of b is %d",
              tensor_a.ori_shape[idx_k_of_a], tensor_b.ori_shape[idx_k_of_b]);
      return false;
    }

    *m = std::ceil(static_cast<double>(tensor_a.ori_shape[idx_m_of_a]) / block_in);
    *k = std::ceil(static_cast<double>(tensor_a.ori_shape[idx_k_of_a]) / block_reduce);
    *n = std::ceil(static_cast<double>(tensor_b.ori_shape[idx_n_of_b]) / block_out);

    return true;
}

/*
 * @brief: tiling function of gemm
 * @param [in] op_type: op_type of the gemm
 * @param [in] op_paras: inputs/outputs/atts of the gemm
 * @param [in] op_compile_info: compile time generated info of the gemm
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool GEMMTiling(const std::string &op_type, const TeOpParas &op_paras, const json &compile_info,
                OpRunInfo& run_info) {
  if (!CheckGEMMOpPara(op_type, op_paras)) {
    return false;
  }
  if (!CheckGEMMCompileInfo(op_type, compile_info)) {
    return false;
  }

  auto tensor_a = op_paras.inputs[0].tensor[0];
  auto tensor_b = op_paras.inputs[1].tensor[0];
  int64_t m, k, n;
  if (!CalcGEMMMkn(op_type, compile_info, tensor_a, tensor_b, &m, &k, &n)) {
    return false;
  }

  auto dynamic_mode = compile_info["dynamic_mode"].get<std::string>();

  if (dynamic_mode != "dynamic_mkn") {
    OP_LOGE(op_type.c_str(), "dynamic_mode: %s is not supported", dynamic_mode.c_str());
    return false;
  }

  string tiling_id("-1");
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
      if (in_range) {
        tiling_id = element.key();
        break;
      }
    }
  } else {
    OP_LOGD(op_type.c_str(), "MatMul/MatMulV2 match tiling in repository");
  }

  if (tiling_id == "-1") {
    OP_LOGE(op_type.c_str(),
            "This shape is not covered by any tiling, "
            "please modify range and recompile");
    return false;
  }

  run_info.block_dim = static_cast<int32_t>(compile_info["block_dim"][tiling_id]);
  ByteBufferPut(run_info.tiling_data, stoi(tiling_id));
  ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(m));
  ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(k));
  ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(n));

  OP_LOGD(op_type.c_str(), "MatMul/MatMulV2 match repo/cost_model tiling_id %s m %lld k %lld n %lld",
          tiling_id.c_str(), m, k, n);
  return true;
}

// register tiling interface of the gemm
REGISTER_OP_TILING_FUNC_BUFFERED(MatMulV2, GEMMTiling);
}  // namespace optiling
