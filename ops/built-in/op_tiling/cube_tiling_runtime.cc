/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
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
 * \file cube_tiling_runtime.cc
 * \brief
 */
#include "cube_tiling_runtime.h"

#include "error_log.h"
#include "error_util.h"
#include "op_log.h"
#include "graph/utils/type_utils.h"

namespace {
using nlohmann::json;
using std::map;
using std::string;
using std::tuple;
using std::vector;

// batchmatmul/matmul mkn, dx/dw nhw
const auto kIsShapeInRange3 = [](const int64_t *shape, const vector<int64_t> &range) -> bool {
  return shape[0] >= range[0] && shape[0] <= range[1] && shape[1] >= range[2] && shape[1] <= range[3] &&
         shape[2] >= range[4] && shape[2] <= range[5];
};

// batchmatmul mkn batch, conv3d ndhw
const auto kIsShapeInRange4 = [](const int64_t *shape, const vector<int64_t> &range) -> bool {
  return shape[0] >= range[0] && shape[0] <= range[1] && shape[1] >= range[2] && shape[1] <= range[3] &&
         shape[2] >= range[4] && shape[2] <= range[5] && shape[3] >= range[6] && shape[3] <= range[7];
};

// batchmatmul/matmul mkn, dx/dw nhw
const auto kCalcDist3 = [](const int64_t *shape, const vector<int64_t> &seeds) -> int64_t {
  return abs(shape[0] - seeds[0]) + abs(shape[1] - seeds[1]) + abs(shape[2] - seeds[2]);
};

// batchmatmul mkn batch, conv3d ndhw
const auto kCalcDist4 = [](const int64_t *shape, const vector<int64_t> &seeds) -> int64_t {
  return abs(shape[0] - seeds[0]) + abs(shape[1] - seeds[1]) + abs(shape[2] - seeds[2]) + abs(shape[3] - seeds[3]);
};

#define CHECK_TILING_IN_REPO(tiling_id, is_shape_in_range, calc_dist) \
  do {                                                                \
    int64_t min_distance = std::numeric_limits<int64_t>::max();       \
    size_t idx = 0;                                                   \
    size_t tiling_id_idx = kInvalidTilingId;                          \
    while (idx < repo_seeds.size() && idx < repo_range.size()) {      \
      if (is_shape_in_range(shape, repo_range[idx])) {                \
        int64_t dist = calc_dist(shape, repo_seeds[idx]);             \
        if (dist < min_distance) {                                    \
          min_distance = dist;                                        \
          tiling_id_idx = idx;                                        \
        }                                                             \
      }                                                               \
      ++idx;                                                          \
    }                                                                 \
    if (tiling_id_idx != kInvalidTilingId) {                          \
      tiling_id = repo_tiling_ids[tiling_id_idx];                     \
    }                                                                 \
  } while (0)

#define CHECK_TILING_IN_COST_MODEL(tiling_id, is_shape_in_range) \
  do {                                                           \
    for (size_t idx = 0; idx < cost_range.size(); ++idx) {       \
      if (is_shape_in_range(shape, cost_range[idx])) {           \
        tiling_id = cost_tiling_ids[idx];                        \
        break;                                                   \
      }                                                          \
    }                                                            \
  } while (0)
}  // namespace

namespace optiling {
bool CubeCompileInfo::AnalyzeCompileInfo(const char *op_name, const char *compile_info_str) {
  OP_TILING_CHECK(compile_info_str == nullptr, CUBE_INNER_ERR_REPORT(op_name, "null compile info"), return false);
  OP_LOGD(op_name, "compile info: %s", compile_info_str);
  try {
    auto compile_info = json::parse(compile_info_str);
    if (compile_info.type() == json::value_t::object) {
      OP_TILING_CHECK(!(AnalyzeExtendInfo(compile_info) && AnalyzeCommonCompileInfo(compile_info)),
                      CUBE_INNER_ERR_REPORT(op_name, "Parse compile value fail"), return false);
    } else {
      for (size_t idx = 0; idx < compile_info.size(); ++idx) {
        OP_TILING_CHECK(!(AnalyzeExtendInfo(compile_info[idx]) && AnalyzeCommonCompileInfo(compile_info[idx])),
                        CUBE_INNER_ERR_REPORT(op_name, "Parse compile value fail"), return false);
      }
    }
  } catch (...) {
    CUBE_INNER_ERR_REPORT(op_name, "get unknown exception, please check compile info json.");
    return false;
  }

  return true;
}

bool CubeCompileInfo::AnalyzeCommonCompileInfo(const nlohmann::json &compile_info) {
  // json structure: {"repo_seeds": {"10114": [..]}, "repo_range": {"10114": [...]}, "cost_range": {"10115": [...]},
  //                  "block_dim": {"10114": 32, "10115": 32}}
  const auto &tmp_block_dim = compile_info["block_dim"].get<map<string, uint32_t>>();
  bool is_digit = true;
  auto is_digit_func = [&is_digit](char element) { is_digit = is_digit && isdigit(element) != 0;};
  for (auto it = tmp_block_dim.begin(); it != tmp_block_dim.end(); ++it) {
    is_digit = true;
    std::for_each(it->first.begin(), it->first.end(), is_digit_func);
    if (is_digit) {  // filter CORE_NUM in binary mode
      block_dim.insert({stoull(it->first), it->second});
    }
  }

  if (compile_info.contains("repo_seeds") && compile_info.contains("repo_range")) {
    const auto &tmp_repo_seeds = compile_info["repo_seeds"].get<map<string, vector<int64_t>>>();
    const auto &tmp_repo_range = compile_info["repo_range"].get<map<string, vector<int64_t>>>();
    auto repo_seeds_iter = tmp_repo_seeds.begin();
    auto repo_range_iter = tmp_repo_range.begin();

    // maybe in loop, can't reserve space here
    while (repo_seeds_iter != tmp_repo_seeds.end() && repo_range_iter != tmp_repo_range.end()) {
      if (repo_seeds_iter->first != repo_range_iter->first) {
        return false;
      }

      repo_seeds.emplace_back(repo_seeds_iter->second);
      repo_range.emplace_back(repo_range_iter->second);
      repo_tiling_ids.emplace_back(stoull(repo_seeds_iter->first));
      ++repo_seeds_iter;
      ++repo_range_iter;
    }
  }

  if (compile_info.contains("cost_range")) {
    const auto &tmp_cost_range = compile_info["cost_range"].get<map<string, vector<int64_t>>>();
    // maybe in loop, can't reserve space here
    for (auto it = tmp_cost_range.begin(); it != tmp_cost_range.end(); ++it) {
      cost_range.emplace_back(it->second);
      cost_tiling_ids.emplace_back(stoull(it->first));
    }
  }

  return true;
}

bool CubeCompileInfo::CheckRangeSize(size_t shape_dim_num) const {
  // for security
  auto check_range = [shape_dim_num](const vector<int64_t> &range) { return range.size() < (shape_dim_num << 1);};
  auto check_seeds = [shape_dim_num](const vector<int64_t> &seeds) { return seeds.size() < shape_dim_num;};

  return std::find_if(repo_range.begin(), repo_range.end(), check_range) == repo_range.end() &&
         std::find_if(repo_seeds.begin(), repo_seeds.end(), check_seeds) == repo_seeds.end() &&
         std::find_if(cost_range.begin(), cost_range.end(), check_range) == cost_range.end();
}

uint64_t CubeCompileInfo::CheckTilingInRepo(const char *op_name, const int64_t *shape, size_t dim_num) const {
  uint64_t tiling_id = kInvalidTilingId;
  switch (dim_num) {
    case 3: {  // shape size 3
      // batchmatmul/matmul mkn, dx/dw nhw
      CHECK_TILING_IN_REPO(tiling_id, kIsShapeInRange3, kCalcDist3);
      break;
    }
    case 4: {  // shape size 4
      // batchmatmul mkn batch, conv3d ndhw
      CHECK_TILING_IN_REPO(tiling_id, kIsShapeInRange4, kCalcDist4);
      break;
    }
    default:
      OP_LOGW(op_name, "not support check dim num %zu in repo", dim_num);
      break;
  }

  OP_LOGD(op_name, "get tiling_id %lu from repo range/seeds", tiling_id);
  return tiling_id;
}

uint64_t CubeCompileInfo::CheckTilingInCostModel(const char *op_name, const int64_t *shape, size_t dim_num) const {
  uint64_t tiling_id = kInvalidTilingId;
  switch (dim_num) {
    case 3: {  // shape size 3
      // batchmatmul/matmul mkn, dx/dw nhw
      CHECK_TILING_IN_COST_MODEL(tiling_id, kIsShapeInRange3);
      break;
    }
    case 4: {  // shape size 4
      // batchmatmul mkn batch, conv3d ndhw
      CHECK_TILING_IN_COST_MODEL(tiling_id, kIsShapeInRange4);
      break;
    }
    default:
      OP_LOGW(op_name, "not support check dim num %zu in cost range", dim_num);
      break;
  }

  OP_LOGD(op_name, "get tiling_id %llu from cost_range", tiling_id);
  return tiling_id;
}

string TensorDesc2String(const gert::StorageShape *shape, const gert::CompileTimeTensorDesc *tensor) {
  if (shape == nullptr || tensor == nullptr) {
    return "nil ";
  }

  std::ostringstream oss;
  oss << "(dtype: " << ge::TypeUtils::DataTypeToSerialString(tensor->GetDataType()) << "),";
  oss << "(shape:" << ge::Shape2String(shape->GetStorageShape()) << "),";
  oss << "(ori_shape:" << ge::Shape2String(shape->GetOriginShape()) << "),";
  oss << "(format: " << ge::TypeUtils::FormatToSerialString(tensor->GetStorageFormat()) << "),";
  oss << "(ori_format: " << ge::TypeUtils::FormatToSerialString(tensor->GetOriginFormat()) << ") ";

  return oss.str();
}

string DebugTilingContext(gert::TilingContext *context) {
  std::ostringstream oss;
  for (size_t i = 0; i < context->GetComputeNodeInfo()->GetInputsNum(); ++i) {
    oss << "input" << i << ": ";
    oss << TensorDesc2String(context->GetInputShape(i), context->GetInputDesc(i));
  }

  for (size_t i = 0; i < context->GetComputeNodeInfo()->GetOutputsNum(); ++i) {
    oss << "output" << i << ": ";
    oss << TensorDesc2String(context->GetOutputShape(i), context->GetOutputDesc(i));
  }
  return oss.str();
}
}  // namespace optiling
