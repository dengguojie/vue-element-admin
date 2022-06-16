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

// dynamic batch
const auto kIsShapeInRange1 = [](const int64_t *shape, const vector<int64_t> &range) -> bool {
  return shape[0] >= range[0] && shape[0] <= range[1];
};

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

// skip batch for conv
const auto kConvCalcDist3 = [](const int64_t *shape, const vector<int64_t> &seeds) -> int64_t {
  return abs(shape[1] - seeds[1]) + abs(shape[2] - seeds[2]);
};

// batchmatmul mkn batch, conv3d ndhw
const auto kCalcDist4 = [](const int64_t *shape, const vector<int64_t> &seeds) -> int64_t {
  return abs(shape[0] - seeds[0]) + abs(shape[1] - seeds[1]) + abs(shape[2] - seeds[2]) + abs(shape[3] - seeds[3]);
};

// skip batch for conv
const auto kConvCalcDist4 = [](const int64_t *shape, const vector<int64_t> &seeds) -> int64_t {
  return abs(shape[1] - seeds[1]) + abs(shape[2] - seeds[2]) + abs(shape[3] - seeds[3]);
};
}  // namespace

namespace optiling {
template <class IsShapeInRange, class CalcDist>
uint64_t MatchRepoTiling(const int64_t *shape, const vector<vector<int64_t>> &repo_seeds,
                         const vector<vector<int64_t>> &repo_range, const IsShapeInRange &is_shape_in_range,
                         const CalcDist &calc_dist) {
  int64_t min_distance = std::numeric_limits<int64_t>::max();
  size_t idx = 0;
  size_t tiling_id_idx = kInvalidTilingId;
  while (idx < repo_seeds.size() && idx < repo_range.size()) {
    if (is_shape_in_range(shape, repo_range[idx])) {
      int64_t dist = calc_dist(shape, repo_seeds[idx]);
      if (dist < min_distance) {
        min_distance = dist;
        tiling_id_idx = idx;
      }
    }
    ++idx;
  }

  return tiling_id_idx;
}

template <class IsShapeInRange>
uint64_t MatchCostModelTiling(const int64_t *shape, const vector<vector<int64_t>> &cost_range,
                              const vector<uint64_t> &tiling_ids, const IsShapeInRange &is_shape_in_range) {
  for (size_t idx = 0; idx < cost_range.size(); ++idx) {
    if (is_shape_in_range(shape, cost_range[idx])) {
      return tiling_ids[idx];
    }
  }

  return kInvalidTilingId;
}

CubeTilingType ParseTilingType(const json &compile_info) {
  if (!compile_info.contains("tiling_type")) {
    return CUBE_DYNAMIC_SHAPE_TILING;
  }

  const auto &tiling_type = compile_info["tiling_type"];
  if (tiling_type == "default_tiling") {
    return CUBE_DEFAULT_TILING;
  }

  if (tiling_type == "binary") {
    return CUBE_BINARY_TILING;
  }

  return CUBE_DYNAMIC_SHAPE_TILING;
}

void GetVarFlagsFromCompileInfo(const nlohmann::json &compile_info, uint32_t &var_bit_flags) {
  std::vector<std::string> vars = compile_info.at("_vars").begin().value().get<std::vector<std::string>>();
  for (auto it = kVar2Flag.begin(); it != kVar2Flag.end(); ++it) {
    if (std::find(vars.begin(), vars.end(), it->first) != vars.end()) {
      var_bit_flags = var_bit_flags | it->second;
    }
  }
}

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

bool CubeCompileInfo::AnalyzeCommonCompileInfo(const json &compile_info) {
  // json structure: {"repo_seeds": {"10114": [..]}, "repo_range": {"10114": [...]}, "cost_range": {"10115": [...]},
  //                  "block_dim": {"10114": 32, "10115": 32}}
  correct_range_flag = compile_info.contains("correct_range_flag") && compile_info["correct_range_flag"].is_boolean() &&
                       compile_info["correct_range_flag"];
  tiling_type = ParseTilingType(compile_info);

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

  // for dynamic batch
  if (compile_info.contains("tiling_range")) {
    const auto &tmp_tiling_range = compile_info["tiling_range"].get<map<string, vector<int64_t>>>();
    // maybe in loop, can't reserve space here
    for (auto it = tmp_tiling_range.begin(); it != tmp_tiling_range.end(); ++it) {
      batch_range.emplace_back(it->second);
      batch_tiling_ids.emplace_back(stoull(it->first));
    }
  }

  if (compile_info.contains("default_range")) {
    const auto &tmp_default_range = compile_info["default_range"].get<map<string, vector<int64_t>>>();
    for (auto it = tmp_default_range.begin(); it != tmp_default_range.end(); ++it) {
      default_range = it->second;
      default_tiling_id = stoull(it->first);
      break;
    }
  }

  return true;
}

bool CubeCompileInfo::CheckRangeSize(size_t shape_dim_num) const {
  // for security
  auto check_range = [shape_dim_num](const vector<int64_t> &range) { return range.size() < (shape_dim_num << 1); };
  auto check_seeds = [shape_dim_num](const vector<int64_t> &seeds) { return seeds.size() < shape_dim_num; };
  auto check_batch_range = [](const vector<int64_t> &range) { return range.size() < 2; };  // range size 2 for dim num 1

  return std::find_if(repo_range.begin(), repo_range.end(), check_range) == repo_range.end() &&
         std::find_if(repo_seeds.begin(), repo_seeds.end(), check_seeds) == repo_seeds.end() &&
         std::find_if(cost_range.begin(), cost_range.end(), check_range) == cost_range.end() &&
         std::find_if(batch_range.begin(), batch_range.end(), check_batch_range) == batch_range.end();
}

uint64_t CubeCompileInfo::CheckTilingInRepo(const char *op_name, const int64_t *shape, size_t dim_num,
                                            bool conv) const {
  uint64_t tiling_id_idx = kInvalidTilingId;
  switch (dim_num) {
    case 3: {  // shape size 3
      // batchmatmul/matmul mkn, dx/dw nhw
      if (conv) {
        tiling_id_idx = MatchRepoTiling(shape, repo_seeds, repo_range, kIsShapeInRange3, kConvCalcDist3);
      } else {
        tiling_id_idx = MatchRepoTiling(shape, repo_seeds, repo_range, kIsShapeInRange3, kCalcDist3);
      }
      break;
    }
    case 4: {  // shape size 4
      // batchmatmul mkn batch, conv3d ndhw
      if (conv) {
        tiling_id_idx = MatchRepoTiling(shape, repo_seeds, repo_range, kIsShapeInRange4, kConvCalcDist4);
      } else {
        tiling_id_idx = MatchRepoTiling(shape, repo_seeds, repo_range, kIsShapeInRange4, kCalcDist4);
      }
      break;
    }
    default:
      OP_LOGW(op_name, "not support check dim num %zu in repo", dim_num);
      break;
  }

  uint64_t tiling_id = (tiling_id_idx != kInvalidTilingId) ? repo_tiling_ids[tiling_id_idx] : kInvalidTilingId;
  OP_LOGD(op_name, "get tiling_id %lu from repo range/seeds", tiling_id);
  return tiling_id;
}

uint64_t CubeCompileInfo::CheckTilingInCostModel(const char *op_name, const int64_t *shape, size_t dim_num) const {
  uint64_t tiling_id = kInvalidTilingId;
  switch (dim_num) {
    case 3: {  // shape size 3
      // batchmatmul/matmul mkn, dx/dw nhw
      tiling_id = MatchCostModelTiling(shape, cost_range, cost_tiling_ids, kIsShapeInRange3);
      break;
    }
    case 4: {  // shape size 4
      // batchmatmul mkn batch, conv3d ndhw
      tiling_id = MatchCostModelTiling(shape, cost_range, cost_tiling_ids, kIsShapeInRange4);
      break;
    }
    default:
      OP_LOGW(op_name, "not support check dim num %zu in cost range", dim_num);
      break;
  }

  OP_LOGD(op_name, "get tiling_id %lu from cost_range", tiling_id);
  return tiling_id;
}

uint64_t CubeCompileInfo::CheckDefaultTiling(const char *op_name, const int64_t *shape, size_t dim_num) const {
  uint64_t tiling_id = kInvalidTilingId;
  switch (dim_num) {
    case 1:
      // dynamic batch for dx/dw
      if (kIsShapeInRange1(shape, default_range)) {
        tiling_id = default_tiling_id;
      }
      break;
    case 3:  // shape size 3
      // batchmatmul/matmul mkn, dx/dw nhw
      if (kIsShapeInRange3(shape, default_range)) {
        tiling_id = default_tiling_id;
      }
      break;
    case 4:  // shape size 4
      // batchmatmul mkn batch, conv3d ndhw
      if (kIsShapeInRange4(shape, default_range)) {
        tiling_id = default_tiling_id;
      }
      break;
    default:
      OP_LOGW(op_name, "not support check dim num %zu in default tiling", dim_num);
      break;
  }

  OP_LOGD(op_name, "get tiling_id %lu from default tiling", tiling_id);
  return tiling_id;
}

uint64_t CubeCompileInfo::CubeTilingBatch(const char *op_name, const int64_t *shape) const {
  uint64_t tiling_id = kInvalidTilingId;
  for (size_t idx = 0; idx < batch_range.size(); ++idx) {
    if (kIsShapeInRange1(shape, batch_range[idx])) {
      tiling_id = batch_tiling_ids[idx];
    }
  }

  OP_LOGD(op_name, "get tiling_id %lu for dynamic batch", tiling_id);
  return tiling_id;
}

bool Conv2DBackPropCompileInfo::AnalyzeExtendInfo(const json &compile_info) {
  if (compile_info.contains("tiling_type") && compile_info["tiling_type"] == "binary") {
    repo_binary_flag = true;
    OP_TILING_CHECK(!compile_info.contains("block_dim"),
                    CUBE_INNER_ERR_REPORT("Conv2DBackpropInput", "get block_dim failed"), return false);

    // dx/dw should use same field
    if (compile_info["block_dim"].contains("CORE_NUM")) {
      core_num = compile_info["block_dim"]["CORE_NUM"];  // dx
    }
    if (compile_info.contains("max_core_num")) {
      core_num = compile_info["max_core_num"];  // dw
    }

    if (compile_info.contains("binary_mode")) {
      binary_mode = compile_info["binary_mode"];
    }
    if (compile_info.contains("aub_num")) {
      aub_num = compile_info["aub_num"];
    }
    if (compile_info.contains("cub_num")) {
      cub_num = compile_info["cub_num"];
    }
    if (compile_info.contains("ub_size")) {
      ub_size = compile_info["ub_size"];
    }
  } else {
    GetVarFlagsFromCompileInfo(compile_info, var_bit_flags);
  }

  return true;
}

ge::graphStatus ParseConv2DBackpropCompileInfo(gert::KernelContext *context) {
  OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("nil", "context is null"),
                  return ge::GRAPH_FAILED);
  auto compute_node = reinterpret_cast<const gert::ComputeNodeInfo *>(context->GetComputeNodeExtend());
  OP_TILING_CHECK(compute_node == nullptr, CUBE_INNER_ERR_REPORT("nil", "compute_node is null"),
                  return ge::GRAPH_FAILED);
  auto op_name = compute_node->GetNodeName();
  auto compile_info = context->GetOutputPointer<Conv2DBackPropCompileInfo>(0);
  auto json_str = context->GetInputStrPointer(0);
  OP_TILING_CHECK(compile_info == nullptr || json_str == nullptr,
                  CUBE_INNER_ERR_REPORT(op_name, "compile_info or json is null"), return ge::GRAPH_FAILED);
  OP_TILING_CHECK(!compile_info->AnalyzeCompileInfo(op_name, json_str),
                  CUBE_INNER_ERR_REPORT(op_name, "failed to analyze compile info"), return ge::GRAPH_FAILED);

  OP_TILING_CHECK(!compile_info->CheckRangeSize(3),  // 3: nhw
                  CUBE_INNER_ERR_REPORT(op_name, "repo_range/repo_seeds/cost_range invalid"), return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CubeTiling(const int64_t *input_shape, size_t intput_shape_dim_num, const gert::Shape &var_value,
                           const CubeCompileInfo &compile_info, gert::TilingContext *context) {
  const char *op_name = context->GetNodeName();
  uint64_t tiling_id = kInvalidTilingId;

  if (compile_info.tiling_type == CUBE_DEFAULT_TILING) {
    tiling_id = compile_info.CheckDefaultTiling(op_name, input_shape, intput_shape_dim_num);
  } else if (var_value.GetDimNum() != 1) {
    tiling_id = compile_info.CheckTilingInRepo(op_name, input_shape, intput_shape_dim_num, true);
    if (tiling_id == kInvalidTilingId) {
      tiling_id = compile_info.CheckTilingInCostModel(op_name, input_shape, intput_shape_dim_num);
    }
  } else {
    tiling_id = compile_info.CubeTilingBatch(op_name, input_shape);
  }

  if (tiling_id == kInvalidTilingId) {
    if (compile_info.correct_range_flag) {
      CUBE_INNER_ERR_REPORT(op_name,
                            "The original range does not meet requirements,"
                            "new range is generated during op compile, but the shape is not covered by new range");
    }

    CUBE_INNER_ERR_REPORT(op_name, "This shape is not covered by any tiling, please modify range and recompile");
    return ge::GRAPH_FAILED;
  }

  auto it = compile_info.block_dim.find(tiling_id);
  OP_TILING_CHECK(it == compile_info.block_dim.end(),
                  CUBE_INNER_ERR_REPORT(op_name, "failed to get block dim for tiling id %lu", tiling_id),
                  return ge::GRAPH_FAILED);

  context->SetBlockDim(it->second);
  context->SetTilingKey(tiling_id);
  auto tiling_data = context->GetRawTilingData();
  for (size_t idx = 0; idx < var_value.GetDimNum(); ++idx) {
    tiling_data->Append(static_cast<int32_t>(var_value.GetDim(idx)));
  }

  return ge::GRAPH_SUCCESS;
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
