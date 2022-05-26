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
 * \file cube_tiling_runtime.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_TILING_RUNTIME_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_TILING_RUNTIME_H_

#include <map>
#include <vector>
#include <nlohmann/json.hpp>

#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_context.h"

namespace optiling {
constexpr uint64_t kInvalidTilingId = std::numeric_limits<uint64_t>::max();

class CubeCompileInfo {
 public:
  CubeCompileInfo() = default;
  virtual ~CubeCompileInfo() = default;

  bool AnalyzeCompileInfo(const char *op_name, const char *compile_info_str);
  bool CheckRangeSize(size_t shape_dim_num) const;
  uint64_t CheckTilingInRepo(const char *op_name, const int64_t *shape, size_t dim_num) const;
  uint64_t CheckTilingInCostModel(const char *op_name, const int64_t *shape, size_t dim_num) const;

  virtual bool AnalyzeExtendInfo(const nlohmann::json &compile_info) = 0;
  bool AnalyzeCommonCompileInfo(const nlohmann::json &compile_info);

  std::vector<std::vector<int64_t>> repo_seeds;
  std::vector<std::vector<int64_t>> repo_range;
  std::vector<std::vector<int64_t>> cost_range;
  std::vector<uint64_t> repo_tiling_ids;
  std::vector<uint64_t> cost_tiling_ids;
  std::map<uint64_t, uint32_t> block_dim;
};

std::string TensorDesc2String(const gert::StorageShape *shape, const gert::CompileTimeTensorDesc *tensor);
std::string DebugTilingContext(gert::TilingContext *context);
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_TILING_RUNTIME_H_