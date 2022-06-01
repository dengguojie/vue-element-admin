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
#include <nlohmann/json.hpp>
#include <vector>

#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_context.h"

namespace optiling {
constexpr uint64_t kInvalidTilingId = std::numeric_limits<uint64_t>::max();
constexpr uint32_t kVarBatchN = 0x01;
constexpr uint32_t kVarDxH = 0x02;
constexpr uint32_t kVarDxW = 0x04;

const std::map<std::string, uint32_t> kVar2Flag = {{"batch_n", kVarBatchN}, {"dx_h", kVarDxH}, {"dx_w", kVarDxW}};

enum CubeTilingType {
  CUBE_DYNAMIC_SHAPE_TILING,
  CUBE_DEFAULT_TILING,
  CUBE_BINARY_TILING,
};

class CubeCompileInfo {
 public:
  CubeCompileInfo() = default;
  virtual ~CubeCompileInfo() = default;

  bool AnalyzeCompileInfo(const char *op_name, const char *compile_info_str);
  bool CheckRangeSize(size_t shape_dim_num) const;
  uint64_t CheckTilingInRepo(const char *op_name, const int64_t *shape, size_t dim_num, bool conv = false) const;
  uint64_t CheckTilingInCostModel(const char *op_name, const int64_t *shape, size_t dim_num) const;
  uint64_t CheckDefaultTiling(const char *op_name, const int64_t *shape, size_t dim_num) const;
  uint64_t CubeTilingBatch(const char *op_name, const int64_t *shape) const;

  virtual bool AnalyzeExtendInfo(const nlohmann::json &compile_info) = 0;
  bool AnalyzeCommonCompileInfo(const nlohmann::json &compile_info);

  bool correct_range_flag = false;
  CubeTilingType tiling_type = CUBE_DYNAMIC_SHAPE_TILING;
  uint64_t default_tiling_id = kInvalidTilingId;
  std::vector<int64_t> default_range;
  std::vector<std::vector<int64_t>> repo_seeds;
  std::vector<std::vector<int64_t>> repo_range;
  std::vector<std::vector<int64_t>> cost_range;
  std::vector<std::vector<int64_t>> batch_range;  // for dynamic batch
  std::vector<uint64_t> repo_tiling_ids;
  std::vector<uint64_t> cost_tiling_ids;
  std::vector<uint64_t> batch_tiling_ids;  // for dynamic batch
  std::map<uint64_t, uint32_t> block_dim;
};

class Conv2DBackPropCompileInfo : public optiling::CubeCompileInfo {
 public:
  Conv2DBackPropCompileInfo() = default;
  ~Conv2DBackPropCompileInfo() override = default;

  bool AnalyzeExtendInfo(const nlohmann::json &compile_info) override;

  bool repo_binary_flag = false;
  uint32_t var_bit_flags = 0;
  int32_t core_num = 0;
  int32_t aub_num = 0;
  int32_t cub_num = 1;
  int32_t binary_mode = 1;
  int32_t ub_size = 0;
};

void GetVarFlagsFromCompileInfo(const nlohmann::json &compile_info, uint32_t &var_bit_flags);
ge::graphStatus ParseConv2DBackpropCompileInfo(gert::KernelContext *context);
ge::graphStatus CubeTiling(const int64_t *input_shape, size_t input_shape_dim_num, const gert::Shape &var_value,
                           const CubeCompileInfo &compile_info, gert::TilingContext *context);

std::string TensorDesc2String(const gert::StorageShape *shape, const gert::CompileTimeTensorDesc *tensor);
std::string DebugTilingContext(gert::TilingContext *context);
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_TILING_RUNTIME_H_