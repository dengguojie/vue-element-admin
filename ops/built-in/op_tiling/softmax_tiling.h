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
 * \file softmax_tiling.h
 * \brief
 */

#ifndef SOFTMAX_TILING_H
#define SOFTMAX_TILING_H

#include <cmath>
#include <vector>

#include "vector_tiling.h"

namespace optiling {

struct TilingInfo {
  int32_t block_dim;
  int32_t block_tiling_axis;
  int64_t block_tiling_factor;
  int32_t ub_tiling_axis;
  int64_t ub_tiling_factor;
};

struct TilingNewInfo {
  int32_t core_num;
  int32_t block_tiling_axis;
  int64_t block_tiling_outer;
  int64_t block_tiling_inner;
  int32_t ub_tiling_axis;
  int64_t ub_tiling_outer;
  int64_t ub_tiling_inner;
};

struct ReorderInfo {
  std::vector<int64_t> reorder_input_shape{std::vector<int64_t>(10, 0)};
  std::vector<int32_t> fused_block_tiling_axis;
  // pos after reorder : pos before reorder
  //    vector.idx     :      vector[idx]
  std::vector<int32_t> reorderPos_oriPos{std::vector<int32_t>(10, 0)};
};

struct CompileInfo {
  bool is_const = false;
  bool is_const_post = false;
  bool atomic = false;
  bool is_keep_dims = false;
  int64_t max_ub_count;
  int32_t core_num;
  int32_t reduce_block_size;
};

class Softmax {
 public:
  explicit Softmax(const std::string& _op_type, const TeOpParas& _op_paras, const nlohmann::json& _op_info,
                  OpRunInfo& _run_info)
      : op_type(_op_type), op_paras(_op_paras), op_info(_op_info), run_info(_run_info) {
  }
  ~Softmax() {
  }
  bool Init();
  bool DoTiling();
  bool WriteTilingData();
  bool FusedReduceAxis();
  bool GetCompileInfo();
  bool ProcessTiling();
  bool GetUbTilingInfo();
  bool GetBlockTilingInfo();

 private:
  int32_t GetBlockSize(std::string dtypeUB);
  bool IsInVector(std::vector<int32_t>& input, int32_t value);

 private:
  const std::string& op_type;
  const TeOpParas& op_paras;
  const nlohmann::json& op_info;
  OpRunInfo& run_info;
  CompileInfo compileInfo;
  TilingInfo tilingInfo;
  TilingNewInfo tilingNewInfo;
  ReorderInfo reorderInfo;

  std::vector<int64_t> input_shape_ori;
  std::vector<int32_t> reduce_axis_ori{std::vector<int32_t>(10, 0)};
  std::vector<int64_t> input_shape{std::vector<int64_t>(10, 0)};
  std::vector<int32_t> reduce_axis{std::vector<int32_t>(10, 0)};
  std::vector<int64_t> output_shape{std::vector<int64_t>(10, 0)};

  bool is_last_axis_reduce;
  std::string output_dtypeUB;
  int64_t total_output_count;
  int64_t total_reduce_count;
  int32_t pattern;
  int32_t set_reduce_mean_cof_flag;
  int32_t block_size;
  float reduce_mean_cof;
};
}  // namespace optiling

#endif  // SOFTMAX_TILING_H
