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
constexpr int32_t INT_NUM_TWO = 2;
constexpr int32_t INT_NUM_THREE = 3;
constexpr int32_t BLOCK_SIZE_INT8 = 8;
constexpr int32_t BLOCK_SIZE_FLOAT16 = 16;
constexpr int32_t BLOCK_SIZE_FLOAT = 32;
constexpr int32_t PTTERN_30 = 300000000;
constexpr int32_t PTTERN_40 = 400000000;
constexpr int32_t PTTERN_50 = 500000000;

struct SoftmaxTilingInfo {
  int32_t block_dim{0};
  int32_t block_tiling_axis{-1};
  int64_t block_tiling_factor{1};
  int32_t ub_tiling_axis{-1};
  int64_t ub_tiling_factor{1};
};

struct SoftmaxCompileInfo {
  bool is_const{false};
  bool is_const_post{false};
  bool atomic{false};
  bool is_keep_dims{false};
  int64_t max_ub_count{0};
  int32_t core_num{1};
  int32_t reduce_block_size{0};
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
  SoftmaxCompileInfo compileInfo;
  SoftmaxTilingInfo tilingInfo;

  std::vector<int64_t> input_shape_ori;
  std::vector<int32_t> reduce_axis_ori{std::vector<int32_t>(10, 0)};
  std::vector<int64_t> input_shape{std::vector<int64_t>(10, 0)};
  std::vector<int32_t> reduce_axis{std::vector<int32_t>(10, 0)};
  std::vector<int64_t> output_shape{std::vector<int64_t>(10, 0)};

  bool is_last_axis_reduce{false};
  std::string output_dtypeUB;
  int32_t pattern{0};
  int32_t block_size{1};
};
}  // namespace optiling

#endif  // SOFTMAX_TILING_H
