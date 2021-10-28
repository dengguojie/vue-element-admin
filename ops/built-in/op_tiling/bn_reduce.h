/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file bn_reduce.h
 * \brief
 */

#ifndef BN_REDUCE_TILING_H
#define BN_REDUCE_TILING_H


#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "op_tiling.h"

namespace optiling {

constexpr int32_t SMALL_SHAPE_THRESHOLD = 1024;
constexpr int32_t FUSED_NON_REDUCE_AXIS = 0;
constexpr int32_t FUSED_REDUCE_AXIS = 1;
constexpr int32_t H_W_THRESHOLD = 100;
constexpr int32_t DEFAULT_VECTOR_CAPACITY_10 = 10;

struct TilingInfo {
  int32_t block_dim{-1};
  int32_t block_tiling_axis{-1};
  int64_t block_tiling_factor{-1};
  int32_t ub_tiling_axis{-1};
  int64_t ub_tiling_factor{-1};
};

struct ReorderInfo {
  std::vector<int64_t> reorder_input_shape{std::vector<int64_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};
  std::vector<int32_t> fused_block_tiling_axis;
  // pos after reorder : pos before reorder
  // vector.idx : vector[idx]
  std::vector<int32_t> reorderPos_oriPos{std::vector<int32_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};
};

struct CompileInfo {
  bool is_const = false;
  bool is_const_post = false;
  bool atomic = false;
  bool is_keep_dims = false;
  int64_t max_ub_count;
  int32_t core_num;
  int32_t reduce_block_size;
  bool customised = false;
};

class BNReduce {
public:
  explicit BNReduce(const std::string& _op_type, const TeOpParas& _op_paras,
                    const nlohmann::json& _op_info, OpRunInfo& _run_info)
                    : op_type(_op_type), op_paras(_op_paras), op_info(_op_info), run_info(_run_info),
                    is_customised(false), is_fuse_hn(false), saved_customised_ub_axis(0),
                    saved_customised_ub_factor(0) {
  }

  ~BNReduce() {
  }
  bool Init();
  bool DoTiling();
  bool WriteTilingData();
  bool ConstInputProcPost();
  bool FusedReduceAxis();
  bool GetCompileInfo();
  bool ChooseAtomic();

  bool ProcessAtomicTiling();
  void ProcessReorderAxis(const int32_t fused_type);
  bool GetUbTilingInfo();
  bool GetAtomicBlockTilingInfo();
  bool GetAtomicBlockDim();

  int32_t CalcTilingKey();
private:
  int32_t CalcPattern(const std::vector<int64_t>& input, const std::vector<int32_t>& axis);
  int64_t GetReorderInputShapeMul(const int32_t axis_index, const int32_t block_tiling_axis_in_reorder);
  int32_t CalcConstPattern(const std::vector<int32_t>& reduce_axis);
  bool IsInVector(const std::vector<int32_t>& input, const int32_t value);
  void EliminateOne();

  bool DoCustomisedTiling();
  bool DoDefaultTiling();
  bool DoGeneralTiling();

  int64_t CustomisedGetNearestFactor(const int64_t dim, const int64_t split_size);
  bool CustomisedGetUBTiling();
  bool CustomisedGetBlockTiling();
  int64_t GetClosedFactor(const int64_t inner_loop);

private:
  const std::string& op_type;
  const TeOpParas& op_paras;
  const nlohmann::json& op_info;
  OpRunInfo& run_info;
  CompileInfo compileInfo;
  TilingInfo tilingInfo;
  ReorderInfo reorderInfo;

  std::vector<int64_t> input_shape_ori{std::vector<int64_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};
  std::vector<int32_t> reduce_axis_ori{std::vector<int32_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};
  std::vector<int64_t> input_shape{std::vector<int64_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};
  std::vector<int32_t> reduce_axis{std::vector<int32_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};
  std::vector<int64_t> output_shape{std::vector<int64_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};

  // assistant
  std::vector<int64_t> normalize_shape{std::vector<int64_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};
  std::vector<int32_t> normalize_axis{std::vector<int32_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};
  std::vector<int32_t> reduce_flag{std::vector<int32_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};

  int32_t pattern{-1};
  int32_t block_size{-1};
  bool is_customised;
  bool is_fuse_hn;
  int32_t saved_customised_ub_axis;
  int64_t saved_customised_ub_factor;
};
}  // namespace optiling

#endif  // BN_REDUCE_TILING_H
