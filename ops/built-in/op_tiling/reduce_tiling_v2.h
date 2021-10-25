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
 * \file reduce_tiling_v2.h
 * \brief
 */

#ifndef REDUCE_TILING_V2_H
#define REDUCE_TILING_V2_H

#include <cmath>
#include <vector>

#include "vector_tiling.h"

namespace optiling {
namespace utils {
const int32_t SMALL_SHAPE_THRESHOLD = 1024;
const int32_t FUSED_NON_REDUCE_AXIS = 0;
const int32_t FUSED_REDUCE_AXIS = 1;
const int32_t BASE_2 = 2;
const int32_t BASE_4 = 4;
const int32_t BASE_10 = 10;
const int32_t EIGHTY_PERCENT = 0.8;
const int32_t MAX_INTEGER = 2147483647;
const int32_t EMPTY_SCHEDULE_UB_TILING_FACTOR_128 = 128;
const int32_t ARRAY_INDEX_0 = 0;
const int32_t ARRAY_INDEX_1 = 1;
const int32_t ARRAY_INDEX_2 = 2;
const int32_t ARRAY_INDEX_3 = 3;
const int32_t ARRAY_INDEX_4 = 4;
const int32_t DEFAULT_VECTOR_CAPACITY_10 = 10;
const int32_t ARRAY_FIRST_POS = 0;
const int32_t DEFAULT_CAPACITY_EMPTY = 0;
const int32_t NO_DIM = 0;
const int32_t TILINGKEY_NONE_REDUCE_AXIS = 2147483646;

struct TilingInfoReduce {
  int32_t block_dim{-1};
  int32_t block_tiling_axis{-1};
  int64_t block_tiling_factor{-1};
  int32_t ub_tiling_axis{-1};
  int64_t ub_tiling_factor{-1};
};

struct ReorderInfoReduce {
  std::vector<int64_t> reorder_input_shape{std::vector<int64_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};
  std::vector<int32_t> fused_block_tiling_axis;
  // pos after reorder : pos before reorder
  // vector.idx:vector[idx]
  std::vector<int32_t> reorderPos_oriPos{std::vector<int32_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};
};

struct CompileInfoReduce {
  bool is_const{false};
  bool is_const_post{false};
  bool atomic{false};
  bool is_keep_dims{false};
  bool const_atomic_flag{false};
  uint32_t idx_before_reduce{0};
  int64_t zero_ub_factor{-1};
  int64_t max_ub_count{-1};
  int32_t core_num{-1};
  int32_t min_block_size{-1};
  int32_t coef{-1};
  uint32_t const_block_dims{0};
  uint idx{0};
  std::vector<int32_t> pattern_info;
  std::vector<int32_t> ub_info_rf;
  std::vector<int32_t> ub_info;
  std::pair<bool, std::vector<int32_t>> ori_axis;
  std::pair<bool, uint32_t> axes_idx;
  std::pair<bool, int32_t> compile_pattern;
};

struct GeInfoReduce {
  ge::DataType axes_type;
  ge::Tensor axis_tensor;
};

class Reduce {
 public:
  explicit Reduce(const std::string& _op_type, const ge::Operator& _op_paras, const nlohmann::json& _json_info,
                  utils::OpRunInfo& _run_info)
      : op_type(_op_type), op_paras(_op_paras), json_info(_json_info), run_info(_run_info) {
  }
  ~Reduce() {
  }

  bool GetCompileInfoForProcessControl();
  bool GetCompileInfoForCalculate();
  bool GetGeInfo();
  bool SetInit();
  bool MatchPattern();
  bool IsZero();
  bool DoZeroBranch();
  bool DoConstRunTimeBranch();
  bool DoTiling();
  bool NewDoTiling(const OpInfo& op_info);
  bool WriteTilingData();
  void FusedReduceAxis();
  void ChooseAtomic();
  void ChooseUBInfo();

  bool TilingProcess();
  bool ProcessAtomicTiling();
  bool ProcessNormalTiling();
  bool FineTuning();

  void ProcessReorderAxis(int32_t fused_type);
  bool GetUbTilingInfo();
  bool GetAtomicBlockTilingInfo();
  bool GetAtomicBlockDim();

  bool GetBlockTilingInfo();
  void GetNotMulCoreBlockTiling();
  bool SetAttrVars(int32_t key);
  int32_t CalcTilingKey();
  std::vector<int64_t> GetInputShape();
  std::vector<int32_t> GetReduceAxis();

 private:
  int32_t CalcPattern(std::vector<int64_t>& input, std::vector<int32_t>& axis);
  int64_t GetReorderInputShapeMul(int32_t axis_index, int32_t block_tiling_axis_in_reorder);
  int64_t GetAlignShapeMul(int32_t axis_index);
  int64_t GetShapeMul(std::vector<int64_t>& shape, int32_t axis_index);
  bool CalcBlockDim(std::vector<int64_t>& out, int32_t tiling_axis, int64_t tiling_factor, int32_t& block_dim);
  int32_t GetRealBlockTilingAxis(std::vector<int64_t>& shape, int32_t idx);
  int32_t CalcConstPattern(std::vector<int32_t>& reduce_axis);
  bool IsInVector(std::vector<int32_t>& input, int32_t value);
  void EliminateOne();
  void GetConstValue(const ge::Tensor& data, const ge::DataType& dtype, std::vector<int32_t>& const_vec);

 private:
  const std::string& op_type;
  const ge::Operator& op_paras;
  const nlohmann::json& json_info;
  utils::OpRunInfo& run_info;
  CompileInfoReduce compileInfo;
  GeInfoReduce geInfo;
  TilingInfoReduce tilingInfo;
  ReorderInfoReduce reorderInfo;

  bool exit_zero_axis{false};
  bool exit_non_reduce_zero_axis{false};
  int64_t fusion_dim_value{1};
  int64_t zero_tiling_key{0};

  std::vector<int64_t> input_shape_ori;
  std::vector<int32_t> reduce_axis_ori{std::vector<int32_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};
  std::vector<int64_t> input_shape{std::vector<int64_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};
  std::vector<int32_t> reduce_axis{std::vector<int32_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};
  std::vector<int64_t> output_shape{std::vector<int64_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};

  // assistant
  std::vector<int64_t> normalize_shape{std::vector<int64_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};
  std::vector<int32_t> normalize_axis{std::vector<int32_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};
  std::vector<int32_t> reduce_flag{std::vector<int32_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};

  bool is_last_axis_reduce{false};
  int64_t total_output_count{-1};
  int64_t total_reduce_count{-1};
  int32_t pattern{-1};
  int32_t block_size{-1};

  int32_t ubSizeA{-1};
  int32_t ubSizeB{-1};
};
}  // namespace utils
}  // namespace optiling

#endif  // REDUCE_TILING_V2_H
