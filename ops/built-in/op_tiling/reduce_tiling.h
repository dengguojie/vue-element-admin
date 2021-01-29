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
 * \file reduce_tiling.h
 * \brief
 */

#ifndef REDUCE_TILING_H
#define REDUCE_TILING_H

#include <cmath>
#include <vector>

#include "vector_tiling.h"
#include "../fusion_pass/common/fp16_t.hpp"
#include "op_log.h"
#include "error_log.h"

namespace optiling {

const int32_t REDUCE_MEAN_COF_FP32 = 1;
const int32_t REDUCE_MEAN_COF_FP16 = 2;
const int32_t SMALL_SHAPE_THRESHOLD = 1024;
const int32_t FUSED_NON_REDUCE_AXIS = 0;
const int32_t FUSED_REDUCE_AXIS = 1;

struct TilingInfo {
  int32_t block_dim;
  int32_t block_tiling_axis;
  int64_t block_tiling_factor;
  int32_t ub_tiling_axis;
  int64_t ub_tiling_factor;
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
  int32_t min_block_size;
  int32_t coef;
};

class Reduce {
 public:
  explicit Reduce(const std::string& _op_type, const TeOpParas& _op_paras, const nlohmann::json& _op_info,
                  OpRunInfo& _run_info)
      : op_type(_op_type), op_paras(_op_paras), op_info(_op_info), run_info(_run_info) {
  }
  ~Reduce() {
  }
  bool Init();
  bool DoTiling();
  bool WriteTilingData();
  bool ConstInputProcPost();
  bool FusedReduceAxis();
  bool GetCompileInfo();
  bool ChooseAtomic();

  bool ProcessAtomicTiling();
  bool ProcessNormalTiling();
  bool SpecialUBTiling();

  void ProcessReorderAxis(int32_t fused_type);
  bool GetUbTilingInfo();
  bool GetAtomicBlockTilingInfo();
  bool GetAtomicBlockDim();

  bool GetBlockTilingInfo();
  void GetNotMulCoreBlockTiling();
  int32_t CalcTilingKey();

 private:
  int32_t CalcPattern(std::vector<int64_t>& input, std::vector<int32_t>& axis);
  int32_t GetBlockSize(std::string dtypeUB);
  int64_t GetReorderInputShapeMul(int32_t axis_index, int32_t block_tiling_axis_in_reorder);
  int64_t GetAlignShapeMul(int32_t axis_index);
  int64_t GetShapeMul(std::vector<int64_t>& shape, int32_t axis_index);
  int32_t GetBlockDim(std::vector<int64_t>& out, int32_t tiling_axis, int64_t tiling_factor);
  int32_t GetRealBlockTilingAxis(std::vector<int64_t>& shape, int32_t idx);
  int32_t CalcConstPattern(std::vector<int32_t>& reduce_axis);
  bool IsInVector(std::vector<int32_t>& input, int32_t value);
  void EliminateOne();

 private:
  const std::string& op_type;
  const TeOpParas& op_paras;
  const nlohmann::json& op_info;
  OpRunInfo& run_info;
  CompileInfo compileInfo;
  TilingInfo tilingInfo;
  ReorderInfo reorderInfo;

  std::vector<int64_t> input_shape_ori;
  std::vector<int32_t> reduce_axis_ori{std::vector<int32_t>(10, 0)};
  std::vector<int64_t> input_shape{std::vector<int64_t>(10, 0)};
  std::vector<int32_t> reduce_axis{std::vector<int32_t>(10, 0)};
  std::vector<int64_t> output_shape{std::vector<int64_t>(10, 0)};

  // assistant
  std::vector<int64_t> normalize_shape{std::vector<int64_t>(10, 0)};
  std::vector<int32_t> normalize_axis{std::vector<int32_t>(10, 0)};
  std::vector<int32_t> reduce_flag{std::vector<int32_t>(10, 0)};

  bool is_last_axis_reduce;
  int64_t total_output_count;
  int64_t total_reduce_count;
  int32_t pattern;
  int32_t set_reduce_mean_cof_flag;
  int32_t block_size;
  float reduce_mean_cof;

  int32_t ubSizeA;
  int32_t ubSizeB;
};
}  // namespace optiling

#endif  // REDUCE_TILING_H
