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
 * \file reduce_tiling_v3.h
 * \brief
 */

#ifndef REDUCE_TILING_V3_H
#define REDUCE_TILING_V3_H
#include <cmath>
#include <vector>
#include "graph/utils/op_desc_utils.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/kernel_run_context.h"

#include "vector_tiling.h"
#include "auto_tiling.h"
#include "auto_tiling_context.h"

namespace optiling {
namespace v3 {
constexpr int32_t DEFAULT_VECTOR_CAPACITY_10 = 10;

struct ReduceReorderInfo {
  std::vector<int64_t> reorder_input_shape{std::vector<int64_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};
  std::vector<int64_t> fused_block_tiling_axis;
  // pos after reorder : pos before reorder
  // vector.idx:vector[idx]
  std::vector<int32_t> reorderPos_oriPos{std::vector<int32_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};
};

struct ReduceCompileInfo: AutoTilingCompileInfo {
  public:
    ReduceCompileInfo() = default;
    ReduceCompileInfo(const char* op_type, const nlohmann::json& json_info);
    ~ReduceCompileInfo() override = default;
    bool is_const{false};
    bool is_const_post{false};
    bool atomic{false};
    bool is_keep_dims{false};
    int32_t reduce_axes_type{-1};
    uint32_t idx_before_reduce{0};
    int64_t zero_ub_factor{-1};
    int32_t core_num{-1};
    int32_t min_block_size{-1};
    int32_t coef{-1};
    int32_t pad_max_entire_size{-1};
    bool support_transpose{false};
    int32_t reduce_dtype_byte{-1};
    bool group_reduce{false};
    int32_t workspace_size{0};
    std::vector<int32_t> pattern_info;
    std::vector<int32_t> ub_info_rf;
    std::vector<int32_t> ub_info;
    std::vector<int32_t> ub_info_pad;
    std::vector<int32_t> ub_info_transpose;
    std::pair<bool, std::vector<int64_t>> ori_axis;
    std::pair<bool, uint32_t> axes_idx;
    std::pair<bool, int32_t> compile_pattern;
    std::unordered_map<std::string, uint32_t> block_dim_map;
    std::unordered_map<std::string, bool> atomic_flags_map;
    bool Parse(const char* op_type, const nlohmann::json& json_info);
    bool parsed_success{true};

  private:
    bool GetCompileInfoForProcessControl(const nlohmann::json& json_info);
    bool GetCompileInfoForConst(const nlohmann::json& json_info);
    bool GetCompileInfoForCalculate(const char* op_type, const nlohmann::json& json_info);
    bool GetCompileInfoForRunInfo(const nlohmann::json& json_info);
};

struct ReduceTilingInfo {
  int32_t block_dim{-1};
  int32_t block_tiling_axis{-1};
  int64_t block_tiling_factor{-1};
  int32_t ub_tiling_axis{-1};
  int64_t ub_tiling_factor{-1};
  int32_t sch_type{0};
  uint32_t const_block_dims{0};
  bool const_atomic_flag{false};
  bool atomic{false};
  bool group_reduce{false};
  int64_t max_ub_count{-1};
  uint idx{0};
};

template <typename T>
class Reduce {
 public:
  explicit Reduce(T* _context, const OpInfoImpl* _op_info)
    : context(_context),
      op_info(_op_info) {
  }
  ~Reduce() {
  }

  bool DoTiling();
  bool DoTiling(const OpInfoImpl& op_info);

 private:
  int32_t CalcPattern(std::vector<int64_t>& input, std::vector<int64_t>& axis);
  int64_t GetReorderInputShapeMul(int32_t axis_index, int32_t block_tiling_axis_in_reorder);
  int64_t GetAlignShapeMul(int32_t axis_index) const;
  int64_t GetShapeMul(std::vector<int64_t>& shape, int32_t axis_index);
  bool CalcBlockDim(std::vector<int64_t>& out, int32_t tiling_axis, int64_t tiling_factor, int32_t& block_dim);
  int32_t GetRealBlockTilingAxis(std::vector<int64_t>& shape, int32_t idx);
  int32_t CalcConstPattern(std::vector<int64_t>& reduce_axis);
  bool GetBlockTilingInfoX(int64_t cur_block_factor, int64_t i, int64_t right_total_num);
  bool GetBlockTilingInfoY(int32_t left_block_dim, int64_t i,
                           int64_t max_block_tilling_factor, int64_t right_total_num);
  bool GetBlockTilingInfoInner(int64_t i, int64_t cur_block_dim,
                               int64_t cur_block_factor, int64_t right_total_num);
  bool GetBlockTilingInfoLessThanCoreNum(int32_t left_block_dim, uint32_t i, int64_t right_total_num );
  bool GetReduceAxisTensor();

  bool IsInVector(std::vector<int64_t>& input, int32_t value);
  void EliminateOne();
  bool CheckCompileInfoForCalculate();
  bool GetGeInfo();
  bool SetInit();
  bool MatchPattern();
  bool IsZero();
  bool DoZeroBranch();
  bool DoConstRunTimeBranch();
  bool DoReduceTiling();
  bool DoReduceTiling(const OpInfoImpl& op_info);
  bool WriteWorkspace();
  bool WriteTilingData();
  bool WriteConstTilingData();
  bool WriteDynamicTilingData();
  void FusedReduceAxis();
  void ChooseAtomic();
  bool ChooseGroupAxis();
  void ChooseUBInfo();
  void GetReduceShapeCommonInfo();

  bool TilingProcess();
  bool ProcessAtomicTiling();
  bool ProcessNormalTiling();
  bool ProcessGroupTiling();
  bool FineTuning();

  void ProcessReorderAxis(int32_t fused_type);
  bool GetUbTilingInfo();
  bool GetGroupBlockTilingInfo();
  bool GetGroupUbTilingInfo();
  bool GetAtomicBlockTilingInfo();
  bool GetAtomicBlockDim();

  bool GetBlockTilingInfo();
  void GetNotMulCoreBlockTiling();
  int32_t CalcTilingKey();
  bool IsReducePadCase() const;
  bool IsEnableReducePad() const;
  bool IsReduceTransposeCase();
  bool GetInputShapeFromOpShape(OpShape shape);
  bool GetInputShapeOri();
  bool GetInputShapeOri(const OpInfoImpl& op_info);
  bool GetReduceAxisOri(const OpInfoImpl& op_info);

 private:
  std::string op_type;
  T* context;
  const OpInfoImpl* op_info{nullptr};
  const ReduceCompileInfo* compileInfo;
  ReduceTilingInfo reduceTilingInfo;
  ReduceReorderInfo reorderInfo;

  bool exit_zero_axis{false};
  bool exit_non_reduce_zero_axis{false};
  int64_t fusion_dim_value{1};
  int64_t zero_tiling_key{0};
  bool is_reduce_pad_case{false};
  bool is_reduce_transpose_case{false};

  std::vector<int64_t> input_shape_ori;
  std::vector<int64_t> reduce_axis_ori{std::vector<int64_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};
  std::vector<int64_t> input_shape{std::vector<int64_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};
  std::vector<int64_t> reduce_axis{std::vector<int64_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};
  std::vector<int64_t> output_shape{std::vector<int64_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};

  // assistant
  std::vector<int64_t> normalize_shape{std::vector<int64_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};
  std::vector<int64_t> normalize_axis{std::vector<int64_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};
  std::vector<int32_t> reduce_flag{std::vector<int32_t>(DEFAULT_VECTOR_CAPACITY_10, 0)};

  bool is_last_axis_reduce{false};
  int64_t total_output_count{-1};
  int64_t total_reduce_count{-1};
  int32_t pattern{-1};
  int32_t block_size{-1};

  int32_t ubSizeA{-1};
  int32_t ubSizeB{-1};
};

template class Reduce<AutoTilingContext>;
template class Reduce<AutoTilingOp>;
}  // namespace v3

class ReduceTilingHandler: public AutoTilingHandler {
  public:
    ReduceTilingHandler(const std::string& op_info, const std::string& pattern, const nlohmann::json& json_info)
                         : AutoTilingHandler(op_info, pattern), compileInfo(op_info.c_str(), json_info) {}
  bool DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const override;
  bool DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info, const OpInfo& op_info) const override;
  bool ParsedSuccess() const {
    return compileInfo.parsed_success;
  }

  ~ReduceTilingHandler() override = default;

  private:
  const v3::ReduceCompileInfo compileInfo;
};
}  // namespace optiling
#endif  // REDUCE_TILING_V3_H