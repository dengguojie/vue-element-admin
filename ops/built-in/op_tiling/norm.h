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
 * \file norm.h
 * \brief
 */

#ifndef NORM_TILING_H
#define NORM_TILING_H

#include <cmath>
#include <vector>

#include "vector_tiling.h"

namespace optiling {

static const std::size_t NORM_MAX_DIM_LEN = 8;
static const std::size_t NORM_MAX_INPUT_NUMS = 70;
static const std::size_t NORM_MAX_WORKSPACE_NUMS = 20;
static const std::size_t NORM_MAX_VAR_NUMS = 40;
static const int64_t NORM_FAKE_WORKSPACE_SIZE = 32;
static const int32_t NORM_NONE_SPLIT_KEY = 9;
static const int32_t NORM_REDUCE_PATTERN_WEIGHT = 1000;
static const int32_t NORM_COMMON_SCH_TYPE = 0;
static const int32_t NORM_PARTIAL_REORDER_SCH_TYPE = 1;
static const int32_t NORM_ALIGNED_IN_UB_SCH_TYPE = 2;

struct NormCompileInfo {
  // construct func
  NormCompileInfo() = default;
  NormCompileInfo(const std::string& op_type, const nlohmann::json &compile_info);
  // check value
  bool Check();

  std::string norm_op_type;
  bool check_success{true};
  // reduce and broadcast axis
  std::vector<int32_t> ori_reduce_axis;
  std::vector<int32_t> ori_broadcast_axis;
  bool is_broadcast_axis_known{false};
  // graph info
  std::vector<int32_t> input_type;
  bool exist_output_after_reduce{false};
  bool exist_workspace_after_reduce{false};
  std::unordered_map<std::string, std::vector<int32_t>> available_ub_size;
  // common info
  int32_t core_num{-1};
  int32_t min_block_size{-1};
  int32_t pad_max_entire_size{-1};
  // workspace info
  std::unordered_map<std::string, std::vector<int32_t>> workspace_info;
  // norm vars
  std::unordered_map<std::string, std::vector<int32_t>> norm_vars;
  // fuse axis
  bool is_fuse_axis{true};
  // const
  bool is_const{false};
  bool is_const_post{false};
  std::unordered_map<std::string, int32_t> const_block_dims;
};

struct NormTilingInfo {
  int32_t block_dim{-1};
  int32_t block_tiling_axis{-1};
  int64_t block_tiling_factor{-1};
  int32_t ub_tiling_axis{-1};
  int64_t ub_tiling_factor{-1};
};

struct NormReorderInfo {
  std::vector<int64_t> reorder_input_shape{std::vector<int64_t>(NORM_MAX_DIM_LEN, 0)};
  std::vector<int32_t> fused_block_tiling_axis;
  // pos after reorder : pos before reorder
  //    vector.idx     :      vector[idx]
  std::vector<int32_t> reorderPos_oriPos{std::vector<int32_t>(NORM_MAX_DIM_LEN, 0)};
};

enum NormBroadcastMode {
  NO_BROADCAST = 0,
  SAME_REDUCE = 1,
  OPPOSITE_REDUCE = 2,
  ALL_BROADCAST = 3,
  OTHERS = 4
};

class Norm {
  public:
    explicit Norm(const std::string& op_type, const ge::Operator& op_paras,
                  const NormCompileInfo& compileInfo, utils::OpRunInfo& run_info)
        : op_type(op_type), op_paras(op_paras), compileInfo(compileInfo), run_info(run_info) {
    }
    ~Norm() = default;
    bool DoTiling();

  private:
    bool GetInput();
    bool Init();
    bool GetNormPattern();
    bool EliminateOne();
    bool FusedAxis();
    bool GetUbSizeInfo();
    bool ProcessTiling();
    bool JudgeCurDimSplitBlock(const int64_t& right_product, const std::size_t& index);
    bool JudgeCurDimSplitBlock(const int64_t& left_product, const int64_t& right_product,
                               const std::size_t& index, const int64_t& max_block_factor = 0);
    bool GetPartialReorderBlockTilingInfo();
    bool GetWorkspaceBlockTilingInfo();
    bool GetBlockTilingInfo();
    bool ProcessReorderAxis();
    bool GetPartialReorderUbTilingInfo();
    bool JudgeNormalCurDimSplitUb(const std::size_t& index);
    bool JudgeWorkspaceCurDimSplitUb(const std::size_t& index);
    bool GetUbTilingInfo();
    bool NeedRefineBlockTiling();
    bool CalcTiling();
    bool ConstInputProcPost();
    bool CalcTilingKey();
    bool CalcWorkspace();
    bool WriteTilingData();

    bool IsNeedPartialReorder();
    bool IsNeedWorkspace();
    bool IsNeedAlignedInUb();
    bool GetVarValue();
    bool CalcInputAlignShape();
    NormBroadcastMode JudgeBroadcastMode(const std::array<int64_t, NORM_MAX_DIM_LEN>& before_shape);
    int32_t CalcMinEliminateOneIndex();
    int32_t GetBlockDim(int32_t tiling_axis, int64_t tiling_factor);
    int64_t CalcReorderShapeProduct(int32_t axis_index, bool exclude_reduce_axis);
    int64_t CalcReorderAlignShapeProduct(int32_t axis_index);

    const std::string& op_type;
    const ge::Operator& op_paras;
    const NormCompileInfo& compileInfo;
    utils::OpRunInfo& run_info;
    NormTilingInfo tilingInfo;
    NormReorderInfo reorderInfo;

    std::array<std::array<int64_t, NORM_MAX_DIM_LEN>, NORM_MAX_INPUT_NUMS> before_broadcast_shapes{};
    std::vector<int64_t> input_shape_ori{std::vector<int64_t>(NORM_MAX_DIM_LEN, 0)};
    std::vector<int32_t> reduce_axis_ori{std::vector<int32_t>(NORM_MAX_DIM_LEN, 0)};
    std::vector<int32_t> broadcast_axis_ori{std::vector<int32_t>(NORM_MAX_DIM_LEN, 0)};
    std::vector<int64_t> input_shape{std::vector<int64_t>(NORM_MAX_DIM_LEN, 0)};
    std::vector<int32_t> reduce_axis{std::vector<int32_t>(NORM_MAX_DIM_LEN, 0)};
    std::vector<int32_t> broadcast_axis{std::vector<int32_t>(NORM_MAX_DIM_LEN, 0)};

    // assistant
    std::vector<int64_t> input_align_shape{std::vector<int64_t>(NORM_MAX_DIM_LEN, 0)};
    std::vector<int64_t> workspace{std::vector<int64_t>(NORM_MAX_WORKSPACE_NUMS, 0)};
    std::vector<int32_t> var_value{std::vector<int32_t>(NORM_MAX_VAR_NUMS, 0)};

    bool is_last_axis_reduce{false};
    bool is_need_workspace{false};
    bool is_partial_reorder{false};
    bool is_continuous_data_move{false};
    bool is_discontinuous_reduce_axis{false};
    bool is_align_and_remove_pad{false};

    int64_t after_reduce_align_shape_product{-1};
    int64_t after_reduce_shape_product{-1};
    int64_t reduce_product{-1};

    int32_t last_r_axis_index{-1};
    int32_t first_a_axis_index{-1};
    int32_t block_tiling_axis_index_in_reorder{-1};
    int32_t last_a_axis_index_in_reorder{-1};

    int32_t reduce_pattern{0};
    int32_t broadcast_pattern{0};
    int32_t norm_pattern{0};
    int32_t sch_type{0};
    int32_t db{0};
    int32_t block_size{-1};
    int64_t ub_size{-1};
    int32_t tiling_key{0};
    std::string tiling_key_str;
    std::size_t max_dim_len{0};
    std::size_t before_broadcast_input_num{0};

    int32_t common_max_ub_count{-1};
    int32_t workspace_max_ub_count{-1};
    int32_t pad_max_ub_count{-1};
};

class NormTilingHandler: public AutoTilingHandler {
  public:
  NormTilingHandler(const std::string& o, const std::string& p, const nlohmann::json& c)
    : AutoTilingHandler(o, p), norm_compile_info(o, c) {}
  ~NormTilingHandler() = default;
  bool DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const override;
  bool DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info, const OpInfo& op_info) const override;
  bool ParsedSuccess() {return norm_compile_info.check_success;};

  private:
  const NormCompileInfo norm_compile_info;
};

std::shared_ptr<AutoTilingHandler> CreateNormTilingHandler(const std::string& op_type,
                                                               const std::string& pattern,
                                                               const nlohmann::json& parsed_compile_info);

}  // namespace optiling

#endif  // NORM_TILING_H
