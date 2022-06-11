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
 * \file broadcast_v3.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_BROADCAST_H_
#define OPS_BUILT_IN_OP_TILING_BROADCAST_H_

#include <vector>
#include <string>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "vector_tiling.h"
#include "elewise_v3.h"
#include "rl_tune.h"

#include "external/graph/operator.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/kernel_run_context.h"

#include "auto_tiling.h"
#include "auto_tiling_context.h"

namespace optiling {
namespace v3 {
constexpr size_t B_MAX_DIM_LEN = 16;
constexpr size_t B_MAX_INPUT_NUMS = 70;

struct BroadcastCompileInfo : AutoTilingCompileInfo{
  ElewiseCompileInfo pure_elewise_compile_info;

  std::pair<bool, std::unordered_map<std::string, std::vector<int64_t>>> base_info_compile;
  std::vector<bool> flag_info_compile;
  int64_t ub_factor_align {-1};
  bool contains_elewise_sch {false};
  std::pair<bool, std::unordered_map<std::string, std::vector<int64_t>>> elewise_vars_compile;
  std::pair<bool, std::vector<int64_t>> const_block_dims_compile;
  std::pair<bool, std::vector<std::vector<int64_t>>> const_shapes_compile;
  std::pair<bool, std::vector<std::vector<size_t>>> fusion_index_compile;
  std::pair<bool, std::vector<bool>> broadcast_axis_compile;
  std::pair<bool, std::string> soc_version;

  // rl bank info
  std::pair<bool, std::vector<std::pair<rl::RlPattern, std::vector<rl::RlBankInfo>>>> bank_info_pair;

  BroadcastCompileInfo() = default;
  ~BroadcastCompileInfo() override = default;
  BroadcastCompileInfo(const std::string& op_type, const nlohmann::json& outer_compile_info);

 private:
  void ParseElewiseInfos(const nlohmann::json& outer_compile_info);
  bool Parse(const nlohmann::json& outer_compile_info);
};

enum class Pattern {
  ORIGINAL = 0,
  COMMON = 100,
  COMMON_BROADCAST = 120,
  COMMON_BROADCAST_COMMON = 121,
  BROADCAST = 200,
  BROADCAST_COMMON = 210,
  BROADCAST_SCALAR = 230,
  SCALAR_BROADCAST = 320,
  UNKNWON_UNKNOWN = 999
};

template <typename T>
class Broadcast {
  public:
    explicit Broadcast(T* _context, const OpInfoImpl* _op_info)
        : context(_context),op_info(_op_info) {}
    ~Broadcast() = default;
    bool BroadcastTiling();
    bool BroadcastTiling(const OpInfo& op_info);

  private:
    bool Init();
    bool InitCompileInfo();
    bool InitOpInOutInfo();
    bool GenerateOutputShape();
    bool TryMatchAllUnknown();
    void TrySwitchToElewise();
    void TrySwitchToPerfPattern();
    void TrySwitchToPerfPatternMilan();
    void FusionContinuousAxis(std::vector<int64_t>& fused_shape_x, std::vector<int64_t>& fused_shape_y);
    void MulTrySwitchToPerfPattern();
    void MulTrySwitchToPerfPatternMilan();
    void MulFusionContinuousAxis(std::vector<std::vector<int64_t>>& fusion_shapes, size_t& fusion_length);
    void GenerateAllUnknown(const std::vector<int64_t>& out_shape, const std::vector<bool>& brc_axis,
                              const int64_t split_axis, const int64_t split_factor);
    bool CalcSplitFactor(std::vector<int64_t>& out_shape, const std::vector<bool>& brc_axis, const int64_t ele_in_block,
                           int64_t& split_axis, int64_t& split_factor);
    bool RefineShapesForBroadcast();
    bool CalcTiling();
    bool DoBlockTiling();
    int64_t SplitUb(const int64_t& max_ub_shape, const int64_t& ele_in_block);
    int64_t FindLowestMiddle();
    bool DoUbTiling();
    bool MilanUbTiling();
    bool DefaultUbTiling();
    void AdjustUbTiling(const int64_t under_ub_shape, const int64_t limit);
    void CheckUpdateUbTiling();
    void OptimizeUbTiling();
    void CalcKey();
    bool IsNeedDoubleBuffer() const;

    bool CompletedShapes();
    bool CompletedShapes(const std::vector<std::vector<int64_t>>& op_input_shapes);
    bool GetOutputType();
    bool CheckInputs();
    bool MatchConstShape(const std::vector<int64_t>& const_shapes,size_t& key_index);
    bool CalcConstKey(const bool is_support_broadcast);
    bool IsEmptyTensor();
    bool WriteConstTiling();
    bool ModifyTiling();

    bool DoTiling();
    bool WriteTilingData() const;

    bool TryMatchRlBank();
    bool WriteRlTilingData(const rl::RlBankInfo& rl_bank_info);
    bool DoRlUbTiling(const rl::RlBankInfo& rl_bank_info,
                      const int64_t rl_ub_split_axis, const int64_t rl_block_split_axis,
                      std::array<int64_t, rl::RL_TOTAL_SHAPE_DIM_LEN>& fused_output_shape,
                      int64_t& under_ub_split_shape);
    bool DoRlTiling(const rl::RlBankInfo& rl_bank_info);

  private:
    T* context;
    const OpInfoImpl* op_info{nullptr};
    const char* op_type;
    const BroadcastCompileInfo* broadcast_compile_info;
    ge::DataType in_type{ge::DataType::DT_FLOAT16};
    ge::DataType out_type{ge::DataType::DT_FLOAT16};
    size_t max_output_shape_size{1};
    size_t input_num{0};
    size_t dim_len{0};
    bool is_multi_output{false};
    std::array<std::array<int64_t, B_MAX_DIM_LEN>, B_MAX_INPUT_NUMS> input_shapes{};
    std::vector<std::vector<size_t>> fusion_index{};
    std::vector<std::vector<int64_t>> fusion_shapes{};
    std::vector<int64_t> output_shape{};
    std::array<bool, B_MAX_DIM_LEN> broadcast_axis{};
    int64_t key{-1};
    int64_t output_size{1};
    int64_t multi_core_output{1};
    int64_t block_axis{-1};
    int64_t ub_axis{-1};
    int64_t block_dims{1};
    int64_t ub_factor{1};
    int64_t block_factor{1};
    int64_t max_available_ub{0};
    int64_t max_available_ub_db{0};
    size_t original_dim_len{0};
    Pattern s_pattern{Pattern::ORIGINAL};
    bool is_const{false};
    bool only_const_tiling{false};
    bool need_tiling_cut{true};
    bool need_single_core{false};
    bool need_double_buffer{false};
    bool need_block_align{false};
    bool is_milan_soc{false};

    int64_t max_dtype_compile{0};
    int64_t core_num_compile{0};
    bool is_support_broadcast_compile{false};
    bool is_support_absorbable_broadcast_compile{false};
    bool use_special_pattern_compile{false};
    bool is_unknown_rank_compile{false};
    bool has_all_unknown_compile{false};

    // rl
    bool hit_rl_bank{false};
    int64_t rl_ub_factor{1};
    int64_t rl_block_factor{1};
    int64_t rl_block_dim{1};
    std::array<int32_t, rl::RL_MAX_VARS_NUM> rl_tiling_data{};
};
template class Broadcast<AutoTilingContext>;
template class Broadcast<AutoTilingOp>;
} // namespace v3
class BroadcastTilingHandler: public AutoTilingHandler {
  public:
  BroadcastTilingHandler(const std::string& o, const std::string& p, const nlohmann::json& c)
    : AutoTilingHandler(o, p), broadcast_compile_info(o, c) {}
  ~BroadcastTilingHandler() = default;
  bool DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const override;
  bool DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info, const OpInfo& op_info) const override;

  private:
  const v3::BroadcastCompileInfo broadcast_compile_info;
};
} // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_BROADCAST_H_
