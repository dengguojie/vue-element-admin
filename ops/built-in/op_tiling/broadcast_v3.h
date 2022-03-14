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

#include "external/graph/operator.h"

namespace optiling {
namespace v3 {
constexpr size_t B_MAX_DIM_LEN = 16;
constexpr size_t B_MAX_INPUT_NUMS = 70;

struct BroadcastCompileInfo {
  ElewiseCompileInfo pure_elewise_compile_info;

  std::pair<bool, std::unordered_map<std::string, std::vector<int64_t>>> base_info_compile;
  std::vector<bool> flag_info_compile;
  bool outs_uint1_compile {false};
  bool contains_elewise_sch {false};
  std::pair<bool, std::unordered_map<std::string, std::vector<int64_t>>> elewise_vars_compile;
  std::unordered_map<std::uint64_t, vector<VarAttr>> var_attr_map;
  std::pair<bool, std::vector<int64_t>> const_block_dims_compile;

  std::pair<bool, std::vector<std::vector<int64_t>>> const_shapes_compile;
  std::pair<bool, std::vector<std::vector<size_t>>> fusion_index_compile;
  std::pair<bool, std::vector<bool>> broadcast_axis_compile;
  std::pair<bool, std::string> soc_version;

  BroadcastCompileInfo() = default;
  BroadcastCompileInfo(const std::string& op_type, const nlohmann::json& outer_compile_info);
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

class Broadcast {
  public:
    explicit Broadcast(const std::string& op_type,
                   const ge::Operator& op_paras,
                   const BroadcastCompileInfo& broadcast_compile_info,
                   utils::OpRunInfo& run_info)
    : op_type(op_type), op_paras(op_paras), broadcast_compile_info(broadcast_compile_info), run_info(run_info) {}
    ~Broadcast() = default;
    bool BroadcastTiling();
    bool BroadcastTiling(const OpInfo& op_info);

  private:
    bool Init();
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
    void CheckUpdateBlockTiling();
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
    bool GetType();
    bool CheckInputs(bool& is_pure_elementwise);
    bool MatchConstShape(const std::vector<int64_t>& const_shapes,size_t& key_index);
    bool CalcConstKey(const bool is_support_broadcast);
    bool IsEmptyTensor();
    bool WriteConstTiling();
    bool ModifyTiling();

    bool DoTiling();
    bool WriteTilingData() const;

  private:
    const std::string& op_type;
    const ge::Operator& op_paras;
    const BroadcastCompileInfo& broadcast_compile_info;
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
    utils::OpRunInfo& run_info;

    int64_t max_dtype_compile{0};
    int64_t core_num_compile{0};
    bool is_support_broadcast_compile{false};
    bool is_support_absorbable_broadcast_compile{false};
    bool use_special_pattern_compile{false};
    bool is_unknown_rank_compile{false};
    bool has_all_unknown_compile{false};
};
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
