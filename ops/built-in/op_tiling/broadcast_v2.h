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
 * \file broadcast.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_BROADCAST_H_
#define OPS_BUILT_IN_OP_TILING_BROADCAST_H_

#include <vector>
#include <string>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "external/graph/operator.h"

namespace optiling {
namespace utils {

static const size_t B_MAX_DIM_LEN = 16;
static const size_t B_MAX_INPUT_NUMS = 70;

struct CompileInfo {
  int64_t max_dtype{0};
  int64_t core_num{0};
  bool is_support_broadcast{false};
  bool is_support_absorbable_broadcast{false};
  bool use_special_pattern{false};
  bool is_unknown_rank{false};
  bool has_all_unknown{false};
};

enum Pattern {
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
  static const int64_t BLOCK_SIZE = 32;
  static const size_t MAX_UNKNOWN_RANK = 8;
  static const int64_t DOUBLE_BUFFER_SIZE = 2;
  static const int64_t BLOCK_NUM = 8;
  static const int64_t MAX_REPEAT_TIMES = 8;
  static const int64_t N_LAST_BROADCAST_THRESHOLD = 1024;
  static const int64_t LAST_AND_N_LAST_FACTOR = 7;
  static const int64_t MAX_PATTERN_DIM = 3;
  static const int64_t SPECIAL_BROADCAST_INPUT_NUMS = 2;
  static const int64_t BROADCAST_BASE_KEY = 2;
  static const int64_t ELEWISE_REPEATE_NUMS = 128;
  static const int64_t ELEWISE_UINT1_REPEATE_NUMS = 256;
  static const int64_t NONE_BRC_AXIS_OPTIMIZE_BLOCK_NUMS = 3;
  static constexpr float MIDDLE_AXIS_OPTIMIZE_BLOCK_NUMS = 1.5;

 public:
  explicit Broadcast(const std::string& _op_type, const ge::Operator& _op_paras, const nlohmann::json& _compile_info,
                     const std::vector<bool>& _flag_info,
                     const ge::DataType _in_type,
                     const ge::DataType _out_type,
                     size_t _max_output_shape_size,
                     size_t _input_num,
                     size_t _dim_len,
                     bool is_multi_output,
                     std::array<std::array<int64_t, B_MAX_DIM_LEN>, B_MAX_INPUT_NUMS>& _input_shapes)
      : op_type(_op_type),
        op_paras(_op_paras),
        compile_info(_compile_info),
        flag_info(_flag_info),
        in_type(_in_type),
        out_type(_out_type),
        max_output_shape_size(_max_output_shape_size),
        input_num(_input_num),
        dim_len(_dim_len),
        input_shapes(_input_shapes) {
  }
  ~Broadcast() {
  }
  bool DoTiling();
  bool WriteTilingData(utils::OpRunInfo& run_info) const;

 private:
  bool Init();
  bool GenerateOutputShape();
  bool TryMatchAllUnknown();
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

 private:
  const std::string& op_type;
  const ge::Operator& op_paras;
  const nlohmann::json& compile_info;
  const std::vector<bool>& flag_info;
  const ge::DataType in_type;
  const ge::DataType out_type;
  size_t max_output_shape_size{1};
  size_t input_num{0};
  size_t dim_len{0};
  bool is_multi_output{false};
  std::array<std::array<int64_t, B_MAX_DIM_LEN>, B_MAX_INPUT_NUMS>& input_shapes;
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
  CompileInfo compileInfo;
  bool is_const{false};
  bool only_const_tiling{false};
  bool need_multi_core{true};
  bool need_double_buffer{false};
  bool need_block_align{false};
  bool is_milan_soc{false};
};

}  // namespace utils
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_BROADCAST_H_
