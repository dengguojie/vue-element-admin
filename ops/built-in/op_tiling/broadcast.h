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
 * \file broadcast.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_BROADCAST_H_
#define OPS_BUILT_IN_OP_TILING_BROADCAST_H_

#include <vector>
#include <string>

#include <nlohmann/json.hpp>
#include "graph/debug/ge_log.h"
#include "op_tiling.h"

namespace optiling {

static const size_t B_MAX_DIM_LEN = 16;
static const size_t B_MAX_INPUT_NUMS = 70;

struct CompileInfo {
  int64_t ub_size{0};
  int64_t max_dtype{0};
  int64_t coexisting_quantity{0};
  int64_t core_num{0};
  bool is_support_broadcast{false};
  bool is_support_absorbable_broadcast{false};
  bool use_special_pattern{false};
  bool fusion_flag{false};
};

enum Pattern {
  ORIGINAL = 0,
  COMMON = 100,
  COMMON_BROADCAST = 120,
  COMMON_BROADCAST_COMMON = 121,
  BROADCAST = 200,
  BROADCAST_COMMON = 210,
  BROADCAST_SCALAR = 230,
  SCALAR_BROADCAST = 320
};

class Broadcast {
public:
  static const int64_t BLOCK_SIZE = 32;
  static const int64_t DOUBLE_BUFFER_SIZE = 2;
  static const int64_t N_LAST_BROADCAST_THRESHOLD = 512;
  static const int64_t LAST_AND_N_LAST_FACTOR = 7;
  static const int64_t MAX_PATTERN_DIM = 3;
  static const int64_t SPECIAL_BROADCAST_INPUT_NUMS = 2;
  static const int64_t BROADCAST_BASE_KEY = 2;
  static const int64_t ELEWISE_REPEATE_NUMS = 128;
  static const int64_t ELEWISE_UINT1_REPEATE_NUMS = 256;
  static constexpr float LAST_AND_N_LAST_BASE = 1.5;

public:
  explicit Broadcast(const std::string& _op_type, const TeOpParas& _op_paras,
                     const nlohmann::json& _op_info, const std::vector<bool>& _flag_info,
                     size_t _input_num, size_t _dim_len,
                     std::array<std::array<int64_t, B_MAX_DIM_LEN>, B_MAX_INPUT_NUMS>& _input_shapes)
      : op_type(_op_type), op_paras(_op_paras), op_info(_op_info), flag_info(_flag_info),
        input_num(_input_num), dim_len(_dim_len), input_shapes(_input_shapes) {
  }
  ~Broadcast() {
  }
  bool DoTiling();
  bool WriteTilingData(OpRunInfo& run_info) const;

private:
  bool Init();
  bool GenerateOutputShape();
  bool TrySwitchToPerfPattern();
  void FusionContinuousAxis(std::vector<int64_t>& fused_shape_x, std::vector<int64_t>& fused_shape_y);
  bool MulTrySwitchToPerfPattern();
  void MulFusionContinuousAxis(std::vector<std::vector<int64_t>>& fusion_shapes, size_t& fusion_length);
  bool BroadcastShapes();
  bool RefineShapesForBroadcast();
  bool CalcTiling();
  bool DoBlockTiling();
  void CheckUpdateBlockTiling();
  bool DoUbTiling();
  void AdjustUbTiling(const int64_t under_ub_shape, const int64_t limit);
  void CheckUpdateUbTiling();
  void CalcKey();
  bool IsNeedDoubleBuffer() const;

private:
  const std::string& op_type;
  const TeOpParas& op_paras;
  const nlohmann::json& op_info;
  const std::vector<bool>& flag_info;
  size_t input_num{0};
  size_t dim_len{0};
  std::array<std::array<int64_t, B_MAX_DIM_LEN>, B_MAX_INPUT_NUMS>& input_shapes;
  std::vector<std::vector<size_t>> fusion_index{};
  std::vector<int64_t> output_shape{};
  std::array<bool, B_MAX_DIM_LEN> broadcast_axis{};
  size_t max_output_shape_size{1};
  int64_t key{-1};
  int64_t output_size{1};
  int64_t multi_core_output{1};
  int64_t block_axis{-1};
  int64_t ub_axis{-1};
  int64_t block_dims{1};
  int64_t ub_factor{1};
  int64_t block_factor{1};
  int64_t max_available_ub{0};
  Pattern s_pattern{Pattern::ORIGINAL};
  std::string in_type;
  std::string out_type;
  CompileInfo compileInfo;
  bool is_const{false};
  bool only_const_tiling{false};
  bool need_multi_core{true};
  bool need_double_buffer{false};
  bool is_multi_output{false};
};

}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_BROADCAST_H_
