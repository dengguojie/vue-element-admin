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
 * \file eletwise.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_ELETWISE_H_
#define OPS_BUILT_IN_OP_TILING_ELETWISE_H_

#include <vector>
#include <string>

#include <nlohmann/json.hpp>
#include "graph/debug/ge_log.h"
#include "op_tiling.h"

namespace optiling {

// a block size in D
const int32_t BLOCK_SIZE = 32;
const int32_t DOUBLE_BUFFER_SIZE = 2;
const int32_t N_LAST_BROADCAST_THRESHOLD = 512;
const int32_t MAX_PATTERN_DIM = 3;
const int32_t SPECIAL_BROADCAST_INPUT_NUMS = 2;
const int32_t BROADCAST_BASE_KEY = 2;
const size_t MAX_DIM_LEN = 16;
const size_t MAX_INPUT_NUMS = 70;

struct CompileInfo {
  int32_t ub_size{0};
  int32_t max_dtype{0};
  int32_t coexisting_quantity{0};
  int32_t core_num{0};
  int32_t fusion_flag{-1};
  bool is_support_broadcast{false};
  bool is_support_absorbable_broadcast{false};
  bool use_special_pattern{false};
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

class Eletwise {
public:
  explicit Eletwise(const std::string& _op_type, const TeOpParas& _op_paras, const nlohmann::json& _op_info)
      : op_type(_op_type), op_paras(_op_paras), op_info(_op_info) {
  }
  ~Eletwise() {
  }
  bool Init();
  bool GetCompletedShapes();
  bool GenerateOutputShape();
  bool TrySwitchToPerfPattern();
  bool MulTrySwitchToPerfPattern();
  void MulFusionContinuousAxis(std::vector<std::vector<int64_t>>& fusion_shapes, size_t& fusion_length);
  bool BroadcastShapes();
  bool RefineShapesForBroadcast();
  bool CalcTiling();
  bool DoBlockTiling();
  bool DoUbTiling();
  void CalcKey();
  bool CalcConstKey();
  bool WriteTilingData(OpRunInfo& run_info);
  bool DoTiling();
  bool IsNeedDoubleBuffer();

private:
  const std::string& op_type;
  const TeOpParas& op_paras;
  const nlohmann::json& op_info;
  bool is_const{false};
  bool only_const_tiling{false};
  bool need_multi_core{true};
  bool need_double_buffer{false};
  CompileInfo compileInfo;
  int32_t max_available_ub{0};
  std::string in_type;
  std::string out_type;
  int32_t key{0};
  int32_t block_axis{-1};
  int32_t ub_axis{-1};
  int32_t block_dims{1};
  int32_t ub_factor{1};
  int32_t block_factor{1};
  int32_t broadcast_aixs{-1};
  int32_t block_axis_output{-1};
  Pattern s_pattern{Pattern::ORIGINAL};
  std::array<std::array<int64_t, MAX_DIM_LEN>, MAX_INPUT_NUMS> input_shapes{};
  size_t input_num{0};
  size_t dim_len{0};
  std::vector<int64_t> output_shape;
};

}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_ELETWISE_H_
