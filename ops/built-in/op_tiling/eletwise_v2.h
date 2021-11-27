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
 * \file eletwise_v2.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_ELETWISE_H_
#define OPS_BUILT_IN_OP_TILING_ELETWISE_H_

#include <vector>
#include <string>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "external/graph/operator.h"
#include "broadcast_v2.h"
#include "vector_tiling.h"

namespace optiling {
namespace utils {
class Eletwise {
 public:
  static const int64_t ELEWISE_REPEATE_NUMS = 128;
  static const int64_t ELEWISE_UINT1_REPEATE_NUMS = 256;
  static const int64_t BLOCK_SIZE = 32;
  static const int64_t DOUBLE_BUFFER_SIZE = 2;

 public:
  explicit Eletwise(const std::string& _op_type,
                    std::array<std::array<int64_t, B_MAX_DIM_LEN>, B_MAX_INPUT_NUMS>& _input_shapes,
                    const nlohmann::json& _compile_info, const ge::DataType _in_type,
                    const ge::DataType _out_type, const std::vector<bool>& _flag_info)
      : op_type(_op_type), input_shapes(_input_shapes), compile_info(_compile_info),
      in_type(_in_type), out_type(_out_type), flag_info(_flag_info) {
  }
  ~Eletwise() {
  }
  bool WriteTilingData(utils::OpRunInfo& run_info) const;
  bool DoTiling();

 private:
  bool Init();
  bool GenerateOutputShape();
  bool CalcTiling();
  bool DoBlockTiling();
  bool DoUbTiling();
  void CalcKey();

 private:
  const std::string& op_type;
  std::array<std::array<int64_t, B_MAX_DIM_LEN>, B_MAX_INPUT_NUMS>& input_shapes;
  const nlohmann::json& compile_info;
  const ge::DataType in_type;
  const ge::DataType out_type;
  const std::vector<bool>& flag_info;
  std::vector<int64_t> output_shape{};
  int64_t key{-1};
  int64_t max_available_ub{0};
  int64_t max_available_ub_db{0};
  int64_t block_axis{-1};
  int64_t ub_axis{-1};
  int64_t block_dims{1};
  int64_t ub_factor{1};
  int64_t block_factor{1};
  int64_t max_dtype{0};
  int64_t core_num{0};
  bool only_const_tiling{false};
  bool use_special_pattern{false};
  bool need_multi_core{true};
  bool need_double_buffer{false};
};
}  // namespace utils
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_ELETWISE_H_
