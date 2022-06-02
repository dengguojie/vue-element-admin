/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
 * \file split_dsl.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_SPLIT_DSL_H_
#define OPS_BUILT_IN_OP_TILING_SPLIT_DSL_H_

#include <vector>
#include <string>

#include <nlohmann/json.hpp>
#include "vector_tiling.h"
#include "auto_tiling.h"
#include "auto_tiling_context.h"

namespace optiling {
namespace split {
constexpr size_t MAX_DIM_LEN = 8;
constexpr int64_t BLOCK_SIZE_BYTES = 32;
constexpr int64_t MAX_INPUT_NUM = 63;
constexpr int64_t MAX_TILING_NUM = 256;

struct SplitCompileInfo : AutoTilingCompileInfo {
  int64_t core_num{0};
  int64_t ub_size{0};
  int64_t ori_axis{-1};
  int64_t const_block_dims{1};
  bool is_const{false};
  bool only_const_tiling{false};
  bool is_avg_split{false};
  bool split_is_const{false};
  std::vector<bool> split_vars;

  SplitCompileInfo() = default;
  explicit SplitCompileInfo(const nlohmann::json& json_compile_info);
  ~SplitCompileInfo() override = default;
  bool Parse(const char* op_type, const nlohmann::json& json_compile_info);
};

template <typename T>
class Split {
 public:
  explicit Split(T* _context, const OpInfoImpl* _op_info)
      : context(_context),
        op_info(_op_info) {
  }
  ~Split() = default;
  bool SplitTiling();

 private:
  void ProcessConst() const;
  bool DoTiling();
  bool GenerateBaseInput(const OpShape& shape, int64_t& axis);
  bool GenerateOutputShape();
  bool GenerateBaseInputFromOp(const std::vector<int64_t>& shape, int64_t& ori_axis);
  bool GenerateOutputShapeFromOp();
  bool IsAllAlign();
  bool CalcTiling();
  bool CalcSingleOutputInfo();
  bool CalcAllAlignInfo();
  bool CalcGeneralInfo();
  void DoBaseBlockTiling();
  void DoBaseUbTiling();
  void DoBlockTiling();
  void DoUbTiling();
  void CalcKey();
  bool WriteConstTilingData();
  bool WriteTilingData();
  void WriteCutData(size_t& tiling_num);
  void WriteShapeData(size_t& tiling_num);

 private:
  T* context;
  const OpInfoImpl* op_info{nullptr};
  std::string op_type;
  const SplitCompileInfo* c_info{nullptr};
  ge::DataType dtype{ge::DataType::DT_FLOAT16};
  int64_t min_split_shape{INT64_MAX};
  int64_t output_nums{1};
  int64_t block_axis{0};
  int64_t max_available_ub{0};
  int64_t block_factor{-1};
  int64_t avg_block_factor{-1};
  int64_t block_dims{1};
  int64_t col_limit{1};
  int64_t row_limit{1};
  uint64_t tiling_key{0};
  bool need_multi_core{false};
  bool is_base{false};
  bool is_empty{false};
  bool only_cut_m{false};
  bool is_all_align{false};
  bool is_all_align_copy{false};
  bool is_single_output{false};
  std::array<int64_t, MAX_DIM_LEN> input_shapes{};
  std::array<std::array<int64_t, MAX_DIM_LEN>, MAX_INPUT_NUM> output_shapes{};
  std::array<int64_t, MAX_INPUT_NUM> low_ub_factors{};
  std::array<int64_t, MAX_INPUT_NUM> high_ub_factors{};
  std::array<int32_t, MAX_TILING_NUM> tiling_data{};
};
}  // namespace split

class SplitDslTilingHandler : public AutoTilingHandler {
 public:
  SplitDslTilingHandler(const std::string& o, const std::string& p, const nlohmann::json& c)
      : AutoTilingHandler(o, p), compile_info(c) {
  }
  ~SplitDslTilingHandler() = default;
  bool DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const override;
  bool DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info, const OpInfo& op_info) const override;

 private:
  const split::SplitCompileInfo compile_info;
};
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_SPLIT_DSL_H_
