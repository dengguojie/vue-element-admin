/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file transpose_dsl.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_TRANSPOSE_DSL_H_
#define OPS_BUILT_IN_OP_TILING_TRANSPOSE_DSL_H_

#include <vector>
#include <string>

#include <nlohmann/json.hpp>
#include "vector_tiling.h"
#include "external/graph/operator.h"

namespace optiling {
namespace transpose {
constexpr size_t MAX_DIM_LEN = 8;

struct CompileInfo {
  int64_t core_num{0};
  int64_t ub_size{0};
  bool only_const_tiling{false};
  std::vector<int64_t> permute{};
  std::vector<int64_t> ori_permute{};
  std::vector<int64_t> mergeable{};
  std::vector<bool> transpose_vars{};
};

struct UbTilingParams {
  int64_t input_index{0};
  int64_t output_index{0};
  int64_t cur_ub_size{0};
  int64_t input_ub_size{0};
  int64_t output_ub_size{0};
  bool no_update_cross{false};
};

struct CrossUbTilingParams {
  int64_t input_index{0};
  int64_t output_index{0};
  int64_t input_ub_size{0};
  int64_t output_ub_size{0};
  int64_t cross_ub_in{0};
  int64_t cross_ub_out{0};
  bool can_update_ub{false};
};

struct AdjustTilingParams {
  int64_t max_ub{0};
  int64_t all_in_ub{0};
  int64_t input_in_ub{0};
  int64_t output_in_ub{0};
  int64_t input_index{0};
};

class Transpose {
 public:
  explicit Transpose(const std::string& _op_type, const ge::Operator& _op_paras, const nlohmann::json& _compile_info,
                     const OpInfo& _op_info, utils::OpRunInfo& _run_info, const bool _has_op_info)
      : op_type(_op_type),
        op_paras(_op_paras),
        compile_info(_compile_info),
        op_info(_op_info),
        run_info(_run_info),
        has_op_info(_has_op_info) {
  }
  ~Transpose() {
  }
  bool TransposeTiling();

 private:
  bool ParseCompileInfo();
  bool GenerateOutputShape();
  bool GenerateOutputShapeFromOp();
  bool ProcessConst(bool& is_const);
  bool DoTiling();
  bool CalcTiling();
  bool DoBlockAndUbTiling();
  bool DoBlockTiling();
  bool DoUbTilingNoCross();
  bool DoUbTiling(int64_t available_ub);
  void AdjustUbTiling();
  void DoStoreAlignBlockTiling();
  void DoStoreAlignUbTiling(int64_t available_ub);
  void CalcKey();
  void CalcInUbSize(transpose::AdjustTilingParams& adjustTilingParams);
  bool AdjustJudge(const transpose::AdjustTilingParams& adjustTilingParams, bool greater_input);
  void UpdateFactorOne(transpose::AdjustTilingParams& adjustTilingParams);
  void AdjustInputFactor(transpose::AdjustTilingParams& adjustTilingParams);
  void AdjustOutputFactor(transpose::AdjustTilingParams& adjustTilingParams);
  void UbNoOverlap(int64_t output_in_ub);
  void InputUbTiling(transpose::UbTilingParams& tilingParams);
  void OutputUbTiling(transpose::UbTilingParams& tilingParams);
  bool UbSplitSameAxis(transpose::UbTilingParams& tilingParams);
  bool InputCrossUbTiling(transpose::CrossUbTilingParams& tilingParams);
  bool OutputCrossUbTiling(transpose::CrossUbTilingParams& tilingParams);
  void CrossUbUpdateSameAxis(transpose::CrossUbTilingParams& tilingParams);
  bool WriteTilingData() const;

 private:
  const std::string& op_type;
  const ge::Operator& op_paras;
  const nlohmann::json& compile_info;
  const OpInfo& op_info;
  utils::OpRunInfo& run_info;
  transpose::CompileInfo c_info;
  int64_t max_available_ub{0};
  int64_t max_available_cross_ub{0};
  int64_t input_available_ub{1};
  int64_t output_available_ub{1};
  int64_t output_size{1};
  int64_t block_axis{-1};
  int64_t low_ub_axis{-1};
  int64_t high_ub_axis{-1};
  int64_t block_factor{-1};
  int64_t low_ub_factor{-1};
  int64_t high_ub_factor{-1};
  int64_t block_dims{1};
  int64_t tiling_key{-1};
  int64_t max_dim_shape{-1};
  ge::DataType dtype{ge::DataType::DT_FLOAT16};
  bool is_last_transpose{false};
  bool is_nlast_align{false};
  bool is_nlast_no_conv{false};
  bool need_multi_core{false};
  bool first_last_transpose{false};
  bool is_pure_copy{false};
  bool has_op_info{false};
  std::array<int64_t, transpose::MAX_DIM_LEN> input_shapes{};
  std::array<int64_t, transpose::MAX_DIM_LEN> output_shapes{};
  std::array<int64_t, transpose::MAX_DIM_LEN> have_splits{};
};
}  // namespace transpose

/*
 * @brief: tiling function of transpose operator
 * @param [in] op_type: op_type of the transpose operator
 * @param [in] op_paras: inputs/outputs/atts of the transpose operator
 * @param [in] op_info: compile time generated info of the transpose operator
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool TransposeDsl(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& compile_info,
                  utils::OpRunInfo& run_info);

class TransposeDslTilingHandler : public AutoTilingHandler {
 public:
  TransposeDslTilingHandler(const std::string& o, const std::string& p, const nlohmann::json& c)
      : AutoTilingHandler(o, p), compile_info(c) {
  }
  ~TransposeDslTilingHandler() override = default;
  bool DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const override;
  bool DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info, const OpInfo& op_info) const override;

 private:
  const nlohmann::json compile_info{};
};
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_TRANSPOSE_DSL_H_
