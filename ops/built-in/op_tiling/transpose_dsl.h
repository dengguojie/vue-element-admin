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
#include "auto_tiling.h"
#include "auto_tiling_context.h"

namespace optiling {
namespace transpose {
constexpr size_t MAX_DIM_LEN = 8;

struct TransposeCompileInfo : AutoTilingCompileInfo {
  int64_t core_num{0};
  int64_t ub_size{0};
  int32_t const_block_dims{1};
  bool only_const_tiling{false};
  bool is_const{false};
  std::vector<int64_t> permute{};
  std::vector<int64_t> ori_permute{};
  std::vector<int64_t> mergeable{};
  std::vector<bool> transpose_vars{};

  TransposeCompileInfo() = default;
  explicit TransposeCompileInfo(const nlohmann::json& json_compile_info);
  ~TransposeCompileInfo() override = default;
  bool Parse(const char* op_type, const nlohmann::json& json_compile_info);
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

template <typename T>
class Transpose {
 public:
  explicit Transpose(T* _context, const OpInfoImpl* _op_info)
      : context(_context),
        op_info(_op_info) {
  }
  ~Transpose() = default;
  bool TransposeTiling();

 private:
  void Init();
  void GenerateShape(int64_t cur_dim_value, size_t cur_index, int64_t& real_index, bool& is_first_in);
  bool GenerateOutputShape();
  bool GenerateOutputShapeFromOp();
  void ProcessConst() const;
  bool DoTiling();
  bool CalcOutputSize();
  bool CalcTiling();
  bool DoBlockAndUbTiling();
  bool DoBlockTiling();
  bool InitCrossUbTilingParams(CrossUbTilingParams& tilingParams);
  bool DoUbTilingNoCross();
  bool InitUbTilingParams(UbTilingParams& ubTilingParams, int64_t available_ub);
  bool DoUbTiling(int64_t available_ub);
  void AdjustUbTiling();
  void DoStoreAlignBlockTiling();
  void DoStoreAlignUbTiling(int64_t available_ub);
  void CalcKey();
  void CalcInUbSize(transpose::AdjustTilingParams& adjustTilingParams) const;
  bool AdjustJudge(const transpose::AdjustTilingParams& adjustTilingParams, bool greater_input) const;
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
  void WriteConstTilingData() const;
  bool WriteTilingData() const;

 private:
  T* context;
  const OpInfoImpl* op_info{nullptr};
  const TransposeCompileInfo* c_info{nullptr};
  std::string op_type;
  int64_t core_num{0};
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
  uint64_t tiling_key{0};
  int64_t max_dim_shape{-1};
  ge::DataType dtype{ge::DataType::DT_FLOAT16};
  bool is_last_transpose{false};
  bool is_nlast_align{false};
  bool is_nlast_no_conv{false};
  bool need_multi_core{false};
  bool first_last_transpose{false};
  bool is_pure_copy{false};
  std::array<int64_t, transpose::MAX_DIM_LEN> input_shapes{};
  std::array<int64_t, transpose::MAX_DIM_LEN> output_shapes{};
  std::array<int64_t, transpose::MAX_DIM_LEN> have_splits{};
  std::vector<int64_t> permute;
  std::vector<int64_t> ori_permute;
};

template class Transpose<AutoTilingContext>;
template class Transpose<AutoTilingOp>;
}  // namespace transpose

class TransposeDslTilingHandler : public AutoTilingHandler {
 public:
  TransposeDslTilingHandler(const std::string& o, const std::string& p, const nlohmann::json& c)
      : AutoTilingHandler(o, p), compile_info(c) {
  }
  ~TransposeDslTilingHandler() override = default;
  bool DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const override;
  bool DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info, const OpInfo& op_info) const override;

 private:
  const transpose::TransposeCompileInfo compile_info;
};
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_TRANSPOSE_DSL_H_
