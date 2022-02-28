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
#include "external/graph/operator.h"

namespace optiling {
namespace split {
constexpr size_t MAX_DIM_LEN = 8;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t MAX_INPUT_NUM = 63;
constexpr int64_t MAX_TILING_NUM = 256;

struct CompileInfo {
  int64_t core_num{0};
  int64_t ub_size{0};
  int64_t ori_axis{-1};
  int64_t const_block_dims{1};
  bool is_const{false};
  bool only_const_tiling{false};
  bool is_avg_split{false};
  bool split_is_const{false};
  std::vector<bool> split_vars;

  CompileInfo() = default;
  explicit CompileInfo(const nlohmann::json& compile_info);
};

class Split {
 public:
  explicit Split(const std::string& _op_type, const ge::Operator& _op_paras, const CompileInfo& _compile_info,
                 const OpInfo& _op_info, utils::OpRunInfo& _run_info)
      : op_type(_op_type),
        op_paras(_op_paras),
        c_info(_compile_info),
        op_info(_op_info),
        run_info(_run_info) {
  }
  ~Split() = default;
  bool SplitTiling();

 private:
  void ProcessConst() const;
  bool DoTiling();
  bool GenerateOutputShape();
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
  void WriteConstTilingData();
  bool WriteTilingData();
  void WriteCutData(int64_t& tiling_num);
  void WriteShapeData(int64_t& tiling_num);

 private:
  const std::string& op_type;
  const ge::Operator& op_paras;
  const split::CompileInfo& c_info;
  const OpInfo& op_info;
  utils::OpRunInfo& run_info;
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

/*
 * @brief: tiling function of split operator
 * @param [in] op_type: op_type of the split operator
 * @param [in] op_paras: inputs/outputs/atts of the split operator
 * @param [in] compile_info: compile time generated info of the split operator
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool SplitDsl(const std::string& op_type, const ge::Operator& op_paras, const split::CompileInfo& compile_info,
              utils::OpRunInfo& run_info);

/*
 * @brief: tiling function of split operator
 * @param [in] op_type: op_type of the split operator
 * @param [in] op_paras: inputs/outputs/atts of the split operator
 * @param [in] compile_info: compile time generated info of the split operator
 * @param [in] op_info: operator info, example shapes and axis
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool SplitDsl(const std::string& op_type, const ge::Operator& op_paras, const split::CompileInfo& compile_info,
              utils::OpRunInfo& run_info, const OpInfo& op_info);

class SplitDslTilingHandler : public AutoTilingHandler {
 public:
  SplitDslTilingHandler(const std::string& o, const std::string& p, const nlohmann::json& c)
      : AutoTilingHandler(o, p), compile_info(c) {
  }
  ~SplitDslTilingHandler() = default;
  bool DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const override;
  bool DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info, const OpInfo& op_info) const override;

 private:
  const split::CompileInfo compile_info;
};
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_SPLIT_DSL_H_
