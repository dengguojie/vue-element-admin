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
 * \file concat_dsl.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CONCAT_DSL_H_
#define OPS_BUILT_IN_OP_TILING_CONCAT_DSL_H_

#include <vector>
#include <string>

#include <nlohmann/json.hpp>
#include "vector_tiling.h"
#include "external/graph/operator.h"

namespace optiling {
namespace concat {
constexpr size_t MAX_DIM_LEN = 8;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t MAX_INPUT_NUM = 48;

struct CompileInfo {
  int64_t core_num{0};
  int64_t ub_size{0};
  int64_t ori_axis{-1};
  bool only_const_tiling{false};
  std::vector<std::vector<bool>> concat_vars;
  std::vector<size_t> align_vars;
};

class Concat {
 public:
  explicit Concat(const std::string& _op_type, const ge::Operator& _op_paras, const nlohmann::json& _compile_info,
                  const OpInfo& _op_info, utils::OpRunInfo& _run_info, const bool _has_op_info)
      : op_type(_op_type),
        op_paras(_op_paras),
        compile_info(_compile_info),
        op_info(_op_info),
        run_info(_run_info),
        has_op_info(_has_op_info) {
  }
  ~Concat() = default;
  bool ConcatTiling();

 private:
  bool ProcessConst(bool& is_const);
  bool DoTiling();
  bool ParseCompileInfo();
  bool GenerateOutputShape();
  bool GenerateOutputShapeFromOp();
  bool CalcTiling();
  void DoBlockTiling();
  void DoUbTiling();
  void DoAllAlignUbTiling();
  void DoNoAlignUbTiling(const int64_t factor_n);
  void DoGeneralUbTiling(const int64_t factor_n);
  void DoOneConcatUbTiling();
  void CalcInputPattern(int64_t col_limit, int64_t& ge_factor_n, int64_t& lt_factor_n);
  void DoUbSplitZeroAxis(const int64_t factor_m);
  void CheckAndUpdateTiling();
  void UpdateTiling();
  bool CheckZeroBlockTiling();
  bool CheckOneBlockTiling();
  void CalcFactor();
  void CalcOffsets();
  void CalcKey();
  void WriteConstTilingData() const;
  bool WriteTilingData() const;

 private:
  const std::string& op_type;
  const ge::Operator& op_paras;
  const nlohmann::json& compile_info;
  const OpInfo& op_info;
  utils::OpRunInfo& run_info;
  concat::CompileInfo c_info;
  int64_t input_nums{0};
  int64_t max_available_ub{0};
  int64_t block_axis{-1};
  int64_t block_factor{-1};
  int64_t low_ub_factor{-1};
  int64_t high_ub_factor{-1};
  int64_t block_dims{1};
  int64_t real_factor_n{1};
  int64_t factor_col{1};
  int64_t ori_output_col{1};
  uint64_t tiling_key{0};
  ge::DataType dtype{ge::DataType::DT_FLOAT16};
  const bool has_op_info{false};
  bool need_multi_core{false};
  bool is_one_concat{false};
  bool use_one_concat{false};
  bool read_align_no_ub{false};
  bool all_concat_align{false};
  bool all_half_align{false};
  bool no_align{false};
  bool all_one_concat{false};
  bool is_empty{false};
  std::array<std::array<int64_t, MAX_DIM_LEN>, MAX_INPUT_NUM> input_shapes{};
  std::array<int64_t, MAX_DIM_LEN> output_shapes{};
  std::array<int64_t, MAX_INPUT_NUM> align_factors{};
  std::array<int64_t, MAX_INPUT_NUM> offsets{};
};
}  // namespace concat

/*
 * @brief: tiling function of concat operator
 * @param [in] op_type: op_type of the concat operator
 * @param [in] op_paras: inputs/outputs/atts of the concat operator
 * @param [in] compile_info: compile time generated info of the concat operator
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool ConcatDsl(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& compile_info,
               utils::OpRunInfo& run_info);

/*
 * @brief: tiling function of concat operator
 * @param [in] op_type: op_type of the concat operator
 * @param [in] op_paras: inputs/outputs/atts of the concat operator
 * @param [in] compile_info: compile time generated info of the concat operator
 * @param [in] op_info: operator info, example shapes and axis
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool ConcatDsl(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& compile_info,
               utils::OpRunInfo& run_info, const OpInfo& op_info);

class ConcatDslTilingHandler : public AutoTilingHandler {
 public:
  ConcatDslTilingHandler(const std::string& o, const std::string& p, const nlohmann::json& c)
      : AutoTilingHandler(o, p), compile_info(c) {
  }
  ~ConcatDslTilingHandler() = default;
  bool DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const override;
  bool DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info, const OpInfo& op_info) const override;

 private:
  const nlohmann::json compile_info;
};
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CONCAT_DSL_H_
