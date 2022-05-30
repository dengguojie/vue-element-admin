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
 * \file elewise_v3.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_ELEWISE_V3_H_
#define OPS_BUILT_IN_OP_TILING_ELEWISE_V3_H_

#include <vector>
#include <string>
#include <unordered_set>
#include "vector_tiling.h"
#include "external/graph/operator.h"

namespace optiling {
namespace v3 {

struct ElewiseCompileInfo {
  ElewiseCompileInfo() = default;
  ElewiseCompileInfo(const std::string& op_type, const nlohmann::json& outer_compile_info);

  // required compile_info
  uint32_t classify_inputs_num{0};
  uint32_t flag_info_size{0};
  bool only_const_tiling{false};
  int64_t ub_factor_align{-1};
  // optional compile_info
  bool classify_const_mode{false};
  bool support_broadcast{false};
  bool absorbable_broadcast{false};
  std::pair<bool, std::unordered_map<std::string, std::vector<int64_t>>> base_info;
  std::pair<bool, std::vector<int64_t>> const_block_dims;
  std::pair<bool, std::unordered_map<std::string, std::vector<int64_t>>> elewise_vars;
  VarAttrWrap varAttrWrap;

 private:
  // required compile info parser functions
  void ParseClassifyNum(const nlohmann::json& outer_compile_info);
  void ParseFlagInfo(const nlohmann::json& outer_compile_info);
  void ParseUbFactorAlign(const nlohmann::json& outer_compile_info);
  void ParseRequiredCompileInfo(const nlohmann::json& outer_compile_info);
  // optional compile info parser function
  bool ParseVarsAttr(const nlohmann::json& outer_compile_info);
  void ParseBaseInfo(const nlohmann::json& outer_compile_info);
  void ParseConstCompileInfo(const nlohmann::json& outer_compile_info);
  void ParseElewiseVar(const nlohmann::json& outer_compile_info);
  bool ParseOptionalCompileInfo(const nlohmann::json& outer_compile_info);
};

enum class ElewisePattern {
  CONST = 000,
  COMMON = 100,
  BROADCAST = 200,
  BROADCAST_SCALAR = 230,
  SCALAR_BROADCAST = 320,
  UNKNOWN = 666
};

class Elewise {
 public:
  explicit Elewise(const std::string& op_type,
                   const ge::Operator& op_paras,
                   const ElewiseCompileInfo& compile_info,
                   utils::OpRunInfo& run_info)
      : op_type(op_type), op_paras(op_paras), compile_info(compile_info), run_info(run_info) {}
  ~Elewise() = default;
  bool DoTiling();
  bool DoTiling(const OpInfo& op_info);
  void SetBroadcastPattern(const ElewisePattern& pattern);

 private:
  bool CheckCompileInfo();
  void GetOutputDtype();
  void GetCheckInputs(std::vector<uint32_t>& check_list);
  void GetCheckInputs(std::vector<uint32_t>& check_list, const OpInfo& op_info);
  bool GetShapeUnderCheck(std::vector<uint32_t>& check_list);
  bool GetShapeUnderCheck(std::vector<uint32_t>& check_list, const OpInfo& op_info);
  bool GetInOutShapes();
  bool GetInOutShapes(const OpInfo& op_info);
  bool WriteKnownData();
  bool CalcConstKey();
  bool ConstModeTiling();
  bool EmptyModeTiling();
  bool CalcPatternKey();
  bool ParseBaseInfo();
  void CalcMultiCore();
  void DoBlockTiling();
  bool DoUbTiling();
  void CalcTilingKey();
  bool WriteTilingData() const;
  bool SpecialModeTiling();

 private:
  const std::string& op_type;
  const ge::Operator& op_paras;
  const ElewiseCompileInfo& compile_info;
  utils::OpRunInfo& run_info;
  // input infos
  uint32_t input_num{0};
  std::vector<int64_t> input_fuse_shapes{};
  std::unordered_set<int64_t> fuse_diff_shapes{};
  // output infos
  int64_t out_shape{1};
  ge::DataType out_dtype{ge::DataType::DT_MAX};
  // base infos
  int64_t core_num{-1};
  int64_t max_dtype{-1};
  int64_t max_available_ub{-1};
  int64_t max_available_ub_db{-1};
  // tiling infos
  bool need_multi_core{true};
  bool need_double_buffer{false};
  uint64_t tiling_key{1};
  int64_t block_dims{1};
  int64_t block_factor{1};
  int64_t ub_factor{1};
  bool broadcast_dispatch{false};
  ElewisePattern classify_pattern{ElewisePattern::UNKNOWN};
};

ElewisePattern GetDispatchPattern(std::vector<std::vector<int64_t>> elewise_inputs, const uint32_t& classify_nums);
}  // namespace v3

class ElewiseTilingHandler: public AutoTilingHandler {
 public:
  ElewiseTilingHandler(const std::string& o, const std::string& p, const nlohmann::json& c)
      : AutoTilingHandler(o, p), elewise_compile_info(o, c) {}
  ~ElewiseTilingHandler() = default;
  bool DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const override;
  bool DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info, const OpInfo& op_info) const override;

 private:
  const v3::ElewiseCompileInfo elewise_compile_info;
};
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_ELEWISE_V3_H_
