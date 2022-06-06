/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include <nlohmann/json.hpp>
#include "vector_tiling.h"
#include "auto_tiling.h"
#include "auto_tiling_context.h"
#include "rl_tune.h"

namespace optiling {
namespace v3 {

struct ElewiseCompileInfo : AutoTilingCompileInfo{
  ElewiseCompileInfo() = default;
  ElewiseCompileInfo(const std::string& op_type, const nlohmann::json& outer_compile_info);
  ~ElewiseCompileInfo() override = default;

  bool Parse(const char* op_type, const nlohmann::json& json_compile_info);

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
  // rl bank info
  std::pair<bool, std::vector<std::pair<rl::RlPattern, std::vector<rl::RlBankInfo>>>> bank_info_pair;

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

template <typename T>
class Elewise {
 public:
  explicit Elewise(T* _context, const OpInfoImpl* _op_info)
      : context(_context),
        op_info(_op_info) {
  }
  ~Elewise() = default;
  bool DoTiling();
  bool DoTiling(const OpInfo& op_info);
  void SetBroadcastPattern(const ElewisePattern& pattern);

 private:
  void GetOutputDtype();
  bool CheckCompileInfo();
  void GetCheckInputs(std::vector<uint32_t>& check_list);
  bool GetShapeUnderCheckCustom(std::vector<uint32_t>& check_list);
  bool GetShapeUnderCheck(std::vector<uint32_t>& check_list);
  bool GetInOutShapes();
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
  bool WriteRlTilingData(const rl::RlBankInfo& rl_bank_info) const;
  bool DoRlTiling(const rl::RlBankInfo& rl_bank_info);
  bool TryMatchRlBank();

 private:
  T* context;
  const OpInfoImpl* op_info{nullptr};
  const char* op_type;
  const ElewiseCompileInfo* compile_info;
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
  // rl
  bool hit_rl_bank{false};
  int64_t rl_ub_factor{1};
  int64_t rl_block_factor{1};
  int64_t rl_block_dim{1};
};

ElewisePattern GetDispatchPattern(std::vector<std::vector<int64_t>> elewise_inputs, const uint32_t& classify_nums);

template class Elewise<AutoTilingContext>;
template class Elewise<AutoTilingOp>;
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
