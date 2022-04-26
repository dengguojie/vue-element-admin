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
#include "vector_tiling.h"
#include "external/graph/operator.h"

namespace optiling {
namespace v3 {

struct ElewiseCompileInfo {
  ElewiseCompileInfo() = default;
  ElewiseCompileInfo(const std::string& op_type, const nlohmann::json& outer_compile_info);
  void SetBroadcastPattern(const bool& is_broadcast_pattern);

  // required compile_info
  uint32_t flag_info_size{0};
  bool only_const_tiling{false};
  bool is_const_shapes{false};
  bool use_special_pattern{true};
  int64_t ub_factor_align{-1};
  // optional base_info
  int64_t pattern_key{-1};
  int64_t core_num{-1};
  int64_t max_dtype{-1};
  int64_t max_available_ub{-1};
  int64_t max_available_ub_db{-1};
  // const_core_dims
  int64_t const_block_dims{-1};
  // elewise_vars_size
  uint32_t elewise_vars_size{0};
  // tiling info from broadcast
  bool broadcast_pattern{false};
  VarAttrWrap varAttrWrap;

 private:
  void ParseFlagInfoSize(const nlohmann::json& outer_compile_info);
  void ParseBaseInfo(const nlohmann::json& outer_compile_info);
  void ParseConstDims(const nlohmann::json& outer_compile_info);
  void ParseElewiseVarSize(const nlohmann::json& outer_compile_info);
  bool ParseAttrVars(const nlohmann::json& outer_compile_info);
  void ParseUbFactorAlign(const nlohmann::json& outer_compile_info);
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

 private:
  const int64_t GetElementByType(const ge::DataType& dtype) const;
  bool CheckCompileInfo();
  bool CheckInOutNum();
  bool CheckInOutNum(const OpInfo& op_info);
  bool GetShapeUnderCheck();
  bool GetShapeUnderCheck(const OpInfo& op_info);
  void GetShapeFromBroadcast(const OpInfo& op_info);
  bool GetOutShape();
  bool GetOutShape(const OpInfo& op_info);
  void GetOutputDtype();
  bool WriteKnownData();
  bool DoConstTiling();
  bool DoEmptyTiling();
  void CalcMultiCore();
  void DoBlockTiling();
  bool DoUbTiling();
  void CalcCommonKey();
  bool DoCommonTiling();
  bool WriteCommonData() const;

 private:
  const std::string& op_type;
  const ge::Operator& op_paras;
  const ElewiseCompileInfo& compile_info;
  utils::OpRunInfo& run_info;
  // tiling info
  bool need_multi_core{true};
  bool need_double_buffer{false};
  uint64_t tiling_key{1};
  int64_t block_dims{1};
  int64_t ub_factor{1};
  int64_t block_factor{1};
  int64_t out_shape{1};
  ge::DataType out_dtype{ge::DataType::DT_MAX};
};
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
