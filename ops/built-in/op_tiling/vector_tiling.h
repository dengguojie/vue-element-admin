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
 * \file vector_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_VECTOR_TILING_H_
#define OPS_BUILT_IN_OP_TILING_VECTOR_TILING_H_

#include <string>
#include <map>
#include <chrono>
#include <memory>
#include <cstdlib>
#include <nlohmann/json.hpp>

#include "register/op_tiling_registry.h"
#include "register/op_tiling_attr_utils.h"
#include "op_log.h"
#include "error_log.h"
#include "graph/debug/ge_log.h"
#include "vector_tiling_log.h"

#include "vector_op_info.h"
#include "op_tiling.h"

namespace optiling {

class AutoTilingHandler: public CompileInfoBase {
public:
  AutoTilingHandler(const std::string& o, const std::string& p) : op_type(o), pattern(p) {}
  virtual bool DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const = 0;
  virtual bool DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info, const OpInfo& op_info) const = 0;
  virtual ~AutoTilingHandler() = default;

protected:
  const std::string op_type;
  const std::string pattern;
};

struct VarAttr {
  VarAttr(const std::string& _name, int32_t _index,
          const std::string& _type, const std::string& _src_type, const int32_t& _length):
          name (_name), index(_index), type (_type), src_type (_src_type), length (_length) {}
  std::string name;
  int32_t index;
  std::string type;
  std::string src_type;
  int32_t length{0};
};


class VarAttrWrap {
public:
  bool ParseVarAttr(const nlohmann::json& json_info);
  bool WriteVarAttrs(const uint64_t tiling_key, const std::string& op_type, const ge::Operator& op_paras,
                     utils::OpRunInfo& run_info) const;

private:
  bool SetVarAttrs(const std::string& op_type, const ge::Operator& op_paras,
                   utils::OpRunInfo& run_info, const vector<VarAttr>& var_attrs) const;

private:
  int32_t mode{-1};
  std::vector<VarAttr> var_attrs;
  std::unordered_map<std::uint64_t, vector<VarAttr>> var_attr_map;
};

std::shared_ptr<AutoTilingHandler> CreateAutoTilingHandler(const std::string& op_type, const std::string& pattern,
                                                           const nlohmann::json& parsed_compile_info);
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_VECTOR_TILING_H_
