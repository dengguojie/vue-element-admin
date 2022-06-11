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
 * \file vector_tiling_rt2.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_VECTOR_TILING_RT2_H_
#define OPS_BUILT_IN_OP_TILING_VECTOR_TILING_RT2_H_


#include <vector>
#include <unordered_map>
#include <nlohmann/json.hpp>

#include "graph/types.h"
#include "graph/operator.h"
#include "register/op_tiling_info.h"
#include "exe_graph/runtime/tiling_context.h"
#include "vector_op_info.h"

namespace optiling {
enum class SchPattern {
    ELETWISE,
    BROADCAST,
    COMMONREDUCE,
    TUPLEREDUCE,
    NORM,
    CONCAT,
    SPLIT,
    SLICE,
    GATHER,
    TRANSPOSE,
    TRANSDATA,
    DEFAULT
};


struct VarAttr_rt2 {
  VarAttr_rt2(const std::string& _name, size_t _index,
          const std::string& _type, const std::string& _src_type, const int32_t& _length):
          name (_name), index(_index), type (_type), src_type (_src_type), length (_length) {}
  std::string name;
  size_t index;
  std::string type;
  std::string src_type;
  int32_t length{0};
};

class VarAttrWrap_rt2 {
public:
  bool ParseVarAttr(const nlohmann::json& json_info);
  bool WriteVarAttrs(const uint64_t tiling_key, const std::string& op_type, const ge::Operator& op_paras,
                     utils::OpRunInfo& run_info) const;
  bool WriteVarAttrs(const uint64_t tiling_key, gert::TilingContext& context) const;

private:
  bool SetVarAttrs(const std::string& op_type, const ge::Operator& op_paras,
                   utils::OpRunInfo& run_info, const std::vector<VarAttr_rt2>& var_attrs) const;
  bool SetVarAttrs( const std::vector<VarAttr_rt2>& var_attrs, gert::TilingContext& context) const;

private:
  int32_t mode{-1};
  std::vector<VarAttr_rt2> var_attrs;
  std::unordered_map<std::uint64_t, std::vector<VarAttr_rt2>> var_attr_map;
};
struct AutoTilingCompileInfo {
  SchPattern pattern;
  VarAttrWrap_rt2 var_attr_wrap;

public:
  AutoTilingCompileInfo() {}
  virtual ~AutoTilingCompileInfo() {}
};

class OpInfoImpl {
public:
  explicit OpInfoImpl() = default;
  explicit OpInfoImpl(const AutoTilingCompileInfo* _compile_info) : compile_info(_compile_info) {}
  ~OpInfoImpl() = default;

  void SetInputShape(const std::vector<gert::Shape>* _op_input_ge_shapes);
  void SetInputShape(const std::vector<std::vector<int64_t>>* _op_input_shapes);
  void SetAxes(const std::vector<int32_t>* _op_axes);
  void SetAxes(const std::vector<int64_t>* _op_axes);
  void SetInputType(const ge::DataType* _op_in_type);
  const std::vector<std::vector<int64_t>>* GetInputShape() const;
  const std::vector<gert::Shape>* GetInputGeShape() const;
  const std::vector<int64_t>* GetAxes() const;
  const ge::DataType* GetInType() const;
  const AutoTilingCompileInfo* GetCompileInfo() const;

  // Compatible code, please do not use
  const std::vector<std::vector<int64_t>>& GetInputShapeD() const;
  // Compatible code, please do not use
  const std::vector<std::vector<int32_t>>& GetReduceAxesD() const;
  // Compatible code, please do not use
  const ge::DataType& GetInTypeD() const;

private:
  std::vector<std::vector<int64_t>> op_input_shapes;
  std::vector<gert::Shape> op_input_ge_shapes;
  std::vector<int64_t> op_axes;
  std::vector<std::vector<int32_t>> op_axes_d;
  const std::vector<std::vector<int64_t>>* op_input_shapes_ptr{nullptr};
  const std::vector<gert::Shape>* op_input_ge_shapes_ptr{nullptr};
  const ge::DataType* op_in_type{nullptr};
  const std::vector<int64_t>* op_axes_ptr{nullptr};
  const AutoTilingCompileInfo* compile_info{nullptr};
};

class OpInfoImplGetter {
public:
  static std::shared_ptr<OpInfoImpl> GetOpInfoImpl(const OpInfo* obj) {
    if (obj == nullptr) {
      return nullptr;
    }
    return obj->op_info_impl;
  }
};
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_VECTOR_TILING_RT2_H_
