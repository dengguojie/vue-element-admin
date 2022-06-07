/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file vector_tiling_rt2.cc
 * \brief tiling function of vector ops
 */

#include "vector_tiling_rt2.h"
#include "register/op_tiling_attr_utils.h"
#include "vector_tiling_log.h"
#include "error_log.h"


namespace optiling {
namespace {
constexpr int32_t VAR_ATTR_MODE_NOT_EXIST = -1;
constexpr int32_t VAR_ATTR_MODE_CONSISTENT = 0;
constexpr int32_t VAR_ATTR_MODE_INDEPENDENT = 1;
}

gert::AttrDataType GetGeAttrDataType(const std::string& attr_data_type) {
  const std::unordered_map<std::string, gert::AttrDataType> attr_data_type_map {
    {"bool", gert::AttrDataType::kBool},
    {"str", gert::AttrDataType::kString},
    {"string", gert::AttrDataType::kString},
    {"int", gert::AttrDataType::kInt32},
    {"int32", gert::AttrDataType::kInt32},
    {"int64", gert::AttrDataType::kInt64},
    {"uint", gert::AttrDataType::kUint32},
    {"uint32", gert::AttrDataType::kUint32},
    {"float", gert::AttrDataType::kFloat32},
    {"float32", gert::AttrDataType::kFloat32},
    {"float16", gert::AttrDataType::kFloat16},
    {"list_bool", gert::AttrDataType::kListBool},
    {"list_str", gert::AttrDataType::kListString},
    {"list_string", gert::AttrDataType::kListString},
    {"list_int", gert::AttrDataType::kListInt32},
    {"list_int32", gert::AttrDataType::kListInt32},
    {"list_int64", gert::AttrDataType::kListInt64},
    {"list_uint", gert::AttrDataType::kListUint32},
    {"list_uint32", gert::AttrDataType::kListUint32},
    {"list_float", gert::AttrDataType::kListFloat32},
    {"list_float32", gert::AttrDataType::kListFloat32},
    {"list_float16", gert::AttrDataType::kListFloat16},
    {"list_list_int", gert::AttrDataType::kListListInt32},
    {"list_list_int32", gert::AttrDataType::kListListInt32},
    {"list_list_int64", gert::AttrDataType::kListListInt64}
  };
  if (attr_data_type_map.find(attr_data_type) == attr_data_type_map.end()) {
    return gert::AttrDataType::kTypeEnd;
  }
  return attr_data_type_map.at(attr_data_type);
}
void OpInfoImpl::SetInputShape(const std::vector<gert::Shape>* _op_input_ge_shapes) {
  op_input_ge_shapes_ptr = _op_input_ge_shapes;
  if (_op_input_ge_shapes) {
    op_input_shapes.resize(_op_input_ge_shapes->size());
    for (size_t i = 0; i < _op_input_ge_shapes->size(); i++) {
      std::vector<int64_t> shape(_op_input_ge_shapes->at(i).GetDimNum());
      for (size_t j = 0; j < _op_input_ge_shapes->at(i).GetDimNum(); j++) {
        shape[j] = _op_input_ge_shapes->at(i).GetDim(j);
      }
      op_input_shapes[i] = shape;
    }
  }
}

void OpInfoImpl::SetInputShape(const std::vector<std::vector<int64_t>>* _op_input_shapes) {
  op_input_shapes_ptr = _op_input_shapes;
  if (_op_input_shapes) {
    op_input_ge_shapes.resize(_op_input_shapes->size());
    for (size_t i = 0; i < _op_input_shapes->size(); i++) {
      gert::Shape shape;
      for (size_t j = 0; j < _op_input_shapes->at(i).size(); j++) {
        shape.AppendDim(_op_input_shapes->at(i)[j]);
      }
      op_input_ge_shapes[i] = shape;
    }
  }
}

void OpInfoImpl::SetAxes(const std::vector<int64_t>* _op_axes) {
  op_axes_ptr = _op_axes;
}

void OpInfoImpl::SetAxes(const std::vector<int32_t>* _op_axes) {
  if (_op_axes) {
    op_axes_d = {*_op_axes};
    op_axes.resize(_op_axes->size());
    for (size_t i = 0; i < _op_axes->size(); i++) {
      op_axes[i] = static_cast<int64_t>(_op_axes->at(i));
    }
  }
}

void OpInfoImpl::SetInputType(const ge::DataType* _op_in_type) {
  op_in_type = _op_in_type;
}

const std::vector<std::vector<int64_t>>* OpInfoImpl::GetInputShape() const {
  if (op_input_shapes_ptr) {
    return op_input_shapes_ptr;
  }
  if (op_input_shapes.empty()) {
    return nullptr;
  }
  return &op_input_shapes;
}

const std::vector<gert::Shape>* OpInfoImpl::GetInputGeShape() const {
  if (op_input_ge_shapes_ptr) {
    return op_input_ge_shapes_ptr;
  }
  if (op_input_ge_shapes.empty()) {
    return nullptr;
  }
  return &op_input_ge_shapes;
}

const std::vector<int64_t>* OpInfoImpl::GetAxes() const {
  if (op_axes_ptr) {
    return op_axes_ptr;
  }
  if (op_axes.empty()) {
    return nullptr;
  }
  return &op_axes;
}

const ge::DataType* OpInfoImpl::GetInType() const {
  return op_in_type;
}

const AutoTilingCompileInfo* OpInfoImpl::GetCompileInfo() const {
  return compile_info;
}

bool VarAttrWrap_rt2::ParseVarAttr(const nlohmann::json& json_info) {
  if (json_info.count("_var_attr_mode") <= 0) {
    return true;
  }

  mode = json_info.at("_var_attr_mode").get<std::int32_t>();
  if (mode == VAR_ATTR_MODE_CONSISTENT) {
    const auto& json_var_attrs = json_info.at("_var_attrs");
    var_attrs.reserve(json_var_attrs.size());
    for (const auto& var : json_var_attrs) {
      var_attrs.emplace_back(var.at("name"), var.at("index"), var.at("type"), var.at("src_type"), var.at("length"));
    }
  } else {
    const auto& json_var_attr_map = json_info.at("_var_attrs");
    for (const auto& item : json_var_attr_map.items()) {
      const std::uint64_t& tiling_key = std::stoull(item.key());
      const auto& json_var_attrs = item.value();

      const auto& ret = var_attr_map.emplace(std::piecewise_construct,
                                             std::forward_as_tuple(tiling_key),
                                             std::forward_as_tuple());
      auto& var_attrs_of_map = ret.first->second;
      var_attrs_of_map.reserve(json_var_attrs.size());
      for (const auto& var : json_var_attrs) {
        var_attrs_of_map.emplace_back(var.at("name"), var.at("index"), var.at("type"), var.at("src_type"),
                                      var.at("length"));
      }
    }
  }
  return true;
}

bool VarAttrWrap_rt2::WriteVarAttrs(const uint64_t tiling_key, gert::TilingContext& context) const {
     if (mode == VAR_ATTR_MODE_CONSISTENT) {
    return SetVarAttrs(var_attrs, context);
  }

  if (mode == VAR_ATTR_MODE_INDEPENDENT) {
    if (var_attr_map.count(tiling_key) < 0) {
      OP_LOGD(context.GetNodeType(), "TilingKey of %d do not has var attrs.", tiling_key);
      return true;
    } else {
      return SetVarAttrs(var_attr_map.at(tiling_key), context);
    }
  }

  return true;
}

bool VarAttrWrap_rt2::SetVarAttrs(const std::vector<VarAttr_rt2>& var_attrs, gert::TilingContext& context) const {
  try {
    for (VarAttr_rt2 var_attr : var_attrs) {
      gert::AttrDataType ge_var_src_type = GetGeAttrDataType(var_attr.src_type);
      gert::AttrDataType ge_var_type = GetGeAttrDataType(var_attr.type);

      gert::TilingData* tiling_data = context.GetRawTilingData();
      ge::graphStatus graphStatus = tiling_data->AppendConvertedAttrVal(context.GetAttrs(),
                                                                        var_attr.index,
                                                                        ge_var_src_type,
                                                                        ge_var_type);
      if (graphStatus != ge::GRAPH_SUCCESS) {
        VECTOR_INNER_ERR_REPORT_TILIING(context.GetNodeType(),
                                        "getAttrVars from GE error. Error message: var index is %zu , var type is %s, "
                                        "var_src_type is %s",
                                        var_attr.index, var_attr.type.c_str(), var_attr.src_type.c_str());
        return false;
      }
    }
  } catch (const std::exception &e) {
    VECTOR_INNER_ERR_REPORT_TILIING(context.GetNodeType(), "SetAttrVars error. Error message: %s", e.what());
    return false;
  }
  return true;
}

bool VarAttrWrap_rt2::WriteVarAttrs(const uint64_t tiling_key, const std::string& op_type,
                                    const ge::Operator& op_paras, utils::OpRunInfo& run_info) const {
  if (mode == VAR_ATTR_MODE_CONSISTENT) {
    return SetVarAttrs(op_type, op_paras, run_info, var_attrs);
  }

  if (mode == VAR_ATTR_MODE_INDEPENDENT) {
    if (var_attr_map.count(tiling_key) < 0) {
      OP_LOGD(op_type.c_str(), "TilingKey of %d do not has var attrs.", tiling_key);
      return true;
    } else {
      return SetVarAttrs(op_type, op_paras, run_info, var_attr_map.at(tiling_key));
    }
  }

  return true;
}

bool VarAttrWrap_rt2::SetVarAttrs(const std::string& op_type, const ge::Operator& op_paras,
                                  utils::OpRunInfo& run_info, const vector<VarAttr_rt2>& var_attrs) const {
  try {
    for (VarAttr_rt2 varAttr : var_attrs) {
      const char* var_name = varAttr.name.c_str();
      const char* var_type = varAttr.type.c_str();
      const char* var_src_type = varAttr.src_type.c_str();

      AttrDataPtr data;
      ge::graphStatus graphStatus = GetOperatorAttrValue(op_paras, var_name, var_src_type, data, var_type);
      if (graphStatus != ge::GRAPH_SUCCESS) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type,
                                        "getAttrVars from FE error. Error message: var name is %s , var type is %s, "
                                        "var_src_type is %s",
                                        var_name, var_type, var_src_type);
        return false;
      }

      run_info.AddTilingData(reinterpret_cast<const char*>(data->GetData()), data->GetSize());
    }
  } catch (const std::exception &e) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "SetAttrVars error. Error message: %s", e.what());
    return false;
  }
  return true;
}

// Compatible code, please do not use
const std::vector<std::vector<int64_t>>& OpInfoImpl::GetInputShapeD() const {
  return *op_input_shapes_ptr;
}

// Compatible code, please do not use
const std::vector<std::vector<int32_t>>& OpInfoImpl::GetReduceAxesD() const {
  return op_axes_d;
}

// Compatible code, please do not use
const ge::DataType& OpInfoImpl::GetInTypeD() const {
  return *op_in_type;
}
} // namespace optiling