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
 * \file vector_tiling.cc
 * \brief tiling function of vector ops
 */
#include "vector_tiling.h"
#include "tiling_handler.h"

namespace optiling {
namespace {
  constexpr int32_t VAR_ATTR_MODE_NOT_EXIST = -1;
  constexpr int32_t VAR_ATTR_MODE_CONSISTENT = 0;
  constexpr int32_t VAR_ATTR_MODE_INDEPENDENT = 1;
}
const std::vector<vector<int32_t>> OpInfo::dummy_variable;

std::shared_ptr<AutoTilingHandler> CreateAutoTilingHandler(const std::string& op_type, const std::string& pattern,
                                                           const nlohmann::json& parsed_compile_info) {
  OP_LOGI(op_type.c_str(), "Entering AutoTiling Compile Info Parser for pattern %s", pattern.c_str());
  try {
    if (pattern == "CommReduce") {
      return CreateReduceTilingHandler(op_type, pattern, parsed_compile_info);
    } else if (pattern == "ElemWise") {
      return CreateElewiseTilingHandler(op_type, pattern, parsed_compile_info);
    } else if (pattern == "Broadcast") {
      return CreateBroadcastTilingHandler(op_type, pattern, parsed_compile_info);
    } else if (pattern == "Norm") {
      return CreateNormTilingHandler(op_type, pattern, parsed_compile_info);
    } else if (pattern == "Gather") {
        return CreateGatherTilingHandler(op_type, pattern, parsed_compile_info);
    } else if (pattern == "Slice") {
      return CreateSliceTilingHandler(op_type, pattern, parsed_compile_info);
    }else if (pattern == "Transpose") {
      return CreateTransposeDslTilingHandler(op_type, pattern, parsed_compile_info);
    } else if (pattern == "Transdata") {
      return CreateTransdataTilingHandler(op_type, pattern, parsed_compile_info);
    } else if (pattern == "Concat") {
      return CreateConcatDslTilingHandler(op_type, pattern, parsed_compile_info);
    } else if (pattern == "Split") {
      return CreateSplitDslTilingHandler(op_type, pattern, parsed_compile_info);
    } else {
      OP_LOGE(op_type.c_str(), "Pattern %s is not supported by AutoTiling Compile Info Parser", pattern.c_str());
      return std::shared_ptr<AutoTilingHandler>(nullptr);
    }
  } catch (...) {
    OP_LOGE(op_type.c_str(), "Unknown Exception encountered when parsing Compile Info of pattern %s", pattern.c_str());
    return std::shared_ptr<AutoTilingHandler>(nullptr);
  }
}

bool VarAttrWrap::ParseVarAttr(const nlohmann::json& json_info) {
  if (json_info.count("_var_attr_mode") <= 0) {
    return true;
  }

  mode = json_info.at("_var_attr_mode").get<std::int32_t>();
  if (mode == VAR_ATTR_MODE_CONSISTENT) {
    const auto& json_var_attrs = json_info.at("_var_attrs");
    var_attrs.reserve(json_var_attrs.size());
    for (const auto& var : json_var_attrs) {
      var_attrs.emplace_back(var.at("name"), var.at("type"), var.at("src_type"), var.at("length"));
    }
  } else {
    const auto& json_var_attr_map = json_info.at("_var_attrs");
    for (const auto& item : json_var_attr_map.items()) {
      const std::uint64_t& tiling_key = std::stoull(item.key());
      const auto& json_var_attrs = item.value();

      const auto& ret = var_attr_map.emplace(std::piecewise_construct,
                                             forward_as_tuple(tiling_key),
                                             forward_as_tuple());
      auto& var_attrs_of_map = ret.first->second;
      var_attrs_of_map.reserve(json_var_attrs.size());
      for (const auto& var : json_var_attrs) {
        var_attrs_of_map.emplace_back(var.at("name"), var.at("type"), var.at("src_type"), var.at("length"));
      }
    }
  }

  return true;
}

bool VarAttrWrap::WriteVarAttrs(const uint64_t tiling_key, const std::string& op_type, const ge::Operator& op_paras,
                                utils::OpRunInfo& run_info) const {
  if (mode == VAR_ATTR_MODE_NOT_EXIST) {
    return true;
  }

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
}

bool VarAttrWrap::SetVarAttrs(const std::string& op_type, const ge::Operator& op_paras,
                              utils::OpRunInfo& run_info, const vector<VarAttr>& var_attrs) const {
  try {
    for (VarAttr varAttr : var_attrs) {
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
} // namespace optiling