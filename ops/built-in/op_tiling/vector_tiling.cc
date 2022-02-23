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
    }else if (pattern == "Transpose") {
      return CreateTransposeDslTilingHandler(op_type, pattern, parsed_compile_info);
    } else if (pattern == "Transdata") {
      return CreateTransdataTilingHandler(op_type, pattern, parsed_compile_info);
    } else if (pattern == "Concat") {
      return CreateConcatDslTilingHandler(op_type, pattern, parsed_compile_info);
    } else {
      OP_LOGE(op_type.c_str(), "Pattern %s is not supported by AutoTiling Compile Info Parser", pattern.c_str());
      return std::shared_ptr<AutoTilingHandler>(nullptr);
    }
  } catch (...) {
    OP_LOGE(op_type.c_str(), "Unknown Exception encountered when parsing Compile Info of pattern %s", pattern.c_str());
    return std::shared_ptr<AutoTilingHandler>(nullptr);
  }
}

bool ParseVarAttr(const nlohmann::json& json_info,
                  std::unordered_map<std::uint64_t, vector<VarAttr>>& var_attr_map) {
  const auto& tiling_key_map = json_info.at("_attr_vars");
  for (const auto& item : tiling_key_map.items()) {
    const std::uint64_t& tiling_key = std::stoull(item.key());
    const auto& all_attr_vars = item.value();

    const auto& ret = var_attr_map.emplace(std::piecewise_construct, forward_as_tuple(tiling_key), forward_as_tuple());
    auto& var_attr_list = ret.first->second;
    var_attr_list.reserve(all_attr_vars.size());
    for (const auto& var : all_attr_vars) {
      var_attr_list.emplace_back(var.at("name"), var.at("type"), var.at("src_type"), var.at("length"));
    }
  }

  return true;
}

bool SetAttrVars(const std::string& op_type, const ge::Operator& op_paras,
                 utils::OpRunInfo& run_info, const std::vector<VarAttr>& all_attr_vars) {
  try {
      for (VarAttr varAttr : all_attr_vars) {
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
