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
 * \file tiling_dispatch.cpp
 * \brief tiling function of ops
 */
#include "vector_tiling.h"
#ifdef ASCEND_OPTILING_UT
#include "tests/ut/op_tiling_test/mock_tiling_dispatch.h"
#endif

namespace optiling {


/*
 * @brief: tiling function of ops
 * @param [in] op_type: opType of ops
 * @param [in] op_paras: inputs/outputs/attrs of ops
 * @param [in] compile_info: compile time generated info of ops
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool AutoTiling(const ge::Operator& op_paras, const void* compile_info,
                utils::OpRunInfo& run_info) {
#ifdef ASCEND_OPTILING_UT
  OP_LOGI("AutoTiling", "Entering AutoTiling Dispatcher.");
#else
  OP_LOGI("AutoTiling", "Entering AutoTiling Dispatcher UT Mode.");
#endif
  if (compile_info != nullptr) {
    const AutoTilingCompileInfo& outer_compile_info = *(static_cast<const AutoTilingCompileInfo*>(compile_info));
    const std::string& op_type = outer_compile_info.op_type;
    const std::string& pattern = outer_compile_info.pattern;
    const std::shared_ptr<void> inner_compile_info = outer_compile_info.compile_info;
    if (inner_compile_info == nullptr) {
      OP_LOGE(op_type.c_str(),
              "AutoTiling Dispatcher received nullptr for Compile Info of pattern %s", pattern.c_str());
      return false;
    }
    OP_LOGI(op_type.c_str(), "Compile Info of pattern %s received successfully", pattern.c_str());
    bool ret = false;
    if (pattern == "CommReduce") {
      ret = ReduceTiling(op_type, op_paras, *static_pointer_cast<nlohmann::json>(inner_compile_info), run_info);
    } else if (pattern == "ElemWise") {
      ret = EletwiseTiling(op_type, op_paras, *static_pointer_cast<nlohmann::json>(inner_compile_info), run_info);
    } else if (pattern == "Broadcast") {
      ret = EletwiseTiling(op_type, op_paras, *static_pointer_cast<nlohmann::json>(inner_compile_info), run_info);
    } else if (pattern == "Norm") {
      ret = NormTiling(op_type, op_paras, *static_pointer_cast<nlohmann::json>(inner_compile_info), run_info);
    }  else if (pattern == "Transpose") {
      ret = TransposeDsl(op_type, op_paras, *static_pointer_cast<nlohmann::json>(inner_compile_info), run_info);
    } else {
      ret = false;
      OP_LOGE(op_type.c_str(), "Compile Info of pattern %s is not supported", pattern.c_str());
    }
    return ret;
  } else {
    OP_LOGE("AutoTiling", "AutoTiling Dispatcher received nullptr, Compile Info Parser is not working properly.");
    return false;
  }
}

void* AutoTilingCompileInfoParser(const ge::Operator& op_paras, const ge::AscendString& compile_info_str) {
#ifdef ASCEND_OPTILING_UT  
  OP_LOGI("AutoTiling", "Entering AutoTiling Compile Info Parser.");
#else
  OP_LOGI("AutoTiling", "Entering AutoTiling Compile Info Parser UT Mode.");
#endif
  std::string unknown = "UNKNOWN";
  std::string& op_type = unknown;
  ge::AscendString ascend_op_type;
  ge::graphStatus ret = op_paras.GetOpType(ascend_op_type);
  if (ret == ge::GRAPH_SUCCESS) {
    op_type = ascend_op_type.GetString();
  } else {
    OP_LOGE("UNKNOWN_OP_TYPE", "get op type failed");
    return nullptr;
  }

  // Default parsing sequence, parse compile info directly to nlohmann::json object
  const nlohmann::json& parsed_compile_info = nlohmann::json::parse(compile_info_str.GetString());
  OP_TILING_CHECK((parsed_compile_info.find("_pattern") == parsed_compile_info.end()),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "compile info not contain [_pattern]"),
                  return nullptr);
  OP_TILING_CHECK(!parsed_compile_info["_pattern"].is_string(),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "compile info[_pattern] not is string"),
                  return nullptr);
  const string& pattern = parsed_compile_info["_pattern"];
  std::shared_ptr<void> compile_info;
  OP_LOGI(op_type.c_str(), "Entering AutoTiling Compile Info Parser for pattern %s", pattern.c_str());
  try {
    if (pattern == "CommReduce") {
      compile_info = std::make_shared<nlohmann::json>(parsed_compile_info);
    } else if (pattern == "ElemWise") {
      compile_info = std::make_shared<nlohmann::json>(parsed_compile_info);
    } else if (pattern == "Broadcast") {
      compile_info = std::make_shared<nlohmann::json>(parsed_compile_info);
    } else if (pattern == "Norm") {
      compile_info = std::make_shared<nlohmann::json>(parsed_compile_info);
    } else if (pattern == "Transpose") {
      compile_info = std::make_shared<nlohmann::json>(parsed_compile_info);
    } else {
      OP_LOGE(op_type.c_str(), "Pattern %s is not supported by AutoTiling Compile Info Parser", pattern.c_str());
      return nullptr;
    }
  } catch(...) {
    OP_LOGE(op_type.c_str(), "Unknown Exception encountered when parsing Compile Info of pattern %s", pattern.c_str());
    return nullptr;
  }
  return new AutoTilingCompileInfo(op_type, pattern, compile_info);
}

// register AutoTiling
REGISTER_OP_TILING_V3(AutoTiling, optiling::AutoTiling, optiling::AutoTilingCompileInfoParser);

}  // namespace optiling
