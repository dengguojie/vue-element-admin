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
 * \file auto_tiling.cpp
 * \brief Auto-Tiling Private Declarations and Implementations
 */
#include "vector_tiling.h"
#include "auto_tiling.h"

namespace optiling {

static thread_local std::vector<std::shared_ptr<AutoTilingHandler>> handler_container;

/*
 * @brief: tiling function of ops
 * @param [in] op_type: opType of ops
 * @param [in] op_paras: inputs/outputs/attrs of ops
 * @param [in] compile_info: compile time generated info of ops
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool AutoTiling(const ge::Operator& op_paras, const void* handler,
                utils::OpRunInfo& run_info) {
  OP_LOGI("AutoTiling", "Entering AutoTiling Dispatcher.");
  if (handler == nullptr) {
    OP_LOGE("AutoTiling", "AutoTiling Dispatcher received nullptr, CompileInfo Parser is not working properly.");
    return false;
  }
  const AutoTilingHandler& _handler = *(static_cast<const AutoTilingHandler*>(handler));
#ifdef ASCEND_OPTILING_UT
  return true;
#else
  return _handler.DoTiling(op_paras, run_info);
#endif
}

void* AutoTilingHandlerParser(const ge::Operator& op_paras, const ge::AscendString& compile_info_str) {
  // Print Info Log and get Pattern+OpType
  OP_LOGI("AutoTiling", "Entering AutoTilingHandler Parser.");
  ge::AscendString ascend_op_type;
  ge::graphStatus ret = op_paras.GetOpType(ascend_op_type);
  if (ret != ge::GRAPH_SUCCESS) {
    OP_LOGE("UNKNOWN_OP_TYPE", "get op type failed");
    return nullptr;
  }
  const std::string& op_type = ascend_op_type.GetString();

  // Default parsing sequence, parse compile info directly to nlohmann::json object
  const nlohmann::json& parsed_compile_info = nlohmann::json::parse(compile_info_str.GetString());
  OP_TILING_CHECK((parsed_compile_info.find("_pattern") == parsed_compile_info.end()),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "compile info not contain [_pattern]"),
                  return nullptr);
  OP_TILING_CHECK(!parsed_compile_info["_pattern"].is_string(),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "compile info[_pattern] not is string"),
                  return nullptr);
  const string& pattern = parsed_compile_info["_pattern"];
  std::shared_ptr<AutoTilingHandler> p_handler = CreateAutoTilingHandler(op_type, pattern,
                                                                         parsed_compile_info);
  handler_container.push_back(p_handler);
  return p_handler.get();
}

// register AutoTiling
REGISTER_OP_TILING_V3(AutoTiling, AutoTiling, AutoTilingHandlerParser);

}  // namespace optiling
