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
#include <string>

#include <nlohmann/json.hpp>
#include "graph/debug/ge_log.h"
#include "op_tiling.h"
#include "error_log.h"
#include "vector_tiling.h"
#include "op_log.h"

namespace optiling {
/*
 * @brief: tiling function of ops
 * @param [in] op_type: opType of ops
 * @param [in] op_paras: inputs/outputs/attrs of ops
 * @param [in] op_info: compile time generated info of ops
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool AutoTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                utils::OpRunInfo& run_info) {
  GELOGI("op[%s] tiling running.", op_type.c_str());

  OP_TILING_CHECK((op_info.find("_pattern") == op_info.end()),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "compile info not contain [_pattern]"), return false);
  OP_TILING_CHECK(!op_info["_pattern"].is_string(),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "compile info[_pattern] not is string"), return false);
  const std::string& pattern = op_info["_pattern"];

  GELOGI("op[%s] tiling pattern.", pattern.c_str());

  bool ret = false;
  if (pattern == "CommReduce") {
    ret = ReduceTiling(op_type, op_paras, op_info, run_info);
  } else if (pattern == "ElemWise" || pattern == "Broadcast") {
    ret = EletwiseTiling(op_type, op_paras, op_info, run_info);
  } else if (pattern == "Norm") {
    ret = NormTiling(op_type, op_paras, op_info, run_info);
  } else {
    ret = false;
    GELOGI("Auto tiling not supported patten: %s.", pattern.c_str());
  }

  return ret;
}

// register autoTiling
REGISTER_OP_TILING_FUNC_BUFFERED_V2(AutoTiling, AutoTiling);
}  // namespace optiling
