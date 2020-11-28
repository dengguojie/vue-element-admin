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
bool AutoTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                OpRunInfo& run_info) {
  GELOGI("op[%s] tiling running.", op_type.c_str());

  const std::string& pattern = op_info["_pattern"].get<std::string>();

  GELOGI("op[%s] tiling pattern.", pattern.c_str());

  bool ret = false;
  if (pattern == "CommReduce") {
    ret = ReduceTiling(op_type, op_paras, op_info, run_info);
  } else if (pattern == "ElemWise") {
    ret = EletwiseTiling(op_type, op_paras, op_info, run_info);
  } else {
    ret = false;
    GELOGI("Auto tiling not supported patten: %s.", pattern.c_str());
  }

  return ret;
}

// register autoTiling
REGISTER_OP_TILING_FUNC_BUFFERED(AutoTiling, AutoTiling);

}  // namespace optiling
