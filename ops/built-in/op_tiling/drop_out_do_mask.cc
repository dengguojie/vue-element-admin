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
 * \file drop_out_do_mask.cpp
 * \brief
 */
#include <string>
#include <math.h>
#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "error_log.h"

// the minest num for one core
const int64_t CORE_MINEST_NUM = 128;

namespace optiling {

bool GetDropOutDoMaskCompileParams(const nlohmann::json& opCompileInfo, int64_t& coreNum) {
  using namespace nlohmann;
  auto allVars = opCompileInfo["vars"];
  if (allVars.count("core_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("DropOutDoMaskTiling", "GetCompileParams, get core_num error");
    return false;
  }
  coreNum = allVars["core_num"].get<std::int64_t>();

  return true;
}

void SetRuningParams(const int64_t& core_used_num,
                     const int64_t& num_per_core,
                     const int64_t& num_tail_core,
                     OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, core_used_num);
  ByteBufferPut(runInfo.tiling_data, num_per_core);
  ByteBufferPut(runInfo.tiling_data, num_tail_core);
}

bool DropOutDoMaskTiling(const std::string& op_type,
                         const TeOpParas& op_paras,
                         const nlohmann::json& op_info,
                         OpRunInfo& run_info) {
  using namespace ge;
  if (op_paras.inputs.size() != 3 || op_paras.inputs[0].tensor.empty() || op_paras.inputs[1].tensor.empty() ||
      op_paras.inputs[2].tensor.empty()) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "inputs size is not 3 or some input are empty");
      return false;
  }
  const std::vector<int64_t>& var_shape = op_paras.inputs[0].tensor[0].shape;
  std::string input_dtype = op_paras.inputs[0].tensor[0].dtype;

  int64_t var_num = var_shape.size() == 0 ? 1 : std::accumulate(var_shape.begin(), var_shape.end(),
                                                                1, std::multiplies<int64_t>());

  int64_t core_num = 0;
  if (!GetDropOutDoMaskCompileParams(op_info, core_num)) {
    VECTOR_INNER_ERR_REPORT_TILIING("DropOutDoMaskTiling", "GetCompileParams, get core_num error");
    return false;
  }

  int64_t sigment_total = (var_num + CORE_MINEST_NUM - 1) / CORE_MINEST_NUM;
  int64_t sigment_per_core = (sigment_total + core_num - 1) / core_num;
  int64_t core_used_num = sigment_per_core == 0 ? 1 : (sigment_total + sigment_per_core - 1) / sigment_per_core;
  int64_t num_per_core = sigment_per_core * CORE_MINEST_NUM;
  int64_t num_tail_core = var_num - (num_per_core * (core_used_num - 1));
  GELOGD("op [DropOutDoMaskTiling] : CompileParams, core_used_num = %d.", core_used_num);
  GELOGD("op [DropOutDoMaskTiling] : CompileParams, num_per_core = %d.", num_per_core);
  GELOGD("op [DropOutDoMaskTiling] : CompileParams, num_tail_core = %d.", num_tail_core);
  SetRuningParams(core_used_num, num_per_core, num_tail_core, run_info);

  run_info.block_dim = core_num;
  std::vector<int64_t> workspace;
  run_info.workspaces = workspace;

  GELOGI("op[%s] tiling run success.", op_type.c_str());

  return true;
}

// register tiling interface of the DropOutDoMask op.
REGISTER_OP_TILING_FUNC_BUFFERED(DropOutDoMask, DropOutDoMaskTiling);
}  // namespace optiling

