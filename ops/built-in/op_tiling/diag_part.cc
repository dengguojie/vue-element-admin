/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file diag_part.cc
 * \brief
 */
#include <nlohmann/json.hpp>
#include "graph/debug/ge_log.h"
#include "../op_proto/util/error_util.h"
#include "graph/utils/op_desc_utils.h"
#include "error_log.h"
#include "op_log.h"
#include "op_tiling.h"
#include <math.h>

namespace optiling {
struct TilingParam {
  // input x line num
  int64_t input_num = 0;
  // use aicore num
  int64_t act_core_num = 0;
  // each aicore need compute num except last aicore
  int64_t one_core_num = 0;
  // last aicore need compute num
  int64_t last_core_num = 0;
};

void DiagPartInItRunningParams(TilingParam& param) {
  param.input_num = 0;
  param.act_core_num = 0;
  param.one_core_num = 0;
  param.last_core_num = 0;
}

static void PrintTilingParam(const TilingParam& param) {
  OP_LOGD("DiagPartTiling", "(input_num, act_core_num, one_core_num, last_core_num):(%d, %d, %d, %d)", param.input_num,
          param.act_core_num, param.one_core_num, param.last_core_num);
}

static void CalTilingParam(TilingParam& param, int64_t input_num, int64_t aicore_num) {
  param.input_num = sqrt(input_num);
  int num_per_core = (param.input_num + aicore_num - 1) / aicore_num;
  num_per_core = max(1, num_per_core);
  num_per_core = ((num_per_core + 128 - 1) / 128) * 128;
  param.act_core_num = (param.input_num + num_per_core - 1) / num_per_core;
  param.one_core_num = num_per_core;
  param.last_core_num = param.input_num - (param.act_core_num - 1) * num_per_core;
}

static void SetTilingParam(const TilingParam& param, utils::OpRunInfo& run_info) {
  run_info.AddTilingData(param.input_num);
  run_info.AddTilingData(param.act_core_num);
  run_info.AddTilingData(param.one_core_num);
  run_info.AddTilingData(param.last_core_num);
}

bool DiagPartGetCompileInfo(const nlohmann::json& op_compile_info, int64_t& core_num) {
  auto all_vars = op_compile_info["vars"];
  if (all_vars.count("core_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("DiagPart", "Get core_num failed");
    return false;
  }
  core_num = all_vars["core_num"].get<std::int64_t>();
  return true;
}

bool DiagPartTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                utils::OpRunInfo& run_info) {
  OP_LOGI(op_type.c_str(), "DIAGPARTTiling running");

  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  vector<int64_t> input_shape = operator_info->MutableInputDesc(0)->MutableShape().GetDims();

  int64_t value_num = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int64_t>());

  int64_t core_num = 0;
  if (!DiagPartGetCompileInfo(op_info, core_num)) {
    return false;
  }

  const map<string, int64_t&> compile_params = {{"core_num", core_num}};
  for (auto& param : compile_params) {
    const auto& name = param.first;
    OP_LOGD(op_type.c_str(), "DiagPartGetCompileInfo %s", name.c_str());
  }

  TilingParam param;
  DiagPartInItRunningParams(param);

  CalTilingParam(param, value_num, core_num);
  SetTilingParam(param, run_info);
  PrintTilingParam(param);

  run_info.SetBlockDim(param.act_core_num);
  OP_LOGI(op_type.c_str(), "DIAGPARTTiling run success.");
  return true;
}
// register tiling interface of the DropOutDoMask op.
REGISTER_OP_TILING_FUNC_BUFFERED_V2(DiagPart, DiagPartTiling);
}  // namespace optiling
