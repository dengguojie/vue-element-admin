/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#include "op_tiling_util.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "error_log.h"
#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"

// the minest num for one core
const int64_t CORE_MINEST_NUM = 128;
static const std::vector<std::string> COMPILE_INFO_KEY = {"core_num"};

namespace optiling {
bool GetDropOutDoMaskCompileParams(const std::string& op_type, const std::vector<int64_t>& op_info, int64_t& coreNum) {
  OP_TILING_CHECK(
      op_info.size() != COMPILE_INFO_KEY.size(),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "the compile info num is not equal expect compile_info(%zu), is %zu",
                                      COMPILE_INFO_KEY.size(), op_info.size()),
      return false);
  coreNum = op_info[0];

  return true;
}

void SetRuningParams(const int64_t& core_used_num, const int64_t& num_per_core, const int64_t& num_tail_core,
                     utils::OpRunInfo& runInfo) {
  runInfo.AddTilingData(core_used_num);
  runInfo.AddTilingData(num_per_core);
  runInfo.AddTilingData(num_tail_core);
}

bool DropOutDoMaskTiling(const std::string& op_type, const ge::Operator& op_paras, const std::vector<int64_t>& op_info,
                         utils::OpRunInfo& run_info) {
  using namespace ge;
  PROFILING_TILING_INIT(op_type.c_str());

  auto operator_info = OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get op_info failed."),
                  return false);

  auto var_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(var_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get var_desc failed."), return false);

  const GeShape& var_shape = var_desc->MutableShape();
  PROFILING_TILING_AFTER_GET_SHAPE_REG();
  int64_t var_num = var_shape.GetDimNum() == 0 ? 1 : GetTensorSize(var_shape);
  int64_t core_num = 0;
  if (!GetDropOutDoMaskCompileParams(op_type, op_info, core_num)) {
    VECTOR_INNER_ERR_REPORT_TILIING("DropOutDoMaskTiling", "GetCompileParams, get core_num error");
    return false;
  }
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  int64_t sigment_total = (var_num + CORE_MINEST_NUM - 1) / CORE_MINEST_NUM;
  int64_t sigment_per_core = (sigment_total + core_num - 1) / core_num;
  int64_t core_used_num = sigment_per_core == 0 ? 1 : (sigment_total + sigment_per_core - 1) / sigment_per_core;
  int64_t num_per_core = sigment_per_core * CORE_MINEST_NUM;
  int64_t num_tail_core = var_num - (num_per_core * (core_used_num - 1));
  PROFILING_TILING_AFTER_CALCU_TILING_REG();

  GELOGD("op [DropOutDoMaskTiling] : CompileParams, core_used_num = %d.", core_used_num);
  GELOGD("op [DropOutDoMaskTiling] : CompileParams, num_per_core = %d.", num_per_core);
  GELOGD("op [DropOutDoMaskTiling] : CompileParams, num_tail_core = %d.", num_tail_core);
  SetRuningParams(core_used_num, num_per_core, num_tail_core, run_info);

  run_info.SetBlockDim(core_num);
  PROFILING_TILING_END();
  GELOGI("op[%s] tiling run success.", op_type.c_str());

  return true;
}

// register tiling interface of the DropOutDoMask op.
REGISTER_OP_TILING_V3_WITH_VECTOR(DropOutDoMask, DropOutDoMaskTiling, COMPILE_INFO_KEY, NO_OPTIONAL_VALUE);
}  // namespace optiling
