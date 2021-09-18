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
 * \file assign.cc
 * \brief
 */
#include <string>
#include <math.h>
#include <algorithm>
#include <nlohmann/json.hpp>
#include "op_tiling_util.h"

#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "error_log.h"
#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"

// the minimum num for one core
const int64_t CORE_MINIMUM_NUM = 4;
const int64_t BLOCK_SIZE = 32;  // one block size is 32Bytes

using namespace ge;
namespace optiling {
bool GetAssignCompileParams(const nlohmann::json& op_compile_info, int64_t& core_num, int64_t& ub_size) {
  using namespace nlohmann;
  const nlohmann::json& all_vars = op_compile_info["vars"];
  if (all_vars.count("core_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("Assign", "GetAssignCompileParams, get core_num error");
    return false;
  }
  core_num = all_vars["core_num"].get<std::int64_t>();

  if (all_vars.count("ub_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("Assign", "GetAssignCompileParams, get ub_size error");
    return false;
  }
  ub_size = all_vars["ub_size"].get<std::int64_t>();

  return true;
}

bool AssignTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                  utils::OpRunInfo& run_info) {
  using namespace ge;
  PROFILING_TILING_INIT(op_type.c_str());
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get op_info failed."), return false);

  auto value_desc = operator_info->MutableInputDesc(1);
  OP_TILING_CHECK(value_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get value_desc failed."), return false);

  ge::GeShape value_shape = value_desc->MutableShape();
  ge::DataType input_dtype = value_desc->GetDataType();
  PROFILING_TILING_AFTER_GET_SHAPE_REG();

  int64_t value_num = value_shape.GetShapeSize();

  int64_t core_num = 0;
  int64_t ub_size = 0;
  if (!GetAssignCompileParams(op_info, core_num, ub_size)) {
    return false;
  }
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  int64_t ele_size = GetSizeByDataType(input_dtype);

  int64_t ele_per_block = BLOCK_SIZE / ele_size;
  int64_t block_count = (value_num + ele_per_block - 1) / ele_per_block;

  int64_t sigment_total = (block_count + CORE_MINIMUM_NUM - 1) / CORE_MINIMUM_NUM;
  int64_t sigment_per_core = (sigment_total + core_num - 1) / core_num;

  int64_t core_used_num = sigment_per_core == 0 ? 1 : (sigment_total + sigment_per_core - 1) / sigment_per_core;
  int64_t block_per_core = sigment_per_core * CORE_MINIMUM_NUM;
  int64_t block_tail_core = block_count - (block_per_core * (core_used_num - 1));

  OP_LOGD(op_type.c_str(), "CompileParams, core_used_num = %d, block_per_core = %d, block_tail_core = %d",
          core_used_num, block_per_core, block_tail_core);
  PROFILING_TILING_AFTER_CALCU_TILING_REG();

  run_info.AddTilingData(core_used_num);
  run_info.AddTilingData(block_per_core);
  run_info.AddTilingData(block_tail_core);

  run_info.SetBlockDim(core_num);
  PROFILING_TILING_END();
  OP_LOGI(op_type.c_str(), "tiling run success.");
  return true;
}

// register tiling interface of the Assign op.
REGISTER_OP_TILING_FUNC_BUFFERED_V2(Assign, AssignTiling);
}  // namespace optiling
