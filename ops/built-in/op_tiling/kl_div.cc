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

#include "../fusion_pass/common/fp16_t.hpp"
#include "vector_tiling.h"
#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"
#include "op_tiling_util.h"

#include "error_log.h"
namespace optiling {
struct KLDivCompileInfo {
  std::shared_ptr<AutoTilingHandler> tiling_handler;
  ge::DataType reduce_mean_cof_dtype;
  bool has_reduce_mean_cof_dtype;
};

bool KLDivTiling(const std::string& op_type, const ge::Operator& op_paras,
                 const KLDivCompileInfo& parsed_info, utils::OpRunInfo& run_info) {
  OP_LOGD(op_type, "Enter KLDivTiling");
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "parsed_info.tiling_handler nullptr, error!"),
                  return false);
  bool ret = parsed_info.tiling_handler->DoTiling(op_paras, run_info);
  PROFILING_TILING_INIT(op_type.c_str());
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetOpDescFromOperator return nullptr!"), return false);
  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input 0 opdesc failed"),
                  return false);
  auto dim0 = input_desc->MutableShape().GetDim(0);
  if (dim0 == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "The dim0 cannot be zero!");
    return false;
  }
  if (parsed_info.has_reduce_mean_cof_dtype) {
    float reduce_mean_cof = 1.0;
    if (parsed_info.reduce_mean_cof_dtype == ge::DT_FLOAT) {
      reduce_mean_cof = reduce_mean_cof / dim0;
      run_info.AddTilingData(reduce_mean_cof);
    } else if (parsed_info.reduce_mean_cof_dtype == ge::DT_FLOAT16) {
      reduce_mean_cof = reduce_mean_cof / dim0;
      fe::fp16_t reduce_mean_cof_fp16 = reduce_mean_cof;
      run_info.AddTilingData(reduce_mean_cof_fp16);
      run_info.AddTilingData((uint16_t)0);
    }
    OP_LOGD(op_type, "The value of reduce_mean_cof is: %f", reduce_mean_cof);
  }
  PROFILING_TILING_AFTER_GET_SHAPE_REG();
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();
  PROFILING_TILING_AFTER_CALCU_TILING_REG();
  PROFILING_TILING_END();

  OP_LOGD(op_type, "Exit KLDivTiling");
  return ret;
}

static bool ParseJsonCompileInfo(const std::string& op_type, const nlohmann::json& compile_info,
                                 KLDivCompileInfo& parsed_info) {
  parsed_info.tiling_handler = CreateAutoTilingHandler(op_type, PATTERN_REDUCE, compile_info);
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CreateAutoTilingHandler return nullptr"), return false);
  std::string dtype;
  parsed_info.has_reduce_mean_cof_dtype = false;
  
  if (GetCompileValue(compile_info, "reduce_mean_cof_dtype", dtype)) {
    parsed_info.has_reduce_mean_cof_dtype = true;
    parsed_info.reduce_mean_cof_dtype = (dtype == "float16") ? ge::DT_FLOAT16 : ge::DT_FLOAT;
  }
  return true;
}

REGISTER_OP_TILING_V3_CUSTOM(KLDiv, KLDivTiling, ParseJsonCompileInfo, KLDivCompileInfo);
}  // namespace optiling
