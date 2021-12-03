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

#include <iostream>
#include "../fusion_pass/common/fp16_t.hpp"
#include "eletwise.h"
#include "vector_tiling.h"
#include "op_tiling_util.h"

namespace optiling {

struct LpLossCompileInfo {
  std::shared_ptr<AutoTilingHandler> tiling_handler;
  std::vector<int32_t> reduce_axis;
  ge::DataType dtype;
  bool reduction_is_none;
};

bool LpLossIsInAxis(std::vector<int32_t>& input, int32_t value) {
  for (uint32_t i = 0; i < input.size(); i++) {
    if (input[i] == value) {
      return true;
    }
  }
  return false;
}

bool LpLossTiling(const std::string& op_type, const ge::Operator& op_paras,
                              const LpLossCompileInfo& parsed_info, utils::OpRunInfo& run_info) {
  PROFILING_TILING_INIT(op_type.c_str());
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "parsed_info.tiling_handler nullptr, error!"),
                  return false);
  if (parsed_info.reduction_is_none) {
    bool ret = parsed_info.tiling_handler->DoTiling(op_paras, run_info);
    return ret;
  }
  PROFILING_TILING_AFTER_GET_SHAPE_REG();
  bool ret = parsed_info.tiling_handler->DoTiling(op_paras, run_info);
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get op_info failed."),
                  return false);

  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_desc failed."),
                  return false);

  const auto& input_shape = input_desc->MutableShape();

  std::vector<int32_t> reduce_axis = parsed_info.reduce_axis;
  int32_t max_value = static_cast<int32_t>(input_shape.GetDimNum());
  int32_t min_value = -1 * max_value;
  for (size_t i = 0; i < reduce_axis.size(); i++) {
    if (reduce_axis[i] >= max_value || reduce_axis[i] < min_value) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "value of axis is illegal.");
      return false;
    }
    if (reduce_axis[i] < 0) {
      reduce_axis[i] = max_value + reduce_axis[i];
    }
  }
  PROFILING_TILING_AFTER_CALCU_TILING_REG();
  // reduce_mean_cof is not required when handling pure dma_copy case
  if (input_shape.GetDim(0) == 1 && reduce_axis[0] == 0) {
    return ret;
  }

  float reduce_mean_cof = 1.0;
  for (uint32_t i = 0; i < input_shape.GetDimNum(); i++) {
    if (LpLossIsInAxis(reduce_axis, i)) {
      OP_TILING_CHECK(input_shape.GetDim(i) == 0, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input_shape cannot include 0."),
                      return false);
      reduce_mean_cof = reduce_mean_cof / input_shape.GetDim(i);
    }
  }
  if (parsed_info.dtype == DT_FLOAT) {
    run_info.AddTilingData((float)reduce_mean_cof);
  } else if (parsed_info.dtype == DT_FLOAT16) {
    fe::fp16_t reduce_mean_cof_fp16 = reduce_mean_cof;
    run_info.AddTilingData(reduce_mean_cof_fp16);
    run_info.AddTilingData((uint16_t)0);
  }
  OP_LOGD(op_type.c_str(), "reduce mean cof:%f", reduce_mean_cof);
  PROFILING_TILING_END();

  return ret;
}

static bool ParseJsonCompileInfo(const std::string& op_type, const nlohmann::json& compile_info,
                                 LpLossCompileInfo& parsed_info) {
  parsed_info.reduction_is_none = false;
  std::string reduction;
  OP_TILING_CHECK(!GetCompileValue(compile_info, "reduction", reduction),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ParseJsonCompileInfo, get reduction error"), return false);
  if (reduction == "none") {
    parsed_info.reduction_is_none = true;
    parsed_info.tiling_handler = CreateAutoTilingHandler(op_type, PATTERN_ELEMWISE, compile_info);
    OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CreateAutoTilingHandler return nullptr"),
                    return false);
  } else {
    parsed_info.tiling_handler = CreateAutoTilingHandler(op_type, PATTERN_REDUCE, compile_info);
    OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CreateAutoTilingHandler return nullptr"),
                    return false);
    OP_TILING_CHECK(!GetCompileValue(compile_info, "_ori_axis", parsed_info.reduce_axis),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ParseJsonCompileInfo, get _ori_axis error"),
                    return false);
    std::string dtype;
    OP_TILING_CHECK(!GetCompileValue(compile_info, "reduce_mean_cof_dtype", dtype),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ParseJsonCompileInfo get reduce_mean_cof_dtype error"),
                    return false);
    parsed_info.dtype = (dtype == "float32") ? ge::DT_FLOAT : ge::DT_FLOAT16;
  }

  return true;
}

REGISTER_OP_TILING_V3_CUSTOM(LpLoss, LpLossTiling, ParseJsonCompileInfo, LpLossCompileInfo);
}  // namespace optiling