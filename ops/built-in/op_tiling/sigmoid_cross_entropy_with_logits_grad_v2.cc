/*
 * Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include "reduce_tiling.h"
#include "eletwise.h"
#include "../fusion_pass/common/fp16_t.hpp"
#include "error_log.h"
#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"

namespace optiling {

bool SigmoidCrossEntropyWithLogitsGradV2Tiling(const std::string& op_type, const ge::Operator& op_paras,
                                               const nlohmann::json& op_info, utils::OpRunInfo& run_info) {
  bool ret = EletwiseTiling(op_type, op_paras, op_info, run_info);
  PROFILING_TILING_INIT(op_type.c_str());
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetOpDescFromOperator return nullptr!"), return false);
  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input 0 opdesc failed"), return false);
  std::vector<int64_t> input_shape = input_desc->MutableShape().GetDims();
  // reduce_mean_cof is not required when handling pure dma_copy case
  if (input_shape.empty() || input_shape[0] == 1) {
    return ret;
  }

  if (op_info.count("reduce_mean_cof_dtype") > 0) {
    const std::string& reduce_mean_cof_dtype = op_info.at("reduce_mean_cof_dtype").get<std::string>();
    if (reduce_mean_cof_dtype == "float32") {
      float reduce_mean_cof = 1.0;
      for (uint32_t i = 0; i < input_shape.size(); i++) {
        if (input_shape[i] != 0) {
          reduce_mean_cof = reduce_mean_cof / input_shape[i];
        } else {
          VECTOR_INNER_ERR_REPORT_TILIING(op_type, "the shape[%d] is 0,do not supported", i);
          return false;
        }
      }
      run_info.AddTilingData(reduce_mean_cof);
      OP_LOGD(op_type.c_str(), "reduce mean cof:%f", reduce_mean_cof);
    } else if (reduce_mean_cof_dtype == "float16") {
      float reduce_mean_cof = 1.0;
      for (uint32_t i = 0; i < input_shape.size(); i++) {
        if (input_shape[i] != 0) {
          reduce_mean_cof = reduce_mean_cof / input_shape[i];
        } else {
          VECTOR_INNER_ERR_REPORT_TILIING(op_type, "the shape[%d] is 0,do not supported", i);
          return false;
        }
      }
      fe::fp16_t reduce_mean_cof_fp16 = reduce_mean_cof;
      run_info.AddTilingData((fe::fp16_t)reduce_mean_cof_fp16);
      run_info.AddTilingData((uint16_t)0);
      OP_LOGD(op_type.c_str(), "reduce mean cof:%f", reduce_mean_cof);
    }
  }
  PROFILING_TILING_AFTER_GET_SHAPE_REG();
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();
  PROFILING_TILING_AFTER_CALCU_TILING_REG();
  PROFILING_TILING_END();

  return ret;
}

REGISTER_OP_TILING_FUNC_BUFFERED_V2(SigmoidCrossEntropyWithLogitsGradV2, SigmoidCrossEntropyWithLogitsGradV2Tiling);
}  // namespace optiling
