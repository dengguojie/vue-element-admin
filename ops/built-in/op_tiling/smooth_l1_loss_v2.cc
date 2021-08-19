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

#include <iostream>
namespace optiling {
bool SmoothL1LossV2Tiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                          OpRunInfo& run_info) {
  bool ret = false;
  std::vector<int64_t> input_shape = op_paras.inputs[0].tensor[0].shape;
  float reduce_mean_cof = 1.0;
  std::string reduction, reduce_mean_cof_dtype;
  
  if (op_info.count("reduction") > 0 && op_info.count("reduce_mean_cof_dtype") > 0) {
    reduction = op_info.at("reduction").get<std::string>();
    reduce_mean_cof_dtype = op_info.at("reduce_mean_cof_dtype").get<std::string>();
  } else {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Compile_info[reduction] or Compile_info[reduce_mean_cof_dtype] not exist.");
    return false;
  }

  if (reduction == "sum" || reduction == "mean") {
    Reduce reduce(op_type, op_paras, op_info, run_info);
    ret = reduce.DoTiling() && reduce.WriteTilingData();
    if (reduction == "mean" && reduce_mean_cof_dtype == "float32") {
      for (uint32_t i = 0; i < input_shape.size(); i++) {
        reduce_mean_cof = reduce_mean_cof / input_shape[i];
      }
      ByteBufferPut(run_info.tiling_data, (float)reduce_mean_cof);
      OP_LOGD(op_type.c_str(), "reduce mean cof:%f", reduce_mean_cof);
    } else if (reduction == "mean" && reduce_mean_cof_dtype == "float16") {
      for (uint32_t i = 0; i < input_shape.size(); i++) {
        reduce_mean_cof = reduce_mean_cof / input_shape[i];
      }
      fe::fp16_t reduce_mean_cof_fp16;
      reduce_mean_cof_fp16 = reduce_mean_cof;
      ByteBufferPut(run_info.tiling_data, (fe::fp16_t)reduce_mean_cof_fp16);
      ByteBufferPut(run_info.tiling_data, (uint16_t)0);
      OP_LOGD(op_type.c_str(), "reduce mean cof:%f", reduce_mean_cof);
    }
  } else if (reduction == "none"){
    ret = EletwiseTiling(op_type, op_paras, op_info, run_info);
  }
  return ret;
}

REGISTER_OP_TILING_FUNC_BUFFERED(SmoothL1LossV2, SmoothL1LossV2Tiling);
}  // namespace optiling