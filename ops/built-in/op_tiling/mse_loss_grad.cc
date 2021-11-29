/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.

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

namespace optiling {
bool MseLossGradTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                       OpRunInfo& run_info) {
  bool ret = EletwiseTiling(op_type, op_paras, op_info, run_info);
  std::vector<int64_t> input_shape = op_paras.inputs[0].tensor[0].shape;
  // reduce_mean_cof is not required when handling pure dma_copy case
  if (input_shape[0] == 1) {
    return ret;
  }

  if (op_info.count("reduce_mean_cof_dtype") > 0) {
    const std::string& reduce_mean_cof_dtype = op_info.at("reduce_mean_cof_dtype").get<std::string>();
    float reduce_mean_cof = 1.0;
    if (reduce_mean_cof_dtype == "float32") {
      for (uint32_t i = 0; i < input_shape.size(); i++) {
        if (input_shape[i] == 0) {
          VECTOR_INNER_ERR_REPORT_TILIING("mes_loss_grad", "inputshape = 0 is not support");
          return false;
        }

        reduce_mean_cof = reduce_mean_cof / input_shape[i];
      }
      reduce_mean_cof = 2.0 * reduce_mean_cof;
      ByteBufferPut(run_info.tiling_data, (float)reduce_mean_cof);
      OP_LOGD(op_type.c_str(), "reduce mean cof:%f", reduce_mean_cof);
    } else if (reduce_mean_cof_dtype == "float16") {
      for (uint32_t i = 0; i < input_shape.size(); i++) {
        if(input_shape[i] == 0){
          VECTOR_INNER_ERR_REPORT_TILIING("mse_loss_grad", "inputshape = 0 is not support");
          return false;
        }
          reduce_mean_cof = reduce_mean_cof / input_shape[i];
      }
      reduce_mean_cof = 2.0 * reduce_mean_cof;
      fe::fp16_t reduce_mean_cof_fp16 = reduce_mean_cof;
      ByteBufferPut(run_info.tiling_data, (fe::fp16_t)reduce_mean_cof_fp16);
      ByteBufferPut(run_info.tiling_data, (uint16_t)0);
      OP_LOGD(op_type.c_str(), "reduce mean cof:%f", reduce_mean_cof);
    }
  }

  return ret;
}

REGISTER_OP_TILING_FUNC_BUFFERED(MseLossGrad, MseLossGradTiling);
}  // namespace optiling
