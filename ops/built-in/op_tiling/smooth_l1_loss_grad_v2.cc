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

#include "reduce_tiling.h"
#include "eletwise.h"
#include "../fusion_pass/common/fp16_t.hpp"

namespace optiling {
bool SmoothL1LossGradV2Tiling(const std::string& op_type,
                              const TeOpParas& op_paras,
                              const nlohmann::json& op_info,
                              OpRunInfo& run_info) {
  bool ret = EletwiseTiling(op_type, op_paras, op_info, run_info);
  std::vector<int64_t> input_shape = op_paras.inputs[0].tensor[0].shape;
  if (op_info.count("reduce_mean_cof_dtype") > 0) {
    const std::string& reduce_mean_cof_dtype = op_info.at("reduce_mean_cof_dtype").get<std::string>();
    float reduce_mean_cof = 1.0;
    if (reduce_mean_cof_dtype == "float32") {
      for (uint32_t i = 0; i < input_shape.size(); i++) {
        reduce_mean_cof = reduce_mean_cof / input_shape[i];
      }

      ByteBufferPut(run_info.tiling_data, (float)reduce_mean_cof);
      OP_LOGD(op_type.c_str(), "reduce mean cof:%f", reduce_mean_cof);
    } else if (reduce_mean_cof_dtype == "float16") {
      for (uint32_t i = 0; i < input_shape.size(); i++) {
        reduce_mean_cof = reduce_mean_cof / input_shape[i];
      }

      fe::fp16_t reduce_mean_cof_fp16 = reduce_mean_cof;
      ByteBufferPut(run_info.tiling_data, (fe::fp16_t)reduce_mean_cof_fp16);
      ByteBufferPut(run_info.tiling_data, (uint16_t)0);
      OP_LOGD(op_type.c_str(), "reduce mean cof:%f", reduce_mean_cof);
    }
  }

  return ret;
}

REGISTER_OP_TILING_FUNC_BUFFERED(SmoothL1LossGradV2, SmoothL1LossGradV2Tiling);
}  // namespace optiling
