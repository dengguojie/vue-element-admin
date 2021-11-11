/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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

#include <sstream>
#include <cctype>
#include <iostream>

#include "reduce_tiling.h"
#include "eletwise.h"
#include "../fusion_pass/common/fp16_t.hpp"

namespace optiling {
bool IsInAxisReduceStdWithMean(std::vector<int32_t>& input, int32_t value) {
  for (uint32_t i = 0; i < input.size(); i++) {
    if (input[i] == value) {
      return true;
    }
  }
  return false;
}

bool ReduceStdWithMeanTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                             OpRunInfo& run_info) {
  Reduce reduce(op_type, op_paras, op_info, run_info);
  bool ret = reduce.DoTiling();
  ret = ret && reduce.WriteTilingData();

  std::vector<int64_t> input_shape = reduce.GetInputShape();
  std::vector<int32_t> reduce_axis = reduce.GetReduceAxis();
  // reduce_mean_cof is not required when handling pure dma_copy case
  if (input_shape[0] == 1 && reduce_axis[0] == 0) {
    return ret;
  }

  const std::string& attr_unbiased = op_info.at("attr_unbiased").get<std::string>();

  if (op_info.count("reduce_mean_cof_dtype") > 0) {
    float reduce_mean_cof = 1.0;
    float input_shape_mul = 1.0;
    const std::string& reduce_mean_cof_dtype = op_info.at("reduce_mean_cof_dtype").get<std::string>();
    for (uint32_t i = 0; i < input_shape.size(); i++) {
      if (IsInAxisReduceStdWithMean(reduce_axis, i)) {
        input_shape_mul = input_shape_mul * input_shape[i];
      }
    }
    if (reduce_mean_cof_dtype == "float32") {
      if (attr_unbiased == "true") {
        reduce_mean_cof = reduce_mean_cof / (input_shape_mul - 1);
      } else {
        reduce_mean_cof = reduce_mean_cof / (input_shape_mul);
      }
      ByteBufferPut(run_info.tiling_data, (float)reduce_mean_cof);
      OP_LOGD(op_type.c_str(), "reduce mean cof:%f", reduce_mean_cof);
    } else if (reduce_mean_cof_dtype == "float16") {
        if (attr_unbiased == "true") {
          reduce_mean_cof = reduce_mean_cof / (input_shape_mul - 1);
        } else {
          reduce_mean_cof = reduce_mean_cof / (input_shape_mul);
        }
      std::cout << "tiling std reduce_mean_cof is start " << std::endl;
      std::cout << reduce_mean_cof << std::endl;
      std::cout << "tiling std reduce_mean_cof is end " << std::endl;
      fe::fp16_t reduce_mean_cof_fp16 = reduce_mean_cof;
      ByteBufferPut(run_info.tiling_data, (fe::fp16_t)reduce_mean_cof_fp16);
      ByteBufferPut(run_info.tiling_data, (uint16_t)0);
      OP_LOGD(op_type.c_str(), "reduce mean cof:%f", reduce_mean_cof);
    }
  }

  return ret;
}

REGISTER_OP_TILING_FUNC_BUFFERED(ReduceStdWithMean, ReduceStdWithMeanTiling);
}  // namespace optiling
