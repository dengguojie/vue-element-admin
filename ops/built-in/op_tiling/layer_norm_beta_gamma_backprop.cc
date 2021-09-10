/* Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include <algorithm>
#include <nlohmann/json.hpp>
#include <string>

#include "../op_proto/util/error_util.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "op_tiling.h"

namespace optiling {
enum tiling_key { NO_SPLIT = 0, SPLIT_REDUCE = 1, SPLIT_REDUCE_I = 2 };

bool LayerNormBetaGammaBackpropTiling(const std::string& op_type, const TeOpParas& op_paras,
                                      const nlohmann::json& op_compile_info_json, OpRunInfo& run_info) {
  GELOGI("LayerNormBetaGammaBackprop Tiling running.");
  std::vector<int64_t> input_shape = op_paras.inputs[0].tensor[0].shape;
  if (input_shape.size() < 2) {
    GELOGE(ge::FAILED, "LayerNormBetaGammaBackprop currrently not support input shape size less than 2.");
    return false;
  }
  int32_t dim_0 = input_shape[0];
  int32_t dim_1 = input_shape[1];

  ByteBufferPut(run_info.tiling_data, dim_0);
  ByteBufferPut(run_info.tiling_data, dim_1);
  run_info.block_dim = dim_0;
  run_info.tiling_key = 1;

  GELOGI("LayerNormBetaGammaBackprop Tiling end.");
  return true;
}
bool LayerNormBetaGammaBackpropV2Tiling(const std::string& op_type, const TeOpParas& op_paras,
                                        const nlohmann::json& op_compile_info_json, OpRunInfo& run_info) {
  GELOGI("LayerNormBetaGammaBackpropV2 Tiling running.");
  std::vector<int64_t> input_shape = op_paras.inputs[0].tensor[0].shape;
  if (input_shape.size() < 2) {
    GELOGE(ge::FAILED, "LayerNormBetaGammaBackprop currrently not support input shape size less than 2.");
    return false;
  }
  int32_t dim_0 = input_shape[0];
  int32_t dim_1 = input_shape[1];
  int32_t reduce_dim = dim_0 * dim_1;
  int32_t core_num = op_compile_info_json["core_num"].get<int32_t>();
  int32_t max_reduce_factor = op_compile_info_json["max_reduce_factor"].get<int32_t>();
  OP_LOGI(op_type.c_str(), "core_num = %d, max_reduce_factor = %d", core_num, max_reduce_factor);
  if (core_num <= 0 || max_reduce_factor<= 0) {
    return false;
  }
  int32_t factor = (reduce_dim + core_num - 1) / core_num;
  ByteBufferPut(run_info.tiling_data, reduce_dim);
  if (reduce_dim <= core_num) {
    run_info.tiling_key = NO_SPLIT;
    run_info.block_dim = reduce_dim;
  } else if(reduce_dim <= core_num * max_reduce_factor) {
    run_info.tiling_key = SPLIT_REDUCE;
    run_info.block_dim = core_num;
    ByteBufferPut(run_info.tiling_data, factor);
  } else {
    run_info.tiling_key = SPLIT_REDUCE_I;
    run_info.block_dim = core_num;
    ByteBufferPut(run_info.tiling_data, factor);
  }
  GELOGI("LayerNormBetaGammaBackpropV2 Tiling end.");
  return true;
}
REGISTER_OP_TILING_FUNC_BUFFERED(LayerNormBetaGammaBackprop, LayerNormBetaGammaBackpropTiling);
REGISTER_OP_TILING_FUNC_BUFFERED(LayerNormBetaGammaBackpropV2, LayerNormBetaGammaBackpropV2Tiling);
}  // namespace optiling
