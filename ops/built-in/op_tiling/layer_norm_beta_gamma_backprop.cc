/*
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

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
REGISTER_OP_TILING_FUNC_BUFFERED(LayerNormBetaGammaBackprop, LayerNormBetaGammaBackpropTiling);
}  // namespace optiling
