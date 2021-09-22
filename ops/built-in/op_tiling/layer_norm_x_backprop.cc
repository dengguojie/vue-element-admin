/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.

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
#include <iostream>

#include "../op_proto/util/error_util.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "op_tiling.h"
#include "error_log.h"

namespace optiling {
  bool GetLayerNormXBackpropCompileParams(const std::string& op_type, const nlohmann::json& op_info,
                                          int32_t& core_num, int32_t& ub_size, int32_t& max_dtype)
  {
    using namespace nlohmann;
    if (op_info == nullptr) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_info is null");
      return false;
    }
    if (op_info.count("CORE_NUM") == 0 || op_info.count("UB_SIZE") == 0 || op_info.count("MAX_DTYPE") == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CORE_NUM or UB_SIZE or MAX_DTYPE is null");
      return false;
    }
    core_num = op_info["CORE_NUM"].get<std::int32_t>();
    ub_size = op_info["UB_SIZE"].get<std::int32_t>();
    max_dtype = op_info["MAX_DTYPE"].get<std::int32_t>();

    return true;
  }

  bool LayerNormXBackpropTiling(const std::string &op_type, const TeOpParas &op_paras, const nlohmann::json &op_info,
                                OpRunInfo &run_info)
  {
    GELOGI("LayerNormXBackpropTiling running.");
    std::vector<int64_t> input_shape = op_paras.inputs[0].tensor[0].shape;
    int32_t fmap_x0 = input_shape[0];
    int32_t fmap_x1 = input_shape[1];
    int32_t core_num = 0;
    int32_t ub_size = 0;
    int32_t max_dtype = 0;
    int32_t CUT_AXIS_ONE_TILING_KEY = 10000;

    bool ret = GetLayerNormXBackpropCompileParams(op_type, op_info, core_num, ub_size, max_dtype);
    if (!ret) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetLayerNormXBackpropCompileParams failed.");
      return false;
    }
    GELOGI("op[%s] GetLayerNormXBackpropCompileParams success.", op_type.c_str());

    ByteBufferPut(run_info.tiling_data, fmap_x0);
    ByteBufferPut(run_info.tiling_data, fmap_x1);

    run_info.block_dim = core_num;
    run_info.tiling_key = CUT_AXIS_ONE_TILING_KEY;
    GELOGI("LayerNormXBackpropTiling end.");

    return true;
  }
    REGISTER_OP_TILING_FUNC_BUFFERED(LayerNormXBackprop, LayerNormXBackpropTiling);
}
