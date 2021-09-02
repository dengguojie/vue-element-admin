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
#include <iostream>
#include "../op_proto/util/error_util.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "op_tiling.h"

namespace optiling {
  bool GetLayerNormXBackpropV2CompileParams(const std::string& op_type, const nlohmann::json& op_info,
                                            int32_t& core_num, int32_t& ub_size, int32_t& max_dtype) {
    using namespace nlohmann;
    if (op_info == nullptr) {
      ge::OpsGetCompileParamsErrReport("LayerNormXBackpropV2", "op_info");
      OP_LOGE(op_type.c_str(), "op_info is null");
      return false;
    }
    if (op_info.count("CORE_NUM") == 0 || op_info.count("UB_SIZE") == 0 || op_info.count("MAX_DTYPE") == 0)
    {
      ge::OpsGetCompileParamsErrReport("LayerNormXBackpropV2", "CORE_NUM_UB_SIZE_MAX_DTYPE");
      OP_LOGE(op_type.c_str(), "CORE_NUM or UB_SIZE or MAX_DTYPE is null");
      return false;
    }
    core_num = op_info["CORE_NUM"].get<std::int32_t>();
    ub_size = op_info["UB_SIZE"].get<std::int32_t>();
    max_dtype = op_info["MAX_DTYPE"].get<std::int32_t>();

    return true;
  }

  bool LayerNormXBackpropV2Tiling(const std::string &op_type, const TeOpParas &op_paras, const nlohmann::json &op_info,
                                  OpRunInfo &run_info) {
    GELOGI("LayerNormXBackpropV2Tiling running.");
    std::vector<int64_t> input_shape = op_paras.inputs[0].tensor[0].shape;
    int32_t fmap_x0 = input_shape[0];
    int32_t fmap_x1 = input_shape[1];
    int32_t core_num = 0;
    int32_t ub_size = 0;
    int32_t max_dtype = 0;

    bool ret = GetLayerNormXBackpropV2CompileParams(op_type, op_info, core_num, ub_size, max_dtype);
    if (!ret) {
      OP_LOGE("op[%s] GetLayerNormXBackpropV2CompileParams failed.", op_type.c_str());
      return false;
    }
    GELOGI("op[%s] GetLayerNormXBackpropV2CompileParams success.", op_type.c_str());

    ByteBufferPut(run_info.tiling_data, fmap_x0);
    ByteBufferPut(run_info.tiling_data, fmap_x1);

    run_info.block_dim = core_num;
    run_info.tiling_key = 10000;
    GELOGI("LayerNormXBackpropTilingV2 end.");
    return true;
  }
    REGISTER_OP_TILING_FUNC_BUFFERED(LayerNormXBackpropV2, LayerNormXBackpropV2Tiling);
}
