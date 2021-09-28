/* Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
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
enum tiling_key {
  NO_REDUCE = 400,
  ALL_DYNAMIC_NO_SPLIT = 200,
  ALL_DYNAMIC_SPLIT_NORMAL = 201,
  ALL_DYNAMIC_SPLIT_REDUCE = 202,
  ALL_DYNAMIC_SPLIT_REDUCE_SPLIT_NORMAL = 203,
  ALL_DYNAMIC_SPLIT_REDUCE_I = 204,
  DYNAMIC_REDUCE_NO_SPLIT = 100,
  DYNAMIC_REDUCE_SPLIT_REDUCE = 101,
  DYNAMIC_REDUCE_SPLIT_REDUCE_I = 102,
  DYNAMIC_NORMAL_NO_SPLIT = 300,
  DYNAMIC_NORMAL_SPLIT_NORMAL = 301,
  DYNAMIC_NORMAL_SPLIT_REDUCE = 302,
  DYNAMIC_NORMAL_SPLIT_REDUCE_SPLIT_NORMAL = 303,
  DYNAMIC_NORMAL_SPLIT_REDUCE_I = 304,
};

bool LayerNormBetaGammaBackpropTiling(const std::string& op_type, const TeOpParas& op_paras,
                                      const nlohmann::json& op_compile_info_json, OpRunInfo& run_info)
{
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

void NoReduceTiling(int32_t fused_dim, int32_t core_num, OpRunInfo& run_info)
{
  int32_t block_factor;
  int32_t ub_factor;
  int32_t block_dim = 1;
  if (fused_dim < core_num * 16) {
    block_factor =  fused_dim;
    ub_factor = fused_dim;
  } else {
    block_factor = (fused_dim + core_num - 1) / core_num;
    ub_factor = 16;
    block_dim = core_num;
  }
  ByteBufferPut(run_info.tiling_data, fused_dim);
  ByteBufferPut(run_info.tiling_data, block_factor);
  ByteBufferPut(run_info.tiling_data, ub_factor);
  run_info.tiling_key = NO_REDUCE;
  run_info.block_dim = block_dim;
}

void AllDynamicTiling(int32_t reduce_dim, int32_t normal_dim, int32_t core_num, int32_t max_reduce_factor,
                      int32_t max_last_factor, OpRunInfo& run_info)
{
  ByteBufferPut(run_info.tiling_data, reduce_dim);
  ByteBufferPut(run_info.tiling_data, normal_dim);
  if (reduce_dim <= core_num) {
    // do not split reduce
    run_info.block_dim = reduce_dim;
    if (normal_dim <= max_last_factor) {
      // no split reduce no split normal
      run_info.tiling_key = ALL_DYNAMIC_NO_SPLIT;
    } else {
      // no split reduce split normal
      run_info.tiling_key = ALL_DYNAMIC_SPLIT_NORMAL;
    }
  } else {
    run_info.block_dim = core_num;
    int32_t factor = (reduce_dim + core_num - 1) / core_num;
    if (normal_dim > max_last_factor) {
      // split reduce split normal
      run_info.tiling_key = ALL_DYNAMIC_SPLIT_REDUCE_SPLIT_NORMAL;
      ByteBufferPut(run_info.tiling_data, factor);
    } else if (factor > max_reduce_factor) {
      // split reduce_i no split normak,  max_reduce_factor is calc by normal_dim = max_last_factor
      run_info.tiling_key = ALL_DYNAMIC_SPLIT_REDUCE_I;
      ByteBufferPut(run_info.tiling_data, factor);
      int factor_i = (factor + max_reduce_factor - 1) / max_reduce_factor;
      ByteBufferPut(run_info.tiling_data, factor_i);
    } else {
      // split reduce no split normal
      run_info.tiling_key = ALL_DYNAMIC_SPLIT_REDUCE;
      ByteBufferPut(run_info.tiling_data, factor);
    }
  }
}

void DynamicReduceTiling(int32_t reduce_dim, int32_t normal_dim, int32_t core_num, int32_t max_reduce_factor,
                         int32_t max_last_factor, OpRunInfo& run_info)
{
  int32_t factor = (reduce_dim + core_num - 1) / core_num;
  ByteBufferPut(run_info.tiling_data, reduce_dim);
  if (reduce_dim <= core_num) {
    run_info.tiling_key = DYNAMIC_REDUCE_NO_SPLIT;
    run_info.block_dim = reduce_dim;
  } else if (normal_dim > max_last_factor || reduce_dim <= core_num * max_reduce_factor) {
    run_info.tiling_key = DYNAMIC_REDUCE_SPLIT_REDUCE;
    run_info.block_dim = core_num;
    ByteBufferPut(run_info.tiling_data, factor);
  } else {
    run_info.tiling_key = DYNAMIC_REDUCE_SPLIT_REDUCE_I;
    run_info.block_dim = core_num;
    ByteBufferPut(run_info.tiling_data, factor);
  }
}

void DynamicNormalTiling(int32_t reduce_dim, int32_t normal_dim, int32_t core_num, int32_t max_reduce_factor,
                         int32_t max_last_factor, OpRunInfo& run_info)
{
  ByteBufferPut(run_info.tiling_data, normal_dim);
  int32_t factor = (reduce_dim + core_num - 1) / core_num;
  if (reduce_dim <= core_num) {
    run_info.block_dim = reduce_dim;
    if (normal_dim <= max_last_factor) {
      // no split reduce no split normal
      run_info.tiling_key = DYNAMIC_NORMAL_NO_SPLIT;
    } else {
      run_info.tiling_key = DYNAMIC_NORMAL_SPLIT_NORMAL;
    }
  } else {
    run_info.block_dim = core_num;
    if (normal_dim > max_last_factor) {
      // split reduce split normal
      run_info.tiling_key = DYNAMIC_NORMAL_SPLIT_REDUCE_SPLIT_NORMAL;
    } else if (factor > max_reduce_factor) {
      run_info.tiling_key = DYNAMIC_NORMAL_SPLIT_REDUCE_I;
    } else {
      run_info.tiling_key = DYNAMIC_NORMAL_SPLIT_REDUCE;
    }
  }
}

bool LayerNormBetaGammaBackpropV2Tiling(const std::string& op_type, const TeOpParas& op_paras,
                                        const nlohmann::json& op_compile_info_json, OpRunInfo& run_info)
{
  std::vector<int64_t> input_shape = op_paras.inputs[0].tensor[0].shape;
  std::vector<int64_t> shape_gamma = op_compile_info_json["shape_gamma"].get<std::vector<int64_t>>();
  int32_t core_num = op_compile_info_json["core_num"].get<int32_t>();
  int32_t max_reduce_factor = op_compile_info_json["max_reduce_factor"].get<int32_t>();
  int32_t max_last_factor = op_compile_info_json["max_last_factor"].get<int32_t>();
  bool dynamic_reduce = op_compile_info_json["dynamic_reduce"].get<bool>();
  bool dynamic_normal = op_compile_info_json["dynamic_normal"].get<bool>();
  int32_t normal_dim = 1;
  int32_t reduce_dim = 1;
  int32_t i;
  if (core_num <= 0) {
    GELOGE(ge::FAILED, "Get invalid core_num.");
    return false;
  }
  for (i = 0; i < input_shape.size() - shape_gamma.size(); i++) {
    reduce_dim *= input_shape[i];
  }
  for (; i < input_shape.size(); i++) {
    normal_dim *= input_shape[i];
  }
  if (input_shape.size() == shape_gamma.size()) {
    NoReduceTiling(normal_dim, core_num, run_info);
  } else if (dynamic_reduce && dynamic_normal) {
    AllDynamicTiling(reduce_dim, normal_dim, core_num, max_reduce_factor, max_last_factor, run_info);
  } else if (dynamic_reduce) {
    DynamicReduceTiling(reduce_dim, normal_dim, core_num, max_reduce_factor, max_last_factor, run_info);
  } else if (dynamic_normal) {
    DynamicNormalTiling(reduce_dim, normal_dim, core_num, max_reduce_factor, max_last_factor, run_info);
  }
  return true;
}
REGISTER_OP_TILING_FUNC_BUFFERED(LayerNormBetaGammaBackprop, LayerNormBetaGammaBackpropTiling);
REGISTER_OP_TILING_FUNC_BUFFERED(LayerNormBetaGammaBackpropV2, LayerNormBetaGammaBackpropV2Tiling);
}  // namespace optiling
