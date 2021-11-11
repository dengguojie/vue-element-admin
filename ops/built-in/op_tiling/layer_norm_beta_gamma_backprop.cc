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
#include "../op_proto/util/op_common_util.h"
#include "graph/debug/ge_log.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "op_tiling.h"
#include "op_tiling_util.h"

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

struct opInfoV1 {
  /* data */
  int32_t core_num;
  int32_t ub_size;
  int32_t batch_cols_padding;
  int32_t k_num;
};

struct opInfoV2 {
  /* data */
  int32_t core_num;
  int32_t max_reduce_factor;
  int32_t max_last_factor;
  std::vector<int64_t> shape_gamma;
  bool dynamic_reduce;
  bool dynamic_normal;
};

bool LayerNormBetaGammaBackpropParseFunc(const std::string& op_type, const nlohmann::json& compile_info,
                                         opInfoV1& compile_value) {
  using namespace nlohmann;
  OP_TILING_CHECK(compile_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_compile_info_json is null"),
                  return false);
  auto vars = compile_info["vars"];

  OP_TILING_CHECK(!GetCompileValue(vars, "core_num", compile_value.core_num),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormBetaGammaBackpropParseFunc get core_num error"),
                  return false);
  OP_TILING_CHECK(!GetCompileValue(vars, "ub_size", compile_value.ub_size),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormBetaGammaBackpropParseFunc get ub_size error"),
                  return false);
  OP_TILING_CHECK(
      !GetCompileValue(vars, "batch_cols_padding", compile_value.batch_cols_padding),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormBetaGammaBackpropParseFunc get batch_cols_padding error"),
      return false);
  OP_TILING_CHECK(!GetCompileValue(vars, "k_num", compile_value.k_num),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormBetaGammaBackpropParseFunc get k_num error"),
                  return false);
  OP_LOGI(op_type.c_str(), "GetCompileParams success.");
  return true;
}

bool LayerNormBetaGammaBackpropV2ParseFunc(const std::string& op_type, const nlohmann::json& compile_info,
                                           opInfoV2& compile_value) {
  OP_TILING_CHECK(!GetCompileValue(compile_info, "core_num", compile_value.core_num),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormBetaGammaBackpropV2ParseFunc, get core_num error"),
                  return false);
  OP_TILING_CHECK(
      !GetCompileValue(compile_info, "max_reduce_factor", compile_value.max_reduce_factor),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormBetaGammaBackpropV2ParseFunc, get max_reduce_factor error"),
      return false);
  OP_TILING_CHECK(
      !GetCompileValue(compile_info, "max_last_factor", compile_value.max_last_factor),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormBetaGammaBackpropV2ParseFunc, get max_last_factor error"),
      return false);
  OP_TILING_CHECK(
      !GetCompileValue(compile_info, "shape_gamma", compile_value.shape_gamma),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormBetaGammaBackpropV2ParseFunc, get shape_gamma error"),
      return false);

  OP_TILING_CHECK(
      !GetCompileValue(compile_info, "dynamic_reduce", compile_value.dynamic_reduce),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormBetaGammaBackpropV2ParseFunc, get dynamic_reduce error"),
      return false);
  OP_TILING_CHECK(
      !GetCompileValue(compile_info, "dynamic_normal", compile_value.dynamic_normal),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormBetaGammaBackpropV2ParseFunc, get dynamic_normal error"),
      return false);
  OP_LOGI(op_type.c_str(), "GetCompileParams success.");
  return true;
}

bool LayerNormBetaGammaBackpropTiling(const std::string& op_type, const ge::Operator& op_paras,
                                      const opInfoV1& op_compile_info_json, utils::OpRunInfo& run_info) {
  GELOGI("LayerNormBetaGammaBackprop Tiling running.");
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  auto input0_desc = operator_info->MutableInputDesc(0);
  std::vector<int64_t> input_shape = input0_desc->MutableShape().GetDims();
  if (input_shape.size() < 2) {
    GELOGE(ge::FAILED, "LayerNormBetaGammaBackprop currrently not support input shape size less than 2.");
    return false;
  }
  int32_t dim_0 = input_shape[0];
  int32_t dim_1 = input_shape[1];

  run_info.AddTilingData(dim_0);
  run_info.AddTilingData(dim_1);
  run_info.SetBlockDim(dim_0);
  run_info.SetTilingKey(1);

  GELOGI("LayerNormBetaGammaBackprop Tiling end.");
  return true;
}

void NoReduceTiling(int32_t fused_dim, int32_t core_num, utils::OpRunInfo& run_info) {
  int32_t block_factor;
  int32_t ub_factor;
  int32_t block_dim = 1;
  if (fused_dim < core_num * 16) {
    block_factor = fused_dim;
    ub_factor = fused_dim;
  } else {
    block_factor = (fused_dim + core_num - 1) / core_num;
    ub_factor = 16;
    block_dim = core_num;
  }
  run_info.AddTilingData(fused_dim);
  run_info.AddTilingData(block_factor);
  run_info.AddTilingData(ub_factor);
  run_info.SetTilingKey(NO_REDUCE);
  run_info.SetBlockDim(block_dim);
}

void AllDynamicTiling(int32_t reduce_dim, int32_t normal_dim, int32_t core_num, int32_t max_reduce_factor,
                      int32_t max_last_factor, utils::OpRunInfo& run_info) {
  run_info.AddTilingData(reduce_dim);
  run_info.AddTilingData(normal_dim);
  if (reduce_dim <= core_num) {
    // do not split reduce
    run_info.SetBlockDim(reduce_dim);
    if (normal_dim <= max_last_factor) {
      // no split reduce no split normal
      run_info.SetTilingKey(ALL_DYNAMIC_NO_SPLIT);
    } else {
      // no split reduce split normal
      run_info.SetTilingKey(ALL_DYNAMIC_SPLIT_NORMAL);
    }
  } else {
    run_info.SetBlockDim(core_num);
    int32_t factor = (reduce_dim + core_num - 1) / core_num;
    if (normal_dim > max_last_factor) {
      // split reduce split normal
      run_info.SetTilingKey(ALL_DYNAMIC_SPLIT_REDUCE_SPLIT_NORMAL);
      run_info.AddTilingData(factor);
    } else if (factor > max_reduce_factor) {
      // split reduce_i no split normak,  max_reduce_factor is calc by normal_dim = max_last_factor
      run_info.SetTilingKey(ALL_DYNAMIC_SPLIT_REDUCE_I);
      run_info.AddTilingData(factor);
      int factor_i = (factor + max_reduce_factor - 1) / max_reduce_factor;
      run_info.AddTilingData(factor_i);
    } else {
      // split reduce no split normal
      run_info.SetTilingKey(ALL_DYNAMIC_SPLIT_REDUCE);
      run_info.AddTilingData(factor);
    }
  }
}

void DynamicReduceTiling(int32_t reduce_dim, int32_t normal_dim, int32_t core_num, int32_t max_reduce_factor,
                         int32_t max_last_factor, utils::OpRunInfo& run_info) {
  int32_t factor = (reduce_dim + core_num - 1) / core_num;
  run_info.AddTilingData(reduce_dim);
  if (reduce_dim <= core_num) {
    run_info.SetTilingKey(DYNAMIC_REDUCE_NO_SPLIT);
    run_info.SetBlockDim(reduce_dim);
  } else if (normal_dim > max_last_factor || reduce_dim <= core_num * max_reduce_factor) {
    run_info.SetTilingKey(DYNAMIC_REDUCE_SPLIT_REDUCE);
    run_info.SetBlockDim(core_num);
    run_info.AddTilingData(factor);
  } else {
    run_info.SetTilingKey(DYNAMIC_REDUCE_SPLIT_REDUCE_I);
    run_info.SetBlockDim(core_num);
    run_info.AddTilingData(factor);
  }
}

void DynamicNormalTiling(int32_t reduce_dim, int32_t normal_dim, int32_t core_num, int32_t max_reduce_factor,
                         int32_t max_last_factor, utils::OpRunInfo& run_info) {
  run_info.AddTilingData(normal_dim);
  int32_t factor = (reduce_dim + core_num - 1) / core_num;
  if (reduce_dim <= core_num) {
    run_info.SetBlockDim(reduce_dim);
    if (normal_dim <= max_last_factor) {
      // no split reduce no split normal
      run_info.SetTilingKey(DYNAMIC_NORMAL_NO_SPLIT);
    } else {
      run_info.SetTilingKey(DYNAMIC_NORMAL_SPLIT_NORMAL);
    }
  } else {
    run_info.SetBlockDim(core_num);
    if (normal_dim > max_last_factor) {
      // split reduce split normal
      run_info.SetTilingKey(DYNAMIC_NORMAL_SPLIT_REDUCE_SPLIT_NORMAL);
    } else if (factor > max_reduce_factor) {
      run_info.SetTilingKey(DYNAMIC_NORMAL_SPLIT_REDUCE_I);
    } else {
      run_info.SetTilingKey(DYNAMIC_NORMAL_SPLIT_REDUCE);
    }
  }
}

bool LayerNormBetaGammaBackpropV2Tiling(const std::string& op_type, const ge::Operator& op_paras,
                                        const opInfoV2& op_compile_info_json, utils::OpRunInfo& run_info) {
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  auto input0_desc = operator_info->MutableInputDesc(0);
  std::vector<int64_t> input_shape = input0_desc->MutableShape().GetDims();

  std::vector<int64_t> shape_gamma = op_compile_info_json.shape_gamma;
  int32_t core_num = op_compile_info_json.core_num;
  int32_t max_reduce_factor = op_compile_info_json.max_reduce_factor;
  int32_t max_last_factor = op_compile_info_json.max_last_factor;
  bool dynamic_reduce = op_compile_info_json.dynamic_reduce;
  bool dynamic_normal = op_compile_info_json.dynamic_normal;
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
REGISTER_OP_TILING_V3_CUSTOM(LayerNormBetaGammaBackprop, LayerNormBetaGammaBackpropTiling,
                             LayerNormBetaGammaBackpropParseFunc, opInfoV1);
REGISTER_OP_TILING_V3_CUSTOM(LayerNormBetaGammaBackpropV2, LayerNormBetaGammaBackpropV2Tiling,
                             LayerNormBetaGammaBackpropV2ParseFunc, opInfoV2);
}  // namespace optiling
