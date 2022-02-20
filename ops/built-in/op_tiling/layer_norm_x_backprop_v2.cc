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
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

#include "../op_proto/util/error_util.h"
#include "../op_proto/util/op_common_util.h"
#include "graph/debug/ge_log.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "op_tiling.h"
#include "op_tiling_util.h"

namespace {
  constexpr int32_t INDEX_TWO = 2;
  constexpr int32_t BASE_INPUT_SIZE = 3;
}

namespace optiling {
struct opInfo {
  /* data */
  int32_t UB_SIZE;
  int32_t CORE_NUM;
  int32_t MAX_DTYPE;
  int32_t COEXISTING_QUANTITY;
};

bool LayerNormXBackpropV2ParseFunc(const std::string& op_type, const nlohmann::json& compile_info,
                                   opInfo& compile_value) {
  OP_TILING_CHECK(!GetCompileValue(compile_info, "UB_SIZE", compile_value.UB_SIZE),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormXBackpropV2ParseFunc, get UB_SIZE error"),
                  return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "CORE_NUM", compile_value.CORE_NUM),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormXBackpropV2ParseFunc, get CORE_NUM error"),
                  return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "MAX_DTYPE", compile_value.MAX_DTYPE),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormXBackpropV2ParseFunc, get MAX_DTYPE error"),
                  return false);
  OP_TILING_CHECK(
      !GetCompileValue(compile_info, "COEXISTING_QUANTITY", compile_value.COEXISTING_QUANTITY),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormXBackpropV2ParseFunc, get COEXISTING_QUANTITY error"),
      return false);
  OP_LOGI(op_type.c_str(), "GetCompileParams success.");
  return true;
}

bool GetLayerNormXBackpropV2CompileParams(const opInfo& op_info, int32_t& core_num,
                                          int32_t& ub_size, int32_t& max_dtype) {
  core_num = op_info.CORE_NUM;
  ub_size = op_info.UB_SIZE;
  max_dtype = op_info.MAX_DTYPE;

  return true;
}

bool LayerNormXBackpropV2Tiling(const std::string& op_type, const ge::Operator& op_paras, const opInfo& op_info,
                                utils::OpRunInfo& run_info) {
  GELOGI("LayerNormXBackpropV2Tiling running.");
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  auto input0_desc = operator_info->MutableInputDesc(0);
  std::vector<int64_t> input_shape = input0_desc->MutableShape().GetDims();
  int32_t fmap_x0 = input_shape[0];
  int32_t fmap_x1 = input_shape[1];
  int32_t fmap_muli = input_shape[1] * input_shape[INDEX_TWO];
  int32_t core_num = 0;
  int32_t ub_size = 0;
  int32_t max_dtype = 0;
  int32_t THREE_DIMEN_KEY = 10000;
  int32_t FOUR_DIMEN_KEY = 20000;

  bool ret = GetLayerNormXBackpropV2CompileParams(op_info, core_num, ub_size, max_dtype);
  if (!ret) {
    OP_LOGE(op_type.c_str(), "GetLayerNormXBackpropV2CompileParams failed");
    return false;
  }
  GELOGI("op[%s] GetLayerNormXBackpropV2CompileParams success.", op_type.c_str());

  run_info.AddTilingData(fmap_x0);
  run_info.AddTilingData(fmap_x1);
  run_info.AddTilingData(fmap_muli);
  run_info.SetBlockDim(core_num);
  if (input_shape.size() > BASE_INPUT_SIZE) {
    run_info.SetTilingKey(FOUR_DIMEN_KEY);
  } else if (input_shape.size() == BASE_INPUT_SIZE) {
    run_info.SetTilingKey(THREE_DIMEN_KEY);
  }

  GELOGI("LayerNormXBackpropTilingV2 end.");
  return true;
}
REGISTER_OP_TILING_V3_CUSTOM(LayerNormXBackpropV2, LayerNormXBackpropV2Tiling, LayerNormXBackpropV2ParseFunc, opInfo);
}  // namespace optiling
