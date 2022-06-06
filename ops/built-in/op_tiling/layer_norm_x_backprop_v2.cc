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
#include <cmath>

#include "error_util.h"
#include "../op_proto/util/op_common_util.h"
#include "graph/debug/ge_log.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "op_tiling.h"
#include "op_tiling_util.h"
#include "vector_tiling.h"

namespace optiling {
struct LayerNormXBackpropV2CompileInfo {
  std::shared_ptr<AutoTilingHandler> tiling_handler;
  bool reduce_mean_cof;
  bool unknown_mode;
};

bool LayerNormXBackpropV2ParseFunc(const std::string& op_type, const nlohmann::json& compile_info,
                                   LayerNormXBackpropV2CompileInfo& parsed_info) {
  parsed_info.tiling_handler = CreateAutoTilingHandler(op_type, PATTERN_NORM, compile_info);
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CreateAutoTilingHandler return nullptr"), return false);
  bool unknown_mode = false;
  bool reduce_mean_cof = false;

  if (GetCompileValue(compile_info, "unknown_mode", unknown_mode)) {
    unknown_mode = true;
  }
  if (GetCompileValue(compile_info, "reduce_mean_cof", reduce_mean_cof)) {
    reduce_mean_cof = true;
  }

  parsed_info.unknown_mode = unknown_mode;
  parsed_info.reduce_mean_cof = reduce_mean_cof;
  OP_LOGI(op_type.c_str(), "GetCompileParams success.");
  return true;
}

bool GetReduceAxis(std::vector<int64_t> input_x_shape, std::vector<int64_t> input_mean_shape,
                   ge::Format input_x_format, std::vector<int32_t>& reduce_axis,
                   std::vector<int64_t>& shape_x_nz) {
  int32_t rank = input_x_shape.size();
  if (input_x_format == FORMAT_FRACTAL_NZ) {
    int32_t nz_begin = rank - 4;
    for (int32_t i = 0; i < nz_begin; i++) {
      shape_x_nz.push_back(input_x_shape[i]);
    }
    shape_x_nz.push_back(input_x_shape[nz_begin]);
    shape_x_nz.push_back(input_x_shape[nz_begin+1]);
    shape_x_nz.push_back(input_x_shape[nz_begin+2]);
    shape_x_nz.push_back(input_x_shape[nz_begin+2]);

    std::vector<int64_t> shape_mean_nz = {};
    int32_t len_mean = input_mean_shape.size();
    int32_t mean_nz_begin = len_mean - 2;
    for (int32_t i = 0; i < mean_nz_begin; i++) {
      shape_mean_nz.push_back(input_mean_shape[i]);
    }
    shape_mean_nz.push_back(1);
    shape_mean_nz.push_back(input_x_shape[nz_begin+1]);
    shape_mean_nz.push_back(input_x_shape[nz_begin+2]);
    shape_mean_nz.push_back(1);

    int32_t x_nz_size = shape_x_nz.size();
    for (int32_t i = 0; i < x_nz_size; i++) {
      if (shape_x_nz[i] != shape_mean_nz[i]) {
        reduce_axis.push_back(i);
      }
    }
  } else {
    for (int32_t i = 0; i < rank; i++) {
      int64_t xtem = input_x_shape[i];
      int64_t mean = input_mean_shape[i];
      if (xtem != mean) {
        reduce_axis.push_back(i);
      }
    } 
  }
  return true;
}

bool LayerNormXBackpropV2UnknownAxisTiling(const std::string& op_type, const ge::Operator& op_paras,
                                           const LayerNormXBackpropV2CompileInfo& parsed_info,
                                           utils::OpRunInfo& run_info) {
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetOpDescFromOperator return nullptr"),
                  return false);
  auto input_x_desc = operator_info->MutableInputDesc(1);
  OP_TILING_CHECK(input_x_desc == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get input_x_desc return nullptr"),
                  return false);
  auto input_mean_desc = operator_info->MutableInputDesc(3);
  OP_TILING_CHECK(input_mean_desc == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get input_mean_desc return nullptr"),
                  return false);

  std::vector<int64_t> input_x_shape = input_x_desc->MutableShape().GetDims();
  std::vector<int64_t> input_mean_shape = input_mean_desc->MutableShape().GetDims();
  ge::Format input_x_format = input_x_desc->GetFormat();

  std::vector<int32_t> reduce_axis = {};
  std::vector<int64_t> shape_x_nz = {};
  GetReduceAxis(input_x_shape, input_mean_shape, input_x_format, reduce_axis, shape_x_nz);

  std::vector<std::vector<int64_t>> shapes = {input_x_shape};
  std::vector<std::vector<int32_t>> axes = {reduce_axis};
  ge::DataType input_x_dtype = input_x_desc->GetDataType();

  OpInfo norm_info(shapes, input_x_dtype, axes);
  OP_TILING_CHECK(!parsed_info.tiling_handler->DoTiling(op_paras, run_info, norm_info),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, 
                                                  "LayerNormXBackpropV2UnknownAxisTiling, do DoTiling failed"),
                  return false);
  if (parsed_info.reduce_mean_cof) {
    float mean_num = 1.0;
    int32_t reduce_axis_size = reduce_axis.size();
    if (input_x_format == FORMAT_FRACTAL_NZ) {
      for (int32_t i = 0; i < reduce_axis_size; i++) {
        mean_num *= shape_x_nz[reduce_axis[i]];
      }
    } else {
      for (int32_t i = 0; i < reduce_axis_size; i++) {
        mean_num *= input_x_shape[reduce_axis[i]];
      }
    }
    
    float mean_cof = pow(mean_num, -1);
    float mean_cof2 = pow(mean_num, -1) * 2;

    run_info.AddTilingData(mean_cof);
    run_info.AddTilingData(mean_cof2);
  }
  GELOGI("LayerNormXBackpropV2UnknownAxisTiling end.");
  return true;
}

bool LayerNormXBackpropV2Tiling(const std::string& op_type, const ge::Operator& op_paras,
                                const LayerNormXBackpropV2CompileInfo& parsed_info,
                                utils::OpRunInfo& run_info) {
  GELOGI("LayerNormXBackpropV2Tiling running.");
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetOpDescFromOperator return nullptr"),
                  return false);
  auto input_x_desc = operator_info->MutableInputDesc(1);
  OP_TILING_CHECK(input_x_desc == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get input_x_desc return nullptr"),
                  return false);
  auto input_mean_desc = operator_info->MutableInputDesc(3);
  OP_TILING_CHECK(input_x_desc == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get input_mean_desc return nullptr"),
                  return false);

  if (parsed_info.unknown_mode) {
    return LayerNormXBackpropV2UnknownAxisTiling(op_type, op_paras, parsed_info, run_info);
  }

  bool ret = parsed_info.tiling_handler->DoTiling(op_paras, run_info);
  if (!ret) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormXBackpropV2 tiling failed.");
    return false;
  }

  if (parsed_info.reduce_mean_cof) {
    std::vector<int64_t> input_x_shape = input_x_desc->MutableShape().GetDims();
    std::vector<int64_t> input_mean_shape = input_mean_desc->MutableShape().GetDims();
    ge::Format input_x_format = input_x_desc->GetFormat();

    std::vector<int32_t> reduce_axis = {};
    std::vector<int64_t> shape_x_nz = {};

    GetReduceAxis(input_x_shape, input_mean_shape, input_x_format, reduce_axis, shape_x_nz);
    float mean_num = 1.0;
    int32_t reduce_axis_size = reduce_axis.size();
    if (input_x_format == FORMAT_FRACTAL_NZ) {
      for (int32_t i = 0; i < reduce_axis_size; i++) {
        mean_num *= shape_x_nz[reduce_axis[i]];
      }
    } else {
      for (int32_t i = 0; i < reduce_axis_size; i++) {
        mean_num *= input_x_shape[reduce_axis[i]];
      }
    }
    
    float mean_cof = pow(mean_num, -1);
    float mean_cof2 = pow(mean_num, -1) * 2;

    run_info.AddTilingData(mean_cof);
    run_info.AddTilingData(mean_cof2);
  }

  GELOGI("LayerNormXBackpropTilingV2 end.");
  return true;
}
REGISTER_OP_TILING_V3_CUSTOM(LayerNormXBackpropV2, LayerNormXBackpropV2Tiling,
                             LayerNormXBackpropV2ParseFunc, LayerNormXBackpropV2CompileInfo);
}  // namespace optiling
