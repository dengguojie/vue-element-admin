/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include "reduce_tiling.h"
#include "../fusion_pass/common/fp16_t.hpp"
#include "error_log.h"
#include "op_tiling_util.h"
#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"

namespace optiling {

struct ReduceMeanCompileInfo {
  std::shared_ptr<AutoTilingHandler> tiling_handler;
  ge::DataType reduce_mean_cof_dtype;
  bool has_reduce_mean_cof_dtype;
  int _idx_before_reduce;
  int axes_idx;
  std::vector<int32_t> _ori_axis;
};

bool IsInVector(const std::vector<int32_t>& input, int32_t value) {
  for (uint32_t i = 0; i < input.size(); i++) {
    if (input[i] == value) {
      return true;
    }
  }
  return false;
}

bool IsPureMove(const std::vector<int64_t>& input_shape, const std::vector<int32_t>& reduce_axis) {
  for (uint32_t i = 0; i < input_shape.size(); i++) {
    if (IsInVector(reduce_axis, i)) {
      if (input_shape[i] != 1) {
        return false;
      }
    }
  }
  return true;
}

bool GetInputShape(const std::string& op_type, const ge::Operator& op_paras,
                   const ReduceMeanCompileInfo& parsed_info, std::vector<int64_t>& input_shape_ori) {
  int idx = parsed_info._idx_before_reduce;
  // CHECK INPUT
  OP_TILING_CHECK((idx < 0),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "idx is invalid index for inputs"), return false);
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetOpDescFromOperator return nullptr!"), return false);

  auto input_desc = operator_info->MutableInputDesc(idx);
  OP_TILING_CHECK(input_desc == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input %d opdesc failed", idx), return false);
  input_shape_ori = input_desc->MutableShape().GetDims();
  return true;
}

bool GetReduceAxis(const std::string& op_type, const ge::Operator& op_paras, const ReduceMeanCompileInfo& parsed_info,
                   const std::vector<int64_t>& input_shape_ori, std::vector<int32_t>& reduce_axis_ori) {
  // Get ori reduce axis
  if(op_type == "ReduceMean") {
    int axes_idx = parsed_info.axes_idx;
    OP_TILING_CHECK(!ops::GetConstIntData(op_paras, axes_idx, reduce_axis_ori),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetConstIntData %d error!", axes_idx), return false);
  } else {
    // axis_known
    reduce_axis_ori = parsed_info._ori_axis;
  }

  // Convert reduce axis (-1 -> length+1)
  // CHECK AXIS VALUE
  int32_t max_value = int32_t(input_shape_ori.size());
  int32_t min_value = -1 * max_value;
  for (size_t i = 0; i < reduce_axis_ori.size(); i++) {
    bool is_illegal_case = reduce_axis_ori[i] >= max_value || reduce_axis_ori[i] < min_value;
    if (is_illegal_case) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "value of axis is illegal.");
      return false;
    }
    if (reduce_axis_ori[i] < 0) {
      reduce_axis_ori[i] = input_shape_ori.size() + reduce_axis_ori[i];
    }
  }
  return true;
}

bool ReduceMeanTiling(const std::string& op_type, const ge::Operator& op_paras,
                      const ReduceMeanCompileInfo& parsed_info, utils::OpRunInfo& run_info) {
  PROFILING_TILING_INIT(op_type.c_str());
  bool ret = true;
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetOpDescFromOperator return nullptr!"), return false);
  std::vector<int64_t> input_shape{std::vector<int64_t>(10, 0)};
  std::vector<int32_t> reduce_axis{std::vector<int32_t>(10, 0)};
  ret = ret && GetInputShape(op_type, op_paras, parsed_info, input_shape);
  ScalarToShape(input_shape);
  PROFILING_TILING_AFTER_GET_SHAPE_REG();
  ret = ret && GetReduceAxis(op_type, op_paras, parsed_info, input_shape, reduce_axis);
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();
  std::vector<std::vector<int64_t>> shapes = {input_shape};
  std::vector<std::vector<int32_t>> axes{reduce_axis};
  ge::DataType type = operator_info->MutableInputDesc(0)->GetDataType();
  OpInfo eletwise_info(shapes, type, axes);
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "parsed_info.tiling_handler nullptr, error!"),
                  return false);
  ret = parsed_info.tiling_handler->DoTiling(op_paras, run_info, eletwise_info);
  // reduce_mean_cof is not required when handling pure dma_copy case
  if (IsPureMove(input_shape, reduce_axis)) {
    return ret;
  }
  PROFILING_TILING_AFTER_CALCU_TILING_REG();
  float reduce_mean_cof = 1.0;
  if (parsed_info.has_reduce_mean_cof_dtype) {
    if (parsed_info.reduce_mean_cof_dtype == ge::DT_FLOAT) {
      for (uint32_t i = 0; i < input_shape.size(); i++) {
        if (input_shape[i] == 0) {
          OP_LOGD(op_type.c_str(), "reduce mean shape is 0");
          return ret;
        }
        if (IsInVector(reduce_axis, i)) {
          reduce_mean_cof = reduce_mean_cof / input_shape[i];
        }
      }
      run_info.AddTilingData((float)reduce_mean_cof);
      OP_LOGD(op_type.c_str(), "reduce mean cof:%f", reduce_mean_cof);
    } else {
      for (uint32_t i = 0; i < input_shape.size(); i++) {
        if (input_shape[i] == 0) {
          OP_LOGD(op_type.c_str(), "reduce mean shape is 0, dtype is fp16");
          return ret;
        }
        if (IsInVector(reduce_axis, i)) {
        if (input_shape[i] == 0) {
          VECTOR_INNER_ERR_REPORT_TILIING("reduce_mean", "inputshape = 0 is not support");
          return false;
        }
          reduce_mean_cof = reduce_mean_cof / input_shape[i];
        }
      }
      fe::fp16_t reduce_mean_cof_fp16 = reduce_mean_cof;
      run_info.AddTilingData((fe::fp16_t)reduce_mean_cof_fp16);
      run_info.AddTilingData((uint16_t)0);
      OP_LOGD(op_type.c_str(), "reduce mean cof:%f", reduce_mean_cof);
    }
  }
  PROFILING_TILING_END();

  return ret;
}

static bool ParseJsonCompileInfo(const std::string& op_type,
                          const nlohmann::json& compile_info,
                          ReduceMeanCompileInfo& parsed_info) {
  parsed_info.tiling_handler = CreateAutoTilingHandler(op_type, PATTERN_REDUCE, compile_info);
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CreateAutoTilingHandler return nullptr"),
                  return false);
  std::string dtype;
  parsed_info.has_reduce_mean_cof_dtype = false;
  if(GetCompileValue(compile_info, "reduce_mean_cof_dtype", dtype)) {
    parsed_info.has_reduce_mean_cof_dtype = true;
    parsed_info.reduce_mean_cof_dtype = (dtype == "float32") ? ge::DT_FLOAT : ge::DT_FLOAT16;
  }
  parsed_info._idx_before_reduce = 0;
  GetCompileValue(compile_info, "_idx_before_reduce", parsed_info._idx_before_reduce);
  //ReduceMean input axes index is 1
  parsed_info.axes_idx = 1;
  GetCompileValue(compile_info, "axes_idx", parsed_info.axes_idx);
  if(op_type == "ReduceMeanD") {
    OP_TILING_CHECK(!GetCompileValue(compile_info, "_ori_axis", parsed_info._ori_axis),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ParseJsonCompileInfo, get _ori_axis error"),
                    return false);
  }
  return true;
}

REGISTER_OP_TILING_V3_CUSTOM(ReduceMean, ReduceMeanTiling, ParseJsonCompileInfo, ReduceMeanCompileInfo);
REGISTER_OP_TILING_V3_CUSTOM(ReduceMeanD, ReduceMeanTiling, ParseJsonCompileInfo, ReduceMeanCompileInfo);
}  // namespace optiling
