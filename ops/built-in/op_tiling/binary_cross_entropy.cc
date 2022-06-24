/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#include "../fusion_pass/common/fp16_t.hpp"
#include "vector_tiling.h"
#include "op_tiling_util.h"

namespace optiling {
struct BroadcastCompileInfo {
  std::shared_ptr<AutoTilingHandler> tiling_handler;
  ge::DataType dtype;
  bool reduction_is_none;
};

bool IsInAxis(std::vector<int32_t>& input, int32_t value) {
  for (uint32_t i = 0; i < input.size(); i++) {
    if (input[i] == value) {
      return true;
    }
  }
  return false;
}

bool BinaryCrossEntropyTiling(const std::string& op_type, const ge::Operator& op_paras,
                              const BroadcastCompileInfo& parsed_info, utils::OpRunInfo& run_info) {
  PROFILING_TILING_INIT(op_type.c_str());
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "parsed_info.tiling_handler nullptr, error!"), return false);
  if (parsed_info.reduction_is_none) {
    bool ret = parsed_info.tiling_handler->DoTiling(op_paras, run_info);
    return ret;
  }

  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get op_info failed."),
                  return false);

  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_desc failed."),
                  return false);

  ge::GeShape input_shape = input_desc->MutableShape();
  ge::DataType type = input_desc->GetDataType();
  int32_t input_size = static_cast<int32_t>(input_shape.GetDimNum());
  PROFILING_TILING_AFTER_GET_SHAPE_REG();

  std::vector<int32_t> reduce_axis{};
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  for (int32_t i = 0; i < input_size; i++) {
    reduce_axis.insert(reduce_axis.end(), i);
  }

  vector<vector<int64_t>> input_shapes = {input_shape.GetDims()};
  vector<vector<int32_t>> input_axes = {reduce_axis};
  OpInfo reduce_info(input_shapes, type, input_axes);
  bool ret = parsed_info.tiling_handler->DoTiling(op_paras, run_info, reduce_info);
  PROFILING_TILING_AFTER_CALCU_TILING_REG();

  if (parsed_info.dtype == DT_FLOAT || parsed_info.dtype == DT_FLOAT16) {
    float reduce_mean_cof = 1.0;
    for (int32_t i = 0; i < input_size; i++) {
      if (IsInAxis(reduce_axis, i)) {
        OP_TILING_CHECK(input_shape.GetDim(i) == 0,
                        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input_shape cannot include 0."), return false);
        reduce_mean_cof = reduce_mean_cof / input_shape.GetDim(i);
      }
    }
    if (parsed_info.dtype == DT_FLOAT) {
      run_info.AddTilingData((float)reduce_mean_cof);
    } else if (parsed_info.dtype == DT_FLOAT16) {
      fe::fp16_t reduce_mean_cof_fp16 = reduce_mean_cof;
      run_info.AddTilingData(reduce_mean_cof_fp16);
      run_info.AddTilingData((uint16_t)0);
    }
    OP_LOGD(op_type.c_str(), "reduce mean cof:%f", reduce_mean_cof);
  }

  PROFILING_TILING_END();

  return ret;
}

static bool ParseJsonCompileInfo(const std::string& op_type, const nlohmann::json& compile_info,
                                 BroadcastCompileInfo& parsed_info) {
  parsed_info.reduction_is_none = false;
  std::string reduction;
  OP_TILING_CHECK(!GetCompileValue(compile_info, "reduction", reduction),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ParseJsonCompileInfo, get reduction error"), return false);
  if (reduction == "none") {
    parsed_info.reduction_is_none = true;
    parsed_info.tiling_handler = CreateAutoTilingHandler(op_type, PATTERN_ELEMWISE, compile_info);
    OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CreateAutoTilingHandler return nullptr"), return false);
  } else {
    parsed_info.tiling_handler = CreateAutoTilingHandler(op_type, PATTERN_REDUCE, compile_info);
    OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CreateAutoTilingHandler return nullptr"), return false);
    std::string dtype;
    parsed_info.dtype = ge::DT_MAX;
    if (GetCompileValue(compile_info, "reduce_mean_cof_dtype", dtype)) {
      parsed_info.dtype = (dtype == "float32") ? ge::DT_FLOAT : ge::DT_FLOAT16;
    }
  }

  return true;
}

REGISTER_OP_TILING_V3_CUSTOM(BinaryCrossEntropy, BinaryCrossEntropyTiling, ParseJsonCompileInfo, BroadcastCompileInfo);
}  // namespace optiling
