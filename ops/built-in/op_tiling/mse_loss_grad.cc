/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include "reduce_tiling.h"
#include "eletwise.h"
#include "../fusion_pass/common/fp16_t.hpp"
#include "op_tiling_util.h"

namespace optiling {

struct MseLossGradCompileInfo {
  std::shared_ptr<AutoTilingHandler> tiling_handler;
  ge::DataType dtype;
};

bool MseLossGradTiling(const std::string& op_type, const ge::Operator& op_paras,
                                  const MseLossGradCompileInfo& parsed_info, utils::OpRunInfo& run_info) {
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get op_info failed."),
                  return false);

  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_desc failed."),
                  return false);

  const std::vector<int64_t> input_shape = input_desc->MutableShape().GetDims();

  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "parsed_info.tiling_handler nullptr, error!"),
                  return false);
  bool ret = parsed_info.tiling_handler->DoTiling(op_paras, run_info);
  // reduce_mean_cof is not required when handling pure dma_copy case

  float reduce_mean_cof = 1.0;
  for (uint32_t i = 0; i < input_shape.size(); i++) {
    OP_TILING_CHECK(input_shape[i] == 0, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input_shape cannot include 0."),
                    return false);
    reduce_mean_cof = reduce_mean_cof / input_shape[i];
  }
  if (parsed_info.dtype == ge::DT_FLOAT) {
    run_info.AddTilingData((float)reduce_mean_cof);
  } else if (parsed_info.dtype == ge::DT_FLOAT16) {
    fe::fp16_t reduce_mean_cof_fp16 = reduce_mean_cof;
    run_info.AddTilingData((fe::fp16_t)reduce_mean_cof_fp16);
    run_info.AddTilingData((uint16_t)0);
  }
  OP_LOGD(op_type.c_str(), "reduce mean cof:%f", reduce_mean_cof);

  return ret;
}

static bool ParseJsonCompileInfo(const std::string& op_type, const nlohmann::json& compile_info,
                                 MseLossGradCompileInfo& parsed_info) {
  parsed_info.tiling_handler = CreateAutoTilingHandler(op_type, PATTERN_BROADCAST, compile_info);
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CreateAutoTilingHandler return nullptr"),
                  return false);
  std::string dtype;
  OP_TILING_CHECK(!GetCompileValue(compile_info, "reduce_mean_cof_dtype", dtype),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ParseJsonCompileInfo get reduce_mean_cof_dtype error"),
                  return false);
  parsed_info.dtype = (dtype == "float32") ? ge::DT_FLOAT : ge::DT_FLOAT16;

  return true;
}

REGISTER_OP_TILING_V3_CUSTOM(MseLossGrad, MseLossGradTiling, ParseJsonCompileInfo, MseLossGradCompileInfo);
}  // namespace optiling
