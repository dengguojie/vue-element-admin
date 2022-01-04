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

#include <iostream>
#include "../fusion_pass/common/fp16_t.hpp"
#include "vector_tiling.h"
#include "op_tiling_util.h"

namespace optiling {
struct L1LossGradCompileInfo {
  std::shared_ptr<AutoTilingHandler> tiling_handler;
  ge::DataType dtype;
};

bool L1LossGradTiling(const std::string& op_type, const ge::Operator& op_paras,
                      const L1LossGradCompileInfo& parsed_info, utils::OpRunInfo& run_info) {
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
                                 L1LossGradCompileInfo & parsed_info) {
  parsed_info.tiling_handler = CreateAutoTilingHandler(op_type, PATTERN_ELEMWISE, compile_info);
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CreateAutoTilingHandler return nullptr"),
                  return false);
  std::string dtype;
  parsed_info.dtype = ge::DT_MAX;
  if (GetCompileValue(compile_info, "reduce_mean_cof_dtype", dtype)) {
    parsed_info.dtype = (dtype == "float32") ? ge::DT_FLOAT : ge::DT_FLOAT16;
  }

  return true;
}

REGISTER_OP_TILING_V3_CUSTOM(L1LossGrad, L1LossGradTiling, ParseJsonCompileInfo, L1LossGradCompileInfo);
}  // namespace optiling
