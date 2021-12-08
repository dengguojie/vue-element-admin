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

/*!
 * \file bn_reduce_grad.cpp
 * \brief
 */
#include "eletwise.h"
#include <algorithm>
#include "vector_tiling.h"
#include "error_log.h"
#include "../fusion_pass/common/fp16_t.hpp"
#include "op_tiling_util.h"

namespace optiling {

struct BnTrainingReduceGradCompileInfo {
  std::shared_ptr<AutoTilingHandler> tiling_handler;
  int64_t have_reduce_mean_cof_dtype;
};

bool BnTrainingReduceGradTiling(const std::string& op_type, const ge::Operator& op_paras,
                                const BnTrainingReduceGradCompileInfo& parsed_info, utils::OpRunInfo& run_info) {
  PROFILING_TILING_INIT(op_type.c_str());
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "parsed_info.tiling_handler nullptr, error!"),
                  return false);
  bool ret = parsed_info.tiling_handler->DoTiling(op_paras, run_info);
  if (!ret) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "bn_training_reduce_grad tiling failed.");
    return false;
  }
  PROFILING_TILING_AFTER_GET_SHAPE_REG();
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get op_info failed."),
                  return false);

  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_desc failed."),
                  return false);

  const std::vector<int64_t>& input_x_shapes = input_desc->MutableShape().GetDims();
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  float reduce_mean_cof = 1.0;
  int64_t num = input_x_shapes[0] * input_x_shapes[2] * input_x_shapes[3];
  if (num == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "bn_training_reduce_grad invalid dim value 0. (%ld,%ld,%ld,%ld,%ld)",
                                    input_x_shapes[0], input_x_shapes[1], input_x_shapes[2], input_x_shapes[3],
                                    input_x_shapes[4]);
    return false;
  }
  reduce_mean_cof = reduce_mean_cof / num;
  PROFILING_TILING_AFTER_CALCU_TILING_REG();

  if (parsed_info.have_reduce_mean_cof_dtype) {
    run_info.AddTilingData(reduce_mean_cof);
    run_info.AddTilingData(-reduce_mean_cof);

    OP_LOGD(op_type, "bn_training_reduce_grad write tilingdata num_rec= %f", reduce_mean_cof);
  }
  PROFILING_TILING_END();

  return true;
}

static bool ParseJsonCompileInfo(const std::string& op_type, const nlohmann::json& compile_info,
                                 BnTrainingReduceGradCompileInfo& parsed_info) {
  parsed_info.tiling_handler = CreateAutoTilingHandler(op_type, PATTERN_BROADCAST, compile_info);
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CreateAutoTilingHandler return nullptr"),
                  return false);
  // get core_num value
  std::string dtype;
  parsed_info.have_reduce_mean_cof_dtype = false;
  OP_TILING_CHECK(!GetCompileValue(compile_info, "reduce_mean_cof_dtype", dtype),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ParseJsonCompileInfo, get reduce_mean_cof_dtype error"),
                  return false);
  parsed_info.have_reduce_mean_cof_dtype = true;
  return true;
}

// register tiling interface of the bn_training_reduce_grad op.
REGISTER_OP_TILING_V3_CUSTOM(BNTrainingReduceGrad, BnTrainingReduceGradTiling, ParseJsonCompileInfo,
                             BnTrainingReduceGradCompileInfo);
}  // namespace optiling
