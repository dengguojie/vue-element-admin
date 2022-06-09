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
#include <algorithm>
#include "vector_tiling.h"
#include "error_log.h"
#include "../fusion_pass/common/fp16_t.hpp"
#include "op_tiling_util.h"

namespace optiling {
const int64_t INDEX_0 = 0;
const int64_t INDEX_1 = 1;
const int64_t INDEX_2 = 2;
const int64_t INDEX_3 = 3;
const int64_t INDEX_4 = 4;
struct BnTrainingReduceGradCompileInfo {
  std::shared_ptr<AutoTilingHandler> tiling_handler;
};

bool BnTrainingReduceGradTiling(const std::string& op_type, const ge::Operator& op_paras,
                                const BnTrainingReduceGradCompileInfo& parsed_info, utils::OpRunInfo& run_info) {
  PROFILING_TILING_INIT(op_type.c_str());
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "parsed_info.tiling_handler nullptr, error!"), return false);
  bool ret = parsed_info.tiling_handler->DoTiling(op_paras, run_info);
  if (!ret) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "training_reduce_grad tiling failed.");
    return false;
  }
  PROFILING_TILING_AFTER_GET_SHAPE_REG();
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get op_info failed."),
                  return false);

  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_desc failed."),
                  return false);

  const GeShape& input_x_shapes = input_desc->MutableShape();
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  int64_t num = 1;
  float reduce_mean_cof = 1.0;
  ge::Format input_x_format = input_desc->GetFormat();
  if (input_x_format == FORMAT_NDC1HWC0) {
    num = input_x_shapes.GetDim(INDEX_0) * input_x_shapes.GetDim(INDEX_1) *
          input_x_shapes.GetDim(INDEX_3) * input_x_shapes.GetDim(INDEX_4);
    if (op_type == "INTrainingReduceGrad") {
      num = input_x_shapes.GetDim(INDEX_1) * input_x_shapes.GetDim(INDEX_3) * input_x_shapes.GetDim(INDEX_4);
    }
  } else {
    num = input_x_shapes.GetDim(INDEX_0) * input_x_shapes.GetDim(INDEX_2) *
          input_x_shapes.GetDim(INDEX_3);
    if (op_type == "INTrainingReduceGrad") {
      num = input_x_shapes.GetDim(INDEX_3) * input_x_shapes.GetDim(INDEX_4);
    }
  }

  if (num == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "training_reduce_grad invalid dim value 0. (%s)",
                                    input_x_shapes.ToString().c_str());
    return false;
  }
  reduce_mean_cof = reduce_mean_cof / num;
  PROFILING_TILING_AFTER_CALCU_TILING_REG();

  run_info.AddTilingData(reduce_mean_cof);
  run_info.AddTilingData(-reduce_mean_cof);

  OP_LOGD(op_type, "bn_training_reduce_grad write tilingdata num_rec= %f", reduce_mean_cof);

  PROFILING_TILING_END();

  return true;
}

static bool ParseJsonCompileInfo(const std::string& op_type, const nlohmann::json& compile_info,
                                 BnTrainingReduceGradCompileInfo& parsed_info) {
  parsed_info.tiling_handler = CreateAutoTilingHandler(op_type, PATTERN_BROADCAST, compile_info);
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CreateAutoTilingHandler return nullptr"), return false);
  return true;
}

// register tiling interface of the bn_training_reduce_grad op.
REGISTER_OP_TILING_V3_CUSTOM(BNTrainingReduceGrad, BnTrainingReduceGradTiling, ParseJsonCompileInfo,
                             BnTrainingReduceGradCompileInfo);
REGISTER_OP_TILING_V3_CUSTOM(BN3DTrainingReduceGrad, BnTrainingReduceGradTiling, ParseJsonCompileInfo,
                             BnTrainingReduceGradCompileInfo);
REGISTER_OP_TILING_V3_CUSTOM(INTrainingReduceGrad, BnTrainingReduceGradTiling, ParseJsonCompileInfo,
                             BnTrainingReduceGradCompileInfo);
}  // namespace optiling
