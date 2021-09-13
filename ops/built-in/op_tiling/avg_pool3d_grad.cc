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
 * \file avg_pool3d_grad.cc
 * \brief tiling function of avg_pool3d_grad
 */
#include <string>
#include <nlohmann/json.hpp>
#include <limits>
#include "cube_tiling.h"
#include "op_log.h"
#include "op_tiling.h"
#include "error_log.h"

namespace {
  constexpr int32_t kAvgPool3DGradDimLimit = 6;
  constexpr int32_t kAvgPool3DGradDedyInputIdx = 1;
}

namespace optiling {
/*
 * @brief: tiling function of avg_pool3d_grad
 * @param [in] op_type: op_type of the avg_pool3d_grad
 * @param [in] op_paras: inputs/outputs/atts of the avg_pool3d_grad
 * @param [in] compile_info: compile time generated info of the avg_pool3d_grad
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool AvgPool3DGradTiling(const std::string& op_type,
                         const ge::Operator& op_paras,
                         const nlohmann::json& compile_info,
                         utils::OpRunInfo& run_info) {
  if ((op_paras.GetInputsSize() <= kAvgPool3DGradDedyInputIdx) ||
      (op_paras.GetInputDesc(kAvgPool3DGradDedyInputIdx).GetShape().GetDimNum() != kAvgPool3DGradDimLimit) ||
      (op_paras.GetOutputDesc(0).GetShape().GetDimNum() != kAvgPool3DGradDimLimit)) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "param check failed");
    return false;
  }

  return Conv3DCommonTiling("AvgPool3DGrad",
                            op_paras.GetOutputDesc(0).GetShape().GetDims(),
                            op_paras.GetInputDesc(kAvgPool3DGradDedyInputIdx).GetShape().GetDims(),
                            compile_info,
                            run_info);
}

REGISTER_OP_TILING_FUNC_BUFFERED_V2(AvgPool3DGrad, AvgPool3DGradTiling);
}  // namespace optiling
