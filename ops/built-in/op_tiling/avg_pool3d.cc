/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file avg_pool3d.cc
 * \brief tiling function of avg_pool3d
 */
#include <string>
#include <nlohmann/json.hpp>
#include "cube_tiling.h"
#include "op_log.h"
#include "op_tiling.h"
#include "../op_proto/util/error_util.h"

namespace {
  constexpr int32_t SHAPE_SIZE_6D = 6;
}

namespace optiling {
/*
 * @brief: tiling function of avg_pool3d
 * @param [in] op_type: op_type of the avg_pool3d
 * @param [in] op_paras: inputs/outputs/atts of the avg_pool3d
 * @param [in] compile_info: compile time generated info of the avg_pool3d
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool AvgPool3DTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& compile_info,
                     OpRunInfo& run_info) {
  if (op_paras.inputs.empty() || op_paras.outputs.empty() || op_paras.inputs[0].tensor.empty() ||
      op_paras.outputs[0].tensor.empty() || (op_paras.inputs[0].tensor[0].shape.size() < SHAPE_SIZE_6D) ||
      op_paras.outputs[0].tensor[0].shape.size() < SHAPE_SIZE_6D) {
    CUBE_INNER_ERR_REPORT(op_type.c_str(), "invalid inputs' shape dims.");
    return false;
  }
  return Conv3DCommonTiling("AvgPool3D", op_paras.inputs[0].tensor[0].shape,
                            op_paras.outputs[0].tensor[0].shape,
                            compile_info, run_info);
}

REGISTER_OP_TILING_FUNC_BUFFERED(AvgPool3D, AvgPool3DTiling);
}  // namespace optiling
