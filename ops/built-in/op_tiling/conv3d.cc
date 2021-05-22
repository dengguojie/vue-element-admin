/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file conv3d.cpp
 * \brief tiling function of conv3d
 */
#include <string>
#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"
#include "cube_tiling.h"
#include "op_log.h"
#include "../op_proto/util/error_util.h"

namespace {
  const unsigned int SHAPE_SIZE_6HD = 6;
}

namespace optiling {
/*
 * @brief: tiling function of conv3d
 * @param [in] op_type: op_type of the conv3d
 * @param [in] op_paras: inputs/outputs/atts of the conv3d
 * @param [in] op_compile_info: compile time generated info of the conv3d
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool Conv3DTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                  OpRunInfo& runInfo) {
  if (opParas.inputs.empty() || opParas.outputs.empty() || opParas.inputs[0].tensor.empty() ||
      opParas.outputs[0].tensor.empty() || (opParas.inputs[0].tensor[0].shape.size() < SHAPE_SIZE_6HD) ||
      (opParas.outputs[0].tensor[0].shape.size() < SHAPE_SIZE_6HD)) {
    return false;
  }

  if (opCompileInfo.empty()) {
    CUBE_INNER_ERR_REPORT(opType.c_str(), "op compile info is empty");
    return false;
  }

  if (opCompileInfo.contains("fmap_c1") && opParas.inputs[0].tensor[0].shape[2] != opCompileInfo["fmap_c1"]) {
    CUBE_INNER_ERR_REPORT(opType.c_str(), "not support, input x channel should be equal to filter * groups");
    return false;
  }

  nlohmann::json opInfo;
  deal_with_compile_info(opCompileInfo, opInfo);

  return Conv3DCommonTiling("Conv3D", opParas.inputs[0].tensor[0].shape,
                            opParas.outputs[0].tensor[0].shape,
                            opInfo, runInfo);
}

// register tiling interface of the conv3d
REGISTER_OP_TILING_FUNC_BUFFERED(Conv3D, Conv3DTiling);
}  // namespace optiling