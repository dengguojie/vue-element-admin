/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
bool Conv3DTiling(const std::string& opType,
                  const ge::Operator& opParas,
                  const nlohmann::json& opCompileInfo,
                  utils::OpRunInfo& runInfo) {
  if (opParas.GetInputsSize() == 0 ||
      opParas.GetOutputsSize() == 0 ||
      (opParas.GetInputDesc(0).GetShape().GetDimNum() < SHAPE_SIZE_6HD) ||
      (opParas.GetOutputDesc(0).GetShape().GetDimNum() < SHAPE_SIZE_6HD)) {
    return false;
  }

  if (opCompileInfo.empty()) {
    CUBE_INNER_ERR_REPORT(opType.c_str(), "op compile info is empty");
    return false;
  }

  // the dim index of input x channel is 2
  if (opCompileInfo.contains("fmap_c1") && opParas.GetInputDesc(0).GetShape().GetDim(2) != opCompileInfo["fmap_c1"]) {
    CUBE_INNER_ERR_REPORT(opType.c_str(), "not support, input x channel should be equal to filter * groups");
    return false;
  }

  nlohmann::json opInfo;
  deal_with_compile_info(opCompileInfo, opInfo);

  return Conv3DCommonTiling("Conv3D",
                            opParas.GetInputDesc(0).GetShape().GetDims(),
                            opParas.GetOutputDesc(0).GetShape().GetDims(),
                            opInfo,
                            runInfo);
}

// register tiling interface of the conv3d
REGISTER_OP_TILING_FUNC_BUFFERED_V2(Conv3D, Conv3DTiling);
}  // namespace optiling
