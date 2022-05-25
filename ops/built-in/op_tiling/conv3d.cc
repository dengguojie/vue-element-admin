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
#include "error_util.h"

namespace {
  const unsigned int SHAPE_SIZE_6HD = 6;
  static const int kDimIndex = 2;
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
  GELOGD("Start Conv3DTiling process");
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
  int32_t input_channel_dim = opParas.GetInputDesc(0).GetShape().GetDim(kDimIndex);
  if (opCompileInfo.contains("fmap_c1")) {
    int32_t fmap_c1 = opCompileInfo["fmap_c1"];
    OP_LOGE_IF(input_channel_dim != fmap_c1, false, opType,
               "unsupported input, input x channel [%d] should be equal to filter * groups [%d]",
               input_channel_dim, fmap_c1);
  } else {
    OP_LOGE(opType, "compile_info does not contain the key value of the fmap_c1");
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
