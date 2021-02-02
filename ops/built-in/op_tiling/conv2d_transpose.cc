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
 * \file conv2d_transpose.cc
 * \brief tiling function of conv2d_transpose
 */
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"
#include "cube_tiling.h"
#include "op_log.h"

namespace optiling {
/*
 * @brief: tiling function of conv2d_transpose
 * @param [in] op_type: op_type of conv2d_transpose
 * @param [in] op_paras: inputs/outputs/atts of conv2d_transpose
 * @param [in] op_compile_info: compile time generated info of conv2d_transpose
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool Conv2DTransposeTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                         OpRunInfo& runInfo) {
  int32_t nDim = 0;
  int32_t hDim = 2;
  int32_t wDim = 3;

  if (opParas.inputs.empty() || opParas.outputs.empty() || opParas.inputs.size() < 2 ||
      opParas.inputs[1].tensor.empty() ||  opParas.outputs[0].tensor.empty() ||
      opParas.inputs[1].tensor[0].shape.empty() || opParas.inputs[1].tensor[0].shape.size() < 4 ||
      opParas.outputs[0].tensor[0].shape.empty() || opParas.outputs[0].tensor[0].shape.size() < 4) {
    return false;
  }

  GELOGD("Current format is %s, Ori format is %s", opParas.outputs[0].tensor[0].format.c_str(),
         opParas.outputs[0].tensor[0].ori_format.c_str());

  int32_t n =  opParas.outputs[0].tensor[0].shape[nDim];
  int32_t dedyH = opParas.inputs[1].tensor[0].shape[hDim];
  int32_t dedyW = opParas.inputs[1].tensor[0].shape[wDim];
  int32_t dxH = opParas.outputs[0].tensor[0].shape[hDim];
  int32_t dxW = opParas.outputs[0].tensor[0].shape[wDim];

  int32_t tilingID = CubeTiling(opType, {n, dxH, dxW}, opCompileInfo, runInfo);
  GELOGD("tiling_data is %d, %d, %d, %d, %d, %d", tilingID, n, dxH, dxW, dedyH, dedyW);
  try {
    runInfo.tiling_key = tilingID;
    int status = opCompileInfo["push_status"];
    if (status == 0) {
      ByteBufferPut(runInfo.tiling_data, tilingID);
    }
  } catch (const std::exception &e) {
    GE_LOGE("op [%s]: get push_status error. Error message: %s", opType.c_str(), e.what());
    return false;
  }
  std::vector<std::string> varMap = opCompileInfo.at("_vars")["10000"];

  if (std::find(varMap.begin(), varMap.end(), "batch_n") != varMap.end()) {
    ByteBufferPut(runInfo.tiling_data, n);
  }
  if (std::find(varMap.begin(), varMap.end(), "dx_h") != varMap.end()) {
    ByteBufferPut(runInfo.tiling_data, dedyH);
    ByteBufferPut(runInfo.tiling_data, dxH);
  }
  if (std::find(varMap.begin(), varMap.end(), "dx_w") != varMap.end()) {
    ByteBufferPut(runInfo.tiling_data, dedyW);
    ByteBufferPut(runInfo.tiling_data, dxW);
  }

  return true;
}

// register tiling interface of the conv2d_transpose
REGISTER_OP_TILING_FUNC_BUFFERED(Conv2DTranspose, Conv2DTransposeTiling);
}  // namespace optiling
