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
#include "register/op_tiling.h"
#include "graph/debug/ge_log.h"
#include "conv_tiling.h"
#include "op_log.h"

namespace {
  inline bool IsShapeInRange(int32_t d, int32_t h, int32_t w, const std::vector<int32_t> &range) {
    if (range.empty()) {
      return false;
    }

    return d >= range[0] && d <= range[1] && h >= range[2] && h <= range[3] && w >= range[4] && w <= range[5];
  }

  int32_t GetDynamicDHWTiling(const std::vector<int32_t>& curShape, const std::string& dynamicMode,
                              const nlohmann::json& opInfo, optiling::OpRunInfo& runInfo) {
    if (curShape.size() < 3) {
      return 0;
    }

    std::string tilingID("0");
    int32_t d = curShape[0];
    int32_t h = curShape[1];
    int32_t w = curShape[2];
    auto& tilingSeeds = opInfo.at("repo_seeds");
    auto& repoRange = opInfo.at("repo_range");
    int32_t minDist = 1000000;
    for (auto it = tilingSeeds.begin(); it != tilingSeeds.end(); it++) {
      std::vector<int32_t> seed = it.value().get<std::vector<int32_t>>();
      auto& range = repoRange[it.key()];
      if (IsShapeInRange(d, h, w, range)) {
        int32_t dist = abs(curShape[0] - seed[0]) + abs(curShape[1] - seed[1]) + abs(curShape[2] - seed[2]);
        if (dist < minDist) {
          tilingID = it.key();
          minDist = dist;
        }
      }
    }

    if (tilingID == "0") {
      auto& costRange = opInfo.at("cost_range");
      for (auto it = costRange.begin(); it != costRange.end(); it++) {
        auto& range = it.value();
        if (IsShapeInRange(d, h, w, range)) {
          tilingID = it.key();
          break;
        }
      }
    }

    if (tilingID != "0") {
      runInfo.block_dim = (uint32_t)opInfo["block_dim"][tilingID];
    }
    return std::stoi(tilingID);
  }
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
      opParas.outputs[0].tensor.empty() || (opParas.inputs[0].tensor[0].shape.size() < 3) ||
      (opParas.outputs[0].tensor[0].shape.size() < 3)) {
    return false;
  }
  std::string mode = opCompileInfo["dynamic_mode"].get<std::string>();
  GELOGD("dynamic_mode is [%s]", mode.c_str());
  GELOGD("Current format is %s, Ori format is %s", opParas.inputs[0].tensor[0].format.c_str(),
         opParas.inputs[0].tensor[0].ori_format.c_str());

  int32_t tilingId = 0;
  if (mode == "dynamic_dhw") {
    int32_t d = opParas.inputs[0].tensor[0].shape[1];
    int32_t h = opParas.inputs[0].tensor[0].shape[3];
    int32_t w = opParas.inputs[0].tensor[0].shape[4];
    int32_t outD = opParas.outputs[0].tensor[0].shape[1];
    int32_t outH = opParas.outputs[0].tensor[0].shape[3];
    int32_t outW = opParas.outputs[0].tensor[0].shape[4];
    tilingId = GetDynamicDHWTiling({d, h, w}, mode, opCompileInfo, runInfo);
    ByteBufferPut(runInfo.tiling_data, tilingId);
    ByteBufferPut(runInfo.tiling_data, d);
    ByteBufferPut(runInfo.tiling_data, h);
    ByteBufferPut(runInfo.tiling_data, w);
    ByteBufferPut(runInfo.tiling_data, outD);
    ByteBufferPut(runInfo.tiling_data, outH);
    ByteBufferPut(runInfo.tiling_data, outW);

    GELOGD("tiling_data is %d, %d, %d, %d, %d, %d, %d", tilingId, d, h, w, outD, outH, outW);
  } else if (mode == "dynamic_batch") {
    int32_t batch = opParas.inputs[0].tensor[0].shape[0];
    tilingId = ConvTiling({batch}, mode, opCompileInfo, runInfo);
    ByteBufferPut(runInfo.tiling_data, tilingId);
    ByteBufferPut(runInfo.tiling_data, batch);

    GELOGD("Input info is %d, %d", tilingId, batch);
  } else {
    OP_LOGE(opType.c_str(), "op ScatterAddTiling is not supported");
    return false;
  }

  if (tilingId == 0) {
    OP_LOGE(opType.c_str(),
            "This shape is not covered by any tiling, "
            "please modify range and recompile");
    return false;
  }
  return true;
}

// register tiling interface of the conv3d
REGISTER_OP_TILING_FUNC(Conv3D, Conv3DTiling);
}  // namespace optiling
