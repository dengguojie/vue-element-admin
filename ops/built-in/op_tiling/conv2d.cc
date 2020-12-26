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
 * \file conv2d.cpp
 * \brief tiling function of conv2d
 */
#include <string>
#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"

using namespace std;
namespace optiling {
/*
 * @brief: tiling function of conv2d
 * @param [in] op_type: op_type of the conv2d
 * @param [in] op_paras: inputs/outputs/atts of the conv2d
 * @param [in] op_compile_info: compile time generated info of the conv2d
 * @param [out] run_info: result data
 * @return bool: success or not
 */
int32_t g_nDim = 0;
int32_t g_hDim = 1;
int32_t g_wDim = 3;
int32_t g_nMinDim = 0;
int32_t g_nMaxDim = 1;
int32_t g_hMinDim = 2;
int32_t g_hMaxDim = 3;
int32_t g_wMinDim = 4;
int32_t g_wMaxDim = 5;

string ConvTilingBatch(const std::vector<int32_t>& curShape, const nlohmann::json& opInfo) {
  std::string tilingID("0");
  auto& tilingRange = opInfo.at("tiling_range");
  for (auto it = tilingRange.begin(); it != tilingRange.end(); it++) {
    auto& range = it.value();
    if (curShape[g_nDim] >= range[g_nMinDim] && curShape[g_nDim] <= range[g_nMaxDim]) {
      tilingID = it.key();
    }
  }
  return tilingID;
}

string ConvTilingCostModel(const std::vector<int32_t>& curShape, const nlohmann::json& opInfo) {
  std::string tilingID("0");
  auto& costRange = opInfo.at("cost_range");
  for (auto it = costRange.begin(); it != costRange.end(); it++) {
    auto& range = it.value();

    if (curShape[g_nDim] >= range[g_nMinDim] && curShape[g_nDim] <= range[g_nMaxDim] &&
        curShape[g_hDim] >= range[g_hMinDim] && curShape[g_hDim] <= range[g_hMaxDim] &&
        curShape[g_wDim] >= range[g_wMinDim] && curShape[g_wDim] <= range[g_wMaxDim]) {
      tilingID = it.key();
    }
  }
  return tilingID;
}

string ConvTilingNHW(const std::vector<int32_t>& curShape, const nlohmann::json& opInfo) {
  int32_t seedHDim = 0;
  int32_t seedWDim = 1;
  int32_t minDist = 1000000;

  std::string tilingID("0");
  auto& repoRange = opInfo.at("repo_range");
  auto& tilingSeeds = opInfo.at("repo_seeds");

  for (auto it = tilingSeeds.begin(); it != tilingSeeds.end(); it++) {
    std::vector<int32_t> seed = it.value().get<std::vector<int32_t>>();
    auto& range = repoRange[it.key()];
    if (curShape[g_nDim] >= range[g_nMinDim] && curShape[g_nDim] <= range[g_nMaxDim] &&
        curShape[g_hDim] >= range[g_hMinDim] && curShape[g_hDim] <= range[g_hMaxDim] && 
        curShape[g_wDim] >= range[g_wMinDim] && curShape[g_wDim] <= range[g_wMaxDim]) {
        int32_t dist = abs(curShape[g_hDim] - seed[seedHDim]) + abs(curShape[g_wDim] - seed[seedWDim]);
        if (dist < minDist) {
          tilingID = it.key();
          minDist = dist;
        }
    }
  }
  if (tilingID != "0") {
    return tilingID;
  }

  tilingID = ConvTilingCostModel(curShape, opInfo);
  return tilingID;
}

int32_t ConvTiling(const std::string& opType, const std::vector<int32_t>& curShape, 
                   const nlohmann::json& opInfo, OpRunInfo& runInfo) {
  std::vector<std::string> varMap = opInfo.at("_vars")["10000"];
  std::string tilingID("0");

  if (varMap.size() != 1) {
    tilingID = ConvTilingNHW(curShape, opInfo);
  } else {
    tilingID = ConvTilingBatch(curShape, opInfo);
  }
  if (tilingID == "0") {
    OP_LOGE(opType.c_str(),
            "This shape is not covered by any tiling, "
            "please modify range and recompile");
    return false;
  }

  runInfo.block_dim = (uint32_t)opInfo["block_dim"][tilingID];
  return std::stoi(tilingID);
}

bool Conv2DTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                  OpRunInfo& runInfo) {
  int32_t hDim = 2;
  int32_t wDim = 3;

  if (opParas.inputs.empty() || opParas.outputs.empty() || opParas.inputs[0].tensor.empty() ||
      opParas.outputs[0].tensor.empty() || opParas.inputs[0].tensor[0].shape.empty() ||
      opParas.outputs[0].tensor[0].shape.empty()) {
    return false;
  }

  GELOGD("Current format is %s, Ori format is %s", opParas.inputs[0].tensor[0].format.c_str(),
         opParas.inputs[0].tensor[0].ori_format.c_str());

  int32_t n = opParas.inputs[0].tensor[0].shape[g_nDim];
  int32_t h = opParas.inputs[0].tensor[0].shape[hDim];
  int32_t w = opParas.inputs[0].tensor[0].shape[wDim];
  int32_t outH = opParas.outputs[0].tensor[0].shape[hDim];
  int32_t outW = opParas.outputs[0].tensor[0].shape[wDim];

  int32_t tilingID = ConvTiling(opType, {n, h, outH, w, outW}, opCompileInfo, runInfo);
  
  GELOGD("tiling _data is %d, %d, %d, %d, %d, %d", tilingID, n, h, w, outH, outW);

  ByteBufferPut(runInfo.tiling_data, tilingID);
  std::vector<std::string> varMap = opCompileInfo.at("_vars")["10000"];

  if (std::find(varMap.begin(), varMap.end(), "batch_n") != varMap.end()) {
    ByteBufferPut(runInfo.tiling_data, n);
  }
  if (std::find(varMap.begin(), varMap.end(), "fmap_h") != varMap.end()) {
    ByteBufferPut(runInfo.tiling_data, h);
    ByteBufferPut(runInfo.tiling_data, outH);
  }
  if (std::find(varMap.begin(), varMap.end(), "fmap_w") != varMap.end()) {
    ByteBufferPut(runInfo.tiling_data, w);
    ByteBufferPut(runInfo.tiling_data, outW);
  }

  return true;
}

// register tiling interface of the conv2d
REGISTER_OP_TILING_FUNC_BUFFERED(Conv2D, Conv2DTiling);
}  // namespace optiling
