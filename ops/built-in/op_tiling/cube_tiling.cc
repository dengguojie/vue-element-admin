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
 * \file cube_tiling.cpp
 * \brief
 */
#include <stdlib.h>
#include <string>
#include "cube_tiling.h"

namespace optiling {

int32_t g_nDim = 0;
int32_t g_hDim = 1;
int32_t g_wDim = 2;
int32_t g_nMinDim = 0;
int32_t g_nMaxDim = 1;
int32_t g_hMinDim = 2;
int32_t g_hMaxDim = 3;
int32_t g_wMinDim = 4;
int32_t g_wMaxDim = 5;

string CubeTilingBatch(const std::vector<int32_t>& curShape, const nlohmann::json& opInfo) {
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

string CubeTilingCostModel(const std::vector<int32_t>& curShape, const nlohmann::json& opInfo) {
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

string CubeTilingNHW(const std::vector<int32_t>& curShape, const nlohmann::json& opInfo) {
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

  tilingID = CubeTilingCostModel(curShape, opInfo);
  return tilingID;
}


int32_t CubeTiling(const std::string& opType, const std::vector<int32_t>& curShape, const nlohmann::json& opInfo,
                   OpRunInfo& runInfo) {
    std::vector<std::string> varMap = opInfo.at("_vars")["10000"];
    std::string tilingID("0");

    if (opInfo["tiling_type"] == "default_tiling") {
        tilingID = opInfo["default_range"].begin().key();
    } else if (varMap.size() != 1) {
        tilingID = CubeTilingNHW(curShape, opInfo);
    } else {
        tilingID = CubeTilingBatch(curShape, opInfo);
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
}  // namespace optiling
