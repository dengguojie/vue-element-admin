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
 * \file conv_tiling.cpp
 * \brief
 */
#include <stdlib.h>

#include "conv_tiling.h"

namespace optiling {

int32_t ConvTiling(const std::vector<int32_t>& curShape, const std::string& dynamicMode, const nlohmann::json& opInfo,
                   OpRunInfo& runInfo) {
  if (curShape.empty()) {
    return 0;
  }
  std::string tilingID("0");
  if (dynamicMode == "dynamic_hw") {
    int32_t h = curShape[0];
    int32_t w = curShape[1];
    auto& tilingSeeds = opInfo.at("repo_seeds");
    auto& repoRange = opInfo.at("repo_range");
    int32_t minDist = 1000000;
    for (auto it = tilingSeeds.begin(); it != tilingSeeds.end(); it++) {
      std::vector<int32_t> seed = it.value().get<std::vector<int32_t>>();
      auto& range = repoRange[it.key()];
      if (h >= range[0] && h <= range[1] && w >= range[2] && w <= range[3]) {
        int32_t dist = abs(curShape[0] - seed[0]) + abs(curShape[1] - seed[1]);
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
        if (h >= range[0] && h <= range[1] && w >= range[2] && w <= range[3]) {
          tilingID = it.key();
          break;
        }
      }
    }
  } else if (dynamicMode == "dynamic_batch") {
    int32_t curB = curShape[0];
    auto& tilingCase = opInfo.at("tiling_range");
    for (auto it = tilingCase.begin(); it != tilingCase.end(); it++) {
      auto& range = it.value();
      if (curB >= range[0] && curB <= range[1]) {
        tilingID = it.key();
      }
    }
  }
  if (tilingID != "0") {
    runInfo.block_dim = (uint32_t)opInfo["block_dim"][tilingID];
  }
  return std::stoi(tilingID);
}

}  // namespace optiling
