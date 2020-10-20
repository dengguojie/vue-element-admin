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
#include "conv_tiling.h"

namespace optiling {

int32_t ConvTiling(const std::vector<int32_t>& curShape, const std::string& dynamicMode, const nlohmann::json& opInfo,
                   OpRunInfo& runInfo) {
  if (curShape.empty()) {
    return 0;
  }
  int32_t tilingID = 0;
  if (dynamicMode == "dynamic_hw") {
    int32_t h = curShape[0];
    int32_t w = curShape[1];
    auto& tilingSeeds = opInfo.at("repo_seeds");
    for (auto it = tilingSeeds.begin(); it != tilingSeeds.end(); it++) {
      auto& seed = it.value();
      if (h == seed[0] && w == seed[1]) {
        tilingID = std::stoi(it.key());
        runInfo.block_dim = (uint32_t)opInfo["block_dim"][it.key()];
      }
    }
    if (tilingID == 0) {
      auto& tilingCases = opInfo.at("tiling_range");
      for (auto it = tilingCases.begin(); it != tilingCases.end(); it++) {
        auto& ranges = it.value();
        for (auto itRange = ranges.begin(); itRange != ranges.end(); itRange++) {
          if (h >= (*itRange)[0] && h <= (*itRange)[1] && w >= (*itRange)[2] && w <= (*itRange)[3]) {
            tilingID = std::stoi(it.key());
            runInfo.block_dim = (uint32_t)opInfo["block_dim"][it.key()];
          }
        }
      }
    }
  }
  if (dynamicMode == "dynamic_batch") {
    int32_t curB = curShape[0];
    auto& tilingCase = opInfo.at("tiling_range");
    for (auto it = tilingCase.begin(); it != tilingCase.end(); it++) {
      auto& range = it.value();
      if (curB >= range[0] && curB <= range[1]) {
        tilingID = std::stoi(it.key());
        runInfo.block_dim = (uint32_t)opInfo["block_dim"][it.key()];
      }
    }
  }
  return tilingID;
}

}  // namespace optiling
