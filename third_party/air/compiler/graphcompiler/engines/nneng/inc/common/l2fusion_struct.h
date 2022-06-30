/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef FUSION_ENGINE_INC_COMMON_L2FUSION_STRUCT_H_
#define FUSION_ENGINE_INC_COMMON_L2FUSION_STRUCT_H_

#include <map>
#include <string>
#include "runtime/kernel.h"

namespace fe {
struct L2Data {
  uint32_t l2Index;
  uint64_t l2Addr;
  uint64_t l2PageNum;
};

using L2DataMap = std::map<uint64_t, L2Data>;   // the key is ddr addr
using L2DataPair = std::pair<uint64_t, L2Data>;  // the key is ddr addr

struct TaskL2Info {
  std::string nodeName;
  rtL2Ctrl_t l2ctrl;

  L2DataMap input;
  L2DataMap output;
  uint32_t isUsed;
};

using TaskL2InfoMap = std::map<std::string, TaskL2Info>;    // the key is nodeName
using TaskL2InfoPair = std::pair<std::string, TaskL2Info>;  // the key is nodeName
}  // namespace fe
#endif  // FUSION_ENGINE_INC_COMMON_L2FUSION_STRUCT_H_
