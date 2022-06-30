/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef AIR_COMPILER_GRAPH_COMPILER_ENGINES_NNENG_INC_COMMON_CMO_ID_GEN_STRATEGY_H_
#define AIR_COMPILER_GRAPH_COMPILER_ENGINES_NNENG_INC_COMMON_CMO_ID_GEN_STRATEGY_H_

#include <mutex>
#include "graph/node.h"
#include "graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
class CMOIdGenStrategy {
 public:
  CMOIdGenStrategy();
  ~CMOIdGenStrategy();

  CMOIdGenStrategy(const CMOIdGenStrategy &) = delete;
  CMOIdGenStrategy &operator=(const CMOIdGenStrategy &) = delete;

  static CMOIdGenStrategy& Instance();
  Status Finalize();
  uint16_t GenerateCMOId(const ge::Node &node);
  void UpdateReuseMap(const int64_t &topo_id, const uint16_t &cmo_id);

 private:

  uint32_t GetAtomicId() const;

  std::mutex cmo_id_mutex_;
  std::unordered_map<uint16_t, int64_t> reuse_cmo_id_map_;
};
}  // namespace fe
#endif  // AIR_COMPILER_GRAPH_COMPILER_ENGINES_NNENG_INC_COMMON_CMO_ID_GEN_STRATEGY_H_
