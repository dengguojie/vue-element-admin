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
 * \file fusion_removenode_registry.h
 * \brief all remove node pass.
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_COMMON_FUSION_REMOVENODE_REGISTRY_H_
#define OPS_BUILT_IN_FUSION_PASS_COMMON_FUSION_REMOVENODE_REGISTRY_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <functional>
#include "graph/utils/node_utils.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
using std::function;

struct LinkIndexPair {
  uint32_t inAnchorIndex;
  uint32_t outAnchorIndex;
};

class FusionRemoveNodeRegister {
 public:
  explicit FusionRemoveNodeRegister(std::string opType) : opType_(opType), preCheckFunc_(nullptr){};
  FusionRemoveNodeRegister() = default;

  virtual ~FusionRemoveNodeRegister() = default;

  FusionRemoveNodeRegister& AddLinkPair(uint32_t inAnchorIndex, uint32_t outAnchorIndex);
  FusionRemoveNodeRegister& SetPreCheckFunc(std::function<Status(ge::NodePtr)> preCheckFunc);
  void GetParameters(std::vector<LinkIndexPair>& linkPairVec, std::function<Status(ge::NodePtr)>& preCheckFunc);
  const std::string& GetOpType() const {
    return opType_;
  }

 private:
  std::string opType_;
  std::vector<LinkIndexPair> linkPairVec_;
  std::function<Status(ge::NodePtr)> preCheckFunc_;
};

class FusionRemoveNodeRegistry {
 public:
  FusionRemoveNodeRegistry() = default;
  virtual ~FusionRemoveNodeRegistry() = default;

  static FusionRemoveNodeRegistry* Instance();

  Status SetRemoveNodeRegister(const std::string& opType, const FusionRemoveNodeRegister& reg);

  Status GetRegisterByOpType(const std::string& opType, FusionRemoveNodeRegister& reg) const;

 private:
  std::map<std::string, FusionRemoveNodeRegister> removeNodeMap_;
};

class RemoveNodeReciever {
 public:
  RemoveNodeReciever(FusionRemoveNodeRegister& reg_data);
  virtual ~RemoveNodeReciever() = default;
};

#define REGISTER_REMOVENODE(opType) REGISTER_REMOVENODE_UNIQ_HELPER(__COUNTER__, opType)
#define REGISTER_REMOVENODE_UNIQ_HELPER(ctr, name) REGISTER_REMOVENODE_UNIQ(ctr, name)
#define REGISTER_REMOVENODE_UNIQ(ctr, opType) \
  static RemoveNodeReciever register_removenode##ctr __attribute__((unused)) = FusionRemoveNodeRegister(opType)
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_COMMON_FUSION_REMOVENODE_REGISTRY_H_
