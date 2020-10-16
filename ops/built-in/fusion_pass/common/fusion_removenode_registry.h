/**
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @brief all remove node pass.
 *
 * @version 1.0
 *
 */

#ifndef FUSION_REMOVE_NODE_REGISTRY_H_
#define FUSION_REMOVE_NODE_REGISTRY_H_

#include <cstdio>
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
  explicit FusionRemoveNodeRegister(std::string opType)
          : opType_(opType), preCheckFunc_(nullptr) {};
  FusionRemoveNodeRegister() = default;

  virtual ~FusionRemoveNodeRegister() = default;

  FusionRemoveNodeRegister &AddLinkPair(uint32_t inAnchorIndex,
                                        uint32_t outAnchorIndex);
  FusionRemoveNodeRegister &SetPreCheckFunc(
          std::function<bool(ge::NodePtr)> preCheckFunc);
  void GetParameters(std::vector<LinkIndexPair> &linkPairVec,
                     std::function<bool(ge::NodePtr)> &preCheckFunc);
  const std::string& GetOpType() const { return opType_; }

private:
  std::string opType_;
  std::vector<LinkIndexPair> linkPairVec_;
  std::function<bool(ge::NodePtr)> preCheckFunc_;
};

class FusionRemoveNodeRegistry {
 public:
  FusionRemoveNodeRegistry() = default;
  virtual ~FusionRemoveNodeRegistry() = default;

  static FusionRemoveNodeRegistry *Instance();

  Status SetRemoveNodeRegister(const std::string &opType,
                               FusionRemoveNodeRegister &reg);

  Status GetRegisterByOpType(const std::string &opType,
                             FusionRemoveNodeRegister &reg);

 private:
  std::map<std::string, FusionRemoveNodeRegister> removeNodeMap_;
};

class RemoveNodeReciever {
 public:
  RemoveNodeReciever(FusionRemoveNodeRegister &reg);
  virtual ~RemoveNodeReciever() = default;
};

#define REGISTER_REMOVENODE(opType) \
  REGISTER_REMOVENODE_UNIQ_HELPER(__COUNTER__, opType)
#define REGISTER_REMOVENODE_UNIQ_HELPER(ctr, name) \
  REGISTER_REMOVENODE_UNIQ(ctr, name)
#define REGISTER_REMOVENODE_UNIQ(ctr, opType)            \
  static RemoveNodeReciever register_removenode##ctr \
      __attribute__((unused)) = FusionRemoveNodeRegister(opType)
}  // namespace fe
#endif  // FUSION_REMOVE_NODE_REGISTRY_H_
