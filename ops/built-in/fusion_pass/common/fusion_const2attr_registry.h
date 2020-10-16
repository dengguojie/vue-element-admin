/**
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief all const2attr pass.
 *
 * @version 1.0
 *
 */

#ifndef GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_FUSION_CONST2ATTR_REGISTRY_H_
#define GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_FUSION_CONST2ATTR_REGISTRY_H_

#include <cstdio>
#include <map>
#include <unordered_map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "pattern_fusion_util.h"

namespace fe {
class FusionConst2AttrOpRegister {
  friend class PatternFusionUtil;
  friend class OpConst2AttrOpReciever;
  friend class FusionConst2AttrOpRegistry;

 public:
  explicit FusionConst2AttrOpRegister(std::string opType)
      : opType_(std::move(opType)),
        needCheckSupported_(false){};
  FusionConst2AttrOpRegister() = default;

  virtual ~FusionConst2AttrOpRegister() = default;

  FusionConst2AttrOpRegister &OriginOpType(const std::string &oriOpType);
  FusionConst2AttrOpRegister &NeedCheckSupported(const bool needCheck);
  FusionConst2AttrOpRegister &SetConstToAttr(int attrIndex,
                                             const std::string &attrName,
                                             const std::string &attrType);
  FusionConst2AttrOpRegister &SetPreCheckFunc(std::function<bool(ge::NodePtr)> preCheckFunc);
  void GetAttrInfo(std::string &opType, bool &needCheckSupported,
                   std::vector<PassAttrInfo> &attrVec,
                   std::function<bool(ge::NodePtr)> &preCheckFunc);

 private:
  std::string opType_;
  std::string oriOpType_;
  bool needCheckSupported_;
  std::vector<PassAttrInfo> attrVec_;
  std::function<bool(ge::NodePtr)> preCheckFunc_{nullptr};
};

class FusionConst2AttrOpRegistry {
 public:
  FusionConst2AttrOpRegistry() = default;
  virtual ~FusionConst2AttrOpRegistry() = default;

  static FusionConst2AttrOpRegistry *Instance();

  Status SetConst2AttrRegister(const string &opType,
                               FusionConst2AttrOpRegister &const2AttrRegister);
  Status GetRegisterByOriType(const string &oriOpType,
                              FusionConst2AttrOpRegister &reg);

 private:
  std::unordered_map<string, FusionConst2AttrOpRegister> const2AttrOpMap_;
};

class OpConst2AttrOpReciever {
 public:
  OpConst2AttrOpReciever(FusionConst2AttrOpRegister &reg);
  virtual ~OpConst2AttrOpReciever() = default;
};

#define REGISTER_CONST2ATTR(opType) \
  REGISTER_CONST2ATTR_UNIQ_HELPER(__COUNTER__, opType)
#define REGISTER_CONST2ATTR_UNIQ_HELPER(ctr, name) \
  REGISTER_CONST2ATTR_UNIQ(ctr, name)
#define REGISTER_CONST2ATTR_UNIQ(ctr, opType)            \
  static OpConst2AttrOpReciever register_const2attr##ctr \
      __attribute__((unused)) = FusionConst2AttrOpRegister(opType)
}  // namespace fe
#endif
