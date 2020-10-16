/**
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief all addoutput pass.
 *
 * @version 1.0
 *
 */

#ifndef BUILT_IN_FUSION_PASS_FUSION_ADDOUTPUT_REGISTRY_H_
#define BUILT_IN_FUSION_PASS_FUSION_ADDOUTPUT_REGISTRY_H_

#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <functional>
#include "pattern_fusion_util.h"
#include "graph/compute_graph.h"

namespace fe {
class FusionAddOutputOpRegister {
  friend class PatternFusionUtil;
  friend class OpAddOutputOpReciever;
  friend class FusionAddOutputOpRegistry;

 public:
  explicit FusionAddOutputOpRegister(std::string opType)
      : opType_(std::move(opType)){};
  FusionAddOutputOpRegister() = default;

  virtual ~FusionAddOutputOpRegister() = default;

  FusionAddOutputOpRegister &SetAddOutput(uint32_t inputInfoIndex,
                                          const std::string &inputInfoName);
  FusionAddOutputOpRegister &SetPreCheckFunc(std::function<bool(ge::NodePtr)> preCheckFunc);
  void GetInputInfo(std::vector<PassInputInfo> &inputInfoVec,
                    std::function<bool(ge::NodePtr)> &preCheckFunc);

 private:
  std::string opType_;
  std::vector<PassInputInfo> inputInfoVec_;
  std::function<bool(ge::NodePtr)> preCheckFunc_{nullptr};
};

class FusionAddOutputOpRegistry {
 public:
  FusionAddOutputOpRegistry() = default;
  virtual ~FusionAddOutputOpRegistry() = default;

  static FusionAddOutputOpRegistry *Instance();

  Status SetAddOutputRegister(const string &opType,
                              FusionAddOutputOpRegister &addOutputRegister);
  Status GetRegisterByOpType(const string &oriOpType,
                             FusionAddOutputOpRegister &reg);

 private:
  std::unordered_map<string, FusionAddOutputOpRegister> addOutputOpMap_;
};

class OpAddOutputOpReciever {
 public:
  OpAddOutputOpReciever(FusionAddOutputOpRegister &reg);
  virtual ~OpAddOutputOpReciever() = default;
};

#define REGISTER_ADDOUTPUT(opType) \
  REGISTER_ADDOUTPUT_UNIQ_HELPER(__COUNTER__, opType)
#define REGISTER_ADDOUTPUT_UNIQ_HELPER(ctr, name) \
  REGISTER_ADDOUTPUT_UNIQ(ctr, name)
#define REGISTER_ADDOUTPUT_UNIQ(ctr, opType)           \
  static OpAddOutputOpReciever register_addoutput##ctr \
      __attribute__((unused)) = FusionAddOutputOpRegister(opType)
}  // namespace fe
#endif
