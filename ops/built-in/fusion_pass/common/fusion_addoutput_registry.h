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
 * \file fusion_addoutput_registry.h
 * \brief all addoutput pass.
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_COMMON_FUSION_ADDOUTPUT_REGISTRY_H_
#define OPS_BUILT_IN_FUSION_PASS_COMMON_FUSION_ADDOUTPUT_REGISTRY_H_

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
  explicit FusionAddOutputOpRegister(std::string opType) : opType_(std::move(opType)){};
  FusionAddOutputOpRegister() = default;

  virtual ~FusionAddOutputOpRegister() = default;

  FusionAddOutputOpRegister& SetAddOutput(uint32_t inputInfoIndex, const std::string& inputInfoName);
  FusionAddOutputOpRegister& SetPreCheckFunc(std::function<Status(ge::NodePtr)> preCheckFunc);
  void GetInputInfo(std::vector<PassInputInfo>& inputInfoVec, std::function<Status(ge::NodePtr)>& preCheckFunc);

 private:
  std::string opType_;
  std::vector<PassInputInfo> inputInfoVec_;
  std::function<Status(ge::NodePtr)> preCheckFunc_{nullptr};
};

class FusionAddOutputOpRegistry {
 public:
  FusionAddOutputOpRegistry() = default;
  virtual ~FusionAddOutputOpRegistry() = default;

  static FusionAddOutputOpRegistry* Instance();

  Status SetAddOutputRegister(const string& opType, const FusionAddOutputOpRegister& reg);
  Status GetRegisterByOpType(const string& opType, FusionAddOutputOpRegister& reg) const;

 private:
  std::unordered_map<string, FusionAddOutputOpRegister> addOutputOpMap_;
};

class OpAddOutputOpReciever {
 public:
  OpAddOutputOpReciever(const FusionAddOutputOpRegister& reg);
  virtual ~OpAddOutputOpReciever() = default;
};

#define REGISTER_ADDOUTPUT(opType) REGISTER_ADDOUTPUT_UNIQ_HELPER(__COUNTER__, opType)
#define REGISTER_ADDOUTPUT_UNIQ_HELPER(ctr, name) REGISTER_ADDOUTPUT_UNIQ(ctr, name)
#define REGISTER_ADDOUTPUT_UNIQ(ctr, opType) \
  static OpAddOutputOpReciever register_addoutput##ctr __attribute__((unused)) = FusionAddOutputOpRegister(opType)
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_COMMON_FUSION_ADDOUTPUT_REGISTRY_H_
