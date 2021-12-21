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
 * \file fusion_removenode_registry.cpp
 * \brief all const2attr registry.
 */
#include "fusion_removenode_registry.h"
#include "op_log.h"

namespace fe {
FusionRemoveNodeRegistry* FusionRemoveNodeRegistry::Instance() {
  static FusionRemoveNodeRegistry instance;
  return &instance;
}

Status FusionRemoveNodeRegistry::SetRemoveNodeRegister(const std::string& opType, FusionRemoveNodeRegister& reg) {
  if (removeNodeMap_.find(opType) == removeNodeMap_.end()) {
    removeNodeMap_.insert(make_pair(opType, reg));
    OP_LOGD(opType.c_str(), "Remove node pass of op[%s] register successfully!", opType.c_str());
  } else {
    OP_LOGI(opType.c_str(), "Remove node pass of op[%s] has been registered.", opType.c_str());
  }
  return SUCCESS;
}

Status FusionRemoveNodeRegistry::GetRegisterByOpType(const std::string& opType, FusionRemoveNodeRegister& reg) {
  auto iter = removeNodeMap_.find(opType);
  if (iter != removeNodeMap_.end()) {
    reg = iter->second;
  } else {
    return FAILED;
  }
  return SUCCESS;
}

void FusionRemoveNodeRegister::GetParameters(std::vector<LinkIndexPair>& linkPairVec,
                                             std::function<Status(ge::NodePtr)>& preCheckFunc) {
  linkPairVec = linkPairVec_;
  preCheckFunc = preCheckFunc_;
}

FusionRemoveNodeRegister& FusionRemoveNodeRegister::AddLinkPair(uint32_t inAnchorIndex, uint32_t outAnchorIndex) {
  LinkIndexPair pair = {inAnchorIndex, outAnchorIndex};
  linkPairVec_.push_back(pair);
  return *this;
}

FusionRemoveNodeRegister& FusionRemoveNodeRegister::SetPreCheckFunc(std::function<Status(ge::NodePtr)> preCheckFunc) {
  preCheckFunc_ = preCheckFunc;
  return *this;
}

RemoveNodeReciever::RemoveNodeReciever(FusionRemoveNodeRegister& reg_data) {
  FusionRemoveNodeRegistry::Instance()->SetRemoveNodeRegister(reg_data.GetOpType(), reg_data);
}
}  // namespace fe
