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
 * \file fusion_const2attr_registry.cpp
 * \brief all const2attr registry.
 */
#include "fusion_const2attr_registry.h"
#include "graph/compute_graph.h"
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "pattern_fusion_util.h"
#include "op_log.h"

namespace fe {
FusionConst2AttrOpRegistry* FusionConst2AttrOpRegistry::Instance() {
  static FusionConst2AttrOpRegistry instance;
  return &instance;
}

OpConst2AttrOpReciever::OpConst2AttrOpReciever(FusionConst2AttrOpRegister& reg_data) {
  FusionConst2AttrOpRegistry::Instance()->SetConst2AttrRegister(reg_data.oriOpType_, reg_data);
}

Status FusionConst2AttrOpRegistry::SetConst2AttrRegister(const string& opType, FusionConst2AttrOpRegister& reg) {
  if (const2AttrOpMap_.find(opType) == const2AttrOpMap_.end()) {
    const2AttrOpMap_.insert(make_pair(opType, reg));
    OP_LOGD(opType.c_str(), "%s register successfully!", opType.c_str());
  } else {
    OP_LOGW(opType.c_str(), "%s has registered already!", opType.c_str());
  }
  return SUCCESS;
}
Status FusionConst2AttrOpRegistry::GetRegisterByOriType(const string& oriOpType, FusionConst2AttrOpRegister& reg) {
  if (const2AttrOpMap_.find(oriOpType) != const2AttrOpMap_.end()) {
    reg = const2AttrOpMap_[oriOpType];
    OP_LOGD(oriOpType.c_str(), "%s find in registery", oriOpType.c_str());
  } else {
    return FAILED;
  }
  return SUCCESS;
}
void FusionConst2AttrOpRegister::GetAttrInfo(std::string& opType, bool& needCheckSupported,
                                             std::vector<PassAttrInfo>& attrVec,
                                             std::function<Status(ge::NodePtr)>& preCheckFunc) {
  opType = opType_;
  needCheckSupported = needCheckSupported_;
  attrVec = attrVec_;
  preCheckFunc = preCheckFunc_;
}

FusionConst2AttrOpRegister& FusionConst2AttrOpRegister::OriginOpType(const std::string& oriOpType) {
  oriOpType_ = oriOpType;
  return *this;
}

FusionConst2AttrOpRegister& FusionConst2AttrOpRegister::NeedCheckSupported(const bool needCheck) {
  needCheckSupported_ = needCheck;
  return *this;
}

FusionConst2AttrOpRegister& FusionConst2AttrOpRegister::SetConstToAttr(int attrIndex, const std::string& attrName,
                                                                       const std::string& attrType) {
  PassAttrInfo attrInfo = {attrIndex, attrName, attrType};
  attrVec_.push_back(attrInfo);
  return *this;
}

FusionConst2AttrOpRegister& FusionConst2AttrOpRegister::SetPreCheckFunc(std::function<Status(ge::NodePtr)> preCheckFunc) {
  preCheckFunc_ = preCheckFunc;
  return *this;
}
}  // namespace fe
