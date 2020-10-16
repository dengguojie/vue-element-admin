/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief all addoutput registry.
 *
 * @version 1.0
 *
 */

#include "fusion_addoutput_registry.h"
#include "graph/compute_graph.h"
#include "register/graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "pattern_fusion_util.h"
#include "op_log.h"

namespace fe {
FusionAddOutputOpRegistry *FusionAddOutputOpRegistry::Instance() {
  static FusionAddOutputOpRegistry instance;
  return &instance;
}

OpAddOutputOpReciever::OpAddOutputOpReciever(
    FusionAddOutputOpRegister &reg_data) {
  FusionAddOutputOpRegistry::Instance()->SetAddOutputRegister(reg_data.opType_,
                                                              reg_data);
}

Status FusionAddOutputOpRegistry::SetAddOutputRegister(
    const string &opType, FusionAddOutputOpRegister &reg) {
  if (addOutputOpMap_.find(opType) == addOutputOpMap_.end()) {
    addOutputOpMap_.insert(make_pair(opType, reg));
    OP_LOGD(opType.c_str(), "%s register successfully!", opType.c_str());
  } else {
    OP_LOGW(opType.c_str(), "%s has registered already!", opType.c_str());
  }
  return SUCCESS;
}

Status FusionAddOutputOpRegistry::GetRegisterByOpType(
    const string &opType, FusionAddOutputOpRegister &reg) {
  if (addOutputOpMap_.find(opType) != addOutputOpMap_.end()) {
    reg = addOutputOpMap_[opType];
    OP_LOGD(opType.c_str(), "%s find in registry", opType.c_str());
  } else {
    return FAILED;
  }
  return SUCCESS;
}

void FusionAddOutputOpRegister::GetInputInfo(
    std::vector<PassInputInfo> &inputInfoVec,
    std::function<bool(ge::NodePtr)> &preCheckFunc) {
  inputInfoVec = inputInfoVec_;
  preCheckFunc = preCheckFunc_;
}

FusionAddOutputOpRegister &FusionAddOutputOpRegister::SetAddOutput(
    uint32_t inputInfoIndex, const std::string &inputInfoName) {
  PassInputInfo inputInfo = {inputInfoIndex, inputInfoName};
  inputInfoVec_.push_back(inputInfo);
  return *this;
}

FusionAddOutputOpRegister &FusionAddOutputOpRegister::SetPreCheckFunc(
        std::function<bool(ge::NodePtr)> preCheckFunc) {
  preCheckFunc_ = preCheckFunc;
  return *this;
}

}  // namespace fe
