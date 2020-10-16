/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this
 * file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "graph/utils/op_desc_utils.h"
#include "register/register.h"
#include "op_log.h"
#include "common/util/error_manager/error_manager.h"

namespace domi {
Status DepthwiseConv2DMappingFn(const Message* op_src, ge::Operator& op) {
  AutoMappingFn(op_src, op);
  auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDesc tensorDescW = op_dsc->GetInputDesc(1);
  tensorDescW.SetOriginFormat(ge::FORMAT_HWCN);
  tensorDescW.SetFormat(ge::FORMAT_HWCN);
  auto ret = op_dsc->UpdateInputDesc(1, tensorDescW);
  if (ret != ge::GRAPH_SUCCESS) {
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["param_name"] = "updating filter's format";
    err_map["rule_desc"] = "update filter's format";
    err_map["format"] = "failed";
    std::string report_error_code = "E50012";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    OP_LOGE("[Plugin][ERROR]update filter format failed\n");
    return FAILED;
  }
  std::vector<int32_t> padList = {0,0,0,0};
  op.SetAttr("pads", padList);

  return SUCCESS;
}

REGISTER_CUSTOM_OP("DepthwiseConv2D")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DepthwiseConv2dNative")
    .ParseParamsFn(DepthwiseConv2DMappingFn)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
