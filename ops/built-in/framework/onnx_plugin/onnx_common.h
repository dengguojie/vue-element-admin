/* Copyright (C) 2020. Huawei Technologies Co., Ltd. All
rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <string>
#include <vector>
#include <map>
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/operator.h"
#include "all_ops.h"
#include "graph.h"

namespace domi {
std::string ONNX_PARSER_MODULE = "onnx_plugin";
std::string ONNX_PLUGIN_ERR_CODE = "E79999";
std::string ONNX_PLUGIN_WARNING_CODE = "W79999";
std::string ONNX_PLUGIN_INFO_CODE = "I79999";

void OnnxPluginLogE(const std::string& op_name, const std::string& err_detail) {
  map<string, string> errMap;
  errMap["report_module"] = ONNX_PARSER_MODULE;
  errMap["op_name"] = op_name;
  errMap["description"] = err_detail;
  std::string reportErrorCode = ONNX_PLUGIN_ERR_CODE;
  ErrorManager::GetInstance().ReportErrMessage(reportErrorCode, errMap);
}

void OnnxPluginLogW(const std::string& op_name, const std::string& warn_detail) {
  map<string, string> warnMap;
  warnMap["report_module"] = ONNX_PARSER_MODULE;
  warnMap["op_name"] = op_name;
  warnMap["description"] = warn_detail;
  std::string reportErrorCode = ONNX_PLUGIN_WARNING_CODE;
  ErrorManager::GetInstance().ReportErrMessage(reportErrorCode, warnMap);
}

void OnnxPluginLogI(const std::string& op_name, const std::string& info_detail) {
  map<string, string> infoMap;
  infoMap["report_module"] = ONNX_PARSER_MODULE;
  infoMap["op_name"] = op_name;
  infoMap["description"] = info_detail;
  std::string reportErrorCode = ONNX_PLUGIN_INFO_CODE;
  ErrorManager::GetInstance().ReportErrMessage(reportErrorCode, infoMap);
}
}  // namespace domi