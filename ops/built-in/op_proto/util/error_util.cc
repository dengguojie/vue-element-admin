/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file error_util.cpp
 * \brief
 */
#include <map>
#include "common/util/error_manager/error_manager.h"
#include "error_util.h"
#include "error_code.h"
#include "op_log.h"

using namespace std;
using namespace ge;

namespace ge {

std::string GetViewErrorCodeStr(ge::ViewErrorCode errCode) {
  return "E" + std::to_string(errCode);
}

std::string GetShapeErrMsg(uint32_t index,
                           const std::string& wrong_shape,
                           const std::string& correct_shape)
{
  std::string msg = ConcatString(index, "th input has wrong shape",
      wrong_shape, ", it should be ", correct_shape);
  return msg;
}

std::string GetAttrValueErrMsg(const std::string& attr_name, const std::string& wrong_val,
                               const std::string& correct_val)
{
  std::string msg = ConcatString("attr[", attr_name, "], has wrong value[",
      wrong_val, "], it should be ", correct_val);
  return msg;
}

std::string GetAttrSizeErrMsg(const std::string& attr_name, const std::string& wrong_size,
                              const std::string& correct_size)
{
  std::string msg = ConcatString("attr[", attr_name, "], has wrong size[",
      wrong_size, "], it should be ", correct_size);
  return msg;
}

std::string GetInputInvalidErrMsg(const std::string& param_name) {
  std::string msg = ConcatString("get ", param_name, " failed");
  return msg;
}

std::string GetShapeSizeErrMsg(uint32_t index, const std::string& wrong_shape_size,
                               const std::string& correct_shape_size) {
  std::string msg =
      ConcatString(index, "th input has wrong shape size ", wrong_shape_size, ", it should be ", correct_shape_size);
  return msg;
}

std::string GetInputFormatNotSupportErrMsg(const std::string& param_name, const std::string& expected_format_list,
                                           const std::string& data_format) {
  std::string msg = ConcatString("[", param_name, "], has wrong format [", data_format, "], it should be in ",
                                 expected_format_list);
  return msg;
}

std::string GetInputDtypeNotSupportErrMsg(const std::string& param_name, const std::string& expected_dtype_list,
                                          const std::string& data_dtype) {
  std::string msg =
      ConcatString("[", param_name, "], has wrong dtype [", data_dtype, "], it should be in ", expected_dtype_list);
  return msg;
}

std::string GetInputDTypeErrMsg(const std::string& param_name, const std::string& expected_dtype,
                                const std::string& data_dtype) {
  std::string msg =
      ConcatString("[", param_name, "], has wrong dtype [", data_dtype, "], it should be ", expected_dtype);
  return msg;
}

std::string GetInputFormatErrMsg(const std::string& param_name, const std::string& expected_format,
                                 const std::string& data_format) {
  std::string msg =
      ConcatString("[", param_name, "], has wrong format [", data_format, "], it should be in ", expected_format);
  return msg;
}

std::string SetAttrErrMsg(const std::string& param_name) {
  std::string msg = ConcatString("set param [", param_name, "] failed");
  return msg;
}

std::string UpdateParamErrMsg(const std::string& param_name) {
  std::string msg = ConcatString("update [", param_name, "] failed");
  return msg;
}

std::string GetParamOutRangeErrMsg(const std::string& param_name, const std::string& range,
                                   const std::string& real_value) {
  std::string msg = ConcatString("the parameter [", param_name, "] should be in the range of [", range,
                                 "], but actually is ", real_value);
  return msg;
}

std::string OtherErrMsg(const std::string& error_detail) {
  std::string msg = error_detail;
  return msg;
}

void OpsMissInputErrReport(const std::string& op_name, const std::string& param_name) {
  map<string, string> err_map;
  err_map["op_name"] = op_name;
  err_map["param_name"] = param_name;
  std::string report_error_code = GetViewErrorCodeStr(ViewErrorCode::INVALID_MISS_INPUT);
  ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
}

void OpsInputFormatErrReport(const std::string& op_name, const std::string& param_name,
                             const std::string& expected_format_list, const std::string& data_format) {
  map<string, string> err_map;
  err_map["op_name"] = op_name;
  err_map["param_name"] = param_name;
  err_map["expected_format_list"] = expected_format_list;
  err_map["format"] = data_format;
  std::string report_error_code = GetViewErrorCodeStr(ViewErrorCode::INVALID_INPUT_FORMAT);
  ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
}

void OpsInputDtypeErrReport(const std::string& op_name, const std::string& param_name,
                            const std::string& expected_data_type_list, const std::string& data_type) {
  map<string, string> err_map;
  err_map["op_name"] = op_name;
  err_map["param_name"] = param_name;
  err_map["expected_data_type_list"] = expected_data_type_list;
  err_map["data_type"] = data_type;
  std::string report_error_code = GetViewErrorCodeStr(ViewErrorCode::INVALID_INPUT_DTYPE);
  ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
}

void OpsInputTypeErrReport(const std::string& op_name, const std::string& param_name, const std::string& param_type,
                           const std::string& actual_type) {
  map<string, string> err_map;
  err_map["op_name"] = op_name;
  err_map["param_name"] = param_name;
  err_map["param_type"] = param_type;
  err_map["actual_type"] = actual_type;
  std::string report_error_code = GetViewErrorCodeStr(ViewErrorCode::INVALID_INPUT_TYPE);
  ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
}

void OpsGetAttrErrReport(const std::string& op_name, const std::string& param_name) {
  map<string, string> err_map;
  err_map["op_name"] = op_name;
  err_map["param_name"] = param_name;
  std::string report_error_code = GetViewErrorCodeStr(ViewErrorCode::INVALID_GET_ATTR);
  ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
}

void OpsSetAttrErrReport(const std::string& op_name, const std::string& param_name) {
  map<string, string> err_map;
  err_map["op_name"] = op_name;
  err_map["param_name"] = param_name;
  std::string report_error_code = GetViewErrorCodeStr(ViewErrorCode::INVALID_SET_ATTR);
  ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
}

void OpsAttrValueErrReport(const std::string& op_name, const std::string& param_name, const std::string& excepted_value,
                           const std::string& input_value) {
  map<string, string> err_map;
  err_map["op_name"] = op_name;
  err_map["param_name"] = param_name;
  err_map["excepted_value"] = excepted_value;
  err_map["input_value"] = input_value;
  std::string report_error_code = GetViewErrorCodeStr(ViewErrorCode::INVALID_OPS_ATTR_VALUE);
  ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
}

void OpsOPUpdateErrReport(const std::string& op_name, const std::string& param_name) {
  map<string, string> err_map;
  err_map["op_name"] = op_name;
  err_map["param_name"] = param_name;
  std::string report_error_code = GetViewErrorCodeStr(ViewErrorCode::FAILED_UPDATE_OP);
  ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
}

void OpsInputShapeErrReport(const std::string& op_name, const std::string& rule_desc, const std::string& param_name,
                            const std::string& param_value) {
  map<string, string> err_map;
  err_map["op_name"] = op_name;
  err_map["rule_desc"] = rule_desc;
  err_map["param_name"] = param_name;
  err_map["param_value"] = param_value;
  std::string report_error_code = GetViewErrorCodeStr(ViewErrorCode::INVALID_SHAPE);
  ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
}

void OpsOneInputShapeErrReport(const std::string& op_name, const std::string& param_name,
                               const std::string& error_detail) {
  map<string, string> err_map;
  err_map["op_name"] = op_name;
  err_map["param_name"] = param_name;
  err_map["error_detail"] = error_detail;
  std::string report_error_code = GetViewErrorCodeStr(ViewErrorCode::INVALID_ONE_INPUT_SHAPE);
  ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
}

void OpsTwoInputShapeErrReport(const std::string& op_name, const std::string& param_name1,
                               const std::string& param_name2, const std::string& error_detail) {
  map<string, string> err_map;
  err_map["op_name"] = op_name;
  err_map["param_name1"] = param_name1;
  err_map["param_name2"] = param_name2;
  err_map["error_detail"] = error_detail;
  std::string report_error_code = GetViewErrorCodeStr(ViewErrorCode::INVALID_TWO_INPUT_SHAPE);
  ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
}

void OpsOneOutputShapeErrReport(const std::string& op_name, const std::string& param_name,
                                const std::string& error_detail) {
  map<string, string> err_map;
  err_map["op_name"] = op_name;
  err_map["param_name"] = param_name;
  err_map["error_detail"] = error_detail;
  std::string report_error_code = GetViewErrorCodeStr(ViewErrorCode::INVALID_ONE_OUTPUT_SHAPE);
  ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
}

void OpsGetCompileParamsErrReport(const std::string& op_name, const std::string& param_name) {
  map<string, string> err_map;
  err_map["op_name"] = op_name;
  err_map["param_name"] = param_name;
  std::string report_error_code = GetViewErrorCodeStr(ViewErrorCode::FAILED_GET_COMPILIE_PARAMS);
  ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
}

void OpsInputShapeSizeErrReport(const std::string& op_name, const std::string& input_name, const std::string& max_value,
                                const std::string& real_value) {
  map<string, string> err_map;
  err_map["op_name"] = op_name;
  err_map["input_name"] = input_name;
  err_map["max_value"] = max_value;
  err_map["real_value"] = real_value;
  std::string report_error_code = GetViewErrorCodeStr(ViewErrorCode::INVALID_SHAPE_SIZE);
  ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
}

void OpsInputShapeDimErrReport(const std::string& op_name, const std::string& param_name, const std::string& max_value,
                               const std::string& min_value, const std::string& real_value) {
  map<string, string> err_map;
  err_map["op_name"] = op_name;
  err_map["param_name"] = param_name;
  err_map["max_value"] = max_value;
  err_map["min_value"] = min_value;
  err_map["real_value"] = real_value;
  std::string report_error_code = GetViewErrorCodeStr(ViewErrorCode::INVALID_SHAPE_DIM);
  ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
}

void OpsInputShapeBroadcastErrReport(const std::string& op_name, const std::string& input1_name,
                                     const std::string& input2_name, const std::string& input1_shape,
                                     const std::string& input2_shape) {
  map<string, string> err_map;
  err_map["op_name"] = op_name;
  err_map["input1_name"] = input1_name;
  err_map["input2_name"] = input2_name;
  err_map["input1_shape"] = input1_shape;
  err_map["input2_shape"] = input2_shape;
  std::string report_error_code = GetViewErrorCodeStr(ViewErrorCode::INVALID_BROADCAST_SHAPE);
  ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
}

void TbeInputDataTypeErrReport(const std::string& op_name, const std::string& param_name,
                               const std::string& expected_dtype_list, const std::string& dtype) {
  map<string, string> err_map;
  err_map["op_name"] = op_name;
  err_map["param_name"] = param_name;
  err_map["expected_dtype_list"] = expected_dtype_list;
  err_map["dtype"] = dtype;
  std::string report_error_code = "E50034";
  ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
}

void OpsTwoInputDtypeErrReport(const std::string& op_name, const std::string& input1_name,
                               const std::string& input2_name, const std::string& input1_dtype,
                               const std::string& input2_dtype) {
  map<string, string> err_map;
  err_map["op_name"] = op_name;
  err_map["input1_name"] = input1_name;
  err_map["input2_name"] = input2_name;
  err_map["input1_dtype"] = input1_dtype;
  err_map["input2_dtype"] = input2_dtype;
  std::string report_error_code = GetViewErrorCodeStr(ViewErrorCode::INVALID_TWO_INPUT_DTYPE);
  ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
}

void OpsAippErrReport(const std::string& aipp_output_H, const std::string& aipp_output_W, const std::string& data_H,
                      const std::string& data_W) {
  map<string, string> err_map;
  err_map["aipp_output_H"] = aipp_output_H;
  err_map["aipp_output_W"] = aipp_output_W;
  err_map["data_H"] = data_H;
  err_map["data_W"] = data_W;
  std::string report_error_code = GetViewErrorCodeStr(ViewErrorCode::INVALID_AIPP_ERROR);
  ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
}

void OpsConvAttrValueErrReport(const std::string& op_name, const std::string& param_name, const std::string& expected_value,
                           const std::string& input_value) {
  map<string, string> err_map;
  err_map["op_name"] = op_name;
  err_map["param_name"] = param_name;
  err_map["expected_value"] = expected_value;
  err_map["input_value"] = input_value;
  std::string report_error_code = GetViewErrorCodeStr(ViewErrorCode::INVALID_CONV_ATTR_VALUE);
  ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
}

void OpsConvSetAttrErrReport(const std::string& op_name, const std::string& param1_name,
                           const std::string& param2_name) {
  map<string, string> err_map;
  err_map["op_name"] = op_name;
  err_map["param1_name"] = param1_name;
  err_map["param2_name"] = param2_name;
  std::string report_error_code = GetViewErrorCodeStr(ViewErrorCode::INVALID_CONV_SET_ATTR);
  ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
}

void OpsConvShapeErrReport(const std::string& op_name, const std::string& description) {
  map<string, string> err_map;
  err_map["op_name"] = op_name;
  err_map["description"] = description;
  std::string report_error_code = GetViewErrorCodeStr(ViewErrorCode::INVALID_CONV_SHAPE);
  ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
}

void GeInfershapeErrReport(const std::string& op_name, const std::string& op_type, const std::string& value,
                           const std::string& reason) {
  std::string report_error_code = GetViewErrorCodeStr(ViewErrorCode::INVALID_INFER_SHAPE);
  ErrorManager::GetInstance().ATCReportErrMessage(report_error_code, {"opname", "optype", "value", "reason"},
                                                  {op_name, op_type, value, reason});
}

void CommonRuntimeErrLog(const std::string& opname, const std::string& description){
  map<string, string> err_map;
  err_map["op_name"] = opname;
  err_map["description"] = description;
  OP_LOGE(opname.c_str(), description);
  (void)ErrorManager::GetInstance().ReportErrMessage("E50058", err_map);
}

}  // namespace ge
