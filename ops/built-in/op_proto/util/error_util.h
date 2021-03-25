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
 * \file error_util.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_UTIL_ERROR_UTIL_H_
#define OPS_BUILT_IN_OP_PROTO_UTIL_ERROR_UTIL_H_

#include <sstream>
#include <string>
#include <vector>
#include "common/util/error_manager/error_manager.h"
#include "error_code.h"
#include "securec.h"
#include "operator.h"

#define AICPU_INFER_SHAPE_CALL_ERR_REPORT(op_name, err_msg) \
  do { \
    OP_LOGE(op_name.c_str(), "%s", err_msg.c_str()); \
    REPORT_CALL_ERROR(GetViewErrorCodeStr(ViewErrorCode::AICPU_INFER_SHAPE_ERROR), \
      "%s", ConcatString("op[", op_name, "], ", err_msg).c_str()); \
  } while(0);

#define AICPU_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg) \
  do { \
    OP_LOGE(op_name.c_str(), "%s", err_msg.c_str()); \
    REPORT_INNER_ERROR(GetViewErrorCodeStr(ViewErrorCode::AICPU_INFER_SHAPE_ERROR), \
      "%s", ConcatString("op[", op_name, "], ", err_msg).c_str()); \
  } while(0);

#define VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg)\
  do { \
    REPORT_INNER_ERROR("E" + std::to_string(ViewErrorCode.VECTOR_INNER_ERROR), err_msg); \
    OP_LOGE(op_name.c_str(), "%s", err_msg.c_str()); \
  } while (0);

namespace ge {

/*
 * get debug string of vector
 * param[in] v vector
 * return vector's debug string
 */
template <typename T>
std::string DebugString(const std::vector<T>& v) {
  std::ostringstream oss;
  oss << "[";
  if (v.size() > 0) {
    for (size_t i = 0; i < v.size() - 1; ++i) {
      oss << v[i] << ", ";
    }
    oss << v[v.size() - 1];
  }
  oss << "]";
  return oss.str();
}

/*
 * str cat util function
 * param[in] params need concat to string
 * return concatted string
 */
template <typename T>
std::string ConcatString(T arg) {
  std::ostringstream oss;
  oss << arg;
  return oss.str();
}

template <typename T, typename... Ts>
std::string ConcatString(T arg, Ts... arg_left) {
  std::ostringstream oss;
  oss << arg;
  oss << ConcatString(arg_left...);
  return oss.str();
}

std::string GetViewErrorCodeStr(ge::ViewErrorCode errCode);

std::string GetShapeErrMsg(uint32_t index, const std::string& wrong_shape,
                           const std::string& correct_shape);

std::string GetAttrValueErrMsg(const std::string& attr_name, const std::string& wrong_val,
                               const std::string& correct_val);

std::string GetAttrSizeErrMsg(const std::string& attr_name, const std::string& wrong_size,
                              const std::string& correct_size);

/*
 * report input shape error of infer shape
 * param[in] index the index of input
 * param[in] opname op name
 * param[in] wrong_shape wrong input shape
 * param[in] correct_shape correct input shape
 * return void
 */
void ShapeErrReport(uint32_t index, const std::string& opname, const std::string& wrong_shape,
                    const std::string& correct_shape);

/*
 * report attr value error of infer shape
 * param[in] attrname the attr name
 * param[in] opname op name
 * param[in] wrong_value wrong attr value
 * param[in] correct_value correct attr value
 * return void
 */
void AttrValueErrReport(const std::string& attrName, const std::string& opname, const std::string& wrong_value,
                        const std::string& correct_value);

/*
 * report attr size error of infer shape
 * param[in] attrname the attr name
 * param[in] opname op name
 * param[in] wrong_size wrong attr size
 * param[in] correct_size correct attr size
 * return void
 */
void AttrSizeErrReport(const std::string& attrName, const std::string& opname, const std::string& wrong_size,
                       const std::string& correct_size);

/*
 * report common error of infer shape
 * param[in] opname op name
 * param[in] err_msg error message
 * return void
 */
void InferShapeOtherErrReport(const std::string& opname, const std::string& err_msg);

void OpsMissInputErrReport(const std::string& op_name, const std::string& param_name);

void OpsInputFormatErrReport(const std::string& op_name, const std::string& param_name,
                             const std::string& expected_format_list, const std::string& data_format);

void OpsInputDtypeErrReport(const std::string& op_name, const std::string& param_name,
                            const std::string& expected_data_type_list, const std::string& data_type);

void OpsInputTypeErrReport(const std::string& op_name, const std::string& param_name, const std::string& param_type,
                           const std::string& actual_type);

void OpsGetAttrErrReport(const std::string& op_name, const std::string& param_name);

void OpsSetAttrErrReport(const std::string& op_name, const std::string& param_name);

void OpsAttrValueErrReport(const std::string& op_name, const std::string& param_name, const std::string& excepted_value,
                           const std::string& input_value);

void OpsOPUpdateErrReport(const std::string& op_name, const std::string& param_name);

void OpsInputShapeErrReport(const std::string& op_name, const std::string& rule_desc, const std::string& param_name,
                            const std::string& param_value);

void OpsOneInputShapeErrReport(const std::string& op_name, const std::string& param_name,
                               const std::string& error_detail);

void OpsTwoInputShapeErrReport(const std::string& op_name, const std::string& param_name1,
                               const std::string& param_name2, const std::string& error_detail);

void OpsOneOutputShapeErrReport(const std::string& op_name, const std::string& param_name,
                                const std::string& error_detail);

void OpsGetCompileParamsErrReport(const std::string& op_name, const std::string& param_name);

void OpsInputShapeSizeErrReport(const std::string& op_name, const std::string& input_name, const std::string& max_value,
                                const std::string& real_value);

void OpsInputShapeDimErrReport(const std::string& op_name, const std::string& param_name, const std::string& max_value,
                               const std::string& min_value, const std::string& real_value);

void OpsInputShapeBroadcastErrReport(const std::string& op_name, const std::string& input1_name,
                                     const std::string& input2_name, const std::string& input1_shape,
                                     const std::string& input2_shape);

void TbeInputDataTypeErrReport(const std::string& op_name, const std::string& param_name,
                               const std::string& expected_dtype_list, const std::string& dtype);

void OpsTwoInputDtypeErrReport(const std::string& op_name, const std::string& input1_name,
                               const std::string& input2_name, const std::string& input1_dtype,
                               const std::string& input2_dtype);

void OpsAippErrReport(const std::string& aipp_output_H, const std::string& aipp_output_W, const std::string& data_H,
                      const std::string& data_W);

void OpsConvAttrValueErrReport(const std::string& op_name, const std::string& param_name, const std::string& expected_value,
                           const std::string& input_value);

void OpsConvSetAttrErrReport(const std::string& op_name, const std::string& param1_name,
                           const std::string& param2_name);

void OpsConvShapeErrReport(const std::string& op_name, const std::string& description);

void GeInfershapeErrReport(const std::string& op_name, const std::string& op_type, const std::string& value,
                           const std::string& reason);
/*
 * log common runtime error
 * param[in] opname op name
 * param[in] error description
 * return void
 */
void CommonRuntimeErrLog(const std::string& opname, const std::string& description);
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_UTIL_ERROR_UTIL_H_
