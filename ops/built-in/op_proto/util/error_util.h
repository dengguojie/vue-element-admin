/* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use
 * this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef OP_PROTO_ERROR_UTIL_H_
#define OP_PROTO_ERROR_UTIL_H_

#include <sstream>
#include <string>
#include <vector>
#include "operator.h"

namespace ge
{

/*
* get debug string of vector
* param[in] v vector
* return vector's debug string
*/
template<typename T>
std::string DebugString(const std::vector<T> &v)
{
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
template<typename T>
std::string Strcat(T arg)
{
    std::ostringstream oss;
    oss << arg;
    return oss.str();
}

template<typename T, typename... Ts>
std::string Strcat(T arg, Ts... arg_left)
{
    std::ostringstream oss;
    oss << arg;
    oss << Strcat(arg_left...);
    return oss.str();
}

/*
* report input shape error of infer shape
* param[in] index the index of input
* param[in] opname op name
* param[in] wrong_shape wrong input shape
* param[in] correct_shape correct input shape
* return void
*/
void ShapeErrReport(uint32_t index,
                    const std::string &opname,
                    const std::string &wrong_shape,
                    const std::string &correct_shape);

/*
* report attr value error of infer shape
* param[in] attrname the attr name
* param[in] opname op name
* param[in] wrong_value wrong attr value
* param[in] correct_value correct attr value
* return void
*/
void AttrValueErrReport(const std::string &attrName,
                        const std::string &opname,
                        const std::string &wrong_value,
                        const std::string &correct_value);

/*
* report attr size error of infer shape
* param[in] attrname the attr name
* param[in] opname op name
* param[in] wrong_size wrong attr size
* param[in] correct_size correct attr size
* return void
*/
void AttrSizeErrReport(const std::string &attrName,
                       const std::string &opname,
                       const std::string &wrong_size,
                       const std::string &correct_size);

/*
* report common error of infer shape
* param[in] opname op name
* param[in] err_msg error message
* return void
*/
void InferShapeOtherErrReport(const std::string &opname, const std::string &err_msg);

void OpsMissInputErrReport(const std::string &op_name,
                       const std::string &param_name);


void OpsInputFormatErrReport(const std::string &op_name,
                       const std::string &param_name,
                       const std::string &expected_format_list,
                       const std::string &data_format);


void OpsInputDtypeErrReport(const std::string &op_name,
                       const std::string &param_name,
                       const std::string &expected_data_type_list,
                       const std::string &data_type);


void OpsInputTypeErrReport(const std::string &op_name,
                       const std::string &param_name,
                       const std::string &param_type,
                       const std::string &actual_type);


void OpsGetAttrErrReport(const std::string &op_name,
                       const std::string &param_name);


void OpsSetAttrErrReport(const std::string &op_name,
                       const std::string &param_name);


void OpsAttrValueErrReport(const std::string &op_name,
                       const std::string &param_name,
                       const std::string &excepted_value,
                       const std::string &input_value);


void OpsOPUpdateErrReport(const std::string &op_name,
                       const std::string &param_name);


void OpsInputShapeErrReport(const std::string &op_name,
                       const std::string &rule_desc,
                       const std::string &param_name,
                       const std::string &param_value);


void OpsOneInputShapeErrReport(const std::string &op_name,
                       const std::string &param_name,
                       const std::string &error_detail);


void OpsTwoInputShapeErrReport(const std::string &op_name,
                       const std::string &param_name1,
                       const std::string &param_name2,
                       const std::string &error_detail);


void OpsOneOutputShapeErrReport(const std::string &op_name,
                       const std::string &param_name,
                       const std::string &error_detail);


void OpsGetCompileParamsErrReport(const std::string &op_name,
                       const std::string &param_name);


void OpsInputShapeSizeErrReport(const std::string &op_name,
                       const std::string &input_name,
                       const std::string &max_value,
                       const std::string &real_value);


void OpsInputShapeDimErrReport(const std::string &op_name,
                       const std::string &param_name,
                       const std::string &max_value,
                       const std::string &min_value,
                       const std::string &real_value);

void OpsInputShapeBroadcastErrReport(const std::string &op_name,
                       const std::string &input1_name,
                       const std::string &input2_name,
                       const std::string &input1_shape,
                       const std::string &input2_shape);


void TbeInputDataTypeErrReport(const std::string &op_name,
                        const std::string &param_name,
                        const std::string &expected_dtype_list,
                        const std::string &dtype);


void OpsTwoInputDtypeErrReport(const std::string &op_name,
                       const std::string &input1_name,
                       const std::string &input2_name,
                       const std::string &input1_dtype,
                       const std::string &input2_dtype);

void OpsAippErrReport(const std::string &aipp_output_H,
                       const std::string &aipp_output_W,
                       const std::string &data_H,
                       const std::string &data_W);

void GeInfershapeErrReport(const std::string &op_name,
                           const std::string &op_type,
                           const std::string &value,
                           const std::string &reason);
}


#endif  // OP_PROTO_ERROR_UTIL_H_
