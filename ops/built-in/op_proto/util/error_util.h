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
#include "graph/operator.h"


#define AICPU_INFER_SHAPE_CALL_ERR_REPORT(op_name, err_msg) \
  do { \
    OP_LOGE(op_name, "%s", err_msg.c_str()); \
    REPORT_CALL_ERROR(GetViewErrorCodeStr(ViewErrorCode::AICPU_INFER_SHAPE_ERROR), \
      "%s", ConcatString("op[", op_name, "], ", err_msg).c_str()); \
  } while (0)

#define AICPU_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg) \
  do { \
    OP_LOGE(op_name, "%s", err_msg.c_str()); \
    REPORT_INNER_ERROR(GetViewErrorCodeStr(ViewErrorCode::AICPU_INFER_SHAPE_ERROR), \
      "%s", ConcatString("op[", op_name, "], ", err_msg).c_str()); \
  } while (0)

#define VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg)\
  do { \
    OP_LOGE(op_name, "%s", err_msg.c_str()); \
    REPORT_INNER_ERROR(GetViewErrorCodeStr(ViewErrorCode::VECTOR_INNER_ERROR), \
      "%s", ConcatString("op[", op_name, "], ", err_msg).c_str()); \
  } while (0)

#define VECTOR_FUSION_INNER_ERR_REPORT(op_name, err_msg, ...)\
  do { \
    OP_LOGE(op_name, err_msg, ##__VA_ARGS__); \
    REPORT_INNER_ERROR(GetViewErrorCodeStr(ViewErrorCode::VECTOR_INNER_ERROR), \
    "op[%s], " err_msg, get_op_name(op_name), ##__VA_ARGS__); \
  } while (0)

#define CUBE_INNER_ERR_REPORT(op_name, err_msg, ...) \
  do { \
      OP_LOGE(op_name, err_msg, ##__VA_ARGS__); \
      REPORT_INNER_ERROR("E69999", "op[%s], " err_msg, op_name, ##__VA_ARGS__); \
  } while(0)

#define CUBE_CALL_ERR_REPORT(op_name, err_msg, ...) \
  do { \
      OP_LOGE(op_name, err_msg, ##__VA_ARGS__); \
      REPORT_CALL_ERROR("E69999", "op[%s], " err_msg, op_name, ##__VA_ARGS__); \
  } while(0)

#define CUBE_INNER_ERR_REPORT_PLUGIN(op_name, err_msg, ...) \
  do { \
      OP_LOGE(op_name, err_msg, ##__VA_ARGS__); \
      REPORT_INNER_ERROR("E59999", "op[%s], " err_msg, op_name, ##__VA_ARGS__); \
  } while(0)

#define CUBE_CALL_ERR_REPORT_PLUGIN(op_name, err_msg, ...) \
  do { \
      OP_LOGE(op_name, err_msg, ##__VA_ARGS__); \
      REPORT_CALL_ERROR("E59999", "op[%s], " err_msg, op_name, ##__VA_ARGS__); \
  } while(0)

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

std::string GetInputInvalidErrMsg(const std::string& param_name);
std::string GetShapeSizeErrMsg(uint32_t index, const std::string& wrong_shape_size,
                               const std::string& correct_shape_size);

std::string GetInputFormatNotSupportErrMsg(const std::string& param_name, const std::string& expected_format_list,
                                           const std::string& data_format);

std::string GetInputDtypeNotSupportErrMsg(const std::string& param_name, const std::string& expected_dtype_list,
                                          const std::string& data_dtype);

std::string GetInputDTypeErrMsg(const std::string& param_name, const std::string& expected_dtype,
                                const std::string& data_dtype);

std::string GetInputFormatErrMsg(const std::string& param_name, const std::string& expected_format,
                                 const std::string& data_format);

std::string SetAttrErrMsg(const std::string& param_name);
std::string UpdateParamErrMsg(const std::string& param_name);

std::string GetParamOutRangeErrMsg(const std::string& param_name, const std::string& range,
                                   const std::string& real_value);

std::string OtherErrMsg(const std::string& error_detail);

void TbeInputDataTypeErrReport(const std::string& op_name, const std::string& param_name,
                               const std::string& expected_dtype_list, const std::string& dtype);

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
