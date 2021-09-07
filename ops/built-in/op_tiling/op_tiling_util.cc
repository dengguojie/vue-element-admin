/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file op_tiling_util.cc
 * \brief tiling function of op
 */

#include "op_tiling_util.h"
#include <functional>
#include <ge_error_codes.h>
#include <graph/utils/type_utils.h>

namespace optiling {
using namespace std;

/*
 * @brief: get datatype string from enum
 * @param [in] type: enum datatype
 * @return string: datatype string
 */
std::string to_string(const ge::DataType& type) {
  return ge::TypeUtils::DataTypeToSerialString(type);
}

/*
 * @brief: get format string from enum
 * @param [in] format: enum format
 * @return string: format string
 */
std::string to_string(const ge::Format& format) {
  return ge::TypeUtils::FormatToSerialString(format);
}

/*
 * @brief: read constvalue from paras store into values
 * @param [in] paras: ge::Operator
 * @param [in] input_name: constvalue name
 * @param [out] values: vector to store return values.
 * @return bool: flag of success or not
 */
bool GetConstValue(const ge::Operator& paras, const string& input_name, std::vector<int64_t>& values) {
  const string& op_type = paras.GetOpType();
  ge::Tensor const_tensor;
  if (paras.GetInputConstData(input_name, const_tensor) != ge::GRAPH_SUCCESS) {
    OP_LOGE(op_type.c_str(), "constvalue [%s] not exists.", input_name.c_str());
    return false;
  }

  auto dtype = const_tensor.GetTensorDesc().GetDataType();
  auto data = const_tensor.GetData();
  auto size = const_tensor.GetSize();
  if (data == nullptr || size == 0) {
    OP_LOGE(op_type.c_str(), "constvalue [%s] nullptr or size=0.", input_name.c_str());
    return false;
  }
  values.clear();
  switch (dtype) {
    case DT_INT64: {
      size_t count = size / sizeof(int64_t);
      const int64_t* data_addr = reinterpret_cast<const int64_t*>(data);
      for (size_t i = 0; i < count; i++) {
        values.push_back(*data_addr);
        data_addr++;
      }
    } break;
    case DT_INT32: {
      size_t count = size / sizeof(int32_t);
      const int32_t* data_addr = reinterpret_cast<const int32_t*>(data);
      for (size_t i = 0; i < count; i++) {
        values.push_back(*data_addr);
        data_addr++;
      }
    } break;
    default: {
      OP_LOGE(op_type, "GetConstValue of dtype[%s] has not implement.", to_string(dtype).c_str());
      return false;
    } break;
  }
  return true;
}

int64_t GetByteLenByString(const std::string& data_type) {
  auto find_it = STR_TO_DATATYPE.find(data_type);
  if (find_it != STR_TO_DATATYPE.end()) {
    return GetSizeByDataType(find_it->second);
  }
  OP_LOGW("GetByteLen", "con not get the dtype[%s] in ge::DataType list. will return 0", data_type.c_str());
  return 0;
}

}  // namespace optiling
