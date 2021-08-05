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
 * \file op_tiling_util.h
 * \brief
 */

#ifndef CANN_OPS_BUILT_IN_OP_TILING_OP_TILING_UTIL_H_
#define CANN_OPS_BUILT_IN_OP_TILING_OP_TILING_UTIL_H_

#include <vector>
#include "op_tiling.h"
#include "external/graph/operator.h"

using optiling::ByteBuffer;
using namespace ge;

/*
 * @brief: read constvalue from paras store into values
 * @param [in] paras: ge::Operator
 * @param [in] input_name: constvalue name
 * @param [out] values: vector to store return values.
  * @return bool: flag of success or not
 */
 
bool GetConstValue(const ge::Operator& paras, const string& input_name, std::vector<int64_t>& values);
/*
 * @brief: get datatype string from enum
 * @param [in] type: enum datatype
  * @return string: datatype string
 */
 
std::string to_string(const ge::DataType& type);
/*
 * @brief: get format string from enum
 * @param [in] format: enum format
  * @return string: format string
 */
std::string to_string(const ge::Format& format);

template <typename T>
string to_string(const ByteBuffer& tiling_data) {
  auto data = tiling_data.str();
  string result="(";
  const T *data_addr = reinterpret_cast<const T*>(data.c_str());
  for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
    result += std::to_string(*data_addr);
    data_addr++;
    result += ",";
  }
  result += ")";

  return result;
}

#endif  // CANN_OPS_BUILT_IN_OP_TILING_OP_TILING_UTIL_H_
