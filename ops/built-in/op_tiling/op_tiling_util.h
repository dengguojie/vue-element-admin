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

namespace optiling {
using optiling::ByteBuffer;
using namespace ge;

const std::map<std::string, DataType> STR_TO_DATATYPE = {{"float", DT_FLOAT},
                                                         {"float32", DT_FLOAT},
                                                         {"float16", DT_FLOAT16},
                                                         {"int8", DT_INT8},
                                                         {"int16", DT_INT16},
                                                         {"int32", DT_INT32},
                                                         {"int64", DT_INT64},
                                                         {"uint8", DT_UINT8},
                                                         {"uint16", DT_UINT16},
                                                         {"uint32", DT_UINT32},
                                                         {"uint64", DT_UINT64},
                                                         {"bool", DT_BOOL},
                                                         {"double", DT_DOUBLE},
                                                         {"dual", DT_DUAL},
                                                         {"dual_sub_int8", DT_DUAL_SUB_INT8},
                                                         {"dual_sub_uint8", DT_DUAL_SUB_UINT8},
                                                         {"int4", DT_INT4},
                                                         {"bfloat16", DT_BF16}};

/*
 * @brief: read input shapes from paras
 * @param [in] paras: ge::Operator
 * @return vector<vector<int64_t>>: shapes vector of inputs
 */
vector<vector<int64_t>> GetInputShapes(const ge::Operator& paras);

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
  string result = "(";
  const T* data_addr = reinterpret_cast<const T*>(data.c_str());
  for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
    result += std::to_string(*data_addr);
    data_addr++;
    result += ",";
  }
  result += ")";

  return result;
}

/*
 * @brief: get Byte size base on dtype(string)
 * @param [in] op_type: string dtype
 * @return int64_t: byte len
 */
int64_t GetByteLenByString(const std::string& op_type);
}  // namespace optiling
#endif  // CANN_OPS_BUILT_IN_OP_TILING_OP_TILING_UTIL_H_
