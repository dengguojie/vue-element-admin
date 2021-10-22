/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#include "graph/utils/op_desc_utils.h"

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

int64_t GetByteLenByString(const std::string& data_type) {
  auto find_it = STR_TO_DATATYPE.find(data_type);
  if (find_it != STR_TO_DATATYPE.end()) {
    return GetSizeByDataType(find_it->second);
  }
  OP_LOGW("GetByteLen", "con not get the dtype[%s] in ge::DataType list. will return 0", data_type.c_str());
  return 0;
}

vector<vector<int64_t>> GetInputShapes(const ge::Operator& paras) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(paras);
  if (op_desc == nullptr)
    return {};

  vector<vector<int64_t>> shapes;
  int count = op_desc->GetInputsSize();
  for (int i = 0; i < count; i++) {
    auto ptr = op_desc->MutableInputDesc(i);
    shapes.emplace_back(ptr->MutableShape().GetDims());
  }

  return shapes;
}

int64_t GetDataBlockElems(const ge::DataType& dtype) {
  int64_t dataBlock = 0;
  if (dtype == DT_FLOAT || dtype == DT_INT32 || dtype == DT_UINT32) {
    dataBlock = 8;
  } else if (dtype == DT_FLOAT16 || dtype == DT_INT16 || dtype == DT_UINT16) {
    dataBlock = 16;
  } else if (dtype == DT_INT8 || dtype == DT_UINT8) {
    dataBlock = 32;
  } else if (dtype == DT_INT64 || dtype == DT_UINT64) {
    dataBlock = 4;
  }
  return dataBlock;
}

}  // namespace optiling
