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
 * \file op_const.h
 * \brief
 */

#ifndef CANN_OPS_BUILT_IN_OPS_CONST_H_
#define CANN_OPS_BUILT_IN_OPS_CONST_H_

#include <vector>
#include "external/graph/operator.h"
#include "graph/utils/op_desc_utils.h"

namespace ops {
using namespace ge;

template <typename T1, typename T2>
static void GetDataToVector(const uint8_t* const_data, size_t data_size, std::vector<T1>& result) {
  size_t size = data_size / sizeof(T2);
  result.resize(size);
  const T2* data = reinterpret_cast<const T2*>(const_data);
  for (size_t i = 0; i < size; i++) {
    result[i] = *(data + i);
  }
}

/*
 * @brief: read constvalue from paras store into values
 * @param [in] paras: ge::Operator
 * @param [in] const_input_idx: constvalue axes index
 * @param [out] values: vector to store return values.
 * @return bool: flag of success or not
 */
template <typename T>
bool GetConstIntData(const ge::Operator& paras, const int64_t const_input_idx, std::vector<T>& values) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(paras);
  ConstGeTensorBarePtr const_tensor = OpDescUtils::GetInputConstData(paras, const_input_idx);
  if (const_tensor == nullptr) {
    auto input_name = op_desc->GetInputNameByIndex(const_input_idx);
    OP_LOGW("GetConstIntData", "constvalue [%s] is not exists.", input_name.c_str());
    return false;
  }

  const auto& tensor_data = const_tensor->GetData();
  auto data = tensor_data.GetData();
  if (data == nullptr) {
    auto input_name = op_desc->GetInputNameByIndex(const_input_idx);
    OP_LOGW("GetConstIntData", "constvalue [%s] is nullptr.", input_name.c_str());
    return false;
  }
  auto size = tensor_data.GetSize();
  DataType dtype = op_desc->MutableInputDesc(const_input_idx)->GetDataType();
  switch (dtype) {
    case DT_UINT64:
      GetDataToVector<T, uint64_t>(data, size, values);
      break;
    case DT_INT64:
      GetDataToVector<T, int64_t>(data, size, values);
      break;
    case DT_UINT32:
      GetDataToVector<T, uint32_t>(data, size, values);
      break;
    case DT_INT32:
      GetDataToVector<T, int32_t>(data, size, values);
      break;
    default: {
      OP_LOGW("GetConstIntData", "GetConstValue of dtype[%d] has not implement.", dtype);
      return false;
    } break;
  }
  return true;
}

}  // namespace ops
#endif  // CANN_OPS_BUILT_IN_OPS_CONST_H_
