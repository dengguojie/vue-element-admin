/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file runtime2_util.h
 * \brief
 */

#ifndef CANN_OPS_BUILT_IN_OP_TILING_RUNTIME2_UTIL_H_
#define CANN_OPS_BUILT_IN_OP_TILING_RUNTIME2_UTIL_H_

#include <nlohmann/json.hpp>
#include "register/op_impl_registry.h"
#include "runtime/continuous_vector.h"
#include "runtime/storage_shape.h"
#include "runtime/tiling_context.h"
#include "runtime/tiling_parse_context.h"
#include "context_util.h"
#include "op_util.h"
#include "../error_log.h"
#include "../fusion_pass/common/fp16_t.hpp"

namespace optiling {
template <typename T>
T* MutableCompileInfo(gert::TilingParseContext* context) {
  return context->GetCompiledInfo<T>();
}

/*
 * @brief: Calculate reduce cof value
 * @param [in] input_shape: gert::Shape, the input shape for reduce
 * @param [in] reduce_axis: const std::vector<int32_t>, the reduce axes num
 * @param [out] reduce_mean_cof: the result of reduce cof value
 * @return bool: true or false;
 */
template <typename T>
bool CalcReduceMeanCof(const gert::Shape& input_shape, const std::vector<T>& reduce_axis,
                       float& reduce_mean_cof) {
  const size_t dim_len = input_shape.GetDimNum();
  const size_t ori_reduce_axis_len = reduce_axis.size();
  // init reduce_mean_cof is 1.0
  reduce_mean_cof = 1.0;
  for (size_t i = 0; i < ori_reduce_axis_len; i++) {
    OP_TILING_CHECK(
        !ops::IsDimValid(dim_len, reduce_axis[i]),
        VECTOR_INNER_ERR_REPORT_TILIING("CalcReduceMeanCof", "%s",
                                        ops::GenInvalidDimMsg("reduce_axis", i, dim_len, reduce_axis[i]).c_str()),
        return false);

    // convert reduce axis (like: -1 -> (dim_len - 1))
    T single_reduce_axis = reduce_axis[i] < 0 ? reduce_axis[i] + dim_len : reduce_axis[i];

    int64_t reduce_dim = input_shape.GetDim(single_reduce_axis);
    OP_TILING_CHECK(reduce_dim == 0, OP_LOGI("CalcReduceMeanCof", "the reduce dim is 0, will ignore reduce_mean_cof"),
                    return true);
    reduce_mean_cof = reduce_mean_cof / reduce_dim;
  }
  OP_LOGD("CalcReduceMeanCof", "CalcReduceMeanCof cof is %1f", reduce_mean_cof);

  return true;
}

/*
 * @brief: add reduce cof value after the tiling data
 * @param [in] input_shape: gert::Shape, the input shape for reduce
 * @param [in] input_dtype: ge::DataType,  the input dtype for reduce
 * @param [in] reduce_axis: const std::vector<int32_t>, the reduce axes num
 * @param [out] tiling_data: gert::TilingData, the tiling data, will add the cof value to th lasy TilingData
 * @return bool: true or false;
 */
template <typename T>
bool AddReduceMeanCof(const gert::Shape& input_shape, const ge::DataType input_dtype,
                      const std::vector<T>& reduce_axis, gert::TilingData* tiling_data) {
  float reduce_mean_cof = 1.0;
  bool calcu_flag = CalcReduceMeanCof(input_shape, reduce_axis, reduce_mean_cof);
  OP_LOGD("AddReduceMeanCof", "AddReduceMeanCof dtype is %s", ops::ToString(input_dtype).c_str());
  switch (input_dtype) {
    case ge::DT_FLOAT:
      tiling_data->Append((float)reduce_mean_cof);
      return calcu_flag;
    case ge::DT_FLOAT16:
      tiling_data->Append((fe::fp16_t)reduce_mean_cof);
      tiling_data->Append((uint16_t)0);
      return calcu_flag;
    default:
      OP_LOGW("AddReduceMeanCof", "Only support [DT_FLOAT, DT_FLOAT16], but is [%s]",
              ops::ToString(input_dtype).c_str());
      return false;
  }
}

/*
 * @brief: get the json class of compile info from context
 * @param [in] context: gert::TilingContext
 * @return bool: std::unique_ptr<nlohmann::json>;
 */
std::unique_ptr<nlohmann::json> GetJsonObj(gert::TilingParseContext* context);

/*
 * @brief: add workspace size to context
 * @param [in] context: gert::TilingContext
 * @param [in] workspace: size_t the workspace num
 * @return bool: true or false;
 */
bool AddWorkspace(gert::TilingContext* context, const size_t workspace);

/*
 * @brief: ceil(u_value/d_value)*d_value
 *         eg. CeilAlign(4,3) -> 6, CeilAlign(4,2) -> 4, CeilAlign(4,0) -> 4
 * @param [in] u_value: int64_t
 * @param [in] d_value: int64_t
 * @return int64: ceil
 */
int64_t CeilAlign(int64_t u_value, int64_t d_value);

/*
 * @brief: if d_value == 0 return u_value, else return u_value % d_value
 * @param [in] u_value: int64_t
 * @param [in] d_value: int64_t
 * @return int64: ceil
 */
int64_t GetRemainder(int64_t u_value, int64_t d_value);

/*
 * @brief: calculate the shape size of shape from begin to end
 * @param [in] shape: gert::Shape, the input shape
 * @param [in] begin: size_t, the begin point
 * @param [in] end: size_t, the end point
 * @return int64: the total shape size for begin to end
 */
int64_t GetPartShapeSize(const gert::Shape& shape, size_t begin, size_t end);

/*
 * @brief: for debug print runtime2 tiling data
 * @param [in] context: gert::TilingContext
 * @param [out] string: tiling data string
 * @return string: result
 */
template <typename T>
std::string GetTilingDataString(gert::TilingContext* context) {
  auto tiling_data = context->GetRawTilingData();
  auto data_size = tiling_data->GetDataSize();
  std::string result;
  const T *data = reinterpret_cast<const T*>(tiling_data->GetData());
  size_t len = data_size / sizeof(T);
  for (size_t i = 0; i < len; i++) {
    result += std::to_string(data[i]);
    result += " ";
  }
  return result;
}
}  // namespace optiling
#endif  // CANN_OPS_BUILT_IN_OP_TILING_RUNTIME2_UTIL_H_
