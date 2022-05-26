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

namespace optiling {
template <typename T>
T* MutableCompileInfo(gert::TilingParseContext* context) {
  return context->GetCompiledInfo<T>();
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
 * @brief: add reduce cof value after the tiling data
 * @param [in] input_shape: gert::Shape, the input shape for reduce
 * @param [in] input_dtype: ge::DataType,  the input dtype for reduce
 * @param [in] reduce_axis: const std::vector<int32_t>, the reduce axes num
 * @param [out] tiling_data: gert::TilingData, the tiling data, will add the cof value to th lasy TilingData
 * @return bool: true or false;
 */
bool AddReducMeanCof(const gert::Shape& input_shape, const ge::DataType input_dtype,
                     const std::vector<int32_t>& reduce_axis, gert::TilingData* tiling_data);
}  // namespace optiling
#endif  // CANN_OPS_BUILT_IN_OP_TILING_RUNTIME2_UTIL_H_
