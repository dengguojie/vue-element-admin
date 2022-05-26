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
 * \file op_util.h
 * \brief
 */

#ifndef CANN_OPS_BUILT_IN_OP_UTIL_H_
#define CANN_OPS_BUILT_IN_OP_UTIL_H_

#include <memory>
#include <utility>
#include "runtime/shape.h"
#include "error_util.h"

namespace ops {

template <typename _T, typename... _Args>
inline std::shared_ptr<_T> make_shared_nothrow(_Args&&... __args) noexcept(
    noexcept(_T(std::forward<_Args>(__args)...))) {
  try {
    return std::make_shared<_T>(std::forward<_Args>(__args)...);
  } catch (...) {
    return std::shared_ptr<_T>();
  }
}

template <typename T1, typename T2>
bool IsDimValid(const T1 shape_size, const T2 dim_value) {
  int64_t minimum_num = static_cast<int64_t>(shape_size) * (-1);
  int64_t maximum_num = static_cast<int64_t>(shape_size) - 1;

  return static_cast<int64_t>(dim_value) >= minimum_num && static_cast<int64_t>(dim_value) <= maximum_num;
}

template <typename T1, typename T2>
std::string GenInvalidDimMsg(const std::string dim_name, const T1 shape_size, const T2 dim_value) {
  std::string wrong_val = ge::ConcatString(static_cast<int64_t>(dim_value));
  // will be "[-rank, rank)"
  std::string neg_rank = ge::ConcatString(static_cast<int64_t>(shape_size) * (-1));
  std::string expect_val =
      ge::ConcatString("[", neg_rank, ", ", ge::ConcatString(static_cast<int64_t>(shape_size)), ")");

  return ge::GetAttrValueErrMsg(dim_name, wrong_val, expect_val);
}

template <typename T1, typename T2>
std::string GenInvalidDimMsg(const std::string dim_name, const size_t dim_idx, const T1 shape_size,
                             const T2 dim_value) {
  std::string invalid_dim_name = ge::ConcatString(dim_name, "[", ge::ConcatString(dim_idx), "]");

  return GenInvalidDimMsg(invalid_dim_name, shape_size, dim_value);
}

template <typename T>
T CeilDiv(T x, T y) {
  return y == 0 ? x : (x + y - 1) / y;
}

/*
 * @brief: get datatype string from enum
 * @param [in] type: enum datatype
 * @return string: datatype string
 */
std::string ToString(const ge::DataType& type);

/*
 * @brief: get format string from enum
 * @param [in] format: enum format
 * @return string: format string
 */
std::string ToString(const ge::Format& format);

/*
 * @brief: get shape string from gert::Shape, for debug
 * @param [in] format: enum format
 * @return string: shape string
 */
std::string ToString(const gert::Shape& shape);

/*
 * @brief: trans the gert::Shape to vector<int64_t>
 * @param [in] format: gert::Shape
 * @return vector<int64_t>: the vector shape
 */
std::vector<int64_t> ToVector(const gert::Shape& shape);
}  // namespace ops
#endif  // CANN_OPS_BUILT_IN_OP_UTIL_H_
