/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file vector_op_info.cc
 * \brief tiling function of vector ops
 */

#include "vector_op_info.h"
#include "register/op_tiling_attr_utils.h"
#include "vector_tiling_log.h"
#include "vector_tiling_rt2.h"

namespace optiling {
OpInfo::OpInfo(const std::vector<std::vector<int64_t>>& _op_input_shapes,
               const ge::DataType& _op_in_type,
               const std::vector<std::vector<int32_t>>& _op_axes) {
  op_info_impl = std::make_shared<OpInfoImpl>();
  op_info_impl->SetInputShape(&_op_input_shapes);
  op_info_impl->SetInputType(&_op_in_type);
  op_info_impl->SetAxes(&_op_axes[0]);
}

OpInfo::OpInfo(const std::vector<std::vector<int64_t>>& _op_input_shapes,
               const ge::DataType& _op_in_type) {
  op_info_impl = std::make_shared<OpInfoImpl>();
  op_info_impl->SetInputShape(&_op_input_shapes);
  op_info_impl->SetInputType(&_op_in_type);
}

OpInfo::OpInfo(const AutoTilingCompileInfo* _compile_info) {
  op_info_impl = std::make_shared<OpInfoImpl>(_compile_info);
}

void OpInfo::SetInputShape(const std::vector<gert::Shape>* _op_input_ge_shapes) {
  op_info_impl->SetInputShape(_op_input_ge_shapes);
}

void OpInfo::SetInputShape(const std::vector<std::vector<int64_t>>* _op_input_shapes) {
  op_info_impl->SetInputShape(_op_input_shapes);
}

void OpInfo::SetAxes(const std::vector<int64_t>* _op_axes) {
  op_info_impl->SetAxes(_op_axes);
}

void OpInfo::SetInputType(ge::DataType* _op_in_type) {
  op_info_impl->SetInputType(_op_in_type);
}

// Compatible code, please do not use
const std::vector<std::vector<int64_t>>& OpInfo::GetInputShape() const {
  return op_info_impl->GetInputShapeD();
}

// Compatible code, please do not use
const std::vector<std::vector<int32_t>>& OpInfo::GetReduceAxes() const {
  return op_info_impl->GetReduceAxesD();
}

// Compatible code, please do not use
const ge::DataType& OpInfo::GetInType() const {
  return op_info_impl->GetInTypeD();
}
 } // namespace optiling