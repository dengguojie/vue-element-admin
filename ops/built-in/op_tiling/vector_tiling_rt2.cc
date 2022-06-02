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
 * \file vector_tiling_rt2.cc
 * \brief tiling function of vector ops
 */

#include "vector_tiling_rt2.h"
#include "register/op_tiling_attr_utils.h"
#include "vector_tiling_log.h"
#include "error_log.h"


namespace optiling {
namespace {
constexpr int32_t VAR_ATTR_MODE_NOT_EXIST = -1;
constexpr int32_t VAR_ATTR_MODE_CONSISTENT = 0;
constexpr int32_t VAR_ATTR_MODE_INDEPENDENT = 1;
}

void OpInfoImpl::SetInputShape(const std::vector<gert::Shape>* _op_input_ge_shapes) {
  op_input_ge_shapes_ptr = _op_input_ge_shapes;
  if (_op_input_ge_shapes) {
    op_input_shapes.resize(_op_input_ge_shapes->size());
    for (size_t i = 0; i < _op_input_ge_shapes->size(); i++) {
      std::vector<int64_t> shape(_op_input_ge_shapes->at(i).GetDimNum());
      for (size_t j = 0; j < _op_input_ge_shapes->at(i).GetDimNum(); j++) {
        shape[j] = _op_input_ge_shapes->at(i).GetDim(j);
      }
      op_input_shapes[i] = shape;
    }
  }
}

void OpInfoImpl::SetInputShape(const std::vector<std::vector<int64_t>>* _op_input_shapes) {
  op_input_shapes_ptr = _op_input_shapes;
  if (_op_input_shapes) {
    op_input_ge_shapes.resize(_op_input_shapes->size());
    for (size_t i = 0; i < _op_input_shapes->size(); i++) {
      gert::Shape shape;
      for (size_t j = 0; j < _op_input_shapes->at(i).size(); j++) {
        shape.AppendDim(_op_input_shapes->at(i)[j]);
      }
      op_input_ge_shapes[i] = shape;
    }
  }
}

void OpInfoImpl::SetAxes(const std::vector<int64_t>* _op_axes) {
  op_axes_ptr = _op_axes;
}

void OpInfoImpl::SetAxes(const std::vector<int32_t>* _op_axes) {
  if (_op_axes) {
    op_axes_d = {*_op_axes};
    op_axes.resize(_op_axes->size());
    for (size_t i = 0; i < _op_axes->size(); i++) {
      op_axes[i] = static_cast<int64_t>(_op_axes->at(i));
    }
  }
}

void OpInfoImpl::SetInputType(const ge::DataType* _op_in_type) {
  op_in_type = _op_in_type;
}

const std::vector<std::vector<int64_t>>* OpInfoImpl::GetInputShape() const {
  if (op_input_shapes_ptr) {
    return op_input_shapes_ptr;
  }
  if (op_input_shapes.empty()) {
    return nullptr;
  }
  return &op_input_shapes;
}

const std::vector<gert::Shape>* OpInfoImpl::GetInputGeShape() const {
  if (op_input_ge_shapes_ptr) {
    return op_input_ge_shapes_ptr;
  }
  if (op_input_ge_shapes.empty()) {
    return nullptr;
  }
  return &op_input_ge_shapes;
}

const std::vector<int64_t>* OpInfoImpl::GetAxes() const {
  if (op_axes_ptr) {
    return op_axes_ptr;
  }
  if (op_axes.empty()) {
    return nullptr;
  }
  return &op_axes;
}

const ge::DataType* OpInfoImpl::GetInType() const {
  return op_in_type;
}

const AutoTilingCompileInfo* OpInfoImpl::GetCompileInfo() const {
  return compile_info;
}

// Compatible code, please do not use
const std::vector<std::vector<int64_t>>& OpInfoImpl::GetInputShapeD() const {
  return *op_input_shapes_ptr;
}

// Compatible code, please do not use
const std::vector<std::vector<int32_t>>& OpInfoImpl::GetReduceAxesD() const {
  return op_axes_d;
}

// Compatible code, please do not use
const ge::DataType& OpInfoImpl::GetInTypeD() const {
  return *op_in_type;
}
} // namespace optiling