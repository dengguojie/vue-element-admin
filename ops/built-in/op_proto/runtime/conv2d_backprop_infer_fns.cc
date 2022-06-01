/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
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
 * \file conv2d_backprop_infer_fns.cc
 * \brief
 */
#include <map>

#include "error_util.h"
#include "../util/util.h"
#include "graph/utils/type_utils.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_log.h"
#include "register/op_impl_registry.h"

using ge::Format;
using ge::GRAPH_FAILED;
using ge::graphStatus;
namespace {
constexpr size_t kConv2dDimSizeLimit = 4;
}  // namespace
namespace gert {
graphStatus InferShapeForConvBackprop(InferShapeContext *context, size_t const_tensor_idx,
                                      const char *const_tensor_name, size_t dim_num) {
  const auto op_name = context->GetNodeName();
  const auto y_shape = context->GetOutputShape(0);
  CHECK_PTR_NULL(y_shape, "y shape", return GRAPH_FAILED);

  auto const_tensor = context->GetInputTensor(const_tensor_idx);
  CHECK(const_tensor == nullptr, CUBE_INNER_ERR_REPORT(op_name, "get null %s tensor", const_tensor_name),
        return GRAPH_FAILED);
  size_t const_tensor_dim_num = const_tensor->GetStorageShape().GetDimNum();
  CHECK(const_tensor_dim_num != dim_num,
        CUBE_INNER_ERR_REPORT(op_name, "%s dim num %zu invalid", const_tensor_name, const_tensor_dim_num),
        return GRAPH_FAILED);
  y_shape->SetDimNum(dim_num);

  auto dtype = const_tensor->GetDataType();
  if (dtype == ge::DT_INT32) {
    auto tensor_data = const_tensor->GetData<int32_t>();
    CHECK(tensor_data == nullptr, CUBE_INNER_ERR_REPORT(op_name, "get null %s tensor data", const_tensor_name),
          return GRAPH_FAILED);
    for (size_t idx = 0; idx < const_tensor_dim_num; ++idx) {
      y_shape->SetDim(idx, tensor_data[idx]);
    }
  } else if (dtype == ge::DT_INT64) {
    auto tensor_data = const_tensor->GetData<int64_t>();
    CHECK(tensor_data == nullptr, CUBE_INNER_ERR_REPORT(op_name, "get null %s tensor data", const_tensor_name),
          return GRAPH_FAILED);
    for (size_t idx = 0; idx < const_tensor_dim_num; ++idx) {
      y_shape->SetDim(idx, tensor_data[idx]);
    }
  } else {
    CUBE_INNER_ERR_REPORT(op_name, "tensor %s not support dtype %s", const_tensor_name,
                          ge::TypeUtils::DataTypeToSerialString(dtype).c_str());
    return GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

graphStatus InferShapeForConv2DBackpropInput(InferShapeContext *context) {
  return InferShapeForConvBackprop(context, 0, "input_sizes", kConv2dDimSizeLimit);
}

IMPL_OP(Conv2DBackpropInput).InferShape(InferShapeForConv2DBackpropInput).InputsDataDependency({0});
IMPL_OP(DepthwiseConv2DBackpropInput).InferShape(InferShapeForConv2DBackpropInput).InputsDataDependency({0});
}  // namespace gert
