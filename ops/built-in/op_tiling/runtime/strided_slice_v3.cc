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
 * \file strided_slice_v3.ccc
 * \brief dynamic shape tiling of strided_slice_v3
 */
#include "strided_slice_v3.h"
#include <numeric>
#include <vector>
#include "strided_slice.h"
#include "runtime2_util.h"
#include "op_tiling_util.h"

namespace {
const std::string OP_NAME = "StridedSliceV3";
const int INDEX_X = 0;
const int INDEX_BEGIN = 1;
const int INDEX_END = 2;
const int INDEX_AXES = 3;
const int INDEX_STRIDES = 4;
const int INDEX_Y = 0;
}  // namespace

namespace optiling {
template <typename T>
static void PositiveAxisImpl(int32_t input_dims, const gert::Tensor* axes_tensor, std::vector<int64_t>& new_axes) {
  int32_t axes_size = static_cast<int32_t>(axes_tensor->GetShapeSize());
  const T* data = axes_tensor->GetData<T>();
  for (int32_t i = 0; i < axes_size; i++) {
    int32_t value = static_cast<int32_t>(data[i]);
    if (value >= 0 && value < input_dims) {
      new_axes.push_back(value);
    } else if (value < 0 && value >= -input_dims) {
      new_axes.push_back(value + input_dims);
    }
  }
  return;
}

static std::vector<int64_t> ConstructValidAxis(const gert::Tensor* axes_tensor, int32_t input_dims) {
  std::vector<int64_t> new_axes;
  if (!axes_tensor || !(axes_tensor->GetShapeSize())) {
    new_axes.resize(input_dims);
    std::iota(new_axes.begin(), new_axes.end(), 0);
    return new_axes;
  }
  if (axes_tensor->GetDataType() == ge::DT_INT32) {
    PositiveAxisImpl<int32_t>(input_dims, axes_tensor, new_axes);
  } else {
    PositiveAxisImpl<int64_t>(input_dims, axes_tensor, new_axes);
  }
  return new_axes;
}

static void ConstructSliceShape(const gert::Shape& shape, int32_t dim_num, std::vector<int64_t>& param) {
  param.resize(dim_num);
  for (int32_t i = 0; i < dim_num; i++) {
    param[i] = shape.GetDim(i);
  }
  return;
}

static int64_t GetConstIndexValue(const gert::Tensor* tensor, int32_t idx) {
  // idx must be valid
  int64_t value = 0;
  if (tensor->GetDataType() == ge::DT_INT32) {
    const int32_t* data = tensor->GetData<int32_t>();
    value = static_cast<int64_t>(data[idx]);
  } else {
    const int64_t* data = tensor->GetData<int64_t>();
    value = data[idx];
  }
  OP_LOGD(OP_NAME.c_str(), "const tensor[%lld] is %ld.", idx, value);
  return value;
}

static int64_t GetConstIndexValue(const gert::Tensor* tensor, int32_t idx, int64_t input_size, int64_t clip_lower,
                                  int64_t clip_upper) {
  // idx must be valid
  int64_t value = 0;
  if (tensor->GetDataType() == ge::DT_INT32) {
    const int32_t* data = tensor->GetData<int32_t>();
    value = static_cast<int64_t>(data[idx]);
  } else {
    const int64_t* data = tensor->GetData<int64_t>();
    value = data[idx];
  }
  if (value < 0) {
    value += input_size;
  }

  // clamp value
  if (value < clip_lower) {
    value = clip_lower;
  } else if (value > clip_upper) {
    value = clip_upper;
  }
  OP_LOGD(OP_NAME.c_str(), "const tensor[%lld] is %ld.", idx, value);
  return value;
}

static void ConstructStrideList(const gert::Tensor* stride_tensor, int64_t dim_num, const std::vector<int64_t>& axes,
                                std::vector<int64_t>& stride) {
  stride.assign(dim_num, 1);
  if (!stride_tensor) {
    OP_LOGD(OP_NAME, "Stride tensor is null. Set stride as 1.");
    return;
  }
  int32_t stride_size = static_cast<int32_t>(stride_tensor->GetShapeSize());
  for (int32_t i = 0; i < dim_num && i < stride_size; i++) {
    int64_t axes_value = axes[i];
    stride[axes_value] = GetConstIndexValue(stride_tensor, i);
  }
  return;
}

static void ConstructBeginList(const gert::Tensor* begin_tensor, const gert::Shape& x_shape,
                               const std::vector<int64_t>& axes, std::vector<int64_t>& begin_vec) {
  const int32_t dim_num = static_cast<int32_t>(x_shape.GetDimNum());
  begin_vec.assign(dim_num, 0);
  const int32_t begins_size = static_cast<int32_t>(begin_tensor->GetShapeSize());
  for (int32_t i = 0; i < dim_num && i < begins_size; i++) {
    int64_t axes_value = axes[i];
    int64_t clip_upper = x_shape.GetDim(axes_value);
    begin_vec[axes_value] = GetConstIndexValue(begin_tensor, i, clip_upper, 0, clip_upper);
  }
  return;
}

static void ConstructEndList(const gert::Tensor* end_tensor, const gert::Shape& x_shape,
                             const std::vector<int64_t>& axes, std::vector<int64_t>& end_vec) {
  const int32_t dim_num = static_cast<int32_t>(x_shape.GetDimNum());
  end_vec.resize(dim_num);
  const int32_t end_size = static_cast<int32_t>(end_tensor->GetShapeSize());
  for (int32_t i = 0; i < dim_num; i++) {
    int64_t axes_value = axes[i];
    int64_t clip_upper = x_shape.GetDim(axes_value);
    int64_t end_value = clip_upper;
    if (i < end_size) {
      end_value = GetConstIndexValue(end_tensor, i, clip_upper, 0, clip_upper);
    }
    end_vec[axes_value] = end_value;
  }
  return;
}

static void AppendTilingData(const SliceParameters& params, gert::TilingData* tiling_data) {
  size_t shape_length = params.input.size();
  tiling_data->Append<int64_t>(params.tiling_mode);
  tiling_data->Append<int64_t>(shape_length);
  const vector<int64_t>* tiling_params[] = {
      &params.input, &params.output_shape, &params.begin_list, &params.end_list, &params.stride_list,
  };

  for (auto item : tiling_params) {
    tiling_data->Append<int64_t>(item->data(), item->size());
  }
  return;
}

ge::graphStatus TilingPrepareForStridedSliceV3(gert::TilingParseContext* context) {
  auto compile_info = MutableCompileInfo<StridedSliceV3CompileInfo>(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  std::unique_ptr<nlohmann::json> parsed_object_cinfo = GetJsonObj(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, parsed_object_cinfo);
  const nlohmann::json& vars = (*parsed_object_cinfo)["vars"];
  OP_TILING_CHECK(vars.empty(), VECTOR_INNER_ERR_REPORT_TILIING(OP_NAME, "get vars failed."), return ge::GRAPH_FAILED);
  GetCompileValue(vars, "block_dim", compile_info->block_dim);
  GetCompileValue(vars, "ub_size", compile_info->ub_size);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForStridedSliceV3(gert::TilingContext* context) {
  const gert::StorageShape* x_storage = context->GetInputShape(INDEX_X);
  OPS_CHECK_NULL_WITH_CONTEXT(context, x_storage);
  const gert::StorageShape* y_storage = context->GetOutputShape(INDEX_Y);
  OPS_CHECK_NULL_WITH_CONTEXT(context, y_storage);

  struct SliceParameters slice_param;
  const gert::Shape x_shape = x_storage->GetOriginShape();
  int32_t input_dim_num = static_cast<int32_t>(x_shape.GetDimNum());
  ConstructSliceShape(x_shape, input_dim_num, slice_param.input);
  const gert::Shape y_shape = y_storage->GetOriginShape();
  ConstructSliceShape(y_shape, input_dim_num, slice_param.output_shape);

  std::vector<int64_t> new_axes = ConstructValidAxis(context->GetInputTensor(INDEX_AXES), input_dim_num);
  const gert::Tensor* begin_tensor = context->GetInputTensor(INDEX_BEGIN);
  OPS_CHECK_NULL_WITH_CONTEXT(context, begin_tensor);
  ConstructBeginList(begin_tensor, x_shape, new_axes, slice_param.begin_list);
  const gert::Tensor* end_tensor = context->GetInputTensor(INDEX_END);
  OPS_CHECK_NULL_WITH_CONTEXT(context, end_tensor);
  ConstructEndList(end_tensor, x_shape, new_axes, slice_param.end_list);
  ConstructStrideList(context->GetInputTensor(INDEX_STRIDES), input_dim_num, new_axes, slice_param.stride_list);

  MakePerformanceParams(slice_param);
  OP_LOGD(OP_NAME.c_str(), "perf slice params: %s", slice_param.to_string().c_str());

  auto compile_info = reinterpret_cast<const StridedSliceV3CompileInfo*>(context->GetCompileInfo());
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  const gert::Tensor* x_tensor = context->GetInputTensor(INDEX_X);
  OPS_CHECK_NULL_WITH_CONTEXT(context, x_tensor);
  SetTilingMode(slice_param, compile_info->block_dim, x_tensor->GetDataType(), compile_info->ub_size, OP_NAME);
  OP_LOGD(OP_NAME.c_str(), "set tiling_mode params: %s", slice_param.to_string().c_str());

  auto tiling_data = context->GetRawTilingData();
  OPS_CHECK_NULL_WITH_CONTEXT(context, tiling_data);
  AppendTilingData(slice_param, tiling_data);
  context->SetBlockDim(compile_info->block_dim);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(StridedSliceV3)
    .Tiling(TilingForStridedSliceV3)
    .TilingParse<StridedSliceV3CompileInfo>(TilingPrepareForStridedSliceV3);
}  // namespace optiling
