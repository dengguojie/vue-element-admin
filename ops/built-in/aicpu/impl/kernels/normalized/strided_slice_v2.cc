/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
#include "strided_slice_v2.h"

#include <algorithm>
#include <numeric>

#include "Eigen/Core"
#include "cpu_kernel_utils.h"
#include "strided_slice.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/kernel_util.h"

namespace {
const char *kStridedSliceV2 = "StridedSliceV2";
const char *kAxes = "axes";
const char *kStrides = "strides";
const char *kBeginMask = "begin_mask";
const char *kEndMask = "end_mask";
const char *kEllipsisMask = "ellipsis_mask";
const char *kNewAxisMask = "new_axis_mask";
const char *kShrinkAxisMask = "shrink_axis_mask";
}  // namespace

namespace aicpu {
uint32_t StridedSliceV2CpuKernel::CheckParam(const Tensor *begin,
                                             const Tensor *end,
                                             const Tensor *axes,
                                             const Tensor *strides) {
  DataType begin_type = begin->GetDataType();
  KERNEL_CHECK_FALSE((begin_type == end->GetDataType()),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Expect begin and end to be same data type, but got begin "
                     "data type[%d], end data type[%d]",
                     begin_type, end->GetDataType());
  auto begin_shape = begin->GetTensorShape();
  KERNEL_CHECK_NULLPTR(begin_shape, KERNEL_STATUS_PARAM_INVALID,
                       "Get input begin shape failed")
  auto end_shape = end->GetTensorShape();
  KERNEL_CHECK_NULLPTR(end_shape, KERNEL_STATUS_PARAM_INVALID,
                       "Get input end shape failed")
  KERNEL_CHECK_FALSE(
      ((begin_shape->GetDims() == 1) && (end_shape->GetDims() == 1) &&
       (begin_shape->NumElements() == end_shape->NumElements())),
      KERNEL_STATUS_PARAM_INVALID,
      "Expect begin and end to be 1d  equal size tensor, but got begin "
      "dims[%u] and element number[%lld], end dim size[%u] and element "
      "number[%lld].",
      begin_shape->GetDims(), begin_shape->NumElements(), end_shape->GetDims(),
      end_shape->NumElements());

  if (strides != nullptr) {
    KERNEL_CHECK_FALSE(
        (begin_type == strides->GetDataType()), KERNEL_STATUS_PARAM_INVALID,
        "Expect begin and strides to be same data type, but got begin "
        "data type[%d], strides data type[%d]",
        begin_type, strides->GetDataType());
    auto strides_shape = strides->GetTensorShape();
    KERNEL_CHECK_NULLPTR(strides_shape, KERNEL_STATUS_PARAM_INVALID,
                         "Get input strides shape failed")
    KERNEL_CHECK_FALSE(
        ((strides_shape->GetDims() == 1) &&
         (begin_shape->NumElements() == strides_shape->NumElements())),
        KERNEL_STATUS_PARAM_INVALID,
        "Expect begin and strides to be 1d  equal size tensor, but got begin "
        "dims[%u] and element number[%lld], strides dim size[%u] and element "
        "number[%lld].",
        begin_shape->GetDims(), begin_shape->NumElements(),
        strides_shape->GetDims(), strides_shape->NumElements());
  }

  if (axes != nullptr) {
    KERNEL_CHECK_FALSE(
        (begin_type == axes->GetDataType()), KERNEL_STATUS_PARAM_INVALID,
        "Expect begin and axes to be same data type, but got begin "
        "data type[%d], axes data type[%d]",
        begin_type, axes->GetDataType());
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t StridedSliceV2CpuKernel::BuildBeginParam(
    const std::shared_ptr<TensorShape> &x_shape, const Tensor *begin,
    std::vector<int64_t> &begin_vec) {
  int32_t x_dims = x_shape->GetDims();
  T *begin_data = static_cast<T *>(begin->GetData());
  KERNEL_CHECK_NULLPTR(begin_data, KERNEL_STATUS_PARAM_INVALID,
                       "Get input begin data failed")
  for (int64_t i = 0; i < begin->NumElements(); ++i) {
    begin_vec.push_back(static_cast<int64_t>(begin_data[i]));
  }

  for (int32_t i = static_cast<int32_t>(begin_vec.size()); i < x_dims; ++i) {
    begin_vec.push_back(0);
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t StridedSliceV2CpuKernel::BuildEndParam(
    const std::shared_ptr<TensorShape> &x_shape, const Tensor *end,
    std::vector<int64_t> &end_vec) {
  int32_t x_dims = x_shape->GetDims();
  T *end_data = static_cast<T *>(end->GetData());
  KERNEL_CHECK_NULLPTR(end_data, KERNEL_STATUS_PARAM_INVALID,
                       "Get input end data failed")
  for (int64_t i = 0; i < end->NumElements(); ++i) {
    end_vec.push_back(static_cast<int64_t>(end_data[i]));
  }

  for (int32_t i = static_cast<int32_t>(end_vec.size()); i < x_dims; ++i) {
    end_vec.push_back(x_shape->GetDimSize(i));
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t StridedSliceV2CpuKernel::BuildStridesParam(
    const std::shared_ptr<TensorShape> &x_shape, const Tensor *strides,
    std::vector<int64_t> &strides_vec) {
  int32_t x_dims = x_shape->GetDims();
  if (strides == nullptr) {
    for (int32_t i = 0; i < x_dims; ++i) {
      strides_vec.push_back(1);
    }
  } else {
    T *strides_data = static_cast<T *>(strides->GetData());
    KERNEL_CHECK_NULLPTR(strides_data, KERNEL_STATUS_PARAM_INVALID,
                         "Get input strides data failed")
    for (int64_t i = 0; i < strides->NumElements(); ++i) {
      strides_vec.push_back(static_cast<int64_t>(strides_data[i]));
    }

    for (int32_t i = static_cast<int32_t>(strides_vec.size()); i < x_dims;
         ++i) {
      strides_vec.push_back(1);
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t StridedSliceV2CpuKernel::BuildAxesParam(
    const std::shared_ptr<TensorShape> &x_shape, const Tensor *axes,
    std::vector<int64_t> &axes_vec) {
  int32_t x_dims = x_shape->GetDims();
  if (axes == nullptr) {
    axes_vec.resize(x_dims);
    std::iota(axes_vec.begin(), axes_vec.end(), 0);
  } else {
    T *axes_data = static_cast<T *>(axes->GetData());
    KERNEL_CHECK_NULLPTR(axes_data, KERNEL_STATUS_PARAM_INVALID,
                         "Get input axes data failed")
    for (int64_t i = 0; i < axes->NumElements(); ++i) {
      T axes_value = axes_data[i] < 0 ? axes_data[i] + x_dims : axes_data[i];
      KERNEL_CHECK_FALSE(
          ((axes_value >= 0) && (axes_value < x_dims)),
          KERNEL_STATUS_PARAM_INVALID,
          "Check axes[%lld] value[%lld] failed, must be in range [-%d, %d]", i,
          static_cast<int64_t>(axes_value), x_dims, x_dims - 1);
      KERNEL_CHECK_FALSE(
          (std::find(axes_vec.begin(), axes_vec.end(), axes_value) ==
           axes_vec.end()),
          KERNEL_STATUS_PARAM_INVALID,
          "Check value failed, axes[%lld] value[%lld] is repeated", i,
          static_cast<int64_t>(axes_value));
      axes_vec.push_back(static_cast<int64_t>(axes_value));
    }

    for (int32_t i = 0; i < x_dims; ++i) {
      if (std::find(axes_vec.begin(), axes_vec.end(), i) == axes_vec.end()) {
        axes_vec.push_back(static_cast<int64_t>(i));
      }
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t StridedSliceV2CpuKernel::BuildParam(
    const Tensor *x, const Tensor *begin, const Tensor *end, const Tensor *axes,
    const Tensor *strides, std::vector<int64_t> &begin_vec,
    std::vector<int64_t> &end_vec, std::vector<int64_t> &strides_vec) {
  auto x_shape = x->GetTensorShape();

  std::vector<int64_t> begin_ret;
  uint32_t ret = BuildBeginParam<T>(x_shape, begin, begin_ret);
  KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                     "Build begin parameter failed, ret[%u]", ret);

  std::vector<int64_t> end_ret;
  ret = BuildEndParam<T>(x_shape, end, end_ret);
  KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                     "Build end parameter failed, ret[%u]", ret);

  std::vector<int64_t> strides_ret;
  ret = BuildStridesParam<T>(x_shape, strides, strides_ret);
  KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                     "Build strides parameter failed, ret[%u]", ret);

  std::vector<int64_t> axes_ret;
  ret = BuildAxesParam<T>(x_shape, axes, axes_ret);
  KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                     "Build axes parameter failed, ret[%u]", ret);

  begin_vec.resize(axes_ret.size());
  end_vec.resize(axes_ret.size());
  strides_vec.resize(axes_ret.size());
  for (size_t i = 0; i < axes_ret.size(); ++i) {
    T axes_value = axes_ret[i];
    begin_vec[axes_value] = begin_ret[i];
    end_vec[axes_value] = end_ret[i];
    strides_vec[axes_value] = strides_ret[i];
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t StridedSliceV2CpuKernel::CheckAndBuildParam(
    const Tensor *x, const Tensor *begin, const Tensor *end, const Tensor *axes,
    const Tensor *strides, std::vector<int64_t> &begin_vec,
    std::vector<int64_t> &end_vec, std::vector<int64_t> &strides_vec) {
  uint32_t ret = CheckParam(begin, end, axes, strides);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }

  return BuildParam<T>(x, begin, end, axes, strides, begin_vec, end_vec,
                       strides_vec);
}

uint32_t StridedSliceV2CpuKernel::DoStridedSliceV2(
    CpuKernelContext &ctx, const std::vector<int64_t> &begin_vec,
    const std::vector<int64_t> &end_vec,
    const std::vector<int64_t> &strides_vec) {
  uint32_t ret = KERNEL_STATUS_OK;
#define STRIDED_SLICE_V2_CASE(DT, T)                                  \
  case (DT): {                                                        \
    ret = StridedSliceCpuKernel::CalStridedSlice<T>(                  \
        ctx, begin_vec, end_vec, strides_vec,                         \
        ctx.Input(0), ctx.Output(0));                                 \
    break;                                                            \
  }

  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    STRIDED_SLICE_V2_CASE(DT_INT8, int8_t)
    STRIDED_SLICE_V2_CASE(DT_INT16, int16_t)
    STRIDED_SLICE_V2_CASE(DT_INT32, int32_t)
    STRIDED_SLICE_V2_CASE(DT_INT64, int64_t)
    STRIDED_SLICE_V2_CASE(DT_UINT8, uint8_t)
    STRIDED_SLICE_V2_CASE(DT_UINT16, uint16_t)
    STRIDED_SLICE_V2_CASE(DT_UINT32, uint32_t)
    STRIDED_SLICE_V2_CASE(DT_UINT64, uint64_t)
    STRIDED_SLICE_V2_CASE(DT_FLOAT16, Eigen::half)
    STRIDED_SLICE_V2_CASE(DT_FLOAT, float)
    STRIDED_SLICE_V2_CASE(DT_DOUBLE, double)
    default:
      KERNEL_LOG_ERROR("%s kernel data type [%s] not support.", kStridedSliceV2,
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
#undef STRIDED_SLICE_V2_CASE
  return ret;
}

uint32_t StridedSliceV2CpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *x = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(x, KERNEL_STATUS_PARAM_INVALID, "Get input x failed")
  auto x_shape = x->GetTensorShape();
  KERNEL_CHECK_NULLPTR(x_shape, KERNEL_STATUS_PARAM_INVALID,
                       "Get input x shape failed")
  Tensor *begin = ctx.Input(1);
  KERNEL_CHECK_NULLPTR(begin, KERNEL_STATUS_PARAM_INVALID,
                       "Get input begin failed")
  Tensor *end = ctx.Input(2);
  KERNEL_CHECK_NULLPTR(end, KERNEL_STATUS_PARAM_INVALID, "Get input end failed")

  Tensor *axes = nullptr;
  Tensor *strides = nullptr;
  uint32_t input_size = ctx.GetInputsSize();
  KERNEL_LOG_INFO("Input size[%u]", input_size);
  for (uint32_t i = 3; i < input_size; ++i) {
    Tensor *tmp = ctx.Input(i);
    KERNEL_CHECK_NULLPTR(tmp, KERNEL_STATUS_PARAM_INVALID,
                         "Get input[%u] failed", i)
    std::string name = CpuKernelUtils::GetTensorName(tmp);
    KERNEL_LOG_INFO("Input[%u] info, name[%s]", i, name.c_str());
    if (name == kAxes) {
      axes = tmp;
    } else if (name == kStrides) {
      strides = tmp;
    }
  }

  int64_t begin_mask_value = 0;
  AttrValue *begin_mask = ctx.GetAttr(kBeginMask);
  if (begin_mask != nullptr) {
    begin_mask_value = begin_mask->GetInt();
  }

  int64_t end_mask_value = 0;
  AttrValue *end_mask = ctx.GetAttr(kEndMask);
  if (end_mask != nullptr) {
    end_mask_value = end_mask->GetInt();
  }

  int64_t ellipsis_mask_value = 0;
  AttrValue *ellipsis_mask = ctx.GetAttr(kEllipsisMask);
  if (ellipsis_mask != nullptr) {
    ellipsis_mask_value = ellipsis_mask->GetInt();
  }

  int64_t new_axis_mask_value = 0;
  AttrValue *new_axis_mask = ctx.GetAttr(kNewAxisMask);
  if (new_axis_mask != nullptr) {
    new_axis_mask_value = new_axis_mask->GetInt();
  }

  int64_t shrink_axis_mask_value = 0;
  AttrValue *shrink_axis_mask = ctx.GetAttr(kShrinkAxisMask);
  if (shrink_axis_mask != nullptr) {
    shrink_axis_mask_value = shrink_axis_mask->GetInt();
  }

  std::vector<int64_t> begin_vec;
  std::vector<int64_t> end_vec;
  std::vector<int64_t> strides_vec;
  std::vector<int64_t> x_shape_value = x_shape->GetDimSizes();

  DataType type = begin->GetDataType();
  uint32_t ret = KERNEL_STATUS_OK;
  if (type == DT_INT32) {
    ret = CheckAndBuildParam<int32_t>(x, begin, end, axes, strides, begin_vec,
                                      end_vec, strides_vec);
    KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), ret,
                       "Check and build parameter failed, ret[%u]", ret);
  } else if (type == DT_INT64) {
    ret = CheckAndBuildParam<int64_t>(x, begin, end, axes, strides, begin_vec,
                                      end_vec, strides_vec);
    KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), ret,
                       "Check and build parameter failed, ret[%u]", ret);
  } else {
    KERNEL_LOG_ERROR(
        "Unsupported input begin data_type[%d], only support DT_INT32 and "
        "DT_INT64.",
        type);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  KERNEL_LOG_INFO("Check and build parameters success.");

  ret = StridedSliceCpuKernel::InitParamsWithMasks(
      x_shape_value, begin_mask_value, end_mask_value, ellipsis_mask_value,
      new_axis_mask_value, shrink_axis_mask_value, begin_vec, end_vec,
      strides_vec);
  KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), ret,
                     "Init parameters with masks failed, ret[%u]", ret);

  return DoStridedSliceV2(ctx, begin_vec, end_vec, strides_vec);
}

REGISTER_CPU_KERNEL(kStridedSliceV2, StridedSliceV2CpuKernel);
}  // namespace aicpu
