/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021.All rights reserved.
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

#include "grid_sampler_2d.h"

#include "cmath"
#include "cpu_kernel_utils.h"
#include "iostream"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
using namespace std;
namespace {
const char *kGridSampler2D = "GridSampler2D";
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 1;
const int64_t Max_Size = 64 * 1024;
const int64_t Max_Size2 = 8 * 1024;

#define GRIDSAMPLER2D_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                               \
    uint32_t result = GridSampler2DCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                           \
      KERNEL_LOG_ERROR("GridSampler2D kernel compute failed."); \
      return result;                                            \
    }                                                           \
    break;                                                      \
  }

#define GRIDSAMPLER2D_COMPUTE_CASE_HALF(DTYPE, CTX)             \
  case (DTYPE): {                                               \
    uint32_t result = GridSampler2DCompute_half(CTX);           \
    if (result != KERNEL_STATUS_OK) {                           \
      KERNEL_LOG_ERROR("GridSampler2D kernel compute failed."); \
      return result;                                            \
    }                                                           \
    break;                                                      \
  }
}  // namespace

namespace aicpu {
uint32_t GridSampler2DCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "GridSampler2D check input and output number failed.");
  Tensor *x_data = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(x_data->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input[0] failed.")
  Tensor *grid_data = ctx.Input(1);
  KERNEL_CHECK_NULLPTR(grid_data->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input[1] failed.")
  Tensor *y_data = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(y_data->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input[0] failed.")
  DataType x_type = x_data->GetDataType();
  DataType grid_type = grid_data->GetDataType();
  if (x_type != grid_type) {
    KERNEL_LOG_ERROR(
        "Input[0] and input[1] must have same dtype, but get [%s] and [%s].",
        DTypeStr(x_type).c_str(), DTypeStr(grid_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  switch (x_type) {
    GRIDSAMPLER2D_COMPUTE_CASE_HALF(DT_FLOAT16, ctx)
    GRIDSAMPLER2D_COMPUTE_CASE(DT_FLOAT, float, ctx)
    GRIDSAMPLER2D_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("Input data type[%s] not supported.",
                       DTypeStr(x_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t GetAttrValue(CpuKernelContext &ctx, std::string &interpolation_mode,
                      std::string &padding_mode, bool &align_corners) {
  AttrValue *attr1 = ctx.GetAttr("interpolation_mode");
  if (attr1 == nullptr) {
    interpolation_mode = "bilinear";
  } else {
    interpolation_mode = attr1->GetString();
  }
  if (interpolation_mode != "bilinear" && interpolation_mode != "nearest") {
    KERNEL_LOG_ERROR(
        "The value of interpolation_mode must be bilinear or nearest, but get "
        "[%s].",
        interpolation_mode.c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  AttrValue *attr2 = ctx.GetAttr("padding_mode");
  if (attr2 == nullptr) {
    padding_mode = "zeros";
  } else {
    padding_mode = attr2->GetString();
  }
  if (padding_mode != "zeros" && padding_mode != "border" &&
      padding_mode != "reflection") {
    KERNEL_LOG_ERROR(
        "The value of padding_mode must be one of border, reflection and "
        "zeros, but get [%s].",
        padding_mode.c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  AttrValue *attr3 = ctx.GetAttr("align_corners");
  if (attr3 == nullptr) {
    align_corners = false;
  } else {
    align_corners = attr3->GetBool();
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t GridSampler2DCpuKernel::GridSampler2DCompute(CpuKernelContext &ctx) {
  Tensor *x_data = ctx.Input(0);
  auto x_data_addr = reinterpret_cast<T *>(x_data->GetData());
  auto x_shape = x_data->GetTensorShape();
  auto x_dims = x_shape->GetDimSizes();
  int64_t x_stride[4];
  int64_t stride_tmp = 1;
  for (int32_t i = 3; i > -1; i--) {
    x_stride[i] = stride_tmp;
    stride_tmp *= x_dims[i];
  }
  Tensor *grid_data = ctx.Input(1);
  auto grid_data_addr = reinterpret_cast<T *>(grid_data->GetData());
  auto grid_shape = grid_data->GetTensorShape();
  auto grid_dims = grid_shape->GetDimSizes();
  if (grid_dims[0] != x_dims[0] || grid_dims[INPUT_NUM3] != INPUT_NUM2) {
    KERNEL_LOG_ERROR("The shape of grid [%d, %d, %d,  %d] is invalid.",
                     grid_dims[0], grid_dims[1], grid_dims[INPUT_NUM2],
                     grid_dims[INPUT_NUM3]);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int64_t grid_stride[4];
  stride_tmp = 1;
  for (int32_t i = 3; i > -1; i--) {
    grid_stride[i] = stride_tmp;
    stride_tmp *= grid_dims[i];
  }
  Tensor *y_data = ctx.Output(0);
  auto y_data_addr = reinterpret_cast<T *>(y_data->GetData());
  auto y_shape = y_data->GetTensorShape();
  auto y_dims = y_shape->GetDimSizes();
  int64_t y_stride[4];
  stride_tmp = 1;
  for (int32_t i = 3; i > -1; i--) {
    y_stride[i] = stride_tmp;
    stride_tmp *= y_dims[i];
  }
  uint64_t y_data_size = y_data->GetDataSize();
  std::string interpolation_mode;
  std::string padding_mode;
  bool align_corners;
  GetAttrValue(ctx, interpolation_mode, padding_mode, align_corners);
  if (y_data_size < Max_Size) {
    Call1(x_data_addr, y_data_addr, grid_data_addr, x_dims, y_dims, y_stride,
          x_stride, grid_stride, interpolation_mode, padding_mode,
          align_corners);
  } else {
    Call2(ctx, x_data_addr, y_data_addr, grid_data_addr, x_dims, y_dims,
          y_stride, x_stride, grid_stride, interpolation_mode, padding_mode,
          align_corners);
  }
  return KERNEL_STATUS_OK;
}

uint32_t GridSampler2DCpuKernel::GridSampler2DCompute_half(
    CpuKernelContext &ctx) {
  Tensor *x_data = ctx.Input(0);
  auto x_data_addr = reinterpret_cast<Eigen::half *>(x_data->GetData());
  auto x_shape = x_data->GetTensorShape();
  auto x_dims = x_shape->GetDimSizes();
  int64_t x_stride[4];
  int64_t stride_tmp = 1;
  for (int32_t i = 3; i > -1; i--) {
    x_stride[i] = stride_tmp;
    stride_tmp *= x_dims[i];
  }
  Tensor *grid_data = ctx.Input(1);
  auto grid_data_addr = reinterpret_cast<Eigen::half *>(grid_data->GetData());
  auto grid_shape = grid_data->GetTensorShape();
  auto grid_dims = grid_shape->GetDimSizes();
  if (grid_dims[0] != x_dims[0] || grid_dims[INPUT_NUM3] != INPUT_NUM2) {
    KERNEL_LOG_ERROR("The shape of grid [%d, %d, %d, %d] is invalid.",
                     grid_dims[0], grid_dims[1], grid_dims[INPUT_NUM2],
                     grid_dims[INPUT_NUM3]);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int64_t grid_stride[4];
  stride_tmp = 1;
  for (int32_t i = 3; i > -1; i--) {
    grid_stride[i] = stride_tmp;
    stride_tmp *= grid_dims[i];
  }
  Tensor *y_data = ctx.Output(0);
  auto y_data_addr = reinterpret_cast<Eigen::half *>(y_data->GetData());
  auto y_shape = y_data->GetTensorShape();
  auto y_dims = y_shape->GetDimSizes();
  int64_t y_stride[4];
  stride_tmp = 1;
  for (int32_t i = 3; i > -1; i--) {
    y_stride[i] = stride_tmp;
    stride_tmp *= y_dims[i];
  }
  uint64_t y_data_size = y_data->GetDataSize();
  std::string interpolation_mode;
  std::string padding_mode;
  bool align_corners;
  GetAttrValue(ctx, interpolation_mode, padding_mode, align_corners);
  if (y_data_size < Max_Size2) {
    Call1Half(x_data_addr, y_data_addr, grid_data_addr, x_dims, y_dims,
              y_stride, x_stride, grid_stride, interpolation_mode, padding_mode,
              align_corners);
  } else {
    Call2Half(ctx, x_data_addr, y_data_addr, grid_data_addr, x_dims, y_dims,
              y_stride, x_stride, grid_stride, interpolation_mode, padding_mode,
              align_corners);
  }
  return KERNEL_STATUS_OK;
}

uint32_t GridSampler2DCpuKernel::Call2Half(
    CpuKernelContext &ctx, Eigen::half *x_data_addr, Eigen::half *y_data_addr,
    Eigen::half *grid_data_addr, std::vector<int64_t> x_dims,
    std::vector<int64_t> y_dims, int64_t *y_stride, int64_t *x_stride,
    int64_t *grid_stride, std::string interpolation_mode,
    std::string padding_mode, bool align_corners) {
  uint32_t data_num = y_dims[0] * y_dims[2] * y_dims[3];
  uint32_t min_core_num = 1;
  size_t max_core_num =
      std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
  if (max_core_num > data_num) {
    max_core_num = data_num;
  }
  auto shard_compute = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      int64_t y_iter[4] = {0};
      y_iter[1] = static_cast<int64_t>(i);
      int64_t count = 3;
      while (y_iter[1] > 0) {
        if (count == 1) {
          count--;
        }
        y_iter[count] = y_iter[1] % y_dims[count];
        y_iter[1] /= y_dims[count--];
      }
      const int64_t y_c = y_dims[1];
      int64_t grid_offset = y_iter[0] * grid_stride[0] +
                            y_iter[2] * grid_stride[1] +
                            y_iter[3] * grid_stride[2];
      float x = static_cast<float>(grid_data_addr[grid_offset]);
      float y =
          static_cast<float>(grid_data_addr[grid_offset + grid_stride[3]]);
      x = grid_sampler_compute_source_index(x, x_dims[3], padding_mode,
                                            align_corners);
      y = grid_sampler_compute_source_index(y, x_dims[2], padding_mode,
                                            align_corners);
      auto x_ptr_NC = y_iter[0] * x_stride[0];
      auto y_ptr_NCHW = y_iter[0] * y_stride[0] + y_iter[2] * y_stride[2] +
                        y_iter[3] * y_stride[3];

      if (interpolation_mode == "bilinear") {
        bilinear_half(x, y, x_data_addr, y_data_addr, y_c, x_dims, y_stride,
                      x_stride, x_ptr_NC, y_ptr_NCHW);
      } else if (interpolation_mode == "nearest") {
        nearest_half(x, y, x_data_addr, y_data_addr, y_c, x_dims, y_stride,
                     x_stride, x_ptr_NC, y_ptr_NCHW);
      }
    }
  };
  if (max_core_num == 0) {
    max_core_num = 1;
  }
  KERNEL_HANDLE_ERROR(
      CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num,
                                  shard_compute),
      "GridSampler2D Compute failed.");
  return KERNEL_STATUS_OK;
}

uint32_t GridSampler2DCpuKernel::Call1Half(
    Eigen::half *x_data_addr, Eigen::half *y_data_addr,
    Eigen::half *grid_data_addr, std::vector<int64_t> x_dims,
    std::vector<int64_t> y_dims, int64_t *y_stride, int64_t *x_stride,
    int64_t *grid_stride, std::string interpolation_mode,
    std::string padding_mode, bool align_corners) {
  int64_t y_iter[4] = {0};
  const int64_t y_c = y_dims[1];
  auto NextIndex = [&]() {
    int64_t carry = 1;
    for (int32_t id = 3; id > -1; id--) {
      if (id == 1) {
        continue;
      }
      int64_t iter_val = y_iter[id] + carry;
      if (iter_val == y_dims[id]) {
        y_iter[id] = 0;
      } else {
        y_iter[id] = iter_val;
        carry = 0;
        break;
      }
    }
    return (carry == 0);
  };
  do {
    int64_t grid_offset = y_iter[0] * grid_stride[0] +
                          y_iter[2] * grid_stride[1] +
                          y_iter[3] * grid_stride[2];
    float x = static_cast<float>(grid_data_addr[grid_offset]);
    float y = static_cast<float>(grid_data_addr[grid_offset + grid_stride[3]]);
    x = grid_sampler_compute_source_index(x, x_dims[INPUT_NUM3], padding_mode,
                                          align_corners);
    y = grid_sampler_compute_source_index(y, x_dims[INPUT_NUM2], padding_mode,
                                          align_corners);
    auto x_ptr_NC = y_iter[0] * x_stride[0];
    auto y_ptr_NCHW = y_iter[0] * y_stride[0] + y_iter[2] * y_stride[2] +
                      y_iter[3] * y_stride[3];
    if (interpolation_mode == "bilinear") {
      bilinear_half(x, y, x_data_addr, y_data_addr, y_c, x_dims, y_stride,
                    x_stride, x_ptr_NC, y_ptr_NCHW);
    } else if (interpolation_mode == "nearest") {
      nearest_half(x, y, x_data_addr, y_data_addr, y_c, x_dims, y_stride,
                   x_stride, x_ptr_NC, y_ptr_NCHW);
    } else {
      KERNEL_LOG_ERROR(
          "The value of interpolation_mode must be bilinear or nearest, but "
          "get [%s].",
          interpolation_mode.c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  } while (NextIndex());
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t GridSampler2DCpuKernel::Call2(
    CpuKernelContext &ctx, T *x_data_addr, T *y_data_addr, T *grid_data_addr,
    std::vector<int64_t> x_dims, std::vector<int64_t> y_dims, int64_t *y_stride,
    int64_t *x_stride, int64_t *grid_stride, std::string interpolation_mode,
    std::string padding_mode, bool align_corners) {
  uint32_t data_num = y_dims[0] * y_dims[2] * y_dims[3];
  uint32_t min_core_num = 1;
  uint32_t max_core_num =
      std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
  if (max_core_num > data_num) {
    max_core_num = data_num;
  }
  auto shard_compute = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      int64_t y_iter[4] = {0};
      y_iter[1] = static_cast<int64_t>(i);
      int64_t count = 3;
      while (y_iter[1] > 0) {
        if (count == 1) {
          count--;
        }
        y_iter[count] = y_iter[1] % y_dims[count];
        y_iter[1] /= y_dims[count--];
      }
      const int64_t y_c = y_dims[1];
      int64_t grid_offset = y_iter[0] * grid_stride[0] +
                            y_iter[2] * grid_stride[1] +
                            y_iter[3] * grid_stride[2];
      T x = grid_data_addr[grid_offset];
      T y = grid_data_addr[grid_offset + grid_stride[3]];
      x = grid_sampler_compute_source_index(x, x_dims[3], padding_mode,
                                            align_corners);
      y = grid_sampler_compute_source_index(y, x_dims[2], padding_mode,
                                            align_corners);
      auto x_ptr_NC = y_iter[0] * x_stride[0];
      auto y_ptr_NCHW = y_iter[0] * y_stride[0] + y_iter[2] * y_stride[2] +
                        y_iter[3] * y_stride[3];
      if (interpolation_mode == "bilinear") {
        bilinear(x, y, x_data_addr, y_data_addr, y_c, x_dims, y_stride,
                 x_stride, x_ptr_NC, y_ptr_NCHW);
      } else if (interpolation_mode == "nearest") {
        nearest(x, y, x_data_addr, y_data_addr, y_c, x_dims, y_stride, x_stride,
                x_ptr_NC, y_ptr_NCHW);
      }
    }
  };
  if (max_core_num == 0) max_core_num = 1;
  KERNEL_HANDLE_ERROR(
      CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num,
                                  shard_compute),
      "GridSampler2D Compute failed.");
  return KERNEL_STATUS_OK;
}

template <typename T>
void GridSampler2DCpuKernel::Call1(
    T *x_data_addr, T *y_data_addr, T *grid_data_addr,
    std::vector<int64_t> x_dims, std::vector<int64_t> y_dims, int64_t *y_stride,
    int64_t *x_stride, int64_t *grid_stride, std::string interpolation_mode,
    std::string padding_mode, bool align_corners) {
  int64_t y_iter[4] = {0};
  const int64_t y_c = y_dims[1];
  auto NextIndex = [&]() {
    int64_t carry = 1;
    for (int32_t id = 3; id > -1; id--) {
      if (id == 1) {
        continue;
      }
      int64_t iter_val = y_iter[id] + carry;
      if (iter_val == y_dims[id]) {
        y_iter[id] = 0;
      } else {
        y_iter[id] = iter_val;
        carry = 0;
        break;
      }
    }
    return (carry == 0);
  };

  do {
    int64_t grid_offset = y_iter[0] * grid_stride[0] +
                          y_iter[2] * grid_stride[1] +
                          y_iter[3] * grid_stride[2];
    T x = grid_data_addr[grid_offset];
    T y = grid_data_addr[grid_offset + grid_stride[INPUT_NUM3]];
    x = grid_sampler_compute_source_index(x, x_dims[INPUT_NUM3], padding_mode,
                                          align_corners);
    y = grid_sampler_compute_source_index(y, x_dims[INPUT_NUM2], padding_mode,
                                          align_corners);
    auto x_ptr_NC = y_iter[0] * x_stride[0];
    auto y_ptr_NCHW = y_iter[0] * y_stride[0] +
                      y_iter[INPUT_NUM2] * y_stride[INPUT_NUM2] +
                      y_iter[INPUT_NUM3] * y_stride[INPUT_NUM3];
    if (interpolation_mode == "bilinear") {
      bilinear(x, y, x_data_addr, y_data_addr, y_c, x_dims, y_stride, x_stride,
               x_ptr_NC, y_ptr_NCHW);
    } else if (interpolation_mode == "nearest") {
      nearest(x, y, x_data_addr, y_data_addr, y_c, x_dims, y_stride, x_stride,
              x_ptr_NC, y_ptr_NCHW);
    }
  } while (NextIndex());
}
void GridSampler2DCpuKernel::nearest_half(
    float x, float y, Eigen::half *x_data_addr, Eigen::half *y_data_addr,
    int64_t y_c, std::vector<int64_t> x_dims, int64_t *y_stride,
    int64_t *x_stride, int64_t x_ptr_NC, int64_t y_ptr_NCHW) {
  int64_t x_nearest = static_cast<int64_t>(std::round(x));
  int64_t y_nearest = static_cast<int64_t>(std::round(y));
  for (int64_t c = 0; c < y_c;
       ++c, x_ptr_NC += x_stride[1], y_ptr_NCHW += y_stride[1]) {
    if (within_bounds_2d(y_nearest, x_nearest, x_dims[INPUT_NUM2],
                         x_dims[INPUT_NUM3])) {
      auto x_index =
          x_ptr_NC + y_nearest * x_stride[2] + x_nearest * x_stride[3];
      y_data_addr[y_ptr_NCHW] = x_data_addr[x_index];
    } else {
      y_data_addr[y_ptr_NCHW] = static_cast<Eigen::half>(0);
    }
  }
}

void GridSampler2DCpuKernel::bilinear_half(
    float x, float y, Eigen::half *x_data_addr, Eigen::half *y_data_addr,
    int64_t y_c, std::vector<int64_t> x_dims, int64_t *y_stride,
    int64_t *x_stride, int64_t x_ptr_NC, int64_t y_ptr_NCHW) {
  int64_t x_tnw = static_cast<int64_t>(std::floor(x));
  int64_t y_tnw = static_cast<int64_t>(std::floor(y));

  int64_t x_tne = x_tnw + 1;
  int64_t y_tne = y_tnw;

  int64_t x_tsw = x_tnw;
  int64_t y_tsw = y_tnw + 1;

  int64_t x_tse = x_tnw + 1;
  int64_t y_tse = y_tnw + 1;

  Eigen::half tnw = static_cast<Eigen::half>((x_tse - x) * (y_tse - y));
  Eigen::half tne = static_cast<Eigen::half>((x - x_tsw) * (y_tsw - y));
  Eigen::half tsw = static_cast<Eigen::half>((x_tne - x) * (y - y_tne));
  Eigen::half tse = static_cast<Eigen::half>((x - x_tnw) * (y - y_tnw));

  for (int64_t c = 0; c < y_c;
       ++c, x_ptr_NC += x_stride[1], y_ptr_NCHW += y_stride[1]) {
    y_data_addr[y_ptr_NCHW] = static_cast<Eigen::half>(0);
    if (within_bounds_2d(y_tnw, x_tnw, x_dims[INPUT_NUM2],
                         x_dims[INPUT_NUM3])) {
      auto x_index = x_ptr_NC + y_tnw * x_stride[2] + x_tnw * x_stride[3];
      y_data_addr[y_ptr_NCHW] += x_data_addr[x_index] * tnw;
    }
    if (within_bounds_2d(y_tne, x_tne, x_dims[INPUT_NUM2],
                         x_dims[INPUT_NUM3])) {
      auto x_index = x_ptr_NC + y_tne * x_stride[2] + x_tne * x_stride[3];
      y_data_addr[y_ptr_NCHW] += x_data_addr[x_index] * tne;
    }
    if (within_bounds_2d(y_tsw, x_tsw, x_dims[INPUT_NUM2],
                         x_dims[INPUT_NUM3])) {
      auto x_index = x_ptr_NC + y_tsw * x_stride[2] + x_tsw * x_stride[3];
      y_data_addr[y_ptr_NCHW] += x_data_addr[x_index] * tsw;
    }
    if (within_bounds_2d(y_tse, x_tse, x_dims[INPUT_NUM2],
                         x_dims[INPUT_NUM3])) {
      auto x_index = x_ptr_NC + y_tse * x_stride[2] + x_tse * x_stride[3];
      y_data_addr[y_ptr_NCHW] += x_data_addr[x_index] * tse;
    }
  }
}

template <typename T>
void GridSampler2DCpuKernel::nearest(T x, T y, T *x_data_addr, T *y_data_addr,
                                     int64_t y_c, std::vector<int64_t> x_dims,
                                     int64_t *y_stride, int64_t *x_stride,
                                     int64_t x_ptr_NC, int64_t y_ptr_NCHW) {
  int64_t x_nearest = static_cast<int64_t>(std::round(x));
  int64_t y_nearest = static_cast<int64_t>(std::round(y));
  for (int64_t c = 0; c < y_c;
       c++, x_ptr_NC += x_stride[1], y_ptr_NCHW += y_stride[1]) {
    if (within_bounds_2d(y_nearest, x_nearest, x_dims[INPUT_NUM2],
                         x_dims[INPUT_NUM3])) {
      auto x_index =
          x_ptr_NC + y_nearest * x_stride[2] + x_nearest * x_stride[3];
      y_data_addr[y_ptr_NCHW] = x_data_addr[x_index];
    } else {
      y_data_addr[y_ptr_NCHW] = static_cast<T>(0);
    }
  }
}

template <typename T>
void GridSampler2DCpuKernel::bilinear(T x, T y, T *x_data_addr, T *y_data_addr,
                                      int64_t y_c, std::vector<int64_t> x_dims,
                                      int64_t *y_stride, int64_t *x_stride,
                                      int64_t x_ptr_NC, int64_t y_ptr_NCHW) {
  int64_t x_tnw = static_cast<int64_t>(std::floor(x));
  int64_t y_tnw = static_cast<int64_t>(std::floor(y));

  int64_t x_tne = x_tnw + 1;
  int64_t y_tne = y_tnw;

  int64_t x_tsw = x_tnw;
  int64_t y_tsw = y_tnw + 1;

  int64_t x_tse = x_tnw + 1;
  int64_t y_tse = y_tnw + 1;

  T tnw = (x_tse - x) * (y_tse - y);
  T tne = (x - x_tsw) * (y_tsw - y);
  T tsw = (x_tne - x) * (y - y_tne);
  T tse = (x - x_tnw) * (y - y_tnw);

  for (int64_t c = 0; c < y_c;
       ++c, x_ptr_NC += x_stride[1], y_ptr_NCHW += y_stride[1]) {
    y_data_addr[y_ptr_NCHW] = static_cast<T>(0);
    if (within_bounds_2d(y_tnw, x_tnw, x_dims[INPUT_NUM2],
                         x_dims[INPUT_NUM3])) {
      auto x_index = x_ptr_NC + y_tnw * x_stride[2] + x_tnw * x_stride[3];
      y_data_addr[y_ptr_NCHW] += x_data_addr[x_index] * tnw;
    }
    if (within_bounds_2d(y_tne, x_tne, x_dims[INPUT_NUM2],
                         x_dims[INPUT_NUM3])) {
      auto x_index = x_ptr_NC + y_tne * x_stride[2] + x_tne * x_stride[3];
      y_data_addr[y_ptr_NCHW] += x_data_addr[x_index] * tne;
    }
    if (within_bounds_2d(y_tsw, x_tsw, x_dims[INPUT_NUM2],
                         x_dims[INPUT_NUM3])) {
      auto x_index = x_ptr_NC + y_tsw * x_stride[2] + x_tsw * x_stride[3];
      y_data_addr[y_ptr_NCHW] += x_data_addr[x_index] * tsw;
    }
    if (within_bounds_2d(y_tse, x_tse, x_dims[INPUT_NUM2],
                         x_dims[INPUT_NUM3])) {
      auto x_index = x_ptr_NC + y_tse * x_stride[2] + x_tse * x_stride[3];
      y_data_addr[y_ptr_NCHW] += x_data_addr[x_index] * tse;
    }
  }
}

template <typename T>
T GridSampler2DCpuKernel::grid_sampler_compute_source_index(
    T coord, int64_t size, std::string padding_mode, bool align_corners) {
  if (align_corners) {
    coord = ((coord + 1.f) / INPUT_NUM2) * (size - 1);
  } else {
    coord = ((coord + 1.f) * size - 1) / INPUT_NUM2;
  }
  if (padding_mode == "border") {
    coord =
        std::min(static_cast<T>(size - 1), std::max(coord, static_cast<T>(0)));
  } else if (padding_mode == "reflection") {
    if (align_corners) {
      coord = reflect_coordinates(coord, 0, INPUT_NUM2 * (size - 1));
    } else {
      coord = reflect_coordinates(coord, -1, INPUT_NUM2 * size - 1);
    }
    coord =
        std::min(static_cast<T>(size - 1), std::max(coord, static_cast<T>(0)));
  }
  return coord;
}

template <typename T>
T GridSampler2DCpuKernel::reflect_coordinates(T coord, int64_t twice_low,
                                              int64_t twice_high) {
  if (twice_low == twice_high) {
    return static_cast<T>(0);
  }
  T min = static_cast<T>(twice_low) / 2;
  T span = static_cast<T>(twice_high - twice_low) / 2;
  coord = std::fabs(coord - min);
  T extra = std::fmod(coord, span);
  int64_t flips = static_cast<int64_t>(std::floor(coord / span));
  if (flips % INPUT_NUM2 == 0) {
    return extra + min;
  } else {
    return span - extra + min;
  }
}

bool GridSampler2DCpuKernel::within_bounds_2d(int64_t h, int64_t w, int64_t H,
                                              int64_t W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

REGISTER_CPU_KERNEL(kGridSampler2D, GridSampler2DCpuKernel);
}  // namespace aicpu
