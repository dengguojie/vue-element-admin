/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "grid_sampler_3d.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "cmath"

namespace {
const char *kGridSampler3D = "GridSampler3D";
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 1;
const uint64_t kOutputSize = 64 * 1024;
const uint32_t kTwo = 2;
const uint32_t kThree = 3;
const uint32_t kFour = 4;
const uint32_t kFive = 5;

#define GRIDSAMPLER3D_COMPUTE_CASE(DTYPE, TYPE, CTX)                  \
  case(DTYPE):{                                                       \
    uint32_t result = GridSampler3DCompute<TYPE>(CTX);                \
    if (result != KERNEL_STATUS_OK) {                                 \
      KERNEL_LOG_ERROR("GridSampler3D kernel compute failed.");       \
      return result;                                                  \
    }                                                                 \
    break;                                                            \
  }

#define GRIDSAMPLER3D_COMPUTE_CASE_HALF(DTYPE, CTX)                   \
  case(DTYPE):{                                                       \
    uint32_t result = GridSampler3DComputeHalf(CTX);                  \
    if (result != KERNEL_STATUS_OK) {                                 \
      KERNEL_LOG_ERROR("GridSampler3D kernel compute failed.");       \
      return result;                                                  \
    }                                                                 \
    break;                                                            \
  }
}

namespace aicpu{
uint32_t GridSampler3DCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "GridSampler3D check input and output number failed.");
  Tensor *x_data = ctx.Input(0);
  Tensor *grid_data = ctx.Input(1);
  DataType x_type = x_data->GetDataType();
  DataType grid_type = grid_data->GetDataType();
  if (x_type != grid_type) {
    KERNEL_LOG_ERROR("Input[0] and input[1] must have same dtype, but get [%s] and [%s].",
                     DTypeStr(x_type).c_str(), DTypeStr(grid_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  switch(x_type) {
    GRIDSAMPLER3D_COMPUTE_CASE_HALF(DT_FLOAT16, ctx)
    GRIDSAMPLER3D_COMPUTE_CASE(DT_FLOAT, float, ctx)
    GRIDSAMPLER3D_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("Input data type[%s] not supported.",DTypeStr(x_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t GridSampler3DCpuKernel::GridSampler3DCompute(CpuKernelContext &ctx) {
  Tensor *x_data = ctx.Input(0);
  auto x_data_addr = reinterpret_cast<T *>(x_data->GetData());
  auto x_shape = x_data->GetTensorShape();
  auto x_dims = x_shape->GetDimSizes();
  Tensor *grid_data = ctx.Input(1);
  auto grid_data_addr = reinterpret_cast<T *>(grid_data->GetData());
  auto grid_shape = grid_data->GetTensorShape();
  auto grid_dims = grid_shape->GetDimSizes();
  if (grid_dims[0] != x_dims[0]) {
    KERNEL_LOG_ERROR("The first dimension of the shape of x must be equal to that of grid, but got [%d] and [%d].",
                     x_dims[0], grid_dims[0]);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *y_data = ctx.Output(0);
  auto y_data_addr = reinterpret_cast<T *>(y_data->GetData());
  auto y_shape = y_data->GetTensorShape();
  auto y_dims = y_shape->GetDimSizes();
  uint64_t y_data_size = y_data->GetDataSize();
  std::vector<int64_t> x_stride = stride_comput(x_dims);
  std::vector<int64_t> grid_stride = stride_comput(grid_dims);
  std::vector<int64_t> y_stride = stride_comput(y_dims);
  std::string interpolation_mode, padding_mode;
  bool align_corners;
  if (!check_attr(interpolation_mode, padding_mode, align_corners, ctx)) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (y_data_size < kOutputSize) {
    std::vector<int64_t> y_iter(kFive, 0);
    const int64_t y_c = y_dims[1];
    do {
      int64_t grid_offset = y_iter[0] * grid_stride[0] + y_iter[kTwo] * grid_stride[1] +
                            y_iter[kThree] * grid_stride[kTwo] + y_iter[kFour] * grid_stride[kThree];
      T x = grid_data_addr[grid_offset];
      T y = grid_data_addr[grid_offset + grid_stride[kFour]];
      T z = grid_data_addr[grid_offset + kTwo * grid_stride[kFour]];
      x = grid_sampler_compute_source_index(x, x_dims[kFour], padding_mode, align_corners);
      y = grid_sampler_compute_source_index(y, x_dims[kThree], padding_mode, align_corners);
      z = grid_sampler_compute_source_index(z, x_dims[kTwo], padding_mode, align_corners);
      auto x_ptr_NC = y_iter[0] * x_stride[0];
      auto y_ptr_NCDHW = y_iter[0] * y_stride[0] + y_iter[kTwo] * y_stride[kTwo] + y_iter[kThree] * y_stride[kThree] +
                         y_iter[kFour] * y_stride[kFour];
      std::vector<T *> addr = {x_data_addr, y_data_addr};
      std::vector<T> location = {x, y, z};
      if (interpolation_mode == "bilinear") {
        bilinear_compute(addr, location, y_c, x_dims, x_ptr_NC, x_stride, y_ptr_NCDHW, y_stride);
      } else if (interpolation_mode == "nearest") {
        nearest_compute(addr, location, y_c, x_dims, x_ptr_NC, x_stride, y_ptr_NCDHW, y_stride);
      }
    } while (NextIndex(y_dims, y_iter));
  } else {
    size_t data_num = y_dims[0] * y_dims[kTwo] * y_dims[kThree] * y_dims[kFour];
    uint32_t min_core_num = 1;
    size_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kTwo);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto shard_compute = [&](size_t start, size_t end){
      for (size_t i = start; i < end; i++){
        std::vector<int64_t> y_iter(kFive, 0);
        y_iter[1] = static_cast<int64_t>(i);
        int64_t count = kFour;
        while (y_iter[1] > 0 && count != 1) {
          y_iter[count] = y_iter[1] % y_dims[count];
          y_iter[1] /= y_dims[count--];
        }
        y_iter[0] = y_iter[1] % y_dims[0];
        y_iter[1] /= y_dims[0];
        const int64_t y_c = y_dims[1];
        int64_t grid_offset = y_iter[0] * grid_stride[0] + y_iter[kTwo] * grid_stride[1] +
                              y_iter[kThree] * grid_stride[kTwo] + y_iter[kFour] * grid_stride[kThree];
        T x = grid_data_addr[grid_offset];
        T y = grid_data_addr[grid_offset + grid_stride[kFour]];
        T z = grid_data_addr[grid_offset + kTwo * grid_stride[kFour]];
        x = grid_sampler_compute_source_index(x, x_dims[kFour], padding_mode, align_corners);
        y = grid_sampler_compute_source_index(y, x_dims[kThree], padding_mode, align_corners);
        z = grid_sampler_compute_source_index(z, x_dims[kTwo], padding_mode, align_corners);
        auto x_ptr_NC = y_iter[0] * x_stride[0];
        auto y_ptr_NCDHW = y_iter[0] * y_stride[0] + y_iter[kTwo] * y_stride[kTwo] +
                           y_iter[kThree] * y_stride[kThree] + y_iter[kFour] * y_stride[kFour];
        std::vector<T *> addr = {x_data_addr, y_data_addr};
        std::vector<T> location = {x, y, z};
        if (interpolation_mode == "bilinear") {
          bilinear_compute(addr, location, y_c, x_dims, x_ptr_NC, x_stride, y_ptr_NCDHW, y_stride);
        } else if (interpolation_mode == "nearest") {
          nearest_compute(addr, location, y_c, x_dims, x_ptr_NC, x_stride, y_ptr_NCDHW, y_stride);
        }
      }
    };
    if (max_core_num == 0) {KERNEL_LOG_ERROR("max_core_num could not be 0.");}
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_compute),
                        "GridSampler3D Compute failed.")
  }
  return KERNEL_STATUS_OK;
}

bool GridSampler3DCpuKernel::check_attr(std::string &interpolation_mode, std::string &padding_mode,
                                        bool &align_corners, CpuKernelContext &ctx) {
  AttrValue *attr1 = ctx.GetAttr("interpolation_mode");
  if (attr1 == nullptr) {
    interpolation_mode = "bilinear";
  } else {
    interpolation_mode = attr1->GetString();
  }
  if (interpolation_mode != "bilinear" && interpolation_mode != "nearest"){
    KERNEL_LOG_ERROR("The value of interpolation_mode must be bilinear or nearest, but get [%s].",
                     interpolation_mode.c_str());
    return false;
  }
  AttrValue *attr2 = ctx.GetAttr("padding_mode");
  if (attr2 == nullptr) {
    padding_mode = "zeros";
  } else {
    padding_mode = attr2->GetString();
  }
  if (padding_mode != "zeros" && padding_mode != "border" && padding_mode != "reflection"){
    KERNEL_LOG_ERROR("The value of padding_mode must be one of border, reflection and zeros, but get [%s].",
                     padding_mode.c_str());
    return false;
  }
  AttrValue *attr3 = ctx.GetAttr("align_corners");
  if (attr3 == nullptr) {
    align_corners = false;
  } else {
    align_corners = attr3->GetBool();
  }
  return true;
}

std::vector<int64_t> GridSampler3DCpuKernel::stride_comput(const std::vector<int64_t> &shape) {
  std::vector<int64_t> stride;
  size_t size = shape.size();
  int64_t stride_tmp = 1;
  for (int32_t i = size - 1; i > -1; --i) {
    stride.insert(stride.begin(), stride_tmp);
    stride_tmp *= shape[i];
  }
  return stride;
}

bool GridSampler3DCpuKernel::NextIndex(const std::vector<int64_t> &shape, std::vector<int64_t> &iter) {
  int64_t carry = 1;
  for (int32_t id = shape.size() - 1; id > -1; --id) {
    if (id == 1) {
      continue;
    }
    int64_t iter_val = iter[id] + carry;
    if (iter_val == shape[id]) {
      iter[id] = 0;
    } else {
      iter[id] = iter_val;
      carry = 0;
      break;
    }
  }
  return (carry == 0);
}

uint32_t GridSampler3DCpuKernel::GridSampler3DComputeHalf(CpuKernelContext &ctx) {
  Tensor *x_data = ctx.Input(0);
  auto x_data_addr = reinterpret_cast<Eigen::half *>(x_data->GetData());
  auto x_shape = x_data->GetTensorShape();
  auto x_dims = x_shape->GetDimSizes();
  Tensor *grid_data = ctx.Input(1);
  auto grid_data_addr = reinterpret_cast<Eigen::half *>(grid_data->GetData());
  auto grid_shape = grid_data->GetTensorShape();
  auto grid_dims = grid_shape->GetDimSizes();
  if (grid_dims[0] != x_dims[0]) {
    KERNEL_LOG_ERROR("The first dimension of the shape of x must be equal to that of grid, but got [%d] and [%d].",
                     x_dims[0], grid_dims[0]);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *y_data = ctx.Output(0);
  auto y_data_addr = reinterpret_cast<Eigen::half *>(y_data->GetData());
  auto y_shape = y_data->GetTensorShape();
  auto y_dims = y_shape->GetDimSizes();
  uint64_t y_data_size = y_data->GetDataSize();
  std::vector<int64_t> x_stride = stride_comput(x_dims);
  std::vector<int64_t> grid_stride = stride_comput(grid_dims);
  std::vector<int64_t> y_stride = stride_comput(y_dims);
  std::string interpolation_mode, padding_mode;
  bool align_corners;
  if (!check_attr(interpolation_mode, padding_mode, align_corners, ctx)) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (y_data_size < kOutputSize) {
    std::vector<int64_t> y_iter(kFive, 0);
    const int64_t y_c = y_dims[1];
    do {
      int64_t grid_offset = y_iter[0] * grid_stride[0] + y_iter[kTwo] * grid_stride[1] +
                            y_iter[kThree] * grid_stride[kTwo] + y_iter[kFour] * grid_stride[kThree];
      float x = static_cast<float>(grid_data_addr[grid_offset]);
      float y = static_cast<float>(grid_data_addr[grid_offset + grid_stride[kFour]]);
      float z = static_cast<float>(grid_data_addr[grid_offset + kTwo * grid_stride[kFour]]);
      x = grid_sampler_compute_source_index(x, x_dims[kFour], padding_mode, align_corners);
      y = grid_sampler_compute_source_index(y, x_dims[kThree], padding_mode, align_corners);
      z = grid_sampler_compute_source_index(z, x_dims[kTwo], padding_mode, align_corners);
      auto x_ptr_NC = y_iter[0] * x_stride[0];
      auto y_ptr_NCDHW = y_iter[0] * y_stride[0] + y_iter[kTwo] * y_stride[kTwo] +
                         y_iter[kThree] * y_stride[kThree] + y_iter[kFour] * y_stride[kFour];
      std::vector<Eigen::half *> addr = {x_data_addr, y_data_addr};
      std::vector<float> location = {x, y, z};
      if (interpolation_mode == "bilinear") {
        bilinear_compute_half(addr, location, y_c, x_dims, x_ptr_NC, x_stride, y_ptr_NCDHW, y_stride);
      } else if (interpolation_mode == "nearest") {
        nearest_compute_half(addr, location, y_c, x_dims, x_ptr_NC, x_stride, y_ptr_NCDHW, y_stride);
      }
    } while (NextIndex(y_dims, y_iter));
  } else {
    size_t data_num = y_dims[0] * y_dims[kTwo] * y_dims[kThree] * y_dims[kFour];
    uint32_t min_core_num = 1;
    size_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kTwo);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto shard_compute = [&](size_t start, size_t end){
      for (size_t i = start; i < end; i++){
        int64_t y_iter[kFive] = {0};
        y_iter[1] = static_cast<int64_t>(i);
        int64_t count = kFour;
        while (y_iter[1] > 0 && count != 1) {
          y_iter[count] = y_iter[1] % y_dims[count];
          y_iter[1] /= y_dims[count--];
        }
        y_iter[0] = y_iter[1] % y_dims[0];
        y_iter[1] /= y_dims[0];
        const int64_t y_c = y_dims[1];
        int64_t grid_offset = y_iter[0] * grid_stride[0] + y_iter[kTwo] * grid_stride[1] +
                              y_iter[kThree] * grid_stride[kTwo] + y_iter[kFour] * grid_stride[kThree];
        float x = static_cast<float>(grid_data_addr[grid_offset]);
        float y = static_cast<float>(grid_data_addr[grid_offset + grid_stride[kFour]]);
        float z = static_cast<float>(grid_data_addr[grid_offset + kTwo * grid_stride[kFour]]);
        x = grid_sampler_compute_source_index(x, x_dims[kFour], padding_mode, align_corners);
        y = grid_sampler_compute_source_index(y, x_dims[kThree], padding_mode, align_corners);
        z = grid_sampler_compute_source_index(z, x_dims[kTwo], padding_mode, align_corners);
        auto x_ptr_NC = y_iter[0] * x_stride[0];
        auto y_ptr_NCDHW = y_iter[0] * y_stride[0] + y_iter[kTwo] * y_stride[kTwo] +
                           y_iter[kThree] * y_stride[kThree] + y_iter[kFour] * y_stride[kFour];
        std::vector<Eigen::half *> addr = {x_data_addr, y_data_addr};
        std::vector<float> location = {x, y, z};
        if (interpolation_mode == "bilinear") {
          bilinear_compute_half(addr, location, y_c, x_dims, x_ptr_NC, x_stride, y_ptr_NCDHW, y_stride);
        } else if (interpolation_mode == "nearest") {
          nearest_compute_half(addr, location, y_c, x_dims, x_ptr_NC, x_stride, y_ptr_NCDHW, y_stride);
        }
      }
    };
    if (max_core_num == 0) {KERNEL_LOG_ERROR("max_core_num could not be 0.");}
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_compute),
                        "GridSampler3D Compute failed.")
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
void GridSampler3DCpuKernel::bilinear_compute(std::vector<T *> addr, std::vector<T> location, const int64_t y_c,
                                              std::vector<int64_t> x_dims, int64_t x_ptr_NC,
                                              std::vector<int64_t> x_stride, int64_t y_ptr_NCDHW,
                                              std::vector<int64_t> y_stride){
  T *x_data_addr = addr[0];
  T *y_data_addr = addr[1];
  T x = location[0];
  T y = location[1];
  T z = location[kTwo];
  int64_t x_tnw = static_cast<int64_t>(std::floor(x));
  int64_t y_tnw = static_cast<int64_t>(std::floor(y));
  int64_t z_tnw = static_cast<int64_t>(std::floor(z));
  int64_t x_tne = x_tnw + 1;
  int64_t y_tne = y_tnw;
  int64_t z_tne = z_tnw;
  int64_t x_tsw = x_tnw;
  int64_t y_tsw = y_tnw + 1;
  int64_t z_tsw = z_tnw;
  int64_t x_tse = x_tnw + 1;
  int64_t y_tse = y_tnw + 1;
  int64_t z_tse = z_tnw;
  int64_t x_bnw = x_tnw;
  int64_t y_bnw = y_tnw;
  int64_t z_bnw = z_tnw + 1;
  int64_t x_bne = x_tnw + 1;
  int64_t y_bne = y_tnw;
  int64_t z_bne = z_tnw + 1;
  int64_t x_bsw = x_tnw;
  int64_t y_bsw = y_tnw + 1;
  int64_t z_bsw = z_tnw + 1;
  int64_t x_bse = x_tnw + 1;
  int64_t y_bse = y_tnw + 1;
  int64_t z_bse = z_tnw + 1;
  T tnw = (x_bse - x) * (y_bse - y) * (z_bse - z);
  T tne = (x - x_bsw) * (y_bsw - y) * (z_bsw - z);
  T tsw = (x_bne - x) * (y - y_bne) * (z_bne - z);
  T tse = (x - x_bnw) * (y - y_bnw) * (z_bnw - z);
  T bnw = (x_tse - x) * (y_tse - y) * (z - z_tse);
  T bne = (x - x_tsw) * (y_tsw - y) * (z - z_tsw);
  T bsw = (x_tne - x) * (y - y_tne) * (z - z_tne);
  T bse = (x - x_tnw) * (y - y_tnw) * (z - z_tnw);
  for (int64_t c = 0; c < y_c; c++, x_ptr_NC += x_stride[1], y_ptr_NCDHW += y_stride[1]) {
    y_data_addr[y_ptr_NCDHW] = static_cast<T>(0);
    if (within_bounds_3d(z_tnw, y_tnw, x_tnw, x_dims)) {
      auto x_index = x_ptr_NC + z_tnw * x_stride[kTwo] + y_tnw * x_stride[kThree] + x_tnw * x_stride[kFour];
      y_data_addr[y_ptr_NCDHW] += x_data_addr[x_index] * tnw;
    }
    if (within_bounds_3d(z_tne, y_tne, x_tne, x_dims)) {
      auto x_index = x_ptr_NC + z_tne * x_stride[kTwo] + y_tne * x_stride[kThree] + x_tne * x_stride[kFour];
      y_data_addr[y_ptr_NCDHW] += x_data_addr[x_index] * tne;
    }
    if (within_bounds_3d(z_tsw, y_tsw, x_tsw, x_dims)) {
      auto x_index = x_ptr_NC + z_tsw * x_stride[kTwo] + y_tsw * x_stride[kThree] + x_tsw * x_stride[kFour];
      y_data_addr[y_ptr_NCDHW] += x_data_addr[x_index] * tsw;
    }
    if (within_bounds_3d(z_tse, y_tse, x_tse, x_dims)) {
      auto x_index = x_ptr_NC + z_tse * x_stride[kTwo] + y_tse * x_stride[kThree] + x_tse * x_stride[kFour];
      y_data_addr[y_ptr_NCDHW] += x_data_addr[x_index] * tse;
    }
    if (within_bounds_3d(z_bnw, y_bnw, x_bnw, x_dims)) {
      auto x_index = x_ptr_NC + z_bnw * x_stride[kTwo] + y_bnw * x_stride[kThree] + x_bnw * x_stride[kFour];
      y_data_addr[y_ptr_NCDHW] += x_data_addr[x_index] * bnw;
    }
    if (within_bounds_3d(z_bne, y_bne, x_bne, x_dims)) {
      auto x_index = x_ptr_NC + z_bne * x_stride[kTwo] + y_bne * x_stride[kThree] + x_bne * x_stride[kFour];
      y_data_addr[y_ptr_NCDHW] += x_data_addr[x_index] * bne;
    }
    if (within_bounds_3d(z_bsw, y_bsw, x_bsw, x_dims)) {
      auto x_index = x_ptr_NC + z_bsw * x_stride[kTwo] + y_bsw * x_stride[kThree] + x_bsw * x_stride[kFour];
      y_data_addr[y_ptr_NCDHW] += x_data_addr[x_index] * bsw;
    }
    if (within_bounds_3d(z_bse, y_bse, x_bse, x_dims)) {
      auto x_index = x_ptr_NC + z_bse * x_stride[kTwo] + y_bse * x_stride[kThree] + x_bse * x_stride[kFour];
      y_data_addr[y_ptr_NCDHW] += x_data_addr[x_index] * bse;
    }
  }
}

void GridSampler3DCpuKernel::bilinear_compute_half(std::vector<Eigen::half *> addr, std::vector<float> location,
                                                   const int64_t y_c, std::vector<int64_t> x_dims, int64_t x_ptr_NC,
                                                   std::vector<int64_t> x_stride, int64_t y_ptr_NCDHW,
                                                   std::vector<int64_t> y_stride){
  Eigen::half *x_data_addr = addr[0];
  Eigen::half *y_data_addr = addr[1];
  float x = location[0];
  float y = location[1];
  float z = location[kTwo];
  int64_t x_tnw = static_cast<int64_t>(std::floor(x));
  int64_t y_tnw = static_cast<int64_t>(std::floor(y));
  int64_t z_tnw = static_cast<int64_t>(std::floor(z));
  int64_t x_tne = x_tnw + 1;
  int64_t y_tne = y_tnw;
  int64_t z_tne = z_tnw;
  int64_t x_tsw = x_tnw;
  int64_t y_tsw = y_tnw + 1;
  int64_t z_tsw = z_tnw;
  int64_t x_tse = x_tnw + 1;
  int64_t y_tse = y_tnw + 1;
  int64_t z_tse = z_tnw;
  int64_t x_bnw = x_tnw;
  int64_t y_bnw = y_tnw;
  int64_t z_bnw = z_tnw + 1;
  int64_t x_bne = x_tnw + 1;
  int64_t y_bne = y_tnw;
  int64_t z_bne = z_tnw + 1;
  int64_t x_bsw = x_tnw;
  int64_t y_bsw = y_tnw + 1;
  int64_t z_bsw = z_tnw + 1;
  int64_t x_bse = x_tnw + 1;
  int64_t y_bse = y_tnw + 1;
  int64_t z_bse = z_tnw + 1;
  Eigen::half tnw = static_cast<Eigen::half>((x_bse - x) * (y_bse - y) * (z_bse - z));
  Eigen::half tne = static_cast<Eigen::half>((x - x_bsw) * (y_bsw - y) * (z_bsw - z));
  Eigen::half tsw = static_cast<Eigen::half>((x_bne - x) * (y - y_bne) * (z_bne - z));
  Eigen::half tse = static_cast<Eigen::half>((x - x_bnw) * (y - y_bnw) * (z_bnw - z));
  Eigen::half bnw = static_cast<Eigen::half>((x_tse - x) * (y_tse - y) * (z - z_tse));
  Eigen::half bne = static_cast<Eigen::half>((x - x_tsw) * (y_tsw - y) * (z - z_tsw));
  Eigen::half bsw = static_cast<Eigen::half>((x_tne - x) * (y - y_tne) * (z - z_tne));
  Eigen::half bse = static_cast<Eigen::half>((x - x_tnw) * (y - y_tnw) * (z - z_tnw));
  for (int64_t c = 0; c < y_c; c++, x_ptr_NC += x_stride[1], y_ptr_NCDHW += y_stride[1]) {
    y_data_addr[y_ptr_NCDHW] = static_cast<Eigen::half>(0);
    if (within_bounds_3d(z_tnw, y_tnw, x_tnw, x_dims)) {
      auto x_index = x_ptr_NC + z_tnw * x_stride[kTwo] + y_tnw * x_stride[kThree] + x_tnw * x_stride[kFour];
      y_data_addr[y_ptr_NCDHW] += x_data_addr[x_index] * tnw;
    }
    if (within_bounds_3d(z_tne, y_tne, x_tne, x_dims)) {
      auto x_index = x_ptr_NC + z_tne * x_stride[kTwo] + y_tne * x_stride[kThree] + x_tne * x_stride[kFour];
      y_data_addr[y_ptr_NCDHW] += x_data_addr[x_index] * tne;
    }
    if (within_bounds_3d(z_tsw, y_tsw, x_tsw, x_dims)) {
      auto x_index = x_ptr_NC + z_tsw * x_stride[kTwo] + y_tsw * x_stride[kThree] + x_tsw * x_stride[kFour];
      y_data_addr[y_ptr_NCDHW] += x_data_addr[x_index] * tsw;
    }
    if (within_bounds_3d(z_tse, y_tse, x_tse, x_dims)) {
      auto x_index = x_ptr_NC + z_tse * x_stride[kTwo] + y_tse * x_stride[kThree] + x_tse * x_stride[kFour];
      y_data_addr[y_ptr_NCDHW] += x_data_addr[x_index] * tse;
    }
    if (within_bounds_3d(z_bnw, y_bnw, x_bnw, x_dims)) {
      auto x_index = x_ptr_NC + z_bnw * x_stride[kTwo] + y_bnw * x_stride[kThree] + x_bnw * x_stride[kFour];
      y_data_addr[y_ptr_NCDHW] += x_data_addr[x_index] * bnw;
    }
    if (within_bounds_3d(z_bne, y_bne, x_bne, x_dims)) {
      auto x_index = x_ptr_NC + z_bne * x_stride[kTwo] + y_bne * x_stride[kThree] + x_bne * x_stride[kFour];
      y_data_addr[y_ptr_NCDHW] += x_data_addr[x_index] * bne;
    }
    if (within_bounds_3d(z_bsw, y_bsw, x_bsw, x_dims)) {
      auto x_index = x_ptr_NC + z_bsw * x_stride[kTwo] + y_bsw * x_stride[kThree] + x_bsw * x_stride[kFour];
      y_data_addr[y_ptr_NCDHW] += x_data_addr[x_index] * bsw;
    }
    if (within_bounds_3d(z_bse, y_bse, x_bse, x_dims)) {
      auto x_index = x_ptr_NC + z_bse * x_stride[kTwo] + y_bse * x_stride[kThree] + x_bse * x_stride[kFour];
      y_data_addr[y_ptr_NCDHW] += x_data_addr[x_index] * bse;
    }
  }
}

template <typename T>
void GridSampler3DCpuKernel::nearest_compute(std::vector<T *> addr, std::vector<T> location, const int64_t y_c,
                                             std::vector<int64_t> x_dims, int64_t x_ptr_NC,
                                             std::vector<int64_t> x_stride, int64_t y_ptr_NCDHW,
                                             std::vector<int64_t> y_stride){
  T *x_data_addr = addr[0];
  T *y_data_addr = addr[1];
  T x = location[0];
  T y = location[1];
  T z = location[kTwo];
  int64_t x_nearest = static_cast<int64_t>(std::round(x));
  int64_t y_nearest = static_cast<int64_t>(std::round(y));
  int64_t z_nearest = static_cast<int64_t>(std::round(z));
  for (int64_t c = 0; c < y_c; c++, x_ptr_NC += x_stride[1], y_ptr_NCDHW += y_stride[1]) {
    if (within_bounds_3d(z_nearest, y_nearest, x_nearest, x_dims)) {
      auto x_index = x_ptr_NC + z_nearest * x_stride[kTwo] + y_nearest * x_stride[kThree] + x_nearest * x_stride[kFour];
      y_data_addr[y_ptr_NCDHW] = x_data_addr[x_index];
    } else {
      y_data_addr[y_ptr_NCDHW] = static_cast<T>(0);
    }
  }
}

void GridSampler3DCpuKernel::nearest_compute_half(std::vector<Eigen::half *> addr, std::vector<float> location,
                                                  const int64_t y_c, std::vector<int64_t> x_dims, int64_t x_ptr_NC,
                                                  std::vector<int64_t> x_stride, int64_t y_ptr_NCDHW,
                                                  std::vector<int64_t> y_stride){
  Eigen::half *x_data_addr = addr[0];
  Eigen::half *y_data_addr = addr[1];
  float x = location[0];
  float y = location[1];
  float z = location[kTwo];
  int64_t x_nearest = static_cast<int64_t>(std::round(x));
  int64_t y_nearest = static_cast<int64_t>(std::round(y));
  int64_t z_nearest = static_cast<int64_t>(std::round(z));
  for (int64_t c = 0; c < y_c; c++, x_ptr_NC += x_stride[1], y_ptr_NCDHW += y_stride[1]) {
    if (within_bounds_3d(z_nearest, y_nearest, x_nearest, x_dims)) {
      auto x_index = x_ptr_NC + z_nearest * x_stride[kTwo] + y_nearest * x_stride[kThree] + x_nearest * x_stride[kFour];
      y_data_addr[y_ptr_NCDHW] = x_data_addr[x_index];
    } else {
      y_data_addr[y_ptr_NCDHW] = static_cast<Eigen::half>(0);
    }
  }
}

template <typename T>
T GridSampler3DCpuKernel::grid_sampler_compute_source_index(
    T coord, int64_t size, std::string padding_mode, bool align_corners) {
  if (align_corners) {
    coord = ((coord + 1.f) / kTwo) * (size - 1);
  } else {
    coord = ((coord + 1.f) * size - 1) / kTwo;
  }
  if (padding_mode == "border") {
    coord = std::min(static_cast<T>(size - 1), std::max(coord, static_cast<T>(0)));
  } else if (padding_mode == "reflection") {
    if (align_corners) {
      coord = reflect_coordinates(coord, 0, kTwo * (size - 1));
    } else {
      coord = reflect_coordinates(coord, -1, kTwo * size - 1);
    }
    coord = std::min(static_cast<T>(size - 1), std::max(coord, static_cast<T>(0)));
  }
  return coord;
}

template <typename T>
T GridSampler3DCpuKernel::reflect_coordinates(T coord, int64_t twice_low, int64_t twice_high) {
  if (twice_low == twice_high) {
    return static_cast<T>(0);
  }
  T min = static_cast<T>(twice_low) / kTwo;
  T span = static_cast<T>(twice_high - twice_low) / kTwo;
  coord = std::fabs(coord - min);
  T extra = std::fmod(coord, span);
  int64_t flips = static_cast<int64_t>(std::floor(coord / span));
  if (flips % kTwo == 0) {
    return extra + min;
  } else {
    return span - extra + min;
  }
}

bool GridSampler3DCpuKernel::within_bounds_3d(int64_t d, int64_t h, int64_t w, const std::vector<int64_t> &shape) {
  return d >= 0 && d < shape[kTwo] && h >= 0 && h < shape[kThree] && w >= 0 && w < shape[kFour];
}
REGISTER_CPU_KERNEL(kGridSampler3D, GridSampler3DCpuKernel);
}
