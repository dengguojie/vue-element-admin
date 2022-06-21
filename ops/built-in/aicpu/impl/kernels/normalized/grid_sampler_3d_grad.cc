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

#include "grid_sampler_3d_grad.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "cmath"

namespace {
const char *kGridSampler3DGrad = "GridSampler3DGrad";
const uint32_t kInputNum = 3;
const uint32_t kOutputNum = 2;
const uint64_t kOutputSize = 4 * 256;
const uint32_t kTwo = 2;
const uint32_t kThree = 3;
const uint32_t kFour = 4;

bool check_shape(std::vector<int64_t> grad_shape, std::vector<int64_t> x_shape, std::vector<int64_t> grid_shape) {
  if (grad_shape[0] != x_shape[0] || grid_shape[0] != x_shape[0]) {
    KERNEL_LOG_ERROR(
        "The first dimension of the shape of grad, x and grid must be consistent, but got [%d], [%d] and [%d].",
        grad_shape[0], x_shape[0], grid_shape[0]);
    return false;
  }
  std::vector<int64_t> output_shape {x_shape[0], x_shape[1], grid_shape[1], grid_shape[kTwo], grid_shape[kThree]};
  for (size_t i = 0; i < output_shape.size(); ++i) {
    if (grad_shape[i] != output_shape[i]) {
      KERNEL_LOG_ERROR("The shape of grad must be the same as that of output which is ([%d], [%d], [%d], [%d], [%d]), \
                       but got ([%d], [%d], [%d], [%d], [%d]).", output_shape[0], output_shape[1], output_shape[kTwo],
                       output_shape[kThree], output_shape[kFour], grad_shape[0], grad_shape[1], grad_shape[kTwo],
                       grad_shape[kThree], grad_shape[kFour]);
      return false;
    }
  }
  return true;
}

#define SAFE_ADD_3D_FOR_DX(PTR)                                                \
  safe_add_3d(&dx_data_addr[PTR], loc_tnw, dx_stride, x_dims, tnw * grad_out); \
  safe_add_3d(&dx_data_addr[PTR], loc_tne, dx_stride, x_dims, tne * grad_out); \
  safe_add_3d(&dx_data_addr[PTR], loc_tsw, dx_stride, x_dims, tsw * grad_out); \
  safe_add_3d(&dx_data_addr[PTR], loc_tse, dx_stride, x_dims, tse * grad_out); \
  safe_add_3d(&dx_data_addr[PTR], loc_bnw, dx_stride, x_dims, bnw * grad_out); \
  safe_add_3d(&dx_data_addr[PTR], loc_bne, dx_stride, x_dims, bne * grad_out); \
  safe_add_3d(&dx_data_addr[PTR], loc_bsw, dx_stride, x_dims, bsw * grad_out); \
  safe_add_3d(&dx_data_addr[PTR], loc_bse, dx_stride, x_dims, bse * grad_out);

#define GRIDSAMPLER3DGRAD_COMPUTE_CASE(DTYPE, TYPE, CTX)                       \
  case(DTYPE):{                                                                \
    uint32_t result = GridSampler3DGradCompute<TYPE>(CTX);                     \
    if (result != KERNEL_STATUS_OK) {                                          \
      KERNEL_LOG_ERROR("GridSampler3DGrad kernel compute failed.");            \
      return result;                                                           \
    }                                                                          \
    break;                                                                     \
  }

#define GRIDSAMPLER3DGRAD_COMPUTE_CASE_HALF(DTYPE, CTX)                        \
  case(DTYPE):{                                                                \
    uint32_t result = GridSampler3DGradComputeHalf(CTX);                       \
    if (result != KERNEL_STATUS_OK) {                                          \
      KERNEL_LOG_ERROR("GridSampler3DGrad kernel compute failed.");            \
      return result;                                                           \
    }                                                                          \
    break;                                                                     \
  }
}

namespace aicpu{
uint32_t GridSampler3DGradCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "GridSampler3DGrad check input and output number failed.");
  Tensor *grad_data = ctx.Input(0); 
  Tensor *x_data = ctx.Input(1);
  Tensor *grid_data = ctx.Input(kTwo);
  DataType grad_type = grad_data->GetDataType();
  DataType x_type = x_data->GetDataType();
  DataType grid_type = grid_data->GetDataType();
  if (grad_type != x_type || x_type != grid_type) {
    KERNEL_LOG_ERROR("Input[0], input[1] and input[2] must have same dtype, but get [%s], [%s] and [%s].",
                     DTypeStr(grad_type).c_str(), DTypeStr(x_type).c_str(), DTypeStr(grid_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  switch(x_type) {
    GRIDSAMPLER3DGRAD_COMPUTE_CASE_HALF(DT_FLOAT16, ctx)
    GRIDSAMPLER3DGRAD_COMPUTE_CASE(DT_FLOAT, float, ctx)
    GRIDSAMPLER3DGRAD_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("Input data type[%s] not supported.",DTypeStr(x_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

bool GridSampler3DGradCpuKernel::check_attr(std::string &interpolation_mode, std::string &padding_mode,
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

std::vector<int64_t> GridSampler3DGradCpuKernel::stride_comput(const std::vector<int64_t> &shape) {
  std::vector<int64_t> stride;
  size_t size = shape.size();
  int64_t stride_tmp = 1;
  for (int32_t i = size - 1; i > -1; --i) {
    stride.insert(stride.begin(), stride_tmp);
    stride_tmp *= shape[i];
  }
  return stride;
}

template <typename T>
uint32_t GridSampler3DGradCpuKernel::GridSampler3DGradCompute(CpuKernelContext &ctx) {
  Tensor *grad_data = ctx.Input(0);
  auto grad_data_addr = reinterpret_cast<T *>(grad_data->GetData());
  auto grad_shape = grad_data->GetTensorShape();
  auto grad_dims = grad_shape->GetDimSizes();
  std::vector<int64_t> grad_stride = stride_comput(grad_dims);
  Tensor *x_data = ctx.Input(1);
  auto x_data_addr = reinterpret_cast<T *>(x_data->GetData());
  auto x_shape = x_data->GetTensorShape();
  auto x_dims = x_shape->GetDimSizes();
  std::vector<int64_t> x_stride = stride_comput(x_dims);
  Tensor *grid_data = ctx.Input(kTwo);
  auto grid_data_addr = reinterpret_cast<T *>(grid_data->GetData());
  auto grid_shape = grid_data->GetTensorShape();
  auto grid_dims = grid_shape->GetDimSizes();
  uint64_t grid_data_size = grid_data->GetDataSize();
  if (!check_shape(grad_dims, x_dims, grid_dims)) {return KERNEL_STATUS_PARAM_INVALID;}
  std::vector<int64_t> grid_stride = stride_comput(grid_dims);
  Tensor *dx_data = ctx.Output(0);
  auto dx_data_addr = reinterpret_cast<T *>(dx_data->GetData());
  auto dx_shape = dx_data->GetTensorShape();
  auto dx_dims = dx_shape->GetDimSizes();
  std::vector<int64_t> dx_stride = stride_comput(dx_dims);
  int64_t dx_data_num = dx_data->NumElements();
  for (int64_t i = 0; i < dx_data_num; ++i){
    dx_data_addr[i] = static_cast<T>(0);
  }
  Tensor *dgrid_data = ctx.Output(1);
  auto dgrid_data_addr = reinterpret_cast<T *>(dgrid_data->GetData());
  auto dgrid_shape = dgrid_data->GetTensorShape();
  auto dgrid_dims = dgrid_shape->GetDimSizes();
  std::vector<int64_t> dgrid_stride = stride_comput(dgrid_dims);
  std::string interpolation_mode, padding_mode;
  bool align_corners;
  if (!check_attr(interpolation_mode, padding_mode, align_corners, ctx)) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto shard_compute = [&](size_t start, size_t end){
    for (size_t i = start; i < end; ++i){
      int64_t n = static_cast<int64_t>(i);
      int64_t x_ptr_N = n * x_stride[0];
      const int64_t loc_num = grid_dims[1] * grid_dims[kTwo] * grid_dims[kThree];
      const int64_t iter_begin = n * loc_num;
      const int64_t iter_end = (n + 1) * loc_num;
      for (int64_t iter = iter_begin; iter < iter_end; ++iter) {
        const int64_t d = iter / (grid_dims[kTwo] * grid_dims[kThree]) % grid_dims[1];
        const int64_t h = iter / grid_dims[kThree] % grid_dims[kTwo];
        const int64_t w = iter % grid_dims[kThree];
        const std::vector<int64_t> NDHW = {n, d, h, w};
        int64_t grid_ptr_NDHW = iter * grid_dims[kFour];
        T x = grid_data_addr[grid_ptr_NDHW];
        T y = grid_data_addr[grid_ptr_NDHW + 1];
        T z = grid_data_addr[grid_ptr_NDHW + kTwo];
        T gx_mult, gy_mult, gz_mult;
        x = grid_sampler_compute_source_index_set_grad(x, x_dims[kFour], padding_mode, align_corners, &gx_mult);
        y = grid_sampler_compute_source_index_set_grad(y, x_dims[kThree], padding_mode, align_corners, &gy_mult);
        z = grid_sampler_compute_source_index_set_grad(z, x_dims[kTwo], padding_mode, align_corners, &gz_mult);
        std::vector<T> location {x, y, z};
        if (interpolation_mode == "bilinear") {
          std::vector<T *> addr {grad_data_addr, x_data_addr, dx_data_addr, dgrid_data_addr};
          std::vector<std::vector<int64_t>> strides = {grad_stride, x_stride, dx_stride, dgrid_stride};
          bilinear_compute(addr, location, NDHW, grid_ptr_NDHW, x_ptr_N, x_dims, strides, gx_mult, gy_mult, gz_mult);
        } else if (interpolation_mode == "nearest") {
          std::vector<T *> addr {grad_data_addr, dx_data_addr, dgrid_data_addr};
          std::vector<std::vector<int64_t>> vecs = {x_dims, grad_stride, dx_stride, dgrid_stride};
          nearest_compute(addr, location, NDHW, grid_ptr_NDHW, vecs);
        }
      }
    }
  };
  if (grid_data_size / kThree < kOutputSize) {
    shard_compute(0, x_dims[0]);
  } else {
    size_t data_num = grid_dims[0];
    uint32_t min_core_num = 1;
    size_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kTwo);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    if (max_core_num == 0) {KERNEL_LOG_ERROR("max_core_num could not be 0.");}
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_compute),
                        "GridSampler3DGrad Compute failed.")
  }
  return KERNEL_STATUS_OK;
}

uint32_t GridSampler3DGradCpuKernel::GridSampler3DGradComputeHalf(CpuKernelContext &ctx) {
  Tensor *grad_data = ctx.Input(0);
  auto grad_data_addr = reinterpret_cast<Eigen::half *>(grad_data->GetData());
  auto grad_shape = grad_data->GetTensorShape();
  auto grad_dims = grad_shape->GetDimSizes();
  std::vector<int64_t> grad_stride = stride_comput(grad_dims);
  Tensor *x_data = ctx.Input(1);
  auto x_data_addr = reinterpret_cast<Eigen::half *>(x_data->GetData());
  auto x_shape = x_data->GetTensorShape();
  auto x_dims = x_shape->GetDimSizes();
  std::vector<int64_t> x_stride = stride_comput(x_dims);
  Tensor *grid_data = ctx.Input(kTwo);
  auto grid_data_addr = reinterpret_cast<Eigen::half *>(grid_data->GetData());
  auto grid_shape = grid_data->GetTensorShape();
  auto grid_dims = grid_shape->GetDimSizes();
  uint64_t grid_data_size = grid_data->GetDataSize();
  if (!check_shape(grad_dims, x_dims, grid_dims)) {return KERNEL_STATUS_PARAM_INVALID;}
  std::vector<int64_t> grid_stride = stride_comput(grid_dims);
  Tensor *dx_data = ctx.Output(0);
  auto dx_data_addr = reinterpret_cast<Eigen::half *>(dx_data->GetData());
  auto dx_shape = dx_data->GetTensorShape();
  auto dx_dims = dx_shape->GetDimSizes();
  std::vector<int64_t> dx_stride = stride_comput(dx_dims);
  int64_t dx_data_num = dx_data->NumElements();
  for (int64_t i = 0; i < dx_data_num; ++i){
    dx_data_addr[i] = static_cast<Eigen::half>(0);
  }
  Tensor *dgrid_data = ctx.Output(1);
  auto dgrid_data_addr = reinterpret_cast<Eigen::half *>(dgrid_data->GetData());
  auto dgrid_shape = dgrid_data->GetTensorShape();
  auto dgrid_dims = dgrid_shape->GetDimSizes();
  std::vector<int64_t> dgrid_stride = stride_comput(dgrid_dims);
  std::string interpolation_mode, padding_mode;
  bool align_corners;
  if (!check_attr(interpolation_mode, padding_mode, align_corners, ctx)) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto shard_compute = [&](size_t start, size_t end){
    for (size_t i = start; i < end; ++i){
      int64_t n = static_cast<int64_t>(i);
      int64_t x_ptr_N = n * x_stride[0];
      const int64_t loc_num = grid_dims[1] * grid_dims[kTwo] * grid_dims[kThree];
      const int64_t iter_begin = n * loc_num;
      const int64_t iter_end = (n + 1) * loc_num;
      for (int64_t iter = iter_begin; iter < iter_end; ++iter) {
        const int64_t d = iter / (grid_dims[kTwo] * grid_dims[kThree]) % grid_dims[1];
        const int64_t h = iter / grid_dims[kThree] % grid_dims[kTwo];
        const int64_t w = iter % grid_dims[kThree];
        const std::vector<int64_t> NDHW = {n, d, h, w};
        int64_t grid_ptr_NDHW = iter * grid_dims[kFour];
        float x = static_cast<float>(grid_data_addr[grid_ptr_NDHW]);
        float y = static_cast<float>(grid_data_addr[grid_ptr_NDHW + 1]);
        float z = static_cast<float>(grid_data_addr[grid_ptr_NDHW + kTwo]);
        float gx_mult, gy_mult, gz_mult;
        x = grid_sampler_compute_source_index_set_grad(x, x_dims[kFour], padding_mode, align_corners, &gx_mult);
        y = grid_sampler_compute_source_index_set_grad(y, x_dims[kThree], padding_mode, align_corners, &gy_mult);
        z = grid_sampler_compute_source_index_set_grad(z, x_dims[kTwo], padding_mode, align_corners, &gz_mult);
        std::vector<float> location {x, y, z};
        if (interpolation_mode == "bilinear") {
          std::vector<Eigen::half *> addr {grad_data_addr, x_data_addr, dx_data_addr, dgrid_data_addr};
          std::vector<std::vector<int64_t>> strides = {grad_stride, x_stride, dx_stride, dgrid_stride};
          bilinear_compute_half(addr, location, NDHW, grid_ptr_NDHW, x_ptr_N, x_dims, strides, gx_mult, gy_mult,
                                gz_mult);
        } else if (interpolation_mode == "nearest") {
          std::vector<Eigen::half *> addr {grad_data_addr, dx_data_addr, dgrid_data_addr};
          std::vector<std::vector<int64_t>> vecs = {x_dims, grad_stride, dx_stride, dgrid_stride};
          nearest_compute_half(addr, location, NDHW, grid_ptr_NDHW, vecs);
        }
      }
    }
  };
  if (grid_data_size / kThree < kOutputSize) {
    shard_compute(0, x_dims[0]);
  } else {
    size_t data_num = grid_dims[0];
    uint32_t min_core_num = 1;
    size_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kTwo);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    if (max_core_num == 0) {KERNEL_LOG_ERROR("max_core_num could not be 0.");}
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_compute),
                        "GridSampler3DGrad Compute failed.")
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
void GridSampler3DGradCpuKernel::bilinear_compute(const std::vector<T *> &addr, const std::vector<T> &location,
    const std::vector<int64_t> &NDHW, const int64_t &dgrid_ptr_NDHW, const int64_t &x_ptr_N,
    const std::vector<int64_t> &x_dims, const std::vector<std::vector<int64_t>> &strides, T gx_mult, T gy_mult,
    T gz_mult) {
  T *grad_data_addr = addr[0], *x_data_addr = addr[1], *dx_data_addr = addr[kTwo], *dgrid_data_addr = addr[kThree];
  T x = location[0], y = location[1], z = location[kTwo];
  const std::vector<int64_t> grad_stride = strides[0];
  const std::vector<int64_t> x_stride = strides[1];
  const std::vector<int64_t> dx_stride = strides[kTwo];
  const std::vector<int64_t> dgrid_stride = strides[kThree];
  int64_t x_tnw = static_cast<int64_t>(std::floor(x));
  int64_t y_tnw = static_cast<int64_t>(std::floor(y));
  int64_t z_tnw = static_cast<int64_t>(std::floor(z));
  std::vector<int64_t> loc_tnw = {x_tnw, y_tnw, z_tnw};
  std::vector<int64_t> loc_tne = {x_tnw + 1, y_tnw, z_tnw};
  std::vector<int64_t> loc_tsw = {x_tnw, y_tnw + 1, z_tnw};
  std::vector<int64_t> loc_tse = {x_tnw + 1, y_tnw + 1, z_tnw};
  std::vector<int64_t> loc_bnw = {x_tnw, y_tnw, z_tnw + 1};
  std::vector<int64_t> loc_bne = {x_tnw + 1, y_tnw, z_tnw + 1};
  std::vector<int64_t> loc_bsw = {x_tnw, y_tnw + 1, z_tnw + 1};
  std::vector<int64_t> loc_bse = {x_tnw + 1, y_tnw + 1, z_tnw + 1};
  T tnw = (loc_bse[0] - x) * (loc_bse[1] - y) * (loc_bse[kTwo] - z);
  T tne = (x - loc_bsw[0]) * (loc_bsw[1] - y) * (loc_bsw[kTwo] - z);
  T tsw = (loc_bne[0] - x) * (y - loc_bne[1]) * (loc_bne[kTwo] - z);
  T tse = (x - loc_bnw[0]) * (y - loc_bnw[1]) * (loc_bnw[kTwo] - z);
  T bnw = (loc_tse[0] - x) * (loc_tse[1] - y) * (z - loc_tse[kTwo]);
  T bne = (x - loc_tsw[0]) * (loc_tsw[1] - y) * (z - loc_tsw[kTwo]);
  T bsw = (loc_tne[0] - x) * (y - loc_tne[1]) * (z - loc_tne[kTwo]);
  T bse = (x - loc_tnw[0]) * (y - loc_tnw[1]) * (z - loc_tnw[kTwo]);
  T gx = static_cast<T>(0), gy = static_cast<T>(0), gz = static_cast<T>(0);
  int64_t grad_ptr_NCDHW = NDHW[0] * grad_stride[0] + NDHW[1] * grad_stride[kTwo] + NDHW[kTwo] * grad_stride[kThree] +
                           NDHW[kThree] * grad_stride[kFour];
  int64_t dx_ptr_NC = NDHW[0] * dx_stride[0];
  int64_t x_ptr_NC = x_ptr_N;
  for (int64_t c = 0; c < x_dims[1]; ++c, grad_ptr_NCDHW += grad_stride[1], x_ptr_NC += x_stride[1],
       dx_ptr_NC += dx_stride[1]) {
    T grad_out = grad_data_addr[grad_ptr_NCDHW];
    SAFE_ADD_3D_FOR_DX(dx_ptr_NC);
    if (within_bounds_3d(loc_tnw, x_dims)) {
      auto x_index = x_ptr_NC + z_tnw * x_stride[kTwo] + y_tnw * x_stride[kThree] + x_tnw * x_stride[kFour];
      gx -= x_data_addr[x_index] * (loc_bse[1] - y) * (loc_bse[kTwo] - z) * grad_out;
      gy -= x_data_addr[x_index] * (loc_bse[0] - x) * (loc_bse[kTwo] - z) * grad_out;
      gz -= x_data_addr[x_index] * (loc_bse[0] - x) * (loc_bse[1] - y) * grad_out; 
    }
    if (within_bounds_3d(loc_tne, x_dims)) {
      auto x_index = x_ptr_NC + loc_tne[kTwo] * x_stride[kTwo] + loc_tne[1] * x_stride[kThree] +
                     loc_tne[0] * x_stride[kFour];
      gx += x_data_addr[x_index] * (loc_bsw[1] - y) * (loc_bsw[kTwo] - z) * grad_out;
      gy -= x_data_addr[x_index] * (x - loc_bsw[0]) * (loc_bsw[kTwo] - z) * grad_out;
      gz -= x_data_addr[x_index] * (x - loc_bsw[0]) * (loc_bsw[1] - y) * grad_out;
    }
    if (within_bounds_3d(loc_tsw, x_dims)) {
      auto x_index = x_ptr_NC + loc_tsw[kTwo] * x_stride[kTwo] + loc_tsw[1] * x_stride[kThree] +
                     loc_tsw[0] * x_stride[kFour];
      gx -= x_data_addr[x_index] * (y - loc_bne[1]) * (loc_bne[kTwo] - z) * grad_out;
      gy += x_data_addr[x_index] * (loc_bne[0] - x) * (loc_bne[kTwo] - z) * grad_out;
      gz -= x_data_addr[x_index] * (loc_bne[0] - x) * (y - loc_bne[1]) * grad_out;
    }
    if (within_bounds_3d(loc_tse, x_dims)) {
      auto x_index = x_ptr_NC + loc_tse[kTwo] * x_stride[kTwo] + loc_tse[1] * x_stride[kThree] +
                     loc_tse[0] * x_stride[kFour];
      gx += x_data_addr[x_index] * (y - loc_bnw[1]) * (loc_bnw[kTwo] - z) * grad_out;
      gy += x_data_addr[x_index] * (x - loc_bnw[0]) * (loc_bnw[kTwo] - z) * grad_out;
      gz -= x_data_addr[x_index] * (x - loc_bnw[0]) * (y - loc_bnw[1]) * grad_out;
    }
    if (within_bounds_3d(loc_bnw, x_dims)) {
      auto x_index = x_ptr_NC + loc_bnw[kTwo] * x_stride[kTwo] + loc_bnw[1] * x_stride[kThree] +
                     loc_bnw[0] * x_stride[kFour];
      gx -= x_data_addr[x_index] * (loc_tse[1] - y) * (z - loc_tse[kTwo]) * grad_out;
      gy -= x_data_addr[x_index] * (loc_tse[0] - x) * (z - loc_tse[kTwo]) * grad_out;
      gz += x_data_addr[x_index] * (loc_tse[0] - x) * (loc_tse[1] - y) * grad_out;
    }
    if (within_bounds_3d(loc_bne, x_dims)) {
      auto x_index = x_ptr_NC + loc_bne[kTwo] * x_stride[kTwo] + loc_bne[1] * x_stride[kThree] +
                     loc_bne[0] * x_stride[kFour];
      gx += x_data_addr[x_index] * (loc_tsw[1] - y) * (z - loc_tsw[kTwo]) * grad_out;
      gy -= x_data_addr[x_index] * (x - loc_tsw[0]) * (z - loc_tsw[kTwo]) * grad_out;
      gz += x_data_addr[x_index] * (x - loc_tsw[0]) * (loc_tsw[1] - y) * grad_out;
    }
    if (within_bounds_3d(loc_bsw, x_dims)) {
      auto x_index = x_ptr_NC + loc_bsw[kTwo] * x_stride[kTwo] + loc_bsw[1] * x_stride[kThree] +
                     loc_bsw[0] * x_stride[kFour];
      gx -= x_data_addr[x_index] * (y - loc_tne[1]) * (z - loc_tne[kTwo]) * grad_out;
      gy += x_data_addr[x_index] * (loc_tne[0] - x) * (z - loc_tne[kTwo]) * grad_out;
      gz += x_data_addr[x_index] * (loc_tne[0] - x) * (y - loc_tne[1]) * grad_out;
    }
    if (within_bounds_3d(loc_bse, x_dims)) {
      auto x_index = x_ptr_NC + loc_bse[kTwo] * x_stride[kTwo] + loc_bse[1] * x_stride[kThree] +
                     loc_bse[0] * x_stride[kFour];
      gx += x_data_addr[x_index] * (y - y_tnw) * (z - z_tnw) * grad_out;
      gy += x_data_addr[x_index] * (x - x_tnw) * (z - z_tnw) * grad_out;
      gz += x_data_addr[x_index] * (x - x_tnw) * (y - y_tnw) * grad_out;
    }
  }
  dgrid_data_addr[dgrid_ptr_NDHW] = gx_mult * gx;
  dgrid_data_addr[dgrid_ptr_NDHW + 1] = gy_mult * gy;
  dgrid_data_addr[dgrid_ptr_NDHW + kTwo] = gz_mult * gz;
}

void GridSampler3DGradCpuKernel::bilinear_compute_half(const std::vector<Eigen::half *> &addr,
    const std::vector<float> &location, const std::vector<int64_t> &NDHW, const int64_t &dgrid_ptr_NDHW,
    const int64_t &x_ptr_N, const std::vector<int64_t> &x_dims, const std::vector<std::vector<int64_t>> &strides,
    float gx_mult, float gy_mult, float gz_mult) {
  Eigen::half *grad_data_addr = addr[0], *x_data_addr = addr[1];
  Eigen::half *dx_data_addr = addr[kTwo], *dgrid_data_addr = addr[kThree];
  float x = location[0], y = location[1], z = location[kTwo];
  const std::vector<int64_t> grad_stride = strides[0];
  const std::vector<int64_t> x_stride = strides[1];
  const std::vector<int64_t> dx_stride = strides[kTwo];
  const std::vector<int64_t> dgrid_stride = strides[kThree];
  int64_t x_tnw = static_cast<int64_t>(std::floor(x));
  int64_t y_tnw = static_cast<int64_t>(std::floor(y));
  int64_t z_tnw = static_cast<int64_t>(std::floor(z));
  std::vector<int64_t> loc_tnw = {x_tnw, y_tnw, z_tnw};
  std::vector<int64_t> loc_tne = {x_tnw + 1, y_tnw, z_tnw};
  std::vector<int64_t> loc_tsw = {x_tnw, y_tnw + 1, z_tnw};
  std::vector<int64_t> loc_tse = {x_tnw + 1, y_tnw + 1, z_tnw};
  std::vector<int64_t> loc_bnw = {x_tnw, y_tnw, z_tnw + 1};
  std::vector<int64_t> loc_bne = {x_tnw + 1, y_tnw, z_tnw + 1};
  std::vector<int64_t> loc_bsw = {x_tnw, y_tnw + 1, z_tnw + 1};
  std::vector<int64_t> loc_bse = {x_tnw + 1, y_tnw + 1, z_tnw + 1};
  Eigen::half tnw = static_cast<Eigen::half>((loc_bse[0] - x) * (loc_bse[1] - y) * (loc_bse[kTwo] - z));
  Eigen::half tne = static_cast<Eigen::half>((x - loc_bsw[0]) * (loc_bsw[1] - y) * (loc_bsw[kTwo] - z));
  Eigen::half tsw = static_cast<Eigen::half>((loc_bne[0] - x) * (y - loc_bne[1]) * (loc_bne[kTwo] - z));
  Eigen::half tse = static_cast<Eigen::half>((x - loc_bnw[0]) * (y - loc_bnw[1]) * (loc_bnw[kTwo] - z));
  Eigen::half bnw = static_cast<Eigen::half>((loc_tse[0] - x) * (loc_tse[1] - y) * (z - loc_tse[kTwo]));
  Eigen::half bne = static_cast<Eigen::half>((x - loc_tsw[0]) * (loc_tsw[1] - y) * (z - loc_tsw[kTwo]));
  Eigen::half bsw = static_cast<Eigen::half>((loc_tne[0] - x) * (y - loc_tne[1]) * (z - loc_tne[kTwo]));
  Eigen::half bse = static_cast<Eigen::half>((x - loc_tnw[0]) * (y - loc_tnw[1]) * (z - loc_tnw[kTwo]));
  Eigen::half gx = static_cast<Eigen::half>(0), gy = static_cast<Eigen::half>(0), gz = static_cast<Eigen::half>(0);
  int64_t grad_ptr_NCDHW = NDHW[0] * grad_stride[0] + NDHW[1] * grad_stride[kTwo] + NDHW[kTwo] * grad_stride[kThree] +
                           NDHW[kThree] * grad_stride[kFour];
  int64_t dx_ptr_NC = NDHW[0] * dx_stride[0];
  int64_t x_ptr_NC = x_ptr_N;
  for (int64_t c = 0; c < x_dims[1]; ++c, grad_ptr_NCDHW += grad_stride[1], x_ptr_NC += x_stride[1],
       dx_ptr_NC += dx_stride[1]) {
    Eigen::half grad_out = grad_data_addr[grad_ptr_NCDHW];
    SAFE_ADD_3D_FOR_DX(dx_ptr_NC);
    if (within_bounds_3d(loc_tnw, x_dims)) {
      auto x_index = x_ptr_NC + z_tnw * x_stride[kTwo] + y_tnw * x_stride[kThree] + x_tnw * x_stride[kFour];
      gx -= x_data_addr[x_index] * static_cast<Eigen::half>((loc_bse[1] - y) * (loc_bse[kTwo] - z)) * grad_out;
      gy -= x_data_addr[x_index] * static_cast<Eigen::half>((loc_bse[0] - x) * (loc_bse[kTwo] - z)) * grad_out;
      gz -= x_data_addr[x_index] * static_cast<Eigen::half>((loc_bse[0] - x) * (loc_bse[1] - y)) * grad_out; 
    }
    if (within_bounds_3d(loc_tne, x_dims)) {
      auto x_index = x_ptr_NC + loc_tne[kTwo] * x_stride[kTwo] + loc_tne[1] * x_stride[kThree] +
                     loc_tne[0] * x_stride[kFour];
      gx += x_data_addr[x_index] * static_cast<Eigen::half>((loc_bsw[1] - y) * (loc_bsw[kTwo] - z)) * grad_out;
      gy -= x_data_addr[x_index] * static_cast<Eigen::half>((x - loc_bsw[0]) * (loc_bsw[kTwo] - z)) * grad_out;
      gz -= x_data_addr[x_index] * static_cast<Eigen::half>((x - loc_bsw[0]) * (loc_bsw[1] - y)) * grad_out;
    }
    if (within_bounds_3d(loc_tsw, x_dims)) {
      auto x_index = x_ptr_NC + loc_tsw[kTwo] * x_stride[kTwo] + loc_tsw[1] * x_stride[kThree] +
                     loc_tsw[0] * x_stride[kFour];
      gx -= x_data_addr[x_index] * static_cast<Eigen::half>((y - loc_bne[1]) * (loc_bne[kTwo] - z)) * grad_out;
      gy += x_data_addr[x_index] * static_cast<Eigen::half>((loc_bne[0] - x) * (loc_bne[kTwo] - z)) * grad_out;
      gz -= x_data_addr[x_index] * static_cast<Eigen::half>((loc_bne[0] - x) * (y - loc_bne[1])) * grad_out;
    }
    if (within_bounds_3d(loc_tse, x_dims)) {
      auto x_index = x_ptr_NC + loc_tse[kTwo] * x_stride[kTwo] + loc_tse[1] * x_stride[kThree] +
                     loc_tse[0] * x_stride[kFour];
      gx += x_data_addr[x_index] * static_cast<Eigen::half>((y - loc_bnw[1]) * (loc_bnw[kTwo] - z)) * grad_out;
      gy += x_data_addr[x_index] * static_cast<Eigen::half>((x - loc_bnw[0]) * (loc_bnw[kTwo] - z)) * grad_out;
      gz -= x_data_addr[x_index] * static_cast<Eigen::half>((x - loc_bnw[0]) * (y - loc_bnw[1])) * grad_out;
    }
    if (within_bounds_3d(loc_bnw, x_dims)) {
      auto x_index = x_ptr_NC + loc_bnw[kTwo] * x_stride[kTwo] + loc_bnw[1] * x_stride[kThree] +
                     loc_bnw[0] * x_stride[kFour];
      gx -= x_data_addr[x_index] * static_cast<Eigen::half>((loc_tse[1] - y) * (z - loc_tse[kTwo])) * grad_out;
      gy -= x_data_addr[x_index] * static_cast<Eigen::half>((loc_tse[0] - x) * (z - loc_tse[kTwo])) * grad_out;
      gz += x_data_addr[x_index] * static_cast<Eigen::half>((loc_tse[0] - x) * (loc_tse[1] - y)) * grad_out;
    }
    if (within_bounds_3d(loc_bne, x_dims)) {
      auto x_index = x_ptr_NC + loc_bne[kTwo] * x_stride[kTwo] + loc_bne[1] * x_stride[kThree] +
                     loc_bne[0] * x_stride[kFour];
      gx += x_data_addr[x_index] * static_cast<Eigen::half>((loc_tsw[1] - y) * (z - loc_tsw[kTwo])) * grad_out;
      gy -= x_data_addr[x_index] * static_cast<Eigen::half>((x - loc_tsw[0]) * (z - loc_tsw[kTwo])) * grad_out;
      gz += x_data_addr[x_index] * static_cast<Eigen::half>((x - loc_tsw[0]) * (loc_tsw[1] - y)) * grad_out;
    }
    if (within_bounds_3d(loc_bsw, x_dims)) {
      auto x_index = x_ptr_NC + loc_bsw[kTwo] * x_stride[kTwo] + loc_bsw[1] * x_stride[kThree] +
                     loc_bsw[0] * x_stride[kFour];
      gx -= x_data_addr[x_index] * static_cast<Eigen::half>((y - loc_tne[1]) * (z - loc_tne[kTwo])) * grad_out;
      gy += x_data_addr[x_index] * static_cast<Eigen::half>((loc_tne[0] - x) * (z - loc_tne[kTwo])) * grad_out;
      gz += x_data_addr[x_index] * static_cast<Eigen::half>((loc_tne[0] - x) * (y - loc_tne[1])) * grad_out;
    }
    if (within_bounds_3d(loc_bse, x_dims)) {
      auto x_index = x_ptr_NC + loc_bse[kTwo] * x_stride[kTwo] + loc_bse[1] * x_stride[kThree] +
                     loc_bse[0] * x_stride[kFour];
      gx += x_data_addr[x_index] * static_cast<Eigen::half>((y - y_tnw) * (z - z_tnw)) * grad_out;
      gy += x_data_addr[x_index] * static_cast<Eigen::half>((x - x_tnw) * (z - z_tnw)) * grad_out;
      gz += x_data_addr[x_index] * static_cast<Eigen::half>((x - x_tnw) * (y - y_tnw)) * grad_out;
    }
  }
  dgrid_data_addr[dgrid_ptr_NDHW] = static_cast<Eigen::half>(gx_mult) * gx;
  dgrid_data_addr[dgrid_ptr_NDHW + 1] = static_cast<Eigen::half>(gy_mult) * gy;
  dgrid_data_addr[dgrid_ptr_NDHW + kTwo] = static_cast<Eigen::half>(gz_mult) * gz;
}

template <typename T>
void GridSampler3DGradCpuKernel::nearest_compute(const std::vector<T *> &addr, const std::vector<T> &location,
    const std::vector<int64_t> &NDHW, const int64_t &dgrid_ptr_NDHW, const std::vector<std::vector<int64_t>> &vecs) {
  T *grad_data_addr = addr[0], *dx_data_addr = addr[1], *dgrid_data_addr = addr[kTwo];
  T x = location[0], y = location[1], z = location[kTwo];
  const std::vector<int64_t> x_dims = vecs[0];
  const std::vector<int64_t> grad_stride = vecs[1];
  const std::vector<int64_t> dx_stride = vecs[kTwo];
  const std::vector<int64_t> dgrid_stride = vecs[kThree];
  int64_t x_nearest = static_cast<int64_t>(std::round(x));
  int64_t y_nearest = static_cast<int64_t>(std::round(y));
  int64_t z_nearest = static_cast<int64_t>(std::round(z));
  std::vector<int64_t> loc_nearest = {x_nearest, y_nearest, z_nearest};
  int64_t grad_ptr_NCDHW = NDHW[0] * grad_stride[0] + NDHW[1] * grad_stride[kTwo] + NDHW[kTwo] * grad_stride[kThree] +
                           NDHW[kThree] * grad_stride[kFour];
  int64_t dx_ptr_NC = NDHW[0] * dx_stride[0];
  for (int64_t c = 0; c < x_dims[1]; ++c, grad_ptr_NCDHW += grad_stride[1], dx_ptr_NC += dx_stride[1]) {
    safe_add_3d(&dx_data_addr[dx_ptr_NC], loc_nearest, dx_stride, x_dims, grad_data_addr[grad_ptr_NCDHW]);
  }
  dgrid_data_addr[dgrid_ptr_NDHW] = static_cast<T>(0);
  dgrid_data_addr[dgrid_ptr_NDHW + 1] = static_cast<T>(0);
  dgrid_data_addr[dgrid_ptr_NDHW + kTwo] = static_cast<T>(0);
}

void GridSampler3DGradCpuKernel::nearest_compute_half(const std::vector<Eigen::half *> &addr,
    const std::vector<float> &location, const std::vector<int64_t> &NDHW, const int64_t &dgrid_ptr_NDHW,
    const std::vector<std::vector<int64_t>> &vecs) {
  Eigen::half *grad_data_addr = addr[0], *dx_data_addr = addr[1], *dgrid_data_addr = addr[kTwo];
  float x = location[0], y = location[1], z = location[kTwo];
  const std::vector<int64_t> x_dims = vecs[0];
  const std::vector<int64_t> grad_stride = vecs[1];
  const std::vector<int64_t> dx_stride = vecs[kTwo];
  const std::vector<int64_t> dgrid_stride = vecs[kThree];
  int64_t x_nearest = static_cast<int64_t>(std::round(x));
  int64_t y_nearest = static_cast<int64_t>(std::round(y));
  int64_t z_nearest = static_cast<int64_t>(std::round(z));
  std::vector<int64_t> loc_nearest = {x_nearest, y_nearest, z_nearest};
  int64_t grad_ptr_NCDHW = NDHW[0] * grad_stride[0] + NDHW[1] * grad_stride[kTwo] + NDHW[kTwo] * grad_stride[kThree] +
                           NDHW[kThree] * grad_stride[kFour];
  int64_t dx_ptr_NC = NDHW[0] * dx_stride[0];
  for (int64_t c = 0; c < x_dims[1]; ++c, grad_ptr_NCDHW += grad_stride[1], dx_ptr_NC += dx_stride[1]) {
    safe_add_3d(&dx_data_addr[dx_ptr_NC], loc_nearest, dx_stride, x_dims, grad_data_addr[grad_ptr_NCDHW]);
  }
  dgrid_data_addr[dgrid_ptr_NDHW] = static_cast<Eigen::half>(0);
  dgrid_data_addr[dgrid_ptr_NDHW + 1] = static_cast<Eigen::half>(0);
  dgrid_data_addr[dgrid_ptr_NDHW + kTwo] = static_cast<Eigen::half>(0);
}

template <typename T>
T GridSampler3DGradCpuKernel::grid_sampler_compute_source_index_set_grad(
    T coord, int64_t size, std::string padding_mode, bool align_corners, T *grad_x) {
  T grad_clip, grad_refl;
  if (align_corners) {
    *grad_x = static_cast<T>(size - 1) / kTwo;
    coord = ((coord + 1) / kTwo) * (size - 1);
  } else {
    *grad_x = static_cast<T>(size) / kTwo;
    coord = ((coord + 1) * size - 1) / kTwo;
  }
  if (padding_mode == "border") {
    coord = clip_coordinates_set_grad(coord, size, &grad_clip);
    *grad_x = (*grad_x) * grad_clip;
  } else if (padding_mode == "reflection") {
    if (align_corners) {
      coord = reflect_coordinates_set_grad(coord, 0, kTwo * (size - 1), &grad_refl);
    } else {
      coord = reflect_coordinates_set_grad(coord, -1, kTwo * size - 1, &grad_refl);
    }
    coord = clip_coordinates_set_grad(coord, size, &grad_clip);
    *grad_x = (*grad_x) * grad_refl * grad_clip;
  }
  return coord;
}

template <typename T>
T GridSampler3DGradCpuKernel::clip_coordinates_set_grad(T x, int64_t clip_limit, T *grad_x) {
  if (x <= static_cast<T>(0)) {
    *grad_x = static_cast<T>(0);
    return static_cast<T>(0);
  } else {
    T max = static_cast<T>(clip_limit - 1);
    if (x >= max) {
      *grad_x = static_cast<T>(0);
      return max;
    } else {
      *grad_x = static_cast<T>(1);
      return x;
    }
  }
}

template <typename T>
T GridSampler3DGradCpuKernel::reflect_coordinates_set_grad(T x, int64_t twice_low, int64_t twice_high, T *grad_x) {
  if (twice_low == twice_high) {
    *grad_x = static_cast<T>(0);
    return static_cast<T>(0);
  }
  int64_t grad_x_mult_;
  T min = static_cast<T>(twice_low) / kTwo;
  T span = static_cast<T>(twice_high - twice_low) / kTwo;
  x = x - min;
  if (x < static_cast<T>(0)){
    grad_x_mult_ = -1;
    x = -x;
  } else {
    grad_x_mult_ = 1;
  }
  T extra = std::fmod(x, span);
  int64_t flips = static_cast<int64_t>(std::floor(x / span));
  if (flips % kTwo == 0) {
    *grad_x = static_cast<T>(grad_x_mult_);
    return extra + min;
  } else {
    *grad_x = static_cast<T>(-grad_x_mult_);
    return span - extra + min;
  }
}

template <typename T>
void GridSampler3DGradCpuKernel::safe_add_3d(T *data, const std::vector<int64_t> &loc,
                                             const std::vector<int64_t> &stride,
                                             const std::vector<int64_t> &shape, T delta) {
  const int64_t x = loc[0];
  const int64_t y = loc[1];
  const int64_t z = loc[2];
  if (within_bounds_3d(loc, shape)) {
    data[z * stride[kTwo] + y * stride[kThree] + x * stride[kFour]] += delta;
  }
}

bool GridSampler3DGradCpuKernel::within_bounds_3d(const std::vector<int64_t> &loc, const std::vector<int64_t> &shape) {
  const int64_t x = loc[0];
  const int64_t y = loc[1];
  const int64_t z = loc[2];
  return z >= 0 && z < shape[kTwo] && y >= 0 && y < shape[kThree] && x >= 0 && x < shape[kFour];
}
REGISTER_CPU_KERNEL(kGridSampler3DGrad, GridSampler3DGradCpuKernel);
}
