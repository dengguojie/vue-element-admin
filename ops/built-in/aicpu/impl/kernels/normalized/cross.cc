/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
#include "cross.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const int64_t kParallelNum = 2 * 1024;
const char *kCross = "Cross";
const int64_t kThreeNum = 3;
#define Cross_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                       \
    uint32_t result = CrossCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                   \
      KERNEL_LOG_ERROR("Cross kernel compute failed."); \
      return result;                                    \
    }                                                   \
    break;                                              \
  }
}  // namespace

namespace aicpu {
uint32_t CrossCpuKernel::GetDimAndCheck(const CpuKernelContext &ctx) {
  auto input1_data_shape = ctx.Input(0)->GetTensorShape();
  auto input2_data_shape = ctx.Input(1)->GetTensorShape();
  AttrValue *dim_attr = ctx.GetAttr("dim");
  if (dim_attr == nullptr) {
    for (int64_t i = 0; i < input1_data_shape->GetDims(); i++) {
      if (input1_data_shape->GetDimSize(i) == kThreeNum) {
        dim_ = i;
        break;
      }
      if (i == input1_data_shape->GetDims() - 1 &&
          input1_data_shape->GetDimSize(i) != kThreeNum) {
        KERNEL_LOG_ERROR("The size of inputs dim should be 3,but got [%d].",
                         input1_data_shape->GetDimSize(i));
        return KERNEL_STATUS_PARAM_INVALID;
      }
    }
  } else {
    dim_ = dim_attr->GetInt();
  }

  if (input1_data_shape->GetDims() != input2_data_shape->GetDims()) {
    KERNEL_LOG_ERROR("The shape of two inputs must have the same size.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  for (int64_t i = 0; i < input1_data_shape->GetDims(); ++i) {
    if (input1_data_shape->GetDimSize(i) != input2_data_shape->GetDimSize(i)) {
      KERNEL_LOG_ERROR("input1 and input2 must have the same shape value.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  if (dim_ < -(input1_data_shape->GetDims()) ||
      dim_ > (input1_data_shape->GetDims()) - 1) {
    KERNEL_LOG_ERROR("dim should between [%d] and [%d],but got [%d].",
                     -(input1_data_shape->GetDims()),
                     (input1_data_shape->GetDims()) - 1, dim_);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (dim_ < 0) {
    dim_ = input1_data_shape->GetDims() + dim_;
  }
  if (input1_data_shape->GetDimSize(dim_) != kThreeNum &&
      input2_data_shape->GetDimSize(dim_) != kThreeNum) {
    KERNEL_LOG_ERROR("The size of inputs dim should be 3,but got [%d].",
                     input1_data_shape->GetDimSize(dim_));
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t CrossCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Cross check input and output number failed.");
  KERNEL_HANDLE_ERROR(GetDimAndCheck(ctx),
                      "[%s] check params failed.", kCross);
  auto output_type = ctx.Output(0)->GetDataType();
  switch (output_type) {
    Cross_COMPUTE_CASE(DT_DOUBLE, double, ctx);
    Cross_COMPUTE_CASE(DT_FLOAT, float, ctx);
    Cross_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx);
    Cross_COMPUTE_CASE(DT_INT8, int8_t, ctx);
    Cross_COMPUTE_CASE(DT_INT16, int16_t, ctx);
    Cross_COMPUTE_CASE(DT_INT32, int32_t, ctx);
    Cross_COMPUTE_CASE(DT_INT64, int64_t, ctx);
    Cross_COMPUTE_CASE(DT_UINT8, uint8_t, ctx);
    Cross_COMPUTE_CASE(DT_UINT16, uint16_t, ctx);
    Cross_COMPUTE_CASE(DT_UINT32, uint32_t, ctx);
    Cross_COMPUTE_CASE(DT_UINT64, uint64_t, ctx);
    Cross_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx);
    Cross_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx);
    default:
      KERNEL_LOG_ERROR("Cross kernel data type [%s] not support.",
                       DTypeStr(output_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T1>
uint32_t CrossCpuKernel::CrossCompute(CpuKernelContext &ctx) {
  Tensor *input1_data = ctx.Input(0);
  auto input1_data_addr = reinterpret_cast<T1 *>(input1_data->GetData());
  auto input1_data_shape = input1_data->GetTensorShape();
  const std::vector<int64_t> input1_data_shape_dims =
      input1_data_shape->GetDimSizes();
  int64_t input1_data_num = input1_data->NumElements();

  Tensor *input2_data = ctx.Input(1);
  auto input2_data_addr = reinterpret_cast<T1 *>(input2_data->GetData());
  auto input2_data_shape = input2_data->GetTensorShape();
  auto inpu2_data_dim_sizes = input2_data_shape->GetDimSizes();
  std::vector<int64_t> input2_data_shape_dims = inpu2_data_dim_sizes;

  Tensor *output_data = ctx.Output(0);
  auto output_data_addr = reinterpret_cast<T1 *>(output_data->GetData());
  auto output_data_shape = output_data->GetTensorShape();
  std::vector<int64_t> output_data_shape_dims =
      output_data_shape->GetDimSizes();
  int64_t dim = dim_;
  int64_t total = input1_data_num / 3;
  auto a_dims_num = input1_data_shape->GetDims();
  const int64_t n = a_dims_num;
  int64_t a_stride[n];
  int64_t stride_tmp = 1;
  for (int64_t i = n - 1; i > -1; i--) {
    a_stride[i] = stride_tmp;
    stride_tmp *= input1_data_shape_dims[i];
  }
  int64_t input1_data_stride = a_stride[dim];
  int64_t b_stride[n];
  stride_tmp = 1;
  for (int64_t i = n - 1; i > -1; i--) {
    b_stride[i] = stride_tmp;
    stride_tmp *= input2_data_shape_dims[i];
  }
  int64_t input2_data_stride = b_stride[dim];
  int64_t r_stride[n];
  stride_tmp = 1;
  for (int64_t i = n - 1; i > -1; i--) {
    r_stride[i] = stride_tmp;
    stride_tmp *= output_data_shape_dims[i];
  }
  int64_t output_data_stride = r_stride[dim];
  auto cross_shard = [&](int64_t start, int64_t end) {
    const int64_t input1_data_dim = input1_data_shape->GetDims();
    std::vector<int64_t> position_in_dims(input1_data_dim);
    int64_t index_in_curr_dim = start;
    int64_t input1_data_start = 0;
    int64_t input2_data_start = 0;
    int64_t output_data_start = 0;
    for (int64_t i = 0; i < input1_data_dim; i++) {
      if (i == dim) {
        continue;
      }
      position_in_dims[i] =
          index_in_curr_dim % input1_data_shape->GetDimSize(i);
      input1_data_start +=
          (index_in_curr_dim % input1_data_shape->GetDimSize(i)) * a_stride[i];
      input2_data_start +=
          (index_in_curr_dim % input2_data_shape->GetDimSize(i)) * b_stride[i];
      output_data_start +=
          (index_in_curr_dim % output_data_shape->GetDimSize(i)) * r_stride[i];
      index_in_curr_dim = index_in_curr_dim / input1_data_shape->GetDimSize(i);
    }
    while (start < end) {
      output_data_addr[output_data_start + 0 * output_data_stride] =
          input1_data_addr[input1_data_start + 1 * input1_data_stride] *
              input2_data_addr[input2_data_start + 2 * input2_data_stride] -
          input1_data_addr[input1_data_start + 2 * input1_data_stride] *
              input2_data_addr[input2_data_start + 1 * input2_data_stride];
      output_data_addr[output_data_start + 1 * output_data_stride] =
          input1_data_addr[input1_data_start + 2 * input1_data_stride] *
              input2_data_addr[input2_data_start + 0 * input2_data_stride] -
          input1_data_addr[input1_data_start + 0 * input1_data_stride] *
              input2_data_addr[input2_data_start + 2 * input2_data_stride];
      output_data_addr[output_data_start + 2 * output_data_stride] =
          input1_data_addr[input1_data_start + 0 * input1_data_stride] *
              input2_data_addr[input2_data_start + 1 * input2_data_stride] -
          input1_data_addr[input1_data_start + 1 * input1_data_stride] *
              input2_data_addr[input2_data_start + 0 * input2_data_stride];
      start++;
      for (int i = 0; i < input1_data_dim; i++) {
        if (i == dim) {
          continue;
        }
        position_in_dims[i]++;
        input1_data_start += a_stride[i];
        input2_data_start += b_stride[i];
        output_data_start += r_stride[i];
        if (position_in_dims[i] == input1_data_shape->GetDimSize(i) &&
            i != input1_data_shape->GetDims() - 1) {
          input1_data_start -= position_in_dims[i] * a_stride[i];
          input2_data_start -= position_in_dims[i] * b_stride[i];
          output_data_start -= position_in_dims[i] * r_stride[i];
          position_in_dims[i] = 0;
        } else {
          break;
        }
      }
    }
  };
  if (total > kParallelNum) {
    const int64_t max_core_num = std::max(
        static_cast<int64_t>(1),
        static_cast<int64_t>(aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2));
    const int64_t per_unit_size = total / std::min(total, max_core_num);
    KERNEL_HANDLE_ERROR(
        CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, cross_shard),
        "Cross compute failed");
  } else {
    cross_shard(0, total);
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kCross, CrossCpuKernel);
}  // namespace aicpu
