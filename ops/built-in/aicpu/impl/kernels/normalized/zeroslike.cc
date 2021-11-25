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
#include "zeroslike.h"
#include <cstring>

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "Eigen/Dense"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *kZerosLike = "ZerosLike";

#define ZEROSLIKE_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                           \
    uint32_t result = ZerosLikePartCompute<TYPE>(CTX);      \
    if (result != KERNEL_STATUS_OK) {                       \
      KERNEL_LOG_ERROR("ZerosLike kernel compute failed."); \
      return result;                                        \
    }                                                       \
  break;                                                    \
  }
}  // namespace

namespace aicpu {
uint32_t ZerosLikeCpuKernel::Compute(CpuKernelContext &ctx) {
     // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "[%s] check input and output failed.", kZerosLike);
  KERNEL_HANDLE_ERROR(ZerosLikeCheck(ctx), "[%s] check params failed.",
                        kZerosLike);
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    ZEROSLIKE_COMPUTE_CASE(DT_BOOL, bool, ctx)
    ZEROSLIKE_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    ZEROSLIKE_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    ZEROSLIKE_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    ZEROSLIKE_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    ZEROSLIKE_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    ZEROSLIKE_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    ZEROSLIKE_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    ZEROSLIKE_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    ZEROSLIKE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    ZEROSLIKE_COMPUTE_CASE(DT_FLOAT, float, ctx)
    ZEROSLIKE_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    ZEROSLIKE_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    ZEROSLIKE_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
  default:
    KERNEL_LOG_ERROR("ZerosLike kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t ZerosLikeCpuKernel::ZerosLikeCheck(CpuKernelContext &ctx) {
  auto input_0 = ctx.Input(0);
  auto output_0 = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(input_0->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input data failed.")
  KERNEL_CHECK_NULLPTR(output_0->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get output data failed")
  KERNEL_CHECK_NULLPTR(input_0->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input tensor shape failed.")
  std::vector<int64_t> shape_x = input_0->GetTensorShape()->GetDimSizes();
  size_t shape_size = shape_x.size();
  KERNEL_CHECK_FALSE((shape_size > 0), KERNEL_STATUS_PARAM_INVALID,
                     "Input must be at least rank 1, got [%zu].",
                     shape_x.size())
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t ZerosLikeCpuKernel::ZerosLikePartCompute(CpuKernelContext &ctx) {
  size_t data_num = ctx.Input(0)->NumElements();
  Tensor *y = ctx.Output(0);
  auto y_addr = y->GetData();
  memset(y_addr, 0 , data_num*sizeof(T));
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kZerosLike, ZerosLikeCpuKernel);
}  // namespace aicpu
