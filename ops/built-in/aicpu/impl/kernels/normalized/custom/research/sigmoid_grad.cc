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
#include "sigmoid_grad.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *kSigmoidGrad = "SigmoidGrad";
constexpr int64_t kParallelDataNums = 128 * 1024;
constexpr int64_t kParallelDataNumsMid1 = 512 * 1024;
constexpr int64_t kParallelDataNumsMid2 = 1024 * 1024;

#define SigmoidGrad_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                             \
    uint32_t result = SigmoidGradCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                         \
      KERNEL_LOG_ERROR("SigmoidGrad kernel compute failed."); \
      return result;                                          \
    }                                                         \
    break;                                                    \
  }
}  // namespace

namespace aicpu {
uint32_t SigmoidGradCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "[%s] check input and output failed.", kSigmoidGrad);
  KERNEL_HANDLE_ERROR(SigmoidGradCheck(ctx), "[%s] check params failed.",
                      kSigmoidGrad);
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    SigmoidGrad_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    SigmoidGrad_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    SigmoidGrad_COMPUTE_CASE(DT_FLOAT, float, ctx)
    SigmoidGrad_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    SigmoidGrad_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default :
      KERNEL_LOG_ERROR("SigmoidGrad kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t SigmoidGradCpuKernel::SigmoidGradCheck(CpuKernelContext &ctx) {
  auto input_0 = ctx.Input(0);
  auto input_1 = ctx.Input(1);
  auto output = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(input_0->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input 0 data failed.")
  KERNEL_CHECK_NULLPTR(input_1->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input 1 data failed.")
  KERNEL_CHECK_NULLPTR(output->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get output data failed")
  DataType input0_type = input_0->GetDataType();
  DataType input1_type = input_1->GetDataType();
  KERNEL_CHECK_FALSE((input0_type == input1_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input0 [%s] need be same with "
                     "input1 [%s].",
                     DTypeStr(input0_type).c_str(),
                     DTypeStr(input1_type).c_str())

  auto input0_size = input_0->GetDataSize();
  auto input1_size = input_1->GetDataSize();
  auto output_size = output->GetDataSize();
  KERNEL_CHECK_FALSE((input0_size == output_size), KERNEL_STATUS_PARAM_INVALID,
                     "The data size of output [%llu] need be same with "
                     "input0 [%llu].",
                     output_size, input0_size)

  KERNEL_CHECK_FALSE((input0_size == input1_size), KERNEL_STATUS_PARAM_INVALID,
                     "The data size of input1 [%llu] need be same with "
                     "input0 [%llu].",
                     input1_size, input0_size)

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SigmoidGradCpuKernel::SigmoidGradCompute(CpuKernelContext &ctx) {
  auto input_y = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto input_dy = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto output_z = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  size_t data_num = ctx.Input(0)->NumElements();
  int64_t data_size = data_num * sizeof(T);
  if (data_size <= kParallelDataNums) {
    T one_trans = static_cast<T>(1.0);
    for (size_t i = 0; i < data_num; i++) {
      auto y_idx = input_y + i;    // i-th value of input0
      auto dy_idx = input_dy + i;  // i-th value of input1
      *(output_z + i) = (one_trans - (*y_idx)) * (*y_idx) * (*dy_idx);
    }
  } else {
    uint32_t min_core_num = 1;
    uint32_t max_core_num =
        std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);

    if (data_size <= kParallelDataNumsMid1) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    } else if (data_size <= kParallelDataNumsMid2) {
      max_core_num = std::min(max_core_num, 6U);  // up to 6 cpu cores
    }

    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    T one_trans = static_cast<T>(1.0);
    auto shard_SigmoidGrad = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        auto y_idx = input_y + i;    // i-th value of input0
        auto dy_idx = input_dy + i;  // i-th value of input1
        *(output_z + i) = (one_trans - (*y_idx)) * (*y_idx) * (*dy_idx);
      }
    };
    KERNEL_HANDLE_ERROR(
        CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num,
                                    shard_SigmoidGrad),
        "SigmoidGrad Compute failed.")
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kSigmoidGrad, SigmoidGradCpuKernel);
}  // namespace aicpu