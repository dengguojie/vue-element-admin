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
#include "elugrad.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *KEluGrad = "EluGrad";

#define ELUGRAD_COMPUTE_CASE(DTYPE, TYPE, CTX)           \
  case (DTYPE): {                                        \
    uint32_t result = EluGradCompute<TYPE>(CTX);         \
    if (result != KERNEL_STATUS_OK) {                    \
      KERNEL_LOG_ERROR("EluGrad kernel compute failed"); \
      return result;                                     \
    }                                                    \
    break;                                               \
  }
}  // namespace

namespace aicpu {
uint32_t EluGradCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "EluGrad check input and output number failed.");
  KERNEL_HANDLE_ERROR(EluGradCheck(ctx), "[%s] check params failed.", KEluGrad);
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    ELUGRAD_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    ELUGRAD_COMPUTE_CASE(DT_FLOAT, float, ctx)
    ELUGRAD_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("EluGrad kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
uint32_t EluGradCpuKernel::EluGradCheck(CpuKernelContext &ctx) {
  KERNEL_CHECK_NULLPTR(ctx.Input(0)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input 0 failed.")
  KERNEL_CHECK_NULLPTR(ctx.Input(1)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input 1 failed.")
  KERNEL_CHECK_NULLPTR(ctx.Output(0)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get output failed.")
  DataType input0_type = ctx.Input(0)->GetDataType();
  DataType input1_type = ctx.Input(1)->GetDataType();
  DataType output0_type = ctx.Output(0)->GetDataType();
  KERNEL_CHECK_FALSE(
      (input0_type == input1_type), KERNEL_STATUS_PARAM_INVALID,
      "The data type of input0 [%s] need be same with input1 [%s]",
      DTypeStr(input0_type).c_str(), DTypeStr(input1_type).c_str())
  KERNEL_CHECK_FALSE(
      (input0_type == output0_type), KERNEL_STATUS_PARAM_INVALID,
      "The data type of input0 [%s] need be same with output0 [%s]",
      DTypeStr(input0_type).c_str(), DTypeStr(output0_type).c_str())
  return KERNEL_STATUS_OK;
}
template <typename T>
uint32_t EluGradCpuKernel::EluGradCompute(CpuKernelContext &ctx) {
  auto input_0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto input_1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto output = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  auto data_type = ctx.Input(0)->GetDataType();
  int64_t data_num = ctx.Output(0)->NumElements();
  int64_t data_size = data_num * sizeof(T);
  auto num0 = static_cast<T>(0);
  auto num1 = static_cast<T>(1);
  if ((data_type == DT_FLOAT16 && data_size <= 128 * 1024) ||
      (data_type == DT_FLOAT && data_size <= 64 * 1024) ||
      (data_type == DT_DOUBLE && data_size <= 64 * 1024)) {
    for (int64_t index = 0; index < data_num; ++index) {
      *(output + index) =
          (*(input_1 + index)) < num0
              ? ((*(input_1 + index)) + num1) * (*(input_0 + index))
              : *(input_0 + index);
    }
  } else {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(
      min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto shard_elugrad = [&](size_t start, size_t end) {
      for (size_t index = start; index < end; ++index) {
        *(output + index) =
            (*(input_1 + index)) < num0
                ? ((*(input_1 + index)) + num1) * (*(input_0 + index))
                : *(input_0 + index);
      }
    };
    KERNEL_HANDLE_ERROR(
        CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num,
                                    shard_elugrad),
        "EluGrad Compute failed.")
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(KEluGrad, EluGradCpuKernel);
}  // namespace aicpu