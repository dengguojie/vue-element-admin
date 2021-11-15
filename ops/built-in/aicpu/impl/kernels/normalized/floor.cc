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
#include "floor.h"

#include <stdint.h>

#include "Eigen/Dense"
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "kernel_util.h"
#include "log.h"
#include "securec.h"
#include "status.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *kFloor = "Floor";

#define FLOOR_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                       \
    uint32_t result = FloorCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                   \
      KERNEL_LOG_ERROR("Floor kernel compute failed."); \
      return result;                                    \
    }                                                   \
    break;                                              \
  }
}  // namespace

namespace aicpu {
uint32_t FloorCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "[%s] check input and output failed.", kFloor);
  KERNEL_HANDLE_ERROR(FloorCheck(ctx), "[%s] check params failed.", kFloor);
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    FLOOR_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    FLOOR_COMPUTE_CASE(DT_FLOAT, float, ctx)
    FLOOR_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("Floor kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t FloorCpuKernel::FloorCheck(CpuKernelContext &ctx) {
  KERNEL_CHECK_NULLPTR(ctx.Input(0)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input data failed.")
  KERNEL_CHECK_NULLPTR(ctx.Output(0)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get output data failed.")
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t FloorCpuKernel::FloorCompute(CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t data_num = ctx.Output(0)->NumElements();
  for (int64_t i = 0; i < data_num; i++) {
    auto x_index = input_x + i;  // i-th value of input0
    *(output_y + i) = Eigen::numext::floor((*x_index));
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kFloor, FloorCpuKernel);
}  // namespace aicpu
