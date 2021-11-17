/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#include "not_equal.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "utils/equal_util.h"
namespace {
constexpr uint32_t kOutputNum = 1;
constexpr uint32_t kInputNum = 2;
const char *kNotEqual = "NotEqual";
const bool kFlag = false;

#define EQUAL_COMPUTE_CASE(DTYPE, TYPE, CTX, Flag)                                \
  case (DTYPE): {                                                                 \
    uint32_t result = EqualCompute<TYPE>(CTX, Flag);                              \
    if (result != KERNEL_STATUS_OK) {                                             \
      KERNEL_LOG_ERROR("NotEqual kernel compute failed, result = [%d].", result); \
      return result;                                                              \
    }                                                                             \
    break;                                                                        \
  }
}

namespace aicpu {
uint32_t NotEqualCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Check NotEqual params failed.");

  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    EQUAL_COMPUTE_CASE(DT_INT8, int8_t, ctx, kFlag)
    EQUAL_COMPUTE_CASE(DT_INT16, int16_t, ctx, kFlag)
    EQUAL_COMPUTE_CASE(DT_INT32, int32_t, ctx, kFlag)
    EQUAL_COMPUTE_CASE(DT_INT64, int64_t, ctx, kFlag)
    EQUAL_COMPUTE_CASE(DT_UINT8, uint8_t, ctx, kFlag)
    EQUAL_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx, kFlag)
    EQUAL_COMPUTE_CASE(DT_FLOAT, float, ctx, kFlag)
    EQUAL_COMPUTE_CASE(DT_DOUBLE, double, ctx, kFlag)
    EQUAL_COMPUTE_CASE(DT_BOOL, bool, ctx, kFlag)
    default:
      KERNEL_LOG_WARN("NotEqual kernel data type [%u] not support.", data_type);
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kNotEqual, NotEqualCpuKernel);
}  // namespace aicpu