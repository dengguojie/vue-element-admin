/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef _AICPU_AICPU_DEVICE_CPU_KERNELS_UTILS_MATH_UTIL_H_
#define _AICPU_AICPU_DEVICE_CPU_KERNELS_UTILS_MATH_UTIL_H_

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "log.h"
#include "status.h"
#include "cpu_context.h"

namespace aicpu {
const uint32_t kThreadNum = 32;
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 1;
const uint32_t kFirstInputIndex = 0;
const uint32_t kSecondInputIndex = 1;
const uint32_t kFirstOutputIndex = 0;

// attr name
const std::string ATTR_NAME_DTYPE = "dtype";
const std::string ATTR_NAME_RANDOM_UNIFORM_SEED = "seed";
const std::string ATTR_NAME_RANDOM_UNIFORM_SEED2 = "seed2";

/// @ingroup math_util
/// @brief normal check for calculation
/// @param [in] ctx  context
/// @return uint32_t
inline uint32_t NormalCheck(CpuKernelContext &ctx) {
  if ((ctx.GetInputsSize() != kInputNum) || (ctx.GetOutputsSize() != kOutputNum)) {
    KERNEL_LOG_ERROR("Unexpected %s node, input size: %u, output size: %u",
                     ctx.GetOpType().c_str(), ctx.GetInputsSize(), ctx.GetOutputsSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  Tensor *input_0 = ctx.Input(kFirstInputIndex);
  KERNEL_CHECK_NULLPTR(input_0, KERNEL_STATUS_PARAM_INVALID, "Get input 0 failed")
  Tensor *input_1 = ctx.Input(kSecondInputIndex);
  KERNEL_CHECK_NULLPTR(input_1, KERNEL_STATUS_PARAM_INVALID, "Get input 1 failed")

  if (input_0->GetDataType() != input_1->GetDataType()) {
    KERNEL_LOG_ERROR("Data type of inputs for %s node not matched, data_type_0:%d, data_type_1:%d",
                     ctx.GetOpType().c_str(), input_0->GetDataType(), input_1->GetDataType());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if ((input_0->GetDataSize() == 0) || (input_1->GetDataSize() == 0)) {
    KERNEL_LOG_ERROR("Data size of input0 is %llu, input1 is %llu.", input_0->GetDataSize(), input_1->GetDataSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  Tensor *output = ctx.Output(kFirstOutputIndex);
  KERNEL_CHECK_NULLPTR(output, KERNEL_STATUS_PARAM_INVALID, "Get output failed")

  return KERNEL_STATUS_OK;
}
}  // namespace aicpu
#endif  // _AICPU_AICPU_DEVICE_CPU_KERNELS_UTILS_MATH_UTIL_H_
