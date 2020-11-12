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

#include "reshape_kernels.h"

#include <securec.h>
#include "cpu_types.h"
#include "log.h"
#include "status.h"

namespace {
const char *RESHAPE = "Reshape";
}

namespace aicpu {
uint32_t ReshapeCpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *input_tensor = ctx.Input(0);
  if (input_tensor == nullptr) {
    KERNEL_LOG_ERROR("get input:0 failed");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  Tensor *output_tensor = ctx.Output(0);
  if (output_tensor == nullptr) {
    KERNEL_LOG_ERROR("get output:0 failed");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto input_data = input_tensor->GetData();
  auto output_data = output_tensor->GetData();
  uint64_t input_size = input_tensor->GetDataSize();
  uint64_t output_size = output_tensor->GetDataSize();
  if (input_size != output_size) {
    KERNEL_LOG_ERROR(
        "input data size:%lld must be equal to output data size:%lld.",
        input_size, output_size);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  int cpyRet = memcpy_s(output_data, output_size, input_data, input_size);
  if (cpyRet < 0) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(RESHAPE, ReshapeCpuKernel);
}  // namespace aicpu
