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
#include "assign.h"
#include <securec.h>
#include "log.h"
#include "status.h"

namespace {
constexpr char *kAssign = "Assign";
}

namespace aicpu {
uint32_t AssignCpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *ref_tensor = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(ref_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "Get input[0] failed.");
  void *ref_data = ref_tensor->GetData();
  KERNEL_CHECK_NULLPTR(ref_data, KERNEL_STATUS_PARAM_INVALID,
                       "Get input[0] data failed.");
  uint64_t ref_size = ref_tensor->GetDataSize();
  Tensor *value_tensor = ctx.Input(1);
  KERNEL_CHECK_NULLPTR(value_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "Get input[1] failed.");
  void *value_data = value_tensor->GetData();
  KERNEL_CHECK_NULLPTR(value_data, KERNEL_STATUS_PARAM_INVALID,
                       "Get input[1] data failed.");
  uint64_t value_size = value_tensor->GetDataSize();
  KERNEL_CHECK_FALSE((value_size == ref_size), KERNEL_STATUS_PARAM_INVALID,
                     "Input[0] datasize is not equal to input[1] data size,"
                     "data size of input[0] is [%llu], data size of input[1] is [%llu].",
                     ref_size, value_size);
  if (value_size > 0) {
    auto mem_ret = memcpy_s(ref_data, ref_size, value_data, value_size);
    KERNEL_CHECK_FALSE((mem_ret == EOK), KERNEL_STATUS_PARAM_INVALID,
                       "Memcpy size[%zu] from input[1] to input[0] failed.",
                       value_size);
  }
  Tensor *output_tensor = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(output_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "Get output[0] failed.");
  output_tensor->SetData(ref_data);
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kAssign, AssignCpuKernel);
}  // namespace aicpu