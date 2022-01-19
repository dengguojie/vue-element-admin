/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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

#include "environ_create.h"
#include "utils/environ_manager.h"

namespace {
const char *kEnvironCreate = "EnvironCreate";
}

namespace aicpu {
uint32_t EnvironCreateCpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *output_tensor = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(output_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "Get output[0] failed.");
  auto output_shape = output_tensor->GetTensorShape()->GetDimSizes();
  // Check the output handle.
  if (!EnvironMgr::GetInstance().IsScalarTensor(output_shape)) {
    KERNEL_LOG_ERROR("The output is not scalar tensor.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // Generate an unique handle.
  int64_t env_handle = EnvironMgr::GetInstance().Create();
  KERNEL_LOG_DEBUG("Create env handle:%d ", env_handle);
  auto *output_data = reinterpret_cast<int64_t *>(output_tensor->GetData());
  output_data[0] = env_handle;

  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kEnvironCreate, EnvironCreateCpuKernel);
}  // namespace aicpu
