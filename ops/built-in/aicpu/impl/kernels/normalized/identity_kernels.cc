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

#include "identity_kernels.h"
#include <securec.h>
#include <map>
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/sparse_tensor.h"

namespace {
const char *IDENTITY = "Identity";
}

namespace aicpu {

uint32_t IdentityCpuKernel::DoCompute() {
  if (inputs_.size() == 0 || outputs_.size() == 0) {
    KERNEL_LOG_ERROR("IdentityCpuKernel::IdentityTask: input or output is empty.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // init
  void *input = (void *)inputs_[0]->GetData();
  void *output = (void *)outputs_[0]->GetData();
  if (input == NULL || output == NULL) {
    KERNEL_LOG_ERROR("IdentityCpuKernel::IdentityTask: input or output is NULL.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int64_t data_size = outputs_[0]->GetDataSize();
  int ret = memcpy_s(output, data_size, input, data_size);
  if (ret < 0) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

uint32_t IdentityCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("IdentityCpuKernel::GetInputAndCheck start!");
  // get input Tensors
  const int num_input = 1;
  for (int i = 0; i < num_input; ++i) {
    Tensor *tensor = ctx.Input(i);
    if (tensor == nullptr) {
      KERNEL_LOG_ERROR(
          "IdentityCpuKernel::GetInputAndCheck: get input tensor[%d] failed", i);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    inputs_.push_back(tensor);
  }
  // get output Tensors
  const int num_output = 1;
  for (int i = 0; i < num_output; ++i) {
    Tensor *tensor = ctx.Output(i);
    if (tensor == nullptr) {
      KERNEL_LOG_ERROR(
          "IdentityCpuKernel::GetInputAndCheck: get output tensor[%d] failed", i);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    outputs_.push_back(tensor);
  }
  KERNEL_LOG_INFO("IdentityCpuKernel::GetInputAndCheck success!");
  return KERNEL_STATUS_OK;
}

uint32_t IdentityCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("IdentityCpuKernel::Compute start!!");

  uint32_t res = GetInputAndCheck(ctx);
  if (res != KERNEL_STATUS_OK) {
    return res;
  }

  res = DoCompute();
  if (res != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("IdentityCpuKernel::Compute failed");
    return res;
  }

  KERNEL_LOG_INFO("IdentityCpuKernel::Compute success!!");
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(IDENTITY, IdentityCpuKernel);
}  // namespace aicpu
