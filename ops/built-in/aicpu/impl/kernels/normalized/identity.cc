/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021.All rights reserved.
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

#include "identity.h"

#include "securec.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kIdentityInputNum = 1;
constexpr uint32_t kIdentityOutputNum = 1;
const char *kIdentity = "Identity";
}

namespace aicpu {
uint32_t IdentityCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kIdentityInputNum, kIdentityOutputNum),
                      "[%s] check params failed.", kIdentity);

  // parse params
  void *input_data = ctx.Input(0)->GetData();
  KERNEL_CHECK_NULLPTR(input_data, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get input_data[0] failed.", kIdentity);
  void *output_data = ctx.Output(0)->GetData();
  KERNEL_CHECK_NULLPTR(output_data, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get output_data[0] failed.", kIdentity);
  int64_t input_size = ctx.Input(0)->GetDataSize();
  int64_t output_size = ctx.Output(0)->GetDataSize();
  if (output_size < input_size) {
    KERNEL_LOG_WARN("[%s] output size [%ld] less than input size [%ld].",
                    kIdentity, output_size, input_size);
    input_size = output_size;
  } else if (output_size > input_size) {
    KERNEL_LOG_ERROR("[%s] output size [%ld] greater than input size [%ld].",
                     kIdentity, output_size, input_size);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // do copy
  if (output_data != input_data) {
    int cpret = memcpy_s(output_data, output_size, input_data, input_size);
    KERNEL_CHECK_FALSE(
        (cpret == EOK), KERNEL_STATUS_INNER_ERROR,
        "[%s] memcpy_s to output failed, destMax [%ld], count [%ld].",
        kIdentity, output_size, input_size);
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kIdentity, IdentityCpuKernel);
}  // namespace aicpu
