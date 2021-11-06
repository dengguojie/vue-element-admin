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
#include "no_op.h"
#include "log.h"
#include "status.h"

namespace {
const char *kNoOp = "NoOp";
}

namespace aicpu {
uint32_t NoOpCpuKernel::Compute(CpuKernelContext &ctx) {
    KERNEL_LOG_INFO("[%s] no need to compute.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kNoOp, NoOpCpuKernel);
}  // namespace aicpu