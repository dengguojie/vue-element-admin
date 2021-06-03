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
#ifndef AICPU_FOLDING_FOLDING_
#define AICPU_FOLDING_FOLDING_

#include <stdint.h>

#include <map>
#include <string>

#include "graph/utils/op_desc_utils.h"
#include "register/host_cpu_context.h"
#include "register/register.h"

extern "C" {
__attribute__((visibility("default"))) int32_t InitCpuConstantFolding(
    ge::HostCpuOp *(*create_fn)());

__attribute__((visibility("default"))) int32_t CpuConstantFoldingCompute(
    ge::Operator &op, const std::map<std::string, const ge::Tensor> &inputs,
    std::map<std::string, ge::Tensor> outputs);

__attribute__((visibility("default"))) uint32_t RunHostCpuKernel(void *param);
}
#endif  // AICPU_FOLDING_FOLDING_
