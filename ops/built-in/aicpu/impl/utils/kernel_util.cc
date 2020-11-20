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

#include "kernel_util.h"

namespace aicpu {
uint32_t NormalMathCheck(CpuKernelContext &ctx) {
  const uint32_t kInputNum = 2;
  const uint32_t kOutputNum = 1;

  if ((ctx.GetInputsSize() != kInputNum) ||
      (ctx.GetOutputsSize() != kOutputNum)) {
    KERNEL_LOG_ERROR("Unexpected %s node, input size: %u, output size: %u",
                     ctx.GetOpType().c_str(), ctx.GetInputsSize(),
                     ctx.GetOutputsSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  Tensor *input_0 = ctx.Input(kFirstInputIndex);
  KERNEL_CHECK_NULLPTR(input_0, KERNEL_STATUS_PARAM_INVALID,
                       "Get input 0 failed")
  Tensor *input_1 = ctx.Input(kSecondInputIndex);
  KERNEL_CHECK_NULLPTR(input_1, KERNEL_STATUS_PARAM_INVALID,
                       "Get input 1 failed")

  if (input_0->GetDataType() != input_1->GetDataType()) {
    KERNEL_LOG_ERROR(
        "Data type of inputs for %s node not matched, data_type_0:%d, "
        "data_type_1:%d",
        ctx.GetOpType().c_str(), input_0->GetDataType(),
        input_1->GetDataType());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if ((input_0->GetDataSize() == 0) || (input_1->GetDataSize() == 0)) {
    KERNEL_LOG_ERROR("Data size of input0 is %llu, input1 is %llu.",
                     input_0->GetDataSize(), input_1->GetDataSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  Tensor *output = ctx.Output(kFirstOutputIndex);
  KERNEL_CHECK_NULLPTR(output, KERNEL_STATUS_PARAM_INVALID, "Get output failed")

  return KERNEL_STATUS_OK;
}

uint32_t NormalCheck(CpuKernelContext &ctx,
                            const uint32_t inputs_nums,
                            const uint32_t outputs_nums) {
  if (inputs_nums != kDynamicInput) {
    KERNEL_CHECK_FALSE((ctx.GetInputsSize() != inputs_num),
                       KERNEL_STATUS_PARAM_INVALID,
                       "%s need %u inputs, but got %u.",
                       inputs_num, ctx.GetInputsSize());
    for (uint32_t i = 0; i < inputs_num; ++i) {
      Tensor *input = ctx.Input(i);
      KERNEL_CHECK_NULLPTR(input, KERNEL_STATUS_INNER_ERROR,
                           "%s get input:%u failed.", i);
    }
  }

  if (outputs_nums != kDynamicOutput) {
    KERNEL_CHECK_FALSE((ctx.GetOutputsSize() != outputs_num),
                       KERNEL_STATUS_PARAM_INVALID,
                       "%s need %u outputs, but got %u.",
                       outputs_num, ctx.GetOutputsSize());
    for (uint32_t i = 0; i < outputs_num; ++i) {
      Tensor *output = ctx.Output(i);
      KERNEL_CHECK_NULLPTR(output, KERNEL_STATUS_INNER_ERROR,
                           "%s get output:%u failed.", i);
    }
  }
  return KERNEL_STATUS_OK;
}
}  // namespace aicpu
