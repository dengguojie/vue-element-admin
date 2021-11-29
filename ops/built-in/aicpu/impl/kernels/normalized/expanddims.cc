/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
#include "expanddims.h"

#include <securec.h>
#include "log.h"
#include "status.h"

namespace {
const char *kExpandDims = "ExpandDims";
const size_t kExpandDimsOutputDescNum = 1;
const size_t kExpandDimsInputNum = 2;
}

namespace aicpu {
uint32_t ExpandDimsCpuKernel::Compute(CpuKernelContext &ctx) {
  if ((ctx.GetInputsSize() != kExpandDimsInputNum) ||
      (ctx.GetOutputsSize() != kExpandDimsOutputDescNum)) {
    KERNEL_LOG_WARN(
        "Unexpected ExpandDims node, node input size[%zu], node output size"
        "[%zu], node name[%s]",
        ctx.GetInputsSize(), ctx.GetOutputsSize(), ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  Tensor *output = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(output, KERNEL_STATUS_PARAM_INVALID,
                       "Get output[0] failed")

  uint8_t *output_data = reinterpret_cast<uint8_t *>(output->GetData());
  KERNEL_CHECK_NULLPTR(output_data, KERNEL_STATUS_PARAM_INVALID,
                       "Get output[0] data failed")
  uint64_t output_data_size = output->GetDataSize();
  // print output tensor information, and will be deleted
  KERNEL_LOG_INFO("ExpandDims op[%s] output tensor data size is [%llu]",
                  ctx.GetOpType().c_str(), output_data_size);
  auto shape = output->GetTensorShape();
  KERNEL_CHECK_NULLPTR(shape, KERNEL_STATUS_PARAM_INVALID,
                       "Get tensor shape failed")

  size_t data_dim_size = shape->GetDims();
  KERNEL_LOG_INFO("ExpandDims op[%s] output tensor dim size is [%zu]",
                  ctx.GetOpType().c_str(), data_dim_size);

  Tensor *input = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(input, KERNEL_STATUS_PARAM_INVALID,
                       "Get input[0] failed")
  uint8_t *input_data = reinterpret_cast<uint8_t *>(input->GetData());
  uint64_t input_data_size = input->GetDataSize();
  KERNEL_LOG_INFO("ExpandDims op[%s] input tensor input_size is [%zu]",
                  ctx.GetOpType().c_str(), input_data_size);

  if (output_data_size != input_data_size) {
    KERNEL_LOG_ERROR(
        "Input data size[%llu] is not equal to output data size[%llu].",
        input_data_size, output_data_size);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  for (uint64_t i = 0; i < input_data_size; i++) {
    output_data[i] = input_data[i];
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kExpandDims, ExpandDimsCpuKernel);
}  // namespace aicpu
