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

#include <algorithm>

namespace aicpu {
const std::map<std::string, DataType> dtype_maps {
  { "DT_FLOAT" , DT_FLOAT },
  { "DT_FLOAT16" , DT_FLOAT16 },
  { "DT_INT8" , DT_INT8 },
  { "DT_INT16" , DT_INT16 },
  { "DT_UINT16" , DT_UINT16 },
  { "DT_UINT8" , DT_UINT8 },
  { "DT_INT32" , DT_INT32 },
  { "DT_INT64" , DT_INT64 },
  { "DT_UINT32" , DT_UINT32 },
  { "DT_UINT64" , DT_UINT64 },
  { "DT_BOOL" , DT_BOOL },
  { "DT_DOUBLE" , DT_DOUBLE },
  { "DT_STRING" , DT_STRING },
  { "DT_DUAL_SUB_INT8" , DT_DUAL_SUB_INT8 },
  { "DT_DUAL_SUB_UINT8" , DT_DUAL_SUB_UINT8 },
  { "DT_COMPLEX64" , DT_COMPLEX64 },
  { "DT_COMPLEX128" , DT_COMPLEX128 },
  { "DT_QINT8" , DT_QINT8 },
  { "DT_QINT16" , DT_QINT16 },
  { "DT_QINT32" , DT_QINT32 },
  { "DT_QUINT8" , DT_QUINT8 },
  { "DT_QUINT16" , DT_QUINT16 },
  { "DT_RESOURCE" , DT_RESOURCE },
  { "DT_STRING_REF" , DT_STRING_REF },
  { "DT_DUAL" , DT_DUAL },
  { "DT_UNDEFINED" , DT_UNDEFINED }
};

uint32_t NormalMathCheck(CpuKernelContext &ctx) {
  const uint32_t kInputNum = 2;
  const uint32_t kOutputNum = 1;

  if ((ctx.GetInputsSize() != kInputNum) ||
      (ctx.GetOutputsSize() != kOutputNum)) {
    KERNEL_LOG_ERROR("[%s] unexpected node, input size [%u], output size [%u]",
                     ctx.GetOpType().c_str(), ctx.GetInputsSize(),
                     ctx.GetOutputsSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  Tensor *input_0 = ctx.Input(kFirstInputIndex);
  KERNEL_CHECK_NULLPTR(input_0, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] Get input[0] failed", ctx.GetOpType().c_str());
  Tensor *input_1 = ctx.Input(kSecondInputIndex);
  KERNEL_CHECK_NULLPTR(input_1, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] Get input[1] failed", ctx.GetOpType().c_str());

  if (input_0->GetDataType() != input_1->GetDataType()) {
    KERNEL_LOG_ERROR(
        "[%s] dtype of inputs not matched, data_type[0] [%d], "
        "data_type[1] [%d]",
        ctx.GetOpType().c_str(), input_0->GetDataType(),
        input_1->GetDataType());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if ((input_0->GetDataSize() == 0) || (input_1->GetDataSize() == 0)) {
    KERNEL_LOG_ERROR("[%s] data size of input[0] [%llu], input[1] [%llu].",
                     ctx.GetOpType().c_str(), input_0->GetDataSize(),
                     input_1->GetDataSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  Tensor *output = ctx.Output(kFirstOutputIndex);
  KERNEL_CHECK_NULLPTR(output, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get output failed", ctx.GetOpType().c_str());

  return KERNEL_STATUS_OK;
}

uint32_t NormalCheck(CpuKernelContext &ctx,
                     const uint32_t inputs_num,
                     const uint32_t outputs_num) {
  if (inputs_num != kDynamicInput) {
    KERNEL_CHECK_FALSE((ctx.GetInputsSize() == inputs_num),
                       KERNEL_STATUS_PARAM_INVALID,
                       "[%s] need [%u] inputs, but got [%u].",
                       ctx.GetOpType().c_str(),
                       inputs_num, ctx.GetInputsSize());
    for (uint32_t i = 0; i < inputs_num; ++i) {
      Tensor *input = ctx.Input(i);
      KERNEL_CHECK_NULLPTR(input, KERNEL_STATUS_INNER_ERROR,
                           "[%s] get input[%u] failed.",
                           ctx.GetOpType().c_str(), i);
    }
  }

  if (outputs_num != kDynamicOutput) {
    KERNEL_CHECK_FALSE((ctx.GetOutputsSize() == outputs_num),
                       KERNEL_STATUS_PARAM_INVALID,
                       "[%s] need [%u] outputs, but got [%u].",
                       ctx.GetOpType().c_str(),
                       outputs_num, ctx.GetOutputsSize());
    for (uint32_t i = 0; i < outputs_num; ++i) {
      Tensor *output = ctx.Output(i);
      KERNEL_CHECK_NULLPTR(output, KERNEL_STATUS_INNER_ERROR,
                           "[%s] get output[%u] failed.",
                           ctx.GetOpType().c_str(), i);
    }
  }
  return KERNEL_STATUS_OK;
}

DataType GetDataType(std::string dtype_str) {
  auto iter = dtype_maps.find(dtype_str);
  if (iter != dtype_maps.end()) {
    return iter->second;
  } else {
    return DT_UNDEFINED;
  }
}

std::string GetDataType(DataType dtype) {
  auto iter = std::find_if(dtype_maps.begin(), dtype_maps.end(),
      [dtype](const std::map<std::string, DataType>::value_type &kv) {
        return (kv.second == dtype);
      } );
  if (iter != dtype_maps.end()) {
    return iter->first;
  } else {
    return std::string("DT_UNDEFINED");
  }
}
}  // namespace aicpu
