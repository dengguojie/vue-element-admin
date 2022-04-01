/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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

#include "clip_by_value.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const std::uint32_t kClipByValueInputNum {3u};
const std::uint32_t kClipByValueOutputNum {1u};
const std::uint32_t inputIdx0 {0u};
const std::uint32_t inputIdx1 {1u};
const std::uint32_t inputIdx2 {2u};
const std::uint32_t outputIdx0 {0u};
const char *ClipByValue = "ClipByValue";
}
namespace aicpu {
namespace detail {
template <typename T>
std::uint32_t DoComputeClipByValue(const CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("DoComputeClipByValue::Start");
  auto input_x = reinterpret_cast<T*>(ctx.Input(inputIdx0)->GetData());
  auto input_min = reinterpret_cast<T*>(ctx.Input(inputIdx1)->GetData());
  auto input_max = reinterpret_cast<T*>(ctx.Input(inputIdx2)->GetData());
  auto output_y = reinterpret_cast<T*>(ctx.Output(outputIdx0)->GetData());
  int64_t data_num = ctx.Output(outputIdx0)->NumElements();
  auto input_min_index = input_min;
  auto input_max_index = input_max;
  bool isMulNum = false;

  if (ctx.Input(inputIdx1)->NumElements() > 1) {
    isMulNum = true;
  }

  KERNEL_LOG_INFO("data_num : [%llu], ctx.Input(1)->NumElements():[%llu]", data_num, ctx.Input(inputIdx1)->NumElements());
  for (int64_t i = 0; i < data_num; i++) {
    auto x_index = input_x + i;
    auto y_index = output_y + i;

    if (isMulNum) {
      input_min_index = input_min + i;
      input_max_index = input_max + i;
    }

    KERNEL_LOG_INFO("*x_index : [%lf], *input_min_index : [%lf], *input_max_index : [%lf]",
        (static_cast<double>(*x_index)), (static_cast<double>(*input_min_index)),
        (static_cast<double>(*input_max_index)));
    if (*x_index < *input_min_index) {
      *y_index = *input_min_index;
    } else if (*x_index > *input_max_index) {
      *y_index = *input_max_index;
    } else {
      *y_index = *x_index;
    }
    KERNEL_LOG_INFO("*x_index : [%lf], *y_index : [%lf]", (static_cast<double>(*x_index)), (static_cast<double>(*y_index)));
  }

  KERNEL_LOG_INFO("DoComputeClip::Stop");
  return KERNEL_STATUS_OK;
}

std::uint32_t ExtraCheckClipByValue(const CpuKernelContext &ctx) {
  for (int64_t i = 0; i < kClipByValueInputNum; i++) {
    if (ctx.Input(i)->GetData() == nullptr) {
      KERNEL_LOG_ERROR("ctx.Input(%llu)->GetData() == nullptr", i);
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  if (ctx.Output(outputIdx0)->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get output data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (ctx.Input(inputIdx0)->GetDataType() != ctx.Output(outputIdx0)->GetDataType()) {
    KERNEL_LOG_ERROR(
        "The data type of the input [%s] need be the same as the ouput [%s].",
        DTypeStr(ctx.Input(inputIdx0)->GetDataType()).c_str(),
        DTypeStr(ctx.Output(outputIdx0)->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  for (int64_t i = 1; i < kClipByValueInputNum; i++) {
    if (ctx.Input(inputIdx0)->GetDataType() != ctx.Input(i)->GetDataType()) {
      KERNEL_LOG_ERROR(
          "The data type of the input [%s] need be the same [%s].",
          DTypeStr(ctx.Input(i)->GetDataType()).c_str(),
          DTypeStr(ctx.Input(inputIdx0)->GetDataType()).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  if (ctx.Input(inputIdx0)->GetDataSize() != ctx.Output(outputIdx0)->GetDataSize()) {
    KERNEL_LOG_ERROR(
        "The data size of the input [%llu] need be the same as the ouput [%llu].",
        ctx.Input(inputIdx0)->GetDataSize(), ctx.Output(outputIdx0)->GetDataSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  KERNEL_LOG_INFO("NumElements[0] : [%llu], NumElements[1] : [%llu], NumElements[2] : [%llu]",
      ctx.Input(inputIdx0)->NumElements(), ctx.Input(inputIdx1)->NumElements(),
      ctx.Input(inputIdx2)->NumElements());

  if ((ctx.Input(inputIdx1)->NumElements() == 1) && (ctx.Input(inputIdx2)->NumElements() == 1)) {
    return KERNEL_STATUS_OK;
  }

  if ((ctx.Input(inputIdx1)->NumElements() == ctx.Input(inputIdx2)->NumElements()) &&
      (ctx.Input(inputIdx0)->NumElements() == ctx.Input(inputIdx1)->NumElements())) {
    return KERNEL_STATUS_OK;
  }

  return KERNEL_STATUS_PARAM_INVALID;
}

std::uint32_t CheckClipByValue(CpuKernelContext &ctx, std::uint32_t inputs_num,
                               std::uint32_t outputs_num) {
  return NormalCheck(ctx, inputs_num, outputs_num)
             ? KERNEL_STATUS_PARAM_INVALID
             : ExtraCheckClipByValue(ctx);
}

std::uint32_t ComputeClipByValue(const CpuKernelContext &ctx) {
  DataType input_type {ctx.Input(inputIdx0)->GetDataType()};
  switch (input_type) {
    case DT_FLOAT:
      return DoComputeClipByValue<float>(ctx);
    case DT_DOUBLE:
      return DoComputeClipByValue<double>(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported input data type [%s].", DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail


std::uint32_t ClipByValueCpuKernel::Compute(CpuKernelContext &ctx)  {
  return detail::CheckClipByValue(ctx, kClipByValueInputNum, kClipByValueOutputNum)
             ? KERNEL_STATUS_PARAM_INVALID
             : detail::ComputeClipByValue(ctx);
}

REGISTER_CPU_KERNEL(ClipByValue, ClipByValueCpuKernel);
}  // namespace aicpu