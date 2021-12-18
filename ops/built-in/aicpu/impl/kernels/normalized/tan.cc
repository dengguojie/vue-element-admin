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

#include "tan.h"

#include <complex>
#include <unsupported/Eigen/CXX11/Tensor>
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const std::uint32_t kTanInputNum{1};
const std::uint32_t kTanOutputNum{1};
const std::int64_t ParallelNum{1024};
const char *kTan{"Tan"};
}  // namespace

namespace aicpu {
namespace detail {
template <typename T>
inline std::uint32_t ComputeTanKernel(const CpuKernelContext &ctx) {
  using i64 = std::int64_t;
  const auto ParallelFor = aicpu::CpuKernelUtils::ParallelFor;
  auto input = static_cast<T *>(ctx.Input(0)->GetData());
  auto output = static_cast<T *>(ctx.Output(0)->GetData());
  i64 total = ctx.Input(0)->NumElements();
  uint32_t cores = aicpu::CpuKernelUtils::GetCPUNum(ctx);
  if (total > ParallelNum) {
    i64 per_unit_size{total / std::min(std::max(1L, cores - 2L), total)};
    return ParallelFor(ctx, total, per_unit_size, [&](i64 begin, i64 end) {
      std::transform(input + begin, input + end, output + begin, Eigen::numext::tan<T>);
    });
  } else if (cores != 0) {
    std::transform(input, input + total, output, Eigen::numext::tan<T>);
  } else {
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
inline std::uint32_t ComputeTan(const CpuKernelContext &ctx) {
  uint32_t result = ComputeTanKernel<T>(ctx);
  if (result != 0) {
    KERNEL_LOG_ERROR("Tan compute failed.");
  }
  return result;
}

inline std::uint32_t TanExtraCheck(const CpuKernelContext &ctx) {
  if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
    KERNEL_LOG_ERROR(
        "The data type of the input [%s] need be the same as the ouput [%s].",
        DTypeStr(ctx.Input(0)->GetDataType()).c_str(),
        DTypeStr(ctx.Output(0)->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(0)->GetDataSize() != ctx.Output(0)->GetDataSize()) {
    KERNEL_LOG_ERROR(
        "The data size of the input [%llu] need be the same as the ouput "
        "[%llu].",
        ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(0)->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get input data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Output(0)->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get output data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  std::vector<int64_t> input_dims =
      ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> output_dims =
      ctx.Output(0)->GetTensorShape()->GetDimSizes();
  if (input_dims.size() != output_dims.size()) {
    KERNEL_LOG_ERROR(
        "The data dim size of the input [%llu] need be the same as the output "
        "[%llu].",
        input_dims.size(), output_dims.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  for (size_t index = 0; index < input_dims.size(); index++) {
    if (input_dims[index] != output_dims[index]) {
      KERNEL_LOG_ERROR(
          "The data dim of the input need be the same as the output.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  return KERNEL_STATUS_OK;
}

std::uint32_t TanCheck(CpuKernelContext &ctx, uint32_t inputs_num,
                       uint32_t outputs_num) {
  return NormalCheck(ctx, inputs_num, outputs_num)
             ? KERNEL_STATUS_PARAM_INVALID
             : TanExtraCheck(ctx);
}
// DT_FLOAT16, DT_FLOAT, DT_DOUBLE
// DT_COMPLEX64, DT_COMPLEX128
std::uint32_t TanCompute(const CpuKernelContext &ctx) {
  DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case DT_FLOAT16:
      return ComputeTan<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeTan<std::float_t>(ctx);
    case DT_DOUBLE:
      return ComputeTan<std::double_t>(ctx);
    case DT_COMPLEX64:
      return ComputeTan<std::complex<std::float_t> >(ctx);
    case DT_COMPLEX128:
      return ComputeTan<std::complex<std::double_t> >(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported input data type [%s].",
                       DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

std::uint32_t TanCpuKernel::Compute(CpuKernelContext &ctx) {
  return detail::TanCheck(ctx, kTanInputNum, kTanOutputNum)
             ? KERNEL_STATUS_PARAM_INVALID
             : detail::TanCompute(ctx);
}

REGISTER_CPU_KERNEL(kTan, TanCpuKernel);
}  // namespace aicpu
