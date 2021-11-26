/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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

#include "asin.h"

#include <unsupported/Eigen/CXX11/Tensor>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const std::uint32_t kAsinInputNum{1};
const std::uint32_t kAsinOutputNum{1};
const char *kAsin{"Asin"};
}  // namespace

namespace internal {
template <typename T>
inline T ScalarAsin(T x) {
  return std::asin(x);
}

template <>
inline Eigen::half ScalarAsin(Eigen::half x) {
  const Eigen::half val{
      static_cast<Eigen::half>(std::asin(static_cast<std::float_t>(x)))};
  return val;
}
}  // namespace internal

namespace aicpu {
namespace detail {
template <typename T>
inline std::uint32_t ComputeAsinKernel(const CpuKernelContext &ctx) {
  using i64 = std::int64_t;
  const auto ParallelFor = aicpu::CpuKernelUtils::ParallelFor;
  const auto ScalarAsin = internal::ScalarAsin<T>;
  auto input = static_cast<T *>(ctx.Input(0)->GetData());
  auto output = static_cast<T *>(ctx.Output(0)->GetData());
  i64 total = ctx.Input(0)->NumElements();
  uint32_t cores = aicpu::CpuKernelUtils::GetCPUNum(ctx);
  i64 per_unit_size{total / std::min(std::max(1L, cores - 2L), total)};
  return ParallelFor(ctx, total, per_unit_size, [&](i64 begin, i64 end) {
    std::transform(input + begin, input + end, output + begin, ScalarAsin);
  });
}

template <typename T>
inline std::uint32_t ComputeAsin(const CpuKernelContext &ctx) {
  uint32_t result = ComputeAsinKernel<T>(ctx);
  if (result != 0) {
    KERNEL_LOG_ERROR("Asin compute failed.");
  }
  return result;
}

inline std::uint32_t ExtraCheck(const CpuKernelContext &ctx) {
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
  return KERNEL_STATUS_OK;
}

std::uint32_t Check(CpuKernelContext &ctx, uint32_t inputs_num,
                    uint32_t outputs_num) {
  return NormalCheck(ctx, kAsinInputNum, kAsinOutputNum)
             ? KERNEL_STATUS_PARAM_INVALID
             : ExtraCheck(ctx);
}

std::uint32_t Compute(const CpuKernelContext &ctx) {
  DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case DT_FLOAT16:
      return ComputeAsin<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeAsin<std::float_t>(ctx);
    case DT_DOUBLE:
      return ComputeAsin<std::double_t>(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported input data type [%s].",
                       DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

std::uint32_t AsinCpuKernel::Compute(CpuKernelContext &ctx) {
  return detail::Check(ctx, kAsinInputNum, kAsinOutputNum)
             ? KERNEL_STATUS_PARAM_INVALID
             : detail::Compute(ctx);
}

REGISTER_CPU_KERNEL(kAsin, AsinCpuKernel);
}  // namespace aicpu
