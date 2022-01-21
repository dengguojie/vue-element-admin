/**
 * Copyright 2021 Jilin University
 * Copyright 2020 Huawei Technologies Co., Ltd.
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

#include "abs.h"

#include <unsupported/Eigen/CXX11/Tensor>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const std::uint32_t kAbsInputNum{1u};
const std::uint32_t kAbsOutputNum{1u};
const char *kAbs{"Abs"};
const std::int64_t kAbsParallelNum{512 * 1024};
}  // namespace

namespace aicpu {
namespace detail {
template <typename T>
inline T ScalarAbs(T x) {
  return std::abs(x);
}

template <>
inline Eigen::half ScalarAbs(Eigen::half x) {
  const Eigen::half val{
      static_cast<Eigen::half>(std::abs(static_cast<std::float_t>(x)))};
  return Eigen::half_impl::isnan(val) ? Eigen::half{0.0f} : val;
}

template <>
inline std::int32_t ScalarAbs(std::int32_t x) {
  std::int32_t y = x >> 31;
  return (x ^ y) - y;
}

template <>
inline std::int64_t ScalarAbs(std::int64_t x) {
  std::int64_t y = x >> 63;
  return (x ^ y) - y;
}

template <>
inline std::float_t ScalarAbs(std::float_t x) {
  *(std::uint32_t *)&x &= 0x7fffffff;
  return x;
}

template <>
inline std::double_t ScalarAbs(std::double_t x) {
  *(std::uint64_t *)&x &= ~(1ULL << 63);
  return x;
}

inline std::uint32_t ParallelForAbs(
    const CpuKernelContext &ctx, std::int64_t total, std::int64_t per_unit_size,
    const std::function<void(std::int64_t, std::int64_t)> &work) {
  if (total > kAbsParallelNum)
    return aicpu::CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, work);
  else
    work(0, total);
  return KERNEL_STATUS_OK;
}

template <typename T>
inline std::uint32_t ComputeAbsKernel(const CpuKernelContext &ctx) {
  T *input0{static_cast<T *>(ctx.Input(0)->GetData())};
  T *output{static_cast<T *>(ctx.Output(0)->GetData())};
  std::int64_t total{ctx.Input(0)->NumElements()};
  std::uint32_t cores{aicpu::CpuKernelUtils::GetCPUNum(ctx)};
  std::int64_t per_unit_size{total / std::min(std::max(1L, cores - 2L), total)};
  return ParallelForAbs(ctx, total, per_unit_size,
                        [&](std::int64_t begin, std::int64_t end) {
                          std::transform(input0 + begin, input0 + end,
                                         output + begin, ScalarAbs<T>);
                        });
}

template <typename T>
inline std::uint32_t ComputeAbs(const CpuKernelContext &ctx) {
  std::uint32_t result{ComputeAbsKernel<T>(ctx)};
  if (result != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Abs compute failed.");
  }
  return result;
}

inline std::uint32_t ExtraCheckAbs(const CpuKernelContext &ctx) {
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
  return KERNEL_STATUS_OK;
}

inline std::uint32_t CheckAbs(CpuKernelContext &ctx, std::uint32_t inputs_num,
                              std::uint32_t outputs_num) {
  return NormalCheck(ctx, kAbsInputNum, kAbsOutputNum)
             ? KERNEL_STATUS_PARAM_INVALID
             : ExtraCheckAbs(ctx);
}

inline std::uint32_t ComputeAbs(const CpuKernelContext &ctx) {
  DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case DT_INT8:
      return ComputeAbs<std::int8_t>(ctx);
    case DT_INT16:
      return ComputeAbs<std::int16_t>(ctx);
    case DT_INT32:
      return ComputeAbs<std::int32_t>(ctx);
    case DT_INT64:
      return ComputeAbs<std::int64_t>(ctx);
    case DT_FLOAT16:
      return ComputeAbs<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeAbs<std::float_t>(ctx);
    case DT_DOUBLE:
      return ComputeAbs<std::double_t>(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported input data type [%s].",
                       DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

std::uint32_t AbsCpuKernel::Compute(CpuKernelContext &ctx) {
  return detail::CheckAbs(ctx, kAbsInputNum, kAbsOutputNum)
             ? KERNEL_STATUS_PARAM_INVALID
             : detail::ComputeAbs(ctx);
}

REGISTER_CPU_KERNEL(kAbs, AbsCpuKernel);
}  // namespace aicpu