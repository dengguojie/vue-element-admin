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

#include "cosh.h"

#include <unsupported/Eigen/CXX11/Tensor>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const std::uint32_t kCoshInputNum{1};
const std::uint32_t kCoshOutputNum{1};
const char *kCosh{"Cosh"};
const std::int64_t kCoshParallelNum{64 * 1024};
}  // namespace

namespace aicpu {
namespace detail {
template <typename T>
inline T ScalarCosh(T x) {
  return std::cosh(x);
}

template <>
inline Eigen::half ScalarCosh(Eigen::half x) {
  const Eigen::half val {
      static_cast<Eigen::half>(std::cosh(static_cast<std::float_t>(x)))};
  return Eigen::half_impl::isnan(val) ? Eigen::half{0.0f} : val;
}

inline std::uint32_t ParallelForCosh(
    const CpuKernelContext &ctx, std::int64_t total, std::int64_t perUnitSize,
    const std::function<void(std::int64_t, std::int64_t)> &work) {
  if (total > kCoshParallelNum)
    return aicpu::CpuKernelUtils::ParallelFor(ctx, total, perUnitSize, work);
  else
    work(0, total);
  return KERNEL_STATUS_OK;
}
template <typename T>
inline std::uint32_t ComputeCoshKernel(const CpuKernelContext &ctx) {
  T *input0{static_cast<T *>(ctx.Input(0)->GetData())};
  T *output{static_cast<T *>(ctx.Output(0)->GetData())};
  std::int64_t total{ctx.Input(0)->NumElements()};
  std::uint32_t cores{aicpu::CpuKernelUtils::GetCPUNum(ctx)};
  std::int64_t per_unit_size{total / std::min(std::max(1L, cores - 2L), total)};
  return ParallelForCosh(ctx, total, per_unit_size,
                         [&](std::int64_t begin, std::int64_t end) {
                           std::transform(input0 + begin, input0 + end,
                                          output + begin, ScalarCosh<T>);
                         });
}
template <typename T>
inline std::uint32_t ComputeCosh(const CpuKernelContext &ctx) {
  std::uint32_t result{ComputeCoshKernel<T>(ctx)};
  if (result != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Cosh compute failed.");
  }
  return result;
}

inline std::uint32_t ExtraCheckCosh(const CpuKernelContext &ctx) {
  if (ctx.Input(0)->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get input data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Output(0)->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get output data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
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

inline std::uint32_t CheckCosh(CpuKernelContext &ctx, std::uint32_t inputs_num,
                               std::uint32_t outputs_num) {
  return NormalCheck(ctx, kCoshInputNum, kCoshOutputNum)
             ? KERNEL_STATUS_PARAM_INVALID
             : ExtraCheckCosh(ctx);
}

inline std::uint32_t ComputeCosh(const CpuKernelContext &ctx) {
  DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case DT_FLOAT16:
      return ComputeCosh<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeCosh<std::float_t>(ctx);
    case DT_DOUBLE:
      return ComputeCosh<std::double_t>(ctx);
    case DT_COMPLEX64:
      return ComputeCosh<std::complex<std::float_t>>(ctx);
    case DT_COMPLEX128:
      return ComputeCosh<std::complex<std::double_t>>(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported input data type [%s].",
                       DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

std::uint32_t CoshCpuKernel::Compute(CpuKernelContext &ctx) {
  return detail::CheckCosh(ctx, kCoshInputNum, kCoshOutputNum)
             ? KERNEL_STATUS_PARAM_INVALID
             : detail::ComputeCosh(ctx);
}

REGISTER_CPU_KERNEL(kCosh, CoshCpuKernel);
}  // namespace aicpu