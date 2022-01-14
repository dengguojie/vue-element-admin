/**
 * Copyright 2021 Huawei Technologies Co., Ltd.
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

#include "softplus.h"

#include <complex>
#include <unsupported/Eigen/CXX11/Tensor>
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const std::uint32_t kSoftplusInputNum{1};
const std::uint32_t kSoftplusOutputNum{1};
const std::uint32_t ParallelNum{20480};
const std::uint32_t DoubleParallelNums{8196};
const char *kSoftplus{"Softplus"};
}  // namespace

namespace aicpu {
namespace detail {
template <typename T>
inline std::uint32_t ComputeSoftplusKernel(const CpuKernelContext &ctx,
                                           DataType &input_type) {
  const auto ParallelFor = aicpu::CpuKernelUtils::ParallelFor;
  T *input = (T *)(ctx.Input(0)->GetData());
  T *output = (T *)(ctx.Output(0)->GetData());
  std::int64_t total = ctx.Input(0)->NumElements();
  std::uint64_t total_size = ctx.Input(0)->GetDataSize();
  uint32_t cores = aicpu::CpuKernelUtils::GetCPUNum(ctx);
  bool parallel_flag = false;
  switch (input_type) {
    case DT_DOUBLE: {
      if (total_size > DoubleParallelNums * sizeof(T)) {
        parallel_flag = true;
      }
    }
    default: {
      if (total_size > ParallelNum * sizeof(T)) {
        parallel_flag = true;
      }
    }
  }
  if (parallel_flag) {
    int64_t per_unit_size{total / std::min(std::max(1L, cores - 2L), total)};
    return ParallelFor(
        ctx, total, per_unit_size, [&](int64_t begin, int64_t end) {
          int64_t length = end - begin;
          Eigen::TensorMap<Eigen::Tensor<T, 1>> tensor_x(input + begin, length);
          Eigen::TensorMap<Eigen::Tensor<T, 1>> tensor_y(output + begin,
                                                         length);
          static const T threshold =
              Eigen::numext::log(Eigen::NumTraits<T>::epsilon()) + T(2);
          auto too_large = tensor_x > tensor_x.constant(-threshold);
          auto too_small = tensor_x < tensor_x.constant(threshold);
          auto tensor_x_exp = tensor_x.exp();
          tensor_y = too_large.select(
              tensor_x, too_small.select(tensor_x_exp, tensor_x_exp.log1p()));
        });
  } else if (cores != 0) {
    Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Aligned> tensor_x(input,
                                                                   total);
    Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Aligned> tensor_y(output,
                                                                   total);
    static const T threshold =
        Eigen::numext::log(Eigen::NumTraits<T>::epsilon()) + T(2);
    auto too_large = tensor_x > tensor_x.constant(-threshold);
    auto too_small = tensor_x < tensor_x.constant(threshold);
    auto features_exp = tensor_x.exp();
    tensor_y = too_large.select(
        tensor_x,                       // softplus(x) ~= x for x large
        too_small.select(features_exp,  // softplus(x) ~= exp(x) for x small
                         features_exp.log1p()));
  } else {
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
inline std::uint32_t ComputeSoftplus(const CpuKernelContext &ctx,
                                     DataType &input_type) {
  uint32_t result = ComputeSoftplusKernel<T>(ctx, input_type);
  if (result != 0) {
    KERNEL_LOG_ERROR("Softplus compute failed.");
  }
  return result;
}

inline std::uint32_t SoftplusExtraCheck(const CpuKernelContext &ctx) {
  if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
    KERNEL_LOG_ERROR(
        "The data type of the input [%s] need be the same as the ouput [%s].",
        DTypeStr(ctx.Input(0)->GetDataType()).c_str(),
        DTypeStr(ctx.Output(0)->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  KERNEL_CHECK_NULLPTR(ctx.Input(0)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input data failed.")
  KERNEL_CHECK_NULLPTR(ctx.Output(0)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get output data failed.")
  std::vector<int64_t> input_dims =
      ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> output_dims =
      ctx.Output(0)->GetTensorShape()->GetDimSizes();
  if (input_dims.size() != output_dims.size()) {
    KERNEL_LOG_ERROR(
        "The data dim of the input size [%llu] need be the same as the output "
        "size [%llu].",
        input_dims.size(), output_dims.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  for (size_t index = 0; index < input_dims.size(); index++) {
    if (input_dims[index] != output_dims[index]) {
      KERNEL_LOG_ERROR(
          "The data dim[%llu]=%lld of the input need be the same as the output "
          "dim[%llu]=%lld.",
          index, input_dims[index], index, output_dims[index]);
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  return KERNEL_STATUS_OK;
}

std::uint32_t SoftplusCheck(CpuKernelContext &ctx, uint32_t inputs_num,
                            uint32_t outputs_num) {
  return NormalCheck(ctx, kSoftplusInputNum, kSoftplusOutputNum)
             ? KERNEL_STATUS_PARAM_INVALID
             : SoftplusExtraCheck(ctx);
}
std::uint32_t SoftplusCompute(const CpuKernelContext &ctx) {
  DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case DT_FLOAT16:
      return ComputeSoftplus<Eigen::half>(ctx, input_type);
    case DT_FLOAT:
      return ComputeSoftplus<std::float_t>(ctx, input_type);
    case DT_DOUBLE:
      return ComputeSoftplus<std::double_t>(ctx, input_type);
    default:
      KERNEL_LOG_ERROR("Unsupported input data type [%s].",
                       DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

std::uint32_t SoftplusCpuKernel::Compute(CpuKernelContext &ctx) {
  return detail::SoftplusCheck(ctx, kSoftplusInputNum, kSoftplusOutputNum)
             ? KERNEL_STATUS_PARAM_INVALID
             : detail::SoftplusCompute(ctx);
}

REGISTER_CPU_KERNEL(kSoftplus, SoftplusCpuKernel);
}  // namespace aicpu
