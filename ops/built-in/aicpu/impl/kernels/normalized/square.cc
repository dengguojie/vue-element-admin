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

#include "square.h"

#include <complex>
#include <unsupported/Eigen/CXX11/Tensor>
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const std::uint32_t kSquareInputNum{1};
const std::uint32_t kSquareOutputNum{1};
// parallel 3 level
const std::int64_t Int32ParallelNums[3] = {96000, 192000, 384000};
const std::int64_t Int64ParallelNums[3] = {24000, 48000, 96000};
const std::int64_t Float16ParallelNums[3] = {24000, 48000, 96000};
const std::int64_t FloatParallelNums[3] = {192000, 384000, 512000};
const std::int64_t DoubleParallelNums[3] = {48000, 96000, 256000};
const std::int64_t ComplexParallelNums[3] = {24000, 96000, 192000};
const char *kSquare{"Square"};
}  // namespace

namespace aicpu {
namespace detail {
template <typename T>
inline std::uint32_t ComputeSquareKernel(const CpuKernelContext &ctx,
                                         DataType &input_type) {
  const auto ParallelFor = aicpu::CpuKernelUtils::ParallelFor;
  T *input = (T *)(ctx.Input(0)->GetData());
  T *output = (T *)(ctx.Output(0)->GetData());
  std::int64_t total = ctx.Input(0)->NumElements();
  std::uint64_t total_size = ctx.Input(0)->GetDataSize();
  uint32_t cores = aicpu::CpuKernelUtils::GetCPUNum(ctx);
  bool parallel_flag = false;
  std::int64_t ParallelNums[3];
  switch (input_type) {
    case DT_INT32: {
      memcpy(ParallelNums, Int32ParallelNums, 3 * sizeof(std::int64_t));
    }; break;
    case DT_INT64: {
      memcpy(ParallelNums, Int64ParallelNums, 3 * sizeof(std::int64_t));
    }; break;
    case DT_FLOAT16: {
      memcpy(ParallelNums, Float16ParallelNums, 3 * sizeof(std::int64_t));
    }; break;
    case DT_FLOAT: {
      memcpy(ParallelNums, FloatParallelNums, 3 * sizeof(std::int64_t));
    }; break;
    case DT_DOUBLE: {
      memcpy(ParallelNums, DoubleParallelNums, 3 * sizeof(std::int64_t));
    }; break;
    case DT_COMPLEX64: {
      memcpy(ParallelNums, ComplexParallelNums, 3 * sizeof(std::int64_t));
    }; break;
    case DT_COMPLEX128: {
      memcpy(ParallelNums, ComplexParallelNums, 3 * sizeof(std::int64_t));
    }; break;
  }
  if (total_size > ParallelNums[2] * sizeof(T)) {
    parallel_flag = true;
  } else if (total_size > ParallelNums[1] * sizeof(T)) {
    parallel_flag = true;
    cores = 8;
  } else if (total_size > ParallelNums[0] * sizeof(T)) {
    parallel_flag = true;
    cores = 6;
  }
  if (parallel_flag) {
    std::int64_t per_unit_size{total /
                               std::min(std::max(1L, cores - 2L), total)};
    if (input_type == DT_INT64 || input_type == DT_FLOAT16) {
      return ParallelFor(
          ctx, total, per_unit_size, [&](std::int64_t begin, std::int64_t end) {
            T *inner_input = input + begin;
            T *inner_output = output + begin;
            for (size_t index = begin; index < end; ++index) {
              *(inner_output) = (*(inner_input)) * (*(inner_input));
              inner_input++;
              inner_output++;
            }
          });
    } else {
      return ParallelFor(
          ctx, total, per_unit_size, [&](std::int64_t begin, std::int64_t end) {
            std::int64_t length = end - begin;
            Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Aligned> tensor_x(
                input + begin, length);
            Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Aligned> tensor_y(
                output + begin, length);
            tensor_y = tensor_x.square();
          });
    }
  } else if (cores != 0) {
    if (input_type == DT_INT64 || input_type == DT_FLOAT16) {
      for (size_t index = 0; index < total; ++index) {
        *(output) = (*(input)) * (*(input));
        input++;
        output++;
      }
    } else {
      Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Aligned> tensor_x(input,
                                                                     total);
      Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Aligned> tensor_y(output,
                                                                     total);
      tensor_y = tensor_x.square();
    }
  } else {
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
inline std::uint32_t ComputeSquare(const CpuKernelContext &ctx,
                                   DataType &input_type) {
  uint32_t result = ComputeSquareKernel<T>(ctx, input_type);
  if (result != 0) {
    KERNEL_LOG_ERROR("Square compute failed.");
  }
  return result;
}

inline std::uint32_t SquareExtraCheck(const CpuKernelContext &ctx) {
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

std::uint32_t SquareCheck(CpuKernelContext &ctx, uint32_t inputs_num,
                          uint32_t outputs_num) {
  return NormalCheck(ctx, kSquareInputNum, kSquareOutputNum)
             ? KERNEL_STATUS_PARAM_INVALID
             : SquareExtraCheck(ctx);
}
// DT_FLOAT16, DT_FLOAT, DT_DOUBLE,DT_INT32, DT_INT64, DT_COMPLEX64,
// DT_COMPLEX128
std::uint32_t SquareCompute(const CpuKernelContext &ctx) {
  DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case DT_INT32:
      return ComputeSquare<std::int32_t>(ctx, input_type);
    case DT_INT64:
      return ComputeSquare<std::int64_t>(ctx, input_type);
    case DT_FLOAT16:
      return ComputeSquare<Eigen::half>(ctx, input_type);
    case DT_FLOAT:
      return ComputeSquare<std::float_t>(ctx, input_type);
    case DT_DOUBLE:
      return ComputeSquare<std::double_t>(ctx, input_type);
    case DT_COMPLEX64:
      return ComputeSquare<std::complex<std::float_t> >(ctx, input_type);
    case DT_COMPLEX128:
      return ComputeSquare<std::complex<std::double_t> >(ctx, input_type);
    default:
      KERNEL_LOG_ERROR("Unsupported input data type [%s].",
                       DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

std::uint32_t SquareCpuKernel::Compute(CpuKernelContext &ctx) {
  return detail::SquareCheck(ctx, kSquareInputNum, kSquareOutputNum)
             ? KERNEL_STATUS_PARAM_INVALID
             : detail::SquareCompute(ctx);
}

REGISTER_CPU_KERNEL(kSquare, SquareCpuKernel);
}  // namespace aicpu
