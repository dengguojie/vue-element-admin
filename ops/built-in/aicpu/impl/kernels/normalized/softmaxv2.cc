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
#define EIGEN_USE_THREADS
#define EIGEN_USE_SIMPLE_THREAD_POOL
#include "softmaxv2.h"

#include <complex>
#include <cstring>
#include <unsupported/Eigen/CXX11/Tensor>
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"
#include "securec.h"

namespace {
const std::uint32_t kSoftmaxV2InputNum{1};
const std::uint32_t kSoftmaxV2OutputNum{1};
const std::int64_t paralled_data_num{2048};
const char *const kSoftmaxV2 = "SoftmaxV2";
}  // namespace

namespace aicpu {
namespace detail {
template <typename T>
inline std::uint32_t ComputeSoftmaxV2Kernel(const CpuKernelContext &ctx) {
  auto input = static_cast<T *>(ctx.Input(0)->GetData());
  auto output = static_cast<T *>(ctx.Output(0)->GetData());
  // axes default values = [-1]
  std::vector<std::int64_t> axes{-1};
  if (ctx.GetAttr("axes") != nullptr) {
    axes = ctx.GetAttr("axes")->GetListInt();
  }
  std::int64_t total = ctx.Input(0)->NumElements();
  std::vector<std::int64_t> dims = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  uint32_t cores = aicpu::CpuKernelUtils::GetCPUNum(ctx);
  if (cores < 1) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  int64_t dim_size = static_cast<int64_t>(dims.size());
  // pivot is the axes value
  std::int64_t pivot, inner_size{1}, outer_size{1}, length{1};
  pivot = (axes[0] >= 0 ? axes[0] : dim_size + axes[0]);
  for (int64_t index = 0; index < dim_size; index++) {
    if (index > pivot) {
      inner_size *= static_cast<std::int64_t>(dims[index]);
    }
    if (index < pivot) {
      outer_size *= static_cast<int64_t>(dims[index]);
    }
  }
  length = inner_size * outer_size;
  T dims_exp_sum[length];
  T dims_maximum[length];
  (void)memset_s(dims_exp_sum, length * sizeof(T), 0, length * sizeof(T));
  bool parallel_flag = false;
  if (total > paralled_data_num) {
    parallel_flag = true;
  }
  // Note: the shape of Eigen::Tensor logits and softmax is reverse of input
  // Tensor
  if (!parallel_flag) {
    Eigen::TensorMap<Eigen::Tensor<T, 3>, Eigen::Aligned> logits(input, inner_size, (int)dims[pivot], outer_size);
    Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Aligned> dims_sum(dims_exp_sum, length);
    Eigen::TensorMap<Eigen::Tensor<T, 2>, Eigen::Aligned> dims_max(dims_maximum, inner_size, outer_size);
    Eigen::array<int, 1> softmax_axes{{1}};
    dims_max = logits.maximum(softmax_axes);
    for (int64_t index = 0, index_dst = 0, index_batch = 0, count_step = 0; index < total; index++) {
      if (index % inner_size == 0 && index != 0) {
        count_step++;
        if (count_step == static_cast<int64_t>(dims[pivot])) {
          count_step = 0;
          index_batch += inner_size;
        }
        index_dst = index_batch;
      }
      *(output + index) = Eigen::numext::exp(*(input + index) - dims_maximum[index_dst]);
      dims_exp_sum[index_dst] += (*(output + index));
      index_dst++;
    }
    dims_sum = dims_sum.inverse();
    for (int64_t index = 0, index_dst = 0, index_batch = 0, count_step = 0;
         index < total; index++) {
      if (index % inner_size == 0 && index != 0) {
        count_step++;
        if (count_step == dims[pivot]) {
          count_step = 0;
          index_batch += inner_size;
        }
        index_dst = index_batch;
      }
      *(output + index) = (*(output + index)) * (dims_exp_sum[index_dst]);
      index_dst++;
    }
  } else {
    std::int64_t per_unit_size{length / std::min(std::max(1L, cores - 2L), length)};
    const T constant_one(1.0);
    KERNEL_HANDLE_ERROR(aicpu::CpuKernelUtils::ParallelFor(
        ctx, length, per_unit_size, [&](std::int64_t begin, std::int64_t end) {
          int64_t tmp = static_cast<int64_t>(dims[pivot]);
          for (int64_t index = begin, dim_length = tmp, outer_index, index_base; index < end; ++index) {
            outer_index = index / inner_size;
            index_base = outer_index * dim_length * inner_size + index % inner_size;
            dims_maximum[index] = *(input + index_base);
            for (int64_t inner_index = 0, index_dst = index_base;
                 inner_index < dim_length; ++inner_index) {
              if (*(input + index_dst) > dims_maximum[index]) {
                dims_maximum[index] = *(input + index_dst);
              }
              index_dst += inner_size;
            }
            for (int64_t inner_index = 0, index_dst = index_base;
                 inner_index < dim_length; ++inner_index) {
              *(output + index_dst) = Eigen::numext::exp(*(input + index_dst) - dims_maximum[index]);
              dims_exp_sum[index] += (*(output + index_dst));
              index_dst += inner_size;
            }
            dims_exp_sum[index] = constant_one / dims_exp_sum[index];
            for (int64_t inner_index = 0, index_dst = index_base;
                 inner_index < dim_length; ++inner_index) {
              *(output + index_dst) = *(output + index_dst) * dims_exp_sum[index];
              index_dst += inner_size;
            }
          }
        }), "CpuKernelUtils::ParallelFor failed.");
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
inline std::uint32_t ComputeSoftmaxV2(const CpuKernelContext &ctx) {
  std::uint32_t result = ComputeSoftmaxV2Kernel<T>(ctx);
  if (result != 0) {
    KERNEL_LOG_ERROR("SoftmaxV2 compute failed.");
  }
  return result;
}

inline std::uint32_t SoftmaxV2ExtraCheck(const CpuKernelContext &ctx) {
  if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
    KERNEL_LOG_ERROR("The data type of the input [%s] need be the same as the ouput [%s].",
        DTypeStr(ctx.Input(0)->GetDataType()).c_str(), DTypeStr(ctx.Output(0)->GetDataType()).c_str());
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
  std::vector<int64_t> input_dims = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> output_dims = ctx.Output(0)->GetTensorShape()->GetDimSizes();
  if (input_dims.size() != output_dims.size()) {
    KERNEL_LOG_ERROR("The data dim of the input size [%llu] need be the same as the output "
        "size [%llu].", input_dims.size(), output_dims.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  for (size_t index = 0; index < input_dims.size(); index++) {
    if (input_dims[index] != output_dims[index]) {
      KERNEL_LOG_ERROR("The input data dim[%llu]=%lld need be the same as the output "
          "dim[%llu]=%lld.", index, input_dims[index], index, output_dims[index]);
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  if (ctx.GetAttr("axes") != nullptr) {
    std::vector<std::int64_t> axes = ctx.GetAttr("axes")->GetListInt();
    if (axes.size() != 1) {
      KERNEL_LOG_ERROR("The Attributes axes size is %lld, but size must be 1.", axes.size());
      return KERNEL_STATUS_PARAM_INVALID;
    }
    auto input = ctx.Input(0)->GetTensorShape()->GetDimSizes();
    std::int64_t dim = 0;
    int64_t size = static_cast<int64_t>(input.size());
    dim = (axes[0] > 0) ? axes[0] : axes[0] + size;
    if ((dim < 0 || dim >= size) && axes[0] >= size) {
      KERNEL_LOG_ERROR("The Attributes axes[0] dim=%ld is out of range of input size %ld.", axes[0], size);
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  return KERNEL_STATUS_OK;
}

std::uint32_t SoftmaxV2Check(CpuKernelContext &ctx) {
  return NormalCheck(ctx, kSoftmaxV2InputNum, kSoftmaxV2OutputNum)
             ? KERNEL_STATUS_PARAM_INVALID
             : SoftmaxV2ExtraCheck(ctx);
}
// DT_FLOAT16, DT_FLOAT, DT_DOUBLE
std::uint32_t SoftmaxV2Compute(const CpuKernelContext &ctx) {
  DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case DT_FLOAT16:
      return ComputeSoftmaxV2<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeSoftmaxV2<std::float_t>(ctx);
    case DT_DOUBLE:
      return ComputeSoftmaxV2<std::double_t>(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported input data type [%s].", DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

std::uint32_t SoftmaxV2CpuKernel::Compute(CpuKernelContext &ctx) {
  return detail::SoftmaxV2Check(ctx)
             ? KERNEL_STATUS_PARAM_INVALID
             : detail::SoftmaxV2Compute(ctx);
}

REGISTER_CPU_KERNEL(kSoftmaxV2, SoftmaxV2CpuKernel);
}  // namespace aicpu
