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
#include "logsoftmaxv2.h"

#include <securec.h>
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kLogSoftmaxV2InputNum = 1;
const uint32_t kLogSoftmaxV2OutputNum = 1;
const uint32_t dimType1 = 1;
const uint32_t dimType2 = 2;
const uint32_t dimType3 = 3;
const int64_t paralled_data_size = 4 * 1024;
const char* kLogSoftmaxV2 = "LogSoftmaxV2";
#define LOGSOFTMAXV2_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                              \
    uint32_t result = LogSoftmaxV2Compute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                          \
      KERNEL_LOG_ERROR("LogSoftmaxV2 kernel compute failed."); \
      return result;                                           \
    }                                                          \
    break;                                                     \
  }
}  // namespace

namespace aicpu {
uint32_t LogSoftmaxV2CpuKernel::Compute(CpuKernelContext& ctx) {
  // check params
  KERNEL_HANDLE_ERROR(
      NormalCheck(ctx, kLogSoftmaxV2InputNum, kLogSoftmaxV2OutputNum),
      "[%s] check input and output failed.", kLogSoftmaxV2);
  // parse params
  KERNEL_HANDLE_ERROR(LogSoftmaxV2Check(ctx), "[%s] check params failed.",
                      kLogSoftmaxV2);
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    LOGSOFTMAXV2_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    LOGSOFTMAXV2_COMPUTE_CASE(DT_FLOAT, float, ctx)
    LOGSOFTMAXV2_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("LogSoftmaxV2 kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t LogSoftmaxV2CpuKernel::LogSoftmaxV2Check(CpuKernelContext& ctx) {
  KERNEL_CHECK_NULLPTR(ctx.Input(0)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "get input failed.");
  KERNEL_CHECK_NULLPTR(ctx.Input(0)->GetTensorShape(),
                       KERNEL_STATUS_PARAM_INVALID,
                       "Get input tensor shape failed.")
  KERNEL_CHECK_NULLPTR(ctx.Output(0)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "get output failed.");
  KERNEL_CHECK_NULLPTR(ctx.GetAttr("axes"), KERNEL_STATUS_PARAM_INVALID,
                       "get exclusive failed.");
  std::vector<std::int64_t> axes_data = ctx.GetAttr("axes")->GetListInt();
  KERNEL_CHECK_FALSE((axes_data.size() == 1), KERNEL_STATUS_PARAM_INVALID,
                     "axes is out of shape");
  int64_t axes = axes_data[0];
  KERNEL_CHECK_FALSE((axes < ctx.Input(0)->GetTensorShape()->GetDims()),
                     KERNEL_STATUS_PARAM_INVALID,
                     "axes is larger than input dims - 1");
  KERNEL_CHECK_FALSE((axes >= -ctx.Input(0)->GetTensorShape()->GetDims()),
                     KERNEL_STATUS_PARAM_INVALID,
                     "axes is lower than -input dims");
  std::vector<int64_t> shape_input =
      ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_output =
      ctx.Output(0)->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((shape_input.size() != 0), KERNEL_STATUS_PARAM_INVALID,
                     "Input must be at least rank 1, got [%zu].",
                     shape_input.size())
  KERNEL_CHECK_FALSE(
      (shape_input.size() == shape_output.size()), KERNEL_STATUS_PARAM_INVALID,
      "The output shape size should be same as the output shape size")
  return KERNEL_STATUS_OK;
}
template <typename T>
uint32_t LogSoftmaxV2CpuKernel::LogSoftmaxV2Compute(CpuKernelContext& ctx) {
  const auto ParallelFor{aicpu::CpuKernelUtils::ParallelFor};
  auto input = static_cast<T*>(ctx.Input(0)->GetData());
  auto output = static_cast<T*>(ctx.Output(0)->GetData());
  std::vector<std::int64_t> axes{-1};
  if (ctx.GetAttr("axes") != nullptr) {
    axes = ctx.GetAttr("axes")->GetListInt();
  }
  std::int64_t total = ctx.Input(0)->NumElements();
  std::vector<std::int64_t> dims =
      ctx.Input(0)->GetTensorShape()->GetDimSizes();
  uint32_t cores = aicpu::CpuKernelUtils::GetCPUNum(ctx);
  if (cores < 1) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  std::int64_t pivot, inner_size{1}, outer_size{1}, length{1};
  if (axes[0] >= 0) {
    pivot = axes[0];
  } else {
    pivot = dims.size() + axes[0];
  }
  int64_t size = dims.size();
  for (int64_t index = 0; index < size; index++) {
    if (index > pivot) {
      inner_size *= dims[index];
    }
    if (index < pivot) {
      outer_size *= dims[index];
    }
  }
  length = inner_size * outer_size;
  T dims_exp_sum[length];
  T dims_maximum[length];
  memset_s(dims_exp_sum, length * sizeof(T), 0, length * sizeof(T));
  int64_t data_size = total * sizeof(T);
  if (data_size <= paralled_data_size) {
    Eigen::TensorMap<Eigen::Tensor<T, dimType3>, Eigen::Aligned> logits(
        input, inner_size, (int)dims[pivot], outer_size);
    Eigen::TensorMap<Eigen::Tensor<T, dimType1>, Eigen::Aligned> dims_sum(dims_exp_sum,
                                                                   length);
    Eigen::TensorMap<Eigen::Tensor<T, dimType2>, Eigen::Aligned> dims_max(
        dims_maximum, inner_size, outer_size);
    Eigen::array<int, 1> softmax_axes{{1}};
    dims_max = logits.maximum(softmax_axes);
    for (int64_t index = 0, index_dst = 0, index_batch = 0, step = 0;
         index < total; index++) {
      if (index % inner_size == 0 && index != 0) {
        step++;
        if (step == dims[pivot]) {
          step = 0;
          index_batch += inner_size;
        }
        index_dst = index_batch;
      }
      *(output + index) =
          Eigen::numext::exp(*(input + index) - dims_maximum[index_dst]);
      dims_exp_sum[index_dst] += (*(output + index));
      index_dst++;
    }
    dims_sum = dims_sum.inverse();
    for (int64_t index = 0, index_dst = 0, index_batch = 0, step = 0;
         index < total; index++) {
      if (index % inner_size == 0 && index != 0) {
        step++;
        if (step == dims[pivot]) {
          step = 0;
          index_batch += inner_size;
        }
        index_dst = index_batch;
      }
      *(output + index) = (*(output + index)) * (dims_exp_sum[index_dst]);
      *(output + index) = Eigen::numext::log(*(output + index));
      index_dst++;
    }
  } else {
    std::int64_t per_unit_size{length /
                               std::min(std::max(1L, cores - 2L), length)};
    const T constant_one(1.0);
    ParallelFor(
        ctx, length, per_unit_size, [&](std::int64_t begin, std::int64_t end) {
          for (int64_t index = begin, dim_length = dims[pivot], outer_index,
                      index_base;
               index < end; ++index) {
            outer_index = index / inner_size;
            index_base =
                outer_index * dim_length * inner_size + index % inner_size;
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
              *(output + index_dst) = Eigen::numext::exp(*(input + index_dst) -
                                                         dims_maximum[index]);
              dims_exp_sum[index] += (*(output + index_dst));
              index_dst += inner_size;
            }
            dims_exp_sum[index] = constant_one / dims_exp_sum[index];
            for (int64_t inner_index = 0, index_dst = index_base;
                 inner_index < dim_length; ++inner_index) {
              *(output + index_dst) =
                  *(output + index_dst) * dims_exp_sum[index];
              *(output + index_dst) = Eigen::numext::log(*(output + index_dst));
              index_dst += inner_size;
            }
          }
        });
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kLogSoftmaxV2, LogSoftmaxV2CpuKernel);
}  // namespace aicpu