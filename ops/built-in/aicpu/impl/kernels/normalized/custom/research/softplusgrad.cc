/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "softplusgrad.h"

#include <complex>
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kSoftplusGrad = "SoftplusGrad";
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const int64_t DataFloatParallelNum = 61440;
const int64_t DataDefaultParallelNum = 12288;
}  // namespace

namespace aicpu {
uint32_t SoftplusGradCpuKernel::Compute(CpuKernelContext &ctx) {
  if (NormalMathCheck(ctx) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(0)->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get input gradients data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(1)->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get input features data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Output(0)->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get output data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *input_1 = ctx.Input(kSecondInputIndex);
  Tensor *output = ctx.Output(kFirstOutputIndex);
  std::vector<int64_t> input_0_dims =
      ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> input_1_dims =
      ctx.Input(1)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> output_dims =
      ctx.Output(0)->GetTensorShape()->GetDimSizes();
  if ((input_0_dims.size() != input_1_dims.size()) ||
      (input_0_dims.size() != output_dims.size())) {
    KERNEL_LOG_ERROR(
        "The data shapes of the input must be equal, but the dimension size "
        "input0, input1, output are %d, %d and %d",
        input_0_dims.size(), input_1_dims.size(), output_dims.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  for (size_t index = 0; index < input_0_dims.size(); index++) {
    if (input_0_dims[index] != input_1_dims[index]) {
      KERNEL_LOG_ERROR(
          "The data dim[%llu]=%lld of the input0 need be the same as the "
          "input1 dim[%llu]=%lld.",
          index, input_0_dims[index], index, input_1_dims[index]);
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  for (size_t index = 0; index < input_0_dims.size(); index++) {
    if (input_0_dims[index] != output_dims[index]) {
      KERNEL_LOG_ERROR(
          "The data dim[%llu]=%lld of the input0 need be the same as the "
          "output dim[%llu]=%lld.",
          index, input_0_dims[index], index, output_dims[index]);
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  if (input_1->GetDataType() != output->GetDataType()) {
    KERNEL_LOG_ERROR(
        "The data type of the input [%s],input [%s], output [%s] must be the "
        "same type.",
        DTypeStr(ctx.Input(0)->GetDataType()).c_str(),
        DTypeStr(ctx.Input(1)->GetDataType()).c_str(),
        DTypeStr(ctx.Output(0)->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // choose compute function depend on dataType
  auto data_type =
      static_cast<DataType>(ctx.Input(kFirstInputIndex)->GetDataType());
  switch (data_type) {
    case DT_FLOAT16:
      return SoftplusGradCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return SoftplusGradCompute<float>(ctx);
    case DT_DOUBLE:
      return SoftplusGradCompute<double>(ctx);
    default:
      KERNEL_LOG_ERROR(
          "[%s] Data type of input is not support, input data type is [%s].",
          ctx.GetOpType().c_str(), DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename T>
uint32_t SoftplusGradCpuKernel::SoftplusGradCompute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO(
      "[%s] Input[0] data size is [%llu], input[1] data size is [%llu], output "
      "data size is [%llu].",
      ctx.GetOpType().c_str(), ctx.Input(0)->GetDataSize(),
      ctx.Input(1)->GetDataSize(), ctx.Output(0)->GetDataSize());
  int64_t gradients_total = ctx.Input(0)->NumElements();
  int64_t features_total = ctx.Input(1)->NumElements();
  int64_t total = std::min(gradients_total, features_total);
  std::uint64_t total_size = ctx.Input(0)->GetDataSize();
  ctx.Output(0)->GetTensorShape()->SetDimSizes(
      ctx.Input(0)->GetTensorShape()->GetDimSizes());
  uint32_t cores = aicpu::CpuKernelUtils::GetCPUNum(ctx);
  T *gradients = (T *)(ctx.Input(0)->GetData());
  T *features = (T *)(ctx.Input(1)->GetData());
  T *backprops = (T *)(ctx.Output(0)->GetData());
  bool muilt_core_flag = false;
  DataType input_type{ctx.Output(0)->GetDataType()};
  // Determine whether to enable multi-core parallel computing
  switch (input_type) {
    case DT_FLOAT: {
      if (total_size > DataFloatParallelNum * sizeof(T)) {
        muilt_core_flag = true;
      }
    }; break;
    default: {
      if (total_size > DataDefaultParallelNum * sizeof(T)) {
        muilt_core_flag = true;
      };
      break;
    }
  }
  // Eigen::Array
  if (muilt_core_flag) {
    const auto ParallelFor = aicpu::CpuKernelUtils::ParallelFor;
    std::int64_t per_unit_size{total /
                               std::min(std::max(1L, cores - 2L), total)};
    auto shard_softplusgrad = [&](std::int64_t begin, std::int64_t end) {
      std::int64_t length = end - begin;
      Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > array_gradients(
          gradients + begin, length, 1);
      Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > array_features(
          features + begin, length, 1);
      Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > array_backprops(
          backprops + begin, length, 1);
      array_backprops = array_gradients / ((-array_features).exp() + T(1));
    };
    KERNEL_HANDLE_ERROR(
        ParallelFor(ctx, total, per_unit_size, shard_softplusgrad),
        "SoftplusGrad Compute failed.")
    return KERNEL_STATUS_OK;
  } else if (cores != 0) {
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > array_gradients(gradients,
                                                                    total, 1);
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > array_features(features,
                                                                   total, 1);
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > array_backprops(backprops,
                                                                    total, 1);
    array_backprops = array_gradients / ((-array_features).exp() + T(1));
  } else {
    KERNEL_LOG_ERROR("SoftplusGrad compute failed.");
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kSoftplusGrad, SoftplusGradCpuKernel);
}  // namespace aicpu