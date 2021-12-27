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
#include "tanhgrad.h"

#include <complex>
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kTanhGrad = "TanhGrad";
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const int64_t DataFloat16ParallelNum = 4096;
const int64_t DataComplexParallelNum = 8192;
const int64_t DataDefaultParallelNum = 32768;
}  // namespace

namespace aicpu {
uint32_t TanhGradCpuKernel::Compute(CpuKernelContext &ctx) {
  if (NormalMathCheck(ctx) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(0)->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get input y data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(1)->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get input dy data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Output(0)->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get output data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *input_0 = ctx.Input(kFirstInputIndex);
  Tensor *input_1 = ctx.Input(kSecondInputIndex);
  Tensor *output = ctx.Output(kFirstOutputIndex);
  if ((input_0->GetDataSize() == 0) || (input_1->GetDataSize() == 0)) {
    KERNEL_LOG_INFO("[%s] Input is empty tensor.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_OK;
  }
  if (input_1->GetDataType() != output->GetDataType()) {
    KERNEL_LOG_ERROR(
        "The data type of the input [%s],input [%s], output [%s] must be the same type.",
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
      return TanhGradCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return TanhGradCompute<float>(ctx);
    case DT_DOUBLE:
      return TanhGradCompute<double>(ctx);
    case DT_COMPLEX64:
      return TanhGradCompute<std::complex<std::float_t> >(ctx);
    case DT_COMPLEX128:
      return TanhGradCompute<std::complex<std::double_t> >(ctx);
    default:
      KERNEL_LOG_ERROR(
          "[%s] Data type of input is not support, input data type is [%s].",
          ctx.GetOpType().c_str(), DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename T>
uint32_t TanhGradCpuKernel::TanhGradCompute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO(
      "[%s] Input[0] data size is [%llu], input[1] data size is [%llu], output "
      "data size is [%llu].",
      ctx.GetOpType().c_str(), ctx.Input(0)->GetDataSize(),
      ctx.Input(1)->GetDataSize(), ctx.Output(0)->GetDataSize());

  int64_t input_y_total = ctx.Input(0)->NumElements();
  int64_t input_dy_total = ctx.Input(1)->NumElements();
  int64_t total = std::min(input_y_total, input_dy_total);
  ctx.Output(0)->GetTensorShape()->SetDimSizes(
      ctx.Input(0)->GetTensorShape()->GetDimSizes());
  uint32_t cores = aicpu::CpuKernelUtils::GetCPUNum(ctx);
  T *input_y = static_cast<T *>(ctx.Input(0)->GetData());
  T *input_dy = static_cast<T *>(ctx.Input(1)->GetData());
  T *output = static_cast<T *>(ctx.Output(0)->GetData());
  bool muilt_core_flag = false;
  DataType input_type{ctx.Output(0)->GetDataType()};
  // Determine whether to enable multi-core parallel computing
  switch (input_type) {
    case DT_FLOAT16: {
      if (total > DataFloat16ParallelNum) {
        muilt_core_flag = true;
      }
    }; break;
    case DT_COMPLEX64: {
      if (total > DataComplexParallelNum) {
        muilt_core_flag = true;
      }
    }; break;
    case DT_COMPLEX128: {
      if (total > DataComplexParallelNum) {
        muilt_core_flag = true;
      }
    }; break;
    default: {
      if (total > DataDefaultParallelNum) {
        muilt_core_flag = true;
      }
    }
  }
  // Eigen::Array
  if (muilt_core_flag) {
    const auto ParallelFor = aicpu::CpuKernelUtils::ParallelFor;
    std::int64_t per_unit_size{total / std::min(std::max(1L, cores - 2L), total)};
    auto shard_tanhgrad = [&](std::int64_t begin, std::int64_t end) {
      std::int64_t length = end - begin;
      Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > array_y(input_y + begin,
                                                              length, 1);
      Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > array_dy(input_dy + begin,
                                                               length, 1);
      Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > array_z(output + begin,
                                                              length, 1);
      array_z = array_dy * (T(1.0) - array_y * array_y);
    };
    KERNEL_HANDLE_ERROR(ParallelFor(ctx, total, per_unit_size, shard_tanhgrad),
                        "TanhGrad Compute failed.")
    return KERNEL_STATUS_OK;
  } else if (cores != 0) {
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > array_y(input_y, total, 1);
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > array_dy(input_dy, total,
                                                             1);
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > array_z(output, total, 1);
    array_z = array_dy * (T(1.0) - array_y * array_y);
  } else {
    KERNEL_LOG_ERROR("TanhGrad compute failed.");
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kTanhGrad, TanhGradCpuKernel);
}  // namespace aicpu
