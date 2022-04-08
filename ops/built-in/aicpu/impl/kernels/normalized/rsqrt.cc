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
#include "rsqrt.h"

#include <cfloat>
#include <complex>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kRsqrt = "Rsqrt";
const size_t kRsqrtInputNum = 1;
const size_t kRsqrtOutputNum = 1;
constexpr int64_t kParallelDataNums = 8 * 1024;
constexpr int64_t kParallelComplexDataNums = 4 * 1024;
}  // namespace

namespace aicpu {
uint32_t RsqrtCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kRsqrtOutputNum, kRsqrtInputNum),
                      "Check Rsqrt params failed.");
  if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
    KERNEL_LOG_ERROR(
        "The data type of the input [%s] need be the same as the ouput [%s]",
        DTypeStr(ctx.Input(0)->GetDataType()).c_str(),
        DTypeStr(ctx.Output(0)->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(0)->GetDataSize() != ctx.Output(0)->GetDataSize()) {
    KERNEL_LOG_ERROR(
        "The data size of the input [%llu] need be the same as the ouput "
        "[%llu]",
        ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *x = ctx.Input(0);
  Tensor *y = ctx.Output(0);
  int64_t data_num = x->NumElements();
  DataType data_type = x->GetDataType();
  uint32_t res = KERNEL_STATUS_OK;

  switch (data_type) {
    case DT_FLOAT16:
      res = RsqrtCompute<Eigen::half>(x, y, data_num, ctx);
      break;
    case DT_FLOAT:
      res = RsqrtCompute<float>(x, y, data_num, ctx);
      break;
    case DT_DOUBLE:
      res = RsqrtCompute<double>(x, y, data_num, ctx);
      break;
    case DT_COMPLEX64:
      res = RsqrtComputeComplex<std::complex<float>>(x, y, data_num, ctx);
      break;
    case DT_COMPLEX128:
      res = RsqrtComputeComplex<std::complex<double>>(x, y, data_num, ctx);
      break;
    default:
      KERNEL_LOG_ERROR("Rsqrt invalid input type [%s]",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  if (res != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t RsqrtCpuKernel::RsqrtCompute(Tensor *x, Tensor *y, int64_t data_num,
                                      CpuKernelContext &ctx) const {
  auto input_x = reinterpret_cast<T *>(x->GetData());
  KERNEL_CHECK_NULLPTR(input_x, KERNEL_STATUS_PARAM_INVALID, "Get input data failed")
  auto output_y = reinterpret_cast<T *>(y->GetData());
  KERNEL_CHECK_NULLPTR(output_y, KERNEL_STATUS_PARAM_INVALID, "Get output data failed")
  if (data_num <= kParallelDataNums) {
    for (int64_t i = 0; i < data_num; i++) {
      if (x->GetDataType() == DT_FLOAT16) {
        if ((Eigen::half)input_x[i] == Eigen::half{0.0f}) {
          KERNEL_LOG_ERROR("Rsqrt kernel input[%ld] cannot be 0", i);
          return KERNEL_STATUS_PARAM_INVALID;
        }
      } else if (x->GetDataType() == DT_FLOAT) {
        if ((std::fabs(static_cast<float>(input_x[i])) < FLT_EPSILON)) {
          KERNEL_LOG_ERROR("Rsqrt kernel input[%ld] cannot be 0", i);
          return KERNEL_STATUS_PARAM_INVALID;
        }
      } else if (x->GetDataType() == DT_DOUBLE) {
        if ((std::fabs(static_cast<double>(input_x[i])) < DBL_EPSILON)) {
          KERNEL_LOG_ERROR("Rsqrt kernel input[%ld] cannot be 0", i);
          return KERNEL_STATUS_PARAM_INVALID;
        }
      }
      output_y[i] = static_cast<T>(1) / (sqrt(input_x[i]));
    }
  } else {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(
        min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto shard_rsqrt = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        if (x->GetDataType() == DT_FLOAT16) {
          if ((Eigen::half)input_x[i] == Eigen::half{0.0f}) {
            KERNEL_LOG_ERROR("Rsqrt kernel input[%zu] cannot be 0", i);
          }
        } else if (x->GetDataType() == DT_FLOAT) {
          if ((std::fabs(static_cast<float>(input_x[i])) < FLT_EPSILON)) {
            KERNEL_LOG_ERROR("Rsqrt kernel input[%zu] cannot be 0", i);
          }
        } else if (x->GetDataType() == DT_DOUBLE) {
          if ((std::fabs(static_cast<double>(input_x[i])) < DBL_EPSILON)) {
            KERNEL_LOG_ERROR("Rsqrt kernel input[%zu] cannot be 0", i);
          }
        }
        output_y[i] = static_cast<T>(1) / (sqrt(input_x[i]));
      }
    };
    KERNEL_HANDLE_ERROR(
        CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num,
                                    shard_rsqrt),
        "Rsqrt Compute failed.")
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t RsqrtCpuKernel::RsqrtComputeComplex(Tensor *x, Tensor *y,
                                             int64_t data_num,
                                             CpuKernelContext &ctx) const {
  auto input_x = reinterpret_cast<T *>(x->GetData());
  KERNEL_CHECK_NULLPTR(input_x, KERNEL_STATUS_PARAM_INVALID,
                       "Get input data failed")
  auto output_y = reinterpret_cast<T *>(y->GetData());
  KERNEL_CHECK_NULLPTR(output_y, KERNEL_STATUS_PARAM_INVALID,
                       "Get output data failed")
  if (data_num <= kParallelComplexDataNums) {
    for (int64_t i = 0; i < data_num; i++) {
      output_y[i] =
          sqrt(conj(input_x[i])) / sqrt(input_x[i].real() * input_x[i].real() +
                                        input_x[i].imag() * input_x[i].imag());
    }
  } else {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(
        min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }

    auto shard_rsqrt = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        output_y[i] = sqrt(conj(input_x[i])) /
                      sqrt(input_x[i].real() * input_x[i].real() +
                           input_x[i].imag() * input_x[i].imag());
      }
    };
    KERNEL_HANDLE_ERROR(
        CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num,
                                    shard_rsqrt),
        "Rsqrt Compute failed.")
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kRsqrt, RsqrtCpuKernel);
}  // namespace aicpu

