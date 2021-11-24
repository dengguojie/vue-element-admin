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
#include "diag.h"

#include <atomic>
#include "unsupported/Eigen/CXX11/Tensor"

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const char *kDiag = "Diag";
const char *kDiagPart = "DiagPart";
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 1;

template <typename T>
void RangeDiag(int64_t start, int64_t end, int64_t size, T *input, T *out) {
  std::fill(out + size * start, out + size * end, T());
  for (int64_t i = start; i < end; ++i) {
    out[(1 + size) * i] = input[i];
  }
}

template <typename T>
void RangeDiagPart(int64_t start, int64_t end, int64_t size, T *input, T *out) {
  for (int64_t i = start; i < end; ++i) {
    out[i] = input[(1 + size) * i];
  }
}
}  // namespace

namespace aicpu {
uint32_t DiagCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Check Diag params failed.");
  Tensor *input_tensor = ctx.Input(0);
  KERNEL_CHECK_FALSE((input_tensor->GetTensorShape()->GetDims() != 0),
                     KERNEL_STATUS_INNER_ERROR,
                     "Input must be at least rank 1, but got rank 0")
  int64_t out_data_num = 0;
  KERNEL_CHECK_ASSIGN_64S_MULTI(input_tensor->NumElements(),
                                input_tensor->NumElements(), out_data_num,
                                KERNEL_STATUS_INNER_ERROR);

  Tensor *output_tensor = ctx.Output(0);
  KERNEL_CHECK_FALSE((output_tensor->NumElements() >= out_data_num),
                     KERNEL_STATUS_INNER_ERROR,
                     "The output elements number[%ld] must be greater than "
                     "or equal to the square of the input elements number[%ld]",
                     output_tensor->NumElements(), input_tensor->NumElements())

  auto input_data = input_tensor->GetData();
  auto output_data = output_tensor->GetData();
  DataType input_type = input_tensor->GetDataType();
  std::atomic<bool> shard_ret(true);
  auto shard = [&](int64_t start, int64_t end) {
    switch (input_type) {
      case DT_FLOAT16:
        RangeDiag(start, end, input_tensor->NumElements(),
                  static_cast<Eigen::half *>(input_data),
                  static_cast<Eigen::half *>(output_data));
        break;
      case DT_FLOAT:
        RangeDiag(start, end, input_tensor->NumElements(),
                  static_cast<float *>(input_data),
                  static_cast<float *>(output_data));
        break;
      case DT_DOUBLE:
        RangeDiag(start, end, input_tensor->NumElements(),
                  static_cast<double *>(input_data),
                  static_cast<double *>(output_data));
        break;
      case DT_INT32:
        RangeDiag(start, end, input_tensor->NumElements(),
                  static_cast<int32_t *>(input_data),
                  static_cast<int32_t *>(output_data));
        break;
      case DT_INT64:
        RangeDiag(start, end, input_tensor->NumElements(),
                  static_cast<int64_t *>(input_data),
                  static_cast<int64_t *>(output_data));
        break;
      default:
        KERNEL_LOG_ERROR("Unsupported input data type[%s]",
                         DTypeStr(input_type).c_str());
        shard_ret.store(false);
        return;
    }
  };

  uint32_t ret =
      CpuKernelUtils::ParallelFor(ctx, input_tensor->NumElements(), 1, shard);
  if ((ret != KERNEL_STATUS_OK) || (!shard_ret.load())) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kDiag, DiagCpuKernel);

uint32_t DiagPartCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Check DiagPart params failed.");
  Tensor *input_tensor = ctx.Input(0);
  int32_t input_dims = input_tensor->GetTensorShape()->GetDims();
  KERNEL_CHECK_FALSE(
      (input_dims > 0 && input_dims % 2 == 0), KERNEL_STATUS_INNER_ERROR,
      "Input rank must be even and positive, but got rank[%d]", input_dims)
  Tensor *output_tensor = ctx.Output(0);
  int64_t input_data_num = 0;
  KERNEL_CHECK_ASSIGN_64S_MULTI(output_tensor->NumElements(),
                                output_tensor->NumElements(), input_data_num,
                                KERNEL_STATUS_INNER_ERROR);
  KERNEL_CHECK_FALSE(
      (input_tensor->NumElements() >= input_data_num),
      KERNEL_STATUS_INNER_ERROR,
      "The input elements number[%ld] must be greater than "
      "or equal to the square of the output elements number[%ld]",
      output_tensor->NumElements(), input_tensor->NumElements())
  auto input_data = input_tensor->GetData();
  auto output_data = output_tensor->GetData();
  DataType input_type = input_tensor->GetDataType();
  std::atomic<bool> shard_ret(true);
  auto shard = [&](int64_t start, int64_t end) {
    switch (input_type) {
      case DT_FLOAT16:
        RangeDiagPart(start, end, output_tensor->NumElements(),
                      static_cast<Eigen::half *>(input_data),
                      static_cast<Eigen::half *>(output_data));
        break;
      case DT_FLOAT:
        RangeDiagPart(start, end, output_tensor->NumElements(),
                      static_cast<float *>(input_data),
                      static_cast<float *>(output_data));
        break;
      case DT_DOUBLE:
        RangeDiagPart(start, end, output_tensor->NumElements(),
                      static_cast<double *>(input_data),
                      static_cast<double *>(output_data));
        break;
      case DT_INT32:
        RangeDiagPart(start, end, output_tensor->NumElements(),
                      static_cast<int32_t *>(input_data),
                      static_cast<int32_t *>(output_data));
        break;
      case DT_INT64:
        RangeDiagPart(start, end, output_tensor->NumElements(),
                      static_cast<int64_t *>(input_data),
                      static_cast<int64_t *>(output_data));
        break;
      default:
        KERNEL_LOG_ERROR("Unsupported input data type[%s]",
                         DTypeStr(input_type).c_str());
        shard_ret.store(false);
        return;
    }
  };

  uint32_t ret =
      CpuKernelUtils::ParallelFor(ctx, output_tensor->NumElements(), 1, shard);
  if ((ret != KERNEL_STATUS_OK) || (!shard_ret.load())) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kDiagPart, DiagPartCpuKernel);
}  // namespace aicpu
