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
template <typename T>
void CallDiagCalc(int64_t start, int64_t end, Tensor *&input_tensor,
                  Tensor *&output_tensor) {
  RangeDiag(start, end, input_tensor->NumElements(),
            static_cast<T *>(input_tensor->GetData()),
            static_cast<T *>(output_tensor->GetData()));
}

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

  std::map<int, std::function<void(int64_t, int64_t, Tensor *&, Tensor *&)>>
      calls;
  calls[DT_FLOAT16] = CallDiagCalc<Eigen::half>;
  calls[DT_FLOAT] = CallDiagCalc<float>;
  calls[DT_DOUBLE] = CallDiagCalc<double>;
  calls[DT_INT32] = CallDiagCalc<int32_t>;
  calls[DT_INT64] = CallDiagCalc<int64_t>;
  calls[DT_COMPLEX64] = CallDiagCalc<std::complex<float>>;
  calls[DT_COMPLEX128] = CallDiagCalc<std::complex<double>>;

  DataType input_type = input_tensor->GetDataType();
  std::atomic<bool> shard_ret(true);
  if (calls.find(input_type) == calls.end()) {
    KERNEL_LOG_ERROR("Unsupported input data type[%s]",
                     DTypeStr(input_type).c_str());
    shard_ret.store(false);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto shard = [&](int64_t start, int64_t end) {
    calls[input_type](start, end, input_tensor, output_tensor);
  };

  uint32_t ret =
      CpuKernelUtils::ParallelFor(ctx, input_tensor->NumElements(), 1, shard);
  if ((ret != KERNEL_STATUS_OK) || (!shard_ret.load())) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kDiag, DiagCpuKernel);

template <typename T>
void CallDiagPartCalc(int64_t start, int64_t end, Tensor *&input_tensor,
                      Tensor *&output_tensor) {
  RangeDiagPart(start, end, output_tensor->NumElements(),
                static_cast<T *>(input_tensor->GetData()),
                static_cast<T *>(output_tensor->GetData()));
}

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

  std::map<int, std::function<void(int64_t, int64_t, Tensor *&, Tensor *&)>>
      calls;
  calls[DT_FLOAT16] = CallDiagPartCalc<Eigen::half>;
  calls[DT_FLOAT] = CallDiagPartCalc<float>;
  calls[DT_DOUBLE] = CallDiagPartCalc<double>;
  calls[DT_INT32] = CallDiagPartCalc<int32_t>;
  calls[DT_INT64] = CallDiagPartCalc<int64_t>;
  calls[DT_COMPLEX64] = CallDiagPartCalc<std::complex<float>>;
  calls[DT_COMPLEX128] = CallDiagPartCalc<std::complex<double>>;

  std::atomic<bool> shard_ret(true);
  DataType input_type = input_tensor->GetDataType();
  if (calls.find(input_type) == calls.end()) {
    KERNEL_LOG_ERROR("Unsupported input data type[%s]",
                     DTypeStr(input_type).c_str());
    shard_ret.store(false);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto shard = [&](int64_t start, int64_t end) {
    calls[input_type](start, end, input_tensor, output_tensor);
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
