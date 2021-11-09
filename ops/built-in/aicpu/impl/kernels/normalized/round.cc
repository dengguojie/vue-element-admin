/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All right reserved.
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
#include "round.h"

#include "unsupported/Eigen/CXX11/Tensor"
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"

namespace {
const char *kRound = "Round";

template <typename T>
const T ScalarRound(const T &x) {
  bool isInt = Eigen::NumTraits<T>::IsInteger;
  if (isInt) {
    return x;
  }

  T round_val = Eigen::numext::floor(x);
  const T fraction = x - round_val;
  if (fraction > T(.5)) {
    round_val += T(1.0);
  } else if (fraction == T(.5)) {
    const T nearest_even_int =
        round_val - T(2) * Eigen::numext::floor(T(.5) * x);
    bool is_odd = (nearest_even_int == T(1));
    if (is_odd) {
      round_val += T(1);
    }
  }
  return round_val;
}

template <typename T>
void RangeRound(int64_t start, int64_t end, T *input, T *out) {
  for (int64_t i = start; i < end; ++i) {
    out[i] = ScalarRound<T>(input[i]);
  }
}
}

namespace aicpu {
bool RoundCpuKernel::CheckSupported(DataType input_type) {
  switch (input_type) {
    case DT_FLOAT16:
    case DT_FLOAT:
    case DT_DOUBLE:
    case DT_INT32:
    case DT_INT64:
      return true;
    default:
      KERNEL_LOG_ERROR("Unsupported input data type[%d]", input_type);
      return false;
  }
}

uint32_t RoundCpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *input_tensor = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(input_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "Get input[0] failed")
  Tensor *output_tensor = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(output_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "Get output[0] failed")
  auto input_data = input_tensor->GetData();
  KERNEL_CHECK_NULLPTR(input_data, KERNEL_STATUS_PARAM_INVALID,
                       "Get input[0] data failed")
  auto output_data = output_tensor->GetData();
  KERNEL_CHECK_NULLPTR(output_data, KERNEL_STATUS_PARAM_INVALID,
                       "Get output[0] data failed")

  DataType input_type = input_tensor->GetDataType();
  if (!CheckSupported(input_type)) {
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto shardCopy = [&](int64_t start, int64_t end) {
    switch (input_type) {
      case DT_FLOAT16:
        RangeRound(start, end, static_cast<Eigen::half *>(input_data),
                   static_cast<Eigen::half *>(output_data));
        break;
      case DT_FLOAT:
        RangeRound(start, end, static_cast<float *>(input_data),
                   static_cast<float *>(output_data));
        break;
      case DT_DOUBLE:
        RangeRound(start, end, static_cast<double *>(input_data),
                   static_cast<double *>(output_data));
        break;
      case DT_INT32:
        RangeRound(start, end, static_cast<int32_t *>(input_data),
                   static_cast<int32_t *>(output_data));
        break;
      case DT_INT64:
        RangeRound(start, end, static_cast<int64_t *>(input_data),
                   static_cast<int64_t *>(output_data));
        break;
      default:
        KERNEL_LOG_ERROR("Unsupported input data type[%d]", input_type);
        return;
    }
  };

  uint32_t ret = CpuKernelUtils::ParallelFor(ctx, input_tensor->NumElements(),
                                             1, shardCopy);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kRound, RoundCpuKernel);
}  // namespace aicpu
