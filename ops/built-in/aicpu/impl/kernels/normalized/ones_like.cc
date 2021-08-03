/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "ones_like.h"

#include <atomic>
#include "unsupported/Eigen/CXX11/Tensor"

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const char *kOnesLike = "OnesLike";
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 1;

template <typename T>
void RangeOnesLike(int64_t start, int64_t end, T *out) {
  for (int64_t i = start; i < end; ++i) {
    out[i] = T(1);
  }
}
}  // namespace

namespace aicpu {
uint32_t OnesLikeCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Check OnesLike params failed.");
  Tensor *input_tensor = ctx.Input(0);
  Tensor *output_tensor = ctx.Output(0);
  auto output_data = output_tensor->GetData();

  DataType input_type = input_tensor->GetDataType();
  std::atomic<bool> shard_ret(true);
  auto shard = [&](int64_t start, int64_t end) {
    switch (input_type) {
      case DT_FLOAT16:
        RangeOnesLike(start, end, static_cast<Eigen::half *>(output_data));
        break;
      case DT_FLOAT:
        RangeOnesLike(start, end, static_cast<float *>(output_data));
        break;
      case DT_DOUBLE:
        RangeOnesLike(start, end, static_cast<double *>(output_data));
        break;
      case DT_INT8:
        RangeOnesLike(start, end, static_cast<int8_t *>(output_data));
        break;
      case DT_INT16:
        RangeOnesLike(start, end, static_cast<int16_t *>(output_data));
        break;
      case DT_INT32:
        RangeOnesLike(start, end, static_cast<int32_t *>(output_data));
        break;
      case DT_INT64:
        RangeOnesLike(start, end, static_cast<int64_t *>(output_data));
        break;
      case DT_UINT8:
        RangeOnesLike(start, end, static_cast<uint8_t *>(output_data));
        break;
      case DT_UINT16:
        RangeOnesLike(start, end, static_cast<uint16_t *>(output_data));
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

REGISTER_CPU_KERNEL(kOnesLike, OnesLikeCpuKernel);
}  // namespace aicpu
