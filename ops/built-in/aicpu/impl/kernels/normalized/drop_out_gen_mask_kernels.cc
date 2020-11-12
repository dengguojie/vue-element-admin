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

#include "drop_out_gen_mask_kernels.h"

#include <memory.h>
#include <cfloat>
#include <ctime>
#include <random>

#include "cpu_types.h"
#include "log.h"
#include "status.h"

namespace {
const char *DropOutGenMask = "DropOutGenMask";
}

namespace aicpu {
std::random_device e;

uint32_t DropOutGenMaskCpuKernel::Compute(CpuKernelContext &ctx) {
  uint64_t tmp_count = 1;
  Tensor *shape_tensor = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(shape_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "get input:0 failed.")

  auto input_shape = shape_tensor->GetTensorShape();
  KERNEL_CHECK_NULLPTR(input_shape, KERNEL_STATUS_PARAM_INVALID,
                       "get input_shape failed.")

  DataType shape_dt = static_cast<DataType>(shape_tensor->GetDataType());
  for (int j = 0; j < input_shape->GetDims(); j++) {
    tmp_count *= input_shape->GetDimSize(j);
  }

  uint64_t count = 1;
  if (shape_dt == DT_INT32) {
    auto input0 = reinterpret_cast<int32_t *>(shape_tensor->GetData());
    for (uint64_t index = 0; index < tmp_count; index++) {
      count *= input0[index];
    }
  } else {
    auto input0 = reinterpret_cast<int64_t *>(shape_tensor->GetData());
    for (uint64_t index = 0; index < tmp_count; index++) {
      count *= input0[index];
    }
  }

  Tensor *prob_tensor = ctx.Input(1);
  KERNEL_CHECK_NULLPTR(prob_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "get input:1 failed.")

  DataType dt = static_cast<DataType>(prob_tensor->GetDataType());
  float keep_prob = 0;
  if (dt == DT_FLOAT16) {
    keep_prob = *reinterpret_cast<float *>(prob_tensor->GetData());
  } else {
    keep_prob = *reinterpret_cast<float *>(prob_tensor->GetData());
  }
  KERNEL_LOG_INFO("DropOutGenMask mask count and pro: %d %f", count, keep_prob);

  Tensor *out_tensor = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(out_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "get output:0 failed.")

  std::default_random_engine e(time(0));
  std::bernoulli_distribution b(keep_prob);
  const uint8_t mask[8] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};
  uint64_t byteCount = count >> 3;
  uint8_t *out = reinterpret_cast<uint8_t *>(out_tensor->GetData());
  for (uint64_t i = 0; i < byteCount; ++i) {
    out[i] = 0x00;
    for (const auto &m : mask) {
      if (b(e)) {
        out[i] = out[i] | m;
      }
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(DropOutGenMask, DropOutGenMaskCpuKernel);
}  // namespace aicpu
