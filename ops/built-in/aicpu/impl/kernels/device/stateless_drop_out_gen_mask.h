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

#ifndef AICPU_KERNELS_DEVICE_STATELESS_DROP_OUT_GEN_MASK
#define AICPU_KERNELS_DEVICE_STATELESS_DROP_OUT_GEN_MASK

#include "cpu_kernel.h"
#include "cpu_types.h"

namespace aicpu {
class StatelessDropOutGenMaskCpuKernel : public CpuKernel {
 public:
  StatelessDropOutGenMaskCpuKernel() = default;
  ~StatelessDropOutGenMaskCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  void StatelessDropOutGenMaskKernel(const uint64_t count, const float prob,
                                     const uint8_t *offset, const uint8_t *key,
                                     uint8_t *out);
  uint32_t DoCompute(CpuKernelContext &ctx, float prob, int64_t seed,
                     int64_t seed1, uint8_t *out_buff, uint64_t outputSize);
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_DEVICE_STATELESS_DROP_OUT_GEN_MASK