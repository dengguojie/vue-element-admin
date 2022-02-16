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

#ifndef AICPU_KERNELS_NORMALIZED_COMPARE_AND_BITPACK_H_
#define AICPU_KERNELS_NORMALIZED_COMPARE_AND_BITPACK_H_

#include "cpu_kernel.h"

namespace aicpu {
class CompareAndBitpackCpuKernel : public CpuKernel {
public:
  CompareAndBitpackCpuKernel() = default;
  ~CompareAndBitpackCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

private:
  static uint32_t CheckParam(Tensor *x, Tensor *y, Tensor *z,
                             CpuKernelContext &ctx);

  template <typename T>
  static uint32_t CompareCompute(Tensor *x, Tensor *y, Tensor *z,
                                 CpuKernelContext &ctx);

  static uint32_t BoolCompute(Tensor *x, Tensor *y, Tensor *z,
                              CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
