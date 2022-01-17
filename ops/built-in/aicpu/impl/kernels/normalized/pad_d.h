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
#ifndef AICPU_KERNELS_NORMALIZED_PAD_D_H
#define AICPU_KERNELS_NORMALIZED_PAD_D_H

#include "cpu_kernel.h"
#include <vector>
#include "cpu_kernel_utils.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace aicpu {
class PadDCpuKernel : public CpuKernel {
public:
  PadDCpuKernel() = default;
  ~PadDCpuKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

private:
  int64_t multi(int64_t x, int64_t rank, std::vector<int64_t> &dims_y);

  int64_t sumLR(int64_t x, std::vector<int64_t> &vec);

  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);

};
} // namespace aicpu
#endif