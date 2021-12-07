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
#ifndef AICPU_KERNELS_NORMALIZED_CUMSUM_H_
#define AICPU_KERNELS_NORMALIZED_CUMSUM_H_

#include "cpu_kernel.h"

namespace aicpu {
class CumsumCpuKernel : public CpuKernel {
 public:
  CumsumCpuKernel() = default;
  ~CumsumCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext& ctx) override;

 private:
  uint32_t CumsumCheck(CpuKernelContext& ctx);
  
  void CumsumGetAttr(CpuKernelContext &ctx, bool &exclusive, bool &reverse);

  template <typename T>
  uint32_t CumsumCompute(CpuKernelContext& ctx);
  template <typename T, typename T2>
  uint32_t CumsumCompute2(CpuKernelContext& ctx);
};
}  // namespace aicpu
#endif
