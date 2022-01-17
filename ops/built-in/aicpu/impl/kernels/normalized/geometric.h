/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021.All rights reserved.
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

#ifndef AICPU_KERNELS_NORMALIZED_GEOMETRIC_H_
#define AICPU_KERNELS_NORMALIZED_GEOMETRIC_H_

#include "cpu_kernel.h"
#include "utils/bcast.h"

namespace aicpu {
class GeometricCpuKernel : public CpuKernel {
 public:
  GeometricCpuKernel() = default;
  ~GeometricCpuKernel() override = default;

  protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);

  uint32_t ExtraParamCheck(CpuKernelContext &ctx);
  int32_t attr_seed_ = 0;
  float p_ = 0.5;
};
}  // namespace aicpu
#endif
