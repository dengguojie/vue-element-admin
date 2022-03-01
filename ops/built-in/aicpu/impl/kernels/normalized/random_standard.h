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
#ifndef AICPU_KERNELS_NORMALIZED_RANDOM_STANDARD_H
#define AICPU_KERNELS_NORMALIZED_RANDOM_STANDARD_H
#define EIGEN_USE_THREADS
#define EIGEN_USE_SIMPLE_THREAD_POOL

#include "cpu_kernel.h"

namespace aicpu {
class RandomStandardCpuKernel : public CpuKernel {
 public:
  RandomStandardCpuKernel() = default;
  ~RandomStandardCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  /**
   * @brief generate data from Eigen
   * @param ctx cpu kernel context
   * @param output using to output data
   * @return status if success
   */
  template <typename T>
  void Generate(CpuKernelContext &ctx, Tensor *output);
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_RANDOM_STANDARD_H
