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

#ifndef AICPU_KERNELS_NORMALIZED_REALDIV_H
#define AICPU_KERNELS_NORMALIZED_REALDIV_H

#include "cpu_kernel.h"

namespace aicpu {
class RealDivKernel : public CpuKernel {
 public:
  ~RealDivKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  uint32_t ComputeRealdiv(Tensor *x, Tensor *y, Tensor *z,
                          CpuKernelContext &ctx);

  uint32_t ComputeDiffType(Tensor *x, Tensor *y, Tensor *z, DataType data_type,
                           CpuKernelContext &ctx);

  template <typename T>
  uint32_t ComputeDiffShape(int64_t dim, T *x_addr, T *y_addr, T *z_addr,
                            std::vector<int64_t> &x_dim_size,
                            std::vector<int64_t> &y_dim_size,
                            std::vector<int64_t> &z_dim_size,
                            CpuKernelContext &ctx);

  template <typename T, int32_t dim>
  uint32_t DoCompute(T *x_addr, T *y_addr, T *z_addr,
                     std::vector<int64_t> &x_dim_size,
                     std::vector<int64_t> &y_dim_size,
                     std::vector<int64_t> &z_dim_size, CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
