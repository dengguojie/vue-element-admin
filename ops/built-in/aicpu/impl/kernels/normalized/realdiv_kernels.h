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

#ifndef _AICPU_REALDIV_KERNELS_H_
#define _AICPU_REALDIV_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
class RealDivCpuKernel : public CpuKernel {
 public:
  ~RealDivCpuKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  uint32_t ComputeRealdiv(Tensor *x, Tensor *y, Tensor *z);

  uint32_t ComputeDiffType(Tensor *x, Tensor *y, Tensor *z, DataType dataType);

  template <typename T>
  uint32_t ComputeDiffShape(int64_t dim, T *xAddr, T *yAddr, T *zAddr,
                            std::vector<int64_t> &xDimSize,
                            std::vector<int64_t> &yDimSize,
                            std::vector<int64_t> &zDimSize);

  template <typename T, int32_t dim>
  void DoCompute(T *xAddr, T *yAddr, T *zAddr, std::vector<int64_t> &xDimSize,
                 std::vector<int64_t> &yDimSize,
                 std::vector<int64_t> &zDimSize);
};
}  // namespace aicpu
#endif
