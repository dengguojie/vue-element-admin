/**
 * Copyright(c) Huawei Technologies Co., Ltd.2021-2021.All rights reserved.
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

#ifndef AICPU_KERNELS_NORMALIZED_TRIU_H_
#define AICPU_KERNELS_NORMALIZED_TRIU_H_

#include "cpu_kernel.h"

namespace aicpu {
class TriuCpuKernel : public CpuKernel {
 public:
  TriuCpuKernel() = default;
  ~TriuCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t ValidParam(CpuKernelContext &ctx);

  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);

  template <typename T>
  void SetResult(CpuKernelContext &ctx, int64_t matrix_start,
                 int64_t matrix_end);

  template <typename MatrixMap>
  void SetResultDiagonalMinus(MatrixMap output, MatrixMap input,
                              int32_t diagonal_, int64_t matrix_height,
                              int64_t matrix_width);

  template <typename MatrixMap, typename T>
  void SetResultDiagonaPositive(MatrixMap output, int32_t diagonal_,
                                int64_t matrix_height, int64_t matrix_width);
  int32_t diagonal_ = 0;
};
}  // namespace aicpu
#endif
