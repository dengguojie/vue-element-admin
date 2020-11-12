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

#ifndef _AICPU_TOPK_KERNELS_H_
#define _AICPU_TOPK_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
struct MatrixInfo {
  std::vector<int> matrix_shape;
  int32_t matrix_type;
};
class TopKCpuKernel : public CpuKernel {
 public:
  ~TopKCpuKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t GetInputAndCheck(CpuKernelContext &ctx);
  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);
  template <typename T>
  static uint32_t TopKForNVector(T *input, T *value, int32_t *indices, int col,
                                 int k, int n, bool sorted);
  int32_t k_;
  bool sorted_;
  Tensor *input_tensor_;
  Tensor *output_values_;
  Tensor *output_indices_;
  MatrixInfo matrix_info_;
  int32_t col_ = 0;
  int32_t row_ = 0;
};
}  // namespace aicpu
#endif
