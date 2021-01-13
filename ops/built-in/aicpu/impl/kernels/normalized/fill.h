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

#ifndef AICPU_KERNELS_NORMALIZED_FILL_H_
#define AICPU_KERNELS_NORMALIZED_FILL_H_

#include "cpu_kernel.h"

namespace aicpu {
class FillCpuKernel : public CpuKernel {
 public:
  FillCpuKernel() = default;
  ~FillCpuKernel() override = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  /**
   * @brief calc dims from input dims tensor
   * @param dims_tensor input dims tensor
   * @param dims output shape dims
   * @return status if success
   */
  template <typename T>
  uint32_t CalcDims(const Tensor *dims_tensor, std::vector<int64_t> &dims);

  /**
   * @brief fill output from input value
   * @param ctx cpu kernel context
   * @param output output Tensor
   * @param value input value
   * @return status if success
   */
  template <typename T, int32_t OPTION>
  uint32_t Assign(CpuKernelContext &ctx, const Tensor *output, const T &value);
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_FILL_H_
