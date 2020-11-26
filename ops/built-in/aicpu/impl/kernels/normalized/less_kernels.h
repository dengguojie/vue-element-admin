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

#ifndef _AICPU_LESS_KERNELS_H_
#define _AICPU_LESS_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {

class LessCpuKernel : public CpuKernel {
 public:
  ~LessCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t GetInputAndCheck(CpuKernelContext &ctx);
  template <typename T>
  uint32_t DoCompute();
  template <typename T, const int32_t rank>
  uint32_t DoRealCompute();
  std::vector<int64_t> GetDimSize(std::shared_ptr<TensorShape> input_shape);
  size_t GetSize(std::vector<int64_t> dim_size);

 private:
  Tensor *x1_ = nullptr;
  Tensor *x2_ = nullptr;
  Tensor *y_ = nullptr;
  int32_t x_dtype_;
};
}  // namespace aicpu
#endif
