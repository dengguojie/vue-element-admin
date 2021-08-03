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
#ifndef AICPU_KERNELS_NORMALIZED_GREATER_EQUAL_H_
#define AICPU_KERNELS_NORMALIZED_GREATER_EQUAL_H_

#include "cpu_kernel.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/bcast.h"

namespace aicpu {
template <typename T>
using TensorMap =
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>,
                     Eigen::Aligned>;

class GreaterEqualCpuKernel : public CpuKernel {
 public:
  GreaterEqualCpuKernel() = default;
  ~GreaterEqualCpuKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);

  template <typename T, int32_t RANK>
  uint32_t BroadcastCompute(TensorMap<T> &x, TensorMap<T> &y,
                            TensorMap<bool> &out, Bcast &bcast);
};
}  // namespace aicpu
#endif