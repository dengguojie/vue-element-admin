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
#ifndef AICPU_KERNELS_NORMALIZED_INPLACE_TOP_K_DISTANCE_H_
#define AICPU_KERNELS_NORMALIZED_INPLACE_TOP_K_DISTANCE_H_

#include "cpu_kernel.h"
#include "utils/bcast.h"
#include <vector>
namespace aicpu {
class InplaceTopKDistanceCpuKernel : public CpuKernel {
 public:
  InplaceTopKDistanceCpuKernel() = default;

  ~InplaceTopKDistanceCpuKernel() override = default;

  uint32_t Compute(CpuKernelContext& ctx) override;

 private:
  class Inputs {
   public:
    Tensor* topk_pq_distance = nullptr;
    Tensor* topk_pq_index = nullptr;
    Tensor* topk_pq_ivf = nullptr;
    Tensor* pq_distance = nullptr;
    Tensor* pq_index = nullptr;
    Tensor* pq_ivf = nullptr;
    AttrValue* order = nullptr;
  };

  template <typename T>
  class Item {
   public:
    T value;
    int32_t index;
    int32_t ivf;
  };

  uint32_t GetInputAndCheck(CpuKernelContext& ctx, Inputs& inputs);

  template <typename T>
  uint32_t DoCompute(CpuKernelContext& ctx, Inputs& inputs);

  template <typename T>
  uint32_t ModifyInput(std::vector<Item<T>> items, Inputs& inputs, CpuKernelContext& ctx);
};
}  // namespace aicpu
#endif
