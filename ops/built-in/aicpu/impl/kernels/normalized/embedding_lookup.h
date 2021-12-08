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
#ifndef AICPU_KERNELS_NORMALIZED_EMBEDDING_LOOKUP_H_
#define AICPU_KERNELS_NORMALIZED_EMBEDDING_LOOKUP_H_

#include "cpu_kernel.h"

namespace aicpu {
class EmbeddingLookuptMsCpuKernel : public CpuKernel {
 public:
  ~EmbeddingLookuptMsCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t GetInputAndCheck(CpuKernelContext &ctx);
  uint32_t GetInput(CpuKernelContext &ctx);

  template <typename T>
  uint32_t DoComputeForEachType(CpuKernelContext &ctx);

  std::vector<void *> ioAddrs_;
  int64_t offset_ = 0;
  int64_t out_size_ = 1;
  int64_t p_size_ = 1;
  int64_t i_size_ = 1;
  int64_t val_size_ = 1;
  int64_t axis_size_ = 1;
  DataType param_type_ = DT_INT32;
  DataType index_type_ = DT_INT32;
};
}  // namespace aicpu
#endif
