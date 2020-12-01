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
#ifndef _AICPU_TRANSDATA_KERNEL_H_
#define _AICPU_TRANSDATA_KERNEL_H_

#include "cpu_kernel.h"

namespace aicpu {
class TransDataCpuKernel : public CpuKernel {
 public:
  ~TransDataCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;
 private:
  template <typename T>
  uint32_t DealData(T *input_Data, T *output_data, Tensor *input_tensor,
                    Tensor *out_put_tensor, int64_t group);
};
}  // namespace aicpu
#endif