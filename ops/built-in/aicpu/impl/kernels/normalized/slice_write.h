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
#ifndef AICPU_KERNELS_NORMALIZED_SLICE_WRITE_H_
#define AICPU_KERNELS_NORMALIZED_SLICE_WRITE_H_

#include "cpu_kernel.h"

namespace aicpu {
class SliceWriteCpuKernel : public CpuKernel {
 public:
  SliceWriteCpuKernel() = default;
  ~SliceWriteCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

private:
  bool CheckValueSupported(DataType input_x_type, DataType input_value_type);
  uint32_t Check(const Tensor *x, const Tensor *value,
    int64_t row_offset, int64_t col_offset);
  uint32_t GetBeginValue(const Tensor *begin, int64_t &row_offset,
    int64_t &col_offset);

};
}  // namespace aicpu
#endif
