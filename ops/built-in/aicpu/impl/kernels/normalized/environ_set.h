/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef AICPU_KERNELS_NORMALIZED_ENVIRON_SET_H
#define AICPU_KERNELS_NORMALIZED_ENVIRON_SET_H

#include <vector>
#include <string>
#include <memory>
#include "cpu_kernel.h"
#include "utils/environ.h"

namespace aicpu {
class EnvironSetCpuKernel : public CpuKernel {
 public:
  EnvironSetCpuKernel() : value_type_attr_(kObjectTypeTensorType), value_size_(0),
                          output_handle_(nullptr), input_handle_(nullptr),
                          input_value_(nullptr) {}
  ~EnvironSetCpuKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t InitKernel(const CpuKernelContext &ctx);

  int32_t value_type_attr_;
  size_t value_size_;
  Tensor *output_handle_;
  Tensor *input_handle_;
  Tensor *input_value_;
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_ENVIRON_SET_H
