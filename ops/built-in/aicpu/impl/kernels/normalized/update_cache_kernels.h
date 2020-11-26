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

#ifndef _AICPU_UPDATE_CACHE_KERNELS_H_
#define _AICPU_UPDATE_CACHE_KERNELS_H_

#include <math.h>
#include <vector>
#include "cpu_kernel.h"

namespace aicpu {

class UpdateCacheCpuKernel : public CpuKernel {
 public:
  ~UpdateCacheCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t DoCompute();

  uint32_t GetInputAndCheck(CpuKernelContext &ctx);

  int64_t batch_size_ = 1;
  int64_t update_size_ = 1;
  int64_t update_length_ = 1;

  std::vector<Tensor *> inputs_;
  std::vector<Tensor *> outputs_;
  DataType param_type_ = DT_FLOAT;
  DataType indices_type_ = DT_INT32;
};

}  // namespace aicpu
#endif