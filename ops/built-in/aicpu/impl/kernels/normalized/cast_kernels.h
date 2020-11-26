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

#ifndef _AICPU_CAST_KERNELS_H_
#define _AICPU_CAST_KERNELS_H_

#include "cpu_kernel.h"
#include "cpu_types.h"

namespace aicpu {
class CastCpuKernel : public CpuKernel {
 public:
  ~CastCpuKernel() = default;
  uint32_t TransferType(int64_t start, int64_t end);
  uint32_t Compute(CpuKernelContext &ctx) override;
  void SetMap();

 private:
  std::map<int, std::map<int, std::function<uint32_t(Tensor *&, Tensor *&,
                                                     int64_t &, int64_t &)>>>
      calls_;
  Tensor *xTensor_;
  Tensor *yTensor_;
  DataType xDataType_;
  DataType yDataType_;
  int64_t xDataSize_ = 1;
  int64_t yDataSize_ = 1;
};
}  // namespace aicpu
#endif
