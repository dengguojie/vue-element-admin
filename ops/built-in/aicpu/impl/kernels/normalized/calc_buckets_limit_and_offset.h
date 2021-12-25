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

#ifndef AICPU_KERNELS_NORMALIZED_CALC_BUCKETS_LIMIT_AND_OFFSET_H
#define AICPU_KERNELS_NORMALIZED_CALC_BUCKETS_LIMIT_AND_OFFSET_H

#include <vector>

#include "cpu_kernel.h"

namespace {
const char *kCalcBucketsLimitAndOffset = "CalcBucketsLimitAndOffset";
const uint32_t kInputNum = 3;
const uint32_t kOutputNum = 2;
}  // namespace

namespace aicpu {

class CalcBucketsLimitAndOffsetCpuKernel : public CpuKernel {
 public:
  CalcBucketsLimitAndOffsetCpuKernel() = default;
  ~CalcBucketsLimitAndOffsetCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t InitParams(CpuKernelContext &ctx);
  template <typename T>
  uint32_t DoCompute();

 private:
  int64_t input_num_elements_[kInputNum]{0};
  void *datas_[kInputNum + kOutputNum]{nullptr};
  int64_t total_limit_{0};
};
}  // namespace aicpu
#endif
