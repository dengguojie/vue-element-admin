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

#ifndef _AICPU_LOGGING_KERNELS_H_
#define _AICPU_LOGGING_KERNELS_H_

#include <string>
#include "cpu_kernel.h"
namespace aicpu {
class AssertCpuKernel : public CpuKernel {
 public:
  ~AssertCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t summarize_ = 0;
};

static std::string SummarizeValue(Tensor &t, int64_t max_entries,
                                  bool print_v2 = false);
template <typename T>
static std::string SummarizeArray(int64_t limit, int64_t num_elts, Tensor &t,
                                  bool print_v2);
template <typename T>
void PrintOneDim(int dim_index, std::shared_ptr<TensorShape> shape,
                 int64_t limit, int shape_size, const T *data,
                 int64_t *data_index, std::string &result);
}

#endif  //_AICPU_LOGGING_KERNELS_H_
