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

#ifndef _AICPU_AICPU_DEVICE_CPU_KERNELS_KERNELS_ADD_KERNEL_H_
#define _AICPU_AICPU_DEVICE_CPU_KERNELS_KERNELS_ADD_KERNEL_H_
#define EIGEN_USE_THREADS
#define EIGEN_USE_SIMPLE_THREAD_POOL

#include "cpu_kernel.h"
#include "cpu_types.h"
#include "utils/bcast.h"

namespace aicpu {
class AddCpuKernel : public CpuKernel {
 public:
  AddCpuKernel() = default;
  ~AddCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  static uint32_t AddCompute(CpuKernelContext &ctx);

  template <int32_t RANK, typename T>
  static void AddCalculate(CalcInfo &calc_info);
};
}  // namespace aicpu
#endif  // _AICPU_AICPU_DEVICE_CPU_KERNELS_KERNELS_ADD_KERNEL_H_
