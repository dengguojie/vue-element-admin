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

#ifndef AICPU_KERNELS_NORMALIZED_ADDCMUL_H_
#define AICPU_KERNELS_NORMALIZED_ADDCMUL_H_

#include "cpu_kernel.h"
#include "utils/bcast.h"

namespace aicpu {
struct CalcInfo {
  bool isNeedBcast;
  Bcast *bcast;
  int64_t elements_nums;
  BcastShapeType type;
};
class AddcmulCpuKernel : public CpuKernel {
 public:
  AddcmulCpuKernel() = default;
  ~AddcmulCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t AddcmulParamCheck(CpuKernelContext &ctx);

  template <typename T>
  void SpecialCompute(BcastShapeType type, int64_t start, int64_t end,
                      CpuKernelContext &ctx, CalcInfo &mul_info);

  template <typename T>
  void AddBcastCompute(Bcast &add_bcast, int64_t start, int64_t end,
                       CpuKernelContext &ctx, CalcInfo &mul_info);

  template <typename T>
  uint32_t NoBcastCompute(CpuKernelContext &ctx, CalcInfo &mul_info);

  template <typename T>
  uint32_t BcastCompute(CpuKernelContext &ctx, Bcast &bcast,
                        CalcInfo &mul_info);

  template <typename T>
  uint32_t AddcmulCompute(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
