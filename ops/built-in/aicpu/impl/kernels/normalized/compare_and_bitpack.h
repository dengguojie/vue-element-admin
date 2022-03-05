/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 */

#ifndef AICPU_KERNELS_DEVICE_COMPARE_AND_BITPACK_H
#define AICPU_KERNELS_DEVICE_COMPARE_AND_BITPACK_H

#include "cpu_kernel.h"
#include "cpu_types.h"

namespace aicpu {
class CompareAndBitpackCpuKernel : public CpuKernel {
 public:
  CompareAndBitpackCpuKernel() = default;
  ~CompareAndBitpackCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t ParaCheck(CpuKernelContext &ctx) const;

  template <typename T>
  uint32_t CompareAndBitpackCompute(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_DEVICE_COMPARE_AND_BITPACK_H
