/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 */

#ifndef AICPU_KERNELS_DEVICE_LINSPACE_H
#define AICPU_KERNELS_DEVICE_LINSPACE_H

#include "cpu_kernel.h"
#include "cpu_types.h"
#include "utils/bcast.h"

namespace aicpu {
class LinSpaceCpuKernel : public CpuKernel {
 public:
  LinSpaceCpuKernel() = default;
  ~LinSpaceCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_DEVICE_LINSPACE_H