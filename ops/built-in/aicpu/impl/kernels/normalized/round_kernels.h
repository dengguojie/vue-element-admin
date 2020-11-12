/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of round
 */

#ifndef _AICPU_ROUND_KERNELS_H_
#define _AICPU_ROUND_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
class RoundCpuKernel : public CpuKernel {
 public:
  ~RoundCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  bool CheckSupported(DataType input_type);
};
}  // namespace aicpu
#endif
