
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: api of Conv2D
 */

#ifndef _CONV2D_KERNELS_H_
#define _CONV2D_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
class Conv2DCpuKernel : public CpuKernel {
public:
    ~Conv2DCpuKernel() = default;
    virtual uint32_t Compute(CpuKernelContext &ctx) override;
};
} // namespace aicpu
#endif
