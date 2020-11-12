/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of expanddims
 */

#ifndef _AICPU_EXPANDDIMS_KERNELS_H_
#define _AICPU_EXPANDDIMS_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
class ExpandDimsCpuKernel : public CpuKernel {
public:
    ~ExpandDimsCpuKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;
};
} // namespace aicpu
#endif
