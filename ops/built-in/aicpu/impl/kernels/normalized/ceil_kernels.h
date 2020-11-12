/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of ceil, y = ceil(x)
 */

#ifndef _AICPU_CEIL_KERNELS_H_
#define _AICPU_CEIL_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
class CeilKernel : public CpuKernel {
public:
    ~CeilKernel() = default;

    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    template <typename T>
    void ComputeCeil(Tensor *x, Tensor *y, uint64_t dataSize);
};
}  // namespace aicpu
#endif
