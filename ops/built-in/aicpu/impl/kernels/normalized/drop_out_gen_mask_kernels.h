/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of drop out gen mask
 */


#ifndef _AICPU_TEST_ADD_KERNELS_H_
#define _AICPU_TEST_ADD_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
class DropOutGenMaskCpuKernel : public CpuKernel {
public:
    ~DropOutGenMaskCpuKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    uint64_t seed0_;
    uint64_t seed1_;
    float keep_prob_;
    uint64_t count_;
    uint8_t *out_;
};
}
#endif
