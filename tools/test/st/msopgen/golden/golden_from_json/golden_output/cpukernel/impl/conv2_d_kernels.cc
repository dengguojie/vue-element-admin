
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: implement of Conv2D
 */
#include "conv2_d_kernels.h"

namespace  {
const char *CONV2_D = "Conv2D";
}

namespace aicpu  {
uint32_t Conv2DCpuKernel::Compute(CpuKernelContext &ctx)
{
    return 0;
}

REGISTER_CPU_KERNEL(CONV2_D, Conv2DCpuKernel);
} // namespace aicpu
