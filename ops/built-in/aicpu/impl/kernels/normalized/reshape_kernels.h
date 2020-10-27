/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of reshape
 */

#ifndef _AICPU_RESHAPE_KERNELS_H_
#define _AICPU_RESHAPE_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
struct MatrixInfo {
    std::vector<int> matrix_shape;
    int32_t matrix_type;
};

class ReshapeCpuKernel : public CpuKernel {
public:
    ~ReshapeCpuKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    MatrixInfo matrix_info_;
    size_t input_size_ = 0;
};
} // namespace aicpu
#endif
