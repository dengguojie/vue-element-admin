/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of reshape
 */

#ifndef _AICPU_GATHER_KERNELS_H_
#define _AICPU_GATHER_KERNELS_H_

#include <vector>
#include "cpu_kernel.h"

namespace aicpu {
class GatherKernel : public CpuKernel {
public:
    ~GatherKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

protected:
    uint32_t DoCompute();

    uint32_t GetInputAndCheck(CpuKernelContext &ctx);

    std::vector<Tensor *> inputs_;
    std::vector<Tensor *> outputs_;
    DataType param_type_ = DT_INT32;
    DataType index_type_ = DT_INT32;
};

} // namespace aicpu
#endif
