/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of CacheSwapTable
 */

#ifndef _AICPU_CACHE_SWAP_TABLE_KERNELS_H_
#define _AICPU_CACHE_SWAP_TABLE_KERNELS_H_

#include <math.h>
#include <vector>
#include "cpu_kernel.h"

namespace aicpu
{

    class CacheSwapTableKernel : public CpuKernel
    {
    public:
        ~CacheSwapTableKernel() = default;
        uint32_t Compute(CpuKernelContext &ctx) override;

    protected:
        uint32_t DoCompute();

        uint32_t GetInputAndCheck(CpuKernelContext &ctx);

        int64_t batch_size_ = 1;
        int64_t one_line_col_ = 1;
        int64_t output_size_ = 1;

        std::vector<Tensor *> inputs_;
        std::vector<Tensor *> outputs_;
        DataType param_type_ = DT_FLOAT;
        DataType indices_type_ = DT_INT32;
    };

} // namespace aicpu
#endif