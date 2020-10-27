/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of UpdateCache
 */

#ifndef _AICPU_UPDATE_CACHE_KERNELS_H_
#define _AICPU_UPDATE_CACHE_KERNELS_H_

#include <math.h>
#include <vector>
#include "cpu_kernel.h"

namespace aicpu
{

    class UpdateCacheKernel : public CpuKernel
    {
    public:
        ~UpdateCacheKernel() = default;
        uint32_t Compute(CpuKernelContext &ctx) override;

    protected:
        uint32_t DoCompute();

        uint32_t GetInputAndCheck(CpuKernelContext &ctx);

        int64_t batch_size_ = 1;
        int64_t update_size_ = 1;
        int64_t update_length_ = 1;

        std::vector<Tensor *> inputs_;
        std::vector<Tensor *> outputs_;
        DataType param_type_ = DT_FLOAT;
        DataType indices_type_ = DT_INT32;
    };

} // namespace aicpu
#endif