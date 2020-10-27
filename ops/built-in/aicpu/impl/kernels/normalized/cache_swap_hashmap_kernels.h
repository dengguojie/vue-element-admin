/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of CacheSwapHashmap
 */

#ifndef _AICPU_CACHE_SWAP_HASHMAP_KERNELS_H_
#define _AICPU_CACHE_SWAP_HASHMAP_KERNELS_H_

#include <math.h>
#include <vector>
#include "cpu_kernel.h"
#include "search_cache_idx_kernels.h"

namespace aicpu
{

    class CacheSwapHashmapKernel : public CpuKernel
    {
    public:
        ~CacheSwapHashmapKernel() = default;
        uint32_t Compute(CpuKernelContext &ctx) override;

    protected:
        uint32_t DoCompute();

        uint32_t GetInputAndCheck(CpuKernelContext &ctx);

        int64_t batch_size_ = 1;
        int64_t hashmap_length_ = 1;

        std::vector<Tensor *> inputs_;
        std::vector<Tensor *> outputs_;
        DataType param_type_ = DT_INT32;
    };

} // namespace aicpu
#endif