/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of reshape
 */

#ifndef _AICPU_UNIQUE_KERNELS_H_
#define _AICPU_UNIQUE_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
    class UniqueWithPadKernel : public CpuKernel {
        public:
        ~UniqueWithPadKernel() = default;
        uint32_t Compute(CpuKernelContext &ctx) override;

        private:
        uint32_t GetInputAndCheck(CpuKernelContext &ctx);
        uint32_t DoCompute();
        template <typename T> uint32_t UniqueWithPadTask();
        Tensor *input_tensor_;
        int64_t p_size_ = 1;
        int32_t matrix_type_;
        Tensor *input_padding_;
        Tensor *output_values_;
        Tensor *output_indices_;
    };
} // namespace aicpu
#endif