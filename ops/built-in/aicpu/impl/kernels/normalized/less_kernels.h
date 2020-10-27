/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of reshape
 */

#ifndef _AICPU_LESS_KERNELS_H_
#define _AICPU_LESS_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {

    class LessKernel : public CpuKernel {
        public:
        ~LessKernel() = default;
        uint32_t Compute(CpuKernelContext &ctx) override;

        private:
        uint32_t GetInputAndCheck(CpuKernelContext &ctx);
        template <typename T> uint32_t DoCompute();
        template <typename T, const int32_t rank> uint32_t DoRealCompute();
        std::vector<int64_t> GetDimSize(std::shared_ptr<TensorShape> input_shape);
        size_t GetSize(std::vector<int64_t> dim_size);
        Tensor *x1_;
        Tensor *x2_;
        Tensor *y_;
        int32_t x_dtype_;
    };
} // namespace aicpu
#endif
