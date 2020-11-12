/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of reshape
 */

#ifndef _AICPU_TOPKV2_KERNELS_H_
#define _AICPU_TOPKV2_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
    struct MatrixInfo {
        std::vector<int> matrix_shape;
        int32_t matrix_type;
    };

    class TopKV2CpuKernel : public CpuKernel {
        public:
        ~TopKV2CpuKernel() = default;
        uint32_t Compute(CpuKernelContext &ctx) override;

        private:
        uint32_t GetInputAndCheck(CpuKernelContext &ctx);
        template <typename T> uint32_t DoCompute(T data_type);
        int32_t k_;
        bool sorted_;
        Tensor *input_tensor_;
        Tensor *output_values_;
        Tensor *output_indices_;
        MatrixInfo matrix_info_;
        int32_t col_ = 0;
        int32_t row_ = 0;
    };
} // namespace aicpu
#endif
