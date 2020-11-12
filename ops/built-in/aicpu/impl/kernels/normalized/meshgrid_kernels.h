/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of reshape
 */

#ifndef _AICPU_MESHGRID_KERNELS_H_
#define _AICPU_MESHGRID_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {

    class MeshgridCpuKernel : public CpuKernel {
        public:
        ~MeshgridCpuKernel() = default;
        uint32_t Compute(CpuKernelContext &ctx) override;

        private:
        uint32_t GetInputAndCheck(CpuKernelContext &ctx);
        template <typename T> uint32_t DoCompute(T data_type);

        std::vector<void *> ioAddrs_;
        DataType input_type_ = DT_INT32;
        std::string indexing_;
        size_t ndim_ = 0;
        std::vector<int> bcast_;
    };
} // namespace aicpu
#endif
