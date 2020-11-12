/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of realdiv, z = x / y
 */

#ifndef _AICPU_REALDIV_KERNELS_H_
#define _AICPU_REALDIV_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
class RealDivKernel : public CpuKernel {
public:
    ~RealDivKernel() = default;

    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    template <typename T>
    uint32_t ComputeRealdiv(Tensor *x, Tensor *y, Tensor *z);

    uint32_t ComputeDiffType(Tensor *x, Tensor *y, Tensor *z, DataType dataType);

    template <typename T>
    uint32_t ComputeDiffShape(int64_t dim, T *xAddr, T *yAddr, T *zAddr,
                                         std::vector<int64_t> &xDimSize,
                                         std::vector<int64_t> &yDimSize,
                                         std::vector<int64_t> &zDimSize);

    template <typename T, int32_t dim>
    void DoCompute(T *xAddr, T *yAddr, T *zAddr,
                 std::vector<int64_t> &xDimSize,
                 std::vector<int64_t> &yDimSize,
                 std::vector<int64_t> &zDimSize);
};
}  // namespace aicpu
#endif
