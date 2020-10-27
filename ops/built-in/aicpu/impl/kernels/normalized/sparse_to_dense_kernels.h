/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of sparse to dense
 */

#ifndef _AICPU_SPARSETODENSE_KERNELS_H_
#define _AICPU_SPARSETODENSE_KERNELS_H_

#include "cpu_kernel.h"
#include "utils/sparse_tensor.h"

namespace aicpu {

class SparseToDenseCpuKernel : public CpuKernel {
public:
    ~SparseToDenseCpuKernel() = default;

protected:

    /*
     * valid sparse to dense param
     * @param st: sparse tensor
     * @param indices: indices tensor
     * @param output: output tensor
     * @return uint32_t: 0->success other->failed
     */
    template <typename ValueT>
    uint32_t EigenSparseToDense(SparseTensor &st, Tensor *indices, Tensor *output)
    {
        if (indices->GetDataType() == DT_INT32) {
            return st.ToDense<int32_t, ValueT>(output);
        } else {
            return st.ToDense<int64_t, ValueT>(output);
        }
    }

    /*
     * valid sparse to dense param
     * @param st: sparse tensor
     * @param indices: indices tensor
     * @param output: output tensor
     * @return uint32_t: 0->success other->failed
     */
    uint32_t SparseToDense(SparseTensor &st, Tensor *indices, Tensor *output);

    /*
     * valid sparse to dense param
     * @param ctx: cpu kernel context
     * @return uint32_t: 0->success other->failed
     */
    uint32_t ValidParam(CpuKernelContext &ctx);

    /*
     * compute sparse to dense
     * @param ctx: cpu kernel context
     * @return uint32_t: 0->success other->failed
     */
    uint32_t Compute(CpuKernelContext &ctx) override;
};

} // namespace aicpu
#endif
