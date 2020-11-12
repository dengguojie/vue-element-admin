/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of SearchCacheIdx
 */

#ifndef _AICPU_SEARCH_CACHE_IDX_KERNELS_H_
#define _AICPU_SEARCH_CACHE_IDX_KERNELS_H_

#include <math.h>
#include <vector>
#include "cpu_kernel.h"

#define NULLTAG 0

namespace aicpu
{

    template <typename T>
    struct HashmapEntry
    {
        T key;
        T value;
        T step;
        T tag;

        bool IsEmpty()
        {
            if (this->tag == NULLTAG)
                return true;
            else
                return false;
        }

        bool IsUsing(const T &train_step)
        {
            if (this->step >= (train_step - 1))
                return true;
            else
                return false;
        }

        bool IsKey(const T &emb_idx)
        {
            if (this->key == emb_idx)
                return true;
            else
                return false;
        }

        void SetEmpty()
        {
            this->tag = NULLTAG;
        }
    };

    template <class T>
    T HashFunc(const T &key, const int64_t &length)
    {
        return (T)(((0.6180339 * key) - floor(0.6180339 * key)) * length);
    }

    class SearchCacheIdxKernel : public CpuKernel
    {
    public:
        ~SearchCacheIdxKernel() = default;
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