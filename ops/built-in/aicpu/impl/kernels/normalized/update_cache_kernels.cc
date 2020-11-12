/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: implement of update_cache_kernels.h
 */

#include <map>
#include <chrono>
#include <securec.h>
#include "update_cache_kernels.h"
#include "utils/sparse_tensor.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"

namespace
{
    const char *UPDATE_CACHE = "UpdateCache";
}

namespace aicpu
{

    template <typename T>
    uint32_t UpdateCacheTask(std::vector<Tensor *> &inputs_, std::vector<Tensor *> &outputs_, int64_t &batch_size_,
                             int64_t &update_length_, int &type_size)
    {
        auto start = std::chrono::high_resolution_clock::now();

        if (inputs_.size() == 0 || outputs_.size() == 0)
        {
            KERNEL_LOG_ERROR("UpdateCacheKernel::UpdateCacheTask: input or output is empty.");
            return KERNEL_STATUS_PARAM_INVALID;
        }

        char *input_x = reinterpret_cast<char *>(inputs_[0]->GetData());
        T *indices = reinterpret_cast<T *>(inputs_[1]->GetData());      //16000*39
        char *update = reinterpret_cast<char *>(inputs_[2]->GetData()); //16000*39*80
        T max_num = *reinterpret_cast<T *>(inputs_[3]->GetData());

        int64_t one_length_size = type_size * update_length_;
        KERNEL_LOG_INFO("UpdateCache one_length_size %d.", one_length_size);

        for (int64_t i = 0; i < batch_size_; ++i)
        {
            if (indices[i] < 0 || indices[i] >= max_num)
                continue;
            char *tmp = update + i * one_length_size;
            int ret = memcpy_s(input_x + indices[i] * one_length_size, one_length_size, tmp, one_length_size);
            if (ret != 0)
            {
                KERNEL_LOG_ERROR("UpdateCache memcpy failed, result %d.", ret);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();

        KERNEL_LOG_INFO("UpdateCache execute %fms.", std::chrono::duration<double, std::milli>(end - start).count());
        return KERNEL_STATUS_OK;
    }

    uint32_t UpdateCacheKernel::DoCompute()
    {
        std::map<int, std::function<uint32_t(std::vector<Tensor *> &, std::vector<Tensor *> &, int64_t &, int64_t &, int &)>> calls;
        calls[DT_INT32] = UpdateCacheTask<int32_t>;
        calls[DT_INT64] = UpdateCacheTask<int64_t>;

        if (calls.find(indices_type_) == calls.end())
        {
            KERNEL_LOG_ERROR("UpdateCacheKernel op don't support indices tensor types: %s", typeid(indices_type_).name());
            return KERNEL_STATUS_PARAM_INVALID;
        }

        int type_size = 4;
        if (param_type_ == DT_FLOAT || param_type_ == DT_INT32)
            type_size = 4;
        if (param_type_ == DT_DOUBLE || param_type_ == DT_INT64)
            type_size = 8;

        return calls[indices_type_](inputs_, outputs_, batch_size_, update_length_, type_size);
    }

    uint32_t UpdateCacheKernel::GetInputAndCheck(CpuKernelContext &ctx)
    {
        KERNEL_LOG_INFO("UpdateCacheKernel::GetInputAndCheck start!");

        // get input Tensors
        const int num_input = 4;
        for (int i = 0; i < num_input; ++i)
        {
            Tensor *tensor = ctx.Input(i);
            if (tensor == nullptr)
            {
                KERNEL_LOG_ERROR("UpdateCacheKernel::GetInputAndCheck: get input tensor[%d] failed", i);
                return KERNEL_STATUS_PARAM_INVALID;
            }
            inputs_.push_back(tensor);
        }
        // get output Tensors
        const int num_output = 1;
        for (int i = 0; i < num_output; ++i)
        {
            Tensor *tensor = ctx.Output(i);
            if (tensor == nullptr)
            {
                KERNEL_LOG_ERROR("UpdateCacheKernel::GetInputAndCheck: get output tensor[%d] failed", i);
                return KERNEL_STATUS_PARAM_INVALID;
            }
            outputs_.push_back(tensor);
        }
        // get param type
        param_type_ = static_cast<DataType>(inputs_[0]->GetDataType());
        indices_type_ = static_cast<DataType>(inputs_[1]->GetDataType());
        KERNEL_LOG_INFO("UpdateCacheKernel::GetInputAndCheck success!");

        std::shared_ptr<TensorShape> param_shape = ctx.Input(0)->GetTensorShape();
        std::shared_ptr<TensorShape> indices_shape = ctx.Input(1)->GetTensorShape();
        std::shared_ptr<TensorShape> update_shape = ctx.Input(2)->GetTensorShape();

        // 16000*39*80
        for (int i = 0; i < update_shape->GetDims(); ++i)
        {
            update_size_ *= update_shape->GetDimSize(i);
        }

        for (int i = 0; i < indices_shape->GetDims(); ++i)
        {
            batch_size_ *= indices_shape->GetDimSize(i);
        }

        update_length_ = update_size_ / batch_size_; // 80
        return KERNEL_STATUS_OK;
    }

    uint32_t UpdateCacheKernel::Compute(CpuKernelContext &ctx)
    {
        KERNEL_LOG_INFO("UpdateCacheKernel::Compute start!!");

        uint32_t res = GetInputAndCheck(ctx);
        if (res != KERNEL_STATUS_OK)
        {
            return res;
        }

        res = DoCompute();
        if (res != KERNEL_STATUS_OK)
        {
            KERNEL_LOG_ERROR("UpdateCacheKernel::Compute failed");
            return res;
        }

        KERNEL_LOG_INFO("UpdateCacheKernel::Compute success!!");
        return KERNEL_STATUS_OK;
    }
    REGISTER_CPU_KERNEL(UPDATE_CACHE, UpdateCacheKernel);
} // namespace aicpu
