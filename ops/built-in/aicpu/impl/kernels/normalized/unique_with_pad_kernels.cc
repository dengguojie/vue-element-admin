/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: implement of reshape
 */

#include "unique_with_pad_kernels.h"

#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include <ctime>
#include <memory.h>
#include <unordered_map>

using std::string;

namespace {
const char *UNIQUE = "UniqueWithPad";
}

namespace aicpu {
uint32_t UniqueWithPadKernel::Compute(CpuKernelContext &ctx)
{
    KERNEL_LOG_INFO("UniqueWithPadKernel::Compute start!!");

    uint32_t res = GetInputAndCheck(ctx);
    if (res != KERNEL_STATUS_OK) {
        return res;
    }

    res = DoCompute();
    if (res != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("UniqueWithPadKernel::Compute failed");
        return res;
    }

    KERNEL_LOG_INFO("UniqueWithPadKernel::Compute success!!");
    return KERNEL_STATUS_OK;
}

uint32_t UniqueWithPadKernel::DoCompute()
{
    uint32_t res;
    switch (matrix_type_) {
        case DT_INT32: {
            res = UniqueWithPadTask<int32_t>();
            break;
        }
        case DT_INT64: {
            res = UniqueWithPadTask<int64_t>();
            break;
        }
        default: {
            KERNEL_LOG_ERROR("UniqueWithPad op don't support input tensor types: %s", typeid(matrix_type_).name());
            return KERNEL_STATUS_PARAM_INVALID;
        }
    }

    return res;
}

template <typename T>
uint32_t UniqueWithPadKernel::UniqueWithPadTask()
{
    clock_t start, end;
    start = clock();
    T *a = reinterpret_cast<T *>(input_tensor_->GetData());
    T padding = *static_cast<T *>(input_padding_->GetData());
    T *out = reinterpret_cast<T *>(output_values_->GetData());
    T *idx_vec = reinterpret_cast<T *>(output_indices_->GetData());
    for (int64_t i = 0; i < p_size_; ++i) {
        out[i] = padding;
    }
    std::unordered_map<T, int> uniq;
    uniq.reserve(2 * p_size_);
    for (int64_t i = 0, j = 0; i < p_size_; ++i) {
        auto it = uniq.emplace(a[i], j);
        idx_vec[i] = it.first->second;
        if (it.second) {
            ++j;
        }
    }
    for (const auto &it : uniq) {
        out[it.second] = it.first;
    }
    end = clock();
    KERNEL_LOG_INFO("UniqueWithPad execute %f ms.", (float)(end - start) * 1000 / CLOCKS_PER_SEC);
    return KERNEL_STATUS_OK;
}

uint32_t UniqueWithPadKernel::GetInputAndCheck(CpuKernelContext &ctx)
{

    KERNEL_LOG_INFO("UniqueWithPadKernel::GetInputAndCheck start!! ");

    // get input_tensor
    input_tensor_ = ctx.Input(0);
    if (input_tensor_ == nullptr) {
        KERNEL_LOG_ERROR("get input:0 failed");
        KERNEL_LOG_INFO("UniqueWithPadKernel::GetInputAndCheck failed!! ");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    std::shared_ptr<TensorShape> input_shape = input_tensor_->GetTensorShape();
    int32_t input_rank = input_shape->GetDims();
    for (int32_t i = 0; i < input_rank; ++i) {
        p_size_ *= input_shape->GetDimSize(i);
    }
    matrix_type_ = static_cast<DataType>(input_tensor_->GetDataType());

    // get padding
    input_padding_ = ctx.Input(1);
    if (input_padding_ == nullptr) {
        KERNEL_LOG_ERROR("get input:1 failed");
        KERNEL_LOG_INFO("UniqueWithPadKernel::GetInputAndCheck failed!! ");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    //get output
    output_values_ = ctx.Output(0);
    if (output_values_ == nullptr) {
        KERNEL_LOG_ERROR("get output:0 failed");
        KERNEL_LOG_INFO("UniqueWithPadKernel::GetInputAndCheck failed!! ");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    output_indices_ = ctx.Output(1);
    if (output_indices_ == nullptr) {
        KERNEL_LOG_ERROR("get output:1 failed");
        KERNEL_LOG_INFO("UniqueWithPadKernel::GetInputAndCheck failed!! ");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    KERNEL_LOG_INFO("UniqueWithPadKernel::GetInputAndCheck success!! ");

    return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(UNIQUE, UniqueWithPadKernel);
} // namespace aicpu