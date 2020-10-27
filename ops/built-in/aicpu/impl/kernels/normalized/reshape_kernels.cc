/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: implement of reshape
 */

#include "reshape_kernels.h"

#include <securec.h>
#include "cpu_types.h"
#include "log.h"
#include "status.h"


namespace {
const char *RESHAPE = "Reshape";
}

namespace aicpu {
uint32_t ReshapeCpuKernel::Compute(CpuKernelContext &ctx)
{
    Tensor *input_tensor = ctx.Input(0);
    if (input_tensor == nullptr) {
        KERNEL_LOG_ERROR("get input:0 failed");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    Tensor *output_tensor = ctx.Output(0);
    if (output_tensor == nullptr) {
        KERNEL_LOG_ERROR("get output:0 failed");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    auto input_data = input_tensor->GetData();
    auto output_data = output_tensor->GetData();
    auto input_shape = input_tensor->GetTensorShape();
    if (input_shape == nullptr) {
        KERNEL_LOG_ERROR("get input_shape failed");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    input_size_ = 1;

    matrix_info_.matrix_type = input_tensor->GetDataType();
    for (int i = 0; i < input_shape->GetDims(); ++i) {
        matrix_info_.matrix_shape.push_back(input_shape->GetDimSize(i));
        input_size_ *= input_shape->GetDimSize(i);
    }

    size_t type_size = GetSizeByDataType(static_cast<DataType>(matrix_info_.matrix_type));
    if (type_size < 1) {
        KERNEL_LOG_ERROR("don't support input tensor types");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    int cpyRet = memcpy_s(output_data, input_size_ * type_size, input_data, input_size_ * type_size);
    if (cpyRet < 0) {
        return KERNEL_STATUS_INNER_ERROR;
    }
    return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(RESHAPE, ReshapeCpuKernel);
} // namespace aicpu
