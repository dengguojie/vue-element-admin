/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: implement of expanddims
 */

#include "expanddims_kernel.h"

#include <securec.h>
#include "log.h"
#include "status.h"


namespace {
const char *EXPANDDIMS = "ExpandDims";
const size_t kExpandDimsOutputDescNum = 1;
const size_t kExpandDimsInputNum = 2;
}

namespace aicpu {
uint32_t ExpandDimsCpuKernel::Compute(CpuKernelContext &ctx)
{
    KERNEL_LOG_INFO("ExpandDims folding kernel in.");
    if ((ctx.GetInputsSize() != kExpandDimsInputNum) || (ctx.GetOutputsSize() != kExpandDimsOutputDescNum)) {
        KERNEL_LOG_WARN("Unexpected ExpandDims node, node input size: %zu, node output size: %zu, node name: %s",
            ctx.GetInputsSize(), ctx.GetOutputsSize(), ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    Tensor *output = ctx.Output(0);
    if (output == nullptr) {
        KERNEL_LOG_ERROR("get output:0 failed");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    uint8_t *output_data =  reinterpret_cast<uint8_t *>(output->GetData());
    uint64_t output_data_size = output->GetDataSize();
    // print output tensor information, and will be deleted
    KERNEL_LOG_INFO("ExpandDims op %s output tensor data size is %llu", ctx.GetOpType().c_str(), output_data_size);
    auto shape = output->GetTensorShape();
    if (shape == nullptr) {
        KERNEL_LOG_ERROR("get tensor shape failed");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    size_t data_dim_size = shape->GetDims();
    KERNEL_LOG_INFO("ExpandDims op %s output tensor dim size is %zu", ctx.GetOpType().c_str(), data_dim_size);

    Tensor *input = ctx.Input(0);
    if (input == nullptr) {
        KERNEL_LOG_ERROR("get input:0 failed");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    uint8_t *input_data = reinterpret_cast<uint8_t *>(input->GetData());
    uint64_t input_data_size = input->GetDataSize();
    KERNEL_LOG_INFO("ExpandDims op %s input tensor input_size is %zu", ctx.GetOpType().c_str(), input_data_size);

    if (output_data_size != input_data_size) {
        KERNEL_LOG_ERROR("input data size:%llu is not equal to output data size:%llu.",
            input_data_size, output_data_size);
        return KERNEL_STATUS_PARAM_INVALID;
    }

    for (uint64_t i = 0; i < input_data_size; i++) {
        output_data[i] = input_data[i];
    }

    KERNEL_LOG_INFO("ExpandDims folding kernel success.");
    return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(EXPANDDIMS, ExpandDimsCpuKernel);
} // namespace aicpu
