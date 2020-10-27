/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: implement of ceil, y = ceil(x)
 */
#include "ceil_kernels.h"

#include <securec.h>
#include <stdint.h>

#include "cpu_types.h"
#include "status.h"
#include "log.h"
#include "Eigen/Dense"

namespace {
    const char *CEIL  = "Ceil";               // op name
    const size_t K_CEIL_OUTPUT_DESC_NUM = 1;  // output dims
    const size_t K_CEIL_INPUT_NUM = 1;        // input dims
}

namespace aicpu {
uint32_t CeilKernel::Compute(CpuKernelContext &ctx)
{
    KERNEL_LOG_INFO("CeilKernel::Compute begin.");

    if ((ctx.GetInputsSize() != K_CEIL_INPUT_NUM) || (ctx.GetOutputsSize() != K_CEIL_OUTPUT_DESC_NUM)) {
        KERNEL_LOG_ERROR("Unexpected Ceil node, node input size: %zu, node output size: %zu, node name: %s",
            ctx.GetInputsSize(), ctx.GetOutputsSize(), ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    Tensor *x = ctx.Input(0);
    if (x == nullptr) {
        KERNEL_LOG_ERROR("Ceil input x is null");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    Tensor *y = ctx.Output(0);
    if (y == nullptr) {
        KERNEL_LOG_ERROR("Ceil output y is null");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    uint64_t dataSize = x->GetDataSize();
    KERNEL_LOG_INFO("Ceil input size:%llu.", dataSize);
    DataType dataType = DataType(x->GetDataType());
    KERNEL_LOG_INFO("Ceil input type:%d", dataType);

    switch (dataType) {
        case DT_FLOAT16:
            ComputeCeil<Eigen::half>(x, y, dataSize);
            break;
        case DT_FLOAT:
            ComputeCeil<float>(x, y, dataSize);
            break;
        case DT_DOUBLE:
            ComputeCeil<double>(x, y, dataSize);
            break;
        default:
            KERNEL_LOG_ERROR("Ceil invalid input type:%d", dataType);
            break;
    }

    KERNEL_LOG_INFO("CeilKernel::Compute end.");
    return KERNEL_STATUS_OK;
}

template <typename T>
void CeilKernel::ComputeCeil(Tensor *x, Tensor *y, uint64_t dataSize)
{
    auto xAddr = x->GetData();
    auto yAddr = y->GetData();
    // 1 represents row vector
    Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>> mapX((T *)xAddr, dataSize / sizeof(T));
    Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>> mapY((T *)yAddr, dataSize / sizeof(T));
    mapY=mapX.array().ceil().matrix();
    KERNEL_LOG_INFO("CeilKernel::Compute success.");
}

REGISTER_CPU_KERNEL(CEIL, CeilKernel);
}  // namespace aicpu
