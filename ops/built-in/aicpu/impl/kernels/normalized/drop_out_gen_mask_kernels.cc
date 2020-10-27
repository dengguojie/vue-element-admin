/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: implement of drop out gen mask
 */

#include "drop_out_gen_mask_kernels.h"

#include <cfloat>
#include <ctime>
#include <random>
#include <memory.h>

#include "cpu_types.h"
#include "status.h"
#include "log.h"

namespace {
const char *DropOutGenMask = "DropOutGenMask";
}

namespace aicpu {
std::random_device e;

uint32_t DropOutGenMaskCpuKernel::Compute(CpuKernelContext &ctx)
{
    AttrValue *seed0 = ctx.GetAttr("seed");
    KERNEL_CHECK_NULLPTR(seed0, KERNEL_STATUS_PARAM_INVALID, "get attr:seed failed.")

    AttrValue *seed1 = ctx.GetAttr("seed2");
    KERNEL_CHECK_NULLPTR(seed1, KERNEL_STATUS_PARAM_INVALID, "get attr:seed2 failed.")

    seed0_ = seed0->GetInt();
    seed1_ = seed1->GetInt();
    if (seed0_ == 0 && seed1_ == 0) {
        seed0_ = e();
        seed1_ = e();
    }
    uint64_t tmp_count = 1;
    Tensor *shape_tensor = ctx.Input(0);
    KERNEL_CHECK_NULLPTR(shape_tensor, KERNEL_STATUS_PARAM_INVALID, "get input:0 failed.")

    auto input_shape = shape_tensor->GetTensorShape();
    KERNEL_CHECK_NULLPTR(input_shape, KERNEL_STATUS_PARAM_INVALID, "get input_shape failed.")

    DataType shape_dt = static_cast<DataType>(shape_tensor->GetDataType());
    for (int j = 0; j < input_shape->GetDims(); j++) {
        tmp_count *= input_shape->GetDimSize(j);
    }

    if (shape_dt == DT_INT32) {
        auto input0 = reinterpret_cast<int32_t *>(shape_tensor->GetData());
        count_ = 1;
        for (uint64_t index = 0; index < tmp_count; index++) {
            count_ *= input0[index];
        }
    } else {
        auto input0 = reinterpret_cast<int64_t *>(shape_tensor->GetData());
        count_ = 1;
        for (uint64_t index = 0; index < tmp_count; index++) {
            count_ *= input0[index];
        }
    }

    Tensor *prob_tensor = ctx.Input(1);
    KERNEL_CHECK_NULLPTR(prob_tensor, KERNEL_STATUS_PARAM_INVALID, "get input:1 failed.")

    DataType dt = static_cast<DataType>(prob_tensor->GetDataType());
    if (dt == DT_FLOAT16) {
        keep_prob_ = *reinterpret_cast<float *>(prob_tensor->GetData());
    } else {
        keep_prob_ = *reinterpret_cast<float *>(prob_tensor->GetData());
    }
    KERNEL_LOG_INFO("DropOutGenMask mask count and pro: %d %f", count_, keep_prob_);

    Tensor *out_tensor = ctx.Output(0);
    KERNEL_CHECK_NULLPTR(out_tensor, KERNEL_STATUS_PARAM_INVALID, "get output:0 failed.")

    std::default_random_engine e(time(0));
    std::bernoulli_distribution b(keep_prob_);
    const uint8_t mask[8] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};
    uint64_t byteCount = count_ >> 3;
    out_ = reinterpret_cast<uint8_t *>(out_tensor->GetData());
    for (uint64_t i = 0; i < byteCount; ++i) {
        out_[i] = 0x00;
        for (const auto &m : mask) {
            if (b(e)) {
                out_[i] = out_[i] | m;
            }
        }
    }
    out_ = nullptr;
    return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(DropOutGenMask, DropOutGenMaskCpuKernel);
} // namespace aicpu
