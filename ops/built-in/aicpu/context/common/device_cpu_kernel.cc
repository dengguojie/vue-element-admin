/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: implement of device cpu kernel
 */

#include "device_cpu_kernel.h"

#include <limits.h>
#include "aicpu_task_struct.h"
#include "cce/fwk_adpt_struct.h"
#include "status.h"
#include "log.h"
#include "cpu_kernel_register.h"
#include "cpu_kernel_utils.h"
#include "cpu_kernel.h"
#include "cpu_node_def.h"

using namespace aicpu;
namespace {
// max param len limit 10k.
constexpr uint32_t MAX_PARAM_LEN = 10240;
// max io address num limit 1024
constexpr uint32_t MAX_IO_ADDR_NUMPARAM_LEN = 1024;

uint32_t ParseIoAddrAndNodeDefStr(AicpuParamHead *paramHead, std::vector<uint64_t> &ioAddrs, std::string &nodeDefStr)
{
    auto paramBase = reinterpret_cast<uint8_t *>(paramHead);
    uint8_t *extendParamBase = paramBase + sizeof(AicpuParamHead);
    uint32_t extendParamLen = paramHead->length - sizeof(AicpuParamHead);

    if (paramHead->ioAddrNum > 0) {
        if (paramHead->ioAddrNum > MAX_IO_ADDR_NUMPARAM_LEN) {
            KERNEL_LOG_ERROR("param ioAddrNum=%u is over %u.", paramHead->ioAddrNum, MAX_IO_ADDR_NUMPARAM_LEN);
            return KERNEL_STATUS_PARAM_INVALID;
        }

        uint32_t addrLen = paramHead->ioAddrNum * sizeof(uint64_t);
        if (extendParamLen < addrLen) {
            KERNEL_LOG_ERROR("extend param is not enough for io addr, ioAddrNum=%u, extendParamLen=%u.",
                paramHead->ioAddrNum, extendParamLen);
            return KERNEL_STATUS_PARAM_INVALID;
        }

        auto ioAddrBase = reinterpret_cast<uint64_t *>(extendParamBase);
        for (uint32_t i = 0; i < paramHead->ioAddrNum; ++i) {
            ioAddrs.push_back(ioAddrBase[i]);
        }
        extendParamBase = extendParamBase + addrLen;
        extendParamLen -= addrLen;
    }

    if (extendParamLen < sizeof(uint32_t)) {
        KERNEL_LOG_ERROR("extend param is not enough for addr, needLen=%u, extendParamLen=%u.", sizeof(uint32_t),
            extendParamLen);
        return KERNEL_STATUS_PARAM_INVALID;
    }

    uint32_t nodeDefLen = *reinterpret_cast<uint32_t *>(extendParamBase);
    extendParamBase += sizeof(uint32_t);
    nodeDefStr = std::string(reinterpret_cast<char *>(extendParamBase), nodeDefLen);
    return KERNEL_STATUS_OK;
}

uint32_t ParseExtShapeType(const FWKAdapter::ExtInfo *extInfo, bool &unknownShape)
{
    if (extInfo->infoLen != sizeof(int32_t)) {
        KERNEL_LOG_ERROR("parse ext shape type failed as infoLen must be %zu but %u.", sizeof(int32_t),
            extInfo->infoLen);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    unknownShape = true;
    KERNEL_LOG_INFO("Kernel has unknown shape.");
    return KERNEL_STATUS_OK;
}

uint32_t ParseExtInputShape(const CpuKernelContext &ctx, FWKAdapter::ExtInfo *extInfo,
    std::vector<FWKAdapter::ShapeAndType *> &inputShapeAndType)
{
    // no overflow
    uint32_t inputsSize = ctx.GetInputsSize();
    KERNEL_LOG_INFO("Parse extend input shape, input size:%u.", inputsSize);
    auto needLen = inputsSize * sizeof(FWKAdapter::ShapeAndType);
    if (extInfo->infoLen != needLen) {
        KERNEL_LOG_ERROR("parse ext input shape failed, as infoLen must be input_num[%d] * sizeof(ShapeAndType)[%zu],"
            " but %u.",
            inputsSize, sizeof(FWKAdapter::ShapeAndType), extInfo->infoLen);
        return KERNEL_STATUS_PARAM_INVALID;
    }

    auto input = reinterpret_cast<FWKAdapter::ShapeAndType *>(extInfo->infoMsg);
    for (uint32_t index = 0; index < inputsSize; ++index) {
        inputShapeAndType.emplace_back(&input[index]);
    }
    return KERNEL_STATUS_OK;
}

uint32_t ParseExtOutputShape(const CpuKernelContext &ctx, FWKAdapter::ExtInfo *extInfo,
    std::vector<FWKAdapter::ShapeAndType *> &outputShapeAndType)
{
    // no overflow
    uint32_t outputsSize = ctx.GetOutputsSize();
    KERNEL_LOG_INFO("Parse extend output shape, output size:%u.", outputsSize);
    auto needLen = outputsSize * sizeof(FWKAdapter::ShapeAndType);
    if (extInfo->infoLen != needLen) {
        KERNEL_LOG_ERROR("parse ext output shape failed as infoLen must be output num[%d] * sizeof(ShapeAndType)[%zu], "
            "but %u.", outputsSize, sizeof(FWKAdapter::ShapeAndType), extInfo->infoLen);
        return KERNEL_STATUS_PARAM_INVALID;
    }

    auto output = reinterpret_cast<FWKAdapter::ShapeAndType *>(extInfo->infoMsg);
    for (uint32_t index = 0; index < outputsSize; ++index) {
        outputShapeAndType.emplace_back(&output[index]);
    }
    return KERNEL_STATUS_OK;
}

uint32_t UpdateInputShape(CpuKernelContext &ctx, const std::vector<FWKAdapter::ShapeAndType *> &inputShapeAndType)
{
    for (uint32_t i = 0; i < ctx.GetInputsSize(); ++i) {
        std::vector<int64_t> dims;
        for (uint32_t index = 0; index < FWKAdapter::kMaxShapeDims; ++index) {
            // LLONG_MIN for dim end flag
            if (inputShapeAndType[i]->dims[index] == LLONG_MIN) {
                break;
            }
            int64_t dimValue = inputShapeAndType[i]->dims[index];
            KERNEL_LOG_INFO("Update extend input[%u] shape[%u]=%lld.", i, index, dimValue);
            dims.emplace_back(dimValue);
        }

        Tensor *input = ctx.Input(i);
        if (input == nullptr) {
            KERNEL_LOG_ERROR("get input:%u failed.", i);
            return KERNEL_STATUS_PARAM_INVALID;
        }

        auto shape = input->GetTensorShape();
        if (shape == nullptr) {
            KERNEL_LOG_ERROR("get input:%u shape failed.", i);
            return KERNEL_STATUS_PARAM_INVALID;
        }

        shape->SetDimSizes(dims);
    }
    return KERNEL_STATUS_OK;
}

uint32_t UpdateOutputShape(CpuKernelContext &ctx, const std::vector<FWKAdapter::ShapeAndType *> &outputShapeAndType)
{
    for (uint32_t i = 0; i < ctx.GetOutputsSize(); ++i) {
        std::vector<int64_t> dims;
        for (uint32_t index = 0; index < FWKAdapter::kMaxShapeDims; ++index) {
            // LLONG_MIN for dim end flag
            if (outputShapeAndType[i]->dims[index] == LLONG_MIN) {
                break;
            }
            int64_t dimValue = outputShapeAndType[i]->dims[index];
            KERNEL_LOG_INFO("Update extend output[%u] shape[%u]=%lld.", i, index, dimValue);
            dims.emplace_back(dimValue);
        }

        Tensor *output = ctx.Output(i);
        if (output == nullptr) {
            KERNEL_LOG_ERROR("get output:%u failed.", i);
            return KERNEL_STATUS_PARAM_INVALID;
        }

        auto shape = output->GetTensorShape();
        if (shape == nullptr) {
            KERNEL_LOG_ERROR("get output:%u shape failed.", i);
            return KERNEL_STATUS_PARAM_INVALID;
        }

        shape->SetDimSizes(dims);
    }
    return KERNEL_STATUS_OK;
}

uint32_t UpdateInputAndOutputShape(CpuKernelContext &ctx, bool unknownShape,
    const std::vector<FWKAdapter::ShapeAndType *> &inputShapeAndType,
    const std::vector<FWKAdapter::ShapeAndType *> &outputShapeAndType)
{
    if (unknownShape) {
        uint32_t ret = UpdateInputShape(ctx, inputShapeAndType);
        if (ret != KERNEL_STATUS_OK) {
            return ret;
        }

        ret = UpdateOutputShape(ctx, outputShapeAndType);
        if (ret != KERNEL_STATUS_OK) {
            return ret;
        }
    }

    return KERNEL_STATUS_OK;
}

uint32_t ParseExtAndUpdateShape(CpuKernelContext &ctx, AicpuParamHead *paramHead, bool &unknownShape,
    std::vector<FWKAdapter::ShapeAndType *> &inputShapeAndType,
    std::vector<FWKAdapter::ShapeAndType *> &outputShapeAndType)
{
    KERNEL_LOG_INFO("Parse extend info and update shape begin");
    uint32_t offset = 0;
    FWKAdapter::ExtInfo *extInfoPtr = nullptr;
    char *extInfoBuf = reinterpret_cast<char *>(static_cast<uintptr_t>(paramHead->extInfoAddr));
    while (offset + sizeof(FWKAdapter::ExtInfo) <= paramHead->extInfoLength) {
        extInfoPtr = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf + offset);
        if (extInfoPtr == nullptr) {
            KERNEL_LOG_ERROR("extInfo is nullptr, extInfoLength=%u, extInfoAddr=%p, offset=%zu.",
                paramHead->extInfoLength, paramHead->extInfoAddr, offset);
            return KERNEL_STATUS_PARAM_INVALID;
        }

        uint32_t ret = KERNEL_STATUS_OK;
        switch (extInfoPtr->infoType) {
            case FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE:
                ret = ParseExtShapeType(extInfoPtr, unknownShape);
                break;
            case FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE:
                ret = ParseExtInputShape(ctx, extInfoPtr, inputShapeAndType);
                break;
            case FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE:
                ret = ParseExtOutputShape(ctx, extInfoPtr, outputShapeAndType);
                break;
            default:
                KERNEL_LOG_INFO("ignore infoType=%d, infoLen=%u.", extInfoPtr->infoType, extInfoPtr->infoLen);
                break;
        }

        if (ret != KERNEL_STATUS_OK) {
            return ret;
        }

        // not overflow
        offset += FWKAdapter::kExtInfoHeadSize;
        offset += extInfoPtr->infoLen;
    }

    uint32_t ret = UpdateInputAndOutputShape(ctx, unknownShape, inputShapeAndType, outputShapeAndType);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }

    return KERNEL_STATUS_OK;
}

uint32_t UpdateFWKOutputShape(const CpuKernelContext &ctx, bool unknownShape,
    std::vector<FWKAdapter::ShapeAndType *> &outputShapeAndType)
{
    if (unknownShape) {
        for (uint32_t i = 0; i < ctx.GetOutputsSize(); ++i) {
            Tensor *output = ctx.Output(i);
            KERNEL_CHECK_NULLPTR(output, KERNEL_STATUS_PARAM_INVALID, "get output:%u failed.", i)
            auto shape = output->GetTensorShape();
            KERNEL_CHECK_NULLPTR(shape, KERNEL_STATUS_PARAM_INVALID, "get output:%u shape failed.", i)

            for (int32_t index = 0; index < shape->GetDims(); ++index) {
                outputShapeAndType[i]->dims[index] = shape->GetDimSize(index);
            }
        }
    }
    return KERNEL_STATUS_OK;
}

uint32_t SetTensorDataAndSize(CpuKernelContext &ctx, const std::vector<uint64_t> &ioAddrs)
{
    if (ioAddrs.size() != ctx.GetInputsSize() + ctx.GetOutputsSize()) {
        KERNEL_LOG_ERROR("addr number:%zu is not equal to the sum of inputs:%zu and output:%zu.", ioAddrs.size(),
            ctx.GetInputsSize(), ctx.GetOutputsSize());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    size_t addrIndex = 0;
    for (size_t i = 0; i < ctx.GetInputsSize(); i++, addrIndex++) {
        Tensor *input = ctx.Input(i);
        if (input == nullptr) {
            KERNEL_LOG_ERROR("get input:%u failed.", i);
            return KERNEL_STATUS_PARAM_INVALID;
        }
        input->SetData(reinterpret_cast<void *>(static_cast<uintptr_t>(ioAddrs[addrIndex])));
        int64_t calcDataSize = input->CalcDataSizeByShape();
        uint64_t dataSize = calcDataSize < 0 ? 0 : calcDataSize;
        input->SetDataSize(dataSize);
        KERNEL_LOG_INFO("set input:%u addr:%llu success.", i, ioAddrs[addrIndex]);
    }

    for (size_t i = 0; i < ctx.GetOutputsSize(); i++, addrIndex++) {
        Tensor *output = ctx.Output(i);
        if (output == nullptr) {
            KERNEL_LOG_ERROR("get output:%u failed.", i);
            return KERNEL_STATUS_PARAM_INVALID;
        }
        output->SetData(reinterpret_cast<void *>(static_cast<uintptr_t>(ioAddrs[addrIndex])));
        int64_t calcDataSize = output->CalcDataSizeByShape();
        uint64_t dataSize = calcDataSize < 0 ? 0 : calcDataSize;
        output->SetDataSize(dataSize);
        KERNEL_LOG_INFO("set output:%u addr:%llu success.", i, ioAddrs[addrIndex]);
    }
    return KERNEL_STATUS_OK;
}
}

extern "C" {
__attribute__((visibility("default"))) uint32_t RunCpuKernel(void *param)
{
    KERNEL_LOG_INFO("RunCpuKernel C begin");
    if (param == nullptr) {
        KERNEL_LOG_ERROR("param is null.");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    // parse param_len
    AicpuParamHead *paramHead = static_cast<AicpuParamHead *>(param);
    if ((paramHead->length < sizeof(AicpuParamHead)) || (paramHead->length > MAX_PARAM_LEN)) {
        KERNEL_LOG_ERROR("param length=%u not in [%zu, %u].", paramHead->length, sizeof(AicpuParamHead), MAX_PARAM_LEN);
        return KERNEL_STATUS_PARAM_INVALID;
    }

    std::vector<uint64_t> ioAddrs;
    std::string strData;
    uint32_t ret = ParseIoAddrAndNodeDefStr(paramHead, ioAddrs, strData);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }

    auto nodedef = CpuKernelUtils::CreateNodeDef();
    KERNEL_CHECK_NULLPTR(nodedef, KERNEL_STATUS_INNER_ERROR, "Create node def failed.")

    if (!nodedef->ParseFromString(strData)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }

    CpuKernelContext ctx(DEVICE);
    ret = ctx.Init(nodedef.get());
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }

    ret = SetTensorDataAndSize(ctx, ioAddrs);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }

    bool unknownShape = false;
    std::vector<FWKAdapter::ShapeAndType *> inputShapeAndType;
    std::vector<FWKAdapter::ShapeAndType *> outputShapeAndType;
    ret = ParseExtAndUpdateShape(ctx, paramHead, unknownShape, inputShapeAndType, outputShapeAndType);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }

    ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }

    return UpdateFWKOutputShape(ctx, unknownShape, outputShapeAndType);
}
}
