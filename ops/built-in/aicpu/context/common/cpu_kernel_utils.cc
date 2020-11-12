/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: implement of cpu kernel utils
 */

#include "cpu_kernel_utils.h"
#include "tensor_impl.h"
#include "tensor_shape_impl.h"
#include "attr_value_impl.h"
#include "node_def_impl.h"
#include "log.h"
#include "status.h"
#include "device.h"
#include "sharder.h"

namespace aicpu {
/*
 * construct Tensor for memory self-management.
 */
std::shared_ptr<Tensor> CpuKernelUtils::CreateTensor()
{
    auto protoPtr = new (std::nothrow) aicpuop::Tensor();
    if (protoPtr == nullptr) {
        KERNEL_LOG_ERROR("new Tensor proto failed");
        return std::shared_ptr<Tensor>(nullptr);
    }

    auto wrapperPtr = new (std::nothrow) TensorImpl(protoPtr, [](aicpuop::Tensor *p) { delete p; });
    if (wrapperPtr == nullptr) {
        KERNEL_LOG_ERROR("new TensorProto failed");
        delete protoPtr;
        return std::shared_ptr<Tensor>(nullptr);
    }

    auto classPtr = new (std::nothrow) Tensor(wrapperPtr);
    if (classPtr == nullptr) {
        KERNEL_LOG_ERROR("new Tensor failed");
        delete wrapperPtr;
        return std::shared_ptr<Tensor>(nullptr);
    }

    return std::shared_ptr<Tensor>(classPtr);
}

std::shared_ptr<Tensor> CpuKernelUtils::CreateTensor(TensorImpl *tensor)
{
    if (tensor == nullptr) {
        KERNEL_LOG_ERROR("tensor is null");
        return std::shared_ptr<Tensor>(nullptr);
    }

    auto classPtr = new (std::nothrow) Tensor(tensor);
    if (classPtr == nullptr) {
        KERNEL_LOG_ERROR("new Tensor failed");
        return std::shared_ptr<Tensor>(nullptr);
    }

    return std::shared_ptr<Tensor>(classPtr);
}

/*
 * get tensor impl.
 */
std::shared_ptr<TensorImpl> CpuKernelUtils::GetImpl(const Tensor *tensor)
{
    return tensor->impl_;
}

std::shared_ptr<TensorShape> CpuKernelUtils::CreateTensorShape()
{
    auto protoPtr = new (std::nothrow) aicpuop::TensorShape();
    if (protoPtr == nullptr) {
        KERNEL_LOG_ERROR("new TensorShape proto failed");
        return std::shared_ptr<TensorShape>(nullptr);
    }

    auto wrapperPtr = new (std::nothrow) TensorShapeImpl(protoPtr, [](aicpuop::TensorShape *p) { delete p; });
    if (wrapperPtr == nullptr) {
        KERNEL_LOG_ERROR("new TensorShapeImpl failed");
        delete protoPtr;
        return std::shared_ptr<TensorShape>(nullptr);
    }

    auto classPtr = new (std::nothrow) TensorShape(wrapperPtr);
    if (classPtr == nullptr) {
        KERNEL_LOG_ERROR("new TensorShape failed");
        delete wrapperPtr;
        return std::shared_ptr<TensorShape>(nullptr);
    }

    return std::shared_ptr<TensorShape>(classPtr);
}

std::shared_ptr<TensorShape> CpuKernelUtils::CreateTensorShape(TensorShapeImpl *tensorShape)
{
    if (tensorShape == nullptr) {
        KERNEL_LOG_ERROR("tensor Shape Proto is null");
        return std::shared_ptr<TensorShape>(nullptr);
    }

    auto classPtr = new (std::nothrow) TensorShape(tensorShape);
    if (classPtr == nullptr) {
        KERNEL_LOG_ERROR("new TensorShape failed");
        return std::shared_ptr<TensorShape>(nullptr);
    }

    return std::shared_ptr<TensorShape>(classPtr);
}

/*
 * get tensor shape impl.
 */
std::shared_ptr<TensorShapeImpl> CpuKernelUtils::GetImpl(const TensorShape *tensorShape)
{
    return tensorShape->impl_;
}

/*
 * construct AttrValue for memory self-management.
 */
std::shared_ptr<AttrValue> CpuKernelUtils::CreateAttrValue()
{
    auto protoPtr = new (std::nothrow) aicpuop::AttrValue();
    if (protoPtr == nullptr) {
        KERNEL_LOG_ERROR("new AttrValue proto failed");
        return std::shared_ptr<AttrValue>(nullptr);
    }

    auto wrapperPtr = new (std::nothrow) AttrValueImpl(protoPtr, [](aicpuop::AttrValue *p) { delete p; });
    if (wrapperPtr == nullptr) {
        KERNEL_LOG_ERROR("new AttrValueImpl failed");
        delete protoPtr;
        return std::shared_ptr<AttrValue>(nullptr);
    }

    auto classPtr = new (std::nothrow) AttrValue(wrapperPtr);
    if (classPtr == nullptr) {
        KERNEL_LOG_ERROR("new AttrValue failed");
        delete wrapperPtr;
        return std::shared_ptr<AttrValue>(nullptr);
    }

    return std::shared_ptr<AttrValue>(classPtr);
}

std::shared_ptr<AttrValue> CpuKernelUtils::CreateAttrValue(AttrValueImpl *impl)
{
    if (impl == nullptr) {
        KERNEL_LOG_ERROR("impl is null");
        return std::shared_ptr<AttrValue>(nullptr);
    }

    auto classPtr = new (std::nothrow) AttrValue(impl);
    if (classPtr == nullptr) {
        KERNEL_LOG_ERROR("new AttrValue failed");
        return std::shared_ptr<AttrValue>(nullptr);
    }

    return std::shared_ptr<AttrValue>(classPtr);
}

/*
 * get attr value impl.
 */
std::shared_ptr<AttrValueImpl> CpuKernelUtils::GetImpl(const AttrValue *attrValue)
{
    return attrValue->impl_;
}

/*
 * construct NodeDef for memory self-management.
 */
std::shared_ptr<NodeDef> CpuKernelUtils::CreateNodeDef()
{
    auto protoPtr = new (std::nothrow) aicpuop::NodeDef();
    if (protoPtr == nullptr) {
        KERNEL_LOG_ERROR("new NodeDef proto failed");
        return std::shared_ptr<NodeDef>(nullptr);
    }

    auto wrapperPtr = new (std::nothrow) NodeDefImpl(protoPtr, [](aicpuop::NodeDef *p) { delete p; });
    if (wrapperPtr == nullptr) {
        KERNEL_LOG_ERROR("new NodeDefImpl failed");
        delete protoPtr;
        return std::shared_ptr<NodeDef>(nullptr);
    }

    auto classPtr = new (std::nothrow) NodeDef(wrapperPtr);
    if (classPtr == nullptr) {
        KERNEL_LOG_ERROR("new NodeDef failed");
        delete wrapperPtr;
        return std::shared_ptr<NodeDef>(nullptr);
    }

    return std::shared_ptr<NodeDef>(classPtr);
}

/*
 * ParallelFor shards the "total" units of work.
 * @return uint32_t: 0->sucess other->failed
 */
uint32_t CpuKernelUtils::ParallelFor(const CpuKernelContext &ctx, int64_t total, int64_t perUnitSize,
    const std::function<void(int64_t, int64_t)> &work)
{
    if(ctx.device_ == nullptr) {
        KERNEL_LOG_ERROR("device is null.");
        return KERNEL_STATUS_INNER_ERROR;
    }

    const Sharder *sharder = ctx.device_->GetSharder();
    if(sharder == nullptr) {
        KERNEL_LOG_ERROR("get sharder is null.");
        return KERNEL_STATUS_INNER_ERROR;
    }

    sharder->ParallelFor(total, perUnitSize, work);
    return KERNEL_STATUS_OK;
}

/*
 * Get CPU number
 * @return CPU number
 */
uint32_t CpuKernelUtils::GetCPUNum(const CpuKernelContext &ctx)
{
    if(ctx.device_ == nullptr) {
        KERNEL_LOG_ERROR("device is null.");
        return 0;
    }

    const Sharder *sharder = ctx.device_->GetSharder();
    if(sharder == nullptr) {
        KERNEL_LOG_ERROR("get sharder is null.");
        return 0;
    }

    return sharder->GetCPUNum();
}
} // namespace aicpu
