/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: implement of tensor
 */

#include "tensor_impl.h"

#include "proto/me_tensor_shape.pb.h"
#include "cpu_kernel_utils.h"
#include "tensor_shape_impl.h"
#include "cpu_types.h"
#include "log.h"

namespace aicpu {
/*
 * get tensor shape value of tensor.
 */
std::shared_ptr<TensorShape> TensorImpl::GetTensorShape() const
{
    aicpuop::TensorShape *tensorShape = tensor_->mutable_tensor_shape();
    if (tensorShape == nullptr) {
        KERNEL_LOG_ERROR("Protobuf mutable tensor shape is null.");
        return std::shared_ptr<TensorShape>(nullptr);
    }

    TensorShapeImpl *impl = new (std::nothrow) TensorShapeImpl(tensorShape);
    if (impl == nullptr) {
        KERNEL_LOG_ERROR("create TensorShapeImpl failed.");
        return std::shared_ptr<TensorShape>(nullptr);
    }

    auto aicpuShape = CpuKernelUtils::CreateTensorShape(impl);
    if (aicpuShape == nullptr) {
        delete impl;
    }
    return aicpuShape;
}

/*
 * set tensor shape value to tensor.
 */
bool TensorImpl::SetTensorShape(const TensorShape *shape)
{
    KERNEL_CHECK_NULLPTR(shape, false, "tensor shape is null")

    aicpuop::TensorShape *tensorShape = tensor_->mutable_tensor_shape();
    KERNEL_CHECK_NULLPTR(tensorShape, false, "Protobuf mutable tensor shape is null")
    auto impl = CpuKernelUtils::GetImpl(shape);
    KERNEL_CHECK_NULLPTR(impl, false, "get impl is null")

    auto proto = impl->GetProto();
    KERNEL_CHECK_NULLPTR(proto, false, "get proto is null")

    *tensorShape = *(proto);
    return true;
}

/*
 * get data type value of tensor.
 */
DataType TensorImpl::GetDataType() const
{
    return static_cast<DataType>(tensor_->tensor_type());
}

/*
 * set data type value to tensor.
 */
void TensorImpl::SetDataType(DataType type)
{
    tensor_->set_tensor_type(type);
}

/*
 * get data ptr of tensor.
 */
void *TensorImpl::GetData() const
{
    return reinterpret_cast<void *>(static_cast<uintptr_t>(tensor_->data_ptr()));
}

/*
 * set data ptr to tensor.
 */
void TensorImpl::SetData(void *addr)
{
    tensor_->set_data_ptr(static_cast<uint64_t>(reinterpret_cast<intptr_t>(addr)));
}

/*
 * get data size of tensor.
 */
uint64_t TensorImpl::GetDataSize() const
{
    return tensor_->data_size();
}

/*
 * set data size to tensor.
 */
void TensorImpl::SetDataSize(uint64_t size)
{
    tensor_->set_data_size(size);
}

/*
 * calculate data size by tensor shape.
 */
int64_t TensorImpl::CalcDataSizeByShape() const
{
    int64_t dataSize = NumElements();
    int32_t elementSize = GetSizeByDataType(static_cast<DataType>(GetDataType()));
    if ((dataSize < 0) || (elementSize < 0)) {
        KERNEL_LOG_WARN("get tensor element number:%ld or element type size:%d less than 0.", dataSize, elementSize);
        return -1;
    }

    KERNEL_CHECK_ASSIGN_64S_MULTI(dataSize, elementSize, dataSize, -1);
    return dataSize;
}

/*
 * get data elements number.
 */
int64_t TensorImpl::NumElements() const
{
    auto shape = GetTensorShape();
    if (shape == nullptr) {
        KERNEL_LOG_ERROR("get tensor shape failed.");
        return -1;
    }

    return shape->NumElements();
}

aicpuop::Tensor *TensorImpl::GetProto() const
{
    return tensor_.get();
}
}