/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: implement of attr value
 */

#include "attr_value_impl.h"

#include "cpu_kernel_utils.h"
#include "tensor_impl.h"
#include "tensor_shape_impl.h"
#include "log.h"

namespace aicpu {
/*
 * get string value of attr.
 */
std::string AttrValueImpl::GetString() const
{
    return attrValue_->s();
}

/*
 * get string list size of attr.
 */
int32_t AttrValueImpl::ListStringSize() const
{
    auto array = attrValue_->array();
    return array.s_size();
}

/*
 * get string list value of attr.
 */
std::vector<std::string> AttrValueImpl::GetListString() const
{
    std::vector<std::string> ret;
    auto array = attrValue_->array();
    for (int32_t i = 0; i < array.s_size(); i++) {
        ret.emplace_back(array.s(i));
    }
    return ret;
}

/*
 * set string list value to attr.
 */
void AttrValueImpl::SetListString(const std::vector<std::string> &bytes)
{
    auto array = attrValue_->mutable_array();
    KERNEL_CHECK_NULLPTR_VOID(array, "Protobuf mutable array is nullptr")
    for (const std::string &s : bytes) {
        array->add_s(s);
    }
}

/*
 * set string value to attr.
 */
void AttrValueImpl::SetString(const std::string &byte)
{
    attrValue_->set_s(byte);
}

/*
 * attr add string value to list.
 */
void AttrValueImpl::AddListString(const std::string &str)
{
    auto array = attrValue_->mutable_array();
    KERNEL_CHECK_NULLPTR_VOID(array, "Protobuf mutable array is nullptr")
    array->add_s(str);
}

/*
 * get int value of attr.
 */
int64_t AttrValueImpl::GetInt() const
{
    return attrValue_->i();
}

/*
 * get int list value of attr.
 */
std::vector<int64_t> AttrValueImpl::GetListInt() const
{
    std::vector<int64_t> ret;
    auto array = attrValue_->array();
    for (int32_t i = 0; i < array.i_size(); i++) {
        ret.emplace_back(array.i(i));
    }
    return ret;
}

/*
 * attr add int value to list.
 */
void AttrValueImpl::AddListInt(int64_t i)
{
    auto array = attrValue_->mutable_array();
    KERNEL_CHECK_NULLPTR_VOID(array, "Protobuf mutable array is nullptr")
    array->add_i(i);
}

/*
 * get int list size of attr.
 */
int32_t AttrValueImpl::ListIntSize() const
{
    auto array = attrValue_->array();
    return array.i_size();
}

/*
 * set int value to attr.
 */
void AttrValueImpl::SetInt(int64_t i)
{
    attrValue_->set_i(i);
}

/*
 * set int list value to attr.
 */
void AttrValueImpl::SetListInt(const std::vector<int64_t> &list)
{
    auto array = attrValue_->mutable_array();
    KERNEL_CHECK_NULLPTR_VOID(array, "Protobuf mutable array is nullptr")
    for (const int64_t &i : list) {
        array->add_i(i);
    }
}

/*
 * get float value of attr.
 */
float AttrValueImpl::GetFloat() const
{
    return attrValue_->f();
}

/*
 * get float list value of attr.
 */
std::vector<float> AttrValueImpl::GetListFloat() const
{
    std::vector<float> ret;
    auto array = attrValue_->array();
    for (int32_t i = 0; i < array.f_size(); i++) {
        ret.emplace_back(array.f(i));
    }
    return ret;
}

/*
 * attr add float value to list.
 */
void AttrValueImpl::AddListFloat(float f)
{
    auto array = attrValue_->mutable_array();
    KERNEL_CHECK_NULLPTR_VOID(array, "Protobuf mutable array is nullptr")
    array->add_f(f);
}

/*
 * set float value to attr.
 */
void AttrValueImpl::SetFloat(float f)
{
    attrValue_->set_f(f);
}

/*
 * get float list size of attr.
 */
int32_t AttrValueImpl::ListFloatSize() const
{
    auto array = attrValue_->array();
    return array.f_size();
}

/*
 * set float list value to attr.
 */
void AttrValueImpl::SetListFloat(const std::vector<float> &list)
{
    auto array = attrValue_->mutable_array();
    KERNEL_CHECK_NULLPTR_VOID(array, "Protobuf mutable array is nullptr")
    for (const float &f : list) {
        array->add_f(f);
    }
}

/*
 * get bool value of attr.
 */
bool AttrValueImpl::GetBool() const
{
    return attrValue_->b();
}

/*
 * get bool list value of attr.
 */
std::vector<bool> AttrValueImpl::GetListBool() const
{
    std::vector<bool> ret;
    auto array = attrValue_->array();
    for (int32_t i = 0; i < array.b_size(); i++) {
        ret.push_back(array.b(i));
    }
    return ret;
}

/*
 * attr add bool value to list.
 */
void AttrValueImpl::AddListBool(bool b)
{
    auto array = attrValue_->mutable_array();
    KERNEL_CHECK_NULLPTR_VOID(array, "Protobuf mutable array is nullptr")
    array->add_b(b);
}

/*
 * get bool list size of attr.
 */
int32_t AttrValueImpl::ListBoolSize() const
{
    auto array = attrValue_->array();
    return array.b_size();
}

/*
 * set bool value to attr.
 */
void AttrValueImpl::SetBool(bool b)
{
    attrValue_->set_b(b);
}

/*
 * set bool list value to attr.
 */
void AttrValueImpl::SetListBool(const std::vector<bool> &list)
{
    auto array = attrValue_->mutable_array();
    KERNEL_CHECK_NULLPTR_VOID(array, "Protobuf mutable array is nullptr")
    for (const bool &b : list) {
        array->add_b(b);
    }
}

/*
 * get data type value of attr.
 */
DataType AttrValueImpl::GetDataType() const
{
    return static_cast<DataType>(attrValue_->type());
}

/*
 * get data type list value of attr.
 */
std::vector<DataType> AttrValueImpl::GetListDataType() const
{
    std::vector<DataType> ret;
    auto array = attrValue_->array();
    for (int32_t i = 0; i < array.type_size(); i++) {
        ret.emplace_back(static_cast<DataType>(array.type(i)));
    }
    return ret;
}

/*
 * attr add data type value to list.
 */
void AttrValueImpl::AddListDataType(DataType type)
{
    auto array = attrValue_->mutable_array();
    KERNEL_CHECK_NULLPTR_VOID(array, "Protobuf mutable array is nullptr")
    array->add_type(type);
}

/*
 * get data type list size of attr.
 */
int32_t AttrValueImpl::ListDataTypeSize() const
{
    auto array = attrValue_->array();
    return array.type_size();
}

/*
 * set data type value to attr.
 */
void AttrValueImpl::SetDataType(DataType type)
{
    attrValue_->set_type(type);
}

/*
 * set data type list value to attr.
 */
void AttrValueImpl::SetListDataType(const std::vector<DataType> &list)
{
    auto array = attrValue_->mutable_array();
    KERNEL_CHECK_NULLPTR_VOID(array, "Protobuf mutable array is nullptr")
    for (const DataType &type : list) {
        array->add_type(type);
    }
}

/*
 * set tensor shape value to attr.
 */
bool AttrValueImpl::SetTensorShape(const TensorShape *shape)
{
    KERNEL_CHECK_NULLPTR(shape, false, "shape is null")

    auto tensorShape = attrValue_->mutable_shape();
    KERNEL_CHECK_NULLPTR(tensorShape, false, "Protobuf mutable tensor shape is null")
    auto impl = CpuKernelUtils::GetImpl(shape);
    KERNEL_CHECK_NULLPTR(impl, false, "get impl is null")
    auto proto = impl->GetProto();
    KERNEL_CHECK_NULLPTR(proto, false, "get proto is null")
    *tensorShape = *(impl->GetProto());
    return true;
}

/*
 * set tensor shape list value to attr.
 */
uint32_t AttrValueImpl::SetListTensorShape(const std::vector<TensorShape *> &list)
{
    uint32_t ret = 0;
    auto array = attrValue_->mutable_array();
    KERNEL_CHECK_NULLPTR(array, ret, "Protobuf mutable array is nullptr")

    for (size_t i = 0; i < list.size(); i++) {
        auto tmpShape = array->add_shape();
        if ((list[i] == nullptr) || (tmpShape == nullptr)) {
            KERNEL_LOG_ERROR("shape[%zu] is null or protobuf add shape ret null.", i);
        } else {
            auto impl = CpuKernelUtils::GetImpl(list[i]);
            if ((impl == nullptr) || (impl->GetProto() == nullptr)) {
                KERNEL_LOG_ERROR("get list[%zu] impl or proto is null.", i);
                continue;
            }
            *tmpShape = *(impl->GetProto());
            ret++;
        }
    }

    return ret;
}

/*
 * attr add tensor shape value to list.
 */
std::shared_ptr<TensorShape> AttrValueImpl::AddListTensorShape()
{
    auto array = attrValue_->mutable_array();
    if (array == nullptr) {
        KERNEL_LOG_ERROR("Protobuf mutable array is nullptr.");
        return std::shared_ptr<TensorShape>(nullptr);
    }

    auto shape = array->add_shape();
    if (shape == nullptr) {
        KERNEL_LOG_ERROR("Protobuf mutable array add shape is nullptr.");
        return std::shared_ptr<TensorShape>(nullptr);
    }

    TensorShapeImpl *impl = new (std::nothrow) TensorShapeImpl(shape);
    if (impl == nullptr) {
        KERNEL_LOG_ERROR("create TensorShapeImpl failed.");
        return std::shared_ptr<TensorShape>(nullptr);
    }

    auto tensorShape = CpuKernelUtils::CreateTensorShape(impl);
    if (tensorShape == nullptr) {
        delete impl;
    }
    return tensorShape;
}

/*
 * get tensor shape value of attr.
 */
std::shared_ptr<TensorShape> AttrValueImpl::GetTensorShape() const
{
    auto shape = attrValue_->mutable_shape();
    if (shape == nullptr) {
        KERNEL_LOG_ERROR("Protobuf mutable shape is nullptr.");
        return std::shared_ptr<TensorShape>(nullptr);
    }

    TensorShapeImpl *impl = new (std::nothrow) TensorShapeImpl(shape);
    if (impl == nullptr) {
        KERNEL_LOG_ERROR("create TensorShapeImpl failed.");
        return std::shared_ptr<TensorShape>(nullptr);
    }

    auto tensorShape = CpuKernelUtils::CreateTensorShape(impl);
    if (tensorShape == nullptr) {
        delete impl;
    }
    return tensorShape;
}

/*
 * get tensor shape list value of attr.
 */
std::vector<TensorShape> AttrValueImpl::GetListTensorShape() const
{
    std::vector<TensorShape> ret;
    auto array = attrValue_->mutable_array();
    KERNEL_CHECK_NULLPTR(array, ret, "Protobuf mutable array is nullptr")
    for (int32_t i = 0; i < array->shape_size(); i++) {
        auto shape = array->mutable_shape(i);
        if (shape == nullptr) {
            KERNEL_LOG_ERROR("Protobuf mutable shape[%d] is nullptr.", i);
            return std::vector<TensorShape>();
        }

        TensorShapeImpl *impl = new (std::nothrow) TensorShapeImpl(shape);
        if (impl == nullptr) {
            KERNEL_LOG_ERROR("create TensorShapeImpl[%d] failed.", i);
            return std::vector<TensorShape>();
        } else {
            auto tensorShape = CpuKernelUtils::CreateTensorShape(impl);
            if (tensorShape == nullptr) {
                delete impl;
                return std::vector<TensorShape>();
            }
            ret.emplace_back(*tensorShape);
        }
    }
    return ret;
}

/*
 * get tensor shape list size of attr.
 */
int32_t AttrValueImpl::ListTensorShapeSize() const
{
    auto array = attrValue_->array();
    return array.shape_size();
}

/*
 * set tensor value to attr.
 */
bool AttrValueImpl::SetTensor(const Tensor *tensor)
{
    KERNEL_CHECK_NULLPTR(tensor, false, "tensor is null")
    auto tensorPtr = attrValue_->mutable_tensor();
    KERNEL_CHECK_NULLPTR(tensorPtr, false, "Protobuf mutable tensor is nullptr")
    auto impl = CpuKernelUtils::GetImpl(tensor);
    KERNEL_CHECK_NULLPTR(impl, false, "get impl is nullptr")
    auto proto = impl->GetProto();
    KERNEL_CHECK_NULLPTR(proto, false, "get proto is nullptr")
    *tensorPtr = *(proto);
    return true;
}

/*
 * set tensor list value to attr.
 */
uint32_t AttrValueImpl::SetListTensor(const std::vector<Tensor *> &list)
{
    uint32_t ret = 0;
    auto array = attrValue_->mutable_array();
    KERNEL_CHECK_NULLPTR(array, ret, "Protobuf mutable array is nullptr")
    for (size_t i = 0; i < list.size(); i++) {
        auto tensorPtr = array->add_tensor();
        if ((list[i] == nullptr) || (tensorPtr == nullptr)) {
            KERNEL_LOG_WARN("tensor[%zu] is null or protobuf add tensor ret null.", i);
        } else {
            auto impl = CpuKernelUtils::GetImpl(list[i]);
            if ((impl == nullptr) || (impl->GetProto() == nullptr)) {
                KERNEL_LOG_WARN("get list[%zu] impl or proto is null.", i);
                continue;
            }
            *tensorPtr = *(impl->GetProto());
            ret++;
        }
    }
    return ret;
}

/*
 * attr add tensor value to list.
 */
std::shared_ptr<Tensor> AttrValueImpl::AddListTensor()
{
    auto array = attrValue_->mutable_array();
    if (array == nullptr) {
        KERNEL_LOG_ERROR("Protobuf mutable array is nullptr.");
        return std::shared_ptr<Tensor>(nullptr);
    }

    auto tensor = array->add_tensor();
    if (tensor == nullptr) {
        KERNEL_LOG_ERROR("Protobuf mutable array add tensor is nullptr.");
        return std::shared_ptr<Tensor>(nullptr);
    }

    TensorImpl *impl = new (std::nothrow) TensorImpl(tensor);
    if (impl == nullptr) {
        KERNEL_LOG_ERROR("create TensorImpl failed.");
        return std::shared_ptr<Tensor>(nullptr);
    }

    auto aicpuTensor = CpuKernelUtils::CreateTensor(impl);
    if (aicpuTensor == nullptr) {
        delete impl;
    }
    return aicpuTensor;
}

/*
 * get tensor value of attr.
 */
std::shared_ptr<Tensor> AttrValueImpl::GetTensor() const
{
    auto tensor = attrValue_->mutable_tensor();
    if (tensor == nullptr) {
        KERNEL_LOG_ERROR("Protobuf mutable tensor is nullptr.");
        return std::shared_ptr<Tensor>(nullptr);
    }

    TensorImpl *impl = new (std::nothrow) TensorImpl(tensor);
    if (impl == nullptr) {
        KERNEL_LOG_ERROR("create TensorImpl failed.");
        return std::shared_ptr<Tensor>(nullptr);
    }

    auto aicpuTensor = CpuKernelUtils::CreateTensor(impl);
    if (aicpuTensor == nullptr) {
        delete impl;
    }
    return aicpuTensor;
}

/*
 * get tensor list value of attr.
 */
std::vector<Tensor> AttrValueImpl::GetListTensor() const
{
    std::vector<Tensor> ret;
    auto array = attrValue_->mutable_array();
    KERNEL_CHECK_NULLPTR(array, ret, "Protobuf mutable array is nullptr")
    for (int32_t i = 0; i < array->tensor_size(); i++) {
        auto tensor = array->mutable_tensor(i);
        if (tensor == nullptr) {
            KERNEL_LOG_ERROR("Protobuf mutable tensor is nullptr.");
            return std::vector<Tensor>();
        }

        TensorImpl *impl = new (std::nothrow) TensorImpl(tensor);
        if (impl == nullptr) {
            KERNEL_LOG_ERROR("create TensorImpl[%d] failed.", i);
            return std::vector<Tensor>();
        } else {
            auto aicpuTensor = CpuKernelUtils::CreateTensor(impl);
            if (aicpuTensor == nullptr) {
                delete impl;
                return std::vector<Tensor>();
            }
            ret.emplace_back(*aicpuTensor);
        }
    }
    return ret;
}

/*
 * get tensor list size of attr.
 */
int32_t AttrValueImpl::ListTensorSize() const
{
    auto array = attrValue_->array();
    return array.tensor_size();
}

/*
 * get attr proto.
 */
aicpuops::AttrValue *AttrValueImpl::GetProto() const
{
    return attrValue_.get();
}
} // namespace aicpu
