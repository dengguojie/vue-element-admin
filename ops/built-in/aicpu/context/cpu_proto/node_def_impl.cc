/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: implement of node def
 */
#include "node_def_impl.h"

#include "log.h"
#include "status.h"
#include "tensor_impl.h"
#include "attr_value_impl.h"
#include "cpu_kernel_utils.h"

namespace aicpu {
/*
 * parse parameter from string.
 */
bool NodeDefImpl::ParseFromString(const std::string &str)
{
    if (!nodeDef_->ParseFromString(str)) {
        KERNEL_LOG_ERROR("ParseFromString failed");
        return false;
    }

    return true;
}

/*
 * serialize string to node def.
 */
bool NodeDefImpl::SerializeToString(std::string &str) const
{
    if (!nodeDef_->SerializeToString(&str)) {
        KERNEL_LOG_ERROR("SerializeToString failed");
        return false;
    }

    return true;
}

/*
 * set op type to node def.
 */
void NodeDefImpl::SetOpType(const std::string &op)
{
    nodeDef_->set_op(op);
}

/*
 * get op type of node def.
 */
std::string NodeDefImpl::GetOpType() const
{
    return nodeDef_->op();
}

/*
 * add input tensor to node def.
 */
std::shared_ptr<Tensor> NodeDefImpl::AddInputs()
{
    auto tensor = nodeDef_->add_inputs();
    if (tensor == nullptr) {
        KERNEL_LOG_ERROR("Protobuf node def add tensor is nullptr.");
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
 * add output tensor to node def.
 */
std::shared_ptr<Tensor> NodeDefImpl::AddOutputs()
{
    auto tensor = nodeDef_->add_outputs();
    if (tensor == nullptr) {
        KERNEL_LOG_ERROR("Protobuf node def add tensor is nullptr.");
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
 * add attr to node def.
 */
bool NodeDefImpl::AddAttrs(const std::string &name, const AttrValue *attr)
{
    if (attr == nullptr) {
        KERNEL_LOG_ERROR("attr is null.");
        return false;
    }

    auto attrs = nodeDef_->mutable_attrs();
    KERNEL_CHECK_NULLPTR(attrs, false, "Protobuf mutable attrs is null")
    auto impl = CpuKernelUtils::GetImpl(attr);
    auto pair =
        attrs->insert(google::protobuf::Map<std::string, aicpuop::AttrValue>::value_type(name, *(impl->GetProto())));
    if (!pair.second) {
        KERNEL_LOG_ERROR("Nodedef insert attr %s to nodeDef failed.", name.c_str());
        return false;
    }
    return true;
}

/*
 * get input tensor size of node def.
 */
int32_t NodeDefImpl::InputsSize() const
{
    return nodeDef_->inputs_size();
}

/*
 * get output tensor size of node def.
 */
int32_t NodeDefImpl::OutputsSize() const
{
    return nodeDef_->outputs_size();
}

/*
 * get input tensor of node def.
 */
std::shared_ptr<Tensor> NodeDefImpl::MutableInputs(int32_t index) const
{
    if ((index >= InputsSize()) || (index < 0)) {
        KERNEL_LOG_ERROR("index:%d should be less than input tensors size:%d and noe less than 0.", index,
            InputsSize());
        return std::shared_ptr<Tensor>(nullptr);
    }

    auto tensor = nodeDef_->mutable_inputs(index);
    if (tensor == nullptr) {
        KERNEL_LOG_ERROR("Protobuf node def mutable inputs[%d] tensor is nullptr.", index);
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
 * get output tensor of node def.
 */
std::shared_ptr<Tensor> NodeDefImpl::MutableOutputs(int32_t index) const
{
    if ((index >= OutputsSize()) || (index < 0)) {
        KERNEL_LOG_ERROR("index:%d should be less than output tensors size:%d and noe less than 0.", index,
            OutputsSize());
        return std::shared_ptr<Tensor>(nullptr);
    }

    auto tensor = nodeDef_->mutable_outputs(index);
    if (tensor == nullptr) {
        KERNEL_LOG_ERROR("Protobuf node def mutable outputs[%d] tensor is nullptr.", index);
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
 * get attr of node def.
 */
std::unordered_map<std::string, std::shared_ptr<AttrValue>> NodeDefImpl::Attrs() const
{
    std::unordered_map<std::string, std::shared_ptr<AttrValue>> ret;
    auto attrsMap = nodeDef_->mutable_attrs();
    KERNEL_CHECK_NULLPTR(attrsMap, ret, "Protobuf mutable attrs is null")

    for (auto it = attrsMap->begin(); it != attrsMap->end(); ++it) {
        aicpuop::AttrValue *attr = &(it->second);
        AttrValueImpl *impl = new (std::nothrow) AttrValueImpl(attr);
        if (impl == nullptr) {
            KERNEL_LOG_WARN("create AttrValueImpl failed.");
        }

        auto attrValue = CpuKernelUtils::CreateAttrValue(impl);
        if (attrValue == nullptr) {
            KERNEL_LOG_WARN("create CreateAttrValue failed.");
            delete impl;
        }
        (void)ret.insert(std::make_pair(it->first, attrValue));
    }

    return ret;
}
} // namespace aicpu
