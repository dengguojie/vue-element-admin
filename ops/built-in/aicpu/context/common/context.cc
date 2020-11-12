/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: implement of context
 */

#include "cpu_context.h"

#include "proto/me_node_def.pb.h"
#include "proto/me_attr.pb.h"
#include "device.h"
#include "sharder.h"
#include "status.h"
#include "log.h"
#include "cpu_node_def.h"

namespace aicpu {
CpuKernelContext::CpuKernelContext(DeviceType type)
{
    Device *device = new (std::nothrow) Device(type);
    if (device != nullptr) {
        device_.reset(device);
    }
}

uint32_t CpuKernelContext::Init(NodeDef *nodeDef)
{
    if (nodeDef == nullptr) {
        KERNEL_LOG_ERROR("node def is null.");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    op_ = nodeDef->GetOpType();
    KERNEL_LOG_INFO("construct the ctx of the op:%s.", op_.c_str());
    for (int32_t i = 0; i < nodeDef->InputsSize(); i++) {
        auto input = nodeDef->MutableInputs(i);
        if (input == nullptr) {
            KERNEL_LOG_ERROR("get input:%d tensor failed in op:%s.", i, op_.c_str());
            return KERNEL_STATUS_PARAM_INVALID;
        }
        inputs_.emplace_back(std::move(input));
    }

    for (int32_t i = 0; i < nodeDef->OutputsSize(); i++) {
        auto output = nodeDef->MutableOutputs(i);
        if (output == nullptr) {
            KERNEL_LOG_ERROR("get output:%d tensor failed in op:%s.", i, op_.c_str());
            return KERNEL_STATUS_PARAM_INVALID;
        }
        outputs_.emplace_back(std::move(output));
    }

    auto attrMap = nodeDef->Attrs();
    for (auto iter = attrMap.begin(); iter != attrMap.end(); ++iter) {
        auto attrValuePtr = iter->second;
        if (attrValuePtr == nullptr) {
            KERNEL_LOG_ERROR("get attr:%s failed in op:%s.", iter->first.c_str(), op_.c_str());
            return KERNEL_STATUS_PARAM_INVALID;
        }
        auto ret = attrs_.insert(std::make_pair(iter->first, std::move(attrValuePtr)));
        if (ret.second != true) {
            KERNEL_LOG_ERROR("insert attr:%s failed in op:%s.", iter->first.c_str(), op_.c_str());
            return KERNEL_STATUS_INNER_ERROR;
        }
    }

    return KERNEL_STATUS_OK;
}

/*
 * get op type.
 * @return string: op type
 */
std::string CpuKernelContext::GetOpType() const
{
    return op_;
}

/*
 * get input tensor.
 * @return Tensor *: not null->success, null->failed
 */
Tensor *CpuKernelContext::Input(uint32_t index) const
{
    if (index >= inputs_.size()) {
        KERNEL_LOG_ERROR("index:%u should be less than input tensors size:%zu.", index, inputs_.size());
        return nullptr;
    }

    return inputs_[index].get();
}

/*
 * get output tensor.
 * @return Tensor *: not null->success, null->failed
 */
Tensor *CpuKernelContext::Output(uint32_t index) const
{
    if (index >= outputs_.size()) {
        KERNEL_LOG_ERROR("index:%u should be less than output tensors size:%zu.", index, outputs_.size());
        return nullptr;
    }

    return outputs_[index].get();
}

/*
 * get attr.
 * @return AttrValue *: not null->success, null->failed
 */
AttrValue *CpuKernelContext::GetAttr(std::string name) const
{
    auto it = attrs_.find(name);
    if (it == attrs_.end()) {
        KERNEL_LOG_WARN("attr:%s is not exist.", name.c_str());
        return nullptr;
    }

    return (it->second).get();
}

/*
 * get input size.
 * @return uint32_t: input size
 */
uint32_t CpuKernelContext::GetInputsSize() const
{
    return inputs_.size();
}

/*
 * get output size.
 * @return uint32_t: output size
 */
uint32_t CpuKernelContext::GetOutputsSize() const
{
    return outputs_.size();
}
}
