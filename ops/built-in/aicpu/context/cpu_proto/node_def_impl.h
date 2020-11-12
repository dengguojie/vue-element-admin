/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of node def impl
 */

#ifndef CPU_KERNEL_NODE_DEF_IMPL_H
#define CPU_KERNEL_NODE_DEF_IMPL_H
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include "proto/me_node_def.pb.h"

#include "cpu_tensor.h"
#include "cpu_attr_value.h"

namespace aicpu {
class NodeDefImpl {
    friend class CpuKernelUtils;

public:
    NodeDefImpl(
        aicpuop::NodeDef *nodeDef, std::function<void(aicpuop::NodeDef *)> delFunc = [](aicpuop::NodeDef *p) {})
        : nodeDef_(nodeDef, delFunc)
    {}

    ~NodeDefImpl() = default;
    NodeDefImpl(const NodeDefImpl &) = delete;
    NodeDefImpl(NodeDefImpl &&) = delete;
    NodeDefImpl &operator=(const NodeDefImpl &) = delete;
    NodeDefImpl &operator=(NodeDefImpl &&) = delete;

    /*
     * parse parameter from string.
     * @return bool: true->success, false->failed
     */
    bool ParseFromString(const std::string &str);

    /*
     * serialize string to node def.
     * @return bool: true->success, false->failed
     */
    bool SerializeToString(std::string &str) const;

    /*
     * set op type to node def.
     * @param op: op type
     */
    void SetOpType(const std::string &op);

    /*
     * get op type of node def.
     * @return string: op type
     */
    std::string GetOpType() const;

    /*
     * add input tensor to node def.
     * @return shared_ptr<Tensor>: not null->success, null->failed
     */
    std::shared_ptr<Tensor> AddInputs();

    /*
     * add output tensor to node def.
     * @return shared_ptr<Tensor>: not null->success, null->failed
     */
    std::shared_ptr<Tensor> AddOutputs();

    /*
     * add attr to node def.
     * @param name: attr name
     * @param attr: attr need to add
     * @return bool: true->success, false->failed
     */
    bool AddAttrs(const std::string &name, const AttrValue *attr);

    /*
     * get input tensor size of node def.
     * @return int32_t: input tensor size of node def
     */
    int32_t InputsSize() const;

    /*
     * get output tensor size of node def.
     * @return int32_t: input tensor size of node def
     */
    int32_t OutputsSize() const;

    /*
     * get input tensor of node def.
     * @param index: index of input tensor
     * @return shared_ptr<Tensor>: input tensor ptr of node def
     */
    std::shared_ptr<Tensor> MutableInputs(int32_t index) const;

    /*
     * get output tensor of node def.
     * @param index: index of output tensor
     * @return shared_ptr<Tensor>: output tensor ptr of node def
     */
    std::shared_ptr<Tensor> MutableOutputs(int32_t index) const;

    /*
     * get attr of node def.
     * @return std::unordered_map<std::string, std::shared_ptr<AttrValue>>: attrs of node def
     */
    std::unordered_map<std::string, std::shared_ptr<AttrValue> > Attrs() const;

private:
    std::shared_ptr<aicpuop::NodeDef> nodeDef_ { nullptr };
};
} // namespace aicpu
#endif // CPU_KERNEL_NODE_DEF_IMPL_H
