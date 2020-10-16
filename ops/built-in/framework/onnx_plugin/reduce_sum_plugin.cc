/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include "op_log.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include <string>
#include <vector>

namespace domi {

Status ParseParamsReduceSum(const Message *op_src, ge::Operator &op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE("ReduceSum", "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    std::vector<int> v_axis;
    bool set_axes_flag = false;
    bool keep_dims = true;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "axes" && attr.type() == ge::onnx::AttributeProto::INTS) {
            for (int i = 0; i < attr.ints_size(); i++) {
                v_axis.push_back(attr.ints(i));
            }
            set_axes_flag = true;
        }
        else if (attr.name() == "keepdims" && attr.type() == ge::onnx::AttributeProto::INT) {
            if (attr.i() != 1) {
                keep_dims = false;
            }
        }
    }
    if (set_axes_flag) {
        op_dest.SetAttr("axes", v_axis);
    }
    else {
        OP_LOGI("ReduceSum", "onnx ReduceSum op has no axes attr, use default.");
    }
    op_dest.SetAttr("keep_dims", keep_dims);

    return SUCCESS;
}

// register ReduceSum op info to GE
REGISTER_CUSTOM_OP("ReduceSumD")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::11::ReduceSum")
    .ParseParamsFn(ParseParamsReduceSum)
    .ImplyType(ImplyType::TVM);
}  // namespace domi