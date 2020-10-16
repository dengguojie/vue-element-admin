/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this
 * file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http:// www.apache.org/licenses/LICENSE-2.0
 */

#include "op_log.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include <string>

namespace domi {

static const int DEFAULT_AXIS = 0;

Status ParseParamsGather(const Message* op_src, ge::Operator& op_dest) {
    OP_LOGI("Gather", "[PLUGIN_GATHER]---------ParseParams Gather start----------");
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (nullptr == node) {
        OP_LOGE("Gather", "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    int axis_val = 0;
    bool set_axis_flag = false;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
            axis_val = attr.i();
            set_axis_flag = true;
            break;
        }
    }
    if (!set_axis_flag) {
        OP_LOGI("Gather", "onnx Gather op has no axis attr.");
        axis_val = DEFAULT_AXIS;
    }
    op_dest.SetAttr("axis", axis_val);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("GatherV2D")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::11::Gather")
    .ParseParamsFn(ParseParamsGather)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
