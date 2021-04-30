/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permission and
 * limitations under the License.
 */
#include "op_log.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include <string>
#include <vector>
 
namespace domi {
Status parse_params_hard_max(const Message *op_src, ge::Operator &op_dest)
{
    const ge::onnx::NodeProto *node = reinterpret_cast<const ge::onnx::NodeProto *>(op_src);
    if (node == nullptr) {
        OP_LOGE("HardMax", "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    int axis = -1;
    for (auto attr : node->attribute()) {
        if (attr.name() == "axis") {
            axis = attr.i();
        }
    }
    op_dest.SetAttr("axis", axis);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("HardMax")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::HardMax",
                   "ai.onnx::9::HardMax",
                   "ai.onnx::10::HardMax",
                   "ai.onnx::11::HardMax",
                   "ai.onnx::12::HardMax",
                   "ai.onnx::13::HardMax"})
    .ParseParamsFn(parse_params_hard_max)
    .ImplyType(ImplyType::TVM);
}


