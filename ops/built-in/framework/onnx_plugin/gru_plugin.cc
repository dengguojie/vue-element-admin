/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file gru_plugin.cpp
 * \brief
 */
#include "op_log.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include <string>

namespace domi {

Status ParseParamsCommonGRU(const Message *op_src, ge::Operator &op_dest) {
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);

    if (nullptr == node) {
        OP_LOGE("GRU", "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    
    int hidden_size = 0;
    for (const auto& attr: node->attribute()) {
        if (attr.name() == "hidden_size" && attr.type() == ge::onnx::AttributeProto::INT) {
            hidden_size = attr.i();
        }
    }
    op_dest.SetAttr("hidden_size", hidden_size);

    return SUCCESS;
}

REGISTER_CUSTOM_OP("CommonGRU")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::11::GRU")
    .ParseParamsFn(ParseParamsCommonGRU)
    .ImplyType(ImplyType::TVM);

}  // namespace domi