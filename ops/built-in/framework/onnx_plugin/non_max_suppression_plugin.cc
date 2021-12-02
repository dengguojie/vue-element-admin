/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file non_max_suppression_plugin.cc
 * \brief
 */
#include "onnx_common.h"
#include "array_ops.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;

Status ParseParamsNonMaxSuppression(const Message *op_src, ge::Operator &op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    int center_point_box = 0;
    for(const auto &attr : node->attribute()){
       if(attr.name()=="center_point_box" && attr.type()==ge::onnx::AttributeProto::INT) {
           center_point_box = attr.i();

       }
    }
    op_dest.SetAttr("center_point_box",center_point_box);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("NonMaxSuppressionV6")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::10::NonMaxSuppression",
                   "ai.onnx::11::NonMaxSuppression",
                   "ai.onnx::12::NonMaxSuppression",
                   "ai.onnx::13::NonMaxSuppression"})
    .ParseParamsFn(ParseParamsNonMaxSuppression)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
