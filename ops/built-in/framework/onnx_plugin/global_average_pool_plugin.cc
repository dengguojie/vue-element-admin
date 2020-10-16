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

#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/operator.h"
#include "op_log.h"
#include <string>
#include <vector>

namespace domi {

static const int DEFAULT_AXIS = 0;

Status ParseParamsGlobalAveragePool(const Message* op_src, ge::Operator& op_dest) 
{
    OP_LOGI("GlobalAveragePool", "[PLUGIN_GLOBALAVERAGEPOOL]---------ParseParams GlobalAveragePool start----------");
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (nullptr == node) {
        OP_LOGE("GlobalAveragePool", "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    std::vector<int> strides;
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    op_dest.SetAttr("strides", strides);

    string padding = "VALID";
    op_dest.SetAttr("padding", padding);

    string data_format = "NCHW";
    op_dest.SetAttr("data_format", data_format);

    std::vector<int> ksize;
    ksize.push_back(1);
    ksize.push_back(1);
    ksize.push_back(-1);
    ksize.push_back(-1);
    op_dest.SetAttr("ksize", ksize);

    return SUCCESS;
}

REGISTER_CUSTOM_OP("AvgPool")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::11::GlobalAveragePool")
    .ParseParamsFn(ParseParamsGlobalAveragePool)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
