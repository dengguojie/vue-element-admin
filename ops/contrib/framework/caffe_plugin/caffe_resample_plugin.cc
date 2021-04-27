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

#include "register/register.h"

namespace domi {
    Status ParseParamsResample(const ge::Operator &op_src, ge::Operator &op_dst)
    {
        bool antialias = true;
        int width = 1;
        int height = 1;
        int type = 1;
        if (ge::GRAPH_SUCCESS == op_src.GetAttr("antialias", antialias)) {
            op_dst.SetAttr("antialias", antialias);
        }
        if (ge::GRAPH_SUCCESS == op_src.GetAttr("width", width)) {
            op_dst.SetAttr("width", width);
        }
        if (ge::GRAPH_SUCCESS == op_src.GetAttr("height", height)) {
            op_dst.SetAttr("height", height);
        }
        if (ge::GRAPH_SUCCESS == op_src.GetAttr("type", type)) {
            op_dst.SetAttr("type", type);
        }
        return SUCCESS;
    }

    REGISTER_CUSTOM_OP("Resample")
    .FrameworkType(CAFFE)
    .OriginOpType("Resample")
    .ParseParamsByOperatorFn(ParseParamsResample)
    .ImplyType(ImplyType::TVM);
}  // namespace domi

