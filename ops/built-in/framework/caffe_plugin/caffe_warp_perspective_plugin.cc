/* Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"


using namespace ge;

namespace domi
{
// Caffe ParseParams
Status ParseParams_WarpPerspective(const Message* op_origin, ge::Operator& op_dest)
{
    const caffe::LayerParameter* layer =
        dynamic_cast<const caffe::LayerParameter*>(op_origin);

    if (nullptr == layer) {
        OP_LOGE("WarpPerspective", "Dynamic cast op_src to LayerParameter failed.");
        return FAILED;
    }

    const caffe::WarpPerspectiveParameter& param = layer->warp_perspective_param();
    if(param.has_out_height()) {
        op_dest.SetAttr("out_height", (int)param.out_height());
    }
    else {
        OP_LOGE("WarpPerspective Get out_height failed.");
        return FAILED;
    }

    if(param.has_out_width()) {
        op_dest.SetAttr("out_width", (int)param.out_width());
    }
    else {
        OP_LOGE("WarpPerspective Get out_width failed.");
        return FAILED;
    }

    if(param.has_constant()) {
        op_dest.SetAttr("constant", (float)param.constant());
    }

    if(param.has_border_type()) {
        op_dest.SetAttr("border_type", (string)param.border_type());
    }

    return SUCCESS;
}


REGISTER_CUSTOM_OP("WarpPerspective")
    .FrameworkType(CAFFE)  // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("WarpPerspective")  // // Reduction indicates the type name of the operator in the caffe framework.
    .ParseParamsFn(ParseParams_WarpPerspective)  // AutoMappingFn indicates automatic mapping the parameters of op.
    .ImplyType(ImplyType::TVM);

}  // namespace domi
