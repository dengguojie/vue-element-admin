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
Status ParseParamsCrop(const Message* op_origin, ge::Operator& op_dest)
{
    // trans op_src to op_dest
    const caffe::LayerParameter* layer =
        dynamic_cast<const caffe::LayerParameter*>(op_origin);

    if (nullptr == layer) {
        OP_LOGE("Crop", "Dynamic cast op_src to LayerParameter failed.");
        return FAILED;
    }

    const caffe::CropParameter& param = layer->crop_param();
    int axis = 2;
    if(param.has_axis()) {
        axis = (int)param.axis();
    }
    op_dest.SetAttr("axis", axis);

    std::vector<int64_t> v_offsets;
    int offsetSize = param.offset_size();
    if(offsetSize==0){
        v_offsets.push_back(0);

    } else{
        for (int32_t i = 0; i < param.offset_size(); i++) {
            v_offsets.push_back(param.offset(i));

        }
    }
    op_dest.SetAttr("offsets", v_offsets);

    return SUCCESS;
}

REGISTER_CUSTOM_OP("Crop")
    .FrameworkType(CAFFE)  // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("Crop")  // // Reduction indicates the type name of the operator in the caffe framework.
    .ParseParamsFn(ParseParamsCrop)  // AutoMappingFn indicates automatic mapping the parameters of op.
    .ImplyType(ImplyType::TVM);
}  // namespace domi
