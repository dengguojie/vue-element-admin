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
Status ParseParams_ROIPooling(const Message* op_origin, ge::Operator& op_dest)
{
    // trans op_src to op_dest
    const caffe::LayerParameter* layer =
        dynamic_cast<const caffe::LayerParameter*>(op_origin);

    if (nullptr == layer) {
        OP_LOGE("ROIPooling",
            "Dynamic cast op_src to LayerParameter failed.");
        return FAILED;
    }

    const caffe::ROIPoolingParameter& param = layer->roi_pooling_param();

    if(param.has_pooled_h()) {
        op_dest.SetAttr("pooled_h", (int)param.pooled_h());
    }
    if(param.has_pooled_w()) {
        op_dest.SetAttr("pooled_w", (int)param.pooled_w());
    }
    if (param.pooled_w() <= 0 || param.pooled_h() <= 0){
        OP_LOGE("ROIPooling",
            " pooled_h or pooled_w  must be > 0");
        return FAILED;
    }
    if(param.has_spatial_scale_h() && param.has_spatial_scale_w()) {
        op_dest.SetAttr("spatial_scale_h", (float)param.spatial_scale_h());
        op_dest.SetAttr("spatial_scale_w", (float)param.spatial_scale_w());
    }
    else if (param.has_spatial_scale()) {
        op_dest.SetAttr("spatial_scale_h", (float)param.spatial_scale());
        op_dest.SetAttr("spatial_scale_w", (float)param.spatial_scale());
    }
    else {
        // the default value is 0.0625 of spatial_scale
        op_dest.SetAttr("spatial_scale_h", (float)0.0625);
        op_dest.SetAttr("spatial_scale_w", (float)0.0625);
    }


    return SUCCESS;
}

// test_reduction is the type name of the operator in the OM model. 
// It can be specified randomly and cannot be the same as an existing type name. It is case sensitive. 
REGISTER_CUSTOM_OP("ROIPooling")
    .FrameworkType(CAFFE)  // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("ROIPooling")  // // Reduction indicates the type name of the operator in the caffe framework.
    .ParseParamsFn(ParseParams_ROIPooling)  // AutoMappingFn indicates automatic mapping the parameters of op.
    .ImplyType(ImplyType::TVM);
}  // namespace domi
