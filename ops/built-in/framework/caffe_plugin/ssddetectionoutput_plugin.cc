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
Status ParseParams_SSDDetectionOutput(const Message* op_origin, ge::Operator& op_dest)
{
    // trans op_src to op_dest
    const caffe::LayerParameter* layer =
        dynamic_cast<const caffe::LayerParameter*>(op_origin);

    if (nullptr == layer) {
        OP_LOGE("SSDDetectionOutput",
            "Dynamic cast op_src to LayerParameter failed");
        return FAILED;
    }
    const caffe::SSDDetectionOutputParameter& param = layer->ssddetectionoutput_param();

    if(param.has_num_classes()) {
        op_dest.SetAttr("num_classes", param.num_classes());
    }
    if(param.has_share_location()) {
        op_dest.SetAttr("share_location", param.share_location());
    }
    if(param.has_background_label_id()) {
        op_dest.SetAttr("background_label_id", param.background_label_id());
    }
    if(param.has_iou_threshold()) {
        op_dest.SetAttr("iou_threshold", param.iou_threshold());
    }
    if(param.has_top_k()) {
        op_dest.SetAttr("top_k", param.top_k());
    }
    if(param.has_eta()) {
        op_dest.SetAttr("eta", param.eta());
    }
    if(param.has_variance_encoded_in_target()) {
        op_dest.SetAttr("variance_encoded_in_target", param.variance_encoded_in_target());
    }
    if(param.has_code_type()) {
        op_dest.SetAttr("code_type", param.code_type());
    }
    if(param.has_keep_top_k()) {
        op_dest.SetAttr("keep_top_k", param.keep_top_k());
    }
    if(param.has_confidence_threshold()) {
        op_dest.SetAttr("confidence_threshold", param.confidence_threshold());
    }

    return SUCCESS;
}
// test_reduction is the type name of the operator in the OM model.
// It can be specified randomly and cannot be the same as an existing type name. It is case sensitive.
REGISTER_CUSTOM_OP("SSDDetectionOutput")
    .FrameworkType(CAFFE)  // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("SSDDetectionOutput")  // Reduction indicates the type name of the operator in the caffe framework.
    .ParseParamsFn(ParseParams_SSDDetectionOutput)  // AutoMappingFn indicates automatic mapping the param.has_ters of op.
    .ImplyType(ImplyType::TVM);
}  // namespace domi
