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
Status ParseParams_FSRDetectionOutput(const Message* op_origin, ge::Operator& op_dest)
{
    // trans op_src to op_dest
    const caffe::LayerParameter* layer =
        dynamic_cast<const caffe::LayerParameter*>(op_origin);

    if (nullptr == layer) {
        OP_LOGE("FSRDetectionOutput",
            "Dynamic cast op_src to LayerParameter failed");
        return FAILED;
    }

    const caffe::FSRDetectionOutputParameter& param = layer->fsrdetectionoutput_param();
   if(param.has_num_classes()) {
        op_dest.SetAttr("num_classes", param.num_classes());
    }
   if(param.has_score_threshold()) {
        op_dest.SetAttr("score_threshold", param.score_threshold());
    }
    if(param.has_iou_threshold()) {
        op_dest.SetAttr("iou_threshold", param.iou_threshold());
    }
    if(param.has_batch_rois()) {
        op_dest.SetAttr("batch_rois", (int)param.batch_rois());
    }
    return SUCCESS;
}
// test_reduction is the type name of the operator in the OM model. 
// It can be specified randomly and cannot be the same as an existing type name. It is case sensitive. 
REGISTER_CUSTOM_OP("FSRDetectionOutput") 
    .FrameworkType(CAFFE)  // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("FSRDetectionOutput")  // Reduction indicates the type name of the operator in the caffe framework.
    .ParseParamsFn(ParseParams_FSRDetectionOutput)  // AutoMappingFn indicates automatic mapping the parameters of op.
    .ImplyType(ImplyType::TVM);
}  // namespace domi
