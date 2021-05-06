/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: ssd detection output framework cpp file
 * Author: Huawei
 * Create: 2020-6-11
 */

#include "register/register.h"

namespace domi {
Status ParseParamsSSDDetectionOutputV2(const ge::Operator &op_src, ge::Operator &op_dst)
{
    int num_classes = 0;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("num_classes", num_classes)) {
        op_dst.SetAttr("num_classes", num_classes);
    }

    bool share_location = false;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("share_location", share_location)) {
        op_dst.SetAttr("share_location", share_location);
    }

    int background_label_id = 0;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("background_label_id", background_label_id)) {
        op_dst.SetAttr("background_label_id", background_label_id);
    }

    float nms_threshold = 0.0;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("iou_threshold", nms_threshold)) {
        op_dst.SetAttr("iou_threshold", nms_threshold);
    }

    int top_k = 0;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("top_k", top_k)) {
        op_dst.SetAttr("top_k", top_k);
    }

    float eta = 0;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("eta", eta)) {
        op_dst.SetAttr("eta", eta);
    }

    bool variance_encoded_in_target = false;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("variance_encoded_in_target", variance_encoded_in_target)) {
        op_dst.SetAttr("variance_encoded_in_target", variance_encoded_in_target);
    }

    int code_type = 0;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("code_type", code_type)) {
        op_dst.SetAttr("code_type", code_type);
    }

    int keep_top_k = 0;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("keep_top_k", keep_top_k)) {
        op_dst.SetAttr("keep_top_k", keep_top_k);
    }

    float confidence_threshold = 0.0;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("confidence_threshold", confidence_threshold)) {
        op_dst.SetAttr("confidence_threshold", confidence_threshold);
    }

    return SUCCESS;
}

REGISTER_CUSTOM_OP("SSDDetectionOutputV2")
    .FrameworkType(CAFFE)            
    .OriginOpType("DetectionOutput") // name in caffe module
    .ParseParamsByOperatorFn(ParseParamsSSDDetectionOutputV2) 
    .ImplyType(ImplyType::TVM);
}
