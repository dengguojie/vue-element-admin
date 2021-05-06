/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: RoiAlign caffe plugin cpp file
 * Author: Huawei
 * Create: 2020-6-11
 */

#include "register/register.h"

using namespace ge;

namespace domi {
Status ParseParamsROIAlign(const ge::Operator &op_src, ge::Operator &op_dst)
{   
    int pooled_h = 0;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("pooled_h", pooled_h)) {
        op_dst.SetAttr("pooled_h", pooled_h);
    }
    int pooled_w = 0;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("pooled_w", pooled_w)) {
        op_dst.SetAttr("pooled_w", pooled_w);
    }
    float spatial_scale = 0;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("spatial_scale", spatial_scale)) {
        op_dst.SetAttr("spatial_scale", spatial_scale);
    }
    return SUCCESS;
}

REGISTER_CUSTOM_OP("ROIAlignTIK")
    .FrameworkType(CAFFE)               // type: CAFFE, TENSORFLOW
    .OriginOpType("ROIAlign")           // name in caffe module
    .ParseParamsByOperatorFn(ParseParamsROIAlign) // ParseParamsFn need to realize for caffe
    .ImplyType(ImplyType::TVM);
}
