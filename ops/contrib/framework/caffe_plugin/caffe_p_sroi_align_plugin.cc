/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: PsRoiAlign caffe plugin cpp file
 * Author: Huawei
 * Create: 2020-6-11
 */

#include <memory>
#include <string>
#include <vector>
#include "register/register.h"

using namespace ge;

namespace domi {
Status ParseParamsPSROIAlign(const ge::Operator &op_src, ge::Operator &op_dst)
{
    float spatial_scale = 0.f;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("spatial_scale", spatial_scale)) {
        op_dst.SetAttr("spatial_scale", spatial_scale);
    }
    int output_dim = 0;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("output_dim", output_dim)) {
        op_dst.SetAttr("output_dim", output_dim);
    }
    int group_size = 0;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("group_size", group_size)) {
        op_dst.SetAttr("group_size", group_size);
    }
    int sample_num = 0;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("sample_num", sample_num)) {
        op_dst.SetAttr("sample_num", sample_num);
    }
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PSROIAlign")
    .FrameworkType(CAFFE)                 // type: CAFFE, TENSORFLOW
    .OriginOpType("PSROIAlign")           // name in caffe module
    .ParseParamsByOperatorFn(ParseParamsPSROIAlign) // AutoMappingFn for Tensorflow
    .ImplyType(ImplyType::TVM);
}
