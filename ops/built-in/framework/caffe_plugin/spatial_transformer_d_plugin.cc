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

#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/operator.h"
#include <memory>
#include <string>
#include <vector>
#include "op_log.h"

using namespace ge;

namespace domi {

Status ParseParamsSpatialTransformer(const Message* op_src, ge::Operator& op_dest) {
    OP_LOGI("SpatialTransformer",
    "[PLUGIN_STN]------------ParseParams SpatialTransformer Start---------------");
    const caffe::LayerParameter* layer =
    dynamic_cast<const caffe::LayerParameter*>(op_src);

    if (nullptr == layer) {
        OP_LOGE("SpatialTransformer", "Dynamic cast op_src to LayerParameter failed.");
        return FAILED;
    }

    const caffe::SpatialTransformerParameter& param = layer->st_param();
    // set output size
    std::vector<int64_t> output_size = {-1, -1};
    if (param.has_output_h()) {
        output_size[0] = param.output_h();
    }
    if (param.has_output_w()) {
        output_size[1] = param.output_w();
    }
    op_dest.SetAttr("output_size", (output_size));

    // set default_theta and use_default_theta
    std::vector<float> default_theta;
    std::vector<bool> use_default_theta = {false, false, false, false, false, false};
    if (param.has_theta_1_1()) {
        default_theta.push_back(param.theta_1_1());
        use_default_theta[0] = true;
    }
    if (param.has_theta_1_2()) {
        default_theta.push_back(param.theta_1_2());
        use_default_theta[1] = true;
    }
    if (param.has_theta_1_3()) {
        default_theta.push_back(param.theta_1_3());
        use_default_theta[2] = true;
    }
    if (param.has_theta_2_1()) {
        default_theta.push_back(param.theta_2_1());
        use_default_theta[3] = true;
    }
    if (param.has_theta_2_2()) {
        default_theta.push_back(param.theta_2_2());
        use_default_theta[4] = true;
    }
    if (param.has_theta_2_3()) {
        default_theta.push_back(param.theta_2_3());
        use_default_theta[5] = true;
    }
    op_dest.SetAttr("default_theta", (default_theta));
    op_dest.SetAttr("use_default_theta", (use_default_theta));

    // set align_corners
    op_dest.SetAttr("align_corners", false);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("SpatialTransformerD")
    .FrameworkType(CAFFE)
    .OriginOpType("SpatialTransformer")
    .ParseParamsFn(ParseParamsSpatialTransformer)
    .ImplyType(ImplyType::TVM);
}  // namespace domi

