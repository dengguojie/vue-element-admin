/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2018. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file roipooling_plugin.cpp
 * \brief
 */
#include <string>
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"
#include "../../op_proto/util/error_util.h"

namespace domi {
// Caffe ParseParams
Status ParseParams_ROIPooling(const Message* op_origin, ge::Operator& op_dest) {
  // trans op_src to op_dest
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);
  const float DEFAULT_SPATIAL_SCALE = 0.0625;
  if (layer == nullptr) {
    OP_LOGE("ROIPooling", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }

  const caffe::ROIPoolingParameter& param = layer->roi_pooling_param();

  if (param.has_pooled_h()) {
    op_dest.SetAttr("pooled_h", static_cast<int>(param.pooled_h()));
  }
  if (param.has_pooled_w()) {
    op_dest.SetAttr("pooled_w", static_cast<int>(param.pooled_w()));
  }
  if (param.pooled_w() <= 0 || param.pooled_h() <= 0) {
    ge::OpsInputShapeErrReport(op_dest.GetName(), "pooled_h or pooled_w must be larger than 0",
                               "pooled_h",to_string(param.pooled_h()) + ", pooled_w is " + to_string(param.pooled_w()));
    OP_LOGE("ROIPooling", " pooled_h or pooled_w  must be > 0");
    return FAILED;
  }
  if (param.has_spatial_scale_h() && param.has_spatial_scale_w()) {
    op_dest.SetAttr("spatial_scale_h", static_cast<float>(param.spatial_scale_h()));
    op_dest.SetAttr("spatial_scale_w", static_cast<float>(param.spatial_scale_w()));
  } else if (param.has_spatial_scale()) {
    op_dest.SetAttr("spatial_scale_h", static_cast<float>(param.spatial_scale()));
    op_dest.SetAttr("spatial_scale_w", static_cast<float>(param.spatial_scale()));
  } else {
    // the default value is 0.0625 of spatial_scale
    op_dest.SetAttr("spatial_scale_h", static_cast<float>(DEFAULT_SPATIAL_SCALE));
    op_dest.SetAttr("spatial_scale_w", static_cast<float>(DEFAULT_SPATIAL_SCALE));
  }

  return SUCCESS;
}

// test_reduction is the type name of the operator in the OM model.
// It can be specified randomly and cannot be the same as an existing type name. It is case sensitive.
REGISTER_CUSTOM_OP("ROIPooling")
    .FrameworkType(CAFFE)        // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("ROIPooling")  // // Reduction indicates the type name of the operator in the caffe framework.
    .ParseParamsFn(ParseParams_ROIPooling)  // AutoMappingFn indicates automatic mapping the parameters of op.
    .ImplyType(ImplyType::TVM);
}  // namespace domi
