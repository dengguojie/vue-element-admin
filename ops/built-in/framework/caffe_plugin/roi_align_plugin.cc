/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
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
namespace domi {
static const int DEFAULT_SAMPLE_NUM = 2;
static const int DEFAULT_ROI_END_MODE = 0;
Status ParseParamsROIAlign(const Message* op_origin,
                                         ge::Operator& op_dest) {
  OP_LOGI("enter into ParseParamsROIAlign ------begin!!\n");
  // trans op_src to op_dest
  const caffe::LayerParameter* layer = \
    dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (nullptr == layer) {
    OP_LOGI("Dynamic cast op_src to LayerParameter failed.\n");
    return FAILED;
  }
  // get layer
  const caffe::ROIAlignParameter& param = \
    layer->roi_align_param();

  if (param.has_spatial_scale()) {
    op_dest.SetAttr("spatial_scale", param.spatial_scale());
  }
  if (param.has_pooled_h()) {
    op_dest.SetAttr("pooled_height", param.pooled_h());
  }
  if (param.has_pooled_w()) {
    op_dest.SetAttr("pooled_width", param.pooled_w());
  }

  int sample_num;
  if (param.has_sampling_ratio()) {
   sample_num = param.sampling_ratio();
  } else {
   sample_num = DEFAULT_SAMPLE_NUM;
  }
  op_dest.SetAttr("sample_num", sample_num);

  int roi_end_mode = DEFAULT_ROI_END_MODE;
  if (param.has_roi_end_mode()) {
    if (param.roi_end_mode()!=0 && param.roi_end_mode()!=1){
      OP_LOGE("roi_end_mode only support 0 and 1 now!! FAIL!\n");
    } else {
      roi_end_mode = param.roi_end_mode();
    }
  }
  op_dest.SetAttr("roi_end_mode", roi_end_mode);

  OP_LOGI("ParseParamsROIAlign ------end!!\n");
  return SUCCESS;
}

// register ROIAlign op info to GE
REGISTER_CUSTOM_OP("ROIAlign")
    .FrameworkType(CAFFE)
    .OriginOpType("ROIAlign")
    .ParseParamsFn(ParseParamsROIAlign)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
