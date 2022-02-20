/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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
 * \file roi_align_plugin.cpp
 * \brief
 */
#include <string>
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"
#include "../../op_proto/util/error_util.h"

namespace domi {
static const int DEFAULT_SAMPLE_NUM = 2;
static const int DEFAULT_ROI_END_MODE = 0;
Status ParseParamsROIAlign(const Message* op_origin, ge::Operator& op_dest)
{
  OP_LOGI("enter into ParseParamsROIAlign ------begin!!\n");
  // trans op_src to op_dest
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (layer == nullptr) {
    OP_LOGI("Dynamic cast op_src to LayerParameter failed.\n");
    return FAILED;
  }
  // get layer
  const caffe::ROIAlignParameter& param = layer->roi_align_param();

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
    if (param.roi_end_mode() != 0 && param.roi_end_mode() != 1) {
      OP_LOGE("ParseParamsROIAlign", "roi_end_mode only support 0 and 1 now!! FAIL!\n");
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
