/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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
 * \file caffe_proposal_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {
// Parse the parameters from caffe model, and pass them to the inner model.
Status ParseParamsProposal(const Message* op_origin, ge::Operator& op_dest) {
  OP_LOGI("Proposal", "enter into ParseParamsProposal ------begin!!");
  // trans op_src to op_dest
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);

  // Ckeck operator parameter's validity
  if (nullptr == layer) {
    OP_LOGE("Proposal", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }

  // get layer
  const caffe::ProposalParameter& param = layer->proposal_param();

  if (param.has_feat_stride()) {
    op_dest.SetAttr("feat_stride", param.feat_stride());
  }

  if (param.has_base_size()) {
    op_dest.SetAttr("base_size", param.base_size());
  }

  if (param.has_min_size()) {
    op_dest.SetAttr("min_size", param.min_size());
  }

  if (param.ratio_size() > 0) {
    std::vector<float> v_ratio;
    for (int32_t i = 0; i < param.ratio_size(); i++) {
      v_ratio.push_back((param.ratio(i)));
    }
    op_dest.SetAttr("ratio", v_ratio);
  }

  if (param.scale_size() > 0) {
    std::vector<float> v_scale;
    for (int32_t i = 0; i < param.scale_size(); i++) {
      v_scale.push_back((param.scale(i)));
    }
    op_dest.SetAttr("scale", v_scale);
  }

  if (param.has_pre_nms_topn()) {
    op_dest.SetAttr("pre_nms_topn", param.pre_nms_topn());
  }
  if (param.has_post_nms_topn()) {
    op_dest.SetAttr("post_nms_topn", param.post_nms_topn());
  }
  if (param.has_iou_threshold()) {
    op_dest.SetAttr("iou_threshold", param.iou_threshold());
  }

  if (param.has_output_actual_rois_num()) {
    op_dest.SetAttr("output_actual_rois_num", param.output_actual_rois_num());
  }
  OP_LOGI("Proposal", "ParseParamsProposal ------end!!");

  return SUCCESS;
}

/**
 * Register the op plugin
 * REGISTER_CUSTOM_OP:    Operator type name in om model, can be any but not duplicate with existence. case sensitive
 * FrameworkType:       Enum type, only support CAFFE
 * OriginOpType:        name of the operator type name in CAFFE
 * ParseParamsFn:      Op parameters parse function
 * InferShapeAndTypeFn: Set output description and datatype function
 * TEBinBuildFn:    Class name of op parser
 * ImplyType:           Instantiation type, TVM
 */
REGISTER_CUSTOM_OP("Proposal")
    .FrameworkType(CAFFE)
    .OriginOpType("Proposal")
    .ParseParamsFn(ParseParamsProposal)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
