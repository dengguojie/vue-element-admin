/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file caffe_prior_box_plugin.cpp
 * \brief
 */
#include <string>
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"
#include "../../op_proto/util/error_util.h"

namespace domi {
static const float DEFAULT_OFFSET = 0.5;
static const float DEFAULT_STEP = 0.0;
static const int DEFAULT_SIZE = 0;
// Caffe ParseParams
Status ParseParamsPriorBox(const Message* op_origin, ge::Operator& op_dest) {
  OP_LOGI("enter into ParseParamsPriorBox ------begin!!\n");
  // trans op_src to op_dest
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (nullptr == layer) {
    OP_LOGE("Dynamic cast op_src to LayerParameter failed.\n");
    return FAILED;
  }
  // get layer
  const caffe::PriorBoxParameter& param = layer->prior_box_param();

  int n = layer->bottom_size();
  if (n != 2) {
    ge::OpsInputShapeErrReport(op_dest.GetName(), "PriorBox Layer need take two input parameters",
                               "input", to_string(n));
    OP_LOGE(op_dest.GetName().c_str(), "(2 vs. %d) PriorBox Layer takes 2 input.", n);
    return FAILED;
  }

  bool flip = true;
  if (param.has_flip()) {
    flip = param.flip();
  }
  op_dest.SetAttr("flip", flip);

  bool clip = false;
  if (param.has_clip()) {
    clip = param.clip();
  }
  op_dest.SetAttr("clip", clip);

  if (param.has_img_h() || param.has_img_w()) {
    if (param.has_img_size()) {
      ge::OpsInputShapeErrReport(op_dest.GetName(), "set either img_size or img_h/img_w should be specified",
                                 "img_size and img_h/img_w", "set both");
      OP_LOGE("PriorBox", "Either img_size or img_h/img_w should be specified; not both.");
      return FAILED;
    }
    if (param.img_h() < 0) {
      ge::OpsAttrValueErrReport(op_dest.GetName(), "img_h", "larger than 0",
                                to_string(param.img_h()));
      OP_LOGE("PriorBox", "img_h should be larger than 0.");
      return FAILED;
    }
    op_dest.SetAttr("img_h", param.img_h());

    if (param.img_w() < 0) {
      ge::OpsAttrValueErrReport(op_dest.GetName(), "img_w", "larger than 0",
                                to_string(param.img_h()));
      OP_LOGE("PriorBox", "img_w should be larger than 0.");
      return FAILED;
    }
    op_dest.SetAttr("img_w", param.img_w());
  } else if (param.has_img_size()) {
    if (param.img_size() < 0) {
      ge::OpsAttrValueErrReport(op_dest.GetName(), "img_size", "larger than 0",
                                to_string(param.img_size()));
      OP_LOGE("PriorBox", "img_size should be larger than 0.");
      return FAILED;
    }
    op_dest.SetAttr("img_h", param.img_size());
    op_dest.SetAttr("img_w", param.img_size());
  } else {
    op_dest.SetAttr("img_h", DEFAULT_SIZE);
    op_dest.SetAttr("img_w", DEFAULT_SIZE);
  }

  if (param.has_step_h() || param.has_step_w()) {
    if (param.has_step()) {
      ge::OpsInputShapeErrReport(op_dest.GetName(), "set either step or step_h/step_w should be specified",
                                 "step and step_h/step_w", "set both");
      OP_LOGE("PriorBox", "Either step or step_h/step_w should be specified; not both.");
      return FAILED;
    }
    if (param.step_h() < 0) {
      ge::OpsAttrValueErrReport(op_dest.GetName(), "step_h", "larger than 0",
                                to_string(param.step_h()));
      OP_LOGE("PriorBox", "step_h should be larger than 0.");
      return FAILED;
    }
    op_dest.SetAttr("step_h", param.step_h());

    if (param.step_w() < 0) {
      ge::OpsAttrValueErrReport(op_dest.GetName(), "step_w", "larger than 0",
                                to_string(param.step_w()));
      OP_LOGE("PriorBox", "step_w should be larger than 0.");
      return FAILED;
    }
    op_dest.SetAttr("step_w", param.step_w());
  } else if (param.has_step()) {
    if (param.step() < 0) {
      ge::OpsAttrValueErrReport(op_dest.GetName(), "step", "larger than 0",
                                to_string(param.step()));
      OP_LOGE("PriorBox", "step should be larger than 0.");
      return FAILED;
    }
    op_dest.SetAttr("step_h", param.step());
    op_dest.SetAttr("step_w", param.step());
  } else {
    op_dest.SetAttr("step_h", DEFAULT_STEP);
    op_dest.SetAttr("step_w", DEFAULT_STEP);
  }

  float offset = 0.0;
  if (param.has_offset()) {
    offset = param.offset();
  } else {
    offset = DEFAULT_OFFSET;
  }
  op_dest.SetAttr("offset", offset);

  std::vector<float> v_min_size;
  if (param.min_size_size() > 0) {
    for (int32_t i = 0; i < param.min_size_size(); i++) {
      v_min_size.push_back(param.min_size(i));
      if (param.min_size(i) <= 0) {
        ge::OpsAttrValueErrReport(op_dest.GetName(), "min_size", "positive",
                                  to_string(param.min_size(i)));
        OP_LOGE("PriorBox", "min_size must be positive.");
        return FAILED;
      }
    }
    op_dest.SetAttr("min_size", v_min_size);
  } else {
    ge::OpsGetAttrErrReport(op_dest.GetName(), "min_size");
    OP_LOGE("PriorBox", "Must provide min_size.");
    return FAILED;
  }

  std::vector<float> v_max_size;
  if (param.max_size_size() > 0) {
    if (param.max_size_size() != param.min_size_size()) {
      return FAILED;
    }
    for (int32_t i = 0; i < param.max_size_size(); i++) {
      v_max_size.push_back(param.max_size(i));
      if (param.max_size(i) <= param.min_size(i)) {
        ge::OpsAttrValueErrReport(op_dest.GetName(), "max_size", "greater than min_size",
                                  to_string(param.max_size(i)));
        OP_LOGE("PriorBox", "max_size must be greater than min_size.");
        return FAILED;
      }
    }
  }
  op_dest.SetAttr("max_size", v_max_size);

  std::vector<float> v_aspect_ratio;
  v_aspect_ratio.push_back(1.0);
  if (param.aspect_ratio_size() > 0) {
    for (int32_t i = 0; i < param.aspect_ratio_size(); i++) {
      v_aspect_ratio.push_back((param.aspect_ratio(i)));
    }
  }
  op_dest.SetAttr("aspect_ratio", v_aspect_ratio);

  if (param.variance_size() > 0) {
    std::vector<float> v_variance;
    for (int32_t i = 0; i < param.variance_size(); i++) {
      v_variance.push_back((param.variance(i)));
    }
    op_dest.SetAttr("variance", v_variance);
  }

  OP_LOGI("ParseParamsPriorBox ------end!!\n");
  return SUCCESS;
}

// register PriorBox op info to GE
REGISTER_CUSTOM_OP("PriorBox")
    .FrameworkType(CAFFE)
    .OriginOpType("PriorBox")
    .ParseParamsFn(ParseParamsPriorBox)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
