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
static bool SetImgAttr(const caffe::PriorBoxParameter& param, ge::Operator& op_dest) {
  if (param.has_img_h() || param.has_img_w()) {
    if (param.has_img_size()) {
      OP_LOGE("PriorBox", "Either img_size or img_h/img_w should be specified; not both.");
      return false;
    }
    if (param.img_h() < 0) {
      OP_LOGE("PriorBox", "img_h should be larger than 0.");
      return false;
    }
    op_dest.SetAttr("img_h", param.img_h());

    if (param.img_w() < 0) {
      OP_LOGE("PriorBox", "img_w should be larger than 0.");
      return false;
    }
    op_dest.SetAttr("img_w", param.img_w());
  } else if (param.has_img_size()) {
    if (param.img_size() < 0) {
      OP_LOGE("PriorBox", "img_size should be larger than 0.");
      return false;
    }
    op_dest.SetAttr("img_h", param.img_size());
    op_dest.SetAttr("img_w", param.img_size());
  } else {
    op_dest.SetAttr("img_h", DEFAULT_SIZE);
    op_dest.SetAttr("img_w", DEFAULT_SIZE);
  }
  return true;
}
static bool setStepAttr(const caffe::PriorBoxParameter& param, ge::Operator& op_dest) {
  if (param.has_step_h() || param.has_step_w()) {
    if (param.has_step()) {
      OP_LOGE("PriorBox", "Either step or step_h/step_w should be specified; not both.");
      return false;
    }
    if (param.step_h() < 0) {
      OP_LOGE("PriorBox", "step_h should be larger than 0.");
      return false;
    }
    op_dest.SetAttr("step_h", param.step_h());

    if (param.step_w() < 0) {
      OP_LOGE("PriorBox", "step_w should be larger than 0.");
      return false;
    }
    op_dest.SetAttr("step_w", param.step_w());
  } else if (param.has_step()) {
    if (param.step() < 0) {
      OP_LOGE("PriorBox", "step should be larger than 0.");
      return false;
    }
    op_dest.SetAttr("step_h", param.step());
    op_dest.SetAttr("step_w", param.step());
  } else {
    op_dest.SetAttr("step_h", DEFAULT_STEP);
    op_dest.SetAttr("step_w", DEFAULT_STEP);
  }
  return true;
}
static bool setStrideAttr(const caffe::PriorBoxParameter& param, ge::Operator& op_dest) {
  std::vector<float> v_min_size;
  if (param.min_size_size() > 0) {
    for (int32_t i = 0; i < param.min_size_size(); i++) {
      v_min_size.push_back(param.min_size(i));
      if (param.min_size(i) <= 0) {
        OP_LOGE("PriorBox", "min_size must be positive.");
        return false;
      }
    }
    op_dest.SetAttr("min_size", v_min_size);
  } else {
    OP_LOGE("PriorBox", "Must provide min_size.");
    return false;
  }

  std::vector<float> v_max_size;
  if (param.max_size_size() > 0) {
    if (param.max_size_size() != param.min_size_size()) {
      return FAILED;
    }
    for (int32_t i = 0; i < param.max_size_size(); i++) {
      v_max_size.push_back(param.max_size(i));
      if (param.max_size(i) <= param.min_size(i)) {
        OP_LOGE("PriorBox", "max_size must be greater than min_size.");
        return false;
      }
    }
  }
  op_dest.SetAttr("max_size", v_max_size);
  return true;
}

// Caffe ParseParams
Status ParseParamsPriorBox(const Message* op_origin, ge::Operator& op_dest) {
  OP_LOGI("enter into ParseParamsPriorBox ------begin!!\n");
  // trans op_src to op_dest
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (layer == nullptr) {
    OP_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to LayerParameter failed.\n");
    return FAILED;
  }
  // get layer
  const caffe::PriorBoxParameter& param = layer->prior_box_param();

    if (!SetImgAttr(param, op_dest)) {
    OP_LOGE(op_dest.GetName().c_str(), "set Img failed.");
    return FAILED;
  }
  if (!setStepAttr(param, op_dest)) {
    OP_LOGE(op_dest.GetName().c_str(), "set Step failed.");
    return FAILED;
  }
  if (!setStrideAttr(param, op_dest)) {
    OP_LOGE(op_dest.GetName().c_str(), "set Stride failed.");
    return FAILED;
  }


  const int DEFAULT_BOTTOM_SIZE = 2;
  int n = layer->bottom_size();
  if (n != DEFAULT_BOTTOM_SIZE) {
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

  float offset = 0.0;
  if (param.has_offset()) {
    offset = param.offset();
  } else {
    offset = DEFAULT_OFFSET;
  }
  op_dest.SetAttr("offset", offset);

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
