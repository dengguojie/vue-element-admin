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
 * \file lrn_plugin.cpp
 * \brief
 */
#include <string>
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"
#include "../../op_proto/util/error_util.h"

namespace domi {
Status ParseParamsLrn(const Message* op_origin, ge::Operator& op_dest)
{
   // Parse parameters
  const float lrnDefaultBeta = 0.75;
  const uint32_t lrnDefaultNormRegion = 0;
  const uint32_t lrnDefaultLocalSize = 5;
  const float lrnDefaultBias = 1.0;
  const float lrnDefaultAlpha = 1.0;
  const string LRN_ACROSS_CHANNELS = "ACROSS_CHANNELS";
  const string LRN_WITHIN_CHANNEL = "WITHIN_CHANNEL";

  // trans op_src to op_dest
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);

  // Ckeck operator parameter's validity
  if (layer == nullptr) {
    OP_LOGE(op_dest.GetName().c_str(), "convert src op failed.");
    return FAILED;
  }

  // get layer
  const caffe::LRNParameter& param = layer->lrn_param();

  if (param.has_beta()) {
    op_dest.SetAttr("beta", param.beta());
  } else {
    op_dest.SetAttr("beta", lrnDefaultBeta);
  }

  uint32_t localSize = lrnDefaultLocalSize;
  if (param.has_local_size()) {
    localSize = param.local_size();
    if (localSize % 2 == 0) {
      OP_LOGE(op_dest.GetName().c_str(), "LRN only supports odd values for localSize.");
      return FAILED;
    }
  }
  int64_t depthRadius = (localSize - 1) / 2;
  op_dest.SetAttr("depth_radius", depthRadius);

  uint32_t normRegion = lrnDefaultNormRegion;
  if (param.has_norm_region()) {
    normRegion = param.norm_region();
  }

  if (param.has_k()) {
    op_dest.SetAttr("bias", param.k());
  } else {
    op_dest.SetAttr("bias", lrnDefaultBias);
  }

  float alpha = lrnDefaultAlpha;
  if (param.has_alpha()) {
    alpha = param.alpha();
  }
  op_dest.SetAttr("alpha", alpha / localSize);

  if (normRegion == lrnDefaultNormRegion) {
    op_dest.SetAttr("norm_region", LRN_ACROSS_CHANNELS);
  } else {
    op_dest.SetAttr("norm_region", LRN_WITHIN_CHANNEL);
  }

  return SUCCESS;
}

REGISTER_CUSTOM_OP("LRN")
    .FrameworkType(CAFFE)
    .OriginOpType("LRN")
    .ParseParamsFn(ParseParamsLrn)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
