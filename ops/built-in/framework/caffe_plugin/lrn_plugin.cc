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

Status ParseParamsLrn(const Message* op_origin, ge::Operator& op_dest) {
   // Parse parameters
  const float LRN_DEFAULT_BETA = 0.75;
  const uint32_t LRN_DEFAULT_NORM_REGION = 0;
  const uint32_t LRN_DEFAULT_LOCAL_SIZE = 5;
  const float LRN_DEFAULT_BIAS = 1.0;
  const float LRN_DEFAULT_ALPHA = 1.0;
  const string LRN_ACROSS_CHANNELS = "ACROSS_CHANNELS";
  const string LRN_WITHIN_CHANNEL = "WITHIN_CHANNEL";

  // trans op_src to op_dest
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);

  // Ckeck operator parameter's validity
  if (nullptr == layer) {
    OP_LOGE(op_dest.GetName().c_str(), "convert src op failed.");
    return FAILED;
  }

  // get layer
  const caffe::LRNParameter& param = layer->lrn_param();

  if (param.has_beta()) {
    op_dest.SetAttr("beta", param.beta());
  } else {
    op_dest.SetAttr("beta", LRN_DEFAULT_BETA);
  }

  uint32_t local_size = LRN_DEFAULT_LOCAL_SIZE;
  if (param.has_local_size()) {
    local_size = param.local_size();
    if (0 == local_size % 2) {
      ge::OpsAttrValueErrReport(op_dest.GetName(), "local_size", "odd value",
                                to_string(local_size));
      OP_LOGE(op_dest.GetName().c_str(), "LRN only supports odd values for local_size.");
      return FAILED;
    }
  }
  int64_t depth_radius = (local_size - 1) / 2;
  op_dest.SetAttr("depth_radius", depth_radius);

  uint32_t norm_region = LRN_DEFAULT_NORM_REGION;
  if (param.has_norm_region()) {
    norm_region = param.norm_region();
  }

  if (param.has_k()) {
    op_dest.SetAttr("bias", param.k());
  } else {
    op_dest.SetAttr("bias", LRN_DEFAULT_BIAS);
  }

  float alpha = LRN_DEFAULT_ALPHA;
  if (param.has_alpha()) {
    alpha = param.alpha();
  }
  op_dest.SetAttr("alpha", alpha / local_size);

  if (norm_region == LRN_DEFAULT_NORM_REGION) {
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
