/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the
License.
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
#include "common/util/error_manager/error_manager.h"

using namespace ge;
namespace domi {

const float LRN_DEFAULT_BETA = 0.75;
const uint32_t LRN_DEFAULT_NORM_REGION = 0;
const uint32_t LRN_DEFAULT_LOCAL_SIZE = 5;
const float LRN_DEFAULT_BIAS = 1.0;
const float LRN_DEFAULT_ALPHA = 1.0;
const std::string ACROSS_CHANNELS = "ACROSS_CHANNELS";
const std::string WITHIN_CHANNEL = "WITHIN_CHANNEL";


// Parse parameters
Status ParseParamsLrn(const Message* op_origin, ge::Operator& op_dest) {
    // trans op_src to op_dest
  const caffe::LayerParameter* layer = \
      dynamic_cast<const caffe::LayerParameter*>(op_origin);

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
      OP_LOGE(op_dest.GetName().c_str(),
        "LRN only supports odd values for local_size.");
      map<string, string> err_map;
      err_map["op_name"] = "LRN";
      err_map["param_name"] = "local_size";
      err_map["excepted_value"] = "odd value";
      err_map["input_value"] = to_string(local_size);
      std::string report_error_code = "E70007";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);     
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
    op_dest.SetAttr("norm_region", ACROSS_CHANNELS);
  } else {
    op_dest.SetAttr("norm_region", WITHIN_CHANNEL);
  }

  return SUCCESS;
}

REGISTER_CUSTOM_OP("LRN")
  .FrameworkType(CAFFE)
  .OriginOpType("LRN")
  .ParseParamsFn(ParseParamsLrn)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
