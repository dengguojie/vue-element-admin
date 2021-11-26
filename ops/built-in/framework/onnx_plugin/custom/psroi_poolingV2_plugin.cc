/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "graph/operator.h"
#include "register/register.h"
#include "json.hpp"
#include "./../onnx_common.h"

using namespace ge;
using json = nlohmann::json;
namespace domi {
Status ParseOnnxParamsPSROIPoolingV2(const ge::Operator& op_src, ge::Operator& op_dest) {
  AscendString attrs_string;
  float spatial_scale = 0;
  int output_dim = 0;
  int group_size = 0;
  int attr_num = 0;
  if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
    json attrs = json::parse(attrs_string.GetString());
    for (json& attr : attrs["attribute"]) {
      if (attr["name"] == "spatial_scale") {
        std::string spatial = attr["f"];
        spatial_scale = atof(spatial.c_str());
        ++attr_num; 
      } else if (attr["name"] == "output_dim") {
        output_dim = attr["i"];
        ++attr_num;
      } else if (attr["name"] == "group_size") {
        group_size = attr["i"];
        ++attr_num;
      }
    }
  }

  if (attr_num != 3) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Node must have attr spatial_scale、output_dim、group_size");
    return FAILED;
  }
  op_dest.SetAttr("spatial_scale", spatial_scale);
  op_dest.SetAttr("output_dim", output_dim);
  op_dest.SetAttr("group_size", group_size);

  if (ChangeFormatFromOnnx(op_dest, 0, ge::FORMAT_NCHW, true) != SUCCESS ||
      ChangeFormatFromOnnx(op_dest, 1, ge::FORMAT_NCHW, true) != SUCCESS ||
      ChangeFormatFromOnnx(op_dest, 0, ge::FORMAT_NCHW, false) != SUCCESS) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "ChangeFormatFromOnnx fail");
    return FAILED;
  }
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PSROIPoolingV2")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::PSROIPooling"),
                   ge::AscendString("ai.onnx::9::PSROIPooling"),
                   ge::AscendString("ai.onnx::10::PSROIPooling"),
                   ge::AscendString("ai.onnx::11::PSROIPooling"),
                   ge::AscendString("ai.onnx::12::PSROIPooling"),
                   ge::AscendString("ai.onnx::13::PSROIPooling")})
    .ParseParamsByOperatorFn(ParseOnnxParamsPSROIPoolingV2)
    .ImplyType(ImplyType::TVM);
}  // domi