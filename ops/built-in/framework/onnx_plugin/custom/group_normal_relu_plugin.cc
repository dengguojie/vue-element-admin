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
Status ParseOnnxParamsGroupNormRelu(const ge::Operator& op_src, ge::Operator& op_dest) {
  AscendString attrs_string;
  int num_groups = 0;
  float eps = 0;
  if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
    json attrs = json::parse(attrs_string.GetString());
    for (json& attr : attrs["attribute"]) {
      if (attr["name"] == "eps") {
        std::string eps_str = attr["f"];
        eps = atof(eps_str.c_str());
      } else if (attr["name"] == "num_groups") {
        num_groups = attr["i"];
      }
    }
  }

  op_dest.SetAttr("num_groups", num_groups);
  op_dest.SetAttr("eps", eps);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("GroupNormRelu")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::GroupNormRelu"),
                   ge::AscendString("ai.onnx::9::GroupNormRelu"),
                   ge::AscendString("ai.onnx::10::GroupNormRelu"),
                   ge::AscendString("ai.onnx::11::GroupNormRelu"),
                   ge::AscendString("ai.onnx::12::GroupNormRelu"),
                   ge::AscendString("ai.onnx::13::GroupNormRelu")})
    .ParseParamsByOperatorFn(ParseOnnxParamsGroupNormRelu)
    .ImplyType(ImplyType::TVM);
}  // domi