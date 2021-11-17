/* Copyright (C) 2020. Huawei Technologies Co., Ltd. All
rights reserved.
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
#include "../onnx_common.h"
#include "json.hpp"
using namespace std;
using namespace ge;
using json = nlohmann::json;
namespace domi {
const int kTypeFloat = 1;

template <typename T>
void GetAttrListFromJson(json& attr, std::vector<T>& val, std::string& dtype) {
  int num = attr[dtype].size();
  for (int i = 0; i < num; ++i) {
    val.push_back(attr[dtype][i].get<T>());
  }
}

Status ParseParamsBoundingBoxDecode(const ge::Operator& op_src, ge::Operator& op_dest) {
  std::vector<int64_t> max_shape;
  std::vector<float> means;
  std::vector<float> stds;
  float wh_ratio_clip = 0.016f;
  std::string wh_ratio_clip_str;
  std::string dtype = "ints";
  ge::AscendString attrs_string;
  if (ge::GRAPH_SUCCESS == op_src.GetAttr("attribute", attrs_string)) {
    json attrs = json::parse(attrs_string.GetString());
    for (json attr : attrs["attribute"]) {
      if (attr["name"] == "max_shape") {
        dtype = "ints";
        GetAttrListFromJson(attr, max_shape, dtype);
      }

      if (attr["name"] == "means") {
        dtype = "floats";
        GetAttrListFromJson(attr, means, dtype);
      }

      if (attr["name"] == "stds") {
        dtype = "floats";
        GetAttrListFromJson(attr, stds, dtype);
      }

      if (attr["name"] == "wh_ratio_clip" && attr["type"] == kTypeFloat) {
        wh_ratio_clip_str = attr["f"];  // float type in json has accuracy loss, so we use string type to store it
        wh_ratio_clip = atof(wh_ratio_clip_str.c_str());
        op_dest.SetAttr("wh_ratio_clip", wh_ratio_clip);
      }
    }
  }
  if (max_shape.empty() || means.empty() || stds.empty()) {
    return FAILED;
  }

  op_dest.SetAttr("max_shape", max_shape);
  op_dest.SetAttr("means", means);
  op_dest.SetAttr("stds", stds);

  return SUCCESS;
}

// register op info to GE
REGISTER_CUSTOM_OP("BoundingBoxDecode")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::BoundingBoxDecode"),
                   ge::AscendString("ai.onnx::9::BoundingBoxDecode"),
                   ge::AscendString("ai.onnx::10::BoundingBoxDecode"),
                   ge::AscendString("ai.onnx::11::BoundingBoxDecode"),
                   ge::AscendString("ai.onnx::12::BoundingBoxDecode"),
                   ge::AscendString("ai.onnx::13::BoundingBoxDecode")})
    .ParseParamsByOperatorFn(ParseParamsBoundingBoxDecode)
    .ImplyType(ImplyType::TVM);
}  // namespace domi