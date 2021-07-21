/* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this
 * file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http:// www.apache.org/licenses/LICENSE-2.0
 */
#include "../onnx_common.h"

using namespace std;
using namespace ge;
using ge::Operator;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
using OpDesc = std::shared_ptr<ge::OpDesc>;

Status ParseParamsInt8GivenTensorFill(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (nullptr == node) {
    ONNX_PLUGIN_LOGE("ParseParamsInt8GivenTensorFill", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  
  std::vector<int32_t> vals;
  std::vector<int64_t> dims;
  int zero_point = 0;
  std::string str_vals = "";
  for (auto &attr : node->attribute()) {
    if (attr.name() == "values") {
      str_vals = attr.s();
    } else if (attr.name() == "shape") {
      int num = attr.ints_size();
      for (int i = 0; i < num; ++i) {
        dims.push_back(attr.ints(i));
      }
    } else if (attr.name().find("zero_point") != std::string::npos) {
      zero_point = attr.i();
    }
  }
  

  if (str_vals.empty()) {
    ONNX_PLUGIN_LOGE("ParseParamsInt8GivenTensorFill", "Must have attr values");
    return FAILED;
  }

  stringstream sstr(str_vals);
  std::string token;
  while (getline(sstr, token, ',')) {
    if (!token.empty()) {
      vals.push_back(atoi(token.c_str()));
    }
    token = "";
  }
  
  if (zero_point != 0) {
    int size = vals.size();
    for (int i = 0; i < size; ++i) {
      vals[i] -= zero_point;
    }
  }

  if (dims.empty()) {
    dims.push_back((int64_t)vals.size());
  }

  ge::Shape shape(dims);
  auto data_type = zero_point == 0 ? ge::DT_UINT8 : ge::DT_INT8;
  TensorDesc tensorDesc(shape, ge::FORMAT_NCHW, data_type);
  if (data_type == ge::DT_INT8) {
    std::vector<int8_t> vals_int;
    int size = vals.size();
    for (int i = 0; i < size; ++i) {
      vals_int.push_back((int8_t)vals[i]);
    }
    ge::Tensor value_tensor(tensorDesc, reinterpret_cast<uint8_t*>(vals_int.data()), vals_int.size() * sizeof(int8_t));
    op_dest.SetAttr("value", value_tensor);
  } else {
    std::vector<uint8_t> vals_uint;
    int size = vals.size();
    for (int i = 0; i < size; ++i) {
      vals_uint.push_back((uint8_t)vals[i]);
    }
    ge::Tensor value_tensor(tensorDesc, vals_uint.data(), vals_uint.size() * sizeof(uint8_t));
    op_dest.SetAttr("value", value_tensor);
  }
  return SUCCESS;
}


REGISTER_CUSTOM_OP("Constant")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::Int8GivenTensorFill",
                   "ai.onnx::9::Int8GivenTensorFill",
                   "ai.onnx::10::Int8GivenTensorFill",
                   "ai.onnx::11::Int8GivenTensorFill",
                   "ai.onnx::12::Int8GivenTensorFill",
                   "ai.onnx::13::Int8GivenTensorFill"})
    .ParseParamsFn(ParseParamsInt8GivenTensorFill)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
