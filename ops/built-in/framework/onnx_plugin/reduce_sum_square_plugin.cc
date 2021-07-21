/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file reduce_sum_square_plugin.cc
 * \brief
 */

#include "onnx_common.h"

namespace domi {

Status parse_params_reduce_sum_square(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  std::vector<int32_t> v_axes;
  bool keep_dims = true;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "axes" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (int i = 0; i<attr.ints_size(); i++){
        v_axes.push_back(attr.ints(i));
      }
    } else if (attr.name() == "keepdims" && attr.type() == ge::onnx::AttributeProto::INT) {
      if (attr.i() != 1) {
        keep_dims = true;
      }
    }
  }

  op_dest.SetAttr("axis", v_axes);
  op_dest.SetAttr("keep_dims", keep_dims);
  return SUCCESS;
}

// register ReduceSumSquare op info to GE
REGISTER_CUSTOM_OP("SquareSumV1")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::1::ReduceSumSquare",
                 "ai.onnx::8::ReduceSumSquare",
                 "ai.onnx::9::ReduceSumSquare",
                 "ai.onnx::10::ReduceSumSquare",
                 "ai.onnx::11::ReduceSumSquare",
                 "ai.onnx::12::ReduceSumSquare",
                 "ai.onnx::13::ReduceSumSquare"})
  .ParseParamsFn(parse_params_reduce_sum_square)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
