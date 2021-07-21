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
 * \file reduce_mean_plugin.cpp
 * \brief
 */
#include "onnx_common.h"

namespace domi {

Status ParseParamsReduceL2(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int p_num = 2;
  std::vector<int> v_axes = {};
  bool keep_dims = true;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "axes" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (int i = 0; i < attr.ints_size(); i++) {
        v_axes.push_back(attr.ints(i));
      }
    } else if (attr.name() == "keepdims" && attr.type() == ge::onnx::AttributeProto::INT) {
      keep_dims = (attr.i() == 1);
    }
  }

  op_dest.SetAttr("axes", v_axes);
  op_dest.SetAttr("keepdim", keep_dims);
  op_dest.SetAttr("p", p_num);
  return SUCCESS;
}

// register ReduceMean op info to GE
REGISTER_CUSTOM_OP("LpNorm")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::ReduceL2",
                   "ai.onnx::9::ReduceL2",
                   "ai.onnx::10::ReduceL2",
                   "ai.onnx::11::ReduceL2",
                   "ai.onnx::12::ReduceL2",
                   "ai.onnx::13::ReduceL2"})
    .ParseParamsFn(ParseParamsReduceL2)
    .ImplyType(ImplyType::TVM);
}
