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
 * \file batch_normalization_plugin.cpp
 * \brief
 */
#include "onnx_common.h"

namespace domi {

Status ParseParamsBatchNorm(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  float epsilon = 0.00001;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "epsilon" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      epsilon = attr.f();
      break;
    }
  }
  op_dest.SetAttr("epsilon", epsilon);

  const std::string data_format = "NCHW";
  op_dest.SetAttr("data_format", data_format);

  const bool is_training = false;
  op_dest.SetAttr("is_training", is_training);
  op_dest.SetAttr("onnx", "onnx");
  return SUCCESS;
}

// register ReduceMean op info to GE
REGISTER_CUSTOM_OP("BatchNorm")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::BatchNormalization",
                   "ai.onnx::9::BatchNormalization",
                   "ai.onnx::10::BatchNormalization",
                   "ai.onnx::11::BatchNormalization",
                   "ai.onnx::12::BatchNormalization",
                   "ai.onnx::13::BatchNormalization"})
    .ParseParamsFn(ParseParamsBatchNorm)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
