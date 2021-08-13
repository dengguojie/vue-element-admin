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
 * \file common_lstm_plugin.cpp
 * \brief
 */
#include "onnx_common.h"

namespace domi {

Status ParseParamsCommonLSTM(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (nullptr == node) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int hidden_size = 0;
  bool hidden_size_flag = false;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "hidden_size" && attr.type() == ge::onnx::AttributeProto::INT) {
      hidden_size_flag = true;
    }
    if (attr.name() == "direction" && attr.type() == ge::onnx::AttributeProto::STRING) {
      op_dest.SetAttr("direction", attr.s());
    }
  }
  if (!hidden_size_flag) {
    OP_LOGD(op_dest.GetName().c_str(), "onnx LSTM op has no hidden_size attr.");
    hidden_size = 0;
  }
  op_dest.SetAttr("hidden_size", hidden_size);

  return SUCCESS;
}

REGISTER_CUSTOM_OP("CommonLSTM")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::LSTM",
                   "ai.onnx::9::LSTM",
                   "ai.onnx::10::LSTM",
                   "ai.onnx::11::LSTM",
                   "ai.onnx::12::LSTM",
                   "ai.onnx::13::LSTM"})
    .ParseParamsFn(ParseParamsCommonLSTM)
    .ImplyType(ImplyType::TVM);
}  // namespace domi