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
 * \file global_average_pool_plugin.cpp
 * \brief
 */
#include "onnx_common.h"

namespace domi {

static const int DEFAULT_AXIS = 0;

Status ParseParamsGlobalAveragePool(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (nullptr == node) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  std::vector<int> strides{1, 1, 1, 1};
  op_dest.SetAttr("strides", strides);

  string padding = "VALID";
  op_dest.SetAttr("padding", padding);

  string data_format = "NCHW";
  op_dest.SetAttr("data_format", data_format);

  std::vector<int> ksize{1, 1, -1, -1};
  op_dest.SetAttr("ksize", ksize);

  return SUCCESS;
}

REGISTER_CUSTOM_OP("AvgPool")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::GlobalAveragePool",
                   "ai.onnx::9::GlobalAveragePool",
                   "ai.onnx::10::GlobalAveragePool",
                   "ai.onnx::11::GlobalAveragePool",
                   "ai.onnx::12::GlobalAveragePool",
                   "ai.onnx::13::GlobalAveragePool"})
    .ParseParamsFn(ParseParamsGlobalAveragePool)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
