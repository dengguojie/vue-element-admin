/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file affine_grid.cc
 * \brief
 */
#include "onnx_common.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;

Status ParseParamsEmbeddingBag(const Message *op_src, ge::Operator &op_dest) {

  const NodeProto *node = dynamic_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  // set attr mode_value
  std::string mode_value;
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "mode" && attr.type() == ge::onnx::AttributeProto::STRING) {
      mode_value = attr.s();
      op_dest.SetAttr("mode", mode_value);
    }
  }
  // set attr scale_grad_by_freq
  bool scale_grad_by_freq = false;
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "scale_grad_by_freq" && attr.i() != 0) {
      scale_grad_by_freq = true;
      break;
    }
  }
  op_dest.SetAttr("scale_grad_by_freq", scale_grad_by_freq);
  // set attr sparse
  bool sparse = false;
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "sparse" && attr.i() != 0) {
      sparse = true;
      break;
    }
  }
  op_dest.SetAttr("sparse", sparse);
  // set attr include_last_offset
  bool include_last_offset = false;
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "include_last_offset" && attr.i() != 0) {
      include_last_offset = true;
      break;
    }
  }
  op_dest.SetAttr("include_last_offset", include_last_offset);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("EmbeddingBag")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::EmbeddingBag",
                    "ai.onnx::9::EmbeddingBag",
                    "ai.onnx::10::EmbeddingBag",
                    "ai.onnx::11::EmbeddingBag",
                    "ai.onnx::12::EmbeddingBag",
                    "ai.onnx::13::EmbeddingBag"})
    .ParseParamsFn(ParseParamsEmbeddingBag)
    .ImplyType(ImplyType::TVM);
} // namespace domi
