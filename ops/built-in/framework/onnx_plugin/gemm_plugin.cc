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
 * \file gemm_plugin.cpp
 * \brief
 */
#include "onnx_common.h"

namespace domi {

Status ParseParamsGemm(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node =
      dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  bool trans_a = false;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "transA" && attr.i() != 0) {
      trans_a = true;
      break;
    }
  }
  bool trans_b = false;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "transB" && attr.i() != 0) {
      trans_b = true;
      break;
    }
  }

  op_dest.SetAttr("transpose_x1", trans_a);
  op_dest.SetAttr("transpose_x2", trans_b);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  if (op_desc == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Get op desc failed.");
    return FAILED;
  }
  //The fmap should be NCHW
  ge::GeTensorDesc output_y_desc = op_desc->GetOutputDesc(0);
  output_y_desc.SetOriginFormat(ge::FORMAT_NCHW);
  output_y_desc.SetFormat(ge::FORMAT_NCHW);
  op_desc->UpdateOutputDesc(0, output_y_desc);

  return SUCCESS;
}

// register Gemm op info to GE
REGISTER_CUSTOM_OP("MatMulV2")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::Gemm",
                   "ai.onnx::9::Gemm",
                   "ai.onnx::10::Gemm",
                   "ai.onnx::11::Gemm",
                   "ai.onnx::12::Gemm",
                   "ai.onnx::13::Gemm"})
    .ParseParamsFn(ParseParamsGemm)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
