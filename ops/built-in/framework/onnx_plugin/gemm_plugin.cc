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
#include <string>
#include <vector>
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"

namespace domi {

Status ParseParamsGemm(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node =
      dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("Gemm", "Dynamic cast op_src to NodeProto failed.");
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

  return SUCCESS;
}

// register Gemm op info to GE
REGISTER_CUSTOM_OP("MatMulV2")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::9::Gemm",
                   "ai.onnx::10::Gemm",
                   "ai.onnx::11::Gemm",
                   "ai.onnx::12::Gemm",
                   "ai.onnx::13::Gemm"})
    .ParseParamsFn(ParseParamsGemm)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
