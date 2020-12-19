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
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;
using OpDesc = std::shared_ptr<ge::OpDesc>;
static const int DATA_TYPE_INT32 = 6;
Status ParseParamsMultinomial(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("ParseParamsMultinomial",
            "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  int data_type = DATA_TYPE_INT32;
  float seed = 0;
  int sample_size = 1;
  for (auto attr : node->attribute()) {
    if (attr.name() == "dtype") {
      data_type = attr.i();
    } else if (attr.name() == "sample_size") {
      sample_size = attr.i();
    } else if (attr.name() == "seed") {
      seed = attr.f();
    }
  }
  op_dest.SetAttr("dtype", data_type);
  op_dest.SetAttr("sample_size", sample_size);
  op_dest.SetAttr("seed", seed);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("MultinomialFuss")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::11::Multinomial")
  .ParseParamsFn(ParseParamsMultinomial)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
