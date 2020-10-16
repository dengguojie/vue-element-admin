/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
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

#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"

namespace domi {
static const int DEFAULT_CONCAT_DIM = 1;

Status ParseParamsConcat(const Message* op_src, ge::Operator& op_dest) {
  OP_LOGI("Concat",
          "[PLUGIN_CONCAT]------------ParseParams Concatstart---------------");
  const caffe::LayerParameter* layer =
      dynamic_cast<const caffe::LayerParameter*>(op_src);

  if (nullptr == layer) {
    OP_LOGE("Concat", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }

  const caffe::ConcatParameter& param = layer->concat_param();
  int concat_dim = 0;
  if (param.has_axis()) {
    concat_dim = param.axis();
  } else if (param.has_concat_dim()) {
    concat_dim = param.concat_dim();
  } else {
    OP_LOGI("Concat", "Caffe Concat has no axis nor concat_dim.");
    concat_dim = DEFAULT_CONCAT_DIM;
  }
  op_dest.SetAttr("concat_dim", concat_dim);

  int n = layer->bottom_size();
  OP_LOGI("Concat",
          "[PLUGIN_CONCAT]--------------bottom_size=%d---------------", n);
  op_dest.SetAttr("N", n);
  std::shared_ptr<ge::OpDesc> op_desc =
      ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  op_desc->AddDynamicInputDesc("x", n);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("ConcatD")
    .FrameworkType(CAFFE)
    .OriginOpType("Concat")
    .ParseParamsFn(ParseParamsConcat)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
