/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {

Status ParseParamAscendAntiQuant(const Message *op_src, ge::Operator &op_dest) {
  const caffe::LayerParameter* layer = \
      dynamic_cast<const caffe::LayerParameter*>(op_src);
  // Ckeck operator parameter's validity
  if (nullptr == layer) {
    OP_LOGE(op_dest.GetName().c_str(), "convert src op failed.");
    return FAILED;
  }

  // get layer
  const caffe::QuantParameter& param = layer->quant_param();

  if (param.has_scale()) {
    op_dest.SetAttr("scale", param.scale());
  } else {
    op_dest.SetAttr("scale", float(1));
  }

  if (param.has_offset()) {
    op_dest.SetAttr("offset", float(*((signed char*)param.offset().c_str())) * float(-1.0));
  } else {
    op_dest.SetAttr("offset", float(0));
  }
  OP_LOGW(op_dest.GetName().c_str(), "****** offset: %f", float(*((signed char*)param.offset().c_str())) * float(-1.0));

  op_dest.SetAttr("sqrt_mode", false);

  return SUCCESS;
}

// register AscendDequant op info to GE
REGISTER_CUSTOM_OP("AscendAntiQuant")
  .FrameworkType(CAFFE)
  .OriginOpType("AntiQuant")
  .ParseParamsFn(ParseParamAscendAntiQuant)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
