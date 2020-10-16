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

/*
 * parse tile parameters
 * param[in] op_src  source op description
 * param[out] op_dst destination op description
 * return SUCCESS:parse success
 *        FAILED: parse failed
 */
Status ParseParamsTileWithAxis(const Message* op_src, ge::Operator& op_dst) {
  const caffe::LayerParameter* layer =
      static_cast<const caffe::LayerParameter*>(op_src);
  if (layer == nullptr) {
    OP_LOGI("TileWithAxis",
            "[TileWithAxis_Plugin] cast op_src to LayerParameter failed.");
    return FAILED;
  }

  const caffe::TileParameter& param = layer->tile_param();
  if (param.has_axis()) {
    op_dst.SetAttr("axis", (int64_t)(param.axis()));
  } else {
    op_dst.SetAttr("axis", (int64_t)(1));
  }

  if (param.has_tiles()) {
     op_dst.SetAttr("tiles", (int64_t)(param.tiles()));
  }

  return SUCCESS;
}

// register TileWithAxis op info to GE
REGISTER_CUSTOM_OP("TileWithAxis")
    .FrameworkType(CAFFE)
    .OriginOpType("Tile")
    .ParseParamsFn(ParseParamsTileWithAxis);
}  // namespace domi
