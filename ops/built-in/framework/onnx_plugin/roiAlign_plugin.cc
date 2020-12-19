/* Copyright (c) Huawei Technologies Co., Ltd. 2012-2020. All rights reserved.
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
#include <string>
#include <vector>

#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
namespace domi {
using NodeProto = ge::onnx::NodeProto;
Status ParseParamsRoiAlign(const Message *op_src, ge::Operator &op_dest) {
  const NodeProto *node = reinterpret_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE("RoiAlign", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int output_height_value = 1;
  op_dest.SetAttr("pooled_height", output_height_value);
  int output_width_value = 1;
  op_dest.SetAttr("pooled_width", output_width_value);
  int sampling_ratio_value = 0;
  op_dest.SetAttr("sample_num", sampling_ratio_value);
  float spatial_scale_value = 1.0;
  op_dest.SetAttr("spatial_scale", spatial_scale_value);

  int default_roi_end_mode_value = 0;
  op_dest.SetAttr("roi_end_mode", default_roi_end_mode_value);

  for (auto attr : node->attribute()) {
    if (attr.name() == "output_height" &&
        attr.type() == ge::onnx::AttributeProto::INT) {
      output_height_value = attr.i();
      op_dest.SetAttr("pooled_height", output_height_value);
    } else if (attr.name() == "output_width") {
      output_width_value = attr.i();
      op_dest.SetAttr("pooled_width", output_width_value);
    } else if (attr.name() == "sampling_ratio") {
      sampling_ratio_value = attr.i();
      op_dest.SetAttr("sample_num", sampling_ratio_value);
    } else if (attr.name() == "spatial_scale") {
      spatial_scale_value = attr.f();
      op_dest.SetAttr("spatial_scale", spatial_scale_value);
    }
  }

  return SUCCESS;
}
// register ROIAlign op info to GE
REGISTER_CUSTOM_OP("ROIAlign")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::11::RoiAlign")
  .ParseParamsFn(ParseParamsRoiAlign)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
