/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file max_roi_pool_plugin.cc
 * \brief
 */
#include "onnx_common.h"
#include "array_ops.h"

namespace domi {

Status ParseParamsMaxRoiPool(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
      ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
      return FAILED;
  }
  float spatial_scale = 1.0;
  std::vector<int> pooled_shape;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "pooled_shape" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (int i = 0; i < attr.ints_size(); i++) {
        pooled_shape.push_back(attr.ints(i));
      }
    } else if (attr.name() == "spatial_scale" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      spatial_scale = static_cast<float>(attr.f());
    }
  }
  if (pooled_shape.size() != 2) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Obtain attr pooled_shape failed.");
    return FAILED;
  }

  op_dest.SetAttr("pooled_h", pooled_shape[0]);
  op_dest.SetAttr("pooled_w", pooled_shape[1]);
  op_dest.SetAttr("spatial_scale_h", spatial_scale);
  op_dest.SetAttr("spatial_scale_w", spatial_scale);
  return SUCCESS;
}

//register MaxRoiPool op info to GE
REGISTER_CUSTOM_OP("ROIPooling")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::1::MaxRoiPool",
                   "ai.onnx::8::MaxRoiPool",
                   "ai.onnx::9::MaxRoiPool",
                   "ai.onnx::10::MaxRoiPool",
                   "ai.onnx::11::MaxRoiPool",
                   "ai.onnx::12::MaxRoiPool",
                   "ai.onnx::13::MaxRoiPool"})
    .ParseParamsFn(ParseParamsMaxRoiPool)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
