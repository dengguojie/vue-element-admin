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
 * \file mean_variance_normalization_plugin.cpp
 * \brief
 */
#include "onnx_common.h"

namespace domi {
static const int OUTPUT_SIZE = 1;
struct MeanVarianceNormalizationAttr {
  std::vector<int> v_axes = {};
  std::vector<int> DefaultAxes = {0, 2, 3};

  bool set_axes_flag = false;
};

Status UpdateAttrFromOnnx(const ge::onnx::NodeProto* node, MeanVarianceNormalizationAttr& node_attr) {
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "axes" && attr.type() == ge::onnx::AttributeProto::INTS) {
      node_attr.set_axes_flag = true;
      for (int i = 0; i < attr.ints_size(); i++) {
        node_attr.v_axes.push_back(attr.ints(i));    
      } 
    }
  }
  return SUCCESS;
}

Status SetAttrToDesc(ge::Operator& op_dest, MeanVarianceNormalizationAttr& node_attr) {
  if (node_attr.set_axes_flag) {
    op_dest.SetAttr("axes", node_attr.v_axes);
  } else {
    OP_LOGD(op_dest.GetName().c_str(), "onnx MeanVarianceNormalization op has no axes attr, set it to [0, 2, 3].");
    op_dest.SetAttr("axes", node_attr.DefaultAxes);
  }

  return SUCCESS;
}

Status ParseParamsMeanVarianceNormalization(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  int op_output_size = node->output_size();
  if (op_output_size != OUTPUT_SIZE) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "The output of Indices is not support, transforming failed.");
    return FAILED;
  }
  MeanVarianceNormalizationAttr node_attr;
  if (UpdateAttrFromOnnx(node, node_attr) != SUCCESS) {
    return FAILED;
  }

  if (SetAttrToDesc(op_dest, node_attr) != SUCCESS) {
    return FAILED;
  }

  return SUCCESS;
}

REGISTER_CUSTOM_OP("MVNV2")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::9::MeanVarianceNormalization",
                   "ai.onnx::10::MeanVarianceNormalization",
                   "ai.onnx::11::MeanVarianceNormalization",
                   "ai.onnx::12::MeanVarianceNormalization",
                   "ai.onnx::13::MeanVarianceNormalization"})
    .ParseParamsFn(ParseParamsMeanVarianceNormalization)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
