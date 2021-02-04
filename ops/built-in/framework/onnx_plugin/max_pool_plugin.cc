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
 * \file max_pool_plugin.cpp
 * \brief
 */
#include <string>
#include <vector>

#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"

#include "op_log.h"

namespace domi {
static const int OUTPUT_SIZE = 1;
struct MaxPoolAttr {
  int64_t v_ceil_mode = 0;
  std::vector<int> v_ksizes = {};
  std::vector<int> v_strides = {};
  std::vector<int> v_pads = {};
  std::string v_pad = "CALCULATED";
  std::vector<int> DefaultStride = {1, 1, 1, 1};
  std::vector<int> DefaultPads = {0, 0, 0, 0};
  std::vector<int> v_dilations = {};
  int v_storage_order = 0;

  bool set_ksizes_flag = false;
  bool set_strides_flag = false;
  bool set_pads_flag = false;
};

Status UpdateAttrFromOnnx(const ge::onnx::NodeProto* node, MaxPoolAttr& node_attr) {
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "kernel_shape" && attr.type() == ge::onnx::AttributeProto::INTS) {
      node_attr.v_ksizes.push_back(1);
      node_attr.v_ksizes.push_back(1);
      if (attr.ints_size() == 2) {
        for (int i = 0; i < attr.ints_size(); i++) {
          node_attr.v_ksizes.push_back(attr.ints(i));
        }
      } else if (attr.ints_size() == 1) {
        node_attr.v_ksizes.push_back(attr.ints(0));
        node_attr.v_ksizes.push_back(attr.ints(0));
      } else {
        OP_LOGE("MaxPool", "the lenth of attr kernel size is greater then 2, may be it is 3D MaxPool.");
        return FAILED;
      }
      node_attr.set_ksizes_flag = true;
    } else if (attr.name() == "strides" && attr.type() == ge::onnx::AttributeProto::INTS) {
      node_attr.v_strides.push_back(1);
      node_attr.v_strides.push_back(1);
      if (attr.ints_size() == 2) {
        for (int i = 0; i < attr.ints_size(); i++) {
          node_attr.v_strides.push_back(attr.ints(i));
        }
      } else if (attr.ints_size() == 1) {
        node_attr.v_strides.push_back(attr.ints(0));
        node_attr.v_strides.push_back(attr.ints(0));
      } else {
        OP_LOGE("MaxPool", "the lenth of attr strides is greater then 2, may be it is 3D MaxPool.");
        return FAILED;
      }
      node_attr.set_strides_flag = true;
    } else if (attr.name() == "auto_pad" && attr.type() == ge::onnx::AttributeProto::STRING) {
      if (attr.s() == "VALID") {
        node_attr.v_pad = "VALID";
      } else if (attr.s() == "NOTSET") {
        node_attr.v_pad = "CALCULATED";
      } else if (attr.s() == "SAME_UPPER") {
        node_attr.v_pad = "SAME";
      } else if (attr.s() == "SAME_LOWER") {
        node_attr.v_pad = "SAME";
        OP_LOGW("MaxPool", "value of auto_pad is same_lower, the accuracy error will be large!");
      } else {
        OP_LOGE("MaxPool", "value of auto_pad is invalid, transform failed.");
        return FAILED;
      }
    } else if (attr.name() == "pads" && attr.type() == ge::onnx::AttributeProto::INTS) {
      if (attr.ints_size() == 4) {
        // adjust padding order from NCHW to NHWC.
        std::vector<int> rank = {0, 2, 3, 1};
        for (auto i : rank) {
          node_attr.v_pads.push_back(attr.ints(i));
        }
      } else if (attr.ints_size() == 1) {
        node_attr.v_pads.push_back(attr.ints(0));
        node_attr.v_pads.push_back(attr.ints(0));
        node_attr.v_pads.push_back(attr.ints(0));
        node_attr.v_pads.push_back(attr.ints(0));
      } else {
        OP_LOGE("MaxPool", "the lenth of attr pads is not equal to 4 or 1, transform failed.");
        return FAILED;
      }
      node_attr.set_pads_flag = true;
    } else if (attr.name() == "ceil_mode" && attr.type() == ge::onnx::AttributeProto::INT) {
      node_attr.v_ceil_mode = attr.i();
    } else if (attr.name() == "dilations" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (int i = 0; i < attr.ints_size(); i++) {
        node_attr.v_dilations.push_back(attr.ints(i));
      }
    } else if (attr.name() == "storage_order" && attr.type() == ge::onnx::AttributeProto::INT) {
      node_attr.v_storage_order = attr.i();
    }
  }
  return SUCCESS;
}

Status SetAttrToDesc(ge::Operator& op_dest, MaxPoolAttr& node_attr) {
  if (node_attr.v_storage_order != 0) {
    OP_LOGE("MaxPool", "the storage order used column is not supported, failed to transfrom.");
    return FAILED;
  }

  if (node_attr.v_dilations.begin() != node_attr.v_dilations.end()) {
    for (auto i : node_attr.v_dilations) {
      if (i != 1) {
        OP_LOGE("MaxPool", "the value of attr dilations is not 1, failed to transfrom.");
        return FAILED;
      }
    }
  }

  if (node_attr.v_ceil_mode == 0) {
    op_dest.SetAttr("ceil_mode", false);
  } else {
    op_dest.SetAttr("ceil_mode", true);
  }

  if (node_attr.set_ksizes_flag) {
    // check whether the parameter of kernel shape is valid.
    if (node_attr.v_ksizes[1] * node_attr.v_ksizes[2] > 255) {
      OP_LOGE("MaxPool", "the value of kernel shape is out of constraints, failed to transfrom.");
      return FAILED;
    }
    op_dest.SetAttr("ksize", node_attr.v_ksizes);
  } else {
    OP_LOGI("MaxPool", "onnx MaxPool op has no ksize attr, set it to 1.");
    op_dest.SetAttr("ksize", node_attr.DefaultStride);
  }

  if (node_attr.set_strides_flag) {
    // check whether the parameter of strides is valid.
    if (node_attr.v_strides[1] > 63 || node_attr.v_strides[2] > 63) {
      OP_LOGE("MaxPool", "the value of strides is out of constraints, failed to transfrom.");
      return FAILED;
    }
    op_dest.SetAttr("strides", node_attr.v_strides);
  } else {
    OP_LOGI("MaxPool", "onnx MaxPool op has no strides attr, use default.");
    op_dest.SetAttr("strides", node_attr.DefaultStride);
  }

  if (node_attr.set_pads_flag) {
    op_dest.SetAttr("pads", node_attr.v_pads);
  } else {
    OP_LOGI("MaxPool", "onnx MaxPool op has no pads attr, use default.");
    op_dest.SetAttr("pads", node_attr.DefaultPads);
  }

  op_dest.SetAttr("padding_mode", node_attr.v_pad);
  return SUCCESS;
}

Status ParseParamsMaxPool(const Message* op_src, ge::Operator& op_dest) {
  OP_LOGI("MaxPool", "[PLUGIN_MaxPool]--------------ParseParamsMaxPool  start---------------");
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (nullptr == node) {
    OP_LOGE("MaxPool", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  int op_output_size = node->output_size();
  if (op_output_size != OUTPUT_SIZE) {
    OP_LOGE("MaxPool", "The output of Indices is not support, transforming failed.");
    return FAILED;
  }
  MaxPoolAttr node_attr;
  if (UpdateAttrFromOnnx(node, node_attr) != SUCCESS) {
    return FAILED;
  }

  if (SetAttrToDesc(op_dest, node_attr) != SUCCESS) {
    return FAILED;
  }

  OP_LOGI("MaxPool", "--------------ParseParamsMaxPool  end---------------");

  return SUCCESS;
}

REGISTER_CUSTOM_OP("MaxPoolV3")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::11::MaxPool")
    .ParseParamsFn(ParseParamsMaxPool)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
