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
struct AvgPoolAttr {
  std::vector<int> ksizes = {};
  std::vector<int> strides = {};
  std::vector<int> pads = {};
  std::string auto_pad = "SAME";
  int64_t ceil_mode = 0;
};

struct AvgPoolAttrFlag {
  bool set_ksizes_flag = false;
  bool set_strides_flag = false;
  bool set_pads_flag = false;
};

static const char DATA_FORMAT[] = "NCHW";
// kernel_shape数组的合理大小
static const int KERNEL_SHAPE_SIZE = 2;
// strides数组的合理大小
static const int STRIDES_SIZE = 2;
// pads数组的合理大小
static const int PADS_SIZE = 4;

Status SetKernelAttr(int size, std::vector<int> data, AvgPoolAttr &node_attr,
                     AvgPoolAttrFlag &is_node_attrset) {
  if (size == KERNEL_SHAPE_SIZE) {
    for (int i = 0; i < size; i++) {
      node_attr.ksizes.push_back(data[i]);
    }
  } else if (size == 1) {
    node_attr.ksizes.push_back(data[0]);
    node_attr.ksizes.push_back(data[0]);
  }
  is_node_attrset.set_ksizes_flag = true;
  return SUCCESS;
}

Status SetStridesAttr(int size, std::vector<int> data, AvgPoolAttr &node_attr,
                      AvgPoolAttrFlag &is_node_attrset) {
  if (size == STRIDES_SIZE) {
    for (int i = 0; i < size; i++) {
      node_attr.strides.push_back(data[i]);
    }
  } else if (size == 1) {
    node_attr.strides.push_back(data[0]);
    node_attr.strides.push_back(data[0]);
  }
  is_node_attrset.set_strides_flag = true;
  return SUCCESS;
}

Status SetPadsAttr(int size, std::vector<int> data, AvgPoolAttr &node_attr,
                   AvgPoolAttrFlag &is_node_attrset) {
  if (size == PADS_SIZE) {
    for (int i = 0; i < size; i++) {
      node_attr.pads.push_back(data[i]);
    }
  } else if (size == 1) {
    node_attr.pads.push_back(data[0]);
    node_attr.pads.push_back(data[0]);
    node_attr.pads.push_back(data[0]);
    node_attr.pads.push_back(data[0]);
  }
  is_node_attrset.set_pads_flag = true;
  return SUCCESS;
}

Status CheckAttribute(const NodeProto *node, AvgPoolAttr &node_attr,
                      AvgPoolAttrFlag &is_node_attrset) {
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "auto_pad" &&
        attr.type() == ge::onnx::AttributeProto::STRING) {
      if (attr.s() == "VALID") {
        node_attr.auto_pad = "VALID";
      } else {
        node_attr.auto_pad = "SAME";
      }
      continue;
    }

    if (attr.name() == "ceil_mode" &&
        attr.type() == ge::onnx::AttributeProto::INT) {
      node_attr.ceil_mode = attr.i();
      continue;
    }

    int size = attr.ints_size();
    std::vector<int> data;
    for (int i = 0; i < size; i++) {
      data.push_back(attr.ints(i));
    }

    if (attr.name() == "kernel_shape" &&
        attr.type() == ge::onnx::AttributeProto::INTS) {
      SetKernelAttr(size, data, node_attr, is_node_attrset);
    }
    if (attr.name() == "strides" &&
        attr.type() == ge::onnx::AttributeProto::INTS) {
      SetStridesAttr(size, data, node_attr, is_node_attrset);
    }
    if (attr.name() == "pads" &&
        attr.type() == ge::onnx::AttributeProto::INTS) {
      SetPadsAttr(size, data, node_attr, is_node_attrset);
    }
  }
  return SUCCESS;
}

Status ParseParamsAveragePool(const Message *op_src, ge::Operator &op_dest) {
  const NodeProto *node = reinterpret_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE("AveragePool", "reinterpret_cast op_src to NodeProto failed.");
    return FAILED;
  }

  std::vector<int> default_stride = {1, 1};
  std::vector<int> default_pads = {0, 0, 0, 0};

  AvgPoolAttr node_attr;
  AvgPoolAttrFlag is_node_attrset;

  CheckAttribute(node, node_attr, is_node_attrset);

  op_dest.SetAttr("mode", 1);  // 0:max pooling or 1:avg pooling

  // 和pooling的ceil mode属性是反的
  if (node_attr.ceil_mode == 0) {
    op_dest.SetAttr("ceil_mode", 1);
  } else {
    op_dest.SetAttr("ceil_mode", 0);
  }

  if (is_node_attrset.set_ksizes_flag) {
    op_dest.SetAttr("window", node_attr.ksizes);
  } else {
    OP_LOGI("AveragePool", "onnx AveragePool op has no ksize attr");
    op_dest.SetAttr("window", default_stride);
  }

  if (is_node_attrset.set_strides_flag) {
    op_dest.SetAttr("stride", node_attr.strides);
  } else {
    OP_LOGI("AveragePool", "onnx AveragePool use default.");
    op_dest.SetAttr("strides", default_stride);
  }

  if (is_node_attrset.set_pads_flag) {
    op_dest.SetAttr("pad", node_attr.pads);
  } else {
    OP_LOGI("AveragePool", "onnx AveragePool use default.");
    op_dest.SetAttr("pad", default_pads);
  }
  return SUCCESS;
}

REGISTER_CUSTOM_OP("Pooling")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::11::AveragePool")
  .ParseParamsFn(ParseParamsAveragePool)
  .ImplyType(ImplyType::TVM);
}  //  namespace domi
