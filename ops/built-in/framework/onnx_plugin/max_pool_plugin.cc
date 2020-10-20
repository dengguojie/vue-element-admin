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

Status ParseParamsMaxPool(const Message* op_src, ge::Operator& op_dest) {
  OP_LOGI("MaxPool", "[PLUGIN_MaxPool]--------------ParseParamsMaxPool  start---------------");
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (nullptr == node) {
    OP_LOGE("MaxPool", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  int64_t ceil_mode = 0;
  std::vector<int> v_ksizes = {};
  std::vector<int> v_strides = {};
  std::vector<int> v_pads = {};
  std::string v_pad = "SAME";
  std::vector<int> DefaultStride = {1, 1};
  std::vector<int> DefaultPads = {1, 1, 1, 1};

  bool set_ksizes_flag = false;
  bool set_strides_flag = false;
  bool set_pads_flag = false;

  for (const auto& attr : node->attribute()) {
    if (attr.name() == "kernel_shape" && attr.type() == ge::onnx::AttributeProto::INTS) {
      if (attr.ints_size() == 2) {
        for (int i = 0; i < attr.ints_size(); i++) {
          v_ksizes.push_back(attr.ints(i));
        }
      } else if (attr.ints_size() == 1) {
        v_ksizes.push_back(attr.ints(0));
        v_ksizes.push_back(attr.ints(0));
      }
      set_ksizes_flag = true;
    } else if (attr.name() == "strides" && attr.type() == ge::onnx::AttributeProto::INTS) {
      if (attr.ints_size() == 2) {
        for (int i = 0; i < attr.ints_size(); i++) {
          v_strides.push_back(attr.ints(i));
        }
      } else if (attr.ints_size() == 1) {
        v_strides.push_back(attr.ints(0));
        v_strides.push_back(attr.ints(0));
      }
      set_strides_flag = true;
    } else if (attr.name() == "auto_pad" && attr.type() == ge::onnx::AttributeProto::STRING) {
      if (attr.s() == "VALID") {
        v_pad = "VALID";
      } else {
        v_pad = "SAME";
      }
    } else if (attr.name() == "pads" && attr.type() == ge::onnx::AttributeProto::INTS) {
      if (attr.ints_size() == 4) {
        for (int i = 0; i < attr.ints_size(); i++) {
          v_pads.push_back(attr.ints(i));
        }
      } else if (attr.ints_size() == 1) {
        v_pads.push_back(attr.ints(0));
        v_pads.push_back(attr.ints(0));
        v_pads.push_back(attr.ints(0));
        v_pads.push_back(attr.ints(0));
      }
      set_pads_flag = true;
    } else if (attr.name() == "ceil_mode" && attr.type() == ge::onnx::AttributeProto::INT) {
      ceil_mode = attr.i();
    }
  }

  if (ceil_mode == 0) {
    op_dest.SetAttr("ceil_mode", 1);
  } else {
    op_dest.SetAttr("ceil_mode", 0);
  }

  if (set_ksizes_flag) {
    op_dest.SetAttr("window", v_ksizes);
  } else {
    OP_LOGI("MaxPool", "onnx MaxPool op has no ksize attr");
    op_dest.SetAttr("window", DefaultStride);
  }

  if (set_strides_flag) {
    op_dest.SetAttr("stride", v_strides);
  } else {
    OP_LOGI("MaxPool", "onnx MaxPool op has no strides attr, use default.");
    op_dest.SetAttr("strides", DefaultStride);
  }

  if (set_pads_flag) {
    op_dest.SetAttr("pad", v_pads);
  } else {
    OP_LOGI("MaxPool", "onnx MaxPool op has no pads attr, use default.");
    op_dest.SetAttr("pad", DefaultPads);
  }

  OP_LOGI("MaxPool", "--------------ParseParamsMaxPool  end---------------");

  return SUCCESS;
}

REGISTER_CUSTOM_OP("Pooling")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::11::MaxPool")
    .ParseParamsFn(ParseParamsMaxPool)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
