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
 * \file yolo_plugin.cpp
 * \brief
 */
#include <string>
#include <vector>

#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"

using namespace std;
using namespace ge;

namespace domi {
Status ParseParamsYolo(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("Yolo", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int boxes = 3;
  int coords = 4;
  int classes = 80;
  std::string yolo_version = "V3";
  bool softmax = false;
  bool background = false;
  bool softmaxtree = false;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "boxes" && attr.type() == ge::onnx::AttributeProto::INT) {
      boxes = attr.i();
    } else if (attr.name() == "coords" && attr.type() == ge::onnx::AttributeProto::INT) {
      coords = attr.i();
    } else if (attr.name() == "classes" && attr.type() == ge::onnx::AttributeProto::INT) {
      classes = attr.i();
    } else if (attr.name() == "yolo_version" && attr.type() == ge::onnx::AttributeProto::STRING) {
      yolo_version = attr.s();
    }
  }

  op_dest.SetAttr("boxes", boxes);
  op_dest.SetAttr("coords", coords);
  op_dest.SetAttr("classes", classes);
  op_dest.SetAttr("yolo_version", yolo_version);
  op_dest.SetAttr("softmax", softmax);
  op_dest.SetAttr("background", background);
  op_dest.SetAttr("softmaxtree", softmaxtree);

  return SUCCESS;
}

// register Yolo op info to GE
REGISTER_CUSTOM_OP("Yolo")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::Yolo",
                   "ai.onnx::9::Yolo",
                   "ai.onnx::10::Yolo",
                   "ai.onnx::11::Yolo",
                   "ai.onnx::12::Yolo",
                   "ai.onnx::13::Yolo"})
    .ParseParamsFn(ParseParamsYolo)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
