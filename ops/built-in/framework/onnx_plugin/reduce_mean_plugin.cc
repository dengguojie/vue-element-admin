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
 * \file reduce_mean_plugin.cpp
 * \brief
 */
#include <string>
#include <vector>

#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"

#include "op_log.h"

namespace domi {

Status ParseParamsReduceMean(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("ReduceMean", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  std::vector<int> v_axis;
  bool set_axes_flag = false;
  bool keep_dims = true;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "axes" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (int i = 0; i < attr.ints_size(); i++) {
        v_axis.push_back(attr.ints(i));
      }
      set_axes_flag = true;
    } else if (attr.name() == "keepdims" && attr.type() == ge::onnx::AttributeProto::INT) {
      if (attr.i() != 1) {
        keep_dims = false;
      }
    }
  }
  if (set_axes_flag) {
    op_dest.SetAttr("axes", v_axis);
  } else {
    OP_LOGI("ReduceMean", "onnx ReduceMean op has no axes attr, use default.");
  }
  op_dest.SetAttr("keep_dims", keep_dims);

  return SUCCESS;
}

// register ReduceMean op info to GE
REGISTER_CUSTOM_OP("ReduceMeanD")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::11::ReduceMean")
    .ParseParamsFn(ParseParamsReduceMean)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
