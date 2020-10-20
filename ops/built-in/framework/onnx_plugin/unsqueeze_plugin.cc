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
 * \file unsqueeze_plugin.cpp
 * \brief
 */
#include <string>
#include <vector>

#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"

#include "op_log.h"

using domi::ONNX;

namespace domi {

Status ParseParamsUnsqueeze(const Message* op_src, ge::Operator& op_dest) {
  OP_LOGI("Unsqueeze", "[PLUGIN_CONCAT]---------ParseParams Unsqueeze start----------");
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("Unsqueeze", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  bool has_axes = false;
  bool has_dims = false;

  std::vector<int64_t> axes;
  std::string attr_name = "axes";
  for (auto it : node->attribute()) {
    if (it.name() != attr_name) {
      continue;
    }

    has_axes = true;
    int dims_size = it.ints_size();
    if (dims_size == 0) {
      break;
    }
    has_dims = true;

    for (int64_t dim : it.ints()) {
      axes.push_back(dim);
      OP_LOGI("Unsqueeze", "Dim: %ld", dim);
    }
  }

  if (!has_axes || !has_dims) {
    OP_LOGE("Unsqueeze", "axes is invalid!");
    return PARAM_INVALID;
  }

  op_dest.SetAttr(attr_name, axes);
  return SUCCESS;
}

// register Add op info to GE
REGISTER_CUSTOM_OP("Unsqueeze")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::11::Unsqueeze")
    .ParseParamsFn(ParseParamsUnsqueeze)
    .ImplyType(ImplyType::GELOCAL);
}  // namespace domi
