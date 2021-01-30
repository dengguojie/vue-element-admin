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
 * \file concat_plugin.cpp
 * \brief
 */
#include <string>
#include <vector>

#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"

#include "op_log.h"

namespace domi {

static const int DEFAULT_CONCAT_DIM = 1;

Status ParseParamsConcat(const Message* op_src, ge::Operator& op_dest) {
  OP_LOGI("Concat", "[PLUGIN_CONCAT]---------ParseParams Concat start----------");
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (nullptr == node) {
    OP_LOGE("Concat", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int concat_dim = 0;
  bool set_axis_flag = false;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      concat_dim = attr.i();
      set_axis_flag = true;
      break;
    }
  }
  if (!set_axis_flag) {
    OP_LOGI("Concat", "onnx Concat op has no axis attr.");
    concat_dim = DEFAULT_CONCAT_DIM;
  }
  op_dest.SetAttr("concat_dim", concat_dim);

  int n = node->input_size();
  OP_LOGI("Concat", "[PLUGIN_CONCAT]----------input_size=%d----------", n);
  op_dest.SetAttr("N", n);
  std::shared_ptr<ge::OpDesc> op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  op_desc->AddDynamicInputDesc("x", n);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("ConcatD")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::11::Concat")
    .ParseParamsFn(ParseParamsConcat)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
