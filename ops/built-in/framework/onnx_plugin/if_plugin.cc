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
#include "onnx_common.h"
#include "array_ops.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;
Status ParseParamsIf(const Message *op_src, ge::Operator &op_dest) {
  const ge::onnx::NodeProto *node = dynamic_cast<const ge::onnx::NodeProto *>(op_src);
  std::shared_ptr<ge::OpDesc> op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  int in_size = node->input_size() - 1;
  if (in_size < 0) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "If input num is less than 0.");
    return FAILED;
  }
  int out_size = node->output_size();
  (void)op_desc->AddDynamicInputDesc("input", in_size);
  (void)op_desc->AddDynamicOutputDesc("output", out_size);
  return SUCCESS;
}
Status ParseSubgraphPostFnIf(const std::string& subgraph_name, const ge::Graph& graph) {
  AutoMappingSubgraphIOIndexFunc auto_mapping_subgraph_index_func =
    FrameworkRegistry::Instance().GetAutoMappingSubgraphIOIndexFunc(ONNX);
  if (auto_mapping_subgraph_index_func == nullptr) {
    ONNX_PLUGIN_LOGE("If", "auto mapping subgraph func is nullptr!");
    return FAILED;
  }
  return auto_mapping_subgraph_index_func(graph,
      [&](int data_index, int &parent_index) -> Status {
        parent_index = data_index + 1;
        return SUCCESS;
      },
      [&](int output_index, int &parent_index) -> Status {
        parent_index = output_index;
        return SUCCESS;
      });
}

// register if op info to GE
REGISTER_CUSTOM_OP("If")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::8::If",
                 "ai.onnx::9::If",
                 "ai.onnx::10::If",
                 "ai.onnx::11::If",
                 "ai.onnx::12::If",
                 "ai.onnx::13::If"})
  .ParseParamsFn(ParseParamsIf)
  .ParseSubgraphPostFn(ParseSubgraphPostFnIf)
  .ImplyType(ImplyType::GELOCAL);
}  // namespace domi
