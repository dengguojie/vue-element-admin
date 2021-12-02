/* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this
 * file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http:// www.apache.org/licenses/LICENSE-2.0
 */
#include "onnx_common.h"
#include "array_ops.h"
#include "random_ops.h"
using namespace std;
using namespace ge;
using ge::Operator;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
using OpDesc = std::shared_ptr<ge::OpDesc>;
static const int DATA_TYPE_INT32 = 6;
Status ParseParamsMultinomialCall(const Message* op_src, ge::Operator& op_dest) {
 const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  std::shared_ptr<ge::OpDesc> op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  if (op_desc == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "op_desc is null");
    return FAILED;
  }
  
  op_desc->AddDynamicInputDesc("x", 1);
  op_desc->AddDynamicOutputDesc("y", 1);
  int data_type = DATA_TYPE_INT32;
  float seed = 0;
  int sample_size = 1;
  for (auto attr : node->attribute()) {
    if (attr.name() == "dtype") {
      data_type = attr.i();
    } else if (attr.name() == "sample_size") {
      sample_size = attr.i();
    } else if (attr.name() == "seed") {
      seed = attr.f();
    }
  }
  ge::AttrUtils::SetInt(op_desc, "dtype", data_type);
  ge::AttrUtils::SetInt(op_desc, "sample_size", sample_size);
  ge::AttrUtils::SetFloat(op_desc, "seed", seed);
  op_dest.SetAttr("original_type", "ai.onnx::11::Multinomial");
  return SUCCESS;
}

Status ParseOpToGraphMultinomial(const ge::Operator &op, Graph &graph) {
  std::shared_ptr<ge::OpDesc> op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc == nullptr) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "op_desc is null");
    return FAILED;
  }
  int sample_size = 1;
  int data_type = DATA_TYPE_INT32;
  float seed = 0;
  ge::AttrUtils::GetInt(op_desc, "sample_size", sample_size);
  ge::AttrUtils::GetInt(op_desc, "dtype", data_type);
  ge::AttrUtils::GetFloat(op_desc, "seed", seed);
  ge::DataType dtype_om = GetOmDtypeFromOnnxDtype(data_type);
  if (dtype_om == ge::DT_UNDEFINED) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "dtype[%d] is wrong,please select right dtype", data_type);
    return FAILED;
  }
  int int_seed = static_cast<int>(seed);
  std::vector<int64_t> dims = {};
  ge::Tensor tensor = Scalar2Tensor(sample_size, dims, ge::DT_INT32);
  auto data_op = op::Data("input1").set_attr_index(0);
  auto const_op = op::Const("data1").set_attr_value(tensor);
  auto muti_op = op::Multinomial("Muti")
                 .set_input_logits(data_op)
                 .set_input_num_samples(const_op)
                 .set_attr_dtype(dtype_om)
                 .set_attr_seed(int_seed);
  std::vector<ge::Operator> inputs{data_op};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
  output_indexs.emplace_back(muti_op, std::vector<size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::8::Multinomial",
                 "ai.onnx::9::Multinomial",
                 "ai.onnx::10::Multinomial",
                 "ai.onnx::11::Multinomial",
                 "ai.onnx::12::Multinomial",
                 "ai.onnx::13::Multinomial"})
  .ParseParamsFn(ParseParamsMultinomialCall)
  .ParseOpToGraphFn(ParseOpToGraphMultinomial)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
