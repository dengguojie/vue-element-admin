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

using namespace std;
using namespace ge;
using ge::Operator;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
using OpDesc = std::shared_ptr<ge::OpDesc>;

Status ParseParamsEyeLike(const Message *op_src, ge::Operator &op_dest) {
  const NodeProto *node = reinterpret_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int dtype = 1;
  int k = 0;
  for (auto attr : node->attribute()) {
    if (attr.name() == "dtype") {
      dtype = attr.i();
    } else if(attr.name() == "k") {
      k = attr.i();
    }
  }

  op_dest.SetAttr("dtype", dtype);
  op_dest.SetAttr("k", k);
  op_dest.SetAttr("original_type", "ai.onnx::11::EyeLike");
  OpDesc op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  op_desc->AddDynamicInputDesc("x", 1);
  op_desc->AddDynamicOutputDesc("y", 1);
  return SUCCESS;
}

Status ParseOpToGraphEyeLike(const ge::Operator& op, Graph& graph) {
  //TODO(ONNX PLUGIN) This is a Interim plan, The review did not take complexity into account.
  //Follow will add new optype eyelike to adapte 
  int dtype = 1;
  op.GetAttr("dtype", dtype);
  std::map<int, ge::DataType> dmap = {{1, ge::DT_FLOAT}, {10, ge::DT_FLOAT16}, {11, ge::DT_DOUBLE},
                                      {3, ge::DT_INT8}, {5, ge::DT_INT16}, {6, ge::DT_INT32},
                                      {7, ge::DT_INT64}, {2, ge::DT_UINT8}, {4, ge::DT_UINT16},
                                      {12, ge::DT_UINT32}, {13, ge::DT_UINT64}, {9, ge::DT_BOOL}};
  
  int32_t k = 0;
  op.GetAttr("k", k);
  ge::Shape shape({1});
  TensorDesc tensor_desc(shape, ge::FORMAT_ND, ge::DT_INT32);
  ge::Tensor k_tensor(tensor_desc, reinterpret_cast<uint8_t*>(&k), sizeof(int32_t));

  std::vector<int64_t> dims;
  ge::Shape shape1(dims);
  int32_t pad = 0;
  TensorDesc tensor_desc1(shape1, ge::FORMAT_ND, dmap[dtype]);
  ge::Tensor pad_tensor(tensor_desc1, reinterpret_cast<uint8_t*>(&pad), sizeof(int32_t));

  int32_t fill = 1;
  TensorDesc tensor_desc2(shape1, ge::FORMAT_ND, ge::DT_INT32);
  ge::Tensor fill_tensor(tensor_desc2, reinterpret_cast<uint8_t*>(&fill), sizeof(int32_t));

  int32_t split_dim = 0;
  TensorDesc tensor_desc3(shape1, ge::FORMAT_ND, ge::DT_INT32);
  ge::Tensor split_dim_tensor(tensor_desc3, reinterpret_cast<uint8_t*>(&split_dim), sizeof(int32_t));
  
  auto data0 = op::Data("data").set_attr_index(0);
  auto const_op = op::Const("const").set_attr_value(k_tensor);
  
  auto shape_op = op::Shape("shape").set_input_x(data0).set_attr_dtype(ge::DT_INT32);
  auto const_op_split = op::Const("const3").set_attr_value(split_dim_tensor);
  auto split_op = op::Split("split").create_dynamic_output_y(2)
                                    .set_input_x(shape_op)
                                    .set_input_split_dim(const_op_split)
                                    .set_attr_num_split(2);
 
  auto add_op = op::Adds("add").set_input_x(split_op, 0).set_attr_value(k);
  auto const_op2 = op::Const("const2").set_attr_value(fill_tensor);
  auto fill_op = op::Fill("fill").set_input_dims(add_op).set_input_value(const_op2);
  auto cast_op = op::Cast("cast").set_input_x(fill_op).set_attr_dst_type(dmap[dtype]);

  auto squeeze_op = op::Squeeze("squeeze").set_input_x(split_op, 0).set_attr_axis(0);
  auto squeeze_op1 = op::Squeeze("squeeze1").set_input_x(split_op, 1).set_attr_axis(0);
  
  auto const_op1 = op::Const("const1").set_attr_value(pad_tensor);
  auto maxtrix_op = op::MatrixDiagV2("matrix_diagv2").set_input_diagonal(cast_op)
                                                     .set_input_k(const_op)
                                                     .set_input_num_rows(squeeze_op)
                                                     .set_input_num_cols(squeeze_op1)
                                                     .set_input_padding_value(const_op1);

  auto input_desc1 = maxtrix_op.GetInputDesc(2);
  input_desc1.SetShape(ge::Shape(dims));
  maxtrix_op.UpdateInputDesc("num_rows", input_desc1);
  std::vector<ge::Operator> inputs = {data0};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
  output_indexs.emplace_back(maxtrix_op, std::vector<size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}
// register ROIAlign op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::8::EyeLike",
                 "ai.onnx::9::EyeLike",
                 "ai.onnx::10::EyeLike",
                 "ai.onnx::11::EyeLike",
                 "ai.onnx::12::EyeLike",
                 "ai.onnx::13::EyeLike"})
  .ParseParamsFn(ParseParamsEyeLike)
  .ParseOpToGraphFn(ParseOpToGraphEyeLike)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
