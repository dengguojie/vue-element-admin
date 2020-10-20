/* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use
 * this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

/*!
 *\file max_pool_v3_grad.cpp
 *\brief infer of MaxPoolV3Grad
 */
#include "max_pool_v3_grad.h"
#include "op_log.h"

namespace ge {
bool check_two_input_dtyep_same(const Operator& op, const string& input_name1, const string& input_name2) {
  auto input_type_orig_input = op.GetInputDesc(input_name1).GetDataType();
  auto input_type_orig_output = op.GetInputDesc(input_name2).GetDataType();
  if (input_type_orig_input != input_type_orig_output) {
    return false;
  }
  return true;
}

std::vector<int64_t> GetAttrValue(const ge::Operator& op, const std::string& key_name) {
  std::vector<int64_t> list;
  return list;
}

bool CheckListEmpty(const std::string& opName, const std::vector<int64_t>& list, const std::string& attrName) {
  if (list.empty()) {
    return false;
  }
  return true;
}

IMPLEMT_VERIFIER(MaxPoolV3Grad, MaxPoolV3GradVerify) {
  if (!check_two_input_dtyep_same(op, "orig_input", "orig_output") ||
      !check_two_input_dtyep_same(op, "orig_input", "grad")) {
    OP_LOGE(op.GetName().c_str(), "The shape of orig_input orig_output and grad must be same!");
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
    if (data_format != "NCHW" && data_format != "NHWC") {
      OP_LOGE(op.GetName().c_str(), "The data_format should be NCHW or NHWC!");
      return GRAPH_FAILED;
    }
  }
  std::vector<int64_t> ksize;
  ksize = GetAttrValue(op, "ksize");
  if (!CheckListEmpty(op.GetName(), ksize, "ksize")) {
    OP_LOGE(op.GetName().c_str(), "The ksize is empty!");
    return GRAPH_FAILED;
  }
  if (ksize.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "The size of ksize should be 4!");
    return GRAPH_FAILED;
  }
  if (data_format == "NCHW" && (ksize[0] != 1 || ksize[1] != 1)) {
    OP_LOGE(op.GetName().c_str(), "The first and second dim of ksize must be 1!");
    return GRAPH_FAILED;
  }
  if (data_format == "NHWC" && (ksize[0] != 1 || ksize[3] != 1)) {
    OP_LOGE(op.GetName().c_str(), "The first and fourth dim of ksize must be 1!");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), strides, "strides")) {
    OP_LOGE(op.GetName().c_str(), "The strides is empty!");
    return GRAPH_FAILED;
  }
  if (strides.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "The size of strides should be 4!");
    return GRAPH_FAILED;
  }
  if (data_format == "NCHW" && (strides[0] != 1 || strides[1] != 1)) {
    OP_LOGE(op.GetName().c_str(), "The first and second dim of strides must be 1!");
    return GRAPH_FAILED;
  }
  if (data_format == "NHWC" && (strides[0] != 1 || strides[3] != 1)) {
    OP_LOGE(op.GetName().c_str(), "The first and fourth dim of ksize must be 1!");
    return GRAPH_FAILED;
  }
  std::string padding_mod;
  if (ge::GRAPH_SUCCESS != op.GetAttr("padding_mod", padding_mod)) {
    OP_LOGE(op.GetName().c_str(), "The padding_mod is empty!");
    return GRAPH_FAILED;
  }
  if (padding_mod != "SAME" && padding_mod != "VALID" && padding_mod != "CALCULATED") {
    OP_LOGE(op.GetName().c_str(), "The value of padding_mode must be in 'SAME' 'VALID' or 'CALCULATED'!");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> pads;
  ksize = GetAttrValue(op, "pads");
  if (!CheckListEmpty(op.GetName(), pads, "pads")) {
    OP_LOGE(op.GetName().c_str(), "The pads is empty!");
    return GRAPH_FAILED;
  }
  if (pads.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "The size of pads should be 4!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(MaxPoolV3Grad, MaxPoolV3GradInferShape) {
  auto shapeX1 = op.GetInputDesc("orig_input").GetShape();
  auto inputType = op.GetInputDesc("orig_input").GetDataType();

  TensorDesc td = op.GetOutputDesc("out_grad");
  td.SetShape(shapeX1);
  td.SetDataType(inputType);
  (void)op.UpdateOutputDesc("out_grad", td);
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(MaxPoolV3Grad, MaxPoolV3GradInferShape);
VERIFY_FUNC_REG(MaxPoolV3Grad, MaxPoolV3GradVerify);
}  // namespace ge
