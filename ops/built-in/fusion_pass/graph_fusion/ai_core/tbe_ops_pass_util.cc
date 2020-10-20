/**
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file tbe_ops_pass_util.cpp
 *
 * @brief util for ops pass
 *
 * @version 1.0
 *
 */
#include "tbe_ops_pass_util.h"

#include <vector>
#include <string>
#include "graph/utils/op_desc_utils.h"

using namespace std;

bool HasUnKnowDimShape(const ge::NodePtr &node_ptr) {
  if (!node_ptr) {
    return false;
  }

  auto op = ge::OpDescUtils::CreateOperatorFromNode(node_ptr);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (!op_desc) {
    return false;
  }

  for (const auto &ptr : op_desc->GetAllInputsDescPtr()) {
    auto ge_shape = ptr->GetShape();
    for (const auto &dim : ge_shape.GetDims()) {
      if (dim == ge::UNKNOWN_DIM) {
        return true;
      }
    }
  }

  for (const auto &ptr : op_desc->GetAllOutputsDescPtr()) {
    auto ge_shape = ptr->GetShape();
    for (const auto &dim : ge_shape.GetDims()) {
      if (dim == ge::UNKNOWN_DIM) {
        return true;
      }
    }
  }

  return false;
}

void ClearOpInferDepends(const ge::NodePtr &node_ptr) {
  if (!node_ptr) {
    return;
  }

  auto op = ge::OpDescUtils::CreateOperatorFromNode(node_ptr);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (!op_desc) {
    return;
  }

  vector<string> dummyVec;
  op_desc->SetOpInferDepends(dummyVec);
}



