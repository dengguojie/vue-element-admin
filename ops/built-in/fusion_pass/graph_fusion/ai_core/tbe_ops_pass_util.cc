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

#include <algorithm>
#include <vector>
#include <string>
#include "graph/utils/op_desc_utils.h"
#include "pattern_fusion_util.h"
#include "op_log.h"

using namespace std;

bool HasUnKnowDimShape(const ge::NodePtr &node_ptr) {
  FUSION_PASS_CHECK(node_ptr == nullptr, FUSION_PASS_LOGE("node is null."), return false);

  auto op = ge::OpDescUtils::CreateOperatorFromNode(node_ptr);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  FUSION_PASS_CHECK(op_desc == nullptr, FUSION_PASS_LOGE("op desc is null."), return false);

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

bool HasUnKnowShape(const ge::NodePtr &node_ptr) {
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
            if (dim == ge::UNKNOWN_DIM || dim == ge::UNKNOWN_DIM_NUM) {
                return true;
            }
        }
    }

    for (const auto &ptr : op_desc->GetAllOutputsDescPtr()) {
        auto ge_shape = ptr->GetShape();
        for (const auto &dim : ge_shape.GetDims()) {
            if (dim == ge::UNKNOWN_DIM || dim == ge::UNKNOWN_DIM_NUM) {
                return true;
            }
        }
    }

    return false;
}

void ClearOpInferDepends(const ge::NodePtr &node_ptr) {
  FUSION_PASS_CHECK(node_ptr == nullptr, FUSION_PASS_LOGE("node is null."), return);

  auto op = ge::OpDescUtils::CreateOperatorFromNode(node_ptr);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  FUSION_PASS_CHECK(op_desc == nullptr, FUSION_PASS_LOGE("op desc is null."), return);

  vector<string> dummyVec;
  op_desc->SetOpInferDepends(dummyVec);
}

bool IsUnknownShape(const std::vector<int64_t>& shape) {
  if (shape == ge::UNKNOWN_RANK) {
    return true;
  }

  auto found = std::find(shape.begin(), shape.end(), ge::UNKNOWN_DIM);
  return found != shape.end();
}

void RemoveInputDesc(ge::OpDescPtr op_desc, uint32_t index) {
  auto& input_name_index_map = op_desc->MutableAllInputName();
  auto found = input_name_index_map.end();
  for (auto iter = input_name_index_map.begin(); iter != input_name_index_map.end(); ++iter) {
    if (iter->second == index) {
      found = iter;
    } else if (iter->second > index) {
      iter->second--;
    }
  }

  if (found != input_name_index_map.end()) {
    input_name_index_map.erase(found);
  }

  ge::OpDescUtils::ClearInputDesc(op_desc, index);
}

bool GetIntConstValueFromTensor(const Operator& op, const Tensor& const_tensor, const DataType& dtype,
                                std::vector<int64_t>& const_data) {
  size_t size = 0;
  if (dtype == ge::DT_INT32) {
    int32_t* const_data_ptr = (int32_t*)const_tensor.GetData();
    if (const_data_ptr == nullptr) {
      OP_LOGW(op.GetName().c_str(), "const_data_ptr is null");
      return false;
    }
    size = const_tensor.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back((int32_t)((*(const_data_ptr + i))));
      OP_LOGD(op.GetName().c_str(), "const type is int32 idx:value = %d:%d", i, (int32_t)(*(const_data_ptr + i)));
    }
  } else if (dtype == ge::DT_INT64) {
    int64_t* const_data_ptr = (int64_t*)const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back(((int64_t)(*(const_data_ptr + i))));
      OP_LOGD(op.GetName().c_str(), "const type is int64 idx:value = %d:%d", i, (int32_t)(*(const_data_ptr + i)));
    }
  } else {
    OP_LOGW(op.GetName().c_str(), "do not support get int value for this type %d", dtype);
    return false;
  }
  return true;
}

bool GetIntConstValue(const ge::NodePtr& fused_node, const string& const_name,
                      std::vector<int64_t>&  const_value) {
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fused_node);
  OP_LOGD(op.GetName().c_str(), "begin to get const value for input name(%s)", const_name.c_str());
  Tensor const_tensor;
  if (ge::GRAPH_SUCCESS != op.GetInputConstData(const_name, const_tensor)) {
    OP_LOGW(op.GetName().c_str(), "get const tensor of name(%s) from op failed", const_name.c_str());
    return false;
  }
  DataType dtype = op.GetInputDesc(const_name).GetDataType();

  if (!GetIntConstValueFromTensor(op, const_tensor, dtype, const_value)) {
    OP_LOGW(op.GetName().c_str(), "get const Value of name(%s) from tensor failed", const_name.c_str());
    return false;
  }
  OP_LOGD(op.GetName().c_str(), "end to get const value for input name(%s)", const_name.c_str());
  return true;
}
