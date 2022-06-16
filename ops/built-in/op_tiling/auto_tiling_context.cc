/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file auto_tiling_context.cc
 * \brief
 */

#include "auto_tiling_context.h"
#include "vector_tiling_log.h"
#include "error_log.h"
#include "op_log.h"
#include "op_const.h"

namespace optiling {
size_t OpShape::GetDimNum() const {
  if (rt_shape != nullptr) {
    return rt_shape->GetDimNum();
  }
  return ge_shape->GetDimNum();
}

int64_t OpShape::GetDim(size_t idx) const {
  if (rt_shape != nullptr) {
    OP_LOGD("AutoTiling", "AutoTiling shape value: shape[%lld]=%lld", idx, rt_shape->GetDim(idx));
    return rt_shape->GetDim(idx);
  }
  OP_LOGD("AutoTiling", "AutoTiling shape value: shape[%lld]=%lld", idx, ge_shape->GetDim(idx));
  return ge_shape->GetDim(idx);
}

int64_t OpShape::GetShapeSize() const {
  if (rt_shape != nullptr) {
    return rt_shape->GetShapeSize();
  }
  return ge_shape->GetShapeSize();
}

bool OpShape::Empty() const {
  return rt_shape != nullptr && ge_shape != nullptr;
}

bool AutoTilingOp::GetInputDataType(size_t idx, ge::DataType& dtype) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(*op_paras);
  if (op_desc == nullptr) {
    return false;
  }
  auto input_desc = op_desc->MutableInputDesc(idx);
  if (input_desc == nullptr) {
    return false;
  }
  dtype = input_desc->GetDataType();
  return true;
}

bool AutoTilingOp::GetInputDataType(const OpInfoImpl* op_info, ge::DataType& dtype) {
  if (op_info && op_info->GetInType()) {
    dtype = *op_info->GetInType();
    return true;
  } else {
    return GetInputDataType(static_cast<size_t>(0), dtype);
  }
}

bool AutoTilingOp::GetOutputDataType(size_t idx, ge::DataType& dtype) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(*op_paras);
  if (op_desc == nullptr) {
    return false;
  }
  auto output_desc = op_desc->MutableOutputDesc(idx);
  if (output_desc == nullptr) {
    return false;
  }
  dtype = output_desc->GetDataType();
  return true;
}

size_t AutoTilingOp::GetInputNums() {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(*op_paras);
  if (op_desc == nullptr) {
    return 0;
  }
  return op_desc->GetAllInputsSize();
}

size_t AutoTilingOp::GetInputNums(const OpInfoImpl* op_info) {
  if (op_info && op_info->GetInputShape()) {
    OP_LOGD(op_type, "Get custom input shape num success");
    return op_info->GetInputShape()->size();
  } else {
    return GetInputNums();
  }
}

size_t AutoTilingOp::GetOutputNums() {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(*op_paras);
  if (op_desc == nullptr) {
    return 0;
  }
  return op_desc->GetOutputsSize();
}

OpShape AutoTilingOp::GetInputShape(size_t idx) {
  OP_LOGD(op_type, "Get input shape %lld:", idx);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(*op_paras);
  if (op_desc == nullptr) {
    return OpShape();
  }
  auto input_desc = op_desc->MutableInputDesc(idx);
  if (input_desc == nullptr) {
    return OpShape();
  }
  const ge::GeShape& shape = input_desc->GetShape();
  OpShape opShape(&shape);
  return opShape;
}

OpShape AutoTilingOp::GetOriginInputShape(size_t idx) {
  OP_LOGD(op_type, "Get origin input shape %lld:", idx);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(*op_paras);
  if (op_desc == nullptr) {
    return OpShape();
  }
  auto input_desc = op_desc->MutableInputDesc(idx);
  if (input_desc == nullptr) {
    return OpShape();
  }
  const ge::GeShape& shape = input_desc->GetOriginShape();
  OpShape opShape(&shape);
  return opShape;
}
OpShape AutoTilingOp::GetOutputShape(size_t idx) {
  OP_LOGD(op_type, "Get output shape %lld:", idx);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(*op_paras);
  if (op_desc == nullptr) {
    return OpShape();
  }
  auto output_desc = op_desc->MutableOutputDesc(idx);
  if (output_desc == nullptr) {
    return OpShape();
  }
  const ge::GeShape& shape = output_desc->GetShape();
  OpShape opShape(&shape);
  return opShape;
}

const char* AutoTilingOp::GetOpType() {
  return op_type;
}

const AutoTilingCompileInfo* AutoTilingOp::GetCompileInfo() {
  return compile_info;
}

bool AutoTilingOp::SetBlockDim(uint32_t block_dims) {
  run_info->SetBlockDim(block_dims);
  return true;
}

bool AutoTilingOp::SetTilingKey(uint64_t tiling_key) {
  run_info->SetTilingKey(tiling_key);
  return true;
}

bool AutoTilingOp::SetNeedAtomic(bool flag) {
  run_info->SetClearAtomic(flag);
  return true;
}

void AutoTilingOp::SetCompileInfo(const AutoTilingCompileInfo* _compile_info) {
  compile_info = _compile_info;
}

bool AutoTilingOp::GetAttr(const char* name, size_t index, int64_t& value) {
  if (op_paras->GetAttr(name, value) == ge::GRAPH_FAILED) {
    return false;
  }
  return true;
}

bool AutoTilingOp::GetAttr(const char* name, size_t index, std::vector<int64_t>& values) {
  if (op_paras->GetAttr(name, values) == ge::GRAPH_FAILED) {
    return false;
  }
  return true;
}

bool AutoTilingOp::GetConstInput(const char* name, size_t index, int64_t& value) {
  if (name != nullptr) {
    index = ge::OpDescUtils::GetOpDescFromOperator(*op_paras)->GetInputIndexByName("axes");
  }
  std::vector<int64_t> values{0};
  if (!ops::GetConstIntData(*op_paras, index, values)) {
    return false;
  }
  if (values.empty()) {
    return false;
  }
  value = values[0];
  return true;
}

bool AutoTilingOp::GetConstInput(const char* name, size_t index, std::vector<int64_t>& values) {
  constexpr size_t max_integer = 2147483647;
  if (name != nullptr && index == max_integer) {
    index = ge::OpDescUtils::GetOpDescFromOperator(*op_paras)->GetInputIndexByName("axes");
  }
  if (!ops::GetConstIntData(*op_paras, index, values)) {
    return false;
  }
  return true;
}

const ge::Operator* AutoTilingOp::GetOpParas() {
  return op_paras;
}

utils::OpRunInfo* AutoTilingOp::GetRunInfo() {
  return run_info;
}

bool AutoTilingContext::GetInputDataType(size_t idx, ge::DataType& dtype) {
  auto input_desc = context->GetInputDesc(idx);
  if (input_desc == nullptr) {
    return false;
  }
  dtype = input_desc->GetDataType();
  return true;
}

bool AutoTilingContext::GetInputDataType(const OpInfoImpl* op_info, ge::DataType& dtype) {
  if (op_info && op_info->GetInType()) {
    dtype = *op_info->GetInType();
    return true;
  } else {
    return GetInputDataType(static_cast<size_t>(0), dtype);
  }
}

bool AutoTilingContext::GetOutputDataType(size_t idx, ge::DataType& dtype) {
  auto output_desc = context->GetInputDesc(idx);
  if (output_desc == nullptr) {
    return false;
  }
  dtype = output_desc->GetDataType();
  return true;
}

size_t AutoTilingContext::GetInputNums() {
  return context->GetComputeNodeInputNum();
}

size_t AutoTilingContext::GetInputNums(const OpInfoImpl* op_info) {
  if (op_info && op_info->GetInputShape()) {
    OP_LOGD(context->GetNodeType(), "Get custom input shape num success");
    return op_info->GetInputShape()->size();
  } else {
    return GetInputNums();
  }
}

size_t AutoTilingContext::GetOutputNums() {
  auto compute_node_info = context->GetComputeNodeInfo();
  if (compute_node_info == nullptr) {
    return 0;
  }
  return compute_node_info->GetOutputsNum();
}

OpShape AutoTilingContext::GetInputShape(size_t idx) {
  OP_LOGD(context->GetNodeType(), "Get input shape %lld:", idx);
  auto input_shape = context->GetInputShape(idx);
  if (input_shape == nullptr) {
    return OpShape();
  }
  const gert::Shape& shape = input_shape->GetStorageShape();
  OpShape opShape(&shape);
  return opShape;
}
OpShape AutoTilingContext::GetOriginInputShape(size_t idx) {
  OP_LOGD(context->GetNodeType(), "Get origin input shape %lld:", idx);
  auto input_shape = context->GetInputShape(idx);
  if (input_shape == nullptr) {
    return OpShape();
  }
  const gert::Shape& shape = input_shape->GetOriginShape();
  OpShape opShape(&shape);
  return opShape;
}

OpShape AutoTilingContext::GetOutputShape(size_t idx) {
  OP_LOGD(context->GetNodeType(), "Get output shape %lld:", idx);
  const gert::Shape& shape = context->GetOutputShape(idx)->GetStorageShape();
  OpShape opShape(&shape);
  return opShape;
}

const char* AutoTilingContext::GetOpType() {
  return context->GetNodeType();
}

const AutoTilingCompileInfo* AutoTilingContext::GetCompileInfo() {
  if (compile_info != nullptr) {
    return compile_info;
  }
  return reinterpret_cast<const AutoTilingCompileInfo*>(context->GetCompileInfo());
}

bool AutoTilingContext::SetBlockDim(uint32_t block_dims) {
  if (context->SetBlockDim(block_dims) == ge::GRAPH_FAILED) {
    return false;
  }
  return true;
}

bool AutoTilingContext::SetTilingKey(uint64_t tiling_key) {
  if (context->SetTilingKey(tiling_key) == ge::GRAPH_FAILED) {
    return false;
  }
  return true;
}

bool AutoTilingContext::SetNeedAtomic(bool flag) {
  if (context->SetNeedAtomic(flag) == ge::GRAPH_FAILED) {
    return false;
  }
  return true;
}

void AutoTilingContext::SetCompileInfo(const AutoTilingCompileInfo* _compile_info) {
  compile_info = _compile_info;
}

bool AutoTilingContext::GetAttr(const char* name, size_t index, int64_t& value) {
  auto* attrs = context->GetAttrs();
  if (attrs == nullptr) {
    return false;
  }
  const auto* attr_value = attrs->GetAttrPointer<int64_t>(index);
  if (attr_value == nullptr) {
    return false;
  }
  value = *attr_value;
  return true;
}

bool AutoTilingContext::GetAttr(const char* name, size_t index, std::vector<int64_t>& values) {
  auto* attrs = context->GetAttrs();
  if (attrs == nullptr) {
    return false;
  }
  auto* attr_value = attrs->GetAttrPointer<gert::ContinuousVector>(index);
  if (attr_value == nullptr) {
    return false;
  }
  auto data = reinterpret_cast<const int64_t*>(attr_value->GetData());
  for (size_t i = 0; i < attr_value->GetSize(); i++) {
    values.push_back(data[i]);
  }
  return true;
}

bool AutoTilingContext::GetConstInput(const char* name, size_t index, int64_t& value) {
  std::vector<int64_t> const_values;
  if (GetConstInput(name, index, const_values)) {
    if (!const_values.empty()) {
      value = const_values[0];
      return true;
    }
  }
  return false;
}

template <typename T>
bool GetData(const gert::Tensor* tensor, std::vector<int64_t>& values) {
  size_t shape_size = tensor->GetShapeSize();
  values.resize(shape_size);
  auto* tensor_data = tensor->GetData<T>();
  if (tensor_data == nullptr) {
    return false;
  }
  for (size_t i = 0; i < shape_size; i++) {
    values[i] = static_cast<int64_t>(*(tensor_data + i));
  }
  return true;
}

bool AutoTilingContext::GetConstInput(const char* name, size_t index, std::vector<int64_t>& values) {
  auto* tensor = context->GetInputTensor(index);
  if (tensor == nullptr) {
    return false;
  }
  if (tensor->GetDataType() == ge::DT_INT32) {
    return GetData<int32_t>(tensor, values);
  } else if (tensor->GetDataType() == ge::DT_INT64) {
    return GetData<int64_t>(tensor, values);
  } else if (tensor->GetDataType() == ge::DT_UINT32) {
    return GetData<uint32_t>(tensor, values);
  } else if (tensor->GetDataType() == ge::DT_UINT64) {
    return GetData<uint64_t>(tensor, values);
  }
  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeType(), "Get const input error, unsupport data type");
  return false;
}

bool AutoTilingOp::WriteVarAttrs(const uint64_t tiling_key) const {
  return compile_info->var_attr_wrap.WriteVarAttrs(tiling_key, op_type, *op_paras, *run_info);
}

bool AutoTilingContext::WriteVarAttrs(const uint64_t tiling_key) {
  const AutoTilingCompileInfo *autoTilingCompileInfo =
                            dynamic_cast<const AutoTilingCompileInfo *>(GetCompileInfo());
  return autoTilingCompileInfo->var_attr_wrap.WriteVarAttrs(tiling_key, *context);
}
const ge::Operator* AutoTilingContext::GetOpParas() {
  return nullptr;
}

utils::OpRunInfo* AutoTilingContext::GetRunInfo() {
  return nullptr;
}
} // namespace optiling
