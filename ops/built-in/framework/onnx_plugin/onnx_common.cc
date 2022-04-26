/* Copyright (C) 2020. Huawei Technologies Co., Ltd. All
rights reserved.
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
  enum DataTypeOnnx {
  DTO_FLOAT = 1,     // float type
  DTO_UINT8 = 2,     // uint8 type
  DTO_INT8 = 3,      // int8 type
  DTO_UINT16 = 4,    // uint16 type
  DTO_INT16 = 5,     // int16 type
  DTO_INT32 = 6,     // int32 type
  DTO_INT64 = 7,     // int64 type
  DTO_STRING = 8,    // string type
  DTO_BOOL = 9,      // bool type
  DTO_FLOAT16 = 10,  // float16 type
  DTO_DOUBLE = 11,   // double type
  DTO_UINT32 = 12,   // uint32 type
  DTO_UINT64 = 13,   // uint64 type
  DTO_UNDEFINED
};

std::map<int, ge::DataType> onnx2om_dtype_map = {{DTO_UINT8, ge::DT_UINT8}, {DTO_UINT16, ge::DT_UINT16},
                                                 {DTO_UINT32, ge::DT_UINT32}, {DTO_UINT64, ge::DT_UINT64},
                                                 {DTO_INT8, ge::DT_INT8}, {DTO_INT16, ge::DT_INT16},
                                                 {DTO_INT32, ge::DT_INT32}, {DTO_INT64, ge::DT_INT64},
                                                 {DTO_FLOAT16, ge::DT_FLOAT16}, {DTO_FLOAT, ge::DT_FLOAT},
                                                 {DTO_DOUBLE, ge::DT_DOUBLE}, {DTO_STRING, ge::DT_STRING},
                                                 {DTO_BOOL, ge::DT_BOOL}};


ge::DataType GetOmDtypeFromOnnxDtype(int onnx_type) {
  auto dto_type = static_cast<DataTypeOnnx>(onnx_type);
  if (onnx2om_dtype_map.find(dto_type) == onnx2om_dtype_map.end()) {
    return ge::DT_UNDEFINED;
  }
  return onnx2om_dtype_map[dto_type];
}

Status ChangeFormatFromOnnx(ge::Operator& op, const int idx, ge::Format format, bool is_input) {
  auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (op_dsc == nullptr) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "Get op_desc from operator failed.");
    return FAILED;
  }

  if (is_input) {
    ge::GeTensorDesc org_tensor = op_dsc->GetInputDesc(idx);
    org_tensor.SetOriginFormat(format);
    org_tensor.SetFormat(format);
    auto ret = op_dsc->UpdateInputDesc(idx, org_tensor);
    if (ret != ge::GRAPH_SUCCESS) {
      ONNX_PLUGIN_LOGE(op_dsc->GetName().c_str(), "change input format failed.");
      return FAILED;
    }
  } else {
    ge::GeTensorDesc org_tensor_y = op_dsc->GetOutputDesc(idx);
    org_tensor_y.SetOriginFormat(format);
    org_tensor_y.SetFormat(format);
    auto ret_y = op_dsc->UpdateOutputDesc(idx, org_tensor_y);
    if (ret_y != ge::GRAPH_SUCCESS) {
      ONNX_PLUGIN_LOGE(op_dsc->GetName().c_str(), "change output format failed.");
      return FAILED;
    }
  }
  return SUCCESS;
}
}  //  namespace domi
