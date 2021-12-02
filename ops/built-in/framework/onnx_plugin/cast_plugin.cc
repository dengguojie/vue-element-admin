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

enum DataTypeOm {
  DT_FLOAT = 0,    // float type
  DT_FLOAT16 = 1,  // float16 type
  DT_INT8 = 2,     // int8 type
  DT_INT32 = 3,    // int32 type
  DT_UINT8 = 4,    // uint8 type
  DT_INT16 = 6,    // int16 type
  DT_UINT16 = 7,   // uint16 type
  DT_UINT32 = 8,   // uint32 type
  DT_INT64 = 9,    // int64 type
  DT_UINT64 = 10,  // uint64 type
  DT_DOUBLE = 11,  // double type
  DT_BOOL = 12,    // bool type
  DT_STRING = 13,  // string type
  DT_UNDEFINED
};

std::map<DataTypeOnnx, DataTypeOm> g_data_type_dic = {
    {DTO_FLOAT, DT_FLOAT},     {DTO_UINT8, DT_UINT8},   {DTO_INT8, DT_INT8},
    {DTO_UINT16, DT_UINT16},   {DTO_INT16, DT_INT16},   {DTO_INT32, DT_INT32},
    {DTO_INT64, DT_INT64},     {DTO_STRING, DT_STRING}, {DTO_BOOL, DT_BOOL},
    {DTO_FLOAT16, DT_FLOAT16}, {DTO_DOUBLE, DT_DOUBLE}, {DTO_UINT32, DT_UINT32},
    {DTO_UINT64, DT_UINT64}};

int32_t TransDataTypeFromOnnxToOm(int onnx) {
  auto dto_onnx = static_cast<DataTypeOnnx>(onnx);
  int32_t om = -1;
  auto it = g_data_type_dic.find(dto_onnx);
  if (it == g_data_type_dic.end()) {
    return om;
  }
  om = static_cast<int32_t>(it->second);
  return om;
}

Status ParseParamsCast(const Message *op_src, ge::Operator &op_dst) {
  const NodeProto *node = reinterpret_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dst.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  bool bfind_to = false;
  for (auto attr : node->attribute()) {
    if (attr.name() == "to") {
      bfind_to = true;
      int onnx = attr.i();
      int32_t om = TransDataTypeFromOnnxToOm(onnx);
      if (om == -1) {
        ONNX_PLUGIN_LOGE(op_dst.GetName().c_str(), "TransDataTypeFromOnnxToOm failed, onnx = %d", onnx);
        return FAILED;
      }
      op_dst.SetAttr("dst_type", om);
      break;
    }
  }

  if (!bfind_to) {
    ONNX_PLUGIN_LOGE(op_dst.GetName().c_str(), "Message op_src do not have attribute to.");
    return FAILED;
  }
  return SUCCESS;
}

REGISTER_CUSTOM_OP("Cast")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::8::Cast",
                 "ai.onnx::9::Cast",
                 "ai.onnx::10::Cast",
                 "ai.onnx::11::Cast",
                 "ai.onnx::12::Cast",
                 "ai.onnx::13::Cast"})
  .ParseParamsFn(ParseParamsCast)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
