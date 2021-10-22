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

#ifndef OPS_BUILT_IN_FRAMEWORK_ONNX_PLUGIN_ONNX_COMMON_H_
#define OPS_BUILT_IN_FRAMEWORK_ONNX_PLUGIN_ONNX_COMMON_H_

#include <string>
#include <vector>
#include <map>
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/operator.h"
#include "graph.h"
#include "all_ops.h"
#include "op_log.h"

namespace domi {
#define ONNX_PLUGIN_LOGE(op_name, err_msg, ...) \
  do { \
      REPORT_INNER_ERROR("E79999", "onnx_plugin op_name[%s], " err_msg, op_name, ##__VA_ARGS__); \
  } while(0)

template<typename T>
ge::Tensor Vec2Tensor(vector<T>& vals, const vector<int64_t>& dims, ge::DataType dtype, ge::Format format = ge::FORMAT_ND) {
  ge::Shape shape(dims);
  ge::TensorDesc desc(shape, format, dtype);
  ge::Tensor tensor(desc, reinterpret_cast<uint8_t*>(vals.data()), vals.size() * sizeof(T));
  return tensor;
}

template<typename T>
ge::Tensor Scalar2Tensor(T val, const vector<int64_t>& dims, ge::DataType dtype, ge::Format format = ge::FORMAT_ND) {
  ge::Shape shape(dims);
  ge::TensorDesc desc(shape, format, dtype);
  ge::Tensor tensor(desc, reinterpret_cast<uint8_t*>(&val), sizeof(T));
  return tensor;
}

ge::DataType GetOmDtypeFromOnnxDtype(int onnx_type);
Status ChangeFormatFromOnnx(std::shared_ptr<ge::OpDesc>& op_dsc, const int idx, ge::Format format, bool is_input);
}  // namespace domi
#endif  //  OPS_BUILT_IN_FRAMEWORK_ONNX_PLUGIN_ONNX_COMMON_H_