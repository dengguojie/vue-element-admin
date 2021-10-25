/*
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include "error_log.h"
#include "reduce_tiling.h"
#include "op_tiling_util.h"
#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"
#include <unordered_map>

namespace optiling {
static std::string GetShape(std::vector<int64_t> shape) {
  std::string res;
  for (auto it = shape.begin(); it < shape.end(); it++) {
    res += std::to_string(*it);
    res += ", ";
  }
  return res;
}

bool BiasAddGradTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                       utils::OpRunInfo& run_info) {
  PROFILING_TILING_INIT(op_type.c_str());
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetOpDescFromOperator return nullptr!"), return false);

  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input0 desc failed"), return false);
  std::vector<int64_t> shape = input_desc->MutableShape().GetDims();
  std::vector<int64_t> ori_shape = input_desc->GetOriginShape().GetDims();
  ge::Format format = input_desc->GetFormat();
  ge::Format ori_format = input_desc->GetOriginFormat();
  PROFILING_TILING_AFTER_GET_SHAPE_REG();
  std::unordered_map<char, int> zip_shape;
  std::vector<int64_t> new_shape = shape;

  OP_LOGI("BiasAddGrad",
          "input format [%s], ori_format [%s], shape: [%s], ori_shape: [%s], "
          "and ori_format lens not compare ori_shape lens.",
          to_string(format).c_str(), to_string(ori_format).c_str(), GetShape(shape).c_str(), GetShape(ori_shape).c_str());
  if (format == ge::FORMAT_FRACTAL_Z or format == ge::FORMAT_FRACTAL_Z_3D) {
    uint64_t target_shape = 4;
    if (shape.size() == target_shape) {
      std::string str_ori_format = to_string(ori_format);
      if (str_ori_format.size() != ori_shape.size()) {
        OP_LOGD("BiasAddGrad",
                "input format [%s], ori_format [%s], shape: [%s], ori_shape: [%s], "
                "and ori_format lens not compare ori_shape lens.",
                to_string(format).c_str(), str_ori_format.c_str(), GetShape(shape).c_str(),
                GetShape(ori_shape).c_str());
        return false;
      } else {
        for (uint64_t i = 0; i < str_ori_format.size(); ++i) {
          zip_shape[str_ori_format[i]] = ori_shape[i];
        }
        int64_t shape_h_dim = zip_shape['H'];
        int64_t shape_w_dim = zip_shape['W'];
        int64_t shape_c1_dim = shape[0] / (shape_h_dim * shape_w_dim);
        new_shape = std::vector<int64_t>{shape_c1_dim, shape_h_dim, shape_w_dim};
        new_shape.insert(new_shape.end(), shape.begin() + 1, shape.end());
        if (format == ge::FORMAT_FRACTAL_Z_3D) {
          int64_t shape_d_dim = zip_shape['D'];
          shape_c1_dim = new_shape[0] / shape_d_dim;
          new_shape.insert(new_shape.begin(), {shape_d_dim});
          new_shape[1] = shape_c1_dim;
        }
      }
    }
  }
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();
  std::vector<std::vector<int64_t>> shapes = {new_shape};
  ge::DataType type = input_desc->GetDataType();
  OpInfo eletwise_info(shapes, type);

  bool ret = ReduceTiling(op_type, op_paras, op_info, run_info, eletwise_info);
  PROFILING_TILING_AFTER_CALCU_TILING_REG();
  PROFILING_TILING_END();
  return ret;
}

REGISTER_OP_TILING_FUNC_BUFFERED_V2(BiasAddGrad, BiasAddGradTiling);
}  // namespace optiling
