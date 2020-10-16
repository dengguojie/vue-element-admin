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
 * @file concat_v2.cpp
 *
 * @brief dynamic shape tiling of concat_v2
 *
 * @version 1.0
 *
 */

#include <nlohmann/json.hpp>
#include <securec.h>
#include "register/op_tiling.h"
#include "op_log.h"

namespace optiling {
using namespace ge;
using namespace std;

static vector<vector<int64_t>> GetInputShapes(const TeOpParas &paras) {
  vector<vector<int64_t>> shapes;
  for (const auto &input : paras.inputs) {
    if (input.tensor.empty()) {
      return {};
    }

    if (input.arg_type == TA_LIST) {
      OP_LOGI("concat", "TA_LIST");
      for (const auto& tensor : input.tensor) {
        shapes.push_back(tensor.shape);
      }
    } else if (input.arg_type == TA_SINGLE) {
      OP_LOGI("concat", "TA_SINGLE");
      shapes.push_back(input.tensor[0].shape);
    }
  }

  return shapes;
}

static bool GetConcatDim(const TeOpParas &paras, int32_t &dim) {
  if (paras.const_inputs.find("concat_dim") == paras.const_inputs.end()) {
    return false;
  }

  if (EOK != memcpy_s(&dim, sizeof(dim), std::get<0>(paras.const_inputs.at(
      "concat_dim")), sizeof(dim))) {
    return false;
  }
  return true;
}

static string to_string(const ByteBuffer &tiling_data) {
  auto data = tiling_data.str();
  string result;
  int64_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
    if (EOK == memcpy_s(&tmp, sizeof(tmp), data.c_str()+ i, sizeof(tmp))) {
      result += std::to_string(tmp);
    } else {
      result += "ERROR";
    }

    result += " ";
  }

  return result;
}

static bool CheckParams(const string &op,
                 const vector<vector<int64_t>> &input_shapes,
                 int32_t dim) {
  if (input_shapes.empty()) {
    OP_LOGE(op.c_str(), "The input count should be more than 0");
    return false;
  }

  auto shape_length = input_shapes[0].size();
  for (const auto &input_shape : input_shapes) {
    if (input_shape.size() != shape_length) {
      OP_LOGE(op.c_str(), "The length of each shape must be equal");
      return false;
    }
  }

  int max_dim = static_cast<int32_t>(shape_length);
  if (dim >= max_dim or dim < 1 - max_dim) {
    OP_LOGE(op.c_str(), "In op[%s], the parameter[%s] "
                        "should be [between %d and %d], "
                        "but actually is [%d].",
            op.c_str(),
            "concat_dim",
            min(max_dim-1, -max_dim),
            max(max_dim-1, -max_dim),
            dim);

    return false;
  }

  size_t new_dims = dim > 0 ? dim : max_dim + dim;
  const auto &shape = input_shapes[0];
  for (size_t i = 1; i < input_shapes.size(); i++) {
    for (size_t j = 0; j < shape_length; j++) {
      if (j != new_dims && input_shapes[i][j] != shape[j]) {
        OP_LOGE(op.c_str(), "dims must equal except concat dim.");

        return false;
      }
    }
  }

  return true;
}

struct TilingParam {
  int64_t axis = 0;
  int64_t out_dims = 1;
  int64_t max_inner_dims = 0;
  int64_t min_inner_dims = 0;
  int64_t output_inner_length = 1;
  int64_t input_count = 0;
  int64_t reserve1 = 0;
  int64_t reserve2 = 0;

  // list of pair with inner_dims and output_idx
  vector<pair<int64_t, int64_t>> input_tiling_info;
  int index = 0;
  void encode(ByteBuffer& tiling_data) {
    ByteBufferPut(tiling_data, axis);
    ByteBufferPut(tiling_data, out_dims);
    ByteBufferPut(tiling_data, max_inner_dims);
    ByteBufferPut(tiling_data, min_inner_dims);
    ByteBufferPut(tiling_data, output_inner_length);
    ByteBufferPut(tiling_data, input_count);
    ByteBufferPut(tiling_data, reserve1);
    ByteBufferPut(tiling_data, reserve2);

    for (const auto &item: input_tiling_info) {
      ByteBufferPut(tiling_data, item.first);
      ByteBufferPut(tiling_data, item.second);
    }
  }
};

static bool GetTilingParam(const vector<vector<int64_t>> &input_shapes,
                                  int32_t concat_dim,
                                  TilingParam& tiling_param) {
  if (input_shapes.empty()) {
    OP_LOGE("concat", "The input count should be more than 0");
    return false;
  }

  if (concat_dim < 0) {
    concat_dim += input_shapes[0].size();
  }

  auto input_count = input_shapes.size();
  tiling_param.max_inner_dims = 0;
  tiling_param.axis = 1;
  tiling_param.input_count = input_shapes.size();
  tiling_param.out_dims = accumulate(input_shapes[0].begin(),
                                     input_shapes[0].begin() + concat_dim,
                                     1,
                                     std::multiplies<int64_t>());
  tiling_param.min_inner_dims = accumulate(input_shapes[0].begin() + concat_dim,
                                           input_shapes[0].end(),
                                           (int64_t)1,
                                           multiplies<int64_t>());
  int64_t output_index = 0;
  for (size_t i=0; i < input_count; i++) {
    auto inner_dims = accumulate(input_shapes[i].begin() + concat_dim,
                                 input_shapes[i].end(),
                                 (int64_t)1,
                                 multiplies<int64_t>());
    tiling_param.max_inner_dims = max(tiling_param.max_inner_dims, inner_dims);
    tiling_param.min_inner_dims = min(tiling_param.min_inner_dims, inner_dims);

    tiling_param.input_tiling_info.emplace_back(
        pair<int64_t, int64_t>(inner_dims, output_index));

    output_index += inner_dims;

  }

  tiling_param.output_inner_length = output_index;

  return true;

}

template<typename OUT_TYPE>
static OUT_TYPE GetCompileInfo(const nlohmann::json &op_info,
                               const std::string &name) {
  const nlohmann::json &allVars = op_info["vars"];
  if (allVars.empty()) {
    return 0;
  }

  if (allVars.count(name) == 0) {
    return 0;
  }

  return allVars[name].get<OUT_TYPE>();
}

/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] opParas: inputs/outputs/atts of the op
 * @param [in] op_info: compile time generated info of the op
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool ConcatV2Tiling(const std::string &opType, const TeOpParas &opParas,
                    const nlohmann::json &op_info, OpRunInfo &runInfo) {
  OP_LOGI(opType.c_str(), "ConcatV2Tiling running.");

  vector<vector<int64_t>> input_shapes = GetInputShapes(opParas);
  if (input_shapes.empty()) {
    OP_LOGE(opType.c_str(), "Get input shapes failed.");
    return false;
  }

  int32_t concat_dim = 0;
  if (!GetConcatDim(opParas, concat_dim)) {
    OP_LOGE(opType.c_str(), "concat_dim not exists.");
    return false;
  }
  OP_LOGD(opType.c_str(), "concat_dim=%d.", concat_dim);

  OP_LOGD(opType.c_str(), "to check params.");
  if (!CheckParams(opType, input_shapes, concat_dim)) {
    return false;
  }

  OP_LOGD(opType.c_str(), "GetCompileInfo input_size.");
  auto input_size = GetCompileInfo<size_t>(op_info, "input_size");
  if (input_size != input_shapes.size()) {
    OP_LOGE(opType.c_str(),
            "check input size failed. "
            "Input_size in compile is %zd, but in params is %zd",
            input_size, input_shapes.size());
    return false;
  }

  OP_LOGD(opType.c_str(), "GetCompileInfo block_dim.");
  auto block_dim = GetCompileInfo<int32_t>(op_info, "block_dim");
  if (block_dim == 0) {
    OP_LOGE(opType.c_str(), "block dim should not be 0.");
    return false;
  }

  OP_LOGD(opType.c_str(), "GetTilingParam.");
  TilingParam tiling_param;
  if (!GetTilingParam(input_shapes, concat_dim, tiling_param)) {
    return false;
  }

  OP_LOGD(opType.c_str(), "encode TilingParam.");
  tiling_param.encode(runInfo.tiling_data);
  OP_LOGD(opType.c_str(), "TilingParam:%s.",
          to_string(runInfo.tiling_data).c_str());

  // block_dim, not need for concat tiling
  runInfo.block_dim = block_dim;

  // workspace, null for tik op
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;
  OP_LOGI(opType.c_str(), "tiling run success.");

  return true;
}

// register tiling interface of the Concat, ConcatV2 op.
REGISTER_OP_TILING_FUNC(ConcatV2D, ConcatV2Tiling);
REGISTER_OP_TILING_FUNC(ConcatD, ConcatV2Tiling);
}



