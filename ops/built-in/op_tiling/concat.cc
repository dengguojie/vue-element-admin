/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file concat.cpp
 * \brief dynamic shape tiling of concat_v2
 */
#include <map>

#include <nlohmann/json.hpp>
#include "op_tiling.h"

#include "op_log.h"
#include "../op_proto/util/error_util.h"
#include "error_log.h"

namespace optiling {
using namespace ge;
using namespace std;

static vector<vector<int64_t>> GetInputShapes(const TeOpParas& paras) {
  vector<vector<int64_t>> shapes;
  for (const auto& input : paras.inputs) {
    if (input.tensor.empty()) {
      return {};
    }

    if (input.arg_type == TA_LIST) {
      OP_LOGD("concat", "TA_LIST");
      for (const auto& tensor : input.tensor) {
        shapes.emplace_back(tensor.shape);
      }
    } else if (input.arg_type == TA_SINGLE) {
      OP_LOGD("concat", "TA_SINGLE");
      shapes.emplace_back(input.tensor[0].shape);
    }
  }

  return shapes;
}

static string to_string(const ByteBuffer& tiling_data) {
  auto data = tiling_data.str();
  string result;
  const int64_t *data_addr = reinterpret_cast<const int64_t*>(data.c_str());
  for (size_t i = 0; i < data.length() / sizeof(int64_t); i++) {
    result += std::to_string(*data_addr);
    data_addr++;
    result += " ";
  }

  return result;
}

static string to_string(const vector<vector<int64_t>>& shapes) {
  std::string shapes_string = "[";
  for (const auto& shape : shapes) {
    shapes_string += ge::DebugString<int64_t>(shape);
    shapes_string += ",";
  }

  shapes_string += "]";
  return shapes_string;
}

static bool CheckParams(const string& op, const vector<vector<int64_t>>& input_shapes, int32_t dim) {
  if (input_shapes.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op, "The input count should be more than 0");
    return false;
  }

  auto shape_length = input_shapes[0].size();
  for (const auto& input_shape : input_shapes) {
    if (input_shape.size() != shape_length) {
      VECTOR_INNER_ERR_REPORT_TILIING(op, "The length of each shape must be equal");
      return false;
    }
  }

  int max_dim = static_cast<int32_t>(shape_length);
  if (dim >= max_dim or dim < -max_dim) {
    VECTOR_INNER_ERR_REPORT_TILIING(op, "the parameter[%s] should be [between %d and %d], but actually is [%d].",
            "concat_dim", min(max_dim - 1, -max_dim), max(max_dim - 1, -max_dim), dim);
    return false;
  }

  size_t new_dims = dim >= 0 ? dim : max_dim + dim;
  const auto& shape = input_shapes[0];
  for (size_t i = 1; i < input_shapes.size(); i++) {
    for (size_t j = 0; j < shape_length; j++) {
      if (j != new_dims && input_shapes[i][j] != shape[j]) {
        VECTOR_INNER_ERR_REPORT_TILIING(op, "dims must equal except concat dim[%d], input_values[%s]", dim,
                to_string(input_shapes).c_str());
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

    for (const auto& item : input_tiling_info) {
      ByteBufferPut(tiling_data, item.first);
      ByteBufferPut(tiling_data, item.second);
    }
  }
};

static bool GetTilingParam(const vector<vector<int64_t>>& input_shapes, int32_t concat_dim, TilingParam& tiling_param) {
  if (input_shapes.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING("concat", "The input count should be more than 0");
    return false;
  }

  if (concat_dim < 0) {
    concat_dim += input_shapes[0].size();
  }

  auto input_count = input_shapes.size();
  tiling_param.max_inner_dims = 0;
  tiling_param.axis = 1;
  tiling_param.input_count = input_shapes.size();
  tiling_param.out_dims =
      accumulate(input_shapes[0].begin(), input_shapes[0].begin() + concat_dim, 1, std::multiplies<int64_t>());
  tiling_param.min_inner_dims =
      accumulate(input_shapes[0].begin() + concat_dim, input_shapes[0].end(), (int64_t)1, multiplies<int64_t>());
  int64_t output_index = 0;
  for (size_t i = 0; i < input_count; i++) {
    auto inner_dims =
        accumulate(input_shapes[i].begin() + concat_dim, input_shapes[i].end(), (int64_t)1, multiplies<int64_t>());
    tiling_param.max_inner_dims = max(tiling_param.max_inner_dims, inner_dims);
    tiling_param.min_inner_dims = min(tiling_param.min_inner_dims, inner_dims);

    tiling_param.input_tiling_info.emplace_back(pair<int64_t, int64_t>(inner_dims, output_index));

    output_index += inner_dims;
  }

  tiling_param.output_inner_length = output_index;

  return true;
}

template <typename T>
static bool GetCompileInfo(const nlohmann::json& op_info, const std::string& name, T& value) {
  const nlohmann::json& allVars = op_info["vars"];
  if (allVars.empty()) {
    return false;
  }

  if (allVars.count(name) == 0) {
    return false;
  }

  value = allVars[name].get<T>();
  return true;
}

static void Pack2ConcatParams(const std::string& opType, vector<vector<int64_t>>& input_shapes, int32_t& concat_dim) {
  if (opType != "Pack") {
    return;
  }

  if (concat_dim < -1) {
    concat_dim += 1;
    return;
  }

  if (!input_shapes.empty()) {
    int32_t shape_length = static_cast<int32_t>(input_shapes[0].size());
    if (concat_dim == -1 || concat_dim == shape_length) {
      for (auto& shape : input_shapes) {
        shape.push_back(1);
      }
    }
  }
}

/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] opParas: inputs/outputs/atts of the op
 * @param [in] op_info: compile time generated info of the op
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool ConcatV2Tiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& op_info,
                    OpRunInfo& runInfo) {
  OP_LOGD(opType.c_str(), "ConcatV2Tiling running.");

  vector<vector<int64_t>> input_shapes = GetInputShapes(opParas);
  if (input_shapes.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "Get input shapes failed.");
    return false;
  }

  int32_t concat_dim = 0;
  int32_t input_size = 0;
  int32_t block_dim = 0;

  const map<string, int32_t&> compile_params = {
      {"concat_dim", concat_dim},
      {"input_size", input_size},
      {"block_dim", block_dim},
  };

  for (auto& param : compile_params) {
    const auto& name = param.first;
    OP_LOGD(opType.c_str(), "GetCompileInfo %s.", name.c_str());
    if (!GetCompileInfo<int32_t>(op_info, name, param.second)) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileInfo %s failed.", name.c_str());
      return false;
    }
    OP_LOGD(opType.c_str(), "%s=%d.", name.c_str(), param.second);
  }

  Pack2ConcatParams(opType, input_shapes, concat_dim);
  OP_LOGD(opType.c_str(), "to check params.");
  if (!CheckParams(opType, input_shapes, concat_dim)) {
    return false;
  }

  if (input_size != static_cast<int32_t>(input_shapes.size())) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType,
            "check input size failed. "
            "Input_size in compile is %d, but in params is %zd",
            input_size, input_shapes.size());
    return false;
  }

  OP_LOGD(opType.c_str(), "GetTilingParam.");
  TilingParam tiling_param;
  if (!GetTilingParam(input_shapes, concat_dim, tiling_param)) {
    return false;
  }

  OP_LOGD(opType.c_str(), "encode TilingParam.");
  tiling_param.encode(runInfo.tiling_data);
  OP_LOGD(opType.c_str(), "TilingParam:%s.", to_string(runInfo.tiling_data).c_str());

  // block_dim, not need for concat tiling
  runInfo.block_dim = block_dim;

  // workspace, null for tik op
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;
  OP_LOGD(opType.c_str(), "tiling run success.");

  return true;
}

// register tiling interface of the Concat, ConcatV2 op.
REGISTER_OP_TILING_FUNC_BUFFERED(ConcatV2D, ConcatV2Tiling);
REGISTER_OP_TILING_FUNC_BUFFERED(ConcatD, ConcatV2Tiling);

// register tiling interface of the Pack op.
REGISTER_OP_TILING_FUNC_BUFFERED(Pack, ConcatV2Tiling);
}  // namespace optiling
