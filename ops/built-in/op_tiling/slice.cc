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
 * \file slice.cpp
 * \brief dynamic shape tiling of slice
 */
#include <string>
#include <map>
#include <cmath>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"

#include "../op_proto/util/error_util.h"
#include "../op_proto/util/op_common_util.h"
#include "op_log.h"

namespace optiling {
using namespace ge;

struct SliceParameters {
  std::vector<int64_t> input;
  std::vector<int64_t> output_shape;
  std::vector<int64_t> begin_list;
  std::vector<int64_t> end_list;
  std::vector<int64_t> stride_list;

  std::string to_string() const {
    string result = "input_shape:" + ops::to_string(input);
    result += " output_shape:" + ops::to_string(output_shape);
    result += " begin:" + ops::to_string(begin_list);
    result += " end:" + ops::to_string(end_list);
    result += " stride:" + ops::to_string(stride_list);
    return result;
  }
};

static void MakePerformanceParams(SliceParameters &parameters) {
  SliceParameters perf_params;
  bool last_same = false;
  size_t perf_size = 0;
  for (size_t i = 0; i < parameters.input.size(); i++) {
    const auto input_shape_i = parameters.input[i];
    const auto output_shape_i = parameters.output_shape[i];
    const auto begin_i = parameters.begin_list[i];
    const auto end_i = parameters.end_list[i];
    const auto stride_i = parameters.stride_list[i];
    if (input_shape_i != output_shape_i) {
      perf_params.input.push_back(input_shape_i);
      perf_params.output_shape.push_back(output_shape_i);
      perf_params.begin_list.push_back(begin_i);
      perf_params.end_list.push_back(end_i);
      perf_params.stride_list.push_back(stride_i);
      last_same = false;
      perf_size++;
      continue;
    }

    if (!last_same) {
      last_same = true;
      perf_params.input.push_back(input_shape_i);
      perf_params.output_shape.push_back(output_shape_i);
      perf_params.begin_list.push_back(begin_i);
      perf_params.end_list.push_back(end_i);
      perf_params.stride_list.push_back(stride_i);
      perf_size++;
      continue;
    }

    const auto perf_index = perf_size - 1;
    perf_params.input[perf_index] *= input_shape_i;
    perf_params.output_shape[perf_index] *= output_shape_i;
    perf_params.begin_list[perf_index] = 0;
    perf_params.end_list[perf_index] = perf_params.input[perf_index];
    perf_params.stride_list[perf_index] = 1;
  }

  if (perf_params.input.size() > 1 &&
      perf_params.input.back() == perf_params.output_shape.back() &&
      perf_params.stride_list[perf_params.input.size() - 2] == 1) {
    const auto last_second_index = perf_params.input.size() - 2;
    perf_params.input[last_second_index] *= perf_params.input.back();
    perf_params.output_shape[last_second_index] *= perf_params.output_shape.back();
    perf_params.begin_list[last_second_index] *= perf_params.input.back();
    perf_params.end_list[last_second_index] *= perf_params.input.back();
    perf_params.stride_list[last_second_index] = 1;

    perf_params.input.pop_back();
    perf_params.output_shape.pop_back();
    perf_params.begin_list.pop_back();
    perf_params.end_list.pop_back();
    perf_params.stride_list.pop_back();
  }

  if (parameters.input.size() > perf_params.input.size()) {
    const auto input_len = parameters.input.size();
    parameters.input.assign(input_len, 1);
    parameters.output_shape.assign(input_len, 1);
    parameters.begin_list.assign(input_len, 0);
    parameters.end_list.assign(input_len, 1);
    parameters.stride_list.assign(input_len, 1);

    const auto delta = input_len - perf_params.input.size();
    std::copy(perf_params.input.begin(), perf_params.input.end(), parameters.input.begin() + delta);
    std::copy(perf_params.output_shape.begin(), perf_params.output_shape.end(), parameters.output_shape.begin() + delta);
    std::copy(perf_params.begin_list.begin(), perf_params.begin_list.end(), parameters.begin_list.begin() + delta);
    std::copy(perf_params.end_list.begin(), perf_params.end_list.end(), parameters.end_list.begin() + delta);
    std::copy(perf_params.stride_list.begin(), perf_params.stride_list.end(), parameters.stride_list.begin() + delta);
  }
}

static void SetRuningParams(const SliceParameters& params, OpRunInfo& runInfo) {
  const vector<int64_t>* tiling_params[] = {
      &params.input, &params.output_shape, &params.begin_list, &params.end_list, &params.stride_list,
  };

  for (auto item : tiling_params) {
    for (auto x : *item) {
      ByteBufferPut(runInfo.tiling_data, x);
    }
  }
}

static bool GetConstValue(const TeOpParas& paras, const string& name, const string& dtype, vector<int64_t>& values) {
  values.clear();
  if (paras.const_inputs.count(name) == 0 || std::get<0>(paras.const_inputs.at(name)) == nullptr) {
    return false;
  }

  auto size = std::get<1>(paras.const_inputs.at(name));
  if (dtype == "int64") {
    int count = size / sizeof(int64_t);
    const int64_t *data_addr = reinterpret_cast<const int64_t*>(std::get<0>(paras.const_inputs.at(name)));
    for (int i=0; i<count; i++) {
      values.push_back(*data_addr);
      data_addr++;
    }
  } else if (dtype == "int32") {
    int count = size / sizeof(int32_t);
    const int32_t *data_addr = reinterpret_cast<const int32_t*>(std::get<0>(paras.const_inputs.at(name)));
    for (int i=0; i<count; i++) {
      values.push_back(*data_addr);
      data_addr++;
    }
  }

  return true;
}

template <typename OUT_TYPE>
static OUT_TYPE GetCompileInfo(const nlohmann::json& op_info, const std::string& name) {
  const nlohmann::json& allVars = op_info["vars"];
  if (allVars.empty()) {
    return 0;
  }

  if (allVars.count(name) == 0) {
    return 0;
  }

  return allVars[name].get<OUT_TYPE>();
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

bool SliceTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                 OpRunInfo& runInfo) {
  using namespace nlohmann;
  OP_LOGD(opType.c_str(), "SliceTiling running.");

  if (opCompileInfo == nullptr) {
    OP_LOGE(opType.c_str(), "SliceTiling: opCompileInfo json error.");
    ge::OpsGetCompileParamsErrReport(opType, "compile json");
    return false;
  }
  if (opParas.inputs.empty() || opParas.inputs[0].tensor.empty()) {
    OP_LOGE(opType.c_str(), "SliceTiling: input shape error.");
    ge::OpsMissInputErrReport(opType, "x");
    return false;
  }

  SliceParameters slice_params_output;
  slice_params_output.input = opParas.inputs[0].tensor[0].shape;
  map<string, std::pair<int, vector<int64_t>&>> const_params = {
      {"offsets", {1, slice_params_output.begin_list}},
      {"size", {2, slice_params_output.end_list}},
  };

  for (auto& item : const_params) {
    auto& name = item.first;
    int index = item.second.first;
    auto& values = item.second.second;
    if (!GetConstValue(opParas, name, opParas.inputs[index].tensor[0].dtype, values)) {
      OP_LOGE(opType.c_str(), "Get %s values failed", name.c_str());
      ge::OpsGetAttrErrReport(opType, name);
      return false;
    }
  }

  if (slice_params_output.begin_list.size() != slice_params_output.end_list.size() ||
      slice_params_output.begin_list.size() != slice_params_output.input.size()) {
    OP_LOGE(opType.c_str(), "length of input_shape, offsets and size must be equal.");
    ge::OpsInputShapeErrReport(
        opType, "length of x's shape, offsets and size must be equal.", "[x][offsets][size]",
        ConcatString("[", slice_params_output.input.size(), "]", "[", slice_params_output.begin_list.size(), "]", "[",
                     slice_params_output.end_list.size(), "]"));
    return false;
  }

  for (size_t index = 0; index < slice_params_output.end_list.size(); index++) {
    if (slice_params_output.end_list[index] == -1) {
      slice_params_output.end_list[index] = slice_params_output.input[index] - slice_params_output.begin_list[index];
    }
  }

  for (size_t i=0; i<slice_params_output.begin_list.size(); i++) {
    if (slice_params_output.begin_list[i] < 0 ||
        slice_params_output.begin_list[i] + slice_params_output.end_list[i] < slice_params_output.begin_list[i] ||
        slice_params_output.begin_list[i] + slice_params_output.end_list[i] > slice_params_output.input[i]) {
      OP_LOGE(opType.c_str(),
              "Requirements: 0<=offsets[i]<= offsets[i]+size[i]<=input_shape[i], offsets:%s, size:%s, input_shape:%s",
              DebugString(slice_params_output.begin_list).c_str(), DebugString(slice_params_output.end_list).c_str(),
              DebugString(slice_params_output.input).c_str());
      ge::OpsAttrValueErrReport(opType, "offsets", "[0<=offsets[i]<= offsets[i]+size[i]<=input_shape[i]]",
                                ConcatString(DebugString(slice_params_output.begin_list).c_str(),
                                             DebugString(slice_params_output.end_list).c_str(),
                                             DebugString(slice_params_output.input).c_str()));
      return false;
    }
  }

  // in slice end_list is size values
  slice_params_output.output_shape = slice_params_output.end_list;

  // calculate end list
  for (size_t i = 0; i < slice_params_output.begin_list.size(); i++) {
    slice_params_output.end_list[i] += slice_params_output.begin_list[i];
  }

  slice_params_output.stride_list.assign(slice_params_output.end_list.size(), 1);

  OP_LOGD(opType.c_str(), "origin slice params: %s", slice_params_output.to_string().c_str());
  MakePerformanceParams(slice_params_output);
  OP_LOGD(opType.c_str(), "perf slice params: %s", slice_params_output.to_string().c_str());
  SetRuningParams(slice_params_output, runInfo);
  OP_LOGD(opType.c_str(), "tiling param:%s", to_string(runInfo.tiling_data).c_str());
  int32_t core_num = GetCompileInfo<int32_t>(opCompileInfo, "block_dim");
  runInfo.block_dim = core_num;
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;
  OP_LOGD(opType.c_str(), "tiling run success.");

  return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(Slice, SliceTiling);
}  // namespace optiling
