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
 * \file strided_slice.cpp
 * \brief dynamic shape tiling of strided_slice
 */
#include <string>
#include <map>
#include <cmath>

#include <nlohmann/json.hpp>
#include "register/op_tiling.h"
#include "graph/debug/ge_log.h"

#include "op_log.h"
#include "../op_proto/util/error_util.h"
#include "../op_proto/strided_slice_infer_shape.h"

namespace optiling {
using namespace ge;

/*
 * @brief: tiling function of StridedSlice
 * @param [in] opType: opType of the StridedSlice
 * @param [in] opParas: inputs/outputs/atts of the StridedSlice
 * @param [in] opCompileInfo: compile time generated info of the StridedSlice
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
const int32_t BYTE_BLOCK = 32;

struct SliceParameters {
  std::vector<int64_t> input;
  std::vector<int64_t> output_shape;
  std::vector<int64_t> begin_list;
  std::vector<int64_t> end_list;
  std::vector<int64_t> stride_list;
};

struct SliceMasks {
  uint32_t begin_mask = 0;
  uint32_t end_mask = 0;
  uint32_t ellipsis_mask = 0;
  uint32_t new_axis_mask = 0;
  uint32_t shrink_axis_mask = 0;
};

static string to_string(const ByteBuffer& tiling_data) {
  auto data = tiling_data.str();
  string result;
  const int64_t *data_addr = reinterpret_cast<const int64_t*>(data.c_str());
  for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
    result += std::to_string(*data_addr);
    data_addr++;
    result += " ";
  }

  return result;
}

static bool GetStridedSliceSocParams(const std::string& opType, const nlohmann::json& opCompileInfo, int32_t& core_num,
                                     uint32_t& begin_mask, uint32_t& end_mask, uint32_t& ellipsis_mask,
                                     uint32_t& new_axis_mask, uint32_t& shrink_axis_mask) {
  using namespace nlohmann;
  const auto& allVars = opCompileInfo["vars"];
  if (allVars.count("block_dim") == 0) {
    OP_LOGE(opType.c_str(), "GetCompileParams, get block_dim error");
    ge::OpsGetCompileParamsErrReport(opType, "block_dim");
    return false;
  }
  core_num = allVars["block_dim"].get<std::int32_t>();

  if (allVars.count("begin_mask") == 0) {
    OP_LOGE(opType.c_str(), "GetCompileParams, get begin_mask error");
    ge::OpsGetCompileParamsErrReport(opType, "begin_mask");
    return false;
  }
  begin_mask = allVars["begin_mask"].get<std::uint32_t>();

  if (allVars.count("end_mask") == 0) {
    OP_LOGE(opType.c_str(), "GetCompileParams, get end_mask error");
    ge::OpsGetCompileParamsErrReport(opType, "end_mask");
    return false;
  }
  end_mask = allVars["end_mask"].get<std::uint32_t>();

  if (allVars.count("ellipsis_mask") == 0) {
    OP_LOGE(opType.c_str(), "GetCompileParams, get ellipsis_mask error");
    ge::OpsGetCompileParamsErrReport(opType, "ellipsis_mask");
    return false;
  }
  ellipsis_mask = allVars["ellipsis_mask"].get<std::uint32_t>();

  if (allVars.count("new_axis_mask") == 0) {
    OP_LOGE(opType.c_str(), "GetCompileParams, get new_axis_mask error");
    ge::OpsGetCompileParamsErrReport(opType, "new_axis_mask");
    return false;
  }
  new_axis_mask = allVars["new_axis_mask"].get<std::uint32_t>();

  if (allVars.count("shrink_axis_mask") == 0) {
    OP_LOGE(opType.c_str(), "GetCompileParams, get shrink_axis_mask error");
    ge::OpsGetCompileParamsErrReport(opType, "shrink_axis_mask");
    return false;
  }
  shrink_axis_mask = allVars["shrink_axis_mask"].get<std::int32_t>();

  return true;
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

static void MakeSameDims(SliceParameters &parameters, const vector<int64_t> &shape) {
  if (parameters.input.size() < shape.size()) {
    return;
  }

  bool same_size = parameters.input.size() == parameters.begin_list.size() &&
      parameters.input.size() == parameters.end_list.size() &&
      parameters.input.size() == parameters.stride_list.size();

  if (!same_size) {
    return;
  }

  std::vector<int64_t> tmp_input;
  std::vector<int64_t> tmp_begin_list;
  std::vector<int64_t> tmp_end_list;
  std::vector<int64_t> tmp_stride_list;
  std::vector<int64_t> tmp_output_shape;
  bool wrong_dim = false;
  for (size_t i=0,j=0; i<parameters.input.size(); i++) {
    if (j < shape.size() && parameters.input[i] == shape[j]) {
      tmp_input.push_back(parameters.input[i]);
      tmp_begin_list.push_back(parameters.begin_list[i]);
      tmp_end_list.push_back(parameters.end_list[i]);
      tmp_stride_list.push_back(parameters.stride_list[i]);
      auto interval = parameters.end_list[i] - parameters.begin_list[i];
      auto stride_i = parameters.stride_list[i];
      int64_t output_size = interval / stride_i + (interval % stride_i != 0 ? 1 : 0);
      tmp_output_shape.push_back(output_size);
      j++;
      continue;
    }

    if (parameters.input[i] != 1) {
      wrong_dim = true;
    }
  }

  if (!wrong_dim) {
    parameters.input = tmp_input;
    parameters.begin_list = tmp_begin_list;
    parameters.end_list = tmp_end_list;
    parameters.stride_list = tmp_stride_list;
    parameters.output_shape = tmp_output_shape;
  }
}

bool StridedSliceTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                        OpRunInfo& runInfo) {
  using namespace nlohmann;
  OP_LOGD(opType.c_str(), "StridedSliceTiling running.");
  if (opCompileInfo == nullptr) {
    OP_LOGE(opType.c_str(), "StridedSliceDTiling: opCompileInfo json error.");
    ge::OpsGetCompileParamsErrReport(opType, "compile json");
    return false;
  }
  if (opParas.inputs.empty() || opParas.inputs[0].tensor.empty()) {
    OP_LOGE(opType.c_str(), "StridedSliceTiling: input shape error.");
    ge::OpsMissInputErrReport(opType, "x");
    return false;
  }

  struct SliceParameters slice_params = {};
  struct SliceMasks slice_masks = {};

  const std::vector<int64_t>& input_shape = opParas.inputs[0].tensor[0].shape;
  map<string, std::pair<int, vector<int64_t>&>> const_params = {
      {"begin", {1, slice_params.begin_list}},
      {"end", {2, slice_params.end_list}},
      {"strides", {3, slice_params.stride_list}},
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

  for (size_t i = 0; i < slice_params.stride_list.size(); i++) {
    if (slice_params.stride_list[i] == 0) {
      OP_LOGE(opType.c_str(), "StridedSliceTiling: the dim of stride must be non-zero");
      ge::OpsAttrValueErrReport(opType, "strides", "non-zero", "0");
      return false;
    }

    if (slice_params.stride_list[i] != 1) {
      OP_LOGE(opType.c_str(), "StridedSliceTiling: stride[%zu] must be 1, but it is %lld", i,
              slice_params.stride_list[i]);
      ge::OpsAttrValueErrReport(opType, "strides", "1", ConcatString(slice_params.stride_list[i]));
      return false;
    }
  }

  int32_t core_num = 0;
  bool flag = GetStridedSliceSocParams(opType, opCompileInfo, core_num, slice_masks.begin_mask,
                                       slice_masks.end_mask, slice_masks.ellipsis_mask,
                                       slice_masks.new_axis_mask, slice_masks.shrink_axis_mask);
  if (!flag) {
    OP_LOGE(opType.c_str(), "StridedSliceTiling: get soc params error");

    return false;
  }

  StridedSliceParams input_params = {
      input_shape,
      slice_params.begin_list,
      slice_params.end_list,
      slice_params.stride_list,
      vector<pair<int64_t, int64_t>>(),
      slice_masks.begin_mask,
      slice_masks.end_mask,
      slice_masks.ellipsis_mask,
      slice_masks.new_axis_mask,
      slice_masks.shrink_axis_mask
  };

  vector<int64_t> output_shape;
  vector<pair<int64_t, int64_t>> output_ranges;

  if (!StridedSliceCommonInferShape(opType, input_params, output_shape, output_ranges)) {
    OP_LOGE(opType.c_str(), "StridedSliceCommonInferShape failed.");
    return false;
  }

  slice_params.input = input_params.input_shape;
  slice_params.begin_list = input_params.begin;
  slice_params.end_list = input_params.end;
  slice_params.stride_list = input_params.strides;
  slice_params.output_shape = output_shape;

  MakeSameDims(slice_params, input_shape);
  SetRuningParams(slice_params, runInfo);
  OP_LOGD(opType.c_str(), "tiling param:%s", to_string(runInfo.tiling_data).c_str());
  runInfo.block_dim = core_num;
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;
  OP_LOGD(opType.c_str(), "tiling run success.");

  return true;
}

REGISTER_OP_TILING_FUNC(StridedSlice, StridedSliceTiling);
}  // namespace optiling
