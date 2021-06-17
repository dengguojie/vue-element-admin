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

#include "strided_slice.h"

#include <string>
#include <map>
#include <cmath>

#include <nlohmann/json.hpp>
#include "graph/debug/ge_log.h"

#include "op_log.h"
#include "../op_proto/util/error_util.h"
#include "../op_proto/util/op_common_util.h"
#include "../op_proto/strided_slice_infer_shape.h"

namespace optiling {
using namespace ge;

const int32_t BYTE_BLOCK = 32;
const size_t MAX_SUPPORTED_DIMS = 8;

std::string SliceParameters::to_string() const {
  string result = "input_shape:" + ops::to_string(input);
  result += " output_shape:" + ops::to_string(output_shape);
  result += " begin:" + ops::to_string(begin_list);
  result += " end:" + ops::to_string(end_list);
  result += " stride:" + ops::to_string(stride_list);
  result += " tiling_mode:" + std::to_string(tiling_mode);
  return result;
}

struct SliceMasks {
  uint32_t begin_mask = 0;
  uint32_t end_mask = 0;
  uint32_t ellipsis_mask = 0;
  uint32_t new_axis_mask = 0;
  uint32_t shrink_axis_mask = 0;
};

static string to_string(const ByteBuffer& tiling_data) {
  auto data = tiling_data.str();
  string result = "(";
  const int64_t* data_addr = reinterpret_cast<const int64_t*>(data.c_str());
  for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
    result += std::to_string(*data_addr);
    data_addr++;
    result += ", ";
  }
  result += ")";

  return result;
}

static bool GetStridedSliceSocParams(const std::string& opType, const nlohmann::json& opCompileInfo, int32_t& core_num,
                                     uint32_t& begin_mask, uint32_t& end_mask, uint32_t& ellipsis_mask,
                                     uint32_t& new_axis_mask, uint32_t& shrink_axis_mask, int32_t& ub_size) {
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

  if (allVars.count("ub_size") == 0) {
    OP_LOGE(opType.c_str(), "GetCompileParams, get ub_size error");
    ge::OpsGetCompileParamsErrReport(opType, "ub_size");
    return false;
  }
  ub_size = allVars["ub_size"].get<std::int32_t>();
  return true;
}

static void MakeSameDims(SliceParameters& parameters, const vector<int64_t>& shape) {
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
  for (size_t i = 0, j = 0; i < parameters.input.size(); i++) {
    if (j < shape.size() && parameters.input[i] == shape[j]) {
      tmp_input.push_back(parameters.input[i]);
      tmp_begin_list.push_back(parameters.begin_list[i]);
      tmp_end_list.push_back(parameters.end_list[i]);
      tmp_stride_list.push_back(parameters.stride_list[i]);
      auto interval = parameters.end_list[i] - parameters.begin_list[i];
      auto stride_i = parameters.stride_list[i];
      if (stride_i == 0) {
        stride_i = 1;
      }
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

static void MakePerformanceParams(SliceParameters& parameters) {
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

  if (perf_params.input.size() > 1 && perf_params.input.back() == perf_params.output_shape.back() &&
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

  parameters = perf_params;
}

/*
 * @brief: tiling function of StridedSlice
 * @param [in] opType: opType of the StridedSlice
 * @param [in] opParas: inputs/outputs/atts of the StridedSlice
 * @param [in] opCompileInfo: compile time generated info of the StridedSlice
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool StridedSliceTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                        OpRunInfo& runInfo) {
  using namespace nlohmann;
  OP_LOGD(opType.c_str(), "StridedSliceTiling running.");

  if (opParas.inputs.empty() || opParas.inputs[0].tensor.empty()) {
    OP_LOGE(opType.c_str(), "StridedSliceTiling: input shape error.");
    ge::OpsMissInputErrReport(opType, "x");
    return false;
  }

  struct SliceParameters slice_params = {};
  struct SliceMasks slice_masks = {};

  const std::vector<int64_t>& input_shape = opParas.inputs[0].tensor[0].shape;
  if (input_shape.size() > MAX_SUPPORTED_DIMS) {
    OP_LOGE(opType, "StridedSliceTiling: input shape error, max supported dims is %zu, actual is %zu.",
            MAX_SUPPORTED_DIMS, input_shape.size());
    ge::OpsInputShapeSizeErrReport(opType, "x", std::to_string(MAX_SUPPORTED_DIMS), std::to_string(input_shape.size()));
    return false;
  }

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
  int32_t ub_size = 0;
  bool flag = GetStridedSliceSocParams(opType, opCompileInfo, core_num, slice_masks.begin_mask, slice_masks.end_mask,
                                       slice_masks.ellipsis_mask, slice_masks.new_axis_mask,
                                       slice_masks.shrink_axis_mask, ub_size);
  OP_LOGD(opType.c_str(), "param ub_size: %d", ub_size);
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
      slice_masks.shrink_axis_mask,
      true,
      true,
      true,
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
  OP_LOGD(opType.c_str(), "origin slice params: %s", slice_params.to_string().c_str());

  SetSliceTilingData(opType, slice_params, runInfo, opParas, core_num, ub_size);
  runInfo.block_dim = core_num;
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;
  OP_LOGD(opType.c_str(), "tiling run success.");

  return true;
}

void SetSliceTilingData(const string& opType, SliceParameters& slice_params, OpRunInfo& runInfo,
                        const TeOpParas& opParas, int32_t core_num, int32_t ub_size) {
  MakePerformanceParams(slice_params);
  OP_LOGD(opType.c_str(), "perf slice params: %s", slice_params.to_string().c_str());

  SetTilingMode(slice_params, core_num, opParas.inputs[0].tensor[0].dtype, ub_size, opType);
  OP_LOGD(opType.c_str(), "set tiling_mode params: %s", slice_params.to_string().c_str());

  SetRuningParams(slice_params, runInfo);
  OP_LOGD(opType.c_str(), "tiling param:%s", to_string(runInfo.tiling_data).c_str());
}

REGISTER_OP_TILING_FUNC_BUFFERED(StridedSlice, StridedSliceTiling);
}  // namespace optiling
