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
  uint32_t beginmask = 0;
  uint32_t endmask = 0;
  uint32_t ellipsismask = 0;
  uint32_t newaxismask = 0;
  uint32_t shrinkaxismask = 0;
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

static void StridedSliceInferShape(const std::string& opType, const std::vector<int64_t>& input_shape,
                                   struct SliceParameters& slice_params, struct SliceMasks& slicemasks) {
  int32_t base_number = 2.0;
  int32_t newbeginmask = 0;
  int32_t newendmask = 0;
  int32_t newshrinkmask = 0;
  int32_t newaxismask = 0;
  uint32_t begin_len = slice_params.begin_list.size();
  uint32_t dim_num = input_shape.size();
  slice_params.input = input_shape;
  // compute the right_move of begin end stride and masks
  // because of non-zero ellipsismask
  int32_t right_move = std::max<int64_t>(dim_num - begin_len, 0);
  if (dim_num < begin_len && slicemasks.newaxismask != 0) {
    dim_num = begin_len;
  }

  // rebuild the begin end stride of new_axis,
  // because ignored when new_axis is true.
  if (slicemasks.newaxismask != 0) {
    for (uint32_t i = 0; i < dim_num; i++) {
      if ((slicemasks.newaxismask & ((uint64_t)pow(base_number, i))) == ((uint64_t)pow(base_number, i))) {
        slice_params.begin_list[i] = 0;
        slice_params.end_list[i] = 1;
        slice_params.stride_list[i] = 1;
        if ((slicemasks.shrinkaxismask & ((uint64_t)pow(base_number, i))) == ((uint64_t)pow(base_number, i))) {
          slicemasks.shrinkaxismask -= (uint64_t)pow(base_number, i);
        }
      }
    }
  }
  if (slicemasks.ellipsismask != 0) {
    for (size_t i = 0; i < dim_num; i++) {
      if ((slicemasks.ellipsismask & ((uint64_t)pow(base_number, i))) == ((uint64_t)pow(base_number, i))) {
        if ((slicemasks.shrinkaxismask & ((uint64_t)pow(base_number, i))) == ((uint64_t)pow(base_number, i))) {
          slicemasks.shrinkaxismask -= (uint64_t)pow(base_number, i);
        }
        if ((slicemasks.newaxismask & ((uint64_t)pow(base_number, i))) == ((uint64_t)pow(base_number, i))) {
          slicemasks.newaxismask -= (uint64_t)pow(base_number, i);
        }
        if ((slicemasks.beginmask & ((uint64_t)pow(base_number, i))) == ((uint64_t)pow(base_number, i))) {
          slicemasks.beginmask -= (uint64_t)pow(base_number, i);
        }
        if ((slicemasks.endmask & ((uint64_t)pow(base_number, i))) == ((uint64_t)pow(base_number, i))) {
          slicemasks.endmask -= (uint64_t)pow(base_number, i);
        }
      }
    }
  }
  int32_t tmp_shrink = 0;
  if (slicemasks.shrinkaxismask != 0) {
    for (size_t i = 0; i < dim_num; i++) {
      if ((slicemasks.shrinkaxismask & ((uint64_t)pow(base_number, i))) == ((uint64_t)pow(base_number, i))) {
        if (begin_len > i) {
          tmp_shrink += (uint64_t)pow(base_number, i);
        }
      }
    }
    slicemasks.shrinkaxismask = tmp_shrink;
  }
  int32_t tmp_new_axis = 0;
  if (slicemasks.newaxismask != 0) {
    for (size_t i = 0; i < dim_num; i++) {
      if ((slicemasks.newaxismask & ((uint64_t)pow(base_number, i))) == ((uint64_t)pow(base_number, i))) {
        tmp_new_axis += 1;
      }
      right_move += tmp_new_axis;
    }
  }
  if (slicemasks.ellipsismask != 0) {
    uint32_t bitellipsis = (uint64_t)log2(slicemasks.ellipsismask);
    for (size_t i = 0; i < dim_num; i++) {
      if ((slicemasks.beginmask & (1 << i)) && (bitellipsis >= i)) {
        newbeginmask += (uint64_t)pow(base_number, i);
      } else if ((slicemasks.beginmask & (1 << i)) && (bitellipsis < i)) {
        newbeginmask += (uint64_t)pow(base_number, i + right_move);
      }
      if ((slicemasks.endmask & (1 << i)) && (bitellipsis >= i)) {
        newendmask += (uint64_t)pow(base_number, i);
      } else if ((slicemasks.endmask & (1 << i)) && (bitellipsis < i)) {
        newendmask += (uint64_t)pow(base_number, i + right_move);
      }
      if ((slicemasks.shrinkaxismask & (1 << i)) && (bitellipsis >= i)) {
        newshrinkmask += (uint64_t)pow(base_number, i);
      } else if ((slicemasks.shrinkaxismask & (1 << i)) && (bitellipsis < i)) {
        newshrinkmask += (uint64_t)pow(base_number, i + right_move);
      }
      if ((slicemasks.newaxismask & (1 << i)) && (bitellipsis >= i)) {
        newaxismask += (uint64_t)pow(base_number, i);
      } else if ((slicemasks.newaxismask & (1 << i)) && (bitellipsis < i)) {
        newaxismask += (uint64_t)pow(base_number, i + right_move);
      }
    }
    slicemasks.beginmask = newbeginmask;
    slicemasks.endmask = newendmask;
    slicemasks.shrinkaxismask = newshrinkmask;
    slicemasks.newaxismask = newaxismask;
  }
  for (size_t i = 0; i < dim_num; i++) {
    if ((slicemasks.newaxismask & ((uint64_t)pow(base_number, i))) == ((uint64_t)pow(base_number, i))) {
      slice_params.input.insert(slice_params.input.begin() + i, 1);
    }
  }
  uint32_t bitellipsis = (uint64_t)log2(slicemasks.ellipsismask);
  if (slicemasks.ellipsismask != 0 && bitellipsis > begin_len - 1) {
    if (begin_len < dim_num) {
      for (size_t i = 0; i < dim_num - begin_len; i++) {
        slice_params.begin_list.push_back(0);
        slice_params.end_list.push_back(slice_params.input[begin_len + i]);
        slice_params.stride_list.push_back(1);
        begin_len += 1;
      }
    }
    if (slicemasks.ellipsismask != 0) {
      for (size_t i = 0; i < dim_num; i++) {
        if ((slicemasks.ellipsismask & ((uint64_t)pow(base_number, i))) == ((uint64_t)pow(base_number, i))) {
          uint32_t ellipsis_dim = i;
          slice_params.begin_list[i] = 0;
          slice_params.end_list[i] = input_shape[i];
          slice_params.stride_list[i] = 1;
          if ((slicemasks.shrinkaxismask & ((uint64_t)pow(base_number, i))) == ((uint64_t)pow(base_number, i))) {
            slicemasks.shrinkaxismask -= (uint64_t)pow(base_number, i);
          }
          if (begin_len < dim_num + tmp_new_axis) {
            uint32_t begin_len_tmp = begin_len;
            for (size_t j = 1; j <= dim_num + tmp_new_axis - begin_len_tmp; j++) {
              slice_params.begin_list.insert(slice_params.begin_list.begin() + ellipsis_dim + j, 0);
              slice_params.end_list.insert(slice_params.end_list.begin() + ellipsis_dim + j,
                                           slice_params.input[ellipsis_dim + j]);
              slice_params.stride_list.insert(slice_params.stride_list.begin() + ellipsis_dim + j, 1);
            }
          }
        }
      }
    }
  } else {
    if (slicemasks.ellipsismask != 0) {
      for (size_t i = 0; i < dim_num; i++) {
        if ((slicemasks.ellipsismask & ((uint64_t)pow(base_number, i))) == ((uint64_t)pow(base_number, i))) {
          uint32_t ellipsis_dim = i;
          slice_params.begin_list[i] = 0;
          slice_params.end_list[i] = input_shape[i];
          slice_params.stride_list[i] = 1;
          if ((slicemasks.shrinkaxismask & ((uint64_t)pow(base_number, i))) == ((uint64_t)pow(base_number, i))) {
            slicemasks.shrinkaxismask -= (uint64_t)pow(base_number, i);
          }
          if (begin_len < dim_num + tmp_new_axis) {
            uint32_t begin_len_tmp = begin_len;
            for (size_t j = 1; j <= dim_num + tmp_new_axis - begin_len_tmp; j++) {
              slice_params.begin_list.insert(slice_params.begin_list.begin() + ellipsis_dim + j, 0);
              slice_params.end_list.insert(slice_params.end_list.begin() + ellipsis_dim + j,
                                           slice_params.input[ellipsis_dim + j]);
              slice_params.stride_list.insert(slice_params.stride_list.begin() + ellipsis_dim + j, 1);
              begin_len += 1;
            }
          }
        }
      }
    }
    if (begin_len < slice_params.input.size()) {
      for (size_t i = 0; i < slice_params.input.size() - begin_len; i++) {
        slice_params.begin_list.push_back(0);
        slice_params.end_list.push_back(slice_params.input[begin_len + i]);
        slice_params.stride_list.push_back(1);
      }
    }
  }
  for (size_t i = 0; i < dim_num; i++) {
    if (slice_params.begin_list[i] < 0) {
      slice_params.begin_list[i] = input_shape[i] + slice_params.begin_list[i];
    }
    if (slice_params.end_list[i] < 0) {
      slice_params.end_list[i] = input_shape[i] + slice_params.end_list[i];
    }
  }
  for (size_t i = 0; i < dim_num; i++) {
    if ((slicemasks.beginmask & ((uint64_t)pow(base_number, i))) == ((uint64_t)pow(base_number, i))) {
      if (slice_params.stride_list[i] > 0) {
        slice_params.begin_list[i] = 0;
      }
      if (slice_params.stride_list[i] < 0) {
        slice_params.begin_list[i] = slice_params.input[i];
      }
    }
    if ((slicemasks.endmask & ((uint64_t)pow(base_number, i))) == ((uint64_t)pow(base_number, i))) {
      if (slice_params.stride_list[i] > 0) {
        slice_params.end_list[i] = slice_params.input[i];
      }
      if (slice_params.stride_list[i] < 0) {
        slice_params.end_list[i] = 0;
      }
    }
    if ((slicemasks.ellipsismask & ((uint64_t)pow(base_number, i))) == ((uint64_t)pow(base_number, i))) {
      slice_params.begin_list[i] = 0;
      slice_params.end_list[i] = input_shape[i];
      slice_params.stride_list[i] = 1;
    }
  }

  uint32_t new_axis_flag = 0;
  for (size_t i = 0; i < dim_num; i++) {
    if ((slicemasks.newaxismask & ((uint64_t)pow(base_number, i))) == ((uint64_t)pow(base_number, i))) {
      new_axis_flag += 1;
    }
  }

  for (size_t i = 0; i < slice_params.input.size(); i++) {
    if ((slicemasks.shrinkaxismask & ((uint64_t)pow(base_number, i))) == ((uint64_t)pow(base_number, i))) {
      slice_params.end_list[i] = slice_params.begin_list[i] + 1;
    }
  }

  for (size_t i = 0; i < slice_params.begin_list.size(); i++) {
    if (slice_params.stride_list[i] > 0) {
      if (slice_params.begin_list[i] >= slice_params.end_list[i]) {
        slice_params.begin_list[i] = slice_params.end_list[i];
      }

      if (slice_params.end_list[i] > slice_params.input[i]) {
        slice_params.end_list[i] = slice_params.input[i];
      }
      if (slice_params.end_list[i] == 0) {
        slice_params.begin_list[i] = slice_params.end_list[i];
      }
      if (slice_params.begin_list[i] < 0 && slice_params.end_list[i] >= 0) {
        slice_params.begin_list[i] = 0;
        if (slice_params.end_list[i] >= slice_params.input[i]) {
          slice_params.end_list[i] = slice_params.input[i];
        }
      }
    }
    if (slice_params.stride_list[i] < 0) {
      if (slice_params.begin_list[i] >= slice_params.input[i]) {
        if (slice_params.end_list[i] >= 0) {
          slice_params.begin_list[i] = slice_params.input[i] - 1;
        }
        if (slice_params.end_list[i] < 0) {
          slice_params.begin_list[i] = slice_params.input[i];
          slice_params.end_list[i] = 0;
        }
      }
      if (slice_params.begin_list[i] == 0) {
        if (slice_params.begin_list[i] <= slice_params.end_list[i]) {
          slice_params.begin_list[i] = slice_params.end_list[i];
        }
        if (slice_params.begin_list[i] > slice_params.end_list[i]) {
          slice_params.begin_list[i] = 0;
          slice_params.end_list[i] = -1;
        }
      }
    }
  }
}

static void MakeSameDims(SliceParameters &parameters, const vector<int64_t> &shape) {
  if (parameters.input.size() <= shape.size()) {
    return;
  }

  bool same_size = parameters.input.size() == parameters.begin_list.size() &&
      parameters.input.size() == parameters.end_list.size() &&
      parameters.input.size() == parameters.stride_list.size() &&
      parameters.input.size() == parameters.output_shape.size();

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
    if (parameters.input[i] == shape[j]) {
      tmp_input.push_back(parameters.input[i]);
      tmp_begin_list.push_back(parameters.begin_list[i]);
      tmp_end_list.push_back(parameters.end_list[i]);
      tmp_stride_list.push_back(parameters.stride_list[i]);
      tmp_output_shape.push_back(parameters.output_shape[i]);
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

  struct SliceParameters slice_params_output = {};
  struct SliceMasks slicemasks_output = {};

  const std::vector<int64_t>& input_shape = opParas.inputs[0].tensor[0].shape;
  map<string, std::pair<int, vector<int64_t>&>> const_params = {
      {"begin", {1, slice_params_output.begin_list}},
      {"end", {2, slice_params_output.end_list}},
      {"strides", {3, slice_params_output.stride_list}},
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

  for (size_t i = 0; i < slice_params_output.stride_list.size(); i++) {
    if (slice_params_output.stride_list[i] == 0) {
      OP_LOGE(opType.c_str(), "StridedSliceTiling: the dim of stride must be non-zero");
      ge::OpsAttrValueErrReport(opType, "strides", "non-zero", "0");
      return false;
    }

    if (slice_params_output.stride_list[i] != 1) {
      OP_LOGE(opType.c_str(), "StridedSliceTiling: stride[%zu] must be 1, but it is %lld", i,
              slice_params_output.stride_list[i]);
      ge::OpsAttrValueErrReport(opType, "strides", "1", ConcatString(slice_params_output.stride_list[i]));
      return false;
    }
  }

  int32_t core_num = 0;
  bool flag = GetStridedSliceSocParams(opType, opCompileInfo, core_num, slicemasks_output.beginmask,
                                       slicemasks_output.endmask, slicemasks_output.ellipsismask,
                                       slicemasks_output.newaxismask, slicemasks_output.shrinkaxismask);
  if (!flag) {
    OP_LOGE(opType.c_str(), "StridedSliceTiling: get soc params error");

    return false;
  }

  int32_t ellipsis_dim = 0;
  auto shape_len = input_shape.size();
  if (slice_params_output.begin_list.size() != slice_params_output.end_list.size()) {
    OP_LOGE(opType.c_str(), "StridedSliceTiling:end's shape and begin's shape must be match");
    ge::OpsInputShapeErrReport(
        opType, "length of begin and end must be equal.", "[begin][end]",
        ConcatString("[", slice_params_output.begin_list.size(), "]", "[", slice_params_output.end_list.size(), "]"));
    return false;
  }

  if (slicemasks_output.ellipsismask != 0) {
    for (size_t i = 0; i < shape_len; ++i) {
      if ((slicemasks_output.ellipsismask & ((uint64_t)pow(2.0, i))) == (uint64_t)pow(2.0, i)) {
        ellipsis_dim += 1;
      }
    }
    if (ellipsis_dim > 1) {
      OP_LOGE(opType.c_str(), "StridedSliceTiling:only support 1 dim of ellipsis!");
      return false;
    }
  }
  if (shape_len < slice_params_output.begin_list.size()) {
    if (slicemasks_output.newaxismask != 0) {
      for (size_t i = 0; i < slice_params_output.begin_list.size(); i++) {
        if ((slicemasks_output.newaxismask & ((uint64_t)pow(2.0, i))) != (uint64_t)pow(2.0, i)) {
          OP_LOGE(opType.c_str(), "StridedSliceTiling: the length of begin not more than the length of input");
          return false;
        }
      }
    }
    if (slicemasks_output.newaxismask == 0) {
      OP_LOGE(opType.c_str(), "StridedSliceTiling: the length of begin not more than the length of input");
      return false;
    }
  }

  StridedSliceInferShape(opType, input_shape, slice_params_output, slicemasks_output);

  uint32_t shrinkaxismaskTemp = 0;
  int32_t base_number = 2.0;
  if (slicemasks_output.shrinkaxismask != 0) {
    for (size_t i = 0; i < shape_len; ++i) {
      if ((slice_params_output.end_list[i] - slice_params_output.begin_list[i]) == 0) {
        shrinkaxismaskTemp += pow(base_number, i);
      }
    }
  }
  slicemasks_output.shrinkaxismask = slicemasks_output.shrinkaxismask | shrinkaxismaskTemp;
  std::vector<int64_t> outputshapelist;
  // Convert the target data into a double type by multiply '1.0'
  double changeToDouble = 1.0;
  for (size_t i = 0; i < slice_params_output.begin_list.size(); ++i) {
    int32_t dim = (int64_t)(ceil((slice_params_output.end_list[i] - slice_params_output.begin_list[i]) /
                                 (slice_params_output.stride_list[i] * changeToDouble)));
    dim = std::max<int64_t>(dim, int64_t(0));
    if (((slicemasks_output.shrinkaxismask & ((uint64_t)pow(2.0, i))) != ((uint64_t)pow(2.0, i))) ||
        ((slicemasks_output.newaxismask & ((uint64_t)pow(2.0, i))) != ((uint64_t)pow(2.0, i)))) {
      // get outputshape
      outputshapelist.push_back(dim);
    }
  }

  if (slicemasks_output.shrinkaxismask == 0 && slicemasks_output.newaxismask == 0) {
    if (slice_params_output.begin_list.size() > slice_params_output.input.size()) {
      for (size_t i = 0; i < slice_params_output.begin_list.size() - slice_params_output.input.size(); i++) {
        outputshapelist.erase(outputshapelist.begin() + i + slice_params_output.input.size());
      }
    }
  }
  // shrink_axis_mask != 0
  if (slicemasks_output.shrinkaxismask > 0) {
    int32_t shrink_flag = 0;
    for (size_t i = 0; i < shape_len; i++) {
      if ((slicemasks_output.shrinkaxismask & ((uint64_t)pow(base_number, i))) == ((uint64_t)pow(base_number, i))) {
        outputshapelist.erase(outputshapelist.begin() + i - shrink_flag);
        shrink_flag += 1;
      }
    }
  }
  if (slicemasks_output.shrinkaxismask > 0) {
    for (size_t i = 0; i < shape_len; i++) {
      if ((slicemasks_output.shrinkaxismask & ((uint64_t)pow(base_number, i))) == ((uint64_t)pow(base_number, i))) {
        outputshapelist.insert(outputshapelist.begin() + i, 1);
      }
    }
  }

  slice_params_output.output_shape = outputshapelist;
  MakeSameDims(slice_params_output, input_shape);
  SetRuningParams(slice_params_output, runInfo);
  OP_LOGD(opType.c_str(), "tiling param:%s", to_string(runInfo.tiling_data).c_str());
  runInfo.block_dim = core_num;
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;
  OP_LOGD(opType.c_str(), "tiling run success.");

  return true;
}

REGISTER_OP_TILING_FUNC(StridedSlice, StridedSliceTiling);
}  // namespace optiling
