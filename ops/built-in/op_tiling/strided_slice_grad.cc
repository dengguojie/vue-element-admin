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
 * \file strided_slice_grad.cpp
 * \brief
 */
#include <map>

#include <securec.h>

#include "op_log.h"
#include "pad_common.h"

namespace optiling {

struct SliceParameters {
  std::vector<int64_t> input;
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

static bool GetConstValue(const TeOpParas& paras, const std::string& name, const std::string& dtype,
                          std::vector<int64_t>& values) {
  values.clear();
  if (paras.const_inputs.count(name) == 0) {
    return false;
  }

  auto size = std::get<1>(paras.const_inputs.at(name));
  if (dtype == "int64") {
    int count = size / sizeof(int64_t);
    values.resize(count);
    if (EOK != memcpy_s(values.data(), count * sizeof(int64_t), std::get<0>(paras.const_inputs.at(name)),
                        std::get<1>(paras.const_inputs.at(name)))) {
      return false;
    }
  } else if (dtype == "int32") {
    int count = size / sizeof(int32_t);
    std::vector<int32_t> tmp(count, 0);
    if (EOK != memcpy_s(tmp.data(), count * sizeof(int32_t), std::get<0>(paras.const_inputs.at(name)),
                        std::get<1>(paras.const_inputs.at(name)))) {
      return false;
    }
    values.insert(values.end(), tmp.begin(), tmp.end());
  }

  return true;
}

static bool GetStridedSliceSocParams(const std::string& opType, const nlohmann::json& opCompileInfo, int32_t& maxCore,
                                     int32_t& ubSize, uint32_t& begin_mask, uint32_t& end_mask, uint32_t& ellipsis_mask,
                                     uint32_t& new_axis_mask, uint32_t& shrink_axis_mask) {
  using namespace nlohmann;
  const auto& allVars = opCompileInfo["vars"];

  if (allVars.count("maxCore") == 0) {
    OP_LOGE(opType.c_str(), "op[SSGTiling] : GetCompileParams, get maxCore error");
    return false;
  }
  maxCore = allVars["maxCore"].get<std::int32_t>();

  if (allVars.count("ubSize") == 0) {
    OP_LOGE(opType.c_str(), "op[SSGTiling] : GetCompileParams, get ubSize error");
    return false;
  }
  ubSize = allVars["ubSize"].get<std::int32_t>();

  if (allVars.count("begin_mask") == 0) {
    OP_LOGE(opType.c_str(), "op[SSGTiling] : GetCompileParams, get begin_mask error");
    return false;
  }
  begin_mask = allVars["begin_mask"].get<std::uint32_t>();

  if (allVars.count("end_mask") == 0) {
    OP_LOGE(opType.c_str(), "op [SSGTiling] : GetCompileParams, get end_mask error");
    return false;
  }
  end_mask = allVars["end_mask"].get<std::uint32_t>();

  if (allVars.count("ellipsis_mask") == 0) {
    OP_LOGE(opType.c_str(), "op [StridedSliceTiling] : GetCompileParams, get ellipsis_mask error");
    return false;
  }
  ellipsis_mask = allVars["ellipsis_mask"].get<std::uint32_t>();

  if (allVars.count("new_axis_mask") == 0) {
    OP_LOGE(opType.c_str(), "op [StridedSliceTiling] : GetCompileParams, get new_axis_mask error");
    return false;
  }
  new_axis_mask = allVars["new_axis_mask"].get<std::uint32_t>();

  if (allVars.count("shrink_axis_mask") == 0) {
    OP_LOGE(opType.c_str(), "op [StridedSliceTiling] : GetCompileParams, get new_axis_mask error");
    return false;
  }
  shrink_axis_mask = allVars["shrink_axis_mask"].get<std::int32_t>();

  return true;
}

static void StridedSliceInferShape(const std::string& opType, const std::vector<int64_t> input_shape,
                                   struct SliceParameters& slice_params, struct SliceMasks& slicemasks) {
  int32_t base_number = 2;
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

void _printTensorValue(std::vector<int64_t>& in, int64_t len, std::string name) {
  using namespace std;
  string vec_str;
  for (auto item : in) {
    vec_str += to_string(item);
    vec_str += ",";
  }
  OP_LOGI("Op[StrideSliceGrad]", "Func[_printTensorValue] [%s]: [%s].", name.c_str(), vec_str.c_str());
}

bool CheckTensorValue(const std::string& opType, SliceParameters& slice_params_output) {
  int64_t length_shape = slice_params_output.input.size();
  int64_t length_begin = slice_params_output.begin_list.size();
  int64_t length_end = slice_params_output.end_list.size();
  int64_t length_strides = slice_params_output.stride_list.size();

  // rule of length
  if (length_shape != length_begin && length_end != length_strides && length_begin != length_end) {
    OP_LOGE(opType.c_str(), "tensors' shapes are not matched");
    return false;
  }

  // rule of value
  int64_t begin_i = 0;
  int64_t shape_i = 0;
  int64_t end_i = 0;

  for (int64_t i = 0; i < length_shape; i++) {
    begin_i = slice_params_output.begin_list[i];
    end_i = slice_params_output.end_list[i];
    shape_i = slice_params_output.input[i];

    begin_i = (begin_i < 0) ? begin_i + shape_i : begin_i;
    end_i = (end_i < 0) ? end_i + shape_i : end_i;

    if (not(begin_i >= 0 && end_i <= shape_i && begin_i <= end_i)) {
      OP_LOGE(opType.c_str(), "Bound Over: begin[%d]:%d, end[%d]:%d, shape_x[%d]:%d", i, begin_i, i, end_i, i, shape_i);
      return false;
    }
  }

  for (auto item : slice_params_output.stride_list) {
    if (item != 1) {
      OP_LOGE(opType.c_str(), "value of strides must be 1 not %d", item);
      return false;
    }
  }
  return true;
}

bool StridedSliceGradTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                            OpRunInfo& runInfo) {
  using namespace std;
  OP_LOGI(opType.c_str(), "SSG_Tiling running.");

  ///////////////////////////////////////////////////////////
  // Get Info of tensors from Func "StridedSliceGrad"(SSG).//
  //////////////////////////////////////////////////////////
  struct SliceParameters slice_params_output = {};
  struct SliceMasks slicemasks_output = {};

  map<string, std::pair<int, vector<int64_t>&>> const_params = {
      {"shape", {0, slice_params_output.input}},
      {"begin", {1, slice_params_output.begin_list}},
      {"end", {2, slice_params_output.end_list}},
      {"strides", {3, slice_params_output.stride_list}},
  };

  // Get ConstInput
  for (auto& item : const_params) {
    auto& name = item.first;
    int index = item.second.first;
    auto& values = item.second.second;
    if (!GetConstValue(opParas, name, opParas.inputs[index].tensor[0].dtype, values)) {
      OP_LOGE(opType.c_str(), "Get %s values failed", name.c_str());
      return false;
    }
  }

  // Print Org_Tensor'values
  int64_t length_shape = slice_params_output.input.size();
  int64_t length_begin = slice_params_output.begin_list.size();
  int64_t length_end = slice_params_output.end_list.size();
  int64_t length_strides = slice_params_output.stride_list.size();
  _printTensorValue(slice_params_output.input, length_shape, "shape");
  _printTensorValue(slice_params_output.begin_list, length_begin, "begin");
  _printTensorValue(slice_params_output.end_list, length_end, "end");
  _printTensorValue(slice_params_output.stride_list, length_strides, "strides");

  // Get CompileParams
  int32_t maxCore = 0;
  int32_t ubSize = 0;
  bool flag = GetStridedSliceSocParams(opType, opCompileInfo, maxCore, ubSize, slicemasks_output.beginmask,
                                       slicemasks_output.endmask, slicemasks_output.ellipsismask,
                                       slicemasks_output.newaxismask, slicemasks_output.shrinkaxismask);
  if (!flag) {
    return false;
  }

  // Calc Paddings
  bool cond0 = slice_params_output.input.size() > slice_params_output.begin_list.size();
  bool cond1 = slice_params_output.begin_list.size() == 2;
  bool cond2 = slicemasks_output.ellipsismask == 1;
  bool cond3 = slicemasks_output.shrinkaxismask == 2;
  uint64_t length = slice_params_output.input.size();
  std::vector<std::vector<int64_t>> padding(length, std::vector<int64_t>(2));
  // input_shape is not change, it will be used in calc padding.
  std::vector<int64_t> input_shape = slice_params_output.input;
  vector<int64_t> ori_begin = slice_params_output.begin_list;
  int choose_padding = 0;

  // Get InferShape: produce begin_shape, end_shape for calc padding
  StridedSliceInferShape(opType, input_shape, slice_params_output, slicemasks_output);

  if (cond0 && cond1 && cond2 && cond3) {
    choose_padding = 1;
    for (uint64_t i = 0; i < length; i++) {
      padding[i][0] = 0;
      padding[i][1] = 0;
    }
    padding[length - 1][0] = ori_begin[1];
    padding[length - 1][1] = input_shape[length - 1] - ori_begin[1] - 1;
  } else {
    int64_t begin_i = 0;
    int64_t shape_i = 0;
    int64_t end_i = 0;

    for (uint64_t i = 0; i < length; i++) {
      begin_i = slice_params_output.begin_list[i];
      end_i = slice_params_output.end_list[i];
      shape_i = input_shape[i];

      begin_i = (begin_i < 0) ? begin_i + shape_i : begin_i;
      end_i = (end_i < 0) ? end_i + shape_i : end_i;

      padding[i][0] = begin_i;
      padding[i][1] = shape_i - end_i;
    }
  }

  // check_rule: shape, begin, and, strides
  bool success = CheckTensorValue(opType, slice_params_output);
  if (!success) {
    return false;
  }

  ////////////////////////////
  /////////////PadD///////////
  ////////////////////////////
  // Get inShape outShape dtype padding
  padCommon pad;
  std::vector<int64_t> inShape = opParas.inputs[4].tensor[0].shape;
  if (choose_padding == 1){
    GELOGD("op[%s] inShape need insert the last dim that is 1.", opType.c_str());
    inShape.push_back(1);
  }
  std::vector<int64_t> outShape = input_shape;
  const std::string dtype = opParas.inputs[4].tensor[0].dtype;
  int numBit = pad._numBit(dtype);
  bool ValidTensor = pad.CheckTensor(inShape, outShape);
  if (!ValidTensor) {
    return false;
  }

  /////////////////////////////////
  //---Get Params for Running----//
  /////////////////////////////////
  PadDTilingParams runParams;
  pad.InitTilingParams(runParams, int(inShape.size()));

  // Discriminate Align(1) and Not Align(0).
  runParams.branch =
      pad.CheckBranch(inShape, outShape, padding, numBit, 0) * pad.CheckBranch(inShape, outShape, padding, numBit, 1);

  // Get Params In Circulation Layer
  pad.GetDepth(inShape, outShape, padding, runParams.depth, maxCore, numBit, runParams.branch);
  pad.GetCirculateParams("top", numBit, maxCore, inShape, outShape, padding, runParams);
  pad.GetCirculateParams("bottom", numBit, maxCore, inShape, outShape, padding, runParams);

  // Get Params In Recursion Layer
  if (runParams.branch == 1) {
    pad.GetRecurCore(runParams, inShape, outShape, padding, maxCore, numBit, ubSize);
  } else {
    pad.GetRecurCorePro(runParams, inShape, outShape, padding, maxCore, numBit, ubSize);
  }

  pad.SetRunningParams(runParams, runInfo);
  pad.PrintRunningParams(runParams);

  runInfo.block_dim = uint32_t(maxCore);
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;
  GELOGI("op[%s] tiling run success.", opType.c_str());
  return true;
}

// register tiling interface of the StridedSliceGrad op.
REGISTER_OP_TILING_FUNC_BUFFERED(StridedSliceGrad, StridedSliceGradTiling);
}  // namespace optiling
