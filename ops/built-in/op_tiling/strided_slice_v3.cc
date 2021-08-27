/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file strided_slice_v3.cpp
 * \brief dynamic shape tiling of strided_slice_v3
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
#include "error_log.h"
#include "strided_slice.h"

namespace optiling {
using namespace ge;

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

bool StridedSliceV3Tiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                 OpRunInfo& runInfo) {
  using namespace nlohmann;
  OP_LOGD(opType.c_str(), "StridedSliceV3Tiling running.");

  if (opParas.inputs.empty() || opParas.inputs[0].tensor.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "StridedSliceV3Tiling: input shape error.");
    return false;
  }

  SliceParameters slice_params_output;
  slice_params_output.input = opParas.inputs[0].tensor[0].shape;

  //reading tensor begin end axes from tiling start
  std::vector<int64_t> input_axes_values;
  map<string, std::pair<int, vector<int64_t>&>> const_params = {
      {"begin", {1, slice_params_output.begin_list}},
      {"end", {2, slice_params_output.end_list}},
      {"axes", {4, input_axes_values}},
  };

  bool axes_exist = true;
  for (auto& item : const_params) {
    auto& name = item.first;
    int index = item.second.first;
    auto& values = item.second.second;
    if (!GetConstValue(opParas, name, opParas.inputs[index].tensor[0].dtype, values)) {
      if(name == "axes"){
        axes_exist = false;
        continue;
      }
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "Get %s values failed", name.c_str());
      return false;
    }
  }
  //reading tensor begin end axes from tiling end
  if(!axes_exist){
    for(size_t i = 0; i < slice_params_output.begin_list.size(); ++i){
      input_axes_values.push_back(static_cast<int64_t>(i));
    }
  }

  //the len of axes,begin,end must be the same  
  if (slice_params_output.begin_list.size() != slice_params_output.end_list.size()|| 
      slice_params_output.begin_list.size() != input_axes_values.size()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "length of axes, offsets and size must be equal.");
    ge::OpsInputShapeErrReport(
        opType, "length of axes,begin list and end list must be equal.", "[begin][end]",
        ConcatString("[", input_axes_values.size(), "]", "[", slice_params_output.begin_list.size(), "]", "[",
                     slice_params_output.end_list.size(), "]"));
    return false;
  }

  if(slice_params_output.input.size()!=slice_params_output.end_list.size()){

    int64_t begin_len = slice_params_output.input.size();
    //init begin list with 0
    std::vector<int64_t> processed_begin(begin_len, 0);
    //init end list with shape max
    std::vector<int64_t> processed_end = slice_params_output.input;
    //init to max shape,like[20,10,5]
    //assign the begin end values accoring to the axes values 
    //begin[0,0] end [10,5] axes [0,-2]  with  shape [20,10,5]
    //   |             |            |
    //   V             V            V
    // [0,0,0]        [10,5,5]     [0,1]
    for (size_t i = 0; i < input_axes_values.size(); ++i){
      int64_t axes_index = input_axes_values[i];
      // negative axes 
      if(axes_index < 0){
        axes_index = begin_len + axes_index;
      }
      // axes out of boundary
      if(axes_index >= begin_len){
        axes_index = begin_len - 1;
        OP_LOGD(opType.c_str(), "Pos Axes Value Out Of Boudary:%ld ,origin values: %ld", axes_index, input_axes_values[i]);
      }
      // axes out of boundary
      if(axes_index < 0){
        axes_index = 0;
        OP_LOGD(opType.c_str(), "Neg Axes Value Out Of Boudary:%ld ,origin values: %ld",  axes_index, input_axes_values[i]);
      }
      processed_end[axes_index] = slice_params_output.end_list[i];
      processed_begin[axes_index] = slice_params_output.begin_list[i];
    }
    //assign the proceseed value back to slice params
    slice_params_output.begin_list.assign(processed_begin.begin(),processed_begin.end());
    slice_params_output.end_list.assign(processed_end.begin(),processed_end.end());

  }
  
  // for start list processing
  for (size_t index = 0; index < slice_params_output.end_list.size(); index++) {
    //for shape[2,3,4,5] with end list [-1,-1,-1,-1]--->[1,2,3,4]
    if (slice_params_output.end_list[index] < 0) {
      slice_params_output.end_list[index] = slice_params_output.input[index] + slice_params_output.end_list[index];
    }
    //for end list out of boundary like [INT_MAX,INT_MAX,INT_MAX,INT_MAX] is cut to [2,3,4,5]
    if(slice_params_output.end_list[index] > slice_params_output.input[index]){
      slice_params_output.end_list[index] = slice_params_output.input[index];    
    }
  }
  // for end list processing
  for (size_t index = 0; index < slice_params_output.begin_list.size(); index++) {
    //for shape[2,3,4,5] with end list [-1,-1,-1,-1]--->[1,2,3,4]
    if (slice_params_output.begin_list[index] < 0) {
      slice_params_output.begin_list[index] = slice_params_output.input[index] + slice_params_output.begin_list[index];
    }
    //for end list out of boundary like [INT_MIN,INT_MIN,INT_MIN,INT_MIN] is cut to [0,0,0,0]
    if(slice_params_output.begin_list[index] < 0){
      slice_params_output.begin_list[index] = 0;  
    }
  }


  // get the output shape？
  slice_params_output.output_shape.resize(slice_params_output.input.size());
  for (size_t index = 0; index < slice_params_output.begin_list.size(); index++) {
    slice_params_output.output_shape[index] = slice_params_output.end_list[index] - slice_params_output.begin_list[index];
  }

  //setting the  default stride,now only support 1
  slice_params_output.stride_list.assign(slice_params_output.end_list.size(), 1);
  OP_LOGD(opType.c_str(), "origin stridedslicev3 params for stridedslice: %s", slice_params_output.to_string().c_str());

  int32_t core_num = GetCompileInfo<int32_t>(opCompileInfo, "block_dim");
  OP_TILING_CHECK(core_num == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "core_num cannot be zero."),
                  return false);
  int32_t ub_size = GetCompileInfo<int32_t>(opCompileInfo, "ub_size");
  SetSliceTilingData(opType, slice_params_output, runInfo, opParas, core_num, ub_size);

  runInfo.block_dim = core_num;
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;
  OP_LOGD(opType.c_str(), "tiling run success.");

  return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(StridedSliceV3, StridedSliceV3Tiling);
}  // namespace optiling
