/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#include "op_tiling_util.h"
#include "graph/debug/ge_log.h"

#include "../op_proto/util/error_util.h"
#include "../op_proto/util/op_common_util.h"
#include "op_log.h"
#include "error_log.h"
#include "strided_slice.h"
#include "graph/utils/op_desc_utils.h"
#include "vector_tiling_profiling.h"

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

bool SliceTiling(const std::string& opType, const ge::Operator& opParas, const nlohmann::json& opCompileInfo,
                 utils::OpRunInfo& runInfo) {
  using namespace nlohmann;
  OP_LOGD(opType.c_str(), "SliceTiling running.");

  PROFILING_TILING_INIT(opType.c_str());
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(opParas);
  if (operator_info == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "operator_info is null.");
    return false;
  }
  auto opdesc = operator_info->MutableInputDesc(0);
  if (opdesc == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "opdesc is null.");
    return false;
  }

  SliceParameters slice_params_output;
  slice_params_output.input = opdesc->MutableShape().GetDims();
  map<string, std::pair<int, vector<int64_t>&>> const_params = {
      {"offsets", {1, slice_params_output.begin_list}},
      {"size", {2, slice_params_output.end_list}},
  };

  for (auto& item : const_params) {
    auto& name = item.first;
    auto& values = item.second.second;
    if (!GetConstValue(opParas, name, values)) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "Get %s values failed", name.c_str());
      return false;
    }
  }

  if (slice_params_output.begin_list.size() != slice_params_output.end_list.size() ||
      slice_params_output.begin_list.size() != slice_params_output.input.size()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "length of input_shape, offsets and size must be equal.");
    ge::OpsInputShapeErrReport(
        opType, "length of x's shape, offsets and size must be equal.", "[x][offsets][size]",
        ConcatString("[", slice_params_output.input.size(), "]", "[", slice_params_output.begin_list.size(), "]", "[",
                     slice_params_output.end_list.size(), "]"));
    return false;
  }

  PROFILING_TILING_AFTER_GET_SHAPE_REG();

  int32_t core_num = GetCompileInfo<int32_t>(opCompileInfo, "block_dim");
  OP_TILING_CHECK(core_num == 0, VECTOR_INNER_ERR_REPORT_TILIING(opType, "core_num cannot be zero."), return false);
  int32_t ub_size = GetCompileInfo<int32_t>(opCompileInfo, "ub_size");
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  for (size_t index = 0; index < slice_params_output.end_list.size(); index++) {
    if (slice_params_output.end_list[index] == -1) {
      slice_params_output.end_list[index] = slice_params_output.input[index] - slice_params_output.begin_list[index];
    }
  }

  for (size_t i = 0; i < slice_params_output.begin_list.size(); i++) {
    if (slice_params_output.begin_list[i] < 0 ||
        slice_params_output.begin_list[i] + slice_params_output.end_list[i] < slice_params_output.begin_list[i] ||
        slice_params_output.begin_list[i] + slice_params_output.end_list[i] > slice_params_output.input[i]) {
      VECTOR_INNER_ERR_REPORT_TILIING(
          opType,
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

  ge::DataType input_dtype = opdesc->GetDataType();
  SetSliceTilingData(opType, slice_params_output, runInfo, input_dtype, core_num, ub_size);

  PROFILING_TILING_AFTER_CALCU_TILING_REG();

  runInfo.SetBlockDim(core_num);
  OP_LOGD(opType.c_str(), "tiling run success.");

  PROFILING_TILING_END();

  return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED_V2(Slice, SliceTiling);
}  // namespace optiling
