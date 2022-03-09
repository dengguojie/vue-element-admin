/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#include <algorithm>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <math.h>

#include "../op_proto/util/error_util.h"
#include "../op_proto/util/op_common_util.h"
#include "graph/debug/ge_log.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "op_tiling.h"
#include "op_tiling_util.h"

namespace {
  constexpr int32_t TILING_FACTOR_0 = 1;
  constexpr int32_t FLOAT16_BYTE = 2;
}

namespace optiling {
struct ExtractImagePatchOpInfo {
  int32_t ubSize;
  int32_t coreNum;
  int32_t avgSplitUbSize;
  int32_t resUbNum;
  int32_t alignBlockSize;
  int32_t workspaceOutput;
  int32_t workspaceFilter;
  int32_t workspaceC;
};

bool ExtractImagePatchesParseFunc(const std::string& op_type, const nlohmann::json& compile_info,
                                  ExtractImagePatchOpInfo& op_info) {
  OP_TILING_CHECK(!GetCompileValue(compile_info, "ubSize", op_info.ubSize),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ExtractImagePatchesParseFunc, get ubSize error"),
                  return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "coreNum", op_info.coreNum),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ExtractImagePatchesParseFunc, get coreNum error"),
                  return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "avgSplitUbSize", op_info.avgSplitUbSize),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ExtractImagePatchesParseFunc, get avgSplitUbSize error"),
                  return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "resUbNum", op_info.resUbNum),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ExtractImagePatchesParseFunc, get resUbNum error"),
                  return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "alignBlockSize", op_info.alignBlockSize),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ExtractImagePatchesParseFunc, get alignBlockSize error"),
                  return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "workspaceOutput", op_info.workspaceOutput),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ExtractImagePatchesParseFunc, get workspaceOutput error"),
                  return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "workspaceFilter", op_info.workspaceFilter),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ExtractImagePatchesParseFunc, get workspaceFilter error"),
                  return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "workspaceC", op_info.workspaceC),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ExtractImagePatchesParseFunc, get workspaceC error"),
                  return false);

  OP_LOGI(op_type.c_str(), "GetCompileParams success.");
  return true;
}

int32_t CalMultiCoreFactor(const int32_t& device_core_num, vector<int64_t>& input_shape){
  float input_shape_0 = static_cast<float>(input_shape[0]);
  int32_t min_need_core_num = ceil(input_shape_0/TILING_FACTOR_0);
  int32_t used_core_num = (min_need_core_num > device_core_num) ? device_core_num : min_need_core_num;
  int32_t input_shape_per_core = ceil(input_shape_0/used_core_num);
  int32_t multi_core_factor_0 = (input_shape_per_core > TILING_FACTOR_0) ? input_shape_per_core : TILING_FACTOR_0;
  return multi_core_factor_0;
}

bool ExtractImagePatchesTiling(const std::string& op_type, const ge::Operator& op_paras,
                               const ExtractImagePatchOpInfo& op_info, utils::OpRunInfo& run_info) {
  GELOGI("ExtractImagePatchesTiling running.");
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  auto input0_desc = operator_info->MutableInputDesc(0);
  std::vector<int64_t> input_shape = input0_desc->MutableShape().GetDims();
  int32_t device_core_num = op_info.coreNum;
  int32_t ubSize = op_info.ubSize;
  int32_t avg_split_ub_size = op_info.avgSplitUbSize;
  int32_t res_ub_num = op_info.resUbNum;
  int32_t align_block_size = op_info.alignBlockSize;
  int32_t workspace_output = op_info.workspaceOutput;
  int32_t workspace_filter = op_info.workspaceFilter;
  int32_t workspace_c = op_info.workspaceC;

  int32_t fmap_x0 = input_shape[0];
  int32_t multi_core_factor_0 = CalMultiCoreFactor(device_core_num, input_shape);
  int32_t max_dtype = 0;
  int32_t FOUR_DIMEN_KEY = 10000;
  int32_t workspaces = FLOAT16_BYTE * fmap_x0 * workspace_output * workspace_filter * workspace_c;
  GELOGI("op[%s] Get ExtractImagePatchesCompileParams success.", op_type.c_str());

  run_info.AddTilingData(fmap_x0);
  run_info.AddTilingData(multi_core_factor_0);
  run_info.SetBlockDim(device_core_num);
  run_info.SetTilingKey(FOUR_DIMEN_KEY);
  run_info.AddWorkspace(workspaces);

  GELOGI("ExtractImagePatchesTiling end.");
  return true;
}
REGISTER_OP_TILING_V3_CUSTOM(ExtractImagePatches, ExtractImagePatchesTiling, ExtractImagePatchesParseFunc, ExtractImagePatchOpInfo);
}  // namespace optiling
