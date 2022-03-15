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
  int32_t coreNum;
  bool inSpecialDevice;
  vector<int32_t> workspaceDimen;
  int32_t realC;
  vector<int32_t> ksizeHW;
  vector<int32_t> strideHW;
  vector<int32_t> dilateHW;
};

bool ExtractImagePatchesParseFunc(const std::string& op_type, const nlohmann::json& compile_info,
                                  ExtractImagePatchOpInfo& op_info) {
  OP_TILING_CHECK(!GetCompileValue(compile_info, "coreNum", op_info.coreNum),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ExtractImagePatchesParseFunc, get coreNum error"),
                  return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "inSpecialDevice", op_info.inSpecialDevice),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ExtractImagePatchesParseFunc, get inSpecialDevice error"),
                  return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "workspaceDimen", op_info.workspaceDimen),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ExtractImagePatchesParseFunc, get workspaceDimen error"),
                  return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "realC", op_info.realC),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ExtractImagePatchesParseFunc, get realC error"),
                  return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "ksizeHW", op_info.ksizeHW),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ExtractImagePatchesParseFunc, get ksizeHW error"),
                  return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "strideHW", op_info.strideHW),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ExtractImagePatchesParseFunc, get strideHW error"),
                  return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "dilateHW", op_info.dilateHW),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ExtractImagePatchesParseFunc, get dilateHW error"),
                  return false);

  OP_LOGI(op_type.c_str(), "GetCompileParams success.");
  return true;
}

int32_t CalMultiCoreFactor(const int32_t& device_core_num, vector<int64_t>& input_shape) {
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
  int32_t workspace_output = op_info.workspaceDimen[0];
  int32_t workspace_filter = op_info.workspaceDimen[1];
  int32_t workspace_c = op_info.workspaceDimen[2];
  int32_t real_c = op_info.realC;
  int32_t ksize_h = op_info.ksizeHW[0];
  int32_t ksize_w = op_info.ksizeHW[1];
  int32_t stride_h = op_info.strideHW[0];
  int32_t stride_w = op_info.strideHW[1];
  int32_t dilate_h = op_info.dilateHW[0];
  int32_t dilate_w = op_info.dilateHW[1];
  bool in_special_device = op_info.inSpecialDevice;

  int32_t fmap_x0 = input_shape[0];
  int32_t multi_core_factor_0 = CalMultiCoreFactor(device_core_num, input_shape);
  int32_t FOUR_DIMEN_KEY = 10000;
  int32_t SPECIAL_DIMEN_KEY = 20000;
  int32_t workspaces = FLOAT16_BYTE * fmap_x0 * workspace_output * workspace_filter * workspace_c;
  GELOGI("op[%s] Get ExtractImagePatchesCompileParams success.", op_type.c_str());

  run_info.AddTilingData(fmap_x0);
  run_info.AddTilingData(multi_core_factor_0);
  run_info.SetBlockDim(device_core_num);
  run_info.AddWorkspace(workspaces);
  if (input_shape[0] == 128 && input_shape[2] == 32 && input_shape[3] == 32 && real_c == 3 &&\
    in_special_device && ksize_h == 2 && ksize_w == 2 && stride_h == 2 && stride_w == 2 &&\
    dilate_h == 1 && dilate_w == 1) {
    run_info.SetTilingKey(SPECIAL_DIMEN_KEY);
  } else {
    run_info.SetTilingKey(FOUR_DIMEN_KEY);
  }

  GELOGI("ExtractImagePatchesTiling end.");
  return true;
}
REGISTER_OP_TILING_V3_CUSTOM(ExtractImagePatches, ExtractImagePatchesTiling,
                             ExtractImagePatchesParseFunc, ExtractImagePatchOpInfo);
}  // namespace optiling
