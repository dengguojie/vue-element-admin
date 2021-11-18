/* Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include <algorithm>
#include <nlohmann/json.hpp>
#include <string>

#include "../op_proto/util/error_util.h"
#include "../op_proto/util/op_common_util.h"
#include "graph/debug/ge_log.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "op_tiling.h"
#include "op_tiling_util.h"

namespace {
  constexpr int32_t WO_CUT_64 = 64;
  constexpr int32_t WO_CUT_32 = 32;
  constexpr int32_t WO_CUT_8 = 8;
  constexpr int32_t WO_CUT_2 = 2;
  constexpr int32_t BASE_DIM_NUM = 2;
}

namespace optiling {
// CASE1 CASE2 CASE3 CASE7 CASE8 CASE9 cut nc1h axis for block parallel, CASE1 CASE3 CASE7 CASE8 CASE9 cut nc1h_in
// and CASE2 cut wo.
// CASE4 CASE5 cut wo for block parallel, CASE4 cut wo_block_in.
enum tiling_case { CASE1, CASE2, CASE3, CASE4, CASE5, CASE6, CASE7, CASE8, CASE9 };
struct opInfo {
  int32_t core_num;
  int32_t max_w_in_ub;
  int32_t ksize;
  int32_t strides;
  int32_t pad_l;
  int32_t pad_r;
  bool ceil_mode;
};

bool AvgPool1DParseFunc(const std::string& op_type, const nlohmann::json& compile_info, opInfo& compile_value) {
  OP_TILING_CHECK(!GetCompileValue(compile_info, "core_num", compile_value.core_num),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "AvgPool1DParseFunc get core_num error"), return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "max_w_in_ub", compile_value.max_w_in_ub),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "AvgPool1DParseFunc get max_w_in_ub error"), return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "ksize", compile_value.ksize),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "AvgPool1DParseFunc get ksize error"), return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "strides", compile_value.strides),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "AvgPool1DParseFunc get strides error"), return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "pad_l", compile_value.pad_l),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "AvgPool1DParseFunc get pad_l error"), return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "pad_r", compile_value.pad_r),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "AvgPool1DParseFunc get pad_r error"), return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "ceil_mode", compile_value.ceil_mode),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "AvgPool1DParseFunc get ceil_mode error"), return false);
  OP_LOGI(op_type.c_str(), "GetCompileParams success.");
  return true;
}
vector<int32_t> GetTilingData(int32_t nc1h, int32_t wo, int32_t max_w_in_ub, int32_t core_num, int32_t& block_dim) {
  vector<int32_t> tiling_data;
  if (nc1h >= core_num) {
    // do not need to cut wo for block parallel
    int32_t nc1h_factor = (nc1h + core_num - 1) / core_num;
    int32_t nc1h_parts = (nc1h + nc1h_factor - 1) / nc1h_factor;
    block_dim = nc1h_parts;
    if (wo > max_w_in_ub) {
      // case 2 cut wo
      int32_t wo_factor = max_w_in_ub;
      tiling_data.push_back(CASE2);
      tiling_data.push_back(nc1h_factor);
      tiling_data.push_back(wo_factor);
    } else if (wo <= max_w_in_ub / WO_CUT_64) {
      tiling_data.push_back(CASE9);
      tiling_data.push_back(nc1h_factor);
    } else if (wo <= max_w_in_ub / WO_CUT_32) {
      tiling_data.push_back(CASE8);
      tiling_data.push_back(nc1h_factor);
    } else if (wo <= max_w_in_ub / WO_CUT_8) {
      tiling_data.push_back(CASE7);
      tiling_data.push_back(nc1h_factor);
    } else if (wo <= max_w_in_ub / WO_CUT_2) {
      tiling_data.push_back(CASE1);
      tiling_data.push_back(nc1h_factor);
    } else {
      tiling_data.push_back(CASE3);
      tiling_data.push_back(nc1h_factor);
    }
  } else {
    if (wo < BASE_DIM_NUM) {
      // wo dim is not large enough to cut for block parallel
      block_dim = nc1h;
      tiling_data.push_back(CASE6);
    } else {
      int32_t wo_parts = core_num / nc1h;
      int32_t wo_block_factor = wo_parts < wo ? wo / wo_parts : 1;
      int32_t wo_real_parts = (wo + wo_block_factor - 1) / wo_block_factor;
      block_dim = nc1h * wo_real_parts;
      if (wo_block_factor > max_w_in_ub) {
        // wo_block_factor need to cut further to storage in ub
        int32_t wo_block_in_factor = max_w_in_ub;
        tiling_data.push_back(CASE4);
        tiling_data.push_back(wo_block_factor);
        tiling_data.push_back(wo_block_in_factor);
      } else {
        // wo_block_factor,C0 can be calculated whole
        tiling_data.push_back(CASE5);
        tiling_data.push_back(wo_block_factor);
      }
    }
  }
  return tiling_data;
}

bool AvgPool1DTiling(const std::string& op_type, const ge::Operator& op_paras, const opInfo& op_compile_info_json,
                     utils::OpRunInfo& run_info) {
  GELOGI("AvgPool1DTiling running.");
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  std::vector<int64_t> input_shape = operator_info->MutableInputDesc(0)->MutableShape().GetDims();
  std::vector<int64_t> output_shape = operator_info->MutableOutputDesc(0)->MutableShape().GetDims();
  int32_t fmap_n = input_shape[0];
  int32_t fmap_c1 = input_shape[1];
  int32_t fmap_h = input_shape[2];
  int32_t fmap_w = input_shape[3];
  int32_t wo = output_shape[3];
  int32_t fmap_nc1h = fmap_n * fmap_c1 * fmap_h;
  int32_t core_num = op_compile_info_json.core_num;
  int32_t max_w_in_ub = op_compile_info_json.max_w_in_ub;
  int32_t ksize = op_compile_info_json.ksize;
  int32_t strides = op_compile_info_json.strides;
  int32_t pad_l = op_compile_info_json.pad_l;
  int32_t pad_r = op_compile_info_json.pad_r;
  bool ceil_mode = op_compile_info_json.ceil_mode;
  int32_t fmap_wo;
  if (ceil_mode) {
    fmap_wo = (fmap_w + pad_l + pad_r - ksize + strides - 1) / strides + 1;
  } else {
    fmap_wo = (fmap_w + pad_l + pad_r - ksize) / strides + 1;
  }
  if (pad_l) {
    if ((fmap_wo - 1) * strides >= fmap_w + pad_l) {
      fmap_wo -= 1;
    }
  }

  int32_t block_dim = 0;
  vector<int32_t> tiling_data = GetTilingData(fmap_nc1h, wo, max_w_in_ub, core_num, block_dim);
  run_info.AddTilingData(fmap_nc1h);
  run_info.AddTilingData(fmap_w);
  run_info.AddTilingData(fmap_wo);
  for (size_t i = 1; i < tiling_data.size(); i++) {
    run_info.AddTilingData(tiling_data[i]);
  }

  run_info.SetBlockDim(block_dim);
  run_info.SetTilingKey(tiling_data[0]);

  GELOGI("AvgPool1DTiling end.");
  return true;
}
REGISTER_OP_TILING_V3_CUSTOM(AvgPool1DD, AvgPool1DTiling, AvgPool1DParseFunc, opInfo);
}  // namespace optiling
