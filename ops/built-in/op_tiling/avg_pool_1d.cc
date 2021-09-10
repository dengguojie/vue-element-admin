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
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "op_tiling.h"

namespace optiling {
// CASE1 CASE2 CASE3 CASE7 CASE8 CASE9 cut nc1h axis for block parallel, CASE1 CASE3 CASE7 CASE8 CASE9 cut nc1h_in
// and CASE2 cut wo.
// CASE4 CASE5 cut wo for block parallel, CASE4 cut wo_block_in.
enum tiling_case {CASE1, CASE2, CASE3, CASE4, CASE5, CASE6, CASE7, CASE8, CASE9};

vector<int32_t> GetTilingData(int32_t nc1h, int32_t wo, int32_t max_w_in_ub, int32_t core_num, int32_t &block_dim) {
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
    } else if (wo <= max_w_in_ub / 64) {
      tiling_data.push_back(CASE9);
      tiling_data.push_back(nc1h_factor);
    } else if (wo <= max_w_in_ub / 32) {
      tiling_data.push_back(CASE8);
      tiling_data.push_back(nc1h_factor);
    } else if (wo <= max_w_in_ub / 8) {
      tiling_data.push_back(CASE7);
      tiling_data.push_back(nc1h_factor);
    } else if (wo <= max_w_in_ub / 2) {
      tiling_data.push_back(CASE1);
      tiling_data.push_back(nc1h_factor);
    } else {
      tiling_data.push_back(CASE3);
      tiling_data.push_back(nc1h_factor);
    }
  } else {
    if (wo < 2) {
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

bool AvgPool1DTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_compile_info_json,
                     OpRunInfo& run_info) {
  GELOGI("AvgPool1DTiling running.");
  std::vector<int64_t> input_shape = op_paras.inputs[0].tensor[0].shape;
  std::vector<int64_t> output_shape = op_paras.outputs[0].tensor[0].shape;
  int32_t fmap_n = input_shape[0];
  int32_t fmap_c1 = input_shape[1];
  int32_t fmap_h = input_shape[2];
  int32_t fmap_w = input_shape[3];
  int32_t wo = output_shape[3];
  int32_t fmap_nc1h = fmap_n * fmap_c1 * fmap_h;
  int32_t core_num = op_compile_info_json["core_num"].get<int32_t>();
  int32_t max_w_in_ub = op_compile_info_json["max_w_in_ub"].get<int32_t>();
  int32_t ksize = op_compile_info_json["ksize"].get<int32_t>();
  int32_t strides = op_compile_info_json["strides"].get<int32_t>();
  int32_t pad_l = op_compile_info_json["pad_l"].get<int32_t>();
  int32_t pad_r = op_compile_info_json["pad_r"].get<int32_t>();
  bool ceil_mode = op_compile_info_json["ceil_mode"].get<bool>();
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
  ByteBufferPut(run_info.tiling_data, fmap_nc1h);
  ByteBufferPut(run_info.tiling_data, fmap_w);
  ByteBufferPut(run_info.tiling_data, fmap_wo);
  for (size_t i = 1; i < tiling_data.size(); i++) {
    ByteBufferPut(run_info.tiling_data, tiling_data[i]);
  }

  run_info.block_dim = block_dim;
  run_info.tiling_key = tiling_data[0];

  GELOGI("AvgPool1DTiling end.");
  return true;
}
REGISTER_OP_TILING_FUNC_BUFFERED(AvgPool1DD, AvgPool1DTiling);
}  // namespace optiling
