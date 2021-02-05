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
 * \file resize_nearest_neighbor_v2.cpp
 * \brief
 */
#include <string>
#include <math.h>
#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "op_log.h"

namespace optiling {

struct ResizeNearestNeighborV2TilingParams
{
    int64_t tiling_key;
    int64_t input_batch;
    int64_t input_c1;
    int64_t input_height;
    int64_t input_weight;
    int64_t output_height;
    int64_t output_weight;
    int64_t cut_batch_c1_num;
    int64_t cut_height_num;
    int64_t cut_weight_num;
};

bool GetResizeNearestNeighborV2CompileParams(const nlohmann::json& compile_info,
                                             int64_t& core_num,
                                             int64_t& max_w_len,
                                             int64_t& align_corners,
                                             int64_t& half_pixel_centers) {
  using namespace nlohmann;
  auto allVars = compile_info["vars"];
  if (allVars.count("core_num") == 0) {
    OP_LOGE("op [ResizeNearestNeighborV2Tiling] : GetCompileParams, get core_num error");
    return false;
  }
  core_num = allVars["core_num"].get<std::int64_t>();
  if (allVars.count("max_w_len") == 0) {
    OP_LOGE("op [ResizeNearestNeighborV2Tiling] : GetCompileParams, get max_w_len error");
    return false;
  }
  max_w_len = allVars["max_w_len"].get<std::int64_t>();
  if (allVars.count("align_corners") == 0) {
    OP_LOGE("op [ResizeNearestNeighborV2Tiling] : GetCompileParams, get align_corners error");
    return false;
  }
  align_corners = allVars["align_corners"].get<std::int64_t>();
  if (allVars.count("half_pixel_centers") == 0) {
    OP_LOGE("op [ResizeNearestNeighborV2Tiling] : GetCompileParams, get half_pixel_centers error");
    return false;
  }
  half_pixel_centers = allVars["half_pixel_centers"].get<std::int64_t>();
  return true;
}

bool ResizeNearestNeighborV2Tiling(const std::string& op_type,
                                   const TeOpParas& op_paras,
                                   const nlohmann::json& op_info,
                                   OpRunInfo& run_info) {
  using namespace ge;

  OP_LOGI(op_type.c_str(), "tiling run begin.");

  if (op_paras.inputs.empty()) {
    OP_LOGE(op_type.c_str(), "Length of inputs is empty.");
    return false;
  }
  if (op_paras.outputs.empty()) {
    OP_LOGE(op_type.c_str(), "Length of outputs is empty.");
    return false;
  }
  const std::vector<int64_t>& input_shape = op_paras.inputs[0].tensor[0].shape;
  const std::vector<int64_t>& output_shape = op_paras.outputs[0].tensor[0].shape;
  ResizeNearestNeighborV2TilingParams tiling_params;
  int64_t core_num = 0;
  int64_t max_w_len = 0;
  int64_t align_corners = 0;
  int64_t half_pixel_centers = 0;
  if (!GetResizeNearestNeighborV2CompileParams(op_info, core_num, max_w_len,
                                               align_corners, half_pixel_centers)) {
    OP_LOGE(op_type.c_str(), "get compile info from json failed.");
    return false;
  }
  if (input_shape.size() != 5) {
    OP_LOGE(op_type.c_str(), "the input shape size must be 5(NC1HWC0).");
    return false;
  }
  if (output_shape.size() != 5) {
    OP_LOGE(op_type.c_str(), "the output shape size must be 5(NC1HWC0).");
    return false;
  }
  tiling_params.input_batch = input_shape[0];
  tiling_params.input_c1 = input_shape[1];
  tiling_params.input_height = input_shape[2];
  tiling_params.output_height = output_shape[2];
  tiling_params.input_weight = input_shape[3];
  tiling_params.output_weight = output_shape[3];

  // cut num
  tiling_params.cut_batch_c1_num = 1;
  tiling_params.cut_height_num = 1;
  tiling_params.cut_weight_num = 1;

  // calcu tiling
  // check whether h,w to nh,nw
  bool is_h_nh = ((tiling_params.output_height % tiling_params.input_height == 0
                   &&  align_corners + half_pixel_centers == 0)
                  || (tiling_params.output_height == tiling_params.input_height));
  bool is_nh_h = (tiling_params.input_height % tiling_params.output_height == 0
                  &&  align_corners + half_pixel_centers == 0);
  bool is_w_nw = ((tiling_params.output_weight % tiling_params.input_weight == 0
                   &&  align_corners + half_pixel_centers == 0)
                  || tiling_params.output_weight == tiling_params.input_weight);
  auto image_batch_c1 = tiling_params.input_batch * tiling_params.input_c1;
  tiling_params.tiling_key = is_w_nw ? 110000 : 100000;

  if (is_h_nh && is_w_nw) {
    // process h * w, so cut by nc1 first from input
    tiling_params.tiling_key = 200000;
    auto cut_batch_c1_sigment = (image_batch_c1 + core_num - 1) / core_num;
    tiling_params.cut_batch_c1_num = (image_batch_c1 + cut_batch_c1_sigment - 1) / cut_batch_c1_sigment;
    auto left_core_num = core_num - tiling_params.cut_batch_c1_num;
    if (left_core_num != 0) {
      // charge whether continue cut weight
      left_core_num = (core_num  + tiling_params.cut_batch_c1_num - 1) / tiling_params.cut_batch_c1_num;
      tiling_params.cut_batch_c1_num = core_num / left_core_num;
      auto cut_w_sigment = (tiling_params.output_weight + left_core_num - 1) / left_core_num;
      tiling_params.cut_weight_num = (tiling_params.output_weight + cut_w_sigment - 1) / cut_w_sigment;
    }
  } else {
    // process nc1 * w
    // 1. first cut by w to get more nc1
    // 2. second cut h
    auto nc_max = (image_batch_c1 + core_num - 1) / core_num;
    nc_max = (image_batch_c1 + nc_max - 1) / nc_max;
    auto h_max = (tiling_params.output_height + core_num - 1) / core_num;
    h_max = (tiling_params.output_height + h_max - 1) / h_max;
    auto w_max = (tiling_params.output_weight + core_num - 1) / core_num;
    w_max = (tiling_params.output_weight + w_max - 1) / w_max;
    for (int64_t i = 0; i < core_num; ++i) {
      int64_t cut_weight_num_tmp = pow(2, i);
      if (cut_weight_num_tmp > core_num) {
        tiling_params.cut_weight_num = w_max;
        break;
      }
      int64_t w_sigment = (tiling_params.output_weight + cut_weight_num_tmp - 1) / cut_weight_num_tmp;
      if (w_sigment <= 256) {
        tiling_params.cut_weight_num = min(cut_weight_num_tmp, w_max);
        break;
      }
    }
    // cut h
    auto left_core_num = core_num / tiling_params.cut_weight_num;
    tiling_params.cut_height_num = min(left_core_num, h_max);
    tiling_params.cut_height_num = tiling_params.cut_height_num != 0 ? tiling_params.cut_height_num : 1;
  }

  OP_LOGD(op_type.c_str(), "tiling_data, tiling_key = %d.", tiling_params.tiling_key);
  OP_LOGD(op_type.c_str(), "tiling_data, input_batch_c1 = %d.", image_batch_c1);
  OP_LOGD(op_type.c_str(), "tiling_data, input_height = %d.", tiling_params.input_height);
  OP_LOGD(op_type.c_str(), "tiling_data, input_weight = %d.", tiling_params.input_weight);
  OP_LOGD(op_type.c_str(), "tiling_data, output_height = %d.", tiling_params.output_height);
  OP_LOGD(op_type.c_str(), "tiling_data, output_weight = %d.", tiling_params.output_weight);
  OP_LOGD(op_type.c_str(), "tiling_data, cut_batch_c1_num = %d.", tiling_params.cut_batch_c1_num);
  OP_LOGD(op_type.c_str(), "tiling_data, cut_height_num = %d.", tiling_params.cut_height_num);
  OP_LOGD(op_type.c_str(), "tiling_data, cut_weight_num = %d.", tiling_params.cut_weight_num);

  ByteBufferPut(run_info.tiling_data, tiling_params.tiling_key);
  ByteBufferPut(run_info.tiling_data, tiling_params.input_batch);
  ByteBufferPut(run_info.tiling_data, tiling_params.input_c1);
  ByteBufferPut(run_info.tiling_data, tiling_params.input_height);
  ByteBufferPut(run_info.tiling_data, tiling_params.input_weight);
  ByteBufferPut(run_info.tiling_data, tiling_params.output_height);
  ByteBufferPut(run_info.tiling_data, tiling_params.output_weight);
  ByteBufferPut(run_info.tiling_data, tiling_params.cut_batch_c1_num);
  ByteBufferPut(run_info.tiling_data, tiling_params.cut_height_num);
  ByteBufferPut(run_info.tiling_data, tiling_params.cut_weight_num);
  run_info.block_dim = core_num;
  std::vector<int64_t> workspace;
  run_info.workspaces = workspace;
  OP_LOGI(op_type.c_str(), "tiling run success.");
  return true;
}

// register tiling interface of the ResizeNearestNeighborV2 op.
REGISTER_OP_TILING_FUNC_BUFFERED(ResizeNearestNeighborV2, ResizeNearestNeighborV2Tiling);
}  // namespace optiling
