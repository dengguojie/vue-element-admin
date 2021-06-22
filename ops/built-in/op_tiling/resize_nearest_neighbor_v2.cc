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
#include "error_log.h"

namespace optiling {

// tiling_key format: 000000
// 1. Reserved, default 1
// 2. h align flag, 0: h -> x.x*h, 1: h -> nh, 2: nh -> h, 3: h = h
// 3. w align flag, 0: w -> x.x*w, 1: w -> nw, 2: nw -> w, 3: w = w
// 4. src stride flag, 0: can not copy with stride 1: can copy with stride
// 5. des stride flag, 0: can not copy with stride 1: can copy with stride
// 6. Reserved, default 0
const int64_t DEFAULT_TILING_MODE = 100000;
const int64_t HEIGHT_ALIGN_FLAG = 10000;
const int64_t WEIGHT_ALIGN_FLAG = 1000;
const int64_t WIDTH_ALIGN_FLAG = 100;
const int64_t BIG_TO_SMALL_FLAG = 10;

struct ResizeNearestNeighborV2TilingParams {
  int64_t tiling_key;
  int64_t input_batch;
  int64_t input_c1;
  int64_t input_height;
  int64_t input_weight;
  int64_t output_height;
  int64_t output_weight;
  // cut core num by batch * C1
  int64_t cut_batch_c1_num;
  // cut core num by height
  int64_t cut_height_num;
  // cut core num by weight
  int64_t cut_weight_num;
};

struct ResizeNearestNeighborV2CompileParams {
  int64_t core_num;
  int64_t max_w_len;
  int64_t align_corners;
  int64_t half_pixel_centers;
  std::string op_type;
};

static bool GetResizeNearestNeighborV2CompileParams(const nlohmann::json& compile_info,
                                                    ResizeNearestNeighborV2CompileParams& compile_params) {
  using namespace nlohmann;
  auto allVars = compile_info["vars"];
  if (allVars.count("core_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "GetCompileParams, get core_num error");
    return false;
  }
  compile_params.core_num = allVars["core_num"].get<std::int64_t>();
  if (allVars.count("max_w_len") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "GetCompileParams, get max_w_len error");
    return false;
  }
  compile_params.max_w_len = allVars["max_w_len"].get<std::int64_t>();
  if (allVars.count("align_corners") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "GetCompileParams, get align_corners error");
    return false;
  }
  compile_params.align_corners = allVars["align_corners"].get<std::int64_t>();
  if (allVars.count("half_pixel_centers") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "GetCompileParams, get half_pixel_centers error");
    return false;
  }
  compile_params.half_pixel_centers = allVars["half_pixel_centers"].get<std::int64_t>();
  return true;
}

static void GetTilingParamForHW2MHNW(const ResizeNearestNeighborV2CompileParams& compile_params,
                                     ResizeNearestNeighborV2TilingParams& tiling_params) {
  auto image_batch_c1 = tiling_params.input_batch * tiling_params.input_c1;
  auto cut_batch_c1_sigment = (image_batch_c1 + compile_params.core_num - 1) / compile_params.core_num;
  tiling_params.cut_batch_c1_num = (image_batch_c1 + cut_batch_c1_sigment - 1) / cut_batch_c1_sigment;
  auto left_core_num = compile_params.core_num - tiling_params.cut_batch_c1_num;
  if (left_core_num != 0) {
    if (tiling_params.input_weight > left_core_num * 4) {
      // charge whether continue cut weight
      left_core_num = (compile_params.core_num + tiling_params.cut_batch_c1_num - 1) / tiling_params.cut_batch_c1_num;
      tiling_params.cut_batch_c1_num = compile_params.core_num / left_core_num;
      auto cut_w_sigment = (tiling_params.input_weight + left_core_num - 1) / left_core_num;
      int64_t min_w_sigment = 16;
      cut_w_sigment = max(min_w_sigment, cut_w_sigment);
      tiling_params.cut_weight_num = (tiling_params.input_weight + cut_w_sigment - 1) / cut_w_sigment;
    }
  }
  left_core_num = compile_params.core_num / (tiling_params.cut_batch_c1_num * tiling_params.cut_weight_num);
  // continue cut height
  if (left_core_num > 1) {
    auto cut_h_sigment = (tiling_params.input_height + left_core_num - 1) / left_core_num;
    auto h_max = (tiling_params.input_height + cut_h_sigment - 1) / cut_h_sigment;
    tiling_params.cut_height_num = min(left_core_num, h_max);
  }
}

static void GetTilingParamForNW2W(const ResizeNearestNeighborV2CompileParams& compile_params,
                                  ResizeNearestNeighborV2TilingParams& tiling_params) {
  // cut weight first
  if (tiling_params.output_weight <= compile_params.core_num) {
    tiling_params.cut_weight_num = tiling_params.output_weight;
  } else {
    for (int64_t i = (compile_params.core_num - 1); i > 0; i--) {
      if (tiling_params.output_weight % i == 0) {
        tiling_params.cut_weight_num = i;
        break;
      }
    }
  }
  auto left_core_num = compile_params.core_num - tiling_params.cut_weight_num;
  if (left_core_num != 0) {
    // continue cut height
    left_core_num = compile_params.core_num / tiling_params.cut_weight_num;
    auto cut_h_sigment = (tiling_params.input_height + left_core_num - 1) / left_core_num;
    auto h_max = (tiling_params.input_height + cut_h_sigment - 1) / cut_h_sigment;
    tiling_params.cut_height_num = min(left_core_num, h_max);
  }
  left_core_num = compile_params.core_num / (tiling_params.cut_weight_num * tiling_params.input_height);
  // continue cut height
  auto image_batch_c1 = tiling_params.input_batch * tiling_params.input_c1;
  if (left_core_num > 1) {
    auto cut_batch_c1_sigment = (image_batch_c1 + left_core_num - 1) / left_core_num;
    tiling_params.cut_batch_c1_num = (image_batch_c1 + cut_batch_c1_sigment - 1) / cut_batch_c1_sigment;
    tiling_params.cut_batch_c1_num = min(left_core_num, tiling_params.cut_batch_c1_num);
  }
}

static void GetTilingParamForDefault(const ResizeNearestNeighborV2CompileParams& compile_params, const bool is_w_nw,
                                     ResizeNearestNeighborV2TilingParams& tiling_params) {
  auto h_max = (tiling_params.output_height + compile_params.core_num - 1) / compile_params.core_num;
  h_max = (tiling_params.output_height + h_max - 1) / h_max;
  auto image_batch_c1 = tiling_params.input_batch * tiling_params.input_c1;
  auto cut_weight_total = is_w_nw ? tiling_params.input_weight : tiling_params.output_weight;
  auto w_max = (cut_weight_total + compile_params.core_num - 1) / compile_params.core_num;
  w_max = (cut_weight_total + w_max - 1) / w_max;
  for (int64_t i = 0; i < compile_params.core_num; ++i) {
    int64_t cut_weight_num_tmp = pow(2, i);
    if (cut_weight_num_tmp > compile_params.core_num) {
      tiling_params.cut_weight_num = min(cut_weight_num_tmp / 2, w_max);
      break;
    }
    int64_t w_sigment = (cut_weight_total + cut_weight_num_tmp - 1) / cut_weight_num_tmp;
    if (w_sigment <= 256) {
      tiling_params.cut_weight_num = min(cut_weight_num_tmp, w_max);
      break;
    }
  }
  auto left_core_num = compile_params.core_num / tiling_params.cut_weight_num;
  auto nc_max = (image_batch_c1 + left_core_num - 1) / left_core_num;
  nc_max = (image_batch_c1 + nc_max - 1) / nc_max;
  // when w_cut * NC1 > compile_params.max_w_sigment, will cut NC1 first
  int64_t w_sigment = (cut_weight_total + tiling_params.cut_weight_num - 1) / tiling_params.cut_weight_num;
  auto image_batch_c1_w = image_batch_c1 * min(w_sigment, int64_t(128));
  auto nc_cut_by_compile = (image_batch_c1_w + compile_params.max_w_len - 1) / compile_params.max_w_len;
  if (image_batch_c1_w > compile_params.max_w_len) {
    tiling_params.cut_batch_c1_num = min(nc_cut_by_compile, nc_max);
    tiling_params.cut_batch_c1_num = max(int64_t(1), tiling_params.cut_batch_c1_num);
    left_core_num = compile_params.core_num / (tiling_params.cut_weight_num * tiling_params.cut_batch_c1_num);
  }
  // cut h
  tiling_params.cut_height_num = min(left_core_num, h_max);
  tiling_params.cut_height_num = tiling_params.cut_height_num != 0 ? tiling_params.cut_height_num : 1;
}

static void GetTilingParamForResizeNearestNeighborV2GradDefault(
    const ResizeNearestNeighborV2CompileParams& compile_params, ResizeNearestNeighborV2TilingParams& tiling_params) {
  auto h_max = (tiling_params.input_height + compile_params.core_num - 1) / compile_params.core_num;
  h_max = (tiling_params.input_height + h_max - 1) / h_max;
  auto image_batch_c1 = tiling_params.input_batch * tiling_params.input_c1;
  auto w_max = (tiling_params.input_weight + compile_params.core_num - 1) / compile_params.core_num;
  w_max = (tiling_params.input_weight + w_max - 1) / w_max;
  for (int64_t i = 0; i < compile_params.core_num; ++i) {
    int64_t cut_weight_num_tmp = pow(2, i);
    if (cut_weight_num_tmp > compile_params.core_num) {
      tiling_params.cut_weight_num = min(cut_weight_num_tmp / 2, w_max);
      break;
    }
    int64_t w_sigment = (tiling_params.input_weight + cut_weight_num_tmp - 1) / cut_weight_num_tmp;
    if (w_sigment <= 256) {
      tiling_params.cut_weight_num = min(cut_weight_num_tmp, w_max);
      break;
    }
  }
  auto left_core_num = compile_params.core_num / tiling_params.cut_weight_num;
  auto nc_max = (image_batch_c1 + left_core_num - 1) / left_core_num;
  nc_max = (image_batch_c1 + nc_max - 1) / nc_max;
  // when w_cut * NC1 > compile_params.max_w_len, will cut NC1 first
  int64_t w_sigment = (tiling_params.input_weight + tiling_params.cut_weight_num - 1) / tiling_params.cut_weight_num;
  auto image_batch_c1_w = image_batch_c1 * min(w_sigment, int64_t(128));
  auto nc_cut_by_compile = (image_batch_c1_w + compile_params.max_w_len - 1) / compile_params.max_w_len;
  if (image_batch_c1_w > compile_params.max_w_len) {
    tiling_params.cut_batch_c1_num = min(nc_cut_by_compile, nc_max);
    tiling_params.cut_batch_c1_num = max(int64_t(1), tiling_params.cut_batch_c1_num);
    left_core_num = compile_params.core_num / (tiling_params.cut_weight_num * tiling_params.cut_batch_c1_num);
  }
  // cut h
  tiling_params.cut_height_num = min(left_core_num, h_max);
  tiling_params.cut_height_num = tiling_params.cut_height_num != 0 ? tiling_params.cut_height_num : 1;
  left_core_num = compile_params.core_num /
                  (tiling_params.cut_weight_num * tiling_params.cut_batch_c1_num * tiling_params.cut_height_num);
  int64_t cut_weight_height_num = tiling_params.cut_weight_num * tiling_params.cut_height_num;
  if (left_core_num > 1 && image_batch_c1 > 1 && cut_weight_height_num < 17) {
    left_core_num = compile_params.core_num / cut_weight_height_num;
    nc_max = (image_batch_c1 + left_core_num - 1) / left_core_num;
    nc_max = (image_batch_c1 + nc_max - 1) / nc_max;
    tiling_params.cut_batch_c1_num = min(left_core_num, nc_max);
  }
}

static bool GetTilingParam(const ResizeNearestNeighborV2CompileParams& compile_params,
                           ResizeNearestNeighborV2TilingParams& tiling_params) {
  // check whether h,w to nh,nw
  bool is_h_nh = ((tiling_params.output_height % tiling_params.input_height == 0 &&
                   compile_params.align_corners + compile_params.half_pixel_centers == 0) ||
                  (tiling_params.output_height == tiling_params.input_height));
  bool is_w_nw = ((tiling_params.output_weight % tiling_params.input_weight == 0 &&
                   compile_params.align_corners + compile_params.half_pixel_centers == 0) ||
                  tiling_params.output_weight == tiling_params.input_weight);
  // h is not h-> mh and  w -> nw, will run output cut branch
  // is the n is too large (> 100), will not use hign performance branch
  is_w_nw = (!is_h_nh && tiling_params.output_weight > tiling_params.input_weight * 100) ? false : is_w_nw;
  // h is h-> mh and  w -> nw, n > max_w_len,  will not use hign performance branch
  is_w_nw = (is_h_nh && tiling_params.output_weight > tiling_params.input_weight * compile_params.max_w_len) ? false
                                                                                                             : is_w_nw;
  int64_t h_tiling_align_flag = 0;
  int64_t w_tiling_align_flag = is_w_nw ? 1 : 0;

  if (is_h_nh && is_w_nw) {
    // process h * w, so cut by nc1 first from input
    h_tiling_align_flag = 1;
    if (tiling_params.output_weight == tiling_params.input_weight) {
      w_tiling_align_flag = 3;
    }
    GetTilingParamForHW2MHNW(compile_params, tiling_params);
  } else {
    // process nc1 * w
    // 1. first cut by w to get more nc1
    // 2. second cut h
    GetTilingParamForDefault(compile_params, is_w_nw, tiling_params);
  }
  tiling_params.tiling_key =
      tiling_params.tiling_key + h_tiling_align_flag * HEIGHT_ALIGN_FLAG + w_tiling_align_flag * WEIGHT_ALIGN_FLAG;
  return true;
}

static bool GetTilingParamResizeNearestNeighborV2Grad(const ResizeNearestNeighborV2CompileParams& compile_params,
                                                      ResizeNearestNeighborV2TilingParams& tiling_params) {
  // check whether h,w to nh,nw
  bool is_h_nh = ((tiling_params.output_height % tiling_params.input_height == 0 &&
                   compile_params.align_corners + compile_params.half_pixel_centers == 0) ||
                  (tiling_params.output_height == tiling_params.input_height));
  bool is_w_nw = ((tiling_params.output_weight % tiling_params.input_weight == 0 &&
                   compile_params.align_corners + compile_params.half_pixel_centers == 0) ||
                  tiling_params.output_weight == tiling_params.input_weight);
  // h is not h-> mh and  w -> nw, will run output cut branch
  // is the n is too large (> 100), will not use hign performance branch
  is_w_nw = (!is_h_nh && tiling_params.output_weight > tiling_params.input_weight * 100) ? false : is_w_nw;
  // h is h-> mh and  w -> nw, n > max_w_len,  will not use hign performance branch
  is_w_nw = (is_h_nh && tiling_params.output_weight > tiling_params.input_weight * compile_params.max_w_len) ? false
                                                                                                             : is_w_nw;
  int64_t h_tiling_align_flag = 0;
  int64_t w_tiling_align_flag = is_w_nw ? 1 : 0;

  bool is_nw_w = ((tiling_params.input_weight % tiling_params.output_weight == 0) &&
                  (compile_params.align_corners + compile_params.half_pixel_centers == 0));
  is_nw_w = (tiling_params.input_weight > tiling_params.output_weight * 120) ? false : is_nw_w;
  bool is_big_to_small = tiling_params.input_weight > tiling_params.output_weight;
  int64_t w_align_flag = (is_big_to_small && is_nw_w) ? 1 : 0;
  int64_t is_big_to_small_flag = is_big_to_small ? 1 : 0;

  if (is_h_nh && is_w_nw) {
    // process h * w, so cut by nc1 first from input
    h_tiling_align_flag = 1;
    GetTilingParamForHW2MHNW(compile_params, tiling_params);
  } else if (w_align_flag) {
    GetTilingParamForNW2W(compile_params, tiling_params);
  } else {
    // process nc1 * w
    // 1. first cut by w to get more nc1
    // 2. second cut h
    GetTilingParamForResizeNearestNeighborV2GradDefault(compile_params, tiling_params);
  }
  tiling_params.tiling_key = tiling_params.tiling_key + h_tiling_align_flag * HEIGHT_ALIGN_FLAG +
                             w_tiling_align_flag * WEIGHT_ALIGN_FLAG + w_align_flag * WIDTH_ALIGN_FLAG +
                             is_big_to_small_flag * BIG_TO_SMALL_FLAG;
  return true;
}

static void PrintTilingParams(const std::string& op_type, const ResizeNearestNeighborV2TilingParams& tiling_params,
                              const ResizeNearestNeighborV2CompileParams& compile_params) {
  // print tiling_params
  OP_LOGD(op_type, "tiling_data, tiling_key = %d.", tiling_params.tiling_key);
  OP_LOGD(op_type, "tiling_data, input_batch_c1 = %d.", tiling_params.input_batch * tiling_params.input_c1);
  OP_LOGD(op_type, "tiling_data, input_height = %d.", tiling_params.input_height);
  OP_LOGD(op_type, "tiling_data, input_weight = %d.", tiling_params.input_weight);
  OP_LOGD(op_type, "tiling_data, output_height = %d.", tiling_params.output_height);
  OP_LOGD(op_type, "tiling_data, output_weight = %d.", tiling_params.output_weight);
  OP_LOGD(op_type, "tiling_data, cut_batch_c1_num = %d.", tiling_params.cut_batch_c1_num);
  OP_LOGD(op_type, "tiling_data, cut_height_num = %d.", tiling_params.cut_height_num);
  OP_LOGD(op_type, "tiling_data, cut_weight_num = %d.", tiling_params.cut_weight_num);

  // print compile_params
  OP_LOGD(op_type, "compile_data, core_num = %d.", compile_params.core_num);
  OP_LOGD(op_type, "compile_data, max_w_len = %d.", compile_params.max_w_len);
  OP_LOGD(op_type, "compile_data, align_corners = %d.", compile_params.align_corners);
  OP_LOGD(op_type, "compile_data, half_pixel_centers = %d.", compile_params.half_pixel_centers);
}

void SetTilingParams(const ResizeNearestNeighborV2TilingParams& tiling_params, OpRunInfo& run_info) {
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
}

static bool ResizeNearestNeighborV2Tiling(const std::string& op_type, const TeOpParas& op_paras,
                                          const nlohmann::json& op_info, OpRunInfo& run_info) {
  using namespace ge;
  OP_LOGI(op_type, "tiling run begin.");

  if (op_paras.inputs.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Length of inputs is empty.");
    return false;
  }
  if (op_paras.outputs.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Length of outputs is empty.");
    return false;
  }
  // get input_shape and output_shape
  const std::vector<int64_t>& input_shape = op_paras.inputs[0].tensor[0].shape;
  const std::vector<int64_t>& output_shape = op_paras.outputs[0].tensor[0].shape;
  if (input_shape.size() != 5) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "the input shape size must be 5(NC1HWC0).");
    return false;
  }
  if (output_shape.size() != 5) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "the output shape size must be 5(NC1HWC0).");
    return false;
  }

  // get compile data begin
  ResizeNearestNeighborV2CompileParams compile_params;
  // init compile data
  compile_params.core_num = 0;
  compile_params.max_w_len = 0;
  compile_params.align_corners = 0;
  compile_params.half_pixel_centers = 0;
  compile_params.op_type = op_type;
  // get compile data
  if (!GetResizeNearestNeighborV2CompileParams(op_info, compile_params)) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile info from nlohmann json failed.");
    return false;
  }
  // get compile data end

  // get tiling data begin
  ResizeNearestNeighborV2TilingParams tiling_params;
  // init tiling data
  tiling_params.tiling_key = DEFAULT_TILING_MODE;
  tiling_params.input_batch = input_shape[0];
  tiling_params.input_c1 = input_shape[1];
  tiling_params.input_height = input_shape[2];
  tiling_params.output_height = output_shape[2];
  tiling_params.input_weight = input_shape[3];
  tiling_params.output_weight = output_shape[3];
  tiling_params.cut_batch_c1_num = 1;
  tiling_params.cut_height_num = 1;
  tiling_params.cut_weight_num = 1;

  // calcu tiling
  bool get_tiling_result = false;
  if (op_type == "ResizeNearestNeighborV2" || op_type == "ResizeBilinearV2") {
    get_tiling_result = GetTilingParam(compile_params, tiling_params);
  } else {
    get_tiling_result = GetTilingParamResizeNearestNeighborV2Grad(compile_params, tiling_params);
  }
  if (!get_tiling_result) {
    PrintTilingParams(op_type, tiling_params, compile_params);
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get tiling data failed.");
    return false;
  }

  // get tiling data end
  PrintTilingParams(op_type, tiling_params, compile_params);
  SetTilingParams(tiling_params, run_info);
  run_info.block_dim = compile_params.core_num;
  std::vector<int64_t> workspace;
  run_info.workspaces = workspace;
  OP_LOGI(op_type, "tiling run success.");

  return true;
}

// register tiling interface of the ResizeNearestNeighborV2 op.
REGISTER_OP_TILING_FUNC_BUFFERED(ResizeNearestNeighborV2, ResizeNearestNeighborV2Tiling);
// register tiling interface of the ResizeBilinearV2 op.
REGISTER_OP_TILING_FUNC_BUFFERED(ResizeBilinearV2, ResizeNearestNeighborV2Tiling);
// register tiling interface of the ResizeNearestNeighborV2Grad op.
REGISTER_OP_TILING_FUNC_BUFFERED(ResizeNearestNeighborV2Grad, ResizeNearestNeighborV2Tiling);
}  // namespace optiling
