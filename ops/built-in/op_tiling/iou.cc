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

/*!
 * \file bounding_box_decode.cc
 * \brief dynamic shape tiling of bounding_box_decode
 */
#include<string>
#include <cmath>
#include <map>
#include <nlohmann/json.hpp>
#include "op_log.h"
#include "op_tiling.h"
#include "error_log.h"

using namespace ge;
using namespace std;
namespace optiling{
const int32_t TILING_MODE_1 = 1;
const int32_t TILING_MODE_2 = 2;

static const int32_t GTBOX_SEGMENT = 4096 * 4;
static const int32_t BBOX_SEGMENT = 4096 * 4;
struct IouTilingParams
{
  int32_t tiling_mode;
  int32_t point_per_core;
  int32_t core_tail_num;
  int32_t core_num;
  int32_t area_x0_size;

  int32_t bboxes_num;
  int32_t gtboxes_num;

  int32_t bb_loop;
  int32_t bb_tail;
  int32_t bb_tail_offset;
};

void InitTilingParams(IouTilingParams& tiling_params) {
  tiling_params.tiling_mode = 0;
  tiling_params.point_per_core = 0;
  tiling_params.core_tail_num = 0;
  tiling_params.core_num = 0;
  tiling_params.area_x0_size = 0;
  tiling_params.bboxes_num = 0;
  tiling_params.gtboxes_num = 0;
  tiling_params.bb_loop = 0;
  tiling_params.bb_tail = 0;
  tiling_params.bb_tail_offset = 0;
}

void PrintTilingParams(const IouTilingParams& tiling_params) {
  GELOGD("Op Tiling [Iou]: tiling_mode=%d.", tiling_params.tiling_mode);
  GELOGD("Op Tiling [Iou]: point_per_core=%d.", tiling_params.point_per_core);
  GELOGD("Op Tiling [Iou]: core_tail_num=%d.", tiling_params.core_tail_num);
  GELOGD("Op Tiling [Iou]: core_num=%d.", tiling_params.core_num);
  GELOGD("Op Tiling [Iou]: area_x0_size=%d.", tiling_params.area_x0_size);
  GELOGD("Op Tiling [Iou]: bboxes_num=%d.", tiling_params.bboxes_num);
  GELOGD("Op Tiling [Iou]: gtboxes_num=%d.", tiling_params.gtboxes_num);
  GELOGD("Op Tiling [Iou]: bboxes_num=%d.", tiling_params.bb_loop);
  GELOGD("Op Tiling [Iou]: gtboxes_num=%d.", tiling_params.bb_tail);
  GELOGD("Op Tiling [Iou]: gtboxes_num=%d.", tiling_params.bb_tail_offset);
}

void SetRunningInfo(const IouTilingParams& tiling_params, OpRunInfo& run_info) {
  ByteBufferPut(run_info.tiling_data, tiling_params.tiling_mode);
  ByteBufferPut(run_info.tiling_data, tiling_params.point_per_core);
  ByteBufferPut(run_info.tiling_data, tiling_params.core_tail_num);
  ByteBufferPut(run_info.tiling_data, tiling_params.core_num);
  ByteBufferPut(run_info.tiling_data, tiling_params.area_x0_size);
  ByteBufferPut(run_info.tiling_data, tiling_params.bboxes_num);
  ByteBufferPut(run_info.tiling_data, tiling_params.gtboxes_num);
  ByteBufferPut(run_info.tiling_data, tiling_params.bb_loop);
  ByteBufferPut(run_info.tiling_data, tiling_params.bb_tail);
  ByteBufferPut(run_info.tiling_data, tiling_params.bb_tail_offset);
}

int32_t GetCeilInt(int32_t num1, int32_t num2) {
  return (num1 + num2 - 1) / num2;
}

int32_t GetAlignInt32(int32_t origin_num, int32_t align_num) {
  return GetCeilInt(origin_num, align_num) * align_num;
}

int32_t GetAreaUbSize(int32_t bb_ub_segment_point, int32_t max_eliments) {
  return GetAlignInt32(bb_ub_segment_point, max_eliments);
}

int32_t GetGtAreaUbSize(int32_t gt_ub_segment_point, int32_t max_eliments) {
  return GetAlignInt32(gt_ub_segment_point, max_eliments);
}

int32_t GetAreaX0Size(int32_t area_ub_size, int32_t gt_area_ub_size) {
  return area_ub_size > gt_area_ub_size ? area_ub_size : gt_area_ub_size;
}

int32_t GetGtBoxUbSegment(string& dtype) {
  if (dtype == "float16") {
    return GTBOX_SEGMENT;
  } else {
    return GTBOX_SEGMENT / 2;
  }
}

int32_t GetBBoxUbSegment(string& dtype, bool product) {
  int32_t bbox_ub_segment = BBOX_SEGMENT;
  if (dtype == "float32") {
    bbox_ub_segment /= 2;
  }
  if (!product) {
    bbox_ub_segment /= 2;
  }
  return bbox_ub_segment;
}

int32_t GetMaxEliments(string& dtype) {
  if (dtype == "float16") {
    return 16 * 8;
  } else {
    return 8 * 8;
  }
}

int32_t GetBBLoop(int32_t bboxes_num, int32_t bbox_ub_segment) {
  return bboxes_num * 4 / bbox_ub_segment;
}

int32_t GetBBTail(int32_t bboxes_num, int32_t bbox_ub_segment) {
  return bboxes_num * 4 % bbox_ub_segment;
}

int32_t GetBBTailOffset(int32_t bb_loop, int32_t& bb_tail, int32_t bbox_ub_segment, int32_t min_point_per_core) {
  int32_t min_segment = min_point_per_core * 4;
  if (0 < bb_tail && bb_tail < min_segment && bb_loop != 0) {
    int32_t bb_tail_offset = bb_loop * bbox_ub_segment + bb_tail - min_segment;
    bb_tail = min_segment;
    return bb_tail_offset;
  } else if (bb_tail % min_segment != 0 && bb_loop != 0) {
    int32_t bb_tail_offset = bb_loop * bbox_ub_segment + (bb_tail % min_segment) - min_segment;
    bb_tail = GetAlignInt32(bb_tail, min_segment);
    return bb_tail_offset;
  } else {
    int32_t bb_tail_offset = bb_loop * bbox_ub_segment;
    return bb_tail_offset;
  }
}

int32_t GetElimentsPerBlock(string& dtype) {
  if (dtype == "float16") {
    return 16;
  } else {
    return 8;
  }
}

int32_t GetPointPerCoreMode2(int32_t gtboxes_num, int32_t bboxes_num, int32_t full_core_num,
                             int32_t eliments_per_block, int32_t min_point_per_core) {
  int32_t core_num;
  if (bboxes_num < eliments_per_block) {
    core_num = 1;
  } else {
    core_num = full_core_num;
  }

  int32_t point_per_core = GetCeilInt(gtboxes_num, core_num);
  if (bboxes_num < min_point_per_core) {
    if (point_per_core < min_point_per_core) {
      point_per_core = min_point_per_core;
    }
    point_per_core = GetAlignInt32(point_per_core, min_point_per_core);
  }
  return point_per_core;
}

int32_t GetCoreTailNumMode2(int32_t gtboxes_num, int32_t point_per_core) {
  return gtboxes_num % point_per_core;
}

int32_t GetCoreNumMode2(int32_t gtboxes_num, int32_t point_per_core) {
  return GetCeilInt(gtboxes_num, point_per_core);
}

int32_t GetPointPerCoreMode1(int32_t bboxes_num, int32_t full_core_num, int32_t min_point_per_core) {
  int32_t point_per_core = GetCeilInt(bboxes_num, full_core_num);
  if (point_per_core < min_point_per_core) {
    return min_point_per_core;
  } else {
    return GetAlignInt32(point_per_core, min_point_per_core);
  }
}

int32_t GetCoreTailNumMode1(int32_t bboxes_num, int32_t point_per_core) {
  return bboxes_num % point_per_core;
}

int32_t GetCoreNumMode1(int32_t bboxes_num, int32_t point_per_core) {
  return GetCeilInt(bboxes_num, point_per_core);
}

int32_t GetTilingMode(int32_t gtboxes_num) {
  if (gtboxes_num * 4 <= GTBOX_SEGMENT) {
    return TILING_MODE_1;
  } else {
    return TILING_MODE_2;
  }
}

int32_t GetMinPointPerCore(string& dtype) {
  if (dtype == "float16") {
    return 16;
  } else {
    return 8;
  }
}

void CalRunningInfo(IouTilingParams& tiling_params, int32_t full_core_num, const vector<int64_t>& bboxes_shape,
                    const vector<int64_t>& gtboxes_shape, string dtype, bool product) {
  int32_t bboxes_num = bboxes_shape[0];
  int32_t gtboxes_num = gtboxes_shape[0];
  int32_t min_point_per_core = GetMinPointPerCore(dtype);
  int32_t tiling_mode = GetTilingMode(gtboxes_num);
  int32_t bb_ub_segment_point = GetBBoxUbSegment(dtype, product) / 4;
  int32_t gt_ub_segment_point = GetBBoxUbSegment(dtype, product) / 4;
  int32_t max_eliments = GetMaxEliments(dtype);
  int32_t area_ub_size = GetAreaUbSize(bb_ub_segment_point, max_eliments);
  int32_t gt_area_ub_size = GetGtAreaUbSize(gt_ub_segment_point, max_eliments);
  int32_t area_x0_size = GetAreaX0Size(area_ub_size, gt_area_ub_size);
  int32_t point_per_core, core_tail_num, core_num;
  int32_t bb_ub_segment, bb_loop, bb_tail;
  if (tiling_mode == TILING_MODE_1) {
    point_per_core = GetPointPerCoreMode1(bboxes_num, full_core_num, min_point_per_core);
    core_tail_num = GetCoreTailNumMode1(bboxes_num, point_per_core);
    core_num = GetCoreNumMode1(bboxes_num, point_per_core);
    bb_ub_segment = GetBBoxUbSegment(dtype, product);
    bb_loop = GetBBLoop(point_per_core, bb_ub_segment);
    bb_tail = GetBBTail(point_per_core, bb_ub_segment);
  } else {
    int32_t eliments_per_block = GetElimentsPerBlock(dtype);
    point_per_core = GetPointPerCoreMode2(gtboxes_num, bboxes_num,
                                          full_core_num, eliments_per_block, min_point_per_core);
    core_tail_num = GetCoreTailNumMode2(gtboxes_num, point_per_core);
    core_num = GetCoreNumMode2(gtboxes_num, point_per_core);
    bb_ub_segment = GetBBoxUbSegment(dtype, product);
    bb_loop = GetBBLoop(bboxes_num, bb_ub_segment);
    bb_tail = GetBBTail(bboxes_num, bb_ub_segment);
  }
  int32_t bb_tail_offset = GetBBTailOffset(bb_loop, bb_tail, bb_ub_segment, min_point_per_core);

  tiling_params.tiling_mode = tiling_mode;
  tiling_params.point_per_core = point_per_core;
  tiling_params.core_tail_num = core_tail_num;
  tiling_params.core_num = core_num;
  tiling_params.area_x0_size = area_x0_size;

  tiling_params.bboxes_num = bboxes_num;
  tiling_params.gtboxes_num = gtboxes_num;

  tiling_params.bb_loop = bb_loop;
  tiling_params.bb_tail = bb_tail;
  tiling_params.bb_tail_offset = bb_tail_offset;
}

bool CalCompileInfo(const string& op_type, const nlohmann::json& op_info,
                    int32_t& full_core_num, bool& product) {
  using namespace nlohmann;
  auto all_vars = op_info["vars"];
  if (all_vars.count("full_core_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "IouTiling: GetCompileInfo, get full_core_num error.");
    return false;
  }
  if (all_vars.count("product") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "IouTiling: GetCompileInfo, get product error.");
    return false;
  }
  full_core_num = all_vars["full_core_num"].get<int32_t>();
  product = all_vars["product"].get<bool>();
  return true;
}

bool IouTiling(const string& op_type, const TeOpParas& op_paras, 
    const nlohmann::json& op_info, OpRunInfo& run_info) {
  OP_LOGD(op_type.c_str(), "IouTiling running.");

  IouTilingParams tiling_params;
  InitTilingParams(tiling_params);
  int32_t full_core_num;
  bool product = false;
  bool get_compile_info = CalCompileInfo(op_type, op_info, full_core_num, product);
  if (!get_compile_info) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "IouTiling: GetCompileInfo error.");
    return false;
  }

  const vector<int64_t>& bboxes_shape = op_paras.inputs[0].tensor[0].shape;
  const string dtype = op_paras.inputs[0].tensor[0].dtype;
  const vector<int64_t>& gtboxes_shape = op_paras.inputs[1].tensor[0].shape;

  CalRunningInfo(tiling_params, full_core_num, bboxes_shape, gtboxes_shape, dtype, product);

  SetRunningInfo(tiling_params, run_info);
  PrintTilingParams(tiling_params);
  run_info.block_dim = tiling_params.core_num;

  OP_LOGD(op_type.c_str(), "IouTiling end.");
  return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(Iou, IouTiling);
}  // namespace optiling
