/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file image_projective_transform.cc
 * \brief dynamic shape tiling of image_projective_transform
 */
#include <string>
#include <math.h>
#include <nlohmann/json.hpp>
#include "op_tiling_util.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "error_log.h"
#include "graph/utils/op_desc_utils.h"

namespace optiling {
using namespace ge;
using namespace std;

struct ImageProjectiveTransformParams {
  int32_t tiling_mode;
  int32_t act_core_num;
  int32_t input_b;
  int32_t input_h;
  int32_t input_w;
  int32_t input_c;
  int32_t input_size;
  int32_t output_h;
  int32_t output_w;
  int32_t ub_height;
  int32_t ub_repeat_time;
  int32_t ub_repeat_left;
  int32_t imgnum_cal_repeat;
  int32_t imgnum_cal_ceil_repeat;
  int32_t imgnum_cal_left;
};

struct CompileINfoParams {
  // get Compile info
  int32_t ub_ele = 0;
  int32_t core_num;
  int32_t trans_dtype_size;
  int32_t block_byte_size;
};

const int64_t TILING_MODE_0 = 0;
const int64_t TILING_MODE_1 = 1;
const int64_t TILING_MODE_2 = 2;
const int64_t TILING_MODE_3 = 3;
const int64_t TILING_MODE_4 = 4;
const int64_t TILING_MODE_5 = 5;
const int64_t TILING_MODE_6 = 6;
const int64_t TILING_MODE_7 = 7;
const int64_t TILING_MODE_8 = 8;
const int64_t TILING_MODE_9 = 9;
const int64_t IMAGE_INDEX_0 = 0;
const int64_t IMAGE_INDEX_1 = 1;
const int64_t IMAGE_INDEX_2 = 2;
const int64_t IMAGE_INDEX_3 = 3;
const int64_t OUTPUT_SHAPE_SIZE = 2;
const int64_t OUTPUT_SHAPE_H = 0;
const int64_t OUTPUT_SHAPE_W = 1;
const int64_t COMPILE_INFO_0 = 0;
const int64_t COMPILE_INFO_1 = 1;
const int64_t COMPILE_INFO_2 = 2;
const int64_t COMPILE_INFO_3 = 3;

static const std::vector<std::string> COMPILE_INFO_KEY = {"ub_ele", "core_num", "trans_dtype_size", "block_byte_size"};

void InitTilingParams(ImageProjectiveTransformParams& params) {
  params.tiling_mode = 0;
  params.act_core_num = 0;
  params.input_b = 0;
  params.input_h = 0;
  params.input_w = 0;
  params.input_c = 0;
  params.input_size = 0;
  params.output_h = 0;
  params.output_w = 0;
  params.ub_height = 0;
  params.ub_repeat_time = 0;
  params.ub_repeat_left = 0;
  params.imgnum_cal_repeat = 0;
  params.imgnum_cal_ceil_repeat = 0;
  params.imgnum_cal_left = 0;
}

bool GetCompileInfo(const std::string& op_type, const std::vector<int64_t>& op_compile_info,
                    CompileINfoParams& compile_info_param) {
  OP_TILING_CHECK(COMPILE_INFO_KEY.size() != op_compile_info.size(),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "parse op_compile_info failed."), return false);

  compile_info_param.ub_ele = static_cast<int32_t>(op_compile_info[COMPILE_INFO_0]);
  compile_info_param.core_num = static_cast<int32_t>(op_compile_info[COMPILE_INFO_1]);
  compile_info_param.trans_dtype_size = static_cast<int32_t>(op_compile_info[COMPILE_INFO_2]);
  compile_info_param.block_byte_size = static_cast<int32_t>(op_compile_info[COMPILE_INFO_3]);

  return true;
}

int32_t CalTilingMode(const ImageProjectiveTransformParams& tiling_params, const GeShape& images_shape,
                      const GeShape& transform_shape, const CompileINfoParams& compile_info_param,
                      const ge::DataType& dtype) {
  int32_t tiling_mode = TILING_MODE_4;
  int32_t transform_b = transform_shape.GetDim(IMAGE_INDEX_0);
  int32_t images_b = images_shape.GetDim(IMAGE_INDEX_0);
  int32_t input_c = images_shape.GetDim(IMAGE_INDEX_3);
  int32_t ub_ele = compile_info_param.ub_ele;
  int32_t output_hw = tiling_params.output_h * tiling_params.output_w;
  int32_t input_hw = tiling_params.input_h * tiling_params.input_w;
  int32_t input_size = input_hw * input_c;
  int32_t output_size = output_hw * input_c;
  int32_t block_size = 0;
  int64_t data_block = GetDataBlockElems(dtype);
  block_size = static_cast<int32_t>(data_block);

  if (images_b <= compile_info_param.core_num && input_size < ub_ele && output_hw < ub_ele &&
      output_size > block_size) {
    tiling_mode = transform_b == 1 ? TILING_MODE_0 : TILING_MODE_1;
  }

  else if (images_b > compile_info_param.core_num && input_size < ub_ele && output_hw < ub_ele &&
           output_size > block_size) {
    tiling_mode = transform_b == 1 ? TILING_MODE_2 : TILING_MODE_3;
  }

  else if (images_b <= compile_info_param.core_num && (input_size > ub_ele || output_hw > ub_ele) &&
           output_size > block_size) {
    tiling_mode = transform_b == 1 ? TILING_MODE_4 : TILING_MODE_5;
  }

  else if (images_b > compile_info_param.core_num && (input_size > ub_ele || output_hw > ub_ele) &&
           output_size > block_size) {
    tiling_mode = transform_b == 1 ? TILING_MODE_6 : TILING_MODE_7;
  }

  else if (output_size < block_size && input_size < ub_ele) {
    tiling_mode = transform_b == 1 ? TILING_MODE_8 : TILING_MODE_9;
  }

  return tiling_mode;
}

void CalRunningInfo(ImageProjectiveTransformParams& tiling_params, const CompileINfoParams& compile_info_param,
                    const GeShape& images_shape, const GeShape& transform_shape, const ge::DataType& dtype) {
  int32_t input_h = images_shape.GetDim(IMAGE_INDEX_1);
  int32_t input_b = images_shape.GetDim(IMAGE_INDEX_0);
  int32_t input_w = images_shape.GetDim(IMAGE_INDEX_2);
  int32_t input_c = images_shape.GetDim(IMAGE_INDEX_3);
  int32_t input_size = input_h * input_w * input_c;

  int32_t imgnum_cal_repeat = 0;
  int32_t imgnum_cal_ceil_repeat = 0;
  int32_t imgnum_cal_left = 0;
  imgnum_cal_repeat = input_b / compile_info_param.core_num;
  imgnum_cal_ceil_repeat = imgnum_cal_repeat + 1;
  imgnum_cal_left = input_b % compile_info_param.core_num;
  tiling_params.imgnum_cal_repeat = imgnum_cal_repeat;
  tiling_params.imgnum_cal_ceil_repeat = imgnum_cal_ceil_repeat;
  tiling_params.imgnum_cal_left = imgnum_cal_left;

  int32_t ub_height = 0;
  int32_t ub_repeat_time = 0;
  int32_t ub_repeat_left = 0;
  int32_t ub_ele = compile_info_param.ub_ele;
  int32_t output_h = tiling_params.output_h;
  int32_t output_w = tiling_params.output_w;
  ub_height = ub_ele / output_w;
  ub_repeat_time = output_h / ub_height;
  ub_repeat_left = output_h % ub_height;
  tiling_params.ub_height = ub_height;
  tiling_params.ub_repeat_time = ub_repeat_time;
  tiling_params.ub_repeat_left = ub_repeat_left;

  int32_t act_core_num = 0;
  if (input_b > compile_info_param.core_num) {
    act_core_num = compile_info_param.core_num;
  } else {
    act_core_num = input_b;
  }
  tiling_params.act_core_num = act_core_num;

  tiling_params.input_h = input_h;
  tiling_params.input_b = input_b;
  tiling_params.input_w = input_w;
  tiling_params.input_c = input_c;
  tiling_params.input_size = input_size;
  tiling_params.tiling_mode = CalTilingMode(tiling_params, images_shape, transform_shape, compile_info_param, dtype);
}

void SetRunningInfo(const ImageProjectiveTransformParams& tiling_params, utils::OpRunInfo& run_info) {
  run_info.AddTilingData(tiling_params.tiling_mode);
  run_info.AddTilingData(tiling_params.act_core_num);
  run_info.AddTilingData(tiling_params.input_b);
  run_info.AddTilingData(tiling_params.input_h);
  run_info.AddTilingData(tiling_params.input_w);
  run_info.AddTilingData(tiling_params.input_c);
  run_info.AddTilingData(tiling_params.input_size);
  run_info.AddTilingData(tiling_params.output_h);
  run_info.AddTilingData(tiling_params.output_w);
  run_info.AddTilingData(tiling_params.ub_height);
  run_info.AddTilingData(tiling_params.ub_repeat_time);
  run_info.AddTilingData(tiling_params.ub_repeat_left);
  run_info.AddTilingData(tiling_params.imgnum_cal_repeat);
  run_info.AddTilingData(tiling_params.imgnum_cal_ceil_repeat);
  run_info.AddTilingData(tiling_params.imgnum_cal_left);
}

void PrintTilingParams(const ImageProjectiveTransformParams& tiling_params) {
  OP_LOGD("ImageProjectiveTransform", "[ImageProjectiveTransformTiling] : tiling_mode=%ld.",
          tiling_params.tiling_mode);
  OP_LOGD("ImageProjectiveTransform", "[ImageProjectiveTransformTiling] : act_core_num=%ld.",
          tiling_params.act_core_num);
  OP_LOGD("ImageProjectiveTransform", "[ImageProjectiveTransformTiling] : input_b=%ld.", tiling_params.input_b);
  OP_LOGD("ImageProjectiveTransform", "[ImageProjectiveTransformTiling] : input_h=%ld.", tiling_params.input_h);
  OP_LOGD("ImageProjectiveTransform", "[ImageProjectiveTransformTiling] : input_b=%ld.", tiling_params.input_w);
  OP_LOGD("ImageProjectiveTransform", "[ImageProjectiveTransformTiling] : input_h=%ld.", tiling_params.input_c);
  OP_LOGD("ImageProjectiveTransform", "[ImageProjectiveTransformTiling] : input_size=%ld.", tiling_params.input_size);
  OP_LOGD("ImageProjectiveTransform", "[ImageProjectiveTransformTiling] : input_b=%ld.", tiling_params.output_h);
  OP_LOGD("ImageProjectiveTransform", "[ImageProjectiveTransformTiling] : input_h=%ld.", tiling_params.output_w);
  OP_LOGD("ImageProjectiveTransform", "[ImageProjectiveTransformTiling] : ub_height=%ld.", tiling_params.ub_height);
  OP_LOGD("ImageProjectiveTransform", "[ImageProjectiveTransformTiling] : ub_repeat_time=%ld.",
          tiling_params.ub_repeat_time);
  OP_LOGD("ImageProjectiveTransform", "[ImageProjectiveTransformTiling] : ub_repeat_left=%ld.",
          tiling_params.ub_repeat_left);
  OP_LOGD("ImageProjectiveTransform", "[ImageProjectiveTransformTiling] : imgnum_cal_repeat=%ld.",
          tiling_params.imgnum_cal_repeat);
  OP_LOGD("ImageProjectiveTransform", "[ImageProjectiveTransformTiling] : imgnum_cal_ceil_repeat=%ld.",
          tiling_params.imgnum_cal_ceil_repeat);
  OP_LOGD("ImageProjectiveTransform", "[ImageProjectiveTransformTiling] : imgnum_cal_left=%ld.",
          tiling_params.imgnum_cal_left);
}

bool ImageProjectiveTransformTiling(const std::string& op_type, const ge::Operator& opParas,
                                    const std::vector<int64_t>& op_compile_info, utils::OpRunInfo& run_info) {
  OP_LOGI(op_type.c_str(), "ImageProjectiveTransformTiling running.");

  CompileINfoParams tiling_params;
  OP_TILING_CHECK(!GetCompileInfo(op_type, op_compile_info, tiling_params),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetCompileInfo error."), return false);

  auto operator_info = OpDescUtils::GetOpDescFromOperator(opParas);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get op_info failed."),
                  return false);

  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_images failed."),
                  return false);
  const GeShape& images_shape = input_desc->MutableShape();
  ge::DataType image_dtype = input_desc->GetDataType();

  input_desc = operator_info->MutableInputDesc(1);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_transforms failed."),
                  return false);
  const GeShape& transform_shape = input_desc->MutableShape();

  ImageProjectiveTransformParams params;
  InitTilingParams(params);

  vector<int64_t> output_data;
  if (!ops::GetConstIntData(opParas, OUTPUT_SHAPE_SIZE, output_data)) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get const stride failed!");
    return false;
  } else if (output_data.size() != OUTPUT_SHAPE_SIZE) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "the dimension count should be 2! size = %ld", output_data.size());
    return false;
  } else {
    params.output_h = static_cast<int32_t>(output_data[OUTPUT_SHAPE_H]);
    params.output_w = static_cast<int32_t>(output_data[OUTPUT_SHAPE_W]);
  }

  CalRunningInfo(params, tiling_params, images_shape, transform_shape, image_dtype);
  SetRunningInfo(params, run_info);
  PrintTilingParams(params);

  run_info.SetBlockDim(params.act_core_num);
  OP_LOGI(op_type.c_str(), "ImageProjectiveTransformTiling run success.");
  return true;
}

// register tiling interface of the ResizeBilinearV2Grad op.
REGISTER_OP_TILING_V3_WITH_VECTOR(ImageProjectiveTransform, ImageProjectiveTransformTiling, COMPILE_INFO_KEY,
                                  NO_OPTIONAL_VALUE);
REGISTER_OP_TILING_V3_WITH_VECTOR(ImageProjectiveTransformV2, ImageProjectiveTransformTiling, COMPILE_INFO_KEY,
                                  NO_OPTIONAL_VALUE);
}  // namespace optiling
