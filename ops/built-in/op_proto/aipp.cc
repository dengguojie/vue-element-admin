/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file aipp.cpp
 * \brief
 */
#include "inc/aipp.h"
#include "graph/operator_reg.h"
#include "util/util.h"
#include "op_log.h"
#include "proto/insert_op.pb.h"
#include <nlohmann/json.hpp>

#include "graph/utils/graph_utils.h"
#include "./util/error_util.h"
#include "graph/utils/type_utils.h"
#include "register/infer_data_slice_registry.h"
#include "graph/common_error_codes.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {

void GetOutputHeightWidth(::domi::AippOpParams* aipp_op_params, uint64_t* output_height, uint64_t* output_width) {
  if (aipp_op_params->crop()) {
    *output_height = aipp_op_params->crop_size_h() ? aipp_op_params->crop_size_h() : *output_height;
    *output_width = aipp_op_params->crop_size_w() ? aipp_op_params->crop_size_w() : *output_width;
  }

  if (aipp_op_params->resize()) {
    *output_height = aipp_op_params->resize_output_h() ? aipp_op_params->resize_output_h() : *output_height;
    *output_width = aipp_op_params->resize_output_w() ? aipp_op_params->resize_output_w() : *output_width;
  }

  if (!(aipp_op_params->crop()) && !(aipp_op_params->resize())) {
    *output_height = aipp_op_params->src_image_size_h() ? aipp_op_params->src_image_size_h() : *output_height;
    *output_width = aipp_op_params->src_image_size_w() ? aipp_op_params->src_image_size_w() : *output_width;
  }

  if (aipp_op_params->padding()) {
    uint64_t left_padding_size = aipp_op_params->left_padding_size() ? aipp_op_params->left_padding_size() : 0;
    uint64_t right_padding_size = aipp_op_params->right_padding_size() ? aipp_op_params->right_padding_size() : 0;
    uint64_t top_padding_size = aipp_op_params->top_padding_size() ? aipp_op_params->top_padding_size() : 0;
    uint64_t bottom_padding_size = aipp_op_params->bottom_padding_size() ? aipp_op_params->bottom_padding_size() : 0;

    *output_height = *output_height + top_padding_size + bottom_padding_size;
    *output_width = *output_width + left_padding_size + right_padding_size;
  }
}

uint64_t GetSrcImageSizeDtype(::domi::AippOpParams* aipp_op_params, uint64_t batch, uint64_t c1, uint64_t height,
                              uint64_t width, ge::DataType* src_img_dtype) {
  uint64_t size = 0;
  uint64_t src_image_size_h = aipp_op_params->src_image_size_h() ? aipp_op_params->src_image_size_h() : height;
  uint64_t src_image_size_w = aipp_op_params->src_image_size_w() ? aipp_op_params->src_image_size_w() : width;

  if (aipp_op_params->input_format()) {
    if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_YUV420SP_U8) {
      size = batch * 3 * src_image_size_h * src_image_size_w / 2;
      *src_img_dtype = DT_UINT8;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_XRGB8888_U8) {
      size = batch * 4 * src_image_size_h * src_image_size_w;
      *src_img_dtype = DT_UINT8;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RGB888_U8) {
      size = batch * 3 * src_image_size_h * src_image_size_w;
      *src_img_dtype = DT_UINT8;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_YUV400_U8) {
      size = batch * src_image_size_h * src_image_size_w;
      *src_img_dtype = DT_UINT8;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_NC1HWC0DI_FP16) {
      size = batch * c1 * src_image_size_h * src_image_size_w * 4 * 2;
      *src_img_dtype = DT_FLOAT16;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_NC1HWC0DI_S8) {
      size = batch * c1 * src_image_size_h * src_image_size_w * 4;
      *src_img_dtype = DT_INT8;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_ARGB8888_U8) {
      size = batch * 4 * src_image_size_h * src_image_size_w;
      *src_img_dtype = DT_UINT8;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_YUYV_U8) {
      size = batch * 2 * src_image_size_h * src_image_size_w;
      *src_img_dtype = DT_UINT8;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_YUV422SP_U8) {
      size = batch * 2 * src_image_size_h * src_image_size_w;
      *src_img_dtype = DT_UINT8;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_AYUV444_U8) {
      size = batch * 4 * src_image_size_h * src_image_size_w;
      *src_img_dtype = DT_UINT8;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RAW10) {
      size = batch * src_image_size_h * src_image_size_w * 2;
      *src_img_dtype = DT_UINT16;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RAW12) {
      size = batch * src_image_size_h * src_image_size_w * 2;
      *src_img_dtype = DT_UINT16;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RAW16) {
      size = batch * src_image_size_h * src_image_size_w * 2;
      *src_img_dtype = DT_UINT16;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RAW24) {
      size = batch * src_image_size_h * src_image_size_w * 4;
      *src_img_dtype = DT_UINT32;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RGB16) {
      size = batch * src_image_size_h * src_image_size_w * 3;
      *src_img_dtype = DT_UINT16;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RGB20) {
      size = batch * src_image_size_h * src_image_size_w * 3;
      *src_img_dtype = DT_UINT32;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RGB24) {
      size = batch * src_image_size_h * src_image_size_w * 3;
      *src_img_dtype = DT_UINT32;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RGB8_IR) {
      size = batch * src_image_size_h * src_image_size_w * 4;
      *src_img_dtype = DT_UINT8;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RGB16_IR) {
      size = batch * src_image_size_h * src_image_size_w * 4;
      *src_img_dtype = DT_UINT16;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RGB24_IR) {
      size = batch * src_image_size_h * src_image_size_w * 4;
      *src_img_dtype = DT_UINT32;
    } else {
      OP_LOGE("Aipp", "Input format of AIPP conf is undefined!");
    }
  }

  return size;
}

std::vector<int32_t> GetAclInputDims(::domi::AippOpParams* aipp_op_params, uint64_t batch, uint64_t srcImageHeight,
                                     uint64_t srcImageWidth) {
  uint64_t channel = 3;
  uint64_t height = srcImageHeight;
  uint64_t width = srcImageWidth;
  std::vector<int32_t> aclInputDims;

  if (aipp_op_params->input_format()) {
    if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_YUV420SP_U8) {
      height = srcImageHeight * 3 / 2;
      channel = 1;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_XRGB8888_U8) {
      channel = 4;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RGB888_U8) {
      channel = 3;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_YUV400_U8) {
      channel = 1;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_ARGB8888_U8) {
      channel = 4;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_YUYV_U8) {
      channel = 2;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_YUV422SP_U8) {
      height = srcImageHeight * 2;
      channel = 1;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_AYUV444_U8) {
      channel = 4;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RAW10) {
      channel = 1;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RAW12) {
      channel = 1;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RAW16) {
      channel = 1;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RAW24) {
      channel = 1;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RGB16) {
      channel = 3;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RGB20) {
      channel = 3;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RGB24) {
      channel = 3;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RGB8_IR) {
      channel = 4;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RGB16_IR) {
      channel = 4;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RGB24_IR) {
      channel = 4;
    } else {
      OP_LOGE("Aipp", "Input format of AIPP conf is undefined!");
    }
  }

  aclInputDims.push_back(batch);
  aclInputDims.push_back(height);
  aclInputDims.push_back(width);
  aclInputDims.push_back(channel);

  return aclInputDims;
}

uint64_t GetChannel(::domi::AippOpParams* aipp_op_params) {
  if (aipp_op_params->input_format()) {
    if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_XRGB8888_U8) {
      return 4;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_YUV400_U8) {
      return 1;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_ARGB8888_U8) {
      return 4;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_AYUV444_U8) {
      return 4;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RAW10) {
      return 1;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RAW12) {
      return 1;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RAW16) {
      return 1;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RAW24) {
      return 1;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RGB8_IR) {
      return 4;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RGB16_IR) {
      return 4;
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RGB24_IR) {
      return 4;
    } else {
      return 3;
    }
  }

  return 3;
}

void SetAippMode(nlohmann::json& root, ::domi::AippOpParams* aipp_op_params) {
  if (aipp_op_params->aipp_mode()) {
    if (aipp_op_params->aipp_mode() == ::domi::AippOpParams_AippMode_static_) {
      root["aipp_mode"] = "static";
    } else if (aipp_op_params->aipp_mode() == ::domi::AippOpParams_AippMode_dynamic) {
      root["aipp_mode"] = "dynamic";
    } else {
      OP_LOGE("Aipp", "aipp_mode must be configured as static or dynamic!");
    }
  }
}

void SetInputFormat(nlohmann::json& root, ::domi::AippOpParams* aipp_op_params) {
  if (aipp_op_params->input_format()) {
    if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_YUV420SP_U8) {
      root["input_format"] = "YUV420SP_U8";
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_XRGB8888_U8) {
      root["input_format"] = "XRGB8888_U8";
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RGB888_U8) {
      root["input_format"] = "RGB888_U8";
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_YUV400_U8) {
      root["input_format"] = "YUV400_U8";
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_NC1HWC0DI_FP16) {
      root["input_format"] = "NC1HWC0DI_FP16";
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_NC1HWC0DI_S8) {
      root["input_format"] = "NC1HWC0DI_S8";
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_ARGB8888_U8) {
      root["input_format"] = "ARGB8888_U8";
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_YUYV_U8) {
      root["input_format"] = "YUYV_U8";
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_YUV422SP_U8) {
      root["input_format"] = "YUV422SP_U8";
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_AYUV444_U8) {
      root["input_format"] = "AYUV444_U8";
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RAW10) {
      root["input_format"] = "RAW10";
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RAW12) {
      root["input_format"] = "RAW12";
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RAW16) {
      root["input_format"] = "RAW16";
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RAW24) {
      root["input_format"] = "RAW24";
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RGB16) {
      root["input_format"] = "RGB16";
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RGB20) {
      root["input_format"] = "RGB20";
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RGB24) {
      root["input_format"] = "RGB24";
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RGB8_IR) {
      root["input_format"] = "RGB8_IR";
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RGB16_IR) {
      root["input_format"] = "RGB16_IR";
    } else if (aipp_op_params->input_format() == ::domi::AippOpParams_InputFormat_RGB24_IR) {
      root["input_format"] = "RGB24_IR";
    } else {
      OP_LOGE("Aipp", "Input format of AIPP conf is undefined!");
    }
  }
}

void SetSrcImageSize(nlohmann::json& root, ::domi::AippOpParams* aipp_op_params) {
  // src_image_size_w int32
  if (aipp_op_params->src_image_size_w()) {
    root["src_image_size_w"] = aipp_op_params->src_image_size_w();
  }
  // src_image_size_h int32
  if (aipp_op_params->src_image_size_h()) {
    root["src_image_size_h"] = aipp_op_params->src_image_size_h();
  }
}

void SetSwitch(nlohmann::json& root, ::domi::AippOpParams* aipp_op_params) {
  // csc_switch bool
  if (aipp_op_params->csc_switch()) {
    root["csc_switch"] = aipp_op_params->csc_switch();
  }
  // rbuv_swap_switch bool
  if (aipp_op_params->rbuv_swap_switch()) {
    root["rbuv_swap_switch"] = aipp_op_params->rbuv_swap_switch();
  }

  // ax_swap_switch bool
  if (aipp_op_params->ax_swap_switch()) {
    root["ax_swap_switch"] = aipp_op_params->ax_swap_switch();
  }
}

void SetMatrix(nlohmann::json& root, ::domi::AippOpParams* aipp_op_params) {
  // matrix_r0c0 repeated int32
  if (aipp_op_params->matrix_r0c0_size()) {
    root["matrix_r0c0"] = aipp_op_params->matrix_r0c0(0);
  }
  // matrix_r0c1 repeated int32
  if (aipp_op_params->matrix_r0c1_size()) {
    root["matrix_r0c1"] = aipp_op_params->matrix_r0c1(0);
  }
  // matrix_r0c2 repeated int32
  if (aipp_op_params->matrix_r0c2_size()) {
    root["matrix_r0c2"] = aipp_op_params->matrix_r0c2(0);
  }
  // matrix_r1c0 repeated int32
  if (aipp_op_params->matrix_r1c0_size()) {
    root["matrix_r1c0"] = aipp_op_params->matrix_r1c0(0);
  }
  // matrix_r1c1 repeated int32
  if (aipp_op_params->matrix_r1c1_size()) {
    root["matrix_r1c1"] = aipp_op_params->matrix_r1c1(0);
  }
  // matrix_r1c2 repeated int32
  if (aipp_op_params->matrix_r1c2_size()) {
    root["matrix_r1c2"] = aipp_op_params->matrix_r1c2(0);
  }
  // matrix_r2c0 repeated int32
  if (aipp_op_params->matrix_r2c0_size()) {
    root["matrix_r2c0"] = aipp_op_params->matrix_r2c0(0);
  }
  // matrix_r2c1 repeated int32
  if (aipp_op_params->matrix_r2c1_size()) {
    root["matrix_r2c1"] = aipp_op_params->matrix_r2c1(0);
  }
  // matrix_r2c2 repeated int32
  if (aipp_op_params->matrix_r2c2_size()) {
    root["matrix_r2c2"] = aipp_op_params->matrix_r2c2(0);
  }
}

void SetInputBias(nlohmann::json& root, ::domi::AippOpParams* aipp_op_params) {
  // input_bias_0 repeated int32
  if (aipp_op_params->input_bias_0_size()) {
    root["input_bias_0"] = aipp_op_params->input_bias_0(0);
  }
  // input_bias_1 repeated int32
  if (aipp_op_params->input_bias_1_size()) {
    root["input_bias_1"] = aipp_op_params->input_bias_1(0);
  }
  // input_bias_2 repeated int32
  if (aipp_op_params->input_bias_2_size()) {
    root["input_bias_2"] = aipp_op_params->input_bias_2(0);
  }
}

void SetOutputBias(nlohmann::json& root, ::domi::AippOpParams* aipp_op_params) {
  // output_bias_0 repeated int32
  if (aipp_op_params->output_bias_0_size()) {
    root["output_bias_0"] = aipp_op_params->output_bias_0(0);
  }
  // output_bias_1 repeated int32
  if (aipp_op_params->output_bias_1_size()) {
    root["output_bias_1"] = aipp_op_params->output_bias_1(0);
  }
  // output_bias_2 repeated int32
  if (aipp_op_params->output_bias_2_size()) {
    root["output_bias_2"] = aipp_op_params->output_bias_2(0);
  }
}

void SetMeanChn(nlohmann::json& root, ::domi::AippOpParams* aipp_op_params) {
  // mean_chn_0 int32
  if (aipp_op_params->mean_chn_0()) {
    root["mean_chn_0"] = aipp_op_params->mean_chn_0();
  }
  // mean_chn_1 int32
  if (aipp_op_params->mean_chn_1()) {
    root["mean_chn_1"] = aipp_op_params->mean_chn_1();
  }
  // mean_chn_2 int32
  if (aipp_op_params->mean_chn_2()) {
    root["mean_chn_2"] = aipp_op_params->mean_chn_2();
  }
  // mean_chn_3 int32
  if (aipp_op_params->mean_chn_3()) {
    root["mean_chn_3"] = aipp_op_params->mean_chn_3();
  }
}

void SetVarReciChn(nlohmann::json& root, ::domi::AippOpParams* aipp_op_params) {
  // var_reci_chn_0 repeated float
  if (aipp_op_params->var_reci_chn_0_size()) {
    root["var_reci_chn_0"] = aipp_op_params->var_reci_chn_0(0);
  }
  // var_reci_chn_1 repeated float
  if (aipp_op_params->var_reci_chn_1_size()) {
    root["var_reci_chn_1"] = aipp_op_params->var_reci_chn_1(0);
  }
  // var_reci_chn_2 repeated float
  if (aipp_op_params->var_reci_chn_2_size()) {
    root["var_reci_chn_2"] = aipp_op_params->var_reci_chn_2(0);
  }
  // var_reci_chn_3 repeated float
  if (aipp_op_params->var_reci_chn_3_size()) {
    root["var_reci_chn_3"] = aipp_op_params->var_reci_chn_3(0);
  }
}

void SetMinChn(nlohmann::json& root, ::domi::AippOpParams* aipp_op_params) {
  // min_chn_0 float
  if (aipp_op_params->min_chn_0()) {
    root["min_chn_0"] = aipp_op_params->min_chn_0();
  }
  // min_chn_1 float
  if (aipp_op_params->min_chn_1()) {
    root["min_chn_1"] = aipp_op_params->min_chn_1();
  }
  // min_chn_2 float
  if (aipp_op_params->min_chn_2()) {
    root["min_chn_2"] = aipp_op_params->min_chn_2();
  }
  // min_chn_3 float
  if (aipp_op_params->min_chn_3()) {
    root["min_chn_3"] = aipp_op_params->min_chn_3();
  }
}

void SetCrop(nlohmann::json& root, ::domi::AippOpParams* aipp_op_params) {
  // crop
  if (aipp_op_params->crop()) {
    root["crop"] = aipp_op_params->crop();
  }
  // load_start_pos_h
  if (aipp_op_params->load_start_pos_h()) {
    root["load_start_pos_h"] = aipp_op_params->load_start_pos_h();
  }

  // load_start_pos_w
  if (aipp_op_params->load_start_pos_w()) {
    root["load_start_pos_w"] = aipp_op_params->load_start_pos_w();
  }

  // crop_size_h
  if (aipp_op_params->crop_size_h()) {
    root["crop_size_h"] = aipp_op_params->crop_size_h();
  }

  // crop_size_h
  if (aipp_op_params->crop_size_w()) {
    root["crop_size_w"] = aipp_op_params->crop_size_w();
  }
}

void SetResize(nlohmann::json& root, ::domi::AippOpParams* aipp_op_params) {
  // resize
  if (aipp_op_params->resize()) {
    root["resize"] = aipp_op_params->resize();
  }
  // resize_output_h
  if (aipp_op_params->resize_output_h()) {
    root["resize_output_h"] = aipp_op_params->resize_output_h();
  }

  // resize_output_w
  if (aipp_op_params->resize_output_w()) {
    root["resize_output_w"] = aipp_op_params->resize_output_w();
  }
}

void SetPadding(nlohmann::json& root, ::domi::AippOpParams* aipp_op_params) {
  // padding
  if (aipp_op_params->padding()) {
    root["padding"] = aipp_op_params->padding();
    if (aipp_op_params->padding_value()) {
      root["padding_value"] = aipp_op_params->padding_value();
    }
  }
  // left_padding_size
  if (aipp_op_params->left_padding_size()) {
    root["left_padding_size"] = aipp_op_params->left_padding_size();
  }

  // right_padding_size
  if (aipp_op_params->right_padding_size()) {
    root["right_padding_size"] = aipp_op_params->right_padding_size();
  }

  // top_padding_size
  if (aipp_op_params->top_padding_size()) {
    root["top_padding_size"] = aipp_op_params->top_padding_size();
  }

  // bottom_padding_size
  if (aipp_op_params->bottom_padding_size()) {
    root["bottom_padding_size"] = aipp_op_params->bottom_padding_size();
  }
}

IMPLEMT_VERIFIER(Aipp, AippVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(Aipp, AippInfer) {
  int64_t has_infered_verified = 0;
  if (op.GetAttr("has_infered_verified", has_infered_verified) == GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "This aipp has infered, return success");
    return GRAPH_SUCCESS;
  }
  std::string aipp_config_path;
  op.GetAttr("aipp_config_path", aipp_config_path);
  char resolved_file_path[PATH_MAX] = {0x00};
  if (realpath(aipp_config_path.c_str(), resolved_file_path) == nullptr) {
    OP_LOGE(op.GetName().c_str(), "invalid insert op conf file path:%s.", aipp_config_path.c_str());
    return GRAPH_FAILED;
  }

  // protobuff message to json
  std::shared_ptr<domi::InsertNewOps> insert_op_conf_(new (std::nothrow) domi::InsertNewOps());
  if (insert_op_conf_ == nullptr) {
    OP_LOGE(op.GetName().c_str(), "insert_op_conf_ is null!");
    return GRAPH_FAILED;
  }

  bool ret = GraphUtils::ReadProtoFromTextFile(aipp_config_path.c_str(), insert_op_conf_.get());
  if (!ret) {
    OP_LOGE(op.GetName().c_str(), "Read AIPP conf file error!");
    return GRAPH_FAILED;
  }
  int64_t index = 0;
  if (op.GetAttr("current_aipp_index", index) == GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "Get current aipp index %d", index);
  }
  if (index >= insert_op_conf_->aipp_op_size()) {
    OP_LOGE(op.GetName().c_str(), "current_aipp_index %d is invalid", index);
    OpsGetAttrErrReport(op.GetName().c_str(), "current_aipp_index");
    return GRAPH_FAILED;
  }
  ::domi::AippOpParams* aipp_op_params = insert_op_conf_->mutable_aipp_op(index);
  if (aipp_op_params == nullptr) {
    std::string err_msg = GetInputInvalidErrMsg("aipp_op_params");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  nlohmann::json root;
  // aipp_mode AippMode
  SetAippMode(root, aipp_op_params);

  // related_input_rank uint32
  if (aipp_op_params->related_input_rank()) {
    root["related_input_rank"] = aipp_op_params->related_input_rank();
  }

  // input_format InputFormat
  SetInputFormat(root, aipp_op_params);

  // src_image_size_w int32  src_image_size_h int32
  SetSrcImageSize(root, aipp_op_params);

  // csc_switch bool rbuv_swap_switch bool
  SetSwitch(root, aipp_op_params);

  // matrix
  SetMatrix(root, aipp_op_params);

  // input_bias
  SetInputBias(root, aipp_op_params);

  // output_bias
  SetOutputBias(root, aipp_op_params);

  // mean_chn
  SetMeanChn(root, aipp_op_params);

  // var_reci_chn
  SetVarReciChn(root, aipp_op_params);

  // min_chn
  SetMinChn(root, aipp_op_params);

  // crop
  SetCrop(root, aipp_op_params);

  // resize
  SetResize(root, aipp_op_params);

  // padding
  SetPadding(root, aipp_op_params);

  // raw_rgbir_to_f16_n int32
  if (aipp_op_params->raw_rgbir_to_f16_n()) {
    root["raw_rgbir_to_f16_n"] = aipp_op_params->raw_rgbir_to_f16_n();
  }

  auto aipp_config_json = root.dump();
  op.set_attr_aipp_config_path(aipp_config_json);

  auto images_desc = op.GetInputDesc("images");
  auto images_shape = images_desc.GetShape().GetDims();
  uint64_t batch = 0;
  // uint64_t channel = 0;
  uint64_t height = 0;
  uint64_t width = 0;
  uint64_t c1 = 0;
  uint64_t c0 = 0;

  uint32_t src_image_size = 0;
  if (aipp_op_params->aipp_mode() == ::domi::AippOpParams_AippMode_dynamic) {
    OP_LOGI(op.GetName().c_str(), "aipp dynamic config!");

    (void)op.UpdateOutputDesc("features", images_desc);

    src_image_size = aipp_op_params->max_src_image_size() ? aipp_op_params->max_src_image_size() : 0;
    OP_LOGI(op.GetName().c_str(), "dynamic aipp_real_size is %u", src_image_size);

    // Set size to tensordesc
    images_desc.SetSize(src_image_size);

    vector<int64_t> shape_dync;
    shape_dync.push_back(1);
    shape_dync.push_back(src_image_size);

    images_desc.SetShape(Shape(shape_dync));
    images_desc.SetOriginShape(Shape(shape_dync));

    images_desc.SetDataType(DT_UINT8);
    images_desc.SetFormat(FORMAT_NHWC);
    images_desc.SetOriginFormat(FORMAT_NHWC);

    (void)op.UpdateInputDesc("images", images_desc);
    op.SetAttr("has_infered_verified", 1);
    return GRAPH_SUCCESS;
  }

  auto imagesDimNum = images_desc.GetShape().GetDimNum();
  if (((images_desc.GetFormat() == FORMAT_NCHW || images_desc.GetFormat() == FORMAT_NHWC) && imagesDimNum < 4)
      || (images_desc.GetFormat() == FORMAT_NC1HWC0_C04 && imagesDimNum < 5)) {
      OpsOneInputShapeErrReport(op.GetName(), "images shape dims", "The input shape of images is invalid");
      OP_LOGE(op.GetName().c_str(), "The input shape of images is invalid");
      return GRAPH_FAILED;
  }
  if (images_desc.GetFormat() == FORMAT_NCHW) {
    batch = images_shape[0];
    height = images_shape[2];
    width = images_shape[3];
  } else if (images_desc.GetFormat() == FORMAT_NHWC) {
    batch = images_shape[0];
    height = images_shape[1];
    width = images_shape[2];
  } else if (images_desc.GetFormat() == FORMAT_NC1HWC0_C04) {
    batch = images_shape[0];
    c1 = images_shape[1];
    height = images_shape[2];
    width = images_shape[3];
    c0 = images_shape[4];
  } else {
    OpsInputFormatErrReport(op.GetName(), "images", "NCHW, NHWC or NC1HWC0_C04",
                            ge::TypeUtils::FormatToSerialString(images_desc.GetFormat()));
    OP_LOGE(op.GetName().c_str(), "aipp input format only support NCHW, NHWC, NC1HWC0_C04.");
    return GRAPH_FAILED;
  }

  uint64_t real_channel = 1;
  if (images_desc.GetFormat() != FORMAT_NC1HWC0_C04) {
    real_channel = GetChannel(aipp_op_params);
    OP_LOGI(op.GetName().c_str(), "real_channel:%d", (int)real_channel);
  }

  uint64_t output_height = height;
  uint64_t output_width = width;
  (void)GetOutputHeightWidth(aipp_op_params, &output_height, &output_width);

  OP_LOGI(op.GetName().c_str(), "aipp output_height:%d, aipp output_width:%d, data's height:%d, data's width:%d",
          (int)output_height, (int)output_width, (int)height, (int)width);

  if (output_height != height || output_width != width) {
    OpsAippErrReport(ConcatString(output_height), ConcatString(output_width), ConcatString(height),
                     ConcatString(width));
    OP_LOGE(op.GetName().c_str(),
            "the data output H and W is not equal with aipp output H and W."
            "aipp output_height:%d, aipp output_width:%d, data's height:%d, data's width:%d",
            (int)output_height, (int)output_width, (int)height, (int)width);

    return GRAPH_FAILED;
  }

  ge::DataType src_image_dtype = DT_UINT8;
  src_image_size = GetSrcImageSizeDtype(aipp_op_params, batch, c1, height, width, &src_image_dtype);
  // Set size to tensordesc
  images_desc.SetSize(src_image_size);
  OP_LOGI(op.GetName().c_str(), "aipp_real_size is %u", src_image_size);

  (void)op.UpdateOutputDesc("features", images_desc);

  uint64_t src_image_size_h = aipp_op_params->src_image_size_h() ? aipp_op_params->src_image_size_h() : height;
  uint64_t src_image_size_w = aipp_op_params->src_image_size_w() ? aipp_op_params->src_image_size_w() : width;
  vector<int64_t> shape;
  if (images_desc.GetFormat() == FORMAT_NCHW) {
    shape.push_back(batch);
    shape.push_back(src_image_size_h);
    shape.push_back(src_image_size_w);
    shape.push_back(real_channel);
    images_desc.SetFormat(FORMAT_NHWC);
    images_desc.SetOriginFormat(FORMAT_NHWC);
  } else if (images_desc.GetFormat() == FORMAT_NHWC) {
    shape.push_back(batch);
    shape.push_back(src_image_size_h);
    shape.push_back(src_image_size_w);
    shape.push_back(real_channel);
    images_desc.SetFormat(FORMAT_NHWC);
    images_desc.SetOriginFormat(FORMAT_NHWC);
  } else if (images_desc.GetFormat() == FORMAT_NC1HWC0_C04) {
    shape.push_back(batch);
    shape.push_back(c1);
    shape.push_back(src_image_size_h);
    shape.push_back(src_image_size_w);
    shape.push_back(c0);
  }

  images_desc.SetShape(Shape(shape));
  images_desc.SetOriginShape(Shape(shape));
  images_desc.SetDataType(src_image_dtype);
  (void)op.UpdateInputDesc("images", images_desc);
  op.SetAttr("has_infered_verified", 1);

  std::vector<int32_t> aclInputDims;
  aclInputDims = GetAclInputDims(aipp_op_params, batch, src_image_size_h, src_image_size_w);
  OP_LOGI(op.GetName().c_str(), "aclInputDims size: %d", aclInputDims.size());
  if (aclInputDims.size() >= 4) {
    OP_LOGI(op.GetName().c_str(), "aclInputDims: %d, %d, %d, %d", aclInputDims[0], aclInputDims[1], aclInputDims[2],
            aclInputDims[3]);
    op.SetAttr("input_dims", aclInputDims);
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_INFER_DATA_SLICE(Aipp, AippInferDataSlice) {
  OP_LOGI(op.GetName().c_str(), "AippInferDataSlice start.");

  auto images_desc = op.GetInputDesc("images");
  auto input_format = images_desc.GetFormat();

  if (input_format != FORMAT_NHWC && input_format != FORMAT_NCHW && input_format != FORMAT_NC1HWC0_C04) {
    OP_LOGE(op.GetName().c_str(), "aipp input format only support NCHW, NHWC, NC1HWC0_C04.");
    OpsInputFormatErrReport(op.GetName(), "images", "NCHW, NHWC or NC1HWC0_C04", ConcatString(input_format));
    return GRAPH_FAILED;
  }

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  vector<vector<int64_t>> images_data_slice = {{}, {}, {}, {}};
  vector<vector<int64_t>> output_data_slice = {{}, {}, {}, {}, {}};
  GeTensorDescPtr tensor_desc_in = op_desc->MutableInputDesc("images");
  GeTensorDescPtr tensor_desc_out = op_desc->MutableOutputDesc("features");

  if (!ge::AttrUtils::GetListListInt(tensor_desc_out, ge::ATTR_NAME_DATA_SLICE, output_data_slice)) {
    OP_LOGI(op.GetName().c_str(), "no data slice, use default as {{}, {}, {}, {}, {}}");
    return GRAPH_FAILED;
  }

  for (unsigned i = 0; i < output_data_slice.size(); i++) {
    if (output_data_slice[i].size() > 0) {
      if (output_data_slice[i].size() != 2) {
        OP_LOGE(op.GetName().c_str(), "data slice format input size should be 2.");
        return GRAPH_FAILED;
      }

      if (i == 0) {
        int64_t n_start = output_data_slice[i][0];
        int64_t n_end = output_data_slice[i][1];
        images_data_slice[i] = {n_start, n_end};

        if (input_format == FORMAT_NC1HWC0_C04) {
          images_data_slice.push_back({});
        }
        if (!AttrUtils::SetListListInt(tensor_desc_in, ge::ATTR_NAME_DATA_SLICE, images_data_slice)) {
          OP_LOGE(op.GetName().c_str(), "images data_slice set failed.");
          return GRAPH_FAILED;
        }
      } else {
        OP_LOGI(op.GetName().c_str(), "only support cut in n");
        return NOT_SUPPORT_SLICE;
      }
    }
  }

  OP_LOGI(op.GetName().c_str(), "AippInferDataSlice success.");
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Aipp, AippInfer);
VERIFY_FUNC_REG(Aipp, AippVerify);
INFER_DATA_SLICE_FUNC_REG(Aipp, AippInferDataSlice);

COMMON_INFER_FUNC_REG(AippData, ELMTWISE_INFER_SHAPEANDTYPE("data", "out"));
}  // namespace ge
