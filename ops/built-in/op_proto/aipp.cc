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
#include "op_const.h"
#include "proto/insert_op.pb.h"
#include <nlohmann/json.hpp>

#include "graph/utils/graph_utils.h"
#include "./util/error_util.h"
#include "graph/utils/type_utils.h"
#include "register/infer_data_slice_registry.h"
#include "graph/common_error_codes.h"
#include "graph/debug/ge_attr_define.h"


namespace ge {

namespace {
  constexpr uint8_t kChannel1 = 1;
  constexpr uint8_t kChannel3 = 3;
  constexpr uint8_t kChannel4 = 4;

  constexpr uint8_t kFormatYuv420sp = 1;
  constexpr uint8_t kFormatXrgb8888 = 2;
  constexpr uint8_t kFormatRgb888 = 5;
  constexpr uint8_t kFormatArgb8888 = 6;
  constexpr uint8_t kFormatYuyv = 7;
  constexpr uint8_t kFormatYuv422sp = 8;
  constexpr uint8_t kFormatAyuv444 = 9;
  constexpr uint8_t kFormatYuv400 = 10;

  constexpr int32_t kParamsHeadSize = 64;
  constexpr int32_t kSize = 4;
  constexpr int32_t kBatchNumIndex = 4;
  constexpr int32_t kSrcImageSizeWIndex = 8;
  constexpr int32_t kSrcImageSizeHIndex = 12;
  constexpr int32_t kScfSwitchIndex = 1;
  constexpr int32_t kPaddingSwitchIndex = 2;
  constexpr int32_t kCropStartPosWIndex = 8;
  constexpr int32_t kCropStartPosHIndex = 12;
  constexpr int32_t kCropSizeWIndex = 16;
  constexpr int32_t kCropSizeHIndex = 20;
  constexpr int32_t kScfInputSizeWIndex = 24;
  constexpr int32_t kScfInputSizeHIndex = 28;
  constexpr int32_t kScfOutputSizeWIndex = 32;
  constexpr int32_t kScfOutputSizeHIndex = 36;
  constexpr int32_t kPaddingSizeTopIndex = 40;
  constexpr int32_t kPaddingSizeBottomIndex = 44;
  constexpr int32_t kPaddingSizeLeftIndex = 48;
  constexpr int32_t kPaddingSizeRightIndex = 52;

  constexpr int64_t kMinCropSize = 8;
}

struct AippParams {
  uint8_t inputFormat;
  int8_t batchNum;
  int32_t srcImageSizeW;
  int32_t srcImageSizeH;

  int8_t cropSwitch;
  int8_t scfSwitch;
  int8_t paddingSwitch;
  int32_t cropStartPosW;
  int32_t cropStartPosH;
  int32_t cropSizeW;
  int32_t cropSizeH;
  int32_t scfInputSizeW;
  int32_t scfInputSizeH;
  int32_t scfOutputSizeW;
  int32_t scfOutputSizeH;
  int32_t paddingSizeTop;
  int32_t paddingSizeBottom;
  int32_t paddingSizeLeft;
  int32_t paddingSizeRight;
};

void InitAippParams(AippParams& params) {
  params.inputFormat = 0;
  params.batchNum = 0;
  params.srcImageSizeW = 0;
  params.srcImageSizeH = 0;

  params.cropSwitch = 0;
  params.scfSwitch = 0;
  params.paddingSwitch = 0;
  params.cropStartPosW = 0;
  params.cropStartPosH = 0;
  params.cropSizeW = 0;
  params.cropSizeH = 0;
  params.scfInputSizeW = 0;
  params.scfInputSizeH = 0;
  params.scfOutputSizeW = 0;
  params.scfOutputSizeH = 0;
  params.paddingSizeTop = 0;
  params.paddingSizeBottom = 0;
  params.paddingSizeLeft = 0;
  params.paddingSizeRight = 0;
}

void ParseAippParams(const GeTensor* params_tensor, AippParams& aippParams) {
  const uint8_t* constData = params_tensor->GetData().GetData();

  aippParams.inputFormat = *constData;
  aippParams.batchNum = *((int8_t*)constData + kBatchNumIndex);
  aippParams.srcImageSizeW = *((int32_t*)constData + (kSrcImageSizeWIndex / kSize));
  aippParams.srcImageSizeH = *((int32_t*)constData + (kSrcImageSizeHIndex / kSize));

  aippParams.cropSwitch = *((int8_t*)constData + kParamsHeadSize);
  aippParams.scfSwitch = *((int8_t*)constData + (kParamsHeadSize + kScfSwitchIndex));
  aippParams.paddingSwitch = *((int8_t*)constData + (kParamsHeadSize + kPaddingSwitchIndex));
  aippParams.cropStartPosW = *((int32_t*)constData + (kParamsHeadSize + kCropStartPosWIndex) / kSize);
  aippParams.cropStartPosH = *((int32_t*)constData + (kParamsHeadSize + kCropStartPosHIndex) / kSize);
  aippParams.cropSizeW = *((int32_t*)constData + (kParamsHeadSize + kCropSizeWIndex) / kSize);
  aippParams.cropSizeH = *((int32_t*)constData + (kParamsHeadSize + kCropSizeHIndex) / kSize);
  aippParams.scfInputSizeW = *((int32_t*)constData + (kParamsHeadSize + kScfInputSizeWIndex) / kSize);
  aippParams.scfInputSizeH = *((int32_t*)constData + (kParamsHeadSize + kScfInputSizeHIndex) / kSize);
  aippParams.scfOutputSizeW = *((int32_t*)constData + (kParamsHeadSize + kScfOutputSizeWIndex) / kSize);
  aippParams.scfOutputSizeH = *((int32_t*)constData + (kParamsHeadSize + kScfOutputSizeHIndex) / kSize);
  aippParams.paddingSizeTop = *((int32_t*)constData + (kParamsHeadSize + kPaddingSizeTopIndex) / kSize);
  aippParams.paddingSizeBottom = *((int32_t*)constData + (kParamsHeadSize + kPaddingSizeBottomIndex) / kSize);
  aippParams.paddingSizeLeft = *((int32_t*)constData + (kParamsHeadSize + kPaddingSizeLeftIndex) / kSize);
  aippParams.paddingSizeRight = *((int32_t*)constData + (kParamsHeadSize + kPaddingSizeRightIndex) / kSize);
}

void PrintAippParams(Operator& op, AippParams& aippParams) {
  OP_LOGD(TbeGetName(op).c_str(), "aippParams  inputFormat=%d", aippParams.inputFormat);
  OP_LOGD(TbeGetName(op).c_str(), "aippParams  batchNum=%d", aippParams.batchNum);
  OP_LOGD(TbeGetName(op).c_str(), "aippParams  srcImageSizeW=%d", aippParams.srcImageSizeW);
  OP_LOGD(TbeGetName(op).c_str(), "aippParams  srcImageSizeH=%d", aippParams.srcImageSizeH);
  OP_LOGD(TbeGetName(op).c_str(), "aippParams  cropSwitch=%d", aippParams.cropSwitch);
  OP_LOGD(TbeGetName(op).c_str(), "aippParams  scfSwitch=%d", aippParams.scfSwitch);
  OP_LOGD(TbeGetName(op).c_str(), "aippParams  paddingSwitch=%d", aippParams.paddingSwitch);
  OP_LOGD(TbeGetName(op).c_str(), "aippParams  cropStartPosW=%d", aippParams.cropStartPosW);
  OP_LOGD(TbeGetName(op).c_str(), "aippParams  cropStartPosH=%d", aippParams.cropStartPosH);
  OP_LOGD(TbeGetName(op).c_str(), "aippParams  cropSizeW=%d", aippParams.cropSizeW);
  OP_LOGD(TbeGetName(op).c_str(), "aippParams  cropSizeH=%d", aippParams.cropSizeH);
  OP_LOGD(TbeGetName(op).c_str(), "aippParams  scfInputSizeW=%d", aippParams.scfInputSizeW);
  OP_LOGD(TbeGetName(op).c_str(), "aippParams  scfInputSizeH=%d", aippParams.scfInputSizeH);
  OP_LOGD(TbeGetName(op).c_str(), "aippParams  scfOutputSizeW=%d", aippParams.scfOutputSizeW);
  OP_LOGD(TbeGetName(op).c_str(), "aippParams  scfOutputSizeH=%d", aippParams.scfOutputSizeH);
  OP_LOGD(TbeGetName(op).c_str(), "aippParams  paddingSizeTop=%d", aippParams.paddingSizeTop);
  OP_LOGD(TbeGetName(op).c_str(), "aippParams  paddingSizeBottom=%d", aippParams.paddingSizeBottom);
  OP_LOGD(TbeGetName(op).c_str(), "aippParams  paddingSizeLeft=%d", aippParams.paddingSizeLeft);
  OP_LOGD(TbeGetName(op).c_str(), "aippParams  paddingSizeRight=%d", aippParams.paddingSizeRight);
}

int64_t GetDynamicShapeChannel(uint8_t inputFormat) {
  uint8_t channel = kChannel3;
  switch (inputFormat) {
    case kFormatXrgb8888:
    case kFormatArgb8888:
    case kFormatAyuv444:
      channel = kChannel4;
      break;
    case kFormatYuv400:
      channel = kChannel1;
      break;
    default:
      channel = kChannel3;
      break;
  }
  return channel;
}

bool GetDynamicShapeOutputHW(AippParams& aippParams, int64_t* outputH, int64_t* outputW) {
  if (outputH == nullptr || outputW == nullptr) {
    OP_LOGE("Aipp", "outputH or outputW is null!");
    return false;
  }
  OP_LOGD("Aipp", "GetDynamicShapeOutputHW  srcImageSizeH[%ld], srcImageSizeW[%ld]", *outputH, *outputW);
  if (*outputH < kMinCropSize || *outputW < kMinCropSize) {
    OP_LOGE("Aipp", "srcImageSizeH[%ld], srcImageSizeW[%ld] must be greater than or equal to 8", *outputH, *outputW);
    return false;
  }
  if (aippParams.cropSwitch > 0) {
    *outputH = aippParams.cropSizeH ? aippParams.cropSizeH : *outputH;
    *outputW = aippParams.cropSizeW ? aippParams.cropSizeW : *outputW;
    if (*outputH < kMinCropSize || *outputW < kMinCropSize) {
      OP_LOGE("Aipp", "cropSizeH[%ld], cropSizeW[%ld] must be greater than or equal to 8", *outputH, *outputW);
      return false;
    }
  }

  if (aippParams.scfSwitch > 0) {
    *outputH = aippParams.scfOutputSizeH ? aippParams.scfOutputSizeH : *outputH;
    *outputW = aippParams.scfOutputSizeW ? aippParams.scfOutputSizeW : *outputW;
  }

  if (aippParams.paddingSwitch > 0) {
    *outputH = *outputH + aippParams.paddingSizeTop + aippParams.paddingSizeBottom;
    *outputW = *outputW + aippParams.paddingSizeLeft + aippParams.paddingSizeRight;
  }

  return true;
}

bool CheckImageInputFormat(AippParams& aippParams) {
  if (aippParams.inputFormat != kFormatYuv420sp && aippParams.inputFormat != kFormatYuv400 &&
      aippParams.inputFormat != kFormatRgb888 && aippParams.inputFormat != kFormatXrgb8888) {
    OP_LOGE("Aipp", "inputFormat only support yuv420sp, yuv400, rgb888, xrgb8888");
    return false;
  }
  return true;
}

int64_t GetDynamicShapeSrcSize(AippParams& aippParams, int64_t batch, ge::DataType* src_img_dtype) {
  int64_t size = 0;
  int64_t srcImageSizeH = aippParams.srcImageSizeH;
  int64_t srcImageSizeW = aippParams.srcImageSizeW;

  switch (aippParams.inputFormat) {
    case kFormatYuv420sp:
      size = batch * 3 * srcImageSizeH * srcImageSizeW / 2;
      *src_img_dtype = DT_UINT8;
      break;
    case kFormatXrgb8888:
    case kFormatArgb8888:
    case kFormatAyuv444:
      size = batch * 4 * srcImageSizeH * srcImageSizeW;
      *src_img_dtype = DT_UINT8;
      break;
    case kFormatYuyv:
    case kFormatYuv422sp:
      size = batch * 2 * srcImageSizeH * srcImageSizeW;
      *src_img_dtype = DT_UINT8;
      break;
    case kFormatYuv400:
      size = batch * srcImageSizeH * srcImageSizeW;
      *src_img_dtype = DT_UINT8;
      break;
    default:
      size = batch * 3 * srcImageSizeH * srcImageSizeW;
      *src_img_dtype = DT_UINT8;
      break;
  }
  return size;
}

void GetOutputHeightWidth(::domi::AippOpParams* aipp_op_params, int64_t* output_height, int64_t* output_width) {
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
    int64_t left_padding_size = aipp_op_params->left_padding_size() ? aipp_op_params->left_padding_size() : 0;
    int64_t right_padding_size = aipp_op_params->right_padding_size() ? aipp_op_params->right_padding_size() : 0;
    int64_t top_padding_size = aipp_op_params->top_padding_size() ? aipp_op_params->top_padding_size() : 0;
    int64_t bottom_padding_size = aipp_op_params->bottom_padding_size() ? aipp_op_params->bottom_padding_size() : 0;

    *output_height = *output_height + top_padding_size + bottom_padding_size;
    *output_width = *output_width + left_padding_size + right_padding_size;
  }
}

int64_t GetSrcImageSizeDtype(::domi::AippOpParams* aipp_op_params, int64_t batch, int64_t c1, int64_t height,
                             int64_t width, ge::DataType* src_img_dtype) {
  int64_t size = 0;
  int64_t src_image_size_h = aipp_op_params->src_image_size_h() ? aipp_op_params->src_image_size_h() : height;
  int64_t src_image_size_w = aipp_op_params->src_image_size_w() ? aipp_op_params->src_image_size_w() : width;

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

std::vector<int32_t> GetAclInputDims(::domi::AippOpParams* aipp_op_params, int64_t batch, int64_t srcImageHeight,
                                     int64_t srcImageWidth) {
  int64_t channel = 3;
  int64_t height = srcImageHeight;
  int64_t width = srcImageWidth;
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

int64_t GetChannel(::domi::AippOpParams* aipp_op_params) {
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

static void SaveImagesDesc(Operator& op, const std::string& aipp_config_path) {
  OP_LOGD(TbeGetName(op).c_str(), "aipp infershape, SaveImagesDesc start");
  int64_t has_infered_verified = 0;
  if (op.GetAttr("has_infered_verified", has_infered_verified) != GRAPH_SUCCESS) {
    OP_LOGD(TbeGetName(op).c_str(), "SaveImagesDesc, aipp_config_path is %s", aipp_config_path.c_str());
    op.SetAttr("aipp_config_file_path", aipp_config_path);

    auto images_desc = op.GetInputDesc("images");
    Tensor images_tensor(images_desc);
    OP_LOGD(TbeGetName(op).c_str(),
            "SaveImagesDesc, images_shape is %s, images_dtype value is %d, images_format value is %d",
            to_string(images_desc.GetShape().GetDims()).c_str(),
            images_desc.GetDataType(), images_desc.GetFormat());
    op.SetAttr("aipp_images_tensor_bak", images_tensor);
  }
  OP_LOGD(TbeGetName(op).c_str(), "aipp infershape, SaveImagesDesc end");
}

static void GetImagesDesc(Operator& op, TensorDesc& images_desc) {
  OP_LOGD(TbeGetName(op).c_str(), "aipp infershape, GetImagesDesc start");
  Tensor images_tensor;
  if (op.GetAttr("aipp_images_tensor_bak", images_tensor) == GRAPH_SUCCESS) {
    OP_LOGD(TbeGetName(op).c_str(), "GetImagesDesc, get aipp_images_tensor_bak success");
    images_desc = images_tensor.GetTensorDesc();
    OP_LOGD(TbeGetName(op).c_str(),
            "GetImagesDesc, images_shape is %s, images_dtype value is %d, images_format value is %d",
            to_string(images_desc.GetShape().GetDims()).c_str(),
            images_desc.GetDataType(), images_desc.GetFormat());
  } else {
    OP_LOGD(TbeGetName(op).c_str(), "GetImagesDesc, first, get images_desc from images");
    images_desc = op.GetInputDesc("images");
  }
  OP_LOGD(TbeGetName(op).c_str(), "aipp infershape, GetImagesDesc end");
}

static graphStatus DynamicShapeInfershape(Operator& op, const GeTensor* params_data) {
  OP_LOGI(TbeGetName(op).c_str(), "aipp infershape, aipp dynamic shape start");
  // parse params data
  AippParams aippParams;
  InitAippParams(aippParams);
  ParseAippParams(params_data, aippParams);
  PrintAippParams(op, aippParams);
  if (!CheckImageInputFormat(aippParams)) {
    std::string err_msg = OtherErrMsg("aipp dynamic shape, image input format is not support");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto images_desc = op.GetInputDesc("images");
  auto features_desc = op.GetOutputDesc("features");
  auto features_format = features_desc.GetFormat();
  if (features_format != FORMAT_NCHW && features_format != FORMAT_NHWC) {
    OP_LOGE(TbeGetName(op).c_str(), "aipp dynamic shape, input format only support NCHW, NHWC.");
    return GRAPH_FAILED;
  }

  std::vector<std::pair<int64_t, int64_t>> features_shape_range;
  features_desc.GetShapeRange(features_shape_range);
  if (features_shape_range.size() != 5) {
    OP_LOGE(TbeGetName(op).c_str(), "aipp dynamic shape, the size of features shape range is invalid.");
    return GRAPH_FAILED;
  }
  OP_LOGD(TbeGetName(op).c_str(), "features shape range is %s", to_string(features_shape_range).c_str());
  int64_t n_start = features_shape_range[0].first;
  int64_t n_end = features_shape_range[0].second;
  int64_t h_start = features_shape_range[2].first;
  int64_t h_end = features_shape_range[2].second;
  int64_t w_start = features_shape_range[3].first;
  int64_t w_end = features_shape_range[3].second;
  OP_LOGD(TbeGetName(op).c_str(), "features shape range is n: [%ld,%ld], height: [%ld,%ld], width: [%ld,%ld]",
          n_start, n_end, h_start, h_end, w_start, w_end);

  int64_t batch = aippParams.batchNum;
  int64_t real_channel = GetDynamicShapeChannel(aippParams.inputFormat);
  int64_t inputH = aippParams.srcImageSizeH;
  int64_t inputW = aippParams.srcImageSizeW;
  int64_t outputH = inputH;
  int64_t outputW = inputW;
  if (!GetDynamicShapeOutputHW(aippParams, &outputH, &outputW)) {
    std::string err_msg = OtherErrMsg("aipp dynamic shape, GetDynamicShapeOutputHW error");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  OP_LOGD(TbeGetName(op).c_str(), "aipp dynamic shape, batch=%ld, real_channel=%ld", batch, real_channel);
  OP_LOGI(TbeGetName(op).c_str(), "aipp dynamic shape, srcImageSizeH=%ld, srcImageSizeW=%ld, outputH=%ld, outputW=%ld",
          inputH, inputW, outputH, outputW);
  if ((n_start < 1 || (n_end != -1 && (batch < n_start || batch > n_end)))
      || (h_start < 1 || (h_end != -1 && (outputH < h_start || outputH > h_end)))
      || (w_start < 1 || (w_end != -1 && (outputW < w_start || outputW > w_end)))) {
    OP_LOGE(TbeGetName(op).c_str(), "The output shape is invalid, out of range. batch is %ld, batch range is [%ld,%ld],"
            "height is %ld, height range is [%ld,%ld], width is %ld, width range is [%ld,%ld]",
            batch, n_start, n_end, outputH, h_start, h_end, outputW, w_start, w_end);
    return GRAPH_FAILED;
  }

  vector<int64_t> shape;
  if (features_desc.GetFormat() == FORMAT_NCHW) {
    OP_LOGD(TbeGetName(op).c_str(), "features Format is NCHW");
    shape.push_back(batch);
    shape.push_back(real_channel);
    shape.push_back(outputH);
    shape.push_back(outputW);
  } else if (features_desc.GetFormat() == FORMAT_NHWC) {
    OP_LOGD(TbeGetName(op).c_str(), "features Format is NHWC");
    shape.push_back(batch);
    shape.push_back(outputH);
    shape.push_back(outputW);
    shape.push_back(real_channel);
  } else {
    OP_LOGE(TbeGetName(op).c_str(), "aipp dynamic shape, features format is not support.");
    return GRAPH_FAILED;
  }
  features_desc.SetShape(Shape(shape));
  (void)op.UpdateOutputDesc("features", features_desc);

  ge::DataType src_image_dtype = DT_UINT8;
  int64_t src_img_size = GetDynamicShapeSrcSize(aippParams, batch, &src_image_dtype);
  OP_LOGD(TbeGetName(op).c_str(), "aipp dynamic shape, src_img_size is %ld", src_img_size);
  images_desc.SetSize(src_img_size);
  images_desc.SetDataType(src_image_dtype);
  images_desc.SetShape(Shape({batch, inputH, inputW, real_channel}));
  images_desc.SetOriginShape(Shape({batch, inputH, inputW, real_channel}));
  images_desc.SetFormat(FORMAT_NHWC);
  images_desc.SetOriginFormat(FORMAT_NHWC);
  (void)op.UpdateInputDesc("images", images_desc);

  OP_LOGI(TbeGetName(op).c_str(), "aipp infer, dynamic shape, success");
  return GRAPH_SUCCESS;
}

static graphStatus DynamicModeInfershape(Operator& op, ::domi::AippOpParams* aipp_op_params) {
  OP_LOGI(TbeGetName(op).c_str(), "aipp infershape, aipp dynamic mode, start");
  int64_t src_image_size = aipp_op_params->max_src_image_size() ? aipp_op_params->max_src_image_size() : 0;
  OP_LOGD(TbeGetName(op).c_str(), "dynamic aipp_real_size is %ld", src_image_size);
  if (src_image_size <= 0) {
    OP_LOGE(TbeGetName(op).c_str(), "the max_src_image_size must be set to a value greater than 0 in cfg");
    return GRAPH_FAILED;
  }

  TensorDesc images_desc;
  GetImagesDesc(op, images_desc);
  OP_LOGD(TbeGetName(op).c_str(), "images_desc size is %u", images_desc.GetSize());
  auto images_shape = images_desc.GetShape().GetDims();
  if (IsUnknownRankShape(images_shape)) {
    OP_LOGE(TbeGetName(op).c_str(), "The shape is unknown rank, not support!");
    return GRAPH_FAILED;
  }
  if (IsUnKnownShape(images_shape)) {
    OP_LOGD(TbeGetName(op).c_str(), "aipp dynamic shape, set depend");
    const vector<string> depend_name = {"params"};
    PREPARE_DYNAMIC_SHAPE(depend_name);
  }
  (void)op.UpdateOutputDesc("features", images_desc);

  // Set size to tensordesc
  images_desc.SetSize(src_image_size);

  std::vector<std::pair<int64_t, int64_t>> images_shape_range;
  images_desc.GetShapeRange(images_shape_range);
  if (!images_shape_range.empty() && images_shape_range.size() == 4) {
    OP_LOGI(TbeGetName(op).c_str(), "images shape range is %s", to_string(images_shape_range).c_str());
    std::vector<std::pair<int64_t, int64_t>> images_shape_range_new;
    images_shape_range_new.push_back(std::pair<int64_t, int64_t>(1, 1));
    images_shape_range_new.push_back(std::pair<int64_t, int64_t>(1, src_image_size));
    images_desc.SetShapeRange(images_shape_range_new);
    OP_LOGI(TbeGetName(op).c_str(), "after, images shape range is %s", to_string(images_shape_range_new).c_str());
  }

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
  OP_LOGI(TbeGetName(op).c_str(), "aipp infer, dynamic mode, success");

  return GRAPH_SUCCESS;
}

static graphStatus StaticInferShape(Operator& op, ::domi::AippOpParams* aipp_op_params) {
  OP_LOGI(TbeGetName(op).c_str(), "aipp infershape, aipp static mode, start");
  TensorDesc images_desc;
  GetImagesDesc(op, images_desc);
  auto images_shape = images_desc.GetShape().GetDims();

  int64_t batch = 0;
  int64_t height = 0;
  int64_t width = 0;
  int64_t c1 = 0;
  int64_t c0 = 0;

  auto imagesDimNum = images_desc.GetShape().GetDimNum();
  if (((images_desc.GetFormat() == FORMAT_NCHW || images_desc.GetFormat() == FORMAT_NHWC) && imagesDimNum < 4)
      || (images_desc.GetFormat() == FORMAT_NC1HWC0_C04 && imagesDimNum < 5)) {
    OP_LOGE(TbeGetName(op).c_str(), "The input shape of images is invalid");
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
    OP_LOGE(TbeGetName(op).c_str(), "aipp input format only support NCHW, NHWC, NC1HWC0_C04.");
    return GRAPH_FAILED;
  }
  OP_LOGI(TbeGetName(op).c_str(), "batch=%ld, height=%ld, width=%ld", batch, height, width);

  int64_t real_channel = 1;
  if (images_desc.GetFormat() != FORMAT_NC1HWC0_C04) {
    real_channel = GetChannel(aipp_op_params);
    OP_LOGI(TbeGetName(op).c_str(), "real_channel:%d", (int)real_channel);
  }

  int64_t output_height = height;
  int64_t output_width = width;
  (void)GetOutputHeightWidth(aipp_op_params, &output_height, &output_width);

  OP_LOGI(TbeGetName(op).c_str(), "aipp output_height:%d, aipp output_width:%d, data's height:%d, data's width:%d",
          (int)output_height, (int)output_width, (int)height, (int)width);

  if (output_height != height || output_width != width) {
    OP_LOGE(TbeGetName(op).c_str(), "the data output H and W is not equal with aipp output H and W."
            "aipp output_height:%d, aipp output_width:%d, data's height:%d, data's width:%d",
            (int)output_height, (int)output_width, (int)height, (int)width);

    return GRAPH_FAILED;
  }

  ge::DataType src_image_dtype = DT_UINT8;
  int64_t src_image_size = GetSrcImageSizeDtype(aipp_op_params, batch, c1, height, width, &src_image_dtype);
  // set size to tensordesc
  images_desc.SetSize(src_image_size);
  OP_LOGI(TbeGetName(op).c_str(), "aipp_real_size is %ld", src_image_size);

  (void)op.UpdateOutputDesc("features", images_desc);

  int64_t src_image_size_h = aipp_op_params->src_image_size_h() ? aipp_op_params->src_image_size_h() : height;
  int64_t src_image_size_w = aipp_op_params->src_image_size_w() ? aipp_op_params->src_image_size_w() : width;
  OP_LOGD(TbeGetName(op).c_str(), "src_image_size_h=%ld, src_image_size_w=%ld", src_image_size_h, src_image_size_w);
  vector<int64_t> shape;
  if (images_desc.GetFormat() == FORMAT_NCHW || images_desc.GetFormat() == FORMAT_NHWC) {
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
  OP_LOGI(TbeGetName(op).c_str(), "aclInputDims size: %d", aclInputDims.size());
  if (aclInputDims.size() >= 4) {
    OP_LOGI(TbeGetName(op).c_str(), "aclInputDims: %d, %d, %d, %d", aclInputDims[0], aclInputDims[1], aclInputDims[2],
            aclInputDims[3]);
    op.SetAttr("input_dims", aclInputDims);
  }
  OP_LOGI(TbeGetName(op).c_str(), "aipp infer, static mode, success");

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Aipp, AippVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AippInfer) {
  OP_LOGI(TbeGetName(op).c_str(), "AippInfer start");
  std::string aipp_config_path;
  if (op.GetAttr("aipp_config_path", aipp_config_path) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "AippInfer, failed to get attr aipp_config_path");
    return GRAPH_FAILED;
  }
  SaveImagesDesc(op, aipp_config_path);

  if (nlohmann::json::accept(aipp_config_path)) {
    OP_LOGD(TbeGetName(op).c_str(), "AippInfer, aipp_config_path is json");
    int64_t modePosition = aipp_config_path.find("aipp_mode");
    int64_t dynamicPosition = aipp_config_path.find("dynamic");
    if (modePosition > 0 && dynamicPosition > 0) {
      // aipp_config_path: {"aipp_mode":"dynamic"}
      OP_LOGD(TbeGetName(op).c_str(), "AippInfer, aipp dynamic mode");
      const GeTensor* params_tensor_p = OpDescUtils::GetInputConstData(op, 1);
      if (params_tensor_p != nullptr) {
        // running state
        OP_LOGD(TbeGetName(op).c_str(), "AippInfer, running, do DynamicShapeInfershape");
        return DynamicShapeInfershape(op, params_tensor_p);
      }
    }

    // compiling state
    if (op.GetAttr("aipp_config_file_path", aipp_config_path) != GRAPH_SUCCESS) {
      OP_LOGE(TbeGetName(op).c_str(), "AippInfer, failed to get attr aipp_config_file_path");
      return GRAPH_FAILED;
    }
  }
  OP_LOGD(TbeGetName(op).c_str(), "AippInfer, aipp_config_path is %s", aipp_config_path.c_str());

  char resolved_file_path[PATH_MAX] = {0x00};
  if (realpath(aipp_config_path.c_str(), resolved_file_path) == nullptr) {
    OP_LOGE(TbeGetName(op).c_str(), "invalid insert op conf file path:%s.", aipp_config_path.c_str());
    return GRAPH_FAILED;
  }

  // protobuff message to json
  std::shared_ptr<domi::InsertNewOps> insert_op_conf_(new (std::nothrow) domi::InsertNewOps());
  if (insert_op_conf_ == nullptr) {
    OP_LOGE(TbeGetName(op).c_str(), "insert_op_conf_ is null!");
    return GRAPH_FAILED;
  }

  bool ret = GraphUtils::ReadProtoFromTextFile(aipp_config_path.c_str(), insert_op_conf_.get());
  if (!ret) {
    OP_LOGE(TbeGetName(op).c_str(), "Read AIPP conf file error!");
    return GRAPH_FAILED;
  }
  int64_t index = 0;
  op.GetAttr("current_aipp_index", index);
  OP_LOGD(TbeGetName(op).c_str(), "AippInfer, current_aipp_index is %ld", index);

  if (index >= insert_op_conf_->aipp_op_size()) {
    OP_LOGE(TbeGetName(op).c_str(), "current_aipp_index %ld is invalid", index);
    return GRAPH_FAILED;
  }
  ::domi::AippOpParams* aipp_op_params = insert_op_conf_->mutable_aipp_op(index);
  if (aipp_op_params == nullptr) {
    std::string err_msg = GetInputInvalidErrMsg("aipp_op_params");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
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
  op.SetAttr("aipp_config_path", aipp_config_json);

  if (aipp_op_params->aipp_mode() == ::domi::AippOpParams_AippMode_dynamic) {
    return DynamicModeInfershape(op, aipp_op_params);
  } else {
    return StaticInferShape(op, aipp_op_params);
  }
}

IMPLEMT_INFER_DATA_SLICE(Aipp, AippInferDataSlice) {
  OP_LOGI(TbeGetName(op).c_str(), "AippInferDataSlice start.");

  auto images_desc = op.GetInputDesc("images");
  auto input_format = images_desc.GetFormat();

  if (input_format != FORMAT_NHWC && input_format != FORMAT_NCHW && input_format != FORMAT_NC1HWC0_C04) {
    OP_LOGE(TbeGetName(op).c_str(), "aipp input format only support NCHW, NHWC, NC1HWC0_C04.");
    return GRAPH_FAILED;
  }

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  vector<vector<int64_t>> images_data_slice = {{}, {}, {}, {}};
  vector<vector<int64_t>> output_data_slice = {{}, {}, {}, {}, {}};
  GeTensorDescPtr tensor_desc_in = op_desc->MutableInputDesc("images");
  GeTensorDescPtr tensor_desc_out = op_desc->MutableOutputDesc("features");

  if (!ge::AttrUtils::GetListListInt(tensor_desc_out, ge::ATTR_NAME_DATA_SLICE, output_data_slice)) {
    OP_LOGI(TbeGetName(op).c_str(), "no data slice, use default as {{}, {}, {}, {}, {}}");
    return GRAPH_FAILED;
  }

  for (unsigned i = 0; i < output_data_slice.size(); i++) {
    if (output_data_slice[i].size() > 0) {
      if (output_data_slice[i].size() != 2) {
        OP_LOGE(TbeGetName(op).c_str(), "data slice format input size should be 2.");
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
          OP_LOGE(TbeGetName(op).c_str(), "images data_slice set failed.");
          return GRAPH_FAILED;
        }
      } else {
        OP_LOGI(TbeGetName(op).c_str(), "only support cut in n");
        return NOT_SUPPORT_SLICE;
      }
    }
  }

  OP_LOGI(TbeGetName(op).c_str(), "AippInferDataSlice success.");
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Aipp, AippInfer);
VERIFY_FUNC_REG(Aipp, AippVerify);
INFER_DATA_SLICE_FUNC_REG(Aipp, AippInferDataSlice);

COMMON_INFER_FUNC_REG(AippData, ELMTWISE_INFER_SHAPEANDTYPE("data", "out"));
}  // namespace ge
