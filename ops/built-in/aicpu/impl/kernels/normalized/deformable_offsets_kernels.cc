/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "deformable_offsets_kernels.h"

#include <memory.h>
#include <cfloat>
#include <ctime>
#include <random>

#include "cpu_types.h"
#include "log.h"
#include "status.h"

namespace {
const char *DEFORMABLE_OFFSETS = "DeformableOffsets";
}

namespace aicpu {
uint32_t DeformableOffsetsCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("DeformableOffsetsCpuKernel::Compute start!! ");

  uint32_t res = GetInputParam(ctx);
  if (res != KERNEL_STATUS_OK) {
    return res;
  }

  res = ParseInputParam();
  if (res != KERNEL_STATUS_OK) {
    return res;
  }

  res = CheckInputParam();
  if (res != KERNEL_STATUS_OK) {
    return res;
  }

  DataType x_data_type = x_tensor->GetDataType();
  if (x_data_type != DT_FLOAT16) {
    KERNEL_LOG_ERROR(
        "The current x input type only supports DT_FLOAT16, x data type is %d.",
        x_data_type);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  DataType offsets_data_type = offsets_tensor->GetDataType();

  auto x_data = reinterpret_cast<Eigen::half *>(x_tensor->GetData());
  KERNEL_CHECK_NULLPTR(x_data, KERNEL_STATUS_PARAM_INVALID,
                       "input x get data failed.");
  auto offsets_data_ptr = offsets_tensor->GetData();
  KERNEL_CHECK_NULLPTR(offsets_data_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "input offset get data failed.");
  auto y_data = reinterpret_cast<Eigen::half *>(y_tensor->GetData());
  KERNEL_CHECK_NULLPTR(y_data, KERNEL_STATUS_PARAM_INVALID,
                       "input y get data failed.");

  switch (offsets_data_type) {
    case DT_FLOAT16: {
      auto offsets_data = reinterpret_cast<Eigen::half *>(offsets_data_ptr);
      return DoCompute<Eigen::half>(x_data, offsets_data, y_data);
    }
    case DT_FLOAT: {
      auto offsets_data = reinterpret_cast<float *>(offsets_data_ptr);
      return DoCompute<float>(x_data, offsets_data, y_data);
    }
    default: {
      KERNEL_LOG_ERROR("input offsets datatype error, offsets data type is %d.",
                       offsets_data_type);
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  KERNEL_LOG_INFO("DeformableOffsetsCpuKernel::Compute end!! ");
  return KERNEL_STATUS_OK;
}

uint32_t DeformableOffsetsCpuKernel::GetInputParam(CpuKernelContext &ctx) {
  // get x
  x_tensor = ctx.Input(0);
  if (x_tensor == nullptr) {
    KERNEL_LOG_ERROR("get input:x failed");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // get offsets
  offsets_tensor = ctx.Input(1);
  if (offsets_tensor == nullptr) {
    KERNEL_LOG_ERROR("get input:offsets failed");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // get attr: strides
  AttrValue *strides = ctx.GetAttr("strides");
  KERNEL_CHECK_NULLPTR(strides, KERNEL_STATUS_PARAM_INVALID,
                       "get attr:strides failed.");
  stride_list_ = strides->GetListInt();

  // get attr: pads
  AttrValue *pads = ctx.GetAttr("pads");
  KERNEL_CHECK_NULLPTR(pads, KERNEL_STATUS_PARAM_INVALID,
                       "get attr:pads failed.");
  pads_list_ = pads->GetListInt();

  // get attr: ksize
  AttrValue *ksize = ctx.GetAttr("ksize");
  KERNEL_CHECK_NULLPTR(ksize, KERNEL_STATUS_PARAM_INVALID,
                       "get attr:ksize failed.");
  ksize_list_ = ksize->GetListInt();

  // get attr: dilations
  AttrValue *dilations = ctx.GetAttr("dilations");
  KERNEL_CHECK_NULLPTR(dilations, KERNEL_STATUS_PARAM_INVALID,
                       "get attr:dilations failed.");
  dilation_list_ = dilations->GetListInt();

  // get attr: data_format
  AttrValue *xDataFormat = ctx.GetAttr("data_format");
  KERNEL_CHECK_NULLPTR(xDataFormat, KERNEL_STATUS_PARAM_INVALID,
                       "get attr:data_format failed.");
  data_format_ = xDataFormat->GetString();

  // get attr: deformable_groups
  AttrValue *deformGroups = ctx.GetAttr("deformable_groups");
  KERNEL_CHECK_NULLPTR(deformGroups, KERNEL_STATUS_PARAM_INVALID,
                       "get attr:deformable_groups failed.");
  deformable_groups_ = deformGroups->GetInt();

  y_tensor = ctx.Output(0);
  if (y_tensor == nullptr) {
    KERNEL_LOG_ERROR("get output:y failed");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t DeformableOffsetsCpuKernel::ParseInputParam() {
  const int32_t STRIDE_SIZE_LIMIT = 2;
  const int32_t PAD_SIZE_LIMIT = 4;
  const int32_t KSIZE_SIZE_LIMIT = 2;

  if (stride_list_.size() == STRIDE_SIZE_LIMIT) {
    if (stride_list_[0] <= 0 || stride_list_[1] <= 0) {
      KERNEL_LOG_ERROR(
          "input strideList[0]:%d or stride_list_[1]:%d value error, at least "
          "1.",
          stride_list_[0], stride_list_[1]);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    param_.strideH = stride_list_[0];
    param_.strideW = stride_list_[1];
  } else {
    KERNEL_LOG_ERROR("input strideList's size error, strideList's size:%d.",
                     stride_list_.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (pads_list_.size() == PAD_SIZE_LIMIT) {
    param_.padUp = pads_list_[0];
    param_.padDown = pads_list_[1];
    param_.padLeft = pads_list_[2];
    param_.padRight = pads_list_[3];
  } else {
    KERNEL_LOG_ERROR("input padsList's size error, padsList's size:%d.",
                     pads_list_.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ksize_list_.size() == KSIZE_SIZE_LIMIT) {
    param_.ksizeX = ksize_list_[0];
    param_.ksizeY = ksize_list_[1];
  } else {
    KERNEL_LOG_ERROR("input ksizeList's size error, ksizeList's size:%d.",
                     ksize_list_.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto xShape = x_tensor->GetTensorShape();
  if (xShape == nullptr) {
    KERNEL_LOG_ERROR("get xShape failed");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto offsetsShape = offsets_tensor->GetTensorShape();
  if (offsetsShape == nullptr) {
    KERNEL_LOG_ERROR("get offsetsShape failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto yShape = y_tensor->GetTensorShape();
  if (yShape == nullptr) {
    KERNEL_LOG_ERROR("get y_tensor failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if ("NCHW" == data_format_) {
    x_.batch = xShape->GetDimSize(0);
    x_.channels = xShape->GetDimSize(1);
    x_.height = xShape->GetDimSize(2);
    x_.width = xShape->GetDimSize(3);
    param_.dilationsH = dilation_list_[2];
    param_.dilationsW = dilation_list_[3];
  } else if ("NHWC" == data_format_) {
    x_.batch = xShape->GetDimSize(0);
    x_.height = xShape->GetDimSize(1);
    x_.width = xShape->GetDimSize(2);
    x_.channels = xShape->GetDimSize(3);
    param_.dilationsH = dilation_list_[1];
    param_.dilationsW = dilation_list_[2];
  } else {
    KERNEL_LOG_ERROR("input dayta_format should be 'NCHW' or 'NHWC'.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  offset_.batch = offsetsShape->GetDimSize(0);
  offset_.channels = offsetsShape->GetDimSize(1);
  offset_.height = offsetsShape->GetDimSize(2);
  offset_.width = offsetsShape->GetDimSize(3);

  y_.batch = yShape->GetDimSize(0);
  y_.channels = yShape->GetDimSize(1);
  y_.height = yShape->GetDimSize(2);
  y_.width = yShape->GetDimSize(3);

  return KERNEL_STATUS_OK;
}

uint32_t DeformableOffsetsCpuKernel::CheckInputParam() {
  if (param_.padUp > 1) {
    param_.padUp = 1;
  }
  if (param_.padDown > 1) {
    param_.padDown = 1;
  }
  if (param_.padRight > 1) {
    param_.padRight = 1;
  }
  if (param_.padLeft > 1) {
    param_.padLeft = 1;
  }

  if ((0 == param_.ksizeX % 2) || (0 == param_.ksizeY % 2)) {
    KERNEL_LOG_ERROR(
        "ksize does not support even numbers temporarily, ksizeX:%d, "
        "ksizeY:%d.",
        param_.ksizeX, param_.ksizeY);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (deformable_groups_ == 0 || 0 != x_.channels % deformable_groups_) {
    KERNEL_LOG_ERROR("input featuremap can not Divide deformableGroups:%d.",
                     deformable_groups_);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (offset_.channels !=
      deformable_groups_ * ksize_list_[0] * ksize_list_[1] * 3) {
    KERNEL_LOG_ERROR("offsets_channels should be deformable_group*k_x*k_y*3.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (x_.batch != offset_.batch || x_.height != offset_.height ||
      x_.width != offset_.width) {
    KERNEL_LOG_ERROR(
        "input x's or offset's shape error, x's shape is [%d %d %d %d], "
        "offset's shape is[%d %d %d %d].",
        x_.batch, x_.channels, x_.height, x_.width, offset_.batch,
        offset_.channels, offset_.height, offset_.width);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t DeformableOffsetsCpuKernel::DoCompute(Eigen::half *inputDataX,
                                               T *inputDataOffsets,
                                               Eigen::half *inputDataY) {
  const int kOffsetIndexMax =
      offset_.batch * offset_.channels * offset_.height * offset_.width;
  const int kTotalOffsetDataTypes = 3;
  const int kOffsetStrideIndex = kTotalOffsetDataTypes * param_.ksizeX *
                                 param_.ksizeY * offset_.height * offset_.width;
  for (int b = 0; b < y_.batch; b++) {
    Eigen::half *inputX = inputDataX + b * x_.channels * x_.height * x_.width;
    Eigen::half *inputY = inputDataY + b * y_.channels * y_.height * y_.width;
    for (int g = 0; g < deformable_groups_; g++) {
      int offset_index = b * offset_.channels * offset_.height * offset_.width +
                         g * kOffsetStrideIndex;
      if (offset_index < 0 || offset_index >= kOffsetIndexMax) {
        KERNEL_LOG_ERROR("calculation offsets index over flow.");
        return KERNEL_STATUS_INNER_ERROR;
      }
      T *input_offset = inputDataOffsets + offset_index;
      int64_t groupLen = x_.channels / deformable_groups_;
      for (int l = 0; l < groupLen; l++) {
        int currentAxis = g * groupLen + l;
        uint32_t res =
            ComputePosition<T>(inputX, input_offset, inputY, currentAxis);
        if (res != KERNEL_STATUS_OK) {
          return res;
        }
      }
    }
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t DeformableOffsetsCpuKernel::ComputePosition(const Eigen::half *inputX,
                                                     const T *input_offset,
                                                     Eigen::half *inputY,
                                                     int currentAxis) {
  // formula : (ksizeX + 1) * (dilationsW + 1) + 1
  const int kSizeDilX = (param_.ksizeX - 1) * (param_.dilationsW + 1) + 1;
  const int kSizeDilY = (param_.ksizeY - 1) * (param_.dilationsH + 1) + 1;

  // formula : (width + padLeft + padRight - ksizeX) / strideW + 1
  const int kxPoints =
      ((x_.width + param_.padLeft + param_.padRight - kSizeDilX) /
       param_.strideW) +
      1;
  const int kyPoints =
      ((x_.height + param_.padUp + param_.padDown - kSizeDilY) /
       param_.strideH) +
      1;

  KERNEL_LOG_DEBUG(
      "kxPoints = %d, kyPoints = %d, kSizeDilX = %d, kSizeDilY = %d", kxPoints,
      kyPoints, kSizeDilX, kSizeDilY);
  for (int yDst = 0; yDst < kyPoints; yDst++) {
    for (int xDst = 0; xDst < kxPoints; xDst++) {
      // formula xSrc = xDst * strideW + (ksize_dx - 1) / 2 - padLeft
      int xSrc = xDst * param_.strideW + (kSizeDilX - 1) / 2 - param_.padLeft;
      int ySrc = yDst * param_.strideH + (kSizeDilY - 1) / 2 - param_.padUp;
      uint32_t res = ComputeResult<T>(inputX, input_offset, inputY, xSrc, ySrc,
                                      currentAxis);
      if (res != KERNEL_STATUS_OK) {
        return res;
      }
    }
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t DeformableOffsetsCpuKernel::ComputeResult(const Eigen::half *inputX,
                                                   const T *input_offset,
                                                   Eigen::half *inputY,
                                                   int xSrc, int ySrc,
                                                   int currentAxis) {
  const int kOffsetIndexMax =
      offset_.batch * offset_.channels * offset_.height * offset_.width;
  const int yIndexMax = y_.batch * y_.channels * y_.height * y_.width;
  const int offsetStrideX = 0;
  const int offsetStrideY =
      param_.ksizeX * param_.ksizeY * offset_.height * offset_.width;
  const int offsetStrideDelta =
      2 * param_.ksizeX * param_.ksizeY * offset_.height * offset_.width;
  Eigen::half *currentY = inputY + currentAxis * y_.height * y_.width;
  for (int kx = 0; kx < param_.ksizeX; kx++) {
    int outH = ySrc * param_.ksizeY + kx;
    for (int ky = 0; ky < param_.ksizeY; ky++) {
      int outW = xSrc * param_.ksizeX + ky;
      int mapIndex = kx * param_.ksizeX + ky;
      int mapStride = mapIndex * offset_.height * offset_.width +
                      ySrc * offset_.height + xSrc;

      int offsetXIndex = mapStride + offsetStrideX;
      if (offsetXIndex < 0 || offsetXIndex >= kOffsetIndexMax) {
        KERNEL_LOG_ERROR("calculation offsets index x over flow.");
        return KERNEL_STATUS_INNER_ERROR;
      }
      int offsetYIndex = mapStride + offsetStrideY;
      if (offsetYIndex < 0 || offsetYIndex >= kOffsetIndexMax) {
        KERNEL_LOG_ERROR("calculation offsets index y over flow.");
        return KERNEL_STATUS_INNER_ERROR;
      }
      int offsetDeltaIndex = mapStride + offsetStrideDelta;
      if (offsetDeltaIndex < 0 || offsetDeltaIndex >= kOffsetIndexMax) {
        KERNEL_LOG_ERROR("calculation offsets index delta over flow.");
        return KERNEL_STATUS_INNER_ERROR;
      }

      T offsetsValX = input_offset[offsetXIndex];
      T offsetsValY = input_offset[offsetYIndex];
      T offsetsValDelta = input_offset[offsetDeltaIndex];
      float offsetW =
          static_cast<float>(xSrc) + static_cast<float>(offsetsValX);
      float offsetH =
          static_cast<float>(ySrc) + static_cast<float>(offsetsValY);
      Eigen::half xVal;
      KERNEL_LOG_DEBUG("[currentAxis outH outW]= [%d %d %d]", currentAxis, outH,
                       outW);
      uint32_t res =
          BilinearInterpolate(xVal, inputX, currentAxis, offsetH, offsetW);
      if (res != KERNEL_STATUS_OK) {
        return res;
      }
      int yIndex = outH * y_.width + outW;
      if (yIndex < 0 || yIndex >= yIndexMax) {
        KERNEL_LOG_ERROR(
            "calculation y index over flow, outH:%d, width:%lld, outW:%d.",
            outH, y_.width, outW);
        return KERNEL_STATUS_INNER_ERROR;
      }
      currentY[yIndex] = static_cast<Eigen::half>(
          static_cast<float>(xVal) * static_cast<float>(offsetsValDelta));
    }
  }

  return KERNEL_STATUS_OK;
}

uint32_t DeformableOffsetsCpuKernel::BilinearInterpolate(Eigen::half &out,
                                                         const Eigen::half *in,
                                                         int c_axis, float h,
                                                         float w) {
  if (h <= -1 || x_.height <= h || w <= -1 || x_.width <= w) {
    KERNEL_LOG_ERROR("index overflow h:%f, w:%f", h, w);
    return KERNEL_STATUS_INNER_ERROR;
  }

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h - h_low;
  float lw = w - w_low;
  float hh = 1 - lh;
  float hw = 1 - lw;

  int v1_idx = 0;
  int v2_idx = 0;
  int v3_idx = 0;
  int v4_idx = 0;

  // Calculate bilinear interpolation coordinate points according to the input
  // format type
  if ("NCHW" == data_format_) {
    int idx = c_axis * x_.height * x_.width;
    v1_idx = idx + h_low * x_.width + w_low;
    v2_idx = idx + h_low * x_.width + w_high;
    v3_idx = idx + h_high * x_.width + w_low;
    v4_idx = idx + h_high * x_.width + w_high;
  } else {
    v1_idx = h_low * x_.width * x_.channels + w_low * x_.channels + c_axis;
    v2_idx = h_low * x_.width * x_.channels + w_high * x_.channels + c_axis;
    v3_idx = h_high * x_.width * x_.channels + w_low * x_.channels + c_axis;
    v4_idx = h_high * x_.width * x_.channels + w_high * x_.channels + c_axis;
  }

  Eigen::half v1 = static_cast<Eigen::half>(0);
  if (h_low >= 0 && w_low >= 0) {
    v1 = in[v1_idx];
  }
  Eigen::half v2 = static_cast<Eigen::half>(0);
  if (h_low >= 0 && w_high <= x_.width - 1) {
    v2 = in[v2_idx];
  }
  Eigen::half v3 = static_cast<Eigen::half>(0);
  if (h_high <= x_.height - 1 && w_low >= 0) {
    v3 = in[v3_idx];
  }
  Eigen::half v4 = static_cast<Eigen::half>(0);
  if (h_high <= x_.height - 1 && w_high <= x_.width - 1) {
    v4 = in[v4_idx];
  }

  Eigen::half w1 = static_cast<Eigen::half>(hh * hw);
  Eigen::half w2 = static_cast<Eigen::half>(hh * lw);
  Eigen::half w3 = static_cast<Eigen::half>(lh * hw);
  Eigen::half w4 = static_cast<Eigen::half>(lh * lw);

  out = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(DEFORMABLE_OFFSETS, DeformableOffsetsCpuKernel);
}  // namespace aicpu