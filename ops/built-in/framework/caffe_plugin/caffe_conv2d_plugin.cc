/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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
 * \file caffe_conv2d_plugin.cpp
 * \brief
 */
#include <string>
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"
#include "../../op_proto/util/error_util.h"

namespace domi {
const uint32_t CONV2D_AXIS_NUM = 4;
const uint32_t MAX_PAD_SIZE = 2;
const uint32_t MAX_STRIDE_SIZE = 2;
const uint32_t MAX_DILATION_SIZE = 2;
const uint32_t MAX_KERNEL_SIZE = 2;

/*!
  * @brief Get covolution pad params from caffe proto and convert to tbe conv2d ir
  * @param conv_param the source conv param info from caffe.
  * @param op the dest GE op.
  * @return status whether this operation success.
  */
static bool SetPads(const caffe::ConvolutionParameter& conv_param, ge::Operator& op) {
  const int kDefaultPad = 0;
  int64_t pad[MAX_PAD_SIZE] = {kDefaultPad, kDefaultPad};
  const int kPadSize = conv_param.pad_size();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    if (kPadSize != 0) {
      OP_LOGE(TbeGetName(op), "kPadSize != 0");
      return false;
    }
    pad[0] = conv_param.pad_h();
    pad[1] = conv_param.pad_w();
  } else {
    if (kPadSize == 1 || kPadSize == MAX_PAD_SIZE) {
      for (size_t i = 0; i < MAX_PAD_SIZE; i++) {
        pad[i] = conv_param.pad((kPadSize == 1) ? 0 : i);
      }
    } else if (kPadSize != 0) {
      OP_LOGE(TbeGetName(op).c_str(), "pad size [%d] is not supported.", kPadSize);
      return false;
    }
  }
  std::vector<int64_t> pad_list;
  pad_list.push_back(pad[0]);
  pad_list.push_back(pad[0]);
  pad_list.push_back(pad[1]);
  pad_list.push_back(pad[1]);
  op.SetAttr("pads", (pad_list));

  return true;
}

/*!
  * @brief Get covolution stride params from caffe proto and convert to tbe conv2d
  * @param conv_param the source conv param info from caffe.
  * @param op the dest GE op.
  * @return status whether this operation success.
  */
static bool SetStrides(const caffe::ConvolutionParameter& conv_param, ge::Operator& op) {
  const int kDefaultStride = 1;
  int64_t stride[MAX_STRIDE_SIZE] = {kDefaultStride, kDefaultStride};
  const int kStrideSize = conv_param.stride_size();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    if (kStrideSize != 0) {
      OP_LOGE(TbeGetName(op), "kStrideSize != 0");
      return false;
    }
    stride[0] = conv_param.stride_h();
    stride[1] = conv_param.stride_w();
  } else {
    if (kStrideSize == 1 || kStrideSize == MAX_STRIDE_SIZE) {
      for (size_t i = 0; i < MAX_STRIDE_SIZE; i++) {
        stride[i] = conv_param.stride((kStrideSize == 1) ? 0 : i);
      }
    } else if (kStrideSize != 0) {
      OP_LOGE(TbeGetName(op).c_str(), "stride size [%d] is not supported.", kStrideSize);
      return false;
    }
  }
  std::vector<int64_t> stride_list;
  stride_list.push_back(1);
  stride_list.push_back(1);
  stride_list.push_back(stride[0]);
  stride_list.push_back(stride[1]);
  op.SetAttr("strides", (stride_list));

  return true;
}

/*!
  * @brief Get covolution dilation params from caffe proto and convert to tbe conv2d
  * @param conv_param the source conv param info from caffe.
  * @param op the dest GE op.
  * @return status whether this operation success.
  */
static bool SetDilations(const caffe::ConvolutionParameter& conv_param, ge::Operator& op) {
  const int kDefaultDilation = 1;
  const int kDilationSize = conv_param.dilation_size();
  int64_t dilation[MAX_DILATION_SIZE] = {kDefaultDilation, kDefaultDilation};
  if (kDilationSize == 1 || kDilationSize == MAX_DILATION_SIZE) {
    for (size_t i = 0; i < MAX_DILATION_SIZE; i++) {
      dilation[i] = conv_param.dilation((kDilationSize == 1) ? 0 : i);
    }
  } else if (kDilationSize != 0) {
    OP_LOGE(TbeGetName(op).c_str(), "dilation size [%d] is not supported.", kDilationSize);
    return false;
  }
  std::vector<int64_t> dilation_list;
  dilation_list.push_back(1);
  dilation_list.push_back(1);
  dilation_list.push_back(dilation[0]);
  dilation_list.push_back(dilation[1]);
  op.SetAttr("dilations", (dilation_list));

  return true;
}

/*!
  * @brief Check input parameters that are illegal or not applicable to 2D convolution
  * @param conv_param the source conv param info from caffe.
  * @param op the dest GE op.
  * @return status whether this operation success.
  */
static bool ProcSpecParams(const caffe::ConvolutionParameter& conv_param, ge::Operator& op) {
  int num_output = conv_param.num_output();
  if (num_output < 1) {
    OP_LOGE(TbeGetName(op).c_str(), "num of output should be positive.");
    return false;
  }

  int group = conv_param.group();
  if (group < 1 || (group != 0 && num_output % group != 0)) {
    OP_LOGE(TbeGetName(op).c_str(), "group should be positive and divisible by num of output.");
    return false;
  }
  op.SetAttr("groups", static_cast<int64_t>(group));

  const int kKernelSize = conv_param.kernel_size_size();
  int kernel[MAX_KERNEL_SIZE] = {0, 0};
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    if (kKernelSize != 0) {
      OP_LOGE(TbeGetName(op).c_str(), "set kernel_size or kernel_h/w, not both.");
      return false;
    }
    kernel[0] = conv_param.kernel_h();
    kernel[1] = conv_param.kernel_w();
  } else {
    if (kKernelSize == 1 || kKernelSize == MAX_KERNEL_SIZE) {
      for (size_t i = 0; i < MAX_KERNEL_SIZE; i++) {
        kernel[i] = conv_param.kernel_size((kKernelSize == 1) ? 0 : i);
      }
    } else {
      OP_LOGE(TbeGetName(op).c_str(), "kernel size [%d] is not supported.", kKernelSize);
      return false;
    }
  }

  for (size_t i = 0; i < MAX_KERNEL_SIZE; i++) {
    if (kernel[i] < 1) {
      OP_LOGE(TbeGetName(op).c_str(), "kernel dimensions should be positive.");
      return false;
    }
  }

  int channelAxis = conv_param.axis();
  if ((channelAxis + CONV2D_AXIS_NUM) % CONV2D_AXIS_NUM != 1) {
    OP_LOGE(TbeGetName(op).c_str(),
            "only support 2D convolution and C-channel on the second"
            " axis.");
    return false;
  }

  bool forceNdIm2col = conv_param.force_nd_im2col();
  if (forceNdIm2col) {
    OP_LOGE(TbeGetName(op).c_str(), "only support 2D convolution.");
    return false;
  }

  return true;
}

/*!
  * @brief Replace GE ParseParams fuction to process graph conv2d node attrs
  * @param op_src the source op info from caffe.
  * @param op the dest GE op.
  * @return status whether this operation success.
  */
static Status ParseParamsConv2D(const Message* op_src, ge::Operator& op) {
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_src);
  if (layer == nullptr) {
    OP_LOGE(TbeGetName(op).c_str(), "convert src op failed.");
    return FAILED;
  }

  if (layer->bottom_size() != 1) {
    OP_LOGE(TbeGetName(op).c_str(), "Convolution layer bottom num(%d) must be 1", layer->bottom_size());
    return FAILED;
  }

  if (layer->top_size() != 1) {
    OP_LOGE(TbeGetName(op).c_str(), "Convolution layer top num(%d) must be 1", layer->top_size());
    return FAILED;
  }

  const caffe::ConvolutionParameter& conv_param = layer->convolution_param();

  if (!(ProcSpecParams(conv_param, op) && SetPads(conv_param, op) && SetStrides(conv_param, op) &&
        SetDilations(conv_param, op))) {
    OP_LOGE(TbeGetName(op).c_str(), "Convolution layer set spec params/pads/strides/dilation failed.");
    return FAILED;
  }

  return SUCCESS;
}

REGISTER_CUSTOM_OP("Conv2D")
    .FrameworkType(CAFFE)              // type: CAFFE, TENSORFLOW
    .OriginOpType({"Convolution", "DepthwiseConvolution"})       // name in caffe module
    .ParseParamsFn(ParseParamsConv2D)  // AutoMappingFn for Tensorflow,
    // ParseParamsFn need to realize for caffe
    .ImplyType(ImplyType::TVM);
}  // namespace domi
