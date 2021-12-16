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
 * \file caffe_deconv_plugin.cpp
 * \brief
 */
#include <string>
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"
#include "../../op_proto/util/axis_util.h"
#include "../../op_proto/util/error_util.h"

namespace domi {
static bool SetPads(const caffe::ConvolutionParameter& convParam, const ge::AscendString& op_name, ge::Operator& op) {
  const int MAX_PAD_SIZE = 2;
  std::vector<int64_t> vec;
  const int pSize = convParam.pad_size();
  const int kDefaultPad = 0;
  int64_t pad[2] = {kDefaultPad, kDefaultPad};
  if (convParam.has_pad_h() || convParam.has_pad_w()) {
    if (pSize != 0) {
      ge::OpsConvSetAttrErrReport(op_name.GetString(), "pad", "pad_h/w");
      OP_LOGE(op_name.GetString(), "set either pad or pad_h/w, not both.");
      return false;
    }
    pad[0] = convParam.pad_h();
    pad[1] = convParam.pad_w();
  } else {
    if (pSize == 1 || pSize == 2) {
      for (size_t i = 0; i < 2; i++) {
        pad[i] = convParam.pad((pSize == 1) ? 0 : i);
      }
    } else if (pSize != 0) {
      ge::OpsConvAttrValueErrReport(op_name.GetString(), "pad_size", "[0,1," + to_string(MAX_PAD_SIZE) + "]",
                                    to_string(pSize));
      OP_LOGE(op_name.GetString(), "pad size is invalid, actual is: %d.", pSize);
      return false;
    }
  }
  vec.push_back(pad[0]);
  vec.push_back(pad[0]);
  vec.push_back(pad[1]);
  vec.push_back(pad[1]);
  op.SetAttr("pads", (vec));
  return true;
}

static bool SetStrides(const caffe::ConvolutionParameter& convParam, const ge::AscendString& op_name,
                       ge::Operator& op) {
  const int MAX_STRIDE_SIZE = 2;
  std::vector<int64_t> vec;
  const int sSize = convParam.stride_size();
  const int kDefaultStride = 1;
  int64_t stride[2] = {kDefaultStride, kDefaultStride};
  if (convParam.has_stride_h() || convParam.has_stride_w()) {
    if (sSize != 0) {
      ge::OpsConvSetAttrErrReport(op_name.GetString(), "stride", "stride_h/w");
      OP_LOGE(op_name.GetString(), "set either stride or stride_h/w, not both");
      return false;
    }
    stride[0] = convParam.stride_h();
    stride[1] = convParam.stride_w();
  } else {
    if (sSize == 1 || sSize == 2) {
      for (size_t i = 0; i < 2; i++) {
        stride[i] = convParam.stride((sSize == 1) ? 0 : i);
      }
    } else if (sSize != 0) {
      ge::OpsConvAttrValueErrReport(op_name.GetString(), "stride_size", "[0,1," + to_string(MAX_STRIDE_SIZE) + "]",
                                    to_string(sSize));
      OP_LOGE(op_name.GetString(), "stride size is invalid, actual is: %d.", sSize);
      return false;
    }
  }
  vec.push_back(stride[0]);
  vec.push_back(stride[1]);
  op.SetAttr("strides", (vec));
  return true;
}

static bool SetDilations(const caffe::ConvolutionParameter& convParam, const ge::AscendString& op_name,
                         ge::Operator& op) {
  const int MAX_DILATION_SIZE = 2;
  std::vector<int64_t> vec;
  const int dSize = convParam.dilation_size();
  const int kDefaultDilation = 1;
  int64_t dilation[2] = {kDefaultDilation, kDefaultDilation};
  if (dSize == 1 || dSize == 2) {
    for (size_t i = 0; i < 2; i++) {
      dilation[i] = convParam.dilation((dSize == 1) ? 0 : i);
    }
  } else if (dSize != 0) {
    ge::OpsConvAttrValueErrReport(op_name.GetString(), "dilation_size", "[0,1," + to_string(MAX_DILATION_SIZE) + "]",
                                  to_string(dSize));
    OP_LOGE(op_name.GetString(), "dilation size is invalid, actual is: %d.", dSize);
    return false;
  }
  vec.push_back(1);
  vec.push_back(1);
  vec.push_back(dilation[0]);
  vec.push_back(dilation[1]);
  op.SetAttr("dilations", (vec));
  return true;
}

Status ParseParamsDeconv(const Message* op_src, ge::Operator& op) {
  ge::AscendString op_name;
  CHECK(op.GetName(op_name) != ge::GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return FAILED);

  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_src);
  if (layer == nullptr) {
    ge::OpsConvShapeErrReport(op_name.GetString(), "convert src op failed.");
    OP_LOGE(op_name.GetString(), "convert src op failed.");
    return FAILED;
  }

  if (layer->bottom_size() != 1) {
    ge::OpsConvAttrValueErrReport(op_name.GetString(), "Deconvolution layer bottom num", "1",
                                  to_string(layer->bottom_size()));
    OP_LOGE(op_name.GetString(), "Deconvolution layer bottom num(%d) must be 1", layer->bottom_size());
    return FAILED;
  }

  if (layer->top_size() != 1) {
    ge::OpsConvAttrValueErrReport(op_name.GetString(), "Deconvolution layer top num", "1",
                                  to_string(layer->top_size()));
    OP_LOGE(op_name.GetString(), "Deconvolution layer top num(%d) must be 1", layer->top_size());
    return FAILED;
  }

  const caffe::ConvolutionParameter& convParam = layer->convolution_param();

  if (!SetPads(convParam, op_name, op)) {
    ge::OpsConvShapeErrReport(op_name.GetString(), "Set pads failed.");
    OP_LOGE(op_name.GetString(), "set pads failed.");
    return FAILED;
  }
  if (!SetStrides(convParam, op_name, op)) {
    ge::OpsConvShapeErrReport(op_name.GetString(), "Set strides failed.");
    OP_LOGE(op_name.GetString(), "set strides failed.");
    return FAILED;
  }
  if (!SetDilations(convParam, op_name, op)) {
    ge::OpsConvShapeErrReport(op_name.GetString(), "Set dilations failed.");
    OP_LOGE(op_name.GetString(), "set dilations failed.");
    return FAILED;
  }

  if (convParam.has_group()) {
    uint32_t group = convParam.group();
    op.SetAttr("groups", group);
  }

  return SUCCESS;
}

REGISTER_CUSTOM_OP("Deconvolution")
    .FrameworkType(CAFFE)              // type: CAFFE, TENSORFLOW
    .OriginOpType("Deconvolution")     // name in caffe module
    .ParseParamsFn(ParseParamsDeconv)  // AutoMappingFn for Tensorflow, ParseParamsFn need to realize for caffe
    .ImplyType(ImplyType::TVM);
}  // namespace domi
