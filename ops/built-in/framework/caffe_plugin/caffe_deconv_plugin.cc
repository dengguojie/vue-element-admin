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
#include "error_util.h"

namespace domi {
static bool SetPads(const caffe::ConvolutionParameter& convParam, const ge::AscendString& op_name, ge::Operator& op) {
  const int kMaxPadSize = 2;
  std::vector<int64_t> vec;
  const int pSize = convParam.pad_size();
  const int kDefaultPad = 0;
  int64_t pad[2] = {kDefaultPad, kDefaultPad};
  if (convParam.has_pad_h() || convParam.has_pad_w()) {
    CHECK(pSize != 0,
          OP_LOGE(op_name.GetString(), "one of pad or pad_h/w needs to be set, not both."),
          return false);
    pad[0] = convParam.pad_h();
    pad[1] = convParam.pad_w();
  } else {
    if (pSize == 1 || pSize == 2) {
      for (size_t i = 0; i < kMaxPadSize; i++) {
        pad[i] = convParam.pad((pSize == 1) ? 0 : i);
      }
    } else if (pSize != 0) {
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
  std::vector<int64_t> vec;
  const int sSize = convParam.stride_size();
  const int kDefaultStride = 1;
  int64_t stride[2] = {kDefaultStride, kDefaultStride};
  if (convParam.has_stride_h() || convParam.has_stride_w()) {
    CHECK(sSize != 0,
          OP_LOGE(op_name.GetString(), "one of stride or stride_h/w needs to be set, not both."),
          return false);
    stride[0] = convParam.stride_h();
    stride[1] = convParam.stride_w();
  } else {
    if (sSize == 1 || sSize == 2) {
      for (size_t i = 0; i < 2; i++) {
        stride[i] = convParam.stride((sSize == 1) ? 0 : i);
      }
    } else if (sSize != 0) {
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
  std::vector<int64_t> vec;
  const int dSize = convParam.dilation_size();
  const int kDefaultDilation = 1;
  int64_t dilation[2] = {kDefaultDilation, kDefaultDilation};
  if (dSize == 1 || dSize == 2) {
    for (size_t i = 0; i < 2; i++) {
      dilation[i] = convParam.dilation((dSize == 1) ? 0 : i);
    }
  } else if (dSize != 0) {
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
  CHECK(layer == nullptr, OP_LOGE(op_name.GetString(), "failed to convert src op."), return FAILED);

  if (layer->bottom_size() != 1) {
    OP_LOGE(op_name.GetString(), "Deconvolution layer bottom num(%d) must be 1", layer->bottom_size());
    return FAILED;
  }

  if (layer->top_size() != 1) {
    OP_LOGE(op_name.GetString(), "Deconvolution layer top num(%d) must be 1", layer->top_size());
    return FAILED;
  }

  const caffe::ConvolutionParameter& convParam = layer->convolution_param();

  if (!SetPads(convParam, op_name, op)) {
    return FAILED;
  }
  if (!SetStrides(convParam, op_name, op)) {
    return FAILED;
  }
  if (!SetDilations(convParam, op_name, op)) {
    return FAILED;
  }

  if (convParam.has_group()) {
    uint32_t group = convParam.group();
    op.SetAttr("groups", group);
  }
  OP_LOGD(op_name.GetString(), "op[Deconv] caffe plugin parsed successfully");

  return SUCCESS;
}

REGISTER_CUSTOM_OP("Deconvolution")
    .FrameworkType(CAFFE)              // type: CAFFE, TENSORFLOW
    .OriginOpType("Deconvolution")     // name in caffe module
    .ParseParamsFn(ParseParamsDeconv)  // AutoMappingFn for Tensorflow, ParseParamsFn need to realize for caffe
    .ImplyType(ImplyType::TVM);
}  // namespace domi
