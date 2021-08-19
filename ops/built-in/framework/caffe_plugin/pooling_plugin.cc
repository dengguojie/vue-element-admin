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
 * \file pooling_plugin.cpp
 * \brief
 */
#include <string>
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"
#include "../../op_proto/util/error_util.h"

namespace domi {
Status ParseParamsPooling(const Message* op_src, ge::Operator& op_dest) {
  OP_LOGI("Pooling", "--------------ParseParamsPooling  start---------------");

  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_src);

  if (layer == nullptr) {
    OP_LOGE("Pooling", "Dynamic cast op_src to LayerParameter failed");
    return FAILED;
  }

  const caffe::PoolingParameter& pooling_param = layer->pooling_param();

  // set gloable pooling default value
  bool global_pooling = false;
  if (pooling_param.has_global_pooling()) {
    global_pooling = pooling_param.global_pooling();
  }

  if (global_pooling) {
    if (pooling_param.has_kernel_size() || pooling_param.has_kernel_h() || pooling_param.has_kernel_w()) {
      ge::OpsInputShapeErrReport(op_dest.GetName(), "In the Global_pooling condition, Filter size cannot be specified",
                                 "Filter size", "specified");
      OP_LOGE("Pooling", "With Global_pooling: true Filter size cannot specified");
      return FAILED;
    }
  } else {
    // preserve original caffe logic to prevent rewrite errors
    bool condition =
        !pooling_param.has_kernel_size() != !(pooling_param.has_kernel_h() && pooling_param.has_kernel_w());
    if (!condition) {
      ge::OpsInputShapeErrReport(op_dest.GetName(), "Filter size is kernel_size or kernel_h and kernel_w",
                                 "kernel_size and kernel_h/kernel_w", "set both");
      OP_LOGE("Pooling", "Filter size is kernel_size OR kernel_h and kernel_w; not both");
      return FAILED;
    }

    // preserve original caffe logic to prevent rewrite errors
    condition = pooling_param.has_kernel_size() || (pooling_param.has_kernel_h() && pooling_param.has_kernel_w());
    if (!condition) {
      ge::OpsInputShapeErrReport(op_dest.GetName(), "For non-square filters, kernel_h and kernel_w are required",
                                 "kernel_size or kernel_h/kernel_w", "missed");
      OP_LOGE("Pooling", "For non-square filters both kernel_h and kernel_w are required.");
      return FAILED;
    }
  }

  // preserve original caffe logic to prevent rewrite errors
  bool condition = (!pooling_param.has_pad() && pooling_param.has_pad_h() && pooling_param.has_pad_w()) ||
                   (!pooling_param.has_pad_h() && !pooling_param.has_pad_w());
  if (!condition) {
    ge::OpsInputShapeErrReport(op_dest.GetName(), "set either pad or pad_h/pad_w should be specified",
                               "pad and pad_h/pad_w", "set both");
    OP_LOGE("Pooling", "pad is pad OR pad_h and pad_w are required.");
    return FAILED;
  }

  // preserve original caffe logic to prevent rewrite errors
  condition = ((!pooling_param.has_stride() && pooling_param.has_stride_h() && pooling_param.has_stride_w()) ||
               (!pooling_param.has_stride_h() && !pooling_param.has_stride_w()));
  if (!condition) {
    ge::OpsInputShapeErrReport(op_dest.GetName(), "set either stride or stride_h/stride_w should be specified",
                               "stride and stride_h/stride_w", "set both");
    OP_LOGE("Pooling", "Stride is stride OR stride_h and stride_w are required.");
    return FAILED;
  }

  // parse global_pooling
  if (global_pooling) {
    op_dest.SetAttr("global_pooling", true);
  } else {
    op_dest.SetAttr("global_pooling", false);
  }

  // parse round_mode to ceil_mode
  int64_t ceil_mode = 0;
  if (pooling_param.has_round_mode()) {
    ceil_mode = static_cast<int64_t>(pooling_param.round_mode());
  } else {
    ceil_mode = static_cast<int64_t>(0);
  }
  op_dest.SetAttr("ceil_mode", ceil_mode);

  int64_t kernel_h = 1;
  int64_t kernel_w = 1;
  if (global_pooling) {
    // default value will set at PoolingInferShape, because can't get input H and input W here
  } else {
    if (pooling_param.has_kernel_size()) {
      kernel_h = pooling_param.kernel_size();
      kernel_w = kernel_h;
    } else {
      kernel_h = pooling_param.kernel_h();
      kernel_w = pooling_param.kernel_w();
    }
  }

  if (kernel_h < 0) {
    ge::OpsAttrValueErrReport(op_dest.GetName(), "kernel_h", "not be zero",
                              to_string(kernel_h));
    OP_LOGE("Pooling", "Filter H dimensions cannot be zero.");
    return FAILED;
  }

  if (kernel_w < 0) {
    ge::OpsAttrValueErrReport(op_dest.GetName(), "kernel_w", "not be zero",
                              to_string(kernel_w));
    OP_LOGE("Pooling", "Filter W dimensions cannot be zero.");
    return FAILED;
  }

  std::vector<int64_t> vec_window;
  vec_window.push_back(kernel_h);
  vec_window.push_back(kernel_w);
  op_dest.SetAttr("window", (vec_window));

  int64_t pad_h = 0;
  int64_t pad_w = 0;
  if (!pooling_param.has_pad_h()) {
    pad_h = pooling_param.pad();
    pad_w = pad_h;
  } else {
    pad_h = pooling_param.pad_h();
    pad_w = pooling_param.pad_w();
  }

  std::vector<int64_t> vec_pad;
  vec_pad.push_back(pad_h);
  vec_pad.push_back(pad_h);
  vec_pad.push_back(pad_w);
  vec_pad.push_back(pad_w);
  op_dest.SetAttr("pad", (vec_pad));

  int64_t dilation = 1;
  std::vector<int64_t> vec_dilation;
  vec_dilation.push_back(dilation);
  vec_dilation.push_back(dilation);
  vec_dilation.push_back(dilation);
  vec_dilation.push_back(dilation);
  op_dest.SetAttr("dilation", (vec_dilation));

  int64_t stride_h = 1;
  int64_t stride_w = 1;
  if (!pooling_param.has_stride_h()) {
    stride_h = pooling_param.stride();
    stride_w = stride_h;
  } else {
    stride_h = pooling_param.stride_h();
    stride_w = pooling_param.stride_w();
  }
  std::vector<int64_t> vec_stride;
  vec_stride.push_back(stride_h);
  vec_stride.push_back(stride_w);
  op_dest.SetAttr("stride", (vec_stride));

  if (global_pooling) {
    // preserve original caffe logic to prevent rewrite errors
    bool conditionGloable = (pad_h == 0) && (pad_w == 0) && (stride_h == 1) && (stride_w == 1);
    if (!conditionGloable) {
      ge::OpsInputShapeErrReport(op_dest.GetName(), "In the Global_pooling condition, only pad = 0 and stride = 1",
                                 "pad", to_string(pad_h) + ", stride is " + to_string(stride_h));
      OP_LOGE("Pooling", "With Global_pooling: true; only pad = 0 and stride = 1");
      return FAILED;
    }
  }

  // parse pool to mode
  int64_t mode = 0;
  if (pooling_param.has_pool()) {
    mode = static_cast<int64_t>(pooling_param.pool());
  }
  op_dest.SetAttr("mode", mode);

  if (pad_h != 0 || pad_w != 0) {
    if ((mode != 0) && (mode != 1)) {
      ge::OpsInputShapeErrReport(op_dest.GetName(), "Padding implemented only for average and max pooling",
                                 "mode", to_string(mode));
      OP_LOGE("Pooling", "Padding implemented only for average and max pooling.");
      return FAILED;
    }

    if (pad_h > kernel_h) {
      ge::OpsAttrValueErrReport(op_dest.GetName(), "pad_h", "less than kernel_h",
                                to_string(pad_h));
      OP_LOGE("Pooling", "pad_h should less than kernel_h.");
      return FAILED;
    }

    if (pad_w > kernel_w) {
      ge::OpsAttrValueErrReport(op_dest.GetName(), "pad_w", "less than kernel_w",
                                to_string(pad_w));
      OP_LOGE("Pooling", "pad_w should less than kernel_w.");
      return FAILED;
    }
  }
  OP_LOGI("Pooling", "--------------ParseParamsPooling  end---------------");

  return SUCCESS;
}

REGISTER_CUSTOM_OP("Pooling")
    .FrameworkType(CAFFE)
    .OriginOpType("Pooling")
    .ParseParamsFn(ParseParamsPooling)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
