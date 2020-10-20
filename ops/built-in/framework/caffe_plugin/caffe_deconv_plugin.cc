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

/*!
 * \file caffe_deconv_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"
#include "common/util/error_manager/error_manager.h"

namespace domi {
static bool SetPads(const caffe::ConvolutionParameter& convParam, ge::Operator& op) {
  const int MAX_PAD_SIZE = 2;
  std::vector<int64_t> vec;
  const int pSize = convParam.pad_size();
  const int kDefaultPad = 0;
  int64_t pad[2] = {kDefaultPad, kDefaultPad};
  if (convParam.has_pad_h() || convParam.has_pad_w()) {
    if (pSize != 0) {
      map<string, string> err_map;
      err_map["op_name"] = op.GetName();
      err_map["param1_name"] = "pad";
      err_map["param2_name"] = "pad_h/w";
      std::string report_error_code = "E50057";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      OP_LOGE(op.GetName().c_str(), "set either pad or pad_h/w, not both.");
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
      map<string, string> err_map;
      err_map["op_name"] = op.GetName();
      err_map["param_name"] = "pad_size";
      err_map["expected_value"] = "[0,1," + to_string(MAX_PAD_SIZE) + "]";
      err_map["input_value"] = to_string(pSize);
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      OP_LOGE(op.GetName().c_str(), "pad size is invalid, actual is: %d.", pSize);
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

static bool SetStrides(const caffe::ConvolutionParameter& convParam, ge::Operator& op) {
  const int MAX_STRIDE_SIZE = 2;
  std::vector<int64_t> vec;
  const int sSize = convParam.stride_size();
  const int kDefaultStride = 1;
  int64_t stride[2] = {kDefaultStride, kDefaultStride};
  if (convParam.has_stride_h() || convParam.has_stride_w()) {
    if (sSize != 0) {
      map<string, string> err_map;
      err_map["op_name"] = op.GetName();
      err_map["param1_name"] = "stride";
      err_map["param2_name"] = "stride_h/w";
      std::string report_error_code = "E50057";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      OP_LOGE(op.GetName().c_str(), "set either stride or stride_h/w, not both");
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
      map<string, string> err_map;
      err_map["op_name"] = op.GetName();
      err_map["param_name"] = "stride_size";
      err_map["expected_value"] = "[0,1," + to_string(MAX_STRIDE_SIZE) + "]";
      err_map["input_value"] = to_string(sSize);
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      OP_LOGE(op.GetName().c_str(), "stride size is invalid, actual is: %d.", sSize);
      return false;
    }
  }
  vec.push_back(stride[0]);
  vec.push_back(stride[1]);
  op.SetAttr("strides", (vec));
  return true;
}

static bool SetDilations(const caffe::ConvolutionParameter& convParam, ge::Operator& op) {
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
    map<string, string> err_map;
    err_map["op_name"] = op.GetName();
    err_map["param_name"] = "dilation_size";
    err_map["expected_value"] = "[0,1," + to_string(MAX_DILATION_SIZE) + "]";
    err_map["input_value"] = to_string(dSize);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    OP_LOGE(op.GetName().c_str(), "dilation size is invalid, actual is: %d.", dSize);
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
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_src);
  if (layer == nullptr) {
    map<string, string> err_map;
    err_map["op_name"] = op.GetName();
    err_map["description"] = "convert src op failed";
    std::string report_error_code = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    OP_LOGE(op.GetName().c_str(), "convert src op failed.");
    return FAILED;
  }

  if (layer->bottom_size() != 1) {
    map<string, string> err_map;
    err_map["op_name"] = op.GetName();
    err_map["param_name"] = "Deconvolution layer bottom num";
    err_map["expected_value"] = "1";
    err_map["input_value"] = to_string(layer->bottom_size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    OP_LOGE(op.GetName().c_str(), "Deconvolution layer bottom num(%d) must be 1", layer->bottom_size());
    return FAILED;
  }

  if (layer->top_size() != 1) {
    map<string, string> err_map;
    err_map["op_name"] = op.GetName();
    err_map["param_name"] = "Deconvolution layer top num";
    err_map["expected_value"] = "1";
    err_map["input_value"] = to_string(layer->top_size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    OP_LOGE(op.GetName().c_str(), "Deconvolution layer top num(%d) must be 1", layer->top_size());
    return FAILED;
  }

  const caffe::ConvolutionParameter& convParam = layer->convolution_param();

  if (!SetPads(convParam, op)) {
    map<string, string> err_map;
    err_map["op_name"] = op.GetName();
    err_map["description"] = "set pads failed.";
    std::string report_error_code = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    OP_LOGE(op.GetName().c_str(), "set pads failed.");
    return FAILED;
  }
  if (!SetStrides(convParam, op)) {
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "set strides failed.";
    std::string report_error_code = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    OP_LOGE(op.GetName().c_str(), "set strides failed.");
    return FAILED;
  }
  if (!SetDilations(convParam, op)) {
    map<string, string> err_map;
    err_map["op_name"] = op.GetName();
    err_map["description"] = "set dilations failed.";
    std::string report_error_code = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    OP_LOGE(op.GetName().c_str(), "set dilations failed.");
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
