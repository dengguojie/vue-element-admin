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
 * \file extract_image_patches_plugin.cpp
 * \brief
 */
#include "graph/utils/op_desc_utils.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {

namespace {
  const size_t kKsizeLength = 4;
  const size_t kStridesLength = 4;
  const size_t kRatesLength = 4;
}

Status ExtractImagePatchesMappingFn(const Message* op_src, ge::Operator& op) {
  if (AutoMappingFn(op_src, op) != SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "AutoMappingFn failed.");
    return FAILED;
  }
  auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (op_dsc == nullptr) {
    OP_LOGE(op.GetName().c_str(), "GetOpDescFromOperator got nullptr failed.");
    return FAILED;
  }
  ge::GeTensorDesc tensor_descw = op_dsc->GetInputDesc(0);
  ge::GeTensorDesc tensor_descw1 = op_dsc->GetOutputDesc(0);
  tensor_descw.SetOriginFormat(ge::FORMAT_NHWC);
  tensor_descw1.SetOriginFormat(ge::FORMAT_NHWC);
  tensor_descw.SetFormat(ge::FORMAT_NHWC);
  tensor_descw1.SetFormat(ge::FORMAT_NHWC);
  auto ret = op_dsc->UpdateInputDesc(0, tensor_descw);
  auto ret1 = op_dsc->UpdateOutputDesc(0, tensor_descw1);
  if (ret != ge::GRAPH_SUCCESS || ret1 != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateInputDesc or UpdateOutputDesc failed.");
    return FAILED;
  }

  std::vector<int64_t> ksize;
  if (op.GetAttr("ksizes", ksize) != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get ksizes attr failed.");
    return FAILED;
  }
  if (ksize.size() != kKsizeLength) {
    OP_LOGE(op.GetName().c_str(), "Ksize has an incorrected length.");
    return FAILED;
  }
  vector<int64_t> ksize_hw = {ksize[1], ksize[2]};
  op.SetAttr("ksizes", ksize_hw);

  vector<int64_t> strides;
  if (op.GetAttr("strides", strides) != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get strides attr failed.");
    return FAILED;
  }
  if (strides.size() != kStridesLength) {
    OP_LOGE(op.GetName().c_str(), "Strides has an incorrected length.");
    return FAILED;
  }
  vector<int64_t> strides_hw = {strides[1], strides[2]};
  op.SetAttr("strides", strides_hw);

  vector<int64_t> rates;
  if (op.GetAttr("rates", rates) != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get rates attr failed.");
    return FAILED;
  }
  if (rates.size() != kRatesLength) {
    OP_LOGE(op.GetName().c_str(), "rates has an incorrected length.");
    return FAILED;
  }
  vector<int64_t> rates_hw = {rates[1], rates[2]};
  op.SetAttr("dilations", rates_hw);

  std::string padding = "";
  if (op.GetAttr("padding", padding) != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get padding attr failed.");
    return FAILED;
  }
  if (padding != "SAME" && padding != "VALID") {
    OP_LOGE(op.GetName().c_str(), "TF padding pattern is incorrected.");
    return FAILED;
  }
  op.SetAttr("padding_mode", padding);

  return SUCCESS;
}

REGISTER_CUSTOM_OP("Im2col")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ExtractImagePatches")
    .ParseParamsFn(ExtractImagePatchesMappingFn)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
