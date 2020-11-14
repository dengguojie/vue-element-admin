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
 * \file nn_calculation_ops.cpp
 * \brief
 */
#define CHECK_FORMAT(format)                                                     \
  {                                                                              \
    if (ge::FORMAT_RESERVED == format) {                                         \
      OP_LOGE(op.GetName().c_str(), "get format failed:%s:%d", #format, format); \
      return false;                                                              \
    }                                                                            \
  }

#include "./nn_calculation_ops.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "./util/error_util.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "util/util.h"
#include "register/infer_data_slice_registry.h"
#include "graph/common_error_codes.h"
#include "graph/debug/ge_attr_define.h"
namespace ge {

namespace {
  const int32_t kConv3dDimSizeLimit = 5;
  const int32_t kConv3dLengthPadsLimit = 6;
  const int32_t kConv3dStridesSizeLimit = 5;
  const int32_t kConv3dInputSizeLimit = 5;
  const int32_t kConv3dPadsSizeLimit = 6;
  const int32_t kConv3dDataSlice = 6;
}

// ----------------LSTM begin-------------------

IMPLEMT_VERIFIER(LSTM, LSTMInferShape) {
  return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(LSTM, LSTMVerify) {
  int32_t output_size = 0;
  int32_t input_size = op.GetInputsSize();
  bool expose_hidden = false;

  if (ge::GRAPH_SUCCESS != op.GetAttr("expose_hidden", expose_hidden)) {
    OpsGetAttrErrReport(op.GetName(), "expose_hidden");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr expose_hidden failed!");
  }

  if (input_size == 9) {
    output_size = op.GetInputDesc(3).GetShape().GetDim(2);
  } else if (input_size == 7 && expose_hidden) {
    output_size = op.GetInputDesc(2).GetShape().GetDim(2);
  } else if (input_size == 7) {
    output_size = op.GetInputDesc(6).GetShape().GetDim(1);
  } else {
    output_size = op.GetInputDesc(4).GetShape().GetDim(1);
  }

  ge::TensorDesc inputXTensorDesc = op.GetInputDesc(0);

  vector<int64_t> hDims;

  hDims.push_back(inputXTensorDesc.GetShape().GetDim(0));
  hDims.push_back(inputXTensorDesc.GetShape().GetDim(1));
  hDims.push_back(output_size);

  TensorDesc outputHTensorDesc = op.GetOutputDesc(0);
  outputHTensorDesc.SetShape(ge::Shape(hDims));
  (void)op.UpdateOutputDesc("h", outputHTensorDesc);

  if (expose_hidden) {
    int32_t c_index = 0;
    int32_t h_index = 0;
    if (input_size == 9) {
      h_index = 3;
      c_index = 4;
    } else {
      h_index = 2;
      c_index = 3;
    }
    ge::TensorDesc inputHTensorDesc = op.GetInputDesc(h_index);
    ge::TensorDesc inputCTensorDesc = op.GetInputDesc(c_index);
    ge::Shape shape = inputCTensorDesc.GetShape();
    ge::Shape shapeH = inputHTensorDesc.GetShape();

    TensorDesc outputHtTensorDesc = op.GetOutputDesc(1);
    TensorDesc outputCtTensorDesc = op.GetOutputDesc(2);

    outputCtTensorDesc.SetShape(shape);
    outputHtTensorDesc.SetShape(shapeH);

    (void)op.UpdateOutputDesc("h_t", outputHtTensorDesc);
    (void)op.UpdateOutputDesc("c_t", outputCtTensorDesc);
  }

  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(LSTM, LSTMInferShape);
VERIFY_FUNC_REG(LSTM, LSTMVerify);
// ----------------LSTM end------------------

// ----------------DepthwiseConv2d Op-------------------
// Obtains the value of the constant tensor.

static void InferHWDepthwiseConv2D(int32_t input, int32_t kernel, int32_t pad, int32_t stride,
                          int32_t dilation, vector<int64_t> output_slice, vector<int64_t>& data_slice) {
  // calc start rule: (i_start + pad_h)/stride_h = output_start
  int64_t i_start = output_slice[0] * stride - pad;
  if (i_start < 0) {
    i_start = 0;
  }
  // calc end rule: (iend_start + pad_h)/stride_h = output_end
  // iend_end = iend_start + dilation*(kernel_h-1)
  int64_t i_end = output_slice[1] * stride - pad + dilation * (kernel - 1);
  if (i_end >= input) {
    i_end = input - 1;
  }
  data_slice = {i_start, i_end};
}
/*!
  * @brief provide DepthwiseConv2D operator slice data
  * @param DepthwiseConv2D Operator type.
  * @param DepthwiseConv2D slice data function
  * @return Status The processing flow result.
  */
IMPLEMT_INFER_DATA_SLICE(DepthwiseConv2D, DepthwiseConv2DInferDataSlice) {
  OP_LOGD(op.GetName().c_str(), "Enter DepthwiseConv2D InferDataSlice");
  // get input h/w, filter h/w, pad_h,pad_w, stride_h, stride_w, dilation_h,dilation_w
  auto x_tensor = op.GetInputDesc("x");
  auto w_tensor = op.GetInputDesc("filter");

  auto x_shape = x_tensor.GetShape().GetDims();
  auto w_shape = w_tensor.GetShape().GetDims();

  auto x_format = x_tensor.GetFormat();
  auto w_format = w_tensor.GetFormat();

  std::vector<int32_t> stride_list;
  std::vector<int32_t> dilation_list;
  std::vector<int32_t> pad_list;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stride_list) || GRAPH_SUCCESS != op.GetAttr("dilations", dilation_list)
      || GRAPH_SUCCESS != op.GetAttr("pads", pad_list)){
    return GRAPH_FAILED;
  }

  int32_t ih = 0;
  int32_t iw = 0;
  int32_t kh = 0;
  int32_t kw = 0;
  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;
  int32_t padt = pad_list[0];
  int32_t padb = pad_list[1];
  int32_t padl = pad_list[2];
  int32_t padr = pad_list[3];

  if (x_format == FORMAT_NCHW) {
    ih = x_shape[2];
    iw = x_shape[3];
    strh = stride_list[2];
    strw = stride_list[3];
    dilh = dilation_list[2];
    dilw = dilation_list[3];
  } else if (x_format == FORMAT_NHWC) {
    ih = x_shape[1];
    iw = x_shape[2];
    strh = stride_list[1];
    strw = stride_list[2];
    dilh = dilation_list[1];
    dilw = dilation_list[2];
  }

  if (w_format == FORMAT_NCHW) {
    kh = w_shape[2];
    kw = w_shape[3];
  } else if (w_format == FORMAT_NHWC) {
    kh = w_shape[1];
    kw = w_shape[2];
  } else if (w_format == FORMAT_HWCN) {
    kh = w_shape[0];
    kw = w_shape[1];
  }

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
  vector<vector<int64_t>> y_data_slice;
  vector<vector<int64_t>> x_data_slice = {{}, {}, {}, {}, {}};
  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op.GetName().c_str(), "no data slice, not need infer input");
    return GRAPH_FAILED;
  }
  bool need_infer = false;
  bool have_slice = false;
  for(int i=0; i < y_data_slice.size(); i++) {
    if (y_data_slice[i].size() > 0) {
      have_slice = true;
      if (i == 2) {
        need_infer = true;
        vector<int64_t> ih_slice;
        InferHWDepthwiseConv2D(ih, kh, padt, strh, dilh, y_data_slice[i], ih_slice);
        OP_LOGD(op.GetName().c_str(), "DepthwiseConv2D h axis slice ori_scope is [%d,%d], calced output scope is [%d,%d]",
                ih_slice[0], ih_slice[1], y_data_slice[i][0], y_data_slice[i][1]);
        x_data_slice[i] = ih_slice;
      } else if (i == 3) {
        need_infer = true;
        vector<int64_t> iw_slice;
        InferHWDepthwiseConv2D(iw, kw, padl, strw, dilw, y_data_slice[i], iw_slice);
        OP_LOGD(op.GetName().c_str(), "DepthwiseConv2D w axis slice ori_scope is [%d,%d], calced output scope is [%d,%d]",
                iw_slice[0], iw_slice[1], y_data_slice[i][0], y_data_slice[i][1]);
        x_data_slice[i] = iw_slice;
      }
    }
  }
  if (have_slice == false) {
    return GRAPH_FAILED;
  }
  if (need_infer == false) {
    return NO_OVERLAP_DIM;
  } else{
    if(!AttrUtils::SetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }
  OP_LOGD(op.GetName().c_str(), "Calc DepthwiseConv2D InferDataSlice end!");
}

INFER_DATA_SLICE_FUNC_REG(DepthwiseConv2D, DepthwiseConv2DInferDataSlice);

static std::vector<int64_t> GetAttrValue(const ge::Operator& op, const std::string& key_name) {
  std::vector<int64_t> list;
  if (ge::GRAPH_SUCCESS != op.GetAttr(key_name, list)) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue failed!");
  }

  return list;
}

static bool CheckListEmpty(const std::string& opName, const std::vector<int64_t>& list, const std::string& attrName) {
  if (list.empty()) {
    OP_LOGE(opName.c_str(), "the %s is empty !", attrName.c_str());
    return false;
  }

  return true;
}

static bool GetPadDepthwiseConv2D(ge::Operator& op, int64_t inH, int64_t inW, int64_t filterH, int64_t filterW,
                                  int64_t strideH, int64_t strideW, int64_t dilationH, int64_t dilationW,
                                  int64_t& padtop, int64_t& padbottom, int64_t& padleft, int64_t& padright) {
  std::string padStr;
  std::vector<int64_t> padList;
  if (GRAPH_SUCCESS == op.GetAttr("_padding", padList)) {
    op.SetAttr("pads", padList);
  } else if (GRAPH_SUCCESS == op.GetAttr("padding", padStr)) {
    if (padStr.compare("SAME") == 0) {
      int64_t effective_filter_h = (filterH - 1) * dilationH + 1;
      int64_t effective_filter_w = (filterW - 1) * dilationW + 1;
      int64_t out_h = (inH + strideH - 1) / strideH;
      int64_t out_w = (inW + strideW - 1) / strideW;
      int64_t pad_h = std::max((out_h - 1) * strideH + effective_filter_h - inH, (int64_t)0);
      int64_t pad_w = std::max((out_w - 1) * strideW + effective_filter_w - inW, (int64_t)0);
      padList.push_back(pad_h / 2);
      padList.push_back(pad_h / 2 + pad_h % 2);
      padList.push_back(pad_w / 2);
      padList.push_back(pad_w / 2 + pad_w % 2);
    } else if (padStr.compare("VALID") == 0) {
      padList.push_back(0);
      padList.push_back(0);
      padList.push_back(0);
      padList.push_back(0);
    } else {
      OP_LOGE(op.GetName().c_str(),
              "padding should be SAME or VALID."
              " actual is: %s.",
              padStr.c_str());
      return false;
    }
    op.SetAttr("pads", padList);
  }
  std::vector<int64_t> padVec;
  if (op.GetAttr("pads", padVec) == ge::GRAPH_FAILED) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue padding failed!");
    return false;
  }
  auto pSize = padVec.size();
  if (pSize != 4) {
    OP_LOGE(op.GetName().c_str(),
            "pads list should be 4d."
            " actual is: %d.",
            (int)pSize);
    return false;
  }
  padtop = padVec[0];
  padbottom = padVec[1];
  padleft = padVec[2];
  padright = padVec[3];
  if (padtop < 0 || padbottom < 0 || padleft < 0 || padright < 0) {
    OP_LOGE(op.GetName().c_str(),
            "pads should be positive, "
            " actual is [%d,%d,%d,%d].",
            padtop, padbottom, padleft, padright);
    return false;
  }

  return true;
}

// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(DepthwiseConv2D, DepthwiseConv2DVerify) {
  auto xTensor = op.GetInputDesc(0);
  auto wTensor = op.GetInputDesc(1);

  auto xShape = xTensor.GetShape().GetDims();
  auto wShape = wTensor.GetShape().GetDims();

  if (xShape.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "input x shape should be 4d. input x shape size is %d", (int)xShape.size());
    return GRAPH_FAILED;
  }

  if (wShape.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "input filter shape should be 4d. input filter shape size is %d", (int)wShape.size());
    return GRAPH_FAILED;
  }

  auto xDtype = xTensor.GetDataType();
  auto wDtype = wTensor.GetDataType();

  if (xDtype != wDtype) {
    OP_LOGE(op.GetName().c_str(), "input x dtype(%d) is differ from filter dtype(%d).", (int)xDtype, (int)wDtype);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dilation;
  dilation = GetAttrValue(op, "dilations");
  if (!CheckListEmpty(op.GetName(), dilation, "dilations")) {
    OP_LOGE(op.GetName().c_str(), "Get dilations failed!");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> stride;
  stride = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), stride, "strides")) {
    OP_LOGE(op.GetName().c_str(), "Get stride failed!");
    return GRAPH_FAILED;
  }
  if (stride.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "stride dim(%d) must be 4!", (int)stride.size());
    return GRAPH_FAILED;
  }

  std::vector<int64_t> pads;
  pads = GetAttrValue(op, "pads");
  if (!CheckListEmpty(op.GetName(), pads, "pads")) {
    OP_LOGE(op.GetName().c_str(), "Get pads failed!");
    return GRAPH_FAILED;
  }
  if (pads.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "attr pads(%d) is too large", (int)pads.size());
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
    if (data_format != "NCHW" && data_format != "NHWC") {
      OP_LOGE(op.GetName().c_str(), "attr data_format(%s) only support NCHW and NHWC", data_format.c_str());
      return GRAPH_FAILED;
    }
  }
  // attr offset_x need not check
  return GRAPH_SUCCESS;
}

static map<int, std::string> format2str = {
    {ge::FORMAT_NCHW, "NCHW"},   {ge::FORMAT_NHWC, "NHWC"},   {ge::FORMAT_HWCN, "HWCN"},  {ge::FORMAT_DHWNC, "DHWNC"},
    {ge::FORMAT_DHWCN, "DHWCN"}, {ge::FORMAT_NDHWC, "NDHWC"}, {ge::FORMAT_NCDHW, "NCDHW"}};

static bool GetDimInFormat(const std::string& opName, const std::string& formatStr, const std::string& dimName,
                           int64_t& dimPosition) {
  dimPosition = formatStr.find(dimName);
  if (dimPosition < 0) {
    OP_LOGE(opName.c_str(), "Position(%s) is invalid: %d, which format is %s.", dimName.c_str(), dimPosition,
            formatStr.c_str());
    return false;
  }
  return true;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DepthwiseConv2DInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter op_proto inferfunction!");

  int64_t nPosition = 0;
  int64_t cPosition = 0;
  int64_t hPosition = 0;
  int64_t wPosition = 0;
  int64_t inN = 0;
  int64_t inC = 0;
  int64_t inH = 0;
  int64_t inW = 0;
  int64_t outH = 0;
  int64_t outW = 0;
  int64_t filterH = 0;
  int64_t filterW = 0;
  int64_t dilationH = 0;
  int64_t dilationW = 0;
  int64_t strideH = 0;
  int64_t strideW = 0;
  int64_t effectiveFilterH = 0;
  int64_t effectiveFilterW = 0;
  int64_t padtop = 0;
  int64_t padbottom = 0;
  int64_t padleft = 0;
  int64_t padright = 0;

  std::vector<int64_t> dilation;
  dilation = GetAttrValue(op, "dilations");
  if (!CheckListEmpty(op.GetName(), dilation, "dilations")) {
    OP_LOGE(op.GetName().c_str(), "Get dilations failed!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> stride;
  stride = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), stride, "strides")) {
    OP_LOGE(op.GetName().c_str(), "Get stride failed!");
    return GRAPH_FAILED;
  }

  std::string dataFormat = "";
  if (ge::GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
    OP_LOGE(op.GetName().c_str(), "get data_format attr failed");
    return GRAPH_FAILED;
  }

  auto tensorDescIn = op.GetInputDesc(0);
  auto tensorDescW = op.GetInputDesc(1);

  auto shapeIn = tensorDescIn.GetShape();
  auto shapeW = tensorDescW.GetShape();

  auto dataTypeIn = tensorDescIn.GetDataType();

  if (!GetDimInFormat(op.GetName(), dataFormat, "N", nPosition)) {
    return GRAPH_FAILED;
  }

  if (!GetDimInFormat(op.GetName(), dataFormat, "C", cPosition)) {
    return GRAPH_FAILED;
  }

  if (!GetDimInFormat(op.GetName(), dataFormat, "H", hPosition)) {
    return GRAPH_FAILED;
  }
  if (!GetDimInFormat(op.GetName(), dataFormat, "W", wPosition)) {
    return GRAPH_FAILED;
  }

  // NC1HWC0(NCHW)
  inN = shapeIn.GetDim(nPosition);
  inC = shapeIn.GetDim(cPosition);
  inH = shapeIn.GetDim(hPosition);
  inW = shapeIn.GetDim(wPosition);

  Format filterFormat = tensorDescW.GetFormat();
  std::string filterFormatStr = format2str[filterFormat];
  int64_t fhPosition = 0;
  int64_t fwPosition = 0;
  if (!GetDimInFormat(op.GetName(), filterFormatStr, "H", fhPosition)) {
    return GRAPH_FAILED;
  }
  if (!GetDimInFormat(op.GetName(), filterFormatStr, "W", fwPosition)) {
    return GRAPH_FAILED;
  }

  filterH = shapeW.GetDim(fhPosition);
  filterW = shapeW.GetDim(fwPosition);

  dilationH = dilation.at(hPosition);
  dilationW = dilation.at(wPosition);
  strideH = stride.at(hPosition);
  strideW = stride.at(wPosition);

  if (false == GetPadDepthwiseConv2D(op, inH, inW, filterH, filterW, strideH, strideW, dilationH, dilationW, padtop,
                                     padbottom, padleft, padright)) {
    OP_LOGE(op.GetName().c_str(), "get pads attrs failed.");
    return GRAPH_FAILED;
  }

  effectiveFilterH = (filterH - 1) * dilationH + 1;
  effectiveFilterW = (filterW - 1) * dilationW + 1;
  outH = (inH + padtop + padbottom - effectiveFilterH) / strideH + 1;
  outW = (inW + padleft + padright - effectiveFilterW) / strideW + 1;

  vector<int64_t> shapeOut;
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  auto formatOut = tensordesc_output.GetFormat();

  // NC1HWC0(NCHW/NHWC)
  if (formatOut == FORMAT_NCHW) {
    shapeOut.push_back(inN);
    shapeOut.push_back(inC);
    shapeOut.push_back(outH);
    shapeOut.push_back(outW);
  } else if (formatOut == FORMAT_NHWC) {
    shapeOut.push_back(inN);
    shapeOut.push_back(outH);
    shapeOut.push_back(outW);
    shapeOut.push_back(inC);
  } else {
    OP_LOGE(op.GetName().c_str(),
            "output y format should be NCHW or NHWC."
            " actual is: %d",
            (int)formatOut);
    return GRAPH_FAILED;
  }

  tensordesc_output.SetShape(Shape(shapeOut));
  if (dataTypeIn == ge::DT_INT8) {
    tensordesc_output.SetDataType(ge::DT_INT32);
  } else {
    tensordesc_output.SetDataType(dataTypeIn);
  }
  (void)op.UpdateOutputDesc("y", tensordesc_output);

  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DepthwiseConv2D, DepthwiseConv2DInferShape);

// Registered verify function
VERIFY_FUNC_REG(DepthwiseConv2D, DepthwiseConv2DVerify);

static graphStatus VerifyDepthwiseConv2DbpPadding(ge::Operator& op) {
  std::string pad;
  if (GRAPH_SUCCESS == op.GetAttr("padding", pad)) {
    if (pad.compare("SAME") != 0 && pad.compare("VALID") != 0) {
      OP_LOGE(op.GetName().c_str(), "padding must be SAME or VALID. actual is: %s", pad.c_str());
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "get padding failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

static graphStatus VerifyDepthwiseConv2DbpPads(ge::Operator& op) {
  std::vector<int64_t> pads;
  if (GRAPH_SUCCESS == op.GetAttr("pads", pads)) {
    if (pads.size() < 4) {
      OP_LOGE(op.GetName().c_str(), "op pads's size is illegal,pads.size:%d", pads.size());
      return GRAPH_FAILED;
    }
    OP_LOGI(op.GetName().c_str(), "op pads: top:%d,bottom:%d,left:%d,right:%d", pads[0], pads[1], pads[2], pads[3]);
    if (pads[0] < 0 || pads[1] < 0 || pads[2] < 0 || pads[3] < 0) {
      OP_LOGE(op.GetName().c_str(), "op get pads is illegal");
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "op get pads failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// ----------------DepthwiseConv2DBackpropInputD Op-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(DepthwiseConv2DBackpropInputD, DepthwiseConv2DBackpropInputDVerify) {
  std::vector<int64_t> input_size;
  input_size = GetAttrValue(op, "input_size");
  if (!CheckListEmpty(op.GetName(), input_size, "input_size")) {
    return GRAPH_FAILED;
  }

  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), strides, "strides")) {
    return GRAPH_FAILED;
  }
  if (strides.size() != DIM_SIZE4) {
    OP_LOGE(op.GetName().c_str(), "strides must be NCHW!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dilations;
  dilations = GetAttrValue(op, "dilations");
  if (!CheckListEmpty(op.GetName(), dilations, "dilations")) {
    return GRAPH_FAILED;
  }
  if (dilations.size() != DIM_SIZE4) {
    OP_LOGE(op.GetName().c_str(), "dilations must be NCHW!");
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
    if (data_format != "NCHW" && data_format != "NHWC") {
      OP_LOGE(op.GetName().c_str(), "attr data_format(%s) only support NCHW and NHWC", data_format.c_str());
      return GRAPH_FAILED;
    }
  }
  if (GRAPH_SUCCESS != VerifyDepthwiseConv2DbpPads(op)) {
    return GRAPH_FAILED;
  }

  if (op.GetInputDesc(0).GetDataType() != op.GetInputDesc(1).GetDataType()) {
    OP_LOGE(op.GetName().c_str(), "The type of filter and out_backprop must be same!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DepthwiseConv2DBackpropInputDInferShape) {
  std::vector<int64_t> input_size;
  input_size = GetAttrValue(op, "input_size");

  DataType output_dtype = op.GetInputDesc("out_backprop").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("input_grad");
  tensordesc_output.SetShape(Shape(input_size));
  tensordesc_output.SetDataType(output_dtype);
  (void)op.UpdateOutputDesc("input_grad", tensordesc_output);

  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), strides, "strides")) {
    return GRAPH_FAILED;
  }
  if (strides.size() != DIM_SIZE4) {
    OP_LOGE(op.GetName().c_str(), "strides must be NCHW!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dilations;
  dilations = GetAttrValue(op, "dilations");
  if (!CheckListEmpty(op.GetName(), dilations, "dilations")) {
    return GRAPH_FAILED;
  }
  if (dilations.size() != DIM_SIZE4) {
    OP_LOGE(op.GetName().c_str(), "dilations must be NCHW!");
    return GRAPH_FAILED;
  }

  std::string dataFormat = "";
  if (ge::GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
    OP_LOGE(op.GetName().c_str(), "get data_format attr failed");
    return GRAPH_FAILED;
  }

  int64_t hPosition = 0;
  int64_t wPosition = 0;
  int64_t fhPosition = 0;
  int64_t fwPosition = 0;
  int64_t inH = 0;
  int64_t inW = 0;
  int64_t filterH = 0;
  int64_t filterW = 0;
  int64_t dilationH = 0;
  int64_t dilationW = 0;
  int64_t strideH = 0;
  int64_t strideW = 0;
  int64_t padtop = 0;
  int64_t padbottom = 0;
  int64_t padleft = 0;
  int64_t padright = 0;

  auto tensorDescW = op.GetInputDesc(0);
  auto shapeW = tensorDescW.GetShape();

  Format filterFormat = tensorDescW.GetFormat();
  std::string filterFormatStr = format2str[filterFormat];
  if (!GetDimInFormat(op.GetName(), filterFormatStr, "H", fhPosition)) {
    return GRAPH_FAILED;
  }
  if (!GetDimInFormat(op.GetName(), filterFormatStr, "W", fwPosition)) {
    return GRAPH_FAILED;
  }

  filterH = shapeW.GetDim(fhPosition);
  filterW = shapeW.GetDim(fwPosition);

  if (!GetDimInFormat(op.GetName(), dataFormat, "H", hPosition)) {
    return GRAPH_FAILED;
  }
  if (!GetDimInFormat(op.GetName(), dataFormat, "W", wPosition)) {
    return GRAPH_FAILED;
  }

  // NC1HWC0(NCHW)
  inH = input_size[hPosition];
  inW = input_size[wPosition];

  dilationH = dilations.at(hPosition);
  dilationW = dilations.at(wPosition);
  strideH = strides.at(hPosition);
  strideW = strides.at(wPosition);

  if (false == GetPadDepthwiseConv2D(op, inH, inW, filterH, filterW, strideH, strideW, dilationH, dilationW, padtop,
                                     padbottom, padleft, padright)) {
    OP_LOGE(op.GetName().c_str(), "update pads attrs failed.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DepthwiseConv2DBackpropInputD, DepthwiseConv2DBackpropInputDInferShape);
// Registered verify function
VERIFY_FUNC_REG(DepthwiseConv2DBackpropInputD, DepthwiseConv2DBackpropInputDVerify);

// ----------------DepthwiseConv2DBackpropInput Op-------------------
// Obtains the value of the constant tensor.
static void GetConstValue(const Tensor& const_tensor, const DataType& dtype, std::vector<int64_t>& const_data) {
  const uint8_t* constData = const_tensor.GetData();
  size_t size;
  if (dtype == ge::DT_INT32) {
    size = const_tensor.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back(*((int32_t*)constData + i));
    }
  } else {
    size = const_tensor.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back(*((int64_t*)constData + i));
    }
  }
}

// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(DepthwiseConv2DBackpropInput, DepthwiseConv2DBackpropInputVerify) {
  Tensor input_size_tensor;
  if (GRAPH_SUCCESS != op.GetInputConstData("input_size", input_size_tensor)) {
    OP_LOGE(op.GetName().c_str(), "Get constdata failed");
    return GRAPH_FAILED;
  }
  DataType dtype = op.GetInputDesc("input_size").GetDataType();

  std::vector<int64_t> input_size;
  GetConstValue(input_size_tensor, dtype, input_size);
  if (!CheckListEmpty(op.GetName(), input_size, "input_size")) {
    return GRAPH_FAILED;
  }

  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), strides, "strides")) {
    return GRAPH_FAILED;
  }
  if (strides.size() != DIM_SIZE4) {
    OP_LOGE(op.GetName().c_str(), "strides must be NCHW!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dilations;
  dilations = GetAttrValue(op, "dilations");
  if (!CheckListEmpty(op.GetName(), dilations, "dilations")) {
    return GRAPH_FAILED;
  }
  if (dilations.size() != DIM_SIZE4) {
    OP_LOGE(op.GetName().c_str(), "dilations must be NCHW!");
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
    if (data_format != "NCHW" && data_format != "NHWC") {
      OP_LOGE(op.GetName().c_str(), "attr data_format(%s) only support NCHW and NHWC", data_format.c_str());
      return GRAPH_FAILED;
    }
  }

  if (VerifyDepthwiseConv2DbpPadding(op) == GRAPH_SUCCESS || VerifyDepthwiseConv2DbpPads(op) == GRAPH_SUCCESS) {
    return GRAPH_SUCCESS;
  } else {
    return GRAPH_FAILED;
  }

  if (op.GetInputDesc(1).GetDataType() != op.GetInputDesc(2).GetDataType()) {
    OP_LOGE(op.GetName().c_str(), "The type of filter and out_backprop must be same!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DepthwiseConv2DBackpropInputInferShape) {
  Tensor input_size_tensor;
  if (ge::GRAPH_SUCCESS != op.GetInputConstData("input_size", input_size_tensor)) {
    OP_LOGE(op.GetName().c_str(), "get constdata failed");
    return GRAPH_FAILED;
  }
  DataType dtype = op.GetInputDesc("input_size").GetDataType();
  std::vector<int64_t> input_size;
  GetConstValue(input_size_tensor, dtype, input_size);

  DataType output_dtype = op.GetInputDesc("out_backprop").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("input_grad");
  tensordesc_output.SetShape(Shape(input_size));
  tensordesc_output.SetDataType(output_dtype);
  (void)op.UpdateOutputDesc("input_grad", tensordesc_output);

  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), strides, "strides")) {
    return GRAPH_FAILED;
  }
  if (strides.size() != DIM_SIZE4) {
    OP_LOGE(op.GetName().c_str(), "strides must be NCHW!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dilations;
  dilations = GetAttrValue(op, "dilations");
  if (!CheckListEmpty(op.GetName(), dilations, "dilations")) {
    return GRAPH_FAILED;
  }
  if (dilations.size() != DIM_SIZE4) {
    OP_LOGE(op.GetName().c_str(), "dilations must be NCHW!");
    return GRAPH_FAILED;
  }

  std::string dataFormat = "";
  if (ge::GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
    OP_LOGE(op.GetName().c_str(), "get data_format attr failed");
    return GRAPH_FAILED;
  }

  int64_t hPosition = 0;
  int64_t wPosition = 0;
  int64_t fhPosition = 0;
  int64_t fwPosition = 0;
  int64_t inH = 0;
  int64_t inW = 0;
  int64_t filterH = 0;
  int64_t filterW = 0;
  int64_t dilationH = 0;
  int64_t dilationW = 0;
  int64_t strideH = 0;
  int64_t strideW = 0;
  int64_t padtop = 0;
  int64_t padbottom = 0;
  int64_t padleft = 0;
  int64_t padright = 0;

  auto tensorDescW = op.GetInputDesc(1);
  auto shapeW = tensorDescW.GetShape();

  Format filterFormat = tensorDescW.GetFormat();
  std::string filterFormatStr = format2str[filterFormat];
  if (!GetDimInFormat(op.GetName(), filterFormatStr, "H", fhPosition)) {
    return GRAPH_FAILED;
  }
  if (!GetDimInFormat(op.GetName(), filterFormatStr, "W", fwPosition)) {
    return GRAPH_FAILED;
  }

  filterH = shapeW.GetDim(fhPosition);
  filterW = shapeW.GetDim(fwPosition);

  if (!GetDimInFormat(op.GetName(), dataFormat, "H", hPosition)) {
    return GRAPH_FAILED;
  }
  if (!GetDimInFormat(op.GetName(), dataFormat, "W", wPosition)) {
    return GRAPH_FAILED;
  }

  // NC1HWC0(NCHW)
  inH = input_size[hPosition];
  inW = input_size[wPosition];

  dilationH = dilations.at(hPosition);
  dilationW = dilations.at(wPosition);
  strideH = strides.at(hPosition);
  strideW = strides.at(wPosition);

  if (false == GetPadDepthwiseConv2D(op, inH, inW, filterH, filterW, strideH, strideW, dilationH, dilationW, padtop,
                                     padbottom, padleft, padright)) {
    OP_LOGE(op.GetName().c_str(), "update pads attrs failed.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DepthwiseConv2DBackpropInput, DepthwiseConv2DBackpropInputInferShape);
// Registered verify function
VERIFY_FUNC_REG(DepthwiseConv2DBackpropInput, DepthwiseConv2DBackpropInputVerify);

// ----------------DepthwiseConv2DBackpropFilterD Op-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(DepthwiseConv2DBackpropFilterD, DepthwiseConv2DBackpropFilterDVerify) {
  std::vector<int64_t> filter_size;
  filter_size = GetAttrValue(op, "filter_size");
  if (!CheckListEmpty(op.GetName(), filter_size, "filter_size")) {
    return GRAPH_FAILED;
  }
  if (filter_size.size() != DIM_SIZE4) {
    OP_LOGE(op.GetName().c_str(), "Filter_size must be HWCK!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), strides, "strides")) {
    return GRAPH_FAILED;
  }
  if (strides.size() != DIM_SIZE4) {
    OP_LOGE(op.GetName().c_str(), "strides must be 4!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dilations;
  dilations = GetAttrValue(op, "dilations");
  if (!CheckListEmpty(op.GetName(), dilations, "dilations")) {
    return GRAPH_FAILED;
  }
  if (dilations.size() != DIM_SIZE4) {
    OP_LOGE(op.GetName().c_str(), "dilations must be 4!");
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
    if (data_format != "NCHW" && data_format != "NHWC") {
      OP_LOGE(op.GetName().c_str(), "attr data_format(%s) only support NCHW and NHWC", data_format.c_str());
      return GRAPH_FAILED;
    }
  }
  if (GRAPH_SUCCESS != VerifyDepthwiseConv2DbpPads(op)) {
    return GRAPH_FAILED;
  }

  if (op.GetInputDesc(0).GetDataType() != op.GetInputDesc(1).GetDataType()) {
    OP_LOGE(op.GetName().c_str(), "The type of input and out_backprop must be same!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DepthwiseConv2DBackpropFilterDInferShape) {
  std::vector<int64_t> filter_size;
  filter_size = GetAttrValue(op, "filter_size");

  DataType output_dtype = op.GetInputDesc("out_backprop").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("filter_grad");
  tensordesc_output.SetShape(Shape(filter_size));
  tensordesc_output.SetDataType(output_dtype);
  (void)op.UpdateOutputDesc("filter_grad", tensordesc_output);

  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), strides, "strides")) {
    return GRAPH_FAILED;
  }
  if (strides.size() != DIM_SIZE4) {
    OP_LOGE(op.GetName().c_str(), "strides must be NCHW!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dilations;
  dilations = GetAttrValue(op, "dilations");
  if (!CheckListEmpty(op.GetName(), dilations, "dilations")) {
    return GRAPH_FAILED;
  }
  if (dilations.size() != DIM_SIZE4) {
    OP_LOGE(op.GetName().c_str(), "dilations must be NCHW!");
    return GRAPH_FAILED;
  }

  std::string dataFormat = "";
  if (ge::GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
    OP_LOGE(op.GetName().c_str(), "get data_format attr failed");
    return GRAPH_FAILED;
  }

  int64_t hPosition = 0;
  int64_t wPosition = 0;
  int64_t fhPosition = 0;
  int64_t fwPosition = 0;
  int64_t inH = 0;
  int64_t inW = 0;
  int64_t filterH = 0;
  int64_t filterW = 0;
  int64_t dilationH = 0;
  int64_t dilationW = 0;
  int64_t strideH = 0;
  int64_t strideW = 0;
  int64_t padtop = 0;
  int64_t padbottom = 0;
  int64_t padleft = 0;
  int64_t padright = 0;

  Format filterFormat = tensordesc_output.GetFormat();
  std::string filterFormatStr = format2str[filterFormat];
  if (!GetDimInFormat(op.GetName(), filterFormatStr, "H", fhPosition)) {
    return GRAPH_FAILED;
  }
  if (!GetDimInFormat(op.GetName(), filterFormatStr, "W", fwPosition)) {
    return GRAPH_FAILED;
  }
  filterH = filter_size[fhPosition];
  filterW = filter_size[fwPosition];

  auto tensorDescIn = op.GetInputDesc(0);
  auto shapeIn = tensorDescIn.GetShape();

  if (!GetDimInFormat(op.GetName(), dataFormat, "H", hPosition)) {
    return GRAPH_FAILED;
  }
  if (!GetDimInFormat(op.GetName(), dataFormat, "W", wPosition)) {
    return GRAPH_FAILED;
  }

  // NC1HWC0(NCHW)
  inH = shapeIn.GetDim(hPosition);
  inW = shapeIn.GetDim(wPosition);

  dilationH = dilations.at(hPosition);
  dilationW = dilations.at(wPosition);
  strideH = strides.at(hPosition);
  strideW = strides.at(wPosition);

  if (false == GetPadDepthwiseConv2D(op, inH, inW, filterH, filterW, strideH, strideW, dilationH, dilationW, padtop,
                                     padbottom, padleft, padright)) {
    OP_LOGE(op.GetName().c_str(), "update pads attrs failed.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DepthwiseConv2DBackpropFilterD, DepthwiseConv2DBackpropFilterDInferShape);
// Registered verify function
VERIFY_FUNC_REG(DepthwiseConv2DBackpropFilterD, DepthwiseConv2DBackpropFilterDVerify);

// ----------------DepthwiseConv2DBackpropFilter Op-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(DepthwiseConv2DBackpropFilter, DepthwiseConv2DBackpropFilterVerify) {
  Tensor filter_size_tensor;
  if (GRAPH_SUCCESS != op.GetInputConstData("filter_size", filter_size_tensor)) {
    OP_LOGE(op.GetName().c_str(), "Get constdata failed");
    return GRAPH_FAILED;
  }
  DataType dtype = op.GetInputDesc("filter_size").GetDataType();
  std::vector<int64_t> filter_size;
  GetConstValue(filter_size_tensor, dtype, filter_size);
  if (!CheckListEmpty(op.GetName(), filter_size, "filter_size")) {
    return GRAPH_FAILED;
  }
  if (filter_size.size() != DIM_SIZE4) {
    OP_LOGE(op.GetName().c_str(), "Filter_size must be 4!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), strides, "strides")) {
    return GRAPH_FAILED;
  }
  if (strides.size() != DIM_SIZE4) {
    OP_LOGE(op.GetName().c_str(), "strides must be 4!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dilations;
  dilations = GetAttrValue(op, "dilations");
  if (!CheckListEmpty(op.GetName(), dilations, "dilations")) {
    return GRAPH_FAILED;
  }
  if (dilations.size() != DIM_SIZE4) {
    OP_LOGE(op.GetName().c_str(), "dilations must be 4!");
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
    if (data_format != "NCHW" && data_format != "NHWC") {
      OP_LOGE(op.GetName().c_str(), "attr data_format(%s) only support NCHW and NHWC", data_format.c_str());
      return GRAPH_FAILED;
    }
  }
  if (GRAPH_SUCCESS == VerifyDepthwiseConv2DbpPadding(op) || GRAPH_SUCCESS == VerifyDepthwiseConv2DbpPads(op)) {
    return GRAPH_SUCCESS;
  } else {
    return GRAPH_FAILED;
  }

  if (op.GetInputDesc(0).GetDataType() != op.GetInputDesc(2).GetDataType()) {
    OP_LOGE(op.GetName().c_str(), "The type of input and out_backprop must be same!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DepthwiseConv2DBackpropFilterInferShape) {
  Tensor filter_size_tensor;
  if (GRAPH_SUCCESS != op.GetInputConstData("filter_size", filter_size_tensor)) {
    OP_LOGE(op.GetName().c_str(), "Get constdata failed");
    return GRAPH_FAILED;
  }
  DataType dtype = op.GetInputDesc("filter_size").GetDataType();
  std::vector<int64_t> filter_size;
  GetConstValue(filter_size_tensor, dtype, filter_size);

  DataType output_dtype = op.GetInputDesc("out_backprop").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("filter_grad");
  tensordesc_output.SetShape(Shape(filter_size));
  tensordesc_output.SetDataType(output_dtype);
  (void)op.UpdateOutputDesc("filter_grad", tensordesc_output);

  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), strides, "strides")) {
    return GRAPH_FAILED;
  }
  if (strides.size() != DIM_SIZE4) {
    OP_LOGE(op.GetName().c_str(), "strides must be 4!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dilations;
  dilations = GetAttrValue(op, "dilations");
  if (!CheckListEmpty(op.GetName(), dilations, "dilations")) {
    return GRAPH_FAILED;
  }
  if (dilations.size() != DIM_SIZE4) {
    OP_LOGE(op.GetName().c_str(), "dilations must be 4!");
    return GRAPH_FAILED;
  }

  std::string dataFormat = "";
  if (ge::GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
    OP_LOGE(op.GetName().c_str(), "get data_format attr failed");
    return GRAPH_FAILED;
  }

  int64_t hPosition = 0;
  int64_t wPosition = 0;
  int64_t fhPosition = 0;
  int64_t fwPosition = 0;
  int64_t inH = 0;
  int64_t inW = 0;
  int64_t filterH = 0;
  int64_t filterW = 0;
  int64_t dilationH = 0;
  int64_t dilationW = 0;
  int64_t strideH = 0;
  int64_t strideW = 0;
  int64_t padtop = 0;
  int64_t padbottom = 0;
  int64_t padleft = 0;
  int64_t padright = 0;

  Format filterFormat = tensordesc_output.GetFormat();
  std::string filterFormatStr = format2str[filterFormat];
  if (!GetDimInFormat(op.GetName(), filterFormatStr, "H", fhPosition)) {
    return GRAPH_FAILED;
  }
  if (!GetDimInFormat(op.GetName(), filterFormatStr, "W", fwPosition)) {
    return GRAPH_FAILED;
  }
  filterH = filter_size[fhPosition];
  filterW = filter_size[fwPosition];

  auto tensorDescIn = op.GetInputDesc(0);
  auto shapeIn = tensorDescIn.GetShape();

  if (!GetDimInFormat(op.GetName(), dataFormat, "H", hPosition)) {
    return GRAPH_FAILED;
  }
  if (!GetDimInFormat(op.GetName(), dataFormat, "W", wPosition)) {
    return GRAPH_FAILED;
  }

  // NC1HWC0(NCHW)
  inH = shapeIn.GetDim(hPosition);
  inW = shapeIn.GetDim(wPosition);

  dilationH = dilations.at(hPosition);
  dilationW = dilations.at(wPosition);
  strideH = strides.at(hPosition);
  strideW = strides.at(wPosition);

  if (false == GetPadDepthwiseConv2D(op, inH, inW, filterH, filterW, strideH, strideW, dilationH, dilationW, padtop,
                                     padbottom, padleft, padright)) {
    OP_LOGE(op.GetName().c_str(), "update pads attrs failed.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DepthwiseConv2DBackpropFilter, DepthwiseConv2DBackpropFilterInferShape);
// Registered verify function
VERIFY_FUNC_REG(DepthwiseConv2DBackpropFilter, DepthwiseConv2DBackpropFilterVerify);

// --------------------------------BiasAddGrad---------------------------------
IMPLEMT_COMMON_INFERFUNC(BiasAddGradInferShape) {
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  ge::TensorDesc tensor_desc_x = op.GetInputDesc("x");
  ge::Shape shape = tensor_desc_x.GetShape();
  size_t dim_num = shape.GetDimNum();

  std::string data_format;
  if (ge::GRAPH_SUCCESS != op.GetAttr("data_format", data_format)) {
    OpsGetAttrErrReport(op.GetName(), "data_format");
    printf(
        "[Plugin][ERROR]The bias add grad op GetOpAttr"
        "data_format failed!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dim_vec;
  if (data_format == "NHWC") {
    if (dim_num < DIM_SIZE2 || dim_num > DIM_SIZE8) {
      OpsInputShapeDimErrReport(op.GetName(), "x", ConcatString(DIM_SIZE8), ConcatString(DIM_SIZE2),
                                ConcatString(dim_num));
      OP_LOGE(
          "[Plugin][ERROR]The bias add grad op dimension(%lu) is not"
          "supported when format is NHWC!",
          dim_num);
      return GRAPH_FAILED;
    }
    dim_vec.push_back(shape.GetDim(dim_num - 1));
  } else if (data_format == "NCHW") {
    if (dim_num < DIM_SIZE2) {
      OP_LOGE(
          "[Plugin][ERROR]The bias add grad op dimension(%lu) is not"
          "supported when format is NCHW!",
          dim_num);
      return GRAPH_FAILED;
    }
    dim_vec.push_back(shape.GetDim(1));
  } else {
    string expected_format_list = ConcatString("NHWC, NCHW");
    OpsInputFormatErrReport(op.GetName(), "x", expected_format_list, data_format);
    OP_LOGE(
        "[Plugin][ERROR]The bias add grad op data format(%s) is not"
        "supported!",
        data_format.c_str());
    return GRAPH_FAILED;
  }

  tensordesc_output.SetShape(ge::Shape(dim_vec));
  tensordesc_output.SetRealDimCnt(1);

  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BiasAddGrad, BiasAddGradInferShape);
// ------------------------------BiasAddGrad END-----------------------------------

//============================Conv2Dbackprop===============================
#define ALIGN_CONV2DBP(x_1, x_2) ((((x_1) + (x_2)-1) / (x_2)) * (x_2))

#define CHECK_POSITION(position)                                                       \
  {                                                                                    \
    if (position < 0) {                                                                \
      OP_LOGE(op.GetName().c_str(), "get position failed:%s:%d", #position, position); \
      return false;                                                                    \
    }                                                                                  \
  }

static bool getStrideDilationHW(ge::Operator& op, int32_t& stride_h, int32_t& stride_w, int32_t& dilation_h,
                                int32_t& dilation_w) {
  const int32_t DIM_SIZE_LIMIT = 4;
  std::string dataFormat = "";
  int32_t hPosition = 0;
  int32_t wPosition = 0;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", dataFormat)) {
    if (dataFormat != "NCHW" && dataFormat != "NHWC") {
      OP_LOGE(op.GetName().c_str(), "conv2DBackprop's dataFormat error, should be HCHW OR NHWC!");
      map<string, string> err_map;
      err_map["param"] = "data_Format";
      err_map["op_name"] = "conv2Dbp";
      err_map["expected_format_list"] = "HCHW OR NHWC";
      err_map["format"] = dataFormat;
      std::string report_error_code = "E50002";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return false;
    }

  } else {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr dataFormat failed!");
    map<string, string> err_map;
    err_map["op_name"] = "conv2Dbp";
    err_map["param_name"] = "data_Format";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  // get H & W position
  hPosition = dataFormat.find("H");
  wPosition = dataFormat.find("W");
  if (hPosition < 0 || wPosition < 0) {
    OP_LOGE(op.GetName().c_str(), "Get hPosition or wPosition failed!");
    map<string, string> err_map;
    err_map["op_name"] = "conv2Dbp";
    err_map["param_name"] = "hPosition and wPosition of dataFormat";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  std::vector<int32_t> strideList;
  if (GRAPH_SUCCESS == op.GetAttr("strides", strideList)) {
    if (strideList.empty()) {
      OP_LOGE(op.GetName().c_str(), "strideList from op is empty!");
      map<string, string> err_map;
      err_map["op_name"] = "conv2Dbp";
      err_map["param_name"] = "stride_List";
      std::string report_error_code = "E50030";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return false;
    }
    stride_h = strideList[hPosition];
    stride_w = strideList[wPosition];

  } else {
    OP_LOGE(op.GetName().c_str(), "Get strides list failed!");
    map<string, string> err_map;
    err_map["op_name"] = "conv2Dbp";
    err_map["param_name"] = "stride_List";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  // check dilations shape
  std::vector<int32_t> dilationsList;
  if (GRAPH_SUCCESS == op.GetAttr("dilations", dilationsList)) {
    if (dilationsList.size() != DIM_SIZE_LIMIT) {
      OP_LOGE(op.GetName().c_str(), "dilationsList list should be 4d");
      map<string, string> err_map;
      err_map["op_name"] = "Conv2DBackpropInput";
      err_map["param_name"] = "dilationsList";
      err_map["expected_length"] = std::to_string(DIM_SIZE_LIMIT);
      err_map["length"] = std::to_string(dilationsList.size());
      std::string report_error_code = "E50035";
      (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return false;
    }
    dilation_h = dilationsList[hPosition];
    dilation_w = dilationsList[wPosition];

  } else {
    OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
    map<string, string> err_map;
    err_map["op_name"] = "Conv2DBackpropInput";
    err_map["param_name"] = "dilations";
    std::string report_error_code = "E50030";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  return true;
}

template <typename T1, typename T2>
static bool SetPadListByPaddingConv2dbp(ge::Operator& op, std::vector<T1>& inputSizes, Format inputFormat,
                                        std::vector<T2>& filterSizes, Format filterFormat) {
  OP_LOGI(op.GetName().c_str(), "SetPadListByPaddingConv2dbp begin.");
  if (filterSizes.size() < 4 || inputSizes.size() < 4) {
    OP_LOGE(op.GetName().c_str(), "filter_sizes or inputSizes is illegal");
    map<string, string> err_map;
    err_map["op_name"] = "conv2Dbp";
    err_map["param_name"] = "filterSizes and inputSizes";
    err_map["expected_length"] = "4";
    err_map["length"] = std::to_string(filterSizes.size()) + " and " + std::to_string(inputSizes.size());
    std::string report_error_code = "E50035";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  CHECK_FORMAT(inputFormat);
  CHECK_FORMAT(filterFormat);

  int32_t stride_h = 0;
  int32_t stride_w = 0;
  int32_t dilation_h = 0;
  int32_t dilation_w = 0;
  if (false == getStrideDilationHW(op, stride_h, stride_w, dilation_h, dilation_w)) {
    OP_LOGE(op.GetName().c_str(), "op get strides failed.");
    map<string, string> err_map;
    err_map["op_name"] = "conv2Dbp";
    err_map["param_name"] = "stridet";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  std::string inputFormatStr = format2str[inputFormat];
  int32_t hInputPosition = inputFormatStr.find("H");
  CHECK_POSITION(hInputPosition);
  int32_t wInputPosition = inputFormatStr.find("W");
  CHECK_POSITION(wInputPosition);
  int32_t dx_h = inputSizes[hInputPosition];
  int32_t dx_w = inputSizes[wInputPosition];

  std::string filterFormatStr = format2str[filterFormat];
  int32_t hFilterPosition = filterFormatStr.find("H");
  CHECK_POSITION(hFilterPosition);
  int32_t wFilterPosition = filterFormatStr.find("W");
  CHECK_POSITION(wFilterPosition);
  int32_t filter_h = filterSizes[hFilterPosition];
  int32_t filter_w = filterSizes[wFilterPosition];

  int32_t filter_dilation_h = (filter_h - 1) * dilation_h + 1;
  int32_t filter_dilation_w = (filter_w - 1) * dilation_w + 1;
  std::string padding;
  std::vector<int32_t> pads;
  if (GRAPH_SUCCESS == op.GetAttr("padding", padding)) {
    int pad_h = 0;
    int32_t pad_up = 0;
    int32_t pad_down = 0;
    int pad_w = 0;
    int32_t pad_left = 0;
    int32_t pad_right = 0;
    if (padding == "SAME") {
      pad_h = std::max(ALIGN_CONV2DBP(dx_h, stride_h) - stride_h + filter_dilation_h - dx_h, 0);
      pad_up = pad_h / 2;
      pad_down = pad_h - pad_up;
      pad_w = std::max(ALIGN_CONV2DBP(dx_w, stride_w) - stride_w + filter_dilation_w - dx_w, 0);
      pad_left = pad_w / 2;
      pad_right = pad_w - pad_left;
    }
    pads.push_back(pad_up);
    pads.push_back(pad_down);
    pads.push_back(pad_left);
    pads.push_back(pad_right);

    op.SetAttr("pads", pads);
  }

  if (GRAPH_SUCCESS == op.GetAttr("pads", pads)) {
    if (pads.size() < 4) {
      OP_LOGE(op.GetName().c_str(), "op pads's size is illegal.");
      map<string, string> err_map;
      err_map["op_name"] = "conv2Dbp";
      err_map["param_name"] = "pads";
      err_map["expected_length"] = "4";
      err_map["length"] = std::to_string(pads.size());
      std::string report_error_code = "E50035";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return false;
    }
    if (pads[0] < 0 || pads[1] < 0 || pads[2] < 0 || pads[3] < 0) {
      OP_LOGE(op.GetName().c_str(), "op get pads is illegal");
      map<string, string> err_map;
      err_map["op_name"] = "conv2Dbp";
      err_map["param_name"] = "pads";
      err_map["expected_value"] = ">= 0";
      err_map["input_value"] = "[" + std::to_string(pads[0]) + ", " + std::to_string(pads[1]) + ", " +
                               std::to_string(pads[2]) + ", " + std::to_string(pads[3]) + ']';
      std::string report_error_code = "E50029";
      (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return false;
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "op get pads failed.");
    map<string, string> err_map;
    err_map["op_name"] = "conv2Dbp";
    err_map["param_name"] = "pads";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  OP_LOGI(op.GetName().c_str(), "op set pads succ.");
  return true;
}

static graphStatus VerifyConvPadding(ge::Operator& op) {
  std::string pad;
  if (GRAPH_SUCCESS == op.GetAttr("padding", pad)) {
    if (pad.compare("SAME") != 0 && pad.compare("VALID") != 0) {
      OP_LOGE(op.GetName().c_str(), "padding must be SAME or VALID.");
      map<string, string> err_map;
      err_map["op_name"] = op.GetName().c_str();
      err_map["expected_pad_mode"] = "SAME or VALID";
      err_map["actual_pad_mode"] = pad;
      std::string report_error_code = "E50050";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGW(op.GetName().c_str(), "get padding failed. try to get pads.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

static graphStatus VerifyConv2dbpPads(ge::Operator& op) {
  std::vector<int> pads;
  if (GRAPH_SUCCESS == op.GetAttr("pads", pads)) {
    if (pads.size() < 4) {
      OP_LOGE(op.GetName().c_str(), "op pads's size is illegal.");
      map<string, string> err_map;
      err_map["op_name"] = "conv2Dbp";
      err_map["param_name"] = "pads";
      err_map["expected_length"] = "4";
      err_map["length"] = std::to_string(pads.size());
      std::string report_error_code = "E50035";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
    if (pads[0] < 0 || pads[1] < 0 || pads[2] < 0 || pads[3] < 0) {
      OP_LOGE(op.GetName().c_str(), "op get pads is illegal");
      map<string, string> err_map;
      err_map["op_name"] = "conv2Dbp";
      err_map["param_name"] = "pads";
      err_map["expected_value"] = ">= 0";
      err_map["input_value"] = "[" + std::to_string(pads[0]) + ", " + std::to_string(pads[1]) + ", " +
                               std::to_string(pads[2]) + ", " + std::to_string(pads[3]) + ']';
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "op get pads failed.");
    map<string, string> err_map;
    err_map["op_name"] = "conv2Dbp";
    err_map["param_name"] = "pads";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
// ----------------Conv2DBackpropInput-------------------
static graphStatus VerifyConv2dbpInputCommon(const ge::Operator& op) {
  auto filter_desc = op.GetInputDesc("filter");
  auto out_backprop_desc = op.GetInputDesc("out_backprop");

  auto filter_dtype = filter_desc.GetDataType();
  auto out_backprop_dtype = out_backprop_desc.GetDataType();
  auto filter_shape = filter_desc.GetShape().GetDims();
  auto out_backprop_shape = out_backprop_desc.GetShape().GetDims();

  const int32_t DIM_SIZE_LIMIT = 4;
  const int32_t DIM_STRIDES_LIMIT = 4;

  // check input dtype
  if (filter_dtype != out_backprop_dtype) {
    OP_LOGE(op.GetName().c_str(), "filter's dtype should equal to out_backprop's dtype.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DBackpropInput";
    err_map["param1_name"] = "filter";
    err_map["param2_name"] = "out_backprop";
    err_map["param1_value"] = std::to_string(filter_dtype);
    err_map["param2_value"] = std::to_string(out_backprop_dtype);
    err_map["attr_name"] = "dtype";
    std::string report_error_code = "E50031";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check input tensor shape
  if (filter_shape.size() != DIM_SIZE_LIMIT) {
    OP_LOGE(op.GetName().c_str(), "filter's shape should be 4d.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DBackpropInput";
    err_map["param_name"] = "filterShape";
    err_map["expected_length"] = "4";
    err_map["length"] = std::to_string(filter_shape.size());
    std::string report_error_code = "E50035";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  if (out_backprop_shape.size() != DIM_SIZE_LIMIT) {
    OP_LOGE(op.GetName().c_str(), "out_backprop's shape should be 4d.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DBackpropInput";
    err_map["param_name"] = "out_backprop's shape";
    err_map["expected_length"] = "4";
    err_map["length"] = std::to_string(out_backprop_shape.size());
    std::string report_error_code = "E50035";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check strides shape
  std::vector<int32_t> stride_list;
  if (GRAPH_SUCCESS == op.GetAttr("strides", stride_list)) {
    if (stride_list.size() != DIM_STRIDES_LIMIT) {
      OP_LOGE(op.GetName().c_str(), "strides should be 4d.");
      map<std::string, std::string> err_map;
      err_map["op_name"] = "Conv2DBackpropInput";
      err_map["param_name"] = "strides's shape";
      err_map["expected_length"] = std::to_string(DIM_STRIDES_LIMIT);
      err_map["length"] = std::to_string(stride_list.size());
      std::string report_error_code = "E50035";
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "get strides list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DBackpropInput";
    err_map["param_name"] = "strides list";
    std::string report_error_code = "E50030";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check dilations shape
  std::vector<int32_t> dilations_list;
  if (GRAPH_SUCCESS == op.GetAttr("dilations", dilations_list)) {
    if (dilations_list.size() != DIM_SIZE_LIMIT) {
      OP_LOGE(op.GetName().c_str(), "dilations_list list should be 4d");
      map<std::string, std::string> err_map;
      err_map["op_name"] = "Conv2DBackpropInput";
      err_map["param_name"] = "dilations_list";
      err_map["expected_length"] = std::to_string(DIM_SIZE_LIMIT);
      err_map["length"] = std::to_string(dilations_list.size());
      std::string report_error_code = "E50035";
      (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DBackpropInput";
    err_map["param_name"] = "dilations";
    std::string report_error_code = "E50030";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(Conv2DBackpropInputInfer)
  OP_LOGI(op.GetName().c_str(), "Enter Conv2DBackpropInput inferfunction!");
  std::vector<std::string> input_infer_depends = {"input_size"};
  op_desc->SetOpInferDepends(input_infer_depends);
  std::vector<int64_t> dy_sizes = op.GetInputDesc("out_backprop").GetShape().GetDims();
  bool is_dynamic = false;
  // when static op or dynamic op phase_running, is_dynamic == False
  if (std::find(dy_sizes.begin(), dy_sizes.end(), -1) != dy_sizes.end()) {
    is_dynamic = true;
  }

  std::vector<int64_t> input_sizes;
  auto y_desc = op.GetOutputDesc("y");
  if (!is_dynamic) {
    Tensor input_sizes_tensor;
    if (GRAPH_SUCCESS != op.GetInputConstData("input_size", input_sizes_tensor)) {
      OP_LOGE(op.GetName().c_str(), "get input_size tensor failed.");
      map<std::string, std::string> err_map;
      err_map["op_name"] = "Conv2DBackpropInput";
      err_map["param_name"] = "input_size";
      std::string report_error_code = "E50030";
      (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
    // get shape for output from input_size
    auto input_sizes_desc = op.GetInputDesc("input_size");
    DataType dtype = input_sizes_desc.GetDataType();
    GetConstValue(input_sizes_tensor, dtype, input_sizes);
  } else {
    OP_LOGD(op.GetName().c_str(), "dynamic shape set range");
    std::vector<std::pair<int64_t, int64_t>> dx_range;
    op.GetInputDesc("input_size").GetShapeRange(dx_range);
    if (!dx_range.empty()) {
      y_desc.SetShapeRange(dx_range);
    }
    for (size_t i = 0; i < dx_range.size(); i++) {
      if (dx_range[i].first == dx_range[i].second) {
        input_sizes.push_back(dx_range[i].first);
      } else {
        input_sizes.push_back(-1);
      }
      OP_LOGD(op.GetName().c_str(), "dx Range[%u] is (%lld, %lld)", i, dx_range[i].first, dx_range[i].second);
    }
  }

  int64_t groups = 1;
  if (GRAPH_SUCCESS != op.GetAttr("groups", groups)) {
    OP_LOGI(op.GetName().c_str(), "no groups setting, use groups as 1");
  }

  // set dtype of output desc
  auto out_backprop_dtype = op.GetInputDesc("out_backprop").GetDataType();
  if (out_backprop_dtype == DT_INT8) {
    y_desc.SetDataType(DT_INT32);
  } else {
    y_desc.SetDataType(out_backprop_dtype);
  }

  // set shape of output desc, input_size should match the format of y
  if (input_sizes.size() == 4) {
    std::vector<int64_t> y_shape;
    y_shape.push_back(input_sizes[0]);
    y_shape.push_back(input_sizes[1]);
    y_shape.push_back(input_sizes[2]);
    y_shape.push_back(input_sizes[3]);
    y_desc.SetShape(ge::Shape(y_shape));
  }

  auto dx_shape = y_desc.GetShape().GetDims();
  for (size_t i = 0; i < dx_shape.size(); i++) {
    OP_LOGD(op.GetName().c_str(), "dx_shape [%u] is %d", i, (int32_t)dx_shape[i]);
  }

  // update output desc
  if (GRAPH_SUCCESS != op.UpdateOutputDesc("y", y_desc)) {
    OP_LOGE(op.GetName().c_str(), "update output desc failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DBackpropInput";
    err_map["param_name"] = "updating result";
    err_map["rule_desc"] = "updata OutputDesc";
    err_map["param_value"] = "failed";
    std::string report_error_code = "E50012";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> filter_sizes = op.GetInputDesc("filter").GetShape().GetDims();
  Format filter_format = op.GetInputDesc("filter").GetFormat();
  Format input_format = y_desc.GetFormat();
  CHECK_FORMAT(filter_format);
  CHECK_FORMAT(input_format);

  // update pads list by padding[SAME,VALID]
  if (!is_dynamic) {
    if (false == SetPadListByPaddingConv2dbp(op, input_sizes, input_format, filter_sizes, filter_format)) {
      OP_LOGE(op.GetName().c_str(), "update pads list by padding failed.");
      map<std::string, std::string> err_map;
      err_map["op_name"] = "Conv2DBackpropInput";
      err_map["param_name"] = "updding result";
      err_map["rule_desc"] = "updata pads list by padding";
      err_map["param_value"] = "failed";
      std::string report_error_code = "E50012";
      (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  }

  OP_LOGI(op.GetName().c_str(), "Leaving Conv2DBackpropInput inferfunction!");
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

IMPLEMT_VERIFIER(Conv2DBackpropInput, Conv2DBackpropInputVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter Conv2DBackpropInput verifyfunction!");
  if (GRAPH_SUCCESS != VerifyConv2dbpInputCommon(op)) {
    return GRAPH_FAILED;
  }
  // check padding value
  if (GRAPH_SUCCESS == VerifyConvPadding(op) || GRAPH_SUCCESS == VerifyConv2dbpPads(op)) {
    OP_LOGI(op.GetName().c_str(), "Leaving Conv2DBackpropInput verifyfunction!");
    return GRAPH_SUCCESS;
  } else {
    OP_LOGE(op.GetName().c_str(), "Leaving Conv2DBackpropInput verifyfunction!");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "verify pads and verify padding";
    err_map["op_name"] = "Conv2DBackpropInput";
    err_map["excepted_value"] = std::to_string(GRAPH_SUCCESS);
    err_map["output_value"] = std::to_string(GRAPH_FAILED);
    std::string report_error_code = "E50029";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
}

INFER_FUNC_REG(Conv2DBackpropInput, Conv2DBackpropInputInfer);
VERIFY_FUNC_REG(Conv2DBackpropInput, Conv2DBackpropInputVerify);

// ----------------Conv2DBackpropInputD-------------------
IMPLEMT_INFERFUNC(Conv2DBackpropInputD, Conv2DBackpropInputDInfer) {
  OP_LOGI(op.GetName().c_str(), "Enter Conv2DBackpropInputD inferfunction!");
  const int32_t DIM_SIZE_LIMIT = 4;

  auto out_backprop_desc = op.GetInputDesc("out_backprop");
  // get dtype for output from out_backprop
  auto out_backprop_dtype = out_backprop_desc.GetDataType();
  // get shape for output from input_size
  std::vector<int32_t> input_sizes;
  if (GRAPH_SUCCESS == op.GetAttr("input_size", input_sizes)) {
    if (input_sizes.size() != DIM_SIZE_LIMIT) {
      OP_LOGE(op.GetName().c_str(), "input_size list should be 4d.");
      map<std::string, std::string> err_map;
      err_map["op_name"] = "Conv2DBackpropInput";
      err_map["param_name"] = "input_size";
      err_map["expected_length"] = "4";
      err_map["length"] = std::to_string(input_sizes.size());
      std::string report_error_code = "E50035";
      (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "get input_size list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DBackpropInput";
    err_map["param_name"] = "input_size";
    std::string report_error_code = "E50030";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // set dtype of output desc
  auto y_desc = op.GetOutputDesc("y");
  if (out_backprop_dtype == DT_INT8) {
    y_desc.SetDataType(DT_INT32);
  } else {
    y_desc.SetDataType(out_backprop_dtype);
  }

  // set shape of output desc, input_size should match the format of y
  std::vector<int64_t> y_shape;
  y_shape.push_back(input_sizes[0]);
  y_shape.push_back(input_sizes[1]);
  y_shape.push_back(input_sizes[2]);
  y_shape.push_back(input_sizes[3]);
  y_desc.SetShape(ge::Shape(y_shape));

  // update output desc
  if (GRAPH_SUCCESS != op.UpdateOutputDesc("y", y_desc)) {
    OP_LOGE(op.GetName().c_str(), "update output desc failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DBackpropInput";
    err_map["param_name"] = "updating result";
    err_map["rule_desc"] = "updata OutputDesc";
    err_map["param_value"] = "failed";
    std::string report_error_code = "E50012";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> filter_sizes = op.GetInputDesc("filter").GetShape().GetDims();
  Format filter_format = op.GetInputDesc("filter").GetFormat();
  Format input_format = y_desc.GetFormat();
  CHECK_FORMAT(filter_format);
  CHECK_FORMAT(input_format);

  // update pads list by padding[SAME,VALID]
  if (false == SetPadListByPaddingConv2dbp(op, input_sizes, input_format, filter_sizes, filter_format)) {
    OP_LOGE(op.GetName().c_str(), "Conv2DBackpropInputD update pads list by padding failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DBackpropInput";
    err_map["param_name"] = "updating result";
    err_map["rule_desc"] = "updata pads list by padding";
    err_map["param_value"] = "failed";
    std::string report_error_code = "E50012";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  OP_LOGI(op.GetName().c_str(), "Leaving Conv2DBackpropInputD inferfunction!");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv2DBackpropInputD, Conv2DBackpropInputDVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter Conv2DBackpropInputD verifyfunction!");
  if (GRAPH_SUCCESS != VerifyConv2dbpInputCommon(op)) {
    return GRAPH_FAILED;
  }
  // check padding value
  if (GRAPH_SUCCESS != VerifyConv2dbpPads(op)) {
    return GRAPH_FAILED;
  }
  OP_LOGI(op.GetName().c_str(), "Leaving Conv2DBackpropInputD verifyfunction!");
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Conv2DBackpropInputD, Conv2DBackpropInputDInfer);
VERIFY_FUNC_REG(Conv2DBackpropInputD, Conv2DBackpropInputDVerify);

// ----------------Conv2DBackpropFilter-------------------
static graphStatus VerifyConv2dbpFilterCommon(const ge::Operator& op) {
  auto x_desc = op.GetInputDesc("x");
  auto out_backprop_desc = op.GetInputDesc("out_backprop");
  auto x_dtype = x_desc.GetDataType();
  auto out_backprop_dtype = out_backprop_desc.GetDataType();
  auto x_shape = x_desc.GetShape().GetDims();
  auto out_backprop_shape = out_backprop_desc.GetShape().GetDims();

  const int32_t DIM_SIZE_LIMIT = 4;
  const int32_t DIM_STRIDES_LIMIT = 4;

  // check input dtype
  if (x_dtype != out_backprop_dtype) {
    OP_LOGE(op.GetName().c_str(), "filter's dtype should equal to out_backprop's dtype.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DBackpropFilter";
    err_map["param1_name"] = "filter";
    err_map["param2_name"] = "out_backprop";
    err_map["param1_value"] = std::to_string(x_dtype);
    err_map["param2_value"] = std::to_string(out_backprop_dtype);
    err_map["attr_name"] = "dtype";
    std::string report_error_code = "E50031";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check input tensor shape
  if (x_shape.size() != DIM_SIZE_LIMIT) {
    OP_LOGE(op.GetName().c_str(), "x's shape should be 4d.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DBackpropFilter";
    err_map["param_name"] = "x's shape";
    err_map["expected_length"] = "4";
    err_map["length"] = std::to_string(x_shape.size());
    std::string report_error_code = "E50035";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  if (out_backprop_shape.size() != DIM_SIZE_LIMIT) {
    OP_LOGE(op.GetName().c_str(), "out_backprop's shape should be 4d.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DBackpropFilter";
    err_map["param_name"] = "out_backprop's shape";
    err_map["expected_length"] = "4";
    err_map["length"] = std::to_string(out_backprop_shape.size());
    std::string report_error_code = "E50035";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check strides shape
  std::vector<int32_t> stride_list;
  if (GRAPH_SUCCESS == op.GetAttr("strides", stride_list)) {
    if (stride_list.size() != DIM_STRIDES_LIMIT) {
      OP_LOGE(op.GetName().c_str(), "strides should be 4d.");
      map<std::string, std::string> err_map;
      err_map["op_name"] = "Conv2DBackpropFilter";
      err_map["param_name"] = "strides's shape";
      err_map["expected_length"] = std::to_string(DIM_STRIDES_LIMIT);
      err_map["length"] = std::to_string(stride_list.size());
      std::string report_error_code = "E50035";
      (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "get strides list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DBackpropFilter";
    err_map["param_name"] = "strides list";
    std::string report_error_code = "E50030";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check dilations shape
  std::vector<int32_t> dilations_list;
  if (GRAPH_SUCCESS == op.GetAttr("dilations", dilations_list)) {
    if (dilations_list.size() != DIM_SIZE_LIMIT) {
      OP_LOGE(op.GetName().c_str(), "dilations_list list should be 4d.");
      map<std::string, std::string> err_map;
      err_map["op_name"] = "Conv2DBackpropFilter";
      err_map["param_name"] = "dilations_list";
      err_map["expected_length"] = std::to_string(DIM_SIZE_LIMIT);
      err_map["length"] = std::to_string(dilations_list.size());
      std::string report_error_code = "E50035";
      (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DBackpropFilter";
    err_map["param_name"] = "dilations";
    std::string report_error_code = "E50035";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(Conv2DBackpropFilterInfer)
  OP_LOGI(op.GetName().c_str(), "Enter Conv2DBackpropFilter inferfunction!");
  std::vector<std::string> input_infer_depends = {"filter_size"};
  op_desc->SetOpInferDepends(input_infer_depends);

  Tensor filter_sizes_tensor;
  if (GRAPH_SUCCESS != op.GetInputConstData("filter_size", filter_sizes_tensor)) {
    OP_LOGE(op.GetName().c_str(), "get filter_size tensor failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DBackpropFilter";
    err_map["param_name"] = "filter_size";
    std::string report_error_code = "E50030";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  // get shape for output from filter_size
  auto filter_sizes_desc = op.GetInputDesc("filter_size");
  DataType dtype = filter_sizes_desc.GetDataType();
  std::vector<int64_t> filter_sizes;
  GetConstValue(filter_sizes_tensor, dtype, filter_sizes);

  // set dtype of output desc
  auto y_desc = op.GetOutputDesc("y");
  auto out_backprop_dtype = op.GetInputDesc("out_backprop").GetDataType();
  y_desc.SetDataType(out_backprop_dtype);
  // set shape of output desc, filter_size should match the format of y
  std::vector<int64_t> y_shape;
  y_shape.push_back(filter_sizes[0]);
  y_shape.push_back(filter_sizes[1]);
  y_shape.push_back(filter_sizes[2]);
  y_shape.push_back(filter_sizes[3]);
  y_desc.SetShape(ge::Shape(y_shape));

  // update output desc
  if (GRAPH_SUCCESS != op.UpdateOutputDesc("y", y_desc)) {
    OP_LOGE(op.GetName().c_str(), "update output desc failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DBackpropFilter";
    err_map["param_name"] = "updating result";
    err_map["rule_desc"] = "updata OutputDesc";
    err_map["param_value"] = "failed";
    std::string report_error_code = "E50012";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> x_sizes = op.GetInputDesc("x").GetShape().GetDims();
  Format x_format = op.GetInputDesc("x").GetFormat();
  Format filter_format = y_desc.GetFormat();
  CHECK_FORMAT(x_format);
  CHECK_FORMAT(filter_format);

  bool is_dynamic = false;
  if (std::find(x_sizes.begin(), x_sizes.end(), -1) != x_sizes.end()) {
    is_dynamic = true;
  }

  if (!is_dynamic) {
    // update pads list by padding[SAME,VALID]
    if (false == SetPadListByPaddingConv2dbp(op, x_sizes, x_format, filter_sizes, filter_format)) {
      OP_LOGE(op.GetName().c_str(), "update pads list by padding failed.");
      map<std::string, std::string> err_map;
      err_map["op_name"] = "Conv2DBackpropFilter";
      err_map["param_name"] = "updding result";
      err_map["rule_desc"] = "updata pads list by padding";
      err_map["param_value"] = "failed";
      std::string report_error_code = "E50012";
      (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGD(op.GetName().c_str(), "Do not update pads in dynamic_shape");
  }
  OP_LOGI(op.GetName().c_str(), "Leaving Conv2DBackpropFilter inferfunction!");
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

IMPLEMT_VERIFIER(Conv2DBackpropFilter, Conv2DBackpropFilterVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter Conv2DBackpropFilter verifyfunction!");
  if (GRAPH_SUCCESS != VerifyConv2dbpFilterCommon(op)) {
    return GRAPH_FAILED;
  }
  // check padding value
  if (GRAPH_SUCCESS == VerifyConvPadding(op) || GRAPH_SUCCESS == VerifyConv2dbpPads(op)) {
    OP_LOGI(op.GetName().c_str(), "Leaving Conv2DBackpropFilter verifyfunction!");
    return GRAPH_SUCCESS;
  } else {
    OP_LOGE(op.GetName().c_str(), "Leaving Conv2DBackpropFilter verifyfunction!");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "verify pads and verify padding result";
    err_map["op_name"] = "Conv2DBackpropInput";
    err_map["rule_desc"] = "verify pads and verify padding";
    err_map["param_value"] = "failed";
    std::string report_error_code = "E50012";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
}

INFER_FUNC_REG(Conv2DBackpropFilter, Conv2DBackpropFilterInfer);
VERIFY_FUNC_REG(Conv2DBackpropFilter, Conv2DBackpropFilterVerify);

// ----------------Conv2DBackpropFilterD-------------------
IMPLEMT_INFERFUNC(Conv2DBackpropFilterD, Conv2DBackpropFilterDInfer) {
  OP_LOGI(op.GetName().c_str(), "Enter Conv2DBackpropFilterD inferfunction!");
  const int32_t DIM_SIZE_LIMIT = 4;

  // get shape for output from filter_size
  std::vector<int32_t> filter_sizes;
  if (GRAPH_SUCCESS == op.GetAttr("filter_size", filter_sizes)) {
    if (filter_sizes.size() != DIM_SIZE_LIMIT) {
      OP_LOGE(op.GetName().c_str(), "filter_size list should be 4d.");
      map<std::string, std::string> err_map;
      err_map["op_name"] = "Conv2DBackpropFilter";
      err_map["param_name"] = "filter_size";
      err_map["expected_length"] = "4";
      err_map["length"] = std::to_string(filter_sizes.size());
      std::string report_error_code = "E50035";
      (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "get filter_size list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DBackpropFilter";
    err_map["param_name"] = "filter_size";
    std::string report_error_code = "E50030";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  auto y_desc = op.GetOutputDesc("y");
  // set dtype of output desc
  y_desc.SetDataType(DT_FLOAT);

  // set shape of output desc, filter_size should match the format of y
  std::vector<int64_t> y_shape;
  y_shape.push_back(filter_sizes[0]);
  y_shape.push_back(filter_sizes[1]);
  y_shape.push_back(filter_sizes[2]);
  y_shape.push_back(filter_sizes[3]);
  y_desc.SetShape(ge::Shape(y_shape));

  // update output desc
  if (GRAPH_SUCCESS != op.UpdateOutputDesc("y", y_desc)) {
    OP_LOGE(op.GetName().c_str(), "update output desc failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DBackpropFilter";
    err_map["param_name"] = "updating result";
    err_map["rule_desc"] = "updata OutputDesc";
    err_map["param_value"] = "failed";
    std::string report_error_code = "E50012";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> x_sizes = op.GetInputDesc("x").GetShape().GetDims();
  Format x_format = op.GetInputDesc("x").GetFormat();
  Format filter_format = y_desc.GetFormat();
  CHECK_FORMAT(x_format);
  CHECK_FORMAT(filter_format);

  // update pads list by padding[SAME,VALID]
  if (false == SetPadListByPaddingConv2dbp(op, x_sizes, x_format, filter_sizes, filter_format)) {
    OP_LOGE(op.GetName().c_str(), "update pads list by padding failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DBackpropFilter";
    err_map["param_name"] = "updding result";
    err_map["rule_desc"] = "updata pads list by padding";
    err_map["param_value"] = "failed";
    std::string report_error_code = "E50012";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  OP_LOGI(op.GetName().c_str(), "Leaving Conv2DBackpropFilterD inferfunction!");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv2DBackpropFilterD, Conv2DBackpropFilterDVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter Conv2DBackpropFilterD verifyfunction!");
  if (GRAPH_SUCCESS != VerifyConv2dbpFilterCommon(op)) {
    return GRAPH_FAILED;
  }
  // check padding value
  if (GRAPH_SUCCESS != VerifyConv2dbpPads(op)) {
    return GRAPH_FAILED;
  }
  OP_LOGI(op.GetName().c_str(), "Leaving Conv2DBackpropFilterD verifyfunction!");
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Conv2DBackpropFilterD, Conv2DBackpropFilterDInfer);
VERIFY_FUNC_REG(Conv2DBackpropFilterD, Conv2DBackpropFilterDVerify);

// --------------------------Conv2D------------------------------
/*!
  * @brief Convert different framework pad param to ir pads:
  *
  * [_padding]: 4D lsit, format sensitive, need convert to pads
  * [padding]: 'SAME' or 'VALID', need convert to pads
  * [pads]: 4D list, format sensitive, no need convert
  *
  * @param op Conv2D operator.
  * @param ih, iw  Input images H/W size.
  * @param kh, kw  Input filter H/W size.
  * @param strh, strw  Input stride H/W value.
  * @param dilh, dilw  Input dilation H/W value.
  * @param padt, padb, padl, padr Top, bottom, left, right padding.
  * @return bool Whether the pads setting is correct.
  */
static bool GetPadConv2D(ge::Operator& op, int32_t ih, int32_t iw, int32_t kh, int32_t kw, int32_t strh, int32_t strw,
                         int32_t dilh, int32_t dilw, int32_t& padt, int32_t& padb, int32_t& padl, int32_t& padr) {
  std::string pad_str;
  std::vector<int32_t> pad_list;
  if (GRAPH_SUCCESS == op.GetAttr("padding", pad_str) && pad_str.compare("EXPLICIT") != 0) {
    if (pad_str.compare("SAME") == 0) {
      int32_t tails_h = ih % strh;
      int32_t tails_w = iw % strw;
      int32_t dkh = dilh * (kh - 1) + 1;
      int32_t dkw = dilw * (kw - 1) + 1;
      int32_t pad_h = std::max((tails_h > 0 ? dkh - tails_h : dkh - strh), 0);
      int32_t pad_w = std::max((tails_w > 0 ? dkw - tails_w : dkw - strw), 0);
      pad_list.push_back(pad_h / 2);
      pad_list.push_back(pad_h / 2 + pad_h % 2);
      pad_list.push_back(pad_w / 2);
      pad_list.push_back(pad_w / 2 + pad_w % 2);
    } else if (pad_str.compare("VALID") == 0) {
      pad_list.push_back(0);
      pad_list.push_back(0);
      pad_list.push_back(0);
      pad_list.push_back(0);
    } else {
      OP_LOGE(op.GetName().c_str(),
              "padding should be SAME or VALID."
              " actual is: %s.",
              pad_str.c_str());
      map<string, string> err_map;
      err_map["op_name"] = op.GetName().c_str();
      err_map["expected_pad_mode"] = "SAME or VALID";
      err_map["actual_pad_mode"] = pad_str;
      std::string report_error_code = "E50050";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return false;
    }
    op.SetAttr("pads", pad_list);
  }

  // handle attr auto_pad from ONNX
  if (GRAPH_SUCCESS == op.GetAttr("auto_pad", pad_str)) {
    if (pad_str.compare("SAME_UPPER") == 0) {
      int32_t tails_h = ih % strh;
      int32_t tails_w = iw % strw;
      int32_t dkh = dilh * (kh - 1) + 1;
      int32_t dkw = dilw * (kw - 1) + 1;
      int32_t pad_h = std::max((tails_h > 0 ? dkh - tails_h : dkh - strh), 0);
      int32_t pad_w = std::max((tails_w > 0 ? dkw - tails_w : dkw - strw), 0);
      pad_list.push_back(pad_h / 2);
      pad_list.push_back(pad_h / 2 + pad_h % 2);
      pad_list.push_back(pad_w / 2);
      pad_list.push_back(pad_w / 2 + pad_w % 2);
      op.SetAttr("pads", pad_list);
    } else if (pad_str.compare("SAME_LOWER") == 0) {
      int32_t tails_h = ih % strh;
      int32_t tails_w = iw % strw;
      int32_t dkh = dilh * (kh - 1) + 1;
      int32_t dkw = dilw * (kw - 1) + 1;
      int32_t pad_h = std::max((tails_h > 0 ? dkh - tails_h : dkh - strh), 0);
      int32_t pad_w = std::max((tails_w > 0 ? dkw - tails_w : dkw - strw), 0);
      pad_list.push_back(pad_h / 2 + pad_h % 2);
      pad_list.push_back(pad_h / 2);
      pad_list.push_back(pad_w / 2 + pad_w % 2);
      pad_list.push_back(pad_w / 2);
      op.SetAttr("pads", pad_list);
    } else if (pad_str.compare("NOTSET") == 0) {
    } else if (pad_str.compare("VALID") == 0) {
      pad_list.push_back(0);
      pad_list.push_back(0);
      pad_list.push_back(0);
      pad_list.push_back(0);
      op.SetAttr("pads", pad_list);
    } else {
      OP_LOGE(op.GetName().c_str(),
              "padding should be SAME or VALID."
              " actual is: %s.",
              pad_str.c_str());
      map<string, string> err_map;
      err_map["op_name"] = op.GetName().c_str();
      err_map["expected_pad_mode"] = "NOTSET, SAME_UPPER, SAME_LOWER or VALID";
      err_map["actual_pad_mode"] = pad_str;
      std::string report_error_code = "E50050";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);

      return false;
    }
  }

  std::vector<int32_t> pads_list;
  op.GetAttr("pads", pads_list);
  auto p_size = pads_list.size();
  if (pads_list.empty() || p_size != 4) {
    OP_LOGE(op.GetName().c_str(), "pads list should be 4D. actual is: %u.", p_size);
    map<string, string> err_map;
    err_map["param_name"] = "pads";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_value"] = "4D";
    err_map["input_value"] = std::to_string(p_size) + "D.";
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  padt = pads_list[0];
  padb = pads_list[1];
  padl = pads_list[2];
  padr = pads_list[3];
  if(op.GetOpType() == "Conv2D") {
    int32_t ho = (ih + padt + padb - (kh - 1) * dilh - 1) / strh + 1;
    int32_t wo = (iw + padl + padr - (kw - 1) * dilw - 1) / strw + 1;
    int32_t hr = (ih + padt + padb - (kh - 1) * dilh - 1) % strh;
    int32_t wr = (iw + padl + padr - (kw - 1) * dilw - 1) % strw;
    if ((ho == 1 && hr <= padb) || (wo == 1 && wr <= padr)) {
      if (ho == 1 && hr <= padb) {
          padb -= hr;
          pads_list[1] = padb;
      }
      if (wo == 1 && wr <= padr) {
          padr -= wr;
          pads_list[3] = padr;
      }
      op.SetAttr("pads", pads_list);
    }
  }
  if (padt < 0 || padb < 0 || padl < 0 || padr < 0) {
    OP_LOGE(op.GetName().c_str(),
            "pads should be positive, "
            " actual is [%d,%d,%d,%d].",
            padt, padb, padl, padr);
    map<string, string> err_map;
    err_map["param_name"] = "pads";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_value"] = "positive";
    err_map["input_value"] =
        std::to_string(padt) + ", " + std::to_string(padb) + ", " + std::to_string(padl) + ", " + std::to_string(padr);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  return true;
}

/*!
  * @brief Get 2D(H/W) stride and dilation params to infershape output.
  *
  * [strides]: 4D list, format sensitive, according to first input tensor format
  * [dilations]: 4D list, format sensitive
  *
  * @param op Conv2D operator.
  * @param refer Valid value reference format.
  * @param strh, strw  Input stride H/W value.
  * @param dilh, dilw  Input dilation H/W value.
  * @return bool Whether the strides, dilations settings are correct.
  */
static bool GetAttrsConv2D(ge::Operator& op, Format refer, int32_t& strh, int32_t& strw, int32_t& dilh, int32_t& dilw) {
  std::vector<int32_t> stride_list;
  op.GetAttr("strides", stride_list);
  auto s_size = stride_list.size();
  if (stride_list.empty() || s_size != 4) {
    OP_LOGE(op.GetName().c_str(), "strides list should be 4D. actual is: %u.", s_size);
    map<string, string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_value"] = "4D";
    err_map["input_value"] = std::to_string(s_size) + "D.";
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  std::vector<int32_t> dilation_list;
  op.GetAttr("dilations", dilation_list);
  auto d_size = dilation_list.size();
  if (dilation_list.empty() || d_size != 4) {
    OP_LOGE(op.GetName().c_str(), "dilations list should be 4D. actual is: %u.", d_size);
    map<string, string> err_map;
    err_map["param_name"] = "dilations";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_value"] = "4D";
    err_map["input_value"] = std::to_string(d_size) + "D.";
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (refer == FORMAT_NCHW) {
    strh = stride_list[2];
    strw = stride_list[3];
    dilh = dilation_list[2];
    dilw = dilation_list[3];
  } else if (refer == FORMAT_NHWC) {
    strh = stride_list[1];
    strw = stride_list[2];
    dilh = dilation_list[1];
    dilw = dilation_list[2];
  }
  if (strh <= 0 || strw <= 0) {
    OP_LOGE(op.GetName().c_str(),
            "strides should be positive,"
            " actual is [%d,%d].",
            strh, strw);
    map<string, string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_value"] = "positive";
    err_map["input_value"] = std::to_string(strh) + ", " + std::to_string(strw);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  if (dilh <= 0 || dilw <= 0) {
    OP_LOGE(op.GetName().c_str(),
            "dilations should be positive,"
            " actual is [%d,%d].",
            dilh, dilw);
    map<string, string> err_map;
    err_map["param_name"] = "dilations";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_value"] = "positive";
    err_map["input_value"] = std::to_string(dilh) + ", " + std::to_string(dilw);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  return true;
}

/*!
  * @brief Infer output shape and dtype, dtype is same to first input tensor, Output
  *        format is set by ge parser process already.
  * @param Conv2DInfer Conv2D infershape function.
  * @return Status The processing flow result.
  */
IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(Conv2DInfer)
  OP_LOGD(op.GetName().c_str(), "Enter Conv2DInfer.");
  auto x_tensor = op.GetInputDesc("x");
  auto w_tensor = op.GetInputDesc("filter");

  auto x_shape = x_tensor.GetShape().GetDims();
  auto w_shape = w_tensor.GetShape().GetDims();
  if (x_shape.size() != 4 || w_shape.size() != 4) {
    return GRAPH_FAILED;
  }
  auto x_format = x_tensor.GetFormat();
  auto w_format = w_tensor.GetFormat();
  CHECK_FORMAT(x_format);
  CHECK_FORMAT(w_format);

  int32_t in = 0;
  int32_t ic = 0;
  int32_t ih = 0;
  int32_t iw = 0;
  int32_t kn = 0;
  int32_t kc = 0;
  int32_t kh = 0;
  int32_t kw = 0;
  if (x_format == FORMAT_NCHW) {
    in = x_shape[0];
    ic = x_shape[1];
    ih = x_shape[2];
    iw = x_shape[3];
  } else if (x_format == FORMAT_NHWC) {
    in = x_shape[0];
    ic = x_shape[3];
    ih = x_shape[1];
    iw = x_shape[2];
  } else {
    OP_LOGE(op.GetName().c_str(),
            "input x format should be NCHW or NHWC."
            " actual is: %s",
            TypeUtils::FormatToSerialString(x_format).c_str());
    map<string, string> err_map;
    err_map["param"] = "x";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_format_list"] = "NCHW or NHWC";
    err_map["format"] = TypeUtils::FormatToSerialString(x_format);
    std::string report_error_code = "E50002";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  if (w_format == FORMAT_NCHW) {
    kn = w_shape[0];
    kc = w_shape[1];
    kh = w_shape[2];
    kw = w_shape[3];
  } else if (w_format == FORMAT_NHWC) {
    kn = w_shape[0];
    kc = w_shape[3];
    kh = w_shape[1];
    kw = w_shape[2];
  } else if (w_format == FORMAT_HWCN) {
    kn = w_shape[3];
    kc = w_shape[2];
    kh = w_shape[0];
    kw = w_shape[1];
  } else {
    OP_LOGE(op.GetName().c_str(),
            "input filter format should be NCHW, NHWC or HWCN."
            " actual is: %s",
            TypeUtils::FormatToSerialString(w_format).c_str());
    map<string, string> err_map;
    err_map["param"] = "filter";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_format_list"] = "NCHW, NHWC or HWCN";
    err_map["format"] = TypeUtils::FormatToSerialString(w_format);
    std::string report_error_code = "E50002";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  auto bias_tensor = op.GetInputDesc("bias");
  auto bias_shape = bias_tensor.GetShape().GetDims();
  if (bias_shape.size() == 1 && bias_shape[0] != kn) {
    OP_LOGE(op.GetName().c_str(), "input bias size should be equal to out_channels.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "input bias size should be equal to out_channels.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  } else if (bias_shape.size() > 1) {
    OP_LOGE(op.GetName().c_str(), "input bias shape should be 1D.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "input bias shape should be 1D.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // set data_format: copy value of x_format to data_format
  // data_format will be used to infer position of H/W
  // in strides and dilations(previously used ori_format)
  std::string data_format;
  std::string attr_data_format = "data_format";
  std::string data_format_NCHW = "NCHW";
  std::string data_format_NHWC = "NHWC";

  if (GRAPH_SUCCESS == op.GetAttr(attr_data_format, data_format)) {
    OP_LOGI(data_format.c_str(), "conv before set data_format");
  }

  if (x_format == ge::FORMAT_NCHW) {
    op.SetAttr(attr_data_format, data_format_NCHW);
  } else {
    op.SetAttr(attr_data_format, data_format_NHWC);
  }

  op.GetAttr(attr_data_format, data_format);
  OP_LOGI(data_format.c_str(), "conv after set data_format");

  int64_t groups = 1;
  op.GetAttr("groups", groups);
  if (ic != kc * groups) {
    OP_LOGE(op.GetName().c_str(),
            "x channel should be equal to filter channel*groups. "
            "x format is: %s, filter format is: %s, "
            "x shape is: [%d,%d,%d,%d], filter shape is: [%d,%d,%d,%d], "
            "groups is: %d.",
            TypeUtils::FormatToSerialString(x_format).c_str(), TypeUtils::FormatToSerialString(w_format).c_str(),
            (int)x_shape[0], (int)x_shape[1], (int)x_shape[2], (int)x_shape[3], (int)w_shape[0], (int)w_shape[1],
            (int)w_shape[2], (int)w_shape[3], (int)groups);
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["x_shape"] = std::to_string(x_shape[0]) + ", " + std::to_string(x_shape[1]) + ", " +
                         std::to_string(x_shape[2]) + ", " + std::to_string(x_shape[3]);
    err_map["filter_shape"] = std::to_string(w_shape[0]) + ", " + std::to_string(w_shape[1]) + ", " +
                              std::to_string(w_shape[2]) + ", " + std::to_string(w_shape[3]);
    err_map["groups"] = std::to_string(groups);

    std::string report_error_code = "E50059";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  if (kn % groups != 0) {
    OP_LOGE(op.GetName().c_str(), "out_channels should be divisible by groups.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "out_channels should be divisible by groups.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;
  int32_t padt = 0;
  int32_t padb = 0;
  int32_t padl = 0;
  int32_t padr = 0;
  if (!GetAttrsConv2D(op, x_format, strh, strw, dilh, dilw) ||
      !GetPadConv2D(op, ih, iw, kh, kw, strh, strw, dilh, dilw, padt, padb, padl, padr)) {
    return GRAPH_FAILED;
  }

  int64_t ihPad = (ih + padt + padb - dilh * (kh - 1) - 1);
  int64_t iwPad = (iw + padl + padr - dilw * (kw - 1) - 1);
  int64_t oh = ihPad / strh + 1;
  int64_t ow = iwPad / strw + 1;
  if ((ih > 0) && (kh > 0) && (iw > 0) && (kw > 0)) {
    if ((ihPad < 0) || (iwPad < 0)) {
      OP_LOGE(op.GetName().c_str(), "image size after padding should be greater than or equal to filter size.");
      map<string, string> err_map;
      err_map["op_name"] = op.GetName().c_str();
      err_map["description"] = "image size after padding should be greater than or equal to filter size.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    }
  }

  vector<int64_t> y_shape;
  auto y_tensor = op.GetOutputDesc("y");
  auto y_format = y_tensor.GetFormat();
  CHECK_FORMAT(y_format)
  if (y_format == FORMAT_NCHW) {
    y_shape.push_back(in);
    y_shape.push_back(kn);
    y_shape.push_back(oh);
    y_shape.push_back(ow);
  } else if (y_format == FORMAT_NHWC) {
    y_shape.push_back(in);
    y_shape.push_back(oh);
    y_shape.push_back(ow);
    y_shape.push_back(kn);
  } else {
    OP_LOGE(op.GetName().c_str(),
            "output y format should be NCHW or NHWC."
            " actual is: %s",
            TypeUtils::FormatToSerialString(y_format).c_str());
    map<string, string> err_map;
    err_map["param"] = "y";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_format_list"] = "NCHW or NHWC";
    err_map["format"] = TypeUtils::FormatToSerialString(y_format);
    std::string report_error_code = "E50002";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  y_tensor.SetShape(Shape(y_shape));
  auto x_dtype = x_tensor.GetDataType();
  if (x_dtype == ge::DT_INT8) {
    y_tensor.SetDataType(ge::DT_INT32);
  } else {
    y_tensor.SetDataType(x_dtype);
  }

  // set Range
  bool is_dynamic = false;
  // when static op or dynamic op phase_running, is_dynamic == False
  if (std::find(x_shape.begin(), x_shape.end(), -1) != x_shape.end()) {
    is_dynamic = true;
  }
  if (is_dynamic) {
    size_t idx_n = 0;
    size_t idx_h = 0;
    size_t idx_w = 0;
    size_t idx_c = 0;
    if (x_format == FORMAT_NHWC) {
      idx_h = 1;
      idx_w = 2;
      idx_c = 3;
    } else if (x_format == FORMAT_NCHW) {
      idx_c = 1;
      idx_h = 2;
      idx_w = 3;
    }

    // update pads if padding is SAME
    std::string pad_str;
    if (GRAPH_SUCCESS == op.GetAttr("padding", pad_str) && pad_str == "SAME" && x_shape[idx_n] != -1) {
      op.SetAttr("pads", {-1, -1, -1, -1});
      OP_LOGD(op.GetName().c_str(), "set pads to {-1, -1, -1, -1} when padding is SAME in dynamic_shape");
    }

    OP_LOGD(op.GetName().c_str(), "dynamic shape set range");
    std::vector<std::pair<int64_t, int64_t>> fm_range;
    x_tensor.GetShapeRange(fm_range);
    for (size_t i = 0; i < fm_range.size(); i++) {
      OP_LOGD(op.GetName().c_str(), "fmap Range[%u] is (%lld, %lld)", i, fm_range[i].first, fm_range[i].second);
    }

    if (fm_range.empty()) {
      OP_LOGE(op.GetName().c_str(), "fm_range is empty.");
      map<string, string> err_map;
      err_map["op_name"] = op.GetName().c_str();
      err_map["description"] = "fm_range is empty.";
      std::string report_error_code = "E50058";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }

    std::vector<std::pair<int64_t, int64_t>> out_range(fm_range);
    out_range[idx_c] = std::make_pair((int64_t)kn, (int64_t)kn);
    if (x_shape[idx_h] == -1) {
      y_shape[idx_h] = -1;
      int64_t low_h = fm_range[idx_h].first;
      int64_t high_h = fm_range[idx_h].second;
      if (pad_str == "SAME") {
        out_range[idx_h].first = (low_h + strh -1) / strh;
        out_range[idx_h].second = (high_h + strh -1) / strh;
      } else {
        out_range[idx_h].first = (low_h + padt + padb - dilh * (kh - 1) - 1) / strh + 1;
        out_range[idx_h].second = (high_h + padt + padb - dilh * (kh - 1) - 1) / strh + 1;
      }
    }
    if (x_shape[idx_w] == -1) {
      y_shape[idx_w] = -1;
      int64_t low_w = fm_range[idx_w].first;
      int64_t high_w = fm_range[idx_w].second;
      if (pad_str == "SAME") {
        out_range[idx_w].first = (low_w + strw -1) / strw;
        out_range[idx_w].second = (high_w + strw -1) / strw;
      } else {
        out_range[idx_w].first = (low_w + padl + padr - dilw * (kw - 1) - 1) / strw + 1;
        out_range[idx_w].second = (high_w + padl + padr - dilw * (kw - 1) - 1) / strw + 1;
      }
    }
    y_tensor.SetShape(Shape(y_shape));
    y_tensor.SetShapeRange(out_range);
    for (size_t i = 0; i < out_range.size(); i++) {
      OP_LOGD(op.GetName().c_str(), "output Range[%u] is (%lld, %lld)", i, out_range[i].first, out_range[i].second);
    }
  }

  if (GRAPH_SUCCESS != op.UpdateOutputDesc("y", y_tensor)) {
    OP_LOGE(op.GetName().c_str(), "update output desc failed.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "update output desc failed.";
    std::string report_error_code = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  OP_LOGD(op.GetName().c_str(), "Leave Conv2DInfer.");
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

/*!
  * @brief Verify the required 2 input tensor, optional bias ignored, verify
  *        strides and dilations attrs, pads ignored.
  * @param Conv2D Operator type.
  * @param Conv2DVerify Input validity check function.
  * @return Status The processing flow result.
  */
IMPLEMT_VERIFIER(Conv2D, Conv2DVerify) {
  OP_LOGD(op.GetName().c_str(), "Enter Conv2DVerify.");
  auto x_tensor = op.GetInputDesc("x");
  auto w_tensor = op.GetInputDesc("filter");
  auto x_shape = x_tensor.GetOriginShape().GetDims();
  auto w_shape = w_tensor.GetOriginShape().GetDims();
  auto offset_w_tensor = op.GetInputDesc("offset_w");

  if (offset_w_tensor.GetOriginShape().GetDims().size() != 0) {
    OP_LOGE(op.GetName().c_str(), "input offset_w is not supported.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "input offset_w is not supported.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  if (x_shape.size() != 4) {
    if (x_shape.size() == 0) {
      OP_LOGE(op.GetName().c_str(), "input x shape is empty.");
      map<string, string> err_map;
      err_map["op_name"] = op.GetName().c_str();
      err_map["description"] = "input x shape is empty.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    } else {
      OP_LOGE(op.GetName().c_str(), "input x shape should be 4D.");
      map<string, string> err_map;
      err_map["op_name"] = op.GetName().c_str();
      err_map["description"] = "input x shape should be 4D.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    }
    return GRAPH_FAILED;
  }
  if (w_shape.size() != 4) {
    if (w_shape.size() == 0) {
      OP_LOGE(op.GetName().c_str(), "input filter shape is empty.");
      map<string, string> err_map;
      err_map["op_name"] = op.GetName().c_str();
      err_map["description"] = "input filter shape is empty.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    } else {
      OP_LOGE(op.GetName().c_str(), "input filter shape should be 4D.");
      map<string, string> err_map;
      err_map["op_name"] = op.GetName().c_str();
      err_map["description"] = "input filter shape should be 4D.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    }
    return GRAPH_FAILED;
  }

  auto x_dtype = x_tensor.GetDataType();
  auto w_dtype = w_tensor.GetDataType();

  if (x_dtype != w_dtype) {
    OP_LOGE(op.GetName().c_str(),
            "input x dtype is differ from filter dtype."
            " actual x dtype is: %d filter dtype is: %d",
            (int)x_dtype, (int)w_dtype);
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["param1"] = "x";
    err_map["param1_data_type"] = std::to_string(x_dtype);
    err_map["param2"] = "filter";
    err_map["param2_data_type"] = std::to_string(w_dtype);
    err_map["rule"] = "input x dtype is same as filter dtype";
    std::string report_error_code = "E50004";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int32_t> stride_list;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stride_list)) {
    OP_LOGE(op.GetName().c_str(), "get strides list failed.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "get strides list failed.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  std::vector<int32_t> dilation_list;
  if (GRAPH_SUCCESS != op.GetAttr("dilations", dilation_list)) {
    OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "get dilations list failed.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  OP_LOGD(op.GetName().c_str(), "Leave Conv2DVerify.");
  return GRAPH_SUCCESS;
}

static void InferHWConv2D(int32_t input, int32_t kernel, int32_t pad, int32_t stride,
                          int32_t dilation, vector<int64_t> output_slice, vector<int64_t>& data_slice) {
  // calc start rule: (i_start + pad_h)/stride_h = output_start
  int64_t i_start = output_slice[0] * stride - pad;
  if (i_start < 0) {
    i_start = 0;
  }
  // calc end rule: (iend_start + pad_h)/stride_h = output_end
  // iend_end = iend_start + dilation*(kernel_h-1)
  int64_t i_end = output_slice[1] * stride - pad + dilation * (kernel - 1);
  if (i_end >= input) {
    i_end = input - 1;
  }
  data_slice = {i_start, i_end};
}
/*!
  * @brief provide Conv2D operator slice data
  * @param Conv2D Operator type.
  * @param Conv2DInferDataSlice slice data function
  * @return Status The processing flow result.
  */
IMPLEMT_INFER_DATA_SLICE(Conv2D, Conv2DInferDataSlice) {
  OP_LOGD(op.GetName().c_str(), "Enter Conv2D InferDataSlice");
  // get input h/w, filter h/w, pad_h,pad_w, stride_h, stride_w, dilation_h,dilation_w
  auto x_tensor = op.GetInputDesc("x");
  auto w_tensor = op.GetInputDesc("filter");

  auto x_shape = x_tensor.GetShape().GetDims();
  auto w_shape = w_tensor.GetShape().GetDims();

  auto x_format = x_tensor.GetFormat();
  auto w_format = w_tensor.GetFormat();

  std::vector<int32_t> stride_list;
  std::vector<int32_t> dilation_list;
  std::vector<int32_t> pad_list;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stride_list) || GRAPH_SUCCESS != op.GetAttr("dilations", dilation_list)
      || GRAPH_SUCCESS != op.GetAttr("pads", pad_list)){
    return GRAPH_FAILED;
  }

  int32_t ih = 0;
  int32_t iw = 0;
  int32_t kh = 0;
  int32_t kw = 0;
  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;
  int32_t padt = pad_list[0];
  int32_t padb = pad_list[1];
  int32_t padl = pad_list[2];
  int32_t padr = pad_list[3];

  if (x_format == FORMAT_NCHW) {
    ih = x_shape[2];
    iw = x_shape[3];
    strh = stride_list[2];
    strw = stride_list[3];
    dilh = dilation_list[2];
    dilw = dilation_list[3];
  } else if (x_format == FORMAT_NHWC) {
    ih = x_shape[1];
    iw = x_shape[2];
    strh = stride_list[1];
    strw = stride_list[2];
    dilh = dilation_list[1];
    dilw = dilation_list[2];
  }

  if (w_format == FORMAT_NCHW) {
    kh = w_shape[2];
    kw = w_shape[3];
  } else if (w_format == FORMAT_NHWC) {
    kh = w_shape[1];
    kw = w_shape[2];
  } else if (w_format == FORMAT_HWCN) {
    kh = w_shape[0];
    kw = w_shape[1];
  }

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
  vector<vector<int64_t>> y_data_slice;
  vector<vector<int64_t>> x_data_slice = {{}, {}, {}, {}, {}};
  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op.GetName().c_str(), "no data slice, not need infer input");
    return GRAPH_FAILED;
  }
  bool need_infer = false;
  bool have_slice = false;
  for(int i=0; i < y_data_slice.size(); i++) {
    if (y_data_slice[i].size() > 0) {
      have_slice = true;
      if (i == 2) {
        need_infer = true;
        vector<int64_t> ih_slice;
        InferHWConv2D(ih, kh, padt, strh, dilh, y_data_slice[i], ih_slice);
        OP_LOGD(op.GetName().c_str(), "conv2d h axis slice ori_scope is [%d,%d], calced output scope is [%d,%d]",
                ih_slice[0], ih_slice[1], y_data_slice[i][0], y_data_slice[i][1]);
        x_data_slice[i] = ih_slice;
      } else if (i == 3) {
        need_infer = true;
        vector<int64_t> iw_slice;
        InferHWConv2D(iw, kw, padl, strw, dilw, y_data_slice[i], iw_slice);
        OP_LOGD(op.GetName().c_str(), "conv2d w axis slice ori_scope is [%d,%d], calced output scope is [%d,%d]",
                iw_slice[0], iw_slice[1], y_data_slice[i][0], y_data_slice[i][1]);
        x_data_slice[i] = iw_slice;
      }
    }
  }
  if (have_slice == false) {
    return GRAPH_FAILED;
  }
  if (need_infer == false) {
    return NO_OVERLAP_DIM;
  } else{
    if(!AttrUtils::SetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }
  OP_LOGD(op.GetName().c_str(), "Calc Conv2D InferDataSlice end!");
}

INFER_DATA_SLICE_FUNC_REG(Conv2D, Conv2DInferDataSlice);
INFER_FUNC_REG(Conv2D, Conv2DInfer);
VERIFY_FUNC_REG(Conv2D, Conv2DVerify);

// --------------------------Conv2DCompress------------------------------

/*
 * Infer output shape and dtype, dtype is same to first input tensor
 * Output format is set by ge parser process already
 */
IMPLEMT_INFERFUNC(Conv2DCompress, Conv2DCompressInfer) {
  OP_LOGD(op.GetName().c_str(), "Enter Conv2DCompressInfer.");
  auto xTensor = op.get_input_desc_x();
  auto wTensor = op.get_input_desc_filter_compress();

  auto xShape = xTensor.GetShape().GetDims();
  auto wShape = wTensor.GetShape().GetDims();
  auto xFormat = xTensor.GetFormat();
  auto wFormat = wTensor.GetFormat();
  CHECK_FORMAT(xFormat);
  CHECK_FORMAT(wFormat);

  int32_t in = 0;
  int32_t ic = 0;
  int32_t ih = 0;
  int32_t iw = 0;
  int32_t kn = 0;
  int32_t kc = 0;
  int32_t kh = 0;
  int32_t kw = 0;
  if (xFormat == FORMAT_NCHW) {
    in = xShape[0];
    ic = xShape[1];
    ih = xShape[2];
    iw = xShape[3];
  } else if (xFormat == FORMAT_NHWC) {
    in = xShape[0];
    ic = xShape[3];
    ih = xShape[1];
    iw = xShape[2];
  } else {
    OP_LOGE(op.GetName().c_str(), "input x format should be NCHW or NHWC.");
    return GRAPH_FAILED;
  }

  if (wFormat == FORMAT_NCHW) {
    kn = wShape[0];
    kc = wShape[1];
    kh = wShape[2];
    kw = wShape[3];
  } else if (wFormat == FORMAT_NHWC) {
    kn = wShape[0];
    kc = wShape[3];
    kh = wShape[1];
    kw = wShape[2];
  } else if (wFormat == FORMAT_HWCN) {
    kn = wShape[3];
    kc = wShape[2];
    kh = wShape[0];
    kw = wShape[1];
  } else {
    OP_LOGE(op.GetName().c_str(), "input filter format should be NCHW, NHWC or HWCN.");
    return GRAPH_FAILED;
  }

  // set data_format: copy value of xFormat to data_format
  // data_format will be used to infer position of H/W
  // in strides and dilations(previously used ori_format)
  std::string data_format;
  std::string attr_data_format = "data_format";
  std::string data_format_NCHW = "NCHW";
  std::string data_format_NHWC = "NHWC";

  if (GRAPH_SUCCESS == op.GetAttr(attr_data_format, data_format)) {
    OP_LOGI(data_format.c_str(), "conv compress before set data_format");
  }

  if (xFormat == ge::FORMAT_NCHW) {
    op.SetAttr(attr_data_format, data_format_NCHW);
  } else {
    op.SetAttr(attr_data_format, data_format_NHWC);
  }

  op.GetAttr(attr_data_format, data_format);
  OP_LOGI(data_format.c_str(), "conv compress after set data_format");

  int64_t groups = 1;
  if (GRAPH_SUCCESS != op.GetAttr("groups", groups)) {
    OP_LOGI(op.GetName().c_str(), "no groups setting, use groups as 1");
  }
  OP_LOGI(op.GetName().c_str(), "groups is %lld", groups);

  if (ic != kc * groups) {
    OP_LOGE(op.GetName().c_str(), "input x channel should be equal to filter. ");
    return GRAPH_FAILED;
  }

  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;
  int32_t padt = 0;
  int32_t padb = 0;
  int32_t padl = 0;
  int32_t padr = 0;
  if (false == GetAttrsConv2D(op, xFormat, strh, strw, dilh, dilw)) {
    OP_LOGE(op.GetName().c_str(), "get attrs failed.");
    return GRAPH_FAILED;
  }
  if (false == GetPadConv2D(op, ih, iw, kh, kw, strh, strw, dilh, dilw, padt, padb, padl, padr)) {
    OP_LOGE(op.GetName().c_str(), "get pads attrs failed.");
    return GRAPH_FAILED;
  }

  int64_t oh = (ih + padt + padb - dilh * (kh - 1) - 1) / strh + 1;
  int64_t ow = (iw + padl + padr - dilw * (kw - 1) - 1) / strw + 1;

  vector<int64_t> yShape;
  auto yTensor = op.get_output_desc_y();
  auto yFormat = yTensor.GetFormat();
  CHECK_FORMAT(yFormat)
  if (yFormat == FORMAT_NCHW) {
    yShape.push_back(in);
    yShape.push_back(kn);
    yShape.push_back(oh);
    yShape.push_back(ow);
  } else if (yFormat == FORMAT_NHWC) {
    yShape.push_back(in);
    yShape.push_back(oh);
    yShape.push_back(ow);
    yShape.push_back(kn);
  } else {
    OP_LOGE(op.GetName().c_str(), "output y format should be NCHW or NHWC.");
    return GRAPH_FAILED;
  }
  yTensor.SetShape(Shape(yShape));
  auto xDtype = xTensor.GetDataType();
  if (xDtype == ge::DT_INT8) {
    yTensor.SetDataType(ge::DT_INT32);
  } else {
    yTensor.SetDataType(xDtype);
  }
  if (GRAPH_SUCCESS != op.update_output_desc_y(yTensor)) {
    OP_LOGE(op.GetName().c_str(), "update output desc failed.");
    return GRAPH_FAILED;
  }

  OP_LOGD(op.GetName().c_str(), "Leave Conv2DCompressInfer.");
  return GRAPH_SUCCESS;
}

/*
 * Verify the required 2 input tensor, optional bias ignored
 * Verify strides and dilations attrs, pads ignored
 */
IMPLEMT_VERIFIER(Conv2DCompress, Conv2DCompressVerify) {
  OP_LOGD(op.GetName().c_str(), "Enter Conv2DCompressVerify.");
  auto xTensor = op.get_input_desc_x();
  auto wTensor = op.get_input_desc_filter_compress();

  auto xShape = xTensor.GetShape().GetDims();
  auto wShape = wTensor.GetShape().GetDims();

  if (xShape.size() != 4) {
    if (xShape.size() == 0) {
      OP_LOGE(op.GetName().c_str(), "input x shape is empty.");
    } else {
      OP_LOGE(op.GetName().c_str(), "only support 2D compress convolution.");
    }
    return GRAPH_FAILED;
  }
  if (wShape.size() != 4) {
    if (wShape.size() == 0) {
      OP_LOGE(op.GetName().c_str(), "input filter_compress shape is empty.");
    } else {
      OP_LOGE(op.GetName().c_str(), "only support 2D compress convolution.");
    }
    return GRAPH_FAILED;
  }

  auto xDtype = xTensor.GetDataType();
  auto wDtype = wTensor.GetDataType();

  if (xDtype != wDtype) {
    OP_LOGE(op.GetName().c_str(),
            "input x dtype is differ from filter_compress dtype."
            " actual x dtype is: %d filter dtype is: %d",
            (int)xDtype, (int)wDtype);
    return GRAPH_FAILED;
  }

  std::vector<int32_t> strideList;
  if (GRAPH_SUCCESS != op.GetAttr("strides", strideList)) {
    OP_LOGE(op.GetName().c_str(), "get strides list failed.");
    return GRAPH_FAILED;
  }
  std::vector<int32_t> dilationList;
  if (GRAPH_SUCCESS != op.GetAttr("dilations", dilationList)) {
    OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
    return GRAPH_FAILED;
  }

  OP_LOGD(op.GetName().c_str(), "Leave Conv2DCompressVerify.");
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Conv2DCompress, Conv2DCompressInfer);
VERIFY_FUNC_REG(Conv2DCompress, Conv2DCompressVerify);

// ----------------DeformableConv2d-----------------------
/*!
  * @brief Get 2D(H/W) stride, dilation and pad params to infershape output.
  *
  * [strides]: 4D list, format sensitive, according to first input tensor format
  * [dilations]: 4D list, format sensitive
  * [pads]: 4D list
  *
  * @param op DeformableConv2D operator.
  * @param refer Valid value reference format.
  * @param strh, strw  Input stride H/W value.
  * @param dilh, dilw  Input dilation H/W value.
  * @param padt, padb, padl, padr  Input top/bottom/left/right pad value.
  * @return bool Whether the strides, dilations settings are correct.
  */
static bool GetAttrsDfmConv2D(ge::Operator& op, Format refer, int32_t& strh, int32_t& strw, int32_t& dilh, int32_t& dilw,
                              int32_t& padt, int32_t& padb, int32_t& padl, int32_t& padr) {
  std::vector<int32_t> stride_list;
  op.GetAttr("strides", stride_list);
  auto s_size = stride_list.size();
  if (stride_list.empty() || s_size != 4) {
    OP_LOGE(op.GetName().c_str(), "strides list should be 4D. actual is: %u.", s_size);
    map<string, string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_value"] = "4D";
    err_map["input_value"] = std::to_string(s_size) + "D.";
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  std::vector<int32_t> dilation_list;
  op.GetAttr("dilations", dilation_list);
  auto d_size = dilation_list.size();
  if (dilation_list.empty() || d_size != 4) {
    OP_LOGE(op.GetName().c_str(), "dilations list should be 4D. actual is: %u.", d_size);
    map<string, string> err_map;
    err_map["param_name"] = "dilations";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_value"] = "4D";
    err_map["input_value"] = std::to_string(d_size) + "D.";
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  int32_t strn = 0;
  int32_t strc = 0;
  int32_t diln = 0;
  int32_t dilc = 0;
  if (refer == FORMAT_NCHW) {
    strn = stride_list[0];
    strc = stride_list[1];
    strh = stride_list[2];
    strw = stride_list[3];
    diln = dilation_list[0];
    dilc = dilation_list[1];
    dilh = dilation_list[2];
    dilw = dilation_list[3];
  } else if (refer == FORMAT_NHWC) {
    strn = stride_list[0];
    strc = stride_list[3];
    strh = stride_list[1];
    strw = stride_list[2];
    diln = dilation_list[0];
    dilc = dilation_list[3];
    dilh = dilation_list[1];
    dilw = dilation_list[2];
  }
  if (strh <= 0 || strw <= 0) {
    OP_LOGE(op.GetName().c_str(),
            "strides should be positive,"
            " actual is [%d,%d].",
            strh, strw);
    map<string, string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_value"] = "positive";
    err_map["input_value"] = std::to_string(strh) + ", " + std::to_string(strw);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  if (strn != 1 || strc != 1) {
    OP_LOGE(op.GetName().c_str(), "strides N/C dimensions must be set to 1.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "strides N/C dimensions must be set to 1.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  if (dilh <= 0 || dilw <= 0) {
    OP_LOGE(op.GetName().c_str(),
            "dilations should be positive,"
            " actual is [%d,%d].",
            dilh, dilw);
    map<string, string> err_map;
    err_map["param_name"] = "dilations";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_value"] = "positive";
    err_map["input_value"] = std::to_string(dilh) + ", " + std::to_string(dilw);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  if (diln != 1 || dilc != 1) {
    OP_LOGE(op.GetName().c_str(), "dilations N/C dimensions must be set to 1.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "dilations N/C dimensions must be set to 1.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  std::vector<int32_t> pads_list;
  op.GetAttr("pads", pads_list);
  auto p_size = pads_list.size();
  if (pads_list.empty() || p_size != 4) {
    OP_LOGE(op.GetName().c_str(), "pads list should be 4D. actual is: %u.", p_size);
    map<string, string> err_map;
    err_map["param_name"] = "pads";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_value"] = "4D";
    err_map["input_value"] = std::to_string(p_size) + "D.";
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  padt = pads_list[0];
  padb = pads_list[1];
  padl = pads_list[2];
  padr = pads_list[3];
  if (padt < 0 || padb < 0 || padl < 0 || padr < 0) {
    OP_LOGE(op.GetName().c_str(),
            "pads should be positive, "
            " actual is [%d,%d,%d,%d].",
            padt, padb, padl, padr);
    map<string, string> err_map;
    err_map["param_name"] = "pads";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_value"] = "positive";
    err_map["input_value"] =
        std::to_string(padt) + ", " + std::to_string(padb) + ", " + std::to_string(padl) + ", " + std::to_string(padr);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  return true;
}

/*!
  * @brief Infer output shape and dtype, dtype is same to first input tensor
  *        Output format is set by ge parser process already
  * @param DeformableConv2D Operator type.
  * @param DeformableConv2DInfer Infer function.
  * @return Status The processing flow result.
  */
IMPLEMT_INFERFUNC(DeformableConv2D, DeformableConv2DInfer){
  OP_LOGD(op.GetName().c_str(), "Enter DeformableConv2DInfer");
  auto x_tensor = op.GetInputDesc("x");
  auto w_tensor = op.GetInputDesc("filter");

  auto x_shape = x_tensor.GetShape().GetDims();
  auto w_shape = w_tensor.GetShape().GetDims();
  if (x_shape.size() != 4 || w_shape.size() != 4) {
    return GRAPH_FAILED;
  }
  auto x_format = x_tensor.GetFormat();
  auto w_format  = w_tensor.GetFormat();
  CHECK_FORMAT(x_format);
  CHECK_FORMAT(w_format);

  int32_t in = 0;
  int32_t ic = 0;
  int32_t ih = 0;
  int32_t iw = 0;
  int32_t kn = 0;
  int32_t kc = 0;
  int32_t kh = 0;
  int32_t kw = 0;

  // set fm
  if (x_format == FORMAT_NCHW) {
    in = x_shape[0];
    ic = x_shape[1];
    ih = x_shape[2];
    iw = x_shape[3];
  } else if (x_format == FORMAT_NHWC) {
    in = x_shape[0];
    ic = x_shape[3];
    ih = x_shape[1];
    iw = x_shape[2];
  } else {
    OP_LOGE(op.GetName().c_str(), "input x format should be NCHW or NHWC. actual is: %s",
            TypeUtils::FormatToSerialString(x_format).c_str());
    map<string, string> err_map;
    err_map["param"] = "x";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_format_list"] = "NCHW or NHWC";
    err_map["format"] = TypeUtils::FormatToSerialString(x_format);
    std::string report_error_code = "E50002";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // set kernel
  if (w_format == FORMAT_NCHW) {
    kn = w_shape[0];
    kc = w_shape[1];
    kh = w_shape[2];
    kw = w_shape[3];
  } else if (w_format == FORMAT_HWCN) {
    kn = w_shape[3];
    kc = w_shape[2];
    kh = w_shape[0];
    kw = w_shape[1];
  } else {
    OP_LOGE(op.GetName().c_str(),
            "input filter format should be NCHW or HWCN. actual is: %s",
            TypeUtils::FormatToSerialString(w_format).c_str());
    map<string, string> err_map;
    err_map["param"] = "filter";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_format_list"] = "NCHW, NHWC or HWCN";
    err_map["format"] = TypeUtils::FormatToSerialString(w_format);
    std::string report_error_code = "E50002";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  auto bias_tensor = op.GetInputDesc("bias");
  auto bias_shape = bias_tensor.GetShape().GetDims();
  if (bias_shape.size() == 1 && bias_shape[0] != kn) {
    OP_LOGE(op.GetName().c_str(), "input bias size should be equal to out_channels.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "input bias size should be equal to out_channels.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  } else if (bias_shape.size() > 1) {
    OP_LOGE(op.GetName().c_str(), "input bias shape should be 1D.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "input bias shape should be 1D.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // set data_format: copy value of x_format to data_format
  // data_format will be used to infer position of H/W
  // in strides and dilations(previously used ori_format)
  std::string data_format;
  std::string attr_data_format = "data_format";
  std::string data_format_NCHW = "NCHW";
  std::string data_format_NHWC = "NHWC";

  if (x_format == ge::FORMAT_NCHW) {
    op.SetAttr(attr_data_format, data_format_NCHW);
  } else {
    op.SetAttr(attr_data_format, data_format_NHWC);
  }

  // set strides, dilations, pad
  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;
  int32_t padt = 0;
  int32_t padb = 0;
  int32_t padl = 0;
  int32_t padr = 0;
  if (!GetAttrsDfmConv2D(op, x_format, strh, strw, dilh, dilw, padt, padb, padl, padr)) {
    return GRAPH_FAILED;
  }

  int64_t ih_pad = (ih + padt + padb - dilh * (kh - 1) - 1);
  int64_t iw_pad = (iw + padl + padr - dilw * (kw - 1) - 1);
  int64_t oh = ih_pad / strh + 1;
  int64_t ow = iw_pad / strw + 1;
  if ((ih > 0) && (kh > 0) && (iw > 0) && (kw > 0)) {
    if ((ih_pad < 0) || (iw_pad < 0)) {
      OP_LOGE(op.GetName().c_str(), "image size after padding should be greater than or equal to filter size.");
      map<string, string> err_map;
      err_map["op_name"] = op.GetName().c_str();
      err_map["description"] = "image size after padding should be greater than or equal to filter size.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    }
  }

  int64_t groups = 1;
  op.GetAttr("groups", groups);
  if (ic != kc * groups) {
    OP_LOGE(op.GetName().c_str(),
            "x channel should be equal to filter channel*groups. "
            "x format is: %s, filter format is: %s, "
            "x shape is: [%d,%d,%d,%d], filter shape is: [%d,%d,%d,%d], "
            "groups is: %d.",
            TypeUtils::FormatToSerialString(x_format).c_str(), TypeUtils::FormatToSerialString(w_format).c_str(),
            (int)x_shape[0], (int)x_shape[1], (int)x_shape[2], (int)x_shape[3], (int)w_shape[0], (int)w_shape[1],
            (int)w_shape[2], (int)w_shape[3], (int)groups);
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["x_shape"] = std::to_string(x_shape[0]) + ", " + std::to_string(x_shape[1]) + ", " +
                         std::to_string(x_shape[2]) + ", " + std::to_string(x_shape[3]);
    err_map["filter_shape"] = std::to_string(w_shape[0]) + ", " + std::to_string(w_shape[1]) + ", " +
                              std::to_string(w_shape[2]) + ", " + std::to_string(w_shape[3]);
    err_map["groups"] = std::to_string(groups);

    std::string report_error_code = "E50059";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  if (kn % groups != 0) {
    OP_LOGE(op.GetName().c_str(), "out_channels should be divisible by groups.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "out_channels should be divisible by groups.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  auto offset_tensor = op.GetInputDesc("offsets");
  auto offset_format  = offset_tensor.GetFormat();
  auto offset_shape = offset_tensor.GetShape().GetDims();
  if (offset_shape.size() != 4) {
    return GRAPH_FAILED;
  }
  int32_t dfm_group = 1;
  op.GetAttr("deformable_groups", dfm_group);
  if (dfm_group < 1 || ic % dfm_group != 0) {
    OP_LOGE(op.GetName().c_str(), "deformable_groups should be positive and can divide in_channels");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "deformable_groups should be positive and can divide in_channels.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  std::vector<int64_t> exp_shape;
  if (offset_format == FORMAT_NCHW) {
    exp_shape.push_back(in);
    exp_shape.push_back(dfm_group * kh * kw * 3);
    exp_shape.push_back(ih);
    exp_shape.push_back(iw);
  } else if (offset_format == FORMAT_NHWC) {
    exp_shape.push_back(in);
    exp_shape.push_back(ih);
    exp_shape.push_back(iw);
    exp_shape.push_back(dfm_group * kh * kw * 3);
  } else {
    OP_LOGE(op.GetName().c_str(), "input offsets format should be NCHW or NHWC. actual is: %s",
            TypeUtils::FormatToSerialString(offset_format).c_str());
    map<string, string> err_map;
    err_map["param"] = "offsets";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_format_list"] = "NCHW or NHWC";
    err_map["format"] = TypeUtils::FormatToSerialString(offset_format);
    std::string report_error_code = "E50002";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  if (exp_shape != offset_shape) {
    OP_LOGE(op.GetName().c_str(), "input offsets shape should be [%d,%d,%d,%d].",
      (int)exp_shape[0], (int)exp_shape[1], (int)exp_shape[2], (int)exp_shape[3]);
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "input offsets shape should be [" + std::to_string(exp_shape[0]) + ", " +
                             std::to_string(exp_shape[1]) + ", " + std::to_string(exp_shape[2]) + ", " +
                             std::to_string(exp_shape[3]) + "].";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  vector<int64_t> y_shape;
  auto y_tensor = op.GetOutputDesc("y");
  auto y_format = y_tensor.GetFormat();
  CHECK_FORMAT(y_format)
  if (y_format == FORMAT_NCHW) {
    y_shape.push_back(in);
    y_shape.push_back(kn);
    y_shape.push_back(oh);
    y_shape.push_back(ow);
  } else if (y_format == FORMAT_NHWC) {
    y_shape.push_back(in);
    y_shape.push_back(oh);
    y_shape.push_back(ow);
    y_shape.push_back(kn);
  } else {
    OP_LOGE(op.GetName().c_str(), "output y format should be NCHW or NHWC. actual is: %s",
            TypeUtils::FormatToSerialString(y_format).c_str());
    map<string, string> err_map;
    err_map["param"] = "y";
    err_map["op_name"] = op.GetName().c_str();
    err_map["expected_format_list"] = "NCHW or NHWC";
    err_map["format"] = TypeUtils::FormatToSerialString(y_format);
    std::string report_error_code = "E50002";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  y_tensor.SetShape(Shape(y_shape));
  auto x_dtype = x_tensor.GetDataType();
  if (x_dtype == ge::DT_INT8) {
    y_tensor.SetDataType(ge::DT_INT32);
  }else{
    y_tensor.SetDataType(x_dtype);
  }

  if (GRAPH_SUCCESS != op.UpdateOutputDesc("y", y_tensor)) {
    OP_LOGE(op.GetName().c_str(), "update output desc failed.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "update output desc failed.";
    std::string report_error_code = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  OP_LOGD(op.GetName().c_str(), "Leave DeformableConv2DInfer.");
  return GRAPH_SUCCESS;
}

/*!
  * @brief Verify the required 3 input tensor, optional bias ignored, verify
  *        strides and dilations attrs, pads ignored.
  * @param DeformableConv2D Operator type.
  * @param DeformableConv2DVerify Input validity check function.
  * @return Status The processing flow result.
  */
IMPLEMT_VERIFIER(DeformableConv2D, DeformableConv2DVerify){
  OP_LOGD(op.GetName().c_str(), "Enter DeformableConv2DVerify.");
  auto x_tensor = op.GetInputDesc("x");
  auto w_tensor = op.GetInputDesc("filter");
  auto offset_tensor = op.GetInputDesc("offsets");
  auto x_shape = x_tensor.GetOriginShape().GetDims();
  auto w_shape = w_tensor.GetOriginShape().GetDims();
  auto offset_shape = offset_tensor.GetOriginShape().GetDims();

  if (x_shape.size() != 4) {
    if (x_shape.size() == 0) {
      OP_LOGE(op.GetName().c_str(), "input x shape is empty.");
      map<string, string> err_map;
      err_map["op_name"] = op.GetName().c_str();
      err_map["description"] = "input x shape is empty.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    } else {
      OP_LOGE(op.GetName().c_str(), "input x shape should be 4D.");
      map<string, string> err_map;
      err_map["op_name"] = op.GetName().c_str();
      err_map["description"] = "input x shape should be 4D.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    }
    return GRAPH_FAILED;
  }
  if (w_shape.size() != 4) {
    if (w_shape.size() == 0) {
      OP_LOGE(op.GetName().c_str(), "input filter shape is empty.");
      map<string, string> err_map;
      err_map["op_name"] = op.GetName().c_str();
      err_map["description"] = "input filter shape is empty.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    } else {
      OP_LOGE(op.GetName().c_str(), "input filter shape should be 4D.");
      map<string, string> err_map;
      err_map["op_name"] = op.GetName().c_str();
      err_map["description"] = "input filter shape should be 4D.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    }
    return GRAPH_FAILED;
  }
  if (offset_shape.size() != 4) {
    if (offset_shape.size() == 0) {
      OP_LOGE(op.GetName().c_str(), "input offsets shape is empty.");
      map<string, string> err_map;
      err_map["op_name"] = op.GetName().c_str();
      err_map["description"] = "input offsets shape is empty.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    } else {
      OP_LOGE(op.GetName().c_str(), "input offsets shape should be 4D.");
      map<string, string> err_map;
      err_map["op_name"] = op.GetName().c_str();
      err_map["description"] = "input offsets shape should be 4D.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    }
    return GRAPH_FAILED;
  }

  auto x_dtype = x_tensor.GetDataType();
  auto w_dtype = w_tensor.GetDataType();

  if (x_dtype != w_dtype) {
    OP_LOGE(op.GetName().c_str(),
            "input x dtype is differ from filter dtype. actual x dtype is: %d filter dtype is: %d",
            (int)x_dtype, (int)w_dtype);
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["param1"] = "x";
    err_map["param1_data_type"] = std::to_string(x_dtype);
    err_map["param2"] = "filter";
    err_map["param2_data_type"] = std::to_string(w_dtype);
    err_map["rule"] = "input x dtype is same as filter dtype";
    std::string report_error_code = "E50004";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int32_t> stride_list;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stride_list)) {
    OP_LOGE(op.GetName().c_str(), "get strides list failed.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "get strides list failed.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  std::vector<int32_t> dilation_list;
  if (GRAPH_SUCCESS != op.GetAttr("dilations", dilation_list)) {
    OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "get dilations list failed.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  OP_LOGD(op.GetName().c_str(), "Leave DeformableConv2DVerify.");
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DeformableConv2D, DeformableConv2DInfer);
VERIFY_FUNC_REG(DeformableConv2D, DeformableConv2DVerify);
// ----------------Deconvolution--------------------------
static bool GetAttrsDeconv(ge::Operator& op, Format refer, int32_t& strh, int32_t& strw, int32_t& dilh, int32_t& dilw) {
  std::vector<int32_t> stride_list;
  op.GetAttr("strides", stride_list);
  auto s_size = stride_list.size();
  if (s_size != 2) {
    OP_LOGE(op.GetName().c_str(),
            "strides list should be 2d."
            " actual is: %d.",
            s_size);
    string sizealue = ConcatString(s_size);
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["param_name"] = "sSize";
    err_map["expected_value"] = "2d";
    err_map["input_value"] = sizealue;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  std::vector<int32_t> dilation_list;
  op.GetAttr("dilations", dilation_list);
  auto d_size = dilation_list.size();
  if (d_size != 4) {
    OP_LOGE(op.GetName().c_str(),
            "dilations list should be 4d."
            " actual is: %d.",
            d_size);
    string realvalue = ConcatString(d_size);
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["param_name"] = "dSize";
    err_map["expected_value"] = "4d";
    err_map["input_value"] = realvalue;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  strh = stride_list[0];
  strw = stride_list[1];
  if (refer == FORMAT_NCHW) {
    dilh = dilation_list[2];
    dilw = dilation_list[3];
  } else {
    return false;
  }
  if (strh <= 0 || strw <= 0) {
    OP_LOGE(op.GetName().c_str(),
            "strides should be positive, "
            " actual is [%d,%d].",
            strh, strw);
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["param_name"] = "strides";
    err_map["expected_value"] = "positive";
    err_map["input_value"] = "[" + std::to_string(strh) + "," + std::to_string(strw) + "]";
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  return true;
}

static void InferHWDeoconv(int32_t kernel, int32_t dilation, int32_t stride, vector<int32_t>& pads,
                           vector<int64_t>& output, vector<int64_t>& input, int32_t index, int32_t input_max) {
  int32_t dilate_kernel = (kernel - 1) * dilation + 1;
  int32_t pad_out = kernel - pads[index] - 1;
  int32_t start = std::ceil(static_cast<float>(output[0] - pad_out) / stride);
  int32_t end = (output[1] + dilate_kernel - 1 - pad_out) / stride;
  start = std::max(static_cast<int32_t>(0), start);
  end = std::min(end, input_max - 1);
  input = {start, end};
  int32_t oh = output[1] - output[0] + 1;
  int32_t ih = end - start + 1;
  if (output[0] != 0) {
    int32_t pad_pre_deconv = (stride - (output[0] - pad_out) % stride) % stride;
    pads[index] = kernel - pad_pre_deconv - 1;
  }
  pads[index+1] = stride * (ih - 1) + dilate_kernel - oh - pads[index];
}

IMPLEMT_INFER_DATA_SLICE(Deconvolution, DeconvolutionInferDataSlice) {
  OP_LOGD(op.GetName().c_str(), "Enter Deconvolution InferDataSlice.");

  auto x_tensor = op.get_input_desc_x();
  auto w_tensor = op.get_input_desc_filter();
  auto x_format = x_tensor.GetFormat();
  auto x_shape = x_tensor.GetShape().GetDims();
  auto w_shape = w_tensor.GetShape().GetDims();
  auto x_dtype = x_tensor.GetDataType();
  int32_t ih = x_shape[2];
  int32_t iw = x_shape[3];
  int32_t kh = w_shape[2];
  int32_t kw = w_shape[3];
  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;

  if (false == GetAttrsDeconv(op, x_format, strh, strw, dilh, dilw)) {
    OP_LOGE(op.GetName().c_str(), "get attrs failed.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "get attrs failed.";
    std::string report_error_code = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  vector<int32_t> pad_list;
  op.GetAttr("pads", pad_list);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
  GeTensorDescPtr tensor_desc_w = op_desc->MutableInputDesc("filter");

  vector<vector<int64_t>> y_data_slice;
  vector<vector<int64_t>> x_data_slice = {{}, {}, {}, {}, {}};
  vector<vector<int64_t>> w_data_slice = {{}, {}, {}, {}};

  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op.GetName().c_str(), "no data slice, not need infer input");
    return GRAPH_FAILED;
  }

  bool need_infer = false;
  bool have_slice = false;

  for(int i = 0; i < y_data_slice.size(); i++) {
    if (y_data_slice[i].size() > 0) {
      have_slice = true;
      if (i == 1 && x_dtype != DT_INT8) {
        int64_t cin_start = y_data_slice[i][0] * kh * kw;
        int64_t cin_end = (y_data_slice[i][1] + 1)*kh*kw - 1;
        w_data_slice[0] = {cin_start, cin_end};
        if(!AttrUtils::SetListListInt(tensor_desc_w, ge::ATTR_NAME_DATA_SLICE, w_data_slice)) {
          return GRAPH_FAILED;
        }
        OP_LOGI(op.GetName().c_str(), "infer input in Cin success");
        return GRAPH_SUCCESS;
      } else if(i == 2 && (kh != 1 || strh != 1)) {
        vector<int64_t> input_h;
        InferHWDeoconv(kh, dilh, strh, pad_list, y_data_slice[i], input_h, 0, ih);
        x_data_slice[i] = input_h;
        if(!AttrUtils::SetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
          return GRAPH_FAILED;
        }
        op.SetAttr("pads", pad_list);
        OP_LOGI(op.GetName().c_str(), "infer input in H success");
        return GRAPH_SUCCESS;
      } else if(i == 3 && (kw != 1 || strw != 1)) {
        vector<int64_t> input_w;
        InferHWDeoconv(kw, dilw, strw, pad_list, y_data_slice[i], input_w, 2, iw);
        x_data_slice[i] = input_w;
        if(!AttrUtils::SetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
          return GRAPH_FAILED;
        }
        op.SetAttr("pads", pad_list);
        OP_LOGI(op.GetName().c_str(), "infer input in W success");
        return GRAPH_SUCCESS;
      } else if (i == 4) {
        OP_LOGI(op.GetName().c_str(), "cannot support cut in block_C");
        return NOT_SUPPORT_SLICE;
      }
    }
  }

  if (have_slice == false) {
    OP_LOGI(op.GetName().c_str(), "no data slice, not need infer input");
    return GRAPH_FAILED;
  } else {
    OP_LOGI(op.GetName().c_str(), "data slice without overlap, not need infer input");
    return NO_OVERLAP_DIM;
  }
  return GRAPH_FAILED;
}

IMPLEMT_INFERFUNC(Deconvolution, DeconvolutionInfer) {
  OP_LOGD(op.GetName().c_str(), "Enter DeconvolutionInfer.");
  auto x_tensor = op.get_input_desc_x();
  auto w_tensor = op.get_input_desc_filter();

  auto x_shape = x_tensor.GetShape().GetDims();
  auto w_shape = w_tensor.GetShape().GetDims();
  auto x_format = x_tensor.GetFormat();
  auto w_format = w_tensor.GetFormat();
  CHECK_FORMAT(x_format);
  CHECK_FORMAT(w_format);

  int32_t in = 0;
  int32_t ic = 0;
  int32_t ih = 0;
  int32_t iw = 0;
  int32_t kn = 0;
  int32_t kc = 0;
  int32_t kh = 0;
  int32_t kw = 0;
  if (x_format == FORMAT_NCHW) {
    in = x_shape[0];
    ic = x_shape[1];
    ih = x_shape[2];
    iw = x_shape[3];
  } else {
    OP_LOGE(op.GetName().c_str(),
            "input x format should be NCHW"
            " actual is: %d",
            x_format);
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["param_name"] = "xFormat";
    err_map["expected_format_list"] = "[NCHW]";
    err_map["format"] = ConcatString(x_format);
    std::string report_error_code = "E50033";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  if (w_format == FORMAT_NCHW) {
    kn = w_shape[0];
    kc = w_shape[1];
    kh = w_shape[2];
    kw = w_shape[3];
  } else {
    OP_LOGE(op.GetName().c_str(),
            "input filter format should be NCHW"
            " actual is: %d",
            w_format);
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["param_name"] = "wFormat";
    err_map["expected_format_list"] = "[NCHW]";
    err_map["format"] = ConcatString(w_format);
    std::string report_error_code = "E50033";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  int64_t groups = 1;
  if (GRAPH_SUCCESS != op.GetAttr("groups", groups)) {
    OP_LOGI(op.GetName().c_str(), "no groups setting, use groups as 1");
  }
  OP_LOGI(op.GetName().c_str(), "groups is %lld", groups);

  if (ic != kn) {
    OP_LOGE(op.GetName().c_str(),
            "input x channel should be equal to filter. "
            "x format is: %d, filter format is: %d "
            "x shape is: [%d,%d,%d,%d], filter shape is: [%d,%d,%d,%d].",
            x_format, w_format, x_shape[0], x_shape[1], x_shape[2], x_shape[3], w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["attr_name"] = "channel";
    err_map["param1_name"] = "x";
    err_map["param1_value"] = std::to_string(ic);
    err_map["param2_name"] = "filter";
    err_map["param2_value"] = std::to_string(kn);
    std::string report_error_code = "E50031";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;
  int32_t padt = 0;
  int32_t padb = 0;
  int32_t padl = 0;
  int32_t padr = 0;
  if (false == GetAttrsDeconv(op, x_format, strh, strw, dilh, dilw)) {
    OP_LOGE(op.GetName().c_str(), "get attrs failed.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "get attrs failed.";
    std::string report_error_code = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  if (false == GetPadConv2D(op, ih, iw, kh, kw, strh, strw, dilh, dilw, padt, padb, padl, padr)) {
    OP_LOGE(op.GetName().c_str(), "get pads attrs failed.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "get pads attrs failed.";
    std::string report_error_code = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  int khext = dilh * (kh - 1) + 1;
  int kwext = dilw * (kw - 1) + 1;
  int64_t oh = strh * (ih - 1) + khext - padt - padb;
  int64_t ow = strw * (iw - 1) + kwext - padl - padr;

  vector<int64_t> y_shape;
  auto y_tensor = op.get_output_desc_y();
  auto y_format = y_tensor.GetFormat();
  CHECK_FORMAT(y_format)
  if (y_format == FORMAT_NCHW) {
    y_shape.push_back(in);
    y_shape.push_back(kc * groups);
    y_shape.push_back(oh);
    y_shape.push_back(ow);
  } else {
    OP_LOGE(op.GetName().c_str(),
            "output y format should be NCHW."
            " actual is: %d",
            y_format);
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["param_name"] = "yFormat";
    err_map["expected_format_list"] = "[NCHW]";
    err_map["format"] = ConcatString(y_format);
    std::string report_error_code = "E50033";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  y_tensor.SetShape(Shape(y_shape));
  auto x_dtype = x_tensor.GetDataType();
  if (x_dtype == DT_INT8) {
    y_tensor.SetDataType(DT_INT32);
  } else {
    y_tensor.SetDataType(x_dtype);
  }
  if (GRAPH_SUCCESS != op.update_output_desc_y(y_tensor)) {
    OP_LOGE(op.GetName().c_str(), "update output desc failed.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "update output desc failed.";
    std::string report_error_code = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  OP_LOGD(op.GetName().c_str(), "Leave DeconvolutionInfer.");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Deconvolution, DeconvolutionVerify) {
  OP_LOGD(op.GetName().c_str(), "Enter DeconvolutionVerify.");
  auto x_tensor = op.get_input_desc_x();
  auto w_tensor = op.get_input_desc_filter();

  auto x_shape = x_tensor.GetShape().GetDims();
  auto w_shape = w_tensor.GetShape().GetDims();
  if (x_shape.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "input x shape should be 4d.");
    string xvalue = ConcatString(x_shape.size());
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["param_name"] = "xShape";
    err_map["expected_value"] = "4d";
    err_map["input_value"] = xvalue;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  if (w_shape.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "input filter shape should be 4d.");
    string wvalue = ConcatString(w_shape.size());
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["param_name"] = "wShape";
    err_map["expected_value"] = "4d";
    err_map["input_value"] = wvalue;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  auto x_dtype = x_tensor.GetDataType();
  auto w_dtype = w_tensor.GetDataType();
  if (x_dtype != w_dtype) {
    OP_LOGE(op.GetName().c_str(),
            "input x dtype is differ from filter dtype."
            " actual x dtype is: %d filter dtype is: %d",
            (int)x_dtype, (int)w_dtype);
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["param1"] = "x";
    err_map["param1_data_type"] = std::to_string(x_dtype);
    err_map["param2"] = "filter";
    err_map["param2_data_type"] = std::to_string(w_dtype);
    err_map["rule"] = "input x dtype is same as filter dtype";
    std::string report_error_code = "E50004";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int32_t> stride_list;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stride_list)) {
    OP_LOGE(op.GetName().c_str(), "get strides list failed.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "get strides list failed.";
    std::string report_error_code = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  std::vector<int32_t> dilation_list;
  if (GRAPH_SUCCESS != op.GetAttr("dilations", dilation_list)) {
    OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "get dilations list failed.";
    std::string report_error_code = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  OP_LOGD(op.GetName().c_str(), "Leave DeconvolutionVerify.");
  return GRAPH_SUCCESS;
}

INFER_DATA_SLICE_FUNC_REG(Deconvolution, DeconvolutionInferDataSlice);
INFER_FUNC_REG(Deconvolution, DeconvolutionInfer);
VERIFY_FUNC_REG(Deconvolution, DeconvolutionVerify);

// ---------------------------Conv3D---------------------------
static bool GetPadConv3D(ge::Operator& op, int32_t id, int32_t ih, int32_t iw,
                         int32_t kd, int32_t kh, int32_t kw, int32_t strd,
                         int32_t strh, int32_t strw, int32_t dild, const int32_t dilh,
                         int32_t dilw, int32_t& padf, int32_t& padba, int32_t& padt,
                         int32_t& padb, int32_t& padl, int32_t& padr) {
  std::string pad_str;
  std::vector<int32_t> pad_list;
  if (GRAPH_SUCCESS == op.GetAttr("_padding", pad_list)) {
    op.SetAttr("pads", pad_list);
  } else if (GRAPH_SUCCESS == op.GetAttr("padding", pad_str)) {
    if (pad_str.compare("SAME") == 0) {
      int32_t tails_d = id % strd;
      int32_t tails_h = ih % strh;
      int32_t tails_w = iw % strw;
      int32_t pad_d = std::max((tails_d > 0 ? kd - tails_d : kd - strd), 0);
      int32_t pad_h = std::max((tails_h > 0 ? kh - tails_h : kh - strh), 0);
      int32_t pad_w = std::max((tails_w > 0 ? kw - tails_w : kw - strw), 0);
      pad_list.push_back(pad_d / 2);
      pad_list.push_back(pad_d / 2 + pad_d % 2);
      pad_list.push_back(pad_h / 2);
      pad_list.push_back(pad_h / 2 + pad_h % 2);
      pad_list.push_back(pad_w / 2);
      pad_list.push_back(pad_w / 2 + pad_w % 2);
    } else if (pad_str.compare("VALID") == 0) {
      for (int32_t i = 0; i < 6; i++)
        pad_list.push_back(0);
    } else {
      OP_LOGE(op.GetName().c_str(), "padding should be SAME or VALID.");
      map<std::string, std::string> err_map;
      err_map["param_name"] = "padding";
      err_map["op_name"] = "Conv3d";
      err_map["Expected_value"] = "SAME or VALID";
      err_map["input_value"] = pad_str;
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return false;
    }
    op.SetAttr("pads", pad_list);
  }
  std::vector<int32_t> pad_vec;
  if (GRAPH_SUCCESS != op.GetAttr("pads", pad_vec)) {
    OP_LOGE(op.GetName().c_str(), "get pads failed");
    return false;
  }
  auto p_size = pad_vec.size();
  if (p_size != kConv3dPadsSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "pads list should be 6d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "pads_list";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "6d";
    err_map["input_value"] = std::to_string(p_size);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  padf = pad_vec[0];
  padba = pad_vec[1];
  padt = pad_vec[2];
  padb = pad_vec[3];
  padl = pad_vec[4];
  padr = pad_vec[5];
  if (padf < 0 || padba < 0 || padt < 0 || padb < 0 || padl < 0 || padr < 0) {
    OP_LOGE(op.GetName().c_str(), "pads should be positive");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "pads_list";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "positive";
    err_map["input_value"] = std::to_string(padf) + " " + std::to_string(padba) + " " + std::to_string(padt) + " " +
                             std::to_string(padb) + " " + std::to_string(padl) + " " + std::to_string(padr);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  return true;
}

static bool GetAttrsConv3D(ge::Operator& op, Format refer,  int32_t& strd,
                           int32_t& strh, int32_t& strw, int32_t& dild,
                           int32_t& dilh, int32_t& dilw) {
  std::vector<int32_t> stride_list;
  if (op.GetAttr("strides", stride_list) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get strides list failed.");
    return false;
  }
  auto s_size = stride_list.size();
  if (s_size != kConv3dStridesSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "strides list should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "stride_list";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "5d";
    err_map["input_value"] = std::to_string(s_size);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  // get data_format, not used for now temporarily
  std::string data_format;
  if (op.GetAttr("data_format", data_format) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The Conv3D op GetOpAttr data_format failed!");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "data_format";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "NDHWC";
    err_map["input_value"] = data_format;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  std::vector<int32_t> dilation_list;
  if (op.GetAttr("dilations", dilation_list) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
    return false;
  }
  auto d_size = dilation_list.size();
  if (d_size != kConv3dDimSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "dilations list should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "dilation_list";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "5d";
    err_map["input_value"] = std::to_string(d_size);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (refer == FORMAT_NCDHW) {
    strd = stride_list[2];
    strh = stride_list[3];
    strw = stride_list[4];
    dild = dilation_list[2];
    dilh = dilation_list[3];
    dilw = dilation_list[4];
  } else if (refer == FORMAT_NDHWC) {
    strd = stride_list[1];
    strh = stride_list[2];
    strw = stride_list[3];
    dild = dilation_list[1];
    dilh = dilation_list[2];
    dilw = dilation_list[3];
  }
  if (strd <= 0 || strh <= 0 || strw <= 0) {
    OP_LOGE(op.GetName().c_str(), "strides should be positive.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "positive";
    err_map["input_value"] = std::to_string(strd) + " " + std::to_string(strh) + " " + std::to_string(strw);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  if (dild != 1 || dilh != 1 || dilw != 1) {
    OP_LOGE(op.GetName().c_str(), "dilations only support 1 now.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "dilations";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "1, 1, 1";
    err_map["input_value"] = std::to_string(dild) + " " + std::to_string(dilh) + " " + std::to_string(dilw);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  return true;
}
IMPLEMT_INFERFUNC(Conv3D, Conv3DInfer) {
  OP_LOGD(op.GetName().c_str(), "Enter Conv3DInfer.");

  auto x_tensor = op.get_input_desc_x();
  auto x_format = x_tensor.GetFormat();
  auto x_shape = x_tensor.GetShape().GetDims();

  if (x_shape.size() != kConv3dInputSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "x_shape's shape should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "x_shape";
    err_map["op_name"] = "Conv3DInfer";
    err_map["excepted_value"] = std::to_string(5);
    err_map["input_value"] = std::to_string(x_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  int32_t in = 0;
  int32_t ic = 0;
  int32_t id = 0;
  int32_t ih = 0;
  int32_t iw = 0;
  if (x_format == FORMAT_NCDHW) {
    in = x_shape[0];
    ic = x_shape[1];
    id = x_shape[2];
    ih = x_shape[3];
    iw = x_shape[4];
  } else if (x_format == FORMAT_NDHWC) {
    in = x_shape[0];
    ic = x_shape[4];
    id = x_shape[1];
    ih = x_shape[2];
    iw = x_shape[3];
  } else {
    OP_LOGE(op.GetName().c_str(), "input x format should be NCDHW or NDHWC.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "x_format";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "NCDHW or NDHWC";
    err_map["input_value"] = x_format;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  auto w_tensor = op.get_input_desc_filter();
  auto w_shape = w_tensor.GetShape().GetDims();
  auto w_format = w_tensor.GetFormat();
  if (w_shape.size() != kConv3dInputSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "w_shape's shape should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "w_shape";
    err_map["op_name"] = "Conv3DInfer";
    err_map["excepted_value"] = std::to_string(5);
    err_map["input_value"] = std::to_string(w_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  int32_t kn = 0;
  int32_t kc = 0;
  int32_t kd = 0;
  int32_t kh = 0;
  int32_t kw = 0;
  if (w_format == FORMAT_NCDHW) {
    kn = w_shape[0];
    kc = w_shape[1];
    kd = w_shape[2];
    kh = w_shape[3];
    kw = w_shape[4];
  } else if (w_format == FORMAT_NDHWC) {
    kn = w_shape[0];
    kc = w_shape[4];
    kd = w_shape[1];
    kh = w_shape[2];
    kw = w_shape[3];
  } else if (w_format == FORMAT_DHWCN) {
    kn = w_shape[4];
    kc = w_shape[3];
    kd = w_shape[0];
    kh = w_shape[1];
    kw = w_shape[2];
  } else {
    OP_LOGE(op.GetName().c_str(), "input filter format should be NCDHW, NDHWC or DHWCN.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "wFormat";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "NCDHW or NDHWC or DHWCN";
    err_map["input_value"] = w_format;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  int64_t group = 1;
  if (GRAPH_SUCCESS != op.GetAttr("groups", group)) {
    OP_LOGI(op.GetName().c_str(), "no group setting, use group as 1");
  }

  if (ic != kc * group) {
    OP_LOGE(op.GetName().c_str(), "input x channel should be equal to filter.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3d";
    err_map["channel_of_x"] = std::to_string(ic);
    err_map["channel_of_filter"] = std::to_string(kc * group);
    std::string report_error_code = "E50039";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  int32_t strd = 0;
  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dild = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;
  if (!GetAttrsConv3D(op, x_format, strd, strh, strw, dild, dilh, dilw)) {
    OP_LOGE(op.GetName().c_str(), "get attrs failed.");
    return GRAPH_FAILED;
  }

  int32_t padf = 0;
  int32_t padba = 0;
  int32_t padt = 0;
  int32_t padb = 0;
  int32_t padl = 0;
  int32_t padr = 0;
  if (!GetPadConv3D(op, id, ih, iw, kd, kh, kw, strd, strh, strw, dild, dilh, dilw, padf, padba, padt, padb,
                            padl, padr)) {
    OP_LOGE(op.GetName().c_str(), "get pads attrs failed.");
    return GRAPH_FAILED;
  }

  int64_t od = (id + padf + padba - dild * (kd - 1) - 1) / strd + 1;
  int64_t oh = (ih + padt + padb - dilh * (kh - 1) - 1) / strh + 1;
  int64_t ow = (iw + padl + padr - dilw * (kw - 1) - 1) / strw + 1;

  vector<int64_t> y_shape;
  auto y_tensor = op.get_output_desc_y();
  auto y_format = y_tensor.GetFormat();
  CHECK_FORMAT(y_format)
  if (y_format == FORMAT_NCDHW) {
    y_shape.push_back(in);
    y_shape.push_back(kn);
    y_shape.push_back(od);
    y_shape.push_back(oh);
    y_shape.push_back(ow);
  } else if (y_format == FORMAT_NDHWC) {
    y_shape.push_back(in);
    y_shape.push_back(od);
    y_shape.push_back(oh);
    y_shape.push_back(ow);
    y_shape.push_back(kn);
  } else {
    OP_LOGE(op.GetName().c_str(), "output y format should be NCDHW or NDHWC.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "yFormat";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "NCDHW or NDHWC";
    err_map["input_value"] = y_format;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  y_tensor.SetShape(Shape(y_shape));
  auto x_dtype = x_tensor.GetDataType();
  y_tensor.SetDataType(x_dtype);
  if (GRAPH_SUCCESS != op.update_output_desc_y(y_tensor)) {
    OP_LOGE(op.GetName().c_str(), "update output desc failed.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "output_desc_y";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = GRAPH_SUCCESS;
    err_map["output_value"] = op.update_output_desc_y(y_tensor);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  OP_LOGD(op.GetName().c_str(), "Leave Conv3DInfer.");
  return GRAPH_SUCCESS;
}

static void InferHWConv3d(int32_t kernel,
                          int32_t dilation,
                          int32_t stride,
                          int32_t input_size,
                          const vector<int64_t>& output,
                          vector<int64_t>& input,
                          vector<int32_t>& pad_list,
                          uint32_t pad_idx) {
  int32_t kernel_size = (kernel - 1) * dilation + 1;
  int32_t pad_h = pad_list[pad_idx];
  input[0] = std::max(stride * output[0] - pad_h, 0L);
  input[1] = std::min(stride * output[1] - pad_h + kernel_size - 1,
                      static_cast<int64_t>(input_size - 1));

  pad_list[pad_idx] = std::max(static_cast<int32_t>(pad_h - stride * output[0]), 0);
  pad_list[pad_idx + 1] = std::max(static_cast<int32_t>(
                                      stride * output[1] - pad_h +
                                      kernel_size - input_size),
                                   0);
}

IMPLEMT_INFER_DATA_SLICE(Conv3D, Conv3DInferSliceData) {
  OP_LOGD(op.GetName().c_str(), "Enter Conv3DInferSliceData.");

  auto x_format = op.get_input_desc_x().GetFormat();
  int32_t strd = 0;
  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dild = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;
  if (!GetAttrsConv3D(op, x_format, strd, strh, strw, dild, dilh, dilw)) {
    OP_LOGE(op.GetName().c_str(), "get attrs failed.");
    return GRAPH_FAILED;
  }

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
  GeTensorDescPtr tensor_desc_w = op_desc->MutableInputDesc("filter");

  vector<vector<int64_t>> y_data_slice = {{}, {}, {}, {}, {}, {}};
  vector<vector<int64_t>> x_data_slice = {{}, {}, {}, {}, {}, {}};
  vector<vector<int64_t>> w_data_slice = {{}, {}, {}, {}};
  vector<vector<int64_t>> bias_data_slice = {{}};

  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice) != GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "no data slice, need not infer input.");
    return GRAPH_FAILED;
  }

  // check data_slice attr
  if(y_data_slice.size() != kConv3dDataSlice) {
    OP_LOGE(op.GetName().c_str(), "y_data_slice's size should be 6.");
    return GRAPH_FAILED;
  }

  // no support for C0 axis
  if(y_data_slice[kConv3dDataSlice - 1].size() != 0) {
    OP_LOGE(op.GetName().c_str(), "no support for cut C0 axis.");
    return NOT_SUPPORT_SLICE;
  }

  // check valid slice num in data slice
  int32_t valid_cnt = 0;
  for(uint32_t i = 0; i < y_data_slice.size(); ++i) {
    if(y_data_slice[i].size() == 0) {
      continue;
    }
    if(y_data_slice[i].size() != 2) {
      OP_LOGE(op.GetName().c_str(), "data slice format input size should be 2.");
      return GRAPH_FAILED;
    }
    valid_cnt ++;
  }
  if(valid_cnt == 0) {
    OP_LOGE(op.GetName().c_str(), "data slice is empty.");
    return GRAPH_FAILED;
  }
  if(valid_cnt != 1) {
    OP_LOGE(op.GetName().c_str(), "valid slice range num is more than 1.");
    return GRAPH_FAILED;
  }

  vector<int32_t> pad_list;
  op.GetAttr("pads", pad_list);
  bool needUpdateX = false;
  bool needUpdateW = false;

  auto x_shape = op.get_input_desc_x().GetShape().GetDims();
  std::string x_format_str = format2str[x_format];
  int32_t d_input_position = x_format_str.find("D");
  CHECK_POSITION(d_input_position);
  int32_t h_input_position = x_format_str.find("H");
  CHECK_POSITION(h_input_position);
  int32_t w_input_position = x_format_str.find("W");
  CHECK_POSITION(w_input_position);
  int32_t id = x_shape[d_input_position];
  int32_t ih = x_shape[h_input_position];
  int32_t iw = x_shape[w_input_position];

  auto filter_format = op.get_input_desc_filter().GetFormat();
  auto w_shape = op.get_input_desc_filter().GetShape().GetDims();
  std::string filter_format_str = format2str[filter_format];
  int32_t d_filter_position = filter_format_str.find("D");
  CHECK_POSITION(d_filter_position);
  int32_t h_filter_position = filter_format_str.find("H");
  CHECK_POSITION(h_filter_position);
  int32_t w_filter_position = filter_format_str.find("W");
  CHECK_POSITION(w_filter_position);
  int32_t kd = w_shape[d_filter_position];
  int32_t kh = w_shape[h_filter_position];
  int32_t kw = w_shape[w_filter_position];

  // cut N
  if(y_data_slice[0].size() != 0) {
    x_data_slice[0] = y_data_slice[0];
    needUpdateX = true;
  }

  // cut D
  if(y_data_slice[1].size() != 0) {
    x_data_slice[1].clear();
    x_data_slice[1].resize(2);
    InferHWConv3d(kd, dild, strd, id,
                  y_data_slice[1], x_data_slice[1], pad_list, 0);
    needUpdateX = true;
  }

  // cut Cout
  if(y_data_slice[2].size() != 0) {
    w_data_slice[1] = y_data_slice[2];
    bias_data_slice[0] = y_data_slice[2];
    needUpdateW = true;
  }

  // cut H
  if(y_data_slice[3].size() != 0) {
    x_data_slice[3].clear();
    x_data_slice[3].resize(2);
    InferHWConv3d(kh, dilh, strh, ih,
                  y_data_slice[3], x_data_slice[3], pad_list, 2);
    needUpdateX = true;
  }

  // cut W
  if(y_data_slice[4].size() != 0) {
    x_data_slice[4].clear();
    x_data_slice[4].resize(2);
    InferHWConv3d(kw, dilw, strw, iw,
                  y_data_slice[4], x_data_slice[4], pad_list, 4);
    needUpdateX = true;
  }

  // check update flag
  if(!needUpdateX && !needUpdateW) {
    OP_LOGE(op.GetName().c_str(), "there's no update in desc.");
    return GRAPH_FAILED;
  }

  // update data slice attr
  if(needUpdateX) {
    if(!AttrUtils::SetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
      OP_LOGE(op.GetName().c_str(), "set x data slice attr failed.");
      return GRAPH_FAILED;
    }
  }
  if(needUpdateW){
    if(!AttrUtils::SetListListInt(tensor_desc_w, ge::ATTR_NAME_DATA_SLICE, w_data_slice)) {
      OP_LOGE(op.GetName().c_str(), "set w data slice attr failed");
      return GRAPH_FAILED;
    }
  }

  // update pads attr info
  op.SetAttr("pads", pad_list);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv3D, Conv3DVerify) {
  OP_LOGD(op.GetName().c_str(), "Enter Conv3DVerify.");
  auto x_tensor = op.get_input_desc_x();
  auto w_tensor = op.get_input_desc_filter();

  auto x_shape = x_tensor.GetShape().GetDims();
  auto w_shape = w_tensor.GetShape().GetDims();
  if (x_shape.size() != kConv3dInputSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "input x shape should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "xShape_size";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = std::to_string(5);
    err_map["output_value"] = std::to_string(x_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  if (w_shape.size() != kConv3dInputSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "input filter shape should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "w_shape_size";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = std::to_string(5);
    err_map["output_value"] = std::to_string(w_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  auto x_dtype = x_tensor.GetDataType();
  auto w_dtype = w_tensor.GetDataType();

  if (x_dtype != w_dtype) {
    OP_LOGE(op.GetName().c_str(), "input x dtype is differ from filter dtype.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3d";
    err_map["attr_name"] = "dtype";
    err_map["param1_name"] = "input x";
    err_map["param2_name"] = "weight";
    err_map["param1_value"] = std::to_string(x_dtype);
    err_map["param2_value"] = std::to_string(w_dtype);
    std::string report_error_code = "E50031";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int32_t> stride_list;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stride_list)) {
    OP_LOGE(op.GetName().c_str(), "get strides list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3d";
    err_map["op_name"] = "strides";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  std::vector<int32_t> dilation_list;
  if (GRAPH_SUCCESS != op.GetAttr("dilations", dilation_list)) {
    OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3d";
    err_map["op_name"] = "dilations";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  OP_LOGD(op.GetName().c_str(), "Leave Conv3DVerify.");
  return GRAPH_SUCCESS;
}

INFER_DATA_SLICE_FUNC_REG(Conv3D, Conv3DInferSliceData);
INFER_FUNC_REG(Conv3D, Conv3DInfer);
VERIFY_FUNC_REG(Conv3D, Conv3DVerify);

// -----------------------------conv3dbp_common_check-----------------------------
template <typename T1, typename T2>
static bool SetPadListByPaddingConv3dbp(ge::Operator& op, const std::vector<T1>& input_sizes, Format input_format,
                                        const std::vector<T2>& filter_sizes, Format filter_format) {
  if (filter_sizes.size() < kConv3dInputSizeLimit || input_sizes.size() < kConv3dInputSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "filter_sizes or inputSizes is illegal");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "filter_size and inputsize";
    err_map["op_name"] = "Conv3dbp";
    err_map["excepted_value"] = std::to_string(5);
    err_map["input_value"] = std::to_string(filter_sizes.size()) + " " + std::to_string(input_sizes.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  CHECK_FORMAT(input_format);
  CHECK_FORMAT(filter_format);

  std::vector<int32_t> stride_list;
  if (GRAPH_FAILED == op.GetAttr("strides", stride_list)) {
    OP_LOGE(op.GetName().c_str(), "op get strides failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbp";
    err_map["param_name"] = "strides";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (stride_list.size() < kConv3dInputSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "op get strides failed.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = "Conv3dbp";
    err_map["excepted_value"] = std::to_string(3);
    err_map["input_value"] = std::to_string(stride_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  std::string input_format_str = format2str[input_format];
  int32_t h_input_position = input_format_str.find("H");
  CHECK_POSITION(h_input_position);
  int32_t w_input_position = input_format_str.find("W");
  CHECK_POSITION(w_input_position);
  int32_t d_input_position = input_format_str.find("D");
  CHECK_POSITION(d_input_position);
  int32_t dx_h = input_sizes[h_input_position];
  int32_t dx_w = input_sizes[w_input_position];
  int32_t dx_d = input_sizes[d_input_position];

  int32_t stride_h = stride_list[h_input_position];
  int32_t stride_w = stride_list[w_input_position];
  int32_t stride_d = stride_list[d_input_position];

  std::string filter_format_str = format2str[filter_format];
  int32_t h_filter_position = filter_format_str.find("H");
  CHECK_POSITION(h_filter_position);
  int32_t w_filter_position = filter_format_str.find("W");
  CHECK_POSITION(w_filter_position);
  int32_t d_filter_position = filter_format_str.find("D");
  CHECK_POSITION(d_filter_position);

  int32_t filter_h = filter_sizes[h_filter_position];
  int32_t filter_w = filter_sizes[w_filter_position];
  int32_t filter_d = filter_sizes[d_filter_position];

  std::string padding;
  std::vector<int32_t> pads;
  if (GRAPH_SUCCESS == op.GetAttr("padding", padding)) {
    OP_LOGI(op.GetName().c_str(), "op get padding succ.");
    int pad_h = 0;
    int32_t pad_up = 0;
    int32_t pad_down = 0;
    int pad_w = 0;
    int32_t pad_left = 0;
    int32_t pad_right = 0;
    int pad_d = 0;
    int32_t pad_head = 0;
    int32_t pad_tail = 0;
    if (padding == "SAME") {
      pad_h = std::max(ALIGN_CONV2DBP(dx_h, stride_h) - stride_h + filter_h - dx_h, 0);
      pad_up = pad_h / 2;
      pad_down = pad_h - pad_up;
      pad_w = std::max(ALIGN_CONV2DBP(dx_w, stride_w) - stride_w + filter_w - dx_w, 0);
      pad_left = pad_w / 2;
      pad_right = pad_w - pad_left;
      pad_d = std::max(ALIGN_CONV2DBP(dx_d, stride_d) - stride_d + filter_d - dx_d, 0);
      pad_head = pad_d / 2;
      pad_tail = pad_d - pad_head;
    }

    pads.push_back(pad_head);
    pads.push_back(pad_tail);
    pads.push_back(pad_up);
    pads.push_back(pad_down);
    pads.push_back(pad_left);
    pads.push_back(pad_right);

    op.SetAttr("pads", pads);
  }
  if (GRAPH_SUCCESS == op.GetAttr("pads", pads)) {
    if (pads.size() < kConv3dPadsSizeLimit) {
      OP_LOGE(op.GetName().c_str(), "op pads's size is illegal,pads.");
      map<std::string, std::string> err_map;
      err_map["param_name"] = "pads";
      err_map["op_name"] = "Conv3dbp";
      err_map["excepted_value"] = std::to_string(6);
      err_map["input_value"] = std::to_string(pads.size());
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return false;
    }
    if (pads[0] < 0 || pads[1] < 0 || pads[2] < 0 || pads[3] < 0 || pads[4] < 0 || pads[5] < 0) {
      OP_LOGE(op.GetName().c_str(), "op get pads is illegal");
      map<std::string, std::string> err_map;
      err_map["param_name"] = "pads";
      err_map["op_name"] = "Conv3dbp";
      err_map["excepted_value"] = "Non-negative";
      err_map["input_value"] = std::to_string(pads[0]) + " " + std::to_string(pads[1]) + " " + std::to_string(pads[2]) +
                               " " + std::to_string(pads[3]) + " " + std::to_string(pads[4]) + " " +
                               std::to_string(pads[5]);
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return false;
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "op get pads failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbp";
    err_map["param_name"] = "pads";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  OP_LOGI(op.GetName().c_str(), "op set pads succ.");
  return true;
}

static graphStatus VerifyConv3dbpInputCommon(const ge::Operator& op) {
  auto filter_desc = op.GetInputDesc("filter");
  auto filter_dtype = filter_desc.GetDataType();
  auto out_backprop_desc = op.GetInputDesc("out_backprop");
  auto out_backprop_dtype = out_backprop_desc.GetDataType();

  // check input dtype
  if (filter_dtype != out_backprop_dtype) {
    OP_LOGE(op.GetName().c_str(), "filter's dtype should equal to outBackprop's dtype.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbpInput";
    err_map["attr_name"] = "dtype";
    err_map["param1_name"] = "filter";
    err_map["param2_name"] = "outBackprop";
    err_map["param1_value"] = std::to_string(filter_dtype);
    err_map["param2_value"] = std::to_string(out_backprop_dtype);
    std::string report_error_code = "E50031";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check input tensor shape
  auto filter_shape = filter_desc.GetShape().GetDims();
  if (filter_shape.size() != kConv3dDimSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "filter's shape should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "filterShape_size";
    err_map["op_name"] = "Conv3dbpInput";
    err_map["excepted_value"] = std::to_string(5);
    err_map["input_value"] = std::to_string(filter_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  auto out_backprop_shape = out_backprop_desc.GetShape().GetDims();
  if (out_backprop_shape.size() != kConv3dDimSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "outBackprop's shape should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "outBackpropShape_size";
    err_map["op_name"] = "Conv3dbpInput";
    err_map["excepted_value"] = std::to_string(5);
    err_map["input_value"] = std::to_string(out_backprop_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check strides shape
  std::vector<int32_t> stride_list;
  if (GRAPH_SUCCESS == op.GetAttr("strides", stride_list)) {
    if (stride_list.size() != kConv3dDimSizeLimit) {
      OP_LOGE(op.GetName().c_str(), "strides should be 5d.");
      map<std::string, std::string> err_map;
      err_map["param_name"] = "strides";
      err_map["op_name"] = "Conv3dbpInput";
      err_map["excepted_value"] = std::to_string(5);
      err_map["input_value"] = std::to_string(stride_list.size());
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "get strides list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbpInput";
    err_map["param_name"] = "strides";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check dilations shape
  std::vector<int32_t> dilations_list;
  if (GRAPH_SUCCESS == op.GetAttr("dilations", dilations_list)) {
    if (dilations_list.size() != kConv3dDimSizeLimit) {
      OP_LOGE(op.GetName().c_str(), "dilations_list should be 5d.");
      map<std::string, std::string> err_map;
      err_map["param_name"] = "dilations";
      err_map["op_name"] = "Conv3dbpInput";
      err_map["excepted_value"] = std::to_string(5);
      err_map["input_value"] = std::to_string(dilations_list.size());
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbpInput";
    err_map["param_name"] = "dilations";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
static graphStatus VerifyConv3dbpPads(const ge::Operator& op) {
  std::vector<int> pads;
  if (GRAPH_SUCCESS == op.GetAttr("pads", pads)) {
    if (pads.size() < kConv3dLengthPadsLimit) {
      OP_LOGE(op.GetName().c_str(), "op pads's size is illegal,pads.");
      map<std::string, std::string> err_map;
      err_map["param_name"] = "pads";
      err_map["op_name"] = "Conv3dbpInput";
      err_map["excepted_value"] = std::to_string(6);
      err_map["input_value"] = std::to_string(pads.size());
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }

    if (pads[0] < 0 || pads[1] < 0 || pads[2] < 0 || pads[3] < 0 || pads[4] < 0 || pads[5] < 0) {
      OP_LOGE(op.GetName().c_str(), "op get pads is illegal");
      map<std::string, std::string> err_map;
      err_map["param_name"] = "pads";
      err_map["op_name"] = "Conv3dbpInput";
      err_map["excepted_value"] = "positive";
      err_map["input_value"] = std::to_string(pads[0]) + " " + std::to_string(pads[1]) + " " + std::to_string(pads[2]) +
                               " " + std::to_string(pads[3]) + " " + std::to_string(pads[4]) + " " +
                               std::to_string(pads[5]);
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "op get pads failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbpInput";
    err_map["param_name"] = "pads";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(Conv3DBackpropInput, Conv3DBackpropInputInfer) {
  OP_LOGI(op.GetName().c_str(), "Enter Conv3DBackpropInput inferfunction!");

  Tensor input_sizes_tensor;
  if (GRAPH_SUCCESS != op.GetInputConstData("input_size", input_sizes_tensor)) {
    OP_LOGE(op.GetName().c_str(), "get input_size tensor failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbpInput";
    err_map["param_name"] = "inputSizesTensor";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // get shape for output from input_sizes
  auto input_sizes_desc = op.GetInputDesc("input_size");
  DataType dtype = input_sizes_desc.GetDataType();
  std::vector<int64_t> input_sizes;
  GetConstValue(input_sizes_tensor, dtype, input_sizes);
  // std::vector<int64_t> inputSizes = op.GetInputDesc("input_sizes").GetShape().GetDims();

  // set dtype of output desc
  auto out_backprop_dtype = op.GetInputDesc("out_backprop").GetDataType();
  auto y_desc = op.GetOutputDesc("y");
  y_desc.SetDataType(out_backprop_dtype);
  // set shape of output desc, input_sizes should match the format of y
  std::vector<int64_t> y_shape;
  for (auto i : input_sizes) {
    y_shape.push_back(i);
  }
  y_desc.SetShape(ge::Shape(y_shape));

  // update output desc
  if (GRAPH_SUCCESS != op.UpdateOutputDesc("y", y_desc)) {
    OP_LOGE(op.GetName().c_str(), "update output desc failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbpInput";
    err_map["param_name"] = "output y";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> filter_sizes = op.GetInputDesc("filter").GetShape().GetDims();
  Format filter_format = op.GetInputDesc("filter").GetFormat();
  Format input_format = y_desc.GetFormat();
  // update pads list by padding[SAME,VALID]
  if (!SetPadListByPaddingConv3dbp(op, input_sizes, input_format, filter_sizes, filter_format)) {
    OP_LOGE(op.GetName().c_str(), "update pads list by padding failed.");
    return GRAPH_FAILED;
  }

  OP_LOGI(op.GetName().c_str(), "Leaving Conv3DBackpropInput inferfunction!");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv3DBackpropInput, Conv3DBackpropInputVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter Conv3DBackpropInput verifyfunction!");
  if (GRAPH_SUCCESS != VerifyConv3dbpInputCommon(op)) {
    return GRAPH_FAILED;
  }
  // check padding value
  if (GRAPH_SUCCESS == VerifyConvPadding(op) || GRAPH_SUCCESS == VerifyConv3dbpPads(op)) {
    OP_LOGI(op.GetName().c_str(), "Leaving Conv3DBackpropInput verifyfunction!");
    return GRAPH_SUCCESS;
  } else {
    OP_LOGE(op.GetName().c_str(), "Leaving Conv3DBackpropInput verifyfunction!");
    return GRAPH_FAILED;
  }
}

INFER_FUNC_REG(Conv3DBackpropInput, Conv3DBackpropInputInfer);
VERIFY_FUNC_REG(Conv3DBackpropInput, Conv3DBackpropInputVerify);

// ----------------Conv3DBackpropInputD-------------------
IMPLEMT_INFERFUNC(Conv3DBackpropInputD, Conv3DBackpropInputDInfer) {
  OP_LOGI(op.GetName().c_str(), "Enter Conv3DBackpropInputD inferfunction!");
  // get shape for output from input_size
  std::vector<int32_t> input_sizes;
  if (GRAPH_SUCCESS == op.GetAttr("input_size", input_sizes)) {
    if (input_sizes.size() != kConv3dDimSizeLimit) {
      OP_LOGE(op.GetName().c_str(), "input_size list should be 5d.");
      map<std::string, std::string> err_map;
      err_map["param_name"] = "input_size";
      err_map["op_name"] = "Conv3dbpInput";
      err_map["excepted_value"] = std::to_string(5);
      err_map["input_value"] = std::to_string(input_sizes.size());
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "get input_size list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbpInput";
    err_map["param_name"] = "input_size";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  auto out_backprop_desc = op.GetInputDesc("out_backprop");
  // get dtype for output from out_backprop
  auto out_backprop_dtype = out_backprop_desc.GetDataType();
  // set dtype of output desc
  auto y_desc = op.GetOutputDesc("y");
  y_desc.SetDataType(out_backprop_dtype);
  // set shape of output desc, input_size should match the format of y
  std::vector<int64_t> out_shape;
  for (auto i : input_sizes) {
    out_shape.push_back(i);
  }
  y_desc.SetShape(ge::Shape(out_shape));

  // update output desc
  if (GRAPH_SUCCESS != op.UpdateOutputDesc("y", y_desc)) {
    OP_LOGE(op.GetName().c_str(), "update output desc failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbpInput";
    err_map["param_name"] = "output y";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> filter_sizes = op.GetInputDesc("filter").GetShape().GetDims();
  Format filter_format = op.GetInputDesc("filter").GetFormat();
  Format input_format = y_desc.GetFormat();
  // update pads list by padding[SAME,VALID]
  if (!SetPadListByPaddingConv3dbp(op, input_sizes, input_format, filter_sizes, filter_format)) {
    OP_LOGE(op.GetName().c_str(), "Conv3DBackpropInputD update pads list by padding failed.");
    return GRAPH_FAILED;
  }
  OP_LOGI(op.GetName().c_str(), "Leaving Conv3DBackpropInputD inferfunction!");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv3DBackpropInputD, Conv3DBackpropInputDVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter Conv3DBackpropInputD verifyfunction!");
  if (GRAPH_SUCCESS != VerifyConv3dbpInputCommon(op)) {
    return GRAPH_FAILED;
  }
  // check padding value
  if (GRAPH_SUCCESS != VerifyConv3dbpPads(op)) {
    return GRAPH_FAILED;
  }
  OP_LOGI(op.GetName().c_str(), "Leaving Conv3DBackpropInputD verifyfunction!");
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Conv3DBackpropInputD, Conv3DBackpropInputDInfer);
VERIFY_FUNC_REG(Conv3DBackpropInputD, Conv3DBackpropInputDVerify);

// ----------------Conv3DBackpropFilter-------------------
static graphStatus VerifyConv3dbpFilterCommon(const ge::Operator& op) {
  auto x_desc = op.GetInputDesc("x");
  auto out_backprop_desc = op.GetInputDesc("out_backprop");

  // check input dtype
  auto x_dtype = x_desc.GetDataType();
  auto out_backprop_dtype = out_backprop_desc.GetDataType();
  if (x_dtype != out_backprop_dtype) {
    OP_LOGE(op.GetName().c_str(), "x's dtype should equal to out_backprop's dtype.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbpFilter";
    err_map["attr_name"] = "dtype";
    err_map["param1_name"] = "input x";
    err_map["param2_name"] = "outBackprop";
    err_map["param1_value"] = std::to_string(x_dtype);
    err_map["param2_value"] = std::to_string(out_backprop_dtype);
    std::string report_error_code = "E50031";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check input tensor shape
  auto x_shape = x_desc.GetShape().GetDims();
  if (x_shape.size() != kConv3dDimSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "x's shape should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "x_shape";
    err_map["op_name"] = "Conv3dbpFilter";
    err_map["excepted_value"] = std::to_string(5);
    err_map["input_value"] = std::to_string(x_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  auto out_backprop_shape = out_backprop_desc.GetShape().GetDims();
  if (out_backprop_shape.size() != kConv3dDimSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "out_backprop's shape should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "out_backprop_shape_size";
    err_map["op_name"] = "Conv3dbpFilter";
    err_map["excepted_value"] = std::to_string(5);
    err_map["input_value"] = std::to_string(out_backprop_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check strides shape
  std::vector<int32_t> stride_list;
  if (GRAPH_SUCCESS == op.GetAttr("strides", stride_list)) {
    if (stride_list.size() != kConv3dStridesSizeLimit) {
      OP_LOGE(op.GetName().c_str(), "strides should be 5d.");
      map<std::string, std::string> err_map;
      err_map["param_name"] = "stride_list";
      err_map["op_name"] = "Conv3dbpFilter";
      err_map["excepted_value"] = std::to_string(5);
      err_map["input_value"] = std::to_string(stride_list.size());
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "get strides list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbpFilter";
    err_map["param_name"] = "strides";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check dilations shape
  std::vector<int32_t> dilations_list;
  if (GRAPH_SUCCESS == op.GetAttr("dilations", dilations_list)) {
    if (dilations_list.size() != kConv3dDimSizeLimit) {
      OP_LOGE(op.GetName().c_str(), "dilations_list should be 5d.");
      map<std::string, std::string> err_map;
      err_map["param_name"] = "dilations";
      err_map["op_name"] = "Conv3dbpFilter";
      err_map["excepted_value"] = std::to_string(5);
      err_map["input_value"] = std::to_string(dilations_list.size());
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbpFilter";
    err_map["param_name"] = "dilations";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
IMPLEMT_INFERFUNC(Conv3DBackpropFilter, Conv3DBackpropFilterInfer) {
  OP_LOGI(op.GetName().c_str(), "Enter Conv3DBackpropFilter Infer Function!");

  Tensor filter_sizes_tensor;
  if (GRAPH_SUCCESS != op.GetInputConstData("filter_size", filter_sizes_tensor)) {
    OP_LOGE(op.GetName().c_str(), "get filter_size tensor failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbpFilter";
    err_map["param_name"] = "filter_tensor";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  // get shape for output from filter_size
  auto filter_sizes_desc = op.GetInputDesc("filter_size");
  DataType dtype = filter_sizes_desc.GetDataType();
  std::vector<int64_t> filter_sizes;
  GetConstValue(filter_sizes_tensor, dtype, filter_sizes);

  if (filter_sizes.size() != kConv3dInputSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "filter_sizes's shape should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "filter_sizes";
    err_map["op_name"] = "Conv3dbpFilter";
    err_map["excepted_value"] = std::to_string(5);
    err_map["input_value"] = std::to_string(filter_sizes.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // set dtype of output desc
  auto y_desc = op.GetOutputDesc("y");
  auto out_backprop_dtype = op.GetInputDesc("out_backprop").GetDataType();
  y_desc.SetDataType(out_backprop_dtype);
  // set shape of output desc, filter_size should match the format of y
  std::vector<int64_t> y_shape;
  y_shape.push_back(filter_sizes[0]);
  y_shape.push_back(filter_sizes[1]);
  y_shape.push_back(filter_sizes[2]);
  y_shape.push_back(filter_sizes[3]);
  y_shape.push_back(filter_sizes[4]);
  y_desc.SetShape(ge::Shape(y_shape));

  // update output desc
  if (GRAPH_SUCCESS != op.UpdateOutputDesc("y", y_desc)) {
    OP_LOGE(op.GetName().c_str(), "update output desc failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbpFilter";
    err_map["param_name"] = "output_y";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  std::vector<int64_t> x_sizes = op.GetInputDesc("x").GetShape().GetDims();
  Format x_format = op.GetInputDesc("x").GetFormat();
  Format filter_format = y_desc.GetFormat();
  // update pads list by padding[SAME,VALID]
  if (!SetPadListByPaddingConv3dbp(op, x_sizes, x_format, filter_sizes, filter_format)) {
    OP_LOGE(op.GetName().c_str(), "update pads list by padding failed.");
    return GRAPH_FAILED;
  }
  OP_LOGI(op.GetName().c_str(), "Leaving Conv3DBackpropFilter infer function!");
  return GRAPH_SUCCESS;
}

static graphStatus VerifyConv3dbpFilterPads(const ge::Operator& op) {
  std::vector<int> pads;
  if (GRAPH_SUCCESS == op.GetAttr("pads", pads)) {
    if (pads.size() < kConv3dLengthPadsLimit) {
      OP_LOGE(op.GetName().c_str(), "op pads's size is illegal.");
      map<std::string, std::string> err_map;
      err_map["param_name"] = "pads";
      err_map["op_name"] = "Conv3dbpFilter";
      err_map["excepted_value"] = std::to_string(6);
      err_map["input_value"] = std::to_string(pads.size());
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }

    if (pads[0] < 0 || pads[1] < 0 || pads[2] < 0 || pads[3] < 0 || pads[4] < 0 || pads[5] < 0) {
      OP_LOGE(op.GetName().c_str(), "op get pads is illegal");
      map<std::string, std::string> err_map;
      err_map["param_name"] = "pads";
      err_map["op_name"] = "Conv3dbpFilter";
      err_map["excepted_value"] = "positive";
      err_map["input_value"] = std::to_string(pads[0]) + " " + std::to_string(pads[1]) + " " + std::to_string(pads[2]) +
                               " " + std::to_string(pads[3]) + " " + std::to_string(pads[4]) + " " +
                               std::to_string(pads[5]);
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "op get pads failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbpFilter";
    err_map["param_name"] = "pads";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv3DBackpropFilter, Conv3DBackpropFilterVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter Conv3DBackpropFilter verify function!");
  if (GRAPH_SUCCESS != VerifyConv3dbpFilterCommon(op)) {
    return GRAPH_FAILED;
  }
  // check padding value
  if (GRAPH_SUCCESS == VerifyConvPadding(op) || GRAPH_SUCCESS == VerifyConv3dbpPads(op)) {
    OP_LOGI(op.GetName().c_str(), "Leaving Conv3DBackpropFilter verify function!");
    return GRAPH_SUCCESS;
  } else {
    OP_LOGE(op.GetName().c_str(), "Leaving Conv3DBackpropFilter verify function!");
    return GRAPH_FAILED;
  }
}

INFER_FUNC_REG(Conv3DBackpropFilter, Conv3DBackpropFilterInfer);
VERIFY_FUNC_REG(Conv3DBackpropFilter, Conv3DBackpropFilterVerify);

// ----------------Conv3DBackpropFilterD-------------------
IMPLEMT_INFERFUNC(Conv3DBackpropFilterD, Conv3DBackpropFilterDInfer) {
  OP_LOGI(op.GetName().c_str(), "Enter Conv3DBackpropFilterD inferfunction!");

  // get shape for output from filter_size
  std::vector<int32_t> filter_sizes;
  if (GRAPH_SUCCESS == op.GetAttr("filter_size", filter_sizes)) {
    if (filter_sizes.size() != kConv3dDimSizeLimit) {
      OP_LOGE(op.GetName().c_str(), "filter_size list should be 5d.");
      map<std::string, std::string> err_map;
      err_map["param_name"] = "filter_sizes";
      err_map["op_name"] = "Conv3dbpFilter";
      err_map["excepted_value"] = std::to_string(5);
      err_map["input_value"] = std::to_string(filter_sizes.size());
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "get filter_size list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbpFilter";
    err_map["param_name"] = "filter_size";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // set dtype of output desc
  auto y_desc = op.GetOutputDesc("y");
  y_desc.SetDataType(DT_FLOAT);

  // set shape of output desc, filter_size should match the format of y
  std::vector<int64_t> y_shape;
  y_shape.push_back(filter_sizes[0]);
  y_shape.push_back(filter_sizes[1]);
  y_shape.push_back(filter_sizes[2]);
  y_shape.push_back(filter_sizes[3]);
  y_shape.push_back(filter_sizes[4]);
  y_desc.SetShape(ge::Shape(y_shape));

  // update output desc
  if (GRAPH_SUCCESS != op.UpdateOutputDesc("y", y_desc)) {
    OP_LOGE(op.GetName().c_str(), "update output desc failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbpFilter";
    err_map["param_name"] = "output y";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> x_sizes = op.GetInputDesc("x").GetShape().GetDims();
  Format x_format = op.GetInputDesc("x").GetFormat();
  Format filter_format = y_desc.GetFormat();
  // update pads list by padding[SAME,VALID]
  if (!SetPadListByPaddingConv3dbp(op, x_sizes, x_format, filter_sizes, filter_format)) {
    OP_LOGE(op.GetName().c_str(), "update pads list by padding failed.");
    return GRAPH_FAILED;
  }

  OP_LOGI(op.GetName().c_str(), "Leaving Conv3DBackpropFilterD inferfunction!");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv3DBackpropFilterD, Conv3DBackpropFilterDVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter Conv3DBackpropFilterD verifyfunction!");
  if (GRAPH_SUCCESS != VerifyConv3dbpFilterCommon(op)) {
    return GRAPH_FAILED;
  }
  // check padding value
  if (GRAPH_SUCCESS != VerifyConv3dbpFilterPads(op)) {
    return GRAPH_FAILED;
  }
  OP_LOGI(op.GetName().c_str(), "Leaving Conv3DBackpropFilterD verifyfunction!");
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Conv3DBackpropFilterD, Conv3DBackpropFilterDInfer);
VERIFY_FUNC_REG(Conv3DBackpropFilterD, Conv3DBackpropFilterDVerify);

// -----------------------------conv3d_transpose_common_check----------------------------
template <typename T1>
static bool CheckVectorAllZero(const std::vector<T1>& list)
{
    for (const auto& iter : list) {
        if (iter != 0) {
            return false;
        }
    }
    return true;
}

template <typename T1, typename T2, typename T3>
static bool SetInputsizeListConv3dtranspose(ge::Operator& op, const std::vector<T1>& x_sizes, Format x_format,
                                            const std::vector<T2>& filter_sizes, Format filter_format,
                                            const std::vector<T3>& input_sizes, Format input_format) {
  if (x_sizes.size() != kConv3dInputSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "x_sizes is illegal");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "x_sizes";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(5);
    err_map["input_value"] = std::to_string(x_sizes.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (input_sizes.size() != kConv3dInputSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "input_sizes is illegal");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "input_sizes";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(5);
    err_map["input_value"] = std::to_string(input_sizes.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (filter_sizes.size() != kConv3dInputSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "filter_sizes is illegal");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "filter_size";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(5);
    err_map["input_value"] = std::to_string(filter_sizes.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  std::vector<int32_t> stride_list;
  if (op.GetAttr("strides", stride_list) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "op get strides failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dTranspose";
    err_map["param_name"] = "strides";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (stride_list.size() != kConv3dInputSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "op get strides failed.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(5);
    err_map["input_value"] = std::to_string(stride_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  std::vector<int32_t> dilations_list;
  if (op.GetAttr("dilations", dilations_list) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "op get dilation failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dTranspose";
    err_map["param_name"] = "dilations";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (dilations_list.size() != kConv3dInputSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "op get dilation failed.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "dilations";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(5);
    err_map["input_value"] = std::to_string(stride_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  std::vector<int32_t> output_padding_list;
  if (op.GetAttr("output_padding", output_padding_list) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "op get outputpadding failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dTranspose";
    err_map["param_name"] = "output_padding";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (output_padding_list.size() != kConv3dInputSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "op get outputpadding failed.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "output_padding";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(5);
    err_map["input_value"] = std::to_string(output_padding_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  std::string x_format_str = format2str[x_format];
  int32_t h_input_position = x_format_str.find("H");
  CHECK_POSITION(h_input_position);
  int32_t w_input_position = x_format_str.find("W");
  CHECK_POSITION(w_input_position);
  int32_t d_input_position = x_format_str.find("D");
  CHECK_POSITION(d_input_position);
  int32_t c_input_position = x_format_str.find("C");
  CHECK_POSITION(c_input_position);
  int32_t n_input_position = x_format_str.find("N");
  CHECK_POSITION(n_input_position);
  int32_t dy_h = x_sizes[h_input_position];
  int32_t dy_w = x_sizes[w_input_position];
  int32_t dy_d = x_sizes[d_input_position];
  int32_t dy_n = x_sizes[n_input_position];

  int32_t stride_h = stride_list[h_input_position];
  int32_t stride_w = stride_list[w_input_position];
  int32_t stride_d = stride_list[d_input_position];

  int32_t dilation_h = dilations_list[h_input_position];
  int32_t dilation_w = dilations_list[w_input_position];
  int32_t dilation_d = dilations_list[d_input_position];

  int32_t outputpadding_h = output_padding_list[h_input_position];
  int32_t outputpadding_w = output_padding_list[w_input_position];
  int32_t outputpadding_d = output_padding_list[d_input_position];

  std::string filter_format_str = format2str[filter_format];
  int32_t h_filter_position = filter_format_str.find("H");
  CHECK_POSITION(h_filter_position);
  int32_t w_filter_position = filter_format_str.find("W");
  CHECK_POSITION(w_filter_position);
  int32_t d_filter_position = filter_format_str.find("D");
  CHECK_POSITION(d_filter_position);
  int32_t c_filter_position = filter_format_str.find("C");
  CHECK_POSITION(c_filter_position);

  int32_t filter_h = filter_sizes[h_filter_position];
  int32_t filter_w = filter_sizes[w_filter_position];
  int32_t filter_d = filter_sizes[d_filter_position];
  int32_t filter_c = filter_sizes[c_filter_position];

  std::vector<int32_t> pads_list;
  if (op.GetAttr("pads", pads_list) == GRAPH_FAILED) {
    OP_LOGE(op.GetName().c_str(), "op get pads failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dTranspose";
    err_map["param_name"] = "pads";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (pads_list.size() != kConv3dPadsSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "op get pads failed.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "pads";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(6);
    err_map["input_value"] = std::to_string(pads_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  int32_t pad_head = pads_list[0];
  int32_t pad_tail = pads_list[1];
  int32_t pad_up = pads_list[2];
  int32_t pad_down = pads_list[3];
  int32_t pad_left = pads_list[4];
  int32_t pad_right = pads_list[5];

  std::vector<int32_t> output;
  int32_t output_h = 0;
  int32_t output_w = 0;
  int32_t output_d = 0;
  int32_t output_n = 0;
  int32_t output_c = 0;
  if (!CheckVectorAllZero(input_sizes)) {
    output_h = input_sizes[h_input_position];
    output_w = input_sizes[w_input_position];
    output_d = input_sizes[d_input_position];
    output_n = input_sizes[n_input_position];
    output_c = input_sizes[c_input_position];

  } else {
    output_d = stride_d * (dy_d - 1) + outputpadding_d + ((filter_d - 1) * dilation_d + 1) - pad_head - pad_tail;
    output_h = stride_h * (dy_h - 1) + outputpadding_h + ((filter_h - 1) * dilation_h + 1) - pad_up - pad_down;
    output_w = stride_w * (dy_w - 1) + outputpadding_w + ((filter_w - 1) * dilation_w + 1) - pad_left - pad_right;
    output_n = dy_n;
    output_c = filter_c;
  }

  if (x_format == FORMAT_NCDHW) {
    output.push_back(output_n);
    output.push_back(output_c);
    output.push_back(output_d);
    output.push_back(output_h);
    output.push_back(output_w);
  } else if (x_format == FORMAT_NDHWC) {
    output.push_back(output_n);
    output.push_back(output_d);
    output.push_back(output_h);
    output.push_back(output_w);
    output.push_back(output_c);
  } else {
    OP_LOGE(op.GetName().c_str(), "input_size format should be NCDHW or NDHWC.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "input_size";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "NCDHW or NDHWC";
    err_map["input_value"] = x_format;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // set input_size shape to dedx
  op.SetAttr("dedx", output);

  return true;
}

static bool GetAttrsConv3DTranspose(ge::Operator& op, Format refer,  int32_t& strd,
                                    int32_t& strh, int32_t& strw, int32_t& dild,
                                    int32_t& dilh, int32_t& dilw) {
  std::vector<int32_t> stride_list;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stride_list)) {
    OP_LOGE(op.GetName().c_str(), "get strides list failed.");
    return false;
  }
  auto s_size = stride_list.size();
  if (s_size != kConv3dStridesSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "strides list should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "stride_list";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = "5d";
    err_map["input_value"] = std::to_string(s_size);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  // get data_format, not used for now temporarily
  std::string data_format;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", data_format)) {
    OP_LOGE(op.GetName().c_str(), "The Conv3D op GetOpAttr data_format failed!");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "data_format";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = "NDHWC";
    err_map["input_value"] = data_format;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  std::vector<int32_t> dilation_list;
  if (GRAPH_SUCCESS != op.GetAttr("dilations", dilation_list)) {
    OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
    return false;
  }
  auto d_size = dilation_list.size();
  if (d_size != kConv3dDimSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "dilations list should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "dilation_list";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = "5d";
    err_map["input_value"] = std::to_string(d_size);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (refer == FORMAT_NCDHW) {
    strd = stride_list[2];
    strh = stride_list[3];
    strw = stride_list[4];
    dild = dilation_list[2];
    dilh = dilation_list[3];
    dilw = dilation_list[4];
  } else if (refer == FORMAT_NDHWC) {
    strd = stride_list[1];
    strh = stride_list[2];
    strw = stride_list[3];
    dild = dilation_list[1];
    dilh = dilation_list[2];
    dilw = dilation_list[3];
  }
  if (strd <= 0 || strh <= 0 || strw <= 0) {
    OP_LOGE(op.GetName().c_str(), "strides should be positive.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = "positive";
    err_map["input_value"] = std::to_string(strd) + " " + std::to_string(strh) + " " + std::to_string(strw);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  return true;
}

static graphStatus VerifyConv3dTransposeInput(const ge::Operator& op) {
  auto filter_desc = op.GetInputDesc("filter");
  auto filter_dtype = filter_desc.GetDataType();
  auto x_desc = op.GetInputDesc("x");
  auto x_dtype = x_desc.GetDataType();

  // check input dtype
  if (filter_dtype != x_dtype) {
    OP_LOGE(op.GetName().c_str(), "filter's dtype should equal to x's dtype.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dTranspose";
    err_map["attr_name"] = "dtype";
    err_map["param1_name"] = "filter";
    err_map["param2_name"] = "x";
    err_map["param1_value"] = std::to_string(filter_dtype);
    err_map["param2_value"] = std::to_string(x_dtype);
    std::string report_error_code = "E50031";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check input tensor shape
  auto filter_shape = filter_desc.GetShape().GetDims();
  if (filter_shape.size() != kConv3dDimSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "filter's shape should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "filterShape_size";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(5);
    err_map["input_value"] = std::to_string(filter_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  auto x_shape = x_desc.GetShape().GetDims();
  if (x_shape.size() != kConv3dDimSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "x's shape should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "xShape_size";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(5);
    err_map["input_value"] = std::to_string(x_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check strides shape
  std::vector<int32_t> stride_list;
  if (op.GetAttr("strides", stride_list) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get strides list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dTranspose";
    err_map["param_name"] = "strides";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  if (stride_list.size() != kConv3dDimSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "strides should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(5);
    err_map["input_value"] = std::to_string(stride_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check dilations shape
  std::vector<int32_t> dilations_list;
  if (op.GetAttr("dilations", dilations_list) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dTranspose";
    err_map["param_name"] = "dilations";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  if (dilations_list.size() != kConv3dDimSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "dilationsList list should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "dilations";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(5);
    err_map["input_value"] = std::to_string(dilations_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int32_t> output_padding_list;
  if (op.GetAttr("output_padding", output_padding_list) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get output_padding list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dTranspose";
    err_map["param_name"] = "output_padding";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  if (output_padding_list.size() != kConv3dDimSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "output_paddingList list should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "output_padding";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(5);
    err_map["input_value"] = std::to_string(output_padding_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}
// ----------------Conv3DTranspose-------------------
IMPLEMT_INFERFUNC(Conv3DTranspose, Conv3DTransposeInfer) {
  Tensor input_sizes_tensor;
  if (op.GetInputConstData("input_size", input_sizes_tensor) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get input_size tensor failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dTranspose";
    err_map["param_name"] = "inputSizesTensor";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // get shape for output from input_sizes
  auto input_sizes_desc = op.GetInputDesc("input_size");
  DataType dtype = input_sizes_desc.GetDataType();
  std::vector<int64_t> input_sizes;
  GetConstValue(input_sizes_tensor, dtype, input_sizes);

  Format filter_format = op.GetInputDesc("filter").GetFormat();
  CHECK_FORMAT(filter_format);

  auto y_desc = op.GetOutputDesc("y");
  Format input_format = y_desc.GetFormat();
  CHECK_FORMAT(input_format);

  Format x_format = op.GetInputDesc("x").GetFormat();
  CHECK_FORMAT(x_format);

  std::vector<int64_t> filter_sizes = op.GetInputDesc("filter").GetShape().GetDims();
  std::vector<int64_t> x_sizes = op.GetInputDesc("x").GetShape().GetDims();
  if (SetInputsizeListConv3dtranspose(op, x_sizes, x_format, filter_sizes, filter_format, input_sizes, input_format) ==
      false) {
    OP_LOGE(op.GetName().c_str(),
            "Conv3DTranspose update pads list by padding failed or calculate input sizes failed.");
    return GRAPH_FAILED;
  }

  // set dtype of x
  auto x_dtype = op.GetInputDesc("x").GetDataType();
  y_desc.SetDataType(x_dtype);
  // set shape of output desc, input_sizes should match the format of y
  std::vector<int64_t> y_shape;
  for (auto i : input_sizes) {
    y_shape.push_back(i);
  }
  y_desc.SetShape(ge::Shape(y_shape));

  // update output desc
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update output desc failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3DTranspose";
    err_map["param_name"] = "output y";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv3DTranspose, Conv3DTransposeVerify) {
  if (VerifyConv3dTransposeInput(op) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  // check padding value
  if (VerifyConv3dbpPads(op) == GRAPH_SUCCESS) {
    return GRAPH_SUCCESS;
  } else {
    OP_LOGE(op.GetName().c_str(), "Leaving Conv3DTranspose verifyfunction!");
    return GRAPH_FAILED;
  }
}

INFER_FUNC_REG(Conv3DTranspose, Conv3DTransposeInfer);
VERIFY_FUNC_REG(Conv3DTranspose, Conv3DTransposeVerify);
// ----------------Conv3DTransposeD-------------------
IMPLEMT_INFERFUNC(Conv3DTransposeD, Conv3DTransposeDInfer) {
  // get shape for output from input_size
  std::vector<int32_t> input_sizes;
  if (op.GetAttr("input_size", input_sizes) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get input_size list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dTranspose";
    err_map["param_name"] = "input_size";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  if (input_sizes.size() != kConv3dDimSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "input_size list should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "input_size";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(5);
    err_map["input_value"] = std::to_string(input_sizes.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  Format filter_format = op.GetInputDesc("filter").GetFormat();
  CHECK_FORMAT(filter_format);

  auto y_desc = op.GetOutputDesc("y");
  Format input_format = y_desc.GetFormat();
  CHECK_FORMAT(input_format);

  Format x_format = op.GetInputDesc("x").GetFormat();
  CHECK_FORMAT(x_format);
  // update pads list by padding[SAME,VALID] and calculate input_size
  std::vector<int64_t> filter_sizes = op.GetInputDesc("filter").GetShape().GetDims();
  std::vector<int64_t> x_sizes = op.GetInputDesc("x").GetShape().GetDims();
  if (SetInputsizeListConv3dtranspose(op, x_sizes, x_format, filter_sizes, filter_format, input_sizes, input_format) ==
      false) {
    OP_LOGE(op.GetName().c_str(),
            "Conv3DTransposeD update pads list by padding failed or calculate input sizes failed.");
    return GRAPH_FAILED;
  }
  // get dtype for output from x
  auto x_desc = op.GetInputDesc("x");
  auto x_dtype = x_desc.GetDataType();
  // set dtype of output desc
  y_desc.SetDataType(x_dtype);
  // set shape of output desc, input_size should match the format of y
  std::vector<int32_t> dedx;
  if (op.GetAttr("dedx", dedx) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get dedx list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dTranspose";
    err_map["param_name"] = "dedx";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  if (dedx.size() != kConv3dDimSizeLimit) {
    OP_LOGE(op.GetName().c_str(), "dedx list should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "dedx";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(5);
    err_map["input_value"] = std::to_string(dedx.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> out_shape;
  for (auto i : dedx) {
    out_shape.push_back(i);
  }

  y_desc.SetShape(ge::Shape(out_shape));
  // update input_size shape
  op.SetAttr("input_size", dedx);

  // update output desc
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update output desc failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dTranspose";
    err_map["param_name"] = "output y";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

static void InferHWConv3dTransposeD(int32_t kernel,
                                    int32_t dilation,
                                    int32_t stride,
                                    int32_t input_size,
                                    const vector<int64_t>& output,
                                    vector<int64_t>& input,
                                    vector<int32_t>& pad_list,
                                    uint32_t pad_idx) {
  int32_t kernel_size = (kernel - 1) * dilation + 1;
  int32_t pad_out = kernel_size - pad_list[pad_idx] - 1;

  input[0] = std::max(static_cast<int64_t>(
                        std::ceil(
                          static_cast<float>(output[0] - pad_out) /
                          static_cast<float>(stride))),
                      0L);
  input[1] = std::min((output[1] + kernel_size - 1 - pad_out) / static_cast<int64_t>(stride),
                      static_cast<int64_t>(input_size - 1));

  int32_t oh = static_cast<int32_t>(output[1] - output[0] + 1);
  int32_t ih = static_cast<int32_t>(input[1] - input[0] + 1);

  pad_list[pad_idx] = kernel_size - static_cast<int32_t>(
                                      input[0] * stride + pad_out - output[0]) - 1;
  pad_list[pad_idx + 1] = std::max(stride * (ih - 1) + kernel_size -
                                      oh - pad_list[pad_idx],
                                   0);
}

IMPLEMT_INFER_DATA_SLICE(Conv3DTransposeD, Conv3DTransposeDInfereDataSlice) {
  OP_LOGD(op.GetName().c_str(), "Enter Conv3DTransposeDInfereDataSlice.");

  auto x_format = op.get_input_desc_x().GetFormat();
  int32_t strd = 0;
  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dild = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;

  if (!GetAttrsConv3DTranspose(op, x_format, strd, strh, strw, dild, dilh, dilw)) {
    OP_LOGE(op.GetName().c_str(), "get attrs failed.");
    return GRAPH_FAILED;
  }

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
  GeTensorDescPtr tensor_desc_w = op_desc->MutableInputDesc("filter");

  vector<vector<int64_t>> y_data_slice = {{}, {}, {}, {}, {}, {}};
  vector<vector<int64_t>> x_data_slice = {{}, {}, {}, {}, {}, {}};
  vector<vector<int64_t>> w_data_slice = {{}, {}, {}, {}};
  vector<vector<int64_t>> bias_data_slice = {{}};

  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice) != GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "no data slice, not need infer input");
    return GRAPH_FAILED;
  }

  // check data_slice attr
  if (y_data_slice.size() != kConv3dDataSlice) {
    OP_LOGE(op.GetName().c_str(), "y_data_slice's size should be 6.");
    return GRAPH_FAILED;
  }

  // no support C0 axis
  if (y_data_slice[kConv3dDataSlice - 1].size() != 0) {
    OP_LOGE(op.GetName().c_str(), "no support to cut C0 axis.");
    return NOT_SUPPORT_SLICE;
  }

  // check valid slice num in data slice
  int32_t valid_cnt = 0;
  for (uint32_t i = 0; i < y_data_slice.size(); ++i) {
    if (y_data_slice[i].size() == 0) {
      continue;
    }
    if (y_data_slice[i].size() != 2) {
      OP_LOGE(op.GetName().c_str(), "data slice format input size should be 2.");
      return GRAPH_FAILED;
    }
    valid_cnt ++;
  }
  if (valid_cnt == 0) {
    OP_LOGE(op.GetName().c_str(), "data slice is empty.");
    return GRAPH_FAILED;
  }
  if (valid_cnt != 1) {
    OP_LOGE(op.GetName().c_str(), "valid slice range num is more than 1.");
    return GRAPH_FAILED;
  }

  vector<int32_t> pad_list;
  op.GetAttr("pads", pad_list);

  auto x_shape = op.get_input_desc_x().GetShape().GetDims();
  std::string x_format_str = format2str[x_format];
  int32_t d_input_position = x_format_str.find("D");
  CHECK_POSITION(d_input_position);
  int32_t h_input_position = x_format_str.find("H");
  CHECK_POSITION(h_input_position);
  int32_t w_input_position = x_format_str.find("W");
  CHECK_POSITION(w_input_position);
  int32_t id = x_shape[d_input_position];
  int32_t ih = x_shape[h_input_position];
  int32_t iw = x_shape[w_input_position];

  auto filter_format = op.get_input_desc_filter().GetFormat();
  auto w_shape = op.get_input_desc_filter().GetShape().GetDims();
  std::string filter_format_str = format2str[filter_format];
  int32_t d_filter_position = filter_format_str.find("D");
  CHECK_POSITION(d_filter_position);
  int32_t h_filter_position = filter_format_str.find("H");
  CHECK_POSITION(h_filter_position);
  int32_t w_filter_position = filter_format_str.find("W");
  CHECK_POSITION(w_filter_position);
  int32_t kd = w_shape[d_filter_position];
  int32_t kh = w_shape[h_filter_position];
  int32_t kw = w_shape[w_filter_position];

  bool needUpdateX = false;
  bool needUpdateW = false;
  // cut N
  if (y_data_slice[0].size() != 0) {
    x_data_slice[0] = y_data_slice[0];
    needUpdateX = true;
  }

  // cut D
  if (y_data_slice[1].size() != 0) {
    x_data_slice[1].clear();
    x_data_slice[1].resize(2);
    InferHWConv3dTransposeD(kd, dild, strd, id,
                           y_data_slice[1], x_data_slice[1],
                           pad_list, 0);
    needUpdateX = true;
  }

  // cut H
  if (y_data_slice[3].size() != 0) {
    x_data_slice[3].clear();
    x_data_slice[3].resize(2);
    InferHWConv3dTransposeD(kh, dilh, strh, ih,
                           y_data_slice[3], x_data_slice[3],
                           pad_list, 2);
    needUpdateX = true;
  }

  // cut W
  if (y_data_slice[4].size() != 0) {
    x_data_slice[4].clear();
    x_data_slice[4].resize(2);
    InferHWConv3dTransposeD(kw, dilw, strw, iw,
                           y_data_slice[4], x_data_slice[4],
                           pad_list, 4);
    needUpdateX = true;
  }

  // cut Cout
  if (y_data_slice[2].size() != 0) {
    w_data_slice[0].clear();
    w_data_slice[0].resize(2);
    w_data_slice[0][0] = y_data_slice[2][0] * static_cast<int64_t>(kh * kw);
    w_data_slice[0][1] = y_data_slice[2][1] * static_cast<int64_t>(kh * kw);
    bias_data_slice[0] = y_data_slice[2];
    needUpdateW = true;
  }

  // check update flag
  if (!needUpdateX && !needUpdateW) {
    OP_LOGE(op.GetName().c_str(), "there's no update in desc.");
    return GRAPH_FAILED;
  }

  // update data slice attr
  if (needUpdateX) {
    if(!AttrUtils::SetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
      OP_LOGE(op.GetName().c_str(), "set x data slice attr failed.");
      return GRAPH_FAILED;
    }
  }
  if (needUpdateW) {
    if(!AttrUtils::SetListListInt(tensor_desc_w, ge::ATTR_NAME_DATA_SLICE, w_data_slice)) {
      OP_LOGE(op.GetName().c_str(), "set w data slice attr failed");
      return GRAPH_FAILED;
    }
  }

  // update pads attr info
  op.SetAttr("pads", pad_list);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv3DTransposeD, Conv3DTransposeDVerify) {
  if (VerifyConv3dTransposeInput(op) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  // check padding value
  if (VerifyConv3dbpPads(op) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}
INFER_DATA_SLICE_FUNC_REG(Conv3DTransposeD, Conv3DTransposeDInfereDataSlice);
INFER_FUNC_REG(Conv3DTransposeD, Conv3DTransposeDInfer);
VERIFY_FUNC_REG(Conv3DTransposeD, Conv3DTransposeDVerify);

// ----------------Conv2DTransposeD-------------------
template <typename T1, typename T2, typename T3>
static bool SetInputsizeListConv2DTranspose(ge::Operator& op, const std::vector<T1>& x_sizes, Format x_format,
                                            const std::vector<T2>& filter_sizes, Format filter_format,
                                            const std::vector<T3>& input_sizes, Format input_format) {
  // the shape of input_size may be 4
  const int32_t INPUT_SIZE_LIMIT = 4;
  const int32_t PADS_SIZE_LIMIT = 4;

  if (filter_sizes.size() != INPUT_SIZE_LIMIT) {
    OP_LOGE(op.GetName().c_str(), "filter_sizes is illegal.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "filter_size";
    err_map["op_name"] = "Conv2DTranspose";
    err_map["excepted_value"] = std::to_string(4);
    err_map["input_value"] = std::to_string(filter_sizes.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  std::vector<int32_t> stride_list;
  if (op.GetAttr("strides", stride_list) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "op get strides failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DTranspose";
    err_map["param_name"] = "strides";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (stride_list.size() != INPUT_SIZE_LIMIT) {
    OP_LOGE(op.GetName().c_str(), "stride_list size is illegal.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = "Conv2DTranspose";
    err_map["excepted_value"] = std::to_string(4);
    err_map["input_value"] = std::to_string(stride_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  std::vector<int32_t> dilations_list;
  if (op.GetAttr("dilations", dilations_list) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "op get dilation failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DTranspose";
    err_map["param_name"] = "dilations";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (dilations_list.size() != INPUT_SIZE_LIMIT) {
    OP_LOGE(op.GetName().c_str(), "dilations_list size is illegal.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "dilations";
    err_map["op_name"] = "Conv2DTranspose";
    err_map["excepted_value"] = std::to_string(4);
    err_map["input_value"] = std::to_string(stride_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  std::vector<int32_t> output_padding_list;
  if (op.GetAttr("output_padding", output_padding_list) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "op get outputpadding failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DTranspose";
    err_map["param_name"] = "output_padding";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (output_padding_list.size() != INPUT_SIZE_LIMIT) {
    OP_LOGE(op.GetName().c_str(), "outputpadding size is illegal.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "output_padding";
    err_map["op_name"] = "Conv2DTranspose";
    err_map["excepted_value"] = std::to_string(4);
    err_map["input_value"] = std::to_string(output_padding_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  std::string x_format_str = format2str[x_format];
  int32_t h_input_position = x_format_str.find("H");
  CHECK_POSITION(h_input_position);
  int32_t w_input_position = x_format_str.find("W");
  CHECK_POSITION(w_input_position);
  int32_t c_input_position = x_format_str.find("C");
  CHECK_POSITION(c_input_position);
  int32_t n_input_position = x_format_str.find("N");
  CHECK_POSITION(n_input_position);
  int32_t dy_h = x_sizes[h_input_position];
  int32_t dy_w = x_sizes[w_input_position];
  int32_t dy_n = x_sizes[n_input_position];

  int32_t stride_h = stride_list[h_input_position];
  int32_t stride_w = stride_list[w_input_position];

  int32_t dilation_h = dilations_list[h_input_position];
  int32_t dilation_w = dilations_list[w_input_position];

  int32_t outputpadding_h = output_padding_list[h_input_position];
  int32_t outputpadding_w = output_padding_list[w_input_position];

  std::string filter_format_str = format2str[filter_format];
  int32_t h_filter_position = filter_format_str.find("H");
  CHECK_POSITION(h_filter_position);
  int32_t w_filter_position = filter_format_str.find("W");
  CHECK_POSITION(w_filter_position);
  int32_t c_filter_position = filter_format_str.find("C");
  CHECK_POSITION(c_filter_position);

  int32_t filter_h = filter_sizes[h_filter_position];
  int32_t filter_w = filter_sizes[w_filter_position];
  int32_t filter_c = filter_sizes[c_filter_position];

  std::vector<int32_t> pads_list;
  if (op.GetAttr("pads", pads_list) == GRAPH_FAILED) {
    OP_LOGE(op.GetName().c_str(), "op get pads failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DTranspose";
    err_map["param_name"] = "pads";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (pads_list.size() != PADS_SIZE_LIMIT) {
    OP_LOGE(op.GetName().c_str(), "op get pads_list failed.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "output_padding";
    err_map["op_name"] = "Conv2DTranspose";
    err_map["excepted_value"] = std::to_string(4);
    err_map["input_value"] = std::to_string(pads_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  int32_t pad_up = pads_list[0];
  int32_t pad_down = pads_list[1];
  int32_t pad_left = pads_list[2];
  int32_t pad_right = pads_list[3];

  std::vector<int32_t> output;
  int32_t output_h = 0;
  int32_t output_w = 0;
  int32_t output_n = 0;
  int32_t output_c = 0;
  if (!CheckVectorAllZero(input_sizes)) {
    output_h = input_sizes[h_input_position];
    output_w = input_sizes[w_input_position];
    output_n = input_sizes[n_input_position];
    output_c = input_sizes[c_input_position];

  } else {
    output_h = stride_h * (dy_h - 1) + outputpadding_h + ((filter_h - 1) * dilation_h + 1) - pad_up - pad_down;
    output_w = stride_w * (dy_w - 1) + outputpadding_w + ((filter_w - 1) * dilation_w + 1) - pad_left - pad_right;
    output_n = dy_n;
    output_c = filter_c;
  }

  if (x_format == FORMAT_NCHW) {
    output.push_back(output_n);
    output.push_back(output_c);
    output.push_back(output_h);
    output.push_back(output_w);
  } else if (x_format == FORMAT_NHWC) {
    output.push_back(output_n);
    output.push_back(output_h);
    output.push_back(output_w);
    output.push_back(output_c);
  } else {
    OP_LOGE(op.GetName().c_str(), "inputSize format should be NCHW or NHWC.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "inputSize";
    err_map["op_name"] = "Conv2DTranspose";
    err_map["excepted_value"] = "NCHW or NHWC";
    err_map["input_value"] = x_format;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // set input_size shape to dedx
  op.SetAttr("dedx", output);

  return true;
}

static graphStatus VerifyConv2DTransposeInput(const ge::Operator& op) {
  auto filter_desc = op.GetInputDesc("filter");
  auto x_desc = op.GetInputDesc("x");

  auto filter_dtype = filter_desc.GetDataType();
  auto x_dtype = x_desc.GetDataType();
  auto filter_shape = filter_desc.GetShape().GetDims();
  auto x_shape = x_desc.GetShape().GetDims();

  const int32_t DIM_SIZE_LIMIT = 4;

  // check input dtype
  if (filter_dtype != x_dtype) {
    OP_LOGE(op.GetName().c_str(), "filter's dtype should equal to x's dtype.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DTranspose";
    err_map["attr_name"] = "dtype";
    err_map["param1_name"] = "filter";
    err_map["param2_name"] = "x";
    err_map["param1_value"] = std::to_string(filter_dtype);
    err_map["param2_value"] = std::to_string(x_dtype);
    std::string report_error_code = "E50031";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check input tensor shape
  if (filter_shape.size() != DIM_SIZE_LIMIT) {
    OP_LOGE(op.GetName().c_str(), "filter's shape should be 4d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "filterShape_size";
    err_map["op_name"] = "Conv2DTranspose";
    err_map["excepted_value"] = std::to_string(4);
    err_map["input_value"] = std::to_string(filter_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  if (x_shape.size() != DIM_SIZE_LIMIT) {
    OP_LOGE(op.GetName().c_str(), "x's shape should be 4d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "xShape_size";
    err_map["op_name"] = "Conv2DTranspose";
    err_map["excepted_value"] = std::to_string(4);
    err_map["input_value"] = std::to_string(x_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check strides shape
  std::vector<int32_t> stride_list;
  if (op.GetAttr("strides", stride_list) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get strides list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DTranspose";
    err_map["param_name"] = "strides";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  if (stride_list.size() != DIM_SIZE_LIMIT) {
    OP_LOGE(op.GetName().c_str(), "strides should be 4d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = "Conv2DTranspose";
    err_map["excepted_value"] = std::to_string(4);
    err_map["input_value"] = std::to_string(stride_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check dilations shape
  std::vector<int32_t> dilations_list;
  if (op.GetAttr("dilations", dilations_list) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DTranspose";
    err_map["param_name"] = "dilations";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  if (dilations_list.size() != DIM_SIZE_LIMIT) {
    OP_LOGE(op.GetName().c_str(), "dilations_list list should be 4d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "dilations";
    err_map["op_name"] = "Conv2DTranspose";
    err_map["excepted_value"] = std::to_string(4);
    err_map["input_value"] = std::to_string(dilations_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int32_t> output_padding_list;
  if (op.GetAttr("output_padding", output_padding_list) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get output_padding list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DTranspose";
    err_map["param_name"] = "output_padding";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  if (output_padding_list.size() != DIM_SIZE_LIMIT) {
    OP_LOGE(op.GetName().c_str(), "output_paddingList list should be 4d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "output_padding";
    err_map["op_name"] = "Conv2DTranspose";
    err_map["excepted_value"] = std::to_string(4);
    err_map["input_value"] = std::to_string(output_padding_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}
// ----------------Conv2DTranspose-------------------
IMPLEMT_INFERFUNC(Conv2DTranspose, Conv2DTransposeInfer) {
  auto input_sizes_desc = op.GetInputDesc("input_size");
  auto y_desc = op.GetOutputDesc("y");

  Tensor input_sizes_tensor;
  if (op.GetInputConstData("input_size", input_sizes_tensor) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get input_size tensor failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DTranspose";
    err_map["param_name"] = "input_sizes_tensor";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // get shape for output from input_sizes
  DataType dtype = input_sizes_desc.GetDataType();
  std::vector<int64_t> input_sizes;
  GetConstValue(input_sizes_tensor, dtype, input_sizes);

  // set dtype of x
  auto x_dtype = op.GetInputDesc("x").GetDataType();
  std::vector<int64_t> filter_sizes = op.GetInputDesc("filter").GetShape().GetDims();
  std::vector<int64_t> x_sizes = op.GetInputDesc("x").GetShape().GetDims();
  Format filter_format = op.GetInputDesc("filter").GetFormat();
  Format input_format = y_desc.GetFormat();
  Format x_format = op.GetInputDesc("x").GetFormat();
  CHECK_FORMAT(filter_format);
  CHECK_FORMAT(input_format);
  CHECK_FORMAT(x_format);
  // update pads list by padding[SAME,VALID] and calculate input_size
  if (!SetInputsizeListConv2DTranspose(op, x_sizes, x_format, filter_sizes, filter_format, input_sizes, input_format)) {
    OP_LOGE(op.GetName().c_str(), "Set Conv2DTranspose InputsizeList failed.");
    return GRAPH_FAILED;
  }
  // set out type
  if (x_dtype == DT_INT8) {
    y_desc.SetDataType(DT_INT32);
  } else {
    y_desc.SetDataType(x_dtype);
  }
  // set shape of output desc, input_sizes should match the format of y
  std::vector<int64_t> y_shape;
  for (auto i : input_sizes) {
    y_shape.push_back(i);
  }
  y_desc.SetShape(ge::Shape(y_shape));

  // update output desc
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update output desc failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DTranspose";
    err_map["param_name"] = "output y";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv2DTranspose, Conv2DTransposeVerify) {
  if (VerifyConv2DTransposeInput(op) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  // check padding value
  if (VerifyConvPadding(op) == GRAPH_SUCCESS || VerifyConv2dbpPads(op) == GRAPH_SUCCESS) {
    return GRAPH_SUCCESS;
  } else {
    OP_LOGE(op.GetName().c_str(), "Leaving Conv2DTranspose verifyfunction!");
    return GRAPH_FAILED;
  }
}

INFER_FUNC_REG(Conv2DTranspose, Conv2DTransposeInfer);
VERIFY_FUNC_REG(Conv2DTranspose, Conv2DTransposeVerify);
// ----------------Conv2DTransposeD-------------------
IMPLEMT_INFERFUNC(Conv2DTransposeD, Conv2DTransposeDInfer) {
  const int32_t DIM_SIZE_LIMIT = 4;

  // get shape for output from input_size
  std::vector<int32_t> input_sizes;
  if (op.GetAttr("input_size", input_sizes) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get input_size list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DTranspose";
    err_map["param_name"] = "input_size";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  if (input_sizes.size() != DIM_SIZE_LIMIT) {
    OP_LOGE(op.GetName().c_str(), "input_size list should be 4d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "input_size";
    err_map["op_name"] = "Conv2DTranspose";
    err_map["excepted_value"] = std::to_string(4);
    err_map["input_value"] = std::to_string(input_sizes.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> filter_sizes = op.GetInputDesc("filter").GetShape().GetDims();
  std::vector<int64_t> x_sizes = op.GetInputDesc("x").GetShape().GetDims();
  Format filter_format = op.GetInputDesc("filter").GetFormat();
  auto x_desc = op.GetInputDesc("x");
  // get dtype for output from x
  auto x_dtype = x_desc.GetDataType();
  auto y_desc = op.GetOutputDesc("y");
  Format input_format = y_desc.GetFormat();
  Format x_format = op.GetInputDesc("x").GetFormat();
  CHECK_FORMAT(filter_format);
  CHECK_FORMAT(input_format);
  CHECK_FORMAT(x_format);
  // update pads list by padding[SAME,VALID] and calculate input_size
  if (!SetInputsizeListConv2DTranspose(op, x_sizes, x_format, filter_sizes, filter_format, input_sizes, input_format)) {
    OP_LOGE(op.GetName().c_str(), "Set Conv2DTranspose InputsizeList failed.");
    return GRAPH_FAILED;
  }
  // set dtype of output desc
  if (x_dtype == DT_INT8) {
    y_desc.SetDataType(DT_INT32);
  } else {
    y_desc.SetDataType(x_dtype);
  }
  // set shape of output desc, input_size should match the format of y
  std::vector<int32_t> dedx;
  if (op.GetAttr("dedx", dedx) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get dedx list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DTranspose";
    err_map["param_name"] = "dedx";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  if (dedx.size() != DIM_SIZE_LIMIT) {
    OP_LOGE(op.GetName().c_str(), "dedx list should be 4d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "dedx";
    err_map["op_name"] = "Conv2DTranspose";
    err_map["excepted_value"] = std::to_string(4);
    err_map["input_value"] = std::to_string(dedx.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> out_shape;
  for (auto i : dedx) {
    out_shape.push_back(i);
  }

  y_desc.SetShape(ge::Shape(out_shape));
  // update input_size shape
  op.SetAttr("input_size", dedx);

  // update output desc
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update output desc failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DTranspose";
    err_map["param_name"] = "output y";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv2DTransposeD, Conv2DTransposeDVerify) {
  if (VerifyConv2DTransposeInput(op) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  // check padding value
  if (VerifyConv2dbpPads(op) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Conv2DTransposeD, Conv2DTransposeDInfer);
VERIFY_FUNC_REG(Conv2DTransposeD, Conv2DTransposeDVerify);

static graphStatus VerifyDeformableOffsetsInput(const ge::Operator& op) {
    const int32_t DIM_SIZE_LIMIT = 4;
    const int32_t INPUTLIST_SIZE_LIMIT = 2;

    auto xDesc = op.GetInputDesc("x");
    auto offsetsDesc = op.GetInputDesc("offsets");

    auto xShape = xDesc.GetShape().GetDims();
    auto offsetsShape = offsetsDesc.GetShape().GetDims();

    // check input tensor shape
    if (xShape.size() != DIM_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "x's shape should be 4d, x's shape is %d.", xShape.size());
        map<std::string, std::string> err_map;
        err_map["param_name"] = "xShape_size";
        err_map["op_name"] = "DeformableOffsets";
        err_map["excepted_value"] = std::to_string(4);
        err_map["input_value"] = std::to_string(xShape.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    if (offsetsShape.size() != DIM_SIZE_LIMIT) {
    OP_LOGE(op.GetName().c_str(), "offsets's shape should be 4d, offsets's shape is %d.", offsetsShape.size());
    map<std::string, std::string> err_map;
    err_map["param_name"] = "offsetsShape_size";
    err_map["op_name"] = "DeformableOffsets";
    err_map["excepted_value"] = std::to_string(4);
    err_map["input_value"] = std::to_string(offsetsShape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
    }

    // check strides size
    std::vector<int32_t> strideList;
    if (op.GetAttr("strides", strideList) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "get strides list failed.");
        map<std::string, std::string> err_map;
        err_map["op_name"] = "DeformableOffsets";
        err_map["param_name"] = "strides";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    if (strideList.size() != INPUTLIST_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "strides size should be 2.");
        map<std::string, std::string> err_map;
        err_map["param_name"] = "strides";
        err_map["op_name"] = "DeformableOffsets";
        err_map["excepted_value"] = std::to_string(2);
        err_map["input_value"] = std::to_string(strideList.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // check pads size
    std::vector<int32_t> padsList;
    if (op.GetAttr("pads", padsList) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "get pads list failed.");
        map<std::string, std::string> err_map;
        err_map["op_name"] = "DeformableOffsets";
        err_map["param_name"] = "pads";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    if (padsList.size() != DIM_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "pads size should be 4.");
        map<std::string, std::string> err_map;
        err_map["param_name"] = "pads";
        err_map["op_name"] = "DeformableOffsets";
        err_map["excepted_value"] = std::to_string(4);
        err_map["input_value"] = std::to_string(strideList.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // check ksize size
    std::vector<int32_t> ksizeList;
    if (op.GetAttr("ksize", ksizeList) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "get ksize list failed.");
        map<std::string, std::string> err_map;
        err_map["op_name"] = "DeformableOffsets";
        err_map["param_name"] = "ksize";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    if (ksizeList.size() != INPUTLIST_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "ksize size should be 2.");
        map<std::string, std::string> err_map;
        err_map["param_name"] = "ksize";
        err_map["op_name"] = "DeformableOffsets";
        err_map["excepted_value"] = std::to_string(2);
        err_map["input_value"] = std::to_string(ksizeList.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    // check dilations size
    std::vector<int32_t> dilationsList;
    if (op.GetAttr("dilations", dilationsList) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "get dilations list failed.");
        map<std::string, std::string> err_map;
        err_map["op_name"] = "DeformableOffsets";
        err_map["param_name"] = "dilations";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    if (dilationsList.size() != DIM_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "dilationsList size should be 4.");
        map<std::string, std::string> err_map;
        err_map["param_name"] = "dilations";
        err_map["op_name"] = "DeformableOffsets";
        err_map["excepted_value"] = std::to_string(4);
        err_map["input_value"] = std::to_string(dilationsList.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}
//----------------DeformableOffsets-------------------
IMPLEMT_INFERFUNC(DeformableOffsets, DeformableOffsetsInfer)
{
    const int32_t DIM_SIZE_LIMIT = 4;
    const int32_t KSIZE_SIZE_LIMIT = 2;

    auto xDesc = op.GetInputDesc("x");
    auto yDesc = op.GetOutputDesc("y");

    std::vector<int64_t> xSizes = xDesc.GetShape().GetDims();
    if (xSizes.size() != DIM_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "input x should be 4d.");
        map<string, string> err_map;
        err_map["param_name"] = "x";
        err_map["op_name"] = "DeformableOffsets";
        err_map["excepted_value"] = std::to_string(4);
        err_map["input_value"] = std::to_string(xSizes.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    Format xFormat = xDesc.GetFormat();
    CHECK_FORMAT(xFormat);

    std::vector<int32_t> ksizeList;
    if (op.GetAttr("ksize", ksizeList) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "op get ksize failed.");
        map<string, string> err_map;
        err_map["op_name"] = "DeformableOffsets";
        err_map["param_name"] = "ksize";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    if (ksizeList.size() != KSIZE_SIZE_LIMIT) {
        OP_LOGE(op.GetName().c_str(), "input ksize should be 2d.");
        map<string, string> err_map;
        err_map["param_name"] = "ksize";
        err_map["op_name"] = "DeformableOffsets";
        err_map["excepted_value"] = std::to_string(4);
        err_map["input_value"] = std::to_string(ksizeList.size());
        std::string report_error_code = "E50029";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }
    int32_t ksizeX = ksizeList[0];
    int32_t ksizeY = ksizeList[1];

    std::string xFormatStr = format2str[xFormat];
    int32_t hInputPosition = xFormatStr.find("H");
    CHECK_POSITION(hInputPosition);
    int32_t wInputPosition = xFormatStr.find("W");
    CHECK_POSITION(wInputPosition);
    int32_t cInputPosition = xFormatStr.find("C");
    CHECK_POSITION(cInputPosition);
    int32_t nInputPosition = xFormatStr.find("N");
    CHECK_POSITION(nInputPosition);
    int32_t dy_h = xSizes[hInputPosition];
    int32_t dy_w = xSizes[wInputPosition];
    int32_t dy_c = xSizes[cInputPosition];
    int32_t dy_n = xSizes[nInputPosition];

    int64_t out_h = static_cast<int64_t>(ksizeY * dy_h);
    int64_t out_w = static_cast<int64_t>(ksizeX * dy_w);
    int64_t out_c = static_cast<int64_t>(dy_c);
    int64_t out_n = static_cast<int64_t>(dy_n);

    std::vector<int64_t> yShape;
    auto yFormat = yDesc.GetFormat();
    CHECK_FORMAT(yFormat)
    if (yFormat == FORMAT_NCHW) {
        yShape.push_back(out_n);
        yShape.push_back(out_c);
        yShape.push_back(out_h);
        yShape.push_back(out_w);
    } else if (yFormat == FORMAT_NHWC) {
        yShape.push_back(out_n);
        yShape.push_back(out_h);
        yShape.push_back(out_w);
        yShape.push_back(out_c);
    }
    yDesc.SetShape(ge::Shape(yShape));

    auto xDtype = xDesc.GetDataType();
    yDesc.SetDataType(xDtype);

    // update output desc
    if (op.UpdateOutputDesc("y", yDesc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update output desc failed.");
        map<string, string> err_map;
        err_map["op_name"] = "DeformableOffsets";
        err_map["param_name"] = "output y";
        std::string report_error_code = "E50030";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(DeformableOffsets, DeformableOffsetsVerify)
{
    if (VerifyDeformableOffsetsInput(op) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DeformableOffsets, DeformableOffsetsInfer);
VERIFY_FUNC_REG(DeformableOffsets, DeformableOffsetsVerify);

} // namespace ge

