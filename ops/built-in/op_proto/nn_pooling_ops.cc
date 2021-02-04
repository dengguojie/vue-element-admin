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
 * \file nn_pooling_ops.cpp
 * \brief
 */
/* reslove the complexity of pooling fuction. */
#include "inc/nn_pooling_ops.h"
#include <string.h>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include "graph/operator.h"
#include "op_log.h"
#include "common/util/error_manager/error_manager.h"
#include "util/common_shape_fns.h"
#include "util/error_util.h"
#include "util/util.h"
#include "register/infer_data_slice_registry.h"
#include "graph/common_error_codes.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {

namespace {
  constexpr size_t kAvgpool3DGradOriShapeDim = 5;
  constexpr size_t kAvgpool3DGradKsizeDim = 3;
  constexpr size_t kAvgpool3DGradStridesDim = 3;
  constexpr size_t kAvgpool3DGradPadsDim = 6;
  constexpr size_t kAvgpool3DGradShapeDim = 6;
  map<int, std::string> format2str = {
    {ge::FORMAT_NCHW, "NCHW"}, {ge::FORMAT_NHWC, "NHWC"}, {ge::FORMAT_HWCN, "HWCN"},
    {ge::FORMAT_DHWNC, "DHWNC"}, {ge::FORMAT_DHWCN, "DHWCN"}, {ge::FORMAT_NDHWC, "NDHWC"},
    {ge::FORMAT_NCDHW, "NCDHWS"}
  };
}

// Obtains the value of the attr.
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

static bool Construct3DPadsByPadding(std::string opName, ge::Operator& op, int32_t id, int32_t ih, int32_t iw,
                                     int32_t kd, int32_t kh, int32_t kw, int32_t strd, int32_t strh, int32_t strw) {
  std::string padStr;
  std::vector<int32_t> padList;
  int32_t padf = 0;
  int32_t padba = 0;
  int32_t padt = 0;
  int32_t padb = 0;
  int32_t padl = 0;
  int32_t padr = 0;
  if (GRAPH_SUCCESS == op.GetAttr("padding", padStr)) {
    if (padStr.compare("SAME") == 0) {
      int32_t tails_d = id % strd;
      int32_t tails_h = ih % strh;
      int32_t tails_w = iw % strw;
      int32_t pad_d = std::max((tails_d > 0 ? kd - tails_d : kd - strd), 0);
      int32_t pad_h = std::max((tails_h > 0 ? kh - tails_h : kh - strh), 0);
      int32_t pad_w = std::max((tails_w > 0 ? kw - tails_w : kw - strw), 0);
      padList.push_back(pad_d / 2);
      padList.push_back(pad_d / 2 + pad_d % 2);
      padList.push_back(pad_h / 2);
      padList.push_back(pad_h / 2 + pad_h % 2);
      padList.push_back(pad_w / 2);
      padList.push_back(pad_w / 2 + pad_w % 2);
    } else if (padStr.compare("VALID") == 0) {
      for (int32_t i = 0; i < 6; i++) {
        padList.push_back(0);
      }
    } else {
      map<string, string> err_map;
      err_map["param_name"] = "padding";
      err_map["op_name"] = opName;
      err_map["Expected_value"] = "SAME or VALID";
      err_map["input_value"] = padStr;
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return false;
    }
    op.SetAttr("pads", padList);
  }

  std::vector<int32_t> padVec;
  if (GRAPH_SUCCESS != op.GetAttr("pads", padVec)) {
    OP_LOGE(op.GetName().c_str(), "Failed to get pads!");
    return false;
  }

  auto pSize = padVec.size();
  // Check padVec.empty() for CODEX which is unnecessary
  if (padVec.empty() || (pSize != 6)) {
    map<string, string> err_map;
    err_map["param_name"] = "pads list";
    err_map["op_name"] = opName;
    err_map["excepted_value"] = "6d";
    err_map["input_value"] = std::to_string(pSize);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  padf = padVec[0];
  padba = padVec[1];
  padt = padVec[2];
  padb = padVec[3];
  padl = padVec[4];
  padr = padVec[5];
  if (padf < 0 || padba < 0 || padt < 0 || padb < 0 || padl < 0 || padr < 0) {
    map<string, string> err_map;
    err_map["param_name"] = "pads_list";
    err_map["op_name"] = "MaxPool3DGrad";
    err_map["excepted_value"] = "positive";
    err_map["input_value"] = std::to_string(padf) + " " + std::to_string(padba) + " " + std::to_string(padt) + " " +
                             std::to_string(padb) + " " + std::to_string(padl) + " " + std::to_string(padr);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  return true;
}

static bool GetPadsByPadding(Operator& op, int32_t id, int32_t ih, int32_t iw, int32_t kd, int32_t kh, int32_t kw,
                             int32_t strd, int32_t strh, int32_t strw, vector<int32_t> &pads) {
  std::string pad_str;
  std::vector<int32_t> pad_vec;
  int32_t pad_int;
  if (op.GetAttr("padding", pad_str) == GRAPH_SUCCESS) {
    if (pad_str.compare("SAME") == 0) {
      int32_t tails_d = id % strd;
      int32_t tails_h = ih % strh;
      int32_t tails_w = iw % strw;
      int32_t pad_d = std::max((tails_d > 0 ? kd - tails_d : kd - strd), 0);
      int32_t pad_h = std::max((tails_h > 0 ? kh - tails_h : kh - strh), 0);
      int32_t pad_w = std::max((tails_w > 0 ? kw - tails_w : kw - strw), 0);
      pads.push_back(pad_d / 2);
      pads.push_back(pad_d / 2 + pad_d % 2);
      pads.push_back(pad_h / 2);
      pads.push_back(pad_h / 2 + pad_h % 2);
      pads.push_back(pad_w / 2);
      pads.push_back(pad_w / 2 + pad_w % 2);
      return true;
    } else if (pad_str.compare("VALID") == 0) {
      for (int32_t i = 0; i < 6; i++) {
        pads.push_back(0);
      }
      return true;
    }
  }

  if (op.GetAttr("padding", pad_vec) == GRAPH_SUCCESS) {
    if (pad_vec.size() == 3) {
      pads.push_back(pad_vec[0]);
      pads.push_back(pad_vec[0]);
      pads.push_back(pad_vec[1]);
      pads.push_back(pad_vec[1]);
      pads.push_back(pad_vec[2]);
      pads.push_back(pad_vec[2]);
      return true;
    }
  }
  if (op.GetAttr("padding", pad_int) == GRAPH_SUCCESS) {
    pads.push_back(pad_int);
    pads.push_back(pad_int);
    pads.push_back(pad_int);
    pads.push_back(pad_int);
    pads.push_back(pad_int);
    pads.push_back(pad_int);
    return true;
  }
  return false;
}

static bool GetStridesAndKSize(Operator& op, Format refer, int32_t& strd, int32_t& strh, int32_t& strw,
                               int32_t& kd, int32_t& kh, int32_t& kw) {
  std::vector<int32_t> stride_list;
  if (op.GetAttr("strides", stride_list) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Failed to get strides!");
    return false;
  }

  std::vector<int32_t> ksize_list;
  if (op.GetAttr("ksize", ksize_list) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Failed to get ksize!");
    return false;
  }

  if (ksize_list.size() == 1) {
    kd = ksize_list[0];
    kh = ksize_list[0];
    kw = ksize_list[0];
  } else if (ksize_list.size() == 3) {
    kd = ksize_list[0];
    kh = ksize_list[1];
    kw = ksize_list[2];
  } else if (ksize_list.size() == 5) {
    if (refer == FORMAT_NCDHW) {
      kd = ksize_list[2];
      kh = ksize_list[3];
      kw = ksize_list[4];
    } else if(refer == FORMAT_NDHWC) {
      kd = ksize_list[1];
      kh = ksize_list[2];
      kw = ksize_list[3];
    } else {
      // DHWCN
      kd = ksize_list[0];
      kh = ksize_list[1];
      kw = ksize_list[2];
    }
  }

  if (stride_list.size() == 1) {
    strd = stride_list[0];
    strh = stride_list[0];
    strw = stride_list[0];
  } else if (stride_list.size() == 3) {
    strd = stride_list[0];
    strh = stride_list[1];
    strw = stride_list[2];
  } else if (stride_list.size() == 5) {
    if (refer == FORMAT_NCDHW) {
      strd = stride_list[2];
      strh = stride_list[3];
      strw = stride_list[4];
    } else if (refer == FORMAT_NDHWC) {
      strd = stride_list[1];
      strh = stride_list[2];
      strw = stride_list[3];
    } else {
      // DHWCN
      strd = stride_list[0];
      strh = stride_list[1];
      strw = stride_list[2];
    }
  }

  return true;
}

static graphStatus GetWindowedOutputSizeVerboseV2(int64_t input_size, int64_t filter_size,
                                                  int64_t dilation_rate, int64_t stride,
                                                  const std::string padding_type, bool ceil_mode,
                                                  int64_t* output_size,
                                                  int64_t* padding_before,
                                                  int64_t* padding_after) {
  if (stride <= 0) {
    OP_LOGE("Stride must be > 0, but got [%lld]", stride);
    return GRAPH_FAILED;
  }
  if (dilation_rate < 1) {
    OP_LOGE("Dilation rate must be >= 1, but got [%lld]", dilation_rate);
    return GRAPH_FAILED;
  }
  int64_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  if (padding_type == "VALID") {
    *output_size = (input_size - effective_filter_size + stride) / stride;
    *padding_before = 0;
    *padding_after = 0;
  } else if (padding_type == "EXPLICIT") {
     *output_size = (input_size + *padding_before + *padding_after -
                    effective_filter_size + stride) /
                    stride;
  } else if (padding_type == "SAME") {
    *output_size = (input_size + stride - 1) / stride;
    const int64_t padding_needed =
        std::max(int64_t{0}, (*output_size - 1) * stride +
                 effective_filter_size - input_size);
    // For odd values of total padding, add more padding at the 'right'
    // side of the given dimension.
    *padding_before = padding_needed / 2;
    *padding_after = padding_needed - *padding_before;
  } else if (padding_type == "CALCULATED") {
    if (ceil_mode) {
      *output_size = (input_size -effective_filter_size + *padding_before + *padding_after + stride - 1) / stride + 1;
    } else {
      *output_size = (input_size -effective_filter_size + *padding_before + *padding_after) / stride + 1;
    }
    const int64_t padding_needed =
        std::max(int64_t{0}, (*output_size - 1) * stride +
                 effective_filter_size - input_size);
    *padding_after = std::max(int64_t{0}, padding_needed - *padding_before);
  } else {
    OP_LOGE("Padding [%s] is invaild.", padding_type.c_str());
    return GRAPH_FAILED;
  }
  if (*output_size < 0) {
    OP_LOGE("Computed output size would be negative, but got [%lld].",
            *output_size);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Dilation2D, Dilation2DVerify) {
  auto x_shape = op.GetInputDesc("x").GetShape();
  Format input_format = op.GetInputDesc("x").GetFormat();
  auto filter_shape = op.GetInputDesc("filter").GetShape();
  std::vector<int64_t> filter_list = filter_shape.GetDims();

  if (!CheckTwoInputDtypeSame(op, "x", "filter")) {
    OP_LOGE(op.GetName().c_str(), "Two input dtypes must be same.");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(0), 4, x_shape, op.GetName().c_str())
      != GRAPH_SUCCESS) {
    OpsOneInputShapeErrReport(op.GetName(), "X Shape Size", "XShape Size != 4");
    OP_LOGE(op.GetName().c_str(), "The rank of x must be 4.");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 3, filter_shape, op.GetName().c_str())
      != GRAPH_SUCCESS) {
    OpsOneInputShapeErrReport(op.GetName(), "Filter Shape Size", "FilterShape Size != 3");
    OP_LOGE(op.GetName().c_str(), "The rank of filter must be 3.");
    return GRAPH_FAILED;
  }

  std::string data_format;
  if (op.GetAttr("data_format", data_format) != GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "data_format");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr data_format failed!");
    return GRAPH_FAILED;
  }
  if (data_format != "NHWC" && data_format != "NCHW") {
    OpsAttrValueErrReport(op.GetName(), "data_format", "NHWC,NCHW", data_format);
    OP_LOGE(op.GetName().c_str(), "data_format[%s] is invalid!", data_format.c_str());
    return GRAPH_FAILED;
  }
  if((input_format == FORMAT_NCHW && data_format != "NCHW") || (input_format == FORMAT_NHWC && data_format != "NHWC")) {
    InferShapeOtherErrReport(op.GetName(), "Input format and data_format is not same");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> strides;
  if (op.GetAttr("strides", strides) != GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "strides");
    OP_LOGE(op.GetName().c_str(), "Get attr strides failed");
    return GRAPH_FAILED;
  }
  if (strides.size() != 4) {
    AttrSizeErrReport("strides", op.GetName(), ConcatString(strides.size()), "4");
    OP_LOGE(op.GetName().c_str(), "Attr strides(%u) must be 4", strides.size());
    return GRAPH_FAILED;
  }

  std::vector<int64_t> rates;
  if (op.GetAttr("rates", rates) != GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "rates");
    OP_LOGE(op.GetName().c_str(), "Get attr rates failed");
    return GRAPH_FAILED;
  }
  if (rates.size() != 4) {
    AttrSizeErrReport("rates", op.GetName(), ConcatString(rates.size()), "4");
    OP_LOGE(op.GetName().c_str(), "Attr rates(%u) must be 4", rates.size());
    return GRAPH_FAILED;
  }

  std::string padding_mode;
  if (op.GetAttr("padding_mode", padding_mode) != GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "padding_mode");
    OP_LOGE(op.GetName().c_str(), "Get padding failed!");
    return GRAPH_FAILED;
  }
  if (padding_mode != "SAME" && padding_mode != "VALID" && padding_mode != "CALCULATED") {
    OpsAttrValueErrReport(op.GetName(), "padding_mode", "SAME,VALID,CALCULATED", padding_mode);
    OP_LOGE(op.GetName().c_str(), "Attr padding(%s) only support SAME,VALID,CALCULATED", padding_mode.c_str());
    return GRAPH_FAILED;
  }

  std::vector<int64_t> pads;
  if (op.GetAttr("pads", pads) != GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "pads");
    OP_LOGE(op.GetName().c_str(), "Get attr pads failed");
    return GRAPH_FAILED;
  }
  if (pads.size() != 4) {
    AttrSizeErrReport("pads", op.GetName(), ConcatString(pads.size()), "4");
    OP_LOGE(op.GetName().c_str(), "Attr pads(%u) must be 4", pads.size());
    return GRAPH_FAILED;
  }

  bool ceil_mode;
  if (op.GetAttr("ceil_mode", ceil_mode) != GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "ceil_mode");
    OP_LOGE(op.GetName().c_str(), "Get attr ceil_mode failed");
    return GRAPH_FAILED;
  }

  int64_t window_h;
  int64_t window_w;
  int64_t rate_n;
  int64_t rate_c;
  int64_t stride_n;
  int64_t stride_c;
  if (input_format == FORMAT_NHWC) {
    rate_n = rates[0];
    rate_c = rates[3];
    stride_n = strides[0];
    stride_c = strides[3];
    window_h = (filter_list[0] - 1) * rates[1] + 1;
    window_w = (filter_list[1] - 1) * rates[2] + 1;
  } else {
    rate_n = rates[0];
    rate_c = rates[1];
    stride_n = strides[0];
    stride_c = strides[1];
    window_h = (filter_list[1] - 1) * rates[2] + 1;
    window_w = (filter_list[2] - 1) * rates[3] + 1;
  }
  if (rate_n != 1 || rate_c != 1) {
    OpsAttrValueErrReport(op.GetName(), "rates", "1", ConcatString(rate_n, ",",rate_c));
    OP_LOGE(op.GetName().c_str(), "rates[%d,%d] of NC is invalid!", rate_n, rate_c);
    return GRAPH_FAILED;
  }
  if (stride_n != 1 || stride_c != 1) {
    OpsAttrValueErrReport(op.GetName(), "strides", "1", ConcatString(stride_n, ",",stride_c));
    OP_LOGE(op.GetName().c_str(), "strides[%d,%d] of NC is invalid!", stride_n, stride_c);
    return GRAPH_FAILED;
  }
  if (padding_mode == "CALCULATED" &&
      (pads[0] >= window_h || pads[1] >= window_h || pads[2] >= window_w || pads[3] >= window_w)) {
    InferShapeOtherErrReport(op.GetName(), "pads must be less than window size when using CALCULATED mode");
    OP_LOGE(op.GetName().c_str(), "pads must be less than window size when using CALCULATED mode");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}
IMPLEMT_INFERFUNC(Dilation2D, Dilation2DInfer) {
  auto strides =  op.get_attr_strides();
  auto rates = op.get_attr_rates();
  auto padding_mode = op.get_attr_padding_mode();
  auto pads = op.get_attr_pads();
  auto ceil_mode = op.get_attr_ceil_mode();
  Shape x_shape = op.GetInputDesc("x").GetShape();
  Shape filter_shape = op.GetInputDesc("filter").GetShape();
  auto data_type = op.GetInputDesc("x").GetDataType();
  Format input_format = op.GetInputDesc("x").GetFormat();
  TensorDesc output_desc = op.GetOutputDesc("y");

  int32_t stride_rows;
  int32_t stride_cols;
  int32_t rate_rows;
  int32_t rate_cols;
  int64_t batch_size_dim;
  int64_t in_rows_dim;
  int64_t in_cols_dim;
  int64_t filter_rows_dim;
  int64_t filter_cols_dim;
  int64_t output_depth_dim;
  int64_t unused;
  int32_t x_h_dim;
  int32_t x_w_dim;
  int32_t filter_h_dim;
  int32_t filter_w_dim;
  if (input_format == FORMAT_NHWC) {
    stride_rows = strides[1];
    stride_cols = strides[2];
    rate_rows = rates[1];
    rate_cols = rates[2];
    batch_size_dim = x_shape.GetDim(0);
    in_rows_dim = x_shape.GetDim(1);
    in_cols_dim = x_shape.GetDim(2);
    filter_rows_dim = filter_shape.GetDim(0);
    filter_cols_dim = filter_shape.GetDim(1);
    output_depth_dim = filter_shape.GetDim(2);
    unused = x_shape.GetDim(3);
    x_h_dim = 1;
    x_w_dim = 2;
    filter_h_dim = 0;
    filter_w_dim = 1;
  } else {
    stride_rows = strides[2];
    stride_cols = strides[3];
    rate_rows = rates[2];
    rate_cols = rates[3];
    batch_size_dim = x_shape.GetDim(0);
    in_rows_dim = x_shape.GetDim(2);
    in_cols_dim = x_shape.GetDim(3);
    filter_rows_dim = filter_shape.GetDim(1);
    filter_cols_dim = filter_shape.GetDim(2);
    output_depth_dim = filter_shape.GetDim(0);
    unused = x_shape.GetDim(1);
    x_h_dim = 2;
    x_w_dim = 3;
    filter_h_dim = 1;
    filter_w_dim = 2;
  }

  if (!ValueKnown(x_shape, x_h_dim) || !ValueKnown(x_shape, x_w_dim) ||
      !ValueKnown(filter_shape, filter_h_dim) || !ValueKnown(filter_shape, filter_w_dim)) {
    Shape output_shape({batch_size_dim, -1, -1, output_depth_dim});
    output_desc.SetShape(output_shape);
    output_desc.SetDataType(data_type);
    return op.UpdateOutputDesc("y", output_desc);
  }

  if (Merge(unused, output_depth_dim, unused) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Merge unused and output_depth_dim failed.");
    return GRAPH_FAILED;
  }

  auto filter_rows_eff = filter_rows_dim +
                         (filter_rows_dim - 1) * (rate_rows - 1);
  auto filter_cols_eff = filter_cols_dim +
                         (filter_cols_dim - 1) * (rate_cols - 1);

  int64_t output_rows = 0;
  int64_t output_cols = 0;
  int64_t padding_before = 0;
  int64_t padding_after = 0;

  if (padding_mode == "CALCULATED") {
    padding_before = pads[0];
    padding_after = pads[1];
  } else {
    padding_before = 0;
    padding_after = 0;
  }
  if(GetWindowedOutputSizeVerboseV2(
      in_rows_dim, filter_rows_eff, 1, stride_rows, padding_mode, ceil_mode, &output_rows,
      &padding_before, &padding_after) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  if (padding_mode == "CALCULATED") {
    padding_before = pads[2];
    padding_after = pads[3];
  } else {
    padding_before = 0;
    padding_after = 0;
  }
  if(GetWindowedOutputSizeVerboseV2(
      in_cols_dim, filter_cols_eff, 1, stride_cols, padding_mode, ceil_mode, &output_cols,
      &padding_before, &padding_after) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  if (input_format == FORMAT_NHWC) {
    Shape y_shape({batch_size_dim, output_rows, output_cols, output_depth_dim});
    output_desc.SetShape(y_shape);
  } else {
    Shape y_shape({batch_size_dim, output_depth_dim, output_rows, output_cols});
    output_desc.SetShape(y_shape);
  }
  output_desc.SetDataType(data_type);
  return op.UpdateOutputDesc("y", output_desc);
}
VERIFY_FUNC_REG(Dilation2D, Dilation2DVerify);
INFER_FUNC_REG(Dilation2D, Dilation2DInfer);
// ----------------Pooling-------------------
IMPLEMT_INFERFUNC(Pooling, PoolingInferShape) {
  auto globalPooling = op.get_attr_global_pooling();
  auto window = op.get_attr_window();
  auto pad = op.get_attr_pad();
  auto stride = op.get_attr_stride();
  auto ceilMode = op.get_attr_ceil_mode();

  auto xShape = op.get_input_desc_x().GetShape().GetDims();
  auto xDtype = op.get_input_desc_x().GetDataType();
  auto xFormat = op.get_input_desc_x().GetFormat();

  int64_t inputN = 0;
  int64_t inputC = 0;
  int64_t inputH = 0;
  int64_t inputW = 0;

  if (xFormat == FORMAT_NCHW) {
    inputN = xShape[0];
    inputC = xShape[1];
    inputH = xShape[2];
    inputW = xShape[3];
  } else if (xFormat == FORMAT_NHWC) {
    inputN = xShape[0];
    inputC = xShape[3];
    inputH = xShape[1];
    inputW = xShape[2];
  } else {
    OP_LOGE(op.GetName().c_str(),
            "xFormat should be NCHW or NHWC."
            " actual is: %d",
            (int)xFormat);
    OpsInputFormatErrReport(op.GetName(), "xFormat", "NCHW or NHWC", ConcatString(xFormat));
    return GRAPH_FAILED;
  }

  int64_t strH = stride[0];
  int64_t strW = stride[1];

  int64_t windowH = window[0];
  int64_t windowW = window[1];

  // update globalPooling default value
  if (globalPooling) {
    windowH = inputH;
    windowW = inputW;
    (void)op.set_attr_window({windowH, windowW});
  }

  int64_t padT = pad[0];
  int64_t padB = pad[1];
  int64_t padL = pad[2];
  int64_t padR = pad[3];

  if ((padT != padB) || (padL != padR)) {
    InferShapeOtherErrReport(op.GetName(), "padT should equals padB, and padL should equals padR in caffe!");
    OP_LOGE(op.GetName().c_str(), "padT should equals padB, and padL should equals padR in caffe");
    return GRAPH_FAILED;
  }

  // init output
  int64_t outputH = inputH;
  int64_t outputW = inputW;
  int64_t outputN = inputN;
  int64_t outputC = inputC;

  // update output
  if (ceilMode == 0) {
    outputH = static_cast<int64_t>(std::ceil((inputH + padT + padB - windowH) * 1.0f / strH)) + 1;
    outputW = static_cast<int64_t>(std::ceil((inputW + padL + padR - windowW) * 1.0f / strW)) + 1;
  } else if (ceilMode == 1) {
    outputH = static_cast<int64_t>(std::floor((inputH + padT + padB - windowH) * 1.0f / strH)) + 1;
    outputW = static_cast<int64_t>(std::floor((inputW + padL + padR - windowW) * 1.0f / strW)) + 1;
  } else {
    OP_LOGE(op.GetName().c_str(), "Unknown rounding mode.");
    return GRAPH_FAILED;
  }

  bool hasPad = padT || padL;
  if (hasPad) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((outputH - 1) * strH >= inputH + padT) {
      --outputH;
    }
    if ((outputW - 1) * strW >= inputW + padL) {
      --outputW;
    }

    // CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
    bool conditionH = ((outputH - 1) * strH) <= inputH + padT;
    if (!conditionH) {
      InferShapeOtherErrReport(op.GetName(), "CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_) failed!");
      OP_LOGE(op.GetName().c_str(), "CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_) failed!");
      return GRAPH_FAILED;
    }

    // CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
    bool conditionW = ((outputW - 1) * strW) <= inputW + padL;
    if (!conditionW) {
      InferShapeOtherErrReport(op.GetName(), "CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_) failed!");
      OP_LOGE(op.GetName().c_str(), "CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_) failed!");
      return GRAPH_FAILED;
    }
  }

  auto outdesc = op.GetOutputDesc("y");
  auto yFormat = outdesc.GetFormat();
  vector<int64_t> yShape;
  if (yFormat == FORMAT_NCHW) {
    yShape.push_back(outputN);
    yShape.push_back(outputC);
    yShape.push_back(outputH);
    yShape.push_back(outputW);
  } else if (yFormat == FORMAT_NHWC) {
    yShape.push_back(outputN);
    yShape.push_back(outputH);
    yShape.push_back(outputW);
    yShape.push_back(outputC);
  } else {
    OP_LOGE(op.GetName().c_str(),
            "yFormat should be NCHW or NHWC."
            " actual is: %d",
            (int)yFormat);
    OpsInputFormatErrReport(op.GetName(), "xFormat", "NCHW or NHWC", ConcatString(xFormat));
    return GRAPH_FAILED;
  }

  outdesc.SetShape(Shape(yShape));
  outdesc.SetDataType(ge::DataType(xDtype));

  if (GRAPH_SUCCESS != op.update_output_desc_y(outdesc)) {
    OpsOPUpdateErrReport(op.GetName(), "Output Description");
    OP_LOGE(op.GetName().c_str(), "update output desc failed.");
    return GRAPH_FAILED;
  }

  OP_LOGD(op.GetName().c_str(), "Leave PoolingInfer.");
  return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(Pooling, PoolingVerify) {
  auto window = op.get_attr_window();
  auto stride = op.get_attr_stride();
  auto xShape = op.get_input_desc_x().GetShape().GetDims();
  auto pad = op.get_attr_pad();
  if (xShape.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "pooling check input size is invalid");
    OpsOneInputShapeErrReport(op.GetName(), "X Shape Size", "XShape Size != 4");
    return GRAPH_FAILED;
  }
  if (window.size() != 2) {
    OP_LOGE(op.GetName().c_str(), "pooling check window size is invalid");
    OpsAttrValueErrReport(op.GetName(), "Window Size", "2", ConcatString(window.size()));
    return GRAPH_FAILED;
  }
  if (stride.size() != 2) {
    OP_LOGE(op.GetName().c_str(), "pooling check stride size is invalid");
    OpsAttrValueErrReport(op.GetName(), "Pooling Stride", "2", ConcatString(stride.size()));
    return GRAPH_FAILED;
  }
  if (pad.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "pooling check pad size is invalid");
    OpsAttrValueErrReport(op.GetName(), "Pooling Pad Size", "4", ConcatString(pad.size()));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFORMAT_FUNC(Pooling, PoolingInferFormat) {
  OP_LOGD(op.GetName().c_str(), "Enter Pooling op_proto infer format function!");
  std::string dataFormat;
  TensorDesc tensordesc_input = op.GetInputDesc("x");
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  if (GRAPH_SUCCESS == op.GetAttr("data_format", dataFormat)) {
    if(dataFormat != "NHWC" && dataFormat != "NCHW") {
      OP_LOGE(op.GetName().c_str(), "dataFormat only support 'NHWC', 'NCHW'.");
      OpsInputFormatErrReport(op.GetName(), "data", "NCHW or NHWC", ConcatString(dataFormat));
      return GRAPH_FAILED;
    }
    if(dataFormat == "NHWC" && tensordesc_input.GetOriginFormat() == Format::FORMAT_ND) {
      tensordesc_input.SetOriginFormat(FORMAT_NHWC);
      tensordesc_output.SetOriginFormat(FORMAT_NHWC);
      (void)op.UpdateOutputDesc("x", tensordesc_input);
      (void)op.UpdateOutputDesc("y", tensordesc_output);
    }
    if(dataFormat == "NCHW" && tensordesc_input.GetOriginFormat() == Format::FORMAT_ND) {
      tensordesc_input.SetOriginFormat(FORMAT_NCHW);
      tensordesc_output.SetOriginFormat(FORMAT_NCHW);
      (void)op.UpdateOutputDesc("x", tensordesc_input);
      (void)op.UpdateOutputDesc("y", tensordesc_output);
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "data_format is null, please check!");
    OpsInputFormatErrReport(op.GetName(), "data", "NHWC, NCHW", "NULL");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

static void InferHWPooling(int64_t kernel, int64_t dilation, int64_t& pad_pre, int64_t& pad_after,
                           int64_t stride, vector<int64_t>& output, vector<int64_t>& input,
                           int64_t& ori_input, int64_t& ori_output) {
  if (kernel != 1) {
    int64_t first_start = 0;
    int64_t second_start = 0;
    int64_t first_end = 0;
    int64_t second_end = 0;
    int64_t start = 0;
    int64_t end = 0;
    first_start = output[0] * stride - pad_pre;
    second_start = output[1] * stride - pad_pre;
    first_end = std::min(first_start + kernel, ori_input + pad_pre);
    second_end = std::min(second_start + kernel, ori_input);

    start = std::max(first_start, int64_t(0));
    end = second_end - 1;
    input = {start, end};

    if (output[0] == 0) {
       if (output[0] != ori_output - 1) {
         pad_after = 0;
       }
    } else {
      if (output[0] != ori_output - 1) {
         pad_pre = 0;
         pad_after = 0;
       } else {
         pad_pre = 0;
       }
    }
  } else {
    input = output;
  }
}

IMPLEMT_INFER_DATA_SLICE(Pooling, PoolingInferDataSlice) {
  auto globalPooling = op.get_attr_global_pooling();
  auto window = op.get_attr_window();
  auto pad = op.get_attr_pad();
  auto stride = op.get_attr_stride();
  auto ceilMode = op.get_attr_ceil_mode();
  auto xShape = op.get_input_desc_x().GetShape().GetDims();
  auto xFormat = op.get_input_desc_x().GetFormat();

  int64_t dilationH = 0;
  int64_t inputN = 0;
  int64_t inputC = 0;
  int64_t inputH = 0;
  int64_t inputW = 0;

  if (xFormat == FORMAT_NCHW) {
    inputN = xShape[0];
    inputC = xShape[1];
    inputH = xShape[2];
    inputW = xShape[3];
  } else if (xFormat == FORMAT_NHWC) {
    inputN = xShape[0];
    inputC = xShape[3];
    inputH = xShape[1];
    inputW = xShape[2];
  } else {
    OP_LOGE(op.GetName().c_str(),
            "xFormat should be NCHW or NHWC."
            " actual is: %d",
            (int)xFormat);
    OpsInputFormatErrReport(op.GetName(), "xFormat", "NCHW or NHWC", ConcatString(xFormat));
    return GRAPH_FAILED;
  }

  int64_t outputH = inputH;
  int64_t outputW = inputW;
  int64_t strH = stride[0];
  int64_t strW = stride[1];
  int64_t windowH = window[0];
  int64_t windowW = window[1];
  int64_t padT = pad[0];
  int64_t padB = pad[1];
  int64_t padL = pad[2];
  int64_t padR = pad[3];
   // update output
  if (ceilMode == 0) {
    outputH = static_cast<int64_t>(std::ceil((inputH + padT + padB - windowH) * 1.0f / strH)) + 1;
    outputW = static_cast<int64_t>(std::ceil((inputW + padL + padR - windowW) * 1.0f / strW)) + 1;
  } else if (ceilMode == 1) {
    outputH = static_cast<int64_t>(std::floor((inputH + padT + padB - windowH) * 1.0f / strH)) + 1;
    outputW = static_cast<int64_t>(std::floor((inputW + padL + padR - windowW) * 1.0f / strW)) + 1;
  } else {
    OP_LOGE(op.GetName().c_str(), "Unknown rounding mode.");
    return GRAPH_FAILED;
  }

  bool hasPad = padT || padL;
  if (hasPad) {
    if ((outputH - 1) * strH >= inputH + padT) {
      --outputH;
    }
    if ((outputW - 1) * strW >= inputW + padL) {
      --outputW;
    }
  }

  vector<vector<int64_t>> y_data_slice = {{}, {}, {}, {}, {}};
  vector<vector<int64_t>> x_data_slice = {{}, {}, {}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_out = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_in = op_desc->MutableInputDesc("x");
  if (!ge::AttrUtils::GetListListInt(tensor_desc_out, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op.GetName().c_str(), "no data slice, use default as {{}, {}, {}, {}, {}}");
    return GRAPH_FAILED;
  }

  for(unsigned i = 0; i < y_data_slice.size(); i++) {
    if (y_data_slice[i].size() > 0) {
      if (i == 0) {
        return NO_OVERLAP_DIM;
      } else if (i == 1 or i == 3 or i == 4) {
        return NOT_SUPPORT_SLICE;
      } else if (i == 2) {
        vector<int64_t> input_h;
        InferHWPooling(windowH, dilationH, pad[0], pad[1], stride[0], y_data_slice[i], input_h, inputH, outputH);
        x_data_slice[i] = input_h;
      }
    }
  }

  for(unsigned i = 0; i < x_data_slice.size(); i++) {
    if (x_data_slice[i].size() > 0) {
      if(!AttrUtils::SetListListInt(tensor_desc_in, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
        return GRAPH_FAILED;
      }
      (void)op.set_attr_pad(pad);
      return GRAPH_SUCCESS;
    }
    return NO_OVERLAP_DIM;
  }

   return NO_OVERLAP_DIM;
}

INFER_FUNC_REG(Pooling, PoolingInferShape);
VERIFY_FUNC_REG(Pooling, PoolingVerify);
INFER_FORMAT_FUNC_REG(Pooling, PoolingInferFormat);
INFER_DATA_SLICE_FUNC_REG(Pooling, PoolingInferDataSlice);
// ----------------Pooling-------------------

// ----------------AvgPool-------------------
IMPLEMT_VERIFIER(AvgPool, AvgPoolVerify) {
  auto ksize = op.get_attr_ksize();
  auto strides = op.get_attr_strides();
  auto xShape = op.get_input_desc_x().GetShape().GetDims();
  bool invalidParam = (xShape.size() != 4 || ksize.size() != 4 || strides.size() != 4);
  if (invalidParam) {
    InferShapeOtherErrReport(op.GetName(), "xShape size != 4 or kSize size() != 4, strides size() != 4, please check!");
    OP_LOGE(op.GetName().c_str(), "xShape size != 4 or kSize size() != 4, strides size() != 4, please check!");
    return GRAPH_FAILED;
  }
  std::string dataFormat;
  if (GRAPH_SUCCESS == op.GetAttr("data_format", dataFormat)) {
    if (dataFormat != "NHWC" && dataFormat != "NCHW" && dataFormat != "NC1HWC0") {
      string expected_format_list = ConcatString("NHWC,NCHW,NC1HWC0");
      OpsInputFormatErrReport(op.GetName(), "data_format", expected_format_list, dataFormat);
      OP_LOGE(op.GetName().c_str(),
              "dataFormat only "
              "support 'NHWC', 'NCHW' and 'NC1HWC0'.");
      return GRAPH_FAILED;
    } else {
      if (dataFormat == "NHWC") {
        if (ksize[0] != 1 || ksize[3] != 1) {
          InferShapeOtherErrReport(
              op.GetName(), "Only supports pooling across width/height, and other ksize dimension should be one");
          OP_LOGE(op.GetName().c_str(),
                  "Only supports pooling across "
                  "width/height, and other ksize dimension should be one");
          return GRAPH_FAILED;
        }
        if (strides[0] != 1 || strides[3] != 1) {
          InferShapeOtherErrReport(
              op.GetName(), "Only supports pooling across width/height, and other strides dimension should be one");
          OP_LOGE(op.GetName().c_str(),
                  "Only supports pooling across "
                  "width/height, and other strides dimension should be one");
          return GRAPH_FAILED;
        }
      } else {
        if (ksize[0] != 1 || ksize[1] != 1) {
          InferShapeOtherErrReport(
              op.GetName(), "Only supports pooling across width/height, and other ksize dimension should be one");
          OP_LOGE(op.GetName().c_str(),
                  "Only supports pooling across "
                  "width/height, and other ksize dimension should be one");
          return GRAPH_FAILED;
        }
        if (strides[0] != 1 || strides[1] != 1) {
          InferShapeOtherErrReport(
              op.GetName(), "Only supports pooling across width/height, and other strides dimension should be one");
          OP_LOGE(op.GetName().c_str(),
                  "Only supports pooling across "
                  "width/height, and other strides dimension should be one");
          return GRAPH_FAILED;
        }
      }
    }
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(AvgPool, AvgPoolInferShape) {
  const size_t DIM_SIZE1 = 1;
  const size_t DIM_SIZE2 = 2;
  const size_t DIM_SIZE3 = 3;
  const size_t DIM_SIZE4 = 4;
  auto inputTensorDesc = op.GetInputDesc("x");
  auto shape = inputTensorDesc.GetShape();
  Format input_format = inputTensorDesc.GetFormat();
  // get input ksize
  std::vector<int32_t> ksizeList;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksizeList)) {
    OpsGetAttrErrReport(op.GetName(), "ksize");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ksizeList failed!");
    return GRAPH_FAILED;
  }

  if (ksizeList.size() != DIM_SIZE4) {
    OpsAttrValueErrReport(op.GetName(), "length of ksize", ConcatString(DIM_SIZE4),
                          ConcatString((size_t)ksizeList.size()));
    OP_LOGE(op.GetName().c_str(),
            "length of ksize must be equal to the "
            "length of shape!");
    return GRAPH_FAILED;
  }

  // get input strides
  std::vector<int32_t> stridesList;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stridesList)) {
    OpsGetAttrErrReport(op.GetName(), "strides");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr stridesList failed!");
    return GRAPH_FAILED;
  }

  if (stridesList.size() != DIM_SIZE4) {
    OpsAttrValueErrReport(op.GetName(), "length of strides", ConcatString(DIM_SIZE4),
                          ConcatString((size_t)stridesList.size()));
    OP_LOGE(op.GetName().c_str(),
            "length of strides must be equal to "
            "the length of shape!");
    return GRAPH_FAILED;
  }

  // get input data_format
  std::string dataFormat;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
    OpsGetAttrErrReport(op.GetName(), "data_format");
    OP_LOGE(op.GetName().c_str(),
            "The AvgPool op GetOpAttr data_format "
            "failed!");
    return GRAPH_FAILED;
  }

  // get input paddingMode
  std::string paddingMode;
  if (GRAPH_SUCCESS != op.GetAttr("padding", paddingMode)) {
    OpsGetAttrErrReport(op.GetName(), "padding");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr padding failed!");
    return GRAPH_FAILED;
  }

  if (paddingMode != "SAME" && paddingMode != "VALID") {
    string expected_format_list = ConcatString("SAME,VALID");
    OpsInputFormatErrReport(op.GetName(), "padding", expected_format_list, paddingMode);
    OP_LOGE(op.GetName().c_str(),
            "AvgPool can only support SAME or VALID "
            "padding mode!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dims_input = shape.GetDims();

  // set for global avg pool
  if (dataFormat == "NHWC") {
    if (ksizeList[1] == -1 && ksizeList[2] == -1) {
      ksizeList[1] = dims_input[1];
      ksizeList[2] = dims_input[2];
    }
  } else {
    if (ksizeList[2] == -1 && ksizeList[3] == -1) {
      ksizeList[2] = dims_input[2];
      ksizeList[3] = dims_input[3];
    }
  }
  op.SetAttr("ksize", ksizeList);

  // set output shape
  std::vector<int64_t> dimVector;
  if (input_format == FORMAT_NHWC) {
    if (paddingMode == "SAME") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == DIM_SIZE1) || (i == DIM_SIZE2)) {
          if (dataFormat == "NHWC") {
            int64_t dims = (dims_input[i] + stridesList[i] - 1) / stridesList[i];
            dimVector.push_back(dims);
          } else {
            int64_t dims = (dims_input[i] + stridesList[i + 1] - 1) / stridesList[i + 1];
            dimVector.push_back(dims);
          }
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    } else {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == DIM_SIZE1) || (i == DIM_SIZE2)) {
          if (dataFormat == "NHWC") {
            int64_t dims = (dims_input[i] - ksizeList[i] + 1 + (stridesList[i] - 1)) / stridesList[i];
            dimVector.push_back(dims);
          } else {
            int64_t dims = (dims_input[i] - ksizeList[i + 1] + 1 + (stridesList[i + 1] - 1)) / stridesList[i + 1];
            dimVector.push_back(dims);
          }
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    }
  } else {
    if (paddingMode == "SAME") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == DIM_SIZE2) || (i == DIM_SIZE3)) {
          if (dataFormat == "NHWC") {
            int64_t dims = (dims_input[i] + stridesList[i - 1] - 1) / stridesList[i - 1];
            dimVector.push_back(dims);
          } else {
            int64_t dims = (dims_input[i] + stridesList[i] - 1) / stridesList[i];
            dimVector.push_back(dims);
          }
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    } else {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == DIM_SIZE2) || (i == DIM_SIZE3)) {
          if (dataFormat == "NHWC") {
            int64_t dims = (dims_input[i] - ksizeList[i - 1] + 1 + (stridesList[i - 1] - 1)) / stridesList[i - 1];
            dimVector.push_back(dims);
          } else {
            int64_t dims = (dims_input[i] - ksizeList[i] + 1 + (stridesList[i] - 1)) / stridesList[i];
            dimVector.push_back(dims);
          }
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    }
  }
  TensorDesc td = op.GetOutputDesc("y");
  DataType inputDtype = inputTensorDesc.GetDataType();
  Shape outputShape(dimVector);
  td.SetShape(outputShape);
  td.SetDataType(inputDtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

static void InferHWAvgpool(int64_t kernel,int64_t stride, vector<int64_t>& output, vector<int64_t>& input,
                           int64_t& ori_input) {
    int64_t first_start = 0;
    int64_t second_start = 0;
    int64_t first_end = 0;
    int64_t second_end = 0;
    int64_t start = 0;
    int64_t end = 0;
    first_start = output[0] * stride;
    second_start = output[1] * stride;
    first_end = std::min(first_start + kernel, ori_input);
    second_end = std::min(second_start + kernel, ori_input);
    start = std::max(first_start, int64_t(0));
    end = second_end - 1;
    input = {start, end};
}

IMPLEMT_INFER_DATA_SLICE(AvgPool, AvgPoolInferDataSlice){
  auto inputTensorDesc = op.GetInputDesc("x");
  auto shape = inputTensorDesc.GetShape();
  std::vector<int64_t> dims_input = shape.GetDims();

  std::vector<int64_t> ksizeList;
  std::vector<int64_t> stridesList;
  std::string dataFormat;
  std::string paddingMode;
  op.GetAttr("ksize", ksizeList);
  op.GetAttr("strides", stridesList);
  op.GetAttr("data_format", dataFormat);
  op.GetAttr("padding", paddingMode);

  int64_t inputH = 0;
  int64_t inputW = 0;
  int64_t windowH = 0;
  int64_t windowW = 0;
  int64_t strideH = 0;
  int64_t strideW = 0;
  int64_t dilationH = 0;

  if (dataFormat == "NHWC") {
    inputH = dims_input[1];
    inputW = dims_input[2];
    windowH = ksizeList[1];
    windowW = ksizeList[2];
    strideH = stridesList[1];
    strideW = stridesList[2];
  } else if(dataFormat == "NCHW") {
    inputH = dims_input[2];
    inputW = dims_input[3];
    windowH = ksizeList[2];
    windowW = ksizeList[3];
    strideH = stridesList[2];
    strideW = stridesList[3];
  }

  if (dataFormat == "NHWC" && ksizeList[0] == inputH && ksizeList[1] == inputW) {
    return NO_OVERLAP_DIM;
  }
  if (dataFormat == "NCHW" && ksizeList[0] == inputH && ksizeList[1] == inputW) {
    return NO_OVERLAP_DIM;
  }
  if (paddingMode == "SAME") {
    return NO_OVERLAP_DIM;
  }

  vector<vector<int64_t>> y_data_slice = {{}, {}, {}, {}, {}};
  vector<vector<int64_t>> x_data_slice = {{}, {}, {}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_out = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_in = op_desc->MutableInputDesc("x");
  if (!ge::AttrUtils::GetListListInt(tensor_desc_out, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op.GetName().c_str(), "no data slice, use default as {{}, {}, {}, {}, {}}");
    return GRAPH_FAILED;
  }

  for(unsigned i = 0; i < y_data_slice.size(); i++) {
    if (y_data_slice[i].size() > 0) {
      if (i == 0) {
        return NO_OVERLAP_DIM;
      } else if (i == 1 or i == 3 or i == 4){
        return NOT_SUPPORT_SLICE;
      } else if (i == 2) {
        vector<int64_t> input_h;
        InferHWAvgpool(windowH, strideH, y_data_slice[i], input_h, inputH);
        x_data_slice[i] = input_h;
      }
    }
  }

  for(unsigned i = 0; i < x_data_slice.size(); i++) {
    if (x_data_slice[i].size() > 0) {
      if(!AttrUtils::SetListListInt(tensor_desc_in, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
        return GRAPH_FAILED;
      }
      return GRAPH_SUCCESS;
    }
    return NO_OVERLAP_DIM;
  }

  return NO_OVERLAP_DIM;
}

INFER_FUNC_REG(AvgPool, AvgPoolInferShape);
VERIFY_FUNC_REG(AvgPool, AvgPoolVerify);
INFER_DATA_SLICE_FUNC_REG(AvgPool, AvgPoolInferDataSlice);
// ----------------AvgPool-------------------

// -------------------AvgPoolV2--------------------
IMPLEMT_VERIFIER(AvgPoolV2, AvgPoolV2Verify) {
  const size_t DIM_SIZE4 = 4;
  auto inputTensorDesc = op.GetInputDesc("x");
  auto shape = inputTensorDesc.GetShape();

  // get input kszie
  std::vector<int32_t> ksizeList;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksizeList)) {
    OpsGetAttrErrReport(op.GetName(), "ksize");
    OP_LOGE(op.GetName().c_str(), "Failed to get attr ksize");
    return GRAPH_FAILED;
  }
  if (ksizeList.size() != DIM_SIZE4) {
    AttrSizeErrReport("ksize", op.GetName(), ConcatString(ksizeList.size()), "4");
    OP_LOGE(op.GetName().c_str(), "Size[%d] of attr ksize is invalid.", ksizeList.size());
    return GRAPH_FAILED;
  }

  // get input strides
  std::vector<int32_t> stridesList;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stridesList)) {
    OpsGetAttrErrReport(op.GetName(), "strides");
    OP_LOGE(op.GetName().c_str(), "Failed to get attr strides");
    return GRAPH_FAILED;
  }
  if (stridesList.size() != DIM_SIZE4) {
    AttrSizeErrReport("strides", op.GetName(), ConcatString(stridesList.size()), "4");
    OP_LOGE(op.GetName().c_str(), "Size[%d] of attr strides is invalid.", stridesList.size());
    return GRAPH_FAILED;
  }

  // get input paddingMode
  std::string paddingMode;
  if (GRAPH_SUCCESS != op.GetAttr("padding_mode", paddingMode)) {
    OpsGetAttrErrReport(op.GetName(), "padding_mode");
    OP_LOGE(op.GetName().c_str(), "Failed to get attr padding_mode");
    return GRAPH_FAILED;
  }
  if (paddingMode != "SAME" && paddingMode != "VALID" && paddingMode != "CALCULATED") {
    OpsAttrValueErrReport(op.GetName(), "padding_mode", "SAME, VALID and CALCULATED", paddingMode);
    OP_LOGE(op.GetName().c_str(), "padding_mode[%s] is invalid.", paddingMode.c_str());
    return GRAPH_FAILED;
  }

  // get input pads
  std::vector<int32_t> padVec;
  if (GRAPH_SUCCESS != op.GetAttr("pads", padVec)) {
    OpsGetAttrErrReport(op.GetName(), "pads");
    OP_LOGE(op.GetName().c_str(), "Failed to get attr pads");
    return GRAPH_FAILED;
  }
  if (padVec.size() != DIM_SIZE4) {
    AttrSizeErrReport("pads", op.GetName(), ConcatString(padVec.size()), "4");
    OP_LOGE(op.GetName().c_str(), "Size[%d] of attr pads is invalid.", padVec.size());
    return GRAPH_FAILED;
  }

  // get input data_format
  std::string dataFormat;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
    OpsGetAttrErrReport(op.GetName(), "data_format");
    OP_LOGE(op.GetName().c_str(), "Failed to get attr data_format");
    return GRAPH_FAILED;
  }
  if (dataFormat != "NHWC" && dataFormat != "NCHW" && dataFormat != "NC1HWC0") {
    OpsAttrValueErrReport(op.GetName(), "data_format", "NHWC, NCHW and NC1HWC0", dataFormat);
    OP_LOGE(op.GetName().c_str(), "data_format[%s] is invalid.", dataFormat.c_str());
    return GRAPH_FAILED;
  }
  if (dataFormat == "NHWC") {
    if (ksizeList[0] != 1 || ksizeList[3] != 1) {
      OpsAttrValueErrReport(op.GetName(), "ksize", "1", ConcatString(ksizeList[0], ",", ksizeList[3]));
      OP_LOGE(op.GetName().c_str(), "ksize[%d,%d] of NC is invalid", ksizeList[0], ksizeList[3]);
      return GRAPH_FAILED;
    }
    if (stridesList[0] != 1 || stridesList[3] != 1) {
      OpsAttrValueErrReport(op.GetName(), "strides", "1", ConcatString(stridesList[0], ",", stridesList[3]));
      OP_LOGE(op.GetName().c_str(), "strides[%d,%d] of NC is invalid", stridesList[0], stridesList[3]);
      return GRAPH_FAILED;
    }
  }
  if (dataFormat == "NCHW" || dataFormat == "NC1HWC0") {
    if (ksizeList[0] != 1 || ksizeList[1] != 1) {
      OpsAttrValueErrReport(op.GetName(), "ksize", "1", ConcatString(ksizeList[0], ",", ksizeList[1]));
      OP_LOGE(op.GetName().c_str(), "ksize[%d,%d] of NC is invalid", ksizeList[0], ksizeList[1]);
      return GRAPH_FAILED;
    }
    if (stridesList[0] != 1 || stridesList[1] != 1) {
      OpsAttrValueErrReport(op.GetName(), "strides", "1", ConcatString(stridesList[0], ",", stridesList[1]));
      OP_LOGE(op.GetName().c_str(), "strides[%d,%d] of NC is invalid", stridesList[0], stridesList[1]);
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AvgPoolV2InferShape) {
  const size_t DIM_SIZE1 = 1;
  const size_t DIM_SIZE2 = 2;
  const size_t DIM_SIZE3 = 3;
  auto inputTensorDesc = op.GetInputDesc("x");
  auto shape = inputTensorDesc.GetShape();
  Format input_format = inputTensorDesc.GetFormat();

  // get input kszie
  std::vector<int32_t> ksizeList;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksizeList)) {
    OpsGetAttrErrReport(op.GetName(), "ksize");
    OP_LOGE(op.GetName().c_str(), "Failed to get attr ksize");
    return GRAPH_FAILED;
  }

  // get input strides
  std::vector<int32_t> stridesList;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stridesList)) {
    OpsGetAttrErrReport(op.GetName(), "strides");
    OP_LOGE(op.GetName().c_str(), "Failed to get attr strides");
    return GRAPH_FAILED;
  }

  // get input data_format
  std::string dataFormat;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
    OpsGetAttrErrReport(op.GetName(), "data_format");
    OP_LOGE(op.GetName().c_str(), "Failed to get attr data_format");
    return GRAPH_FAILED;
  }

  // get input paddingMode
  std::string paddingMode;
  if (GRAPH_SUCCESS != op.GetAttr("padding_mode", paddingMode)) {
    OpsGetAttrErrReport(op.GetName(), "padding_mode");
    OP_LOGE(op.GetName().c_str(), "Failed to get attr padding_mode");
    return GRAPH_FAILED;
  }

  // get input strides
  std::vector<int32_t> padVec;
  if (GRAPH_SUCCESS != op.GetAttr("pads", padVec)) {
    OpsGetAttrErrReport(op.GetName(), "pads");
    OP_LOGE(op.GetName().c_str(), "Failed to get attr pads");
    return GRAPH_FAILED;
  }

  // get input global_padding
  bool globalPooling;
  if (GRAPH_SUCCESS != op.GetAttr("global_pooling", globalPooling)) {
    OpsGetAttrErrReport(op.GetName(), "global_pooling");
    OP_LOGE(op.GetName().c_str(), "Failed to get attr global_pooling");
    return GRAPH_FAILED;
  }

  // get input ceilMode
  bool ceilMode;
  if (GRAPH_SUCCESS != op.GetAttr("ceil_mode", ceilMode)) {
    OpsGetAttrErrReport(op.GetName(), "ceil_mode");
    OP_LOGE(op.GetName().c_str(), "Failed to get attr ceil_mode");
    return GRAPH_FAILED;
  }

  // input format mast equals to data_format
  if ((input_format == FORMAT_NCHW && dataFormat != "NCHW") || (input_format == FORMAT_NHWC && dataFormat != "NHWC")) {
    InferShapeOtherErrReport(op.GetName(), "Input format and dataFormat is not same.");
    OP_LOGE(op.GetName().c_str(), "Input format and dataFormat is not same.");
    return GRAPH_FAILED;
  }

  // INFER
  std::vector<int64_t> dims_input = shape.GetDims();
  // set output shape
  std::vector<int64_t> dimVector;
  int64_t window_h, window_w;
  if (FORMAT_NHWC == input_format) {
    if (globalPooling) {
      window_h = dims_input[1];
      window_w = dims_input[2];
      stridesList[1] = dims_input[1];
      stridesList[2] = dims_input[2];
      padVec = {0, 0, 0, 0};
    } else {
      window_h = (int64_t)ksizeList[1];
      window_w = (int64_t)ksizeList[2];
    }
  } else {
    if (globalPooling) {
      window_h = dims_input[2];
      window_w = dims_input[3];
      stridesList[2] = dims_input[2];
      stridesList[3] = dims_input[3];
      padVec = {0, 0, 0, 0};
    } else {
      window_h = (int64_t)ksizeList[2];
      window_w = (int64_t)ksizeList[3];
    }
  }
  if (FORMAT_NHWC == input_format) {
    if (paddingMode == "SAME") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if (DIM_SIZE1 == i || DIM_SIZE2 == i) {
          int64_t dims = (dims_input[i] + stridesList[i] - 1) / stridesList[i];
          dimVector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    } else if (paddingMode == "VALID") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if (DIM_SIZE1 == i) {
          int64_t dims = (dims_input[i] - window_h + 1 + stridesList[i] - 1) / stridesList[i];
          dimVector.push_back(dims);
        } else if (DIM_SIZE2 == i) {
          int64_t dims = (dims_input[i] - window_w + 1 + stridesList[i] - 1) / stridesList[i];
          dimVector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    } else {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if (ceilMode) {
          if (DIM_SIZE1 == i) {
            int64_t dims = (dims_input[i] - window_h + padVec[0] + padVec[1] + stridesList[i] - 1) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else if (DIM_SIZE2 == i) {
            int64_t dims = (dims_input[i] - window_w + padVec[2] + padVec[3] + stridesList[i] - 1) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else {
            int64_t dims = dims_input[i];
            dimVector.push_back(dims);
          }
        } else {
          if (DIM_SIZE1 == i) {
            int64_t dims = (dims_input[i] - window_h + padVec[0] + padVec[1]) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else if (DIM_SIZE2 == i) {
            int64_t dims = (dims_input[i] - window_w + padVec[2] + padVec[3]) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else {
            int64_t dims = dims_input[i];
            dimVector.push_back(dims);
          }
        }
      }
    }
  } else {
    // NCHW
    if (paddingMode == "SAME") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if (DIM_SIZE2 == i || DIM_SIZE3 == i) {
          int64_t dims = (dims_input[i] + stridesList[i] - 1) / stridesList[i];
          dimVector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    } else if (paddingMode == "VALID") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if (DIM_SIZE2 == i) {
          int64_t dims = (dims_input[i] - window_h + 1 + stridesList[i] - 1) / stridesList[i];
          dimVector.push_back(dims);
        } else if (DIM_SIZE3 == i) {
          int64_t dims = (dims_input[i] - window_w + 1 + stridesList[i] - 1) / stridesList[i];
          dimVector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    } else {
      if (ceilMode) {
        for (size_t i = 0; i < dims_input.size(); i++) {
          if (DIM_SIZE2 == i) {
            int64_t dims = (dims_input[i] - window_h + padVec[0] + padVec[1] + stridesList[i] - 1) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else if (DIM_SIZE3 == i) {
            int64_t dims = (dims_input[i] - window_w + padVec[2] + padVec[3] + stridesList[i] - 1) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else {
            int64_t dims = dims_input[i];
            dimVector.push_back(dims);
          }
        }
      } else {
        for (size_t i = 0; i < dims_input.size(); i++) {
          if (DIM_SIZE2 == i) {
            int64_t dims = (dims_input[i] - window_h + padVec[0] + padVec[1]) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else if (DIM_SIZE3 == i) {
            int64_t dims = (dims_input[i] - window_w + padVec[2] + padVec[3]) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else {
            int64_t dims = dims_input[i];
            dimVector.push_back(dims);
          }
        }
      }
    }
  }
  TensorDesc td = op.GetOutputDesc("y");
  DataType inputDtype = inputTensorDesc.GetDataType();
  Shape outputShape(dimVector);
  td.SetShape(outputShape);
  td.SetDataType(inputDtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AvgPoolV2, AvgPoolV2InferShape);
VERIFY_FUNC_REG(AvgPoolV2, AvgPoolV2Verify);
// ----------------AvgPoolV2 End-------------------

// ----------------AvgPool3D-------------------
IMPLEMT_VERIFIER(AvgPool3D, AvgPool3DVerify) {
  auto ksize = op.get_attr_ksize();
  auto strides = op.get_attr_strides();
  auto x_shape = op.get_input_desc_x().GetShape().GetDims();
  bool invalid_param = (x_shape.size() != 5 || ksize.size() > 5 || strides.size() > 5);
  if (invalid_param) {
    OP_LOGE(op.GetName().c_str(), "AvgPool3D check x_shape or ksize or strides size is invalid!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(AvgPool3D, AvgPool3DInferShape) {
  auto input_tensor_desc = op.GetInputDesc("x");
  auto shape = input_tensor_desc.GetShape();
  Format input_format = input_tensor_desc.GetFormat();
  std::vector<int64_t> dims_input = shape.GetDims();
  int32_t id = 0;
  int32_t ih = 0;
  int32_t iw = 0;
  int32_t in = 0;
  int32_t ic = 0;
  int32_t kd = 0;
  int32_t kh = 0;
  int32_t kw = 0;
  int32_t strd = 0;
  int32_t strh = 0;
  int32_t strw = 0;

  // get input dims
  if (input_format == FORMAT_NDHWC) {
    in = dims_input[0];
    id = dims_input[1];
    ih = dims_input[2];
    iw = dims_input[3];
    ic = dims_input[4];
  } else if (input_format == FORMAT_NCDHW) {
    in = dims_input[0];
    ic = dims_input[1];
    id = dims_input[2];
    ih = dims_input[3];
    iw = dims_input[4];
  } else if (input_format == FORMAT_DHWCN) {
    id = dims_input[0];
    ih = dims_input[1];
    iw = dims_input[2];
    ic = dims_input[3];
    in = dims_input[4];
  } else {
    map<string, string> err_map;
    err_map["param_name"] = "input_format";
    err_map["op_name"] = "AvgPool3D";
    err_map["excepted_value"] = "NDHWC or NCDHW or DHWCN";
    err_map["input_value"] = input_format;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  if (!GetStridesAndKSize(op, input_format, strd, strh, strw, kd, kh, kw)) {
    OP_LOGE(op.GetName().c_str(), "Failed to get attr strides or ksize in AvgPool3D.");
    return GRAPH_FAILED;
  }
  if (strd == 0 || strh == 0 || strw == 0 || kd == 0 || kh == 0 || kw == 0) {
    OP_LOGE(op.GetName().c_str(), "Strides or ksize invalid.");
    return GRAPH_FAILED;
  }

  std::vector<int32_t> pad_vec;
  if (GetPadsByPadding(op, id, ih, iw, kd, kh, kw, strd, strh, strw, pad_vec)) {
    op.SetAttr("pads",pad_vec);
  } else if (op.GetAttr("pads", pad_vec) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get padding or pads failed.");
    return GRAPH_FAILED;
  }
  if (pad_vec.size() != 6) {
    OP_LOGE(op.GetName().c_str(), "pads size must be 6.");
    return GRAPH_FAILED;
  }

  std::string pad_str;
  if (op.GetAttr("padding", pad_str) == GRAPH_SUCCESS) {
    if (pad_str.compare("SAME") == 0) {
      // tensorflow same padding mode, set count_include_pad = false
      op.SetAttr("count_include_pad", false);
    }
  }

  bool ceil_mode = false;
  op.GetAttr("ceil_mode", ceil_mode);

  // set output shape
  std::vector<int64_t> dim_vector;
  int64_t outd = 0;
  int64_t outh = 0;
  int64_t outw = 0;
  if (ceil_mode) {
    outd = (id + pad_vec[0] + pad_vec[1] - kd + strd - 1) / strd + 1;
    outh = (ih + pad_vec[2] + pad_vec[3] - kh + strh - 1) / strh + 1;
    outw = (iw + pad_vec[4] + pad_vec[5] - kw + strw - 1) / strw + 1;
    if ((outd - 1) * strd >= id + pad_vec[0]) {
      outd--;
    }
    if ((outh - 1) * strh >= ih + pad_vec[2]) {
      outh--;
    }
    if ((outw - 1) * strw >= iw + pad_vec[4]) {
      outw--;
    }

  } else {
    outd = (id + pad_vec[0] + pad_vec[1] - kd) / strd + 1;
    outh = (ih + pad_vec[2] + pad_vec[3] - kh) / strh + 1;
    outw = (iw + pad_vec[4] + pad_vec[5] - kw) / strw + 1;
  }
  if (input_format == FORMAT_NDHWC) {
    dim_vector.push_back(in);
    dim_vector.push_back(outd);
    dim_vector.push_back(outh);
    dim_vector.push_back(outw);
    dim_vector.push_back(ic);
  } else if (input_format == FORMAT_NCDHW) {
    dim_vector.push_back(in);
    dim_vector.push_back(ic);
    dim_vector.push_back(outd);
    dim_vector.push_back(outh);
    dim_vector.push_back(outw);
  } else if (input_format == FORMAT_DHWCN) {
    dim_vector.push_back(outd);
    dim_vector.push_back(outh);
    dim_vector.push_back(outw);
    dim_vector.push_back(ic);
    dim_vector.push_back(in);
  }

  TensorDesc tensor_out = op.GetOutputDesc("y");
  DataType input_dtype = input_tensor_desc.GetDataType();
  Shape output_shape(dim_vector);
  tensor_out.SetShape(output_shape);
  tensor_out.SetDataType(input_dtype);
  if (op.UpdateOutputDesc("y", tensor_out) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    OpsOPUpdateErrReport(op.GetName(), "y");
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AvgPool3D, AvgPool3DInferShape);
VERIFY_FUNC_REG(AvgPool3D, AvgPool3DVerify);
// ----------------AvgPool3D-------------------

// ----------------AvgPool3DD-------------------
IMPLEMT_VERIFIER(AvgPool3DD, AvgPool3DDVerify) {
  auto ksize = op.get_attr_ksize();
  auto strides = op.get_attr_strides();
  auto x_shape = op.get_input_desc_x().GetShape().GetDims();
  bool invalid_param = (x_shape.size() != 5 || ksize.size() > 5 || strides.size() > 5);
  if (invalid_param) {
    OP_LOGE(op.GetName().c_str(), "AvgPool3DD check x_shap or ksize or strides size is invalid!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(AvgPool3DD, AvgPool3DDInferShape) {
  auto input_tensor_desc = op.GetInputDesc("x");
  auto shape = input_tensor_desc.GetShape();
  Format input_format = input_tensor_desc.GetFormat();
  std::vector<int64_t> dims_input = shape.GetDims();
  int32_t id = 0;
  int32_t ih = 0;
  int32_t iw = 0;
  int32_t in = 0;
  int32_t ic = 0;
  int32_t kd = 0;
  int32_t kh = 0;
  int32_t kw = 0;
  int32_t strd = 0;
  int32_t strh = 0;
  int32_t strw = 0;

  // get input dims
  if (input_format == FORMAT_NDHWC) {
    in = dims_input[0];
    id = dims_input[1];
    ih = dims_input[2];
    iw = dims_input[3];
    ic = dims_input[4];
  } else if (input_format == FORMAT_NCDHW) {
    in = dims_input[0];
    ic = dims_input[1];
    id = dims_input[2];
    ih = dims_input[3];
    iw = dims_input[4];
  } else if (input_format == FORMAT_DHWCN) {
    id = dims_input[0];
    ih = dims_input[1];
    iw = dims_input[2];
    ic = dims_input[3];
    in = dims_input[4];
  } else {
    map<string, string> err_map;
    err_map["param_name"] = "input_format";
    err_map["op_name"] = "AvgPool3DD";
    err_map["excepted_value"] = "NDHWC or NCDHW or DHWCN";
    err_map["input_value"] = input_format;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  if (!GetStridesAndKSize(op, input_format, strd, strh, strw, kd, kh, kw)) {
    OP_LOGE(op.GetName().c_str(), "Failed to get attr strides or ksize in AvgPool3DD.");
    return GRAPH_FAILED;
  }
  if (strd == 0 || strh == 0 || strw == 0 || kd == 0 || kh == 0 || kw == 0) {
    OP_LOGE(op.GetName().c_str(), "Strides or ksize invalid.");
    return GRAPH_FAILED;
  }

  std::vector<int32_t> pad_vec;
  if (GetPadsByPadding(op, id, ih, iw, kd, kh, kw, strd, strh, strw, pad_vec)) {
    op.SetAttr("pads",pad_vec);
  } else if (op.GetAttr("pads", pad_vec) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get padding or pads failed.");
    return GRAPH_FAILED;
  }
  if (pad_vec.size() != 6) {
    OP_LOGE(op.GetName().c_str(), "pads size must be 6.");
    return GRAPH_FAILED;
  }

  std::string pad_str;
  if (op.GetAttr("padding", pad_str) == GRAPH_SUCCESS) {
    if (pad_str.compare("SAME") == 0) {
      // tensorflow same padding mode, set count_include_pad = false
      op.SetAttr("count_include_pad", false);
    }
  }

  bool ceil_mode = false;
  op.GetAttr("ceil_mode", ceil_mode);

  // set output shape
  std::vector<int64_t> dim_vector;
  int64_t outd = 0;
  int64_t outh = 0;
  int64_t outw = 0;
  if (ceil_mode) {
    outd = (id + pad_vec[0] + pad_vec[1] - kd + strd - 1) / strd + 1;
    outh = (ih + pad_vec[2] + pad_vec[3] - kh + strh - 1) / strh + 1;
    outw = (iw + pad_vec[4] + pad_vec[5] - kw + strw - 1) / strw + 1;
    if ((outd - 1) * strd >= id + pad_vec[0]) {
      outd--;
    }
    if ((outh - 1) * strh >= ih + pad_vec[2]) {
      outh--;
    }
    if ((outw - 1) * strw >= iw + pad_vec[4]) {
      outw--;
    }

  } else {
    outd = (id + pad_vec[0] + pad_vec[1] - kd) / strd + 1;
    outh = (ih + pad_vec[2] + pad_vec[3] - kh) / strh + 1;
    outw = (iw + pad_vec[4] + pad_vec[5] - kw) / strw + 1;
  }
  if (input_format == FORMAT_NDHWC) {
    dim_vector.push_back(in);
    dim_vector.push_back(outd);
    dim_vector.push_back(outh);
    dim_vector.push_back(outw);
    dim_vector.push_back(ic);
  } else if (input_format == FORMAT_NCDHW) {
    dim_vector.push_back(in);
    dim_vector.push_back(ic);
    dim_vector.push_back(outd);
    dim_vector.push_back(outh);
    dim_vector.push_back(outw);
  } else if (input_format == FORMAT_DHWCN) {
    dim_vector.push_back(outd);
    dim_vector.push_back(outh);
    dim_vector.push_back(outw);
    dim_vector.push_back(ic);
    dim_vector.push_back(in);
  }

  TensorDesc tensor_out = op.GetOutputDesc("y");
  DataType input_dtype = input_tensor_desc.GetDataType();
  Shape output_shape(dim_vector);
  tensor_out.SetShape(output_shape);
  tensor_out.SetDataType(input_dtype);
  if (op.UpdateOutputDesc("y", tensor_out) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    OpsOPUpdateErrReport(op.GetName(), "y");
  }
  return GRAPH_SUCCESS;
}

static void InferHDAvgPool3DD(int32_t kernel, int32_t pad, int32_t stride, int32_t origin_input,
                              const vector<int64_t>& output_slice, vector<int64_t>& input_slice) {
  // size of output_slice is greater than 1
  int64_t slice_start = output_slice[0] * stride - pad;
  if (slice_start < 0) {
    slice_start = 0;
  }

  int64_t slice_end = output_slice[1] * stride - pad + (kernel - 1);
  if (slice_end >= origin_input) {
    slice_end = origin_input - 1;
  }
  input_slice = {slice_start, slice_end};
}
/*!
 * @brief provide AvgPool3DD operator slice data
 * @param AvgPool3DD Operator type.
 * @param AvgPool3DDInferDataSlice slice data function
 * @return Status The processing flow result.
 */
IMPLEMT_INFER_DATA_SLICE(AvgPool3DD, AvgPool3DDInferDataSlice) {
  OP_LOGI(op.GetName().c_str(), "Enter AvgPool3DD InferDataSlice");

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_in_ptr = op_desc->MutableInputDesc("x");
  GeTensorDescPtr tensor_out_ptr = op_desc->MutableOutputDesc("y");
  auto shape_in = tensor_in_ptr->GetShape();

  // get input data_format
  std::string data_format;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", data_format)) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr data_format failed!");
    return GRAPH_FAILED;
  }

  auto x_format = tensor_in_ptr->GetOriginFormat();
  const std::map<std::string, Format> format_map{{"NDHWC", FORMAT_NDHWC}, {"NCDHW", FORMAT_NCDHW}};

  // Set ori_format as data_format if input ori_format is the default value ND.
  if (x_format == FORMAT_ND) {
    tensor_in_ptr->SetOriginFormat(format_map.at(data_format));
    tensor_in_ptr->SetFormat(format_map.at(data_format));
  }

  x_format = tensor_in_ptr->GetOriginFormat();
  if (x_format != FORMAT_NDHWC && x_format != FORMAT_NCDHW) {
    OP_LOGE(op.GetName().c_str(), "Input x format only support NDHWC or NCDHW");
    return GRAPH_FAILED;
  }

  std::map<char, int> idx_map{{'N', 0}, {'D', 1}, {'H', 2}, {'W', 3}, {'C', 4}};
  if (x_format == FORMAT_NCDHW) {
    idx_map = {{'N', 0}, {'C', 1}, {'D', 2}, {'H', 3}, {'W', 4}};
  }

  int32_t input_d = shape_in.GetDim(idx_map['D']);
  int32_t input_h = shape_in.GetDim(idx_map['H']);
  int32_t input_w = shape_in.GetDim(idx_map['W']);
  int32_t filter_d = 0;
  int32_t filter_h = 0;
  int32_t filter_w = 0;
  int32_t stride_d = 0;
  int32_t stride_h = 0;
  int32_t stride_w = 0;

  if (GetStridesAndKSize(op, x_format, stride_d, stride_h, stride_w, filter_d, filter_h, filter_w)) {
    OP_LOGE(op.GetName().c_str(), "Failed to get attr strides or ksize in AvgPool3D.");
    return GRAPH_FAILED;
  }

  // construct pads attr
  if (!Construct3DPadsByPadding("AvgPool3DD", op, input_d, input_h, input_w, filter_d, filter_h, filter_w, stride_d,
                                stride_h, stride_w)) {
    OP_LOGE(op.GetName().c_str(), "Failed to construct pads in AvgPool3D.");
    return GRAPH_FAILED;
  }

  std::vector<int32_t> pad_list;
  if (op.GetAttr("pads", pad_list) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Failed to get pads!");
    return GRAPH_FAILED;
  }
  if (pad_list.size() < 3) {
    OP_LOGE(op.GetName().c_str(), "The pad_list do not have enough elements!");
    return GRAPH_FAILED;
  }
  int32_t pad_d = pad_list[0];
  int32_t pad_h = pad_list[2];

  vector<vector<int64_t>> y_data_slice = {{}, {}, {}, {}, {}, {}};
  vector<vector<int64_t>> x_data_slice = {{}, {}, {}, {}, {}, {}};
  if (!AttrUtils::GetListListInt(tensor_out_ptr, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op.GetName().c_str(), "No data slice, not need infer input");
    return GRAPH_FAILED;
  }

  bool need_infer = false;
  bool have_slice = false;
  for (unsigned idx = 0; idx < y_data_slice.size(); idx++) {
    if (y_data_slice[idx].size() > 1) {
      have_slice = true;
      if (idx == 1) {
        need_infer = true;
        vector<int64_t> slice_data_d;
        InferHDAvgPool3DD(filter_d, pad_d, stride_d, input_d, y_data_slice[idx], slice_data_d);
        OP_LOGD(op.GetName().c_str(), "AvgPool3DD d axis slice ori_scope is [%d, %d], calced output scope is [%d, %d]",
                slice_data_d[0], slice_data_d[1], y_data_slice[idx][0], y_data_slice[idx][1]);
        x_data_slice[idx] = slice_data_d;
      } else if (idx == 3) {
        need_infer = true;
        vector<int64_t> slice_data_h;
        InferHDAvgPool3DD(filter_h, pad_h, stride_h, input_h, y_data_slice[idx], slice_data_h);
        OP_LOGD(op.GetName().c_str(), "AvgPool3DD h axis slice ori_scope is [%d, %d], calced output scope is [%d, %d]",
                slice_data_h[0], slice_data_h[1], y_data_slice[idx][0], y_data_slice[idx][1]);
        x_data_slice[idx] = slice_data_h;
      }
    }
  }

  if (!have_slice) {
    return GRAPH_FAILED;
  }
  if (!need_infer) {
    return NO_OVERLAP_DIM;
  } else {
    if (!AttrUtils::SetListListInt(tensor_in_ptr, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }
  OP_LOGI(op.GetName().c_str(), "Calc AvgPool3DD InferDataSlice end!");
}

INFER_FUNC_REG(AvgPool3DD, AvgPool3DDInferShape);
VERIFY_FUNC_REG(AvgPool3DD, AvgPool3DDVerify);
INFER_DATA_SLICE_FUNC_REG(AvgPool3DD, AvgPool3DDInferDataSlice);
// ----------------AvgPool3D-------------------

// ----------------AvgPool3DGradD-------------------
IMPLEMT_VERIFIER(AvgPool3DGradD, AvgPool3DGradDVerify) {
  std::vector<int64_t> orig_input_shape = GetAttrValue(op, "orig_input_shape");
  if (orig_input_shape.size() != kAvgpool3DGradOriShapeDim) {
    OP_LOGE(op.GetName().c_str(), "orig_input_shape length is not %zu.", kAvgpool3DGradOriShapeDim);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> ksize = GetAttrValue(op, "ksize");
  if (ksize.size() != kAvgpool3DGradKsizeDim) {
    OP_LOGE(op.GetName().c_str(), "ksize size is not %zu.", kAvgpool3DGradKsizeDim);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> strides = GetAttrValue(op, "strides");
  if (strides.size() != kAvgpool3DGradStridesDim) {
    OP_LOGE(op.GetName().c_str(), "strides size is not %zu.", kAvgpool3DGradStridesDim);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> pads;
  if (op.GetAttr("pads", pads) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get pads failed!");
    return GRAPH_FAILED;
  }
  if (pads.size() != kAvgpool3DGradPadsDim) {
    OP_LOGE(op.GetName().c_str(), "Attr pads size is not %zu.", kAvgpool3DGradPadsDim);
    return GRAPH_FAILED;
  }

  std::string data_format;
  if (op.GetAttr("data_format", data_format) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get attr data_format failed.");
    return GRAPH_FAILED;
  }
  if (data_format != "NDHWC" && data_format != "NCDHW") {
    OP_LOGE(op.GetName().c_str(), "Attr data_format(%s) only support NDHWC.", data_format.c_str());
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(AvgPool3DGradD, AvgPool3DGradDInferShape) {
  std::vector<int64_t> orig_input_shape = GetAttrValue(op, "orig_input_shape");
  if (orig_input_shape.size() != kAvgpool3DGradOriShapeDim) {
    OP_LOGE(op.GetName().c_str(), "Get orig_input_shape failed!");
    return GRAPH_FAILED;
  }
  TensorDesc grads_desc = op.GetInputDesc("grads");

  std::vector<int64_t> ksize = GetAttrValue(op, "ksize");
  if (!CheckListEmpty(op.GetName(), ksize, "ksize")) {
    return GRAPH_FAILED;
  }

  std::vector<int64_t> strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), strides, "strides")) {
    return GRAPH_FAILED;
  }

  std::vector<int64_t> pads;
  if (op.GetAttr("pads", pads) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get pads failed!");
    return GRAPH_FAILED;
  }

  std::string data_format;
  if (op.GetAttr("data_format", data_format) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get data_format failed!");
    return GRAPH_FAILED;
  }
  if (data_format != "NDHWC" && data_format != "NCDHW") {
    OP_LOGE(op.GetName().c_str(), "Attr data_format(%s) only support NDHWC or NCDHW.",
            data_format.c_str());
    return GRAPH_FAILED;
  }

  TensorDesc mean_desc = op.GetInputDesc("multiplier");
  if (data_format == "NDHWC") {
    mean_desc.SetOriginFormat(FORMAT_NDHWC);
    mean_desc.SetFormat(FORMAT_NDHWC);
  } else if (data_format == "NCDHW") {
    mean_desc.SetOriginFormat(FORMAT_NCDHW);
    mean_desc.SetFormat(FORMAT_NCDHW);
  } else {
    OP_LOGE(op.GetName().c_str(), "Attr data_format(%s) only support NDHWC or NCDHW", data_format.c_str());
    return GRAPH_FAILED;
  }

  Shape grads_shape = grads_desc.GetShape();
  DataType grads_dtype = grads_desc.GetDataType();
  mean_desc.SetShape(grads_shape);
  mean_desc.SetOriginShape(grads_shape);
  mean_desc.SetDataType(grads_dtype);
  if (op.UpdateInputDesc("multiplier", mean_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Fail to update input multiplier!");
    return GRAPH_FAILED;
  }

  string::size_type c_position = data_format.find("C");
  if (c_position == std::string::npos) {
    return GRAPH_FAILED;
  }
  vector<int64_t> kernel_shape = {ksize.at(2), ksize.at(0), ksize.at(1), 1,
                                  orig_input_shape.at(c_position)};

  TensorDesc filter_desc = op.GetInputDesc("filter");
  filter_desc.SetShape(Shape(kernel_shape));
  filter_desc.SetOriginShape(Shape(kernel_shape));
  filter_desc.SetOriginFormat(FORMAT_DHWCN);
  filter_desc.SetFormat(FORMAT_DHWCN);
  filter_desc.SetDataType(grads_dtype);
  if (op.UpdateInputDesc("filter", filter_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Fail to update input:filter desc!");
    return GRAPH_FAILED;
  }

  TensorDesc output_desc = op.GetOutputDesc("output");
  output_desc.SetShape(Shape(orig_input_shape));
  output_desc.SetDataType(grads_dtype);
  if (op.UpdateOutputDesc("output", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Fail to update output:output desc!");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

VERIFY_FUNC_REG(AvgPool3DGradD, AvgPool3DGradDVerify);
INFER_FUNC_REG(AvgPool3DGradD, AvgPool3DGradDInferShape);

// ----------------AvgPool3DGrad-------------------
IMPLEMT_VERIFIER(AvgPool3DGrad, AvgPool3DGradVerify) {
  Tensor orig_input_shape_tensor;
  if (op.GetInputConstData("orig_input_shape", orig_input_shape_tensor) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get constdata failed");
    return GRAPH_FAILED;
  }

  DataType dtype = op.GetInputDesc("orig_input_shape").GetDataType();
  std::vector<int64_t> orig_input_shape;
  GetConstValue(op, orig_input_shape_tensor, dtype, orig_input_shape);
  if (orig_input_shape.size() != kAvgpool3DGradOriShapeDim) {
    OP_LOGE(op.GetName().c_str(), "Input:orig_input_shape has an incorreted length");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> ksize = GetAttrValue(op, "ksize");
  if (ksize.size() != kAvgpool3DGradKsizeDim) {
    OP_LOGE(op.GetName().c_str(), "Attr:ksize has an incorreted length");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> strides = GetAttrValue(op, "strides");
  if (strides.size() != kAvgpool3DGradStridesDim) {
    OP_LOGE(op.GetName().c_str(), "Attr:strides has an incorreted length");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> pads;
  if (op.GetAttr("pads", pads) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get pads failed!");
    return GRAPH_FAILED;
  }
  if (pads.size() != kAvgpool3DGradPadsDim) {
    OP_LOGE(op.GetName().c_str(), "Attr:pads has an incorreted length");
    return GRAPH_FAILED;
  }

  std::string data_format;
  if (op.GetAttr("data_format", data_format) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get attr data_format failed.");
    return GRAPH_FAILED;
  }
  if (data_format != "NDHWC" && data_format != "NCDHW") {
    OP_LOGE(op.GetName().c_str(), "Attr data_format(%s) only support NDHWC or NCDHW.", data_format.c_str());
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

void AvgPool3dGradCalcPads(const vector<int64_t> &grads_shape,
                           const string &grads_format,
                           const vector<int64_t> &fmap_shape,
                           const string &fmap_format,
                           const vector<int64_t> &ksize_hwd,
                           const vector<int64_t> &strides_hwd,
                           vector<int64_t> &pads) {
  int64_t pads_d = 0;
  int64_t pads_h = 0;
  int64_t pads_w = 0;

  pads_d = std::max((grads_shape[grads_format.find("D")] - 1) * strides_hwd[2] +
                     ksize_hwd[2] - fmap_shape[fmap_format.find("D")], 0L);
  pads_h = std::max((grads_shape[grads_format.find("H")] - 1) * strides_hwd[0] +
                     ksize_hwd[0] - fmap_shape[fmap_format.find("H")], 0L);
  pads_w = std::max((grads_shape[grads_format.find("W")] - 1) * strides_hwd[1] +
                     ksize_hwd[1] - fmap_shape[fmap_format.find("W")], 0L);
  pads[0] = pads_d / 2;
  pads[1] = pads_d - pads[0];
  pads[2] = pads_h / 2;
  pads[3] = pads_h - pads[2];
  pads[4] = pads_w / 2;
  pads[5] = pads_w - pads[4];
}

IMPLEMT_INFERFUNC(AvgPool3DGrad, AvgPool3DGradInferShape) {
  Tensor orig_input_shape_tensor;
  if (op.GetInputConstData("orig_input_shape", orig_input_shape_tensor) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get constdata failed.");
    return GRAPH_FAILED;
  }

  DataType dtype = op.GetInputDesc("orig_input_shape").GetDataType();
  std::vector<int64_t> orig_input_shape;
  GetConstValue(op, orig_input_shape_tensor, dtype, orig_input_shape);
  DataType output_dtype = op.GetInputDesc("grads").GetDataType();
  TensorDesc output_desc = op.GetOutputDesc("output");
  output_desc.SetShape(Shape(orig_input_shape));
  output_desc.SetDataType(output_dtype);
  if (op.UpdateOutputDesc("output", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> ksize = GetAttrValue(op, "ksize");
  if (ksize.size() != kAvgpool3DGradKsizeDim) {
    OP_LOGE(op.GetName().c_str(), "Attr:ksize has an incorrected length.");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> strides = GetAttrValue(op,"strides");
  if (strides.size() != kAvgpool3DGradStridesDim) {
    OP_LOGE(op.GetName().c_str(), "Attr:strides has an incorrected length.");
    return GRAPH_FAILED;
  }

  std::string padding;
  if (op.GetAttr("padding", padding) == GRAPH_SUCCESS) {
    if (padding != "SAME" && padding != "VALID") {
      OP_LOGE(op.GetName().c_str(), "Padding pattern is incorrected, only support SAME and VALID.");
      return GRAPH_FAILED;
    }
    vector<int64_t> pads(kAvgpool3DGradPadsDim, 0);
    if (padding == "SAME") {
      AvgPool3dGradCalcPads(op.GetInputDesc("grads").GetOriginShape().GetDims(),
                            format2str[op.GetInputDesc("grads").GetFormat()],
                            orig_input_shape,
                            format2str[op.GetInputDesc("orig_input_shape").GetFormat()],
                            ksize, strides,
                            pads);
    }
    op.SetAttr("pads", pads);
  }

  std::vector<int64_t> pads;
  if (op.GetAttr("pads", pads) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get pads failed.");
    return GRAPH_FAILED;
  }
  if (pads.size() != kAvgpool3DGradPadsDim) {
    OP_LOGE(op.GetName().c_str(), "Attr pads has an incorrected length.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AvgPool3DGrad, AvgPool3DGradInferShape);
VERIFY_FUNC_REG(AvgPool3DGrad, AvgPool3DGradVerify);

// ----------------MaxPool-------------------
static void UpdateDimAndRange(const int64_t& ksize, const int64_t& strides, int64_t& dim_size,
                              std::pair<int64_t, int64_t>& dim_range) {
  if (dim_size != -1) {
    int64_t output_dim_size = (dim_size - ksize + strides) / strides;
    dim_range = std::pair<int64_t, int64_t>{output_dim_size, output_dim_size};
    dim_size = output_dim_size;
  } else {
    int64_t first_range = dim_range.first == 1 ? 1 : (dim_range.first - ksize + strides) / strides;
    int64_t second_range = dim_range.second == -1 ? -1 : (dim_range.second - ksize + strides) / strides;
    dim_range = std::pair<int64_t, int64_t>{first_range, second_range};
  }
}

IMPLEMT_INFERFUNC(MaxPool, MaxPoolInferShape) {
  auto node = NodeUtils::GetNodeFromOperator(op);
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_shape = input_desc->MutableShape();
  auto input_format = input_desc->GetFormat();
  auto input_dtype = input_desc->GetDataType();
  auto output_desc = op_info->MutableOutputDesc("y");
  output_desc->SetDataType(input_dtype);
  // get input ksize
  std::vector<int32_t> ksize;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksize)) {
    OpsGetAttrErrReport(op.GetName(), "ksize");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ksize failed!");
    return GRAPH_FAILED;
  }
  // get input strides
  std::vector<int32_t> strides;
  if (GRAPH_SUCCESS != op.GetAttr("strides", strides)) {
    OpsGetAttrErrReport(op.GetName(), "strides");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr strides failed!");
    return GRAPH_FAILED;
  }
  // get input data_format
  std::string data_format;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", data_format)) {
    OpsGetAttrErrReport(op.GetName(), "data_format");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr data_format failed!");
    return GRAPH_FAILED;
  }
  // get input padding
  std::string padding;
  if (GRAPH_SUCCESS != op.GetAttr("padding", padding)) {
    OpsGetAttrErrReport(op.GetName(), "padding");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr padding failed!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> input_dims = input_shape.GetDims();
  std::vector<std::pair<int64_t, int64_t>> input_range;

  // dynamic case, input shape is -2, output is [-1, -1, -1, -1], only support NHWC or NCHW
  if (IsUnknownRankShape(input_dims)) {
    OP_LOGW(op.GetName().c_str(), "the input os unkown rank, will set the input [-1, -1, -1, -1].");
    input_dims = {-1, -1, -1, -1};
  } else {
    input_desc->GetShapeRange(input_range);
  }
  MakeUpShapeRange(input_dims, input_range);

  // set output shape
  std::vector<int64_t> output_dims;
  std::vector<std::pair<int64_t, int64_t>> output_range;

  auto input_h_dim = input_format == FORMAT_NHWC ? 1 : 2;
  auto input_w_dim = input_format == FORMAT_NHWC ? 2 : 3;
  auto strides_h_dim = data_format == "NHWC" ? 1 : 2;
  auto strides_w_dim = data_format == "NHWC" ? 2 : 3;

  if (padding != "VALID") {
    ksize[strides_h_dim] = 1;
    ksize[strides_w_dim] = 1;
  }

  for (size_t i = 0; i < input_dims.size(); i++) {
    int64_t dim_size = input_dims[i];
    auto dim_range = input_range[i];
    if (i == input_h_dim) {
      UpdateDimAndRange(ksize[strides_h_dim], strides[strides_h_dim], dim_size, dim_range);
    } else if (i == input_w_dim) {
      UpdateDimAndRange(ksize[strides_w_dim], strides[strides_w_dim], dim_size, dim_range);
    }
    output_dims.push_back(dim_size);
    output_range.push_back(dim_range);
  }

  output_desc->SetShape(GeShape(output_dims));
  output_desc->SetShapeRange(output_range);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MaxPool, MaxPoolVerify) {
  // check ksize
  std::vector<int32_t> ksize;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksize)) {
    OpsGetAttrErrReport(op.GetName(), "ksize");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ksize failed!");
    return GRAPH_FAILED;
  }
  if (ksize.size() != 4) {
    OpsAttrValueErrReport(op.GetName(), "length of ksize", ConcatString(4), ConcatString((size_t)ksize.size()));
    OP_LOGE(op.GetName().c_str(), "The length of ksize must be equal to the length of shape!");
    return GRAPH_FAILED;
  }
  // check strides
  std::vector<int32_t> strides;
  if (GRAPH_SUCCESS != op.GetAttr("strides", strides)) {
    OpsGetAttrErrReport(op.GetName(), "strides");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr strides failed!");
    return GRAPH_FAILED;
  }
  if (strides.size() != 4) {
    OpsAttrValueErrReport(op.GetName(), "length of strides", ConcatString(4), ConcatString(strides.size()));
    OP_LOGE(op.GetName().c_str(), "The length of strides must be equal to the length of shape!");
    return GRAPH_FAILED;
  }
  // check data_format
  std::string data_format;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", data_format)) {
    OpsGetAttrErrReport(op.GetName(), "data_format");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr data_format failed!");
    return GRAPH_FAILED;
  }
  if (data_format != "NHWC" && data_format != "NCHW" && data_format != "NC1HWC0") {
    string expected_format_list = ConcatString("NHWC,NCHW,NC1HWC0");
    OpsInputFormatErrReport(op.GetName(), "data_format", expected_format_list, data_format);
    OP_LOGE(op.GetName().c_str(), "data_format only support 'NHWC','NCHW' and 'NC1HWC0'.");
    return GRAPH_FAILED;
  }
  if (data_format == "NHWC") {
    if ((ksize[0] != 1) || (ksize[3] != 1) || (strides[0] != 1) || (strides[3] != 1)) {
      OP_LOGE(op.GetName().c_str(), "Pooling across width/height and other ksize dimension should be one");
      return GRAPH_FAILED;
    }
  }
  if ((data_format == "NCHW") || (data_format == "NC1HWC0")) {
    if ((ksize[0] != 1) || (ksize[1] != 1) || (strides[0] != 1) || (strides[1] != 1)) {
      OP_LOGE(op.GetName().c_str(), "Pooling across width/height and other ksize dimension should be one");
      return GRAPH_FAILED;
    }
  }
  // check padding
  std::string padding;
  if (GRAPH_SUCCESS != op.GetAttr("padding", padding)) {
    OpsGetAttrErrReport(op.GetName(), "padding");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr padding failed!");
    return GRAPH_FAILED;
  }
  if (padding != "SAME" && padding != "VALID") {
    string expected_format_list = ConcatString("SAME,VALID");
    OpsInputFormatErrReport(op.GetName(), "padding", expected_format_list, padding);
    OP_LOGE(op.GetName().c_str(), "padding only support SAME or VALID padding mode!");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

static void InferHWMaxPool(int64_t kernel, int64_t stride, vector<int64_t>& output, vector<int64_t>& input,
                           int64_t& ori_input) {
  int64_t first_start = 0;
  int64_t second_start = 0;
  int64_t first_end = 0;
  int64_t second_end = 0;
  int64_t start = 0;
  int64_t end = 0;
  first_start = output[0] * stride;
  second_start = output[1] * stride;
  first_end = std::min(first_start + kernel, ori_input);
  second_end = std::min(second_start + kernel, ori_input);
  start = std::max(first_start, int64_t(0));
  end = second_end - 1;
  input = {start, end};
}

IMPLEMT_INFER_DATA_SLICE(MaxPool, MaxPoolInferDataSlice) {
  auto inputTensorDesc = op.GetInputDesc("x");
  auto shape = inputTensorDesc.GetShape();
  std::vector<int64_t> dims_input = shape.GetDims();

  std::vector<int64_t> ksizeList;
  std::vector<int64_t> stridesList;
  std::string dataFormat;
  std::string paddingMode;
  op.GetAttr("ksize", ksizeList);
  op.GetAttr("strides", stridesList);
  op.GetAttr("data_format", dataFormat);
  op.GetAttr("padding", paddingMode);

  int64_t inputH = 0;
  int64_t inputW = 0;
  int64_t windowH = 0;
  int64_t windowW = 0;
  int64_t strideH = 0;
  int64_t strideW = 0;
  int64_t dilationH = 0;

  if (dataFormat == "NHWC") {
    inputH = dims_input[1];
    inputW = dims_input[2];
    windowH = ksizeList[1];
    windowW = ksizeList[2];
    strideH = stridesList[1];
    strideW = stridesList[2];
  } else if (dataFormat == "NCHW") {
    inputH = dims_input[2];
    inputW = dims_input[3];
    windowH = ksizeList[2];
    windowW = ksizeList[3];
    strideH = stridesList[2];
    strideW = stridesList[3];
  }

  if (dataFormat == "NHWC" && ksizeList[0] == inputH && ksizeList[1] == inputW) {
    return NO_OVERLAP_DIM;
  }
  if (dataFormat == "NCHW" && ksizeList[0] == inputH && ksizeList[1] == inputW) {
    return NO_OVERLAP_DIM;
  }
  if (paddingMode == "SAME") {
    return NO_OVERLAP_DIM;
  }

  vector<vector<int64_t>> y_data_slice = {{}, {}, {}, {}, {}};
  vector<vector<int64_t>> x_data_slice = {{}, {}, {}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_out = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_in = op_desc->MutableInputDesc("x");
  if (!ge::AttrUtils::GetListListInt(tensor_desc_out, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op.GetName().c_str(), "no data slice, use default as {{}, {}, {}, {}, {}}");
    return GRAPH_FAILED;
  }

  for (unsigned i = 0; i < y_data_slice.size(); i++) {
    if (y_data_slice[i].size() > 0) {
      if (i == 0) {
        return NO_OVERLAP_DIM;
      } else if (i == 1 or i == 3 or i == 4) {
        return NOT_SUPPORT_SLICE;
      } else if (i == 2) {
        vector<int64_t> input_h;
        InferHWMaxPool(windowH, strideH, y_data_slice[i], input_h, inputH);
        x_data_slice[i] = input_h;
      }
    }
  }

  for (unsigned i = 0; i < x_data_slice.size(); i++) {
    if (x_data_slice[i].size() > 0) {
      if (!AttrUtils::SetListListInt(tensor_desc_in, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
        return GRAPH_FAILED;
      }
      return GRAPH_SUCCESS;
    }
    return NO_OVERLAP_DIM;
  }

  return NO_OVERLAP_DIM;
}

INFER_FUNC_REG(MaxPool, MaxPoolInferShape);
VERIFY_FUNC_REG(MaxPool, MaxPoolVerify);
INFER_DATA_SLICE_FUNC_REG(MaxPool, MaxPoolInferDataSlice);
// ----------------MaxPool-------------------

// ----------------MaxPool3D-------------------
IMPLEMT_INFERFUNC(MaxPool3D, MaxPool3DInferShape) {
  const size_t DIM_SIZE1 = 1;
  const size_t DIM_SIZE3 = 3;
  const size_t DIM_SIZE5 = 5;
  auto inputTensorDesc = op.GetInputDesc("x");
  auto shape = inputTensorDesc.GetShape();
  Format input_format = inputTensorDesc.GetFormat();
  // get input ksize
  std::vector<int32_t> ksizeList;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksizeList)) {
    OpsGetAttrErrReport(op.GetName(), "ksize");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ksizeList failed!");
    return GRAPH_FAILED;
  }

  if ((ksizeList.size() != DIM_SIZE1) && (ksizeList.size() != DIM_SIZE3) && (ksizeList.size() != DIM_SIZE5)) {
    string excepted_value = ConcatString(DIM_SIZE1, DIM_SIZE3, DIM_SIZE5);
    OpsAttrValueErrReport(op.GetName(), "length of ksize", excepted_value, ConcatString((size_t)ksizeList.size()));
    OP_LOGE(op.GetName().c_str(), "length of ksize must be  1 or 3 or 5!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> ksizeTempList;
  if (ksizeList.size() == DIM_SIZE1) {
    ksizeTempList.push_back(1);
    ksizeTempList.push_back(ksizeList[0]);
    ksizeTempList.push_back(ksizeList[0]);
    ksizeTempList.push_back(ksizeList[0]);
    ksizeTempList.push_back(1);
  }

  if (ksizeList.size() == DIM_SIZE3) {
    ksizeTempList.push_back(1);
    ksizeTempList.push_back(ksizeList[0]);
    ksizeTempList.push_back(ksizeList[1]);
    ksizeTempList.push_back(ksizeList[2]);
    ksizeTempList.push_back(1);
  }

  if (ksizeList.size() == DIM_SIZE5) {
    ksizeTempList.push_back(ksizeList[0]);
    ksizeTempList.push_back(ksizeList[1]);
    ksizeTempList.push_back(ksizeList[2]);
    ksizeTempList.push_back(ksizeList[3]);
    ksizeTempList.push_back(ksizeList[4]);
  }

  // get input strides
  std::vector<int32_t> stridesList;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stridesList)) {
    OpsGetAttrErrReport(op.GetName(), "strides");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr stridesList failed!");
    return GRAPH_FAILED;
  }

  if ((stridesList.size() != DIM_SIZE1) && (stridesList.size() != DIM_SIZE3) && (stridesList.size() != DIM_SIZE5)) {
    string excepted_value = ConcatString(DIM_SIZE1, DIM_SIZE3, DIM_SIZE5);
    OpsAttrValueErrReport(op.GetName(), "length of strides", excepted_value, ConcatString((size_t)stridesList.size()));
    OP_LOGE(op.GetName().c_str(), "length of strides must be  1 or 3 or 5!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> stridesTempList;
  if (stridesList.size() == DIM_SIZE1) {
    stridesTempList.push_back(1);
    stridesTempList.push_back(stridesList[0]);
    stridesTempList.push_back(stridesList[0]);
    stridesTempList.push_back(stridesList[0]);
    stridesTempList.push_back(1);
  }

  if (stridesList.size() == DIM_SIZE3) {
    stridesTempList.push_back(1);
    stridesTempList.push_back(stridesList[0]);
    stridesTempList.push_back(stridesList[1]);
    stridesTempList.push_back(stridesList[2]);
    stridesTempList.push_back(1);
  }

  if (stridesList.size() == DIM_SIZE5) {
    stridesTempList.push_back(stridesList[0]);
    stridesTempList.push_back(stridesList[1]);
    stridesTempList.push_back(stridesList[2]);
    stridesTempList.push_back(stridesList[3]);
    stridesTempList.push_back(stridesList[4]);
  }

  // get input data_format
  std::string dataFormat;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
    OpsGetAttrErrReport(op.GetName(), "data_format");
    OP_LOGE(op.GetName().c_str(),
            "The MaxPool3D op GetOpAttr data_format "
            "failed!");
    return GRAPH_FAILED;
  }

  // get input paddingMode
  std::string paddingMode;
  if (GRAPH_SUCCESS != op.GetAttr("padding", paddingMode)) {
    OpsGetAttrErrReport(op.GetName(), "padding");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr padding failed!");
    return GRAPH_FAILED;
  }

  if (paddingMode != "SAME" && paddingMode != "VALID") {
    string excepted_value = ConcatString("SAME,VALID");
    OpsAttrValueErrReport(op.GetName(), "padding", excepted_value, paddingMode);
    OP_LOGE(op.GetName().c_str(),
            "MaxPool3D can only support SAME or VALID "
            "padding mode!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dims_input = shape.GetDims();
  // set output shape
  std::vector<int64_t> dimVector;
  if (input_format == FORMAT_NDHWC) {
    if (paddingMode == "SAME") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        // D H W calculate, N C stride default 1
        int64_t dims = (dims_input[i] + stridesTempList[i] - 1) / stridesTempList[i];
        dimVector.push_back(dims);
      }
    } else {
      for (size_t i = 0; i < dims_input.size(); i++) {
        // D H W calculate, N C stride default 1
        int64_t dims = (dims_input[i] - ksizeTempList[i] + 1 + (stridesTempList[i] - 1)) / stridesTempList[i];
        dimVector.push_back(dims);
      }
    }
  }

  TensorDesc td = op.GetOutputDesc("y");
  DataType inputDtype = inputTensorDesc.GetDataType();
  Shape outputShape(dimVector);
  td.SetShape(outputShape);
  td.SetDataType(inputDtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MaxPool3D, MaxPool3DVerify) {
  // verify in infer func
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MaxPool3D, MaxPool3DInferShape);
VERIFY_FUNC_REG(MaxPool3D, MaxPool3DVerify);
// ----------------MaxPool3D-------------------

// ---------------------MaxPool3DGradGrad---------------------
static bool GetAttrsMaxPool3DGradGrad(ge::Operator& op, Format refer, int32_t& strd, int32_t& strh, int32_t& strw,
                                      int32_t& kd, int32_t& kh, int32_t& kw) {
  std::vector<int32_t> strideList;
  if (GRAPH_SUCCESS != op.GetAttr("strides", strideList)) {
    OP_LOGE(op.GetName().c_str(), "Failed to get strides!");
    return GRAPH_FAILED;
  }

  std::vector<int32_t> ksizeList;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksizeList)) {
    OP_LOGE(op.GetName().c_str(), "Failed to get ksize!");
    return GRAPH_FAILED;
  }

  if (ksizeList.size() == 1) {
    kd = kh = kw = ksizeList[0];
  } else if (ksizeList.size() == 3) {
    kd = ksizeList[0];
    kh = ksizeList[1];
    kw = ksizeList[2];
  } else if (ksizeList.size() == 5) {
    if (refer == FORMAT_NCDHW) {
      kd = ksizeList[2];
      kh = ksizeList[3];
      kw = ksizeList[4];
    } else {
      kd = ksizeList[1];
      kh = ksizeList[2];
      kw = ksizeList[3];
    }
  }

  if (strideList.size() == 1) {
    strd = strh = strw = strideList[0];
  } else if (strideList.size() == 3) {
    strd = strideList[0];
    strh = strideList[1];
    strw = strideList[2];
  } else if (strideList.size() == 5) {
    if (refer == FORMAT_NCDHW) {
      strd = strideList[2];
      strh = strideList[3];
      strw = strideList[4];
    } else {
      strd = strideList[1];
      strh = strideList[2];
      strw = strideList[3];
    }
  }

  return true;
}

IMPLEMT_VERIFIER(MaxPool3DGradGrad, MaxPool3DGradGradVerify) {
  OP_LOGD(op.GetName().c_str(), "Entering MaxPool3DGradGradDVerify");
  if (!CheckTwoInputDtypeSame(op, "orig_x", "orig_y") || !CheckTwoInputDtypeSame(op, "orig_x", "grads")) {
    OP_LOGE(op.GetName().c_str(), "MaxPool3DGradGrad, two input dtypes must be same");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> ksize;
  ksize = GetAttrValue(op, "ksize");
  if (!CheckListEmpty(op.GetName(), ksize, "ksize")) {
    OP_LOGE(op.GetName().c_str(), "get attr ksize failed");
    return GRAPH_FAILED;
  }
  if (ksize.size() != 1 && ksize.size() != 3 && ksize.size() != 5) {
    OP_LOGE(op.GetName().c_str(), "attr ksize(%d) must be 1 3 or 5", (int)ksize.size());
    return GRAPH_FAILED;
  }
  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), strides, "strides")) {
    OP_LOGE(op.GetName().c_str(), "get attr strides failed");
    return GRAPH_FAILED;
  }
  if (strides.size() != 1 && strides.size() != 3 && strides.size() != 5) {
    OP_LOGE(op.GetName().c_str(), "attr strides(%d) must be 1 3 or 5", (int)strides.size());
    return GRAPH_FAILED;
  }

  std::string data_format;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
    if (data_format != "NCDHW" && data_format != "NDHWC") {
      OP_LOGE(op.GetName().c_str(), "attr data_format(%s) only support NCDHW and NDHWC", data_format.c_str());
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(MaxPool3DGradGrad, MaxPool3DGradGradInferShape) {
  OP_LOGD(op.GetName().c_str(), "Entering MaxPool3DGradGradInferShape");
  auto input_orig_out = op.GetInputDesc("orig_y");
  auto shape_orig_out = input_orig_out.GetShape();
  auto type_orig_out = input_orig_out.GetDataType();

  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(shape_orig_out);
  td.SetDataType(type_orig_out);
  (void)op.UpdateOutputDesc("y", td);

  // get input DHW
  auto xShape = op.GetInputDesc("orig_x").GetShape().GetDims();
  auto xFormat = op.GetInputDesc("orig_x").GetFormat();
  int32_t id = 0;
  int32_t ih = 0;
  int32_t iw = 0;

  if ((xFormat == FORMAT_NCDHW) && (xShape.size() >= 5)) {
    id = xShape[2];
    ih = xShape[3];
    iw = xShape[4];
  } else if ((xFormat == FORMAT_NDHWC) && (xShape.size() >= 5)) {
    id = xShape[1];
    ih = xShape[2];
    iw = xShape[3];
  } else {
    map<string, string> err_map;
    err_map["param_name"] = "xFormat";
    err_map["op_name"] = "MaxPool3DGradGrad";
    err_map["excepted_value"] = "NCDHW or NDHWC";
    err_map["input_value"] = xFormat;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // get ksize and strides
  int32_t kd = 0;
  int32_t kh = 0;
  int32_t kw = 0;
  int32_t strd = 0;
  int32_t strh = 0;
  int32_t strw = 0;
  if (false == GetAttrsMaxPool3DGradGrad(op, xFormat, strd, strh, strw, kd, kh, kw)) {
    OP_LOGE(op.GetName().c_str(), "Failed to get attr in MaxPool3DGradGrad");
    return GRAPH_FAILED;
  }
  if ((strd == 0) || (strh == 0) || (strw == 0)) {
    OP_LOGE(op.GetName().c_str(), "strd/strh/strw should not be zero");
    return GRAPH_FAILED;
  }
  // construct pads attr
  if (false == Construct3DPadsByPadding("MaxPool3DGradGradD", op, id, ih, iw, kd, kh, kw, strd, strh, strw)) {
    OP_LOGE(op.GetName().c_str(), "Failed to get pads in MaxPool3DGradGrad");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MaxPool3DGradGrad, MaxPool3DGradGradInferShape);
VERIFY_FUNC_REG(MaxPool3DGradGrad, MaxPool3DGradGradVerify);
// ---------------------MaxPool3DGradGrad---------------------

// ----------------MaxPoolGrad-------------------
IMPLEMT_VERIFIER(MaxPoolGrad, MaxPoolGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2") || !CheckTwoInputDtypeSame(op, "x1", "grad")) {
    return GRAPH_FAILED;
  }
  std::vector<int64_t> ksize;
  ksize = GetAttrValue(op, "ksize");
  if (!CheckListEmpty(op.GetName(), ksize, "ksize")) {
    OP_LOGE(op.GetName().c_str(), "Get attr ksize failed");
    return GRAPH_FAILED;
  }
  if (ksize.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "Attr ksize(%u) must be 4", ksize.size());
    return GRAPH_FAILED;
  }
  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), strides, "strides")) {
    OP_LOGE(op.GetName().c_str(), "Get attr strides failed");
    return GRAPH_FAILED;
  }
  if (strides.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "Attr strides(%u) must be 4", strides.size());
    return GRAPH_FAILED;
  }
  std::string padding;
  if (op.GetAttr("padding", padding) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get padding failed!");
    return GRAPH_FAILED;
  }
  if (padding != "SAME" && padding != "VALID") {
    OP_LOGE(op.GetName().c_str(), "Attr padding(%s) only support SAME and VALID", padding.c_str());
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_SUCCESS) {
    if (data_format != "NCHW" && data_format != "NHWC") {
      OP_LOGE(op.GetName().c_str(), "Attr data_format(%s) only support NCHW and NHWC", data_format.c_str());
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(MaxPoolGrad, MaxPoolGradInferShape) {
  if (OneInOneOutDynamicInfer(op, "x1", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

INFER_FUNC_REG(MaxPoolGrad, MaxPoolGradInferShape);
VERIFY_FUNC_REG(MaxPoolGrad, MaxPoolGradVerify);
// ---------------------MaxPoolGrad---------------------

// ---------------------MaxPoolGradGrad---------------------
IMPLEMT_VERIFIER(MaxPoolGradGrad, MaxPoolGradGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2") || !CheckTwoInputDtypeSame(op, "x1", "grad")) {
    OP_LOGE(op.GetName().c_str(), "Two input dtypes must be same");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> ksize;
  ksize = GetAttrValue(op, "ksize");
  if (!CheckListEmpty(op.GetName(), ksize, "ksize")) {
    OP_LOGE(op.GetName().c_str(), "Get attr ksize failed");
    return GRAPH_FAILED;
  }
  if (ksize.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "Attr ksize(%u) must be 4", ksize.size());
    return GRAPH_FAILED;
  }
  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), strides, "strides")) {
    OP_LOGE(op.GetName().c_str(), "Get attr strides failed");
    return GRAPH_FAILED;
  }
  if (strides.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "Attr strides(%u) must be 4", strides.size());
    return GRAPH_FAILED;
  }
  std::string padding;
  if (op.GetAttr("padding", padding) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get padding failed!");
    return GRAPH_FAILED;
  }
  if (padding != "SAME" && padding != "VALID") {
    OP_LOGE(op.GetName().c_str(), "Attr padding(%s) only support SAME and VALID", padding.c_str());
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_SUCCESS) {
    if (data_format != "NCHW" && data_format != "NHWC") {
      OP_LOGE(op.GetName().c_str(), "Attr data_format(%s) only support NCHW and NHWC", data_format.c_str());
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(MaxPoolGradGrad, MaxPoolGradGradInferShape) {
  auto input_x2 = op.GetInputDesc("x2");
  auto shape_x2 = input_x2.GetShape();
  auto type_x2 = input_x2.GetDataType();

  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(shape_x2);
  tensordesc_output.SetDataType(type_x2);
  if (op.UpdateOutputDesc("y", tensordesc_output) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MaxPoolGradGrad, MaxPoolGradGradInferShape);
VERIFY_FUNC_REG(MaxPoolGradGrad, MaxPoolGradGradVerify);
// ---------------------MaxPoolGradGrad---------------------

// ---------------------MaxPoolExt2---------------------
IMPLEMT_INFERFUNC(MaxPoolExt2, MaxPoolExt2InferShape) {
  const size_t DIM_SIZE2 = 2;
  const size_t DIM_SIZE3 = 3;
  const size_t DIM_SIZE4 = 4;
  auto inputTensorDesc = op.GetInputDesc("x");
  auto shape = inputTensorDesc.GetShape();
  Format input_format = inputTensorDesc.GetFormat();
  // get input ksize
  std::vector<int32_t> ksizeList;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksizeList)) {
    OpsGetAttrErrReport(op.GetName(), "ksize");
    OP_LOGW(op.GetName().c_str(), "GetOpAttr ksizeList failed!");
    std::vector<int64_t> outputVec = {-1, -1, -1, -1, -1};
    TensorDesc td = op.GetOutputDesc("y");
    DataType inputDtype = inputTensorDesc.GetDataType();
    Shape outputShape(outputVec);
    td.SetShape(outputShape);
    td.SetDataType(inputDtype);
    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
  }

  if (ksizeList.size() != DIM_SIZE4) {
    string excepted_value = ConcatString("equal to the length of x'shape[", DIM_SIZE4, "]");
    OpsAttrValueErrReport(op.GetName(), "ksize'length", excepted_value, ConcatString(ksizeList.size()));
    OP_LOGE(op.GetName().c_str(),
            "length of ksize must be equal to the "
            "length of shape!");
    return GRAPH_FAILED;
  }

  // get input strides
  std::vector<int32_t> stridesList;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stridesList)) {
    OpsGetAttrErrReport(op.GetName(), "strides");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr stridesList failed!");
    return GRAPH_FAILED;
  }

  if (stridesList.size() != DIM_SIZE4) {
    string excepted_value = ConcatString("equal to the length of x'shape[", DIM_SIZE4, "]");
    OpsAttrValueErrReport(op.GetName(), "strides'length", excepted_value, ConcatString(stridesList.size()));
    OP_LOGE(op.GetName().c_str(),
            "length of strides must be equal to the "
            "length of shape!");
    return GRAPH_FAILED;
  }

  // get input data_format
  std::string dataFormat;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
    OpsGetAttrErrReport(op.GetName(), "data_format");
    OP_LOGE(op.GetName().c_str(),
            "The MaxPool op GetOpAttr data_format "
            "failed!");
    return GRAPH_FAILED;
  }

  // get input paddingMode
  std::string paddingMode;
  if (GRAPH_SUCCESS != op.GetAttr("padding", paddingMode)) {
    OpsGetAttrErrReport(op.GetName(), "padding");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr padding failed!");
    return GRAPH_FAILED;
  }

  if (paddingMode != "SAME" && paddingMode != "VALID") {
    string excepted_value = ConcatString("SAME, VALID");
    OpsAttrValueErrReport(op.GetName(), "padding", excepted_value, paddingMode);
    OP_LOGE(op.GetName().c_str(),
            "MaxPool can only support SAME or VALID "
            "padding mode!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dims_input = shape.GetDims();
  // set output shape
  std::vector<int64_t> dimVector;
  if (input_format == FORMAT_NHWC) {
    if (paddingMode == "SAME") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == DIM_SIZE1) || (i == DIM_SIZE2)) {
          if (dataFormat == "NHWC") {
            int64_t dims = (dims_input[i] + stridesList[i] - 1) / stridesList[i];
            dimVector.push_back(dims);
          } else {
            int64_t dims = (dims_input[i] + stridesList[i + 1] - 1) / stridesList[i + 1];
            dimVector.push_back(dims);
          }
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    } else {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == DIM_SIZE1) || (i == DIM_SIZE2)) {
          if (dataFormat == "NHWC") {
            int64_t dims = (dims_input[i] - ksizeList[i] + 1 + (stridesList[i] - 1)) / stridesList[i];
            dimVector.push_back(dims);
          } else {
            int64_t dims = (dims_input[i] - ksizeList[i + 1] + 1 + (stridesList[i + 1] - 1)) / stridesList[i + 1];
            dimVector.push_back(dims);
          }
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    }
  } else {
    if (paddingMode == "SAME") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == DIM_SIZE2) || (i == DIM_SIZE3)) {
          if (dataFormat == "NHWC") {
            int64_t dims = (dims_input[i] + stridesList[i - 1] - 1) / stridesList[i - 1];
            dimVector.push_back(dims);
          } else {
            int64_t dims = (dims_input[i] + stridesList[i] - 1) / stridesList[i];
            dimVector.push_back(dims);
          }
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    } else {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == DIM_SIZE2) || (i == DIM_SIZE3)) {
          if (dataFormat == "NHWC") {
            int64_t dims = (dims_input[i] - ksizeList[i - 1] + 1 + (stridesList[i - 1] - 1)) / stridesList[i - 1];
            dimVector.push_back(dims);
          } else {
            int64_t dims = (dims_input[i] - ksizeList[i] + 1 + (stridesList[i] - 1)) / stridesList[i];
            dimVector.push_back(dims);
          }
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    }
  }
  TensorDesc td = op.GetOutputDesc("y");
  DataType inputDtype = inputTensorDesc.GetDataType();
  Shape outputShape(dimVector);
  td.SetShape(outputShape);
  td.SetDataType(inputDtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

static void InferHWMaxPoolExt2(int64_t kernel,int64_t stride, vector<int64_t>& output, vector<int64_t>& input,
                           int64_t& ori_input) {
    int64_t first_start = 0;
    int64_t second_start = 0;
    int64_t first_end = 0;
    int64_t second_end = 0;
    int64_t start = 0;
    int64_t end = 0;
    first_start = output[0] * stride;
    second_start = output[1] * stride;
    first_end = std::min(first_start + kernel, ori_input);
    second_end = std::min(second_start + kernel, ori_input);
    start = std::max(first_start, int64_t(0));
    end = second_end - 1;
    input = {start, end};
}

IMPLEMT_INFER_DATA_SLICE(MaxPoolExt2, MaxPoolExt2InferDataSlice){
  auto inputTensorDesc = op.GetInputDesc("x");
  auto shape = inputTensorDesc.GetShape();
  std::vector<int64_t> dims_input = shape.GetDims();

  std::vector<int64_t> ksizeList;
  std::vector<int64_t> stridesList;
  std::string dataFormat;
  std::string paddingMode;
  op.GetAttr("ksize", ksizeList);
  op.GetAttr("strides", stridesList);
  op.GetAttr("data_format", dataFormat);
  op.GetAttr("padding", paddingMode);

  int64_t inputH = 0;
  int64_t inputW = 0;
  int64_t windowH = 0;
  int64_t windowW = 0;
  int64_t strideH = 0;
  int64_t strideW = 0;
  int64_t dilationH = 0;

  if (dataFormat == "NHWC") {
    inputH = dims_input[1];
    inputW = dims_input[2];
    windowH = ksizeList[1];
    windowW = ksizeList[2];
    strideH = stridesList[1];
    strideW = stridesList[2];
  } else if(dataFormat == "NCHW") {
    inputH = dims_input[2];
    inputW = dims_input[3];
    windowH = ksizeList[2];
    windowW = ksizeList[3];
    strideH = stridesList[2];
    strideW = stridesList[3];
  }

  if (dataFormat == "NHWC" && ksizeList[0] == inputH && ksizeList[1] == inputW) {
    return NO_OVERLAP_DIM;
  }
  if (dataFormat == "NCHW" && ksizeList[0] == inputH && ksizeList[1] == inputW) {
    return NO_OVERLAP_DIM;
  }
  if (paddingMode == "SAME") {
    return NO_OVERLAP_DIM;
  }

  vector<vector<int64_t>> y_data_slice = {{}, {}, {}, {}, {}};
  vector<vector<int64_t>> x_data_slice = {{}, {}, {}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_out = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_in = op_desc->MutableInputDesc("x");
  if (!ge::AttrUtils::GetListListInt(tensor_desc_out, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op.GetName().c_str(), "no data slice, use default as {{}, {}, {}, {}, {}}");
    return GRAPH_FAILED;
  }

  for(unsigned i = 0; i < y_data_slice.size(); i++) {
    if (y_data_slice[i].size() > 0) {
      if (i == 0) {
        return NO_OVERLAP_DIM;
      } else if (i == 1 or i == 3 or i == 4){
        return NOT_SUPPORT_SLICE;
      } else if (i == 2) {
        vector<int64_t> input_h;
        InferHWMaxPoolExt2(windowH, strideH, y_data_slice[i], input_h, inputH);
        x_data_slice[i] = input_h;
      }
    }
  }

  for(unsigned i = 0; i < x_data_slice.size(); i++) {
    if (x_data_slice[i].size() > 0) {
      if(!AttrUtils::SetListListInt(tensor_desc_in, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
        return GRAPH_FAILED;
      }
      return GRAPH_SUCCESS;
    }
    return NO_OVERLAP_DIM;
  }

  return NO_OVERLAP_DIM;
}

INFER_FUNC_REG(MaxPoolExt2, MaxPoolExt2InferShape);
INFER_DATA_SLICE_FUNC_REG(MaxPoolExt2, MaxPoolExt2InferDataSlice);
// ---------------------MaxPoolExt2---------------------

// ---------------------MaxPoolGradWithArgmax---------------------
IMPLEMT_VERIFIER(MaxPoolGradWithArgmax, MaxPoolGradWithArgmaxVerify) {
  const size_t DIM_SIZE4 = 4;
  // get input ksize
  std::vector<int32_t> ksizeList;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksizeList)) {
    OpsGetAttrErrReport(op.GetName(), "ksize");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ksizeList failed!");
    return GRAPH_FAILED;
  }
  if (ksizeList.size() < DIM_SIZE4) {
    string excepted_value = ConcatString("more than[", DIM_SIZE4, "]");
    OpsAttrValueErrReport(op.GetName(), "ksize'length", excepted_value, ConcatString(ksizeList.size()));
    OP_LOGE(op.GetName().c_str(), "length of ksize must be more than 4");
    return GRAPH_FAILED;
  }
  // get input strides
  std::vector<int32_t> stridesList;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stridesList)) {
    OpsGetAttrErrReport(op.GetName(), "strides");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr stridesList failed!");
    return GRAPH_FAILED;
  }

  if (stridesList.size() < DIM_SIZE4) {
    string excepted_value = ConcatString("more than[", DIM_SIZE4, "]");
    OpsAttrValueErrReport(op.GetName(), "strides'length", excepted_value, ConcatString(stridesList.size()));
    OP_LOGE(op.GetName().c_str(), "length of strides must be more than 4");
    return GRAPH_FAILED;
  }
  if ((ksizeList[0] != 1) || (ksizeList[3] != 1) || (stridesList[0] != 1) || (stridesList[3] != 1)) {
    OP_LOGE(op.GetName().c_str(),
            "MaxPoolGradWithArgmax only supports pooling "
            "across width/height, and other ksize "
            "dimension should be one");
    return GRAPH_FAILED;
  }
  if ((ksizeList[1] * ksizeList[2]) > 255) {
    OpsAttrValueErrReport(op.GetName(), "window", "<= 255", ConcatString((ksizeList[1] * ksizeList[2])));
    OP_LOGE(op.GetName().c_str(),
            "invalid window params, window_h*window_w "
            "should be <= 255");
    return GRAPH_FAILED;
  }
  // get input paddingMode
  std::string paddingMode;
  if (GRAPH_SUCCESS != op.GetAttr("padding", paddingMode)) {
    OpsGetAttrErrReport(op.GetName(), "padding");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr padding failed!");
    return GRAPH_FAILED;
  }
  if (paddingMode != "SAME" && paddingMode != "VALID") {
    string excepted_value = ConcatString("SAME, VALID");
    OpsAttrValueErrReport(op.GetName(), "padding", excepted_value, paddingMode);
    OP_LOGE(op.GetName().c_str(),
            "MaxPool can only support SAME or VALID "
            "padding mode!");
    return GRAPH_FAILED;
  }
  if (!CheckTwoInputDtypeSame(op, "x", "grad")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(MaxPoolGradWithArgmax, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
VERIFY_FUNC_REG(MaxPoolGradWithArgmax, MaxPoolGradWithArgmaxVerify);
// ---------------------MaxPoolGradWithArgmax---------------------

// ---------------------MaxPoolV2---------------------
static void GetMaxPoolV2ConstData(const Tensor& data, const DataType& dtype, std::vector<int64_t>& const_vec) {
  const uint8_t* constData = data.GetData();
  size_t size;
  if (dtype == ge::DT_INT32) {
    size = data.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; ++i) {
      const_vec.push_back(*((int32_t*)(constData) + i));
    }
  } else {
    size = data.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; ++i) {
      const_vec.push_back(*((int64_t*)(constData) + i));
    }
  }
}

IMPLEMT_INFERFUNC(MaxPoolV2, MaxPoolV2InferShape) {
  const size_t DIM_SIZE2 = 2;
  const size_t DIM_SIZE3 = 3;
  const size_t DIM_SIZE4 = 4;
  auto inputTensorDesc = op.GetInputDesc("x");
  auto shape = inputTensorDesc.GetShape();
  Format input_format = inputTensorDesc.GetFormat();
  // get input ksize
  Tensor ksizeData;
  if (ge::GRAPH_SUCCESS != op.GetInputConstData("ksize", ksizeData)) {
    OP_LOGE(op.GetName().c_str(), "get const data failed");
    return GRAPH_FAILED;
  }
  DataType ksizeDtype = op.GetInputDesc("ksize").GetDataType();
  std::vector<int64_t> ksizeList;
  GetMaxPoolV2ConstData(ksizeData, ksizeDtype, ksizeList);
  if (ksizeList.size() != DIM_SIZE4) {
    OP_LOGE(op.GetName().c_str(),
            "length of ksize %zu must be equal to the "
            "length of shape!",
            ksizeList.size());
    return GRAPH_FAILED;
  }

  // get input strides
  Tensor stridesData;
  if (ge::GRAPH_SUCCESS != op.GetInputConstData("strides", stridesData)) {
    OP_LOGE(op.GetName().c_str(), "get constdata failed");
    return GRAPH_FAILED;
  }
  DataType stridesDtype = op.GetInputDesc("strides").GetDataType();
  std::vector<int64_t> stridesList;
  GetMaxPoolV2ConstData(stridesData, stridesDtype, stridesList);
  if (stridesList.size() != DIM_SIZE4) {
    OP_LOGE(op.GetName().c_str(),
            "length of strides must be equal to the "
            "length of shape!");
    return GRAPH_FAILED;
  }
  // get input data_format
  std::string dataFormat;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
    OP_LOGE(op.GetName().c_str(),
            "The MaxPool op GetOpAttr data_format "
            "failed!");
    return GRAPH_FAILED;
  }
  if (dataFormat != "NHWC" && dataFormat != "NCHW" && dataFormat != "NC1HWC0") {
    OP_LOGE(op.GetName().c_str(),
            "data_format only "
            "support 'NHWC','NCHW' and 'NC1HWC0'.");
    return GRAPH_FAILED;
  }
  if (dataFormat == "NHWC") {
    if ((ksizeList[0] != 1) || (ksizeList[3] != 1) || (stridesList[0] != 1) || (stridesList[3] != 1)) {
      OP_LOGE(op.GetName().c_str(),
              "MaxPool only supports pooling across width/height"
              "and other ksize dimension should be one");
      return GRAPH_FAILED;
    }
  }
  if ((dataFormat == "NCHW") || (dataFormat == "NC1HWC0")) {
    if ((ksizeList[0] != 1) || (ksizeList[1] != 1) || (stridesList[0] != 1) || (stridesList[1] != 1)) {
      OP_LOGE(op.GetName().c_str(),
              "MaxPool only supports pooling across width/height"
              "and other ksize dimension should be one");
      return GRAPH_FAILED;
    }
  }

  // get input paddingMode
  std::string paddingMode;
  if (GRAPH_SUCCESS != op.GetAttr("padding", paddingMode)) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr padding failed!");
    return GRAPH_FAILED;
  }

  if (paddingMode != "SAME" && paddingMode != "VALID") {
    OP_LOGE(op.GetName().c_str(),
            "MaxPool can only support SAME or VALID "
            "padding mode!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dims_input = shape.GetDims();
  // set output shape
  std::vector<int64_t> dimVector;
  if (input_format == FORMAT_NHWC) {
    if (paddingMode == "SAME") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == DIM_SIZE1) || (i == DIM_SIZE2)) {
          if (dataFormat == "NHWC") {
            int64_t dims = (dims_input[i] + stridesList[i] - 1) / stridesList[i];
            dimVector.push_back(dims);
          } else {
            int64_t dims = (dims_input[i] + stridesList[i + 1] - 1) / stridesList[i + 1];
            dimVector.push_back(dims);
          }
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    } else {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == DIM_SIZE1) || (i == DIM_SIZE2)) {
          if (dataFormat == "NHWC") {
            int64_t dims = (dims_input[i] - ksizeList[i] + 1 + (stridesList[i] - 1)) / stridesList[i];
            dimVector.push_back(dims);
          } else {
            int64_t dims = (dims_input[i] - ksizeList[i + 1] + 1 + (stridesList[i + 1] - 1)) / stridesList[i + 1];
            dimVector.push_back(dims);
          }
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    }
  } else {
    if (paddingMode == "SAME") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == DIM_SIZE2) || (i == DIM_SIZE3)) {
          if (dataFormat == "NHWC") {
            int64_t dims = (dims_input[i] + stridesList[i - 1] - 1) / stridesList[i - 1];
            dimVector.push_back(dims);
          } else {
            int64_t dims = (dims_input[i] + stridesList[i] - 1) / stridesList[i];
            dimVector.push_back(dims);
          }
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    } else {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == DIM_SIZE2) || (i == DIM_SIZE3)) {
          if (dataFormat == "NHWC") {
            int64_t dims = (dims_input[i] - ksizeList[i - 1] + 1 + (stridesList[i - 1] - 1)) / stridesList[i - 1];
            dimVector.push_back(dims);
          } else {
            int64_t dims = (dims_input[i] - ksizeList[i] + 1 + (stridesList[i] - 1)) / stridesList[i];
            dimVector.push_back(dims);
          }
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    }
  }
  TensorDesc td = op.GetOutputDesc("y");
  DataType inputDtype = inputTensorDesc.GetDataType();
  Shape outputShape(dimVector);
  td.SetShape(outputShape);
  td.SetDataType(inputDtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MaxPoolV2, MaxPoolV2InferShape);
// ---------------------MaxPoolV2---------------------

int64_t CeilDev(int64_t value, int64_t factor) {
  int64_t value_num = 0;
  if (value % factor == 0) {
    value_num = value / factor;
  } else {
    value_num = value / factor + 1;
  }
  return value_num;
}

// ---------------------MaxPoolWithArgmax---------------------
IMPLEMT_INFERFUNC(MaxPoolWithArgmax, MaxPoolWithArgmaxInferShape) {
  const size_t DIM_SIZE1 = 1;
  const size_t DIM_SIZE2 = 2;
  const size_t DIM_SIZE3 = 3;
  const size_t DIM_SIZE4 = 4;
  ge::TensorDesc inputTensorDesc = op.GetInputDesc(0);
  Format input_format = inputTensorDesc.GetFormat();
  ge::Shape shape = inputTensorDesc.GetShape();
  int32_t in_size_h = 0;
  int32_t in_size_w = 0;
  if (input_format == FORMAT_NHWC) {
    in_size_h = shape.GetDim(1);
    in_size_w = shape.GetDim(2);
  } else {
    in_size_h = shape.GetDim(2);
    in_size_w = shape.GetDim(3);
  }
  // get input ksize
  std::vector<int32_t> ksizeList;
  if (op.GetAttr("ksize", ksizeList) != ge::GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "ksize");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ksizeList failed!");
    return GRAPH_FAILED;
  }

  if (ksizeList.size() != DIM_SIZE4) {
    string excepted_value = ConcatString("equal to the length of x'shape[", DIM_SIZE4, "]");
    OpsAttrValueErrReport(op.GetName(), "ksize'length", excepted_value, ConcatString(ksizeList.size()));
    OP_LOGE(op.GetName().c_str(),
            "length of ksize must be equal to"
            "the length of shape!");
    return GRAPH_FAILED;
  }

  // get input strides
  std::vector<int32_t> stridesList;
  if (op.GetAttr("strides", stridesList) != ge::GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "strides");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr stridesList failed!");
    return GRAPH_FAILED;
  }

  if (stridesList.size() != DIM_SIZE4) {
    string excepted_value = ConcatString("equal to the length of x'shape[", DIM_SIZE4, "]");
    OpsAttrValueErrReport(op.GetName(), "strides'length", excepted_value, ConcatString(stridesList.size()));
    OP_LOGE(op.GetName().c_str(),
            "length of strides must be equal to"
            "the length of shape!");
    return GRAPH_FAILED;
  }
  if ((ksizeList[0] != 1) || (ksizeList[3] != 1) || (stridesList[0] != 1) || (stridesList[3] != 1)) {
    OP_LOGE(op.GetName().c_str(),
            "MaxPoolWithArgmax only supports pooling "
            "across width/height, and other ksize "
            "dimension should be one");
    return GRAPH_FAILED;
  }
  if ((ksizeList[1] * ksizeList[2]) > 255) {
    OpsAttrValueErrReport(op.GetName(), "window", "<= 255", ConcatString((ksizeList[1] * ksizeList[2])));
    OP_LOGE(op.GetName().c_str(),
            "invalid window params, window_h*window_w "
            "should be <= 255");
    return GRAPH_FAILED;
  }
  // get input paddingMode
  std::string paddingMode;
  if (op.GetAttr("padding", paddingMode) != ge::GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "padding");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr padding failed!");
    return GRAPH_FAILED;
  }
  if (paddingMode != "SAME" && paddingMode != "VALID") {
    string excepted_value = ConcatString("SAME, VALID");
    OpsAttrValueErrReport(op.GetName(), "padding", excepted_value, paddingMode);
    OP_LOGE(op.GetName().c_str(),
            "MaxPoolWithArgmax can only support"
            "SAME or VALID padding mode!");
    return GRAPH_FAILED;
  }
  if (((ksizeList[1] > in_size_h) || (ksizeList[2] > in_size_w)) && (paddingMode == "VALID")) {
    OP_LOGE(op.GetName().c_str(), "when padding is VALID, ksize must be not less than input size.");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> dims_input = shape.GetDims();
  // set output max shape
  std::vector<int64_t> dimVector;
  if (input_format == FORMAT_NHWC) {
    if (paddingMode == "SAME") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == DIM_SIZE1) || (i == DIM_SIZE2)) {
          int64_t dims = (dims_input[i] + stridesList[i] - 1) / stridesList[i];
          dimVector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    } else {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == DIM_SIZE1) || (i == DIM_SIZE2)) {
          int64_t dims = (dims_input[i] - ksizeList[i] + 1 + (stridesList[i] - 1)) / stridesList[i];
          dimVector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    }
  } else {
    if (paddingMode == "SAME") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == DIM_SIZE2) || (i == DIM_SIZE3)) {
          int64_t dims = (dims_input[i] + stridesList[i - 1] - 1) / stridesList[i - 1];
          dimVector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    } else {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == DIM_SIZE2) || (i == DIM_SIZE3)) {
          int64_t dims = (dims_input[i] - ksizeList[i - 1] + 1 + (stridesList[i - 1] - 1)) / stridesList[i - 1];
          dimVector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    }
  }
  TensorDesc outputMaxTensorDesc = op.GetOutputDesc("y");
  TensorDesc outputMaskTensorDesc = op.GetOutputDesc("argmax");
  Shape outputMaxShape(dimVector);
  DataType inputDtype = inputTensorDesc.GetDataType();
  outputMaxTensorDesc.SetShape(outputMaxShape);
  outputMaxTensorDesc.SetDataType(inputDtype);
  outputMaskTensorDesc.SetShape(outputMaxShape);
  outputMaskTensorDesc.SetDataType(DT_INT64);
  (void)op.UpdateOutputDesc("y", outputMaxTensorDesc);
  (void)op.UpdateOutputDesc("argmax", outputMaskTensorDesc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MaxPoolWithArgmax, MaxPoolWithArgmaxInferShape);
// ---------------------MaxPoolWithArgmax---------------------

// ---------------------Mask2Argmax---------------------
IMPLEMT_INFERFUNC(Mask2Argmax, Mask2ArgmaxInferShape) {
  const size_t DIM_SIZE1 = 1;
  const size_t DIM_SIZE2 = 2;
  const size_t DIM_SIZE3 = 3;
  const size_t DIM_SIZE4 = 4;
  ge::TensorDesc inputTensorDesc = op.GetInputDesc(0);
  Format input_format = inputTensorDesc.GetFormat();
  ge::Shape shape = inputTensorDesc.GetShape();
  int32_t in_size_h = 0;
  int32_t in_size_w = 0;
  if (input_format == FORMAT_NHWC) {
    in_size_h = shape.GetDim(1);
    in_size_w = shape.GetDim(2);
  } else {
    in_size_h = shape.GetDim(2);
    in_size_w = shape.GetDim(3);
  }
  // get input ksize
  std::vector<int32_t> ksizeList;
  if (op.GetAttr("ksize", ksizeList) != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ksizeList failed!");
    return GRAPH_FAILED;
  }

  if (ksizeList.size() != DIM_SIZE4) {
    OP_LOGE(op.GetName().c_str(),
            "length of ksize must be equal to"
            "the length of shape!");
    return GRAPH_FAILED;
  }

  // get input strides
  std::vector<int32_t> stridesList;
  if (op.GetAttr("strides", stridesList) != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr stridesList failed!");
    return GRAPH_FAILED;
  }

  if (stridesList.size() != DIM_SIZE4) {
    OP_LOGE(op.GetName().c_str(),
            "length of strides must be equal to"
            "the length of shape!");
    return GRAPH_FAILED;
  }
  if ((ksizeList[0] != 1) || (ksizeList[3] != 1) || (stridesList[0] != 1) || (stridesList[3] != 1)) {
    OP_LOGE(op.GetName().c_str(),
            "Mask2Argmax only supports pooling "
            "across width/height, and other ksize "
            "dimension should be one");
    return GRAPH_FAILED;
  }
  if ((ksizeList[1] * ksizeList[2]) > 255) {
    OP_LOGE(op.GetName().c_str(),
            "invalid window params, window_h*window_w "
            "should be <= 255");
    return GRAPH_FAILED;
  }

  // get input paddingMode
  std::string paddingMode;
  if (op.GetAttr("padding", paddingMode) != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr padding failed!");
    return GRAPH_FAILED;
  }

  if (paddingMode != "SAME" && paddingMode != "VALID") {
    OP_LOGE(op.GetName().c_str(),
            "Mask2Argmax can only support"
            "SAME or VALID padding mode!");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> dims_input = shape.GetDims();
  // set output max shape
  std::vector<int64_t> dimVector;
  if (input_format == FORMAT_NHWC) {
    if (paddingMode == "SAME") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == DIM_SIZE1) || (i == DIM_SIZE2)) {
          int64_t dims = (dims_input[i] + stridesList[i] - 1) / stridesList[i];
          dimVector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    } else {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == DIM_SIZE1) || (i == DIM_SIZE2)) {
          int64_t dims = (dims_input[i] - ksizeList[i] + 1 + (stridesList[i] - 1)) / stridesList[i];
          dimVector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    }
  } else {
    if (paddingMode == "SAME") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == DIM_SIZE2) || (i == DIM_SIZE3)) {
          int64_t dims = (dims_input[i] + stridesList[i - 1] - 1) / stridesList[i - 1];
          dimVector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    } else {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == DIM_SIZE2) || (i == DIM_SIZE3)) {
          int64_t dims = (dims_input[i] - ksizeList[i - 1] + 1 + (stridesList[i - 1] - 1)) / stridesList[i - 1];
          dimVector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    }
  }

  TensorDesc outputArgmaxTensorDesc = op.GetOutputDesc("argmax");
  Shape outputArgmaxShape(dimVector);
  DataType outputArgmaxDtype = DT_FLOAT;
  outputArgmaxTensorDesc.SetShape(outputArgmaxShape);
  outputArgmaxTensorDesc.SetDataType(outputArgmaxDtype);

  (void)op.UpdateOutputDesc("argmax", outputArgmaxTensorDesc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Mask2Argmax, Mask2ArgmaxInferShape);
// ---------------------Mask2Argmax---------------------

// ----------------------MaxPoolGradGradWithArgmax-----------------------
IMPLEMT_VERIFIER(MaxPoolGradGradWithArgmax, MaxPoolGradGradWithArgmaxVerify) {
  DataType input_x_type = op.GetInputDesc("x").GetDataType();
  DataType input_grad_type = op.GetInputDesc("grad").GetDataType();
  if (input_x_type != input_grad_type) {
    OP_LOGE(op.GetName().c_str(), "The max_pool_grad_grad_with_argmax op inputs should have the same dtype!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(MaxPoolGradGradWithArgmax, MaxPoolGradGradWithArgmaxInferShape) {
  auto input_tensor_desc = op.GetInputDesc("x");
  auto shape = input_tensor_desc.GetShape();
  Format input_format = input_tensor_desc.GetFormat();
  // get input ksize
  std::vector<int32_t> ksize_list;
  if (op.GetAttr("ksize", ksize_list) != GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "ksize");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ksize_list failed!");
    return GRAPH_FAILED;
  }

  if (ksize_list.size() != 4) {
    OpsAttrValueErrReport(op.GetName(), "ksize", ConcatString(4), ConcatString(ksize_list.size()));
    OP_LOGE(op.GetName().c_str(), "Length of ksize must be equal to the length of shape!");
    return GRAPH_FAILED;
  }

  // get input strides
  std::vector<int32_t> strides_list;
  if (op.GetAttr("strides", strides_list) != GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "strides");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr strides_list failed!");
    return GRAPH_FAILED;
  }
  if (strides_list.size() != 4) {
    OpsAttrValueErrReport(op.GetName(), "strides", ConcatString(4), ConcatString(strides_list.size()));
    OP_LOGE(op.GetName().c_str(), "Length of strides must be equal to the length of shape!");
    return GRAPH_FAILED;
  }
  for (auto i = 0; i < strides_list.size(); i++) {
    if (strides_list[i] == 0) {
      OP_LOGE(op.GetName().c_str(), "strides_list has element which equals to 0");
      return GRAPH_FAILED;
    }
  }
  // get input padding_mode
  std::string padding_mode;
  if (op.GetAttr("padding", padding_mode) != GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "padding");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr padding failed!");
    return GRAPH_FAILED;
  }

  if (padding_mode != "SAME" && padding_mode != "VALID") {
    string excepted_value = ConcatString("SAME, VALID");
    OpsAttrValueErrReport(op.GetName(), "padding_mode", excepted_value, padding_mode);
    OP_LOGE(op.GetName().c_str(), "MaxPoolGradGradWithArgmax can only support SAME or VALID padding mode!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dims_input = shape.GetDims();
  // set output shape
  std::vector<int64_t> dim_vector;
  if (input_format == FORMAT_NHWC) {
    if (padding_mode == "SAME") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == 1) || (i == 2)) {
          int64_t dims = (dims_input[i] + strides_list[i] - 1) / strides_list[i];
          dim_vector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dim_vector.push_back(dims);
        }
      }
    } else {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == 1) || (i == 2)) {
          int64_t dims = (dims_input[i] - ksize_list[i] + 1 + (strides_list[i] - 1)) / strides_list[i];
          dim_vector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dim_vector.push_back(dims);
        }
      }
    }
  } else {
    if (padding_mode == "SAME") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == 2) || (i == 3)) {
          int64_t dims = (dims_input[i] + strides_list[i - 1] - 1) / strides_list[i - 1];
          dim_vector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dim_vector.push_back(dims);
        }
      }
    } else {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == 2) || (i == 3)) {
          int64_t dims = (dims_input[i] - ksize_list[i - 1] + 1 + (strides_list[i - 1] - 1)) / strides_list[i - 1];
          dim_vector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dim_vector.push_back(dims);
        }
      }
    }
  }
  TensorDesc out_tensor_desc = op.GetOutputDesc("y");
  DataType input_dtype = input_tensor_desc.GetDataType();
  Shape output_shape(dim_vector);
  out_tensor_desc.SetShape(output_shape);
  out_tensor_desc.SetDataType(input_dtype);
  if (op.UpdateOutputDesc("y", out_tensor_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MaxPoolGradGradWithArgmax, MaxPoolGradGradWithArgmaxInferShape);
VERIFY_FUNC_REG(MaxPoolGradGradWithArgmax, MaxPoolGradGradWithArgmaxVerify);
// ----------------------MaxPoolGradGradWithArgmax-----------------------

static bool GetDimInFormat(const std::string& format_str, const std::string& dim_name,
                           std::string::size_type& dim_position) {
  dim_position = format_str.find(dim_name);
  return dim_position != std::string::npos;
}

static bool GetOriOutput(ge::Operator& op, std:: vector<std::string::size_type> position,
                         std::vector<int64_t> input_shape, std::vector<int64_t> filter, std::vector<int64_t> strides,
                         std::string pad_str, std::vector<int64_t>& ori_output_shape) {
  int64_t input_h = 0;
  int64_t input_w = 0;
  int64_t filter_h = 0;
  int64_t filter_w = 0;
  int64_t stride_h = 0;
  int64_t stride_w = 0;
  int64_t ori_output_h = 0;
  int64_t ori_output_w = 0;

  if (position[2] < input_shape.size() && position[3] < input_shape.size()) {
    input_h = input_shape.at(position[2]);
    input_w = input_shape.at(position[3]);
  } else {
    OP_LOGE(op.GetName().c_str(), "Input shape subscript out of bounds!");
    return false;
  }

  if (position[2] < filter.size() && position[3] < filter.size()) {
    filter_h = filter.at(position[2]);
    filter_w = filter.at(position[3]);
  } else {
    OP_LOGE(op.GetName().c_str(), "Filter subscript out of bounds!");
    return false;
  }

  if (position[2] < strides.size() && position[3] < strides.size()) {
    stride_h = strides.at(position[2]);
    stride_w = strides.at(position[3]);
  } else {
    OP_LOGE(op.GetName().c_str(), "Strides subscript out of bounds!");
    return false;
  }

  if (stride_h != 0 && stride_w != 0) {
    if (pad_str.compare("SAME") == 0) {
      ori_output_h = (input_h + stride_h - 1) / stride_h;
      ori_output_w = (input_w + stride_w - 1) / stride_w;
    } else if (pad_str.compare("VALID") == 0) {
      ori_output_h = (input_h - filter_h + stride_h) / stride_h;
      ori_output_w = (input_w - filter_w + stride_w) / stride_w;
    } else {
      OP_LOGE(op.GetName().c_str(),
              "Padding should be SAME or VALID. Actual is: %s.", pad_str.c_str());
      return false;
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "Strides cannot be 0!");
    return false;
  }

  ori_output_shape.clear();
  ori_output_shape = {input_shape.at(position[0]), input_shape.at(position[1]), ori_output_h, ori_output_w};

  return true;
}

// ---------------------AvgPoolGrad---------------------
IMPLEMT_VERIFIER(AvgPoolGrad, AvgPoolGradVerify) {
  Tensor orig_input_shape_tensor;
  if (op.GetInputConstData("orig_input_shape", orig_input_shape_tensor) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get constdata filed");
    return GRAPH_FAILED;
  }

  DataType dtype = op.GetInputDesc("orig_input_shape").GetDataType();
  std::vector<int64_t> orig_input_size;
  GetConstValue(op, orig_input_shape_tensor, dtype, orig_input_size);
  if (!CheckListEmpty(op.GetName(), orig_input_size, "orig_input_shape")) {
    return GRAPH_FAILED;
  }
  std::vector<int64_t> ksize;
  ksize = GetAttrValue(op, "ksize");
  if (!CheckListEmpty(op.GetName(), ksize, "ksize")) {
    return GRAPH_FAILED;
  }
  if (ksize.size() < 4) {
    OP_LOGE(op.GetName().c_str(), "Attr ksize(%u) is too small", ksize.size());
  }
  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), strides, "strides")) {
    return GRAPH_FAILED;
  }
  if (strides.size() < 4) {
    OP_LOGE(op.GetName().c_str(), "Attr strides(%u) is too small", strides.size());
  }

  std::string padding;
  if (op.GetAttr("padding", padding) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get padding failed!");
    return GRAPH_FAILED;
  }
  if (padding != "SAME" && padding != "VALID") {
    OP_LOGE(op.GetName().c_str(), "Attr padding(%s) only support SAME and VALID", padding.c_str());
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_SUCCESS) {
    if (data_format != "NCHW" && data_format != "NHWC") {
      OP_LOGE(op.GetName().c_str(), "Attr data_format(%s) only support NCHW and NHWC", data_format.c_str());
      return GRAPH_FAILED;
    }
  }

  TensorDesc tensordesc_input = op.GetInputDesc("input_grad");
  Shape input_grad_shape = tensordesc_input.GetShape();

  string::size_type n_position{0};
  string::size_type c_position{0};
  string::size_type h_position{0};
  string::size_type w_position{0};

  if (!GetDimInFormat(data_format, "N", n_position)) {
    return GRAPH_FAILED;
  }

  if (!GetDimInFormat(data_format, "C", c_position)) {
    return GRAPH_FAILED;
  }

  if (!GetDimInFormat(data_format, "H", h_position)) {
    return GRAPH_FAILED;
  }

  if (!GetDimInFormat(data_format, "W", w_position)) {
    return GRAPH_FAILED;
  }

  std::vector<std::string::size_type> position{n_position, c_position, h_position, w_position};
  std::vector<int64_t> ori_output_shape;

  if (!GetOriOutput(op, position, orig_input_size, ksize, strides, padding, ori_output_shape)) {
    OP_LOGE(op.GetName().c_str(), "Get origin output failed.");
    return GRAPH_FAILED;
  }

  if (input_grad_shape.GetDim(n_position) != ori_output_shape[0] ||
      input_grad_shape.GetDim(c_position) != ori_output_shape[1] ||
      input_grad_shape.GetDim(h_position) != ori_output_shape[2] ||
      input_grad_shape.GetDim(w_position) != ori_output_shape[3]) {
    OP_LOGE(op.GetName().c_str(), "Input grad shape is wrong!");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AvgPoolGradInferShape) {
  Tensor orig_input_shape_tensor;
  if (op.GetInputConstData("orig_input_shape", orig_input_shape_tensor) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get constdata failed");
    return GRAPH_FAILED;
  }
  DataType dtype = op.GetInputDesc("orig_input_shape").GetDataType();
  std::vector<int64_t> orig_input_size;
  GetConstValue(op, orig_input_shape_tensor, dtype, orig_input_size);
  DataType output_dtype = op.GetInputDesc("input_grad").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("out_grad");
  tensordesc_output.SetShape(Shape(orig_input_size));
  tensordesc_output.SetDataType(output_dtype);
  if (op.UpdateOutputDesc("out_grad", tensordesc_output) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AvgPoolGrad, AvgPoolGradInferShape);
VERIFY_FUNC_REG(AvgPoolGrad, AvgPoolGradVerify);
// ---------------------AvgPoolGrad---------------------

// ---------------------AvgPoolGradD---------------------
IMPLEMT_VERIFIER(AvgPoolGradD, AvgPoolGradDVerify) {
  std::vector<int64_t> orig_input_size;
  orig_input_size = GetAttrValue(op, "orig_input_shape");
  if (!CheckListEmpty(op.GetName(), orig_input_size, "orig_input_shape")) {
    return GRAPH_FAILED;
  }
  std::vector<int64_t> ksize;
  ksize = GetAttrValue(op, "ksize");
  if (!CheckListEmpty(op.GetName(), ksize, "ksize")) {
    return GRAPH_FAILED;
  }
  if (ksize.size() < 4) {
    OP_LOGE(op.GetName().c_str(), "Attr ksize(%u) is too small", ksize.size());
  }
  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), strides, "strides")) {
    return GRAPH_FAILED;
  }

  std::string padding;
  if (op.GetAttr("padding", padding) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get padding failed!");
    return GRAPH_FAILED;
  }
  if (padding != "SAME" && padding != "VALID") {
    OP_LOGE(op.GetName().c_str(), "Attr padding(%s) must in SAME and VALID", padding.c_str());
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_SUCCESS) {
    if (data_format != "NCHW" && data_format != "NHWC") {
      OP_LOGE(op.GetName().c_str(), "Attr data_format(%s) only support NCHW and NHWC", data_format.c_str());
      return GRAPH_FAILED;
    }
  }

  TensorDesc tensordesc_input = op.GetInputDesc("input_grad");
  Shape input_grad_shape = tensordesc_input.GetShape();

  string::size_type n_position{0};
  string::size_type c_position{0};
  string::size_type h_position{0};
  string::size_type w_position{0};

  if (!GetDimInFormat(data_format, "N", n_position)) {
    return GRAPH_FAILED;
  }

  if (!GetDimInFormat(data_format, "C", c_position)) {
    return GRAPH_FAILED;
  }

  if (!GetDimInFormat(data_format, "H", h_position)) {
    return GRAPH_FAILED;
  }

  if (!GetDimInFormat(data_format, "W", w_position)) {
    return GRAPH_FAILED;
  }

  std::vector<std::string::size_type> position{n_position, c_position, h_position, w_position};
  std::vector<int64_t> ori_output_shape;

  if (!GetOriOutput(op, position, orig_input_size, ksize, strides, padding, ori_output_shape)) {
    OP_LOGE(op.GetName().c_str(), "Get origin output failed.");
    return GRAPH_FAILED;
  }

  if (input_grad_shape.GetDim(n_position) != ori_output_shape[0] ||
      input_grad_shape.GetDim(c_position) != ori_output_shape[1] ||
      input_grad_shape.GetDim(h_position) != ori_output_shape[2] ||
      input_grad_shape.GetDim(w_position) != ori_output_shape[3]) {
    OP_LOGE(op.GetName().c_str(), "Input grad shape is wrong!");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AvgPoolGradDInferShape) {
  std::vector<int64_t> orig_input_size;
  orig_input_size = GetAttrValue(op, "orig_input_shape");

  TensorDesc tensordesc_input = op.GetInputDesc("input_grad");
  Shape input_grad_shape = tensordesc_input.GetShape();
  DataType input_grad_dtype = tensordesc_input.GetDataType();

  std::vector<int64_t> ksize;
  ksize = GetAttrValue(op, "ksize");
  if (!CheckListEmpty(op.GetName(), ksize, "ksize")) {
    return GRAPH_FAILED;
  }
  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), strides, "strides")) {
    return GRAPH_FAILED;
  }
  std::string padding;
  if (op.GetAttr("padding", padding) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get padding failed!");
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (op.GetAttr("data_format", data_format) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get data_format failed!");
    return GRAPH_FAILED;
  }

  string::size_type n_position{0};
  string::size_type c_position{0};
  string::size_type h_position{0};
  string::size_type w_position{0};
  int64_t kernel_n{0};
  int64_t kernel_c{0};
  int64_t kernel_h{0};
  int64_t kernel_w{0};

  if (!GetDimInFormat(data_format, "N", n_position)) {
    return GRAPH_FAILED;
  }

  if (!GetDimInFormat(data_format, "C", c_position)) {
    return GRAPH_FAILED;
  }

  if (!GetDimInFormat(data_format, "H", h_position)) {
    return GRAPH_FAILED;
  }

  if (!GetDimInFormat(data_format, "W", w_position)) {
    return GRAPH_FAILED;
  }

  TensorDesc tensordesc_mean = op.GetInputDesc("mean_matrix");
  if (data_format == "NHWC") {
    tensordesc_mean.SetFormat(FORMAT_NHWC);
    tensordesc_mean.SetOriginFormat(FORMAT_NHWC);
  } else if (data_format == "NCHW") {
    tensordesc_mean.SetFormat(FORMAT_NCHW);
    tensordesc_mean.SetOriginFormat(FORMAT_NCHW);
  } else {
    OP_LOGE(op.GetName().c_str(), "Attr data_format(%s) only support NCHW and NHWC", data_format.c_str());
  }

  tensordesc_mean.SetShape(input_grad_shape);
  tensordesc_mean.SetOriginShape(input_grad_shape);
  tensordesc_mean.SetDataType(input_grad_dtype);
  if (op.UpdateInputDesc("mean_matrix", tensordesc_mean) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Fail to update input mean_matrix!");
    return GRAPH_FAILED;
  }

  kernel_h = ksize.at(h_position);
  kernel_w = ksize.at(w_position);
  kernel_c = orig_input_size.at(c_position);
  kernel_n = 1;
  vector<int64_t> kernel_shape{kernel_h, kernel_w, kernel_c, kernel_n};

  TensorDesc tensordesc_kernel = op.GetInputDesc("kernel_matrix");
  tensordesc_kernel.SetShape(Shape(kernel_shape));
  tensordesc_kernel.SetOriginShape(Shape(kernel_shape));
  tensordesc_kernel.SetOriginFormat(FORMAT_HWCN);
  tensordesc_kernel.SetFormat(FORMAT_HWCN);
  tensordesc_kernel.SetDataType(input_grad_dtype);
  if (op.UpdateInputDesc("kernel_matrix", tensordesc_kernel) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Fail to update input kernel_matrix!");
    return GRAPH_FAILED;
  }

  TensorDesc tensordesc_output = op.GetOutputDesc("out_grad");
  tensordesc_output.SetShape(Shape(orig_input_size));
  tensordesc_output.SetDataType(input_grad_dtype);
  if (op.UpdateOutputDesc("out_grad", tensordesc_output) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Fail to update output out_grad!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AvgPoolGradD, AvgPoolGradDInferShape);
VERIFY_FUNC_REG(AvgPoolGradD, AvgPoolGradDVerify);
// ---------------------AvgPoolGradD end---------------------

IMPLEMT_VERIFIER(AvgPoolV2Grad, AvgPoolV2GradVerify) {
  Tensor orig_input_shape_tensor;
  if (GRAPH_SUCCESS != op.GetInputConstData("orig_input_shape", orig_input_shape_tensor)) {
    OP_LOGE(op.GetName().c_str(), "get constdata filed");
    return GRAPH_FAILED;
  }
  DataType dtype = op.GetInputDesc("orig_input_shape").GetDataType();

  std::vector<int64_t> orig_input_size;
  GetConstValue(op, orig_input_shape_tensor, dtype, orig_input_size);
  if (!CheckListEmpty(op.GetName(), orig_input_size, "orig_input_shape")) {
    OP_LOGE(op.GetName().c_str(), "orig_input_shape is empty!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> ksize;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksize)) {
    OpsGetAttrErrReport(op.GetName(), "ksize");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ksize failed!");
    return GRAPH_FAILED;
  }
  if (ksize.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "Size of ksize must be 4!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> strides;
  if (GRAPH_SUCCESS != op.GetAttr("strides", strides)) {
    OpsGetAttrErrReport(op.GetName(), "strides");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr strides failed!");
    return GRAPH_FAILED;
  }
  if (strides.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "Size of strides must be 4!");
    return GRAPH_FAILED;
  }

  std::string padding_mode;
  if (ge::GRAPH_SUCCESS != op.GetAttr("padding_mode", padding_mode)) {
    OpsGetAttrErrReport(op.GetName(), "padding_mode");
    OP_LOGE(op.GetName().c_str(), "Get padding_mode failed!");
    return GRAPH_FAILED;
  }
  if (padding_mode != "SAME" && padding_mode != "VALID" && padding_mode != "CALCULATED") {
    OP_LOGE(op.GetName().c_str(), "attr padding_mode(%s) only support SAME VALID and CALCULATED", padding_mode.c_str());
    return GRAPH_FAILED;
  }

  std::vector<int64_t> pads;
  if (GRAPH_SUCCESS != op.GetAttr("pads", pads)) {
    OpsGetAttrErrReport(op.GetName(), "pads");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr pads failed!");
    return GRAPH_FAILED;
  }
  if (pads.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "Size of pads must be 4!");
    return GRAPH_FAILED;
  }

  // get input ceilMode
  bool ceilMode;
  if (GRAPH_SUCCESS != op.GetAttr("ceil_mode", ceilMode)) {
    OpsGetAttrErrReport(op.GetName(), "ceil_mode");
    OP_LOGE(op.GetName().c_str(), "The AvgPoolV2Grad op GetOpAttr ceil_mode failed!");
    return GRAPH_FAILED;
  }

  if (ceilMode && padding_mode == "VALID") {
    OP_LOGE(op.GetName().c_str(),
            "When Attr(padding_mode) is VALID, Attr(ceil_mode) must be False. Received ceil_mode: True.");
  }

  std::string data_format;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", data_format)) {
    OpsGetAttrErrReport(op.GetName(), "data_format");
    OP_LOGE(op.GetName().c_str(),
            "The AvgPoolV2Grad op GetOpAttr data_format failed!");
    return GRAPH_FAILED;
  }

  if (data_format != "NCHW" && data_format != "NHWC") {
    OP_LOGE(op.GetName().c_str(), "attr data_format(%s) only support NCHW and NHWC", data_format.c_str());
    return GRAPH_FAILED;
  }

  if (data_format == "NCHW") {
    if (ksize[0] != 1 || ksize[1] != 1 || strides[0] != 1 || strides[1] != 1) {
      OP_LOGE(op.GetName().c_str(),
              "AvgPoolV2Grad only supports pooling across width/height"
              "and other ksize dimension should be one");
      return GRAPH_FAILED;
    }
    if (padding_mode == "CALCULATED" &&
        (pads[0] >= ksize[2] || pads[1] >= ksize[2] || pads[2] >= ksize[3] || pads[3] >= ksize[3])) {
      OP_LOGE(op.GetName().c_str(), "Pads must be less then ksize when using CALCULATED mode!");
      return GRAPH_FAILED;
    }
  }

  if (data_format == "NHWC") {
    if (ksize[0] != 1 || ksize[3] != 1 || strides[0] != 1 || strides[3] != 1) {
      OP_LOGE(op.GetName().c_str(),
              "AvgPoolV2Grad only supports pooling across width/height"
              "and other ksize dimension should be one");
      return GRAPH_FAILED;
    }
    if (padding_mode == "CALCULATED" &&
        (pads[0] >= ksize[1] || pads[1] >= ksize[1] || pads[2] >= ksize[2] || pads[3] >= ksize[2])) {
      OP_LOGE(op.GetName().c_str(), "Pads must be less then ksize when using CALCULATED mode!");
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AvgPoolV2GradInferShape) {
  Tensor orig_input_shape_tensor;
  if (GRAPH_SUCCESS != op.GetInputConstData("orig_input_shape", orig_input_shape_tensor)) {
    return GRAPH_FAILED;
  }
  DataType dtype = op.GetInputDesc("orig_input_shape").GetDataType();

  std::vector<int64_t> orig_input_size;
  GetConstValue(op, orig_input_shape_tensor, dtype, orig_input_size);
  DataType output_dtype = op.GetInputDesc("input_grad").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("out_grad");
  tensordesc_output.SetShape(Shape(orig_input_size));
  tensordesc_output.SetDataType(output_dtype);
  (void)op.UpdateOutputDesc("out_grad", tensordesc_output);
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(AvgPoolV2Grad, AvgPoolV2GradInferShape);
// Registered verify function
VERIFY_FUNC_REG(AvgPoolV2Grad, AvgPoolV2GradVerify);
// ----------------AvgPoolV2Grad-------------------

IMPLEMT_VERIFIER(AvgPoolV2GradD, AvgPoolV2GradDVerify) {
  // get attr orig_input_shape
  std::vector<int64_t> orig_input_shape;
  if (GRAPH_SUCCESS != op.GetAttr("orig_input_shape", orig_input_shape)) {
    OpsGetAttrErrReport(op.GetName(), "orig_input_shape");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr orig_input_shape failed!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> ksize;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksize)) {
    OpsGetAttrErrReport(op.GetName(), "ksize");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ksize failed!");
    return GRAPH_FAILED;
  }
  if (ksize.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "Size of ksize must be 4!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> strides;
  if (GRAPH_SUCCESS != op.GetAttr("strides", strides)) {
    OpsGetAttrErrReport(op.GetName(), "strides");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr strides failed!");
    return GRAPH_FAILED;
  }
  if (strides.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "Size of strides must be 4!");
    return GRAPH_FAILED;
  }

  std::string padding_mode;
  if (ge::GRAPH_SUCCESS != op.GetAttr("padding_mode", padding_mode)) {
    OpsGetAttrErrReport(op.GetName(), "padding_mode");
    OP_LOGE(op.GetName().c_str(), "Get padding_mode failed!");
    return GRAPH_FAILED;
  }
  if (padding_mode != "SAME" && padding_mode != "VALID" && padding_mode != "CALCULATED") {
    OP_LOGE(op.GetName().c_str(), "attr padding_mode(%s) only support SAME VALID and CALCULATED", padding_mode.c_str());
    return GRAPH_FAILED;
  }

  std::vector<int64_t> pads;
  if (GRAPH_SUCCESS != op.GetAttr("pads", pads)) {
    OpsGetAttrErrReport(op.GetName(), "pads");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr pads failed!");
    return GRAPH_FAILED;
  }
  if (pads.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "Size of pads must be 4!");
    return GRAPH_FAILED;
  }

  // get input ceilMode
  bool ceilMode;
  if (GRAPH_SUCCESS != op.GetAttr("ceil_mode", ceilMode)) {
    OpsGetAttrErrReport(op.GetName(), "ceil_mode");
    OP_LOGE(op.GetName().c_str(), "The AvgPoolV2GradD op GetOpAttr ceil_mode failed!");
    return GRAPH_FAILED;
  }

  if (ceilMode && padding_mode == "VALID") {
    OP_LOGE(op.GetName().c_str(),
            "When Attr(padding_mode) is VALID, Attr(ceil_mode) must be False. Received ceil_mode: True.");
  }

  std::string data_format;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", data_format)) {
    OpsGetAttrErrReport(op.GetName(), "data_format");
    OP_LOGE(op.GetName().c_str(),
            "The AvgPoolV2GradD op GetOpAttr data_format failed!");
    return GRAPH_FAILED;
  }

  if (data_format != "NCHW" && data_format != "NHWC") {
    OP_LOGE(op.GetName().c_str(), "attr data_format(%s) only support NCHW and NHWC", data_format.c_str());
    return GRAPH_FAILED;
  }

  if (data_format == "NCHW") {
    if (ksize[0] != 1 || ksize[1] != 1 || strides[0] != 1 || strides[1] != 1) {
      OP_LOGE(op.GetName().c_str(),
              "AvgPoolV2GradD only supports pooling across width/height"
              "and other ksize dimension should be one");
      return GRAPH_FAILED;
    }
    if (padding_mode == "CALCULATED" &&
        (pads[0] >= ksize[2] || pads[1] >= ksize[2] || pads[2] >= ksize[3] || pads[3] >= ksize[3])) {
      OP_LOGE(op.GetName().c_str(), "Pads must be less then ksize when using CALCULATED mode!");
      return GRAPH_FAILED;
    }
  }

  if (data_format == "NHWC") {
    if (ksize[0] != 1 || ksize[3] != 1 || strides[0] != 1 || strides[3] != 1) {
      OP_LOGE(op.GetName().c_str(),
              "AvgPoolV2GradD only supports pooling across width/height"
              "and other ksize dimension should be one");
      return GRAPH_FAILED;
    }
    if (padding_mode == "CALCULATED" &&
        (pads[0] >= ksize[1] || pads[1] >= ksize[1] || pads[2] >= ksize[2] || pads[3] >= ksize[2])) {
      OP_LOGE(op.GetName().c_str(), "Pads must be less then ksize when using CALCULATED mode!");
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AvgPoolV2GradDInferShape) {
  // get attr orig_input_size
  std::vector<int64_t> orig_input_size;
  orig_input_size = GetAttrValue(op, "orig_input_shape");
  DataType output_dtype = op.GetInputDesc("input_grad").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("out_grad");
  tensordesc_output.SetShape(Shape(orig_input_size));
  tensordesc_output.SetDataType(output_dtype);
  (void)op.UpdateOutputDesc("out_grad", tensordesc_output);
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(AvgPoolV2GradD, AvgPoolV2GradDInferShape);
// Registered verify function
VERIFY_FUNC_REG(AvgPoolV2GradD, AvgPoolV2GradDVerify);
// ----------------AvgPoolV2GradD-------------------

IMPLEMT_VERIFIER(Upsample, UpsampleVerify) {
  return GRAPH_SUCCESS;
}
IMPLEMT_INFERFUNC(Upsample, UpsampleInferShape) {
  TensorDesc tensordesc_output = op.GetInputDesc("x");
  uint32_t stride_h = 2;
  uint32_t stride_w = 2;
  if (op.GetAttr("stride_h", stride_h) != ge::GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "GetOpAttr stride failed, set stride_h default value");
    stride_h = 2;
  }
  if (op.GetAttr("stride_w", stride_w) != ge::GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "GetOpAttr stride failed, set stride_w default value");
    stride_w = 2;
  }
  ge::Shape shape = tensordesc_output.GetShape();
  std::vector<int64_t> dims_input = shape.GetDims();
  std::vector<int64_t> dimVector;
  for (size_t i = 0; i < dims_input.size(); i++) {
    if (i == 2) {
      int64_t dims = dims_input[i] * stride_h;
      dimVector.push_back(dims);
    } else if (i == 3) {
      int64_t dims = dims_input[i] * stride_w;
      dimVector.push_back(dims);
    } else {
      int64_t dims = dims_input[i];
      dimVector.push_back(dims);
    }
  }

  Shape outputMaxShape(dimVector);
  tensordesc_output.SetShape(outputMaxShape);
  (void)op.UpdateOutputDesc("y", tensordesc_output);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Upsample, UpsampleInferShape);
VERIFY_FUNC_REG(Upsample, UpsampleVerify);

IMPLEMT_INFERFUNC(FractionalMaxPoolGrad, FractionalMaxPoolGradInfer) {
  Shape input_shape;
  if (WithRank(op.GetInputDesc(0), 4, input_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    ShapeErrReport(0, op.GetName(), DebugString(op.GetInputDesc(0).GetShape().GetDims()), "4D");
    return GRAPH_FAILED;
  }

  auto type = op.GetInputDesc("orig_input").GetDataType();
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(Shape(input_shape));
  output_desc.SetDataType(type);
  if (op.UpdateOutputDesc("y", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Update y desc failed.");
    OpsOPUpdateErrReport(op.GetName(), "y");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(FractionalMaxPoolGrad, FractionalMaxPoolGradInfer);

IMPLEMT_INFERFUNC(FractionalAvgPool, FractionalAvgPoolInfer) {
  Shape input;
  if (WithRank(op.GetInputDesc(0), 4, input, op.GetName().c_str()) != GRAPH_SUCCESS) {
    ShapeErrReport(0, op.GetName(), DebugString(op.GetInputDesc(0).GetShape().GetDims()), "4D");
    return GRAPH_FAILED;
  }
  std::vector<float> pooling_ratio;
  op.GetAttr("pooling_ratio", pooling_ratio);
  if (pooling_ratio.size() != 4) {
    AttrSizeErrReport("pooling_ratio", op.GetName(), ConcatString(pooling_ratio.size()), "4");
    OP_LOGE(op.GetName().c_str(), "pooling_ratio field must specify 4 dimensions.");
    return GRAPH_PARAM_INVALID;
  }
  auto x_dims = op.GetInputDesc(0).GetShape().GetDims();
  std::vector<int64_t> dims;
  dims.reserve(4);
  for (int i = 0; i < 4; ++i) {
    auto val = static_cast<int64_t>(x_dims[i] / pooling_ratio[i]);
    if (val < 0) {
      string err_msg = ConcatString("size computed for ", i, "th dim is ", val, ", please check");
      OP_LOGE(op.GetName().c_str(), "%s.", err_msg.c_str());
      InferShapeOtherErrReport(op.GetName(), err_msg);
      return GRAPH_PARAM_INVALID;
    }
    dims.push_back(val);
  }
  Shape out(dims);
  Shape row_pooling_sequence;
  (void)Vector(dims[1] + 1, row_pooling_sequence);
  Shape col_pooling_sequence;
  (void)Vector(dims[2] + 1, col_pooling_sequence);

  DataType type = op.GetInputDesc("x").GetDataType();

  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(out);
  y_desc.SetDataType(type);
  op.UpdateOutputDesc("y", y_desc);

  TensorDesc row_desc = op.GetOutputDesc("row_pooling_sequence");
  row_desc.SetShape(row_pooling_sequence);
  row_desc.SetDataType(DT_INT64);
  op.UpdateOutputDesc("row_pooling_sequence", row_desc);

  TensorDesc col_desc = op.GetOutputDesc("col_pooling_sequence");
  col_desc.SetShape(col_pooling_sequence);
  col_desc.SetDataType(DT_INT64);
  op.UpdateOutputDesc("col_pooling_sequence", col_desc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(FractionalAvgPool, FractionalAvgPoolInfer);

IMPLEMT_INFERFUNC(FractionalMaxPool, FractionalMaxPoolInfer) {
  auto tensor = op.get_input_desc_x();
  Shape input_value;

  if (WithRank(tensor, 4, input_value, op.GetName().c_str()) != GRAPH_SUCCESS) {
    ShapeErrReport(0, op.GetName(), DebugString(tensor.GetShape().GetDims()), "4D");
    OP_LOGE(op.GetName().c_str(), "input value must be 4-D.");
    return GRAPH_FAILED;
  }
  std::vector<float> pooling_ratio;
  pooling_ratio = op.get_attr_pooling_ratio();
  if (pooling_ratio.size() != 4) {
    AttrSizeErrReport("pooling_ratio", op.GetName(), ConcatString(pooling_ratio.size()), "4");
    OP_LOGE(op.GetName().c_str(), "pooling_ratio field must specify 4-D.");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> output_dims;
  for (int i = 0; i < 4; ++i) {
    int64_t dim = input_value.GetDim(i);
    if (dim != UNKNOWN_DIM) {
      auto real_dim = static_cast<int64_t>(dim / pooling_ratio[i]);
      if (real_dim < 0) {
        string err_msg = ConcatString("size computed for ", i, "th dim is ", real_dim, ", please check");
        OP_LOGE(op.GetName().c_str(), "%s.", err_msg.c_str());
        InferShapeOtherErrReport(op.GetName(), err_msg);
        return GRAPH_FAILED;
      }
      output_dims.push_back(real_dim);
    } else {
      output_dims.push_back(UNKNOWN_DIM);
    }
  }

  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(Shape(output_dims));
  y_desc.SetDataType(op.GetInputDesc("x").GetDataType());
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "fail to update output y.");
    return GRAPH_FAILED;
  }

  TensorDesc row_pooling_desc = op.GetOutputDesc("row_pooling_sequence");
  row_pooling_desc.SetShape(Shape({output_dims[1] + 1}));
  row_pooling_desc.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("row_pooling_sequence", row_pooling_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "fail to  update output row_pooling_sequence.");
    return GRAPH_FAILED;
  }

  TensorDesc col_pooling_desc = op.GetOutputDesc("col_pooling_sequence");
  col_pooling_desc.SetShape(Shape({output_dims[2] + 1}));
  col_pooling_desc.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("col_pooling_sequence", col_pooling_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "fail to update output col_pooling_sequence.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(FractionalMaxPool, FractionalMaxPoolInfer);

IMPLEMT_INFERFUNC(DataFormatVecPermute, DataFormatVecPermuteInfer) {
  DataType y_type = op.GetInputDesc("x").GetDataType();
  TensorDesc desc = op.GetOutputDesc("y");
  desc.SetShape(op.GetInputDesc("x").GetShape());
  desc.SetDataType(y_type);

  return op.UpdateOutputDesc("y", desc);
}

INFER_FUNC_REG(DataFormatVecPermute, DataFormatVecPermuteInfer);

IMPLEMT_INFERFUNC(FractionalAvgPoolGrad, FractionalAvgPoolGradInfer) {
  Tensor tensor;
  if (op.GetInputConstData("orig_input_tensor_shape", tensor) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input orig_input_tensor_shape GetInputConstData failed");
    return GRAPH_FAILED;
  }

  Shape result;
  if (MakeShapeFromShapeTensor(tensor, result, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Make shape from ShapeTensor failed");
    return GRAPH_FAILED;
  }

  DataType y_type = op.GetInputDesc("out_backprop").GetDataType();
  TensorDesc out_desc = op.GetOutputDesc("y");
  out_desc.SetShape(Shape(result));
  out_desc.SetDataType(y_type);
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update y desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(FractionalAvgPoolGrad, FractionalAvgPoolGradInfer);

IMPLEMT_INFERFUNC(NthElement, NthElementInfer) {
  Shape x_shape;
  auto x_tensor = op.get_input_desc_x();
  if (WithRankAtLeast(x_tensor, 1, x_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    ShapeErrReport(0, op.GetName(), DebugString(op.GetInputDesc(0).GetShape().GetDims()), "at least 1D");
    OP_LOGE(op.GetName().c_str(), "The rank of input x must be at least 1.");
    return GRAPH_FAILED;
  }

  Tensor n_tensor;
  if (op.GetInputConstData("n", n_tensor) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get input n value error.");
    return GRAPH_FAILED;
  }

  int64_t n_dim = 0;
  if (MakeDimForScalarInput(n_tensor, n_dim, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "MakeDimForScalarInput get input n value error.");
    return GRAPH_FAILED;
  }

  int64_t existing = x_shape.GetDimNum();
  int64_t last_input_dim = x_shape.GetDim(existing - 1);
  if ((last_input_dim != ge::UNKNOWN_DIM) && (n_dim != ge::UNKNOWN_DIM) && (last_input_dim <= n_dim)) {
    OP_LOGE(op.GetName().c_str(), "Input must have last dim > n=%lld,but inputLastDim is %lld", n_dim, last_input_dim);
    return GRAPH_FAILED;
  }

  Shape output_shape;
  if (SubShape(x_shape, 0, -1, 1, output_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get SubShape error.");
    return GRAPH_FAILED;
  }

  TensorDesc y_tensor = op.GetOutputDesc("y");
  y_tensor.SetDataType(x_tensor.GetDataType());
  y_tensor.SetShape(output_shape);
  return op.UpdateOutputDesc("y", y_tensor);
}

INFER_FUNC_REG(NthElement, NthElementInfer);

// ----------------MaxPool3DGrad-------------------
static bool GetPadMaxPool3DGrad(ge::Operator& op, int32_t id, int32_t ih, int32_t iw, int32_t kd, int32_t kh,
                                int32_t kw, int32_t strd, int32_t strh, int32_t strw, int32_t& padf, int32_t& padba,
                                int32_t& padt, int32_t& padb, int32_t& padl, int32_t& padr) {
  std::string padStr;
  std::vector<int32_t> padList;
  if (GRAPH_SUCCESS == op.GetAttr("padding", padStr)) {
    if (padStr.compare("SAME") == 0) {
      int32_t tails_d = id % strd;
      int32_t tails_h = ih % strh;
      int32_t tails_w = iw % strw;
      int32_t pad_d = std::max((tails_d > 0 ? kd - tails_d : kd - strd), 0);
      int32_t pad_h = std::max((tails_h > 0 ? kh - tails_h : kh - strh), 0);
      int32_t pad_w = std::max((tails_w > 0 ? kw - tails_w : kw - strw), 0);
      padList.push_back(pad_d / 2);
      padList.push_back(pad_d / 2 + pad_d % 2);
      padList.push_back(pad_h / 2);
      padList.push_back(pad_h / 2 + pad_h % 2);
      padList.push_back(pad_w / 2);
      padList.push_back(pad_w / 2 + pad_w % 2);
    } else if (padStr.compare("VALID") == 0) {
      for (int32_t i = 0; i < 6; i++)
        padList.push_back(0);
    } else {
      map<string, string> err_map;
      err_map["param_name"] = "padding";
      err_map["op_name"] = "MaxPool3DGrad";
      err_map["Expected_value"] = "SAME or VALID";
      err_map["input_value"] = padStr;
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return false;
    }
    op.SetAttr("pads", padList);
  }
  std::vector<int32_t> padVec;
  if (op.GetAttr("pads", padVec) != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Failed to get pads!");
    return false;
  }
  auto pSize = padVec.size();
  if (pSize != 6) {
    map<string, string> err_map;
    err_map["param_name"] = "pads list";
    err_map["op_name"] = "MaxPool3DGrad";
    err_map["excepted_value"] = "6d";
    err_map["input_value"] = std::to_string(pSize);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  padf = padVec[0];
  padba = padVec[1];
  padt = padVec[2];
  padb = padVec[3];
  padl = padVec[4];
  padr = padVec[5];
  if (padf < 0 || padba < 0 || padt < 0 || padb < 0 || padl < 0 || padr < 0) {
    map<string, string> err_map;
    err_map["param_name"] = "pads_list";
    err_map["op_name"] = "MaxPool3DGrad";
    err_map["excepted_value"] = "positive";
    err_map["input_value"] = std::to_string(padf) + " " + std::to_string(padba) + " " + std::to_string(padt) + " " +
                             std::to_string(padb) + " " + std::to_string(padl) + " " + std::to_string(padr);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  return true;
}

static bool GetAttrsMaxPool3DGrad(ge::Operator& op, Format refer, int32_t& strd, int32_t& strh, int32_t& strw,
                                  int32_t& kd, int32_t& kh, int32_t& kw) {
  std::vector<int32_t> strideList;
  if (GRAPH_SUCCESS != op.GetAttr("strides", strideList)) {
    OP_LOGE(op.GetName().c_str(), "Failed to get strides!");
    return GRAPH_FAILED;
  }
  std::vector<int32_t> ksizeList;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksizeList)) {
    OP_LOGE(op.GetName().c_str(), "Failed to get ksize!");
    return GRAPH_FAILED;
  }

  if (refer == FORMAT_NCDHW) {
    strd = strideList[2];
    strh = strideList[3];
    strw = strideList[4];
    kd = ksizeList[2];
    kh = ksizeList[3];
    kw = ksizeList[4];
  } else if (refer == FORMAT_NDHWC) {
    strd = strideList[1];
    strh = strideList[2];
    strw = strideList[3];
    kd = ksizeList[1];
    kh = ksizeList[2];
    kw = ksizeList[3];
  } else {
    map<string, string> err_map;
    err_map["param_name"] = "refer";
    err_map["op_name"] = "MaxPool3DGrad";
    err_map["excepted_value"] = "NCDHW or NDHWC";
    err_map["input_value"] = refer;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  return true;
}

IMPLEMT_VERIFIER(MaxPool3DGrad, MaxPool3DGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "orig_x", "orig_y") || !CheckTwoInputDtypeSame(op, "orig_x", "grads")) {
    return GRAPH_FAILED;
  }
  std::vector<int64_t> ksize;
  ksize = GetAttrValue(op, "ksize");
  if (!CheckListEmpty(op.GetName(), ksize, "ksize")) {
    OP_LOGE(op.GetName().c_str(), "get attr ksize failed");
    return GRAPH_FAILED;
  }
  if (ksize.size() != 5) {
    OP_LOGE(op.GetName().c_str(), "attr ksize(%d) must be 5", (int)ksize.size());
    return GRAPH_FAILED;
  }
  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), strides, "strides")) {
    OP_LOGE(op.GetName().c_str(), "get attr strides failed");
    return GRAPH_FAILED;
  }
  if (strides.size() != 5) {
    OP_LOGE(op.GetName().c_str(), "attr strides(%d) must be 5", (int)strides.size());
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
    if (data_format != "NDHWC" && data_format != "NCDHW") {
      OP_LOGE(op.GetName().c_str(), "attr data_format(%s) only support NDHWC and NCDHW", data_format.c_str());
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(MaxPool3DGrad, MaxPool3DGradInferShape) {
  auto shapeX1 = op.GetInputDesc("orig_x").GetShape();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(shapeX1);
  td.SetDataType(DT_FLOAT);
  (void)op.UpdateOutputDesc("y", td);

  // get input DHW
  auto xShape = op.GetInputDesc("orig_x").GetShape().GetDims();
  auto xFormat = op.GetInputDesc("orig_x").GetFormat();
  int32_t id = 0;
  int32_t ih = 0;
  int32_t iw = 0;

  if (xFormat == FORMAT_NCDHW) {
    id = xShape[2];
    ih = xShape[3];
    iw = xShape[4];
  } else if (xFormat == FORMAT_NDHWC) {
    id = xShape[1];
    ih = xShape[2];
    iw = xShape[3];
  } else {
    map<string, string> err_map;
    err_map["param_name"] = "xFormat";
    err_map["op_name"] = "MaxPool3DGrad";
    err_map["excepted_value"] = "NCDHW or NDHWC";
    err_map["input_value"] = xFormat;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // get ksize and strides
  int32_t kd = 0;
  int32_t kh = 0;
  int32_t kw = 0;
  int32_t strd = 0;
  int32_t strh = 0;
  int32_t strw = 0;
  if (false == GetAttrsMaxPool3DGrad(op, xFormat, strd, strh, strw, kd, kh, kw)) {
    OP_LOGE(op.GetName().c_str(), "get attrs failed.");
    return GRAPH_FAILED;
  }
  // get pad
  int32_t padf = 0;
  int32_t padba = 0;
  int32_t padt = 0;
  int32_t padb = 0;
  int32_t padl = 0;
  int32_t padr = 0;
  if ((strd == 0) || (strh == 0) || (strw == 0)) {
    OP_LOGE(op.GetName().c_str(), "strd/strh/strw should not be zero");
    return GRAPH_FAILED;
  }
  if (false == GetPadMaxPool3DGrad(op, id, ih, iw, kd, kh, kw, strd, strh, strw, padf, padba, padt, padb, padl, padr)) {
    OP_LOGE(op.GetName().c_str(), "get pads attrs failed.");
    return GRAPH_FAILED;
  }

  OP_LOGD(op.GetName().c_str(), "Leave MaxPool3DGrad.");
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MaxPool3DGrad, MaxPool3DGradInferShape);
VERIFY_FUNC_REG(MaxPool3DGrad, MaxPool3DGradVerify);
// ---------------------MaxPool3DGrad---------------------

// ----------------AvgPool1D-------------------
IMPLEMT_VERIFIER(AvgPool1D, AvgPool1DVerify) {
  std::vector<int64_t> pads;
  pads = GetAttrValue(op, "pads");
  if (!CheckListEmpty(op.GetName(), pads, "pads")) {
    return GRAPH_FAILED;
  }
  int64_t k_size;
  if (op.GetAttr("ksize", k_size) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get ksize failed!");
    return GRAPH_FAILED;
  }
  int64_t strides;
  if (op.GetAttr("strides", strides) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get strides failed!");
    return GRAPH_FAILED;
  }
  bool ceil_mode{false};
  if (op.GetAttr("ceil_mode", ceil_mode) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get ceil_mode failed!");
    return GRAPH_FAILED;
  }
  bool count_include_pad{false};
  if (op.GetAttr("count_include_pad", count_include_pad) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get count_include_pad failed!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
IMPLEMT_INFERFUNC(AvgPool1D, AvgPool1DInfer) {
  TensorDesc output_tensor_desc = op.GetOutputDesc("y");
  auto input_tensor = op.GetInputDesc("x");
  Format input_format = input_tensor.GetFormat();
  auto input_shape = input_tensor.GetShape();
  int64_t input_w_size{0};

  if (input_format == FORMAT_NHWC) {
    input_w_size = input_shape.GetDim(2);
  } else if (input_format == FORMAT_NCHW) {
    input_w_size = input_shape.GetDim(3);
  }
  DataType input_type = input_tensor.GetDataType();
  uint32_t ksize{0};
  uint32_t strides{1};
  bool ceil_mode{false};
  if (op.GetAttr("ksize", ksize) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ksize failed");
    return GRAPH_FAILED;
  }
  if (op.GetAttr("strides", strides) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr strides failed");
    return GRAPH_FAILED;
  }
  if (strides == 0) {
    OP_LOGE(op.GetName().c_str(), "Value of strides should not 0");
    return GRAPH_FAILED;
  }
  // get input ksize
  std::vector<int32_t> pads_list;
  if (op.GetAttr("pads", pads_list) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr pads_list failed!");
    return GRAPH_FAILED;
  }
  if (op.GetAttr("ceil_mode", ceil_mode) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ceil_mode failed");
    return GRAPH_FAILED;
  }
  if (pads_list.size() < 2) {
    OP_LOGE(op.GetName().c_str(), "Size of pads_list must greater than 1!");
    return GRAPH_FAILED;
  }
  uint32_t padl = pads_list[0];
  uint32_t padr = pads_list[1];
  uint32_t output_w_size = 0;
  if (ceil_mode) {
    output_w_size = (input_w_size + padl + padr - ksize + strides - 1) / strides + 1;
  } else {
    output_w_size = ((input_w_size + padl + padr) - ksize) / strides + 1;
  }
  if (padl) {
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
    // existing bug in pytorch code
    // padl = 0 and strides is big, but kernel is small, return nan
    if (((static_cast<int64_t>(output_w_size) - 1) * static_cast<int64_t>(strides)) >=
        (input_w_size + static_cast<int64_t>(padl))) {
      output_w_size--;
    }
  }
  padr = (output_w_size - 1) * strides + ksize - input_w_size - padl;

  if (input_format != FORMAT_NHWC && input_format != FORMAT_NCHW) {
    OP_LOGE(op.GetName().c_str(), "Input format only support NCHW or NHWC");
    return GRAPH_FAILED;
  }

  vector<int64_t> dim_vec;
  if (input_format == FORMAT_NHWC) {
    dim_vec.push_back(input_shape.GetDim(0));
    dim_vec.push_back(input_shape.GetDim(1));
    dim_vec.push_back(output_w_size);
    dim_vec.push_back(input_shape.GetDim(3));
  } else if (input_format == FORMAT_NCHW) {
    dim_vec.push_back(input_shape.GetDim(0));
    dim_vec.push_back(input_shape.GetDim(1));
    dim_vec.push_back(input_shape.GetDim(2));
    dim_vec.push_back(output_w_size);
  }
  if (dim_vec.size() == 0) {
    OP_LOGE(op.GetName().c_str(), "Input format is not NCHW or NHWC");
    return GRAPH_FAILED;
  }

  Shape output_shape = Shape(dim_vec);
  DataType output_dtype = input_type;
  output_tensor_desc.SetShape(output_shape);
  output_tensor_desc.SetDataType(output_dtype);
  if (op.UpdateOutputDesc("y", output_tensor_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(AvgPool1D, AvgPool1DInfer);
VERIFY_FUNC_REG(AvgPool1D, AvgPool1DVerify);
// ----------------AvgPool1D END-------------------

// ----------------AvgPool1DD-------------------
IMPLEMT_VERIFIER(AvgPool1DD, AvgPool1DDVerify) {
  if (!CheckTwoInputDtypeSame(op, "x", "assist_matrix")) {
    OP_LOGE(op.GetName().c_str(), "Check matrix input dtype!");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> pads;
  pads = GetAttrValue(op, "pads");
  if (!CheckListEmpty(op.GetName(), pads, "pads")) {
    return GRAPH_FAILED;
  }
  if (pads.size() != 2) {
    OP_LOGE(op.GetName().c_str(), "Pads size should be two");
    return GRAPH_FAILED;
  }
  int64_t k_size;
  if (op.GetAttr("ksize", k_size) != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get k_size failed!");
    return GRAPH_FAILED;
  }
  int64_t strides;
  if (op.GetAttr("strides", strides) != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get strides failed!");
    return GRAPH_FAILED;
  }
  if (strides == 0) {
    OP_LOGE(op.GetName().c_str(), "Value of strides should not 0");
    return GRAPH_FAILED;
  }
  bool ceil_mode{false};
  if (op.GetAttr("ceil_mode", ceil_mode) != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get ceil_mode failed!");
    return GRAPH_FAILED;
  }
  bool count_include_pad{false};
  if (op.GetAttr("count_include_pad", count_include_pad) != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get count_include_pad failed!");
    return GRAPH_FAILED;
  }

  int64_t padl = pads[0];
  int64_t padr = pads[1];
  int64_t w_output{0};

  auto input_tensor = op.GetInputDesc("x");
  Format input_format = input_tensor.GetFormat();
  auto input_shape = input_tensor.GetShape();
  int64_t input_w_size{0};
  if (input_format == FORMAT_NHWC) {
    input_w_size = input_shape.GetDim(2);
  } else if (input_format == FORMAT_NCHW) {
    input_w_size = input_shape.GetDim(3);
  }

  if (ceil_mode) {
    w_output = (input_w_size + padl + padr - k_size + strides - 1) / strides + 1;
  } else {
    w_output = ((input_w_size + padl + padr) - k_size) / strides + 1;
  }
  if (padl) {
    // ensure that the last pooling starts inside the image needed to avoid problems in ceil mode
    // existing bug in pytorch code padl = 0 and strides is big, but kernel is small, return Nan
    if (((w_output - 1) * strides) >= (input_w_size + padl)) {
      w_output--;
    }
  }
  auto matrix_tensor = op.GetInputDesc("assist_matrix");
  auto matrix_shape = matrix_tensor.GetShape();
  if (w_output != matrix_shape.GetDim(3)) {
    OP_LOGE(op.GetName().c_str(), "Check matrix shape W dimension");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(AvgPool1DD, AvgPool1DDInfer) {
  TensorDesc output_tensor_desc = op.GetOutputDesc("y");
  auto input_tensor = op.GetInputDesc("x");
  Format input_format = input_tensor.GetFormat();
  auto input_shape = input_tensor.GetShape();
  auto input_w_size = 0;

  if (input_format == FORMAT_NHWC) {
    input_w_size = input_shape.GetDim(2);
  } else if (input_format == FORMAT_NCHW) {
    input_w_size = input_shape.GetDim(3);
  }
  DataType input_type = input_tensor.GetDataType();
  uint32_t ksize{0};
  uint32_t strides{1};
  bool ceil_mode{false};

  if (op.GetAttr("ksize", ksize) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ksize failed, set ksize default value");
  }
  if (op.GetAttr("strides", strides) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr strides failed, set strides default value");
  }
  if (strides == 0) {
    OP_LOGE(op.GetName().c_str(), "Value of strides should not 0");
    return GRAPH_FAILED;
  }
  // get input ksize
  std::vector<int32_t> pads_list;
  if (op.GetAttr("pads", pads_list) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr pads_list failed!");
    return GRAPH_FAILED;
  }
  if (pads_list.size() < 2) {
    OP_LOGE(op.GetName().c_str(), "Attr pads should has 2 elements at least!");
    return GRAPH_FAILED;
  }
  if (op.GetAttr("ceil_mode", ceil_mode) != GRAPH_SUCCESS) {
    OP_LOGW(op.GetName().c_str(), "GetOpAttr ceil_mode failed, set ceil_mode default value");
  }
  uint32_t padl = pads_list[0];
  uint32_t padr = pads_list[1];
  uint32_t output_w_size{0};
  if (ceil_mode) {
    output_w_size = (input_w_size + padl + padr - ksize + strides - 1) / strides + 1;
  } else {
    output_w_size = ((input_w_size + padl + padr) - ksize) / strides + 1;
  }
  if (padl) {
    // ensure that the last pooling starts inside the image needed to avoid problems in ceil mode existing bug in
    // pytorch code padl = 0 and strides is big, but kernel is small, return Nan
    if (((static_cast<int64_t>(output_w_size) - 1) * static_cast<int64_t>(strides)) >=
        (static_cast<int64_t>(input_w_size) + static_cast<int64_t>(padl))) {
      output_w_size--;
    }
  }
  padr = (output_w_size - 1) * strides + ksize - input_w_size - padl;

  vector<int64_t> dim_vec;
  if (input_format == FORMAT_NHWC) {
    dim_vec.push_back(input_shape.GetDim(0));
    dim_vec.push_back(input_shape.GetDim(1));
    dim_vec.push_back(output_w_size);
    dim_vec.push_back(input_shape.GetDim(3));
  } else if (input_format == FORMAT_NCHW) {
    dim_vec.push_back(input_shape.GetDim(0));
    dim_vec.push_back(input_shape.GetDim(1));
    dim_vec.push_back(input_shape.GetDim(2));
    dim_vec.push_back(output_w_size);
  }
  if (dim_vec.size() == 0) {
    OP_LOGE(op.GetName().c_str(), "Input format is not NCHW or NHWC");
    return GRAPH_FAILED;
  }
  Shape output_shape = ge::Shape(dim_vec);
  DataType output_dtype = input_type;
  output_tensor_desc.SetShape(output_shape);
  output_tensor_desc.SetDataType(output_dtype);
  if (op.UpdateOutputDesc("y", output_tensor_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(AvgPool1DD, AvgPool1DDInfer);
VERIFY_FUNC_REG(AvgPool1DD, AvgPool1DDVerify);
// ----------------AvgPool1DD END-------------------

// ----------------------MaxPoolGradWithArgmaxV2-----------------------
IMPLEMT_VERIFIER(MaxPoolGradWithArgmaxV2, MaxPoolGradWithArgmaxV2Verify) {
  return GRAPH_SUCCESS;
}
IMPLEMT_COMMON_INFERFUNC(MaxPoolGradWithArgmaxV2InferShape) {
  TensorDesc output_y = op.GetOutputDesc("y");

  auto tensorDesc = op.GetInputDesc("x");
  auto shape = tensorDesc.GetShape();
  output_y.SetShape(shape);

  (void)op.UpdateOutputDesc("y", output_y);
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(MaxPoolGradWithArgmaxV2, MaxPoolGradWithArgmaxV2InferShape);
VERIFY_FUNC_REG(MaxPoolGradWithArgmaxV2, MaxPoolGradWithArgmaxV2Verify);
// ----------------------MaxPoolGradWithArgmaxV2-----------------------

// ---------------------MaxPoolWithArgmaxV2---------------------

int cal_max(int input_size, int pad, int dilation, int kernel_size, int stride, bool ceil_mode) {
  int max_size = 0;
  int temp = 0;

  if (stride == 0) {
    return 0;
  }

  temp = input_size + 2 * pad - dilation * (kernel_size - 1) - 1;
  if (ceil_mode) {
    max_size = (temp + stride - 1) / stride + 1;
  } else {
    max_size = temp / stride + 1;
  }
  return max_size;
}

int ceil(int a, int b) {
  int r = 0;
  if (b == 0) {
    return 0;
  }

  if (a % b == 0) {
    r = a / b;
  } else {
    r = a / b + 1;
  }

  return r;
}

void cal_mask(int max_h, int max_w, int kernel_h, int kernel_w, int input_c0, int* mask_h, int* mask_w) {
  int max_mul = 0;
  max_mul = max_h * max_w;
  *mask_h = kernel_h * kernel_w;
  *mask_w = ceil(max_mul, input_c0) + 1;
}

IMPLEMT_VERIFIER(MaxPoolWithArgmaxV2, MaxPoolWithArgmaxV2Verify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(MaxPoolWithArgmaxV2InferShape) {
  TensorDesc output_max = op.GetOutputDesc("y");
  TensorDesc output_mask = op.GetOutputDesc("argmax");

  auto tensorDesc = op.GetInputDesc(0);
  auto shape = tensorDesc.GetShape();

  std::vector<int64_t> max_vec;
  std::vector<int64_t> mask_vec;
  std::vector<int64_t> pads;
  std::vector<int64_t> dilation;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> strides;

  int batch_size, c1_size, input_h, input_w, pad_h, pad_w, dilation_h, dilation_w, kernel_h, kernel_w;
  int max_h, max_w, mask_h, mask_w, stride_h, stride_w;
  int input_c0 = 16;
  bool ceil_mode;

  op.GetAttr("ksize", kernel_size);
  op.GetAttr("strides", strides);
  op.GetAttr("pads", pads);
  op.GetAttr("dilation", dilation);
  op.GetAttr("ceil_mode", ceil_mode);
  batch_size = shape.GetDim(0);
  c1_size = shape.GetDim(1);
  input_h = shape.GetDim(2);
  input_w = shape.GetDim(3);

  pad_h = pads[1];
  pad_w = pads[2];
  dilation_h = dilation[1];
  dilation_w = dilation[2];
  stride_h = strides[1];
  stride_w = strides[2];
  kernel_h = kernel_size[1];
  kernel_w = kernel_size[2];

  max_h = cal_max(input_h, pad_h, dilation_h, kernel_h, stride_h, ceil_mode);
  max_w = cal_max(input_w, pad_w, dilation_w, kernel_w, stride_w, ceil_mode);

  cal_mask(max_h, max_w, kernel_h, kernel_w, input_c0, &mask_h, &mask_w);

  max_vec.push_back(batch_size);
  max_vec.push_back(c1_size);
  max_vec.push_back(max_h);
  max_vec.push_back(max_w);

  mask_vec.push_back(batch_size);
  mask_vec.push_back(c1_size);
  mask_vec.push_back(mask_h);
  mask_vec.push_back(mask_w);

  ge::Shape max_shape = ge::Shape(max_vec);
  ge::Shape mask_shape = ge::Shape(mask_vec);

  output_max.SetShape(max_shape);
  output_max.SetDataType(op.GetInputDesc("x").GetDataType());
  output_max.SetFormat(op.GetInputDesc("x").GetFormat());
  output_mask.SetShape(mask_shape);
  output_mask.SetFormat(op.GetInputDesc("x").GetFormat());

  (void)op.UpdateOutputDesc("y", output_max);
  (void)op.UpdateOutputDesc("argmax", output_mask);
  return GRAPH_SUCCESS;
}
// Registered inferfunction
COMMON_INFER_FUNC_REG(MaxPoolWithArgmaxV2, MaxPoolWithArgmaxV2InferShape);

// Registered verify function
VERIFY_FUNC_REG(MaxPoolWithArgmaxV2, MaxPoolWithArgmaxV2Verify);
// ---------------------MaxPoolWithArgmaxV2---------------------


// ----------------------MaxPoolV3------------------------------
IMPLEMT_VERIFIER(MaxPoolV3, MaxPoolV3Verify) {
  auto inputTensorDesc = op.GetInputDesc("x");
  auto shape = inputTensorDesc.GetShape();
  Format input_format = inputTensorDesc.GetFormat();

  // Verify
  std::vector<int64_t> ksize;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksize)) {
    OpsGetAttrErrReport(op.GetName(), "ksize");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ksize failed!");
    return GRAPH_FAILED;
  }
  if (ksize.size() < 4) {
    return GRAPH_FAILED;
  }

  std::vector<int64_t> strides;
  if (GRAPH_SUCCESS != op.GetAttr("strides", strides)) {
    OpsGetAttrErrReport(op.GetName(), "strides");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr strides failed!");
    return GRAPH_FAILED;
  }
  if (strides.size() < 4) {
    return GRAPH_FAILED;
  }

  std::string padding_mode;
  if (ge::GRAPH_SUCCESS != op.GetAttr("padding_mode", padding_mode)) {
    OpsGetAttrErrReport(op.GetName(), "padding_mode");
    OP_LOGE(op.GetName().c_str(), "Get padding_mode failed!");
    return GRAPH_FAILED;
  }
  if (padding_mode != "SAME" && padding_mode != "VALID" && padding_mode != "CALCULATED") {
    OP_LOGE(op.GetName().c_str(), "attr padding_mode(%s) only support SAME VALID and CALCULATED", padding_mode.c_str());
    return GRAPH_FAILED;
  }

  std::vector<int64_t> pads;
  if (GRAPH_SUCCESS != op.GetAttr("pads", pads)) {
    OpsGetAttrErrReport(op.GetName(), "pads");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr pads failed!");
    return GRAPH_FAILED;
  }
  if (pads.size() < 4) {
    return GRAPH_FAILED;
  }

  std::string data_format;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", data_format)) {
    OpsGetAttrErrReport(op.GetName(), "data_format");
    OP_LOGE(op.GetName().c_str(),
            "The AvgPoolGradD op GetOpAttr data_format "
            "failed!");
    return GRAPH_FAILED;
  }

  bool ceilMode;
  if (GRAPH_SUCCESS != op.GetAttr("ceil_mode", ceilMode)) {
    return GRAPH_FAILED;
  }

  if (data_format != "NCHW" && data_format != "NHWC") {
    OP_LOGE(op.GetName().c_str(), "attr data_format(%s) only support NCHW and NHWC", data_format.c_str());
    return GRAPH_FAILED;
  }
  if (data_format == "NCHW") {
    if (ksize[0] != 1 || ksize[1] != 1 || strides[0] != 1 || strides[1] != 1) {
      OP_LOGE(op.GetName().c_str(),
              "MaxPoolV3Grad only supports pooling across width/height"
              "and other ksize dimension should be one");
      return GRAPH_FAILED;
    }
    if (padding_mode == "CALCULATED" &&
        (pads[0] >= ksize[2] || pads[1] >= ksize[2] || pads[2] >= ksize[3] || pads[3] >= ksize[3])) {
      OP_LOGE(op.GetName().c_str(), "Pads must be less then ksize when using CALCULATED mode!");
      return GRAPH_FAILED;
    }
  }

  if (data_format == "NHWC") {
    if (ksize[0] != 1 || ksize[3] != 1 || strides[0] != 1 || strides[3] != 1) {
      OP_LOGE(op.GetName().c_str(),
              "MaxPoolV3 only supports pooling across width/height"
              "and other ksize dimension should be one");
      return GRAPH_FAILED;
    }
    if (padding_mode == "CALCULATED" &&
        (pads[0] >= ksize[1] || pads[1] >= ksize[1] || pads[2] >= ksize[2] || pads[3] >= ksize[2])) {
      OP_LOGE(op.GetName().c_str(), "Pads must be less then ksize when using CALCULATED mode!");
      return GRAPH_FAILED;
    }
  }

  // input format mast equals to data_format
  if ((input_format == FORMAT_NCHW && data_format != "NCHW") ||
      (input_format == FORMAT_NHWC && data_format != "NHWC")) {
    OP_LOGE(op.GetName().c_str(), "Format of input must be same with dataFormat!");
    return GRAPH_FAILED;
  }

  if(ceilMode && padding_mode == "VALID") {
    OP_LOGE(op.GetName().c_str(), "When padding_mode is 'VALID', ceil_mode must be False");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

// infer
IMPLEMT_COMMON_INFERFUNC(MaxPoolV3InferShape) {
  const size_t DIM_SIZE1 = 1;
  const size_t DIM_SIZE2 = 2;
  const size_t DIM_SIZE3 = 3;
  auto inputTensorDesc = op.GetInputDesc("x");
  auto shape = inputTensorDesc.GetShape();
  Format input_format = inputTensorDesc.GetFormat();

  // get input kszie
  std::vector<int32_t> ksizeList;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksizeList)) {
    return GRAPH_FAILED;
  }

  // get input strides
  std::vector<int32_t> stridesList;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stridesList)) {
    return GRAPH_FAILED;
  }

  // get input data_format
  std::string dataFormat;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
    return GRAPH_FAILED;
  }

  // get input paddingMode
  std::string paddingMode;
  if (GRAPH_SUCCESS != op.GetAttr("padding_mode", paddingMode)) {
    return GRAPH_FAILED;
  }

  // get input pads
  std::vector<int32_t> padVec;
  if (GRAPH_SUCCESS != op.GetAttr("pads", padVec)) {
    return GRAPH_FAILED;
  }

  // get input global_padding
  bool globalPooling;
  if (GRAPH_SUCCESS != op.GetAttr("global_pooling", globalPooling)) {
    return GRAPH_FAILED;
  }

  // get input ceilMode
  bool ceilMode;
  if (GRAPH_SUCCESS != op.GetAttr("ceil_mode", ceilMode)) {
    return GRAPH_FAILED;
  }

  // input format mast equals to data_format
  if ((input_format == FORMAT_NCHW && dataFormat != "NCHW") || (input_format == FORMAT_NHWC && dataFormat != "NHWC")) {
    return GRAPH_FAILED;
  }

  // INFER
  std::vector<int64_t> dims_input = shape.GetDims();
  // set output shape
  std::vector<int64_t> dimVector;
  int64_t window_h, window_w;
  if (FORMAT_NHWC == input_format) {
    if (globalPooling) {
      paddingMode = "VALID";
      window_h = dims_input[1];
      window_w = dims_input[2];
      padVec[0] = 0;
      padVec[1] = 0;
      padVec[2] = 0;
      padVec[3] = 0;
    } else {
      window_h = (int64_t)ksizeList[1];
      window_w = (int64_t)ksizeList[2];
    }
  } else {
    if (globalPooling) {
      paddingMode = "VALID";
      window_h = dims_input[2];
      window_w = dims_input[3];
      padVec[0] = 0;
      padVec[1] = 0;
      padVec[2] = 0;
      padVec[3] = 0;
    } else {
      window_h = (int64_t)ksizeList[2];
      window_w = (int64_t)ksizeList[3];
    }
  }
  if (FORMAT_NHWC == input_format) {
    if (paddingMode == "SAME") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if (DIM_SIZE1 == i || DIM_SIZE2 == i) {
          int64_t dims = (dims_input[i] + stridesList[i] - 1) / stridesList[i];
          dimVector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    } else if (paddingMode == "VALID") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if (DIM_SIZE1 == i) {
          int64_t dims = (dims_input[i] - window_h + 1 + stridesList[i] - 1) / stridesList[i];
          dimVector.push_back(dims);
        } else if (DIM_SIZE2 == i) {
          int64_t dims = (dims_input[i] - window_w + 1 + stridesList[i] - 1) / stridesList[i];
          dimVector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    } else {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if (ceilMode) {
          if (DIM_SIZE1 == i) {
            int64_t dims = (dims_input[i] - window_h + padVec[0] + padVec[1] + stridesList[i] - 1) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else if (DIM_SIZE2 == i) {
            int64_t dims = (dims_input[i] - window_w + padVec[2] + padVec[3] + stridesList[i] - 1) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else {
            int64_t dims = dims_input[i];
            dimVector.push_back(dims);
          }
        } else {
          if (DIM_SIZE1 == i) {
            int64_t dims = (dims_input[i] - window_h + padVec[0] + padVec[1]) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else if (DIM_SIZE2 == i) {
            int64_t dims = (dims_input[i] - window_w + padVec[2] + padVec[3]) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else {
            int64_t dims = dims_input[i];
            dimVector.push_back(dims);
          }
        }
      }
    }
  } else {
    // NCHW
    if (paddingMode == "SAME") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if (DIM_SIZE2 == i || DIM_SIZE3 == i) {
          int64_t dims = (dims_input[i] + stridesList[i] - 1) / stridesList[i];
          dimVector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    } else if (paddingMode == "VALID") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if (DIM_SIZE2 == i) {
          int64_t dims = (dims_input[i] - window_h + 1 + stridesList[i] - 1) / stridesList[i];
          dimVector.push_back(dims);
        } else if (DIM_SIZE3 == i) {
          int64_t dims = (dims_input[i] - window_w + 1 + stridesList[i] - 1) / stridesList[i];
          dimVector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    } else {
      if (ceilMode) {
        for (size_t i = 0; i < dims_input.size(); i++) {
          if (DIM_SIZE2 == i) {
            int64_t dims = (dims_input[i] - window_h + padVec[0] + padVec[1] + stridesList[i] - 1) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else if (DIM_SIZE3 == i) {
            int64_t dims = (dims_input[i] - window_w + padVec[2] + padVec[3] + stridesList[i] - 1) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else {
            int64_t dims = dims_input[i];
            dimVector.push_back(dims);
          }
        }
      } else {
        for (size_t i = 0; i < dims_input.size(); i++) {
          if (DIM_SIZE2 == i) {
            int64_t dims = (dims_input[i] - window_h + padVec[0] + padVec[1]) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else if (DIM_SIZE3 == i) {
            int64_t dims = (dims_input[i] - window_w + padVec[2] + padVec[3]) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else {
            int64_t dims = dims_input[i];
            dimVector.push_back(dims);
          }
        }
      }
    }
  }
  TensorDesc td = op.GetOutputDesc("y");
  DataType inputDtype = inputTensorDesc.GetDataType();
  Shape outputShape(dimVector);
  td.SetShape(outputShape);
  td.SetDataType(inputDtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(MaxPoolV3, MaxPoolV3InferShape);
VERIFY_FUNC_REG(MaxPoolV3, MaxPoolV3Verify);
// ----------------------MaxPoolV3------------------------------

// ----------------------MaxPoolV3Grad--------------------------
bool check_two_input_dtyep_same(const Operator& op, const string& input_name1, const string& input_name2) {
  auto input_type_orig_input = op.GetInputDesc(input_name1).GetDataType();
  auto input_type_orig_output = op.GetInputDesc(input_name2).GetDataType();
  if (input_type_orig_input != input_type_orig_output) {
    return false;
  }
  return true;
}

IMPLEMT_VERIFIER(MaxPoolV3Grad, MaxPoolV3GradVerify) {
  if (!check_two_input_dtyep_same(op, "orig_input", "orig_output") ||
      !check_two_input_dtyep_same(op, "orig_input", "grad")) {
    OP_LOGE(op.GetName().c_str(), "The shape of orig_input orig_output and grad must be same!");
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
    if (data_format != "NCHW" && data_format != "NHWC") {
      OP_LOGE(op.GetName().c_str(), "The data_format should be NCHW or NHWC!");
      return GRAPH_FAILED;
    }
  }
  std::vector<int64_t> ksize;
  ksize = GetAttrValue(op, "ksize");
  if (!CheckListEmpty(op.GetName(), ksize, "ksize")) {
    OP_LOGE(op.GetName().c_str(), "The ksize is empty!");
    return GRAPH_FAILED;
  }
  if (ksize.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "The size of ksize should be 4!");
    return GRAPH_FAILED;
  }
  if (data_format == "NCHW" && (ksize[0] != 1 || ksize[1] != 1)) {
    OP_LOGE(op.GetName().c_str(), "The first and second dim of ksize must be 1!");
    return GRAPH_FAILED;
  }
  if (data_format == "NHWC" && (ksize[0] != 1 || ksize[3] != 1)) {
    OP_LOGE(op.GetName().c_str(), "The first and fourth dim of ksize must be 1!");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), strides, "strides")) {
    OP_LOGE(op.GetName().c_str(), "The strides is empty!");
    return GRAPH_FAILED;
  }
  if (strides.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "The size of strides should be 4!");
    return GRAPH_FAILED;
  }
  if (data_format == "NCHW" && (strides[0] != 1 || strides[1] != 1)) {
    OP_LOGE(op.GetName().c_str(), "The first and second dim of strides must be 1!");
    return GRAPH_FAILED;
  }
  if (data_format == "NHWC" && (strides[0] != 1 || strides[3] != 1)) {
    OP_LOGE(op.GetName().c_str(), "The first and fourth dim of ksize must be 1!");
    return GRAPH_FAILED;
  }
  std::string padding_mode;
  if (ge::GRAPH_SUCCESS != op.GetAttr("padding_mode", padding_mode)) {
    OP_LOGE(op.GetName().c_str(), "The padding_mode is empty!");
    return GRAPH_FAILED;
  }
  if (padding_mode != "SAME" && padding_mode != "VALID" && padding_mode != "CALCULATED") {
    OP_LOGE(op.GetName().c_str(), "The value of padding_mode must be in 'SAME' 'VALID' or 'CALCULATED'!");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> pads;
  pads = GetAttrValue(op, "pads");
  if (!CheckListEmpty(op.GetName(), pads, "pads")) {
    OP_LOGE(op.GetName().c_str(), "The pads is empty!");
    return GRAPH_FAILED;
  }
  if (pads.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "The size of pads should be 4!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(MaxPoolV3Grad, MaxPoolV3GradInferShape) {
  auto shapeX1 = op.GetInputDesc("orig_input").GetShape();
  auto inputType = op.GetInputDesc("orig_input").GetDataType();

  TensorDesc td = op.GetOutputDesc("out_grad");
  td.SetShape(shapeX1);
  td.SetDataType(inputType);
  (void)op.UpdateOutputDesc("out_grad", td);
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(MaxPoolV3Grad, MaxPoolV3GradInferShape);
VERIFY_FUNC_REG(MaxPoolV3Grad, MaxPoolV3GradVerify);
// ----------------------MaxPoolV3Grad--------------------------

// ------------AdaptiveMaxPool2d Op Begin----------------
IMPLEMT_INFERFUNC(AdaptiveMaxPool2d, AdaptiveMaxPool2dInferShape) {
  OP_LOGI(op.GetName().c_str(), " AdaptiveMaxPool2d inferShape begin!");
  const size_t DIM_SIZE2 = 2;
  auto input_tensor_desc = op.GetInputDesc("x");
  auto shape = input_tensor_desc.GetShape();
  // get output_size
  std::vector<int64_t> ouput_size_list;
  if (GRAPH_SUCCESS != op.GetAttr("output_size", ouput_size_list)) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ouput_size_list failed!");
    return GRAPH_FAILED;
  }
  // check output size
  if (ouput_size_list.size() != DIM_SIZE2) {
    OP_LOGE(op.GetName().c_str(), "length of output_size must be 2");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> dims_input = shape.GetDims();
  // set output shape
  std::vector<int64_t> dim_vector;
  for (size_t i = 0; i < dims_input.size(); i++) {
    int64_t dims = dims_input[i];
    dim_vector.push_back(dims);
  }
  size_t index0 = dims_input.size() - 2;
  size_t index1 = dims_input.size() - 1;
  dim_vector[index0] = ouput_size_list[0];
  dim_vector[index1] = ouput_size_list[1];
  TensorDesc td = op.GetOutputDesc("y");
  DataType input_dtype = input_tensor_desc.GetDataType();
  Shape output_shape(dim_vector);
  td.SetShape(output_shape);
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(AdaptiveMaxPool2d, AdaptiveMaxPool2dVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AdaptiveMaxPool2d, AdaptiveMaxPool2dInferShape);
VERIFY_FUNC_REG(AdaptiveMaxPool2d, AdaptiveMaxPool2dVerify);
// ------------AdaptiveMaxPool2d Op End----------------

// ------------AdaptiveAvgPool2d Op Begin----------------
IMPLEMT_INFERFUNC(AdaptiveAvgPool2d, AdaptiveAvgPool2dInferShape) {
  OP_LOGI(op.GetName().c_str(), " AdaptiveAvgPool2d inferShape begin!");
  const size_t DIM_SIZE1 = 1;
  const size_t DIM_SIZE2 = 2;
  const size_t DIM_SIZE3 = 3;
  const size_t DIM_SIZE4 = 4;
  auto input_tensor_desc = op.GetInputDesc("x");
  auto shape = input_tensor_desc.GetShape();
  Format input_format = input_tensor_desc.GetFormat();
  // get output_size
  std::vector<int64_t> ouput_size_list;
  if (GRAPH_SUCCESS != op.GetAttr("output_size", ouput_size_list)) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ouput_size_list failed!");
    return GRAPH_FAILED;
  }
  // check output size
  if (ouput_size_list.size() != DIM_SIZE2) {
    OP_LOGE(op.GetName().c_str(), "length of output_size must be 2");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> dims_input = shape.GetDims();
  // set output shape
  std::vector<int64_t> dim_vector;
  for (size_t i = 0; i < dims_input.size(); i++) {
    int64_t dims = dims_input[i];
    dim_vector.push_back(dims);
  }
  size_t index0 = dims_input.size() - 2;
  size_t index1 = dims_input.size() - 1;
  dim_vector[index0] = ouput_size_list[0];
  dim_vector[index1] = ouput_size_list[1];
  TensorDesc td = op.GetOutputDesc("y");
  DataType input_dtype = input_tensor_desc.GetDataType();
  Shape output_shape(dim_vector);
  td.SetShape(output_shape);
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(AdaptiveAvgPool2d, AdaptiveAvgPool2dVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AdaptiveAvgPool2d, AdaptiveAvgPool2dInferShape);
VERIFY_FUNC_REG(AdaptiveAvgPool2d, AdaptiveAvgPool2dVerify);
// ------------AdaptiveAvgPool2d Op End----------------

// ------------AdaptiveAvgPool2dGrad Op Begin----------------
IMPLEMT_INFERFUNC(AdaptiveAvgPool2dGrad, AdaptiveAvgPool2dGradInferShape) {
  OP_LOGI(op.GetName().c_str(), " AdaptiveAvgPool2dGrad inferShape begin!");
  // get orig_input_shape
  std::vector<int64_t> ori_shape;
  if (GRAPH_SUCCESS != op.GetAttr("orig_input_shape", ori_shape)) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr orig_input_shape failed!");
    return GRAPH_FAILED;
  }
  // get output size
  if (ori_shape.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "length of orig_input_shape must be 4");
    return GRAPH_FAILED;
  }

  TensorDesc output_grad = op.GetOutputDesc("output_grad");
  DataType input_dtype = op.GetInputDesc("input_grad").GetDataType();
  Shape output_shape(ori_shape);
  output_grad.SetShape(output_shape);
  output_grad.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("output_grad", output_grad);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(AdaptiveAvgPool2dGrad, AdaptiveAvgPool2dGradVerify) {
  auto input_tensor_desc = op.GetInputDesc("input_grad");
  auto grad_input_shape = input_tensor_desc.GetShape();
  std::vector<int64_t> dims_input = grad_input_shape.GetDims();
  if (dims_input.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "length of input_grad must be 4");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AdaptiveAvgPool2dGrad, AdaptiveAvgPool2dGradInferShape);
VERIFY_FUNC_REG(AdaptiveAvgPool2dGrad, AdaptiveAvgPool2dGradVerify);
// ------------AdaptiveAvgPool2dGrad Op End----------------

// ------------max_pool_grad_with_argmaxv1 Op Begin----------------
IMPLEMT_VERIFIER(MaxPoolGradWithArgmaxV1, MaxPoolGradWithArgmaxV1Verify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(MaxPoolGradWithArgmaxV1InferShape) {
  TensorDesc output_y = op.GetOutputDesc("y");
  auto tensor_desc = op.GetInputDesc("x");
  auto shape = tensor_desc.GetShape();
  output_y.SetShape(shape);
  (void)op.UpdateOutputDesc("y", output_y);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(MaxPoolGradWithArgmaxV1,
                      MaxPoolGradWithArgmaxV1InferShape);
VERIFY_FUNC_REG(MaxPoolGradWithArgmaxV1, MaxPoolGradWithArgmaxV1Verify);
// ------------max_pool_grad_with_argmaxv1 Op End----------------

// ------------MaxPoolWithArgmaxV1 Op Begin----------------
struct MaxPoolWithArgmaxParam {
  int input_size;
  int pad;
  int dilation;
  int kernel_size;
  int stride;
  bool ceil_mode;
};

int CalMax(const MaxPoolWithArgmaxParam &maxpool) {
  const uint32_t G_DIM_C = 1;
  const uint32_t G_DIM_H = 2;
  int max_size = 0;
  int temp = 0;
  int input_size = maxpool.input_size;
  int pad = maxpool.pad;
  int dilation = maxpool.dilation;
  int kernel_size = maxpool.kernel_size;
  int stride = maxpool.stride;
  bool ceil_mode = maxpool.ceil_mode;
  if (stride == 0) {
    return 0;
  }
  temp =
      input_size + G_DIM_H * pad - dilation * (kernel_size - G_DIM_C) - G_DIM_C;
  if (ceil_mode) {
    max_size = (temp + stride - G_DIM_C) / stride + G_DIM_C;
  } else {
    max_size = temp / stride + G_DIM_C;
  }
  return max_size;
}

int CalCeil(int a, int b) {
  const uint32_t G_DIM_C = 1;
  int r = 0;
  if (b == 0) {
    return 0;
  }

  if (a % b == 0) {
    r = a / b;
  } else {
    r = a / b + G_DIM_C;
  }
  return r;
}

int CalMaskH(int max_h, int max_w, int kernel_h, int kernel_w, int input_c0) {
  int mask_h = 0;
  mask_h = kernel_h * kernel_w;
  return mask_h;
}

int CalMaskW(int max_h, int max_w, int kernel_h, int kernel_w, int input_c0) {
  const uint32_t G_DIM_C = 1;
  int max_mul = 0;
  int mask_w = 0;
  max_mul = max_h * max_w;
  mask_w = CalCeil(max_mul, input_c0) + G_DIM_C;
  return mask_w;
}

IMPLEMT_VERIFIER(MaxPoolWithArgmaxV1, MaxPoolWithArgmaxV1Verify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(MaxPoolWithArgmaxV1InferShape) {
  TensorDesc output_max = op.GetOutputDesc("y");
  TensorDesc output_mask = op.GetOutputDesc("argmax");

  auto tensor_desc = op.GetInputDesc(0);
  auto shape = tensor_desc.GetShape();

  std::vector<int64_t> vec_max, vec_mask, vec_pads, vec_dilation, vec_kernel,
      vec_strides;
  int batch_size, c1_size, input_h, input_w, kernel_h, kernel_w;
  int input_c0 = 16;
  bool ceil_mode = false;
  op.GetAttr("ksize", vec_kernel);
  op.GetAttr("strides", vec_strides);
  op.GetAttr("pads", vec_pads);
  op.GetAttr("dilation", vec_dilation);
  op.GetAttr("ceil_mode", ceil_mode);
  const uint32_t G_DIM_N = 0;
  const uint32_t G_DIM_C = 1;
  const uint32_t G_DIM_H = 2;
  const uint32_t G_DIM_W = 3;
  batch_size = shape.GetDim(G_DIM_N);
  c1_size = shape.GetDim(G_DIM_C);
  input_h = shape.GetDim(G_DIM_H);
  input_w = shape.GetDim(G_DIM_W);
  kernel_h = vec_kernel[G_DIM_C];
  kernel_w = vec_kernel[G_DIM_H];

  MaxPoolWithArgmaxParam maxpool_h = {input_h,
                                      static_cast<int>(vec_pads[G_DIM_C]),
                                      static_cast<int>(vec_dilation[G_DIM_C]),
                                      kernel_h,
                                      static_cast<int>(vec_strides[G_DIM_C]),
                                      ceil_mode};
  MaxPoolWithArgmaxParam maxpool_w = {input_w,
                                      static_cast<int>(vec_pads[G_DIM_H]),
                                      static_cast<int>(vec_dilation[G_DIM_H]),
                                      kernel_w,
                                      static_cast<int>(vec_strides[G_DIM_H]),
                                      ceil_mode};
  int max_h = CalMax(maxpool_h);
  int max_w = CalMax(maxpool_w);
  int mask_h = CalMaskH(max_h, max_w, kernel_h, kernel_w, input_c0);
  int mask_w = CalMaskW(max_h, max_w, kernel_h, kernel_w, input_c0);

  vec_max.push_back(batch_size);
  vec_max.push_back(c1_size);
  vec_max.push_back(max_h);
  vec_max.push_back(max_w);
  vec_mask.push_back(batch_size);
  vec_mask.push_back(c1_size);
  vec_mask.push_back(mask_h);
  vec_mask.push_back(mask_w);

  ge::Shape max_shape = ge::Shape(vec_max);
  ge::Shape mask_shape = ge::Shape(vec_mask);

  output_max.SetShape(max_shape);
  output_max.SetDataType(op.GetInputDesc("x").GetDataType());
  output_max.SetFormat(op.GetInputDesc("x").GetFormat());
  output_mask.SetShape(mask_shape);
  output_mask.SetFormat(op.GetInputDesc("x").GetFormat());

  (void)op.UpdateOutputDesc("y", output_max);
  (void)op.UpdateOutputDesc("argmax", output_mask);
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(MaxPoolWithArgmaxV1, MaxPoolWithArgmaxV1InferShape);

// Registered verify function
VERIFY_FUNC_REG(MaxPoolWithArgmaxV1, MaxPoolWithArgmaxV1Verify);
// ------------MaxPoolWithArgmaxV1 Op End----------------

}  // namespace ge
