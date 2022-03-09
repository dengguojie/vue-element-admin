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
#include <cmath>
#include "axis_util.h"
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
  constexpr size_t kAvgPoolGradOriShapeDim = 4;
  constexpr size_t kAvgPool3DGradOriShapeDim = 5;
  constexpr size_t kAvgPool3DGradKsizeDim = 5;
  constexpr size_t kAvgPool3DGradStridesDim = 5;
  constexpr size_t kAvgPool3DGradPadsDim = 6;
  constexpr size_t kAvgPool3DGradShapeDim = 6;
  constexpr size_t kAvgPool3DOriShapeDim = 5;
  constexpr size_t kAvgPool3DShapeDim = 6;
  constexpr size_t kAvgPool3DPadsDim = 6;
  constexpr size_t kAvgPool3DGradDataSlice = 6;
  const int64_t kDynamicRangeLowerBound = 1;
  const int64_t kDynamicRangeUpperBound = 4096;
  map<int, std::string> format2str = {
    {ge::FORMAT_NCHW, "NCHW"}, {ge::FORMAT_NHWC, "NHWC"}, {ge::FORMAT_HWCN, "HWCN"},
    {ge::FORMAT_DHWNC, "DHWNC"}, {ge::FORMAT_DHWCN, "DHWCN"}, {ge::FORMAT_NDHWC, "NDHWC"},
    {ge::FORMAT_NCDHW, "NCDHW"}
  };
  map<int, std::string> dtype2str = {
    {ge::DT_FLOAT, "FLOAT"}, {ge::DT_FLOAT16, "FLOAT16"}, {ge::DT_INT8, "INT8"},
    {ge::DT_INT16, "INT16"}, {ge::DT_UINT16, "UINT16"}, {ge::DT_UINT8, "UINT8"},
    {ge::DT_INT32, "INT32"}, {ge::DT_INT64, "INT64"}, {ge::DT_UINT32, "UINT32"},
    {ge::DT_UINT64, "UINT64"}
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
      if (strd == 0 || strh == 0 || strw == 0) {
        return false;
      }
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
    OP_LOGE("GetWindowedOutputSizeVerboseV2", "Stride must be > 0, but got [%ld]", stride);
    return GRAPH_FAILED;
  }
  if (dilation_rate < 1) {
    OP_LOGE("GetWindowedOutputSizeVerboseV2", "Dilation rate must be >= 1, but got [%ld]", dilation_rate);
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
    OP_LOGE("GetWindowedOutputSizeVerboseV2", "Padding [%s] is invaild.", padding_type.c_str());
    return GRAPH_FAILED;
  }
  if (*output_size < 0) {
    OP_LOGE("GetWindowedOutputSizeVerboseV2", "Computed output size would be negative, but got [%ld].",
            *output_size);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

static bool reset_range(ge::Operator& op, const std::string& tensor_name) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto tensor_desc = op_desc->MutableInputDesc(tensor_name);
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  tensor_desc->GetShapeRange(shape_range);
  if (shape_range.empty()) {
    return false;
  }
  std::vector<int64_t> shape_sizes = tensor_desc->MutableShape().GetDims();
  if (shape_range.size() != shape_sizes.size()) {
    return false;
  }
  bool reset_flag = false;
  for (size_t i = 0; i < shape_sizes.size(); i++) {
    if (shape_sizes[i] > 0 && (shape_range[i].first != shape_sizes[i] || shape_range[i].second != shape_sizes[i])) {
      reset_flag = true;
      shape_range[i].first = shape_sizes[i];
      shape_range[i].second = shape_sizes[i];
    }
  }

  if (reset_flag) {
    tensor_desc->SetShapeRange(shape_range);
    ge::AscendString op_name;
    (void)op.GetName(op_name);
    OP_LOGW(op_name.GetString(), "%s range does not match the shape value, has been fixed.", tensor_name.c_str());
  }
  return reset_flag;
}

//--------- Dilation2D ---------------
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
    OP_LOGE(op.GetName().c_str(), "The rank of x must be 4.");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 3, filter_shape, op.GetName().c_str())
      != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of filter must be 3.");
    return GRAPH_FAILED;
  }

  std::string data_format = "NHWC";
  op.GetAttr("data_format", data_format);
  if (data_format != "NHWC" && data_format != "NCHW") {
    OP_LOGE(op.GetName().c_str(), "data_format[%s] is invalid!", data_format.c_str());
    return GRAPH_FAILED;
  }
  if((input_format == FORMAT_NCHW && data_format != "NCHW") || (input_format == FORMAT_NHWC && data_format != "NHWC")) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), string("Input format and data_format is not same"));
    return GRAPH_FAILED;
  }
  std::vector<int64_t> strides;
  if (op.GetAttr("strides", strides) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get attr strides failed");
    return GRAPH_FAILED;
  }
  if (strides.size() != 4) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        GetAttrSizeErrMsg("strides", ConcatString(strides.size()), "4"));
    return GRAPH_FAILED;
  }

  std::vector<int64_t> rates;
  if (op.GetAttr("rates", rates) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get attr rates failed");
    return GRAPH_FAILED;
  }
  if (rates.size() != 4) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        GetAttrSizeErrMsg("rates", ConcatString(rates.size()), "4"));
    return GRAPH_FAILED;
  }

  std::string padding_mode = "SAME";
  op.GetAttr("padding_mode", padding_mode);
  if (padding_mode != "SAME" && padding_mode != "VALID" && padding_mode != "CALCULATED") {
    OP_LOGE(op.GetName().c_str(), "Attr padding(%s) only support SAME,VALID,CALCULATED", padding_mode.c_str());
    return GRAPH_FAILED;
  }

  std::vector<int64_t> pads = {0, 0, 0, 0};
  op.GetAttr("pads", pads);
  if (pads.size() != 4) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        GetAttrSizeErrMsg("pads", ConcatString(pads.size()), "4"));
    OP_LOGE(op.GetName().c_str(), "Attr pads(%lu) must be 4", pads.size());
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
    OP_LOGE(op.GetName().c_str(), "rates[%ld, %ld] of NC is invalid!", rate_n, rate_c);
    return GRAPH_FAILED;
  }
  if (stride_n != 1 || stride_c != 1) {
    OP_LOGE(op.GetName().c_str(), "strides[%ld, %ld] of NC is invalid!", stride_n, stride_c);
    return GRAPH_FAILED;
  }
  if (padding_mode == "CALCULATED" &&
      (pads[0] >= window_h || pads[1] >= window_h || pads[2] >= window_w || pads[3] >= window_w)) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), string("pads must be less than window size when using CALCULATED mode"));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(Dilation2D, Dilation2DInfer) {
  auto strides =  op.get_attr_strides();
  auto rates = op.get_attr_rates();
  std::string padding_mode = "SAME";
  op.GetAttr("padding_mode", padding_mode);
  std::vector<int64_t> pads = {0, 0, 0, 0};
  op.GetAttr("pads", pads);
  bool ceil_mode = false;
  op.GetAttr("ceil_mode", ceil_mode);
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
  if (GetWindowedOutputSizeVerboseV2(
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
  if (GetWindowedOutputSizeVerboseV2(
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

//--------- Dilation2DBackpropFilter ---------------
IMPLEMT_VERIFIER(Dilation2DBackpropFilter, Dilation2DBackpropFilterVerify) {
  auto x_shape = op.GetInputDesc("x").GetShape();
  Format input_format = op.GetInputDesc("x").GetFormat();
  auto filter_shape = op.GetInputDesc("filter").GetShape();
  std::vector<int64_t> filter_list = filter_shape.GetDims();
  auto out_backprop_shape = op.GetInputDesc("out_backprop").GetShape();
  Format out_backprop_format = op.GetInputDesc("out_backprop").GetFormat();

  if (!CheckTwoInputDtypeSame(op, "x", "filter")) {
    OP_LOGE(op.GetName().c_str(), "Two input dtypes must be same.");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(0), 4, x_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of x must be 4.");
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(1), 3, filter_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of filter must be 3.");
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(2), 4, out_backprop_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of out_backprop must be 4.");
    return GRAPH_FAILED;
  }

  std::string data_format;
  if (op.GetAttr("data_format", data_format) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr data_format failed!");
    return GRAPH_FAILED;
  }
  if (data_format != "NHWC" && data_format != "NCHW") {
    OP_LOGE(op.GetName().c_str(), "data_format[%s] is invalid!", data_format.c_str());
    return GRAPH_FAILED;
  }

  if ((input_format == FORMAT_NCHW && out_backprop_format == FORMAT_NCHW && data_format != "NCHW") ||
      (input_format == FORMAT_NHWC && out_backprop_format == FORMAT_NHWC && data_format != "NHWC")) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), string("Input format and out_backprop_format and data_format is not same"));
  }

  std::vector<int64_t> strides;
  if (op.GetAttr("strides", strides) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get attr strides failed");
    return GRAPH_FAILED;
  }
  if (strides.size() != 4) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        GetAttrSizeErrMsg("strides", ConcatString(strides.size()), "4"));
    return GRAPH_FAILED;
  }

  std::vector<int64_t> rates;
  if (op.GetAttr("rates", rates) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get attr rates failed");
    return GRAPH_FAILED;
  }

  if (rates.size() != 4) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        GetAttrSizeErrMsg("rates", ConcatString(rates.size()), "4"));
    return GRAPH_FAILED;
  }

  std::string padding_mode;
  if (op.GetAttr("padding_mode", padding_mode) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get padding failed!");
    return GRAPH_FAILED;
  }

  if (padding_mode != "SAME" && padding_mode != "VALID" && padding_mode != "CALCULATED") {
    OP_LOGE(op.GetName().c_str(), "Attr padding(%s) only support SAME,VALID,CALCULATED", padding_mode.c_str());
    return GRAPH_FAILED;
  }

  std::vector<int64_t> pads;
  if (op.GetAttr("pads", pads) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get attr pads failed");
    return GRAPH_FAILED;
  }
  if (pads.size() != 4) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        GetAttrSizeErrMsg("pads", ConcatString(pads.size()), "4"));
    return GRAPH_FAILED;
  }

  bool ceil_mode;
  if (op.GetAttr("ceil_mode", ceil_mode) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get attr ceil_mode failed");
    return GRAPH_FAILED;
  }

  int64_t window_h;
  int64_t window_w;
  int64_t rate_n;
  int64_t rate_c;
  int64_t stride_n;
  int64_t stride_c;

  if (data_format == "NHWC") {
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
    OP_LOGE(op.GetName().c_str(), "rates[%ld, %ld] of NC is invalid!", rate_n, rate_c);
    return GRAPH_FAILED;
  }

  if (stride_n != 1 || stride_c != 1) {
    OP_LOGE(op.GetName().c_str(), "strides[%ld, %ld] of NC is invalid!", stride_n, stride_c);
    return GRAPH_FAILED;
  }

  if (padding_mode == "CALCULATED" &&
      (pads[0] >= window_h || pads[1] >= window_h || pads[2] >= window_w || pads[3] >= window_w)) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), string("pads must be less than window size when using CALCULATED mode"));
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(Dilation2DBackpropFilter, Dilation2DBackpropFilterInfer) {
  auto strides = op.get_attr_strides();
  auto rates = op.get_attr_rates();
  auto padding_mode = op.get_attr_padding_mode();
  auto pads = op.get_attr_pads();
  auto data_format = op.get_attr_data_format();
  Shape x_shape = op.GetInputDesc("x").GetShape();
  Shape filter_shape = op.GetInputDesc("filter").GetShape();
  auto data_type = op.GetInputDesc("x").GetDataType();
  TensorDesc output_desc = op.GetOutputDesc("y");

  int64_t batch_size_dim;
  int64_t filter_rows_dim;
  int64_t filter_cols_dim;
  int64_t output_depth_dim;
  int64_t unused;
  int32_t x_h_dim;
  int32_t x_w_dim;
  int32_t filter_h_dim;
  int32_t filter_w_dim;

  if (data_format == "NHWC") {
    batch_size_dim = 1;
    filter_rows_dim = filter_shape.GetDim(0);
    filter_cols_dim = filter_shape.GetDim(1);
    output_depth_dim = filter_shape.GetDim(2);
    unused = x_shape.GetDim(3);
    x_h_dim = 1;
    x_w_dim = 2;
    filter_h_dim = 0;
    filter_w_dim = 1;
  } else {
    batch_size_dim = 1;
    filter_rows_dim = filter_shape.GetDim(1);
    filter_cols_dim = filter_shape.GetDim(2);
    output_depth_dim = filter_shape.GetDim(0);
    unused = x_shape.GetDim(1);
    x_h_dim = 2;
    x_w_dim = 3;
    filter_h_dim = 1;
    filter_w_dim = 2;
  }

  if (!ValueKnown(x_shape, x_h_dim) || !ValueKnown(x_shape, x_w_dim) || !ValueKnown(filter_shape, filter_h_dim) ||
      !ValueKnown(filter_shape, filter_w_dim)) {
    Shape output_shape({batch_size_dim, -1, -1, output_depth_dim});
    output_desc.SetShape(output_shape);
    output_desc.SetDataType(data_type);
    return op.UpdateOutputDesc("y", output_desc);
  }

  if (Merge(unused, output_depth_dim, unused) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Merge unused and output_depth_dim failed.");
    return GRAPH_FAILED;
  }

  if (data_format == "NHWC") {
    Shape y_shape({batch_size_dim, filter_rows_dim, filter_cols_dim, output_depth_dim});
    output_desc.SetShape(y_shape);
  } else {
    Shape y_shape({batch_size_dim, output_depth_dim, filter_rows_dim, filter_cols_dim});
    output_desc.SetShape(y_shape);
  }
  output_desc.SetDataType(data_type);
  return op.UpdateOutputDesc("y", output_desc);
}

VERIFY_FUNC_REG(Dilation2DBackpropFilter, Dilation2DBackpropFilterVerify);
INFER_FUNC_REG(Dilation2DBackpropFilter, Dilation2DBackpropFilterInfer);

//--------- Dilation2DBackpropInput ---------------
IMPLEMT_VERIFIER(Dilation2DBackpropInput, Dilation2DBackpropInputVerify) {
  auto x_shape = op.GetInputDesc("x").GetShape();
  Format input_format = op.GetInputDesc("x").GetFormat();
  auto filter_shape = op.GetInputDesc("filter").GetShape();
  std::vector<int64_t> filter_list = filter_shape.GetDims();
  auto out_backprop_shape = op.GetInputDesc("out_backprop").GetShape();
  Format out_backprop_format = op.GetInputDesc("out_backprop").GetFormat();

  if (!CheckTwoInputDtypeSame(op, "x", "filter")) {
    OP_LOGE(op.GetName().c_str(), "Two input dtypes must be same.");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(0), 4, x_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of x must be 4.");
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(1), 3, filter_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of filter must be 3.");
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(2), 4, out_backprop_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of out_backprop must be 4.");
    return GRAPH_FAILED;
  }

  std::string data_format;
  if (op.GetAttr("data_format", data_format) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr data_format failed!");
    return GRAPH_FAILED;
  }
  if (data_format != "NHWC" && data_format != "NCHW") {
    OP_LOGE(op.GetName().c_str(), "data_format[%s] is invalid!", data_format.c_str());
    return GRAPH_FAILED;
  }

  if ((input_format == FORMAT_NCHW && out_backprop_format == FORMAT_NCHW && data_format != "NCHW") ||
      (input_format == FORMAT_NHWC && out_backprop_format == FORMAT_NHWC && data_format != "NHWC")) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), string("Input format and out_backprop_format and data_format is not same"));
  }

  std::vector<int64_t> strides;
  if (op.GetAttr("strides", strides) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get attr strides failed");
    return GRAPH_FAILED;
  }
  if (strides.size() != 4) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        GetAttrSizeErrMsg("strides", ConcatString(strides.size()), "4"));
    return GRAPH_FAILED;
  }

  std::vector<int64_t> rates;
  if (op.GetAttr("rates", rates) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get attr rates failed");
    return GRAPH_FAILED;
  }

  if (rates.size() != 4) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        GetAttrSizeErrMsg("rates", ConcatString(rates.size()), "4"));
    return GRAPH_FAILED;
  }

  std::string padding_mode;
  if (op.GetAttr("padding_mode", padding_mode) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get padding failed!");
    return GRAPH_FAILED;
  }

  if (padding_mode != "SAME" && padding_mode != "VALID" && padding_mode != "CALCULATED") {
    OP_LOGE(op.GetName().c_str(), "Attr padding(%s) only support SAME,VALID,CALCULATED", padding_mode.c_str());
    return GRAPH_FAILED;
  }

  std::vector<int64_t> pads;
  if (op.GetAttr("pads", pads) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get attr pads failed");
    return GRAPH_FAILED;
  }
  if (pads.size() != 4) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        GetAttrSizeErrMsg("pads", ConcatString(pads.size()), "4"));
    return GRAPH_FAILED;
  }

  bool ceil_mode;
  if (op.GetAttr("ceil_mode", ceil_mode) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get attr ceil_mode failed");
    return GRAPH_FAILED;
  }

  int64_t window_h;
  int64_t window_w;
  int64_t rate_n;
  int64_t rate_c;
  int64_t stride_n;
  int64_t stride_c;

  if (data_format == "NHWC") {
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
    OP_LOGE(op.GetName().c_str(), "rates[%ld, %ld] of NC is invalid!", rate_n, rate_c);
    return GRAPH_FAILED;
  }

  if (stride_n != 1 || stride_c != 1) {
    OP_LOGE(op.GetName().c_str(), "strides[%ld, %ld] of NC is invalid!", stride_n, stride_c);
    return GRAPH_FAILED;
  }

  if (padding_mode == "CALCULATED" &&
      (pads[0] >= window_h || pads[1] >= window_h || pads[2] >= window_w || pads[3] >= window_w)) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), string("pads must be less than window size when using CALCULATED mode"));
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(Dilation2DBackpropInput, Dilation2DBackpropInputInfer) {
  auto strides = op.get_attr_strides();
  auto rates = op.get_attr_rates();
  auto padding_mode = op.get_attr_padding_mode();
  auto pads = op.get_attr_pads();
  auto data_format = op.get_attr_data_format();
  Shape x_shape = op.GetInputDesc("x").GetShape();
  Shape filter_shape = op.GetInputDesc("filter").GetShape();
  auto data_type = op.GetInputDesc("x").GetDataType();
  TensorDesc output_desc = op.GetOutputDesc("y");

  int64_t batch_size_dim;
  int64_t in_rows_dim;
  int64_t in_cols_dim;
  int64_t output_depth_dim;
  int64_t unused;
  int32_t x_h_dim;
  int32_t x_w_dim;
  int32_t filter_h_dim;
  int32_t filter_w_dim;

  if (data_format == "NHWC") {
    batch_size_dim = x_shape.GetDim(0);
    in_rows_dim = x_shape.GetDim(1);
    in_cols_dim = x_shape.GetDim(2);
    output_depth_dim = filter_shape.GetDim(2);
    unused = x_shape.GetDim(3);
    x_h_dim = 1;
    x_w_dim = 2;
    filter_h_dim = 0;
    filter_w_dim = 1;
  } else {
    batch_size_dim = x_shape.GetDim(0);
    in_rows_dim = x_shape.GetDim(2);
    in_cols_dim = x_shape.GetDim(3);
    output_depth_dim = filter_shape.GetDim(0);
    unused = x_shape.GetDim(1);
    x_h_dim = 2;
    x_w_dim = 3;
    filter_h_dim = 1;
    filter_w_dim = 2;
  }

  if (!ValueKnown(x_shape, x_h_dim) || !ValueKnown(x_shape, x_w_dim) || !ValueKnown(filter_shape, filter_h_dim) ||
      !ValueKnown(filter_shape, filter_w_dim)) {
    Shape output_shape({batch_size_dim, -1, -1, output_depth_dim});
    output_desc.SetShape(output_shape);
    output_desc.SetDataType(data_type);
    return op.UpdateOutputDesc("y", output_desc);
  }

  if (Merge(unused, output_depth_dim, unused) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Merge unused and output_depth_dim failed.");
    return GRAPH_FAILED;
  }

  if (data_format == "NHWC") {
    Shape y_shape({batch_size_dim, in_rows_dim, in_cols_dim, output_depth_dim});
    output_desc.SetShape(y_shape);
  } else {
    Shape y_shape({batch_size_dim, output_depth_dim, in_rows_dim, in_cols_dim});
    output_desc.SetShape(y_shape);
  }
  output_desc.SetDataType(data_type);
  return op.UpdateOutputDesc("y", output_desc);
}

VERIFY_FUNC_REG(Dilation2DBackpropInput, Dilation2DBackpropInputVerify);
INFER_FUNC_REG(Dilation2DBackpropInput, Dilation2DBackpropInputInfer);

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
    string expected_format_list = ConcatString("NCHW, NHWC");
    std::string err_msg = GetInputFormatNotSupportErrMsg("xFormat", expected_format_list, ConcatString(xFormat));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg_padT = GetAttrValueErrMsg("padT", ConcatString(padT), ConcatString(padB));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg_padT);
    std::string err_msg_padL = GetAttrValueErrMsg("padL", ConcatString(padL), ConcatString(padR));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg_padL);
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
    std::string err_msg = OtherErrMsg("Unknown rounding mode.");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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

    bool conditionH = ((outputH - 1) * strH) <= inputH + padT;
    if (!conditionH) {
      std::string err_msg = OtherErrMsg("CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_) failed!");
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }

    bool conditionW = ((outputW - 1) * strW) <= inputW + padL;
    if (!conditionW) {
      std::string err_msg = OtherErrMsg("CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_) failed!");
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    string expected_format_list = ConcatString("NCHW or NHWC");
    std::string err_msg = GetInputFormatNotSupportErrMsg("yFormat", expected_format_list, ConcatString(yFormat));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  outdesc.SetShape(Shape(yShape));
  outdesc.SetDataType(ge::DataType(xDtype));

  if (GRAPH_SUCCESS != op.update_output_desc_y(outdesc)) {
    std::string err_msg = UpdateParamErrMsg("output desc");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = GetShapeSizeErrMsg(0, std::to_string(xShape.size()), ConcatString(4));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (window.size() != 2) {
    std::string err_msg = GetShapeSizeErrMsg(3, std::to_string(window.size()), ConcatString(2));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (stride.size() != 2) {
    std::string err_msg = GetShapeSizeErrMsg(4, std::to_string(stride.size()), ConcatString(2));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (pad.size() != 4) {
    std::string err_msg = GetShapeSizeErrMsg(5, std::to_string(pad.size()), ConcatString(4));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
      string expected_format_list = ConcatString("NHWC, NCHW");
      std::string err_msg = GetInputFormatNotSupportErrMsg("dataFormat", expected_format_list, dataFormat);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    string expected_format_list = ConcatString("NHWC, NCHW");
    std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, ConcatString("null"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFER_DATA_SLICE(Pooling, PoolingInferDataSlice) {
  OP_LOGD(op.GetName().c_str(), "Enter PoolingInferDataSlice.");
  vector<vector<int64_t>> y_data_slice = {{}, {}, {}, {}, {}};
  vector<vector<int64_t>> x_data_slice = {{}, {}, {}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_out = op_desc->MutableOutputDesc(0);
  GeTensorDescPtr tensor_desc_in = op_desc->MutableInputDesc(0);
  if (!ge::AttrUtils::GetListListInt(tensor_desc_out, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op.GetName().c_str(), "no data slice, use default as {{}, {}, {}, {}, {}}");
    return GRAPH_FAILED;
  }
  for(unsigned i = 0; i < y_data_slice.size(); i++) {
    if (y_data_slice[i].size() >= 2) {
      if (i == 0) {
        int64_t n_start = y_data_slice[i][0];
        int64_t n_end = y_data_slice[i][1];
        x_data_slice[i] = {n_start, n_end};

        if (!AttrUtils::SetListListInt(tensor_desc_in, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
          std::string err_msg = SetAttrErrMsg("x_data_slice");
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
          return GRAPH_FAILED;
        }
      } else if (i == 2) {
        int64_t out_h_start = y_data_slice[i][0];
        int64_t out_h_end = y_data_slice[i][1];

        OP_LOGD(op.GetName().c_str(), "Get cut h output range[%d, %d]", out_h_start, out_h_end);

        auto xFormat = op.get_input_desc_x().GetFormat();
        auto xShape = op.get_input_desc_x().GetShape().GetDims();
        int64_t inputH = 0;
        if (xFormat == FORMAT_NC1HWC0) {
          inputH = xShape[2];
        } else {
          OP_LOGI(op.GetName().c_str(), "x format is not NC1HWC0");
          return GRAPH_FAILED;
        }

        // get attr
        auto globalPooling = op.get_attr_global_pooling();
        auto window = op.get_attr_window();
        auto pad = op.get_attr_pad();
        auto stride = op.get_attr_stride();
        if (globalPooling) {
          OP_LOGI(op.GetName().c_str(), "global pooling can't cut h");
          return NOT_SUPPORT_SLICE;
        }
        if (pad.size() != 4) {
          OP_LOGI(op.GetName().c_str(), "pad size error");
          return NOT_SUPPORT_SLICE;
        }
        int64_t strH = stride[0];
        int64_t windowH = window[0];
        int64_t padT = pad[0];

        int64_t in_h_start = strH * out_h_start - padT;
        in_h_start = in_h_start > 0 ? in_h_start : 0;
        int64_t in_h_end = strH * out_h_end - padT + windowH;
        in_h_end = in_h_end < inputH ? in_h_end : inputH;
        in_h_end = in_h_end - 1;
        x_data_slice[i] = {in_h_start, in_h_end};

        OP_LOGD(op.GetName().c_str(), "Get cut h input range[%d, %d]", in_h_start, in_h_end);

        vector<int32_t> new_pad_lists = {0, 0, 0, 0};
        if (in_h_start == 0) {
          new_pad_lists[0] = pad[0];
          new_pad_lists[2] = pad[2];
          new_pad_lists[3] = pad[3];
        } else if (in_h_start != 0 && in_h_end != (inputH - 1)) {
          new_pad_lists[2] = pad[2];
          new_pad_lists[3] = pad[3];
        } else {
          new_pad_lists[1] = pad[1];
          new_pad_lists[2] = pad[2];
          new_pad_lists[3] = pad[3];
        }
        op.SetAttr("pad", new_pad_lists);
        OP_LOGD(op.GetName().c_str(), "pooling new pad lists is [%d, %d, %d, %d]", new_pad_lists[0],
                new_pad_lists[1], new_pad_lists[2], new_pad_lists[3]);

        if (!AttrUtils::SetListListInt(tensor_desc_in, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
          std::string err_msg = SetAttrErrMsg("x_data_slice");
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
          return GRAPH_FAILED;
        }
      } else {
        OP_LOGI(op.GetName().c_str(), "only support cut in n and h");
        return NOT_SUPPORT_SLICE;
      }
    }
  }
  OP_LOGD(op.GetName().c_str(), "PoolingInferDataSlice success");
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Pooling, PoolingInferShape);
VERIFY_FUNC_REG(Pooling, PoolingVerify);
INFER_FORMAT_FUNC_REG(Pooling, PoolingInferFormat);
INFER_DATA_SLICE_FUNC_REG(Pooling, PoolingInferDataSlice);
// ----------------Pooling-------------------

/*!
  * Simply get value range in grade list.
  */
static bool GetSingleRange(ge::Operator& op, const std::vector<int64_t>& grade,
                          const int64_t& value, int64_t& low, int64_t& high) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);

  size_t min_size = 2;
  CHECK(grade.size() < min_size, OP_LOGE(op_name.GetString(), "input grade size smaller then %lu", min_size),
        return false);
  // grade is in ascending order
  size_t last = grade.size() - 1;
  CHECK(value > grade[last],
        OP_LOGE(op_name.GetString(), "input value %ld is out the range of %ld", value, grade[last]), return false);
  // if it is the right boundary value, use the right closed interval
  if (value == grade[last]) {
    low = grade[last - 1] + 1;
    high = grade[last];
    return true;
  }
  for (auto n : grade) {
    if (value > n) {
      low = n + 1;
    }
    if (value <= n) {
      high = n;
      break;
    }
  }
  return true;
}

/*!
  * Make sure that output shape is larger than 0
  */
bool CorrectFuzzyCompileRangeStart(ge::Operator& op, ge::GeTensorDescPtr& x_tensor,
                                        std::vector<std::pair<int64_t, int64_t>>& input_range,
                                        int32_t kh, int32_t kw) {
  auto x_shape = x_tensor->MutableShape().GetDims();
  // only support 4D shape
  auto x_format = x_tensor->GetFormat();
  size_t idx_h = 0;
  size_t idx_w = 0;
  if (x_format == FORMAT_NHWC) {
    idx_h = 1;
    idx_w = 2;
  } else {
    idx_h = 2;
    idx_w = 3;
  }
  std::string pads_str;
  op.GetAttr("padding", pads_str);

  int64_t low_h = kh;
  int64_t low_w = kw;
  // get the smallest shape value allowed
  if (pads_str == "SAME") {
    // pad is dynamic, get value from shape or range
    if (x_shape[idx_h] != -1) {
      // shape is static
      low_h = x_shape[idx_h] < low_h ? x_shape[idx_h] : low_h;
    } else {
      // shape is dynamic
      low_h = input_range[idx_h].first < low_h ? input_range[idx_h].first : low_h;
    }
    if (x_shape[idx_w] != -1) {
      low_w = x_shape[idx_w] < low_w ? x_shape[idx_w] : low_w;
    } else {
      low_w = input_range[idx_w].first < low_w ? input_range[idx_w].first : low_w;
    }
  }
  // get larger one for left range
  input_range[idx_h].first = input_range[idx_h].first > low_h ? input_range[idx_h].first : low_h;
  input_range[idx_w].first = input_range[idx_w].first > low_w ? input_range[idx_w].first : low_w;
  return true;
}

// ----------------AvgPool-------------------
IMPLEMT_VERIFIER(AvgPool, AvgPoolVerify) {
  auto ksize = op.get_attr_ksize();
  auto strides = op.get_attr_strides();
  auto x_shape = op.get_input_desc_x().GetShape().GetDims();
  bool unknown_rank = IsUnknownRankShape(x_shape);
  bool invalid_param = ((!unknown_rank && x_shape.size() != 4) || ksize.size() != 4 || strides.size() != 4);
  if (invalid_param) {
    std::string err_msg = OtherErrMsg("xShape size != 4 or kSize size() != 4, strides size() != 4");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
    if (data_format != "NHWC" && data_format != "NCHW" && data_format != "NC1HWC0") {
      string expected_format_list = ConcatString("NHWC,NCHW,NC1HWC0");
      std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    } else {
      if (data_format == "NHWC") {
        if (ksize[0] != 1 || ksize[3] != 1) {
          string expected_pool_list = ConcatString("width,height");
          std::string err_msg1 = GetAttrSizeErrMsg(TbeGetName(op),data_format, expected_pool_list);
          std::string err_msg = ConcatString(err_msg1,"and other ksize dimension should be one");
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
          return GRAPH_FAILED;
        }
        if (strides[0] != 1 || strides[3] != 1) {
          string expected_pool_list = ConcatString("width,height");
          std::string err_msg1 = GetAttrSizeErrMsg(TbeGetName(op), data_format, expected_pool_list);
          std::string err_msg = ConcatString(err_msg1,"and other strides dimension should be one");
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
         return GRAPH_FAILED;
        }
      } else {
        if (ksize[0] != 1 || ksize[1] != 1) {
          string expected_pool_list = ConcatString("width,height");
          std::string err_msg1 = GetAttrSizeErrMsg(TbeGetName(op), data_format, expected_pool_list);
          std::string err_msg = ConcatString(err_msg1,"and other ksize dimension should be one");
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
          return GRAPH_FAILED;
        }
        if (strides[0] != 1 || strides[1] != 1) {
          string expected_pool_list = ConcatString("width,height");
          std::string err_msg1 = GetAttrSizeErrMsg(TbeGetName(op), data_format, expected_pool_list);
          std::string err_msg = ConcatString(err_msg1,"and other strides dimension should be one");
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
          return GRAPH_FAILED;
        }
      }
    }
  }
  return GRAPH_SUCCESS;
}

static void SetAvgPoolOutRange(const std::string pad_str, const vector<int32_t>& attr_params,
                              const bool& ceil_mode,
                              const std::vector<std::pair<int64_t, int64_t>>& input_range,
                              std::vector<std::pair<int64_t, int64_t>>& output_range) {
  size_t attrIdx = 0;
  int32_t stride = attr_params[attrIdx++];
  int32_t pad = attr_params[attrIdx++];
  int32_t kernel = attr_params[attrIdx++];
  int32_t input_position = attr_params[attrIdx++];
  int32_t low = input_range[input_position].first;
  int32_t high = input_range[input_position].second;
  if (pad_str == "SAME") {
    output_range[input_position].first = (low + stride - 1) / stride;
    output_range[input_position].second = (high + stride - 1) / stride;;
  } else if (pad_str == "VALID") {
    if (ceil_mode) {
      output_range[input_position].first = (low - kernel + pad + stride - 1) / stride + 1;
      output_range[input_position].second = (high - kernel + pad + stride - 1) / stride + 1;
    } else {
      output_range[input_position].first = (low - kernel + 1 + (stride - 1) + pad) / stride;
      output_range[input_position].second = (high - kernel + 1 + (stride - 1) + pad) / stride;
    }
  } else {
    if (ceil_mode) {
      output_range[input_position].first = (low - kernel + pad + stride - 1) / stride + 1;
      output_range[input_position].second = (high - kernel + pad + stride - 1) / stride + 1;
    } else {
      output_range[input_position].first = (low - kernel + pad) / stride + 1;
      output_range[input_position].second = (high - kernel + pad) / stride + 1;
    }
  }
  output_range[input_position].first = std::max(output_range[input_position].first, kDynamicRangeLowerBound);
  output_range[input_position].second = std::min(output_range[input_position].second, kDynamicRangeUpperBound);
  if (high == -1) {
    output_range[input_position].second = high;
  }
}

static bool SetAvgPoolOutShapeRange(ge::Operator& op, ge::GeTensorDescPtr& input_tensor_desc, ge::GeTensorDescPtr& td,
                                    Format input_format, const std::string& data_format,
                                    const std::vector<int32_t>& ksize_list, const std::vector<int32_t>& strides_list,
                                    const std::vector<int32_t>& pad_list, const std::vector<int64_t>& dims_input,
                                    std::vector<int64_t> dim_vector, const bool& ceil_mode, const bool& global_pooling,
                                    const std::string& padding_mode) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);

  std::string input_format_str = format2str[input_format];

  if (input_format_str != "NHWC") {
    input_format_str = "NCHW";
  }
  int32_t h_input_position = input_format_str.find("H");
  int32_t w_input_position = input_format_str.find("W");
  int32_t c_input_position = input_format_str.find("C");
  int64_t input_h = dims_input[h_input_position];
  int64_t input_w = dims_input[w_input_position];

  int32_t h_output_position = data_format.find("H");
  int32_t w_output_position = data_format.find("W");

  int32_t strh = strides_list[h_output_position];
  int32_t strw = strides_list[w_output_position];
  int32_t padt = pad_list[0];
  int32_t padb = pad_list[1];
  int32_t padl = pad_list[2];
  int32_t padr = pad_list[3];
  int32_t kh = ksize_list[h_output_position];
  int32_t kw = ksize_list[w_output_position];

  // update pads if padding is SAME
  if (padding_mode == "SAME" && (input_h == -1 || input_w == -1)) {
    op.SetAttr("pads", {-1, -1, -1, -1});
    OP_LOGD(op_name.GetString(), "set pads to {-1, -1, -1, -1} when padding is SAME in dynamic_shape");
  }

  OP_LOGD(op_name.GetString(), "dynamic shape set range");
  std::vector<std::pair<int64_t, int64_t>> Input_range;
  input_tensor_desc->GetShapeRange(Input_range);
  if (input_h == -1 && !global_pooling) {
    dim_vector[h_output_position] = -1;
  }
  if (input_w == -1 && !global_pooling) {
    dim_vector[w_output_position] = -1;
  }
  if (!Input_range.empty() && dims_input.size() == Input_range.size()) {
    std::vector<std::pair<int64_t, int64_t>> output_range(Input_range);
    output_range[c_input_position] = Input_range[c_input_position];
    output_range[h_input_position] = std::make_pair(dim_vector[h_input_position], dim_vector[h_input_position]);
    output_range[w_input_position] = std::make_pair(dim_vector[w_input_position], dim_vector[w_input_position]);
    if (input_h == -1 && !global_pooling) {
      vector<int32_t> attrParams_h = {strh, padt + padb, kh, h_input_position, h_output_position};
      SetAvgPoolOutRange(padding_mode, attrParams_h, ceil_mode, Input_range, output_range);
    }
    if (input_w == -1 && !global_pooling) {
      vector<int32_t> attrParams_w = {strw, padl + padr, kw, w_input_position, w_output_position};
      SetAvgPoolOutRange(padding_mode, attrParams_w, ceil_mode, Input_range, output_range);
    }
    td->SetShapeRange(output_range);
  }
  td->SetShape(GeShape(dim_vector));
  return true;
}

IMPLEMT_INFERFUNC(AvgPool, AvgPoolInferShape) {
  OP_LOGD(TbeGetName(op), "Enter AvgPoolInferShape");
  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::ConstGeTensorDescPtr input_desc = op_desc->GetInputDescPtr(0);
  if (input_desc == nullptr) {
    std::string err_msg = OtherErrMsg("can't get input desc");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  const GeShape &input_shape = input_desc->GetShape();
  Format input_format = input_desc->GetFormat();
  DataType input_dtype = input_desc->GetDataType();
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc(0);
  if (output_desc == nullptr) {
    std::string err_msg = OtherErrMsg("can't get output desc");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  GeShape &output_shape = output_desc->MutableShape();

  // get input ksize
  std::vector<int32_t> ksize_list;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksize_list)) {
    std::string err_msg = GetInputInvalidErrMsg("ksize_list");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (ksize_list.size() != DIM_SIZE4) {
    std::string err_msg1 = GetAttrValueErrMsg("ksize_list", std::to_string(ksize_list.size()), ConcatString(DIM_SIZE4));
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // get input strides
  std::vector<int32_t> strides_list;
  if (GRAPH_SUCCESS != op.GetAttr("strides", strides_list)) {
    std::string err_msg = GetInputInvalidErrMsg("strides_list");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (strides_list.size() != DIM_SIZE4) {
    std::string err_msg = GetAttrValueErrMsg("strides_list", std::to_string(strides_list.size()), ConcatString(DIM_SIZE4));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // get input data_format
  std::string data_format;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", data_format)) {
    std::string err_msg = GetInputInvalidErrMsg("data_format");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // get input paddingMode
  std::string padding_mode;
  if (GRAPH_SUCCESS != op.GetAttr("padding", padding_mode)) {
    std::string err_msg = GetInputInvalidErrMsg("padding");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (padding_mode != "SAME" && padding_mode != "VALID") {
    string expected_format_list = ConcatString("SAME,VALID");
    std::string err_msg = GetInputFormatNotSupportErrMsg(TbeGetName(op), expected_format_list, padding_mode);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int32_t n_position = 0;
  int32_t c_position = 0;
  int32_t h_position = 0;
  int32_t w_position = 0;
  if (input_format != FORMAT_NHWC) {
    // regard input format NCHW
    n_position = 0;
    c_position = 1;
    h_position = 2;
    w_position = 3;
  } else {
    n_position = 0;
    h_position = 1;
    w_position = 2;
    c_position = 3;
  }
  int64_t input_n = input_shape.GetDim(n_position);
  int64_t input_c = input_shape.GetDim(c_position);
  int64_t input_h = input_shape.GetDim(h_position);
  int64_t input_w = input_shape.GetDim(w_position);

  int32_t stride_h = 0;
  int32_t stride_w = 0;
  int32_t window_h = 0;
  int32_t window_w = 0;
  int32_t h_attr_position = 0;
  int32_t w_attr_position = 0;
  int64_t output_n = 0;
  int64_t output_h = 0;
  int64_t output_w = 0;
  int64_t output_c = 0;

  if (data_format == "NCHW") {
    window_h = ksize_list[2];
    window_w = ksize_list[3];
    stride_h = strides_list[2];
    stride_w = strides_list[3];
    h_attr_position = 2;
    w_attr_position = 3;
  } else if (data_format == "NHWC") {
    window_h = ksize_list[1];
    window_w = ksize_list[2];
    stride_h = strides_list[1];
    stride_w = strides_list[2];
    h_attr_position = 1;
    w_attr_position = 2;
  } else {
    std::string err_msg = OtherErrMsg("attr data format is wrong.");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  bool unknown_dim_num = input_shape.IsUnknownDimNum();
  if (!unknown_dim_num) {
    // set for global avg pool
    if (window_h == -1 && window_w == -1) {
      ksize_list[h_attr_position] = input_h;
      ksize_list[w_attr_position] = input_w;
      window_h = input_h;
      window_w = input_w;
    }
    op.SetAttr("ksize", ksize_list);
    output_n = input_n;
    output_c = input_c;
    if (padding_mode == "SAME") {
      output_h = (input_h + stride_h - 1) / stride_h;
      output_w = (input_w + stride_w - 1) / stride_w;
    } else {
      output_h = (input_h - window_h + 1 + (stride_h - 1)) / stride_h;
      output_w = (input_w - window_w + 1 + (stride_w - 1)) / stride_w;
    }
  } else {
    output_n = -1;
    output_c = -1;
    output_h = -1;
    output_w = -1;
  }

  output_desc->SetDataType(input_dtype);
  output_shape.SetDimNum(4);
  output_shape.SetDim(n_position, output_n);
  output_shape.SetDim(c_position, output_c);
  output_shape.SetDim(h_position, output_h);
  output_shape.SetDim(w_position, output_w);
  output_desc->SetShape(output_shape);


  bool is_dynamic = (input_n == -1) || (input_c == -1) || (input_h == -1) || (input_w == -1);
  if (is_dynamic) {
    std::vector<int32_t> pad_vec = {0, 0, 0, 0};
    bool is_global_pool = false;
    bool ceil_mode = false;
    auto input_tensor_desc = op_desc->MutableInputDesc("x");
    if (!SetAvgPoolOutShapeRange(op, input_tensor_desc, output_desc, input_format, data_format,
                                      ksize_list, strides_list, pad_vec, input_shape.GetDims(), output_shape.GetDims(),
                                      ceil_mode, is_global_pool, padding_mode)) {
      return GRAPH_FAILED;
    }
  }
  // fuzz build allow shape dim -1 with range
  // fuzz_build switch
  bool fuzz_build = false;
  op.GetAttr(ge::ATTR_NAME_FUZZ_BUILD, fuzz_build);
  // fuzz build
  if ((!unknown_dim_num) && fuzz_build) {
    OP_LOGD(TbeGetName(op), "start fuzz build.");
  }
  OP_LOGD(TbeGetName(op), "Leave AvgPoolInferShape");
  return GRAPH_SUCCESS;
}

static void InferHWAvgpool(int64_t kernel,int64_t stride, vector<int64_t>& output, vector<int64_t>& input,
                           int64_t& ori_input) {
    int64_t first_start = 0;
    int64_t second_start = 0;
    int64_t second_end = 0;
    int64_t start = 0;
    int64_t end = 0;
    first_start = output[0] * stride;
    second_start = output[1] * stride;
    second_end = std::min(second_start + kernel, ori_input);
    start = std::max(first_start, int64_t(0));
    end = second_end - 1;
    input = {start, end};
}

IMPLEMT_INFER_DATA_SLICE(AvgPool, AvgPoolInferDataSlice){
  OP_LOGD(TbeGetName(op), "Enter AvgPoolInferDataSlice.");
  auto inputTensorDesc = op.GetInputDesc("x");
  auto shape = inputTensorDesc.GetShape();
  std::vector<int64_t> dims_input = shape.GetDims();
  auto inputFormat = inputTensorDesc.GetFormat();

  int64_t inputH = 0;
  int64_t inputW = 0;
  if (inputFormat == FORMAT_NC1HWC0) {
    inputH = dims_input[2];
    inputW = dims_input[3];
  } else {
    OP_LOGE(TbeGetName(op), "Invalid inputFormat.");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> ksizeList;
  std::vector<int64_t> stridesList;
  std::string dataFormat;
  std::string paddingMode;
  op.GetAttr("ksize", ksizeList);
  op.GetAttr("strides", stridesList);
  op.GetAttr("data_format", dataFormat);
  op.GetAttr("padding", paddingMode);

  int64_t windowH = 0;
  int64_t windowW = 0;
  int64_t strideH = 0;
  if (dataFormat == "NHWC") {
    windowH = ksizeList[1];
    windowW = ksizeList[2];
    strideH = stridesList[1];
  } else if(dataFormat == "NCHW") {
    windowH = ksizeList[2];
    windowW = ksizeList[3];
    strideH = stridesList[2];
  } else {
    OP_LOGE(TbeGetName(op), "Invalid dataFormat.");
    return GRAPH_FAILED;
  }

  if (windowH == inputH && windowW == inputW) {
    OP_LOGD(TbeGetName(op), "Global pool can't calculate over lap.");
    return NO_OVERLAP_DIM;
  }
  if (paddingMode == "SAME") {
    OP_LOGD(TbeGetName(op), "Padding mode is same, can't calculate over lap.");
    return NO_OVERLAP_DIM;
  }

  vector<vector<int64_t>> y_data_slice = {{}, {}, {}, {}, {}};
  vector<vector<int64_t>> x_data_slice = {{}, {}, {}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_out = op_desc->MutableOutputDesc(0);
  GeTensorDescPtr tensor_desc_in = op_desc->MutableInputDesc(0);
  if (!ge::AttrUtils::GetListListInt(tensor_desc_out, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGE(TbeGetName(op), "no data slice, use default.");
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
        OP_LOGE(TbeGetName(op), "Set x data slice failed.");
        return GRAPH_FAILED;
      }
      return GRAPH_SUCCESS;
    }
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
    std::string err_msg = GetInputInvalidErrMsg("ksize");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (ksizeList.size() != DIM_SIZE4) {
    std::string err_msg = GetAttrSizeErrMsg("ksizeList", std::to_string(ksizeList.size()), std::to_string(DIM_SIZE4));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // get input strides
  std::vector<int32_t> stridesList;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stridesList)) {
    std::string err_msg = GetInputInvalidErrMsg("strides");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (stridesList.size() != DIM_SIZE4) {
    std::string err_msg = GetAttrSizeErrMsg("stridesList()", std::to_string(stridesList.size()), std::to_string(DIM_SIZE4));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // get input paddingMode
  std::string paddingMode;
  if (GRAPH_SUCCESS != op.GetAttr("padding_mode", paddingMode)) {
    std::string err_msg = GetInputInvalidErrMsg("padding_mode");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (paddingMode != "SAME" && paddingMode != "VALID" && paddingMode != "CALCULATED") {
    string expected_padding_list = ConcatString("SAME, VALID, CALCULATED");
    std::string err_msg = GetAttrValueErrMsg("paddingMode", paddingMode, expected_padding_list);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // get input pads
  std::vector<int32_t> padVec;
  if (GRAPH_SUCCESS != op.GetAttr("pads", padVec)) {
    std::string err_msg = GetInputInvalidErrMsg("pads");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (padVec.size() != DIM_SIZE4) {
    std::string err_msg = GetAttrSizeErrMsg("padVec", std::to_string(padVec.size()), std::to_string(DIM_SIZE4));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // get input data_format
  std::string dataFormat;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
    std::string err_msg = GetInputInvalidErrMsg("data_format");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (dataFormat != "NHWC" && dataFormat != "NCHW" && dataFormat != "NC1HWC0") {
    string expected_dataformat_list = ConcatString("NHWC, VALID,NCHW NC1HWC0");
    std::string err_msg = GetAttrValueErrMsg("dataFormat", dataFormat, expected_dataformat_list);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (dataFormat == "NHWC") {
    if (ksizeList[0] != 1 || ksizeList[3] != 1) {
      string expected_ksizeList_list = ConcatString("ksizeList[0]=1", ",", "ksizeList[3]=1");
      string ksizeList_list = ConcatString(ksizeList[0], ",", ksizeList[3]);
      std::string err_msg = GetAttrValueErrMsg("ksizeList", ksizeList_list, expected_ksizeList_list);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    if (stridesList[0] != 1 || stridesList[3] != 1) {
      string expected_stridesList_list = ConcatString("stridesList[0]=1", ",", "stridesList[3]=1");
      string stridesList_list = ConcatString(stridesList[0], ",", stridesList[3]);
      std::string err_msg = GetAttrValueErrMsg("stridesList", stridesList_list, expected_stridesList_list);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    if (stridesList[1] < 1 || stridesList[2] < 1) {
      string expected_stridesList_list = ConcatString("stridesList[1] > 0", ",", "stridesList[3] > 0");
      string stridesList_list = ConcatString(stridesList[1], ",", stridesList[2]);
      std::string err_msg = GetAttrValueErrMsg("stridesList", stridesList_list, expected_stridesList_list);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }
  if (dataFormat == "NCHW" || dataFormat == "NC1HWC0") {
    if (ksizeList[0] != 1 || ksizeList[1] != 1) {
      string expected_ksizeList_list = ConcatString("ksizeList[0]=1", ",", "ksizeList[3]=1");
      string ksizeList_list = ConcatString(ksizeList[0], ",", ksizeList[3]);
      std::string err_msg = GetAttrValueErrMsg("ksizeList", ksizeList_list, expected_ksizeList_list);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    if (stridesList[0] != 1 || stridesList[1] != 1) {
      string expected_stridesList_list = ConcatString("stridesList[0]=1", ",", "stridesList[3]=1");
      string stridesList_list = ConcatString(stridesList[0], ",", stridesList[3]);
      std::string err_msg = GetAttrValueErrMsg("stridesList", stridesList_list, expected_stridesList_list);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    if (stridesList[2] < 1 || stridesList[3] < 1) {
      string expected_stridesList_list = ConcatString("stridesList[2] > 0", ",", "stridesList[3] > 0");
      string stridesList_list = ConcatString(stridesList[2], ",", stridesList[3]);
      std::string err_msg = GetAttrValueErrMsg("stridesList", stridesList_list, expected_stridesList_list);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AvgPoolV2InferShape) {
  OP_LOGD(op.GetName().c_str(), "Enter AvgPoolV2InferShape");

  const size_t DIM_SIZE4 = 4;
  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::ConstGeTensorDescPtr input_desc = op_desc->GetInputDescPtr(0);
  if (input_desc == nullptr) {
    std::string err_msg = OtherErrMsg("can't get input desc");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  const GeShape &input_shape = input_desc->GetShape();
  Format input_format = input_desc->GetFormat();
  DataType input_dtype = input_desc->GetDataType();
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc(0);
  if (output_desc == nullptr) {
    std::string err_msg = OtherErrMsg("can't get output desc");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  GeShape &output_shape = output_desc->MutableShape();

  // get input kszie
  std::vector<int32_t> ksize_list;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksize_list)) {
    std::string err_msg = GetInputInvalidErrMsg("ksize");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // get input strides
  std::vector<int32_t> strides_list;
  if (GRAPH_SUCCESS != op.GetAttr("strides", strides_list)) {
    std::string err_msg = GetInputInvalidErrMsg("strides");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // get input data_format
  std::string data_format;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", data_format)) {
    std::string err_msg = GetInputInvalidErrMsg("data_format");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // get input padding_mode
  std::string padding_mode;
  if (GRAPH_SUCCESS != op.GetAttr("padding_mode", padding_mode)) {
    std::string err_msg = GetInputInvalidErrMsg("padding_mode");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);

    return GRAPH_FAILED;
  }

  // get input strides
  std::vector<int32_t> pad_vec;
  if (GRAPH_SUCCESS != op.GetAttr("pads", pad_vec)) {
    std::string err_msg = GetInputInvalidErrMsg("pads");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // get input global_padding
  bool global_pooling;
  if (GRAPH_SUCCESS != op.GetAttr("global_pooling", global_pooling)) {
    std::string err_msg = GetInputInvalidErrMsg("global_pooling");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // get input ceil_mode
  bool ceil_mode;
  if (GRAPH_SUCCESS != op.GetAttr("ceil_mode", ceil_mode)) {
    std::string err_msg = GetInputInvalidErrMsg("ceil_mode");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // input format mast equals to data_format
  if ((input_format == FORMAT_NCHW && data_format != "NCHW") || (input_format == FORMAT_NHWC && data_format != "NHWC")) {
    string err_msg1 = ConcatString("Input format and data_format is not same. input_format:",input_format, ", data_format:",data_format);
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // INFER
  bool unknownRank = input_shape.IsUnknownDimNum();
  if (!unknownRank && input_shape.GetDimNum() != DIM_SIZE4) {
    std::string err_msg = GetAttrSizeErrMsg("x", std::to_string(input_shape.GetDimNum()), std::to_string(DIM_SIZE4));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  int32_t position_n = data_format.find("N");
  int32_t position_c = data_format.find("C");
  int32_t position_w = data_format.find("W");
  int32_t position_h = data_format.find("H");
  int64_t input_n = input_shape.GetDim(position_n);
  int64_t input_c = input_shape.GetDim(position_c);
  int64_t input_h = input_shape.GetDim(position_h);
  int64_t input_w = input_shape.GetDim(position_w);
  int64_t output_n = 0;
  int64_t output_c = 0;
  int64_t output_h = 0;
  int64_t output_w = 0;

  int32_t window_h, window_w;
  if (!unknownRank) {
    if (global_pooling) {
      window_h = input_h;
      window_w = input_w;
      strides_list[position_h] = input_h;
      strides_list[position_w] = input_w;
      pad_vec = {0, 0, 0, 0};
    } else {
      window_h = ksize_list[position_h];
      window_w = ksize_list[position_w];
    }
    output_n = input_n;
    output_c = input_c;

    int32_t stride_h = strides_list[position_h];
    int32_t stride_w = strides_list[position_w];
    if (padding_mode == "SAME") {
      output_h = (input_h + stride_h - 1) / stride_h;
      output_w = (input_w + stride_w - 1) / stride_w;
    } else if (padding_mode == "VALID") {
      if (ceil_mode) {
        output_h = (input_h - window_h + pad_vec[0] + pad_vec[1] + stride_h - 1) / stride_h + 1;
        output_w = (input_w - window_w + pad_vec[2] + pad_vec[3] + stride_w - 1) / stride_w + 1;
      } else {
        output_h = (input_h - window_h + 1 + stride_h - 1) / stride_h;
        output_w = (input_w - window_w + 1 + stride_w - 1) / stride_w;
      }
    } else {
      for (size_t i = 0; i < input_shape.GetDimNum(); i++) {
        if (ceil_mode) {
          output_h = (input_h - window_h + pad_vec[0] + pad_vec[1] + stride_h - 1) / stride_h + 1;
          output_w = (input_w - window_w + pad_vec[2] + pad_vec[3] + stride_w - 1) / stride_w + 1;
        } else {
          output_h = (input_h - window_h + pad_vec[0] + pad_vec[1]) / stride_h + 1;
          output_w = (input_w - window_w + pad_vec[2] + pad_vec[3]) / stride_w + 1;
        }
      }
    }
  } else {
    output_n = -1;
    output_c = -1;
    output_h = -1;
    output_w = -1;
  }

  output_shape.SetDimNum(4);
  output_shape.SetDim(position_n, output_n);
  output_shape.SetDim(position_c, output_c);
  output_shape.SetDim(position_h, output_h);
  output_shape.SetDim(position_w, output_w);
  output_desc->SetShape(output_shape);
  if (input_dtype == ge::DT_INT8) {
    output_desc->SetDataType(ge::DT_INT32);
  } else {
    output_desc->SetDataType(input_dtype);
  }


  bool isDynamic = (input_n == -1) || (input_c == -1) || (input_h == -1) || (input_w == -1);
  if (isDynamic) {
    auto input_tensor_desc = op_desc->MutableInputDesc("x");
    if (!SetAvgPoolOutShapeRange(op, input_tensor_desc, output_desc, input_format, data_format,
                                 ksize_list, strides_list, pad_vec, input_shape.GetDims(), output_shape.GetDims(),
                                 ceil_mode, global_pooling, padding_mode)) {
      return GRAPH_FAILED;
    }
  }
  OP_LOGD(op.GetName().c_str(), "Leave AvgPoolV2InferShape");
  return GRAPH_SUCCESS;
}

// cal new_pad_value for avgpool_v2 according to conv(data slice info)
static void InferHWAvgpoolv2(int32_t input, int32_t kernel, int32_t pad, int32_t stride,
                             int32_t dilation, vector<int64_t> output_slice, vector<int64_t>& data_slice,
                             int64_t* start_add_pad, int64_t* end_add_pad)
{
    if ((start_add_pad == nullptr) || (end_add_pad == nullptr)) {
        OP_LOGE("InferHWAvgpoolv2:", "Get start_add_pad or end_add_pad failed.");
        return;
    }
    OP_LOGD("InferHWAvgpoolv2:", "input:%d kernel:%d pad:%d stride:%d dilation:%d start_add_pad:%ld end_add_pad:%ld\n",
            input, kernel, pad, stride, dilation, *start_add_pad, *end_add_pad);
    if (output_slice.size() >= 2) { // output slice has 2 dim(h_start, h_end)
        // calc start rule: (i_start + pad_h)/stride_h = output_start
        int64_t i_start = output_slice[0] * stride - pad;
        if (i_start < 0) {
            *start_add_pad = -i_start;
            i_start = 0;
        } else {
            *start_add_pad = 0;
        }
        // calc end rule: (iend_start + pad_h)/stride_h = output_end
        // iend_end = iend_start + dilation*(kernel_h-1)
        int64_t i_end = output_slice[1] * stride - pad + dilation * (kernel - 1);
        if (i_end >= input) {
            *end_add_pad = i_end - input + 1;
            i_end = input - 1;
        } else {
            *end_add_pad = 0;
        }
        data_slice = {i_start, i_end};
        OP_LOGD("InferHWAvgpoolv2:", "i_start:%ld i_end:%ld output_slice[0]:%ld output_slice[1]:%ld\n",
                i_start, i_end, output_slice[0], output_slice[1]);
    }
}

static void get_corrected_pad(int64_t* input_pad)
{
    if (input_pad == nullptr) {
        OP_LOGE("get_corrected_pad:", "Get input_pad failed.");
        return;
    }
    if (*input_pad < 0) {
        *input_pad = 0;
    }
}

  // The code here refers to the implementation of _calculate_pads before calling conv2d_compute in avg_pool_v2.py
static void calculate_pad(string padding, int64_t input_h, int64_t input_w, int64_t stride_h,
                          int64_t stride_w, int64_t ksize_h, int64_t ksize_w, vector<int64_t> dilations,
                          vector<int64_t>& pad, vector<int64_t> pads, bool ceilMode)
{
    if ((pads.size() < 4) || (dilations.size() < 4)) {
        OP_LOGE("calculate_pad:", "Get pad_vec or  dilations failed.");
    } else if ((stride_h == 0) || (stride_w == 0)) {
        OP_LOGE("calculate_pad:", "Divisor stride_h or  stride_w is 0.");
    } else {
        OP_LOGD("calculate_pad:", "stride_h:%d stride_w:%d ksize_h:%d ksize_w:%d\n", stride_h, stride_w, ksize_h, ksize_w);
        int64_t output_h = 0;
        int64_t output_w = 0;
        int64_t pad_row = 0;
        int64_t pad_col = 0;
        int64_t pad_top = 0;
        int64_t pad_bottom = 0;
        int64_t pad_left = 0;
        int64_t pad_right = 0;
        int64_t ho = 0;
        int64_t wo = 0;
        if (padding == "SAME") {
            output_h = floor((input_h + stride_h - 1) / stride_h);
            output_w = floor((input_w + stride_w - 1) / stride_w);
            pad_row = (output_h - 1) * stride_h + ((ksize_h - 1) * dilations[0] + 1) - input_h;
            pad_col = (output_w - 1) * stride_w + ((ksize_w - 1) * dilations[1] + 1) - input_w;
            pad_top = floor(pad_row / 2);
            pad_bottom = pad_row - pad_top;
            pad_left = floor(pad_col / 2);
            pad_right = pad_col - pad_left;
            get_corrected_pad(&pad_top);
            get_corrected_pad(&pad_bottom);
            get_corrected_pad(&pad_left);
            get_corrected_pad(&pad_right);
            pad = {pad_top, pad_bottom, pad_left, pad_right};
        } else if (padding == "CALCULATED") {
            pad_top= pads[0];
            pad_bottom= pads[1];
            pad_left= pads[2];
            pad_right= pads[3];

            if (ceilMode) {
                ho = floor((input_h - ksize_h + pad_top + pad_bottom + stride_h - 1) / stride_h) + 1;
                wo = floor((input_w - ksize_w + pad_left + pad_right + stride_w - 1) / stride_w) + 1;
                pad_bottom = (ho - 1) * stride_h + ksize_h - input_h - pad_top;
                pad_right = (wo - 1) * stride_w + ksize_w - input_w - pad_left;
            } else {
                ho = floor((input_h - ksize_h + pad_top + pad_bottom) / stride_h) + 1;
                wo = floor((input_w - ksize_w + pad_left + pad_right) / stride_w) + 1;
                pad_bottom = (ho - 1) * stride_h + ksize_h - input_h - pad_top;
                pad_right = (wo - 1) * stride_w + ksize_w - input_w - pad_left;
                get_corrected_pad(&pad_bottom);
                get_corrected_pad(&pad_right);
            }
            pad = {pad_top, pad_bottom, pad_left, pad_right};

        } else {
            pad = {0, 0, 0, 0};
        }
    }
}

IMPLEMT_INFER_DATA_SLICE(AvgPoolV2, AvgPoolV2InferDataSlice){
    AscendString op_name;
    CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);
    OP_LOGD(op_name.GetString(), "Enter AvgPoolV2InferDataSlice");
    auto inputTensorDesc = op.GetInputDescByName("x");
    auto shape = inputTensorDesc.GetShape();
    std::vector<int64_t> dims_input = shape.GetDims();
    auto inputFormat = inputTensorDesc.GetFormat();

    if (IsUnknownRankShape(dims_input)) {
        OP_LOGD(op_name.GetString(), "input x shape [-2] do not support split.");
        return GRAPH_FAILED;
    }

    // get input ksize
    std::vector<int64_t> ksizeList;
    if (GRAPH_SUCCESS != op.GetAttr("ksize", ksizeList)) {
        OP_LOGD(op_name.GetString(), "failed to get op Attr ksizeList");
        return GRAPH_FAILED;
    }
    // get input strides
    std::vector<int64_t> stridesList;
    if (GRAPH_SUCCESS != op.GetAttr("strides", stridesList)) {
        OP_LOGD(op_name.GetString(), "failed to get op Attr stridesList");
        return GRAPH_FAILED;
    }
    // get input data_format
    std::string dataFormat;
    if (GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
        OP_LOGD(op_name.GetString(), "failed to get op Attr dataFormat");
        return GRAPH_FAILED;
    }
    // get input padding_mode
    std::string paddingMode;
    if (GRAPH_SUCCESS != op.GetAttr("padding_mode", paddingMode)) {
        OP_LOGD(op_name.GetString(), "failed to get op Attr padding_mode");
        return GRAPH_FAILED;
    }

    // get input ceil_mode
    bool ceil_mode;
    if (GRAPH_SUCCESS != op.GetAttr("ceil_mode", ceil_mode)) {
        OP_LOGD(op_name.GetString(), "failed to get op Attr ceil_mode");
        return GRAPH_FAILED;
    }
    // get input pads
    std::vector<int64_t> pad_vec;
    if (GRAPH_SUCCESS != op.GetAttr("pads", pad_vec)) {
        OP_LOGD(op_name.GetString(), "failed to get op Attr pad_vec");
        return GRAPH_FAILED;
    }

    if ((dims_input.size() < 4) || (ksizeList.size() < 4) || (stridesList.size() < 4) || (pad_vec.size() < 4)) {
        OP_LOGE(op_name.GetString(), "dims_input size or ksize size or strides size or pad size less then 4.");
        return GRAPH_FAILED;
    }

    int64_t inputH = 0;
    int64_t inputW = 0;
    if (inputFormat == FORMAT_NC1HWC0) {
        inputH = dims_input[2];
        inputW = dims_input[3];
    } else {
        OP_LOGE(op_name.GetString(), "Invalid inputFormat.");
        return GRAPH_FAILED;
    }

    int64_t windowH = 0;
    int64_t windowW = 0;
    int64_t strideH = 0;
    int64_t strideW = 0;
    if (dataFormat == "NHWC") {
        windowH = ksizeList[1];
        windowW = ksizeList[2];
        strideH = stridesList[1];
        strideW = stridesList[2];
    } else if (dataFormat == "NCHW") {
        windowH = ksizeList[2];
        windowW = ksizeList[3];
        strideH = stridesList[2];
        strideW = stridesList[3];
    } else {
        OP_LOGE(op_name.GetString(), "Invalid dataFormat.");
        return GRAPH_FAILED;
    }
    if (windowH == inputH && windowW == inputW) {
        OP_LOGD(op_name.GetString(), "Global pool can't calculate over lap.");
        return NO_OVERLAP_DIM;
    }

    vector<int64_t> dilations = {1, 1, 1, 1};
    vector<int64_t> pad_list;
    calculate_pad(paddingMode, inputH, inputW, strideH, strideW, windowH,
                  windowW, dilations, pad_list, pad_vec, ceil_mode);
    if (pad_list.size() < 4) {
        OP_LOGE(op_name.GetString(), "pad_list size less then 4.");
        return GRAPH_FAILED;
    }
    OP_LOGD(op_name.GetString(), "avgpoolv2 calculate_pad_list  is [%d,%d,%d,%d], ori_pad_vec is [%d,%d,%d,%d]",
            pad_list[0], pad_list[1], pad_list[2], pad_list[3], pad_vec[0], pad_vec[1], pad_vec[2], pad_vec[3]);

    vector<vector<int64_t>> y_data_slice = {{}, {}, {}, {}, {}};
    vector<vector<int64_t>> x_data_slice = {{}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    GeTensorDescPtr tensor_desc_out = op_desc->MutableOutputDesc("y");
    GeTensorDescPtr tensor_desc_in = op_desc->MutableInputDesc("x");
    if (!ge::AttrUtils::GetListListInt(tensor_desc_out, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
        OP_LOGE(op_name.GetString(), "no data slice, use default.");
        return GRAPH_FAILED;
    }

    int64_t padt = pad_list[0];
    int64_t padl = pad_list[2];
    int64_t dilh = 1;
    int64_t dilw = 1;
    bool haveSlice = false;
    vector<int64_t> new_pad_lists = pad_list;
    for (unsigned i = 0; i < y_data_slice.size(); i++) {
        if (y_data_slice[i].size() > 0) {
            haveSlice = true;
            if (i == 2) {
                vector<int64_t> ih_slice;
                int64_t top_add_pad = pad_list[0];
                int64_t bom_add_pad = pad_list[1];
                InferHWAvgpoolv2(inputH, windowH, padt, strideH, dilh, y_data_slice[i],
                                 ih_slice, &top_add_pad, &bom_add_pad);
                OP_LOGD(op_name.GetString(), "Avgpoolv2 h axis slice ori_scope is [%d,%d], output scope is [%d,%d]",
                        ih_slice[0], ih_slice[1], y_data_slice[i][0], y_data_slice[i][1]);
                new_pad_lists[0] = top_add_pad;
                new_pad_lists[1] = bom_add_pad;
                x_data_slice[i] = ih_slice;
            } else if (i == 3) {
                vector<int64_t> iw_slice;
                int64_t left_add_pad = pad_list[2];
                int64_t right_add_pad = pad_list[3];
                InferHWAvgpoolv2(inputW, windowW, padl, strideW, dilw, y_data_slice[i],
                                iw_slice, &left_add_pad, &right_add_pad);
                OP_LOGD(op_name.GetString(), "Avgpoolv2 w axis slice ori_scope is [%d,%d], output scope is [%d,%d]",
                        iw_slice[0], iw_slice[1], y_data_slice[i][0], y_data_slice[i][1]);
                new_pad_lists[2] = left_add_pad;
                new_pad_lists[3] = right_add_pad;
                x_data_slice[i] = iw_slice;
            }
        }
    }
    op.SetAttr("pads", new_pad_lists);
    OP_LOGD(op_name.GetString(), "avgpoolv2 new pad lists is [%d,%d,%d,%d]", new_pad_lists[0],
            new_pad_lists[1], new_pad_lists[2], new_pad_lists[3]);
    for (unsigned i = 0; i < x_data_slice.size(); i++) {
        if (x_data_slice[i].size() > 0) {
            if (!AttrUtils::SetListListInt(tensor_desc_in, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
                OP_LOGE(op_name.GetString(), "Set x data slice failed.");
                return GRAPH_FAILED;
            }
            return GRAPH_SUCCESS;
        }
    }

    if (!haveSlice) {
        return GRAPH_FAILED;
  }
    OP_LOGD(op_name.GetString(), "AvgPoolV2InferDataSlice success");
    return NO_OVERLAP_DIM;
}


COMMON_INFER_FUNC_REG(AvgPoolV2, AvgPoolV2InferShape);
VERIFY_FUNC_REG(AvgPoolV2, AvgPoolV2Verify);
INFER_DATA_SLICE_FUNC_REG(AvgPoolV2, AvgPoolV2InferDataSlice);
// ----------------AvgPoolV2 End-------------------

// ----------------AvgPool3D-------------------
bool IsAllVal(const vector<int64_t>& vec, int64_t val) {
  for (auto i : vec) {
    if (i != val) {
      return false;
    }
  }
  return true;
}

IMPLEMT_VERIFIER(AvgPool3D, AvgPool3DVerify) {
  ge::AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto x_shape = op_desc->MutableInputDesc("x")->MutableShape().GetDims();
  auto ksize = op.get_attr_ksize();
  auto strides = op.get_attr_strides();
  bool invalid_param = x_shape.size() != 5;
  bool invalid_ksize = ksize.size() != 1 && ksize.size() != 3 && ksize.size() != 5;
  bool invalid_strides = strides.size() != 1 && strides.size() !=3 && strides.size() != 5;
  bool invalid_pads = false;
  std::vector<int32_t> pads;
  if (op.GetAttr("pads", pads) == GRAPH_SUCCESS) {
    invalid_pads = pads.size() != 6;
  }
  if (invalid_param || invalid_ksize || invalid_strides || invalid_pads) {
    OP_LOGE(op_name.GetString(), "AvgPool3D check x_shape or ksize or strides size or pads size is invalid!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

void AvgPool3DCalOutputRange(int32_t ksize, int32_t stride, const string& padding,
                             std::pair<int64_t, int64_t>& fmap_range,
                             std::pair<int64_t, int64_t> &output_range) {
  fmap_range.first = fmap_range.first == -1 ? kDynamicRangeUpperBound : fmap_range.first;
  fmap_range.second = fmap_range.second == -1 ? kDynamicRangeUpperBound : fmap_range.second;
  if (padding == "SAME") {
    output_range.first = std::max((fmap_range.first + stride - 1) / stride, kDynamicRangeLowerBound);
    output_range.second = std::min((fmap_range.second + stride - 1) / stride, kDynamicRangeUpperBound);
  } else {
    output_range.first = std::max((fmap_range.first - ksize) / stride + 1, kDynamicRangeLowerBound);
    output_range.second = std::min((fmap_range.second - ksize) / stride + 1, kDynamicRangeUpperBound);
  }
}

bool GetAvgPool3DOutputRange(ge::OpDescPtr &op_desc, const vector<int64_t> &ksize_dhw,
                             const vector<int64_t> &strides_dhw, const string& padding, const string& data_format,
                             vector<int64_t>& output_shape, vector<pair<int64_t, int64_t>>& output_range) {
  auto fmap_desc = op_desc->MutableInputDesc("x");
  Format fmap_format = fmap_desc->GetFormat();
  CHECK(fmap_format != FORMAT_NDHWC && fmap_format != FORMAT_NCDHW,
        OP_LOGE("AvgPool3D", "unsupport fmap desc format."),
        return false);
  CHECK_KEY_IN_MAP(format2str, fmap_desc->GetFormat(), "fmap_format", return false);
  string fmap_format_str = format2str[fmap_desc->GetFormat()];
  vector<pair<int64_t, int64_t>> fmap_range;
  fmap_desc->GetShapeRange(fmap_range);
  CHECK(fmap_range.size() != kAvgPool3DOriShapeDim,
        OP_LOGE("AvgPool3D", "input x range dim is invalid, expect:%lu, real:%lu.",
                kAvgPool3DOriShapeDim, fmap_range.size()),
        return false);
  auto n_position = data_format.find("N");
  auto c_position = data_format.find("C");
  auto d_position = data_format.find("D");
  auto h_position = data_format.find("H");
  auto w_position = data_format.find("W");

  auto f_n_position = fmap_format_str.find("N");
  auto f_c_position = fmap_format_str.find("C");
  auto f_d_position = fmap_format_str.find("D");
  auto f_h_position = fmap_format_str.find("H");
  auto f_w_position = fmap_format_str.find("W");

  if (n_position == std::string::npos || c_position == std::string::npos || d_position == std::string::npos ||
      h_position == std::string::npos || w_position == std::string::npos) {
        return false;
      }
  if (f_n_position == std::string::npos || f_c_position == std::string::npos || f_d_position == std::string::npos ||
      f_h_position == std::string::npos || f_w_position == std::string::npos) {
        return false;
      }
  if  (output_range.size() < data_format.size() || fmap_range.size() < fmap_format_str.size()) {
    return false;
  }

  output_range[n_position] = fmap_range[f_n_position];
  output_range[c_position] = fmap_range[f_c_position];
  AvgPool3DCalOutputRange(ksize_dhw[0], strides_dhw[0], padding, fmap_range[f_d_position],
                          output_range[d_position]);
  AvgPool3DCalOutputRange(ksize_dhw[1], strides_dhw[1], padding, fmap_range[f_h_position],
                          output_range[h_position]);
  AvgPool3DCalOutputRange(ksize_dhw[2], strides_dhw[2], padding, fmap_range[f_w_position],
                          output_range[w_position]);
  for (size_t i = 0; i < fmap_range.size(); ++i) {
    output_shape[i] = output_range[i].first == output_range[i].second ? output_range[i].first : -1;
  }
  return true;
}

void UpdateAvgPool3DPads(int32_t id, int32_t ih, int32_t iw, int32_t kd,
                         int32_t kh, int32_t kw, int32_t strd, int32_t strh,
                         int32_t strw, vector<int64_t>& pads) {
  int32_t pads_d = 0;
  int32_t pads_h = 0;
  int32_t pads_w = 0;

  pads_d = std::max((id + strd - 1) / strd * strd + kd - strd - id, 0);
  pads_h = std::max((ih + strh - 1) / strh * strh + kh - strh - ih, 0);
  pads_w = std::max((iw + strw - 1) / strw * strw + kw - strw - iw, 0);

  pads[0] = pads_d / 2;
  pads[1] = pads_d - pads[0];
  pads[2] = pads_h / 2;
  pads[3] = pads_h - pads[2];
  pads[4] = pads_w / 2;
  pads[5] = pads_w - pads[4];
}

IMPLEMT_INFERFUNC(AvgPool3D, AvgPool3DInferShape) {
  ge::AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  auto input_tensor_desc = op.GetInputDescByName("x");
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
    OP_LOGE(op_name.GetString(), "Failed to get attr strides or ksize in AvgPool3D.");
    return GRAPH_FAILED;
  }
  if (strd == 0 || strh == 0 || strw == 0 || kd == 0 || kh == 0 || kw == 0) {
    OP_LOGE(op_name.GetString(), "Strides or ksize invalid.");
    return GRAPH_FAILED;
  }

  bool is_dynamic = IsUnknown(dims_input);
  if (is_dynamic) {
    string padding;
    vector<int64_t> pads(kAvgPool3DPadsDim, 0);
    if (op.GetAttr("padding", padding) == GRAPH_SUCCESS) {
      CHECK(padding != "SAME" && padding != "VALID",
            OP_LOGE(op_name.GetString(),
                    "Padding is invalid, expect SAME or VALID, but real: %s.",
                    padding.c_str()),
            return GRAPH_FAILED);
    } else if (op.GetAttr("pads", pads) == GRAPH_SUCCESS) {
      padding = IsAllVal(pads, 0) ? "VALID" : "SAME";
    } else {
      OP_LOGE(op_name.GetString(),
              "Can't get padding or pads attribute.");
      return GRAPH_FAILED;
    }
    auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
    std::vector<int64_t> output_shape(kAvgPool3DOriShapeDim, 0);
    std::vector<std::pair<int64_t, int64_t>> output_range(kAvgPool3DOriShapeDim);
    std::vector<int64_t> ksize_dhw = {kd, kh, kw};
    std::vector<int64_t> strides_dhw = {strd, strh, strw};

    CHECK_KEY_IN_MAP(format2str, input_format, "input_format", return GRAPH_FAILED);
    std::string dx_format_str = format2str[input_format];
    CHECK(!GetAvgPool3DOutputRange(op_desc, ksize_dhw, strides_dhw, padding, dx_format_str,
                            output_shape, output_range),
          OP_LOGE(op_name.GetString(), "Set dynamic output shape and range failed."),
          return GRAPH_FAILED);

    pads.assign(kAvgPool3DPadsDim, 0);
    if (padding == "SAME") {
      if (id != -1 && ih != -1 && iw != -1) {
        UpdateAvgPool3DPads(id, ih, iw, kd, kh, kw, strd, strh, strw, pads);
      } else {
        pads.assign(kAvgPool3DPadsDim, -1);
      }
    }
    op.SetAttr("pads", pads);
    auto output_desc = op_desc->MutableOutputDesc("y");
    DataType input_dtype = op_desc->MutableInputDesc("x")->GetDataType();
    output_desc->SetShape(GeShape(output_shape));
    output_desc->SetShapeRange(output_range);
    output_desc->SetDataType(input_dtype);
    return GRAPH_SUCCESS;
  }

  std::vector<int32_t> pad_vec;
  if (GetPadsByPadding(op, id, ih, iw, kd, kh, kw, strd, strh, strw, pad_vec)) {
    op.SetAttr("pads",pad_vec);
  } else {
    CHECK(op.GetAttr("pads", pad_vec) != GRAPH_SUCCESS, OP_LOGE(op_name.GetString(), "get padding or pads failed."),
          return GRAPH_FAILED);
  }

  CHECK(pad_vec.size() != 6, OP_LOGE(op_name.GetString(), "pads size must be 6."), return GRAPH_FAILED);

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

  TensorDesc tensor_out = op.GetOutputDescByName("y");
  DataType input_dtype = input_tensor_desc.GetDataType();
  Shape output_shape(dim_vector);
  tensor_out.SetShape(output_shape);
  tensor_out.SetDataType(input_dtype);
  if (op.UpdateOutputDesc("y", tensor_out) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
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

  if (!GetStridesAndKSize(op, x_format, stride_d, stride_h, stride_w, filter_d, filter_h, filter_w)) {
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
static bool CheckGlobal(const int32_t d_dim,
                        const int32_t h_dim,
                        const int32_t w_dim) {
  if (d_dim == 1 && h_dim == 1 && w_dim == 1) {
    return true;
  }
  return false;
}

static graphStatus VerifyDataSlice(const ge::Operator& op, const vector<vector<int64_t>>& data_slice) {
  ge::AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  // check data_slice attr
  CHECK(data_slice.size() != kAvgPool3DGradDataSlice,
        CUBE_INNER_ERR_REPORT(op_name.GetString(), "data_slice's size should be 6."), return GRAPH_FAILED);

  // no support C0 axis
  CHECK(data_slice[kAvgPool3DGradDataSlice - 1].size() != 0,
        CUBE_INNER_ERR_REPORT(op_name.GetString(), "no support to cut C0 axis."), return NOT_SUPPORT_SLICE);

  // check valid slice num in data slice
  int32_t valid_cnt = 0;
  for (uint32_t i = 0; i < data_slice.size(); ++i) {
    if (data_slice[i].size() == 0) {
      continue;
    }

    CHECK(data_slice[i].size() != 2,
          CUBE_INNER_ERR_REPORT(op_name.GetString(), "data slice format input size should be 2."), return GRAPH_FAILED);
    valid_cnt ++;
  }
  if (valid_cnt == 0) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "data slice is empty.");
    return GRAPH_FAILED;
  }
  if (valid_cnt != 1) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "valid slice range num is more than 1.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

static bool GetAttrsAvgPool3DGrad(ge::Operator& op, const string& data_format,
                                  int32_t& strd, int32_t& strh, int32_t& strw) {
  ge::AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);

  std::vector<int32_t> stride_list;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stride_list)) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "get strides list failed.");
    return false;
  }
  auto s_size = stride_list.size();
  if (s_size != kAvgPool3DGradStridesDim) {
    OP_LOGE(op_name.GetString(), "strides list should be 3d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "stride_list";
    err_map["op_name"] = "AvgPool3DGrad";
    err_map["excepted_value"] = "3d";
    err_map["input_value"] = std::to_string(s_size);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  strd = stride_list[data_format.find("D")];
  strh = stride_list[data_format.find("H")];
  strw = stride_list[data_format.find("W")];

  if (strd <= 0 || strh <= 0 || strw <= 0) {
    OP_LOGE(op_name.GetString(), "strides should be positive.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = "AvgPool3DGrad";
    err_map["excepted_value"] = "positive";
    err_map["input_value"] = std::to_string(strd) + " " + std::to_string(strh) + " " + std::to_string(strw);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  return true;
}

static void InferHWAvgPool3DGrad(int64_t kernel_size,
                                 int64_t stride,
                                 int64_t input_size,
                                 const vector<int64_t>& output,
                                 vector<int64_t>& input,
                                 vector<int64_t>& multiplier,
                                 vector<int64_t>& pad_list,
                                 uint64_t pad_idx) {
  int64_t pad_out = kernel_size - pad_list[pad_idx] - 1;

  input[0] = std::min(std::max(static_cast<int64_t>(
                               std::ceil(
                                  static_cast<float>(output[0] - pad_out) /
                                  static_cast<float>(stride))),
                                0L),
                      static_cast<int64_t>(input_size - 1));
  input[1] = std::min((output[1] + kernel_size - 1 - pad_out) / static_cast<int64_t>(stride),
                      static_cast<int64_t>(input_size - 1));

  multiplier[0] = input[0];
  multiplier[1] = input[1];

  int64_t oh = static_cast<int64_t>(output[1] - output[0] + 1);
  int64_t ih = static_cast<int64_t>(input[1] - input[0] + 1);

  pad_list[pad_idx] = kernel_size - static_cast<int64_t>(
                                      input[0] * stride + pad_out - output[0]) - 1;
  pad_list[pad_idx + 1] = std::max(stride * (ih - 1) + kernel_size -
                                      oh - pad_list[pad_idx],
                                   0L);
}

IMPLEMT_VERIFIER(AvgPool3DGradD, AvgPool3DGradDVerify) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  std::vector<int64_t> orig_input_shape = GetAttrValue(op, "orig_input_shape");
  if (orig_input_shape.size() != kAvgPool3DGradOriShapeDim) {
    OP_LOGE(op_name.GetString(), "orig_input_shape length is not %zu.", kAvgPool3DGradOriShapeDim);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> ksize = GetAttrValue(op, "ksize");
  if (ksize.size() != kAvgPool3DGradKsizeDim) {
    OP_LOGE(op_name.GetString(), "ksize size is not %zu.", kAvgPool3DGradKsizeDim);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> strides = GetAttrValue(op, "strides");
  if (strides.size() != kAvgPool3DGradStridesDim) {
    OP_LOGE(op_name.GetString(), "strides size is not %zu.", kAvgPool3DGradStridesDim);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> pads;
  if (op.GetAttr("pads", pads) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "Get pads failed!");
    return GRAPH_FAILED;
  }
  if (pads.size() != kAvgPool3DGradPadsDim) {
    OP_LOGE(op_name.GetString(), "Attr pads size is not %zu.", kAvgPool3DGradPadsDim);
    return GRAPH_FAILED;
  }

  std::string data_format;
  if (op.GetAttr("data_format", data_format) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "get attr data_format failed.");
    return GRAPH_FAILED;
  }
  if (data_format != "NDHWC" && data_format != "NCDHW") {
    OP_LOGE(op_name.GetString(), "Attr data_format(%s) only support NDHWC.", data_format.c_str());
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(AvgPool3DGradD, AvgPool3DGradDInferShape) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  std::vector<int64_t> orig_input_shape = GetAttrValue(op, "orig_input_shape");
  if (orig_input_shape.size() != kAvgPool3DGradOriShapeDim) {
    OP_LOGE(op_name.GetString(), "Get orig_input_shape failed!");
    return GRAPH_FAILED;
  }
  TensorDesc grads_desc = op.GetInputDesc("grads");

  std::vector<int64_t> ksize = GetAttrValue(op, "ksize");
  if (!CheckListEmpty(op_name.GetString(), ksize, "ksize")) {
    return GRAPH_FAILED;
  }

  std::vector<int64_t> strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op_name.GetString(), strides, "strides")) {
    return GRAPH_FAILED;
  }

  std::vector<int64_t> pads;
  if (op.GetAttr("pads", pads) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "Get pads failed!");
    return GRAPH_FAILED;
  }

  std::string data_format;
  CHECK(op.GetAttr("data_format", data_format) != GRAPH_SUCCESS,
        OP_LOGE(op_name.GetString(), "Get data_format failed!"), return GRAPH_FAILED);
  CHECK(data_format != "NDHWC" && data_format != "NCDHW",
        OP_LOGE(op_name.GetString(), "Attr data_format(%s) only support NDHWC or NCDHW.", data_format.c_str()),
        return GRAPH_FAILED);

  TensorDesc mean_desc = op.GetInputDesc("multiplier");
  if (data_format == "NDHWC") {
    mean_desc.SetOriginFormat(FORMAT_NDHWC);
    mean_desc.SetFormat(FORMAT_NDHWC);
  } else if (data_format == "NCDHW") {
    mean_desc.SetOriginFormat(FORMAT_NCDHW);
    mean_desc.SetFormat(FORMAT_NCDHW);
  } else {
    OP_LOGE(op_name.GetString(), "Attr data_format(%s) only support NDHWC or NCDHW", data_format.c_str());
    return GRAPH_FAILED;
  }

  Shape grads_shape = grads_desc.GetShape();
  DataType grads_dtype = grads_desc.GetDataType();
  mean_desc.SetShape(grads_shape);
  mean_desc.SetOriginShape(grads_shape);
  mean_desc.SetDataType(grads_dtype);
  if (op.UpdateInputDesc("multiplier", mean_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "Fail to update input multiplier!");
    return GRAPH_FAILED;
  }

  string::size_type c_position = data_format.find("C");
  if (c_position == std::string::npos) {
    return GRAPH_FAILED;
  }
  int64_t kd = ksize.at(data_format.find("D"));
  int64_t kh = ksize.at(data_format.find("H"));
  int64_t kw = ksize.at(data_format.find("W"));
  vector<int64_t> kernel_shape = {kd, kh, kw, 1,
                                  orig_input_shape.at(c_position)};

  TensorDesc filter_desc = op.GetInputDesc("filter");
  filter_desc.SetShape(Shape(kernel_shape));
  filter_desc.SetOriginShape(Shape(kernel_shape));
  filter_desc.SetOriginFormat(FORMAT_DHWCN);
  filter_desc.SetFormat(FORMAT_DHWCN);
  filter_desc.SetDataType(grads_dtype);
  if (op.UpdateInputDesc("filter", filter_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "Fail to update input:filter desc!");
    return GRAPH_FAILED;
  }

  TensorDesc output_desc = op.GetOutputDesc("output");
  output_desc.SetShape(Shape(orig_input_shape));
  output_desc.SetDataType(grads_dtype);
  if (op.UpdateOutputDesc("output", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "Fail to update output:output desc!");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_INFER_DATA_SLICE(AvgPool3DGradD, AvgPool3DGradDInferDataSlice) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGD(op_name.GetString(), "Enter AvgPool3DGradDInferDataSlice.");

  int32_t strd = 0;
  int32_t strh = 0;
  int32_t strw = 0;

  string data_format;
  CHECK(op.GetAttr("data_format", data_format) != GRAPH_SUCCESS,
        OP_LOGE(op_name.GetString(), "get attr data_format failed."),
        return GRAPH_FAILED);
  CHECK(data_format != "NDHWC" && data_format != "NCDHW",
        OP_LOGE(op_name.GetString(),
                "Attr data_format(%s) only support NDHWC or NCDHW.",
                data_format.c_str()),
        return GRAPH_FAILED);

  if (!GetAttrsAvgPool3DGrad(op, data_format, strd, strh, strw)) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "get attrs failed.");
    return GRAPH_FAILED;
  }

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);
  GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("output");
  GeTensorDescPtr tensor_desc_grads = op_desc->MutableInputDesc("grads");
  GeTensorDescPtr tensor_desc_multiplier = op_desc->MutableInputDesc("multiplier");
  CHECK_PTR_NULL(tensor_desc_y, "tensor y desc", return GRAPH_FAILED);
  CHECK_PTR_NULL(tensor_desc_grads, "tensor grads desc", return GRAPH_FAILED);

  vector<vector<int64_t>> y_data_slice(6, vector<int64_t>(0));
  vector<vector<int64_t>> grads_data_slice(6, vector<int64_t>(0));
  vector<vector<int64_t>> multiplier_data_slice(6, vector<int64_t>(0));

  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op_name.GetString(), "no data slice, not need infer input");
    return GRAPH_FAILED;
  }

  graphStatus ret = VerifyDataSlice(op, y_data_slice);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }

  vector<int64_t> pad_list;
  op.GetAttr("pads", pad_list);

  auto x_format = tensor_desc_grads->GetOriginFormat();
  auto x_shape = op.get_input_desc_grads().GetShape().GetDims();

  CHECK_KEY_IN_MAP(format2str, x_format, "x_format", return GRAPH_FAILED);
  std::string x_format_str = format2str[x_format];

  int32_t d_input_position = x_format_str.find("D");
  int32_t h_input_position = x_format_str.find("H");
  int32_t w_input_position = x_format_str.find("W");
  int32_t id = x_shape[d_input_position];
  int32_t ih = x_shape[h_input_position];
  int32_t iw = x_shape[w_input_position];

  if (CheckGlobal(id, ih, iw)) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "global mode not support cut");
    return NOT_SUPPORT_SLICE;
  }

  std::vector<int64_t> ksize = GetAttrValue(op, "ksize");
  int64_t kd = ksize.at(data_format.find("D"));
  int64_t kh = ksize.at(data_format.find("H"));
  int64_t kw = ksize.at(data_format.find("W"));

  bool needUpdateX = false;
  // cut N
  if (y_data_slice[0].size() != 0) {
    grads_data_slice[0] = y_data_slice[0];
    multiplier_data_slice[0] = y_data_slice[0];
    needUpdateX = true;
  }

  // cut D
  if (y_data_slice[1].size() != 0 && pad_list.size() > 2) {
    grads_data_slice[1].clear();
    grads_data_slice[1].resize(2);
    multiplier_data_slice[1].clear();
    multiplier_data_slice[1].resize(2);
    InferHWAvgPool3DGrad(kd, strd, id, y_data_slice[1], grads_data_slice[1],
                         multiplier_data_slice[1], pad_list, 0);
    needUpdateX = true;
  }

  // cut H
  if (y_data_slice[3].size() != 0 && pad_list.size() > 4) {
    grads_data_slice[3].clear();
    grads_data_slice[3].resize(2);
    multiplier_data_slice[3].clear();
    multiplier_data_slice[3].resize(2);
    InferHWAvgPool3DGrad(kh, strh, ih, y_data_slice[3], grads_data_slice[3],
                         multiplier_data_slice[3], pad_list, 2);
    needUpdateX = true;
  }

  // cut W
  if (y_data_slice[4].size() != 0 && pad_list.size() == kAvgPool3DGradPadsDim) {
    grads_data_slice[4].clear();
    grads_data_slice[4].resize(2);
    multiplier_data_slice[4].clear();
    multiplier_data_slice[4].resize(2);
    InferHWAvgPool3DGrad(kw, strw, iw, y_data_slice[4], grads_data_slice[4],
                         multiplier_data_slice[4], pad_list, 4);
    needUpdateX = true;
  }

  // check update flag
  if (!needUpdateX) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "there's no update in desc.");
    return GRAPH_FAILED;
  }

  // update data slice attr
  if (needUpdateX) {
    if(!AttrUtils::SetListListInt(tensor_desc_grads, ge::ATTR_NAME_DATA_SLICE, grads_data_slice)) {
      CUBE_INNER_ERR_REPORT(op_name.GetString(), "set grads data slice attr failed.");
      return GRAPH_FAILED;
    }
    CHECK_PTR_NULL(tensor_desc_multiplier, "tensor multipiler desc", return GRAPH_FAILED);
    if(!AttrUtils::SetListListInt(tensor_desc_multiplier, ge::ATTR_NAME_DATA_SLICE, multiplier_data_slice)) {
      CUBE_INNER_ERR_REPORT(op_name.GetString(), "set multiplier data slice attr failed.");
      return GRAPH_FAILED;
    }
  }

  // update pads attr info
  op.SetAttr("pads", pad_list);
  return GRAPH_SUCCESS;
}

INFER_DATA_SLICE_FUNC_REG(AvgPool3DGradD, AvgPool3DGradDInferDataSlice);
VERIFY_FUNC_REG(AvgPool3DGradD, AvgPool3DGradDVerify);
INFER_FUNC_REG(AvgPool3DGradD, AvgPool3DGradDInferShape);

// ----------------AvgPool3DGrad-------------------
IMPLEMT_VERIFIER(AvgPool3DGrad, AvgPool3DGradVerify) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  std::vector<int64_t> ksize = GetAttrValue(op, "ksize");
  if (ksize.size() != kAvgPool3DGradKsizeDim) {
    OP_LOGE(op_name.GetString(), "Attr:ksize has an incorreted length");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> strides = GetAttrValue(op, "strides");
  if (strides.size() != kAvgPool3DGradStridesDim) {
    OP_LOGE(op_name.GetString(), "Attr:strides has an incorreted length");
    return GRAPH_FAILED;
  }

  std::string data_format;
  CHECK(op.GetAttr("data_format", data_format) != GRAPH_SUCCESS,
        OP_LOGE(op_name.GetString(), "get attr data_format failed."), return GRAPH_FAILED);

  if (data_format != "NDHWC" && data_format != "NCDHW") {
    OP_LOGE(op_name.GetString(),
            "Attr data_format(%s) only support NDHWC or NCDHW.",
            data_format.c_str());
    return GRAPH_FAILED;
  }

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);
  auto orig_shape_desc = op_desc->MutableInputDesc("orig_input_shape");
  CHECK_PTR_NULL(orig_shape_desc, "orig_shape desc", return GRAPH_FAILED);
  vector<int64_t> orig_input_dim = orig_shape_desc->MutableShape().GetDims();
  if (orig_input_dim.size() != kAvgPool3DGradOriShapeDim &&
        orig_input_dim.size() != 1) {
    OP_LOGE(op_name.GetString(), "orig_input_shape's dim expect: %u or %lu, but real: %lu.",
            1, kAvgPool3DGradOriShapeDim, orig_input_dim.size());
    return GRAPH_FAILED;
  }

  auto grads_desc = op_desc->MutableInputDesc("grads");
  CHECK_PTR_NULL(grads_desc, "grads desc", return GRAPH_FAILED);
  vector<int64_t> grads_shape = grads_desc->MutableShape().GetDims();
  if (grads_shape.size() != kAvgPool3DGradOriShapeDim &&
        !IsUnknownRankShape(grads_shape)) {
    OP_LOGE(op_name.GetString(), "grads_shape's dim expect: %lu, but real: %lu.",
            kAvgPool3DGradOriShapeDim, grads_shape.size());
    return GRAPH_FAILED;
  }

  string padding;
  std::vector<int64_t> pads;
  if (op.GetAttr("pads", pads) == GRAPH_SUCCESS) {
    if (pads.size() != kAvgPool3DGradPadsDim) {
      OP_LOGE(op_name.GetString(), "Attr:pads size expect: %lu, but real: %lu.",
              kAvgPool3DGradPadsDim, pads.size());
      return GRAPH_FAILED;
    }
  } else if (op.GetAttr("padding", padding) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "op attrs must contains padding or pads.");
    return GRAPH_FAILED;
  } else if (padding != "SAME" && padding != "VALID") {
    OP_LOGE(op_name.GetString(),
            "Attr padding(%s) only SAME NDHWC or VALID.",
            padding.c_str());
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

void AvgPool3DGradCalcPads(const vector<int64_t> &fmap_shape,
                           const string &fmap_format,
                           const vector<int64_t> &ksize,
                           const vector<int64_t> &strides,
                           vector<int64_t> &pads,
                           const string &data_format) {
  int64_t pads_d = 0;
  int64_t pads_h = 0;
  int64_t pads_w = 0;
  int64_t stride_d = strides[data_format.find("D")];
  int64_t stride_h = strides[data_format.find("H")];
  int64_t stride_w = strides[data_format.find("W")];

  pads_d = std::max((fmap_shape[fmap_format.find("D")] + stride_d - 1) /
                     stride_d * stride_d + ksize[data_format.find("D")] -
                     stride_d - fmap_shape[fmap_format.find("D")], 0L);
  pads_h = std::max((fmap_shape[fmap_format.find("H")] + stride_h - 1) /
                     stride_h * stride_h + ksize[data_format.find("H")] -
                     stride_h - fmap_shape[fmap_format.find("H")], 0L);
  pads_w = std::max((fmap_shape[fmap_format.find("W")] + stride_w - 1) /
                     stride_w * stride_w +ksize[data_format.find("W")] -
                     stride_w - fmap_shape[fmap_format.find("W")], 0L);

  pads[0] = pads_d / 2;
  pads[1] = pads_d - pads[0];
  pads[2] = pads_h / 2;
  pads[3] = pads_h - pads[2];
  pads[4] = pads_w / 2;
  pads[5] = pads_w - pads[4];
}

void AvgPool3DGradCalDedxRange(int64_t ksize, int64_t stride, const string &padding,
                               std::pair<int64_t, int64_t> &grads_range,
                               std::pair<int64_t, int64_t> &fmap_range) {
  grads_range.first = grads_range.first == -1 ? kDynamicRangeUpperBound : grads_range.first;
  grads_range.second = grads_range.second == -1 ? kDynamicRangeUpperBound : grads_range.second;
  if (padding == "SAME") {
    fmap_range.first = std::max(stride * (grads_range.first - 1) + 1, kDynamicRangeLowerBound);
    fmap_range.second = std::min(stride * grads_range.second, kDynamicRangeUpperBound);
  } else {
    fmap_range.first = std::max(stride * (grads_range.first - 1) + ksize, kDynamicRangeLowerBound);
    fmap_range.second = std::min(stride * (grads_range.second - 1) + ksize + stride - 1,
                                 kDynamicRangeUpperBound);
  }
}

bool SetAvgPool3DGradOutputRange(ge::OpDescPtr &op_desc,
                                 const vector<int64_t> &ksize, const vector<int64_t> &strides,
                                 const string &padding, const string &output_format,
                                 vector<int64_t> &fmap_shape) {
  auto orig_input_desc = op_desc->MutableInputDesc("orig_input_shape");
  auto grads_desc = op_desc->MutableInputDesc("grads");
  auto output_desc = op_desc->MutableOutputDesc("output");
  CHECK_PTR_NULL(orig_input_desc, "input desc", return false);
  CHECK_PTR_NULL(grads_desc, "grad desc", return false);
  CHECK_PTR_NULL(output_desc, "output desc", return false);

  std::vector<std::pair<int64_t, int64_t>> grads_range;
  std::vector<int64_t> grads_shape = grads_desc->MutableShape().GetDims();
  if (!IsUnknown(grads_shape)) {
    for (auto i : grads_shape) {
      grads_range.push_back(std::make_pair(i, i));
    }
  } else {
    grads_desc->GetShapeRange(grads_range);
  }
  if (grads_range.size() != kAvgPool3DGradOriShapeDim) {
    OP_LOGE("AvgPool3dGrad",
            "grads range size is invalid, expect :%lu, real: %lu.",
            kAvgPool3DGradOriShapeDim,
            grads_range.size());
    return false;
  }
  CHECK_KEY_IN_MAP(format2str, grads_desc->GetFormat(), "grad_format", return false);
  string grads_format = format2str[grads_desc->GetFormat()];

  std::vector<std::pair<int64_t, int64_t>> fmap_range(kAvgPool3DGradOriShapeDim);

  std::vector<std::pair<int64_t, int64_t>> pre_op_range;
  orig_input_desc->GetValueRange(pre_op_range);

  std::vector<int64_t> output_shape = output_desc->MutableShape().GetDims();
  std::vector<std::pair<int64_t, int64_t>> orig_shape_range;
  orig_input_desc->GetShapeRange(orig_shape_range);

  if (IsUnknownRankShape(grads_shape)) {
    // not supported -2 yet
    fmap_shape.assign(kAvgPool3DGradOriShapeDim, -1);
    OP_LOGE("AvgPool3dGrad", "AvgPool3dGrad has not spport grads shape -2.");
    return false;
  } else if ((pre_op_range.size() == kAvgPool3DGradOriShapeDim) &&
             (grads_range.size() == kAvgPool3DGradOriShapeDim)) {
    // try get prevent op output shape range
    fmap_range = pre_op_range;
  } else if (orig_shape_range.size() == kAvgPool3DGradOriShapeDim) {
    // try get orig_input_shape range and ifx dedy range
    fmap_range = orig_shape_range;
  } else {
    // use dedy's shape and range calculate dedx's shape and range
    fmap_range[output_format.find("N")] = grads_range[grads_format.find("N")];
    fmap_range[output_format.find("C")] = grads_range[grads_format.find("C")];
    AvgPool3DGradCalDedxRange(ksize[output_format.find("D")], strides[output_format.find("D")],
                              padding, grads_range[grads_format.find("D")],
                              fmap_range[output_format.find("D")]);
    AvgPool3DGradCalDedxRange(ksize[output_format.find("H")], strides[output_format.find("H")],
                              padding, grads_range[grads_format.find("H")],
                              fmap_range[output_format.find("H")]);
    AvgPool3DGradCalDedxRange(ksize[output_format.find("W")], strides[output_format.find("W")],
                              padding, grads_range[grads_format.find("W")],
                              fmap_range[output_format.find("W")]);
  }
  // set output range
  output_desc->SetShapeRange(fmap_range);
  fmap_shape.assign(kAvgPool3DGradOriShapeDim, -1);
  bool is_all_const = true;
  for (size_t i = 0; i < fmap_range.size(); ++i) {
    fmap_shape[i] = fmap_range[i].first == fmap_range[i].second ? fmap_range[i].first : -1;
    is_all_const = is_all_const && fmap_shape[i] != -1;
  }
  // fe judge dynamic mode by detect whether shape contains -1,
  // if there is no -1 in shape, it will run static impl ops.
  if (is_all_const) {
    fmap_shape[output_format.find("N")] = -1;
  }
  return true;
}

IMPLEMT_INFERFUNC(AvgPool3DGrad, AvgPool3DGradInferShape) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto grads_desc = op_info->MutableInputDesc("grads");
  auto orig_shape_desc = op_info->MutableInputDesc("orig_input_shape");
  CHECK_PTR_NULL(op_info, "op desc", return GRAPH_FAILED);
  CHECK_PTR_NULL(grads_desc, "grads desc", return GRAPH_FAILED);
  CHECK_PTR_NULL(orig_shape_desc, "orig_shape desc", return GRAPH_FAILED);
  auto grads_dtype = grads_desc->GetDataType();

  std::vector<std::string> const_inputs = {"orig_input_shape"};
  op_info->SetOpInferDepends(const_inputs);
  bool is_dynamic = false;

  Tensor orig_shape_tensor;
  vector<int64_t> fmap_shape;
  vector<int64_t> orig_input_dim = orig_shape_desc->MutableShape().GetDims();
  if (op.GetInputConstData("orig_input_shape", orig_shape_tensor) == GRAPH_SUCCESS) {
    // orig_input_shape is const
    DataType dtype = orig_shape_desc->GetDataType();
    GetConstValue(op, orig_shape_tensor, dtype, fmap_shape);
  } else if (orig_input_dim.size() == kAvgPool3DGradOriShapeDim) {
    fmap_shape = orig_input_dim;
    is_dynamic = IsUnKnownShape(fmap_shape);
  } else {
    is_dynamic = true;
    OP_LOGI(op_name.GetString(), "infer shape in dynamic mode.");
    fmap_shape.assign(kAvgPool3DGradOriShapeDim, -1);
  }

  auto output_desc = op_info->MutableOutputDesc("output");
  CHECK_PTR_NULL(output_desc, "output desc", return GRAPH_FAILED);
  auto output_shape = output_desc->MutableShape();
  output_desc->SetDataType(grads_dtype);
  // get ksize attr
  std::vector<int64_t> ksize = GetAttrValue(op, "ksize");
  if (ksize.size() != kAvgPool3DGradKsizeDim) {
    OP_LOGE(op_name.GetString(), "Attr:ksize has an incorrected length.");
    return GRAPH_FAILED;
  }
  // get strides attr
  std::vector<int64_t> strides = GetAttrValue(op, "strides");
  if (strides.size() != kAvgPool3DGradStridesDim) {
    OP_LOGE(op_name.GetString(), "Attr:strides has an incorrected length.");
    return GRAPH_FAILED;
  }
  // get data format attr
  std::string data_format;
  if (op.GetAttr("data_format", data_format) != GRAPH_SUCCESS ||
     (data_format != "NDHWC" && data_format != "NCDHW")) {
    OP_LOGE(op_name.GetString(),
           "Op attr:data_format only support NDHWC and NCDHW.");
    return GRAPH_FAILED;
  }
  // get padding attr
  std::string padding;
  bool has_padding_attr = true;
  vector<int64_t> pads(kAvgPool3DGradPadsDim, 0);
  if (op.GetAttr("padding", padding) == GRAPH_SUCCESS) {
    if (padding != "SAME" && padding != "VALID") {
      OP_LOGE(op_name.GetString(), "Padding pattern is incorrected, only support SAME and VALID.");
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGD(op_name.GetString(), "Get padding attr failed.");
    // pads attr
    CHECK(op.GetAttr("pads", pads) != GRAPH_SUCCESS, OP_LOGE(op_name.GetString(), "Get attr:pads failed."),
          return GRAPH_FAILED);
    has_padding_attr = false;
    padding = IsAllVal(pads, 0) ? "VALID" : "SAME";
  }
  Format orig_shape_format = orig_shape_desc->GetFormat();
  string orig_shape_format_str = format2str[orig_shape_format];
  // set range
  if (is_dynamic && !SetAvgPool3DGradOutputRange(op_info, ksize, strides, padding,
                                                 data_format, fmap_shape)) {
    OP_LOGE(op_name.GetString(), "Calculate output range.");
    return GRAPH_FAILED;
  }

  if (fmap_shape.size() == kAvgPool3DGradOriShapeDim) {
    output_desc->SetShape(GeShape(fmap_shape));
  } else {
    OP_LOGE(op_name.GetString(), "output shape size is incollected. real:%lu expect:%lu.",
            kAvgPool3DGradOriShapeDim, fmap_shape.size());
    return GRAPH_FAILED;
  }

  if ((is_dynamic || has_padding_attr) && padding == "SAME") {
    if (!is_dynamic || (fmap_shape[orig_shape_format_str.find("D")] != -1 &&
                        fmap_shape[orig_shape_format_str.find("H")] != -1 &&
                        fmap_shape[orig_shape_format_str.find("W")] != -1)){
      OP_LOGD(op_name.GetString(), "Fix padding attr.");
      AvgPool3DGradCalcPads(fmap_shape, orig_shape_format_str, ksize, strides, pads, data_format);
    } else {
      pads.assign(kAvgPool3DGradPadsDim, -1);
    }
  }

  op.SetAttr("pads", pads);
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
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc(0);
  auto input_shape = input_desc->MutableShape();
  auto input_format = input_desc->GetFormat();
  auto input_dtype = input_desc->GetDataType();
  auto output_desc = op_info->MutableOutputDesc(0);
  output_desc->SetDataType(input_dtype);
  // get input ksize
  std::vector<int32_t> ksize;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksize)) {
    std::string err_msg = GetInputInvalidErrMsg("ksize");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  // get input strides
  std::vector<int32_t> strides;
  if (GRAPH_SUCCESS != op.GetAttr("strides", strides)) {
    std::string err_msg = GetInputInvalidErrMsg("strides");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  // get input data_format
  std::string data_format;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", data_format)) {
    std::string err_msg = GetInputInvalidErrMsg("data_format");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  // get input padding
  std::string padding;
  if (GRAPH_SUCCESS != op.GetAttr("padding", padding)) {
    std::string err_msg = GetInputInvalidErrMsg("padding");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
  if ((strides[strides_h_dim] == 0) || (strides[strides_w_dim] == 0)) {
    std::string err_msg = GetInputInvalidErrMsg("padding");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (padding != "VALID") {
    ksize[strides_h_dim] = 1;
    ksize[strides_w_dim] = 1;
  }

  // set ksize for global max pool
  bool unknowRank = IsUnknownRankShape(input_dims);
  if (!unknowRank) {
    if (ksize[input_h_dim] == -1 && ksize[input_w_dim] == -1) {
      ksize[input_h_dim] = input_dims[input_h_dim];
      ksize[input_w_dim] = input_dims[input_w_dim];
      op.SetAttr("ksize", ksize);
    }
  }

  for (size_t i = 0; i < input_dims.size(); i++) {
    int64_t dim_size = input_dims[i];
    auto dim_range = input_range[i];
    if (static_cast<int64_t>(i) == input_h_dim) {
      UpdateDimAndRange(ksize[strides_h_dim], strides[strides_h_dim], dim_size, dim_range);
    } else if (static_cast<int64_t>(i) == input_w_dim) {
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
    std::string err_msg = GetInputInvalidErrMsg("ksize");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (ksize.size() != 4) {
    std::string err_msg = GetAttrSizeErrMsg("ksize", ConcatString(ksize.size()), ConcatString(4));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  // check strides
  std::vector<int32_t> strides;
  if (GRAPH_SUCCESS != op.GetAttr("strides", strides)) {
    std::string err_msg = GetInputInvalidErrMsg("strides");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (strides.size() != 4) {
    std::string err_msg = GetAttrSizeErrMsg("strides", ConcatString(strides.size()), ConcatString(4));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  // check data_format
  std::string data_format;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", data_format)) {
    std::string err_msg = GetInputInvalidErrMsg("data_format");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (data_format != "NHWC" && data_format != "NCHW" && data_format != "NC1HWC0") {
    string expected_format_list = ConcatString("NHWC, NCHW, NC1HWC0");
    std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (data_format == "NHWC") {
    if ((ksize[0] != 1) || (ksize[3] != 1) || (strides[0] != 1) || (strides[3] != 1)) {
      std::string err_msg = OtherErrMsg("Pooling across width/height and other ksize dimension should be one");
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }
  if ((data_format == "NCHW") || (data_format == "NC1HWC0")) {
    if ((ksize[0] != 1) || (ksize[1] != 1) || (strides[0] != 1) || (strides[1] != 1)) {
      string err_msg = ConcatString("Pooling across width/height and other ksize dimension should be one, ksize[0]:",ksize[0], ", ksize[1]:",ksize[0],", strides[0]:",strides[1],", strides[1]:",strides[1]);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }
  // check padding
  std::string padding;
  if (GRAPH_SUCCESS != op.GetAttr("padding", padding)) {
    std::string err_msg = GetInputInvalidErrMsg("padding");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (padding != "SAME" && padding != "VALID") {
    string expected_format_list = ConcatString("SAME or VALID");
    std::string err_msg = GetAttrValueErrMsg("padding", padding, expected_format_list);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

static void InferHWMaxPool(int64_t kernel, int64_t stride, vector<int64_t>& output, vector<int64_t>& input,
                           int64_t& ori_input) {
  int64_t first_start = 0;
  int64_t second_start = 0;
  int64_t second_end = 0;
  int64_t start = 0;
  int64_t end = 0;
  first_start = output[0] * stride;
  second_start = output[1] * stride;
  second_end = std::min(second_start + kernel, ori_input);
  start = std::max(first_start, int64_t(0));
  end = second_end - 1;
  input = {start, end};
}

IMPLEMT_INFER_DATA_SLICE(MaxPool, MaxPoolInferDataSlice) {
  OP_LOGD(op.GetName().c_str(), "Enter MaxPoolInferDataSlice.");
  auto inputTensorDesc = op.GetInputDesc("x");
  auto shape = inputTensorDesc.GetShape();
  std::vector<int64_t> dims_input = shape.GetDims();
  auto inputFormat = inputTensorDesc.GetFormat();

  int64_t inputH = 0;
  int64_t inputW = 0;
  if (inputFormat == FORMAT_NC1HWC0) {
    inputH = dims_input[2];
    inputW = dims_input[3];
  } else {
    OP_LOGE(op.GetName().c_str(), "Invalid inputFormat.");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> ksizeList;
  std::vector<int64_t> stridesList;
  std::string dataFormat;
  std::string paddingMode;
  op.GetAttr("ksize", ksizeList);
  op.GetAttr("strides", stridesList);
  op.GetAttr("data_format", dataFormat);
  op.GetAttr("padding", paddingMode);

  int64_t windowH = 0;
  int64_t windowW = 0;
  int64_t strideH = 0;

  if (dataFormat == "NHWC") {
    windowH = ksizeList[1];
    windowW = ksizeList[2];
    strideH = stridesList[1];
  } else if (dataFormat == "NCHW") {
    windowH = ksizeList[2];
    windowW = ksizeList[3];
    strideH = stridesList[2];
  } else {
    OP_LOGE(op.GetName().c_str(), "Invalid dataFormat.");
    return GRAPH_FAILED;
  }

  if (windowH == inputH && windowW == inputW) {
    OP_LOGD(op.GetName().c_str(), "Global pool can't calculate over lap.");
    return NO_OVERLAP_DIM;
  }
  if (paddingMode == "SAME") {
    OP_LOGD(op.GetName().c_str(), "Padding mode is same, can't calculate over lap.");
    return NO_OVERLAP_DIM;
  }

  vector<vector<int64_t>> y_data_slice = {{}, {}, {}, {}, {}};
  vector<vector<int64_t>> x_data_slice = {{}, {}, {}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_out = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_in = op_desc->MutableInputDesc("x");
  if (!ge::AttrUtils::GetListListInt(tensor_desc_out, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGE(op.GetName().c_str(), "no data slice, use default as.");
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
        OP_LOGE(op.GetName().c_str(), "Set x data slice failed.");
        return GRAPH_FAILED;
      }
      return GRAPH_SUCCESS;
    }
  }

  return NO_OVERLAP_DIM;
}

INFER_FUNC_REG(MaxPool, MaxPoolInferShape);
VERIFY_FUNC_REG(MaxPool, MaxPoolVerify);
INFER_DATA_SLICE_FUNC_REG(MaxPool, MaxPoolInferDataSlice);
// ----------------MaxPool-------------------

// ----------------MaxPool3D-------------------
IMPLEMT_INFERFUNC(MaxPool3D, MaxPool3DInferShape) {
  DYNAMIC_SHAPE_NOT_SUPPORTED(op);
  const size_t DIM_SIZE1 = 1;
  const size_t DIM_SIZE3 = 3;
  const size_t DIM_SIZE5 = 5;
  auto inputTensorDesc = op.GetInputDesc("x");
  auto shape = inputTensorDesc.GetShape();
  Format input_format = inputTensorDesc.GetFormat();

  size_t input_dims = shape.GetDims().size();
  if (input_dims != DIM_SIZE5) {
    string excepted_value = ConcatString(DIM_SIZE5);
    OP_LOGE(op.GetName().c_str(), "length of x should be 5!");
    return GRAPH_FAILED;
  }

  // get input ksize
  std::vector<int32_t> ksizeList;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksizeList)) {
    std::string err_msg = GetInputInvalidErrMsg("ksize");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if ((ksizeList.size() != DIM_SIZE1) && (ksizeList.size() != DIM_SIZE3) && (ksizeList.size() != DIM_SIZE5)) {
    string excepted_size = ConcatString(DIM_SIZE1, " ", DIM_SIZE3, " or ", DIM_SIZE5);
    std::string err_msg = GetAttrSizeErrMsg("ksizeList", ConcatString(ksizeList.size()), excepted_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // get input strides
  std::vector<int32_t> stridesList;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stridesList)) {
    std::string err_msg = GetInputInvalidErrMsg("ksize");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if ((stridesList.size() != DIM_SIZE1) && (stridesList.size() != DIM_SIZE3) && (stridesList.size() != DIM_SIZE5)) {
    string excepted_size = ConcatString(DIM_SIZE1, " or ", DIM_SIZE3, " or ", DIM_SIZE5);
    std::string err_msg = GetAttrSizeErrMsg("stridesList", ConcatString(stridesList.size()), excepted_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  std::string dataFormat;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
    std::string err_msg = GetInputInvalidErrMsg("data_format");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (dataFormat != "NDHWC" && dataFormat != "NCDHW") {
    string expected_format_list = ConcatString("NDHWC, NCDHW");
    std::string err_msg = GetInputFormatNotSupportErrMsg("dataFormat", expected_format_list, dataFormat);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> ksizeTempList;
  if (dataFormat == "NDHWC") {
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
  }else {
    if (ksizeList.size() == DIM_SIZE1) {
      ksizeTempList.push_back(1);
      ksizeTempList.push_back(1);
      ksizeTempList.push_back(ksizeList[0]);
      ksizeTempList.push_back(ksizeList[0]);
      ksizeTempList.push_back(ksizeList[0]);
    }

    if (ksizeList.size() == DIM_SIZE3) {
      ksizeTempList.push_back(1);
      ksizeTempList.push_back(1);
      ksizeTempList.push_back(ksizeList[0]);
      ksizeTempList.push_back(ksizeList[1]);
      ksizeTempList.push_back(ksizeList[2]);
    }

    if (ksizeList.size() == DIM_SIZE5) {
      ksizeTempList.push_back(ksizeList[0]);
      ksizeTempList.push_back(ksizeList[1]);
      ksizeTempList.push_back(ksizeList[2]);
      ksizeTempList.push_back(ksizeList[3]);
      ksizeTempList.push_back(ksizeList[4]);
    }
  }

  std::vector<int64_t> stridesTempList;
  if (dataFormat == "NDHWC") {
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
  }else {
    if (stridesList.size() == DIM_SIZE1) {
      stridesTempList.push_back(1);
      stridesTempList.push_back(1);
      stridesTempList.push_back(stridesList[0]);
      stridesTempList.push_back(stridesList[0]);
      stridesTempList.push_back(stridesList[0]);
    }

    if (stridesList.size() == DIM_SIZE3) {
      stridesTempList.push_back(1);
      stridesTempList.push_back(1);
      stridesTempList.push_back(stridesList[0]);
      stridesTempList.push_back(stridesList[1]);
      stridesTempList.push_back(stridesList[2]);
    }
    if (stridesList.size() == DIM_SIZE5) {
      stridesTempList.push_back(stridesList[0]);
      stridesTempList.push_back(stridesList[1]);
      stridesTempList.push_back(stridesList[2]);
      stridesTempList.push_back(stridesList[3]);
      stridesTempList.push_back(stridesList[4]);
    }
  }

  std::vector<int32_t> padsList;
  if (GRAPH_SUCCESS != op.GetAttr("pads", padsList)) {
    std::string err_msg = GetInputInvalidErrMsg("pads");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (padsList.size() != 6) {
    string excepted_value = ConcatString(6);
    std::string err_msg = GetAttrSizeErrMsg("padsList", ConcatString(padsList.size()), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // get input paddingMode
  std::string paddingMode;
  if (GRAPH_SUCCESS != op.GetAttr("padding", paddingMode)) {
    std::string err_msg = GetInputInvalidErrMsg("padding");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (paddingMode != "SAME" && paddingMode != "VALID" && paddingMode != "CALCULATED") {
    string excepted_value = ConcatString("SAME VALID or CALCULATED");
    std::string err_msg = GetAttrValueErrMsg("paddingMode", paddingMode, excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int ceilMode = 0;
  if (GRAPH_SUCCESS != op.GetAttr("ceil_mode", ceilMode)) {
    std::string err_msg = GetInputInvalidErrMsg("ceil_mode");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (ceilMode != 0 && ceilMode != 1) {
    string excepted_value = ConcatString("0 or 1");
    std::string err_msg = GetAttrValueErrMsg("ceilMode", ConcatString(ceilMode), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dims_input = shape.GetDims();
  std::vector<int64_t> dimVector;
  int64_t dims = 1;
  int pad = 0;
  // set output shape
  if (paddingMode == "CALCULATED") {
    if (ceilMode == 0) {
      if (input_format == FORMAT_NCDHW) {
        for (size_t i = 0; i < dims_input.size(); i++) {
          if (i == 0 || i == 1) {
              dims = dims_input[i];
          }else {
            if(i == 2){
              pad = padsList[0];
            }
            if(i == 3){
              pad = padsList[2];
            }
            if(i == 4){
              pad = padsList[4];
            }
            dims = ((dims_input[i] + 2 * pad - (ksizeTempList[i] - 1) - 1) / stridesTempList[i]) + 1;
          }
          dimVector.push_back(dims);
        }
      }else if (input_format == FORMAT_NDHWC) {
        for (size_t i = 0; i < dims_input.size(); i++) {
          if (i == 0 || i == 4) {
              dims = dims_input[i];
          }else {
            if(i == 1){
              pad = padsList[0];
            }
            if(i == 2){
              pad = padsList[2];
            }
            if(i == 3){
              pad = padsList[4];
            }
            dims = ((dims_input[i] + 2 * pad - (ksizeTempList[i] - 1) - 1) / stridesTempList[i]) + 1;
          }
          dimVector.push_back(dims);
        }
      }
    } else {
      if (input_format == FORMAT_NCDHW) {
        for (size_t i = 0; i < dims_input.size(); i++) {
          if (i == 0 || i == 1) {
              dims = dims_input[i];
          }else {
            if (i == 2) {
              pad = padsList[0];
            }
            if (i == 3) {
              pad = padsList[2];
            }
            if (i == 4) {
              pad = padsList[4];
            }
            dims = ((dims_input[i] + 2 * pad - (ksizeTempList[i] - 1) - 1 + stridesTempList[i] - 1) /\
                     stridesTempList[i]) + 1;
          }
          dimVector.push_back(dims);
        }
      }else if (input_format == FORMAT_NDHWC) {
        for (size_t i = 0; i < dims_input.size(); i++) {
          if (i == 0 || i == 4) {
              dims = dims_input[i];
          }else {
            if (i == 1) {
              pad = padsList[0];
            }
            if (i == 2) {
              pad = padsList[2];
            }
            if (i == 3) {
              pad = padsList[4];
            }
            dims = ((dims_input[i] + 2 * pad - (ksizeTempList[i] - 1) - 1 + stridesTempList[i] - 1) /\
                     stridesTempList[i]) + 1;
          }
          dimVector.push_back(dims);
        }
      }
    }
  } else {
    if (input_format == FORMAT_NDHWC) {
      if (paddingMode == "SAME") {
        for (size_t i = 0; i < dims_input.size(); i++) {
          int64_t dims = 1;
          if (i == 0 || i == 4) {
            dims = dims_input[i];
          } else {
            dims = (dims_input[i] + stridesTempList[i] - 1) / stridesTempList[i];
          }
          dimVector.push_back(dims);
        }
      } else {
        for (size_t i = 0; i < dims_input.size(); i++) {
          int64_t dims = 1;
          if (i == 0 || i == 4) {
            dims = dims_input[i];
          } else {
            dims = (dims_input[i] - ksizeTempList[i] + 1 + (stridesTempList[i] - 1)) / stridesTempList[i];
          }
          dimVector.push_back(dims);
        }
      }
    } else if (input_format == FORMAT_NCDHW) {
      if (paddingMode == "SAME") {
        for (size_t i = 0; i < dims_input.size(); i++) {
          int64_t dims = 1;
          if (i == 0 || i == 1) {
            dims = dims_input[i];
          } else {
            dims = (dims_input[i] + stridesTempList[i] - 1) / stridesTempList[i];
          }
          dimVector.push_back(dims);
        }
      } else {
        for (size_t i = 0; i < dims_input.size(); i++) {
          int64_t dims = 1;
          if (i == 0 || i == 1) {
            dims = dims_input[i];
          } else {
            dims = (dims_input[i] - ksizeTempList[i] + 1 + (stridesTempList[i] - 1)) / stridesTempList[i];
          }
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

IMPLEMT_VERIFIER(MaxPool3D, MaxPool3DVerify) {
  // verify in infer func
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MaxPool3D, MaxPool3DInferShape);
VERIFY_FUNC_REG(MaxPool3D, MaxPool3DVerify);
// ----------------MaxPool3D-------------------

//-------------------MaxPool3DWithArgmax---------------------
IMPLEMT_INFERFUNC(MaxPool3DWithArgmax, MaxPool3DWithArgmaxInferShape){
  TensorDesc inputDesc = op.GetInputDesc("x");
  auto inputShape = inputDesc.GetShape().GetDims();
  DataType inputDtype = inputDesc.GetDataType();
  TensorDesc argmaxDesc = op.GetOutputDesc("argmax");
  TensorDesc outputDesc = op.GetOutputDesc("y");
  std::vector<int64_t> stridesList;
  op.GetAttr("strides", stridesList);
  std::vector<int64_t> kernelList;
  op.GetAttr("ksize", kernelList);
  int64_t dOut = (inputShape[1] - kernelList[2]) / stridesList[2] + 1;
  int64_t hOut = (inputShape[3] - kernelList[3]) / stridesList[3] + 1;
  int64_t wOut = (inputShape[4] - kernelList[4]) / stridesList[4] + 1;
  int64_t alignedBmLine;
  alignedBmLine = (wOut * hOut % 16 == 0) ? (wOut * hOut) : (((int64_t)(wOut * hOut / 16) + 1) * 16);
  std::vector<int64_t> argShapeVec;
  argShapeVec.push_back(inputShape[0]);
  argShapeVec.push_back(dOut);
  argShapeVec.push_back(inputShape[2] * kernelList[2] * kernelList[3] * kernelList[4]);
  argShapeVec.push_back((int64_t)(alignedBmLine / 16));
  argShapeVec.push_back(inputShape[5]);
  Shape argmaxShape(argShapeVec);
  argmaxDesc.SetShape(argmaxShape);
  argmaxDesc.SetDataType(DT_UINT16);
  (void)op.UpdateOutputDesc("argmax", argmaxDesc);
  std::vector<int64_t> outShapeVec {inputShape[0], dOut, inputShape[2], hOut, wOut, inputShape[5]};
  Shape outputShape(outShapeVec);
  outputDesc.SetShape(outputShape);
  outputDesc.SetDataType(inputDtype);
  (void)op.UpdateOutputDesc("y", outputDesc);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MaxPool3DWithArgmax, MaxPool3DWithArgmaxVerify) {
  // verify in infer func
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MaxPool3DWithArgmax, MaxPool3DWithArgmaxInferShape);
VERIFY_FUNC_REG(MaxPool3DWithArgmax, MaxPool3DWithArgmaxVerify);
//-------------------MaxPool3DWithArgmax---------------------

// ---------------------MaxPool3DGradGrad---------------------
static bool GetAttrsMaxPool3DGradGrad(ge::Operator& op, Format refer, int32_t& strd, int32_t& strh, int32_t& strw,
                                      int32_t& kd, int32_t& kh, int32_t& kw) {
  std::vector<int32_t> strideList;
  if (GRAPH_SUCCESS != op.GetAttr("strides", strideList)) {
    std::string err_msg = GetInputInvalidErrMsg("strides");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<int32_t> ksizeList;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksizeList)) {
    std::string err_msg = GetInputInvalidErrMsg("ksize");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = OtherErrMsg("MaxPool3DGradGrad, two input dtypes must be same");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<int64_t> ksize;
  ksize = GetAttrValue(op, "ksize");
  if (!CheckListEmpty(op.GetName(), ksize, "ksize")) {
    std::string err_msg = GetInputInvalidErrMsg("ksize");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (ksize.size() != 1 && ksize.size() != 3 && ksize.size() != 5) {
    string excepted_size = ConcatString("1 or 3 or 5");
    std::string err_msg = GetAttrSizeErrMsg("ksize", ConcatString(ksize.size()), excepted_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), strides, "strides")) {
    std::string err_msg = GetInputInvalidErrMsg("strides");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (strides.size() != 1 && strides.size() != 3 && strides.size() != 5) {
    string excepted_size = ConcatString("1 3 or 5");
    std::string err_msg = GetAttrSizeErrMsg("strides", ConcatString(strides.size()), excepted_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  std::string data_format;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
    if (data_format != "NCDHW" && data_format != "NDHWC") {
      string expected_format_list = ConcatString("NCDHW, NDHWC");
      std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = GetInputInvalidErrMsg("attr in MaxPool3DGradGrad");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if ((strd == 0) || (strh == 0) || (strw == 0)) {
    string err_msg = ConcatString("strd/strh/strw should not be zero, strd:",strd, ", strh:",strh,", strw:",strw);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  // construct pads attr
  if (false == Construct3DPadsByPadding("MaxPool3DGradGradD", op, id, ih, iw, kd, kh, kw, strd, strh, strw)) {
    std::string err_msg = GetInputInvalidErrMsg("pads in MaxPool3DGradGrad");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = GetInputInvalidErrMsg("ksize");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (ksize.size() != 4) {
    std::string err_msg = GetAttrSizeErrMsg("ksize", ConcatString(ksize.size()), ConcatString(4));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), strides, "strides")) {
    std::string err_msg = GetInputInvalidErrMsg("strides");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (strides.size() != 4) {
    std::string err_msg = GetAttrSizeErrMsg("strides", ConcatString(strides.size()), ConcatString(4));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::string padding;
  if (op.GetAttr("padding", padding) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("padding");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (padding != "SAME" && padding != "VALID") {
    string excepted_value = ConcatString("SAME or VALID");
    std::string err_msg = GetAttrValueErrMsg("padding", padding, excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_SUCCESS) {
    if (data_format != "NCHW" && data_format != "NHWC") {
      string expected_format_list = ConcatString("NCHW, NHWC");
      std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = OtherErrMsg("Two input dtypes must be same");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<int64_t> ksize;
  ksize = GetAttrValue(op, "ksize");
  if (!CheckListEmpty(op.GetName(), ksize, "ksize")) {
    std::string err_msg = GetInputInvalidErrMsg("ksize");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (ksize.size() != 4) {
    std::string err_msg = GetAttrSizeErrMsg("ksize", ConcatString(ksize.size()), ConcatString(4));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), strides, "strides")) {
    std::string err_msg = GetInputInvalidErrMsg("strides");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (strides.size() != 4) {
    std::string err_msg = GetAttrSizeErrMsg("strides", ConcatString(strides.size()), ConcatString(4));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::string padding;
  if (op.GetAttr("padding", padding) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("padding");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (padding != "SAME" && padding != "VALID") {
    string excepted_value = ConcatString("SAME or VALID");
    std::string err_msg = GetAttrValueErrMsg("padding", padding, excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_SUCCESS) {
    if (data_format != "NCHW" && data_format != "NHWC") {
      string expected_format_list = ConcatString("NCHW, NHWC");
      std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = UpdateParamErrMsg("OutputDesc run");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    string excepted_size = ConcatString("equal to the length of x'shape[", DIM_SIZE4, "]");
    std::string err_msg = GetAttrSizeErrMsg("ksizeList", ConcatString(ksizeList.size()), excepted_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // get input strides
  std::vector<int32_t> stridesList;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stridesList)) {
    std::string err_msg = GetInputInvalidErrMsg("strides");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (stridesList.size() != DIM_SIZE4) {
    string excepted_size = ConcatString("equal to the length of x'shape[", DIM_SIZE4, "]");
    std::string err_msg = GetAttrSizeErrMsg("stridesList", ConcatString(stridesList.size()), excepted_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // get input data_format
  std::string dataFormat;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
    std::string err_msg = GetInputInvalidErrMsg("data_format");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // get input paddingMode
  std::string paddingMode;
  if (GRAPH_SUCCESS != op.GetAttr("padding", paddingMode)) {
    std::string err_msg = GetInputInvalidErrMsg("padding");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (paddingMode != "SAME" && paddingMode != "VALID") {
    string excepted_value = ConcatString("SAME or VALID");
    std::string err_msg = GetAttrValueErrMsg("paddingMode", ConcatString(paddingMode), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    int64_t second_end = 0;
    int64_t start = 0;
    int64_t end = 0;
    first_start = output[0] * stride;
    second_start = output[1] * stride;
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
  int64_t strideH = 0;

  if (dataFormat == "NHWC") {
    inputH = dims_input[1];
    inputW = dims_input[2];
    windowH = ksizeList[1];
    strideH = stridesList[1];
  } else if(dataFormat == "NCHW") {
    inputH = dims_input[2];
    inputW = dims_input[3];
    windowH = ksizeList[2];
    strideH = stridesList[2];
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
    std::string err_msg = GetInputInvalidErrMsg("ksize");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (ksizeList.size() < DIM_SIZE4) {
    string correct_size = ConcatString("more than [", DIM_SIZE4, "]");
    std::string err_msg = GetAttrSizeErrMsg("ksizeList", ConcatString(ksizeList.size()), correct_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  // get input strides
  std::vector<int32_t> stridesList;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stridesList)) {
    std::string err_msg = GetInputInvalidErrMsg("strides");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (stridesList.size() < DIM_SIZE4) {
    string correct_size = ConcatString("more than[", DIM_SIZE4, "]");
    std::string err_msg = GetAttrSizeErrMsg("stridesList", ConcatString(stridesList.size()), correct_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if ((ksizeList[0] != 1) || (ksizeList[3] != 1) || (stridesList[0] != 1) || (stridesList[3] != 1)) {
    std::string err_msg = OtherErrMsg("MaxPoolGradWithArgmax only supports pooling "
                                      "across width/height, and other ksize "
                                      "dimension should be one");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if ((ksizeList[1] * ksizeList[2]) > 255) {
    string excepted_value = ConcatString("less than or equal to 255");
    std::string err_msg = GetAttrValueErrMsg("ksizeList[1] * ksizeList[2]", ConcatString(ksizeList[1] * ksizeList[2]), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  // get input paddingMode
  std::string paddingMode;
  if (GRAPH_SUCCESS != op.GetAttr("padding", paddingMode)) {
    std::string err_msg = GetInputInvalidErrMsg("padding");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (paddingMode != "SAME" && paddingMode != "VALID") {
    string excepted_value = ConcatString("SAME, VALID");
    std::string err_msg = GetAttrValueErrMsg("paddingMode", paddingMode, excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = "get input[ksize] const data failed";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  DataType ksizeDtype = op.GetInputDesc("ksize").GetDataType();
  std::vector<int64_t> ksizeList;
  GetMaxPoolV2ConstData(ksizeData, ksizeDtype, ksizeList);
  if (ksizeList.size() != DIM_SIZE4) {
    std::string err_msg = ConcatString("input ksize data size[",
      ksizeList.size() ,"] must be equal to ", DIM_SIZE4);
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // get input strides
  Tensor stridesData;
  if (ge::GRAPH_SUCCESS != op.GetInputConstData("strides", stridesData)) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
      std::string("get input[strides] const data failed"));
    return GRAPH_FAILED;
  }
  DataType stridesDtype = op.GetInputDesc("strides").GetDataType();
  std::vector<int64_t> stridesList;
  GetMaxPoolV2ConstData(stridesData, stridesDtype, stridesList);
  if (stridesList.size() != DIM_SIZE4) {
    std::string err_msg = ConcatString("input strides data size[",
      stridesList.size() ,"] must be equal to ", DIM_SIZE4);
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  // get input data_format
  std::string dataFormat;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
      std::string("get attr[data_format] failed"));
    return GRAPH_FAILED;
  }
  if (dataFormat != "NHWC" && dataFormat != "NCHW" && dataFormat != "NC1HWC0") {
    std::string err_msg = ConcatString("check attr[data_format] value[",
      dataFormat ,"] failed, only support[NHWC/NCHW/NC1HWC0]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (dataFormat == "NHWC") {
    if ((ksizeList[0] != 1) || (ksizeList[3] != 1) || (stridesList[0] != 1) || (stridesList[3] != 1)) {
      std::string err_msg = ConcatString("check input[ksize/strides] value ",
        "failed, ksize[0]:[", ksizeList[0], "], ksize[3]:[", ksizeList[3], "]",
        "strides[0]:[", stridesList[0], "], strides[3]:[", stridesList[3], "]",
        "must be one.");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }
  if ((dataFormat == "NCHW") || (dataFormat == "NC1HWC0")) {
    if ((ksizeList[0] != 1) || (ksizeList[1] != 1) || (stridesList[0] != 1) || (stridesList[1] != 1)) {
      std::string err_msg = ConcatString("check input[ksize/strides] value ",
        "failed, ksize[0]:[", ksizeList[0], "], ksize[1]:[", ksizeList[1], "]",
        "strides[0]:[", stridesList[0], "], strides[1]:[", stridesList[1], "]",
        "must be one.");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }

  // get input paddingMode
  std::string paddingMode;
  if (GRAPH_SUCCESS != op.GetAttr("padding", paddingMode)) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
      std::string("get attr[padding] failed"));
    return GRAPH_FAILED;
  }

  if (paddingMode != "SAME" && paddingMode != "VALID") {
    OP_LOGE(op.GetName().c_str(),
            "MaxPool can only support SAME or VALID "
            "padding mode!");
    std::string err_msg = ConcatString("check attr[padding] value[",
      paddingMode, "] failed, only support[SAME/VALID]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = GetInputInvalidErrMsg("ksize");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (ksizeList.size() != DIM_SIZE4) {
    string excepted_value = ConcatString("DIM_SIZE4");
    std::string err_msg = GetAttrSizeErrMsg("ksizeList", std::to_string(ksizeList.size()), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // get input strides
  std::vector<int32_t> stridesList;
  if (op.GetAttr("strides", stridesList) != ge::GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("strides");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (stridesList.size() != DIM_SIZE4) {
    string excepted_value = ConcatString("DIM_SIZE4");
    std::string err_msg = GetAttrSizeErrMsg("stridesList", std::to_string(stridesList.size()), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if ((ksizeList[0] != 1) || (ksizeList[3] != 1) || (stridesList[0] != 1) || (stridesList[3] != 1)) {
    std::string err_msg = OtherErrMsg("MaxPoolWithArgmax only supports pooling "
                                      "across width/height, and other ksize "
                                      "dimension should be one");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if ((ksizeList[1] * ksizeList[2]) > 255) {
    string excepted_value = ConcatString("(ksizeList[1] * ksizeList[2]) <= 255");
    int32_t wrong_value = ksizeList[1] * ksizeList[2];
    std::string err_msg = GetAttrValueErrMsg("ksizeList[1] and ksizeList[2]", std::to_string(wrong_value), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  // get input paddingMode
  std::string paddingMode;
  if (op.GetAttr("padding", paddingMode) != ge::GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("padding");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (paddingMode != "SAME" && paddingMode != "VALID") {
    string excepted_value = ConcatString("SAME, VALID");
    std::string err_msg = GetAttrValueErrMsg("paddingMode", paddingMode, excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (((ksizeList[1] > in_size_h) || (ksizeList[2] > in_size_w)) && (paddingMode == "VALID")) {
    std::string err_msg = OtherErrMsg("when padding is VALID, ksize must be not less than input size.");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
  // get input ksize
  std::vector<int32_t> ksizeList;
  if (op.GetAttr("ksize", ksizeList) != ge::GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("ksizeList");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (ksizeList.size() != DIM_SIZE4) {
    std::string err_msg = GetAttrSizeErrMsg("ksizeList", std::to_string(ksizeList.size()), ConcatString(DIM_SIZE4));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // get input strides
  std::vector<int32_t> stridesList;
  if (op.GetAttr("strides", stridesList) != ge::GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("strides");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (stridesList.size() != DIM_SIZE4) {
    std::string err_msg = GetAttrSizeErrMsg("stridesList", std::to_string(stridesList.size()), ConcatString(DIM_SIZE4));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if ((ksizeList[0] != 1) || (ksizeList[3] != 1) || (stridesList[0] != 1) || (stridesList[3] != 1)) {
    std::string err_msg = OtherErrMsg("Mask2Argmax only supports pooling "
                                      "across width/height, and other ksize "
                                      "dimension should be one");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if ((ksizeList[1] * ksizeList[2]) > 255) {
    string excepted_value = ConcatString("less than or equal to 255");
    std::string err_msg = GetAttrValueErrMsg("ksizeList[1] * ksizeList[2]", ConcatString(ksizeList[1] * ksizeList[2]), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // get input paddingMode
  std::string paddingMode;
  if (op.GetAttr("padding", paddingMode) != ge::GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("padding");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (paddingMode != "SAME" && paddingMode != "VALID") {
    string excepted_value = ConcatString("SAME or VALID");
    std::string err_msg = GetAttrValueErrMsg("paddingMode", paddingMode, excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = OtherErrMsg("The max_pool_grad_grad_with_argmax op inputs should have the same dtype!");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = GetInputInvalidErrMsg("ksize_list");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (ksize_list.size() != 4) {
    std::string err_msg = GetAttrSizeErrMsg("ksize_list", ConcatString(ksize_list.size()), ConcatString(4));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // get input strides
  std::vector<int32_t> strides_list;
  if (op.GetAttr("strides", strides_list) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("strides");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (strides_list.size() != 4) {
    std::string err_msg = GetAttrSizeErrMsg("strides_list", ConcatString(strides_list.size()), ConcatString(4));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  for (size_t i = 0; i < strides_list.size(); i++) {
    if (strides_list[i] == 0) {
      string excepted_value = ConcatString("not equal to 0");
      std::string err_msg = GetAttrValueErrMsg(ConcatString("strides_list[", i, "]"), ConcatString(strides_list[i]), excepted_value);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }
  // get input padding_mode
  std::string padding_mode;
  if (op.GetAttr("padding", padding_mode) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("padding");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (padding_mode != "SAME" && padding_mode != "VALID") {
    string excepted_value = ConcatString("SAME or VALID");
    std::string err_msg = GetAttrValueErrMsg("padding_mode", padding_mode, excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = UpdateParamErrMsg("OutputDesc run");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
static void set_avg_pool_grad_out_range(const std::string& pad_str,
                                        size_t idx,
                                        const vector<int64_t>& attrParams,
                                        const std::vector<std::pair<int64_t, int64_t>>& input_grad_range,
                                        std::vector<std::pair<int64_t, int64_t>>& output_range) {
  size_t attrIdx = 0;
  int32_t stride = attrParams[attrIdx++];
  int32_t kernel = attrParams[attrIdx++];
  int64_t low = input_grad_range[idx].first;
  int64_t high = input_grad_range[idx].second;
  if (pad_str == "SAME") {
    output_range[idx].first = stride * (low - 1) + 1;
    output_range[idx].second = stride * high;
  } else {
    output_range[idx].first = stride * (low - 1) + kernel;
    output_range[idx].second = stride * (high - 1) + kernel + stride - 1;
  }
  output_range[idx].first = std::max(output_range[idx].first, kDynamicRangeLowerBound);
  output_range[idx].second = std::min(output_range[idx].second, kDynamicRangeUpperBound);
  if (high == -1) {
    output_range[idx].second = high;
  }
}

static bool set_avg_pool_grad_out_shape_range(ge::Operator& op, const std::string& pad_str,
                                              const std::vector<int64_t>& input_grad_shape,
                                              const std::string& data_format,
                                              std::vector<std::pair<int64_t, int64_t>>& input_grad_range,
                                              const std::vector<int64_t>& ksize,
                                              std::vector<std::pair<int64_t, int64_t>>& output_range,
                                              ge::GeTensorDescPtr& tensordesc_output,
                                              ge::GeTensorDescPtr& input_grad_desc,
                                              bool& unknown_rank) {
  ge::AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);
  if (input_grad_range.size() != input_grad_shape.size() && input_grad_range.size() != 0 && !unknown_rank) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(),
      "length of range(%zu) in dyanmic shape must be equal to the length of shape(%zu), or equal to 0.",
      input_grad_range.size(), input_grad_range.size());
    return false;
  }
  if (input_grad_range.size() == 0 && !unknown_rank) {
    input_grad_range.resize(input_grad_shape.size());
    for (size_t i = 0; i < input_grad_shape.size(); i++) {
      input_grad_range[i].first = std::max(input_grad_shape[i], kDynamicRangeLowerBound);
      input_grad_range[i].second = input_grad_shape[i];
    }
    input_grad_desc->SetShapeRange(input_grad_range);
  }
  if (!output_range.empty() && output_range.size() == kAvgPoolGradOriShapeDim && !unknown_rank) {
    tensordesc_output->SetShapeRange(output_range);
  } else {
    std::vector<int64_t> output_sizes = tensordesc_output->MutableShape().GetDims();
    if (output_sizes.empty() || output_sizes.size() != 4) {
      OP_LOGE(op_name.GetString(), "output_sizes list should be 4D. actual is: %lu.", output_sizes.size());
      map<string, string> err_map;
      err_map["param_name"] = "output_sizes";
      err_map["op_name"] = op_name.GetString();
      err_map["expected_value"] = "4D";
      err_map["input_value"] = std::to_string(output_sizes.size()) + "D.";
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return false;
    }
    int32_t h_position = data_format.find("H");
    int32_t w_position = data_format.find("W");
    int32_t c_position = data_format.find("C");
    int32_t n_position = data_format.find("N");

    if (unknown_rank) {
      vector<int64_t> output_shape;
      output_shape.resize(kAvgPoolGradOriShapeDim);
      output_shape[n_position] = -1;
      output_shape[h_position] = -1;
      output_shape[w_position] = -1;
      output_shape[c_position] = -1;
      tensordesc_output->SetShape(GeShape(output_shape));
      return true;
    }

    int64_t output_h = output_sizes[h_position];
    int64_t output_w = output_sizes[w_position];

    int64_t filter_h = ksize[h_position];
    int64_t filter_w = ksize[w_position];

    std::vector<int64_t> strides;
    strides = GetAttrValue(op, "strides");
    int64_t stride_h = strides[h_position];
    int64_t stride_w = strides[w_position];

    output_range.resize(kAvgPoolGradOriShapeDim);
    output_range[c_position] = std::make_pair(input_grad_shape[c_position], input_grad_shape[c_position]);
    output_range[h_position] = std::make_pair(output_h, output_h);
    output_range[w_position] = std::make_pair(output_w, output_w);
    output_range[n_position] = std::make_pair(input_grad_shape[n_position], input_grad_shape[n_position]);
    if (!input_grad_range.empty() && input_grad_range.size() == input_grad_range.size()) {
      output_range[n_position] = input_grad_range[n_position];
      if (output_h == -1) {
        vector<int64_t> attr_params_h = {stride_h, filter_h};
        set_avg_pool_grad_out_range(pad_str, h_position, attr_params_h, input_grad_range, output_range);
      }
      if (output_w == -1) {
        vector<int64_t> attr_params_w = {stride_w, filter_w};
        set_avg_pool_grad_out_range(pad_str, w_position, attr_params_w, input_grad_range, output_range);
      }
      tensordesc_output->SetShapeRange(output_range);
    }
  }
  return true;
}

static void reset_avg_pool_grad_out_shape(const std::vector<int64_t>&dy_sizes,
                                          const std::string& dy_format, std::vector<int64_t>& input_sizes,
                                          const std::string& input_format) {
  int32_t h_input_position = input_format.find("H");
  int32_t w_input_position = input_format.find("W");
  int32_t n_input_position = input_format.find("N");
  int32_t h_dy_position = dy_format.find("H");
  int32_t w_dy_position = dy_format.find("W");
  int32_t n_dy_position = dy_format.find("N");
  if (dy_sizes[n_dy_position] == -1) {
    input_sizes[n_input_position] = -1;
  }
  if (dy_sizes[h_dy_position] == -1) {
    input_sizes[h_input_position] = -1;
  }
  if (dy_sizes[w_dy_position] == -1) {
    input_sizes[w_input_position] = -1;
  }
}

IMPLEMT_VERIFIER(AvgPoolGrad, AvgPoolGradVerify) {
  ge::AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  std::vector<int64_t> ksize;
  ksize = GetAttrValue(op, "ksize");
  if (!CheckListEmpty(op_name.GetString(), ksize, "ksize")) {
    return GRAPH_FAILED;
  }
  if (ksize.size() < 4) {
    OP_LOGE(op_name.GetString(), "Attr ksize(%lu) is too small", ksize.size());
  }
  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op_name.GetString(), strides, "strides")) {
    return GRAPH_FAILED;
  }
  if (strides.size() < 4) {
    OP_LOGE(op_name.GetString(), "Attr strides(%lu) is too small", strides.size());
    return GRAPH_FAILED;
  }

  std::string padding;
  CHECK(op.GetAttr("padding", padding) != GRAPH_SUCCESS, OP_LOGE(op_name.GetString(), "Get padding failed!"),
        return GRAPH_FAILED);
  if (padding != "SAME" && padding != "VALID") {
    OP_LOGE(op_name.GetString(), "Attr padding(%s) only support SAME and VALID", padding.c_str());
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_SUCCESS) {
    if (data_format != "NCHW" && data_format != "NHWC") {
      OP_LOGE(op_name.GetString(), "Attr data_format(%s) only support NCHW and NHWC", data_format.c_str());
      return GRAPH_FAILED;
    }
  }

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

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AvgPoolGradInferShape) {
  ge::AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGI(op_name.GetString(), "Enter AvgPoolGrad inferfunction!");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto input_sizes_desc = op_desc->MutableInputDesc("orig_input_shape");
  auto input_grad_desc = op_desc->MutableInputDesc("input_grad");
  auto tensordesc_output = op_desc->MutableOutputDesc("out_grad");
  auto input_grad_shape = input_grad_desc->MutableShape().GetDims();
  std::vector<std::string> input_infer_depends = {"orig_input_shape"};
  op_desc->SetOpInferDepends(input_infer_depends);
  Tensor orig_input_shape_tensor;
  std::vector<int64_t> orig_input_size;
  bool is_input_size_const = false;
  bool is_dynamic =false;

  if (op.GetInputConstData("orig_input_shape", orig_input_shape_tensor) == GRAPH_SUCCESS) {
    OP_LOGD(op_name.GetString(), "Get constdata success");
    DataType dtype = input_sizes_desc->GetDataType();
    GetConstValue(op, orig_input_shape_tensor, dtype, orig_input_size);
    is_input_size_const = true;
  }
  if (IsUnKnownShape(input_grad_shape)) {
    // when static op or dynamic op phase_running, is_dynamic == False
    is_dynamic = true;
    reset_range(op, "input_grad");
  }
  DataType output_dtype = input_grad_desc->GetDataType();
  bool unknown_rank = IsUnknownRankShape(input_grad_shape);
  if (is_dynamic || (!is_input_size_const && unknown_rank)) {
    // get shape for output from input_size
    std::string pad_str;
    if (!unknown_rank && GRAPH_SUCCESS == op.GetAttr("padding", pad_str) && pad_str == "SAME") {
      op.SetAttr("pads", {-1, -1, -1, -1});
    } else if (!unknown_rank && GRAPH_SUCCESS == op.GetAttr("padding", pad_str) && pad_str == "VALID") {
      op.SetAttr("pads", {0, 0, 0, 0});
    }
    std::string data_format;
    op.GetAttr("data_format", data_format);
    std::vector<std::pair<int64_t, int64_t>> input_grad_range;
    input_grad_desc->GetShapeRange(input_grad_range);
    std::vector<std::pair<int64_t, int64_t>> output_range;
    input_sizes_desc->GetValueRange(output_range);
    std::vector<int64_t> ksize;
    ksize = GetAttrValue(op, "ksize");
    if (!set_avg_pool_grad_out_shape_range(op, pad_str, input_grad_shape, data_format, input_grad_range, ksize,
                                          output_range, tensordesc_output, input_grad_desc, unknown_rank)) {
      return GRAPH_FAILED;
    }
    for (size_t i = 0; i < output_range.size(); i++) {
      if (output_range[i].first == output_range[i].second) {
        orig_input_size.push_back(output_range[i].first);
      } else {
        orig_input_size.push_back(-1);
      }
    }
    if (!unknown_rank) {
      reset_avg_pool_grad_out_shape(input_grad_shape, data_format, orig_input_size, data_format);
    }
  }
  if (orig_input_size.size() == input_grad_shape.size()) {
    tensordesc_output->SetShape(GeShape(orig_input_size));
  }
  tensordesc_output->SetDataType(output_dtype);

  OP_LOGD(op_name.GetString(), "Leave AvgPoolGrad inferfunction!");
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
    OP_LOGE(op.GetName().c_str(), "Attr ksize(%lu) is too small", ksize.size());
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
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ksize failed!");
    return GRAPH_FAILED;
  }
  if (ksize.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "Size of ksize must be 4!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> strides;
  if (GRAPH_SUCCESS != op.GetAttr("strides", strides)) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr strides failed!");
    return GRAPH_FAILED;
  }
  if (strides.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "Size of strides must be 4!");
    return GRAPH_FAILED;
  }

  std::string padding_mode;
  if (ge::GRAPH_SUCCESS != op.GetAttr("padding_mode", padding_mode)) {
    OP_LOGE(op.GetName().c_str(), "Get padding_mode failed!");
    return GRAPH_FAILED;
  }
  if (padding_mode != "SAME" && padding_mode != "VALID" && padding_mode != "CALCULATED") {
    OP_LOGE(op.GetName().c_str(), "attr padding_mode(%s) only support SAME VALID and CALCULATED", padding_mode.c_str());
    return GRAPH_FAILED;
  }

  std::vector<int64_t> pads;
  if (GRAPH_SUCCESS != op.GetAttr("pads", pads)) {
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
    OP_LOGE(op.GetName().c_str(), "The AvgPoolV2Grad op GetOpAttr ceil_mode failed!");
    return GRAPH_FAILED;
  }

  if (ceilMode && padding_mode == "VALID") {
    OP_LOGE(op.GetName().c_str(),
            "When Attr(padding_mode) is VALID, Attr(ceil_mode) must be False. Received ceil_mode: True.");
  }

  std::string data_format;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", data_format)) {
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
    OP_LOGE(op.GetName().c_str(), "GetOpAttr orig_input_shape failed!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> ksize;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksize)) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ksize failed!");
    return GRAPH_FAILED;
  }
  if (ksize.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "Size of ksize must be 4!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> strides;
  if (GRAPH_SUCCESS != op.GetAttr("strides", strides)) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr strides failed!");
    return GRAPH_FAILED;
  }
  if (strides.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "Size of strides must be 4!");
    return GRAPH_FAILED;
  }

  std::string padding_mode;
  if (ge::GRAPH_SUCCESS != op.GetAttr("padding_mode", padding_mode)) {
    OP_LOGE(op.GetName().c_str(), "Get padding_mode failed!");
    return GRAPH_FAILED;
  }
  if (padding_mode != "SAME" && padding_mode != "VALID" && padding_mode != "CALCULATED") {
    OP_LOGE(op.GetName().c_str(), "attr padding_mode(%s) only support SAME VALID and CALCULATED", padding_mode.c_str());
    return GRAPH_FAILED;
  }

  std::vector<int64_t> pads;
  if (GRAPH_SUCCESS != op.GetAttr("pads", pads)) {
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
    OP_LOGE(op.GetName().c_str(), "The AvgPoolV2GradD op GetOpAttr ceil_mode failed!");
    return GRAPH_FAILED;
  }

  if (ceilMode && padding_mode == "VALID") {
    OP_LOGE(op.GetName().c_str(),
            "When Attr(padding_mode) is VALID, Attr(ceil_mode) must be False. Received ceil_mode: True.");
  }

  std::string data_format;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", data_format)) {
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
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),ConcatString("call WithRank failed, ",
        GetShapeErrMsg(0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "4D")));
    return GRAPH_FAILED;
  }

  auto type = op.GetInputDesc("orig_input").GetDataType();
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(Shape(input_shape));
  output_desc.SetDataType(type);
  if (op.UpdateOutputDesc("y", output_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        std::string("update output[y] desc failed"));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(FractionalMaxPoolGrad, FractionalMaxPoolGradInfer);

IMPLEMT_INFERFUNC(FractionalAvgPool, FractionalAvgPoolInfer) {
  Shape input;
  if (WithRank(op.GetInputDesc(0), 4, input, op.GetName().c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), ConcatString("Call WithRank function failed, ",
        GetShapeErrMsg(0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "4D")));
    return GRAPH_FAILED;
  }
  std::vector<float> pooling_ratio;
  op.GetAttr("pooling_ratio", pooling_ratio);
  if (pooling_ratio.size() != 4) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        GetAttrValueErrMsg("pooling_ratio", ConcatString(pooling_ratio.size()), "4"));
    return GRAPH_PARAM_INVALID;
  }
  auto x_dims = op.GetInputDesc(0).GetShape().GetDims();
  std::vector<int64_t> dims;
  dims.reserve(4);
  for (int i = 0; i < 4; ++i) {
    int64_t val = ge::UNKNOWN_DIM;
    if (x_dims[i] != ge::UNKNOWN_DIM) {
      val = static_cast<int64_t>(x_dims[i] / pooling_ratio[i]);
      if (val < 0) {
        AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
          ConcatString("size computed for ", i, "th dim is ", val, ", should be >= 0"));
        return GRAPH_PARAM_INVALID;
      }
    }

    OP_LOGI(op.GetName().c_str(), "i = %d, x_dims[i] = %lld, pooling_ratio[i] = %f, val = %lld",
        i, x_dims[i], pooling_ratio[i], val);
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
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), ConcatString("call WithRank failed, ",
        GetShapeErrMsg(0, DebugString(tensor.GetShape().GetDims()), "4D")));
    return GRAPH_FAILED;
  }
  std::vector<float> pooling_ratio;
  pooling_ratio = op.get_attr_pooling_ratio();
  if (pooling_ratio.size() != 4) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        GetAttrSizeErrMsg("pooling_ratio", DebugString(tensor.GetShape().GetDims()), "4D"));
    return GRAPH_FAILED;
  }

  std::vector<int64_t> output_dims;
  for (int i = 0; i < 4; ++i) {
    int64_t dim = input_value.GetDim(i);
    if (dim != UNKNOWN_DIM) {
      auto real_dim = static_cast<int64_t>(dim / pooling_ratio[i]);
      if (real_dim < 0) {
        string err_msg = ConcatString("size computed for ", i, "th dim of output[y] is ", real_dim);
        AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        std::string("update output[y] desc failed."));
    return GRAPH_FAILED;
  }

  TensorDesc row_pooling_desc = op.GetOutputDesc("row_pooling_sequence");
  row_pooling_desc.SetShape(Shape({output_dims[1] + 1}));
  row_pooling_desc.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("row_pooling_sequence", row_pooling_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        std::string("update output[row_pooling_sequence] desc failed."));
    return GRAPH_FAILED;
  }

  TensorDesc col_pooling_desc = op.GetOutputDesc("col_pooling_sequence");
  col_pooling_desc.SetShape(Shape({output_dims[2] + 1}));
  col_pooling_desc.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("col_pooling_sequence", col_pooling_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        std::string("update output[col_pooling_sequence] desc failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(FractionalMaxPool, FractionalMaxPoolInfer);

IMPLEMT_INFERFUNC(DataFormatVecPermute, DataFormatVecPermuteInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto x_desc = op_desc->MutableInputDesc(0);

  std::vector<std::pair<int64_t, int64_t>> range;
  if (x_desc->GetShapeRange(range) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  DataType y_type = x_desc->GetDataType();

  auto y_desc = op_desc->MutableOutputDesc(0);
  y_desc->SetShape(x_desc->GetShape());
  y_desc->SetShapeRange(range);
  y_desc->SetDataType(y_type);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DataFormatVecPermute, DataFormatVecPermuteInfer);

IMPLEMT_INFERFUNC(FractionalAvgPoolGrad, FractionalAvgPoolGradInfer) {
  Tensor tensor;
  if (op.GetInputConstData("orig_input_tensor_shape", tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        ConcatString("get const data from input[orig_input_tensor_shape] failed"));
    return GRAPH_FAILED;
  }

  Shape result;
  if (MakeShapeFromShapeTensor(tensor, result, op.GetName().c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
        ConcatString("call MakeShapeFromShapeTensor function failed to make",
            " shape from input[orig_input_tensor_shape] data"));
    return GRAPH_FAILED;
  }

  DataType y_type = op.GetInputDesc("out_backprop").GetDataType();
  TensorDesc out_desc = op.GetOutputDesc("y");
  out_desc.SetShape(Shape(result));
  out_desc.SetDataType(y_type);
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        std::string("update output[y] desc failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(FractionalAvgPoolGrad, FractionalAvgPoolGradInfer);

IMPLEMT_INFERFUNC(NthElement, NthElementInfer) {
  Shape x_shape;
  auto x_tensor = op.get_input_desc_x();
  if (WithRankAtLeast(x_tensor, 1, x_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
      "failed to call WithRankAtLeast function, ",
      "input[x] rank must be at least 1D, but got rank[",
      op.get_input_desc_x().GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  Tensor n_tensor;
  if (op.GetInputConstData("n", n_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
      std::string("get const data[n] failed"));
    return GRAPH_FAILED;
  }

  int64_t n_dim = 0;
  if (MakeDimForScalarInput(n_tensor, n_dim, op.GetName().c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
      std::string("failed to call MakeDimForScalarInput function, "
      "get input n shape failed"));
    return GRAPH_FAILED;
  }

  int64_t existing = x_shape.GetDimNum();
  int64_t last_input_dim = x_shape.GetDim(existing - 1);
  if ((last_input_dim != ge::UNKNOWN_DIM) && (n_dim != ge::UNKNOWN_DIM) && (last_input_dim <= n_dim)) {
    std::string err_msg = ConcatString("input[x] last dim value[",
      last_input_dim, "] must be greater than [", n_dim, "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  Shape output_shape;
  if (SubShape(x_shape, 0, -1, 1, output_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
      "failed to call SubShape function, input[x] shape[",
      DebugString(x_shape.GetDims()), "], start[0], end[-1], stride[1]");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
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
    } else if (padStr.compare("CALCULATED") == 0) {
      // Pytorch
      if (op.GetAttr("pads", padList) != ge::GRAPH_SUCCESS) {
        std::string err_msg = GetInputInvalidErrMsg("pads");
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
        return false;
      }
      return true;
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
    std::string err_msg = GetInputInvalidErrMsg("pads");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = GetInputInvalidErrMsg("strides");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<int32_t> ksizeList;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksizeList)) {
    std::string err_msg = GetInputInvalidErrMsg("ksize");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = GetInputInvalidErrMsg("ksize");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (ksize.size() != 5) {
    std::string err_msg = GetAttrSizeErrMsg("ksize", ConcatString(ksize.size()), ConcatString(5));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), strides, "strides")) {
    std::string err_msg = GetInputInvalidErrMsg("strides");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (strides.size() != 5) {
    std::string err_msg = GetAttrSizeErrMsg("strides", ConcatString(strides.size()), ConcatString(5));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
    if (data_format != "NDHWC" && data_format != "NCDHW") {
      string expected_format_list = ConcatString("NDHWC, NCDHW");
      std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = GetInputInvalidErrMsg("attrs");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = GetAttrValueErrMsg("strd/strh/strw", ConcatString(strd, "/", strh, "/", strw), ConcatString(0));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (false == GetPadMaxPool3DGrad(op, id, ih, iw, kd, kh, kw, strd, strh, strw, padf, padba, padt, padb, padl, padr)) {
    std::string err_msg = GetInputInvalidErrMsg("pads attrs");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto input_x_desc = op_desc->GetInputDescPtr(0);
  auto output_y_desc = op_desc->MutableOutputDesc(0);
  Format input_format = input_x_desc->GetFormat();
  const GeShape &input_shape = input_x_desc->GetShape();
  GeShape &output_shape = output_y_desc->MutableShape();

  auto input_w_size = 0;
  input_w_size = input_shape.GetDim(3);

  uint32_t ksize{0};
  uint32_t strides{1};
  bool ceil_mode{false};

  if (!AttrUtils::GetInt(op_desc, "ksize", ksize)) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ksize failed, set ksize default value");
  }
  if (!AttrUtils::GetInt(op_desc, "strides", strides)) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr strides failed, set strides default value");
  }
  if (strides == 0) {
    OP_LOGE(op.GetName().c_str(), "Value of strides should not 0");
    return GRAPH_FAILED;
  }
  // get input ksize
  std::vector<int32_t> pads_list;
  if (!AttrUtils::GetListInt(op_desc, "pads", pads_list)) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr pads_list failed!");
    return GRAPH_FAILED;
  }
  if (pads_list.size() < 2) {
    OP_LOGE(op.GetName().c_str(), "Attr pads should has 2 elements at least!");
    return GRAPH_FAILED;
  }
  if (!AttrUtils::GetBool(op_desc, "ceil_mode", ceil_mode)) {
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

  size_t rank = input_shape.GetDimNum();
  output_shape.SetDimNum(rank);

  for (size_t i = 0; i < rank; i++) {
    if (i == 3) {
      output_shape.SetDim(i, output_w_size);
      continue;
    }
    int64_t dim_size = input_shape.GetDim(i);
    output_shape.SetDim(i, dim_size);
  }
  output_y_desc->SetDataType(input_x_desc->GetDataType());

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
  auto op_desc_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc_x = op_desc_info->MutableInputDesc("x");
  auto output_desc_y = op_desc_info->MutableOutputDesc("y");
  vector<int64_t> x_shape = input_desc_x->MutableShape().GetDims();
  DataType input_dtype = input_desc_x->GetDataType();
  std::vector<std::pair<int64_t, int64_t>> x_range;
  input_desc_x->GetShapeRange(x_range);

  MakeUpShapeRange(x_shape, x_range);
  output_desc_y->SetShape(GeShape(x_shape));
  output_desc_y->SetShapeRange(x_range);
  output_desc_y->SetDataType(input_dtype);
  return GRAPH_SUCCESS;
}
// Registered inferfunction
COMMON_INFER_FUNC_REG(MaxPoolGradWithArgmaxV2, MaxPoolGradWithArgmaxV2InferShape);
// Registered verify function
VERIFY_FUNC_REG(MaxPoolGradWithArgmaxV2, MaxPoolGradWithArgmaxV2Verify);
// ----------------------MaxPoolGradWithArgmaxV2-----------------------

// ---------------------MaxPoolWithArgmaxV2---------------------
static int64_t DivRtn(int64_t x, int64_t y) {
  int64_t q = x / y;
  int64_t r = x % y;
  if ((r != 0) && ((r < 0) != (y < 0))) {
    --q;
  }
  return q;
}

static void UpdateMaxDimAndRange(const int64_t& ksize, const int64_t& strides, bool ceil_mode, int32_t pad_a,
                                 int64_t& dim_size, std::pair<int64_t, int64_t>& dim_range) {
  if (dim_size != -1) {
    int64_t output_dim_size = 0;
    int64_t exact_size = dim_size + 2 * pad_a - (ksize - 1) - 1 + ((ceil_mode == 1) ? (strides - 1) : 0);
    output_dim_size = DivRtn(exact_size, strides) + 1;
    if (pad_a > 0) {
      if ((output_dim_size - 1) * strides >= dim_size + pad_a) {
        output_dim_size = output_dim_size - 1;
      }
    }
    dim_range = std::pair<int64_t, int64_t>{output_dim_size, output_dim_size};
    dim_size = output_dim_size;
  } else {
    int64_t first_range = 0;
    int64_t second_range = 0;
    if (dim_range.first == 1) {
      first_range = 1;
    } else {
      int64_t exact_size = dim_range.first + 2 * pad_a - (ksize - 1) - 1 + ((ceil_mode == 1) ? (strides - 1) : 0);
      first_range = DivRtn(exact_size, strides) + 1;
      if (pad_a > 0) {
        if ((first_range - 1) * strides >= dim_range.first + pad_a) {
          first_range = first_range - 1;
        }
      }
    }
    if (dim_range.second == -1) {
      second_range = -1;
    } else {
      int64_t exact_size = dim_range.second + 2 * pad_a - (ksize - 1) - 1 + ((ceil_mode == 1) ? (strides - 1) : 0);
      second_range = DivRtn(exact_size, strides) + 1;
      if (pad_a > 0) {
        if ((second_range - 1) * strides >= dim_range.second + pad_a) {
          second_range = second_range - 1;
        }
      }
    }
    dim_range = std::pair<int64_t, int64_t>{first_range, second_range};
  }
}

static void UpdateMaskW(const int64_t& out_h, const int64_t& out_w,
                        const int64_t& first_h, const int64_t& first_w,
                        const int64_t& second_h, const int64_t& second_w,
                        int64_t& dim_size, std::pair<int64_t, int64_t>& dim_range) {
  if (dim_size != -1) {
    int64_t mask_w = out_h * out_w;
    if (mask_w % 16 != 0) {
      mask_w = mask_w/16 + 1;
    } else {
      mask_w = mask_w/16;
    }
    mask_w = mask_w + 1;
    dim_range = std::pair<int64_t, int64_t>{mask_w, mask_w};
    dim_size = mask_w;
  } else {
    int64_t first_range = 0;
    int64_t second_range = 0;
    if (dim_range.first == 1) {
      first_range = 1;
    } else {
      int64_t mask_w = first_h * first_w;
      if (mask_w % 16 != 0) {
        mask_w = mask_w/16 + 1;
      } else {
        mask_w = mask_w/16;
      }
      first_range = mask_w + 1;
    }
    if (dim_range.second == -1) {
      second_range = -1;
    } else {
      int64_t mask_w = second_h * second_w;
      if (mask_w % 16 != 0) {
        mask_w = mask_w/16 + 1;
      } else {
        mask_w = mask_w/16;
      }
      second_range = mask_w + 1;
    }
    dim_range = std::pair<int64_t, int64_t>{first_range, second_range};
  }
}


IMPLEMT_VERIFIER(MaxPoolWithArgmaxV2, MaxPoolWithArgmaxV2Verify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(MaxPoolWithArgmaxV2InferShape) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  if (op_info == nullptr) {
    OP_LOGE(op.GetName().c_str(), "Get op_info failed.");
    return GRAPH_FAILED;
  }
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_shape = input_desc->MutableShape();
  auto input_format = input_desc->GetFormat();
  DataType input_dtype = input_desc->GetDataType();
  auto output_max = op_info->MutableOutputDesc("y");
  auto output_mask = op_info->MutableOutputDesc("argmax");

  std::vector<int32_t> ksize;
  if (op.GetAttr("ksize", ksize) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("ksize");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (ksize.size() != 4) {
    GE_OP_LOGE(op.GetName().c_str(), "Length of ksize must be 4!");
    return GRAPH_FAILED;
  }

  std::vector<int32_t> strides;
  if (GRAPH_SUCCESS != op.GetAttr("strides", strides)) {
    std::string err_msg = GetInputInvalidErrMsg("strides");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (strides.size() != 4) {
    GE_OP_LOGE(op.GetName().c_str(), "Length of strides must be 4!");
    return GRAPH_FAILED;
  }

  std::vector<int32_t> pads;
  if (GRAPH_SUCCESS != op.GetAttr("pads", pads)) {
    std::string err_msg = GetInputInvalidErrMsg("pads");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (pads.size() != 4) {
    GE_OP_LOGE(op.GetName().c_str(), "Length of pads must be 4!");
    return GRAPH_FAILED;
  }

  int32_t pad_top = pads[2];
  int32_t pad_left = pads[3];

  bool ceil_mode = 0;
  if (GRAPH_SUCCESS != op.GetAttr("ceil_mode", ceil_mode)) {
    std::string err_msg = GetInputInvalidErrMsg("ceil_mode");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> input_dims = input_shape.GetDims();
  std::vector<std::pair<int64_t, int64_t>> input_range;
  if (IsUnknownRankShape(input_dims)) {
    OP_LOGW(op.GetName().c_str(), "the input os unknown rank, will set the input [-1, -1, -1, -1].");
    input_dims = {-1, -1, -1, -1};
  } else {
    input_desc->GetShapeRange(input_range);
  }
  MakeUpShapeRange(input_dims, input_range);

  std::vector<int64_t> max_dims;
  std::vector<std::pair<int64_t, int64_t>> max_range;
  std::vector<int64_t> mask_dims;
  std::vector<std::pair<int64_t, int64_t>> mask_range;

  auto input_h_dim = input_format == FORMAT_NHWC ? 1 : 2;
  auto input_w_dim = input_format == FORMAT_NHWC ? 2 : 3;
  auto strides_h_dim = input_format == FORMAT_NHWC ? 1 : 2;
  auto strides_w_dim = input_format == FORMAT_NHWC ? 2 : 3;

  for (size_t i = 0; i < input_dims.size(); i++) {
    int64_t max_dim_size = input_dims[i];
    auto max_dim_range = input_range[i];
    if (i == input_h_dim) {
      UpdateMaxDimAndRange(ksize[strides_h_dim], strides[strides_h_dim], ceil_mode,
                                  pad_top, max_dim_size, max_dim_range);
    } else if (i == input_w_dim) {
      UpdateMaxDimAndRange(ksize[strides_w_dim], strides[strides_w_dim], ceil_mode,
                                  pad_left, max_dim_size, max_dim_range);
    }
    max_dims.push_back(max_dim_size);
    max_range.push_back(max_dim_range);
  }

  for (size_t i = 0; i < input_dims.size(); i++) {
    int64_t mask_dim_size = input_dims[i];
    auto mask_dims_range = input_range[i];
    if (i == input_h_dim) {
      mask_dim_size = ksize[strides_h_dim] * ksize[strides_w_dim];
      mask_dims_range = std::pair<int64_t, int64_t>{mask_dim_size, mask_dim_size};
    } else if (i == input_w_dim) {
      UpdateMaskW(max_dims[strides_h_dim], max_dims[strides_w_dim],
       max_range[strides_h_dim].first, max_range[strides_w_dim].first,
       max_range[strides_h_dim].second, max_range[strides_w_dim].second,
       mask_dim_size, mask_dims_range);
    }
    mask_dims.push_back(mask_dim_size);
    mask_range.push_back(mask_dims_range);
  }

  output_max->SetShape(GeShape(max_dims));
  output_max->SetShapeRange(max_range);
  output_max->SetDataType(input_dtype);
  output_mask->SetShape(GeShape(mask_dims));
  output_mask->SetShapeRange(mask_range);
  output_mask->SetDataType(DT_UINT16);
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(MaxPoolWithArgmaxV2, MaxPoolWithArgmaxV2InferShape);

// Registered verify function
VERIFY_FUNC_REG(MaxPoolWithArgmaxV2, MaxPoolWithArgmaxV2Verify);
// ---------------------MaxPoolWithArgmaxV2---------------------


// ----------------------MaxPoolV3------------------------------
static void SAMEUpdateDimAndRange(const int64_t& ksize, const int64_t& strides, int64_t& dim_size,
                                  std::pair<int64_t, int64_t>& dim_range) {
  // warning strides divisor cannnot be 0
  CHECK(strides == 0, OP_LOGW("SAMEUpdateDimAndRange", "strides divisor cannnot be 0"), return);
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


static void CALCULATEUpdateDimAndRange(const int64_t& ksize, const int64_t& strides, bool ceil_mode, int32_t pad_a,
                                       int32_t pad_b, int64_t& dim_size, std::pair<int64_t, int64_t>& dim_range) {
  // warning strides divisor cannnot be 0
  CHECK(strides == 0, OP_LOGW("CALCULATEUpdateDimAndRange", "strides divisor cannnot be 0"), return);
  if (dim_size != -1) {
    int64_t output_dim_size = 0;
    if (ceil_mode) {
      output_dim_size = (dim_size + pad_a + pad_b - ksize + strides + strides - 1) / strides;
    } else {
      output_dim_size = (dim_size + pad_a + pad_b - ksize + strides) / strides;
    }
    dim_range = std::pair<int64_t, int64_t>{output_dim_size, output_dim_size};
    dim_size = output_dim_size;
  } else {
    int64_t first_range = 0;
    int64_t second_range = 0;
    if (ceil_mode) {
      first_range =
          dim_range.first == 1 ? 1 : (dim_range.first + pad_a + pad_b - ksize + strides + strides - 1) / strides;
      second_range =
          dim_range.second == -1 ? -1 : (dim_range.second + pad_a + pad_b - ksize + strides + strides - 1) / strides;
    } else {
      first_range = dim_range.first == 1 ? 1 : (dim_range.first + pad_a + pad_b - ksize + strides) / strides;
      second_range = dim_range.second == -1 ? -1 : (dim_range.second + pad_a + pad_b - ksize + strides) / strides;
    }
    dim_range = std::pair<int64_t, int64_t>{first_range, second_range};
  }
}


IMPLEMT_VERIFIER(MaxPoolV3, MaxPoolV3Verify) {
  auto inputTensorDesc = op.GetInputDesc("x");
  auto shape = inputTensorDesc.GetShape();
  Format input_format = inputTensorDesc.GetFormat();

  // Verify
  std::vector<int64_t> ksize;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksize)) {
    std::string err_msg = GetInputInvalidErrMsg("ksize");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (ksize.size() < 4) {
    return GRAPH_FAILED;
  }

  std::vector<int64_t> strides;
  if (GRAPH_SUCCESS != op.GetAttr("strides", strides)) {
    std::string err_msg = GetInputInvalidErrMsg("strides");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (strides.size() < 4) {
    return GRAPH_FAILED;
  }

  std::string padding_mode;
  if (ge::GRAPH_SUCCESS != op.GetAttr("padding_mode", padding_mode)) {
    std::string err_msg = GetInputInvalidErrMsg("padding_mode");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (padding_mode != "SAME" && padding_mode != "VALID" && padding_mode != "CALCULATED") {
    string excepted_value = ConcatString("SAME or VALID or CALCULATED");
    std::string err_msg = GetAttrValueErrMsg("padding_mode", padding_mode, excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> pads;
  if (GRAPH_SUCCESS != op.GetAttr("pads", pads)) {
    std::string err_msg = GetInputInvalidErrMsg("pads");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (pads.size() < 4) {
    return GRAPH_FAILED;
  }

  std::string data_format;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", data_format)) {
    std::string err_msg = GetInputInvalidErrMsg("data_format");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  bool ceilMode;
  if (GRAPH_SUCCESS != op.GetAttr("ceil_mode", ceilMode)) {
    return GRAPH_FAILED;
  }

  if (data_format != "NCHW" && data_format != "NHWC" && data_format != "NC1HWC0") {
    string expected_format_list = ConcatString("NCHW, NHWC, NC1HWC0");
    std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (data_format == "NCHW" || data_format == "NC1HWC0") {
    if (ksize[0] != 1 || ksize[1] != 1 || strides[0] != 1 || strides[1] != 1) {
      std::string err_msg = OtherErrMsg("MaxPoolV3Grad only supports pooling across width/height"
                                        "and other ksize dimension should be one");
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    if (padding_mode == "CALCULATED" &&
        (pads[0] >= ksize[2] || pads[1] >= ksize[2] || pads[2] >= ksize[3] || pads[3] >= ksize[3])) {
      std::string err_msg = OtherErrMsg("Pads must be less then ksize when using CALCULATED mode!");
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }
  if (data_format == "NHWC") {
    if (ksize[0] != 1 || ksize[3] != 1 || strides[0] != 1 || strides[3] != 1) {
      std::string err_msg = OtherErrMsg("MaxPoolV3 only supports pooling across width/height"
                                        "and other ksize dimension should be one");
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    if (padding_mode == "CALCULATED" &&
        (pads[0] >= ksize[1] || pads[1] >= ksize[1] || pads[2] >= ksize[2] || pads[3] >= ksize[2])) {
      std::string err_msg = OtherErrMsg("Pads must be less then ksize when using CALCULATED mode!");
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }

  // input format mast equals to data_format
  if ((input_format == FORMAT_NCHW && data_format != "NCHW") ||
      (input_format == FORMAT_NHWC && data_format != "NHWC")) {
      string err_msg1 = ConcatString("Format of input must be same with dataFormat! input_format:",input_format, ", data_format:",data_format);
      std::string err_msg = OtherErrMsg(err_msg1);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
  }

  if(ceilMode && padding_mode == "VALID") {
    std::string err_msg = OtherErrMsg("When padding_mode is 'VALID', ceil_mode must be False");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

// infer
IMPLEMT_COMMON_INFERFUNC(MaxPoolV3InferShape) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  if (op_info == nullptr) {
    OP_LOGE(op.GetName().c_str(), "Get op_info failed.");
    return GRAPH_FAILED;
  }
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_shape = input_desc->MutableShape();
  auto input_format = input_desc->GetFormat();
  auto input_dtype = input_desc->GetDataType();
  auto output_desc = op_info->MutableOutputDesc("y");
  output_desc->SetDataType(input_dtype);

  std::vector<int32_t> ksize;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksize)) {
    std::string err_msg = GetInputInvalidErrMsg("ksize");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<int32_t> strides;
  if (GRAPH_SUCCESS != op.GetAttr("strides", strides)) {
    std::string err_msg = GetInputInvalidErrMsg("strides");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  std::string data_format;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", data_format)) {
    std::string err_msg = GetInputInvalidErrMsg("data_format");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  std::string padding_mode;
  if (GRAPH_SUCCESS != op.GetAttr("padding_mode", padding_mode)) {
    std::string err_msg = GetInputInvalidErrMsg("padding_mode");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<int32_t> pads;
  if (GRAPH_SUCCESS != op.GetAttr("pads", pads)) {
    std::string err_msg = GetInputInvalidErrMsg("pads");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (pads.size() != 4) {
    GE_OP_LOGE(op.GetName().c_str(), "Length of pads must be 4!");
    return GRAPH_FAILED;
  }
  int32_t pad_top = pads[0];
  int32_t pad_bottom = pads[1];
  int32_t pad_left = pads[2];
  int32_t pad_right = pads[3];

  bool global_pooling;
  if (GRAPH_SUCCESS != op.GetAttr("global_pooling", global_pooling)) {
    std::string err_msg = GetInputInvalidErrMsg("global_pooling");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  bool ceil_mode;
  if (GRAPH_SUCCESS != op.GetAttr("ceil_mode", ceil_mode)) {
    std::string err_msg = GetInputInvalidErrMsg("ceil_mode");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if ((input_format == FORMAT_NCHW && data_format != "NCHW") ||
      (input_format == FORMAT_NHWC && data_format != "NHWC") ) {
    GE_OP_LOGE(op.GetName().c_str(), "input_format and data_format should be same.");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> input_dims = input_shape.GetDims();
  std::vector<std::pair<int64_t, int64_t>> input_range;
  if (IsUnknownRankShape(input_dims)) {
    OP_LOGW(op.GetName().c_str(), "the input os unknown rank, will set the input [-1, -1, -1, -1].");
    input_dims = {-1, -1, -1, -1};
  } else {
    input_desc->GetShapeRange(input_range);
  }
  MakeUpShapeRange(input_dims, input_range);

  std::vector<int64_t> output_dims;
  std::vector<std::pair<int64_t, int64_t>> output_range;

  auto input_h_dim = input_format == FORMAT_NHWC ? 1 : 2;
  auto input_w_dim = input_format == FORMAT_NHWC ? 2 : 3;
  auto strides_h_dim = data_format == "NHWC" ? 1 : 2;
  auto strides_w_dim = data_format == "NHWC" ? 2 : 3;

  if (global_pooling) {
    for (size_t i = 0; i < input_dims.size(); i++) {
      if (i == input_h_dim || i == input_w_dim) {
        output_dims.push_back(1);
        output_range.push_back({1, 1});
      } else {
        output_dims.push_back(input_dims[i]);
        output_range.push_back(input_range[i]);
      }
    }
  } else {
    if (strides[strides_h_dim] <= 0 || strides[strides_w_dim] <= 0) {
      OP_LOGE(op.GetName().c_str(), "strides h and w must greater than 0.");
      return GRAPH_FAILED;
    }
    if (padding_mode == "SAME" || padding_mode == "VALID") {
      ksize[strides_h_dim] = 1;
      ksize[strides_w_dim] = 1;
      for (size_t i = 0; i < input_dims.size(); i++) {
        int64_t dim_size = input_dims[i];
        auto dim_range = input_range[i];
        if (i == input_h_dim) {
          SAMEUpdateDimAndRange(ksize[strides_h_dim], strides[strides_h_dim], dim_size, dim_range);
        } else if (i == input_w_dim) {
          SAMEUpdateDimAndRange(ksize[strides_w_dim], strides[strides_w_dim], dim_size, dim_range);
        }
        output_dims.push_back(dim_size);
        output_range.push_back(dim_range);
      }
    } else {
      for (size_t i = 0; i < input_dims.size(); i++) {
        int64_t dim_size = input_dims[i];
        auto dim_range = input_range[i];
        if (i == input_h_dim) {
          CALCULATEUpdateDimAndRange(ksize[strides_h_dim], strides[strides_h_dim], ceil_mode,
                                     pad_top, pad_bottom, dim_size, dim_range);
        } else if (i == input_w_dim) {
          CALCULATEUpdateDimAndRange(ksize[strides_w_dim], strides[strides_w_dim], ceil_mode,
                                     pad_left, pad_right, dim_size, dim_range);
        }
        output_dims.push_back(dim_size);
        output_range.push_back(dim_range);
      }
    }
  }
  output_desc->SetShape(GeShape(output_dims));
  output_desc->SetShapeRange(output_range);
  return GRAPH_SUCCESS;
}

IMPLEMT_INFER_DATA_SLICE(MaxPoolV3, MaxPoolV3InferDataSlice) {
  // max_pool_v3 can cut n axis now
  OP_LOGD(TbeGetName(op), "Enter MaxPoolV3 InferDataSlice");
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto tensor_desc_y = op_desc->MutableOutputDesc("y");
  CHECK(!tensor_desc_y, OP_LOGI(TbeGetName(op), "get output desc failed."), return GRAPH_FAILED);
  auto tensor_desc_x = op_desc->MutableInputDesc("x");
  CHECK(!tensor_desc_x, OP_LOGI(TbeGetName(op), "get input desc failed."), return GRAPH_FAILED);
  vector<vector<int64_t>> y_data_slice;
  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(TbeGetName(op), "no data slice, not need infer input");
    return GRAPH_FAILED;
  }

  if (!y_data_slice.empty()) {
    if (!AttrUtils::SetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
      OP_LOGD(TbeGetName(op), "Set x_data slice failed!");
      return GRAPH_FAILED;
    }
    OP_LOGD(TbeGetName(op), "Calc MaxPoolV3 InferDataSlice end!");
    return GRAPH_SUCCESS;
  }

  OP_LOGD(TbeGetName(op), "Calc MaxPoolV3 InferDataSlice end!");
  return NO_OVERLAP_DIM;
}

COMMON_INFER_FUNC_REG(MaxPoolV3, MaxPoolV3InferShape);
VERIFY_FUNC_REG(MaxPoolV3, MaxPoolV3Verify);
INFER_DATA_SLICE_FUNC_REG(MaxPoolV3, MaxPoolV3InferDataSlice);
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
    std::string err_msg = OtherErrMsg("The shape of orig_input orig_output and grad must be same!");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
    if (data_format != "NCHW" && data_format != "NHWC") {
      string expected_format_list = ConcatString("NCHW, NHWC");
      std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }
  std::vector<int64_t> ksize;
  ksize = GetAttrValue(op, "ksize");
  if (!CheckListEmpty(op.GetName(), ksize, "ksize")) {
    std::string err_msg = OtherErrMsg("The ksize is empty!");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (ksize.size() != 4) {
    string excepted_size = ConcatString("4");
    std::string err_msg = GetAttrSizeErrMsg("ksize", std::to_string(ksize.size()), excepted_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (data_format == "NCHW" && (ksize[0] != 1 || ksize[1] != 1)) {
    string wrong_value = ConcatString(ksize[0], " and ", ksize[1]);
    std::string err_msg = GetAttrValueErrMsg("ksize[0] and ksize[1]", wrong_value, ConcatString("1"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (data_format == "NHWC" && (ksize[0] != 1 || ksize[3] != 1)) {
    string wrong_value = ConcatString(ksize[0], " and ", ksize[3]);
    std::string err_msg = GetAttrValueErrMsg("ksize[0] and ksize[3]", wrong_value, ConcatString("1"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), strides, "strides")) {
    std::string err_msg = OtherErrMsg("The strides is empty!");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (strides.size() != 4) {
    string excepted_size = ConcatString("4");
    std::string err_msg = GetAttrSizeErrMsg("strides", std::to_string(strides.size()), excepted_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (data_format == "NCHW" && (strides[0] != 1 || strides[1] != 1)) {
    string wrong_value = ConcatString(strides[0], " and ", strides[1]);
    std::string err_msg = GetAttrValueErrMsg("strides[0] and strides[1]", wrong_value, ConcatString("1"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (data_format == "NHWC" && (strides[0] != 1 || strides[3] != 1)) {
    string wrong_value = ConcatString(strides[0], " and ", strides[3]);
    std::string err_msg = GetAttrValueErrMsg("strides[0] and strides[3]", wrong_value, ConcatString("1"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::string padding_mode;
  if (ge::GRAPH_SUCCESS != op.GetAttr("padding_mode", padding_mode)) {
    std::string err_msg = OtherErrMsg("The padding_mode is empty!");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (padding_mode != "SAME" && padding_mode != "VALID" && padding_mode != "CALCULATED") {
    string excepted_value = ConcatString("SAME or VALID or CALCULATED");
    std::string err_msg = GetAttrValueErrMsg("padding_mode", padding_mode, excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<int64_t> pads;
  pads = GetAttrValue(op, "pads");
  if (!CheckListEmpty(op.GetName(), pads, "pads")) {
    std::string err_msg = OtherErrMsg("The pads is empty!");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (pads.size() != 4) {
    string excepted_size = ConcatString("4");
    std::string err_msg = GetAttrSizeErrMsg("pads", std::to_string(pads.size()), excepted_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
  OP_LOGI(op.GetName().c_str(),
          " AdaptiveMaxPool2d inferShape dims: [%u] ", dims_input.size());

  size_t index0 = dims_input.size() - 2;
  size_t index1 = dims_input.size() - 1;
  Format input_format = input_tensor_desc.GetFormat();
  OP_LOGI(op.GetName().c_str(), " AdaptiveMaxPool2d inferShape format: [%u] ", input_format);
  if ((input_format == FORMAT_NC1HWC0) || (input_format == FORMAT_NHWC)) {
    index0 = dims_input.size() - 3;
    index1 = dims_input.size() - 2;
  }
  dim_vector[index0] = ouput_size_list[0];
  dim_vector[index1] = ouput_size_list[1];

  TensorDesc td = op.GetOutputDesc("y");
  DataType input_dtype = input_tensor_desc.GetDataType();
  Shape output_shape(dim_vector);
  td.SetShape(output_shape);
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);

  TensorDesc out1_td = op.GetOutputDesc("argmax");
  DataType out1_dtype = out1_td.GetDataType();
  OP_LOGI(op.GetName().c_str(),  " AdaptiveMaxPool2d inferShape argmax dtype: [%u] ", out1_dtype);
  if (out1_dtype == DT_UNDEFINED) {
      out1_td.SetDataType(DT_INT32);
      OP_LOGI(op.GetName().c_str(), " AdaptiveMaxPool2d inferShape set argmax dtype: [%u] ", DT_INT32);
  }
  out1_td.SetShape(output_shape);
  (void)op.UpdateOutputDesc("argmax", out1_td);

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

IMPLEMT_VERIFIER(AdaptiveAvgPool2d, AdaptiveAvgPool2dVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AdaptiveAvgPool2d, AdaptiveAvgPool2dInferShape);
VERIFY_FUNC_REG(AdaptiveAvgPool2d, AdaptiveAvgPool2dVerify);
// ------------AdaptiveAvgPool2d Op End----------------

// ------------AdaptiveAvgPool2dGrad Op Begin----------------
IMPLEMT_INFERFUNC(AdaptiveAvgPool2dGrad, AdaptiveAvgPool2dGradInferShape) {
    // get orig_input_shape
    std::vector<int64_t> ori_shape;
    if (GRAPH_SUCCESS != op.GetAttr("orig_input_shape", ori_shape)) {
        OP_LOGE(op.GetName().c_str(), "GetOpAttr orig_input_shape failed!");
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
    std::vector<int64_t> dims_input = op.GetInputDesc("input_grad").GetShape().GetDims();
    std::vector<int64_t> ori_shape;
    if (GRAPH_SUCCESS != op.GetAttr("orig_input_shape", ori_shape)) {
        OP_LOGE(op.GetName().c_str(), "GetOpAttr orig_input_shape failed!");
        return GRAPH_FAILED;
    }
    if (dims_input.size() != ori_shape.size()) {
        OP_LOGE(op.GetName().c_str(), "The shape of grad and orig_input must be same!");
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
  auto op_desc_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc_x = op_desc_info->MutableInputDesc("x");
  auto output_desc_y = op_desc_info->MutableOutputDesc("y");
  vector<int64_t> x_shape = input_desc_x->MutableShape().GetDims();
  DataType input_dtype = input_desc_x->GetDataType();
  std::vector<std::pair<int64_t, int64_t>> x_range;
  input_desc_x->GetShapeRange(x_range);

  MakeUpShapeRange(x_shape, x_range);
  output_desc_y->SetShape(GeShape(x_shape));
  output_desc_y->SetShapeRange(x_range);
  output_desc_y->SetDataType(input_dtype);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(MaxPoolGradWithArgmaxV1,
                      MaxPoolGradWithArgmaxV1InferShape);
VERIFY_FUNC_REG(MaxPoolGradWithArgmaxV1, MaxPoolGradWithArgmaxV1Verify);
// ------------max_pool_grad_with_argmaxv1 Op End----------------

// ------------MaxPoolWithArgmaxV1 Op Begin----------------
struct MaxPoolWithArgmaxParam {
  int pad;
  int dilation;
  int kernel_size;
  int stride;
  bool ceil_mode;
};

int CalMax(const MaxPoolWithArgmaxParam &maxpool, int input_size) {
  const uint32_t G_DIM_C = 1;
  const uint32_t G_DIM_H = 2;
  int max_size = 0;
  int temp = 0;
  int pad = maxpool.pad;
  int dilation = maxpool.dilation;
  int kernel_size = maxpool.kernel_size;
  int stride = maxpool.stride;
  bool ceil_mode = maxpool.ceil_mode;
  if (stride == 0) {
     return 0;
  }
  temp = input_size + G_DIM_H * pad - dilation * (kernel_size - G_DIM_C) - G_DIM_C;
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

int CalMaskH(int max_h, int kernel_h, int kernel_w) {
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
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto output_max = op_info->MutableOutputDesc("y");
  auto output_mask = op_info->MutableOutputDesc("argmax");
  auto shape = input_desc->MutableShape();
  DataType input_dtype = input_desc->GetDataType();

  std::vector<int64_t> input_dims = shape.GetDims();
  std::vector<std::pair<int64_t, int64_t>> input_range;

  if (IsUnknownRankShape(input_dims)) {
      OP_LOGW(op.GetName().c_str(), "the input os unkown rank, will set the input [-1, -1, -1, -1].");
      input_dims = {-1, -1, -1, -1};
  } else {
      input_desc->GetShapeRange(input_range);
  }
  MakeUpShapeRange(input_dims, input_range);

  std::vector<int64_t> vec_max, vec_mask, vec_pads, vec_dilation, vec_kernel, vec_strides;
  std::vector<std::pair<int64_t, int64_t>> max_range, mask_range;
  std::pair<int64_t, int64_t> max_h_range_dim, mask_h_range_dim, max_w_range_dim, mask_w_range_dim;

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

  MaxPoolWithArgmaxParam maxpool_h = {static_cast<int>(vec_pads[G_DIM_C]), static_cast<int>(vec_dilation[G_DIM_C]),
  kernel_h, static_cast<int>(vec_strides[G_DIM_C]), ceil_mode};
  MaxPoolWithArgmaxParam maxpool_w = {static_cast<int>(vec_pads[G_DIM_H]), static_cast<int>(vec_dilation[G_DIM_H]),
  kernel_w, static_cast<int>(vec_strides[G_DIM_H]), ceil_mode};

  int max_h = -1;
  int max_w = -1;
  int mask_h = -1;
  int mask_w = -1;
  int max_h_range0 = 1;
  int max_h_range1 = -1;
  auto dim_h_range = input_range[2];
  auto dim_w_range = input_range[3];

  if (input_h != -1) {
      max_h = CalMax(maxpool_h, input_h);
      mask_h = CalMaskH(max_h, kernel_h, kernel_w);
      max_h_range_dim = std::pair<int64_t, int64_t>{max_h, max_h};
      mask_h_range_dim = std::pair<int64_t, int64_t>{mask_h, mask_h};
  } else {
      max_h_range0 = dim_h_range.first == 1 ? 1 : CalMax(maxpool_h, dim_h_range.first);
      max_h_range1 = dim_h_range.second == -1 ? -1 : CalMax(maxpool_h, dim_h_range.second);
      int mask_h_rang0 = dim_h_range.first == 1 ? 1 : CalMaskH(max_h_range0, kernel_h, kernel_w);
      int mask_h_rang1 = dim_h_range.second == -1 ? -1 : CalMaskH(max_h_range1, kernel_h, kernel_w);
      max_h_range_dim = std::pair<int64_t, int64_t>{max_h_range0, max_h_range1};
      mask_h_range_dim = std::pair<int64_t, int64_t>{mask_h_rang0, mask_h_rang1};
  }

  if (input_w != -1) {
      max_w = CalMax(maxpool_w, input_w);
      mask_w = CalMaskW(max_h, max_w, kernel_h, kernel_w, input_c0);
      max_w_range_dim = std::pair<int64_t, int64_t>{max_w, max_w};
      mask_w_range_dim = std::pair<int64_t, int64_t>{mask_w, mask_w};
  } else {
      int max_w_range0 = dim_w_range.first == 1 ? 1 : CalMax(maxpool_w, dim_w_range.first);
      int max_w_range1 = dim_w_range.second == -1 ? -1 : CalMax(maxpool_w, dim_w_range.second);
      int mask_w_rang0 = dim_w_range.first == 1 ? 1 : CalMaskW(max_h_range0, max_w_range0, kernel_h,
                                                               kernel_w, input_c0);
      int mask_w_rang1 = dim_w_range.second == -1 ? -1 : CalMaskW(max_h_range1, max_w_range1, kernel_h,
                                                                 kernel_w, input_c0);
      max_w_range_dim = std::pair<int64_t, int64_t>{max_w_range0, max_w_range1};
      mask_w_range_dim = std::pair<int64_t, int64_t>{mask_w_rang0, mask_w_rang1};
  }

  vec_max.push_back(batch_size);
  vec_max.push_back(c1_size);
  vec_max.push_back(max_h);
  vec_max.push_back(max_w);
  vec_mask.push_back(batch_size);
  vec_mask.push_back(c1_size);
  vec_mask.push_back(mask_h);
  vec_mask.push_back(mask_w);
  max_range.push_back(input_range[0]);
  max_range.push_back(input_range[1]);
  max_range.push_back(max_h_range_dim);
  max_range.push_back(max_w_range_dim);
  mask_range.push_back(input_range[0]);
  mask_range.push_back(input_range[1]);
  mask_range.push_back(mask_h_range_dim);
  mask_range.push_back(mask_w_range_dim);

  OP_LOGD(op.GetName().c_str(), "max_shape[%d,%d]", vec_max[2], vec_max[3]);
  OP_LOGD(op.GetName().c_str(), "mask_shape[%d,%d]", vec_mask[2], vec_mask[3]);
  OP_LOGD(op.GetName().c_str(), "max_range[%d,%d]", max_h_range_dim.first, max_h_range_dim.second);
  OP_LOGD(op.GetName().c_str(), "max_range[%d,%d]", max_w_range_dim.first, max_w_range_dim.second);
  OP_LOGD(op.GetName().c_str(), "mask_range[%d,%d]", mask_h_range_dim.first, mask_h_range_dim.second);
  OP_LOGD(op.GetName().c_str(), "mask_range[%d,%d]", mask_w_range_dim.first, mask_w_range_dim.second);

  output_max->SetShape(GeShape(vec_max));
  output_max->SetShapeRange(max_range);
  output_max->SetDataType(input_dtype);
  output_mask->SetShape(GeShape(vec_mask));
  output_mask->SetShapeRange(mask_range);
  output_mask->SetDataType(DT_UINT16);
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(MaxPoolWithArgmaxV1, MaxPoolWithArgmaxV1InferShape);

// Registered verify function
VERIFY_FUNC_REG(MaxPoolWithArgmaxV1, MaxPoolWithArgmaxV1Verify);
// ------------MaxPoolWithArgmaxV1 Op End----------------
// ----------------SubSample begin-------------------
IMPLEMT_COMMON_INFERFUNC(SubSampleInferShape) {
  OP_LOGI("SubSample", " SubSample inferShape begin!");
  std::vector<int64_t> labels_shape = op.GetInputDesc("labels").GetShape().GetDims();
  std::vector<int64_t> output_shape;
  output_shape.push_back(labels_shape[0]);
  // update output info
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(ge::Shape(output_shape));
  output_desc.SetDataType(ge::DT_INT32);
  output_desc.SetOriginFormat(ge::FORMAT_ND);
  output_desc.SetFormat(ge::FORMAT_ND);
  (void)op.UpdateOutputDesc("y", output_desc);
  OP_LOGI("SubSample", " SubSample inferShape end!");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SubSample, SubSampleVerify) {
  return GRAPH_SUCCESS;
}
// Registered inferfunction
COMMON_INFER_FUNC_REG(SubSample, SubSampleInferShape);
// Registered verify function
VERIFY_FUNC_REG(SubSample, SubSampleVerify);
// ----------------SubSample end-------------------
// ----------------SubSampleLabels begin-------------------
IMPLEMT_COMMON_INFERFUNC(SubSampleLabelsInferShape) {
  OP_LOGI("SubSampleLabels", " SubSampleLabels inferShape begin!");
  std::vector<int64_t> sub_labels_shape = op.GetInputDesc("labels").GetShape().GetDims();
  std::vector<int64_t> sub_output_shape;
  sub_output_shape.push_back(sub_labels_shape[0]);
  // update output info
  TensorDesc sub_output_desc = op.GetOutputDesc("y");
  sub_output_desc.SetShape(ge::Shape(sub_output_shape));
  sub_output_desc.SetDataType(ge::DT_INT32);
  sub_output_desc.SetOriginFormat(ge::FORMAT_ND);
  sub_output_desc.SetFormat(ge::FORMAT_ND);
  (void)op.UpdateOutputDesc("y", sub_output_desc);
  OP_LOGI("SubSampleLabels", " SubSampleLabels inferShape end!");
  return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(SubSampleLabels, SubSampleLabelsVerify) {
  return GRAPH_SUCCESS;
}
// Registered inferfunction
COMMON_INFER_FUNC_REG(SubSampleLabels, SubSampleLabelsInferShape);
// Registered verify function
VERIFY_FUNC_REG(SubSampleLabels, SubSampleLabelsVerify);
// ----------------SubSampleLabels end-------------------
// ----------------GlobalLpPool start-------------------
IMPLEMT_COMMON_INFERFUNC(GlobalLpPoolInfer) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto output = op_info->MutableOutputDesc("y");
  auto shape = input_desc->MutableShape();
  DataType input_dtype = input_desc->GetDataType();

  std::vector<int64_t> input_dims = shape.GetDims();
  std::vector<std::pair<int64_t, int64_t>> input_range;

  input_desc->GetShapeRange(input_range);
  MakeUpShapeRange(input_dims, input_range);

  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_range;
  std::pair<int64_t, int64_t> range_dim;
  range_dim = std::pair<int64_t, int64_t>{1, 1};

  int batch_size = shape.GetDim(0);
  int c_size = shape.GetDim(1);

  output_shape.push_back(batch_size);
  output_shape.push_back(c_size);
  output_range.push_back(input_range[0]);
  output_range.push_back(input_range[1]);

  if (input_dims.size() == 4) {
    output_shape.push_back(1);
    output_shape.push_back(1);
    output_range.push_back(range_dim);
    output_range.push_back(range_dim);
  } else {
    output_shape.push_back(1);
    output_shape.push_back(1);
    output_shape.push_back(1);
    output_range.push_back(range_dim);
    output_range.push_back(range_dim);
    output_range.push_back(range_dim);
  }

  output->SetShape(GeShape(output_shape));
  output->SetShapeRange(output_range);
  output->SetDataType(input_dtype);

  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(GlobalLpPool, GlobalLpPoolInfer);
// ----------------GlobalLpPool end------------------

//-----------------GlobalAveragePool start----------------
IMPLEMT_INFERFUNC(GlobalAveragePool, GlobalAveragePoolInferShape) {
  TensorDesc input_desc = op.GetInputDesc("x");
  TensorDesc output_desc = op.GetOutputDesc("y");

  int64_t num_shape = 1;
  vector<int64_t> output_shape;
  DataType input_dtype = input_desc.GetDataType();
  Format input_format = input_desc.GetFormat();
  vector<int64_t> input_shape = input_desc.GetShape().GetDims();
  int64_t x_shape = input_desc.GetShape().GetDims().size();
  output_shape.push_back(input_shape[0]);
  output_shape.push_back(input_shape[1]);

  if (x_shape == 5 && (input_format == FORMAT_NCDHW || input_format == FORMAT_ND)) {
    output_shape.push_back(num_shape);
    output_shape.push_back(num_shape);
    output_shape.push_back(num_shape);
    if (input_format == FORMAT_ND) {
      OP_LOGW(op.GetName().c_str(), "input format is ND, but use format is NCDHW");
    }
  } else if (x_shape == 4 && (input_format == FORMAT_NCHW || input_format == FORMAT_ND)) {
    output_shape.push_back(num_shape);
    output_shape.push_back(num_shape);
    if (input_format == FORMAT_ND) {
      OP_LOGW(op.GetName().c_str(), "input format is ND, but use format is NCHW");
    }
  } else if (x_shape == 3 && input_format == FORMAT_ND) {
    output_shape.push_back(num_shape);
  } else {
    OP_LOGE(op.GetName().c_str(), "x_shape or format is error");
    return GRAPH_FAILED;
  }

  output_desc.SetShape(ge::Shape(output_shape));
  output_desc.SetDataType(input_dtype);
  output_desc.SetFormat(input_format);

  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(GlobalAveragePool, GlobalAveragePoolVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(GlobalAveragePool, GlobalAveragePoolInferShape);
VERIFY_FUNC_REG(GlobalAveragePool, GlobalAveragePoolVerify);
//-----------------GlobalAveragePool end----------------
}  // namespace ge
