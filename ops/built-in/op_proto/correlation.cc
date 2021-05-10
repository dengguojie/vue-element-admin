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
 * \file correlation.cpp
 * \brief
 */

#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include "inc/correlation.h"

#define CHECK_FORMAT(format)  \
{                                         \
  if (ge::FORMAT_RESERVED == format) {    \
    return false;     \
  }                     \
}

namespace ge
{
//--------------------------Correlation------------------------------
/*
 * Verify the required 2 input tensor, optional bias ignored
 * Verify groups attr
*/
IMPLEMT_VERIFIER(Correlation, CorrelationVerify) {
  auto x_tensor = op.get_input_desc_x();
  auto w_tensor = op.get_input_desc_filter();
  auto x_shape = x_tensor.GetShape().GetDims();
  auto w_shape = w_tensor.GetShape().GetDims();

  if (x_shape.size() != 4) {
    return GRAPH_FAILED;
  }
  if (w_shape.size() != 4) {
    return GRAPH_FAILED;
  }

  auto x_format = x_tensor.GetFormat();
  auto w_format  = w_tensor.GetFormat();
  CHECK_FORMAT(x_format);
  CHECK_FORMAT(w_format);

  int x_batchnum = 0;
  int w_batchnum = 0;
  int x_channel_num = 0;
  int w_channel_num = 0;

  if (w_format == FORMAT_NCHW) {
    x_batchnum = x_shape[0];
    x_channel_num = x_shape[1];
    w_batchnum = w_shape[0];
    w_channel_num = w_shape[1];
  } else if (w_format == FORMAT_NHWC) {
    x_batchnum = x_shape[0];
    x_channel_num = x_shape[3];
    w_batchnum = w_shape[0];
    w_channel_num = w_shape[3];
  } else if (w_format == FORMAT_HWCN) {
    x_batchnum = x_shape[3];
    x_channel_num = x_shape[2];
    w_batchnum = w_shape[3];
    w_channel_num = w_shape[2];
  } else {
    return GRAPH_FAILED;
  }

  if (x_batchnum != w_batchnum) {
    return GRAPH_FAILED;
  }

  if (x_channel_num != w_channel_num) {
    return GRAPH_FAILED;
  }

  auto x_datatype = x_tensor.GetDataType();
  auto w_datatype = w_tensor.GetDataType();

  if (x_datatype != w_datatype) {
    return GRAPH_FAILED;
  }

  int64_t groups = 1;
  if (GRAPH_SUCCESS != op.GetAttr("groups", groups)) {
    return GRAPH_FAILED;
  }
  if (groups != 1 && groups != x_channel_num) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(Correlation, CorrelationInfer) {
  auto x_tensor = op.get_input_desc_x();
  auto w_tensor = op.get_input_desc_filter();
  auto x_shape = x_tensor.GetShape().GetDims();
  auto w_shape = w_tensor.GetShape().GetDims();
  auto x_format = x_tensor.GetFormat();
  auto w_format  = w_tensor.GetFormat();
  CHECK_FORMAT(x_format);
  CHECK_FORMAT(w_format);

  int32_t in = 0;
  int32_t ic = 0;
  int32_t ih = 0;
  int32_t iw = 0;
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
    return GRAPH_FAILED;
  }

  if (w_format == FORMAT_NCHW) {
    kc = w_shape[1];
    kh = w_shape[2];
    kw = w_shape[3];
  } else if (w_format == FORMAT_NHWC) {
    kc = w_shape[3];
    kh = w_shape[1];
    kw = w_shape[2];
  } else if (w_format == FORMAT_HWCN) {
    kc = w_shape[2];
    kh = w_shape[0];
    kw = w_shape[1];
  } else {
    return GRAPH_FAILED;
  }

  int64_t groups = 1;
  if (GRAPH_SUCCESS != op.GetAttr("groups", groups)) {
    return GRAPH_FAILED;
  }
  int32_t strh = 1;
  int32_t strw = 1;
  int32_t dilh = 1;
  int32_t dilw = 1;
  int32_t padt = 0;
  int32_t padb = 0;
  int32_t padl = 0;
  int32_t padr = 0;
  int64_t oh = (ih + padt + padb - dilh * (kh - 1) - 1) / strh + 1;
  int64_t ow = (iw + padl + padr - dilw * (kw - 1) - 1) / strw + 1;
  int64_t oc = 1;
  if (groups == 1) {
    oc = kc / ic;
  } else {
    oc = kc;
  }

  vector<int64_t> vec_y_shape;
  auto y_tensor = op.GetOutputDesc(0);
  auto y_format = y_tensor.GetFormat();
  CHECK_FORMAT(y_format);
  if (y_format == FORMAT_NCHW) {
    vec_y_shape.push_back(in);
    vec_y_shape.push_back(oc);
    vec_y_shape.push_back(oh);
    vec_y_shape.push_back(ow);
  } else if (y_format == FORMAT_NHWC) {
    vec_y_shape.push_back(in);
    vec_y_shape.push_back(oh);
    vec_y_shape.push_back(ow);
    vec_y_shape.push_back(oc);
  } else {
    return GRAPH_FAILED;
  }
  y_tensor.SetShape(Shape(vec_y_shape));
  auto x_datatype = x_tensor.GetDataType();
  if (x_datatype == ge::DT_INT8) {
    y_tensor.SetDataType(ge::DT_INT32);
  }else{
    y_tensor.SetDataType(x_datatype);
  }
  if (GRAPH_SUCCESS != op.update_output_desc_y(y_tensor)) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Correlation, CorrelationInfer);
VERIFY_FUNC_REG(Correlation, CorrelationVerify);

} // namespace ge
