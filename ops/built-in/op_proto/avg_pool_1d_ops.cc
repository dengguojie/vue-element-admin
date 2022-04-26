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
 * \file avg_pool_1d.cc
 * \brief
 */
/* reslove the complexity of pooling fuction. */
#include "inc/avg_pool_1d_ops.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "common/util/error_manager/error_manager.h"
#include "graph/common_error_codes.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/operator.h"
#include "inc/nn_pooling_ops.h"
#include "op_log.h"
#include "register/infer_data_slice_registry.h"
#include "util/common_shape_fns.h"
#include "util/error_util.h"
#include "util/util.h"

namespace ge {
IMPLEMT_INFERFUNC(AvgPool1DAvgMatrix, AvgPool1DAvgMatrixInfer) {
  TensorDesc output_tensor_desc = op.GetOutputDesc("y");
  auto input_tensor = op.GetInputDesc("x");
  Format input_format = input_tensor.GetFormat();
  auto input_shape = input_tensor.GetShape();
  int64_t input_w_size = 0;
  Shape output_shape;
  vector<int64_t> dim_vec;
  DataType input_type = input_tensor.GetDataType();
  if (input_format == FORMAT_NHWC) {
    input_w_size = input_shape.GetDim(2);
  } else if (input_format == FORMAT_NCHW) {
    input_w_size = input_shape.GetDim(3);
  } else {
      OP_LOGE(TbeGetName(op).c_str(), "Input format only support NCHW or NHWC");
      return GRAPH_FAILED;
  }
  //dim_w  not equal to  ge::UNKNOWN_DIM
  if (input_w_size != ge::UNKNOWN_DIM) {
    int32_t ksize = 0;
    int32_t strides = 1;
    bool ceil_mode = false;
    if (op.GetAttr("ksize", ksize) != GRAPH_SUCCESS) {
      OP_LOGE(TbeGetName(op).c_str(), "GetOpAttr ksize failed");
      return GRAPH_FAILED;
    }
    if (op.GetAttr("strides", strides) != GRAPH_SUCCESS) {
      OP_LOGE(TbeGetName(op).c_str(), "GetOpAttr strides failed");
      return GRAPH_FAILED;
    }
    if (strides == 0) {
      OP_LOGE(TbeGetName(op).c_str(), "Value of strides should not be 0");
      return GRAPH_FAILED;
    }
    // get input ksize
    std::vector<int32_t> pads_list;
    if (op.GetAttr("pads", pads_list) != GRAPH_SUCCESS) {
      OP_LOGE(TbeGetName(op).c_str(), "GetOpAttr pads_list failed!");
      return GRAPH_FAILED;
    }
    if (op.GetAttr("ceil_mode", ceil_mode) != GRAPH_SUCCESS) {
      OP_LOGE(TbeGetName(op).c_str(), "GetOpAttr ceil_mode failed");
      return GRAPH_FAILED;
    }
    if (pads_list.size() < 2) {
      OP_LOGE(TbeGetName(op).c_str(), "Size of pads_list must greater than 1!");
      return GRAPH_FAILED;
    }
    int32_t padl = pads_list[0];
    int32_t padr = pads_list[1];
    int32_t output_w_size = 0;
    if (ceil_mode) {
      output_w_size =
          (input_w_size + padl + padr - ksize + strides - 1) / strides + 1;
    } else {
      output_w_size = ((input_w_size + padl + padr) - ksize) / strides + 1;
    }
    if (padl) {
      if (((static_cast<int64_t>(output_w_size) - 1) *
           static_cast<int64_t>(strides)) >=
          (input_w_size + static_cast<int64_t>(padl))) {
        output_w_size--;
      }
    }
    padr = (output_w_size - 1) * strides + ksize - input_w_size - padl;

    if (input_format != FORMAT_NHWC && input_format != FORMAT_NCHW) {
      OP_LOGE(TbeGetName(op).c_str(), "Input format only support NCHW or NHWC "
      ", input format is [%d]", input_format);
      return GRAPH_FAILED;
    }
    if (input_format == FORMAT_NHWC) {
      dim_vec.push_back(1);
      dim_vec.push_back(1);
      dim_vec.push_back(output_w_size);
      dim_vec.push_back(16);
    } else if (input_format == FORMAT_NCHW) {
      dim_vec.push_back(1);
      dim_vec.push_back(16);
      dim_vec.push_back(1);
      dim_vec.push_back(output_w_size);
    }
    output_shape = Shape(dim_vec);
  } else {
  //dims_w is ge::UNKNOWN_DIM
    if (input_format == FORMAT_NHWC) {
      dim_vec.push_back(1);
      dim_vec.push_back(1);
      dim_vec.push_back(ge::UNKNOWN_DIM);
      dim_vec.push_back(16);
    } else if (input_format == FORMAT_NCHW) {
      dim_vec.push_back(1);
      dim_vec.push_back(16);
      dim_vec.push_back(1);
      dim_vec.push_back(ge::UNKNOWN_DIM);
    }
    output_shape = Shape(dim_vec);
  }
  if (!ShapeFullDefined(output_shape)) {
    std::vector<std::pair<int64_t, int64_t>> y_range;
    for (const int64_t& y_dim : dim_vec) {
      y_range.push_back(y_dim == UNKNOWN_DIM ? std::pair<int64_t, int64_t>{1, -1} :
                                               std::pair<int64_t, int64_t>{y_dim, y_dim});
    }
    output_tensor_desc.SetShapeRange(y_range);
  }
  DataType output_dtype = input_type;
  output_tensor_desc.SetShape(output_shape);
  output_tensor_desc.SetDataType(output_dtype);
  if (op.UpdateOutputDesc("y", output_tensor_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(),
            "UpdateOutputDesc failed. Need check whether the names of outputs "
            "are matched.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(AvgPool1DAvgMatrix, AvgPool1DAvgMatrixInfer);
}  // namespace ge