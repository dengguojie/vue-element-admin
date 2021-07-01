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
 * \file matrix_calculation_ops.cpp
 * \brief
 */
#include "inc/matrix_calculation_ops.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <sstream>
#include <utility>
#include <vector>

#include "util/util.h"
#include "util/common_shape_fns.h"
#include "util/array_ops_shape_fns.h"
#include "util/error_util.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/common_error_codes.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common/util/error_manager/error_manager.h"
#include "op_log.h"
#include "register/infer_data_slice_registry.h"

using namespace std;

namespace ge {
// ----------------FullyConnection-------------------

bool InferFC5HD(vector<vector<int64_t>>& x_data_slice, vector<vector<int64_t>>& w_data_slice,
                vector<vector<int64_t>>& y_data_slice, int32_t& infer_x, int32_t& infer_w) {
  for (int i = 0; i < y_data_slice.size(); i++) {
    if (y_data_slice[i].size() > 0) {
      if (i == 0) {
        x_data_slice[i] = y_data_slice[i];
        infer_x = 1;
      } else if (i == 1) {
        w_data_slice[i] = y_data_slice[i];
        infer_w = 1;
      }
    }
  }
}

bool InferFC5HD2NZ(vector<vector<int64_t>>& x_data_slice, vector<vector<int64_t>>& w_data_slice,
                   vector<vector<int64_t>>& y_data_slice, int32_t& infer_x, int32_t& infer_w,
                   const vector<int64_t>& x_shape) {
  for (int i = 0; i < y_data_slice.size(); i++) {
    if (y_data_slice[i].size() > 0) {
      if (i == 0) {
        w_data_slice[1] = y_data_slice[i];
        infer_w = 1;
      } else if (i == 1 && y_data_slice[i].size() == 2) {
        int64_t m_start = y_data_slice[i][0] * 16;
        int64_t m_end = std::min(y_data_slice[i][1]*16 + 15, x_shape[0] - 1);
        x_data_slice[0] = {m_start, m_end};
        infer_x = 1;
      }
    }
  }
}

bool InferFCNZ(vector<vector<int64_t>>& x_data_slice, vector<vector<int64_t>>& w_data_slice,
              vector<vector<int64_t>>& y_data_slice, int32_t& infer_x, int32_t& infer_w,
              const int64_t axis) {
  for (int i = 0; i < y_data_slice.size(); i++) {
    if (y_data_slice[i].size() > 0) {
      if (axis == 2) {
        if (i == 0 || i == 2){
          x_data_slice[i] = y_data_slice[i];
          infer_x = 1;
        } else if (i == 1) {
          w_data_slice[i] = y_data_slice[i];
          infer_w = 1;
        }
      } else {
        if (i == 0) {
          w_data_slice[1] = y_data_slice[i];
          infer_w = 1;
        } else if (i == 1) {
          x_data_slice[i] = y_data_slice[i];
          infer_x = 1;
        }
      }
    }
  }
}


bool InferFullyConnectionDataSlice(ge::Operator& op) {
  auto x_tensor = op.GetInputDesc("x");
  auto w_tensor = op.GetInputDesc("w");
  auto y_tensor = op.GetOutputDesc("y");

  auto x_shape = x_tensor.GetShape().GetDims();
  auto w_shape = w_tensor.GetShape().GetDims();
  auto x_format = x_tensor.GetFormat();
  auto y_format = y_tensor.GetFormat();

  int64_t num_output;
  int64_t axis;
  if (GRAPH_SUCCESS != op.GetAttr("num_output", num_output) || GRAPH_SUCCESS != op.GetAttr("axis", axis)){
    return false;
  }
  vector<vector<int64_t>> x_data_slice;
  if (x_format == FORMAT_NC1HWC0 || axis == 2) {
    x_data_slice = {{}, {}, {}, {}, {}};
  } else {
    x_data_slice = {{}, {}, {}, {}};
  }
  vector<vector<int64_t>> w_data_slice = {{}, {}, {}, {}};

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
  GeTensorDescPtr tensor_desc_w = op_desc->MutableInputDesc("w");
  vector<vector<int64_t>> y_data_slice;
  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op.GetName().c_str(), "no data slice, not need infer input");
    return false;
  }

  int32_t infer_x = 0;
  int32_t infer_w = 0;
  if (y_format == FORMAT_NC1HWC0) {
    OP_LOGI(op.GetName().c_str(), "infer dataslice from 5HD to 5HD");
    InferFC5HD(x_data_slice, w_data_slice, y_data_slice, infer_x, infer_w);
  } else if (x_format == FORMAT_NC1HWC0) {
    OP_LOGI(op.GetName().c_str(), "infer dataslice from 5HD to NZ");
    InferFC5HD2NZ(x_data_slice, w_data_slice, y_data_slice, infer_x, infer_w, x_shape);
  } else {
    OP_LOGI(op.GetName().c_str(), "infer dataslice from NZ to NZ");
    InferFCNZ(x_data_slice, w_data_slice, y_data_slice, infer_x, infer_w, axis);
  }

  if (infer_x == 0 && infer_w == 0) {
    OP_LOGI(op.GetName().c_str(), "no data slice, not need infer input");
    return false;
  }

  if (infer_x == 1) {
    if(!AttrUtils::SetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
      return false;
    }
    OP_LOGI(op.GetName().c_str(), "infer input x success");
  }
  if (infer_w == 1) {
    if(!AttrUtils::SetListListInt(tensor_desc_w, ge::ATTR_NAME_DATA_SLICE, w_data_slice)) {
      return false;
    }
    num_output = (w_data_slice[1][1] - w_data_slice[1][0] + 1) * w_shape[2];
    op.SetAttr("num_output", num_output);
    OP_LOGI(op.GetName().c_str(), "infer input w success");
  }
  return true;
}

IMPLEMT_VERIFIER(FullyConnection, FullyConnectionVerify) {
  auto xShape = op.get_input_desc_x().GetShape().GetDims();
  auto wShape = op.get_input_desc_w().GetShape().GetDims();
  bool transpose = op.get_attr_transpose();
  int xDim = xShape.size();
  int axis = op.get_attr_axis();
  int axis_new;

  if (axis < 0) {
    axis_new = axis + xDim;
  } else {
    axis_new = axis;
  }

  // check axis
  if (axis_new != 1 && axis_new != 2) {
    std::string err_msg = OtherErrMsg(ConcatString("Attr axis is wrong, the original value of axis ",axis," is not supported."));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int kShape = 1;
  int reduceStart;
  if (axis_new == 2) {
    reduceStart = 2;
  } else {
    reduceStart = 1;
  }
  for (int i = reduceStart; i < xDim; i++) {
    kShape *= xShape[i];
  }

  // check wShape size
  if (wShape.size() != 2) {
    std::string err_msg = GetAttrValueErrMsg("wShape.size()", std::to_string(wShape.size()), "2");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // check km and kn shape
  if (wShape.size() == 2) {
    if (!transpose) {
      if (kShape != wShape[1]) {
        string err_msg1 = ConcatString("weight K must equal to input K! kShape:",kShape, ", wShape[1]:",wShape[1]);
        std::string err_msg = OtherErrMsg(err_msg1);
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
        return GRAPH_FAILED;
      }
    } else {
      if (kShape != wShape[0]) {
        string err_msg1 = ConcatString("weight K must equal to input K! kShape:",kShape, ", wShape[0]:",wShape[0]);
        std::string err_msg = OtherErrMsg(err_msg1);
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
        return GRAPH_FAILED;
      }
    }
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(FullyConnection, FullyConnectionInfer) {
  auto outDesc = op.GetOutputDesc("y");
  auto weightDesc = op.GetInputDesc("w");
  auto xDesc = op.GetInputDesc("x");

  auto xShape = op.GetInputDesc("x").GetShape().GetDims();
  auto wShape = op.GetInputDesc("w").GetShape().GetDims();
  auto xDtype = op.GetInputDesc("x").GetDataType();
  bool transpose = op.get_attr_transpose();
  int axis = op.get_attr_axis();
  int xDim = xShape.size();
  int axis_new;

  if (axis < 0) {
    axis_new = axis + xDim;
  } else {
    axis_new = axis;
  }
  if (axis_new != 1 && axis_new != 2) {
    std::string err_msg = GetAttrValueErrMsg("axis_new", std::to_string(axis_new), "1 or 2");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  op.SetAttr("axis", axis_new);

  vector<int64_t> changedWeightShape;
  vector<int64_t> yShape;
  vector<int64_t> changedXShape;

  if (axis_new == 2) {
    if (yShape.empty()) {
      yShape.push_back(xShape[0]);
      yShape.push_back(xShape[1]);
    }
    changedXShape.push_back(xShape[0]);
    changedXShape.push_back(xShape[1]);
    if (xShape.size() >= 3) {
      int km_shape = 1;
      for (int i = 2; i < xDim; i++) {
        km_shape *= xShape[i];
      }
      changedXShape.push_back(km_shape);
    } else {
      CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Not enough info about M and K!\n");
      return GRAPH_FAILED;
    }

    if (!transpose) {
      changedWeightShape.push_back(wShape[0]);
      changedWeightShape.push_back(changedXShape[2]);
      changedWeightShape.push_back(1);
      changedWeightShape.push_back(1);
      yShape.push_back(wShape[0]);
    } else {
      changedWeightShape.push_back(changedXShape[2]);
      changedWeightShape.push_back(1);
      changedWeightShape.push_back(1);
      changedWeightShape.push_back(wShape[1]);
      yShape.push_back(wShape[1]);
      weightDesc.SetFormat(ge::FORMAT_CHWN);
      weightDesc.SetOriginFormat(ge::FORMAT_CHWN);
    }

    xDesc.SetShape(ge::Shape(changedXShape));
    xDesc.SetOriginShape(ge::Shape(changedXShape));
    (void)op.UpdateInputDesc("x", xDesc);
  } else {
    if (yShape.empty()) {
      yShape.push_back(xShape[0]);
    }
    if (xShape.size() == 2) {
      xShape.push_back(1);
      xShape.push_back(1);
    } else if (xShape.size() == 3) {
      xShape.push_back(1);
    }

    if (!transpose) {
      changedWeightShape.push_back(wShape[0]);
      changedWeightShape.push_back(xShape[1]);
      changedWeightShape.push_back(xShape[2]);
      changedWeightShape.push_back(xShape[3]);
      yShape.push_back(wShape[0]);
    } else {
      changedWeightShape.push_back(xShape[1]);
      changedWeightShape.push_back(xShape[2]);
      changedWeightShape.push_back(xShape[3]);
      changedWeightShape.push_back(wShape[1]);
      yShape.push_back(wShape[1]);
      weightDesc.SetFormat(ge::FORMAT_CHWN);
      weightDesc.SetOriginFormat(ge::FORMAT_CHWN);
    }

    if (xShape[0] == 1) {
      yShape.push_back(1);
      yShape.push_back(1);
    }
  }

  weightDesc.SetShape(ge::Shape(changedWeightShape));
  weightDesc.SetOriginShape(ge::Shape(changedWeightShape));
  (void)op.UpdateInputDesc("w", weightDesc);

  outDesc.SetShape(ge::Shape(yShape));
  if (xDtype == ge::DT_INT8) {
    outDesc.SetDataType(ge::DT_INT32);
  } else {
    outDesc.SetDataType(ge::DataType(xDtype));
  }
  (void)op.UpdateOutputDesc("y", outDesc);

  return GRAPH_SUCCESS;
}

IMPLEMT_INFER_DATA_SLICE(FullyConnection, FullyConnectionInferDataSlice) {
  OP_LOGD(op.GetName().c_str(), "Enter FullyConnection InferDataSlice");
  if (InferFullyConnectionDataSlice(op)) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

INFER_FUNC_REG(FullyConnection, FullyConnectionInfer);
VERIFY_FUNC_REG(FullyConnection, FullyConnectionVerify);
INFER_DATA_SLICE_FUNC_REG(FullyConnection, FullyConnectionInferDataSlice);

// ----------------FullyConnectionCompress-------------------
IMPLEMT_VERIFIER(FullyConnectionCompress, FullyConnectionCompressVerify) {
  auto xShape = op.get_input_desc_x().GetShape().GetDims();
  auto wShape = op.get_input_desc_w().GetShape().GetDims();
  bool transpose = op.get_attr_transpose();
  int xDim = xShape.size();

  int kShape = 1;
  for (int i = 1; i < xDim; i++) {
    kShape *= xShape[i];
  }

  // check wShape size
  if (wShape.size() != 1 && wShape.size() != 2) {
    OpsOneOutputShapeErrReport(op.GetName(), "W shape", "wShape Compress size must equal to 1 or 2!");
    OP_LOGE(op.GetName().c_str(), "wShape Compress size must equal to 1 or 2!\n");
    return GRAPH_FAILED;
  }

  // check km and kn shape
  if (wShape.size() == 2) {
    if (!transpose) {
      if (kShape != wShape[1]) {
        OpsTwoInputShapeErrReport(op.GetName(), "X Shape", "W Shape", "weight Compress K must equal to input K!");
        OP_LOGE(op.GetName().c_str(), "weight Compress K must equal to input K!\n");
        return GRAPH_FAILED;
      }
    } else {
      if (kShape != wShape[0]) {
        OpsTwoInputShapeErrReport(op.GetName(), "X Shape", "W Shape", "weight Compress K must equal to input K!");
        OP_LOGE(op.GetName().c_str(), "weight Compress K must equal to input K!\n");
        return GRAPH_FAILED;
      }
    }
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(FullyConnectionCompress, FullyConnectionCompressInfer) {
  auto outDesc = op.GetOutputDesc("y");
  auto weightDesc = op.GetInputDesc("w");

  auto xShape = op.GetInputDesc("x").GetShape().GetDims();
  auto wShape = op.GetInputDesc("w").GetShape().GetDims();
  auto xDtype = op.GetInputDesc("x").GetDataType();
  bool transpose = op.get_attr_transpose();

  if (xShape.size() < 1 || wShape.size() < 1) {
    OpsTwoInputShapeErrReport(op.GetName(), "X Shape", "W Shape", "Input Shape or Weight Shape should >= 1!");
    OP_LOGE(op.GetName().c_str(), "Invalid Shape size, xShape size is %u, wShape size is %u.", xShape.size(),
            wShape.size());
    return GRAPH_FAILED;
  }

  vector<int64_t> changedWeightShape;
  vector<int64_t> yShape;
  if (yShape.empty()) {
    yShape.push_back(xShape[0]);
  }

  if (xShape.size() == 2) {
    xShape.push_back(1);
    xShape.push_back(1);
  } else if (xShape.size() == 3) {
    xShape.push_back(1);
  }
  if (!transpose) {
    changedWeightShape.push_back(wShape[0]);
    changedWeightShape.push_back(xShape[1]);
    changedWeightShape.push_back(xShape[2]);
    changedWeightShape.push_back(xShape[3]);
    yShape.push_back(wShape[0]);
  } else {
    changedWeightShape.push_back(xShape[1]);
    changedWeightShape.push_back(xShape[2]);
    changedWeightShape.push_back(xShape[3]);
    changedWeightShape.push_back(wShape[1]);
    yShape.push_back(wShape[1]);
    weightDesc.SetFormat(ge::FORMAT_CHWN);
    weightDesc.SetOriginFormat(ge::FORMAT_CHWN);
  }

  weightDesc.SetShape(ge::Shape(changedWeightShape));
  weightDesc.SetOriginShape(ge::Shape(changedWeightShape));
  (void)op.UpdateInputDesc("w", weightDesc);

  outDesc.SetShape(ge::Shape(yShape));
  if (xDtype == ge::DT_INT8) {
    outDesc.SetDataType(ge::DT_INT32);
  } else {
    outDesc.SetDataType(ge::DataType(xDtype));
  }
  (void)op.UpdateOutputDesc("y", outDesc);

  return GRAPH_SUCCESS;
}

IMPLEMT_INFER_DATA_SLICE(FullyConnectionCompress, FullyConnectionCompressInferDataSlice) {
  OP_LOGD(op.GetName().c_str(), "Enter FullyConnectionCompress InferDataSlice");
  if (InferFullyConnectionDataSlice(op)) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

INFER_FUNC_REG(FullyConnectionCompress, FullyConnectionCompressInfer);

VERIFY_FUNC_REG(FullyConnectionCompress, FullyConnectionCompressVerify);

INFER_DATA_SLICE_FUNC_REG(FullyConnectionCompress, FullyConnectionCompressInferDataSlice);

// ----------------Matmul-------------------
string GetMatMulInfo(const Operator &op, const std::string &name_attr) {
  auto desc_a = op.GetInputDesc("x1");
  auto desc_b = op.GetInputDesc("x2");
  ge::Shape shape_a = desc_a.GetShape();
  ge::Shape shape_b = desc_b.GetShape();
  ge::Shape ori_shape_a = desc_a.GetOriginShape();
  ge::Shape ori_shape_b = desc_b.GetOriginShape();
  std::vector<std::pair<int64_t, int64_t>> range_a;
  std::vector<std::pair<int64_t, int64_t>> range_b;
  desc_a.GetShapeRange(range_a);
  desc_b.GetShapeRange(range_b);

  bool trans_a = false, trans_b = false;
  op.GetAttr(name_attr + "_x1", trans_a);
  op.GetAttr(name_attr + "_x2", trans_b);

  std::ostringstream oss;
  oss << "shape of a: ";
  for (int i = 0; i < shape_a.GetDimNum(); i++) {
      oss << shape_a.GetDim(i) << ", ";
  }
  oss << "ori shape of a: ";
  for (int i = 0; i < ori_shape_a.GetDimNum(); i++) {
      oss << ori_shape_a.GetDim(i) << ", ";
  }
  oss << std::endl;
  oss << "shape of b: ";
  for (int i = 0; i < shape_b.GetDimNum(); i++) {
      oss << shape_b.GetDim(i) << ", ";
  }
  oss << "ori shape of b: ";
  for (int i = 0; i < ori_shape_b.GetDimNum(); i++) {
      oss << ori_shape_b.GetDim(i) << ", ";
  }
  oss << std::endl;
  oss << "trans_a " << trans_a << " trans_b " << trans_b << std::endl;
  oss << "range of a: (";
  for (int i = 0; i < range_a.size(); i++) {
      oss << "(" << range_a[i].first << ", " << range_a[i].second << ") ";
  }
  oss << ")" << std::endl;
  oss << "range of b: (";
  for (int i = 0; i < range_b.size(); i++) {
      oss << "(" << range_b[i].first << ", " << range_b[i].second << "), ";
  }
  oss << ")" << std::endl;
  return oss.str();
}

static const std::pair<int64_t, int64_t> FULL_RANGE = {1, -1};
static const std::pair<int64_t, int64_t> EMPTY_RANGE = {0, 0};
static const int64_t NORMALIZE_INFINITE_RANGE = std::numeric_limits<int64_t>::max();
static const std::pair<int64_t, int64_t> NORMALIZE_FULL_RANGE = {1, NORMALIZE_INFINITE_RANGE};
static const int64_t VALUE_UNKNOWN_RANK = -2;
static const int64_t INFINITE_RANGE = -1;
static const int32_t MAX_RANGE = std::numeric_limits<int32_t>::max();
static const std::vector<int64_t> BATCH_GRAR = {0, 1, 3, 7, 15, 31, MAX_RANGE};
static const std::vector<int64_t> SHAPE_GEAR = {0, 16*3, 16*7, 16*15, 16*31, 16*63, 16*127, 16*191,
                                                16*255, 16*511, 16*767, 16*1023, MAX_RANGE};

bool IsDimValid(int64_t dim) {
  return dim >= VALUE_UNKNOWN_RANK && dim != 0;
}

bool IsRangeValid(const std::vector<int64_t> &shape, const std::vector<std::pair<int64_t, int64_t>> &range,
                  const string &op_name, bool is_strict=true) {
  if (shape.empty() || shape == UNKNOWN_RANK || range.empty()) {
    return true;
  }

  if (std::find(shape.begin(), shape.end(), UNKNOWN_DIM) != shape.end()) {
    if (range.size() != shape.size()) {
      CUBE_CALL_ERR_REPORT(op_name.c_str(),
        "length of range(%zu) in dynamic shape scene must be equal to the length of shape(%zu), or equal to 0.",
        range.size(), shape.size());
      return false;
    }

    for (size_t i = 0; i < range.size(); ++i) {
      if (shape[i] == -1 && range[i].second != -1 && range[i].first > range[i].second) {
        CUBE_CALL_ERR_REPORT(op_name.c_str(),
                "%zu-th range(%lld, %lld) is invalid.", i, range[i].first, range[i].second);
        return false;
      }
    }
  }

  // vector op do not update range when shape is fixed
  if (!is_strict) {
    return true;
  }

  if (range.size() != shape.size()) {
    return false;
  }
  for (size_t i = 0; i < range.size(); ++i) {
    if (shape[i] != range[i].first or shape[i] != range[i].second) {
      return false;
    }
  }
  return true;
}

bool IsUnknownShape(const std::vector<int64_t> &shape) {
  if (shape.empty()) {
    return false;
  }

  if (shape == UNKNOWN_RANK || std::find(shape.begin(), shape.end(), UNKNOWN_DIM) != shape.end()) {
    return true;
  }
  return false;
}

void NormalizeRange(const std::string &op_name, const int64_t dim,
                    const std::pair<int64_t, int64_t> &shape_range,
                    std::pair<int64_t, int64_t> &range) {
  if (dim != UNKNOWN_DIM) {
    range = {dim, dim};
    return ;
  }

  if (shape_range == EMPTY_RANGE) {
    range = {1, NORMALIZE_INFINITE_RANGE};
    OP_LOGW(
        op_name.c_str(),
        "[InferShape] the dimension is -1 and no range is provided, therefore, the range is assumed to be [1, %lld]",
        NORMALIZE_INFINITE_RANGE);
  } else if (shape_range.second == INFINITE_RANGE) {
    range = {shape_range.first, NORMALIZE_INFINITE_RANGE};
  } else {
    range = shape_range;
  }
}

bool IntersectDimensionAndRange(const std::string &op_name,
                                const int64_t dim_a,
                                const int64_t dim_b,
                                const std::pair<int64_t, int64_t> &range_a,
                                const std::pair<int64_t, int64_t> &range_b,
                                int64_t &dim,
                                std::pair<int64_t, int64_t> &range) {
  // | b\a        | -1,(y1,y2)                      | y          |
  // | ---------- | ------------------------------- | ---------- |
  // | -1,(x1,x2) | -1,(max(x1,y1),min(x2,y)) check | y check    |
  // | x          | x check                         | x==y check |

  if (dim_a > 0 && dim_b > 0) {
    if (dim_a != dim_b || range_a != range_b) {
      CUBE_INNER_ERR_REPORT(op_name.c_str(), "[InferShape] dimensions a(%ld) and b(%ld) must be same", dim_a, dim_b);
      return false;
    }
    dim = dim_a;
    range = range_a;
    return true;
  }

  if (dim_a == UNKNOWN_DIM && dim_b == UNKNOWN_DIM) {
    auto lower_bound = std::max(range_a.first, range_b.first);
    auto upper_bound = std::min(range_a.second, range_b.second);
    if (lower_bound > upper_bound) {
      CUBE_INNER_ERR_REPORT(op_name.c_str(), "[InferShape] range a(%ld, %ld) and b(%ld, %ld) must have intersections",
        range_a.first, range_a.second, range_b.first, range_b.second);
      return false;
    }

    range.first = lower_bound;
    range.second = upper_bound;
    return true;
  }

  if (dim_a == UNKNOWN_DIM) {
    if (range_a.first <= dim_b && dim_b <= range_a.second) {
      dim = dim_b;
      range = range_b;
      return true;
    }
    CUBE_INNER_ERR_REPORT(op_name.c_str(), "[InferShape] dimension(%ld) must be in range(%ld, %ld)",
                          dim_b, range_a.first, range_b.second);
    return false;
  }
  if (range_b.first <= dim_a && dim_a <= range_b.second) {
    dim = dim_a;
    range = range_a;
    return true;
  }
  CUBE_INNER_ERR_REPORT(op_name.c_str(), "[InferShape] dimension(%ld) must be in range(%ld, %ld)",
    dim_a, range_b.first, range_b.second);
  return false;
}

bool BroadcastDimensionAndRange(const std::string &op_name,
                                const int64_t dim_a,
                                const int64_t dim_b,
                                const std::pair<int64_t, int64_t> &range_a,
                                const std::pair<int64_t, int64_t> &range_b,
                                int64_t &dim,
                                std::pair<int64_t, int64_t> &range) {
  // | b\a        | -1,(1,y)        | -1,(y1,y2)                       | 0          | 1          | y          |
  // | ---------- | --------------- | -------------------------------- | ---------- | ---------- | ---------- |
  // | -1,(1,x)   | -1,(1,max(x,y)) | -1,(y1,y2)                       | -1,(1,x)   | -1,(1,x)   | y check    |
  // | -1,(x1,x2) | -1,(x1,x2)      | -1,(max(x1,y1),min(x2,y2)) check | -1,(x1,x2) | -1,(x1,x2) | y check    |
  // | 0          | -1,(1,y)        | -1,(y1,y2)                       | 0          | 1          | y          |
  // | 1          | -1,(1,y)        | -1,(y1,y2)                       | 1          | 1          | y          |
  // | x          | x check         | x check                          | x          | x          | x==y check |

  if (dim_a == 0) {
    dim = dim_b;
    range = range_b;
    return true;
  }
  if (dim_b == 0) {
    dim = dim_a;
    range = range_a;
    return true;
  }

  if (dim_a == 1) {
    dim = dim_b;
    range = range_b;
    return true;
  }
  if (dim_b == 1) {
    dim = dim_a;
    range = range_a;
    return true;
  }

  if (dim_a > 1 && dim_b > 1) {
    if (dim_a != dim_b) {
      CUBE_INNER_ERR_REPORT(op_name.c_str(), "[InferShape] dimensions a(%ld) and b(%ld) must be equal",
        dim_a, dim_b);
      return false;
    }
    dim = dim_a;
    range = range_a;
    return true;
  }
  if (dim_a > 1) {
    if (range_b.first <= dim_a && dim_a <= range_b.second) {
      dim = dim_a;
      range = range_a;
      return true;
    }
    CUBE_INNER_ERR_REPORT(op_name.c_str(), "[InferShape] dimension(%ld) must be in range(%ld, %ld)",
      dim_a, range_b.first, range_b.second);
    return false;
  }
  if (dim_b > 1) {
    if (range_a.first <= dim_b && dim_b <= range_a.second) {
      dim = dim_b;
      range = range_b;
      return true;
    }
    CUBE_INNER_ERR_REPORT(op_name.c_str(), "[InferShape] dimension(%ld) must be in range(%ld, %ld)",
      dim_b, range_a.first, range_a.second);
    return false;
  }

  if (range_a.first == 1 && range_b.first == 1) {
    dim = UNKNOWN_DIM;
    range = {1, std::max(range_a.second, range_b.second)};
    return true;
  }
  if (range_a.first > 1 && range_b.first > 1) {
    auto lower_bound = std::max(range_a.first, range_b.first);
    auto upper_bound = std::min(range_a.second, range_b.second);
    if (lower_bound > upper_bound) {
      CUBE_INNER_ERR_REPORT(op_name.c_str(), "[InferShape] range a(%ld, %ld) and b(%ld, %ld) must have intersections",
        range_a.first, range_a.second, range_b.first, range_b.second);
      return false;
    }
    dim = UNKNOWN_DIM;
    range = {lower_bound, upper_bound};
    return true;
  }
  if (range_a.first > 1) {
    dim = dim_a;
    range = range_a;
  } else {
    dim = dim_b;
    range = range_b;
  }

  return true;
}

class InferShapeMatMul {
 public:
  bool GetShapeRangeOfOutput();
  InferShapeMatMul(const string &op_name, const vector<int64_t> &shape_a, const vector<int64_t> &shape_b,
                   const vector<int64_t> &shape_bias, vector<std::pair<int64_t, int64_t>> &range_a,
                   vector<std::pair<int64_t, int64_t>> &range_b,
                   vector<std::pair<int64_t, int64_t>> &range_bias, bool trans_a, bool trans_b,
                   vector<int64_t> &shape_out, vector<std::pair<int64_t, int64_t>> &range_out, bool has_batch);

 private:
  bool PrecheckShapeAndRange(const string &op_name);
  bool NormalizeShapeAndRange();
  bool InferMKN();
  bool InferBatch();
  void SimplifyShapeAndRange();
  bool IsStaticShape();
  void InitializeShapeAndRange(const vector<int64_t> &shape,
                               vector<int64_t> &infer_shape,
                               const vector<std::pair<int64_t, int64_t>> &range,
                               vector<std::pair<int64_t, int64_t>> &infer_range);
  bool NormalizeRangeOfMatMul(const vector<int64_t> &shape,
                              const vector<int64_t> &infer_shape,
                              const vector<std::pair<int64_t, int64_t>> &range,
                              vector<std::pair<int64_t, int64_t>> &infer_range);

  static const int64_t base_len;
  const string& op_name;
  const vector<int64_t> &shape_a;
  const vector<int64_t> &shape_b;
  const vector<int64_t> &shape_bias;
  vector<std::pair<int64_t, int64_t>> &range_a;
  vector<std::pair<int64_t, int64_t>> &range_b;
  vector<std::pair<int64_t, int64_t>> &range_bias;
  bool trans_a;
  bool trans_b;
  vector<int64_t> &shape_out;
  vector<std::pair<int64_t, int64_t>> &range_out;
  bool has_batch;
  int64_t num_dim;

  vector<int64_t> infer_shape_a;
  vector<int64_t> infer_shape_b;
  vector<int64_t> infer_shape_bias;
  vector<std::pair<int64_t, int64_t>> infer_range_a;
  vector<std::pair<int64_t, int64_t>> infer_range_b;
  vector<std::pair<int64_t, int64_t>> infer_range_bias;
};

const int64_t InferShapeMatMul::base_len = 2;

bool InferShapeMatMul::IsStaticShape() {
  if (!IsUnknownShape(shape_a) && !IsUnknownShape(shape_b) && !IsUnknownShape(shape_bias)) {
    return true;
  }
  return false;
}

InferShapeMatMul::InferShapeMatMul(const string &op_name, const vector<int64_t> &shape_a,
                                   const vector<int64_t> &shape_b, const vector<int64_t> &shape_bias,
                                   vector<std::pair<int64_t, int64_t>> &range_a,
                                   vector<std::pair<int64_t, int64_t>> &range_b,
                                   vector<std::pair<int64_t, int64_t>> &range_bias,
                                   bool trans_a, bool trans_b,
                                   vector<int64_t> &shape_out, vector<std::pair<int64_t, int64_t>> &range_out,
                                   bool has_batch)
    : op_name(op_name),
      shape_a(shape_a),
      shape_b(shape_b),
      shape_bias(shape_bias),
      range_a(range_a),
      range_b(range_b),
      range_bias(range_bias),
      trans_a(trans_a),
      trans_b(trans_b),
      shape_out(shape_out),
      range_out(range_out),
      has_batch(has_batch) {
  num_dim = std::max(std::max(shape_a.size(), shape_b.size()), shape_bias.size());
  num_dim = std::max(base_len, num_dim);

  infer_shape_a = vector<int64_t>(num_dim);
  infer_range_a = vector<std::pair<int64_t, int64_t>>(num_dim);

  infer_shape_b = vector<int64_t>(num_dim);
  infer_range_b = vector<std::pair<int64_t, int64_t>>(num_dim);

  if (!shape_bias.empty()) {
    infer_shape_bias = vector<int64_t>(num_dim);
    infer_range_bias = vector<std::pair<int64_t, int64_t>>(num_dim);
  }

  shape_out = vector<int64_t>(num_dim);
  range_out = vector<std::pair<int64_t, int64_t>>(num_dim);

  if (IsStaticShape()) {
    range_a = {};
    range_b = {};
    range_bias = {};
  }
}

void InferShapeMatMul::InitializeShapeAndRange(const vector<int64_t> &shape,
                                               vector<int64_t> &infer_shape,
                                               const vector<std::pair<int64_t, int64_t>> &range,
                                               vector<std::pair<int64_t, int64_t>> &infer_range) {
  // deal with shape (-2) range {}
  auto valid_offset = shape == UNKNOWN_RANK ? 0 : infer_shape.size() - shape.size();
  if (shape == UNKNOWN_RANK) {
    fill(infer_shape.begin() + valid_offset, infer_shape.end(), UNKNOWN_DIM);
    fill(infer_range.begin() + valid_offset, infer_range.end(), NORMALIZE_FULL_RANGE);
  } else {
    copy(shape.begin(), shape.end(), infer_shape.begin() + valid_offset);
    copy(range.begin(), range.end(), infer_range.begin() + valid_offset);
  }
}

bool InferShapeMatMul::NormalizeRangeOfMatMul(const vector<int64_t> &shape,
                                              const vector<int64_t> &infer_shape,
                                              const vector<std::pair<int64_t, int64_t>> &range,
                                              vector<std::pair<int64_t, int64_t>> &infer_range) {
  // deal with empty range
  vector<std::pair<int64_t, int64_t>> preprocess_range;
  if (shape != UNKNOWN_RANK && range.empty()) {
    for (auto i = 0; i < shape.size(); ++i) {
      if (shape[i] == -1) {
        preprocess_range.push_back({1, NORMALIZE_INFINITE_RANGE});
      } else {
        preprocess_range.push_back({shape[i], shape[i]});
      }
    }
  } else {
    preprocess_range = range;
  }

  auto valid_offset = infer_range.size() - preprocess_range.size();
  for (auto i = valid_offset; i < infer_range.size(); ++i) {
    if (infer_shape[i] < VALUE_UNKNOWN_RANK || infer_shape[i] == 0) {
      OpsInputShapeErrReport(op_name.c_str(), std::to_string(i - valid_offset) + "-th dim is not supported", "dim",
                             std::to_string(infer_shape[i]));
      OP_LOGE(op_name.c_str(), "[InferShape] %d-th dim(%d) is not supported, please check the output of upper operator",
              i - valid_offset, infer_shape[i]);
      return false;
    }
    if (infer_range[i].first == UNKNOWN_DIM && infer_range[i].second == UNKNOWN_DIM) {
      OpsInputShapeErrReport(op_name.c_str(), "range is not supported", "range", "(-1, -1)");
      OP_LOGE(op_name.c_str(),
              "[InferShape] range like (-1, -1) is not supported, please check the output of upper operator");
      return false;
    }

    NormalizeRange(op_name, infer_shape[i], preprocess_range[i - valid_offset], infer_range[i]);
  }
  return true;
}

bool InferShapeMatMul::PrecheckShapeAndRange(const string &op_name) {
  bool res = true;
  res &= IsRangeValid(shape_a, range_a, op_name, false);
  res &= IsRangeValid(shape_b, range_b, op_name, false);
  res &= IsRangeValid(shape_bias, range_bias, op_name, false);
  return res;
}

bool InferShapeMatMul::NormalizeShapeAndRange() {
  InitializeShapeAndRange(shape_a, infer_shape_a, range_a, infer_range_a);
  InitializeShapeAndRange(shape_b, infer_shape_b, range_b, infer_range_b);
  InitializeShapeAndRange(shape_bias, infer_shape_bias, range_bias, infer_range_bias);

  bool res = true;
  res &= NormalizeRangeOfMatMul(shape_a, infer_shape_a, range_a, infer_range_a);
  res &= NormalizeRangeOfMatMul(shape_b, infer_shape_b, range_b, infer_range_b);
  res &= NormalizeRangeOfMatMul(shape_bias, infer_shape_bias, range_bias, infer_range_bias);

  return res;
}

bool InferShapeMatMul::InferMKN() {
  int64_t idx_m = trans_a ? num_dim - 1 : num_dim - 2;
  int64_t idx_k_a = trans_a ? num_dim - 2 : num_dim - 1;
  int64_t idx_k_b = trans_b ? num_dim - 1 : num_dim - 2;
  int64_t idx_n_b = trans_b ? num_dim - 2 : num_dim - 1;

  auto m = infer_shape_a[idx_m];
  auto k_a = infer_shape_a[idx_k_a];
  auto k_b = infer_shape_b[idx_k_b];
  auto n_b = infer_shape_b[idx_n_b];
  auto n = n_b;

  int64_t k;
  std::pair<int64_t, int64_t> range_k, range_n = infer_range_b[idx_n_b];
  OP_LOGI(op_name.c_str(), "[InferShape] start check the k dim!");
  if (k_a > 0 && k_b > 0 && k_a != k_b) {
    OpsInputShapeErrReport(op_name.c_str(), "The k-axis of a and b tensors must be the same", "a and b", "");
    OP_LOGE(op_name.c_str(), "[InferShape] The k-axis of a(%lld) and b(%lld) tensors must be the same", k_a, k_b);
    return false;
  } else if (k_a < 0 && k_b < 0) {
    if (!IntersectDimensionAndRange(op_name, k_a, k_b, infer_range_a[idx_k_a], infer_range_b[idx_k_b], k, range_k)) {
      OP_LOGE(op_name.c_str(), "[InferShape] The intersection of the k-axis of tensor a and b is invalid");
      return false;
    }
  }
  OP_LOGD(op_name.c_str(), "[InferShape] start check the bias input dim!");
  if (!shape_bias.empty()) {
    int64_t idx_n_bias = num_dim - 1;
    int64_t n_bias = infer_shape_bias[idx_n_bias];
    if (!IsDimValid(n_bias)) {
      OpsInputShapeErrReport(op_name.c_str(), "The dimension must be -2, -1 or greater than 0", "a and b", "");
      OP_LOGE(op_name.c_str(), "[InferShape] dimension must be -2, -1 or greater than 0");
      return false;
    }

    if (!IntersectDimensionAndRange(op_name, n_b, n_bias, infer_range_b[idx_n_b], infer_range_bias[idx_n_bias], n,
                                    range_n)) {
      OP_LOGE(op_name.c_str(), "[InferShape] The intersection of the n-axis of tensor b and bias is invalid");
      return false;
    }
  }

  shape_out[num_dim - 2] = m;
  shape_out[num_dim - 1] = n;
  range_out[num_dim - 2] = infer_range_a[idx_m];
  range_out[num_dim - 1] = range_n;

  return true;
}

bool InferShapeMatMul::InferBatch() {
  for (auto i = 0; i < num_dim - 2; ++i) {
    if (!BroadcastDimensionAndRange(op_name, infer_shape_a[i], infer_shape_b[i], infer_range_a[i], infer_range_b[i],
                                    shape_out[i], range_out[i])) {
      CUBE_INNER_ERR_REPORT(op_name.c_str(),
        "[InferShape] The broadcst operation for tensor a and b on the n-th dimension is failed");
      return false;
    }

    if (!shape_bias.empty()) {
      if (!BroadcastDimensionAndRange(op_name, shape_out[i], infer_shape_bias[i], range_out[i], infer_range_bias[i],
                                      shape_out[i], range_out[i])) {
        CUBE_INNER_ERR_REPORT(op_name.c_str(),
          "[InferShape] The broadcst operation for tensor out and bias on the n-th dimension is failed");
        return false;
      }
    }
  }
  return true;
}

void InferShapeMatMul::SimplifyShapeAndRange() {
  if (std::find(shape_out.begin(), shape_out.end(), UNKNOWN_DIM) == shape_out.end()) {
    range_out = {};
    return;
  }

  for (int i = 0; i < range_out.size(); i++) {
    if (range_out[i].first == range_out[i].second) {
      shape_out[i] = range_out[i].first;
    }

    // reverse normalize
    if (range_out[i].second == NORMALIZE_INFINITE_RANGE) {
      range_out[i] = {range_out[i].first, INFINITE_RANGE};
    }
  }
}

bool InferShapeMatMul::GetShapeRangeOfOutput() {
  if (!has_batch && shape_a == UNKNOWN_RANK && shape_b == UNKNOWN_RANK &&
      (shape_bias.empty() || shape_bias == UNKNOWN_RANK)) {
    shape_out = UNKNOWN_RANK;
    range_out = {};
    OP_LOGW(op_name.c_str(), "[InferShape] cannot derive any shape and range information of output");
    return true;
  }
  if (has_batch && (shape_a == UNKNOWN_RANK || shape_b == UNKNOWN_RANK || shape_bias == UNKNOWN_RANK)) {
    shape_out = UNKNOWN_RANK;
    range_out = {};
    OP_LOGW(op_name.c_str(), "[InferShape] cannot derive any shape and range information of output");
    return true;
  }

  if (!PrecheckShapeAndRange(op_name)) {
    return false;
  }

  if (!NormalizeShapeAndRange()) {
    return false;
  }

  if (!InferMKN()) {
    return false;
  }

  if (!InferBatch()) {
    return false;
  }

  SimplifyShapeAndRange();

  return true;
}

void GetMatmulShapeGear(int64_t dim_size,
                        const std::vector<int64_t> &shape_gear,
                        std::pair<int64_t, int64_t> &range) {
  int position = 1;
  while (position < shape_gear.size() && shape_gear[position] < dim_size) {
    position++;
  }
  range = {shape_gear[position - 1] + 1, shape_gear[position]};
}

int32_t CalculateMatmulShapeRange(const std::vector<int64_t> &shape,
                                  std::vector<std::pair<int64_t, int64_t>> &single_point_range) {
  for (int i = 0; i < shape.size() - 2; i++) {
    if (shape[i] > MAX_RANGE) {
      return -1;
    }
    GetMatmulShapeGear(shape[i], BATCH_GRAR, single_point_range[i]);
  }
  for (int i = shape.size() - 2; i < shape.size(); i++) {
    if (shape[i] > MAX_RANGE) {
      return -1;
    }
    GetMatmulShapeGear(shape[i], SHAPE_GEAR, single_point_range[i]);
  }
  return 0;
}

graphStatus GetMatMulOutputShape(Operator &op,
                                 std::vector<int64_t> &shape_out,
                                 std::vector<std::pair<int64_t, int64_t>> &shape_range_out,
                                 const std::string &name_attr, bool has_batch) {
  ge::TensorDesc desc_a = op.GetInputDesc("x1");
  ge::TensorDesc desc_b = op.GetInputDesc("x2");
  auto shape_a = desc_a.GetShape().GetDims();
  auto shape_b = desc_b.GetShape().GetDims();
  std::vector<std::pair<int64_t, int64_t>> shape_range_a;
  std::vector<std::pair<int64_t, int64_t>> shape_range_b;
  desc_a.GetShapeRange(shape_range_a);
  desc_b.GetShapeRange(shape_range_b);

  ge::TensorDesc desc_bias;
  std::vector<std::pair<int64_t, int64_t>> shape_range_bias;
  vector<int64_t> shape_bias;
  if (ge::GRAPH_SUCCESS == op.TryGetInputDesc("bias", desc_bias)) {
    shape_bias = desc_bias.GetShape().GetDims();
    desc_bias.GetShapeRange(shape_range_bias);
  }

  bool trans_a = false;
  if (ge::GRAPH_SUCCESS != op.GetAttr(name_attr + "_x1", trans_a)) {
    OpsGetAttrErrReport(op.GetName(), name_attr + "_x1");
    OP_LOGE(op.GetName().c_str(), "[Plugin][ERROR]%s GetOpAttr %s_x1 failed!",
            op.GetName().c_str(), name_attr.c_str());
    return GRAPH_FAILED;
  }
  if(shape_a.size() == 1 && shape_a[0] > 0) {
    shape_a.insert(shape_a.begin(), 1);
    shape_range_a.insert(shape_range_a.begin(), make_pair<int64_t, int64_t>(1, 1));
  }
  if(shape_b.size() == 1 && shape_b[0] > 0) {
    shape_b.push_back(1);
    shape_range_b.push_back(make_pair<int64_t, int64_t>(1, 1));
  }
  bool trans_b = false;
  if (ge::GRAPH_SUCCESS != op.GetAttr(name_attr + "_x2", trans_b)) {
    OpsGetAttrErrReport(op.GetName(), name_attr + "_x2");
    OP_LOGE(op.GetName().c_str(), "[Plugin][ERROR]%s GetOpAttr %s_x2 failed!",
            op.GetName().c_str(), name_attr.c_str());
    return GRAPH_FAILED;
  }

  bool fuzzy_flag = false;
  bool is_static_shape = !IsUnKnownShape(shape_a) && !IsUnKnownShape(shape_b) && !IsUnknownShape(shape_bias);
  if (ge::GRAPH_SUCCESS == op.GetAttr(ge::ATTR_NAME_FUZZ_BUILD, fuzzy_flag) && fuzzy_flag && is_static_shape) {
    int shape_a_length = shape_a.size();
    int shape_b_length = shape_b.size();
    std::vector<std::pair<int64_t, int64_t>> single_point_range_a(shape_a_length);
    std::vector<std::pair<int64_t, int64_t>> single_point_range_b(shape_b_length);
    int32_t calc_range_a = CalculateMatmulShapeRange(shape_a, single_point_range_a);
    if (calc_range_a < 0) {
      OP_LOGE(op.GetName().c_str(), "shape of input_x1 is too large");
      return GRAPH_FAILED;
    }
    int32_t calc_range_b = CalculateMatmulShapeRange(shape_b, single_point_range_b);
    if (calc_range_b < 0) {
      OP_LOGE(op.GetName().c_str(), "shape of input_x2 is too large");
      return GRAPH_FAILED;
    }
    shape_range_a = single_point_range_a;
    shape_range_b = single_point_range_b;
    desc_a.SetShapeRange(shape_range_a);
    desc_b.SetShapeRange(shape_range_b);
    (void) op.UpdateInputDesc("x1", desc_a);
    (void) op.UpdateInputDesc("x2", desc_b);
    std::string transpose_name = "transpose";
    if (shape_a_length > 2 || shape_b_length > 2) {
      transpose_name = "adj";
    }
    OP_LOGD(op.GetName().c_str(), "%s", GetMatMulInfo(op, transpose_name).c_str());
  }
  auto obj = InferShapeMatMul(op.GetName(), shape_a, shape_b, shape_bias, shape_range_a, shape_range_b,
                              shape_range_bias, trans_a, trans_b, shape_out, shape_range_out, has_batch);
  if (!obj.GetShapeRangeOfOutput()) {
    return GRAPH_FAILED;
  }
  OP_LOGD(op.GetName().c_str(), "fuzzy_flag = %d", fuzzy_flag);
  return GRAPH_SUCCESS;
}

bool InferMatmulInputNZ(const Operator &op,
                        vector<vector<int64_t>> &output,
                        bool trans_a, bool trans_b) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_x1 = op_desc->MutableInputDesc("x1");
  GeTensorDescPtr tensor_desc_x2 = op_desc->MutableInputDesc("x2");
  vector<vector<int64_t>> x1_data_slice = {{}, {}, {}, {}};
  vector<vector<int64_t>> x2_data_slice = {{}, {}, {}, {}};
  for(int i = 0; i < output.size(); i++) {
    if (output[i].size() > 1) {
      if (i == 0) {
        if (!trans_b) {
          x2_data_slice[1] = output[i];
        } else {
          x2_data_slice[0] = output[i];
        }
        if(!AttrUtils::SetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice)) {
          return false;
        }
        OP_LOGI(op.GetName().c_str(), "infer input in N success");
        return true;
      } else if (i == 1) {
        if (!trans_a) {
          x1_data_slice[0] = output[i];
        } else {
          x1_data_slice[1] = output[i];
        }
        if(!AttrUtils::SetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice)) {
          return false;
        }
        OP_LOGI(op.GetName().c_str(), "infer input in M success");
        return true;
      } else {
        OP_LOGI(op.GetName().c_str(), "cannot support cut in block_n and block_m");
        return false;
      }
    }
  }
  OP_LOGI(op.GetName().c_str(), "no data slice, not need infer input");
  return false;
}

bool InferMatmulInputND(const Operator &op,
                        vector<vector<int64_t>> &output,
                        bool trans_a, bool trans_b) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_x1 = op_desc->MutableInputDesc("x1");
  GeTensorDescPtr tensor_desc_x2 = op_desc->MutableInputDesc("x2");
  vector<vector<int64_t>> x1_data_slice = {{}, {}};
  vector<vector<int64_t>> x2_data_slice = {{}, {}};
  for(int i = 0; i < output.size(); i++) {
    if (output[i].size() > 1) {
      if (i == 0) {
        if (!trans_a) {
          x1_data_slice[0] = output[i];
        } else {
          x1_data_slice[1] = output[i];
        }
        if(!AttrUtils::SetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice)) {
          return false;
        }
        OP_LOGI(op.GetName().c_str(), "infer input in M success");
        return true;
      } else if (i == 1) {
        if (!trans_b) {
          x2_data_slice[1] = output[i];
        } else {
          x2_data_slice[0] = output[i];
        }
        if(!AttrUtils::SetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice)) {
          return false;
        }
        OP_LOGI(op.GetName().c_str(), "infer input in N success");
        return true;
      }
    }
  }
  OP_LOGI(op.GetName().c_str(), "no data slice, not need infer input");
  return false;
}

bool InferMatmul(const Operator &op) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");

  bool trans_a = false;
  if (ge::GRAPH_SUCCESS != op.GetAttr("transpose_x1", trans_a)) {
    OpsGetAttrErrReport(op.GetName(), "transpose_x1");
    OP_LOGE(op.GetName().c_str(), "[Plugin][ERROR]%s GetOpAttr transpose_x1 failed!",
            op.GetName().c_str());
    return false;
  }
  bool trans_b = false;
  if (ge::GRAPH_SUCCESS != op.GetAttr("transpose_x2", trans_b)) {
    OpsGetAttrErrReport(op.GetName(), "transpose_x2");
    OP_LOGE(op.GetName().c_str(), "[Plugin][ERROR]%s GetOpAttr transpose_x2 failed!",
            op.GetName().c_str());
    return false;
  }

  Format x1_format = op.GetInputDesc("x1").GetFormat();
  Format x2_format = op.GetInputDesc("x2").GetFormat();
  if (x1_format == FORMAT_FRACTAL_NZ) {
    trans_a = !trans_a;
  }
  if (x2_format == FORMAT_FRACTAL_NZ) {
    trans_b = !trans_b;
  }

  vector<vector<int64_t>> y_data_slice;

  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op.GetName().c_str(), "no data slice, not need infer input");
    return false;
  }

  if (x1_format == FORMAT_FRACTAL_NZ) {
    if(!InferMatmulInputNZ(op, y_data_slice, trans_a, trans_b)) {
      return false;
    }
    return true;
  } else {
    if(!InferMatmulInputND(op, y_data_slice, trans_a, trans_b)) {
      return false;
    }
    return true;
  }
}

bool InferBatchMatmulInputNZ(const Operator &op,
                             vector<vector<int64_t>> &output,
                             bool trans_a, bool trans_b,
                             size_t x1_dims, size_t x2_dims) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_x1 = op_desc->MutableInputDesc("x1");
  GeTensorDescPtr tensor_desc_x2 = op_desc->MutableInputDesc("x2");
  vector<vector<int64_t>> x1_data_slice(x1_dims);
  vector<vector<int64_t>> x2_data_slice(x2_dims);
  size_t y_dims = output.size();

  for(int i = 0; i < y_dims; i++) {
    if (output[i].size() > 1) {
      if (i == y_dims - 4) {
        // split n
        if (!trans_b) {
          x2_data_slice[x2_dims - 3] = output[i];
        } else {
          x2_data_slice[x2_dims - 4] = output[i];
        }
        if(!AttrUtils::SetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice)) {
          return false;
        }
        OP_LOGI(op.GetName().c_str(), "infer input in N success");
        return true;
      } else if (i == y_dims - 3) {
        if (!trans_a) {
          x1_data_slice[x1_dims - 4] = output[i];
        } else {
          x1_data_slice[x2_dims - 3] = output[i];
        }
        if(!AttrUtils::SetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice)) {
          return false;
        }
        OP_LOGI(op.GetName().c_str(), "infer input in M success");
        return true;
      } else if (i < y_dims - 4){
        // split batch
        x1_data_slice[i] = output[i];
        if(!AttrUtils::SetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice)) {
          return false;
        }
        if (x2_dims == x1_dims) {
          x2_data_slice[i] = output[i];
          if(!AttrUtils::SetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice)) {
            return false;
          }
        }
        OP_LOGI(op.GetName().c_str(), "infer input in batch success");
        return true;
      } else {
        OP_LOGI(op.GetName().c_str(), "cannot support cut in block_n and block_m");
        return false;
      }
    }
  }
  OP_LOGI(op.GetName().c_str(), "no data slice, not need infer input");
  return false;
}

bool InferBatchMatmulInputND(const Operator &op,
                             vector<vector<int64_t>> &output,
                             bool trans_a, bool trans_b,
                             size_t x1_dims, size_t x2_dims) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_x1 = op_desc->MutableInputDesc("x1");
  GeTensorDescPtr tensor_desc_x2 = op_desc->MutableInputDesc("x2");
  vector<vector<int64_t>> x1_data_slice(x1_dims);
  vector<vector<int64_t>> x2_data_slice(x2_dims);
  size_t y_dims = output.size();

  for(int i = 0; i < y_dims; i++) {
    if (output[i].size() > 1) {
      if (i == y_dims - 2) {
        // split m
        if (!trans_a) {
          x1_data_slice[x1_dims - 2] = output[i];
        } else {
          x1_data_slice[x1_dims - 1] = output[i];
        }
        if(!AttrUtils::SetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice)) {
          return false;
        }
        OP_LOGI(op.GetName().c_str(), "infer input in M success");
        return true;
      } else if (i == y_dims - 1) {
        // split n
        if (!trans_b) {
          x2_data_slice[x2_dims - 1] = output[i];
        } else {
          x2_data_slice[x2_dims - 2] = output[i];
        }
        if(!AttrUtils::SetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice)) {
          return false;
        }
        OP_LOGI(op.GetName().c_str(), "infer input in N success");
        return true;
      } else {
        x1_data_slice[i] = output[i];
        if(!AttrUtils::SetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice)) {
          return false;
        }
        if (x2_dims == x1_dims) {
          x2_data_slice[i] = output[i];
          if(!AttrUtils::SetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice)) {
            return false;
          }
        }
        OP_LOGI(op.GetName().c_str(), "infer input in Batch success");
        return true;
      }
    }
  }
  OP_LOGI(op.GetName().c_str(), "no data slice, not need infer input");
  return false;
}

bool InferBatchMatmul(const Operator &op) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");

  bool trans_a = false;
  bool trans_b = false;
  if (ge::GRAPH_SUCCESS != op.GetAttr("adj_x1", trans_a)) {
    OpsGetAttrErrReport(op.GetName(), "transposeA");
    OP_LOGE(op.GetName().c_str(), "[Plugin][ERROR]%s GetOpAttr transposeA failed!",
            op.GetName().c_str());
    return false;
  }
  if (ge::GRAPH_SUCCESS != op.GetAttr("adj_x2", trans_b)) {
    OpsGetAttrErrReport(op.GetName(), "transposeB");
    OP_LOGE(op.GetName().c_str(), "[Plugin][ERROR]%s GetOpAttr transposeB failed!",
            op.GetName().c_str());
    return false;
  }

  Format x1_format = op.GetInputDesc("x1").GetFormat();
  Format x2_format = op.GetInputDesc("x2").GetFormat();
  size_t x1_dims = op.GetInputDesc("x1").GetShape().GetDimNum();
  size_t x2_dims = op.GetInputDesc("x2").GetShape().GetDimNum();
  if (x1_format == FORMAT_FRACTAL_NZ) {
    trans_a = !trans_a;
  }
  if (x2_format == FORMAT_FRACTAL_NZ) {
    trans_b = !trans_b;
  }

  vector<vector<int64_t>> y_data_slice;
  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op.GetName().c_str(), "no data slice, not need infer input");
    return false;
  }

  if (x1_format == FORMAT_FRACTAL_NZ) {
    if(!InferBatchMatmulInputNZ(op, y_data_slice, trans_a, trans_b, x1_dims, x2_dims)) {
      return false;
    }
    return true;
  } else {
    if(!InferBatchMatmulInputND(op, y_data_slice, trans_a, trans_b, x1_dims, x2_dims)) {
      return false;
    }
    return true;
  }
}

// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(MatMul, MatMulVerify) {
  std::vector<DataType> support_list;
  support_list.push_back(DT_FLOAT16);
  support_list.push_back(DT_FLOAT);
  support_list.push_back(DT_INT32);
  if (CheckInputDataType(op, "x1", support_list) == false) {
    return GRAPH_FAILED;
  }
  if (CheckInputDataType(op, "x2", support_list) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(MatMulInferShape) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto tensordesc_output = op_desc->MutableOutputDesc("y");
  auto tensordesc_x1 = op_desc->GetInputDesc("x1");
  auto tensordesc_x2 = op_desc->GetInputDesc("x2");

  auto dtype = tensordesc_x1.GetDataType();

  OP_LOGD(op.GetName().c_str(), "start judge the dtype for matmul!");
  if (dtype == DT_FLOAT) {
    OP_LOGW(op.GetName().c_str(), "[Plugin][WARNING]MatMul fp32 op has poor performance!");
  }

  OP_LOGD(op.GetName().c_str(), "%s", GetMatMulInfo(op, "transpose").c_str());

  std::vector<int64_t> shape_out;
  std::vector<std::pair<int64_t, int64_t>> shape_range_out;
  if (GRAPH_SUCCESS != GetMatMulOutputShape(op, shape_out, shape_range_out, "transpose", false)) {
    return GRAPH_FAILED;
  }

  ge::GeShape shape_out_desc{shape_out};
  tensordesc_output->SetShapeRange(shape_range_out);
  tensordesc_output->SetShape(shape_out_desc);
  tensordesc_output->SetOriginShape(shape_out_desc);
  tensordesc_output->SetDataType(tensordesc_x1.GetDataType());
  return GRAPH_SUCCESS;
}

// the slice infer
IMPLEMT_INFER_DATA_SLICE(MatMul, MatMulInferDataSlice) {
  OP_LOGD(op.GetName().c_str(), "Enter Matmul InferDataSlice.");
  if (!InferMatmul(op)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(MatMul, MatMulInferShape);

// Registered verify function
VERIFY_FUNC_REG(MatMul, MatMulVerify);

// Registered slice function
INFER_DATA_SLICE_FUNC_REG(MatMul, MatMulInferDataSlice);
// ----------------Matmul-------------------
// ----------------Matmul-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(MatMulV2, MatMulV2Verify) {
  std::vector<DataType> support_list;
  support_list.push_back(DT_FLOAT16);
  support_list.push_back(DT_FLOAT);
  support_list.push_back(DT_INT32);
  support_list.push_back(DT_INT8);
  if (CheckInputDataType(op, "x1", support_list) == false) {
    return GRAPH_FAILED;
  }
  if (CheckInputDataType(op, "x2", support_list) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(MatMulV2InferShape) {
  OP_LOGD(op.GetName().c_str(), "[MatMulV2 Infershape] Start matmul infershape.");
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto tensordesc_output = op_desc->MutableOutputDesc("y");
  auto tensordesc_x1 = op_desc->MutableInputDesc("x1");
  auto tensordesc_x2 = op_desc->MutableInputDesc("x2");

  auto shape_x1 = tensordesc_x1->GetShape();
  auto shape_x2 = tensordesc_x2->GetShape();
  auto dtype = tensordesc_x1->GetDataType();
  OP_LOGD(op.GetName().c_str(), "[MatMulV2 Infershape] Check the input dtype.");
  if (dtype == DT_FLOAT) {
    OP_LOGW(op.GetName().c_str(), "[Plugin][WARNING]MatMul fp32 op has poor performance!");
  }
  OP_LOGD(op.GetName().c_str(), "[MatMulV2 Infershape] Check the input shape length.");
  if (shape_x1.GetDims() != UNKNOWN_RANK && shape_x1.GetDims().size() != 2 && shape_x1.GetDims().size() != 4) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "[Plugin][ERROR]Matmul the first input dims is not 2 or 4!");
    return GRAPH_FAILED;
  }

  OP_LOGD(op.GetName().c_str(), "%s", GetMatMulInfo(op, "transpose").c_str());

  std::vector<int64_t> shape_out;
  std::vector<std::pair<int64_t, int64_t>> shape_range_out;
  OP_LOGD(op.GetName().c_str(), "[MatMulV2 Infershape] Check the transpose attr.");
  if (GRAPH_SUCCESS != GetMatMulOutputShape(op, shape_out, shape_range_out, "transpose", false)) {
    return GRAPH_FAILED;
  }
  OP_LOGD(op.GetName().c_str(), "[MatMulV2 Infershape] The transpose attr is Ok.");
  ge::GeShape shape_out_desc{shape_out};
  OP_LOGD(op.GetName().c_str(), "[MatMulV2 Infershape] Start to set output shape.");
  tensordesc_output->SetShape(shape_out_desc);
  tensordesc_output->SetOriginShape(shape_out_desc);
  tensordesc_output->SetShapeRange(shape_range_out);
  if (tensordesc_x1->GetDataType() == ge::DT_INT8) {
    tensordesc_output->SetDataType(ge::DT_INT32);
  } else {
    tensordesc_output->SetDataType(tensordesc_x1->GetDataType());
  }
  OP_LOGD(op.GetName().c_str(), "[MatMulV2 Infershape] End MatMulV2 infershape.");
  return GRAPH_SUCCESS;
}
IMPLEMT_INFERFORMAT_FUNC(MatMulV2, MatMulV2InferFormat) {
  OP_LOGD(op.GetName().c_str(), "[MatMulV2 Inferformat] Finaly input format is %d", FORMAT_ND);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);

  auto tensordesc_input = op_desc->MutableInputDesc("x1");
  tensordesc_input->SetOriginFormat(FORMAT_ND);
  tensordesc_input->SetFormat(FORMAT_ND);

  auto tensordesc_input_2 = op_desc->MutableInputDesc("x2");
  tensordesc_input_2->SetOriginFormat(FORMAT_ND);
  tensordesc_input_2->SetFormat(FORMAT_ND);

  return GRAPH_SUCCESS;
}
// the slice infer
IMPLEMT_INFER_DATA_SLICE(MatMulV2, MatMulV2InferDataSlice) {
  OP_LOGD(op.GetName().c_str(), "Enter MatmulV2 InferDataSlice.");
  if (!InferMatmul(op)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FORMAT_FUNC_REG(MatMulV2, MatMulV2InferFormat);
// Registered inferfunction
COMMON_INFER_FUNC_REG(MatMulV2, MatMulV2InferShape);

// Registered verify function
VERIFY_FUNC_REG(MatMulV2, MatMulV2Verify);

// Registered slice function
INFER_DATA_SLICE_FUNC_REG(MatMulV2, MatMulV2InferDataSlice);
// ----------------Matmul-------------------
// ----------------GEMM-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(GEMM, GemmVerify) {
  std::vector<DataType> support_list;
  std::vector<DataType> support_list_ab;

  support_list.push_back(DT_FLOAT16);
  support_list.push_back(DT_INT8);

  support_list_ab.push_back(DT_FLOAT);
  support_list_ab.push_back(DT_INT32);
  support_list_ab.push_back(DT_FLOAT16);
  OP_LOGD(op.GetName().c_str(), "[GEMM Verify] Start GEMM Verify.");

  if (CheckInputDataType(op, "a", support_list) == false) {
    TbeInputDataTypeErrReport(op.GetName().c_str(), "a", "float16,int8",
                              DataTypeToStringDesc(op.GetInputDesc("a").GetDataType()));
    return GRAPH_FAILED;
  }
  if (CheckInputDataType(op, "b", support_list) == false) {
    TbeInputDataTypeErrReport(op.GetName().c_str(), "b", "float16,int8",
                              DataTypeToStringDesc(op.GetInputDesc("b").GetDataType()));
    return GRAPH_FAILED;
  }
  if (CheckInputDataType(op, "c", support_list_ab) == false) {
    TbeInputDataTypeErrReport(op.GetName().c_str(), "c", "float32,int32,float16",
                              DataTypeToStringDesc(op.GetInputDesc("c").GetDataType()));
    return GRAPH_FAILED;
  }
  if (CheckInputDataType(op, "alpha", support_list_ab) == false) {
    TbeInputDataTypeErrReport(op.GetName().c_str(), "alpha", "float32,int32,float16",
                              DataTypeToStringDesc(op.GetInputDesc("alpha").GetDataType()));
    return GRAPH_FAILED;
  }
  if (CheckInputDataType(op, "beta", support_list_ab) == false) {
    TbeInputDataTypeErrReport(op.GetName().c_str(), "beta", "float32,int32,float16",
                              DataTypeToStringDesc(op.GetInputDesc("beta").GetDataType()));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(GemmInferShape) {
  OP_LOGD(op.GetName().c_str(), "[GEMM Infershape] Start GEMM infershape.");
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  ge::TensorDesc inputTensorDescC = op.GetInputDesc("c");
  DataType dtype = inputTensorDescC.GetDataType();
  ge::Shape shapeC = inputTensorDescC.GetShape();
  ge::Shape outputShape(shapeC);
  tensordesc_output.SetDataType(dtype);

  tensordesc_output.SetShape(outputShape);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

// not change inferformat
IMPLEMT_INFERFORMAT_FUNC(GEMM, GemmInferFormat) { return GRAPH_SUCCESS;}
INFER_FORMAT_FUNC_REG(GEMM, GemmInferFormat);

// Registered inferfunction
COMMON_INFER_FUNC_REG(GEMM, GemmInferShape);

// Registered verify function
VERIFY_FUNC_REG(GEMM, GemmVerify);
// ----------------GEMM-------------------

// ----------------BatchMatMul-------------------

// Check the dtype and attr of the input tensor description.
// ----------------BatchMatMul-------------------

// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(BatchMatMul, BatchMatMulVerify) {
  return GRAPH_SUCCESS;
}

void modify_batchmatmul_outputshape(vector<int64_t> &shape_out, const vector<int64_t> &shape_y) {
  vector<int64_t> shape_out_new(2);
  if (shape_y.size() >= 2) {
    shape_out = shape_y;
  }
  if (shape_y.size() == 1 && shape_out.size() >= 2) {
    copy(shape_out.end() - 2, shape_out.end(), shape_out_new.begin());
    shape_out = shape_out_new;
  }
}

graphStatus CommonBatchMatMulInferShape(Operator &op) {
  OP_LOGD(op.GetName().c_str(), "%s", GetMatMulInfo(op, "adj").c_str());

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto tensordesc_out = op_desc->MutableOutputDesc("y");
  auto tensordesc_x1 = op_desc->GetInputDesc("x1");
  auto tensordesc_x2 = op_desc->GetInputDesc("x2");

  ge::TensorDesc tensordesc_bias;
  vector<int64_t> shape_bias;
  if (ge::GRAPH_SUCCESS == op.TryGetInputDesc("bias", tensordesc_bias)) {
    shape_bias = tensordesc_bias.GetShape().GetDims();
  }

  auto shape_x1 = tensordesc_x1.GetShape().GetDims();
  auto shape_x2 = tensordesc_x2.GetShape().GetDims();

  size_t dim_num_x1 = shape_x1.size();
  size_t dim_num_x2 = shape_x2.size();

  bool trans_a = false;
  bool trans_b = false;
  if (ge::GRAPH_SUCCESS != op.GetAttr("adj_x1", trans_a)) {
    OpsGetAttrErrReport(op.GetName(), "transposeA");
    printf("GetOpAttr transpose_a or transpose_a failed!");
    return GRAPH_FAILED;
  }
  if (ge::GRAPH_SUCCESS != op.GetAttr("adj_x2", trans_b)) {
    OpsGetAttrErrReport(op.GetName(), "transposeB");
    printf("GetOpAttr transpose_a or transpose_b failed!");
    return GRAPH_FAILED;
  }

  int dim_num = std::max(dim_num_x1, dim_num_x2);
  bool any_unknown_rank = shape_x1 == UNKNOWN_RANK || shape_x1 == UNKNOWN_RANK || shape_bias == UNKNOWN_RANK;
  if (!any_unknown_rank && (dim_num < 1 || dim_num > 8)) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "[Infershape]The shape can only be in the range of 1 to 8.");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> shape_out;
  std::vector<std::pair<int64_t, int64_t>> shape_range_out;
  if (GRAPH_SUCCESS != GetMatMulOutputShape(op, shape_out, shape_range_out, "adj", true)) {
    return GRAPH_FAILED;
  }

  OP_LOGD(op.GetName().c_str(), "modify output shape with ori_shape");

  tensordesc_out->SetShape(ge::GeShape(shape_out));
  tensordesc_out->SetShapeRange(shape_range_out);
  if (tensordesc_x1.GetDataType() == ge::DT_INT8) {
    tensordesc_out->SetDataType(ge::DT_INT32);
  } else {
    tensordesc_out->SetDataType(tensordesc_x1.GetDataType());
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(BatchMatMulInferShape) {
  return CommonBatchMatMulInferShape(op);
}

// the slice infer
IMPLEMT_INFER_DATA_SLICE(BatchMatMul, BatchMatMulInferDataSlice) {
  OP_LOGD(op.GetName().c_str(), "Enter BatchMatmul InferDataSlice.");
  if (!InferBatchMatmul(op)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(BatchMatMul, BatchMatMulInferShape);

// Registered verify function
VERIFY_FUNC_REG(BatchMatMul, BatchMatMulVerify);

// Registered slice function
INFER_DATA_SLICE_FUNC_REG(BatchMatMul, BatchMatMulInferDataSlice);

// ----------------BatchMatMulV2-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(BatchMatMulV2, BatchMatMulV2Verify) {
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(BatchMatMulV2InferShape) {
  return CommonBatchMatMulInferShape(op);
}

IMPLEMT_INFERFORMAT_FUNC(BatchMatMulV2, BatchMatMulV2InferFormat) {
  OP_LOGD(op.GetName().c_str(), "[BatchMatMulV2 Inferformat] Finaly input format is %d", FORMAT_ND);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);

  auto tensordesc_input = op_desc->MutableInputDesc("x1");
  tensordesc_input->SetOriginFormat(FORMAT_ND);
  tensordesc_input->SetFormat(FORMAT_ND);

  auto tensordesc_input_2 = op_desc->MutableInputDesc("x2");
  tensordesc_input_2->SetOriginFormat(FORMAT_ND);
  tensordesc_input_2->SetFormat(FORMAT_ND);

  return GRAPH_SUCCESS;
}

// the slice infer
IMPLEMT_INFER_DATA_SLICE(BatchMatMulV2, BatchMatMulV2InferDataSlice) {
  OP_LOGD(op.GetName().c_str(), "Enter BatchMatmulV2 InferDataSlice.");
  if (!InferBatchMatmul(op)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Registered inferfunction
INFER_FORMAT_FUNC_REG(BatchMatMulV2, BatchMatMulV2InferFormat);
COMMON_INFER_FUNC_REG(BatchMatMulV2, BatchMatMulV2InferShape);

// Registered verify function
VERIFY_FUNC_REG(BatchMatMulV2, BatchMatMulV2Verify);

// Registered slice function
INFER_DATA_SLICE_FUNC_REG(BatchMatMulV2, BatchMatMulV2InferDataSlice);

// - ---------------L2Loss-------------------
IMPLEMT_COMMON_INFERFUNC(L2LossInferShape) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  DataType input_dtype = input_desc->GetDataType();
  auto output_desc = op_info->MutableOutputDesc("y");

  std::vector<std::pair<int64_t, int64_t>> input_range;
  std::vector<int64_t> shape_vector;
  output_desc->SetShape(GeShape(shape_vector));
  output_desc->SetOriginShape(GeShape(shape_vector));
  output_desc->SetShapeRange(input_range);
  output_desc->SetDataType(input_dtype);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(L2Loss, L2LossInferShape);
// --------------L2Loss END-----------------

// ----------------DiagPart-------------------
IMPLEMT_COMMON_INFERFUNC(DiagPartInferShape) {
  Shape shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  std::vector<int64_t> dim_vector;
  for (size_t i = 0; i < shape.GetDimNum() / 2; i++) {
    dim_vector.push_back(shape.GetDim(i));
  }
  Shape output_shape(dim_vector);
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(output_shape);
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(DiagPart, DiagPartInferShape);
// ----------------DiagPart END-------------------

// ----------------DiagPartD-------------------
IMPLEMT_COMMON_INFERFUNC(DiagPartDInferShape) {
  Shape shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  std::vector<int64_t> dim_vector;
  for (size_t i = 0; i < shape.GetDimNum() / 2; i++) {
    dim_vector.push_back(shape.GetDim(i));
  }
  Shape output_shape(dim_vector);
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(output_shape);
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(DiagPartD, DiagPartDInferShape);
// ----------------DiagPartD END-------------------

// ---------------MatrixDiag-------------------
IMPLEMT_COMMON_INFERFUNC(MatrixDiagInferShape) {
  Shape input_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  int64_t dimsInput = input_shape.GetDimNum() - 1;
  int64_t dimNums1 = input_shape.GetDim(dimsInput);

  vector<int64_t> dimInfo = input_shape.GetDims();
  std::vector<int64_t> dim_vec;
  for (size_t j = 0; j < input_shape.GetDimNum(); ++j) {
    dim_vec.push_back(dimInfo[j]);
  }
  dim_vec.push_back(dimNums1);
  td.SetShape(ge::Shape(dim_vec));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MatrixDiag, MatrixDiagVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixDiag, MatrixDiagInferShape);
VERIFY_FUNC_REG(MatrixDiag, MatrixDiagVerify);
// ----------------MatrixDiag END----------------

// ---------------MatrixDiagD-------------------
IMPLEMT_COMMON_INFERFUNC(MatrixDiagDInferShape) {
  Shape x_shape = op.GetInputDesc("x").GetShape();
  Shape assist_shape = op.GetInputDesc("assist").GetShape();
  TensorDesc td = op.GetOutputDesc("y");
  std::vector<int64_t> dims_x = x_shape.GetDims();
  std::vector<int64_t> dims_assist = assist_shape.GetDims();
  if (dims_x.size() < dims_assist.size()) {
    std::vector<int64_t> dims_tmp = dims_x;
    dims_x = dims_assist;
    dims_assist = dims_tmp;
  }

  if (dims_x.size() != dims_assist.size()) {
    int dec = dims_x.size() - dims_assist.size();
    for (int i = 0; i < dec; i++) {
      dims_assist.insert(dims_assist.begin(), (int64_t)1);
    }
  }

  std::vector<int64_t> dim_vec;
  for (size_t i = 0; i < dims_x.size(); i++) {
    if ((dims_x[i] != dims_assist[i]) && (dims_x[i] != 1) && (dims_assist[i] != 1)) {
      std::string err_msg = OtherErrMsg(ConcatString("The dimensions does not match the broadcast rule(", dims_x[i], ", ", dims_assist[i], ")"));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    }

    int64_t dims = dims_x[i] > dims_assist[i] ? dims_x[i] : dims_assist[i];
    dim_vec.push_back(dims);
  }
  td.SetShape(ge::Shape(dim_vec));
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MatrixDiagD, MatrixDiagDVerify) {
  DataType input_diagonal_dtype = op.GetInputDesc(0).GetDataType();
  DataType input_help_dtype = op.GetInputDesc(1).GetDataType();
  if (input_diagonal_dtype != input_help_dtype) {
    string err_msg1 = ConcatString("the inputs of diagonal and help should be the same dtype! input_diagonal_dtype:",input_diagonal_dtype, ", input_diagonal_dtype:",input_diagonal_dtype);
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixDiagD, MatrixDiagDInferShape);
VERIFY_FUNC_REG(MatrixDiagD, MatrixDiagDVerify);
// ----------------MatrixDiagD END----------------

// ----------------MatrixDiagPart--------------------
IMPLEMT_COMMON_INFERFUNC(MatrixDiagPartInferShape) {
  Shape shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  Format input_format = op.GetInputDesc("x").GetFormat();
  std::vector<int64_t> dim_vector;
  int64_t dimsInput_1 = shape.GetDimNum() - 1;
  int64_t dimsInput_2 = shape.GetDimNum() - 2;
  int64_t dimNums_1 = shape.GetDim(dimsInput_1);
  int64_t dimNums_2 = shape.GetDim(dimsInput_2);
  if (dimNums_1 > dimNums_2) {
    for (size_t i = 0; i < shape.GetDimNum() - 1; i++) {
      dim_vector.push_back(shape.GetDim(i));
    }
  } else {
    for (size_t i = 0; i < shape.GetDimNum() - 2; i++) {
      dim_vector.push_back(shape.GetDim(i));
    }
    dim_vector.push_back(dimNums_1);
  }
  Shape output_shape(dim_vector);
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(output_shape);
  td.SetDataType(input_dtype);
  td.SetFormat(input_format);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MatrixDiagPart, MatrixDiagPartVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixDiagPart, MatrixDiagPartInferShape);
VERIFY_FUNC_REG(MatrixDiagPart, MatrixDiagPartVerify);
// ------------------MatrixDiagPart END---------------------

// ----------------MatrixDiagPartD--------------------
IMPLEMT_COMMON_INFERFUNC(MatrixDiagPartDInferShape) {
  Shape shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  Format input_format = op.GetInputDesc("x").GetFormat();
  std::vector<int64_t> dim_vector;
  int64_t dimsInput_1 = shape.GetDimNum() - 1;
  int64_t dimsInput_2 = shape.GetDimNum() - 2;
  int64_t dimNums_1 = shape.GetDim(dimsInput_1);
  int64_t dimNums_2 = shape.GetDim(dimsInput_2);
  if (dimNums_1 > dimNums_2) {
    for (size_t i = 0; i < shape.GetDimNum() - 1; i++) {
      dim_vector.push_back(shape.GetDim(i));
    }
  } else {
    for (size_t i = 0; i < shape.GetDimNum() - 2; i++) {
      dim_vector.push_back(shape.GetDim(i));
    }
    dim_vector.push_back(dimNums_1);
  }
  Shape output_shape(dim_vector);
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(output_shape);
  td.SetDataType(input_dtype);
  td.SetFormat(input_format);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MatrixDiagPartD, MatrixDiagPartDVerify) {
  DataType input_diagonal_dtype = op.GetInputDesc(0).GetDataType();
  DataType input_help_dtype = op.GetInputDesc(1).GetDataType();
  if (input_diagonal_dtype != input_help_dtype) {
    string err_msg1 = ConcatString("the inputs of diagonal and help should be the same dtype! input_diagonal_dtype:",input_diagonal_dtype, ", input_help_dtype:",input_help_dtype);
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixDiagPartD, MatrixDiagPartDInferShape);
VERIFY_FUNC_REG(MatrixDiagPartD, MatrixDiagPartDVerify);
// ------------------MatrixDiagPart ENDD---------------------

// ---------------MatrixSetDiag--------------
IMPLEMT_COMMON_INFERFUNC(MatrixSetDiagInferShape) {
  Shape input_shape = op.GetInputDesc(0).GetShape();
  DataType input_dtype = op.GetInputDesc(0).GetDataType();
  TensorDesc td = op.GetOutputDesc(0);
  td.SetShape(ge::Shape(input_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MatrixSetDiag, MatrixSetDiagVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixSetDiag, MatrixSetDiagInferShape);
VERIFY_FUNC_REG(MatrixSetDiag, MatrixSetDiagVerify);
// ----------------MatrixSetDiag END----------------

// ---------------MatrixSetDiagD--------------
IMPLEMT_COMMON_INFERFUNC(MatrixSetDiagDInferShape) {
  Shape input_shape = op.GetInputDesc(0).GetShape();
  DataType input_dtype = op.GetInputDesc(0).GetDataType();
  TensorDesc td = op.GetOutputDesc(0);
  td.SetShape(ge::Shape(input_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MatrixSetDiagD, MatrixSetDiagDVerify) {
  DataType input_matrix_dtype = op.GetInputDesc(0).GetDataType();
  DataType input_diagonal_dtype = op.GetInputDesc(1).GetDataType();
  DataType input_help_dtype = op.GetInputDesc(2).GetDataType();
  if ((input_matrix_dtype != input_diagonal_dtype) || (input_matrix_dtype != input_help_dtype)) {
    string err_msg1 = ConcatString("the inputs of matrix and diagonal should be the same dtype! input_matrix_dtype:",input_matrix_dtype, ", input_diagonal_dtype:",input_diagonal_dtype,", input_help_dtype:",input_help_dtype);
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixSetDiagD, MatrixSetDiagDInferShape);
VERIFY_FUNC_REG(MatrixSetDiagD, MatrixSetDiagDVerify);
// ----------------MatrixSetDiag ENDD----------------

// -----------------ScatterNdUpdate-----------------
IMPLEMT_VERIFIER(ScatterNdUpdate, ScatterNdUpdateVerify) {
  if (!CheckTwoInputDtypeSame(op, "var", "updates")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ScatterNdUpdateInferShape) {
  // main part of shape infer
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ge::GeShape var_shape = op_desc->MutableInputDesc("var")->GetShape();
  std::vector<std::pair<int64_t, int64_t>> var_shape_range;
  op_desc->MutableInputDesc("var")->GetShapeRange(var_shape_range);
  DataType input_dtype = op_desc->MutableInputDesc("var")->GetDataType();
  GeTensorDescPtr td = op_desc->MutableOutputDesc("var");
  td->SetShape(var_shape);
  td->SetDataType(input_dtype);
  td->SetShapeRange(var_shape_range);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ScatterNdUpdate, ScatterNdUpdateInferShape);
VERIFY_FUNC_REG(ScatterNdUpdate, ScatterNdUpdateVerify);
// -------------------ScatterNdUpdate END----------------

// -----------------TensorScatterUpdate-----------------
IMPLEMT_VERIFIER(TensorScatterUpdate, TensorScatterUpdateVerify) {
  if (!CheckTwoInputDtypeSame(op, "x", "updates")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(TensorScatterUpdate, TensorScatterUpdateInferShape) {
  Shape var_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(ge::Shape(var_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorScatterUpdate, TensorScatterUpdateInferShape);
VERIFY_FUNC_REG(TensorScatterUpdate, TensorScatterUpdateVerify);
// -------------------TensorScatterUpdate END----------------

// -------------------ScatterElements------------------------
IMPLEMT_VERIFIER(ScatterElements, ScatterElementsVerify) {
  if (!CheckTwoInputDtypeSame(op, "data", "updates")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(ScatterElements, ScatterElementsInferShape) {
  Shape data_shape = op.GetInputDesc("data").GetShape();
  DataType input_dtype = op.GetInputDesc("data").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(ge::Shape(data_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ScatterElements, ScatterElementsInferShape);
VERIFY_FUNC_REG(ScatterElements, ScatterElementsVerify);
// -------------------ScatterElements END--------------------

// ------------------ScatterAdd---------------------
IMPLEMT_VERIFIER(ScatterAdd, ScatterAddVerify) {
  if (!CheckTwoInputDtypeSame(op, "var", "updates")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ScatterAddInferShape) {
  // main part of shape infer
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ge::GeShape var_shape = op_desc->MutableInputDesc("var")->GetShape();
  std::vector<std::pair<int64_t, int64_t>> var_shape_range;
  op_desc->MutableInputDesc("var")->GetShapeRange(var_shape_range);
  DataType input_dtype = op_desc->MutableInputDesc("var")->GetDataType();
  GeTensorDescPtr td = op_desc->MutableOutputDesc("var");
  td->SetShape(var_shape);
  td->SetDataType(input_dtype);
  td->SetShapeRange(var_shape_range);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ScatterAdd, ScatterAddInferShape);
VERIFY_FUNC_REG(ScatterAdd, ScatterAddVerify);
// --------------ScatterAdd END------------------

// ------------------ScatterScalar--------------------
IMPLEMT_COMMON_INFERFUNC(ScatterScalarInferShape) {
  // main part of shape infer
  Shape index_shape = op.GetInputDesc("index").GetShape();
  DataType index_dtype = op.GetInputDesc("index").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(ge::Shape(index_shape));
  td.SetDataType(index_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ScatterScalar, ScatterScalarInferShape);
// --------------ScatterScalar END------------------

// ------------------ScatterTensor---------------------
IMPLEMT_VERIFIER(ScatterTensor, ScatterTensorVerify) {
  if (!CheckTwoInputDtypeSame(op, "index", "src")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ScatterTensorInferShape) {
  // main part of shape infer
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ge::GeShape index_shape = op_desc->MutableInputDesc("index")->GetShape();
  DataType input_dtype = op_desc->MutableInputDesc("index")->GetDataType();
  GeTensorDescPtr td = op_desc->MutableOutputDesc("y");
  td->SetShape(index_shape);
  td->SetDataType(input_dtype);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ScatterTensor, ScatterTensorInferShape);
VERIFY_FUNC_REG(ScatterTensor, ScatterTensorVerify);
// --------------ScatterTensor END------------------

// ------------------ScatterDiv---------------------
IMPLEMT_VERIFIER(ScatterDiv, ScatterDivVerify) {
  if (!CheckTwoInputDtypeSame(op, "var", "updates")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ScatterDivInferShape) {
  // main part of shape infer
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ge::GeShape var_shape = op_desc->MutableInputDesc("var")->GetShape();
  std::vector<std::pair<int64_t, int64_t>> var_shape_range;
  op_desc->MutableInputDesc("var")->GetShapeRange(var_shape_range);
  DataType input_dtype = op_desc->MutableInputDesc("var")->GetDataType();
  GeTensorDescPtr td = op_desc->MutableOutputDesc("var");
  td->SetShape(var_shape);
  td->SetDataType(input_dtype);
  td->SetShapeRange(var_shape_range);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ScatterDiv, ScatterDivInferShape);
VERIFY_FUNC_REG(ScatterDiv, ScatterDivVerify);
// --------------ScatterDiv END------------------

// ----------------ScatterNdAdd------------
IMPLEMT_VERIFIER(ScatterNdAdd, ScatterNdAddVerify) {
  if (!CheckTwoInputDtypeSame(op, "var", "updates")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ScatterNdAddInferShape) {
  // main part of shape infer
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ge::GeShape var_shape = op_desc->MutableInputDesc("var")->GetShape();
  std::vector<std::pair<int64_t, int64_t>> var_shape_range;
  op_desc->MutableInputDesc("var")->GetShapeRange(var_shape_range);
  DataType input_dtype = op_desc->MutableInputDesc("var")->GetDataType();
  GeTensorDescPtr td = op_desc->MutableOutputDesc("var");
  td->SetShape(var_shape);
  td->SetDataType(input_dtype);
  td->SetShapeRange(var_shape_range);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ScatterNdAdd, ScatterNdAddInferShape);
VERIFY_FUNC_REG(ScatterNdAdd, ScatterNdAddVerify);
// ------------------ScatterNdAdd END------------------

// ----------------TensorScatterAdd------------
IMPLEMT_VERIFIER(TensorScatterAdd, TensorScatterAddVerify) {
  if (!CheckTwoInputDtypeSame(op, "x", "updates")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(TensorScatterAddInferShape) {
  Shape var_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(ge::Shape(var_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(TensorScatterAdd, TensorScatterAddInferShape);
VERIFY_FUNC_REG(TensorScatterAdd, TensorScatterAddVerify);
// ------------------TensorScatterAdd END------------------

// -------------------ScatterNdSub-------------------
IMPLEMT_VERIFIER(ScatterNdSub, ScatterNdSubVerify) {
  if (!CheckTwoInputDtypeSame(op, "var", "updates")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ScatterNdSubInferShape) {
  // main part of shape infer
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ge::GeShape var_shape = op_desc->MutableInputDesc("var")->GetShape();
  std::vector<std::pair<int64_t, int64_t>> var_shape_range;
  op_desc->MutableInputDesc("var")->GetShapeRange(var_shape_range);
  DataType input_dtype = op_desc->MutableInputDesc("var")->GetDataType();
  GeTensorDescPtr td = op_desc->MutableOutputDesc("var");
  td->SetShape(var_shape);
  td->SetDataType(input_dtype);
  td->SetShapeRange(var_shape_range);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ScatterNdSub, ScatterNdSubInferShape);
VERIFY_FUNC_REG(ScatterNdSub, ScatterNdSubVerify);
// ---------------ScatterNdSub END-----------------

// -------------------TensorScatterSub-------------------
IMPLEMT_VERIFIER(TensorScatterSub, TensorScatterSubVerify) {
  if (!CheckTwoInputDtypeSame(op, "x", "updates")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(TensorScatterSubInferShape) {
  Shape var_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(ge::Shape(var_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(TensorScatterSub, TensorScatterSubInferShape);
VERIFY_FUNC_REG(TensorScatterSub, TensorScatterSubVerify);
// ---------------TensorScatterSub END-----------------

// ----------------ScatterSub---------------------
IMPLEMT_VERIFIER(ScatterSub, ScatterSubVerify) {
  if (!CheckTwoInputDtypeSame(op, "var", "updates")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ScatterSubInferShape) {
  // main part of shape infer
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ge::GeShape var_shape = op_desc->MutableInputDesc("var")->GetShape();
  std::vector<std::pair<int64_t, int64_t>> var_shape_range;
  op_desc->MutableInputDesc("var")->GetShapeRange(var_shape_range);
  DataType input_dtype = op_desc->MutableInputDesc("var")->GetDataType();
  GeTensorDescPtr td = op_desc->MutableOutputDesc("var");
  td->SetShape(var_shape);
  td->SetDataType(input_dtype);
  td->SetShapeRange(var_shape_range);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ScatterSub, ScatterSubInferShape);
VERIFY_FUNC_REG(ScatterSub, ScatterSubVerify);
// --------------------ScatterSub END-----------------

// ---------------ConfusionMatrix-----------------
IMPLEMT_COMMON_INFERFUNC(ConfusionMatrixInferShape) {
  int64_t num_classes;
  auto output_dtype = DT_FLOAT;
  std::string output_dtype_str = "float32";
  if (op.GetAttr("num_classes", num_classes) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("num_classes");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (op.GetAttr("dtype", output_dtype_str) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("dtype");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (output_dtype_str == "float32") {
    output_dtype = DT_FLOAT;
  } else if (output_dtype_str == "int32") {
    output_dtype = DT_INT32;
  } else if (output_dtype_str == "int8") {
    output_dtype = DT_INT8;
  } else if (output_dtype_str == "float16") {
    output_dtype = DT_FLOAT16;
  } else if (output_dtype_str == "uint8") {
    output_dtype = DT_UINT8;
  } else {
    string expected_data_type_list = ConcatString("float32, int32, int8, float16, uint8");
    std::string err_msg = GetInputDtypeNotSupportErrMsg("dtype", expected_data_type_list, output_dtype_str);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<int64_t> out_shape;
  out_shape.push_back(num_classes);
  out_shape.push_back(num_classes);
  vector<int64_t> y_shape(out_shape);
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(ge::Shape(y_shape));
  td.SetDataType((DataType)output_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ConfusionMatrix, ConfusionMatrixInferShape);
// ------------------ConfusionMatrix END------------------

// ------------------ScatterMul---------------------
IMPLEMT_VERIFIER(ScatterMul, ScatterMulVerify) {
  if (!CheckTwoInputDtypeSame(op, "var", "updates")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ScatterMulInferShape) {
  // main part of shape infer
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ge::GeShape var_shape = op_desc->MutableInputDesc("var")->GetShape();
  std::vector<std::pair<int64_t, int64_t>> var_shape_range;
  op_desc->MutableInputDesc("var")->GetShapeRange(var_shape_range);
  DataType input_dtype = op_desc->MutableInputDesc("var")->GetDataType();
  GeTensorDescPtr td = op_desc->MutableOutputDesc("var");
  td->SetShape(var_shape);
  td->SetDataType(input_dtype);
  td->SetShapeRange(var_shape_range);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ScatterMul, ScatterMulInferShape);
VERIFY_FUNC_REG(ScatterMul, ScatterMulVerify);
// --------------ScatterMul END------------------

// ------------------ScatterUpdate---------------------
IMPLEMT_VERIFIER(ScatterUpdate, ScatterUpdateVerify) {
  if (!CheckTwoInputDtypeSame(op, "var", "updates")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ScatterUpdateInferShape) {
  // main part of shape infer
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ge::GeShape var_shape = op_desc->MutableInputDesc("var")->GetShape();
  std::vector<std::pair<int64_t, int64_t>> var_shape_range;
  op_desc->MutableInputDesc("var")->GetShapeRange(var_shape_range);
  DataType input_dtype = op_desc->MutableInputDesc("var")->GetDataType();
  GeTensorDescPtr td = op_desc->MutableOutputDesc("var");
  td->SetShape(var_shape);
  td->SetDataType(input_dtype);
  td->SetShapeRange(var_shape_range);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ScatterUpdate, ScatterUpdateInferShape);
VERIFY_FUNC_REG(ScatterUpdate, ScatterUpdateVerify);
// --------------ScatterUpdate END------------------

// ------------------ScatterMin---------------------
IMPLEMT_VERIFIER(ScatterMin, ScatterMinVerify) {
  if (!CheckTwoInputDtypeSame(op, "var", "updates")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ScatterMinInferShape) {
  // main part of shape infer
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ge::GeShape var_shape = op_desc->MutableInputDesc("var")->GetShape();
  std::vector<std::pair<int64_t, int64_t>> var_shape_range;
  op_desc->MutableInputDesc("var")->GetShapeRange(var_shape_range);
  DataType input_dtype = op_desc->MutableInputDesc("var")->GetDataType();
  GeTensorDescPtr td = op_desc->MutableOutputDesc("var");
  td->SetShape(var_shape);
  td->SetDataType(input_dtype);
  td->SetShapeRange(var_shape_range);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ScatterMin, ScatterMinInferShape);
VERIFY_FUNC_REG(ScatterMin, ScatterMinVerify);
// --------------ScatterMin END------------------

// ------------------ScatterMax---------------------
IMPLEMT_VERIFIER(ScatterMax, ScatterMaxVerify) {
  if (!CheckTwoInputDtypeSame(op, "var", "updates")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ScatterMaxInferShape) {
  // main part of shape infer
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ge::GeShape var_shape = op_desc->MutableInputDesc("var")->GetShape();
  std::vector<std::pair<int64_t, int64_t>> var_shape_range;
  op_desc->MutableInputDesc("var")->GetShapeRange(var_shape_range);
  DataType input_dtype = op_desc->MutableInputDesc("var")->GetDataType();
  GeTensorDescPtr td = op_desc->MutableOutputDesc("var");
  td->SetShape(var_shape);
  td->SetDataType(input_dtype);
  td->SetShapeRange(var_shape_range);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ScatterMax, ScatterMaxInferShape);
VERIFY_FUNC_REG(ScatterMax, ScatterMaxVerify);
// --------------ScatterMax END------------------

bool FullyDefined(Shape s) {
  auto dims = s.GetDims();
  if (dims == ge::UNKNOWN_SHAPE) {
    return false;
  }
  for (auto& dim : dims) {
    if (dim == ge::UNKNOWN_DIM) {
      return false;
    }
  }
  return true;
}

IMPLEMT_COMMON_INFERFUNC(MatrixDiagPartV2InferShape) {
  Shape input_shape;
  auto input_tensor_desc = op.GetInputDesc(0);
  if (WithRankAtLeast(input_tensor_desc, 2, input_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
      "failed to call WithRankAtLeast function, ",
      "input[input] rank must be at least 2D, but got rank[",
      op.GetInputDesc(0).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  Shape k_shape;
  auto k_tensor_desc = op.GetInputDesc(1);
  if (WithRankAtMost(k_tensor_desc, 1, k_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
      "failed to call WithRankAtMost function, ",
      "input[k] rank must be at most 1D, but got rank[",
      op.GetInputDesc(1).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  Shape padding_value_shape;
  auto padding_value_tensor_desc = op.GetInputDesc(2);
  if (WithRank(padding_value_tensor_desc, 0, padding_value_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call WithRank function, ",
      "input[padding_value] rank must be 0, but got rank[",
      op.GetInputDesc(2).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc output_desc = op.GetOutputDesc("diagonal");

  Tensor k_tensor;
  std::vector<std::string> input_infer_depends = {"k"};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends(input_infer_depends);
  if (op.GetInputConstData("k", k_tensor) != GRAPH_SUCCESS || !RankKnown(input_shape) || !FullyDefined(k_shape)) {
    output_desc.SetShape(Shape(ge::UNKNOWN_SHAPE));
    output_desc.SetDataType(input_tensor_desc.GetDataType());
    (void)op.UpdateOutputDesc("diagonal", output_desc);
    return GRAPH_SUCCESS;
  }

  int32_t lower_diag_index = 0;
  int32_t upper_diag_index = 0;
  if (k_shape.GetDimNum() == 0) {
    lower_diag_index = *(reinterpret_cast<int32_t*>(k_tensor.GetData()));
    upper_diag_index = lower_diag_index;
  } else {
    auto k_dims = k_shape.GetDims();
    int64_t num_elements = k_dims[0];
    if (num_elements == 1) {
      int32_t* data = reinterpret_cast<int32_t*>(k_tensor.GetData());
      lower_diag_index = *data;
      upper_diag_index = lower_diag_index;
    } else if (num_elements == 2) {
      int32_t* data = reinterpret_cast<int32_t*>(k_tensor.GetData());
      lower_diag_index = *(data);
      upper_diag_index = *(data + 1);
    } else {
      std::string err_msg = ConcatString(
        "the input [k] must be scalar or a vector with one or two elements, ",
        "but it has [", num_elements, "] elements. ");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_PARAM_INVALID;
    }
  }

  if (lower_diag_index > upper_diag_index) {
    std::string err_msg = ConcatString(
      "the variable [lower_diag_index] of input [k] 1th value[",
      lower_diag_index,  "] must be  not greater than [upper_diag_index] ",
      "of input [k] 2th value[", upper_diag_index, "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_PARAM_INVALID;
  }

  auto input_dims = input_shape.GetDims();
  const int32_t input_rank = input_shape.GetDimNum();
  const int32_t num_rows = input_dims[input_rank - 2];
  const int32_t num_cols = input_dims[input_rank - 1];
  int64_t max_diag_len = ge::UNKNOWN_DIM;
  if (num_rows != ge::UNKNOWN_DIM && num_cols != ge::UNKNOWN_DIM) {
    if (lower_diag_index != 0 && (-num_rows >= lower_diag_index || lower_diag_index >= num_cols)) {
      std::string err_msg = ConcatString(
        "the variable [lower_diag_index] of input [k] value[",
        lower_diag_index, "] is illegal, ",
        "should be 0 or in range(", -num_rows, ", ", num_cols, ")");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_PARAM_INVALID;
    }
    if (upper_diag_index != 0 && (-num_rows >= upper_diag_index || upper_diag_index >= num_cols)) {
      std::string err_msg = ConcatString(
        "the variable [upper_diag_index] of input [k] value[",
        upper_diag_index, "] is illegal, ",
        "should be 0 or in range(", -num_rows, ", ", num_cols, ")");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_PARAM_INVALID;
    }
    max_diag_len = std::min(num_rows + std::min(upper_diag_index, 0), num_cols - std::max(lower_diag_index, 0));
  }

  std::vector<int64_t> output_dims;
  for (int32_t i = 0; i < input_rank - 2; ++i) {
    output_dims.push_back(input_dims[i]);
  }
  if (lower_diag_index < upper_diag_index) {
    output_dims.push_back(upper_diag_index - lower_diag_index + 1);
  }
  output_dims.push_back(max_diag_len);
  output_desc.SetShape(Shape(output_dims));
  output_desc.SetDataType(input_tensor_desc.GetDataType());
  (void)op.UpdateOutputDesc("diagonal", output_desc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixDiagPartV2, MatrixDiagPartV2InferShape);

IMPLEMT_COMMON_INFERFUNC(MatrixSetDiagV2InferShape) {
  Shape input_shape;
  auto input_tensor_desc = op.GetInputDesc(0);
  if (WithRankAtLeast(input_tensor_desc, 2, input_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
      "failed to call WithRankAtLeast function, ",
      "input[input] rank must be at least 2D, but got rank[",
      op.GetInputDesc(0).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  Shape diagonal_shape;
  auto diagonal_tensor_desc = op.GetInputDesc(1);
  if (WithRankAtLeast(diagonal_tensor_desc, 1, diagonal_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
      "failed to call WithRankAtLeast function, ",
      "input[diagonal] rank must be at least 1D, but got rank[",
      op.GetInputDesc(1).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  Shape k_shape;
  auto k_tensor_desc = op.GetInputDesc(2);
  if (WithRankAtMost(k_tensor_desc, 1, k_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
      "failed to call WithRankAtMost function, ",
      "input[k] rank must be at most 1D, but got rank[",
      op.GetInputDesc(2).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int32_t lower_diag_index = 0;
  int32_t upper_diag_index = 0;
  bool k_index_known = false;
  Tensor k_tensor;
  if (op.GetInputConstData("k", k_tensor) == GRAPH_SUCCESS && FullyDefined(k_shape)) {
    k_index_known = true;
    if (k_shape.GetDimNum() == 0) {
      lower_diag_index = *(reinterpret_cast<int32_t*>(k_tensor.GetData()));
      upper_diag_index = lower_diag_index;
    } else {
      auto k_dims = k_shape.GetDims();
      int64_t num_elements = k_dims[0];
      if (num_elements == 1) {
        int32_t* data = reinterpret_cast<int32_t*>(k_tensor.GetData());
        lower_diag_index = *data;
        upper_diag_index = lower_diag_index;
      } else if (num_elements == 2) {
        int32_t* data = reinterpret_cast<int32_t*>(k_tensor.GetData());
        lower_diag_index = *(data);
        upper_diag_index = *(data + 1);
      } else {
        std::string err_msg = ConcatString(
          "the input[k] must be a scalar or vector with one or two elements, ",
          "but it has [", num_elements, "] elements. ");
        AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
        return GRAPH_PARAM_INVALID;
      }
    }

    if (lower_diag_index > upper_diag_index) {
      std::string err_msg = ConcatString(
        "the variable [lower_diag_index] of input[k] 1th value[",
        lower_diag_index, "] must be ",
        "not greater than [upper_diag_index] of input[k] 2th value[",
        upper_diag_index, "]");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_PARAM_INVALID;
    }
  }

  if (RankKnown(input_shape)) {
    auto input_rank = input_shape.GetDimNum();
    if (k_index_known) {
      if (WithRank(diagonal_tensor_desc, (lower_diag_index == upper_diag_index ? input_rank - 1 : input_rank),
                   diagonal_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        std::string err_msg = ConcatString(
          "failed to call WithRank function, ",
          "input[diagonal] rank must be [",
          (lower_diag_index == upper_diag_index ? input_rank - 1 : input_rank),
          "], but got rank[", diagonal_tensor_desc.GetShape().GetDimNum(), "]");
        AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
        return GRAPH_FAILED;
      } else {
        if (WithRankAtLeast(diagonal_tensor_desc, input_rank - 1, diagonal_shape, op.GetName().c_str()) !=
            GRAPH_SUCCESS) {
          std::string err_msg = ConcatString(
            "failed to call WithRankAtLeast function, ",
            "input[diagonal] rank must be at least ", input_rank - 1,
            "D, but got rank[", diagonal_tensor_desc.GetShape().GetDimNum(), "]");
          AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
          return GRAPH_FAILED;
        }

        if (WithRankAtMost(diagonal_tensor_desc, input_rank, diagonal_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
          std::string err_msg = ConcatString(
            "failed to call WithRankAtMost function, ",
            "input[diagonal] rank must be at most ",
            input_rank,"D, but got rank[",
            diagonal_tensor_desc.GetShape().GetDimNum(), "]");
          AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
          return GRAPH_FAILED;
        }
      }

      auto input_dims = input_shape.GetDims();
      const int32_t num_rows = input_dims[input_rank - 2];
      const int32_t num_cols = input_dims[input_rank - 1];
      if (num_rows != ge::UNKNOWN_DIM && num_cols != ge::UNKNOWN_DIM) {
        if (lower_diag_index != 0 && (-num_rows >= lower_diag_index || lower_diag_index >= num_cols)) {
          std::string err_msg = ConcatString(
            "the variable [lower_diag_index] of input[k] value[",
            lower_diag_index, "] is illegal, ",
            "should be 0 or in range(", -num_rows, ", ", num_cols, ")");
          AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
          return GRAPH_PARAM_INVALID;
        }
        if (upper_diag_index != 0 && (-num_rows >= upper_diag_index || upper_diag_index >= num_cols)) {
          std::string err_msg = ConcatString(
            "the variable [upper_diag_index] of input[k] value[",
            upper_diag_index, "] is illegal, ",
            "should be 0 or in range(", -num_rows, ", ", num_cols, ")");
          AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
          return GRAPH_PARAM_INVALID;
        }
      }
    }
  }

  auto output_desc = op.GetOutputDesc("output");
  Shape output_shape = input_shape;
  if (RankKnown(diagonal_shape) && !FullyDefined(input_shape)) {
    Shape diagonal_prefix_shape;
    if (SubShape(diagonal_shape, 0, (lower_diag_index == upper_diag_index ? -1 : -2), 1, diagonal_prefix_shape,
                 op.GetName().c_str()) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString(
        "failed to call SubShape function, input[diagonal] shape",
        DebugString(diagonal_shape.GetDims()),
        ", end[", (lower_diag_index == upper_diag_index ? -1 : -2),
        "] is invaild");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }

    if (Concatenate(diagonal_prefix_shape, UnknownShapeOfRank(2), diagonal_shape) != GRAPH_SUCCESS) {
      std:: string err_msg = ConcatString("failed to call Concatenate function ",
        "to concatenate prefix diagonal shape",
        DebugString(diagonal_prefix_shape.GetDims()),
        " and 2D unknown_shape");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }

    if (Merge(input_shape, diagonal_shape, output_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString(
        "failed to call Merge function to merge the 0th input[input] shape",
        DebugString(input_shape.GetDims()),
        " and the 1st input[diagonal]'s shape",
        DebugString(diagonal_shape.GetDims()));
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }
  output_desc.SetShape(output_shape);
  output_desc.SetDataType(input_tensor_desc.GetDataType());
  (void)op.UpdateOutputDesc("output", output_desc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixSetDiagV2, MatrixSetDiagV2InferShape);

IMPLEMT_COMMON_INFERFUNC(MatrixDiagV2InferShape) {
  Shape diagonal_shape;
  auto diagonal_tensor_desc = op.GetInputDesc(0);
  if (WithRankAtLeast(diagonal_tensor_desc, 1, diagonal_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call WithRankAtLeast function, ",
      "input[diagonal] rank must be at least 1D, but got rank[",
      op.GetInputDesc(0).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  Shape k_shape;
  auto k_tensor_desc = op.GetInputDesc(1);
  if (WithRankAtMost(k_tensor_desc, 1, k_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
      "failed to call WithRankAtMost function, ",
      "input[k] rank must be at most 1D, but got rank[",
      op.GetInputDesc(1).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  Shape num_rows_shape;
  auto num_rows_tensor_desc = op.GetInputDesc(2);
  if (WithRank(num_rows_tensor_desc, 0, num_rows_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call WithRank function, ",
      "input[num_rows] rank must be 0, but got rank[",
      op.GetInputDesc(2).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  Shape num_cols_shape;
  auto num_cols_tensor_desc = op.GetInputDesc(3);
  if (WithRank(num_cols_tensor_desc, 0, num_cols_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call WithRank function, ",
      "input[num_cols] rank must be 0, but got rank[",
      op.GetInputDesc(3).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  Shape padding_value_shape;
  auto padding_value_tensor_desc = op.GetInputDesc(4);
  if (WithRank(padding_value_tensor_desc, 0, padding_value_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call WithRank function, ",
      "input[padding_value] rank must be 0, but got rank[",
      op.GetInputDesc(4).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  auto output_desc = op.GetOutputDesc("output");
  Tensor k_tensor;
  std::vector<std::string> input_infer_depends = {"k", "num_rows", "num_cols"};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends(input_infer_depends);
  if (op.GetInputConstData("k", k_tensor) != GRAPH_SUCCESS || !RankKnown(diagonal_shape) || !FullyDefined(k_shape)) {
    output_desc.SetShape(Shape(ge::UNKNOWN_SHAPE));
    output_desc.SetDataType(diagonal_tensor_desc.GetDataType());
    (void)op.UpdateOutputDesc("output", output_desc);
    return GRAPH_SUCCESS;
  }

  int32_t lower_diag_index = 0;
  int32_t upper_diag_index = 0;
  if (k_shape.GetDimNum() == 0) {
    lower_diag_index = *(reinterpret_cast<int32_t*>(k_tensor.GetData()));
    upper_diag_index = lower_diag_index;
  } else {
    auto k_dims = k_shape.GetDims();
    int32_t num_elements = k_dims[0];
    if (num_elements == 1) {
      int32_t* data = reinterpret_cast<int32_t*>(k_tensor.GetData());
      lower_diag_index = *data;
      upper_diag_index = lower_diag_index;
    } else if (num_elements == 2) {
      int32_t* data = reinterpret_cast<int32_t*>(k_tensor.GetData());
      lower_diag_index = *(data);
      upper_diag_index = *(data + 1);
    } else {
      std::string err_msg = ConcatString(
        "the input[k] must be scalar or a vector with one or two elements, ",
        "but it has [", num_elements, "] elements. ");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_PARAM_INVALID;
    }
  }

  if (lower_diag_index > upper_diag_index) {
    std::string err_msg = ConcatString(
      "the variable [lower_diag_index] of input[k] 1th value[",
      lower_diag_index, "] must be ",
      "not greater than [upper_diag_index] of input[k] 2th value[",
      upper_diag_index, "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_PARAM_INVALID;
  }

  auto diagonal_dims = diagonal_shape.GetDims();
  const int32_t diagonal_rank = diagonal_shape.GetDimNum();
  if (lower_diag_index < upper_diag_index) {
    const int64_t num_diags = diagonal_dims[diagonal_rank - 2];
    const int64_t other_dim = diagonal_dims[diagonal_rank - 1];
    if (num_diags != (upper_diag_index - lower_diag_index + 1)) {
      std::string err_msg = ConcatString(
        "the number of rows of [diagonal] doesn't match the number of ",
        "diagonals implied from [d_lower] and [d_upper].",
        " num_diags is [" , num_diags, "], d_lower is [", lower_diag_index,
        "], d_upper is [", upper_diag_index, "], other_dim is [", other_dim, "]");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_PARAM_INVALID;
    }
  }

  int32_t num_rows = ge::UNKNOWN_DIM;
  Tensor num_rows_tensor;
  if (op.GetInputConstData("num_rows", num_rows_tensor) == GRAPH_SUCCESS) {
    num_rows = *(reinterpret_cast<int32_t*>(num_rows_tensor.GetData()));
  }

  int32_t num_cols = ge::UNKNOWN_DIM;
  Tensor num_cols_tensor;
  if (op.GetInputConstData("num_cols", num_cols_tensor) == GRAPH_SUCCESS) {
    num_cols = *(reinterpret_cast<int32_t*>(num_cols_tensor.GetData()));
  }

  const int32_t max_diag_len = diagonal_dims[diagonal_rank - 1];
  const int32_t min_num_rows = max_diag_len - std::min(upper_diag_index, 0);
  const int32_t min_num_cols = max_diag_len + std::max(lower_diag_index, 0);

  if (num_rows == ge::UNKNOWN_DIM && num_cols == ge::UNKNOWN_DIM) {
    num_rows = std::max(min_num_rows, min_num_cols);
    num_cols = num_rows;
  }

  if (num_rows == ge::UNKNOWN_DIM) {
    num_rows = min_num_rows;
  } else if (num_rows < min_num_rows) {
    std::string err_msg = ConcatString(
      "input[num_rows] value[", num_rows, "] must be not less than ",
      "min_num_rows[", min_num_rows, "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_PARAM_INVALID;
  }

  if (num_cols == ge::UNKNOWN_DIM) {
    num_cols = min_num_cols;
  } else if (num_cols < min_num_cols) {
    std::string err_msg = ConcatString(
      "input[num_cols] value[", num_cols, "] must be not less than ",
      "min_num_cols[", min_num_cols, "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_PARAM_INVALID;
  }

  if (num_rows != min_num_rows && num_cols != min_num_cols &&
      min_num_rows != ge::UNKNOWN_DIM && min_num_cols != ge::UNKNOWN_DIM) {
    std::string err_msg = ConcatString(
      "input[num_rows] value[", num_rows, "] and ",
      "input[num_cols] value[", num_cols, "] ",
      "are not equal with min_num_rows[", min_num_rows, "] and min_num_cols[",
      min_num_rows, "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_PARAM_INVALID;
  }

  Shape output_shape;
  OP_LOGI(op.GetName().c_str(), "num_rows: ", num_rows, " num_cols: ", num_cols);
  if (lower_diag_index == upper_diag_index) {
    if (ReplaceDim(diagonal_shape, diagonal_rank - 1, num_rows, output_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString(
        "failed to call ReplaceDim function, replace input[diagonal] dim, ",
        "index[", diagonal_rank - 1, "],  replace value[", num_rows, "]");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    if (Concatenate(output_shape, Shape({num_cols}), output_shape) != GRAPH_SUCCESS) {
      std:: string err_msg = ConcatString(
        "failed to call Concatenate function, output shape",
        DebugString(output_shape.GetDims()),
        " and another shape", DebugString(Shape({num_cols}).GetDims()));
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  } else {
    if (ReplaceDim(diagonal_shape, diagonal_rank - 2, num_rows, output_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString(
        "failed to call ReplaceDim function, replace input[diagonal] dim, ",
        "index[", diagonal_rank - 2, "], replace value[", num_rows, "]");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    if (ReplaceDim(output_shape, diagonal_rank - 1, num_cols, output_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString(
        "failed to call ReplaceDim function, replace output[output] dim, ",
        "index[", diagonal_rank - 1, "], replace value[", num_cols, "]");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }
  output_desc.SetShape(output_shape);
  output_desc.SetDataType(diagonal_tensor_desc.GetDataType());
  (void)op.UpdateOutputDesc("output", output_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixDiagV2, MatrixDiagV2InferShape);

// ----------------IndexAdd Begin-------------------
bool InferShapeAndTypeIndexAdd(Operator& op) {
  TensorDesc output_desc = op.GetOutputDesc("var_out");
  DataType var_dtype = op.GetInputDesc("var").GetDataType();
  Format var_format = op.GetInputDesc("var").GetFormat();
  ge::Shape var_shape = op.GetInputDesc("var").GetShape();
  std::vector<int64_t> var_dims = var_shape.GetDims();

  ge::Shape updates_shape = op.GetInputDesc("updates").GetShape();
  std::vector<int64_t> updates_dims = updates_shape.GetDims();

  if (updates_dims != var_dims) {
    OP_LOGE(op.GetName().c_str(), "var_dims not equal updates dims");
    return GRAPH_FAILED;
  }

  ge::Shape output_shape = ge::Shape(var_dims);
  output_desc.SetShape(output_shape);
  output_desc.SetDataType(var_dtype);
  output_desc.SetFormat(var_format);
  op.UpdateOutputDesc("var_out", output_desc);

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(IndexAdd, IndexAddVerify) {
  DataType var_dtype = op.GetInputDesc("var").GetDataType();
  DataType indices_dtype = op.GetInputDesc("indices").GetDataType();
  DataType updates_dtype = op.GetInputDesc("updates").GetDataType();
  DataType var_out_dtype = op.GetInputDesc("var_out").GetDataType();
  if (var_dtype != var_out_dtype || var_dtype != updates_dtype) {
    OP_LOGE(op.GetName().c_str(),
            "The input shape of var var_out updates is equal, please check!");
    return GRAPH_FAILED;
  }
  if (indices_dtype != DT_INT32) {
    OP_LOGE(op.GetName().c_str(),
            "The input shape of indices is not int32, please check!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(IndexAddInferShape) {
  if (InferShapeAndTypeIndexAdd(op) == GRAPH_FAILED) {
    OP_LOGE(op.GetName().c_str(), "index_add infer shape failed!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(IndexAdd, IndexAddInferShape);
// Registered verify function
VERIFY_FUNC_REG(IndexAdd, IndexAddVerify);
// ----------------IndexAdd END---------------------

// ----------------IndexPut Begin-------------------
bool InferShapeAndTypeIndexPut(Operator& op) {
  TensorDesc output_desc = op.GetOutputDesc("y");
  DataType x1_dtype = op.GetInputDesc("x1").GetDataType();
  Format x1_format = op.GetInputDesc("x1").GetFormat();
  ge::Shape x1_shape = op.GetInputDesc("x1").GetShape();
  std::vector<int64_t> x1_dims = x1_shape.GetDims();

  ge::Shape x2_shape = op.GetInputDesc("x2").GetShape();
  std::vector<int64_t> x2_dims = x2_shape.GetDims();

  if (x2_dims != x1_dims) {
    OP_LOGE(op.GetName().c_str(), "x1_dims not equal x2_dims");
    return GRAPH_FAILED;
  }

  ge::Shape output_shape = ge::Shape(x1_dims);
  output_desc.SetShape(output_shape);
  output_desc.SetDataType(x1_dtype);
  output_desc.SetFormat(x1_format);
  op.UpdateOutputDesc("y", output_desc);

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(IndexPut, IndexPutVerify) {
  DataType x1_dtype = op.GetInputDesc("x1").GetDataType();
  DataType indices_dtype = op.GetInputDesc("indices").GetDataType();
  DataType x2_dtype = op.GetInputDesc("x2").GetDataType();
  if (x1_dtype != x2_dtype) {
    OP_LOGE(op.GetName().c_str(),
            "The input shape of x1 x2 y is equal, please check!");
    return GRAPH_FAILED;
  }
  if (indices_dtype != DT_INT32) {
    OP_LOGE(op.GetName().c_str(),
            "The input shape of indices is not int32, please check!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(IndexPutInferShape) {
  if (InferShapeAndTypeIndexPut(op) == GRAPH_FAILED) {
    OP_LOGE(op.GetName().c_str(), "index_put infer shape failed!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(IndexPut, IndexPutInferShape);
// Registered verify function
VERIFY_FUNC_REG(IndexPut, IndexPutVerify);
// ----------------IndexPut END---------------------

// ---------------Triu--------------
IMPLEMT_COMMON_INFERFUNC(TriuInferShape) {
  Shape input_shape = op.GetInputDesc(0).GetShape();
  DataType input_dtype = op.GetInputDesc(0).GetDataType();
  TensorDesc td = op.GetOutputDesc(0);
  td.SetShape(ge::Shape(input_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Triu, TriuVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Triu, TriuInferShape);
VERIFY_FUNC_REG(Triu, TriuVerify);
// ----------------Triu END----------------
// ---------------Tril--------------
IMPLEMT_COMMON_INFERFUNC(TrilInferShape) {
    Shape input_shape = op.GetInputDesc(0).GetShape();
    DataType input_dtype = op.GetInputDesc(0).GetDataType();
    TensorDesc td = op.GetOutputDesc(0);
    td.SetShape(ge::Shape(input_shape));
    td.SetDataType(input_dtype);
    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Tril, TrilVerify) { return GRAPH_SUCCESS; }

INFER_FUNC_REG(Tril, TrilInferShape);
VERIFY_FUNC_REG(Tril, TrilVerify);
// ----------------Tril END----------------
// ----------------Einsum-------------------
// check if there is an ellipsis
bool is_ellispis(std::string ori_str, std::string target) {
    std::string::size_type idx;
    idx = ori_str.find(target);
    if (idx == std::string::npos){
        return false;
    } else {
        return true;
    }
}

// remove spaces from the string
void trim_whitespace(std::string &s)
{
    int index = 0;
    if(!s.empty()) {
        while((index = s.find(' ',index)) != string::npos) {
            s.erase(index,1);
        }
    }
}

//  cut the string in half
void split_equ(std::string eqn,std::string &in_equ,std::string &out_equ) {
    size_t pos = 0;
    if ((pos = eqn.find("->")) != std::string::npos) {
        in_equ = eqn.substr(0,pos);
        out_equ = eqn.substr(pos+2);
    } else {
        return;
    }
}
// gets an array of input strings
void get_in_equ_list(std::string in_equ,vector<std::string> &in_equ_list) {
    std::stringstream equ_stream(in_equ);
    std::string term;
    while (! equ_stream.eof()) {
        std::getline(equ_stream, term, ',');
        in_equ_list.push_back(term);
    }
}

// Scenes with ellipses
void map_with_ellipsis(std::string equ_temp,vector<int64_t> equ_tensor_temp,
                        map<std::string, int64_t> &equ_map,
                        map<std::string, vector<int64_t>> &ellipsis_map) {
    int64_t equ_temp_size = equ_temp.size();
    int64_t equ_tensor_temp_size = equ_tensor_temp.size();
    int64_t ell_size = equ_tensor_temp_size + 3 - equ_temp_size;
    std::string dot = "...";
    vector<int64_t> ell_list;
    std::string equ_key = "A";

    if (equ_temp[0] == dot[0]) {
        for (int64_t i = 0; i < ell_size; i++){
            ell_list.push_back(equ_tensor_temp[i]);
            ellipsis_map[dot] = ell_list;
        }
        for (int64_t i = 0; i < (equ_tensor_temp_size - ell_size); i++) {
            equ_key[0] = equ_temp[i+3];
            equ_map[equ_key] = equ_tensor_temp[i+ell_size];
        }
    } else if (equ_temp[equ_temp_size - 1] == dot[0]){
        for (int64_t i = (equ_tensor_temp_size - ell_size); i < equ_tensor_temp_size; i++) {
            ell_list.push_back(equ_tensor_temp[i]);
            ellipsis_map[dot] = ell_list;
        }
        for (int64_t i = 0; i < (equ_tensor_temp_size - ell_size); i++) {
            equ_key[0] = equ_temp[i];
            equ_map[equ_key] = equ_tensor_temp[i];
        }
    } else {
        int64_t start_index = equ_temp.find_first_of(".");
        for (int64_t i = 0; i < start_index; i++) {
            equ_key[0] = equ_temp[i];
            equ_map[equ_key] = equ_tensor_temp[i];
        }
        for (int64_t j = start_index; j < (start_index+ell_size); j++) {
            ell_list.push_back(equ_tensor_temp[j]);
            ellipsis_map[dot] = ell_list;
        }
        for (int64_t k = 0; k < (equ_tensor_temp_size - ell_size - start_index); k++) {
            equ_key[0] = equ_temp[k+start_index+3];
            equ_map[equ_key] = equ_tensor_temp[k+start_index+ell_size];
        }
    }
}

// Scenes without ellipses
void map_without_ellipsis(std::string equ_temp,vector<int64_t> equ_tensor_temp,
                        map<std::string, int64_t> &equ_map) {
    int64_t equ_temp_size = equ_temp.size();
    int64_t equ_tensor_temp_size = equ_tensor_temp.size();
    if (equ_temp_size != equ_tensor_temp_size) {
    }
    for (int64_t i = 0; i < equ_temp_size; i++) {
        std::string equ_key = "A";
        equ_key[0] = equ_temp[i];
        equ_map[equ_key] = equ_tensor_temp[i];
    }
}

// Output shape with ellipsis
void output_with_ellipsis(vector<int64_t> &output_shape,std::string out_equ,
                            map<std::string, int64_t> equ_map,
                            map<std::string, vector<int64_t>> ellipsis_map) {
    int64_t out_equ_size = out_equ.size();
    std::string dot = "...";
    vector<int64_t> ell_list;
    ell_list = ellipsis_map[dot];
    int64_t ell_size = ell_list.size();
    std::string equ_key = "A";
    if (out_equ[0] == dot[0]) {
        for (int64_t i = 0; i < ell_size; i++) {
            output_shape.push_back(ell_list[i]);
        }
        for (int64_t i = 3; i < out_equ_size; i++) {
            equ_key[0] = out_equ[i];
            output_shape.push_back(equ_map[equ_key]);
        }
    } else if (out_equ[out_equ_size - 1] == dot[0]){
        for (int64_t i = 0; i < (out_equ_size-3); i++) {
            equ_key[0] = out_equ[i];
            output_shape.push_back(equ_map[equ_key]);
        }
        for (int64_t i = 0; i < ell_size; i++) {
            output_shape.push_back(ell_list[i]);
        }
    } else {
        int64_t start_index = out_equ.find_first_of(".");
        for (int64_t i = 0; i < start_index; i++) {
            equ_key[0] = out_equ[i];
            output_shape.push_back(equ_map[equ_key]);
        }
        for (int64_t i = 0; i < ell_size; i++) {
            output_shape.push_back(ell_list[i]);
        }
        for (int64_t i = (start_index+3); i < out_equ_size; i++) {
            equ_key[0] = out_equ[i];
            output_shape.push_back(equ_map[equ_key]);
        }
    }
}

// Output shape without ellipsis
void output_without_ellipsis(vector<int64_t> &output_shape,std::string out_equ,
                            map<std::string, int64_t> equ_map) {
    int64_t out_equ_size = out_equ.size();
    int64_t val;
    std::string equ_key = "A";
    for (int64_t i = 0; i < out_equ_size; i++)
    {   equ_key[0] = out_equ[i];
        val = equ_map[equ_key];
        output_shape.push_back(val);
    }

}

// einsum infer shape
void einsum_infer_shape(std::string eqn, vector<vector<int64_t>> tensor_list,vector<int64_t> &output_shape) {

    trim_whitespace(eqn);
    int64_t tensor_size = tensor_list.size();
// define two maps to hold the corresponding characters
    map<std::string, int64_t> equ_map;
    map<std::string, vector<int64_t>> ellipsis_map;
    std::string in_equ;
    std::string out_equ;
// Split string
    split_equ(eqn,in_equ,out_equ);
// gets a list of input strings
    vector<std::string> in_equ_list;
    get_in_equ_list(in_equ,in_equ_list);

    int64_t in_equ_size = in_equ_list.size();
    if (in_equ_size != tensor_size) {
        return;
    }
    std::string equ_temp;
    vector<int64_t> equ_tensor_temp;
    std::string targets = "...";
    for (int64_t i = 0; i < in_equ_size; i++)
    {
        equ_temp = in_equ_list[i];
        equ_tensor_temp = tensor_list[i];
        if (is_ellispis(equ_temp, targets)) {
            map_with_ellipsis(equ_temp,equ_tensor_temp,equ_map,ellipsis_map);
        } else {
            map_without_ellipsis(equ_temp,equ_tensor_temp,equ_map);
        }
    }
    if (out_equ.size() == 0) {
        return;
    } else {
        if (is_ellispis(out_equ,targets)) {
            output_with_ellipsis(output_shape, out_equ, equ_map,ellipsis_map);
        } else {
            output_without_ellipsis(output_shape, out_equ, equ_map);
        }
    }
}

IMPLEMT_COMMON_INFERFUNC(EinsumInferShape) {
    auto x0_type = op.GetDynamicInputDesc("x", 0).GetDataType();
    // get attr equation
    std::string equation;
    if (op.GetAttr("equation", equation) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "GetOpAttr equation failed");
        return GRAPH_FAILED;
    }
    // set tensor_size attr
    int64_t tensor_size = op.GetInputsSize();
    op.SetAttr("N", tensor_size);
    vector<vector<int64_t>> tensor_list;
    for (size_t i = 0; i < op.GetInputsSize(); i++) {
        auto xi_dims = op.GetDynamicInputDesc("x", i).GetShape().GetDims();
        tensor_list.push_back(xi_dims);
    }
    vector<int64_t> output_shape;
    einsum_infer_shape(equation,tensor_list,output_shape);
    // updata output shape and dtype
    TensorDesc output_desc = op.GetOutputDesc("y");
    output_desc.SetShape(ge::Shape(output_shape));
    output_desc.SetDataType(x0_type);
    CHECK(op.UpdateOutputDesc("y",output_desc) != GRAPH_SUCCESS,
         OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."),
         return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(Einsum, EinsumVerify) {
    return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(Einsum, EinsumInferShape);
VERIFY_FUNC_REG(Einsum, EinsumVerify);
// ----------------Einsum-------------------

// ---------------Eye----------------------------
static bool CheckRows(const Operator &op, const string &attr_num_rows)
{
    int64_t num_rows;
    op.GetAttr(attr_num_rows, num_rows);
    if (num_rows <= 0) {
        return false;
    }
    return true;
}

static bool CheckBatchShape(const Operator &op, const string &attr_batch_shape)
{
    std::vector<int64_t> batch_shape;
    op.GetAttr(attr_batch_shape, batch_shape);
    for (int i = 0; i < batch_shape.size(); ++i) {
        if (batch_shape[i] <= 0) {
            OP_LOGE(op.GetName().c_str(), "the value of batch_shape less than 0.\n");
            return false;
        }
    }
    return true;
}

IMPLEMT_COMMON_INFERFUNC(EyeInferShape)
{
    TensorDesc td = op.GetOutputDesc("y");
    int64_t num_rows, num_columns;
    std::vector<int64_t> batch_shape;
    op.GetAttr("num_rows", num_rows);
    op.GetAttr("num_columns", num_columns);
    op.GetAttr("batch_shape", batch_shape);

    if (!CheckRows(op, "num_rows") || !CheckBatchShape(op, "batch_shape")) {
        return GRAPH_FAILED;
    }
    if (num_columns <= 0) {
        num_columns = num_rows;
    }
    std::vector<int64_t> dim_vec;
    for (int i = 0; i < batch_shape.size(); ++i) {
        dim_vec.push_back(batch_shape[i]);
    }
    dim_vec.push_back(num_rows);
    dim_vec.push_back(num_columns);
    td.SetShape(ge::Shape(dim_vec));
    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Eye, EyeVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Eye, EyeInferShape);

VERIFY_FUNC_REG(Eye, EyeVerify);
//--------------Eye END-------------------------------

// ----------------FillDiagonal-------------------
IMPLEMT_COMMON_INFERFUNC(FillDiagonalInferShape) {
  Shape x_shape = op.GetInputDesc("x").GetShape();
  DataType x_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(ge::Shape(x_shape));
  td.SetDataType(x_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(FillDiagonal, FillDiagonalInferShape);
// ----------------FillDiagonal END-------------------

}  // namespace ge
