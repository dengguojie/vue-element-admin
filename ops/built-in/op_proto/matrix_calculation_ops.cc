/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
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
#include <array>

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
// MatMul & BatchMatMul input index
const int64_t kMatMulInputX1Index = 0;
const int64_t kMatMulInputX2Index = 1;
const int64_t kMatMulInputBiasIndex = 2;

// BatchMatmul
const int64_t kBatchMatmulMaxShapeSize = 8;

// Einsum
const int64_t kEinsumOffsetLength = 3;

// FullyConnection
const int64_t kSliceRangeLength = 2;
const int64_t kChannelZero = 16;
const int64_t kAxisReduceIdx = 2;
const int64_t kWeightIndexH = 2;

// MatrixDiagPartV2InferShape
const int64_t kInputPadIndex = 2;
const int64_t kInputKMaxSize = 2;

// MatrixSetDiagV2
const int64_t kInputKIndex = 2;

// MatrixDiagV2
const int64_t kInputNumRowsIndex = 2;
const int64_t kInputNumColsIndex = 3;
const int64_t kInputPaddingIndex = 4;

// MatMul FRACTAL_Z
const int64_t BLOCK_SIZE = 16;

// Matmul/BatchMatmul m_indx n_index
const int64_t kOutputMIndex = -2;
const int64_t kOutputNIndex = -1;

template <typename T>
static std::string VectorToString(const std::vector<T> &dims) {
  std::stringstream ss;
  for (auto iter = dims.begin(); iter != dims.end(); ++iter) {
    ss << *iter;
    if (iter != dims.end() - 1) {
      ss << ", ";
    }
  }
  return ss.str();
}

// ----------------FullyConnection-------------------

bool InferFC5HD(vector<vector<int64_t>>& x_data_slice, vector<vector<int64_t>>& w_data_slice,
                vector<vector<int64_t>>& y_data_slice, int32_t& infer_x, int32_t& infer_w) {
  for (size_t i = 0; i < y_data_slice.size(); i++) {
    if (y_data_slice[i].size() > 0) {
      if (static_cast<int64_t>(i) == 0) {
        x_data_slice[i] = y_data_slice[i];
        infer_x = 1;
      } else if (static_cast<int64_t>(i) == 1) {
        w_data_slice[i] = y_data_slice[i];
        infer_w = 1;
      }
    }
  }
  return true;
}

bool InferFC5HD2NZ(vector<vector<int64_t>>& x_data_slice, vector<vector<int64_t>>& w_data_slice,
                   vector<vector<int64_t>>& y_data_slice, int32_t& infer_x, int32_t& infer_w,
                   const vector<int64_t>& x_shape) {
  for (size_t i = 0; i < y_data_slice.size(); i++) {
    if (y_data_slice[i].size() > 0) {
      if (static_cast<int64_t>(i) == 0) {
        w_data_slice[1] = y_data_slice[i];
        infer_w = 1;
      } else if (static_cast<int64_t>(i) == 1 && y_data_slice[i].size() == kSliceRangeLength) {
        int64_t m_start = y_data_slice[i][0] * kChannelZero;
        int64_t m_end = std::min((y_data_slice[i][1] + 1) * kChannelZero - 1, x_shape[0] - 1);
        x_data_slice[0] = {m_start, m_end};
        infer_x = 1;
      }
    }
  }
  return true;
}

bool InferFCNZ(vector<vector<int64_t>>& x_data_slice, vector<vector<int64_t>>& w_data_slice,
               vector<vector<int64_t>>& y_data_slice, int32_t& infer_x, int32_t& infer_w,
               const int64_t axis) {
  for (size_t i = 0; i < y_data_slice.size(); i++) {
    if (y_data_slice[i].size() <= 0) {
      continue;
    }
    if (axis == kAxisReduceIdx) {
      // 0 and 2 is index of m and m0
      if (static_cast<int64_t>(i) == 0 || static_cast<int64_t>(i) == 2) {
        x_data_slice[i] = y_data_slice[i];
        infer_x = 1;
      } else if (static_cast<int64_t>(i) == 1) {
        w_data_slice[i] = y_data_slice[i];
        infer_w = 1;
      }
    } else {
      if (static_cast<int64_t>(i) == 0) {
        w_data_slice[1] = y_data_slice[i];
        infer_w = 1;
      } else if (static_cast<int64_t>(i) == 1) {
        x_data_slice[i] = y_data_slice[i];
        infer_x = 1;
      }
    }
  }
  return true;
}


bool InferFullyConnectionDataSlice(ge::Operator& op) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return false);
  auto x_tensor = op.GetInputDescByName("x");
  auto w_tensor = op.GetInputDescByName("w");
  auto y_tensor = op.GetOutputDescByName("y");

  auto x_shape = x_tensor.GetShape().GetDims();
  auto w_shape = w_tensor.GetShape().GetDims();
  auto x_format = x_tensor.GetFormat();
  auto y_format = y_tensor.GetFormat();

  int64_t num_output;
  int64_t axis;
  if (GRAPH_SUCCESS != op.GetAttr("num_output", num_output) || GRAPH_SUCCESS != op.GetAttr("axis", axis)) {
    return false;
  }
  vector<vector<int64_t> > x_data_slice;
  if (x_format == FORMAT_NC1HWC0 || axis == kAxisReduceIdx) {
    x_data_slice = {{}, {}, {}, {}, {}};
  } else {
    x_data_slice = {{}, {}, {}, {}};
  }
  vector<vector<int64_t> > w_data_slice = {{}, {}, {}, {}};

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
  GeTensorDescPtr tensor_desc_w = op_desc->MutableInputDesc("w");
  vector<vector<int64_t> > y_data_slice;
  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(opName.GetString(), "no data slice, not need infer input");
    return false;
  }

  int32_t infer_x = 0;
  int32_t infer_w = 0;
  if (y_format == FORMAT_NC1HWC0) {
    OP_LOGI(opName.GetString(), "infer dataslice from 5HD to 5HD");
    InferFC5HD(x_data_slice, w_data_slice, y_data_slice, infer_x, infer_w);
  } else if (x_format == FORMAT_NC1HWC0) {
    OP_LOGI(opName.GetString(), "infer dataslice from 5HD to NZ");
    InferFC5HD2NZ(x_data_slice, w_data_slice, y_data_slice, infer_x, infer_w, x_shape);
  } else {
    OP_LOGI(opName.GetString(), "infer dataslice from NZ to NZ");
    InferFCNZ(x_data_slice, w_data_slice, y_data_slice, infer_x, infer_w, axis);
  }

  if (infer_x == 0 && infer_w == 0) {
    OP_LOGI(opName.GetString(), "no data slice, not need infer input");
    return false;
  }

  if (infer_x == 1) {
    if (!AttrUtils::SetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
      return false;
    }
    OP_LOGI(opName.GetString(), "infer input x success");
  }
  if (infer_w == 1) {
    if (!AttrUtils::SetListListInt(tensor_desc_w, ge::ATTR_NAME_DATA_SLICE, w_data_slice)) {
      return false;
    }
    num_output = (w_data_slice[1][1] - w_data_slice[1][0] + 1) * w_shape[kWeightIndexH];
    op.SetAttr("num_output", num_output);
    OP_LOGI(opName.GetString(), "infer input w success");
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
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);

  if (axis < 0) {
    axis_new = axis + xDim;
  } else {
    axis_new = axis;
  }

  // check axis
  if (axis_new != 1 && axis_new != 2) {
    std::string err_msg =
        OtherErrMsg(ConcatString("Attr axis is wrong, the original value of axis ", axis, " is not supported."));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), err_msg);
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
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), err_msg);
    return GRAPH_FAILED;
  }

  // check km and kn shape
  if (wShape.size() == 2) {
    if (!transpose) {
      if (kShape != wShape[1]) {
        string err_msg1 = ConcatString("weight K must equal to input K! kShape:", kShape, ", wShape[1]:", wShape[1]);
        std::string err_msg = OtherErrMsg(err_msg1);
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), err_msg);
        return GRAPH_FAILED;
      }
    } else {
      if (kShape != wShape[0]) {
        string err_msg1 = ConcatString("weight K must equal to input K! kShape:", kShape, ", wShape[0]:", wShape[0]);
        std::string err_msg = OtherErrMsg(err_msg1);
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), err_msg);
        return GRAPH_FAILED;
      }
    }
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(FullyConnection, FullyConnectionInfer) {
  auto outDesc = op.GetOutputDescByName("y");
  auto weightDesc = op.GetInputDescByName("w");
  auto xDesc = op.GetInputDescByName("x");

  auto xShape = op.GetInputDescByName("x").GetShape().GetDims();
  auto wShape = op.GetInputDescByName("w").GetShape().GetDims();
  auto xDtype = op.GetInputDescByName("x").GetDataType();
  bool transpose = op.get_attr_transpose();
  int axis = op.get_attr_axis();
  int xDim = xShape.size();
  int axis_new;
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);

  axis_new = (axis < 0) ? (axis + xDim) : axis;
  if (axis_new != 1 && axis_new != 2) {
    std::string err_msg = GetAttrValueErrMsg("axis_new", std::to_string(axis_new), "1 or 2");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), err_msg);
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
      CUBE_INNER_ERR_REPORT(opName.GetString(), "Not enough info about M and K!\n");
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

    if (transpose) {
      changedWeightShape.push_back(xShape[1]);
      changedWeightShape.push_back(xShape[2]);
      changedWeightShape.push_back(xShape[3]);
      changedWeightShape.push_back(wShape[1]);
      yShape.push_back(wShape[1]);
      weightDesc.SetFormat(ge::FORMAT_CHWN);
      weightDesc.SetOriginFormat(ge::FORMAT_CHWN);
    } else {
      changedWeightShape.push_back(wShape[0]);
      changedWeightShape.push_back(xShape[1]);
      changedWeightShape.push_back(xShape[2]);
      changedWeightShape.push_back(xShape[3]);
      yShape.push_back(wShape[0]);
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
  auto outDataType = (xDtype == ge::DT_INT8 || xDtype == ge::DT_INT4) ? ge::DT_INT32 : ge::DataType(xDtype);
  outDesc.SetDataType(outDataType);
  (void)op.UpdateOutputDesc("y", outDesc);

  return GRAPH_SUCCESS;
}

IMPLEMT_INFER_DATA_SLICE(FullyConnection, FullyConnectionInferDataSlice) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  OP_LOGD(opName.GetString(), "Enter FullyConnection InferDataSlice");
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
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
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
    OP_LOGE(opName.GetString(), "wShape Compress size must equal to 1 or 2!\n");
    return GRAPH_FAILED;
  }

  // check km and kn shape
  if (wShape.size() == 2) {
    if (!transpose) {
      if (kShape != wShape[1]) {
        OP_LOGE(opName.GetString(), "weight Compress K must equal to input K!\n");
        return GRAPH_FAILED;
      }
    } else {
      if (kShape != wShape[0]) {
        OP_LOGE(opName.GetString(), "weight Compress K must equal to input K!\n");
        return GRAPH_FAILED;
      }
    }
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(FullyConnectionCompress, FullyConnectionCompressInfer) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  auto outDesc = op.GetOutputDescByName("y");
  auto weightDesc = op.GetInputDescByName("w");

  auto xShape = op.GetInputDescByName("x").GetShape().GetDims();
  auto wShape = op.GetInputDescByName("w").GetShape().GetDims();
  auto xDtype = op.GetInputDescByName("x").GetDataType();
  bool transpose = op.get_attr_transpose();

  if (xShape.size() < 1 || wShape.size() < 1) {
    OP_LOGE(opName.GetString(), "Invalid Shape size, xShape size is %lu, wShape size is %lu.", xShape.size(),
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
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  OP_LOGD(opName.GetString(), "Enter FullyConnectionCompress InferDataSlice");
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
  auto desc_a = op.GetInputDescByName("x1");
  auto desc_b = op.GetInputDescByName("x2");
  ge::Shape shape_a = desc_a.GetShape();
  ge::Shape shape_b = desc_b.GetShape();
  ge::Shape ori_shape_a = desc_a.GetOriginShape();
  ge::Shape ori_shape_b = desc_b.GetOriginShape();
  std::vector<std::pair<int64_t, int64_t>> range_a;
  std::vector<std::pair<int64_t, int64_t>> range_b;
  desc_a.GetShapeRange(range_a);
  desc_b.GetShapeRange(range_b);

  bool trans_a = false, trans_b = false;
  op.GetAttr((name_attr + "_x1").c_str(), trans_a);
  op.GetAttr((name_attr + "_x2").c_str(), trans_b);

  std::ostringstream oss;
  oss << "shape of a: ";
  for (size_t i = 0; i < shape_a.GetDimNum(); i++) {
      oss << shape_a.GetDim(i) << ", ";
  }
  oss << "ori shape of a: ";
  for (size_t i = 0; i < ori_shape_a.GetDimNum(); i++) {
      oss << ori_shape_a.GetDim(i) << ", ";
  }
  oss << std::endl;
  oss << "shape of b: ";
  for (size_t i = 0; i < shape_b.GetDimNum(); i++) {
      oss << shape_b.GetDim(i) << ", ";
  }
  oss << "ori shape of b: ";
  for (size_t i = 0; i < ori_shape_b.GetDimNum(); i++) {
      oss << ori_shape_b.GetDim(i) << ", ";
  }
  oss << std::endl;
  oss << "trans_a " << trans_a << " trans_b " << trans_b << std::endl;
  oss << "range of a: (";
  for (size_t i = 0; i < range_a.size(); i++) {
      oss << "(" << range_a[i].first << ", " << range_a[i].second << ") ";
  }
  oss << ")" << std::endl;
  oss << "range of b: (";
  for (size_t i = 0; i < range_b.size(); i++) {
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

class InferShapeMatMul {
 public:
  InferShapeMatMul(const AscendString &op_name,
                   const GeShape &shape_a,
                   const GeShape &shape_b,
                   vector<std::pair<int64_t, int64_t>> &range_a,
                   vector<std::pair<int64_t, int64_t>> &range_b,
                   bool trans_a,
                   bool trans_b,
                   GeShape &shape_out,
                   vector<std::pair<int64_t, int64_t>> &range_out,
                   ConstGeTensorDescPtr tensordesc_bias = nullptr) :
      op_name(op_name),
      shape_a(shape_a),
      shape_b(shape_b),
      range_a(range_a),
      range_b(range_b),
      trans_a(trans_a),
      trans_b(trans_b),
      shape_out(shape_out),
      range_out(range_out),
      tensordesc_bias(tensordesc_bias) {
      if (tensordesc_bias != nullptr) {
        shape_bias = tensordesc_bias->GetShape();
        tensordesc_bias->GetShapeRange(range_bias);
      }
    }

  ~InferShapeMatMul() {}

  static graphStatus VerifyInputs(const Operator &op);
  static bool IsRangeValid(const GeShape &shape,
                          const std::vector<std::pair<int64_t, int64_t>> &range);
  static void SimplifyShapeAndRange(GeShape& local_shape_out,
                                    vector<std::pair<int64_t, int64_t>>& range_out);

  bool InferShape();

 protected:
  static const int64_t BASE_LEN = 2;

  bool InitializeShapeAndRange(const GeShape& shape,
                              const vector<std::pair<int64_t, int64_t>>& range,
                              std::array<int64_t, BASE_LEN>& infer_shape,
                              std::array<std::pair<int64_t, int64_t>, BASE_LEN>& infer_range);
  bool InferMKN() const;
  bool InferBias() const;

  const AscendString& op_name;
  const GeShape &shape_a;
  const GeShape &shape_b;
  vector<std::pair<int64_t, int64_t>> &range_a;
  vector<std::pair<int64_t, int64_t>> &range_b;
  bool trans_a;
  bool trans_b;
  GeShape &shape_out;
  vector<std::pair<int64_t, int64_t>> &range_out;
  ConstGeTensorDescPtr tensordesc_bias;
  GeShape shape_bias;
  vector<std::pair<int64_t, int64_t>> range_bias;

  std::array<int64_t, BASE_LEN> infer_shape_a;
  std::array<int64_t, BASE_LEN> infer_shape_b;
  std::array<std::pair<int64_t, int64_t>, BASE_LEN> infer_range_a;
  std::array<std::pair<int64_t, int64_t>, BASE_LEN> infer_range_b;
};

bool InferShapeMatMul::IsRangeValid(const GeShape &shape, const std::vector<std::pair<int64_t, int64_t>> &range) {
  if (shape.GetDimNum() == 0 || shape.IsUnknownDimNum() || range.empty()) {
    return true;
  }

  if (shape.IsUnknownShape()) {
    if (range.size() != shape.GetDimNum()) {
      return false;
    }

    for (size_t i = 0; i < range.size(); ++i) {
      CHECK(shape.GetDim(i) < UNKNOWN_DIM, OP_LOGE("", "Invalid dim size"), return false);
      CHECK(shape.GetDim(i) == UNKNOWN_DIM && range[i].second != INFINITE_RANGE && range[i].first > range[i].second,
            OP_LOGE("", "Invalid dim size"), return false);
      CHECK(shape.GetDim(i) == UNKNOWN_DIM && range[i].second == INFINITE_RANGE && range[i].first == INFINITE_RANGE,
            OP_LOGE("", "Invalid dim size"), return false);
    }
  }

  return true;
}

graphStatus InferShapeMatMul::VerifyInputs(const Operator &op) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);

  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);

  ge::ConstGeTensorDescPtr tensordesc_x1 = op_desc->GetInputDescPtr(kMatMulInputX1Index);
  ge::ConstGeTensorDescPtr tensordesc_x2 = op_desc->GetInputDescPtr(kMatMulInputX2Index);
  const GeShape& shape_x1 = tensordesc_x1->GetShape();
  const GeShape& shape_x2 = tensordesc_x2->GetShape();
  std::vector<std::pair<int64_t, int64_t>> shape_range_x1;
  std::vector<std::pair<int64_t, int64_t>> shape_range_x2;
  if (shape_x1.IsUnknownShape() || shape_x2.IsUnknownShape()) {
    tensordesc_x1->GetShapeRange(shape_range_x1);
    tensordesc_x2->GetShapeRange(shape_range_x2);
  }

  CHECK(!IsRangeValid(shape_x1, shape_range_x1),
        OP_LOGE(opName.GetString(), "Precheck input shape failed."), return GRAPH_FAILED);
  CHECK(!IsRangeValid(shape_x2, shape_range_x2),
        OP_LOGE(opName.GetString(), "Precheck input shape failed."), return GRAPH_FAILED);

  ge::ConstGeTensorDescPtr tensordesc_bias = op_desc->GetInputDescPtr(kMatMulInputBiasIndex);
  if (tensordesc_bias != nullptr) {
    const GeShape& shape_bias = tensordesc_bias->GetShape();
    std::vector<std::pair<int64_t, int64_t>> shape_range_bias;
    if (shape_bias.IsUnknownShape()) {
      tensordesc_bias->GetShapeRange(shape_range_bias);
    }
    CHECK(!IsRangeValid(shape_bias, shape_range_bias),
        OP_LOGE(opName.GetString(), "Precheck input shape failed."), return GRAPH_FAILED);
  }

  return GRAPH_SUCCESS;
}

void InferShapeMatMul::SimplifyShapeAndRange(GeShape& local_shape_out, vector<std::pair<int64_t, int64_t>>& range_out) {
  CHECK(!local_shape_out.IsUnknownShape(), range_out.clear(), return);
  for (size_t i = 0; i < range_out.size(); i++) {
    if (range_out[i].first == range_out[i].second) {
      local_shape_out.SetDim(i, range_out[i].first);
    }
    // reverse normalize
    if (range_out[i].second == NORMALIZE_INFINITE_RANGE) {
      range_out[i] = {range_out[i].first, INFINITE_RANGE};
    }
  }
}

bool InferShapeMatMul::InferBias() const {
  // --------------------------------------------------------------------------------------
  // | bias \ n       |      {-2}      |          -1, (y1, y2)              |     y       |
  // | -------------- | --------------------------------------------------- | ----------- |
  // | {} or {-2}     |      {-2}      |          -1, (y1, y2)              |     y       |
  // | -1, (x1, x2)   |  -1, (x1, x2)  | -1, (max(x1,y1), min(x2,y2)) check |  y check    |
  // |       x        |       x        |             x check                |  x==y check |
  // --------------------------------------------------------------------------------------
  int64_t bias_dim = shape_bias.GetDimNum();
  CHECK(shape_bias.GetDimNum() == 0 || shape_bias.IsUnknownDimNum(), NULL, return true);

  if (shape_out.IsUnknownDimNum()) {
    if (shape_bias.GetDim(bias_dim - 1) > 0) {
      shape_out.SetDimNum(BASE_LEN);
      shape_out.SetDim(0, UNKNOWN_DIM);
      shape_out.SetDim(1, shape_bias.GetDim(bias_dim - 1));
      range_out = {FULL_RANGE, {shape_bias.GetDim(bias_dim - 1), shape_bias.GetDim(bias_dim - 1)}};
      return true;
    }
    CHECK(shape_bias.GetDim(bias_dim - 1) == UNKNOWN_DIM && range_bias.empty(), NULL, return true);
    shape_out.SetDimNum(BASE_LEN);
    shape_out.SetDim(0, UNKNOWN_DIM);
    shape_out.SetDim(1, shape_bias.GetDim(bias_dim - 1));
    range_out = {FULL_RANGE, range_bias.back()};
    return true;
  }

  if (shape_bias.GetDim(bias_dim - 1) == UNKNOWN_DIM && shape_out.GetDim(1) == UNKNOWN_DIM) {
    // Compatible with old version
    CHECK((range_bias.empty() || range_bias.back() == FULL_RANGE), NULL, return true);
    auto range_out_upper = range_out.back().second == INFINITE_RANGE ?
      NORMALIZE_INFINITE_RANGE : range_out.back().second;
    auto range_bias_upper = range_bias.back().second == INFINITE_RANGE ?
      NORMALIZE_INFINITE_RANGE : range_bias.back().second;
    auto lower_bound = std::max(range_out.back().first, range_bias.back().first);
    auto upper_bound = std::min(range_out_upper, range_bias_upper);
    if (lower_bound <= upper_bound) {
      range_out.back() = {lower_bound, upper_bound};
      return true;
    }
    CUBE_INNER_ERR_REPORT(
        op_name.GetString(), "[InferShape] range n [%ld, %ld] and bias [%ld, %ld] must have intersections",
        range_out.back().first, range_out.back().second, range_bias.back().first, range_bias.back().second);
    return false;
  }

  if (shape_bias.GetDim(bias_dim - 1) == UNKNOWN_DIM) {
    CHECK(range_bias.empty(), NULL, return true);
    auto range_bias_lower = range_bias.back().first;
    auto range_bias_upper = range_bias.back().second == INFINITE_RANGE ?
      NORMALIZE_INFINITE_RANGE : range_bias.back().second;
    CHECK(range_bias_lower <= shape_out.GetDim(1) && shape_out.GetDim(1) <= range_bias_upper, NULL, return true);
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "[InferShape] dimension n [%ld] must be in range [%ld, %ld]",
                          shape_out.GetDim(1), range_bias_lower, range_bias_upper);
    return false;
  }

  if (shape_out.GetDim(1) == UNKNOWN_DIM) {
    auto range_out_lower = range_out.back().first;
    auto range_out_upper = range_out.back().second == INFINITE_RANGE ?
      NORMALIZE_INFINITE_RANGE : range_out.back().second;
    if (range_out_lower <= shape_bias.GetDim(bias_dim - 1) && shape_bias.GetDim(bias_dim - 1) <= range_out_upper) {
      shape_out.SetDim(1, shape_bias.GetDim(bias_dim - 1));
      range_out.back() = {shape_bias.GetDim(bias_dim - 1), shape_bias.GetDim(bias_dim - 1)};
      return true;
    }
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "[InferShape] dimension bias [%ld] must be in range [%ld, %ld]",
                          shape_bias.GetDim(bias_dim - 1), range_out_lower, range_out_upper);
    return false;
  }

  CHECK(shape_bias.GetDim(bias_dim - 1) != shape_out.GetDim(1), OP_LOGE(op_name.GetString(),
      "[InferShape] The dimension of n [%ld] and bias [%ld] tensors must be the same",
      shape_out.GetDim(1), shape_bias.GetDim(bias_dim - 1)), return false);

  return true;
}

bool InferShapeMatMul::InferMKN() const {
  int64_t idx_m = trans_a ? 1 : 0;
  int64_t idx_k_a = trans_a ? 0 : 1;
  int64_t idx_k_b = trans_b ? 1 : 0;
  int64_t idx_n_b = trans_b ? 0 : 1;

  auto m = infer_shape_a[idx_m];
  auto k_a = infer_shape_a[idx_k_a];
  auto k_b = infer_shape_b[idx_k_b];
  auto n = infer_shape_b[idx_n_b];

  // ka = -1, kb = -1
  if (k_a == UNKNOWN_DIM && k_b == UNKNOWN_DIM) {
    auto lower_bound = std::max(infer_range_a[idx_k_a].first, infer_range_b[idx_k_b].first);
    auto upper_bound = std::min(infer_range_a[idx_k_a].second, infer_range_b[idx_k_b].second);
    if (lower_bound > upper_bound) {
      CUBE_INNER_ERR_REPORT(op_name.GetString(),
                            "[InferShape] range k_a [%ld, %ld] and k_b [%ld, %ld] must have intersections",
                            infer_range_a[idx_k_a].first, infer_range_a[idx_k_a].second, infer_range_b[idx_k_b].first,
                            infer_range_b[idx_k_b].second);
      return false;
    }
    // ka = -1, kb != -1
  } else if (k_a == UNKNOWN_DIM) {
    if (infer_range_a[idx_k_a].first > k_b || k_b > infer_range_a[idx_k_a].second) {
      CUBE_INNER_ERR_REPORT(op_name.GetString(), "[InferShape] dimension of k_b [%ld] must be in range [%ld, %ld]",
                            k_b, infer_range_a[idx_k_a].first, infer_range_a[idx_k_a].second);
      return false;
    }
  // ka != -1, kb = -1
  } else if (k_b == UNKNOWN_DIM) {
    if (infer_range_b[idx_k_b].first > k_a || k_a > infer_range_b[idx_k_b].second) {
      CUBE_INNER_ERR_REPORT(op_name.GetString(), "[InferShape] dimension of k_a [%ld] must be in range [%ld, %ld]",
                            k_a, infer_range_b[idx_k_b].first, infer_range_b[idx_k_b].second);
      return false;
    }
  // ka != -1, kb != -1
  } else {
    if (k_a != k_b) {
      OP_LOGE(op_name.GetString(), "[InferShape] The k-axis of x1 [%ld] and x2 [%ld] tensors must be the same", k_a,
              k_b);
      return false;
    }
  }
  shape_out.SetDimNum(BASE_LEN);
  shape_out.SetDim(0, m);
  shape_out.SetDim(1, n);
  range_out = {infer_range_a[idx_m], infer_range_b[idx_n_b]};

  return true;
}

bool InferShapeMatMul::InitializeShapeAndRange(const GeShape& shape,
                                               const vector<std::pair<int64_t, int64_t>>& range,
                                               std::array<int64_t, BASE_LEN>& infer_shape,
                                               std::array<std::pair<int64_t, int64_t>, BASE_LEN>& infer_range) {
  // Dynamic shape: {-2}
  // infer_shape: {-1, ..., -1}; infer_range: {{1, MAX}, ..., {1, MAX}}
  if (shape.IsUnknownDimNum()) {
    infer_shape = {UNKNOWN_DIM, UNKNOWN_DIM};
    infer_range = {NORMALIZE_FULL_RANGE, NORMALIZE_FULL_RANGE};
    return true;
  }
  // Dynamic shape: {..., -1, ...}
  infer_shape = {shape.GetDim(0), shape.GetDim(1)};
  infer_range = {NORMALIZE_FULL_RANGE, NORMALIZE_FULL_RANGE};
  for (size_t i = 0; i < shape.GetDimNum(); ++i) {
    CHECK(shape.GetDim(i) < UNKNOWN_DIM, OP_LOGE("", "Invalid dim size"), return false);
    // shape[i] > 0
    if (shape.GetDim(i) > 0) {
      infer_range[i] = {shape.GetDim(i), shape.GetDim(i)};
      continue;
    }
    // shape[i] == -1
    if (range.empty()) {
      // range {} -> infer_range {1, MAX}
      infer_range[i] = NORMALIZE_FULL_RANGE;
      continue;
    }
    if (range[i] == EMPTY_RANGE) {
      // range {0, 0} -> infer_range {1, MAX}
      infer_range[i] = NORMALIZE_FULL_RANGE;
    } else if (range[i].second == INFINITE_RANGE) {
      // range {a, -1} -> infer_range {a, MAX}
      infer_range[i] = {range[i].first, NORMALIZE_INFINITE_RANGE};
    } else {
      // range {a, b} -> infer_range {a, b}
      infer_range[i] = range[i];
    }
  }

  return true;
}

bool InferShapeMatMul::InferShape() {
  // 1) Static shape
  if (!shape_a.IsUnknownShape() && !shape_b.IsUnknownShape()) {
    int64_t idx_m = trans_a ? 1 : 0;
    int64_t idx_k_a = trans_a ? 0 : 1;
    int64_t idx_k_b = trans_b ? 1 : 0;
    int64_t idx_n = trans_b ? 0 : 1;

    if (shape_a.GetDim(idx_k_a) != shape_b.GetDim(idx_k_b)) {
      OP_LOGE(op_name.GetString(), "[InferShape] The k-axis of a(%ld) and b(%ld) tensors must be the same",
              shape_a.GetDim(idx_k_a), shape_b.GetDim(idx_k_b));
      return false;
    }
    shape_out.SetDimNum(BASE_LEN);
    shape_out.SetDim(0, shape_a.GetDim(idx_m));
    shape_out.SetDim(1, shape_b.GetDim(idx_n));
    range_out.clear();
    if (tensordesc_bias != nullptr) {
      CHECK(!InferBias(), OP_LOGE(op_name.GetString(), "Infer bias failed."), return false);
    }
    return true;
  }
  // 2) Dynamic shape
  if (shape_a.IsUnknownDimNum() && shape_b.IsUnknownDimNum()) {
    shape_out.SetIsUnknownDimNum();
    range_out.clear();
    OP_LOGW(op_name.GetString(), "[InferShape] cannot derive any shape and range information of output");
    if (tensordesc_bias != nullptr) {
      CHECK(!InferBias(), OP_LOGE(op_name.GetString(), "Infer bias failed."), return false);
    }
    return true;
  }
  // Initialize infer_shape & infer_range
  CHECK(!InitializeShapeAndRange(shape_a, range_a, infer_shape_a, infer_range_a),
    OP_LOGE(op_name.GetString(), "Initialize infer_shape & infer_range failed."), return false);
  CHECK(!InitializeShapeAndRange(shape_b, range_b, infer_shape_b, infer_range_b),
    OP_LOGE(op_name.GetString(), "Initialize infer_shape & infer_range failed."), return false);
  // 3) Infer output shape
  CHECK(!InferMKN(), OP_LOGE(op_name.GetString(), "Failed to infer output shape."), return false);
  // 4) InferBias
  if (tensordesc_bias != nullptr) {
    CHECK(!InferBias(), OP_LOGE(op_name.GetString(), "Infer bias failed."), return false);
  }
  // 5) Postprocess
  SimplifyShapeAndRange(shape_out, range_out);

  return true;
}

class InferShapeBatchMatMul {
 public:
  InferShapeBatchMatMul(const AscendString& op_name, const GeShape& shape_a, const GeShape& shape_b,
                        vector<std::pair<int64_t, int64_t>>& range_a, vector<std::pair<int64_t, int64_t>>& range_b,
                        bool trans_a, bool trans_b, GeShape& shape_out, vector<std::pair<int64_t, int64_t>>& range_out,
                        ConstGeTensorDescPtr tensordesc_bias = nullptr)
      : op_name(op_name),
        shape_a(shape_a),
        shape_b(shape_b),
        range_a(range_a),
        range_b(range_b),
        trans_a(trans_a),
        trans_b(trans_b),
        shape_out(shape_out),
        range_out(range_out),
        tensordesc_bias(tensordesc_bias) {
    num_dima = shape_a.GetDimNum();
    num_dimb = shape_b.GetDimNum();
    num_dim = std::max(num_dima, num_dimb);
    if (tensordesc_bias != nullptr) {
      shape_bias = tensordesc_bias->GetShape();
      tensordesc_bias->GetShapeRange(range_bias);
      num_dim_bias = shape_bias.GetDimNum();
      num_dim = std::max(num_dim, num_dim_bias);
    }
    shape_out.SetDimNum(num_dim);
    range_out = vector<std::pair<int64_t, int64_t>>(num_dim);
  };

  ~InferShapeBatchMatMul() {};
  bool InferShape();

 protected:
  static const int64_t BASE_LEN = 8;
  int64_t num_dim;
  int64_t num_dima;
  int64_t num_dimb;
  int64_t num_dim_bias;
  bool InitializeShapeAndRange(const GeShape& shape, const vector<std::pair<int64_t, int64_t>>& range,
                               std::array<std::pair<int64_t, int64_t>, BASE_LEN>& infer_range);
  bool InferBatchStatic() const;
  bool InferBatch() const;
  bool InferMKN() const;
  bool InferBias();

  const AscendString& op_name;
  const GeShape& shape_a;
  const GeShape& shape_b;
  vector<std::pair<int64_t, int64_t>>& range_a;
  vector<std::pair<int64_t, int64_t>>& range_b;
  bool trans_a;
  bool trans_b;
  GeShape& shape_out;
  vector<std::pair<int64_t, int64_t>>& range_out;
  ConstGeTensorDescPtr tensordesc_bias;
  GeShape shape_bias;
  vector<std::pair<int64_t, int64_t>> range_bias;

  std::array<std::pair<int64_t, int64_t>, BASE_LEN> infer_range_a;
  std::array<std::pair<int64_t, int64_t>, BASE_LEN> infer_range_b;
  std::array<std::pair<int64_t, int64_t>, BASE_LEN> infer_range_bias;
};

bool InferShapeBatchMatMul::InferMKN() const {
  // use index - 2 to get m_dim
  int64_t idx_m = num_dima - 2;
  // use index - 1 to get k_dim of a
  int64_t idx_k_a = num_dima - 1;
  // use index - 2 to get k_dim of b
  int64_t idx_k_b = num_dimb - 2;
  // use index - 1 to get n_dim
  int64_t idx_n_b = num_dimb - 1;
  if (trans_a) {
    // use index - 1 to get m_dim when transposed
    idx_m = num_dima - 1;
    // use index - 2 to get k_dim of a when transposed
    idx_k_a = num_dima - 2;
  }
  if (trans_b) {
    // use index - 1 to get k_dim of b when transposed
    idx_k_b = num_dimb - 1;
    // use index - 2 to get n_dim when transposed
    idx_n_b = num_dimb - 2;
  }

  auto m = shape_a.GetDim(idx_m);
  auto k_a = shape_a.GetDim(idx_k_a);
  auto k_b = shape_b.GetDim(idx_k_b);
  auto n = shape_b.GetDim(idx_n_b);

  // ka = -1, kb = -1
  if (k_a == UNKNOWN_DIM && k_b == UNKNOWN_DIM) {
    auto lower_bound = std::max(infer_range_a[idx_k_a].first, infer_range_b[idx_k_b].first);
    auto upper_bound = std::min(infer_range_a[idx_k_a].second, infer_range_b[idx_k_b].second);
    // infer_range_ka & infer_range_kb != NULL
    CHECK((lower_bound > upper_bound),
          CUBE_INNER_ERR_REPORT(op_name.GetString(),
                                "[InferShape] range k_a[%ld, %ld] and k_b[%ld, %ld] must have intersections",
                                infer_range_a[idx_k_a].first, infer_range_a[idx_k_a].second,
                                infer_range_b[idx_k_b].first, infer_range_b[idx_k_b].second),
          return false);
  }
  // ka = -1, kb != -1
  else if (k_a == UNKNOWN_DIM) {
    CHECK((infer_range_a[idx_k_a].first > k_b || k_b > infer_range_a[idx_k_a].second),
          CUBE_INNER_ERR_REPORT(op_name.GetString(), "[InferShape] dimension of k_b (%ld) must be in range[%ld, %ld]",
                                k_b, infer_range_a[idx_k_a].first, infer_range_a[idx_k_a].second),
          return false);
  }
  // ka = -1, kb != -1
  else if (k_b == UNKNOWN_DIM) {
    CHECK((infer_range_b[idx_k_b].first > k_a || k_a > infer_range_b[idx_k_b].second),
          CUBE_INNER_ERR_REPORT(op_name.GetString(), "[InferShape] dimension of k_a (%ld) must be in range[%ld, %ld]",
                                k_a, infer_range_b[idx_k_b].first, infer_range_b[idx_k_b].second),
          return false);
  }
  // ka != -1, kb != -1
  else if (k_a != k_b) {
    OP_LOGE(op_name.GetString(), "[InferShape] The k-axis of a(%ld) and b(%ld) tensors must be the same", k_a, k_b);
    return false;
  }

  // use index - 2 to get m_dim
  shape_out.SetDim(num_dim - 2, m);
  // use index - 1 to get n_dim
  shape_out.SetDim(num_dim - 1, n);
  // use index - 2 to get m_range
  range_out[num_dim - 2] = infer_range_a[idx_m];
  // use index - 1 to get n_range
  range_out[num_dim - 1] = infer_range_b[idx_n_b];
  return true;
}

void CopyOutShapeFromInputShape(const GeShape& shape_in, GeShape& shape_out, int64_t valid_offset) {
  for (auto i = 0; i < valid_offset; ++i) {
    shape_out.SetDim(i, shape_in.GetDim(i));
  }
}

bool InferShapeBatchMatMul::InferBatchStatic() const {
  auto valid_offset = num_dim - std::min(num_dima, num_dimb);
  const GeShape& shape_long = num_dima < num_dimb ? shape_b : shape_a;
  const GeShape& shape_short = num_dima < num_dimb ? shape_a : shape_b;
  int64_t shape_value_long;
  int64_t shape_value_short;

  CopyOutShapeFromInputShape(shape_long, shape_out, valid_offset);
  // use index - 2 to get index of m
  for (auto i = valid_offset; i < num_dim - 2; ++i) {
    shape_value_short = shape_short.GetDim(i - valid_offset);
    shape_value_long = shape_long.GetDim(i);
    if (shape_value_short > 1 && shape_value_long > 1 && shape_value_short != shape_value_long) {
      return false;
    }
    shape_out.SetDim(i, std::max(shape_value_short, shape_value_long));
  }
  return true;
}

bool InferShapeBatchMatMul::InitializeShapeAndRange(const GeShape& shape,
                                                    const vector<std::pair<int64_t, int64_t>>& range,
                                                    std::array<std::pair<int64_t, int64_t>, BASE_LEN>& infer_range) {
  // Dynamic shape: {..., -1, ...}
  for (size_t i = 0; i < shape.GetDimNum(); ++i) {
    int64_t shape_value = shape.GetDim(i);
    // shape[i] < -1
    CHECK(shape_value < UNKNOWN_DIM, OP_LOGE("", "Invalid dim size"), return false);
    // shape[i] > 0
    if (shape_value > 0) {
      infer_range[i] = {shape_value, shape_value};
      continue;
    }
    // shape[i] == -1
    if (range.empty()) {
      // range {} -> infer_range {1, MAX}
      infer_range[i] = NORMALIZE_FULL_RANGE;
      continue;
    }
    if (range[i].second == INFINITE_RANGE) {
      // range {a, -1} -> infer_range {a, MAX}
      infer_range[i] = {range[i].first, NORMALIZE_INFINITE_RANGE};
    } else {
      // range {a, b} -> infer_range {a, b}
      infer_range[i] = range[i];
    }
  }

  return true;
}

bool BroadcastBatchDimAndRange(const AscendString& op_name, const int64_t dim_a, const int64_t dim_b,
                               const std::pair<int64_t, int64_t>& range_a, const std::pair<int64_t, int64_t>& range_b,
                               int64_t& dim, std::pair<int64_t, int64_t>& range) {
  // | b\a        | -1,(1,y)        | -1,(y1,y2)                       | 1          | y          |
  // | ---------- | --------------- | -------------------------------- | ---------- | ---------- |
  // | -1,(1,x)   | -1,(1,max(x,y)) | -1,(y1,y2)                       | -1,(1,x)   | y check    |
  // | -1,(x1,x2) | -1,(x1,x2)      | -1,(max(x1,y1),min(x2,y2)) check | -1,(x1,x2) | y check    |
  // | 1          | -1,(1,y)        | -1,(y1,y2)                       | 1          | y          |
  // | x          | x check         | x check                          | x          | x==y check |

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

  bool bothStaticDim = dim_a > 1 && dim_b > 1;
  if (bothStaticDim) {
    CHECK((dim_a != dim_b),
          CUBE_INNER_ERR_REPORT(op_name.GetString(), "[InferShape] dimensions a(%ld) and b(%ld) must be equal", dim_a,
                                dim_b),
          return false);

    dim = dim_a;
    range = range_a;
    return true;
  }
  if (dim_a > 1) {
    CHECK((range_b.first > dim_a || dim_a > range_b.second),
          CUBE_INNER_ERR_REPORT(op_name.GetString(), "[InferShape] dimension(%ld) must be in range[%ld, %ld]", dim_a,
                                range_b.first, range_b.second),
          return false);

    dim = dim_a;
    range = range_a;
    return true;
  }
  if (dim_b > 1) {
    CHECK((range_a.first > dim_b || dim_b > range_a.second),
          CUBE_INNER_ERR_REPORT(op_name.GetString(), "[InferShape] dimension(%ld) must be in range[%ld, %ld]", dim_b,
                                range_a.first, range_a.second),
          return false);

    dim = dim_b;
    range = range_b;
    return true;
  }

  bool bothRangeStartFrom1 = range_a.first == 1 && range_b.first == 1;
  if (bothRangeStartFrom1) {
    dim = UNKNOWN_DIM;
    range = {1, std::max(range_a.second, range_b.second)};
    return true;
  }

  bool bothRangeGreat1 = range_a.first > 1 && range_b.first > 1;
  if (bothRangeGreat1) {
    auto lower_bound = std::max(range_a.first, range_b.first);
    auto upper_bound = std::min(range_a.second, range_b.second);
    CHECK((lower_bound > upper_bound),
          CUBE_INNER_ERR_REPORT(op_name.GetString(),
                                "[InferShape] range a[%ld, %ld] and b[%ld, %ld] must have intersections", range_a.first,
                                range_a.second, range_b.first, range_b.second),
          return false);

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

bool InferShapeBatchMatMul::InferBatch() const {
  auto valid_offset = num_dim - std::min(num_dima, num_dimb);
  int64_t shape_value_out;

  if (num_dima < num_dim) {
    CopyOutShapeFromInputShape(shape_b, shape_out, valid_offset);
    copy(infer_range_b.begin(), infer_range_b.begin() + valid_offset, range_out.begin());
    // stop before num_dim - 2 so as to avoid traversing axis m, n
    for (auto i = valid_offset; i < num_dim - 2; ++i) {
      CHECK((!BroadcastBatchDimAndRange(op_name, shape_a.GetDim(i - valid_offset), shape_b.GetDim(i),
                                        infer_range_a[i - valid_offset], infer_range_b[i], shape_value_out,
                                        range_out[i])),
            NULL, return false);

      shape_out.SetDim(i, shape_value_out);
    }
    return true;
  }
  CopyOutShapeFromInputShape(shape_a, shape_out, valid_offset);
  copy(infer_range_a.begin(), infer_range_a.begin() + valid_offset, range_out.begin());
  // stop before num_dim - 2 so as to avoid traversing axis m, n
  for (auto i = valid_offset; i < num_dim - 2; ++i) {
    CHECK((!BroadcastBatchDimAndRange(op_name, shape_a.GetDim(i), shape_b.GetDim(i - valid_offset), infer_range_a[i],
                                      infer_range_b[i - valid_offset], shape_value_out, range_out[i])),
          NULL, return false);

    shape_out.SetDim(i, shape_value_out);
  }
  return true;
}

bool InferNDimWithBias(const AscendString& op_name, const int64_t dim_a, const int64_t dim_b,
                       const std::pair<int64_t, int64_t>& range_a, const std::pair<int64_t, int64_t>& range_b,
                       int64_t& dim, std::pair<int64_t, int64_t>& range) {
  // | b\a        | -1,(y1,y2)                      | y          |
  // | ---------- | ------------------------------- | ---------- |
  // | -1,(x1,x2) | -1,(max(x1,y1),min(x2,y)) check | y check    |
  // | x          | x check                         | x==y check |

  // shape_bias_n > 0 && n > 0
  if (dim_a > 0 && dim_b > 0) {
    CHECK(dim_a != dim_b,
          CUBE_INNER_ERR_REPORT(op_name.GetString(), "[InferShape] dimensions a(%ld) and b(%ld) must be same", dim_a,
                                dim_b),
          return false);
    dim = dim_a;
    range = range_a;
    return true;
  }

  // shape_bias_n = -1 && n = -1
  if (dim_a == UNKNOWN_DIM && dim_b == UNKNOWN_DIM) {
    auto lower_bound = std::max(range_a.first, range_b.first);
    auto upper_bound = std::min(range_a.second, range_b.second);
    CHECK((lower_bound > upper_bound),
          CUBE_INNER_ERR_REPORT(op_name.GetString(),
                                "[InferShape] range a[%ld, %ld] and b[%ld, %ld] must have intersections", range_a.first,
                                range_a.second, range_b.first, range_b.second),
          return false);

    range.first = lower_bound;
    range.second = upper_bound;
    return true;
  }

  // shape_bias_n = -1 && n > 0
  if (dim_a == UNKNOWN_DIM) {
    CHECK((range_a.first > dim_b || dim_b > range_a.second),
          CUBE_INNER_ERR_REPORT(op_name.GetString(), "[InferShape] dimension(%ld) must be in range[%ld, %ld]", dim_b,
                                range_a.first, range_a.second),
          return false);

    dim = dim_b;
    range = range_b;
    return true;
  }

  // shape_bias_n > 0 && n = -1
  CHECK((range_b.first > dim_a || dim_a > range_b.second),
        CUBE_INNER_ERR_REPORT(op_name.GetString(), "[InferShape] dimension(%ld) must be in range[%ld, %ld]", dim_a,
                              range_b.first, range_b.second),
        return false);

  dim = dim_a;
  range = range_a;
  return true;
}

bool InferShapeBatchMatMul::InferBias() {
  int64_t shape_value_out = shape_out.GetDim(num_dim - 1);
  // 1) shape_bias = {}
  CHECK(num_dim_bias == 0, NULL, return true);

  CHECK(!InitializeShapeAndRange(shape_bias, range_bias, infer_range_bias),
        OP_LOGE(op_name.GetString(), "Initialize infer_shape & infer_range failed."), return false);

  // 2) infer n with bias
  CHECK(!InferNDimWithBias(op_name, shape_bias.GetDim(num_dim_bias - 1), shape_out.GetDim(num_dim - 1),
                           infer_range_bias[num_dim_bias - 1], range_out[num_dim - 1], shape_value_out,
                           range_out[num_dim - 1]),
        NULL, return false);

  shape_out.SetDim(num_dim - 1, shape_value_out);

  // 3) infer batch with bias
  auto valid_offset = num_dim - std::min(num_dim_bias, std::max(num_dima, num_dimb));
  if (num_dim_bias < num_dim) {
    // stop before num_dim - 2 so as to avoid traversing axis m, n
    for (auto i = valid_offset; i < num_dim - 2; ++i) {
      CHECK(!BroadcastBatchDimAndRange(op_name, shape_bias.GetDim(i - valid_offset), shape_out.GetDim(i),
                                       infer_range_bias[i - valid_offset], range_out[i], shape_value_out, range_out[i]),
            NULL, return false);

      shape_out.SetDim(i, shape_value_out);
    }
    return true;
  }
  CopyOutShapeFromInputShape(shape_bias, shape_out, valid_offset);
  copy(infer_range_bias.begin(), infer_range_bias.begin() + valid_offset, range_out.begin());
  // stop before num_dim - 2 so as to avoid traversing axis m, n
  for (auto i = valid_offset; i < num_dim - 2; ++i) {
    CHECK(!BroadcastBatchDimAndRange(op_name, shape_bias.GetDim(i), shape_out.GetDim(i - valid_offset),
                                     infer_range_bias[i], range_out[i - valid_offset], shape_value_out, range_out[i]),
          NULL, return false);

    shape_out.SetDim(i, shape_value_out);
  }
  return true;
}

bool InferShapeBatchMatMul::InferShape() {
  // 1) Static shape
  if (!shape_a.IsUnknownShape() && !shape_b.IsUnknownShape()) {
    // using index - 2 to get m_dim
    int64_t idx_m = num_dima - 2;
    int64_t idx_k_a = num_dima - 1;
    // using index - 2 to get k_dim
    int64_t idx_k_b = num_dimb - 2;
    int64_t idx_n = num_dimb - 1;
    if (trans_a) {
      idx_m = num_dima - 1;
      // using index - 2 to get k_dim
      idx_k_a = num_dima - 2;
    }
    if (trans_b) {
      idx_k_b = num_dimb - 1;
      // using index - 2 to get n_dim
      idx_n = num_dimb - 2;
    }

    if (shape_a.GetDim(idx_k_a) != shape_b.GetDim(idx_k_b)) {
      OP_LOGE(op_name.GetString(), "[InferShape] The k-axis of a(%ld) and b(%ld) tensors must be the same",
              shape_a.GetDim(idx_k_a), shape_b.GetDim(idx_k_b));
      return false;
    }
    CHECK(!InferBatchStatic(), OP_LOGE(op_name.GetString(), "Failed to infer Batch."), return false);

    // using index - 2 to get m_dim in shape_out
    shape_out.SetDim((num_dim - 2), shape_a.GetDim(idx_m));
    shape_out.SetDim((num_dim - 1), shape_b.GetDim(idx_n));
    if (tensordesc_bias != nullptr) {
      CHECK(!InferBias(), OP_LOGE(op_name.GetString(), "Infer bias failed."), return false);
    }
    range_out.clear();
    return true;
  }
  // 2) Dynamic shape
  // 2.1) has {-2}
  bool hasUnKnownDimNum = shape_a.IsUnknownDimNum() || shape_b.IsUnknownDimNum() ||
                          ((tensordesc_bias != nullptr) && shape_bias.IsUnknownDimNum());
  if (hasUnKnownDimNum) {
    shape_out.SetIsUnknownDimNum();
    range_out.clear();
    OP_LOGW(op_name.GetString(), "[InferShape] cannot derive any shape and range information of output");
    return true;
  }

  // 2.2) {..., -1, ...}
  // Initialize infer_shape & infer_range
  CHECK(!InitializeShapeAndRange(shape_a, range_a, infer_range_a),
        OP_LOGE(op_name.GetString(), "Initialize infer_shape & infer_range failed."), return false);
  CHECK(!InitializeShapeAndRange(shape_b, range_b, infer_range_b),
        OP_LOGE(op_name.GetString(), "Initialize infer_shape & infer_range failed."), return false);

  // 3) Infer Batch
  CHECK(!InferBatch(), OP_LOGE(op_name.GetString(), "Failed to infer Batch."), return false);

  // 4) Infer output shape
  CHECK(!InferMKN(), OP_LOGE(op_name.GetString(), "Failed to infer output shape."), return false);

  if (tensordesc_bias != nullptr) {
    // 5) InferBias
    CHECK(!InferBias(), OP_LOGE(op_name.GetString(), "Infer bias failed."), return false);
  }

  // 6) Postprocess
  InferShapeMatMul::SimplifyShapeAndRange(shape_out, range_out);

  return true;
}

bool InferMatmulInputNZ(const Operator &op,
                        vector<vector<int64_t>> &output,
                        bool trans_a, bool trans_b) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return false);
  GeTensorDescPtr tensor_desc_x1 = op_desc->MutableInputDesc(0);
  GeTensorDescPtr tensor_desc_x2 = op_desc->MutableInputDesc(1);
  vector<vector<int64_t> > x1_data_slice = {{}, {}, {}, {}};
  vector<vector<int64_t> > x2_data_slice = {{}, {}, {}, {}};
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return false);
  for (size_t i = 0; i < output.size(); i++) {
    if (output[i].size() <= 1) {
      continue;
    }
    if (i == 0) {
      if (!trans_b) {
        x2_data_slice[1] = output[i];
      } else {
        x2_data_slice[0] = output[i];
      }
      if (!AttrUtils::SetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice)) {
        return false;
      }
      OP_LOGD(opName.GetString(), "infer input in N success");
      return true;
    } else if (i == 1) {
      if (!trans_a) {
        x1_data_slice[0] = output[i];
      } else {
        x1_data_slice[1] = output[i];
      }
      CHECK(!AttrUtils::SetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice),
        OP_LOGE("", "SetListListInt failed."), return false);
      OP_LOGD(opName.GetString(), "infer input in M success");
      return true;
    } else {
      OP_LOGD(opName.GetString(), "cannot support cut in block_n and block_m");
      return false;
    }
  }
  OP_LOGD(opName.GetString(), "no data slice, not need infer input");
  return false;
}

bool InferMatmulInputND(const Operator &op,
                        vector<vector<int64_t>> &output,
                        bool trans_a, bool trans_b) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return false);
  GeTensorDescPtr tensor_desc_x1 = op_desc->MutableInputDesc(0);
  GeTensorDescPtr tensor_desc_x2 = op_desc->MutableInputDesc(1);
  vector<vector<int64_t> > x1_data_slice = {{}, {}};
  vector<vector<int64_t> > x2_data_slice = {{}, {}};
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return false);
  for (size_t i = 0; i < output.size(); i++) {
    if (output[i].size() <= 1) {
      continue;
    }
    if (i == 0) {
      if (!trans_a) {
        x1_data_slice[0] = output[i];
      } else {
        x1_data_slice[1] = output[i];
      }
      CHECK(!AttrUtils::SetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice),
        OP_LOGE("", "SetListListInt failed."), return false);
      OP_LOGD(opName.GetString(), "infer input in M success");
      return true;
    } else if (i == 1) {
      if (!trans_b) {
        x2_data_slice[1] = output[i];
      } else {
        x2_data_slice[0] = output[i];
      }
      CHECK(!AttrUtils::SetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice),
        OP_LOGE("", "SetListListInt failed."), return false);
      OP_LOGD(opName.GetString(), "infer input in N success");
      return true;
    }
  }
  OP_LOGD(opName.GetString(), "no data slice, not need infer input");
  return false;
}

bool InferMatmul(const Operator &op) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return false);
  GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc(0);
  CHECK_PTR_NULL(tensor_desc_y, "tensor y desc", return false);
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return false);

  bool trans_a = false;
  if (!AttrUtils::GetBool(op_desc, "transpose_x1", trans_a)) {
    OP_LOGE(opName.GetString(), "[Plugin][ERROR]%s GetOpAttr transpose_x1 failed!",
            opName.GetString());
    return false;
  }
  bool trans_b = false;
  if (!AttrUtils::GetBool(op_desc, "transpose_x2", trans_b)) {
    OP_LOGE(opName.GetString(), "[Plugin][ERROR]%s GetOpAttr transpose_x2 failed!",
            opName.GetString());
    return false;
  }

  const Format& x1_format = op_desc->GetInputDescPtr(0)->GetFormat();
  const Format& x2_format = op_desc->GetInputDescPtr(1)->GetFormat();

  if (x1_format == FORMAT_FRACTAL_NZ) {
    trans_a = !trans_a;
  }
  if (x2_format == FORMAT_FRACTAL_NZ) {
    trans_b = !trans_b;
  }

  vector<vector<int64_t> > y_data_slice;

  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(opName.GetString(), "no data slice, not need infer input");
    return false;
  }

  if (x1_format == FORMAT_FRACTAL_NZ) {
    if (!InferMatmulInputNZ(op, y_data_slice, trans_a, trans_b)) {
      return false;
    }
  } else {
    if (!InferMatmulInputND(op, y_data_slice, trans_a, trans_b)) {
      return false;
    }
  }
  return true;
}

bool InferBatchMatmulInputNZ(const Operator &op,
                             vector<vector<int64_t>> &output,
                             bool trans_a, bool trans_b,
                             size_t x1_dims, size_t x2_dims) {
  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return false);
  GeTensorDescPtr tensor_desc_x1 = op_desc->MutableInputDesc(0);
  GeTensorDescPtr tensor_desc_x2 = op_desc->MutableInputDesc(1);
  vector<vector<int64_t> > x1_data_slice(x1_dims);
  vector<vector<int64_t> > x2_data_slice(x2_dims);
  size_t y_dims = output.size();
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return false);

  for (size_t i = 0; i < y_dims; i++) {
    if (output[i].size() <= 1) {
      continue;
    }
    // using index -4 to get n_dim of output
    if (i == y_dims - 4) {
      // split n
      if (!trans_b) {
        // using index -3 to get n_dim of x2
        x2_data_slice[x2_dims - 3] = output[i];
      } else {
        // using index -4 to get n_dim of x2
        x2_data_slice[x2_dims - 4] = output[i];
      }
      CHECK(!AttrUtils::SetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice),
        OP_LOGE("", "SetListListInt failed."), return false);
      OP_LOGI(opName.GetString(), "infer input in N success");
      return true;
    // using index -3 to get m_dim of output
    } else if (i == y_dims - 3) {
      if (!trans_a) {
        // using index -4 to get m_dim of x1
        x1_data_slice[x1_dims - 4] = output[i];
      } else {
        // using index -3 to get m_dim of x1
        x1_data_slice[x1_dims - 3] = output[i];
      }
      if (!AttrUtils::SetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice)) {
        return false;
      }
      OP_LOGI(opName.GetString(), "infer input in M success");
      return true;
    // using index -4 to get batch_dim of output
    } else if (i < y_dims - 4) {
      // split batch
      x1_data_slice[i] = output[i];
      CHECK(!AttrUtils::SetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice),
        OP_LOGE("", "SetListListInt failed."), return false);
      if (x2_dims == x1_dims) {
        x2_data_slice[i] = output[i];
        CHECK(!AttrUtils::SetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice),
          OP_LOGE(opName.GetString(), "set data of x2 failed."),
          return false);
      }

      OP_LOGI(opName.GetString(), "infer input in batch success");
      return true;
    } else {
      OP_LOGI(opName.GetString(), "cannot support cut in block_n and block_m");
      return false;
    }
  }
  OP_LOGI(opName.GetString(), "no data slice, not need infer input");
  return false;
}

bool InferBatchMatmulInputND(const Operator &op,
                             vector<vector<int64_t>> &output,
                             bool trans_a, bool trans_b,
                             size_t x1_dims, size_t x2_dims) {
  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_x1 = op_desc->MutableInputDesc(0);
  GeTensorDescPtr tensor_desc_x2 = op_desc->MutableInputDesc(1);
  vector<vector<int64_t> > x1_data_slice(x1_dims);
  vector<vector<int64_t> > x2_data_slice(x2_dims);
  size_t y_dims = output.size();
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return false);

  for (size_t i = 0; i < y_dims; i++) {
    if (output[i].size() <= 1) {
      continue;
    }
    // using index - 2 to get m_dim of output
    if (i == y_dims - 2) {
      // split m
      if (!trans_a) {
        // using index - 2 to get m_dim of x1
        x1_data_slice[x1_dims - 2] = output[i];
      } else {
        x1_data_slice[x1_dims - 1] = output[i];
      }
      CHECK(!AttrUtils::SetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice),
        OP_LOGE("", "SetListListInt failed."), return false);
      OP_LOGI(opName.GetString(), "infer input in M success");
      return true;
    } else if (i == y_dims - 1) {
      // split n
      if (!trans_b) {
        x2_data_slice[x2_dims - 1] = output[i];
      } else {
        // using index - 2 to get n_dim of x2
        x2_data_slice[x2_dims - 2] = output[i];
      }
      CHECK(!AttrUtils::SetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice),
        OP_LOGE("", "SetListListInt failed."), return false);
      OP_LOGI(opName.GetString(), "infer input in N success");
      return true;
    } else {
      x1_data_slice[i] = output[i];
      CHECK(!AttrUtils::SetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice),
        OP_LOGE("", "SetListListInt failed."), return false);
      if (x2_dims == x1_dims) {
        x2_data_slice[i] = output[i];
        CHECK(!AttrUtils::SetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice),
          OP_LOGE(opName.GetString(), "set data of x2 failed."),
          return false);
      }
      OP_LOGI(opName.GetString(), "infer input in Batch success");
      return true;
    }
  }
  OP_LOGI(opName.GetString(), "no data slice, not need infer input");
  return false;
}

bool InferBatchMatmul(const Operator &op) {
  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return false);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc(0);
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return false);

  bool trans_a = false;
  if (!AttrUtils::GetBool(op_desc, "adj_x1", trans_a)) {
    OP_LOGE(opName.GetString(), "[Plugin][ERROR]%s GetOpAttr transposeA failed!",
            opName.GetString());
    return false;
  }

  bool trans_b = false;
  if (!AttrUtils::GetBool(op_desc, "adj_x2", trans_b)) {
    OP_LOGE(opName.GetString(), "[Plugin][ERROR]%s GetOpAttr transposeB failed!",
            opName.GetString());
    return false;
  }

  ge::ConstGeTensorDescPtr tensordesc_x1 = op_desc->GetInputDescPtr(0);
  ge::ConstGeTensorDescPtr tensordesc_x2 = op_desc->GetInputDescPtr(1);
  Format x1_format = tensordesc_x1->GetFormat();
  Format x2_format = tensordesc_x2->GetFormat();
  const GeShape& shape_x1 = tensordesc_x1->GetShape();
  const GeShape& shape_x2 = tensordesc_x2->GetShape();
  size_t x1_dims = shape_x1.GetDimNum();
  size_t x2_dims = shape_x2.GetDimNum();
  if (x1_format == FORMAT_FRACTAL_NZ) {
    trans_a = !trans_a;
  }
  if (x2_format == FORMAT_FRACTAL_NZ) {
    trans_b = !trans_b;
  }

  vector<vector<int64_t> > y_data_slice;
  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(opName.GetString(), "no data slice, not need infer input");
    return false;
  }

  if (x1_format == FORMAT_FRACTAL_NZ) {
    if (!InferBatchMatmulInputNZ(op, y_data_slice, trans_a, trans_b, x1_dims, x2_dims)) {
      return false;
    }
    return true;
  } else {
    if (!InferBatchMatmulInputND(op, y_data_slice, trans_a, trans_b, x1_dims, x2_dims)) {
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
  support_list.push_back(DT_BF16);
  if (CheckInputDataType(op, "x1", support_list) == false) {
    return GRAPH_FAILED;
  }
  if (CheckInputDataType(op, "x2", support_list) == false) {
    return GRAPH_FAILED;
  }
  return InferShapeMatMul::VerifyInputs(op);
}

bool SetMatMulOutputDtype(const AscendString& opName, const ge::GeTensorDescPtr tensordesc_x1,
                          ge::GeTensorDescPtr tensordesc_output) {
  ge::DataType input_dtype = tensordesc_x1->GetDataType();
  ge::DataType output_dtype = tensordesc_output->GetDataType();
  if (input_dtype == DT_FLOAT) {
    OP_LOGW(opName.GetString(), "[Plugin][WARNING]MatMul fp32 op has poor performance!");
  }
  // Split K scenario modifies the output dtype of MatMul as DT_FLOAT in Pytorch Adapther if input dtype is
  // DT_FLOAT16. Also the input format is ND and the output format is Fractal_NZ.
  bool split_k_dtype_correct = input_dtype == DT_FLOAT16 && output_dtype == DT_FLOAT;
  bool split_k_format_correct = false;
  Format input_format = tensordesc_x1->GetFormat();
  Format output_format = tensordesc_output->GetFormat();
  if (AttrUtils::HasAttr(tensordesc_x1, ge::ATTR_NAME_STORAGE_FORMAT) &&
      AttrUtils::HasAttr(tensordesc_output, ge::ATTR_NAME_STORAGE_FORMAT)) {
    Format input_storage_format;
    int64_t input_storage_format_val;
    Format output_storage_format;
    int64_t output_storage_format_val;
    if (!AttrUtils::GetInt(tensordesc_x1, ge::ATTR_NAME_STORAGE_FORMAT, input_storage_format_val)) {
      OP_LOGE(opName.GetString(), "[Plugin][ERROR]%s GetOpAttr %s for input failed!", opName.GetString(),
              "storage_format");
      return false;
    }
    if (!AttrUtils::GetInt(tensordesc_output, ge::ATTR_NAME_STORAGE_FORMAT, output_storage_format_val)) {
      OP_LOGE(opName.GetString(), "[Plugin][ERROR]%s GetOpAttr %s for output failed!", opName.GetString(),
              "storage_format");
      return false;
    }
    input_storage_format = static_cast<Format>(input_storage_format_val);
    output_storage_format = static_cast<Format>(output_storage_format_val);
    if (input_storage_format == FORMAT_ND && output_storage_format == FORMAT_FRACTAL_NZ) {
      split_k_format_correct = true;
    }
  }
  if (input_format == FORMAT_ND && output_format == FORMAT_FRACTAL_NZ) {
    // Used in dynamic tuning.
    split_k_format_correct = true;
  }
  bool split_k = split_k_dtype_correct && split_k_format_correct;
  if (split_k) {
    // The following scenario is used in MatMul split K
    OP_LOGD(opName.GetString(), "split K is needed");
    tensordesc_output->SetDataType(output_dtype);
  } else {
    tensordesc_output->SetDataType(input_dtype);
  }
  return true;
}

// remove completed dimensions
void ReshapeOutput(GeShape& shape_out, vector<std::pair<int64_t, int64_t>>& shape_range_out,
                   const size_t index) {
  size_t shape_range_dim = shape_range_out.size();
  if (shape_range_dim > 0 && (shape_range_dim + index) >= 0) {
    shape_range_out.erase(shape_range_out.begin() + shape_range_dim + index);
  }

  size_t shape_out_dim = shape_out.GetDimNum();
  vector<int64_t> tmp_shape_out(shape_out_dim);
  for (size_t i = 0; i < shape_out_dim; i++) {
    tmp_shape_out[i] = shape_out.GetDim(i);
  }
  tmp_shape_out.erase(tmp_shape_out.begin() + shape_out_dim + index);

  shape_out.SetDimNum(shape_out_dim - 1);
  for (size_t j = 0; j < shape_out_dim - 1; j++) {
    shape_out.SetDim(j, tmp_shape_out[j]);
  }
}

// check flag to deal with complemented situation
void InferComplementedOutput(GeShape& shape_out, vector<std::pair<int64_t, int64_t>>& shape_range_out,
                             const bool shape_x1_reshape_flag, const bool shape_x2_reshape_flag) {
  size_t index = 0;

  // comfirm completed dimension m_index =-2,n_index = -1;
  if (shape_x1_reshape_flag && !shape_x2_reshape_flag) {
    index = kOutputMIndex;
  }

  if (!shape_x1_reshape_flag && shape_x2_reshape_flag) {
    index = kOutputNIndex;
  }

  if (index != 0) {
    ReshapeOutput(shape_out, shape_range_out, index);
  }
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(MatMulInferShape) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);
  ge::GeTensorDescPtr tensordesc_output = op_desc->MutableOutputDesc(0);
  ge::GeTensorDescPtr tensordesc_x1 = op_desc->MutableInputDesc(0);
  ge::GeTensorDescPtr tensordesc_x2 = op_desc->MutableInputDesc(1);

  OP_LOGD(opName.GetString(), "start judge the dtype for matmul!");
  OP_LOGD(opName.GetString(), "%s", GetMatMulInfo(op, "transpose").c_str());
  GeShape shape_x1(tensordesc_x1->MutableShape());
  GeShape shape_x2(tensordesc_x2->MutableShape());
  std::vector<std::pair<int64_t, int64_t>> shape_range_x1;
  std::vector<std::pair<int64_t, int64_t>> shape_range_x2;
  if (shape_x1.IsUnknownShape() || shape_x2.IsUnknownShape()) {
    tensordesc_x1->GetShapeRange(shape_range_x1);
    tensordesc_x2->GetShapeRange(shape_range_x2);
  }

  bool trans_a = false;
  if (!AttrUtils::GetBool(op_desc, "transpose_x1", trans_a)) {
    OP_LOGE(opName.GetString(), "[Plugin][ERROR]%s GetOpAttr %s_x1 failed!", opName.GetString(), "transpose_x1");
    return GRAPH_FAILED;
  }
  bool trans_b = false;
  if (!AttrUtils::GetBool(op_desc, "transpose_x2", trans_b)) {
    OP_LOGE(opName.GetString(), "[Plugin][ERROR]%s GetOpAttr %s_x2 failed!", opName.GetString(), "transpose_x2");
    return GRAPH_FAILED;
  }

  bool shape_x1_reshape_flag = false;
  if (shape_x1.GetDimNum() == 1 && shape_x1.GetDim(0) > 0) {
    int64_t ori_dim = shape_x1.GetDim(0);
    shape_x1.SetDimNum(2);
    shape_x1.SetDim(0, 1);
    shape_x1.SetDim(1, ori_dim);
    shape_range_x1.insert(shape_range_x1.begin(), make_pair<int64_t, int64_t>(1, 1));
    shape_x1_reshape_flag = true;
  }

  bool shape_x2_reshape_flag = false;
  if (shape_x2.GetDimNum() == 1 && shape_x2.GetDim(0) > 0) {
    int64_t ori_dim = shape_x2.GetDim(0);
    shape_x2.SetDimNum(2);
    shape_x2.SetDim(0, ori_dim);
    shape_x2.SetDim(1, 1);
    shape_range_x2.push_back(make_pair<int64_t, int64_t>(1, 1));
    shape_x2_reshape_flag = true;
  }

  GeShape& shape_out = tensordesc_output->MutableShape();
  std::vector<std::pair<int64_t, int64_t>> shape_range_out;
  InferShapeMatMul inferHelper(opName, shape_x1, shape_x2, shape_range_x1, shape_range_x2, trans_a, trans_b, shape_out,
                               shape_range_out);
  CHECK(!inferHelper.InferShape(), OP_LOGE(opName.GetString(), "Failed to infer output shape"), return GRAPH_FAILED);

  InferComplementedOutput(shape_out, shape_range_out, shape_x1_reshape_flag, shape_x2_reshape_flag);

  tensordesc_output->SetShapeRange(shape_range_out);
  if (!SetMatMulOutputDtype(opName, tensordesc_x1, tensordesc_output)) {
    OP_LOGE(opName.GetString(), "[Plugin][ERROR]%s SetMatMulOutputDtype failed!", opName.GetString());
    return GRAPH_FAILED;
  }
  OP_LOGD(opName.GetString(), "the output data type is %s",
          DataTypeToStringDesc(tensordesc_output->GetDataType()).c_str());
  return GRAPH_SUCCESS;
}

// the slice infer
IMPLEMT_INFER_DATA_SLICE(MatMul, MatMulInferDataSlice) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  OP_LOGD(opName.GetString(), "Enter Matmul InferDataSlice.");
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
  support_list.push_back(DT_INT4);
  support_list.push_back(DT_BF16);
  if (CheckInputDataType(op, "x1", support_list) == false) {
    return GRAPH_FAILED;
  }
  if (CheckInputDataType(op, "x2", support_list) == false) {
    return GRAPH_FAILED;
  }

  return InferShapeMatMul::VerifyInputs(op);
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(MatMulV2InferShape) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  OP_LOGD(opName.GetString(), "[MatMulV2 Infershape] Start matmul infershape.");
  OP_LOGD(opName.GetString(), "%s", GetMatMulInfo(op, "transpose").c_str());

  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensordesc_output = op_desc->MutableOutputDesc(0);
  ge::GeTensorDescPtr tensordesc_x1 = op_desc->MutableInputDesc(0);
  ge::GeTensorDescPtr tensordesc_x2 = op_desc->MutableInputDesc(1);
  ge::DataType dtype = tensordesc_x1->GetDataType();
  OP_LOGD(opName.GetString(), "[MatMulV2 Infershape] Check the input dtype.");
  if (dtype == DT_FLOAT) {
    OP_LOGW(opName.GetString(), "[Plugin][WARNING]MatMul fp32 op has poor performance!");
  }

  GeShape shape_x1(tensordesc_x1->MutableShape());
  GeShape shape_x2(tensordesc_x2->MutableShape());
  std::vector<std::pair<int64_t, int64_t>> shape_range_x1;
  std::vector<std::pair<int64_t, int64_t>> shape_range_x2;
  if (shape_x1.IsUnknownShape() || shape_x2.IsUnknownShape()) {
    tensordesc_x1->GetShapeRange(shape_range_x1);
    tensordesc_x2->GetShapeRange(shape_range_x2);
  }

  bool shape_x2_reshape_flag = false;
  if (shape_x2.GetDimNum() == 1 && shape_x2.GetDim(0) > 0) {
    int64_t ori_dim = shape_x2.GetDim(0);
    shape_x2.SetDimNum(2);
    shape_x2.SetDim(0, ori_dim);
    shape_x2.SetDim(1, 1);
    shape_range_x2.push_back(make_pair<int64_t, int64_t>(1, 1));
    shape_x2_reshape_flag = true;
  }

  bool shape_x1_reshape_flag = false;
  if (shape_x1.GetDimNum() == 1 && shape_x1.GetDim(0) > 0) {
    int64_t ori_dim = shape_x1.GetDim(0);
    shape_x1.SetDimNum(2);
    shape_x1.SetDim(0, 1);
    shape_x1.SetDim(1, ori_dim);
    shape_range_x1.insert(shape_range_x1.begin(), make_pair<int64_t, int64_t>(1, 1));
    shape_x1_reshape_flag = true;
  }

  bool trans_b = false;
  if (!AttrUtils::GetBool(op_desc, "transpose_x2", trans_b)) {
    OP_LOGE(opName.GetString(), "[Plugin][ERROR]%s GetOpAttr %s_x2 failed!",
            opName.GetString(), "transpose_x2");
    return GRAPH_FAILED;
  }

  bool trans_a = false;
  if (!AttrUtils::GetBool(op_desc, "transpose_x1", trans_a)) {
    OP_LOGE(opName.GetString(), "[Plugin][ERROR]%s GetOpAttr %s_x1 failed!",
            opName.GetString(), "transpose_x1");
    return GRAPH_FAILED;
  }

  int64_t input_size = 0;
  int64_t hidden_size = 0;
  bool input_size_flag = AttrUtils::GetInt(op_desc, "input_size", input_size);
  bool hidden_size_flag = AttrUtils::GetInt(op_desc, "hidden_size", hidden_size);
  OP_LOGD(opName.GetString(), "input_size[%lld], hidden_size[%lld]", input_size, hidden_size);
  if (input_size_flag && hidden_size_flag) {
    shape_x2.SetDim(1, shape_x1.GetDim(1));
    int64_t align_dim = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE +
                        (hidden_size + BLOCK_SIZE) / BLOCK_SIZE * BLOCK_SIZE;
    shape_x2.SetDim(0, align_dim);
  }

  OP_LOGD(opName.GetString(), "[MatMulV2 Infershape] Check the input shape length.");
  if (shape_x1.GetDims() != UNKNOWN_RANK && shape_x1.GetDims().size() != 2 && shape_x1.GetDims().size() != 4) {
    CUBE_INNER_ERR_REPORT(opName.GetString(), "[Plugin][ERROR]Matmul the first input dims is not 2 or 4!");
    return GRAPH_FAILED;
  }

  GeShape& shape_out = tensordesc_output->MutableShape();
  std::vector<std::pair<int64_t, int64_t>> shape_range_out;
  ge::ConstGeTensorDescPtr tensordesc_bias = op_desc->GetInputDescPtr(kMatMulInputBiasIndex);
  InferShapeMatMul inferHelper(opName, shape_x1, shape_x2, shape_range_x1, shape_range_x2,
                               trans_a, trans_b, shape_out, shape_range_out, tensordesc_bias);
  CHECK(!inferHelper.InferShape(), OP_LOGE(opName.GetString(), "Failed to infer output shape"), return GRAPH_FAILED);

  InferComplementedOutput(shape_out, shape_range_out, shape_x1_reshape_flag, shape_x2_reshape_flag);

  OP_LOGD(opName.GetString(), "[MatMulV2 Infershape] Start to set output shape.");
  tensordesc_output->SetShapeRange(shape_range_out);
  if (tensordesc_x1->GetDataType() == ge::DT_INT8 || tensordesc_x1->GetDataType() == ge::DT_INT4) {
    tensordesc_output->SetDataType(ge::DT_INT32);
  } else {
    tensordesc_output->SetDataType(tensordesc_x1->GetDataType());
  }
  OP_LOGD(opName.GetString(), "[MatMulV2 Infershape] End MatMulV2 infershape.");

  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFORMAT_FUNC(MatMulV2, MatMulV2InferFormat) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  OP_LOGD(opName.GetString(), "[MatMulV2 Inferformat] Finaly input format is %d", FORMAT_ND);
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
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  OP_LOGD(opName.GetString(), "Enter MatmulV2 InferDataSlice.");
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
  support_list.push_back(DT_FLOAT);
  support_list.push_back(DT_INT8);
  support_list.push_back(DT_INT32);
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  OP_LOGD(opName.GetString(), "[GEMM Verify] Start GEMM Verify.");

  if (CheckInputDataType(op, "a", support_list) == false) {
    return GRAPH_FAILED;
  }
  if (CheckInputDataType(op, "b", support_list) == false) {
    TbeInputDataTypeErrReport(opName.GetString(), "b", "float16,float32,int8,int32",
                              DataTypeToStringDesc(op.GetInputDescByName("b").GetDataType()));
    return GRAPH_FAILED;
  }
  if (CheckInputDataType(op, "c", support_list) == false) {
    TbeInputDataTypeErrReport(opName.GetString(), "c", "float16,float32,int8,int32",
                              DataTypeToStringDesc(op.GetInputDescByName("c").GetDataType()));
    return GRAPH_FAILED;
  }
  if (CheckInputDataType(op, "alpha", support_list) == false) {
    TbeInputDataTypeErrReport(opName.GetString(), "alpha", "float16,float32,int8,int32",
                              DataTypeToStringDesc(op.GetInputDescByName("alpha").GetDataType()));
    return GRAPH_FAILED;
  }
  if (CheckInputDataType(op, "beta", support_list) == false) {
    TbeInputDataTypeErrReport(opName.GetString(), "beta", "float16,float32,int8,int32",
                              DataTypeToStringDesc(op.GetInputDescByName("beta").GetDataType()));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(GemmInferShape) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  OP_LOGD(opName.GetString(), "[GEMM Infershape] Start GEMM infershape.");
  TensorDesc tensordesc_output = op.GetOutputDescByName("y");
  ge::TensorDesc inputTensorDescC = op.GetInputDescByName("c");
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
  return InferShapeMatMul::VerifyInputs(op);
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(BatchMatMulInferShape) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  OP_LOGD(opName.GetString(), "%s", GetMatMulInfo(op, "adj").c_str());

  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);
  ge::GeTensorDescPtr tensordesc_out = op_desc->MutableOutputDesc(0);
  CHECK_PTR_NULL(tensordesc_out, "tensor out desc", return GRAPH_FAILED);
  ge::GeTensorDescPtr tensordesc_x1 = op_desc->MutableInputDesc(0);
  CHECK_PTR_NULL(tensordesc_x1, "tensor x1 desc", return GRAPH_FAILED);
  ge::GeTensorDescPtr tensordesc_x2 = op_desc->MutableInputDesc(1);
  CHECK_PTR_NULL(tensordesc_x2, "tensor x2 desc", return GRAPH_FAILED);
  ge::DataType dtype = tensordesc_x1->GetDataType();

  GeShape shape_x1(tensordesc_x1->MutableShape());
  GeShape shape_x2(tensordesc_x2->MutableShape());
  std::vector<std::pair<int64_t, int64_t>> shape_range_x1;
  std::vector<std::pair<int64_t, int64_t>> shape_range_x2;
  if (shape_x1.IsUnknownShape() || shape_x2.IsUnknownShape()) {
    tensordesc_x1->GetShapeRange(shape_range_x1);
    tensordesc_x2->GetShapeRange(shape_range_x2);
  }

  bool trans_a = false;
  if (!AttrUtils::GetBool(op_desc, "adj_x1", trans_a)) {
    OP_LOGE(opName.GetString(), "[Plugin][ERROR]%s GetOpAttr %s_x1 failed!", opName.GetString(), "adj_x1");
    return GRAPH_FAILED;
  }

  bool trans_b = false;
  if (!AttrUtils::GetBool(op_desc, "adj_x2", trans_b)) {
    OP_LOGE(opName.GetString(), "[Plugin][ERROR]%s GetOpAttr %s_x2 failed!", opName.GetString(), "adj_x2");
    return GRAPH_FAILED;
  }

  size_t dim_num_x1 = shape_x1.GetDimNum();
  size_t dim_num_x2 = shape_x2.GetDimNum();
  auto dim_num = std::max(dim_num_x1, dim_num_x2);
  bool any_unknown_rank = shape_x1.IsUnknownDimNum() || shape_x2.IsUnknownDimNum();
  if (!any_unknown_rank && (dim_num < 1 || dim_num > kBatchMatmulMaxShapeSize)) {
    CUBE_INNER_ERR_REPORT(opName.GetString(), "[Infershape]The shape can only be in the range of 1 to 8.");
    return GRAPH_FAILED;
  }

  bool shape_x1_reshape_flag = false;
  if (shape_x1.GetDimNum() == 1 && shape_x1.GetDim(0) > 0) {
    int64_t ori_dim = shape_x1.GetDim(0);
    shape_x1.SetDimNum(2);
    shape_x1.SetDim(0, 1);
    shape_x1.SetDim(1, ori_dim);
    shape_range_x1.insert(shape_range_x1.begin(), make_pair<int64_t, int64_t>(1, 1));
    shape_x1_reshape_flag = true;
  }

  bool shape_x2_reshape_flag = false;
  if (shape_x2.GetDimNum() == 1 && shape_x2.GetDim(0) > 0) {
    int64_t ori_dim = shape_x2.GetDim(0);
    shape_x2.SetDimNum(2);
    shape_x2.SetDim(0, ori_dim);
    shape_x2.SetDim(1, 1);
    shape_range_x2.push_back(make_pair<int64_t, int64_t>(1, 1));
    shape_x2_reshape_flag = true;
  }

  GeShape& shape_out = tensordesc_out->MutableShape();
  std::vector<std::pair<int64_t, int64_t>> shape_range_out;
  InferShapeBatchMatMul BatchMatMulInfer(opName, shape_x1, shape_x2, shape_range_x1, shape_range_x2, trans_a, trans_b,
                                         shape_out, shape_range_out);
  CHECK(!BatchMatMulInfer.InferShape(), OP_LOGE(opName.GetString(), "Failed to infer output shape"),
        return GRAPH_FAILED);

  InferComplementedOutput(shape_out, shape_range_out, shape_x1_reshape_flag, shape_x2_reshape_flag);

  tensordesc_out->SetShapeRange(shape_range_out);
  if (dtype == ge::DT_INT8) {
    tensordesc_out->SetDataType(ge::DT_INT32);
  } else {
    tensordesc_out->SetDataType(dtype);
  }
  return GRAPH_SUCCESS;
}

// the slice infer
IMPLEMT_INFER_DATA_SLICE(BatchMatMul, BatchMatMulInferDataSlice) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  OP_LOGD(opName.GetString(), "Enter BatchMatmul InferDataSlice.");
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
  return InferShapeMatMul::VerifyInputs(op);
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(BatchMatMulV2InferShape) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  OP_LOGD(opName.GetString(), "%s", GetMatMulInfo(op, "adj").c_str());

  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);
  ge::GeTensorDescPtr tensordesc_out = op_desc->MutableOutputDesc(0);
  CHECK_PTR_NULL(tensordesc_out, "tensor out desc", return GRAPH_FAILED);
  ge::GeTensorDescPtr tensordesc_x1 = op_desc->MutableInputDesc(0);
  CHECK_PTR_NULL(tensordesc_x1, "tensor x1 desc", return GRAPH_FAILED);
  ge::GeTensorDescPtr tensordesc_x2 = op_desc->MutableInputDesc(1);
  CHECK_PTR_NULL(tensordesc_x2, "tensor x2 desc", return GRAPH_FAILED);
  ge::DataType dtype = tensordesc_x1->GetDataType();

  GeShape shape_x1(tensordesc_x1->MutableShape());
  GeShape shape_x2(tensordesc_x2->MutableShape());
  std::vector<std::pair<int64_t, int64_t>> shape_range_x1;
  std::vector<std::pair<int64_t, int64_t>> shape_range_x2;
  if (shape_x1.IsUnknownShape() || shape_x2.IsUnknownShape()) {
    tensordesc_x1->GetShapeRange(shape_range_x1);
    tensordesc_x2->GetShapeRange(shape_range_x2);
  }

  bool trans_a = false;
  if (!AttrUtils::GetBool(op_desc, "adj_x1", trans_a)) {
    OP_LOGE(opName.GetString(), "[Plugin][ERROR]%s GetOpAttr %s_x1 failed!", opName.GetString(), "adj_x1");
    return GRAPH_FAILED;
  }

  bool trans_b = false;
  if (!AttrUtils::GetBool(op_desc, "adj_x2", trans_b)) {
    OP_LOGE(opName.GetString(), "[Plugin][ERROR]%s GetOpAttr %s_x2 failed!", opName.GetString(), "adj_x2");
    return GRAPH_FAILED;
  }

  size_t dim_num_x1 = shape_x1.GetDimNum();
  size_t dim_num_x2 = shape_x2.GetDimNum();
  auto dim_num = std::max(dim_num_x1, dim_num_x2);
  bool any_unknown_rank = shape_x1.IsUnknownDimNum() || shape_x2.IsUnknownDimNum();
  ge::ConstGeTensorDescPtr tensordesc_bias = op_desc->GetInputDescPtr(kMatMulInputBiasIndex);
  if (tensordesc_bias != nullptr) {
    const GeShape& shape_bias = tensordesc_bias->GetShape();
    any_unknown_rank = any_unknown_rank || shape_bias.IsUnknownDimNum();
  }
  if (!any_unknown_rank && (dim_num < 1 || dim_num > 8)) {
    CUBE_INNER_ERR_REPORT(opName.GetString(), "[Infershape]The shape can only be in the range of 1 to 8.");
    return GRAPH_FAILED;
  }

  bool shape_x1_reshape_flag = false;
  if (shape_x1.GetDimNum() == 1 && shape_x1.GetDim(0) > 0) {
    int64_t ori_dim = shape_x1.GetDim(0);
    shape_x1.SetDimNum(2);
    shape_x1.SetDim(0, 1);
    shape_x1.SetDim(1, ori_dim);
    shape_range_x1.insert(shape_range_x1.begin(), make_pair<int64_t, int64_t>(1, 1));
    shape_x1_reshape_flag = true;
  }

  bool shape_x2_reshape_flag = false;
  if (shape_x2.GetDimNum() == 1 && shape_x2.GetDim(0) > 0) {
    int64_t ori_dim = shape_x2.GetDim(0);
    shape_x2.SetDimNum(2);
    shape_x2.SetDim(0, ori_dim);
    shape_x2.SetDim(1, 1);
    shape_range_x2.push_back(make_pair<int64_t, int64_t>(1, 1));
    shape_x2_reshape_flag = true;
  }

  GeShape& shape_out = tensordesc_out->MutableShape();
  std::vector<std::pair<int64_t, int64_t>> shape_range_out;
  InferShapeBatchMatMul BatchMatMulV2Infer(opName, shape_x1, shape_x2, shape_range_x1, shape_range_x2, trans_a, trans_b,
                                           shape_out, shape_range_out, tensordesc_bias);
  CHECK(!BatchMatMulV2Infer.InferShape(), OP_LOGE(opName.GetString(), "Failed to infer output shape"),
        return GRAPH_FAILED);

  InferComplementedOutput(shape_out, shape_range_out, shape_x1_reshape_flag, shape_x2_reshape_flag);

  tensordesc_out->SetShapeRange(shape_range_out);
  if (dtype == ge::DT_INT8) {
    tensordesc_out->SetDataType(ge::DT_INT32);
  } else {
    tensordesc_out->SetDataType(dtype);
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFORMAT_FUNC(BatchMatMulV2, BatchMatMulV2InferFormat) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  OP_LOGD(opName.GetString(), "[BatchMatMulV2 Inferformat] Finaly input format is %d", FORMAT_ND);
  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);

  ge::GeTensorDescPtr tensordesc_input = op_desc->MutableInputDesc(0);
  CHECK_PTR_NULL(tensordesc_input, "tensordesc input", return GRAPH_FAILED);
  ge::GeTensorDescPtr tensordesc_input_2 = op_desc->MutableInputDesc(1);
  CHECK_PTR_NULL(tensordesc_input_2, "tensordesc input_2", return GRAPH_FAILED);

  tensordesc_input->SetOriginFormat(FORMAT_ND);
  tensordesc_input->SetFormat(FORMAT_ND);
  tensordesc_input_2->SetOriginFormat(FORMAT_ND);
  tensordesc_input_2->SetFormat(FORMAT_ND);

  return GRAPH_SUCCESS;
}

// the slice infer
IMPLEMT_INFER_DATA_SLICE(BatchMatMulV2, BatchMatMulV2InferDataSlice) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  OP_LOGD(opName.GetString(), "Enter BatchMatmulV2 InferDataSlice.");
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
  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_desc == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT("DiagPart", GetInputInvalidErrMsg("op_desc")),
        return GRAPH_FAILED);
  ge::ConstGeTensorDescPtr input_x_desc = op_desc->GetInputDescPtr(0);
  CHECK(input_x_desc == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT("DiagPart", GetInputInvalidErrMsg("x")),
        return GRAPH_FAILED);
  const GeShape &input_shape = input_x_desc->GetShape();
  const int64_t input_to_output_dims_times = 2;
  int64_t output_shape_len = input_shape.GetDimNum() / input_to_output_dims_times;
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc(0);
  GeShape &output_shape = output_desc->MutableShape();
  DataType input_dtype = input_x_desc->GetDataType();

  if (input_shape.IsUnknownDimNum()) {
    output_desc->SetShape(input_shape);
  }else {
    output_shape.SetDimNum(output_shape_len);
    for (int64_t i = 0; i < output_shape_len; i++) {
      output_shape.SetDim(i, input_shape.GetDim(i));
    }
  }
  if (input_shape.IsUnknownShape()) {
    std::vector<std::pair<int64_t, int64_t>> shape_range;
    input_x_desc->GetShapeRange(shape_range);
    for (unsigned i = 0; i < shape_range.size(); i++) {
      if (shape_range[i].first > 0) {
        shape_range[i].first = shape_range[i].first;
      }
      if (shape_range[i].second > 0) {
        shape_range[i].second = shape_range[i].second;
      }
    }
    output_desc->SetShapeRange(shape_range);
  }
  output_desc->SetShape(output_shape);
  output_desc->SetDataType(input_dtype);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(DiagPart, DiagPartInferShape);
// ----------------DiagPart END-------------------

// ----------------DiagPartD-------------------
IMPLEMT_COMMON_INFERFUNC(DiagPartDInferShape) {
  Shape shape = op.GetInputDescByName("x").GetShape();
  DataType input_dtype = op.GetInputDescByName("x").GetDataType();
  std::vector<int64_t> dim_vector;
  for (size_t i = 0; i < shape.GetDimNum() / 2; i++) {
    dim_vector.push_back(shape.GetDim(i));
  }
  Shape output_shape(dim_vector);
  TensorDesc td = op.GetOutputDescByName("y");
  td.SetShape(output_shape);
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(DiagPartD, DiagPartDInferShape);
// ----------------DiagPartD END-------------------

// ---------------MatrixDiag-------------------
IMPLEMT_COMMON_INFERFUNC(MatrixDiagInferShape) {
  Shape input_shape = op.GetInputDescByName("x").GetShape();
  DataType input_dtype = op.GetInputDescByName("x").GetDataType();
  TensorDesc td = op.GetOutputDescByName("y");
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
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  Shape x_shape = op.GetInputDescByName("x").GetShape();
  Shape assist_shape = op.GetInputDescByName("assist").GetShape();
  TensorDesc td = op.GetOutputDescByName("y");
  std::vector<int64_t> dims_x = x_shape.GetDims();
  std::vector<int64_t> dims_assist = assist_shape.GetDims();
  if (dims_x.size() < dims_assist.size()) {
    std::vector<int64_t> dims_tmp = dims_x;
    dims_x = dims_assist;
    dims_assist = dims_tmp;
  }

  if (dims_x.size() != dims_assist.size()) {
    auto dec = dims_x.size() - dims_assist.size();
    for (size_t i = 0; i < dec; i++) {
      dims_assist.insert(dims_assist.begin(), (int64_t)1);
    }
  }

  std::vector<int64_t> dim_vec;
  for (size_t i = 0; i < dims_x.size(); i++) {
    if ((dims_x[i] != dims_assist[i]) && (dims_x[i] != 1) && (dims_assist[i] != 1)) {
      std::string err_msg = OtherErrMsg(ConcatString("The dimensions does not match the broadcast rule(",
        dims_x[i], ", ", dims_assist[i], ")"));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), err_msg);
    }

    int64_t dims = dims_x[i] > dims_assist[i] ? dims_x[i] : dims_assist[i];
    dim_vec.push_back(dims);
  }
  td.SetShape(ge::Shape(dim_vec));
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MatrixDiagD, MatrixDiagDVerify) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  DataType input_diagonal_dtype = op.GetInputDesc(0).GetDataType();
  DataType input_help_dtype = op.GetInputDesc(1).GetDataType();
  if (input_diagonal_dtype != input_help_dtype) {
    string err_msg1 = ConcatString("the inputs of diagonal and help should be the same dtype! input_diagonal_dtype:",
      input_diagonal_dtype, ", input_diagonal_dtype:", input_diagonal_dtype);
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixDiagD, MatrixDiagDInferShape);
VERIFY_FUNC_REG(MatrixDiagD, MatrixDiagDVerify);
// ----------------MatrixDiagD END----------------
// ----------------AttentionScore------------
IMPLEMT_COMMON_INFERFUNC(AttentionScoreInferShape) {
  auto attention_score_output = op.GetOutputDescByName("attention_score");
  auto softmax_output = op.GetOutputDescByName("softmax_output");
  vector<int64_t> query_dims = op.GetInputDescByName("query").GetShape().GetDims();
  vector<int64_t> padding_mask_dims = op.GetInputDescByName("padding_mask").GetShape().GetDims();
  if (query_dims.size() != 4 || padding_mask_dims.size() != 4) {
    OP_LOGE(op.GetName().c_str(),
            "The input query and padding_mask only support 4D.");
    return GRAPH_FAILED;
  }
  vector<int64_t> attention_score_dims = {query_dims[0] * query_dims[2], query_dims[3] * query_dims[1]};
  attention_score_output.SetShape(ge::Shape(attention_score_dims));
  attention_score_output.SetDataType(ge::DT_FLOAT16);
  (void)op.UpdateOutputDesc("attention_score", attention_score_output);

  vector<int64_t> softmax_dims = {query_dims[0], query_dims[1], query_dims[2], padding_mask_dims[3]};
  softmax_output.SetShape(ge::Shape(softmax_dims));
  softmax_output.SetDataType(ge::DT_FLOAT16);
  (void)op.UpdateOutputDesc("softmax_output", softmax_output);

  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(AttentionScore, AttentionScoreInferShape);
// ----------------AttentionScore End------------
// ----------------MatrixDiagPart--------------------
IMPLEMT_COMMON_INFERFUNC(MatrixDiagPartInferShape) {
  Shape shape = op.GetInputDescByName("x").GetShape();
  DataType input_dtype = op.GetInputDescByName("x").GetDataType();
  Format input_format = op.GetInputDescByName("x").GetFormat();
  std::vector<int64_t> dim_vector;
  int64_t dimsInput_1 = shape.GetDimNum() - 1;
  // using index - 2 to get input2
  int64_t dimsInput_2 = shape.GetDimNum() - 2;
  int64_t dimNums_1 = shape.GetDim(dimsInput_1);
  int64_t dimNums_2 = shape.GetDim(dimsInput_2);
  if (dimNums_1 > dimNums_2) {
    for (size_t i = 0; i < shape.GetDimNum() - 1; i++) {
      dim_vector.push_back(shape.GetDim(i));
    }
  } else {
    // using index - 2 to get input2
    for (size_t i = 0; i < shape.GetDimNum() - 2; i++) {
      dim_vector.push_back(shape.GetDim(i));
    }
    dim_vector.push_back(dimNums_1);
  }
  Shape output_shape(dim_vector);
  TensorDesc td = op.GetOutputDescByName("y");
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
  Shape shape = op.GetInputDescByName("x").GetShape();
  DataType input_dtype = op.GetInputDescByName("x").GetDataType();
  Format input_format = op.GetInputDescByName("x").GetFormat();
  std::vector<int64_t> dim_vector;
  int64_t dimsInput_1 = shape.GetDimNum() - 1;
  // using index - 2 to get input2
  int64_t dimsInput_2 = shape.GetDimNum() - 2;
  int64_t dimNums_1 = shape.GetDim(dimsInput_1);
  int64_t dimNums_2 = shape.GetDim(dimsInput_2);
  if (dimNums_1 > dimNums_2) {
    for (size_t i = 0; i < shape.GetDimNum() - 1; i++) {
      dim_vector.push_back(shape.GetDim(i));
    }
  } else {
    // using index - 2 to get input2
    for (size_t i = 0; i < shape.GetDimNum() - 2; i++) {
      dim_vector.push_back(shape.GetDim(i));
    }
    dim_vector.push_back(dimNums_1);
  }
  Shape output_shape(dim_vector);
  TensorDesc td = op.GetOutputDescByName("y");
  td.SetShape(output_shape);
  td.SetDataType(input_dtype);
  td.SetFormat(input_format);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MatrixDiagPartD, MatrixDiagPartDVerify) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  DataType input_diagonal_dtype = op.GetInputDesc(0).GetDataType();
  DataType input_help_dtype = op.GetInputDesc(1).GetDataType();
  if (input_diagonal_dtype != input_help_dtype) {
    string err_msg1 = ConcatString("the inputs of diagonal and help should be the same dtype! input_diagonal_dtype:",
      input_diagonal_dtype, ", input_help_dtype:", input_help_dtype);
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), err_msg);
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
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  DataType input_matrix_dtype = op.GetInputDesc(0).GetDataType();
  DataType input_diagonal_dtype = op.GetInputDesc(1).GetDataType();
  DataType input_help_dtype = op.GetInputDesc(2).GetDataType();
  if ((input_matrix_dtype != input_diagonal_dtype) || (input_matrix_dtype != input_help_dtype)) {
    string err_msg1 = ConcatString("the inputs of matrix and diagonal should be the same dtype! input_matrix_dtype:",
      input_matrix_dtype, ", input_diagonal_dtype:", input_diagonal_dtype, ", input_help_dtype:", input_help_dtype);
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), err_msg);
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
  Shape var_shape = op.GetInputDescByName("x").GetShape();
  DataType input_dtype = op.GetInputDescByName("x").GetDataType();
  TensorDesc td = op.GetOutputDescByName("y");
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
  return GRAPH_SUCCESS;
}
IMPLEMT_INFERFUNC(ScatterElements, ScatterElementsInferShape) {
  Shape data_shape = op.GetInputDescByName("data").GetShape();
  DataType input_dtype = op.GetInputDescByName("data").GetDataType();
  TensorDesc td = op.GetOutputDescByName("y");
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

// ------------------ScatterAddWithAxis---------------------
IMPLEMT_VERIFIER(ScatterAddWithAxis, ScatterAddWithAxisVerify) {
  if (!CheckTwoInputDtypeSame(op, "var", "updates")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ScatterAddWithAxisInferShape) {
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

COMMON_INFER_FUNC_REG(ScatterAddWithAxis, ScatterAddWithAxisInferShape);
VERIFY_FUNC_REG(ScatterAddWithAxis, ScatterAddWithAxisVerify);
// --------------ScatterAdd END------------------

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
  Shape var_shape = op.GetInputDescByName("x").GetShape();
  DataType input_dtype = op.GetInputDescByName("x").GetDataType();
  TensorDesc td = op.GetOutputDescByName("y");
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
  Shape var_shape = op.GetInputDescByName("x").GetShape();
  DataType input_dtype = op.GetInputDescByName("x").GetDataType();
  TensorDesc td = op.GetOutputDescByName("y");
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
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  int64_t num_classes;
  auto output_dtype = DT_FLOAT;
  std::string output_dtype_str = "float32";
  if (op.GetAttr("num_classes", num_classes) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("num_classes");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), err_msg);
    return GRAPH_FAILED;
  }
  if (op.GetAttr("dtype", output_dtype_str) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("dtype");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), err_msg);
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
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<int64_t> out_shape;
  out_shape.push_back(num_classes);
  out_shape.push_back(num_classes);
  vector<int64_t> y_shape(out_shape);
  TensorDesc td = op.GetOutputDescByName("y");
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
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  Shape input_shape;
  auto input_tensor_desc = op.GetInputDesc(0);
  CHECK(WithRankAtLeast(input_tensor_desc, 2, input_shape, opName.GetString()) != GRAPH_SUCCESS,
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(opName.GetString(), ConcatString(
      "failed to call WithRankAtLeast function, ",
      "input[input] rank must be at least 2D, but got rank[",
      op.GetInputDesc(0).GetShape().GetDimNum(), "]")),
    return GRAPH_FAILED);

  Shape k_shape;
  auto k_tensor_desc = op.GetInputDesc(1);
  CHECK(WithRankAtMost(k_tensor_desc, 1, k_shape, opName.GetString()) != GRAPH_SUCCESS,
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(opName.GetString(), ConcatString(
      "failed to call WithRankAtMost function, ",
      "input[k] rank must be at most 1D, but got rank[",
      op.GetInputDesc(1).GetShape().GetDimNum(), "]")),
    return GRAPH_FAILED);

  Shape padding_value_shape;
  auto padding_value_tensor_desc = op.GetInputDesc(kInputPadIndex);
  CHECK(WithRank(padding_value_tensor_desc, 0, padding_value_shape, opName.GetString()) != GRAPH_SUCCESS,
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(opName.GetString(), ConcatString("failed to call WithRank function, ",
      "input[padding_value] rank must be 0, but got rank[",
      op.GetInputDesc(kInputPadIndex).GetShape().GetDimNum(), "]")),
    return GRAPH_FAILED);

  TensorDesc output_desc = op.GetOutputDescByName("diagonal");

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
    } else if (num_elements == kInputKMaxSize) {
      int32_t* data = reinterpret_cast<int32_t*>(k_tensor.GetData());
      lower_diag_index = *(data);
      upper_diag_index = *(data + 1);
    } else {
      std::string err_msg = ConcatString(
        "the input [k] must be scalar or a vector with one or two elements, ",
        "but it has [", num_elements, "] elements. ");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), err_msg);
      return GRAPH_PARAM_INVALID;
    }
  }

  CHECK(lower_diag_index > upper_diag_index,
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), ConcatString(
      "the variable [lower_diag_index] of input [k] 1th value[",
      lower_diag_index,  "] must be  not greater than [upper_diag_index] ",
      "of input [k] 2th value[", upper_diag_index, "]")),
    return GRAPH_PARAM_INVALID);

  auto input_dims = input_shape.GetDims();
  const int32_t input_rank = input_shape.GetDimNum();
  const int32_t num_rows = input_dims[input_rank - 2];
  const int32_t num_cols = input_dims[input_rank - 1];
  int64_t max_diag_len = ge::UNKNOWN_DIM;
  if (num_rows != ge::UNKNOWN_DIM && num_cols != ge::UNKNOWN_DIM) {
    CHECK(lower_diag_index != 0 && (-num_rows >= lower_diag_index || lower_diag_index >= num_cols),
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), ConcatString(
        "the variable [lower_diag_index] of input [k] value[",
        lower_diag_index, "] is illegal, ",
        "should be 0 or in range(", -num_rows, ", ", num_cols, ")")),
      return GRAPH_PARAM_INVALID);

    CHECK(upper_diag_index != 0 && (-num_rows >= upper_diag_index || upper_diag_index >= num_cols),
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), ConcatString(
        "the variable [upper_diag_index] of input [k] value[",
        upper_diag_index, "] is illegal, ",
        "should be 0 or in range(", -num_rows, ", ", num_cols, ")")),
      return GRAPH_PARAM_INVALID);
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
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  Shape input_shape;
  auto input_tensor_desc = op.GetInputDesc(0);
  CHECK(WithRankAtLeast(input_tensor_desc, 2, input_shape, opName.GetString()) != GRAPH_SUCCESS,
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(opName.GetString(), ConcatString(
      "failed to call WithRankAtLeast function, ",
      "input[input] rank must be at least 2D, but got rank[",
      op.GetInputDesc(0).GetShape().GetDimNum(), "]")),
    return GRAPH_FAILED);

  Shape diagonal_shape;
  auto diagonal_tensor_desc = op.GetInputDesc(1);
  CHECK(WithRankAtLeast(diagonal_tensor_desc, 1, diagonal_shape, opName.GetString()) != GRAPH_SUCCESS,
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(opName.GetString(), ConcatString(
      "failed to call WithRankAtLeast function, ",
      "input[diagonal] rank must be at least 1D, but got rank[",
      op.GetInputDesc(1).GetShape().GetDimNum(), "]")),
    return GRAPH_FAILED);

  Shape k_shape;
  auto k_tensor_desc = op.GetInputDesc(kInputKIndex);
  CHECK(WithRankAtMost(k_tensor_desc, 1, k_shape, opName.GetString()) != GRAPH_SUCCESS,
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(opName.GetString(), ConcatString(
      "failed to call WithRankAtMost function, ",
      "input[k] rank must be at most 1D, but got rank[",
      op.GetInputDesc(kInputKIndex).GetShape().GetDimNum(), "]")),
    return GRAPH_FAILED);

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
        AICPU_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), err_msg);
        return GRAPH_PARAM_INVALID;
      }
    }

    CHECK(lower_diag_index > upper_diag_index,
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), ConcatString(
        "the variable [lower_diag_index] of input[k] 1th value[",
        lower_diag_index, "] must be ",
        "not greater than [upper_diag_index] of input[k] 2th value[",
        upper_diag_index, "]")),
      return GRAPH_PARAM_INVALID);
  }

  if (RankKnown(input_shape)) {
    auto input_rank = input_shape.GetDimNum();
    if (k_index_known) {
      if (WithRank(diagonal_tensor_desc, (lower_diag_index == upper_diag_index ? input_rank - 1 : input_rank),
                   diagonal_shape, opName.GetString()) != GRAPH_SUCCESS) {
        std::string err_msg = ConcatString(
          "failed to call WithRank function, ",
          "input[diagonal] rank must be [",
          (lower_diag_index == upper_diag_index ? input_rank - 1 : input_rank),
          "], but got rank[", diagonal_tensor_desc.GetShape().GetDimNum(), "]");
        AICPU_INFER_SHAPE_CALL_ERR_REPORT(opName.GetString(), err_msg);
        return GRAPH_FAILED;
      } else {
        CHECK(WithRankAtLeast(diagonal_tensor_desc, input_rank - 1, diagonal_shape, opName.GetString()) !=
            GRAPH_SUCCESS,
          AICPU_INFER_SHAPE_CALL_ERR_REPORT(opName.GetString(), ConcatString(
            "failed to call WithRankAtLeast function, ",
            "input[diagonal] rank must be at least ", input_rank - 1,
            "D, but got rank[", diagonal_tensor_desc.GetShape().GetDimNum(), "]")),
          return GRAPH_FAILED);

        CHECK(WithRankAtMost(diagonal_tensor_desc, input_rank, diagonal_shape, opName.GetString()) != GRAPH_SUCCESS,
          AICPU_INFER_SHAPE_CALL_ERR_REPORT(opName.GetString(), ConcatString(
            "failed to call WithRankAtMost function, ",
            "input[diagonal] rank must be at most ",
            input_rank, "D, but got rank[",
            diagonal_tensor_desc.GetShape().GetDimNum(), "]")),
          return GRAPH_FAILED);
      }

      auto input_dims = input_shape.GetDims();
      const int32_t num_rows = input_dims[input_rank - 2];
      const int32_t num_cols = input_dims[input_rank - 1];
      if (num_rows != ge::UNKNOWN_DIM && num_cols != ge::UNKNOWN_DIM) {
        CHECK(lower_diag_index != 0 && (-num_rows >= lower_diag_index || lower_diag_index >= num_cols),
          AICPU_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), ConcatString(
            "the variable [lower_diag_index] of input[k] value[",
            lower_diag_index, "] is illegal, ",
            "should be 0 or in range(", -num_rows, ", ", num_cols, ")")),
          return GRAPH_PARAM_INVALID);

        CHECK(upper_diag_index != 0 && (-num_rows >= upper_diag_index || upper_diag_index >= num_cols),
          AICPU_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), ConcatString(
            "the variable [upper_diag_index] of input[k] value[",
            upper_diag_index, "] is illegal, ",
            "should be 0 or in range(", -num_rows, ", ", num_cols, ")")),
          return GRAPH_PARAM_INVALID);
      }
    }
  }

  auto output_desc = op.GetOutputDescByName("output");
  Shape output_shape = input_shape;
  if (RankKnown(diagonal_shape) && !FullyDefined(input_shape)) {
    Shape diagonal_prefix_shape;
    CHECK(SubShape(diagonal_shape, 0, (lower_diag_index == upper_diag_index ? -1 : -2), 1, diagonal_prefix_shape,
                 opName.GetString()) != GRAPH_SUCCESS,
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(opName.GetString(), ConcatString(
        "failed to call SubShape function, input[diagonal] shape",
        DebugString(diagonal_shape.GetDims()),
        ", end[", (lower_diag_index == upper_diag_index ? -1 : -2),
        "] is invaild")),
      return GRAPH_FAILED);

    CHECK(Concatenate(diagonal_prefix_shape, UnknownShapeOfRank(2), diagonal_shape) != GRAPH_SUCCESS,
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), ConcatString("failed to call Concatenate function ",
        "to concatenate prefix diagonal shape",
        DebugString(diagonal_prefix_shape.GetDims()),
        " and 2D unknown_shape")),
      return GRAPH_FAILED);

    CHECK(Merge(input_shape, diagonal_shape, output_shape, opName.GetString()) != GRAPH_SUCCESS,
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(opName.GetString(), ConcatString(
        "failed to call Merge function to merge the 0th input[input] shape",
        DebugString(input_shape.GetDims()),
        " and the 1st input[diagonal]'s shape",
        DebugString(diagonal_shape.GetDims()))),
      return GRAPH_FAILED);
  }
  output_desc.SetShape(output_shape);
  output_desc.SetDataType(input_tensor_desc.GetDataType());
  (void)op.UpdateOutputDesc("output", output_desc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixSetDiagV2, MatrixSetDiagV2InferShape);

INFER_FUNC_REG(MatrixSetDiagV3, MatrixSetDiagV2InferShape); // 和MatrixSetDiagV2InferShape完全一致

IMPLEMT_COMMON_INFERFUNC(MatrixDiagV2InferShape) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  Shape diagonal_shape;
  auto diagonal_tensor_desc = op.GetInputDesc(0);
  CHECK(WithRankAtLeast(diagonal_tensor_desc, 1, diagonal_shape, opName.GetString()) != GRAPH_SUCCESS,
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(opName.GetString(), ConcatString("failed to call WithRankAtLeast function, ",
      "input[diagonal] rank must be at least 1D, but got rank[",
      op.GetInputDesc(0).GetShape().GetDimNum(), "]")),
    return GRAPH_FAILED);

  Shape k_shape;
  auto k_tensor_desc = op.GetInputDesc(1);
  CHECK(WithRankAtMost(k_tensor_desc, 1, k_shape, opName.GetString()) != GRAPH_SUCCESS,
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(opName.GetString(), ConcatString(
      "failed to call WithRankAtMost function, ",
      "input[k] rank must be at most 1D, but got rank[",
      op.GetInputDesc(1).GetShape().GetDimNum(), "]")),
    return GRAPH_FAILED);

  Shape num_rows_shape;
  auto num_rows_tensor_desc = op.GetInputDesc(kInputNumRowsIndex);
  CHECK(WithRank(num_rows_tensor_desc, 0, num_rows_shape, opName.GetString()) != GRAPH_SUCCESS,
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(opName.GetString(), ConcatString("failed to call WithRank function, ",
      "input[num_rows] rank must be 0, but got rank[",
      op.GetInputDesc(kInputNumRowsIndex).GetShape().GetDimNum(), "]")),
    return GRAPH_FAILED);

  Shape num_cols_shape;
  auto num_cols_tensor_desc = op.GetInputDesc(kInputNumColsIndex);
  CHECK(WithRank(num_cols_tensor_desc, 0, num_cols_shape, opName.GetString()) != GRAPH_SUCCESS,
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(opName.GetString(), ConcatString("failed to call WithRank function, ",
      "input[num_cols] rank must be 0, but got rank[",
      op.GetInputDesc(kInputNumColsIndex).GetShape().GetDimNum(), "]")),
    return GRAPH_FAILED);

  Shape padding_value_shape;
  auto padding_value_tensor_desc = op.GetInputDesc(kInputPaddingIndex);
  CHECK(WithRank(padding_value_tensor_desc, 0, padding_value_shape, opName.GetString()) != GRAPH_SUCCESS,
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(opName.GetString(), ConcatString("failed to call WithRank function, ",
      "input[padding_value] rank must be 0, but got rank[",
      op.GetInputDesc(kInputPaddingIndex).GetShape().GetDimNum(), "]")),
    return GRAPH_FAILED);

  auto output_desc = op.GetOutputDescByName("output");
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
    } else if (num_elements == kInputKMaxSize) {
      int32_t* data = reinterpret_cast<int32_t*>(k_tensor.GetData());
      lower_diag_index = *(data);
      upper_diag_index = *(data + 1);
    } else {
      std::string err_msg = ConcatString(
        "the input[k] must be scalar or a vector with one or two elements, ",
        "but it has [", num_elements, "] elements. ");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), err_msg);
      return GRAPH_PARAM_INVALID;
    }
  }

  CHECK(lower_diag_index > upper_diag_index,
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), ConcatString(
      "the variable [lower_diag_index] of input[k] 1th value[",
      lower_diag_index, "] must be ",
      "not greater than [upper_diag_index] of input[k] 2th value[",
      upper_diag_index, "]")),
    return GRAPH_PARAM_INVALID);

  auto diagonal_dims = diagonal_shape.GetDims();
  const int32_t diagonal_rank = diagonal_shape.GetDimNum();
  if (lower_diag_index < upper_diag_index) {
    const int64_t num_diags = diagonal_dims[diagonal_rank - 2];
    const int64_t other_dim = diagonal_dims[diagonal_rank - 1];
    CHECK(num_diags != (upper_diag_index - lower_diag_index + 1),
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), ConcatString(
        "the number of rows of [diagonal] doesn't match the number of ",
        "diagonals implied from [d_lower] and [d_upper].",
        " num_diags is [", num_diags, "], d_lower is [", lower_diag_index,
        "], d_upper is [", upper_diag_index, "], other_dim is [", other_dim, "]")),
      return GRAPH_PARAM_INVALID);
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
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), err_msg);
    return GRAPH_PARAM_INVALID;
  }

  if (num_cols == ge::UNKNOWN_DIM) {
    num_cols = min_num_cols;
  } else if (num_cols < min_num_cols) {
    std::string err_msg = ConcatString(
      "input[num_cols] value[", num_cols, "] must be not less than ",
      "min_num_cols[", min_num_cols, "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), err_msg);
    return GRAPH_PARAM_INVALID;
  }

  CHECK(num_rows != min_num_rows && num_cols != min_num_cols &&
      min_num_rows != ge::UNKNOWN_DIM && min_num_cols != ge::UNKNOWN_DIM,
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), ConcatString(
      "input[num_rows] value[", num_rows, "] and ",
      "input[num_cols] value[", num_cols, "] ",
      "are not equal with min_num_rows[", min_num_rows, "] and min_num_cols[",
      min_num_rows, "]")),
    return GRAPH_PARAM_INVALID);

  Shape output_shape;
  OP_LOGI(opName.GetString(), "num_rows: ", num_rows, " num_cols: ", num_cols);
  if (lower_diag_index == upper_diag_index) {
    CHECK(ReplaceDim(diagonal_shape, diagonal_rank - 1, num_rows, output_shape, opName.GetString()) != GRAPH_SUCCESS,
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(opName.GetString(), ConcatString(
        "failed to call ReplaceDim function, replace input[diagonal] dim, ",
        "index[", diagonal_rank - 1, "],  replace value[", num_rows, "]")),
      return GRAPH_FAILED);
    CHECK(Concatenate(output_shape, Shape({num_cols}), output_shape) != GRAPH_SUCCESS,
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), ConcatString(
        "failed to call Concatenate function, output shape",
        DebugString(output_shape.GetDims()),
        " and another shape", DebugString(Shape({num_cols}).GetDims()))),
      return GRAPH_FAILED);
  } else {
    CHECK(ReplaceDim(diagonal_shape, diagonal_rank - 2, num_rows, output_shape, opName.GetString()) != GRAPH_SUCCESS,
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(opName.GetString(), ConcatString(
        "failed to call ReplaceDim function, replace input[diagonal] dim, ",
        "index[", diagonal_rank - 2, "], replace value[", num_rows, "]")),
      return GRAPH_FAILED);
    CHECK(ReplaceDim(output_shape, diagonal_rank - 1, num_cols, output_shape, opName.GetString()) != GRAPH_SUCCESS,
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(opName.GetString(), ConcatString(
        "failed to call ReplaceDim function, replace output[output] dim, ",
        "index[", diagonal_rank - 1, "], replace value[", num_cols, "]")),
      return GRAPH_FAILED);
  }
  output_desc.SetShape(output_shape);
  output_desc.SetDataType(diagonal_tensor_desc.GetDataType());
  (void)op.UpdateOutputDesc("output", output_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixDiagV2, MatrixDiagV2InferShape);

// ----------------IndexAdd Begin-------------------
bool InferShapeAndTypeIndexAdd(Operator& op) {
  TensorDesc output_desc = op.GetOutputDescByName("var_out");
  DataType var_dtype = op.GetInputDescByName("var").GetDataType();
  Format var_format = op.GetInputDescByName("var").GetFormat();
  ge::Shape var_shape = op.GetInputDescByName("var").GetShape();
  std::vector<int64_t> var_dims = var_shape.GetDims();

  ge::Shape updates_shape = op.GetInputDescByName("updates").GetShape();
  std::vector<int64_t> updates_dims = updates_shape.GetDims();

  AscendString op_name_str;
  if (GRAPH_SUCCESS !=op.GetName(op_name_str)) {
    OP_LOGE(op_name_str.GetString(), "get op name faild!");
    return false;
  }
  const char *op_name = op_name_str.GetString();

  if (updates_dims != var_dims) {
    OP_LOGE(op_name, "var_dims not equal updates dims");
    return false;
  }

  ge::Shape output_shape = ge::Shape(var_dims);
  output_desc.SetShape(output_shape);
  output_desc.SetDataType(var_dtype);
  output_desc.SetFormat(var_format);
  op.UpdateOutputDesc("var_out", output_desc);

  return true;
}

IMPLEMT_VERIFIER(IndexAdd, IndexAddVerify) {
  DataType var_dtype = op.GetInputDescByName("var").GetDataType();
  DataType indices_dtype = op.GetInputDescByName("indices").GetDataType();
  DataType updates_dtype = op.GetInputDescByName("updates").GetDataType();
  AscendString op_name_str;
  if (GRAPH_SUCCESS !=op.GetName(op_name_str)) {
    OP_LOGE(op_name_str.GetString(), "get op name faild!");
    return false;
  }
  const char *op_name = op_name_str.GetString();
  if (var_dtype != updates_dtype) {
    OP_LOGE(op_name,
            "The input shape of var var_out updates is equal, please check!");
    return GRAPH_FAILED;
  }
  if (indices_dtype != DT_INT32) {
    OP_LOGE(op_name,
            "The input shape of indices is not int32, please check!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(IndexAddInferShape) {
  AscendString op_name_str;
  if (GRAPH_SUCCESS != op.GetName(op_name_str)) {
    OP_LOGE(op_name_str.GetString(), "get op name faild!");
    return false;
  }
  const char *op_name = op_name_str.GetString();
  if (InferShapeAndTypeIndexAdd(op) == false) {
    OP_LOGE(op_name, "index_add infer shape failed!");
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
  TensorDesc output_desc = op.GetOutputDesc(0);
  DataType x1_dtype = op.GetInputDesc(0).GetDataType();
  Format x1_format = op.GetInputDesc(0).GetFormat();
  ge::Shape x1_shape = op.GetInputDesc(0).GetShape();

  AscendString op_name_str;
  if (GRAPH_SUCCESS !=op.GetName(op_name_str)) {
    OP_LOGE(op_name_str.GetString(), "get op name faild!");
    return false;
  }

  output_desc.SetShape(x1_shape);
  output_desc.SetDataType(x1_dtype);
  output_desc.SetFormat(x1_format);
  op.UpdateOutputDesc("y", output_desc);

  return true;
}

IMPLEMT_VERIFIER(IndexPut, IndexPutVerify) {
  DataType x1_dtype = op.GetInputDesc(0).GetDataType();
  DataType x2_dtype = op.GetInputDesc(1).GetDataType();
  AscendString op_name_str;
  if (GRAPH_SUCCESS !=op.GetName(op_name_str)) {
    OP_LOGE(op_name_str.GetString(), "get op name faild!");
    return false;
  }
  const char *op_name = op_name_str.GetString();
  if (x1_dtype != x2_dtype) {
    OP_LOGE(op_name,
            "The input dtype of x1 x2 y is equal, please check!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(IndexPutInferShape) {
  AscendString op_name_str;
  if (GRAPH_SUCCESS != op.GetName(op_name_str)) {
    OP_LOGE(op_name_str.GetString(), "get op name faild!");
    return false;
  }
  const char *op_name = op_name_str.GetString();
  if (InferShapeAndTypeIndexPut(op) == false) {
    OP_LOGE(op_name, "index_put infer shape failed!");
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
bool IsEllispis(const std::string &ori_str, const std::string &target) {
  return (ori_str.find(target) != std::string::npos);
}

// remove spaces from the string
void TrimWhitespace(std::string &s) {
  size_t index = 0;
  if (!s.empty()) {
    while ((index = s.find(' ', index)) != string::npos) {
      s.erase(index, 1);
    }
  }
}

//  cut the string in half
bool SplitEquation(const std::string &eqn, std::string &in_equ, std::string &out_equ) {
  size_t pos = 0;
  if ((pos = eqn.find("->")) != std::string::npos) {
    in_equ = eqn.substr(0, pos);
    // add 2 to get start index of out_equ
    out_equ = eqn.substr(pos + 2);
    return true;
  }
  // cannot support pattern without "->", like "a,a".
  return false;
}

// gets an array of input strings
void GetInSplitEquationList(const std::string &in_equ, std::vector<std::string> &in_equ_list) {
  std::stringstream equ_stream(in_equ);
  std::string term;
  while (!equ_stream.eof()) {
    std::getline(equ_stream, term, ',');
    in_equ_list.push_back(term);
  }
}

// make map between input label and check
bool InsertLabelToMap(const char label, const int64_t shape, std::map<char, int64_t> &equ_map) {
  if (isalpha(label)) {
    if (equ_map.find(label) == equ_map.end()) {
      equ_map[label] = shape;
      OP_LOGD("Einsum", "The shape of input label [%c] is [%ld].", label, shape);
    } else {
      CHECK(equ_map[label] != shape,
            OP_LOGE("Einsum", "Input label [%c] has different shape, which is [%ld] and [%ld].", label, equ_map[label],
                    shape),
            return false);
    }
    return true;
  }
  OP_LOGE("Einsum", "Input label should be A-Za-z, which is [%c].", label);
  return false;
}

// Scenes with ellipses
bool MapWithEllipsis(const std::string &equ_temp, const std::vector<int64_t> &equ_tensor_temp,
                     std::map<char, int64_t> &equ_map, std::set<std::vector<int64_t>> &ellipses_set) {
  auto equ_temp_size = equ_temp.size();
  int64_t equ_tensor_temp_size = equ_tensor_temp.size();
  int64_t ell_size = equ_tensor_temp_size + kEinsumOffsetLength - equ_temp_size;
  std::vector<int64_t> ell_list;

  auto start_index = equ_temp.find_first_of(".");

  for (size_t i = 0; i < start_index; i++) {
    if (!InsertLabelToMap(equ_temp[i], equ_tensor_temp[i], equ_map)) {
      return false;
    }
  }

  if (ell_size > 0) {
    for (int64_t j = 0; j < ell_size; j++) {
      ell_list.push_back(equ_tensor_temp[j + start_index]);
    }
    ellipses_set.insert(ell_list);
    // each inputs ellipses should be same length
    if (ell_size != static_cast<int64_t>((*ellipses_set.begin()).size())) {
      return false;
    }
  }

  for (int64_t k = start_index; k < (equ_tensor_temp_size - ell_size); k++) {
    if (!InsertLabelToMap(equ_temp[k + kEinsumOffsetLength], equ_tensor_temp[k + ell_size], equ_map)) {
      return false;
    }
  }
  return true;
}

// Scenes without ellipses
bool MapWithoutEllipsis(const std::string &equ_temp, const vector<int64_t> &equ_tensor_temp,
                        map<char, int64_t> &equ_map) {
  for (size_t i = 0; i < equ_temp.size(); i++) {
    if (!InsertLabelToMap(equ_temp[i], equ_tensor_temp[i], equ_map)) {
      return false;
    }
  }
  return true;
}

// Output shape with ellipsis
void OutputWithEllipsis(std::vector<int64_t> &output_shape, const std::string &out_equ,
                        const std::map<char, int64_t> &equ_map, const std::vector<int64_t> ell_list) {
  size_t start_index = out_equ.find_first_of(".");
  for (size_t i = 0; i < start_index; i++) {
    output_shape.push_back(equ_map.at(out_equ[i]));
  }
  for (size_t i = 0; i < ell_list.size(); i++) {
    output_shape.push_back(ell_list[i]);
  }
  for (size_t i = (start_index + kEinsumOffsetLength); i < out_equ.size(); i++) {
    output_shape.push_back(equ_map.at(out_equ[i]));
  }
}

// Output shape without ellipsis
void OutputWithoutEllipsis(std::vector<int64_t> &output_shape, const std::string &out_equ,
                           const std::map<char, int64_t> &equ_map) {
  for (size_t i = 0; i < out_equ.size(); i++) {
    output_shape.push_back(equ_map.at(out_equ[i]));
  }
}

bool CheckEllipsisAndDuplicatedLabel(std::string part_eqn, bool &stride_flag) {
  size_t start_idx = part_eqn.find("...");
  if (start_idx != std::string::npos) {
    // erase "..." in equation like "...abc" to "abc"
    part_eqn.erase(start_idx, kEinsumOffsetLength);
    // equations can only contain "..."
    if (part_eqn.find_first_of(".") != std::string::npos) {
      return false;
    }
  }
  // after erase, check duplicated label in equation
  if (!stride_flag) {
    std::set<char> label_set(part_eqn.begin(), part_eqn.end());
    stride_flag = (label_set.size() != part_eqn.size());
  }
  return true;
}

bool CheckEquation(const Operator &op, const std::string &eqn, std::vector<std::string> &in_equ_list,
                   std::string &out_equ) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("Einsum", "GetName failed."), return false);
  const char *einsum_name = op_name.GetString();
  std::string in_equ;
  bool input_stride_flag = false;
  bool output_inflate_flag = false;
  // get input equation and output equation
  CHECK(!SplitEquation(eqn, in_equ, out_equ), OP_LOGE(einsum_name, "Split Equations failed"), return false);

  // split input equaiton list and check
  GetInSplitEquationList(in_equ, in_equ_list);

  for (auto &in_equ_tmp : in_equ_list) {
    OP_LOGD(einsum_name, "Input equation is [%s].", in_equ_tmp.c_str());
    CHECK(in_equ_tmp.empty(),
          OP_LOGE(einsum_name, "Input equation size can not be empty, which is [%zu].", in_equ_tmp.size()),
          return false);
    CHECK(!CheckEllipsisAndDuplicatedLabel(in_equ_tmp, input_stride_flag),
          OP_LOGE(einsum_name, "Input equation is wrong which is [%s].", in_equ_tmp.c_str()), return false);
  }

  // check output equations
  CHECK(out_equ.empty(), OP_LOGE(einsum_name, "Output equation can not be empty"), return false);
  CHECK(!CheckEllipsisAndDuplicatedLabel(out_equ, output_inflate_flag),
        OP_LOGE(einsum_name, "Output equation is wrong which is [%s].", out_equ.c_str()), return false);

  //  equation likes "aab, bc->aac" is not right.
  CHECK(output_inflate_flag && input_stride_flag,
        OP_LOGE(einsum_name, "Duplicated label can not appear in both input and output."), return false);

  CHECK(op.GetInputsSize() != in_equ_list.size(),
        OP_LOGE(einsum_name, "The num of equation's inputs doesn't match with the num of inputs."), return false);
  return true;
}

bool GetEllipsisRes(const std::set<std::vector<int64_t>> &ellipsis_set, std::vector<int64_t> &ellipsis_out) {
  if (ellipsis_set.empty()) {
    return true;
  }
  size_t vec_size = (*ellipsis_set.begin()).size();
  for (size_t i = 0; i < vec_size; i++) {
    int64_t shape_tmp = (*ellipsis_set.begin())[i];
    for (auto &vec_item : ellipsis_set) {
      bool need_broadcast = (vec_item[i] == 1 || shape_tmp == 1);
      if (need_broadcast) {
        shape_tmp = std::max(vec_item[i], shape_tmp);
      } else if (vec_item[i] != shape_tmp) {
        return false;
      }
    }
    ellipsis_out.push_back(shape_tmp);
    OP_LOGD("Einsum", "The output ... shape is [%s].", VectorToString(ellipsis_out).c_str());
  }
  return true;
}

bool CheckOutputNewLabel(std::string &out_equ, std::map<char, int64_t> &equ_map) {
  std::string out_equ_bak(out_equ);
  size_t ellipsis_pos = out_equ_bak.find("...");
  if (ellipsis_pos != std::string::npos) {
    out_equ_bak.erase(ellipsis_pos, kEinsumOffsetLength);
  }
  for (auto i : out_equ_bak) {
    if (equ_map.find(i) == equ_map.end()) {
      return false;
    }
  }
  return true;
}

// einsum infer shape
bool EinsumInferShape(const Operator &op, const std::string &eqn, const std::vector<std::vector<int64_t>> &tensor_list,
                      std::vector<int64_t> &output_shape) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("Einsum", "GetName failed."), return false);
  const char *einsum_name = op_name.GetString();

  // define maps to hold the corresponding characters
  std::map<char, int64_t> equ_map;
  std::vector<int64_t> ellipsis_out;
  std::set<std::vector<int64_t>> ellipsis_set;
  std::vector<std::string> in_equ_list;
  std::string out_equ;

  CHECK(!CheckEquation(op, eqn, in_equ_list, out_equ), OP_LOGE(einsum_name, "Equation is wrong."), return false);

  std::string equ_temp;
  std::vector<int64_t> equ_tensor_temp;
  std::string targets = "...";
  for (size_t i = 0; i < in_equ_list.size(); i++) {
    equ_temp = in_equ_list[i];
    auto equ_temp_size = equ_temp.size();
    equ_tensor_temp = tensor_list[i];
    auto equ_tensor_temp_size = equ_tensor_temp.size();
    if (IsEllispis(equ_temp, targets)) {
      // equation likes "...ab, bc->...ac", ((2,), (2, 3)) is not right.
      CHECK(equ_tensor_temp_size + 3 < equ_temp_size,
            OP_LOGE(einsum_name, "Input tensor size does not match equation."), return false);
      CHECK(!MapWithEllipsis(equ_temp, equ_tensor_temp, equ_map, ellipsis_set),
            OP_LOGE(einsum_name, "Input label and shape is not right."), return false);
    } else {
      CHECK(equ_tensor_temp_size != equ_temp_size, OP_LOGE(einsum_name, "Input tensor size does not match equation."),
            return false);
      CHECK(!MapWithoutEllipsis(equ_temp, equ_tensor_temp, equ_map),
            OP_LOGE(einsum_name, "Input label and shape is not right."), return false);
    }
  }

  // label in out equation must appear in input equation
  CHECK(!CheckOutputNewLabel(out_equ, equ_map),
        OP_LOGE(einsum_name, "Output equation contains new label, The equation is [%s].", out_equ.c_str()),
        return false);

  CHECK(!GetEllipsisRes(ellipsis_set, ellipsis_out), OP_LOGE(einsum_name, "The shape corresponding to ... is wrong"),
        return false);

  if (IsEllispis(out_equ, targets)) {
    OutputWithEllipsis(output_shape, out_equ, equ_map, ellipsis_out);
  } else {
    CHECK(!ellipsis_out.empty(), OP_LOGE(einsum_name, "Output equation should have Ellipsis"), return false);
    OutputWithoutEllipsis(output_shape, out_equ, equ_map);
  }
  return true;
}

void GetRangeInterSection(const std::pair<int64_t, int64_t> &new_range, std::pair<int64_t, int64_t> &result) {
  result.first = std::max(new_range.first, result.first);
  if (result.second == -1 || new_range.second == -1) {
    result.second = std::max(new_range.second, result.second);
  } else {
    result.second = std::min(new_range.second, result.second);
  }
}

void ResetRangewithEmptyInputRange(OpDescPtr &op_desc, const AscendString op_name, size_t i,
                                   vector<std::pair<int64_t, int64_t>> &range) {
  if (!range.empty()) {
    return;
  }
  static std::pair<int64_t, int64_t> default_range = {1, -1};
  auto dims = op_desc->MutableInputDesc(i)->MutableShape().GetDims();
  for (auto dim : dims) {
    if (dim == -1) {
      OP_LOGW(op_name.GetString(), "input%zu tensor has no range but contains -1, use range [1, -1]", i);
      range.push_back(default_range);
    } else {
      range.push_back({dim, dim});
    }
  }
}

void GetEquationRangeInterSection(OpDescPtr &op_desc, const AscendString op_name,
                                  const vector<std::string> &in_equ_list,
                                  std::map<char, std::pair<int64_t, int64_t>> &ranges) {
  for (size_t i = 0; i < in_equ_list.size(); ++i) {
    const std::string &equ_temp = in_equ_list[i];
    vector<std::pair<int64_t, int64_t>> range;
    op_desc->MutableInputDesc(i)->GetShapeRange(range);
    ResetRangewithEmptyInputRange(op_desc, op_name, i, range);

    if (range.size() < equ_temp.size()) {
      OP_LOGW(op_name.GetString(), "skip range, range size: %zu, equation: %s", range.size(), equ_temp.c_str());
      continue;
    }

    for (size_t j = 0; j < equ_temp.size(); ++j) {
      auto it = ranges.find(equ_temp[j]);
      if (it == ranges.end()) {
        ranges.insert({equ_temp[j], range[j]});
      } else {
        GetRangeInterSection(range[j], it->second);
      }
    }
  }
}

void AssembleShapeRange(const std::string &equation, const std::map<char, std::pair<int64_t, int64_t>> &ranges,
                        vector<std::pair<int64_t, int64_t>> &range) {
  static std::pair<int64_t, int64_t> default_range = {1, -1};
  range.reserve(equation.size());
  for (size_t j = 0; j < equation.size(); ++j) {
    auto it = ranges.find(equation[j]);
    if (it == ranges.end()) {
      range.push_back(default_range);
    } else {
      range.push_back(it->second);
    }
  }
}

bool EinsumInferShapeRange(OpDescPtr &op_desc, const AscendString op_name, std::string &eqn,
                           const vector<vector<int64_t>> &tensor_list,
                           vector<std::pair<int64_t, int64_t>> &output_range) {
  std::string targets = "...";
  if (IsEllispis(eqn, targets)) {
    OP_LOGE(op_name.GetString(), "not support equation[%s]", eqn.c_str());
    return false;
  }

  std::string in_equ;
  std::string out_equ;
  // Split string
  SplitEquation(eqn, in_equ, out_equ);
  // gets a list of input strings
  vector<std::string> in_equ_list;
  GetInSplitEquationList(in_equ, in_equ_list);

  if (in_equ_list.size() != tensor_list.size()) {
    // if equation size not equal with tensor size, tf will raise exception
    OP_LOGE(op_name.GetString(), "equation size[%zu] not equal with tensor size[%zu]", in_equ_list.size(),
            tensor_list.size());
    return false;
  }

  std::map<char, std::pair<int64_t, int64_t>> ranges;
  GetEquationRangeInterSection(op_desc, op_name, in_equ_list, ranges);
  for (size_t i = 0; i < in_equ_list.size(); ++i) {
    if (!op_desc->MutableInputDesc(i)->MutableShape().IsUnknownShape()) {
      continue;
    }

    vector<std::pair<int64_t, int64_t>> range;
    AssembleShapeRange(in_equ_list[i], ranges, range);
    op_desc->MutableInputDesc(i)->SetShapeRange(range);
  }

  AssembleShapeRange(out_equ, ranges, output_range);
  return true;
}

IMPLEMT_COMMON_INFERFUNC(EinsumInferShape) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, CUBE_INNER_ERR_REPORT("Einsum", "GetName failed."), return GRAPH_FAILED);
  const char *einsum_name = op_name.GetString();
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto x0_type = op_desc->MutableInputDesc(0)->GetDataType();
  // get attr equation
  std::string equation;
  if (op.GetAttr("equation", equation) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(einsum_name, "GetOpAttr equation failed");
    return GRAPH_FAILED;
  }

  TrimWhitespace(equation);

  CHECK(op.GetInputsSize() > 2, CUBE_INNER_ERR_REPORT(einsum_name, "Input tensors should not exceed 2."),
        return GRAPH_FAILED);

  std::vector<std::vector<int64_t>> tensor_list;
  bool is_unkown_shape = false;
  for (size_t i = 0; i < op.GetInputsSize(); i++) {
    is_unkown_shape = is_unkown_shape || op_desc->MutableInputDesc(i)->MutableShape().IsUnknownShape();
    tensor_list.push_back(std::move(op_desc->MutableInputDesc(i)->MutableShape().GetDims()));
  }

  std::vector<int64_t> output_shape;
  auto output_desc = op_desc->MutableOutputDesc(0);
  if (!is_unkown_shape) {
    CHECK(!EinsumInferShape(op, equation, tensor_list, output_shape),
          CUBE_INNER_ERR_REPORT(einsum_name, "Infershape func failed."), return GRAPH_FAILED);
  } else {
    std::vector<std::pair<int64_t, int64_t>> output_range;
    if (!EinsumInferShapeRange(op_desc, op_name, equation, tensor_list, output_range)) {
      CUBE_INNER_ERR_REPORT(einsum_name, "Infershape func failed.");
      return GRAPH_FAILED;
    }

    output_shape.assign(output_range.size(), -1);
    for (size_t i = 0; i < output_range.size(); i++) {
      if (output_range[i].first == output_range[i].second) {
        output_shape[i] = output_range[i].first;
      }
    }
    output_desc->SetShapeRange(output_range);
  }
  // updata output shape and dtype
  OP_LOGD(einsum_name, "The output shape is [%s].", VectorToString(output_shape).c_str());
  output_desc->SetShape(ge::GeShape(output_shape));
  output_desc->SetDataType(x0_type);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Einsum, EinsumVerify) { return GRAPH_SUCCESS; }
COMMON_INFER_FUNC_REG(Einsum, EinsumInferShape);
VERIFY_FUNC_REG(Einsum, EinsumVerify);
// ----------------Einsum-------------------

// ---------------Eye----------------------------
static bool CheckRows(const Operator &op, const string &attr_num_rows) {
  int64_t num_rows;
  op.GetAttr(attr_num_rows.c_str(), num_rows);
  if (num_rows <= 0) {
    return false;
  }
  return true;
}

static bool CheckBatchShape(const Operator &op, const string &attr_batch_shape) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return false);
  std::vector<int64_t> batch_shape;
  op.GetAttr(attr_batch_shape.c_str(), batch_shape);
  for (size_t i = 0; i < batch_shape.size(); ++i) {
    if (batch_shape[i] <= 0) {
      OP_LOGE(opName.GetString(), "the value of batch_shape less than 0.\n");
      return false;
    }
  }
  return true;
}

IMPLEMT_COMMON_INFERFUNC(EyeInferShape)
{
    TensorDesc td = op.GetOutputDescByName("y");
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
    for (size_t i = 0; i < batch_shape.size(); ++i) {
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
// --------------Eye END-------------------------------

// ----------------FillDiagonal-------------------
IMPLEMT_COMMON_INFERFUNC(FillDiagonalInferShape) {
  Shape x_shape = op.GetInputDescByName("x").GetShape();
  DataType x_dtype = op.GetInputDescByName("x").GetDataType();
  TensorDesc td = op.GetOutputDescByName("y");
  td.SetShape(ge::Shape(x_shape));
  td.SetDataType(x_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(FillDiagonal, FillDiagonalInferShape);
// ----------------FillDiagonal END-------------------

// -----------------------Trace-----------------------
static bool InferShapeAndTypeTrace(Operator& op, const std::string& inputName, const std::string outputName) {
  TensorDesc vOutputDesc = op.GetOutputDescByName(outputName.c_str());
  DataType inputDtype = op.GetInputDescByName(inputName.c_str()).GetDataType();
  ge::Shape inputShape = op.GetInputDescByName(inputName.c_str()).GetShape();
  std::vector<int64_t> inputDims = inputShape.GetDims();
  constexpr int64_t shapeDims = 2;
  if (inputDims.size() != shapeDims) {
    OP_LOGE(outputName.c_str(), "the input shape must is 2-D matrix.\n");
    return false;
  }

  if (inputDtype != DT_FLOAT16 && inputDtype != DT_FLOAT) {
    OP_LOGE(outputName.c_str(), "the input dtype must is float16 or float.\n");
    return false;
  }

  // set output tensor dim
  std::vector<int64_t> dimVec(1, 1);
  ge::Shape outputShape = ge::Shape(dimVec);
  vOutputDesc.SetShape(outputShape);
  vOutputDesc.SetDataType(inputDtype);
  op.UpdateOutputDesc(outputName.c_str(), vOutputDesc);
  return true;
}

IMPLEMT_VERIFIER(Trace, TraceVerify) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  ge::Shape shapeX = op.GetInputDescByName("x").GetShape();
  if (shapeX.GetDimNum() != 2) {
    OP_LOGE(op_name.GetString(), "the input shape must is 2-D matrix.\n");
    return GRAPH_FAILED;
  }
  DataType dtypeX = op.GetInputDescByName("x").GetDataType();
  if (dtypeX != DT_FLOAT16 && dtypeX != DT_FLOAT) {
    OP_LOGE(op_name.GetString(), "the input dtype must is float16 or float.\n");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(TraceInferShape) {
  if (InferShapeAndTypeTrace(op, "x", "y")) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(Trace, TraceInferShape);
VERIFY_FUNC_REG(Trace, TraceVerify);
// ---------------------Trace END----------------------

// ----------------Pinverse Begin------------------------
IMPLEMT_COMMON_INFERFUNC(PinverseInferShape) {
  Shape input_shape = op.GetInputDesc(0).GetShape();
  DataType input_dtype = op.GetInputDesc(0).GetDataType();
  TensorDesc td = op.GetOutputDesc(0);
  td.SetShape(ge::Shape(input_shape));
  auto size = input_shape.GetDimNum();
  int64_t dim_num1 = input_shape.GetDim(size - 2);
  int64_t dim_num2 = input_shape.GetDim(size - 1);
  std::vector<int64_t> dim_vector;
  dim_vector.push_back(dim_num2);
  dim_vector.push_back(dim_num1);
  Shape output_shape(dim_vector);
  td.SetDataType(input_dtype);
  td.SetShape(output_shape);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Pinverse, PinverseVerify)
{
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Pinverse, PinverseInferShape);
VERIFY_FUNC_REG(Pinverse, PinverseVerify);
// ----------------Pinverse END---------------------------------

}  // namespace ge
