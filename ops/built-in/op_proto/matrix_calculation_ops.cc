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
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "util/util.h"
#include "util/common_shape_fns.h"
#include "util/array_ops_shape_fns.h"
#include "util/error_util.h"
#include "graph/utils/node_utils.h"

namespace ge {
// ----------------FullyConnection-------------------
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
    OP_LOGE(op.GetName().c_str(), "Attr axis is wrong, the original value of axis %d is not supported.", axis);
    string realvalue = ConcatString(axis_new);
    OpsAttrValueErrReport(op.GetName().c_str(), "axis", "1 or 2", realvalue);
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
    string realvalue = ConcatString(wShape.size());
    OpsAttrValueErrReport(op.GetName().c_str(), "wShape size", "2", realvalue);
    return GRAPH_FAILED;
  }

  // check km and kn shape
  if (wShape.size() == 2) {
    if (!transpose) {
      if (kShape != wShape[1]) {
        OP_LOGE(op.GetName().c_str(), "weight K must equal to input K!\n");
        OpsTwoInputShapeErrReport(op.GetName(), "X Shape", "W Shape", "weight K must equal to input K!");
        return GRAPH_FAILED;
      }
    } else {
      if (kShape != wShape[0]) {
        OpsTwoInputShapeErrReport(op.GetName(), "X Shape", "W Shape", "weight K must equal to input K!");
        OP_LOGE(op.GetName().c_str(), "weight K must equal to input K!\n");
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
    OpsAttrValueErrReport(op.GetName(), "axis", "1 or 2", ConcatString(axis_new));
    OP_LOGE(op.GetName().c_str(), "Attr axis is wrong, this axis is not supported!\n");
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
      OP_LOGE(op.GetName().c_str(), "Not enough info about M and K!\n");
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

INFER_FUNC_REG(FullyConnection, FullyConnectionInfer);

VERIFY_FUNC_REG(FullyConnection, FullyConnectionVerify);

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

INFER_FUNC_REG(FullyConnectionCompress, FullyConnectionCompressInfer);

VERIFY_FUNC_REG(FullyConnectionCompress, FullyConnectionCompressVerify);

// ----------------Matmul-------------------
string GetMatMulInfo(const Operator &op) {
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
  op.GetAttr("transpose_x1", trans_a);
  op.GetAttr("transpose_x2", trans_b);

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
static const std::pair<int64_t, int64_t> NORMALIZE_FULL_RANGE = {1, std::numeric_limits<int64_t>::max()};
static const int64_t VALUE_UNKNOWN_RANK = -2;
bool IsDimValid(int64_t dim) {
  return dim >= VALUE_UNKNOWN_RANK && dim != 0;
}

void NormalizeRange(const std::string &op_name, const int64_t dim,
                    const std::pair<int64_t, int64_t> &shape_range,
                    std::pair<int64_t, int64_t> &range) {
  if (dim == UNKNOWN_DIM && (shape_range == EMPTY_RANGE || shape_range == FULL_RANGE)) {
    range = NORMALIZE_FULL_RANGE;
    if (shape_range == EMPTY_RANGE) {
      OP_LOGW(
          op_name.c_str(),
          "[InferShape] the dimension is -1 and no range is provided, therefore, the range is assumed to be [1, %lld]",
          NORMALIZE_FULL_RANGE.second);
    }
  } else if (dim > 0) {
    range = {dim, dim};
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
      OP_LOGE(op_name.c_str(), "[InferShape] dimensions a(%lld) and b(%lld) must be same", dim_a, dim_b);
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
      OP_LOGE(op_name.c_str(), "[InferShape] range a(%lld, %lld) and b(%lld, %lld) must have intersections",
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
    OP_LOGE(op_name.c_str(), "[InferShape] dimension(%lld) must be in range(%lld, %lld)", dim_b, range_a.first,
            range_b.second);
    return false;
  }
  if (range_b.first <= dim_a && dim_a <= range_b.second) {
    dim = dim_a;
    range = range_a;
    return true;
  }
  OP_LOGE(op_name.c_str(), "[InferShape] dimension(%lld) must be in range(%lld, %lld)", dim_a, range_b.first,
          range_b.second);
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
      OP_LOGE(op_name.c_str(), "[InferShape] dimensions a(%lld) and b(%lld) must be equal", dim_a, dim_b);
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
    OP_LOGE(op_name.c_str(), "[InferShape] dimension(%lld) must be in range(%lld, %lld)", dim_a, range_b.first,
            range_b.second);
    return false;
  }
  if (dim_b > 1) {
    if (range_a.first <= dim_b && dim_b <= range_a.second) {
      dim = dim_b;
      range = range_b;
      return true;
    }
    OP_LOGE(op_name.c_str(), "[InferShape] dimension(%lld) must be in range(%lld, %lld)", dim_b, range_a.first,
            range_a.second);
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
      OP_LOGE(op_name.c_str(), "[InferShape] range a(%lld, %lld) and b(%lld, %lld) must have intersections",
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
                   const vector<int64_t> &shape_bias, const vector<std::pair<int64_t, int64_t>> &range_a,
                   const vector<std::pair<int64_t, int64_t>> &range_b,
                   const vector<std::pair<int64_t, int64_t>> &range_bias, bool trans_a, bool trans_b,
                   vector<int64_t> &shape_out, vector<std::pair<int64_t, int64_t>> &range_out, bool has_batch);

 private:
  void NormalizeShapeAndRange();
  bool InferMKN();
  bool InferBatch();
  void SimplifyShapeAndRange();

  const string& op_name;
  const vector<int64_t> &shape_a;
  const vector<int64_t> &shape_b;
  const vector<int64_t> &shape_bias;
  const vector<std::pair<int64_t, int64_t>> &range_a;
  const vector<std::pair<int64_t, int64_t>> &range_b;
  const vector<std::pair<int64_t, int64_t>> &range_bias;
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

InferShapeMatMul::InferShapeMatMul(const string &op_name, const vector<int64_t> &shape_a,
                                   const vector<int64_t> &shape_b, const vector<int64_t> &shape_bias,
                                   const vector<std::pair<int64_t, int64_t>> &range_a,
                                   const vector<std::pair<int64_t, int64_t>> &range_b,
                                   const vector<std::pair<int64_t, int64_t>> &range_bias, bool trans_a, bool trans_b,
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
  int64_t base_len = has_batch ? 3 : 2;
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
}

void InferShapeMatMul::NormalizeShapeAndRange() {
  int64_t base_len = has_batch ? 3 : 2;
  if (shape_a == UNKNOWN_RANK) {
    for (int i = num_dim - base_len; i < num_dim; ++i) {
      infer_shape_a[i] = UNKNOWN_DIM;
      infer_range_a[i] = NORMALIZE_FULL_RANGE;
    }
  } else {
      copy(shape_a.begin(), shape_a.end(), infer_shape_a.begin() + num_dim - shape_a.size());
      copy(range_a.begin(), range_a.end(), infer_range_a.begin() + num_dim - range_a.size());
  }

  if (shape_b == UNKNOWN_RANK) {
    for (int i = num_dim - base_len; i < num_dim; ++i) {
      infer_shape_b[i] = UNKNOWN_DIM;
      infer_range_b[i] = NORMALIZE_FULL_RANGE;
    }
  } else {
      copy(shape_b.begin(), shape_b.end(), infer_shape_b.begin() + num_dim - shape_b.size());
      copy(range_b.begin(), range_b.end(), infer_range_b.begin() + num_dim - range_b.size());
  }

  if (!shape_bias.empty()) {
    if (shape_bias == UNKNOWN_RANK) {
      infer_shape_bias[num_dim - 1] = UNKNOWN_DIM;
      infer_range_bias[num_dim - 1] = NORMALIZE_FULL_RANGE;
    } else {
      copy(shape_bias.begin(), shape_bias.end(), infer_shape_bias.begin() + num_dim - shape_bias.size());
      copy(range_bias.begin(), range_bias.end(), infer_range_bias.begin() + num_dim - range_bias.size());
    }
  }

  for (auto i = num_dim - shape_a.size(); i < num_dim; ++i) {
    NormalizeRange(op_name, infer_shape_a[i], infer_range_a[i], infer_range_a[i]);
  }

  for (auto i = num_dim - shape_b.size(); i < num_dim; ++i) {
    NormalizeRange(op_name, infer_shape_b[i], infer_range_b[i], infer_range_b[i]);
  }

  if (!shape_bias.empty()) {
    for (auto i = num_dim - shape_bias.size(); i < num_dim; ++i) {
      NormalizeRange(op_name, infer_shape_bias[i], infer_range_bias[i], infer_range_bias[i]);
    }
  }
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
  if (!IsDimValid(m) || !IsDimValid(k_a) || !IsDimValid(k_b) || !IsDimValid(n_b)) {
    OP_LOGE(op_name.c_str(), "[InferShape] dimension must be -2, -1 or greater than 0");
    return false;
  }

  std::pair<int64_t, int64_t> range_k, range_n = infer_range_b[idx_n_b];
  if (k_a > 0 && k_b > 0 && k_a != k_b) {
    OP_LOGE(op_name.c_str(), "[InferShape] The k-axis of a(%lld) and b(%lld) tensors must be the same", k_a, k_b);
    return false;
  } else if (k_a < 0 && k_b < 0) {
    if (!IntersectDimensionAndRange(op_name, k_a, k_b, infer_range_a[idx_k_a], infer_range_b[idx_k_b], k, range_k)) {
      OP_LOGE(op_name.c_str(), "[InferShape] The intersection of the k-axis of tensor a and b is invalid");
      return false;
    }
  }
  if (!shape_bias.empty()) {
    int64_t idx_n_bias = num_dim - 1;
    int64_t n_bias = infer_shape_bias[idx_n_bias];
    if (!IsDimValid(n_bias)) {
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
  range_out[num_dim - 2] = infer_range_a[idx_m] == NORMALIZE_FULL_RANGE ? FULL_RANGE : infer_range_a[idx_m];
  range_out[num_dim - 1] = range_n == NORMALIZE_FULL_RANGE ? FULL_RANGE : range_n;

  return true;
}

bool InferShapeMatMul::InferBatch() {
  for (auto i = 0; i < num_dim - 2; ++i) {
    if (!BroadcastDimensionAndRange(op_name, infer_shape_a[i], infer_shape_b[i], infer_range_a[i], infer_range_b[i],
                                    shape_out[i], range_out[i])) {
      OP_LOGE(op_name.c_str(),
              "[InferShape] The broadcst operation for tensor a and b on the n-th dimension is failed");
      return false;
    }

    if (!shape_bias.empty()) {
      if (!BroadcastDimensionAndRange(op_name, shape_out[i], infer_shape_bias[i], range_out[i], infer_range_bias[i],
                                      shape_out[i], range_out[i])) {
        OP_LOGE(op_name.c_str(),
                "[InferShape] The broadcst operation for tensor out and bias on the n-th dimension is failed");
        return false;
      }
    }

    if (range_out[i] == NORMALIZE_FULL_RANGE) {
      range_out[i] = FULL_RANGE;
    }
  }
  return true;
}

void InferShapeMatMul::SimplifyShapeAndRange() {
  for (int i = 0; i < range_out.size(); i++) {
    if (range_out[i].first == range_out[i].second) {
      shape_out[i] = range_out[i].first;
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

  NormalizeShapeAndRange();

  if (!InferMKN()) {
    return false;
  }

  if (!InferBatch()) {
    return false;
  }

  SimplifyShapeAndRange();
  return true;
}

graphStatus GetMatMulOutputShape(const Operator &op,
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
  bool trans_b = false;
  if (ge::GRAPH_SUCCESS != op.GetAttr(name_attr + "_x2", trans_b)) {
    OpsGetAttrErrReport(op.GetName(), name_attr + "_x2");
    OP_LOGE(op.GetName().c_str(), "[Plugin][ERROR]%s GetOpAttr %s_x2 failed!",
            op.GetName().c_str(), name_attr.c_str());
    return GRAPH_FAILED;
  }

  auto obj = InferShapeMatMul(op.GetName(), shape_a, shape_b, shape_bias, shape_range_a, shape_range_b,
                              shape_range_bias, trans_a, trans_b, shape_out, shape_range_out, has_batch);
  if (!obj.GetShapeRangeOfOutput()) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
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
  if (dtype == DT_FLOAT) {
    OP_LOGW(op.GetName().c_str(), "[Plugin][WARNING]MatMul fp32 op has poor performance!");
  }

  OP_LOGD(op.GetName().c_str(), "%s", GetMatMulInfo(op).c_str());

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

// Registered inferfunction
COMMON_INFER_FUNC_REG(MatMul, MatMulInferShape);

// Registered verify function
VERIFY_FUNC_REG(MatMul, MatMulVerify);
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
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto tensordesc_output = op_desc->MutableOutputDesc("y");
  auto tensordesc_x1 = op_desc->MutableInputDesc("x1");
  auto tensordesc_x2 = op_desc->MutableInputDesc("x2");

  auto shape_x1 = tensordesc_x1->GetShape();
  auto shape_x2 = tensordesc_x2->GetShape();
  auto dtype = tensordesc_x1->GetDataType();
  if (dtype == DT_FLOAT) {
    OP_LOGW(op.GetName().c_str(), "[Plugin][WARNING]MatMul fp32 op has poor performance!");
  }

  if (shape_x1.GetDims() != UNKNOWN_RANK && shape_x1.GetDims().size() != 2 && shape_x1.GetDims().size() != 4) {
    OP_LOGE(op.GetName().c_str(), "[Plugin][ERROR]Matmul the first input dims is not 2 or 4!");
    return GRAPH_FAILED;
  }

  OP_LOGD(op.GetName().c_str(), "%s", GetMatMulInfo(op).c_str());

  std::vector<int64_t> shape_out;
  std::vector<std::pair<int64_t, int64_t>> shape_range_out;
  if (GRAPH_SUCCESS != GetMatMulOutputShape(op, shape_out, shape_range_out, "transpose", false)) {
    return GRAPH_FAILED;
  }

  ge::GeShape shape_out_desc{shape_out};
  auto input_format = FORMAT_ND;
  auto input_format_1 = FORMAT_ND;
  tensordesc_x1->SetFormat(input_format_1);
  tensordesc_x1->SetOriginFormat(input_format_1);
  tensordesc_x2->SetFormat(input_format);
  tensordesc_x2->SetOriginFormat(input_format);
  tensordesc_output->SetShape(shape_out_desc);
  tensordesc_output->SetOriginShape(shape_out_desc);
  tensordesc_output->SetShapeRange(shape_range_out);
  tensordesc_output->SetFormat(input_format_1);
  tensordesc_output->SetOriginFormat(input_format_1);
  if (tensordesc_x1->GetDataType() == ge::DT_INT8) {
    tensordesc_output->SetDataType(ge::DT_INT32);
  } else {
    tensordesc_output->SetDataType(tensordesc_x1->GetDataType());
  }
  return GRAPH_SUCCESS;
}
IMPLEMT_INFERFORMAT_FUNC(MatMulV2, MatMulV2InferFormat) {
  OP_LOGD(op.GetName().c_str(), "[MatMulV2 Inferformat] Finaly input format is %d", FORMAT_ND);

  ge::TensorDesc tensordesc_input = op.GetInputDesc("x1");
  tensordesc_input.SetOriginFormat(FORMAT_ND);
  tensordesc_input.SetFormat(FORMAT_ND);
  (void)op.UpdateInputDesc("x1", tensordesc_input);

  ge::TensorDesc tensordesc_input_2 = op.GetInputDesc("x2");
  tensordesc_input_2.SetOriginFormat(FORMAT_ND);
  tensordesc_input_2.SetFormat(FORMAT_ND);
  (void)op.UpdateInputDesc("x2", tensordesc_input_2);

  return GRAPH_SUCCESS;
}

INFER_FORMAT_FUNC_REG(MatMulV2, MatMulV2InferFormat);
// Registered inferfunction
COMMON_INFER_FUNC_REG(MatMulV2, MatMulV2InferShape);

// Registered verify function
VERIFY_FUNC_REG(MatMulV2, MatMulV2Verify);
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

graphStatus CommonBatchMatMulInferShape(const Operator &op) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto tensordesc_out = op_desc->MutableOutputDesc("y");
  auto tensordesc_x1 = op_desc->GetInputDesc("x1");
  auto tensordesc_x2 = op_desc->GetInputDesc("x2");

  auto shape_x1 = tensordesc_x1.GetShape();
  auto shape_x2 = tensordesc_x2.GetShape();

  size_t dim_num_x1 = shape_x1.GetDimNum();
  size_t dim_num_x2 = shape_x2.GetDimNum();

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
  bool all_unknown_rank = shape_x1.GetDims() == UNKNOWN_RANK && shape_x1.GetDims() == UNKNOWN_RANK;
  if (!all_unknown_rank && (dim_num < 3 || dim_num > 8)) {
    OP_LOGE(op.GetName().c_str(), "[Infershape]The shape can only be in the range of 3 to 8.");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> shape_out;
  std::vector<std::pair<int64_t, int64_t>> shape_range_out;
  if (GRAPH_SUCCESS != GetMatMulOutputShape(op, shape_out, shape_range_out, "adj", true)) {
    return GRAPH_FAILED;
  }

  tensordesc_out->SetShape(ge::GeShape(shape_out));
  tensordesc_out->SetShapeRange(shape_range_out);
  tensordesc_out->SetFormat(tensordesc_x1.GetFormat());
  tensordesc_out->SetDataType(tensordesc_x1.GetDataType());
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(BatchMatMulInferShape) {
  return CommonBatchMatMulInferShape(op);
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(BatchMatMul, BatchMatMulInferShape);

// Registered verify function
VERIFY_FUNC_REG(BatchMatMul, BatchMatMulVerify);


// ----------------BatchMatMulV2-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(BatchMatMulV2, BatchMatMulV2Verify) {
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(BatchMatMulV2InferShape) {
  return CommonBatchMatMulInferShape(op);
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(BatchMatMulV2, BatchMatMulV2InferShape);

// Registered verify function
VERIFY_FUNC_REG(BatchMatMulV2, BatchMatMulV2Verify);

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
      OpsInputShapeBroadcastErrReport(op.GetName(), "x", "assist", ConcatString(dims_x[i]),
                                      ConcatString(dims_assist[i]));
      OP_LOGE(op.GetName().c_str(),
              "The %s op dimensions does not "
              "match the broadcast rule(%lu %lu).",
              op.GetName().c_str(), dims_x[i], dims_assist[i]);
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
    OpsTwoInputDtypeErrReport(op.GetName(), "input_diagonal", "input_help", ConcatString(input_diagonal_dtype),
                              ConcatString(input_help_dtype));
    OP_LOGE(op.GetName().c_str(),
            "the matrix_diag op inputs "
            "should have the same dtype!\n");
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
    OpsTwoInputDtypeErrReport(op.GetName(), "input_diagonal", "input_help", ConcatString(input_diagonal_dtype),
                              ConcatString(input_help_dtype));
    OP_LOGE(op.GetName().c_str(),
            "the matrix_diag_part op inputs "
            "should have the same dtype!\n");
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
    OpsTwoInputDtypeErrReport(op.GetName(), "input_matrix", "input_help", ConcatString(input_matrix_dtype),
                              ConcatString(input_help_dtype));
    OP_LOGE(op.GetName().c_str(),
            "the matrix_set_part op inputs "
            "should have the same dtype!\n");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixSetDiagD, MatrixSetDiagDInferShape);
VERIFY_FUNC_REG(MatrixSetDiagD, MatrixSetDiagDVerify);
// ----------------MatrixSetDiag ENDD----------------

// -----------------ScatterNdUpdate-----------------
IMPLEMT_INFERFUNC(ScatterNdUpdate, ScatterNdUpdateInferShape) {
  Shape var_shape = op.GetInputDesc("var").GetShape();
  DataType input_dtype = op.GetInputDesc("var").GetDataType();
  TensorDesc td = op.GetOutputDesc("var");
  td.SetShape(ge::Shape(var_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("var", td);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ScatterNdUpdate, ScatterNdUpdateInferShape);
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

IMPLEMT_COMMON_INFERFUNC(ScatterDivInferShape) {
  Shape var_shape = op.GetInputDesc("var").GetShape();
  DataType input_dtype = op.GetInputDesc("var").GetDataType();
  TensorDesc td = op.GetOutputDesc("var");
  td.SetShape(ge::Shape(var_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("var", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(ScatterDiv, ScatterDivVerify) {
  DataType var_dtype = op.GetInputDesc(0).GetDataType();
  DataType updates_dtype = op.GetInputDesc(2).GetDataType();
  if (var_dtype != updates_dtype) {
    OpsTwoInputDtypeErrReport(op.GetName(), "var", "updates", ConcatString(var_dtype), ConcatString(updates_dtype));
    OP_LOGE(op.GetName().c_str(),
            "the scatter_div op inputs "
            "should have the same dtype!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ScatterDiv, ScatterDivInferShape);
VERIFY_FUNC_REG(ScatterDiv, ScatterDivVerify);

// ----------------ScatterNdAdd------------
IMPLEMT_VERIFIER(ScatterNdAdd, ScatterNdAddVerify) {
  if (!CheckTwoInputDtypeSame(op, "var", "updates")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ScatterNdAddInferShape) {
  Shape var_shape = op.GetInputDesc("var").GetShape();
  DataType input_dtype = op.GetInputDesc("var").GetDataType();
  TensorDesc td = op.GetOutputDesc("var");
  td.SetShape(ge::Shape(var_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("var", td);
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
  Shape var_shape = op.GetInputDesc("var").GetShape();
  DataType input_dtype = op.GetInputDesc("var").GetDataType();
  TensorDesc td = op.GetOutputDesc("var");
  td.SetShape(ge::Shape(var_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("var", td);
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
    OpsGetAttrErrReport(op.GetName(), "num_classes");
    OP_LOGE(op.GetName().c_str(), "Op get attr num_classes failed");
    return GRAPH_FAILED;
  }
  if (op.GetAttr("dtype", output_dtype_str) != GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "dtype");
    OP_LOGE(op.GetName().c_str(), "Op get attr dtype failed");
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
    OpsInputDtypeErrReport(op.GetName(), "dtype", expected_data_type_list, output_dtype_str);
    OP_LOGE(" don't supports this dtype.");
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

IMPLEMT_COMMON_INFERFUNC(ScatterMulInferShape) {
  Shape var_shape = op.GetInputDesc("var").GetShape();
  DataType input_dtype = op.GetInputDesc("var").GetDataType();
  TensorDesc td = op.GetOutputDesc("var");
  td.SetShape(ge::Shape(var_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("var", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(ScatterMul, ScatterMulVerify) {
  DataType var_dtype = op.GetInputDesc(0).GetDataType();
  DataType updates_dtype = op.GetInputDesc(2).GetDataType();
  if (var_dtype != updates_dtype) {
    OpsTwoInputDtypeErrReport(op.GetName(), "var", "updates", ConcatString(var_dtype), ConcatString(updates_dtype));
    OP_LOGE(op.GetName().c_str(),
            "the scatter_mul op inputs "
            "should have the same dtype!\n");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ScatterMul, ScatterMulInferShape);
VERIFY_FUNC_REG(ScatterMul, ScatterMulVerify);

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

IMPLEMT_COMMON_INFERFUNC(ScatterMinInferShape) {
  Shape var_shape = op.GetInputDesc("var").GetShape();
  DataType input_dtype = op.GetInputDesc("var").GetDataType();
  TensorDesc td = op.GetOutputDesc("var");
  td.SetShape(ge::Shape(var_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("var", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(ScatterMin, ScatterMinVerify) {
  DataType var_dtype = op.GetInputDesc(0).GetDataType();
  DataType updates_dtype = op.GetInputDesc(2).GetDataType();
  if (var_dtype != updates_dtype) {
    OpsTwoInputDtypeErrReport(op.GetName(), "var", "updates", ConcatString(var_dtype), ConcatString(updates_dtype));
    OP_LOGE(op.GetName().c_str(),
            "the scatter_min op inputs "
            "should have the same dtype!\n");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ScatterMin, ScatterMinInferShape);
VERIFY_FUNC_REG(ScatterMin, ScatterMinVerify);

IMPLEMT_COMMON_INFERFUNC(ScatterMaxInferShape) {
  Shape var_shape = op.GetInputDesc("var").GetShape();
  DataType input_dtype = op.GetInputDesc("var").GetDataType();
  TensorDesc td = op.GetOutputDesc("var");
  td.SetShape(ge::Shape(var_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("var", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(ScatterMax, ScatterMaxVerify) {
  DataType var_dtype = op.GetInputDesc(0).GetDataType();
  DataType updates_dtype = op.GetInputDesc(2).GetDataType();
  if (var_dtype != updates_dtype) {
    OpsTwoInputDtypeErrReport(op.GetName(), "var", "updates", ConcatString(var_dtype), ConcatString(updates_dtype));
    OP_LOGE(op.GetName().c_str(),
            "the scatter_max op inputs "
            "should have the same dtype!\n");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ScatterMax, ScatterMaxInferShape);
VERIFY_FUNC_REG(ScatterMax, ScatterMaxVerify);

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
    OP_LOGE(op.GetName().c_str(), "input input must be at least 2-D, real rank is %lld.",
            input_tensor_desc.GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  Shape k_shape;
  auto k_tensor_desc = op.GetInputDesc(1);
  if (WithRankAtMost(k_tensor_desc, 1, k_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input k must be at most 1-D, real rank is %lld.",
            k_tensor_desc.GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  Shape padding_value_shape;
  auto padding_value_tensor_desc = op.GetInputDesc(2);
  if (WithRank(padding_value_tensor_desc, 0, padding_value_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Input padding_value rank must be 0, real rank is %lld.",
            padding_value_tensor_desc.GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  TensorDesc output_desc = op.GetOutputDesc("diagonal");

  Tensor k_tensor;
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
      OP_LOGE(op.GetName().c_str(), "diag_index must be a vector with one or two elements. It has %lld elements.",
              num_elements);
      return GRAPH_PARAM_INVALID;
    }
  }

  if (lower_diag_index > upper_diag_index) {
    OP_LOGE(op.GetName().c_str(), "lower_diag_index is greater than upper_diag_index");
    return GRAPH_PARAM_INVALID;
  }

  auto input_dims = input_shape.GetDims();
  const int32_t input_rank = input_shape.GetDimNum();
  const int32_t num_rows = input_dims[input_rank - 2];
  const int32_t num_cols = input_dims[input_rank - 1];
  int64_t max_diag_len = ge::UNKNOWN_DIM;
  if (num_rows != ge::UNKNOWN_DIM && num_cols != ge::UNKNOWN_DIM) {
    if (lower_diag_index != 0 && (-num_rows >= lower_diag_index || lower_diag_index >= num_cols)) {
      OP_LOGE(op.GetName().c_str(), "lower_diag_index %lld is out of bound, num_rows: %lld, num_cols: %lld.",
              lower_diag_index, num_rows, num_cols);
      return GRAPH_PARAM_INVALID;
    }
    if (upper_diag_index != 0 && (-num_rows >= upper_diag_index || upper_diag_index >= num_cols)) {
      OP_LOGE(op.GetName().c_str(), "upper_diag_index %lld is out of bound, num_rows: %lld, num_cols: %lld.",
              upper_diag_index, num_rows, num_cols);
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
    OP_LOGE(op.GetName().c_str(), "input input must be at least 2-D, real rank is %lld.",
            input_tensor_desc.GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  Shape diagonal_shape;
  auto diagonal_tensor_desc = op.GetInputDesc(1);
  if (WithRankAtLeast(diagonal_tensor_desc, 1, diagonal_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input diagonal must be at least 1-D, real rank is %lld.",
            diagonal_tensor_desc.GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  Shape k_shape;
  auto k_tensor_desc = op.GetInputDesc(2);
  if (WithRankAtMost(k_tensor_desc, 1, k_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input k rank must be at most 1-D, real rank is %lld.",
            k_tensor_desc.GetShape().GetDimNum());
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
        OP_LOGE(op.GetName().c_str(), "diag_index must be a vector with one or two elements. It has %lld elements",
                num_elements);
        return GRAPH_PARAM_INVALID;
      }
    }

    if (lower_diag_index > upper_diag_index) {
      OP_LOGE(op.GetName().c_str(), "lower_diag_index is greater than upper_diag_index");
      return GRAPH_PARAM_INVALID;
    }
  }

  if (RankKnown(input_shape)) {
    auto input_rank = input_shape.GetDimNum();
    if (k_index_known) {
      if (WithRank(diagonal_tensor_desc, (lower_diag_index == upper_diag_index ? input_rank - 1 : input_rank),
                   diagonal_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input diagonal must be fit with input matrix rank %lld, real rank is %lld.",
                input_rank, diagonal_shape.GetDimNum());
        return GRAPH_FAILED;
      } else {
        if (WithRankAtLeast(diagonal_tensor_desc, input_rank - 1, diagonal_shape, op.GetName().c_str()) !=
            GRAPH_SUCCESS) {
          OP_LOGE(op.GetName().c_str(), "input diagonal must be at least %lld-D, real rank is %lld.", input_rank - 1,
                  diagonal_shape.GetDimNum());
          return GRAPH_FAILED;
        }

        if (WithRankAtMost(diagonal_tensor_desc, input_rank, diagonal_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
          OP_LOGE(op.GetName().c_str(), "input diagonal must be at most %lld-D, real rank is %lld.", input_rank,
                  diagonal_shape.GetDimNum());
          return GRAPH_FAILED;
        }
      }

      auto input_dims = input_shape.GetDims();
      const int32_t num_rows = input_dims[input_rank - 2];
      const int32_t num_cols = input_dims[input_rank - 1];
      if (num_rows != ge::UNKNOWN_DIM && num_cols != ge::UNKNOWN_DIM) {
        if (lower_diag_index != 0 && (-num_rows >= lower_diag_index || lower_diag_index >= num_cols)) {
          OP_LOGE(op.GetName().c_str(), "lower_diag_index is out of bound.");
          return GRAPH_PARAM_INVALID;
        }
        if (upper_diag_index != 0 && (-num_rows >= upper_diag_index || upper_diag_index >= num_cols)) {
          OP_LOGE(op.GetName().c_str(), "upper_diag_index is out of bound.");
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
      OP_LOGE(op.GetName().c_str(), "failed to get subshape from diagonal_shape");
      return GRAPH_FAILED;
    }

    if (Concatenate(diagonal_prefix_shape, UnknownShapeOfRank(2), diagonal_shape) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "failed to concatenate diag_prefix_shape and 2-D unknown_shape");
      return GRAPH_FAILED;
    }

    if (Merge(input_shape, diagonal_shape, output_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "failed to merge input_shape and diagonal_shape.");
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
    OP_LOGE(op.GetName().c_str(), "input diagonal must be at least 1-D, real rank is %lld.",
            diagonal_tensor_desc.GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  Shape k_shape;
  auto k_tensor_desc = op.GetInputDesc(1);
  if (WithRankAtMost(k_tensor_desc, 1, k_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input k rank must be at most 1-D, real rank is %lld.",
            k_tensor_desc.GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  Shape num_rows_shape;
  auto num_rows_tensor_desc = op.GetInputDesc(2);
  if (WithRank(num_rows_tensor_desc, 0, num_rows_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input num_rows rank must be 0, real rank is %lld.",
            num_rows_tensor_desc.GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  Shape num_cols_shape;
  auto num_cols_tensor_desc = op.GetInputDesc(3);
  if (WithRank(num_cols_tensor_desc, 0, num_cols_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input num_cols rank must be 0, real rank is %lld.",
            num_cols_tensor_desc.GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  Shape padding_value_shape;
  auto padding_value_tensor_desc = op.GetInputDesc(4);
  if (WithRank(padding_value_tensor_desc, 0, padding_value_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input padding_value rank must be 0, real rank is %lld.",
            padding_value_tensor_desc.GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  auto output_desc = op.GetOutputDesc("output");
  Tensor k_tensor;
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
      OP_LOGE(op.GetName().c_str(), "diag_index must be a vector with one or two elements. It has %lld elements.",
              num_elements);
      return GRAPH_PARAM_INVALID;
    }
  }

  if (lower_diag_index > upper_diag_index) {
    OP_LOGE(op.GetName().c_str(), "lower_diag_index is greater than upper_diag_index.");
    return GRAPH_PARAM_INVALID;
  }

  auto diagonal_dims = diagonal_shape.GetDims();
  const int32_t diagonal_rank = diagonal_shape.GetDimNum();
  if (lower_diag_index < upper_diag_index) {
    const int64_t num_diags = diagonal_dims[diagonal_rank - 2];
    const int64_t other_dim = diagonal_dims[diagonal_rank - 1];
    if (num_diags != (upper_diag_index - lower_diag_index + 1)) {
      OP_LOGE(op.GetName().c_str(),
              "The number of rows of `diagonal` doesn't match the number of \
               diagonals implied from `d_lower` and `d_upper` \
               num_diags = %lld, d_lower = %lld, d_upper = %lld, other_dim = %lld",
              num_diags, lower_diag_index, upper_diag_index, other_dim);
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
    OP_LOGE(op.GetName().c_str(), "num_rows %d is too small.", num_rows);
    return GRAPH_PARAM_INVALID;
  }

  if (num_cols == ge::UNKNOWN_DIM) {
    num_cols = min_num_cols;
  } else if (num_cols < min_num_cols) {
    OP_LOGE(op.GetName().c_str(), "num_cols %d is too small.", num_cols);
    return GRAPH_PARAM_INVALID;
  }

  if (num_rows != min_num_rows && num_cols != min_num_cols) {
    OP_LOGE(op.GetName().c_str(),
            "num_rows and num_cols are not consistent with lower_diag_index, \
             upper_diag_index, and the length of the given diagonals. \
             num_rows = %lld != min_num_rows = %lld, num_cols = %lld != min_num_cols = %lld",
            num_rows, min_num_rows, num_cols, min_num_cols);
    return GRAPH_PARAM_INVALID;
  }

  Shape output_shape;
  OP_LOGE(op.GetName().c_str(), "num_rows: ", num_rows, " num_cols: ", num_cols);
  if (lower_diag_index == upper_diag_index) {
    if (ReplaceDim(diagonal_shape, diagonal_rank - 1, num_rows, output_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "failed to replacedim from diagonal_shape.");
      return GRAPH_FAILED;
    }
    if (Concatenate(output_shape, Shape({num_cols}), output_shape) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "failed to concatenate betweent outputshape and shape({num_cols}).");
      return GRAPH_FAILED;
    }
  } else {
    if (ReplaceDim(diagonal_shape, diagonal_rank - 2, num_rows, output_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "failed to replacedim from diagonal_shape.");
      return GRAPH_FAILED;
    }
    if (ReplaceDim(output_shape, diagonal_rank - 1, num_cols, output_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "failed to replacedim from output_shape.");
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

}  // namespace ge

