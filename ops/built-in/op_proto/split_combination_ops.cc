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
 * \file split_combination_ops.cpp
 * \brief
 */
#include "inc/split_combination_ops.h"

#include <string>
#include <vector>
#include <set>
#include <cmath>

#include "common/util/error_manager/error_manager.h"
#include "util/op_common_util.h"
#include "util/util.h"
#include "util/error_util.h"
#include "op_log.h"
#include "graph/utils/node_utils.h"
#include "register/infer_data_slice_registry.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
// ----------------Split OP Begin-------------------
static void CalcSplit(const Tensor& data, const DataType& dtype, std::vector<int64_t>& const_vec) {
  const uint8_t* const_data = data.GetData();
  size_t size = data.GetSize() / sizeof(int32_t);
  for (size_t n = 0; n < size; ++n) {
    const_vec.push_back(*((int32_t*)const_data));
  }
}

IMPLEMT_COMMON_INFERFUNC(SplitInferShape) {
  const vector<string> depend_name = {"split_dim"};
  PREPARE_DYNAMIC_SHAPE(depend_name);

  OP_LOGD(op.GetName().c_str(), "SplitInferShape start");
  auto x_inputdesc = op.GetInputDesc("x");
  auto x_shape = x_inputdesc.GetShape();
  auto x_dtype = x_inputdesc.GetDataType();
  TensorDesc td = op.GetDynamicOutputDesc("y", 0);

  std::vector<std::pair<int64_t, int64_t>> x_shape_range;
  std::vector<std::pair<int64_t, int64_t>> out_range;
  x_inputdesc.GetShapeRange(x_shape_range);
  OP_LOGD(op.GetName().c_str(), "SplitInferShape x_shape range is %s", to_string(x_shape_range).c_str());

  int64_t num_split;
  if (op.GetAttr("num_split", num_split) == GRAPH_FAILED) {
    OP_LOGE(op.GetName().c_str(), "get attr num_split failed");
    OpsGetAttrErrReport(op.GetName(), "num_split");
    return GRAPH_FAILED;
  }

  // input shape is [-2]
  if (x_shape.GetDims() == UNKNOWN_RANK) {
    td.SetShape(ge::Shape(UNKNOWN_RANK));
    td.SetDataType(x_dtype);
    for (auto i = 0; i < num_split; ++i) {
      OP_LOGD(op.GetName().c_str(), "SplitInferShape output shape is %s", to_string(td.GetShape()).c_str());
      op.UpdateDynamicOutputDesc("y", i, td);
    }
    return GRAPH_SUCCESS;
  }

  Tensor split_dim_data;
  if (op.GetInputConstData("split_dim", split_dim_data) != GRAPH_SUCCESS) {
    // input split_dim is not const
    OP_LOGD(op.GetName().c_str(), "Get constValue failed of [split_dim]");
    if (x_shape_range.size() == 1) {
      if (x_shape_range[0].second == -1) {
        out_range.push_back(std::pair<int64_t, int64_t>(0, x_shape_range[0].second));
      } else {
        out_range.push_back(std::pair<int64_t, int64_t>(0, ceil((float)x_shape_range[0].second / (float)num_split)));
      }
    } else {
      for(size_t i = 0; i < x_shape_range.size(); ++i) {
        out_range.push_back(std::pair<int64_t, int64_t>(0, x_shape_range[i].second));
      }
    }
    if (num_split > 1) {
      for(size_t i = 0; i < x_shape.GetDimNum(); ++i) {
        x_shape.SetDim(i, -1);
      }
    }
    td.SetShape(x_shape);
    td.SetDataType(x_dtype);
    td.SetShapeRange(out_range);
    for (auto i = 0; i < num_split; ++i) {
      OP_LOGD(op.GetName().c_str(), "SplitInferShape output shape is %s", to_string(x_shape).c_str());
      OP_LOGD(op.GetName().c_str(), "SplitInferShape out_range is %s", to_string(out_range).c_str());
      op.UpdateDynamicOutputDesc("y", i, td);
    }
    return GRAPH_SUCCESS;
  }
  auto split_dim_dtype = op.GetInputDesc("split_dim").GetDataType();
  std::vector<int64_t> split_dim_vec;
  CalcSplit(split_dim_data, split_dim_dtype, split_dim_vec);

  int64_t split_dim = -1;
  if(split_dim_vec.size() > 0){
    split_dim = split_dim_vec[0];
  } else {
    OP_LOGE(op.GetName().c_str(), "size of split_dim_vec must be larger than 0");
    OpsInputShapeErrReport(op.GetName(), "size of split_dim_vec must be larger than 0",
                          "size of split_dim_vec", ConcatString(split_dim_vec.size()));
    return GRAPH_FAILED;
  }
  if (split_dim < 0) {
    split_dim += x_shape.GetDimNum();
  }

 for(size_t i = 0; i < x_shape_range.size(); ++i) {
    if(split_dim == static_cast<int>(i)) {
      int64_t range_left = -1;
      if(x_shape_range[i].first == 1) {
        range_left = x_shape_range[i].first;
      } else {
        range_left = floor((float)x_shape_range[i].first / (float)num_split);
      }

      int64_t range_right = -1;
      if(x_shape_range[i].second == -1 ){
        range_right = x_shape_range[i].second;
      } else {
        range_right = ceil((float)x_shape_range[i].second / (float)num_split);
      }
      out_range.push_back(std::pair<int64_t, int64_t>(range_left, range_right));
    } else {
      out_range.push_back(x_shape_range[i]);
    }
  }

  if (x_shape.GetDim(split_dim) == -1) {
    OP_LOGD(op.GetName().c_str(), "shape at split_dim is -1");
  } else {
    auto length = x_shape.GetDim(split_dim) / num_split;
    OP_LOGD(op.GetName().c_str(), "shape at split_dim is %d", x_shape.GetDim(split_dim));
    x_shape.SetDim(split_dim, length);
  }

  td.SetShape(x_shape);
  td.SetDataType(x_dtype);
  td.SetShapeRange(out_range);

  for (auto i = 0; i < num_split; ++i) {
    OP_LOGD(op.GetName().c_str(), "SplitInferShape output shape is %s", to_string(x_shape).c_str());
    OP_LOGD(op.GetName().c_str(), "SplitInferShape out_range is %s", to_string(out_range).c_str());
    op.UpdateDynamicOutputDesc("y", i, td);
  }
   return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Split, SplitInferShape);
// ----------------Split OP End-------------------

// ----------------SplitD OP Begin-------------------
IMPLEMT_COMMON_INFERFUNC(SplitDInferShape) {
  const vector<string> depends;
  PREPARE_DYNAMIC_SHAPE(depends);

  OP_LOGD(op.GetName().c_str(), "SplitDInferShape");
  auto x_desc = op.GetInputDesc("x");
  auto x_shape = x_desc.GetShape();
  auto x_dtype = x_desc.GetDataType();
  TensorDesc td = op.GetDynamicOutputDesc("y", 0);
  std::vector<std::pair<int64_t, int64_t>> x_shape_range;
  std::vector<std::pair<int64_t, int64_t>> out_range;
  x_desc.GetShapeRange(x_shape_range);
  OP_LOGD(op.GetName().c_str(), "SplitDInferShape x_shape_range is %s", to_string(x_shape_range).c_str());

  int64_t split_dim;
  if (op.GetAttr("split_dim", split_dim) == GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("split_dim");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int64_t num_split;
  if (op.GetAttr("num_split", num_split) == GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("num_split");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // input shape is [-2]
  if (x_shape.GetDims() == UNKNOWN_RANK) {
    td.SetShape(ge::Shape(UNKNOWN_RANK));
    td.SetDataType(x_dtype);
    for (auto i = 0; i < num_split; ++i) {
      OP_LOGD(op.GetName().c_str(), "SplitDInferShape output shape is %s", to_string(td.GetShape()).c_str());
      op.UpdateDynamicOutputDesc("y", i, td);
    }
    return GRAPH_SUCCESS;
  }

  // check attr
  int64_t dim_num = x_shape.GetDimNum();
  if ((split_dim < -dim_num) || (split_dim >= dim_num)) {
    std::string err_msg = GetParamOutRangeErrMsg("split_dim", ConcatString("[", -dim_num, ", ", dim_num, ")"), ConcatString(split_dim));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (num_split < 1) {
    std::string err_msg = GetParamOutRangeErrMsg("num_split", ConcatString("[1,]"), ConcatString(num_split));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (split_dim < 0) {
    split_dim += x_shape.GetDimNum();
  }

  for(size_t i = 0; i < x_shape_range.size(); ++i) {
    if(split_dim == static_cast<int>(i)) {
      int64_t range_left = -1;
      if(x_shape_range[i].first == 1) {
        range_left = x_shape_range[i].first;
      } else {
        range_left = floor((float)x_shape_range[i].first / (float)num_split);
      }

      int64_t range_right = -1;
      if(x_shape_range[i].second == -1 ){
        range_right = x_shape_range[i].second;
      } else {
        range_right = ceil((float)x_shape_range[i].second / (float)num_split);
      }
      out_range.push_back(std::pair<int64_t, int64_t>(range_left, range_right));
    } else {
      out_range.push_back(x_shape_range[i]);
    }
  }

  if (x_shape.GetDim(split_dim) == -1) {
    OP_LOGD(op.GetName().c_str(), "shape at split_dim is -1");
  } else {
    OP_LOGD(op.GetName().c_str(), "shape at split_dim is %d", x_shape.GetDim(split_dim));
    auto length = x_shape.GetDim(split_dim) / num_split;
    x_shape.SetDim(split_dim, length);
  }

  td.SetShape(x_shape);
  td.SetDataType(x_dtype);
  td.SetShapeRange(out_range);

  for (auto i = 0; i < num_split; ++i) {
    OP_LOGD(op.GetName().c_str(), "SplitDInferShape output shape is %s", to_string(x_shape).c_str());
    OP_LOGD(op.GetName().c_str(), "SplitDInferShape out_range is %s", to_string(out_range).c_str());
    op.UpdateDynamicOutputDesc("y", i, td);
  }
   return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SplitD, SplitDInferShape);
// ----------------SplitD OP End-------------------

// ----------------SplitV OP Begin-------------------
static void CalcSplitV(const Tensor& data, const DataType& dtype, std::vector<int64_t>& const_vec) {
  const uint8_t* constData = data.GetData();
  size_t size;
  if (dtype == ge::DT_INT32) {
    size = data.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; ++i) {
      const_vec.push_back(*((int32_t*)constData + i));
    }
  } else {
    size = data.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; ++i) {
      const_vec.push_back(*((int64_t*)constData + i));
    }
  }
}

IMPLEMT_COMMON_INFERFUNC(SplitVInferShape) {
  OP_LOGD(op.GetName().c_str(), "SplitV InferShape start");
  const vector<string> depend_name = {"size_splits", "split_dim"};
  PREPARE_DYNAMIC_SHAPE(depend_name);

  auto x_desc = op.GetInputDesc("x");
  auto x_shape = x_desc.GetShape();
  DataType x_dtype = x_desc.GetDataType();
  TensorDesc td = op.GetDynamicOutputDesc("y", 0);
  TensorDesc tdx = op.GetInputDesc("x");
  int64_t num_split;
  if (op.GetAttr("num_split", num_split) == GRAPH_FAILED) {
    OP_LOGE(op.GetName().c_str(), "get attr num_split failed");
    OpsGetAttrErrReport(op.GetName(), "num_split");
    return GRAPH_FAILED;
  }
  if (num_split <= 0) {
    OP_LOGE(op.GetName().c_str(), "num_split must be greater than 0");
    OpsAttrValueErrReport(op.GetName(), "num_split", "greater than 0", ConcatString(num_split));
    return GRAPH_FAILED;
  }

  // input x shape is [-2], aicpu
  if (x_shape.GetDims() == UNKNOWN_RANK) {
    td.SetShape(ge::Shape(UNKNOWN_RANK));
    td.SetDataType(x_dtype);
    for (auto i = 0; i < num_split; ++i) {
      op.UpdateDynamicOutputDesc("y", i, td);
    }
    return GRAPH_SUCCESS;
  }

  int64_t xDimNum = x_shape.GetDimNum();
  if (xDimNum <= 0) {
    OP_LOGE(op.GetName().c_str(), "size of split_vec must be larger than 0");
    OpsInputShapeErrReport(op.GetName(), "x dim num must be larger than 0",
                           "dim num of x shape", ConcatString(xDimNum));
    return GRAPH_FAILED;
  }

  std::vector<std::pair<int64_t, int64_t>> x_shape_range;
  std::vector<std::pair<int64_t, int64_t>> out_range;
  x_desc.GetShapeRange(x_shape_range);

  Tensor split_dim_data;
  Tensor size_splits_data;
  auto is_split_dim_const = op.GetInputConstData("split_dim", split_dim_data);
  auto is_size_splits_const = op.GetInputConstData("size_splits", size_splits_data);

  if (is_split_dim_const != GRAPH_SUCCESS) {
    // input split_dim and size_splits is not const
    OP_LOGD(op.GetName().c_str(), "SplitVInferShape first");
    for (int64_t i = 0; i < xDimNum; ++i) {
      x_shape.SetDim(i, -1);
    }
    for (size_t i = 0; i < x_shape_range.size(); ++i) {
      out_range.push_back(std::pair<int64_t, int64_t>(0, x_shape_range[i].second));
    }

    td.SetShape(x_shape);
    td.SetDataType(x_dtype);
    td.SetShapeRange(out_range);
    for (auto i = 0; i < num_split; ++i) {
      op.UpdateDynamicOutputDesc("y", i, td);
    }
    OP_LOGD(op.GetName().c_str(), "SplitVInferShape end, split_dim is not const");
    return GRAPH_SUCCESS;
  }

  OP_LOGD(op.GetName().c_str(), "SplitVInferShape, get split_dim value");
  auto split_dim_dtype = op.GetInputDesc("split_dim").GetDataType();
  std::vector<int64_t> split_dim_vec;
  CalcSplitV(split_dim_data, split_dim_dtype, split_dim_vec);

  if (split_dim_vec.size() == 0) {
    OP_LOGE(op.GetName().c_str(), "size of split_dim must be larger than 0");
    OpsInputShapeErrReport(op.GetName(), "size of split_dim must be larger than 0",
                           "size of split_dim", ConcatString(split_dim_vec.size()));
    return GRAPH_FAILED;
  }
  int64_t split_dim = split_dim_vec[0];

  if (split_dim < -xDimNum || split_dim >= xDimNum) {
    OP_LOGE(op.GetName().c_str(), "split_dim is invalid");
    string minValue = ConcatString(-xDimNum);
    string maxValue = ConcatString(xDimNum - 1);
    string excepted_value = ConcatString("in the range of[", minValue, ",", maxValue, "]");
    OpsInputShapeErrReport(op.GetName(), "split_dim", excepted_value, ConcatString(split_dim));
    return GRAPH_FAILED;
  }
  if (split_dim < 0) {
    split_dim += xDimNum;
  }

  if (is_size_splits_const != GRAPH_SUCCESS) {
    x_shape.SetDim(split_dim, -1);

    for (size_t i = 0; i < x_shape_range.size(); ++i) {
      if (split_dim == static_cast<int>(i)) {
        out_range.push_back(std::pair<int64_t, int64_t>(0, x_shape_range[i].second));
      } else {
        out_range.push_back(x_shape_range[i]);
      }
    }

    td.SetShape(x_shape);
    td.SetDataType(x_dtype);
    td.SetShapeRange(out_range);
    for (auto i = 0; i < num_split; ++i) {
      op.UpdateDynamicOutputDesc("y", i, td);
    }
    OP_LOGD(op.GetName().c_str(), "SplitVInferShape end, size_splits is not const");
    return GRAPH_SUCCESS;
  }

  OP_LOGD(op.GetName().c_str(), "SplitVInferShape, get size_splits value");
  auto size_splits_dtype = op.GetInputDesc("size_splits").GetDataType();
  std::vector<int64_t> size_splits_vec;
  CalcSplitV(size_splits_data, size_splits_dtype, size_splits_vec);

  std::vector<int64_t> size_splits(size_splits_vec);
  if (std::count(size_splits.begin(), size_splits.end(), -1) > 1) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), string("the value of size_splits is invalid!"));
    return GRAPH_FAILED;
  }

  int64_t dim = x_shape.GetDim(split_dim);
  if (dim == -1) {
    OP_LOGD(op.GetName().c_str(), "x shape at split_dim is -1");
  } else {
    if (std::find(size_splits.begin(), size_splits.end(), -1) != size_splits.end()) {
      OP_LOGD(op.GetName().c_str(), "-1 is in size_splits");
      int64_t size_splits_sum = 0;
      int64_t temp_index = -1;
      for (size_t i = 0; i < size_splits.size(); ++i) {
        if (size_splits[i] != -1) {
          size_splits_sum += size_splits[i];
        } else {
          temp_index = i;
        }
      }
      if (dim != size_splits_sum && temp_index != -1) {
        size_splits[temp_index] = dim - size_splits_sum;
      }
    }
  }

  for (auto i = 0; i < num_split; ++i) {
    x_shape.SetDim(split_dim, size_splits[i]);
    tdx.SetShape(x_shape);
    tdx.SetOriginShape(x_shape);
    tdx.SetDataType(x_dtype);
    out_range.clear();
    for (size_t j = 0; j < x_shape_range.size(); ++j) {
      if (split_dim == static_cast<int>(j)) {
        out_range.push_back(std::pair<int64_t, int64_t>(size_splits[i], size_splits[i]));
      } else {
        out_range.push_back(x_shape_range[j]);
      }
    }
    tdx.SetShapeRange(out_range);
    op.UpdateDynamicOutputDesc("y", i, tdx);
  }
  OP_LOGD(op.GetName().c_str(), "SplitVInferShape success");
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SplitV, SplitVInferShape);
// ----------------SplitV OP End-------------------

// ----------------SplitVD OP Begin-------------------
IMPLEMT_INFERFUNC(SplitVD, SplitVDInferShape) {
  auto tensordesc = op.GetInputDesc("x");
  auto shape = tensordesc.GetShape();
  DataType inputDtype = tensordesc.GetDataType();
  TensorDesc td = op.GetDynamicOutputDesc("y", 0);

  int64_t split_dim;
  if (op.GetAttr("split_dim", split_dim) == GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("split_dim");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  vector<int64_t> size_splits;
  if (op.GetAttr("size_splits", size_splits) == GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("size_splits");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  int64_t num_split;
  if (op.GetAttr("num_split", num_split) == GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("num_split");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  int64_t dim_num = shape.GetDimNum();
  if ((split_dim < -dim_num) || (split_dim >= dim_num)) {
    std::string err_msg = GetParamOutRangeErrMsg("split_dim", ConcatString("[", -dim_num, ", ", dim_num, ")"), ConcatString(split_dim));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (num_split < 1) {
    std::string err_msg = GetParamOutRangeErrMsg("num_split", ConcatString("[1,]"), ConcatString(num_split));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (split_dim < 0) {
    split_dim += shape.GetDimNum();
  }
  vector<int64_t> adjust_size_splits;
  if (size_splits.size() == 0) {
    int64_t dim = shape.GetDim(split_dim);
    int64_t batch = dim / num_split;
    if (dim % num_split != 0) {
      std::string err_msg = OtherErrMsg("dimvalue should be divisible by num_split");
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    for (auto i = 0; i < num_split; ++i) {
      shape.SetDim(split_dim, batch);
      td.SetShape(shape);
      td.SetDataType(inputDtype);
      op.UpdateDynamicOutputDesc("y", i, td);
      adjust_size_splits.push_back(batch);
    }
  } else if (static_cast<int>(size_splits.size() + 1) == num_split) {
    int64_t dim = shape.GetDim(split_dim);
    int64_t sum = 0;
    for (unsigned int i = 0; i < size_splits.size(); ++i) {
      sum = sum + size_splits[i];
      shape.SetDim(split_dim, size_splits[i]);
      td.SetShape(shape);
      td.SetDataType(inputDtype);
      op.UpdateDynamicOutputDesc("y", i, td);
      adjust_size_splits.push_back(size_splits[i]);
    }
    if (dim - sum > 0) {
      shape.SetDim(split_dim, dim - sum);
      td.SetShape(shape);
      td.SetDataType(inputDtype);
      op.UpdateDynamicOutputDesc("y", size_splits.size(), td);
      adjust_size_splits.push_back(dim - sum);
    }
  } else {
    int64_t dim = shape.GetDim(split_dim);
    int64_t size_splits_sum = 0;
    for (size_t i = 0; i < size_splits.size(); ++i) {
      if (size_splits[i] != -1) {
        size_splits_sum += size_splits[i];
      }
    }
    if (dim != size_splits_sum) {
      for (size_t i = 0; i < size_splits.size(); ++i) {
        if (size_splits[i] == -1) {
          size_splits[i] = dim - size_splits_sum;
        }
      }
    }
    for (auto i = 0; i < num_split; ++i) {
      shape.SetDim(split_dim, size_splits[i]);
      td.SetShape(shape);
      td.SetDataType(inputDtype);
      op.UpdateDynamicOutputDesc("y", i, td);
      adjust_size_splits.push_back(size_splits[i]);
    }
  }
  op.SetAttr("size_splits", adjust_size_splits);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SplitVD, SplitVDInferShape);
// ----------------SplitVD OP End-------------------

// ----------------ConcatV2D OP Begin-------------------
static void JoinShapeRanges(vector<pair<int64_t, int64_t>> &dest_ranges,
                            const vector<pair<int64_t, int64_t>> &src_ranges) {
  auto dest_size = dest_ranges.size();
  auto src_size = src_ranges.size();
  if (dest_size != src_size) {
    return;
  }

  for (size_t i = 0; i < dest_size; i++) {
    dest_ranges[i].first = std::max(dest_ranges[i].first, src_ranges[i].first);
    dest_ranges[i].second = std::min(dest_ranges[i].second, src_ranges[i].second);
  }
}

static vector<pair<int64_t, int64_t>> GetShapeRangesWithUnKnowConcatDim(Operator &op, int64_t num_concat) {
  vector<pair<int64_t, int64_t>> input_shape_ranges;
  vector<vector<pair<int64_t, int64_t>>> all_input_shape_ranges;
  vector<pair<int64_t, int64_t>> output_shape_ranges;
  bool has_shape_ranges = false;
  for (int32_t i = 0; i < num_concat; i++) {
    const auto input_desc = op.GetDynamicInputDesc("x", i);
    (void) input_desc.GetShapeRange(input_shape_ranges);
    OP_LOGD(op.GetName().c_str(), "input shape range:%s", to_string(input_shape_ranges).c_str());
    if (input_shape_ranges.empty()) {
      auto shape_dims = input_desc.GetShape().GetDims();
      MakeUpShapeRange(shape_dims, input_shape_ranges);
    } else {
      has_shape_ranges = true;
    }

    all_input_shape_ranges.push_back(input_shape_ranges);
  }

  if (has_shape_ranges) {
    output_shape_ranges = all_input_shape_ranges[0];
    for (size_t i = 1; i < all_input_shape_ranges.size(); i++) {
      if (output_shape_ranges.size() != all_input_shape_ranges[i].size()) {
        continue;
      }

      for (size_t j = 0; j < output_shape_ranges.size(); j++) {
        output_shape_ranges[j].first = std::max(output_shape_ranges[j].first, all_input_shape_ranges[i][j].first);
        if (output_shape_ranges[j].second == -1 || all_input_shape_ranges[i][j].second == -1) {
          output_shape_ranges[j].second = -1;
        } else {
          output_shape_ranges[j].second = output_shape_ranges[j].second + all_input_shape_ranges[i][j].second;
        }
      }
    }
  }

  return output_shape_ranges;
}

bool JoinShapes(vector<int64_t>& dst_shape, const vector<int64_t>& src_shape, int64_t axis) {
  if (dst_shape == src_shape) {
    return true;
  }

  if (dst_shape.empty() || IsUnknownRankShape(dst_shape)) {
    dst_shape = src_shape;
    return true;
  }

  if (!IsUnknownRankShape(src_shape)) {
    if (dst_shape.size() != src_shape.size()) {
      return false;
    }
    auto shape_dims = dst_shape.size();
    for (size_t i = 0; i < shape_dims; i++) {
      if (dst_shape[i] == src_shape[i]) {
        continue;
      }

      if (axis != static_cast<int64_t>(i) && dst_shape[i] != UNKNOWN_DIM && src_shape[i] != UNKNOWN_DIM) {
        return false;
      }

      if (src_shape[i] != UNKNOWN_DIM) {
        dst_shape[i] = src_shape[i];
      }
    }
  }

  return true;
}

bool ConcatInferShapeCommonStatic(Operator& op, const int64_t dynamic_input_start_idx,
                                  int64_t num_concat, int64_t axis) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc(dynamic_input_start_idx);
  auto output_desc = op_info->MutableOutputDesc(0);
  const GeShape& input_shape = input_desc->MutableShape();
  GeShape& output_shape = output_desc->MutableShape();
  output_shape = input_shape;
  if (output_shape.IsUnknownShape() || num_concat == 1) {
    // dynamic case or the input only one will use dynamic infer func
    return false;
  }

  if (output_shape.IsScalar()) {
    // scalar to shape [1]
    output_shape.SetDimNum(1);
    output_shape.SetDim(0, 1);
  }
  const size_t output_dim = output_shape.GetDimNum();

  if ((axis < -static_cast<int64_t>(output_dim)) || (axis >= static_cast<int64_t>(output_dim))) {
    // axes is valid
    return false;
  }
  if (axis < 0) {
    axis += output_dim;
  }
  int64_t concat_dim_size = output_shape.GetDim(axis);

  for (int64_t input_idx = 1; input_idx < num_concat; input_idx++) {
    auto input_i_desc = op_info->MutableInputDesc(input_idx + dynamic_input_start_idx);
    const GeShape& input_i_shape = input_i_desc->MutableShape();
    if (input_i_shape.IsScalar() && output_dim == 1) {
      concat_dim_size += 1;
      continue;
    }
    if (input_i_shape.IsUnknownShape()) {
      // dynamic case
      return false;
    }
    if (input_i_shape.GetDimNum() != output_dim) {
      // input shape size is not equal output
      return false;
    }
    // check whether the non concat dim is equal
    for (int64_t check_dim = 0; check_dim < output_dim; check_dim++) {
      if (check_dim != axis && input_i_shape.GetDim(check_dim) != output_shape.GetDim(check_dim)) {
        return false;
      }
    }
    concat_dim_size += input_i_shape.GetDim(axis);
  }
  output_shape.SetDim(axis, concat_dim_size);

  // set data type
  output_desc->SetDataType(input_desc->GetDataType());
  return true;
}

static graphStatus ConcatInferShapeCommon(Operator& op, const int64_t dy_input_start_idx,
                                          int64_t num_concat, int64_t axis, bool unknown_axis) {
  if (num_concat <= 0) {
    std::string err_msg = GetAttrValueErrMsg("num_concat", std::to_string(num_concat), ConcatString("num_concat > 0"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // try static infershape directly
  if (!unknown_axis) {
    if (ConcatInferShapeCommonStatic(op, dy_input_start_idx, num_concat, axis)) {
      return GRAPH_SUCCESS;
    }
  }
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  size_t dim_num = 0;
  std::vector<GeTensorDescPtr> input_x_desc;
  const string input_name = "x";
  string input_name_i = "x63";
  for (int64_t input_idx = 0; input_idx < num_concat; input_idx++) {
    input_name_i = input_name + std::to_string(input_idx);
    auto input_desc = op_info->MutableInputDesc(input_name_i);
    if (!input_desc) {
      std::string err_msg = GetInputInvalidErrMsg(input_name_i.c_str());
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }

    input_x_desc.emplace_back(op_info->MutableInputDesc(input_name_i));
  }

  bool all_unknown_rank_shape = true;
  for (const auto& desc : input_x_desc) {
    dim_num = std::max(dim_num, desc->MutableShape().GetDimNum());
    all_unknown_rank_shape = IsUnknownRankShape(desc->MutableShape().GetDims()) && all_unknown_rank_shape;
  }

  if (all_unknown_rank_shape) {
    DataType input_dtype = input_x_desc[0]->GetDataType();
    auto output_desc = op_info->MutableOutputDesc(0);
    output_desc->SetDataType(input_dtype);
    output_desc->SetShape(ge::GeShape(UNKNOWN_RANK));
    OP_LOGD(op.GetName().c_str(), "output shape:%s", to_string(output_desc->GetShape()).c_str());
    return GRAPH_SUCCESS;
  }

  if (unknown_axis) {
    DataType input_dtype = input_x_desc[0]->GetDataType();
    auto output_desc = op_info->MutableOutputDesc(0);
    output_desc->SetDataType(input_dtype);
    vector<int64_t> dimVector(dim_num, -1);
    output_desc->SetShape(ge::GeShape(dimVector));
    auto output_shape_ranges = GetShapeRangesWithUnKnowConcatDim(op, num_concat);
    if (!output_shape_ranges.empty()) {
      output_desc->SetShapeRange(output_shape_ranges);
      OP_LOGD(op.GetName().c_str(), "output shape range:%s", to_string(output_shape_ranges).c_str());
    }
    OP_LOGD(op.GetName().c_str(), "output shape:%s", to_string(output_desc->GetShape()).c_str());
    return GRAPH_SUCCESS;
  }

  if ((axis < -static_cast<int64_t>(dim_num)) || (axis >= static_cast<int64_t>(dim_num))) {
    string range_msg = ConcatString(-dim_num, dim_num);
    std::string err_msg = GetParamOutRangeErrMsg("axis", range_msg, std::to_string(axis));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int64_t non_negative_axis = axis;
  if (non_negative_axis < 0) {
    non_negative_axis += dim_num;
  }

  vector<int64_t> output_shape_dims;
  for (const auto& desc : input_x_desc) {
    auto input_shape_dims = desc->MutableShape().GetDims();
    if (!JoinShapes(output_shape_dims, input_shape_dims, non_negative_axis)) {
      vector<vector<int64_t>> shapes = {output_shape_dims, input_shape_dims};
      std::string err_msg = OtherErrMsg(ConcatString("the input shape dims should be equal except merge axis,"
                                                     "shapes:", ops::to_string(shapes), "axis:", std::to_string(axis)));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }

  int32_t size = 0;
  for (const auto& desc : input_x_desc) {
    if (IsUnknownRankShape(desc->MutableShape().GetDims())) {
      size = -1;
      break;
    }

    auto dim_value = desc->MutableShape().GetDim(non_negative_axis);
    if (dim_value == -1) {
      size = -1;
      break;
    }

    if (size != -1) {
      size += dim_value;
    }
  }

  output_shape_dims[non_negative_axis] = size;
  DataType input_dtype = input_x_desc[0]->GetDataType();
  auto output_desc = op_info->MutableOutputDesc(0);
  output_desc->SetDataType(input_dtype);
  output_desc->SetShape(ge::GeShape(output_shape_dims));
  OP_LOGD(op.GetName().c_str(), "output shape:%s", to_string(output_desc->GetShape()).c_str());

  if (IsUnKnownShape(output_shape_dims)) {
    vector<pair<int64_t, int64_t>> input_shape_ranges;
    vector<pair<int64_t, int64_t>> output_shape_ranges;
    pair<int64_t, int64_t> output_concat_dim_range(0, 0);
    for (const auto& input_desc : input_x_desc) {
      if (IsUnknownRankShape(input_desc->MutableShape().GetDims())) {
        output_concat_dim_range = {0, -1};
        continue;
      }

      input_shape_ranges.clear();
      input_desc->GetShapeRange(input_shape_ranges);
      OP_LOGD(op.GetName().c_str(), "input shape range:%s", to_string(input_shape_ranges).c_str());
      if (input_shape_ranges.empty()) {
        MakeUpShapeRange(input_desc->MutableShape().GetDims(), input_shape_ranges);
      }

      if (static_cast<int64_t>(input_shape_ranges.size()) > non_negative_axis) {
        output_concat_dim_range.first += input_shape_ranges[non_negative_axis].first;
        if (input_shape_ranges[non_negative_axis].second == -1 || output_concat_dim_range.second == -1) {
          output_concat_dim_range.second = -1;
        } else {
          output_concat_dim_range.second += input_shape_ranges[non_negative_axis].second;
        }
      }

      if (output_shape_ranges.empty()) {
        output_shape_ranges = input_shape_ranges;
      } else {
        JoinShapeRanges(output_shape_ranges, input_shape_ranges);
      }
    }

    if (output_concat_dim_range.second != 0 &&
        static_cast<uint64_t>(non_negative_axis) < output_shape_ranges.size()) {
      output_shape_ranges[non_negative_axis] = output_concat_dim_range;
    }

    output_desc->SetShapeRange(output_shape_ranges);
    OP_LOGD(op.GetName().c_str(), "output shape range:%s", to_string(output_shape_ranges).c_str());
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ConcatV2DInferShape) {
  const vector<string> depends;
  PREPARE_DYNAMIC_SHAPE(depends);

  int64_t num_concatext2;
  if (op.GetAttr("N", num_concatext2) == GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("N");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int64_t axis;
  if (op.GetAttr("concat_dim", axis) == GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("axis");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  return ConcatInferShapeCommon(op, 0, num_concatext2, axis, false);
}

COMMON_INFER_FUNC_REG(ConcatV2D, ConcatV2DInferShape);

static graphStatus ConcatInferDataSliceCommon(Operator& op, int64_t num_concat, int64_t axis) {
  if (num_concat <= 0) {
    OP_LOGE(op.GetName().c_str(), "Check N > 0 failed, N is %lld.", num_concat);
    OpsAttrValueErrReport(op.GetName(), "N", ">0", std::to_string(num_concat));
    return GRAPH_FAILED;
  }

  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  OP_LOGE_IF(!op_info, GRAPH_FAILED, op.GetName(), "GetOpDescFromOperator failed.");
  std::vector<GeTensorDescPtr> input_x_desc;
  const string input_name = "x";
  string input_name_i = "x63";
  for (int64_t input_idx = 0; input_idx < num_concat; input_idx++) {
    input_name_i = input_name + std::to_string(input_idx);
    auto input_desc = op_info->MutableInputDesc(input_name_i);
    if (!input_desc) {
      OpsMissInputErrReport(op.GetName(), input_name_i);
      OP_LOGE(op.GetName().c_str(), "Get input desc %s failed.", input_name_i.c_str());
      return GRAPH_FAILED;
    }
    input_x_desc.emplace_back(op_info->MutableInputDesc(input_name_i));
  }

  auto output_desc = op_info->MutableOutputDesc(0);
  OP_LOGE_IF(!output_desc, GRAPH_FAILED, op.GetName(), "Get output desc failed.");
  vector<vector<int64_t>> output_data_slice;
  OP_LOGE_IF(!AttrUtils::GetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, output_data_slice), GRAPH_FAILED,
             op.GetName(), "Output no data slice, not need infer input");

  size_t dim_num = output_desc->GetOriginShape().GetDimNum();

  if ((axis < -static_cast<int64_t>(dim_num)) || (axis >= static_cast<int64_t>(dim_num))) {
    OpsInputShapeDimErrReport(op.GetName(), "axis", ConcatString(dim_num), ConcatString(-dim_num), ConcatString(axis));
    OP_LOGE(op.GetName().c_str(), "Axis[%lld] value out of range[%lld, %lld).", axis, -dim_num, dim_num);
    return GRAPH_FAILED;
  }

  int64_t non_negative_axis = axis;
  if (non_negative_axis < 0) {
    non_negative_axis += dim_num;
  }

  std::string origin_format = ToFormatString(output_desc->GetOriginFormat());
  std::string new_format = ToFormatString(output_desc->GetFormat());
  std::set<string> supported_formats = {"ND", "NC1HWC0", "NCHW", "NHWC"};
  if (supported_formats.count(new_format) != 0 && !output_data_slice.empty()) {
    auto new_non_negative_axis = GetNewAxis4NewFormat(output_desc->GetOriginShape().GetDimNum(), non_negative_axis,
                                                      origin_format, new_format, false);
    OP_LOGE_IF(new_non_negative_axis.empty(), GRAPH_FAILED, op.GetName(),
               "Get new axis from %s to %s failed, origin_shape len is %llu.",
               origin_format.c_str(), new_format.c_str(), output_desc->GetOriginShape().GetDimNum());
    non_negative_axis = new_non_negative_axis[0];
    vector<vector<int64_t>> input_data_slice = output_data_slice;
    for (const auto& desc : input_x_desc) {
      size_t axis_index = static_cast<size_t>(non_negative_axis);
      auto dim_value = desc->MutableShape().GetDim(axis_index);
      input_data_slice[non_negative_axis] = {0, dim_value};
      OP_LOGE_IF(!AttrUtils::SetListListInt(desc, ge::ATTR_NAME_DATA_SLICE, input_data_slice), GRAPH_FAILED,
                 op.GetName(), "Set input(%s) data slice failed", desc->GetName().c_str());
    }

    return GRAPH_SUCCESS;
  }

  return GRAPH_FAILED;
}

IMPLEMT_INFER_DATA_SLICE(ConcatV2D, ConcatV2DInferDataSlice) {
  int64_t num_concat;
  if (op.GetAttr("N", num_concat) == GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "N");
    OP_LOGE(op.GetName().c_str(), "get attr N failed");
    return GRAPH_FAILED;
  }

  int64_t axis;
  if (op.GetAttr("concat_dim", axis) == GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "axis");
    OP_LOGE(op.GetName().c_str(), "get attr axis failed");
    return GRAPH_FAILED;
  }

  return ConcatInferDataSliceCommon(op, num_concat, axis);
}

INFER_DATA_SLICE_FUNC_REG(ConcatV2D, ConcatV2DInferDataSlice);
// ----------------ConcatV2D OP End-------------------

// ----------------ParallelConcat OP Begin-------------------
static std::vector<int64_t> GetAttrValue(const ge::Operator& op, const std::string& key_name) {
  std::vector<int64_t> list;
  if (ge::GRAPH_SUCCESS != op.GetAttr(key_name, list)) {
    std::string err_msg = GetInputInvalidErrMsg("ConstValue");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
  }
  return list;
}

static bool CheckListEmpty(const std::string& opName, const std::vector<int64_t>& list, const std::string& attrName) {
  if (list.empty()) {
    std::string err_msg = OtherErrMsg("the list is empty !");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(opName, err_msg);
    return false;
  }
  return true;
}

IMPLEMT_COMMON_INFERFUNC(ParallelConcatInferShape) {
  auto tensordesc = op.GetDynamicInputDesc("values", 0);
  std::vector<int64_t> shape;
  shape = GetAttrValue(op, "shape");
  int64_t num_1;
  if (!CheckListEmpty(op.GetName(), shape, "shape")) {
    return GRAPH_FAILED;
  }

  if (GRAPH_SUCCESS != op.GetAttr("N", num_1)) {
    std::string err_msg = GetInputInvalidErrMsg("N");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  auto x_shape = tensordesc.GetShape();
  int64_t dimnum;
  dimnum = x_shape.GetDimNum();
  if (shape[0] != num_1) {
    string excepted_value = ConcatString("equal to the num of N[", num_1, "]");
    std::string err_msg = GetAttrValueErrMsg("attr shape[0]", ConcatString(shape[0]), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  for (int64_t i = 1; i < dimnum; i++) {
    if (x_shape.GetDim(i) != shape[i]) {
      string excepted_value = ConcatString("match the output_data's shape[", shape[i], "]");
      string attr_name = ConcatString("x_shape.GetDim(", i, ")");
      std::string err_msg = GetAttrValueErrMsg(attr_name, ConcatString(x_shape.GetDim(i)), excepted_value);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }
  DataType input_dtype = op.GetDynamicInputDesc("values", 0).GetDataType();
  TensorDesc outDesc = op.GetOutputDesc("output_data");
  std::string name_out = outDesc.GetName();
  outDesc.SetShape(ge::Shape(shape));
  outDesc.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("output_data", outDesc);
  OP_LOGI(op.GetName().c_str(), "input shape attr is: %s, set output shape :%s, Obtain REAL OUTPUT SHAPE is %s",
          to_string(ge::Shape(shape)).c_str(), to_string(ge::Shape(outDesc.GetShape())).c_str(),
          to_string(op.GetOutputDesc("output_data").GetShape()).c_str());
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(ParallelConcat, ParallelConcatVerify) {
  int64_t num;
  if (GRAPH_SUCCESS != op.GetAttr("N", num)) {
    std::string err_msg = GetInputInvalidErrMsg("N");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  } else {
    if (op.GetInputsSize() != static_cast<uint64_t>(num)) {
      string excepted_value = ConcatString("same as N[", static_cast<uint64_t>(num), "]");
      std::string err_msg = GetAttrValueErrMsg("values's size", ConcatString(op.GetInputsSize()), excepted_value);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(ParallelConcat, ParallelConcatInferShape);
VERIFY_FUNC_REG(ParallelConcat, ParallelConcatVerify);
// ----------------ParallelConcat OP End-------------------

// ----------------ConcatD OP Begin-------------------
IMPLEMT_COMMON_INFERFUNC(ConcatDInferShape) {
  const vector<string> depends;
  PREPARE_DYNAMIC_SHAPE(depends);

  int64_t num_concat;
  if (op.GetAttr("N", num_concat) == GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("N");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int64_t concat_dim;
  if (op.GetAttr("concat_dim", concat_dim) == GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("concat_dim");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  return ConcatInferShapeCommon(op, 0, num_concat, concat_dim, false);
}

COMMON_INFER_FUNC_REG(ConcatD, ConcatDInferShape);

IMPLEMT_INFER_DATA_SLICE(ConcatD, ConcatDInferDataSlice) {
  int64_t num_concat;
  if (op.GetAttr("N", num_concat) == GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("N");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int64_t concat_dim;
  if (op.GetAttr("concat_dim", concat_dim) == GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("concat_dim");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  return ConcatInferDataSliceCommon(op, num_concat, concat_dim);
}

INFER_DATA_SLICE_FUNC_REG(ConcatD, ConcatDInferDataSlice);
INFER_VALUE_RANGE_DEFAULT_REG(ConcatD);
// ----------------ConcatD OP End-------------------

// ----------------Concat OP Begin-------------------
IMPLEMT_COMMON_INFERFUNC(ConcatInferShape) {
  const vector<string> depend_names = {"concat_dim"};
  PREPARE_DYNAMIC_SHAPE(depend_names);

  int64_t N;
  if (op.GetAttr("N", N) == GRAPH_FAILED) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
                                       string("get attr[N] failed"));
    return GRAPH_FAILED;
  }

  Tensor data;
  bool is_unknown_axis = op.GetInputConstData("concat_dim", data) != GRAPH_SUCCESS;
  OP_LOGD(op.GetName().c_str(), "concat_dim is unknown[%s].", is_unknown_axis ? "true" : "false");
  int64_t axis = 0;
  if (!is_unknown_axis) {
    auto op_info = OpDescUtils::GetOpDescFromOperator(op);
    DataType dtype = op_info->MutableInputDesc("concat_dim")->GetDataType();
    std::vector<int64_t> const_vec;
    if (!GetConstValue(op, data, dtype, const_vec)) {
      is_unknown_axis = true;
      OP_LOGW(op.GetName().c_str(), "Get concat_dim value failed.");
    } else {
      axis = const_vec[0];
    }
  }

  return ConcatInferShapeCommon(op, 1, N, axis, is_unknown_axis);
}

COMMON_INFER_FUNC_REG(Concat, ConcatInferShape);
INFER_VALUE_RANGE_DEFAULT_REG(Concat);
// ----------------Concat OP End-------------------

// ----------------ConcatV2 OP Begin-------------------
IMPLEMT_COMMON_INFERFUNC(ConcatV2InferShape) {
  const vector<string> depend_names = {"concat_dim"};
  PREPARE_DYNAMIC_SHAPE(depend_names);

  int64_t N;
  if (op.GetAttr("N", N) == GRAPH_FAILED) {
    OP_LOGE(op.GetName().c_str(), "get attr N failed");
    OpsGetAttrErrReport(op.GetName(), "N");
    return GRAPH_FAILED;
  }

  Tensor data;
  bool is_unknown_axis = op.GetInputConstData("concat_dim", data) != GRAPH_SUCCESS;
  OP_LOGD(op.GetName().c_str(), "concat_dim is unknown[%s].", is_unknown_axis ? "true" : "false");
  int64_t axis = 0;
  if (!is_unknown_axis) {
    auto op_info = OpDescUtils::GetOpDescFromOperator(op);
    DataType dtype = op_info->MutableInputDesc("concat_dim")->GetDataType();
    std::vector<int64_t> const_vec;
    if (!GetConstValue(op, data, dtype, const_vec)) {
      is_unknown_axis = true;
      OP_LOGW(op.GetName().c_str(), "Get concat_dim value failed.");
    } else {
      axis = const_vec[0];
    }
  }

  return ConcatInferShapeCommon(op, 0, N, axis, is_unknown_axis);
}

COMMON_INFER_FUNC_REG(ConcatV2, ConcatV2InferShape);
INFER_VALUE_RANGE_DEFAULT_REG(ConcatV2);
// ----------------ConcatV2 OP End-------------------

// ----------------Pack OP Begin-------------------
bool JoinShapes(vector<int64_t>& dst_shape, const vector<int64_t>& src_shape) {
  if (dst_shape == src_shape) {
    return true;
  }

  if (dst_shape.empty() || IsUnknownRankShape(dst_shape)) {
    dst_shape = src_shape;
    return true;
  }

  if (!IsUnknownRankShape(src_shape)) {
    if (dst_shape.size() != src_shape.size()) {
      return false;
    }

    auto shape_dims = dst_shape.size();
    for (size_t i = 0; i < shape_dims; i++) {
      if (dst_shape[i] == src_shape[i]) {
        continue;
      }

      if (dst_shape[i] != UNKNOWN_DIM && src_shape[i] != UNKNOWN_DIM) {
        return false;
      }

      if (src_shape[i] != UNKNOWN_DIM) {
        dst_shape[i] = src_shape[i];
      }
    }
  }

  return true;
}

IMPLEMT_COMMON_INFERFUNC(PackInferShape) {
  int64_t pack_num;
  if (op.GetAttr("N", pack_num) == GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("get attr N failed");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (pack_num < 1) {
    string excepted_value = ConcatString("more than or equals to 1");
    std::string err_msg = GetAttrValueErrMsg("pack_num", std::to_string(pack_num), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int64_t axis;
  if (op.GetAttr("axis", axis) == GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("attr axis");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto ge_tensor_desc = op_info->MutableInputDesc(0);
  vector<int64_t> output_shape_dims;
  for (int64_t input_idx = 0; input_idx < pack_num; input_idx++) {
    auto input_shape_dims = op_info->MutableInputDesc(input_idx)->MutableShape().GetDims();
    if (!JoinShapes(output_shape_dims, input_shape_dims)) {
      std::string err_msg = OtherErrMsg( "the input shape dims should be equal");
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }

  // get the output y desc use output idx 0
  auto y_desc = op_info->MutableOutputDesc(0);
  DataType input_dtype = ge_tensor_desc->GetDataType();
  if (IsUnknownRankShape(output_shape_dims)) {
    y_desc->SetShape(ge::GeShape(UNKNOWN_RANK));
    y_desc->SetDataType(input_dtype);
    OP_LOGD(op.GetName().c_str(), "N:%lld, axis:%lld, output shape:%s.", pack_num, axis,
            to_string(y_desc->MutableShape()).c_str());
    return GRAPH_SUCCESS;
  }

  int64_t dim_num = static_cast<int64_t>(output_shape_dims.size());
  if (axis < (-dim_num - 1) || axis > dim_num) {
    string correct_value = ConcatString("in range [", -dim_num - 1, ", ", dim_num, "]");
    std::string err_msg = GetAttrValueErrMsg("axis", ConcatString(axis), correct_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (axis < 0) {
    axis += (dim_num + 1);
  }

  output_shape_dims.reserve(output_shape_dims.size() + 1);
  output_shape_dims.insert(output_shape_dims.begin() + axis, pack_num);
  GeShape x_shape(output_shape_dims);
  y_desc->SetShape(x_shape);
  y_desc->SetOriginShape(x_shape);
  y_desc->SetDataType(input_dtype);

  if (IsUnKnownShape(output_shape_dims)) {
    std::vector<std::pair<int64_t, int64_t>> y_range;
    for (int64_t input_idx = 0; input_idx < pack_num; input_idx++) {
      auto input_shape_dims = op_info->MutableInputDesc(input_idx)->MutableShape().GetDims();
      std::vector<std::pair<int64_t, int64_t>> shape_range;
      op_info->MutableInputDesc(input_idx)->GetShapeRange(shape_range);
      MakeUpShapeRange(input_shape_dims, shape_range);
      if (y_range.empty()) {
        y_range = shape_range;
      } else {
        JoinShapeRanges(y_range, shape_range);
      }
    }

    y_range.reserve(y_range.size() + 1);
    y_range.insert(y_range.begin() + axis, std::pair<int64_t, int64_t>{pack_num, pack_num});
    y_desc->SetShapeRange(y_range);
    OP_LOGD(op.GetName().c_str(), "output shape range:%s.", to_string(y_range).c_str());
  }

  auto p_context = op.GetInferenceContext();
  if (p_context != nullptr) {
    const auto& shapes_and_types = p_context->GetInputHandleShapesAndTypes();
    if (!shapes_and_types.empty()) {
      p_context->SetOutputHandleShapesAndTypes(shapes_and_types);
    }
  }

  OP_LOGD(op.GetName().c_str(), "N:%lld, axis:%lld, output shape:%s.", pack_num, axis,
          to_string(y_desc->MutableShape()).c_str());
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Pack, PackInferShape);
INFER_VALUE_RANGE_DEFAULT_REG(Pack);
// ----------------Pack OP End-------------------

// --------------------ConcatOffset------------------------
IMPLEMT_COMMON_INFERFUNC(ConcatOffsetInferShape) {
  // get attr N
  int num_concat;
  op.GetAttr("N", num_concat);
  if (num_concat < 2) {
    OP_LOGE(op.GetName().c_str(), "The num_concat should be no less than two");
    return GRAPH_FAILED;
  }
  // get the fisrt DynamicInput shape
  const uint32_t start_idx = 1;
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input1_desc = op_info->MutableInputDesc(1);
  DataType input_dtype = input1_desc->GetDataType();
  auto input1_shape = input1_desc->MutableShape().GetDims();
  if (!IsUnknown(input1_shape)) {
    for (auto i = 0; i < num_concat; i++) {
      auto output_desc = op_info->MutableOutputDesc(i);
      output_desc->SetShape(GeShape(input1_shape));
      output_desc->SetDataType(input_dtype);
    }
    return GRAPH_SUCCESS;
  }

  // dynamic shape, will get all inputs and calcu range
  vector<int64_t> dim_size = {};
  std::vector<std::pair<int64_t, int64_t>> input1_range;
  input1_desc->GetShapeRange(input1_range);
  for (auto i = 1; i < num_concat; i++) {
    auto input2_desc = op_info->MutableInputDesc(i + start_idx);
    auto input2_shape = input2_desc->MutableShape().GetDims();
    std::vector<std::pair<int64_t, int64_t>> input2_range;
    if (!IsUnknown(input2_shape)) {
      input1_shape = input2_shape;
      input1_range.clear();
      MakeUpShapeRange(input1_shape, input1_range);
      break;
    }
    input2_desc->GetShapeRange(input2_range);
    FixShapeRangeWithDims(dim_size, input1_shape, input2_shape, input1_range, input2_range);
  }

  for (auto i = 0; i < num_concat; i++) {
    auto output_desc = op_info->MutableOutputDesc(i);
    output_desc->SetShape(GeShape(input1_shape));
    output_desc->SetOriginShape(GeShape(input1_shape));
    output_desc->SetShapeRange(input1_range);
    output_desc->SetDataType(input_dtype);
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ConcatOffset, ConcatOffsetInferShape);
// --------------------ConcatOffset------------------------

// --------------------ConcatOffsetD Op Begin------------------------
IMPLEMT_COMMON_INFERFUNC(ConcatOffsetDInferShape) {
  DataType input_dtype = op.GetDynamicInputDesc("x", 0).GetDataType();
  Shape shape = op.GetDynamicInputDesc("x", 0).GetShape();
  auto tensordesc = op.GetDynamicInputDesc("x", 0);
  int num_concat;
  op.GetAttr("N", num_concat);
  if (num_concat < 2) {
      std::string err_msg = GetAttrSizeErrMsg("num_concat", std::to_string(num_concat), ConcatString("more than or equal to 2"));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
  }
  int64_t concat_dim;
  if (op.GetAttr("concat_dim", concat_dim) == GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("concat_dim");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  tensordesc.SetShape(shape);
  tensordesc.SetDataType(input_dtype);
  for (auto i = 0; i < num_concat; i++) {
    op.UpdateDynamicOutputDesc("y", i, tensordesc);
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ConcatOffsetD, ConcatOffsetDInferShape);
// --------------------ConcatOffsetD Op End------------------------

// --------------------Combinations Op Begin------------------------
static int64_t combi(int64_t input_num, int32_t r) {
  int ans = 1;
  for(int i = input_num; i > input_num - r; i--) ans *= i;
  for(int i = r; i > 1; i--) ans /= i;
  return ans;
}

IMPLEMT_COMMON_INFERFUNC(CombinationsInferShape) {
  Shape shape = op.GetInputDescByName("x").GetShape();
  DataType input_dtype = op.GetInputDescByName("x").GetDataType();
  std::vector<int64_t> dim_vector;
  int64_t dimsInput = shape.GetDimNum() - 1;
  int32_t r;
  if (op.GetAttr("r", r) == GRAPH_FAILED) {
    r = 2;
  }
  if (r < 1) {
      std::string err_msg = GetAttrSizeErrMsg("r", std::to_string(r), ConcatString("more than or equal to 1"));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
  }
  if (r > shape.GetShapeSize()) {
      std::string err_msg = GetAttrSizeErrMsg("r", std::to_string(r), ConcatString("less than or equal to input_num"));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
  }
  bool with_replacement;
  if (op.GetAttr("with_replacement", with_replacement) == GRAPH_FAILED) {
    with_replacement = false;
  }
  if (with_replacement == true) {
    dim_vector.push_back(combi(shape.GetShapeSize(), r) + shape.GetShapeSize());
  } else if (with_replacement == false) {
    dim_vector.push_back(combi(shape.GetShapeSize(), r));
  }
  dim_vector.push_back(r);
  Shape output_shape(dim_vector);
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(output_shape);
  td.SetDataType(input_dtype);
  if (op.UpdateOutputDesc("y", td) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Combinations, CombinationsVerify)
{
  if (op.GetInputDesc("x").GetDataType() != op.GetOutputDesc("y").GetDataType()) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Combinations, CombinationsInferShape);
VERIFY_FUNC_REG(Combinations, CombinationsVerify);
// --------------------Combinations Op End------------------------
}  // namespace ge

