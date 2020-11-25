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
#include "util/util.h"
#include "util/error_util.h"
#include "op_log.h"
#include "graph/utils/node_utils.h"

namespace ge {
// ----------------Split OP Begin-------------------
static void CalcSplit(const Tensor& data, const DataType& dtype, std::vector<int64_t>& const_vec) {
  const uint8_t* constData = data.GetData();
  size_t size = data.GetSize() / sizeof(int32_t);
  for (size_t i = 0; i < size; ++i) {
    const_vec.push_back(*((int32_t*)constData));
  }
}

IMPLEMT_COMMON_INFERFUNC(SplitInferShape) {
  const vector<string> depend_names = {"split_dim"};
  PREPARE_DYNAMIC_SHAPE(depend_names);

  OP_LOGD(op.GetName().c_str(), "SplitInferShape");
  auto x_desc = op.GetInputDesc("x");
  auto x_shape = x_desc.GetShape();
  auto x_dtype = x_desc.GetDataType();
  TensorDesc td = op.GetDynamicOutputDesc("y", 0);

  std::vector<std::pair<int64_t, int64_t>> x_shape_range;
  std::vector<std::pair<int64_t, int64_t>> out_range;
  x_desc.GetShapeRange(x_shape_range);
  OP_LOGD(op.GetName().c_str(), "SplitInferShape x_shape_range is %s", to_string(x_shape_range).c_str());

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
        out_range.push_back(std::pair<int64_t, int64_t>(1, x_shape_range[0].second));
      } else {
        out_range.push_back(std::pair<int64_t, int64_t>(1, ceil((float)x_shape_range[0].second / (float)num_split)));
      }
    } else {
      for(size_t i = 0; i < x_shape_range.size(); ++i) {
        x_shape.SetDim(i, -1);
        out_range.push_back(std::pair<int64_t, int64_t>(1, x_shape_range[i].second));
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
    OP_LOGE(op.GetName().c_str(), "get attr split dim failed");
    OpsGetAttrErrReport(op.GetName(), "split_dim");
    return GRAPH_FAILED;
  }

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
      OP_LOGD(op.GetName().c_str(), "SplitDInferShape output shape is %s", to_string(td.GetShape()).c_str());
      op.UpdateDynamicOutputDesc("y", i, td);
    }
    return GRAPH_SUCCESS;
  }

  // check attr
  int64_t dim_num = x_shape.GetDimNum();
  if ((split_dim < -dim_num) || (split_dim >= dim_num)) {
    OP_LOGE(op.GetName().c_str(), "Axis value out of range");
    OpsInputShapeDimErrReport(op.GetName(), "Axis", ConcatString(dim_num), ConcatString(-dim_num),
                              ConcatString(split_dim));
    return GRAPH_FAILED;
  }
  if (num_split < 1) {
    string excepted_value = ConcatString("in range[1,]");
    OP_LOGE(op.GetName().c_str(), "num_split need greater than or equals to 1");
    OpsAttrValueErrReport(op.GetName(), "num_split", excepted_value, ConcatString(num_split));
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

IMPLEMT_INFERFUNC(SplitV, SplitVInferShape) {
  auto tensordesc = op.GetInputDesc("x");
  auto shape = tensordesc.GetShape();
  DataType inputDtype = tensordesc.GetDataType();
  TensorDesc td = op.GetDynamicOutputDesc("y", 0);

  Tensor data2;
  if (op.GetInputConstData("split_dim", data2) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get constValue failed of [input_split_dim]");
    OpsMissInputErrReport(op.GetName(), "split_dim");
    return GRAPH_FAILED;
  }
  DataType dtype2 = op.GetInputDesc("split_dim").GetDataType();
  std::vector<int64_t> const_vec2;
  CalcSplitV(data2, dtype2, const_vec2);

  Tensor data1;
  if (op.GetInputConstData("size_splits", data1) != GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "Get constValue failed of [input_size_splits]");
    DataType dtype1 = op.GetInputDesc("input_size_splits").GetDataType();
    std::vector<int64_t> const_vec1;
    CalcSplitV(data1, dtype1, const_vec1);

    int64_t split_dim = const_vec2[0];
    std::vector<int64_t> size_splits(const_vec1);

    int64_t num_split;
    if (op.GetAttr("num_split", num_split) == GRAPH_FAILED) {
      OP_LOGE(op.GetName().c_str(), "get attr num_split failed");
      OpsGetAttrErrReport(op.GetName(), "num_split");
      return GRAPH_FAILED;
    }
    if (split_dim < 0) {
      split_dim += shape.GetDimNum();
    }

    for (auto i = 0; i < num_split; ++i) {
      shape.SetDim(split_dim, -1);
      td.SetShape(shape);
      td.SetDataType(inputDtype);
      op.UpdateDynamicOutputDesc("y", i, td);
    }
    return GRAPH_SUCCESS;
  } else {
    DataType dtype1 = op.GetInputDesc("size_splits").GetDataType();
    std::vector<int64_t> const_vec1;
    CalcSplitV(data1, dtype1, const_vec1);

    int64_t split_dim = const_vec2[0];
    std::vector<int64_t> size_splits(const_vec1);

    int64_t num_split;
    if (op.GetAttr("num_split", num_split) == GRAPH_FAILED) {
      OP_LOGE(op.GetName().c_str(), "get attr num_split failed");
      OpsGetAttrErrReport(op.GetName(), "num_split");
      return GRAPH_FAILED;
    }
    if (split_dim < 0) {
      split_dim += shape.GetDimNum();
    }

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
    }
    return GRAPH_SUCCESS;
  }
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
    OpsGetAttrErrReport(op.GetName(), "split_dim");
    OP_LOGE(op.GetName().c_str(), "get attr num_split failed");
  }
  vector<int64_t> size_splits;
  if (op.GetAttr("size_splits", size_splits) == GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "size_splits");
    OP_LOGE(op.GetName().c_str(), "get attr size_splits failed");
  }
  int64_t num_split;
  if (op.GetAttr("num_split", num_split) == GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "num_split");
    OP_LOGE(op.GetName().c_str(), "get attr num_split failed");
  }
  int64_t dim_num = shape.GetDimNum();
  if ((split_dim < -dim_num) || (split_dim >= dim_num)) {
    OpsInputShapeDimErrReport(op.GetName(), "Axis", ConcatString(dim_num), ConcatString(-dim_num),
                              ConcatString(split_dim));
    OP_LOGE(op.GetName().c_str(), "Axis value out of range");
    return GRAPH_FAILED;
  }
  if (num_split < 1) {
    string excepted_value = ConcatString("in range[1,]");
    OpsAttrValueErrReport(op.GetName(), "num_split", excepted_value, ConcatString(num_split));
    OP_LOGE(op.GetName().c_str(), "num_split need greater than or equals to 1");
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
      OP_LOGE(op.GetName().c_str(), "dimvalue should be divisible by num_split");
      OpsInputShapeErrReport(op.GetName(), "dim_value should be divisible by num_split",
                             "dim % num_split", ConcatString(dim % num_split));
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

static graphStatus ConcatInferShapeCommon(Operator& op, int64_t num_concat, int64_t axis) {
  auto tensordesc = op.GetDynamicInputDesc("x", 0);
  auto shape = tensordesc.GetShape();
  int64_t dim_num = shape.GetDimNum();

  vector<int64_t> shape_list;
  vector<std::set<int64_t>> dim_sets(dim_num);
  for (int32_t i = 0; i < num_concat; i++) {
    shape_list = op.GetDynamicInputDesc("x", i).GetShape().GetDims();
    for (int32_t j = 0; j < dim_num; j++) {
      dim_sets[j].insert(shape_list[j]);
      shape.SetDim(j, std::max(shape.GetDim(j), shape_list[j]));
    }
  }

  for (int32_t j = 0; j < dim_num; j++) {
    if ((axis != j)) {
      dim_sets[j].erase(-1);
      if (dim_sets[j].size() <= 1) {
        continue;
      }

      map<string, string> err_map = {
          {"opname", op.GetName()},
          {"err_msg", "All axes must be equal except merge axis,check your shape!"},
      };

      std::string report_error_code = "E35003";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  }

  int32_t size = 0;
  vector<pair<int64_t, int64_t>> input_shape_ranges;
  vector<pair<int64_t, int64_t>> output_shape_ranges;
  pair<int64_t, int64_t> output_concat_dim_range(0, 0);
  bool has_shape_ranges = false;
  for (int32_t i = 0; i < num_concat; i++) {
    const auto input_desc = op.GetDynamicInputDesc("x", i);
    auto dim_value = input_desc.GetShape().GetDim(axis);
    if (dim_value == -1) {
      size = -1;
    } else if (size != -1) {
      size += dim_value;
    }

    (void)input_desc.GetShapeRange(input_shape_ranges);
    OP_LOGD(op.GetName().c_str(), "input shape range:%s", to_string(input_shape_ranges).c_str());
    if (input_shape_ranges.empty()) {
      auto shape_dims = input_desc.GetShape().GetDims();
      MakeUpShapeRange(shape_dims, input_shape_ranges);

      if (dim_value > 0) {
        output_concat_dim_range.first += dim_value;
        output_concat_dim_range.second += dim_value;
      }
    } else {
      has_shape_ranges = true;
      output_concat_dim_range.first += input_shape_ranges[axis].first;
      if (input_shape_ranges[axis].second == -1 || output_concat_dim_range.second == -1) {
        output_concat_dim_range.second = -1;
      } else {
        output_concat_dim_range.second += input_shape_ranges[axis].second;
      }
    }

    if (i == 0) {
      output_shape_ranges = input_shape_ranges;
    } else {
      JoinShapeRanges(output_shape_ranges, input_shape_ranges);
    }
  }
  shape.SetDim(axis, size);
  auto first_input_desc = op.GetDynamicInputDesc("x", 0);
  DataType input_dtype = first_input_desc.GetDataType();

  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(ge::Shape(shape));
  OP_LOGD(op.GetName().c_str(), "output shape:%s", to_string(td.GetShape()).c_str());
  td.SetDataType(input_dtype);

  if (has_shape_ranges && IsUnKnownShape(shape.GetDims()) && static_cast<int>(output_shape_ranges.size()) > axis) {
    output_shape_ranges[axis] = output_concat_dim_range;
    (void)td.SetShapeRange(output_shape_ranges);
    OP_LOGD(op.GetName().c_str(), "output shape range:%s", to_string(output_shape_ranges).c_str());
  }

  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ConcatV2DInferShape) {
  const vector<string> depends;
  PREPARE_DYNAMIC_SHAPE(depends);
  auto tensordesc = op.GetDynamicInputDesc("x", 0);
  int64_t axis;
  int64_t num_concatext2;
  if (op.GetAttr("concat_dim", axis) == GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "axis");
    OP_LOGE(op.GetName().c_str(), "get attr axis failed");
  }
  if (op.GetAttr("N", num_concatext2) == GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "N");
    OP_LOGE(op.GetName().c_str(), "get attr N failed");
  }

  for (int32_t i = 0; i < num_concatext2; i++) {
    const auto input_desc = op.GetDynamicInputDesc("x", i);
    if (input_desc.GetShape().GetDims() == UNKNOWN_RANK) {
      DataType input_dtype = input_desc.GetDataType();

      TensorDesc td = op.GetOutputDesc("y");
      td.SetShape(ge::Shape(UNKNOWN_RANK));
      OP_LOGD(op.GetName().c_str(), "output shape:%s", to_string(td.GetShape()).c_str());
      td.SetDataType(input_dtype);

      (void)op.UpdateOutputDesc("y", td);
      return GRAPH_SUCCESS;
    }
  }

  auto axis1 = axis;
  auto shape = tensordesc.GetShape();
  int64_t dim_num = shape.GetDimNum();
  if ((axis1 < -dim_num) || (axis1 >= dim_num)) {
    OpsInputShapeDimErrReport(op.GetName(), "axis", ConcatString(dim_num), ConcatString(-dim_num), ConcatString(axis1));
    OP_LOGE(op.GetName().c_str(), "Axis value out of range");
    return GRAPH_FAILED;
  }
  if (axis1 < 0) {
    axis1 += shape.GetDimNum();
  }

  return ConcatInferShapeCommon(op, num_concatext2, axis1);
}

COMMON_INFER_FUNC_REG(ConcatV2D, ConcatV2DInferShape);
// ----------------ConcatV2D OP End-------------------

// ----------------ParallelConcat OP Begin-------------------
static std::vector<int64_t> GetAttrValue(const ge::Operator& op, const std::string& key_name) {
  std::vector<int64_t> list;
  if (ge::GRAPH_SUCCESS != op.GetAttr(key_name, list)) {
    OpsGetAttrErrReport(op.GetName(), key_name);
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

IMPLEMT_COMMON_INFERFUNC(ParallelConcatInferShape) {
  auto tensordesc = op.GetDynamicInputDesc("values", 0);
  std::vector<int64_t> shape;
  shape = GetAttrValue(op, "shape");
  int64_t num_1;
  if (!CheckListEmpty(op.GetName(), shape, "shape")) {
    return GRAPH_FAILED;
  }

  if (GRAPH_SUCCESS != op.GetAttr("N", num_1)) {
    OpsGetAttrErrReport(op.GetName(), "N");
    OP_LOGE(op.GetName().c_str(), "GetAttr of N failed.");
    return GRAPH_FAILED;
  }
  auto x_shape = tensordesc.GetShape();
  int64_t dimnum;
  dimnum = x_shape.GetDimNum();
  if (shape[0] != num_1) {
    string excepted_value = ConcatString("equal to the num of N[", num_1, "]");
    OpsAttrValueErrReport(op.GetName(), "output_data's fisrt dim", excepted_value, ConcatString(shape[0]));
    OP_LOGE(op.GetName().c_str(),
            "first dim of output shape must"
            "be equal to the num of input tensors.");
    return GRAPH_FAILED;
  }
  for (int64_t i = 1; i < dimnum; i++) {
    if (x_shape.GetDim(i) != shape[i]) {
      string excepted_value = ConcatString("match the output_data's shape[", shape[i], "]");
      OpsAttrValueErrReport(op.GetName(), "values's shape", excepted_value, ConcatString(x_shape.GetDim(i)));
      OP_LOGE(op.GetName().c_str(),
              "the input shape"
              "do not match the output shape.");
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
    OpsGetAttrErrReport(op.GetName(), "N");
    OP_LOGE(op.GetName().c_str(), "GetAttr of N failed.");
    return GRAPH_FAILED;
  } else {
    if (op.GetInputsSize() != static_cast<uint64_t>(num)) {
      string excepted_value = ConcatString("same as N[", static_cast<uint64_t>(num), "]");
      OpsAttrValueErrReport(op.GetName(), "values's size", excepted_value, ConcatString(op.GetInputsSize()));
      OP_LOGE(op.GetName().c_str(), "input size and N must be same.");
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
  auto tensordesc = op.GetDynamicInputDesc("x", 0);
  auto shape = tensordesc.GetShape();
  int64_t concat_dim;
  int64_t num_concat;
  if (op.GetAttr("concat_dim", concat_dim) == GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "concat_dim");
    OP_LOGE(op.GetName().c_str(), "get attr concat_dim failed");
  }
  if (op.GetAttr("N", num_concat) == GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "N");
    OP_LOGE(op.GetName().c_str(), "get attr N failed");
  }

  for (int32_t i = 0; i < num_concat; i++) {
    const auto input_desc = op.GetDynamicInputDesc("x", i);
    if (input_desc.GetShape().GetDims() == UNKNOWN_RANK) {
      DataType input_dtype = input_desc.GetDataType();

      TensorDesc td = op.GetOutputDesc("y");
      td.SetShape(ge::Shape(UNKNOWN_RANK));
      OP_LOGD(op.GetName().c_str(), "output shape:%s", to_string(td.GetShape()).c_str());
      td.SetDataType(input_dtype);

      (void)op.UpdateOutputDesc("y", td);
      return GRAPH_SUCCESS;
    }
  }

  auto axis = concat_dim;
  int64_t dim_num = shape.GetDimNum();
  if (axis < -dim_num || axis >= dim_num) {
    OpsInputShapeDimErrReport(op.GetName(), "axis", ConcatString(dim_num), ConcatString(-dim_num), ConcatString(axis));
    OP_LOGE(op.GetName().c_str(), "Axis value[%lld] out of range(%lld, %lld]", axis, -dim_num, dim_num);
    return GRAPH_FAILED;
  }
  if (axis < 0) {
    axis += shape.GetDimNum();
  }

  return ConcatInferShapeCommon(op, num_concat, axis);
}

COMMON_INFER_FUNC_REG(ConcatD, ConcatDInferShape);
// ----------------ConcatD OP End-------------------

// ----------------Concat OP Begin-------------------
static void CalcConcat(const Tensor& data, const DataType& dtype, std::vector<int64_t>& const_vec) {
  const uint8_t* constData = data.GetData();
  if (dtype == ge::DT_INT32) {
    const_vec.push_back(*((int32_t*)constData));
  } else {
    const_vec.push_back(*((int64_t*)constData));
  }
}

IMPLEMT_COMMON_INFERFUNC(ConcatInferShape) {
  const vector<string> depend_names = {"concat_dim"};
  PREPARE_DYNAMIC_SHAPE(depend_names);

  int64_t N;
  if (op.GetAttr("N", N) == GRAPH_FAILED) {
    OP_LOGE(op.GetName().c_str(), "get attr N failed");
    OpsGetAttrErrReport(op.GetName(), "N");
  }

  for (int32_t i = 0; i < N; i++) {
    const auto input_desc = op.GetDynamicInputDesc("x", i);
    if (input_desc.GetShape().GetDims() == UNKNOWN_RANK) {
      DataType input_dtype = input_desc.GetDataType();

      TensorDesc td = op.GetOutputDesc("y");
      td.SetShape(ge::Shape(UNKNOWN_RANK));
      OP_LOGD(op.GetName().c_str(), "output shape:%s", to_string(td.GetShape()).c_str());
      td.SetDataType(input_dtype);

      (void) op.UpdateOutputDesc("y", td);
      return GRAPH_SUCCESS;
    }
  }

  auto tensordesc = op.GetDynamicInputDesc("x", 0);
  auto shape = tensordesc.GetShape();
  int64_t dim_num;
  dim_num = shape.GetDimNum();

  Tensor data;
  if (op.GetInputConstData("concat_dim", data) != GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "Get constValue failed of [concat_dim]");
    vector<int64_t> dimVector(dim_num, -1);
    Shape x_shape(dimVector);
    TensorDesc y_desc = op.GetOutputDesc("output_data");
    y_desc.SetShape(ge::Shape(x_shape));
    DataType input_dtype = tensordesc.GetDataType();
    y_desc.SetDataType(input_dtype);
    auto output_shape_ranges = GetShapeRangesWithUnKnowConcatDim(op, N);
    if (!output_shape_ranges.empty()) {
      y_desc.SetShapeRange(output_shape_ranges);
      OP_LOGD(op.GetName().c_str(), "output shape range:%s", to_string(output_shape_ranges).c_str());
    }
    (void) op.UpdateOutputDesc("y", y_desc);
    return GRAPH_SUCCESS;
  }

  DataType dtype = op.GetInputDesc("concat_dim").GetDataType();
  std::vector<int64_t> const_vec;
  CalcConcat(data, dtype, const_vec);
  int64_t concat_dim = const_vec[0];
  int64_t axis = concat_dim;
  if (axis < 0) {
    axis += shape.GetDimNum();
  }

  return ConcatInferShapeCommon(op, N, axis);
}

COMMON_INFER_FUNC_REG(Concat, ConcatInferShape);
// ----------------Concat OP End-------------------

// ----------------ConcatV2 OP Begin-------------------
static void CalcConcatv2(const Tensor& data, const DataType& dtype, std::vector<int64_t>& const_vec) {
  const uint8_t* constData = data.GetData();
  if (dtype == ge::DT_INT32) {
    const_vec.push_back((int64_t)(*((int32_t*)constData)));
  } else {
    const_vec.push_back(*((int64_t*)constData));
  }
}

IMPLEMT_COMMON_INFERFUNC(ConcatV2InferShape) {
  const vector<string> depend_names = {"concat_dim"};
  PREPARE_DYNAMIC_SHAPE(depend_names);

  int64_t N;
  if (op.GetAttr("N", N) == GRAPH_FAILED) {
    OP_LOGE(op.GetName().c_str(), "get attr N failed");
    OpsGetAttrErrReport(op.GetName(), "N");
  }

  for (int32_t i = 0; i < N; i++) {
    const auto input_desc = op.GetDynamicInputDesc("x", i);
    if (input_desc.GetShape().GetDims() == UNKNOWN_RANK) {
      DataType input_dtype = input_desc.GetDataType();

      TensorDesc td = op.GetOutputDesc("y");
      td.SetShape(ge::Shape(UNKNOWN_RANK));
      OP_LOGD(op.GetName().c_str(), "output shape:%s", to_string(td.GetShape()).c_str());
      td.SetDataType(input_dtype);

      (void) op.UpdateOutputDesc("y", td);
      return GRAPH_SUCCESS;
    }
  }

  auto tensordesc = op.GetDynamicInputDesc("x", 0);
  auto shape = tensordesc.GetShape();
  int64_t dim_num;
  dim_num = shape.GetDimNum();
  Tensor data;
  if (op.GetInputConstData("concat_dim", data) != GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "Get constValue failed of [concat_dim]");
    vector<int64_t> dimVector(dim_num, -1);
    Shape x_shape(dimVector);
    TensorDesc y_desc = op.GetOutputDesc("output_data");
    y_desc.SetShape(ge::Shape(x_shape));
    DataType input_dtype = tensordesc.GetDataType();
    y_desc.SetDataType(input_dtype);
    auto output_shape_ranges = GetShapeRangesWithUnKnowConcatDim(op, N);
    if (!output_shape_ranges.empty()) {
      y_desc.SetShapeRange(output_shape_ranges);
      OP_LOGD(op.GetName().c_str(), "output shape range:%s", to_string(output_shape_ranges).c_str());
    }

    (void) op.UpdateOutputDesc("y", y_desc);
    return GRAPH_SUCCESS;
  }
  DataType dtype = op.GetInputDesc("concat_dim").GetDataType();
  std::vector<int64_t> const_vec;
  CalcConcatv2(data, dtype, const_vec);
  int64_t axis = const_vec[0];
  int64_t axis1 = axis;
  if (axis1 < 0) {
    axis1 += shape.GetDimNum();
  }

  return ConcatInferShapeCommon(op, N, axis1);
}

COMMON_INFER_FUNC_REG(ConcatV2, ConcatV2InferShape);
// ----------------ConcatV2 OP End-------------------

// ----------------Pack OP Begin-------------------
IMPLEMT_COMMON_INFERFUNC(PackInferShape) {
  PREPARE_DYNAMIC_SHAPE_WITH_NO_DEPENDS();
  auto ge_tensor_desc = op.GetDynamicInputDesc("x", 0);
  auto shape = ge_tensor_desc.GetShape();
  DataType input_dtype = ge_tensor_desc.GetDataType();
  TensorDesc y_desc = op.GetOutputDesc("y");

  int64_t dimnum;
  dimnum = shape.GetDimNum();
  int64_t axis;
  int64_t pack_num;
  if (op.GetAttr("axis", axis) == GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "axis");
    OP_LOGE(op.GetName().c_str(), "get attr axis failed");
  }
  if (op.GetAttr("N", pack_num) == GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "N");
    OP_LOGE(op.GetName().c_str(), "get attr N failed");
  }
  if (pack_num < 1) {
    OpsAttrValueErrReport(op.GetName(), "N", "greater than or equals to 1", ConcatString(pack_num));
    OP_LOGE(op.GetName().c_str(), "N is out of range");
  }
  if (axis < (-dimnum - 1) || axis > dimnum) {
    string correct_value = ConcatString("in range [", -dimnum - 1, ", ", dimnum, "]");
    AttrValueErrReport("axis", op.GetName(), ConcatString(axis), correct_value);
    OP_LOGE(op.GetName().c_str(), "attr axis is not in range");
    return GRAPH_FAILED;
  }
  if (axis < 0) {
    axis += (dimnum + 1);
  }

  // check unkown_rank and set unkown_rank
  auto shape_dims = ge_tensor_desc.GetShape().GetDims();
  bool is_unkown_rank = shape_dims == UNKNOWN_RANK ? true : false;
  if (is_unkown_rank) {
    y_desc.SetShape(ge::Shape(UNKNOWN_RANK));
    y_desc.SetDataType(input_dtype);
    (void)op.UpdateOutputDesc("y", y_desc);
    return GRAPH_SUCCESS;
  }

  std::vector<std::pair<int64_t, int64_t>> x_range;
  ge_tensor_desc.GetShapeRange(x_range);
  MakeUpShapeRange(shape_dims, x_range);
  std::vector<std::pair<int64_t, int64_t>> y_range;
  vector<int64_t> dimVector;
  for (int64_t i = 0; i < dimnum + 1; i++) {
    if (i < axis) {
      dimVector.push_back(shape.GetDim(i));
      y_range.push_back(x_range[i]);
    } else if (i == axis) {
      dimVector.push_back(pack_num);
      y_range.push_back(std::pair<int64_t, int64_t>{pack_num, pack_num});
    } else {
      dimVector.push_back(shape.GetDim(i - 1));
      y_range.push_back(x_range[i - 1]);
    }
  }
  Shape x_shape(dimVector);

  y_desc.SetShape(ge::Shape(x_shape));
  y_desc.SetOriginShape(ge::Shape(x_shape));
  y_desc.SetDataType(input_dtype);
  y_desc.SetShapeRange(y_range);
  (void)op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Pack, PackInferShape);
// ----------------Pack OP End-------------------

// --------------------ConcatOffset------------------------
IMPLEMT_COMMON_INFERFUNC(ConcatOffsetInferShape) {
  PREPARE_DYNAMIC_SHAPE_WITH_NO_DEPENDS();
  auto x_0_tensordesc = op.GetDynamicInputDesc("x",0);
  int num_concat;
  op.GetAttr("N", num_concat);
  if (num_concat < 2) {
    OP_LOGE(op.GetName().c_str(), "The num_concat should be no less than two");
    return GRAPH_FAILED;
  }

  std::vector<std::pair<int64_t, int64_t>> x_shape_range;
  std::vector<std::pair<int64_t, int64_t>> y_shape_range;
  vector<vector<std::pair<int64_t, int64_t>>> all_input_shape_ranges;
  bool has_shape_ranges = false;

  //unknown_rank
  bool is_unknown_rank = x_0_tensordesc.GetShape().GetDims() == UNKNOWN_RANK ? true : false;
  if(is_unknown_rank){
    y_shape_range.push_back(std::pair<int64_t,int64_t>{1,-1});
    for(auto i = 0; i < num_concat; i++){
      auto y_desc = op.GetDynamicOutputDesc("y",i);
      std::vector<int64_t> oShapeVector;
      oShapeVector.push_back(-1);
      Shape oShape(oShapeVector);
      y_desc.SetShape(oShape);
      y_desc.SetDataType(x_0_tensordesc.GetDataType());
      y_desc.SetShapeRange(y_shape_range);
      op.UpdateDynamicOutputDesc("y", i, y_desc);
    }
    return GRAPH_SUCCESS;
  }

  for (int32_t i = 0; i < num_concat; i++) {
    auto x_desc = op.GetDynamicInputDesc("x",i);
    x_desc.GetShapeRange(x_shape_range);
    if(x_shape_range.empty()){
      auto shape_dims = x_desc.GetShape().GetDims();
      MakeUpShapeRange(shape_dims,x_shape_range);
    }else{
      has_shape_ranges = true;
    }
    all_input_shape_ranges.push_back(x_shape_range);
  }

  if (has_shape_ranges){
    y_shape_range = all_input_shape_ranges[0];
    for(size_t i = 1; i < all_input_shape_ranges.size(); i++){
      for(size_t j = 0; j < y_shape_range.size(); j++){
        y_shape_range[j].first = std::max(y_shape_range[j].first, all_input_shape_ranges[i][j].first);
        if(all_input_shape_ranges[i][j].second == -1){
          continue;
        }else if (y_shape_range[j].second == -1){
          y_shape_range[j].second = all_input_shape_ranges[i][j].second;
        }else{
          y_shape_range[j].second = std::min(y_shape_range[j].second, all_input_shape_ranges[i][j].second);
        }
      }
    }
  }
  for (auto i = 0; i < num_concat; i++) {
    auto y_desc = op.GetDynamicOutputDesc("y",i);
    auto x_desc = op.GetDynamicInputDesc("x",i);
    y_desc.SetShapeRange(y_shape_range);
    y_desc.SetDataType(x_desc.GetDataType());
    if(y_shape_range[0].first == y_shape_range[0].second){
      vector<int64_t> dimVector;
      dimVector.push_back(y_shape_range[0].first);
      ge::Shape y_shape(dimVector);
      y_desc.SetShape(y_shape);
    }else{
      y_desc.SetShape(x_desc.GetShape());
    }
    op.UpdateDynamicOutputDesc("y", i, y_desc);
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
    OpsAttrValueErrReport(op.GetName(), "num_concat", "no less than two", ConcatString(num_concat));
    OP_LOGE(op.GetName().c_str(), "The num_concat should be no less than two");
    return GRAPH_FAILED;
  }
  int64_t concat_dim;
  if (op.GetAttr("concat_dim", concat_dim) == GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "concat_dim");
    OP_LOGE(op.GetName().c_str(), "get attr concat_dim failed");
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
}  // namespace ge
