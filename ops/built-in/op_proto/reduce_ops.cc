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
 * \file reduce_ops.cpp
 * \brief
 */
#include "inc/reduce_ops.h"
#include <string>
#include <vector>
#include "util/util.h"
#include "op_log.h"
#include "./util/error_util.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_log.h"
#include <chrono>

namespace ge {
using std::string;
const static bool prof_switch = std::getenv("REDUCE_INFER_PROF") != nullptr;

// Obtains the value of the constant tensor.
static void GetAllConstValue(const Tensor& data, std::vector<int64_t>& const_vec, ge::DataType axisType) {
  const uint8_t* constData = data.GetData();
  if (axisType == ge::DT_INT32) {
    size_t size = data.GetSize() / sizeof(int32_t);

    for (size_t i = 0; i < size; ++i) {
      const_vec.push_back((int64_t)(*((int32_t*)constData + i)));
    }
  } else if (axisType == ge::DT_INT64) {
    size_t size = data.GetSize() / sizeof(int64_t);

    for (size_t i = 0; i < size; ++i) {
      const_vec.push_back((int64_t)(*((int64_t*)constData + i)));
    }
  }
}

static bool InferReduceShape(const ge::Operator& op, const string& input_name, const string& axis_name,
                             const string& keep_dims_name, ge::TensorDesc& result_desc) {
  // indicates that GE should process related attributes during online infer shape
  vector<string> input_infer_depends = {"axes"};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends(input_infer_depends);

  result_desc = op.GetInputDesc(input_name);
  auto shape = result_desc.GetShape();
  std::vector<int64_t> shapeVector = shape.GetDims();
  int64_t dim_num = shape.GetDimNum();

  if (shapeVector.size() == 0) {
    OP_LOGI(op.GetName().c_str(), "input shape vector size is 0, is scalar.");
    result_desc.SetShape({});
    result_desc.SetShapeRange({});
    return true;
  }

  if (shapeVector[0] == -2) {
    std::vector<int64_t> oShapeVector;
    oShapeVector.push_back(-2);
    Shape oShape(oShapeVector);
    result_desc.SetShape(oShape);
    result_desc.SetShapeRange({});
    return true;
  }

  std::vector<std::pair<int64_t, int64_t>> input_shape_range;
  op_desc->MutableInputDesc(input_name)->GetShapeRange(input_shape_range);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  MakeUpShapeRange(shapeVector, input_shape_range);
  if (input_shape_range.size() != (uint32_t)dim_num) {
    OP_LOGI(op.GetName().c_str(), "reset input shape range.");
    input_shape_range.clear();
    MakeUpShapeRange(shapeVector, input_shape_range);
  }

  bool keep_dims;
  if (GRAPH_SUCCESS != op.GetAttr(keep_dims_name, keep_dims)) {
    OP_LOGE(op.GetName().c_str(), "GetAttr of %s failed.", keep_dims_name.c_str());
    return false;
  }

  ge::TensorDesc axis_desc;
  axis_desc = op.GetInputDesc(axis_name);
  auto axis_shape = axis_desc.GetShape();
  auto axis_type = axis_desc.GetDataType();
  std::vector<int64_t> axis_shapeVector = axis_shape.GetDims();
  int64_t axis_dimNum = axis_shape.GetDimNum();

  if (!axis_shapeVector.empty() && axis_shapeVector[0] > dim_num) {
    OP_LOGE(op.GetName().c_str(), "The size of axisnode must be less than inputx dim_num.");
    return false;
  }

  if (axis_dimNum == 1 && axis_shapeVector[0] == 0) {
    result_desc.SetShape(shape);
    result_desc.SetShapeRange(input_shape_range);
    OP_LOGI(op.GetName().c_str(), "axis dim num is 1 and axis shape vector[0] is 0.");
    return true;
  }

  Tensor data;
  // axis unknown
  if (GRAPH_SUCCESS != op.GetInputConstData(axis_name, data)) {
    OP_LOGI(op.GetName().c_str(), "GetInputConstData of %s failed, enter axis unknown scenario.", axis_name.c_str());

    std::vector<int64_t> oShapeVector;

    if (axis_dimNum > 1) {
      OP_LOGE(op.GetName().c_str(), "The dim number of axis must be one or zero, but actual is %d.", axis_dimNum);
      return false;
    }

    if (keep_dims) {
      for (int64_t item = 0; item < dim_num; ++item) {
        int64_t range_min_value = 1;
        int64_t range_max_value = input_shape_range[item].second;
        output_shape_range.push_back(std::make_pair(range_min_value, range_max_value));

        if (range_max_value == 1) {
          oShapeVector.push_back(1);
        } else {
          oShapeVector.push_back(-1);
        }
      }

      Shape oShape(oShapeVector);
      result_desc.SetShape(oShape);
      result_desc.SetShapeRange(output_shape_range);
    } else {
      if (!axis_shapeVector.empty() && (axis_shapeVector[0] == -1 || axis_shapeVector[0] == -2)) {
        OP_LOGI(op.GetName().c_str(), "Can't get reduce axis number.");

        oShapeVector.push_back(-2);
        Shape oShape(oShapeVector);
        result_desc.SetShape(oShape);
        result_desc.SetShapeRange({});
      } else {
        int64_t output_dimNum = 0;
        if (axis_dimNum == 0) {
          output_dimNum = dim_num - 1;
        } else {
          output_dimNum = dim_num - axis_shapeVector[0];
        }
        OP_LOGI(op.GetName().c_str(), "Get output dim num %d.", output_dimNum);

        int64_t range_min_value = input_shape_range[0].first;
        int64_t range_max_value = input_shape_range[0].second;
        for (uint32_t item = 0; item < shapeVector.size(); ++item) {
          if (input_shape_range[item].first < range_min_value) {
            range_min_value = input_shape_range[item].first;
          }

          if (input_shape_range[item].second == -1) {
            range_max_value = -1;
          }
          if (range_max_value != -1 && input_shape_range[item].second > range_max_value) {
            range_max_value = input_shape_range[item].second;
          }
        }

        for (int64_t item = 0; item < output_dimNum; ++item) {
          oShapeVector.push_back(-1);
          output_shape_range.push_back(std::make_pair(range_min_value, range_max_value));
        }

        Shape oShape(oShapeVector);
        result_desc.SetShape(oShape);
        result_desc.SetShapeRange(output_shape_range);
      }
    }

    // axis known
  } else {
    std::vector<int64_t> axis{};
    size_t size = data.GetSize();
    if (size != 0) {
      GetAllConstValue(data, axis, axis_type);
    }

    // reduce axis is empty, reduce all
    if (axis.size() == 0) {
      for (size_t i = 0; i < shapeVector.size(); ++i) {
        axis.push_back(i);
      }
    }

    // convert reduce axis
    for (size_t i = 0; i < axis.size(); ++i) {
      if (axis[i] < -dim_num || axis[i] > (dim_num - 1)) {
        OP_LOGE(op.GetName().c_str(), "reduce verify failed, axis: %d, dim_num:%d.", axis[i], dim_num);
        return false;
      }
      if (axis[i] < 0) {
        axis[i] = dim_num + axis[i];
      }
    }

    std::vector<int64_t> oShapeVector;
    std::vector<int64_t>::iterator tmp;
    for (int64_t item = 0; item < dim_num; ++item) {
      tmp = std::find(axis.begin(), axis.end(), item);
      if (tmp != axis.end()) {
        // item in axis
        if (keep_dims) {
          // If keepDims is true, current dimesion set to 1
          oShapeVector.push_back(1);
          output_shape_range.push_back(std::make_pair(1, 1));
        }
      } else {
        // item is not in ConstValueAxis
        oShapeVector.push_back(shapeVector[item]);
        output_shape_range.push_back(input_shape_range[item]);
      }
    }

    // clear output shape range during static shape
    bool is_static_shape = true;
    for (uint32_t i = 0; i < shapeVector.size(); ++i) {
      if (shapeVector[i] == -1) {
        is_static_shape = false;
        break;
      }
    }
    if (is_static_shape) {
      output_shape_range.clear();
    }

    Shape oShape(oShapeVector);
    result_desc.SetShape(oShape);
    result_desc.SetShapeRange(output_shape_range);
  }

  return true;
}

static bool CheckReduceInfo(const ge::Operator& op, const size_t& input_size, const size_t& axis_size,
                            const string& keep_dims_name, bool& keep_dims) {
  if (GRAPH_SUCCESS != op.GetAttr(keep_dims_name, keep_dims)) {
    OP_LOGE(op.GetName().c_str(), "GetAttr of %s failed.", keep_dims_name.c_str());
    return false;
  }
  return true;
}

static bool CheckReduceDInfo(const ge::Operator& op, const size_t& input_size, const string& keep_dims_name,
                             const string& axis_name, bool& keep_dims, std::vector<int64_t>& axis) {
  if (GRAPH_SUCCESS != op.GetAttr(keep_dims_name, keep_dims)) {
    OP_LOGE(op.GetName().c_str(), "GetAttr of %s failed.", keep_dims_name.c_str());
    return false;
  }
  if (GRAPH_SUCCESS != op.GetAttr(axis_name, axis)) {
    OpsGetAttrErrReport(op.GetName(), axis_name);
    OP_LOGE(op.GetName().c_str(), "GetAttr of %s failed.", axis_name.c_str());
    return false;
  }
  if (axis.size() > input_size) {
    OP_LOGE(op.GetName().c_str(), "size of axis is illegal.");
    return false;
  }

  return true;
}

template <typename T>
static void GetTensorValue(const GeTensorPtr& data, std::vector<int64_t>& vec_dim) {
  int32_t size = data->GetData().GetSize() / sizeof(T);
  void* data_ptr = (void*)data->GetData().GetData();
  if (data_ptr == nullptr) {
    return;
  }
  for (int32_t i = 0; i < size; i++) {
    T dim = *((T*)data_ptr + i);
    vec_dim.push_back((int64_t)dim);
  }
}

static bool ConvertAxis(std::vector<int64_t>& axis, int64_t input_length) {
  // Convert reduce axis
  for (size_t i = 0; i < axis.size(); ++i) {
    if (axis[i] < -input_length || axis[i] > (input_length - 1)) {
      OP_LOGE("Op[Reduce]", "reduce verify failed, axis: %d, input_length: %d", axis[i], input_length);
      return false;
    }
    if (axis[i] < 0) {
      axis[i] = input_length + axis[i];
    }
  }
  // All Reduce
  if (axis.size() == 0) {
    for (size_t i = 0; i < (size_t)input_length; ++i) {
      axis.push_back(i);
    }
  }
  return true;
}

static void DoKnownBranch(const bool& keep_dims, const DataType& type, std::vector<int64_t>& input_shape,
                          std::vector<int64_t>& axis_value, GeTensorDescPtr& output_desc) {
  /* Work In Situations:
   * 1. runtime for dynamic
   * 2. const case for dynamic
   * 3. static case
   * Don't set range in the branch
   * */
  size_t length = input_shape.size();
  std::vector<int64_t> reduce_flag(length);
  for (auto item : axis_value) {
    reduce_flag[item] = 1;
  }

  std::vector<int64_t> output_shape(length);
  if (keep_dims) {
    for (size_t idx = 0; idx < length; ++idx) {
      output_shape[idx] = reduce_flag[idx] == 1 ? 1 : input_shape[idx];
    }
  } else {
    size_t i0 = 0;
    for (size_t idx = 0; idx < length; ++idx) {
      if (reduce_flag[idx] == 0) {
        output_shape[i0] = input_shape[idx];
        i0++;
      }
    }
    output_shape.resize(i0);
  }

  output_desc->SetShape(GeShape(output_shape));
  output_desc->SetDataType(type);
  return;
}

static void DoAxisKnown(const bool& keep_dims, const std::vector<int64_t>& axis,
                        const std::vector<int64_t>& input_shape,
                        const std::vector<std::pair<int64_t, int64_t>>& input_shape_range,
                        std::vector<int64_t>& output_shape,
                        std::vector<std::pair<int64_t, int64_t>>& output_shape_range) {
  size_t input_length = input_shape.size();
  if (keep_dims) {
    output_shape = input_shape;
    output_shape_range = input_shape_range;
    for (auto item : axis) {
      output_shape[item] = 1;
      output_shape_range[item] = std::make_pair<int64_t, int64_t>(1, 1);
    }
  } else {
    std::vector<int64_t> reduce_flag(input_length);
    for (auto item : axis) {
      reduce_flag[item] = 1;
    }
    for (size_t idx = 0; idx < input_length; ++idx) {
      if (reduce_flag[idx] == 0) {
        output_shape.push_back(input_shape[idx]);
        output_shape_range.push_back(input_shape_range[idx]);
      }
    }
  }

  return;
}

static void DoAxisUnKnown(const bool& keep_dims, const std::vector<int64_t>& axis_shape,
                          const std::vector<int64_t>& input_shape,
                          const std::vector<std::pair<int64_t, int64_t>>& input_shape_range,
                          std::vector<int64_t>& output_shape,
                          std::vector<std::pair<int64_t, int64_t>>& output_shape_range) {
  size_t input_length = input_shape.size();
  size_t axis_length = axis_shape.size();
  if (keep_dims) {
    for (size_t item = 0; item < input_length; ++item) {
      int64_t range_min_value = 1;
      int64_t range_max_value = input_shape_range[item].second;
      output_shape_range.push_back(std::make_pair(range_min_value, range_max_value));
      if (range_max_value == 1) {
        output_shape.push_back(1);
      } else {
        output_shape.push_back(-1);
      }
    }
  } else {
    int64_t output_dimNum = axis_length == 0 ? (int64_t)input_length - 1 : (int64_t)input_length - axis_shape[0];
    int64_t range_min_value = input_shape_range[0].first;
    int64_t range_max_value = input_shape_range[0].second;
    for (size_t item = 0; item < input_shape.size(); ++item) {
      if (input_shape_range[item].first < range_min_value) {
        range_min_value = input_shape_range[item].first;
      }

      if (input_shape_range[item].second == -1) {
        range_max_value = -1;
      }
      if (range_max_value != -1 && input_shape_range[item].second > range_max_value) {
        range_max_value = input_shape_range[item].second;
      }
    }

    for (int64_t item = 0; item < output_dimNum; ++item) {
      output_shape.push_back(-1);
      output_shape_range.push_back(std::make_pair(range_min_value, range_max_value));
    }
  }
  return;
}

static bool InferReduceShapeProcess(const ge::Operator& op, const string& input_name, const string& axis_name,
                                    const string& keep_dims_name) {
  // Get input|output|axis desc
  const vector<string> depends = {"axes"};
  PREPARE_DYNAMIC_SHAPE(depends);
  auto input_desc = op_desc->MutableInputDesc(input_name);
  auto axis_desc = op_desc->MutableInputDesc(axis_name);
  auto output_desc = op_desc->MutableOutputDesc("y");

  vector<int64_t> input_shape = input_desc->MutableShape().GetDims();
  vector<int64_t> axis_shape = axis_desc->MutableShape().GetDims();
  auto input_type = input_desc->GetDataType();
  auto axis_type = axis_desc->GetDataType();
  size_t input_length = input_shape.size();
  size_t axis_length = axis_shape.size();

  if (input_length == 0) {
    output_desc->SetShape({});
    output_desc->SetDataType(input_type);
    return true;
  }
  if (input_shape[0] == -2) {
    std::vector<int64_t> output_shape(1, -2);
    output_desc->SetShape(GeShape(output_shape));
    output_desc->SetDataType(input_type);
    return true;
  }
  if (!axis_shape.empty() && axis_shape[0] == 0) {
    OP_LOGD(op.GetName().c_str(), "axis_shape[0] is 0");
    output_desc->SetShape(GeShape(input_shape));
    output_desc->SetDataType(input_type);
    return true;
  }
  // Get const data
  GeTensorPtr axis_tensor;
  auto node = NodeUtils::GetNodeFromOperator(op);
  auto state = NodeUtils::GetInputConstData(node, axis_name, axis_tensor);
  std::vector<int64_t> axis;
  if (GRAPH_SUCCESS == state) {
    if (axis_type == DT_INT32) {
      GetTensorValue<int32_t>(axis_tensor, axis);
    } else if (axis_type == DT_INT64) {
      GetTensorValue<int64_t>(axis_tensor, axis);
    } else {
      OP_LOGE(op.GetName().c_str(), "axis_type is illegal");
      return false;
    }
    // Convert "-1" -> "length-1";
    if (!ConvertAxis(axis, (int64_t)input_length)) {
      OP_LOGE(op.GetName().c_str(), "axis_value is illegal");
      return false;
    }
  } else {
    OP_LOGD(op.GetName().c_str(), "GetInputConstData Failed");
    if (node == nullptr) {
      OP_LOGE(op.GetName().c_str(), "get null node ptr");
    }
  }

  // Get attr
  bool keep_dims = false;
  if (!CheckReduceInfo(op, input_length, axis_length, keep_dims_name, keep_dims)) {
    OP_LOGE(op.GetName().c_str(), "Inputs and attrs are illegal");
    return false;
  }

  /* Main Process:
   * 1. Special Branch
   * 2. DoKnown Branch
   * 3. UnKnown Branch
   * */
  // Special Branch
  if (!axis_shape.empty() && (axis_shape[0] == -1 || axis_shape[0] == -2) && (!keep_dims)) {
    OP_LOGD(op.GetName().c_str(), "[Special Branch]: axis_shape[0] is -1 or -2.");
    std::vector<int64_t> output_shape;
    output_shape.push_back(-2);
    output_desc->SetShape(GeShape(output_shape));
    output_desc->SetDataType(input_type);
    return true;
  }

  // Special Branch for if axis has redundant axis value
  if (!axis_shape.empty() && axis_shape[0] > static_cast<int64_t>(input_length) && (!keep_dims)) {
    OP_LOGD(op.GetName().c_str(), "[Special Branch]: axis_shape[0] is more than input_length,"
            "if axis has redundant axis value");
    std::vector<int64_t> output_shape;
    output_shape.push_back(-2);
    output_desc->SetShape(GeShape(output_shape));
    output_desc->SetDataType(input_type);
    return true;
  }
  // DoKnown Branch && UnKnown Branch
  if ((!IsUnknown(input_shape)) && (!IsUnknown(axis_shape)) && (GRAPH_SUCCESS == state)) {
    OP_LOGD(op.GetName().c_str(), "[DoKnown Branch]: shape and axis are known.");
    DoKnownBranch(keep_dims, input_type, input_shape, axis, output_desc);
  } else {
    OP_LOGD(op.GetName().c_str(), "[UnKnown Branch]: one of inputs is unknown at least.");
    std::vector<int64_t> output_shape;
    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    std::vector<std::pair<int64_t, int64_t>> input_shape_range;
    input_desc->GetShapeRange(input_shape_range);
    // If InputShapeRange is None, MakeUpShapeRange will set range.
    MakeUpShapeRange(input_shape, input_shape_range);

    // Split as axis known and axis unknown
    if (GRAPH_SUCCESS == state) {
      OP_LOGD(op.GetName().c_str(), "[UnKnown Branch]: axis is known.");
      DoAxisKnown(keep_dims, axis, input_shape, input_shape_range, output_shape, output_shape_range);
    } else {
      OP_LOGD(op.GetName().c_str(), "[UnKnown Branch]: axis is unknown.");
      DoAxisUnKnown(keep_dims, axis_shape, input_shape, input_shape_range, output_shape, output_shape_range);
    }

    output_desc->SetDataType(input_type);
    output_desc->SetShape(GeShape(output_shape));
    output_desc->SetShapeRange(output_shape_range);
  }

  return true;
}

static bool InferReduceDShapeProcess(const ge::Operator& op, const string& input_name, const string& axis_name,
                                     const string& keep_dims_name) {
  // Get input|output desc
  const vector<string> depends = {};
  PREPARE_DYNAMIC_SHAPE(depends);
  auto input_desc = op_desc->MutableInputDesc(input_name);
  auto output_desc = op_desc->MutableOutputDesc("y");

  vector<int64_t> input_shape = input_desc->MutableShape().GetDims();
  auto input_type = input_desc->GetDataType();
  size_t input_length = input_shape.size();

  /* Main Process:
   * 1. Special Branch
   * 2. DoKnown Branch
   * 3. UnKnown Branch
   * */
  // Special Branch
  if (input_length == 0) {
    OP_LOGD(op.GetName().c_str(), "[Special Branch]: input_shape size is 0.");
    output_desc->SetShape({});
    output_desc->SetDataType(input_type);
    return true;
  }
  if (input_shape[0] == -2) {
    OP_LOGD(op.GetName().c_str(), "[Special Branch]: input_shape is -2.");
    std::vector<int64_t> output_shape(1, -2);
    output_desc->SetShape(GeShape(output_shape));
    output_desc->SetDataType(input_type);
    return true;
  }

  // Get attr: axis and keep_dims
  bool keep_dims = false;
  std::vector<int64_t> axis;
  if (!CheckReduceDInfo(op, input_length, keep_dims_name, axis_name, keep_dims, axis)) {
    OP_LOGE(op.GetName().c_str(), "KeepDims or Axis is illegal");
    return false;
  }

  // Convert "-1" -> "length-1";
  if (!ConvertAxis(axis, (int64_t)input_length)) {
    OP_LOGE(op.GetName().c_str(), "axis_value is illegal");
    return false;
  }

  // DoKnown Branch and UnKnown Branch
  if (!IsUnknown(input_shape)) {
    OP_LOGD(op.GetName().c_str(), "[DoKnown Branch]: input_shape is known.");
    DoKnownBranch(keep_dims, input_type, input_shape, axis, output_desc);
  } else {
    OP_LOGD(op.GetName().c_str(), "[UnKnown Branch]: input_shape is unknown.");
    std::vector<int64_t> output_shape(input_length);
    std::vector<std::pair<int64_t, int64_t>> output_shape_range(input_length);
    std::vector<std::pair<int64_t, int64_t>> input_shape_range;
    input_desc->GetShapeRange(input_shape_range);
    // If InputShapeRange is None, MakeUpShapeRange will set range.
    MakeUpShapeRange(input_shape, input_shape_range);

    // MainProcess
    std::vector<int64_t> reduce_flag(input_length);
    for (auto item : axis) {
      reduce_flag[item] = 1;
    }

    if (keep_dims) {
      for (size_t idx = 0; idx < input_length; ++idx) {
        output_shape[idx] = reduce_flag[idx] == 1 ? 1 : input_shape[idx];
        output_shape_range[idx] =
            reduce_flag[idx] == 1 ? std::make_pair<int64_t, int64_t>(1, 1) : input_shape_range[idx];
      }
    } else {
      size_t i0 = 0;
      for (size_t idx = 0; idx < input_length; ++idx) {
        if (reduce_flag[idx] == 0) {
          output_shape[i0] = input_shape[idx];
          output_shape_range[idx] = input_shape_range[idx];
          i0++;
        }
      }
      output_shape.resize(i0);
      output_shape_range.resize(i0);
    }

    output_desc->SetShape(GeShape(output_shape));
    output_desc->SetDataType(input_type);
    output_desc->SetShapeRange(output_shape_range);
  }
  return true;
}

static bool InferReduceDShape(const ge::Operator& op, const string& input_name, const string& axis_name,
                              const string& keep_dims_name, ge::TensorDesc& result_desc) {
  result_desc = op.GetInputDesc(input_name);
  auto shape = result_desc.GetShape();
  std::vector<int64_t> shapeVector = shape.GetDims();
  int64_t dimNum = shape.GetDimNum();

  if (shapeVector.size() == 1 && shapeVector[0] == -2) {
    std::vector<int64_t> oShapeVector;
    oShapeVector.push_back(-2);
    Shape oShape(oShapeVector);
    result_desc.SetShape(oShape);
    return true;
  }

  std::vector<int64_t> axis;
  if (GRAPH_SUCCESS != op.GetAttr(axis_name, axis)) {
    OpsGetAttrErrReport(op.GetName(), axis_name);
    OP_LOGE(op.GetName().c_str(), "GetAttr of %s failed.", axis_name.c_str());
    return false;
  }

  bool keep_dims;
  if (GRAPH_SUCCESS != op.GetAttr(keep_dims_name, keep_dims)) {
    OpsGetAttrErrReport(op.GetName(), keep_dims_name);
    OP_LOGE(op.GetName().c_str(), "GetAttr of %s failed.", keep_dims_name.c_str());
    return false;
  }

  if (axis.empty()) {
    for (size_t i = 0; i < shapeVector.size(); ++i) {
      axis.push_back(i);
    }
  }

  for (size_t i = 0; i < axis.size(); ++i) {
    if (axis[i] < -dimNum || axis[i] > (dimNum - 1)) {
      OpsInputShapeDimErrReport(op.GetName(), "axis", ConcatString(dimNum - 1), ConcatString(-dimNum),
                                ConcatString(axis[i]));
      OP_LOGE(op.GetName().c_str(), "the axis of reduce verify failed.");
      return false;
    }
    if (axis[i] < 0) {
      axis[i] = dimNum + axis[i];
    }
  }

  // infer output shape range
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  std::vector<std::pair<int64_t, int64_t>> input_shape_range;
  op_desc->MutableInputDesc(input_name)->GetShapeRange(input_shape_range);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  MakeUpShapeRange(shapeVector, input_shape_range);
  if (input_shape_range.size() != (uint32_t)dimNum) {
    OP_LOGI(op.GetName().c_str(), "reset input shape range.");
    input_shape_range.clear();
    MakeUpShapeRange(shapeVector, input_shape_range);
  }

  std::vector<int64_t> oShapeVector;
  std::vector<int64_t>::iterator tmp;
  for (int64_t item = 0; item < dimNum; ++item) {
    tmp = std::find(axis.begin(), axis.end(), item);
    if (tmp != axis.end()) {
      // item in axis
      if (keep_dims) {
        // If keepDims is true, current dimesion set to 1
        oShapeVector.push_back(1);
        output_shape_range.push_back(std::make_pair(1, 1));
      }
    } else {
      // item is not in ConstValueAxis
      oShapeVector.push_back(shapeVector[item]);
      output_shape_range.push_back(input_shape_range[item]);
    }
  }

  // clear output shape range during static shape
  bool is_static_shape = true;
  for (uint32_t i = 0; i < shapeVector.size(); ++i) {
    if (shapeVector[i] == -1) {
      is_static_shape = false;
      break;
    }
  }
  if (is_static_shape) {
    output_shape_range.clear();
  }

  Shape oShape(oShapeVector);
  result_desc.SetShape(oShape);
  result_desc.SetShapeRange(output_shape_range);
  return true;
}

// ----------------ReduceAll Op-------------------
// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ReduceAllInferShape) {
  const vector<string> depend_names = {"axes"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  OP_LOGI(op.GetName().c_str(), "Enter ReduceAll proto inferfunction!");
  ge::TensorDesc result_desc;
  if (!InferReduceShape(op, "x", "axes", "keep_dims", result_desc)) {
    return GRAPH_FAILED;
  }
  auto shape = result_desc.GetShape();
  auto dtype = result_desc.GetDataType();
  std::vector<std::pair<int64_t, int64_t>> range;
  result_desc.GetShapeRange(range);

  // update output desc
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(shape);
  output_desc.SetDataType(dtype);
  if (range.size() > 0) {
    output_desc.SetShapeRange(range);
  }
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(ReduceAll, ReduceAllInferShape);
// ----------------ReduceAll END-------------------

// ----------------ReduceAllD Op-------------------
// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ReduceAllDInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ReduceAllD proto inferfunction!");
  ge::TensorDesc result_desc;
  if (!InferReduceDShape(op, "x", "axes", "keep_dims", result_desc)) {
    return GRAPH_FAILED;
  }
  auto shape = result_desc.GetShape();
  auto dtype = result_desc.GetDataType();

  // update output desc
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(shape);
  output_desc.SetDataType(dtype);
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(ReduceAllD, ReduceAllDInferShape);
// ----------------ReduceAllD END-------------------

// ----------------ReduceProd Op-------------------
IMPLEMT_COMMON_INFERFUNC(ReduceProdInferShape) {
  const vector<string> depend_names = {"axes"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  OP_LOGI(op.GetName().c_str(), "Enter ReduceProd proto inferfunction!");
  ge::TensorDesc result_desc;
  if (!InferReduceShape(op, "x", "axes", "keep_dims", result_desc)) {
    return GRAPH_FAILED;
  }
  auto shape = result_desc.GetShape();
  auto dtype = result_desc.GetDataType();
  std::vector<std::pair<int64_t, int64_t>> range;
  result_desc.GetShapeRange(range);

  // update output desc
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(shape);
  output_desc.SetDataType(dtype);
  if (range.size() > 0) {
    output_desc.SetShapeRange(range);
  }
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ReduceProd, ReduceProdInferShape);
INFER_VALUE_RANGE_DEFAULT_REG(ReduceProd);
// ----------------ReduceProd END-------------------

// ----------------ReduceProdD Op-------------------
IMPLEMT_COMMON_INFERFUNC(ReduceProdDInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ReduceProdD proto inferfunction!");
  ge::TensorDesc result_desc;
  if (!InferReduceDShape(op, "x", "axes", "keep_dims", result_desc)) {
    return GRAPH_FAILED;
  }
  auto shape = result_desc.GetShape();
  auto dtype = result_desc.GetDataType();

  // update output desc
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(shape);
  output_desc.SetDataType(dtype);
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ReduceProdD, ReduceProdDInferShape);
// ----------------ReduceProdD END-------------------

// ----------------ReduceMean Op-----------
IMPLEMT_COMMON_INFERFUNC(ReduceMeanInferShape) {
    std::chrono::time_point<std::chrono::steady_clock> before_infer, after_infer;
    if (prof_switch) {
        before_infer = std::chrono::steady_clock::now();
    }
    OP_LOGD(op.GetName().c_str(), "Enter ReduceMeanInferShape");
    if (InferReduceShapeProcess(op, "x", "axes", "keep_dims")) {
        if (prof_switch) {
            after_infer = std::chrono::steady_clock::now();
            auto t0 = std::chrono::duration_cast<std::chrono::microseconds>(after_infer - before_infer).count();
            GEEVENT("[REDUCE_INFER_PROF] op[%s]: total: %d(us)", op.GetName().c_str(), static_cast<int>(t0));
        }
        return GRAPH_SUCCESS;
    }
    return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(ReduceMean, ReduceMeanInferShape);
// ----------------ReduceMean END-----------

// ----------------ReduceMeanD Op-------------------
IMPLEMT_COMMON_INFERFUNC(ReduceMeanDInferShape) {
  std::chrono::time_point<std::chrono::steady_clock> before_infer, after_infer;
  if (prof_switch) {
    before_infer = std::chrono::steady_clock::now();
  }
  OP_LOGD(op.GetName().c_str(), "Enter ReduceMeanDInferShape");
  if (InferReduceDShapeProcess(op, "x", "axes", "keep_dims")) {
    if (prof_switch) {
      after_infer = std::chrono::steady_clock::now();
      auto t0 = std::chrono::duration_cast<std::chrono::microseconds>(after_infer - before_infer).count();
      GEEVENT("[REDUCE_INFER_PROF] op[%s]: total: %d(us)", op.GetName().c_str(), static_cast<int>(t0));
    }
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(ReduceMeanD, ReduceMeanDInferShape);
// ----------------ReduceMeanD END-------------------

// ----------------BNTrainingReduce-------------------
IMPLEMT_COMMON_INFERFUNC(BNTrainingReduceInferShape) {
  auto tensordesc = op.GetInputDesc("x");
  auto shape = tensordesc.GetShape();
  auto format = tensordesc.GetFormat();
  std::vector<int64_t> shapeVector = shape.GetDims();
  size_t dimNum = shapeVector.size();
  std::vector<int64_t> oShapeVector;

  if (format == FORMAT_NHWC) {
    if (dimNum == 4) {
      oShapeVector.push_back(shapeVector[3]);
    } else {
      std::string err_msg = OtherErrMsg(ConcatString("Input x rank[",shapeVector.size(),"] can only support 4 when NHWC."));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  } else if (format == FORMAT_NCHW) {
    if (dimNum >= 2 && dimNum <= 4) {
      oShapeVector.push_back(shapeVector[1]);
    } else {
      string err_msg1 = ConcatString("Input x rank[", shapeVector.size(),"] can only support 2-4 when NCHW.");
      std::string err_msg = OtherErrMsg(err_msg1);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  } else if (format == FORMAT_NDHWC) {
    if (dimNum == 5) {
      oShapeVector.push_back(shapeVector[4]);
    } else {
      std::string err_msg = OtherErrMsg(ConcatString("Input x rank[",shapeVector.size(),"] can only support 5 when NDHWC."));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  } else if (format == FORMAT_NDC1HWC0) {
    if (dimNum == 6) {
      oShapeVector.push_back(1);
      oShapeVector.push_back(1);
      oShapeVector.push_back(shapeVector[2]);
      oShapeVector.push_back(1);
      oShapeVector.push_back(1);
      oShapeVector.push_back(shapeVector[5]);
    } else {
      std::string err_msg = OtherErrMsg(ConcatString("Input x rank[",shapeVector.size(),"] can only support 5 when NDC1HWC0."));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  } else if (format == FORMAT_NCDHW) {
    if (dimNum == 5) {
      oShapeVector.push_back(shapeVector[1]);
    } else {
      std::string err_msg = OtherErrMsg(ConcatString("Input x rank[",shapeVector.size(),"] can only support 5 when NCDHW."));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  } else {
    string expected_format_list = ConcatString("NCHW, NHWC, NDHWC, NCDHW, NDC1HWC0");
    std::string err_msg = GetInputFormatNotSupportErrMsg("format", expected_format_list, ConcatString(format));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc td = op.GetOutputDesc("sum");
  Shape oShape(oShapeVector);
  td.SetShape(oShape);
  td.SetDataType(DT_FLOAT);
  op.UpdateOutputDesc("sum", td);
  op.UpdateOutputDesc("square_sum", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BNTrainingReduce, BNTrainingReduceInferShape);
// ------------------BNTrainingReduce END-----------------

// ----------------BNTrainingReduceGrad-------------------
IMPLEMT_VERIFIER(BNTrainingReduceGrad, BNTrainingReduceGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "grads", "x")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BNTrainingReduceGrad, ELMTWISE_INFER_SHAPEANDTYPE("grads", "y"));

VERIFY_FUNC_REG(BNTrainingReduceGrad, BNTrainingReduceGradVerify);
// ----------------BNTrainingReduceGrad END---------------

// -------------------BNTrainingUpdate--------------------
IMPLEMT_VERIFIER(BNTrainingUpdate, BNTrainingUpdateVerify) {
  return GRAPH_SUCCESS;
}
IMPLEMT_COMMON_INFERFUNC(BNTrainingUpdateInferShape) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto input_x_desc = op_desc->GetInputDescPtr(0);
  auto input_scale_desc = op_desc->GetInputDescPtr(3);

  auto output_y_desc = op_desc->MutableOutputDesc(0);
  auto output_mean_desc = op_desc->MutableOutputDesc(1);
  auto output_variance_desc = op_desc->MutableOutputDesc(2);
  auto output_batch_mean_desc = op_desc->MutableOutputDesc(3);
  auto output_batch_variance_desc = op_desc->MutableOutputDesc(4);

  if (input_x_desc == nullptr || output_y_desc == nullptr || input_scale_desc == nullptr || 
      output_mean_desc == nullptr || output_variance_desc == nullptr || output_batch_mean_desc == nullptr ||output_batch_variance_desc == nullptr) {
    OP_LOGE(op.GetName().c_str(), "[TBE Compiler] Get null node ptr");
    return GRAPH_FAILED;
  }

  output_y_desc->SetShape(input_x_desc->GetShape());
  output_y_desc->SetDataType(input_x_desc->GetDataType());

  output_mean_desc->SetShape(input_scale_desc->GetShape());
  output_mean_desc->SetDataType(input_scale_desc->GetDataType());

  output_variance_desc->SetShape(input_scale_desc->GetShape());
  output_variance_desc->SetDataType(input_scale_desc->GetDataType());

  output_batch_mean_desc->SetShape(input_scale_desc->GetShape());
  output_batch_mean_desc->SetDataType(input_scale_desc->GetDataType());

  output_batch_variance_desc->SetShape(input_scale_desc->GetShape());
  output_batch_variance_desc->SetDataType(input_scale_desc->GetDataType());
  
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BNTrainingUpdate, BNTrainingUpdateInferShape);
VERIFY_FUNC_REG(BNTrainingUpdate, BNTrainingUpdateVerify);
// ------------------BNTrainingUpdate END---------------------

// -------------BNTrainingUpdateV2--------------------
IMPLEMT_VERIFIER(BNTrainingUpdateV2, BNTrainingUpdateV2Verify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(BNTrainingUpdateV2, BNTrainingUpdateV2InferShape) {
  auto shape = op.GetInputDesc("x").GetShape();
  auto output_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(shape);
  td.SetDataType(output_dtype);
  op.UpdateOutputDesc("y", td);

  auto shape_scale = op.GetInputDesc("scale").GetShape();
  auto output_dtype_scale = op.GetInputDesc("scale").GetDataType();

  TensorDesc td_batch_mean = op.GetOutputDesc("batch_mean");
  td_batch_mean.SetShape(shape_scale);
  td_batch_mean.SetDataType(output_dtype_scale);
  op.UpdateOutputDesc("batch_mean", td_batch_mean);

  TensorDesc td_batch_variance = op.GetOutputDesc("batch_variance");
  td_batch_variance.SetShape(shape_scale);
  td_batch_variance.SetDataType(output_dtype_scale);
  op.UpdateOutputDesc("batch_variance", td_batch_variance);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BNTrainingUpdateV2, BNTrainingUpdateV2InferShape);
VERIFY_FUNC_REG(BNTrainingUpdateV2, BNTrainingUpdateV2Verify);
// --------------BNTrainingUpdateV2 End--------------------

// -------------BNTrainingUpdateV3--------------------
IMPLEMT_VERIFIER(BNTrainingUpdateV3, BNTrainingUpdateV3Verify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(BNTrainingUpdateV3, BNTrainingUpdateV3InferShape) {
  auto shape = op.GetInputDesc("x").GetShape();
  auto output_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(shape);
  td.SetDataType(output_dtype);
  op.UpdateOutputDesc("y", td);

  auto shape_scale = op.GetInputDesc("scale").GetShape();
  auto output_dtype_scale = op.GetInputDesc("scale").GetDataType();

  TensorDesc td_batch_mean = op.GetOutputDesc("batch_mean");
  td_batch_mean.SetShape(shape_scale);
  td_batch_mean.SetDataType(output_dtype_scale);
  op.UpdateOutputDesc("batch_mean", td_batch_mean);

  TensorDesc td_batch_variance = op.GetOutputDesc("batch_variance");
  td_batch_variance.SetShape(shape_scale);
  td_batch_variance.SetDataType(output_dtype_scale);
  op.UpdateOutputDesc("batch_variance", td_batch_variance);

  TensorDesc td_reserve_1 = op.GetOutputDesc("reserve_1");
  td_reserve_1.SetShape(shape_scale);
  td_reserve_1.SetDataType(output_dtype_scale);
  op.UpdateOutputDesc("reserve_1", td_reserve_1);

  TensorDesc td_reserve_2 = op.GetOutputDesc("reserve_2");
  td_reserve_2.SetShape(shape_scale);
  td_reserve_2.SetDataType(output_dtype_scale);
  op.UpdateOutputDesc("reserve_2", td_reserve_2);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BNTrainingUpdateV3, BNTrainingUpdateV3InferShape);
VERIFY_FUNC_REG(BNTrainingUpdateV3, BNTrainingUpdateV3Verify);
// --------------BNTrainingUpdateV3 End--------------------

// ----------------BN3DTrainingReduce-------------------
IMPLEMT_COMMON_INFERFUNC(BN3DTrainingReduceInferShape) {
  auto tensordesc = op.GetInputDesc("x");
  auto shape = tensordesc.GetShape();
  auto format = tensordesc.GetFormat();
  std::vector<int64_t> shapeVector = shape.GetDims();
  size_t dimNum = shapeVector.size();
  std::vector<int64_t> oShapeVector;

  if (format == FORMAT_NDHWC) {
    if (dimNum == 5) {
      oShapeVector.push_back(shapeVector[4]);
    } else {
      OpsInputShapeDimErrReport(op.GetName(), "x", "5", "5", ConcatString(dimNum));
      OP_LOGE(op.GetName().c_str(), "Input x rank[%d] can only support 4 when NDHWC.", shapeVector.size());
      return GRAPH_FAILED;
    }
  } else if (format == FORMAT_NCDHW) {
    if (dimNum >= 2 && dimNum <= 5) {
      oShapeVector.push_back(shapeVector[1]);
    } else {
      OpsInputShapeDimErrReport(op.GetName(), "x", "5", "5", ConcatString(dimNum));
      OP_LOGE(op.GetName().c_str(), "Input x rank[%d] can only support 2-4 when NCDHW.", shapeVector.size());
      return GRAPH_FAILED;
    }
  } else if (format == FORMAT_NDC1HWC0) {
    if (dimNum == 6) {
      oShapeVector.push_back(1);
      oShapeVector.push_back(1);
      oShapeVector.push_back(shapeVector[2]);
      oShapeVector.push_back(1);
      oShapeVector.push_back(1);
      oShapeVector.push_back(shapeVector[5]);
    } else {
      OpsInputShapeDimErrReport(op.GetName(), "x", "6", "6", ConcatString(dimNum));
      OP_LOGE(op.GetName().c_str(), "Input x rank[%d] can only support 5 when NDC1HWC0.", shapeVector.size());
      return GRAPH_FAILED;
    }
  } else {
    OpsInputFormatErrReport(op.GetName().c_str(), "inputFormat", "NCHW or NHWC or NCDHW or NDHWC", ConcatString(format));
    OP_LOGE(op.GetName().c_str(), "This op can only support NCHW , NHWC, NDHWC, NCDHW, NDC1HWC0");
    return GRAPH_FAILED;
  }

  TensorDesc td = op.GetOutputDesc("sum");
  Shape oShape(oShapeVector);
  td.SetShape(oShape);
  td.SetDataType(DT_FLOAT);
  op.UpdateOutputDesc("sum", td);
  op.UpdateOutputDesc("square_sum", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BN3DTrainingReduce, BN3DTrainingReduceInferShape);
// ------------------BN3DTrainingReduce END-----------------

// ----------------BN3DTrainingReduceGrad-------------------
IMPLEMT_VERIFIER(BN3DTrainingReduceGrad, BN3DTrainingReduceGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "grads", "x")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BN3DTrainingReduceGrad, ELMTWISE_INFER_SHAPEANDTYPE("grads", "y"));

VERIFY_FUNC_REG(BN3DTrainingReduceGrad, BN3DTrainingReduceGradVerify);
// ----------------BN3DTrainingReduceGrad END---------------

// -------------------BN3DTrainingUpdate--------------------
IMPLEMT_COMMON_INFERFUNC(BN3DTrainingUpdateInferShape) {
  auto shape = op.GetInputDesc("x").GetShape();
  auto output_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(shape);
  td.SetDataType(output_dtype);
  op.UpdateOutputDesc("y", td);

  auto shape_scale = op.GetInputDesc("scale").GetShape();
  auto output_dtype_scale = op.GetInputDesc("scale").GetDataType();

  TensorDesc td_mean = op.GetOutputDesc("mean");
  td_mean.SetShape(shape_scale);
  td_mean.SetDataType(output_dtype_scale);
  op.UpdateOutputDesc("mean", td_mean);

  TensorDesc td_variance = op.GetOutputDesc("variance");
  td_variance.SetShape(shape_scale);
  td_variance.SetDataType(output_dtype_scale);
  op.UpdateOutputDesc("variance", td_variance);

  TensorDesc td_batch_mean = op.GetOutputDesc("batch_mean");
  td_batch_mean.SetShape(shape_scale);
  td_batch_mean.SetDataType(output_dtype_scale);
  op.UpdateOutputDesc("batch_mean", td_batch_mean);

  TensorDesc td_batch_variance = op.GetOutputDesc("batch_variance");
  td_batch_variance.SetShape(shape_scale);
  td_batch_variance.SetDataType(output_dtype_scale);
  op.UpdateOutputDesc("batch_variance", td_batch_variance);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BN3DTrainingUpdate, BN3DTrainingUpdateInferShape);
// ------------------BN3DTrainingUpdate END---------------------

// ------------------------BNInfer--------------------------
IMPLEMT_VERIFIER(BNInfer, BNInferVerify) {
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(BNInfer, BNInferVerify);
IMPLEMT_COMMON_INFERFUNC(BNInferInferShape){
  if(OneInOneOutDynamicInfer(op, "x", {"y"})){
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(BNInfer, BNInferInferShape);
// ----------------------BNInfer End--------------------------

// ------------------BNTrainingUpdateGrad---------------------
IMPLEMT_VERIFIER(BNTrainingUpdateGrad, BNTrainingUpdateGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "grads", "x")) {
    OP_LOGE(op.GetName().c_str(), "[TBE Compiler] the dtype of grads and x should be the same");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
IMPLEMT_COMMON_INFERFUNC(BNTrainingUpdateGradInferShape) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto input_x_desc = op_desc->GetInputDescPtr(1);
  auto input_mean_desc = op_desc->GetInputDescPtr(2);

  auto output_scale_desc = op_desc->MutableOutputDesc(0);
  auto output_offset_desc = op_desc->MutableOutputDesc(1);

  if (input_x_desc == nullptr || input_mean_desc == nullptr ||
      output_scale_desc == nullptr || output_offset_desc == nullptr) {
    OP_LOGE(op.GetName().c_str(), "[TBE Compiler] Get null node ptr");
    return GRAPH_FAILED;
  }

  output_scale_desc->SetShape(input_mean_desc->GetShape());
  output_scale_desc->SetDataType(input_mean_desc->GetDataType());

  output_offset_desc->SetShape(input_mean_desc->GetShape());
  output_offset_desc->SetDataType(input_mean_desc->GetDataType());

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BNTrainingUpdateGrad, BNTrainingUpdateGradInferShape);
VERIFY_FUNC_REG(BNTrainingUpdateGrad, BNTrainingUpdateGradVerify);
// ----------------BNTrainingUpdateGrad END-------------------

// ------------------BN3DTrainingUpdateGrad---------------------
IMPLEMT_VERIFIER(BN3DTrainingUpdateGrad, BN3DTrainingUpdateGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "grads", "x")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
IMPLEMT_COMMON_INFERFUNC(BN3DTrainingUpdateGradInferShape) {
  auto shape = op.GetInputDesc("batch_mean").GetShape();
  auto output_dtype = op.GetInputDesc("batch_mean").GetDataType();

  TensorDesc td_diff_scale = op.GetOutputDesc("diff_scale");
  td_diff_scale.SetShape(shape);
  td_diff_scale.SetDataType(output_dtype);
  op.UpdateOutputDesc("diff_scale", td_diff_scale);

  TensorDesc td_diff_offset = op.GetOutputDesc("diff_offset");
  td_diff_offset.SetShape(shape);
  td_diff_offset.SetDataType(output_dtype);
  op.UpdateOutputDesc("diff_offset", td_diff_scale);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BN3DTrainingUpdateGrad, BN3DTrainingUpdateGradInferShape);
VERIFY_FUNC_REG(BN3DTrainingUpdateGrad, BN3DTrainingUpdateGradVerify);
// ----------------BN3DTrainingUpdateGrad END-------------------

// ----------------BNInferGrad-------------------------
IMPLEMT_VERIFIER(BNInferGrad, BNInferGradVerify) {
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(BNInferGrad, BNInferGradVerify);
IMPLEMT_COMMON_INFERFUNC(BNInferGradInferShape){
  if(OneInOneOutDynamicInfer(op, "grads", {"x_backprop"})){
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(BNInferGrad, BNInferGradInferShape);
// ----------------BNInferGrad End----------------------

// ----------------ReduceSum Op-------------------
IMPLEMT_COMMON_INFERFUNC(ReduceSumInferShape) {
  std::chrono::time_point<std::chrono::steady_clock> before_infer, after_infer;
  if (prof_switch) {
    before_infer = std::chrono::steady_clock::now();
  }
  OP_LOGD(op.GetName().c_str(), "Enter ReduceSumInferShape");
  if (InferReduceShapeProcess(op, "x", "axes", "keep_dims")) {
    if (prof_switch) {
      after_infer = std::chrono::steady_clock::now();
      auto t0 = std::chrono::duration_cast<std::chrono::microseconds>(after_infer - before_infer).count();
      GEEVENT("[REDUCE_INFER_PROF] op[%s]: total: %d(us)", op.GetName().c_str(), static_cast<int>(t0));
    }
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(ReduceSum, ReduceSumInferShape);
// ----------------ReduceSum END-------------------

// ----------------ReduceSumD Op-------------------
IMPLEMT_COMMON_INFERFUNC(ReduceSumDInferShape) {
  std::chrono::time_point<std::chrono::steady_clock> before_infer, after_infer;
  if (prof_switch) {
    before_infer = std::chrono::steady_clock::now();
  }
  OP_LOGD(op.GetName().c_str(), "Enter ReduceSumDInferShape");
  if (InferReduceDShapeProcess(op, "x", "axes", "keep_dims")) {
    if (prof_switch) {
      after_infer = std::chrono::steady_clock::now();
      auto t0 = std::chrono::duration_cast<std::chrono::microseconds>(after_infer - before_infer).count();
      GEEVENT("[REDUCE_INFER_PROF] op[%s]: total: %d(us)", op.GetName().c_str(), static_cast<int>(t0));
    }
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(ReduceSumD, ReduceSumDInferShape);
// ----------------ReduceSumD END-------------------

// ----------------ReduceAny Op-------------------
IMPLEMT_COMMON_INFERFUNC(ReduceAnyInferShape) {
  const vector<string> depend_names = {"axes"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  OP_LOGI(op.GetName().c_str(), "Enter ReduceAny proto inferfunction!");
  ge::TensorDesc result_desc;
  if (!InferReduceShape(op, "x", "axes", "keep_dims", result_desc)) {
    return GRAPH_FAILED;
  }
  auto shape = result_desc.GetShape();
  auto dtype = result_desc.GetDataType();
  std::vector<std::pair<int64_t, int64_t>> range;
  result_desc.GetShapeRange(range);

  // update output desc
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(shape);
  output_desc.SetDataType(dtype);
  if (range.size() > 0) {
    output_desc.SetShapeRange(range);
  }
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ReduceAny, ReduceAnyInferShape);
// ----------------ReduceAny END-------------------

// ----------------ReduceAnyD Op-------------------
IMPLEMT_COMMON_INFERFUNC(ReduceAnyDInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ReduceAnyD proto inferfunction!");
  ge::TensorDesc result_desc;
  if (!InferReduceDShape(op, "x", "axes", "keep_dims", result_desc)) {
    return GRAPH_FAILED;
  }
  std::vector<int64_t> axes_dim;
  if (ge::GRAPH_SUCCESS != op.GetAttr("axes", axes_dim)) {
    std::string err_msg = GetInputInvalidErrMsg("axes");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (axes_dim.size() < DIM_SIZE1 || axes_dim.size() > DIM_SIZE8) {
    string correct_size = ConcatString("must be between 1 and 8");
    std::string err_msg = GetAttrSizeErrMsg("axes_dim", ConcatString(axes_dim.size()), correct_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  auto shape = result_desc.GetShape();
  auto dtype = result_desc.GetDataType();

  // update output desc
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(shape);
  output_desc.SetDataType(dtype);
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ReduceAnyD, ReduceAnyDInferShape);
// ----------------ReduceAnyD END-------------------

// ----------------ReduceMax
IMPLEMT_COMMON_INFERFUNC(ReduceMaxInferShape) {
  std::chrono::time_point<std::chrono::steady_clock> before_infer, after_infer;
  if (prof_switch) {
    before_infer = std::chrono::steady_clock::now();
  }
  OP_LOGD(op.GetName().c_str(), "Enter Start ReduceMaxInferShape");
  if (InferReduceShapeProcess(op, "x", "axes", "keep_dims")) {
    if (prof_switch) {
      after_infer = std::chrono::steady_clock::now();
      auto t0 = std::chrono::duration_cast<std::chrono::microseconds>(after_infer - before_infer).count();
      GEEVENT("[REDUCE_INFER_PROF] opname[%s]: total: %d(us)", op.GetName().c_str(), static_cast<int>(t0));
    }
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(ReduceMax, ReduceMaxInferShape);
// ----------------ReduceMax END

// ----------------ReduceMaxD Op-------------------
IMPLEMT_COMMON_INFERFUNC(ReduceMaxDInferShape) {
  std::chrono::time_point<std::chrono::steady_clock> before_infer, after_infer;
  if (prof_switch) {
    before_infer = std::chrono::steady_clock::now();
  }
  OP_LOGD(op.GetName().c_str(), "Enter ReduceMaxDInferShape");
  if (InferReduceDShapeProcess(op, "x", "axes", "keep_dims")) {
    if (prof_switch) {
      after_infer = std::chrono::steady_clock::now();
      auto t0 = std::chrono::duration_cast<std::chrono::microseconds>(after_infer - before_infer).count();
      GEEVENT("[REDUCE_INFER_PROF] op[%s]: total: %d(us)", op.GetName().c_str(), static_cast<int>(t0));
    }
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(ReduceMaxD, ReduceMaxDInferShape);
// ----------------ReduceMaxD END-------------------

// ----------------ReduceMin Op-----------
IMPLEMT_COMMON_INFERFUNC(ReduceMinInferShape) {
    std::chrono::time_point<std::chrono::steady_clock> before_infer, after_infer;
    if (prof_switch) {
        before_infer = std::chrono::steady_clock::now();
    }
    OP_LOGD(op.GetName().c_str(), "Enter ReduceMinInferShape");
    if (InferReduceShapeProcess(op, "x", "axes", "keep_dims")) {
        if (prof_switch) {
            after_infer = std::chrono::steady_clock::now();
            auto t0 = std::chrono::duration_cast<std::chrono::microseconds>(after_infer - before_infer).count();
            GEEVENT("[REDUCE_INFER_PROF] op[%s]: total: %d(us)", op.GetName().c_str(), static_cast<int>(t0));
        }
        return GRAPH_SUCCESS;
    }
    return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(ReduceMin, ReduceMinInferShape);
// ----------------ReduceMin END-----------

// ----------------ReduceMinD Op-------------------
IMPLEMT_COMMON_INFERFUNC(ReduceMinDInferShape) {
    std::chrono::time_point<std::chrono::steady_clock> before_infer, after_infer;
    if (prof_switch) {
        before_infer = std::chrono::steady_clock::now();
    }
    OP_LOGD(op.GetName().c_str(), "Enter ReduceMinDInferShape");
    if (InferReduceDShapeProcess(op, "x", "axes", "keep_dims")) {
        if (prof_switch) {
            after_infer = std::chrono::steady_clock::now();
            auto t0 = std::chrono::duration_cast<std::chrono::microseconds>(after_infer - before_infer).count();
            GEEVENT("[REDUCE_INFER_PROF] op[%s]: total: %d(us)", op.GetName().c_str(), static_cast<int>(t0));
        }
        return GRAPH_SUCCESS;
    }
    return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(ReduceMinD, ReduceMinDInferShape);
// ----------------ReduceMinD END-------------------

// ----------------EuclideanNorm Op-------------------
IMPLEMT_COMMON_INFERFUNC(EuclideanNormInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter EuclideanNorm proto inferfunction!");
  ge::TensorDesc result_desc;
  if (!InferReduceShape(op, "x", "axes", "keep_dims", result_desc)) {
    return GRAPH_FAILED;
  }
  auto shape = result_desc.GetShape();
  auto dtype = result_desc.GetDataType();

  // update output desc
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(shape);
  output_desc.SetDataType(dtype);
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(EuclideanNorm, EuclideanNormInferShape);
// ----------------EuclideanNorm END-------------------

// ----------------EuclideanNormD Op-------------------
IMPLEMT_COMMON_INFERFUNC(EuclideanNormDInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter EuclideanNormD proto inferfunction!");
  ge::TensorDesc result_desc;
  if (!InferReduceDShape(op, "x", "axes", "keep_dims", result_desc)) {
    return GRAPH_FAILED;
  }
  auto shape = result_desc.GetShape();
  auto dtype = result_desc.GetDataType();

  // update output desc
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(shape);
  output_desc.SetDataType(dtype);
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(EuclideanNormD, EuclideanNormDInferShape);
// ----------------EuclideanNormD END-------------------

// ------------------------INInferV2--------------------------
IMPLEMT_VERIFIER(INInferV2, INInferV2Verify) {
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(INInferV2, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
COMMON_INFER_FUNC_REG(INInferV2, ELMTWISE_INFER_SHAPEANDTYPE("mean", "batch_mean"));
COMMON_INFER_FUNC_REG(INInferV2, ELMTWISE_INFER_SHAPEANDTYPE("variance", "batch_variance"));

VERIFY_FUNC_REG(INInferV2, INInferV2Verify);
// ----------------------INInferV2 End--------------------------

// ------------------------INTrainingReduceV2--------------------------
IMPLEMT_COMMON_INFERFUNC(INTrainingReduceV2InferShape) {
  // x desc
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_shape = input_desc->MutableShape();

  // x dims
  std::vector<int64_t> dims_input = input_shape.GetDims();
  int64_t dim_num = input_shape.GetDimNum();

  // get sum and square_sum output shape for nhwc or ndhwc
  std::vector<int64_t> o_shape_vec;
  for (int i = 0; i < dim_num; i++) {
    if (i != 0 && i != dim_num - 1) {
      o_shape_vec.push_back(1);
    } else {
      o_shape_vec.push_back(dims_input[i]);
    }
  }

  // update sum and square_sum output desc
  auto sum_desc = op_info->MutableOutputDesc("sum");
  auto square_sum_desc = op_info->MutableOutputDesc("square_sum");
  sum_desc->SetShape(GeShape(o_shape_vec));
  square_sum_desc->SetShape(GeShape(o_shape_vec));
  sum_desc->SetDataType(DT_FLOAT);
  square_sum_desc->SetDataType(DT_FLOAT);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(INTrainingReduceV2, INTrainingReduceV2InferShape);
// ----------------------INTrainingReduceV2 End--------------------------

// -------------------INTrainingUpdateV2--------------------
IMPLEMT_COMMON_INFERFUNC(INTrainingUpdateV2InferShape) {
  // x desc and sum desc
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_shape = input_desc->MutableShape();
  auto input_dtype = input_desc->GetDataType();
  auto sum_desc = op_info->MutableInputDesc("sum");
  auto sum_shape = sum_desc->MutableShape();

  // x dims
  std::vector<int64_t> dims_input = input_shape.GetDims();

  // update y output desc
  auto y_desc = op_info->MutableOutputDesc("y");
  y_desc->SetShape(input_shape);
  y_desc->SetDataType(input_dtype);

  // update mean an variance output desc
  auto batch_mean_desc = op_info->MutableOutputDesc("batch_mean");
  auto batch_variance_desc = op_info->MutableOutputDesc("batch_variance");
  batch_mean_desc->SetShape(sum_shape);
  batch_variance_desc->SetShape(sum_shape);
  batch_mean_desc->SetDataType(DT_FLOAT);
  batch_variance_desc->SetDataType(DT_FLOAT);

  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(INTrainingUpdateV2, INTrainingUpdateV2InferShape);
// ------------------INTrainingUpdateV2 END---------------------

// ------------------------INTrainingReduceGrad--------------------------
IMPLEMT_COMMON_INFERFUNC(INTrainingReduceGradInferShape) {
  // x desc
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_shape = input_desc->MutableShape();
  auto input_dtype = input_desc->GetDataType();

  // update output desc
  auto pd_x_desc = op_info->MutableOutputDesc("pd_x");
  pd_x_desc->SetShape(input_shape);
  pd_x_desc->SetDataType(input_dtype);;

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(INTrainingReduceGrad, INTrainingReduceGradInferShape);
// ----------------------INTrainingReduceGrad End--------------------------

// -------------------INTrainingUpdateGrad--------------------
IMPLEMT_COMMON_INFERFUNC(INTrainingUpdateGradInferShape) {
  // variance desc
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("variance");
  auto input_shape = input_desc->MutableShape();
  auto input_dtype = input_desc->GetDataType();

  // update output desc
  auto res_gamma_desc = op_info->MutableOutputDesc("res_gamma");
  auto res_beta_desc = op_info->MutableOutputDesc("res_beta");
  res_gamma_desc->SetShape(input_shape);
  res_beta_desc->SetShape(input_shape);
  res_gamma_desc->SetDataType(input_dtype);
  res_beta_desc->SetDataType(input_dtype);

  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(INTrainingUpdateGrad, INTrainingUpdateGradInferShape);
// ------------------INTrainingUpdateGrad END---------------------

// -------------------INTrainingUpdateGradGammaBeta--------------------
IMPLEMT_COMMON_INFERFUNC(INTrainingUpdateGradGammaBetaInferShape) {
  // res_gamma desc
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("res_gamma");
  auto input_shape = input_desc->MutableShape();
  auto input_dtype = input_desc->GetDataType();

  // res_gamma dims
  std::vector<int64_t> dims_input = input_shape.GetDims();
  if (input_shape.GetDimNum() >= 1) {
    dims_input[0] = 1;
  }

  // update output desc
  auto pd_gamma_desc = op_info->MutableOutputDesc("pd_gamma");
  auto pd_beta_desc = op_info->MutableOutputDesc("pd_beta");
  pd_gamma_desc->SetShape(GeShape(dims_input));
  pd_beta_desc->SetShape(GeShape(dims_input));
  pd_gamma_desc->SetDataType(input_dtype);
  pd_beta_desc->SetDataType(input_dtype);

  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(INTrainingUpdateGradGammaBeta, INTrainingUpdateGradGammaBetaInferShape);
// ------------------INTrainingUpdateGradGammaBeta END---------------------

// ------------------------GNTrainingReduce--------------------------
IMPLEMT_COMMON_INFERFUNC(GNTrainingReduceInferShape) {
  auto inputTensorDesc = op.GetInputDesc("x");
  auto shape = inputTensorDesc.GetShape();

  Format input_format = inputTensorDesc.GetFormat();

  if (input_format != FORMAT_NHWC && input_format != FORMAT_NCHW) {
    string expected_format_list = ConcatString("NHWC, NCHW");
    std::string err_msg = GetInputFormatNotSupportErrMsg("input_format", expected_format_list, ConcatString(input_format));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int64_t num = 2;
  if (GRAPH_SUCCESS != op.GetAttr("num_groups", num)) {
    std::string err_msg = GetInputInvalidErrMsg("num_groups");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
  }

  std::vector<int64_t> dims_input;
  dims_input = shape.GetDims();
  Format dataFormat = inputTensorDesc.GetFormat();
  std::vector<int64_t> dimVector;
  int64_t dimNum = shape.GetDimNum();

  if (dataFormat == FORMAT_NCHW) {
    for (int64_t item = 0; item < dimNum; ++item) {
      if (item == 2 || item == 3) {
        dimVector.push_back(1);
      } else if (item == 0) {
        dimVector.push_back(dims_input[item]);
      } else if (item == 1) {
        dimVector.push_back(num);
        dimVector.push_back(1);
      }
    }
  } else if (dataFormat == FORMAT_NHWC) {
    for (int64_t item = 0; item < dimNum; ++item) {
      if (item == 1 || item == 2) {
        dimVector.push_back(1);
      } else if (item == 0) {
        dimVector.push_back(dims_input[item]);
      } else if (item == 3) {
        dimVector.push_back(num);
        dimVector.push_back(1);
      }
    }
  }

  TensorDesc sum = op.GetOutputDesc("sum");
  sum.SetShape(ge::Shape(dimVector));
  sum.SetDataType(DT_FLOAT);
  (void)op.UpdateOutputDesc("sum", sum);

  TensorDesc square_sum = op.GetOutputDesc("square_sum");
  square_sum.SetShape(ge::Shape(dimVector));
  square_sum.SetDataType(DT_FLOAT);
  (void)op.UpdateOutputDesc("square_sum", square_sum);

  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(GNTrainingReduce, GNTrainingReduceInferShape);
// ----------------------GNTrainingReduce END--------------------------

// -------------------GNTrainingUpdate--------------------
IMPLEMT_COMMON_INFERFUNC(GNTrainingUpdateInferShape) {
  auto inputTensorDesc = op.GetInputDesc("x");
  auto shape = inputTensorDesc.GetShape();
  auto output_dtype = inputTensorDesc.GetDataType();
  Format input_format = inputTensorDesc.GetFormat();

  if (input_format != FORMAT_NHWC && input_format != FORMAT_NCHW) {
    string expected_format_list = ConcatString("NHWC, NCHW");
    std::string err_msg = GetInputFormatNotSupportErrMsg("input_format", expected_format_list, ConcatString(input_format));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int64_t num = 2;
  if (GRAPH_SUCCESS != op.GetAttr("num_groups", num)) {
    std::string err_msg = GetInputInvalidErrMsg("num_groups");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
  }
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(shape);
  td.SetDataType(output_dtype);
  op.UpdateOutputDesc("y", td);

  auto shape_sum = op.GetInputDesc("sum").GetShape();
  auto output_dtype_sum = op.GetInputDesc("sum").GetDataType();

  TensorDesc td_mean = op.GetOutputDesc("batch_mean");
  td_mean.SetShape(shape_sum);
  td_mean.SetDataType(output_dtype_sum);
  op.UpdateOutputDesc("batch_mean", td_mean);

  TensorDesc td_variance = op.GetOutputDesc("batch_variance");
  td_variance.SetShape(shape_sum);
  td_variance.SetDataType(output_dtype_sum);
  op.UpdateOutputDesc("batch_variance", td_variance);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(GNTrainingUpdate, GNTrainingUpdateInferShape);
// ------------------GNTrainingUpdate END---------------------

IMPLEMT_INFERFUNC(ReduceJoin, ReduceJoinInfer) {
  OP_LOGI(op.GetName().c_str(), "Enter ReduceJoin proto inferfunction!");
  ge::TensorDesc result_desc;
  if (!InferReduceShape(op, "input", "reduction_indices", "keep_dims", result_desc)) {
    std::string err_msg = ConcatString("failed to call InferReduceShape function,", 
                                        "input[input], input[reduction_indices], attr[keep_dims]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  auto shape = result_desc.GetShape();
  auto dtype = result_desc.GetDataType();
  std::vector<std::pair<int64_t, int64_t>> range;
  result_desc.GetShapeRange(range);

  // update output desc
  TensorDesc output_desc = op.GetOutputDesc("output");
  output_desc.SetShape(shape);
  output_desc.SetDataType(dtype);
  if (range.size() > 0) {
    output_desc.SetShapeRange(range);
  }
  (void)op.UpdateOutputDesc("output", output_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ReduceJoin, ReduceJoinInfer);

// ----------------ReduceStd Begin-------------------
using std::find;
IMPLEMT_INFERFUNC(ReduceStd, ReduceStdInferShape) {
  TensorDesc tensordesc_input = op.GetInputDescByName("x");
  ge::Shape input_shape = tensordesc_input.GetShape();
  DataType input_dtype = tensordesc_input.GetDataType();
  std::vector<int64_t> dims_input = input_shape.GetDims();
  int64_t dim_num = input_shape.GetDimNum();

  TensorDesc tensordesc_output1 = op.GetOutputDescByName("y1");
  TensorDesc tensordesc_output2 = op.GetOutputDescByName("y2");
  tensordesc_output1.SetDataType(input_dtype);
  tensordesc_output2.SetDataType(input_dtype);

  AscendString op_name;
  if (GRAPH_SUCCESS != op.GetName(op_name)) {
    OP_LOGE("ReduceStd", "op_name get failed.");
    return GRAPH_FAILED;
  }
  const char* op_name_c = op_name.GetString();

  bool keepdim;
  (void)op.GetAttr("keepdim", keepdim);

  // check parameter dim and keepdim
  std::vector<int64_t> axis;
  if (GRAPH_SUCCESS != op.GetAttr("dim", axis)) {
    OP_LOGE(op_name_c, "GE get dim failed");
    return GRAPH_FAILED;
  }

  for (size_t i = 0; i < axis.size(); i++) {
    if (axis[i] < 0) {
      axis[i] = axis[i] + dim_num;
    }
  }

  if (axis.empty()) {
    for (int i = 0; i < dim_num; i++) {
      axis.push_back(i);
    }
  }

  std::vector<int64_t> oshape_vector;
  for (int item = 0; item < dim_num; ++item) {
    if (find(axis.begin(), axis.end(), item) != axis.end()) {
      // item in axis
      if (keepdim == true) {
        // If keepDims is true, current dimesion set to 1
        oshape_vector.push_back(1);
      }
    } else {
      // item is not in ConstValueAxis
      oshape_vector.push_back(dims_input[item]);
    }
  }

  Shape oshape(oshape_vector);
  tensordesc_output1.SetShape(oshape);
  tensordesc_output2.SetShape(oshape);
  (void)op.UpdateOutputDesc("y1", tensordesc_output1);
  (void)op.UpdateOutputDesc("y2", tensordesc_output2);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ReduceStd, ReduceStdInferShape);
// ----------------ReduceStd END---------------------

// ----------------ReduceStdWithMean Begin-------------------
using std::find;
IMPLEMT_INFERFUNC(ReduceStdWithMean, ReduceStdWithMeanInferShape) {
  TensorDesc tensordesc_input = op.GetInputDescByName("x");
  ge::Shape input_shape = tensordesc_input.GetShape();
  DataType input_dtype = tensordesc_input.GetDataType();
  std::vector<int64_t> dims_input = input_shape.GetDims();
  int64_t dim_num = input_shape.GetDimNum();

  TensorDesc tensordesc_output = op.GetOutputDescByName("y");
  tensordesc_output.SetDataType(input_dtype);
  tensordesc_output.SetFormat(tensordesc_input.GetFormat());

  AscendString op_name;
  if (GRAPH_SUCCESS != op.GetName(op_name)) {
    OP_LOGE("ReduceStdWithMean", "op_name get failed.");
    return GRAPH_FAILED;
  }
  const char* op_name_c = op_name.GetString();

  bool keepdim;
  (void)op.GetAttr("keepdim", keepdim);

  // check parameter dim and keepdim
  std::vector<int64_t> axis;
  if (GRAPH_SUCCESS != op.GetAttr("dim", axis)) {
    OP_LOGE(op_name_c, "GE get dim failed");
    return GRAPH_FAILED;
  }

  for (size_t i = 0; i < axis.size(); i++) {
    if (axis[i] < 0) {
      axis[i] = axis[i] + dim_num;
    }
  }

  if (axis.empty()) {
    for (int i = 0; i < dim_num; i++) {
      axis.push_back(i);
    }
  }

  std::vector<int64_t> oshape_vector;
  for (int item = 0; item < dim_num; ++item) {
    if (find(axis.begin(), axis.end(), item) != axis.end()) {
      // item in axis
      if (keepdim == true) {
        // If keepDims is true, current dimesion set to 1
        oshape_vector.push_back(1);
      }
    } else {
      // item is not in ConstValueAxis
      oshape_vector.push_back(dims_input[item]);
    }
  }

  std::vector<std::pair<int64_t, int64_t>> input_mean_range;
  op.GetInputDescByName("mean").GetShapeRange(input_mean_range);

  tensordesc_output.SetShapeRange(input_mean_range);

  Shape oshape(oshape_vector);
  tensordesc_output.SetShape(oshape);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ReduceStdWithMean, ReduceStdWithMeanInferShape);
// ----------------ReduceStdWithMean END---------------------

}  // namespace ge
