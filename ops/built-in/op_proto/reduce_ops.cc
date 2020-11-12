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

namespace ge {
using std::string;

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
  OP_LOGI(op.GetName().c_str(), "Enter ReduceAll proto inferfunction!");
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
  OP_LOGI(op.GetName().c_str(), "Enter ReduceProd proto inferfunction!");
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

COMMON_INFER_FUNC_REG(ReduceProd, ReduceProdInferShape);
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

// ----------------ReduceMean Op-------------------
IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(ReduceMeanInferShape)
  OP_LOGI(op.GetName().c_str(), "Enter ReduceMean proto inferfunction!");
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
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(ReduceMean, ReduceMeanInferShape);
// ----------------ReduceMean END-------------------

// ----------------ReduceMeanD Op-------------------
IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(ReduceMeanDInferShape)
  OP_LOGI(op.GetName().c_str(), "Enter ReduceMeanD proto inferfunction!");
  ge::TensorDesc result_desc;
  if (!InferReduceDShape(op, "x", "axes", "keep_dims", result_desc)) {
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
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

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
      OpsInputShapeDimErrReport(op.GetName(), "x", "4", "4", ConcatString(dimNum));
      OP_LOGE(op.GetName().c_str(), "Input x rank[%d] can only support 4 when NHWC.", shapeVector.size());
      return GRAPH_FAILED;
    }
  } else if (format == FORMAT_NCHW) {
    if (dimNum >= 2 && dimNum <= 4) {
      oShapeVector.push_back(shapeVector[1]);
    } else {
      OpsInputShapeDimErrReport(op.GetName(), "x", "4", "2", ConcatString(dimNum));
      OP_LOGE(op.GetName().c_str(), "Input x rank[%d] can only support 2-4 when NCHW.", shapeVector.size());
      return GRAPH_FAILED;
    }
  } else {
    OpsInputFormatErrReport(op.GetName().c_str(), "inputFormat", "NCHW or NHWC", ConcatString(format));
    OP_LOGE(op.GetName().c_str(), "This op can only support NCHW and NHWC.");
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
IMPLEMT_COMMON_INFERFUNC(BNTrainingUpdateInferShape) {
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

COMMON_INFER_FUNC_REG(BNTrainingUpdate, BNTrainingUpdateInferShape);
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

// ------------------------BNInfer--------------------------
IMPLEMT_VERIFIER(BNInfer, BNInferVerify) {
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BNInfer, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));

VERIFY_FUNC_REG(BNInfer, BNInferVerify);
// ----------------------BNInfer End--------------------------

// ------------------BNTrainingUpdateGrad---------------------
IMPLEMT_VERIFIER(BNTrainingUpdateGrad, BNTrainingUpdateGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "grads", "x")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
IMPLEMT_COMMON_INFERFUNC(BNTrainingUpdateGradInferShape) {
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

COMMON_INFER_FUNC_REG(BNTrainingUpdateGrad, BNTrainingUpdateGradInferShape);
VERIFY_FUNC_REG(BNTrainingUpdateGrad, BNTrainingUpdateGradVerify);
// ----------------BNTrainingUpdateGrad END-------------------

// ----------------BNInferGrad-------------------------
IMPLEMT_VERIFIER(BNInferGrad, BNInferGradVerify) {
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BNInferGrad, ELMTWISE_INFER_SHAPEANDTYPE("grads", "x_backprop"));

VERIFY_FUNC_REG(BNInferGrad, BNInferGradVerify);
// ----------------BNInferGrad End----------------------

// ----------------ReduceSum Op-------------------
IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(ReduceSumInferShape)
  OP_LOGI(op.GetName().c_str(), "Enter ReduceSum proto inferfunction!");
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
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(ReduceSum, ReduceSumInferShape);
// ----------------ReduceSum END-------------------

// ----------------ReduceSumD Op-------------------
IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(ReduceSumDInferShape)
  OP_LOGI(op.GetName().c_str(), "Enter ReduceSumD proto inferfunction!");
  ge::TensorDesc result_desc;
  if (!InferReduceDShape(op, "x", "axes", "keep_dims", result_desc)) {
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
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(ReduceSumD, ReduceSumDInferShape);
// ----------------ReduceSumD END-------------------

// ----------------ReduceAny Op-------------------
IMPLEMT_COMMON_INFERFUNC(ReduceAnyInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ReduceAny proto inferfunction!");
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
    OpsGetAttrErrReport(op.GetName(), "axes");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr failed of ReduceAnyD!");
    return GRAPH_FAILED;
  }

  if (axes_dim.size() < DIM_SIZE1 || axes_dim.size() > DIM_SIZE8) {
    OpsInputShapeDimErrReport(op.GetName(), "axes", ConcatString(DIM_SIZE8), ConcatString(DIM_SIZE1),
                              ConcatString(axes_dim.size()));
    OP_LOGE(op.GetName().c_str(), "axes must be between 1 and 8.");
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

// ----------------ReduceMax Op-------------------
IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(ReduceMaxInferShape)
  OP_LOGI(op.GetName().c_str(), "Enter ReduceMax proto inferfunction!");
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
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(ReduceMax, ReduceMaxInferShape);
// ----------------ReduceMax END-------------------

// ----------------ReduceMaxD Op-------------------
IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(ReduceMaxDInferShape)
  OP_LOGI(op.GetName().c_str(), "Enter ReduceMaxD proto inferfunction!");
  ge::TensorDesc result_desc;
  if (!InferReduceDShape(op, "x", "axes", "keep_dims", result_desc)) {
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
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(ReduceMaxD, ReduceMaxDInferShape);
// ----------------ReduceMaxD END-------------------

// ----------------ReduceMin Op-------------------
IMPLEMT_COMMON_INFERFUNC(ReduceMinInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ReduceMin proto inferfunction!");
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

COMMON_INFER_FUNC_REG(ReduceMin, ReduceMinInferShape);
// ----------------ReduceMin END-------------------

// ----------------ReduceMinD Op-------------------
IMPLEMT_COMMON_INFERFUNC(ReduceMinDInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ReduceMinD proto inferfunction!");
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
  auto inputTensorDesc = op.GetInputDesc("x");
  auto shape = inputTensorDesc.GetShape();
  std::vector<int64_t> dims_input;
  dims_input = shape.GetDims();
  std::vector<int64_t> dimVector;
  int64_t dimNum = shape.GetDimNum();

  for (int64_t item = 0; item < dimNum; ++item) {
    if (item == 2 || item == 3) {
      dimVector.push_back(1);
    } else {
      dimVector.push_back(dims_input[item]);
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

COMMON_INFER_FUNC_REG(INTrainingReduceV2, INTrainingReduceV2InferShape);
// ----------------------INTrainingReduceV2 End--------------------------

// -------------------INTrainingUpdateV2--------------------
IMPLEMT_COMMON_INFERFUNC(INTrainingUpdateV2InferShape) {
  auto shape = op.GetInputDesc("x").GetShape();
  auto output_dtype = op.GetInputDesc("x").GetDataType();

  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(shape);
  td.SetDataType(output_dtype);
  op.UpdateOutputDesc("y", td);

  auto shape_scale = op.GetInputDesc("sum").GetShape();
  auto output_dtype_scale = op.GetInputDesc("sum").GetDataType();

  TensorDesc td_mean = op.GetOutputDesc("batch_mean");
  td_mean.SetShape(shape_scale);
  td_mean.SetDataType(output_dtype_scale);
  op.UpdateOutputDesc("batch_mean", td_mean);

  TensorDesc td_variance = op.GetOutputDesc("batch_variance");
  td_variance.SetShape(shape_scale);
  td_variance.SetDataType(output_dtype_scale);
  op.UpdateOutputDesc("batch_variance", td_variance);

  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(INTrainingUpdateV2, INTrainingUpdateV2InferShape);
// ------------------INTrainingUpdateV2 END---------------------

// ------------------------GNTrainingReduce--------------------------
IMPLEMT_COMMON_INFERFUNC(GNTrainingReduceInferShape) {
  auto inputTensorDesc = op.GetInputDesc("x");
  auto shape = inputTensorDesc.GetShape();

  Format input_format = inputTensorDesc.GetFormat();

  if (input_format != FORMAT_NHWC && input_format != FORMAT_NCHW) {
    OP_LOGE(op.GetName().c_str(),
            "data_format only "
            "support 'NHWC' and 'NCHW'.");
    return GRAPH_FAILED;
  }

  int64_t num = 2;
  if (GRAPH_SUCCESS != op.GetAttr("num_groups", num)) {
    OpsGetAttrErrReport(op.GetName(), "num_groups");
    OP_LOGE(op.GetName().c_str(), "Use default num_groups value!");
    op.SetAttr("num_groups", num);
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
    OP_LOGE(op.GetName().c_str(),
            "data_format only "
            "support 'NHWC' and 'NCHW'.");
    return GRAPH_FAILED;
  }

  int64_t num = 2;
  if (GRAPH_SUCCESS != op.GetAttr("num_groups", num)) {
    OpsGetAttrErrReport(op.GetName(), "num_groups");
    OP_LOGE(op.GetName().c_str(), "Use default num_groups value!");
    op.SetAttr("num_groups", num);
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
// ------------------GNTrainingUpdate END---------------------s
}  // namespace ge
