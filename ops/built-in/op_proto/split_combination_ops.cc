/**
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file split_combination_ops.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "inc/split_combination_ops.h"
#include <string>
#include <vector>
#include "util/util.h"
#include "op_log.h"
#include "./util/error_util.h"
#include "common/util/error_manager/error_manager.h"
namespace ge
{
// ----------------Split OP Begin-------------------
static void CalcSplit(const Tensor& data, const DataType& dtype,
                      std::vector<int64_t>& const_vec) {
  const uint8_t* constData = data.GetData();
  size_t size = data.GetSize() / sizeof(int32_t);
  for (size_t i = 0; i < size; ++i) {
    const_vec.push_back(*((int32_t*)constData));
  }
}

IMPLEMT_INFERFUNC(Split, SplitInferShape) {
  Tensor data;
  if (op.GetInputConstData("split_dim", data) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get constValue failed of [split_dim]");
    return GRAPH_FAILED;
  }
  DataType dtype = op.GetInputDesc("split_dim").GetDataType();
  std::vector<int64_t> const_vec;
  CalcSplit(data, dtype, const_vec);

  auto tensordesc = op.GetInputDesc("x");
  auto shape = tensordesc.GetShape();
  DataType inputDtype = tensordesc.GetDataType();
  TensorDesc td = op.GetDynamicOutputDesc("y", 0);
  int64_t split_dim = const_vec[0];
  int64_t num_split;
  if (op.GetAttr("num_split", num_split) == GRAPH_FAILED) {
    OP_LOGE(op.GetName().c_str(), "get attr num_split failed");
  }
  if (split_dim < 0) {
      split_dim += shape.GetDimNum();
  }

  auto length = shape.GetDim(split_dim) / num_split;
  for (auto i = 0; i < num_split; ++i) {
      shape.SetDim(split_dim, length);
      td.SetShape(shape);
      td.SetDataType(inputDtype);
      op.UpdateDynamicOutputDesc("y", i, td);
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Split, SplitInferShape);
// ----------------Split OP End-------------------

// ----------------SplitD OP Begin-------------------
IMPLEMT_INFERFUNC(SplitD, SplitDInferShape) {
  auto tensordesc = op.GetInputDesc("x");
  auto shape = tensordesc.GetShape();
  DataType inputDtype = tensordesc.GetDataType();
  TensorDesc td = op.GetDynamicOutputDesc("y", 0);
  int64_t split_dim;
  if (op.GetAttr("split_dim", split_dim) == GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "split_dim");
    OP_LOGE(op.GetName().c_str(), "get attr split dim failed");
  }
  int64_t num_split;
  if (op.GetAttr("num_split", num_split) == GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "num_split");
    OP_LOGE(op.GetName().c_str(), "get attr num_split failed");
  }
  int64_t dim_num = shape.GetDimNum();
  if ((split_dim < -dim_num) || (split_dim >= dim_num)){
    OpsInputShapeDimErrReport(op.GetName(), "Axis", Strcat(dim_num), Strcat(-dim_num), Strcat(split_dim));
    OP_LOGE(op.GetName().c_str(), "Axis value out of range");
    return GRAPH_FAILED;
  }
  if (num_split < 1){
      string excepted_value = Strcat("in range[1,]");
      OpsAttrValueErrReport(op.GetName(), "num_split", excepted_value, Strcat(num_split));
      OP_LOGE(op.GetName().c_str(), "num_split need greater than or equals to 1");
      return GRAPH_FAILED;
  }
  if (split_dim < 0) {
    split_dim += shape.GetDimNum();
  }

  auto length = shape.GetDim(split_dim) / num_split;
  for (auto i = 0; i < num_split; ++i) {
    shape.SetDim(split_dim, length);
    td.SetShape(shape);
    td.SetDataType(inputDtype);
    op.UpdateDynamicOutputDesc("y", i, td);
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SplitD, SplitDInferShape);
// ----------------SplitD OP End-------------------

// ----------------SplitV OP Begin-------------------
static void CalcSplitV(const Tensor& data, const DataType& dtype,
                       std::vector<int64_t>& const_vec) {
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
  }else {
    DataType dtype1 = op.GetInputDesc("size_splits").GetDataType();
    std::vector<int64_t> const_vec1;
    CalcSplitV(data1, dtype1, const_vec1);

    int64_t split_dim = const_vec2[0];
    std::vector<int64_t> size_splits(const_vec1);

    int64_t num_split;
    if (op.GetAttr("num_split", num_split) == GRAPH_FAILED) {
    OP_LOGE(op.GetName().c_str(), "get attr num_split failed");
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
  if ((split_dim < -dim_num) || (split_dim >= dim_num)){
    OpsInputShapeDimErrReport(op.GetName(), "Axis", Strcat(dim_num), Strcat(-dim_num), Strcat(split_dim));
    OP_LOGE(op.GetName().c_str(), "Axis value out of range");
    return GRAPH_FAILED;
  }
  if (num_split < 1){
      string excepted_value = Strcat("in range[1,]");
      OpsAttrValueErrReport(op.GetName(), "num_split", excepted_value, Strcat(num_split));
      OP_LOGE(op.GetName().c_str(), "num_split need greater than or equals to 1");
      return GRAPH_FAILED;
  }
  if (split_dim < 0) {
    split_dim += shape.GetDimNum();
  }
  vector<int64_t> adjust_size_splits;
  if(size_splits.size()==0){
    int64_t dim = shape.GetDim(split_dim);
    int64_t batch = dim/num_split;
    for (auto i = 0; i < num_split; ++i) {
      shape.SetDim(split_dim, batch);
      td.SetShape(shape);
      td.SetDataType(inputDtype);
      op.UpdateDynamicOutputDesc("y", i, td);
      adjust_size_splits.push_back(batch);
    }
  }else if(int(size_splits.size()+1)==num_split){
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
    if(dim - sum > 0){
      shape.SetDim(split_dim, dim - sum);
      td.SetShape(shape);
      td.SetDataType(inputDtype);
      op.UpdateDynamicOutputDesc("y", size_splits.size(), td);
      adjust_size_splits.push_back(dim - sum);
    }
  }
  else
  {
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
IMPLEMT_COMMON_INFERFUNC(ConcatV2DInferShape) {
  auto tensordesc = op.GetDynamicInputDesc("x",0);
  int64_t axis;
  int64_t num_concatext2;
  if (op.GetAttr("concat_dim", axis) == GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(),"axis" );
    OP_LOGE(op.GetName().c_str(), "get attr axis failed");
  }
  if (op.GetAttr("N", num_concatext2) == GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "N");
    OP_LOGE(op.GetName().c_str(), "get attr N failed");
  }
  auto axis1 = axis;
  auto shape = tensordesc.GetShape();
  int64_t dim_num = shape.GetDimNum();
  if ((axis1 < -dim_num) || (axis1 >= dim_num)){
    OpsInputShapeDimErrReport(op.GetName(), "axis", Strcat(dim_num), Strcat(-dim_num), Strcat(axis1));
    OP_LOGE(op.GetName().c_str(), "Axis value out of range");
    return GRAPH_FAILED;
  }
  if (axis1 < 0) {
    axis1 += shape.GetDimNum();
  }
  vector<int64_t> first_shape = shape.GetDims();
  vector<int64_t> shape_list;
  for (int32_t i = 0; i < num_concatext2; i++) {
    shape_list = op.GetDynamicInputDesc("x",i).GetShape().GetDims();
    for (int32_t j = 0; j < dim_num; j++) {
      if ((shape_list[j] != first_shape[j]) and (axis1 != j)){
        map<string, string> err_map;
        err_map["opname"] = "ConcatV2D";
        err_map["err_msg"] = "All axes must be equal except merge axis,check your shape!";
        std::string report_error_code = "E35003";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
      }
    }
  }

  int32_t size = 0;
  for (int32_t i = 0; i < num_concatext2; i++) {
    size += op.GetDynamicInputDesc("x",i).GetShape().GetDim(axis1);
  }
  shape.SetDim(axis1, size);
  DataType input_dtype = op.GetDynamicInputDesc("x",0).GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(ge::Shape(shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ConcatV2D, ConcatV2DInferShape);
// ----------------ConcatV2D OP End-------------------

// ----------------ParallelConcat OP Begin-------------------
static std::vector<int64_t> GetAttrValue(const ge::Operator &op,
                                   const std::string &key_name)
{
    std::vector<int64_t> list;
    if (ge::GRAPH_SUCCESS != op.GetAttr(key_name, list))
    {
      OpsGetAttrErrReport(op.GetName(), key_name);
      OP_LOGE(op.GetName().c_str(),"GetOpAttr ConstValue failed!");
    }
  return list;
}

static bool CheckListEmpty(const std::string& opName, const std::vector<int64_t>& list, const std::string& attrName)
{
    if (list.empty())
    {
        OP_LOGE(opName.c_str(),"the %s is empty !", attrName.c_str());
        return false;
    }
  return true;
}

IMPLEMT_COMMON_INFERFUNC(ParallelConcatInferShape) {
  auto tensordesc = op.GetDynamicInputDesc("values",0);
  std::vector<int64_t> shape;
  shape = GetAttrValue(op, "shape");
  int64_t num_1;
  if (!CheckListEmpty(op.GetName(), shape, "shape"))
  {
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
  if (shape[0] != num_1 ) {
    string excepted_value = Strcat("equal to the num of N[", num_1, "]");
    OpsAttrValueErrReport(op.GetName(), "output_data's fisrt dim", excepted_value, Strcat(shape[0]));
    OP_LOGE(op.GetName().c_str(), "first dim of output shape must"
              "be equal to the num of input tensors.");
    return GRAPH_FAILED;
  }
  for (int64_t i = 1; i < dimnum ; i++){
    if (x_shape.GetDim(i) != shape[i]) {
      string excepted_value = Strcat("match the output_data's shape[", shape[i], "]");
      OpsAttrValueErrReport(op.GetName(), "values's shape", excepted_value, Strcat(x_shape.GetDim(i)));
      OP_LOGE(op.GetName().c_str(), "the input shape"
              "do not match the output shape.");
      return GRAPH_FAILED;
    }
  }
  DataType input_dtype = op.GetDynamicInputDesc("values",0).GetDataType();
  TensorDesc outDesc = op.GetOutputDesc("output_data");
  std::string name_out = outDesc.GetName();
  outDesc.SetShape(ge::Shape(shape));
  outDesc.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("output_data", outDesc);
  OP_LOGI(op.GetName().c_str(), "input shape attr is: %s, set output shape :%s, Obtain REAL OUTPUT SHAPE is %s",
                                        to_string(ge::Shape(shape)).c_str(),
                                        to_string(ge::Shape(outDesc.GetShape())).c_str(),
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
      string excepted_value = Strcat("same as N[", static_cast<uint64_t>(num), "]");
      OpsAttrValueErrReport(op.GetName(), "values's size", excepted_value, Strcat(op.GetInputsSize()));
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
  auto tensordesc = op.GetDynamicInputDesc("x",0);
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
  auto axis = concat_dim;
  int64_t dim_num = shape.GetDimNum();
  if (axis < -dim_num||axis >= dim_num){
    OpsInputShapeDimErrReport(op.GetName(), "axis", Strcat(dim_num), Strcat(-dim_num), Strcat(axis));
    OP_LOGE(op.GetName().c_str(), "Axis value out of range");
    return GRAPH_FAILED;
  }
  if (axis < 0) {
      axis += shape.GetDimNum();
  }
  vector<int64_t> first_shape = shape.GetDims();
  vector<int64_t> shape_list;
  for (int32_t i = 0; i < num_concat; i++) {
    shape_list = op.GetDynamicInputDesc("x",i).GetShape().GetDims();
    for (int32_t j = 0; j < dim_num; j++) {
      if ((shape_list[j] != first_shape[j]) and (axis != j)){
        map<string, string> err_map;
        err_map["opname"] = "ConcatD";
        err_map["err_msg"] = "All axes must be equal except merge axis,check your shape!";
        std::string report_error_code = "E35003";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
      }
    }
  }

  int32_t size = 0;
  for (int32_t i = 0; i < num_concat; i++) {
      size += op.GetDynamicInputDesc("x",i).GetShape().GetDim(axis);
  }
  shape.SetDim(axis, size);
  DataType input_dtype = op.GetDynamicInputDesc("x",0).GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(ge::Shape(shape));
  td.SetDataType(input_dtype);
 (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ConcatD, ConcatDInferShape);
// ----------------ConcatD OP End-------------------

// ----------------Concat OP Begin-------------------
static void CalcConcat(const Tensor& data, const DataType& dtype,
                      std::vector<int64_t>& const_vec) {
  const uint8_t* constData = data.GetData();
  if (dtype == ge::DT_INT32) {
    const_vec.push_back(*((int32_t*)constData));
  } else {
    const_vec.push_back(*((int64_t*)constData));
  }
}

IMPLEMT_COMMON_INFERFUNC(ConcatInferShape) {
  auto tensordesc = op.GetDynamicInputDesc("x",0);
  auto shape = tensordesc.GetShape();
  int64_t dimnum;
  dimnum = shape.GetDimNum();

  Tensor data;
  if (op.GetInputConstData("concat_dim", data) != GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "Get constValue failed of [concat_dim]");
    vector<int64_t> dimVector;
    for (int64_t i = 0; i < dimnum ; i++) {
      dimVector.push_back(-1);
    }
    Shape x_shape(dimVector);
    TensorDesc y_desc = op.GetOutputDesc("output_data");
    y_desc.SetShape(ge::Shape(x_shape));
    DataType input_dtype = tensordesc.GetDataType();
    y_desc.SetDataType(input_dtype);
    (void)op.UpdateOutputDesc("output_data", y_desc);
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
  int64_t N;
  int32_t dim_value = 0;
  int32_t size = 0;
  int32_t dim_value1 = 0;

  if (op.GetAttr("N", N) == GRAPH_FAILED) {
    OP_LOGE(op.GetName().c_str(), "get attr N failed");
  }
  vector<int64_t> first_shape = shape.GetDims();
  vector<int64_t> shape_list;
  for (int32_t i = 0; i < N; i++) {
    shape_list = op.GetDynamicInputDesc("x",i).GetShape().GetDims();
    for (int32_t j = 0; j < dimnum; j++) {
      if ((shape_list[j] != first_shape[j]) and (axis != j)){
        map<string, string> err_map;
        err_map["opname"] = "ConcatD";
        err_map["err_msg"] = "All axes must be equal except merge axis,check your shape!";
        std::string report_error_code = "E35003";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
      }
    }
  }


  DataType input_dtype = op.GetDynamicInputDesc("x",0).GetDataType();
  TensorDesc td = op.GetOutputDesc("y");

  vector<int64_t> dimVector1;
  for (int64_t m = 0; m < dimnum; m++ ) {
    dim_value1 = op.GetDynamicInputDesc("x",0).GetShape().GetDim(m);
    if (dim_value1 == -1){
    dimVector1.push_back(-1);
    } else {
      dimVector1.push_back(dim_value1);
    }
  }

  for (int32_t i = 0; i < N; i++) {
      for (int64_t m = 0; m < dimnum; m++ ) {
        dim_value = op.GetDynamicInputDesc("x",i).GetShape().GetDim(m);
        if(dim_value == -1 && m == axis){
          shape.SetDim(axis, -1);
          td.SetShape(shape);
          td.SetDataType(input_dtype);
          (void)op.UpdateOutputDesc("y", td);
          return GRAPH_SUCCESS;
      } else if (dim_value == -1 && m != axis){
        Shape x_shape1(dimVector1);
        TensorDesc y_desc = op.GetOutputDesc("output_data");
        y_desc.SetShape(ge::Shape(x_shape1));
        DataType input_dtype = tensordesc.GetDataType();
        y_desc.SetDataType(input_dtype);
        (void)op.UpdateOutputDesc("output_data", y_desc);
        return GRAPH_SUCCESS;
      }
      }
     size += op.GetDynamicInputDesc("x",i).GetShape().GetDim(axis);
  }
  shape.SetDim(axis, size);
  td.SetShape(shape);
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Concat, ConcatInferShape);
// ----------------Concat OP End-------------------

// ----------------ConcatV2 OP Begin-------------------
static void CalcConcatv2(const Tensor& data, const DataType& dtype,
                         std::vector<int64_t>& const_vec) {
  const uint8_t* constData = data.GetData();
  if (dtype == ge::DT_INT32) {
    const_vec.push_back(*((int32_t*)constData));
  } else {
    const_vec.push_back(*((int64_t*)constData));
  }
}

IMPLEMT_COMMON_INFERFUNC(ConcatV2InferShape) {
  auto tensordesc = op.GetDynamicInputDesc("x",0);
  auto shape = tensordesc.GetShape();
  int64_t dimnum;
  dimnum = shape.GetDimNum();
  Tensor data;
  if (op.GetInputConstData("concat_dim", data) != GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "Get constValue failed of [concat_dim]");

    vector<int64_t> dimVector;
    for (int64_t i = 0; i < dimnum ; i++) {
      dimVector.push_back(-1);
    }
    Shape x_shape(dimVector);
    TensorDesc y_desc = op.GetOutputDesc("output_data");
    y_desc.SetShape(ge::Shape(x_shape));
    DataType input_dtype = tensordesc.GetDataType();
    y_desc.SetDataType(input_dtype);
    (void)op.UpdateOutputDesc("output_data", y_desc);
    return GRAPH_SUCCESS;
  }
  DataType dtype = op.GetInputDesc("concat_dim").GetDataType();
  std::vector<int64_t> const_vec;
  CalcConcatv2(data, dtype, const_vec);
  int64_t axis = const_vec[0];
  int64_t N;
  if (op.GetAttr("N", N) == GRAPH_FAILED) {
    OP_LOGE(op.GetName().c_str(), "get attr N failed");
  }
  int64_t axis1 = axis;
  if (axis1 < 0) {
    axis1 += shape.GetDimNum();
  }
  vector<int64_t> first_shape = shape.GetDims();
  vector<int64_t> shape_list;
  for (int32_t i = 0; i < N; i++) {
    shape_list = op.GetDynamicInputDesc("x",i).GetShape().GetDims();
    for (int32_t j = 0; j < dimnum; j++) {
      if ((shape_list[j] != first_shape[j]) and (axis1 != j)){
        map<string, string> err_map;
        err_map["opname"] = "ConcatV2D";
        err_map["err_msg"] = "All axes must be equal except merge axis,check your shape!";
        std::string report_error_code = "E35003";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
      }
    }
  }

  int32_t size = 0;
  int32_t dim_value = 0;
  int32_t dim_value1 = 0;
  DataType input_dtype = op.GetDynamicInputDesc("x",0).GetDataType();
  TensorDesc td = op.GetOutputDesc("y");

  vector<int64_t> dimVector1;
  for (int64_t m = 0; m < dimnum; m++ ) {
    dim_value1 = op.GetDynamicInputDesc("x",0).GetShape().GetDim(m);
    if (dim_value1 == -1){
    dimVector1.push_back(-1);
    } else {
      dimVector1.push_back(dim_value1);
    }
  }

  for (int32_t i = 0; i < N; i++) {
      for (int32_t m = 0; m < dimnum; m++ ) {
        dim_value = op.GetDynamicInputDesc("x",i).GetShape().GetDim(m);
        if(dim_value == -1 && m == axis1){
          shape.SetDim(axis1, -1);
          td.SetShape(shape);
          td.SetDataType(input_dtype);
          (void)op.UpdateOutputDesc("y", td);
          return GRAPH_SUCCESS;
      } else if (dim_value == -1 && m != axis1){
          Shape x_shape1(dimVector1);
          TensorDesc y_desc = op.GetOutputDesc("output_data");
          y_desc.SetShape(ge::Shape(x_shape1));
          DataType input_dtype = tensordesc.GetDataType();
          y_desc.SetDataType(input_dtype);
          (void)op.UpdateOutputDesc("output_data", y_desc);
          return GRAPH_SUCCESS;
      }
      }
     size += op.GetDynamicInputDesc("x",i).GetShape().GetDim(axis1);
  }
  shape.SetDim(axis1, size);
  td.SetShape(shape);
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ConcatV2, ConcatV2InferShape);
// ----------------ConcatV2 OP End-------------------

// ----------------Pack OP Begin-------------------
IMPLEMT_COMMON_INFERFUNC(PackInferShape) {
  auto ge_tensor_desc = op.GetDynamicInputDesc("x",0);
  auto shape = ge_tensor_desc.GetShape();
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
      OpsAttrValueErrReport(op.GetName(), "N", "greater than or equals to 1", Strcat(pack_num));
      OP_LOGE(op.GetName().c_str(), "N is out of range");
  }
  if (axis < (-dimnum-1) || axis > dimnum) {
    string correct_value = Strcat("in range [", -dimnum-1, ", ", dimnum, "]");
    AttrValueErrReport("axis", op.GetName(), Strcat(axis), correct_value);
    OP_LOGE(op.GetName().c_str(), "attr axis is not in range");
    return GRAPH_FAILED;
  }
  if (axis < 0) {
    axis += (dimnum+1);
  }
  vector<int64_t> dimVector;
  for (int64_t i = 0; i < dimnum+1 ; i++){
  if (i < axis){
    dimVector.push_back(shape.GetDim(i));
  }
  else if (i == axis){
    dimVector.push_back(pack_num);
  }
  else {
    dimVector.push_back(shape.GetDim(i-1));
  }
  }
  Shape x_shape(dimVector);
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(ge::Shape(x_shape));
  DataType input_dtype = ge_tensor_desc.GetDataType();
  y_desc.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Pack, PackInferShape);
// ----------------Pack OP End-------------------

// --------------------ConcatOffset------------------------
IMPLEMT_COMMON_INFERFUNC(ConcatOffsetInferShape) {
  DataType input_dtype = op.GetDynamicInputDesc("x",0).GetDataType();
  Shape shape = op.GetDynamicInputDesc("x",0).GetShape();
  auto tensordesc = op.GetDynamicInputDesc("x",0);
  int num_concat;
  op.GetAttr("N",num_concat);
  if (num_concat < 2) {
    OP_LOGE(op.GetName().c_str(), "The num_concat should be no less than two");
    return GRAPH_FAILED;
  }
  tensordesc.SetShape(shape);
  tensordesc.SetDataType(input_dtype);
  for (auto i = 0; i < num_concat; i++) {
    op.UpdateDynamicOutputDesc("y", i, tensordesc);
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ConcatOffset, ConcatOffsetInferShape);
// --------------------ConcatOffset------------------------

// --------------------ConcatOffsetD Op Begin------------------------
IMPLEMT_COMMON_INFERFUNC(ConcatOffsetDInferShape) {
  DataType input_dtype = op.GetDynamicInputDesc("x",0).GetDataType();
  Shape shape = op.GetDynamicInputDesc("x",0).GetShape();
  auto tensordesc = op.GetDynamicInputDesc("x",0);
  int num_concat;
  op.GetAttr("N",num_concat);
  if (num_concat < 2) {
    OpsAttrValueErrReport(op.GetName(), "num_concat", "no less than two", Strcat(num_concat));
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
