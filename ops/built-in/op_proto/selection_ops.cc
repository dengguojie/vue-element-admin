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
 * @file selection_ops.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include "inc/selection_ops.h"
#include <string>
#include <vector>
#include "util/util.h"
#include "op_log.h"
#include <math.h>
#include "./util/error_util.h"

namespace ge {
static bool CheckListEmpty(const std::string& opName,
                           const std::vector<int64_t>& list,
                           const std::string& attrName) {
  if (list.empty()) {
    OP_LOGE(opName.c_str(), "the %s is empty !", attrName.c_str());
    return false;
  }
  return true;
}
static std::vector<int64_t> GetAttrValue(const ge::Operator& op,
                                         const std::string& key_name) {
  std::vector<int64_t> list;
  if (ge::GRAPH_SUCCESS != op.GetAttr(key_name, list)) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue failed!");
  }
  return list;
}
// ----------------StridedSliceGradD Op Begin-------------------
static graphStatus GetStridedSliceGradValue(const ge::Operator& op,
                                            const std::string& keyName,
                                            vector<int32_t>& multiples) {
  if (ge::GRAPH_SUCCESS != op.GetAttr(keyName, multiples)) {
      OpsGetAttrErrReport(op.GetName(), keyName);
      OP_LOGE(op.GetName().c_str(),
            "Get const(%s) failed from op of 'StridedSliceGrad'!",
            keyName.c_str());
      return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(StridedSliceGradDInferShape) {
  ge::Shape outputShape = op.GetInputDesc("dy").GetShape();
  DataType input_dtype = op.GetInputDesc("dy").GetDataType();
  size_t dimNum = outputShape.GetDimNum();

  // get out shape list from const node
  std::vector<int32_t> outputShapeList;
  if (GRAPH_FAILED == GetStridedSliceGradValue(op, "shape", outputShapeList)) {
    return GRAPH_FAILED;
  }
  std::vector<int32_t> begin;
  if (GRAPH_FAILED == GetStridedSliceGradValue(op, "begin", begin)) {
    return GRAPH_FAILED;
  }
  if (begin.size() < 0 || begin.size() > 8) {
    OP_LOGE(op.GetName().c_str(), "begin size must be more than zero and less than eight!");
    return GRAPH_FAILED;
  }
  std::vector<int32_t> end;
  if (GRAPH_FAILED == GetStridedSliceGradValue(op, "end", end)) {
    return GRAPH_FAILED;
  }
  if (end.size() < 0 || end.size() > 8) {
    OP_LOGE(op.GetName().c_str(), "end size must be more than zero and less than eight!");
    return GRAPH_FAILED;
  }
  std::vector<int32_t> strides;
  if (GRAPH_FAILED == GetStridedSliceGradValue(op, "strides", strides)) {
    return GRAPH_FAILED;
  }
  if (strides.size() < 0 || strides.size() > 8) {
      OpsAttrValueErrReport(op.GetName(), "strides's size", "more than zero and less than eight", Strcat(strides.size()));
      OP_LOGE(op.GetName().c_str(), "strides size must be more than zero and less than eight!");
      return GRAPH_FAILED;
  }
  if (dimNum >= 1 && dimNum <= 8) {
    for (size_t i = 0; i < dimNum; i++) {
      outputShape.SetDim(i, outputShapeList[i]);
    }
  } else {
      OpsInputShapeDimErrReport(op.GetName(), "dy", "8", "1", Strcat(dimNum));
      OP_LOGE(op.GetName().c_str(),
            "The StridedSliceGrad dimension of the input shape is limited to 1"
            " or 8.");
      return GRAPH_FAILED;
  }

  TensorDesc tensordesc_output = op.GetOutputDesc("output");
  tensordesc_output.SetShape(outputShape);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("output", tensordesc_output);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(StridedSliceGradD, StridedSliceGradDInferShape);
// ----------------StridedSliceGradD Op End-------------------

// ----------------StridedSliceGrad Op Begin------------------
IMPLEMT_COMMON_INFERFUNC(StridedSliceGradInferShape) {
  DataType input_dtype = op.GetInputDesc("dy").GetDataType();

  Tensor output_shape_tensor;
  if (op.GetInputConstData("shape", output_shape_tensor) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get constValue failed of [shape]");
    return GRAPH_FAILED;
  }
  DataType dtype = op.GetInputDesc("shape").GetDataType();
  vector<int64_t> outputShapeList;
  GetConstValue(op, output_shape_tensor, dtype, outputShapeList);

  TensorDesc tensordesc_output = op.GetOutputDesc("output");
  ge::Shape outputShape = ge::Shape(outputShapeList);
  tensordesc_output.SetShape(outputShape);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("output", tensordesc_output);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(StridedSliceGrad, StridedSliceGradInferShape);
// ----------------StridedSliceGrad Op End------------------

// -----------------------Tile Op Begin----------------------------------
static void GetTileConstValue(const Operator& op, const Tensor& const_tensor,
                              const DataType& dtype,
                              std::vector<int64_t>& const_data) {
  size_t size = 0;
  if (dtype == ge::DT_INT32) {
    int32_t* const_data_ptr = (int32_t*) const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back((int32_t) ((*(const_data_ptr + i))));
    }
  } else if (dtype == ge::DT_INT64) {
    int64_t* const_data_ptr = (int64_t*) const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back(((int64_t) (*(const_data_ptr + i))));
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "not support this type");
  }
}

static graphStatus TileInferShapeAndType(ge::Operator& op, std::vector<int64_t>& multiples) {
  Shape shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  size_t shape_dim_num = shape.GetDimNum();
  std::vector<std::pair<int64_t,int64_t>> ori_shape_range;
  op.GetInputDesc("x").GetShapeRange(ori_shape_range);

  std::vector<std::pair<int64_t,int64_t>> out_range;
  // add for repeat when len(multiples) > len(shape)
  auto multiplesLen = multiples.size();
  if (shape_dim_num < multiples.size()) {
    vector<int64_t> ShapeList;
    auto lenEror = multiplesLen - shape_dim_num;
    for (size_t i = 0; i < lenEror; i++) {
      ShapeList.push_back(1);
      ori_shape_range.insert(ori_shape_range.begin(), std::make_pair(1, 1));
    }
    for (size_t i = 0; i < shape_dim_num; i++) {
      ShapeList.push_back(shape.GetDim(i));
    }
    shape = Shape(ShapeList);
    shape_dim_num = shape.GetDimNum();;
  }
  // add for repeat end

  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  if (shape_dim_num == 0) {
    tensordesc_output.SetShape(shape);
    tensordesc_output.SetDataType(input_dtype);
    tensordesc_output.SetShapeRange(ori_shape_range);
    (void) op.UpdateOutputDesc("y", tensordesc_output);
  } else if (shape_dim_num == 1) {
    if (shape.GetDim(0) == -1) {
      shape.SetDim(0, shape.GetDim(0));
      out_range.push_back(ori_shape_range[0]);
    } else {
      shape.SetDim(0, shape.GetDim(0) * multiples[0]);
      out_range.push_back(std::make_pair(shape.GetDim(0), shape.GetDim(0)));
    }
    tensordesc_output.SetShape(shape);
    tensordesc_output.SetDataType(input_dtype);
    tensordesc_output.SetShapeRange(out_range);
    (void) op.UpdateOutputDesc("y", tensordesc_output);
  } else if (shape_dim_num == DIM_SIZE2 || shape_dim_num == DIM_SIZE3 ||
             shape_dim_num == DIM_SIZE4 || shape_dim_num == DIM_SIZE5 ||
             shape_dim_num == DIM_SIZE6 || shape_dim_num == DIM_SIZE7 ||
             shape_dim_num == DIM_SIZE8) {
    if (shape_dim_num != multiples.size()) {
      OP_LOGE(op.GetName().c_str(), "the op tile or tile_d:"
                                    "shape_dim_num %lu != multiples size %lu",
              shape_dim_num, multiples.size());
      return GRAPH_FAILED;
    }
    const int64_t int32_max_num = pow(2, 31) - 1;
    for (size_t i = 0; i < shape_dim_num; i++) {
      if (shape.GetDim(i) == -1) {
        shape.SetDim(i, shape.GetDim(i));
        if (multiples[i] * ori_shape_range[i].first > int32_max_num) {
          OP_LOGE(op.GetName().c_str(), "the output shape left range is bigger than int32 max!");
          return GRAPH_FAILED;
        }
        if (multiples[i] * ori_shape_range[i].second > int32_max_num) {
          out_range.push_back(std::make_pair(multiples[i] * ori_shape_range[i].first, -1));
        } else {
          out_range.push_back(std::make_pair(multiples[i] * ori_shape_range[i].first,
          multiples[i] * ori_shape_range[i].second));
        }
      } else {
        shape.SetDim(i, shape.GetDim(i) * multiples[i]);
        out_range.push_back(std::make_pair(shape.GetDim(i), shape.GetDim(i)));
      }
    }
    tensordesc_output.SetShape(shape);
    tensordesc_output.SetDataType(input_dtype);
    tensordesc_output.SetShapeRange(out_range);
    (void) op.UpdateOutputDesc("y", tensordesc_output);
  } else {
    OP_LOGE(op.GetName().c_str(),
            "The tile op infer shape invalid shape size(%lu)\n",
            shape_dim_num);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(TileInferShape) {
  Tensor multiples_tensor;
  if (op.GetInputConstData("multiples", multiples_tensor) != GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "Get constValue failed of [multiples]");
    Shape shape = op.GetInputDesc("x").GetShape();
    size_t dim_num = shape.GetDimNum();
    DataType input_dtype = op.GetInputDesc("x").GetDataType();

    std::vector<int64_t> shape_vector;
    for (size_t item = 0; item < dim_num; ++item) {
      shape_vector.push_back(-1);
    }
    Shape input_shape(shape_vector);

    TensorDesc output_desc = op.GetOutputDesc("y");
    output_desc.SetShape(input_shape);
    output_desc.SetDataType(input_dtype);
    (void) op.UpdateOutputDesc("y", output_desc);

    return GRAPH_SUCCESS;
  }

  DataType dtype = op.GetInputDesc("multiples").GetDataType();
  std::vector<int64_t> multiples;
  GetTileConstValue(op, multiples_tensor, dtype, multiples);
  return TileInferShapeAndType(op, multiples);
}

COMMON_INFER_FUNC_REG(Tile, TileInferShape);
// -----------------------Tile Op end----------------------------------

// -----------------------TileD Op Begin----------------------------------
vector<int64_t> GetTileDConstValue(const ge::Operator& op,
                                   const std::string& key_name) {
  std::vector<int64_t> multiples;
  if (ge::GRAPH_SUCCESS != op.GetAttr(key_name, multiples)) {
    OpsGetAttrErrReport(op.GetName(), key_name);
    OP_LOGE(op.GetName().c_str(), "The tile op GetOpAttr failed!");
  }
  return multiples;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(TileDInferShape) {
  std::vector<int64_t> multiples;
  multiples = GetTileDConstValue(op, "multiples");
  if (multiples.empty()) {
    OP_LOGE(op.GetName().c_str(), "The tile op GetOpAttr"
                                  "ConstValue Value is empty!");
    return GRAPH_FAILED;
  }
  return TileInferShapeAndType(op, multiples);
}
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(TileD, TileDInferShape);
// -----------------------TileD Op end----------------------------------

// -----------------------range Op Begin----------------------------------
static void GetRangeConstValue(const Operator& op, const Tensor& const_tensor,
                               const DataType& dtype,
                               std::vector<float>& const_data) {
  size_t size = 0;
  if (dtype == ge::DT_INT32) {
    int32_t* const_data_ptr = (int32_t*) const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(int32_t);
    for (size_t i =0; i < size; i++) {
      const_data.push_back((int32_t) ((*(const_data_ptr + i))));
    }
  }
  else if (dtype == ge::DT_FLOAT) {
    float* const_data_ptr = (float*) const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(float);
    for (size_t i =0; i < size; ++i) {
      const_data.push_back((float) ((*(const_data_ptr + i))));
    }
  }
  else if (dtype == ge::DT_INT64) {
    int64_t* const_data_ptr = (int64_t*) const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(int64_t);
    for (size_t i =0; i < size; ++i) {
      const_data.push_back((int64_t) ((*(const_data_ptr + i))));
    }
  }
  else {
    OP_LOGE(op.GetName().c_str(), "not support this type");
  }
}

IMPLEMT_COMMON_INFERFUNC(RangeInferShape) {
  Tensor input_start_tensor;
  Tensor input_limit_tensor;
  Tensor input_delta_tensor;
  std::vector<float> start_multiples;
  std::vector<float> limit_multiples;
  std::vector<float> delta_multiples;
  std::vector<int64_t> dimsIn;
  std::vector<std::string> input_infer_depends = {"start", "delta", "limit"};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends(input_infer_depends);

  GeTensorDescPtr y_output = op_desc->MutableOutputDesc(0);
  GeTensorDescPtr start_desc = op_desc->MutableInputDesc(0);
  GeTensorDescPtr limit_desc = op_desc->MutableInputDesc(1);
  GeTensorDescPtr delta_desc = op_desc->MutableInputDesc(2);
  if ((op.GetInputConstData("start", input_start_tensor) != GRAPH_SUCCESS) ||
      (op.GetInputConstData("delta", input_delta_tensor) != GRAPH_SUCCESS) ||
      (op.GetInputConstData("limit", input_limit_tensor) != GRAPH_SUCCESS)) {
    OP_LOGI(op.GetName().c_str(), "Get constValue failed of in [start], [delta],[limit]");
    dimsIn.emplace_back(UNKNOWN_DIM);
    y_output->SetShape(GeShape(dimsIn));
    y_output->SetOriginShape(GeShape(dimsIn));
    y_output->SetShapeRange({std::make_pair(1, -1)});
    DataType start_dtype = start_desc->GetDataType();
    DataType limit_dtype = limit_desc->GetDataType();
    DataType delta_dtype = delta_desc->GetDataType();
    if (start_dtype == ge::DT_INT32 && limit_dtype == ge::DT_INT32 &&
        delta_dtype == ge::DT_INT32) {
      y_output->SetDataType(ge::DT_INT32);
    }
    else if (start_dtype == ge::DT_INT64 && limit_dtype == ge::DT_INT64 &&
        delta_dtype == ge::DT_INT64) {
      y_output->SetDataType(ge::DT_INT64);
    }
    else {
      y_output->SetDataType(ge::DT_FLOAT);
    }

    return GRAPH_SUCCESS;
  } else {
    DataType start_dtype = start_desc->GetDataType();
    DataType limit_dtype = limit_desc->GetDataType();
    DataType delta_dtype = delta_desc->GetDataType();
    GetRangeConstValue(op, input_start_tensor, start_dtype, start_multiples);
    GetRangeConstValue(op, input_limit_tensor, limit_dtype, limit_multiples);
    GetRangeConstValue(op, input_delta_tensor, delta_dtype, delta_multiples);
    if (start_multiples.empty() || limit_multiples.empty() || delta_multiples.empty()) {
      OP_LOGW(op.GetName().c_str(), "the start_multiples_size is %d, the limit_multiples_size is %d,"
              "the delta_multiples_size is %d", start_multiples.size(), limit_multiples.size(),
              delta_multiples.size());

      y_output->SetShape(GeShape({UNKNOWN_DIM}));
      y_output->SetOriginShape(GeShape({UNKNOWN_DIM}));
      y_output->SetShapeRange({std::make_pair(1, -1)});

      return GRAPH_SUCCESS;
    }

    float assist_num = abs(limit_multiples[0] - start_multiples[0]);
    float assist_num_one = abs(delta_multiples[0]);
    int64_t res = 0;
    DataType input_dtype = ge::DT_FLOAT;
    if (start_dtype == ge::DT_INT32 && limit_dtype == ge::DT_INT32 &&
        delta_dtype == ge::DT_INT32) {
      res = int(ceil(float(assist_num)/ assist_num_one));
    }
    else if (start_dtype == ge::DT_INT64 && limit_dtype == ge::DT_INT64 &&
        delta_dtype == ge::DT_INT64) {
      res = int(ceil(float(assist_num)/ assist_num_one));
    }
    else {
      res = ceil(assist_num / assist_num_one);
    }
    dimsIn.emplace_back(res);
    if (start_dtype == ge::DT_INT32 && limit_dtype == ge::DT_INT32 &&
        delta_dtype == ge::DT_INT32) {
      input_dtype = ge::DT_INT32;
    }
    else if (start_dtype == ge::DT_INT64 && limit_dtype == ge::DT_INT64 &&
        delta_dtype == ge::DT_INT64) {
      input_dtype = ge::DT_INT64;
    }
    else {
      input_dtype = ge::DT_FLOAT;
    }
    y_output->SetShape(GeShape(dimsIn));
    y_output->SetOriginShape(GeShape(dimsIn));
    y_output->SetDataType(input_dtype);
    return GRAPH_SUCCESS;
  }
}

COMMON_INFER_FUNC_REG(Range, RangeInferShape);
// -----------------------Range Op End----------------------------------

// -----------------------RangeD Op Begin----------------------------------
IMPLEMT_COMMON_INFERFUNC(RangeDInferShape) {
  float start;
  if (op.GetAttr("start", start) == ge::GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "start");
    OP_LOGE(op.GetName().c_str(),"GetOpAttr ConstValue start failed.");
    return GRAPH_FAILED;
  }
  float limit;
  if (op.GetAttr("limit", limit) == ge::GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "limit");
    OP_LOGE(op.GetName().c_str(),"GetOpAttr ConstValue limit failed.");
    return GRAPH_FAILED;
  }
  float delta;
  if (op.GetAttr("delta", delta) == ge::GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "delta");
    OP_LOGE(op.GetName().c_str(),"GetOpAttr ConstValue delta failed.");
    return GRAPH_FAILED;
  }
  if (limit == start) {
    string excepted_value = Strcat("not equal to limit[", limit, "]");
    OpsAttrValueErrReport(op.GetName(), "start", excepted_value, Strcat(start));
    OP_LOGE(op.GetName().c_str(), "start is not equal to limit");
    return GRAPH_FAILED;
  }
  if (delta == 0) {
    OpsAttrValueErrReport(op.GetName(), "delta", "not equal to zero", Strcat(delta));
    OP_LOGE(op.GetName().c_str(), "the input of delta is not equal to zero");
    return GRAPH_FAILED;
  }
  if (start > limit && delta > 0) {
    string excepted_value = Strcat("more than start[", start, "]");
    OpsAttrValueErrReport(op.GetName(), "limit", excepted_value, Strcat(limit));
    OP_LOGE(op.GetName().c_str(), "requires limit is more than start "
                                  "when delta is more than zero");
    return GRAPH_FAILED;
  }
  if (start < limit && delta < 0) {
    string excepted_value = Strcat("more than limit[", limit, "]");
    OpsAttrValueErrReport(op.GetName(), "start", excepted_value, Strcat(start));
    OP_LOGE(op.GetName().c_str(),"requires start is more than limit "
                                 "when delta is less than zero");
    return GRAPH_FAILED;
  }
  (void)op.UpdateOutputDesc("y", op.GetInputDesc("x"));
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(RangeD, RangeDInferShape);
// -----------------------RangeD Op End----------------------------------

//----------------GatherNd Op-------------------
bool CheckGatherNdInputIndicesSize(const Operator &op, const string &input_name) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto inputIndices = op_desc->MutableInputDesc("indices");
  auto indicesShape = inputIndices->GetShape();
  auto indicesShapeSize = indicesShape.GetDimNum();
  int indicesLastElement = indicesShape.GetDim(indicesShapeSize - 1);
  int indicesPart = 1;
  for (int i = 0; i < indicesLastElement - 1; ++i) {
    indicesPart = indicesPart*indicesShape.GetDim(i);
  }
  if (indicesPart > std::numeric_limits<int>::max()) {
    OpsInputShapeSizeErrReport(op.GetName(), "indices", Strcat(std::numeric_limits<int>::max()), Strcat(indicesPart));
    OP_LOGE(op.GetName().c_str(), "indices has too many elements "
            "for int indexing");
    return false;
  }
  return true;
}

bool CheckGatherNdParamsSize(const Operator &op, int last_dim, int shape_size) {
  if (last_dim > shape_size) {
    OP_LOGE(
        op.GetName().c_str(),
        "indices.shape[-1] must be <= params.rank, but %d and %d.",
        last_dim, shape_size);
    return false;
  }
  return true;
}

IMPLEMT_VERIFIER(GatherNd, GatherNdVerify) {
  if (CheckGatherNdInputIndicesSize(op, "indices") == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(GatherNdInferShape)
  GeTensorDescPtr inputTensorDesc = op_desc->MutableInputDesc("x");
  GeTensorDescPtr outputTensorDesc = op_desc->MutableOutputDesc("y");

  std::vector<std::pair<int64_t,int64_t>> shape_range_x;
  op_desc->MutableInputDesc("x")->GetShapeRange(shape_range_x);
  std::vector<std::pair<int64_t,int64_t>> shape_range_indices;
  op_desc->MutableInputDesc("indices")->GetShapeRange(shape_range_indices);
  std::vector<std::pair<int64_t,int64_t>> out_range;

  auto inputParams = op_desc->MutableInputDesc("x");
  auto inputIndices = op_desc->MutableInputDesc("indices");
  auto paramsShape = inputParams->GetShape();
  auto indicesShape = inputIndices->GetShape();
  auto paramsShapeSize = paramsShape.GetDimNum();
  int indicesShapeSize = indicesShape.GetDimNum();

  vector<int64_t> dimVec;
  vector<int64_t> paramsShapeVec = paramsShape.GetDims();
  vector<int64_t> indicesShapeVec = indicesShape.GetDims();

  MakeUpShapeRange(paramsShapeVec, shape_range_x);
  MakeUpShapeRange(indicesShapeVec, shape_range_indices);

  int indicesLastElement = -2;
  if (!IsUnknownRankShape(indicesShapeVec)) {
    indicesLastElement = indicesShape.GetDim(indicesShapeSize - 1);
  }
  DataType paramsType = inputParams->GetDataType();

  if (indicesLastElement == -1 ||
      indicesLastElement == -2 ||
      IsUnknownRankShape(paramsShapeVec)){
    dimVec.push_back(-2);
    if (shape_range_indices.size() == 0) {
      OP_LOGW(op.GetName().c_str(), "shape range of indices is null, output range can't infer");
    } else {
      for (size_t i=0; i<shape_range_indices.size()-1; i++) {
        out_range.push_back(shape_range_indices[i]);
      }

      std::pair<int64_t, int64_t> range_last = shape_range_indices[shape_range_indices.size() - 1];
      if (range_last.first == range_last.second) {
        indicesLastElement = range_last.second;
      }

      if (indicesLastElement > 0) {
        for (size_t i=indicesLastElement; i<shape_range_x.size(); i++) {
          out_range.push_back(shape_range_x[i]);
        }
      } else {
        for (size_t i=range_last.first; i<shape_range_x.size(); i++) {
          out_range.push_back(std::pair<int64_t,int64_t>(1, -1));
        }
      }
    }
  } else if (!CheckGatherNdParamsSize(
                  op,
                  indicesLastElement,
                  (int)paramsShapeSize)) {
    return GRAPH_FAILED;
  } else {
    for (int i = 0; i < indicesShapeSize - 1; ++i) {
      dimVec.push_back(indicesShape.GetDim(i));
      if ((size_t)i < shape_range_indices.size()) {
        out_range.push_back(shape_range_indices[i]);
      }
    }
    for (size_t i = indicesLastElement; i < paramsShapeSize; ++i) {
      dimVec.push_back(paramsShape.GetDim(i));
      if (i < shape_range_x.size()) {
        out_range.push_back(shape_range_x[i]);
      }
    }
  }

  ge::GeShape outputShape = ge::GeShape(dimVec);
  DataType outputDtype = paramsType;
  outputTensorDesc->SetShape(outputShape);
  outputTensorDesc->SetDataType(outputDtype);
  TensorUtils::SetRealDimCnt(*outputTensorDesc, dimVec.size());
  outputTensorDesc->SetShapeRange(out_range);
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(GatherNd, GatherNdInferShape);
VERIFY_FUNC_REG(GatherNd, GatherNdVerify);
// ----------------GatherNd END----------------

// ----------------GatherV2-------------------
static graphStatus GatherV2InferOptimize(
    ge::Operator& op, int32_t& axis,
    GeTensorDescPtr& x_desc,
    GeTensorDescPtr& indices_desc,
    GeTensorDescPtr& y_desc,
    std::vector<int64_t>& x_shape,
    std::vector<int64_t>& indices_shape,
    std::vector<int64_t>& y_shape,
    std::vector<std::pair<int64_t,int64_t>>& shape_range_x,
    std::vector<std::pair<int64_t,int64_t>>& shape_range_indices,
    std::vector<std::pair<int64_t,int64_t>>& out_range) {
  // real dim cnt has no existing meaning .Original shape has replace its meaning now
  int64_t x_real_dim_cnt = static_cast<int64_t>(x_desc->GetOriginShape().GetDims().size());

  if (IsUnknownRankShape(indices_shape) || IsUnknownRankShape(x_shape)) {
    y_shape.push_back(-2);

    y_desc->SetShape(ge::GeShape(y_shape));
    y_desc->SetDataType(x_desc->GetDataType());

    return GRAPH_SUCCESS;
  }

  if (x_real_dim_cnt < 1) {
    OP_LOGE(op.GetName().c_str(), "x_desc RealDimCnt[%d] not support.",
            x_real_dim_cnt);
    return GRAPH_FAILED;
  }

  if (axis < 0) {
    if (x_real_dim_cnt < -axis) {
      OP_LOGE(op.GetName().c_str(), "x_desc RealDimCnt[%d] < -axis[%d]",
              x_real_dim_cnt, -axis);
      return GRAPH_FAILED;
    }
  } else if (x_real_dim_cnt < axis + 1) {
      OP_LOGE(op.GetName().c_str(), "x_desc RealDimCnt[%d] < axis + 1[%d]",
              x_real_dim_cnt, axis + 1);
    return GRAPH_FAILED;
  }

  int64_t end = axis;
  if (end < 0) {
    end = x_real_dim_cnt + end;
    if (end < 0) {
      OP_LOGE(op.GetName().c_str(), "x_desc RealDimCnt[%d] < axis + 1[%d]",
              x_real_dim_cnt, axis + 1);
      return GRAPH_FAILED;
    }
  }

  for (int i = 0; i < end; i++) {
    y_shape.push_back(x_shape[i]);
    if ((size_t)i < shape_range_x.size()) {
      out_range.push_back(shape_range_x[i]);
    }
  }
  // real dim cnt has no existing meaning .Original shape has replace its meaning now
  auto indices_dim_cnt_unsigned = indices_desc->GetOriginShape().GetDims().size();
  for (size_t i = 0; i < indices_dim_cnt_unsigned; i++) {
    y_shape.push_back(indices_shape[i]);
    if ((size_t)i < shape_range_indices.size()) {
      out_range.push_back(shape_range_indices[i]);
    }
  }

  if (axis != -1) {
    int64_t start = axis + 1;
    int64_t rank = x_real_dim_cnt;
    if (start == 0) {
      OP_LOGE(op.GetName().c_str(), "start[%d] error.", start);
      return GRAPH_FAILED;
    }
    if (start > rank) {
      start = rank;
    }
    if (start < 0) {
      start = rank + start;
      if (start < 0) {
        OP_LOGE(op.GetName().c_str(), "start[%d], rank[%d], error.", start,
                rank);
        return GRAPH_FAILED;
      }
    }
    for (int i = start; i < rank; i++) {
      y_shape.push_back(x_shape[i]);
      if ((size_t)i < shape_range_x.size()) {
        out_range.push_back(shape_range_x[i]);
      }
    }
  }

  y_desc->SetShape(ge::GeShape(y_shape));
  y_desc->SetShapeRange(out_range);
  y_desc->SetDataType(x_desc->GetDataType());
  return GRAPH_SUCCESS;
}

void InferRangeOfUnknownRank(graphStatus result, int32_t& axis,
                             std::vector<std::pair<int64_t,int64_t>>& out_range,
                             std::vector<std::pair<int64_t,int64_t>>& shape_range_x,
                             std::vector<std::pair<int64_t,int64_t>>& shape_range_indices) {
  if (result == GRAPH_SUCCESS) {
    size_t axis_temp = axis>=0?axis:axis+shape_range_x.size();
    for (size_t i=0; i<axis_temp; i++) {
      if (i < shape_range_x.size()) {
        out_range.push_back(shape_range_x[i]);
      }
    }
    for (size_t i=0; i<shape_range_indices.size(); i++) {
      out_range.push_back(shape_range_indices[i]);
    }
    for (size_t i=axis_temp+1; i<shape_range_x.size(); i++) {
      if (i < shape_range_x.size()) {
        out_range.push_back(shape_range_x[i]);
      }
    }
  } else {
    out_range.push_back(std::pair<int64_t,int64_t>(1, -1));
    for (size_t i=0; i<shape_range_indices.size(); i++) {
      out_range.push_back(shape_range_indices[i]);
    }
    out_range.push_back(std::pair<int64_t,int64_t>(1, -1));
  }
  return;
}

IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(GatherV2InferShape)
  vector<string> input_infer_depends = {"axis"};
  op_desc->SetOpInferDepends(input_infer_depends);

  GeTensorDescPtr x_desc = op_desc->MutableInputDesc("x");
  GeTensorDescPtr indices_desc = op_desc->MutableInputDesc("indices");
  GeTensorDescPtr y_desc = op_desc->MutableOutputDesc("y");

  std::vector<int64_t> x_shape = x_desc->MutableShape().GetDims();
  std::vector<int64_t> indices_shape = indices_desc->MutableShape().GetDims();

  std::vector<int64_t> y_shape;

  std::vector<std::pair<int64_t,int64_t>> shape_range_x;
  op_desc->MutableInputDesc("x")->GetShapeRange(shape_range_x);
  std::vector<std::pair<int64_t,int64_t>> shape_range_indices;
  op_desc->MutableInputDesc("indices")->GetShapeRange(shape_range_indices);
  std::vector<std::pair<int64_t,int64_t>> out_range;

  MakeUpShapeRange(x_shape, shape_range_x);
  MakeUpShapeRange(indices_shape, shape_range_indices);

  Tensor axis_tensor;
  int32_t axis = -1;
  graphStatus result = op.GetInputConstData("axis", axis_tensor);
  if (result == GRAPH_SUCCESS)  {
    axis = (int32_t) (*((int32_t *)axis_tensor.GetData()));
  }

  // unknown rank
  if (IsUnknownRankShape(indices_shape) || IsUnknownRankShape(x_shape)) {
    y_shape.push_back(-2);

    // infer shape range
    // unknownshape input with no range can't infer output range
    if (shape_range_x.empty() || shape_range_indices.empty()) {
      OP_LOGW(op.GetName().c_str(),
          "Output range can't infer because that input shape range is empty.");
      y_desc->SetShape(ge::GeShape(y_shape));
      y_desc->SetDataType(x_desc->GetDataType());
    } else {
      InferRangeOfUnknownRank(result, axis, out_range, shape_range_x, shape_range_indices);
      y_desc->SetShape(ge::GeShape(y_shape));
      y_desc->SetDataType(x_desc->GetDataType());
      y_desc->SetShapeRange(out_range);
    }
  } else if (result != GRAPH_SUCCESS) {
    // unknown shape
    OP_LOGI(op.GetName().c_str(), "GetInputConstData(axis) [%d]", result);
    int64_t rank_x = static_cast<int64_t>(x_desc->GetOriginShape().GetDims().size());
    int64_t rank_indices = static_cast<int64_t>(indices_desc->GetOriginShape().GetDims().size());

    // infer shape range
    std::vector<std::pair<int64_t,int64_t>> range_tmp = shape_range_x;
    range_tmp.insert(range_tmp.end(), shape_range_indices.begin(), shape_range_indices.end());
    int64_t min_first{0}, max_second{0};
    for (size_t i=0; i < range_tmp.size(); i++) {
      if (i == 0) {
        min_first = range_tmp[i].first;
        max_second = range_tmp[i].second;
      }
      min_first = min_first < range_tmp[i].first ? min_first : range_tmp[i].first;
      max_second = max_second > range_tmp[i].second ? max_second : range_tmp[i].second;
    }

    for (int i = 0; i < rank_x + rank_indices - 1; i++) {
      y_shape.push_back(-1);
      out_range.push_back(std::pair<int64_t,int64_t>(min_first, max_second));
    }

    y_desc->SetDataType(x_desc->GetDataType());
    y_desc->SetShapeRange(out_range);
    y_desc->SetShape(ge::GeShape(y_shape));
  } else {
    if (GatherV2InferOptimize(op, axis, x_desc, indices_desc, y_desc, x_shape,
        indices_shape, y_shape, shape_range_x, shape_range_indices, out_range) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }
  }
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(GatherV2, GatherV2InferShape);
// ----------------GatherV2 END-------------------

// ----------------GatherV2D-----------------------
static graphStatus GatherV2InferShapeAndType(ge::Operator& op, int32_t& axis) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr indices_desc = op_desc->MutableInputDesc("indices");
  auto indices_shape = indices_desc->GetShape().GetDims();

  std::vector<std::pair<int64_t,int64_t>> shape_range_x;
  op_desc->MutableInputDesc("x")->GetShapeRange(shape_range_x);
  std::vector<std::pair<int64_t,int64_t>> shape_range_indices;
  op_desc->MutableInputDesc("indices")->GetShapeRange(shape_range_indices);
  std::vector<std::pair<int64_t,int64_t>> out_range;

  GeTensorDescPtr x_desc = op_desc->MutableInputDesc("x");
  std::vector<int64_t> y_shape;

  int64_t x_real_dim_cnt = static_cast<int64_t>(x_desc->GetOriginShape().GetDims().size());

  if (IsUnknownRank(op, "indices") || IsUnknownRank(op, "x")) {
    y_shape.push_back(-2);

    GeTensorDescPtr y_desc = op_desc->MutableOutputDesc("y");
    y_desc->SetShape(ge::GeShape(y_shape));
    y_desc->SetDataType(x_desc->GetDataType());

    return GRAPH_SUCCESS;
  }

  if (x_real_dim_cnt < 1) {
    OP_LOGE(op.GetName().c_str(), "x_desc RealDimCnt[%d] not support.",
            x_real_dim_cnt);
    return GRAPH_FAILED;
  }
  auto x_shape = x_desc->GetShape().GetDims();
  if (axis < 0) {
    if (x_real_dim_cnt < -axis) {
      OP_LOGE(op.GetName().c_str(), "x_desc RealDimCnt[%d] < -axis[%d]",
              x_real_dim_cnt, -axis);
      return GRAPH_FAILED;
    }
  } else if (x_real_dim_cnt < axis + 1) {
      OP_LOGE(op.GetName().c_str(), "x_desc RealDimCnt[%d] < axis + 1[%d]",
              x_real_dim_cnt, axis + 1);
    return GRAPH_FAILED;
  }

  int64_t end = axis;
  if (end < 0) {
    end = x_real_dim_cnt + end;
    if (end < 0) {
      OP_LOGE(op.GetName().c_str(), "x_desc RealDimCnt[%d] < axis + 1[%d]",
              x_real_dim_cnt, axis + 1);
      return GRAPH_FAILED;
    }
  }

  for (int i = 0; i < end; i++) {
    y_shape.push_back(x_shape[i]);
    if ((size_t)i < shape_range_x.size()) {
      out_range.push_back(shape_range_x[i]);
    }
  }
  auto indices_dim_cnt_unsigned = static_cast<int64_t>(indices_desc->GetOriginShape().GetDims().size());
  for (int i = 0; i < indices_dim_cnt_unsigned; i++) {
    y_shape.push_back(indices_shape[i]);
    if ((size_t)i < shape_range_indices.size()) {
      out_range.push_back(shape_range_indices[i]);
    }
  }

  if (axis != -1) {
    int64_t start = axis + 1;
    int64_t rank = x_real_dim_cnt;
    if (start == 0) {
      OP_LOGE(op.GetName().c_str(), "start[%d] error.", start);
      return GRAPH_FAILED;
    }
    if (start > rank) {
      start = rank;
    }
    if (start < 0) {
      start = rank + start;
      if (start < 0) {
        OP_LOGE(op.GetName().c_str(), "start[%d], rank[%d], error.", start,
                rank);
        return GRAPH_FAILED;
      }
    }
    for (int i = start; i < rank; i++) {
      y_shape.push_back(x_shape[i]);
      if ((size_t)i < shape_range_x.size()) {
        out_range.push_back(shape_range_x[i]);
      }
    }
  }

  GeTensorDescPtr y_desc = op_desc->MutableOutputDesc("y");
  y_desc->SetShape(ge::GeShape(y_shape));
  y_desc->SetShapeRange(out_range);
  y_desc->SetDataType(x_desc->GetDataType());
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(GatherV2DInferShape)
  GeTensorDescPtr x_desc = op_desc->MutableInputDesc("x");
  int32_t dimnum = 0;
  dimnum = x_desc->GetShape().GetDimNum();
  int32_t axis = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("axis", axis)) {
    OpsGetAttrErrReport(op.GetName(), "axis");
    OP_LOGE("Get const axis failed from op of 'GatherV2'!");
    return GRAPH_FAILED;
  }
  if (!IsUnknownRank(op, "x")) {
    if (axis < -dimnum || axis >= dimnum) {
      OpsInputShapeDimErrReport(op.GetName(), "axis", Strcat(dimnum), Strcat(-dimnum), Strcat(axis));
      OP_LOGE(op.GetName().c_str(), "attr axis is not in range");
      return GRAPH_FAILED;
    }
  }

  if (GatherV2InferShapeAndType(op, axis) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(GatherV2D, GatherV2DInferShape);
// ----------------GatherV2D END-------------------

// ----------------UnsortedSegmentSum-------------------
static void GetUnsortedSegmentSumConstValue(const Tensor& const_tensor,
    const DataType& dtype, int64_t& const_data) {
  if (dtype == ge::DT_INT32){
  int32_t *const_data_ptr = (int32_t *)const_tensor.GetData();
  const_data=(int32_t)((*(const_data_ptr + 0)));
  } else {
    int64_t *const_data_ptr = (int64_t *)const_tensor.GetData();
    const_data=(int64_t)(*(const_data_ptr + 0));
    }
}

IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(UnsortedSegmentSumInferShape)
  vector<string> input_infer_depends = {"num_segments"};
  op_desc->SetOpInferDepends(input_infer_depends);

  Tensor input_num_segments_tensor;
  int64_t input_num_segments;
  DataType input_num_segments_dtype = op_desc->MutableInputDesc("num_segments")->GetDataType();

  std::vector<std::pair<int64_t,int64_t>> shape_range_x;
  op_desc->MutableInputDesc("x")->GetShapeRange(shape_range_x);
  std::vector<std::pair<int64_t,int64_t>> shape_range_seg_id;
  op_desc->MutableInputDesc("segment_ids")->GetShapeRange(shape_range_seg_id);

  std::vector<std::pair<int64_t,int64_t>> out_range;

  if (GRAPH_SUCCESS != op.GetInputConstData("num_segments",
                                            input_num_segments_tensor)) {
    input_num_segments = -1;
    out_range.push_back(std::pair<int64_t,int64_t>(1, -1));
  } else {
    GetUnsortedSegmentSumConstValue(input_num_segments_tensor,
        input_num_segments_dtype, input_num_segments);
    out_range.push_back(std::pair<int64_t,int64_t>(input_num_segments, input_num_segments));
  }

  ge::GeShape shape = op_desc->MutableInputDesc("x")->GetShape();
  ge::GeShape shape_id = op_desc->MutableInputDesc("segment_ids")->GetShape();
  auto shape_vec = shape.GetDims();
  auto shape_id_vec = shape_id.GetDims();

  MakeUpShapeRange(shape_vec, shape_range_x);
  MakeUpShapeRange(shape_id_vec, shape_range_seg_id);

  int64_t dim_idsize_input = shape_id.GetDimNum();
  int64_t dim_size_input=shape.GetDimNum();
  DataType input_dtype = op_desc->MutableInputDesc("x")->GetDataType();
  vector<int64_t> shape_vector;
  if (IsUnknownRankShape(shape_vec) || IsUnknownRankShape(shape_id_vec)) {
    shape_vector.push_back(-2);
    for (size_t i=shape_range_seg_id.size(); i<shape_range_x.size(); i++) {
      out_range.push_back(shape_range_x[i]);
    }
  } else if (dim_idsize_input >1) {
    shape_vector.push_back(input_num_segments);
    for (int i = dim_idsize_input; i < dim_size_input; i++) {
      shape_vector.push_back(shape_vec[i]);
      if ((size_t)i < shape_range_x.size()) {
        out_range.push_back(shape_range_x[i]);
      }
    }
  } else {
    shape_vector = shape_vec;
    shape_vector[0] = input_num_segments;
    for (size_t i=1; i<shape_vector.size(); i++) {
      if (i < shape_range_x.size()) {
        out_range.push_back(shape_range_x[i]);
      }
    }
  }

  GeTensorDescPtr tensordesc_output = op_desc->MutableOutputDesc("y");
  ge::GeShape out_shape = ge::GeShape(shape_vector);
  tensordesc_output->SetShape(out_shape);
  tensordesc_output->SetDataType(input_dtype);
  tensordesc_output->SetShapeRange(out_range);

IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(UnsortedSegmentSum, UnsortedSegmentSumInferShape);
// ----------------UnsortedSegmentSum END----------------

// ----------------UnsortedSegmentSumD-------------------
IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(UnsortedSegmentSumDInferShape)
  int64_t input_num_segments;
  if (ge::GRAPH_SUCCESS != op.GetAttr("num_segments", input_num_segments)) {
      OpsGetAttrErrReport(op.GetName(), "num_segments");
      OP_LOGE(op.GetName().c_str(), "The num_segments"
    "op GetOpAttr ConstValue failed!");
  }
  if (input_num_segments <= 0) {
      OpsAttrValueErrReport(op.GetName(), "num_segments", "reater than 0", Strcat(input_num_segments));
      OP_LOGE(op.GetName().c_str(), "num_segments need greater than 0");
      return GRAPH_FAILED;
  }
  ge::GeShape shape = op_desc->MutableInputDesc("x")->GetShape();
  ge::GeShape shape_id = op_desc->MutableInputDesc("segment_ids")->GetShape();
  int64_t dim_idsize_input = shape_id.GetDimNum();
  int64_t dim_size_input=shape.GetDimNum();
  DataType input_dtype = op_desc->MutableInputDesc("x")->GetDataType();
  vector<int64_t> shape_vector;
  if (IsUnknownRank(op, "x") || IsUnknownRank(op, "segment_ids")) {
    shape_vector.push_back(-2);
  } else if (dim_idsize_input >1) {
    shape_vector.push_back(input_num_segments);
    for (int i = dim_idsize_input; i < dim_size_input; i++) {
      shape_vector.push_back(shape.GetDim(i));
    }
  } else {
    shape_vector = shape.GetDims();
    shape_vector[0] = input_num_segments;
  }

  GeTensorDescPtr tensordesc_output = op_desc->MutableOutputDesc("y");
  ge::GeShape out_shape = ge::GeShape(shape_vector);
  tensordesc_output->SetShape(out_shape);
  tensordesc_output->SetDataType(input_dtype);
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(UnsortedSegmentSumD, UnsortedSegmentSumDInferShape);
// ----------------UnsortedSegmentSumD END------------------

// ----------------StridedSliceD Op Begin-------------------
struct SliceParameters {
  std::vector<int64_t> input;
  std::vector<int64_t> begin_list;
  std::vector<int64_t> end_list;
  std::vector<int64_t> stride_list;
};

struct SliceParametersFormal {
  std::vector<int64_t> begin_list;
  std::vector<int64_t> end_list;
  std::vector<int64_t> stride_list;
};

// define the masks for 'stridedSlice'
struct SliceMasks {
  int64_t beginmask = 0;
  int64_t endmask = 0;
  int64_t ellipsismask = 0;
  int64_t newaxismask = 0;
  int64_t shrinkaxismask = 0;
};

// get value from const node
static graphStatus GetStridedSliceValueInfer(const ge::Operator& op,
                                             const std::string& keyName,
                                             vector<int64_t>& multiples) {
  if (ge::GRAPH_SUCCESS != op.GetAttr(keyName, multiples)) {
    OpsGetAttrErrReport(op.GetName(), keyName);
    OP_LOGE(op.GetName().c_str(),
            "Get const(%s) failed from op of"
            "StridedSlice!\n",
            keyName.c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Get 'begin_list','end_list','stride_list' from const node
static graphStatus GetConstStridedSliceInfer(
    const ge::Operator& op, struct SliceParameters& slice_params,
    struct SliceParametersFormal& slice_paramsFormal) {
  if (GRAPH_FAILED ==
      GetStridedSliceValueInfer(op, "begin", slice_params.begin_list)) {
    return GRAPH_FAILED;
  }

  if (GRAPH_FAILED ==
      GetStridedSliceValueInfer(op, "end", slice_params.end_list)) {
    return GRAPH_FAILED;
  }

  if (GRAPH_FAILED ==
      GetStridedSliceValueInfer(op, "strides", slice_params.stride_list)) {
    return GRAPH_FAILED;
  }
  if (GRAPH_FAILED ==
      GetStridedSliceValueInfer(op, "begin", slice_paramsFormal.begin_list)) {
    return GRAPH_FAILED;
  }

  if (GRAPH_FAILED ==
      GetStridedSliceValueInfer(op, "end", slice_paramsFormal.end_list)) {
    return GRAPH_FAILED;
  }

  if (GRAPH_FAILED == GetStridedSliceValueInfer(
                          op, "strides", slice_paramsFormal.stride_list)) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

// Get relevant masks from const node
static graphStatus GetArgsStridedSliceInfer(const ge::Operator& op,
                                            struct SliceMasks& slicemasks) {
  if (ge::GRAPH_SUCCESS != op.GetAttr("begin_mask", slicemasks.beginmask)) {
    OpsGetAttrErrReport(op.GetName(), "begin_mask");
    OP_LOGE(op.GetName().c_str(),
            "Get attribute 'begin_mask' failed from op of"
            "StridedSlice!\n");
    return GRAPH_FAILED;
  }

  if (ge::GRAPH_SUCCESS != op.GetAttr("end_mask", slicemasks.endmask)) {
    OpsGetAttrErrReport(op.GetName(), "end_mask");
    OP_LOGE(op.GetName().c_str(),
            "Get attribute 'end_mask' failed from op of"
            "StridedSlice!\n");
    return GRAPH_FAILED;
  }

  if (ge::GRAPH_SUCCESS !=
      op.GetAttr("ellipsis_mask", slicemasks.ellipsismask)) {
    OpsGetAttrErrReport(op.GetName(), "ellipsis_mask");
    OP_LOGE(op.GetName().c_str(),
            "Get attribute 'ellipsis_mask' failed from op"
            "of 'StridedSlice'!\n");
    return GRAPH_FAILED;
  }

  if (ge::GRAPH_SUCCESS !=
      op.GetAttr("new_axis_mask", slicemasks.newaxismask)) {
    OpsGetAttrErrReport(op.GetName(), "new_axis_mask");
    OP_LOGE(op.GetName().c_str(),
            "Get attribute 'new_axis_mask' failed from op of"
            "StridedSlice!\n");
    return GRAPH_FAILED;
  }

  if (ge::GRAPH_SUCCESS !=
      op.GetAttr("shrink_axis_mask", slicemasks.shrinkaxismask)) {
    OpsGetAttrErrReport(op.GetName(), "shrink_axis_mask");
    OP_LOGE(op.GetName().c_str(),
            "Get attribute 'shrink_axis_mask' failed from"
            "op of 'StridedSlice'!\n");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

static void GetBeginAndend_listInferPart1(
    const ge::Shape& shape, struct SliceMasks& slicemasks,
    struct SliceParameters& slice_params) {
  ge::Shape inputShape = shape;
  size_t dim_num = shape.GetDimNum();
  size_t begin_len = slice_params.begin_list.size();
  size_t base_number = 2.0;
  size_t newbeginmask = 0;
  size_t newendmask = 0;
  size_t newshrinknmask = 0;
  size_t newnewaxismask = 0;
  slice_params.input = shape.GetDims();

// compute the right_move of begin end stride and masks
// because of non-zero ellipsismask
  size_t right_move = std::max<int64_t>(dim_num - begin_len,0);
  if (dim_num < begin_len && slicemasks.newaxismask != 0){
    dim_num = begin_len;
  }

// rebuild the begin end stride of new_axis,
// because ignored when new_axis is true.
  if (slicemasks.newaxismask != 0) {
    for (size_t i = 0; i < dim_num; i++) {
      if ((slicemasks.newaxismask & ((int64_t)pow(base_number, i))) ==
        ((int64_t)pow(base_number, i))) {
          slice_params.begin_list[i] = 0;
          slice_params.end_list[i] = 1;
          slice_params.stride_list[i] = 1;
          if ((slicemasks.shrinkaxismask & ((int64_t)pow(base_number, i))) ==
            ((int64_t)pow(base_number, i))) {
              slicemasks.shrinkaxismask -= (int64_t)pow(base_number, i);
        }
      }
    }
  }

  size_t tmp_shrink = 0;
  if (slicemasks.shrinkaxismask != 0) {
    for (size_t i = 0; i < dim_num; i++) {
      if ((slicemasks.shrinkaxismask & ((int64_t)pow(base_number, i))) ==
        ((int64_t)pow(base_number, i))) {
          if (begin_len > i) {
            tmp_shrink += (int64_t)pow(base_number, i);
          };
      }
    }
    slicemasks.shrinkaxismask = tmp_shrink;
  }

  if (slicemasks.ellipsismask != 0) {
    size_t bitellipsis = (int64_t)log2(slicemasks.ellipsismask);
    for (size_t i = 0; i < dim_num; i++) {
      if ((slicemasks.beginmask & (1 << i)) && (bitellipsis >= i)) {
        newbeginmask += (int64_t)pow(base_number, i);
      }
      else if ((slicemasks.beginmask & (1 << i)) && (bitellipsis < i)) {
        newbeginmask += (int64_t)pow(base_number, i + right_move);
      }
      if ((slicemasks.endmask & (1 << i)) && (bitellipsis >= i)) {
        newendmask += (int64_t)pow(base_number, i);
      }
      else if ((slicemasks.endmask & (1 << i)) && (bitellipsis < i)) {
        newendmask += (int64_t)pow(base_number, i + right_move);
      }
      if ((slicemasks.shrinkaxismask & (1 << i)) && (bitellipsis >= i)) {
        newshrinknmask += (int64_t)pow(base_number, i);
      }
      else if ((slicemasks.shrinkaxismask & (1 << i)) && (bitellipsis < i)) {
        newshrinknmask += (int64_t)pow(base_number, i + right_move);
      }
      if ((slicemasks.newaxismask & (1 << i)) && (bitellipsis >= i)) {
        newnewaxismask += (int64_t)pow(base_number, i);
      }
      else if ((slicemasks.newaxismask & (1 << i)) && (bitellipsis < i)) {
        newnewaxismask += (int64_t)pow(base_number, i + right_move);
      }
    }
    slicemasks.beginmask = newbeginmask;
    slicemasks.endmask = newendmask;
    slicemasks.shrinkaxismask = newshrinknmask;
    slicemasks.newaxismask = newnewaxismask;
  }

  for (size_t i = 0; i < dim_num; i++) {
    if ((slicemasks.newaxismask & ((int64_t)pow(base_number, i))) ==
        ((int64_t)pow(base_number, i))) {
      slice_params.input.insert(slice_params.input.begin() + i, 1);
    }
  }

  size_t bitellipsis = (int64_t)log2(slicemasks.ellipsismask);
  if (slicemasks.ellipsismask != 0 && bitellipsis > begin_len-1) {
    if (begin_len < dim_num) {
      for (size_t i = 0; i < dim_num - begin_len; i++) {
        slice_params.begin_list.push_back(0);
        slice_params.end_list.push_back(slice_params.input[begin_len + i]);
        slice_params.stride_list.push_back(1);
        begin_len += 1;
      }
    }
    if (slicemasks.ellipsismask != 0) {
      for (size_t i = 0; i < dim_num; i++) {
        if ((slicemasks.ellipsismask & ((int64_t)pow(base_number, i))) ==
          ((int64_t)pow(base_number, i))) {
            size_t ellipsis_dim = i;
            slice_params.begin_list[i] = 0;
            slice_params.end_list[i] = shape.GetDim(i);
            slice_params.stride_list[i] = 1;
            if ((slicemasks.shrinkaxismask & ((int64_t)pow(base_number, i))) ==
              ((int64_t)pow(base_number, i))) {
                slicemasks.shrinkaxismask -= (int64_t)pow(base_number, i);
            }
            if (begin_len < dim_num) {
              size_t begin_len_tmp = begin_len;
              for (size_t j = 1; j <= dim_num - begin_len_tmp; j++) {
                slice_params.begin_list.insert(
                  slice_params.begin_list.begin() + ellipsis_dim + j, 0);
                slice_params.end_list.insert(
                  slice_params.end_list.begin() + ellipsis_dim + j,
                    shape.GetDim(ellipsis_dim + j));
                slice_params.stride_list.insert(
                  slice_params.stride_list.begin() + ellipsis_dim + j, 1);
            }
          }
        }
      }
    }
  }
  else {
    if (slicemasks.ellipsismask != 0) {
      for (size_t i = 0; i < dim_num; i++) {
        if ((slicemasks.ellipsismask & ((int64_t)pow(base_number, i))) ==
            ((int64_t)pow(base_number, i))) {
          size_t ellipsis_dim = i;
          slice_params.begin_list[i] = 0;
          slice_params.end_list[i] = shape.GetDim(i);
          slice_params.stride_list[i] = 1;
          if ((slicemasks.shrinkaxismask & ((int64_t)pow(base_number, i))) ==
            ((int64_t)pow(base_number, i))) {
              slicemasks.shrinkaxismask -= (int64_t)pow(base_number, i);
          }
          if (begin_len < dim_num) {
            size_t begin_len_tmp = begin_len;
            for (size_t j = 1; j <= dim_num - begin_len_tmp; j++) {
              slice_params.begin_list.insert(
                  slice_params.begin_list.begin() + ellipsis_dim + j, 0);
              slice_params.end_list.insert(
                  slice_params.end_list.begin() + ellipsis_dim + j,
                  shape.GetDim(ellipsis_dim + j));
              slice_params.stride_list.insert(
                  slice_params.stride_list.begin() + ellipsis_dim + j, 1);
              begin_len += 1;
            }
          }
        }
      }
    }
    if (begin_len < slice_params.input.size()) {
      for (size_t i = 0; i < slice_params.input.size() - begin_len; i++) {
        slice_params.begin_list.push_back(0);
        slice_params.end_list.push_back(slice_params.input[begin_len + i]);
        slice_params.stride_list.push_back(1);
      }
    }
  }

  for (size_t i = 0; i < dim_num; i++) {
    if (slice_params.begin_list[i] < 0) {
      slice_params.begin_list[i] = shape.GetDim(i) + slice_params.begin_list[i];
    }
    if (slice_params.end_list[i] < 0) {
      slice_params.end_list[i] = shape.GetDim(i) + slice_params.end_list[i];
    }
  }

  for (size_t i = 0; i < dim_num; i++) {
    if ((slicemasks.beginmask & ((int64_t)pow(base_number, i))) ==
        ((int64_t)pow(base_number, i))) {
      if (slice_params.stride_list[i] > 0) {
        slice_params.begin_list[i] = 0;
      }
      if (slice_params.stride_list[i] < 0) {
        slice_params.begin_list[i] = slice_params.input[i];
      }
    }

    if ((slicemasks.endmask & ((int64_t)pow(base_number, i))) ==
        ((int64_t)pow(base_number, i))) {
      if (slice_params.stride_list[i] > 0) {
        slice_params.end_list[i] = slice_params.input[i];
      }
      if (slice_params.stride_list[i] < 0) {
        slice_params.end_list[i] = 0;
      }
    }
    if ((slicemasks.ellipsismask & ((int64_t)pow(base_number, i))) ==
        ((int64_t)pow(base_number, i))) {
      slice_params.begin_list[i] = 0;
      slice_params.end_list[i] = shape.GetDim(i);
      slice_params.stride_list[i] = 1;
    }
  }
}

static void GetBeginAndend_listInferPart2(
    const ge::Shape& shape, struct SliceMasks& slicemasks,
    struct SliceParameters& slice_params) {
  ge::Shape inputShape = shape;
  size_t dim_num = shape.GetDimNum();
  size_t base_number = 2.0;
  size_t new_axis_flag = 0;


  for (size_t i = 0; i < dim_num; i++) {
    if ((slicemasks.newaxismask & ((int64_t)pow(base_number, i))) ==
        ((int64_t)pow(base_number, i))) {
      new_axis_flag += 1;
    }
  }

  for (size_t i = 0; i < slice_params.input.size(); i++) {
    if ((slicemasks.shrinkaxismask & ((int64_t)pow(base_number, i))) ==
        ((int64_t)pow(base_number, i))) {
      slice_params.end_list[i] = slice_params.begin_list[i] + 1;
    }
  }

  for (size_t i = 0; i < slice_params.begin_list.size(); i++) {
    if (slice_params.stride_list[i] > 0) {
      if (slice_params.begin_list[i] >= slice_params.end_list[i]) {
        slice_params.begin_list[i] = slice_params.end_list[i];
      }

      if (slice_params.end_list[i] > slice_params.input[i]) {
        slice_params.end_list[i] = slice_params.input[i];
      }
      if (slice_params.end_list[i] == 0) {
        slice_params.begin_list[i] = slice_params.end_list[i];
      }
      if (slice_params.begin_list[i] < 0 && slice_params.end_list[i] >= 0) {
        slice_params.begin_list[i] = 0;
        if (slice_params.end_list[i] >= slice_params.input[i]) {
          slice_params.end_list[i] = slice_params.input[i];
        }
      }
    }
    if (slice_params.stride_list[i] < 0) {
      if (slice_params.begin_list[i] >= slice_params.input[i]) {
        if (slice_params.end_list[i] >= 0) {
          slice_params.begin_list[i] = slice_params.input[i] - 1;
        }
        if (slice_params.end_list[i] < 0) {
          slice_params.begin_list[i] = slice_params.input[i];
          slice_params.end_list[i] = 0;
        }
      }
      if (slice_params.begin_list[i] == 0) {
        if (slice_params.begin_list[i] <= slice_params.end_list[i]){
          slice_params.begin_list[i] = slice_params.end_list[i];
        }
        if (slice_params.begin_list[i] > slice_params.end_list[i]){
          slice_params.begin_list[i] = 0;
          slice_params.end_list[i] = -1;
        }
      }
    }
  }
}

static graphStatus GetStridedSliceInfer(
    const ge::Operator& op, struct SliceParameters& slice_params,
    struct SliceParametersFormal& slice_paramsFormal) {
  Tensor begin_tensor;
  Tensor end_tensor;
  Tensor stride_tensor;

  if (op.GetInputConstData("begin", begin_tensor) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get constValue failed of [begin]");
    return GRAPH_FAILED;
  }

  if (op.GetInputConstData("end", end_tensor) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get constValue failed of [end]");
    return GRAPH_FAILED;
  }

  if (op.GetInputConstData("strides", stride_tensor) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get constValue failed of [strides]");
    return GRAPH_FAILED;
  }

  DataType dtype = op.GetInputDesc("begin").GetDataType();

  vector<int64_t> begin_list;
  vector<int64_t> end_list;
  vector<int64_t> stride_list;

  GetConstValue(op, begin_tensor, dtype, begin_list);
  GetConstValue(op, end_tensor, dtype, end_list);
  GetConstValue(op, stride_tensor, dtype, stride_list);

  slice_params.begin_list = begin_list;
  slice_paramsFormal.begin_list = begin_list;

  slice_params.end_list = end_list;
  slice_paramsFormal.end_list = end_list;

  slice_params.stride_list = stride_list;
  slice_paramsFormal.stride_list = stride_list;

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(StridedSliceDInferShape) {
  // Get input shape
  ge::Shape shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  size_t dim_num = shape.GetDimNum();

  // Get 'begin_list','end_list','stride_list' from const node
  struct SliceParameters slice_params_output = {};
  struct SliceParametersFormal slice_params_outputFormal = {};
  if (GRAPH_FAILED == GetConstStridedSliceInfer(op, slice_params_output,
                                                slice_params_outputFormal)) {
    return GRAPH_FAILED;
  }

  // Get relevant masks from const node
  struct SliceMasks slicemasks_output = {};
  if (GRAPH_FAILED == GetArgsStridedSliceInfer(op, slicemasks_output)) {
    return GRAPH_FAILED;
  }

  // Deal with 'begin_list' and 'end_list' by corresponding mask
  GetBeginAndend_listInferPart1(shape, slicemasks_output, slice_params_output);
  GetBeginAndend_listInferPart2(shape, slicemasks_output, slice_params_output);

  size_t shrinkaxismaskTemp = 0;
  size_t base_number = 2.0;
  for (size_t i = 0; i < dim_num; ++i) {
    if ((slice_params_output.end_list[i] - slice_params_output.begin_list[i]) ==
        0) {
      shrinkaxismaskTemp += pow(base_number, i);
    }
  }
  slicemasks_output.shrinkaxismask =
      slicemasks_output.shrinkaxismask | shrinkaxismaskTemp;

  std::vector<int64_t> outputlist;
  std::vector<int64_t> outputshapelist;
  // Convert the target data into a double type by multiply '1.0'
  double changeToDouble = 1.0;
  for (size_t i = 0; i < slice_params_output.begin_list.size(); ++i) {
    size_t dim = (int64_t)(ceil(
        (slice_params_output.end_list[i] - slice_params_output.begin_list[i]) /
        (slice_params_output.stride_list[i] * changeToDouble)));
    dim = std::max<int64_t>(dim, int64_t(0));
    if (((slicemasks_output.shrinkaxismask & ((uint64_t)pow(2.0, i))) !=
         ((uint64_t)pow(2.0, i))) ||
        ((slicemasks_output.newaxismask & ((uint64_t)pow(2.0, i))) !=
         ((uint64_t)pow(2.0, i)))) {
      // get outputshape
      outputshapelist.push_back(dim);
      if (dim != 1) {
        // get real dim cnt
        outputlist.push_back(dim);
      }
    }
  }
  if(slicemasks_output.shrinkaxismask == 0 && slicemasks_output.newaxismask == 0){
    if (slice_params_output.begin_list.size() > slice_params_output.input.size()) {
      for (size_t i = 0; i < slice_params_output.begin_list.size() - slice_params_output.input.size(); i++) {
        outputshapelist.erase(outputshapelist.begin() + i + slice_params_output.input.size());
      }
    }
  }

  if (slicemasks_output.shrinkaxismask > 0) {
    size_t shrink_flag = 0;
    for (size_t i = 0; i < dim_num; i++) {
      if (((uint64_t)slicemasks_output.shrinkaxismask & ((uint64_t)pow(base_number, i))) ==
          ((uint64_t)pow(base_number, i))) {
        outputshapelist.erase(outputshapelist.begin() + i - shrink_flag);
        shrink_flag += 1;
      }
    }
  }

  if (outputlist.size() == 0) {
    outputlist.push_back(1);
  }

  ge::Shape outputShape = ge::Shape(outputshapelist);
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(outputShape);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(StridedSliceD, StridedSliceDInferShape);
// ----------------StridedSliceD Op End-------------------

// ----------------stridedSlice Op Begin-------------------
IMPLEMT_COMMON_INFERFUNC(StridedSliceInferShape) {

// Get input shape
  ge::Shape shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  size_t dim_num = shape.GetDimNum();

  if (dim_num == 0) {
    OP_LOGE("Get input x's dimnum is 0");
    return GRAPH_FAILED;
  }
// Get 'begin_list','end_list','stride_list' from const node
  struct SliceParameters slice_params_output = {};
  struct SliceParametersFormal slice_params_outputFormal = {};
  if (GRAPH_FAILED == GetStridedSliceInfer(op, slice_params_output,
                                           slice_params_outputFormal)) {
    OP_LOGW(op.GetName().c_str(), "Get constValue failed of [begin,end,stride]");
    return GRAPH_SUCCESS;
  }

  if (slice_params_output.end_list.size() != slice_params_output.begin_list.size()) {
    OP_LOGE(op.GetName().c_str(), "end shape,begin shape length mismatch!");
    return GRAPH_FAILED;
  }
// Get relevant masks from const node
  struct SliceMasks slicemasks_output = {};
  if (GRAPH_FAILED == GetArgsStridedSliceInfer(op, slicemasks_output)) {
    return GRAPH_FAILED;
  }

  int64_t ellipsis_dim = 0;
  if (slicemasks_output.ellipsismask != 0) {
    for (size_t i = 0; i < dim_num; ++i) {
      if ((slicemasks_output.ellipsismask & ((uint64_t) pow(2.0, i))) == ((uint64_t) pow(2.0, i))) {
        ellipsis_dim += 1;
      }
    }
    if (ellipsis_dim > 1) {
      OP_LOGE(op.GetName().c_str(), "only suppot 1 dim of ellipsis!");
      return GRAPH_FAILED;
    }
  }

// Deal with 'begin_list' and 'end_list' by corresponding mask
  GetBeginAndend_listInferPart1(shape, slicemasks_output, slice_params_output);
  GetBeginAndend_listInferPart2(shape, slicemasks_output, slice_params_output);

  size_t shrinkaxismaskTemp = 0;
  size_t base_number = 2.0;
  if (slicemasks_output.shrinkaxismask != 0) {
    for (size_t i = 0; i < dim_num; ++i) {
      if ((slice_params_output.end_list[i] - slice_params_output.begin_list[i])
          == 0) {
        shrinkaxismaskTemp += pow(base_number, i);
      }
    }
  }
  slicemasks_output.shrinkaxismask = slicemasks_output.shrinkaxismask |
                                     shrinkaxismaskTemp;

  std::vector<int64_t> outputlist;
  std::vector<int64_t> outputshapelist;
// Convert the target data into a double type by multiply '1.0'
  double changeToDouble = 1.0;
  for (size_t i = 0; i < slice_params_output.begin_list.size(); ++i) {
    size_t dim = (int64_t) (ceil((slice_params_output.end_list[i] -
                                  slice_params_output.begin_list[i]) /
                                  (slice_params_output.stride_list[i] *
                                   changeToDouble)));
    dim = std::max<int64_t>(dim, int64_t(0));
    if (((slicemasks_output.shrinkaxismask & ((uint64_t) pow(2.0, i))) !=
         ((uint64_t) pow(2.0, i))) || ((slicemasks_output.newaxismask &
                                        ((uint64_t) pow(2.0, i))) !=
                                       ((uint64_t) pow(2.0, i)))) {
// get outputshape
      outputshapelist.push_back(dim);
      if (dim != 1) {
// get real dim cnt
        outputlist.push_back(dim);
      }
    }
  }
  if(slicemasks_output.shrinkaxismask == 0 && slicemasks_output.newaxismask == 0){
    if (slice_params_output.begin_list.size() > slice_params_output.input.size()) {
      for (size_t i = 0; i < slice_params_output.begin_list.size() - slice_params_output.input.size(); i++) {
        outputshapelist.erase(outputshapelist.begin() + i + slice_params_output.input.size());
      }
    }
  }
// shrink_axis_mask != 0
  if (slicemasks_output.shrinkaxismask > 0) {
    size_t shrink_flag = 0;
    for (size_t i = 0; i < dim_num; i++) {
      if ((slicemasks_output.shrinkaxismask & ((int64_t)pow(base_number, i))) ==
          ((int64_t)pow(base_number, i))) {
        outputshapelist.erase(outputshapelist.begin() + i - shrink_flag);
        shrink_flag += 1;
      }
    }
  }

  if (outputlist.size() == 0) {
    outputlist.push_back(1);
  }
  ge::Shape outputShape = ge::Shape(outputshapelist);
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(outputShape);
  tensordesc_output.SetDataType(input_dtype);
  tensordesc_output.SetRealDimCnt(outputlist.size());
  (void) op.UpdateOutputDesc("y", tensordesc_output);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(StridedSlice, StridedSliceInferShape);
// ----------------StridedSlice Op End-------------------

// ----------------ReverseV2 Op Begin-----------------
IMPLEMT_INFERFUNC(ReverseV2, ReverseV2InferShape) {
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  Shape input_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  tensordesc_output.SetShape(input_shape);
  tensordesc_output.SetDataType(input_dtype);

  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ReverseV2, ReverseV2InferShape);
// ----------------ReverseV2 Op End-------------------

// ----------------ReverseV2D Op Begin---------------
IMPLEMT_INFERFUNC(ReverseV2D, ReverseV2DInferShape) {
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  Shape input_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  tensordesc_output.SetShape(input_shape);
  tensordesc_output.SetDataType(input_dtype);

  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ReverseV2D, ReverseV2DInferShape);
// ----------------ReverseV2D Op End------------------

// ----------------Select----------------------
IMPLEMT_VERIFIER(Select, SelectVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(SelectInferShape) {
  Shape x1_shape = op.GetInputDesc("x2").GetShape();
  DataType input_dtype = op.GetInputDesc("x2").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(Shape(x1_shape));
  td.SetDataType(input_dtype);
  (void) op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Select, SelectInferShape);
VERIFY_FUNC_REG(Select, SelectVerify);
// ---------------Select END-----------------------

// ----------------SelectV2----------------------
bool BroadCastTwoinOneout(const Operator &op,
                       const ge::Shape& shape_x,
                       const ge::Shape& shape_y,
                       std::vector<int64_t>& dim_out) {
  std::vector<int64_t> dim_x = shape_x.GetDims();
  std::vector<int64_t> dim_y = shape_y.GetDims();
  // exchange them
  if (dim_x.size() < dim_y.size()) {
    std::vector<int64_t> dim_tmp = dim_x;
    dim_x = dim_y;
    dim_y = dim_tmp;
  }

  // expand smalll shape
  if (dim_x.size() != dim_y.size()) {
    int dec = dim_x.size() - dim_y.size();
    for (int i = 0; i < dec; i++) {
      dim_y.insert(dim_y.begin(), (int64_t)1);
    }
  }

  // set out dims
  for (size_t i = 0; i < dim_x.size(); i++) {
    if ((dim_x[i] != dim_y[i]) && (dim_x[i] != 1) && (dim_y[i] != 1)) {
      OP_LOGE(op.GetName().c_str(),
              "The %s's dimensions does not match the broadcast rule(%lu %lu).",
              op.GetName().c_str(),
              dim_x[i],
              dim_y[i]);
      return false;
    }

    int64_t dim = dim_x[i] > dim_y[i] ? dim_x[i] : dim_y[i];
    dim_out.push_back(dim);
  }
  return true;
}

IMPLEMT_VERIFIER(SelectV2, SelectV2Verify) {
  if (!CheckTwoInputDtypeSame(op, "then", "else")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}


IMPLEMT_COMMON_INFERFUNC(SelectV2InferShape) {
  Shape x1_shape = op.GetInputDesc("condition").GetShape();
  Shape x2_shape = op.GetInputDesc("then").GetShape();
  Shape x3_shape = op.GetInputDesc("else").GetShape();

  std::vector<int64_t> x1_x2_max;
  if (!BroadCastTwoinOneout(op, x1_shape, x2_shape, x1_x2_max)) {
    return GRAPH_FAILED;
  }
  ge::Shape shape_x1_x2_max = ge::Shape(x1_x2_max);
  std::vector<int64_t> broadcast_shape;
  if (!BroadCastTwoinOneout(op, shape_x1_x2_max, x3_shape, broadcast_shape)) {
    return GRAPH_FAILED;
  }
  DataType input_dtype = op.GetInputDesc("then").GetDataType();
  ge::Shape outputShape = ge::Shape(broadcast_shape);
  TensorDesc tensordesc_output = op.GetOutputDesc("result");
  tensordesc_output.SetShape(outputShape);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("result", tensordesc_output);
  OP_LOGI(op.GetName().c_str(),
          "output shape is: %s, output dtype is:%d.",
          to_string(outputShape).c_str(),input_dtype);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SelectV2, SelectV2InferShape);
VERIFY_FUNC_REG(SelectV2, SelectV2Verify);
// ---------------SelectV2 END-----------------------

// ----------------SegmentMax-------------------
static bool SegmentShapeVerify(const Operator &op,
                               const std::string &input_name,
                               const std::string &segment_ids_name) {
  auto input_shape_dims = op.GetInputDesc("x").GetShape().GetDims();
  auto segment_ids_shape_dims =
      op.GetInputDesc("segment_ids").GetShape().GetDims();

  if (input_shape_dims.empty() || segment_ids_shape_dims.empty()) {
    OP_LOGE(op.GetName().c_str(), "shape of input is empty.");
    return false;
  }

  return true;
}

IMPLEMT_VERIFIER(SegmentMax, SegmentMaxVerify) {
  if (!SegmentShapeVerify(op, "x", "segment_ids")) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(SegmentMaxInferShape) {
  auto input_desc = op.GetInputDesc("x");
  const std::string segment_ids_name = "segment_ids";
  Tensor segment_ids;
  int64_t first_axis_dims;
  if (GRAPH_SUCCESS != op.GetInputConstData(segment_ids_name, segment_ids)) {
    OP_LOGI("segment_max", "GetInputConstData %s failed.",
    segment_ids_name.c_str());
    first_axis_dims = -1;
  } else {
    auto data_type = op.GetInputDesc(segment_ids_name).GetDataType();
    std::vector<int64_t> const_data;
    if (!GetConstIntData(segment_ids, data_type, const_data)) {
      OP_LOGE("segment_max", "invalid data type of segment_ids,"
          "data_type is %d.", (int) data_type);
      return GRAPH_FAILED;
    }
    first_axis_dims = (*std::max_element(const_data.begin(),
        const_data.end())) + 1;
  }

  auto output_shape_dims = input_desc.GetShape().GetDims();
  output_shape_dims[0] = first_axis_dims;
  Shape output_shape(output_shape_dims);
  DataType input_dtype = input_desc.GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(output_shape);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SegmentMax, SegmentMaxInferShape);
VERIFY_FUNC_REG(SegmentMax, SegmentMaxVerify);
// ----------------SegmentMax END-------------------


// ----------------SegmentMaxD-------------------
static bool SegmentDShapeVerify(const Operator &op,
                                const std::string &input_name,
                                const std::string &segment_ids_name) {
  auto input_shape_dims = op.GetInputDesc("x").GetShape().GetDims();

  std::vector<int64_t> segment_ids;
  if (GRAPH_SUCCESS != op.GetAttr(segment_ids_name, segment_ids)) {
    OP_LOGE("segment_max_d", "GetAttr %s failed.", segment_ids_name.c_str());
    return false;
  }

  if (input_shape_dims.empty() || segment_ids.empty()) {
    OP_LOGE("segment_max_d", "shape of input is empty.");
    return false;
  }

  return true;
}

IMPLEMT_VERIFIER(SegmentMaxD, SegmentMaxDVerify) {
  if (!SegmentDShapeVerify(op, "x", "segment_ids")) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(SegmentMaxDInferShape) {
  OP_LOGI("segment_max_d", "enter SegmentMaxDInferShape ......");
  auto input_desc = op.GetInputDesc("x");
  const std::string segment_ids_name = "segment_ids";
  std::vector<int64_t> segment_ids;
  if (GRAPH_SUCCESS != op.GetAttr(segment_ids_name, segment_ids)) {
    OP_LOGE("segment_max_d", "GetAttr %s failed.", segment_ids_name.c_str());
    return GRAPH_FAILED;
  }
  Shape shape = input_desc.GetShape();
  if ((int64_t)segment_ids.size() != (int64_t)shape.GetDim(0)) {
    OP_LOGE(op.GetName().c_str(), "the length of "
            "segment_ids should be equal to shape[0].");
    return GRAPH_FAILED;
  }
  for (size_t dim = 0; dim < segment_ids.size(); dim++) {
    if (dim == 0 && segment_ids[dim] < 0) {
      OP_LOGE(op.GetName().c_str(), "segment_ids must be positive integer");
      return GRAPH_FAILED;
    }
    if (dim > 0 && segment_ids[dim] < segment_ids[dim - 1]) {
      OP_LOGE(op.GetName().c_str(), "segment_ids must "
              "be sorted(from small to large)");
      return GRAPH_FAILED;
    }
  }

  int64_t first_axis_dims = (*std::max_element(segment_ids.begin(),
      segment_ids.end())) + 1;

  auto output_shape_dims = input_desc.GetShape().GetDims();
  output_shape_dims[0] = first_axis_dims;
  for (auto item : output_shape_dims) {
    OP_LOGI("segment_max_d", "shape dims:%lld.", item);
  }
  Shape output_shape(output_shape_dims);
  DataType input_dtype = input_desc.GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(output_shape);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SegmentMaxD, SegmentMaxDInferShape);
VERIFY_FUNC_REG(SegmentMaxD, SegmentMaxDVerify);
// ----------------SegmentMaxD END-------------------

//----------------SliceD Op Begin ----------------------
IMPLEMT_VERIFIER(SliceD, SliceDVerify) {
  std::vector<int64_t> input_size;
  if (ge::GRAPH_SUCCESS != op.GetAttr("size", input_size)) {
    OpsGetAttrErrReport(op.GetName(), "size");
    OP_LOGE(op.GetName().c_str(), "The size op"
      "GetOpAttr ConstValue failed!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> input_begin;
  if (ge::GRAPH_SUCCESS != op.GetAttr("offsets", input_begin)) {
    OpsGetAttrErrReport(op.GetName(), "begin");
    OP_LOGE(op.GetName().c_str(), "The begin op"
      "GetOpAttr ConstValue failed!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(SliceDInferShape) {
  std::vector<int64_t> input_size;
  if (ge::GRAPH_SUCCESS != op.GetAttr("size", input_size)) {
    OpsGetAttrErrReport(op.GetName(), "size");
    OP_LOGE(op.GetName().c_str(), "The size op GetOpAttr"
            "ConstValue failed!");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> input_begin;
  if (ge::GRAPH_SUCCESS != op.GetAttr("offsets", input_begin)) {
    OpsGetAttrErrReport(op.GetName(), "begin");
    OP_LOGE(op.GetName().c_str(), "The input_begin op GetOpAttr ConstValue failed!");
    return GRAPH_FAILED;
  }
  Shape shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  size_t dimNum = shape.GetDimNum();

  if ((int64_t)input_size.size() != (int64_t)dimNum){
    OpsAttrValueErrReport(op.GetName(), "length of size", Strcat((int64_t)dimNum), Strcat((int64_t)input_size.size()));
    OP_LOGE(op.GetName().c_str(), "the length of size"
    "must be equal to shape!");
    return GRAPH_FAILED;
  }
  if ((int64_t) input_begin.size() != (int64_t) dimNum){
    OpsAttrValueErrReport(op.GetName(), "length of begin", Strcat((int64_t) dimNum), Strcat((int64_t) input_begin.size()));
    OP_LOGE(op.GetName().c_str(), "the length of begin"
    "must be equal to shape!");
    return GRAPH_FAILED;
  }
  for (int64_t i = 0; i < (int64_t)dimNum; ++i) {
    if (input_size[i] > shape.GetDim(i) || input_size[i] < -1) {
      string excepted_value = Strcat("in range[0,", shape.GetDim(i), "]");
      OpsAttrValueErrReport(op.GetName(), "size", excepted_value, Strcat(input_size[i]));
      OP_LOGE(op.GetName().c_str(), "size must be greater"
              "than or equal to 0, and less than shape!");
      return GRAPH_FAILED;
    }
    if (input_begin[i] > shape.GetDim(i) || input_begin[i] < -1) {
      string excepted_value = Strcat("in range[-1,", shape.GetDim(i), "] and cannot be equal to 0");
      OpsAttrValueErrReport(op.GetName(), "begin", excepted_value, Strcat(input_begin[i]));
      OP_LOGE(op.GetName().c_str(), "begin must be , greater"
              "than or equal to -1, less than or equal to shape,"
              "and cannot be equal to 0!");
      return GRAPH_FAILED;
    }
  }
  std::vector<int64_t> outputList;
  for (size_t i = 0; i < dimNum; ++i) {
    if ((int)input_size[i] == -1) {
      outputList.push_back((int)(shape.GetDim(i) - input_begin[i]));
    } else {
      outputList.push_back((int)input_size[i]);
      }
  }
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  Shape outputShape(outputList);
  tensordesc_output.SetShape(outputShape);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SliceD, SliceDInferShape);
VERIFY_FUNC_REG(SliceD, SliceDVerify);
//----------------SliceD Op End ----------------------

//----------------Slice Op Begin ----------------------
static void GetSliceConstValue(const Tensor& const_tensor,
    const DataType& dtype, std::vector<int64_t>& const_data) {
  size_t size=0;
  if (dtype == ge::DT_INT32) {
    int32_t *const_data_ptr = (int32_t *)const_tensor.GetData();
  size = const_tensor.GetSize() / sizeof(int32_t);
  for (size_t i = 0; i < size; ++i) {
    const_data.push_back((int32_t)((*(const_data_ptr + i))));
 }
  } else {
      int64_t *const_data_ptr = (int64_t *)const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; ++i) {
          const_data.push_back(((int64_t)(*(const_data_ptr + i))));
    }
  }
}

IMPLEMT_COMMON_INFERFUNC(SliceInferShape) {
  Tensor input_begin_tensor;
  Tensor input_size_tensor;
  Shape shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  size_t dimNum = shape.GetDimNum();
  std::vector<int64_t> outputList;
  std::vector<int64_t> input_begin;
  std::vector<int64_t> input_size;

  if ((op.GetInputConstData("offsets", input_begin_tensor) != GRAPH_SUCCESS) &&
      (op.GetInputConstData("size", input_size_tensor) != GRAPH_SUCCESS)) {
    OP_LOGI(op.GetName().c_str(), "Get constValue failed of [offsets]");
    OP_LOGI(op.GetName().c_str(), "Get constValue failed of [size]");
    for (size_t i = 0; i < dimNum; ++i){
      outputList.push_back(-1);
    }
  }
  else if ((op.GetInputConstData("offsets", input_begin_tensor) != GRAPH_SUCCESS) &&
          (op.GetInputConstData("size", input_size_tensor) == GRAPH_SUCCESS)) {
        DataType input_size_dtype = op.GetInputDesc("size").GetDataType();
        GetSliceConstValue(input_size_tensor, input_size_dtype, input_size);
        for (size_t i = 0; i < dimNum; ++i) {
          if ((int)input_size[i] == -1) {
            outputList.push_back(-1);
        } else {
           outputList.push_back((int)input_size[i]);
          }
        }
    }
  else if ((op.GetInputConstData("offsets", input_begin_tensor) == GRAPH_SUCCESS) &&
          (op.GetInputConstData("size", input_size_tensor) != GRAPH_SUCCESS)) {
        DataType input_begin_dtype = op.GetInputDesc("offsets").GetDataType();
        GetSliceConstValue(input_begin_tensor, input_begin_dtype, input_begin);
        for (size_t i = 0; i < dimNum; ++i){
          outputList.push_back(-1);
        }
      }
  else {
    op.GetInputConstData("offsets", input_begin_tensor);
    op.GetInputConstData("size", input_size_tensor);
    DataType input_begin_dtype = op.GetInputDesc("offsets").GetDataType();
    GetSliceConstValue(input_begin_tensor, input_begin_dtype, input_begin);
    DataType input_size_dtype = op.GetInputDesc("size").GetDataType();
    GetSliceConstValue(input_size_tensor, input_size_dtype, input_size);
    for (size_t i = 0; i < dimNum; ++i) {
    if ((int)input_size[i] == -1) {
      outputList.push_back((int)(shape.GetDim(i) - input_begin[i]));
    } else {
       outputList.push_back((int)input_size[i]);
      }
    }
  }
    TensorDesc tensordesc_output = op.GetOutputDesc("y");
    Shape outputShape(outputList);
    tensordesc_output.SetShape(outputShape);
    tensordesc_output.SetDataType(input_dtype);
    (void)op.UpdateOutputDesc("y", tensordesc_output);
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Slice, SliceInferShape);
//----------------Slice Op End----------------------

// ----------------OneHotD--------------------------
static graphStatus OneHotInferShapeAndType(ge::Operator& op,
                                           DataType& input_type,
                                           std::int64_t& depth,
                                           int32_t axis) {
  ge::Shape indices_shape = op.GetInputDesc(0).GetShape();
  size_t dim_num = indices_shape.GetDimNum();
  std::vector<int64_t> dim_vector;
  if (axis == -1) {
    for (size_t i = 0; i < dim_num; i++) {
      dim_vector.push_back(indices_shape.GetDim(i));
    }
    dim_vector.push_back(depth);
  } else {
    for (size_t i = 0; i <= dim_num; i++) {
      if (i < static_cast<size_t>(axis)) {
        dim_vector.push_back(indices_shape.GetDim(i));
      } else if (i == static_cast<size_t>(axis)) {
        dim_vector.push_back(depth);
      } else {
        dim_vector.push_back(indices_shape.GetDim(i-1));
      }
    }
  }

  Shape out_shape(dim_vector);
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(out_shape);
  tensordesc_output.SetDataType(input_type);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(OneHotDInferShape) {
  std::int64_t depth = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("depth", depth)) {
    OpsGetAttrErrReport(op.GetName(), "depth");
    OP_LOGE(op.GetName().c_str(),"OneHot GetOpAttr depth failed!");
    return GRAPH_FAILED;
  }
  if (depth < 1){
    OpsAttrValueErrReport(op.GetName(), "depth", "greater than or equals to 1", Strcat(depth));
    OP_LOGE(op.GetName().c_str(), "depth need greater than or equals to 1");
    return GRAPH_FAILED;
  }

  ge::Shape indices_shape = op.GetInputDesc(0).GetShape();
  int32_t dim_num = 0;
  dim_num = indices_shape.GetDimNum();
  int32_t axis = -1;
  if (ge::GRAPH_SUCCESS != op.GetAttr("axis", axis)) {
    OpsGetAttrErrReport(op.GetName(), "axis");
    OP_LOGE("Get const axis failed from op of 'OneHotD'!\n");
    return GRAPH_FAILED;
  }
  if (axis < -dim_num || axis > dim_num) {
    OpsInputShapeDimErrReport(op.GetName(), "axis", Strcat(dim_num), Strcat(-dim_num), Strcat(axis));
    OP_LOGE(op.GetName().c_str(), "attr axis is not in range");
    return GRAPH_FAILED;
  }

  DataType input_type = op.GetInputDesc("on_value").GetDataType();
  return OneHotInferShapeAndType(op, input_type, depth, axis);

}

COMMON_INFER_FUNC_REG(OneHotD, OneHotDInferShape);
// ----------------OneHotD END----------------------

// ----------------OneHot---------------------------

IMPLEMT_COMMON_INFERFUNC(OneHotInferShape) {
  Tensor depth_tensor;
  std::int64_t depth = 0;
  DataType dtype = op.GetInputDesc("depth").GetDataType();
  if (ge::GRAPH_SUCCESS != op.GetInputConstData("depth", depth_tensor)) {
    OP_LOGI("Get constdata failed from op of 'OneHot'!\n");
    depth = -1;
  } else {
    if (!GetScalerValue(op, depth_tensor, dtype, depth)){
      OP_LOGE(op.GetName().c_str(), "Get Const Value failed ");
      return GRAPH_FAILED;
    };
  }
  ge::Shape indices_shape = op.GetInputDesc(0).GetShape();
  int32_t dim_num = 0;
  int32_t axis = -1;
  dim_num = indices_shape.GetDimNum();
  if (ge::GRAPH_SUCCESS != op.GetAttr("axis", axis)) {
    OP_LOGE("Get const axis failed from op of 'OneHot'!\n");
    return GRAPH_FAILED;
  }
  if (axis < -dim_num || axis > dim_num) {
    OP_LOGE(op.GetName().c_str(), "attr axis is not in range");
    return GRAPH_FAILED;
  }
  DataType input_type = op.GetInputDesc("on_value").GetDataType();

  return OneHotInferShapeAndType(op, input_type, depth, axis);

}

COMMON_INFER_FUNC_REG(OneHot, OneHotInferShape);
// ----------------OneHot END----------------------

// ----------------TopKD Op Start-------------------
IMPLEMT_COMMON_INFERFUNC(TopKDInferShape) {
  TensorDesc input_tensor_desc = op.GetInputDesc("x");
  TensorDesc value_tensor_desc = op.GetOutputDesc("values");
  TensorDesc indice_tensor_desc = op.GetOutputDesc("indices");

  int32_t k;
  if (ge::GRAPH_SUCCESS != op.GetAttr("k", k))
  {
    LOG_ERROR("[ERROR]op [%s] Attr k is empty !\n", op.GetName().c_str());
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dims_in = input_tensor_desc.GetShape().GetDims();
  if (dims_in.size() > 0) {
    dims_in[dims_in.size() - 1] = k;
  }

  value_tensor_desc.SetShape(ge::Shape(dims_in));
  value_tensor_desc.SetDataType(input_tensor_desc.GetDataType());
  (void)op.UpdateOutputDesc("values", value_tensor_desc);

  indice_tensor_desc.SetShape(ge::Shape(dims_in));
  indice_tensor_desc.SetDataType(DT_INT32);
  (void)op.UpdateOutputDesc("indices", indice_tensor_desc);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(TopKD, TopKDInferShape);
// ----------------TopKD Op End-------------------

//----------------TopK Op-------------------
IMPLEMT_VERIFIER(TopK, TopKVerify)
{
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(TopKInferShape)
{
  TensorDesc input_tensor_desc = op.GetInputDesc("x");
  TensorDesc value_tensor_desc = op.GetOutputDesc("values");
  TensorDesc indice_tensor_desc = op.GetOutputDesc("indices");

  Tensor k_tensor;
  bool unkonwn_dim_flag = false;
  if (GRAPH_SUCCESS != op.GetInputConstData("k", k_tensor)) {
    OP_LOGI(op.GetName().c_str(), "Get constdata failed, unknown dim.");
    unkonwn_dim_flag = true;
  }
  DataType dtype = op.GetInputDesc("k").GetDataType();
  if (dtype != ge::DT_INT32) {
    LOG_ERROR("[ERROR]op [%s] k type Error !\n", op.GetName().c_str());
    return GRAPH_FAILED;
  }
  // Tensor::GetData() return a uint8 ptr. However the definition of k is int32
  // So here use int32* ptr to get the k value
  int64_t k = UNKNOWN_DIM;
  if (!unkonwn_dim_flag && k_tensor.GetData() != nullptr) {
      k = (int64_t)*(int32_t*)k_tensor.GetData();
  }

  std::vector<int64_t> dims_in = input_tensor_desc.GetShape().GetDims();
  if (dims_in.size() > 0) {
    dims_in[dims_in.size() - 1] = k;
  }

  value_tensor_desc.SetShape(ge::Shape(dims_in));
  value_tensor_desc.SetDataType(input_tensor_desc.GetDataType());
  (void)op.UpdateOutputDesc("values", value_tensor_desc);

  indice_tensor_desc.SetShape(ge::Shape(dims_in));
  indice_tensor_desc.SetDataType(DT_INT32);
  (void)op.UpdateOutputDesc("indices", indice_tensor_desc);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(TopK, TopKInferShape);
VERIFY_FUNC_REG(TopK, TopKVerify);
// ----------------TopK Op End-------------------

// ----------------Gather-------------------
IMPLEMT_COMMON_INFERFUNC(GatherInferShape) {
  Shape params_shape = op.GetInputDesc("x").GetShape();
  Shape indices_shape = op.GetInputDesc("indices").GetShape();
  int params_dimnum = op.GetInputDesc("x").GetRealDimCnt();
  int indices_dimnum = op.GetInputDesc("indices").GetRealDimCnt();
  int y_dimnum = params_dimnum + indices_dimnum - 1;
  std::vector<int64_t> dim_vector;
  if (indices_dimnum == 0) {
      indices_dimnum = 1;
      indices_shape = Shape({1});
  }
  if (params_dimnum == 0) {
      params_dimnum = 1;
      params_shape = Shape({1});
  }
  for (int i = 0; i < indices_dimnum; i++) {
    dim_vector.push_back((int64_t)indices_shape.GetDim(i));
  }
  for (int j = 1; j < params_dimnum; j++) {
    dim_vector.push_back((int64_t)params_shape.GetDim(j));
  }

  TensorDesc y_desc = op.GetOutputDesc("y");
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  Shape output_shape = Shape(dim_vector);
  y_desc.SetShape(output_shape);
  y_desc.SetDataType(input_dtype);
  y_desc.SetRealDimCnt(y_dimnum);
  (void)op.UpdateOutputDesc("y", y_desc);
  OP_LOGI(op.GetName().c_str(), "output shape is:%s",
          to_string(output_shape).c_str());
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Gather, GatherInferShape);
// ----------------Gather END-------------------

//----------------ScatterNd-------------------
IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(ScatterNdInferShape)
  vector<string> input_infer_depends = {"shape"};
  op_desc->SetOpInferDepends(input_infer_depends);

  auto output_desc = op_desc->MutableOutputDesc("y");
  auto shape_desc = op_desc->MutableInputDesc("shape");
  std::vector<int64_t> shape_shape = shape_desc->MutableShape().GetDims();
  std::vector<std::pair<int64_t,int64_t>> out_range;
  Tensor shape;
  std::vector<int64_t> const_data;
  if (GRAPH_SUCCESS != op.GetInputConstData("shape", shape)) {
    const_data = {-2};
  } else {
    auto data_type = shape_desc->GetDataType();
    if (!GetConstIntData(shape, data_type, const_data)) {
      USER_GE_LOGE("Invalid data type of shape, data_type is %d.",
                   (int) data_type);
      return GRAPH_FAILED;
    }
  }

  vector<int64_t> shape_dims;
  if (shape_shape.size() == 1 && shape_shape[0] > 0 && IsUnknownRankShape(const_data)) {
    for (int64_t i = 0; i<shape_shape[0]; i++) {
      shape_dims.push_back(-1);
    }
  } else {
    for (size_t i = 0; i < (uint32_t)const_data.size(); ++i) {
      shape_dims.push_back(const_data[i]);
    }
  }

  if (IsUnknownRankShape(shape_dims)) {
    out_range.push_back(std::pair<int64_t,int64_t>(1, -1));
  } else if (IsUnknownVec(shape_dims)) {
    for (size_t i=0; i<shape_dims.size(); i++) {
      if (shape_dims[i] == -1) {
        out_range.push_back(std::pair<int64_t,int64_t>(1, -1));
      } else {
        out_range.push_back(std::pair<int64_t,int64_t>(shape_dims[i], shape_dims[i]));
      }
    }
  }

  GeShape output_shape(shape_dims);
  output_desc->SetShape(output_shape);
  output_desc->SetShapeRange(out_range);
  output_desc->SetDataType(op_desc->MutableInputDesc("x")->GetDataType());
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(ScatterNd, ScatterNdInferShape);
//----------------ScatterNd End-------------------

//----------------ScatterNdD-------------------
IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(ScatterNdDInferShape)
  auto output_desc = op.GetInputDesc("x");
  const std::string shape_name = "shape";
  std::vector<int64_t> shape_out_list;
  if (GRAPH_SUCCESS != op.GetAttr(shape_name, shape_out_list)) {
      OpsGetAttrErrReport(op.GetName(), shape_name);
      USER_GE_LOGE("GetAttr %s failed.", shape_name.c_str());
      return GRAPH_FAILED;
  }
  vector<int64_t> shape_dims;
  for (size_t i = 0; i < (uint32_t)shape_out_list.size(); ++i) {
    shape_dims.push_back(shape_out_list[i]);
  }
  if (shape_out_list.size() != shape_dims.size() ) {
      string excepted_value = Strcat("same with output_y[", shape_dims.size(), "]");
      OpsAttrValueErrReport(op.GetName(), "x'shape", excepted_value, Strcat(shape_out_list.size()));
      OP_LOGE(op.GetName().c_str(),
            "the len of shape must be same with output_y.");
      return GRAPH_FAILED;
  }
  for (int64_t i = 0; i < (int64_t)shape_dims.size(); i++ ) {
  if (shape_out_list[i] != shape_dims[i] ) {
      string excepted_value = Strcat("same with output_y[", shape_dims[i], "]");
      OpsAttrValueErrReport(op.GetName(), "x'shape", excepted_value, Strcat(shape_out_list[i]));
      OP_LOGE(op.GetName().c_str(),
            "shape must be same with output_y.");
      return GRAPH_FAILED;
  }
  }
  Shape output_shape(shape_dims);
  output_desc.SetShape(output_shape);
  op.UpdateOutputDesc("y", output_desc);
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(ScatterNdD, ScatterNdDInferShape);
// ----------------ScatterNdD End-------------------

//----------------InTopKD Op-------------------
bool InTopKDCheckInputX1AndX2(const Operator& op) {
  Shape shape_prediction = op.GetInputDesc("x1").GetShape();
  Shape shape_target = op.GetInputDesc("x2").GetShape();
  int prediction_dim = shape_prediction.GetDimNum();
  if (prediction_dim != DIM_SIZE2) {
    OP_LOGE(op.GetName().c_str(), "predictions must be 2-dimensional"
            "but get %d\n", prediction_dim);
    return GRAPH_FAILED;
  }
  size_t target_dim = shape_target.GetDimNum();
  if (target_dim != DIM_SIZE1) {
    OP_LOGE(op.GetName().c_str(),
            "target must be 1-dimensional, but get %d\n",
             target_dim);
    return GRAPH_FAILED;
  }
  if (shape_prediction.GetDim(0) != shape_target.GetDim(0)){
      OP_LOGE(op.GetName().c_str(),
              "First dimension of prediction must match length of targets"
              "but first dimension of prediction get %d\n",
              shape_prediction.GetDim(0));
    return GRAPH_FAILED;
  }
  return true;
}

bool InTopKDCheckInputAttrK(const Operator& op) {
  int dim_zero = 0;
  Shape shape_k = op.GetInputDesc("k").GetShape();
  int k_dim = shape_k.GetDimNum();
  if (k_dim != dim_zero) {
    OP_LOGE(op.GetName().c_str(),
            "k must be 0 D, but get %d\n",
              k_dim);
    return GRAPH_FAILED;
  }
  return true;
}

IMPLEMT_VERIFIER(InTopKD, InTopKDVerify) {
  if (InTopKDCheckInputX1AndX2(op) == false) {
    return GRAPH_FAILED;
  }
  if (InTopKDCheckInputAttrK(op) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(InTopKDInferShape) {
  Shape shape_target = op.GetInputDesc("x2").GetShape();
  DataType output_dtype = DT_BOOL;
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(shape_target);
  tensordesc_output.SetDataType(output_dtype);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(InTopKD, InTopKDInferShape);
VERIFY_FUNC_REG(InTopKD, InTopKDVerify);
// ---------------InTopKD------------------

// ----------------InTopK Op Start-------------------
bool InTopKCheckInputX1AndX2(const Operator& op) {
  Shape shape_prediction = op.GetInputDesc("x1").GetShape();
  Shape shape_target = op.GetInputDesc("x2").GetShape();
  int prediction_dim = shape_prediction.GetDimNum();
  if (prediction_dim != DIM_SIZE2) {
    OP_LOGE(op.GetName().c_str(), "predictions must be 2-dimensional,"
            "but get %d\n", prediction_dim);
    return GRAPH_FAILED;
  }
  size_t target_dim = shape_target.GetDimNum();
  if (target_dim != DIM_SIZE1) {
    OP_LOGE(op.GetName().c_str(), "target must be 1-dimensional"
            "but get %d\n", target_dim);
    return GRAPH_FAILED;
  }
  if (shape_prediction.GetDim(0) != shape_target.GetDim(0)) {
    OP_LOGE(op.GetName().c_str(),
            "First dimension of prediction must match length of targets,"
            "but first dimension of prediction get %d\n",
            shape_prediction.GetDim(0));
    return GRAPH_FAILED;
  }
  return true;
}

IMPLEMT_VERIFIER(InTopK, InTopKVerify) {
  if (InTopKCheckInputX1AndX2(op) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(InTopKInferShape) {
  DataType output_dtype = DT_BOOL;
  Shape shape_target = op.GetInputDesc("x2").GetShape();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(shape_target);
  tensordesc_output.SetDataType(output_dtype);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(InTopK, InTopKInferShape);
VERIFY_FUNC_REG(InTopK, InTopKVerify);
// ----------------InTopK Op End-------------------

// ----------------StridedSliceAssign-------------------
IMPLEMT_COMMON_INFERFUNC(StridedSliceAssignInferShape) {
  TensorDesc tensordesc_output = op.GetOutputDesc("var");
  tensordesc_output.SetShape(op.GetInputDesc("var").GetShape());
  tensordesc_output.SetDataType(op.GetInputDesc("var").GetDataType());

  (void)op.UpdateOutputDesc("var", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(StridedSliceAssign, StridedSliceAssignInferShape);
// ----------------StridedSliceAssign Op Begin-------------------

// ----------------StridedSliceAssignD Op Begin-------------------
IMPLEMT_COMMON_INFERFUNC(StridedSliceAssignDInferShape) {
  std::vector<int64_t> begin;
  begin = GetAttrValue(op, "begin");
  if (!CheckListEmpty(op.GetName(), begin, "begin")) {
    OP_LOGE(op.GetName().c_str(), "get attr begin failed");
    return GRAPH_FAILED;
  }
  if (begin.size() > 8) {
    OP_LOGE(op.GetName().c_str(), "attr begin(%d) is too large",
            (int)begin.size());
    return GRAPH_FAILED;
  }
  std::vector<int64_t> end;
  end = GetAttrValue(op, "end");
  if (!CheckListEmpty(op.GetName(), end, "end")) {
    OP_LOGE(op.GetName().c_str(), "get attr end failed");
    return GRAPH_FAILED;
  }
  if (end.size() > 8) {
    OP_LOGE(op.GetName().c_str(), "attr end(%d) is too large", (int)end.size());
    return GRAPH_FAILED;
  }
  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), strides, "strides")) {
    OP_LOGE(op.GetName().c_str(), "get attr strides failed");
    return GRAPH_FAILED;
  }
  if (strides.size() > 8) {
    OP_LOGE(op.GetName().c_str(), "attr strides(%d) is too large",
            (int)strides.size());
    return GRAPH_FAILED;
  }
  TensorDesc tensordesc_output = op.GetOutputDesc("var");
  tensordesc_output.SetShape(op.GetInputDesc("var").GetShape());
  tensordesc_output.SetDataType(op.GetInputDesc("var").GetDataType());

  (void)op.UpdateOutputDesc("var", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(StridedSliceAssignD, StridedSliceAssignDInferShape);
// ----------------StridedSliceAssignD Op Begin-------------------

//----------------Cumprod-------------------
IMPLEMT_COMMON_INFERFUNC(CumprodInferShape) {
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(op.GetInputDesc("x").GetShape());
  output_desc.SetDataType(op.GetInputDesc("x").GetDataType());
  op.UpdateOutputDesc("y", output_desc);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Cumprod, CumprodInferShape);
//----------------Cumprod END-------------------

//----------------CumprodD-------------------
IMPLEMT_VERIFIER(CumprodD, CumprodDVerify) {
  int64_t axis;
  if (GRAPH_SUCCESS != op.GetAttr("axis", axis)) {
    OpsGetAttrErrReport(op.GetName(), "axis");
    OP_LOGE(op.GetName().c_str(), "GetAttr of axis failed.");
    return GRAPH_FAILED;
  }
  TensorDesc input_desc = op.GetInputDesc("x");
  int64_t dimnum;
  dimnum = input_desc.GetShape().GetDimNum();
  if (axis < -dimnum || axis >= dimnum) {
    OpsInputShapeDimErrReport(op.GetName(), "axis", Strcat(dimnum), Strcat(-dimnum), Strcat(axis));
    OP_LOGE(op.GetName().c_str(), "attr axis is not in range");
    return GRAPH_FAILED;
    }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(CumprodDInferShape) {
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(op.GetInputDesc("x").GetShape());
  output_desc.SetDataType(op.GetInputDesc("x").GetDataType());
  op.UpdateOutputDesc("y", output_desc);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(CumprodD, CumprodDInferShape);
VERIFY_FUNC_REG(CumprodD, CumprodDVerify);
//----------------CumprodD END-------------------

//----------------Cumsum-------------------
IMPLEMT_COMMON_INFERFUNC(CumsumInferShape) {
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(op.GetInputDesc("x").GetShape());
  output_desc.SetDataType(op.GetInputDesc("x").GetDataType());
  op.UpdateOutputDesc("y", output_desc);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Cumsum, CumsumInferShape);
//----------------Cumsum END-------------------

//----------------CumsumD-------------------
IMPLEMT_VERIFIER(CumsumD, CumsumDVerify) {
  int64_t axis;
  if (GRAPH_SUCCESS != op.GetAttr("axis", axis)) {
    OpsGetAttrErrReport(op.GetName(), "axis");
    OP_LOGE(op.GetName().c_str(), "GetAttr of axis failed.");
    return GRAPH_FAILED;
  }
  TensorDesc input_desc = op.GetInputDesc("x");
  int64_t dimnum;
  dimnum = input_desc.GetShape().GetDimNum();
  if (axis < -dimnum || axis >= dimnum) {
    OpsInputShapeDimErrReport(op.GetName(), "axis", Strcat(dimnum), Strcat(-dimnum), Strcat(axis));
    OP_LOGE(op.GetName().c_str(), "attr axis is not in range");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(CumsumDInferShape) {
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(op.GetInputDesc("x").GetShape());
  output_desc.SetDataType(op.GetInputDesc("x").GetDataType());
  op.UpdateOutputDesc("y", output_desc);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(CumsumD, CumsumDInferShape);
VERIFY_FUNC_REG(CumsumD, CumsumDVerify);
//----------------CumsumD END-------------------


// ----------------InplaceUpdate-------------------
IMPLEMT_COMMON_INFERFUNC(InplaceUpdateInferShape) {
  auto output_desc = op.GetInputDesc("x");
  auto output_shape_dims = output_desc.GetShape().GetDims();
  Shape output_shape(output_shape_dims);
  output_desc.SetShape(output_shape);

  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(InplaceUpdate, InplaceUpdateInferShape);
// ----------------InplaceUpdate END-------------------

// ----------------InplaceUpdateD-------------------
IMPLEMT_COMMON_INFERFUNC(InplaceUpdateDInferShape) {
  auto output_desc = op.GetInputDesc("x");
  auto input_v_desc = op.GetInputDesc("v");
  int64_t dim_value_v;
  dim_value_v = input_v_desc.GetShape().GetDim(0);
  std::vector<int64_t> indices;
  if (op.GetAttr("indices", indices) == GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "indices");
    OP_LOGE(op.GetName().c_str(), "get attr indices failed");
  }

  if ((int64_t)indices.size() != dim_value_v) {
    string excepted_value = Strcat("same as indices[", (int64_t)indices.size(), "]");
    OpsAttrValueErrReport(op.GetName(), "v's length of rank 0", excepted_value, Strcat(dim_value_v));
    OP_LOGE(op.GetName().c_str(), "The length of rank 0 of"
     "tensor v must be the same as length of indices.");
     return GRAPH_FAILED;
  }
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(InplaceUpdateD, InplaceUpdateDInferShape);
// ----------------InplaceUpdateD  END-------------------

// ----------------InplaceAdd-------------------
IMPLEMT_COMMON_INFERFUNC(InplaceAddInferShape) {
  auto output_desc = op.GetInputDesc("x");
  const std::string indices_name = "indices";
  Tensor indices;
  if (GRAPH_SUCCESS != op.GetInputConstData(indices_name, indices)) {
    OP_LOGE("GetInputConstData %s failed.", indices_name.c_str());
    return GRAPH_FAILED;
  }

  auto output_shape_dims = output_desc.GetShape().GetDims();
  Shape output_shape(output_shape_dims);
  output_desc.SetShape(output_shape);

  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(InplaceAdd, InplaceAddInferShape);
// ----------------InplaceAdd  END-------------------

// ----------------InplaceAddD-------------------
IMPLEMT_COMMON_INFERFUNC(InplaceAddDInferShape) {
  auto output_desc = op.GetInputDesc("x");
  auto input_v_desc = op.GetInputDesc("v");
  int64_t dim_value_v;
  dim_value_v = input_v_desc.GetShape().GetDim(0);
  std::vector<int64_t> indices;
  if (op.GetAttr("indices", indices) == GRAPH_FAILED) {
    OpsSetAttrErrReport(op.GetName(), "indices");
    OP_LOGE(op.GetName().c_str(), "get attr indices failed");
  }

  if ((int64_t)indices.size() != dim_value_v) {
    OpsAttrValueErrReport(op.GetName(), "v", Strcat(dim_value_v), Strcat((int64_t)indices.size()));
    OP_LOGE(op.GetName().c_str(), "The length of rank 0 of"
      "tensor v must be the same as length of indices.");
    return GRAPH_FAILED;
  }
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(InplaceAddD, InplaceAddDInferShape);
// ----------------InplaceAddD  END-------------------

// ----------------InplaceSub-------------------
IMPLEMT_COMMON_INFERFUNC(InplaceSubInferShape) {
  auto output_desc = op.GetInputDesc("x");
  const std::string indices_name = "indices";
  Tensor indices;
  if (GRAPH_SUCCESS != op.GetInputConstData(indices_name, indices)) {
    OP_LOGE("GetInputConstData %s failed.", indices_name.c_str());
    return GRAPH_FAILED;
  }

  auto output_shape_dims = output_desc.GetShape().GetDims();
  Shape output_shape(output_shape_dims);
  output_desc.SetShape(output_shape);
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(InplaceSub, InplaceSubInferShape);
// ----------------InplaceSub END-------------------

// ----------------InplaceSubD-------------------
IMPLEMT_COMMON_INFERFUNC(InplaceSubDInferShape) {
  auto output_desc = op.GetInputDesc("x");
  auto input_v_desc = op.GetInputDesc("v");
  int64_t dim_value_v;
  dim_value_v = input_v_desc.GetShape().GetDim(0);
  std::vector<int64_t> indices;
  if (op.GetAttr("indices", indices) == GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "indices");
    OP_LOGE(op.GetName().c_str(), "get attr indices failed");
  }

  if ((int64_t)indices.size() != dim_value_v) {
    string excepted_value = Strcat("same as indices[", (int64_t)indices.size(), "]");
    OpsAttrValueErrReport(op.GetName(), "v's length of rank 0", excepted_value, Strcat(dim_value_v));
    OP_LOGE(op.GetName().c_str(), "The length of rank 0 of"
     "tensor v must be the same as length of indices.");
    return GRAPH_FAILED;
  }
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(InplaceSubD, InplaceSubDInferShape);
// ----------------InplaceSubD  END-------------------

// ----------------UnsortedSegmentMin-------------------
IMPLEMT_COMMON_INFERFUNC(UnsortedSegmentMinInferShape) {
  auto input_desc = op.GetInputDesc("x");
  const std::string num_segments_name = "num_segments";
  Tensor num_segments_data;
  auto data_type = op.GetInputDesc(num_segments_name).GetDataType();
  int64_t num_segments;
  if (GRAPH_SUCCESS != op.GetInputConstData(num_segments_name,
    num_segments_data)) {
    OP_LOGI(op.GetName().c_str(),
           "GetInputConstData %s failed.", num_segments_name.c_str());
    num_segments = -1;
  } else {
    if (!GetScalerValue(op, num_segments_data, data_type, num_segments)) {
      OP_LOGE(op.GetName().c_str(),
             "invalid data type of num_segments_data, data_type is %d.",
             (int) data_type);
      return GRAPH_FAILED;
    }
  }

  Shape shape = op.GetInputDesc("x").GetShape();
  Shape shape_id = op.GetInputDesc("segment_ids").GetShape();
  int64_t dim_idsize_input = shape_id.GetDimNum();
  int64_t dim_size_input=shape.GetDimNum();
  vector<int64_t> shape_vector;
  shape_vector.push_back(num_segments);
  for (int i = dim_idsize_input; i < dim_size_input; i++) {
    shape_vector.push_back(shape.GetDim(i));
  }
  Shape output_shape(shape_vector);
  DataType input_dtype = input_desc.GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(output_shape);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(UnsortedSegmentMin, UnsortedSegmentMinInferShape);
// ----------------UnsortedSegmentMin END-------------------

// ----------------UnsortedSegmentMinD-------------------
IMPLEMT_COMMON_INFERFUNC(UnsortedSegmentMinDInferShape) {
  static const char* op_name = "unsorted_segment_min";
  auto input_desc = op.GetInputDesc("x");
  const std::string num_segments_name = "num_segments";
  int64_t num_segments;
  if (GRAPH_SUCCESS != op.GetAttr(num_segments_name, num_segments)) {
    OP_LOGE(op_name, "GetAttr %s failed.", num_segments_name.c_str());
    return GRAPH_FAILED;
  }

  Shape shape = op.GetInputDesc("x").GetShape();
  Shape shape_id = op.GetInputDesc("segment_ids").GetShape();
  int64_t dim_idsize_input = shape_id.GetDimNum();
  int64_t dim_size_input=shape.GetDimNum();
  vector<int64_t> shape_vector;
  shape_vector.push_back(num_segments);
  for (int i = dim_idsize_input; i < dim_size_input; i++) {
    shape_vector.push_back(shape.GetDim(i));
  }
  Shape output_shape(shape_vector);
  DataType input_dtype = input_desc.GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(output_shape);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(UnsortedSegmentMinD, UnsortedSegmentMinDInferShape);
// ----------------UnsortedSegmentMinD END-------------------

// ----------------UnsortedSegmentMax-------------------
IMPLEMT_COMMON_INFERFUNC(UnsortedSegmentMaxInferShape) {
  auto input_desc = op.GetInputDesc("x");
  const std::string num_segments_name = "num_segments";
  Tensor num_segments_data;
  auto data_type = op.GetInputDesc(num_segments_name).GetDataType();
  int64_t num_segments;
  if (op.GetInputConstData(num_segments_name, num_segments_data) !=
      GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "GetInputConstData %s failed.",
            num_segments_name.c_str());
    num_segments = -1;
  } else {
    if (!GetScalerValue(op, num_segments_data, data_type, num_segments)) {
      OP_LOGE(op.GetName().c_str(),
              "invalid data type of num_segments_data, data_type is %d.",
              (int)data_type);
      return GRAPH_FAILED;
    }
  }
  Shape shape = op.GetInputDesc("x").GetShape();
  Shape shape_id = op.GetInputDesc("segment_ids").GetShape();
  int64_t dim_idsize_input = shape_id.GetDimNum();
  int64_t dim_size_input = shape.GetDimNum();
  vector<int64_t> shape_vector;
  shape_vector.push_back(num_segments);
  for (int i = dim_idsize_input; i < dim_size_input; i++) {
    shape_vector.push_back(shape.GetDim(i));
  }
  Shape output_shape(shape_vector);
  DataType input_dtype = input_desc.GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(output_shape);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(UnsortedSegmentMax, UnsortedSegmentMaxInferShape);
// ----------------UnsortedSegmentMax END-------------------

// ----------------UnsortedSegmentMaxD-------------------
IMPLEMT_COMMON_INFERFUNC(UnsortedSegmentMaxDInferShape) {
  static const char* op_name = "unsorted_segment_max";
  auto input_desc = op.GetInputDesc("x");
  const std::string num_segments_name = "num_segments";
  int64_t num_segments;
  if (op.GetAttr(num_segments_name, num_segments) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "GetAttr %s failed.", num_segments_name.c_str());
    return GRAPH_FAILED;
  }
  Shape shape = op.GetInputDesc("x").GetShape();
  Shape shape_id = op.GetInputDesc("segment_ids").GetShape();
  int64_t dim_idsize_input = shape_id.GetDimNum();
  int64_t dim_size_input = shape.GetDimNum();
  vector<int64_t> shape_vector;
  shape_vector.push_back(num_segments);
  for (int i = dim_idsize_input; i < dim_size_input; i++) {
    shape_vector.push_back(shape.GetDim(i));
  }
  Shape output_shape(shape_vector);
  DataType input_dtype = input_desc.GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(output_shape);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(UnsortedSegmentMaxD, UnsortedSegmentMaxDInferShape);
// ----------------UnsortedSegmentMaxD END-------------------

// ----------------UnsortedSegmentProd-------------------
IMPLEMT_COMMON_INFERFUNC(UnsortedSegmentProdInferShape) {
  auto input_desc = op.GetInputDesc("x");
  const std::string num_segments_name = "num_segments";
  Tensor num_segments_data;
  auto data_type = op.GetInputDesc(num_segments_name).GetDataType();
  int64_t num_segments;
  if (GRAPH_SUCCESS != op.GetInputConstData(num_segments_name,
    num_segments_data)) {
      OP_LOGI(op.GetName().c_str(),
            "GetInputConstData %s failed.", num_segments_name.c_str());
    num_segments = -1;
  } else {
    if (!GetScalerValue(op, num_segments_data, data_type, num_segments)) {
      OP_LOGE(op.GetName().c_str(),
             "invalid data type of num_segments_data, data_type is %d.",
             (int) data_type);
      return GRAPH_FAILED;
    }
  }

  Shape shape = op.GetInputDesc("x").GetShape();
  Shape shape_id = op.GetInputDesc("segment_ids").GetShape();
  int64_t dim_idsize_input = shape_id.GetDimNum();
  int64_t dim_size_input=shape.GetDimNum();
  vector<int64_t> shape_vector;
  shape_vector.push_back(num_segments);
  for (int i = dim_idsize_input; i < dim_size_input; i++) {
    shape_vector.push_back(shape.GetDim(i));
  }
  Shape output_shape(shape_vector);
  DataType input_dtype = input_desc.GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(output_shape);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(UnsortedSegmentProd, UnsortedSegmentProdInferShape);
// ----------------UnsortedSegmentProd END-------------------

// ----------------UnsortedSegmentProdD----------------------
IMPLEMT_COMMON_INFERFUNC(UnsortedSegmentProdDInferShape) {
  static const char* op_name = "unsorted_segment_prod";
  auto input_desc = op.GetInputDesc("x");
  const std::string num_segments_name = "num_segments";
  int64_t num_segments;
  if (GRAPH_SUCCESS != op.GetAttr(num_segments_name, num_segments)) {
    OP_LOGE(op_name, "GetAttr %s failed.", num_segments_name.c_str());
    return GRAPH_FAILED;
  }

  Shape shape = op.GetInputDesc("x").GetShape();
  Shape shape_id = op.GetInputDesc("segment_ids").GetShape();
  int64_t dim_idsize_input = shape_id.GetDimNum();
  int64_t dim_size_input=shape.GetDimNum();
  vector<int64_t> shape_vector;
  shape_vector.push_back(num_segments);
  for (int i = dim_idsize_input; i < dim_size_input; i++) {
    shape_vector.push_back(shape.GetDim(i));
  }
  Shape output_shape(shape_vector);
  DataType input_dtype = input_desc.GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(output_shape);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(UnsortedSegmentProdD, UnsortedSegmentProdDInferShape);

// ----------------scatter_non_aliasing_add-------------------
IMPLEMT_COMMON_INFERFUNC(ScatterNonAliasingAddInferShape) {
  auto input_tensor = op.GetInputDesc("x");
  auto output_desc = op.GetInputDesc("y");
  output_desc.SetShape(op.GetInputDesc("x").GetShape());
  output_desc.SetDataType(op.GetInputDesc("x").GetDataType());
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ScatterNonAliasingAdd, ScatterNonAliasingAddInferShape);
// ------------------scatter_non_aliasing_add END---------------------

//----------------proposal-------------------
IMPLEMT_VERIFIER(Proposal, ProposalVerify) {

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ProposalInferShape) {
  OP_LOGI("propsoal", "infer shape begin---");
  auto cls_prob_shape = op.GetInputDesc("cls_prob").GetShape().GetDims();
  int64_t batch = cls_prob_shape[0];
  DataType input_dtype = op.GetInputDesc("cls_prob").GetDataType();

  int64_t post_nms_topn = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("post_nms_topn", post_nms_topn)) {
    OP_LOGE(op.GetName().c_str(), "get attr failed");
  }

  int64_t tmp_post_nms_topn = post_nms_topn;
  tmp_post_nms_topn = ((post_nms_topn + 15) / 16) * 16;

  std::vector<int64_t> dim_vector;
  dim_vector.push_back(batch);
  dim_vector.push_back(5);
  dim_vector.push_back(tmp_post_nms_topn);
  Shape out_shape_rois(dim_vector);
  TensorDesc rois_desc = op.GetOutputDesc("rois");
  rois_desc.SetShape(out_shape_rois);
  rois_desc.SetDataType(input_dtype);

  Shape out_shape_actual_rois_num({batch, 8});
  TensorDesc actual_rois_num_desc = op.GetOutputDesc("actual_rois_num");
  actual_rois_num_desc.SetShape(out_shape_actual_rois_num);
  actual_rois_num_desc.SetDataType(ge::DT_INT32);

  (void)op.UpdateOutputDesc("rois", rois_desc);
  (void)op.UpdateOutputDesc("actual_rois_num", actual_rois_num_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Proposal, ProposalInferShape);
VERIFY_FUNC_REG(Proposal, ProposalVerify);

//----------------proposal_d-------------------
IMPLEMT_VERIFIER(ProposalD, ProposalDVerify) {

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ProposalDInferShape) {
  OP_LOGI("propsoal", "infer shape begin---");
  auto cls_prob_shape = op.GetInputDesc("cls_prob").GetShape().GetDims();
  int64_t batch = cls_prob_shape[0];
  DataType input_dtype = op.GetInputDesc("cls_prob").GetDataType();

  int64_t post_nms_topn = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("post_nms_topn", post_nms_topn)) {
    OP_LOGE(op.GetName().c_str(), "get attr failed");
  }

  int64_t tmp_post_nms_topn = post_nms_topn;
  tmp_post_nms_topn = ((post_nms_topn + 15) / 16) * 16;

  std::vector<int64_t> dim_vector;
  dim_vector.push_back(batch);
  dim_vector.push_back(5);
  dim_vector.push_back(tmp_post_nms_topn);
  Shape out_shape_rois(dim_vector);
  TensorDesc rois_desc = op.GetOutputDesc("rois");
  rois_desc.SetShape(out_shape_rois);
  rois_desc.SetDataType(input_dtype);

  Shape out_shape_actual_rois_num({batch, 8});
  TensorDesc actual_rois_num_desc = op.GetOutputDesc("actual_rois_num");
  actual_rois_num_desc.SetShape(out_shape_actual_rois_num);
  actual_rois_num_desc.SetDataType(ge::DT_INT32);

  (void)op.UpdateOutputDesc("rois", rois_desc);
  (void)op.UpdateOutputDesc("actual_rois_num", actual_rois_num_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ProposalD, ProposalDInferShape);
VERIFY_FUNC_REG(ProposalD, ProposalDVerify);

//----------------PassThrough-------------------
IMPLEMT_COMMON_INFERFUNC(PassThroughInferShape) {
    // get input depth
    OP_LOGI("pass_through", "infer shape begin---");
    auto inputShape = op.GetInputDesc(0).GetShape().GetDims();
    DataType inputDtype = op.GetInputDesc(0).GetDataType();
    Format inputFormat = op.GetInputDesc(0).GetFormat();
    int64_t stride;
    bool reverse;

    if (GRAPH_SUCCESS != op.GetAttr("stride", stride)) {
        stride = 2;
    }
    if (GRAPH_SUCCESS != op.GetAttr("reverse", reverse)) {
        reverse = false;
    }
    std::vector<int64_t> outputShapeVec;
    if (reverse) {
        if (inputFormat == FORMAT_NCHW) {
            outputShapeVec.push_back(inputShape[0]);
            outputShapeVec.push_back(inputShape[1]/(stride*stride));
            outputShapeVec.push_back(inputShape[2]*stride);
            outputShapeVec.push_back(inputShape[3]*stride);
        } else {
            outputShapeVec.push_back(inputShape[0]);
            outputShapeVec.push_back(inputShape[1]*stride);
            outputShapeVec.push_back(inputShape[2]*stride);
            outputShapeVec.push_back(inputShape[3]/(stride*stride));
        }
    } else {
        if (inputFormat == FORMAT_NCHW) {
            outputShapeVec.push_back(inputShape[0]);
            outputShapeVec.push_back(inputShape[1]*(stride*stride));
            outputShapeVec.push_back(inputShape[2]/stride);
            outputShapeVec.push_back(inputShape[3]/stride);
        } else {
            outputShapeVec.push_back(inputShape[0]);
            outputShapeVec.push_back(inputShape[1]/stride);
            outputShapeVec.push_back(inputShape[2]/stride);
            outputShapeVec.push_back(inputShape[3]*(stride*stride));
        }
    }
    Shape outputShape(outputShapeVec);
    TensorDesc outputDesc = op.GetOutputDesc(0);
    outputDesc.SetShape(outputShape);
    outputDesc.SetDataType(inputDtype);
    outputDesc.SetFormat(inputFormat);
    (void)op.UpdateOutputDesc("y", outputDesc);

    OP_LOGI("pass_through", "infer shape end---");

    return GRAPH_SUCCESS;

}

IMPLEMT_VERIFIER(PassThrough, PassThroughVerify) {
    auto inputShape = op.GetInputDesc(0).GetShape().GetDims();
    Format inputFormat = op.GetInputDesc(0).GetFormat();
    int64_t stride;
    bool reverse;

    if (GRAPH_SUCCESS != op.GetAttr("stride", stride)) {
        stride = 2;
    }
    if (GRAPH_SUCCESS != op.GetAttr("reverse", reverse)) {
        reverse = false;
    }

    if (inputFormat != FORMAT_NCHW && inputFormat != FORMAT_NHWC) {
        OP_LOGE("[ERROR]the PassThrough only support format NCHW&NHWC!");
        OpsInputFormatErrReport(op.GetName().c_str(), "inputFormat", "NCHW or NHWC", Strcat(inputFormat));
        return GRAPH_FAILED;
    }

    if (reverse){
        if (stride < 1) {
            OP_LOGE("[ERROR]the PassThrough op forward do not supported the stride!");
            OpsAttrValueErrReport(op.GetName().c_str(), "stride", "greater than 0", Strcat(stride));
            return GRAPH_FAILED;
        }
        int64_t modC = (inputFormat == FORMAT_NCHW) ?
                       (int64_t)inputShape[1]%(stride*stride) :
                       (int64_t)inputShape[3]%(stride*stride);
        if (modC != 0) {
            OP_LOGE("[ERROR]the PassThrough op forward do not supported the stride!");
            OpsAttrValueErrReport(op.GetName().c_str(), "axis C", "times of stride'squre", Strcat(modC));
            return GRAPH_FAILED;
        }

    } else {
        if (stride < 1) {
            OP_LOGE("[ERROR]the PassThrough op backward do not supported the stride!");
            OpsAttrValueErrReport(op.GetName().c_str(), "stride", "greater than 0", Strcat(stride));
            return GRAPH_FAILED;
        }
        int64_t modH = (inputFormat == FORMAT_NCHW) ?
                       (int64_t)inputShape[2]%stride :
                       (int64_t)inputShape[1]%stride;
        int64_t modW = (inputFormat == FORMAT_NCHW) ?
                       (int64_t)inputShape[3]%stride :
                       (int64_t)inputShape[2]%stride;
        if (modH != 0) {
            OP_LOGE("[ERROR]the PassThrough op backward do not supported the stride!");
            OpsAttrValueErrReport(op.GetName().c_str(), "axis H", "times of stride", Strcat(modH));
            return GRAPH_FAILED;
        }
        if (modW != 0) {
            OP_LOGE("[ERROR]the PassThrough op backward do not supported the stride!");
            OpsAttrValueErrReport(op.GetName().c_str(), "axis W", "times of stride", Strcat(modW));
            return GRAPH_FAILED;
        }

    }

    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(PassThrough, PassThroughInferShape);

VERIFY_FUNC_REG(PassThrough, PassThroughVerify);

//----------------Crop-------------------
IMPLEMT_VERIFIER(Crop, CropVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(CropInferShape) {
  auto inputDesc = op.GetInputDesc("x");
  auto outputDesc = op.GetInputDesc("size");
  ge::Shape outputShape = outputDesc.GetShape();
  ge::Shape inputShape = inputDesc.GetShape();
  int64_t dimNum = inputShape.GetDimNum();
  int64_t axis;
  if (GRAPH_SUCCESS != op.GetAttr("axis", axis)) {
    OP_LOGE("Failed to get attribute axis");
    return GRAPH_FAILED;
  }
  if (axis >= dimNum || axis < -dimNum) {
    OP_LOGE("Failed to check attribute axis");
    return GRAPH_FAILED;
  }
  if (axis < 0) {
    axis += dimNum;
  }
  for (int64_t i = 0; i < axis; i++) {
    outputShape.SetDim(i, inputShape.GetDim(i));
  }
  outputDesc.SetShape(outputShape);
  return op.UpdateOutputDesc("y", outputDesc);
}

COMMON_INFER_FUNC_REG(Crop, CropInferShape);
VERIFY_FUNC_REG(Crop, CropVerify);
//----------------Crop-------------------

/**********************************TileWithAxis**************************************/
IMPLEMT_INFERFUNC(TileWithAxis, TileWithAxisInfer) {
  TensorDesc outputDesc = op.GetOutputDesc("y");
  TensorDesc inputDesc = op.GetInputDesc("x");

  auto input_dType = inputDesc.GetDataType();

  ge::Shape shapeX = inputDesc.GetShape();
  std::vector<int64_t> dimsX = shapeX.GetDims();

  int64_t tiles;
  if (GRAPH_SUCCESS != op.GetAttr("tiles", tiles)) {
    OP_LOGE(op.GetName().c_str(), "Failed to get attribute tiles");
    return GRAPH_FAILED;
  }

  int64_t axis;
  if (GRAPH_SUCCESS != op.GetAttr("axis", axis)) {
    OP_LOGE(op.GetName().c_str(), "Failed to get attribute axis");
    return GRAPH_FAILED;
  }

  Shape originShape = inputDesc.GetOriginShape();
  Format inputFormat = inputDesc.GetFormat();
  Format originFormat = inputDesc.GetOriginFormat();

  if(inputFormat == FORMAT_NC1HWC0){
    if(originShape.GetDimNum() == 4) {
      if (originFormat == FORMAT_NCHW) {
        if (axis < 0) {
          axis = axis - 1;
        }
      } else
        if (originFormat == FORMAT_NHWC) {
          if (axis == -4) {
            axis = -5;
          } else
            if (axis == -1) {
              axis = -4;
            } else
              if (axis == 1) {
                axis = 2;
              } else
                if (axis == 2) {
                  axis = 3;
                } else
                  if (axis == 3) {
                    axis = 1;
                  }
        } else {
          OP_LOGE(op.GetName().c_str(), "5D tensor's origin format should in [NCHW, NHWC]");
          return GRAPH_FAILED;
        }
    }else{
      OP_LOGE(op.GetName().c_str(), "5D tensor's origin shape should be 4D tensor");
      return GRAPH_FAILED;
    }

    if(axis < 0){
      axis = axis + 5;
    }
    if(axis == 1 || axis == 4){
      OP_LOGE(op.GetName().c_str(), "5D tensor's axis is invalid");
      return GRAPH_FAILED;
    }
  }else if(axis < 0){
    axis = axis + dimsX.size();
  }

  dimsX[axis] *= tiles;
  ge::Shape outputShape = ge::Shape(dimsX);

  outputDesc.SetShape(outputShape);
  outputDesc.SetDataType(input_dType);
  op.UpdateOutputDesc("y", outputDesc);

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(TileWithAxis, TileWithAxisVerify) {
  auto xShape = op.GetInputDesc("x").GetShape().GetDims();

  int64_t tiles;
  if (GRAPH_SUCCESS != op.GetAttr("tiles", tiles)) {
    USER_GE_LOGE("Failed to get attribute tiles");
    return GRAPH_FAILED;
  }

  int64_t axis;
  if (GRAPH_SUCCESS != op.GetAttr("axis", axis)) {
    USER_GE_LOGE("Failed to get attribute axis");
    return GRAPH_FAILED;
  }

  bool flag = (axis >= (static_cast<int> (xShape.size()) * (-1)))
  && (axis < static_cast<int> (xShape.size()));
  if (!flag) {
    USER_GE_LOGE("axis must be within range of input rank: axis is %d, shape size is %d.", axis, xShape.size());
    return GRAPH_FAILED;
  }

  if (tiles <= 0) {
    USER_GE_LOGE("tiles must be positive: tiles is %d.", tiles);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

//Registered inferfunction
INFER_FUNC_REG(TileWithAxis, TileWithAxisInfer);

//Registered verify function
VERIFY_FUNC_REG(TileWithAxis, TileWithAxisVerify);

/**********************************TileWithAxis**************************************/
//----------------read_select-------------------
IMPLEMT_COMMON_INFERFUNC(ReadSelectInferShape) {
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(ReadSelect, ReadSelectVerify){
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ReadSelect, ReadSelectInferShape);

VERIFY_FUNC_REG(ReadSelect, ReadSelectVerify);

//----------------write_select-------------------
IMPLEMT_COMMON_INFERFUNC(WriteSelectInferShape) {
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(WriteSelect, WriteSelectVerify){
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(WriteSelect, WriteSelectInferShape);

VERIFY_FUNC_REG(WriteSelect, WriteSelectVerify);

//----------------strided_read-------------------
IMPLEMT_COMMON_INFERFUNC(StridedReadInferShape) {
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(StridedRead, StridedReadVerify){
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(StridedRead, StridedReadInferShape);

VERIFY_FUNC_REG(StridedRead, StridedReadVerify);

//----------------strided_write-------------------
IMPLEMT_COMMON_INFERFUNC(StridedWriteInferShape) {
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(StridedWrite, StridedWriteVerify){
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(StridedWrite, StridedWriteInferShape);

VERIFY_FUNC_REG(StridedWrite, StridedWriteVerify);

//----------------CumulativeLogsumexp-------------------
IMPLEMT_COMMON_INFERFUNC(CumulativeLogsumexpInferShape) {
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(op.GetInputDesc("x").GetShape());
  output_desc.SetDataType(op.GetInputDesc("x").GetDataType());
  op.UpdateOutputDesc("y", output_desc);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(CumulativeLogsumexp, CumulativeLogsumexpInferShape);
//----------------CumulativeLogsumexp END-------------------

//----------------CumulativeLogsumexpD-------------------
IMPLEMT_VERIFIER(CumulativeLogsumexpD, CumulativeLogsumexpDVerify) {
  int64_t axis;
  if (GRAPH_SUCCESS != op.GetAttr("axis", axis)) {
    OP_LOGE(op.GetName().c_str(), "GetAttr of axis failed.");
    return GRAPH_FAILED;
  }
  TensorDesc input_desc = op.GetInputDesc("x");
  int64_t dimnum;
  dimnum = input_desc.GetShape().GetDimNum();
  if (axis < -dimnum || axis >= dimnum) {
    OP_LOGE(op.GetName().c_str(), "attr axis is not in range");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(CumulativeLogsumexpDInferShape) {
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(op.GetInputDesc("x").GetShape());
  output_desc.SetDataType(op.GetInputDesc("x").GetDataType());
  op.UpdateOutputDesc("y", output_desc);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(CumulativeLogsumexpD, CumulativeLogsumexpDInferShape);
VERIFY_FUNC_REG(CumulativeLogsumexpD, CumulativeLogsumexpDVerify);
//----------------CumulativeLogsumexpD END-------------------

} // namespace ge
