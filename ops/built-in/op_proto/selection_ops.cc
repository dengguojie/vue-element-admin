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
 * \file selection_ops.cpp
 * \brief
 */
#include "inc/selection_ops.h"

#include <cmath>
#include <string>
#include <vector>

#include "util/util.h"
#include "util/error_util.h"
#include "op_log.h"
#include "strided_slice_infer_shape.h"
#include "graph/utils/op_desc_utils.h"
#include "register/infer_data_slice_registry.h"
#include "graph/common_error_codes.h"
#include "graph/debug/ge_attr_define.h"
#include "axis_util.h"

#define ELLIPSIS_MASK_UPDATE(mask, new_mask, bit_ellipsis, i, pow_table, \
                             right_mov)                                  \
  do {                                                                   \
    if (((mask) & (1 << i)) && (bit_ellipsis >= i)) {                    \
      new_mask += pow_table[i];                                          \
    } else if (((mask) & (1 << i)) && (bit_ellipsis < i)) {              \
      new_mask += pow_table[i + right_mov];                              \
    }                                                                    \
  } while (0)

namespace ge {
static bool CheckListEmpty(const std::string& op_name, const std::vector<int64_t>& list, const std::string& attr_name) {
  if (list.empty()) {
    OP_LOGE(op_name.c_str(), "The %s is empty !", attr_name.c_str());
    return false;
  }
  return true;
}
static std::vector<int64_t> GetAttrValue(const ge::Operator& op, const std::string& key_name) {
  std::vector<int64_t> list;
  if (op.GetAttr(key_name, list) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue failed!");
  }
  return list;
}
// ----------------StridedSliceGradD Op Begin-------------------
static graphStatus GetStridedSliceGradValue(const ge::Operator& op, const std::string& keyName,
                                            vector<int32_t>& multiples) {
  if (ge::GRAPH_SUCCESS != op.GetAttr(keyName, multiples)) {
    OpsGetAttrErrReport(op.GetName(), keyName);
    OP_LOGE(op.GetName().c_str(), "Get const(%s) failed from op of 'StridedSliceGrad'!", keyName.c_str());
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
    OpsAttrValueErrReport(op.GetName(), "strides's size", "more than zero and less than eight",
                          ConcatString(strides.size()));
    OP_LOGE(op.GetName().c_str(), "strides size must be more than zero and less than eight!");
    return GRAPH_FAILED;
  }
  if (dimNum >= 1 && dimNum <= 8) {
    for (size_t i = 0; i < dimNum; i++) {
      outputShape.SetDim(i, outputShapeList[i]);
    }
  } else {
    OpsInputShapeDimErrReport(op.GetName(), "dy", "8", "1", ConcatString(dimNum));
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
struct SsgMasks {
  uint64_t begin_mask = 0;
  uint64_t end_mask = 0;
  uint64_t ellipsis_mask = 0;
  uint64_t new_axis_mask = 0;
  uint64_t shrink_axis_mask = 0;
};

static graphStatus IsMasksAllZero(const ge::Operator& op, struct SsgMasks& slice_masks, bool& no_mask) {
  map<string, uint64_t&> mask_maps = {
      {"begin_mask", slice_masks.begin_mask},
      {"end_mask", slice_masks.end_mask},
      {"ellipsis_mask", slice_masks.ellipsis_mask},
      {"new_axis_mask", slice_masks.new_axis_mask},
      {"shrink_axis_mask", slice_masks.shrink_axis_mask}
  };

  for (auto& item : mask_maps) {
    int64_t mask_value = 0;
    if (ge::GRAPH_SUCCESS != op.GetAttr(item.first, mask_value)) {
      OpsGetAttrErrReport(op.GetName(), item.first);
      OP_LOGE(op.GetName().c_str(), "Get attribute '%s' failed from op of StridedSlice!", item.first.c_str());
      return GRAPH_FAILED;
    }

    item.second = static_cast<uint64_t>(mask_value);
    if (mask_value != 0) {
      no_mask = false;
    }
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(StridedSliceGradInferShape) {
  OP_LOGD("OP[SSGInferShape]", "SSGInferShape BEGIN.");
  // Set depends if input_tensors are variable.
  DataType input_dtype = op.GetInputDesc("dy").GetDataType();
  Shape shape_dy = op.GetInputDesc("dy").GetShape();

  // Get in_range, Set out_range
  std::vector<std::pair<int64_t, int64_t>> in_range;
  std::vector<std::pair<int64_t, int64_t>> out_range;
  std::pair<int64_t, int64_t> range_value;
  vector<int64_t> outputShapeList;
  Tensor output_shape_tensor;
  op.GetInputDesc("dy").GetShapeRange(in_range);

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends({"shape", "begin", "end", "strides"});
  auto dim_vector = op.GetInputDesc("shape").GetShape().GetDims();
  if (dim_vector.empty()) {
    OP_LOGE(op.GetName().c_str(), "The dims of input is empty!");
    return GRAPH_FAILED;
  }
  if (op.GetInputConstData("shape", output_shape_tensor) != GRAPH_SUCCESS) {
    OP_LOGD("OP[SSGInferShape]", "SSGInferShape BRANCH0.");
    OP_LOGI(op.GetName().c_str(), "Get constValue failed of [shape]");

    // shape_dy is UNKNOWN_RANK:
    if (shape_dy.GetDims() == UNKNOWN_RANK) {
      OP_LOGI("OP[SSGInferShape]", "shape_dy is UNKNOWN_RANK. Couldn't set in_range");

      // shape is not -2 and -1, outputshape should be (-1,...)
      if (dim_vector != UNKNOWN_RANK and dim_vector != UNKNOWN_SHAPE){
        size_t dim_num = dim_vector[0];
        OP_LOGD(op.GetName().c_str(), "dim_num is %d.", dim_num);
        for (size_t dim = 0; dim < dim_num; dim++) {
          outputShapeList.push_back(-1);
        }
        Shape out_shape(outputShapeList);
        TensorDesc tensordesc_output = op.GetOutputDesc("output");
        tensordesc_output.SetShape(out_shape);
        tensordesc_output.SetDataType(input_dtype);
        (void)op.UpdateOutputDesc("output", tensordesc_output);
        return GRAPH_SUCCESS;
      }
      vector<int64_t> shape(1, -2);
      Shape out_shape(shape);
      TensorDesc tensordesc_output = op.GetOutputDesc("output");
      tensordesc_output.SetShape(out_shape);
      tensordesc_output.SetDataType(input_dtype);
      (void)op.UpdateOutputDesc("output", tensordesc_output);
      return GRAPH_SUCCESS;
    }

    // Get shape of tensor, shape must be [n, ]
    // shape is unknown, out_shape is -2
    if (dim_vector == UNKNOWN_SHAPE || dim_vector == UNKNOWN_RANK) {
      OP_LOGD(op.GetName().c_str(), "shape is UNKNOWN_SHAPE or UNKNOWN_RANK");
      vector<int64_t> shape(1, -2);
      Shape out_shape(shape);
      TensorDesc tensordesc_output = op.GetOutputDesc("output");
      tensordesc_output.SetShape(out_shape);
      tensordesc_output.SetDataType(input_dtype);
      (void)op.UpdateOutputDesc("output", tensordesc_output);
      return GRAPH_SUCCESS;
    }

    // special branch: when shape is not const and len of begin < len of dy
    const vector<string> list_names = {"begin", "end", "strides"};
    int64_t begin_len = -1;
    for (const auto& param : list_names) {
      begin_len = std::max(op.GetInputDesc(param).GetShape().GetDim(0), begin_len);
    }

    bool no_mask = true;
    struct SsgMasks slice_masks = {};
    if (GRAPH_FAILED == IsMasksAllZero(op, slice_masks, no_mask)) {
      return GRAPH_FAILED;
    }

    if (no_mask && begin_len > 0 && begin_len < dim_vector[0]) {
      OP_LOGD("OP[SSGInferShape]", "Enter the special branch when shape is not const.");
      vector<int64_t> shape(shape_dy.GetDims());
      for (size_t dim = 0; dim < begin_len; dim++) {
        shape[dim] = -1;
      }

      for (size_t dim = 0; dim < shape.size(); dim++) {
        range_value = std::make_pair(abs(shape[dim]), shape[dim]);
        out_range.push_back(range_value);
      }
      Shape out_shape(shape);
      TensorDesc tensordesc_output = op.GetOutputDesc("output");
      tensordesc_output.SetShape(out_shape);
      tensordesc_output.SetDataType(input_dtype);
      tensordesc_output.SetShapeRange(out_range);
      (void)op.UpdateOutputDesc("output", tensordesc_output);
      return GRAPH_SUCCESS;
    }

    // Set outputShape
    // The SSG's out_range needs to get value of "shape". In compilation, infer_shape will not get
    // value of "shape" while "shape" is variable, so don't need to set range.
    size_t dim_num = dim_vector[0];
    OP_LOGD(op.GetName().c_str(), "dim_num is %d.", dim_num);
    for (size_t dim = 0; dim < dim_num; dim++) {
      outputShapeList.push_back(-1);
    }
    TensorDesc tensordesc_output = op.GetOutputDesc("output");
    ge::Shape out_shape = ge::Shape(outputShapeList);
    tensordesc_output.SetShape(out_shape);
    tensordesc_output.SetDataType(input_dtype);
    (void)op.UpdateOutputDesc("output", tensordesc_output);
    return GRAPH_SUCCESS;
  }

  // if "shape" is -2 || (-1,-1,...) || (A, B, C,...,)
  OP_LOGD("OP[SSGInferShape]", "SSGInferShape BRANCH1.");
  DataType dtype = op.GetInputDesc("shape").GetDataType();
  GetConstValue(op, output_shape_tensor, dtype, outputShapeList);

  // "shape" is -2
  if (outputShapeList == UNKNOWN_RANK) {
    vector<int64_t> shape(1, -2);
    Shape out_shape(shape);
    TensorDesc tensordesc_output = op.GetOutputDesc("output");
    tensordesc_output.SetShape(out_shape);
    tensordesc_output.SetDataType(input_dtype);
    (void)op.UpdateOutputDesc("output", tensordesc_output);
    return GRAPH_SUCCESS;
  }

  // "shape" is not -2
  if (in_range.size() > 0) {
    for (size_t dim = 0; dim < outputShapeList.size(); dim++) {
      range_value = std::make_pair(abs(outputShapeList[dim]), outputShapeList[dim]);
      out_range.push_back(range_value);
    }
  }
  TensorDesc tensordesc_output = op.GetOutputDesc("output");
  ge::Shape outputShape = ge::Shape(outputShapeList);
  tensordesc_output.SetShape(outputShape);
  tensordesc_output.SetDataType(input_dtype);
  if (in_range.size() > 0) {
    OP_LOGD("OP[SSGInferShape]", "SSGInferShape SET_RANGE.");
    tensordesc_output.SetShapeRange(out_range);
  }
  (void)op.UpdateOutputDesc("output", tensordesc_output);
  OP_LOGD("OP[SSGInferShape]", "SSGInferShape END.");
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(StridedSliceGrad, StridedSliceGradInferShape);
// ----------------StridedSliceGrad Op End------------------

// -----------------------Tile Op Begin----------------------------------
static void GetTileConstValue(const Operator& op, const Tensor& const_tensor, const DataType& dtype,
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
  OP_LOGI(op.GetName().c_str(), "Get into multiples attr branch.");
  const int64_t shape_threshold = pow(2, 31) - 1;
  uint64_t multiples_len = multiples.size();
  Shape input_shape = op.GetInputDesc("x").GetShape();
  std::vector<int64_t> input_vector = input_shape.GetDims();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  uint64_t input_len = input_shape.GetDimNum();

  std::vector<std::pair<int64_t, int64_t>> ori_shape_range;
  op.GetInputDesc("x").GetShapeRange(ori_shape_range);
  MakeUpShapeRange(input_vector, ori_shape_range);

  TensorDesc output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> output_vector;
  std::vector<std::pair<int64_t, int64_t>> output_range;

  if (multiples_len < input_len) {
    OP_LOGE(op.GetName().c_str(),
            "the tile multiples len %lu is less than the input len %lu!",
            multiples_len, input_len);
    OpsInputShapeSizeErrReport(op.GetName(), "multiples", "input", ConcatString(multiples_len));
    return GRAPH_FAILED;
  }

  // when input is -2, the output len equals to multiples_len, and the out range is (multiples[i], -1)
  if (IsUnknownRankShape(input_vector)) {
    OP_LOGI(op.GetName().c_str(), "Get into multiples attr and input unknown rank.");
    for (uint64_t i = 0; i < multiples_len; i++) {
      output_vector.push_back(-1);
      output_range.push_back(std::make_pair(multiples[i], -1));
    }
    Shape output_shape(output_vector);
    output_desc.SetShape(output_shape);
    output_desc.SetDataType(input_dtype);
    output_desc.SetShapeRange(output_range);
    (void)op.UpdateOutputDesc("y", output_desc);
    return GRAPH_SUCCESS;
  }

  // input shape not contains -2, align shape and range for input
  if (input_len != multiples_len) {
    OP_LOGI(op.GetName().c_str(), "Get into align input shape with multiples.");
    uint64_t len_diff = multiples_len - input_len;
    input_vector.insert(input_vector.begin(), len_diff, 1);
    ori_shape_range.insert(ori_shape_range.begin(), len_diff, std::make_pair(1, 1));
    input_shape = Shape(input_vector);
    input_len = input_shape.GetDimNum();
  }

  if (input_len == 0) {
    // input is empty and multiples is empty
    OP_LOGI(op.GetName().c_str(), "Get into input and multiples all empty.");
    output_desc.SetShape(input_shape);
    output_desc.SetDataType(input_dtype);
    (void)op.UpdateOutputDesc("y", output_desc);
    return GRAPH_SUCCESS;
  } else if (input_len == 1) {
    if (input_shape.GetDim(0) > 0 && (!multiples.empty())) {
      OP_LOGI(op.GetName().c_str(), "Get into align_input len 1 and input shape > 0.");
      input_shape.SetDim(0, input_shape.GetDim(0) * multiples[0]);
      output_range.push_back(std::make_pair(input_shape.GetDim(0), input_shape.GetDim(0)));
      output_desc.SetShape(input_shape);
      output_desc.SetDataType(input_dtype);
      output_desc.SetShapeRange(output_range);
      (void)op.UpdateOutputDesc("y", output_desc);
      return GRAPH_SUCCESS;
    } else if (input_shape.GetDim(0) == -1 && (!multiples.empty())) {
      OP_LOGI(op.GetName().c_str(), "Get into align_input len 1 and input shape == -1.");
      if (ori_shape_range[0].second == -1 || ori_shape_range[0].second * multiples[0] >= shape_threshold) {
        output_range.push_back(std::make_pair(ori_shape_range[0].first * multiples[0], -1));
      } else {
        output_range.push_back(std::make_pair(ori_shape_range[0].first * multiples[0],
                                              ori_shape_range[0].second * multiples[0]));
      }
      output_desc.SetShape(input_shape);
      output_desc.SetDataType(input_dtype);
      output_desc.SetShapeRange(output_range);
      (void)op.UpdateOutputDesc("y", output_desc);
      return GRAPH_SUCCESS;
    } else {
      OP_LOGE(op.GetName().c_str(), "Illegal input dim value when multiples len is 1.");
      OpsOneInputShapeErrReport(op.GetName(), "input_shape",
                              "input dim value is illegal when multiples_len is 1");
      return GRAPH_FAILED;
    }

  } else if (input_len <= 8 && input_len >= 2) {
    for (uint64_t i = 0; i < input_len; i++) {
      if (input_shape.GetDim(i) > 0) {
        input_shape.SetDim(i, input_shape.GetDim(i) * multiples[i]);
        output_range.push_back(std::make_pair(input_shape.GetDim(i), input_shape.GetDim(i)));
      } else if (input_shape.GetDim(i) == -1) {
        if (ori_shape_range[i].second == -1 || ori_shape_range[i].second * multiples[i] >= shape_threshold) {
          output_range.push_back(std::make_pair(ori_shape_range[i].first * multiples[i], -1));
        } else {
          output_range.push_back(std::make_pair(ori_shape_range[i].first * multiples[i],
                                                ori_shape_range[i].second * multiples[i]));
        }
      }
    }
    output_desc.SetShape(input_shape);
    output_desc.SetDataType(input_dtype);
    output_desc.SetShapeRange(output_range);
    (void)op.UpdateOutputDesc("y", output_desc);
    return GRAPH_SUCCESS;
  } else {
    OP_LOGE(op.GetName().c_str(), "Illegal input len while the input_len is %lu", input_len);
    return GRAPH_FAILED;
  }
}

IMPLEMT_COMMON_INFERFUNC(TileInferShape) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  Tensor multiples_tensor;
  // in order to switch to aicpu when aicore not support
  op_desc->SetOpInferDepends({"multiples"});
  if (op.GetInputConstData("multiples", multiples_tensor) != GRAPH_SUCCESS) {
    // multiples is input tensor, not constInput or attr
    OP_LOGI(op.GetName().c_str(), "Get constValue failed of [multiples]");
    std::vector<int64_t> multiples_vector = op.GetInputDesc("multiples").GetShape().GetDims();

    TensorDesc output_desc = op.GetOutputDesc("y");
    std::vector<int64_t> output_vector;
    std::vector<std::pair<int64_t, int64_t>> output_range;

    Shape input_shape = op.GetInputDesc("x").GetShape();
    std::vector<int64_t> input_vector = input_shape.GetDims();
    DataType input_dtype = op.GetInputDesc("x").GetDataType();
    size_t input_len = input_shape.GetDimNum();

    if (multiples_vector.empty() || (!multiples_vector.empty() && multiples_vector[0] == 0)) {
      OP_LOGE(op.GetName().c_str(), "Get illegal multiples vector value empty or 0!");
      return GRAPH_FAILED;
    } else if (multiples_vector.size() == 1 && multiples_vector[0] > 0) {
      // output_vector length equals to multiples, and all the range is (1, -1)
      OP_LOGI(op.GetName().c_str(), "Get into the branch multiples tensor shape bigger than zero.");
      output_vector.insert(output_vector.begin(), multiples_vector[0], -1);
      output_range.insert(output_range.begin(), multiples_vector[0], std::make_pair(1, -1));
      Shape output_shape(output_vector);
      output_desc.SetShape(output_shape);
      output_desc.SetDataType(input_dtype);
      output_desc.SetShapeRange(output_range);
      (void)op.UpdateOutputDesc("y", output_desc);
      return GRAPH_SUCCESS;
    } else if (multiples_vector.size() == 1 && (multiples_vector[0] == -1 || multiples_vector[0] == -2)) {
      // output depends on the input tensor
      if (input_vector.empty() || UNKNOWN_RANK == input_vector) {
        OP_LOGI(op.GetName().c_str(), "Get into branch multiples unknown and input empty or unknown rank.");
        output_vector = input_vector;
        Shape output_shape(output_vector);
        output_desc.SetShape(output_shape);
        output_desc.SetDataType(input_dtype);
        (void)op.UpdateOutputDesc("y", output_desc);
        return GRAPH_SUCCESS;
      } else {
        OP_LOGI(op.GetName().c_str(), "Get into branch multiples unknown and input static or unknown dims.");
        output_vector.insert(output_vector.begin(), input_len, -1);
        output_range.insert(output_range.begin(), input_len, std::make_pair(1, -1));
        Shape output_shape(output_vector);
        output_desc.SetShape(output_shape);
        output_desc.SetDataType(input_dtype);
        output_desc.SetShapeRange(output_range);
        (void)op.UpdateOutputDesc("y", output_desc);
        return GRAPH_SUCCESS;
      }
    } else {
      OP_LOGE(op.GetName().c_str(), "Illegal input of tensor multiples!");
      return GRAPH_FAILED;
    }
  }

  DataType dtype = op.GetInputDesc("multiples").GetDataType();
  std::vector<int64_t> multiples;
  GetTileConstValue(op, multiples_tensor, dtype, multiples);
  return TileInferShapeAndType(op, multiples);
}

COMMON_INFER_FUNC_REG(Tile, TileInferShape);
// -----------------------Tile Op end----------------------------------

// -----------------------TileD Op Begin----------------------------------
vector<int64_t> GetTileDConstValue(const ge::Operator& op, const std::string& key_name) {
  std::vector<int64_t> multiples;
  if (ge::GRAPH_SUCCESS != op.GetAttr(key_name, multiples)) {
    OpsGetAttrErrReport(op.GetName(), key_name);
    OP_LOGE(op.GetName().c_str(), "op tile_d get attr multiples failed!");
  }
  return multiples;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(TileDInferShape) {
  std::vector<int64_t> multiples;
  multiples = GetTileDConstValue(op, "multiples");
  if (multiples.empty()) {
    OpsGetCompileParamsErrReport(op.GetName(), "multiples");
    OP_LOGE(op.GetName().c_str(), "op tile_d get attr multiples value is empty!");
    return GRAPH_FAILED;
  }
  return TileInferShapeAndType(op, multiples);
}

COMMON_INFER_FUNC_REG(TileD, TileDInferShape);
// -----------------------TileD Op end----------------------------------

// -----------------------range Op Begin----------------------------------
static void GetRangeConstValue(const Operator& op, const Tensor& const_tensor, const DataType& dtype,
                               std::vector<float>& const_data) {
  size_t size = 0;
  if (dtype == ge::DT_INT32) {
    int32_t* const_data_ptr = (int32_t*)const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; i++) {
      const_data.push_back((int32_t)((*(const_data_ptr + i))));
    }
  } else if (dtype == ge::DT_FLOAT) {
    float* const_data_ptr = (float*)const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(float);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back((float)((*(const_data_ptr + i))));
    }
  } else if (dtype == ge::DT_INT64) {
    int64_t* const_data_ptr = (int64_t*)const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back((int64_t)((*(const_data_ptr + i))));
    }
  } else if (dtype == ge::DT_DOUBLE) {
    double* const_data_ptr = (double*)const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(double);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back((double)((*(const_data_ptr + i))));
    }
  } else {
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
    if (start_dtype == ge::DT_INT32 && limit_dtype == ge::DT_INT32 && delta_dtype == ge::DT_INT32) {
      y_output->SetDataType(ge::DT_INT32);
    } else if (start_dtype == ge::DT_INT64 && limit_dtype == ge::DT_INT64 && delta_dtype == ge::DT_INT64) {
      y_output->SetDataType(ge::DT_INT64);
    } else if (start_dtype == ge::DT_DOUBLE && limit_dtype == ge::DT_DOUBLE && delta_dtype == ge::DT_DOUBLE) {
      y_output->SetDataType(ge::DT_DOUBLE);
    } else {
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
      OP_LOGW(op.GetName().c_str(),
              "the start_multiples_size is %d, the limit_multiples_size is %d,"
              "the delta_multiples_size is %d",
              start_multiples.size(), limit_multiples.size(), delta_multiples.size());

      y_output->SetShape(GeShape({UNKNOWN_DIM}));
      y_output->SetOriginShape(GeShape({UNKNOWN_DIM}));
      y_output->SetShapeRange({std::make_pair(1, -1)});

      return GRAPH_SUCCESS;
    }

    float assist_num = std::abs(limit_multiples[0] - start_multiples[0]);
    float assist_num_one = std::abs(delta_multiples[0]);
    int64_t res = 0;
    DataType input_dtype = ge::DT_FLOAT;
    if (assist_num_one < 1e-6) {
      OP_LOGE(op.GetName().c_str(), "the value of delta should not be zero");
      return GRAPH_FAILED;
    }
    if (start_dtype == ge::DT_INT32 && limit_dtype == ge::DT_INT32 && delta_dtype == ge::DT_INT32) {
      res = static_cast<int>(ceil(float(assist_num) / assist_num_one));
    } else if (start_dtype == ge::DT_INT64 && limit_dtype == ge::DT_INT64 && delta_dtype == ge::DT_INT64) {
      res = static_cast<int>(ceil(float(assist_num) / assist_num_one));
    } else if (start_dtype == ge::DT_DOUBLE && limit_dtype == ge::DT_DOUBLE && delta_dtype == ge::DT_DOUBLE) {
      res = static_cast<int>(ceil(double(assist_num) / assist_num_one));
    } else {
      res = static_cast<int>(ceil(assist_num / assist_num_one));
    }
    dimsIn.emplace_back(res);
    if (start_dtype == ge::DT_INT32 && limit_dtype == ge::DT_INT32 && delta_dtype == ge::DT_INT32) {
      input_dtype = ge::DT_INT32;
    } else if (start_dtype == ge::DT_INT64 && limit_dtype == ge::DT_INT64 && delta_dtype == ge::DT_INT64) {
      input_dtype = ge::DT_INT64;
    } else if (start_dtype == ge::DT_DOUBLE && limit_dtype == ge::DT_DOUBLE && delta_dtype == ge::DT_DOUBLE) {
      input_dtype = ge::DT_DOUBLE;
    } else {
      input_dtype = ge::DT_FLOAT;
    }
    y_output->SetShape(GeShape(dimsIn));
    y_output->SetOriginShape(GeShape(dimsIn));
    y_output->SetDataType(input_dtype);
    OP_LOGD(op.GetName().c_str(), "output shape:%s.", to_string(dimsIn).c_str());
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
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue start failed.");
    return GRAPH_FAILED;
  }
  float limit;
  if (op.GetAttr("limit", limit) == ge::GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "limit");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue limit failed.");
    return GRAPH_FAILED;
  }
  float delta;
  if (op.GetAttr("delta", delta) == ge::GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "delta");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue delta failed.");
    return GRAPH_FAILED;
  }
  if (limit == start) {
    string excepted_value = ConcatString("not equal to limit[", limit, "]");
    OpsAttrValueErrReport(op.GetName(), "start", excepted_value, ConcatString(start));
    OP_LOGE(op.GetName().c_str(), "start is not equal to limit");
    return GRAPH_FAILED;
  }
  if (delta == 0) {
    OpsAttrValueErrReport(op.GetName(), "delta", "not equal to zero", ConcatString(delta));
    OP_LOGE(op.GetName().c_str(), "the input of delta is not equal to zero");
    return GRAPH_FAILED;
  }
  if (start > limit && delta > 0) {
    string excepted_value = ConcatString("more than start[", start, "]");
    OpsAttrValueErrReport(op.GetName(), "limit", excepted_value, ConcatString(limit));
    OP_LOGE(op.GetName().c_str(),
            "requires limit is more than start "
            "when delta is more than zero");
    return GRAPH_FAILED;
  }
  if (start < limit && delta < 0) {
    string excepted_value = ConcatString("more than limit[", limit, "]");
    OpsAttrValueErrReport(op.GetName(), "start", excepted_value, ConcatString(start));
    OP_LOGE(op.GetName().c_str(),
            "requires start is more than limit "
            "when delta is less than zero");
    return GRAPH_FAILED;
  }
  (void)op.UpdateOutputDesc("y", op.GetInputDesc("x"));
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(RangeD, RangeDInferShape);
// -----------------------RangeD Op End----------------------------------

// ----------------GatherNd Op-------------------
bool CheckGatherNdInputIndicesSize(const Operator& op, const string& input_name) {
  auto indices_shape = OpDescUtils::GetOpDescFromOperator(op)->MutableInputDesc("indices")->GetShape();
  auto indices_shape_size = indices_shape.GetDimNum();
  int indices_last_element = indices_shape.GetDim(indices_shape_size - 1);
  int64_t indices_part{1};
  for (int i = 0; i < indices_last_element - 1; ++i) {
    indices_part *= static_cast<int64_t>(indices_shape.GetDim(i));
  }
  if (indices_part > std::numeric_limits<int>::max()) {
    OpsInputShapeSizeErrReport(op.GetName(), "indices", ConcatString(std::numeric_limits<int>::max()),
                               ConcatString(indices_part));
    OP_LOGE(op.GetName().c_str(), "Indices has too many elements for int indexing");
    return false;
  }
  return true;
}

bool CheckGatherNdParamsSize(const Operator& op, int last_dim, int shape_size) {
  if (last_dim > shape_size) {
    OP_LOGE(op.GetName().c_str(), "The last dim(%d) of indices must be <= params.rank(%d).", last_dim, shape_size);
    return false;
  }
  return true;
}

IMPLEMT_VERIFIER(GatherNd, GatherNdVerify) {
  if (!CheckGatherNdInputIndicesSize(op, "indices")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(GatherNdInferShape) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc("y");
  std::vector<std::pair<int64_t, int64_t>> shape_range_x;
  op_desc->MutableInputDesc("x")->GetShapeRange(shape_range_x);
  std::vector<std::pair<int64_t, int64_t>> shape_range_indices;
  op_desc->MutableInputDesc("indices")->GetShapeRange(shape_range_indices);
  std::vector<std::pair<int64_t, int64_t>> out_range;
  auto input_params = op_desc -> MutableInputDesc("x");
  auto input_indices = op_desc -> MutableInputDesc("indices");
  auto params_shape = input_params -> GetShape();
  auto indices_shape = input_indices -> GetShape();
  auto params_shape_size = params_shape.GetDimNum();
  int indices_shape_size = indices_shape.GetDimNum();
  vector<int64_t> dim_vec;
  vector<int64_t> params_shape_vec = params_shape.GetDims();
  vector<int64_t> indices_shape_vec = indices_shape.GetDims();
  MakeUpShapeRange(params_shape_vec, shape_range_x);
  MakeUpShapeRange(indices_shape_vec, shape_range_indices);
  int indices_last_element{-2};
  if (!IsUnknownRankShape(indices_shape_vec)) {
    indices_last_element = indices_shape.GetDim(indices_shape_size - 1);
  }
  DataType params_type = input_params->GetDataType();
  if (indices_last_element == -1 || indices_last_element == -2 || IsUnknownRankShape(params_shape_vec)) {
    dim_vec.push_back(-2);
  } else if (!CheckGatherNdParamsSize(op, indices_last_element, (int)params_shape_size)) {
    return GRAPH_FAILED;
  } else {
    for (int i = 0; i < indices_shape_size - 1; ++i) {
      dim_vec.push_back(indices_shape.GetDim(i));
      if ((size_t)i < shape_range_indices.size()) {
        out_range.push_back(shape_range_indices[i]);
      }
    }
    for (size_t i = indices_last_element; i < params_shape_size; ++i) {
      dim_vec.push_back(params_shape.GetDim(i));
      if (i < shape_range_x.size()) {
        out_range.push_back(shape_range_x[i]);
      }
    }
  }
  ge::GeShape output_shape = ge::GeShape(dim_vec);
  DataType output_dtype = params_type;
  output_tensor_desc->SetShape(output_shape);
  output_tensor_desc->SetDataType(output_dtype);
  TensorUtils::SetRealDimCnt(*output_tensor_desc, dim_vec.size());
  if (!IsUnknownRankShape(dim_vec)) {
    output_tensor_desc->SetShapeRange(out_range);
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(GatherNd, GatherNdInferShape);
VERIFY_FUNC_REG(GatherNd, GatherNdVerify);
// ----------------GatherNd END----------------

// ----------------GatherV2-------------------
static graphStatus GatherV2InferOptimize(ge::Operator& op, int64_t& axis, GeTensorDescPtr& x_desc,
                                         GeTensorDescPtr& indices_desc, GeTensorDescPtr& y_desc,
                                         std::vector<int64_t>& x_shape, std::vector<int64_t>& indices_shape,
                                         std::vector<int64_t>& y_shape,
                                         std::vector<std::pair<int64_t, int64_t>>& shape_range_x,
                                         std::vector<std::pair<int64_t, int64_t>>& shape_range_indices,
                                         std::vector<std::pair<int64_t, int64_t>>& out_range) {
  // real dim cnt has no existing meaning .Original shape has replace its meaning now
  int64_t x_real_dim_cnt = static_cast<int64_t>(x_desc->GetOriginShape().GetDims().size());

  if (IsUnknownRankShape(indices_shape) || IsUnknownRankShape(x_shape)) {
    y_shape.push_back(-2);

    y_desc->SetShape(ge::GeShape(y_shape));
    y_desc->SetDataType(x_desc->GetDataType());

    return GRAPH_SUCCESS;
  }

  if (x_real_dim_cnt < 1) {
    OP_LOGE(op.GetName().c_str(), "x_desc RealDimCnt[%d] not support.", x_real_dim_cnt);
    return GRAPH_FAILED;
  }

  if (axis < 0) {
    if (x_real_dim_cnt < -axis) {
      OP_LOGE(op.GetName().c_str(), "x_desc RealDimCnt[%d] < -axis[%d]", x_real_dim_cnt, -axis);
      return GRAPH_FAILED;
    }
  } else if (x_real_dim_cnt < axis + 1) {
    OP_LOGE(op.GetName().c_str(), "x_desc RealDimCnt[%d] < axis + 1[%d]", x_real_dim_cnt, axis + 1);
    return GRAPH_FAILED;
  }

  int64_t end = axis;
  if (end < 0) {
    end = x_real_dim_cnt + end;
    if (end < 0) {
      OP_LOGE(op.GetName().c_str(), "x_desc RealDimCnt[%d] < axis + 1[%d]", x_real_dim_cnt, axis + 1);
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
        OP_LOGE(op.GetName().c_str(), "start[%d], rank[%d], error.", start, rank);
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

IMPLEMT_COMMON_INFERFUNC(GatherV2InferShape) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  vector<string> input_infer_depends = {"axis"};
  op_desc->SetOpInferDepends(input_infer_depends);

  GeTensorDescPtr x_desc = op_desc->MutableInputDesc("x");
  GeTensorDescPtr indices_desc = op_desc->MutableInputDesc("indices");
  GeTensorDescPtr y_desc = op_desc->MutableOutputDesc("y");

  std::vector<int64_t> x_shape = x_desc->MutableShape().GetDims();
  std::vector<int64_t> indices_shape = indices_desc->MutableShape().GetDims();

  std::vector<int64_t> y_shape;

  std::vector<std::pair<int64_t, int64_t>> shape_range_x;
  op_desc->MutableInputDesc("x")->GetShapeRange(shape_range_x);
  std::vector<std::pair<int64_t, int64_t>> shape_range_indices;
  op_desc->MutableInputDesc("indices")->GetShapeRange(shape_range_indices);
  std::vector<std::pair<int64_t, int64_t>> out_range;

  MakeUpShapeRange(x_shape, shape_range_x);
  MakeUpShapeRange(indices_shape, shape_range_indices);

  Tensor axis_tensor;
  int64_t axis = -1;
  DataType axis_dtype = op_desc->MutableInputDesc("axis")->GetDataType();
  graphStatus result = op.GetInputConstData("axis", axis_tensor);
  if (result == GRAPH_SUCCESS) {
    if (axis_dtype == ge::DT_INT64) {
      axis = (int64_t)(*((int64_t*)axis_tensor.GetData()));
    } else {
      axis = (int32_t)(*((int32_t*)axis_tensor.GetData()));
    }
  }

  // unknown rank
  if (IsUnknownRankShape(indices_shape) || IsUnknownRankShape(x_shape)) {
    y_shape.push_back(-2);
    y_desc->SetShape(ge::GeShape(y_shape));
    y_desc->SetDataType(x_desc->GetDataType());
  } else if (result != GRAPH_SUCCESS) {
    // unknown shape
    OP_LOGI(op.GetName().c_str(), "GetInputConstData(axis) [%d]", result);
    int64_t rank_x = static_cast<int64_t>(x_desc->GetOriginShape().GetDims().size());
    int64_t rank_indices = static_cast<int64_t>(indices_desc->GetOriginShape().GetDims().size());

    // infer shape range
    std::vector<std::pair<int64_t, int64_t>> range_tmp = shape_range_x;
    range_tmp.insert(range_tmp.end(), shape_range_indices.begin(), shape_range_indices.end());
    int64_t min_first, max_second;
    for (size_t i = 0; i < range_tmp.size(); i++) {
      if (i == 0) {
        min_first = range_tmp[i].first;
        max_second = range_tmp[i].second;
      }
      min_first = min_first < range_tmp[i].first ? min_first : range_tmp[i].first;
      max_second = max_second > range_tmp[i].second ? max_second : range_tmp[i].second;
    }

    std::pair<int64_t, int64_t> rank_unkown(1, -1);
    int count_rank_x = std::count(shape_range_x.begin(), shape_range_x.end(), rank_unkown);
    int count_rank_indices = std::count(shape_range_indices.begin(), shape_range_indices.end(), rank_unkown);

    for (int i = 0; i < rank_x + rank_indices - 1; i++) {
      y_shape.push_back(-1);
      if (count_rank_x > 0 || count_rank_indices > 0) {
        out_range.push_back(std::pair<int64_t, int64_t>(1, -1));
      } else {
        out_range.push_back(std::pair<int64_t, int64_t>(min_first, max_second));
      }
    }

    y_desc->SetDataType(x_desc->GetDataType());
    y_desc->SetShapeRange(out_range);
    y_desc->SetShape(ge::GeShape(y_shape));
  } else {
    if (GatherV2InferOptimize(op, axis, x_desc, indices_desc, y_desc, x_shape, indices_shape, y_shape, shape_range_x,
                              shape_range_indices, out_range) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(GatherV2, GatherV2InferShape);
// ----------------GatherV2 END-------------------

// ----------------GatherV2D-----------------------
static graphStatus GatherV2InferShapeAndType(ge::Operator& op, int32_t& axis) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr indices_desc = op_desc->MutableInputDesc("indices");
  auto indices_shape = indices_desc->GetShape().GetDims();

  std::vector<std::pair<int64_t, int64_t>> shape_range_x;
  op_desc->MutableInputDesc("x")->GetShapeRange(shape_range_x);
  std::vector<std::pair<int64_t, int64_t>> shape_range_indices;
  op_desc->MutableInputDesc("indices")->GetShapeRange(shape_range_indices);
  std::vector<std::pair<int64_t, int64_t>> out_range;

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
    OP_LOGE(op.GetName().c_str(), "x_desc RealDimCnt[%d] not support.", x_real_dim_cnt);
    return GRAPH_FAILED;
  }
  auto x_shape = x_desc->GetShape().GetDims();
  if (axis < 0) {
    if (x_real_dim_cnt < -axis) {
      OP_LOGE(op.GetName().c_str(), "x_desc RealDimCnt[%d] < -axis[%d]", x_real_dim_cnt, -axis);
      return GRAPH_FAILED;
    }
  } else if (x_real_dim_cnt < axis + 1) {
    OP_LOGE(op.GetName().c_str(), "x_desc RealDimCnt[%d] < axis + 1[%d]", x_real_dim_cnt, axis + 1);
    return GRAPH_FAILED;
  }

  int64_t end = axis;
  if (end < 0) {
    end = x_real_dim_cnt + end;
    if (end < 0) {
      OP_LOGE(op.GetName().c_str(), "x_desc RealDimCnt[%d] < axis + 1[%d]", x_real_dim_cnt, axis + 1);
      return GRAPH_FAILED;
    }
  }

  for (int i = 0; i < end; i++) {
    y_shape.push_back(x_shape[i]);
    if ((size_t)i < shape_range_x.size()) {
      out_range.push_back(shape_range_x[i]);
    }
  }
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
        OP_LOGE(op.GetName().c_str(), "start[%d], rank[%d], error.", start, rank);
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

IMPLEMT_COMMON_INFERFUNC(GatherV2DInferShape) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
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
      OpsInputShapeDimErrReport(op.GetName(), "axis", ConcatString(dimnum), ConcatString(-dimnum), ConcatString(axis));
      OP_LOGE(op.GetName().c_str(), "attr axis is not in range");
      return GRAPH_FAILED;
    }
  }

  if (GatherV2InferShapeAndType(op, axis) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(GatherV2D, GatherV2DInferShape);
// ----------------GatherV2D END-------------------

// ----------------Gather-------------------
IMPLEMT_COMMON_INFERFUNC(GatherInferShape) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr x_desc = op_desc->MutableInputDesc("x");
  std::vector<int64_t> x_shape = x_desc->MutableShape().GetDims();
  DataType x_dtype = op.GetInputDesc("x").GetDataType();
  std::vector<std::pair<int64_t, int64_t>> x_shape_range;
  op_desc->MutableInputDesc("x")->GetShapeRange(x_shape_range);
  MakeUpShapeRange(x_shape, x_shape_range);

  GeTensorDescPtr indices_desc = op_desc->MutableInputDesc("indices");
  std::vector<int64_t> indices_shape = indices_desc->MutableShape().GetDims();
  std::vector<std::pair<int64_t, int64_t>> indices_shape_range;
  op_desc->MutableInputDesc("indices")->GetShapeRange(indices_shape_range);
  MakeUpShapeRange(indices_shape, indices_shape_range);

  GeTensorDescPtr y_desc = op_desc->MutableOutputDesc("y");
  std::vector<int64_t> y_shape;
  std::vector<std::pair<int64_t, int64_t>> y_shape_range;

  int64_t axis = 0;

  // unknown rank
  if (IsUnknownRankShape(indices_shape) || IsUnknownRankShape(x_shape)) {
    y_shape.push_back(-2);
    y_desc->SetShape(ge::GeShape(y_shape));
    y_desc->SetDataType(x_dtype);
  } else {
    if (GatherV2InferOptimize(op, axis, x_desc, indices_desc, y_desc, x_shape, indices_shape, y_shape, x_shape_range,
                              indices_shape_range, y_shape_range) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }
  }
  OP_LOGI(op.GetName().c_str(), "output shape range is:%s", to_string(y_shape_range).c_str());
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Gather, GatherInferShape);
// ----------------Gather END-------------------

// --------------------------GatherElements-------------------------
IMPLEMT_COMMON_INFERFUNC(GatherElementsInferShape) {
    TensorDesc tensordesc_output = op.GetOutputDesc("y");

    tensordesc_output.SetShape(op.GetInputDesc("index").GetShape());
    tensordesc_output.SetDataType(op.GetInputDesc("x").GetDataType());
    (void)op.UpdateOutputDesc("y", tensordesc_output);
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(GatherElements, GatherElementsInferShape);
// --------------------------GatherElements END---------------------

// --------------------------LogSpaceD---------------------
bool InferShapeAndTypeLogSpaceD(Operator& op, const string& input_name, const string& output_name,
                                    const string& attr_name) {
    TensorDesc v_output_desc = op.GetOutputDesc(output_name);
    DataType output_dtype = op.GetInputDesc(input_name).GetDataType();
    Format output_format = op.GetInputDesc(input_name).GetFormat();
    ge::Shape shape = op.GetInputDesc(input_name).GetShape();
    ge::Shape outshape = ge::Shape(shape);
    v_output_desc.SetShape(outshape);
    v_output_desc.SetFormat(output_format);
    int64_t dtype = 1;
    if (op.GetAttr(attr_name, dtype) != GRAPH_SUCCESS) {
        v_output_desc.SetDataType(DT_FLOAT);
    } else {
        if (dtype == 0) {
            v_output_desc.SetDataType(DT_FLOAT16);
        }
        if (dtype == 1) {
            v_output_desc.SetDataType(DT_FLOAT);
        }
    }
    op.UpdateOutputDesc(output_name, v_output_desc);
    return true;
}

IMPLEMT_VERIFIER(LogSpaceD, LogSpaceDVerify)
{
    if (op.GetInputDesc("assist").GetShape().GetDims().size() != 1) {
        OP_LOGE(op.GetName().c_str(), "Input size must be 1.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(LogSpaceDInferShape)
{
    if(InferShapeAndTypeLogSpaceD(op, "assist", "y", "dtype")) {
        return GRAPH_SUCCESS;
    }
    OP_LOGE(op.GetName().c_str(), "should have the assist, y and dtype.");
    return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(LogSpaceD, LogSpaceDInferShape);

VERIFY_FUNC_REG(LogSpaceD, LogSpaceDVerify);
// --------------------------LogSpaceD END---------------------

// ----------------UnsortedSegmentSum-------------------
static void GetUnsortedSegmentSumConstValue(const Tensor& const_tensor, const DataType& dtype, int64_t& const_data) {
  if (dtype == ge::DT_INT32) {
    int32_t* const_data_ptr = (int32_t*)const_tensor.GetData();
    const_data = (int32_t)((*(const_data_ptr + 0)));
  } else {
    int64_t* const_data_ptr = (int64_t*)const_tensor.GetData();
    const_data = (int64_t)(*(const_data_ptr + 0));
  }
}

IMPLEMT_COMMON_INFERFUNC(UnsortedSegmentSumInferShape) {
  vector<string> input_infer_depends = {"num_segments"};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends(input_infer_depends);

  Tensor input_num_segments_tensor;
  int64_t input_num_segments;
  DataType input_num_segments_dtype = op_desc->MutableInputDesc("num_segments")->GetDataType();

  std::vector<std::pair<int64_t, int64_t>> shape_range_x;
  op_desc->MutableInputDesc("x")->GetShapeRange(shape_range_x);
  std::vector<std::pair<int64_t, int64_t>> shape_range_seg_id;
  op_desc->MutableInputDesc("segment_ids")->GetShapeRange(shape_range_seg_id);

  std::vector<std::pair<int64_t, int64_t>> out_range;

  if (GRAPH_SUCCESS != op.GetInputConstData("num_segments", input_num_segments_tensor)) {
    input_num_segments = -1;
    out_range.push_back(std::pair<int64_t, int64_t>(1, -1));
  } else {
    GetUnsortedSegmentSumConstValue(input_num_segments_tensor, input_num_segments_dtype, input_num_segments);
    out_range.push_back(std::pair<int64_t, int64_t>(input_num_segments, input_num_segments));
  }

  ge::GeShape shape = op_desc->MutableInputDesc("x")->GetShape();
  ge::GeShape shape_id = op_desc->MutableInputDesc("segment_ids")->GetShape();
  auto shape_vec = shape.GetDims();
  auto shape_id_vec = shape_id.GetDims();

  MakeUpShapeRange(shape_vec, shape_range_x);
  MakeUpShapeRange(shape_id_vec, shape_range_seg_id);

  int64_t dim_idsize_input = shape_id.GetDimNum();
  int64_t dim_size_input = shape.GetDimNum();
  DataType input_dtype = op_desc->MutableInputDesc("x")->GetDataType();
  vector<int64_t> shape_vector;
  if (IsUnknownRankShape(shape_vec) || IsUnknownRankShape(shape_id_vec)) {
    shape_vector.push_back(-2);
    for (size_t i = shape_range_seg_id.size(); i < shape_range_x.size(); i++) {
      out_range.push_back(shape_range_x[i]);
    }
  } else if (dim_idsize_input > 1) {
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
    for (size_t i = 1; i < shape_vector.size(); i++) {
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

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(UnsortedSegmentSum, UnsortedSegmentSumInferShape);
// ----------------UnsortedSegmentSum END----------------

// ----------------UnsortedSegment-------------------
COMMON_INFER_FUNC_REG(UnsortedSegmentMin, UnsortedSegmentSumInferShape);
COMMON_INFER_FUNC_REG(UnsortedSegmentMax, UnsortedSegmentSumInferShape);
COMMON_INFER_FUNC_REG(UnsortedSegmentProd, UnsortedSegmentSumInferShape);
// ----------------UnsortedSegment END----------------

// ----------------UnsortedSegmentSumD-------------------
IMPLEMT_COMMON_INFERFUNC(UnsortedSegmentSumDInferShape) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  int64_t input_num_segments;
  if (ge::GRAPH_SUCCESS != op.GetAttr("num_segments", input_num_segments)) {
    OpsGetAttrErrReport(op.GetName(), "num_segments");
    OP_LOGE(op.GetName().c_str(),
            "The num_segments"
            "op GetOpAttr ConstValue failed!");
  }
  if (input_num_segments <= 0) {
    OpsAttrValueErrReport(op.GetName(), "num_segments", "reater than 0", ConcatString(input_num_segments));
    OP_LOGE(op.GetName().c_str(), "num_segments need greater than 0");
    return GRAPH_FAILED;
  }
  ge::GeShape shape = op_desc->MutableInputDesc("x")->GetShape();
  ge::GeShape shape_id = op_desc->MutableInputDesc("segment_ids")->GetShape();
  int64_t dim_idsize_input = shape_id.GetDimNum();
  int64_t dim_size_input = shape.GetDimNum();
  DataType input_dtype = op_desc->MutableInputDesc("x")->GetDataType();
  vector<int64_t> shape_vector;
  if (IsUnknownRank(op, "x") || IsUnknownRank(op, "segment_ids")) {
    shape_vector.push_back(-2);
  } else if (dim_idsize_input > 1) {
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
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(UnsortedSegmentSumD, UnsortedSegmentSumDInferShape);
// ----------------UnsortedSegmentSumD END------------------

// ----------------StridedSliceD Op Begin-------------------
struct SliceParameters {
  std::vector<int64_t> input;
  std::vector<int64_t> begin_list;
  std::vector<int64_t> end_list;
  std::vector<int64_t> stride_list;
};

// define the masks for 'stridedSlice'
struct SliceMasks {
  uint64_t begin_mask = 0;
  uint64_t end_mask = 0;
  uint64_t ellipsis_mask = 0;
  uint64_t new_axis_mask = 0;
  uint64_t shrink_axis_mask = 0;
};

// get value from const node
static graphStatus GetStridedSliceListAttrValue(const ge::Operator &op, const std::string &keyName,
                                                vector<int64_t> &multiples) {
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
static graphStatus GetStridedSliceListAttrValues(const ge::Operator &op, struct SliceParameters &slice_params) {
  if (GRAPH_FAILED == GetStridedSliceListAttrValue(op, "begin", slice_params.begin_list)) {
    return GRAPH_FAILED;
  }

  if (GRAPH_FAILED == GetStridedSliceListAttrValue(op, "end", slice_params.end_list)) {
    return GRAPH_FAILED;
  }

  if (GRAPH_FAILED == GetStridedSliceListAttrValue(op, "strides", slice_params.stride_list)) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

// Get relevant masks from const node
static graphStatus GetStridedSliceMasks(const ge::Operator& op, struct SliceMasks& slice_masks) {
  map<string, uint64_t&> mask_maps = {
      {"begin_mask", slice_masks.begin_mask},
      {"end_mask", slice_masks.end_mask},
      {"ellipsis_mask", slice_masks.ellipsis_mask},
      {"new_axis_mask", slice_masks.new_axis_mask},
      {"shrink_axis_mask", slice_masks.shrink_axis_mask}
  };

  for (auto& item : mask_maps) {
    int64_t mask_value = 0;
    if (ge::GRAPH_SUCCESS != op.GetAttr(item.first, mask_value)) {
      OpsGetAttrErrReport(op.GetName(), item.first);
      OP_LOGE(op.GetName().c_str(), "Get attribute '%s' failed from op of StridedSlice!", item.first.c_str());
      return GRAPH_FAILED;
    }

    item.second = static_cast<uint64_t>(mask_value);
  }

  return GRAPH_SUCCESS;
}

static graphStatus GetStridedSliceListConstValues(const ge::Operator& op, struct SliceParameters& slice_params) {
  Tensor const_tensor;

  std::map<std::string, vector<int64_t>&> const_values = {
      {"begin", slice_params.begin_list},
      {"end", slice_params.end_list},
      {"strides", slice_params.stride_list},
  };

  bool all_const = true;
  for (auto& item: const_values) {
    if (op.GetInputConstData(item.first, const_tensor) != GRAPH_SUCCESS) {
      OP_LOGI(op.GetName().c_str(), "[%s] is not constant.", item.first.c_str());
      all_const = false;
      continue;
    }

    item.second.clear();
    auto dtype = op.GetInputDesc(item.first).GetDataType();
    GetConstValue(op, const_tensor, dtype, item.second);
  }

  if (!all_const) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(StridedSliceDInferShape) {
  const vector<string> depends;
  PREPARE_DYNAMIC_SHAPE(depends);

  // Get input shape
  ge::Shape shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  size_t dim_num = shape.GetDimNum();

  // Get 'begin_list','end_list','stride_list' from const node
  struct SliceParameters slice_params;
  if (GRAPH_FAILED == GetStridedSliceListAttrValues(op, slice_params)) {
    return GRAPH_FAILED;
  }

  // Get relevant masks from const node
  struct SliceMasks slice_masks = {};
  if (GRAPH_FAILED == GetStridedSliceMasks(op, slice_masks)) {
    return GRAPH_FAILED;
  }

  StridedSliceParams input_params = {
      shape.GetDims(),
      slice_params.begin_list,
      slice_params.end_list,
      slice_params.stride_list,
      vector<pair<int64_t, int64_t>>(),
      slice_masks.begin_mask,
      slice_masks.end_mask,
      slice_masks.ellipsis_mask,
      slice_masks.new_axis_mask,
      slice_masks.shrink_axis_mask,
      true,
      true,
      true
  };

  vector<int64_t> output_shape;
  vector<pair<int64_t, int64_t>> output_ranges;
  if (!StridedSliceCommonInferShape(op.GetName(), input_params, output_shape, output_ranges)) {
    return GRAPH_FAILED;
  }

  ge::Shape outputShape = ge::Shape(output_shape);
  TensorDesc tensor_desc_output = op.GetOutputDesc("y");
  tensor_desc_output.SetShape(outputShape);
  tensor_desc_output.SetDataType(input_dtype);
  OP_LOGD(op.GetName().c_str(), "output_shape:%s", to_string(tensor_desc_output.GetShape()).c_str());
  (void)op.UpdateOutputDesc("y", tensor_desc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(StridedSliceD, StridedSliceDInferShape);
// ----------------StridedSliceD Op End-------------------

// ----------------stridedSlice Op Begin-------------------
IMPLEMT_COMMON_INFERFUNC(StridedSliceInferShape) {
  const vector<string> depend_names = {"begin", "end", "strides"};
  PREPARE_DYNAMIC_SHAPE(depend_names);

  // Get input shape
  auto input_desc = op.GetInputDesc("x");
  const ge::Shape shape = input_desc.GetShape();
  DataType input_dtype = input_desc.GetDataType();
  int64_t begin_len = -1;
  for (const auto& param : depend_names) {
    begin_len = std::max(op.GetInputDesc(param).GetShape().GetDim(0), begin_len);
  }

  // Get 'begin_list','end_list','stride_list' from const node
  struct SliceParameters slice_params = {};
  bool begin_valid = true;
  bool end_valid = true;
  bool stride_valid = true;
  if (GRAPH_FAILED == GetStridedSliceListConstValues(op, slice_params)) {
    OP_LOGI(op.GetName().c_str(),
            "[begin,end,stride] are not all constant, set to tmp values for inference dynamic shape");
    begin_valid = !slice_params.begin_list.empty();
    end_valid = !slice_params.end_list.empty();
    stride_valid = !slice_params.stride_list.empty();
  }

  OP_LOGD(op.GetName().c_str(), "begin_len:%lld", begin_len);
  if (shape.GetDims() == UNKNOWN_RANK || begin_len == -1 || !stride_valid) {
    TensorDesc output_desc = op.GetOutputDesc("y");
    output_desc.SetDataType(input_dtype);
    ge::Shape outputShape = ge::Shape(UNKNOWN_RANK);
    output_desc.SetShape(outputShape);
    OP_LOGD(op.GetName().c_str(), "output_shape:%s", to_string(output_desc.GetShape()).c_str());
    (void) op.UpdateOutputDesc("y", output_desc);
    return GRAPH_SUCCESS;
  }

  // If  begin is invalid, set begin with begin_len count of 0, for inference output ranges.
  // For example, begin_len is 2 set begin's value to [0, 0]
  if (!begin_valid) {
    slice_params.begin_list.assign(begin_len, 0);
  }

  // If end is invalid, set end with begin_len count with same index of the input shape dims, for inference output
  // ranges. If begin_len greater than the length of input shape, set the end[i] to input_shape.back()
  // which i >= input_shape.size().
  // For example, begin_len is 2 and input shape is (5, 6, 7, 8), set end's value to [5, 6].
  //              begin_len is 5 and input shape is (5, 6, 7, 8), set end's value to [5, 6, 7, 8, 8].
  if (!end_valid) {
    auto shape_dims = shape.GetDims();
    if (begin_len < static_cast<int64_t>(shape_dims.size())) {
      slice_params.end_list.assign(shape_dims.begin(), shape_dims.begin()+begin_len);
    } else {
      slice_params.end_list = shape_dims;
      for (size_t i=shape_dims.size(); i < static_cast<size_t>(begin_len); i++) {
        slice_params.end_list.push_back(shape_dims.back());
      }
    }
  }

  // If stride is invalid, set stride with begin_len count of 1, for inference output ranges.
  // For example, begin_len is 2 set stride's value to [1, 1]
  if (!stride_valid) {
    slice_params.stride_list.assign(begin_len, 1);
  }

  vector<pair<int64_t,int64_t>> input_ranges;
  input_desc.GetShapeRange(input_ranges);
  if (input_ranges.empty()) {
    MakeUpShapeRange(shape.GetDims(), input_ranges);
  }

  size_t dim_num = shape.GetDimNum();

  if (dim_num == 0) {
    OpsInputShapeDimErrReport(op.GetName(), "dims", ConcatString(DIM_SIZE8), ConcatString(DIM_SIZE1),
                              ConcatString(dim_num));
    OP_LOGE("Get input x's dimnum is 0");
    return GRAPH_FAILED;
  }

  OP_LOGD(op.GetName().c_str(), "begin_list:%s", to_string(slice_params.begin_list).c_str());
  OP_LOGD(op.GetName().c_str(), "end_list:%s", to_string(slice_params.end_list).c_str());
  OP_LOGD(op.GetName().c_str(), "stride_list:%s", to_string(slice_params.stride_list).c_str());
  if (slice_params.end_list.size() != slice_params.begin_list.size()) {
    OP_LOGE(op.GetName().c_str(), "end shape,begin shape length mismatch!");
    return GRAPH_FAILED;
  }

  // Get relevant masks from const node
  struct SliceMasks slice_masks = {};
  if (GRAPH_FAILED == GetStridedSliceMasks(op, slice_masks)) {
    return GRAPH_FAILED;
  }

  StridedSliceParams input_params = {
      shape.GetDims(),
      slice_params.begin_list,
      slice_params.end_list,
      slice_params.stride_list,
      input_ranges,
      slice_masks.begin_mask,
      slice_masks.end_mask,
      slice_masks.ellipsis_mask,
      slice_masks.new_axis_mask,
      slice_masks.shrink_axis_mask,
      begin_valid,
      end_valid,
      stride_valid,
  };

  std::vector<int64_t> output_real_dims;
  std::vector<int64_t> output_shape;
  vector<pair<int64_t, int64_t>> output_ranges;
  if (!StridedSliceCommonInferShape(op.GetName(), input_params, output_shape, output_ranges)) {
    return GRAPH_FAILED;
  }

  for (auto dim : output_shape) {
    if (dim != 1) {
      output_real_dims.push_back(dim);
    }
  }

  if (output_real_dims.size() == 0) {
    output_real_dims.push_back(1);
  }

  TensorDesc tensor_desc_output = op.GetOutputDesc("y");
  tensor_desc_output.SetDataType(input_dtype);
  tensor_desc_output.SetRealDimCnt(output_real_dims.size());

  if (IsUnKnownShape(output_shape) && !output_ranges.empty()) {
    tensor_desc_output.SetShapeRange(output_ranges);
  }

  ge::Shape outputShape = ge::Shape(output_shape);
  tensor_desc_output.SetShape(outputShape);
  OP_LOGD(op.GetName().c_str(), "output_ranges:%s", to_string(output_ranges).c_str());
  OP_LOGD(op.GetName().c_str(), "output_shape:%s", to_string(tensor_desc_output.GetShape()).c_str());
  (void) op.UpdateOutputDesc("y", tensor_desc_output);

  auto p_context = op.GetInferenceContext();
  if (p_context != nullptr) {
    const auto& shapes_and_types = p_context->GetInputHandleShapesAndTypes();
    if (!shapes_and_types.empty()) {
      p_context->SetOutputHandleShapesAndTypes(shapes_and_types);
    }
  }

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
  if (TwoInOneOutDynamicInferNoBroadcast(op, "x1", "x2", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(Select, SelectInferShape);
VERIFY_FUNC_REG(Select, SelectVerify);
// ---------------Select END-----------------------

// ----------------SelectV2----------------------
bool BroadCastTwoinOneout(const Operator& op, std::vector<int64_t>& shape_x, std::vector<int64_t>& shape_y,
                          std::vector<std::pair<int64_t, int64_t>>& range_x,
                          std::vector<std::pair<int64_t, int64_t>>& range_y,
                          std::vector<int64_t>& dim_out,
                          std::vector<std::pair<int64_t, int64_t>>& range_out) {
  std::vector<int64_t> dim_x = shape_x;
  std::vector<int64_t> dim_y = shape_y;
  std::vector<std::pair<int64_t, int64_t>> range_x_new = range_x;
  std::vector<std::pair<int64_t, int64_t>> range_y_new = range_y;
  // exchange them
  if (dim_x.size() < dim_y.size()) {
    std::vector<int64_t> dim_tmp = dim_x;
    std::vector<std::pair<int64_t, int64_t>> range_tmp = range_x_new;
    dim_x = dim_y;
    dim_y = dim_tmp;

    range_x_new = range_y_new;
    range_y_new = range_tmp;
  }

  // expand smalll shape
  if (dim_x.size() != dim_y.size()) {
    int dec = dim_x.size() - dim_y.size();
    for (int i = 0; i < dec; i++) {
      dim_y.insert(dim_y.begin(), (int64_t)1);
      range_y_new.insert(range_y_new.begin(), {1, 1});
    }
  }

  // set out dims
  for (size_t i = 0; i < dim_x.size(); i++) {
    if ((dim_x[i] != dim_y[i]) && (dim_x[i] != 1) && (dim_y[i] != 1)) {
      OP_LOGE(op.GetName().c_str(), "The %s's dimensions does not match the broadcast rule(%lu %lu).",
              op.GetName().c_str(), dim_x[i], dim_y[i]);
      return false;
    }

    int64_t dim = std::max(dim_x[i], dim_y[i]);
    std::pair<int64_t, int64_t> range = {0, 0};
    if (range_x_new.size() > i && range_y_new.size() > i) {
      range.first = std::min(range_x_new[i].first, range_y_new[i].first);
      if (range_x_new[i].second == -1 || range_y_new[i].second == -1) {
          range.second = -1;
      } else {
          range.second = std::max(range_x_new[i].second, range_y_new[i].second);
      }
    }
    dim_out.push_back(dim);
    range_out.push_back(range);
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
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto condition_desc = op_info->MutableInputDesc("condition");
  vector<int64_t> condition_shape = condition_desc->MutableShape().GetDims();
  DataType condition_dtype = condition_desc->GetDataType();
  std::vector<std::pair<int64_t, int64_t>> condition_range;
  condition_desc->GetShapeRange(condition_range);

  auto then_desc = op_info->MutableInputDesc("then");
  vector<int64_t> then_shape = then_desc->MutableShape().GetDims();
  DataType then_dtype = then_desc->GetDataType();
  std::vector<std::pair<int64_t, int64_t>> then_range;
  then_desc->GetShapeRange(then_range);

  auto else_desc = op_info->MutableInputDesc("else");
  vector<int64_t> else_shape = else_desc->MutableShape().GetDims();
  DataType else_dtype = else_desc->GetDataType();
  std::vector<std::pair<int64_t, int64_t>> else_range;
  else_desc->GetShapeRange(else_range);

  std::vector<int64_t> condition_then_max_shape;
  std::vector<std::pair<int64_t, int64_t>> condition_then_max_range;

  if (!BroadCastTwoinOneout(op, condition_shape, then_shape, condition_range, then_range,
                            condition_then_max_shape, condition_then_max_range)) {
    return GRAPH_FAILED;
  }
  std::vector<int64_t> max_shape;
  std::vector<std::pair<int64_t, int64_t>> max_range;
  if (!BroadCastTwoinOneout(op, condition_then_max_shape, else_shape, condition_then_max_range,
                            else_range, max_shape, max_range)) {
    return GRAPH_FAILED;
  }
  auto result_desc = op_info->MutableOutputDesc("result");
  result_desc->SetShape(GeShape(max_shape));
  result_desc->SetShapeRange(max_range);
  result_desc->SetDataType(then_dtype);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SelectV2, SelectV2InferShape);
VERIFY_FUNC_REG(SelectV2, SelectV2Verify);
// ---------------SelectV2 END-----------------------

// ----------------SegmentMax-------------------
static bool SegmentShapeVerify(const Operator& op, const std::string& input_name, const std::string& segment_ids_name) {
  auto input_shape_dims = op.GetInputDesc("x").GetShape().GetDims();
  auto segment_ids_shape_dims = op.GetInputDesc("segment_ids").GetShape().GetDims();

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
    OP_LOGI("segment_max", "GetInputConstData %s failed.", segment_ids_name.c_str());
    first_axis_dims = -1;
  } else {
    auto data_type = op.GetInputDesc(segment_ids_name).GetDataType();
    std::vector<int64_t> const_data;
    if (!GetConstIntData(segment_ids, data_type, const_data)) {
      OP_LOGE("segment_max",
              "invalid data type of segment_ids,"
              "data_type is %d.",
              (int)data_type);
      return GRAPH_FAILED;
    }
    first_axis_dims = (*std::max_element(const_data.begin(), const_data.end())) + 1;
  }

  auto output_shape_dims = input_desc.GetShape().GetDims();
  if (output_shape_dims.empty()) {
    OP_LOGE(op.GetName().c_str(), "The dims of input is empty!");
    return GRAPH_FAILED;
  }
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
static bool SegmentDShapeVerify(const Operator& op, const std::string& input_name,
                                const std::string& segment_ids_name) {
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
    OP_LOGE(op.GetName().c_str(),
            "the length of "
            "segment_ids should be equal to shape[0].");
    return GRAPH_FAILED;
  }
  for (size_t dim = 0; dim < segment_ids.size(); dim++) {
    if (dim == 0 && segment_ids[dim] < 0) {
      OP_LOGE(op.GetName().c_str(), "segment_ids must be positive integer");
      return GRAPH_FAILED;
    }
    if (dim > 0 && segment_ids[dim] < segment_ids[dim - 1]) {
      OP_LOGE(op.GetName().c_str(),
              "segment_ids must "
              "be sorted(from small to large)");
      return GRAPH_FAILED;
    }
  }

  int64_t first_axis_dims = (*std::max_element(segment_ids.begin(), segment_ids.end())) + 1;

  auto output_shape_dims = input_desc.GetShape().GetDims();
  if (output_shape_dims.empty()) {
    OP_LOGE(op.GetName().c_str(), "The dims of input is empty!");
    return GRAPH_FAILED;
  }
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

// ----------------SliceD Op Begin ----------------------
IMPLEMT_VERIFIER(SliceD, SliceDVerify) {
  std::vector<int64_t> input_size;
  if (ge::GRAPH_SUCCESS != op.GetAttr("size", input_size)) {
    OpsGetAttrErrReport(op.GetName(), "size");
    OP_LOGE(op.GetName().c_str(),
            "The size op"
            "GetOpAttr ConstValue failed!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> input_begin;
  if (ge::GRAPH_SUCCESS != op.GetAttr("offsets", input_begin)) {
    OpsGetAttrErrReport(op.GetName(), "begin");
    OP_LOGE(op.GetName().c_str(),
            "The begin op"
            "GetOpAttr ConstValue failed!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(SliceDInferShape) {
  const vector<string> depends;
  PREPARE_DYNAMIC_SHAPE(depends);
  std::vector<int64_t> input_size;
  if (ge::GRAPH_SUCCESS != op.GetAttr("size", input_size)) {
    OpsGetAttrErrReport(op.GetName(), "size");
    OP_LOGE(op.GetName().c_str(),
            "The size op GetOpAttr"
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

  if ((int64_t)input_size.size() != (int64_t)dimNum) {
    OpsAttrValueErrReport(op.GetName(), "length of size", ConcatString((int64_t)dimNum),
                          ConcatString((int64_t)input_size.size()));
    OP_LOGE(op.GetName().c_str(),
            "the length of size"
            "must be equal to shape!");
    return GRAPH_FAILED;
  }
  if ((int64_t)input_begin.size() != (int64_t)dimNum) {
    OpsAttrValueErrReport(op.GetName(), "length of begin", ConcatString((int64_t)dimNum),
                          ConcatString((int64_t)input_begin.size()));
    OP_LOGE(op.GetName().c_str(),
            "the length of begin"
            "must be equal to shape!");
    return GRAPH_FAILED;
  }
  for (int64_t i = 0; i < (int64_t)dimNum; ++i) {
    if (input_size[i] > shape.GetDim(i) || input_size[i] < -1) {
      string excepted_value = ConcatString("in range[0,", shape.GetDim(i), "]");
      OpsAttrValueErrReport(op.GetName(), "size", excepted_value, ConcatString(input_size[i]));
      OP_LOGE(op.GetName().c_str(),
              "size must be greater"
              "than or equal to 0, and less than shape!");
      return GRAPH_FAILED;
    }
    if (input_begin[i] > shape.GetDim(i) || input_begin[i] < -1) {
      string excepted_value = ConcatString("in range[-1,", shape.GetDim(i), "] and cannot be equal to 0");
      OpsAttrValueErrReport(op.GetName(), "begin", excepted_value, ConcatString(input_begin[i]));
      OP_LOGE(op.GetName().c_str(),
              "begin must be , greater"
              "than or equal to -1, less than or equal to shape,"
              "and cannot be equal to 0!");
      return GRAPH_FAILED;
    }
  }
  std::vector<int64_t> outputList;
  for (size_t i = 0; i < dimNum; ++i) {
    if ((int)input_size[i] == -1) {
      outputList.push_back(static_cast<int>(shape.GetDim(i) - input_begin[i]));
    } else {
      outputList.push_back(static_cast<int>(input_size[i]));
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
// ----------------SliceD Op End ----------------------

// ---------------SliceDV2 Op Begin ------------------
COMMON_INFER_FUNC_REG(SliceDV2, SliceDInferShape);
// ---------------SliceDV2 Op End --------------------

// ----------------Slice Op Begin ----------------------
static void GetSliceConstValue(const Tensor& const_tensor, const DataType& dtype, std::vector<int64_t>& const_data) {
  size_t size = 0;
  if (dtype == ge::DT_INT32) {
    int32_t* const_data_ptr = (int32_t*)const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back((int32_t)((*(const_data_ptr + i))));
    }
  } else {
    int64_t* const_data_ptr = (int64_t*)const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back(((int64_t)(*(const_data_ptr + i))));
    }
  }
}

IMPLEMT_COMMON_INFERFUNC(SliceInferShape) {
  const vector<string> depend_names = {"offsets", "size"};
  PREPARE_DYNAMIC_SHAPE(depend_names);

  Tensor input_begin_tensor;
  Tensor input_size_tensor;
  auto input_desc = op.GetInputDesc("x");
  const Shape shape = input_desc.GetShape();
  DataType input_dtype = input_desc.GetDataType();
  std::vector<int64_t> input_begin;
  std::vector<int64_t> input_size;

  bool has_offsets = true;
  if (op.GetInputConstData("offsets", input_begin_tensor) != GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "Get offsets failed.");
    has_offsets = false;
  } else {
    DataType input_begin_dtype = op.GetInputDesc("offsets").GetDataType();
    GetSliceConstValue(input_begin_tensor, input_begin_dtype, input_begin);
  }

  bool has_size = true;
  if (op.GetInputConstData("size", input_size_tensor) != GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "Get size failed.");
    has_size = false;
  } else {
    DataType input_size_dtype = op.GetInputDesc("size").GetDataType();
    GetSliceConstValue(input_size_tensor, input_size_dtype, input_size);
  }

  bool is_unknown_rank = !has_size && !has_offsets && shape.GetDims() == UNKNOWN_RANK;
  if (is_unknown_rank) {
    TensorDesc output_desc = op.GetOutputDesc("y");
    output_desc.SetDataType(input_dtype);
    Shape outputShape(UNKNOWN_RANK);
    output_desc.SetShape(outputShape);
    OP_LOGD(op.GetName().c_str(), "output_shape:%s", to_string(output_desc.GetShape()).c_str());
    (void) op.UpdateOutputDesc("y", output_desc);
    return GRAPH_SUCCESS;
  }

  auto shape_dims = shape.GetDims();
  if (shape.GetDims() == UNKNOWN_RANK) {
    shape_dims.assign(std::max(input_begin.size(), input_size.size()), -1);
  }

  size_t dimNum = shape_dims.size();
  std::vector<int64_t> outputList;

  vector<pair<int64_t, int64_t>> ranges;
  input_desc.GetShapeRange(ranges);
  if (ranges.empty()) {
    MakeUpShapeRange(shape_dims, ranges);
  }

  if (!has_size && !has_offsets) {
    for (size_t i = 0; i < dimNum; ++i) {
      outputList.push_back(-1);
      ranges[i].first = 1;
    }
  } else if (!has_offsets && has_size) {
    for (size_t i = 0; i < dimNum; ++i) {
      if (input_size[i] == -1) {
        outputList.push_back(-1);
        ranges[i].first = 1;
      } else {
        outputList.push_back(input_size[i]);
        ranges[i].first = input_size[i];
        ranges[i].second = input_size[i];
      }
    }
  } else if (has_offsets && !has_size) {
    for (size_t i = 0; i < dimNum; ++i) {
      outputList.push_back(-1);
      ranges[i].first = 1;
      if (ranges[i].second != -1) {
        if (shape_dims[i] != -1) {
          ranges[i].second = std::min(ranges[i].second, shape_dims[i]);
        }
        ranges[i].second -= input_begin[i];
      }
    }
  } else {
    for (size_t i = 0; i < dimNum; ++i) {
      if (input_size[i] == -1) {
        if (shape_dims[i] == -1) {
          outputList.push_back(-1);
        } else {
          outputList.push_back(shape_dims[i] - input_begin[i]);
        }

        ranges[i].first = 1;
      } else {
        outputList.push_back(input_size[i]);
        ranges[i].first = input_size[i];
        ranges[i].second = input_size[i];
      }
    }
  }

  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetDataType(input_dtype);
  if (IsUnKnownShape(outputList)) {
    tensordesc_output.SetShapeRange(ranges);
    OP_LOGD(op.GetName().c_str(), "output_ranges:%s", to_string(ranges).c_str());
  }

  Shape outputShape(outputList);
  tensordesc_output.SetShape(outputShape);
  OP_LOGD(op.GetName().c_str(), "output_ranges:%s", to_string(ranges).c_str());
  OP_LOGD(op.GetName().c_str(), "offset:%s", to_string(input_begin).c_str());
  OP_LOGD(op.GetName().c_str(), "size:%s", to_string(input_size).c_str());
  OP_LOGD(op.GetName().c_str(), "output_shape:%s", to_string(tensordesc_output.GetShape()).c_str());
  (void) op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Slice, SliceInferShape);
// ----------------Slice Op End----------------------

// ----------------OneHotD--------------------------
static graphStatus OneHotInferShapeAndType(ge::Operator& op, DataType& input_type, std::int64_t& depth, int32_t axis) {
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
        dim_vector.push_back(indices_shape.GetDim(i - 1));
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
    OP_LOGE(op.GetName().c_str(), "OneHot GetOpAttr depth failed!");
    return GRAPH_FAILED;
  }
  if (depth < 1) {
    OpsAttrValueErrReport(op.GetName(), "depth", "greater than or equals to 1", ConcatString(depth));
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
    OpsInputShapeDimErrReport(op.GetName(), "axis", ConcatString(dim_num), ConcatString(-dim_num), ConcatString(axis));
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
  const vector<string> depend_names = {"depth"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  // get attr axis
  int32_t axis = -1;
  if (ge::GRAPH_SUCCESS != op.GetAttr("axis", axis)) {
    OP_LOGE(op.GetName().c_str(), "Get const axis failed from op of 'OneHot'!\n");
    return GRAPH_FAILED;
  }

  auto node = NodeUtils::GetNodeFromOperator(op);
  // get all Desc info
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  vector<int64_t> input_shape = input_desc->MutableShape().GetDims();

  auto value_desc = op_info->MutableInputDesc("on_value");
  DataType value_dtype = value_desc->GetDataType();

  // output desc and set dtype
  auto output_desc = op_info->MutableOutputDesc("y");
  output_desc->SetDataType(value_dtype);

  if (IsUnknownRankShape(input_shape)) {
    // input is UnknownRank, set output UnknownRank
    OP_LOGW(op.GetName().c_str(), "input shape is UnknownRank, set output UnknownRank");
    output_desc->SetShape(GeShape(input_shape));
    return GRAPH_SUCCESS;
  }
  // update axis to positive number
  // axis = input_shape.size() == 0 ? 0 : axis % (input_shape.size() + 1);

  // get depth const value
  GeTensorPtr depth_tensor = nullptr;
  vector<int64_t> depth_value;
  if (GRAPH_SUCCESS == NodeUtils::GetInputConstData(node, "depth", depth_tensor)) {
    auto const_desc = op_info->MutableInputDesc("depth");
    auto const_dtype = const_desc->GetDataType();
    if (!GetConstValue(op, depth_tensor, const_dtype, depth_value)) {
      OP_LOGW(op.GetName().c_str(), "Get depth const from const tensor failed, set depth -1");
      depth_value.clear();
      depth_value.push_back(-1);
    }
  } else {
    OP_LOGW(op.GetName().c_str(), "Get depth const tensor failed, set depth -1");
    depth_value.clear();
    depth_value.push_back(-1);
  }

  // update output shape
  vector<int64_t> output_shape(input_shape);
  if (-1 == axis) {
    output_shape.insert(output_shape.end(), (int64_t)depth_value[0]);
  } else {
    output_shape.insert(output_shape.begin() + axis, (int64_t)depth_value[0]);
  }
  output_desc->SetShape(GeShape(output_shape));

  // if output shape is dynamic update output range
  if (IsUnknown(output_shape)) {
    output_desc->SetOriginShape(GeShape(output_shape));
    std::vector<std::pair<int64_t, int64_t>> input_range;
    input_desc->GetShapeRange(input_range);
    MakeUpShapeRange(input_shape, input_range);
    std::pair<int64_t, int64_t> depth_range = depth_value[0] == -1 ?
                                              std::pair<int64_t, int64_t>(1, -1):
                                              std::pair<int64_t, int64_t>(depth_value[0], depth_value[0]);
    if (-1 == axis) {
      input_range.insert(input_range.end(), depth_range);
    } else {
      input_range.insert(input_range.begin() + axis, depth_range);
    }
    output_desc->SetShapeRange(input_range);
  }

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(OneHot, OneHotInferShape);
// ----------------OneHot END----------------------

static bool TopKInferCommon(Operator &op, int64_t k) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto output_v_desc = op_info->MutableOutputDesc("values");
  auto output_i_desc = op_info->MutableOutputDesc("indices");

  std::vector<int64_t> dims_in = input_desc->MutableShape().GetDims();
  int32_t dim_size = dims_in.size();
  if (dim_size <= 0) {
    OP_LOGE(op.GetName().c_str(), "The dims_in size should more than 0!");
    return false;
  }

  int32_t dim = dim_size - 1;
  int32_t sorted_axis = dim;
  if (op.GetAttr("dim", dim) == GRAPH_SUCCESS) {
    sorted_axis = dim;
    if (sorted_axis < 0) {
      sorted_axis += dim_size;
    }
    if (sorted_axis >= dim_size) {
      OP_LOGE(op.GetName().c_str(), "Dim is out of shape size.");
      return false;
    }
  }
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  input_desc->GetShapeRange(shape_range);
  if (shape_range.size() > 0) {
    if (k > 0 && sorted_axis < shape_range.size()) {
      shape_range[sorted_axis].first = k;
      shape_range[sorted_axis].second = k;
    }
  } else {
    // input is static shape
    for (int i = 0; i < dims_in.size(); i++) {
      if (i == sorted_axis && k > 0) {
        shape_range.push_back(pair<int64_t, int64_t>(k, k));
      } else {
        shape_range.push_back(pair<int64_t, int64_t>(dims_in[i], dims_in[i]));
      }
    }
  }

  bool unknown_rank = IsUnknownRankShape(dims_in);
  if (unknown_rank) {
    output_v_desc->SetShape(GeShape(UNKNOWN_RANK));
    output_v_desc->SetOriginShape(GeShape(UNKNOWN_RANK));

    output_i_desc->SetShape(GeShape(UNKNOWN_RANK));
    output_i_desc->SetOriginShape(GeShape(UNKNOWN_RANK));
  } else {
    dims_in[sorted_axis] = k;

    output_v_desc->SetShape(GeShape(dims_in));
    output_v_desc->SetShapeRange(shape_range);

    output_i_desc->SetShape(GeShape(dims_in));
    output_i_desc->SetShapeRange(shape_range);
  }
  output_v_desc->SetDataType(input_desc->GetDataType());
  output_i_desc->SetDataType(DT_INT32);
  return true;
}

// ----------------TopKD Op Start-------------------
IMPLEMT_COMMON_INFERFUNC(TopKDInferShape) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);

  int32_t k;
  if (op.GetAttr("k", k) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get attr k failed");
    return GRAPH_FAILED;
  }

  if (TopKInferCommon(op, k) == false) {
    OP_LOGE(op.GetName().c_str(), "TopKInferCommon Failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(TopKD, TopKDInferShape);
// ----------------TopKD Op End-------------------

// ----------------TopK Op-------------------
IMPLEMT_VERIFIER(TopK, TopKVerify) { return GRAPH_SUCCESS; }

IMPLEMT_COMMON_INFERFUNC(TopKInferShape) {
  const vector<string> depend_names = {"k"};
  PREPARE_DYNAMIC_SHAPE(depend_names);

  auto op_info = OpDescUtils::GetOpDescFromOperator(op);

  Tensor k_tensor;
  bool unkonwn_dim_flag{false};
  if (op.GetInputConstData("k", k_tensor) != GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "Get constdata failed, unknown dim.");
    unkonwn_dim_flag = true;
  }

  // Tensor::GetData() return a uint8 ptr. However the definition of k is int32.
  // So here use int32* ptr to get the k value
  int64_t k = UNKNOWN_DIM;
  if (!unkonwn_dim_flag && k_tensor.GetData() != nullptr) {
    DataType dtype = op.GetInputDesc("k").GetDataType();
    if (dtype == DT_INT32) {
      k = static_cast<int64_t>(*(reinterpret_cast<int32_t*>(k_tensor.GetData())));
    } else if (dtype == DT_INT64) {
      k = *(reinterpret_cast<int64_t*>(k_tensor.GetData()));
    } else {
      OP_LOGE(op.GetName().c_str(), "The type of k Error!");
      return GRAPH_FAILED;
    }
  }

  if (TopKInferCommon(op, k) == false) {
    OP_LOGE(op.GetName().c_str(), "TopKInferCommon Failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(TopK, TopKInferShape);
VERIFY_FUNC_REG(TopK, TopKVerify);
// ----------------TopK Op End-------------------

// ----------------TopKV2 Op---------------------
IMPLEMT_VERIFIER(TopKV2, TopKV2Verify) { return GRAPH_SUCCESS; }
IMPLEMT_COMMON_INFERFUNC(TopKV2InferShape) {
  const vector<string> depend_names = {"k"};
  PREPARE_DYNAMIC_SHAPE(depend_names);

  auto op_info = OpDescUtils::GetOpDescFromOperator(op);

  Tensor k_tensor;
  bool unkonwn_dim_flag{false};
  if (op.GetInputConstData("k", k_tensor) != GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "Get constdata failed, unknown dim.");
    unkonwn_dim_flag = true;
  }

  // Tensor::GetData() return a uint8 ptr. However the definition of k is int32.
  // So here use int32* ptr to get the k value
  int64_t k = UNKNOWN_DIM;
  if (!unkonwn_dim_flag && k_tensor.GetData() != nullptr) {
    DataType dtype = op.GetInputDesc("k").GetDataType();
    if (dtype == DT_INT32) {
      k = static_cast<int64_t>(*(reinterpret_cast<int32_t*>(k_tensor.GetData())));
    } else if (dtype == DT_INT64) {
      k = *(reinterpret_cast<int64_t*>(k_tensor.GetData()));
    } else {
      OP_LOGE(op.GetName().c_str(), "The type of k Error!");
      return GRAPH_FAILED;
    }
  }

  if (TopKInferCommon(op, k) == false) {
    OP_LOGE(op.GetName().c_str(), "TopKInferCommon Failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(TopKV2, TopKV2InferShape);
VERIFY_FUNC_REG(TopKV2, TopKV2Verify);
// ----------------TopKV2 Op End-----------------

// ----------------ScatterNd-------------------
IMPLEMT_COMMON_INFERFUNC(ScatterNdInferShape) {
  vector<string> input_infer_depends = {"shape"};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends(input_infer_depends);

  auto output_desc = op_desc->MutableOutputDesc("y");
  auto shape_desc = op_desc->MutableInputDesc("shape");
  std::vector<int64_t> shape_shape = shape_desc->MutableShape().GetDims();
  std::vector<std::pair<int64_t, int64_t>> out_range;
  Tensor shape;
  std::vector<int64_t> const_data;
  if (GRAPH_SUCCESS != op.GetInputConstData("shape", shape)) {
    const_data = {-2};
  } else {
    auto data_type = shape_desc->GetDataType();
    if (!GetConstIntData(shape, data_type, const_data)) {
      USER_GE_LOGE("Invalid data type of shape, data_type is %d.", (int)data_type);
      return GRAPH_FAILED;
    }
  }

  vector<int64_t> shape_dims;
  if (shape_shape.size() == 1 && shape_shape[0] > 0 && IsUnknownRankShape(const_data)) {
    for (int64_t i = 0; i < shape_shape[0]; i++) {
      shape_dims.push_back(-1);
    }
  } else {
    for (size_t i = 0; i < (uint32_t)const_data.size(); ++i) {
      shape_dims.push_back(const_data[i]);
    }
  }

  if (IsUnknownRankShape(shape_dims)) {
    out_range.push_back(std::pair<int64_t, int64_t>(1, -1));
  } else if (IsUnknownVec(shape_dims)) {
    for (size_t i = 0; i < shape_dims.size(); i++) {
      if (shape_dims[i] == -1) {
        out_range.push_back(std::pair<int64_t, int64_t>(1, -1));
      } else {
        out_range.push_back(std::pair<int64_t, int64_t>(shape_dims[i], shape_dims[i]));
      }
    }
  }

  GeShape output_shape(shape_dims);
  output_desc->SetShape(output_shape);
  output_desc->SetShapeRange(out_range);
  output_desc->SetDataType(op_desc->MutableInputDesc("x")->GetDataType());
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ScatterNd, ScatterNdInferShape);
// ----------------ScatterNd End-------------------

// ----------------ScatterNdD-------------------
IMPLEMT_COMMON_INFERFUNC(ScatterNdDInferShape) {
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
  if (shape_out_list.size() != shape_dims.size()) {
    string excepted_value = ConcatString("same with output_y[", shape_dims.size(), "]");
    OpsAttrValueErrReport(op.GetName(), "x'shape", excepted_value, ConcatString(shape_out_list.size()));
    OP_LOGE(op.GetName().c_str(), "the len of shape must be same with output_y.");
    return GRAPH_FAILED;
  }
  for (int64_t i = 0; i < (int64_t)shape_dims.size(); i++) {
    if (shape_out_list[i] != shape_dims[i]) {
      string excepted_value = ConcatString("same with output_y[", shape_dims[i], "]");
      OpsAttrValueErrReport(op.GetName(), "x'shape", excepted_value, ConcatString(shape_out_list[i]));
      OP_LOGE(op.GetName().c_str(), "shape must be same with output_y.");
      return GRAPH_FAILED;
    }
  }
  Shape output_shape(shape_dims);
  output_desc.SetShape(output_shape);
  op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ScatterNdD, ScatterNdDInferShape);
// ----------------ScatterNdD End-------------------

// ----------------InTopKD Op-------------------
bool InTopKDCheckInputX1AndX2(const Operator& op) {
  Shape shape_prediction = op.GetInputDesc("x1").GetShape();
  Shape shape_target = op.GetInputDesc("x2").GetShape();
  size_t prediction_dim = shape_prediction.GetDimNum();
  if (prediction_dim != 2) {
    OP_LOGE(op.GetName().c_str(), "Predictions must be 2-dimensional but get %u", prediction_dim);
    return false;
  }
  size_t target_dim = shape_target.GetDimNum();
  if (target_dim != 1) {
    OP_LOGE(op.GetName().c_str(), "Target must be 1-dimensional, but get %u", target_dim);
    return false;
  }
  if (shape_prediction.GetDim(0) != shape_target.GetDim(0)) {
    OP_LOGE(op.GetName().c_str(),
            "First dimension of prediction must match length of targets, but first dimension of prediction get %d",
            shape_prediction.GetDim(0));
    return false;
  }
  return true;
}

bool InTopKDCheckInputAttrK(const Operator& op) {
  int dim_zero{0};
  Shape shape_k = op.GetInputDesc("k").GetShape();
  int k_dim = shape_k.GetDimNum();
  if (k_dim != dim_zero) {
    OP_LOGE(op.GetName().c_str(), "Attr k must be 0 D, but get %d\n", k_dim);
    return false;
  }
  return true;
}

IMPLEMT_VERIFIER(InTopKD, InTopKDVerify) {
  if (!InTopKDCheckInputX1AndX2(op)) {
    return GRAPH_FAILED;
  }
  if (!InTopKDCheckInputAttrK(op)) {
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
  if (op.UpdateOutputDesc("y", tensordesc_output) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(InTopKD, InTopKDInferShape);
VERIFY_FUNC_REG(InTopKD, InTopKDVerify);
// ---------------InTopKD------------------

// ----------------InTopK Op Start-------------------
bool InTopKCheckInputX1AndX2(const Operator& op) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_prediction = op_info->MutableInputDesc("x1");
  if (input_prediction == nullptr) {
    OP_LOGE(op.GetName().c_str(), "Get constValue failed of [x1].");
    return false;
  }
  auto input_target = op_info->MutableInputDesc("x2");
  if (input_target == nullptr) {
    OP_LOGE(op.GetName().c_str(), "Get constValue failed of [x2].");
    return false;
  }
  auto output_desc = op_info->MutableOutputDesc("y");
  if (output_desc == nullptr) {
    OP_LOGE(op.GetName().c_str(), "Get constValue failed of [y].");
    return false;
  }
  std::vector<int64_t> dims_in_prediction = input_prediction->MutableShape().GetDims();
  std::vector<int64_t> dims_in_target = input_target->MutableShape().GetDims();

  bool unknown_rank = IsUnknownRankShape(dims_in_prediction);
  size_t target_dim = input_target->GetShape().GetDimNum();
  if (target_dim != 1) {
    OP_LOGE(op.GetName().c_str(), "Target must be 1-dimensional, but get %u", target_dim);
    return false;
  }
  if (!unknown_rank) {
    size_t prediction_dim = input_prediction->GetShape().GetDimNum();
    if (prediction_dim != 2) {
      OP_LOGE(op.GetName().c_str(), "Predictions must be 2-dimensional, but get %u", prediction_dim);
      return false;
    }
    if (input_prediction->GetShape().GetDim(0) != -1 && input_target->GetShape().GetDim(0) != -1) {
      if (input_prediction->GetShape().GetDim(0) != input_target->GetShape().GetDim(0)) {
        OP_LOGE(op.GetName().c_str(),
                "First dimension of prediction must match length of targets, but first dimension of prediction get %d",
                input_prediction->GetShape().GetDim(0));
        return false;
      }
    }
  }
  return true;
}

IMPLEMT_VERIFIER(InTopK, InTopKVerify) {
  if (!InTopKCheckInputX1AndX2(op)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(InTopKInferShape) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_prediction = op_info->MutableInputDesc("x1");
  if (input_prediction == nullptr) {
    OP_LOGE(op.GetName().c_str(), "Get constValue failed of [x1].");
    return GRAPH_FAILED;
  }
  auto input_target = op_info->MutableInputDesc("x2");
  if (input_target == nullptr) {
    OP_LOGE(op.GetName().c_str(), "Get constValue failed of [x2].");
    return GRAPH_FAILED;
  }
  auto output_desc = op_info->MutableOutputDesc("y");
  if (output_desc == nullptr) {
    OP_LOGE(op.GetName().c_str(), "Get constValue failed of [y].");
    return GRAPH_FAILED;
  }
  std::vector<std::pair<int64_t, int64_t>> input1_range;
  input_target->GetShapeRange(input1_range);
  std::vector<int64_t> dims_in_prediction = input_prediction->MutableShape().GetDims();
  std::vector<int64_t> dims_in_target = input_target->MutableShape().GetDims();
  bool unknown_rank = IsUnknownRankShape(dims_in_prediction);
  if (!InTopKCheckInputX1AndX2(op)) {
    return GRAPH_FAILED;
  }
  if (unknown_rank) {
    output_desc->SetShape(GeShape(UNKNOWN_RANK));
    output_desc->SetOriginShape(GeShape(UNKNOWN_RANK));
  } else {
    output_desc->SetShape(GeShape(dims_in_target));
    output_desc->SetOriginShape(GeShape(dims_in_target));
    if (input_target->GetShape().GetDim(0) == -1 || input_prediction->GetShape().GetDim(0) == -1 ||
        input_prediction->GetShape().GetDim(1) == -1) {
      if (output_desc->SetShapeRange(input1_range) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "SetShapeRange return failed.");
	return GRAPH_FAILED;
      }
    }
  }
  output_desc->SetDataType(DT_BOOL);
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
  if (op.UpdateOutputDesc("var", tensordesc_output) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(StridedSliceAssign, StridedSliceAssignInferShape);
// ----------------StridedSliceAssign Op Begin-------------------

// ----------------StridedSliceAssignD Op Begin-------------------
IMPLEMT_COMMON_INFERFUNC(StridedSliceAssignDInferShape) {
  std::vector<int64_t> begin;
  begin = GetAttrValue(op, "begin");
  if (!CheckListEmpty(op.GetName(), begin, "begin")) {
    OP_LOGE(op.GetName().c_str(), "Get attr begin failed");
    return GRAPH_FAILED;
  }
  if (begin.size() > 8) {
    OP_LOGE(op.GetName().c_str(), "Attr begin(%u) is too large", begin.size());
    return GRAPH_FAILED;
  }
  std::vector<int64_t> end;
  end = GetAttrValue(op, "end");
  if (!CheckListEmpty(op.GetName(), end, "end")) {
    OP_LOGE(op.GetName().c_str(), "Get attr end failed");
    return GRAPH_FAILED;
  }
  if (end.size() > 8) {
    OP_LOGE(op.GetName().c_str(), "Attr end(%u) is too large", end.size());
    return GRAPH_FAILED;
  }
  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), strides, "strides")) {
    OP_LOGE(op.GetName().c_str(), "Get attr strides failed");
    return GRAPH_FAILED;
  }
  if (strides.size() > 8) {
    OP_LOGE(op.GetName().c_str(), "Attr strides(%u) is too large", strides.size());
    return GRAPH_FAILED;
  }
  TensorDesc tensordesc_output = op.GetOutputDesc("var");
  tensordesc_output.SetShape(op.GetInputDesc("var").GetShape());
  tensordesc_output.SetDataType(op.GetInputDesc("var").GetDataType());
  if (op.UpdateOutputDesc("var", tensordesc_output) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(StridedSliceAssignD, StridedSliceAssignDInferShape);
// ----------------StridedSliceAssignD Op Begin-------------------

// ----------------Cumprod-------------------
IMPLEMT_COMMON_INFERFUNC(CumprodInferShape) {
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(op.GetInputDesc("x").GetShape());
  output_desc.SetDataType(op.GetInputDesc("x").GetDataType());
  op.UpdateOutputDesc("y", output_desc);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Cumprod, CumprodInferShape);
// ----------------Cumprod END-------------------

// ----------------CumprodD-------------------
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
    OpsInputShapeDimErrReport(op.GetName(), "axis", ConcatString(dimnum), ConcatString(-dimnum), ConcatString(axis));
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
// ----------------CumprodD END-------------------

// ----------------Cumsum-------------------
IMPLEMT_COMMON_INFERFUNC(CumsumInferShape) {
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(op.GetInputDesc("x").GetShape());
  output_desc.SetDataType(op.GetInputDesc("x").GetDataType());
  op.UpdateOutputDesc("y", output_desc);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Cumsum, CumsumInferShape);
// ----------------Cumsum END-------------------

// ----------------CumsumD-------------------
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
    OpsInputShapeDimErrReport(op.GetName(), "axis", ConcatString(dimnum), ConcatString(-dimnum), ConcatString(axis));
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
// ----------------CumsumD END-------------------

// ----------------Cummin------------------------
IMPLEMT_COMMON_INFERFUNC(CumminInferShape) {
        TensorDesc output_desc_y = op.GetOutputDesc("y");
        TensorDesc output_desc_argmin = op.GetOutputDesc("indices");
        DataType predict_dtype = op.GetInputDesc("x").GetDataType();
        Format predict_format = op.GetInputDesc("x").GetFormat();
        ge::Shape output_shape = op.GetInputDesc("x").GetShape();
        output_desc_y.SetDataType(predict_dtype);
        output_desc_y.SetFormat(predict_format);
        output_desc_y.SetShape(output_shape);
        output_desc_argmin.SetDataType(DT_INT32);
        output_desc_argmin.SetFormat(predict_format);
        output_desc_argmin.SetShape(output_shape);
        (void)op.UpdateOutputDesc("y", output_desc_y);
        (void)op.UpdateOutputDesc("indices", output_desc_argmin);
        return GRAPH_SUCCESS;
    }

    IMPLEMT_VERIFIER(Cummin, CumminVerify) {
        return GRAPH_SUCCESS;
    }

    COMMON_INFER_FUNC_REG(Cummin, CumminInferShape);
    VERIFY_FUNC_REG(Cummin, CumminVerify);
// ----------------Cummin END----------------------

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
    string excepted_value = ConcatString("same as indices[", (int64_t)indices.size(), "]");
    OpsAttrValueErrReport(op.GetName(), "v's length of rank 0", excepted_value, ConcatString(dim_value_v));
    OP_LOGE(op.GetName().c_str(),
            "The length of rank 0 of"
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
    OpsAttrValueErrReport(op.GetName(), "v", ConcatString(dim_value_v), ConcatString((int64_t)indices.size()));
    OP_LOGE(op.GetName().c_str(),
            "The length of rank 0 of"
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
    string excepted_value = ConcatString("same as indices[", (int64_t)indices.size(), "]");
    OpsAttrValueErrReport(op.GetName(), "v's length of rank 0", excepted_value, ConcatString(dim_value_v));
    OP_LOGE(op.GetName().c_str(),
            "The length of rank 0 of"
            "tensor v must be the same as length of indices.");
    return GRAPH_FAILED;
  }
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(InplaceSubD, InplaceSubDInferShape);
// ----------------InplaceSubD  END-------------------

// ----------------UnsortedSegmentMinD-------------------
IMPLEMT_COMMON_INFERFUNC(UnsortedSegmentMinDInferShape) {
  auto input_desc = op.GetInputDesc("x");
  const std::string kNumSegmentsName = "num_segments";
  int64_t num_segments;
  if (op.GetAttr(kNumSegmentsName, num_segments) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "GetAttr %s failed.", kNumSegmentsName.c_str());
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
  if (op.UpdateOutputDesc("y", tensordesc_output) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(UnsortedSegmentMinD, UnsortedSegmentMinDInferShape);
// ----------------UnsortedSegmentMinD END-------------------

// ----------------UnsortedSegmentMaxD-------------------
IMPLEMT_COMMON_INFERFUNC(UnsortedSegmentMaxDInferShape) {
  auto input_desc = op.GetInputDesc("x");
  const std::string kNumSegmentsName = "num_segments";
  int64_t num_segments;
  if (op.GetAttr(kNumSegmentsName, num_segments) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "GetAttr %s failed.", kNumSegmentsName.c_str());
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
  if (op.UpdateOutputDesc("y", tensordesc_output) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(UnsortedSegmentMaxD, UnsortedSegmentMaxDInferShape);
// ----------------UnsortedSegmentMaxD END-------------------

// ----------------UnsortedSegmentProdD----------------------
IMPLEMT_COMMON_INFERFUNC(UnsortedSegmentProdDInferShape) {
  auto input_desc = op.GetInputDesc("x");
  const std::string kNumSegmentsName = "num_segments";
  int64_t num_segments;
  if (op.GetAttr(kNumSegmentsName, num_segments) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "GetAttr %s failed.", kNumSegmentsName.c_str());
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
  if (op.UpdateOutputDesc("y", tensordesc_output) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(UnsortedSegmentProdD, UnsortedSegmentProdDInferShape);

// ----------------ScatterNDNonAliasingAdd-------------------
IMPLEMT_VERIFIER(ScatterNonAliasingAdd, ScatterNonAliasingAddVerify) {
  if (!CheckTwoInputDtypeSame(op, "x", "updates")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ScatterNonAliasingAddInferShape) {
  // main part of shape infer
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ge::GeShape x_shape = op_desc->MutableInputDesc("x")->GetShape();
  std::vector<std::pair<int64_t, int64_t>> x_shape_range;
  op_desc->MutableInputDesc("x")->GetShapeRange(x_shape_range);
  DataType input_dtype = op_desc->MutableInputDesc("x")->GetDataType();
  GeTensorDescPtr td = op_desc->MutableOutputDesc("y");
  td->SetShape(x_shape);
  td->SetDataType(input_dtype);
  td->SetShapeRange(x_shape_range);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ScatterNonAliasingAdd, ScatterNonAliasingAddInferShape);
VERIFY_FUNC_REG(ScatterNonAliasingAdd, ScatterNonAliasingAddVerify);
// ------------------ScatterNDNonAliasingAdd END---------------------

// ----------------proposal-------------------
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
    OpsGetAttrErrReport(op.GetName(), "post_nms_topn");
    OP_LOGE(op.GetName().c_str(), "get attr failed");
    return GRAPH_FAILED;
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

// ----------------proposal_d-------------------
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
    OpsGetAttrErrReport(op.GetName(), "post_nms_topn");
    OP_LOGE(op.GetName().c_str(), "get attr failed");
    return GRAPH_FAILED;
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

// ----------------PassThrough-------------------
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
  if (stride == 0) {
    OP_LOGE(op.GetName().c_str(), "stride[%d] error.", stride);
    return GRAPH_FAILED;
  }
  if (GRAPH_SUCCESS != op.GetAttr("reverse", reverse)) {
    reverse = false;
  }
  std::vector<int64_t> outputShapeVec;
  if (reverse) {
    if (inputFormat == FORMAT_NCHW) {
      outputShapeVec.push_back(inputShape[0]);
      outputShapeVec.push_back(inputShape[1] / (stride * stride));
      outputShapeVec.push_back(inputShape[2] * stride);
      outputShapeVec.push_back(inputShape[3] * stride);
    } else {
      outputShapeVec.push_back(inputShape[0]);
      outputShapeVec.push_back(inputShape[1] * stride);
      outputShapeVec.push_back(inputShape[2] * stride);
      outputShapeVec.push_back(inputShape[3] / (stride * stride));
    }
  } else {
    if (inputFormat == FORMAT_NCHW) {
      outputShapeVec.push_back(inputShape[0]);
      outputShapeVec.push_back(inputShape[1] * (stride * stride));
      outputShapeVec.push_back(inputShape[2] / stride);
      outputShapeVec.push_back(inputShape[3] / stride);
    } else {
      outputShapeVec.push_back(inputShape[0]);
      outputShapeVec.push_back(inputShape[1] / stride);
      outputShapeVec.push_back(inputShape[2] / stride);
      outputShapeVec.push_back(inputShape[3] * (stride * stride));
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
    OpsInputFormatErrReport(op.GetName().c_str(), "inputFormat", "NCHW or NHWC", ConcatString(inputFormat));
    return GRAPH_FAILED;
  }

  if (reverse) {
    if (stride < 1) {
      OP_LOGE("[ERROR]the PassThrough op forward do not supported the stride!");
      OpsAttrValueErrReport(op.GetName().c_str(), "stride", "greater than 0", ConcatString(stride));
      return GRAPH_FAILED;
    }
    int64_t modC = (inputFormat == FORMAT_NCHW) ? (int64_t)inputShape[1] % (stride * stride)
                                                : (int64_t)inputShape[3] % (stride * stride);
    if (modC != 0) {
      OP_LOGE("[ERROR]the PassThrough op forward do not supported the stride!");
      OpsAttrValueErrReport(op.GetName().c_str(), "axis C", "times of stride'squre", ConcatString(modC));
      return GRAPH_FAILED;
    }

  } else {
    if (stride < 1) {
      OP_LOGE("[ERROR]the PassThrough op backward do not supported the stride!");
      OpsAttrValueErrReport(op.GetName().c_str(), "stride", "greater than 0", ConcatString(stride));
      return GRAPH_FAILED;
    }
    int64_t modH = (inputFormat == FORMAT_NCHW) ? (int64_t)inputShape[2] % stride : (int64_t)inputShape[1] % stride;
    int64_t modW = (inputFormat == FORMAT_NCHW) ? (int64_t)inputShape[3] % stride : (int64_t)inputShape[2] % stride;
    if (modH != 0) {
      OP_LOGE("[ERROR]the PassThrough op backward do not supported the stride!");
      OpsAttrValueErrReport(op.GetName().c_str(), "axis H", "times of stride", ConcatString(modH));
      return GRAPH_FAILED;
    }
    if (modW != 0) {
      OP_LOGE("[ERROR]the PassThrough op backward do not supported the stride!");
      OpsAttrValueErrReport(op.GetName().c_str(), "axis W", "times of stride", ConcatString(modW));
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(PassThrough, PassThroughInferShape);

VERIFY_FUNC_REG(PassThrough, PassThroughVerify);

// ----------------Crop-------------------
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
    OpsGetAttrErrReport(op.GetName().c_str(), "axis");
    OP_LOGE("Failed to get attribute axis");
    return GRAPH_FAILED;
  }
  if (axis >= dimNum || axis < -dimNum) {
    string minvalue = ConcatString(-dimNum);
    string maxvalue = ConcatString(dimNum - 1);
    string excepted_value = ConcatString("in the range of[", minvalue, ",", maxvalue, "]");
    OpsAttrValueErrReport(op.GetName(), "axis", excepted_value, ConcatString(axis));
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
// ----------------Crop-------------------

// ----------------------TileWithAxis-----------------
IMPLEMT_INFERFUNC(TileWithAxis, TileWithAxisInferShape) {
  OP_LOGI("Enter TileWithAxisInferShape");
  TensorDesc outputDesc = op.GetOutputDesc("y");
  TensorDesc inputDesc = op.GetInputDesc("x");

  auto inputDtype = inputDesc.GetDataType();
  outputDesc.SetDataType(inputDtype);

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

  if (inputFormat == FORMAT_NC1HWC0) {
    if (originShape.GetDimNum() == 4) {
      if (originFormat == FORMAT_NCHW) {
        if (axis < 0) {
          axis = axis - 1;
        }
      } else if (originFormat == FORMAT_NHWC) {
        if (axis == -4) {
          axis = -5;
        } else if (axis == -1) {
          axis = -4;
        } else if (axis == 1) {
          axis = 2;
        } else if (axis == 2) {
          axis = 3;
        } else if (axis == 3) {
          axis = 1;
        }
      } else {
        OP_LOGE(op.GetName().c_str(), "5D tensor's origin format should in [NCHW, NHWC]");
        return GRAPH_FAILED;
      }
    } else {
      OP_LOGE(op.GetName().c_str(), "5D tensor's origin shape should be 4D tensor");
      return GRAPH_FAILED;
    }

    if (axis < 0) {
      axis = axis + 5;
    }
    
    if (axis == 1 || axis == 4) {
      OP_LOGE(op.GetName().c_str(), "5D tensor's axis is invalid");
      return GRAPH_FAILED;
    }
  } else if (axis < 0) {
    axis = axis + dimsX.size();
  }

  if (dimsX[axis] != -1) {
    dimsX[axis] = dimsX[axis] * tiles;
  }
  ge::Shape outputShape = ge::Shape(dimsX);
  outputDesc.SetShape(outputShape);

  std::vector<std::pair<int64_t, int64_t>> inputRange;
  inputDesc.GetShapeRange(inputRange);
  std::vector<std::pair<int64_t, int64_t>> outputRange;
  for (size_t i = 0; i < inputRange.size(); i++) {
    if (i == axis) {
      auto range = std::make_pair(inputRange[axis].first * tiles, inputRange[axis].second * tiles);
      outputRange.push_back(range);
    } else {
      outputRange.push_back(inputRange[i]);
    }
  }
  outputDesc.SetShapeRange(outputRange);

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

  bool flag = (axis >= (static_cast<int>(xShape.size()) * (-1))) && (axis < static_cast<int>(xShape.size()));
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

// Registered inferfunction
INFER_FUNC_REG(TileWithAxis, TileWithAxisInferShape);

// Registered verify function
VERIFY_FUNC_REG(TileWithAxis, TileWithAxisVerify);

// ----------------------TileWithAxis-----------------

// ----------------read_select-------------------
IMPLEMT_COMMON_INFERFUNC(ReadSelectInferShape) {
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(ReadSelect, ReadSelectVerify) {
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ReadSelect, ReadSelectInferShape);

VERIFY_FUNC_REG(ReadSelect, ReadSelectVerify);

// ----------------write_select-------------------
IMPLEMT_COMMON_INFERFUNC(WriteSelectInferShape) {
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(WriteSelect, WriteSelectVerify) {
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(WriteSelect, WriteSelectInferShape);

VERIFY_FUNC_REG(WriteSelect, WriteSelectVerify);

// ----------------strided_read-------------------
IMPLEMT_COMMON_INFERFUNC(StridedReadInferShape) {
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(StridedRead, StridedReadVerify) {
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(StridedRead, StridedReadInferShape);

VERIFY_FUNC_REG(StridedRead, StridedReadVerify);

// ----------------strided_write-------------------
IMPLEMT_COMMON_INFERFUNC(StridedWriteInferShape) {
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(StridedWrite, StridedWriteVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_INFER_DATA_SLICE(StridedWrite, StridedWriteInferDataSlice) {
  // strided write can cut n axis now
  OP_LOGD(op.GetName().c_str(), "Enter StridedWrite InferDataSlice");
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
  vector<vector<int64_t>> y_data_slice;
  vector<vector<int64_t>> x_data_slice = {{}, {}, {}, {}, {}};
  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op.GetName().c_str(), "no data slice, not need infer input");
    return GRAPH_FAILED;
  }

  int32_t stride = 1;
  if (GRAPH_SUCCESS != op.GetAttr("stride", stride)) {
    OP_LOGE(op.GetName().c_str(), "can not get attr of stride");
    return GRAPH_FAILED;
  }
  bool have_slice = false;
  for (int i = 0; i < y_data_slice.size(); i++) {
    if (y_data_slice[i].size() > 0) {
      have_slice = true;
      // c1 axis split tensor, ignore stride
      x_data_slice[i] = y_data_slice[i];
      OP_LOGD(op.GetName().c_str(), "Set y_data_slice is on No %d axis", i + 1);
    }
  }
  if (have_slice == false) {
    return GRAPH_FAILED;
  }
  if(!AttrUtils::SetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
    OP_LOGD(op.GetName().c_str(), "Set x_data slice failed!");
    return GRAPH_FAILED;
  }
  OP_LOGD(op.GetName().c_str(), "Calc StridedWrite InferDataSlice end!");
  return GRAPH_SUCCESS;
}

INFER_DATA_SLICE_FUNC_REG(StridedWrite, StridedWriteInferDataSlice);

COMMON_INFER_FUNC_REG(StridedWrite, StridedWriteInferShape);

VERIFY_FUNC_REG(StridedWrite, StridedWriteVerify);

// ----------------CumulativeLogsumexp-------------------
IMPLEMT_COMMON_INFERFUNC(CumulativeLogsumexpInferShape) {
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(op.GetInputDesc("x").GetShape());
  output_desc.SetDataType(op.GetInputDesc("x").GetDataType());
  op.UpdateOutputDesc("y", output_desc);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(CumulativeLogsumexp, CumulativeLogsumexpInferShape);
// ----------------CumulativeLogsumexp END-------------------

// ----------------CumulativeLogsumexpD-------------------
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
// ----------------CumulativeLogsumexpD END-------------------

// ----------------InplaceIndexAdd Begin-------------------
bool InferShapeAndTypeInplaceIndexAdd(Operator& op, const string& input_name1,
                                      const string& output_name) {
  TensorDesc v_output_desc = op.GetOutputDesc(output_name);

  DataType input_dtype = op.GetInputDesc(input_name1).GetDataType();
  Format input_format = op.GetInputDesc(input_name1).GetFormat();

  ge::Shape shape_x = op.GetInputDesc(input_name1).GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();

  ge::Shape output_shape = ge::Shape(dims_x);

  v_output_desc.SetShape(output_shape);
  v_output_desc.SetDataType(input_dtype);
  v_output_desc.SetFormat(input_format);
  op.UpdateOutputDesc(output_name, v_output_desc);

  return true;
}

IMPLEMT_VERIFIER(InplaceIndexAdd, InplaceIndexAddVerify) {
  DataType var_dtype = op.GetInputDesc("var").GetDataType();
  DataType indices_dtype = op.GetInputDesc("indices").GetDataType();
  DataType updates_dtype = op.GetInputDesc("updates").GetDataType();
  DataType var_out_dtype = op.GetInputDesc("var").GetDataType();
  if (var_dtype != var_out_dtype || var_dtype != updates_dtype) {
    OP_LOGE(op.GetName().c_str(),
            "var dtype is not equal to updates dtype, please check!");
    return GRAPH_FAILED;
  }
  if (indices_dtype != DT_INT32) {
    OP_LOGE(op.GetName().c_str(), "indices dtype is not int32, please check!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(InplaceIndexAddInferShape) {
  if (InferShapeAndTypeInplaceIndexAdd(op, "var", "var")) {
    return GRAPH_SUCCESS;
  }
  OP_LOGE(op.GetName().c_str(), "infer shape failed!");
  return GRAPH_FAILED;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(InplaceIndexAdd, InplaceIndexAddInferShape);
// Registered verify function
VERIFY_FUNC_REG(InplaceIndexAdd, InplaceIndexAddVerify);
// ----------------InplaceIndexAdd END---------------------

// ----------------MaskedFill Begin-------------------
IMPLEMT_COMMON_INFERFUNC(InferMaskedFillShape) {
  // ge::Operator op;
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x", "mask", "y", is_dynamic_output)){
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MaskedFill, MaskedFillVerify) {
  auto input_type_mask = op.GetInputDesc("mask").GetDataType();
  if (input_type_mask != DT_BOOL) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(MaskedFill, InferMaskedFillShape);
VERIFY_FUNC_REG(MaskedFill, MaskedFillVerify);
// ----------------MaskedFill END---------------------

// ----------------MaskedSelectV2 Begin-------------------
bool InferShapeAndTypeMaskedSelectV2(Operator& op, const string& input_name1,
                                     const string& input_name2,
                                     const string& output_name) {
  TensorDesc output_desc = op.GetOutputDesc(output_name);
  DataType input_dtype = op.GetInputDesc(input_name1).GetDataType();
  Format input_format = op.GetInputDesc(input_name1).GetFormat();
  ge::Shape shape_x = op.GetInputDesc(input_name1).GetShape();
  ge::Shape shape_y = op.GetInputDesc(input_name2).GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();
  std::vector<int64_t> dims_y = shape_x.GetDims();

  // The small shape is padded with 1.
  if (dims_x.size() != dims_y.size()) {
    int dec = dims_x.size() - dims_y.size();
    for (int i = 0; i < dec; i++) {
      dims_y.insert(dims_y.begin(), (int64_t)1);
    }
  }

  // The value of each dimension in the shape of the output tensor is the
  // larger value of the corresponding dimension in the two inputs.
  std::vector<int64_t> dim_vec;
  for (size_t i = 0; i < dims_x.size(); i++) {
    if ((dims_x[i] != dims_y[i]) && (dims_x[i] != 1) && (dims_y[i] != 1)) {
      OP_LOGE("The shape of x1 and x2 can not broadcast.");
      return false;
    }
    int64_t dims = (dims_x[i] > dims_y[i]) ? dims_x[i] : dims_y[i];
    dim_vec.push_back(dims);
  }

  ge::Shape output_shape = ge::Shape(dim_vec);
  output_desc.SetShape(output_shape);
  output_desc.SetDataType(input_dtype);
  output_desc.SetFormat(input_format);
  op.UpdateOutputDesc(output_name, output_desc);

  return true;
}

IMPLEMT_VERIFIER(MaskedSelectV2, MaskedSelectV2Verify) {
  if (op.GetInputDesc("x").GetDataType() !=
      op.GetOutputDesc("y").GetDataType()) {
    OP_LOGE("x y tensor dtype does not match.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(MaskedSelectV2InferShape) {
  if (InferShapeAndTypeMaskedSelectV2(op, "x", "mask", "y")) {
    return GRAPH_SUCCESS;
  }
  OP_LOGE("The shape of output y does not match that of x1 x2.");
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(MaskedSelectV2, MaskedSelectV2InferShape);
VERIFY_FUNC_REG(MaskedSelectV2, MaskedSelectV2Verify);
// ----------------MaskedSelectV2 END---------------------

// ----------------StridedSliceV2 Begin-------------------
struct SliceParametersFormal {
  std::vector<int64_t> begin_list;
  std::vector<int64_t> end_list;
  std::vector<int64_t> stride_list;
};

// Get relevant masks from const node
static graphStatus GetArgsStridedSliceInfer(const ge::Operator &op,
                                            struct SliceMasks &slice_masks) {
  map<string, uint64_t&> mask_maps = {
    {"begin_mask", slice_masks.begin_mask},
    {"end_mask", slice_masks.end_mask},
    {"ellipsis_mask", slice_masks.ellipsis_mask},
    {"new_axis_mask", slice_masks.new_axis_mask},
    {"shrink_axis_mask", slice_masks.shrink_axis_mask}
  };

  for (auto& item : mask_maps) {
    int64_t mask_value = 0;
    if (ge::GRAPH_SUCCESS != op.GetAttr(item.first, mask_value)) {
      OpsGetAttrErrReport(op.GetName(), item.first);
      OP_LOGE(op.GetName().c_str(), "Get attribute '%s' failed from op of StridedSlice!", item.first.c_str());
      return GRAPH_FAILED;
    }

    item.second = static_cast<uint64_t>(mask_value);
  }
  return GRAPH_SUCCESS;
}

static void GetBeginAndEndListPartOne1(const ge::Shape &shape,
                                       struct SliceMasks &slice_masks,
                                       struct SliceParameters &slice_params,
                                       const vector<int64_t> pow_table,
                                       const int64_t i) {
  size_t dim_num = shape.GetDimNum();
  size_t begin_len = slice_params.begin_list.size();
  int64_t pow_val = 0;
  pow_val = pow_table[i];
  if ((slice_masks.ellipsis_mask & pow_val) == pow_val) {
    size_t ellipsis_dim = i;
    slice_params.begin_list[i] = 0;
    slice_params.end_list[i] = shape.GetDim(i);
    slice_params.stride_list[i] = 1;
    if ((slice_masks.shrink_axis_mask & pow_val) == pow_val) {
      slice_masks.shrink_axis_mask -= pow_val;
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

static void GetBeginAndEndListPartOne2(const ge::Shape &shape,
                                       struct SliceMasks &slice_masks,
                                       struct SliceParameters &slice_params,
                                       const vector<int64_t> pow_table,
                                       const int64_t i) {
  size_t dim_num = shape.GetDimNum();
  size_t begin_len = slice_params.begin_list.size();
  int64_t pow_val = 0;
  pow_val = pow_table[i];
  if ((slice_masks.ellipsis_mask & pow_val) == pow_val) {
    size_t ellipsis_dim = i;
    slice_params.begin_list[i] = 0;
    slice_params.end_list[i] = shape.GetDim(i);
    slice_params.stride_list[i] = 1;
    if ((slice_masks.shrink_axis_mask & pow_val) == pow_val) {
      slice_masks.shrink_axis_mask -= pow_val;
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

static void GetBeginAndEndListPartOne3(const ge::Shape &shape,
                                       struct SliceMasks &slice_masks,
                                       struct SliceParameters &slice_params,
                                       const vector<int64_t> pow_table) {
  size_t dim_num = shape.GetDimNum();

  auto clamp = [](int64_t x, int64_t l, int64_t h) {
    return (x < l) ? l : (x > h) ? h : x;
  };

  for (size_t i = 0; i < dim_num; i++) {
    int64_t stride_i = slice_params.stride_list[i];
    int64_t dim_i = shape.GetDim(i);

    if (slice_params.begin_list[i] < 0) {
      slice_params.begin_list[i] = dim_i + slice_params.begin_list[i];
    }
    if (slice_params.end_list[i] < 0) {
      slice_params.end_list[i] = dim_i + slice_params.end_list[i];
    }

    if (stride_i < 0) {
      slice_params.begin_list[i] =
          clamp(slice_params.begin_list[i], 0, dim_i - 1);
      slice_params.end_list[i] = clamp(slice_params.end_list[i], -1, dim_i);
    } else {
      slice_params.begin_list[i] = clamp(slice_params.begin_list[i], 0, dim_i);
      slice_params.end_list[i] = clamp(slice_params.end_list[i], 0, dim_i);
    }
  }

  int64_t pow_val = 0;
  for (size_t i = 0; i < dim_num; i++) {
    pow_val = pow_table[i];
    if ((slice_masks.begin_mask & pow_val) == pow_val) {
      if (slice_params.stride_list[i] > 0) {
        slice_params.begin_list[i] = 0;
      }
      if (slice_params.stride_list[i] < 0) {
        slice_params.begin_list[i] = slice_params.input[i];
      }
    }

    if ((slice_masks.end_mask & pow_val) == pow_val) {
      if (slice_params.stride_list[i] > 0) {
        slice_params.end_list[i] = slice_params.input[i];
      }
      if (slice_params.stride_list[i] < 0) {
        slice_params.end_list[i] = 0;
      }
    }
    if ((slice_masks.ellipsis_mask & pow_val) == pow_val) {
      slice_params.begin_list[i] = 0;
      slice_params.end_list[i] = shape.GetDim(i);
      slice_params.stride_list[i] = 1;
    }
  }
}

static void GetBeginAndEndListPartOne4(const ge::Shape &shape,
                                       struct SliceMasks &slice_masks,
                                       struct SliceParameters &slice_params,
                                       const vector<int64_t> pow_table) {
  size_t dim_num = shape.GetDimNum();
  size_t begin_len = slice_params.begin_list.size();
  int64_t pow_val = 0;

  if (slice_masks.new_axis_mask != 0) {
    for (size_t i = 0; i < dim_num; i++) {
      pow_val = pow_table[i];
      if ((slice_masks.new_axis_mask & pow_val) == pow_val) {
        slice_params.begin_list[i] = 0;
        slice_params.end_list[i] = 1;
        slice_params.stride_list[i] = 1;
        slice_masks.shrink_axis_mask =
            ((slice_masks.shrink_axis_mask & pow_val) == pow_val)
                ? (slice_masks.shrink_axis_mask - pow_val)
                : slice_masks.shrink_axis_mask;
      }
    }
  }

  size_t tmp_shrink = 0;
  if (slice_masks.shrink_axis_mask != 0) {
    for (size_t i = 0; i < dim_num; i++) {
      pow_val = pow_table[i];
      if ((slice_masks.shrink_axis_mask & pow_val) == pow_val) {
        tmp_shrink = (begin_len > i) ? (tmp_shrink + pow_val) : tmp_shrink;
      }
    }
    slice_masks.shrink_axis_mask = tmp_shrink;
  }

  size_t new_begin_mask = 0;
  size_t new_end_mask = 0;
  size_t new_shrink_n_mask = 0;
  size_t new_new_axis_mask = 0;
  // compute the right_move of begin end stride and masks
  // because of non-zero ellipsis_mask
  size_t right_move = std::max<int64_t>(dim_num - begin_len, 0);
  if (slice_masks.ellipsis_mask != 0) {
    size_t bit_ellipsis = static_cast<int64_t>(log2(slice_masks.ellipsis_mask));
    for (size_t i = 0; i < dim_num; i++) {
      ELLIPSIS_MASK_UPDATE(slice_masks.begin_mask, new_begin_mask, bit_ellipsis,
                           i, pow_table, right_move);
      ELLIPSIS_MASK_UPDATE(slice_masks.end_mask, new_end_mask, bit_ellipsis, i,
                           pow_table, right_move);
      ELLIPSIS_MASK_UPDATE(slice_masks.shrink_axis_mask, new_shrink_n_mask,
                           bit_ellipsis, i, pow_table, right_move);
      ELLIPSIS_MASK_UPDATE(slice_masks.new_axis_mask, new_new_axis_mask,
                           bit_ellipsis, i, pow_table, right_move);
    }
    slice_masks.begin_mask = new_begin_mask;
    slice_masks.end_mask = new_end_mask;
    slice_masks.shrink_axis_mask = new_shrink_n_mask;
    slice_masks.new_axis_mask = new_new_axis_mask;
  }
}

static void GetBeginAndend_listInferPart1(const ge::Shape &shape,
                                          struct SliceMasks &slice_masks,
                                          struct SliceParameters &slice_params,
                                          const vector<int64_t> pow_table) {
  size_t dim_num = shape.GetDimNum();
  size_t begin_len = slice_params.begin_list.size();
  slice_params.input = shape.GetDims();
  if (dim_num < begin_len && slice_masks.new_axis_mask != 0) {
    dim_num = begin_len;
  }

  // rebuild the begin end stride of new_axis,
  // because ignored when new_axis is true.
  int64_t pow_val = 0;
  GetBeginAndEndListPartOne4(shape, slice_masks, slice_params, pow_table);

  for (size_t i = 0; i < dim_num; i++) {
    pow_val = pow_table[i];
    if ((slice_masks.new_axis_mask & pow_val) == pow_val) {
      slice_params.input.insert(slice_params.input.begin() + i, 1);
    }
  }

  size_t bit_ellipsis = static_cast<int64_t>(log2(slice_masks.ellipsis_mask));
  if (slice_masks.ellipsis_mask != 0 && bit_ellipsis > begin_len - 1) {
    if (begin_len < dim_num) {
      for (size_t i = 0; i < dim_num - begin_len; i++) {
        slice_params.begin_list.push_back(0);
        slice_params.end_list.push_back(slice_params.input[begin_len + i]);
        slice_params.stride_list.push_back(1);
        begin_len += 1;
      }
    }
    if (slice_masks.ellipsis_mask != 0) {
      for (size_t i = 0; i < dim_num; i++) {
        GetBeginAndEndListPartOne1(shape, slice_masks, slice_params, pow_table,
                                   i);
      }
    }
  } else {
    if (slice_masks.ellipsis_mask != 0) {
      for (size_t i = 0; i < dim_num; i++) {
        GetBeginAndEndListPartOne2(shape, slice_masks, slice_params, pow_table,
                                   i);
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

  GetBeginAndEndListPartOne3(shape, slice_masks, slice_params, pow_table);
}

static void GetBeginAndend_listInferPart2(const ge::Shape &shape,
                                          struct SliceMasks &slice_masks,
                                          struct SliceParameters &slice_params,
                                          const vector<int64_t> pow_table) {
  size_t dim_num = shape.GetDimNum();
  size_t new_axis_flag = 0;

  for (size_t i = 0; i < dim_num; i++) {
    if ((slice_masks.new_axis_mask & pow_table[i]) == pow_table[i]) {
      new_axis_flag += 1;
    }
  }

  for (size_t i = 0; i < slice_params.input.size(); i++) {
    if ((slice_masks.shrink_axis_mask & pow_table[i]) == pow_table[i]) {
      slice_params.end_list[i] = slice_params.begin_list[i] + 1;
    }
  }
}

static graphStatus GetStridedSliceInfer(
    const ge::Operator &op, struct SliceParameters &slice_params,
    struct SliceParametersFormal &slice_params_formal) {
  Tensor begin_tensor;
  Tensor end_tensor;
  Tensor stride_tensor;
  vector<int64_t> begin_list;
  vector<int64_t> end_list;
  vector<int64_t> stride_list;

  // required input
  if (op.GetInputConstData("begin", begin_tensor) != GRAPH_SUCCESS) {
    OP_LOGW(op.GetName().c_str(), "Get constValue failed of [begin]");
    return GRAPH_FAILED;
  }
  if (op.GetInputConstData("end", end_tensor) != GRAPH_SUCCESS) {
    OP_LOGW(op.GetName().c_str(), "Get constValue failed of [end]");
    return GRAPH_FAILED;
  }
  DataType dtype_begin = op.GetInputDesc("begin").GetDataType();
  DataType dtype_end = op.GetInputDesc("end").GetDataType();
  GetConstValue(op, begin_tensor, dtype_begin, begin_list);
  GetConstValue(op, end_tensor, dtype_end, end_list);

  ge::OpDescPtr slice_desc = OpDescUtils::GetOpDescFromOperator(op);
  if (slice_desc->MutableInputDesc(
          slice_desc->GetInputIndexByName("strides")) != nullptr) {
    if (op.GetInputConstData("strides", stride_tensor) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "Get constValue failed of [strides]");
      return GRAPH_FAILED;
    }
    DataType dtype_stride = op.GetInputDesc("strides").GetDataType();
    GetConstValue(op, stride_tensor, dtype_stride, stride_list);
  } else {
    OP_LOGW(op.GetName().c_str(), "Setting default strides");
    // optional input
    for (size_t i = 0; i < begin_list.size(); i++) {
      stride_list.push_back(1);
    }
  }

  slice_params.begin_list = begin_list;
  slice_params_formal.begin_list = begin_list;

  slice_params.end_list = end_list;
  slice_params_formal.end_list = end_list;

  slice_params.stride_list = stride_list;
  slice_params_formal.stride_list = stride_list;

  return GRAPH_SUCCESS;
}

static graphStatus UpdateParams1(
    const ge::Operator op, struct SliceParameters &slice_params_output,
    struct SliceParametersFormal &slice_params_output_formal) {
  ge::Shape shape = op.GetInputDesc("x").GetShape();
  size_t dim_num = shape.GetDimNum();
  std::vector<std::string> input_infer_depends = {"begin", "end"};
  ge::OpDescPtr slice_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto strides_desc = slice_desc->MutableInputDesc(slice_desc->GetInputIndexByName("strides"));
  if (strides_desc != nullptr) {
    input_infer_depends.push_back("strides");
  }

  auto axes_desc = slice_desc->MutableInputDesc(slice_desc->GetInputIndexByName("axes"));
  if (axes_desc != nullptr) {
    input_infer_depends.push_back("axes");
  }
  slice_desc->SetOpInferDepends(input_infer_depends);

  if (GRAPH_FAILED == GetStridedSliceInfer(op, slice_params_output,
                                           slice_params_output_formal)) {
    OP_LOGW(op.GetName().c_str(),
            "Get constValue failed of [begin,end,[stride]]");
    return GRAPH_SUCCESS;
  }

  // get axes and permute others attr
  Tensor axes_tensor;
  vector<int64_t> axes_list;
  if (axes_desc != nullptr) {
    if (op.GetInputConstData("axes", axes_tensor) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "Get constValue failed of [axes]");
      return GRAPH_FAILED;
    }
    DataType dtype = op.GetInputDesc("axes").GetDataType();
    GetConstValue(op, axes_tensor, dtype, axes_list);

    std::vector<int64_t> new_begins(dim_num, 0);
    std::vector<int64_t> new_ends(dim_num, 0);
    std::vector<int64_t> new_axes(dim_num, dim_num);
    std::vector<int64_t> new_strides(dim_num, 1);
    // begin_list  end_list  stride_list
    int64_t indice = 0;
    for (int32_t i = 0; i < axes_list.size(); i++) {
      indice = (axes_list[i] < 0) ? (axes_list[i] + dim_num) : axes_list[i];
      if (indice < 0) {
        indice = 0;
      } else if (indice > dim_num - 1) {
        indice = dim_num - 1;
      }
      new_axes[indice] = axes_list[i];
      new_begins[indice] = slice_params_output.begin_list[i];
      new_ends[indice] = slice_params_output.end_list[i];
      new_strides[indice] = slice_params_output.stride_list[i];
    }

    std::vector<int64_t> input_shape = shape.GetDims();
    for (int32_t i = 0; i < new_axes.size(); i++) {
      if (new_axes[i] == dim_num) {
        new_ends[i] = input_shape[i];
      }
      if (new_ends[i] > input_shape[i]) {
        new_ends[i] = input_shape[i];
      }
    }

    slice_params_output.begin_list = new_begins;
    slice_params_output.end_list = new_ends;
    slice_params_output.stride_list = new_strides;
  }

  return GRAPH_SUCCESS;
}

static void UpdateParams2(struct SliceMasks slice_masks_output,
                          struct SliceParameters &slice_params_output,
                          std::vector<int64_t> &output_list,
                          std::vector<int64_t> &output_shape_list,
                          const size_t dim_num) {
  size_t base_number = 2.0;
  std::vector<int64_t> pow_table(dim_num, 0);
  for (size_t i = 0; i < dim_num; i++)
    pow_table[i] = static_cast<int64_t>(pow(base_number, i));

  size_t shrink_axis_maskTemp = 0;
  if (slice_masks_output.shrink_axis_mask != 0) {
    for (size_t i = 0; i < dim_num; ++i) {
      if ((slice_params_output.end_list[i] -
           slice_params_output.begin_list[i]) == 0)
        shrink_axis_maskTemp += pow_table[i];
    }
  }
  slice_masks_output.shrink_axis_mask =
      slice_masks_output.shrink_axis_mask | shrink_axis_maskTemp;
  // Convert the target data into a double type by multiply '1.0'
  double change_to_double = 1.0;
  for (size_t i = 0; i < slice_params_output.begin_list.size(); ++i) {
    size_t dim = (int64_t)(ceil(
        (slice_params_output.end_list[i] - slice_params_output.begin_list[i]) /
        (slice_params_output.stride_list[i] * change_to_double)));
    dim = std::max<int64_t>(dim, int64_t(0));
    if (((slice_masks_output.shrink_axis_mask & pow_table[i]) !=
         pow_table[i]) ||
        ((slice_masks_output.new_axis_mask & pow_table[i]) != pow_table[i])) {
      // get outputshape
      output_shape_list.push_back(dim);
      if (dim != 1)
        // get real dim cnt
        output_list.push_back(dim);
    }
  }
  if (slice_masks_output.shrink_axis_mask == 0 &&
      slice_masks_output.new_axis_mask == 0) {
    if (slice_params_output.begin_list.size() >
        slice_params_output.input.size()) {
      for (size_t i = 0; i < slice_params_output.begin_list.size() -
                                 slice_params_output.input.size();
           i++) {
        output_shape_list.erase(output_shape_list.begin() + i +
                                slice_params_output.input.size());
      }
    }
  }
  // shrink_axis_mask != 0
  if (slice_masks_output.shrink_axis_mask > 0) {
    size_t shrink_flag = 0;
    for (size_t i = 0; i < dim_num; i++) {
      if ((slice_masks_output.shrink_axis_mask & pow_table[i]) ==
          pow_table[i]) {
        output_shape_list.erase(output_shape_list.begin() + i - shrink_flag);
        shrink_flag += 1;
      }
    }
  }

  if (output_list.size() == 0) {
    output_list.push_back(1);
  }
}

// ----------------StridedSliceV2 Op Begin-----------------
IMPLEMT_COMMON_INFERFUNC(StridedSliceV2InferShape) {
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
  struct SliceParametersFormal slice_params_output_formal = {};

  if (UpdateParams1(op, slice_params_output, slice_params_output_formal) !=
      GRAPH_SUCCESS) {
    TensorDesc output_desc = op.GetOutputDesc("y");
    output_desc.SetShape(ge::Shape(ge::UNKNOWN_RANK));
    return op.UpdateOutputDesc("y", output_desc);
  }

  op.SetAttr("begin", slice_params_output.begin_list);
  op.SetAttr("end", slice_params_output.end_list);
  op.SetAttr("strides", slice_params_output.stride_list);

  // Get relevant masks from const node
  struct SliceMasks slice_masks_output = {};
  if (GRAPH_FAILED == GetArgsStridedSliceInfer(op, slice_masks_output)) {
    return GRAPH_FAILED;
  }

  int64_t ellipsis_dim = 0;
  if (slice_masks_output.ellipsis_mask != 0) {
    for (size_t i = 0; i < dim_num; ++i) {
      if ((slice_masks_output.ellipsis_mask & ((uint64_t)pow(2.0, i))) ==
          ((uint64_t)pow(2.0, i))) {
        ellipsis_dim += 1;
      }
    }
    if (ellipsis_dim > 1) {
      OP_LOGE(op.GetName().c_str(), "only suppot 1 dim of ellipsis!");
      return GRAPH_FAILED;
    }
  }

  size_t base_number = 2.0;
  std::vector<int64_t> pow_table(dim_num, 0);
  for (size_t i = 0; i < dim_num; i++) {
    pow_table[i] = static_cast<int64_t>(pow(base_number, i));
  }

  // Deal with 'begin_list' and 'end_list' by corresponding mask
  GetBeginAndend_listInferPart1(shape, slice_masks_output, slice_params_output,
                                pow_table);
  GetBeginAndend_listInferPart2(shape, slice_masks_output, slice_params_output,
                                pow_table);

  std::vector<int64_t> output_list;
  std::vector<int64_t> output_shape_list;

  UpdateParams2(slice_masks_output, slice_params_output, output_list,
                output_shape_list, dim_num);

  ge::Shape output_shape = ge::Shape(output_shape_list);
  TensorDesc tensor_desc_output = op.GetOutputDesc("y");
  tensor_desc_output.SetShape(output_shape);
  tensor_desc_output.SetDataType(input_dtype);
  tensor_desc_output.SetRealDimCnt(output_list.size());
  (void)op.UpdateOutputDesc("y", tensor_desc_output);

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(StridedSliceV2, StridedSliceV2Verify) { return GRAPH_SUCCESS; }

INFER_FUNC_REG(StridedSliceV2, StridedSliceV2InferShape);
VERIFY_FUNC_REG(StridedSliceV2, StridedSliceV2Verify);
// ----------------StridedSliceV2 Op End-----------------
// ----------------StridedSliceV2 END---------------------

// ----------------SliceLastDim Begin-------------------
bool InferShapeAndTypeSliceLastDim(Operator& op, const string& x_name,
                                   const string& output_name) {
  TensorDesc v_output_desc = op.GetOutputDesc(output_name);

  DataType input_dtype = op.GetInputDesc(x_name).GetDataType();
  Format input_format = op.GetInputDesc(x_name).GetFormat();

  ge::Shape shape_x = op.GetInputDesc(x_name).GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();

  std::vector<int64_t> dims_output;
  int64_t begin, end, stride, length;
  if (GRAPH_SUCCESS != op.GetAttr("start", begin)) {
    OP_LOGE(op.GetName().c_str(), "Failed to get the length");
    return false;
  }

  if (GRAPH_SUCCESS != op.GetAttr("end", end)) {
    OP_LOGE(op.GetName().c_str(), "Failed to get the length");
    return false;
  }

  if (GRAPH_SUCCESS != op.GetAttr("stride", stride)) {
    OP_LOGE(op.GetName().c_str(), "Failed to get the length");
    return false;
  }
  if (stride == 0) {
    OP_LOGE(op.GetName().c_str(), "stride[%d] error.", stride);
    return GRAPH_FAILED;
  }
  length = (end - begin - 1 + stride) / stride;

  for (size_t i = 0; i < dims_x.size(); i++) {
    int64_t dims = (i == dims_x.size() - 2)
                       ? length
                       : dims_x[i];  // judge if the last second dim
    dims_output.push_back(dims);
  }
  ge::Shape output_shape = ge::Shape(dims_output);

  v_output_desc.SetShape(output_shape);
  v_output_desc.SetDataType(input_dtype);
  v_output_desc.SetFormat(input_format);
  op.UpdateOutputDesc(output_name, v_output_desc);

  return true;
}

IMPLEMT_VERIFIER(SliceLastDim, SliceLastDimVerify) {
  DataType x_dtype = op.GetInputDesc("x").GetDataType();
  DataType out_dtype = op.GetInputDesc("y").GetDataType();

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(SliceLastDimInferShape) {
  if (InferShapeAndTypeSliceLastDim(op, "x", "y")) {
    return GRAPH_SUCCESS;
  }
  OP_LOGE(op.GetName().c_str(), "infer shape failed!");
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(SliceLastDim, SliceLastDimInferShape);
VERIFY_FUNC_REG(SliceLastDim, SliceLastDimVerify);
// ----------------SliceLastDim END---------------------

// ----------------IndexFillD-------------------
bool InferShapeAndTypeIndexFillD(Operator &op, const string &input_name, const string &output_name)
{
    TensorDesc v_output_desc = op.GetOutputDesc(output_name);
    DataType input_dtype = op.GetInputDesc(input_name).GetDataType();
    Format input_format = op.GetInputDesc(input_name).GetFormat();

    //shape of output y is the same as input x
    ge::Shape shape_input = op.GetInputDesc(input_name).GetShape();
    v_output_desc.SetShape(shape_input);
    v_output_desc.SetDataType(input_dtype);
    v_output_desc.SetFormat(input_format);
    op.UpdateOutputDesc(output_name, v_output_desc);

    return true;
}

IMPLEMT_VERIFIER(IndexFillD, IndexFillDVerify)
{
    // check whether the dtype of x and assist1 is the same
    if (op.GetInputDesc("x").GetDataType() != op.GetInputDesc("assist1").GetDataType()
        || op.GetInputDesc("x").GetDataType() != op.GetInputDesc("assist2").GetDataType())
    {
	OP_LOGE(op.GetName().c_str(), "Input dtypes are not the same.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(IndexFillDInferShape)
{
    InferShapeAndTypeIndexFillD(op, "x", "y");
    return GRAPH_SUCCESS;
}

//Registered inferfunction
COMMON_INFER_FUNC_REG(IndexFillD, IndexFillDInferShape);

//Registered verify function
VERIFY_FUNC_REG(IndexFillD, IndexFillDVerify);
//----------------IndexFillD END-------------------

}  // namespace ge
