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
#include "op_const.h"
#include "strided_slice_infer_shape.h"
#include "graph/utils/op_desc_utils.h"
#include "register/infer_data_slice_registry.h"
#include "graph/common_error_codes.h"
#include "graph/debug/ge_attr_define.h"
#include "axis_util.h"
#include "common_shape_fns.h"
#include "util/vector_proto_profiling.h"

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
    std::string err_msg = GetInputInvalidErrMsg(ConcatString(keyName));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    string correct_size = ConcatString("more than zero and less than eight!");
    std::string err_msg = GetAttrSizeErrMsg("begin", ConcatString(begin.size()), correct_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<int32_t> end;
  if (GRAPH_FAILED == GetStridedSliceGradValue(op, "end", end)) {
    return GRAPH_FAILED;
  }
  if (end.size() < 0 || end.size() > 8) {
    string correct_size = ConcatString("more than zero and less than eight!");
    std::string err_msg = GetAttrSizeErrMsg("end", ConcatString(end.size()), correct_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<int32_t> strides;
  if (GRAPH_FAILED == GetStridedSliceGradValue(op, "strides", strides)) {
    return GRAPH_FAILED;
  }
  if (strides.size() < 0 || strides.size() > 8) {
    string correct_size = ConcatString("more than zero and less than eight!");
    std::string err_msg = GetAttrSizeErrMsg("strides", ConcatString(strides.size()), correct_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (dimNum >= 1 && dimNum <= 8) {
    for (size_t i = 0; i < dimNum; i++) {
      outputShape.SetDim(i, outputShapeList[i]);
    }
  } else {
    std::string err_msg = OtherErrMsg("The StridedSliceGrad dimension of the input shape is limited to 1 or 8.");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
      std::string err_msg = GetInputInvalidErrMsg(ConcatString(item.first));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = OtherErrMsg("The dims of input is empty!");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
      for (int64_t  dim = 0; dim < begin_len; dim++) {
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
    // value of "shape" while "shape" is variable, set range as (0,-1).
    size_t dim_num = dim_vector[0];
    OP_LOGD(op.GetName().c_str(), "dim_num is %d.", dim_num);
    for (size_t dim = 0; dim < dim_num; dim++) {
      outputShapeList.push_back(-1);
      out_range.push_back(std::make_pair(0, -1));
    }
    TensorDesc tensordesc_output = op.GetOutputDesc("output");
    ge::Shape out_shape = ge::Shape(outputShapeList);
    tensordesc_output.SetShape(out_shape);
    tensordesc_output.SetDataType(input_dtype);
    tensordesc_output.SetShapeRange(out_range);
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
    if (input_shape.GetDim(0) >= 0 && (!multiples.empty())) {
      OP_LOGI(op.GetName().c_str(), "Get into align_input len 1 and input shape >= 0.");
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
      return GRAPH_FAILED;
    }

  } else if (input_len <= 8 && input_len >= 2) {
    for (uint64_t i = 0; i < input_len; i++) {
      if (input_shape.GetDim(i) >= 0) {
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
    std::string err_msg = GetInputInvalidErrMsg("multiples");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
  }
  return multiples;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(TileDInferShape) {
  std::vector<int64_t> multiples;
  multiples = GetTileDConstValue(op, "multiples");
  if (multiples.empty()) {
    std::string err_msg = OtherErrMsg("op tile_d get attr multiples value is empty!");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    OP_LOGI(op.GetName().c_str(), "Get constValue failed of in input[start], input[delta], input[limit]");
    dimsIn.emplace_back(UNKNOWN_DIM);
    y_output->SetShape(GeShape(dimsIn));
    y_output->SetOriginShape(GeShape(dimsIn));
    y_output->SetShapeRange({std::make_pair(1, -1)});
    DataType start_datatype = start_desc->GetDataType();
    DataType limit_datatype = limit_desc->GetDataType();
    DataType delta_datatype = delta_desc->GetDataType();
    if (start_datatype == ge::DT_INT32 && limit_datatype == ge::DT_INT32 && delta_datatype == ge::DT_INT32) {
      y_output->SetDataType(ge::DT_INT32);
    } else if (start_datatype == ge::DT_INT64 && limit_datatype == ge::DT_INT64 && delta_datatype == ge::DT_INT64) {
      y_output->SetDataType(ge::DT_INT64);
    } else if (start_datatype == ge::DT_DOUBLE && limit_datatype == ge::DT_DOUBLE && delta_datatype == ge::DT_DOUBLE) {
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
              "the start_multiples_size is [%d], the limit_multiples_size is [%d],"
              "the delta_multiples_size is [%d]",
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
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), string("the value of input[delta] should not be zero"));
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
    OP_LOGD(op.GetName().c_str(), "output shape: [%s].", to_string(dimsIn).c_str());
    return GRAPH_SUCCESS;
  }
}

COMMON_INFER_FUNC_REG(Range, RangeInferShape);
// -----------------------Range Op End----------------------------------

// -----------------------RangeD Op Begin----------------------------------
IMPLEMT_COMMON_INFERFUNC(RangeDInferShape) {
  float start;
  if (op.GetAttr("start", start) == ge::GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("start");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  float limit;
  if (op.GetAttr("limit", limit) == ge::GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("limit");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  float delta;
  if (op.GetAttr("delta", delta) == ge::GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("delta");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (limit == start) {
    string excepted_value = ConcatString("not equal to limit[", limit, "]");
    std::string err_msg = GetAttrValueErrMsg("limit", ConcatString("limit"), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (delta == 0) {
    string excepted_value = ConcatString("not equal to 0");
    std::string err_msg = GetAttrValueErrMsg("delta", ConcatString("delta"), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (start > limit && delta > 0) {
    string excepted_value = ConcatString("more than start[", start, "] when delta is more than zero");
    std::string err_msg = GetAttrValueErrMsg("limit", ConcatString("limit"), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (start < limit && delta < 0) {
    string excepted_value = ConcatString("more than start[", limit, "] when delta is less than zero");
    std::string err_msg = GetAttrValueErrMsg("start", ConcatString("start"), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
static graphStatus GatherV2InferOptimize(ge::Operator &op, int64_t &axis, GeTensorDescPtr &x_desc,
                                         GeTensorDescPtr &indices_desc, GeTensorDescPtr &y_desc,
                                         std::vector <int64_t> &x_shape, std::vector <int64_t> &indices_shape,
                                         std::vector <int64_t> &y_shape,
                                         std::vector <std::pair<int64_t, int64_t>> &shape_range_x,
                                         std::vector <std::pair<int64_t, int64_t>> &shape_range_indices,
                                         std::vector <std::pair<int64_t, int64_t>> &out_range,
                                         int64_t batch_dims) {
  // real dim cnt has no existing meaning .Original shape has replace its meaning now
  int64_t x_real_dim_cnt = static_cast<int64_t>(x_desc->GetOriginShape().GetDims().size());

  if (IsUnknownRankShape(indices_shape) || IsUnknownRankShape(x_shape)) {
    y_shape.push_back(-2);

    y_desc->SetShape(ge::GeShape(y_shape));
    y_desc->SetDataType(x_desc->GetDataType());

    return GRAPH_SUCCESS;
  }

  if (x_real_dim_cnt < 1) {
    std::string err_msg = GetAttrValueErrMsg("x_real_dim_cnt", std::to_string(x_real_dim_cnt),
                                             ConcatString("x_real_dim_cnt >= 1"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (axis < 0) {
    if (x_real_dim_cnt < -axis) {
      std::string err_msg = OtherErrMsg(ConcatString("x_desc RealDimCnt[", x_real_dim_cnt,
                                                     "] < -axis[", -axis, "]"));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  } else if (x_real_dim_cnt < axis + 1) {
    std::string err_msg = OtherErrMsg(
        ConcatString("x_desc RealDimCnt[", x_real_dim_cnt, "] < axis + 1[", axis + 1, "]"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (axis < 0) {
    axis = x_real_dim_cnt + axis;
    if (batch_dims > axis) {
      std::string err_msg = OtherErrMsg(ConcatString("batch_dims must be less than or equal to axis", axis));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }

  for (int i = 0; i < axis; i++) {
    y_shape.push_back(x_shape[i]);
    if ((size_t) i < shape_range_x.size()) {
      out_range.push_back(shape_range_x[i]);
    }
  }
  // real dim cnt has no existing meaning .Original shape has replace its meaning now
  auto indices_dim_cnt_unsigned = indices_desc->GetOriginShape().GetDims().size();
  for (size_t i = batch_dims; i < indices_dim_cnt_unsigned; i++) {
    y_shape.push_back(indices_shape[i]);
    if ((size_t) i < shape_range_indices.size()) {
      out_range.push_back(shape_range_indices[i]);
    }
  }

  for (int i = axis + 1; i < x_real_dim_cnt; i++) {
    y_shape.push_back(x_shape[i]);
    if ((size_t) i < shape_range_x.size()) {
      out_range.push_back(shape_range_x[i]);
    }
  }

  y_desc->SetShape(ge::GeShape(y_shape));
  y_desc->SetShapeRange(out_range);
  y_desc->SetDataType(x_desc->GetDataType());
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(GatherV2InferShape) {
    auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
    vector<string> input_infer_depends = { "axis" };
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
        axis = (int64_t)(*((int64_t *) axis_tensor.GetData()));
      } else {
        axis = (int32_t)(*((int32_t *) axis_tensor.GetData()));
      }
    }

    int64_t rank_x = static_cast<int64_t>(x_desc->GetOriginShape().GetDims().size());
    int64_t rank_indices = static_cast<int64_t>(indices_desc->GetOriginShape().GetDims().size());
    int64_t batch_dims = 0;
    if (ge::GRAPH_SUCCESS != static_cast<int64_t>(op.GetAttr("batch_dims", batch_dims))) {
      batch_dims = 0;
      OP_LOGW(op.GetName().c_str(), "GetAttr(batch_dims) failed, set default value to 0.");
    }
    if (batch_dims < -rank_indices || (batch_dims >= rank_indices && rank_indices != 0)) {
      std::string err_msg = OtherErrMsg(ConcatString("Expected batch_dims in the range [", -rank_indices, ",",
                                                     rank_indices, "), but got", batch_dims));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    if (batch_dims < 0) {
      batch_dims = batch_dims + rank_indices;
    }
    if (batch_dims >= rank_x) {
      std::string err_msg = OtherErrMsg(ConcatString("batch_dims must be less than rank(params)", rank_x));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    for (int i = 0; i < batch_dims; i++) {
      if (x_shape[i] != indices_shape[i] && x_shape[i] > 0 && indices_shape[i] > 0) {
        std::string err_msg = OtherErrMsg(ConcatString("params.shape[", i, "]:", x_shape[i],
                                                       "should be equal to indices.shape[", i, "]:",
                                                       indices_shape[i]));
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
        return GRAPH_FAILED;
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

      // infer shape range
      std::vector <std::pair<int64_t, int64_t>> range_tmp = shape_range_x;
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

      std::pair <int64_t, int64_t> rank_unkown(1, -1);
      int count_rank_x = std::count(shape_range_x.begin(), shape_range_x.end(), rank_unkown);
      int count_rank_indices = std::count(shape_range_indices.begin(), shape_range_indices.end(), rank_unkown);

      if (batch_dims != 0) {
        rank_indices = rank_indices - batch_dims;
      }
      for (int i = 0; i < rank_x + rank_indices - 1; i++) {
        y_shape.push_back(-1);
        if (count_rank_x > 0 || count_rank_indices > 0) {
          out_range.push_back(std::pair<int64_t, int64_t>(0, -1));
        } else {
          out_range.push_back(std::pair<int64_t, int64_t>(min_first, max_second));
        }
      }

      y_desc->SetDataType(x_desc->GetDataType());
      y_desc->SetShapeRange(out_range);
      y_desc->SetShape(ge::GeShape(y_shape));
    } else {
      if (GatherV2InferOptimize(op, axis, x_desc, indices_desc, y_desc, x_shape, indices_shape, y_shape, shape_range_x,
                                shape_range_indices, out_range, batch_dims) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
      }
    }

    return GRAPH_SUCCESS;
}
IMPLEMT_INFER_DATA_SLICE(GatherV2, GatherV2InferDataSlice) {
  OP_LOGD(TbeGetName(op), "Enter GatherV2InferDataSlice.");
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  // get indice shape and format
  auto indice_desc = op_info->MutableInputDesc("indices");
  // get y desc
  auto y_desc = op_info->MutableOutputDesc("y");
  // get x desc
  auto x_desc = op_info->MutableInputDesc("x");
  size_t x_dimnum = x_desc->MutableShape().GetDimNum();
  vector<vector<int64_t>> y_data_slice;
  vector<vector<int64_t>> x_data_slice(x_dimnum);
  if (!ge::AttrUtils::GetListListInt(y_desc, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGE(TbeGetName(op), "no data slice, use default as.");
    return GRAPH_FAILED;
  }
  if (!ge::AttrUtils::SetListListInt(indice_desc, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    return GRAPH_FAILED;
  }
  if (!ge::AttrUtils::SetListListInt(x_desc, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(GatherV2, GatherV2InferShape);
INFER_VALUE_RANGE_DEFAULT_REG(GatherV2);
INFER_DATA_SLICE_FUNC_REG(GatherV2, GatherV2InferDataSlice);
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
    std::string err_msg = OtherErrMsg(ConcatString("x_desc RealDimCnt[",x_real_dim_cnt,"] not support."));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  auto x_shape = x_desc->GetShape().GetDims();
  if (axis < 0) {
    if (x_real_dim_cnt < -axis) {
      std::string err_msg = OtherErrMsg(ConcatString("x_desc RealDimCnt[",x_real_dim_cnt,"] < -axis[",-axis,"]"));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  } else if (x_real_dim_cnt < axis + 1) {
    std::string err_msg = OtherErrMsg(ConcatString("x_desc RealDimCnt[",x_real_dim_cnt,"] < axis + 1[",axis + 1,"]"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int64_t end = axis;
  if (end < 0) {
    end = x_real_dim_cnt + end;
    if (end < 0) {
      std::string err_msg = OtherErrMsg(ConcatString("x_desc RealDimCnt[",x_real_dim_cnt,"] < axis + 1[",axis + 1,"]"));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
      std::string err_msg = OtherErrMsg(ConcatString("start[",start,"] error."));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    if (start > rank) {
      start = rank;
    }
    if (start < 0) {
      start = rank + start;
      if (start < 0) {
        std::string err_msg = OtherErrMsg(ConcatString("start[",start,"], rank[",rank,"], error."));
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = GetInputInvalidErrMsg("axis");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (!IsUnknownRank(op, "x")) {
    if (axis < -dimnum || axis >= dimnum) {
      string dim_range = ConcatString(-dimnum,",", dimnum);
      std::string err_msg = GetParamOutRangeErrMsg("axis", dim_range, std::to_string(axis));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
  int64_t batch_dims = 0;
  int64_t rank_indices = static_cast<int64_t>(indices_desc->GetOriginShape().GetDims().size());
  if (ge::GRAPH_SUCCESS != static_cast<int64_t>(op.GetAttr("batch_dims", batch_dims))) {
    batch_dims = 0;
    OP_LOGW(op.GetName().c_str(), "GetAttr(batch_dims) failed, set default value to 0.");
  }
  if (batch_dims < 0) {
    batch_dims = batch_dims + rank_indices;
  }
  if (batch_dims != 0) {
      axis = batch_dims;
  }
  // unknown rank
  if (IsUnknownRankShape(indices_shape) || IsUnknownRankShape(x_shape)) {
    y_shape.push_back(-2);
    y_desc->SetShape(ge::GeShape(y_shape));
    y_desc->SetDataType(x_dtype);
  } else {
    if (GatherV2InferOptimize(op, axis, x_desc, indices_desc, y_desc, x_shape, indices_shape, y_shape, x_shape_range,
                              indices_shape_range, y_shape_range, 0) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }
  }
  OP_LOGI(op.GetName().c_str(), "output shape range is:%s", to_string(y_shape_range).c_str());
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Gather, GatherInferShape);
// ----------------Gather END-------------------

// --------------------------GatherElements-------------------------
static bool InferShapeAndTypeGatherElements(Operator& op, const string& paramName,
                                            const string& indicesName, const string& outputName) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  auto InputDesc = op_desc->MutableInputDesc(paramName.c_str());
  auto IndicesDesc = op_desc->MutableInputDesc(indicesName.c_str());
  auto OutputDecs = op_desc->MutableOutputDesc(outputName.c_str());

  DataType paramType = InputDesc->GetDataType();
  Format paramFormat = InputDesc->GetFormat();
  GeShape paramShape = InputDesc->GetShape();
  GeShape indicesShape = IndicesDesc->GetShape();
  std::vector<std::pair<int64_t, int64_t>> index_range;
  IndicesDesc->GetShapeRange(index_range);

  DataType indices_dtype = IndicesDesc->GetDataType();

  if ((indices_dtype != DT_INT32) && (indices_dtype != DT_INT64)) {
    OP_LOGE("gather_elements", "The indices type is not int32 or int64, please check!");
    return false;
  }

  std::vector<int64_t> params_dims = paramShape.GetDims();
  std::vector<int64_t> indices_dims = indicesShape.GetDims();

  if (params_dims.size() != indices_dims.size()) {
    OP_LOGE("gather_elements", "input dims not equal indices dims");
    return false;
  }

  GeShape output_shape = GeShape(indices_dims);
  OutputDecs->SetDataType(paramType);
  OutputDecs->SetFormat(paramFormat);
  OutputDecs->SetShape(output_shape);
  OutputDecs->SetShapeRange(index_range);

  return true;
}

IMPLEMT_COMMON_INFERFUNC(GatherElementsInferShape) {
  if (InferShapeAndTypeGatherElements(op, "x", "index", "y") == false) {
    OP_LOGE("gather_elements", "gather_elements infer shape failed!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(GatherElements, GatherElementsVerify) {
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(GatherElements, GatherElementsInferShape);
VERIFY_FUNC_REG(GatherElements, GatherElementsVerify);
// --------------------------GatherElements END---------------------

// --------------------------GatherD-------------------------
IMPLEMT_COMMON_INFERFUNC(GatherDInferShape) {
    auto op_info = OpDescUtils::GetOpDescFromOperator(op);
    if (op_info == nullptr) {
      OP_LOGE(op.GetName().c_str(), "op_info should not be nullptr");
      return GRAPH_FAILED;
    }
    auto outputTensordesc = op_info->MutableOutputDesc("y");
    auto x_desc = op_info->MutableInputDesc("x");
    auto indices_desc = op_info->MutableInputDesc("index");

    std::vector<std::pair<int64_t, int64_t>> out_range;
    DataType x_dtype = x_desc->GetDataType();
    vector<int64_t> indices_shape = indices_desc->MutableShape().GetDims();
    indices_desc->GetShapeRange(out_range);

    outputTensordesc->SetShape(GeShape(indices_shape));
    outputTensordesc->SetShapeRange(out_range);
    outputTensordesc->SetDataType(x_dtype);
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(GatherD, GatherDInferShape);
// --------------------------GatherD END---------------------

// --------------------------LogSpaceD---------------------
bool InferShapeAndTypeLogSpaceD(Operator& op, const string& input_name, const string& output_name,
                                    const string& attr_name) {
    TensorDesc v_output_desc = op.GetOutputDesc(output_name);
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

static void GetRealRange(ge::GeShape shape, std::vector<std::pair<int64_t, int64_t>>& range) {
  if (shape.IsUnknownDimNum()) {
    return;
  }
  if (range.empty()) {
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
      int64_t dim = shape.GetDim(i);
      if (dim == -1) {
        range.push_back(std::pair<int64_t, int64_t>(1, -1));
      } else {
        range.push_back(std::pair<int64_t, int64_t>(dim, dim));
      }
    }
  }
}

IMPLEMT_COMMON_INFERFUNC(UnsortedSegmentSumInferShape) {
  PROFILING_PROTO_INIT(op.GetName().c_str());
  vector<string> input_infer_depends = {"num_segments"};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends(input_infer_depends);

  Tensor input_num_segments_tensor;
  int64_t input_num_segments;
  DataType input_num_segments_dtype = op_desc->GetInputDescPtr(2)->GetDataType();

  std::vector<std::pair<int64_t, int64_t>> shape_range_x;
  op_desc->GetInputDescPtr(0)->GetShapeRange(shape_range_x);
  std::vector<std::pair<int64_t, int64_t>> shape_range_seg_id;
  op_desc->GetInputDescPtr(1)->GetShapeRange(shape_range_seg_id);

  std::vector<std::pair<int64_t, int64_t>> out_range;

  if (GRAPH_SUCCESS != op.GetInputConstData("num_segments", input_num_segments_tensor)) {
    input_num_segments = -1;
    out_range.push_back(std::pair<int64_t, int64_t>(0, -1));
  } else {
    GetUnsortedSegmentSumConstValue(input_num_segments_tensor, input_num_segments_dtype, input_num_segments);
    out_range.push_back(std::pair<int64_t, int64_t>(input_num_segments, input_num_segments));
  }

  ge::GeShape shape = op_desc->GetInputDescPtr(0)->GetShape();
  ge::GeShape shape_id = op_desc->GetInputDescPtr(1)->GetShape();

  auto output_desc = op_desc->MutableOutputDesc(0);
  ge::GeShape output_shape = output_desc->MutableShape();
  GetRealRange(shape, shape_range_x);
  GetRealRange(shape_id, shape_range_seg_id);

  int64_t dim_idsize_input = shape_id.GetDimNum();
  int64_t dim_size_input = shape.GetDimNum();
  DataType input_dtype = op_desc->GetInputDescPtr(0)->GetDataType();
  PROFILING_PROTO_AFTER_GET_SHAPE_REG();
  if (shape.IsUnknownDimNum() || shape_id.IsUnknownDimNum()) {
    if (shape.IsUnknownDimNum()) {
      output_desc->SetShape(shape);
      output_desc->SetDataType(input_dtype);
    } else {
      output_desc->SetShape(shape_id);
      output_desc->SetDataType(input_dtype);
    }
    return GRAPH_SUCCESS;
  } else if (dim_idsize_input > 1) {
    size_t rank = dim_size_input - dim_idsize_input + 1;
    size_t idx = 1;
    output_shape.SetDimNum(rank);
    output_shape.SetDim(0, input_num_segments);

    for (int64_t i = dim_idsize_input; i < dim_size_input; i++) {
      int64_t x_dim = shape.GetDim(i);
      output_shape.SetDim(idx, x_dim);
      if ((size_t)i < shape_range_x.size()) {
        out_range.push_back(shape_range_x[i]);
      }
      idx ++;
    }
  } else {
    size_t rank = shape.GetDimNum();
    output_shape.SetDimNum(rank);
    output_shape.SetDim(0, input_num_segments);

    for (size_t i = 1; i < rank; i++) {
      int64_t x_dim = shape.GetDim(i);
      output_shape.SetDim(i, x_dim);
      if ((size_t)i < shape_range_x.size()) {
        out_range.push_back(shape_range_x[i]);
      }
    }
  }

  PROFILING_PROTO_AFTER_INFER_SHAPE_REG();
  output_desc->SetShape(output_shape);
  output_desc->SetDataType(input_dtype);
  output_desc->SetShapeRange(out_range);
  PROFILING_PROTO_END();
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
    std::string err_msg = GetInputInvalidErrMsg("num_segments");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (input_num_segments <= 0) {
    std::string err_msg = GetAttrValueErrMsg("num_segments",  ConcatString(input_num_segments), ConcatString("greater than 0"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
COMMON_INFER_FUNC_REG(UnsortedSegmentMinD, UnsortedSegmentSumDInferShape);
COMMON_INFER_FUNC_REG(UnsortedSegmentMaxD, UnsortedSegmentSumDInferShape);
COMMON_INFER_FUNC_REG(UnsortedSegmentProdD, UnsortedSegmentSumDInferShape);
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
    std::string err_msg = GetInputInvalidErrMsg(ConcatString(keyName));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
      std::string err_msg = GetInputInvalidErrMsg(ConcatString(item.first));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
INFER_VALUE_RANGE_DEFAULT_REG(StridedSliceD);
// ----------------StridedSliceD Op End-------------------

IMPLEMT_INFER_DATA_SLICE(StridedSliceD, StridedSliceDInferDataSlice) {
  // Get input shape
  ge::Shape shape = op.GetInputDesc(0).GetOriginShape();

  // Get relevant masks from const node
  struct SliceMasks slice_masks = {};
  if (GRAPH_FAILED == GetStridedSliceMasks(op, slice_masks)) {
    return GRAPH_FAILED;
  }

  if (slice_masks.new_axis_mask != 0 || slice_masks.shrink_axis_mask != 0) {
    OP_LOGD(TbeGetName(op), "data slice is only support for new_axis_mask == 0 and shrink_axis_mask == 0");
    return NOT_SUPPORT_SLICE;
  }

  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  OP_LOGE_IF(!op_info, GRAPH_FAILED, TbeGetName(op), "GetOpDescFromOperator failed.");
  auto output_desc = op_info->MutableOutputDesc(0);
  OP_LOGE_IF(!output_desc, GRAPH_FAILED, TbeGetName(op), "Get output desc failed.");
  vector<vector<int64_t>> output_data_slice;
  OP_LOGE_IF(!AttrUtils::GetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, output_data_slice), GRAPH_FAILED,
             op.GetName(), "Output no data slice, not need infer input");
  if (!output_data_slice.empty()) {
    auto desc =  op_info->MutableInputDesc(0);
    OP_LOGE_IF(!output_desc, GRAPH_FAILED, TbeGetName(op), "Get input desc failed.");
    OP_LOGE_IF(!AttrUtils::SetListListInt(desc, ge::ATTR_NAME_DATA_SLICE, output_data_slice), GRAPH_FAILED,
                 op.GetName(), "Set input(%s) data slice failed", desc->GetName().c_str());

    return GRAPH_SUCCESS;
  }

  return NO_OVERLAP_DIM;
}

INFER_DATA_SLICE_FUNC_REG(StridedSliceD, StridedSliceDInferDataSlice);


// ----------------stridedSlice Op Begin-------------------
IMPLEMT_COMMON_INFERFUNC(StridedSliceInferShape) {
  const vector<string> depend_names = {"begin", "end", "strides"};
  PREPARE_DYNAMIC_SHAPE(depend_names);

  // Get input shape
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc(0);
  auto &shape = input_desc->MutableShape();
  auto input_dtype = input_desc->GetDataType();
  int64_t begin_len = -1;
  
  auto input_begin = op_info->MutableInputDesc(1)->MutableShape().GetDim(0);
  begin_len = std::max(input_begin, begin_len);
  auto input_end = op_info->MutableInputDesc(2)->MutableShape().GetDim(0);
  begin_len = std::max(input_end, begin_len);
  auto input_strides = op_info->MutableInputDesc(3)->MutableShape().GetDim(0);
  begin_len = std::max(input_strides, begin_len);

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
    auto output_desc = op_info->MutableOutputDesc(0);
    output_desc->SetDataType(input_dtype);
    output_desc->SetShape(ge::GeShape(UNKNOWN_RANK));
    OP_LOGD(op.GetName().c_str(), "output_shape:%s", to_string(output_desc->GetShape()).c_str());
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

  vector<pair<int64_t, int64_t>> input_ranges;
  input_desc->GetShapeRange(input_ranges);
  if (input_ranges.empty()) {
    MakeUpShapeRange(shape.GetDims(), input_ranges);
  }

  OP_LOGD(op.GetName().c_str(), "begin_list:%s", to_string(slice_params.begin_list).c_str());
  OP_LOGD(op.GetName().c_str(), "end_list:%s", to_string(slice_params.end_list).c_str());
  OP_LOGD(op.GetName().c_str(), "stride_list:%s", to_string(slice_params.stride_list).c_str());
  if (slice_params.end_list.size() != slice_params.begin_list.size()) {
    std::string err_msg = OtherErrMsg("end shape, begin shape length mismatch!");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
INFER_VALUE_RANGE_DEFAULT_REG(StridedSlice);
// ----------------StridedSlice Op End-------------------

// ----------------ReverseV2 Op Begin-----------------
IMPLEMT_COMMON_INFERFUNC(ReverseV2InferShape) {
  const vector<string> depend_names = {"axis"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(ReverseV2, ReverseV2InferShape);

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
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
        string("call function CheckTwoInputDtypeSame failed, data type of input[x1] is not same as input[x2]"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(SelectInferShape) {
  if (!TwoInOneOutDynamicInferNoBroadcast(op, "x1", "x2", {"y"})) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
        string("call function TwoInOneOutDynamicInferNoBroadcast failed, update output[y] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
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
  for (size_t i = 0; i < dim_x.size(); i++) {
    if (dim_x[i] == -2 || dim_y[i] == -2) {
      dim_out.push_back(-2);
      return true;
    }
  }
  // set out dims
  for (size_t i = 0; i < dim_x.size(); i++) {
    if ((dim_x[i] != dim_y[i]) && ((dim_x[i] != 1 && dim_x[i] != -1) && (dim_y[i] != 1 && dim_y[i] != -1))) {
      string msg = ConcatString("The dimensions does not match the broadcast rule(", dim_x[i], ", ", dim_y[i], ")");
      std::string err_msg = OtherErrMsg(msg);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return false;
    }

    int64_t dim = std::max(dim_x[i], dim_y[i]);
    if (dim == 1 && (dim_x[i] == -1 || dim_y[i] == -1)) {
      dim = -1;
    }
    std::pair<int64_t, int64_t> range;
    if (dim != -1) {
      range = {dim, dim};
    } else {
      range = {0, -1};
    }
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
  std::vector<std::pair<int64_t, int64_t>> condition_range;
  condition_desc->GetShapeRange(condition_range);

  auto then_desc = op_info->MutableInputDesc("then");
  vector<int64_t> then_shape = then_desc->MutableShape().GetDims();
  DataType then_dtype = then_desc->GetDataType();
  std::vector<std::pair<int64_t, int64_t>> then_range;
  then_desc->GetShapeRange(then_range);

  auto else_desc = op_info->MutableInputDesc("else");
  vector<int64_t> else_shape = else_desc->MutableShape().GetDims();
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
      std::string err_msg = ConcatString("failed to call GetConstIntData function ",
          "due to invalid data type of input[segment_ids]. data_type is ",
          DTypeStr(data_type));
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    first_axis_dims = (*std::max_element(const_data.begin(), const_data.end())) + 1;
  }

  auto output_shape_dims = input_desc.GetShape().GetDims();
  if (output_shape_dims.empty()) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        std::string("the input[x]'s shape should not be empty."));
    return GRAPH_FAILED;
  }
  output_shape_dims[0] = first_axis_dims;
  Shape output_shape(output_shape_dims);
  DataType input_dtype = input_desc.GetDataType();
  TensorDesc tensor_desc_output = op.GetOutputDesc("y");
  tensor_desc_output.SetShape(output_shape);
  tensor_desc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", tensor_desc_output);
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

// ----------------SegmentSum-------------------
static bool SegmentSumShapeVerify(const Operator& op, const std::string& input_name,
                                  const std::string& segment_ids_name) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_shape_dims = op_info->MutableInputDesc("x")->MutableShape().GetDims();
  auto segment_ids_shape_dims = op_info->MutableInputDesc("segment_ids")->MutableShape().GetDims();

  return true;
}

IMPLEMT_VERIFIER(SegmentSum, SegmentSumInferShapeVerifier) {
  if (!SegmentSumShapeVerify(op, "x", "segment_ids")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(SegmentSumInferShape) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_x_desc = op_info->MutableInputDesc("x");
  auto output_desc = op_info->MutableOutputDesc("y");
  auto shape_x = input_x_desc->MutableShape().GetDims();
  auto output_shape_dims = input_x_desc->MutableShape().GetDims();
  if (output_shape_dims.empty()) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        std::string("the input[x]'s shape should not be empty."));
    return GRAPH_FAILED;
  }
  const vector<string> depend_name = {"segment_ids"};
  PREPARE_DYNAMIC_SHAPE(depend_name);
  const std::string segment_ids_name = "segment_ids";
  Tensor segment_ids;
  int64_t first_axis_dims;
  int64_t out_range_first_dims;
  if (GRAPH_SUCCESS != op.GetInputConstData(segment_ids_name, segment_ids)) {
    OP_LOGI("segment_max", "GetInputConstData %s failed.", segment_ids_name.c_str());
    first_axis_dims = -1;
    out_range_first_dims = 0;
  } else {
    auto data_type = op.GetInputDesc(segment_ids_name).GetDataType();
    std::vector<int64_t> const_data;
    if (!GetConstIntData(segment_ids, data_type, const_data)) {
      std::string err_msg = ConcatString("failed to call GetConstIntData function ",
          "due to invalid data type of input[segment_ids]. data_type is ",
          DTypeStr(data_type));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    first_axis_dims = (*std::max_element(const_data.begin(), const_data.end())) + 1;
    out_range_first_dims = first_axis_dims;
  }

  if (IsUnknownRankShape(shape_x)) {
    output_desc->SetShape(GeShape(shape_x));
  } else {
    output_shape_dims[0] = first_axis_dims;
    Shape output_shape(output_shape_dims);
    output_desc->SetShape(GeShape(output_shape_dims));
    if (IsUnKnownShape(shape_x)) {
      std::vector<std::pair<int64_t, int64_t>> shape_range_x;
      std::vector<std::pair<int64_t, int64_t>> output_shape_range;
      output_shape_range.push_back(std::pair<int64_t, int64_t>(out_range_first_dims, first_axis_dims));
      input_x_desc->GetShapeRange(shape_range_x);
      for (size_t i = 1; i < output_shape_dims.size(); i++) {
        if (i < shape_range_x.size()) {
          output_shape_range.push_back(shape_range_x[i]);
        }
      }
      MakeUpShapeRange(output_shape_dims, output_shape_range);
      output_desc->SetShapeRange(output_shape_range);
    }
  }
  DataType input_dtype = input_x_desc->GetDataType();
  output_desc->SetDataType(input_dtype);
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(SegmentSum, SegmentSumInferShape);
VERIFY_FUNC_REG(SegmentSum, SegmentSumInferShapeVerifier);
// ----------------SegmentSum END-------------------

// ----------------SliceD Op Begin ----------------------
IMPLEMT_VERIFIER(SliceD, SliceDVerify) {
  std::vector<int64_t> input_size;
  if (ge::GRAPH_SUCCESS != op.GetAttr("size", input_size)) {
    std::string err_msg = GetInputInvalidErrMsg("ConstValue");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> input_begin;
  if (ge::GRAPH_SUCCESS != op.GetAttr("offsets", input_begin)) {
    std::string err_msg = GetInputInvalidErrMsg("begin");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(SliceDInferShape) {
  const vector<string> depends;
  PREPARE_DYNAMIC_SHAPE(depends);
  std::vector<int64_t> input_size;
  if (ge::GRAPH_SUCCESS != op.GetAttr("size", input_size)) {
    std::string err_msg = GetInputInvalidErrMsg("size");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<int64_t> input_begin;
  if (ge::GRAPH_SUCCESS != op.GetAttr("offsets", input_begin)) {
    std::string err_msg = GetInputInvalidErrMsg("input_begin");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  Shape shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  size_t dimNum = shape.GetDimNum();

  if ((int64_t)input_size.size() != (int64_t)dimNum) {
    std::string err_msg = GetAttrSizeErrMsg("input_size", ConcatString((int64_t)input_size.size()), ConcatString((int64_t)dimNum));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if ((int64_t)input_begin.size() != (int64_t)dimNum) {
    std::string err_msg = GetAttrSizeErrMsg("input_begin", ConcatString((int64_t)input_begin.size()), ConcatString((int64_t)dimNum));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  for (int64_t i = 0; i < (int64_t)dimNum; ++i) {
    if (input_size[i] > shape.GetDim(i) || input_size[i] < -1) {
      string excepted_value = ConcatString("in range[0,", shape.GetDim(i), "]");
      std::string err_msg = GetAttrValueErrMsg("size", ConcatString(input_size[i]), excepted_value);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    if (input_begin[i] > shape.GetDim(i) || input_begin[i] < -1) {
      string excepted_value = ConcatString("in range[-1,", shape.GetDim(i), "] and cannot be equal to 0");
      std::string err_msg = GetAttrValueErrMsg("begin", ConcatString(input_begin[i]), excepted_value);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
      ranges[i].first = 0;
    }
  } else if (!has_offsets && has_size) {
    for (size_t i = 0; i < dimNum; ++i) {
      if (input_size[i] == -1) {
        outputList.push_back(-1);
        ranges[i].first = 0;
      } else {
        outputList.push_back(input_size[i]);
        ranges[i].first = input_size[i];
        ranges[i].second = input_size[i];
      }
    }
  } else if (has_offsets && !has_size) {
    for (size_t i = 0; i < dimNum; ++i) {
      outputList.push_back(-1);
      ranges[i].first = 0;
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

        ranges[i].first = 0;
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
INFER_VALUE_RANGE_DEFAULT_REG(Slice);
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
    OP_LOGE(op.GetName().c_str(), "OneHot GetOpAttr depth failed!");
    return GRAPH_FAILED;
  }
  if (depth < 1) {
    OP_LOGE(op.GetName().c_str(), "depth need greater than or equals to 1");
    return GRAPH_FAILED;
  }

  ge::Shape indices_shape = op.GetInputDesc(0).GetShape();
  int32_t dim_nums = 0;
  dim_nums = indices_shape.GetDimNum();
  int32_t axis = -1;
  if (ge::GRAPH_SUCCESS != op.GetAttr("axis", axis)) {
    OP_LOGE(op.GetName().c_str(), "Get const axis failed from op of 'OneHotD'!\n");
    return GRAPH_FAILED;
  }
  if (axis < -dim_nums || axis > dim_nums) {
    OP_LOGE(op.GetName().c_str(), "attr axis is not in range");
    return GRAPH_FAILED;
  }
  if (axis < -1) {
    OP_LOGE(op.GetName().c_str(), "attr axis must be >= -1");
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
    std::string err_msg = GetInputInvalidErrMsg("Get const axis failed from op of 'OneHot'!\n");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (axis < -1) {
    string correct_size = ConcatString("attr axis(", axis, ") must be >= -1");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), correct_size);
    return GRAPH_FAILED;
  }

  // get all Desc info
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  static const int64_t input_x_idx = 0;
  auto input_desc = op_info->MutableInputDesc(input_x_idx);
  const ge::GeShape& input_shape = input_desc->MutableShape();

  static const int64_t input_on_value_idx = 2;
  auto value_desc = op_info->MutableInputDesc(input_on_value_idx);
  DataType value_dtype = value_desc->GetDataType();

  // output desc and set dtype
  static const int64_t output_y_idx = 0;
  auto output_desc = op_info->MutableOutputDesc(output_y_idx);
  output_desc->SetDataType(value_dtype);

  if (input_shape.IsUnknownDimNum()) {
    // input is UnknownRank, set output UnknownRank
    OP_LOGW("OneHot", "input shape is UnknownRank, set output UnknownRank");
    output_desc->SetShape(input_shape);
    return GRAPH_SUCCESS;
  }
  // update axis to positive number
  int32_t dimnum = input_shape.GetDimNum();
  if (axis > dimnum) {
    string correct_size = ConcatString("attr axis(", axis, ") must be < ", input_shape.GetDimNum());
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), correct_size);
    return GRAPH_FAILED;
  }

  // get depth const value, depth index is 1
  int64_t depth_value = -1;
  static const int64_t input_depth_idx = 1;
  if (!ops::GetConstInt(op, input_depth_idx, depth_value)) {
    OP_LOGW("OneHot", "Get depth const tensor failed, set depth -1");
  }

  // update output shape
  ge::GeShape& output_shape = output_desc->MutableShape();
  output_shape.SetDimNum(dimnum + 1);
  if (-1 == axis) {
    for (int32_t i = 0; i < dimnum; i++) {
      output_shape.SetDim(i, input_shape.GetDim(i));
    }
    output_shape.SetDim(dimnum, depth_value);
  } else {
    while (dimnum > axis) {
      output_shape.SetDim(dimnum, input_shape.GetDim(dimnum - 1));
      dimnum--;
    }
    output_shape.SetDim(axis, depth_value);
    for (int32_t i = 0; i < axis; i++) {
      output_shape.SetDim(i, input_shape.GetDim(i));
    }
  }

  // if output shape is dynamic update output range
  if (output_shape.IsUnknownShape()) {
    output_desc->SetOriginShape(output_shape);
    std::vector<std::pair<int64_t, int64_t>> input_range;
    input_desc->GetShapeRange(input_range);
    MakeUpShapeRange(input_shape, input_range);
    std::pair<int64_t, int64_t> depth_range = depth_value == -1 ?
                                              std::pair<int64_t, int64_t>(1, -1):
                                              std::pair<int64_t, int64_t>(depth_value, depth_value);
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

static void TopKGetShapeRange(std::vector<std::pair<int64_t, int64_t>> &shape_range,
                              const GeShape &input_shape, size_t dim_size, int64_t k,
                              uint32_t sorted_axis) {
  for (size_t i = 0; i < dim_size; i++) {
    if (i == sorted_axis && k > 0) {
      shape_range.push_back(pair<int64_t, int64_t>(k, k));
    } else if (input_shape.GetDim(i) == UNKNOWN_DIM) {
      shape_range.push_back(pair<int64_t, int64_t>(1, -1));
    } else {
      shape_range.push_back(pair<int64_t, int64_t>(input_shape.GetDim(i), input_shape.GetDim(i)));
    }
  }
}

static bool TopKInferCommon(Operator &op, int64_t k) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->GetInputDescPtr(0);
  auto output_v_desc = op_info->MutableOutputDesc(0);
  auto output_i_desc = op_info->MutableOutputDesc(1);

  GeShape &output_v_shape = output_v_desc->MutableShape();
  GeShape &output_i_shape = output_i_desc->MutableShape();

  const GeShape &input_shape = input_desc->GetShape();
  if (input_shape.IsUnknownDimNum()) {
    output_v_desc->SetShape(input_shape);
    output_v_desc->SetDataType(input_desc->GetDataType());
    output_i_desc->SetShape(input_shape);
    output_i_desc->SetDataType(DT_INT32);
    return true;
  }

  int64_t dim_size = input_shape.GetDimNum();
  if (dim_size <= 0) {
    OP_LOGE(op.GetName().c_str(), "The dims_in size should more than 0!");
    return false;
  }

  int64_t dim = dim_size - 1;
  int64_t sorted_axis = dim;
  if (AttrUtils::GetInt(op_info, "dim", dim)) {
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
    if (k > 0 && static_cast<int64_t>(sorted_axis) < static_cast<int64_t>(shape_range.size())) {
      shape_range[sorted_axis].first = k;
      shape_range[sorted_axis].second = k;
    }
  } else {
    // input is static shape
    TopKGetShapeRange(shape_range, input_shape, dim_size, k, static_cast<uint32_t>(sorted_axis));
  }

  output_v_shape.SetDimNum(dim_size);
  output_i_shape.SetDimNum(dim_size);

  for (int64_t i = 0; i < dim_size; i++) {
    if (i == sorted_axis) {
      output_v_shape.SetDim(i, k);
      output_i_shape.SetDim(i, k);
      continue;
    }
    int64_t size = input_shape.GetDim(i);
    output_v_shape.SetDim(i, size);
    output_i_shape.SetDim(i, size);
  }
  
  output_v_desc->SetShape(output_v_shape);
  output_v_desc->SetDataType(input_desc->GetDataType());
  output_i_desc->SetShape(output_i_shape);
  output_i_desc->SetDataType(DT_INT32);
  return true;
}

// ----------------TopKV2D Op---------------------
IMPLEMT_COMMON_INFERFUNC(TopKV2DInferShape) {
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
COMMON_INFER_FUNC_REG(TopKV2D, TopKV2DInferShape);
// ----------------TopKV2D Op End-----------------

// ----------------TopKD Op Start-------------------
IMPLEMT_COMMON_INFERFUNC(TopKDInferShape) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);

  int32_t k;
  if (op.GetAttr("k", k) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("k");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (TopKInferCommon(op, k) == false) {
    std::string err_msg = OtherErrMsg("TopKInferCommon Failed.");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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

  int64_t k = UNKNOWN_DIM;
  static const int64_t input_k_idx = 1;
  if (!(ops::GetConstInt(op, input_k_idx, k))) {
    OP_LOGI(TbeGetName(op), "Get constdata failed, unknown dim.");
  }

  if (TopKInferCommon(op, k) == false) {
    std::string err_msg = OtherErrMsg("TopKInferCommon Failed.");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
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
    USER_GE_LOGE("GetAttr %s failed.", shape_name.c_str());
    return GRAPH_FAILED;
  }
  vector<int64_t> shape_dims;
  for (size_t i = 0; i < (uint32_t)shape_out_list.size(); ++i) {
    shape_dims.push_back(shape_out_list[i]);
  }
  if (shape_out_list.size() != shape_dims.size()) {
    string correct_size = ConcatString("same with output_y[", shape_dims.size(), "]");
    std::string err_msg = GetAttrSizeErrMsg("shape_out_list", ConcatString(shape_out_list.size()), correct_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  for (int64_t i = 0; i < (int64_t)shape_dims.size(); i++) {
    if (shape_out_list[i] != shape_dims[i]) {
      string excepted_value = ConcatString("same with output_y[", shape_dims[i], "]");
      std::string err_msg = GetAttrValueErrMsg("x'shape", ConcatString(shape_out_list[i]), excepted_value);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    OP_LOGE(op.GetName().c_str(), "Target must be 1-dimensional, but get %lu", target_dim);
    return false;
  }
  if (!unknown_rank) {
    size_t prediction_dim = input_prediction->GetShape().GetDimNum();
    if (prediction_dim != 2) {
      OP_LOGE(op.GetName().c_str(), "Predictions must be 2-dimensional, but get %lu", prediction_dim);
      return false;
    }
    if (input_prediction->GetShape().GetDim(0) != -1 && input_target->GetShape().GetDim(0) != -1) {
      if (input_prediction->GetShape().GetDim(0) != input_target->GetShape().GetDim(0)) {
        OP_LOGE(op.GetName().c_str(),
                "First dimension of prediction must match length of targets, but first dimension of prediction get %ld",
                input_prediction->GetShape().GetDim(0));
        return false;
      }
    }
  }
  return true;
}

IMPLEMT_VERIFIER(InTopKD, InTopKDVerify) {
  if (!InTopKDCheckInputX1AndX2(op)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(InTopKDInferShape) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_prediction = op_info->MutableInputDesc("x1");
  if (input_prediction == nullptr) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), std::string("get input[x1] desc failed, input[x1] desc is nullptr."));
    return GRAPH_FAILED;
  }
  auto input_target = op_info->MutableInputDesc("x2");
  if (input_target == nullptr) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), std::string("get input[x2] desc failed, input[x2] desc is nullptr."));
    return GRAPH_FAILED;
  }
  auto output_desc = op_info->MutableOutputDesc("y");
  if (output_desc == nullptr) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), std::string("get output[y] desc failed, output[y] desc is nullptr."));
    return GRAPH_FAILED;
  }
  std::vector<std::pair<int64_t, int64_t>> input1_range;
  input_target->GetShapeRange(input1_range);
  std::vector<int64_t> dims_in_prediction = input_prediction->MutableShape().GetDims();
  std::vector<int64_t> dims_in_target = input_target->MutableShape().GetDims();
  bool unknown_rank = IsUnknownRankShape(dims_in_prediction);
  if (!InTopKDCheckInputX1AndX2(op)) {
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
        AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), std::string("set output[y] shape range failed."));
        return GRAPH_FAILED;
      }
    }
  }
  output_desc->SetDataType(DT_BOOL);
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
    OP_LOGE(op.GetName().c_str(), "Target must be 1-dimensional, but get %lu", target_dim);
    return false;
  }
  if (!unknown_rank) {
    size_t prediction_dim = input_prediction->GetShape().GetDimNum();
    if (prediction_dim != 2) {
      OP_LOGE(op.GetName().c_str(), "Predictions must be 2-dimensional, but get %lu", prediction_dim);
      return false;
    }
    if (input_prediction->GetShape().GetDim(0) != -1 && input_target->GetShape().GetDim(0) != -1) {
      if (input_prediction->GetShape().GetDim(0) != input_target->GetShape().GetDim(0)) {
        OP_LOGE(op.GetName().c_str(),
                "First dimension of prediction must match length of targets, but first dimension of prediction get %ld",
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
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), std::string("get input[x1] desc failed, input[x1] desc is nullptr."));
    return GRAPH_FAILED;
  }
  auto input_target = op_info->MutableInputDesc("x2");
  if (input_target == nullptr) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), std::string("get input[x2] desc failed, input[x2] desc is nullptr."));
    return GRAPH_FAILED;
  }
  auto output_desc = op_info->MutableOutputDesc("y");
  if (output_desc == nullptr) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), std::string("get output[y] desc failed, output[y] desc is nullptr."));
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
        AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), std::string("set output[y] shape range failed."));
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
    std::string err_msg = GetInputInvalidErrMsg("begin");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (begin.size() > 8) {
    string correct_size = ConcatString("less than or equal to 8");
    std::string err_msg = GetAttrSizeErrMsg("begin", ConcatString(begin.size()), correct_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<int64_t> end;
  end = GetAttrValue(op, "end");
  if (!CheckListEmpty(op.GetName(), end, "end")) {
    std::string err_msg = GetInputInvalidErrMsg("end");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (end.size() > 8) {
    string correct_size = ConcatString("less than or equal to 8");
    std::string err_msg = GetAttrSizeErrMsg("end", ConcatString(end.size()), correct_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(op.GetName(), strides, "strides")) {
    std::string err_msg = GetInputInvalidErrMsg("strides");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (strides.size() > 8) {
    string correct_size = ConcatString("less than or equal to 8");
    std::string err_msg = GetAttrSizeErrMsg("strides", ConcatString(strides.size()), correct_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  TensorDesc tensordesc_output = op.GetOutputDesc("var");
  tensordesc_output.SetShape(op.GetInputDesc("var").GetShape());
  tensordesc_output.SetDataType(op.GetInputDesc("var").GetDataType());
  if (op.UpdateOutputDesc("var", tensordesc_output) != GRAPH_SUCCESS) {
    std::string err_msg = UpdateParamErrMsg("var");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(StridedSliceAssignD, StridedSliceAssignDInferShape);
// ----------------StridedSliceAssignD Op Begin-------------------

// ----------------Cumprod-------------------
IMPLEMT_COMMON_INFERFUNC(CumprodInferShape) {
  TensorDesc desc = op.GetInputDesc("x");
  return op.UpdateOutputDesc("y", desc);
}

COMMON_INFER_FUNC_REG(Cumprod, CumprodInferShape);
// ----------------Cumprod END-------------------

// ----------------CumprodD-------------------
IMPLEMT_VERIFIER(CumprodD, CumprodDVerify) {
  int64_t axis;
  if (GRAPH_SUCCESS != op.GetAttr("axis", axis)) {
    std::string err_msg = GetInputInvalidErrMsg("axis");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  TensorDesc input_desc = op.GetInputDesc("x");
  int64_t dimnum;
  dimnum = input_desc.GetShape().GetDimNum();
  if (axis < -dimnum || axis >= dimnum) {
    string dim_range = ConcatString(-dimnum,",", dimnum);
    std::string err_msg = GetParamOutRangeErrMsg("axis", dim_range, std::to_string(axis));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);

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
    std::string err_msg = GetInputInvalidErrMsg("axis");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  TensorDesc input_desc = op.GetInputDesc("x");
  int64_t dimnum;
  dimnum = input_desc.GetShape().GetDimNum();
  if (axis < -dimnum || axis >= dimnum) {
    string dim_range = ConcatString(-dimnum,",", dimnum);
    std::string err_msg = GetParamOutRangeErrMsg("axis", dim_range, std::to_string(axis));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);

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

// ----------------Cummax------------------------
IMPLEMT_COMMON_INFERFUNC(CummaxInferShape) {
  TensorDesc output_desc_y = op.GetOutputDesc("y");
  TensorDesc output_desc_indices = op.GetOutputDesc("indices");
  DataType predict_dtype = op.GetInputDesc("x").GetDataType();
  Format predict_format = op.GetInputDesc("x").GetFormat();
  ge::Shape output_shape = op.GetInputDesc("x").GetShape();
  output_desc_y.SetDataType(predict_dtype);
  output_desc_y.SetFormat(predict_format);
  output_desc_y.SetShape(output_shape);
  output_desc_indices.SetFormat(predict_format);
  output_desc_indices.SetShape(output_shape);
  (void)op.UpdateOutputDesc("y", output_desc_y);
  (void)op.UpdateOutputDesc("indices", output_desc_indices);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Cummax, CummaxVerify) {
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Cummax, CummaxInferShape);
VERIFY_FUNC_REG(Cummax, CummaxVerify);
// ----------------Cummax END----------------------

// ----------------InplaceUpdate-------------------
IMPLEMT_VERIFIER(InplaceUpdate, InplaceUpdateVerify) {
  if (!CheckTwoInputDtypeSame(op, "x", "v")) {
    std::string err_msg = OtherErrMsg("the dtype of x and v should be same!");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(InplaceUpdateInferShape) {
  const int64_t input_x_idx = 0;
  if (OneInOneOutDynamicInfer(op, input_x_idx, {input_x_idx})) {
    return GRAPH_SUCCESS;
  }
  VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), OtherErrMsg("InplaceUpdate OneInOneOutDynamicInfer failed"));
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(InplaceUpdate, InplaceUpdateInferShape);
VERIFY_FUNC_REG(InplaceUpdate, InplaceUpdateVerify);
// ----------------InplaceUpdate END-------------------

// ----------------InplaceUpdateD-------------------
IMPLEMT_COMMON_INFERFUNC(InplaceUpdateDInferShape) {
  auto output_desc = op.GetInputDesc("x");
  auto input_v_desc = op.GetInputDesc("v");
  int64_t dim_value_v;
  dim_value_v = input_v_desc.GetShape().GetDim(0);
  std::vector<int64_t> indices;
  if (op.GetAttr("indices", indices) == GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("indices");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
  }
  if ((int64_t)indices.size() != dim_value_v) {
    string err_msg = ConcatString("The length of rank 0 of tensor v must be the same as length of indices. indices:",(int64_t)indices.size(), ", dim_value_v:",dim_value_v);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
  graphStatus ret = op.GetInputConstData(indices_name, indices);
  OP_LOGI(op.GetName().c_str(), "InplaceAddInferShape get const data from input[%s] ret = %d",
          indices_name.c_str(), static_cast<int>(ret));

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
    std::string err_msg = GetInputInvalidErrMsg("indices");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
  }

  if ((int64_t)indices.size() != dim_value_v) {
    string err_msg1 = ConcatString("bias shape extends x shape when applied, indices:",(int64_t)indices.size(), ", dim_value_v:",dim_value_v);
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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

  graphStatus ret = op.GetInputConstData(indices_name, indices);
  OP_LOGI(op.GetName().c_str(), "InplaceSubInferShape get const data from input[%s] ret = %d",
          indices_name.c_str(), static_cast<int>(ret));

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
    std::string err_msg = GetInputInvalidErrMsg("indices");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
  }

  if ((int64_t)indices.size() != dim_value_v) {
    string err_msg1 = ConcatString("The length of rank 0 of tensor v must be the same as length of indices., indices:",(int64_t)indices.size(), ", dim_value_v:",dim_value_v);
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(InplaceSubD, InplaceSubDInferShape);
// ----------------InplaceSubD  END-------------------

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
    std::string err_msg = GetInputInvalidErrMsg("post_nms_topn");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    string excepted_value = ConcatString("not equal 0");
    std::string err_msg = GetAttrValueErrMsg("stride", ConcatString(stride), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    string expected_format_list = ConcatString("FORMAT_NCHW, FORMAT_NHWC");
    std::string err_msg = GetInputFormatNotSupportErrMsg("inputFormat", expected_format_list, ConcatString(inputFormat));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (reverse) {
    if (stride < 1) {
      string excepted_value = ConcatString("greater than or equal to 1");
      std::string err_msg = GetAttrValueErrMsg("stride", std::to_string(stride), excepted_value);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    int64_t modC = (inputFormat == FORMAT_NCHW) ? (int64_t)inputShape[1] % (stride * stride)
                                                : (int64_t)inputShape[3] % (stride * stride);
    if (modC != 0) {
      string excepted_value = ConcatString("0");
      std::string err_msg = GetAttrValueErrMsg("modC", std::to_string(modC), excepted_value);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }

  } else {
    if (stride < 1) {
      string excepted_value = ConcatString("greater than or equal to 1");
      std::string err_msg = GetAttrValueErrMsg("stride", std::to_string(stride), excepted_value);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    int64_t modH = (inputFormat == FORMAT_NCHW) ? (int64_t)inputShape[2] % stride : (int64_t)inputShape[1] % stride;
    int64_t modW = (inputFormat == FORMAT_NCHW) ? (int64_t)inputShape[3] % stride : (int64_t)inputShape[2] % stride;
    if (modH != 0) {
      string excepted_value = ConcatString("0");
      std::string err_msg = GetAttrValueErrMsg("modH", std::to_string(modH), excepted_value);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    if (modW != 0) {
      string excepted_value = ConcatString("0");
      std::string err_msg = GetAttrValueErrMsg("modW", std::to_string(modW), excepted_value);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = GetInputInvalidErrMsg("axis");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (axis >= dimNum || axis < -dimNum) {
    string minvalue = ConcatString(-dimNum);
    string maxvalue = ConcatString(dimNum - 1);
    string axis_range = ConcatString(-dimNum,",", dimNum);
    std::string err_msg = GetParamOutRangeErrMsg("axis", axis_range, std::to_string(axis));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = GetInputInvalidErrMsg("tiles");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int64_t axis;
  if (GRAPH_SUCCESS != op.GetAttr("axis", axis)) {
    std::string err_msg = GetInputInvalidErrMsg("axis");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
        string expected_format_list = ConcatString("[NCHW, NHWC]");
        std::string err_msg = GetInputFormatNotSupportErrMsg("5D tensor's origin", expected_format_list, ConcatString(originFormat));
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
        return GRAPH_FAILED;
      }
    } else {
      std::string err_msg = OtherErrMsg("5D tensor's origin shape should be 4D tensor");
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }

    if (axis < 0) {
      axis = axis + 5;
    }

    if (axis == 1 || axis == 4) {
      std::string err_msg = OtherErrMsg("5D tensor's axis is invalid");
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    if (static_cast<int64_t>(i) == axis) {
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
    std::string err_msg = GetInputInvalidErrMsg("stride");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  bool have_slice = false;
  for (size_t i = 0; i < y_data_slice.size(); i++) {
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
  if (!AttrUtils::SetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
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
    std::string err_msg = GetInputInvalidErrMsg("axis");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  TensorDesc input_desc = op.GetInputDesc("x");
  int64_t dimnum;
  dimnum = input_desc.GetShape().GetDimNum();
  if (axis < -dimnum || axis >= dimnum) {
    string dim_range = ConcatString(-dimnum,",", dimnum);
    std::string err_msg = GetParamOutRangeErrMsg("axis", dim_range, std::to_string(axis));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);

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
IMPLEMT_INFERFUNC(InplaceIndexAdd, InplaceIndexAddInferShape) {
  DataType var_type = op.GetInputDescByName("var").GetDataType();
  DataType indices_type = op.GetInputDescByName("indices").GetDataType();
  DataType updates_type = op.GetInputDescByName("updates").GetDataType();

  AscendString op_name;
  if (GRAPH_SUCCESS != op.GetName(op_name)) {
    OP_LOGE("InplaceIndexAdd", "op_name get failed.");
    return GRAPH_FAILED;
  }
  const char* op_name_c = op_name.GetString();

  if (var_type != updates_type) {
    OP_LOGE(op_name_c, "var'dtype is not same as updates'dtype.");
    return GRAPH_FAILED;
  }
  if (indices_type != DT_INT32) {
    OP_LOGE(op_name_c, "indices dtype is not int32, please check!");
    return GRAPH_FAILED;
  }
  if (OneInOneOutDynamicInfer(op, "var", {"var"})) {
    return GRAPH_SUCCESS;
  }

  OP_LOGE(op_name_c, "shape of var_out is not same as shape of var.");
  return GRAPH_FAILED;
}
INFER_FUNC_REG(InplaceIndexAdd, InplaceIndexAddInferShape);
// ----------------InplaceIndexAdd END---------------------

// ----------------MaskedFill Begin-------------------
IMPLEMT_COMMON_INFERFUNC(InferMaskedFillShape) {
  // ge::Operator op;
  OP_LOGD(op.GetName().c_str(), "InferMaskedFillShape Begin.");
  bool is_dynamic_output = true;
  const int64_t input_x_idx = 0;
  const int64_t input_mask_idx = 1;
  const int64_t output_y_idx = 0;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, input_x_idx, input_mask_idx,
                                             output_y_idx, is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  OP_LOGD(op.GetName().c_str(), "InferMaskedFillShape End.");

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
      OP_LOGE(op.GetName().c_str(), "The shape of x1 and x2 can not broadcast.");
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
    OP_LOGE(op.GetName().c_str(), "x y tensor dtype does not match.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(MaskedSelectV2InferShape) {
  if (InferShapeAndTypeMaskedSelectV2(op, "x", "mask", "y")) {
    return GRAPH_SUCCESS;
  }
  OP_LOGE(op.GetName().c_str(), "The shape of output y does not match that of x1 x2.");
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(MaskedSelectV2, MaskedSelectV2InferShape);
VERIFY_FUNC_REG(MaskedSelectV2, MaskedSelectV2Verify);
// ----------------MaskedSelectV2 END---------------------


// ----------------MaskedScatter Begin-------------------

IMPLEMT_COMMON_INFERFUNC(MaskedScatterInferShape) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);

  auto output_y = op_info->MutableOutputDesc("y");
  auto input_x = op_info->MutableInputDesc("x");
 
  auto x_dtype = input_x->GetDataType();
  auto x_shape = input_x->GetShape();

  output_y->SetShape(x_shape);
  output_y->SetDataType(x_dtype);
  
  auto input_dims = x_shape.GetDims();
  if (input_dims == UNKNOWN_RANK) {
    std::vector<std::pair<int64_t, int64_t>> x_range;
    input_x->GetShapeRange(x_range);
    output_y->SetShapeRange(x_range);
  }

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(MaskedScatter, MaskedScatterInferShape);
// ----------------MaskedScatter END---------------------


// ----------------StridedSliceV2 Begin-------------------
static graphStatus GetStridedSliceInferConstData(const ge::Operator& op, struct SliceParameters& slice_params,
                                                 std::vector<int64_t>& axes_list) {
  Tensor const_tensor;

  std::map<std::string, vector<int64_t>&> const_values = {
      {"begin", slice_params.begin_list},
      {"end", slice_params.end_list},
      {"strides", slice_params.stride_list},
      {"axes", axes_list},
  };

  auto slice_desc = OpDescUtils::GetOpDescFromOperator(op);
  bool all_const = true;
  for (auto& item: const_values) {
    // avoid null input error when call GetInputConstData
    if (slice_desc->MutableInputDesc(item.first) != nullptr && \
        op.GetInputConstData(item.first, const_tensor) == GRAPH_SUCCESS) {
      item.second.clear();
      auto dtype = op.GetInputDesc(item.first).GetDataType();
      GetConstValue(op, const_tensor, dtype, item.second);
    } else {
      OP_LOGI(op.GetName().c_str(), "[%s] is not constant.", item.first.c_str());
      all_const = false;
    }
  }

  if (!all_const) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// ----------------StridedSliceV2 Op Begin-----------------
IMPLEMT_COMMON_INFERFUNC(StridedSliceV2InferShape) {
  const int64_t MAX_INT = ((uint32_t)(-1))>>1;
  const int64_t MIN_INT = ~MAX_INT;
  vector<string> depend_names = {"begin", "end"};
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  if (op_info->MutableInputDesc("axes") != nullptr) {
    depend_names.push_back("axes");
  }
  if (op_info->MutableInputDesc("strides") != nullptr) {
    depend_names.push_back("strides");
  }
  PREPARE_DYNAMIC_SHAPE(depend_names);

  // Get input shape
  auto input_desc = op.GetInputDesc("x");
  const ge::Shape shape = input_desc.GetShape();
  DataType input_dtype = input_desc.GetDataType();
  int64_t begin_len = -1;
  for (const auto& param : depend_names) {
    begin_len = std::max(op.GetInputDesc(param).GetShape().GetDim(0), begin_len);
  }

  // check the ranks and get the len of final end list len start 
  /* shape must be same with input ranks */
  size_t rank_num = shape.GetDims().size();
  // begin_len = std::max(static_cast<int64_t>(rank_num), begin_len);
  /* read the axes values from const tensor */

  // check the ranks and get the len of final end list len end

  // Get 'begin_list','end_list','stride_list' from const node
  struct SliceParameters slice_params = {};
  std::vector<int64_t> axes_list;
  bool begin_valid = true;
  bool end_valid = true;
  bool stride_valid = true;
  bool axes_valid = true;
  if (GRAPH_FAILED == GetStridedSliceInferConstData(op, slice_params, axes_list)) {
    OP_LOGI(op.GetName().c_str(),
            "[begin,end,stride] are not all constant, set to tmp values for inference dynamic shape");
    begin_valid = !slice_params.begin_list.empty();
    end_valid = !slice_params.end_list.empty();
    stride_valid = !slice_params.stride_list.empty();
    axes_valid = !axes_list.empty();
  }

  OP_LOGD(op.GetName().c_str(), "input stride_valid:%d", stride_valid);
  OP_LOGD(op.GetName().c_str(), "input begin_len:%lld", begin_len);
  if(!stride_valid && begin_len>0){
    auto op_info = OpDescUtils::GetOpDescFromOperator(op);
    if(op_info->MutableInputDesc("strides") == nullptr){
      stride_valid = true;
      slice_params.stride_list.assign(begin_len, 1);
    }
  }
  OP_LOGD(op.GetName().c_str(), "input stride_list:%s", to_string(slice_params.stride_list).c_str());
  
  if (shape.GetDims() == UNKNOWN_RANK) {
    TensorDesc output_desc = op.GetOutputDesc("y");
    output_desc.SetDataType(input_dtype);
    ge::Shape outputShape = ge::Shape(UNKNOWN_RANK);
    output_desc.SetShape(outputShape);
    OP_LOGD(op.GetName().c_str(), "output_shape:%s", to_string(output_desc.GetShape()).c_str());
    (void) op.UpdateOutputDesc("y", output_desc);
    return GRAPH_SUCCESS;
  } else if(begin_len == -1 || !stride_valid){
    TensorDesc output_desc = op.GetOutputDesc("y");
    output_desc.SetDataType(input_dtype);
    ge::Shape outputShape = ge::Shape(std::vector<int64_t>(rank_num,-1));
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
      for (size_t i = shape_dims.size(); i < static_cast<size_t>(begin_len); i++) {
        slice_params.end_list.push_back(shape_dims.back());
      }
    }
  }

  // If stride is invalid, set stride with begin_len count of 1, for inference output ranges.
  // For example, begin_len is 2 set stride's value to [1, 1]
  if (!stride_valid) {
    slice_params.stride_list.assign(begin_len, 1);
  }

  // process end list and begin list accoring to the axes values start
  uint64_t axes_mask = 0;
  uint64_t ends_mask = 0;
  if(axes_valid){
    // clamp x to [MIN_INT ~ MAX_INT]
    auto clamp = [=](int64_t x) {
      return x = x > MAX_INT ? MAX_INT : (x < MIN_INT ? MIN_INT : x);
    };
    // pre fill the values to the vector
    std::vector<int64_t> processed_begin(rank_num, 0);
    std::vector<int64_t> processed_end = shape.GetDims();
    std::vector<int64_t> processed_stride(rank_num, 1);
    // fill the begin end accoring to the axes values 
    for (size_t i = 0; i < axes_list.size(); ++i){
      int64_t axes_index = axes_list[i];
      // negative axes index
      if(axes_index < 0){
        axes_list[i] = axes_index + rank_num;
      }
      // axes out of boundary
      if(axes_index >= static_cast<int64_t>(rank_num)){
        axes_index = rank_num - 1;
        OP_LOGD(op.GetName().c_str(), "Pos Axes Value Out Of Boudary:%s", to_string(axes_list).c_str());
      }
      // axes INT_MIN??? need to process
      if(axes_index < 0){
        axes_index = 0;
        OP_LOGD(op.GetName().c_str(), "Neg Value Out Of Boudary:%s", to_string(axes_list).c_str());
      }
      axes_mask = (1 << axes_index) | axes_mask;
      // clamp value to int32 range that avoid memory alloc fail
      processed_end[axes_index] = clamp(slice_params.end_list[i]);
      processed_begin[axes_index] = clamp(slice_params.begin_list[i]);
      processed_stride[axes_index] = slice_params.stride_list[i];
    }
    // assign the proceseed value back to slice params
    axes_mask = ~axes_mask;
    ends_mask = axes_mask;
    constexpr int64_t MAX_INT64 = ((uint64_t)(-1)) >> 1;
    for (size_t i = 0; i < processed_end.size(); ++i) {
      if (processed_end[i] == MAX_INT64 || processed_end[i] == MAX_INT) {
        ends_mask = (1 << i) | ends_mask;
      }
    }
    slice_params.begin_list.assign(processed_begin.begin(),processed_begin.end());
    slice_params.end_list.assign(processed_end.begin(),processed_end.end());
    slice_params.stride_list.assign(processed_stride.begin(),processed_stride.end());
  }
  //process end list and begin list accoring to the axes values end

  vector<pair<int64_t,int64_t>> input_ranges;
  input_desc.GetShapeRange(input_ranges);
  if (input_ranges.empty()) {
    MakeUpShapeRange(shape.GetDims(), input_ranges);
  }
  size_t dim_num = shape.GetDimNum();

  if (dim_num == 0) {
    std::string err_msg = GetParamOutRangeErrMsg("input x's dimnum", ConcatString("[", DIM_SIZE1, ", ", DIM_SIZE8, "]"),  ConcatString(dim_num));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  OP_LOGD(op.GetName().c_str(), "begin_list:%s", to_string(slice_params.begin_list).c_str());
  OP_LOGD(op.GetName().c_str(), "end_list:%s", to_string(slice_params.end_list).c_str());
  OP_LOGD(op.GetName().c_str(), "stride_list:%s", to_string(slice_params.stride_list).c_str());
  if (slice_params.end_list.size() != slice_params.begin_list.size()) {
    std::string err_msg = OtherErrMsg("end shape, begin shape length mismatch!");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  op.SetAttr("begin", slice_params.begin_list);
  op.SetAttr("end", slice_params.end_list);
  op.SetAttr("strides", slice_params.stride_list);

  // Get relevant masks from const node,  setting all mask to 0 
  struct SliceMasks slice_masks;

  StridedSliceParams input_params = {
      shape.GetDims(),
      slice_params.begin_list,
      slice_params.end_list,
      slice_params.stride_list,
      input_ranges,
      axes_mask,
      ends_mask,
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

COMMON_INFER_FUNC_REG(StridedSliceV2, StridedSliceV2InferShape);
INFER_VALUE_RANGE_DEFAULT_REG(StridedSliceV2);
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
    OP_LOGE(op.GetName().c_str(), "stride[%ld] error.", stride);
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

  // shape of output y is the same as input x
  ge::Shape shape_input = op.GetInputDesc(input_name).GetShape();
  v_output_desc.SetShape(shape_input);
  v_output_desc.SetDataType(input_dtype);
  v_output_desc.SetFormat(input_format);
  op.UpdateOutputDesc(output_name, v_output_desc);

  return true;
}

IMPLEMT_VERIFIER(IndexFillD, IndexFillDVerify)
{
  std::vector<std::string> vec_input;
  vec_input.push_back("x");
  vec_input.push_back("assist1");
  vec_input.push_back("assist2");
  // check whether the dtype of x and assist1 is the same
  if  (!CheckInputDtypeSame(op, vec_input)){
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

// Registered inferfunction
COMMON_INFER_FUNC_REG(IndexFillD, IndexFillDInferShape);

// Registered verify function
VERIFY_FUNC_REG(IndexFillD, IndexFillDVerify);
// ----------------IndexFillD END-------------------

// ----------------AddRowRanges Begin-------------------
bool InferShapeAndTypeAddRowRanges(Operator& op, const string& input_name1,
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

IMPLEMT_VERIFIER(AddRowRanges, AddRowRangesVerify) {
  DataType x_dtype = op.GetInputDesc("x").GetDataType();
  DataType indices_dtype = op.GetInputDesc("indices").GetDataType();
  DataType src_dtype = op.GetInputDesc("src").GetDataType();
  DataType x_out_dtype = op.GetOutputDesc("x").GetDataType();
  if (x_dtype != x_out_dtype || x_dtype != src_dtype) {
    OP_LOGE(op.GetName().c_str(),
            "x dtype is not equal to src dtype, please check!");
    return GRAPH_FAILED;
  }
  if (indices_dtype != DT_INT32) {
    OP_LOGE(op.GetName().c_str(), "indices dtype is not int32, please check!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AddRowRangesInferShape) {
  if (InferShapeAndTypeAddRowRanges(op, "x", "x")) {
    return GRAPH_SUCCESS;
  }
  OP_LOGE(op.GetName().c_str(), "infer shape failed!");
  return GRAPH_FAILED;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(AddRowRanges, AddRowRangesInferShape);
// Registered verify function
VERIFY_FUNC_REG(AddRowRanges, AddRowRangesVerify);
// ----------------AddRowRanges END---------------------

// ----------------MaskedFillRange Begin-------------------
IMPLEMT_COMMON_INFERFUNC(MaskedFillRangeInferShape)
{
  auto input_shape = op.GetInputDesc("x").GetShape();
  auto input_type = op.GetInputDesc("x").GetDataType();
  auto input_dim = input_shape.GetDims().size();

  std::vector<int64_t> vec_dims;
  for (size_t i = 0; i < input_dim; i++) {
    vec_dims.push_back(input_shape.GetDim(static_cast<int64_t>(i)));
  }

  ge::Shape output_shape(vec_dims);

  auto output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(output_shape);
  output_desc.SetDataType(ge::DataType(input_type));
  (void)op.UpdateOutputDesc("y", output_desc);

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MaskedFillRange, MaskedFillRangeVerify)
{
  TensorDesc x_desc = op.get_input_desc_x();
  auto data_type_x = x_desc.GetDataType();
  TensorDesc value_desc = op.get_input_desc_value();
  auto data_type_value = value_desc.GetDataType();
  if (data_type_x != data_type_value) {
    OP_LOGE("MaskedFillRange", "input x and value date type must be equal!");
    return GRAPH_FAILED;
  }

  TensorDesc start_desc = op.get_input_desc_start();
  DataType data_type_start = start_desc.GetDataType();
  TensorDesc end_desc = op.get_input_desc_end();
  DataType data_type_end = end_desc.GetDataType();
  if (data_type_start != data_type_end) {
    OP_LOGE("MaskedFillRange", "input start and end date type must be equal!");
    return GRAPH_FAILED;
  }

  int64_t x_dim = x_desc.GetShape().GetDimNum();
  int64_t axis = std::abs(op.get_attr_axis());
  if (axis >= x_dim) {
    OP_LOGE("MaskedFillRange", "axis is larger than input x dimensions!");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(MaskedFillRange, MaskedFillRangeInferShape);
VERIFY_FUNC_REG(MaskedFillRange, MaskedFillRangeVerify);
// ----------------MaskedFillRange END---------------------

// ----------------InplaceTopKDistance Begin-------------------
IMPLEMT_COMMON_INFERFUNC(InplaceTopKDistanceInferShape) {
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(InplaceTopKDistance, InplaceTopKDistanceVerify) {
  std::vector<int64_t> topk_pq_distance_dims = op.GetInputDesc("topk_pq_distance").GetShape().GetDims();
  std::vector<int64_t> topk_pq_index_dims = op.GetInputDesc("topk_pq_index").GetShape().GetDims();
  std::vector<int64_t> topk_pq_ivf_dims = op.GetInputDesc("topk_pq_ivf").GetShape().GetDims();
  if (!(topk_pq_distance_dims == topk_pq_index_dims && topk_pq_index_dims == topk_pq_ivf_dims)) {
    string msg = ConcatString("The shape of topk_pq_distance is:", DebugString(topk_pq_distance_dims),
                              "The shape of topk_pq_index is:", DebugString(topk_pq_index_dims),
                              "The shape of topk_pq_ivf is:", DebugString(topk_pq_ivf_dims), ". They must be same");
    std::string err_msg = OtherErrMsg(msg);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (topk_pq_distance_dims != UNKNOWN_RANK && topk_pq_distance_dims.size() != 1) {
    string msg = ConcatString("The shape of topk_pq_distance is:", DebugString(topk_pq_distance_dims),
                              "The shape of topk_pq_index is:", DebugString(topk_pq_index_dims),
                              "The shape of topk_pq_ivf is:", DebugString(topk_pq_ivf_dims), ". Dim must be 1");
    std::string err_msg = OtherErrMsg(msg);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> pq_distance_dims = op.GetInputDesc("pq_distance").GetShape().GetDims();
  std::vector<int64_t> pq_index_dims = op.GetInputDesc("pq_index").GetShape().GetDims();
  if (!(pq_distance_dims == pq_index_dims)) {
    string msg = ConcatString("The shape of pq_distance is:", DebugString(topk_pq_distance_dims),
                              "The shape of pq_index is:", DebugString(topk_pq_index_dims),". They must be same");
    std::string err_msg = OtherErrMsg(msg);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (pq_distance_dims != UNKNOWN_RANK && pq_distance_dims.size() != 1) {
    string msg = ConcatString("The shape of pq_distance is:", DebugString(topk_pq_distance_dims),
                              "The shape of pq_index is:", DebugString(topk_pq_index_dims), ". Dim must be 1");
    std::string err_msg = OtherErrMsg(msg);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  OP_LOGE(op.GetName().c_str(), "GRAPH_SUCCESS.");
  return GRAPH_SUCCESS;
}
// Registered inferfunction
COMMON_INFER_FUNC_REG(InplaceTopKDistance, InplaceTopKDistanceInferShape);
// Registered verify function
VERIFY_FUNC_REG(InplaceTopKDistance, InplaceTopKDistanceVerify);
// ----------------InplaceTopKDistance END---------------------

// -----------------TopKPQDistanceMerge Begin----------------------
IMPLEMT_INFERFUNC(TopKPQDistanceMerge, TopKPQDistanceMergeInferShape) {
  int32_t topK = 0;
  if(op.GetAttr("k", topK) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get attr k from op failed");
    return GRAPH_FAILED;
  }
  ge::TensorDesc inputDistanceTensorDesc = op.GetInputDescByName("sorted_distance");
  ge::TensorDesc inputPqivfTensorDesc = op.GetInputDescByName("pq_ivf");
  ge::TensorDesc inputPqindexTensorDesc = op.GetInputDescByName("pq_index");
  DataType distanceDtype = inputDistanceTensorDesc.GetDataType();
  DataType pqIvfDtype = inputPqivfTensorDesc.GetDataType();
  DataType pqIndexDtype = inputPqindexTensorDesc.GetDataType();
 
  vector<int64_t> outputDims = {topK};

  ge::TensorDesc outputDistanceDesc = op.GetOutputDescByName("topk_distance");
  ge::TensorDesc outputIvfDesc = op.GetOutputDescByName("topk_ivf");
  ge::TensorDesc outputIndexDesc = op.GetOutputDescByName("topk_index");
  outputDistanceDesc.SetShape(ge::Shape(outputDims));
  outputIvfDesc.SetShape(ge::Shape(outputDims));
  outputIndexDesc.SetShape(ge::Shape(outputDims));
  outputDistanceDesc.SetDataType(distanceDtype);
  outputIvfDesc.SetDataType(pqIvfDtype);
  outputIndexDesc.SetDataType(pqIndexDtype);
 
  CHECK(op.UpdateOutputDesc("topk_distance", outputDistanceDesc) != GRAPH_SUCCESS,
    OP_LOGE(op.GetName().c_str(), "Update topk_distance outputDesc failed."),
    return GRAPH_FAILED
  );
  CHECK(op.UpdateOutputDesc("topk_ivf", outputIvfDesc) != GRAPH_SUCCESS,
    OP_LOGE(op.GetName().c_str(), "Update topk_ivf outputDesc failed."),
    return GRAPH_FAILED
  );
  CHECK(op.UpdateOutputDesc("topk_index", outputIndexDesc) != GRAPH_SUCCESS,
    OP_LOGE(op.GetName().c_str(), "Update topk_index outputDesc failed."),
    return GRAPH_FAILED
  );
 
  return GRAPH_SUCCESS;
 
}
 
IMPLEMT_VERIFIER(TopKPQDistanceMerge, TopKPQDistanceMergeVerify) {
  const int32_t maxK = 1024;
  std::vector<int64_t> sortedDistanceDims = op.GetInputDescByName("sorted_distance").GetShape().GetDims();
  std::vector<int64_t> pqIvfDims = op.GetInputDescByName("pq_ivf").GetShape().GetDims();
  std::vector<int64_t> pqIndexDims = op.GetInputDescByName("pq_index").GetShape().GetDims();
 
  if (!(sortedDistanceDims == pqIvfDims && pqIvfDims == pqIndexDims)) {
    string msg = ConcatString("The shape of sorted_distance is:", DebugString(sortedDistanceDims),
                              "The shape of pq_ivf is:", DebugString(pqIvfDims),
                              "The shape of pq_index is:", DebugString(pqIndexDims), ".They must be the same");
    std::string  err_msg = OtherErrMsg(msg);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int32_t topK = 0;
  if (op.GetAttr("k", topK) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get attr k from op failed");
    return GRAPH_FAILED;
  }
  if (topK > maxK) {
    string correctValue = ConcatString("not greater than 1024");
    std::string errMsg = GetAttrValueErrMsg("k", ConcatString(topK), correctValue);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), errMsg);
    return GRAPH_FAILED; 
  }
  return GRAPH_SUCCESS;
}
// Registered infershape function 
INFER_FUNC_REG(TopKPQDistanceMerge, TopKPQDistanceMergeInferShape);
// Registered verify function 
VERIFY_FUNC_REG(TopKPQDistanceMerge, TopKPQDistanceMergeVerify);
// -----------------TopKPQDistanceMerge END----------------------

// ----------------StridedSlicev3 Op Begin-------------------
IMPLEMT_COMMON_INFERFUNC(StridedSliceV3InferShape) {
  vector<string> depend_names = {"begin", "end"};
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  if (op_info->MutableInputDesc("axes") != nullptr) {
    depend_names.push_back("axes");
  }
  if (op_info->MutableInputDesc("strides") != nullptr) {
    depend_names.push_back("strides");
  }
  PREPARE_DYNAMIC_SHAPE(depend_names);

  // Get input shape
  auto input_desc = op.GetInputDesc("x");
  const ge::Shape shape = input_desc.GetShape();
  DataType input_dtype = input_desc.GetDataType();
  int64_t begin_len = -1;
  for (const auto& param : depend_names) {
    begin_len = std::max(op.GetInputDesc(param).GetShape().GetDim(0), begin_len);
  }

  // check the ranks and get the len of final end list len start 
  /* shape must be same with input ranks */
  size_t rank_num = shape.GetDims().size();

  // Get 'begin_list','end_list', 'axis_list', 'stride_list' from const node, if exist.
  struct SliceParameters slice_params = {};
  std::vector<int64_t> input_axes_values;
  bool begin_valid = true;
  bool end_valid = true;
  bool stride_valid = true;
  bool axes_valid = true;
  if (GRAPH_FAILED == GetStridedSliceInferConstData(op, slice_params, input_axes_values)) {
    OP_LOGI(op.GetName().c_str(),
            "[begin,end,axis,stride] are not all constant, set to tmp values for inference dynamic shape");
    begin_valid = !slice_params.begin_list.empty();
    end_valid = !slice_params.end_list.empty();
    stride_valid = !slice_params.stride_list.empty();
    axes_valid = !input_axes_values.empty();
  }

  OP_LOGD(op.GetName().c_str(), "input stride_valid:%d", stride_valid);
  OP_LOGD(op.GetName().c_str(), "input begin_len:%lld", begin_len);
  if(!stride_valid && begin_len>0){
    auto op_info = OpDescUtils::GetOpDescFromOperator(op);
    if(op_info->MutableInputDesc("strides") == nullptr){
      stride_valid = true;
      slice_params.stride_list.assign(begin_len, 1);
    }
  }
  OP_LOGD(op.GetName().c_str(), "input stride_list:%s", to_string(slice_params.stride_list).c_str());
  
  if (shape.GetDims() == UNKNOWN_RANK) {
    TensorDesc output_desc = op.GetOutputDesc("y");
    output_desc.SetDataType(input_dtype);
    ge::Shape outputShape = ge::Shape(UNKNOWN_RANK);
    output_desc.SetShape(outputShape);
    OP_LOGD(op.GetName().c_str(), "output_shape:%s", to_string(output_desc.GetShape()).c_str());
    (void) op.UpdateOutputDesc("y", output_desc);
    return GRAPH_SUCCESS;
  }else if(begin_len == -1 || !stride_valid){
    TensorDesc output_desc = op.GetOutputDesc("y");
    output_desc.SetDataType(input_dtype);
    ge::Shape outputShape = ge::Shape(std::vector<int64_t>(rank_num,-1));
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
      for (size_t i = shape_dims.size(); i < static_cast<size_t>(begin_len); i++) {
        slice_params.end_list.push_back(shape_dims.back());
      }
    }
  }

  // If stride is invalid, set stride with begin_len count of 1, for inference output ranges.
  // For example, begin_len is 2 set stride's value to [1, 1]
  if (!stride_valid) {
    slice_params.stride_list.assign(begin_len, 1);
  }

  // process end list and begin list accoring to the axes values start
  uint64_t axes_mask = 0;
  uint64_t ends_mask = 0;
  if (axes_valid) {
    // pre fill the values to the vector
    std::vector<int64_t> processed_begin(rank_num, 0);
    std::vector<int64_t> processed_end = shape.GetDims();
    std::vector<int64_t> processed_stride(rank_num, 1);
    // fill the begin end accoring to the axes values 
    for (size_t i = 0; i < input_axes_values.size(); ++i){
      int64_t axes_index = input_axes_values[i];
      // negative axes index
      if(axes_index < 0){
        input_axes_values[i] = axes_index + rank_num;
      }
      // axes out of boundary
      if(axes_index >= static_cast<int64_t>(rank_num)){
        axes_index = rank_num - 1;
        OP_LOGD(op.GetName().c_str(), "Pos Axes Value Out Of Boudary:%s", to_string(input_axes_values).c_str());
      }
      // axes INT_MIN??? need to process
      if(axes_index < 0){
        axes_index = 0;
        OP_LOGD(op.GetName().c_str(), "Neg Value Out Of Boudary:%s", to_string(input_axes_values).c_str());
      }
      axes_mask = (1<<axes_index)|axes_mask;
      processed_end[axes_index] = slice_params.end_list[i];
      processed_begin[axes_index] = slice_params.begin_list[i];
      processed_stride[axes_index] = slice_params.stride_list[i];
    }
    // assign the proceseed value back to slice params
    axes_mask = ~axes_mask;
    ends_mask = axes_mask;
    constexpr int64_t MAX_INT64 = ((uint64_t)(-1)) >> 1;
    for (size_t i = 0; i < processed_end.size(); ++i) {
      if (processed_end[i] == MAX_INT64) {
        ends_mask = (1 << i) | ends_mask;
      }
    }
    slice_params.begin_list.assign(processed_begin.begin(),processed_begin.end());
    slice_params.end_list.assign(processed_end.begin(),processed_end.end());
    slice_params.stride_list.assign(processed_stride.begin(),processed_stride.end());
  }
  // process end list and begin list accoring to the axes values end

  vector<pair<int64_t,int64_t>> input_ranges;
  input_desc.GetShapeRange(input_ranges);
  if (input_ranges.empty()) {
    MakeUpShapeRange(shape.GetDims(), input_ranges);
  }

  size_t dim_num = shape.GetDimNum();

  if (dim_num == 0) {
    std::string err_msg = GetParamOutRangeErrMsg("input x's dimnum", ConcatString("[", DIM_SIZE1, ", ", DIM_SIZE8, "]"),  ConcatString(dim_num));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  OP_LOGD(op.GetName().c_str(), "begin_list:%s", to_string(slice_params.begin_list).c_str());
  OP_LOGD(op.GetName().c_str(), "end_list:%s", to_string(slice_params.end_list).c_str());
  OP_LOGD(op.GetName().c_str(), "stride_list:%s", to_string(slice_params.stride_list).c_str());
  if (slice_params.end_list.size() != slice_params.begin_list.size()) {
    std::string err_msg = OtherErrMsg("end shape, begin shape length mismatch!");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // Get relevant masks from const node,  setting all mask to 0 
  struct SliceMasks slice_masks;

  StridedSliceParams input_params = {
      shape.GetDims(),
      slice_params.begin_list,
      slice_params.end_list,
      slice_params.stride_list,
      input_ranges,
      axes_mask,
      ends_mask,
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

COMMON_INFER_FUNC_REG(StridedSliceV3, StridedSliceV3InferShape);
INFER_VALUE_RANGE_DEFAULT_REG(StridedSliceV3);
// ----------------StridedSlicev3 Op End-------------------

// ----------------MovingSumWithSigmoidInferShape Begin-------------------
IMPLEMT_COMMON_INFERFUNC(MovingSumWithSigmoidInferShape) {
  OP_LOGD(op.GetName().c_str(), "MovingSumWithSigmoidInferShape start.");
  const vector<string> const_names = {"offset"};
  PREPARE_DYNAMIC_SHAPE(const_names);
  auto ms_op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto input_energy_desc = ms_op_desc->GetInputDescPtr(1);
  auto output_y_desc = ms_op_desc->MutableOutputDesc(0);
  auto energy_dtype = input_energy_desc->GetDataType();
  output_y_desc->SetDataType(energy_dtype);

  auto input_offset_desc = ms_op_desc->GetInputDescPtr(2);
  int64_t batch_size = input_offset_desc->GetShape().GetDim(0) / 2;

  GeShape &output_shape = output_y_desc->MutableShape();
  output_shape.SetDimNum(2);
  auto energy_shape = input_energy_desc->GetShape();
  const GeTensor *data  = OpDescUtils::GetInputConstData(op, 2);
  if (data != nullptr) {
    int64_t beam_sum = 0;
    int64_t frame_sum = 0;
    const int32_t* offset_data = reinterpret_cast<const int32_t*>(data->GetData().GetData());
    for (int64_t i = 0; i < batch_size; i++) {
      beam_sum += offset_data[i];
      frame_sum += offset_data[i + batch_size];
    }
    output_shape.SetDim(0, beam_sum);
    output_shape.SetDim(1, frame_sum);
  } else {
    OP_LOGD(op.GetName().c_str(), "failed to get constValue of [offset].");
    std::vector<int64_t> energy_dims = energy_shape.GetDims();
    if (!IsUnknown(energy_dims)) {
      OP_LOGE(op.GetName().c_str(), "static case is not supported.");
      return GRAPH_FAILED;
    }
    output_shape.SetDim(0, -1);
    output_shape.SetDim(1, -1);

    std::vector<std::pair<int64_t, int64_t>> range_vector;
    range_vector.push_back(std::make_pair(1, -1));
    range_vector.push_back(std::make_pair(1, -1));
    output_y_desc->SetShapeRange(range_vector);
  }

  OP_LOGD(op.GetName().c_str(), "MovingSumWithSigmoidInferShape finish.");
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(MovingSumWithSigmoid, MovingSumWithSigmoidInferShape);
// ----------------MovingSumWithSigmoidInferShape END---------------------

// ------------DynSeqOuter------------------------
IMPLEMT_INFERFUNC(DynSeqOuter, DynSeqOuterInferShape) {
  OP_LOGD(op.GetName().c_str(), "DynSeqOuterInferShape start.");
  const vector<string> const_names = {"seq_len1", "seq_len2"};
  PREPARE_DYNAMIC_SHAPE(const_names);
  auto add_op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto input_energy_desc = add_op_desc->GetInputDescPtr(1);
  auto output_y_desc = add_op_desc->MutableOutputDesc(0);
  auto energy_dtype = input_energy_desc->GetDataType();
  output_y_desc->SetDataType(energy_dtype);

  auto seq_len1_desc = add_op_desc->GetInputDescPtr(2);
  int64_t batch_size = seq_len1_desc->GetShape().GetDim(0);

  GeShape &output_shape = output_y_desc->MutableShape();
  output_shape.SetDimNum(2);
  auto energy_shape = input_energy_desc->GetShape();
  output_shape.SetDim(1, energy_shape.GetDim(1));
  const GeTensor *data1  = OpDescUtils::GetInputConstData(op, 2);
  const GeTensor *data2  = OpDescUtils::GetInputConstData(op, 3);
  if (data1 != nullptr && data2 != nullptr) {
    int64_t bst = 0;
    const int32_t* seq_len1 = reinterpret_cast<const int32_t*>(data1->GetData().GetData());
    const int32_t* seq_len2 = reinterpret_cast<const int32_t*>(data2->GetData().GetData());
    for (int64_t i = 0; i < batch_size; i++) {
      bst += seq_len1[i] * seq_len2[i];
    }
    output_shape.SetDim(0, bst);
  } else {
    OP_LOGD(op.GetName().c_str(), "failed to get constValue of [seq_len1, seq_len2].");
    std::vector<int64_t> energy_dims = energy_shape.GetDims();
    if (!IsUnknown(energy_dims)) {
      OP_LOGE(op.GetName().c_str(), "static case is not supported.");
      return GRAPH_FAILED;
    }
    output_shape.SetDim(0, -1);
    std::vector<std::pair<int64_t, int64_t>> range_vector;
    range_vector.push_back(std::make_pair(1, -1));
    range_vector.push_back(std::make_pair(1, -1));
    output_y_desc->SetShapeRange(range_vector);
  }

  OP_LOGD(op.GetName().c_str(), "DynSeqOuterInferShape finish.");
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(DynSeqOuter, DynSeqOuterInferShape);
// ------------DynSeqOuter Op End-----------------

// ----------------MaskedSelect Begin-------------------
bool InferShapeAndTypeMaskedSelect(Operator& op) {
  OpDescPtr op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr x_input = op_desc->MutableInputDesc(0);
  GeShape x_shape = x_input->GetShape();
  GeTensorDescPtr y_desc = op_desc->MutableOutputDesc(0);
  DataType input_dtype = x_input->GetDataType();
  y_desc->SetDataType(input_dtype);
  std::vector<std::pair<int64_t, int64_t>> range;
  y_desc->SetShape(GeShape({UNKNOWN_DIM}));
  y_desc->SetOriginShape(GeShape({UNKNOWN_DIM}));
  range.emplace_back(std::make_pair(1, x_shape.GetShapeSize()));
  y_desc->SetShapeRange(range);
  return true;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(MaskedSelectInferShape) {
  if (InferShapeAndTypeMaskedSelect(op)) {
    return GRAPH_SUCCESS;
  }
  OP_LOGE(op.GetName().c_str(), "The shape of output y does not match that of x1 x2.");
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(MaskedSelect, MaskedSelectInferShape);
// ----------------MaskedSelect END---------------------
}  // namespace ge

