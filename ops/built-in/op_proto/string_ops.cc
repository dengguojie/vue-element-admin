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
 * \file string_ops.cpp
 * \brief
 */
#include "inc/string_ops.h"
#include "util/common_shape_fns.h"
#include "util/error_util.h"
#include "common_shape_fns.h"
#include "common/inc/op_log.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/operator.h"

namespace ge {
IMPLEMT_INFERFUNC(StringSplit, StringSplitInfer) {
  Shape unused;
  auto tensor_input = op.get_input_desc_input();
  auto tensor_sep = op.get_input_desc_delimiter();

  if (WithRank(tensor_input, 1, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(tensor_input.GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (WithRank(tensor_sep, 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1,
        DebugString(tensor_sep.GetShape().GetDims()), "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  Shape indices_shape;
  Shape values_shape;
  Shape shape;

  auto result = Matrix(ge::UNKNOWN_DIM, 2, indices_shape);
  if (result != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), 
        ConcatString("failed to call Matrix function to generate shape[",
        ge::UNKNOWN_DIM, ", 2] for output[indices]."));
    return GRAPH_FAILED;
  }
  TensorDesc indices_desc = op.get_output_desc_indices();
  indices_desc.SetShape(indices_shape);
  indices_desc.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("indices", indices_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[indices] desc failed."));
    return GRAPH_FAILED;
  }

  result = Vector(ge::UNKNOWN_DIM, values_shape);
  if (result != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), 
        ConcatString("failed to call Vector function to generate shape[",
        ge::UNKNOWN_DIM, "] for output[values]."));
    return GRAPH_FAILED;
  }
  TensorDesc values_desc = op.get_output_desc_values();
  values_desc.SetShape(values_shape);
  values_desc.SetDataType(DT_STRING);
  if (op.UpdateOutputDesc("values", values_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[values] desc failed."));
    return GRAPH_FAILED;
  }

  result = Vector(2, shape);
  if (result != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "generate shape failed !");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), 
        string("failed to call Vector function to generate shape[2] for output[shape]."));
    return GRAPH_FAILED;
  }
  TensorDesc shape_desc = op.get_output_desc_shape();
  shape_desc.SetShape(shape);
  shape_desc.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("shape", shape_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[shape] desc failed."));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StringSplit, StringSplitInfer);

IMPLEMT_INFERFUNC(StaticRegexReplace, StaticRegexReplaceInfer) {
  auto x_desc = op.GetInputDesc("input");
  DataType y_type = x_desc.GetDataType();
  if(y_type != DT_STRING) {
    std::string input_dt = DTypeStr(y_type);
    std::string err_msg = ConcatString("input[input] data type[", input_dt,"] must be DT_STRING");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  TensorDesc desc = op.GetOutputDesc("output");
  std::vector<std::pair<int64_t, int64_t>> range;
  if (x_desc.GetShapeRange(range) != GRAPH_SUCCESS) {
    desc.SetShapeRange(range);
  }
  desc.SetShape(op.GetInputDesc("input").GetShape());
  desc.SetDataType(y_type);

  (void)op.UpdateOutputDesc("output", desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StaticRegexReplace, StaticRegexReplaceInfer);

IMPLEMT_INFERFUNC(StaticRegexFullMatch, StaticRegexFullMatchInfer) {
  auto x_desc = op.GetInputDesc("input");
  DataType x_type = x_desc.GetDataType();
  if(x_type != DT_STRING) {
    std::string input_dt = DTypeStr(x_type);
    std::string err_msg = ConcatString("input[input] data type[", input_dt,"] must be DT_STRING");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  TensorDesc desc = op.GetOutputDesc("output");
  std::vector<std::pair<int64_t, int64_t>> range;
  if (x_desc.GetShapeRange(range) != GRAPH_SUCCESS) {
    desc.SetShapeRange(range);
  }
  desc.SetShape(op.GetInputDesc("input").GetShape());
  desc.SetDataType(DT_BOOL);

  (void)op.UpdateOutputDesc("output", desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StaticRegexFullMatch, StaticRegexFullMatchInfer);

IMPLEMT_INFERFUNC(UnsortedSegmentJoin, UnsortedSegmentJoinInfer) {
  OP_LOGI(op.GetName().c_str(), "Enter UnsortedSegmentJoin proto inferfunction!");
  Shape x_shape = op.GetInputDesc("input").GetShape();
  Shape segment_ids_shape = op.GetInputDesc("segment_ids").GetShape();
  Shape num_segment_shape;
  if (WithRank(op.GetInputDesc("num_segments"), 0, num_segment_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call WithRank function, input[num_segments] rank must be 0, "
        "got rank[",
        num_segment_shape.GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  Shape out_shape;
  // update output desc
  if (RankKnown(segment_ids_shape)) {
    if (MergePrefix(x_shape, segment_ids_shape, x_shape, segment_ids_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString("Call MergePrefix function failed to merge input[input] shape[",
                                          DebugString(x_shape.GetDims()),"] of input[segment_ids] and shape[",
                                          DebugString(segment_ids_shape.GetDims()),"]");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }

    Tensor num_segments_tensor;
    auto result_x = op.GetInputConstData("num_segments", num_segments_tensor);
    if (result_x != GRAPH_SUCCESS) {
      Shape unknown_shape(ge::UNKNOWN_RANK);
      TensorDesc y_desc = op.GetOutputDesc("output");
      y_desc.SetShape(unknown_shape);
      y_desc.SetDataType(DT_STRING);
      (void)op.UpdateOutputDesc("output", y_desc);
      return GRAPH_SUCCESS;
    }
    int64_t num_segments_dim = 0;
    const int32_t *num_segments_data = reinterpret_cast<const int32_t *>(num_segments_tensor.GetData());
    if (num_segments_data == nullptr) {
      num_segments_dim = -1;
    } else {
      num_segments_dim = *num_segments_data;
      if (num_segments_dim < 0) {
        std::string err_msg = ConcatString("the input[num_segments] data must be non-negative, data[",
                           num_segments_dim, "]");
        AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
        return GRAPH_FAILED;
      }
    }
    Shape s_data_suffix;
    int64_t start = segment_ids_shape.GetDimNum();
    int64_t end = std::numeric_limits<int64_t>::max();
    int64_t stride = 1;
    if (SubShape(x_shape, start, end, stride, s_data_suffix, op.GetName().c_str()) != GRAPH_SUCCESS){
      std::string err_msg = ConcatString("failed to call SubShape function, input[input] shape[", 
                                          DebugString(x_shape.GetDims()), "], start[", start,
                                          "] or end[", end, "] or stride[", stride, "] is invaild");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    Concatenate(Shape({num_segments_dim}), s_data_suffix, out_shape);
  } else {
    Shape unknown_shape(ge::UNKNOWN_RANK);
    TensorDesc y_desc = op.GetOutputDesc("output");
    y_desc.SetShape(unknown_shape);
    y_desc.SetDataType(DT_STRING);
    op.UpdateOutputDesc("output", y_desc);
    return GRAPH_SUCCESS;
  }
  TensorDesc output_desc = op.GetOutputDesc("output");
  output_desc.SetShape(out_shape);
  output_desc.SetDataType(DT_STRING);
  (void)op.UpdateOutputDesc("output", output_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(UnsortedSegmentJoin, UnsortedSegmentJoinInfer);

IMPLEMT_INFERFUNC(StringLower, StringLowerInfer) {
  auto x_desc = op.GetInputDesc("input");
  DataType y_type = x_desc.GetDataType();
  if(y_type != DT_STRING) {
    std::string input_dt = DTypeStr(y_type);
    std::string err_msg = ConcatString("input[input] data type[", input_dt,"] must be DT_STRING");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  TensorDesc desc = op.GetOutputDesc("output");
  std::vector<std::pair<int64_t, int64_t>> range;
  if (x_desc.GetShapeRange(range) != GRAPH_SUCCESS) {
    desc.SetShapeRange(range);
  }
  desc.SetShape(op.GetInputDesc("input").GetShape());
  desc.SetDataType(y_type);

  (void)op.UpdateOutputDesc("output", desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StringLower, StringLowerInfer);

IMPLEMT_INFERFUNC(StringUpper, StringUpperInfer) {
  auto x_desc = op.GetInputDesc("input");
  DataType y_type = x_desc.GetDataType();
  if(y_type != DT_STRING) {
    std::string input_dt = DTypeStr(y_type);
    std::string err_msg = ConcatString("input[input] data type[", input_dt,"] must be DT_STRING");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  TensorDesc desc = op.GetOutputDesc("output");
  desc.SetShape(op.GetInputDesc("input").GetShape());
  desc.SetDataType(y_type);
  std::vector<std::pair<int64_t, int64_t>> range;
  if (x_desc.GetShapeRange(range) != GRAPH_SUCCESS) {
    desc.SetShapeRange(range);
  }

  (void)op.UpdateOutputDesc("output", desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StringUpper, StringUpperInfer);

IMPLEMT_INFERFUNC(StringSplitV2, StringSplitV2Infer) {
  Shape unused;
  auto tensor_input = op.get_input_desc_input();
  auto tensor_sep = op.get_input_desc_sep();

  if (WithRank(tensor_input, 1, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(tensor_input.GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (WithRank(tensor_sep, 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1,
        DebugString(tensor_sep.GetShape().GetDims()), "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  Shape indices_shape;
  Shape values_shape;
  Shape shape;

  auto result = Matrix(ge::UNKNOWN_DIM, 2, indices_shape);
  if (result != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), 
        ConcatString("failed to call Matrix function to generate shape[",
        ge::UNKNOWN_DIM, ", 2] of output[indices]."));
    return GRAPH_FAILED;
  }
  TensorDesc indices_desc = op.get_output_desc_indices();
  indices_desc.SetShape(indices_shape);
  indices_desc.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("indices", indices_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[indices] desc failed."));
    return GRAPH_FAILED;
  }

  result = Vector(ge::UNKNOWN_DIM, values_shape);
  if (result != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), 
        ConcatString("failed to call Vector function to generate shape[",
        ge::UNKNOWN_DIM, "] for output[values]."));
    return GRAPH_FAILED;
  }
  TensorDesc values_desc = op.get_output_desc_values();
  values_desc.SetShape(values_shape);
  values_desc.SetDataType(DT_STRING);
  if (op.UpdateOutputDesc("values", values_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[values] desc failed."));
    return GRAPH_FAILED;
  }

  result = Vector(2, shape);
  if (result != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), 
        string("failed to call Vector function to generate shape[2] for output[shape]."));
    return GRAPH_FAILED;
  }
  TensorDesc shape_desc = op.get_output_desc_shape();
  shape_desc.SetShape(shape);
  shape_desc.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("shape", shape_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[shape] desc failed."));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StringSplitV2, StringSplitV2Infer);

IMPLEMT_INFERFUNC(StringNGrams, StringNGramsInfer) {
  const char *op_name = op.GetName().c_str();
  Shape data;
  if (WithRank(op.GetInputDesc(0), 1, data, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(op.GetInputDesc(0).GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  Shape data_splits;
  if (WithRank(op.GetInputDesc(1), 1, data_splits, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1,
        DebugString(op.GetInputDesc(1).GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  Shape ngrams_shape = UnknownShapeOfRank(1);
  TensorDesc ngrams_desc = op.GetOutputDesc(0);
  ngrams_desc.SetShape(ngrams_shape);
  ngrams_desc.SetDataType(DT_STRING);
  ngrams_desc.SetShapeRange({std::pair<int64_t, int64_t>(1, -1)});
  if (op.UpdateOutputDesc("ngrams", ngrams_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[ngrams] desc failed."));
    return GRAPH_FAILED;
  }

  TensorDesc ngrams_splits_desc = op.GetOutputDesc(0);
  ngrams_splits_desc.SetShape(data_splits);
  ngrams_splits_desc.SetDataType(op.GetInputDesc(1).GetDataType());
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  (void) op.GetInputDesc(1).GetShapeRange(shape_range);
  ngrams_splits_desc.SetShapeRange(shape_range);
  if (op.UpdateOutputDesc("ngrams_splits", ngrams_splits_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[ngrams_splits] desc failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StringNGrams, StringNGramsInfer);

IMPLEMT_INFERFUNC(UnicodeDecodeWithOffsets, UnicodeDecodeWithOffsetsInfer) {
  int64_t input_size = op.GetInputDesc(0).GetShape().GetShapeSize();
  if (input_size == 0) {
    input_size = 1;
  }

  int64_t row_splits_num = 0;
  if (Add(input_size, 1, row_splits_num) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call Add function to add dim0[",
        input_size, "] of input[input] with 1.");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  DataType type = DT_FLOAT;
  if (op.GetAttr("Tsplits", type) != GRAPH_SUCCESS) {
    type = DT_INT64;
  }

  TensorDesc row_splits_desc = op.GetOutputDesc("row_splits");
  Shape row_splits_shape({row_splits_num});
  row_splits_desc.SetDataType(type);
  row_splits_desc.SetShape(row_splits_shape);
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  (void) op.GetInputDesc(0).GetShapeRange(shape_range);
  row_splits_desc.SetShapeRange(shape_range);

  if (op.UpdateOutputDesc("row_splits", row_splits_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[row_splits] desc failed."));
    return GRAPH_FAILED;
  }

  TensorDesc char_values_desc = op.GetOutputDesc("char_values");
  char_values_desc.SetDataType(DT_INT32);
  char_values_desc.SetShape(Shape({UNKNOWN_DIM}));
  char_values_desc.SetShapeRange({std::pair<int64_t, int64_t>(1, -1)});
  if (op.UpdateOutputDesc("char_values", char_values_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[char_values] desc failed."));
    return GRAPH_FAILED;
  }

  TensorDesc char_to_byte_starts_desc = op.GetOutputDesc("char_to_byte_starts");
  char_to_byte_starts_desc.SetDataType(DT_INT64);
  char_to_byte_starts_desc.SetShape(Shape({UNKNOWN_DIM}));
  char_to_byte_starts_desc.SetShapeRange({std::pair<int64_t, int64_t>(1, -1)});
  if (op.UpdateOutputDesc("char_to_byte_starts", char_to_byte_starts_desc)
        != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[char_to_byte_starts] desc failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(UnicodeDecodeWithOffsets, UnicodeDecodeWithOffsetsInfer);

IMPLEMT_INFERFUNC(UnicodeDecode, UnicodeDecodeInfer) {
  int64_t input_size = op.GetInputDesc(0).GetShape().GetShapeSize();
  if (input_size == 0) {
    input_size = 1;
  }

  int64_t row_splits_num = 0;
  if (Add(input_size, 1, row_splits_num) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call Add function to add dim0[",
        input_size, "] of input[input] with 1.");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  DataType type = DT_FLOAT;
  if (op.GetAttr("Tsplits", type) != GRAPH_SUCCESS) {
    type = DT_INT64;
  }

  TensorDesc row_splits_desc = op.GetOutputDesc("row_splits");
  Shape row_splits_shape({row_splits_num});
  row_splits_desc.SetDataType(type);
  row_splits_desc.SetShape(row_splits_shape);
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  (void) op.GetInputDesc(0).GetShapeRange(shape_range);
  row_splits_desc.SetShapeRange(shape_range);
  if (op.UpdateOutputDesc("row_splits", row_splits_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[row_splits] desc failed."));
    return GRAPH_FAILED;
  }

  TensorDesc char_values_desc = op.GetOutputDesc("char_values");
  char_values_desc.SetDataType(DT_INT32);
  char_values_desc.SetShape(Shape({UNKNOWN_DIM}));
  char_values_desc.SetShapeRange({std::pair<int64_t, int64_t>(1, -1)});
  if (op.UpdateOutputDesc("char_values", char_values_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[char_values] desc failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(UnicodeDecode, UnicodeDecodeInfer);

IMPLEMT_INFERFUNC(UnicodeTranscode, UnicodeTranscodeInfer) {
  DataType y_type = op.GetInputDesc(0).GetDataType();
  TensorDesc desc = op.GetOutputDesc("output");
  desc.SetShape(op.GetInputDesc(0).GetShape());
  desc.SetDataType(y_type);
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  (void) op.GetInputDesc(0).GetShapeRange(shape_range);
  desc.SetShapeRange(shape_range);

  if (op.UpdateOutputDesc("output", desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[output] desc failed."));
    return GRAPH_FAILED;
  }
  auto p_context = op.GetInferenceContext();
  if (p_context != nullptr) {
    const auto& shapes_and_types = p_context->GetInputHandleShapesAndTypes();
    if (!shapes_and_types.empty()) {
      p_context->SetOutputHandleShapesAndTypes(shapes_and_types);
    }
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(UnicodeTranscode, UnicodeTranscodeInfer);

IMPLEMT_INFERFUNC(UnicodeEncode, UnicodeEncodeInfer) {
  const char *op_name = op.GetName().c_str();

  Shape input_values_shape;
  if (WithRank(op.GetInputDesc(0), 1, input_values_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(op.GetInputDesc(0).GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  Shape splits_shape;
  if (WithRank(op.GetInputDesc(1), 1, splits_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1,
        DebugString(op.GetInputDesc(1).GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<int64_t> output_dims(1);
  if (Subtract(splits_shape.GetDim(0), 1, output_dims[0], op_name) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call Subtract function to substract dim0[",
        splits_shape.GetDim(0), "] of input[input_splits] with 1.");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc desc = op.GetOutputDesc("output");
  desc.SetShape(Shape(output_dims));
  desc.SetDataType(DT_STRING);
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  (void) op.GetInputDesc(1).GetShapeRange(shape_range);
  desc.SetShapeRange(shape_range);

  if (op.UpdateOutputDesc("output", desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[output] desc failed."));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(UnicodeEncode, UnicodeEncodeInfer);

IMPLEMT_INFERFUNC(UnicodeScript, UnicodeScriptInfer) {
  DataType y_type = op.GetInputDesc("x").GetDataType();
  TensorDesc desc = op.GetOutputDesc("y");
  desc.SetShape(op.GetInputDesc("x").GetShape());
  desc.SetDataType(y_type);

  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[y] desc failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(UnicodeScript, UnicodeScriptInfer);

IMPLEMT_INFERFUNC(Substr, SubstrInfer) {
  auto pos_tensor = op.GetInputDesc(1);
  Shape pos_shape = op.GetInputDesc(1).GetShape();
  Shape len_shape = op.GetInputDesc(2).GetShape();
  Shape unused;
  if (WithRank(pos_tensor, len_shape.GetDimNum(), unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("input[pos] rank value[", pos_shape.GetDimNum(),
        "] must same as input[len] rank value[", len_shape.GetDimNum(), "]");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  for (size_t i = 0; i < pos_shape.GetDimNum(); ++i) {
    auto pos_dim = pos_shape.GetDim(i);
    auto len_dim = len_shape.GetDim(i);
    if (pos_dim != len_dim) {
      std::string err_msg = ConcatString("input[pos] dim[", i,
          "] has wrong value[", pos_dim, "], it must same with input[len] dim[", i,
          "] value[", len_dim,"]");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }

  TensorDesc desc = op.GetOutputDesc(0);
  desc.SetDataType(DT_STRING);
  if (op.UpdateOutputDesc("output", desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[output] desc failed."));
    return GRAPH_FAILED;
  }

  auto outputFunc = BROADCAST_INFER("input", "pos", "output");
  return outputFunc(op);
}

INFER_FUNC_REG(Substr, SubstrInfer);

IMPLEMT_INFERFUNC(StringToHashBucketFast, StringToHashBucketFastInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto y_desc = op_desc->MutableOutputDesc(0);

  GeShape y_shape(op_desc->MutableInputDesc(0)->GetShape());
  if (!ShapeFullyDefined(y_shape)) {
    std::vector<std::pair<int64_t, int64_t>> y_range;
    for (const int64_t& y_dim : y_shape.GetDims()) {
      y_range.push_back(y_dim == UNKNOWN_DIM ? std::pair<int64_t, int64_t>{1, -1} :
                                               std::pair<int64_t, int64_t>{y_dim, y_dim});
    }
    y_desc->SetShapeRange(y_range);
  }
  y_desc->SetShape(y_shape);
  y_desc->SetDataType(DT_INT64);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StringToHashBucketFast, StringToHashBucketFastInfer);

IMPLEMT_INFERFUNC(StringToHashBucketStrong, StringToHashBucketStrongInfer) {
  DataType x_type = op.GetInputDesc("x").GetDataType();
  if (x_type != DT_STRING) {
    std::string input_dt = DTypeStr(x_type);
    std::string err_msg = ConcatString("input[x] data type[", input_dt,"] must be DT_STRING");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_PARAM_INVALID;
  }
  TensorDesc desc = op.GetOutputDesc("y");
  desc.SetShape(op.GetInputDesc("x").GetShape());
  desc.SetDataType(DT_INT64);

  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[y] desc failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StringToHashBucketStrong, StringToHashBucketStrongInfer);

IMPLEMT_INFERFUNC(StringToHashBucket, StringToHashBucketInfer) {
  TensorDesc desc = op.GetOutputDesc("y");
  desc.SetShape(op.GetInputDesc(0).GetShape());
  desc.SetDataType(DT_INT64);

  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[y] desc failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StringToHashBucket, StringToHashBucketInfer);

IMPLEMT_INFERFUNC(StringStrip, StringStripInfer) {
  TensorDesc desc = op.GetOutputDesc("y");
  desc.SetShape(op.GetInputDesc(0).GetShape());
  desc.SetDataType(DT_STRING);

  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[y] desc failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StringStrip, StringStripInfer);

IMPLEMT_INFERFUNC(StringLength, StringLengthInfer) {
  TensorDesc desc = op.GetOutputDesc("y");
  desc.SetShape(op.GetInputDesc(0).GetShape());
  desc.SetDataType(DT_INT32);

  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[y] desc failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StringLength, StringLengthInfer);

IMPLEMT_INFERFUNC(StringJoin, StringJoinInfer) {
  size_t input_size = op.GetInputsSize();
  bool all_scalar = true;
  for (size_t i = 0; i < input_size; ++i) {
    if (op.GetInputDesc(i).GetShape().GetDimNum() != 0) {
      all_scalar = false;
    }
  }

  TensorDesc desc = op.GetOutputDesc("y");
  desc.SetDataType(DT_STRING);
  if (all_scalar) {
    desc.SetShape(Shape());
    if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[y] desc failed."));
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }

  Shape out(ge::UNKNOWN_SHAPE);
  for (size_t i = 0; i < input_size; ++i) {
    Shape input_shape = op.GetInputDesc(i).GetShape();
    if ((RankKnown(input_shape)) && (input_shape.GetDimNum() != 0)) {
      if (Merge(out, input_shape, out, op.GetName().c_str()) != GRAPH_SUCCESS) {
        string err_msg = ConcatString("failed to call Merge function to merge output[y] shape",
            DebugString(out.GetDims()), " and ", i, "th dynamic input[x] shape",
            DebugString(input_shape.GetDims()));
        AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
        return GRAPH_FAILED;
      }
    }
  }
  desc.SetShape(out);
  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[y] desc failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StringJoin, StringJoinInfer);

IMPLEMT_INFERFUNC(StringFormat, StringFormatInfer) {
  string template_attr;
  string placeholder_attr;
  if (op.GetAttr("template", template_attr) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("get attr[template] failed."));
    return GRAPH_FAILED;
  }
  if (op.GetAttr("placeholder", placeholder_attr) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("get attr[placeholder] failed."));
    return GRAPH_FAILED;
  }

  std::istringstream str_template(template_attr);
  std::string token;
  size_t pos = -1;
  size_t num_placeholders = 0;
  while (str_template >> token) {
    while ((pos = token.rfind(placeholder_attr)) != std::string::npos) {
      num_placeholders++;
      token.erase(pos, 1);
    }
  }

  if (op.GetInputsSize() != num_placeholders) {
    std::string err_msg = ConcatString("dynamic input[x] number ",
        op.GetInputsSize(), " must match number placeholders[",
        num_placeholders,"] in attr[template].");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc desc = op.GetOutputDesc("y");
  desc.SetDataType(DT_STRING);
  desc.SetShape(Shape());

  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[y] desc failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StringFormat, StringFormatInfer);

IMPLEMT_INFERFUNC(RegexFullMatch, RegexFullMatchInfer) {
  Shape un_used;
  if (WithRank(op.GetInputDesc("pattern"), 0, un_used, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("input[pattern] has wrong shape",
      DebugString(op.GetInputDesc("pattern").GetShape().GetDims()),
      ", it should be scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  Shape x_shape = op.GetInputDesc("x").GetShape();
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(x_shape);
  y_desc.SetDataType(DT_BOOL);
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), std::string("update output[y] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(RegexFullMatch, RegexFullMatchInfer);

IMPLEMT_INFERFUNC(RegexReplace, RegexReplaceInfer) {
  Shape un_used;
  if (WithRank(op.GetInputDesc("pattern"), 0, un_used, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("input[pattern] has wrong shape",
      DebugString(op.GetInputDesc("pattern").GetShape().GetDims()),
      ", it should be scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc("rewrite"), 0, un_used, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("input[rewrite] has wrong shape",
      DebugString(op.GetInputDesc("rewrite").GetShape().GetDims()),
      ", it should be scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  Shape x_shape = op.GetInputDesc("x").GetShape();
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(x_shape);
  y_desc.SetDataType(DT_STRING);
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), std::string("update output[y] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(RegexReplace, RegexReplaceInfer);

IMPLEMT_INFERFUNC(AsString, AsStringInfer) {
  TensorDesc out_desc = op.GetOutputDesc("y");
  out_desc.SetDataType(DT_STRING);
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[y] desc failed."));
    return GRAPH_FAILED;
  }
  return UnchangedShape(op, "x", "y");
}

INFER_FUNC_REG(AsString, AsStringInfer);

IMPLEMT_INFERFUNC(EncodeBase64, EncodeBase64Infer) {
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetDataType(DT_STRING);
  if (op.UpdateOutputDesc("y", output_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[y] desc failed"));
    return GRAPH_FAILED;
  }
  return UnchangedShape(op, "x", "y");
}

INFER_FUNC_REG(EncodeBase64, EncodeBase64Infer);

IMPLEMT_INFERFUNC(DecodeBase64, DecodeBase64Infer) {
  TensorDesc out_desc = op.GetOutputDesc("y");
  out_desc.SetDataType(DT_STRING);
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[y] desc failed."));
    return GRAPH_FAILED;
  }
  return UnchangedShape(op, "x", "y");
}

INFER_FUNC_REG(DecodeBase64, DecodeBase64Infer);

// -----------------StringNormalizer Op-------------------------
IMPLEMT_VERIFIER(StringNormalizer, StringNormalizerVerify) {
  AscendString op_name_str;
  op.GetName(op_name_str);
  const char *op_name = op_name_str.GetString();
  TensorDesc input_desc = op.GetInputDescByName("input");
  auto input_type = input_desc.GetDataType();
  // verify input type
  if (input_type != DT_STRING)
  {
    OP_LOGE(op_name, "input must be UTF-8 string!");
    return GRAPH_FAILED;
  }
  // verify input dims
  std::vector<int64_t> input_shape = input_desc.GetShape().GetDims();
  constexpr int ONEDIMS = 1;
  constexpr int TWODIMS = 2;
  if (input_shape.size() != ONEDIMS && input_shape.size() != TWODIMS) {
    OP_LOGE(op_name, 
            "input dims must be 1 or 2, but get %ld.", input_shape.size());
    return GRAPH_FAILED;
  }
  // verify 2-D input shape
  if (input_shape.size() == TWODIMS && input_shape[0] !=1) {
    OP_LOGE(op_name, 
            "when get 2-D input, expected shape is [1, N],but get [%ld, N]", input_shape[0]);
    return GRAPH_FAILED;
  }
  // verify attr
  std::vector<std::string> stop_words; 
  op.GetAttr("stopwords", stop_words);
  if (stop_words.empty()) {
    OP_LOGI(op_name, 
            "attr::stop_words equals to the default value: {}.");
  }

  bool is_case_sensitive = false; 
  op.GetAttr("is_case_sensitive", is_case_sensitive);
  if (!is_case_sensitive) {
    OP_LOGI(op_name, 
            "attr::is_case_sensitive equals to the default value: false.");
  }

  std::string case_change_action = "NONE";
  op.GetAttr("case_change_action", case_change_action);
  if (case_change_action == "NONE") {
    OP_LOGI(op_name, 
            "attr::case_change_action defaults to \"NONE\"".);
  }
  if ((case_change_action != "LOWER") && 
      (case_change_action != "UPPER") && 
      (case_change_action != "NONE")) {
    OP_LOGE(op_name, 
            "attr::case_change_action is unrecognized, acceptable values are \"LOWER\",\"UPPER\",\"NONE\".");
    return GRAPH_FAILED;
  }

  std::string locale = "C"; 
  op.GetAttr("locale", locale);
  if (locale == "C") {
    OP_LOGI(op_name, 
            "get attr::locale equals to the default value: \"C\".");
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(StringNormalizerInferShape) {
  AscendString op_name_str;
  op.GetName(op_name_str);
  const char *op_name = op_name_str.GetString();
  OP_LOGI(op_name, "Enter StringNormalizer proto inferfunction!");
  TensorDesc input_desc = op.GetInputDescByName("input");
  auto input_type = input_desc.GetDataType();
  auto input_shape = input_desc.GetShape().GetDims();
  constexpr int ONEDIMS = 1;
  constexpr int TWODIMS = 2;
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> range;
  if (input_shape.size() == ONEDIMS) {
    output_shape.emplace_back(UNKNOWN_DIM);
    range.emplace_back(std::make_pair(1, input_shape[0]));
  } else if (input_shape.size() == TWODIMS && input_shape[0] ==1) {
    output_shape.emplace_back(1);
    output_shape.emplace_back(UNKNOWN_DIM);
    range.emplace_back(std::make_pair(1, 1));
    range.emplace_back(std::make_pair(1, input_shape[1]));
  } else {
    OP_LOGE(op_name, 
            "input dims must be 1 or 2, but get %lu.", input_shape.size());
    return GRAPH_FAILED;
  }
  TensorDesc output_desc = op.GetOutputDescByName("output");
  output_desc.SetDataType(input_type);
  Shape output_desc_shape(output_shape);
  output_desc.SetShape(output_desc_shape);
  output_desc.SetShapeRange(range);
  op.UpdateOutputDesc("output", output_desc);
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(StringNormalizer, StringNormalizerInferShape);
VERIFY_FUNC_REG(StringNormalizer, StringNormalizerVerify);
// -----------------StringNormalizer END-------------------------
}  // namespace ge