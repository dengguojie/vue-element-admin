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
    OP_LOGE(op.GetName().c_str(), "The rank of input must be 1");
    return GRAPH_FAILED;
  }

  if (WithRank(tensor_sep, 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of delimiter must be 0");
    return GRAPH_FAILED;
  }

  Shape indices_shape;
  Shape values_shape;
  Shape shape;

  auto result = Matrix(ge::UNKNOWN_DIM, 2, indices_shape);
  if (result != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "generate indices_shape failed !");
    return GRAPH_FAILED;
  }
  TensorDesc indices_desc = op.get_output_desc_indices();
  indices_desc.SetShape(indices_shape);
  indices_desc.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("indices", indices_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update indices desc failed");
    return GRAPH_FAILED;
  }

  result = Vector(ge::UNKNOWN_DIM, values_shape);
  if (result != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "generate values_shape failed !");
    return GRAPH_FAILED;
  }
  TensorDesc values_desc = op.get_output_desc_values();
  values_desc.SetShape(values_shape);
  values_desc.SetDataType(DT_STRING);
  if (op.UpdateOutputDesc("values", values_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update values desc failed");
    return GRAPH_FAILED;
  }

  result = Vector(2, shape);
  if (result != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "generate shape failed !");
    return GRAPH_FAILED;
  }
  TensorDesc shape_desc = op.get_output_desc_shape();
  shape_desc.SetShape(shape);
  shape_desc.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("shape", shape_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update shape desc failed");
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
  Shape segment_ids_shape = op.GetInputDesc("segments_ids").GetShape();
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
    const int64_t* num_segments_data = reinterpret_cast<const int64_t *>(num_segments_tensor.GetData());
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
    OP_LOGE(op.GetName().c_str(), "The rank of input must be 1");
    return GRAPH_FAILED;
  }

  if (WithRank(tensor_sep, 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of sep must be 0");
    return GRAPH_FAILED;
  }

  Shape indices_shape;
  Shape values_shape;
  Shape shape;

  auto result = Matrix(ge::UNKNOWN_DIM, 2, indices_shape);
  if (result != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "generate indices_shape failed !");
    return GRAPH_FAILED;
  }
  TensorDesc indices_desc = op.get_output_desc_indices();
  indices_desc.SetShape(indices_shape);
  indices_desc.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("indices", indices_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update indices desc failed");
    return GRAPH_FAILED;
  }

  result = Vector(ge::UNKNOWN_DIM, values_shape);
  if (result != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "generate values_shape failed !");
    return GRAPH_FAILED;
  }
  TensorDesc values_desc = op.get_output_desc_values();
  values_desc.SetShape(values_shape);
  values_desc.SetDataType(DT_STRING);
  if (op.UpdateOutputDesc("values", values_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update values desc failed");
    return GRAPH_FAILED;
  }

  result = Vector(2, shape);
  if (result != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "generate shape failed !");
    return GRAPH_FAILED;
  }
  TensorDesc shape_desc = op.get_output_desc_shape();
  shape_desc.SetShape(shape);
  shape_desc.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("shape", shape_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update shape desc failed");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StringSplitV2, StringSplitV2Infer);

IMPLEMT_INFERFUNC(UnicodeScript, UnicodeScriptInfer) {
  DataType y_type = op.GetInputDesc("x").GetDataType();
  TensorDesc desc = op.GetOutputDesc("y");
  desc.SetShape(op.GetInputDesc("x").GetShape());
  desc.SetDataType(y_type);

  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update y desc failed.");
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
    OP_LOGE(op.GetName().c_str(), "pos and len must have same rank");
    return GRAPH_FAILED;
  }
  for (size_t i = 0; i < pos_shape.GetDimNum(); ++i) {
    auto pos_dim = pos_shape.GetDim(i);
    auto len_dim = len_shape.GetDim(i);
    if (pos_dim != len_dim) {
      OP_LOGE(op.GetName().c_str(), "pos and len must have same dim");
      return GRAPH_FAILED;
    }
  }

  TensorDesc desc = op.GetOutputDesc(0);
  desc.SetDataType(DT_STRING);
  if (op.UpdateOutputDesc("output", desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update output desc failed.");
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
    OP_LOGE(op.GetName().c_str(), " illegal when input type is not DT_STRING");
    return GRAPH_PARAM_INVALID;
  }
  TensorDesc desc = op.GetOutputDesc("y");
  desc.SetShape(op.GetInputDesc("x").GetShape());
  desc.SetDataType(DT_INT64);

  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update y desc failed.");
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
    OP_LOGE(op.GetName().c_str(), "update y desc failed.");
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
    OP_LOGE(op.GetName().c_str(), "update y desc failed.");
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
    OP_LOGE(op.GetName().c_str(), "update y desc failed.");
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
      OP_LOGE(op.GetName().c_str(), "update y desc failed.");
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }

  Shape out(ge::UNKNOWN_SHAPE);
  for (size_t i = 0; i < input_size; ++i) {
    Shape input_shape = op.GetInputDesc(i).GetShape();
    if ((RankKnown(input_shape)) && (input_shape.GetDimNum() != 0)) {
      if (Merge(out, input_shape, out, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "merge two dimension error.");
        return GRAPH_FAILED;
      }
    }
  }
  desc.SetShape(out);
  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update y desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StringJoin, StringJoinInfer);

IMPLEMT_INFERFUNC(StringFormat, StringFormatInfer) {
  string template_attr;
  string placeholder_attr;
  if (op.GetAttr("template", template_attr) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get attr[template] failed.");
    return GRAPH_FAILED;
  }
  if (op.GetAttr("placeholder", placeholder_attr) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get attr[placeholder] failed.");
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
    OP_LOGE(op.GetName().c_str(), "Num placeholders in template and num inputs must match.");
    return GRAPH_FAILED;
  }

  TensorDesc desc = op.GetOutputDesc("y");
  desc.SetDataType(DT_STRING);
  desc.SetShape(Shape());

  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Update y desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StringFormat, StringFormatInfer);

IMPLEMT_INFERFUNC(RegexFullMatch, RegexFullMatchInfer) {
  Shape un_used;
  if (WithRank(op.GetInputDesc("pattern"), 0, un_used, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input pattern must be 0-D");
    return GRAPH_FAILED;
  }
  Shape x_shape = op.GetInputDesc("x").GetShape();
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(x_shape);
  y_desc.SetDataType(DT_BOOL);
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update y failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(RegexFullMatch, RegexFullMatchInfer);

IMPLEMT_INFERFUNC(RegexReplace, RegexReplaceInfer) {
  Shape un_used;
  if (WithRank(op.GetInputDesc("pattern"), 0, un_used, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input pattern must be 0-D");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc("rewrite"), 0, un_used, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input rewrite must be 0-D");
    return GRAPH_FAILED;
  }
  Shape x_shape = op.GetInputDesc("x").GetShape();
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(x_shape);
  y_desc.SetDataType(DT_STRING);
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update y failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(RegexReplace, RegexReplaceInfer);

IMPLEMT_INFERFUNC(AsString, AsStringInfer) {
  TensorDesc out_desc = op.GetOutputDesc("y");
  out_desc.SetDataType(DT_STRING);
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update y failed");
    return GRAPH_FAILED;
  }
  return UnchangedShape(op, "x", "y");
}

INFER_FUNC_REG(AsString, AsStringInfer);

IMPLEMT_INFERFUNC(EncodeBase64, EncodeBase64Infer) {
  TensorDesc out_desc = op.GetOutputDesc("y");
  out_desc.SetDataType(DT_STRING);
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update y failed");
    return GRAPH_FAILED;
  }
  return UnchangedShape(op, "x", "y");
}

INFER_FUNC_REG(EncodeBase64, EncodeBase64Infer);

IMPLEMT_INFERFUNC(DecodeBase64, DecodeBase64Infer) {
  TensorDesc out_desc = op.GetOutputDesc("y");
  out_desc.SetDataType(DT_STRING);
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update y failed");
    return GRAPH_FAILED;
  }
  return UnchangedShape(op, "x", "y");
}

INFER_FUNC_REG(DecodeBase64, DecodeBase64Infer);
}  // namespace ge