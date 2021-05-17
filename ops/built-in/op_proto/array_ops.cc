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
 * \file array_ops.cpp
 * \brief
 */
#include "inc/array_ops.h"
#include <climits>
#include <unordered_set>
#include <utility>

#include "common/inc/op_log.h"
#include "common_shape_fns.h"
#include "array_ops_shape_fns.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/node_utils.h"
#include "./util/error_util.h"
#include "util/util.h"

namespace ge {
const char* const kShape = "shape";
const char* const kShapeDtype = "shape dtype";
const char* const kAttrShape = "attr shape";
const char* const kAttrDtype = "attr dtype";
const char* const kAttrAxis = "attr axis";
const char* const kAttrNumAxes = "attr num_axes";
const char* const kPreOpInputShapeRange = "_pre_op_in_range";
const int64_t kMaxDimNum = 8;

IMPLEMT_INFERFUNC(MatrixBandPart, MatrixBandPartInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto x_desc = op_desc->MutableInputDesc(0);

  DataType type = x_desc->GetDataType();
  std::vector<std::pair<int64_t, int64_t>> range;
  if (x_desc->GetShapeRange(range) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  auto y_desc = op_desc->MutableOutputDesc(0);
  y_desc->SetShape(x_desc->GetShape());
  y_desc->SetShapeRange(range);
  y_desc->SetDataType(type);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixBandPart, MatrixBandPartInfer);

IMPLEMT_INFERFUNC(UniqueWithCounts, UniqueWithCountsInfer) {
  DataType y_type = op.GetInputDesc("x").GetDataType();
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(Shape({ge::UNKNOWN_DIM}));
  output_desc.SetDataType(y_type);
  if (op.UpdateOutputDesc("y", output_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), string("update description for output[y] failed"));
    return GRAPH_FAILED;
  }

  DataType type;
  if (op.GetAttr("out_idx", type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
                                       string("get attr[out_idx] failed"));
    return GRAPH_FAILED;
  }

  output_desc = op.GetOutputDesc("count");
  output_desc.SetShape(Shape({ge::UNKNOWN_DIM}));
  output_desc.SetDataType(type);

  if (op.UpdateOutputDesc("count", output_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), string("update description for output[count] failed"));
    return GRAPH_FAILED;
  }

  output_desc = op.GetOutputDesc("idx");
  output_desc.SetDataType(type);
  if (op.UpdateOutputDesc("idx", output_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), string("update description for output[idx] failed"));
    return GRAPH_FAILED;
  }
  return UnchangedShape(op, "x", "idx");
}

INFER_FUNC_REG(UniqueWithCounts, UniqueWithCountsInfer);

IMPLEMT_INFERFUNC(Unique, UniqueInfer) {
  OpDescPtr op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr x_input = op_desc->MutableInputDesc(0);

  GeShape x_shape;
  if (WithRank(x_input, 1, x_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(x_input->GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  DataType idx_type;
  if (op.GetAttr("out_idx", idx_type) != GRAPH_SUCCESS) {
    CUBE_CALL_ERR_REPORT(op.GetName().c_str(), "Op get attr out_idx failed");
    return GRAPH_FAILED;
  }

  GeTensorDescPtr idx_desc = op_desc->MutableOutputDesc(1);
  idx_desc->SetShape(x_shape);
  idx_desc->SetOriginShape(x_shape);
  idx_desc->SetDataType(idx_type);

  GeTensorDescPtr y_desc = op_desc->MutableOutputDesc(0);
  y_desc->SetShape(GeShape({UNKNOWN_DIM}));
  y_desc->SetOriginShape(GeShape({UNKNOWN_DIM}));
  y_desc->SetDataType(x_input->GetDataType());
  if (x_shape.GetShapeSize() == UNKNOWN_DIM) {
    return GRAPH_SUCCESS;
  } else {
    std::vector<std::pair<int64_t, int64_t>> range;
    int64_t max_dim = x_shape.GetDim(0);
    range.emplace_back(std::make_pair(1, max_dim));
    y_desc->SetShapeRange(range);
    return GRAPH_SUCCESS;
  }
}

INFER_FUNC_REG(Unique, UniqueInfer);

IMPLEMT_INFERFUNC(UniqueExt2, UniqueExt2Infer) {
  Shape x_shape;
  if (WithRankAtLeast(op.GetInputDesc(0), 1, x_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    ShapeErrReport(0, op.GetName(), DebugString(op.GetInputDesc(0).GetShape().GetDims()), "at least 1D");
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "input x must be more than 1-D");
    return GRAPH_FAILED;
  }

  Shape axis_shape;
  if (WithRank(op.GetInputDesc(1), 1, axis_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    ShapeErrReport(1, op.GetName(), DebugString(op.GetInputDesc(1).GetShape().GetDims()), "1D");
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "input axis must be 1-D");
    return GRAPH_FAILED;
  }

  DataType type = op.GetInputDesc("x").GetDataType();
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(Shape({ge::UNKNOWN_DIM}));
  y_desc.SetDataType(type);
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), string("update description for output y failed"));
    return GRAPH_FAILED;
  }

  DataType idx_type;
  if (op.GetAttr("out_idx", idx_type) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Op get attr out_idx failed");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> axis_dims;
  axis_dims.push_back(ge::UNKNOWN_DIM);
  TensorDesc idx_desc = op.GetOutputDesc("idx");
  idx_desc.SetShape(Shape(axis_dims));
  idx_desc.SetDataType(idx_type);
  if (op.UpdateOutputDesc("idx", idx_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(UniqueExt2, UniqueExt2Infer);

IMPLEMT_INFERFUNC(InvertPermutation, InvertPermutationInfer) {
  Shape shape;
  if (WithRank(op.GetInputDesc(0), 1, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    ShapeErrReport(0, op.GetName(), DebugString(op.GetInputDesc(0).GetShape().GetDims()), "1D");
    std::string err_msg = ConcatString("failed to call WithRank function, ",
      "input[x] rank must be 1D, but got rank[",
      op.GetInputDesc(0).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<std::pair<int64_t, int64_t>> x_range;
  auto status = op.GetInputDesc("x").GetShapeRange(x_range);
  if (status != GRAPH_SUCCESS) {
    return status;
  }

  DataType type = op.GetInputDesc("x").GetDataType();
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(op.GetInputDesc(0).GetShape());
  y_desc.SetShapeRange(x_range);
  y_desc.SetDataType(type);
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(InvertPermutation, InvertPermutationInfer);

IMPLEMT_INFERFUNC(CheckNumerics, CheckNumericsInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto x_desc = op_desc->MutableInputDesc(0);

  std::vector<std::pair<int64_t, int64_t>> range;
  if (x_desc->GetShapeRange(range) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  auto y_desc = op_desc->MutableOutputDesc(0);
  y_desc->SetShape(x_desc->GetShape());
  y_desc->SetShapeRange(range);
  y_desc->SetDataType(x_desc->GetDataType());

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(CheckNumerics, CheckNumericsInfer);

IMPLEMT_INFERFUNC(UnravelIndex, UnravelIndexInfer) {
  auto indices_desc = op.GetInputDesc(0);
  auto dims_desc = op.GetInputDesc(1);

  Shape dims_shape;
  if (WithRank(dims_desc, 1, dims_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    string err_msg = ConcatString("input[dims] must be 1D, real rank is ",
                     dims_shape.GetDimNum());
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  Shape indices_shape;
  if (WithRankAtMost(indices_desc, 1, indices_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    string err_msg = ConcatString("input[indices] must be less than 1D, real rank is ",
                     dims_shape.GetDimNum());
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<int64_t> out_dims({-1, -1});
  std::vector<int64_t> dims_shape_vec = dims_shape.GetDims();
  std::vector<int64_t> indices_shape_vec = indices_shape.GetDims();
  if (indices_shape.GetDimNum() == 0) {
    out_dims[0] = 1;
  } else {
    if (indices_shape_vec != ge::UNKNOWN_RANK && indices_shape_vec != ge::UNKNOWN_SHAPE) {
      out_dims[0] = indices_shape_vec[0];
    }
  }
  if (dims_shape_vec != ge::UNKNOWN_RANK && dims_shape_vec != ge::UNKNOWN_SHAPE) {
    out_dims[1] = dims_shape_vec[0];
  }

  TensorDesc out_desc = op.GetOutputDesc("y");
  out_desc.SetShape(Shape(out_dims));
  out_desc.SetDataType(indices_desc.GetDataType());
  return op.UpdateOutputDesc("y", out_desc);
}

INFER_FUNC_REG(UnravelIndex, UnravelIndexInfer);

IMPLEMT_INFERFUNC(UpperBound, UpperBoundInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc(0), 2, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    ShapeErrReport(0, op.GetName(), DebugString(op.GetInputDesc(0).GetShape().GetDims()), "2D");
    string err_msg = ConcatString(
        "failed to call WithRank function, input[sorted_x] rank must be 2D, "
        "got rank[", op.GetInputDesc(0).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 2, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    ShapeErrReport(1, op.GetName(), DebugString(op.GetInputDesc(1).GetShape().GetDims()), "2D");
    string err_msg = ConcatString(
        "failed to call WithRank function, input[values] rank must be 2D, "
        "got rank[", op.GetInputDesc(1).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  DataType type;
  if (op.GetAttr("out_type", type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
                                       string("get attr[out_type] failed"));
    return GRAPH_FAILED;
  }

  TensorDesc out_desc = op.GetOutputDesc("y");
  out_desc.SetShape(op.GetInputDesc(1).GetShape());
  out_desc.SetDataType(type);
  return op.UpdateOutputDesc("y", out_desc);
}

INFER_FUNC_REG(UpperBound, UpperBoundInfer);

IMPLEMT_INFERFUNC(UniqueWithCountsExt2, UniqueWithCountsExt2Infer) {
  DataType y_type = op.GetInputDesc("x").GetDataType();
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(Shape({ge::UNKNOWN_DIM}));
  output_desc.SetDataType(y_type);
  if (op.UpdateOutputDesc("y", output_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), string("update description for output[y] failed"));
    return GRAPH_FAILED;
  }

  DataType type;
  if (op.GetAttr("out_idx", type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
                                       string("get attr[out_idx] failed"));
    return GRAPH_FAILED;
  }

  output_desc = op.GetOutputDesc("count");
  output_desc.SetShape(Shape({UNKNOWN_DIM}));
  output_desc.SetDataType(type);

  if (op.UpdateOutputDesc("count", output_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), string("update description for output[count] failed"));
    return GRAPH_FAILED;
  }

  output_desc = op.GetOutputDesc("idx");
  output_desc.SetDataType(type);
  if (op.UpdateOutputDesc("idx", output_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), string("update description for output[idx] failed"));
    return GRAPH_FAILED;
  }
  return UnchangedShape(op, "x", "idx");
}

INFER_FUNC_REG(UniqueWithCountsExt2, UniqueWithCountsExt2Infer);

IMPLEMT_INFERFUNC(MirrorPad, MirrorPadInfer) {
  if (PadShapeFn(op) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  DataType type = op.GetInputDesc("x").GetDataType();
  TensorDesc desc = op.GetOutputDesc("y");
  desc.SetDataType(type);
  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MirrorPad, MirrorPadInfer);

IMPLEMT_INFERFUNC(ListDiff, ListDiffInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto x_desc = op_desc->MutableInputDesc(0);
  auto y_desc = op_desc->MutableInputDesc(1);

  Shape unused_shape;
  std::string error_msg;
  if (WithRank(x_desc, 1, unused_shape, op.GetName().c_str()) !=
      GRAPH_SUCCESS) {
    std::string error_msg =
        GetShapeErrMsg(0, DebugString(x_desc->GetShape().GetDims()), "1D");
    error_msg = string("failed to call WithRank function, ") + error_msg;
    return GRAPH_FAILED;
  }

  if (WithRank(y_desc, 1, unused_shape, op.GetName().c_str()) !=
      GRAPH_SUCCESS) {
    std::string error_msg =
        GetShapeErrMsg(1, DebugString(y_desc->GetShape().GetDims()), "1D");
    error_msg = string("failed to call WithRank function, ") + error_msg;
    return GRAPH_FAILED;
  }

  DataType output_type = x_desc->GetDataType();
  DataType index_type;
  if (op.GetAttr("out_idx", index_type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
                                       string("failed to get attr[out_idx]."));
    return GRAPH_FAILED;
  }

  GeShape result({ge::UNKNOWN_DIM});
  auto output_desc = op_desc->MutableOutputDesc(0);
  output_desc->SetShape(GeShape(result));
  output_desc->SetDataType(output_type);

  auto index_desc = op_desc->MutableOutputDesc(1);
  index_desc->SetShape(GeShape(result));
  index_desc->SetDataType(index_type);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ListDiff, ListDiffInfer);

IMPLEMT_INFERFUNC(MirrorPadGrad, MirrorPadGradInfer) {
  return PadGradShapeFn(op);
}

INFER_FUNC_REG(MirrorPadGrad, MirrorPadGradInfer);

IMPLEMT_INFERFUNC(ReverseSequence, ReverseSequenceInfer) {
  Shape input_shape = op.GetInputDesc("x").GetShape();
  TensorDesc seq_lengths_desc = op.GetInputDesc("seq_lengths");
  Shape seq_lengths_shape = op.GetInputDesc("seq_lengths").GetShape();

  // Check whether seq_lengths's rank is equal to 1
  if (WithRank(seq_lengths_desc, 1, seq_lengths_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call WithRank function, input[seq_lengths] rank must "
        "be 1, got rank[", seq_lengths_desc.GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // If rank of x is unknown, set output into unknown shape
  TensorDesc y_desc = op.GetOutputDesc("y");
  if (input_shape.GetDims() == ge::UNKNOWN_SHAPE) {
    y_desc.SetShape(Shape(ge::UNKNOWN_SHAPE));
    return GRAPH_SUCCESS;
  }

  int64_t seq_dim;
  if (op.GetAttr("seq_dim", seq_dim) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Failed to get attribute value.");
    return GRAPH_FAILED;
  }
  int64_t batch_dim;
  if (op.GetAttr("batch_dim", batch_dim) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Failed to get attribute value.");
    return GRAPH_FAILED;
  }
  int64_t input_rank = input_shape.GetDimNum();
  if (seq_dim >= input_rank) {
    string err_msg = GetAttrValueErrMsg("seq_dim", ConcatString(seq_dim),
        ConcatString("< the rank of 0th input[", input_rank, "], 0th input shape is ",
            DebugString(input_shape.GetDims())));
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (batch_dim >= input_rank) {
    string err_msg = GetAttrValueErrMsg("batch_dim", ConcatString(batch_dim),
        ConcatString("< the rank of 0th input[", input_rank, "], 0th input shape is ",
            DebugString(input_shape.GetDims())));
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int64_t batch_dim_dim = input_shape.GetDim(batch_dim);
  if (Merge(batch_dim_dim, seq_lengths_shape.GetDim(0), batch_dim_dim) != GRAPH_SUCCESS) {
    string err_msg = ConcatString(
        "failed to call Merge function, the batch_dim[", batch_dim, "]th dim "
        "value[", batch_dim_dim, "] of 0th input should be equal to 0th dim "
        "value[", seq_lengths_shape.GetDim(0), "] of 1th input. 0th input "
        "shape", DebugString(input_shape.GetDims()), ", 1th input shape",
        DebugString(seq_lengths_shape.GetDims()));
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // Replace batch_dim of input with batch_size
  Shape y_shape;
  ReplaceDim(input_shape, batch_dim, batch_dim_dim, y_shape, op.GetName().c_str());

  DataType x_type = op.GetInputDesc("x").GetDataType();
  y_desc.SetDataType(x_type);
  y_desc.SetShape(y_shape);
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Failed to update y desc.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ReverseSequence, ReverseSequenceInfer);

IMPLEMT_INFERFUNC(Const, ConstInfer) {
  auto const_value = op.get_attr_value();
  auto val_desc = const_value.GetTensorDesc();
  auto dims = val_desc.GetShape().GetDims();
  auto attr_dtype = val_desc.GetDataType();

  TensorDesc out_desc = op.get_output_desc_y();
  out_desc.SetDataType(ge::DataType(attr_dtype));
  out_desc.SetShape(Shape(dims));
  (void)op.update_output_desc_y(out_desc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Const, ConstInfer);

IMPLEMT_INFERFUNC(Constant, ConstantInfer) {
  auto const_value = op.get_attr_value();
  auto val_desc = const_value.GetTensorDesc();
  auto dims = val_desc.GetShape().GetDims();
  auto attr_dtype = val_desc.GetDataType();

  TensorDesc out_desc = op.get_output_desc_y();
  out_desc.SetDataType(ge::DataType(attr_dtype));
  out_desc.SetShape(Shape(dims));
  (void)op.update_output_desc_y(out_desc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Constant, ConstantInfer);

graphStatus ConstAndConstantInferFormat(ge::Operator& op) {
  OP_LOGI(op.GetName().c_str(), "Const infer format start");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto format = op_desc->MutableOutputDesc(0)->GetOriginFormat();
  ConstGeTensorPtr tensor_value;
  if (!AttrUtils::GetTensor(op_desc, "value", tensor_value)) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Get attr value failed!");
    return GRAPH_FAILED;
  }
  if (!tensor_value) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "attr tensor is not exist!");
    return GRAPH_FAILED;
  }
  auto tensor_ptr = const_cast<GeTensor*>(tensor_value.get());
  tensor_ptr->MutableTensorDesc().SetOriginFormat(format);
  tensor_ptr->MutableTensorDesc().SetFormat(format);
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFORMAT_FUNC(Const, ConstInferFormat) {
  return ConstAndConstantInferFormat(op);
}

INFER_FORMAT_FUNC_REG(Const, ConstInferFormat);

IMPLEMT_INFERFUNC(_ParallelConcatStart, ParallelConcatStartInfer) {
  auto attr_shape = op.get_attr_shape();
  auto attr_dtype = op.get_attr_dtype();

  TensorDesc outDesc = op.GetOutputDesc("y");
  outDesc.SetDataType(ge::DataType(attr_dtype));
  outDesc.SetShape(Shape(attr_shape));
  (void)op.UpdateOutputDesc("y", outDesc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(_ParallelConcatStart, ParallelConcatStartInfer);

IMPLEMT_INFERFUNC(Snapshot, SnapshotInferFunc) {
  OP_LOGI(op.GetName().c_str(), "Snapshot infershape start");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc_x = op_desc->MutableInputDesc("x");
  auto output_desc_y = op_desc->MutableOutputDesc("y");

  auto x_dims = input_desc_x->MutableShape().GetDims();
  auto x_type = input_desc_x->GetDataType();
  std::vector<std::pair<int64_t, int64_t>> x_range;
  input_desc_x->GetShapeRange(x_range);
  output_desc_y->SetShape(GeShape(x_dims));
  output_desc_y->SetOriginShape(GeShape(x_dims));
  output_desc_y->SetShapeRange(x_range);
  output_desc_y->SetDataType(x_type);
  OP_LOGI(op.GetName().c_str(), "Snapshot infershape end");
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Snapshot, SnapshotInferFunc);

IMPLEMT_INFERFUNC(GuaranteeConst, GuaranteeConstInfer) {
  TensorDesc tensorDesc = op.GetInputDesc("x");
  (void)op.UpdateOutputDesc("y", tensorDesc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(GuaranteeConst, GuaranteeConstInfer);

IMPLEMT_INFERFUNC(BroadcastArgs, BroadcastArgsInferFunc) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto x1_desc = op_desc->MutableInputDesc("x1");
  auto x2_desc = op_desc->MutableInputDesc("x2");
  auto y_desc = op_desc->MutableOutputDesc("y");
  auto x1_dims = x1_desc->GetShape().GetDims();
  auto x2_dims = x2_desc->GetShape().GetDims();
  auto data_type = x1_desc->GetDataType();
  std::vector<std::pair<int64_t, int64_t>> x1_range;
  std::vector<std::pair<int64_t, int64_t>> x2_range;
  std::vector<std::pair<int64_t, int64_t>> out_range;
  x1_desc->GetShapeRange(x1_range);
  x2_desc->GetShapeRange(x2_range);


  bool data_type_check = ((x1_desc->GetDataType() != DT_INT32 && x1_desc->GetDataType() != DT_INT64) ||
                          (x2_desc->GetDataType() != DT_INT32 && x2_desc->GetDataType() != DT_INT64));
  if (data_type_check) {
    string reason = "x1[" + std::to_string(x1_desc->GetDataType()) + "] + and + x2[" +
                    std::to_string(x1_desc->GetDataType()) + "] must DT_INT32 or DT_INT64";
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), "dtype", reason);
    GE_OP_LOGE(op.GetName().c_str(), "Data type check fail. x1[%u] and x2[%u] must DT_INT32 or DT_INT64",
               x1_desc->GetDataType(), x2_desc->GetDataType());
    return GRAPH_PARAM_INVALID;
  }

  if (x1_dims.size() > 1 || x2_dims.size() > 1) {
    string reason = "x1[" + std::to_string(x1_dims.size()) + "] + and + x2[" + std::to_string(x2_dims.size()) +
                    "] must be less than or equal to 1";
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), "dims", reason);
    GE_OP_LOGE(op.GetName().c_str(), "Size check fail. x1[%u] and x2[%u] must be less than or equal to 1",
               x1_dims.size(), x2_dims.size());
    return GRAPH_PARAM_INVALID;
  }

  if (x1_dims == UNKNOWN_RANK || x2_dims == UNKNOWN_RANK) {
    GE_OP_LOGD(op.GetName().c_str(), "all two inputs are unknown rank!");
    y_desc->SetShape(GeShape(UNKNOWN_SHAPE));
    y_desc->SetOriginShape(GeShape(UNKNOWN_SHAPE));
    y_desc->SetDataType(data_type);
    return GRAPH_SUCCESS;
  }

  if (x1_dims == UNKNOWN_SHAPE && x2_dims == UNKNOWN_SHAPE) {
    GE_OP_LOGD(op.GetName().c_str(), "all two inputs are unknown shape!");
    y_desc->SetShape(GeShape(UNKNOWN_SHAPE));
    y_desc->SetOriginShape(GeShape(UNKNOWN_SHAPE));
    y_desc->SetDataType(data_type);
    y_desc->SetShapeRange(x1_range);
    return GRAPH_SUCCESS;
  } else if (x1_dims == UNKNOWN_SHAPE) {
    GE_OP_LOGD(op.GetName().c_str(), "x1 is unknown shape!");
    int64_t range_max = x2_dims.size();
    std::pair<int64_t, int64_t> pair({1, range_max});
    out_range.emplace_back(pair);
    y_desc->SetShape(GeShape(UNKNOWN_SHAPE));
    y_desc->SetOriginShape(GeShape(UNKNOWN_SHAPE));
    y_desc->SetDataType(data_type);
    y_desc->SetShapeRange(out_range);
    return GRAPH_SUCCESS;
  } else if (x2_dims == UNKNOWN_SHAPE) {
    GE_OP_LOGD(op.GetName().c_str(), "x2 is unknown shape!");
    int64_t range_max = x2_dims.size();
    std::pair<int64_t, int64_t> pair({1, range_max});
    out_range.emplace_back(pair);
    y_desc->SetShape(GeShape(UNKNOWN_SHAPE));
    y_desc->SetOriginShape(GeShape(UNKNOWN_SHAPE));
    y_desc->SetDataType(data_type);
    y_desc->SetShapeRange(out_range);
    return GRAPH_SUCCESS;
  }

  if (x1_dims.empty()) {
    y_desc->SetShape(GeShape(x2_dims));
  } else if (x2_dims.empty()) {
    y_desc->SetShape(GeShape(x1_dims));
  } else {
    auto dims = x1_dims[0] > x2_dims[0] ? x1_dims : x2_dims;
    y_desc->SetShape(GeShape(dims));
  }

  int64_t range_max = x1_dims.size() > x2_dims.size() ? x1_dims.size() : x2_dims.size();
  std::pair<int64_t, int64_t> pair({1, range_max});
  out_range.emplace_back(pair);
  y_desc->SetShapeRange(out_range);
  y_desc->SetDataType(x1_desc->GetDataType());
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BroadcastArgs, BroadcastArgsInferFunc);

IMPLEMT_INFERFUNC(BroadcastGradientArgs, BroadcastGradientArgsInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  auto input_desc_x1 = op_desc->MutableInputDesc("x1");
  auto input_desc_x2 = op_desc->MutableInputDesc("x2");
  auto output_desc_y1 = op_desc->MutableOutputDesc("y1");
  auto output_desc_y2 = op_desc->MutableOutputDesc("y2");
  auto dims_x1 = input_desc_x1->MutableShape().GetDims();
  auto dims_x2 = input_desc_x2->MutableShape().GetDims();
  auto x1_type = input_desc_x1->GetDataType();
  auto x2_type = input_desc_x2->GetDataType();
  std::vector<std::pair<int64_t, int64_t>> x1_range;
  std::vector<std::pair<int64_t, int64_t>> x2_range;
  std::vector<std::pair<int64_t, int64_t>> out_range;
  input_desc_x1->GetShapeRange(x1_range);
  input_desc_x2->GetShapeRange(x2_range);

  if (dims_x1 == UNKNOWN_RANK || dims_x2 == UNKNOWN_RANK) {
    GE_OP_LOGD(op.GetName().c_str(), "all two inputs are unknown rank!");
    output_desc_y1->SetShape(GeShape(UNKNOWN_SHAPE));
    output_desc_y1->SetOriginShape(GeShape(UNKNOWN_SHAPE));
    output_desc_y1->SetDataType(x1_type);
    output_desc_y2->SetShape(GeShape(UNKNOWN_SHAPE));
    output_desc_y2->SetOriginShape(GeShape(UNKNOWN_SHAPE));
    output_desc_y2->SetDataType(x2_type);
    return GRAPH_SUCCESS;
  }
  // Input Dim Num must be equal or smaller than 1
  if (dims_x1 == UNKNOWN_SHAPE && dims_x2 == UNKNOWN_SHAPE) {
    GE_OP_LOGD(op.GetName().c_str(), "all two inputs are unknown shape!");
    output_desc_y1->SetShape(GeShape(UNKNOWN_SHAPE));
    output_desc_y1->SetOriginShape(GeShape(UNKNOWN_SHAPE));
    output_desc_y1->SetDataType(x1_type);
    output_desc_y1->SetShapeRange(x1_range);
    output_desc_y2->SetShape(GeShape(UNKNOWN_SHAPE));
    output_desc_y2->SetOriginShape(GeShape(UNKNOWN_SHAPE));
    output_desc_y2->SetDataType(x2_type);
    output_desc_y2->SetShapeRange(x2_range);
    return GRAPH_SUCCESS;
  } else if (dims_x1 == UNKNOWN_SHAPE) {
    GE_OP_LOGD(op.GetName().c_str(), "x1 is unknown shape!");
    int64_t range_max = dims_x2.size();
    std::pair<int64_t, int64_t> pair({1, range_max});
    out_range.emplace_back(pair);
    output_desc_y1->SetShape(GeShape(UNKNOWN_SHAPE));
    output_desc_y1->SetOriginShape(GeShape(UNKNOWN_SHAPE));
    output_desc_y1->SetDataType(x1_type);
    output_desc_y1->SetShapeRange(out_range);
    output_desc_y2->SetShape(GeShape(UNKNOWN_SHAPE));
    output_desc_y2->SetOriginShape(GeShape(UNKNOWN_SHAPE));
    output_desc_y2->SetDataType(x2_type);
    output_desc_y2->SetShapeRange(out_range);
    return GRAPH_SUCCESS;
  } else if (dims_x2 == UNKNOWN_SHAPE) {
    GE_OP_LOGD(op.GetName().c_str(), "x2 is unknown shape!");
    int64_t range_max = dims_x1.size();
    std::pair<int64_t, int64_t> pair({1, range_max});
    out_range.emplace_back(pair);
    output_desc_y1->SetShape(GeShape(UNKNOWN_SHAPE));
    output_desc_y1->SetOriginShape(GeShape(UNKNOWN_SHAPE));
    output_desc_y1->SetDataType(x1_type);
    output_desc_y1->SetShapeRange(out_range);
    output_desc_y2->SetShape(GeShape(UNKNOWN_SHAPE));
    output_desc_y2->SetOriginShape(GeShape(UNKNOWN_SHAPE));
    output_desc_y2->SetDataType(x2_type);
    output_desc_y2->SetShapeRange(out_range);
    return GRAPH_SUCCESS;
  }

  GE_OP_LOGD(op.GetName().c_str(), "all two inputs are known shape!");
  int64_t range_max = dims_x1.size() == 0 ? 1 : dims_x1.size();
  std::pair<int64_t, int64_t> pair({1, range_max});
  out_range.emplace_back(pair);
  output_desc_y1->SetDataType(x1_type);
  output_desc_y2->SetDataType(x2_type);
  output_desc_y1->SetShape(GeShape(UNKNOWN_SHAPE));
  output_desc_y1->SetOriginShape(GeShape(UNKNOWN_SHAPE));
  output_desc_y2->SetShape(GeShape(UNKNOWN_SHAPE));
  output_desc_y2->SetOriginShape(GeShape(UNKNOWN_SHAPE));
  output_desc_y1->SetShapeRange(out_range);
  output_desc_y2->SetShapeRange(out_range);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BroadcastGradientArgs, BroadcastGradientArgsInfer);

IMPLEMT_INFERFUNC(PreventGradient, PreventGradientInferFunc) {
  OP_LOGI(op.GetName().c_str(), "PreventGradient infershape start");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc_x = op_desc->MutableInputDesc("x");
  auto output_desc_y = op_desc->MutableOutputDesc("y");

  auto x_dims = input_desc_x->MutableShape().GetDims();
  auto x_type = input_desc_x->GetDataType();
  std::vector<std::pair<int64_t, int64_t>> x_range;
  input_desc_x->GetShapeRange(x_range);
  output_desc_y->SetShape(GeShape(x_dims));
  output_desc_y->SetOriginShape(GeShape(x_dims));
  output_desc_y->SetShapeRange(x_range);
  output_desc_y->SetDataType(x_type);
  OP_LOGI(op.GetName().c_str(), "PreventGradient infershape end");
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(PreventGradient, PreventGradientInferFunc);

IMPLEMT_INFERFUNC(StopGradient, StopGradientInferFunc) {
  OP_LOGI(op.GetName().c_str(), "StopGradient infershape start");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc_x = op_desc->MutableInputDesc("x");
  auto output_desc_y = op_desc->MutableOutputDesc("y");

  auto x_dims = input_desc_x->MutableShape().GetDims();
  auto x_type = input_desc_x->GetDataType();
  std::vector<std::pair<int64_t, int64_t>> x_range;
  input_desc_x->GetShapeRange(x_range);
  output_desc_y->SetShape(GeShape(x_dims));
  output_desc_y->SetOriginShape(GeShape(x_dims));
  output_desc_y->SetShapeRange(x_range);
  output_desc_y->SetShapeRange(x_range);
  output_desc_y->SetDataType(x_type);
  OP_LOGI(op.GetName().c_str(), "StopGradient infershape end");
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StopGradient, StopGradientInferFunc);

IMPLEMT_INFERFUNC(ExpandDims, ExpandDimsInfer) {
  std::vector<string> dep_inputs = {"axis"};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto node = NodeUtils::GetNodeFromOperator(op);
  if (node == nullptr) {
    GE_OP_LOGE(op.GetName().c_str(), "get null node ptr");
    return GRAPH_FAILED;
  }
  auto x_desc = op_desc->MutableInputDesc("x");
  auto axis_desc = op_desc->MutableInputDesc("axis");
  auto y_desc = op_desc->MutableOutputDesc("y");

  op_desc->SetOpInferDepends(dep_inputs);
  auto axis_type = axis_desc->GetDataType();
  auto x_type = x_desc->GetDataType();

  if (axis_type != DT_INT32 && axis_type != DT_INT64) {
    string reason = "axis dtype[" + std::to_string(axis_type) + "] must int32 or int64";
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), kAttrDtype, reason);
    GE_OP_LOGE(op.GetName().c_str(), "axis dtype[%d] must int32 or int64", axis_type);
    return GRAPH_PARAM_INVALID;
  }

  bool is_x_unknonwn_rank = x_desc->MutableShape().GetDims() == UNKNOWN_RANK ? true : false;
  if (is_x_unknonwn_rank) {
    GE_OP_LOGD("input x shape is unknown rank!");
    y_desc->SetUnknownDimNumShape();
    y_desc->SetDataType(x_type);
    y_desc->SetOriginDataType(x_type);
    return GRAPH_SUCCESS;
  }

  int64_t axis_nums = axis_desc->MutableShape().GetShapeSize();

  if (axis_nums != 1) {
    // Shape::GetDims().size() == 0, means it's a scalar, its shape is [].
    if (!(axis_nums == 0 && axis_desc->MutableShape().GetDims().size() == 0)) {
      string reason = "axis input must be a tensor with a single value, but [" + std::to_string(axis_nums) + "] nums";
      GeInfershapeErrReport(op.GetName(), op.GetOpType(), "axis", reason);
      GE_OP_LOGE(op.GetName().c_str(), "'axis' input must be a tensor with a single value, but %d nums", axis_nums);
      return GRAPH_PARAM_INVALID;
    }
  }

  GeTensorPtr tensor_axis = nullptr;
  graphStatus status = NodeUtils::GetInputConstData(node, "axis", tensor_axis);
  if (status != GRAPH_SUCCESS) {
    GE_OP_LOGI(op.GetName().c_str(), "Op get input const data of axis failed");
    auto x_shape_size = x_desc->MutableShape().GetDims().size();
    std::vector<int64_t> out_dims(x_shape_size + 1, UNKNOWN_DIM);
    y_desc->SetShape(GeShape(out_dims));
    y_desc->SetOriginShape(GeShape(out_dims));
    y_desc->SetDataType(x_type);
    y_desc->SetOriginDataType(x_type);
    // infer shape range
    std::vector<std::pair<int64_t, int64_t>> x_range;
    (void)x_desc->GetShapeRange(x_range);
    if (x_range.empty()) {
      GE_OP_LOGD(op.GetName().c_str(), "last op does not set shape range!");
      return GRAPH_SUCCESS;
    }
    if (x_range.size() != x_shape_size) {
      GE_OP_LOGE(op.GetName().c_str(),
        "input range size num[%zu] should be same with input shape size[%zu]", x_range.size(), x_shape_size);
      return GRAPH_FAILED;
    }
    int64_t max_range_value = 1;
    for (const auto &ele : x_range) {
      if (ele.second > max_range_value) {
        max_range_value = ele.second;
      }
    }
    std::vector<std::pair<int64_t, int64_t>> y_range(x_shape_size + 1, std::pair<int64_t, int64_t>({1, max_range_value}));
    y_desc->SetShapeRange(y_range);
    return GRAPH_SUCCESS;
  }

  auto pbuff = tensor_axis->GetData().GetData();
  if (pbuff == nullptr) {
    GE_OP_LOGE(op.GetName().c_str(), "no const data when get data from tensor!");
    return GRAPH_FAILED;
  }
  int64_t axis;
  if (axis_type == DT_INT32) {
    axis = *const_cast<int32_t*>(reinterpret_cast<const int32_t*>(pbuff));
  } else if (axis_type == DT_INT64) {
    axis = *const_cast<int64_t*>(reinterpret_cast<const int64_t*>(pbuff));
  }

  std::vector<int64_t> vec_dim;
  int32_t dim_num = x_desc->MutableShape().GetDimNum();
  if (axis < -1 - dim_num || axis > dim_num) {
    string reason = "axis[" + std::to_string(axis) + "] is not in [" + std::to_string(-1 - dim_num) + " , " +
                    std::to_string(dim_num) + "]";
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), "axis", reason);
    GE_OP_LOGE(op.GetName().c_str(), "axis[%d] is not in [%d, %d]", axis, -1 - dim_num, dim_num);
    return GRAPH_PARAM_INVALID;
  }

  if (axis < 0) {
    axis += dim_num + 1;
  }
  for (int i = 0; i < dim_num; i++) {
    vec_dim.push_back(x_desc->MutableShape().GetDim(i));
  }
  vec_dim.emplace(vec_dim.begin() + axis, 1);
  y_desc->SetShape(GeShape(vec_dim));
  y_desc->SetOriginShape(GeShape(vec_dim));
  y_desc->SetDataType(x_type);
  y_desc->SetOriginDataType(x_type);
  // infer shape range
  auto x_shape_size = x_desc->MutableShape().GetDims().size();
  std::vector<std::pair<int64_t, int64_t>> x_range;
  (void)x_desc->GetShapeRange(x_range);
  if (x_range.empty()) {
    GE_OP_LOGD(op.GetName().c_str(), "last op does not set shape range, so break!");
    return GRAPH_SUCCESS;
  }
  if (x_range.size() != x_shape_size) {
    GE_OP_LOGE(op.GetName().c_str(),
      "input range size num[%zu] should be same with input shape size[%zu]", x_range.size(), x_shape_size);
    return GRAPH_FAILED;
  }
  x_range.emplace(x_range.begin() + axis, std::pair<int64_t, int64_t>{1, 1});
  y_desc->SetShapeRange(x_range);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ExpandDims, ExpandDimsInfer);

template <typename T>
static graphStatus ValidateShape(const std::vector<int64_t> &x_shape, const GeTensorPtr& tenosr, int64_t& product,
                                 int& unknow_index, GeShape& output, Operator& op) {
  int64_t dim_num = tenosr->MutableTensorDesc().MutableShape().GetDim(0);
  const T* shape_data = const_cast<T*>(reinterpret_cast<const T*>(tenosr->GetData().GetData()));
  std::vector<int64_t> out_dims = output.GetDims();
  if (shape_data == nullptr) {
    GE_OP_LOGE(op.GetName().c_str(), "truth shape data is invalid");
    return GRAPH_PARAM_INVALID;
  }

  // attr 'allowzero' will be set in onnx reshape op parser.
  int32_t allow_zero = 1;
  (void)op.GetAttr("allowzero", allow_zero);

  for (int64_t i = 0; i < dim_num; i++) {
    OP_LOGD(op.GetName().c_str(), "i: %ld, shape_data[i]: %ld.", i, shape_data[i]);
    if (shape_data[i] == -1) {
      if (unknow_index != -1) {
        string reason = "only one dim may be -1, not both dim[ " + std::to_string(unknow_index) + "] and dim[" +
                        std::to_string(i) + "]";
        GeInfershapeErrReport(op.GetName(), op.GetOpType(), kShape, reason);
        GE_OP_LOGE(op.GetName().c_str(), "Only one dim may be -1, not both dim[%lld] and dim[%lld]", unknow_index, i);
        return GRAPH_PARAM_INVALID;
      }
      unknow_index = i;
      out_dims.push_back(1);
    } else if (shape_data[i] < 0) {
      string reason = "Size[" + std::to_string(i) + "] must be non-negative";
      GeInfershapeErrReport(op.GetName(), op.GetOpType(), kShape, reason);
      GE_OP_LOGE(op.GetName().c_str(), "Size[%lld] must be non-negative", i);
      return GRAPH_PARAM_INVALID;
    } else {
      auto dim = shape_data[i];
      if ((allow_zero == 0) && (shape_data[i] == 0)) {
        dim = x_shape[i];
      }
      if (dim != 0 && product > (INT64_MAX / dim)) {
        string reason = "Mul overflow of int64, product[" + std::to_string(product) + "] dim[" +
                        std::to_string((int64_t)dim) + "]";
        GeInfershapeErrReport(op.GetName(), op.GetOpType(), kShape, reason);
        GE_OP_LOGE(op.GetName().c_str(), "Mul overflow of int64, product[%lld] dim[%lld]", product,
                   (int64_t)dim);
        return GRAPH_PARAM_INVALID;
      }
      out_dims.push_back(dim);
      product *= dim;
    }
  }

  output = GeShape(out_dims);
  return GRAPH_SUCCESS;
}

static graphStatus CaffeReshapeInferShape(const vector<int64_t>& dims, const int64_t& axis, const int64_t& num_axes,
                                          Operator& op) {
  GE_OP_LOGI(op.GetName().c_str(), "Reshape infer shape start");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto x_desc = op_desc->MutableInputDesc("x");
  auto shape_desc = op_desc->MutableInputDesc("shape");
  auto y_desc = op_desc->MutableOutputDesc("y");
  auto x_dims = x_desc->GetShape().GetDims();
  auto data_type = x_desc->GetDataType();

  if (x_dims == UNKNOWN_RANK || dims == UNKNOWN_RANK) {
    GE_OP_LOGD("Input data is unknown_rank");
    y_desc->SetShape(GeShape(UNKNOWN_RANK));
    y_desc->SetOriginShape(GeShape(UNKNOWN_RANK));
    y_desc->SetDataType(data_type);
    return GRAPH_SUCCESS;
  }

  if (x_dims == UNKNOWN_SHAPE) {
    GE_OP_LOGD("Input data is unknown_shape.");
    y_desc->SetShape(GeShape(UNKNOWN_SHAPE));
    y_desc->SetOriginShape(GeShape(UNKNOWN_SHAPE));
    y_desc->SetDataType(data_type);
    return GRAPH_SUCCESS;
  }

  int64_t inferred_axis = -1;
  int64_t constant_count = 1;
  vector<int64_t> copy_axes;

  // parsing dims
  for (size_t i = 0; i < dims.size(); ++i) {
    const int64_t shape_dim_i = dims[i];
    if (shape_dim_i == 0) {
      copy_axes.push_back(i);
    } else if (shape_dim_i == -1) {
      if (inferred_axis != -1) {
        string reason = "only one dim may be -1, not both dim[ " + std::to_string(inferred_axis) + "] and dim[" +
                        std::to_string(i) + "]";
        GeInfershapeErrReport(op.GetName(), op.GetOpType(), kAttrShape, reason);
        GE_OP_LOGE(op.GetName().c_str(), "Only one dim may be -1, not both dim[%ld] and dim[%zu]", inferred_axis, i);
        return GRAPH_PARAM_INVALID;
      }
      inferred_axis = i;
    } else {
      constant_count *= shape_dim_i;
    }
  }

  // parsing start axis and end axis
  Shape bottom_shape = op.GetInputDesc("x").GetShape();
  const int64_t bottom_shape_size = bottom_shape.GetDims().size();
  int64_t start_axis = 0;
  if (axis >= 0) {
    start_axis = axis;
  } else {
    start_axis = axis + bottom_shape_size + 1;
  }
  if (start_axis < 0 || start_axis > bottom_shape_size) {
    int64_t range = -1 - bottom_shape_size;
    // if axis >=0 , axis range [0, bottom_shape_size], else axis < 0, axis range [-1 - bottom_shape_size, -1]
    // axis range [-1 - bottom_shape_size, bottom_shape_size]
    string reason = "axis's range is not in [" + std::to_string(range) + ", " + std::to_string(bottom_shape_size) + "]";
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), kAttrAxis, reason);
    GE_OP_LOGE(op.GetName().c_str(), "reshape param axis is invalid, axis's range is not in [%ld, %ld]", range,
               bottom_shape_size);
    return GRAPH_PARAM_INVALID;
  }

  int64_t end_axis = 0;
  if (num_axes < -1) {
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), kAttrNumAxes, "it must be greater than or equal to -1");
    GE_OP_LOGE(op.GetName().c_str(), "reshape param num_axes is invalid, it must be greater than or equal to -1");
    return GRAPH_PARAM_INVALID;
  } else if (num_axes == -1) {
    end_axis = bottom_shape_size;
  } else {
    end_axis = start_axis + num_axes;
  }
  if (end_axis > bottom_shape_size) {
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), kAttrNumAxes,
                          "num_axes must be less than or equal to " + std::to_string((bottom_shape_size - start_axis)));
    GE_OP_LOGE(op.GetName().c_str(), "reshape param num_axes is invalid, it must be less than or equal to %ld",
               bottom_shape_size - start_axis);
    return GRAPH_PARAM_INVALID;
  }

  // construct top shape
  vector<int64_t> bottom_dims = bottom_shape.GetDims();
  const int64_t num_axes_replaced = end_axis - start_axis;
  const int64_t num_axes_retained = bottom_shape_size - num_axes_replaced;
  const int64_t num_new_axes = dims.size();
  vector<int64_t> top_shape(num_axes_retained + num_new_axes);
  size_t top_shape_index = 0;
  for (int64_t i = 0; i < start_axis; ++i) {
    top_shape[top_shape_index] = bottom_dims[i];
    top_shape_index++;
  }
  for (int64_t i = 0; i < num_new_axes; ++i) {
    top_shape[top_shape_index] = dims[i];
    top_shape_index++;
  }
  for (int64_t i = end_axis; i < bottom_shape_size; ++i) {
    top_shape[top_shape_index] = bottom_dims[i];
    top_shape_index++;
  }
  if (top_shape_index != top_shape.size()) {
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), "infer shape size",
                          "top_shape_index not equal to top_shape size");
    GE_OP_LOGE(op.GetName().c_str(), "reshape infer shape faied, top_shape_index not equal to top_shape size");
    return GRAPH_FAILED;
  }

  // product of [0,start_axis) + [end_axis, bottom_shape_size)
  int64_t explicit_count = constant_count;
  int64_t bottom_count_all = 1;
  for (int i = 0; i < bottom_shape_size; ++i) {
    bottom_count_all *= bottom_dims[i];
    if (i < start_axis || i >= end_axis) {
      explicit_count *= bottom_dims[i];
    }
  }

  // parsing dim 0 and -1
  for (size_t i = 0; i < copy_axes.size(); ++i) {
    const int64_t copy_axis_index = copy_axes[i];
    if ((start_axis + copy_axis_index) >= bottom_shape_size) {
      GeInfershapeErrReport(op.GetName(), op.GetOpType(), kAttrShape,
                            "there was no corresponding bottom axis for dim 0");
      GE_OP_LOGE(op.GetName().c_str(), "there was no corresponding bottom axis for dim 0.");
      return GRAPH_FAILED;
    }
    top_shape[start_axis + copy_axis_index] = bottom_dims[start_axis + copy_axis_index];
    explicit_count *= bottom_dims[start_axis + copy_axis_index];
  }
  if (inferred_axis >= 0) {
    if (bottom_count_all % explicit_count != 0) {
      string reason =
          "The shape of the input cannot be divisible by the product "
          "of the specified dimensions, the product is [" +
          std::to_string(explicit_count) + "]";
      GeInfershapeErrReport(op.GetName(), op.GetOpType(), kAttrShape, reason);
      GE_OP_LOGE(
          op.GetName().c_str(),
          "The shape of the input cannot be divisible by the product of the specified dimensions, the product is %ld",
          explicit_count);
      return GRAPH_FAILED;
    }
    const int64_t inferred_dim = bottom_count_all / explicit_count;
    top_shape[start_axis + inferred_axis] = inferred_dim;
  }

  int64_t top_count_all = 1;
  for (size_t i = 0; i < top_shape.size(); ++i) {
    top_count_all *= top_shape[i];
  }
  if (top_count_all != bottom_count_all) {
    string reason = "output tensor count [ " + std::to_string(top_count_all) + "] does not match input tensor count [" +
                    std::to_string(bottom_count_all) + "].";
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), kAttrShape, reason);
    GE_OP_LOGE(op.GetName().c_str(), "output tensor count %lld does not match input tensor count %ld.", top_count_all,
               bottom_count_all);
    return GRAPH_FAILED;
  }

  // updata output shape info
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(Shape(top_shape));
  td.SetDataType(op.GetInputDesc("x").GetDataType());
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

bool IsEmptyTensor(GeTensorDescPtr tensor_desc) {
  bool is_empty = false;
  for (const auto &dim : tensor_desc->MutableShape().GetDims()) {
    if (dim == 0) {
      is_empty = true;
      break;
    }
  }
  return is_empty;
}

template <typename T>
graphStatus GetOutShapeFromTensor(OpDescPtr op_desc, GeTensorPtr tensor, std::vector<int64_t> &v_out) {
  auto shape_desc = tensor->MutableTensorDesc();
  T* shape_data = const_cast<T*>(reinterpret_cast<const T*>(tensor->GetData().GetData()));
  if (shape_data == nullptr) {
    GE_OP_LOGE(op_desc->GetName().c_str(), "const shape data is invalid");
    return GRAPH_PARAM_INVALID;
  }
  for (int i = 0; i < shape_desc.MutableShape().GetDim(0); i++) {
    v_out.emplace_back(shape_data[i]);
  }
  return GRAPH_SUCCESS;
}

graphStatus EmptyTensorProcess(const Operator &op, const GeTensorDesc &x_desc, const GeTensorPtr &shape_tensor,
                               GeTensorDesc &out_desc) {
  GE_OP_LOGD("Start empty-tensor preprocess!");

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto shape_type = op_desc->MutableInputDesc("shape")->GetDataType();
  std::vector<int64_t> shape_shape;
  graphStatus ret = GRAPH_SUCCESS;

  if (shape_type == DT_INT32) {
    ret = GetOutShapeFromTensor<int32_t>(op_desc, shape_tensor, shape_shape);
  } else if (shape_type == DT_INT64) {
    ret = GetOutShapeFromTensor<int32_t>(op_desc, shape_tensor, shape_shape);
  } else {
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), kShapeDtype,
      "Dim type must be DT_INT32 or DT_INT64.");
    GE_OP_LOGE(op.GetName().c_str(), "Dim type must be DT_INT32 or DT_INT64.");
    return GRAPH_PARAM_INVALID;
  }

  if (ret != GRAPH_SUCCESS) {
    return ret;
  }

  GE_OP_LOGD(op.GetName().c_str(), "x shape: %s shape shape: %s", x_desc.GetShape().ToString().c_str(),
      GeShape(shape_shape).ToString().c_str());

  int64_t num_of_neg_1 = 0;
  int64_t product = 1;
  for (auto &dim : shape_shape) {
    if (dim == -1) { // -1 stand for highest dim here
      num_of_neg_1++;
      dim = 0;
    }
    product *= dim;
  }

  // check valid
  if ((num_of_neg_1 == 0 && product == 0) || (num_of_neg_1 == 1)) {
    out_desc.SetShape(GeShape(shape_shape));
    out_desc.SetOriginShape(GeShape(shape_shape));
    out_desc.SetDataType(x_desc.GetDataType());
    out_desc.SetOriginDataType(x_desc.GetDataType());
    return GRAPH_SUCCESS;
  }
  GE_OP_LOGE(op.GetName().c_str(),
    "Param is invalid!.Please check!Input shape contains -1 num is %ld, product is %ld", num_of_neg_1, product);
  return GRAPH_FAILED;
}

IMPLEMT_INFERFUNC(Reshape, ReshapeInfer) {
  bool zero_flag = false;
  vector<int64_t> attr_dims;
  if (op.GetAttr("shape", attr_dims) == GRAPH_SUCCESS) {
    for (size_t i = 0; i < attr_dims.size(); ++i) {
      if (attr_dims[i] == 0) {
        zero_flag = true;
        break;
      }
    }
  }

  std::vector<string> dep_inputs = {"shape"};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends(dep_inputs);
  auto x_desc = op_desc->MutableInputDesc("x");
  auto y_desc = op_desc->MutableOutputDesc("y");
  auto x_shape = x_desc->GetShape().GetDims();

  int64_t attr_axis = 0;
  op.GetAttr("axis", attr_axis);
  int64_t attr_num_axes = -1;
  op.GetAttr("num_axes", attr_num_axes);

  if (attr_axis != 0 || attr_num_axes != -1 || zero_flag) {
    GE_OP_LOGI(op.GetName().c_str(), "Get reshape_param successfully, shape size is %u, axis is %ld, num_axes is %ld",
               attr_dims.size(), attr_axis, attr_num_axes);
    graphStatus caffe_reshape_ret = CaffeReshapeInferShape(attr_dims, attr_axis, attr_num_axes, op);
    return caffe_reshape_ret;
  }

  GE_OP_LOGI(op.GetName().c_str(), "Reshape infer shape start");
  GeTensorPtr tensor = nullptr;
  auto node = NodeUtils::GetNodeFromOperator(op);
  if (node == nullptr) {
    CUBE_CALL_ERR_REPORT(op.GetName().c_str(), "get null node ptr!");
    return GRAPH_PARAM_INVALID;
  }
  graphStatus state = NodeUtils::GetInputConstData(node, "shape", tensor);
  if (state != GRAPH_SUCCESS) {
    GE_OP_LOGW(op.GetName().c_str(), "Op get input const data of shape failed");
    auto input_shape = op_desc->MutableInputDesc("x")->MutableShape();
    auto shape_input_desc = op_desc->MutableInputDesc("shape");
    auto shape_shape = shape_input_desc->MutableShape();
    // because shape's value stand for output shape, so it should be smaller than 1 dim
    auto shape_rank = shape_shape.GetDims().size();
    if (shape_rank > 1) {
      string reason =
          "shape dim[" + std::to_string(shape_shape.GetDims().size()) + "] should be smaller or equal than 1";
      GeInfershapeErrReport(op.GetName(), op.GetOpType(), kShape, reason);
      GE_OP_LOGE(op.GetName().c_str(), "shape dim[%zu] should be smaller or equal than 1",
                 shape_shape.GetDims().size());
      return GRAPH_PARAM_INVALID;
    }
    if (shape_shape.GetDims() != UNKNOWN_RANK && shape_shape.GetDims() != UNKNOWN_SHAPE) {
      auto x_type = op_desc->MutableInputDesc("x")->GetDataType();
      auto td = op_desc->MutableOutputDesc("y");
      int64_t rank = (shape_rank == 0) ? 0 : shape_shape.GetDims().at(0);
      td->SetShape(GeShape(std::vector<int64_t>(rank, UNKNOWN_DIM)));
      td->SetOriginShape(GeShape(std::vector<int64_t>(rank, UNKNOWN_DIM)));
      td->SetDataType(x_type);
      // calc shape range
      if (input_shape.GetDims() == UNKNOWN_RANK) {
        GE_OP_LOGD("input x is unknown rank!no way to set shape range!");
        return GRAPH_SUCCESS;
      }

      auto input_shape_size = input_shape.GetShapeSize();
      int64_t range_max = 1;
      if (input_shape_size <= 0) {
        // unknown dim , by input shape range calc output range
        std::vector<std::pair<int64_t, int64_t>> x_range;
        (void)op_desc->MutableInputDesc("x")->GetShapeRange(x_range);
        if (x_range.empty()) {
          return GRAPH_SUCCESS;
        }
        ge::array_ops::ReshapeRangeInfer(op, x_range, range_max);
      } else {
        // known dim, shape size as range_max
        range_max = input_shape_size;
      }
      range_max = (range_max > INT32_MAX) ? INT32_MAX : range_max;
      std::vector<std::pair<int64_t, int64_t>> y_range(rank, {1, range_max});
      td->SetShapeRange(y_range);
      return GRAPH_SUCCESS;
    }
    auto x_type = op_desc->MutableInputDesc("x")->GetDataType();
    auto td = op_desc->MutableOutputDesc("y");
    td->SetShape(GeShape({-2}));
    td->SetOriginShape(GeShape({-2}));
    td->SetDataType(x_type);
    return GRAPH_SUCCESS;
  }

  if (IsEmptyTensor(x_desc)) {
    return EmptyTensorProcess(op, *x_desc, tensor, *y_desc);
  }
  std::vector<std::pair<int64_t, int64_t>> x_range;
  std::vector<std::pair<int64_t, int64_t>> y_range;
  op_desc->MutableInputDesc("x")->GetShapeRange(x_range);
  int64_t product = 1;
  int unknow_index = -1;
  GeShape output_shape;

  DataType shape_type = op_desc->MutableInputDesc("shape")->GetDataType();
  int64_t shape_size = op_desc->MutableInputDesc("shape")->MutableShape().GetShapeSize();
  graphStatus ret = GRAPH_SUCCESS;
  if (shape_type == DT_INT32) {
    ret = ValidateShape<int32_t>(x_shape, tensor, product, unknow_index, output_shape, op);
  } else if (shape_type == DT_INT64) {
    ret = ValidateShape<int64_t>(x_shape, tensor, product, unknow_index, output_shape, op);
  } else if (shape_size > 0) {
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), kShapeDtype, "Dim type must be DT_INT32 or DT_INT64.");
    GE_OP_LOGE(op.GetName().c_str(), "Dim type must be DT_INT32 or DT_INT64.");
    return GRAPH_PARAM_INVALID;
  }
  if (ret != GRAPH_SUCCESS) {
    GE_OP_LOGE(op.GetName().c_str(), "ValidateShape failed, ret: %d", ret);
    return ret;
  }

  auto input_shape = op_desc->MutableInputDesc("x")->MutableShape();
  int64_t input_size = input_shape.GetShapeSize();

  // If input tensor is scalar,then input_size will return 0, assign to 1, which means convert scalar to vector.
  if (input_size == 0 && output_shape.GetShapeSize() == 1) {
    input_size = 1;
  }

  if (unknow_index != -1) {
    if (product <= 0) {
      GE_OP_LOGE(op.GetName().c_str(), "Reshape Op can't infer an empty tensor");
      return GRAPH_PARAM_INVALID;
    }
    if (input_shape.GetShapeSize() < 0) {
      GE_OP_LOGI("input x and input shape is all unknown!");
      auto td = op_desc->MutableOutputDesc("y");
      output_shape.SetDim(unknow_index, -1);
      td->SetOriginDataType(op_desc->MutableInputDesc("x")->GetDataType());
      td->SetShape(output_shape);
      td->SetOriginShape(output_shape);
      td->SetDataType(op_desc->MutableInputDesc("x")->GetDataType());
      auto max_input_dims = 1;
      // If last op does not set shape range ,do not set shape range
      if (x_range.empty()) {
        GE_OP_LOGI(op.GetName().c_str(), "input x doesnot have shape range!");
      } else {
        // If last op have already set shape range, try best to infer shape range
        ge::array_ops::ReshapeRangeInfer(op, x_range, y_range, output_shape);
      }

      td->SetShapeRange(y_range);
      return GRAPH_SUCCESS;
    }
    int64_t missing = input_size / product;
    if (product * missing != input_size) {
      string reason = "The shape of the input cannot be divisible from [" + std::to_string(product) + "]";
      GeInfershapeErrReport(op.GetName(), op.GetOpType(), kShape, reason);
      GE_OP_LOGE(op.GetName().c_str(), "The shape of the input cannot be divisible from %lld", product);
      return GRAPH_PARAM_INVALID;
    }
    output_shape.SetDim(unknow_index, missing);
  }
  auto dims = input_shape.GetDims();
  bool is_exist_unknown_shape = false;
  for (auto ele : dims) {
    is_exist_unknown_shape = (ele == -1) ? true : false;
    if (!is_exist_unknown_shape) {
      continue;
    }
  }

  if (SetScalarOutputDesc(string("x"), string("y"), op_desc, output_shape)) {
    return GRAPH_SUCCESS;
  }

  // Shape_size is 0, means shape tensor value is [], implying convert vector/scalar to scalar
  bool convert_to_scalar =
      (shape_size == 0 && (input_size == 1 || (input_size == 0 && input_shape.GetDims().size() == 0)));

  // Output_shape.GetShapeSize() > 0  and input_size <= 0 for dynamic shape
  bool shape_check_ok =
      ((input_size == output_shape.GetShapeSize()) || ((output_shape.GetShapeSize() > 0) && (input_size <= 0)) ||
       (is_exist_unknown_shape && (output_shape.GetShapeSize() > 0)));
  if (!shape_check_ok && !convert_to_scalar) {
    string reason = "Shape size is [" + std::to_string(shape_size) + "], input tensor with [" +
                    std::to_string(input_size) + "] values, is input dynamic shape [" +
                    std::to_string(is_exist_unknown_shape) + "], but requested shape has [" +
                    std::to_string(output_shape.GetShapeSize()) + "] values";
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), kShape, reason);
    GE_OP_LOGE(op.GetName().c_str(),
               "Shape size is %lld, input tensor with %lld values, is input dynamic shape :%d, but \
        requested shape has %lld values",
               shape_size, input_size, is_exist_unknown_shape, output_shape.GetShapeSize());
    return GRAPH_PARAM_INVALID;
  }

  auto td = op_desc->MutableOutputDesc("y");
  td->SetShape(output_shape);
  td->SetOriginShape(output_shape);
  td->SetDataType(op_desc->MutableInputDesc("x")->GetDataType());
  td->SetOriginDataType(op_desc->MutableInputDesc("x")->GetDataType());
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Reshape, ReshapeInfer);

IMPLEMT_INFERFORMAT_FUNC(Reshape, ReshapeInferFormat) {
  GE_OP_LOGI(op.GetName().c_str(), "Reshape infer format start");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto input_descs = op_desc->GetAllInputsDescPtr();
  auto output_descs = op_desc->GetAllOutputsDescPtr();
  for (const auto& input_desc : input_descs) {
    if (input_desc->GetShape().GetDimNum() < 4) {
      input_desc->SetOriginFormat(FORMAT_ND);
      input_desc->SetFormat(FORMAT_ND);
    }
  }
  for (const auto& output_desc : output_descs) {
    if (output_desc->GetShape().GetDimNum() < 4) {
      output_desc->SetOriginFormat(FORMAT_ND);
      output_desc->SetFormat(FORMAT_ND);
    }
  }
  (void)op_desc->DefaultInferFormat();
  for (const auto& input_desc : input_descs) {
    if (input_desc->GetShape().GetDimNum() < 4) {
      input_desc->SetOriginFormat(FORMAT_ND);
      input_desc->SetFormat(FORMAT_ND);
    }
  }
  for (const auto& output_desc : output_descs) {
    if (output_desc->GetShape().GetDimNum() < 4) {
      output_desc->SetOriginFormat(FORMAT_ND);
      output_desc->SetFormat(FORMAT_ND);
    }
  }
  return GRAPH_SUCCESS;
}
INFER_FORMAT_FUNC_REG(Reshape, ReshapeInferFormat);

IMPLEMT_VERIFIER(Squeeze, SqueezeVerify) {
  GE_OP_LOGD("Enter SqueezeVerify");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto axis = op.get_attr_axis();
  auto input_desc_x = op_desc->MutableInputDesc("x");
  auto xShape = input_desc_x->MutableShape().GetDims();

  std::vector<std::pair<int64_t, int64_t>> x_range;
  input_desc_x->GetShapeRange(x_range);
  if ((xShape != UNKNOWN_RANK) && (!x_range.empty()) && (x_range.size() != xShape.size())) {
    // if it has set shape range, it should be same with input dim num
    GE_OP_LOGE("x_shape_range num [%zu] does not match x dims_num [%zu]", x_range.size(), xShape.size());
    return GRAPH_FAILED;
  }

  auto node = NodeUtils::GetNodeFromOperator(op);
  if (node == nullptr) {
    GE_OP_LOGE("node pointer is nullptr");
    return GRAPH_FAILED;
  }
  bool is_unknow = false;
  auto status = NodeUtils::GetNodeUnknownShapeStatus(*node, is_unknow);
  if (status != GRAPH_SUCCESS) {
    GE_OP_LOGE("Get node unknown shape status failed!");
    return GRAPH_FAILED;
  }
  if (is_unknow) {
    // when input is unknown , no way to check param "axis" whether valid. Do check when running
    return GRAPH_SUCCESS;
  }

  if (axis.size() > 0) {
    for (unsigned i = 0; i < axis.size(); i++) {
      if (axis[i] < 0)
        axis[i] += xShape.size();
      bool flag = (0 <= axis[i]) && (axis[i] < static_cast<int64_t>(xShape.size()));
      if (!flag) {
        GeInfershapeErrReport(op.GetName(), op.GetOpType(), kAttrAxis,
                              "axis value is out of range of [-rank(input), rank(input)).");
        GE_OP_LOGE(op.GetName().c_str(), "axis value is out of range of [-rank(input), rank(input)).");
        return GRAPH_FAILED;
      }
      if (!(xShape[axis[i]] == 1)) {
        GeInfershapeErrReport(op.GetName(), op.GetOpType(), kShape, "input shape has dim not equal to 1.");
        GE_OP_LOGE(op.GetName().c_str(), "input shape has dim not equal to 1.");
        return GRAPH_FAILED;
      }
    }
  }
  GE_OP_LOGD("SqueezeVerify Success!");
  return GRAPH_SUCCESS;
}

VERIFY_FUNC_REG(Squeeze, SqueezeVerify);

IMPLEMT_INFERFUNC(Squeeze, SqueezeInfer) {
  GE_OP_LOGD("Enter Squeeze Infershape!");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto axis = op.get_attr_axis();
  auto input_desc_x = op_desc->MutableInputDesc("x");
  auto output_desc_y = op_desc->MutableOutputDesc("y");
  auto input_shape = input_desc_x->MutableShape();
  int64_t dim_size = input_shape.GetDimNum();
  auto x_data_type = input_desc_x->GetDataType();
  int32_t axis_num = axis.size();

  // process -2(UnknownRank)
  if (input_shape.GetDims() == UNKNOWN_RANK) {
    GE_OP_LOGD("Input x shape is -2!");
    output_desc_y->SetShape(GeShape(UNKNOWN_RANK));
    output_desc_y->SetOriginShape(GeShape(UNKNOWN_RANK));
    output_desc_y->SetDataType(x_data_type);
    return GRAPH_SUCCESS;
  }

  std::vector<std::pair<int64_t, int64_t>> x_range;
  std::vector<std::pair<int64_t, int64_t>> y_range;
  input_desc_x->GetShapeRange(x_range);

  std::unordered_set<int32_t> squeeze_dims;
  for (int32_t i = 0; i < axis_num; ++i) {
    int32_t dim = axis[i];
    if (dim < -dim_size || dim >= dim_size) {
      string reason = "Tried to squeeze dim index[" + std::to_string(dim) + "] for tensor with [" +
                      std::to_string(dim_size) + "] dimensions";
      GeInfershapeErrReport(op.GetName(), op.GetOpType(), kAttrAxis, reason);
      GE_OP_LOGE(op.GetName().c_str(), "Tried to squeeze dim index[%d] for tensor with [%lld] dimensions", dim,
                 dim_size);
      return GRAPH_FAILED;
    }
    if (dim < 0) {
      dim = dim_size + dim;
    }
    squeeze_dims.insert(dim);
  }

  vector<int64_t> out_shape;
  for (int i = 0; i < dim_size; i++) {
    auto exist_dim = input_shape.GetDim(i);
    // If squeeze_set is non-empty, only squeeze those dimensions.
    if (!squeeze_dims.empty()) {
      if (squeeze_dims.count(i) > 0) {
        // If dim is -1 and been pointed by axis , do think -1 is 1.because no method to do verify
        if (exist_dim != 1 && exist_dim != UNKNOWN_DIM) {
          string reason = "Can not squeeze dim[" + std::to_string(i) + "], expected a dimension of 1, got [" +
                          std::to_string(exist_dim) + "]";
          GeInfershapeErrReport(op.GetName(), op.GetOpType(), kShape, reason);
          GE_OP_LOGE(op.GetName().c_str(), "Can not squeeze dim[%d], expected a dimension of 1, got %lld", i,
                     exist_dim);
          return GRAPH_FAILED;
        }
      } else {
        out_shape.emplace_back(exist_dim);
        // after verified, it has ensure x_range ele num is same with dims num
        if (!x_range.empty()) {
          y_range.emplace_back(x_range[i]);
        }
      }
    } else {
      // Copy over all non-1-length dimensions.
      // here no methed to ensure which -1 is 1, so do warning
      if (exist_dim != 1) {
        if (exist_dim == -1) {
          GE_OP_LOGW("the [%d] dim is -1, it will not execute squeeze on it! maybe influence result", exist_dim);
        }
        out_shape.emplace_back(exist_dim);
        // after verified, it has ensure x_range ele num is same with dims num
        if (!x_range.empty()) {
          y_range.emplace_back(x_range[i]);
        }
      }
    }
  }

  output_desc_y->SetShape(GeShape(out_shape));
  output_desc_y->SetOriginShape(GeShape(out_shape));
  output_desc_y->SetDataType(x_data_type);
  if (!y_range.empty()) {
    output_desc_y->SetShapeRange(y_range);
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Squeeze, SqueezeInfer);

IMPLEMT_INFERFUNC(Unsqueeze, UnsqueezeInfer) {
  auto axis_arr = op.get_attr_axes();
  auto axis_nums = axis_arr.size();
  if (axis_nums <= 0) {
    string reason = "Axis_nums[" + std::to_string(axis_nums) + "] must be greater than 0";
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), kAttrAxis, reason);
    GE_OP_LOGE(op.GetName().c_str(), "Axis_nums[%zu] must be greater than 0", axis_nums);
    return GRAPH_PARAM_INVALID;
  }
  std::unordered_set<int64_t> values(axis_arr.begin(), axis_arr.end());
  if (values.size() != axis_arr.size()) {
    string reason = "Axis attribute must not contain any duplicates.";
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), kAttrAxis, reason);
    GE_OP_LOGE(op.GetName().c_str(), "Axis attribute must not contain any duplicates.");
    return GRAPH_PARAM_INVALID;
  }
  Shape input_shape = op.get_input_desc_x().GetShape();
  int64_t dim_num = input_shape.GetDimNum() + axis_nums;
  std::vector<int64_t> vec_dim(dim_num, 0);

  for (size_t i = 0; i < axis_nums; i++) {
    int64_t axis = axis_arr[i];
    if ((axis < -dim_num) || (axis > (dim_num - 1))) {
      string reason = "axis[" + std::to_string(axis_nums) + "]'s range is not in [" + std::to_string(-dim_num) + ", " +
                      std::to_string(dim_num - 1) + "]";
      GeInfershapeErrReport(op.GetName(), op.GetOpType(), kAttrAxis, reason);
      GE_OP_LOGE(op.GetName().c_str(), "Axis %ld not in [%ld, %ld]", axis, -dim_num, dim_num);
      return GRAPH_PARAM_INVALID;
    }
    if (axis < 0) {
      axis += dim_num;
    }
    vec_dim.at(axis) = 1;
  }
  int64_t index = 0;
  for (int64_t i = 0; i < dim_num; i++) {
    if (vec_dim.at(i) != 1) {
      vec_dim.at(i) = input_shape.GetDim(index);
      index++;
    }
  }

  TensorDesc td = op.get_output_desc_y();
  td.SetShape(Shape(vec_dim));
  td.SetDataType(op.get_input_desc_x().GetDataType());
  (void)op.update_output_desc_y(td);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Unsqueeze, UnsqueezeInfer);

IMPLEMT_INFERFUNC(Rank, RankInfer) {
  OP_LOGI(op.GetName().c_str(), "Rank infershape start");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto output_desc_y = op_desc->MutableOutputDesc("y");
  std::vector<int64_t> oShapeVector;
  output_desc_y->SetShape(GeShape(oShapeVector));
  output_desc_y->SetOriginShape(GeShape(oShapeVector));
  output_desc_y->SetDataType(DT_INT32);
  OP_LOGI(op.GetName().c_str(), "Rank infershape end");
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Rank, RankInfer);

IMPLEMT_INFERFUNC(Size, SizeInfer) {
  OP_LOGI(op.GetName().c_str(), "Size infershape start");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto output_desc_y = op_desc->MutableOutputDesc("y");
  std::vector<int64_t> oShapeVector;
  output_desc_y->SetShape(GeShape(oShapeVector));

  DataType out_type = DT_INT32;
  GeAttrValue out_type_value;
  op_desc->GetAttr("dtype", out_type_value);
  out_type_value.GetValue<DataType>(out_type);
  output_desc_y->SetDataType(out_type);
  OP_LOGI(op.GetName().c_str(), "Size infershape end");
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Size, SizeInfer);

COMMON_INFER_FUNC_REG(Data, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
COMMON_INFER_FUNC_REG(PlaceHolder, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
COMMON_INFER_FUNC_REG(End, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));

IMPLEMT_INFERFUNC(PlaceholderWithDefault, PlaceholderWithDefaultInfer) {
  TensorDesc input_desc = op.GetInputDesc("x");
  auto dims = input_desc.GetShape().GetDims();
  auto data_type = input_desc.GetDataType();

  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetDataType(ge::DataType(data_type));
  output_desc.SetShape(Shape(dims));
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(PlaceholderWithDefault, PlaceholderWithDefaultInfer);

IMPLEMT_INFERFUNC(Shape, ShapeInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto td = op_desc->MutableOutputDesc("y");
  auto input_dims = op_desc->MutableInputDesc("x")->MutableShape().GetDims();
  if (input_dims == UNKNOWN_RANK) {
    td->SetShape(ge::GeShape(UNKNOWN_SHAPE));
    td->SetOriginShape(ge::GeShape(UNKNOWN_SHAPE));
    td->SetShapeRange(std::vector<std::pair<int64_t, int64_t>>{{1,kMaxDimNum}});
  } else {
    int64_t size = static_cast<int64_t>(input_dims.size());
    std::vector<int64_t> size_v{size};
    td->SetShape(ge::GeShape(size_v));
    td->SetOriginShape(ge::GeShape(size_v));
  }
  uint32_t out_type = DT_INT32;
  (void)op.GetAttr("dtype", out_type);
  td->SetDataType((DataType)out_type);

  std::vector<std::pair<int64_t, int64_t>> inRange;
  op_desc->MutableInputDesc("x")->GetShapeRange(inRange);
  if (!inRange.empty()) {
    std::vector<int64_t> pre_op_range;
    pre_op_range.resize(2*inRange.size());
    for (int i = 0; i < pre_op_range.size(); i = i + 2) {
      pre_op_range[i] = inRange[i/2].first;
      pre_op_range[i + 1] = inRange[i/2].second;
    }
    ge::AttrUtils::SetListInt(*td, kPreOpInputShapeRange, pre_op_range);
    OP_LOGD(op.GetName().c_str(), "Shape op set pre_op_range success");
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Shape, ShapeInfer);

IMPLEMT_INFERFUNC(ShapeN, ShapeNInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  for (size_t i = 0; i < op.GetInputsSize(); i++) {
    auto td = op_desc->MutableOutputDesc(i);
    auto input_dims = op_desc->MutableInputDesc(i)->MutableShape().GetDims();
    if (input_dims == UNKNOWN_RANK) {
      td->SetShape(ge::GeShape(UNKNOWN_SHAPE));
      td->SetOriginShape(ge::GeShape(UNKNOWN_SHAPE));
      td->SetShapeRange(std::vector<std::pair<int64_t, int64_t>>{{1,kMaxDimNum}});
    } else {
      int64_t size = static_cast<int64_t>(input_dims.size());
      GE_OP_LOGD(op.GetName().c_str(), "output value %ld", size);
      std::vector<int64_t> size_v{size};
      td->SetShape(ge::GeShape(size_v));
      td->SetOriginShape(ge::GeShape(size_v));
    }
    uint32_t out_type = DT_INT32;
    (void)op.GetAttr("dtype", out_type);
    td->SetDataType((DataType)out_type);

    std::vector<std::pair<int64_t, int64_t>> inRange;
    op_desc->MutableInputDesc(i)->GetShapeRange(inRange);
    if (!inRange.empty()) {
      std::vector<int64_t> pre_op_range;
      pre_op_range.resize(2*inRange.size());
      for (int i = 0; i < pre_op_range.size(); i = i + 2) {
        pre_op_range[i] = inRange[i/2].first;
        pre_op_range[i + 1] = inRange[i/2].second;
      }
      ge::AttrUtils::SetListInt(*td, kPreOpInputShapeRange, pre_op_range);
      OP_LOGD(op.GetName().c_str(), "ShapeN op set pre_op_range success");
    }
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ShapeN, ShapeNInfer);

IMPLEMT_INFERFUNC(IdentityN, IdentityNInfer) {
  OP_LOGI(op.GetName().c_str(), "IdentityN infershape start");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  for (size_t i = 0; i < op.GetInputsSize(); i++) {

    auto input_desc = op_desc->MutableInputDesc(i);
    auto input_dims = input_desc->MutableShape().GetDims();
    auto output_desc = op_desc->MutableOutputDesc(i);
    auto intput_dtype = input_desc->GetDataType();

    std::vector<std::pair<int64_t, int64_t>> input_range;
    input_desc->GetShapeRange(input_range);
    output_desc->SetShape(GeShape(input_dims));
    output_desc->SetOriginShape(GeShape(input_dims));
    output_desc->SetDataType(intput_dtype);
    output_desc->SetShapeRange(input_range);
  }

  OP_LOGI(op.GetName().c_str(), "IdentityN infershape end");

  return GRAPH_SUCCESS;

}

INFER_FUNC_REG(IdentityN, IdentityNInfer);

IMPLEMT_INFERFUNC(Identity, IdentityInfer) {
  OP_LOGI(op.GetName().c_str(), "Identity infershape start");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc_x = op_desc->MutableInputDesc(0);
  auto output_desc_y = op_desc->MutableOutputDesc(0);

  std::vector<int64_t> vec_dim;
  vec_dim = input_desc_x->MutableShape().GetDims();

  std::vector<std::pair<int64_t, int64_t>> x_range;
  input_desc_x->GetShapeRange(x_range);

  DataType data_type = input_desc_x->GetDataType();

  output_desc_y->SetDataType(data_type);
  output_desc_y->SetShape(GeShape(vec_dim));
  output_desc_y->SetOriginShape(GeShape(vec_dim));
  output_desc_y->SetShapeRange(x_range);
  OP_LOGI(op.GetName().c_str(), "Identity infershape end");
  return GRAPH_SUCCESS;

}

INFER_FUNC_REG(Identity, IdentityInfer);

IMPLEMT_INFERFUNC(ReadVariableOp, ReadVariableOpInfer) {
  TensorDesc input_desc = op.GetInputDesc("x");
  (void)op.UpdateOutputDesc("y", input_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ReadVariableOp, ReadVariableOpInfer);

template <typename T>
static void CaclDims(const Tensor& data, std::vector<int64_t>& vec_dim) {
  int32_t size = data.GetSize() / sizeof(T);
  for (int32_t i = 0; i < size; i++) {
    T dim = *((T*)data.GetData() + i);
    if (dim != 0) {
      vec_dim.push_back(dim);
    } else {
      vec_dim.clear();
      break;
    }
  }
}

template <typename T>
static void CaclDims(const GeTensorPtr& data, std::vector<int64_t>& vec_dim) {
  int32_t size = data->GetData().GetSize() / sizeof(T);
  for (int32_t i = 0; i < size; i++) {
    void* data_ptr = (void*)data->GetData().GetData();
    if (data_ptr == nullptr) {
      return;
    }
    T dim = *((T*)data_ptr + i);
    if (dim != 0) {
      vec_dim.push_back(dim);
    } else {
      vec_dim.clear();
      break;
    }
  }
}

IMPLEMT_INFERFUNC(Empty, EmptyInfer) {
  OP_LOGI(op.GetName().c_str(), "Empty infershape start");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  std::vector<string> dep_inputs = {"shape"};
  op_desc->SetOpInferDepends(dep_inputs);
  auto input_desc_shape = op_desc->MutableInputDesc("shape");
  auto output_desc_y = op_desc->MutableOutputDesc("y");
  auto dtype = op.get_attr_dtype();

  std::vector<std::pair<int64_t, int64_t>> shape_range;
  std::vector<std::pair<int64_t, int64_t>> y_range;
  input_desc_shape->GetShapeRange(shape_range);

  DataType data_type = input_desc_shape->GetDataType();
  std::vector<int64_t> vec_dim;
  if (data_type == DT_INT32) {
    vec_dim = input_desc_shape->MutableShape().GetDims();
  } else {
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), "dtype", "Empty only support shape type 'DT_INT32'");
    GE_OP_LOGE(op.GetName().c_str(), "Empty only support shape type 'DT_INT32'");
    return GRAPH_PARAM_INVALID;
  }

  if (vec_dim == UNKNOWN_RANK) {
    GE_OP_LOGD(op.GetName().c_str(), "all inputs are unknown rank!");
    output_desc_y->SetShape(GeShape(UNKNOWN_SHAPE));
    output_desc_y->SetOriginShape(GeShape(UNKNOWN_SHAPE));
    output_desc_y->SetDataType((DataType)dtype);
    return GRAPH_SUCCESS;
  }

  if (vec_dim == UNKNOWN_SHAPE) {
    GE_OP_LOGD(op.GetName().c_str(), "shape is unknown shape!");
    std::pair<int64_t, int64_t> pair({1, shape_range.size()});
    y_range.emplace_back(pair);
    output_desc_y->SetShape(GeShape(UNKNOWN_SHAPE));
    output_desc_y->SetOriginShape(GeShape(UNKNOWN_SHAPE));
    output_desc_y->SetDataType((DataType)dtype);
    output_desc_y->SetShapeRange(y_range);
    return GRAPH_SUCCESS;
  }

  auto node = NodeUtils::GetNodeFromOperator(op);
  if (node == nullptr) {
    CUBE_CALL_ERR_REPORT(op.GetName().c_str(), "Get null node ptr.");
    return GRAPH_PARAM_INVALID;
  }

  GeTensorPtr shape_data;
  std::vector<int64_t> shape_dims;
  auto result = NodeUtils::GetInputConstData(node, "shape", shape_data);
  if(result == GRAPH_SUCCESS) {
    DataType data_type = shape_data->GetTensorDesc().GetDataType();
    if (data_type == DT_INT32) {
      CaclDims<int32_t>(shape_data,shape_dims);
    } else if (data_type == DT_INT64) {
      CaclDims<int64_t>(shape_data, shape_dims);
    }

    OP_LOGD(op.GetName().c_str(), "Get input const data success.");
    std::pair<int64_t, int64_t> pair({1,shape_range.size()});
    y_range.emplace_back(pair);
    output_desc_y->SetShape(GeShape(shape_dims));
    output_desc_y->SetOriginShape(GeShape(shape_dims));
    output_desc_y->SetDataType((DataType)dtype);
    output_desc_y->SetShapeRange(y_range);
    return GRAPH_SUCCESS;
  } else {
    OP_LOGD(op.GetName().c_str(), "Get input const data failed!");
    std::pair<int64_t, int64_t> pair({1,shape_range.size()});
    y_range.emplace_back(pair);
    output_desc_y->SetShape(GeShape(UNKNOWN_SHAPE));
    output_desc_y->SetOriginShape(GeShape(UNKNOWN_SHAPE));
    output_desc_y->SetDataType((DataType)dtype);
    output_desc_y->SetShapeRange(y_range);
    return GRAPH_SUCCESS;
  }

  output_desc_y->SetShape(GeShape(vec_dim));
  output_desc_y->SetOriginShape(GeShape(vec_dim));
  output_desc_y->SetDataType((DataType)dtype);
  OP_LOGD(op.GetName().c_str(), "Empty infershape end");
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Empty, EmptyInfer);

IMPLEMT_INFERFUNC(LowerBound, LowerBoundInfer) {
  TensorDesc sorted_x_desc = op.GetInputDesc("sorted_x");
  TensorDesc values_desc = op.GetInputDesc("values");
  Shape unused_shape;
  if (WithRank(sorted_x_desc, 2, unused_shape, op.GetName().c_str()) !=
      GRAPH_SUCCESS) {
    ShapeErrReport(0, op.GetName(),
                   DebugString(sorted_x_desc.GetShape().GetDims()), "2D");
    OP_LOGE(op.GetName().c_str(), "Input [sorted_x] rank must be 2.");
    return GRAPH_FAILED;
  }
  if (WithRank(values_desc, 2, unused_shape, op.GetName().c_str()) !=
      GRAPH_SUCCESS) {
    ShapeErrReport(1, op.GetName(),
                   DebugString(values_desc.GetShape().GetDims()), "2D");
    OP_LOGE(op.GetName().c_str(), "Input [values] rank must be 2.");
    return GRAPH_FAILED;
  }

  DataType out_type;
  if (op.GetAttr("out_type", out_type) != GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "out_type");
    OP_LOGE(op.GetName().c_str(), "Get attr [out_type] failed.");
    return GRAPH_FAILED;
  }
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetDataType(out_type);
  y_desc.SetShape(values_desc.GetShape());
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    OpsOPUpdateErrReport(op.GetName(), "y");
    OP_LOGE(op.GetName().c_str(), "Update [y] desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(LowerBound, LowerBoundInfer);

IMPLEMT_INFERFUNC(Where, WhereInfer) {
  OpDescPtr op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr x_desc = op_desc->MutableInputDesc(0);

  GeShape x_shape;
  if (WithRankAtLeast(x_desc, 1, x_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "input x must be at least 1D.");
    return GRAPH_FAILED;
  }

  if (WithRankAtMost(x_desc, 5, x_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "input x must be at most 5D.");
    return GRAPH_FAILED;
  }

  GeTensorDescPtr y_desc = op_desc->MutableOutputDesc(0);
  y_desc->SetDataType(DT_INT64);

  vector<int64_t> y_shape;
  auto input_dims = x_shape.GetDims();
  int64_t input_shape_size = x_shape.GetShapeSize();
  if (input_shape_size != UNKNOWN_DIM) {
    // input shape: known
    y_shape.push_back(UNKNOWN_DIM);
    y_shape.push_back(input_dims.size());

    std::vector<std::pair<int64_t, int64_t>> range;
    int64_t dims_num = x_shape.GetDimNum();
    range.emplace_back(std::make_pair(1, input_shape_size));
    range.emplace_back(std::make_pair(dims_num, dims_num));
    y_desc->SetShapeRange(range);
  } else {
    if (input_dims == UNKNOWN_RANK) {
      // input shape: unknown rank
      y_shape.push_back(UNKNOWN_DIM);
      y_shape.push_back(UNKNOWN_DIM);
    } else {
      // input shape: unknown dims
      y_shape.push_back(UNKNOWN_DIM);
      y_shape.push_back(input_dims.size());
    }
  }

  y_desc->SetShape(GeShape(y_shape));
  y_desc->SetOriginShape(GeShape(y_shape));
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Where, WhereInfer);

IMPLEMT_INFERFUNC(Fingerprint, FingerprintInfer) {
  Shape unused;
  if (WithRankAtLeast(op.GetInputDesc(0), 1, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Input data must be at least 1D.");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Input method rank must be 0.");
    return GRAPH_FAILED;
  }
  int64_t batch = op.GetInputDesc(0).GetShape().GetDim(0);
  int64_t fingerprint_size;
  Tensor method_tensor;
  uint32_t offset = sizeof(uint64_t) * 2;
  int status = op.GetInputConstData("method", method_tensor);
  if (status != GRAPH_SUCCESS) {
    fingerprint_size = UNKNOWN_DIM;
  } else {
    int64_t method_dim;
    method_dim = method_tensor.GetTensorDesc().GetShape().GetDimNum();
    if (method_dim != 0) {
      CUBE_INNER_ERR_REPORT(op.GetName().c_str(),
                            "Input method_tensor rank must be 0, real value is [%ld].", method_dim);
      return GRAPH_FAILED;
    }
    std::string method_string;
    const char *method_data = reinterpret_cast<const char*>(method_tensor.GetData() + offset);

    method_string = method_data;
    if (method_string != "farmhash64") {
      CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Unsupported method, real value is [%s]", method_string.c_str());
      return GRAPH_FAILED;
    }
    fingerprint_size = sizeof(uint64_t);
  }

  Shape shape({batch, fingerprint_size});
  TensorDesc desc = op.GetOutputDesc("y");
  desc.SetShape(shape);
  desc.SetDataType(DT_UINT8);
  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Fail to update output y.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Fingerprint, FingerprintInfer);

IMPLEMT_INFERFUNC(TransShape, TransShapeInfer) {
  TensorDesc y_desc = op.GetOutputDesc("y");
  vector<int64_t> output_shape;
  auto ret = op.GetAttr("outShape", output_shape);
  if (ret != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Failed to get attribute value.");
    return GRAPH_SUCCESS;
  }
  y_desc.SetShape(Shape(output_shape));
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TransShape, TransShapeInfer);

IMPLEMT_INFERFUNC(EditDistance, EditDistanceInfer) {
  std::vector<std::string> input_infer_depends = {"hypothesis_shape", "truth_shape"};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends(input_infer_depends);
  auto hypothesis_indices_desc = op.GetInputDesc(0);
  auto hypothesis_values_desc = op.GetInputDesc(1);
  auto hypothesis_shape_desc = op.GetInputDesc(2);

  if (ValidateSparseTensor(hypothesis_indices_desc, hypothesis_values_desc, hypothesis_shape_desc,
                           op.GetName().c_str()) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "input hypothesis is not sparse tensor");
    return GRAPH_FAILED;
  }

  auto truth_indices_desc = op.GetInputDesc(3);
  auto truth_values_desc = op.GetInputDesc(4);
  auto truth_shape_desc = op.GetInputDesc(5);

  if (ValidateSparseTensor(truth_indices_desc, truth_values_desc, truth_shape_desc, op.GetName().c_str()) !=
      GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "input truth is not sparse tensor");
    return GRAPH_FAILED;
  }

  auto output_desc = op.GetOutputDesc("output");

  Tensor hypothesis_shape_tensor, truth_shape_tensor;
  if (op.GetInputConstData("hypothesis_shape", hypothesis_shape_tensor) != GRAPH_SUCCESS) {
    OP_LOGW(op.GetName().c_str(), "failed to get tensor from input hypothesis shape, return unknown shape");
    output_desc.SetShape(ge::Shape(ge::UNKNOWN_RANK));
    op.UpdateOutputDesc("output", output_desc);
    return GRAPH_SUCCESS;
  }

  if (op.GetInputConstData("truth_shape", truth_shape_tensor) != GRAPH_SUCCESS) {
    OP_LOGW(op.GetName().c_str(), "failed to get tensor from input truth shape, return unknown shape");
    output_desc.SetShape(ge::Shape(ge::UNKNOWN_RANK));
    op.UpdateOutputDesc("output", output_desc);
    return GRAPH_SUCCESS;
  }

  auto hypothesis_shape_num_elements = hypothesis_shape_desc.GetShape().GetShapeSize();
  auto truth_shape_num_elements = truth_shape_desc.GetShape().GetShapeSize();
  if (hypothesis_shape_num_elements != truth_shape_num_elements) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(),
                          "Num elements of hypothesis_shape does not match truth_shape: %ld vs %ld",
                          hypothesis_shape_num_elements, truth_shape_num_elements);
    return GRAPH_PARAM_INVALID;
  }

  int64_t* hypothesis_shape_data = reinterpret_cast<int64_t*>(hypothesis_shape_tensor.GetData());
  if (hypothesis_shape_data == nullptr) {
    CUBE_CALL_ERR_REPORT(op.GetName().c_str(), "hypothesis shape data is invalid");
    return GRAPH_PARAM_INVALID;
  }

  int64_t* truth_shape_data = reinterpret_cast<int64_t*>(truth_shape_tensor.GetData());
  if (truth_shape_data == nullptr) {
    CUBE_CALL_ERR_REPORT(op.GetName().c_str(), "truth shape data is invalid");
    return GRAPH_PARAM_INVALID;
  }

  std::vector<int64_t> output_dims(hypothesis_shape_num_elements - 1);
  for (uint64_t i = 0; i < output_dims.size(); ++i) {
    output_dims[i] = std::max(hypothesis_shape_data[i], truth_shape_data[i]);
  }

  output_desc.SetShape(Shape(output_dims));
  if (op.UpdateOutputDesc("output", output_desc) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "failed to update output output desc");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(EditDistance, EditDistanceInfer);

// ----------------SortV2 Begin-------------------
IMPLEMT_INFERFUNC(SortV2, SortV2InferShape) {
  TensorDesc tensordesc_input = op.GetInputDesc("x");
  Shape input_shape = tensordesc_input.GetShape();
  DataType input_dtype = tensordesc_input.GetDataType();
  std::vector<int64_t> dims_input = input_shape.GetDims();

  TensorDesc tensordesc_output1 = op.GetOutputDesc("y");

  tensordesc_output1.SetShape(ge::Shape(dims_input));

  tensordesc_output1.SetDataType(input_dtype);

  (void)op.UpdateOutputDesc("y", tensordesc_output1);

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SortV2, SortV2Verify) { return GRAPH_SUCCESS; }

INFER_FUNC_REG(SortV2, SortV2InferShape);
VERIFY_FUNC_REG(SortV2, SortV2Verify);
// ----------------SortV2 END---------------------

// ----------------Expand Begin-------------------
template<typename T> static bool ExpandCalDim(const Tensor &data,
                                               std::vector<int64_t> &vec_dim,
                                               std::vector<int64_t> &vec_x) {
  uint32_t size_shape = data.GetSize() / sizeof(T);
  uint32_t size_x = vec_x.size();
  if (size_shape < size_x) {
    uint32_t diff = size_x - size_shape;
    for (int32_t i = 0; i < size_x; i++) {
      if (i < diff) {
        vec_dim.push_back(vec_x[i]);
      } else {
        T dim = *((T *)data.GetData() + (i - diff));
        if ((vec_x[i] != dim) && (vec_x[i] != 1) && (dim != 1)) {
          return false;
        }
        if (vec_x[i] > dim) {
          vec_dim.push_back(vec_x[i]);
        } else {
          vec_dim.push_back(dim);
        }
      }
    }
  } else {
    uint32_t diff = size_shape - size_x;
    for (int32_t i = 0; i < size_shape; i++) {
      T dim = *((T *)data.GetData() + i);
      if (i < diff) {
        vec_dim.push_back(dim);
      } else {
        if ((vec_x[i - diff] != dim) && (vec_x[i-diff] != 1) && (dim != 1)) {
          return false;
        }
        if (vec_x[i - diff] > dim) {
          vec_dim.push_back(vec_x[i - diff]);
        } else {
          vec_dim.push_back(dim);
        }
      }
    }
  }
  return true;
}

IMPLEMT_INFERFUNC(Expand, ExpandInferShape) {
  const vector<string> depend_names = {"shape"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  Shape x_shape = op.GetInputDesc("x").GetShape();
  DataType x_dtype = op.GetInputDesc("x").GetDataType();
  std::vector <int64_t> dims_x = x_shape.GetDims();
  Tensor data;
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  std::vector <int64_t> vec_dim;
  TensorDesc td = op.GetOutputDesc("y");
  if (op.GetInputConstData("shape", data) != GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "Get constValue failed of [shape]");
    auto shape_desc = op_info->MutableInputDesc("shape");
    vector<int64_t> shapedims = shape_desc->MutableShape().GetDims();
    size_t dim_num = shapedims.size();

    DataType input_dtype = op.GetInputDesc("x").GetDataType();

    if (dim_num > 1) {
      OP_LOGE(op.GetName().c_str(), "The dim numbers of constnode are more than one.");
      return GRAPH_FAILED;
    }

    std::vector<int64_t> shape_vector;
    std::vector<std::pair<int64_t, int64_t>> range_vector;

    int64_t max_len = dims_x.size();
    if (shapedims[0] > max_len) {
      max_len = shapedims[0];
    }

    for (int64_t item = 0; item < max_len; ++item) {
      shape_vector.push_back(-1);
      range_vector.push_back(std::make_pair(1, -1));
    }
    auto output_desc = op_info->MutableOutputDesc("y");
    output_desc->SetShape(GeShape(shape_vector));
    output_desc->SetShapeRange(range_vector);
    output_desc->SetDataType(input_dtype);
    return GRAPH_SUCCESS;
    }else {

    DataType data_type = data.GetTensorDesc().GetDataType();
    if (data_type == DT_INT32) {
      if (!ExpandCalDim <int32_t>(data, vec_dim, dims_x)) {
        OP_LOGE(op.GetName().c_str(), "Data shape are not compatible!");
        return GRAPH_FAILED;
      }
    } else if (data_type == DT_INT64) {
      if (!ExpandCalDim <int64_t>(data, vec_dim, dims_x)) {
        OP_LOGE(op.GetName().c_str(), "Data shape are not compatible!");
        return GRAPH_FAILED;
      }
    } else {
      OP_LOGE(op.GetName().c_str(), "Data type not supported!");
      return GRAPH_PARAM_INVALID;
    }

    td.SetShape(ge::Shape(vec_dim));
    td.SetDataType(x_dtype);
    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
  }
}

INFER_FUNC_REG(Expand, ExpandInferShape);
// ----------------Expand END---------------------

// ----------------NonZero Begin-------------------
IMPLEMT_INFERFUNC(NonZero, NonZeroInfer) {
  OpDescPtr op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr x_input = op_desc->MutableInputDesc(0);
  GeShape x_shape = x_input->GetShape();
  GeTensorDescPtr y_desc = op_desc->MutableOutputDesc(0);
  y_desc->SetDataType(DT_INT64);
  bool transpose = false;
  if (op.GetAttr("transpose", transpose) != GRAPH_SUCCESS) {
    OP_LOGW(op.GetName().c_str(), "Failed to get attr[transpose]. Set attr[transpose] to false.");
  }
  if (x_shape.GetDims() == UNKNOWN_RANK) {
    y_desc->SetShape(x_shape);
    y_desc->SetOriginShape(x_shape);
  } else {
    std::vector<std::pair<int64_t, int64_t>> range;
    if (transpose == false) {
      y_desc->SetShape(GeShape({UNKNOWN_DIM, x_shape.GetDimNum()}));
      y_desc->SetOriginShape(GeShape({UNKNOWN_DIM, x_shape.GetDimNum()}));
      if (x_shape.GetShapeSize() == UNKNOWN_DIM) {
        range.emplace_back(std::make_pair(1, -1));
      } else {
        range.emplace_back(std::make_pair(1, x_shape.GetShapeSize()));
      }
      range.emplace_back(std::make_pair(x_shape.GetDimNum(), x_shape.GetDimNum()));
    } else {
      y_desc->SetShape(GeShape({x_shape.GetDimNum(), UNKNOWN_DIM}));
      y_desc->SetOriginShape(GeShape({x_shape.GetDimNum(), UNKNOWN_DIM}));
      range.emplace_back(std::make_pair(x_shape.GetDimNum(), x_shape.GetDimNum()));
      if (x_shape.GetShapeSize() == UNKNOWN_DIM) {
        range.emplace_back(std::make_pair(1, -1));
      } else {
        range.emplace_back(std::make_pair(1, x_shape.GetShapeSize()));
      }
    }
    y_desc->SetShapeRange(range);
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(NonZero, NonZeroInfer);
// ----------------NonZero End-------------------

// ----------------ExpandD Begin-------------------
IMPLEMT_COMMON_INFERFUNC(ExpandDInferShape) {
  Shape x_shape = op.GetInputDesc("x").GetShape();
  DataType x_dtype = op.GetInputDesc("x").GetDataType();
  std::vector<int64_t> shape;
  op.GetAttr("shape", shape);
  std::vector<int64_t> dims_x = x_shape.GetDims();
  TensorDesc td = op.GetOutputDesc("y");

  std::vector<int64_t> dim_vec;
  if (shape.size() < dims_x.size()) {
    std::vector<int64_t> dims_tmp = shape;
    shape = dims_x;
    dims_x = dims_tmp;
  }
  if (shape.size() != dims_x.size()) {
    int dec = shape.size() - dims_x.size();
    for (int i = 0; i < dec; i++) {
      dims_x.insert(dims_x.begin(), (int64_t)1);
    }
  }
  for (size_t i = 0; i < shape.size(); i++) {
    if ((shape[i] != dims_x[i]) && (shape[i] != 1) && (dims_x[i] != 1)) {
      CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "The input shape and attr shape are not compatible.");
      return GRAPH_FAILED;
    }
    if (shape[i] > dims_x[i]) {
      dim_vec.push_back(shape[i]);
    } else {
      dim_vec.push_back(dims_x[i]);
    }
  }
  td.SetShape(ge::Shape(dim_vec));
  td.SetDataType(x_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ExpandD, ExpandDInferShape);
// ----------------ExpandD END---------------------
}  // namespace ge