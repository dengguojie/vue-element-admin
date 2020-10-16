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
 * @file  array_ops.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "inc/array_ops.h"
#include <unordered_set>
#include <utility>

#include "op_log.h"
#include "common_shape_fns.h"
#include "array_ops_shape_fns.h"
#include "graph/utils/tensor_adapter.h"
#include "./util/error_util.h"
#include "util/util.h"

namespace ge {
const char *const kShape = "shape";
const char *const kShapeDtype = "shape dtype";
const char *const kAttrShape = "attr shape";
const char *const kAttrDtype = "attr dtype";
const char *const kAttrAxis = "attr axis";
const char *const kAttrNumAxes = "attr num_axes";

IMPLEMT_INFERFUNC(MatrixBandPart, MatrixBandPartInfer)
{
  if (UnchangedShape(op, "x", "y") != GRAPH_SUCCESS) {
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

INFER_FUNC_REG(MatrixBandPart, MatrixBandPartInfer);

IMPLEMT_INFERFUNC(UniqueWithCounts, UniqueWithCountsInfer) {
  DataType y_type = op.GetInputDesc("x").GetDataType();
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(Shape({UNKNOWN_DIM}));
  output_desc.SetDataType(y_type);
  if (op.UpdateOutputDesc("y", output_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  DataType type;
  if (op.GetAttr("out_idx", type) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Op get attr out_idx failed");
    return GRAPH_FAILED;
  }

  output_desc = op.GetOutputDesc("count");
  output_desc.SetShape(Shape({UNKNOWN_DIM}));
  output_desc.SetDataType(type);

  if (op.UpdateOutputDesc("count", output_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  output_desc = op.GetOutputDesc("idx");
  output_desc.SetDataType(type);
  if (op.UpdateOutputDesc("idx", output_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return UnchangedShape(op, "x", "idx");
}

INFER_FUNC_REG(UniqueWithCounts, UniqueWithCountsInfer);

IMPLEMT_INFERFUNC(Unique, UniqueInfer) {
  OpDescPtr op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr x_input = op_desc->MutableInputDesc(0);

  GeShape x_shape;
  if (WithRank(x_input, 1, x_shape) != GRAPH_SUCCESS) {
    ShapeErrReport(0,
                   op.GetName(),
                   DebugString(x_input->GetShape().GetDims()),
                   "1D");
    OP_LOGE(op.GetName().c_str(), "input x must be 1-D");
    return GRAPH_FAILED;
  }

  DataType idx_type;
  if (op.GetAttr("out_idx", idx_type) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Op get attr out_idx failed");
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
    ShapeErrReport(0,
                   op.GetName(),
                   DebugString(op.GetInputDesc(0).GetShape().GetDims()),
                   "at least 1D");
    OP_LOGE(op.GetName().c_str(), "input x must be more than 1-D");
    return GRAPH_FAILED;
  }

  Shape axis_shape;
  if (WithRank(op.GetInputDesc(1), 1, axis_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    ShapeErrReport(1,
                   op.GetName(),
                   DebugString(op.GetInputDesc(1).GetShape().GetDims()),
                   "1D");
    OP_LOGE(op.GetName().c_str(), "input axis must be 1-D");
    return GRAPH_FAILED;
  }

  DataType type = op.GetInputDesc("x").GetDataType();
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(Shape({ge::UNKNOWN_DIM}));
  y_desc.SetDataType(type);
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  DataType idx_type;
  if (op.GetAttr("out_idx", idx_type) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Op get attr out_idx failed");
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
    ShapeErrReport(0,
                   op.GetName(),
                   DebugString(op.GetInputDesc(0).GetShape().GetDims()),
                   "1D");
    OP_LOGE(op.GetName().c_str(), "input x must be 1-D");
    return GRAPH_FAILED;
  }

  DataType type = op.GetInputDesc("x").GetDataType();
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(op.GetInputDesc(0).GetShape());
  y_desc.SetDataType(type);
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(InvertPermutation, InvertPermutationInfer);

IMPLEMT_INFERFUNC(CheckNumerics, CheckNumericsInfer) {
  DataType type = op.GetInputDesc("x").GetDataType();
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(op.GetInputDesc(0).GetShape());
  y_desc.SetDataType(type);
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(CheckNumerics, CheckNumericsInfer);

IMPLEMT_INFERFUNC(UnravelIndex, UnravelIndexInfer) {
  auto indices_input_dsesc = op.GetInputDesc(0);
  auto dims_input_desc = op.GetInputDesc(1);

  Shape dims_shape;
  if (WithRank(dims_input_desc, 1, dims_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
  ShapeErrReport(1,
                 op.GetName(),
                 DebugString(op.GetInputDesc(1).GetShape().GetDims()),
                 "1D");
    OP_LOGE(op.GetName().c_str(), "dims input rank must be 1D.");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> out_dims;
  Shape indices_shape = indices_input_dsesc.GetShape();
  if (indices_shape.GetDims() == ge::UNKNOWN_SHAPE) {
    // unknown shape
    out_dims = ge::UNKNOWN_SHAPE;
    OP_LOGW(op.GetName().c_str(),
        "Indices input is unknown shape, set output unknown shape.");
  } else if (indices_shape.GetDimNum() == 0) {
    out_dims.push_back(dims_shape.GetDim(0));
  } else {
    int64_t dim_size = indices_shape.GetShapeSize();
    out_dims.push_back(dims_shape.GetDim(0));
    out_dims.push_back(dim_size);
  }

  TensorDesc out_desc = op.GetOutputDesc("y");
  out_desc.SetShape(Shape(out_dims));
  out_desc.SetDataType(indices_input_dsesc.GetDataType());
  return op.UpdateOutputDesc("y", out_desc);
}

INFER_FUNC_REG(UnravelIndex, UnravelIndexInfer);

IMPLEMT_INFERFUNC(UpperBound, UpperBoundInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc(0), 2, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    ShapeErrReport(0,
                   op.GetName(),
                   DebugString(op.GetInputDesc(0).GetShape().GetDims()),
                   "2D");
    OP_LOGE(op.GetName().c_str(), "sorted_x input rank must be 2D.");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 2, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    ShapeErrReport(1,
                   op.GetName(),
                   DebugString(op.GetInputDesc(1).GetShape().GetDims()),
                   "2D");
    OP_LOGE(op.GetName().c_str(), "values input rank must be 2D.");
    return GRAPH_FAILED;
  }

  DataType type;
  if (op.GetAttr("out_type", type) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Op get attr out_type failed");
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
  output_desc.SetShape(Shape({UNKNOWN_DIM}));
  output_desc.SetDataType(y_type);
  if (op.UpdateOutputDesc("y", output_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  DataType type;
  if (op.GetAttr("out_idx", type) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Op get attr out_idx failed");
    return GRAPH_FAILED;
  }

  output_desc = op.GetOutputDesc("count");
  output_desc.SetShape(Shape({UNKNOWN_DIM}));
  output_desc.SetDataType(type);

  if (op.UpdateOutputDesc("count", output_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  output_desc = op.GetOutputDesc("idx");
  output_desc.SetDataType(type);
  if (op.UpdateOutputDesc("idx", output_desc) != GRAPH_SUCCESS) {
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
  auto x = op.GetInputDesc("x");
  auto y = op.GetInputDesc("y");

  Shape unused_shape;
  if (WithRank(x, 1, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    ShapeErrReport(0,
                   op.GetName(),
                   DebugString(x.GetShape().GetDims()),
                   "1D");
    OP_LOGE(op.GetName().c_str(), "input x rank must be 1.");
    return GRAPH_FAILED;
  }

  if (WithRank(y, 1, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    ShapeErrReport(1,
                   op.GetName(),
                   DebugString(y.GetShape().GetDims()),
                   "1D");
    OP_LOGE(op.GetName().c_str(), "input y rank must be 1.");
    return GRAPH_FAILED;
  }

  DataType out_type = op.GetInputDesc("x").GetDataType();
  DataType idx_type;
  if (op.GetAttr("out_idx", idx_type) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Op get attr out_idx failed");
    return GRAPH_FAILED;
  }

  Shape result;
  Vector(ge::UNKNOWN_DIM, result);
  TensorDesc desc_out = op.GetOutputDesc("out");
  desc_out.SetShape(Shape(result));
  desc_out.SetDataType(out_type);
  if (op.UpdateOutputDesc("out", desc_out) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update out desc failed.");
    return GRAPH_FAILED;
  }

  TensorDesc desc_idx = op.GetOutputDesc("idx");
  desc_idx.SetShape(Shape(result));
  desc_idx.SetDataType(idx_type);
  if (op.UpdateOutputDesc("idx", desc_idx) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update idx desc failed.");
    return GRAPH_FAILED;
  }
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
    ShapeErrReport(1,
                   op.GetName(),
                   DebugString(seq_lengths_desc.GetShape().GetDims()),
                   "1D");
    OP_LOGE(op.GetName().c_str(),
            "seq_lengths's rank should be equal to 1.");
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
    OP_LOGE(op.GetName().c_str(), "Failed to get attribute value.");
    return GRAPH_FAILED;
  }
  int64_t batch_dim;
  if (op.GetAttr("batch_dim", batch_dim) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Failed to get attribute value.");
    return GRAPH_FAILED;
  }
  int64_t input_rank = input_shape.GetDimNum();
  if (seq_dim >= input_rank) {
    AttrValueErrReport("seq_dim",
                       op.GetName(),
                       Strcat(seq_dim),
                       Strcat("< the rank of 0th input[", input_rank,
                              "], 0th input shape is ",
                              DebugString(input_shape.GetDims())));

    OP_LOGE(op.GetName().c_str(),
            "seq_dim must be < rank of x: %ld vs. %ld",
            seq_dim,
            input_rank);
    return GRAPH_FAILED;
  }
  if (batch_dim >= input_rank) {
    AttrValueErrReport("batch_dim",
                       op.GetName(),
                       Strcat(batch_dim),
                       Strcat("< the rank of 0th input[", input_rank,
                              "], 0th input shape is ",
                              DebugString(input_shape.GetDims())));
    OP_LOGE(op.GetName().c_str(),
            "batch_dim must be < rank of x: %ld vs. %ld",
            batch_dim,
            input_rank);
    return GRAPH_FAILED;
  }

  int64_t batch_dim_dim = input_shape.GetDim(batch_dim);
  if (Merge(batch_dim_dim, seq_lengths_shape.GetDim(0), batch_dim_dim)
        != GRAPH_SUCCESS) {
    string err_msg = Strcat("the batch_dim[", batch_dim, "]th dim value[",
      batch_dim_dim, "] of 0th input should be equal to 0th dim value[",
      seq_lengths_shape.GetDim(0), "] of 1th input. 0th input shape",
      DebugString(input_shape.GetDims()), ", 1th input shape",
      DebugString(seq_lengths_shape.GetDims()));
    InferShapeOtherErrReport(op.GetName(), err_msg);
    OP_LOGE(op.GetName().c_str(),
            "x.dims[batch_dim] and seq_lengths" \
                " should have the same length.");
    return GRAPH_FAILED;
  }

  // Replace batch_dim of input with batch_size
  Shape y_shape;
  ReplaceDim(input_shape, batch_dim, batch_dim_dim, y_shape, op.GetName().c_str());

  DataType x_type = op.GetInputDesc("x").GetDataType();
  y_desc.SetDataType(x_type);
  y_desc.SetShape(y_shape);
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Failed to update y desc.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ReverseSequence, ReverseSequenceInfer);

IMPLEMT_INFERFUNC(Const, ConstInfer) {
    auto value = op.get_attr_value();
    auto valDesc = value.GetTensorDesc();
    auto dims = valDesc.GetShape().GetDims();
    auto attrDtype = valDesc.GetDataType();

    TensorDesc outDesc = op.get_output_desc_y();
    outDesc.SetDataType(ge::DataType(attrDtype));
    outDesc.SetShape(Shape(dims));
    (void)op.update_output_desc_y(outDesc);

    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Const, ConstInfer);

IMPLEMT_INFERFUNC(Constant, ConstantInfer) {
    auto value = op.get_attr_value();
    auto valDesc = value.GetTensorDesc();
    auto dims = valDesc.GetShape().GetDims();
    auto attrDtype = valDesc.GetDataType();

    TensorDesc outDesc = op.get_output_desc_y();
    outDesc.SetDataType(ge::DataType(attrDtype));
    outDesc.SetShape(Shape(dims));
    (void)op.update_output_desc_y(outDesc);

    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Constant, ConstantInfer);

graphStatus ConstAndConstantInferFormat(ge::Operator &op) {
  OP_LOGI(op.GetName().c_str(), "Const infer format start");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto format = op_desc->MutableOutputDesc(0)->GetOriginFormat();
  ConstGeTensorPtr tensor_value;
  if (!AttrUtils::GetTensor(op_desc, "value", tensor_value)) {
    OP_LOGE(op.GetName().c_str(), "Get attr value failed!");
    return GRAPH_FAILED;
  }
  if (!tensor_value) {
    OP_LOGE(op.GetName().c_str(), "attr tensor is not exist!");
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
    TensorDesc input_desc = op.GetInputDesc("x");
    (void)op.UpdateOutputDesc("y", input_desc);
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
    TensorDesc x1_desc = op.GetInputDesc("x1");
    TensorDesc x2_desc = op.GetInputDesc("x2");

    bool data_type_check = ((x1_desc.GetDataType() != DT_INT32 &&
                            x1_desc.GetDataType() != DT_INT64) ||
                            (x2_desc.GetDataType() != DT_INT32 &&
                            x2_desc.GetDataType() != DT_INT64));
    if (data_type_check) {
        string reason = "x1[" + std::to_string(x1_desc.GetDataType()) +
            "] + and + x2[" + std::to_string(x1_desc.GetDataType()) + "] must DT_INT32 or DT_INT64";
        GeInfershapeErrReport(op.GetName(), op.GetOpType(), "dtype", reason);
        OP_LOGE(op.GetName().c_str(),
            "Data type check fail. x1[%u] and x2[%u] must DT_INT32 or DT_INT64",
            x1_desc.GetDataType(), x2_desc.GetDataType());
        return GRAPH_PARAM_INVALID;
    }

    TensorDesc y_desc = op.GetOutputDesc("y");
    auto x1_dims = x1_desc.GetShape().GetDims();
    auto x2_dims = x2_desc.GetShape().GetDims();

    if (x1_dims.size() > 1 || x2_dims.size() > 1) {
        string reason = "x1[" + std::to_string(x1_dims.size()) +
            "] + and + x2[" + std::to_string(x2_dims.size()) + "] must be less than or equal to 1";
        GeInfershapeErrReport(op.GetName(), op.GetOpType(), "dims", reason);
        OP_LOGE(op.GetName().c_str(),
            "Size check fail. x1[%u] and x2[%u] must be less than or equal to 1",
            x1_dims.size(), x2_dims.size());
        return GRAPH_PARAM_INVALID;
    }

    if (x1_dims.empty() ) {
        y_desc.SetShape(ge::Shape(x2_dims));
    } else if (x2_dims.empty()) {
        y_desc.SetShape(ge::Shape(x1_dims));
    } else {
        auto dims = x1_dims[0] > x2_dims[0] ? x1_dims : x2_dims;
        y_desc.SetShape(ge::Shape(dims));
    }

    y_desc.SetDataType(x1_desc.GetDataType());
    (void)op.UpdateOutputDesc("y", y_desc);

    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BroadcastArgs, BroadcastArgsInferFunc);

IMPLEMT_INFERFUNC(BroadcastGradientArgs, BroadcastGradientArgsInfer) {
    DataType x1_type = op.GetInputDesc("x1").GetDataType();
    TensorDesc y1_tensor_desc = op.GetOutputDesc("y1");
    y1_tensor_desc.SetDataType(x1_type);
    (void)op.UpdateOutputDesc("y1", y1_tensor_desc);

    DataType x2_type = op.GetInputDesc("x2").GetDataType();
    TensorDesc y2_tensor_desc = op.GetOutputDesc("y2");
    y2_tensor_desc.SetDataType(x2_type);
    (void)op.UpdateOutputDesc("y2", y2_tensor_desc);
    return GRAPH_SUCCESS;

}

INFER_FUNC_REG(BroadcastGradientArgs, BroadcastGradientArgsInfer);

IMPLEMT_INFERFUNC(PreventGradient, PreventGradientInferFunc) {
    TensorDesc input_desc = op.GetInputDesc("x");
    (void)op.UpdateOutputDesc("y", input_desc);
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(PreventGradient, PreventGradientInferFunc);

IMPLEMT_INFERFUNC(StopGradient, StopGradientInferFunc) {
    TensorDesc input_desc = op.GetInputDesc("x");
    (void)op.UpdateOutputDesc("y", input_desc);
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StopGradient, StopGradientInferFunc);

IMPLEMT_INFERFUNC(ExpandDims, ExpandDimsInfer) {
  std::vector<string> dep_inputs = {"axis"};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends(dep_inputs);
  auto axis_type = op_desc->MutableInputDesc("axis")->GetDataType();
  auto x_type = op_desc->MutableInputDesc("x")->GetDataType();

  if (axis_type != DT_INT32 && axis_type != DT_INT64) {
    string reason = "axis dtype[" + std::to_string(axis_type) + "] must int32 or int64";
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), kAttrDtype, reason);
    OP_LOGE(op.GetName().c_str(), "axis dtype[%d] must int32 or int64", axis_type);
    return GRAPH_PARAM_INVALID;
  }


  bool is_x_unknonwn_rank =
    op_desc->MutableInputDesc("x")->MutableShape().GetDims() == std::vector<int64_t>({-2}) ? true : false;
  if (is_x_unknonwn_rank) {
    OP_LOGD("input x shape is unknown rank!");
    auto y_desc = op_desc->MutableOutputDesc("y");
    y_desc->SetUnknownDimNumShape();
    y_desc->SetDataType(x_type);
    y_desc->SetOriginDataType(x_type);
    return GRAPH_SUCCESS;
  }

  int64_t axis_nums = op.get_input_desc_axis().GetShape().GetShapeSize();

  if (axis_nums != 1) {
    // Shape::GetDims().size() == 0, means it's a scalar, its shape is [].
    if (!(axis_nums == 0 && op.get_input_desc_axis().GetShape().GetDims().size() == 0)) {
      string reason = "axis input must be a tensor with a single value, but [" + std::to_string(axis_nums) + "] nums";
      GeInfershapeErrReport(op.GetName(), op.GetOpType(), "axis", reason);
      OP_LOGE(op.GetName().c_str(), "'axis' input must be a tensor with a single value, but %d nums", axis_nums);
      return GRAPH_PARAM_INVALID;
    }
  }

  Tensor data;
  graphStatus status = op.GetInputConstData("axis", data);
  if (status != GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "Op get input const data of axis failed");
    std::vector<int64_t> out_dims(op_desc->MutableInputDesc("x")->MutableShape().GetDims().size() + 1, -1);
    auto y_desc = op_desc->MutableOutputDesc("y");
    y_desc->SetShape(GeShape(out_dims));
    y_desc->SetOriginShape(GeShape(out_dims));
    y_desc->SetDataType(x_type);
    y_desc->SetOriginDataType(x_type);
    return GRAPH_SUCCESS;
  }

  int64_t axis = 0;
  if (axis_type == DT_INT32) {
    axis = *(reinterpret_cast<int32_t *>(data.GetData()));
  } else if (axis_type == DT_INT64) {
    axis = *(reinterpret_cast<int64_t *>(data.GetData()));
  } else {
    // not supported
  }

  std::vector<int64_t> vec_dim;
  Shape input_shape = op.get_input_desc_x().GetShape();
  int32_t dim_num = input_shape.GetDimNum();
  if (axis < -1 - dim_num || axis > dim_num) {
    string reason = "axis[" + std::to_string(axis) +
        "] is not in [" + std::to_string(-1 - dim_num) + " , " + std::to_string(dim_num) + "]";
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), "axis", reason);
    OP_LOGE(op.GetName().c_str(), "axis[%d] is not in [%d, %d]",
           axis, -1 - dim_num, dim_num);
    return GRAPH_PARAM_INVALID;
  }

  if (axis < 0) {
    axis += dim_num + 1;
  }
  for (int i = 0; i < dim_num; i++) {
    vec_dim.push_back(input_shape.GetDim(i));
  }
  vec_dim.emplace(vec_dim.begin() + axis, 1);
  auto y_desc = op_desc->MutableOutputDesc("y");
  TensorDesc td = op.get_output_desc_y();
  y_desc->SetShape(GeShape(vec_dim));
  y_desc->SetOriginShape(GeShape(vec_dim));
  y_desc->SetDataType(x_type);
  y_desc->SetOriginDataType(x_type);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ExpandDims, ExpandDimsInfer);

template<typename T>
static graphStatus ValidateShape(const Tensor& tenosr,
                                 int64_t& product,
                                 int& unknow_index,
                                 GeShape& output,
                                 Operator& op) {
  int64_t dim_num = tenosr.GetTensorDesc().GetShape().GetDim(0);
  T *shape_data = const_cast<T *>(reinterpret_cast<const T *>(tenosr.GetData()));
  std::vector<int64_t> out_dims = output.GetDims();
  if (shape_data == nullptr) {
    OP_LOGE(op.GetName().c_str(), "truth shape data is invalid");
    return GRAPH_PARAM_INVALID;
  }

  for (int64_t i = 0; i < dim_num; i++) {
    if (shape_data[i] == -1) {
      if (unknow_index != -1) {
        string reason = "only one dim may be -1, not both dim[ " + std::to_string(unknow_index) +
                        "] and dim[" + std::to_string(i) +"]";
        GeInfershapeErrReport(op.GetName(), op.GetOpType(), kShape, reason);
        OP_LOGE(op.GetName().c_str(), "Only one dim may be -1, not both dim[%lld] and dim[%lld]",
                              unknow_index, i);
        return GRAPH_PARAM_INVALID;
      }
      unknow_index = i;
      out_dims.push_back(1);
    } else if(shape_data[i] < 0) {
      string reason = "Size[" + std::to_string(i) + "] must be non-negative";
      GeInfershapeErrReport(op.GetName(), op.GetOpType(), kShape, reason);
      OP_LOGE(op.GetName().c_str(), "Size[%lld] must be non-negative", i);
      return GRAPH_PARAM_INVALID;
    } else {
      if (shape_data[i] != 0 && product > (INT64_MAX / shape_data[i])) {
        string reason = "Mul overflow of int64, product[" + std::to_string(product) +
            "] shape_data[" + std::to_string((int64_t)shape_data[i]) + "]";
        GeInfershapeErrReport(op.GetName(), op.GetOpType(), kShape, reason);
        OP_LOGE(op.GetName().c_str(), "Mul overflow of int64, product[%lld] shape_data[%lld]",
          product, (int64_t)shape_data[i]);
        return GRAPH_PARAM_INVALID;
      }
      out_dims.push_back(shape_data[i]);
      product *= shape_data[i];
    }
  }

  output = GeShape(out_dims);
  return GRAPH_SUCCESS;
}

static graphStatus CaffeReshapeInferShape(const vector<int64_t>& dims,
                                                 const int64_t& axis,
                                                 const int64_t& num_axes,
                                                 Operator& op) {
  OP_LOGI(op.GetName().c_str(), "Reshape infer shape start");
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
        string reason = "only one dim may be -1, not both dim[ " + std::to_string(inferred_axis) +
                        "] and dim[" + std::to_string(i) + "]";
        GeInfershapeErrReport(op.GetName(), op.GetOpType(), kAttrShape, reason);
        OP_LOGE(op.GetName().c_str(), "Only one dim may be -1, not both dim[%ld] and dim[%zu]", inferred_axis, i);
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
    OP_LOGE(op.GetName().c_str(),
        "reshape param axis is invalid, axis's range is not in [%ld, %ld]", range, bottom_shape_size);
    return GRAPH_PARAM_INVALID;
  }
  int64_t end_axis = 0;
  if (num_axes < -1) {
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), kAttrNumAxes, "it must be greater than or equal to -1");
    OP_LOGE(op.GetName().c_str(), "reshape param num_axes is invalid, it must be greater than or equal to -1");
    return GRAPH_PARAM_INVALID;
  } else if (num_axes == -1){
    end_axis = bottom_shape_size;
  } else {
    end_axis = start_axis + num_axes;
  }
  if (end_axis > bottom_shape_size) {
    GeInfershapeErrReport(op.GetName(), op.GetOpType(),
        kAttrNumAxes, "num_axes must be less than or equal to " + std::to_string((bottom_shape_size - start_axis)));
    OP_LOGE(op.GetName().c_str(),
        "reshape param num_axes is invalid, it must be less than or equal to %ld", bottom_shape_size - start_axis);
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
    GeInfershapeErrReport(op.GetName(), op.GetOpType(),
        "infer shape size", "top_shape_index not equal to top_shape size");
    OP_LOGE(op.GetName().c_str(), "reshape infer shape faied, top_shape_index not equal to top_shape size");
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
      GeInfershapeErrReport(
          op.GetName(), op.GetOpType(), kAttrShape, "there was no corresponding bottom axis for dim 0");
      OP_LOGE(op.GetName().c_str(), "there was no corresponding bottom axis for dim 0.");
      return GRAPH_FAILED;
    }
    top_shape[start_axis + copy_axis_index] = bottom_dims[start_axis + copy_axis_index];
    explicit_count *= bottom_dims[start_axis + copy_axis_index];
  }
  if (inferred_axis >= 0) {
    if (bottom_count_all % explicit_count != 0) {
      string reason = "The shape of the input cannot be divisible by the product "
                      "of the specified dimensions, the product is [" + std::to_string(explicit_count) + "]";
      GeInfershapeErrReport(op.GetName(), op.GetOpType(), kAttrShape, reason);
      OP_LOGE(op.GetName().c_str(),
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
    string reason = "output tensor count [ " +  std::to_string(top_count_all) +
                    "] does not match input tensor count [" + std::to_string(bottom_count_all) + "].";
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), kAttrShape, reason);
    OP_LOGE(op.GetName().c_str(), "output tensor count %lld does not match input tensor count %ld.",
      top_count_all, bottom_count_all );
    return GRAPH_FAILED;
  }

  // updata output shape info
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(Shape(top_shape));
  td.SetDataType(op.GetInputDesc("x").GetDataType());
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
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

  int64_t attr_axis = 0;
  op.GetAttr("axis", attr_axis);
  int64_t attr_num_axes = -1;
  op.GetAttr("num_axes", attr_num_axes);

  if (attr_axis != 0 || attr_num_axes != -1 || zero_flag) {
    OP_LOGI(op.GetName().c_str(), "Get reshape_param successfully, shape size is %u, axis is %ld, num_axes is %ld",
      attr_dims.size(), attr_axis, attr_num_axes);
    graphStatus caffe_reshape_ret = CaffeReshapeInferShape(attr_dims, attr_axis, attr_num_axes, op);
    return caffe_reshape_ret;
  }

  OP_LOGI(op.GetName().c_str(), "Reshape infer shape start");
  Tensor tensor;
  graphStatus state = op.GetInputConstData("shape", tensor);
  if (state != GRAPH_SUCCESS) {
    OP_LOGW(op.GetName().c_str(), "Op get input const data of shape failed");
    auto shape_input_desc = op_desc->MutableInputDesc("shape");
    auto shape_shape = shape_input_desc->MutableShape();
    // because shape's value stand for output shape, so it should be smaller than 1 dim
    if (shape_shape.GetDims().size() > 1) {
      string reason =
          "shape dim[" + std::to_string(shape_shape.GetDims().size()) + "] should be smaller or equal than 1";
      GeInfershapeErrReport(op.GetName(), op.GetOpType(), kShape, reason);
      OP_LOGE(op.GetName().c_str(), "shape dim[%zu] should be smaller or equal than 1",
        shape_shape.GetDims().size());
      return GRAPH_PARAM_INVALID;
    }
    auto x_type = op_desc->MutableInputDesc("x")->GetDataType();
    auto td = op_desc->MutableOutputDesc("y");
    td->SetShape(GeShape({-2}));
    td->SetOriginShape(GeShape({-2}));
    td->SetDataType(x_type);
    return GRAPH_SUCCESS;
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
    ret = ValidateShape<int32_t>(tensor, product, unknow_index, output_shape, op);
  } else if (shape_type == DT_INT64) {
    ret = ValidateShape<int64_t>(tensor, product, unknow_index, output_shape, op);
  } else if (shape_size > 0) {
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), kShapeDtype, "Dim type must be DT_INT32 or DT_INT64.");
    OP_LOGE(op.GetName().c_str(), "Dim type must be DT_INT32 or DT_INT64.");
    return GRAPH_PARAM_INVALID;
  }
  if (ret != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "ValidateShape failed, ret: %d", ret);
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
      OP_LOGE(op.GetName().c_str(), "Reshape Op can't infer an empty tensor");
      return GRAPH_PARAM_INVALID;
    }
    if (input_shape.GetShapeSize() < 0) {
      OP_LOGI("input x and input shape is all unknown!");
      auto td = op_desc->MutableOutputDesc("y");
      output_shape.SetDim(unknow_index, -1);
      td->SetOriginDataType(op_desc->MutableInputDesc("x")->GetDataType());
      td->SetShape(output_shape);
      td->SetOriginShape(output_shape);
      td->SetDataType(op_desc->MutableInputDesc("x")->GetDataType());
      auto max_input_dims = 1;
      // If last op does not set shape range ,do not set shape range
      if (x_range.empty()) {
        OP_LOGI(op.GetName().c_str(), "input x doesnot have shape range!");
      } else {
        // If last op have already set shape range, try best to infer shape range
        for (auto &pair : x_range) {
          if (pair.second < 0) {
            max_input_dims = -1;
            break;
          }
          max_input_dims *= pair.second;
        }
        int64_t left = max_input_dims;
        for (auto dim : output_shape.GetDims()) {
          if (dim < 0) {
            if (max_input_dims == -1) {
              y_range.emplace_back(std::pair<int64_t, int64_t>(1, max_input_dims));
            } else {
              y_range.emplace_back(std::pair<int64_t, int64_t>(1, left));
            }
          } else {
            y_range.emplace_back(std::pair<int64_t, int64_t>(dim, dim));
            if (max_input_dims == -1) {
              continue;
            }
            left = static_cast<int64_t>(((double)max_input_dims + 0.5) / dim);
          }
        }
      }

      td->SetShapeRange(y_range);
      return GRAPH_SUCCESS;
    }
    int64_t missing = input_size / product;
    if (product * missing != input_size) {
      string reason = "The shape of the input cannot be divisible from [" + std::to_string(product) + "]";
      GeInfershapeErrReport(op.GetName(), op.GetOpType(), kShape, reason);
      OP_LOGE(op.GetName().c_str(), "The shape of the input cannot be divisible from %lld", product);
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
  // Shape_size is 0, means shape tensor value is [], implying convert vector/scalar to scalar
  bool convert_to_scalar = (shape_size == 0 &&
                            (input_size == 1 || (input_size == 0 && input_shape.GetDims().size() == 0)));

  // Output_shape.GetShapeSize() > 0  and input_size <= 0 for dynamic shape
  bool shape_check_ok = ((input_size == output_shape.GetShapeSize()) ||
                         ((output_shape.GetShapeSize() > 0) && (input_size <= 0)) ||
                         (is_exist_unknown_shape && (output_shape.GetShapeSize() > 0))
                         );
  if (!shape_check_ok && !convert_to_scalar) {
    string reason = "Shape size is [" + std::to_string(shape_size) + "], input tensor with ["
        + std::to_string(input_size) + "] values, is input dynamic shape [" + std::to_string(is_exist_unknown_shape) +
        "], but requested shape has [" + std::to_string(output_shape.GetShapeSize()) + "] values";
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), kShape, reason);
    OP_LOGE(op.GetName().c_str(), "Shape size is %lld, input tensor with %lld values, is input dynamic shape :%d, but \
       requested shape has %lld values", shape_size, input_size, is_exist_unknown_shape, output_shape.GetShapeSize());
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
  OP_LOGI(op.GetName().c_str(), "Reshape infer format start");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto input_descs = op_desc->GetAllInputsDescPtr();
  auto output_descs = op_desc->GetAllOutputsDescPtr();
  for (const auto &input_desc : input_descs) {
    if (input_desc->GetShape().GetDimNum() < 4) {
      input_desc->SetOriginFormat(FORMAT_ND);
      input_desc->SetFormat(FORMAT_ND);
    }
  }
  for (const auto &output_desc : output_descs) {
    if (output_desc->GetShape().GetDimNum() < 4) {
      output_desc->SetOriginFormat(FORMAT_ND);
      output_desc->SetFormat(FORMAT_ND);
    }
  }
  (void)op_desc->DefaultInferFormat();
  for (const auto &input_desc : input_descs) {
    if (input_desc->GetShape().GetDimNum() < 4) {
      input_desc->SetOriginFormat(FORMAT_ND);
      input_desc->SetFormat(FORMAT_ND);
    }
  }
  for (const auto &output_desc : output_descs) {
    if (output_desc->GetShape().GetDimNum() < 4) {
      output_desc->SetOriginFormat(FORMAT_ND);
      output_desc->SetFormat(FORMAT_ND);
    }
  }
  return GRAPH_SUCCESS;
}
INFER_FORMAT_FUNC_REG(Reshape, ReshapeInferFormat);

IMPLEMT_VERIFIER(Squeeze, SqueezeVerify) {
    auto axis = op.get_attr_axis();
    auto xShape = op.get_input_desc_x().GetShape().GetDims();

    if (axis.size() > 0) {
        for (unsigned i = 0; i < axis.size(); i++) {
            if (axis[i] < 0)
                axis[i] += xShape.size();
            bool flag = (0 <= axis[i]) && (axis[i] < static_cast<int64_t>(xShape.size()));
            if (!flag) {
                GeInfershapeErrReport(op.GetName(), op.GetOpType(), kAttrAxis,
                    "axis value is out of range of [-rank(input), rank(input)).");
                OP_LOGE(op.GetName().c_str(), "axis value is out of range of [-rank(input), rank(input)).");
                return GRAPH_FAILED;
            }
            if (!(xShape[axis[i]] == 1)) {
                GeInfershapeErrReport(
                    op.GetName(), op.GetOpType(), kShape, "input shape has dim not equal to 1.");
                OP_LOGE(op.GetName().c_str(), "input shape has dim not equal to 1.");
                return GRAPH_FAILED;
            }
        }
    }
    return GRAPH_SUCCESS;
}

VERIFY_FUNC_REG(Squeeze, SqueezeVerify);

IMPLEMT_INFERFUNC(Squeeze, SqueezeInfer) {
  auto axis = op.get_attr_axis();
  auto input_shape = op.get_input_desc_x().GetShape();
  int64_t dim_size = op.get_input_desc_x().GetShape().GetDimNum();

  int32_t axis_num = axis.size();
  std::unordered_set<int32_t> squeeze_dims;
  for (int32_t i = 0; i < axis_num; ++i) {
    int32_t dim = axis[i];
    if (dim < -dim_size || dim >= dim_size) {
      string reason = "Tried to squeeze dim index[" + std::to_string(dim) +
          "] for tensor with [" + std::to_string(dim_size) + "] dimensions";
      GeInfershapeErrReport(op.GetName(), op.GetOpType(), kAttrAxis, reason);
      OP_LOGE(op.GetName().c_str(), "Tried to squeeze dim index[%d] for tensor with [%lld] dimensions", dim, dim_size);
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
        if (exist_dim != 1) {
          string reason = "Can not squeeze dim[" + std::to_string(i) +
              "], expected a dimension of 1, got [" + std::to_string(exist_dim) + "]";
          GeInfershapeErrReport(op.GetName(), op.GetOpType(), kShape, reason);
          OP_LOGE(op.GetName().c_str(), "Can not squeeze dim[%d], expected a dimension of 1, got %lld", i, exist_dim);
          return GRAPH_FAILED;
        }
      } else {
        out_shape.push_back(exist_dim);
      }
    } else {
      // Copy over all non-1-length dimensions.
      if (exist_dim != 1) {
        out_shape.push_back(exist_dim);
      }
    }
  }

  TensorDesc td = op.get_output_desc_y();
  td.SetShape(Shape(out_shape));
  td.SetDataType(op.get_input_desc_x().GetDataType());
  (void)op.update_output_desc_y(td);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Squeeze, SqueezeInfer);

IMPLEMT_INFERFUNC(Unsqueeze, UnsqueezeInfer) {
  auto axis_arr = op.get_attr_axes();
  auto axis_nums = axis_arr.size();
  if (axis_nums <= 0) {
    string reason = "Axis_nums[" + std::to_string(axis_nums) + "] must be greater than 0";
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), kAttrAxis, reason);
    OP_LOGE(op.GetName().c_str(), "Axis_nums[%zu] must be greater than 0", axis_nums);
    return GRAPH_PARAM_INVALID;
  }

  std::vector<int64_t> vec_dim;
  Shape input_shape = op.get_input_desc_x().GetShape();
  int64_t dim_num = input_shape.GetDimNum();

  for (int64_t i = 0; i < dim_num; i++) {
    vec_dim.push_back(input_shape.GetDim(i));
  }

  for (size_t i = 0; i < axis_nums; i++) {
    int64_t axis = axis_arr[i];
    if ((axis < -dim_num) || (axis > dim_num)) {
      string reason = "axis[" + std::to_string(axis_nums) + "]'s range is not in [" + std::to_string(-dim_num) +
          ", " +  std::to_string(dim_num) + "]";
      GeInfershapeErrReport(op.GetName(), op.GetOpType(), kAttrAxis, reason);
      OP_LOGE(op.GetName().c_str(), "Axis %ld not in [%ld, %ld]",
              axis, -dim_num, dim_num);
      return GRAPH_PARAM_INVALID;
    }

    if (axis < 0) {
      axis += dim_num + 1;
    }

    vec_dim.emplace(vec_dim.begin() + axis, 1);
  }

  TensorDesc td = op.get_output_desc_y();
  td.SetShape(Shape(vec_dim));
  td.SetDataType(op.get_input_desc_x().GetDataType());
  (void) op.update_output_desc_y(td);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Unsqueeze, UnsqueezeInfer);

IMPLEMT_INFERFUNC(Rank, RankInfer) {
    std::vector<int64_t> oShapeVector;
    Shape oShape(oShapeVector);
    TensorDesc td = op.GetOutputDesc("y");
    td.SetShape(ge::Shape(oShape));
    td.SetDataType(DT_INT32);
    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Rank, RankInfer);

IMPLEMT_INFERFUNC(Size, SizeInfer) {
    TensorDesc td = op.GetOutputDesc("y");
    td.SetShape(ge::Shape());
    uint32_t out_type = DT_INT32;
    (void)op.GetAttr("dtype", out_type);
    td.SetDataType((DataType)out_type);
    (void)op.UpdateOutputDesc("y", td);
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
    } else {
      int64_t size = static_cast<int64_t>(input_dims.size());
      std::vector<int64_t> size_v{size};
      td->SetShape(ge::GeShape(size_v));
      td->SetOriginShape(ge::GeShape(size_v));
    }
    uint32_t out_type = DT_INT32;
    (void)op.GetAttr("dtype", out_type);
    td->SetDataType((DataType)out_type);
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
        } else {
          int64_t size = static_cast<int64_t>(input_dims.size());
          OP_LOGD(op.GetName().c_str(), "output value %ld", size);
          std::vector<int64_t> size_v{size};
          td->SetShape(ge::GeShape(size_v));
          td->SetOriginShape(ge::GeShape(size_v));
        }
        uint32_t out_type = DT_INT32;
        (void)op.GetAttr("dtype", out_type);
        td->SetDataType((DataType)out_type);
    }
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ShapeN, ShapeNInfer);

IMPLEMT_INFERFUNC(IdentityN, IdentityNInfer) {
    for (size_t i = 0; i < op.GetInputsSize(); i++) {
        TensorDesc input_desc = op.GetDynamicInputDesc("x", i);
        (void)op.UpdateDynamicOutputDesc("y", i, input_desc);
    }
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(IdentityN, IdentityNInfer);

IMPLEMT_INFERFUNC(Identity, IdentityInfer) {
    TensorDesc input_desc = op.GetInputDesc("x");
    (void)op.UpdateOutputDesc("y", input_desc);
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Identity, IdentityInfer);

IMPLEMT_INFERFUNC(ReadVariableOp, ReadVariableOpInfer) {
    TensorDesc input_desc = op.GetInputDesc("x");
    (void)op.UpdateOutputDesc("y", input_desc);
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ReadVariableOp, ReadVariableOpInfer);

template<typename T>
static void CaclDims(const Tensor& data, std::vector<int64_t>& vec_dim) {
    int32_t size = data.GetSize() / sizeof(T);
    for (int32_t i = 0; i < size; i++) {
        T dim = *((T *)data.GetData() + i);
        if (dim != 0) {
          vec_dim.push_back(dim);
        } else {
          vec_dim.clear();
          break;
        }
    }
}

IMPLEMT_INFERFUNC(Empty, EmptyInfer) {
    TensorDesc td = op.GetOutputDesc("y");
    Tensor data;
    graphStatus ret = op.GetInputConstData("shape", data);
    if (ret == GRAPH_SUCCESS) {
        DataType data_type = data.GetTensorDesc().GetDataType();
        std::vector<int64_t> vec_dim;
        if (data_type == DT_INT32) {
            CaclDims<int32_t>(data, vec_dim);
        } else {
            GeInfershapeErrReport(op.GetName(), op.GetOpType(), "dtype", "Empty only support shape type 'DT_INT32'");
            OP_LOGE(op.GetName().c_str(), "Empty only support shape type 'DT_INT32'");
            return GRAPH_PARAM_INVALID;
        }
        td.SetShape(Shape(vec_dim));
    }

    uint32_t dtype = DT_INT32;
    (void)op.GetAttr("dtype", dtype);
    td.SetDataType((DataType)dtype);
    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Empty, EmptyInfer);

IMPLEMT_INFERFUNC(LowerBound, LowerBoundInfer) {
  auto sorted_x_tensor = op.get_input_desc_sorted_x();
  auto values_tensor = op.get_input_desc_values();
  Shape unused_shape;
  if (WithRank(sorted_x_tensor, 2, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    ShapeErrReport(0,
                   op.GetName(),
                   DebugString(sorted_x_tensor.GetShape().GetDims()),
                   "2D");
    OP_LOGE(op.GetName().c_str(), "Input sorted_inputs rank must be 2.");
    return GRAPH_FAILED;
  }

  if (WithRank(values_tensor, 2, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    ShapeErrReport(1,
                  op.GetName(),
                  DebugString(values_tensor.GetShape().GetDims()),
                  "2D");
    OP_LOGE(op.GetName().c_str(), "Input values rank must be 2.");
    return GRAPH_FAILED;
  }

  DataType out_type;
  if (op.GetAttr("out_type", out_type) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get Attr out_type error.");
    return GRAPH_FAILED;
  }
  TensorDesc y_tensor = op.GetOutputDesc("y");
  y_tensor.SetDataType(out_type);
  y_tensor.SetShape(values_tensor.GetShape());
  return op.UpdateOutputDesc("y", y_tensor);
}

INFER_FUNC_REG(LowerBound, LowerBoundInfer);

IMPLEMT_INFERFUNC(Where, WhereInfer) {
  OpDescPtr op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr x_desc = op_desc->MutableInputDesc(0);

  GeShape x_shape;
  if (WithRankAtLeast(x_desc, 1, x_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input x must be at least 1D.");
    return GRAPH_FAILED;
  }

  if (WithRankAtMost(x_desc, 5, x_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input x must be at most 5D.");
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
    OP_LOGE(op.GetName().c_str(), "input data must be at least 1D.");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input method rank must be 0.");
    return GRAPH_FAILED;
  }

  int64_t fingerprint_size;
  Tensor method_tensor;
  int status = op.GetInputConstData("method", method_tensor);
  if (status != GRAPH_SUCCESS) {
    fingerprint_size = UNKNOWN_DIM;
  } else {
    int64_t method_dim;
    method_dim = method_tensor.GetTensorDesc().GetShape().GetDimNum();
    if (method_dim != 0) {
      OP_LOGE(op.GetName().c_str(), "input method_tensor rank must be 0, real value is %ld.", method_dim);
      return GRAPH_FAILED;
    }
    std::string method_string;
    const char *method_data = reinterpret_cast<const char*>(method_tensor.GetData() + sizeof(uint64_t));

    method_string = method_data;
    if (method_string != "farmhash64") {
      OP_LOGE(op.GetName().c_str(), "Unsupported method, real value is %s", method_string.c_str());
      return GRAPH_FAILED;
    }
    fingerprint_size = sizeof(uint64_t);
  }

  int64_t batch = op.GetInputDesc(0).GetShape().GetDim(0);
  Shape shape({batch, fingerprint_size});
  TensorDesc desc = op.GetOutputDesc("y");
  desc.SetShape(shape);
  desc.SetDataType(DT_UINT8);
  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "fail to update output y.");
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
    OP_LOGE(op.GetName().c_str(), "Failed to get attribute value.");
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
  auto hypothesis_indices_desc = op.GetInputDesc(0);
  auto hypothesis_values_desc = op.GetInputDesc(1);
  auto hypothesis_shape_desc = op.GetInputDesc(2);

  if (ValidateSparseTensor(hypothesis_indices_desc, hypothesis_values_desc, hypothesis_shape_desc, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input hypothesis is not sparse tensor");
    return GRAPH_FAILED;
  }

  auto truth_indices_desc = op.GetInputDesc(3);
  auto truth_values_desc = op.GetInputDesc(4);
  auto truth_shape_desc = op.GetInputDesc(5);

  if (ValidateSparseTensor(truth_indices_desc, truth_values_desc, truth_shape_desc, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input truth is not sparse tensor");
    return GRAPH_FAILED;
  }

  auto output_desc = op.GetOutputDesc("output");

  Tensor hypothesis_shape_tensor, truth_shape_tensor;
  if (op.GetInputConstData("hypothesis_shape", hypothesis_shape_tensor) != GRAPH_SUCCESS) {
    OP_LOGW(op.GetName().c_str(), "failed to get tensor from input hypothesis shape, return unknown shape");
    output_desc.SetShape(ge::Shape(std::vector<int64_t>{UNKNOWN_SHAPE}));
    op.UpdateOutputDesc("output", output_desc);
    return GRAPH_SUCCESS;
  }

  if (op.GetInputConstData("truth_shape", truth_shape_tensor) != GRAPH_SUCCESS) {
    OP_LOGW(op.GetName().c_str(), "failed to get tensor from input truth shape, return unknown shape");
    output_desc.SetShape(ge::Shape(std::vector<int64_t>{UNKNOWN_SHAPE}));
    op.UpdateOutputDesc("output", output_desc);
    return GRAPH_SUCCESS;
  }

  auto hypothesis_shape_num_elements = hypothesis_shape_desc.GetShape().GetShapeSize();
  auto truth_shape_num_elements = truth_shape_desc.GetShape().GetShapeSize();
  if (hypothesis_shape_num_elements != truth_shape_num_elements) {
    OP_LOGE(op.GetName().c_str(), "Num elements of hypothesis_shape does not match truth_shape: %lld vs %lld",
        hypothesis_shape_num_elements, truth_shape_num_elements);
    return GRAPH_PARAM_INVALID;
  }

  int64_t * hypothesis_shape_data = reinterpret_cast<int64_t*>(hypothesis_shape_tensor.GetData());
  if (hypothesis_shape_data == nullptr) {
    OP_LOGE(op.GetName().c_str(), "hypothesis shape data is invalid");
    return GRAPH_PARAM_INVALID;
  }

  int64_t * truth_shape_data = reinterpret_cast<int64_t*>(truth_shape_tensor.GetData());
  if (truth_shape_data == nullptr) {
    OP_LOGE(op.GetName().c_str(), "truth shape data is invalid");
    return GRAPH_PARAM_INVALID;
  }

  std::vector<int64_t> output_dims(hypothesis_shape_num_elements - 1);
  for (uint64_t i = 0; i < output_dims.size(); ++i) {
    output_dims[i] = std::max(hypothesis_shape_data[i], truth_shape_data[i]);
  }

  output_desc.SetShape(Shape(output_dims));
  if (op.UpdateOutputDesc("output", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "failed to update output output desc");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(EditDistance, EditDistanceInfer);

} // namespace ge

