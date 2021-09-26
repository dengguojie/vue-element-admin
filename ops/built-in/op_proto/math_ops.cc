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
 * \file math_ops.cpp
 * \brief
 */
#include "inc/math_ops.h"
#include "graph/operator.h"
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "util/util.h"
#include "util/error_util.h"
namespace ge {

graphStatus SparseSegmentReductionShapeFn(Operator& op) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeShape x_shape;
  auto x_desc = op_desc->MutableInputDesc(0);
  if (WithRankAtLeast(x_desc, 1, x_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Input x should be at least 1-D.");
    return GRAPH_FAILED;
  }

  GeShape indices_shape;
  auto indices_desc = op_desc->MutableInputDesc(1);
  if (WithRank(indices_desc, 1, indices_shape, op.GetName().c_str())
      != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Input indices must be 1-D.");
    return GRAPH_FAILED;
  }

  GeShape segment_ids_shape;
  auto segment_ids_desc = op_desc->MutableInputDesc(2);
  if (WithRank(segment_ids_desc, 1, segment_ids_shape, op.GetName().c_str())
      != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Input segment_ids must be 1-D.");
    return GRAPH_FAILED;
  }

  GeShape unused;
  if (Merge(indices_shape, segment_ids_shape, unused, op.GetName().c_str()) !=
      GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  GeShape subshape;
  if (SubShape(x_shape, 1, x_shape.GetDimNum(), 1, subshape,
               op.GetName().c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  GeShape out;
  GeShape unknown_dim_shape({ge::UNKNOWN_DIM});
  if (Concatenate(unknown_dim_shape, subshape, out) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  auto y_desc = op_desc->MutableOutputDesc(0);
  y_desc->SetDataType(x_desc->GetDataType());
  y_desc->SetShape(out);

  return GRAPH_SUCCESS;
}

// ---------Power-------------------
IMPLEMT_VERIFIER(Power, PowerVerify) {
    OP_LOGI(op.GetName().c_str(), "Enter PowerVerify.");
    return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(Power, PowerVerify);

IMPLEMT_COMMON_INFERFUNC(PowerInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter PowerInferShape");
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Power, PowerInferShape);
// ---------Power End---------------

IMPLEMT_INFERFUNC(Igamma, IgammaInfer) {
  DataType a_type = op.GetInputDesc("a").GetDataType();
  TensorDesc z_desc = op.GetOutputDesc("z");
  z_desc.SetDataType(a_type);
  if (op.UpdateOutputDesc("z", z_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        std::string("update output[z] desc failed"));
    return GRAPH_FAILED;
  }

  return BROADCAST_INFER("a", "x", "z")(op);
}

INFER_FUNC_REG(Igamma, IgammaInfer);

IMPLEMT_INFERFUNC(Igammac, IgammacInfer) {
  DataType a_type = op.GetInputDesc("a").GetDataType();
  TensorDesc z_desc = op.GetOutputDesc("z");
  z_desc.SetDataType(a_type);
  if (op.UpdateOutputDesc("z", z_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        std::string("update output[z] desc failed"));
    return GRAPH_FAILED;
  }

  return BROADCAST_INFER("a", "x", "z")(op);
}

INFER_FUNC_REG(Igammac, IgammacInfer);

IMPLEMT_INFERFUNC(CompareAndBitpack, CompareAndBitpackInfer) {
  Shape input;
  if (WithRankAtLeast(op.GetInputDesc(0), 1, input, op.GetName().c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), ConcatString("call WithRankAtLeast failed, ",
        GetShapeErrMsg(0, DebugString(op.GetInputDesc(0).GetShape().GetDims()),
            "at least 1D")));
    return GRAPH_FAILED;
  }

  Shape unused;
  if (WithRank(op.GetInputDesc(1), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), ConcatString("call WithRankAtLeast failed, ",
        GetShapeErrMsg(1, DebugString(op.GetInputDesc(1).GetShape().GetDims()),
            "scalar")));
    return GRAPH_FAILED;
  }

  Shape output = input;
  std::vector<int64_t> dims = output.GetDims();
  if ((!dims.empty()) && (dims != UNKNOWN_SHAPE)) {
    size_t len = output.GetDimNum();
    int64_t last_dim = output.GetDim(len - 1);
    int64_t inferred_dim = 0;
    if (Divide(last_dim, int64_t(8), true, inferred_dim, op.GetName().c_str()) != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), ConcatString("call Divide function failed, ",
          "the last dim[", last_dim, "] of 0th input must be the multiple of 8"));
      return GRAPH_FAILED;
    }
    output.SetDim(len - 1, inferred_dim);
  }

  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(output);
  output_desc.SetDataType(DT_UINT8);
  return op.UpdateOutputDesc("y", output_desc);
}

INFER_FUNC_REG(CompareAndBitpack, CompareAndBitpackInfer);

IMPLEMT_INFERFUNC(Bincount, BincountInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends({"size"});

  GeShape unused;
  auto size_desc = op_desc->MutableInputDesc(1);
  if (WithRank(size_desc, 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(
        1, DebugString(size_desc->GetShape().GetDims()), "scalar");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  Tensor tensor;
  int64_t bins = 0;
  if (op.GetInputConstData("size", tensor) != GRAPH_SUCCESS) {
    bins = UNKNOWN_DIM;
  }

  if (bins != UNKNOWN_DIM) {
    if (MakeDimForScalarInput(tensor, bins, op.GetName().c_str()) !=
        GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(
          op.GetName(), string("fail to get dim from tensor of input[size]."));
      return GRAPH_FAILED;
    }
  }

  Shape bins_shape;
  if (Vector(bins, bins_shape) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("fail to gen vector shape according dim bins."));
    return GRAPH_FAILED;
  }

  auto bins_desc = op_desc->MutableOutputDesc(0);
  bins_desc->SetShape(GeShape(bins_shape.GetDims()));
  bins_desc->SetDataType(op_desc->MutableInputDesc(2)->GetDataType());

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Bincount, BincountInfer);

IMPLEMT_INFERFUNC(Betainc, BetaincInfer) {
  const int num_inputs = 3;
  Shape output(UNKNOWN_RANK);
  int num_scalars = 0;
  Shape some_non_scalar;
  for (int i = 0; i < num_inputs; ++i) {
    TensorDesc input_desc = op.GetInputDesc(i);
    Shape input_shape = input_desc.GetShape();
    if (!RankKnown(input_shape)) {
      some_non_scalar = input_shape;
    } else if (input_shape.GetDimNum() == 0) {
      ++num_scalars;
    } else {
      if (Merge(output, input_shape, output, op.GetName().c_str()) !=
          GRAPH_SUCCESS) {
        std::string err_msg =
            ConcatString("failed to call Merge function to merge", i,
                         "th input shape", DebugString(input_shape.GetDims()),
                         " and output[z] shape", DebugString(output.GetDims()));
        AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
        return GRAPH_FAILED;
      }
      some_non_scalar = output;
    }
  }

  if (num_scalars == num_inputs - 1) {
    output = some_non_scalar;
  } else if (num_scalars == num_inputs) {
    TensorDesc a_desc = op.GetInputDesc("a");
    output = a_desc.GetShape();
  }
  DataType a_type = op.GetInputDesc("a").GetDataType();
  TensorDesc z_desc = op.GetOutputDesc("z");
  z_desc.SetShape(output);
  z_desc.SetDataType(a_type);
  if (op.UpdateOutputDesc("z", z_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("fail to update output[z] desc."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Betainc, BetaincInfer);

IMPLEMT_INFERFUNC(Zeta, ZetaInfer) {
  TensorDesc out_desc = op.GetOutputDesc("z");
  DataType x_type = op.GetInputDesc("x").GetDataType();
  out_desc.SetDataType(x_type);
  if (GRAPH_SUCCESS != op.UpdateOutputDesc("z", out_desc)) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        std::string("failed to update output[z] desc."));
    return GRAPH_FAILED;
  }
  auto lambdaFunc = BROADCAST_INFER("x", "q", "z");
  return lambdaFunc(op);
}

INFER_FUNC_REG(Zeta, ZetaInfer);

IMPLEMT_INFERFUNC(Bucketize, BetaincInfer) {
  TensorDesc out_desc = op.GetOutputDesc("y");
  out_desc.SetDataType(DT_INT32);
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), string("update output[y] desc failed"));
    return GRAPH_FAILED;
  }
  return UnchangedShape(op, "x", "y");
}

INFER_FUNC_REG(Bucketize, BetaincInfer);

IMPLEMT_INFERFUNC(SparseSegmentSum, SparseSegmentSumInfer) {
  return SparseSegmentReductionShapeFn(op);
}

INFER_FUNC_REG(SparseSegmentSum, SparseSegmentSumInfer);

IMPLEMT_INFERFUNC(SparseSegmentMean, SparseSegmentMeanInfer) {
  return SparseSegmentReductionShapeFn(op);
}

INFER_FUNC_REG(SparseSegmentMean, SparseSegmentMeanInfer);

IMPLEMT_INFERFUNC(SparseSegmentMeanGrad, SparseSegmentMeanGradInfer) {
  const size_t INPUT_NUM = 4;
  std::string err_msg;
  if (INPUT_NUM != op.GetInputsSize()) {
    err_msg =
        ConcatString("input size should be 4, got[", op.GetInputsSize(), "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  const size_t OUTPUT_NUM = 1;
  if (OUTPUT_NUM != op.GetOutputsSize()) {
    err_msg =
        ConcatString("output size should be 4, got[", op.GetOutputsSize(), "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  std::vector<std::string> input_infer_depends = {"output_dim0"};
  op_desc->SetOpInferDepends(input_infer_depends);

  auto x_desc = op_desc->MutableInputDesc(0);
  GeShape x_ge_shape;
  if (WithRankAtLeast(x_desc, 1, x_ge_shape, op.GetName().c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(0, DebugString(x_desc->GetShape().GetDims()),
                             "at least 1D");
    err_msg = string("failed to call WithRankAtLeast function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  auto indices_desc = op_desc->MutableInputDesc(1);
  GeShape indices_shape;
  if (WithRank(indices_desc, 1, indices_shape, op.GetName().c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(1, DebugString(indices_desc->GetShape().GetDims()),
                             "1D");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  GeShape unused;
  GeShape segment_ids_shape(op_desc->MutableInputDesc(2)->GetShape());
  if (Merge(segment_ids_shape, indices_shape, unused, op.GetName().c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call Merge function to merge input[segment_ids]'s shape",
        DebugString(op_desc->MutableInputDesc(2)->GetShape().GetDims()),
        " and input[indices]'s shape", DebugString(indices_shape.GetDims()));
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  auto unused_desc = op_desc->MutableInputDesc(3);
  if (WithRank(unused_desc, 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(3, DebugString(unused_desc->GetShape().GetDims()),
                             "scalar");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  auto x_shape_dims = x_ge_shape.GetDims();
  Shape x_shape(x_shape_dims);
  Shape subshape;
  if (SubShape(x_shape, 1, x_shape.GetDimNum(), 1, subshape,
               op.GetName().c_str()) != GRAPH_SUCCESS) {
    err_msg =
        ConcatString("failed to call SubShape function to get subshape from ",
                     x_shape.GetDimNum(), " to 1 in input[x] shape",
                     DebugString(x_shape.GetDims()));
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  Tensor dims0_tensor;
  Shape dim0_shape;
  const int32_t* dims0_data;
  if (op.GetInputConstData("output_dim0", dims0_tensor) == GRAPH_SUCCESS) {
    const uint8_t* dims0 = dims0_tensor.GetData();
    dims0_data = reinterpret_cast<const int32_t*>(dims0);
  } else {
    dims0_data = reinterpret_cast<const int32_t*>(&UNKNOWN_DIM);
  }

  dim0_shape = Shape({*dims0_data});

  Shape out;
  if (Concatenate(dim0_shape, subshape, out) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  auto out_dims = out.GetDims();
  GeShape ge_out(out_dims);
  auto out_desc = op_desc->MutableOutputDesc(0);
  out_desc->SetDataType(x_desc->GetDataType());
  out_desc->SetShape(ge_out);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseSegmentMeanGrad, SparseSegmentMeanGradInfer);

IMPLEMT_INFERFUNC(IgammaGradA, IgammaGradAInfer) {
  DataType a_type = op.GetInputDesc("a").GetDataType();
  TensorDesc out_desc = op.GetOutputDesc("z");
  out_desc.SetDataType(a_type);
  if (op.UpdateOutputDesc("z", out_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        std::string("update output[z] desc failed"));
    return GRAPH_FAILED;
  }
  auto lambdaFunc = BROADCAST_INFER("a", "x", "z");
  return lambdaFunc(op);
}

INFER_FUNC_REG(IgammaGradA, IgammaGradAInfer);

IMPLEMT_INFERFUNC(InitData, InitDataInfer) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(InitData, InitDataInfer);

IMPLEMT_INFERFUNC(GetNext, GetNextInfer) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(GetNext, GetNextInfer);

IMPLEMT_INFERFUNC(GetDynamicDims, GetDynamicDimsInfer) {
  // Check inputs size
  Operator::OpInt n_attr;
  if (op.GetAttr("N", n_attr) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get attr N failed");
    return GRAPH_FAILED;
  }
  size_t inputs_size = op.GetInputsSize();
  if (static_cast<int64_t>(inputs_size) != n_attr) {
    OP_LOGE(op.GetName().c_str(), "Inputs size [%zu] must equal attr N [%ld]",
            inputs_size, n_attr);
    return GRAPH_FAILED;
  }

  // Set Output as Vector(unknow_dims_num) of { DT_INT32, DT_INT64 }
  Operator::OpListInt shape_info;
  if (op.GetAttr("shape_info", shape_info) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get attr shape_info failed");
    return GRAPH_FAILED;
  }
  int64_t unknow_dims_num = std::count(shape_info.begin(),
                                       shape_info.end(), -1);
  if (unknow_dims_num == 0) {
    OP_LOGE(op.GetName().c_str(),
            "No need to perform GetDynamicDims in a known shape");
    return GRAPH_FAILED;
  }

  Shape vector_shape;
  if (Vector(unknow_dims_num, vector_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Create output shape failed");
    return GRAPH_FAILED;
  }
  auto dims_desc = op.GetOutputDesc("dims");
  dims_desc.SetShape(vector_shape);
  dims_desc.SetDataType(op.GetInputDesc(0).GetDataType());
  if (op.UpdateOutputDesc("dims", dims_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Update dims desc failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(GetDynamicDims, GetDynamicDimsInfer);

// ----------------Erf-------------------
IMPLEMT_COMMON_INFERFUNC(ErfInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ErfInferShape");
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(Erf, ErfInferShape);
// --------------Erf END------------------

// ----------------Erfc-------------------
IMPLEMT_COMMON_INFERFUNC(ErfcInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ErfcInferShape");
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(Erfc, ErfcInferShape);
// --------------Erfc END-----------------

// Obtains the value of the constant tensor.
static void GetConstValue(const Tensor& const_tensor, const DataType& dtype, std::vector<int64_t>& const_data) {
  const uint8_t* constData = const_tensor.GetData();
  size_t size;
  if (dtype == ge::DT_INT32) {
    size = const_tensor.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back(*((int32_t*)constData + i));
    }
  }
}

IMPLEMT_COMMON_INFERFUNC(HistogramFixedWidthInferShape) {
  std::string dtype_attr;
  if (op.GetAttr("dtype", dtype_attr) == GRAPH_SUCCESS) {
    if (dtype_attr != "int32") {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
          ConcatString("invalid value[", dtype_attr, "], only support int32"));
      return GRAPH_FAILED;
    }
  }
  Tensor nbins_tensor;
  if (ge::GRAPH_SUCCESS != op.GetInputConstData("nbins", nbins_tensor)) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        std::string("get const data from input[nbins] failed"));
    return GRAPH_FAILED;
  }
  DataType dtype = op.GetInputDesc("nbins").GetDataType();
  std::vector<int64_t> nbins;
  GetConstValue(nbins_tensor, dtype, nbins);
  std::vector<int64_t> dim_vector;
  if (nbins.empty()) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        std::string("nbins is empty"));
    return GRAPH_FAILED;
  }
  dim_vector.push_back(nbins[0]);
  Shape output_shape(dim_vector);
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(output_shape);
  td.SetDataType(DT_INT32);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HistogramFixedWidth, HistogramFixedWidthVerify) {
  DataType x_dtype = op.GetInputDesc(0).GetDataType();
  DataType range_dtype = op.GetInputDesc(1).GetDataType();
  if (x_dtype != range_dtype) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        ConcatString("data type of input[x] and input[range] should be same. ",
            DTypeStr(x_dtype), " and ", DTypeStr(range_dtype)));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(HistogramFixedWidth, HistogramFixedWidthInferShape);
VERIFY_FUNC_REG(HistogramFixedWidth, HistogramFixedWidthVerify);

// ----------------HistogramFixedWidthD Op Begin-------------------
IMPLEMT_VERIFIER(HistogramFixedWidthD, HistogramFixedWidthDVerify) {
  if (!CheckTwoInputDtypeSame(op, "x", "range")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(HistogramFixedWidthDInferShape) {
  std::string dtype_attr;
  if (op.GetAttr("dtype", dtype_attr) == GRAPH_SUCCESS) {
    if (dtype_attr != "int32") {
      string expected_dtype_list = ConcatString("int32");
      std::string err_msg = GetInputDtypeNotSupportErrMsg("dtype", expected_dtype_list, dtype_attr);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }
  int64_t nbins;
  if (ge::GRAPH_SUCCESS != op.GetAttr("nbins", nbins)) {
    std::string err_msg = GetInputInvalidErrMsg("nbins");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (nbins <= 0) {
    std::string err_msg = GetAttrValueErrMsg("dims_x", std::to_string(nbins), ConcatString(nbins,">",0));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<int64_t> temp_nbins;
  temp_nbins.push_back(nbins);
  Shape output_shape(temp_nbins);
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(output_shape);
  td.SetDataType(DT_INT32);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(HistogramFixedWidthD, HistogramFixedWidthDInferShape);
VERIFY_FUNC_REG(HistogramFixedWidthD, HistogramFixedWidthDVerify);
// ----------------HistogramFixedWidthD Op End-------------------

IMPLEMT_INFERFUNC(NextAfter, NextAfterInfer) {
  Shape x_shape = op.GetInputDesc("x1").GetShape();
  Shape y_shape = op.GetInputDesc("x2").GetShape();
  TensorDesc out_desc = op.GetOutputDesc("output");
  DataType x_type = op.GetInputDesc("x1").GetDataType();
  DataType y_type = op.GetInputDesc("x2").GetDataType();
  if (x_type != y_type) {
    OP_LOGE(op.GetName().c_str(), "the type of x1 is different from that of x2!");
    return GRAPH_FAILED;
  }

  out_desc.SetDataType(x_type);
  if ((!RankKnown(x_shape)) || (!RankKnown(y_shape))) {
    Shape out_shape(UNKNOWN_SHAPE);
    out_desc.SetShape(out_shape);
    if (op.UpdateOutputDesc("output", out_desc) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "update output failed");
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }

  const size_t rank_x = x_shape.GetDimNum();
  const size_t rank_y = y_shape.GetDimNum();
  const size_t rank_out = std::max(rank_x, rank_y);

  // To compute the broadcast dimensions, zip together x_shape and y_shape
  // and pad with 1 to make them the same length.
  std::vector<int64_t> dims;
  int64_t dim_one = 1;
  if (rank_x != rank_y) {
    OP_LOGI(op.GetName().c_str(), "x1 shape is not equal to x2 shape!");
    dim_one = 1;
  }
  for (size_t i = 0; i < rank_out; ++i) {
    int64_t dim_x;
    if (i < (rank_out - rank_x)) {
      dim_x = dim_one;
    } else {
      // rank_out = rank_x or i >= rank_y - rank_x.
      for (size_t j = 0; j < x_shape.GetDimNum(); ++j) {
        if (x_shape.GetDim(j) == UNKNOWN_DIM) {
          dim_x = UNKNOWN_DIM;
          break;
        }
      }
      if ((i - (rank_out - rank_x)) < 0) {
        dim_x = x_shape.GetDim(rank_x + i - (rank_out - rank_x));
      } else {
        dim_x = x_shape.GetDim(i - (rank_out - rank_x));
      }
    }

    const bool dim_y_is_one = (i < (rank_out - rank_y));
    int64_t dim_y;
    if (dim_y_is_one) {
      dim_y = dim_one;
    } else {
      // rank_out = rank_y or i >= rank_x - rank_y.
      for (size_t j = 0; j < y_shape.GetDimNum(); ++j) {
        if (y_shape.GetDim(j) == UNKNOWN_DIM) {
          dim_y = UNKNOWN_DIM;
          break;
        }
      }
      if ((i - (rank_out - rank_y)) < 0) {
        dim_y = y_shape.GetDim(rank_y + i - (rank_out - rank_y));
      } else {
        dim_y = y_shape.GetDim(i - (rank_out - rank_y));
      }
    }

    if ((dim_x == UNKNOWN_DIM) || (dim_y == UNKNOWN_DIM)) {
      /* One or both dimensions is unknown.
       * If either dimension is greater than 1, assume that the program is
       * correct, and the other dimension will be broadcast to match it.
       * For shape inference, if eliminate the shape checks
       * in this code, assert that the unknown dim is either 1
       * or the same as the known dim.
       * If either dimension is 1, the other dimension is the output.
       */
      if (dim_x > 1) {
        dims.push_back(dim_x);
      } else if (dim_y > 1) {
        dims.push_back(dim_y);
      } else if (dim_x == 1) {
        dims.push_back(dim_y);
      } else if (dim_y == 1) {
        dims.push_back(dim_x);
      } else if (dim_x == dim_y) {
        dims.push_back(dim_x);
      } else {
        dims.push_back(UNKNOWN_DIM);
      }
    } else if ((dim_x == 1) || (dim_y == 1)) {
      // dim_x is dim_one or dim_y is dim_one.
      if ((dim_x == 1) && (!dim_y_is_one)) {
        // broadcast dim_x to dim_y.
        dims.push_back(dim_y);
      } else {
        if (dim_y == 1) {
          // broadcast dim_y to dim_x.
          dims.push_back(dim_x);
        }
      }
    } else {
      int64_t dim;
      if (Merge(dim_x, dim_y, dim) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
      }
      dims.push_back(dim);
    }
  }
  Shape out_shape(dims);
  out_desc.SetShape(out_shape);
  if (op.UpdateOutputDesc("output", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update output failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(NextAfter, NextAfterInfer);

IMPLEMT_INFERFUNC(IsFinite, IsFiniteInfer) {
  TensorDesc out_desc = op.GetOutputDesc("y");
  out_desc.SetDataType(DT_BOOL);
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update y failed");
    return GRAPH_FAILED;
  }
  return UnchangedShape(op, "x", "y");
}

INFER_FUNC_REG(IsFinite, IsFiniteInfer);

IMPLEMT_INFERFUNC(IsInf, IsInfInfer)
{
    TensorDesc out_desc = op.GetOutputDesc("y");
    out_desc.SetDataType(DT_BOOL);
    if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
        AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
                                           string("update output[y] failed."));
        return GRAPH_FAILED;
    }
    return UnchangedShape(op, "x", "y");
}

INFER_FUNC_REG(IsInf, IsInfInfer);

IMPLEMT_INFERFUNC(ComplexAbs, ComplexAbsInfer)
{
    TensorDesc out_desc = op.GetOutputDesc("y");
    DataType Tout;
    if (op.GetAttr("Tout", Tout) != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
                                         string("get attr[Tout] failed"));
    }
    out_desc.SetDataType(Tout);
    (void)op.UpdateOutputDesc("y", out_desc);
    return UnchangedShape(op, "x", "y");
}

INFER_FUNC_REG(ComplexAbs, ComplexAbsInfer);

IMPLEMT_INFERFUNC(IsNan, IsNanInfer) {
  TensorDesc out_desc = op.GetOutputDesc("y");
  out_desc.SetDataType(DT_BOOL);
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
                                       string("update output[y] failed."));
    return GRAPH_FAILED;
  }
  return UnchangedShape(op, "x", "y");
}

INFER_FUNC_REG(IsNan, IsNanInfer);

IMPLEMT_INFERFUNC(Real, RealInfer) {
  TensorDesc out_desc = op.GetOutputDesc("output");
  DataType Tout;
  if (op.GetAttr("Tout", Tout) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
      op.GetName(), std::string("fail to get attr[Tout]"));
    return GRAPH_FAILED;
  }
  out_desc.SetDataType(Tout);
  if (op.UpdateOutputDesc("output", out_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), std::string("update output[output] desc failed"));
    return GRAPH_FAILED;
  }
  return UnchangedShape(op, "input", "output");
}

INFER_FUNC_REG(Real, RealInfer);

IMPLEMT_INFERFUNC(Conj, ConjInfer) {
  TensorDesc desc = op.GetInputDesc("input");
  return op.UpdateOutputDesc("output", desc);
}

INFER_FUNC_REG(Conj, ConjInfer);

// ----------------------NLLLoss------------------------
IMPLEMT_COMMON_INFERFUNC(NLLLossInferShape) {
  std::string reduction;
  if (op.GetAttr("reduction", reduction) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get attr reduction failed.");
    return GRAPH_FAILED;
  }
  
  std::vector<int64_t> scalar;
  std::vector<int64_t> y_shape;
  std::vector<std::pair<int64_t, int64_t>> y_range;

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto x_desc = op_desc->MutableInputDesc(0);
  auto weight_desc = op_desc->MutableInputDesc(2);
  auto y_desc = op_desc->MutableOutputDesc(0);
  auto total_weight_desc = op_desc->MutableOutputDesc(1);
  std::vector<int64_t> x_shape = x_desc->MutableShape().GetDims();

  y_desc->SetDataType(x_desc->GetDataType());
  total_weight_desc->SetShape(GeShape(scalar));
  total_weight_desc->SetDataType(weight_desc->GetDataType());
  
  if (reduction == "none") {
    if (IsUnknownRankShape(x_shape) || x_shape.size() == 2) {
      y_shape.push_back(x_shape[0]);
      y_desc->SetShape(GeShape(y_shape));
    
      if (IsUnKnownShape(x_shape)) {
        std::vector<std::pair<int64_t, int64_t>> x_range;
        x_desc->GetShapeRange(x_range);
        MakeUpShapeRange(x_shape, x_range);
        y_range.push_back(x_range[0]);
        y_desc->SetShapeRange(y_range);
      }

      return GRAPH_SUCCESS;
    }
  }
  
  // reduction "mean" or "sum" or 1D, output y is scalar
  y_desc->SetShape(GeShape(scalar));
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(NLLLoss, NLLLossInferShape);
// --------------------NllLoss END----------------------

// ----------------------NLLLossGrad------------------------
IMPLEMT_COMMON_INFERFUNC(NLLLossGradInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter NLLLossGradInferShape");
  if (OneInOneOutDynamicInfer(op, "x", {"x_grad"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(NLLLossGrad, NLLLossGradInferShape);
// --------------------NLLLossGrad END----------------------

// --------------------Pdist----------------------
IMPLEMT_COMMON_INFERFUNC(PdistInferShape) {
    TensorDesc output_desc = op.GetOutputDesc("y");
    DataType predict_dtype = op.GetInputDesc("x").GetDataType();
    Format predict_format = op.GetInputDesc("x").GetFormat();
    ge::Shape inputshape = op.GetInputDesc("x").GetShape();
    if (inputshape.GetDims().size() != 2) {
        OP_LOGE("The shape of input must be 2.");
        return GRAPH_FAILED;
    }
    int64_t dim_shape = inputshape.GetDim(0);
    int64_t outputshape = 0.5 * ((dim_shape) * (dim_shape-1));
    std::vector<int64_t> yShape({outputshape});
    output_desc.SetDataType(predict_dtype);
    output_desc.SetFormat(predict_format);
    output_desc.SetShape(Shape(yShape));
    (void)op.UpdateOutputDesc("y", output_desc);
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Pdist, PdistVerify) {
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Pdist, PdistInferShape);
VERIFY_FUNC_REG(Pdist, PdistVerify);
// --------------------Pdist END----------------------

// ----------------LpNorm Begin-------------------
IMPLEMT_VERIFIER(LpNorm, LpNormVerify) { return GRAPH_SUCCESS; }
IMPLEMT_COMMON_INFERFUNC(LpNormInfer) {
  auto tensor_input = op.GetInputDesc("x");
  Shape x_shape = tensor_input.GetShape();
  DataType x_type = tensor_input.GetDataType();
  Format x_format = tensor_input.GetFormat();
  size_t dim_num = op.GetInputDesc("x").GetShape().GetDimNum();
  std::vector<int64_t> x_axes = {};
  std::vector<int64_t> new_axes = {};
  std::vector<int64_t> y_vec = {};
  std::vector<int64_t> x_dim_members = x_shape.GetDims();
  bool keep_dim = false;
  int32_t indice;
  (void)op.GetAttr("keepdim", keep_dim);
  if (op.GetAttr("axes", x_axes) != GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "axes will use default value");
  }
  if (x_axes.empty()) {
    for (size_t i = 0; i < dim_num; i++) {
      new_axes.push_back(static_cast<int64_t>(i));
    }
  } else {
    for (size_t i = 0; i < x_axes.size(); i++) {
      indice = (x_axes[i] < 0) ? (x_axes[i] + dim_num) : x_axes[i];
      new_axes.push_back(indice);
    }
  }
  for (size_t i = 0; i < x_shape.GetDimNum(); i++) {
    if (find(new_axes.begin(), new_axes.end(), i) != new_axes.end()) {
      if (keep_dim == true) {
        y_vec.push_back(1);
      }
    } else {
      y_vec.push_back(x_dim_members[i]);
    }
  }
  ge::Shape output_shape(y_vec);
  // update output desc
  ge::TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(output_shape);
  if (x_axes.empty()) {
    std::vector<std::pair<int64_t, int64_t>> o_range;
    output_desc.SetShapeRange(o_range);
  }
  output_desc.SetDataType(x_type);
  output_desc.SetFormat(x_format);
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(LpNorm, LpNormInfer);
VERIFY_FUNC_REG(LpNorm, LpNormVerify);
// ----------------LpNorm END---------------------

// ----------------LpNormReduce Begin-------------------
IMPLEMT_COMMON_INFERFUNC(LpNormReduceInfer) {
  Shape x_shape = op.GetInputDescByName("x").GetShape();
  DataType x_type = op.GetInputDescByName("x").GetDataType();
  Format x_format = op.GetInputDescByName("x").GetFormat();
  size_t dim_num = x_shape.GetDimNum();
  std::vector<int64_t> x_axes = {};
  std::vector<int64_t> new_axes = {};
  std::vector<int64_t> y_vec = {};
  std::vector<int64_t> x_dim_members = x_shape.GetDims();
  bool keep_dim = false;
  int32_t indice;
  (void)op.GetAttr("keepdim", keep_dim);
  if (op.GetAttr("axes", x_axes) != GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "axes will use default value");
  }
  if (x_axes.empty()) {
    for (size_t i = 0; i < dim_num; i++) {
      new_axes.push_back(static_cast<int64_t>(i));
    }
  } else {
    for (size_t i = 0; i < x_axes.size(); i++) {
      indice = (x_axes[i] < 0) ? (x_axes[i] + dim_num) : x_axes[i];
      new_axes.push_back(indice);
    }
  }
  for (size_t i = 0; i < x_shape.GetDimNum(); i++) {
    if (find(new_axes.begin(), new_axes.end(), i) != new_axes.end()) {
      if (keep_dim == true) {
        y_vec.push_back(1);
      }
    } else {
      y_vec.push_back(x_dim_members[i]);
    }
  }
  ge::Shape output_shape(y_vec);
  // update output desc
  ge::TensorDesc output_desc = op.GetOutputDescByName("y");
  output_desc.SetShape(output_shape);
  if (x_axes.empty()) {
    std::vector<std::pair<int64_t, int64_t>> o_range;
    output_desc.SetShapeRange(o_range);
  }
  output_desc.SetDataType(x_type);
  output_desc.SetFormat(x_format);
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(LpNormReduce, LpNormReduceInfer);
// ----------------LpNormReduce END---------------------

// ----------------LpNormUpdate Begin-------------------
IMPLEMT_COMMON_INFERFUNC(LpNormUpdateInfer) {
  Shape x_shape =  op.GetInputDescByName("x").GetShape();
  DataType x_type =  op.GetInputDescByName("x").GetDataType();
  // update output desc
  TensorDesc output_desc = op.GetOutputDescByName("y");
  output_desc.SetShape(ge::Shape(x_shape));
  output_desc.SetDataType(x_type);
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(LpNormUpdate, LpNormUpdateInfer);
// ----------------LpNormUpdate END---------------------

// ----------------Trunc---------------------
IMPLEMT_COMMON_INFERFUNC(TruncInferShape) {
    TensorDesc output_desc = op.GetOutputDesc("output_y");
    DataType predict_dtype = op.GetInputDesc("input_x").GetDataType();
    Format predict_format = op.GetInputDesc("input_x").GetFormat();
    ge::Shape output_shape = op.GetInputDesc("input_x").GetShape();

    output_desc.SetDataType(predict_dtype);
    output_desc.SetFormat(predict_format);
    output_desc.SetShape(output_shape);
    (void)op.UpdateOutputDesc("output_y", output_desc);
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Trunc,TruncVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Trunc, TruncInferShape);
VERIFY_FUNC_REG(Trunc, TruncVerify);
// ----------------Trunc END---------------------

IMPLEMT_INFERFUNC(Complex, ComplexInfer)
{
  TensorDesc out_desc = op.GetOutputDesc("out");
  Shape x_shape = op.GetInputDesc("real").GetShape();
  Shape y_shape = op.GetInputDesc("imag").GetShape();
  DataType x_type = op.GetInputDesc("real").GetDataType();
  DataType y_type = op.GetInputDesc("imag").GetDataType();
  if (x_type != y_type) {
    OP_LOGE(op.GetName().c_str(), "The type of x1 [%d] is different from that of x2 [%d]!", x_type, y_type);
    return GRAPH_FAILED;
  }
  DataType Tout;
  if (op.GetAttr("Tout", Tout) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "Get attr Tout error.");
      return GRAPH_FAILED;
    }

  out_desc.SetDataType(Tout);
  if (op.UpdateOutputDesc("out", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Update out failed");
    return GRAPH_FAILED;
  }

  if ((!RankKnown(x_shape)) || (!RankKnown(y_shape))) {
    Shape out_shape(UNKNOWN_RANK);
    out_desc.SetShape(out_shape);
    if (op.UpdateOutputDesc("out", out_desc) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "Update output failed");
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }

  const size_t rank_x = x_shape.GetDimNum();
  const size_t rank_y = y_shape.GetDimNum();
  const size_t rank_out = std::max(rank_x, rank_y);

  // To compute the broadcast dimensions, zip together x_shape and y_shape
  // and pad with 1 to make them the same length.
  std::vector<int64_t> dims;
  int64_t dim_one = 0;
  if (rank_x != rank_y) {
    OP_LOGI(op.GetName().c_str(), "X1 shape dims [%lld] is not equal to x2 shape dims [%lld]!", rank_x, rank_y);
    dim_one = 1;
  }
  for (size_t i = 0; i < rank_out; ++i) {
    int64_t dim_x = 0;
    if (i < (rank_out - rank_x)) {
      dim_x = dim_one;
    } else {
      // rank_out = rank_x or i >= rank_y - rank_x.
      for (size_t j = 0; j < x_shape.GetDimNum(); ++j) {
        if (x_shape.GetDim(j) == UNKNOWN_DIM) {
          dim_x = UNKNOWN_DIM;
          break;
        }
      }
      if ((i - (rank_out - rank_x)) < 0) {
        dim_x = x_shape.GetDim(rank_x + i - (rank_out - rank_x));
      } else {
        dim_x = x_shape.GetDim(i - (rank_out - rank_x));
      }
    }

    const bool dim_y_is_one = (i < (rank_out - rank_y));
    int64_t dim_y = 0;
    if (dim_y_is_one) {
      dim_y = dim_one;
    } else {
      // rank_out = rank_y or i >= rank_x - rank_y.
      for (size_t j = 0; j < y_shape.GetDimNum(); ++j) {
        if (y_shape.GetDim(j) == UNKNOWN_DIM) {
          dim_y = UNKNOWN_DIM;
          break;
        }
      }
      if ((i - (rank_out - rank_y)) < 0) {
        dim_y = y_shape.GetDim(rank_y + i - (rank_out - rank_y));
      } else {
        dim_y = y_shape.GetDim(i - (rank_out - rank_y));
      }
    }

    if ((dim_x == UNKNOWN_DIM) || (dim_y == UNKNOWN_DIM)) {
      /* One or both dimensions is unknown.
       * If either dimension is greater than 1, assume that the program is
       * correct, and the other dimension will be broadcast to match it.
       * For shape inference, if eliminate the shape checks
       * in this code, assert that the unknown dim is either 1
       * or the same as the known dim.
       * If either dimension is 1, the other dimension is the output.
       */
      if (dim_x > 1) {
        dims.push_back(dim_x);
      } else if (dim_y > 1) {
        dims.push_back(dim_y);
      } else if (dim_x == 1) {
        dims.push_back(dim_y);
      } else if (dim_y == 1) {
        dims.push_back(dim_x);
      } else if (dim_x == dim_y) {
        dims.push_back(dim_x);
      } else {
        dims.push_back(UNKNOWN_DIM);
      }
    } else if ((dim_x == 1) || (dim_y == 1)) {
      // dim_x is dim_one or dim_y is dim_one.
      if ((dim_x == 1) && (!dim_y_is_one)) {
        // broadcast dim_x to dim_y.
        dims.push_back(dim_y);
      } else {
        if (dim_y == 1) {
          // broadcast dim_y to dim_x.
          dims.push_back(dim_x);
        }
      }
    } else {
      int64_t dim = 0;
      if (Merge(dim_x, dim_y, dim) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
      }
      dims.push_back(dim);
    }
  }
  Shape out_shape(dims);
  out_desc.SetShape(out_shape);
  if (op.UpdateOutputDesc("out", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Update output failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Complex, ComplexInfer);


IMPLEMT_INFERFUNC(Imag, ImagInfer)
{
    TensorDesc out_desc = op.GetOutputDesc("output");
    DataType Tout;
    if (op.GetAttr("Tout", Tout) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Get attr Tout error.");
        return GRAPH_FAILED;
    }
    out_desc.SetDataType(Tout);
    if (op.UpdateOutputDesc("output", out_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Update output failed");
        return GRAPH_FAILED;
    }
    return UnchangedShape(op, "input", "output");
}

INFER_FUNC_REG(Imag, ImagInfer);


IMPLEMT_INFERFUNC(Angle, AngleInfer)
{
    TensorDesc out_desc = op.GetOutputDesc("output");
    DataType Tout;
    if (op.GetAttr("Tout", Tout) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Get attr Tout error.");
        return  GRAPH_FAILED;
    }
    out_desc.SetDataType(Tout);
    if (op.UpdateOutputDesc("output", out_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Update output failed");
        return GRAPH_FAILED;
    }
    return UnchangedShape(op, "input", "output");
}

INFER_FUNC_REG(Angle, AngleInfer);

// ----------------SoftMarginLossGrad Begin------------------------
bool infer_shape_and_type_soft_margin_loss_grad(Operator& op,
                                                const string& input_name1,
                                                const string& input_name2,
                                                const string& input_name3,
                                                const string& output_name) {
    TensorDesc v_output_desc = op.GetOutputDesc(output_name);

    DataType input_dtype = op.GetInputDesc(input_name1).GetDataType();
    Format input_format = op.GetInputDesc(input_name1).GetFormat();

    ge::Shape shape_x = op.GetInputDesc(input_name1).GetShape();
    ge::Shape shape_y = op.GetInputDesc(input_name2).GetShape();
    ge::Shape shape_z = op.GetInputDesc(input_name3).GetShape();
    std::vector<int64_t> dims_x = shape_x.GetDims();
    std::vector<int64_t> dims_y = shape_y.GetDims();
    std::vector<int64_t> dims_z = shape_z.GetDims();

    if (dims_y.size() > dims_z.size() && dims_y.size() > dims_x.size()) {
        std::vector<int64_t> dims_tmp = dims_y;
        dims_y = dims_x;
        dims_x = dims_tmp;
    }else if (dims_z.size() > dims_x.size() && dims_z.size() > dims_y.size()) {
        std::vector<int64_t> dims_tmp = dims_z;
        dims_z = dims_x;
        dims_x = dims_tmp;
    }

    if (dims_x.size() != dims_y.size()) {
        int dec = dims_x.size() - dims_y.size();
        for (int i = 0; i < dec; i++) {
            dims_y.insert(dims_y.begin(), (int64_t)1);
        }
    }else if (dims_x.size() != dims_z.size()) {
        int dec = dims_x.size() - dims_z.size();
        for (int i = 0; i < dec; i++) {
            dims_z.insert(dims_z.begin(), (int64_t)1);
        }
    }

    std::vector<int64_t> dim_vec;
    for (size_t i = 0; i < dims_x.size(); i++) {
        if ((dims_x[i] != dims_y[i]) && (dims_x[i] != dims_z[i])
        && (dims_x[i] != 1) && (dims_y[i] != 1) && (dims_z[i] != 1)) {
            OP_LOGE(op.GetName().c_str(), "three input can broatcast \n");
            return false;
        }
        int64_t dims = dims_x[i];
        if((dims_x[i] > dims_y[i]) && dims_x[i] > dims_z[i]){
            dims = dims_x[i];
        }else if((dims_y[i] > dims_x[i]) && dims_y[i] > dims_z[i]){
            dims = dims_y[i];
        }else if((dims_z[i] > dims_x[i]) && dims_z[i] > dims_y[i]){
            dims = dims_z[i];
        }
        dim_vec.push_back(dims);
    }
    ge::Shape output_shape = ge::Shape(dim_vec);

    v_output_desc.SetShape(output_shape);
    v_output_desc.SetDataType(input_dtype);
    v_output_desc.SetFormat(input_format);
    op.UpdateOutputDesc(output_name, v_output_desc);

    return true;
}

IMPLEMT_VERIFIER(SoftMarginLossGrad, SoftMarginLossGradVerify)
{
    if (op.GetInputDesc("predict").GetDataType() != op.GetInputDesc("label").GetDataType() ||
        op.GetInputDesc("predict").GetDataType() != op.GetInputDesc("dout").GetDataType()) {
        OP_LOGE(op.GetName().c_str(), "three input datatype must equal!\n");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(SoftMarginLossGradInferShape)
{
    if(infer_shape_and_type_soft_margin_loss_grad(op, "dout", "predict", "label", "gradient")) {
        return GRAPH_SUCCESS;
    }else{
        return GRAPH_FAILED;
    }
}

COMMON_INFER_FUNC_REG(SoftMarginLossGrad, SoftMarginLossGradInferShape);
VERIFY_FUNC_REG(SoftMarginLossGrad, SoftMarginLossGradVerify);
//---------------SoftMarginLossGrad-------------------

// ----------------Cross begin---------------------------------
bool InferShapeAndTypeCross(Operator& op, const string& input_name1,
                            const string& input_name2, const string& output_name) {
  TensorDesc v_output_desc = op.GetOutputDesc(output_name);
  DataType input_dtype = op.GetInputDesc(input_name1).GetDataType();
  Format input_format = op.GetInputDesc(input_name1).GetFormat();

  ge::Shape shape_x = op.GetInputDesc(input_name1).GetShape();
  ge::Shape shape_y = op.GetInputDesc(input_name2).GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();
  std::vector<int64_t> dims_y = shape_y.GetDims();

  std::vector<int64_t> dim_vec;
  for (size_t i = 0; i < dims_x.size(); i++) {
    if ((dims_x[i] != dims_y[i]) && (dims_x[i] != 1) && (dims_y[i] != 1)) {
        OP_LOGE("Operators need to be broadcast");
        return false;
      }

    int64_t dims = dims_x[i] > dims_y[i] ? dims_x[i] : dims_y[i];
    dim_vec.push_back(dims);
  }
  ge::Shape output_shape = ge::Shape(dim_vec);

  v_output_desc.SetShape(output_shape);
  v_output_desc.SetDataType(input_dtype);
  v_output_desc.SetFormat(input_format);
  op.UpdateOutputDesc(output_name, v_output_desc);

  return true;
}

IMPLEMT_COMMON_INFERFUNC(CrossInferShape)
{
  if(InferShapeAndTypeCross(op, "x1", "x2", "y")) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

IMPLEMT_VERIFIER(Cross, CrossVerify)
{
  if (op.GetInputDesc("x1").GetDataType() != op.GetInputDesc("x2").GetDataType()) {
    OP_LOGE(op.GetName().c_str(), "the two inputs datatype not equal!\n");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Cross, CrossInferShape);
VERIFY_FUNC_REG(Cross, CrossVerify);
// ----------------Cross End------------------------

// ----------------Cdist Begin------------------------
bool infer_shape_cdist(Operator& op,
                     const string& input_name1,
                     const string& input_name2,
                     const string& output_name) {

    TensorDesc output_desc = op.GetOutputDesc(output_name);

    DataType input1_dtype = op.GetInputDesc(input_name1).GetDataType();
    Format input_format = op.GetInputDesc(input_name1).GetFormat();

    ge::Shape shape_x = op.GetInputDesc(input_name1).GetShape();
    ge::Shape shape_y = op.GetInputDesc(input_name2).GetShape();
    std::vector<int64_t> dims_x = shape_x.GetDims();
    std::vector<int64_t> dims_y = shape_y.GetDims();

    if (dims_x.size() != dims_y.size()) {
        OP_LOGE(op.GetName().c_str(), "the two inputs shape not equal!\n");
        return false;
    }
    if ((dims_x.size() < 2) || (dims_y.size() < 2)) {
        OP_LOGE(op.GetName().c_str(), "the two inputs shape size all less than 2!\n");
        return false;
    }

    for(size_t index = 0; index < dims_x.size(); index++) {
        if (dims_x[index] != dims_y[index]) {
            OP_LOGE(op.GetName().c_str(), "the two inputs value not equal!\n");
            return false;
        }
    }

    dims_x.pop_back();
    ge::Shape output_shape = ge::Shape(dims_x);
    output_desc.SetShape(output_shape);
    output_desc.SetDataType(input1_dtype);
    output_desc.SetFormat(input_format);
    op.UpdateOutputDesc(output_name, output_desc);

    return true;
}

IMPLEMT_VERIFIER(Cdist, CdistVerify)
{
    if (op.GetInputDesc("x1").GetDataType() != op.GetInputDesc("x2").GetDataType()) {
        OP_LOGE(op.GetName().c_str(), "the two inputs datatype not equal!\n");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(CdistInferShape)
{
    if (infer_shape_cdist(op, "x1", "x2", "y")) {
        return GRAPH_SUCCESS;
    }
    return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(Cdist, CdistInferShape);
VERIFY_FUNC_REG(Cdist, CdistVerify);
// ----------------Cdist ----------------------------

// ----------------CdistGrad Begin------------------------
IMPLEMT_COMMON_INFERFUNC(CdistGradInferShape) {
    TensorDesc output_desc = op.GetOutputDesc("y");

    DataType input_dtype = op.GetInputDesc("x1").GetDataType();
    Format input_format = op.GetInputDesc("x1").GetFormat();

    ge::Shape grad_shape = op.GetInputDesc("grad").GetShape();
    ge::Shape cdist_shape = op.GetInputDesc("cdist").GetShape();
    ge::Shape input1_shape = op.GetInputDesc("x1").GetShape();
    ge::Shape input2_shape = op.GetInputDesc("x2").GetShape();

    std::vector < int64_t > grad_dim = grad_shape.GetDims();
    std::vector < int64_t > cdist_dim = cdist_shape.GetDims();
    std::vector < int64_t > input1_dim = input1_shape.GetDims();
    std::vector < int64_t > input2_dim = input2_shape.GetDims();

    auto dim_size = input1_dim.size();
    if ((dim_size != 3) && (dim_size != 4)) {
        OP_LOGE(op.GetName().c_str(), "the first inputs datasize not equal 3 and 4!\n");
        return GRAPH_FAILED;
    }
    if ((input2_dim.size() != dim_size) || (grad_dim.size() != dim_size) || (cdist_dim.size() != dim_size)) {
        OP_LOGE(op.GetName().c_str(), "the one of other three inputs datasize not equal the first one \n");
        return GRAPH_FAILED;
    }

    for (size_t i = 0; i < dim_size; i++) {
        auto dim = input1_dim[i];
        if ((input2_dim[i] != dim) || (grad_dim[i] != dim) || (cdist_dim[i] != dim)) {
            OP_LOGE(op.GetName().c_str(), "the one of other three inputs not equal the first one \n");
            return GRAPH_FAILED;
        }
    }
    input1_dim.erase(input1_dim.end() - 2);

    ge::Shape output_shape = ge::Shape(input1_dim);
    output_desc.SetShape(output_shape);
    output_desc.SetDataType(input_dtype);
    output_desc.SetFormat(input_format);
    op.UpdateOutputDesc("y", output_desc);

    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(CdistGrad, CdistGradVerify) {
    DataType x1_dtype = op.GetInputDesc("x1").GetDataType();
    DataType x2_dtype = op.GetInputDesc("x2").GetDataType();
    DataType grad_dtype = op.GetInputDesc("grad").GetDataType();
    DataType cdist_dtype = op.GetInputDesc("cdist").GetDataType();

    if ((x2_dtype != x1_dtype) || (grad_dtype != x1_dtype) || (cdist_dtype != x1_dtype)) {
        OP_LOGE(op.GetName().c_str(), "the four input datatype must not be the same \n");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(CdistGrad, CdistGradInferShape);
VERIFY_FUNC_REG(CdistGrad, CdistGradVerify);
// ----------------CdistGrad END---------------------------------

}  // namespace ge
