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
#include "error_util.h"
#include "data_preprocess.h"
#include "util/vector_proto_profiling.h"
#include <utility>
#include "util/images_ops_shape_fns.h"
#include "op_const.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "axis_util.h"
#include "inc/graph/utils/type_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {

graphStatus SparseSegmentReductionShapeFn(Operator& op) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeShape x_shape;
  auto x_desc = op_desc->MutableInputDesc(0);
  if (WithRankAtLeast(x_desc, 1, x_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Input x should be at least 1-D.");
    return GRAPH_FAILED;
  }

  GeShape indices_shape;
  auto indices_desc = op_desc->MutableInputDesc(1);
  if (WithRank(indices_desc, 1, indices_shape, TbeGetName(op).c_str())
      != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Input indices must be 1-D.");
    return GRAPH_FAILED;
  }

  GeShape segment_ids_shape;
  auto segment_ids_desc = op_desc->MutableInputDesc(2);
  if (WithRank(segment_ids_desc, 1, segment_ids_shape, TbeGetName(op).c_str())
      != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Input segment_ids must be 1-D.");
    return GRAPH_FAILED;
  }

  GeShape unused;
  if (Merge(indices_shape, segment_ids_shape, unused, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  GeShape subshape;
  if (SubShape(x_shape, 1, x_shape.GetDimNum(), 1, subshape,
               TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
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
    OP_LOGI(TbeGetName(op).c_str(), "Enter PowerVerify.");
    return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(Power, PowerVerify);

IMPLEMT_COMMON_INFERFUNC(PowerInferShape) {
  OP_LOGI(TbeGetName(op).c_str(), "Enter PowerInferShape");
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Power, PowerInferShape);
// ---------Power End---------------

IMPLEMT_INFERFUNC(Igamma, IgammaInfer) {
  DataType a_type = op.GetInputDescByName("a").GetDataType();
  TensorDesc z_desc = op.GetOutputDescByName("z");
  z_desc.SetDataType(a_type);
  if (op.UpdateOutputDesc("z", z_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("update output[z] desc failed"));
    return GRAPH_FAILED;
  }

  return BROADCAST_INFER("a", "x", "z")(op);
}

INFER_FUNC_REG(Igamma, IgammaInfer);

IMPLEMT_INFERFUNC(Igammac, IgammacInfer) {
  DataType a_type = op.GetInputDescByName("a").GetDataType();
  TensorDesc z_desc = op.GetOutputDescByName("z");
  z_desc.SetDataType(a_type);
  if (op.UpdateOutputDesc("z", z_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("update output[z] desc failed"));
    return GRAPH_FAILED;
  }

  return BROADCAST_INFER("a", "x", "z")(op);
}

INFER_FUNC_REG(Igammac, IgammacInfer);

IMPLEMT_INFERFUNC(CompareAndBitpack, CompareAndBitpackInfer) {
  Shape input;
  if (WithRankAtLeast(op.GetInputDesc(0), 1, input, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), ConcatString("call WithRankAtLeast failed, ",
        GetShapeErrMsg(0, DebugString(op.GetInputDesc(0).GetShape().GetDims()),
            "at least 1D")));
    return GRAPH_FAILED;
  }

  Shape unused;
  if (WithRank(op.GetInputDesc(1), 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), ConcatString("call WithRankAtLeast failed, ",
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
    if (Divide(last_dim, int64_t(8), true, inferred_dim, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), ConcatString("call Divide function failed, ",
          "the last dim[", last_dim, "] of 0th input must be the multiple of 8"));
      return GRAPH_FAILED;
    }
    output.SetDim(len - 1, inferred_dim);
  }

  TensorDesc output_desc = op.GetOutputDescByName("y");
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
  if (WithRank(size_desc, 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(
        1, DebugString(size_desc->GetShape().GetDims()), "scalar");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  Tensor tensor;
  int64_t bins = 0;
  if (op.GetInputConstData("size", tensor) != GRAPH_SUCCESS) {
    bins = UNKNOWN_DIM;
  }

  if (bins != UNKNOWN_DIM) {
    if (MakeDimForScalarInput(tensor, bins, TbeGetName(op).c_str()) !=
        GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(
          TbeGetName(op), string("fail to get dim from tensor of input[size]."));
      return GRAPH_FAILED;
    }
  }

  Shape bins_shape;
  if (Vector(bins, bins_shape) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("fail to gen vector shape according dim bins."));
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
      if (Merge(output, input_shape, output, TbeGetName(op).c_str()) !=
          GRAPH_SUCCESS) {
        std::string err_msg =
            ConcatString("failed to call Merge function to merge", i,
                         "th input shape", DebugString(input_shape.GetDims()),
                         " and output[z] shape", DebugString(output.GetDims()));
        AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
        return GRAPH_FAILED;
      }
      some_non_scalar = output;
    }
  }

  if (num_scalars == num_inputs - 1) {
    output = some_non_scalar;
  } else if (num_scalars == num_inputs) {
    TensorDesc a_desc = op.GetInputDescByName("a");
    output = a_desc.GetShape();
  }
  DataType a_type = op.GetInputDescByName("a").GetDataType();
  TensorDesc z_desc = op.GetOutputDescByName("z");
  z_desc.SetShape(output);
  z_desc.SetDataType(a_type);
  if (op.UpdateOutputDesc("z", z_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("fail to update output[z] desc."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Betainc, BetaincInfer);

IMPLEMT_INFERFUNC(Zeta, ZetaInfer) {
  TensorDesc out_desc = op.GetOutputDescByName("z");
  DataType x_type = op.GetInputDescByName("x").GetDataType();
  out_desc.SetDataType(x_type);
  if (GRAPH_SUCCESS != op.UpdateOutputDesc("z", out_desc)) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("failed to update output[z] desc."));
    return GRAPH_FAILED;
  }
  auto lambdaFunc = BROADCAST_INFER("x", "q", "z");
  return lambdaFunc(op);
}

INFER_FUNC_REG(Zeta, ZetaInfer);

IMPLEMT_INFERFUNC(Bucketize, BetaincInfer) {
  TensorDesc out_desc = op.GetOutputDescByName("y");
  out_desc.SetDataType(DT_INT32);
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("update output[y] desc failed"));
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
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  const size_t OUTPUT_NUM = 1;
  if (OUTPUT_NUM != op.GetOutputsSize()) {
    err_msg =
        ConcatString("output size should be 4, got[", op.GetOutputsSize(), "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  std::vector<std::string> input_infer_depends = {"output_dim0"};
  op_desc->SetOpInferDepends(input_infer_depends);

  auto x_desc = op_desc->MutableInputDesc(0);
  GeShape x_ge_shape;
  if (WithRankAtLeast(x_desc, 1, x_ge_shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(0, DebugString(x_desc->GetShape().GetDims()),
                             "at least 1D");
    err_msg = string("failed to call WithRankAtLeast function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto indices_desc = op_desc->MutableInputDesc(1);
  GeShape indices_shape;
  if (WithRank(indices_desc, 1, indices_shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(1, DebugString(indices_desc->GetShape().GetDims()),
                             "1D");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  GeShape unused;
  GeShape segment_ids_shape(op_desc->MutableInputDesc(2)->GetShape());
  if (Merge(segment_ids_shape, indices_shape, unused, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call Merge function to merge input[segment_ids]'s shape",
        DebugString(op_desc->MutableInputDesc(2)->GetShape().GetDims()),
        " and input[indices]'s shape", DebugString(indices_shape.GetDims()));
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  auto unused_desc = op_desc->MutableInputDesc(3);
  if (WithRank(unused_desc, 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(3, DebugString(unused_desc->GetShape().GetDims()),
                             "scalar");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto x_shape_dims = x_ge_shape.GetDims();
  Shape x_shape(x_shape_dims);
  Shape subshape;
  if (SubShape(x_shape, 1, x_shape.GetDimNum(), 1, subshape,
               TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    err_msg =
        ConcatString("failed to call SubShape function to get subshape from ",
                     x_shape.GetDimNum(), " to 1 in input[x] shape",
                     DebugString(x_shape.GetDims()));
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
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
  DataType a_type = op.GetInputDescByName("a").GetDataType();
  TensorDesc out_desc = op.GetOutputDescByName("z");
  out_desc.SetDataType(a_type);
  if (op.UpdateOutputDesc("z", out_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
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
  return DataPreprocGetNextCommonInfer(op);
}

INFER_FUNC_REG(GetNext, GetNextInfer);

IMPLEMT_INFERFUNC(GetDynamicDims, GetDynamicDimsInfer) {
  // Check inputs size
  Operator::OpInt n_attr;
  if (op.GetAttr("N", n_attr) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get attr N failed");
    return GRAPH_FAILED;
  }
  size_t inputs_size = op.GetInputsSize();
  if (static_cast<int64_t>(inputs_size) != n_attr) {
    OP_LOGE(TbeGetName(op).c_str(), "Inputs size [%zu] must equal attr N [%ld]",
            inputs_size, n_attr);
    return GRAPH_FAILED;
  }

  // Set Output as Vector(unknow_dims_num) of { DT_INT32, DT_INT64 }
  Operator::OpListInt shape_info;
  if (op.GetAttr("shape_info", shape_info) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get attr shape_info failed");
    return GRAPH_FAILED;
  }
  int64_t unknow_dims_num = std::count(shape_info.begin(),
                                       shape_info.end(), -1);
  if (unknow_dims_num == 0) {
    OP_LOGE(TbeGetName(op).c_str(),
            "No need to perform GetDynamicDims in a known shape");
    return GRAPH_FAILED;
  }

  Shape vector_shape;
  if (Vector(unknow_dims_num, vector_shape) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Create output shape failed");
    return GRAPH_FAILED;
  }
  auto dims_desc = op.GetOutputDescByName("dims");
  dims_desc.SetShape(vector_shape);
  dims_desc.SetDataType(op.GetInputDesc(0).GetDataType());
  if (op.UpdateOutputDesc("dims", dims_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Update dims desc failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(GetDynamicDims, GetDynamicDimsInfer);

// ----------------Erf-------------------
IMPLEMT_COMMON_INFERFUNC(ErfInferShape) {
  OP_LOGI(TbeGetName(op).c_str(), "Enter ErfInferShape");
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(Erf, ErfInferShape);
// --------------Erf END------------------

// ----------------Erfc-------------------
IMPLEMT_COMMON_INFERFUNC(ErfcInferShape) {
  OP_LOGI(TbeGetName(op).c_str(), "Enter ErfcInferShape");
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
  int dtype_attr;
  if (op.GetAttr("dtype", dtype_attr) == GRAPH_SUCCESS) {
    if (dtype_attr != 3) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
          ConcatString("invalid value[", dtype_attr, "], only support int32"));
      return GRAPH_FAILED;
    }
  }
  Tensor nbins_tensor;
  if (ge::GRAPH_SUCCESS != op.GetInputConstData("nbins", nbins_tensor)) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("get const data from input[nbins] failed"));
    return GRAPH_FAILED;
  }
  DataType dtype = op.GetInputDescByName("nbins").GetDataType();
  std::vector<int64_t> nbins;
  GetConstValue(nbins_tensor, dtype, nbins);
  std::vector<int64_t> dim_vector;
  if (nbins.empty()) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("nbins is empty"));
    return GRAPH_FAILED;
  }
  dim_vector.push_back(nbins[0]);
  Shape output_shape(dim_vector);
  TensorDesc td = op.GetOutputDescByName("y");
  td.SetShape(output_shape);
  td.SetDataType(DT_INT32);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HistogramFixedWidth, HistogramFixedWidthVerify) {
  DataType x_dtype = op.GetInputDesc(0).GetDataType();
  DataType range_dtype = op.GetInputDesc(1).GetDataType();
  if (x_dtype != range_dtype) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
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
  int dtype_attr;
  if (op.GetAttr("dtype", dtype_attr) == GRAPH_SUCCESS) {
    if (dtype_attr != 3) {
      OP_LOGE(TbeGetName(op).c_str(), "attr dtype only support int32, but is %d", dtype_attr);
      return GRAPH_FAILED;
    }
  }
  int64_t nbins;
  if (ge::GRAPH_SUCCESS != op.GetAttr("nbins", nbins)) {
    std::string err_msg = GetInputInvalidErrMsg("nbins");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (nbins <= 0) {
    std::string err_msg = GetAttrValueErrMsg("dims_x", std::to_string(nbins), ConcatString(nbins,">",0));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<int64_t> temp_nbins;
  temp_nbins.push_back(nbins);
  Shape output_shape(temp_nbins);
  TensorDesc td = op.GetOutputDescByName("y");
  td.SetShape(output_shape);
  td.SetDataType(DT_INT32);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(HistogramFixedWidthD, HistogramFixedWidthDInferShape);
VERIFY_FUNC_REG(HistogramFixedWidthD, HistogramFixedWidthDVerify);
// ----------------HistogramFixedWidthD Op End-------------------

IMPLEMT_INFERFUNC(NextAfter, NextAfterInfer) {
  Shape x_shape = op.GetInputDescByName("x1").GetShape();
  Shape y_shape = op.GetInputDescByName("x2").GetShape();
  TensorDesc out_desc = op.GetOutputDescByName("output");
  DataType x_type = op.GetInputDescByName("x1").GetDataType();
  DataType y_type = op.GetInputDescByName("x2").GetDataType();
  if (x_type != y_type) {
    OP_LOGE(TbeGetName(op).c_str(), "the type of x1 is different from that of x2!");
    return GRAPH_FAILED;
  }

  out_desc.SetDataType(x_type);
  if ((!RankKnown(x_shape)) || (!RankKnown(y_shape))) {
    Shape out_shape(UNKNOWN_SHAPE);
    out_desc.SetShape(out_shape);
    if (op.UpdateOutputDesc("output", out_desc) != GRAPH_SUCCESS) {
      OP_LOGE(TbeGetName(op).c_str(), "update output failed");
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
    OP_LOGI(TbeGetName(op).c_str(), "x1 shape is not equal to x2 shape!");
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
    OP_LOGE(TbeGetName(op).c_str(), "update output failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(NextAfter, NextAfterInfer);

IMPLEMT_INFERFUNC(IsFinite, IsFiniteInfer) {
  TensorDesc out_desc = op.GetOutputDescByName("y");
  out_desc.SetDataType(DT_BOOL);
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "update y failed");
    return GRAPH_FAILED;
  }
  return UnchangedShape(op, "x", "y");
}

INFER_FUNC_REG(IsFinite, IsFiniteInfer);

IMPLEMT_INFERFUNC(IsInf, IsInfInfer)
{
    TensorDesc out_desc = op.GetOutputDescByName("y");
    out_desc.SetDataType(DT_BOOL);
    if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
        AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                           string("update output[y] failed."));
        return GRAPH_FAILED;
    }
    return UnchangedShape(op, "x", "y");
}

INFER_FUNC_REG(IsInf, IsInfInfer);

IMPLEMT_INFERFUNC(ComplexAbs, ComplexAbsInfer)
{
    TensorDesc out_desc = op.GetOutputDescByName("y");
    DataType Tout;
    if (op.GetAttr("Tout", Tout) != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                         string("get attr[Tout] failed"));
    }
    out_desc.SetDataType(Tout);
    (void)op.UpdateOutputDesc("y", out_desc);
    return UnchangedShape(op, "x", "y");
}

INFER_FUNC_REG(ComplexAbs, ComplexAbsInfer);

IMPLEMT_INFERFUNC(IsNan, IsNanInfer) {
  TensorDesc out_desc = op.GetOutputDescByName("y");
  out_desc.SetDataType(DT_BOOL);
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       string("update output[y] failed."));
    return GRAPH_FAILED;
  }
  return UnchangedShape(op, "x", "y");
}

INFER_FUNC_REG(IsNan, IsNanInfer);

IMPLEMT_INFERFUNC(Real, RealInfer) {
  TensorDesc out_desc = op.GetOutputDescByName("output");
  DataType Tout;
  if (op.GetAttr("Tout", Tout) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
      TbeGetName(op), std::string("fail to get attr[Tout]"));
    return GRAPH_FAILED;
  }
  out_desc.SetDataType(Tout);
  if (op.UpdateOutputDesc("output", out_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), std::string("update output[output] desc failed"));
    return GRAPH_FAILED;
  }
  return UnchangedShape(op, "input", "output");
}

INFER_FUNC_REG(Real, RealInfer);

IMPLEMT_INFERFUNC(Conj, ConjInfer) {
  TensorDesc desc = op.GetInputDescByName("input");
  return op.UpdateOutputDesc("output", desc);
}

INFER_FUNC_REG(Conj, ConjInfer);

// ----------------------NLLLoss------------------------
IMPLEMT_COMMON_INFERFUNC(NLLLossInferShape) {
  std::string reduction;
  if (op.GetAttr("reduction", reduction) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get attr reduction failed.");
    return GRAPH_FAILED;
  }

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto x_desc = op_desc->MutableInputDesc(0);
  auto y_desc = op_desc->MutableOutputDesc(0);
  auto total_weight_desc = op_desc->MutableOutputDesc(1);
  const GeShape& x_shape = x_desc->MutableShape();

  y_desc->SetDataType(x_desc->GetDataType());
  total_weight_desc->SetDataType(x_desc->GetDataType());

  GeShape& y_shape = y_desc->MutableShape();
  GeShape& total_weight_shape = total_weight_desc->MutableShape();

  // set total_weight shape is scalar
  total_weight_shape.SetDimNum(0);
  
  if (reduction == "none") {
    if (x_shape.IsUnknownDimNum()) {
      // x is -2, y is -2
      y_shape.SetIsUnknownDimNum();
      return GRAPH_SUCCESS;
    }
    if (x_shape.GetDimNum() == 2) {
      // x is [x, y], y is [x]
      y_shape.SetDimNum(1);
      y_shape.SetDim(0, x_shape.GetDim(0));
    
      if (x_shape.IsUnknownShape()) {
        // x_shape is dynamic, infer range 
        std::vector<std::pair<int64_t, int64_t>> y_range;
        std::vector<std::pair<int64_t, int64_t>> x_range;
        x_desc->GetShapeRange(x_range);
        std::vector<int64_t> x_shape_vec = x_shape.GetDims();
        MakeUpShapeRange(x_shape_vec, x_range);
        y_range.push_back(x_range[0]);
        y_desc->SetShapeRange(y_range);
      }
      return GRAPH_SUCCESS;
    }
  }
  
  // reduction "mean" or "sum" or 1D, output y is scalar
  y_shape.SetDimNum(0);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(NLLLoss, NLLLossInferShape);
// --------------------NllLoss END----------------------

// ----------------------NLLLossGrad------------------------
IMPLEMT_COMMON_INFERFUNC(NLLLossGradInferShape) {
  OP_LOGI(TbeGetName(op).c_str(), "Enter NLLLossGradInferShape");
  const int64_t input_x_idx = 0;
  const int64_t output_x_grad_idx = 0;
  if (OneInOneOutDynamicInfer(op, input_x_idx, {output_x_grad_idx})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(NLLLossGrad, NLLLossGradInferShape);
// --------------------NLLLossGrad END----------------------

// --------------------Pdist----------------------
IMPLEMT_COMMON_INFERFUNC(PdistInferShape) {
    TensorDesc output_desc = op.GetOutputDescByName("y");
    DataType predict_dtype = op.GetInputDescByName("x").GetDataType();
    Format predict_format = op.GetInputDescByName("x").GetFormat();
    ge::Shape inputshape = op.GetInputDescByName("x").GetShape();
    if (inputshape.GetDims().size() != 2) {
        OP_LOGE(TbeGetName(op).c_str(), "The shape of input must be 2.");
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
  auto tensor_input = op.GetInputDescByName("x");
  Shape x_shape = tensor_input.GetShape();
  DataType x_type = tensor_input.GetDataType();
  Format x_format = tensor_input.GetFormat();
  size_t dim_num = op.GetInputDescByName("x").GetShape().GetDimNum();
  std::vector<int64_t> x_axes = {};
  std::vector<int64_t> new_axes = {};
  std::vector<int64_t> y_vec = {};
  std::vector<int64_t> x_dim_members = x_shape.GetDims();
  bool keep_dim = false;
  int32_t indice;
  (void)op.GetAttr("keepdim", keep_dim);
  if (op.GetAttr("axes", x_axes) != GRAPH_SUCCESS) {
    OP_LOGI(TbeGetName(op).c_str(), "axes will use default value");
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
    OP_LOGI(TbeGetName(op).c_str(), "axes will use default value");
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
    TensorDesc output_desc = op.GetOutputDescByName("output_y");
    DataType predict_dtype = op.GetInputDescByName("input_x").GetDataType();
    Format predict_format = op.GetInputDescByName("input_x").GetFormat();
    ge::Shape output_shape = op.GetInputDescByName("input_x").GetShape();

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
  TensorDesc out_desc = op.GetOutputDescByName("out");
  Shape x_shape = op.GetInputDescByName("real").GetShape();
  Shape y_shape = op.GetInputDescByName("imag").GetShape();
  DataType x_type = op.GetInputDescByName("real").GetDataType();
  DataType y_type = op.GetInputDescByName("imag").GetDataType();
  if (x_type != y_type) {
    OP_LOGE(TbeGetName(op).c_str(), "The type of x1 [%d] is different from that of x2 [%d]!", x_type, y_type);
    return GRAPH_FAILED;
  }
  DataType Tout;
  if (op.GetAttr("Tout", Tout) != GRAPH_SUCCESS) {
      OP_LOGE(TbeGetName(op).c_str(), "Get attr Tout error.");
      return GRAPH_FAILED;
    }

  out_desc.SetDataType(Tout);
  if (op.UpdateOutputDesc("out", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Update out failed");
    return GRAPH_FAILED;
  }

  if ((!RankKnown(x_shape)) || (!RankKnown(y_shape))) {
    Shape out_shape(UNKNOWN_RANK);
    out_desc.SetShape(out_shape);
    if (op.UpdateOutputDesc("out", out_desc) != GRAPH_SUCCESS) {
      OP_LOGE(TbeGetName(op).c_str(), "Update output failed");
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
    OP_LOGI(TbeGetName(op).c_str(), "X1 shape dims [%lld] is not equal to x2 shape dims [%lld]!", rank_x, rank_y);
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
    OP_LOGE(TbeGetName(op).c_str(), "Update output failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Complex, ComplexInfer);


IMPLEMT_INFERFUNC(Imag, ImagInfer)
{
    TensorDesc out_desc = op.GetOutputDescByName("output");
    DataType Tout;
    if (op.GetAttr("Tout", Tout) != GRAPH_SUCCESS) {
        OP_LOGE(TbeGetName(op).c_str(), "Get attr Tout error.");
        return GRAPH_FAILED;
    }
    out_desc.SetDataType(Tout);
    if (op.UpdateOutputDesc("output", out_desc) != GRAPH_SUCCESS) {
        OP_LOGE(TbeGetName(op).c_str(), "Update output failed");
        return GRAPH_FAILED;
    }
    return UnchangedShape(op, "input", "output");
}

INFER_FUNC_REG(Imag, ImagInfer);


IMPLEMT_INFERFUNC(Angle, AngleInfer)
{
    TensorDesc out_desc = op.GetOutputDescByName("output");
    DataType Tout;
    if (op.GetAttr("Tout", Tout) != GRAPH_SUCCESS) {
        OP_LOGE(TbeGetName(op).c_str(), "Get attr Tout error.");
        return  GRAPH_FAILED;
    }
    out_desc.SetDataType(Tout);
    if (op.UpdateOutputDesc("output", out_desc) != GRAPH_SUCCESS) {
        OP_LOGE(TbeGetName(op).c_str(), "Update output failed");
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
    TensorDesc v_output_desc = op.GetOutputDescByName(output_name.c_str());

    DataType input_dtype = op.GetInputDescByName(input_name1.c_str()).GetDataType();
    Format input_format = op.GetInputDescByName(input_name1.c_str()).GetFormat();

    ge::Shape shape_x = op.GetInputDescByName(input_name1.c_str()).GetShape();
    ge::Shape shape_y = op.GetInputDescByName(input_name2.c_str()).GetShape();
    ge::Shape shape_z = op.GetInputDescByName(input_name3.c_str()).GetShape();
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
            OP_LOGE(TbeGetName(op).c_str(), "three input can broatcast \n");
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
    op.UpdateOutputDesc(output_name.c_str(), v_output_desc);

    return true;
}

IMPLEMT_VERIFIER(SoftMarginLossGrad, SoftMarginLossGradVerify)
{
    if (op.GetInputDescByName("predict").GetDataType() != op.GetInputDescByName("label").GetDataType() ||
        op.GetInputDescByName("predict").GetDataType() != op.GetInputDescByName("dout").GetDataType()) {
        OP_LOGE(TbeGetName(op).c_str(), "three input datatype must equal!\n");
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
  TensorDesc v_output_desc = op.GetOutputDescByName(output_name.c_str());
  DataType input_dtype = op.GetInputDescByName(input_name1.c_str()).GetDataType();
  Format input_format = op.GetInputDescByName(input_name1.c_str()).GetFormat();

  ge::Shape shape_x = op.GetInputDescByName(input_name1.c_str()).GetShape();
  ge::Shape shape_y = op.GetInputDescByName(input_name2.c_str()).GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();
  std::vector<int64_t> dims_y = shape_y.GetDims();

  std::vector<int64_t> dim_vec;
  for (size_t i = 0; i < dims_x.size(); i++) {
    if ((dims_x[i] != dims_y[i]) && (dims_x[i] != 1) && (dims_y[i] != 1)) {
        OP_LOGE(TbeGetName(op).c_str(), "Operators need to be broadcast");
        return false;
      }

    int64_t dims = dims_x[i] > dims_y[i] ? dims_x[i] : dims_y[i];
    dim_vec.push_back(dims);
  }
  ge::Shape output_shape = ge::Shape(dim_vec);

  v_output_desc.SetShape(output_shape);
  v_output_desc.SetDataType(input_dtype);
  v_output_desc.SetFormat(input_format);
  op.UpdateOutputDesc(output_name.c_str(), v_output_desc);

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
  if (op.GetInputDescByName("x1").GetDataType() != op.GetInputDescByName("x2").GetDataType()) {
    OP_LOGE(TbeGetName(op).c_str(), "the two inputs datatype not equal!\n");
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

    TensorDesc output_desc = op.GetOutputDescByName(output_name.c_str());

    DataType input1_dtype = op.GetInputDescByName(input_name1.c_str()).GetDataType();
    Format input_format = op.GetInputDescByName(input_name1.c_str()).GetFormat();

    ge::Shape shape_x = op.GetInputDescByName(input_name1.c_str()).GetShape();
    ge::Shape shape_y = op.GetInputDescByName(input_name2.c_str()).GetShape();
    std::vector<int64_t> dims_x = shape_x.GetDims();
    std::vector<int64_t> dims_y = shape_y.GetDims();

    if (dims_x.size() != dims_y.size()) {
        OP_LOGE(TbeGetName(op).c_str(), "the two inputs shape not equal!\n");
        return false;
    }
    if ((dims_x.size() < 2) || (dims_y.size() < 2)) {
        OP_LOGE(TbeGetName(op).c_str(), "the two inputs shape size all less than 2!\n");
        return false;
    }

    for(size_t index = 0; index < dims_x.size(); index++) {
        if (dims_x[index] != dims_y[index]) {
            OP_LOGE(TbeGetName(op).c_str(), "the two inputs value not equal!\n");
            return false;
        }
    }

    dims_x.pop_back();
    ge::Shape output_shape = ge::Shape(dims_x);
    output_desc.SetShape(output_shape);
    output_desc.SetDataType(input1_dtype);
    output_desc.SetFormat(input_format);
    op.UpdateOutputDesc(output_name.c_str(), output_desc);

    return true;
}

IMPLEMT_VERIFIER(Cdist, CdistVerify)
{
    if (op.GetInputDescByName("x1").GetDataType() != op.GetInputDescByName("x2").GetDataType()) {
        OP_LOGE(TbeGetName(op).c_str(), "the two inputs datatype not equal!\n");
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
    TensorDesc output_desc = op.GetOutputDescByName("y");

    DataType input_dtype = op.GetInputDescByName("x1").GetDataType();
    Format input_format = op.GetInputDescByName("x1").GetFormat();

    ge::Shape grad_shape = op.GetInputDescByName("grad").GetShape();
    ge::Shape cdist_shape = op.GetInputDescByName("cdist").GetShape();
    ge::Shape input1_shape = op.GetInputDescByName("x1").GetShape();
    ge::Shape input2_shape = op.GetInputDescByName("x2").GetShape();

    std::vector < int64_t > grad_dim = grad_shape.GetDims();
    std::vector < int64_t > cdist_dim = cdist_shape.GetDims();
    std::vector < int64_t > input1_dim = input1_shape.GetDims();
    std::vector < int64_t > input2_dim = input2_shape.GetDims();

    auto dim_size = input1_dim.size();
    if ((dim_size != 3) && (dim_size != 4)) {
        OP_LOGE(TbeGetName(op).c_str(), "the first inputs datasize not equal 3 and 4!\n");
        return GRAPH_FAILED;
    }
    if ((input2_dim.size() != dim_size) || (grad_dim.size() != dim_size) || (cdist_dim.size() != dim_size)) {
        OP_LOGE(TbeGetName(op).c_str(), "the one of other three inputs datasize not equal the first one \n");
        return GRAPH_FAILED;
    }

    for (size_t i = 0; i < dim_size; i++) {
        auto dim = input1_dim[i];
        if ((input2_dim[i] != dim) || (grad_dim[i] != dim) || (cdist_dim[i] != dim)) {
            OP_LOGE(TbeGetName(op).c_str(), "the one of other three inputs not equal the first one \n");
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
    DataType x1_dtype = op.GetInputDescByName("x1").GetDataType();
    DataType x2_dtype = op.GetInputDescByName("x2").GetDataType();
    DataType grad_dtype = op.GetInputDescByName("grad").GetDataType();
    DataType cdist_dtype = op.GetInputDescByName("cdist").GetDataType();

    if ((x2_dtype != x1_dtype) || (grad_dtype != x1_dtype) || (cdist_dtype != x1_dtype)) {
        OP_LOGE(TbeGetName(op).c_str(), "the four input datatype must not be the same \n");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(CdistGrad, CdistGradInferShape);
VERIFY_FUNC_REG(CdistGrad, CdistGradVerify);
// ----------------CdistGrad END---------------------------------

// ----------------RaggedBincount Begin------------------------
IMPLEMT_COMMON_INFERFUNC(RaggedBincountInferShape) {
    TensorDesc output_desc = op.GetOutputDescByName("output");
    ge::Shape splits_shape = op.GetInputDescByName("splits").GetShape();
    ge::Shape values_shape = op.GetInputDescByName("values").GetShape();
    ge::Shape size_shape = op.GetInputDescByName("size").GetShape();
    DataType weights_dtype = op.GetInputDescByName("weights").GetDataType();
    std::vector<int64_t> splits_dim = splits_shape.GetDims();
    std::vector<int64_t> values_dim = values_shape.GetDims();
    std::vector<int64_t> size_dim = size_shape.GetDims();

    if (splits_dim.size() != 1) {
      OP_LOGE(TbeGetName(op).c_str(), "split dim must be 1\n");
      return GRAPH_FAILED;
    }
    if (values_dim.size() != 2) {
      OP_LOGE(TbeGetName(op).c_str(), "values dim must be 2\n");
      return GRAPH_FAILED;
    }
    if (size_dim.size() != 0) {
      OP_LOGE(TbeGetName(op).c_str(), "size must be scalar\n");
      return GRAPH_FAILED;
    }

    Tensor dims0_tensor;
    const int32_t* dims0_data;
    if (op.GetInputConstData("size", dims0_tensor) == GRAPH_SUCCESS) {
      const uint8_t* dims0 = dims0_tensor.GetData();
      dims0_data = reinterpret_cast<const int32_t*>(dims0);
    } else {
      dims0_data = reinterpret_cast<const int32_t*>(&UNKNOWN_DIM);
    }

    std::vector<int64_t> dims = {splits_dim[0] - 1, *dims0_data};
    output_desc.SetShape(Shape(dims));
    output_desc.SetDataType(weights_dtype);
    op.UpdateOutputDesc("output", output_desc);
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(RaggedBincount, RaggedBincountVerify) {
    DataType weights_dtype = op.GetInputDescByName("weights").GetDataType();
    DataType output_dtype = op.GetOutputDescByName("output").GetDataType();

    if (weights_dtype != output_dtype) {
      OP_LOGE(TbeGetName(op).c_str(), "the weights input datatype must not be same output datatype\n");
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(RaggedBincount, RaggedBincountInferShape);
VERIFY_FUNC_REG(RaggedBincount, RaggedBincountVerify);
// ----------------RaggedBincount END---------------------------------

// ----------------DenseCountSparseOutput Begin--------------------------
IMPLEMT_COMMON_INFERFUNC(DenseCountSparseOutputInferShape) {
    TensorDesc output_indices_desc = op.GetOutputDescByName("output_indices");
    TensorDesc output_values = op.GetOutputDescByName("output_values");
    TensorDesc output_dense_shape_desc = op.GetOutputDescByName("output_dense_shape");

    Format values_format = op.GetInputDescByName("values").GetFormat();
    ge::Shape values_shape = op.GetInputDescByName("values").GetShape();
    ge::Shape weights_shape = op.GetInputDescByName("weights").GetShape();
    DataType values_datatype = op.GetInputDescByName("values").GetDataType();
    DataType weights_datatype = op.GetInputDescByName("weights").GetDataType();

    ge::Shape output;
    if ((RankKnown(weights_shape) == true) && (weights_shape.GetDimNum() == 0)) {
      output = values_shape;
    } else {
      if (Merge(weights_shape, values_shape, output, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
        AICPU_INFER_SHAPE_CALL_ERR_REPORT(
          TbeGetName(op),
          string("failed to call Merge function for shape of weights and values."));
        return GRAPH_FAILED;
      }
    }

    if (values_datatype != DT_INT32 && values_datatype != DT_INT64) {
      OP_LOGE(TbeGetName(op).c_str(), "'values' should have dtype int32 or int64.");
      return GRAPH_FAILED;
    }

    if (weights_datatype != DT_FLOAT && weights_datatype != DT_DOUBLE &&
        weights_datatype != DT_INT32 && weights_datatype != DT_INT64) {
      OP_LOGE(TbeGetName(op).c_str(), "'weights' should have dtype float, double, int32 or int64.");
      return GRAPH_FAILED;
    }

    if ((values_shape.GetDimNum() != 1) && (values_shape.GetDimNum() != 2)) {
      OP_LOGE(TbeGetName(op).c_str(), "'values' must be a 1D or 2D tensor.");
      return GRAPH_FAILED;
    }

    int32_t dim = output.GetDimNum();
    std::vector<int64_t> output_indices_dims = { ge::UNKNOWN_DIM, dim };
    std::vector<int64_t> output_values_dims = { ge::UNKNOWN_DIM };
    std::vector<int64_t> output_dense_shape_dims = { dim };

    output_indices_desc.SetShape(Shape(output_indices_dims));
    output_values.SetShape(Shape(output_values_dims));
    output_dense_shape_desc.SetShape(Shape(output_dense_shape_dims));

    output_indices_desc.SetDataType(DT_INT64);
    output_values.SetDataType(weights_datatype);
    output_dense_shape_desc.SetDataType(DT_INT64);

    output_indices_desc.SetFormat(values_format);
    output_values.SetFormat(values_format);
    output_dense_shape_desc.SetFormat(values_format);

    op.UpdateOutputDesc("output_indices", output_indices_desc);
    op.UpdateOutputDesc("output_values", output_values);
    op.UpdateOutputDesc("output_dense_shape", output_dense_shape_desc);
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(DenseCountSparseOutput, DenseCountSparseOutputInferShape);
// ----------------DenseCountSparseOutput End----------------------------

// ----------------------SparseSegmentSumGrad------------------------
static void GetSparseSegmentSumGradConstValue(const Tensor& const_tensor, const DataType& dtype, int64_t& const_data) {
  if (dtype == ge::DT_INT32) {
    int32_t* const_data_ptr = const_cast<int32_t*>(reinterpret_cast<const int32_t*>(const_tensor.GetData()));
    const_data = static_cast<int32_t>((*(const_data_ptr + 0)));
  } else {
    int64_t* const_data_ptr = const_cast<int64_t*>(reinterpret_cast<const int64_t*>(const_tensor.GetData()));
    const_data = static_cast<int64_t>(*(const_data_ptr + 0));
  }
}

IMPLEMT_COMMON_INFERFUNC(SparseSegmentSumGradInferShape) {
  vector<string> input_infer_depends = {"output_dim0"};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc == nullptr) {
    OP_LOGE(TbeGetName(op).c_str(), "get op_desc failed.");
    return GRAPH_FAILED;
  }
  op_desc->SetOpInferDepends(input_infer_depends);

  Tensor input_output_dim0_tensor;
  int64_t input_output_dim0;
  DataType input_output_dim0_dtype = op_desc->GetInputDescPtr(3)->GetDataType();
  std::vector<std::pair<int64_t, int64_t>> out_range;

  if (GRAPH_SUCCESS != op.GetInputConstData("output_dim0", input_output_dim0_tensor)) {
    input_output_dim0 = -1;
    out_range.push_back(std::pair<int64_t, int64_t>(0, -1));
  } else {
    GetSparseSegmentSumGradConstValue(input_output_dim0_tensor, input_output_dim0_dtype, input_output_dim0);
    out_range.push_back(std::pair<int64_t, int64_t>(input_output_dim0, input_output_dim0));
  }

  ge::GeShape shape = op_desc->GetInputDescPtr(0)->GetShape();
  ge::GeShape shape_id = op_desc->GetInputDescPtr(1)->GetShape();
  ge::GeShape shape_indices = op_desc->GetInputDescPtr(2)->GetShape();

  auto output_desc = op_desc->MutableOutputDesc(0);
  ge::GeShape output_shape = output_desc->MutableShape();

  size_t dim_id_size_input = shape_id.GetDimNum();
  size_t dim_indices_size_input = shape_indices.GetDimNum();
  DataType input_dtype = op_desc->GetInputDescPtr(0)->GetDataType();
  if (dim_id_size_input != 1) {
    OP_LOGE(TbeGetName(op).c_str(), "segment_ids' shape must be 1D.");
    return GRAPH_FAILED;
  }
  if (dim_indices_size_input != 1) {
    OP_LOGE(TbeGetName(op).c_str(), "segment_ids' shape must be 1D.");
    return GRAPH_FAILED;
  }
  if (shape.IsUnknownDimNum()) {
    output_desc->SetShape(shape);
    output_desc->SetDataType(input_dtype);
    return GRAPH_SUCCESS;
  }
  size_t rank = shape.GetDimNum();
  output_shape.SetDimNum(rank);
  output_shape.SetDim(0, input_output_dim0);
  for (size_t i = 1; i < rank; i++) {
    int64_t x_dim = shape.GetDim(i);
    output_shape.SetDim(i, x_dim);
  }
  output_desc->SetShape(output_shape);
  output_desc->SetDataType(input_dtype);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SparseSegmentSumGrad, SparseSegmentSumGradInferShape);
// --------------------SparseSegmentSumGrad END----------------------

// ----------------RaggedCountSparseOutput Begin--------------------------
IMPLEMT_COMMON_INFERFUNC(RaggedCountSparseOutputInferShape) {
    TensorDesc output_indices_desc = op.GetOutputDescByName("output_indices");
    TensorDesc output_values = op.GetOutputDescByName("output_values");
    TensorDesc output_dense_shape_desc = op.GetOutputDescByName("output_dense_shape");

    Format values_format = op.GetInputDescByName("values").GetFormat();
    ge::Shape splits_shape = op.GetInputDescByName("splits").GetShape();
    ge::Shape values_shape = op.GetInputDescByName("values").GetShape();
    ge::Shape weights_shape = op.GetInputDescByName("weights").GetShape();
    DataType splits_datatype = op.GetInputDescByName("splits").GetDataType();
    DataType values_datatype = op.GetInputDescByName("values").GetDataType();
    DataType weights_datatype = op.GetInputDescByName("weights").GetDataType();

    if (splits_datatype != DT_INT64) {
      OP_LOGE(TbeGetName(op).c_str(), "'splits' should have dtype int64.");
      return GRAPH_FAILED;
    }

    if (values_datatype != DT_INT32 && values_datatype != DT_INT64) {
      OP_LOGE(TbeGetName(op).c_str(), "'values' should have dtype int32 or int64.");
      return GRAPH_FAILED;
    }

    if (weights_datatype != DT_FLOAT && weights_datatype != DT_DOUBLE &&
        weights_datatype != DT_INT32 && weights_datatype != DT_INT64) {
      OP_LOGE(TbeGetName(op).c_str(), "'weights' should have dtype float, double, int32 or int64.");
      return GRAPH_FAILED;
    }

    if (splits_shape.GetDimNum() != 1) {
      OP_LOGE(TbeGetName(op).c_str(), "'splits' must be a 1D tensor.");
      return GRAPH_FAILED;
    }

    if ((values_shape.GetDimNum() != 1) && (values_shape.GetDimNum() != 2)) {
      OP_LOGE(TbeGetName(op).c_str(), "'values' must be a 1D or 2D tensor.");
      return GRAPH_FAILED;
    }

    int32_t dim = values_shape.GetDimNum();
    std::vector<int64_t> output_indices_dims = { ge::UNKNOWN_DIM, dim + 1 };
    std::vector<int64_t> output_values_dims = { ge::UNKNOWN_DIM };
    std::vector<int64_t> output_dense_shape_dims = { dim + 1 };

    output_indices_desc.SetShape(Shape(output_indices_dims));
    output_values.SetShape(Shape(output_values_dims));
    output_dense_shape_desc.SetShape(Shape(output_dense_shape_dims));

    output_indices_desc.SetDataType(DT_INT64);
    output_values.SetDataType(weights_datatype);
    output_dense_shape_desc.SetDataType(DT_INT64);

    output_indices_desc.SetFormat(values_format);
    output_values.SetFormat(values_format);
    output_dense_shape_desc.SetFormat(values_format);

    op.UpdateOutputDesc("output_indices", output_indices_desc);
    op.UpdateOutputDesc("output_values", output_values);
    op.UpdateOutputDesc("output_dense_shape", output_dense_shape_desc);
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(RaggedCountSparseOutput, RaggedCountSparseOutputInferShape);
// ----------------RaggedCountSparseOutput End----------------------------

// ----------------------SignBitsUnpack Start----------------------
IMPLEMT_VERIFIER(SignBitsUnpack, SignBitsUnpackVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(SignBitsUnpackInferShape) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  if (op_info == nullptr) {
    OP_LOGE(TbeGetName(op).c_str(), "Get op_info failed.");
    return GRAPH_FAILED;
  }
  auto x_desc = op_info->MutableInputDesc("x");
  auto x_mutable_shape = x_desc->MutableShape();
  std::vector<int64_t> x_dims = x_mutable_shape.GetDims();
  int64_t x_shape = x_dims[0];  
  std::vector<std::pair<int64_t, int64_t>> x_range;
  x_desc->GetShapeRange(x_range);
  auto y_desc = op_info->MutableOutputDesc("y");

  int32_t dim;
  if (op.GetAttr("size", dim) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("dim");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (dim <= 0) {
    std::string err_msg = "dim size must be larger than zero!";
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int32_t dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("dtype");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  DataType y_dtype;
  if (dtype == 0) {
    y_dtype = DT_FLOAT;
  } else if (dtype == 1) {
    y_dtype = DT_FLOAT16;
  } else {
    std::string err_msg = "dtype must be DT_FLOAT(0) or DT_FLOAT16(1)!";
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int64_t pack_rate = 8;
  int64_t y_shape = 0;
  std::pair<int64_t, int64_t> y_dim_range;
  if (x_shape != -1) {
    y_shape = x_shape * pack_rate / dim;
    y_dim_range = std::pair<int64_t, int64_t>{y_shape / dim, y_shape / dim};
  } else {
    y_shape = -1;
    int64_t first_range = 0;
    int64_t second_range = 0;
    if (x_range[0].first == 1) {
      first_range = 1;
    } else {
      first_range = x_range[0].first * pack_rate / dim;
    }
    if (x_range[0].second == -1) {
      second_range = -1;
    } else {
      second_range = x_range[0].second * pack_rate / dim;
    }
    y_dim_range = std::pair<int64_t, int64_t>{first_range, second_range};
  }

  std::vector<int64_t> y_dims;
  std::vector<std::pair<int64_t, int64_t>> y_range;
  y_dims.push_back(dim);
  y_dims.push_back(y_shape);

  y_range.push_back(std::pair<int64_t, int64_t>{dim, dim});
  y_range.push_back(y_dim_range);

  y_desc->SetShape(GeShape(y_dims));
  y_desc->SetShapeRange(y_range);
  y_desc->SetDataType(y_dtype);
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(SignBitsUnpack, SignBitsUnpackInferShape);

// Registered verify function
VERIFY_FUNC_REG(SignBitsUnpack, SignBitsUnpackVerify);
// ----------------------SignBitsUnpack End----------------------

// -------------------ScaledMaskedSoftmax---------------------
IMPLEMT_COMMON_INFERFUNC(ScaledMaskedSoftmaxInferShape) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr input_desc = op_desc->MutableInputDesc(0);
  auto y = op_desc->MutableOutputDesc(0);
  y->SetShape(input_desc->GetShape());
  y->SetDataType(DT_FLOAT16);

  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(ScaledMaskedSoftmax, ScaledMaskedSoftmaxInferShape);
// ------------------ScaledMaskedSoftmax END------------------ 

// -------------------ScaledMaskedSoftmaxGrad---------------------
IMPLEMT_COMMON_INFERFUNC(ScaledMaskedSoftmaxGradInferShape) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr input_desc = op_desc->MutableInputDesc(0);
  auto x_grad = op_desc->MutableOutputDesc(0);
  x_grad->SetShape(input_desc->GetShape());
  x_grad->SetDataType(DT_FLOAT16);

  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(ScaledMaskedSoftmaxGrad, ScaledMaskedSoftmaxGradInferShape);
// ------------------ScaledMaskedSoftmaxGrad END------------------ 
// ----------------------SignBitsPack Start------------------------------
IMPLEMT_VERIFIER(SignBitsPack, SignBitsPackVerify) {
  int32_t dim;
  if (op.GetAttr("size", dim) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "size tiling attribute must exist.\n");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

int64_t pack_rate = 8;
int64_t ceil(int64_t num) {
  if (num % pack_rate == 0) {
    return num / pack_rate;
  } else {
    return num / pack_rate + 1;
  }
}

IMPLEMT_COMMON_INFERFUNC(SignBitsPackInferShape) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  if (op_info == nullptr) {
    OP_LOGE(TbeGetName(op).c_str(),"Get op_info failed.");
    return GRAPH_FAILED;
  }
  auto x_desc = op_info->MutableInputDesc("x");
  auto x_mutable_shape = x_desc->MutableShape();
  std::vector<int64_t> x_dims = x_mutable_shape.GetDims();
  int64_t x_shape = x_dims[0];
  std::vector<std::pair<int64_t, int64_t>> x_range;
  x_desc->GetShapeRange(x_range);
  auto y_desc = op_info->MutableOutputDesc("y");

  int32_t dim;
  if (op.GetAttr("size", dim) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("dim");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (dim <= 0) {
    std::string err_msg = "dim size must be larger than zero!";
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  int64_t y_shape = 0;
  std::pair<int64_t, int64_t> y_dim_range;
  if (x_shape != -1) {
    y_shape = ceil(x_shape) / dim;
    y_dim_range = std::pair<int64_t, int64_t>{y_shape, y_shape};
  } else {
    y_shape = -1;
    int64_t first_range = 0;
    int64_t second_range = 0;
    if (x_range[0].first == 1) {
      first_range = 1;
    } else {
      first_range = ceil(x_range[0].first) / dim;
    }
    if (x_range[0].second == -1) {
      second_range = -1;
    } else {
      second_range = ceil(x_range[0].second) / dim;
    }
    y_dim_range = std::pair<int64_t, int64_t>{first_range, second_range};
  }
  
  DataType y_dtype;
  y_dtype = DT_UINT8;

  std::vector<int64_t> y_dims;
  std::vector<std::pair<int64_t, int64_t>> y_range;
  y_dims.push_back(dim);
  y_dims.push_back(y_shape);

  y_range.push_back(std::pair<int64_t, int64_t>{dim, dim});
  y_range.push_back(y_dim_range);

  y_desc->SetShape(GeShape(y_dims));
  y_desc->SetShapeRange(y_range);
  y_desc->SetDataType(y_dtype);
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(SignBitsPack, SignBitsPackInferShape);

// Registered verify function
VERIFY_FUNC_REG(SignBitsPack, SignBitsPackVerify);
// ----------------------SignBitsPack End------------------------------

// ----------------SobolSample Begin------------------------
IMPLEMT_COMMON_INFERFUNC(SobolSampleInferShape) {
  TensorDesc tensordesc_output = op.GetOutputDescByName("samples");
  Tensor dim_tensor;
  Tensor num_results_tensor;
  if(op.GetInputConstData("dim", dim_tensor) == GRAPH_SUCCESS &&
      op.GetInputConstData("num_results", num_results_tensor) == GRAPH_SUCCESS) {
    auto dim_data = reinterpret_cast<int32_t*>(dim_tensor.GetData());
    auto num_results_data =
        reinterpret_cast<int32_t*>(num_results_tensor.GetData());
    if (*dim_data < 1 || *num_results_data < 1) {
      OP_LOGE(TbeGetName(op).c_str(),
              "Both dim and num_results should be bigger than or equal to 1");
      return GRAPH_FAILED;
    }
    tensordesc_output.SetShape(Shape({*num_results_data, *dim_data}));
  } else {
    tensordesc_output.SetShape(Shape({ge::UNKNOWN_DIM, ge::UNKNOWN_DIM}));
  }

  DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    dtype = DT_FLOAT;
  }
  if ((dtype != DT_FLOAT) && (dtype != DT_DOUBLE)) {
    OP_LOGE(TbeGetName(op).c_str(),
            "The attr 'dtype' must be one of DT_FLOAT, DT_DOUBLE");
    return GRAPH_FAILED;
  }
  (void)tensordesc_output.SetDataType(dtype);
  (void)op.UpdateOutputDesc("samples", tensordesc_output);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SobolSample, SobolSampleVerify)
{
  auto dim_type = op.GetInputDescByName("dim").GetDataType();
  auto num_results_type = op.GetInputDescByName("num_results").GetDataType();
  auto skip_type = op.GetInputDescByName("skip").GetDataType();
  auto dim_dims = op.GetInputDescByName("dim").GetShape().GetDimNum();
  auto num_results_dims = op.GetInputDescByName("num_results").GetShape().GetDimNum();
  auto skip_dims = op.GetInputDescByName("skip").GetShape().GetDimNum();

  if (!(dim_dims == 0 && num_results_dims == 0 && skip_dims == 0)) {
    string err_msg1 = ConcatString("All the inputs should be scalars.");
    std::string err_msg = OtherErrMsg(err_msg1);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (!(dim_type == DT_INT32 && num_results_type == DT_INT32 &&
        skip_type == DT_INT32)) {
    string err_msg1 = ConcatString("Dtype of all the inputs should be int32.");
    std::string err_msg = OtherErrMsg(err_msg1);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SobolSample, SobolSampleInferShape);
VERIFY_FUNC_REG(SobolSample, SobolSampleVerify);
// ----------------SobolSample END---------------------------------

// ----------------SparseCountSparseOutput Begin--------------------------
IMPLEMT_COMMON_INFERFUNC(SparseCountSparseOutputInferShape) {
    TensorDesc output_indices_desc = op.GetOutputDescByName("output_indices");
    TensorDesc output_values = op.GetOutputDescByName("output_values");
    TensorDesc output_dense_shape_desc = op.GetOutputDescByName("output_dense_shape");

    Format values_format = op.GetInputDescByName("values").GetFormat();
    ge::Shape values_shape = op.GetInputDescByName("values").GetShape();
    ge::Shape indices_shape = op.GetInputDescByName("indices").GetShape();
    ge::Shape weights_shape = op.GetInputDescByName("weights").GetShape();
    DataType indices_datatype = op.GetInputDescByName("indices").GetDataType();
    DataType values_datatype = op.GetInputDescByName("values").GetDataType();
    DataType dense_shape_datatype = op.GetInputDescByName("dense_shape").GetDataType();
    DataType weights_datatype = op.GetInputDescByName("weights").GetDataType();

    if (indices_datatype != DT_INT64) {
      OP_LOGE(TbeGetName(op).c_str(), "'indices' should have dtype int64.");
      return GRAPH_FAILED;
    }

    if (values_datatype != DT_INT32 && values_datatype != DT_INT64) {
      OP_LOGE(TbeGetName(op).c_str(), "'values' should have dtype int32 or int64.");
      return GRAPH_FAILED;
    }

    if (dense_shape_datatype != DT_INT64) {
      OP_LOGE(TbeGetName(op).c_str(), "'dense_shape' should have dtype int64.");
      return GRAPH_FAILED;
    }

    if (weights_datatype != DT_FLOAT && weights_datatype != DT_DOUBLE &&
        weights_datatype != DT_INT32 && weights_datatype != DT_INT64) {
      OP_LOGE(TbeGetName(op).c_str(), "'weights' should have dtype float, double, int32 or int64.");
      return GRAPH_FAILED;
    }

    if (indices_shape.GetDimNum() != 2){
      OP_LOGE(TbeGetName(op).c_str(), "'indices' must be a 2D tensor.");
      return GRAPH_FAILED;
    }

    int64_t dim = indices_shape.GetDim(1);
    std::vector<int64_t> output_indices_dims = { ge::UNKNOWN_DIM, dim };
    std::vector<int64_t> output_values_dims = { ge::UNKNOWN_DIM };
    std::vector<int64_t> output_dense_shape_dims = { dim };

    output_indices_desc.SetShape(Shape(output_indices_dims));
    output_values.SetShape(Shape(output_values_dims));
    output_dense_shape_desc.SetShape(Shape(output_dense_shape_dims));

    output_indices_desc.SetDataType(DT_INT64);
    output_values.SetDataType(weights_datatype);
    output_dense_shape_desc.SetDataType(DT_INT64);

    output_indices_desc.SetFormat(values_format);
    output_values.SetFormat(values_format);
    output_dense_shape_desc.SetFormat(values_format);

    op.UpdateOutputDesc("output_indices", output_indices_desc);
    op.UpdateOutputDesc("output_values", output_values);
    op.UpdateOutputDesc("output_dense_shape", output_dense_shape_desc);
    return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(SparseCountSparseOutput, SparseCountSparseOutputInferShape);
// ----------------SparseCountSparseOutput End----------------------------

// ---------------RaggedBinCount Op START-------------------
static bool ValidParamsCheck(const ge::OpDescPtr& op_desc) {
  const size_t input_idx_0 = 0; // splits
  const size_t input_idx_1 = 1; // values
  const size_t input_idx_3 = 3; // weights
  const size_t output_idx_0 = 0;

  CHECK(op_desc->MutableInputDesc(input_idx_0) == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT("RaggedBinCount", 
        OtherErrMsg("input splits is null.")), 
        return false);
  CHECK(op_desc->MutableInputDesc(input_idx_1) == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT("RaggedBinCount", 
        OtherErrMsg("input values is null.")), 
        return false);
  CHECK(op_desc->MutableInputDesc(input_idx_3) == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT("RaggedBinCount", 
        OtherErrMsg("input weights is null.")), 
        return false);
  CHECK(op_desc->MutableOutputDesc(output_idx_0) == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT("RaggedBinCount", 
        OtherErrMsg("output is null.")), 
        return false);

  auto splits_format = op_desc->MutableInputDesc(input_idx_0)->GetFormat();
  OP_LOGD("RaggedBinCount", "get the splits format is %s.", 
          TypeUtils::FormatToSerialString(splits_format).c_str());
  CHECK(splits_format != FORMAT_ND,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT("RaggedBinCount", 
        OtherErrMsg("The splits format is invalid.")),
        return false);

  auto values_format = op_desc->MutableInputDesc(input_idx_1)->GetFormat();
  OP_LOGD("RaggedBinCount", "get the values format is %s", 
          TypeUtils::FormatToSerialString(values_format).c_str());
  CHECK(values_format != FORMAT_ND,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT("RaggedBinCount", 
        OtherErrMsg("The values format is invalid.")),
        return false);

  auto weights_format = op_desc->MutableInputDesc(input_idx_3)->GetFormat();
  OP_LOGD("RaggedBinCount", "get the weights format is %s", 
          TypeUtils::FormatToSerialString(weights_format).c_str());
  CHECK(weights_format != FORMAT_ND,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT("RaggedBinCount", 
        OtherErrMsg("The weights format is invalid.")),
        return false);

  return true;
}

static bool GetSizeData(const ge::Operator& op, const ge::OpDescPtr& op_desc, bool& is_size_const,
                        std::vector<int64_t>& size_tensor, const size_t input_idx_2) {
  const std::vector<string> depend_names = {"size"};
  op_desc->SetOpInferDepends(depend_names);
  const size_t size_len = 1;  

  if (!ops::GetConstIntData(op, input_idx_2, size_tensor)) {
    OP_LOGW("RaggedBinCount", "get const value of input size failed, set size = -1.");
    size_tensor = {-1};
    is_size_const = false;
  }
  OP_LOGD("RaggedBinCount", "the size num must be 1. get the num is %zu.", size_tensor.size());
  CHECK(size_tensor.size() != size_len,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT("RaggedBinCount", 
        OtherErrMsg("the input size num must be 1.")),
        return false);
  
  return true;
}

static bool SetOutputDim(const GeShape& splits_shape, GeShape& output_shape,
                         const size_t splits_shape_len, const int64_t size_data) {
  if (!splits_shape.IsUnknownDimNum()) {
    OP_LOGD("RaggedBinCount", "the splits shape size must be 1. get shape size is %zu.", 
            splits_shape.GetDimNum());
    CHECK(splits_shape.GetDimNum() != splits_shape_len,
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT("RaggedBinCount", 
          OtherErrMsg("The dim of input splits is not 1.")),
          return false);
    output_shape.SetDim(0, splits_shape.GetDim(0) - 1);
  }
  output_shape.SetDim(1, size_data);

  return true;
}

static bool CheckShapeLength(const ge::OpDescPtr& op_desc, const size_t shape_len_max) {
  const size_t input_idx_1 = 1; // values
  const size_t input_idx_3 = 3; // weights

  const GeShape& values_shape = op_desc->MutableInputDesc(input_idx_1)->MutableShape();
  const size_t values_shape_len = values_shape.GetDimNum();
  GeShape& weights_shape = op_desc->MutableInputDesc(input_idx_3)->MutableShape();
  const size_t weights_shape_len = weights_shape.GetDimNum();

  if (!values_shape.IsUnknownDimNum()) {
    OP_LOGD("RaggedBinCount", "the values shape size must be less or equal to 2. get shape size is %zu.",
            values_shape_len);
    CHECK(values_shape_len > shape_len_max,
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT("RaggedBinCount",
          OtherErrMsg("The dim of input values is more than 2.")),
          return false);
  }

  if (!weights_shape.IsUnknownDimNum()) {
    OP_LOGD("RaggedBinCount", "the weights shape size must be less or equal to 2. get shape size is %zu.",
            weights_shape_len);
    CHECK(weights_shape_len > shape_len_max,
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT("RaggedBinCount",
          OtherErrMsg("The dim of input weights is more than 2.")),
          return false);
  }

  if (weights_shape_len != values_shape_len) {
    OP_LOGW("RaggedBinCount", "weights shape is not same as values shape, set weights shape to values shape.");
    weights_shape.SetDimNum(values_shape_len);
  }

  return true;
}

static void InferShapeRange(const ge::OpDescPtr& op_desc,
                            std::vector<std::pair<int64_t, int64_t>>& out_range,
                            const size_t values_shape_len,
                            const size_t values_weights_shape_len_max) {
  const size_t input_idx_0 = 0; // splits
  const size_t input_idx_1 = 1; // values
  const size_t input_idx_3 = 3; // weights
  std::vector<std::pair<int64_t, int64_t>> splits_range;
  std::vector<std::pair<int64_t, int64_t>> values_range;

  auto input_desc_splits = op_desc->MutableInputDesc(input_idx_0);
  const GeShape& splits_shape = input_desc_splits->MutableShape();
  auto input_desc_values = op_desc->MutableInputDesc(input_idx_1);
  const GeShape& values_shape = input_desc_values->MutableShape();
  auto input_desc_weights = op_desc->MutableInputDesc(input_idx_3);
  GeShape& weights_shape = input_desc_weights->MutableShape();

  if (!splits_shape.IsUnknownShape() && !splits_shape.IsUnknownDimNum()) {
    out_range.push_back(std::pair<int64_t, int64_t>(splits_shape.GetDim(0) - 1, splits_shape.GetDim(0) - 1));
  }
  else {
    (void)input_desc_splits->GetShapeRange(splits_range);
    out_range.push_back(std::pair<int64_t, int64_t>(splits_range[0].first - 1, splits_range[0].second - 1));
  }

  weights_shape.SetDim(0, values_shape.GetDim(0));
  if (values_shape_len == values_weights_shape_len_max) {
    weights_shape.SetDim(1, values_shape.GetDim(1));
  }

  if (values_shape.IsUnknownShape()) {
    (void)input_desc_values->GetShapeRange(values_range);
    input_desc_weights->SetShapeRange(values_range);
  }
}

static bool SetOutShapeRange(const ge::OpDescPtr& op_desc, const bool is_size_const,
                             std::vector<std::pair<int64_t, int64_t>>& out_range,
                             const int64_t size_data) {
  const size_t input_idx_2 = 2; // size
  const size_t output_idx_0 = 0;
  const size_t size_len = 1;

  if (!is_size_const) {
    std::vector<std::pair<int64_t, int64_t>> size_value_range;
    auto input_size_data = op_desc->MutableInputDesc(input_idx_2);
    CHECK(input_size_data == nullptr,
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT("RaggedBinCount", 
          OtherErrMsg("input size is null.")),
          return false);
    // means no const size value, will get the size value range
    (void)input_size_data->GetValueRange(size_value_range);
    // the size num must be 1, so the size value range num must be 1
    OP_LOGD("RaggedBinCount", "the size value range num must be 1, but get %zu.", size_value_range.size());
    if (size_value_range.size() != size_len) {
      out_range.push_back(std::pair<int64_t, int64_t>(0, -1));
    }
    else {
      out_range.push_back(size_value_range[0]);
    }
  }
  else {
    out_range.push_back(std::pair<int64_t, int64_t>(size_data, size_data));
  }

  op_desc->MutableOutputDesc(output_idx_0)->SetShapeRange(out_range);
  return true;
}

IMPLEMT_INFERFUNC(RaggedBinCount, RaggedBinCountInferShape) {
  auto opname = TbeGetName(op).c_str();
  OP_LOGD(opname, "Enter RaggedBinCount inferfunction!");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_desc == nullptr, OP_LOGE(opname, "op desc is null."), return GRAPH_FAILED);
  if (!ValidParamsCheck(op_desc)) {
    return GRAPH_FAILED;
  }
  const size_t input_idx_0 = 0; // splits
  const size_t input_idx_1 = 1; // values
  const size_t input_idx_2 = 2; // size
  const size_t input_idx_3 = 3; // weights
  const size_t output_idx_0 = 0;
  static const size_t output_len = 2;
  auto input_desc_splits = op_desc->MutableInputDesc(input_idx_0);
  auto input_desc_values = op_desc->MutableInputDesc(input_idx_1);
  auto input_desc_weights = op_desc->MutableInputDesc(input_idx_3);
  auto output_desc = op_desc->MutableOutputDesc(output_idx_0);
  bool is_size_const = true;
  int64_t size_data = UNKNOWN_DIM;
  std::vector<int64_t> size_tensor;
  if (!GetSizeData(op, op_desc, is_size_const, size_tensor, input_idx_2)) {
    return GRAPH_FAILED;
  }
  size_data = size_tensor[0];
  const GeShape& splits_shape = input_desc_splits->MutableShape();
  const GeShape& values_shape = input_desc_values->MutableShape();
  const size_t splits_shape_len = 1;
  GeShape& output_shape = output_desc->MutableShape();
  output_shape.SetDimNum(output_len);
  if (!SetOutputDim(splits_shape, output_shape, splits_shape_len, size_data)) {
    return GRAPH_FAILED;
  }
  const size_t values_weights_shape_len_max = 2;
  const size_t values_shape_len = values_shape.GetDimNum();
  if (!CheckShapeLength(op_desc, values_weights_shape_len_max)) {
    return GRAPH_FAILED;
  }
  output_desc->SetDataType(input_desc_weights->GetDataType());
  CHECK(!output_shape.IsUnknownShape(), OP_LOGD(opname, "the output is static shape. infer success."),
        return GRAPH_SUCCESS);
  output_shape.SetDim(0, -1);
  OP_LOGD(opname, "the output is dynamic shape. will infer range");
  std::vector<std::pair<int64_t, int64_t>> out_range;
  InferShapeRange(op_desc, out_range, values_shape_len, values_weights_shape_len_max);
  if (!SetOutShapeRange(op_desc, is_size_const, out_range, size_data)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(RaggedBinCount, RaggedBinCountVerify) {
  auto opname = TbeGetName(op).c_str();
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_desc == nullptr, OP_LOGE(opname, "op desc is null."), return false);

  const size_t input_idx_0 = 0; // splits
  const size_t input_idx_1 = 1; // values
  const size_t input_idx_2 = 2; // size
  const size_t input_idx_3 = 3; // weights

  auto splits_descptr = op_desc->MutableInputDesc(input_idx_0); // splits
  GeShape splits_shape_check;
  if (WithRank(splits_descptr, 1, splits_shape_check, opname) != GRAPH_SUCCESS) {
    OP_LOGE(opname, "Expected splits should be 1-D. but get %lu.",
            splits_descptr->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  auto values_descptr = op_desc->MutableInputDesc(input_idx_1); // values
  GeShape values_shape_check;
  if (WithRankAtMost(values_descptr, 2, values_shape_check, opname) != GRAPH_SUCCESS) {
    OP_LOGE(opname, "Expected values should be less 2-D. but get %lu.",
            values_descptr->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  auto size_descptr = op_desc->MutableInputDesc(input_idx_2); // size
  GeShape size_shape_check;
  if (WithRank(size_descptr, 1, size_shape_check, opname) != GRAPH_SUCCESS) {
    OP_LOGE(opname, "Expected size should be 1-D. but get %lu.",
            size_descptr->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  auto weights_descptr = op_desc->MutableInputDesc(input_idx_3); // weights
  GeShape weights_shape_check;
  if (WithRankAtMost(weights_descptr, 2, weights_shape_check, opname) != GRAPH_SUCCESS) {
    OP_LOGE(opname, "Expected weights should be less 2-D. but get %lu.",
            weights_descptr->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  bool binary_output;
  if (op.GetAttr("binary_output", binary_output) != GRAPH_SUCCESS) {
    OP_LOGE(opname, "get attr binary_output failed.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(RaggedBinCount, RaggedBinCountInferShape);
VERIFY_FUNC_REG(RaggedBinCount, RaggedBinCountVerify);
// ---------------RaggedBinCount Op END-------------------
}  // namespace ge
