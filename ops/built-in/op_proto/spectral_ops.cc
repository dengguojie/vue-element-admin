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
 * \file spectral_ops.cpp
 * \brief
 */
#include "inc/spectral_ops.h"
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "util/util.h"
#include "util/error_util.h"

namespace ge {
IMPLEMT_INFERFUNC(IFFT, IFFTInfer) {
  const char *op_name = TbeGetName(op).c_str();
  Shape out;
  if (WithRankAtLeast(op.GetInputDesc(0), 1, out, op_name) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Input out rank must be at least 1.");
    return GRAPH_FAILED;
  }
  DataType type = op.GetInputDesc(0).GetDataType();
  TensorDesc y_desc = op.GetOutputDesc(0);
  y_desc.SetShape(Shape(out));
  y_desc.SetDataType(type);
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Fail to update y.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(IFFT, IFFTInfer);

IMPLEMT_INFERFUNC(RFFT, RFFTInfer) {
  Shape out;
  if (WithRankAtLeast(op.GetInputDesc(0), 1, out, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(
        0, DebugString(op.GetInputDesc(0).GetShape().GetDims()),
        "at least 1D");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  Shape fft_length_input;
  if (WithRank(op.GetInputDesc(1), 1, fft_length_input, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(
        1, DebugString(op.GetInputDesc(1).GetShape().GetDims()),
        "1D");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (fft_length_input.GetDim(0) != 1) {
    if (fft_length_input.GetDim(0) != UNKNOWN_DIM) {
      std::string err_msg = ConcatString(
          "0th dim of input[fft_length] must be 1, real value is ",
		  fft_length_input.GetDim(0));
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }

  TensorDesc y_desc = op.GetOutputDesc("y");
  // unknown shape support
  const char* kFftLengthField = "fft_length";
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends({kFftLengthField});

  int64_t dim_out = out.GetDimNum()-1;
  Tensor fft_length_tensor;
  int status = op.GetInputConstData(kFftLengthField, fft_length_tensor);
  if (status != GRAPH_SUCCESS) {
    out.SetDim(dim_out, UNKNOWN_DIM);
  } else {
    const int32_t* fft_length_as_vec = reinterpret_cast<const int32_t*>(fft_length_tensor.GetData());
    auto dim = fft_length_as_vec[0] != 0 ? fft_length_as_vec[0] / 2 + 1 : fft_length_as_vec[0];
    int64_t dim_replace = static_cast<int64_t>(dim);
    out.SetDim(dim_out, dim_replace);
  }

  y_desc.SetShape(Shape(out));
  y_desc.SetDataType(DT_COMPLEX64);
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), std::string("update output[y] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(RFFT, RFFTInfer);

IMPLEMT_INFERFUNC(IRFFT, IRFFTInfer) {
  Shape out;

  if (WithRankAtLeast(op.GetInputDesc(0), 1, out, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Input out rank must be at least 1.");
    return GRAPH_FAILED;
  }

  Shape fft_length_input;
  if (WithRank(op.GetInputDesc(1), 1, fft_length_input, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Input fft_length_input rank must be 1.");
    return GRAPH_FAILED;
  }

  if (fft_length_input.GetDim(0) != 1) {
    if (fft_length_input.GetDim(0) != UNKNOWN_DIM) {
      OP_LOGE(TbeGetName(op).c_str(), "The fft_length_input dim-0 must be 1, real value is [%ld].",
        fft_length_input.GetDim(0));
      return GRAPH_FAILED;
    }
  }

  TensorDesc y_desc = op.GetOutputDesc("y");
  // unknown shape support
  const char* kFftLengthField = "fft_length";
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends({kFftLengthField});

  int64_t dim_out = out.GetDimNum()-1;
  Tensor fft_length_tensor;
  int status = op.GetInputConstData(kFftLengthField, fft_length_tensor);
  if (status != GRAPH_SUCCESS) {
    out.SetDim(dim_out, UNKNOWN_DIM);
  } else {
    const int32_t* fft_length_as_vec = reinterpret_cast<const int32_t*>(fft_length_tensor.GetData());
    auto dim = fft_length_as_vec[0];
    int64_t dim_replace = static_cast<int64_t>(dim);
    out.SetDim(dim_out, dim_replace);
  }

  y_desc.SetShape(Shape(out));
  y_desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Fail to update output y.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(IRFFT, IRFFTInfer);

IMPLEMT_INFERFUNC(FFT2D, FFT2DInfer) {
  const static int rank = 2;
  Shape out;
  if (WithRankAtLeast(op.GetInputDesc(0), rank, out, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Input out rank must be at least [%d].", rank);
    return GRAPH_FAILED;
  }

  DataType y_type = op.GetInputDesc("x").GetDataType();
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(Shape(out));
  y_desc.SetDataType(y_type);

  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Fail to update output y.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(FFT2D, FFT2DInfer);

IMPLEMT_INFERFUNC(FFT, FFTInfer) {
  const char *op_name = TbeGetName(op).c_str();
  Shape out;
  if (WithRankAtLeast(op.GetInputDesc(0), 1, out, op_name) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Input out rank must be at least 1.");
    return GRAPH_FAILED;
  }
  DataType type = op.GetInputDesc(0).GetDataType();
  TensorDesc y_desc = op.GetOutputDesc(0);
  y_desc.SetShape(Shape(out));
  y_desc.SetDataType(type);
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Fail to update output.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(FFT, FFTInfer);

IMPLEMT_INFERFUNC(IFFT2D, IFFT2DInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto x_desc = op_desc->MutableInputDesc(0);

  GeShape out;
  if (WithRankAtLeast(x_desc, 2, out, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Input rank must be at least 2.");
    return GRAPH_FAILED;
  }
  DataType type = x_desc->GetDataType();

  auto y_desc = op_desc->MutableOutputDesc(0);
  y_desc->SetShape(out);
  y_desc->SetDataType(type);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(IFFT2D, IFFT2DInfer);

}  // namespace ge
