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
 * \file linalg_ops.cpp
 * \brief
 */
#include "inc/linalg_ops.h"
#include "graph/operator.h"
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "linalg_ops_shape_fns.h"
#include "./util/error_util.h"
#include "util/util.h"

namespace ge {
IMPLEMT_INFERFUNC(CholeskyGrad, CholeskyGradInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto x_desc = op_desc->MutableInputDesc(0);

  GeShape y_shape;
  if (MakeBatchSquareMatrix(x_desc, y_shape, op.GetName().c_str()) !=
      GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(),
                          "Op CholeskyGrad first input x tensor make batch square matrix "
                          "failed.");
    return GRAPH_FAILED;
  }

  DataType type = x_desc->GetDataType();
  auto y_desc = op_desc->MutableOutputDesc(0);
  y_desc->SetShape(y_shape);
  y_desc->SetDataType(type);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(CholeskyGrad, CholeskyGradInfer);

IMPLEMT_INFERFUNC(Cholesky, CholeskyInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto x_desc = op_desc->MutableInputDesc(0);

  GeShape y_shape;
  if (MakeBatchSquareMatrix(x_desc, y_shape, op.GetName().c_str()) !=
      GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(),
                          "Op Cholesky first input x's tensor make batch square matrix failed.");
    return GRAPH_FAILED;
  }
  DataType type = x_desc->GetDataType();

  auto y_desc = op_desc->MutableOutputDesc(0);
  y_desc->SetShape(y_shape);
  y_desc->SetDataType(type);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Cholesky, CholeskyInfer);

// ----------------------Ger Starts----------------------
IMPLEMT_VERIFIER(Ger, GerVerify) {
  DataType x1_type = op.GetInputDesc("x1").GetDataType();
  DataType x2_type = op.GetInputDesc("x2").GetDataType();
  if (x1_type != DT_FLOAT16 && x1_type != DT_FLOAT) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Op Ger first input x1's data type should be fp16 or fp32.");
    return GRAPH_FAILED;
  }
  if (x2_type != DT_FLOAT16 && x2_type != DT_FLOAT) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Op Ger second input x2's data type should be fp16 or fp32.");
    return GRAPH_FAILED;
  }
  if (x1_type != x2_type) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Op Ger two inputs' data type doesn't match.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(Ger, GerInfer) {
  DataType x1_type = op.GetInputDesc("x1").GetDataType();
  Shape x1_shape = op.GetInputDesc("x1").GetShape();
  Shape x2_shape = op.GetInputDesc("x2").GetShape();
  if (x1_shape.GetDims().size() != 1 || x2_shape.GetDims().size() != 1) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "The rank of both input should be one dimensional.");
    return GRAPH_FAILED;
  }

  Shape y_shape;
  Concatenate(x1_shape, x2_shape, y_shape);
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(Shape(y_shape));
  y_desc.SetDataType(x1_type);
  op.UpdateOutputDesc("y", y_desc);

  return GRAPH_SUCCESS;
}

// Registered inferfunction
INFER_FUNC_REG(Ger, GerInfer);
// Registered verify function
VERIFY_FUNC_REG(Ger, GerVerify);
// ----------------------Ger End----------------------

IMPLEMT_INFERFUNC(LogMatrixDeterminant, LogMatrixDeterminantInfer) {
  auto x_shape = op.get_input_desc_x().GetShape().GetDims();
  size_t size_num = x_shape.size();
  if (size_num < 2) {
    string err_msg = ConcatString(
        "the rank of input[x] should be greater than 2, but get ", size_num, ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);                                   
    return GRAPH_FAILED;
  }
  if (x_shape[size_num - 1] != x_shape[size_num - 2]) {
    string err_msg = ConcatString(
        "the last two dimension of input[x] should be equal, but get ", x_shape[size_num - 1], " and ", x_shape[size_num - 2], ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg); 
    return GRAPH_FAILED;
  }

  DataType type = op.GetInputDesc("x").GetDataType();

  if (size_num == 2) {
    TensorDesc sign_desc = op.GetOutputDesc("sign");
    sign_desc.SetShape(Shape({1}));
    sign_desc.SetDataType(type);
    op.UpdateOutputDesc("sign", sign_desc);

    TensorDesc y_desc = op.GetOutputDesc("y");
    y_desc.SetShape(Shape({1}));
    y_desc.SetDataType(type);
    op.UpdateOutputDesc("y", y_desc);
  } else {
    vector<int64_t> shape(x_shape.begin(), (x_shape.end() - 2));

    TensorDesc sign_desc = op.GetOutputDesc("sign");
    sign_desc.SetShape(Shape(shape));
    sign_desc.SetDataType(type);
    op.UpdateOutputDesc("sign", sign_desc);

    TensorDesc y_desc = op.GetOutputDesc("y");
    y_desc.SetShape(Shape(shape));
    y_desc.SetDataType(type);
    op.UpdateOutputDesc("y", y_desc);
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(LogMatrixDeterminant, LogMatrixDeterminantInfer);

IMPLEMT_INFERFUNC(MatrixDeterminant, MatrixDeterminantInfer) {
  auto tensor = op.get_input_desc_x();
  Shape s;
  if (WithRankAtLeast(tensor, 2, s, op.GetName().c_str()) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "The rank of x must be at least 2.");
    return GRAPH_FAILED;
  }

  int64_t existing = s.GetDimNum();
  int64_t dim1 = s.GetDim(existing - 1);
  int64_t dim2 = s.GetDim(existing - 2);
  int64_t unused_dim = 0;

  if (Merge(dim1, dim2, unused_dim) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Merge two dimension failed.");
    return GRAPH_FAILED;
  }

  Shape result;
  if (SubShape(s, 0, -2, 1, result, op.GetName().c_str()) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Op MatrixDeterminant Get SubShape Failed.");
    return GRAPH_FAILED;
  }

  DataType type = op.GetInputDesc("x").GetDataType();

  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(Shape(result));
  y_desc.SetDataType(type);
  op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixDeterminant, MatrixDeterminantInfer);

IMPLEMT_INFERFUNC(MatrixInverse, MatrixInverseInfer) {
  auto tensor = op.get_input_desc_x();
  Shape result;

  if (MakeBatchSquareMatrix(tensor, result, op.GetName().c_str()) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Op first input x tensor Make Batch Square Matrix failed.");
    return GRAPH_FAILED;
  }
  DataType type = op.GetInputDesc("x").GetDataType();

  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(Shape(result));
  y_desc.SetDataType(type);
  op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixInverse, MatrixInverseInfer);

IMPLEMT_INFERFUNC(MatrixSolve, MatrixSolveInfer) {
  auto matrix_tensor = op.get_input_desc_matrix();
  auto rhs_tensor = op.get_input_desc_rhs();
  Shape result;
  if (MatrixSolve(matrix_tensor, rhs_tensor, true, result, op.GetName().c_str()) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Op MatrixSolve Call MatrixSolve Infer Shape fns Failed.");
    return GRAPH_FAILED;
  }
  DataType type = op.GetInputDesc("matrix").GetDataType();

  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(Shape(result));
  y_desc.SetDataType(type);
  op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixSolve, MatrixSolveInfer);

IMPLEMT_INFERFUNC(MatrixSolveLs, MatrixSolveLsInfer) {
  auto matrix_tensor = op.get_input_desc_matrix();
  auto rhs_tensor = op.get_input_desc_rhs();
  auto l2_tensor = op.get_input_desc_l2();

  Shape l2;
  if (WithRank(l2_tensor, 0, l2, op.GetName().c_str()) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Op MatrixSolveLs third input l2 must be a scalar.");
    return GRAPH_FAILED;
  }

  Shape result;
  if (MatrixSolve(matrix_tensor, rhs_tensor, false, result, op.GetName().c_str()) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Op MatrixSolveLs Call MatrixSolve Infer Shape fns Failed.");
    return GRAPH_FAILED;
  }

  DataType type = op.GetInputDesc("matrix").GetDataType();

  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(Shape(result));
  y_desc.SetDataType(type);
  op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixSolveLs, MatrixSolveLsInfer);

IMPLEMT_INFERFUNC(MatrixTriangularSolve, MatrixTriangularSolveInfer) {
  auto matrix_tensor = op.get_input_desc_matrix();
  auto rhs_tensor = op.get_input_desc_rhs();
  Shape result;
  if (MatrixSolve(matrix_tensor, rhs_tensor, true, result, op.GetName().c_str()) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Op MatrixTriangularSolve Call MatrixSolve Infer Shape fns Failed.");
    return GRAPH_FAILED;
  }

  DataType type = op.GetInputDesc("matrix").GetDataType();

  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(Shape(result));
  y_desc.SetDataType(type);
  op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixTriangularSolve, MatrixTriangularSolveInfer);

IMPLEMT_INFERFUNC(Qr, QrInfer) {
  auto tensor = op.get_input_desc_x();
  Shape input;
  if (WithRankAtLeast(tensor, 2, input, op.GetName().c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  Shape batch_shape;
  if (SubShape(input, 0, -2, 1, batch_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  int dim_num = input.GetDimNum();
  int m = input.GetDim(dim_num - 2);
  int n = input.GetDim(dim_num - 1);
  Shape q_shape;
  Shape r_shape;
  auto full_matrices = op.get_attr_full_matrices();

  if (full_matrices) {
    // [...,M,M]; [...,M,N], if full_matrices is true
    Shape m_m_shape;
    Shape m_n_shape;
    Matrix(m, m, m_m_shape);
    Matrix(m, n, m_n_shape);

    Concatenate(batch_shape, m_m_shape, q_shape);
    Concatenate(batch_shape, m_n_shape, r_shape);
  } else {
    // [...,M,P]; [...,P,N], if full_matrices is false
    int p = m > n ? n : m;
    Shape m_p_shape;
    Shape p_n_shape;
    Matrix(m, p, m_p_shape);
    Matrix(p, n, p_n_shape);

    Concatenate(batch_shape, m_p_shape, q_shape);
    Concatenate(batch_shape, p_n_shape, r_shape);
  }

  DataType type = op.GetInputDesc("x").GetDataType();
  TensorDesc q_desc = op.GetOutputDesc("q");
  q_desc.SetShape(Shape(q_shape));
  q_desc.SetDataType(type);
  if (op.UpdateOutputDesc("q", q_desc) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Update q desc failed.");
    return GRAPH_FAILED;
  }

  TensorDesc r_desc = op.GetOutputDesc("r");
  r_desc.SetShape(Shape(r_shape));
  r_desc.SetDataType(type);
  if (op.UpdateOutputDesc("r", r_desc) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Update r desc failed.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Qr, QrInfer);

IMPLEMT_INFERFUNC(SelfAdjointEig, SelfAdjointEigInfer) {
  bool judge = false;
  Shape input;
  if (MakeBatchSquareMatrix(op.get_input_desc_x(), input, op.GetName().c_str()) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Input x make batch square matrix failed");
    return GRAPH_FAILED;
  }

  int64_t n;
  size_t dim_size = op.get_input_desc_x().GetShape().GetDimNum();
  int64_t dim0 = op.get_input_desc_x().GetShape().GetDim(dim_size - 2);
  int64_t dim1 = op.get_input_desc_x().GetShape().GetDim(dim_size - 1);
  if (Merge(dim0, dim1, n) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Merge last two dim of input x failed");
    return GRAPH_FAILED;
  }

  Shape batch_shape;
  if (SubShape(input, 0, -2, 1, batch_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "SubShape batch_shape in SelfAdjointEig failed");
    return GRAPH_FAILED;
  }

  vector<int64_t> value_sec_dims;
  value_sec_dims.reserve(1);
  value_sec_dims.push_back(n);
  Shape value_sec_shape(value_sec_dims);
  Shape value_shape;
  judge = (Concatenate(batch_shape, value_sec_shape, value_shape) != GRAPH_SUCCESS);
  if (judge) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Concatenate eigen_value in SelfAdjointEig failed");
    return GRAPH_FAILED;
  }
  TensorDesc value_desc = op.GetOutputDesc("eigen_value");
  value_desc.SetShape(Shape(value_shape));
  value_desc.SetDataType(op.GetInputDesc("x").GetDataType());
  if (op.UpdateOutputDesc("eigen_value", value_desc) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "fail to update output eigen_value.");
    return GRAPH_FAILED;
  }

  const bool compute = op.get_attr_compute_v();
  if (compute) {
    vector<int64_t> vector_sec_dims;
    vector_sec_dims.reserve(2);
    vector_sec_dims.push_back(n);
    vector_sec_dims.push_back(n);
    Shape vector_sec_shape(vector_sec_dims);
    Shape vector_shape;
    judge = (Concatenate(batch_shape, vector_sec_shape, vector_shape) != GRAPH_SUCCESS);
    if (judge) {
      CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Concatenate eigen_vector in SelfAdjointEig failed");
      return GRAPH_FAILED;
    }
    TensorDesc vector_desc = op.GetOutputDesc("eigen_vector");
    vector_desc.SetShape(Shape(vector_shape));
    vector_desc.SetDataType(op.GetInputDesc("x").GetDataType());
    if (op.UpdateOutputDesc("eigen_vector", vector_desc) != GRAPH_SUCCESS) {
      CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "fail to update eigen_vector.");
      return GRAPH_FAILED;
    }
  } else {
    TensorDesc vector_desc = op.GetOutputDesc("eigen_vector");
    vector_desc.SetShape(Shape());
    vector_desc.SetDataType(op.GetInputDesc("x").GetDataType());
    if (op.UpdateOutputDesc("eigen_vector", vector_desc) != GRAPH_SUCCESS) {
      CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "fail to update eigen_vector.");
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SelfAdjointEig, SelfAdjointEigInfer);

// ----------------Slogdet start-------------------
IMPLEMT_VERIFIER(Slogdet, SlogdetVerify) {
    DataType type = op.GetInputDesc("x").GetDataType();
    if (type != DT_FLOAT16 && type != DT_FLOAT && type != DT_DOUBLE) {
        CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Expert a floating point tensor as input.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(Slogdet, SlogdetInfer) {
    auto x_shape = op.get_input_desc_x().GetShape().GetDims();
    size_t size_num = x_shape.size();
    DataType type = op.GetInputDesc("x").GetDataType();
    if (size_num < 2) {
      CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "The rank of x must be greater than 2");
      return GRAPH_FAILED;
    }
    if (x_shape[size_num - 1] != x_shape[size_num - 2]) {
      CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Last two dimension of x are not equal");
      return GRAPH_FAILED;
    }

    if (size_num == 2) {
        TensorDesc sign_desc = op.GetOutputDesc("sign");
        sign_desc.SetShape(Shape({1}));
        sign_desc.SetDataType(type);
        op.UpdateOutputDesc("sign", sign_desc);

        TensorDesc y_desc = op.GetOutputDesc("y");
        y_desc.SetShape(Shape({1}));
        y_desc.SetDataType(type);
        op.UpdateOutputDesc("y", y_desc);
    } else if (x_shape == ge::UNKNOWN_SHAPE) {
        GE_OP_LOGD(op.GetName().c_str(), "x is unknown shape!");
        std::vector<std::pair<int64_t, int64_t>> out_range;
        std::pair<int64_t, int64_t> pair({1, INT64_MAX});
        out_range.emplace_back(pair);
        TensorDesc sign_desc = op.GetOutputDesc("sign");
        sign_desc.SetShape(Shape(ge::UNKNOWN_SHAPE));
        sign_desc.SetOriginShape(Shape(ge::UNKNOWN_SHAPE));
        sign_desc.SetShapeRange(out_range);
        sign_desc.SetDataType(type);
        op.UpdateOutputDesc("sign", sign_desc);

        TensorDesc y_desc = op.GetOutputDesc("y");
        y_desc.SetShape(Shape(ge::UNKNOWN_SHAPE));
        y_desc.SetOriginShape(Shape(ge::UNKNOWN_SHAPE));
        y_desc.SetShapeRange(out_range);
        y_desc.SetDataType(type);
        op.UpdateOutputDesc("y", y_desc);
    } else {
        vector<int64_t> shape(x_shape.begin(), (x_shape.end() - 2));

        TensorDesc sign_desc = op.GetOutputDesc("sign");
        sign_desc.SetShape(Shape(shape));
        sign_desc.SetDataType(type);
        op.UpdateOutputDesc("sign", sign_desc);

        TensorDesc y_desc = op.GetOutputDesc("y");
        y_desc.SetShape(Shape(shape));
        y_desc.SetDataType(type);
        op.UpdateOutputDesc("y", y_desc);
    }
    return GRAPH_SUCCESS;
}

VERIFY_FUNC_REG(Slogdet, SlogdetVerify);
INFER_FUNC_REG(Slogdet, SlogdetInfer);
// ----------------Slogdet END-------------------

IMPLEMT_INFERFUNC(Svd, SvdInfer) {
  Shape input;

  if (WithRankAtLeast(op.get_input_desc_x(), 2, input, op.GetName().c_str()) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "WithRankAtLeast input in Svd failed!");
    return GRAPH_FAILED;
  }

  size_t dim_size = op.get_input_desc_x().GetShape().GetDimNum();
  int64_t m = op.get_input_desc_x().GetShape().GetDim(dim_size - 2);
  int64_t n = op.get_input_desc_x().GetShape().GetDim(dim_size - 1);
  int64_t p = (m < n) ? m : n;

  Shape batch_shape;

  if (SubShape(input, 0, -2, 1, batch_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "SubShape batch_shape in Svd failed!");
    return GRAPH_FAILED;
  }

  Shape sigma_shape;
  vector<int64_t> sigma_dims;
  sigma_dims.reserve(1);
  sigma_dims.push_back(p);
  Shape e_sec_shape(sigma_dims);
  if (Concatenate(batch_shape, e_sec_shape, sigma_shape) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Concatenate sigma_shap in Svd failed!");
    return GRAPH_FAILED;
  }
  TensorDesc sigma_desc = op.GetOutputDesc("sigma");
  sigma_desc.SetShape(Shape(sigma_shape));
  sigma_desc.SetDataType(op.GetInputDesc("x").GetDataType());
  if (op.UpdateOutputDesc("sigma", sigma_desc) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "fail to update output sigma");
    return GRAPH_FAILED;
  }

  const bool compute = op.get_attr_compute_uv();
  if (compute) {
    Shape u_shape;
    Shape v_shape;
    bool full_matrices = op.get_attr_full_matrices();
    vector<int64_t> u_dims;
    vector<int64_t> v_dims;
    u_dims.reserve(2);
    v_dims.reserve(2);
    if (full_matrices) {
      u_dims.push_back(m);
      u_dims.push_back(m);
      Shape u_sec_shape(u_dims);
      if (Concatenate(batch_shape, u_sec_shape, u_shape) != GRAPH_SUCCESS) {
        CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Concatenate uShape with full_matrices = true in Svd failed!");
        return GRAPH_FAILED;
      }
      v_dims.push_back(n);
      v_dims.push_back(n);
      Shape v_sec_shape(v_dims);
      if (Concatenate(batch_shape, v_sec_shape, v_shape) != GRAPH_SUCCESS) {
        CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Concatenate vShape with full_matrices = true in Svd failed!");
        return GRAPH_FAILED;
      }
    } else {
      u_dims.push_back(m);
      u_dims.push_back(p);
      Shape u_sec_shape(u_dims);
      if (Concatenate(batch_shape, u_sec_shape, u_shape) != GRAPH_SUCCESS) {
        CUBE_INNER_ERR_REPORT(op.GetName().c_str(),
          "concatenate uShape with full_matrices = true in Svd failed!");
        return GRAPH_FAILED;
      }
      v_dims.push_back(n);
      v_dims.push_back(p);
      Shape v_sec_shape(v_dims);
      if (Concatenate(batch_shape, v_sec_shape, v_shape) != GRAPH_SUCCESS) {
        CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Concatenate vShape with full_matrices = true in Svd failed!");
        return GRAPH_FAILED;
      }
    }
    TensorDesc u_desc = op.GetOutputDesc("u");
    u_desc.SetShape(Shape(u_shape));
    u_desc.SetDataType(op.GetInputDesc("x").GetDataType());
    if (op.UpdateOutputDesc("u", u_desc) != GRAPH_SUCCESS) {
      CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "fail to update output u.");
      return GRAPH_FAILED;
    }

    TensorDesc v_desc = op.GetOutputDesc("v");
    v_desc.SetShape(Shape(v_shape));
    v_desc.SetDataType(op.GetInputDesc("x").GetDataType());
    if (op.UpdateOutputDesc("v", v_desc) != GRAPH_SUCCESS) {
      CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "fail to update output v");
      return GRAPH_FAILED;
    }
  } else {
    TensorDesc u_desc = op.GetOutputDesc("u");
    u_desc.SetShape(Shape());
    u_desc.SetDataType(op.GetInputDesc("x").GetDataType());
    if (op.UpdateOutputDesc("u", u_desc) != GRAPH_SUCCESS) {
      CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "fail to update output u.");
      return GRAPH_FAILED;
    }

    TensorDesc v_desc = op.GetOutputDesc("v");
    v_desc.SetShape(Shape());
    v_desc.SetDataType(op.GetInputDesc("x").GetDataType());
    if (op.UpdateOutputDesc("v", v_desc) != GRAPH_SUCCESS) {
      CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "fail to update output v");
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Svd, SvdInfer);

IMPLEMT_INFERFUNC(Lu, LuInfer) {
  Shape input;
  if (WithRankAtLeast(op.GetInputDesc(0), 2, input, op.GetName().c_str()) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "input rank must be at least 2.");
    return GRAPH_FAILED;
  }

  int64_t existing = input.GetDimNum();
  int64_t dim1 = input.GetDim(existing - 2);
  int64_t dim2 = input.GetDim(existing - 1);
  int64_t dim_n = 0;

  if (Merge(dim1, dim2, dim_n) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Merge two dimension failed.");
    return GRAPH_FAILED;
  }

  Shape batch_shape;
  if (SubShape(input, 0, -2, 1, batch_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Op Lu Get SubShape Failed.");
    return GRAPH_FAILED;
  }

  Shape lu_shape;
  vector<int64_t> lu_dims;
  lu_dims.reserve(2);
  lu_dims.push_back(dim_n);
  lu_dims.push_back(dim_n);
  Shape lu_sec_shape(lu_dims);
  if (Concatenate(batch_shape, lu_sec_shape, lu_shape) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Concatenate lu_shape failed!");
    return GRAPH_FAILED;
  }

  Shape p_shape;
  vector<int64_t> p_dims;
  p_dims.reserve(1);
  p_dims.push_back(dim_n);
  Shape p_sec_shape(p_dims);
  if (Concatenate(batch_shape, p_sec_shape, p_shape) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Concatenate p_shape failed!");
    return GRAPH_FAILED;
  }

  TensorDesc lu_desc = op.GetOutputDesc("lu");
  lu_desc.SetShape(Shape(lu_shape));
  DataType lu_type = op.GetInputDesc("input").GetDataType();
  lu_desc.SetDataType(lu_type);
  if (op.UpdateOutputDesc("lu", lu_desc) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "fail to update output lu.");
    return GRAPH_FAILED;
  }

  TensorDesc p_desc = op.GetOutputDesc("p");
  p_desc.SetShape(Shape(p_shape));
  DataType p_type;
  if (op.GetAttr("output_idx_type", p_type) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "Get attr output_idx_type error.");
    return GRAPH_FAILED;
  }
  p_desc.SetDataType(p_type);
  if (op.UpdateOutputDesc("p", p_desc) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "fail to update output p");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Lu, LuInfer);

IMPLEMT_INFERFUNC(MatrixSquareRoot, MatrixSquareRootInfer) {
  auto x_tensor_desc = op.GetInputDesc(0);
  Shape y_shape;
  if (MakeBatchSquareMatrix(x_tensor_desc, y_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "fail to MakeBatchSquareMatrix y_shape.");
    return GRAPH_FAILED;
  }

  TensorDesc y_desc = op.GetOutputDesc("y");
  DataType type = op.GetInputDesc("input").GetDataType();
  y_desc.SetShape(Shape(y_shape));
  y_desc.SetDataType(type);
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op.GetName().c_str(), "fail to update output y.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixSquareRoot, MatrixSquareRootInfer);

IMPLEMT_INFERFUNC(TridiagonalSolve, TridiagonalSolveInfer) {
  Shape lhs;
  std::string error_msg;
  TensorDesc diagonals_desc = op.GetInputDesc(0);
  if (WithRankAtLeast(diagonals_desc, 2, lhs, op.GetName().c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRankAtLeast function, ",
        "the rank of input[diagonals] must be at least 2, but get ",
        diagonals_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), error_msg);
    return GRAPH_FAILED;
  }
  Shape rhs;
  TensorDesc rhs_desc = op.GetInputDesc(1);
  if (WithRankAtLeast(rhs_desc, 2, rhs, op.GetName().c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRankAtLeast function, ",
        "the rank of input[rhs] must be at least 2, but get ",
        rhs_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), error_msg);
    return GRAPH_FAILED;
  }

  Shape lhs_batch_shape;
  if (SubShape(lhs, 0, -2, 1, lhs_batch_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
        std::string("fialed to call SubShape function, get input[lhs] shape failed."));
    return GRAPH_FAILED;
  }
  Shape rhs_batch_shape;
  if (SubShape(rhs, 0, -2, 1, rhs_batch_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
        std::string("fialed to call SubShape function, get input[rhs] shape failed."));
    return GRAPH_FAILED;
  }

  if (Merge(lhs_batch_shape, rhs_batch_shape, lhs_batch_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("fialed to call Merge function, ",
        "merge the shape of input[rhs] and input[rhs] failed.");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), error_msg);
    return GRAPH_FAILED;
  }

  int64_t existing_rhs = rhs.GetDimNum();
  int64_t existing_lhs = lhs.GetDimNum();
  int64_t m_rhs = rhs.GetDim(existing_rhs - 2);
  int64_t m_lhs = lhs.GetDim(existing_lhs - 1);
  if (Merge(m_lhs, m_rhs, m_lhs) != GRAPH_SUCCESS) {
    error_msg = ConcatString("fialed to call Merge function, merge dim[",
        (existing_rhs - 2), "] of input[rhs] and dim[", (existing_lhs - 1),
        "] of input[diagonals] failed.");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), error_msg);
    return GRAPH_FAILED;
  }

  int64_t lhs_dim = lhs.GetDim(existing_lhs - 2);
  if (lhs_dim != 3) {
    error_msg = ConcatString("the dim[", (existing_lhs - 2),
        "] of input[diagonals] should be 3, but get ", lhs_dim, ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), error_msg);
    return GRAPH_FAILED;
  }

  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(Shape(rhs));
  DataType type = op.GetInputDesc("diagonals").GetDataType();
  y_desc.SetDataType(type);
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        std::string("fail to update output[y] desc."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TridiagonalSolve, TridiagonalSolveInfer);
}  // namespace ge