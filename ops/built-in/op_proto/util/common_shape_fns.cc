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
 * \file common_shape_fns.cpp
 * \brief
 */
#include "common_shape_fns.h"
#include <vector>
#include <limits>
#include "op_log.h"
#include "graph/utils/op_desc_utils.h"

namespace ge {
graphStatus WithRankAtLeast(const TensorDesc& tensor, int64_t rank, Shape& out, const char* op_name) {
  if (rank > INT32_MAX) {
    OP_LOGE(op_name, "Rank cannot exceed kint32max");
    return GRAPH_FAILED;
  }
  Shape s = tensor.GetShape();
  std::vector<int64_t> dims = s.GetDims();
  // dim.size() convert to be type int64_t can't overflow
  int64_t size = static_cast<int64_t>(dims.size());
  if (!((size >= rank) || (dims == UNKNOWN_SHAPE))) {
    OP_LOGE(op_name, "Shape's rank must be at least %lld", rank);
    return GRAPH_FAILED;
  }
  out = s;
  return GRAPH_SUCCESS;
}

graphStatus WithRankAtLeast(const GeTensorDescPtr& tensorDesc, int64_t rank, GeShape& out_shape) {
  if (rank > INT32_MAX) {
    OP_LOGE("", "Rank cannot exceed kint32max");
    return GRAPH_FAILED;
  }

  GeShape s = tensorDesc->GetShape();
  std::vector<int64_t> dims = s.GetDims();
  // dim.size() convert to be type int64_t can't overflow
  int64_t size = static_cast<int64_t>(dims.size());

  if ((dims != UNKNOWN_RANK) && (size < rank)) {
    OP_LOGE("", "Shape's rank must be at least %lld, current=%lld", rank, size);
    return GRAPH_FAILED;
  }
  out_shape = s;
  return GRAPH_SUCCESS;
}

graphStatus WithRank(const TensorDesc& tensor, int64_t rank, Shape& out, const char* op_name) {
  if (rank > INT32_MAX) {
    OP_LOGE(op_name, "Rank cannot exceed int32max");
    return GRAPH_FAILED;
  }
  Shape s = tensor.GetShape();
  int64_t existing = static_cast<int64_t>(s.GetDimNum());

  if (s.GetDims() == UNKNOWN_RANK) {
    std::vector<int64_t> out_shape(rank, UNKNOWN_DIM);
    out = Shape(out_shape);
    return GRAPH_SUCCESS;
  }

  if (existing != rank) {
    OP_LOGE(op_name, "Shape must be rank %lld", rank);
    return GRAPH_FAILED;
  }
  out = s;
  return GRAPH_SUCCESS;
}

graphStatus WithRank(const GeTensorDescPtr& tensorDesc, int64_t rank, GeShape& out_shape) {
  if (rank > INT32_MAX) {
    OP_LOGE("", "Rank cannot exceed int32max");
    return GRAPH_FAILED;
  }

  GeShape s = tensorDesc->GetShape();
  int64_t existing = static_cast<int64_t>(s.GetDimNum());
  if (s.GetDims() == UNKNOWN_RANK) {
    std::vector<int64_t> out_dims(rank, UNKNOWN_DIM);
    out_shape = GeShape(out_dims);
    return GRAPH_SUCCESS;
  }

  if (existing != rank) {
    OP_LOGE("", "Shape must be rank %lld, current=%lld", rank, existing);
    return GRAPH_FAILED;
  }
  out_shape = s;
  return GRAPH_SUCCESS;
}

graphStatus WithRank(const GeTensorDescPtr& tensorDesc, int64_t rank, Shape& out_shape) {
  if (rank > INT32_MAX) {
    OP_LOGE("", "Rank cannot exceed int32max");
    return GRAPH_FAILED;
  }

  GeShape s = tensorDesc->GetShape();
  int64_t existing = static_cast<int64_t>(s.GetDimNum());
  if (s.GetDims() == UNKNOWN_RANK) {
    std::vector<int64_t> out_dims(rank, UNKNOWN_DIM);
    out_shape = Shape(out_dims);
    return GRAPH_SUCCESS;
  }

  if (existing != rank) {
    OP_LOGE("", "Shape must be rank %lld, current=%lld", rank, existing);
    return GRAPH_FAILED;
  }
  out_shape = Shape(s.GetDims());
  return GRAPH_SUCCESS;
}

graphStatus WithValue(int64_t dim, int64_t value, int64_t& out, const char* op_name) {
  out = value;
  if (dim == UNKNOWN_DIM) {
    return GRAPH_SUCCESS;
  }

  if (dim != value) {
    OP_LOGE(op_name, "Dim and value are not equal: %lld != %lld.", dim, value);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus Merge(int64_t dim1, int64_t dim2, int64_t& out) {
  if (dim1 == dim2) {
    out = dim1;
    return GRAPH_SUCCESS;
  } else if (dim2 == UNKNOWN_DIM) {
    out = dim1;
    return GRAPH_SUCCESS;
  } else if (dim1 == UNKNOWN_DIM) {
    out = dim2;
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

graphStatus Merge(const Shape& s0, const Shape& s1, Shape& out, const char* op_name) {
  // Same shape and unknown rank
  if (s0.GetDims() == s1.GetDims()) {
    out = s0;
    return GRAPH_SUCCESS;
  } else if (!RankKnown(s1)) {
    out = s0;
    return GRAPH_SUCCESS;
  } else if (!RankKnown(s0)) {
    out = s1;
    return GRAPH_SUCCESS;
  }

  const size_t rank = s0.GetDimNum();
  if (s1.GetDimNum() != rank) {
    OP_LOGE(op_name, "Dimension number of two shapes are not equal: %llu != %llu.", rank, s1.GetDimNum());
    return GRAPH_FAILED;
  }

  // Check if each dims equal
  bool return_s0 = true;
  bool return_s1 = true;
  for (size_t i = 0; i < rank; i++) {
    int64_t d0 = s0.GetDim(i);
    int64_t d1 = s1.GetDim(i);
    if (d0 == UNKNOWN_DIM) {
      if (d1 != UNKNOWN_DIM) {
        return_s0 = false;
      }
    } else if (d1 == UNKNOWN_DIM) {
      return_s1 = false;
    } else if (d0 != d1) {
      OP_LOGE(op_name, "Dims %llu are not equal.", rank);
      return GRAPH_FAILED;
    }
  }

  if (return_s0 || return_s1) {
    out = return_s0 ? s0 : s1;
    return GRAPH_SUCCESS;
  }

  // Merge dims
  std::vector<int64_t> dims(rank, 0);
  for (size_t i = 0; i < rank; ++i) {
    // Invariant for merge was checked earlier, so CHECK is ok.
    if (Merge(s0.GetDim(i), s1.GetDim(i), dims[i]) == GRAPH_FAILED) {
      OP_LOGE(op_name, "Failed to merge dims in rank %llu.", i);
      return GRAPH_FAILED;
    }
  }

  out = Shape(dims);
  return GRAPH_SUCCESS;
}

graphStatus Merge(const GeShape& s0, const GeShape& s1, GeShape& out, const char* op_name) {
  // Same shape and unknown rank
  if (s0.GetDims() == s1.GetDims()) {
    out = s0;
    return GRAPH_SUCCESS;
  } else if (!RankKnown(s1)) {
    out = s0;
    return GRAPH_SUCCESS;
  } else if (!RankKnown(s0)) {
    out = s1;
    return GRAPH_SUCCESS;
  }

  const size_t rank = s0.GetDimNum();
  if (s1.GetDimNum() != rank) {
    OP_LOGE(op_name, "Dimension number of two shapes are not equal: %llu != %llu.", rank, s1.GetDimNum());
    return GRAPH_FAILED;
  }

  // Check if each dims equal
  bool return_s0 = true;
  bool return_s1 = true;
  for (size_t i = 0; i < rank; i++) {
    int64_t d0 = s0.GetDim(i);
    int64_t d1 = s1.GetDim(i);
    if (d0 == UNKNOWN_DIM) {
      if (d1 != UNKNOWN_DIM) {
        return_s0 = false;
      }
    } else if (d1 == UNKNOWN_DIM) {
      return_s1 = false;
    } else if (d0 != d1) {
      OP_LOGE(op_name, "Dims %llu are not equal.", rank);
      return GRAPH_FAILED;
    }
  }

  if (return_s0 || return_s1) {
    out = return_s0 ? s0 : s1;
    return GRAPH_SUCCESS;
  }

  // Merge dims
  std::vector<int64_t> dims(rank, 0);
  for (size_t i = 0; i < rank; ++i) {
    // Invariant for merge was checked earlier, so CHECK is ok.
    if (Merge(s0.GetDim(i), s1.GetDim(i), dims[i]) == GRAPH_FAILED) {
      OP_LOGE(op_name, "Failed to merge dims in rank %llu.", i);
      return GRAPH_FAILED;
    }
  }

  out = GeShape(dims);
  return GRAPH_SUCCESS;
}

graphStatus ReplaceDim(const Shape& s, int64_t dim_index_in, int64_t new_dim, Shape& out, const char* op_name) {
  if (!RankKnown(s)) {
    out = Shape(ge::UNKNOWN_SHAPE);
    return GRAPH_SUCCESS;
  }
  int64_t dim_index = dim_index_in;
  if (dim_index < 0) {
    dim_index = (int64_t)s.GetDimNum() + dim_index;
  }
  if (!FastBoundsCheck(dim_index, s.GetDimNum())) {
    out = Shape();
    OP_LOGE(op_name, "Out of range dim_index %ld for shape with %d dimensions", dim_index_in, s.GetDimNum());
    return GRAPH_FAILED;
  }
  std::vector<int64_t> dims = s.GetDims();
  dims[dim_index] = new_dim;
  out = Shape(dims);
  return GRAPH_SUCCESS;
}

template <typename Ta, typename Tb>
bool FastBoundsCheck(const Ta index, const Tb limit) {
  static_assert(std::is_integral<Ta>::value && std::is_integral<Tb>::value,
                "FastBoundsCheck can only be used on integer types.");
  typedef typename std::make_unsigned<decltype(index + limit)>::type UIndex;
  return static_cast<UIndex>(index) < static_cast<UIndex>(limit);
}

graphStatus Add(int64_t dim1, int64_t dim2, int64_t& out) {
  if (dim1 == 0) {
    out = dim2;
  } else if (dim2 == 0) {
    out = dim1;
  } else if ((dim1 == UNKNOWN_DIM) || (dim2 == UNKNOWN_DIM)) {
    out = UNKNOWN_DIM;
  } else {
    const int64_t sum = dim1 + dim2;
    if (sum < 0) {
      return GRAPH_FAILED;
    }
    out = sum;
  }
  return GRAPH_SUCCESS;
}

graphStatus Subtract(int64_t dim1, int64_t dim2, int64_t& out, const char* op_name) {
  if (dim2 == 0) {
    out = dim1;
  } else if ((dim1 == UNKNOWN_DIM) || (dim2 == UNKNOWN_DIM)) {
    out = UNKNOWN_DIM;
  } else {
    if (dim1 < dim2) {
      OP_LOGE(op_name, "Negative dimension size caused by subtracting, dim1=%ld, dim2=%ld", dim1, dim2);
      return GRAPH_FAILED;
    }
    out = dim1 - dim2;
  }
  return GRAPH_SUCCESS;
}

graphStatus SubShape(const Shape& s, int64_t start, int64_t end, int64_t stride, Shape& out, const char* op_name) {
  if (s.GetDimNum() > INT32_MAX) {
    OP_LOGE(op_name, "shape rank cannot exceed kint32max");
    return GRAPH_FAILED;
  }
  const int64_t rank = static_cast<int64_t>(s.GetDimNum());
  TensorDesc tensor(s);
  if (start == 0 && ((tensor.GetRealDimCnt() != -1 && end >= rank) || end == std::numeric_limits<int64_t>::max())) {
    out = s;
    return GRAPH_SUCCESS;
  }

  if (start > rank) {
    start = rank;
  }
  if (end > rank) {
    end = rank;
  }

  if (stride < 0 && start == rank) {
    --start;
  }

  if (start < 0) {
    start = rank + start;
    if (start < 0) {
      OP_LOGE(op_name, "Subshape start out of bounds must be at least 0");
      return GRAPH_FAILED;
    }
  }

  if (end < 0) {
    end = rank + end;
    if (end < 0) {
      OP_LOGE(op_name, "Subshape end out of bounds must be at least 0");
      return GRAPH_FAILED;
    }
  }

  if (!((stride <= 0 || start <= end))) {
    OP_LOGE(op_name, "Subshape must have computed start <= end");
    return GRAPH_FAILED;
  }
  if (!(stride >= 0 || start >= end)) {
    OP_LOGE(op_name, "Subshape must have computed start >= end since stride is negative");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> dims;
  for (int64_t i = start; stride > 0 ? i < end : i > end; i += stride) {
    dims.push_back(s.GetDim(i));
  }
  Shape tmp(dims);
  out = tmp;
  return GRAPH_SUCCESS;
}

graphStatus SubShape(const GeShape& s, size_t start, size_t end, size_t stride, Shape& out, const char* op_name) {
  Shape output_shape;
  auto ret = SubShape(Shape(s.GetDims()), start, end, stride, output_shape, op_name);
  if (ret == GRAPH_SUCCESS) {
    out = output_shape;
  }
  return ret;
}

graphStatus Concatenate(const Shape& s1, const Shape& s2, Shape& out) {
  if (!RankKnown(s1) || !RankKnown(s2)) {
    out = Shape(ge::UNKNOWN_RANK);
    return GRAPH_SUCCESS;
  }
  size_t s1_rank = s1.GetDimNum();
  size_t s2_rank = s2.GetDimNum();
  size_t rank = s1_rank + s2_rank;
  std::vector<int64_t> dims;
  dims.reserve(rank);
  for (size_t i = 0; i < s1_rank; ++i) {
    dims.push_back(s1.GetDim(i));
  }
  for (size_t i = 0; i < s2_rank; ++i) {
    dims.push_back(s2.GetDim(i));
  }
  Shape s(dims);
  out = s;
  return GRAPH_SUCCESS;
}

graphStatus Matrix(int64_t dim1, int64_t dim2, Shape& out) {
  std::vector<int64_t> dims;
  dims.reserve(2);
  dims.push_back(dim1);
  dims.push_back(dim2);
  Shape s(dims);
  out = s;
  return GRAPH_SUCCESS;
}

graphStatus Vector(int64_t dim, Shape& out) {
  std::vector<int64_t> dims;
  dims.reserve(1);
  dims.push_back(dim);
  Shape s(dims);
  out = s;
  return GRAPH_SUCCESS;
}

static graphStatus GetShapeDataFromShapeTensor(Operator& op, const string& dst_name, int64_t rank,
                                               std::vector<int64_t>& data, const char* op_name) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto shape_data_desc = op_desc->MutableInputDesc(dst_name);

  std::vector<std::string> input_infer_depends = {"output_shape"};
  op_desc->SetOpInferDepends(input_infer_depends);

  GeShape shape_data_shape(shape_data_desc->GetShape());
  std::vector<int64_t> dims = shape_data_shape.GetDims();
  DataType data_type = shape_data_desc->GetDataType();

  if (dims.size() != static_cast<size_t>(rank)) {
    OP_LOGE(op_name, "Shape's rank must be %u, but it is %u", rank, dims.size());
    return GRAPH_FAILED;
  }
  int64_t dim_value = rank > 0 ? dims[0] : 1;
  data.clear();
  data.reserve(dim_value);
  Tensor shape_tensor;
  if (data_type == DT_INT32) {
    if (op.GetInputConstData(dst_name, shape_tensor) == GRAPH_SUCCESS) {
      const int32_t* shape_data = reinterpret_cast<const int32_t*>(shape_tensor.GetData());
      for (int64_t i = 0; i < dim_value; i++) {
        data.push_back(static_cast<int64_t>(shape_data[i]));
      }
    } else {
      OP_LOGI(op.GetName().c_str(), "output_shape is not a const tensor.");
      for (int64_t i = 0; i < dim_value; i++) {
        data.push_back(UNKNOWN_DIM);
      }
    }
  } else if (data_type == DT_INT64) {
    if (op.GetInputConstData(dst_name, shape_tensor) == GRAPH_SUCCESS) {
      const int64_t* shape_data = reinterpret_cast<const int64_t*>(shape_tensor.GetData());
      for (int64_t i = 0; i < dim_value; i++) {
        data.push_back(static_cast<int64_t>(shape_data[i]));
      }
    } else {
      OP_LOGI(op.GetName().c_str(), "output_shape is not a const tensor.");
      for (int64_t i = 0; i < dim_value; i++) {
        data.push_back(UNKNOWN_DIM);
      }
    }
  } else {
    OP_LOGE(op_name, "Data type invalid, should be DT_INT32 or DT_INT64");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

static graphStatus GetShapeDataFromConstData(const Tensor& tensor, int64_t rank, std::vector<int64_t>& data,
                                             const char* op_name) {
  TensorDesc shape_data_desc = tensor.GetTensorDesc();
  Shape shape_data_shape = shape_data_desc.GetShape();
  std::vector<int64_t> dims = shape_data_shape.GetDims();
  DataType data_type = shape_data_desc.GetDataType();

  if (dims.size() != static_cast<size_t>(rank)) {
    OP_LOGE(op_name, "Shape's rank must be %u, but it is %u", rank, dims.size());
    return GRAPH_FAILED;
  }
  int64_t dim_value = rank > 0 ? dims[0] : 1;
  data.clear();
  data.reserve(dim_value);
  if (data_type == DT_INT32) {
    const int32_t* shape_data = reinterpret_cast<const int32_t*>(tensor.GetData());
    for (int64_t i = 0; i < dim_value; i++) {
      data.push_back(static_cast<int64_t>(shape_data[i]));
    }
  } else if (data_type == DT_INT64) {
    const int64_t* shape_data = reinterpret_cast<const int64_t*>(tensor.GetData());
    for (int64_t i = 0; i < dim_value; i++) {
      data.push_back(shape_data[i]);
    }
  } else {
    OP_LOGE(op_name, "Data type invalid, should be DT_INT32 or DT_INT64");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

graphStatus MakeShapeFromShapeTensor(const Tensor& tensor, Shape& out, const char* op_name) {
  std::vector<int64_t> shape_data;
  GetShapeDataFromConstData(tensor, 1, shape_data, op_name);
  out = Shape(shape_data);
  return GRAPH_SUCCESS;
}

graphStatus MakeShapeFromShapeTensor(Operator& op, const string& dst_name, GeShape& out, const char* op_name) {
  std::vector<int64_t> shape_data;
  GetShapeDataFromShapeTensor(op, dst_name, 1, shape_data, op_name);
  out = GeShape(shape_data);
  return GRAPH_SUCCESS;
}

graphStatus MakeDimForScalarInput(const Tensor& tensor, int64_t& out, const char* op_name) {
  std::vector<int64_t> shape_data;
  GetShapeDataFromConstData(tensor, 0, shape_data, op_name);
  out = shape_data[0];
  return GRAPH_SUCCESS;
}

graphStatus WithRankAtMost(const TensorDesc& tensor, int64_t rank, Shape& out, const char* op_name) {
  if (rank > INT32_MAX) {
    OP_LOGE(op_name, "Rank cannot exceed kint32max");
    return GRAPH_FAILED;
  }
  Shape s = tensor.GetShape();
  std::vector<int64_t> dims = s.GetDims();
  if (!((dims.size() <= static_cast<size_t>(rank)) || (dims == ge::UNKNOWN_SHAPE))) {
    OP_LOGE(op_name, "Shape's rank must be at most %lld, but it is %u", rank, dims.size());
    return GRAPH_FAILED;
  }
  out = s;
  return GRAPH_SUCCESS;
}

graphStatus WithRankAtMost(const GeTensorDescPtr& tensorDesc, int64_t rank, GeShape& out_shape) {
  if (rank > INT32_MAX) {
    OP_LOGE("", "Rank cannot exceed kint32max");
    return GRAPH_FAILED;
  }

  GeShape s = tensorDesc->GetShape();
  std::vector<int64_t> dims = s.GetDims();
  if ((dims != ge::UNKNOWN_RANK) && (dims.size() > static_cast<size_t>(rank))) {
    OP_LOGE("", "Shape's rank must be at most %lld, but it is %zu", rank, dims.size());
    return GRAPH_FAILED;
  }

  out_shape = s;
  return GRAPH_SUCCESS;
}

graphStatus Scalar(Shape& out) {
  std::vector<int64_t> dims = {};
  Shape s(dims);
  out = s;
  return GRAPH_SUCCESS;
}

graphStatus UnchangedShape(Operator& op, const string input_name, const string output_name) {
  TensorDesc desc = op.GetOutputDesc(output_name);
  desc.SetShape(op.GetInputDesc(input_name).GetShape());
  return op.UpdateOutputDesc(output_name, desc);
}

graphStatus Divide(const int64_t dividend, const int64_t divisor, const bool evenlyDivisible, int64_t& out,
                   const char* op_name) {
  if (divisor == 1) {
    out = dividend;
  } else if ((dividend == ge::UNKNOWN_DIM) || (divisor == ge::UNKNOWN_DIM)) {
    out = ge::UNKNOWN_DIM;
  } else {
    if (divisor <= 0) {
      OP_LOGE(op_name, "Devide's divisor must be positive, but it is %lld", divisor);
      return GRAPH_FAILED;
    }
    if (!((!evenlyDivisible) || (dividend % divisor) == 0)) {
      OP_LOGE(op_name, "Dimension size must be evenly divisible by %lld, but is %lld", divisor, dividend);
      return GRAPH_FAILED;
    }
    out = dividend / divisor;
  }
  return GRAPH_SUCCESS;
}

bool ShapeFullDefined(const Shape& shape) {
  if (!RankKnown(shape)) {
    return false;
  }
  std::vector<int64_t> dims = shape.GetDims();

  for (const auto& dim : dims) {
    if (dim == ge::UNKNOWN_DIM) {
      return false;
    }
  }
  return true;
}

bool ShapeFullyDefined(const GeShape& shape) {
  if (!RankKnown(shape)) {
    return false;
  }

  std::vector<int64_t> dims = shape.GetDims();
  for (const int64_t& dim : dims) {
    if (dim == ge::UNKNOWN_DIM) {
      return false;
    }
  }

  return true;
}

bool RankKnown(const Shape& shape) {
  std::vector<int64_t> dims = shape.GetDims();
  if (dims == ge::UNKNOWN_RANK) {
    return false;
  }
  return true;
}

bool RankKnown(const GeShape& shape) {
  std::vector<int64_t> dims = shape.GetDims();
  if (dims == ge::UNKNOWN_RANK) {
    return false;
  }
  return true;
}

Shape UnknownShapeOfRank(int64_t rank) {
  std::vector<int64_t> dims(rank);
  for (int64_t i = 0; i < rank; ++i) {
    dims[i] = ge::UNKNOWN_DIM;
  }
  return Shape(dims);
}

bool ValueKnown(const Shape& shape, const size_t& dim_index) {
  if (shape.GetDims() == ge::UNKNOWN_SHAPE) {
    return false;
  }
  if (dim_index >= shape.GetDims().size()) {
    return false;
  }
  if (shape.GetDim(dim_index) == ge::UNKNOWN_DIM) {
    return false;
  }

  return true;
}

graphStatus ValidateSparseTensor(const TensorDesc& indices, const TensorDesc& values, const TensorDesc& shape,
                                 const char* op_name) {
  // Validate ranks
  Shape unused_shape;
  if (WithRank(indices, 2, unused_shape, op_name) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "ValidateSparseTensor indices rank must be 2.");
    return GRAPH_FAILED;
  }
  if (WithRank(values, 1, unused_shape, op_name) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "ValidateSparseTensor values rank must be 1.");
    return GRAPH_FAILED;
  }
  if (WithRank(shape, 1, unused_shape, op_name) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "ValidateSparseTensor shape rank must be 1.");
    return GRAPH_FAILED;
  }

  // Number of elements in indices and values must match
  Shape indices_shape = indices.GetShape();
  Shape values_shape = values.GetShape();
  if (ValueKnown(indices_shape, 0)) {
    if (ValueKnown(values_shape, 0)) {
      if (indices_shape.GetDim(0) != values_shape.GetDim(0)) {
        OP_LOGE(op_name, "Number of elements in index and values do not match.");
        return GRAPH_FAILED;
      }
    }
  }

  // Rank embedded in indices must match shape.
  Shape sparse_shape = shape.GetShape();
  if (ValueKnown(indices_shape, 1)) {
    if (ValueKnown(sparse_shape, 0)) {
      if (indices_shape.GetDim(1) != sparse_shape.GetDim(0)) {
        OP_LOGE(op_name, "Index rank and shape rank do not match.");
        return GRAPH_FAILED;
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus DecodeWavShapeFn(Operator& op) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc(0), 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input must be scalar.");
    return GRAPH_FAILED;
  }

  int64_t channels_dim = 0;
  int32_t desired_channels = 0;
  if (op.GetAttr("desired_channels", desired_channels) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "GetAttr desired_channels error.");
    return GRAPH_FAILED;
  }
  if (desired_channels == -1) {
    channels_dim = ge::UNKNOWN_DIM;
  } else {
    if (desired_channels < 0) {
      OP_LOGE(op.GetName().c_str(), "channels must be non-negative.");
      return GRAPH_FAILED;
    }

    channels_dim = static_cast<int64_t>(desired_channels);
  }
  int64_t samples_dim;
  int32_t desired_samples;
  if (op.GetAttr("desired_samples", desired_samples) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "GetAttr desired_samples error.");
    return GRAPH_FAILED;
  }
  if (desired_samples == -1) {
    samples_dim = ge::UNKNOWN_DIM;
  } else {
    if (desired_samples < 0) {
      OP_LOGE(op.GetName().c_str(), "samples must be non-negative.");
      return GRAPH_FAILED;
    }
    samples_dim = static_cast<int64_t>(desired_samples);
  }

  Shape audio_shape({samples_dim, channels_dim});
  Shape sample_rate_shape;
  (void)Scalar(sample_rate_shape);
  TensorDesc audio_tensor = op.GetOutputDesc("audio");
  audio_tensor.SetDataType(DT_FLOAT);
  audio_tensor.SetShape(audio_shape);
  (void)op.UpdateOutputDesc("audio", audio_tensor);
  TensorDesc sample_rate_tensor = op.GetOutputDesc("sample_rate");
  sample_rate_tensor.SetDataType(DT_INT32);
  sample_rate_tensor.SetShape(sample_rate_shape);
  return op.UpdateOutputDesc("sample_rate", sample_rate_tensor);
}

graphStatus EncodeWavShapeFn(Operator& op) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc(0), 2, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Input audio must be rank 2.");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Input sample_rate must be scalar.");
    return GRAPH_FAILED;
  }

  Shape output_shape;
  (void)Scalar(output_shape);
  TensorDesc contents_tensor = op.GetOutputDesc("contents");
  contents_tensor.SetDataType(DT_STRING);
  contents_tensor.SetShape(output_shape);
  return op.UpdateOutputDesc("contents", contents_tensor);
}

graphStatus SparseSegmentReductionShapeFn(Operator& op) {
  Shape x_shape;
  if (WithRankAtLeast(op.GetInputDesc(0), 1, x_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input x should be at least 1-D.");
    return GRAPH_FAILED;
  }
  Shape indices_shape;
  if (WithRank(op.GetInputDesc(1), 1, indices_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input indices must be 1-D.");
    return GRAPH_FAILED;
  }
  Shape segment_ids_shape;
  if (WithRank(op.GetInputDesc(2), 1, segment_ids_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input segment_ids must be 1-D.");
    return GRAPH_FAILED;
  }
  Shape unused;
  if (Merge(indices_shape, segment_ids_shape, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  Shape subshape;
  if (SubShape(x_shape, 1, x_shape.GetDimNum(), 1, subshape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  Shape out;
  Shape unknown_dim_shape({ge::UNKNOWN_DIM});
  if (Concatenate(unknown_dim_shape, subshape, out) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  TensorDesc out_desc = op.GetOutputDesc(0);
  out_desc.SetDataType(op.GetInputDesc(0).GetDataType());
  out_desc.SetShape(out);
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update y failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus SparseSegmentReductionGradShapeFn(Operator& op) {
  Shape x_shape;
  if (WithRankAtLeast(op.GetInputDesc(0), 1, x_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input x should be at least 1-D.");
    return GRAPH_FAILED;
  }
  Shape indices_shape;
  if (WithRank(op.GetInputDesc(1), 1, indices_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input indices must be 1-D.");
    return GRAPH_FAILED;
  }
  Shape unused;
  Shape segment_ids_shape = op.GetInputDesc(2).GetShape();
  if (Merge(segment_ids_shape, indices_shape, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(3), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input output_dim0 must be scalar.");
    return GRAPH_FAILED;
  }
  Shape subshape;
  if (SubShape(x_shape, 1, x_shape.GetDimNum(), 1, subshape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  Tensor dims0_tensor;
  Shape dim0_shape;
  op.GetInputConstData("output_dim0", dims0_tensor);
  const uint8_t* dims0 = dims0_tensor.GetData();
  const int32_t* dims0_data = reinterpret_cast<const int32_t*>(dims0);
  if (*dims0_data < 0) {
    OP_LOGE(op.GetName().c_str(), "Cannot specify a negative value for output_dim0.");
    return GRAPH_FAILED;
  }
  dim0_shape = Shape({*dims0_data});

  Shape out;
  if (Concatenate(dim0_shape, subshape, out) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  TensorDesc out_desc = op.GetOutputDesc(0);
  out_desc.SetDataType(op.GetInputDesc(0).GetDataType());
  out_desc.SetShape(out);
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update y failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus ValidateVariableResourceHandle(Operator& op, std::vector<ShapeAndType>& shape_and_type) {
  auto input_handle = op.GetInferenceContext()->GetInputHandleShapesAndTypes();
  if (input_handle.empty()) {
    Shape unknown_shape(ge::UNKNOWN_SHAPE);
    ShapeAndType shape_and_type(unknown_shape, DT_UNDEFINED);
    std::vector<ShapeAndType> handle_shapes_and_types;
    handle_shapes_and_types.reserve(1);
    handle_shapes_and_types.emplace_back(shape_and_type);
    input_handle.emplace_back(handle_shapes_and_types);
  } else {
    shape_and_type = input_handle[0];
    DataType value_type;
    if (op.GetAttr("dtype", value_type) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "GetAttr dtype failed.");
      return GRAPH_FAILED;
    }
    if (shape_and_type[0].GetDataType() != value_type) {
      OP_LOGE(op.GetName().c_str(), "ValidateVariableResourceHandle read variable with wrong dtype");
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}
}  // namespace ge
