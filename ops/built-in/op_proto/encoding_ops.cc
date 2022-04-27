/**
 * Copyright (C) Huawei Technologies Co., Ltd 2022-2022. All rights reserved.
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
 * \file encoding_ops.cpp
 * \brief
 */
#include "inc/encoding_ops.h"

#include "vector"

#include "op_log.h"
#include "util/common_shape_fns.h"
#include "util/util.h"
#include "util/error_util.h"

namespace ge {
// ---------------LDPCDecode Op start-------------------
IMPLEMT_VERIFIER(LDPCDecode, LDPCDecodeVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(LDPCDecodeInferShape) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS,
        OP_LOGE("LDPCDecode", "Failed to get op name of LDPCDecode"), return GRAPH_FAILED);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  const int64_t valid_num_size_limit = 1;
  const int64_t matrix_info_size_limit = 2;
  const int64_t input_valid_num_id = 0;
  const int64_t input_matrix_info_id = 1;
  const int64_t output_indices_id = 0;
  const int64_t indices_size = 6;
  const int64_t block_size = 512;
  auto valid_num_desc = op_desc->MutableInputDesc(input_valid_num_id);
  auto matrix_info_desc = op_desc->MutableInputDesc(input_matrix_info_id);
  const GeShape &valid_num_shape = valid_num_desc->MutableShape();
  const GeShape &matrix_info_shape = matrix_info_desc->MutableShape();
  auto valid_num_dtype = valid_num_desc->GetDataType();
  auto matrix_info_dtype = matrix_info_desc->GetDataType();

  if (valid_num_shape.GetDimNum() != valid_num_size_limit) {
    OP_LOGE(op_name.GetString(), "Expected dim of valid_num should be 1. real value is %lu.",
            valid_num_shape.GetDimNum());
    return GRAPH_FAILED;
  }

  if (matrix_info_shape.GetDimNum() != matrix_info_size_limit) {
    OP_LOGE(op_name.GetString(), "Expected dim of matrix_info_shape should be 2. real value is %lu.",
            matrix_info_shape.GetDimNum());
    return GRAPH_FAILED;
  }

  if (valid_num_dtype != matrix_info_dtype) {
    OP_LOGE(op_name.GetString(), "Expected dtype of matrix_info and valid_num should be same.");
    return GRAPH_FAILED;
  }

  auto indices_desc = op_desc->MutableOutputDesc(output_indices_id);
  auto &output_shape = indices_desc->MutableShape();
  auto output_dim_0 = valid_num_shape.GetDim(0) * block_size;
  size_t rank = 2;
  output_shape.SetDimNum(rank);
  output_shape.SetDim(0, output_dim_0);
  output_shape.SetDim(1, indices_size);
  indices_desc->SetDataType(valid_num_dtype);
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(LDPCDecode, LDPCDecodeInferShape);

VERIFY_FUNC_REG(LDPCDecode, LDPCDecodeVerify);
// ----------------LDPCDecode END---------------------
}  // namespace ge
