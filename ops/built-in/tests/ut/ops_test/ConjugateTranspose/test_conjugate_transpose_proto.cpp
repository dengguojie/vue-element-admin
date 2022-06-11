/**
* Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

* This program is free software; you can redistribute it and/or modify
* it under the terms of the Apache License Version 2.0. You may not use this
file except in compliance with the License.

* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* Apache License for more details at
* http://www.apache.org/licenses/LICENSE-2.0
*
* @file test_conjugate_transpose_proto.cpp
*
* @brief
*
* @version 1.0
*
*/

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "linalg_ops.h"
#include "array_ops.h"

class conjugate_transpose : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ConjugateTranspose SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ConjugateTranspose TearDown" << std::endl;
  }
};

TEST_F(conjugate_transpose, ConjugateTranspose_infer_shape_0) {
  ge::op::ConjugateTranspose op;
  ge::TensorDesc tensorDesc1;
  ge::Shape shape1({2, 3, 4});
  tensorDesc1.SetDataType(ge::DT_COMPLEX64);
  tensorDesc1.SetShape(shape1);
  op.UpdateInputDesc("x", tensorDesc1);
  std::vector<int64_t> dims_perm{3};
  ge::Tensor constTensor;
  ge::TensorDesc tensor_desc_perm(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  int element_size = dims_perm.size();
  tensor_desc_perm.SetSize(element_size * sizeof(int32_t));
  constTensor.SetTensorDesc(tensor_desc_perm);
  int* conv_perm_tensor_value = new int[element_size];
  for (int i = 0; i < element_size; i++) {
    *(conv_perm_tensor_value + i) = i;
  }
  constTensor.SetData((uint8_t*)conv_perm_tensor_value, element_size * sizeof(int32_t));
  auto const0 = ge::op::Constant("perm").set_attr_value(constTensor);
  op.set_input_perm(const0);
  delete[] conv_perm_tensor_value;
  op.UpdateInputDesc("perm", tensor_desc_perm);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(conjugate_transpose, ConjugateTranspose_infer_shape_2) {
  ge::op::ConjugateTranspose op;
  ge::TensorDesc tensorDesc1;
  ge::Shape shape1({2, 3, 4});
  tensorDesc1.SetDataType(ge::DT_COMPLEX128);
  tensorDesc1.SetShape(shape1);
  op.UpdateInputDesc("x", tensorDesc1);
  std::vector<int64_t> dims_perm{3};
  ge::Tensor constTensor;
  ge::TensorDesc tensor_desc_perm(ge::Shape(), ge::FORMAT_ND, ge::DT_INT64);
  int element_size = dims_perm.size();
  tensor_desc_perm.SetSize(element_size * sizeof(int64_t));
  constTensor.SetTensorDesc(tensor_desc_perm);
  int* conv_perm_tensor_value = new int[element_size];
  for (int i = 0; i < element_size; i++) {
    *(conv_perm_tensor_value + i) = i;
  }
  constTensor.SetData((uint8_t*)conv_perm_tensor_value, element_size * sizeof(int64_t));
  auto const0 = ge::op::Constant("perm").set_attr_value(constTensor);
  op.set_input_perm(const0);
  delete[] conv_perm_tensor_value;
  op.UpdateInputDesc("perm", tensor_desc_perm);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(conjugate_transpose, ConjugateTranspose_infer_shape_3) {
  ge::op::ConjugateTranspose op;
  ge::TensorDesc tensorDesc1;
  ge::Shape shape1({2, 3, 4});
  tensorDesc1.SetDataType(ge::DT_INT32);
  tensorDesc1.SetShape(shape1);
  op.UpdateInputDesc("x", tensorDesc1);
  std::vector<int64_t> dims_perm{3};
  ge::Tensor constTensor;
  ge::TensorDesc tensor_desc_perm(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  int element_size = dims_perm.size();
  tensor_desc_perm.SetSize(element_size * sizeof(int32_t));
  constTensor.SetTensorDesc(tensor_desc_perm);
  int* conv_perm_tensor_value = new int[element_size];
  for (int i = 0; i < element_size; i++) {
    *(conv_perm_tensor_value + i) = i;
  }
  constTensor.SetData((uint8_t*)conv_perm_tensor_value, element_size * sizeof(int32_t));
  auto const0 = ge::op::Constant("perm").set_attr_value(constTensor);
  op.set_input_perm(const0);
  delete[] conv_perm_tensor_value;
  op.UpdateInputDesc("perm", tensor_desc_perm);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
TEST_F(conjugate_transpose, ConjugateTranspose_infer_shape_1) {
  ge::op::ConjugateTranspose op;
  op.UpdateInputDesc("x", create_desc({-2}, ge::DT_COMPLEX64));
  op.UpdateInputDesc("perm", create_desc({-2}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}