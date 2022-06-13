/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_scatter_nd_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "selection_ops.h"

class scatter_nd : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "scatter_nd SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "scatter_nd TearDown" << std::endl;
  }
};

TEST_F(scatter_nd, scatter_nd_infershape_diff_test_1) {
  ge::op::ScatterNd op;
  op.UpdateInputDesc("x", create_desc_with_ori({2, 2, 2}, ge::DT_INT32, ge::FORMAT_ND, {2, 2, 2}, ge::FORMAT_ND));
  /*
  ge::op::Constant shape;
  shape.SetAttr("value", std::vector<int64_t>{1,2,3});
  op.set_input_shape(shape);*/
  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape({3}), ge::FORMAT_ND, ge::DT_INT64);
  constDesc.SetSize(3 * sizeof(int64_t));
  constTensor.SetTensorDesc(constDesc);
  int64_t constData[] = {-1, 2, 3};
  constTensor.SetData((uint8_t*)constData, 3 * sizeof(int64_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_shape(const0);


  ge::TensorDesc tensor_shape = op.GetInputDesc("shape");
  tensor_shape.SetDataType(ge::DT_INT64);

  op.UpdateInputDesc("shape", tensor_shape);
  auto ret = op.InferShapeAndType();
}

TEST_F(scatter_nd, scatter_nd_infershape_diff_test_2) {
  ge::op::ScatterNd op;
  op.UpdateInputDesc("x", create_desc_with_ori({2, 2, 2}, ge::DT_INT32, ge::FORMAT_ND, {2, 2, 2}, ge::FORMAT_ND));


  auto data0 = ge::op::Data().set_attr_index(0);
  op.set_input_shape(data0);


  ge::TensorDesc tensor_shape = op.GetInputDesc("shape");
  tensor_shape.SetDataType(ge::DT_INT64);
  tensor_shape.SetShapeRange({{1,5}});

  op.UpdateInputDesc("shape", tensor_shape);
  auto ret = op.InferShapeAndType();
}
