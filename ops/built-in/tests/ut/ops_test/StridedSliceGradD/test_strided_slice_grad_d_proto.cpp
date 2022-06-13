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
 * @file test_strided_slice_grad_d_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "selection_ops.h"
#include <string>
#include <vector>
using namespace std;

/*
    .INPUT(dy, TensorType::BasicType())
    .OUTPUT(output, TensorType::BasicType())
    .REQUIRED_ATTR(shape, ListInt)
    .REQUIRED_ATTR(begin, ListInt)
    .REQUIRED_ATTR(end, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(begin_mask, Int, 0)
    .ATTR(end_mask, Int, 0)
    .ATTR(ellipsis_mask, Int, 0)
    .ATTR(new_axis_mask, Int, 0)
    .ATTR(shrink_axis_mask, Int, 0)
*/

class strided_slice_grad_d : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "strided_slice_grad_d SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "strided_slice_grad_d TearDown" << std::endl;
  }

  template<typename T>
  void test(const vector<int64_t> &shape,
            const vector<T> &begin,
            const vector<T> &end,
            const vector<T> &strides,
            const vector<int64_t> &dy,
            int32_t begin_mask,
            int32_t end_mask,
            int32_t ellipsis_mask,
            int32_t new_axis_mask,
            int32_t shrink_axis_mask,
            const vector<std::pair<int64_t, int64_t>> &shape_shape_range,
            const vector<std::pair<int64_t, int64_t>> &dy_shape_range,
            vector<int64_t> &output_shape,
            vector<std::pair<int64_t, int64_t>> &output_range) {
    ge::op::StridedSliceGradD op;
    auto dy_tensor_desc = create_desc_shape_range(dy,
                                                  ge::DT_FLOAT16, ge::FORMAT_ND,
                                                  dy,
                                                  ge::FORMAT_ND, dy_shape_range);
    op.UpdateInputDesc("dy", dy_tensor_desc);
    auto begin_dtype = ge::DT_INT32;
    if (sizeof(T) == sizeof(int64_t)) {
      begin_dtype = ge::DT_INT64;
    }

    op.SetAttr("begin", begin);
    op.SetAttr("end", end);
    op.SetAttr("strides", strides);
    op.SetAttr("shape", shape);

    op.SetAttr("begin_mask", begin_mask);
    op.SetAttr("end_mask", end_mask);
    op.SetAttr("ellipsis_mask", ellipsis_mask);
    op.SetAttr("new_axis_mask", new_axis_mask);
    op.SetAttr("shrink_axis_mask", shrink_axis_mask);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("output");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    output_shape = output_desc.GetShape().GetDims();
    EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  }
  template<typename T>
  void testfalse(const vector<int64_t> &shape,
                 const vector<T> &begin,
                 const vector<T> &end,
                 const vector<T> &strides,
                 const vector<int64_t> &dy,
                 int32_t begin_mask,
                 int32_t end_mask,
                 int32_t ellipsis_mask,
                 int32_t new_axis_mask,
                 int32_t shrink_axis_mask,
                 const vector<std::pair<int64_t, int64_t>> &shape_shape_range,
                 const vector<std::pair<int64_t, int64_t>> &dy_shape_range,
                 vector<int64_t> &output_shape,
                 vector<std::pair<int64_t, int64_t>> &output_range) {
    ge::op::StridedSliceGradD op;
    auto dy_tensor_desc = create_desc_shape_range(dy,
                                                  ge::DT_FLOAT16, ge::FORMAT_ND,
                                                  dy,
                                                  ge::FORMAT_ND, dy_shape_range);
    op.UpdateInputDesc("dy", dy_tensor_desc);
    auto begin_dtype = ge::DT_INT32;
    if (sizeof(T) == sizeof(int64_t)) {
      begin_dtype = ge::DT_INT64;
    }

    op.SetAttr("begin", begin);
    op.SetAttr("end", end);
    op.SetAttr("strides", strides);
    op.SetAttr("shape", shape);

    op.SetAttr("begin_mask", begin_mask);
    op.SetAttr("end_mask", end_mask);
    op.SetAttr("ellipsis_mask", ellipsis_mask);
    op.SetAttr("new_axis_mask", new_axis_mask);
    op.SetAttr("shrink_axis_mask", shrink_axis_mask);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
  }
};

TEST_F(strided_slice_grad_d, strided_slice_grad_infer_shape_range_001) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  test<int32_t>({3},
                {},
                {},
                {1, 1, 1},
                {32,13,13},
                0, 0, 0, 0, 0,
                {},
                {},
                output_shape, output_shape_range);
}

TEST_F(strided_slice_grad_d, strided_slice_grad_infer_shape_range_002) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  testfalse<int32_t>({},
                     {},
                     {},
                     {1, 1, 1},
                     {},
                     0, 0, 0, 0, 0,
                     {},
                     {},
                     output_shape, output_shape_range);
}

TEST_F(strided_slice_grad_d, strided_slice_grad_infer_shape_range_003) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  test<int32_t>({-2},
                {},
                {},
                {1, 1, 1},
                {32,13,13},
                0, 0, 0, 0, 0,
                {},
                {},
                output_shape, output_shape_range);
}
