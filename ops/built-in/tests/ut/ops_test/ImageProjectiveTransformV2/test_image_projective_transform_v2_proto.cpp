/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file
 * except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0

 * ImageProjectiveTransformV2 ut case
 */
#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "graph/utils/op_desc_utils.h"
#include "all_ops.h"
#include "common/utils/ut_op_util.h"

class ImageProjectiveTransformV2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ImageProjectiveTransformV2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ImageProjectiveTransformV2 TearDown" << std::endl;
  }
};

using namespace ut_util;
TEST_F(ImageProjectiveTransformV2, ImageProjectiveTransformV2_input_images_ok_test){
  using namespace ge;
  // Graph graph("ReduceSum_1");
  // input images info
  auto images_shape = vector<int64_t>({1, 5, 3, 3});
  auto images_dtype = DT_FLOAT;
  // input transforms info
  auto transform_shape = vector<int64_t>({1, 8});
  auto transform_dtype = DT_FLOAT;
  // input output_shape info
  auto output_shape_shape = vector<int64_t>({2});
  auto output_shape_dtype = DT_INT32;
  vector<uint32_t> output_shape_value = {5, 3};
  // expect result info
  std::vector<int64_t> expected_output_shape = {1, 5, 3, 3};

  // gen ReduceSum op
  auto test_op = op::ImageProjectiveTransformV2("ImageProjectiveTransformV2");
  TENSOR_INPUT_WITH_SHAPE(test_op, images, images_shape, images_dtype, FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(test_op, transforms, transform_shape, transform_dtype, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, output_shape, output_shape_shape, output_shape_dtype, FORMAT_ND, output_shape_value);
  test_op.SetAttr("interpolation",  "NEAREST");
  test_op.SetAttr("fill_mode",  "CONSTANT");

  test_op.InferShapeAndType();
  auto output_desc = test_op.GetOutputDesc(0);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  auto status = test_op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(ImageProjectiveTransformV2, ImageProjectiveTransformV2_input_images_ERR_test){
  using namespace ge;
  // Graph graph("ReduceSum_1");
  // input images info
  auto images_shape = vector<int64_t>({1, 5, 3});
  auto images_dtype = DT_FLOAT;
  // input transforms info
  auto transform_shape = vector<int64_t>({1, 8});
  auto transform_dtype = DT_FLOAT;
  // input output_shape info
  auto output_shape_shape = vector<int64_t>({2});
  auto output_shape_dtype = DT_INT32;
  vector<uint32_t> output_shape_value = {5, 3};
  // expect result info
  std::vector<int64_t> expected_output_shape = {1, 5, 3, 3};

  // gen ReduceSum op
  auto test_op = op::ImageProjectiveTransformV2("ImageProjectiveTransformV2");
  TENSOR_INPUT_WITH_SHAPE(test_op, images, images_shape, images_dtype, FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(test_op, transforms, transform_shape, transform_dtype, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, output_shape, output_shape_shape, output_shape_dtype, FORMAT_ND, output_shape_value);
  test_op.SetAttr("interpolation",  "NEAREST");
  test_op.SetAttr("fill_mode",  "CONSTANT");

  auto ret = test_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
  auto status = test_op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(ImageProjectiveTransformV2, ImageProjectiveTransformV2_outputshape_isnot1D_ERR_test){
  using namespace ge;
  // Graph graph("ReduceSum_1");
  // input images info
  auto images_shape = vector<int64_t>({1, 5, 3, 3});
  auto images_dtype = DT_FLOAT;
  // input transforms info
  auto transform_shape = vector<int64_t>({1, 8});
  auto transform_dtype = DT_FLOAT;
  // input output_shape info
  auto output_shape_shape = vector<int64_t>({3});
  auto output_shape_dtype = DT_INT32;
  vector<uint32_t> output_shape_value = {5, 3, 1};
  // expect result info
  std::vector<int64_t> expected_output_shape = {1, 5, 3, 3};

  // gen ReduceSum op
  auto test_op = op::ImageProjectiveTransformV2("ImageProjectiveTransformV2");
  TENSOR_INPUT_WITH_SHAPE(test_op, images, images_shape, images_dtype, FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(test_op, transforms, transform_shape, transform_dtype, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, output_shape, output_shape_shape, output_shape_dtype, FORMAT_ND, output_shape_value);
  test_op.SetAttr("interpolation",  "NEAREST");
  test_op.SetAttr("fill_mode",  "CONSTANT");

  auto ret = test_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
  auto status = test_op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(ImageProjectiveTransformV2, ImageProjectiveTransformV2_outputshape_interpolation_null){
  using namespace ge;
  // Graph graph("ReduceSum_1");
  // input images info
  auto images_shape = vector<int64_t>({1, 5, 3, 3});
  auto images_dtype = DT_FLOAT;
  // input transforms info
  auto transform_shape = vector<int64_t>({1, 8});
  auto transform_dtype = DT_FLOAT;
  // input output_shape info
  auto output_shape_shape = vector<int64_t>({2});
  auto output_shape_dtype = DT_INT32;
  vector<uint32_t> output_shape_value = {5, 3};
  // expect result info
  std::vector<int64_t> expected_output_shape = {1, 5, 3, 3};

  // gen ReduceSum op
  auto test_op = op::ImageProjectiveTransformV2("ImageProjectiveTransformV2");
  TENSOR_INPUT_WITH_SHAPE(test_op, images, images_shape, images_dtype, FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(test_op, transforms, transform_shape, transform_dtype, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, output_shape, output_shape_shape, output_shape_dtype, FORMAT_ND, output_shape_value);
  test_op.SetAttr("fill_mode",  "CONSTANT");

  test_op.InferShapeAndType();
  auto output_desc = test_op.GetOutputDesc(0);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  auto status = test_op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ImageProjectiveTransformV2, ImageProjectiveTransformV2_input_images_shapeerr_test){
  using namespace ge;
  // Graph graph("ReduceSum_1");
  // input images info
  auto images_shape = vector<int64_t>({1, 5, 3, 3});
  auto images_dtype = DT_FLOAT;
  // input transforms info
  auto transform_shape = vector<int64_t>({1, 8});
  auto transform_dtype = DT_FLOAT;
  // input output_shape info
  auto output_shape_shape = vector<int64_t>({2});
  auto output_shape_dtype = DT_INT32;

  // expect result info
  std::vector<int64_t> expected_output_shape = {1, 5, 3, 3};

  // gen ReduceSum op
  auto test_op = op::ImageProjectiveTransformV2("ImageProjectiveTransformV2");
  TENSOR_INPUT_WITH_SHAPE(test_op, images, images_shape, images_dtype, FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(test_op, transforms, transform_shape, transform_dtype, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(test_op, output_shape, output_shape_shape, output_shape_dtype, FORMAT_ND, {});
  test_op.SetAttr("interpolation",  "NEAREST");
  test_op.SetAttr("fill_mode",  "CONSTANT");

  auto ret = test_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto status = test_op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}
