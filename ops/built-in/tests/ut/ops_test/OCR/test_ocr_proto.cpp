/**
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_ocr_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "ocr_ops.h"

class OCRTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "OCRTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "OCRTest TearDown" << std::endl;
  }
};

TEST_F(OCRTest, BatchEnqueueInferShape) {
  ge::op::BatchEnqueue op;
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(OCRTest, OCRDetectionPreHandleInferShape) {
  ge::op::OCRDetectionPreHandle op;
  op.UpdateInputDesc("img", create_desc({2,2,3}, ge::DT_UINT8));
  op.SetAttr("data_format", "NHWC");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(OCRTest, OCRDetectionPreHandleInferShape1) {
  ge::op::OCRDetectionPreHandle op;
  op.UpdateInputDesc("img", create_desc({3,-1,-1}, ge::DT_UINT8));
  op.SetAttr("data_format", "NCHW");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(OCRTest, OCRDetectionPreHandleInferShapeFail) {
  ge::op::OCRDetectionPreHandle op;
  op.UpdateInputDesc("img", create_desc({2,2,2}, ge::DT_UINT8));
  op.SetAttr("data_format", "NHWC");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(OCRTest, OCRIdentifyPreHandleInferShape) {
  ge::op::OCRIdentifyPreHandle op;
  op.UpdateInputDesc("imgs_data", create_desc({3}, ge::DT_UINT8));
  op.UpdateInputDesc("imgs_offset", create_desc({1}, ge::DT_INT32));
  op.UpdateInputDesc("imgs_size", create_desc({1,3}, ge::DT_INT32));
  
  op.SetAttr("data_format", "NHWC");
  op.SetAttr("size", {1,1});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(OCRTest, OCRIdentifyPreHandleInferShape1) {
  ge::op::OCRIdentifyPreHandle op;
  op.UpdateInputDesc("imgs_data", create_desc({-1}, ge::DT_UINT8));
  op.UpdateInputDesc("imgs_offset", create_desc({-1}, ge::DT_INT32));
  op.UpdateInputDesc("imgs_size", create_desc_shape_range({-1,3}, ge::DT_INT32, ge::FORMAT_ND, {-1, 3},
    ge::FORMAT_ND, {{1, 3}, {3, 3}}));
  
  op.SetAttr("data_format", "NHWC");
  op.SetAttr("size", {1,1});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(OCRTest, OCRIdentifyPreHandleInferShapeFail1) {
  ge::op::OCRIdentifyPreHandle op;
  op.UpdateInputDesc("imgs_data", create_desc({3,2}, ge::DT_UINT8));
  op.UpdateInputDesc("imgs_offset", create_desc({1}, ge::DT_INT32));
  op.UpdateInputDesc("imgs_size", create_desc({1,3}, ge::DT_INT32));
  
  op.SetAttr("data_format", "NHWC");
  op.SetAttr("size", {1,1});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(OCRTest, OCRIdentifyPreHandleInferShapeFail2) {
  ge::op::OCRIdentifyPreHandle op;
  op.UpdateInputDesc("imgs_data", create_desc({3}, ge::DT_UINT8));
  op.UpdateInputDesc("imgs_offset", create_desc({1,2}, ge::DT_INT32));
  op.UpdateInputDesc("imgs_size", create_desc({1,3}, ge::DT_INT32));
  
  op.SetAttr("data_format", "NHWC");
  op.SetAttr("size", {1,1});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(OCRTest, OCRIdentifyPreHandleInferShapeFail3) {
  ge::op::OCRIdentifyPreHandle op;
  op.UpdateInputDesc("imgs_data", create_desc({3}, ge::DT_UINT8));
  op.UpdateInputDesc("imgs_offset", create_desc({1}, ge::DT_INT32));
  op.UpdateInputDesc("imgs_size", create_desc({-1,2}, ge::DT_INT32));
  
  op.SetAttr("data_format", "NHWC");
  op.SetAttr("size", {1,1});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(OCRTest, OCRIdentifyPreHandleInferShapeFail4) {
  ge::op::OCRIdentifyPreHandle op;
  op.UpdateInputDesc("imgs_data", create_desc({3}, ge::DT_UINT8));
  op.UpdateInputDesc("imgs_offset", create_desc({1}, ge::DT_INT32));
  op.UpdateInputDesc("imgs_size", create_desc({-1,3}, ge::DT_INT32));
  
  op.SetAttr("data_format", "NHWC");
  op.SetAttr("size", {1});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(OCRTest, OCRIdentifyPreHandleInferShapeUnknown) {
  ge::op::OCRIdentifyPreHandle op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 100}};
  auto tensor_desc1 = create_desc_shape_range({-1},
                                             ge::DT_UINT8, ge::FORMAT_ND,
                                             {-1},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc2 = create_desc_shape_range({-1},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {-1},
                                             ge::FORMAT_ND, shape_range);
  std::vector<std::pair<int64_t,int64_t>> shape_range2 = {{1, 100}, {3, 3}};
  auto tensor_desc3 = create_desc_shape_range({-1, 3},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {-1, 3},
                                             ge::FORMAT_ND, shape_range2);
  op.UpdateInputDesc("imgs_data", tensor_desc1);
  op.UpdateInputDesc("imgs_offset", tensor_desc2);
  op.UpdateInputDesc("imgs_size", tensor_desc3);
  
  op.SetAttr("data_format", "NHWC");
  op.SetAttr("size", {1,1});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}


TEST_F(OCRTest, OCRRecognitionPreHandleInferShape) {
  ge::op::OCRRecognitionPreHandle op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 100}};
  auto tensor_desc1 = create_desc_shape_range({3},
                                             ge::DT_UINT8, ge::FORMAT_ND,
                                             {3},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("imgs_data", tensor_desc1);
  op.UpdateInputDesc("imgs_offset", create_desc({1}, ge::DT_INT32));
  op.UpdateInputDesc("imgs_size", create_desc({1,3}, ge::DT_INT32));
  op.SetAttr("data_format", "NHWC");
  op.SetAttr("batch_size", 8);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(OCRTest, OCRRecognitionPreHandleInferShapeUnknown) {
  ge::op::OCRRecognitionPreHandle op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 100}};
  auto tensor_desc1 = create_desc_shape_range({-1},
                                             ge::DT_UINT8, ge::FORMAT_ND,
                                             {-1},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("imgs_data", tensor_desc1);
  op.UpdateInputDesc("imgs_offset", create_desc({1}, ge::DT_INT32));
  op.UpdateInputDesc("imgs_size", create_desc({1,3}, ge::DT_INT32));
  op.SetAttr("data_format", "NHWC");
  op.SetAttr("batch_size", 8);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(OCRTest, OCRRecognitionPreHandleInferShapeUnknownHCHW) {
  ge::op::OCRRecognitionPreHandle op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 100}};
  auto tensor_desc1 = create_desc_shape_range({-1},
                                             ge::DT_UINT8, ge::FORMAT_ND,
                                             {-1},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("imgs_data", tensor_desc1);
  op.UpdateInputDesc("imgs_offset", create_desc({1}, ge::DT_INT32));
  op.UpdateInputDesc("imgs_size", create_desc({1,3}, ge::DT_INT32));
  op.SetAttr("data_format", "NCHW");
  op.SetAttr("batch_size", 8);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(OCRTest, OCRRecognitionPreHandleInferShapeBatchSizeFailed) {
  ge::op::OCRRecognitionPreHandle op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 100}};
  auto tensor_desc1 = create_desc_shape_range({-1},
                                             ge::DT_UINT8, ge::FORMAT_ND,
                                             {-1},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("imgs_data", tensor_desc1);
  op.UpdateInputDesc("imgs_offset", create_desc({1}, ge::DT_INT32));
  op.UpdateInputDesc("imgs_size", create_desc({1,3}, ge::DT_INT32));
  op.SetAttr("data_format", "NCHW");
  op.SetAttr("batch_size", 0);
  auto ret = op.InferShapeAndType();
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
}

TEST_F(OCRTest, OCRRecognitionPreHandleInferShapeDimFailed) {
  ge::op::OCRRecognitionPreHandle op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{0, 0}};
  auto tensor_desc1 = create_desc_shape_range({0},
                                             ge::DT_UINT8, ge::FORMAT_ND,
                                             {0},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("imgs_data", tensor_desc1);
  op.UpdateInputDesc("imgs_offset", create_desc({1}, ge::DT_INT32));
  op.UpdateInputDesc("imgs_size", create_desc({1,3}, ge::DT_INT32));
  op.SetAttr("data_format", "NCHW");
  op.SetAttr("batch_size", 8);
  auto ret = op.InferShapeAndType();
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
}

TEST_F(OCRTest, BatchDilatePolysInferShapeUnknown) {
  ge::op::BatchDilatePolys op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 100}};
  auto tensor_desc1 = create_desc_shape_range({-1},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {-1},
                                             ge::FORMAT_ND, shape_range);
  std::vector<std::pair<int64_t,int64_t>> shape_range2 = {{1, 200}, {1, 300}};
  auto tensor_desc2 = create_desc_shape_range({-1, -1},
                                             ge::DT_FLOAT, ge::FORMAT_ND,
                                             {-1, -1},
                                             ge::FORMAT_ND, shape_range2);

  std::vector<std::pair<int64_t,int64_t>> shape_range3 = {{1, 100}};
  auto tensor_desc3 = create_desc_shape_range({-1},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {-1},
                                             ge::FORMAT_ND, shape_range3);
  std::vector<std::pair<int64_t,int64_t>> shape_range4 = {{1, 100}};
  auto tensor_desc4 = create_desc_shape_range({-1},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {-1},
                                             ge::FORMAT_ND, shape_range4);
  op.UpdateInputDesc("polys_data", tensor_desc1);
  op.UpdateInputDesc("polys_offset", tensor_desc3);
  op.UpdateInputDesc("polys_size", tensor_desc4);
  op.UpdateInputDesc("score", tensor_desc2 );
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(OCRTest, BatchDilatePolysInferShape) {
  ge::op::BatchDilatePolys op;
  op.UpdateInputDesc("polys_data", create_desc({2}, ge::DT_INT32));
  op.UpdateInputDesc("polys_offset", create_desc({1}, ge::DT_INT32));
  op.UpdateInputDesc("polys_size", create_desc({1}, ge::DT_INT32));
  op.UpdateInputDesc("score", create_desc({4,4}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(OCRTest, OCRFindContoursInferShape) {
  ge::op::OCRFindContours op;
  op.UpdateInputDesc("img", create_desc({200,300}, ge::DT_UINT8));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(OCRTest, OCRFindContoursInferShapeUnknown) {
  ge::op::OCRFindContours op;
  std::vector<std::pair<int64_t,int64_t>> shape_range ={{1, 200}, {1, 300}};
  auto tensor_desc1 = create_desc_shape_range({-1, -1},
                                             ge::DT_UINT8, ge::FORMAT_ND,
                                             {-1, -1},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("img", tensor_desc1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(OCRTest, DequeueInferShape) {
  ge::op::Dequeue op;
  op.SetAttr("output_type", ge::DT_UINT8);
  op.SetAttr("output_shape", {2,1,1,1});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(OCRTest, OCRDetectionPostHandleShape) {
  ge::op::OCRDetectionPostHandle op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 100}, {1, 100}, {3, 3}};
  auto tensor_desc1 = create_desc_shape_range({50, 50, 3},
                                             ge::DT_UINT8, ge::FORMAT_ND,
                                             {50, 50, 3},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("img", tensor_desc1);
  std::vector<std::pair<int64_t,int64_t>> shape_range2 = {{1, 10}};
  auto tensor_desc2 = create_desc_shape_range({3},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {3},
                                             ge::FORMAT_ND, shape_range2);
  op.UpdateInputDesc("polys_offset", tensor_desc2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(OCRTest, OCRDetectionPostHandleInferShapeUnknown) {
  ge::op::OCRDetectionPostHandle op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 100}, {1, 100}, {3, 3}};
  auto tensor_desc1 = create_desc_shape_range({-1, -1, 3},
                                             ge::DT_UINT8, ge::FORMAT_ND,
                                             {-1, -1, 3},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("img", tensor_desc1);
  std::vector<std::pair<int64_t,int64_t>> shape_range2 = {{1, 10}};
  auto tensor_desc2 = create_desc_shape_range({-1},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {-1},
                                             ge::FORMAT_ND, shape_range2);
  op.UpdateInputDesc("polys_offset", tensor_desc2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(OCRTest, ResizeAndClipPolysInferShape) {
  ge::op::ResizeAndClipPolys op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 100}};
  auto tensor_desc1 = create_desc_shape_range({6},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {6},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("polys_data", tensor_desc1);
  std::vector<std::pair<int64_t,int64_t>> shape_range2 = {{1, 10}};
  auto tensor_desc2 = create_desc_shape_range({1},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {1},
                                             ge::FORMAT_ND, shape_range2);
  op.UpdateInputDesc("polys_offset", tensor_desc2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(OCRTest, ResizeAndClipPolysInferShapeUnknown) {
  ge::op::ResizeAndClipPolys op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 100}};
  auto tensor_desc1 = create_desc_shape_range({-1},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {-1},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("polys_data", tensor_desc1);
  std::vector<std::pair<int64_t,int64_t>> shape_range2 = {{1, 10}};
  auto tensor_desc2 = create_desc_shape_range({-1},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {-1},
                                             ge::FORMAT_ND, shape_range2);
  op.UpdateInputDesc("polys_offset", tensor_desc2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}