/**
 * Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_decode_jpeg_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "op_proto_test_util.h"
#include "util/common_shape_fns.h"
#include "utils/node_utils.h"
#include "utils/op_desc_utils.h"
#include "utils/type_utils.h"
#include "debug/ge_attr_define.h"
#include "tensor.h"
#include "operator.h"
#include "op_desc.h"
#include "ge_tensor.h"
#include "resource_context.h"
#include "image_ops.h"

class DecodeImageV3 : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "DecodeImageV3 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DecodeImageV3 TearDown" << std::endl;
  }
};

TEST_F(DecodeImageV3, decode_image_v3_infer_shape_right_type_0) {
  ge::op::DecodeImage op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {};
  auto tensor_desc = create_desc_shape_range({},
                                             ge::DT_FLOAT, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("contents", tensor_desc);
  op.SetAttr("expand_animations", false);
  op.SetAttr("channels", 3);
  op.SetAttr("dtype", ge::DT_UINT8);
  std::string maxShape("300,100,3");
  op.SetAttr(ge::ATTR_NAME_OP_MAX_SHAPE, maxShape);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  ge::TensorDesc imageDesc = op.GetOutputDesc(0);
  ge::Shape imageShage = imageDesc.GetShape();
  EXPECT_EQ(imageShage.GetDimNum(), 3);
  EXPECT_EQ(imageShage.GetDim(0), -1);
  EXPECT_EQ(imageShage.GetDim(1), -1);
  EXPECT_EQ(imageShage.GetDim(2), 3);
  
  EXPECT_EQ(imageDesc.GetShapeRange(shape_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(shape_range.size(), 3);
  EXPECT_EQ(shape_range[0].first, 1);
  EXPECT_EQ(shape_range[1].first, 1);
  EXPECT_EQ(shape_range[2].first, 1);
  EXPECT_EQ(shape_range[0].second, 300);
  EXPECT_EQ(shape_range[1].second, 100);
  EXPECT_EQ(shape_range[2].second, 3);
}

TEST_F(DecodeImageV3, decode_image_v3_infer_shape_right_type_1) {
  ge::op::DecodeImage op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {};
  auto tensor_desc = create_desc_shape_range({},
                                             ge::DT_FLOAT, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("contents", tensor_desc);
  op.SetAttr("expand_animations", false);
  op.SetAttr("channels", 3);
  op.SetAttr("dtype", ge::DT_UINT8);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  ge::TensorDesc imageDesc = op.GetOutputDesc(0);
  ge::Shape imageShage = imageDesc.GetShape();
  EXPECT_EQ(imageShage.GetDimNum(), 3);
  EXPECT_EQ(imageShage.GetDim(0), -1);
  EXPECT_EQ(imageShage.GetDim(1), -1);
  EXPECT_EQ(imageShage.GetDim(2), 3);
  
  EXPECT_EQ(imageDesc.GetShapeRange(shape_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(shape_range.size(), 0);
}

TEST_F(DecodeImageV3, decode_image_v3_infer_shape_right_type_2) {
  ge::op::DecodeImage op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {};
  auto tensor_desc = create_desc_shape_range({},
                                             ge::DT_FLOAT, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("contents", tensor_desc);
  op.SetAttr("expand_animations", true);
  op.SetAttr("channels", 3);
  op.SetAttr("dtype", ge::DT_UINT8);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  ge::TensorDesc imageDesc = op.GetOutputDesc(0);
  ge::Shape imageShage = imageDesc.GetShape();
  EXPECT_EQ(imageShage.GetDimNum(), 0);
}

TEST_F(DecodeImageV3, decode_image_v3_infer_shape_right_type_3) {
  ge::op::DecodeImage op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {};
  auto tensor_desc = create_desc_shape_range({},
                                             ge::DT_FLOAT, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("contents", tensor_desc);
  op.SetAttr("expand_animations", false);
  op.SetAttr("channels", 0);
  op.SetAttr("dtype", ge::DT_UINT8);
  std::string maxShape("300,100,3");
  op.SetAttr(ge::ATTR_NAME_OP_MAX_SHAPE, maxShape);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  ge::TensorDesc imageDesc = op.GetOutputDesc(0);
  ge::Shape imageShage = imageDesc.GetShape();
  EXPECT_EQ(imageShage.GetDimNum(), 3);
  EXPECT_EQ(imageShage.GetDim(0), -1);
  EXPECT_EQ(imageShage.GetDim(1), -1);
  EXPECT_EQ(imageShage.GetDim(2), -1);
  
  EXPECT_EQ(imageDesc.GetShapeRange(shape_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(shape_range.size(), 3);
  EXPECT_EQ(shape_range[0].first, 1);
  EXPECT_EQ(shape_range[1].first, 1);
  EXPECT_EQ(shape_range[2].first, 1);
  EXPECT_EQ(shape_range[0].second, 300);
  EXPECT_EQ(shape_range[1].second, 100);
  EXPECT_EQ(shape_range[2].second, 3);
}

TEST_F(DecodeImageV3, decode_image_v3_infer_shape_fault_0) {
  ge::op::DecodeImage op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {};
  auto tensor_desc = create_desc_shape_range({},
                                             ge::DT_FLOAT, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("contents", tensor_desc);
  op.SetAttr("expand_animations", false);
  op.SetAttr("channels", 3);
  op.SetAttr("dtype", ge::DT_UINT8);
  std::string maxShape("300,a100,3");
  op.SetAttr(ge::ATTR_NAME_OP_MAX_SHAPE, maxShape);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  ge::TensorDesc imageDesc = op.GetOutputDesc(0);
  ge::Shape imageShage = imageDesc.GetShape();
  EXPECT_EQ(imageShage.GetDimNum(), 3);
  EXPECT_EQ(imageShage.GetDim(0), -1);
  EXPECT_EQ(imageShage.GetDim(1), -1);
  EXPECT_EQ(imageShage.GetDim(2), 3);
  
  EXPECT_EQ(imageDesc.GetShapeRange(shape_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(shape_range.size(), 0);
}

TEST_F(DecodeImageV3, decode_image_v3_infer_shape_fault_1) {
  ge::op::DecodeImage op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {};
  auto tensor_desc = create_desc_shape_range({},
                                             ge::DT_FLOAT, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("contents", tensor_desc);
  op.SetAttr("channels", 2);
  op.SetAttr("expand_animations", false);
  op.SetAttr("dtype", ge::DT_UINT8);
  std::string maxShape("300,a100,3");
  op.SetAttr(ge::ATTR_NAME_OP_MAX_SHAPE, maxShape);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}