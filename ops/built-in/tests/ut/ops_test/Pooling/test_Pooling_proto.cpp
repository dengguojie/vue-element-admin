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
 * @file test_aipp_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "graph/debug/ge_attr_define.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include "nn_pooling_ops.h"
#include "graph/common_error_codes.h"

class Pooling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Pooling Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Pooling Proto Test TearDown" << std::endl;
  }
};


TEST_F(Pooling, Pooling_data_slice_infer1) {
  ge::op::Pooling op;

  auto tensor_desc = create_desc_with_ori({4,1,224,224,16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {4,3,224,224}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);

  auto output_tensor_desc_temp = create_desc_with_ori({4,1,224,224,16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {4,3,224,224}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("y", output_tensor_desc_temp);

  std::vector<std::vector<int64_t>> output_data_slice ={{0,1}, {}, {}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(output_tensor_desc, ge::ATTR_NAME_DATA_SLICE, output_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x = op_desc->MutableInputDesc("x");
  std::vector<std::vector<int64_t>> x_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);

  std::vector<std::vector<int64_t>> expected_x_data_slice = {{0,1}, {}, {}, {}, {}};
  EXPECT_EQ(expected_x_data_slice, x_data_slice);
}

TEST_F(Pooling, Pooling_data_slice_infer2) {
  ge::op::Pooling op;

  // set pooling attr
  std::vector<int32_t> pad_list = {1, 0, 0, 1};
  op.SetAttr("pad", pad_list);
  std::vector<int64_t> window = {3, 3};
  op.SetAttr("window", window);
  std::vector<int64_t> stride = {2, 2};
  op.SetAttr("stride", stride);
  op.SetAttr("global_pooling", false);

  auto tensor_desc = create_desc_with_ori({1,1,224,224,16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1,3,224,224}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);

  auto output_tensor_desc_temp = create_desc_with_ori({1,1,115,115,16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1,3,115,115}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("y", output_tensor_desc_temp);

  std::vector<std::vector<int64_t>> output_data_slice ={{}, {}, {10, 20}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(output_tensor_desc, ge::ATTR_NAME_DATA_SLICE, output_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x = op_desc->MutableInputDesc("x");
  std::vector<std::vector<int64_t>> x_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);

  std::vector<std::vector<int64_t>> expected_x_data_slice = {{}, {}, {19, 41}, {}, {}};
  EXPECT_EQ(expected_x_data_slice, x_data_slice);

  std::vector<int32_t> new_pad_list;
  EXPECT_EQ(op.GetAttr("pad", new_pad_list), ge::GRAPH_SUCCESS);
  std::vector<int32_t> expected_new_pad_list = {0, 0, 0, 1};
  EXPECT_EQ(expected_new_pad_list, new_pad_list);
}

TEST_F(Pooling, Pooling_data_slice_infer3) {
  ge::op::Pooling op;

  // set pooling attr
  std::vector<int32_t> pad_list = {1, 0, 0, 1};
  op.SetAttr("pad", pad_list);
  std::vector<int64_t> window = {3, 3};
  op.SetAttr("window", window);
  std::vector<int64_t> stride = {2, 2};
  op.SetAttr("stride", stride);
  op.SetAttr("global_pooling", false);

  auto tensor_desc = create_desc_with_ori({1,1,224,224,16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1,3,224,224}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);

  auto output_tensor_desc_temp = create_desc_with_ori({1,1,115,115,16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1,3,115,115}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("y", output_tensor_desc_temp);

  std::vector<std::vector<int64_t>> output_data_slice ={{}, {}, {0, 20}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(output_tensor_desc, ge::ATTR_NAME_DATA_SLICE, output_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x = op_desc->MutableInputDesc("x");
  std::vector<std::vector<int64_t>> x_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);

  std::vector<std::vector<int64_t>> expected_x_data_slice = {{}, {}, {0, 41}, {}, {}};
  EXPECT_EQ(expected_x_data_slice, x_data_slice);

  std::vector<int32_t> new_pad_list;
  EXPECT_EQ(op.GetAttr("pad", new_pad_list), ge::GRAPH_SUCCESS);
  std::vector<int32_t> expected_new_pad_list = {1, 0, 0, 1};
  EXPECT_EQ(expected_new_pad_list, new_pad_list);
}

TEST_F(Pooling, Pooling_data_slice_infer4) {
  ge::op::Pooling op;

  // set pooling attr
  std::vector<int32_t> pad_list = {1, 0, 0, 1};
  op.SetAttr("pad", pad_list);
  std::vector<int64_t> window = {3, 3};
  op.SetAttr("window", window);
  std::vector<int64_t> stride = {2, 2};
  op.SetAttr("stride", stride);
  op.SetAttr("global_pooling", false);

  auto tensor_desc = create_desc_with_ori({1,1,224,224,16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1,3,224,224}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);

  auto output_tensor_desc_temp = create_desc_with_ori({1,1,115,115,16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1,3,115,115}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("y", output_tensor_desc_temp);

  std::vector<std::vector<int64_t>> output_data_slice ={{}, {}, {110, 115}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(output_tensor_desc, ge::ATTR_NAME_DATA_SLICE, output_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x = op_desc->MutableInputDesc("x");
  std::vector<std::vector<int64_t>> x_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);

  std::vector<std::vector<int64_t>> expected_x_data_slice = {{}, {}, {219, 223}, {}, {}};
  EXPECT_EQ(expected_x_data_slice, x_data_slice);

  std::vector<int32_t> new_pad_list;
  EXPECT_EQ(op.GetAttr("pad", new_pad_list), ge::GRAPH_SUCCESS);
  std::vector<int32_t> expected_new_pad_list = {0, 0, 0, 1};
  EXPECT_EQ(expected_new_pad_list, new_pad_list);
}

TEST_F(Pooling, Pooling_data_slice_infer5) {
  ge::op::Pooling op;

  // set pooling attr
  std::vector<int32_t> pad_list = {1, 0, 0, 1};
  op.SetAttr("pad", pad_list);
  std::vector<int64_t> window = {3, 3};
  op.SetAttr("window", window);
  std::vector<int64_t> stride = {2, 2};
  op.SetAttr("stride", stride);
  op.SetAttr("global_pooling", false);

  auto tensor_desc = create_desc_with_ori({1,3,224,224}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1,3,224,224}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);

  auto output_tensor_desc_temp = create_desc_with_ori({1,1,115,115,16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1,3,115,115}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("y", output_tensor_desc_temp);

  std::vector<std::vector<int64_t>> output_data_slice ={{}, {}, {110, 115}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(output_tensor_desc, ge::ATTR_NAME_DATA_SLICE, output_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(Pooling, Pooling_data_slice_infer6) {
  ge::op::Pooling op;

  // set pooling attr
  std::vector<int32_t> pad_list = {0, 0, 1, 0, 0, 1};
  op.SetAttr("pad", pad_list);
  std::vector<int64_t> window = {3, 3};
  op.SetAttr("window", window);
  std::vector<int64_t> stride = {2, 2};
  op.SetAttr("stride", stride);
  op.SetAttr("global_pooling", false);

  auto tensor_desc = create_desc_with_ori({1,1,224,224,16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1,3,224,224}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);

  auto output_tensor_desc_temp = create_desc_with_ori({1,1,115,115,16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1,3,115,115}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("y", output_tensor_desc_temp);

  std::vector<std::vector<int64_t>> output_data_slice ={{}, {}, {110, 115}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(output_tensor_desc, ge::ATTR_NAME_DATA_SLICE, output_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::NOT_SUPPORT_SLICE);
}

TEST_F(Pooling, Pooling_data_slice_infer7) {
  ge::op::Pooling op;

  // set pooling attr
  std::vector<int32_t> pad_list = {1, 0, 0, 1};
  op.SetAttr("pad", pad_list);
  std::vector<int64_t> window = {3, 3};
  op.SetAttr("window", window);
  std::vector<int64_t> stride = {2, 2};
  op.SetAttr("stride", stride);
  op.SetAttr("global_pooling", true);

  auto tensor_desc = create_desc_with_ori({1,1,224,224,16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1,3,224,224}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);

  auto output_tensor_desc_temp = create_desc_with_ori({1,1,115,115,16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1,3,115,115}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("y", output_tensor_desc_temp);

  std::vector<std::vector<int64_t>> output_data_slice ={{}, {}, {110, 115}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(output_tensor_desc, ge::ATTR_NAME_DATA_SLICE, output_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::NOT_SUPPORT_SLICE);
}

TEST_F(Pooling, InfershapePooling_001) {
  ge::op::Pooling op;

  auto tensor_desc = create_desc_with_ori({1, 1, 224, 224, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 3, 224, 224},
                                          ge::FORMAT_NC1HWC0);
  op.UpdateInputDesc("x", tensor_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Pooling, InfershapePooling_002) {
  ge::op::Pooling op;
  auto tensor_desc =
      create_desc_with_ori({1, 1, 224, 224, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 3, 224, 224}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("global_pooling", true);
  std::vector<int32_t> pad_list = {0, 1, 1, 0};
  op.SetAttr("pad", pad_list);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Pooling, InfershapePooling_003) {
  ge::op::Pooling op;
  auto tensor_desc =
      create_desc_with_ori({1, 1, 224, 224, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 3, 224, 224}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("ceil_mode", -1);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Pooling, InfershapePooling_004) {
  ge::op::Pooling op;
  std::vector<int64_t> pad_list = {1, 1, 1, 1};
  op.SetAttr("pad", pad_list);
  auto tensor_desc =
      create_desc_with_ori({1, 1, 224, 224, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 3, 224, 224}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int64_t> window = {100, 1};
  op.SetAttr("window", window);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Pooling, InfershapePooling_005) {
  ge::op::Pooling op;
  std::vector<int64_t> pad_list = {1, 1, 1, 1};
  op.SetAttr("pad", pad_list);
  op.SetAttr("ceil_mode", 1);
  auto tensor_desc =
      create_desc_with_ori({1, 224, 224, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 224, 224, 1}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int64_t> window = {1, 100};
  op.SetAttr("window", window);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Pooling, InfershapePooling_006) {
  ge::op::Pooling op;
  auto tensor_desc =
      create_desc_with_ori({1, 1, 224, 224, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 3, 224, 224}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateOutputDesc("y", create_desc_with_ori({}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {}, ge::FORMAT_NCHW));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  std::vector<int64_t> output_data_shape_nchw = {1, 1, 224, 224};
  auto outdesc = op.GetOutputDesc("y");
  EXPECT_EQ(outdesc.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(outdesc.GetShape().GetDims(), output_data_shape_nchw);
}

TEST_F(Pooling, InfershapePooling_007) {
  ge::op::Pooling op;
  auto tensor_desc =
      create_desc_with_ori({1, 1, 224, 224}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 3, 224, 224}, ge::FORMAT_NHWC);
  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateOutputDesc("y", create_desc_with_ori({}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {}, ge::FORMAT_NHWC));

  auto statue = op.VerifyAllAttr(true);
  EXPECT_EQ(statue, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  std::vector<int64_t> output_data_shape_nchw = {1, 1, 224, 224};
  auto outdesc = op.GetOutputDesc("y");
  EXPECT_EQ(outdesc.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(outdesc.GetShape().GetDims(), output_data_shape_nchw);
}

TEST_F(Pooling, VerifyePooling_001) {
  ge::op::Pooling op;
  auto tensor_desc =
      create_desc_with_ori({1, 1, 224, 224, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 3, 224, 224}, ge::FORMAT_NHWC);
  op.UpdateInputDesc("x", tensor_desc);

  auto statue = op.VerifyAllAttr(true);
  EXPECT_EQ(statue, ge::GRAPH_FAILED);
}

TEST_F(Pooling, VerifyePooling_002) {
  ge::op::Pooling op;
  auto tensor_desc =
      create_desc_with_ori({1, 1, 224, 224}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 1, 224, 224}, ge::FORMAT_NHWC);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int64_t> window = {3, 3,3};
  op.SetAttr("window", window);

  auto statue = op.VerifyAllAttr(true);
  EXPECT_EQ(statue, ge::GRAPH_FAILED);
}

TEST_F(Pooling, VerifyePooling_003) {
  ge::op::Pooling op;
  auto tensor_desc =
      create_desc_with_ori({1, 1, 224, 224}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 1, 224, 224}, ge::FORMAT_NHWC);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int64_t> stride = {2, 2,2};
  op.SetAttr("stride", stride);

  auto statue = op.VerifyAllAttr(true);
  EXPECT_EQ(statue, ge::GRAPH_FAILED);
}

TEST_F(Pooling, VerifyePooling_004) {
  ge::op::Pooling op;
  auto tensor_desc =
      create_desc_with_ori({1, 1, 224, 224}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 1, 224, 224}, ge::FORMAT_NHWC);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int32_t> pad_list = {1, 0, 0};
  op.SetAttr("pad", pad_list);

  auto statue = op.VerifyAllAttr(true);
  EXPECT_EQ(statue, ge::GRAPH_FAILED);
}

TEST_F(Pooling, InferformatPooling_001) {
  ge::op::Pooling op;
  auto tensor_desc =
      create_desc_with_ori({1, 1, 224, 224}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 3, 224, 224}, ge::FORMAT_NHWC);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("data_format",true);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_EQ(op_desc->CallInferFormatFunc(op), ge::GRAPH_FAILED);
}

TEST_F(Pooling, InferformatPooling_002) {
  ge::op::Pooling op;
  auto tensor_desc =
      create_desc_with_ori({1, 1, 224, 224}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 3, 224, 224}, ge::FORMAT_NHWC);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("data_format","ND");

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_EQ(op_desc->CallInferFormatFunc(op), ge::GRAPH_FAILED);
}

TEST_F(Pooling, InferformatPooling_003) {
  ge::op::Pooling op;
  auto tensor_desc =
      create_desc_with_ori({1, 1, 224, 224}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 3, 224, 224}, ge::FORMAT_ND);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("data_format","NHWC");

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_EQ(op_desc->CallInferFormatFunc(op), ge::GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->MutableOutputDesc(0)->GetOriginFormat(), ge::FORMAT_NHWC);
}

TEST_F(Pooling, InferformatPooling_004) {
  ge::op::Pooling op;
  auto tensor_desc =
      create_desc_with_ori({1, 1, 224, 224}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 3, 224, 224}, ge::FORMAT_ND);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("data_format","NCHW");

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_EQ(op_desc->CallInferFormatFunc(op), ge::GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->MutableOutputDesc(0)->GetOriginFormat(), ge::FORMAT_NCHW);
}