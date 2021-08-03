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
