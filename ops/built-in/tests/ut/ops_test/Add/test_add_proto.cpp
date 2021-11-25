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
 * @file test_add_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"
#include "graph/debug/ge_attr_define.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"

class add : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "add SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "add TearDown" << std::endl;
  }
};

TEST_F(add, add_infer_shape_fp16) {
  ge::op::Add op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 100}};
  auto tensor_desc = create_desc_shape_range({-1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {64},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x1", tensor_desc);
  op.UpdateInputDesc("x2", tensor_desc);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {2, 100},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(add, add_data_slice_infer1) {
  ge::op::Add op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 100}, {1, 100}, {1, 100},{1, 100}, {1, 100}};
  auto tensor_desc = create_desc_shape_range({16, 16, 16, 16, 16},
                                             ge::DT_FLOAT16,
                                             ge::FORMAT_NC1HWC0,
                                             {16, 256, 16, 16},
                                             ge::FORMAT_NCHW,
                                             shape_range);
  op.UpdateInputDesc("x1", tensor_desc);
  op.UpdateInputDesc("x2", tensor_desc);
  op.UpdateOutputDesc("y", tensor_desc);

  std::vector<std::vector<int64_t>> y_data_slice = {{0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x1 = op_desc->MutableInputDesc("x1");
  auto tensor_desc_x2 = op_desc->MutableInputDesc("x2");
  std::vector<std::vector<int64_t>> x1_data_slice;
  std::vector<std::vector<int64_t>> x2_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice);
  ge::AttrUtils::GetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice);
  std::vector<std::vector<int64_t>> excepted_x1_data_slice = {{0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}};
  std::vector<std::vector<int64_t>> excepted_x2_data_slice = {{0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}};
  EXPECT_EQ(excepted_x1_data_slice, x1_data_slice);
  EXPECT_EQ(excepted_x2_data_slice, x2_data_slice);
}

TEST_F(add, add_data_slice_infer2) {
  ge::op::Add op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 100}, {1, 100}, {1, 100}, {1, 100}, {1, 100}, {1, 100}};
  auto tensor_desc = create_desc_shape_range({16, 16, 16, 16, 16, 16},
                                             ge::DT_FLOAT16,
                                             ge::FORMAT_NDC1HWC0,
                                             {16, 16, 16, 16, 256},
                                             ge::FORMAT_NDHWC,
                                             shape_range);
  op.UpdateInputDesc("x1", tensor_desc);
  op.UpdateInputDesc("x2", tensor_desc);
  op.UpdateOutputDesc("y", tensor_desc);

  std::vector<std::vector<int64_t>> y_data_slice = {{0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x1 = op_desc->MutableInputDesc("x1");
  auto tensor_desc_x2 = op_desc->MutableInputDesc("x2");
  std::vector<std::vector<int64_t>> x1_data_slice;
  std::vector<std::vector<int64_t>> x2_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice);
  ge::AttrUtils::GetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice);
  std::vector<std::vector<int64_t>> excepted_x1_data_slice = {{0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}};
  std::vector<std::vector<int64_t>> excepted_x2_data_slice = {{0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}};
  EXPECT_EQ(excepted_x1_data_slice, x1_data_slice);
  EXPECT_EQ(excepted_x2_data_slice, x2_data_slice);
}

TEST_F(add, add_data_slice_infer4) {
  ge::op::Add op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 100}, {1, 100}, {1, 100}, {1, 100}, {1, 100}, {1, 100}};
  auto tensor_desc = create_desc_shape_range({16, 16, 16, 16, 16, 16},
                                             ge::DT_FLOAT16,
                                             ge::FORMAT_FRACTAL_Z,
                                             {256, 16, 16, 256},
                                             ge::FORMAT_NHWC,
                                             shape_range);
  op.UpdateInputDesc("x1", tensor_desc);
  op.UpdateInputDesc("x2", tensor_desc);
  op.UpdateOutputDesc("y", tensor_desc);

  std::vector<std::vector<int64_t>> y_data_slice = {{0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x1 = op_desc->MutableInputDesc("x1");
  auto tensor_desc_x2 = op_desc->MutableInputDesc("x2");
  std::vector<std::vector<int64_t>> x1_data_slice;
  std::vector<std::vector<int64_t>> x2_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice);
  ge::AttrUtils::GetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice);
  std::vector<std::vector<int64_t>> excepted_x1_data_slice = {{0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}};
  std::vector<std::vector<int64_t>> excepted_x2_data_slice = {{0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}};
  EXPECT_EQ(excepted_x1_data_slice, x1_data_slice);
  EXPECT_EQ(excepted_x2_data_slice, x2_data_slice);
}

TEST_F(add, add_data_slice_infer5) {
  ge::op::Add op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 100}, {1, 100}, {1, 100}, {1, 100}, {1, 100}, {1, 100}};
  auto tensor_desc = create_desc_shape_range({16, 16, 16, 16, 16, 16},
                                             ge::DT_FLOAT16,
                                             ge::FORMAT_FRACTAL_Z,
                                             {256, 256, 16, 16},
                                             ge::FORMAT_NCHW,
                                             shape_range);
  op.UpdateInputDesc("x1", tensor_desc);
  op.UpdateInputDesc("x2", tensor_desc);
  op.UpdateOutputDesc("y", tensor_desc);

  std::vector<std::vector<int64_t>> y_data_slice = {{0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x1 = op_desc->MutableInputDesc("x1");
  auto tensor_desc_x2 = op_desc->MutableInputDesc("x2");
  std::vector<std::vector<int64_t>> x1_data_slice;
  std::vector<std::vector<int64_t>> x2_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice);
  ge::AttrUtils::GetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice);
  std::vector<std::vector<int64_t>> excepted_x1_data_slice = {{0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}};
  std::vector<std::vector<int64_t>> excepted_x2_data_slice = {{0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}};
  EXPECT_EQ(excepted_x1_data_slice, x1_data_slice);
  EXPECT_EQ(excepted_x2_data_slice, x2_data_slice);
}

TEST_F(add, add_data_slice_infer6) {
  ge::op::Add op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 100}, {1, 100}, {1, 100}, {1, 100}, {1, 100}, {1, 100},
                                                         {1, 100}};
  auto tensor_desc = create_desc_shape_range({16, 16, 16, 16, 16, 16, 16},
                                             ge::DT_FLOAT16,
                                             ge::FORMAT_FRACTAL_Z_3D,
                                             {256, 16, 16, 16, 256},
                                             ge::FORMAT_NDHWC,
                                             shape_range);
  op.UpdateInputDesc("x1", tensor_desc);
  op.UpdateInputDesc("x2", tensor_desc);
  op.UpdateOutputDesc("y", tensor_desc);

  std::vector<std::vector<int64_t>> y_data_slice = {{0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x1 = op_desc->MutableInputDesc("x1");
  auto tensor_desc_x2 = op_desc->MutableInputDesc("x2");
  std::vector<std::vector<int64_t>> x1_data_slice;
  std::vector<std::vector<int64_t>> x2_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice);
  ge::AttrUtils::GetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice);
  std::vector<std::vector<int64_t>> excepted_x1_data_slice = {{0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}};
  std::vector<std::vector<int64_t>> excepted_x2_data_slice = {{0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}};
  EXPECT_EQ(excepted_x1_data_slice, x1_data_slice);
  EXPECT_EQ(excepted_x2_data_slice, x2_data_slice);
}

TEST_F(add, add_data_slice_infer7) {
  ge::op::Add op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 100}, {1, 100}, {1, 100}, {1, 100}, {1, 100}, {1, 100}};
  auto tensor_desc = create_desc_shape_range({16, 16, 16, 16, 16, 16},
                                             ge::DT_FLOAT16,
                                             ge::FORMAT_FRACTAL_NZ,
                                             {16, 16, 256, 256},
                                             ge::FORMAT_ND,
                                             shape_range);
  op.UpdateInputDesc("x1", tensor_desc);
  op.UpdateInputDesc("x2", tensor_desc);
  op.UpdateOutputDesc("y", tensor_desc);

  std::vector<std::vector<int64_t>> y_data_slice = {{0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x1 = op_desc->MutableInputDesc("x1");
  auto tensor_desc_x2 = op_desc->MutableInputDesc("x2");
  std::vector<std::vector<int64_t>> x1_data_slice;
  std::vector<std::vector<int64_t>> x2_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice);
  ge::AttrUtils::GetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice);
  std::vector<std::vector<int64_t>> excepted_x1_data_slice = {{0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}};
  std::vector<std::vector<int64_t>> excepted_x2_data_slice = {{0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}};
  EXPECT_EQ(excepted_x1_data_slice, x1_data_slice);
  EXPECT_EQ(excepted_x2_data_slice, x2_data_slice);
}

TEST_F(add, add_data_slice_infer8) {
  ge::op::Add op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 100}, {1, 100}, {1, 100}, {1, 100}, {1, 100}, {1, 100}};
  auto tensor_desc1 = create_desc_shape_range({16, 16, 16, 16, 16, 16},
                                             ge::DT_FLOAT16,
                                             ge::FORMAT_FRACTAL_NZ,
                                             {16, 16, 256, 256},
                                             ge::FORMAT_ND,
                                             shape_range);

  auto tensor_desc2 = create_desc_shape_range({256},
                                             ge::DT_FLOAT16,
                                             ge::FORMAT_ND,
                                             {256},
                                             ge::FORMAT_ND,
                                             {{1, 100}});

  op.UpdateInputDesc("x1", tensor_desc1);
  op.UpdateInputDesc("x2", tensor_desc2);
  op.UpdateOutputDesc("y", tensor_desc1);

  std::vector<std::vector<int64_t>> y_data_slice = {{0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x1 = op_desc->MutableInputDesc("x1");
  auto tensor_desc_x2 = op_desc->MutableInputDesc("x2");
  std::vector<std::vector<int64_t>> x1_data_slice;
  std::vector<std::vector<int64_t>> x2_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice);
  ge::AttrUtils::GetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice);
  std::vector<std::vector<int64_t>> excepted_x1_data_slice = {{0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}};
  std::vector<std::vector<int64_t>> excepted_x2_data_slice = {{0, 64}};
  EXPECT_EQ(excepted_x1_data_slice, x1_data_slice);
  EXPECT_EQ(excepted_x2_data_slice, x2_data_slice);
}

TEST_F(add, add_data_slice_infer9) {
  ge::op::Add op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 100}, {1, 100}, {1, 100}, {1, 100}, {1, 100}, {1, 100}};
  auto tensor_desc1 = create_desc_shape_range({16, 16, 16, 16, 16, 16},
                                             ge::DT_FLOAT16,
                                             ge::FORMAT_FRACTAL_NZ,
                                             {16, 16, 256, 256},
                                             ge::FORMAT_ND,
                                             shape_range);

  auto tensor_desc2 = create_desc_shape_range({256, 256},
                                             ge::DT_FLOAT16,
                                             ge::FORMAT_ND,
                                             {256, 256},
                                             ge::FORMAT_ND,
                                             {{1, 100}, {1, 100}});

  op.UpdateInputDesc("x1", tensor_desc1);
  op.UpdateInputDesc("x2", tensor_desc2);
  op.UpdateOutputDesc("y", tensor_desc1);

  std::vector<std::vector<int64_t>> y_data_slice = {{0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x1 = op_desc->MutableInputDesc("x1");
  auto tensor_desc_x2 = op_desc->MutableInputDesc("x2");
  std::vector<std::vector<int64_t>> x1_data_slice;
  std::vector<std::vector<int64_t>> x2_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice);
  ge::AttrUtils::GetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice);
  std::vector<std::vector<int64_t>> excepted_x1_data_slice = {{}, {}, {}, {}, {}, {}};
  std::vector<std::vector<int64_t>> excepted_x2_data_slice = {{}, {}};
  EXPECT_EQ(excepted_x1_data_slice, x1_data_slice);
  EXPECT_EQ(excepted_x2_data_slice, x2_data_slice);
}

TEST_F(add, add_data_slice_infer10) {
  ge::op::Add op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 100}};
  auto tensor_desc_x1 = create_desc_shape_range({-1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {64},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc_x2 = create_desc_shape_range({-1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {64},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x1", tensor_desc_x1);
  op.UpdateInputDesc("x2", tensor_desc_x2);
  
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(add, add_data_slice_infer11) {
  ge::op::Add op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 100}};
  auto tensor_desc_x1 = create_desc_shape_range({-1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {64},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc_x2 = create_desc_shape_range({-1},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {64},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x1", tensor_desc_x1);
  op.UpdateInputDesc("x2", tensor_desc_x2);
  
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(add, add_data_slice_infer12) {
  ge::op::Add op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 100}, {1, 100}, {1, 100},{1, 100}, {1, 100}};
  auto tensor_desc = create_desc_shape_range({16, 16, 16, 16, 16},
                                             ge::DT_FLOAT16,
                                             ge::FORMAT_NC1HWC0,
                                             {16, 256, 16, 16},
                                             ge::FORMAT_NCHW,
                                             shape_range);
  std::vector<std::pair<int64_t, int64_t>> shape_range_1 = {{1, 1}};
  auto tensor_desc_1 = create_desc_shape_range({1},
                                             ge::DT_FLOAT16,
                                             ge::FORMAT_ND,
                                             {1},
                                             ge::FORMAT_ND,
                                             shape_range_1);
  op.UpdateInputDesc("x1", tensor_desc);
  op.UpdateInputDesc("x2", tensor_desc_1);
  op.UpdateOutputDesc("y", tensor_desc);

  std::vector<std::vector<int64_t>> y_data_slice = {{0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x1 = op_desc->MutableInputDesc("x1");
  auto tensor_desc_x2 = op_desc->MutableInputDesc("x2");
  std::vector<std::vector<int64_t>> x1_data_slice;
  std::vector<std::vector<int64_t>> x2_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice);
  ge::AttrUtils::GetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice);
  std::vector<std::vector<int64_t>> excepted_x1_data_slice = {{0, 4}, {0, 4}, {0, 4}, {0, 4}, {0, 4}};
  std::vector<std::vector<int64_t>> excepted_x2_data_slice = {};
  EXPECT_EQ(excepted_x1_data_slice, x1_data_slice);
  EXPECT_EQ(excepted_x2_data_slice, x2_data_slice);
}
