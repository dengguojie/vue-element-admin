/**
 * Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the
 License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_trans_data_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "graph/common_error_codes.h"
#include "op_proto_test_util.h"
#include "common/utils/ut_profiling_reg.h"
#include "transformation_ops.h"
#include "common/utils/ut_op_common.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_desc_utils.h"

class trans_data : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "trans_data SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "trans_data TearDown" << std::endl;
  }
};

TEST_F(trans_data, trans_data_infer_shape_fp16) {
  ge::op::TransData op;
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  auto tensor_desc =
      create_desc_shape_range({16, 16, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {64}, ge::FORMAT_NCHW, shape_range);
  auto tensor_desc_out = create_desc_shape_range({}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {}, ge::FORMAT_NCHW, shape_range);
  op.UpdateInputDesc("src", tensor_desc);
  op.UpdateOutputDesc("dst", tensor_desc_out);

  auto ret = op.InferShapeAndType();

  // test performance start
  PROFILING_TEST(op.InferShapeAndType, (), 1000, 10);
  // test performance end

  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("dst");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {16, 16, 16, 16};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(trans_data, trans_data_infer_shape_with_diff_format) {
  ge::op::TransData op;
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  auto tensor_desc =
      create_desc_shape_range({16, 16, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {64}, ge::FORMAT_NCHW, shape_range);
  auto tensor_desc_out = create_desc_shape_range({}, ge::DT_FLOAT16, ge::FORMAT_ND, {}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("src", tensor_desc);
  op.UpdateOutputDesc("dst", tensor_desc_out);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("dst");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(trans_data, trans_data_infer_rt2) {
  ge::op::TransData op;
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  auto tensor_desc =
      create_desc_shape_range({16, 16, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {64}, ge::FORMAT_NCHW, shape_range);
  auto tensor_desc_out = create_desc_shape_range({}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {}, ge::FORMAT_NCHW, shape_range);
  op.UpdateInputDesc("src", tensor_desc);
  op.UpdateOutputDesc("dst", tensor_desc_out);
  std::vector<int64_t> expected_output_shape = {16, 16, 16, 16};
  CommonInferShapeOperator(op, {}, {expected_output_shape});
  EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
  auto output0_desc = op.GetOutputDesc(0);
  EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(trans_data, data_slice_infer_nd_5hd_1) {
  ge::op::TransData op;
  std::vector<std::pair<int64_t, int64_t>> shape_src_range = {{4, 4}, {8, 8}, {12, 12}, {24, 24}};
  std::vector<std::pair<int64_t, int64_t>> shape_dst_range = {{4, 4}, {2, 2}, {8, 8}, {12, 12}, {16, 16}};
  auto src_desc = create_desc_shape_range({4, 8, 12, 24},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_NCHW,
                                         {4, 8, 12, 24},
                                         ge::FORMAT_NCHW,
                                         shape_src_range);
  auto dst_desc = create_desc_shape_range({4, 2, 8, 12, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_NC1HWC0,
                                         {4, 2, 8, 12, 16},
                                         ge::FORMAT_NC1HWC0,
                                         shape_dst_range);
  op.UpdateInputDesc("src", src_desc);
  op.SetAttr("src_format", "NCHW");
  op.SetAttr("dst_format", "NC1HWC0");
  op.UpdateOutputDesc("dst", dst_desc);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc("dst");
  std::vector<std::vector<int64_t>> dst_data_slice = {{2, 3}, {}, {}, {}, {}};
  ge::AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, dst_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_src = op_desc->MutableInputDesc("src");
  std::vector<std::vector<int64_t>> src_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_src, ge::ATTR_NAME_DATA_SLICE, src_data_slice);
  std::vector<std::vector<int64_t>> excepted_src_data_slice = {{2, 3}, {0, 7}, {0, 11}, {0, 23}};
  EXPECT_EQ(excepted_src_data_slice, src_data_slice);
}

TEST_F(trans_data, data_slice_infer_5hd_nd_2) {
  ge::op::TransData op;
  std::vector<std::pair<int64_t, int64_t>> shape_src_range = {{4, 4}, {2, 2}, {8, 8}, {12, 12}, {16, 16}};
  std::vector<std::pair<int64_t, int64_t>> shape_dst_range = {{4, 4}, {8, 8}, {12, 12}, {24, 24}};
  auto src_desc = create_desc_shape_range({4, 2, 8, 12, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_NC1HWC0,
                                         {4, 2, 8, 12, 16},
                                         ge::FORMAT_NC1HWC0,
                                         shape_src_range);
  auto dst_desc = create_desc_shape_range({4, 8, 12, 24},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_NCHW,
                                         {4, 8, 12, 24},
                                         ge::FORMAT_NCHW,
                                         shape_dst_range);

  op.UpdateInputDesc("src", src_desc);
  op.SetAttr("src_format", "NC1HWC0");
  op.SetAttr("dst_format", "NCHW");
  op.UpdateOutputDesc("dst", dst_desc);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc("dst");
  std::vector<std::vector<int64_t>> dst_data_slice = {{2, 3}, {}, {}, {}};
  ge::AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, dst_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_src = op_desc->MutableInputDesc("src");
  std::vector<std::vector<int64_t>> src_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_src, ge::ATTR_NAME_DATA_SLICE, src_data_slice);
  std::vector<std::vector<int64_t>> excepted_src_data_slice = {{2, 3}, {0, 1}, {0, 7}, {0, 11}, {0, 15}};
  EXPECT_EQ(excepted_src_data_slice, src_data_slice);
}

TEST_F(trans_data, data_slice_infer_nd_nz_3) {
  ge::op::TransData op;
  std::vector<std::pair<int64_t, int64_t>> shape_src_range = {{37, 37}, {57, 57}};
  std::vector<std::pair<int64_t, int64_t>> shape_dst_range = {{4, 4}, {3, 3}, {16, 16}, {16, 16}};
  auto src_desc = create_desc_shape_range({37, 57},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_NCHW,
                                         {37, 57},
                                         ge::FORMAT_NCHW,
                                         shape_src_range);
  auto dst_desc = create_desc_shape_range({4, 3, 16, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_FRACTAL_NZ,
                                         {4, 3, 16, 16},
                                         ge::FORMAT_FRACTAL_NZ,
                                         shape_dst_range);
  op.UpdateInputDesc("src", src_desc);
  op.SetAttr("src_format", "NCHW");
  op.SetAttr("dst_format", "FRACTAL_NZ");
  op.UpdateOutputDesc("dst", dst_desc);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc("dst");
  std::vector<std::vector<int64_t>> dst_data_slice = {{}, {2, 2}, {}, {}};
  ge::AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, dst_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_src = op_desc->MutableInputDesc("src");
  std::vector<std::vector<int64_t>> src_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_src, ge::ATTR_NAME_DATA_SLICE, src_data_slice);
  std::vector<std::vector<int64_t>> excepted_src_data_slice = {{32, 36}, {0, 56}};
  EXPECT_EQ(excepted_src_data_slice, src_data_slice);
}

TEST_F(trans_data, data_slice_infer_nd_nz_4) {
  ge::op::TransData op;
  std::vector<std::pair<int64_t, int64_t>> shape_src_range = {{37, 37}, {57, 57}};
  std::vector<std::pair<int64_t, int64_t>> shape_dst_range = {{4, 4}, {3, 3}, {16, 16}, {16, 16}};
  auto src_desc = create_desc_shape_range({37, 57},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_NCHW,
                                         {37, 57},
                                         ge::FORMAT_NCHW,
                                         shape_src_range);
  auto dst_desc = create_desc_shape_range({4, 3, 16, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_FRACTAL_NZ,
                                         {4, 3, 16, 16},
                                         ge::FORMAT_FRACTAL_NZ,
                                         shape_dst_range);
  op.UpdateInputDesc("src", src_desc);
  op.SetAttr("src_format", "NCHW");
  op.SetAttr("dst_format", "FRACTAL_NZ");
  op.UpdateOutputDesc("dst", dst_desc);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc("dst");
  std::vector<std::vector<int64_t>> dst_data_slice = {{}, {1, 2}, {}, {}};
  ge::AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, dst_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_src = op_desc->MutableInputDesc("src");
  std::vector<std::vector<int64_t>> src_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_src, ge::ATTR_NAME_DATA_SLICE, src_data_slice);
  std::vector<std::vector<int64_t>> excepted_src_data_slice = {{16, 36}, {0, 56}};
  EXPECT_EQ(excepted_src_data_slice, src_data_slice);
}

TEST_F(trans_data, data_slice_infer_nd_nz_5) {
  ge::op::TransData op;
  std::vector<std::pair<int64_t, int64_t>> shape_src_range = {{7, 7}, {37, 37}, {57, 57}};
  std::vector<std::pair<int64_t, int64_t>> shape_dst_range = {{7, 7}, {4, 4}, {3, 3}, {16, 16}, {16, 16}};
  auto src_desc = create_desc_shape_range({7, 37, 57},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_NCHW,
                                         {7, 37, 57},
                                         ge::FORMAT_NCHW,
                                         shape_src_range);
  auto dst_desc = create_desc_shape_range({7, 4, 3, 16, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_FRACTAL_NZ,
                                         {7, 4, 3, 16, 16},
                                         ge::FORMAT_FRACTAL_NZ,
                                         shape_dst_range);
  op.UpdateInputDesc("src", src_desc);
  op.SetAttr("src_format", "NCHW");
  op.SetAttr("dst_format", "FRACTAL_NZ");
  op.UpdateOutputDesc("dst", dst_desc);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc("dst");
  std::vector<std::vector<int64_t>> dst_data_slice = {{0, 5}, {}, {}, {}, {}};
  ge::AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, dst_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_src = op_desc->MutableInputDesc("src");
  std::vector<std::vector<int64_t>> src_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_src, ge::ATTR_NAME_DATA_SLICE, src_data_slice);
  std::vector<std::vector<int64_t>> excepted_src_data_slice = {{0, 5}, {0, 36}, {0, 56}};
  EXPECT_EQ(excepted_src_data_slice, src_data_slice);
}

TEST_F(trans_data, data_slice_infer_nz_nd_6) {
  ge::op::TransData op;
  std::vector<std::pair<int64_t, int64_t>> shape_src_range = {{4, 4}, {3, 3}, {16, 16}, {16, 16}};
  std::vector<std::pair<int64_t, int64_t>> shape_dst_range = {{37, 37}, {57, 57}};
  auto src_desc = create_desc_shape_range({4, 3, 16, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_FRACTAL_NZ,
                                         {4, 3, 16, 16},
                                         ge::FORMAT_FRACTAL_NZ,
                                         shape_src_range);
  auto dst_desc = create_desc_shape_range({37, 57},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_NHWC,
                                         {37, 57},
                                         ge::FORMAT_NHWC,
                                         shape_dst_range);

  op.UpdateInputDesc("src", src_desc);
  op.SetAttr("src_format", "FRACTAL_NZ");
  op.SetAttr("dst_format", "NHWC");
  op.UpdateOutputDesc("dst", dst_desc);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc("dst");
  std::vector<std::vector<int64_t>> dst_data_slice = {{32, 36}, {}};
  ge::AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, dst_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_src = op_desc->MutableInputDesc("src");
  std::vector<std::vector<int64_t>> src_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_src, ge::ATTR_NAME_DATA_SLICE, src_data_slice);
  std::vector<std::vector<int64_t>> excepted_src_data_slice = {{0, 3}, {2, 2}, {0, 15}, {0, 15}};
  EXPECT_EQ(excepted_src_data_slice, src_data_slice);
}

TEST_F(trans_data, data_slice_infer_nz_nd_7) {
  ge::op::TransData op;
  std::vector<std::pair<int64_t, int64_t>> shape_src_range = {{4, 4}, {3, 3}, {16, 16}, {16, 16}};
  std::vector<std::pair<int64_t, int64_t>> shape_dst_range = {{37, 37}, {57, 57}};
  auto src_desc = create_desc_shape_range({4, 3, 16, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_FRACTAL_NZ,
                                         {4, 3, 16, 16},
                                         ge::FORMAT_FRACTAL_NZ,
                                         shape_src_range);
  auto dst_desc = create_desc_shape_range({37, 57},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_NHWC,
                                         {37, 57},
                                         ge::FORMAT_NHWC,
                                         shape_dst_range);

  op.UpdateInputDesc("src", src_desc);
  op.SetAttr("src_format", "FRACTAL_NZ");
  op.SetAttr("dst_format", "NHWC");
  op.UpdateOutputDesc("dst", dst_desc);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc("dst");
  std::vector<std::vector<int64_t>> dst_data_slice = {{0, 15}, {}};
  ge::AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, dst_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_src = op_desc->MutableInputDesc("src");
  std::vector<std::vector<int64_t>> src_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_src, ge::ATTR_NAME_DATA_SLICE, src_data_slice);
  std::vector<std::vector<int64_t>> excepted_src_data_slice = {{0, 3}, {0, 0}, {0, 15}, {0, 15}};
  EXPECT_EQ(excepted_src_data_slice, src_data_slice);
}

TEST_F(trans_data, data_slice_infer_nz_nd_8) {
  ge::op::TransData op;
  std::vector<std::pair<int64_t, int64_t>> shape_src_range = {{7, 7}, {4, 4}, {3, 3}, {16, 16}, {16, 16}};
  std::vector<std::pair<int64_t, int64_t>> shape_dst_range = {{7, 7}, {37, 37}, {57, 57}};
  auto src_desc = create_desc_shape_range({7, 4, 3, 16, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_FRACTAL_NZ,
                                         {7, 4, 3, 16, 16},
                                         ge::FORMAT_FRACTAL_NZ,
                                         shape_src_range);
  auto dst_desc = create_desc_shape_range({7, 37, 57},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_ND,
                                         {7, 37, 57},
                                         ge::FORMAT_ND,
                                         shape_dst_range);

  op.UpdateInputDesc("src", src_desc);
  op.SetAttr("src_format", "FRACTAL_NZ");
  op.SetAttr("dst_format", "ND");
  op.UpdateOutputDesc("dst", dst_desc);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc("dst");
  std::vector<std::vector<int64_t>> dst_data_slice = {{2, 3}, {}, {}};
  ge::AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, dst_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_src = op_desc->MutableInputDesc("src");
  std::vector<std::vector<int64_t>> src_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_src, ge::ATTR_NAME_DATA_SLICE, src_data_slice);
  std::vector<std::vector<int64_t>> excepted_src_data_slice = {{2, 3}, {0, 3}, {0, 2}, {0, 15}, {0, 15}};
  EXPECT_EQ(excepted_src_data_slice, src_data_slice);
}

TEST_F(trans_data, data_slice_infer_nz_nd_9) {
  ge::op::TransData op;
  std::vector<std::pair<int64_t, int64_t>> shape_src_range = {{4, 4}, {3, 3}, {16, 16}, {16, 16}};
  std::vector<std::pair<int64_t, int64_t>> shape_dst_range = {{37, 37}, {57, 57}};
  auto src_desc = create_desc_shape_range({4, 3, 16, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_FRACTAL_NZ,
                                         {4, 3, 16, 16},
                                         ge::FORMAT_FRACTAL_NZ,
                                         shape_src_range);
  auto dst_desc = create_desc_shape_range({37, 57},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_NHWC,
                                         {37, 57},
                                         ge::FORMAT_NHWC,
                                         shape_dst_range);

  op.UpdateInputDesc("src", src_desc);
  op.SetAttr("src_format", "FRACTAL_NZ");
  op.SetAttr("dst_format", "NHWC");
  op.UpdateOutputDesc("dst", dst_desc);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc("dst");
  std::vector<std::vector<int64_t>> dst_data_slice = {{13, 19}, {}};
  ge::AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, dst_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::NOT_SUPPORT_SLICE);
}

TEST_F(trans_data, data_slice_infer_nz_nd_10) {
  ge::op::TransData op;
  std::vector<std::pair<int64_t, int64_t>> shape_src_range = {{4, 4}, {3, 3}, {32, 32}, {32, 32}};
  std::vector<std::pair<int64_t, int64_t>> shape_dst_range = {{79, 79}, {97, 97}};
  auto src_desc = create_desc_shape_range({4, 3, 32, 32},
                                         ge::DT_INT8,
                                         ge::FORMAT_FRACTAL_NZ,
                                         {4, 3, 32, 32},
                                         ge::FORMAT_FRACTAL_NZ,
                                         shape_src_range);
  auto dst_desc = create_desc_shape_range({79, 97},
                                         ge::DT_INT8,
                                         ge::FORMAT_NHWC,
                                         {79, 97},
                                         ge::FORMAT_NHWC,
                                         shape_dst_range);

  op.UpdateInputDesc("src", src_desc);
  op.SetAttr("src_format", "FRACTAL_NZ");
  op.SetAttr("dst_format", "NHWC");
  op.UpdateOutputDesc("dst", dst_desc);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc("dst");
  std::vector<std::vector<int64_t>> dst_data_slice = {{32, 63}, {}};
  ge::AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, dst_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_src = op_desc->MutableInputDesc("src");
  std::vector<std::vector<int64_t>> src_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_src, ge::ATTR_NAME_DATA_SLICE, src_data_slice);
  std::vector<std::vector<int64_t>> excepted_src_data_slice = {{0, 3}, {1, 1}, {0, 31}, {0, 31}};
  EXPECT_EQ(excepted_src_data_slice, src_data_slice);
}

TEST_F(trans_data, data_slice_infer_nd_6hd_11) {
  ge::op::TransData op;
  std::vector<std::pair<int64_t, int64_t>> shape_src_range = {{4, 4}, {7, 7}, {8, 8}, {12, 12}, {24, 24}};
  std::vector<std::pair<int64_t, int64_t>> shape_dst_range = {{4, 4}, {7, 7}, {2, 2}, {8, 8}, {12, 12}, {16, 16}};
  auto src_desc = create_desc_shape_range({4, 7, 8, 12, 24},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_NDHWC,
                                         {4, 7, 8, 12, 24},
                                         ge::FORMAT_NDHWC,
                                         shape_src_range);
  auto dst_desc = create_desc_shape_range({4, 7, 2, 8, 12, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_NDC1HWC0,
                                         {4, 7, 2, 8, 12, 16},
                                         ge::FORMAT_NDC1HWC0,
                                         shape_dst_range);
  op.UpdateInputDesc("src", src_desc);
  op.SetAttr("src_format", "NDHWC");
  op.SetAttr("dst_format", "NDC1HWC0");
  op.UpdateOutputDesc("dst", dst_desc);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc("dst");
  std::vector<std::vector<int64_t>> dst_data_slice = {{2, 3}, {}, {}, {}, {}, {}};
  ge::AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, dst_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::NOT_SUPPORT_SLICE);
}

TEST_F(trans_data, data_slice_infer_nd_nz_12) {
  ge::op::TransData op;
  std::vector<std::pair<int64_t, int64_t>> shape_src_range = {{7, 7}, {37, 37}, {57, 57}};
  std::vector<std::pair<int64_t, int64_t>> shape_dst_range = {{7, 7}, {4, 4}, {3, 3}, {16, 16}, {16, 16}};
  auto src_desc = create_desc_shape_range({7, 37, 57},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_NCHW,
                                         {7, 37, 57},
                                         ge::FORMAT_NCHW,
                                         shape_src_range);
  auto dst_desc = create_desc_shape_range({7, 4, 3, 16, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_FRACTAL_NZ,
                                         {7, 4, 3, 16, 16},
                                         ge::FORMAT_FRACTAL_NZ,
                                         shape_dst_range);
  op.UpdateInputDesc("src", src_desc);
  op.SetAttr("src_format", "NCHW");
  op.SetAttr("dst_format", "FRACTAL_NZ");
  op.UpdateOutputDesc("dst", dst_desc);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc("dst");
  std::vector<std::vector<int64_t>> dst_data_slice = {{3, 7}, {}, {}, {}, {}};
  ge::AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, dst_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::NOT_SUPPORT_SLICE);
}

TEST_F(trans_data, data_slice_infer_nd_5hd_13) {
  ge::op::TransData op;
  std::vector<std::pair<int64_t, int64_t>> shape_src_range = {{4, 4}, {8, 8}, {12, 12}, {24, 24}};
  std::vector<std::pair<int64_t, int64_t>> shape_dst_range = {{4, 4}, {2, 2}, {8, 8}, {12, 12}, {16, 16}};
  auto src_desc = create_desc_shape_range({4, 8, 12, 24},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_NCHW,
                                         {4, 8, 12, 24},
                                         ge::FORMAT_NCHW,
                                         shape_src_range);
  auto dst_desc = create_desc_shape_range({4, 2, 8, 12, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_NC1HWC0,
                                         {4, 2, 8, 12, 16},
                                         ge::FORMAT_NC1HWC0,
                                         shape_dst_range);
  op.UpdateInputDesc("src", src_desc);
  op.SetAttr("src_format", "NCHW");
  op.SetAttr("dst_format", "NC1HWC0");
  op.UpdateOutputDesc("dst", dst_desc);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc("dst");
  std::vector<std::vector<int64_t>> dst_data_slice = {{2, 2}, {}, {}, {}, {}};
  ge::AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, dst_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_src = op_desc->MutableInputDesc("src");
  std::vector<std::vector<int64_t>> src_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_src, ge::ATTR_NAME_DATA_SLICE, src_data_slice);
  std::vector<std::vector<int64_t>> excepted_src_data_slice = {{2, 2}, {0, 7}, {0, 11}, {0, 23}};
  EXPECT_EQ(excepted_src_data_slice, src_data_slice);
}

TEST_F(trans_data, data_slice_infer_nd_nz_14) {
  ge::op::TransData op;
  std::vector<std::pair<int64_t, int64_t>> shape_src_range = {{37, 37}, {57, 57}};
  std::vector<std::pair<int64_t, int64_t>> shape_dst_range = {{4, 4}, {3, 3}, {16, 16}, {16, 16}};
  auto src_desc = create_desc_shape_range({37, 57},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_NCHW,
                                         {37, 57},
                                         ge::FORMAT_NCHW,
                                         shape_src_range);
  auto dst_desc = create_desc_shape_range({4, 3, 16, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_FRACTAL_NZ,
                                         {4, 3, 16, 16},
                                         ge::FORMAT_FRACTAL_NZ,
                                         shape_dst_range);
  op.UpdateInputDesc("src", src_desc);
  op.SetAttr("src_format", "NCHW");
  op.SetAttr("dst_format", "FRACTAL_NZ");
  op.UpdateOutputDesc("dst", dst_desc);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc("dst");
  std::vector<std::vector<int64_t>> dst_data_slice = {{}, {0, 1}, {}, {}};
  ge::AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, dst_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_src = op_desc->MutableInputDesc("src");
  std::vector<std::vector<int64_t>> src_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_src, ge::ATTR_NAME_DATA_SLICE, src_data_slice);
  std::vector<std::vector<int64_t>> excepted_src_data_slice = {{0, 31}, {0, 56}};
  EXPECT_EQ(excepted_src_data_slice, src_data_slice);
}
