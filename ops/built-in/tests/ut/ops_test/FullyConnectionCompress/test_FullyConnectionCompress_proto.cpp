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
 * @file test_FullyConnectionCompress_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "matrix_calculation_ops.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "op_desc.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

// ---------------FullyConnectionCompress-------------------
class FullyConnectionCompressProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "FullyConnectionCompress Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "FullyConnectionCompress Proto Test TearDown" << std::endl;
  }
};

TEST_F(FullyConnectionCompressProtoTest, fullyConnectionCompressInferShapeTest_1) {
    ge::op::FullyConnectionCompress fullyConnectionCompress;
    fullyConnectionCompress.UpdateInputDesc("x", create_desc_with_ori({1, 16, 1, 1, 32}, ge::DT_INT8, ge::FORMAT_NC1HWC0,{1, 16, 1, 1, 32}, ge::FORMAT_NC1HWC0));
    fullyConnectionCompress.UpdateInputDesc("w", create_desc_with_ori({16, 1, 16, 32}, ge::DT_INT8, ge::FORMAT_FRACTAL_Z,{16, 1, 16, 32}, ge::FORMAT_FRACTAL_Z));
    fullyConnectionCompress.UpdateInputDesc("compress_index", create_desc_with_ori({1}, ge::DT_INT8, ge::FORMAT_ND, {1}, ge::FORMAT_ND));
    fullyConnectionCompress.UpdateOutputDesc("y", create_desc_with_ori({1, 1, 1, 1, 16}, ge::DT_INT32, ge::FORMAT_NC1HWC0,{1, 1, 1, 1, 16}, ge::FORMAT_NC1HWC0));
    auto status = fullyConnectionCompress.InferShapeAndType();
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(FullyConnectionCompressProtoTest, fullyConnectionCompressInferShapeTest_2) {
    ge::op::FullyConnectionCompress fullyConnectionCompress;
    fullyConnectionCompress.UpdateInputDesc("x", create_desc_with_ori({1, 16, 1, 1, 32}, ge::DT_INT8, ge::FORMAT_NC1HWC0,{1, 16, 1, 1, 32}, ge::FORMAT_NC1HWC0));
    fullyConnectionCompress.UpdateInputDesc("w", create_desc_with_ori({}, ge::DT_INT8, ge::FORMAT_FRACTAL_Z,{}, ge::FORMAT_FRACTAL_Z));
    fullyConnectionCompress.UpdateInputDesc("compress_index", create_desc_with_ori({1}, ge::DT_INT8, ge::FORMAT_ND, {1}, ge::FORMAT_ND));
    fullyConnectionCompress.UpdateOutputDesc("y", create_desc_with_ori({1, 1, 1, 1, 16}, ge::DT_INT32, ge::FORMAT_NC1HWC0,{1, 1, 1, 1, 16}, ge::FORMAT_NC1HWC0));
    auto status = fullyConnectionCompress.InferShapeAndType();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FullyConnectionCompressProtoTest, fullyConnectionCompressVerifyTest_1) {
    ge::op::FullyConnectionCompress fullyConnectionCompress;
    fullyConnectionCompress.UpdateInputDesc("x", create_desc_with_ori({4, 2, 2, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0,{4, 16, 1, 1}, ge::FORMAT_NCHW));
    fullyConnectionCompress.UpdateInputDesc("w", create_desc_with_ori({1, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_Z,{32, 16, 1, 1}, ge::FORMAT_NCHW));
    fullyConnectionCompress.UpdateInputDesc("compress_index", create_desc_with_ori({1}, ge::DT_INT8, ge::FORMAT_ND, {1}, ge::FORMAT_ND));
    fullyConnectionCompress.UpdateOutputDesc("y", create_desc_with_ori({4, 2, 1, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0,{4, 32, 1, 1}, ge::FORMAT_NCHW));
    auto status = fullyConnectionCompress.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FullyConnectionCompressProtoTest, fullyConnectionCompressVerifyTest_2) {
    ge::op::FullyConnectionCompress fullyConnectionCompress;
    fullyConnectionCompress.UpdateInputDesc("x", create_desc_with_ori({4, 2, 2, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0,{4, 16, 1, 1}, ge::FORMAT_NCHW));
    fullyConnectionCompress.UpdateInputDesc("w", create_desc_with_ori({1, 1}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_Z,{1, 1}, ge::FORMAT_NCHW));
    fullyConnectionCompress.UpdateInputDesc("compress_index", create_desc_with_ori({1}, ge::DT_INT8, ge::FORMAT_ND, {1}, ge::FORMAT_ND));
    fullyConnectionCompress.UpdateOutputDesc("y", create_desc_with_ori({4, 2, 1, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0,{4, 32, 1, 1}, ge::FORMAT_NCHW));
    auto status = fullyConnectionCompress.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FullyConnectionCompressProtoTest, fullyConnectionCompressSplicDataTest_1) {
    ge::op::FullyConnectionCompress fullyConnectionCompress;
    fullyConnectionCompress.UpdateInputDesc("x", create_desc_with_ori({4, 1, 4, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{4, 64, 16, 1}, ge::FORMAT_NHWC));
    fullyConnectionCompress.UpdateInputDesc("w", create_desc_with_ori({1, 1, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_Z,{8, 16, 1, 1}, ge::FORMAT_HWCN));
    fullyConnectionCompress.UpdateInputDesc("compress_index", create_desc_with_ori({1}, ge::DT_INT8, ge::FORMAT_ND, {1}, ge::FORMAT_ND));
    fullyConnectionCompress.UpdateOutputDesc("y", create_desc_with_ori({4, 1, 4, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{4, 64, 8, 1}, ge::FORMAT_NHWC));
    fullyConnectionCompress.SetAttr("num_output", 8);
    fullyConnectionCompress.SetAttr("transpose", false);
    fullyConnectionCompress.SetAttr("axis", 2);
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {0, 2}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(fullyConnectionCompress);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);

    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    ge::GeTensorDescPtr tensor_desc_w = op_desc->MutableInputDesc("w");
    std::vector<std::vector<int64_t>> x_data_slice;
    std::vector<std::vector<int64_t>> w_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
    ge::AttrUtils::GetListListInt(tensor_desc_w, ge::ATTR_NAME_DATA_SLICE, w_data_slice);

    std::vector<std::vector<int64_t>> expect_x_data_slice = {{}, {}, {0, 2}, {}, {}};
    std::vector<std::vector<int64_t>> expect_w_data_slice = {};
    EXPECT_EQ(expect_x_data_slice, x_data_slice);
    EXPECT_EQ(expect_w_data_slice, w_data_slice);
}
