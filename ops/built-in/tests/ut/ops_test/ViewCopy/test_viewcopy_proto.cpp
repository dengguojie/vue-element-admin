/**
 * Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_viewcopy_proto.cpp
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

class TEST_VIEWCOPY_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "viewcopy test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "viewcopy test TearDown" << std::endl;
  }
};

TEST_F(TEST_VIEWCOPY_UT, viewcopy_infershape) {
    ge::op::ViewCopy op;
    ge::TensorDesc src_td;
    ge::TensorDesc dst_td;
    ge::Shape src_shape({3, 1, 1});
    ge::Shape dst_shape({5, 3, 4, 1});
    src_td.SetDataType(ge::DT_UINT8);
    dst_td.SetDataType(ge::DT_UINT8);
    src_td.SetShape(src_shape);
    src_td.SetOriginShape(src_shape);
    dst_td.SetShape(dst_shape);
    dst_td.SetOriginShape(dst_shape);

    op.UpdateInputDesc("src", src_td);
    op.UpdateInputDesc("dst", dst_td);
    op.UpdateOutputDesc("dst", dst_td);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(TEST_VIEWCOPY_UT, viewcopy_src_size_bigger_than_dst_size) {
    ge::op::ViewCopy op;
    ge::TensorDesc src_size_td;
    ge::TensorDesc dst_size_td;
    ge::Shape src_size_shape({5, 3, 4, 1});
    ge::Shape dst_size_shape({3, 1, 1});
    src_size_td.SetDataType(ge::DT_FLOAT16);
    dst_size_td.SetDataType(ge::DT_UINT8);
    src_size_td.SetShape(src_size_shape);
    src_size_td.SetOriginShape(src_size_shape);
    dst_size_td.SetShape(dst_size_shape);
    dst_size_td.SetOriginShape(dst_size_shape);

    op.UpdateInputDesc("src_size", src_size_td);
    op.UpdateInputDesc("dst_size", dst_size_td);

    auto ret = op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(TEST_VIEWCOPY_UT, viewcopy_size_equal) {
    ge::op::ViewCopy op;
    ge::TensorDesc src_size_td;
    ge::TensorDesc dst_size_td;
    ge::Shape src_size_shape({3, 1, 1});
    ge::Shape dst_size_shape({3, 1, 1});
    src_size_td.SetDataType(ge::DT_UINT8);
    dst_size_td.SetDataType(ge::DT_UINT8);
    src_size_td.SetShape(src_size_shape);
    src_size_td.SetOriginShape(src_size_shape);
    dst_size_td.SetShape(dst_size_shape);
    dst_size_td.SetOriginShape(dst_size_shape);

    op.UpdateInputDesc("src_size", src_size_td);
    op.UpdateInputDesc("dst_size", dst_size_td);

    auto ret = op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(TEST_VIEWCOPY_UT, viewcopy_size_valid_001) {
    ge::op::ViewCopy op;
    ge::TensorDesc src_size_td;
    ge::TensorDesc dst_size_td;
    ge::Shape src_size_shape({3, 1, 1});
    ge::Shape dst_size_shape({5, 3, 4, 1});
    src_size_td.SetDataType(ge::DT_UINT8);
    dst_size_td.SetDataType(ge::DT_UINT8);
    src_size_td.SetShape(src_size_shape);
    src_size_td.SetOriginShape(src_size_shape);
    dst_size_td.SetShape(dst_size_shape);
    dst_size_td.SetOriginShape(dst_size_shape);

    op.UpdateInputDesc("src_size", src_size_td);
    op.UpdateInputDesc("dst_size", dst_size_td);

    auto ret = op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(TEST_VIEWCOPY_UT, viewcopy_size_valid_002) {
    ge::op::ViewCopy op;
    ge::TensorDesc src_size_td;
    ge::TensorDesc dst_size_td;
    ge::Shape src_size_shape({1, 1});
    ge::Shape dst_size_shape({5, 3, 4, 1});
    src_size_td.SetDataType(ge::DT_UINT8);
    dst_size_td.SetDataType(ge::DT_UINT8);
    src_size_td.SetShape(src_size_shape);
    src_size_td.SetOriginShape(src_size_shape);
    dst_size_td.SetShape(dst_size_shape);
    dst_size_td.SetOriginShape(dst_size_shape);

    op.UpdateInputDesc("src_size", src_size_td);
    op.UpdateInputDesc("dst_size", dst_size_td);

    auto ret = op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(TEST_VIEWCOPY_UT, viewcopy_size_valid_003) {
    ge::op::ViewCopy op;
    ge::TensorDesc src_size_td;
    ge::TensorDesc dst_size_td;
    ge::Shape src_size_shape({1, 1});
    ge::Shape dst_size_shape({5, 3, 4, 1});
    src_size_td.SetDataType(ge::DT_UINT8);
    dst_size_td.SetDataType(ge::DT_UINT8);
    src_size_td.SetShape(src_size_shape);
    src_size_td.SetOriginShape(src_size_shape);
    dst_size_td.SetShape(dst_size_shape);
    dst_size_td.SetOriginShape(dst_size_shape);

    op.UpdateInputDesc("src_size", src_size_td);
    op.UpdateInputDesc("dst_size", dst_size_td);

    auto ret = op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(TEST_VIEWCOPY_UT, viewcopy_size_invalid_001) {
    ge::op::ViewCopy op;
    ge::TensorDesc src_size_td;
    ge::TensorDesc dst_size_td;
    ge::Shape src_size_shape({2, 1, 1});
    ge::Shape dst_size_shape({5, 3, 4, 1});
    src_size_td.SetDataType(ge::DT_UINT8);
    dst_size_td.SetDataType(ge::DT_UINT8);
    src_size_td.SetShape(src_size_shape);
    src_size_td.SetOriginShape(src_size_shape);
    dst_size_td.SetShape(dst_size_shape);
    dst_size_td.SetOriginShape(dst_size_shape);

    op.UpdateInputDesc("src_size", src_size_td);
    op.UpdateInputDesc("dst_size", dst_size_td);

    auto ret = op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

