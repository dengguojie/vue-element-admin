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
 * @file test_YoloxBoundingBoxDecode_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"

class yoloxboundingboxdecode : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "YoloxBoundingBoxDecode SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "YoloxBoundingBoxDecode TearDown" << std::endl;
    }
};

TEST_F(yoloxboundingboxdecode, yoloxboundingboxdecode_infershape_test1) {
    ge::op::YoloxBoundingBoxDecode op;
    op.UpdateInputDesc("priors", create_desc({-1, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bboxes", create_desc({-1, -1, 4}, ge::DT_FLOAT16));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  
    auto output_desc = op.GetOutputDesc("decoded_bboxes");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);

    std::vector<int64_t> expected_output_shape = {-1, -1, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(yoloxboundingboxdecode, yoloxboundingboxdecode_infershape_test2) {
    ge::op::YoloxBoundingBoxDecode op;
    op.UpdateInputDesc("priors", create_desc({8400, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bboxes", create_desc({-1, 8400, 4}, ge::DT_FLOAT16));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  
    auto output_desc = op.GetOutputDesc("decoded_bboxes");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);

    std::vector<int64_t> expected_output_shape = {-1, 8400, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(yoloxboundingboxdecode, yoloxboundingboxdecode_infershape_test3) {
    ge::op::YoloxBoundingBoxDecode op;
    op.UpdateInputDesc("priors", create_desc({8400, 4}, ge::DT_FLOAT));
    op.UpdateInputDesc("bboxes", create_desc({2, 8400, 4}, ge::DT_FLOAT));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  
    auto output_desc = op.GetOutputDesc("decoded_bboxes");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {2, 8400, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}


TEST_F(yoloxboundingboxdecode, yoloxboundingboxdecode_infershape_test4) {
    ge::op::YoloxBoundingBoxDecode op;
    op.UpdateInputDesc("priors", create_desc({2, 8400, 4}, ge::DT_FLOAT));
    op.UpdateInputDesc("bboxes", create_desc({2, 8400, 4}, ge::DT_FLOAT));
  
    auto ret = op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}


TEST_F(yoloxboundingboxdecode, yoloxboundingboxdecode_infershape_test5) {
    ge::op::YoloxBoundingBoxDecode op;
    op.UpdateInputDesc("priors", create_desc({8400, 2}, ge::DT_FLOAT));
    op.UpdateInputDesc("bboxes", create_desc({2, 8400, 2}, ge::DT_FLOAT));
  
    auto ret = op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}


TEST_F(yoloxboundingboxdecode, yoloxboundingboxdecode_infershape_test6) {
    ge::op::YoloxBoundingBoxDecode op;
    op.UpdateInputDesc("priors", create_desc({8400, 4}, ge::DT_FLOAT));
    op.UpdateInputDesc("bboxes", create_desc({2, 9600, 4}, ge::DT_FLOAT));
  
    auto ret = op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
