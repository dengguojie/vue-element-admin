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
 * @file test_PtAdd_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "op_proto_test_util.h"
#include "math_ops.h"

class RealTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "Real test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "Real test TearDown" << std::endl;
    }
};

TEST_F(RealTest, infershape_case_00) {
    ge::op::Real op;
    op.SetAttr("Tout", ge::DT_FLOAT);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
