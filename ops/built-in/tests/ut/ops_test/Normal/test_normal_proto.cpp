#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "random_ops.h"

class normal_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "normal SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "normal TearDown" << std::endl;
    }
};

TEST_F(normal_test, normal_infershape_test_1) {
    ge::op::Normal op;
    op.UpdateInputDesc("mean", create_desc_with_ori({2, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("std", create_desc_with_ori({2, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3}, ge::FORMAT_ND));
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDescByName("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {2, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(normal_test, normal_infershape_test_2) {
    ge::op::Normal op;
    op.UpdateInputDesc("mean", create_desc_with_ori({2, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("std", create_desc_with_ori({2, 3}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 3}, ge::FORMAT_ND));
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDescByName("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {2, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
