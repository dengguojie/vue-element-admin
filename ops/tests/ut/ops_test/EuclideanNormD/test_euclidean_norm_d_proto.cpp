#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "reduce_ops.h"

// ----------------EuclideanNorm--------------
class euclidean_norm : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "euclidean_norm SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "euclidean_norm TearDown" << std::endl;
    }
};
/*
TEST_F(euclidean_norm, euclidean_norm_infershape_test) {
    ge::op::EuclideanNorm op;
    op.UpdateInputDesc("x", create_desc_with_ori({1, 2, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 2, 3, 4}, ge::FORMAT_ND));
	op.UpdateInputDesc("axes", create_desc({0,1}, ge::DT_INT32));
    op.SetAttr("keep_dims", false);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}*/


// ----------------EuclideanNormD--------------
class euclidean_norm_d : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "euclidean_norm_d SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "euclidean_norm_d TearDown" << std::endl;
    }
};

TEST_F(euclidean_norm_d, euclidean_norm_d_infershape_test) {
    ge::op::EuclideanNormD op;
    op.UpdateInputDesc("x", create_desc_with_ori({1, 2, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 2, 3, 4}, ge::FORMAT_ND));
    std::vector<int64_t> axes = {0,1};
    op.SetAttr("axes", axes);
    op.SetAttr("keep_dims", false);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
