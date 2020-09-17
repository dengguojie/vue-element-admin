#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"


class cosine_embedding_loss : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "cosine_embedding_loss SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "cosine_embedding_loss TearDown" << std::endl;
    }
};

// normal cases
TEST_F(cosine_embedding_loss, cosine_embedding_loss_infershape_reduce_sum_test) {
    ge::op::CosineEmbeddingLoss op;
    op.UpdateInputDesc("x1", create_desc_with_ori({1, 2, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 2, 3, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("x2", create_desc_with_ori({1, 2, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 2, 3, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("target", create_desc_with_ori({1, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 3, 4}, ge::FORMAT_ND));
    op.SetAttr("margin", 0.3f);
    op.SetAttr("reduction", std::string("sum"));
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {1,};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(cosine_embedding_loss, cosine_embedding_loss_infershape_reduce_mean_test) {
    ge::op::CosineEmbeddingLoss op;
    op.UpdateInputDesc("x1", create_desc_with_ori({1, 2, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 2, 3, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("x2", create_desc_with_ori({1, 2, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 2, 3, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("target", create_desc_with_ori({1, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 3, 4}, ge::FORMAT_ND));
    op.SetAttr("margin", 0.3f);
    op.SetAttr("reduction", std::string("mean"));
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {1,};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(cosine_embedding_loss, cosine_embedding_loss_infershape_reduce_none_test) {
    ge::op::CosineEmbeddingLoss op;
    op.UpdateInputDesc("x1", create_desc_with_ori({1, 2, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 2, 3, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("x2", create_desc_with_ori({1, 2, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 2, 3, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("target", create_desc_with_ori({1, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 3, 4}, ge::FORMAT_ND));
    op.SetAttr("margin", 0.3f);
    op.SetAttr("reduction", std::string("none"));
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {1, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(cosine_embedding_loss, cosine_embedding_loss_infershape_broadcast_valid_test) {
    ge::op::CosineEmbeddingLoss op;
    op.UpdateInputDesc("x1", create_desc_with_ori({1, 2, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 2, 3, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("x2", create_desc_with_ori({2, 2, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 2, 3, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("target", create_desc_with_ori({1, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 3, 4}, ge::FORMAT_ND));
    op.SetAttr("margin", 0.3f);
    op.SetAttr("reduction", std::string("none"));
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

// exception cases
TEST_F(cosine_embedding_loss, cosine_embedding_loss_infershape_broadcast_invalid_test) {
    ge::op::CosineEmbeddingLoss op;
    op.UpdateInputDesc("x1", create_desc_with_ori({1, 2, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 2, 3, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("x2", create_desc_with_ori({2, 3, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3, 3, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("target", create_desc_with_ori({1, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 3, 4}, ge::FORMAT_ND));
    op.SetAttr("margin", 0.3f);
    op.SetAttr("reduction", std::string("none"));
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(cosine_embedding_loss, cosine_embedding_loss_infershape_shape_invalid_test1) {
    ge::op::CosineEmbeddingLoss op;
    op.UpdateInputDesc("x1", create_desc_with_ori({1}, ge::DT_FLOAT16, ge::FORMAT_ND, {1}, ge::FORMAT_ND));
    op.UpdateInputDesc("x2", create_desc_with_ori({1}, ge::DT_FLOAT16, ge::FORMAT_ND, {1}, ge::FORMAT_ND));
    op.UpdateInputDesc("target", create_desc_with_ori({1, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 3, 4}, ge::FORMAT_ND));
    op.SetAttr("margin", 0.3f);
    op.SetAttr("reduction", std::string("none"));
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(cosine_embedding_loss, cosine_embedding_loss_infershape_shape_invalid_test2) {
    ge::op::CosineEmbeddingLoss op;
    op.UpdateInputDesc("x1", create_desc_with_ori({3}, ge::DT_FLOAT16, ge::FORMAT_ND, {3}, ge::FORMAT_ND));
    op.UpdateInputDesc("x2", create_desc_with_ori({2, 3, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3, 3, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("target", create_desc_with_ori({1, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 3, 4}, ge::FORMAT_ND));
    op.SetAttr("margin", 0.3f);
    op.SetAttr("reduction", std::string("none"));
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(cosine_embedding_loss, cosine_embedding_loss_infershape_reduction_invalid_test) {
    ge::op::CosineEmbeddingLoss op;
    op.UpdateInputDesc("x1", create_desc_with_ori({1}, ge::DT_FLOAT16, ge::FORMAT_ND, {1}, ge::FORMAT_ND));
    op.UpdateInputDesc("x2", create_desc_with_ori({2, 3, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3, 3, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("target", create_desc_with_ori({1, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 3, 4}, ge::FORMAT_ND));
    op.SetAttr("margin", 0.3f);
    op.SetAttr("reduction", std::string("max"));
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}
