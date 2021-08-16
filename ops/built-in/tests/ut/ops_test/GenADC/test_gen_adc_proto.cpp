#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "vector_search.h"

class GenADCProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GenADC Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GenADC Proto Test TearDown" << std::endl;
  }
};

TEST_F(GenADCProtoTest, GenADCVerifyTest_1) {
  ge::op::GenADC op;
  op.UpdateInputDesc("query", create_desc({32}, ge::DT_FLOAT16));
  op.UpdateInputDesc("code_book", create_desc({16, 256, 2}, ge::DT_FLOAT16));
  op.UpdateInputDesc("centroids", create_desc({1000000, 32}, ge::DT_FLOAT16));
  op.UpdateInputDesc("bucket_list", create_desc({1024}, ge::DT_INT32));
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(GenADCProtoTest, GenADCVerifyTest_2) {
  ge::op::GenADC op;
  op.UpdateInputDesc("query", create_desc({32}, ge::DT_FLOAT));
  op.UpdateInputDesc("code_book", create_desc({16, 256, 2}, ge::DT_FLOAT));
  op.UpdateInputDesc("centroids", create_desc({1000000, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("bucket_list", create_desc({-1}, ge::DT_INT64));
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(GenADCProtoTest, GenADCVerifyTest_3) {
  ge::op::GenADC op;
  op.UpdateInputDesc("query", create_desc({128}, ge::DT_FLOAT));
  op.UpdateInputDesc("code_book", create_desc({32, 256, 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("centroids", create_desc({1000000, 128}, ge::DT_FLOAT));
  op.UpdateInputDesc("bucket_list", create_desc({37}, ge::DT_INT64));
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(GenADCProtoTest, GenADCVerifyTest_4) {
  ge::op::GenADC op;
  op.UpdateInputDesc("query", create_desc({128}, ge::DT_FLOAT));
  op.UpdateInputDesc("code_book", create_desc({32, 256, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("centroids", create_desc({1000000, 128}, ge::DT_FLOAT));
  op.UpdateInputDesc("bucket_list", create_desc({37}, ge::DT_INT64));
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(GenADCProtoTest, GenADCVerifyTest_5) {
  ge::op::GenADC op;
  op.UpdateInputDesc("query", create_desc({32, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("code_book", create_desc({32, 256, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("centroids", create_desc({1000000, 128}, ge::DT_FLOAT));
  op.UpdateInputDesc("bucket_list", create_desc({37}, ge::DT_INT64));
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(GenADCProtoTest, GenADCVerifyTest_6) {
  ge::op::GenADC op;
  op.UpdateInputDesc("query", create_desc({35}, ge::DT_FLOAT));
  op.UpdateInputDesc("code_book", create_desc({32, 256, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("centroids", create_desc({1000000, 128}, ge::DT_FLOAT));
  op.UpdateInputDesc("bucket_list", create_desc({37}, ge::DT_INT64));
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(GenADCProtoTest, GenADCVerifyTest_7) {
  ge::op::GenADC op;
  op.UpdateInputDesc("query", create_desc({32}, ge::DT_FLOAT16));
  op.UpdateInputDesc("code_book", create_desc({16, 256}, ge::DT_FLOAT16));
  op.UpdateInputDesc("centroids", create_desc({1000000, 32}, ge::DT_FLOAT16));
  op.UpdateInputDesc("bucket_list", create_desc({1024}, ge::DT_INT32));
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(GenADCProtoTest, GenADCVerifyTest_8) {
  ge::op::GenADC op;
  op.UpdateInputDesc("query", create_desc({32}, ge::DT_FLOAT16));
  op.UpdateInputDesc("code_book", create_desc({16, 256, 2}, ge::DT_FLOAT16));
  op.UpdateInputDesc("centroids", create_desc({1000000, 35}, ge::DT_FLOAT16));
  op.UpdateInputDesc("bucket_list", create_desc({1024}, ge::DT_INT32));
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(GenADCProtoTest, GenADCVerifyTest_9) {
  ge::op::GenADC op;
  op.UpdateInputDesc("query", create_desc({32}, ge::DT_FLOAT16));
  op.UpdateInputDesc("code_book", create_desc({16, 256, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("centroids", create_desc({1000000, 32}, ge::DT_FLOAT16));
  op.UpdateInputDesc("bucket_list", create_desc({1024}, ge::DT_INT32));
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(GenADCProtoTest, GenADCInferShapeTest_1) {
  ge::op::GenADC op;
  op.UpdateInputDesc("query",
                     create_desc_shape_range({32}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, {{32, 32}}));
  op.UpdateInputDesc("code_book", create_desc_shape_range({16, 256, 2}, ge::DT_FLOAT16, ge::FORMAT_ND, {16, 256, 2},
                                                          ge::FORMAT_ND, {{16, 16}, {256, 256}, {2, 2}}));
  op.UpdateInputDesc("centroids", create_desc_shape_range({1000000, 32}, ge::DT_FLOAT16, ge::FORMAT_ND, {1000000, 32},
                                                          ge::FORMAT_ND, {{1000000, 1000000}, {32, 32}}));
  op.UpdateInputDesc("bucket_list",
                     create_desc_shape_range({1024}, ge::DT_INT32, ge::FORMAT_ND, {1024}, ge::FORMAT_ND, {{1024, 1024}}));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDescByName("adc_tables");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);

  std::vector<int64_t> expected_output_shape = {1024, 16, 256};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1024, 1024}, {16, 16}, {256, 256}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(GenADCProtoTest, GenADCInferShapeTest_2) {
  ge::op::GenADC op;
  op.UpdateInputDesc("query",
                     create_desc_shape_range({128}, ge::DT_FLOAT, ge::FORMAT_ND, {128}, ge::FORMAT_ND, {{128, 128}}));
  op.UpdateInputDesc("code_book", create_desc_shape_range({32, 256, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {32, 256, 4},
                                                          ge::FORMAT_ND, {{32, 32}, {256, 256}, {4, 4}}));
  op.UpdateInputDesc("centroids", create_desc_shape_range({1000000, 128}, ge::DT_FLOAT, ge::FORMAT_ND, {1000000, 128},
                                                          ge::FORMAT_ND, {{1000000, 1000000}, {128, 128}}));
  op.UpdateInputDesc("bucket_list",
                     create_desc_shape_range({-1}, ge::DT_INT64, ge::FORMAT_ND, {-1}, ge::FORMAT_ND, {{1, 512}}));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDescByName("adc_tables");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);

  std::vector<int64_t> expected_output_shape = {-1, 32, 256};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1, 512}, {32, 32}, {256, 256}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}
