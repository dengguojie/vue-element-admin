#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "linalg_ops.h"

class GerTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "Ger SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Ger TearDown" << std::endl;
  }
};

TEST_F(GerTest, ger_test_case_1) {
  ge::op::Ger op;
  op.UpdateInputDesc("x1", create_desc({10,}, ge::DT_FLOAT16));
  op.UpdateInputDesc("x2", create_desc({20,}, ge::DT_FLOAT16));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);

  std::vector<int64_t> expected_output_shape = {10, 20};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(GerTest, ger_test_case_2) {
  ge::op::Ger op;
  op.UpdateInputDesc("x1", create_desc({10,}, ge::DT_FLOAT));
  op.UpdateInputDesc("x2", create_desc({20,}, ge::DT_FLOAT));
	
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

  std::vector<int64_t> expected_output_shape = {10, 20};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(GerTest, ger_test_case_3) {
  ge::op::Ger op;
  op.UpdateInputDesc("x1", create_desc({1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("x2", create_desc({20,}, ge::DT_FLOAT));
	
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

  std::vector<int64_t> expected_output_shape = {1, 20};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(GerTest, ger_test_case_4) {
  ge::op::Ger op;
  op.UpdateInputDesc("x1", create_desc({-1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("x2", create_desc({20,}, ge::DT_FLOAT));
	
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  
  std::vector<int64_t> expected_output_shape = {-1, 20};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(GerTest, ger_test_case_5) {
  ge::op::Ger op;
  op.UpdateInputDesc("x1", create_desc({-1,}, ge::DT_FLOAT16));
  op.UpdateInputDesc("x2", create_desc({20,}, ge::DT_FLOAT16));
	
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  
  std::vector<int64_t> expected_output_shape = {-1, 20};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(GerTest, ger_test_case_6) {
  ge::op::Ger op;
  op.UpdateInputDesc("x1", create_desc({1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("x2", create_desc({-1,}, ge::DT_FLOAT));
	
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  
  std::vector<int64_t> expected_output_shape = {1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(GerTest, ger_test_case_7) {
  ge::op::Ger op;
  op.UpdateInputDesc("x1", create_desc({-1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("x2", create_desc({-1,}, ge::DT_FLOAT));
	
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  
  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(GerTest, ger_test_case_8) {
  ge::op::Ger op;
  op.UpdateInputDesc("x1", create_desc({10, }, ge::DT_FLOAT16));
  op.UpdateInputDesc("x2", create_desc({20, }, ge::DT_FLOAT));

  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(GerTest, ger_test_case_9) {
  ge::op::Ger op;
  op.UpdateInputDesc("x1", create_desc({10, 10}, ge::DT_FLOAT));
  op.UpdateInputDesc("x2", create_desc({20, 20}, ge::DT_FLOAT));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(GerTest, ger_test_case_10) {
  ge::op::Ger op;
  op.UpdateInputDesc("x1", create_desc({10,}, ge::DT_INT32));
  op.UpdateInputDesc("x2", create_desc({20,}, ge::DT_INT32));

  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(GerTest, ger_test_case_11) {
  ge::op::Ger op;
  op.UpdateInputDesc("x1", create_desc({10,}, ge::DT_FLOAT));
  op.UpdateInputDesc("x2", create_desc({20,}, ge::DT_FLOAT));

  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(GerTest, ger_test_case_12) {
  ge::op::Ger op;
  op.UpdateInputDesc("x1", create_desc({10,}, ge::DT_FLOAT));
  op.UpdateInputDesc("x2", create_desc({20,}, ge::DT_INT32));

  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}